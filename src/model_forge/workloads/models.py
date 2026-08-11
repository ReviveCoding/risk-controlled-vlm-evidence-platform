from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from model_forge.features.asof import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TEXT_FEATURE


class PrevalenceClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X: Any, y: Any) -> PrevalenceClassifier:
        raw_y = np.asarray(y)
        if raw_y.ndim != 1 or len(raw_y) == 0:
            raise ValueError("prevalence classifier requires a nonempty one-dimensional target")
        try:
            numeric_y = raw_y.astype(float)
        except (TypeError, ValueError) as error:
            raise ValueError("prevalence classifier requires a binary target") from error
        if not np.all(np.isfinite(numeric_y)) or not np.all(np.isin(numeric_y, [0.0, 1.0])):
            raise ValueError("prevalence classifier requires a binary target")
        self.prevalence_ = float(np.clip(np.mean(numeric_y), 1e-6, 1 - 1e-6))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        if not hasattr(self, "prevalence_"):
            raise ValueError("prevalence classifier is not fitted")
        positive = np.full(len(X), self.prevalence_)
        return np.column_stack([1 - positive, positive])


class TextColumnSelector(BaseEstimator):
    def fit(self, X: Any, y: Any = None) -> TextColumnSelector:
        return self

    def transform(self, X: Any) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, 0].fillna("").astype(str).to_numpy()
        array = np.asarray(X)
        return array[:, 0].astype(str) if array.ndim == 2 else array.astype(str)


class DeterministicTfidfVectorizer(TfidfVectorizer):
    """Exclude scikit-learn's process-local stop-word cache from artifacts."""

    def __getstate__(self) -> dict[str, Any]:
        state = super().__getstate__()
        state.pop("_stop_words_id", None)
        return state


@dataclass
class PlattCalibrator:
    coefficient: float = 1.0
    intercept: float = 0.0
    fitted_on_partition: str = "unfitted"

    def fit(self, probabilities: np.ndarray, y: np.ndarray, partition: str) -> PlattCalibrator:
        raw_probabilities = np.asarray(probabilities)
        raw_y = np.asarray(y)
        if (
            raw_probabilities.ndim != 1
            or raw_y.ndim != 1
            or len(raw_probabilities) != len(raw_y)
            or len(raw_y) == 0
        ):
            raise ValueError("calibration arrays must be nonempty, aligned, and one-dimensional")
        try:
            probabilities = raw_probabilities.astype(float)
            numeric_y = raw_y.astype(float)
        except (TypeError, ValueError) as error:
            raise ValueError("calibration requires finite probabilities and binary targets") from error
        if (
            not np.all(np.isfinite(probabilities))
            or np.any((probabilities < 0) | (probabilities > 1))
            or not np.all(np.isfinite(numeric_y))
            or not np.all(np.isin(numeric_y, [0.0, 1.0]))
        ):
            raise ValueError("calibration requires [0, 1] probabilities and binary targets")
        if partition != "calibration_policy":
            raise ValueError("calibrator must be fitted only on calibration_policy")
        y = numeric_y.astype(int)
        clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
        logits = np.log(clipped / (1 - clipped)).reshape(-1, 1)
        if len(np.unique(y)) < 2:
            self.coefficient, self.intercept = 1.0, 0.0
        else:
            model = LogisticRegression(C=1e3, solver="lbfgs", random_state=0)
            model.fit(logits, y)
            self.coefficient = float(model.coef_[0, 0])
            self.intercept = float(model.intercept_[0])
        self.fitted_on_partition = partition
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        if self.fitted_on_partition == "unfitted":
            raise ValueError("calibrator is not fitted")
        probabilities = np.asarray(probabilities, dtype=float)
        if (
            probabilities.ndim != 1
            or not np.all(np.isfinite(probabilities))
            or np.any((probabilities < 0) | (probabilities > 1))
        ):
            raise ValueError("calibration transform requires one-dimensional probabilities in [0, 1]")
        clipped = np.clip(probabilities, 1e-6, 1 - 1e-6)
        logits = np.log(clipped / (1 - clipped))
        return expit(self.coefficient * logits + self.intercept)


def _build_sparse_text_candidate(
    config: dict[str, Any],
    seed: int,
    *,
    default_word_features: int,
    default_char_features: int,
    default_c: float,
) -> Pipeline:
    word_features = int(config.get("word_features", default_word_features))
    char_features = int(config.get("char_features", default_char_features))
    features = ColumnTransformer(
        [
            (
                "word",
                Pipeline(
                    [
                        ("select", TextColumnSelector()),
                        (
                            "tfidf",
                            DeterministicTfidfVectorizer(
                                ngram_range=(1, 2),
                                min_df=2,
                                max_features=word_features,
                                sublinear_tf=True,
                            ),
                        ),
                    ]
                ),
                [TEXT_FEATURE],
            ),
            (
                "char",
                Pipeline(
                    [
                        ("select", TextColumnSelector()),
                        (
                            "tfidf",
                            DeterministicTfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(3, 5),
                                min_df=2,
                                max_features=char_features,
                                sublinear_tf=True,
                            ),
                        ),
                    ]
                ),
                [TEXT_FEATURE],
            ),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", min_frequency=2),
                CATEGORICAL_FEATURES,
            ),
            (
                "num",
                Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                NUMERIC_FEATURES,
            ),
        ]
    )
    return Pipeline(
        [
            ("features", features),
            (
                "classifier",
                LogisticRegression(
                    C=float(config.get("C", default_c)),
                    class_weight="balanced",
                    max_iter=int(config.get("max_iter", 500)),
                    solver="liblinear",
                    random_state=seed,
                ),
            ),
        ]
    )


def build_candidate(candidate_id: str, config: dict[str, Any], seed: int) -> BaseEstimator:
    if candidate_id == "W0":
        return PrevalenceClassifier()
    if candidate_id == "W1":
        return _build_sparse_text_candidate(
            config,
            seed,
            default_word_features=3000,
            default_char_features=2000,
            default_c=1.0,
        )
    if candidate_id == "W2":
        return _build_sparse_text_candidate(
            config,
            seed,
            default_word_features=20_000,
            default_char_features=10_000,
            default_c=0.5,
        )
    raise ValueError(f"unknown workload candidate: {candidate_id}")
