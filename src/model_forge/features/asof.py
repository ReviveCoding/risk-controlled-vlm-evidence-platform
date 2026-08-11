from __future__ import annotations

import numpy as np
import pandas as pd

from model_forge.contracts.models import DataContract
from model_forge.data.validation import reject_forbidden_model_features

CATEGORICAL_FEATURES = [
    "product",
    "sub_product",
    "issue",
    "sub_issue",
    "company",
    "submitted_via",
]
NUMERIC_FEATURES = [
    "days_to_send",
    "sent_month",
    "sent_dow",
    "narrative_present",
    "narrative_length",
    "company_prior_count",
    "company_prior_rate",
    "product_prior_count",
    "product_prior_rate",
]
TEXT_FEATURE = "complaint_narrative"
MODEL_FEATURES = [TEXT_FEATURE, *CATEGORICAL_FEATURES, *NUMERIC_FEATURES]


def _past_only_group_stats(
    frame: pd.DataFrame,
    group: str,
    target: str,
    event_time: str,
    label_maturity_days: int,
) -> tuple[pd.Series, pd.Series]:
    """Return stats using only labels whose conservative maturity time has passed."""
    work = pd.DataFrame(
        {
            "group_key": frame[group].fillna("__MISSING__").astype(str),
            "event_time": pd.to_datetime(frame[event_time], errors="raise", utc=True),
            "target": frame[target].astype(float),
        },
        index=frame.index,
    )
    work["label_available_time"] = work["event_time"] + pd.to_timedelta(label_maturity_days, unit="D")

    def matured_totals(grouped_work: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        counts = pd.Series(0.0, index=grouped_work.index)
        sums = pd.Series(0.0, index=grouped_work.index)
        for _, query_rows in grouped_work.groupby("group_key", sort=False, observed=True):
            history = (
                query_rows.groupby("label_available_time", sort=True, observed=True)["target"]
                .agg(row_count="size", target_sum="sum")
                .cumsum()
            )
            available = history.index.to_numpy(dtype="datetime64[ns]")
            queries = query_rows["event_time"].to_numpy(dtype="datetime64[ns]")
            positions = np.searchsorted(available, queries, side="right") - 1
            valid = positions >= 0
            if valid.any():
                indices = query_rows.index.to_numpy()[valid]
                counts.loc[indices] = history["row_count"].to_numpy(dtype=float)[positions[valid]]
                sums.loc[indices] = history["target_sum"].to_numpy(dtype=float)[positions[valid]]
        return counts, sums

    prior_count, prior_sum = matured_totals(work)
    global_grouped = work.groupby("label_available_time", sort=True, observed=True)["target"].agg(
        row_count="size", target_sum="sum"
    )
    global_grouped = global_grouped.cumsum()
    available = global_grouped.index.to_numpy(dtype="datetime64[ns]")
    queries = work["event_time"].to_numpy(dtype="datetime64[ns]")
    positions = np.searchsorted(available, queries, side="right") - 1
    global_prior_count = pd.Series(0.0, index=frame.index)
    global_prior_sum = pd.Series(0.0, index=frame.index)
    valid = positions >= 0
    if valid.any():
        indices = work.index.to_numpy()[valid]
        global_prior_count.loc[indices] = global_grouped["row_count"].to_numpy(dtype=float)[positions[valid]]
        global_prior_sum.loc[indices] = global_grouped["target_sum"].to_numpy(dtype=float)[positions[valid]]
    global_rate = (global_prior_sum + 1.0) / (global_prior_count + 20.0)
    smoothed_rate = (prior_sum + 20.0 * global_rate) / (prior_count + 20.0)
    return prior_count, smoothed_rate.fillna(global_rate).fillna(0.05)


def build_asof_features(frame: pd.DataFrame, contract: DataContract) -> pd.DataFrame:
    ordered = frame.sort_values([contract.event_time, contract.entity_key]).copy()
    sent = pd.to_datetime(ordered[contract.event_time])
    received = pd.to_datetime(ordered[contract.received_time])
    narrative = ordered[TEXT_FEATURE].fillna("").astype(str)
    ordered["days_to_send"] = (sent - received).dt.days.clip(lower=0).astype(float)
    ordered["sent_month"] = sent.dt.month.astype(float)
    ordered["sent_dow"] = sent.dt.dayofweek.astype(float)
    ordered["narrative_present"] = narrative.str.len().gt(0).astype(float)
    ordered["narrative_length"] = narrative.str.len().clip(upper=2000).astype(float)
    company_count, company_rate = _past_only_group_stats(
        ordered,
        "company",
        contract.target,
        contract.event_time,
        contract.label_maturity_days,
    )
    product_count, product_rate = _past_only_group_stats(
        ordered,
        "product",
        contract.target,
        contract.event_time,
        contract.label_maturity_days,
    )
    ordered["company_prior_count"] = np.log1p(company_count)
    ordered["company_prior_rate"] = company_rate
    ordered["product_prior_count"] = np.log1p(product_count)
    ordered["product_prior_rate"] = product_rate
    for column in CATEGORICAL_FEATURES:
        ordered[column] = ordered[column].fillna("__MISSING__").astype(str)
    ordered[TEXT_FEATURE] = narrative
    reject_forbidden_model_features(MODEL_FEATURES, contract)
    result = ordered[[contract.entity_key, contract.event_time, contract.target, *MODEL_FEATURES]].copy()
    return result.sort_index()
