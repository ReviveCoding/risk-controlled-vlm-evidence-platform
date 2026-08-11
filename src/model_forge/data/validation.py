from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from model_forge.contracts.models import DataContract


@dataclass
class ValidationResult:
    valid: bool
    errors: list[str]
    warnings: list[str]
    summary: dict[str, Any]


def validate_dataframe(frame: pd.DataFrame, contract: DataContract) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    missing = sorted(set(contract.required_columns) - set(frame.columns))
    if missing:
        errors.append(f"missing_required_columns:{','.join(missing)}")
        return ValidationResult(False, errors, warnings, {"row_count": len(frame)})
    if frame.empty:
        return ValidationResult(
            False,
            ["empty_dataset"],
            warnings,
            {"row_count": 0, "invalid_schema_count": 1},
        )

    duplicate_count = int(frame[contract.entity_key].duplicated().sum())
    if duplicate_count:
        errors.append(f"duplicate_entity:{duplicate_count}")
    entity = frame[contract.entity_key]
    null_entity = int((entity.isna() | entity.astype(str).str.strip().eq("")).sum())
    if null_entity:
        errors.append(f"null_entity:{null_entity}")
    sent = pd.to_datetime(frame[contract.event_time], errors="coerce", utc=True)
    received = pd.to_datetime(frame[contract.received_time], errors="coerce", utc=True)
    invalid_dates = int((sent.isna() | received.isna()).sum())
    if invalid_dates:
        errors.append(f"invalid_dates:{invalid_dates}")
    invalid_order = int((received > sent).sum())
    if invalid_order:
        errors.append(f"received_after_sent:{invalid_order}")
    numeric_target = pd.to_numeric(frame[contract.target], errors="coerce")
    invalid_target = int((~numeric_target.isin(contract.allowed_target_values)).sum())
    if invalid_target:
        errors.append(f"invalid_target:{invalid_target}")
    missingness = frame[list(contract.required_columns)].isna().mean().sort_values(ascending=False)
    if float(missingness.max()) > 0.40:
        warnings.append("required_column_missingness_above_40pct")
    summary = {
        "row_count": int(len(frame)),
        "feature_source_column_count": int(len(frame.columns) - 1),
        "target_positive_count": int(numeric_target.eq(1).sum()),
        "target_prevalence": (float(numeric_target.mean()) if numeric_target.notna().any() else None),
        "duplicate_count": duplicate_count,
        "invalid_schema_count": len(errors),
        "invalid_date_count": invalid_dates,
        "outlier_extreme_count": int((sent < pd.Timestamp("2010-01-01", tz="UTC")).sum()),
        "missingness": {str(k): float(v) for k, v in missingness.items()},
        "leakage_check_result": "PASS" if not errors else "FAIL",
    }
    return ValidationResult(not errors, errors, warnings, summary)


def assign_temporal_splits(frame: pd.DataFrame, contract: DataContract) -> pd.Series:
    event_time = pd.to_datetime(frame[contract.event_time], errors="raise", utc=True)
    split = pd.Series("out_of_contract", index=frame.index, dtype="object")
    for name, (start, end) in contract.split.items():
        mask = event_time.between(
            pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"), inclusive="both"
        )
        if (split[mask] != "out_of_contract").any():
            raise ValueError(f"overlapping split assignment for {name}")
        split.loc[mask] = name
    return split


def validate_split_readiness(
    frame: pd.DataFrame,
    contract: DataContract,
    partitions: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Fail before training when a locked partition cannot support binary evaluation."""

    assigned = assign_temporal_splits(frame, contract)
    selected_partitions = tuple(contract.split) if partitions is None else partitions
    if (
        not selected_partitions
        or len(set(selected_partitions)) != len(selected_partitions)
        or any(name not in contract.split for name in selected_partitions)
    ):
        raise ValueError("split readiness partitions must be unique contract split names")
    errors: list[str] = []
    distribution: list[dict[str, Any]] = []
    target = pd.to_numeric(frame[contract.target], errors="coerce")
    for name in selected_partitions:
        mask = assigned.eq(name)
        values = target[mask]
        row = {
            "split": name,
            "rows": int(mask.sum()),
            "positives": int(values.eq(1).sum()),
            "negatives": int(values.eq(0).sum()),
        }
        distribution.append(row)
        if row["rows"] == 0:
            errors.append(f"empty_split:{name}")
        elif row["positives"] == 0 or row["negatives"] == 0:
            errors.append(f"single_class_split:{name}")
    return {"status": "PASS" if not errors else "FAIL", "errors": errors, "splits": distribution}


def reject_forbidden_model_features(columns: list[str], contract: DataContract) -> None:
    forbidden = sorted(set(columns) & set(contract.forbidden_model_features))
    if forbidden:
        raise ValueError(f"forbidden model features: {','.join(forbidden)}")
