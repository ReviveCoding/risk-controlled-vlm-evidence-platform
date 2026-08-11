from __future__ import annotations

import csv
import os
import shutil
import tempfile
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from model_forge.contracts.sec_fsds import SecFsdsContract
from model_forge.data.sec_fsds import _read_header, _safe_archive_entries
from model_forge.utils import (
    reject_symlink_components,
    sha256_file,
    sha256_value,
    utc_now,
    write_json,
)

BASE_FORMS = ("10-K", "10-Q")
AMENDMENT_FORMS = {"10-K/A": "10-K", "10-Q/A": "10-Q"}
HORIZONS = (30, 60, 90)
AUDIT_COLUMNS = (
    "original_adsh",
    "cik",
    "form",
    "period",
    "fy",
    "fp",
    "filed",
    "accepted",
    "source_quarter",
    "split",
    "strict_key_complete",
    "strict_key_unique",
    "followup_days_observed",
    "right_censored_30",
    "right_censored_60",
    "right_censored_90",
    "amended_30",
    "amended_60",
    "amended_90",
    "first_amendment_adsh",
    "first_amendment_filed",
    "days_to_first_amendment",
    "prevrpt_audit_only",
    "prevrpt_agrees_with_observed_amendment",
)


def _quarter_end(quarter: str) -> date:
    year = int(quarter[:4])
    month_day = {"1": (3, 31), "2": (6, 30), "3": (9, 30), "4": (12, 31)}[quarter[5]]
    return date(year, *month_day)


def _parse_filed(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y%m%d").date()
    except ValueError as error:
        raise ValueError(f"invalid SEC filed date in label audit: {value!r}") from error


def _parse_accepted(value: str) -> datetime:
    normalized = value.removesuffix(".0")
    try:
        return datetime.strptime(normalized, "%Y-%m-%d %H:%M:%S")
    except ValueError as error:
        raise ValueError(f"invalid SEC accepted timestamp in label audit: {value!r}") from error


def _split_for_filed(value: date) -> str:
    if value <= date(2023, 12, 31):
        return "train"
    if value <= date(2024, 6, 30):
        return "validation"
    if value <= date(2024, 12, 31):
        return "calibration_policy"
    if value <= date(2025, 12, 31):
        return "locked_regression"
    return "prospective"


def _strict_key(row: dict[str, str], base_form: str) -> tuple[str, ...]:
    return (row["cik"], base_form, row["period"], row["fy"], row["fp"])


def _relaxed_key(row: dict[str, str], base_form: str) -> tuple[str, ...]:
    return (row["cik"], base_form, row["period"])


def _read_submissions(
    input_root: Path, contract: SecFsdsContract
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    root = input_root.resolve()
    if not root.is_dir() or root.is_symlink():
        raise ValueError(f"SEC audit input must be a regular directory: {root}")
    expected = {f"{quarter}.zip" for quarter in contract.required_quarters}
    actual = {item.name for item in root.glob("*.zip") if item.is_file()}
    if actual != expected:
        raise ValueError(
            "SEC audit ZIP inventory mismatch; "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )

    submissions: list[dict[str, str]] = []
    archives: list[dict[str, Any]] = []
    expected_header = contract.columns["submissions"]
    member = contract.members["submissions"]
    for quarter in contract.required_quarters:
        path = root / f"{quarter}.zip"
        source_hash = sha256_file(path)
        _safe_archive_entries(path, contract)
        with zipfile.ZipFile(path) as archive:
            if _read_header(archive, member) != expected_header:
                raise ValueError(f"SEC audit SUB header mismatch: {quarter}")
            with archive.open(member) as stream:
                frame = pd.read_csv(
                    stream,
                    sep="\t",
                    dtype=str,
                    keep_default_na=False,
                    na_filter=False,
                    encoding="utf-8-sig",
                )
            if tuple(frame.columns) != expected_header:
                raise ValueError(f"SEC audit SUB columns changed while parsing: {quarter}")
            for record in frame.to_dict(orient="records"):
                row = {str(key): str(value) for key, value in record.items()}
                row["source_quarter"] = quarter
                submissions.append(row)
            # Reading sub.txt to EOF validates the member CRC. Other members are
            # outside this SUB-only feasibility claim and are not decompressed.
        if sha256_file(path) != source_hash:
            raise ValueError(f"SEC audit archive changed while reading: {path.name}")
        archives.append(
            {
                "quarter": quarter,
                "file_name": path.name,
                "bytes": path.stat().st_size,
                "sha256": source_hash,
                "submission_rows": len(frame),
            }
        )
    return submissions, archives


def _combine_sources(
    sources: Iterable[tuple[Path, SecFsdsContract]],
) -> tuple[list[dict[str, str]], list[dict[str, Any]], tuple[str, ...]]:
    submissions: list[dict[str, str]] = []
    archives: list[dict[str, Any]] = []
    quarters: list[str] = []
    fingerprints: list[str] = []
    for root, contract in sources:
        rows, source_archives = _read_submissions(root, contract)
        submissions.extend(rows)
        archives.extend(source_archives)
        quarters.extend(contract.required_quarters)
        fingerprints.append(contract.fingerprint)
    if not quarters or len(set(quarters)) != len(quarters):
        raise ValueError("SEC audit source quarters must be nonempty and non-overlapping")
    expected = []
    start_year, start_quarter = int(min(quarters)[:4]), int(min(quarters)[5])
    end_year, end_quarter = int(max(quarters)[:4]), int(max(quarters)[5])
    year, quarter = start_year, start_quarter
    while (year, quarter) <= (end_year, end_quarter):
        expected.append(f"{year}q{quarter}")
        quarter += 1
        if quarter == 5:
            year += 1
            quarter = 1
    if sorted(quarters) != expected:
        raise ValueError("SEC audit source quarters must form one continuous interval")
    if len({row["adsh"] for row in submissions}) != len(submissions):
        raise ValueError("SEC audit found duplicate accession numbers across source quarters")
    return submissions, archives, tuple(sorted(quarters))


def _audit_rows(
    submissions: list[dict[str, str]], observation_end: date
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    relevant = [row for row in submissions if row["form"] in set(BASE_FORMS) | set(AMENDMENT_FORMS)]
    originals = [row for row in relevant if row["form"] in BASE_FORMS]
    amendments = [row for row in relevant if row["form"] in AMENDMENT_FORMS]

    original_by_strict: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    original_by_relaxed: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    amendment_by_strict: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in originals:
        original_by_strict[_strict_key(row, row["form"])].append(row)
        original_by_relaxed[_relaxed_key(row, row["form"])].append(row)
    for row in amendments:
        base_form = AMENDMENT_FORMS[row["form"]]
        amendment_by_strict[_strict_key(row, base_form)].append(row)

    for rows in amendment_by_strict.values():
        rows.sort(key=lambda row: (_parse_accepted(row["accepted"]), row["adsh"]))

    strict_unmatched = 0
    relaxed_only = 0
    amendment_before_original = 0
    for amendment in amendments:
        base_form = AMENDMENT_FORMS[amendment["form"]]
        strict = original_by_strict.get(_strict_key(amendment, base_form), [])
        prior = [
            row for row in strict if _parse_accepted(row["accepted"]) < _parse_accepted(amendment["accepted"])
        ]
        if prior:
            continue
        if strict:
            amendment_before_original += 1
        else:
            strict_unmatched += 1
            relaxed = original_by_relaxed.get(_relaxed_key(amendment, base_form), [])
            if any(
                _parse_accepted(row["accepted"]) < _parse_accepted(amendment["accepted"]) for row in relaxed
            ):
                relaxed_only += 1

    output_rows: list[dict[str, Any]] = []
    observation_cutoff = datetime.combine(observation_end + timedelta(days=1), time.min)
    for original in sorted(originals, key=lambda row: (row["filed"], row["accepted"], row["adsh"])):
        filed = _parse_filed(original["filed"])
        strict_key = _strict_key(original, original["form"])
        complete = all(strict_key)
        unique = len(original_by_strict[strict_key]) == 1
        future_amendments = [
            row
            for row in amendment_by_strict.get(strict_key, [])
            if _parse_accepted(row["accepted"]) > _parse_accepted(original["accepted"])
        ]
        first = future_amendments[0] if future_amendments else None
        accepted = _parse_accepted(original["accepted"])
        elapsed = _parse_accepted(first["accepted"]) - accepted if first else None
        days_to = elapsed.total_seconds() / 86400 if elapsed is not None else None
        followup = max((observation_cutoff - accepted).total_seconds() / 86400, 0)
        observed_amendment = first is not None and _parse_accepted(first["accepted"]) < observation_cutoff
        prevrpt = original["prevrpt"]
        if prevrpt not in {"0", "1"}:
            raise ValueError(f"invalid SEC prevrpt value: {prevrpt!r}")
        result: dict[str, Any] = {
            "original_adsh": original["adsh"],
            "cik": original["cik"],
            "form": original["form"],
            "period": original["period"],
            "fy": original["fy"],
            "fp": original["fp"],
            "filed": original["filed"],
            "accepted": original["accepted"],
            "source_quarter": original["source_quarter"],
            "split": _split_for_filed(filed),
            "strict_key_complete": int(complete),
            "strict_key_unique": int(unique),
            "followup_days_observed": followup,
            "first_amendment_adsh": first["adsh"] if first else "",
            "first_amendment_filed": first["filed"] if first else "",
            "days_to_first_amendment": days_to if days_to is not None else "",
            "prevrpt_audit_only": prevrpt,
            "prevrpt_agrees_with_observed_amendment": int((prevrpt == "1") == observed_amendment),
        }
        for horizon in HORIZONS:
            censored = accepted + timedelta(days=horizon) >= observation_cutoff
            result[f"right_censored_{horizon}"] = int(censored)
            result[f"amended_{horizon}"] = (
                ""
                if censored or not unique or not complete
                else int(elapsed is not None and elapsed <= timedelta(days=horizon))
            )
        output_rows.append(result)

    split_counts = Counter(row["split"] for row in output_rows)
    horizon_summary: dict[str, Any] = {}
    for horizon in HORIZONS:
        eligible = [
            row
            for row in output_rows
            if row["strict_key_complete"] == 1
            and row["strict_key_unique"] == 1
            and row[f"right_censored_{horizon}"] == 0
        ]
        positives = sum(int(row[f"amended_{horizon}"]) for row in eligible)
        horizon_summary[str(horizon)] = {
            "eligible_originals": len(eligible),
            "positive_amendments": positives,
            "prevalence": positives / len(eligible) if eligible else None,
            "right_censored_originals": sum(int(row[f"right_censored_{horizon}"]) for row in output_rows),
        }
    diagnostics = {
        "all_submission_rows": len(submissions),
        "relevant_submission_rows": len(relevant),
        "original_filings": len(originals),
        "amendment_filings": len(amendments),
        "unique_strict_original_keys": sum(len(rows) == 1 for rows in original_by_strict.values()),
        "ambiguous_strict_original_keys": sum(len(rows) > 1 for rows in original_by_strict.values()),
        "originals_excluded_for_ambiguous_key": sum(
            len(rows) for rows in original_by_strict.values() if len(rows) > 1
        ),
        "originals_excluded_for_incomplete_key": sum(
            not all(_strict_key(row, row["form"])) for row in originals
        ),
        "strict_unmatched_amendments": strict_unmatched,
        "relaxed_key_only_matches": relaxed_only,
        "amendments_not_after_strict_original": amendment_before_original,
        "split_counts": dict(sorted(split_counts.items())),
        "horizons": horizon_summary,
        "prevrpt_agreement_count": sum(row["prevrpt_agrees_with_observed_amendment"] for row in output_rows),
    }
    return output_rows, diagnostics


def publish_sec_label_feasibility_audit(
    historical_input: Path,
    recent_input: Path,
    output: Path,
    historical_contract: SecFsdsContract,
    recent_contract: SecFsdsContract,
) -> dict[str, Any]:
    """Publish a point-in-time amendment-label feasibility audit without training."""

    output = output.resolve(strict=False)
    reject_symlink_components(output, "SEC label-audit output")
    if output.exists():
        raise FileExistsError(f"SEC label-audit output already exists: {output}")
    resolved_inputs = {historical_input.resolve(), recent_input.resolve()}
    if len(resolved_inputs) != 2 or any(
        output == root or output.is_relative_to(root) or root.is_relative_to(output)
        for root in resolved_inputs
    ):
        raise ValueError("SEC label-audit inputs and output must be separate trees")

    submissions, archives, quarters = _combine_sources(
        (
            (historical_input, historical_contract),
            (recent_input, recent_contract),
        )
    )
    observation_end = _quarter_end(max(quarters))
    rows, diagnostics = _audit_rows(submissions, observation_end)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=str(output.parent)))
    try:
        csv_path = stage / "sec_amendment_label_candidates.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=AUDIT_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        manifest = {
            "schema_version": "1.0",
            "status": "PASS",
            "created_at": utc_now(),
            "workload": "sec_future_amendment_risk_feasibility_only",
            "claim_boundary": "label_audit_only_no_features_no_model_no_release_claim",
            "target": "future_10-K/A_or_10-Q/A_after_original_filing",
            "strict_linkage_key": ["cik", "base_form", "period", "fy", "fp"],
            "horizons_days": list(HORIZONS),
            "observation_end": observation_end.isoformat(),
            "source_quarters": list(quarters),
            "source_contract_fingerprints": [
                historical_contract.fingerprint,
                recent_contract.fingerprint,
            ],
            "source_archives": archives,
            "candidate_csv": csv_path.name,
            "candidate_csv_sha256": sha256_file(csv_path),
            "candidate_schema_fingerprint": sha256_value(AUDIT_COLUMNS),
            "feature_exclusions": [
                "prevrpt",
                "amendment_form",
                "amendment_adsh",
                "amendment_filed",
                "all_information_accepted_after_original_filing",
            ],
            "diagnostics": diagnostics,
        }
        write_json(stage / "sec_label_feasibility.json", manifest)
        os.replace(stage, output)
        return {**manifest, "output": str(output)}
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
