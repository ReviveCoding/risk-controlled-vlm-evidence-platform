from __future__ import annotations

import csv
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from model_forge.contracts.sec_fsds import SecFsdsContract
from model_forge.data.sec_label_audit import (
    AMENDMENT_FORMS,
    BASE_FORMS,
    HORIZONS,
    _audit_rows,
    _combine_sources,
    _parse_accepted,
    _parse_filed,
    _quarter_end,
    _strict_key,
)
from model_forge.utils import (
    reject_symlink_components,
    sha256_file,
    sha256_value,
    utc_now,
    write_json,
)

DIAGNOSTIC_COLUMNS = (
    "amendment_adsh",
    "cik",
    "form",
    "period",
    "fy",
    "fp",
    "filed",
    "accepted",
    "source_quarter",
    "classification",
    "strict_prior_original_count",
    "fallback_prior_original_count",
    "selected_original_adsh",
    "selected_original_filed",
    "days_from_selected_original",
    "key_differences",
    "plausible_within_90_days",
)

SPLIT_POSITIVE_MINIMUMS = {
    "train": 500,
    "validation": 100,
    "calibration_policy": 100,
    "locked_regression": 100,
}


def _quarter_start(quarter: str) -> date:
    year = int(quarter[:4])
    month = {"1": 1, "2": 4, "3": 7, "4": 10}[quarter[5]]
    return date(year, month, 1)


def _accepted_before(rows: list[dict[str, str]], amendment: dict[str, str]) -> list[dict[str, str]]:
    accepted = _parse_accepted(amendment["accepted"])
    return [row for row in rows if _parse_accepted(row["accepted"]) < accepted]


def _closest_prior(rows: list[dict[str, str]], amendment: dict[str, str]) -> dict[str, str] | None:
    prior = _accepted_before(rows, amendment)
    return max(prior, key=lambda row: (_parse_accepted(row["accepted"]), row["adsh"])) if prior else None


def _differences(original: dict[str, str], amendment: dict[str, str]) -> str:
    return ",".join(field for field in ("period", "fy", "fp") if original[field] != amendment[field])


def _classify_amendments(
    submissions: list[dict[str, str]], source_start: date
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    originals = [row for row in submissions if row["form"] in BASE_FORMS]
    amendments = [row for row in submissions if row["form"] in AMENDMENT_FORMS]

    by_strict: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    by_period: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    by_fy_fp: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    by_cik_form: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in originals:
        by_strict[_strict_key(row, row["form"])].append(row)
        by_period[(row["cik"], row["form"], row["period"])].append(row)
        by_fy_fp[(row["cik"], row["form"], row["fy"], row["fp"])].append(row)
        by_cik_form[(row["cik"], row["form"])].append(row)

    results: list[dict[str, Any]] = []
    for amendment in sorted(amendments, key=lambda row: (row["filed"], row["accepted"], row["adsh"])):
        base_form = AMENDMENT_FORMS[amendment["form"]]
        strict_rows = by_strict.get(_strict_key(amendment, base_form), [])
        strict_prior = _accepted_before(strict_rows, amendment)
        fallback_prior: list[dict[str, str]] = []
        selected: dict[str, str] | None = None
        key_differences = ""

        if len(strict_prior) == 1:
            classification = "strict_matched_unique"
            selected = strict_prior[0]
        elif len(strict_prior) > 1:
            classification = "strict_matched_ambiguous"
            selected = _closest_prior(strict_prior, amendment)
        elif strict_rows:
            classification = "strict_original_not_prior"
            selected = min(
                strict_rows,
                key=lambda row: (_parse_accepted(row["accepted"]), row["adsh"]),
            )
        elif not all(_strict_key(amendment, base_form)):
            classification = "incomplete_amendment_key"
        else:
            period_rows = by_period.get((amendment["cik"], base_form, amendment["period"]), [])
            period_prior = _accepted_before(period_rows, amendment)
            fy_fp_rows = by_fy_fp.get((amendment["cik"], base_form, amendment["fy"], amendment["fp"]), [])
            fy_fp_prior = _accepted_before(fy_fp_rows, amendment)
            cik_form_rows = by_cik_form.get((amendment["cik"], base_form), [])
            cik_form_prior = _accepted_before(cik_form_rows, amendment)
            if period_prior:
                fallback_prior = period_prior
                classification = "fy_fp_mismatch" if len(period_prior) == 1 else "ambiguous_relaxed_period"
            elif fy_fp_prior:
                fallback_prior = fy_fp_prior
                classification = "period_mismatch" if len(fy_fp_prior) == 1 else "ambiguous_fy_fp"
            elif _parse_filed(amendment["filed"]) < source_start + timedelta(days=90):
                fallback_prior = cik_form_prior
                classification = "left_boundary_possible"
            elif cik_form_prior:
                fallback_prior = cik_form_prior
                classification = "no_linkage_key_match" if len(cik_form_prior) == 1 else "ambiguous_cik_form"
            else:
                classification = "no_prior_original_in_range"
            selected = _closest_prior(fallback_prior, amendment)
            if selected:
                key_differences = _differences(selected, amendment)

        elapsed = (
            _parse_accepted(amendment["accepted"]) - _parse_accepted(selected["accepted"])
            if selected
            else None
        )
        days = elapsed.total_seconds() / 86400 if elapsed is not None else None
        plausible_within_90_days = 0
        if (
            classification
            not in {
                "strict_matched_unique",
                "strict_matched_ambiguous",
                "strict_original_not_prior",
            }
            and elapsed is not None
        ):
            plausible_within_90_days = int(timedelta(0) <= elapsed <= timedelta(days=90))
        results.append(
            {
                "amendment_adsh": amendment["adsh"],
                "cik": amendment["cik"],
                "form": amendment["form"],
                "period": amendment["period"],
                "fy": amendment["fy"],
                "fp": amendment["fp"],
                "filed": amendment["filed"],
                "accepted": amendment["accepted"],
                "source_quarter": amendment["source_quarter"],
                "classification": classification,
                "strict_prior_original_count": len(strict_prior),
                "fallback_prior_original_count": len(fallback_prior),
                "selected_original_adsh": selected["adsh"] if selected else "",
                "selected_original_filed": selected["filed"] if selected else "",
                "days_from_selected_original": days if days is not None else "",
                "key_differences": key_differences,
                "plausible_within_90_days": plausible_within_90_days,
            }
        )

    counts = Counter(row["classification"] for row in results)
    strict_unmatched_classes = {
        "incomplete_amendment_key",
        "fy_fp_mismatch",
        "ambiguous_relaxed_period",
        "period_mismatch",
        "ambiguous_fy_fp",
        "left_boundary_possible",
        "no_linkage_key_match",
        "ambiguous_cik_form",
        "no_prior_original_in_range",
    }
    latency: Counter[str] = Counter()
    for row in results:
        if not row["classification"].startswith("strict_matched"):
            continue
        days = float(row["days_from_selected_original"])
        bucket = "0_30" if days <= 30 else "31_60" if days <= 60 else "61_90" if days <= 90 else "over_90"
        latency[bucket] += 1
    summary = {
        "amendment_filings": len(results),
        "classification_counts": dict(sorted(counts.items())),
        "strict_unmatched_amendments": sum(counts[name] for name in strict_unmatched_classes),
        "strict_unmatched_classifications": sorted(strict_unmatched_classes),
        "plausible_unmatched_within_90_days": sum(int(row["plausible_within_90_days"]) for row in results),
        "strict_match_latency_days": dict(sorted(latency.items())),
    }
    return results, summary


def _split_horizon_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    splits = ("train", "validation", "calibration_policy", "locked_regression", "prospective")
    summary: dict[str, Any] = {}
    for split in splits:
        split_rows = [row for row in rows if row["split"] == split]
        horizons: dict[str, Any] = {}
        for horizon in HORIZONS:
            eligible = [row for row in split_rows if row[f"amended_{horizon}"] != ""]
            positives = sum(int(row[f"amended_{horizon}"]) for row in eligible)
            horizons[str(horizon)] = {
                "eligible_originals": len(eligible),
                "positive_amendments": positives,
                "negative_originals": len(eligible) - positives,
                "prevalence": positives / len(eligible) if eligible else None,
                "right_censored_originals": sum(int(row[f"right_censored_{horizon}"]) for row in split_rows),
            }
        summary[split] = horizons
    return summary


def _readiness_gate(split_summary: dict[str, Any], linkage_summary: dict[str, Any]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for split, minimum in SPLIT_POSITIVE_MINIMUMS.items():
        observed = split_summary[split]["90"]["positive_amendments"]
        checks.append(
            {
                "check": f"minimum_90_day_positives:{split}",
                "status": "PASS" if observed >= minimum else "FAIL",
                "observed": observed,
                "minimum": minimum,
            }
        )
    plausible = linkage_summary["plausible_unmatched_within_90_days"]
    checks.append(
        {
            "check": "plausible_unmatched_within_90_days_reviewed",
            "status": "PASS" if plausible == 0 else "REVIEW_REQUIRED",
            "observed": plausible,
            "required": 0,
        }
    )
    status = (
        "FAIL"
        if any(check["status"] == "FAIL" for check in checks)
        else "REVIEW_REQUIRED"
        if any(check["status"] == "REVIEW_REQUIRED" for check in checks)
        else "PASS"
    )
    return {
        "status": status,
        "target_horizon_days": 90,
        "positive_minimums": SPLIT_POSITIVE_MINIMUMS,
        "checks": checks,
        "interpretation": (
            "No feature engineering or model training is authorized unless this gate is PASS."
        ),
    }


def publish_sec_linkage_diagnostic(
    historical_input: Path,
    recent_input: Path,
    output: Path,
    historical_contract: SecFsdsContract,
    recent_contract: SecFsdsContract,
) -> dict[str, Any]:
    """Publish unmatched-linkage causes and split-level label readiness."""

    output = output.resolve(strict=False)
    reject_symlink_components(output, "SEC linkage-diagnostic output")
    if output.exists():
        raise FileExistsError(f"SEC linkage-diagnostic output already exists: {output}")
    resolved_inputs = {historical_input.resolve(), recent_input.resolve()}
    if len(resolved_inputs) != 2 or any(
        output == root or output.is_relative_to(root) or root.is_relative_to(output)
        for root in resolved_inputs
    ):
        raise ValueError("SEC linkage-diagnostic inputs and output must be separate trees")

    submissions, archives, quarters = _combine_sources(
        ((historical_input, historical_contract), (recent_input, recent_contract))
    )
    observation_end = _quarter_end(max(quarters))
    label_rows, feasibility = _audit_rows(submissions, observation_end)
    diagnostic_rows, linkage = _classify_amendments(submissions, _quarter_start(min(quarters)))
    if linkage["strict_unmatched_amendments"] != feasibility["strict_unmatched_amendments"]:
        raise ValueError("SEC linkage diagnostic disagrees with feasibility audit")
    if (
        linkage["classification_counts"].get("strict_original_not_prior", 0)
        != feasibility["amendments_not_after_strict_original"]
    ):
        raise ValueError("SEC amendment ordering diagnostic disagrees with feasibility audit")
    split_summary = _split_horizon_summary(label_rows)
    gate = _readiness_gate(split_summary, linkage)

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=str(output.parent)))
    try:
        csv_path = stage / "sec_amendment_linkage_diagnostic.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=DIAGNOSTIC_COLUMNS)
            writer.writeheader()
            writer.writerows(diagnostic_rows)
            stream.flush()
            os.fsync(stream.fileno())
        manifest = {
            "schema_version": "1.0",
            "status": "PASS",
            "created_at": utc_now(),
            "workload": "sec_future_amendment_linkage_diagnostic_only",
            "claim_boundary": "diagnostic_only_no_features_no_model_no_release_claim",
            "observation_end": observation_end.isoformat(),
            "source_quarters": list(quarters),
            "source_contract_fingerprints": [
                historical_contract.fingerprint,
                recent_contract.fingerprint,
            ],
            "source_archives": archives,
            "diagnostic_csv": csv_path.name,
            "diagnostic_csv_sha256": sha256_file(csv_path),
            "diagnostic_schema_fingerprint": sha256_value(DIAGNOSTIC_COLUMNS),
            "feasibility_cross_check": feasibility,
            "linkage_diagnostic": linkage,
            "split_horizon_distribution": split_summary,
            "stage_gate": gate,
        }
        write_json(stage / "sec_linkage_diagnostic.json", manifest)
        os.replace(stage, output)
        return {**manifest, "output": str(output)}
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
