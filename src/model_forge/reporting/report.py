from __future__ import annotations

import atexit
import html
import os
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve

from model_forge.utils import atomic_write_text, load_bounded_json, write_json

STYLE = {
    "W0": "#8c8c8c",
    "W1": "#2f6b9a",
    "W2": "#d55e00",
    "C0": "#999999",
    "C1": "#2f6b9a",
    "C2": "#009e73",
    "C3": "#cc79a7",
    "C4": "#d55e00",
}


@lru_cache(maxsize=1)
def _pyplot() -> Any:
    """Import pyplot only when a report is requested.

    CLI discovery, ``--help``, and non-report commands must remain read-only and
    fast. A process-local cache directory is created only for the rendering path.
    """

    if "MPLCONFIGDIR" not in os.environ:
        config_dir = tempfile.mkdtemp(prefix="model-forge-r-matplotlib-")
        os.environ["MPLCONFIGDIR"] = config_dir
        atexit.register(shutil.rmtree, config_dir, ignore_errors=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    return pyplot


def _save(fig: Any, path: Path) -> None:
    plt = _pyplot()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.tight_layout()
        # The complete run is already built in a disposable staging directory. Writing
        # the figure directly avoids long-lived temp files racing workspace sync tools.
        fig.savefig(path, dpi=145, bbox_inches="tight")
        # Windows FlushFileBuffers requires a handle opened with write access.
        # ``r+b`` preserves the completed image while giving os.fsync a valid
        # cross-platform handle.
        with path.open("r+b") as stream:
            os.fsync(stream.fileno())
    finally:
        plt.close(fig)


def _figure_manifest_entry(
    figure_id: str, filename: str, question: str, source: str, interpretation: str, limitation: str
) -> dict[str, str]:
    return {
        "figure_id": figure_id,
        "path": f"figures/{filename}",
        "question": question,
        "data_source": source,
        "generation_command": "model-forge report --output <run-dir>",
        "interpretation": interpretation,
        "limitation": limitation,
    }


def generate_report(output: Path) -> dict[str, Any]:
    plt = _pyplot()
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    comparison = pd.read_csv(output / "candidate_comparison.csv")
    slices = pd.read_csv(output / "slice_metrics.csv")
    scenarios = pd.read_csv(output / "decision_scenarios.csv")
    predictions = pd.read_csv(output / "predictions.csv")
    fault = pd.read_csv(output / "fault_benchmark_results.csv")
    data_quality = load_bounded_json(output / "data_quality.json", label="data-quality evidence")
    fault_summary = load_bounded_json(
        output / "fault_benchmark_summary.json", label="fault-benchmark evidence"
    )
    decision = load_bounded_json(output / "promotion_decision.json", label="promotion-decision evidence")
    metrics_document = load_bounded_json(output / "metrics.json", label="metric evidence")
    release_evidence = load_bounded_json(output / "release_evidence.json", label="release evidence")
    monitoring = load_bounded_json(output / "monitoring_snapshot.json", label="monitoring evidence")
    dataset_identity = load_bounded_json(output / "dataset_identity.json", label="dataset identity")
    run_manifest_path = output / "run_manifest.json"
    run_manifest = (
        load_bounded_json(run_manifest_path, label="run manifest")
        if run_manifest_path.is_file()
        else {
            "run_id": metrics_document.get("run_id"),
            "config_hash": metrics_document.get("config_hash"),
            "source_fingerprint": metrics_document.get("source_fingerprint"),
            "split": data_quality.get("split_boundaries", {}),
        }
    )
    qualification = load_bounded_json(output / "qualification_ablation.json", label="qualification evidence")
    if not isinstance(qualification, list):
        raise ValueError("qualification evidence must be a list")
    evidence_class = str(data_quality.get("evidence_class", "unknown"))
    is_fixture = evidence_class == "measured_synthetic_fixture"
    population_limitation = (
        "Synthetic fixture volumes do not represent CFPB volume."
        if is_fixture
        else "Public CFPB complaints are not a representative sample of consumer experiences."
    )
    entries: list[dict[str, str]] = []

    split_rows = pd.DataFrame(data_quality["split_distribution"])
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.bar(split_rows["split"], split_rows["rows"], color="#4c78a8")
    ax.set(title="Temporal split distribution", ylabel="Rows", xlabel="Locked partition")
    ax.tick_params(axis="x", rotation=25)
    _save(fig, figures / "01_temporal_split.png")
    entries.append(
        _figure_manifest_entry(
            "F01",
            "01_temporal_split.png",
            "Is data distributed across the preregistered temporal partitions?",
            "data_quality.json",
            "All model stages use explicit non-overlapping time partitions.",
            population_limitation,
        )
    )

    missing = pd.DataFrame(
        list(data_quality["validation"]["missingness"].items()), columns=["column", "rate"]
    ).head(12)
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.barh(missing["column"], missing["rate"], color="#72b7b2")
    ax.invert_yaxis()
    ax.set(title="Required-column missingness", xlabel="Missing fraction")
    _save(fig, figures / "02_missingness.png")
    entries.append(
        _figure_manifest_entry(
            "F02",
            "02_missingness.png",
            "Where is source missingness concentrated?",
            "data_quality.json",
            "Missing sub-fields are handled explicitly; key and target fields remain complete.",
            "Fixture missingness is generated, not observed."
            if is_fixture
            else "Narrative availability is opt-in and must not be interpreted as random missingness.",
        )
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.bar(
        comparison["candidate_id"],
        comparison["pr_auc"],
        color=[STYLE.get(v, "#777") for v in comparison["candidate_id"]],
    )
    ax.axhline(
        float(data_quality["validation"]["target_prevalence"]),
        color="black",
        linestyle="--",
        linewidth=1,
        label="overall prevalence",
    )
    ax.set(title="Locked-test predictive comparison", ylabel="PR-AUC", xlabel="Candidate")
    ax.legend()
    _save(fig, figures / "03_model_comparison.png")
    entries.append(
        _figure_manifest_entry(
            "F03",
            "03_model_comparison.png",
            "Does added modeling complexity improve locked-test ranking quality?",
            "candidate_comparison.csv",
            "The champion is selected by the frozen multi-metric rule, not model complexity.",
            "A single public-data snapshot cannot establish production generalization."
            if not is_fixture
            else "Small synthetic data can reverse rankings.",
        )
    )

    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    for candidate in comparison["candidate_id"]:
        part = predictions[predictions["candidate_id"] == candidate]
        observed, predicted = calibration_curve(
            part["target"], part["probability"], n_bins=7, strategy="quantile"
        )
        ax.plot(predicted, observed, marker="o", label=candidate, color=STYLE.get(candidate))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set(
        title="Locked-test calibration",
        xlabel="Mean predicted probability",
        ylabel="Observed fraction",
    )
    ax.legend()
    _save(fig, figures / "04_calibration.png")
    entries.append(
        _figure_manifest_entry(
            "F04",
            "04_calibration.png",
            "Are calibrated probabilities reliability-aligned?",
            "predictions.csv",
            "Calibration is fitted only on the policy partition and evaluated on the locked test.",
            "Quantile bins can hide within-bin error.",
        )
    )

    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    for candidate in comparison["candidate_id"]:
        part = predictions[predictions["candidate_id"] == candidate]
        precision, recall, _ = precision_recall_curve(part["target"], part["probability"])
        ax.plot(recall, precision, label=candidate, color=STYLE.get(candidate))
    ax.set(title="Precision-recall frontier", xlabel="Recall", ylabel="Precision")
    ax.legend()
    _save(fig, figures / "05_precision_recall.png")
    entries.append(
        _figure_manifest_entry(
            "F05",
            "05_precision_recall.png",
            "How do candidates trade precision for recall under class imbalance?",
            "predictions.csv",
            "PR curves expose rare-event ranking quality better than accuracy.",
            "Fixture prevalence is synthetic."
            if is_fixture
            else "Observed prevalence is specific to the frozen CFPB snapshot.",
        )
    )

    worst = (
        slices[slices["metric"] == "pr_auc"]
        .sort_values("value")
        .groupby("candidate_id", as_index=False)
        .first()
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    labels = worst["candidate_id"] + ":" + worst["slice_value"].astype(str)
    ax.bar(labels, worst["value"], color=[STYLE.get(v, "#777") for v in worst["candidate_id"]])
    ax.set(title="Worst eligible slice PR-AUC", ylabel="PR-AUC", xlabel="Candidate : worst slice")
    _save(fig, figures / "06_worst_slice.png")
    entries.append(
        _figure_manifest_entry(
            "F06",
            "06_worst_slice.png",
            "Does aggregate performance conceal a weak subgroup?",
            "slice_metrics.csv",
            "Release evidence includes the worst preregistered eligible slice.",
            "Slices with too few positives are marked ineligible.",
        )
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for candidate, group in scenarios.groupby("candidate_id"):
        ax.plot(
            group["review_capacity"],
            group["positive_capture"],
            marker="o",
            label=candidate,
            color=STYLE.get(candidate),
        )
    ax.set(
        title="Review capacity vs positive capture",
        xlabel="Review capacity",
        ylabel="Positive capture",
    )
    ax.legend()
    _save(fig, figures / "07_capacity_frontier.png")
    entries.append(
        _figure_manifest_entry(
            "F07",
            "07_capacity_frontier.png",
            "What operating capacity captures risk without unbounded review?",
            "decision_scenarios.csv",
            "The 20% point is the preregistered primary workload scenario.",
            "No monetary cost is inferred.",
        )
    )

    fig, ax = plt.subplots(figsize=(6.7, 4.5))
    ax.scatter(comparison["p95_batch_latency_ms"], comparison["pr_auc"], s=90)
    for _, row in comparison.iterrows():
        ax.annotate(
            row["candidate_id"],
            (row["p95_batch_latency_ms"], row["pr_auc"]),
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.set(title="Performance-resource frontier", xlabel="p95 batch latency (ms)", ylabel="PR-AUC")
    _save(fig, figures / "08_performance_latency.png")
    entries.append(
        _figure_manifest_entry(
            "F08",
            "08_performance_latency.png",
            "Is predictive gain worth operational latency?",
            "candidate_comparison.csv",
            "Production selection accounts for quality and latency.",
            "Local batch latency is hardware-specific and not a service SLO.",
        )
    )

    policy_summary = pd.DataFrame(fault_summary["policy_summary"])
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    x = np.arange(len(policy_summary))
    width = 0.25
    ax.bar(x - width, policy_summary["critical_defect_escape_rate"], width, label="CDER")
    ax.bar(x, policy_summary["false_block_rate"], width, label="False block")
    ax.bar(x + width, policy_summary["manual_review_rate"], width, label="Manual review")
    ax.set_xticks(x, policy_summary["policy_id"])
    ax.set(title="Release-policy safety trade-off", ylabel="Rate")
    ax.legend()
    _save(fig, figures / "09_release_policy_comparison.png")
    entries.append(
        _figure_manifest_entry(
            "F09",
            "09_release_policy_comparison.png",
            "Does specialty reduce unsafe promotions within review and false-block constraints?",
            "fault_benchmark_summary.json",
            "C4 must pass all claim gates before a superiority statement is emitted.",
            "Fault cases are synthetic and cannot prove production incident reduction.",
        )
    )

    unsafe = fault[fault["is_unsafe"] == True]  # noqa: E712
    heat = unsafe.pivot_table(index="fault_family", columns="policy_id", values="escaped", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(8.0, 6.2))
    image = ax.imshow(heat.fillna(0).to_numpy(), vmin=0, vmax=1, cmap="YlOrRd")
    ax.set_xticks(range(len(heat.columns)), heat.columns)
    ax.set_yticks(range(len(heat.index)), heat.index)
    ax.set(title="Fault-family escape rate (red is unsafe)")
    fig.colorbar(image, ax=ax, label="Escape rate")
    _save(fig, figures / "10_fault_heatmap.png")
    entries.append(
        _figure_manifest_entry(
            "F10",
            "10_fault_heatmap.png",
            "Which hidden defect families escape each policy?",
            "fault_benchmark_results.csv",
            "Family-level failures cannot be hidden by the aggregate CDER.",
            "Injected faults approximate, rather than reproduce, enterprise incidents.",
        )
    )

    health = unsafe[(unsafe["policy_id"] == "C4") & unsafe["health_score"].notna()]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    if len(health):
        health.boxplot(column="health_score", by="fault_family", ax=ax, rot=65)
        fig.suptitle("")
        ax.set(
            title="RSRH score by hidden fault family",
            xlabel="Fault family",
            ylabel="Reference health",
        )
    else:
        ax.text(0.5, 0.5, "No finite RSRH scores", ha="center", va="center")
        ax.set_axis_off()
    _save(fig, figures / "11_reference_health.png")
    entries.append(
        _figure_manifest_entry(
            "F11",
            "11_reference_health.png",
            "How does reference-aligned health respond across failure modes?",
            "fault_benchmark_results.csv",
            "Low health routes a release to hold/reject while hard gates remain separate.",
            "Health thresholds are benchmark-frozen and require external validation.",
        )
    )

    fig, ax = plt.subplots(figsize=(11.5, 2.8))
    ax.axis("off")
    stages = [
        "Snapshot",
        "Validate",
        "As-of\nfeatures",
        "Train &\ncalibrate",
        "Locked\nevaluate",
        "Compile\nevidence",
        "C0-C4\nqualify",
        "Approve /\nrollback",
        "Monitor",
    ]
    for i, stage in enumerate(stages):
        x0 = 0.03 + i * 0.106
        ax.text(
            x0,
            0.52,
            stage,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=.38", fc="#e8f1f8", ec="#2f6b9a"),
            transform=ax.transAxes,
        )
        if i < len(stages) - 1:
            ax.annotate(
                "",
                xy=(x0 + 0.085, 0.52),
                xytext=(x0 + 0.045, 0.52),
                arrowprops=dict(arrowstyle="->", color="#555"),
                xycoords=ax.transAxes,
            )
    ax.set_title("ModelForge-R governed end-to-end path")
    _save(fig, figures / "12_architecture.png")
    entries.append(
        _figure_manifest_entry(
            "F12",
            "12_architecture.png",
            "How does prediction evidence become a governed product action?",
            "implemented pipeline",
            "Every promotion decision is downstream of locked evidence and upstream of explicit approval.",
            "Databricks execution remains credential-dependent.",
        )
    )

    write_json(output / "figure_manifest.json", entries)
    verdict = fault_summary["claim_verdict"]
    evidence_description = (
        "a deterministic synthetic fixture" if is_fixture else "a checksum-locked public CFPB snapshot"
    )
    reproduction_command = (
        "model-forge demo --config configs/experiment_contract.yaml --clean"
        if is_fixture
        else "model-forge train --dataset <snapshot> --config configs/experiment_contract.yaml --clean"
    )
    table_html = comparison.round(5).to_html(index=False, classes="table")
    policy_html = policy_summary.round(5).to_html(index=False, classes="table")
    split_html = (
        pd.DataFrame(data_quality["split_distribution"]).round(6).to_html(index=False, classes="table")
    )
    qualification_html = pd.DataFrame(
        [
            {
                "policy": item.get("policy_id"),
                "decision": item.get("decision"),
                "reason_codes": ", ".join(item.get("reason_codes", [])),
                "alias_action": item.get("alias_action"),
            }
            for item in qualification
            if isinstance(item, dict)
        ]
    ).to_html(index=False, classes="table", escape=True)
    inference_status = str(release_evidence.get("inference_replay", {}).get("status", "UNKNOWN"))
    independent_status = str(release_evidence.get("independent_rerun", {}).get("status", "UNKNOWN"))
    failed_gates = [
        *[str(value) for value in decision.get("failed_checks", [])],
        *[str(value) for value in decision.get("unevaluated_checks", [])],
        *[str(value) for value in decision.get("reason_codes", [])],
    ]
    failed_gates_html = "".join(f"<li>{html.escape(value)}</li>" for value in failed_gates)
    claim_boundaries = [
        "W2 predictive superiority does not authorize production promotion.",
        "Public CFPB complaints are not representative of all United States consumers.",
        "This research workload is not a consumer adverse-action model.",
        "Synthetic fixture and fault results are not CFPB production evidence.",
        "No hosted CI or Databricks deployment success is claimed by this local report.",
    ]
    claim_html = "".join(f"<li>{html.escape(value)}</li>" for value in claim_boundaries)
    critical_slice = html.escape(str(release_evidence.get("metadata", {}).get("critical_slice")))
    split_boundaries = html.escape(str(run_manifest.get("split", {})))
    images = "\n".join(
        (
            f'<figure><img src="{html.escape(item["path"])}" '
            f'alt="{html.escape(item["question"])}"><figcaption><strong>'
            f"{html.escape(item['question'])}</strong><br>"
            f"{html.escape(item['interpretation'])} Limitation: "
            f"{html.escape(item['limitation'])}</figcaption></figure>"
        )
        for item in entries
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ModelForge-R evidence report</title><style>
body{{font-family:system-ui,sans-serif;max-width:1120px;margin:auto;padding:28px;color:#17202a}}
.banner{{padding:16px;background:#eef5fb;border-left:6px solid #2f6b9a}}
.table{{border-collapse:collapse;width:100%;font-size:14px}}
.table th,.table td{{border:1px solid #ccd6dd;padding:6px;text-align:right}}
.table th:first-child,.table td:first-child{{text-align:left}}
figure{{margin:28px 0;padding:16px;border:1px solid #dfe6eb;border-radius:8px}}
img{{max-width:100%;height:auto}}
figcaption{{font-size:14px;color:#455a64}}
code{{background:#f4f6f7;padding:2px 4px}}
</style></head><body>
<h1>ModelForge-R evidence report</h1>
<div class="banner"><strong>Evidence class:</strong> measured on {html.escape(evidence_description)};
the release-policy fault corpus remains synthetic. This is not production traffic or enterprise control
evidence.<br>
<strong>Release-policy claim verdict:</strong> {html.escape(verdict)}<br>
<strong>Current C4 decision:</strong> {html.escape(decision["decision"])}<br>
<strong>Predictive champion:</strong>
{html.escape(str(metrics_document.get("predictive_champion", "UNKNOWN")))}<br>
<strong>Production champion:</strong> {html.escape(str(release_evidence.get("champion_id", "UNKNOWN")))}</div>
<h2>Verified facts</h2>
<p><strong>Run ID:</strong> <code>{html.escape(str(run_manifest.get("run_id")))}</code><br>
<strong>Dataset hash:</strong>
<code>{html.escape(str(dataset_identity.get("dataset_fingerprint")))}</code><br>
<strong>Config hash:</strong> <code>{html.escape(str(run_manifest.get("config_hash")))}</code><br>
<strong>Source/code hash:</strong> <code>{html.escape(str(run_manifest.get("source_fingerprint")))}</code><br>
<strong>Split boundaries:</strong> <code>{split_boundaries}</code></p>
{split_html}
<h2>Predictive candidates</h2>{table_html}
<h2>Release policies</h2>{policy_html}{qualification_html}
<h2>Failed and incomplete gates</h2><ul>{failed_gates_html or "<li>None recorded</li>"}</ul>
<p><strong>Inference replay:</strong> {html.escape(inference_status)};<br>
<strong>Independent rerun:</strong> {html.escape(independent_status)};<br>
<strong>Critical slice:</strong> {critical_slice}, status
{html.escape(str(release_evidence.get("critical_slice_status")))},
delta {html.escape(str(release_evidence.get("critical_slice_delta")))}.</p>
<h2>Fixture results</h2><p>The C0-C4 fault benchmark shown below is deterministic synthetic evidence.
It is not real-data or production incident evidence.</p>
<h2>Real-data results</h2><p>{html.escape(evidence_description.capitalize())} produced the candidate
metrics above. They are scoped to the immutable dataset identity and configuration hash shown in this
report.</p>
<h2>Partial evidence</h2><p>CFPB metrics use a target-independent probability sample; the complete
eligible population is retained for audit but was not fully scored for model metrics.</p>
<h2>Blocked evidence</h2><p>Independent rerun status:
<strong>{html.escape(independent_status)}</strong>. Missing or failed replay evidence cannot be inferred
as a pass.</p>
<h2>Prospective evidence</h2><p>Prospective stress is monitoring-only and was not used for promotion.
Monitoring action: <strong>{html.escape(str(monitoring.get("action", "UNKNOWN")))}</strong>
({html.escape(str(monitoring.get("reason_code", "UNKNOWN")))}).</p>
<h2>Prohibited claims and exact claim boundary</h2><ul>{claim_html}</ul>
<h2>Diagnostic and decision figures</h2>{images}
<h2>Reproduction</h2><p><code>{html.escape(reproduction_command)}</code></p>
</body></html>"""
    atomic_write_text(output / "report.html", document)
    return {
        "report": str(output / "report.html"),
        "figures": len(entries),
        "claim_verdict": verdict,
    }
