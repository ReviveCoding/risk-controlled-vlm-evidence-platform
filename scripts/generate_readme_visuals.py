from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as element_tree

CAPACITIES = (0.05, 0.10, 0.20, 0.30, 0.40)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _coverage(payload: dict[str, Any]) -> list[dict[str, Any]]:
    records = payload.get("risk_coverage")
    if not isinstance(records, list):
        raise ValueError("risk_coverage must be a list")

    records = sorted(records, key=lambda row: float(row["capacity_fraction"]))
    capacities = tuple(float(row["capacity_fraction"]) for row in records)

    if capacities != CAPACITIES:
        raise ValueError(f"Unexpected review capacities: {capacities}")

    return records


def _record_at(records: list[dict[str, Any]], capacity: float) -> dict[str, Any]:
    for record in records:
        if float(record["capacity_fraction"]) == capacity:
            return record
    raise ValueError(f"Missing review capacity: {capacity}")


def _x(capacity: float, left: float, width: float) -> float:
    return left + ((capacity - 0.05) / 0.35) * width


def _y(value: float, low: float, high: float, top: float, height: float) -> float:
    return top + height - ((value - low) / (high - low)) * height


def _polyline(
    records: list[dict[str, Any]],
    metric: str,
    multiplier: float,
    low: float,
    high: float,
    left: float,
    width: float,
    top: float,
    height: float,
) -> str:
    points: list[str] = []
    for record in records:
        x_value = _x(float(record["capacity_fraction"]), left, width)
        y_value = _y(float(record[metric]) * multiplier, low, high, top, height)
        points.append(f"{x_value:.1f},{y_value:.1f}")
    return " ".join(points)


def _circle_svg(
    records: list[dict[str, Any]],
    metric: str,
    multiplier: float,
    low: float,
    high: float,
    left: float,
    width: float,
    top: float,
    height: float,
    css_class: str,
) -> list[str]:
    circles: list[str] = []
    for record in records:
        x_value = _x(float(record["capacity_fraction"]), left, width)
        y_value = _y(float(record[metric]) * multiplier, low, high, top, height)
        circles.append(f'<circle cx="{x_value:.1f}" cy="{y_value:.1f}" r="4.5" class="{css_class}"/>')
    return circles


def _text(x_value: float, y_value: float, value: str, css_class: str, anchor: str = "start") -> str:
    return (
        f'<text x="{x_value:.1f}" y="{y_value:.1f}" class="{css_class}" '
        f'text-anchor="{anchor}">{html.escape(value)}</text>'
    )


def _panel(
    title: str,
    subtitle: str,
    metric: str,
    multiplier: float,
    low: float,
    high: float,
    ticks: tuple[float, ...],
    suffix: str,
    baseline: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    left: float,
) -> list[str]:
    top = 112.0
    panel_width = 536.0
    panel_height = 352.0
    plot_left = left + 58.0
    plot_top = top + 62.0
    plot_width = panel_width - 84.0
    plot_height = panel_height - 116.0
    selected_x = _x(0.20, plot_left, plot_width)
    parts = [
        (
            f'<rect x="{left:.1f}" y="{top:.1f}" '
            f'width="{panel_width:.1f}" height="{panel_height:.1f}" '
            'class="panel"/>'
        ),
        _text(left + 24, top + 31, title, "panel-title"),
        _text(left + 24, top + 51, subtitle, "panel-subtitle"),
    ]

    for tick in ticks:
        y_value = _y(tick, low, high, plot_top, plot_height)
        parts.append(
            f'<line x1="{plot_left:.1f}" y1="{y_value:.1f}" '
            f'x2="{plot_left + plot_width:.1f}" y2="{y_value:.1f}" class="grid"/>'
        )
        parts.append(_text(plot_left - 9, y_value + 4, f"{tick:.0f}{suffix}", "axis", "end"))

    for capacity in CAPACITIES:
        x_value = _x(capacity, plot_left, plot_width)
        parts.append(
            f'<line x1="{x_value:.1f}" y1="{plot_top:.1f}" '
            f'x2="{x_value:.1f}" y2="{plot_top + plot_height:.1f}" class="grid-light"/>'
        )
        parts.append(_text(x_value, plot_top + plot_height + 24, f"{capacity:.0%}", "axis", "middle"))

    baseline_points = _polyline(
        baseline, metric, multiplier, low, high, plot_left, plot_width, plot_top, plot_height
    )
    candidate_points = _polyline(
        candidate, metric, multiplier, low, high, plot_left, plot_width, plot_top, plot_height
    )

    parts.extend(
        [
            f'<line x1="{plot_left:.1f}" y1="{plot_top + plot_height:.1f}" '
            f'x2="{plot_left + plot_width:.1f}" y2="{plot_top + plot_height:.1f}" class="axis-line"/>',
            f'<line x1="{plot_left:.1f}" y1="{plot_top:.1f}" '
            f'x2="{plot_left:.1f}" y2="{plot_top + plot_height:.1f}" class="axis-line"/>',
            f'<line x1="{selected_x:.1f}" y1="{plot_top:.1f}" '
            f'x2="{selected_x:.1f}" y2="{plot_top + plot_height:.1f}" class="selected-line"/>',
            _text(selected_x, plot_top - 9, "20% selected point", "selected-label", "middle"),
            f'<polyline points="{baseline_points}" class="baseline-line"/>',
            f'<polyline points="{candidate_points}" class="candidate-line"/>',
        ]
    )
    parts.extend(
        _circle_svg(
            baseline,
            metric,
            multiplier,
            low,
            high,
            plot_left,
            plot_width,
            plot_top,
            plot_height,
            "baseline-dot",
        )
    )
    parts.extend(
        _circle_svg(
            candidate,
            metric,
            multiplier,
            low,
            high,
            plot_left,
            plot_width,
            plot_top,
            plot_height,
            "candidate-dot",
        )
    )
    parts.extend(
        [
            f'<circle cx="{left + 24:.1f}" cy="{top + panel_height - 20:.1f}" r="4.5" class="baseline-dot"/>',
            _text(left + 34, top + panel_height - 16, "Frozen handcrafted baseline", "legend"),
            (
                f'<circle cx="{left + panel_width - 192:.1f}" '
                f'cy="{top + panel_height - 20:.1f}" r="4.5" '
                'class="candidate-dot"/>'
            ),
            _text(left + panel_width - 182, top + panel_height - 16, "Learned calibrated router", "legend"),
        ]
    )
    return parts


def render_svg(evidence_dir: Path) -> str:
    baseline = _coverage(_read_json(evidence_dir / "baseline_metrics.json"))
    candidate = _coverage(_read_json(evidence_dir / "candidate_metrics.json"))
    summary = _read_json(evidence_dir / "public_evidence_summary.json")
    selected = summary["selected_operating_point"]

    if float(selected["review_capacity_fraction"]) != 0.20:
        raise ValueError("Expected the public evidence selected operating point to be 20%")

    baseline_selected = _record_at(baseline, 0.20)
    candidate_selected = _record_at(candidate, 0.20)

    expected_values = (
        (baseline_selected["residual_weighted_risk"], selected["baseline"]["residual_weighted_risk"]),
        (candidate_selected["residual_weighted_risk"], selected["candidate"]["residual_weighted_risk"]),
        (baseline_selected["critical_error_capture"], selected["baseline"]["critical_error_capture"]),
        (candidate_selected["critical_error_capture"], selected["candidate"]["critical_error_capture"]),
    )
    if any(float(observed) != float(expected) for observed, expected in expected_values):
        raise ValueError("Public evidence summary disagrees with capacity-curve metrics at 20%")

    baseline_risk = int(float(baseline_selected["residual_weighted_risk"]))
    candidate_risk = int(float(candidate_selected["residual_weighted_risk"]))
    baseline_capture = 100 * float(baseline_selected["critical_error_capture"])
    candidate_capture = 100 * float(candidate_selected["critical_error_capture"])
    risk_reduction = 100 * (baseline_risk - candidate_risk) / baseline_risk
    capture_gain = candidate_capture - baseline_capture
    ci_low = float(summary["paired_bootstrap"]["ci95_low"])
    ci_high = float(summary["paired_bootstrap"]["ci95_high"])

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            '<svg xmlns="http://www.w3.org/2000/svg" width="1200" '
            'height="620" viewBox="0 0 1200 620" role="img" '
            'aria-labelledby="title desc">'
        ),
        '<title id="title">Held-out risk-routing performance across review capacities</title>',
        (
            '<desc id="desc">Comparison of a frozen handcrafted baseline and a '
            "learned calibrated router across five constrained human-review "
            "capacities.</desc>"
        ),
        "<style>",
        ".background { fill: #FFFFFF; }",
        ".title { fill: #111827; font: 700 28px Arial, Helvetica, sans-serif; }",
        ".subtitle { fill: #4B5563; font: 400 15px Arial, Helvetica, sans-serif; }",
        ".panel { fill: #F9FAFB; stroke: #D1D5DB; stroke-width: 1; rx: 14; }",
        ".panel-title { fill: #111827; font: 700 18px Arial, Helvetica, sans-serif; }",
        ".panel-subtitle { fill: #6B7280; font: 400 13px Arial, Helvetica, sans-serif; }",
        ".axis { fill: #4B5563; font: 400 12px Arial, Helvetica, sans-serif; }",
        ".axis-line { stroke: #9CA3AF; stroke-width: 1.2; }",
        ".grid { stroke: #E5E7EB; stroke-width: 1; }",
        ".grid-light { stroke: #F3F4F6; stroke-width: 1; }",
        ".baseline-line { fill: none; stroke: #6B7280; stroke-width: 3; stroke-dasharray: 7 5; }",
        ".candidate-line { fill: none; stroke: #2563EB; stroke-width: 3.5; }",
        ".baseline-dot { fill: #6B7280; }",
        ".candidate-dot { fill: #2563EB; }",
        ".legend { fill: #374151; font: 400 12px Arial, Helvetica, sans-serif; }",
        ".selected-line { stroke: #0F766E; stroke-width: 1.5; stroke-dasharray: 4 4; }",
        ".selected-label { fill: #0F766E; font: 700 12px Arial, Helvetica, sans-serif; }",
        ".callout { fill: #ECFDF5; stroke: #5EEAD4; stroke-width: 1; rx: 12; }",
        ".callout-label { fill: #065F46; font: 700 14px Arial, Helvetica, sans-serif; }",
        ".callout-value { fill: #064E3B; font: 700 24px Arial, Helvetica, sans-serif; }",
        ".callout-detail { fill: #065F46; font: 400 12px Arial, Helvetica, sans-serif; }",
        "</style>",
        '<rect width="1200" height="620" class="background"/>',
        _text(42, 48, "Held-out risk-routing performance across constrained review capacity", "title"),
        _text(
            42,
            74,
            (
                "Synthetic, group-held-out operational-error benchmark. "
                "Figure is generated from public final evidence."
            ),
            "subtitle",
        ),
    ]
    parts.extend(
        _panel(
            "Residual weighted risk",
            "Lower is better",
            "residual_weighted_risk",
            1.0,
            0.0,
            1100.0,
            (0.0, 250.0, 500.0, 750.0, 1000.0),
            "",
            baseline,
            candidate,
            42.0,
        )
    )
    parts.extend(
        _panel(
            "Critical-error capture",
            "Higher is better",
            "critical_error_capture",
            100.0,
            0.0,
            80.0,
            (0.0, 20.0, 40.0, 60.0, 80.0),
            "%",
            baseline,
            candidate,
            622.0,
        )
    )
    parts.extend(
        [
            '<rect x="42" y="504" width="1116" height="82" class="callout"/>',
            _text(70, 532, "Selected operating point: 20% human-review capacity", "callout-label"),
            _text(70, 562, f"{baseline_risk} → {candidate_risk}", "callout-value"),
            _text(224, 562, f"{risk_reduction:.1f}% lower residual weighted risk", "callout-detail"),
            _text(570, 562, f"{baseline_capture:.1f}% → {candidate_capture:.1f}%", "callout-value"),
            _text(774, 562, f"+{capture_gain:.1f} pp critical-error capture", "callout-detail"),
            _text(
                70,
                580,
                f"Paired bootstrap residual-risk difference: 95% CI [{ci_low:.2f}, {ci_high:.2f}]",
                "callout-detail",
            ),
            "</svg>",
        ]
    )
    svg = "\n".join(parts) + "\n"
    element_tree.fromstring(svg)
    return svg


def write_svg(evidence_dir: Path, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_svg(evidence_dir), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate README SVG from public evidence.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("docs/assets/risk_routing_performance.svg"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    expected = render_svg(root / "reports" / "final_run")

    if args.check:
        if not output.is_file() or output.read_text(encoding="utf-8") != expected:
            raise SystemExit(f"Generated SVG is stale or missing: {output}")
        print("README_VISUAL_SVG_CHECK=PASS")
        return 0

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(expected, encoding="utf-8")
    print(f"README_VISUAL_SVG_WRITTEN={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
