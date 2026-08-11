from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class EvidenceCell:
    cell_id: str
    defect_probability: float
    harm: float
    cost: float
    redundancy: float = 0.0
    mandatory: bool = False

    @property
    def value(self) -> float:
        return self.defect_probability * self.harm * (1.0 - min(max(self.redundancy, 0.0), 1.0))


def select_evidence_boundary(cells: list[EvidenceCell], budget: float) -> dict:
    """ABC-inspired evidence expansion/pruning under a fixed evaluation budget."""

    if not math.isfinite(budget) or budget < 0:
        raise ValueError("evidence budget must be finite and nonnegative")
    if len({cell.cell_id for cell in cells}) != len(cells):
        raise ValueError("evidence cell ids must be unique")
    for cell in cells:
        if (
            not cell.cell_id
            or not all(
                math.isfinite(value)
                for value in (cell.defect_probability, cell.harm, cell.cost, cell.redundancy)
            )
            or not 0 <= cell.defect_probability <= 1
            or cell.harm < 0
            or cell.cost < 0
            or not 0 <= cell.redundancy <= 1
        ):
            raise ValueError(f"invalid evidence cell: {cell.cell_id}")

    mandatory = [cell for cell in cells if cell.mandatory]
    mandatory_cost = sum(cell.cost for cell in mandatory)
    if mandatory_cost > budget:
        return {
            "selected": [cell.cell_id for cell in mandatory],
            "cost": mandatory_cost,
            "budget": budget,
            "feasible": False,
            "reason": "MANDATORY_EVIDENCE_EXCEEDS_BUDGET",
        }
    selected = list(mandatory)
    remaining = sorted(
        (cell for cell in cells if not cell.mandatory),
        key=lambda cell: (cell.value / max(cell.cost, 1e-12), cell.value, cell.cell_id),
        reverse=True,
    )
    spent = mandatory_cost
    for cell in remaining:
        if spent + cell.cost <= budget:
            selected.append(cell)
            spent += cell.cost
    return {
        "selected": [cell.cell_id for cell in selected],
        "cost": float(spent),
        "budget": float(budget),
        "feasible": True,
        "value": float(sum(cell.value for cell in selected)),
        "mandatory_preserved": all(cell.cell_id in {item.cell_id for item in selected} for cell in mandatory),
    }
