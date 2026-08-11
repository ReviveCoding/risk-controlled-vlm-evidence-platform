from __future__ import annotations

import numpy as np
import pandas as pd

PRODUCTS = ["Credit card", "Credit reporting", "Mortgage", "Debt collection"]
ISSUES = ["Incorrect information", "Billing dispute", "Communication", "Payment trouble"]
COMPANIES = [f"Company_{letter}" for letter in "ABCDEFGH"]
CHANNELS = ["Web", "Phone", "Referral", "Postal mail"]
STATES = ["NY", "NJ", "CA", "TX", "FL", "IL"]
FIXTURE_SPLIT_ANCHORS = np.array(
    [
        "2019-07-01",
        "2023-07-01",
        "2024-02-01",
        "2024-05-01",
        "2024-08-01",
        "2024-11-01",
        "2025-03-01",
        "2025-09-01",
        "2026-02-01",
        "2026-06-01",
    ],
    dtype="datetime64[D]",
)
FIXTURE_SPLIT_WINDOWS = (
    ("2019-01-01", "2023-12-31"),
    ("2024-01-01", "2024-06-30"),
    ("2024-07-01", "2024-12-31"),
    ("2025-01-01", "2025-12-31"),
    ("2026-01-01", "2026-12-31"),
)


def generate_complaint_fixture(rows: int = 3200, seed: int = 20260717) -> pd.DataFrame:
    """Generate deterministic public-shaped, wholly synthetic complaint records."""

    if type(rows) is not int or rows < len(FIXTURE_SPLIT_ANCHORS):
        raise ValueError(f"rows must be an integer of at least {len(FIXTURE_SPLIT_ANCHORS)}")
    rng = np.random.default_rng(seed)
    start = np.datetime64("2019-01-01")
    end = np.datetime64("2026-07-01")
    random_rows = rows - len(FIXTURE_SPLIT_ANCHORS)
    random_offsets = rng.integers(0, int((end - start).astype(int)), random_rows)
    anchor_offsets = (FIXTURE_SPLIT_ANCHORS - start).astype(int)
    day_offsets = np.sort(np.concatenate([random_offsets, anchor_offsets]))
    sent = pd.to_datetime(start + day_offsets.astype("timedelta64[D]"))
    received = sent - pd.to_timedelta(rng.integers(0, 8, rows), unit="D")
    product = rng.choice(PRODUCTS, rows, p=[0.28, 0.32, 0.18, 0.22])
    issue = rng.choice(ISSUES, rows)
    company = rng.choice(COMPANIES, rows, p=[0.24, 0.19, 0.16, 0.13, 0.10, 0.08, 0.06, 0.04])
    channel = rng.choice(CHANNELS, rows, p=[0.70, 0.12, 0.13, 0.05])
    state = rng.choice(STATES, rows)
    narrative_present = rng.random(rows) < np.where(channel == "Web", 0.72, 0.25)
    urgency = rng.choice(["routine", "urgent", "escalated"], rows, p=[0.70, 0.23, 0.07])

    company_risk = {name: value for name, value in zip(COMPANIES, np.linspace(-1.0, 1.0, 8), strict=False)}
    product_risk = {
        "Credit card": 0.2,
        "Credit reporting": -0.3,
        "Mortgage": 0.7,
        "Debt collection": 0.4,
    }
    year = sent.year.to_numpy()
    logit = np.array([company_risk[v] for v in company])
    logit += np.array([product_risk[v] for v in product])
    logit += 0.55 * (urgency == "escalated") + 0.25 * (urgency == "urgent")
    logit += 0.25 * (channel == "Postal mail") + 0.08 * (year >= 2025)
    logit -= 3.35
    probability = 1.0 / (1.0 + np.exp(-logit))
    target = rng.binomial(1, probability)
    # A governed binary experiment cannot train or evaluate an empty class. Preserve
    # the probabilistic fixture while deterministically repairing only degenerate
    # contract partitions (most commonly seen in deliberately small test fixtures).
    for window_start, window_end in FIXTURE_SPLIT_WINDOWS:
        mask = sent.to_series(index=np.arange(rows)).between(window_start, window_end).to_numpy()
        indices = np.flatnonzero(mask)
        if len(indices) < 2:
            raise RuntimeError("fixture split anchors failed to populate a contract partition")
        if target[indices].sum() == 0:
            target[indices[np.argmax(probability[indices])]] = 1
        elif target[indices].sum() == len(indices):
            target[indices[np.argmin(probability[indices])]] = 0

    narrative = np.where(
        narrative_present,
        np.char.add(
            np.char.add(np.char.add("Synthetic ", urgency.astype(str)), " complaint about "),
            np.char.lower(product.astype(str)),
        ),
        "",
    )
    missing_sub = rng.random(rows) < 0.17
    sub_product = np.asarray(np.char.add(product.astype(str), " subtype"), dtype=object)
    sub_product[missing_sub] = None
    sub_issue = np.asarray(np.char.add(issue.astype(str), " detail"), dtype=object)
    sub_issue[rng.random(rows) < 0.22] = None

    frame = pd.DataFrame(
        {
            "complaint_id": [f"SYN-{seed}-{i:06d}" for i in range(rows)],
            "date_received": received,
            "date_sent_to_company": sent,
            "product": product,
            "sub_product": sub_product,
            "issue": issue,
            "sub_issue": sub_issue,
            "company": company,
            "submitted_via": channel,
            "complaint_narrative": narrative,
            "state": state,
            "tags": np.where(rng.random(rows) < 0.08, "Older American", ""),
            "untimely_response": target.astype(int),
        }
    )
    return frame.sort_values(["date_sent_to_company", "complaint_id"]).reset_index(drop=True)
