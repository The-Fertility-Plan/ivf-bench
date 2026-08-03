"""Outcome consistency metrics for IVF-Bench evaluation."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import numpy as np


def extract_probability(text: str) -> Optional[float]:
    """Extract implantation probability from model response text.

    Looks for patterns like:
    - "20-30%" -> 0.25 (midpoint)
    - "approximately 25%" -> 0.25
    - "0.20 to 0.30" -> 0.25
    - "15%" -> 0.15
    """
    if not text:
        return None

    # Pattern: range like "20-30%" or "20% to 30%"
    range_pct = re.findall(
        r"(\d{1,3}(?:\.\d+)?)\s*(?:%|percent)\s*(?:to|-|–)\s*(\d{1,3}(?:\.\d+)?)\s*(?:%|percent)?",
        text, re.IGNORECASE,
    )
    if range_pct:
        low, high = float(range_pct[-1][0]), float(range_pct[-1][1])
        if 0 < low <= 100 and 0 < high <= 100:
            return round((low + high) / 200, 4)

    # Pattern: single percentage like "25%" or "approximately 25%"
    single_pct = re.findall(
        r"(?:approximately|about|around|roughly|~|≈)?\s*(\d{1,3}(?:\.\d+)?)\s*(?:%|percent)",
        text, re.IGNORECASE,
    )
    if single_pct:
        val = float(single_pct[-1])
        if 0 < val <= 100:
            return round(val / 100, 4)

    # Pattern: decimal like "0.25" or "probability of 0.25"
    decimal = re.findall(
        r"(?:probability|likelihood|chance)\s+(?:of\s+)?(?:approximately\s+|about\s+|around\s+)?"
        r"(0\.\d+)",
        text, re.IGNORECASE,
    )
    if decimal:
        val = float(decimal[-1])
        if 0 < val < 1:
            return round(val, 4)

    return None


def compute_outcome_metrics(
    scores_dir: Path,
    cases_dir: Path,
) -> dict:
    """Compute Brier score and AUROC from extracted probabilities vs real outcomes.

    Returns dict with brier_score, auroc, n_valid, n_total.
    """
    from sklearn.metrics import brier_score_loss, roc_auc_score

    probs = []
    actuals_lb = []
    actuals_cp = []
    n_total = 0
    n_missing = 0

    for score_file in sorted(scores_dir.glob("IVF-BENCH-*.json")):
        case_id = score_file.stem
        case_file = cases_dir / f"{case_id}.json"
        if not case_file.exists():
            continue
        n_total += 1

        score_data = json.loads(score_file.read_text())
        case_data = json.loads(case_file.read_text())

        prob = score_data.get("extracted_probability")
        if prob is None:
            n_missing += 1
            continue

        outcome = case_data["real_outcome"]
        probs.append(prob)
        actuals_lb.append(int(outcome["live_birth"]))
        actuals_cp.append(int(outcome["clinical_pregnancy"]))

    result = {
        "n_total": n_total,
        "n_with_probability": len(probs),
        "n_missing_probability": n_missing,
    }

    if len(probs) < 10:
        result["brier_score_lb"] = None
        result["brier_score_cp"] = None
        result["auroc_lb"] = None
        result["auroc_cp"] = None
        return result

    probs_arr = np.array(probs)
    lb_arr = np.array(actuals_lb)
    cp_arr = np.array(actuals_cp)

    # Brier scores (lower is better, 0 = perfect)
    result["brier_score_lb"] = round(float(brier_score_loss(lb_arr, probs_arr)), 4)
    result["brier_score_cp"] = round(float(brier_score_loss(cp_arr, probs_arr)), 4)

    # AUROC (higher is better, 1.0 = perfect, 0.5 = random)
    # Only compute if both classes present
    if len(set(actuals_lb)) > 1:
        result["auroc_lb"] = round(float(roc_auc_score(lb_arr, probs_arr)), 4)
    else:
        result["auroc_lb"] = None

    if len(set(actuals_cp)) > 1:
        result["auroc_cp"] = round(float(roc_auc_score(cp_arr, probs_arr)), 4)
    else:
        result["auroc_cp"] = None

    return result
