"""Total study cost, computed from the artifacts rather than from memory.

Every response record carries its inference cost and every score record its judge
cost, so the whole bill can be reconstructed. Superseded work (confounded arms,
zero-score failures, quarantined runs) is counted separately: it was really spent,
but it backs no reported number.

Usage: python scripts/study_cost.py
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "data" / "runs"

# Lambda Cloud, 2x H100 80GB SXM5, provisioned 2026-05-13 14:22 to 2026-05-15 17:50
COMPUTE_HOURS, COMPUTE_RATE = 51.5, 8.38

SCORE_DIRS = {
    "scores": "judging, reported",
    "scores_sighted": "judging, sighted judge",
    "scores_sonnet": "judging, cross-judge",
    "scores_stale_cot": "superseded: chain-of-thought confound",
    "scores_zero_judge_failure": "superseded: zero-score failures",
    "scores_opus_judged": "superseded: mixed judge",
    "scores_prompt_confounded": "superseded: ablation prompt confound",
    "scores_grade_confounded": "superseded: grade confound",
    "scores_sonnet_sighted": "judging, cross-judge sighted",
}


def main() -> None:
    inference = defaultdict(float)
    judging = defaultdict(float)

    for run in sorted(RUNS.iterdir()):
        if not run.is_dir() or run.name.startswith("_"):
            continue
        arm = ("ablation arm" if any(s in run.name for s in ("-noimg", "-nograde", "-neither"))
               else "main runs")
        for f in glob.glob(str(run / "responses" / "*.json")):
            inference[arm] += json.loads(Path(f).read_text()).get("cost_usd", 0) or 0
        for sub, label in SCORE_DIRS.items():
            for f in glob.glob(str(run / sub / "*.json")):
                judging[label] += json.loads(Path(f).read_text()).get("judge_cost_usd", 0) or 0

    print("REPORTED WORK")
    compute = COMPUTE_HOURS * COMPUTE_RATE
    print(f"  {'GPU compute (51.5 h x $8.38)':44s} ${compute:9.2f}")
    for k, v in sorted(inference.items()):
        print(f"  {'inference, ' + k:44s} ${v:9.2f}")
    reported = compute + sum(inference.values())
    for label, v in sorted(judging.items()):
        if not label.startswith("superseded"):
            print(f"  {label:44s} ${v:9.2f}")
            reported += v
    print(f"  {'':44s} {'-' * 10}")
    print(f"  {'subtotal, backs a reported number':44s} ${reported:9.2f}")

    # quarantined directories are skipped by the loop above; count them here so
    # the grand total is what was actually spent rather than what was kept
    quarantined = 0.0
    for f in glob.glob(str(RUNS / "_*" / "**" / "*.json"), recursive=True):
        d = json.loads(Path(f).read_text())
        quarantined += (d.get("judge_cost_usd", 0) or 0) + (d.get("cost_usd", 0) or 0)
    if quarantined:
        judging["superseded: quarantined runs"] = quarantined

    print("\nSUPERSEDED WORK (spent, backs nothing reported)")
    sup = 0.0
    for label, v in sorted(judging.items()):
        if label.startswith("superseded"):
            print(f"  {label:44s} ${v:9.2f}")
            sup += v
    print(f"  {'':44s} {'-' * 10}")
    print(f"  {'subtotal, superseded':44s} ${sup:9.2f}")

    print(f"\n  {'TOTAL SPEND':44s} ${reported + sup:9.2f}")
    print(f"\n  Note: our post-trained model and the base 9B on held out were served on the")
    print(f"  GPU line above, so their inference cost is recorded as zero per response.")

    (RUNS / "study_cost.json").write_text(json.dumps(
        {"compute": round(compute, 2),
         "inference": {k: round(v, 2) for k, v in inference.items()},
         "judging": {k: round(v, 2) for k, v in judging.items()},
         "reported_total": round(reported, 2),
         "superseded_total": round(sup, 2),
         "grand_total": round(reported + sup, 2)}, indent=2))
    print(f"\n  wrote {RUNS / 'study_cost.json'}")


if __name__ == "__main__":
    main()
