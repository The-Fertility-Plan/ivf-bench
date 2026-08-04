"""Backfill inference cost on the ablation arms.

The ablation runs were launched without --input-price/--output-price, so every
response recorded its token counts but a cost of zero. The prices are on record
in each main run's run_meta.json, and the arms used the same endpoints, so the
cost is recoverable exactly rather than estimated. Without this the study-cost
table understates what the ablation actually cost.

Usage: python scripts/backfill_ablation_cost.py [--apply]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "data" / "runs"
SUFFIXES = ("-noimg", "-nograde", "-neither")
# arm directory prefix -> the main run whose recorded prices apply
ALIAS = {"google__gemini-2_5-flash": "gemini-2_5-flash"}


def main() -> None:
    apply = "--apply" in sys.argv
    total = 0.0
    for run in sorted(RUNS.iterdir()):
        if not run.is_dir() or not any(run.name.endswith(s) for s in SUFFIXES):
            continue
        base = run.name
        for s in SUFFIXES:
            base = base[: -len(s)] if base.endswith(s) else base
        base = ALIAS.get(base, base)
        main_meta = RUNS / base / "run_meta.json"
        if not main_meta.exists():
            print(f"  !! no main run for {run.name} (looked for {base})")
            continue
        m = json.loads(main_meta.read_text())
        pin, pout = m.get("input_price_per_m", 0.0), m.get("output_price_per_m", 0.0)
        if not (pin or pout):
            print(f"  !! main run {base} has no prices recorded")
            continue

        arm_cost, n = 0.0, 0
        for f in sorted((run / "responses").glob("*.json")):
            d = json.loads(f.read_text())
            if d.get("cost_usd"):
                continue
            c = (d.get("prompt_tokens", 0) * pin
                 + d.get("completion_tokens", 0) * pout) / 1_000_000
            if not c:
                continue
            arm_cost += c
            n += 1
            if apply:
                d["cost_usd"] = c
                f.write_text(json.dumps(d, indent=2))
        if apply and arm_cost:
            am = json.loads((run / "run_meta.json").read_text())
            am["input_price_per_m"] = pin
            am["output_price_per_m"] = pout
            am["cost_backfilled_from"] = base
            (run / "run_meta.json").write_text(json.dumps(am, indent=2))
        total += arm_cost
        print(f"  {run.name:52s} {n:4d} responses  ${arm_cost:7.2f}"
              f"   (in ${pin}/M, out ${pout}/M)")

    print(f"\n  {'ablation inference total':52s}      ${total:7.2f}")
    print("  " + ("written" if apply else "dry run, pass --apply to write"))


if __name__ == "__main__":
    main()
