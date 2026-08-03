"""Check every load-bearing number in the paper against the artifacts.

Run before any submission. Exits non-zero on the first mismatch so a wrong
number cannot survive a build the way several already have.
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TEX = (ROOT / "arxiv" / "ivf_bench.tex").read_text()
R = ["morphological_accuracy", "clinical_integration", "reasoning_coherence",
     "guideline_alignment", "actionability"]


def board(name):
    d = json.loads((ROOT / "data/runs" / name).read_text())
    rows = d if isinstance(d, list) else d.get("models", d.get("leaderboard", []))
    return {r["model"]: r for r in rows}


T, H = board("leaderboard.json"), board("leaderboard_held_out.json")
PA = json.loads((ROOT / "data/runs/paper_analysis.json").read_text())
AB = json.loads((ROOT / "data/runs/image_ablation.json").read_text())
OURS, BASE = "ivf-bench-qwen9b-vlm-orpo", "qwen/qwen3.5-9b"

checks: list[tuple[str, bool, str]] = []


def claim(label, ok, detail=""):
    checks.append((label, bool(ok), detail))


def in_tex(s):
    return s in TEX


# --- corpus -----------------------------------------------------------------
cases = {d: len(glob.glob(str(ROOT / "data" / d / "*.json")))
         for d in ("cases", "validation_cases", "held_out_cases")}
claim("753 cases total", sum(cases.values()) == 753 and in_tex("753"))
claim("550 / 100 / 103 split", (cases["cases"], cases["validation_cases"],
                                cases["held_out_cases"]) == (550, 100, 103))

day = {}
for d in ("cases", "validation_cases", "held_out_cases"):
    for f in glob.glob(str(ROOT / "data" / d / "*.json")):
        c = json.loads(Path(f).read_text())
        day[c["case_id"]] = c["lab_data"]["transfer_day"]
d5, d4 = sum(1 for v in day.values() if v == 5), sum(1 for v in day.values() if v == 4)
claim("641 day-5 and 112 day-4 transfers", d5 == 641 and d4 == 112 and in_tex("641") and in_tex("112"),
      f"{d5}/{d4}")

resp = sum(len(glob.glob(f"{d}/responses/*.json"))
           for d in glob.glob(str(ROOT / "data/runs/*")) if "-noimg" not in d and Path(d).is_dir())
claim("5,194 released responses", resp == 5194 and in_tex("5,194"), str(resp))

# --- leaderboard ------------------------------------------------------------
for lbl, b, key in (("test", T, "4.17"), ("held-out", H, "4.11")):
    claim(f"ours {lbl} {key}", f"{b[OURS]['overall_score']:.2f}" == key and in_tex(key))
claim("Opus held-out 4.01", f"{H['global.anthropic.claude-opus-4-6-v1']['overall_score']:.2f}" == "4.01")
claim("base held-out 3.31", f"{H[BASE]['overall_score']:.2f}" == "3.31")
gain = 100 * (H[OURS]["overall_score"] / H[BASE]["overall_score"] - 1)
claim("24% base-to-trained gain", abs(gain - 24.0) < 0.15 and in_tex("24.0"), f"{gain:.2f}%")
claim("max AUROC 0.562", abs(max(r["auroc"] for r in H.values()) - 0.562) < 5e-4 and in_tex("0.562"))
claim("min AUROC 0.450", abs(min(r["auroc"] for r in H.values()) - 0.450) < 5e-4 and in_tex("0.450"))

# --- the weakest-rubric claim ----------------------------------------------
for lbl, b in (("test", T), ("held-out", H)):
    means = {r: np.mean([x[r] for x in b.values()]) for r in R}
    lowest = min(means, key=means.get)
    claim(f"{lbl}: guideline alignment is the lowest mean rubric",
          lowest == "guideline_alignment",
          f"lowest={lowest} {means[lowest]:.3f}, morph={means['morphological_accuracy']:.3f}")

# --- base rate --------------------------------------------------------------
for lbl, d, want in (("test", "cases", 0.2278), ("held-out", "held_out_cases", 0.2274)):
    y = np.array([int(json.loads(Path(f).read_text())["real_outcome"]["clinical_pregnancy"])
                  for f in glob.glob(str(ROOT / "data" / d / "*.json"))])
    p = y.mean()
    claim(f"{lbl} base-rate Brier {want}", abs(p * (1 - p) - want) < 5e-4 and in_tex(str(want)),
          f"{p*(1-p):.4f}")

worse = [(m, s) for s, b in (("test", T), ("held", H)) for m, r in b.items()
         if r["brier_score"] <= (0.2278 if s == "test" else 0.2274)]
claim("no system beats the base-rate constant", not worse, str(worse))
n_beat_half = sum(1 for b in (T, H) for r in b.values() if r["brier_score"] < 0.25)
claim("nine of sixteen beat a constant 0.50", n_beat_half == 9 and in_tex("nine of the sixteen"),
      str(n_beat_half))

# --- intervals --------------------------------------------------------------
pairs = {(p["higher"], p["lower"]): p for p in PA["held_out"]["adjacent_rank_tests"]}
op = pairs.get((OURS, "global.anthropic.claude-opus-4-6-v1"))
claim("ours over Opus +0.10 [0.03,0.17] p=0.007",
      op and abs(op["diff"] - 0.10) < 0.006 and abs(op["p"] - 0.007) < 0.001 and in_tex("p{=}0.007"),
      str(op and (op["diff"], op["ci95"], op["p"])))

# --- ablation ---------------------------------------------------------------
for m, r in AB.items():
    mm = r["per_rubric"]["morphological_accuracy"]
    for lbl, v in (("overall image", r["overall_with"]), ("overall none", r["overall_without"]),
                   ("morph image", mm["with"]), ("morph none", mm["without"])):
        claim(f"ablation {m} {lbl} {v:.2f}", in_tex(f"{v:.2f}"))
deltas = [r["per_rubric"]["morphological_accuracy"]["delta"] for r in AB.values()]
claim("ablation morphology range 0.07 to 0.23",
      abs(min(deltas) - 0.07) < 0.005 and abs(max(deltas) - 0.23) < 0.005 and in_tex("0.07") and in_tex("0.23"),
      f"{min(deltas):.3f}-{max(deltas):.3f}")

# --- claims that must NOT appear -------------------------------------------
for bad, why in (("one twentieth", "superseded cost ratio"),
                 ("expert Gardner annotation", "labels are silver, not gold"),
                 ("Morphology grounding is the weakest rubric throughout", "false"),
                 ("550 benchmark-derived preference pairs", "trained on 500"),
                 ("carry no information\nabout which patients conceived", "false")):
    claim(f"retracted claim absent: {why}", not in_tex(bad))

# --- report -----------------------------------------------------------------
bad = [c for c in checks if not c[1]]
for label, ok, detail in checks:
    if not ok:
        print(f"  FAIL  {label}" + (f"   [{detail}]" if detail else ""))
print(f"\n{len(checks) - len(bad)}/{len(checks)} checks pass")
if bad:
    sys.exit(1)
print("every load-bearing number in the paper matches the artifacts")
