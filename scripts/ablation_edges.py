"""All four edges of the two-by-two, per rubric, paired on shared cases.

ablation_2x2.py reports the design as conceived. This reports what each edge can
actually support, because one arm turned out not to mean what a two-by-two
assumes it means: a model given neither the image nor the grade declines to
describe the embryo, and the morphology rubric scores an honest refusal as a
middling success rather than as a failure, so the 'neither' arm is not a floor.
The edges that avoid that arm, and the rubrics other than morphology, remain
interpretable, and this script separates them.

Usage: python scripts/ablation_edges.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "data" / "runs"
SCORES = "scores_sighted"
RUBRICS = ["morphological_accuracy", "clinical_integration", "reasoning_coherence",
           "guideline_alignment", "actionability"]
IDS = {Path(f).stem for f in glob.glob(str(ROOT / "data/cases/*.json"))}
RNG = np.random.default_rng(42)

SYSTEMS = {
    "GPT-5.4": "gpt-5_4-2026-03-05",
    "Opus 4.6": "global_anthropic_claude-opus-4-6-v1",
    "Qwen 397B": "qwen__qwen3_5-397b-a17b",
    "Gemini 2.5 Flash": "gemini-2_5-flash",
}
ALIAS = {"gemini-2_5-flash": "google__gemini-2_5-flash"}
ARMS = {"image + grade": "", "grade only": "-noimg",
        "image only": "-nograde", "neither": "-neither"}
# (high arm, low arm, what the difference isolates)
EDGES = [
    ("image + grade", "image only", "the grade, when the model can see the embryo"),
    ("grade only", "neither", "the grade, when the model cannot"),
    ("image + grade", "grade only", "the image, when the model has the grade"),
    ("image only", "neither", "the image, when the model has no grade"),
]


def load(base: str, suffix: str) -> dict[str, dict]:
    run = (ALIAS.get(base, base) + suffix) if suffix else base
    if not (RUNS / run).exists():
        run = base + suffix
    out = {}
    for f in glob.glob(str(RUNS / run / SCORES / "*.json")):
        if Path(f).stem not in IDS:
            continue
        d = json.loads(Path(f).read_text())
        sc = {r["rubric"]: r["score"] for r in d["rubrics"] if r["score"] > 0}
        if len(sc) == len(RUBRICS):
            sc["overall"] = float(np.mean([sc[r] for r in RUBRICS]))
            out[Path(f).stem] = sc
    return out


def paired(a: np.ndarray, b: np.ndarray):
    d = a - b
    idx = RNG.integers(0, len(d), size=(10_000, len(d)))
    m = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), \
        float(2 * min((m <= 0).mean(), (m >= 0).mean()))


def main() -> None:
    data = {lbl: {arm: load(base, sfx) for arm, sfx in ARMS.items()}
            for lbl, base in SYSTEMS.items()}

    print("=" * 90)
    print("ARM MEANS, on the cases each system answered in that arm")
    print("=" * 90)
    print(f"{'system':18s} {'arm':14s} {'n':>4s} {'overall':>8s} " +
          " ".join(f"{r[:5]:>7s}" for r in RUBRICS))
    for lbl, arms in data.items():
        for arm in ARMS:
            t = arms[arm]
            if not t:
                print(f"  {lbl:16s} {arm:14s}  missing")
                continue
            print(f"  {lbl:16s} {arm:14s} {len(t):4d} "
                  f"{np.mean([v['overall'] for v in t.values()]):8.3f} " +
                  " ".join(f"{np.mean([v[r] for v in t.values()]):7.3f}" for r in RUBRICS))
        print()

    out = {}
    for hi, lo, what in EDGES:
        print("=" * 90)
        print(f"EDGE: {hi}  minus  {lo}   isolates {what}")
        print("=" * 90)
        for lbl, arms in data.items():
            if not (arms[hi] and arms[lo]):
                continue
            shared = sorted(set(arms[hi]) & set(arms[lo]))
            row = {"n": len(shared)}
            cells = []
            for metric in ["overall"] + RUBRICS:
                a = np.array([arms[hi][c][metric] for c in shared])
                b = np.array([arms[lo][c][metric] for c in shared])
                d, l, h, p = paired(a, b)
                row[metric] = [round(d, 3), round(l, 3), round(h, 3), round(p, 4)]
                star = "*" if l > 0 or h < 0 else " "
                cells.append(f"{d:+6.3f}{star}")
            print(f"  {lbl:17s} n={len(shared):3d}  " +
                  "  ".join(f"{m[:5]}={c}" for m, c in
                            zip(["overall"] + RUBRICS, cells)))
            out.setdefault(f"{hi} - {lo}", {})[lbl] = row
        print("  * marks a 95% interval excluding zero\n")

    print("=" * 90)
    print("WHY THE 'NEITHER' ARM IS NOT A FLOOR")
    print("=" * 90)
    for lbl, arms in data.items():
        if not arms["neither"]:
            continue
        m = np.mean([v["morphological_accuracy"] for v in arms["neither"].values()])
        f = np.mean([v["morphological_accuracy"] for v in arms["image + grade"].values()])
        print(f"  {lbl:17s} morphology with no inputs {m:.2f}, with both {f:.2f}"
              f"{'   HIGHER WITH NOTHING' if m > f else ''}")
    print("\n  A model handed neither input declines to describe the embryo, and the")
    print("  rubric scores an honest refusal as a middling success, so this arm")
    print("  measures epistemic appropriateness rather than information.")

    (RUNS / "ablation_edges.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RUNS / 'ablation_edges.json'}")


if __name__ == "__main__":
    main()
