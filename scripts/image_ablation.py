"""Compare each model with and without the embryo image on the same cases.

The benchmark hands the model a Gardner grade in text as well as the image, so a
model could score well on morphology grounding without ever looking at the
picture. This runs the paired comparison that settles it: same cases, same judge,
image present or absent.

Run the ablation arms first:
    ivf-bench run <model> --split test --limit N --no-image --run-suffix -noimg
    ivf-bench score <model>-noimg --judge ... --split test

Then:  python scripts/image_ablation.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "data" / "runs"
RUBRICS = [
    "morphological_accuracy",
    "clinical_integration",
    "reasoning_coherence",
    "guideline_alignment",
    "actionability",
]
RNG = np.random.default_rng(42)
N_BOOT = 10_000


def load(run_name: str, case_ids: set[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for sf in (RUNS / run_name / "scores").glob("*.json"):
        if sf.stem not in case_ids:
            continue
        data = json.loads(sf.read_text())
        scores = {r["rubric"]: r["score"] for r in data["rubrics"] if r["score"] > 0}
        if len(scores) != len(RUBRICS):
            continue
        scores["overall"] = float(np.mean([scores[r] for r in RUBRICS]))
        out[sf.stem] = scores
    return out


def paired(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float]:
    d = a - b
    idx = RNG.integers(0, len(d), size=(N_BOOT, len(d)))
    means = d[idx].mean(axis=1)
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), float(p)


def main() -> None:
    case_ids = {f.stem for f in (ROOT / "data/cases").glob("*.json")}
    pairs = [
        ("gpt-5_4-2026-03-05", "gpt-5_4-2026-03-05-noimg", "GPT-5.4"),
        ("global_anthropic_claude-opus-4-6-v1",
         "global_anthropic_claude-opus-4-6-v1-noimg", "Opus 4.6"),
        ("qwen__qwen3_5-397b-a17b", "qwen__qwen3_5-397b-a17b-noimg", "Qwen 397B"),
        ("gemini-2_5-flash", "google__gemini-2_5-flash-noimg", "Gemini 2.5 Flash"),
    ]

    results = {}
    print(f"{'model':18s} {'n':>4s}  {'with image':>11s} {'no image':>9s} "
          f"{'delta':>7s}  {'95% CI':>18s} {'p':>7s}")
    for with_run, without_run, label in pairs:
        if not (RUNS / without_run / "scores").exists():
            print(f"{label:18s}  (ablation arm not scored yet)")
            continue
        w, wo = load(with_run, case_ids), load(without_run, case_ids)
        shared = sorted(set(w) & set(wo))
        if not shared:
            print(f"{label:18s}  (no overlapping scored cases)")
            continue
        a = np.array([w[c]["overall"] for c in shared])
        b = np.array([wo[c]["overall"] for c in shared])
        d, lo, hi, p = paired(a, b)
        print(f"{label:18s} {len(shared):4d}  {a.mean():11.3f} {b.mean():9.3f} "
              f"{d:+7.3f}  [{lo:+.3f}, {hi:+.3f}] {p:7.4f}")

        per_rubric = {}
        for r in RUBRICS:
            ar = np.array([w[c][r] for c in shared])
            br = np.array([wo[c][r] for c in shared])
            dr, lor, hir, pr = paired(ar, br)
            per_rubric[r] = {"with": round(float(ar.mean()), 3),
                             "without": round(float(br.mean()), 3),
                             "delta": round(dr, 3), "ci95": [round(lor, 3), round(hir, 3)],
                             "p": round(pr, 4)}
        results[label] = {"n": len(shared), "overall_with": round(float(a.mean()), 3),
                          "overall_without": round(float(b.mean()), 3),
                          "delta": round(d, 3), "ci95": [round(lo, 3), round(hi, 3)],
                          "p": round(p, 4), "per_rubric": per_rubric}

    if results:
        print(f"\n{'model':18s} morphology grounding: with -> without (delta, p)")
        for label, r in results.items():
            m = r["per_rubric"]["morphological_accuracy"]
            print(f"  {label:16s} {m['with']:.3f} -> {m['without']:.3f} "
                  f"({m['delta']:+.3f}, p={m['p']:.4f})")
        (RUNS / "image_ablation.json").write_text(json.dumps(results, indent=2))
        print(f"\nwrote {RUNS / 'image_ablation.json'}")


if __name__ == "__main__":
    main()
