"""Compare the GPT-5.4 leaderboard against the same responses graded by Sonnet 4.6.

The paper's headline comparison is made by a judge that also authored 92% of the
responses our model was trained to imitate. This asks whether the ordering holds
under a judge from a different model family that wrote none of them.

Usage: python scripts/cross_judge.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "data" / "runs"
RUBRICS = ["morphological_accuracy", "clinical_integration", "reasoning_coherence",
           "guideline_alignment", "actionability"]
RNG = np.random.default_rng(42)
# both judges see the embryo, so the comparison isolates the judge, not its eyesight
JUDGE_DIR = "scores_sonnet_sighted"
IDS = {Path(f).stem for f in glob.glob(str(ROOT / "data/held_out_cases/*.json"))}

NAMES = {
    "gpt-5_4-2026-03-05": "GPT-5.4",
    "ivf-bench-qwen9b-vlm-orpo": "Ours (9B-ORPO)",
    "global_anthropic_claude-opus-4-6-v1": "Opus 4.6",
    "gemini-2_5-flash": "Gemini 2.5 Flash",
    "qwen__qwen3_5-397b-a17b": "Qwen 397B",
    "moonshotai__kimi-k2_5": "Kimi K2.5",
    "global_anthropic_claude-sonnet-4-6": "Sonnet 4.6",
    "qwen__qwen3_5-9b": "Qwen 9B (base)",
}


def load(run: str, subdir: str) -> dict[str, float]:
    out = {}
    for f in glob.glob(str(RUNS / run / subdir / "*.json")):
        cid = Path(f).stem
        if cid not in IDS:
            continue
        d = json.loads(Path(f).read_text())
        sc = {r["rubric"]: r["score"] for r in d["rubrics"] if r["score"] > 0}
        if len(sc) == len(RUBRICS):
            out[cid] = float(np.mean([sc[r] for r in RUBRICS]))
    return out


def paired_ci(a: np.ndarray, b: np.ndarray, n: int = 10_000):
    d = a - b
    idx = RNG.integers(0, len(d), size=(n, len(d)))
    m = d[idx].mean(axis=1)
    p = 2 * min((m <= 0).mean(), (m >= 0).mean())
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), float(p)


def main() -> None:
    gpt, son = {}, {}
    for run in NAMES:
        g, s = load(run, "scores_sighted"), load(run, JUDGE_DIR)
        if g and s:
            gpt[run], son[run] = g, s

    print(f"{'model':18s} {'GPT-5.4 judge':>14s} {'Sonnet judge':>13s} {'n':>5s}")
    order_g, order_s = {}, {}
    for run in NAMES:
        shared = sorted(set(gpt[run]) & set(son[run]))
        g = np.mean([gpt[run][c] for c in shared])
        s = np.mean([son[run][c] for c in shared])
        order_g[run], order_s[run] = g, s
        print(f"  {NAMES[run]:16s} {g:14.3f} {s:13.3f} {len(shared):5d}")

    rg = sorted(order_g, key=lambda k: -order_g[k])
    rs = sorted(order_s, key=lambda k: -order_s[k])
    print(f"\n  GPT-5.4 ordering: {' > '.join(NAMES[k] for k in rg)}")
    print(f"  Sonnet  ordering: {' > '.join(NAMES[k] for k in rs)}")

    rho, p = stats.spearmanr([rg.index(k) for k in NAMES], [rs.index(k) for k in NAMES])
    print(f"\n  rank correlation between the two judges: Spearman rho = {rho:.3f} (p = {p:.4f})")

    OURS, OPUS = "ivf-bench-qwen9b-vlm-orpo", "global_anthropic_claude-opus-4-6-v1"
    # paper_analysis.py is the canonical source for the GPT-5.4-judged interval;
    # recomputing it here with a different RNG history produced a second, slightly
    # different interval for the same quantity, which then reached the paper.
    print("    (the GPT-5.4 interval quoted in the paper comes from "
          "paper_analysis.json, not from this script's resample)")
    print("\n  the paper's headline comparison, ours against Opus 4.6:")
    for lbl, tab in (("GPT-5.4 judge", gpt), ("Sonnet 4.6 judge", son)):
        shared = sorted(set(tab[OURS]) & set(tab[OPUS]))
        a = np.array([tab[OURS][c] for c in shared])
        b = np.array([tab[OPUS][c] for c in shared])
        d, lo, hi, pv = paired_ci(a, b)
        verdict = "ours ahead" if lo > 0 else ("Opus ahead" if hi < 0 else "not separable")
        print(f"    {lbl:18s} {d:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  p={pv:.4f}   {verdict}")

    out = {"per_model": {NAMES[k]: {"gpt": round(order_g[k], 4), "sonnet": round(order_s[k], 4)}
                         for k in NAMES},
           "rank_spearman": round(float(rho), 4)}
    (RUNS / "cross_judge.json").write_text(json.dumps(out, indent=2))
    print(f"\n  wrote {RUNS / 'cross_judge.json'}")


if __name__ == "__main__":
    main()
