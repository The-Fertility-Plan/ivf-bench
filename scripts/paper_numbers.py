"""Every derived number the paper states, recomputed from the sighted-judge scores.

paper_analysis.py covers the bootstrap CIs, rubric structure and verbosity. This
covers the rest: per-rubric averages, the outcome-metric spans, the base-rate
comparison, the post-training table, stated-probability calibration, and the
length-adjusted margin against Opus.

Usage: python scripts/paper_numbers.py
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
SPLITS = {"test": "cases", "held_out": "held_out_cases"}
OURS, OPUS, BASE = ("ivf-bench-qwen9b-vlm-orpo",
                    "global.anthropic.claude-opus-4-6-v1",
                    "qwen/qwen3.5-9b")
RNG = np.random.default_rng(42)


def split_cases(split: str) -> dict[str, dict]:
    d = ROOT / "data" / SPLITS[split]
    return {f.stem: json.loads(f.read_text()) for f in sorted(d.glob("*.json"))}


def load(split: str) -> dict[str, dict[str, dict]]:
    """model -> case_id -> record, excluding ablation arms and judge failures."""
    cases, out = split_cases(split), {}
    for run in sorted(RUNS.iterdir()):
        meta = run / "run_meta.json"
        if not run.is_dir() or run.name.startswith("_") or not meta.exists():
            continue
        m = json.loads(meta.read_text())
        if not m.get("include_image", True) or not m.get("include_grade", True):
            continue
        per_case = {}
        for f in (run / SCORES).glob("*.json"):
            cid = f.stem
            if cid not in cases:
                continue
            d = json.loads(f.read_text())
            sc = {r["rubric"]: r["score"] for r in d["rubrics"] if r["score"] > 0}
            if len(sc) != len(RUBRICS):
                continue
            rf = run / "responses" / f"{cid}.json"
            resp = json.loads(rf.read_text()) if rf.exists() else {}
            per_case[cid] = {
                **sc,
                "overall": float(np.mean([sc[r] for r in RUBRICS])),
                "length": len(resp.get("response_text", "")),
                "prob": d.get("extracted_probability"),
                "outcome": int(bool(cases[cid]["real_outcome"]["clinical_pregnancy"])),
            }
        if per_case:
            out[m["model"]] = per_case
    return out


def brier(tab: dict[str, dict]) -> tuple[float, int]:
    p = [(v["prob"], v["outcome"]) for v in tab.values() if v["prob"] is not None]
    return float(np.mean([(a - b) ** 2 for a, b in p])), len(p)


def auroc(tab: dict[str, dict]) -> float:
    p = [(v["prob"], v["outcome"]) for v in tab.values() if v["prob"] is not None]
    pos = [a for a, b in p if b == 1]
    neg = [a for a, b in p if b == 0]
    if not pos or not neg:
        return float("nan")
    wins = sum((x > y) + 0.5 * (x == y) for x in pos for y in neg)
    return wins / (len(pos) * len(neg))


def main() -> None:
    data = {s: load(s) for s in SPLITS}

    print("=" * 78)
    print("PER-RUBRIC AVERAGES ACROSS SYSTEMS")
    print("=" * 78)
    for split, tabs in data.items():
        print(f"\n{split}:")
        for r in RUBRICS:
            per = {m: np.mean([v[r] for v in t.values()]) for m, t in tabs.items()}
            worst = min(per, key=per.get)
            print(f"  {r:26s} mean {np.mean(list(per.values())):.3f}  "
                  f"max {max(per.values()):.3f}  min {min(per.values()):.3f} ({worst})")
        weakest = sum(min(RUBRICS, key=lambda r: np.mean([v[r] for v in t.values()]))
                      == "morphological_accuracy" for t in tabs.values())
        print(f"  morphology is the single weakest rubric for {weakest}/{len(tabs)} systems")

    print("\n" + "=" * 78)
    print("OUTCOME METRICS")
    print("=" * 78)
    all_auroc, beats_50, total = [], 0, 0
    for split, tabs in data.items():
        print(f"\n{split}:")
        cases = split_cases(split)
        rate = np.mean([bool(c["real_outcome"]["clinical_pregnancy"]) for c in cases.values()])
        const = float(np.mean([(rate - bool(c["real_outcome"]["clinical_pregnancy"])) ** 2
                               for c in cases.values()]))
        print(f"  cohort pregnancy rate {rate:.4f}; constant-at-rate Brier {const:.4f}")
        for m, t in sorted(tabs.items(), key=lambda kv: brier(kv[1])[0]):
            b, n = brier(t)
            a = auroc(t)
            all_auroc.append(a)
            total += 1
            beats_50 += b < 0.25
            mp = np.mean([v["prob"] for v in t.values() if v["prob"] is not None])
            print(f"  {m:38s} Brier {b:.4f} (n={n})  AUROC {a:.4f}  mean p {mp:.3f}"
                  f"{'  beats base rate' if b < const else ''}")
    print(f"\n  AUROC spans {min(all_auroc):.3f} to {max(all_auroc):.3f} "
          f"across {total} model-split combinations")
    print(f"  {beats_50} of {total} beat the constant-0.50 Brier of 0.25, "
          f"{total - beats_50} do not")

    print("\n" + "=" * 78)
    print("POST-TRAINING, HELD-OUT SPLIT")
    print("=" * 78)
    h = data["held_out"]
    for r in ["overall"] + RUBRICS:
        b = np.mean([v[r] for v in h[BASE].values()])
        o = np.mean([v[r] for v in h[OURS].values()])
        print(f"  {r:26s} {b:.3f} -> {o:.3f}   {100*(o-b)/b:+.1f}%")
    bb, _ = brier(h[BASE])
    ob, _ = brier(h[OURS])
    print(f"  {'Brier':26s} {bb:.3f} -> {ob:.3f}   {ob-bb:+.3f}")
    print(f"  {'AUROC':26s} {auroc(h[BASE]):.3f} -> {auroc(h[OURS]):.3f}   "
          f"{auroc(h[OURS])-auroc(h[BASE]):+.3f}")
    print(f"  test-split overall {np.mean([v['overall'] for v in data['test'][BASE].values()]):.3f}"
          f" -> {np.mean([v['overall'] for v in data['test'][OURS].values()]):.3f}")

    print("\n" + "=" * 78)
    print("LENGTH CONTROL, OURS VS OPUS ON HELD OUT")
    print("=" * 78)
    shared = sorted(set(h[OURS]) & set(h[OPUS]))
    la = np.array([h[OURS][c]["length"] for c in shared], float)
    lb = np.array([h[OPUS][c]["length"] for c in shared], float)
    sa = np.array([h[OURS][c]["overall"] for c in shared])
    sb = np.array([h[OPUS][c]["overall"] for c in shared])
    print(f"  mean response length: ours {la.mean():,.0f} chars, Opus {lb.mean():,.0f}")
    print(f"  raw margin {sa.mean()-sb.mean():+.3f}")
    grand = np.concatenate([la, lb]).mean()
    adj = []
    for L, S in ((la, sa), (lb, sb)):
        slope, icpt = np.polyfit(L, S, 1)
        adj.append(icpt + slope * grand)
    print(f"  at a common length of {grand:,.0f} chars: ours {adj[0]:.3f}, "
          f"Opus {adj[1]:.3f}, margin {adj[0]-adj[1]:+.3f}")

    print("\n" + "=" * 78)
    print("JUDGEMENT COUNTS")
    print("=" * 78)
    for split, tabs in data.items():
        print(f"  {split}: {sum(len(t) for t in tabs.values())} judgements, "
              f"{len(tabs)} systems")
    n_resp = sum(len(glob.glob(str(r / "responses" / "*.json")))
                 for r in RUNS.iterdir()
                 if r.is_dir() and not r.name.startswith("_")
                 and (r / "run_meta.json").exists()
                 and all(json.loads((r / "run_meta.json").read_text()).get(k, True)
                         for k in ("include_image", "include_grade")))
    n_abl = sum(len(glob.glob(str(r / "responses" / "*.json")))
                for r in RUNS.iterdir()
                if r.is_dir() and not r.name.startswith("_")
                and (r / "run_meta.json").exists()
                and not all(json.loads((r / "run_meta.json").read_text()).get(k, True)
                            for k in ("include_image", "include_grade")))
    print(f"  leaderboard responses {n_resp}, ablation-arm responses {n_abl}")


if __name__ == "__main__":
    main()
