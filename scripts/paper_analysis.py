"""Zero-cost analyses for the IVF-Bench paper, computed from existing run artifacts.

Produces, for both the test and held-out splits:
  1. Bootstrap 95% CIs on each model's overall score (are adjacent ranks separable?)
  2. Rubric inter-correlation + first-component variance (do 5 rubrics measure 5 things?)
  3. Verbosity bias (does response length predict the judge's score?)
  4. Patient-level robustness (rankings with one embryo per patient)

Writes data/runs/paper_analysis.json and prints a summary.
"""
from __future__ import annotations

import json
from collections import defaultdict
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
SPLIT_DIRS = {"test": "cases", "held_out": "held_out_cases"}
RNG = np.random.default_rng(42)
N_BOOT = 10_000


def load_split_cases(split: str) -> dict[str, dict]:
    """case_id -> case JSON for one split."""
    d = ROOT / "data" / SPLIT_DIRS[split]
    return {f.stem: json.loads(f.read_text()) for f in sorted(d.glob("*.json"))}


def load_records(split: str, cases: dict[str, dict]) -> dict[str, dict[str, dict]]:
    """model -> case_id -> {rubric scores, overall, response length}.

    Applies the same score > 0 filter as leaderboard.py, so judge-failure files
    are dropped rather than counted as zeros.
    """
    out: dict[str, dict[str, dict]] = {}
    for run_dir in sorted(RUNS.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("_"):
            continue
        meta = run_dir / "run_meta.json"
        if not meta.exists():
            continue
        model = json.loads(meta.read_text())["model"]
        per_case: dict[str, dict] = {}
        for sf in (run_dir / "scores").glob("*.json"):
            cid = sf.stem
            if cid not in cases:
                continue
            data = json.loads(sf.read_text())
            scores = {r["rubric"]: r["score"] for r in data["rubrics"] if r["score"] > 0}
            if len(scores) != len(RUBRICS):
                continue  # judge failure or partial parse
            rf = run_dir / "responses" / f"{cid}.json"
            length = len(json.loads(rf.read_text()).get("response_text", "")) if rf.exists() else 0
            per_case[cid] = {
                **scores,
                "overall": float(np.mean([scores[r] for r in RUBRICS])),
                "length": length,
            }
        if per_case:
            out[model] = per_case
    return out


def bootstrap_ci(values: np.ndarray) -> tuple[float, float, float]:
    idx = RNG.integers(0, len(values), size=(N_BOOT, len(values)))
    means = values[idx].mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_diff_ci(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float, float]:
    """Paired bootstrap on a - b over shared cases. Returns mean diff, lo, hi, p."""
    d = a - b
    idx = RNG.integers(0, len(d), size=(N_BOOT, len(d)))
    means = d[idx].mean(axis=1)
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), float(p)


def analyse(split: str) -> dict:
    cases = load_split_cases(split)
    recs = load_records(split, cases)
    res: dict = {"split": split, "n_cases_in_split": len(cases), "models": {}}

    # --- 1. bootstrap CIs -------------------------------------------------
    for model, per_case in recs.items():
        vals = np.array([v["overall"] for v in per_case.values()])
        mean, lo, hi = bootstrap_ci(vals)
        res["models"][model] = {
            "n_scored": len(vals),
            "overall": round(mean, 4),
            "ci95": [round(lo, 4), round(hi, 4)],
        }

    # --- 2. adjacent-rank separability (paired bootstrap) -----------------
    order = sorted(res["models"], key=lambda m: -res["models"][m]["overall"])
    pairs = []
    for hi_m, lo_m in zip(order, order[1:]):
        shared = sorted(set(recs[hi_m]) & set(recs[lo_m]))
        a = np.array([recs[hi_m][c]["overall"] for c in shared])
        b = np.array([recs[lo_m][c]["overall"] for c in shared])
        diff, lo, hi, p = paired_diff_ci(a, b)
        pairs.append({
            "higher": hi_m, "lower": lo_m, "n_shared": len(shared),
            "diff": round(diff, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "p": round(p, 4), "separable": bool(lo > 0),
        })
    res["adjacent_rank_tests"] = pairs

    # --- 3. rubric structure ---------------------------------------------
    mat = np.array([[v[r] for r in RUBRICS] for pc in recs.values() for v in pc.values()])
    corr = np.corrcoef(mat, rowvar=False)
    centred = mat - mat.mean(axis=0)
    eig = np.linalg.svd(centred, compute_uv=False) ** 2
    res["rubric_correlation"] = {
        "matrix": [[round(x, 3) for x in row] for row in corr],
        "labels": RUBRICS,
        "mean_offdiag_r": round(float(corr[np.triu_indices(5, 1)].mean()), 3),
        "pc1_variance_explained": round(float(eig[0] / eig.sum()), 3),
        "n_observations": int(mat.shape[0]),
    }

    # --- 4. verbosity bias ------------------------------------------------
    verb = {}
    for model, per_case in recs.items():
        L = np.array([v["length"] for v in per_case.values()], dtype=float)
        S = np.array([v["overall"] for v in per_case.values()])
        if L.std() == 0:
            continue
        rho, p = stats.spearmanr(L, S)
        verb[model] = {"spearman_rho": round(float(rho), 3), "p": round(float(p), 5),
                       "median_chars": int(np.median(L))}
    allL = np.array([v["length"] for pc in recs.values() for v in pc.values()], dtype=float)
    allS = np.array([v["overall"] for pc in recs.values() for v in pc.values()])
    rho, p = stats.spearmanr(allL, allS)
    res["verbosity_bias"] = {"per_model": verb,
                             "pooled": {"spearman_rho": round(float(rho), 3), "p": float(p)}}

    # --- 5. one-embryo-per-patient robustness -----------------------------
    by_patient: dict[str, list[str]] = defaultdict(list)
    for cid, c in cases.items():
        by_patient[c["image_path"].split("_")[0]].append(cid)
    keep = {sorted(v)[0] for v in by_patient.values()}
    dropped = set(cases) - keep
    rob = {}
    for model, per_case in recs.items():
        vals = np.array([v["overall"] for c, v in per_case.items() if c in keep])
        rob[model] = {"n": len(vals), "overall": round(float(vals.mean()), 4)}
    new_order = sorted(rob, key=lambda m: -rob[m]["overall"])
    res["patient_dedup_robustness"] = {
        "n_dropped_cases": len(dropped),
        "dropped_case_ids": sorted(dropped),
        "scores": rob,
        "rank_order": new_order,
        "rank_changes": [
            {"model": m, "orig_rank": order.index(m) + 1, "dedup_rank": new_order.index(m) + 1}
            for m in order if order.index(m) != new_order.index(m)
        ],
    }
    return res


def main() -> None:
    out = {s: analyse(s) for s in SPLIT_DIRS}
    (RUNS / "paper_analysis.json").write_text(json.dumps(out, indent=2))

    for split, r in out.items():
        print(f"\n{'='*78}\n{split.upper()}  ({r['n_cases_in_split']} cases in split)\n{'='*78}")
        print(f"{'model':38s} {'n':>4s} {'overall':>8s}  95% CI")
        for m in sorted(r["models"], key=lambda m: -r["models"][m]["overall"]):
            v = r["models"][m]
            print(f"{m:38s} {v['n_scored']:4d} {v['overall']:8.3f}  [{v['ci95'][0]:.3f}, {v['ci95'][1]:.3f}]")

        print("\nAdjacent ranks — paired bootstrap on shared cases:")
        for p in r["adjacent_rank_tests"]:
            mark = "SEPARABLE" if p["separable"] else "NOT separable"
            print(f"  {p['higher'][:26]:26s} > {p['lower'][:26]:26s} "
                  f"d={p['diff']:+.3f} [{p['ci95'][0]:+.3f},{p['ci95'][1]:+.3f}] p={p['p']:.3f}  {mark}")

        rc = r["rubric_correlation"]
        print(f"\nRubric structure: mean off-diagonal r={rc['mean_offdiag_r']}, "
              f"PC1 explains {rc['pc1_variance_explained']*100:.1f}% of variance "
              f"(n={rc['n_observations']} judgements)")

        vb = r["verbosity_bias"]
        print(f"Verbosity bias: pooled Spearman rho={vb['pooled']['spearman_rho']} "
              f"(p={vb['pooled']['p']:.2e})")
        for m, v in sorted(vb["per_model"].items(), key=lambda kv: -abs(kv[1]["spearman_rho"]))[:3]:
            print(f"    strongest: {m:34s} rho={v['spearman_rho']:+.3f} median {v['median_chars']} chars")

        pd_ = r["patient_dedup_robustness"]
        print(f"\nOne embryo per patient: dropped {pd_['n_dropped_cases']} cases; "
              f"rank changes: {pd_['rank_changes'] or 'none'}")


if __name__ == "__main__":
    main()
