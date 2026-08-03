"""Two checks that decide headline claims in the paper.

1. A tabular baseline. The paper argues the outcome ceiling reflects missing data
   rather than model capacity. That argument only establishes that the *generated*
   fields carry no signal; it says nothing about the *measured* ones. If logistic
   regression on age, AMH, endometrial thickness, oocyte counts and the Gardner
   grade beats the vision-language models, the claim is false and the correct
   conclusion is that the models underperform a trivial baseline.

2. A length control. The judge rewards verbosity (pooled Spearman 0.61), and our
   post-trained model was trained toward a verbose teacher while Opus 4.6 is the
   one system whose score is uncorrelated with length. If the ORPO-over-Opus
   margin disappears once length is controlled, the headline comparison is a
   length artifact.

Usage: python scripts/tabular_baseline.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RUBRICS = ["morphological_accuracy", "clinical_integration", "reasoning_coherence",
           "guideline_alignment", "actionability"]
RNG = np.random.default_rng(42)


def load_cases(split_dir: str) -> list[dict]:
    return [json.loads(f.read_text()) for f in sorted((ROOT / "data" / split_dir).glob("*.json"))]


def featurize(cases: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Only measured fields. Missing values imputed to the column median."""
    rows, y = [], []
    for c in cases:
        r = c["real_clinical"]; g = c["gardner"]
        rows.append([
            r.get("age"), r.get("amh"), r.get("endometrial_thickness"),
            r.get("cocs_retrieved"), r.get("mii_oocytes"), r.get("transfer_day"),
            g.get("exp"), g.get("icm"), g.get("te"),
        ])
        y.append(int(c["real_outcome"]["clinical_pregnancy"]))
    X = np.array([[np.nan if v is None else float(v) for v in row] for row in rows])
    med = np.nanmedian(X, axis=0)
    idx = np.where(np.isnan(X))
    X[idx] = np.take(med, idx[1])
    return X, np.array(y)


def bootstrap_auroc(y: np.ndarray, p: np.ndarray, n: int = 10_000) -> tuple[float, float, float]:
    obs = roc_auc_score(y, p)
    vals = []
    for _ in range(n):
        i = RNG.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        vals.append(roc_auc_score(y[i], p[i]))
    return obs, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def check_tabular() -> None:
    print("=" * 74)
    print("1. TABULAR BASELINE on measured fields only (9 features)")
    print("=" * 74)
    tr = load_cases("cases")
    te = load_cases("held_out_cases")
    Xtr, ytr = featurize(tr)
    Xte, yte = featurize(te)

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))

    # Honest in-cohort estimate: 5-fold CV on the 550 test-split cases.
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    oof = np.zeros(len(ytr))
    for a, b in cv.split(Xtr, ytr):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        m.fit(Xtr[a], ytr[a])
        oof[b] = m.predict_proba(Xtr[b])[:, 1]
    auc, lo, hi = bootstrap_auroc(ytr, oof)
    print(f"  550-case split, 5-fold cross-validated:")
    print(f"    AUROC {auc:.3f}  95% CI [{lo:.3f}, {hi:.3f}]   Brier {brier_score_loss(ytr, oof):.4f}")

    # Fit on 550, evaluate on the 103 held out.
    model.fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]
    auc2, lo2, hi2 = bootstrap_auroc(yte, p)
    print(f"  fit on 550, evaluated on the 103 held out:")
    print(f"    AUROC {auc2:.3f}  95% CI [{lo2:.3f}, {hi2:.3f}]   Brier {brier_score_loss(yte, p):.4f}")

    base = ytr.mean()
    print(f"\n  reference: cohort rate {base:.3f}; constant predictor Brier "
          f"{np.mean((base - yte) ** 2):.4f}")
    print(f"  reference: best vision-language AUROC on held out = 0.562 (GPT-5.4)")


def load_scores(run: str, ids: set[str]) -> dict[str, dict]:
    out = {}
    for f in (ROOT / "data/runs" / run / "scores").glob("*.json"):
        if f.stem not in ids:
            continue
        d = json.loads(f.read_text())
        sc = {r["rubric"]: r["score"] for r in d["rubrics"] if r["score"] > 0}
        if len(sc) != 5:
            continue
        rf = ROOT / "data/runs" / run / "responses" / f"{f.stem}.json"
        out[f.stem] = {
            "overall": float(np.mean([sc[r] for r in RUBRICS])),
            "len": len(json.loads(rf.read_text()).get("response_text", "")) if rf.exists() else 0,
        }
    return out


def check_length() -> None:
    print()
    print("=" * 74)
    print("2. LENGTH CONTROL on the headline comparison (held-out split)")
    print("=" * 74)
    ids = {f.stem for f in (ROOT / "data/held_out_cases").glob("*.json")}
    ours = load_scores("ivf-bench-qwen9b-vlm-orpo", ids)
    opus = load_scores("global_anthropic_claude-opus-4-6-v1", ids)
    shared = sorted(set(ours) & set(opus))

    a = np.array([ours[c]["overall"] for c in shared])
    b = np.array([opus[c]["overall"] for c in shared])
    la = np.array([ours[c]["len"] for c in shared], dtype=float)
    lb = np.array([opus[c]["len"] for c in shared], dtype=float)

    print(f"  n = {len(shared)}")
    print(f"  mean response length: ours {la.mean():,.0f} chars | Opus {lb.mean():,.0f} chars")
    print(f"  raw margin: {a.mean() - b.mean():+.3f}")

    # Regress score on length within each system, compare length-adjusted means at
    # a common length (the pooled mean).
    common = np.concatenate([la, lb]).mean()
    adj = {}
    for name, sc, ln in (("ours", a, la), ("opus", b, lb)):
        sl, ic, r, p, _ = stats.linregress(ln, sc)
        adj[name] = ic + sl * common
        print(f"  {name:5s}: score-on-length slope {sl:+.2e} per char (r={r:+.3f}, p={p:.3f}), "
              f"adjusted mean at {common:,.0f} chars = {adj[name]:.3f}")
    print(f"  length-adjusted margin: {adj['ours'] - adj['opus']:+.3f}  "
          f"(raw was {a.mean() - b.mean():+.3f})")


if __name__ == "__main__":
    check_tabular()
    check_length()
