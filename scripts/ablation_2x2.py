"""The full two-by-two: does a model read the embryo, or did the grade answer it?

Four conditions per system on the same cases:
    image + grade   the benchmark as run
    grade only      image withheld
    image only      Gardner grade withheld
    neither         both withheld

The single-factor ablation could not separate "cannot read the image" from "the
grade had already answered the question". These four can: if image-only stays
near image+grade, the system reads the embryo; if it falls to the neither floor,
it does not.

Usage: python scripts/ablation_2x2.py
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "data" / "runs"
RUBRICS = ["morphological_accuracy", "clinical_integration", "reasoning_coherence",
           "guideline_alignment", "actionability"]
IDS = {Path(f).stem for f in glob.glob(str(ROOT / "data/cases/*.json"))}
RNG = np.random.default_rng(42)
SCORES = "scores_sighted"   # judge eyesight held constant across all four arms

SYSTEMS = {
    "GPT-5.4":          "gpt-5_4-2026-03-05",
    "Opus 4.6":         "global_anthropic_claude-opus-4-6-v1",
    "Qwen 397B":        "qwen__qwen3_5-397b-a17b",
    "Gemini 2.5 Flash": "gemini-2_5-flash",
}
# the no-image arms were collected under the openrouter slug for gemini
ALIAS = {"gemini-2_5-flash": "google__gemini-2_5-flash"}
CONDITIONS = [("image + grade", ""), ("grade only", "-noimg"),
              ("image only", "-nograde"), ("neither", "-neither")]


def load(run: str) -> dict[str, dict[str, float]]:
    out = {}
    for f in glob.glob(str(RUNS / run / SCORES / "*.json")):
        cid = Path(f).stem
        if cid not in IDS:
            continue
        d = json.loads(Path(f).read_text())
        sc = {r["rubric"]: r["score"] for r in d["rubrics"] if r["score"] > 0}
        if len(sc) == len(RUBRICS):
            sc["overall"] = float(np.mean([sc[r] for r in RUBRICS]))
            out[cid] = sc
    return out


def paired(a: np.ndarray, b: np.ndarray):
    d = a - b
    idx = RNG.integers(0, len(d), size=(10_000, len(d)))
    m = d[idx].mean(axis=1)
    p = 2 * min((m <= 0).mean(), (m >= 0).mean())
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), float(p)


def main() -> None:
    results = {}
    print(f"{'system':18s} {'condition':14s} {'n':>4s} {'overall':>8s} {'morphology':>11s}")
    for label, base in SYSTEMS.items():
        results[label] = {}
        for cond, suffix in CONDITIONS:
            run = (ALIAS.get(base, base) + suffix) if suffix else base
            if not (RUNS / run).exists():
                run = base + suffix
            tab = load(run)
            if not tab:
                print(f"  {label:16s} {cond:14s}  (missing)")
                continue
            results[label][cond] = tab
            ov = np.mean([v["overall"] for v in tab.values()])
            mo = np.mean([v["morphological_accuracy"] for v in tab.values()])
            print(f"  {label:16s} {cond:14s} {len(tab):4d} {ov:8.3f} {mo:11.3f}")
        print()

    print("=" * 78)
    print("THE QUESTION: with the grade withheld, does the image carry the assessment?")
    print("=" * 78)
    out = {}
    for label, conds in results.items():
        if not {"image + grade", "image only", "neither"} <= set(conds):
            continue
        shared = sorted(set(conds["image + grade"]) & set(conds["image only"]) & set(conds["neither"]))
        full = np.array([conds["image + grade"][c]["morphological_accuracy"] for c in shared])
        img = np.array([conds["image only"][c]["morphological_accuracy"] for c in shared])
        nei = np.array([conds["neither"][c]["morphological_accuracy"] for c in shared])
        d1, lo1, hi1, p1 = paired(full, img)   # cost of losing the grade, image kept
        d2, lo2, hi2, p2 = paired(img, nei)    # value of the image once the grade is gone
        # A ratio here is meaningless: the 'neither' arm is not a floor, because a
        # blind judge rewards a model that correctly declines to describe an image
        # it never received. Report the differences themselves.
        recovered = None
        print(f"\n  {label}  (n={len(shared)}, morphology grounding)")
        print(f"    image+grade {full.mean():.2f}   image only {img.mean():.2f}   neither {nei.mean():.2f}")
        print(f"    losing the grade, image kept : {d1:+.3f} [{lo1:+.3f},{hi1:+.3f}] p={p1:.4f}")
        print(f"    the image alone is worth     : {d2:+.3f} [{lo2:+.3f},{hi2:+.3f}] p={p2:.4f}")
        print(f"    NOTE: 'neither' is not a floor; a model with no inputs still scores "
              f"{nei.mean():.2f} by correctly declining to describe what it was not given.")
        out[label] = {"n": len(shared),
                      "full": round(float(full.mean()), 3),
                      "image_only": round(float(img.mean()), 3),
                      "neither": round(float(nei.mean()), 3),
                      "grade_loss": [round(d1, 3), round(lo1, 3), round(hi1, 3), round(p1, 4)],
                      "image_value": [round(d2, 3), round(lo2, 3), round(hi2, 3), round(p2, 4)],
                      "note": "neither is not a floor"}
    (RUNS / "ablation_2x2.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RUNS / 'ablation_2x2.json'}")


if __name__ == "__main__":
    main()
