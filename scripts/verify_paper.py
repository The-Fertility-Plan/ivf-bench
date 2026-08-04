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
import yaml

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


TEX_FLAT = re.sub(r"\s+", " ", TEX)


def in_tex(s):
    """Whitespace-insensitive, so a claim split across a line break still matches."""
    return s in TEX or re.sub(r"\s+", " ", s) in TEX_FLAT


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

ABL = ("-noimg", "-nograde", "-neither")
resp = sum(len(glob.glob(f"{d}/responses/*.json"))
           for d in glob.glob(str(ROOT / "data/runs/*"))
           if Path(d).is_dir() and not Path(d).name.startswith("_")
           and not any(Path(d).name.endswith(a) for a in ABL))
n_abl = sum(len(glob.glob(f"{d}/responses/*.json"))
            for d in glob.glob(str(ROOT / "data/runs/*"))
            if Path(d).is_dir() and any(Path(d).name.endswith(a) for a in ABL))
claim("5,194 released responses", resp == 5194 and in_tex("5,194"), str(resp))
claim(f"{n_abl:,} ablation-arm responses cited", in_tex(f"{n_abl:,}"), str(n_abl))

# --- leaderboard ------------------------------------------------------------
for lbl, b, key in (("test", T, "4.04"), ("held-out", H, "3.97")):
    claim(f"ours {lbl} {key}", f"{b[OURS]['overall_score']:.2f}" == key and in_tex(key))
claim("Opus held-out 3.82", f"{H['global.anthropic.claude-opus-4-6-v1']['overall_score']:.2f}" == "3.82")
claim("base held-out 3.18", f"{H[BASE]['overall_score']:.2f}" == "3.18")
gain = 100 * (H[OURS]["overall_score"] / H[BASE]["overall_score"] - 1)
claim("24.6% base-to-trained gain", abs(gain - 24.6) < 0.15 and in_tex("24.6"), f"{gain:.2f}%")
claim("max AUROC 0.562", abs(max(r["auroc"] for r in H.values()) - 0.562) < 5e-4 and in_tex("0.562"))
claim("min AUROC 0.456", abs(min(r["auroc"] for r in H.values()) - 0.456) < 5e-4 and in_tex("0.456"))

# --- the weakest-rubric claim ----------------------------------------------
for lbl, b in (("test", T), ("held-out", H)):
    means = {r: np.mean([x[r] for x in b.values()]) for r in R}
    lowest = min(means, key=means.get)
    claim(f"{lbl}: morphology grounding is the lowest mean rubric",
          lowest == "morphological_accuracy",
          f"lowest={lowest} {means[lowest]:.3f}, guide={means['guideline_alignment']:.3f}")

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
claim("ours over Opus +0.146 [0.078,0.214] held out",
      op and abs(op["diff"] - 0.146) < 0.006 and op["ci95"][0] > 0
      and in_tex("+0.146") and in_tex("[+0.078, +0.214]"),
      str(op and (op["diff"], op["ci95"], op["p"])))
opt = {(p["higher"], p["lower"]): p for p in PA["test"]["adjacent_rank_tests"]}.get(
    (OURS, "global.anthropic.claude-opus-4-6-v1"))
claim("ours over Opus +0.205 on test",
      opt and abs(opt["diff"] - 0.205) < 0.006 and in_tex("+0.205"),
      str(opt and (opt["diff"], opt["ci95"])))

# --- the two-by-two ablation -------------------------------------------------
EDGES = json.loads((ROOT / "data/runs/ablation_edges.json").read_text())
ARM_RUNS = {
    "GPT-5.4": "gpt-5_4-2026-03-05", "Opus 4.6": "global_anthropic_claude-opus-4-6-v1",
    "Qwen 397B": "qwen__qwen3_5-397b-a17b", "Gemini 2.5 Flash": "gemini-2_5-flash",
}
ARM_ALIAS = {"gemini-2_5-flash": "google__gemini-2_5-flash"}
ARM_SFX = {"image + grade": "", "grade only": "-noimg",
           "image only": "-nograde", "neither": "-neither"}


def arm_scores(base, sfx):
    run = (ARM_ALIAS.get(base, base) + sfx) if sfx else base
    if not (ROOT / "data/runs" / run).exists():
        run = base + sfx
    out = {}
    for f in glob.glob(str(ROOT / "data/runs" / run / "scores_sighted" / "*.json")):
        d = json.loads(Path(f).read_text())
        sc = {r["rubric"]: r["score"] for r in d["rubrics"] if r["score"] > 0}
        if len(sc) == len(R):
            sc["overall"] = float(np.mean([sc[r] for r in R]))
            out[Path(f).stem] = sc
    return out


for lbl, base in ARM_RUNS.items():
    arms = {a: arm_scores(base, s) for a, s in ARM_SFX.items()}
    shared = sorted(set.intersection(*[set(v) for v in arms.values()]))
    claim(f"2x2 {lbl}: all four arms present and overlapping", len(shared) >= 80, str(len(shared)))
    for a in ARM_SFX:
        for metric in ("overall", "morphological_accuracy"):
            v = np.mean([arms[a][c][metric] for c in shared])
            claim(f"2x2 {lbl} {a} {metric} {v:.2f}", in_tex(f"{v:.2f}"), f"{v:.3f}")
    # the arm with no inputs is not a floor on morphology for three of four
    nothing = np.mean([arms["neither"][c]["morphological_accuracy"] for c in shared])
    both = np.mean([arms["image + grade"][c]["morphological_accuracy"] for c in shared])
    if lbl != "GPT-5.4":
        claim(f"2x2 {lbl}: morphology higher with nothing than with both", nothing > both,
              f"{nothing:.3f} vs {both:.3f}")

grade_edge = EDGES["image + grade - image only"]
for lbl, want in (("Opus 4.6", 0.600), ("Gemini 2.5 Flash", 0.340),
                  ("Qwen 397B", 0.228), ("GPT-5.4", -0.130)):
    d = grade_edge[lbl]["morphological_accuracy"][0]
    claim(f"grade edge {lbl} {want:+.2f} on morphology", abs(d - want) < 0.006, f"{d:+.3f}")

# --- generation-budget truncation -------------------------------------------
CUT = {"gemini-2_5-flash": 440, "moonshotai__kimi-k2_5": 197, "qwen__qwen3_5-9b": 94,
       "ivf-bench-qwen9b-vlm-orpo": 74, "qwen__qwen3_5-397b-a17b": 12}
test_ids = {Path(f).stem for f in glob.glob(str(ROOT / "data/cases/*.json"))}
for run, want in CUT.items():
    n = sum(1 for f in glob.glob(str(ROOT / "data/runs" / run / "responses" / "*.json"))
            if Path(f).stem in test_ids
            and json.loads(Path(f).read_text()).get("finish_reason") == "length")
    claim(f"truncation {run} = {want}", n == want and in_tex(str(want)), str(n))
for run in ("gpt-5_4-2026-03-05", "global_anthropic_claude-opus-4-6-v1",
            "global_anthropic_claude-sonnet-4-6"):
    n = sum(1 for f in glob.glob(str(ROOT / "data/runs" / run / "responses" / "*.json"))
            if json.loads(Path(f).read_text()).get("finish_reason") == "length")
    claim(f"never truncated: {run}", n == 0, str(n))

# --- generation budget: the caps really did differ by backend ----------------
BUDGET = {"openrouter": 4096, "bedrock": 12000, "openai": 12000}
never_cut, was_cut = [], []
for d in sorted((ROOT / "data/runs").iterdir()):
    meta = d / "run_meta.json"
    if not d.is_dir() or d.name.startswith("_") or not meta.exists():
        continue
    m = json.loads(meta.read_text())
    if not all(m.get(k, True) for k in ("include_image", "include_grade")):
        continue
    n = sum(1 for f in glob.glob(str(d / "responses" / "*.json"))
            if Path(f).stem in test_ids
            and json.loads(Path(f).read_text()).get("finish_reason") == "length")
    (was_cut if n else never_cut).append((m.get("backend"), d.name))
claim("the three never-truncated systems are exactly the 12,000-token backends",
      all(BUDGET.get(b) == 12000 for b, _ in never_cut)
      and all(BUDGET.get(b) == 4096 for b, _ in was_cut)
      and len(never_cut) == 3,
      f"never={never_cut} cut={[b for b, _ in was_cut]}")
claim("paper states the two budgets", in_tex("12,000") and in_tex("4,096"))

# --- the grade-edge intervals printed in the text match the artifact ---------
for lbl in ("Opus 4.6", "Gemini 2.5 Flash", "Qwen 397B", "GPT-5.4"):
    d, lo, hi, _ = EDGES["image + grade - image only"][lbl]["morphological_accuracy"]
    txt = f"[{lo:+.2f}, {hi:+.2f}]".replace("+", "+")
    claim(f"grade-edge CI for {lbl} printed as {txt}",
          in_tex(f"$[{lo:+.2f}, {hi:+.2f}]$"), txt)

# --- the seventh-place tie rests on the 30 cases only Gemini answered --------
def _sighted(run):
    out = {}
    for f in glob.glob(str(ROOT / "data/runs" / run / "scores_sighted" / "*.json")):
        if Path(f).stem not in test_ids:
            continue
        sc = {r["rubric"]: r["score"]
              for r in json.loads(Path(f).read_text())["rubrics"] if r["score"] > 0}
        if len(sc) == len(R):
            out[Path(f).stem] = float(np.mean([sc[r] for r in R]))
    return out


_q, _g = _sighted("qwen__qwen3_5-9b"), _sighted("gemini-2_5-flash")
_only = sorted(set(_g) - set(_q))
_m = np.mean([_g[c] for c in _only])
_rest = np.mean([_g[c] for c in _g if c not in set(_only)])
claim(f"Gemini averages {_m:.2f} on the {len(_only)} unshared cases against {_rest:.2f} elsewhere",
      in_tex(f"{_m:.2f}") and in_tex(f"{_rest:.2f}"), f"{_m:.3f} vs {_rest:.3f}")

# --- the truncation rate range quoted in the limitations ---------------------
TOTALS = {"gemini-2_5-flash": 550, "moonshotai__kimi-k2_5": 550,
          "qwen__qwen3_5-9b": 520, "ivf-bench-qwen9b-vlm-orpo": 550,
          "qwen__qwen3_5-397b-a17b": 550}
rates = {}
for run, tot in TOTALS.items():
    n = sum(1 for f in glob.glob(str(ROOT / "data/runs" / run / "responses" / "*.json"))
            if Path(f).stem in test_ids
            and json.loads(Path(f).read_text()).get("finish_reason") == "length")
    rates[run] = 100 * n / tot
others = [v for k, v in rates.items() if k != "gemini-2_5-flash"]
# the per-system counts in Table~\ref{tab:truncation} carry this; the percentage
# restatement in the limitations was duplication and has been cut
for _run, _tot in TOTALS.items():
    _n = sum(1 for f in glob.glob(str(ROOT / "data/runs" / _run / "responses" / "*.json"))
             if Path(f).stem in test_ids
             and json.loads(Path(f).read_text()).get("finish_reason") == "length")
    claim(f"truncation count for {_run} ({_n}) appears", in_tex(f"{_n} / {_tot}") or in_tex(str(_n)),
          f"{_n}/{_tot}")
claim("Gemini truncation described as four fifths",
      abs(rates["gemini-2_5-flash"] - 80.0) < 0.5, f"{rates['gemini-2_5-flash']:.1f}%")

# --- cross-judge -------------------------------------------------------------
CJ = json.loads((ROOT / "data/runs/cross_judge.json").read_text())["per_model"]
for name, want_gain in (("Sonnet 4.6", 0.90), ("Opus 4.6", 0.79),
                        ("GPT-5.4", 0.18), ("Ours (9B-ORPO)", 0.16)):
    g = CJ[name]["sonnet"] - CJ[name]["gpt"]
    claim(f"cross-judge gain {name} {want_gain:+.2f}", abs(g - want_gain) < 0.006, f"{g:+.3f}")
gains = {k: v["sonnet"] - v["gpt"] for k, v in CJ.items()}
top2 = sorted(gains, key=gains.get, reverse=True)[:2]
bot2 = sorted(gains, key=gains.get)[:2]
claim("both Anthropic systems gain most from the Anthropic judge",
      set(top2) == {"Sonnet 4.6", "Opus 4.6"}, str(top2))
claim("both GPT-lineage systems gain least",
      set(bot2) == {"GPT-5.4", "Ours (9B-ORPO)"}, str(bot2))

# --- where the larger-budget systems actually finish -------------------------
BIG = {"gpt-5.4-2026-03-05", "global.anthropic.claude-opus-4-6-v1",
       "global.anthropic.claude-sonnet-4-6"}
order = sorted(T, key=lambda m: -T[m]["overall_score"])
pos = [i for i, m in enumerate(order, 1) if m in BIG]
words = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
         6: "sixth", 7: "seventh", 8: "eighth"}
claim("test-split positions of the 12,000-token systems are stated correctly",
      in_tex(f"finish {words[pos[0]]}, {words[pos[1]]} and {words[pos[2]]}"), str(pos))

# --- day of imaging and transfer --------------------------------------------
days = {}
for d in ("cases", "validation_cases", "held_out_cases"):
    for f in glob.glob(str(ROOT / "data" / d / "*.json")):
        c = json.loads(Path(f).read_text())
        days[c["case_id"]] = c["lab_data"]["transfer_day"]
n5 = sum(1 for v in days.values() if v == 5)
n4 = sum(1 for v in days.values() if v == 4)
claim(f"{n5} day-5 and {n4} day-4 cases, and the paper does not claim a single imaging day",
      in_tex(str(n5)) and in_tex(str(n4)) and not in_tex("imaged on day 5"),
      f"{n5}/{n4}")

# --- study cost: every figure quoted, and the arithmetic between them --------
COST = json.loads((ROOT / "data/runs/study_cost.json").read_text())
judging_reported = sum(v for k, v in COST["judging"].items()
                       if not k.startswith("superseded"))
for label, val in (("GPU compute", COST["compute"]),
                   ("main-run inference", COST["inference"]["main runs"]),
                   ("ablation inference", COST["inference"]["ablation arm"]),
                   ("judging subtotal", judging_reported),
                   ("reported total", COST["reported_total"]),
                   ("superseded total", COST["superseded_total"])):
    claim(f"cost quoted to the cent: {label} ${val:,.2f}",
          in_tex(f"{val:,.2f}"), f"{val:.2f}")
# the figures the paper prints must themselves add up, or a reader who sums the
# line items lands somewhere the paper never claims
printed = [float(x.replace(",", "")) for x in
           re.findall(r"\\\$([\d,]+\.\d\d)", TEX)]
for label, parts, total in (
        ("reported", [COST["compute"], COST["inference"]["main runs"],
                      COST["inference"]["ablation arm"], judging_reported],
         COST["reported_total"]),
        ("superseded", [v for k, v in COST["judging"].items()
                        if k.startswith("superseded")], COST["superseded_total"])):
    claim(f"{label} line items sum to the stated subtotal",
          abs(sum(parts) - total) < 0.005, f"{sum(parts):.2f} vs {total:.2f}")
    for v in parts:
        if v:
            claim(f"{label} line item ${v:,.2f} appears verbatim", v in printed, str(v))
parts = COST["compute"] + sum(COST["inference"].values()) + judging_reported
claim("cost components sum to the reported total",
      abs(parts - COST["reported_total"]) < 0.01, f"{parts:.2f}")
claim("reported plus superseded equals the grand total",
      abs(COST["reported_total"] + COST["superseded_total"] - COST["grand_total"]) < 0.01)

# --- the per-rubric extremes the figure caption and prose both assert --------
_hi = sum(1 for r in H.values() if max(R, key=lambda k: r[k]) == "clinical_integration")
_lo = sum(1 for r in H.values() if min(R, key=lambda k: r[k]) == "morphological_accuracy")
_hi_t = sum(1 for r in T.values() if max(R, key=lambda k: r[k]) == "clinical_integration")
claim("clinical integration is the top rubric for all 8 on both splits",
      _hi == 8 and _hi_t == 8, f"held-out {_hi}, test {_hi_t}")
claim("held-out: morphology lowest for 7 of 8, as the caption says",
      _lo == 7 and in_tex("weakest for seven of eight"), str(_lo))

# --- the empty ablation arm: how often it is lowest, and highest -------------
_ARMS = ["image + grade", "grade only", "image only", "neither"]
_low = _high = 0
for lbl, base in ARM_RUNS.items():
    a = {arm: arm_scores(base, ARM_SFX[arm]) for arm in _ARMS}
    sh = sorted(set.intersection(*[set(v) for v in a.values()]))
    ov = [np.mean([a[arm][c]["overall"] for c in sh]) for arm in _ARMS]
    mo = [np.mean([a[arm][c]["morphological_accuracy"] for c in sh]) for arm in _ARMS]
    _low += ov[3] == min(ov)
    _high += mo[3] == max(mo)
claim(f"empty arm lowest on overall for {_low} of 4, highest on morphology for {_high} of 4",
      _low == 1 and _high == 3 and in_tex("lowest on overall\nscore for only one of the four"),
      f"low={_low} high={_high}")

# --- blind vs sighted: the swaps and the one confident disagreement ----------
def _pair_ci(r1, r2, sub, ids):
    def _t(run):
        o = {}
        for f in glob.glob(str(ROOT / "data/runs" / run / sub / "*.json")):
            if Path(f).stem not in ids:
                continue
            sc = {x["rubric"]: x["score"]
                  for x in json.loads(Path(f).read_text())["rubrics"] if x["score"] > 0}
            if len(sc) == len(R):
                o[Path(f).stem] = float(np.mean([sc[k] for k in R]))
        return o
    A, B = _t(r1), _t(r2)
    sh = sorted(set(A) & set(B))
    d = np.array([A[c] for c in sh]) - np.array([B[c] for c in sh])
    rng = np.random.default_rng(42)
    m = d[rng.integers(0, len(d), size=(10_000, len(d)))].mean(axis=1)
    return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


_SON, _KIM = "global_anthropic_claude-sonnet-4-6", "moonshotai__kimi-k2_5"
_b = _pair_ci(_SON, _KIM, "scores", test_ids)
_g = _pair_ci(_SON, _KIM, "scores_sighted", test_ids)
claim("blind and sighted judges separate Sonnet/Kimi on test in opposite directions",
      _b[2] < 0 and _g[1] > 0, f"blind {_b[0]:+.3f}{_b[1:]} sighted {_g[0]:+.3f}{_g[1:]}")
claim("both magnitudes are quoted",
      in_tex(f"{abs(_b[0]):.3f}") and in_tex(f"{abs(_g[0]):.3f}"),
      f"{abs(_b[0]):.3f} / {abs(_g[0]):.3f}")

# --- Gemini's two splits were routed differently -----------------------------
_gm = json.loads((ROOT / "data/runs/provider_manifest.json").read_text())["gemini-2_5-flash"]["per_case"]
_ho = {Path(f).stem for f in glob.glob(str(ROOT / "data/held_out_cases/*.json"))}
_t_named = sum(1 for k, v in _gm.items() if k in test_ids and v)
_h_named = sum(1 for k, v in _gm.items() if k in _ho and v)
claim("Gemini: no provider named on any test call, named on every held-out call",
      _t_named == 0 and _h_named == len(_ho), f"test named {_t_named}, held-out named {_h_named}")

# --- our model is locally served, never "on OpenRouter" ----------------------
# This slipped through once already: the five systems on the 4,096 budget are the
# four OpenRouter baselines plus our own vLLM instance, not five OpenRouter rows.
_flat_tex = re.sub(r"\s+", " ", TEX)
for _bad in ("five on OpenRouter", "five systems served through OpenRouter",
             "all five OpenRouter systems", "five OpenRouter rows"):
    claim(f"does not miscount our local model as an OpenRouter row: '{_bad}'",
          _bad not in _flat_tex)

# --- provider-mixture counts for every OpenRouter row ------------------------
_MIX = {"moonshotai__kimi-k2_5": None, "qwen__qwen3_5-397b-a17b": None,
        "qwen__qwen3_5-9b": None}
_all2 = test_ids | {Path(f).stem for f in glob.glob(str(ROOT / "data/held_out_cases/*.json"))}
for _run in _MIX:
    _p = set()
    for f in glob.glob(str(ROOT / "data/runs" / _run / "raw" / "*.json")):
        if Path(f).stem in _all2:
            k = json.loads(Path(f).read_text()).get("provider")
            if k:
                _p.add(k)
    _MIX[_run] = len(_p)
claim(f"provider counts quoted: Kimi {_MIX['moonshotai__kimi-k2_5']}, "
      f"397B {_MIX['qwen__qwen3_5-397b-a17b']}, 9B {_MIX['qwen__qwen3_5-9b']}",
      all(in_tex(str(v)) for v in _MIX.values()), str(_MIX))
_man = json.loads((ROOT / "data/runs/provider_manifest.json").read_text())
for _r, _lbl in (("moonshotai__kimi-k2_5", "Kimi"), ("qwen__qwen3_5-397b-a17b", "397B"),
                 ("qwen__qwen3_5-9b", "9B"), ("gemini-2_5-flash", "Gemini")):
    _c = _man[_r]["counts"]
    _named_n = sum(v for k, v in _c.items() if k != "(unrecorded)")
    claim(f"{_lbl}: {_named_n} of {sum(_c.values())} calls name a provider, both quoted",
          in_tex(str(_named_n)) and in_tex(str(sum(_c.values()))),
          f"{_named_n}/{sum(_c.values())}")
claim("the largest provider mixture is stated",
      in_tex(f"{max(_MIX.values())} distinct providers"), str(max(_MIX.values())))

# --- which providers actually served the base model --------------------------
_prov = {}
_all_ids = test_ids | {Path(f).stem for f in glob.glob(str(ROOT / "data/held_out_cases/*.json"))}
for f in glob.glob(str(ROOT / "data/runs/qwen__qwen3_5-9b/raw/*.json")):
    if Path(f).stem not in _all_ids:
        continue
    k = json.loads(Path(f).read_text()).get("provider")
    _prov[k] = _prov.get(k, 0) + 1
_named = {k: v for k, v in _prov.items() if k}
claim("base model was served by more than one provider", len(_named) > 1, str(_named))
for k, v in sorted(_named.items(), key=lambda kv: -kv[1]):
    claim(f"provider count quoted: {k} = {v}", in_tex(str(v)), str(v))
claim(f"total recorded calls {sum(_prov.values())} quoted",
      in_tex(str(sum(_prov.values()))), str(sum(_prov.values())))
claim(f"unrecorded-provider count {_prov.get(None, 0)} quoted",
      in_tex(str(_prov.get(None, 0))), str(_prov.get(None, 0)))

# --- the training hyperparameters the paper quotes ---------------------------
_cfg = yaml.safe_load(open(ROOT / "configs/training_orpo_qwen_vlm_final.yaml"))["params"]
claim("LoRA rank 8, alpha 36, three epochs as configured",
      _cfg["lora_r"] == 8 and _cfg["lora_alpha"] == 36 and _cfg["epochs"] == 3
      and in_tex("rank 8 and alpha 36") and in_tex("three epochs"),
      str({k: _cfg[k] for k in ("lora_r", "lora_alpha", "epochs")}))
claim(f"learning rate {_cfg['lr']:.2e} quoted",
      in_tex(f"{_cfg['lr'] * 1e4:.2f}") and in_tex("10^{-4}"), f"{_cfg['lr']:.6g}")
claim(f"preference weight beta {_cfg['dpo_beta']:.3f} quoted",
      in_tex(f"{_cfg['dpo_beta']:.3f}"), f"{_cfg['dpo_beta']:.6g}")

# --- the generated age proxies -----------------------------------------------
_cs = [json.loads(Path(f).read_text()) for d in ("cases", "validation_cases", "held_out_cases")
       for f in glob.glob(str(ROOT / "data" / d / "*.json"))]
_age = np.array([c["real_clinical"]["age"] for c in _cs], float)
_y = np.array([int(bool(c["real_outcome"]["clinical_pregnancy"])) for c in _cs])


def _auroc_abs(v):
    pos, neg = v[_y == 1], v[_y == 0]
    a = sum((x > z) + 0.5 * (x == z) for x in pos for z in neg) / (len(pos) * len(neg))
    return max(a, 1 - a)


for _f, _lbl in (("fsh_iu_l", "basal FSH"), ("partner_age", "partner age")):
    _v = np.array([c["patient_context"][_f] for c in _cs], float)
    _r = np.corrcoef(_age, _v)[0, 1]
    claim(f"{_lbl}: correlation with age {_r:.2f} quoted",
          in_tex(f"{_r:.2f}"), f"{_r:.4f}")
    claim(f"{_lbl}: standalone AUROC {_auroc_abs(_v):.3f} quoted",
          in_tex(f"{_auroc_abs(_v):.3f}"), f"{_auroc_abs(_v):.4f}")

# --- the source-data corruption counts, recomputed from the released CSV ------
import csv as _csv
_M = r"(?:J[a\u00e4]n|Feb|M[a\u00e4]r|Apr|Mai|Jun|Jul|Aug|Sep|Okt|Nov|Dez)"
_date = re.compile(rf"^(?:{_M}\.\d+|\d+\.{_M})$", re.I)
with open(ROOT / "data/raw/Clincial_annotations.csv", encoding="latin-1") as _fh:
    _rows = list(_csv.DictReader(_fh, delimiter=";"))
claim("754 source records", len(_rows) == 754 and in_tex("754"), str(len(_rows)))
_pat = {r["Image"].split("_")[0] for r in _rows}
claim("732 patients in the source", len(_pat) == 732 and in_tex("732"), str(len(_pat)))
for _col, _label in (("AMH", "AMH"), ("Endo", "endometrial thickness")):
    _n = sum(1 for r in _rows if _date.match((r[_col] or "").strip()))
    claim(f"{_n} {_label} values are date-corrupted, as stated",
          in_tex(str(_n)), str(_n))

# --- the preference dataset the paper describes ------------------------------
_train = [json.loads(l) for l in open(ROOT / "data/orpo/train_full.jsonl")]
_ev = [json.loads(l) for l in open(ROOT / "data/orpo/eval_full.jsonl")]
claim("500 train and 50 eval pairs", len(_train) == 500 and len(_ev) == 50,
      f"{len(_train)}/{len(_ev)}")
_n = sum(1 for r in _train if r["chosen_model"] == "gpt-5.4-2026-03-05")
claim(f"GPT-5.4 authored {_n} of 500 chosen sides, {round(100*_n/500)}%",
      in_tex(str(_n)) and in_tex(f"{round(100*_n/500)}\\%"), str(_n))
_gap = np.mean([r["chosen_score"] - r.get("rejected_score", 0) for r in _train + _ev]
               ) if "rejected_score" in _train[0] else np.mean(
    [r["score_gap"] for r in _train + _ev])
claim(f"mean score gap {_gap:.2f}", in_tex(f"{_gap:.2f}"), f"{_gap:.4f}")

# --- serving cost of our model, from measured latency ------------------------
_lat = [json.loads(Path(f).read_text()).get("latency_ms", 0)
        for f in glob.glob(str(ROOT / "data/runs/ivf-bench-qwen9b-vlm-orpo/responses/*.json"))
        if Path(f).stem in test_ids]
_sec = np.mean([x for x in _lat if x]) / 1000
claim(f"{_sec:.0f} seconds per case", in_tex(f"{_sec:.0f} seconds per case"), f"{_sec:.2f}")
claim(f"${8.38 * _sec / 3600:.3f} per case at $8.38/h",
      in_tex(f"{8.38 * _sec / 3600:.3f}"), f"{8.38 * _sec / 3600:.4f}")
claim("GPU line equals 51.5 h x $8.38", abs(51.5 * 8.38 - COST["compute"]) < 0.01)

# --- no point estimate may carry two different intervals ---------------------
# "+0.146 ... [+0.078, +0.214]" in one section and "[+0.078, +0.212]" in another
# is the kind of drift two scripts resampling the same quantity produce.
_flat = re.sub(r"\s+", " ", TEX)
_pairs = re.findall(r"([+-]\d\.\d{3})\$?[^.]{0,80}?\$?\[([+-]\d\.\d{3}), ([+-]\d\.\d{3})\]", _flat)
_seen = {}
_clash = []
for est, lo, hi in _pairs:
    if est in _seen and _seen[est] != (lo, hi):
        _clash.append((est, _seen[est], (lo, hi)))
    _seen.setdefault(est, (lo, hi))
claim("no point estimate is quoted with two different intervals",
      not _clash, str(_clash))

# --- everything the paper says it releases must not be gitignored ------------
_ignored = [l.strip() for l in (ROOT / ".gitignore").read_text().splitlines()
            if l.strip() and not l.startswith("#")]


def _is_ignored(rel):
    import fnmatch
    return any(fnmatch.fnmatch(rel, pat.rstrip("/")) or
               fnmatch.fnmatch(rel, pat.rstrip("/") + "/*") for pat in _ignored)


for _named in ("scores_sighted", "scores", "scores_sonnet_sighted",
               "scores_stale_cot", "scores_zero_judge_failure",
               "scores_opus_judged", "scores_prompt_confounded",
               "scores_grade_confounded"):
    claim(f"released directory {_named} is not gitignored",
          not _is_ignored(f"data/runs/gpt-5_4-2026-03-05/{_named}"), _named)
claim("provider manifest exists and is shipped",
      (ROOT / "data/runs/provider_manifest.json").exists()
      and not _is_ignored("data/runs/provider_manifest.json")
      and in_tex("provider"))

# --- Section 7 announces a number of checks; count the actual ones ------------
_r0 = TEX.index(r"\section{What the benchmark itself gets wrong}")
_r1 = TEX.index(r"\section{Four things we learned")
_sec7 = TEX[_r0:_r1]
_n_checks = len(re.findall(r"\\textbf\{[A-Z][^}]*\?\}", _sec7))
_words = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight"}
claim(f"Section 7 announces its actual check count ({_n_checks})",
      re.search(rf"report {_words[_n_checks]}\s+(such\s+)?checks", _sec7) is not None,
      str(_n_checks))
claim("and how many of them were free",
      in_tex(f"{_words[_n_checks - 1]} of which reuse the collected responses"),
      f"{_n_checks - 1} free")

# --- document hygiene: every float anchored, every citation resolved ---------
_labels = re.findall(r"\\label\{((?:fig|tab):[^}]+)\}", TEX)
_refs = set(re.findall(r"\\ref\{([^}]+)\}", TEX))
claim("every figure and table is referenced from the prose",
      all(l in _refs for l in _labels),
      str([l for l in _labels if l not in _refs]))
claim("every \\ref resolves to a label",
      not (_refs - set(re.findall(r"\\label\{([^}]+)\}", TEX))),
      str(_refs - set(re.findall(r"\\label\{([^}]+)\}", TEX))))
_bib = set(re.findall(r"@\w+\{([^,]+),", (ROOT / "arxiv/references.bib").read_text()))
_cited = {k.strip() for c in re.findall(r"\\cite[tp]?\{([^}]+)\}", TEX) for k in c.split(",")}
claim("citations and bibliography agree in both directions",
      _cited == _bib, str(_cited ^ _bib))
# NB: "\\bibliography" also matches "\\bibliographystyle" in the preamble, which
# silently made this slice empty and the two checks below vacuous.
_body = TEX[TEX.index("\\begin{abstract}"):TEX.index("\\bibliography{")]
assert len(_body) > 10000, "body slice collapsed; the hygiene checks would be vacuous"
_dbl = [m.group(0) for m in re.finditer(r"\b(\w+)\s+\1\b", re.sub(r"\s+", " ", _body), re.I)
        if m.group(1).lower() not in ("had", "that", "is")]
claim("no doubled words in the body", not _dbl, str(_dbl))
# a repeated sentence is what a bad splice leaves behind, and the doubled-word
# check does not see it
_sents = [x.strip() for x in re.split(r"(?<=[.!?]) ", re.sub(r"\s+", " ", _body))
          if len(x.split()) >= 8]
_dupes = {x for x in _sents if _sents.count(x) > 1}
claim("no sentence appears twice in the body", not _dupes, str(list(_dupes)[:2]))
# runaway-length guard. The register here is deliberately long and subordinated,
# so this is set well above normal, at the point where a sentence stops parsing.
_prose = re.sub(r"\\begin\{(table|figure|tikzpicture|axis|tabular|lstlisting)\}.*?"
                r"\\end\{\1\}", " ", _body, flags=re.S)
_prose = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^{}]*\})?", " ", _prose)
_prose = re.sub(r"\s+", " ", re.sub(r"[{}$\\&%]", " ", _prose))
_runaway = [x.strip()[:60] for x in re.split(r"(?<=[.!?]) ", _prose)
            if len(x.split()) > 90]
claim("no prose sentence runs past 90 words", not _runaway, str(_runaway))

# --- claims that must NOT appear -------------------------------------------
for bad, why in (("one twentieth", "superseded cost ratio"),
                 ("expert Gardner annotation", "labels are silver, not gold"),
                 ("Morphology grounding is the weakest rubric throughout", "false"),
                 ("550 benchmark-derived preference pairs", "trained on 500"),
                 ("carry no information\nabout which patients conceived", "false"),
                 ("The ordering survives a\nsecond judge", "it does not, under Sonnet 4.6"),
                 ("does not observe the image", "the judge is sighted now"),
                 ("Two evaluation defects", "there are three"),
                 ("guideline alignment is lowest on both splits", "morphology is")):
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
