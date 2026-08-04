"""Rewrite the leaderboard tables in README.md and the release cards.

Reads data/runs/leaderboard_held_out.json, which is built from the sighted judge,
and replaces the markdown table in each document in place. Every replacement
asserts that its anchor matched, so a silent no-op is impossible.

Usage: python scripts/refresh_cards.py
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "data" / "runs"
COLS = ["overall_score", "morphological_accuracy", "clinical_integration",
        "reasoning_coherence", "guideline_alignment", "actionability"]
NAME = {
    "gpt-5.4-2026-03-05": "GPT-5.4",
    "ivf-bench-qwen9b-vlm-orpo": None,          # filled per document
    "global.anthropic.claude-opus-4-6-v1": "Claude Opus 4.6",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "qwen/qwen3.5-397b-a17b": "Qwen 3.5-397B",
    "moonshotai/kimi-k2.5": "Kimi K2.5",
    "global.anthropic.claude-sonnet-4-6": "Claude Sonnet 4.6",
    "qwen/qwen3.5-9b": "Qwen 3.5-9B (base)",
}
SLUG = {
    "gpt-5.4-2026-03-05": "gpt-5_4-2026-03-05",
    "global.anthropic.claude-opus-4-6-v1": "global_anthropic_claude-opus-4-6-v1",
    "gemini-2.5-flash": "gemini-2_5-flash",
    "qwen/qwen3.5-397b-a17b": "qwen__qwen3_5-397b-a17b",
    "moonshotai/kimi-k2.5": "moonshotai__kimi-k2_5",
    "global.anthropic.claude-sonnet-4-6": "global_anthropic_claude-sonnet-4-6",
    "qwen/qwen3.5-9b": "qwen__qwen3_5-9b",
    "ivf-bench-qwen9b-vlm-orpo": "ivf-bench-qwen9b-vlm-orpo",
}


def board() -> list[dict]:
    d = json.loads((RUNS / "leaderboard_held_out.json").read_text())
    rows = d if isinstance(d, list) else d.get("models", d.get("leaderboard", []))
    return sorted(rows, key=lambda r: -r["overall_score"])


def test_split_cost(model: str) -> float:
    """Inference plus judging over the 550 test cases, as the README column means."""
    ids = {Path(f).stem for f in glob.glob(str(ROOT / "data/cases/*.json"))}
    run = RUNS / SLUG[model]
    tot = 0.0
    for sub, key in (("responses", "cost_usd"), ("scores_sighted", "judge_cost_usd")):
        for f in (run / sub).glob("*.json"):
            if f.stem in ids:
                tot += json.loads(f.read_text()).get(key, 0) or 0
    return tot


def table(ours_label: str, with_rank: bool, with_cost: bool) -> str:
    head = (["#"] if with_rank else []) + ["Model", "Overall", "Morph", "Clinical",
            "Reasoning", "Guideline", "Recommend", "Brier", "AUROC"]
    if with_cost:
        head.append("Cost / 550")
    out = ["| " + " | ".join(head) + " |",
           "|" + "---|" * len(head)]
    for i, r in enumerate(board(), 1):
        name = ours_label if r["model"] == "ivf-bench-qwen9b-vlm-orpo" else NAME[r["model"]]
        cells = ([str(i)] if with_rank else []) + [name]
        cells += [f"{r[c]:.2f}" for c in COLS]
        cells += [f"{r['brier_score']:.3f}", f"{r['auroc']:.3f}"]
        if with_cost:
            c = test_split_cost(r["model"])
            cells.append(f"${c:,.2f}" + ("*" if r["model"] == "ivf-bench-qwen9b-vlm-orpo" else ""))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def swap(path: Path, new_table: str) -> None:
    s = path.read_text()
    m = re.search(r"^\| #? ?\|? ?Model \|.*?(?=\n\n)", s, re.S | re.M)
    if m is None:
        m = re.search(r"^\| Model \|.*?(?=\n\n)", s, re.S | re.M)
    assert m, f"no markdown leaderboard table found in {path.name}"
    path.write_text(s[:m.start()] + new_table + s[m.end():])
    print(f"  rewrote the table in {path.name}")


def main() -> None:
    swap(ROOT / "README.md",
         table("**IVF-Bench-Qwen9B-ORPO (ours)**", with_rank=True, with_cost=True))
    swap(ROOT / "release" / "MODEL_CARD.md",
         table("**This model (9B)**", with_rank=False, with_cost=False))
    print("\n  costs over the 550 test cases, inference plus judging:")
    for r in board():
        print(f"    {NAME[r['model']] or 'ours':22s} ${test_split_cost(r['model']):8,.2f}")


if __name__ == "__main__":
    main()
