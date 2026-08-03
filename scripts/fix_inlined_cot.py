"""Move inlined chain-of-thought out of response_text into reasoning_text.

The locally-served vLLM run stored the model's whole generation in response_text,
so the judge scored ~30% raw reasoning that no other backend exposes. This
rewrites the affected response files in place (originals backed up first) so the
run can be re-judged on the answer alone, with no new inference.

Usage:
    python scripts/fix_inlined_cot.py --dry-run
    python scripts/fix_inlined_cot.py --apply [--run <run-dir-name>]

After applying, delete that run's score files and re-run `ivf-bench score`.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "data" / "runs"


def split_inline_thinking(content: str) -> tuple[str, str | None]:
    """Same rule as runner._split_inline_thinking — keep the two in sync."""
    if not content or "</think>" not in content:
        return content, None
    head, _, answer = content.rpartition("</think>")
    reasoning = head.replace("<think>", "").strip()
    return answer.strip(), reasoning or None


def process(run_dir: Path, apply: bool) -> dict:
    resp_dir = run_dir / "responses"
    backup = run_dir / "responses_raw_with_cot"
    stale_scores = run_dir / "scores_stale_cot"
    affected, cot_len, ans_len, empty = [], [], [], []

    for f in sorted(resp_dir.glob("*.json")):
        data = json.loads(f.read_text())
        text = data.get("response_text") or ""
        if data.get("reasoning_text") or "</think>" not in text:
            continue
        answer, reasoning = split_inline_thinking(text)
        if not answer.strip():
            empty.append(f.stem)  # nothing but reasoning — leave it alone
            continue
        affected.append(f.stem)
        cot_len.append(len(reasoning or ""))
        ans_len.append(len(answer))
        if apply:
            if not backup.exists():
                backup.mkdir()
            if not (backup / f.name).exists():
                shutil.copy2(f, backup / f.name)
            data["response_text"] = answer
            data["reasoning_text"] = reasoning
            f.write_text(json.dumps(data, indent=2))

    # The old score for a rewritten response was computed on text that no longer
    # exists. Leaving it in place would make score_run skip the case and the
    # leaderboard would silently mix old and new judgements, so quarantine it.
    moved = 0
    if apply and affected:
        for cid in affected:
            old = run_dir / "scores" / f"{cid}.json"
            if old.exists():
                stale_scores.mkdir(exist_ok=True)
                shutil.move(str(old), str(stale_scores / f"{cid}.json"))
                moved += 1

    return {
        "run": run_dir.name,
        "n_affected": len(affected),
        "n_answer_only_empty": len(empty),
        "median_cot_chars": int(np.median(cot_len)) if cot_len else 0,
        "median_answer_chars": int(np.median(ans_len)) if ans_len else 0,
        "backup_dir": str(backup) if apply and affected else None,
        "stale_scores_moved": moved,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="write changes (default is dry run)")
    ap.add_argument("--run", default=None, help="limit to one run directory name")
    args = ap.parse_args()

    targets = [d for d in sorted(RUNS.iterdir())
               if d.is_dir() and not d.name.startswith("_") and (d / "responses").exists()
               and (args.run is None or d.name == args.run)]

    for d in targets:
        r = process(d, args.apply)
        if r["n_affected"] or r["n_answer_only_empty"]:
            print(f"{r['run']}: {r['n_affected']} responses with inlined CoT "
                  f"(median {r['median_cot_chars']} chars CoT / {r['median_answer_chars']} chars answer)"
                  + (f", {r['n_answer_only_empty']} skipped as answer-empty" if r["n_answer_only_empty"] else "")
                  + (f"\n  originals backed up to {r['backup_dir']}" if r["backup_dir"] else "")
                  + (f"\n  {r['stale_scores_moved']} now-stale score files moved to scores_stale_cot/"
                     if r.get("stale_scores_moved") else ""))
    if not args.apply:
        print("\nDry run — nothing written. Re-run with --apply to rewrite.")


if __name__ == "__main__":
    main()
