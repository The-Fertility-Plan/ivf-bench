#!/usr/bin/env python3
"""Build ORPO preference dataset from IVF-Bench scored model responses.

For each benchmark case, selects the best-scoring and worst-scoring model
responses as chosen/rejected pairs. Outputs JSONL compatible with
aitraining VLM ORPO (v0.0.53+).

Usage:
    python scripts/build_orpo_dataset.py [OPTIONS]

Output:
    data/orpo/train.jsonl   (500 cases)
    data/orpo/eval.jsonl    (50 cases)
"""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

console = Console()
app = typer.Typer(add_completion=False)

RUNS_DIR = Path("data/runs")
CASES_DIR = Path("data/cases")
IMAGES_DIR = Path("data/raw/Images")
OUTPUT_DIR = Path("data/orpo")

# Models to exclude from preference pair selection (e.g. the one we're training)
EXCLUDE_MODELS: set[str] = set()

RUBRICS = [
    "morphological_accuracy",
    "clinical_integration",
    "reasoning_coherence",
    "guideline_alignment",
    "actionability",
]


def _load_scores(run_dir: Path) -> dict[str, dict]:
    """Load all score files for a model run. Returns {case_id: score_data}."""
    # Deliberately the blind judge, not scores_sighted: the released preference
    # pairs were selected before the judge was shown the embryo, and rebuilding
    # them from a different judge would not reproduce the dataset we trained on.
    # The paper states this in the post-training section.
    scores_dir = run_dir / "scores"
    if not scores_dir.exists():
        return {}
    result = {}
    for f in scores_dir.glob("*.json"):
        data = json.loads(f.read_text())
        case_id = data["case_id"]
        rubric_scores = {r["rubric"]: r["score"] for r in data.get("rubrics", [])}
        if not rubric_scores:
            continue
        avg = sum(rubric_scores.values()) / len(rubric_scores)
        # Skip corrupted scores (all zeros from judge failures)
        if avg == 0.0:
            continue
        result[case_id] = {
            "rubric_scores": rubric_scores,
            "avg_score": avg,
        }
    return result


def _load_response_text(run_dir: Path, case_id: str, include_thinking: bool = True) -> Optional[str]:
    """Load the response text for a specific case from a model run.

    If include_thinking is True and reasoning_text exists, formats as:
        <think>
        {reasoning_text}
        </think>

        {response_text}
    """
    resp_file = run_dir / "responses" / f"{case_id}.json"
    if not resp_file.exists():
        return None
    data = json.loads(resp_file.read_text())
    if data.get("error"):
        return None
    text = data.get("response_text", "")
    if not text.strip():
        return None
    reasoning = data.get("reasoning_text", "")
    if include_thinking and reasoning and reasoning.strip():
        return f"<think>\n{reasoning.strip()}\n</think>\n\n{text}"
    return text


def _discover_models(runs_dir: Path, exclude: set[str]) -> list[tuple[str, Path]]:
    """Discover all scored model runs. Returns [(model_name, run_dir)]."""
    models = []
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        meta_file = d / "run_meta.json"
        scores_dir = d / "scores"
        if meta_file.exists() and scores_dir.exists() and any(scores_dir.glob("*.json")):
            meta = json.loads(meta_file.read_text())
            model_name = meta["model"]
            if model_name not in exclude:
                models.append((model_name, d))
    return models


def _load_case(case_id: str, cases_dir: Path) -> Optional[dict]:
    """Load a benchmark case JSON."""
    f = cases_dir / f"{case_id}.json"
    if not f.exists():
        return None
    return json.loads(f.read_text())


def build_preference_pairs(
    runs_dir: Path,
    cases_dir: Path,
    images_dir: Path,
    exclude_models: set[str],
    min_score_gap: float = 0.0,
    copy_images_to: Optional[Path] = None,
) -> list[dict]:
    """Build ORPO preference pairs from scored model runs.

    For each case, picks the highest-scoring model response as 'chosen'
    and the lowest-scoring as 'rejected'. Skips cases where all models
    tied or the gap is below min_score_gap.
    """
    models = _discover_models(runs_dir, exclude_models)
    if not models:
        console.print("[red]No scored model runs found.[/red]")
        return []

    console.print(f"Found {len(models)} scored models:")
    for name, _ in models:
        console.print(f"  - {name}")

    # Load all scores
    model_scores: dict[str, dict[str, dict]] = {}
    for name, run_dir in models:
        model_scores[name] = _load_scores(run_dir)

    # Find all case IDs from the cases directory
    case_ids = sorted(f.stem for f in cases_dir.glob("IVF-BENCH-*.json"))
    console.print(f"\n{len(case_ids)} benchmark cases found")

    pairs = []
    skipped_no_scores = 0
    skipped_no_gap = 0
    skipped_no_response = 0

    for case_id in case_ids:
        # Collect (model_name, avg_score) for this case
        candidates = []
        for name, _ in models:
            if case_id in model_scores.get(name, {}):
                score_info = model_scores[name][case_id]
                candidates.append((name, score_info["avg_score"]))

        if len(candidates) < 2:
            skipped_no_scores += 1
            continue

        # Sort by score descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_model, best_score = candidates[0]
        worst_model, worst_score = candidates[-1]

        gap = best_score - worst_score
        if gap < min_score_gap:
            skipped_no_gap += 1
            continue

        # Load actual response texts
        best_run_dir = next(d for n, d in models if n == best_model)
        worst_run_dir = next(d for n, d in models if n == worst_model)

        chosen_text = _load_response_text(best_run_dir, case_id)
        rejected_text = _load_response_text(worst_run_dir, case_id)

        if not chosen_text or not rejected_text:
            skipped_no_response += 1
            continue

        # Load the case to get the prompt and image path
        case_data = _load_case(case_id, cases_dir)
        if not case_data:
            skipped_no_response += 1
            continue

        prompt_text = case_data["prompt"]
        image_filename = case_data["image_path"]  # e.g. "0001_01.png"

        # Determine image path for the dataset
        if copy_images_to:
            src = images_dir / image_filename
            dst = copy_images_to / image_filename
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
            img_ref = image_filename  # relative to dataset images dir
        else:
            img_ref = str(images_dir / image_filename)  # absolute path

        pairs.append({
            "images": [img_ref],
            "prompt": prompt_text,
            "chosen": chosen_text,
            "rejected": rejected_text,
            "case_id": case_id,
            "chosen_model": best_model,
            "rejected_model": worst_model,
            "chosen_score": round(best_score, 3),
            "rejected_score": round(worst_score, 3),
            "score_gap": round(gap, 3),
        })

    console.print(f"\nBuilt {len(pairs)} preference pairs")
    if skipped_no_scores:
        console.print(f"  Skipped {skipped_no_scores} (< 2 models scored)")
    if skipped_no_gap:
        console.print(f"  Skipped {skipped_no_gap} (score gap < {min_score_gap})")
    if skipped_no_response:
        console.print(f"  Skipped {skipped_no_response} (missing response text)")

    return pairs


def _to_conversational(rec: dict) -> dict:
    """Convert flat-string ORPO record to conversational format for aitraining VLM ORPO."""
    return {
        "images": rec["images"],
        "prompt": [
            {"role": "user", "content": rec["prompt"]},
        ],
        "chosen": [
            {"role": "assistant", "content": rec["chosen"]},
        ],
        "rejected": [
            {"role": "assistant", "content": rec["rejected"]},
        ],
    }


def _write_jsonl(records: list[dict], path: Path, strip_meta: bool = True) -> None:
    """Write records to JSONL. Optionally strip metadata fields."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            if strip_meta:
                row = _to_conversational(rec)
            else:
                row = dict(rec)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    console.print(f"  Wrote {len(records)} rows → {path}")


@app.command()
def build(
    runs_dir: Path = typer.Option(RUNS_DIR, "--runs-dir"),
    cases_dir: Path = typer.Option(CASES_DIR, "--cases-dir"),
    images_dir: Path = typer.Option(IMAGES_DIR, "--images-dir"),
    output_dir: Path = typer.Option(OUTPUT_DIR, "--output-dir"),
    train_size: int = typer.Option(500, "--train-size"),
    eval_size: int = typer.Option(50, "--eval-size"),
    min_gap: float = typer.Option(0.0, "--min-gap", help="Minimum score gap for inclusion"),
    copy_images: bool = typer.Option(False, "--copy-images", help="Copy images to output dir"),
    seed: int = typer.Option(42, "--seed"),
    exclude: Optional[list[str]] = typer.Option(None, "--exclude", help="Model names to exclude"),
) -> None:
    """Build ORPO preference dataset from IVF-Bench scored runs."""
    exclude_set = set(exclude) if exclude else set(EXCLUDE_MODELS)

    images_out = output_dir / "images" if copy_images else None
    if images_out:
        images_out.mkdir(parents=True, exist_ok=True)

    pairs = build_preference_pairs(
        runs_dir=runs_dir,
        cases_dir=cases_dir,
        images_dir=images_dir,
        exclude_models=exclude_set,
        min_score_gap=min_gap,
        copy_images_to=images_out,
    )

    if not pairs:
        console.print("[red]No preference pairs generated.[/red]")
        raise typer.Exit(1)

    # Deterministic shuffle and split
    rng = random.Random(seed)
    rng.shuffle(pairs)

    total_needed = train_size + eval_size
    if len(pairs) < total_needed:
        console.print(
            f"[yellow]Warning: only {len(pairs)} pairs available, "
            f"need {total_needed} (train={train_size} + eval={eval_size}). "
            f"Using all pairs.[/yellow]"
        )
        eval_actual = min(eval_size, len(pairs) // 10)  # 10% for eval
        train_actual = len(pairs) - eval_actual
    else:
        train_actual = train_size
        eval_actual = eval_size

    train_pairs = pairs[:train_actual]
    eval_pairs = pairs[train_actual:train_actual + eval_actual]

    # Write JSONL files (training format - no metadata)
    _write_jsonl(train_pairs, output_dir / "train.jsonl", strip_meta=True)
    _write_jsonl(eval_pairs, output_dir / "eval.jsonl", strip_meta=True)

    # Also write full metadata version for analysis
    _write_jsonl(train_pairs, output_dir / "train_full.jsonl", strip_meta=False)
    _write_jsonl(eval_pairs, output_dir / "eval_full.jsonl", strip_meta=False)

    # Summary statistics
    console.print()
    table = Table(title="ORPO Dataset Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total pairs", str(len(pairs)))
    table.add_row("Train split", str(len(train_pairs)))
    table.add_row("Eval split", str(len(eval_pairs)))

    # Score gap stats
    gaps = [p["score_gap"] for p in pairs]
    table.add_row("Avg score gap", f"{sum(gaps)/len(gaps):.3f}")
    table.add_row("Min score gap", f"{min(gaps):.3f}")
    table.add_row("Max score gap", f"{max(gaps):.3f}")

    # Model selection frequency
    chosen_counts: dict[str, int] = {}
    rejected_counts: dict[str, int] = {}
    for p in pairs:
        chosen_counts[p["chosen_model"]] = chosen_counts.get(p["chosen_model"], 0) + 1
        rejected_counts[p["rejected_model"]] = rejected_counts.get(p["rejected_model"], 0) + 1

    console.print(table)

    console.print("\n[bold]Chosen model frequency:[/bold]")
    for model, count in sorted(chosen_counts.items(), key=lambda x: -x[1]):
        console.print(f"  {model}: {count} ({count/len(pairs)*100:.1f}%)")

    console.print("\n[bold]Rejected model frequency:[/bold]")
    for model, count in sorted(rejected_counts.items(), key=lambda x: -x[1]):
        console.print(f"  {model}: {count} ({count/len(pairs)*100:.1f}%)")

    console.print(f"\n[bold green]Dataset saved to {output_dir}/[/bold green]")


if __name__ == "__main__":
    app()
