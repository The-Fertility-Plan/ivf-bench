"""Leaderboard generation for IVF-Bench evaluation runs."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.table import Table

from ivf_bench.data.schemas import LeaderboardEntry, RunSummary
from ivf_bench.eval.metrics import compute_outcome_metrics

console = Console()

RUBRIC_KEYS = [
    "morphological_accuracy",
    "clinical_integration",
    "reasoning_coherence",
    "guideline_alignment",
    "actionability",
]


def summarize_run(run_dir: Path, cases_dir: Path, scores_dirname: str = "scores") -> RunSummary:
    """Compute summary statistics for a single model run.

    Only includes responses/scores whose case_id exists in `cases_dir`, so the
    same runs directory can drive separate test/validation/held_out leaderboards.
    """
    meta = json.loads((run_dir / "run_meta.json").read_text())
    responses_dir = run_dir / "responses"
    scores_dir = run_dir / scores_dirname

    split_case_ids = {p.stem for p in cases_dir.glob("IVF-BENCH-*.json")}

    # Aggregate response stats
    total_cost = 0.0
    total_tokens = 0
    latencies = []
    completed = 0
    failed = 0

    for rf in sorted(responses_dir.glob("IVF-BENCH-*.json")):
        if rf.stem not in split_case_ids:
            continue
        data = json.loads(rf.read_text())
        if data.get("error") or not data.get("response_text"):
            failed += 1
        else:
            completed += 1
            total_cost += data.get("cost_usd", 0.0)
            total_tokens += data.get("total_tokens", 0)
            latencies.append(data.get("latency_ms", 0.0))

    # Aggregate rubric scores
    mean_scores: dict[str, float] = {}
    rubric_scores: dict[str, list[float]] = {k: [] for k in RUBRIC_KEYS}

    if scores_dir.exists():
        for sf in sorted(scores_dir.glob("IVF-BENCH-*.json")):
            if sf.stem not in split_case_ids:
                continue
            data = json.loads(sf.read_text())
            for r in data.get("rubrics", []):
                if r["rubric"] in rubric_scores and r["score"] > 0:
                    rubric_scores[r["rubric"]].append(r["score"])

    for key in RUBRIC_KEYS:
        if rubric_scores[key]:
            mean_scores[key] = round(float(np.mean(rubric_scores[key])), 3)
        else:
            mean_scores[key] = 0.0

    overall = round(float(np.mean(list(mean_scores.values()))), 3) if mean_scores else 0.0

    # Outcome metrics
    brier = None
    auroc = None
    if scores_dir.exists():
        metrics = compute_outcome_metrics(scores_dir, cases_dir)
        brier = metrics.get("brier_score_cp")
        auroc = metrics.get("auroc_cp")

    # Add judge costs
    judge_cost = 0.0
    if scores_dir.exists():
        for sf in sorted(scores_dir.glob("IVF-BENCH-*.json")):
            if sf.stem not in split_case_ids:
                continue
            data = json.loads(sf.read_text())
            judge_cost += data.get("judge_cost_usd", 0.0)

    return RunSummary(
        model=meta["model"],
        model_slug=meta["model_slug"],
        total_cases=meta["total_cases"],
        completed=completed,
        failed=failed,
        total_cost_usd=round(total_cost + judge_cost, 4),
        total_tokens=total_tokens,
        avg_latency_ms=round(float(np.mean(latencies)), 1) if latencies else 0.0,
        mean_scores=mean_scores,
        overall_score=overall,
        brier_score=brier,
        auroc=auroc,
    )


def build_leaderboard(runs_dir: Path, cases_dir: Path, scores_dirname: str = "scores") -> list[LeaderboardEntry]:
    """Build leaderboard from all model runs."""
    entries: list[LeaderboardEntry] = []

    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir() or not (run_dir / "run_meta.json").exists():
            continue

        # Ablation arms are separate experiments, not leaderboard entries.
        meta = json.loads((run_dir / "run_meta.json").read_text())
        if not meta.get("include_image", True) or not meta.get("include_grade", True):
            continue

        summary = summarize_run(run_dir, cases_dir, scores_dirname)

        # A run with no usable scores for this split is not a leaderboard row.
        if not any(v > 0 for v in summary.mean_scores.values()):
            continue

        entries.append(LeaderboardEntry(
            rank=0,
            model=summary.model,
            overall_score=summary.overall_score,
            morphological_accuracy=summary.mean_scores.get("morphological_accuracy", 0.0),
            clinical_integration=summary.mean_scores.get("clinical_integration", 0.0),
            reasoning_coherence=summary.mean_scores.get("reasoning_coherence", 0.0),
            guideline_alignment=summary.mean_scores.get("guideline_alignment", 0.0),
            actionability=summary.mean_scores.get("actionability", 0.0),
            outcome_consistency=None,
            brier_score=summary.brier_score,
            auroc=summary.auroc,
            total_cost_usd=summary.total_cost_usd,
            avg_latency_ms=summary.avg_latency_ms,
            cases_completed=summary.completed,
        ))

    # Rank by overall score descending
    entries.sort(key=lambda e: e.overall_score, reverse=True)
    for i, entry in enumerate(entries):
        entry.rank = i + 1

    return entries


def save_leaderboard(entries: list[LeaderboardEntry], output_dir: Path, suffix: str = "") -> Path:
    """Save leaderboard as JSON and Markdown."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    json_path = output_dir / f"leaderboard{suffix}.json"
    json_path.write_text(json.dumps(
        [e.model_dump() for e in entries], indent=2
    ))

    # Markdown
    md_path = output_dir / f"leaderboard{suffix}.md"
    lines = [
        "# IVF-Bench Leaderboard",
        "",
        "| Rank | Model | Overall | Morph | Clinical | Reasoning | Guideline | Action "
        "| Brier | AUROC | Cost | Latency | Cases |",
        "|------|-------|---------|-------|----------|-----------|-----------|-------- "
        "|-------|-------|------|---------|-------|",
    ]

    for e in entries:
        brier_str = f"{e.brier_score:.3f}" if e.brier_score is not None else "—"
        auroc_str = f"{e.auroc:.3f}" if e.auroc is not None else "—"
        lines.append(
            f"| {e.rank} | {e.model} | {e.overall_score:.2f} "
            f"| {e.morphological_accuracy:.2f} | {e.clinical_integration:.2f} "
            f"| {e.reasoning_coherence:.2f} | {e.guideline_alignment:.2f} "
            f"| {e.actionability:.2f} | {brier_str} | {auroc_str} "
            f"| ${e.total_cost_usd:.2f} | {e.avg_latency_ms:.0f}ms | {e.cases_completed} |"
        )

    lines.extend(["", "Generated by IVF-Bench", ""])
    md_path.write_text("\n".join(lines))

    return json_path


def print_leaderboard(entries: list[LeaderboardEntry]) -> None:
    """Print leaderboard to console as a rich table."""
    table = Table(title="IVF-Bench Leaderboard")
    table.add_column("#", style="bold")
    table.add_column("Model", style="cyan")
    table.add_column("Overall", style="bold green")
    table.add_column("Morph")
    table.add_column("Clinical")
    table.add_column("Reason")
    table.add_column("Guide")
    table.add_column("Action")
    table.add_column("Brier", style="yellow")
    table.add_column("AUROC", style="yellow")
    table.add_column("Cost", style="red")
    table.add_column("Latency")
    table.add_column("Cases")

    for e in entries:
        brier = f"{e.brier_score:.3f}" if e.brier_score is not None else "—"
        auroc = f"{e.auroc:.3f}" if e.auroc is not None else "—"
        table.add_row(
            str(e.rank),
            e.model,
            f"{e.overall_score:.2f}",
            f"{e.morphological_accuracy:.2f}",
            f"{e.clinical_integration:.2f}",
            f"{e.reasoning_coherence:.2f}",
            f"{e.guideline_alignment:.2f}",
            f"{e.actionability:.2f}",
            brier,
            auroc,
            f"${e.total_cost_usd:.2f}",
            f"{e.avg_latency_ms:.0f}ms",
            str(e.cases_completed),
        )

    console.print(table)
