"""IVF-Bench CLI: download, split, generate, validate, run, score, leaderboard."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from ivf_bench.config import load_config

load_dotenv()

console = Console()
app = typer.Typer(
    name="ivf-bench",
    help="IVF-Bench: Open benchmark for IVF clinical reasoning VLMs",
    add_completion=False,
)


@app.command()
def download(
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
) -> None:
    """Download and extract the Kromp Blastocyst Dataset from Figshare."""
    config = load_config(config_path)
    from ivf_bench.data.kromp import download_kromp, extract_kromp

    console.print("[bold]Step 1/2: Downloading Kromp dataset...[/bold]")
    download_kromp(config)

    console.print("[bold]Step 2/2: Extracting...[/bold]")
    extract_kromp(config)

    console.print("[bold green]Download complete.[/bold green]")


@app.command()
def split(
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
) -> None:
    """Parse Kromp data and create deterministic splits."""
    config = load_config(config_path)
    from ivf_bench.data.kromp import build_kromp_records
    from ivf_bench.data.splits import create_splits, save_splits

    console.print("[bold]Parsing Kromp dataset...[/bold]")
    records = build_kromp_records(config)
    console.print(f"  {len(records)} clinical records")

    console.print("[bold]Creating splits...[/bold]")
    splits = create_splits(records, config)

    out = save_splits(splits, config.paths.splits_dir)
    console.print(f"[bold green]Splits saved to {out}[/bold green]")

    table = Table(title="Data Splits")
    table.add_column("Split", style="cyan")
    table.add_column("Count", style="green")
    table.add_row("Test", str(len(splits.test)))
    table.add_row("Validation", str(len(splits.validation)))
    table.add_row("Held Out", str(len(splits.held_out)))
    table.add_row("Total", str(len(splits.test) + len(splits.validation) + len(splits.held_out)))
    console.print(table)


@app.command()
def generate(
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
    include_held_out: bool = typer.Option(False, "--include-held-out"),
) -> None:
    """Generate benchmark cases from Kromp data + synthetic context."""
    config = load_config(config_path)
    from ivf_bench.data.kromp import build_kromp_records
    from ivf_bench.data.splits import load_splits
    from ivf_bench.generators.case_builder import build_all_cases, save_cases

    console.print("[bold]Loading records and splits...[/bold]")
    records = build_kromp_records(config)
    splits = load_splits(config.paths.splits_dir)

    console.print("[bold]Building benchmark cases...[/bold]")
    cases = build_all_cases(records, splits, config, include_held_out=include_held_out)

    test_n = sum(1 for c in cases if c.split == "test")
    val_n = sum(1 for c in cases if c.split == "validation")
    held_n = sum(1 for c in cases if c.split == "held_out")
    console.print(f"  Built {len(cases)} cases (test={test_n}, validation={val_n}, held_out={held_n})")

    save_cases(cases, config)


@app.command()
def validate(
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
) -> None:
    """Validate all generated benchmark cases."""
    config = load_config(config_path)
    from ivf_bench.validation.validators import validate_all_cases

    console.print("[bold]Validating cases...[/bold]")
    result = validate_all_cases(config)

    console.print(
        f"\nTotal: {result['total']}, "
        f"Valid: {result['valid']}, "
        f"Invalid: {result['invalid']}"
    )

    if result["issues"]:
        console.print("\n[bold red]Issues found:[/bold red]")
        for case_id, issues in sorted(result["issues"].items()):
            for issue in issues:
                console.print(f"  {case_id}: {issue}")
    else:
        console.print("[bold green]All cases valid.[/bold green]")


# ---------------------------------------------------------------------------
# Phase 2: Evaluation commands
# ---------------------------------------------------------------------------
RUNS_DIR = Path("data/runs")


@app.command()
def run(
    model: str = typer.Argument(..., help="Model ID, e.g. qwen/qwen3.5-397b-a17b or gpt-5.4-2026-03-05"),
    backend: str = typer.Option(
        "openrouter", "--backend", "-b",
        help="Inference backend: openrouter, openai, or gemini",
    ),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
    split: str = typer.Option("test", "--split", "-s"),
    concurrency: int = typer.Option(5, "--concurrency", "-j"),
    limit: Optional[int] = typer.Option(None, "--limit", "-n", help="Max cases to run"),
    input_price: float = typer.Option(0.0, "--input-price", help="$/M input tokens"),
    output_price: float = typer.Option(0.0, "--output-price", help="$/M output tokens"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-run even if responses exist"),
    no_image: bool = typer.Option(False, "--no-image", help="Text-only ablation: drop the embryo image"),
    no_grade: bool = typer.Option(False, "--no-grade", help="Ablation: withhold the Gardner grade"),
    run_suffix: str = typer.Option("", "--run-suffix", help="Suffix for the run directory, e.g. -noimg"),
) -> None:
    """Run VLM inference on benchmark cases via OpenRouter or OpenAI."""
    if backend == "bedrock":
        api_key = ""  # Bedrock uses AWS credentials from env
    elif backend == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            console.print("[bold red]OPENAI_API_KEY not set in .env[/bold red]")
            raise typer.Exit(1)
    elif backend == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            console.print("[bold red]GEMINI_API_KEY not set in .env[/bold red]")
            raise typer.Exit(1)
    else:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            console.print("[bold red]OPENROUTER_API_KEY not set in .env[/bold red]")
            raise typer.Exit(1)

    config = load_config(config_path)
    if split == "validation":
        cases_dir = config.paths.validation_cases_dir
    elif split == "held_out":
        cases_dir = config.paths.held_out_cases_dir
    else:
        cases_dir = config.paths.cases_dir

    from ivf_bench.eval.runner import run_model

    run_dir = asyncio.run(run_model(
        api_key=api_key,
        model=model,
        cases_dir=cases_dir,
        images_dir=config.paths.images_dir,
        runs_dir=RUNS_DIR,
        split=split,
        concurrency=concurrency,
        limit=limit,
        input_price=input_price,
        output_price=output_price,
        backend=backend,
        force_rerun=force,
        include_image=not no_image,
        include_grade=not no_grade,
        run_suffix=run_suffix,
    ))
    console.print(f"\nRun saved to: [bold]{run_dir}[/bold]")


def _get_judge_api_key(backend: str) -> str:
    """Get the appropriate API key for the judge backend."""
    if backend == "openai":
        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            console.print("[bold red]OPENAI_API_KEY not set in .env[/bold red]")
            raise typer.Exit(1)
        return key
    elif backend == "bedrock":
        return ""  # Bedrock uses AWS credentials from env/config
    else:
        key = os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            console.print("[bold red]OPENROUTER_API_KEY not set in .env[/bold red]")
            raise typer.Exit(1)
        return key


@app.command()
def score(
    model: str = typer.Argument(..., help="OpenRouter model ID that was run"),
    judge: str = typer.Option(
        "us.anthropic.claude-sonnet-4-20250514-v1:0", "--judge",
        help="Judge model ID",
    ),
    backend: str = typer.Option(
        "bedrock", "--backend", "-b",
        help="Judge backend: openrouter, bedrock, or openai",
    ),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
    concurrency: int = typer.Option(5, "--concurrency", "-j"),
    limit: Optional[int] = typer.Option(None, "--limit", "-n"),
    judge_input_price: float = typer.Option(3.0, "--judge-input-price", help="Judge $/M input tokens"),
    judge_output_price: float = typer.Option(15.0, "--judge-output-price", help="Judge $/M output tokens"),
    bedrock_region: str = typer.Option("us-east-1", "--bedrock-region"),
    split: str = typer.Option("test", "--split", "-s", help="test / validation / held_out (selects cases_dir)"),
    max_cost: Optional[float] = typer.Option(None, "--max-cost", help="Abort once judge spend exceeds this many USD"),
    scores_dir: str = typer.Option("scores", "--scores-dir", help="Subdirectory to write scores into (use a second name for a second judge)"),
    with_image: bool = typer.Option(False, "--with-image", help="Show the judge the embryo image, so morphology can be scored against it"),
    image_as_reference: bool = typer.Option(False, "--image-as-reference", help="Attach the image to the judge even in an arm where the model was denied it, telling the judge so. Keeps judge eyesight constant across ablation arms"),
) -> None:
    """Score model responses using LLM-as-judge (Health-SCORE rubrics)."""
    api_key = _get_judge_api_key(backend)
    config = load_config(config_path)
    from ivf_bench.eval.runner import _slug
    from ivf_bench.eval.judge import score_run

    run_dir = RUNS_DIR / _slug(model)
    if not run_dir.exists():
        console.print(f"[bold red]No run found for {model} at {run_dir}[/bold red]")
        raise typer.Exit(1)

    if split == "validation":
        cases_dir = config.paths.validation_cases_dir
    elif split == "held_out":
        cases_dir = config.paths.held_out_cases_dir
    else:
        cases_dir = config.paths.cases_dir

    asyncio.run(score_run(
        api_key=api_key,
        judge_model=judge,
        run_dir=run_dir,
        cases_dir=cases_dir,
        concurrency=concurrency,
        limit=limit,
        input_price=judge_input_price,
        output_price=judge_output_price,
        backend=backend,
        bedrock_region=bedrock_region,
        max_cost_usd=max_cost,
        scores_dirname=scores_dir,
        images_dir=config.paths.images_dir if (with_image or image_as_reference) else None,
        image_as_reference=image_as_reference,
    ))


@app.command(name="score-all")
def score_all(
    judge: str = typer.Option(
        "us.anthropic.claude-sonnet-4-20250514-v1:0", "--judge",
        help="Judge model ID",
    ),
    backend: str = typer.Option(
        "bedrock", "--backend", "-b",
        help="Judge backend: openrouter, bedrock, or openai",
    ),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
    concurrency: int = typer.Option(5, "--concurrency", "-j"),
    limit: Optional[int] = typer.Option(None, "--limit", "-n"),
    judge_input_price: float = typer.Option(3.0, "--judge-input-price", help="Judge $/M input tokens"),
    judge_output_price: float = typer.Option(15.0, "--judge-output-price", help="Judge $/M output tokens"),
    bedrock_region: str = typer.Option("us-east-1", "--bedrock-region"),
    max_cost: Optional[float] = typer.Option(None, "--max-cost", help="Abort once judge spend exceeds this many USD (per run)"),
) -> None:
    """Score ALL model runs with LLM-as-judge."""
    api_key = _get_judge_api_key(backend)
    config = load_config(config_path)

    if not RUNS_DIR.exists():
        console.print("[bold red]No runs found.[/bold red]")
        raise typer.Exit(1)

    from ivf_bench.eval.judge import score_run
    import json

    run_dirs = []
    for d in sorted(RUNS_DIR.iterdir()):
        meta_file = d / "run_meta.json"
        if d.is_dir() and meta_file.exists():
            meta = json.loads(meta_file.read_text())
            run_dirs.append((d, meta["model"]))

    if not run_dirs:
        console.print("[yellow]No model runs found.[/yellow]")
        raise typer.Exit(1)

    console.print(f"[bold]Scoring {len(run_dirs)} model runs with {judge} via {backend}[/bold]\n")

    for run_dir, model_name in run_dirs:
        console.print(f"\n{'='*60}")
        console.print(f"[bold cyan]{model_name}[/bold cyan]")
        console.print(f"{'='*60}")
        asyncio.run(score_run(
            api_key=api_key,
            judge_model=judge,
            run_dir=run_dir,
            cases_dir=config.paths.cases_dir,
            concurrency=concurrency,
            limit=limit,
            input_price=judge_input_price,
            output_price=judge_output_price,
            backend=backend,
            bedrock_region=bedrock_region,
            max_cost_usd=max_cost,
        ))

    console.print("\n[bold green]All runs scored.[/bold green]")


@app.command(name="leaderboard")
def leaderboard_cmd(
    config_path: Optional[Path] = typer.Option(None, "--config", "-c"),
    scores_dir: str = typer.Option("scores", "--scores-dir", help="Which scores subdirectory to build from"),
    split: str = typer.Option("test", "--split", "-s", help="test / validation / held_out"),
) -> None:
    """Generate and display the leaderboard from all scored runs."""
    config = load_config(config_path)
    from ivf_bench.eval.leaderboard import (
        build_leaderboard,
        print_leaderboard,
        save_leaderboard,
    )

    if not RUNS_DIR.exists():
        console.print("[bold red]No runs found. Run `ivf-bench run` first.[/bold red]")
        raise typer.Exit(1)

    if split == "validation":
        cases_dir = config.paths.validation_cases_dir
        suffix = "_validation"
    elif split == "held_out":
        cases_dir = config.paths.held_out_cases_dir
        suffix = "_held_out"
    else:
        cases_dir = config.paths.cases_dir
        suffix = ""

    entries = build_leaderboard(RUNS_DIR, cases_dir, scores_dir)
    if not entries:
        console.print("[yellow]No model runs found.[/yellow]")
        raise typer.Exit(1)

    print_leaderboard(entries)
    out = save_leaderboard(entries, RUNS_DIR, suffix=suffix)
    console.print(f"\nSaved to: [bold]{out}[/bold]")
