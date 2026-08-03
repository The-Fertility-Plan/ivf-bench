"""Deterministic data splitting for benchmark construction."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rich.console import Console

from ivf_bench.config import BenchmarkConfig
from ivf_bench.data.schemas import KrompRecord
from ivf_bench.utils.seed import make_rng

console = Console()


@dataclass
class SplitAssignment:
    """Result of splitting the dataset."""
    test: list[str]        # 550 filenames
    validation: list[str]  # 100 filenames
    held_out: list[str]    # remaining (~102) filenames


def create_splits(
    records: list[KrompRecord],
    config: BenchmarkConfig,
) -> SplitAssignment:
    """Split records into test, validation, and held_out sets.

    All records have clinical data. Deterministic shuffle via seed.
    """
    rng = make_rng(config.splits.seed)

    filenames = np.array([r.filename for r in records])
    rng.shuffle(filenames)

    t = config.splits.test_count
    v = config.splits.validation_count

    if len(filenames) < t + v:
        raise ValueError(
            f"Need at least {t + v} records for test+validation, got {len(filenames)}"
        )

    return SplitAssignment(
        test=filenames[:t].tolist(),
        validation=filenames[t : t + v].tolist(),
        held_out=filenames[t + v :].tolist(),
    )


def save_splits(splits: SplitAssignment, output_dir: Path) -> Path:
    """Save split assignments to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "test": sorted(splits.test),
        "validation": sorted(splits.validation),
        "held_out": sorted(splits.held_out),
        "counts": {
            "test": len(splits.test),
            "validation": len(splits.validation),
            "held_out": len(splits.held_out),
            "total": len(splits.test) + len(splits.validation) + len(splits.held_out),
        },
    }
    out_path = output_dir / "splits.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    return out_path


def load_splits(splits_dir: Path) -> SplitAssignment:
    """Load previously saved splits."""
    with open(splits_dir / "splits.json") as f:
        data = json.load(f)
    return SplitAssignment(
        test=data["test"],
        validation=data["validation"],
        held_out=data["held_out"],
    )
