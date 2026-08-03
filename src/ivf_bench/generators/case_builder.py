"""Assemble complete benchmark cases from Kromp data + synthetic context."""
from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.progress import track

from ivf_bench.config import BenchmarkConfig
from ivf_bench.data.schemas import (
    BenchmarkCase,
    KrompRecord,
    LabData,
    RealClinical,
    RealOutcome,
)
from ivf_bench.data.splits import SplitAssignment
from ivf_bench.generators.patient_context import generate_patient_context
from ivf_bench.generators.prompt_template import render_prompt

console = Console()


def build_case(
    idx: int,
    record: KrompRecord,
    split: str,
    config: BenchmarkConfig,
) -> BenchmarkCase:
    """Build a single benchmark case."""
    case_id = f"IVF-BENCH-{idx:04d}"

    lab_data = LabData(
        cocs_retrieved=record.coc,
        mii_oocytes=record.mii,
        transfer_day=record.transfer_day,
    )

    real_clinical = RealClinical(
        age=record.age,
        amh=record.amh,
        endometrial_thickness=record.endometrial_thickness,
        cocs_retrieved=record.coc,
        mii_oocytes=record.mii,
        transfer_day=record.transfer_day,
    )

    real_outcome = RealOutcome(
        biochemical_pregnancy=record.biochemical_pregnancy,
        clinical_pregnancy=record.clinical_pregnancy,
        live_birth=record.live_birth,
    )

    patient_ctx = generate_patient_context(
        case_id=case_id,
        record=record,
        config=config.patient_context,
    )

    return BenchmarkCase(
        case_id=case_id,
        image_path=record.filename,
        gardner=record.gardner,
        lab_data=lab_data,
        patient_context=patient_ctx,
        real_clinical=real_clinical,
        real_outcome=real_outcome,
        split=split,
    )


def build_all_cases(
    records: list[KrompRecord],
    splits: SplitAssignment,
    config: BenchmarkConfig,
    include_held_out: bool = False,
) -> list[BenchmarkCase]:
    """Build benchmark cases for test + validation (+ optionally held_out)."""
    record_map = {r.filename: r for r in records}
    cases: list[BenchmarkCase] = []
    idx = 1

    # Test set
    for fname in sorted(splits.test):
        cases.append(build_case(idx, record_map[fname], "test", config))
        idx += 1

    # Validation set
    for fname in sorted(splits.validation):
        cases.append(build_case(idx, record_map[fname], "validation", config))
        idx += 1

    # Held out (only if explicitly requested)
    if include_held_out:
        for fname in sorted(splits.held_out):
            cases.append(build_case(idx, record_map[fname], "held_out", config))
            idx += 1

    return cases


def save_cases(cases: list[BenchmarkCase], config: BenchmarkConfig) -> Path:
    """Save cases as individual JSON files + a manifest."""
    # Save test cases and validation cases to separate directories
    for case in track(cases, description="Saving cases..."):
        if case.split == "validation":
            out_dir = config.paths.validation_cases_dir
        elif case.split == "held_out":
            out_dir = config.paths.held_out_cases_dir
        else:
            out_dir = config.paths.cases_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        case_dict = case.model_dump(mode="json")
        case_dict["prompt"] = render_prompt(case)

        case_path = out_dir / f"{case.case_id}.json"
        with open(case_path, "w") as f:
            json.dump(case_dict, f, indent=2)

    # Manifest
    manifest = {
        "version": "0.1.0",
        "total_cases": len(cases),
        "test_count": sum(1 for c in cases if c.split == "test"),
        "validation_count": sum(1 for c in cases if c.split == "validation"),
        "held_out_count": sum(1 for c in cases if c.split == "held_out"),
        "cases": [
            {
                "case_id": c.case_id,
                "image_path": c.image_path,
                "split": c.split,
                "gardner": c.gardner.display,
            }
            for c in cases
        ],
    }
    config.paths.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config.paths.manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    console.print(f"[bold green]Saved {len(cases)} cases + manifest[/bold green]")
    return config.paths.cases_dir
