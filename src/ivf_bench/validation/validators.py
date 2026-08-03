"""Validate generated benchmark cases for consistency and completeness."""
from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console

from ivf_bench.config import BenchmarkConfig
from ivf_bench.data.schemas import BenchmarkCase
from ivf_bench.generators.patient_context import SYNTHETIC_FIELDS
from ivf_bench.utils.image import validate_png

console = Console()


def validate_case(case: BenchmarkCase, images_dir: Path) -> list[str]:
    """Validate a single case. Returns list of issues (empty = valid)."""
    issues: list[str] = []

    # Image exists and is valid
    img_path = images_dir / case.image_path
    if not img_path.exists():
        issues.append(f"Image not found: {case.image_path}")
    elif not validate_png(img_path):
        issues.append(f"Invalid PNG: {case.image_path}")

    # Gardner ranges
    if not (0 <= case.gardner.exp <= 6):
        issues.append(f"Expansion out of range: {case.gardner.exp}")
    if not (0 <= case.gardner.icm <= 3):
        issues.append(f"ICM out of range: {case.gardner.icm}")
    if not (0 <= case.gardner.te <= 3):
        issues.append(f"TE out of range: {case.gardner.te}")

    # Patient context plausibility
    ctx = case.patient_context
    if not (18 <= ctx.age <= 50):
        issues.append(f"Age out of range: {ctx.age}")
    if not (15.0 <= ctx.bmi <= 50.0):
        issues.append(f"BMI out of range: {ctx.bmi}")
    if ctx.amh_ng_ml is not None and not (0.01 <= ctx.amh_ng_ml <= 25.0):
        issues.append(f"AMH out of range: {ctx.amh_ng_ml}")
    if not (0.5 <= ctx.fsh_iu_l <= 30.0):
        issues.append(f"FSH out of range: {ctx.fsh_iu_l}")
    if ctx.partner_age < 18:
        issues.append(f"Partner age too low: {ctx.partner_age}")

    # Outcome cascade: LB=1 → HA=1 → SS=1
    # Known anomaly: 480_01.png has LB=1, HA=0 (data entry error in Kromp)
    outcome = case.real_outcome
    if outcome.live_birth and not outcome.clinical_pregnancy:
        if case.image_path != "480_01.png":
            issues.append("Cascade violation: live_birth=True but clinical_pregnancy=False")
    if outcome.clinical_pregnancy and not outcome.biochemical_pregnancy:
        issues.append("Cascade violation: clinical_pregnancy=True but biochemical_pregnancy=False")

    # Synthetic fields list
    if set(ctx.synthetic_fields) != set(SYNTHETIC_FIELDS):
        issues.append(f"Unexpected synthetic_fields: {ctx.synthetic_fields}")

    # Lab data
    if case.lab_data.transfer_day not in (4, 5):
        issues.append(f"Unexpected transfer day: {case.lab_data.transfer_day}")
    if case.lab_data.cocs_retrieved is not None and case.lab_data.cocs_retrieved < 0:
        issues.append(f"COCs negative: {case.lab_data.cocs_retrieved}")
    if case.lab_data.mii_oocytes is not None and case.lab_data.mii_oocytes < 0:
        issues.append(f"MII negative: {case.lab_data.mii_oocytes}")

    return issues


def validate_all_cases(config: BenchmarkConfig) -> dict:
    """Validate all generated cases across both test and validation dirs."""
    images_dir = config.paths.images_dir
    all_issues: dict[str, list[str]] = {}
    total = 0
    valid = 0

    for cases_dir in [config.paths.cases_dir, config.paths.validation_cases_dir]:
        if not cases_dir.exists():
            continue
        for case_file in sorted(cases_dir.glob("IVF-BENCH-*.json")):
            total += 1
            with open(case_file) as f:
                data = json.load(f)
            data.pop("prompt", None)
            case = BenchmarkCase(**data)
            issues = validate_case(case, images_dir)
            if issues:
                all_issues[case.case_id] = issues
            else:
                valid += 1

    return {
        "total": total,
        "valid": valid,
        "invalid": total - valid,
        "issues": all_issues,
    }
