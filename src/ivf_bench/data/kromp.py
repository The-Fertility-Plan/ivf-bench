"""Download, extract, and parse the Kromp Blastocyst Dataset."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from rich.console import Console

from ivf_bench.config import BenchmarkConfig
from ivf_bench.data.schemas import GardnerGrade, KrompRecord
from ivf_bench.utils.download import download_file

console = Console()
logger = logging.getLogger(__name__)

# German month abbreviations → month number (for Excel date corruption fix)
_GERMAN_MONTHS: dict[str, str] = {
    "Jän": "1", "Jan": "1",
    "Feb": "2",
    "Mär": "3", "Mar": "3", "Mrz": "3",
    "Apr": "4",
    "Mai": "5", "May": "5",
    "Jun": "6",
    "Jul": "7",
    "Aug": "8",
    "Sep": "9",
    "Okt": "10", "Oct": "10",
    "Nov": "11",
    "Dez": "12", "Dec": "12",
}

# Regex: "Mon.DD" or "DD.Mon" where Mon is a German/English month abbreviation
_MONTH_PATTERN = re.compile(
    r"^(\d{1,2})\.(" + "|".join(re.escape(m) for m in _GERMAN_MONTHS) + r")$"
    r"|^(" + "|".join(re.escape(m) for m in _GERMAN_MONTHS) + r")\.(\d{1,2})$"
)


def _fix_excel_date(value: str) -> str | None:
    """Fix Excel date corruption in numeric columns.

    German-locale Excel interprets decimal numbers like 1.26 as dates
    (Jän.26 = January 26). This reverses that corruption.

    Examples:
        "Jän.26" → "1.26"
        "10.Sep" → "10.9"
        "08.Mai" → "8.5"
        "Mai.72" → "5.72"
    """
    m = _MONTH_PATTERN.match(value.strip())
    if not m:
        return None
    if m.group(1) and m.group(2):
        # Format: DD.Mon → DD.month_number
        return f"{m.group(1)}.{_GERMAN_MONTHS[m.group(2)]}"
    elif m.group(3) and m.group(4):
        # Format: Mon.DD → month_number.DD
        return f"{_GERMAN_MONTHS[m.group(3)]}.{m.group(4)}"
    return None


def _parse_int_sum(value: object) -> int | None:
    """Parse an integer that might contain a '+' sum (e.g., '7+1' → 8)."""
    if pd.isna(value):
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        if "+" in s:
            try:
                return sum(int(p.strip()) for p in s.split("+"))
            except ValueError:
                pass
    logger.warning("Could not parse integer value: %r", s)
    return None


def _parse_numeric_or_date(value: object) -> float | None:
    """Parse a cell that might be a number, an Excel-corrupted date, or empty."""
    if pd.isna(value):
        return None
    s = str(value).strip()
    if not s:
        return None
    # Try direct float parse first
    try:
        return float(s)
    except ValueError:
        pass
    # Try Excel date fix
    fixed = _fix_excel_date(s)
    if fixed is not None:
        try:
            return float(fixed)
        except ValueError:
            pass
    logger.warning("Could not parse numeric value: %r", s)
    return None


# ---------------------------------------------------------------------------
# Download & extract
# ---------------------------------------------------------------------------
def download_kromp(config: BenchmarkConfig) -> Path:
    """Download the Kromp ZIP from Figshare if not already present."""
    dest = config.paths.kromp_zip
    dest.parent.mkdir(parents=True, exist_ok=True)
    return download_file(
        url=config.kromp.figshare_url,
        dest=dest,
        expected_md5=config.kromp.expected_md5,
    )


def extract_kromp(config: BenchmarkConfig) -> Path:
    """Extract the ZIP. Idempotent via .extracted marker file."""
    zip_path = config.paths.kromp_zip
    extract_to = config.paths.raw_dir
    marker = config.paths.raw_dir / ".extracted"

    if marker.exists():
        console.print("[dim]Already extracted, skipping.[/dim]")
        return extract_to

    console.print(f"Extracting {zip_path.name}...")
    with ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)

    _verify_extraction(config)
    marker.touch()
    return extract_to


def _verify_extraction(config: BenchmarkConfig) -> None:
    """Check that critical files exist after extraction."""
    for path, desc in [
        (config.paths.clinical_csv, "Clincial_annotations.csv"),
        (config.paths.images_dir, "Images directory"),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Expected {desc} at {path}. "
                f"Check contents of {config.paths.raw_dir}"
            )


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------
def parse_clinical_csv(config: BenchmarkConfig) -> list[KrompRecord]:
    """Parse Clincial_annotations.csv (semicolon-delimited, has header).

    Columns: Image;Fond;d;EXP_silver;ICM_silver;TE_silver;AMH;Age;Endo;COC;MII;SS;HA;LB

    The clinical CSV already includes Gardner grades, so no join is needed.
    AMH and Endo columns may have Excel date corruption that is auto-fixed.
    Validates outcome cascade: LB=1 → HA=1 → SS=1.
    """
    df = pd.read_csv(config.paths.clinical_csv, sep=";", dtype=str, encoding="latin-1")
    df.columns = df.columns.str.strip()

    records: list[KrompRecord] = []
    skipped = 0
    cascade_violations = 0
    amh_fixed = 0
    endo_fixed = 0

    for _, row in df.iterrows():
        fname = str(row["Image"]).strip()

        # Check image exists
        img_path = config.paths.images_dir / fname
        if not img_path.exists():
            logger.warning("Image file missing: %s", fname)
            skipped += 1
            continue

        # Parse Gardner grades
        try:
            exp = int(row["EXP_silver"])
            icm = int(row["ICM_silver"])
            te = int(row["TE_silver"])
        except (ValueError, TypeError):
            logger.warning("Invalid Gardner grades for %s: exp=%s icm=%s te=%s",
                           fname, row.get("EXP_silver"), row.get("ICM_silver"), row.get("TE_silver"))
            skipped += 1
            continue

        # Parse outcomes
        ss = int(row["SS"])
        ha = int(row["HA"])
        lb = int(row["LB"])

        # Validate outcome cascade
        if lb == 1 and ha != 1:
            logger.warning("Cascade violation for %s: LB=1 but HA=%d", fname, ha)
            cascade_violations += 1
        if ha == 1 and ss != 1:
            logger.warning("Cascade violation for %s: HA=1 but SS=%d", fname, ss)
            cascade_violations += 1

        # Parse AMH (may be Excel-corrupted)
        raw_amh = row.get("AMH", "")
        amh = _parse_numeric_or_date(raw_amh)
        if amh is not None and raw_amh and _fix_excel_date(str(raw_amh).strip()):
            amh_fixed += 1

        # Parse Endo (may be Excel-corrupted)
        raw_endo = row.get("Endo", "")
        endo = _parse_numeric_or_date(raw_endo)
        if endo is not None and raw_endo and _fix_excel_date(str(raw_endo).strip()):
            endo_fixed += 1

        # Parse other numeric fields
        age = int(row["Age"])
        transfer_day = int(row["d"])
        coc = _parse_int_sum(row.get("COC", ""))
        mii_val = _parse_int_sum(row.get("MII", ""))
        if mii_val is None:
            logger.warning("Missing MII for %s, skipping", fname)
            skipped += 1
            continue
        mii = mii_val

        records.append(KrompRecord(
            filename=fname,
            gardner=GardnerGrade(exp=exp, icm=icm, te=te),
            age=age,
            amh=amh,
            endometrial_thickness=endo,
            coc=coc,
            mii=mii,
            transfer_day=transfer_day,
            biochemical_pregnancy=bool(ss),
            clinical_pregnancy=bool(ha),
            live_birth=bool(lb),
        ))

    if skipped > 0:
        console.print(f"[yellow]Skipped {skipped} records (missing images or invalid grades)[/yellow]")
    if cascade_violations > 0:
        console.print(f"[yellow]{cascade_violations} outcome cascade violations[/yellow]")
    if amh_fixed > 0:
        console.print(f"[dim]Fixed {amh_fixed} Excel-corrupted AMH values[/dim]")
    if endo_fixed > 0:
        console.print(f"[dim]Fixed {endo_fixed} Excel-corrupted Endo values[/dim]")

    console.print(f"[bold]Parsed {len(records)} clinical records[/bold]")

    if len(records) < 700:
        raise ValueError(
            f"Expected >= 700 clinical records, got {len(records)}. "
            "Check CSV parsing and image paths."
        )

    return records


def build_kromp_records(config: BenchmarkConfig) -> list[KrompRecord]:
    """Build KrompRecords from the clinical CSV. Convenience alias."""
    return parse_clinical_csv(config)
