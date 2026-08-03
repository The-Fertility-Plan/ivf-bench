"""Central configuration for IVF-Bench. All tunable parameters in one place."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Paths:
    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    kromp_zip: Path = Path("data/raw/Blastocyst_Dataset.zip")
    images_dir: Path = Path("data/raw/Images")
    clinical_csv: Path = Path("data/raw/Clincial_annotations.csv")
    silver_csv: Path = Path("data/raw/Gardner_train_silver.csv")
    splits_dir: Path = Path("data/splits")
    cases_dir: Path = Path("data/cases")
    validation_cases_dir: Path = Path("data/validation_cases")
    held_out_cases_dir: Path = Path("data/held_out_cases")
    manifest_path: Path = Path("data/manifest.json")


# ---------------------------------------------------------------------------
# Kromp dataset
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class KrompConfig:
    figshare_url: str = "https://ndownloader.figshare.com/files/39348899"
    expected_md5: str = "d19532b4b6bc4792b44738b8930d9ad2"
    expected_image_count: int = 2344
    expected_clinical_count: int = 752


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SplitConfig:
    seed: int = 42
    test_count: int = 550
    validation_count: int = 100
    held_out_count: int = 102


# ---------------------------------------------------------------------------
# Distribution configs (only for fields Kromp doesn't provide)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BMIDistribution:
    mean: float = 25.0
    std: float = 5.0
    min_val: float = 17.0
    max_val: float = 45.0


@dataclass(frozen=True)
class FSHDistribution:
    """Normal, age-dependent. Base ~6 IU/L at age 30, increases with age."""
    base_mean_at_30: float = 6.0
    increase_per_year: float = 0.15
    std: float = 1.5
    min_val: float = 1.0
    max_val: float = 25.0


@dataclass(frozen=True)
class SpermDistribution:
    concentration_mean: float = 50.0
    concentration_std: float = 25.0
    concentration_min: float = 5.0
    concentration_max: float = 200.0
    motility_mean: float = 55.0
    motility_std: float = 15.0
    motility_min: float = 10.0
    motility_max: float = 95.0


@dataclass(frozen=True)
class DiagnosisDistribution:
    categories: tuple[tuple[str, float], ...] = (
        ("tubal_factor", 0.25),
        ("male_factor", 0.25),
        ("unexplained", 0.15),
        ("endometriosis", 0.10),
        ("pcos", 0.12),
        ("diminished_ovarian_reserve", 0.05),
        ("uterine_factor", 0.03),
        ("other", 0.05),
    )


@dataclass(frozen=True)
class ProtocolDistribution:
    categories: tuple[tuple[str, float], ...] = (
        ("antagonist", 0.70),
        ("long_agonist", 0.20),
        ("short_agonist", 0.05),
        ("natural_cycle", 0.03),
        ("other", 0.02),
    )


@dataclass(frozen=True)
class PreviousCyclesDistribution:
    probabilities: tuple[tuple[int, float], ...] = (
        (0, 0.40),
        (1, 0.25),
        (2, 0.15),
        (3, 0.10),
        (4, 0.05),
        (5, 0.05),
    )


@dataclass(frozen=True)
class PartnerAgeConfig:
    # Jelinkova et al. 2026 (DOI: 10.5114/hpr/213967): mean diff 2.4yr, SD 4.3yr, r=0.60
    offset_mean: float = 2.4
    offset_std: float = 4.3
    min_age: float = 20.0
    max_age: float = 65.0


@dataclass(frozen=True)
class SmokingDistribution:
    """Dodge et al. 2015 (n=12,811), Joelsson et al. 2019, Gaskins et al. 2019."""
    categories: tuple[tuple[str, float], ...] = (
        ("never_smoker", 0.65),
        ("former_smoker", 0.25),
        ("current_smoker", 0.10),
    )


@dataclass(frozen=True)
class AlcoholDistribution:
    """Rossi et al. 2011 (n=2,545), Dodge et al. 2015, Joelsson et al. 2019."""
    categories: tuple[tuple[str, float], ...] = (
        ("none_or_rare", 0.40),
        ("light_1_to_3_per_week", 0.35),
        ("moderate_4_to_6_per_week", 0.20),
        ("heavy_7_plus_per_week", 0.05),
    )


@dataclass(frozen=True)
class ExerciseDistribution:
    """Sherwin et al. 2022 (n=229 IPAQ), Dodge et al. 2015, Romanian pilot 2022."""
    categories: tuple[tuple[str, float], ...] = (
        ("sedentary", 0.30),
        ("light_walking_only", 0.25),
        ("moderate_1_to_3x_per_week", 0.25),
        ("active_4_plus_per_week", 0.15),
        ("very_active_daily", 0.05),
    )


@dataclass(frozen=True)
class DietDistribution:
    """Gaskins et al. 2019 EARTH study (n=357), Karayiannis et al. 2018."""
    categories: tuple[tuple[str, float], ...] = (
        ("standard_western", 0.50),
        ("health_conscious", 0.25),
        ("mediterranean", 0.15),
        ("vegetarian_or_vegan", 0.05),
        ("restricted", 0.05),
    )


@dataclass(frozen=True)
class StressDistribution:
    """Holley et al. 2015 (PMC4253550), Klonoff-Cohen & Natarajan 2004."""
    categories: tuple[tuple[str, float], ...] = (
        ("low", 0.15),
        ("mild", 0.30),
        ("moderate", 0.35),
        ("high", 0.15),
        ("severe", 0.05),
    )


@dataclass(frozen=True)
class LifestyleConfig:
    smoking: SmokingDistribution = field(default_factory=SmokingDistribution)
    alcohol: AlcoholDistribution = field(default_factory=AlcoholDistribution)
    exercise: ExerciseDistribution = field(default_factory=ExerciseDistribution)
    diet: DietDistribution = field(default_factory=DietDistribution)
    stress: StressDistribution = field(default_factory=StressDistribution)


@dataclass(frozen=True)
class PatientContextConfig:
    seed: int = 123
    bmi: BMIDistribution = field(default_factory=BMIDistribution)
    fsh: FSHDistribution = field(default_factory=FSHDistribution)
    sperm: SpermDistribution = field(default_factory=SpermDistribution)
    diagnosis: DiagnosisDistribution = field(default_factory=DiagnosisDistribution)
    protocol: ProtocolDistribution = field(default_factory=ProtocolDistribution)
    previous_cycles: PreviousCyclesDistribution = field(default_factory=PreviousCyclesDistribution)
    partner_age: PartnerAgeConfig = field(default_factory=PartnerAgeConfig)
    lifestyle: LifestyleConfig = field(default_factory=LifestyleConfig)


# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BenchmarkConfig:
    paths: Paths = field(default_factory=Paths)
    kromp: KrompConfig = field(default_factory=KrompConfig)
    splits: SplitConfig = field(default_factory=SplitConfig)
    patient_context: PatientContextConfig = field(default_factory=PatientContextConfig)


def load_config(path: Optional[Path] = None) -> BenchmarkConfig:
    """Load config from YAML, falling back to defaults for missing fields."""
    if path is None or not path.exists():
        return BenchmarkConfig()
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return _build_from_dict(BenchmarkConfig, raw)


def _build_from_dict(cls: type, data: dict) -> object:
    """Recursively construct a frozen dataclass from a dict, merging with defaults."""
    if not data:
        return cls()
    kwargs = {}
    default_instance = cls()
    for fld in cls.__dataclass_fields__.values():
        if fld.name not in data:
            continue
        val = data[fld.name]
        if hasattr(fld.type, "__dataclass_fields__") if isinstance(fld.type, type) else False:
            kwargs[fld.name] = _build_from_dict(fld.type, val)
        elif fld.type is Path or (hasattr(fld.type, "__origin__") and fld.type is Path):
            kwargs[fld.name] = Path(val)
        else:
            kwargs[fld.name] = val
    # Merge: start from defaults, override with provided kwargs
    default_dict = {f.name: getattr(default_instance, f.name) for f in cls.__dataclass_fields__.values()}
    default_dict.update(kwargs)
    return cls(**default_dict)
