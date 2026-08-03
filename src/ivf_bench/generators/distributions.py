"""Statistical distribution samplers for synthetic patient context fields.

Only samples fields that Kromp doesn't provide: BMI, FSH, diagnosis,
protocol, previous cycles, partner age, sperm parameters, lifestyle.
"""
from __future__ import annotations

import numpy as np

from ivf_bench.config import (
    BMIDistribution,
    DiagnosisDistribution,
    FSHDistribution,
    LifestyleConfig,
    PartnerAgeConfig,
    PreviousCyclesDistribution,
    ProtocolDistribution,
    SpermDistribution,
)


def _sample_truncated_normal(
    rng: np.random.Generator,
    mean: float,
    std: float,
    min_val: float,
    max_val: float,
) -> float:
    """Sample from truncated normal via rejection sampling."""
    for _ in range(1000):
        val = rng.normal(mean, std)
        if min_val <= val <= max_val:
            return round(val, 1)
    return round(float(np.clip(rng.normal(mean, std), min_val, max_val)), 1)


def _sample_categorical(
    rng: np.random.Generator,
    categories: tuple[tuple, ...],
) -> object:
    """Sample from a categorical distribution."""
    labels = [c[0] for c in categories]
    probs = np.array([c[1] for c in categories], dtype=float)
    probs /= probs.sum()
    return rng.choice(labels, p=probs)


# ---------------------------------------------------------------------------
# Individual samplers
# ---------------------------------------------------------------------------
def sample_bmi(rng: np.random.Generator, cfg: BMIDistribution) -> float:
    return _sample_truncated_normal(rng, cfg.mean, cfg.std, cfg.min_val, cfg.max_val)


def sample_fsh(rng: np.random.Generator, cfg: FSHDistribution, age: int) -> float:
    """Age-dependent FSH. Increases ~0.15 IU/L per year from age 30."""
    age_offset = max(age - 30, 0)
    adjusted_mean = cfg.base_mean_at_30 + age_offset * cfg.increase_per_year
    return _sample_truncated_normal(rng, adjusted_mean, cfg.std, cfg.min_val, cfg.max_val)


def sample_diagnosis(rng: np.random.Generator, cfg: DiagnosisDistribution) -> str:
    return str(_sample_categorical(rng, cfg.categories))


def sample_protocol(rng: np.random.Generator, cfg: ProtocolDistribution) -> str:
    return str(_sample_categorical(rng, cfg.categories))


def sample_previous_cycles(
    rng: np.random.Generator, cfg: PreviousCyclesDistribution
) -> int:
    return int(_sample_categorical(rng, cfg.probabilities))


def sample_previous_outcomes(rng: np.random.Generator, n_cycles: int) -> list[str]:
    """Generate plausible outcomes for previous IVF cycles."""
    if n_cycles == 0:
        return []
    outcomes = ["no_pregnancy", "biochemical_pregnancy", "clinical_miscarriage",
                "ectopic", "live_birth"]
    probs = np.array([0.45, 0.20, 0.15, 0.05, 0.15])
    probs /= probs.sum()
    return rng.choice(outcomes, size=n_cycles, p=probs).tolist()


def sample_partner_age(
    rng: np.random.Generator, cfg: PartnerAgeConfig, female_age: int
) -> int:
    """Partner age correlated with female age (+2-5 years typical)."""
    offset = _sample_truncated_normal(rng, cfg.offset_mean, cfg.offset_std, -5.0, 20.0)
    partner = female_age + offset
    return int(round(float(np.clip(partner, cfg.min_age, cfg.max_age))))


def sample_sperm_concentration(
    rng: np.random.Generator, cfg: SpermDistribution
) -> float:
    return _sample_truncated_normal(
        rng, cfg.concentration_mean, cfg.concentration_std,
        cfg.concentration_min, cfg.concentration_max,
    )


def sample_sperm_motility(
    rng: np.random.Generator, cfg: SpermDistribution
) -> float:
    return _sample_truncated_normal(
        rng, cfg.motility_mean, cfg.motility_std,
        cfg.motility_min, cfg.motility_max,
    )


# Human-readable display labels for lifestyle dimensions
_SMOKING_LABELS = {
    "never_smoker": "never smoker",
    "former_smoker": "former smoker",
    "current_smoker": "current smoker",
}
_ALCOHOL_LABELS = {
    "none_or_rare": "no/rare alcohol",
    "light_1_to_3_per_week": "light drinker (1-3/wk)",
    "moderate_4_to_6_per_week": "moderate drinker (4-6/wk)",
    "heavy_7_plus_per_week": "heavy drinker (7+/wk)",
}
_EXERCISE_LABELS = {
    "sedentary": "sedentary",
    "light_walking_only": "light activity (walking only)",
    "moderate_1_to_3x_per_week": "moderate exercise (1-3x/wk)",
    "active_4_plus_per_week": "active (4+x/wk)",
    "very_active_daily": "very active (daily exercise)",
}
_DIET_LABELS = {
    "standard_western": "standard diet",
    "health_conscious": "health-conscious diet",
    "mediterranean": "Mediterranean diet",
    "vegetarian_or_vegan": "vegetarian/vegan",
    "restricted": "restricted diet",
}
_STRESS_LABELS = {
    "low": "low stress",
    "mild": "mild stress",
    "moderate": "moderate stress",
    "high": "high stress",
    "severe": "severe stress",
}


def sample_lifestyle(rng: np.random.Generator, cfg: LifestyleConfig) -> str:
    """Sample each lifestyle dimension independently and compose a summary string."""
    smoking = str(_sample_categorical(rng, cfg.smoking.categories))
    alcohol = str(_sample_categorical(rng, cfg.alcohol.categories))
    exercise = str(_sample_categorical(rng, cfg.exercise.categories))
    diet = str(_sample_categorical(rng, cfg.diet.categories))
    stress = str(_sample_categorical(rng, cfg.stress.categories))

    parts = [
        _SMOKING_LABELS.get(smoking, smoking),
        _ALCOHOL_LABELS.get(alcohol, alcohol),
        _EXERCISE_LABELS.get(exercise, exercise),
        _DIET_LABELS.get(diet, diet),
        _STRESS_LABELS.get(stress, stress),
    ]
    return "; ".join(parts)
