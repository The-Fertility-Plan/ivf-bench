"""Generate synthetic patient context, augmenting real Kromp clinical data."""
from __future__ import annotations

from ivf_bench.config import PatientContextConfig
from ivf_bench.data.schemas import KrompRecord, PatientContext
from ivf_bench.generators.distributions import (
    sample_bmi,
    sample_fsh,
    sample_diagnosis,
    sample_lifestyle,
    sample_partner_age,
    sample_previous_cycles,
    sample_previous_outcomes,
    sample_protocol,
    sample_sperm_concentration,
    sample_sperm_motility,
)
from ivf_bench.utils.seed import case_seed, make_rng

# Fields that are always synthetic (Kromp doesn't provide these)
SYNTHETIC_FIELDS = [
    "bmi",
    "fsh_iu_l",
    "diagnosis",
    "protocol",
    "previous_cycles",
    "previous_outcomes",
    "partner_age",
    "sperm_concentration_million_ml",
    "sperm_motility_pct",
    "lifestyle_notes",
]


def generate_patient_context(
    case_id: str,
    record: KrompRecord,
    config: PatientContextConfig,
) -> PatientContext:
    """Build a patient context combining real Kromp data with synthetic fields.

    Real (from Kromp): age, AMH, endometrial thickness
    Synthetic: everything else
    """
    seed = case_seed(config.seed, case_id)
    rng = make_rng(seed)

    prev_cycles = sample_previous_cycles(rng, config.previous_cycles)

    return PatientContext(
        # Real fields
        age=record.age,
        amh_ng_ml=record.amh,
        endometrial_thickness_mm=record.endometrial_thickness,
        # Synthetic fields
        bmi=sample_bmi(rng, config.bmi),
        fsh_iu_l=sample_fsh(rng, config.fsh, record.age),
        diagnosis=sample_diagnosis(rng, config.diagnosis),
        protocol=sample_protocol(rng, config.protocol),
        previous_cycles=prev_cycles,
        previous_outcomes=sample_previous_outcomes(rng, prev_cycles),
        partner_age=sample_partner_age(rng, config.partner_age, record.age),
        sperm_concentration_million_ml=sample_sperm_concentration(rng, config.sperm),
        sperm_motility_pct=sample_sperm_motility(rng, config.sperm),
        lifestyle_notes=sample_lifestyle(rng, config.lifestyle),
        synthetic_fields=SYNTHETIC_FIELDS,
    )
