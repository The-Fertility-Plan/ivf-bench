"""Prompt template rendering for benchmark cases."""
from __future__ import annotations

from ivf_bench.data.schemas import BenchmarkCase

EXPANSION_DESCRIPTIONS = {
    0: "early blastocyst, blastocoel less than half the embryo",
    1: "blastocyst, blastocoel at least half the embryo",
    2: "full blastocyst, blastocoel fills the embryo",
    3: "expanded blastocyst, increased size with thinning zona",
    4: "hatching blastocyst, herniating through zona",
    5: "fully hatched blastocyst",
}

ICM_DESCRIPTIONS = {
    "A": "tightly packed, many cells",
    "B": "loosely grouped, several cells",
    "C": "very few cells",
    "D": "degenerative or absent",
}

TE_DESCRIPTIONS = {
    "A": "many cells forming a cohesive epithelium",
    "B": "few cells forming a loose epithelium",
    "C": "very few large cells",
    "D": "degenerative or absent",
}

PROMPT_TEMPLATE = """\
You are an expert reproductive medicine specialist reviewing an IVF case.

## Embryo Image
[IMAGE ATTACHED]

## Laboratory Data
- Gardner Score: {gardner_display} (Expansion: {exp_desc}, ICM: {icm_desc}, TE: {te_desc})
- COCs retrieved: {cocs}
- MII oocytes: {mii}
- Transfer day: Day {transfer_day}

## Patient Profile
- Female age: {age} | BMI: {bmi}
- AMH: {amh} ng/mL | FSH: {fsh} IU/L
- Diagnosis: {diagnosis}
- Stimulation protocol: {protocol}
- Previous IVF cycles: {prev_cycles} ({prev_outcomes})
- Partner age: {partner_age}
- Sperm: {sperm_conc} M/mL, {sperm_mot}% motile
- Endometrial thickness: {endo_mm} mm
- Lifestyle: {lifestyle}

## Task
Provide a comprehensive clinical assessment including:
1. Morphological analysis of the blastocyst image
2. Integration of patient factors with embryo quality
3. Risk factors and favorable indicators for this case
4. Estimated implantation probability range with reasoning
5. Recommendations for this transfer and future cycle planning"""


def render_prompt(case: BenchmarkCase) -> str:
    """Render the prompt template with case data."""
    g = case.gardner
    ctx = case.patient_context
    lab = case.lab_data

    prev_outcomes_str = ", ".join(ctx.previous_outcomes) if ctx.previous_outcomes else "none"

    return PROMPT_TEMPLATE.format(
        gardner_display=g.display,
        exp_desc=EXPANSION_DESCRIPTIONS.get(g.exp, f"stage {g.exp}"),
        icm_desc=ICM_DESCRIPTIONS.get(g.icm_letter, "unknown"),
        te_desc=TE_DESCRIPTIONS.get(g.te_letter, "unknown"),
        cocs=lab.cocs_retrieved,
        mii=lab.mii_oocytes,
        transfer_day=lab.transfer_day,
        age=ctx.age,
        bmi=ctx.bmi,
        amh=ctx.amh_ng_ml if ctx.amh_ng_ml is not None else "N/A",
        fsh=ctx.fsh_iu_l,
        diagnosis=ctx.diagnosis.replace("_", " ").title(),
        protocol=ctx.protocol.replace("_", " ").title(),
        prev_cycles=ctx.previous_cycles,
        prev_outcomes=prev_outcomes_str,
        partner_age=ctx.partner_age,
        sperm_conc=ctx.sperm_concentration_million_ml,
        sperm_mot=ctx.sperm_motility_pct,
        endo_mm=ctx.endometrial_thickness_mm if ctx.endometrial_thickness_mm is not None else "N/A",
        lifestyle=ctx.lifestyle_notes,
    )
