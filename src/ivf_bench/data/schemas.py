"""Pydantic models for Kromp CSV records, benchmark cases, and evaluation."""
from __future__ import annotations

from pydantic import BaseModel, Field
from typing import Optional


# Numeric-to-letter mapping for Gardner ICM/TE grades
_GRADE_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}


class GardnerGrade(BaseModel):
    """Gardner grading system: Expansion (0-6), ICM (0-3 → A-D), TE (0-3 → A-D)."""
    exp: int = Field(ge=0, le=6)
    icm: int = Field(ge=0, le=3)
    te: int = Field(ge=0, le=3)

    @property
    def icm_letter(self) -> str:
        return _GRADE_MAP.get(self.icm, "?")

    @property
    def te_letter(self) -> str:
        return _GRADE_MAP.get(self.te, "?")

    @property
    def display(self) -> str:
        return f"{self.exp}{self.icm_letter}{self.te_letter}"


class KrompRecord(BaseModel):
    """A record from complete.csv joined with clinical_dataset.csv."""
    filename: str
    gardner: GardnerGrade

    # Clinical fields from clinical_dataset.csv (all required after join)
    age: int
    amh: Optional[float] = None  # can be NaN in source
    endometrial_thickness: Optional[float] = None  # Endo column, can be NaN
    coc: Optional[int] = None  # can be empty in source
    mii: int
    transfer_day: int  # Day column: 4 or 5
    biochemical_pregnancy: bool  # SS
    clinical_pregnancy: bool  # HA
    live_birth: bool  # LB


class RealClinical(BaseModel):
    """Raw Kromp clinical values preserved as-is."""
    age: int
    amh: Optional[float] = None
    endometrial_thickness: Optional[float] = None
    cocs_retrieved: Optional[int] = None
    mii_oocytes: int
    transfer_day: int


class PatientContext(BaseModel):
    """Patient profile combining real Kromp data + synthetic augmentation."""
    # Real fields (from Kromp)
    age: int
    amh_ng_ml: Optional[float] = None
    endometrial_thickness_mm: Optional[float] = None

    # Synthetic fields
    bmi: float
    fsh_iu_l: float
    diagnosis: str
    protocol: str
    previous_cycles: int
    previous_outcomes: list[str]
    partner_age: int
    sperm_concentration_million_ml: float
    sperm_motility_pct: float
    lifestyle_notes: str

    # Transparency: which fields are synthetic vs real
    synthetic_fields: list[str] = Field(default_factory=lambda: [
        "bmi", "fsh_iu_l", "diagnosis", "protocol",
        "previous_cycles", "previous_outcomes", "partner_age",
        "sperm_concentration_million_ml", "sperm_motility_pct", "lifestyle_notes",
    ])


class RealOutcome(BaseModel):
    """Real clinical outcome from Kromp."""
    biochemical_pregnancy: bool  # SS
    clinical_pregnancy: bool  # HA (clinical heart activity)
    live_birth: bool  # LB


class LabData(BaseModel):
    """Laboratory data for a case (all real from Kromp)."""
    cocs_retrieved: Optional[int] = None
    mii_oocytes: int
    transfer_day: int  # 4 or 5


class BenchmarkCase(BaseModel):
    """A complete benchmark case."""
    case_id: str = Field(pattern=r"^IVF-BENCH-\d{4}$")
    image_path: str
    gardner: GardnerGrade
    lab_data: LabData
    patient_context: PatientContext
    real_clinical: RealClinical
    real_outcome: RealOutcome
    split: str  # "test", "validation", or "held_out"


# ---------------------------------------------------------------------------
# Phase 2: Evaluation schemas
# ---------------------------------------------------------------------------
class ModelResponse(BaseModel):
    """Raw response from a VLM via OpenRouter."""
    case_id: str
    model: str
    response_text: str
    reasoning_text: Optional[str] = None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: float
    finish_reason: Optional[str] = None
    error: Optional[str] = None


class RubricScore(BaseModel):
    """Score for a single Health-SCORE rubric."""
    rubric: str
    score: int = Field(ge=0, le=5)  # 0 = judge failure
    justification: str


class JudgeResult(BaseModel):
    """Full judge output for one model response."""
    case_id: str
    model: str
    judge_model: str
    rubrics: list[RubricScore]
    extracted_probability: Optional[float] = None  # 0.0-1.0 if parseable
    judge_cost_usd: float = 0.0
    judge_latency_ms: float = 0.0
    judge_prompt_tokens: int = 0
    judge_completion_tokens: int = 0


class RunSummary(BaseModel):
    """Aggregated stats for a single model run."""
    model: str
    model_slug: str
    total_cases: int
    completed: int
    failed: int
    total_cost_usd: float
    total_tokens: int
    avg_latency_ms: float
    mean_scores: dict[str, float] = Field(default_factory=dict)  # rubric -> mean
    overall_score: float = 0.0
    brier_score: Optional[float] = None
    auroc: Optional[float] = None


class LeaderboardEntry(BaseModel):
    """One row in the leaderboard."""
    rank: int
    model: str
    overall_score: float
    morphological_accuracy: float
    clinical_integration: float
    reasoning_coherence: float
    guideline_alignment: float
    actionability: float
    outcome_consistency: Optional[float] = None
    brier_score: Optional[float] = None
    auroc: Optional[float] = None
    total_cost_usd: float
    avg_latency_ms: float
    cases_completed: int
