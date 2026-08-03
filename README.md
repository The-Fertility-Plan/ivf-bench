# IVF-Bench

An open benchmark for measuring how well vision-language models reason about IVF
embryo transfer decisions, plus a 9B open model post-trained on it.

Most machine learning work on embryo selection predicts a score from an image.
This benchmark measures something different: given an embryo image *and* the
patient behind it, can a model describe what it sees, connect it to the patient's
history, give an honest probability, and recommend what to do next? That is the
job a clinician actually does before a transfer, and nobody was measuring it.

**Paper:** `arxiv/ivf_bench.pdf` in this repository (arXiv link on submission)
**Model:** [thefertilityplan/ivf-bench-qwen9b-vlm-orpo](https://huggingface.co/thefertilityplan/ivf-bench-qwen9b-vlm-orpo)
**Preference dataset:** [thefertilityplan/ivf-bench-orpo-qwen9b-clipped](https://huggingface.co/datasets/thefertilityplan/ivf-bench-orpo-qwen9b-clipped)

> **Not for clinical use.** This is a research benchmark. No model here is
> approved by any regulator or validated in a trial, and nothing in this
> repository should inform patient care.

## Results

Held-out split, 103 cases never used for training, tuning, or judge calibration.
Scores are 1 to 5 from a five-rubric GPT-5.4 judge; Brier and AUROC compare the
model's stated probability against the real clinical pregnancy outcome.

| # | Model | Overall | Morph | Clinical | Reasoning | Guideline | Recommend | Brier | AUROC | Cost / 550 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | GPT-5.4 | 4.55 | 3.98 | 5.00 | 4.65 | 4.23 | 4.88 | 0.246 | 0.562 | $61.10 |
| 2 | **IVF-Bench-Qwen9B-ORPO (ours)** | **4.11** | 3.56 | 4.76 | 3.93 | 3.93 | 4.37 | 0.245 | 0.547 | $7.98* |
| 3 | Claude Opus 4.6 | 4.01 | 3.60 | 4.92 | 3.98 | 3.22 | 4.31 | 0.247 | 0.551 | $184.81 |
| 4 | Gemini 2.5 Flash | 3.83 | 3.34 | 4.40 | 3.90 | 3.54 | 3.98 | 0.289 | 0.504 | $7.77 |
| 5 | Qwen 3.5-397B | 3.82 | 3.48 | 4.31 | 3.88 | 3.44 | 3.98 | 0.265 | 0.551 | $10.94 |
| 6 | Kimi K2.5 | 3.73 | 3.48 | 4.42 | 3.80 | 3.04 | 3.90 | 0.254 | 0.518 | $10.99 |
| 7 | Claude Sonnet 4.6 | 3.65 | 3.07 | 4.51 | 3.73 | 2.89 | 4.06 | **0.242** | 0.534 | $38.39 |
| 8 | Qwen 3.5-9B (base) | 3.31 | 2.99 | 3.91 | 3.18 | 2.85 | 3.64 | 0.285 | 0.450 | $6.37 |

\* Our model's cost is not comparable with the rest of the column. Its inference
ran on hardware we already operated, so the $7.98 is almost entirely judge cost,
while every other row is inference plus judge. Per case of inference the honest
figures are $0.044 for our model against $0.310 for Opus 4.6.

Note also that the best Brier score belongs to Sonnet 4.6, not to us, and that
**every model here is worse than a constant predictor** that returns the cohort's
35% pregnancy rate and ignores the embryo entirely (0.227).

Findings worth your attention:

**Models explain well and predict badly, and the ceiling is the data.** Clinical
integration tops out at a perfect 5.00. Outcome AUROC never exceeds 0.562, where
0.5 is a coin flip, and every model scores a worse Brier than a constant set to
the cohort's 35% pregnancy rate. That is not a verdict on the models: the
patient-context fields in these cases are sampled independently of outcome, so
they carry no signal to find. The benchmark is built so those fields can be
swapped from generated to measured when such a dataset exists.

**A 9B open model, post-trained on 500 training pairs, beats Claude Opus 4.6.**
The gap is 0.10 points with a 95% bootstrap interval of [0.03, 0.17] (p=0.007),
at roughly one seventh of the marginal inference cost ($0.044 against $0.310 per
case). Against its own base model it gains 24%, and all five rubrics move, not
one. The comparison is made by a single judge that also authored 92% of the
training targets, which is the largest caveat on it.

**Most of the middle of the leaderboard is a statistical tie.** On the held-out
split, Gemini Flash against Qwen 397B and Kimi against Sonnet are not separable.
Full intervals are reproduced by `scripts/paper_analysis.py`.

**The embryo image adds little once the Gardner grade is supplied in text.**
Withholding it changes morphology grounding by 0.07 to 0.23 points on a
five-point scale, and by nothing measurable for half the systems tested. The
grade is itself a human reading of the image, so this bounds the residual a model
extracts beyond an expert summary rather than showing the image is unnecessary.
Reproduce with `scripts/image_ablation.py`.

The 550-case test leaderboard is in [`data/runs/leaderboard.md`](data/runs/leaderboard.md).

## What is in a case

753 cases built from the public [Kromp blastocyst
dataset](https://doi.org/10.1038/s41597-023-02182-3) (CC BY 4.0). Each one has a
real day-5 embryo image, its silver-standard Gardner grade, real cycle data and outcomes
for that patient, and patient-history fields generated from published population
distributions.

Every generated field is named in a `synthetic_fields` list inside the case, so
you always know which numbers are measured and which are not:

```json
{
  "case_id": "IVF-BENCH-0001",
  "image_path": "0001_01.png",
  "gardner": {"exp": 1, "icm": 3, "te": 3},
  "lab_data": {"cocs_retrieved": 8, "mii_oocytes": 6, "transfer_day": 4},
  "patient_context": {
    "age": 39, "amh_ng_ml": 2.0, "endometrial_thickness_mm": 7.0,
    "bmi": 28.1, "diagnosis": "tubal_factor",
    "synthetic_fields": ["bmi", "fsh_iu_l", "diagnosis", "protocol", "..."]
  },
  "real_outcome": {"biochemical_pregnancy": false, "clinical_pregnancy": false,
                   "live_birth": false},
  "split": "test"
}
```

Splits are 550 test, 100 validation, 103 held out. The held-out split was
untouched through all training and judge calibration.

## Reproducing the benchmark

```bash
git clone https://github.com/The-Fertility-Plan/ivf-bench
cd ivf-bench
python -m venv .venv && .venv/bin/pip install -e .

# 1. Download the Kromp dataset (about 2 GB) and rebuild the cases
.venv/bin/ivf-bench download
.venv/bin/ivf-bench split
.venv/bin/ivf-bench generate
.venv/bin/ivf-bench validate

# 2. Run a model over a split
export OPENAI_API_KEY=...        # or OPENROUTER_API_KEY / AWS credentials
.venv/bin/ivf-bench run gpt-5.4-2026-03-05 --backend openai --split test

# 3. Score the responses with the judge, then build the leaderboard
.venv/bin/ivf-bench score gpt-5.4-2026-03-05 \
    --judge gpt-5.4-2026-03-05 --backend openai --split test -j 4 --max-cost 20
.venv/bin/ivf-bench leaderboard

# 4. Reproduce the paper's statistics (bootstrap CIs, rubric structure,
#    verbosity bias, patient-level robustness). Costs nothing.
.venv/bin/python scripts/paper_analysis.py

# 5. Reproduce the image ablation, after collecting the --no-image arms
.venv/bin/python scripts/image_ablation.py
```

Case generation is deterministic: every synthetic field is seeded from a hash of
the case ID, so step 1 reproduces our exact cases rather than something similar.

### Two things worth knowing before you run this

**Judge failures used to be invisible.** If the judge API returned an error, the
old code wrote a score file full of zeros, and every later run skipped that case
as already done. That silently hollowed out one model's results in our first
pass. The judge now refuses to write a failed score, retries genuine rate limits
with backoff, and fails fast on billing and authentication errors.
`--max-cost` caps spend per run.

**Serve local reasoning models with a reasoning parser.** vLLM without
`--reasoning-parser` returns the chain of thought inside `message.content`, so a
judge scores the thinking along with the answer while hosted APIs hand back the
answer alone. That is not a fair comparison. The runner now splits on `</think>`
automatically, so this is handled for you, but it is worth knowing if you collect
responses through another path.

## Layout

```
src/ivf_bench/          benchmark construction, inference runner, judge, metrics
scripts/                paper statistics, image ablation, preference-pair build,
                        training entrypoint
configs/                ORPO training config for the released model
data/cases/             550 test cases
data/validation_cases/  100 validation cases
data/held_out_cases/    103 held-out cases
data/splits/            split definitions
data/runs/              model responses, judge transcripts, leaderboards
arxiv/                  paper source, bibliography, and built PDF
```

## Training the model

Preference pairs are built from scored benchmark responses: for each case the
highest-scoring model output becomes `chosen` and the lowest-scoring becomes
`rejected`.

```bash
.venv/bin/python scripts/build_orpo_dataset.py     # writes data/orpo/{train,eval}.jsonl
bash scripts/lambda_train.sh configs/training_orpo_qwen_vlm_final.yaml
```

The dataset is uploaded to the Hub as an `Image()`-typed dataset so the trainer
decodes it to PIL; the released copy is linked above if you would rather skip
that step.

Training itself runs on 2x H100 through
[AITraining](https://github.com/monostate/aitraining), which wraps TRL's ORPO
trainer. `configs/` holds the sweep and final configurations.

## Citation

```bibtex
@article{correa2026ivfbench,
  author  = {Correa, Andrew G. A. and Yoon, Brittany},
  title   = {{IVF-Bench}: A Rubric-Based Standard for Evaluating
             Vision-Language Models on {IVF} Clinical Reasoning},
  journal = {arXiv preprint},
  year    = {2026}
}
```

## Licensing

Code is Apache 2.0. Embryo images and the real clinical fields come from Kromp et
al. (2023) and remain CC BY 4.0, so attribute them if you redistribute. Generated
patient-context fields are CC BY 4.0. See [LICENSE](LICENSE).

## Acknowledgement

This benchmark exists because Kromp and colleagues published their data openly.
If you use IVF-Bench, cite their paper as well as ours.
