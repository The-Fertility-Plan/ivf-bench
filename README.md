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
Scores are 1 to 5 from a five-rubric GPT-5.4 judge that is shown the same embryo
image the model was shown; Brier and AUROC compare the
model's stated probability against the real clinical pregnancy outcome. Bold
marks our row for identification only and does not indicate a best value: the
best overall score is GPT-5.4's and the best Brier is Sonnet 4.6's.

| # | Model | Overall | Morph | Clinical | Reasoning | Guideline | Recommend | Brier | AUROC | Cost / 550 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | GPT-5.4 | 4.44 | 3.74 | 5.00 | 4.49 | 4.10 | 4.88 | 0.246 | 0.562 | $57.85 |
| 2 | **IVF-Bench-Qwen9B-ORPO (ours)** | 3.97 | 3.14 | 4.65 | 3.91 | 3.90 | 4.23 | 0.245 | 0.541 | $8.40* |
| 3 | Claude Opus 4.6 | 3.82 | 3.02 | 4.76 | 3.95 | 3.15 | 4.22 | 0.247 | 0.551 | $180.51 |
| 4 | Gemini 2.5 Flash | 3.69 | 2.95 | 4.17 | 3.91 | 3.47 | 3.97 | 0.289 | 0.504 | $8.79 |
| 5 | Qwen 3.5-397B | 3.61 | 2.85 | 4.05 | 3.85 | 3.32 | 3.97 | 0.265 | 0.551 | $11.94 |
| 6 | Claude Sonnet 4.6 | 3.54 | 2.92 | 4.33 | 3.61 | 2.81 | 4.02 | 0.242 | 0.535 | $35.48 |
| 7 | Kimi K2.5 | 3.47 | 2.77 | 4.17 | 3.58 | 2.99 | 3.85 | 0.255 | 0.516 | $12.07 |
| 8 | Qwen 3.5-9B (base) | 3.18 | 2.64 | 3.87 | 3.08 | 2.78 | 3.54 | 0.283 | 0.456 | $7.10 |

\* Our model's cost is not comparable with the rest of the column. Its inference
ran on hardware we already operated, so the $8.40 is almost entirely judge cost,
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

**Post-training a 9B open model on 500 benchmark-derived pairs raises it 24.6%
over its own base, and that holds whoever grades.** Base Qwen 3.5-9B scores 3.18
on the held-out split and the ORPO'd version scores 3.97, with all five rubrics
up and the largest move on guideline alignment (+41%). A second judge from a
different model family puts the same gain at 15.5% (3.57 to 4.13). This is the
claim we would defend hardest: it compares two versions of one model on the same
cases, so no ranking convention can distort it. Two caveats sit inside it. The base
model was decoded greedily and the trained model with the Qwen card's recommended
sampling, because greedy decoding sends the trained model into a repetition loop.
And the base model was served by whichever provider OpenRouter routed to
(Together, Venice and DeepInfra across the run) while ours ran on our own vLLM.
Some unknown part of the gain therefore belongs to decoding or to the serving
stack, and matching both is the clean experiment we did not run.

**The same gain puts it second of eight ahead of Claude Opus 4.6 under our judge,
and fifth under another.** Under GPT-5.4 the gap is +0.15
with a 95% bootstrap interval of [0.08, 0.21], at roughly one seventh of the
marginal inference cost ($0.044 against $0.310 per case). Under Claude Sonnet
4.6, scoring the same 824 held-out judgements, the gap is -0.49 and our model
falls to fifth. Each judge is most generous to its own family: the two Anthropic
systems gain the most from the Anthropic judge (+0.90 and +0.79) and the two
GPT-lineage systems the least (+0.18 and +0.16), and response length does not
explain it. What survives both judges is the comparison that stays inside one
family, our model against its own base: +24.6% under GPT-5.4 and +15.5% under
Sonnet 4.6, with all five rubrics moving. Reproduce with
`scripts/cross_judge.py`.

**Most of the middle of the leaderboard is a statistical tie.** On the held-out
split Sonnet against Kimi is not separable, and on the test split the base Qwen
9B against Gemini Flash is not. Full intervals come from
`scripts/paper_analysis.py`.

**We cannot measure what the embryo image contributes, and we explain why.** We
ran the full 2x2 (image and grade, grade only, image only, neither) and the arm
with no inputs at all turned out not to be a floor: a model given neither input
declines to describe the embryo, and a rubric that asks whether a description is
accurate scores an honest refusal in the middle of the range. Three of four
systems score *higher* on morphology grounding with nothing than with both. An
earlier version of this repo claimed the image is worth 0.07 to 0.23 points; that
number came from a judge that could not see the embryo either, and we retract it.
The one edge that survives is image-and-grade against image-only, where the
expert grade adds 0.23 to 0.60 points for three of four systems. Reproduce with
`scripts/ablation_edges.py`.

**The generation budget was not equal across systems, and of the problems we
could not correct it is the one whose cost we can measure.** It was set per
backend and the settings disagreed:
12,000 output tokens on Bedrock and the OpenAI path, 4,096 on OpenRouter. The
three systems that were never cut off (GPT-5.4, Opus 4.6, Sonnet 4.6) are exactly
the three on the larger budget; all five on the smaller one lost answers mid
sentence, Gemini 2.5 Flash on 440 of 550 test cases, Kimi K2.5 on 197, base Qwen
9B on 94, our model on 74, Qwen 397B on 12. So Gemini's row is not a fair
measurement of that system, and our own margin over Opus is achieved from behind.
Set one budget large enough that nothing reaches it if you rerun this.

**Four leaderboard rows are provider mixtures, not single deployments.** The
systems we reached through OpenRouter were served by whichever third-party host
the router picked per call: 15 distinct providers across the Kimi K2.5 run, 11
across Qwen 397B, 3 across base Qwen 9B. Providers differ in quantisation,
sampling implementation and chat templating, so those rows average over
deployments of the same weights. The Bedrock and OpenAI rows and our own local
vLLM row do not have this problem. We record the provider on every call, so the
mixture is inspectable in the released artifacts. Pin a provider if you rerun
this.

The 550-case test leaderboard is in [`data/runs/leaderboard.md`](data/runs/leaderboard.md).

## What is in a case

753 cases built from the public [Kromp blastocyst
dataset](https://doi.org/10.1038/s41597-023-02182-3) (CC BY 4.0). Each one has a
real embryo image taken on that case's transfer day, its silver-standard Gardner grade, real cycle data and outcomes
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
#    --with-image gives the judge the same embryo the model saw, which is the
#    configuration every reported score comes from
.venv/bin/ivf-bench score gpt-5.4-2026-03-05 \
    --judge gpt-5.4-2026-03-05 --backend openai --split test \
    --scores-dir scores_sighted --with-image -j 4 --max-cost 20
.venv/bin/ivf-bench leaderboard --scores-dir scores_sighted

# 4. Reproduce the paper's statistics (bootstrap CIs, rubric structure,
#    verbosity bias, patient-level robustness). Costs nothing.
.venv/bin/python scripts/paper_analysis.py
.venv/bin/python scripts/paper_numbers.py

# 5. Reproduce the 2x2 ablation, after collecting the withheld-input arms.
#    --image-as-reference keeps the judge's eyesight constant across arms,
#    which is what makes two arms comparable.
.venv/bin/python scripts/ablation_edges.py

# 6. Reproduce the cross-judge check and the full study cost
.venv/bin/python scripts/cross_judge.py
.venv/bin/python scripts/study_cost.py

# 7. Check every number in the paper against the artifacts
.venv/bin/python scripts/verify_paper.py
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
