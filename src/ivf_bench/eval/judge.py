"""Health-SCORE LLM-as-judge scoring for IVF-Bench.

Supports three judge backends:
  - openrouter: OpenRouter API (default)
  - bedrock:    AWS Bedrock Converse API
  - openai:     OpenAI Chat Completions API
"""
from __future__ import annotations

import asyncio
import json
import random
import re
import time
from pathlib import Path
from typing import Optional

import httpx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from ivf_bench.data.schemas import JudgeResult, RubricScore

console = Console()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

RUBRIC_DEFINITIONS = {
    "morphological_accuracy": {
        "name": "Morphological Accuracy",
        "description": (
            "How accurately does the response describe the embryo's morphological features "
            "as visible in the image? Consider: expansion stage assessment, ICM description, "
            "TE description, zona pellucida observations, overall blastocyst appearance."
        ),
        "scale": {
            1: "Major errors or fabricated features not visible in the image",
            2: "Several inaccuracies in morphological description",
            3: "Mostly accurate with some minor errors or omissions",
            4: "Accurate description with good detail",
            5: "Precise, thorough morphological analysis matching the image",
        },
    },
    "clinical_integration": {
        "name": "Clinical Integration",
        "description": (
            "How well does the response integrate patient clinical factors (age, AMH, BMI, "
            "diagnosis, previous outcomes, sperm parameters) with the embryo assessment? "
            "Does it consider how these factors interact with embryo quality?"
        ),
        "scale": {
            1: "No integration of clinical factors with embryo assessment",
            2: "Mentions clinical factors but doesn't connect them to embryo quality",
            3: "Some integration but misses key interactions",
            4: "Good integration of most relevant clinical factors",
            5: "Comprehensive integration showing how clinical factors affect prognosis",
        },
    },
    "reasoning_coherence": {
        "name": "Reasoning Coherence",
        "description": (
            "Is the clinical reasoning logical, consistent, and free of contradictions? "
            "Does the probability estimate follow from the evidence presented? "
            "Are conclusions supported by the observations?"
        ),
        "scale": {
            1: "Contradictory or illogical reasoning",
            2: "Significant logical gaps or inconsistencies",
            3: "Generally coherent with minor logical issues",
            4: "Well-structured reasoning with clear logic",
            5: "Exemplary reasoning chain with all conclusions well-supported",
        },
    },
    "guideline_alignment": {
        "name": "Guideline Alignment",
        "description": (
            "Does the response align with established IVF clinical guidelines "
            "(ESHRE, ASRM, Gardner grading criteria)? Are the recommendations "
            "consistent with evidence-based reproductive medicine?"
        ),
        "scale": {
            1: "Recommendations contradict established guidelines",
            2: "Several recommendations not aligned with guidelines",
            3: "Mostly aligned with some deviations",
            4: "Well-aligned with current clinical guidelines",
            5: "Fully aligned, demonstrates deep knowledge of current guidelines",
        },
    },
    "actionability": {
        "name": "Actionability",
        "description": (
            "Are the recommendations specific, practical, and useful for a clinician? "
            "Does the response provide clear next steps for this transfer and future "
            "cycle planning? Are recommendations appropriately nuanced?"
        ),
        "scale": {
            1: "No actionable recommendations or only generic advice",
            2: "Vague recommendations with little practical value",
            3: "Some useful recommendations but lacking specificity",
            4: "Clear, specific recommendations for this case",
            5: "Highly specific, nuanced recommendations with alternatives considered",
        },
    },
}

JUDGE_PROMPT_TEMPLATE = """\
You are an expert evaluator assessing a Vision-Language Model's clinical analysis of an IVF embryo case.

## Original Case Prompt
{case_prompt}

## Model Response
{model_response}

## Evaluation Task
Score the model's response on each of the following 5 rubrics using a 1-5 scale.

{rubric_text}

## Also Extract
If the model provided a specific implantation probability estimate (a percentage or decimal), \
extract it as a number between 0.0 and 1.0. If no clear numeric estimate was given, output null.

## Required Output Format
Respond with ONLY a JSON object in this exact format (no markdown, no extra text):
{{
  "morphological_accuracy": {{"score": <1-5>, "justification": "<brief reason>"}},
  "clinical_integration": {{"score": <1-5>, "justification": "<brief reason>"}},
  "reasoning_coherence": {{"score": <1-5>, "justification": "<brief reason>"}},
  "guideline_alignment": {{"score": <1-5>, "justification": "<brief reason>"}},
  "actionability": {{"score": <1-5>, "justification": "<brief reason>"}},
  "extracted_probability": <float 0.0-1.0 or null>
}}"""


def _build_rubric_text() -> str:
    parts = []
    for key, rubric in RUBRIC_DEFINITIONS.items():
        lines = [f"### {rubric['name']}"]
        lines.append(rubric["description"])
        lines.append("Scale:")
        for score, desc in rubric["scale"].items():
            lines.append(f"  {score}: {desc}")
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def _parse_judge_response(text: str) -> tuple[list[RubricScore], Optional[float]]:
    text = text.strip()
    if "```" in text:
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            text = match.group(1).strip()

    data = json.loads(text)
    rubrics = []
    for key in RUBRIC_DEFINITIONS:
        entry = data[key]
        rubrics.append(RubricScore(
            rubric=key,
            score=int(entry["score"]),
            justification=entry["justification"],
        ))

    prob = data.get("extracted_probability")
    if prob is not None:
        prob = float(prob)
        if not (0.0 <= prob <= 1.0):
            prob = None
    return rubrics, prob


# ---------------------------------------------------------------------------
# Backend: OpenRouter
# ---------------------------------------------------------------------------
async def _call_openrouter(
    prompt: str,
    judge_model: str,
    api_key: str,
    input_price: float,
    output_price: float,
) -> dict:
    """Returns {text, prompt_tokens, completion_tokens, cost_usd, latency_ms, error}."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/ivf-bench",
        "X-Title": "IVF-Bench Judge",
    }
    payload = {
        "model": judge_model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 2048,
        "temperature": 0.0,
    }

    last_error = None
    for attempt in range(4):
        t0 = time.monotonic()
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    OPENROUTER_URL, headers=headers, json=payload, timeout=120.0,
                )
            latency = (time.monotonic() - t0) * 1000

            if resp.status_code == 429:
                await asyncio.sleep(min(2 ** attempt * 5, 60))
                continue

            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                last_error = str(data["error"])
                continue

            usage = data.get("usage", {})
            pt = usage.get("prompt_tokens", 0)
            ct = usage.get("completion_tokens", 0)
            cost = float(usage.get("total_cost", 0) or 0)
            if cost == 0 and (input_price > 0 or output_price > 0):
                cost = (pt * input_price + ct * output_price) / 1_000_000

            return {
                "text": data["choices"][0]["message"]["content"],
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "cost_usd": cost,
                "latency_ms": latency,
                "error": None,
            }
        except httpx.HTTPStatusError as e:
            last_error = f"HTTP {e.response.status_code}"
            if e.response.status_code >= 500:
                await asyncio.sleep(min(2 ** attempt * 3, 30))
                continue
            break
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            last_error = str(e)
            await asyncio.sleep(min(2 ** attempt * 3, 30))
            continue

    return {"text": "", "prompt_tokens": 0, "completion_tokens": 0,
            "cost_usd": 0, "latency_ms": 0, "error": last_error}


# ---------------------------------------------------------------------------
# Backend: AWS Bedrock
# ---------------------------------------------------------------------------
# Transient-failure handling
#
# A judge call that fails is not just a lost case: score_run used to persist the
# resulting all-zero rubrics, and every later run skipped the case as "done".
# That is how 231 test scores silently became zeros. Retry hard on the errors
# that are worth retrying, and let the caller drop anything still failing.
# ---------------------------------------------------------------------------
# Matched against the error string. Deliberately specific: a bare "500" would
# also match "Limit 500000" in a quota message and retry things that will never
# succeed, and a permanent error like credit_balance_exhausted must fail fast.
_RETRYABLE = (
    "rate_limit", "ratelimit", "rate limit", "too many requests",
    "error code: 429", "status 429", "(429)",
    "throttl", "timeout", "timed out", "connection error", "connection reset",
    "temporarily unavailable", "service unavailable", "internal server error",
    "bad gateway", "gateway timeout", "overloaded", "try again",
)
MAX_JUDGE_ATTEMPTS = 6


# Checked first: OpenAI returns HTTP 429 for an exhausted balance as well as for
# genuine rate limiting, so status alone cannot distinguish "wait" from "never".
_PERMANENT = (
    "insufficient_quota", "credit_balance_exhausted", "no credits remaining",
    "billing", "invalid_api_key", "incorrect api key", "authentication",
    "access denied", "accessdeniedexception", "you do not have access",
)


def _is_retryable(err: str) -> bool:
    e = err.lower()
    if any(t in e for t in _PERMANENT):
        return False
    return any(t in e for t in _RETRYABLE)


async def _with_retry(call, case_id: str, attempts: int = MAX_JUDGE_ATTEMPTS) -> dict:
    """Run an async judge call, retrying transient failures with backoff."""
    result = {"error": "no attempt made"}
    for attempt in range(attempts):
        result = await call()
        if not result["error"]:
            return result
        if not _is_retryable(result["error"]):
            return result
        if attempt == attempts - 1:
            break
        wait = min(2 ** attempt * 5, 90) * (0.75 + 0.5 * random.random())
        console.print(
            f"  [yellow]{case_id}: retryable judge error "
            f"(attempt {attempt + 1}/{attempts}), waiting {wait:.0f}s[/yellow]"
        )
        await asyncio.sleep(wait)
    return result


async def _call_bedrock(
    prompt: str,
    judge_model: str,
    input_price: float,
    output_price: float,
    region: str = "us-east-1",
) -> dict:
    """Call Bedrock Converse API (sync, wrapped in executor for async compat)."""
    import boto3

    def _sync_call() -> dict:
        client = boto3.client("bedrock-runtime", region_name=region)
        t0 = time.monotonic()
        try:
            resp = client.converse(
                modelId=judge_model,
                messages=[{
                    "role": "user",
                    "content": [{"text": prompt}],
                }],
                inferenceConfig={"maxTokens": 2048, "temperature": 0.0},
            )
            latency = (time.monotonic() - t0) * 1000

            text = resp["output"]["message"]["content"][0]["text"]
            usage = resp.get("usage", {})
            pt = usage.get("inputTokens", 0)
            ct = usage.get("outputTokens", 0)
            cost = (pt * input_price + ct * output_price) / 1_000_000

            return {
                "text": text,
                "prompt_tokens": pt,
                "completion_tokens": ct,
                "cost_usd": cost,
                "latency_ms": latency,
                "error": None,
            }
        except Exception as e:
            latency = (time.monotonic() - t0) * 1000
            return {"text": "", "prompt_tokens": 0, "completion_tokens": 0,
                    "cost_usd": 0, "latency_ms": latency, "error": str(e)}

    return await asyncio.get_event_loop().run_in_executor(None, _sync_call)


# ---------------------------------------------------------------------------
# Backend: OpenAI
# ---------------------------------------------------------------------------
async def _call_openai(
    prompt: str,
    judge_model: str,
    api_key: str,
    input_price: float,
    output_price: float,
) -> dict:
    """Call OpenAI Chat Completions API."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key)
    t0 = time.monotonic()
    try:
        resp = await client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=2048,
            temperature=0.0,
        )
        latency = (time.monotonic() - t0) * 1000

        text = resp.choices[0].message.content or ""
        pt = resp.usage.prompt_tokens if resp.usage else 0
        ct = resp.usage.completion_tokens if resp.usage else 0
        cost = (pt * input_price + ct * output_price) / 1_000_000

        return {
            "text": text,
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "cost_usd": cost,
            "latency_ms": latency,
            "error": None,
        }
    except Exception as e:
        latency = (time.monotonic() - t0) * 1000
        return {"text": "", "prompt_tokens": 0, "completion_tokens": 0,
                "cost_usd": 0, "latency_ms": latency, "error": str(e)}


# ---------------------------------------------------------------------------
# Unified judge call
# ---------------------------------------------------------------------------
async def _judge_one(
    backend: str,
    api_key: str,
    judge_model: str,
    case_prompt: str,
    model_response: str,
    case_id: str,
    model: str,
    rubric_text: str,
    input_price: float = 0.0,
    output_price: float = 0.0,
    bedrock_region: str = "us-east-1",
) -> JudgeResult:
    """Score one model response using the judge via any backend."""
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        case_prompt=case_prompt,
        model_response=model_response,
        rubric_text=rubric_text,
    )

    if backend == "bedrock":
        call = lambda: _call_bedrock(  # noqa: E731
            prompt, judge_model, input_price, output_price, bedrock_region)
    elif backend == "openai":
        call = lambda: _call_openai(  # noqa: E731
            prompt, judge_model, api_key, input_price, output_price)
    else:
        call = lambda: _call_openrouter(  # noqa: E731
            prompt, judge_model, api_key, input_price, output_price)

    result = await _with_retry(call, case_id)

    if result["error"]:
        return JudgeResult(
            case_id=case_id,
            model=model,
            judge_model=judge_model,
            rubrics=[
                RubricScore(rubric=k, score=0, justification=f"Judge failed: {result['error']}")
                for k in RUBRIC_DEFINITIONS
            ],
            judge_cost_usd=0.0,
        )

    try:
        rubrics, prob = _parse_judge_response(result["text"])
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        return JudgeResult(
            case_id=case_id,
            model=model,
            judge_model=judge_model,
            rubrics=[
                RubricScore(rubric=k, score=0, justification=f"Parse error: {e}")
                for k in RUBRIC_DEFINITIONS
            ],
            judge_cost_usd=result["cost_usd"],
            judge_latency_ms=result["latency_ms"],
            judge_prompt_tokens=result["prompt_tokens"],
            judge_completion_tokens=result["completion_tokens"],
        )

    return JudgeResult(
        case_id=case_id,
        model=model,
        judge_model=judge_model,
        rubrics=rubrics,
        extracted_probability=prob,
        judge_cost_usd=result["cost_usd"],
        judge_latency_ms=result["latency_ms"],
        judge_prompt_tokens=result["prompt_tokens"],
        judge_completion_tokens=result["completion_tokens"],
    )


# ---------------------------------------------------------------------------
# Score all responses in a run
# ---------------------------------------------------------------------------
async def score_run(
    api_key: str,
    judge_model: str,
    run_dir: Path,
    cases_dir: Path,
    concurrency: int = 5,
    limit: Optional[int] = None,
    input_price: float = 0.0,
    output_price: float = 0.0,
    backend: str = "openrouter",
    bedrock_region: str = "us-east-1",
    max_cost_usd: Optional[float] = None,
) -> Path:
    """Score all responses in a run directory. Returns scores_dir.

    max_cost_usd stops dispatching new judge calls once accumulated spend crosses
    the cap. Already-written scores are kept, so a capped run resumes cleanly.
    """
    responses_dir = run_dir / "responses"
    scores_dir = run_dir / "scores"
    scores_dir.mkdir(exist_ok=True)

    meta = json.loads((run_dir / "run_meta.json").read_text())
    model = meta["model"]

    resp_files = sorted(responses_dir.glob("IVF-BENCH-*.json"))
    rubric_text = _build_rubric_text()

    pending = []
    for rf in resp_files:
        case_id = rf.stem
        score_file = scores_dir / f"{case_id}.json"
        if score_file.exists():
            continue
        # Skip responses whose case file isn't in the active cases_dir
        # (so scoring on held_out doesn't try test-split responses, etc.)
        if not (cases_dir / f"{case_id}.json").exists():
            continue
        resp_data = json.loads(rf.read_text())
        if resp_data.get("error") or not resp_data.get("response_text"):
            continue
        pending.append((case_id, resp_data))

    if limit:
        pending = pending[:limit]

    already_scored = len(resp_files) - len(pending)
    if not pending:
        console.print(f"[green]All responses already scored ({already_scored} total).[/green]")
        return scores_dir

    console.print(
        f"Scoring [bold]{model}[/bold] with [bold]{judge_model}[/bold] "
        f"via [cyan]{backend}[/cyan]: {len(pending)} pending ({already_scored} already done)"
    )

    sem = asyncio.Semaphore(concurrency)
    total_judge_cost = 0.0
    scored = 0
    failed = 0

    capped = 0

    async def process(case_id: str, resp_data: dict) -> None:
        nonlocal total_judge_cost, scored, failed, capped
        async with sem:
            if max_cost_usd is not None and total_judge_cost >= max_cost_usd:
                capped += 1
                progress.update(task, advance=1)
                return

            case_file = cases_dir / f"{case_id}.json"
            case_data = json.loads(case_file.read_text())
            case_prompt = case_data.get("prompt", "")

            result = await _judge_one(
                backend, api_key, judge_model,
                case_prompt, resp_data["response_text"],
                case_id, model, rubric_text,
                input_price=input_price,
                output_price=output_price,
                bedrock_region=bedrock_region,
            )

            total_judge_cost += result.judge_cost_usd

            # A judge API failure yields all-zero rubrics. Persisting that file
            # would make the case look scored forever after, since pending is
            # built from missing score files.
            if all(r.score == 0 for r in result.rubrics):
                failed += 1
                progress.update(task, advance=1)
                return

            scored += 1
            out = scores_dir / f"{case_id}.json"
            out.write_text(result.model_dump_json(indent=2))
            progress.update(task, advance=1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Judging: {model}", total=len(pending))
        await asyncio.gather(*[process(cid, rd) for cid, rd in pending])

    console.print(
        f"\n[bold green]Scored:[/bold green] {scored} cases, "
        f"${total_judge_cost:.4f} judge cost"
    )
    if failed:
        console.print(
            f"[bold red]Judge failed on {failed} cases[/bold red] — not written; "
            f"re-run to retry them."
        )
    if capped:
        console.print(
            f"[bold yellow]Cost cap ${max_cost_usd:.2f} reached[/bold yellow] — "
            f"{capped} cases skipped. Raise --max-cost and re-run to continue."
        )

    return scores_dir
