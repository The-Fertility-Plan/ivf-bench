"""Inference runner for VLM evaluation. Saves full raw API responses."""
from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from pathlib import Path
from typing import Optional

import httpx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from ivf_bench.data.schemas import BenchmarkCase, ModelResponse

console = Console()

# Default is OpenRouter; can be overridden via IVFBENCH_OPENROUTER_URL env var
# (e.g. to a local vLLM server at http://localhost:8000/v1/chat/completions).
OPENROUTER_URL = os.environ.get(
    "IVFBENCH_OPENROUTER_URL",
    "https://openrouter.ai/api/v1/chat/completions",
)


def _slug(model: str) -> str:
    """Convert model ID to filesystem-safe slug."""
    return model.replace("/", "__").replace(".", "_")


def _load_image_b64(images_dir: Path, image_path: str) -> str:
    img = images_dir / image_path
    return base64.b64encode(img.read_bytes()).decode("ascii")


def _strip_grade(text: str) -> str:
    """Remove the Gardner grade line, leaving the rest of the case intact.

    Withholding the grade rather than the image is the other half of the two-by-two
    that separates a model which cannot read the embryo from one for which the
    grade had already answered the question.
    """
    out = []
    for line in text.splitlines():
        if line.strip().startswith("- Gardner Score:"):
            out.append("- Gardner Score: [NOT PROVIDED - assess morphology from the image]")
        else:
            out.append(line)
    return "\n".join(out)


def _build_messages(
    case: BenchmarkCase, images_dir: Path, include_image: bool = True,
    include_grade: bool = True,
) -> list[dict]:
    """Build chat messages with image (works for both OpenRouter and OpenAI).

    include_image=False drops the embryo image and replaces the "[IMAGE ATTACHED]"
    marker in the prompt. This is the ablation that asks how much of a model's
    score comes from looking at the embryo rather than from the Gardner grade and
    patient data it is given in text.
    """
    prompt = case.prompt if include_grade else _strip_grade(case.prompt)
    if not include_image:
        text = prompt.replace(
            "[IMAGE ATTACHED]", "[NO IMAGE PROVIDED - reason from the data below]"
        )
        return [{"role": "user", "content": [{"type": "text", "text": text}]}]

    b64 = _load_image_b64(images_dir, case.image_path)
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]


def _split_inline_thinking(content: str) -> tuple[str, str | None]:
    """Separate an inlined chain-of-thought from the answer.

    A vLLM server without --reasoning-parser returns the whole generation in
    message.content. Qwen thinking templates pre-fill "<think>" in the prompt, so
    the model emits only the closing tag: everything before </think> is reasoning.
    Hosted APIs deliver this in a separate field; without this split the judge
    would score the reasoning too, which no other backend does.
    """
    if not content or "</think>" not in content:
        return content, None
    head, _, answer = content.rpartition("</think>")
    reasoning = head.replace("<think>", "").strip()
    return answer.strip(), reasoning or None


# ---------------------------------------------------------------------------
# OpenRouter backend
# ---------------------------------------------------------------------------
async def _call_openrouter(
    client: httpx.AsyncClient,
    api_key: str,
    model: str,
    messages: list[dict],
    case_id: str,
) -> tuple[ModelResponse, dict | None]:
    """Returns (ModelResponse, raw_api_response_dict)."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/ivf-bench",
        "X-Title": "IVF-Bench",
    }
    # Allow sampling overrides via env (needed for our trained Qwen — greedy
    # decoding traps it in a repetition loop, per Qwen 3.5 model card guidance).
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": int(os.environ.get("IVFBENCH_MAX_TOKENS", "4096")),
        "temperature": float(os.environ.get("IVFBENCH_TEMPERATURE", "0.0")),
    }
    _top_p = os.environ.get("IVFBENCH_TOP_P")
    if _top_p:
        payload["top_p"] = float(_top_p)
    _top_k = os.environ.get("IVFBENCH_TOP_K")
    if _top_k:
        payload["top_k"] = int(_top_k)
    _rp = os.environ.get("IVFBENCH_REPETITION_PENALTY")
    if _rp:
        payload["repetition_penalty"] = float(_rp)

    last_error = None
    for attempt in range(5):
        t0 = time.monotonic()
        try:
            resp = await client.post(
                OPENROUTER_URL, headers=headers, json=payload, timeout=180.0,
            )
            latency = (time.monotonic() - t0) * 1000

            if resp.status_code == 429:
                wait = min(2 ** attempt * 5, 60)
                console.print(f"  [yellow]Rate limited on {case_id}, waiting {wait}s...[/yellow]")
                await asyncio.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()

            if "error" in data:
                return ModelResponse(
                    case_id=case_id, model=model, response_text="",
                    prompt_tokens=0, completion_tokens=0, total_tokens=0,
                    cost_usd=0.0, latency_ms=latency, error=str(data["error"]),
                ), data

            usage = data.get("usage", {})
            choice = data["choices"][0]
            msg = choice["message"]
            text = msg.get("content") or ""
            reasoning = msg.get("reasoning") or msg.get("reasoning_content") or None
            if reasoning is None:
                text, reasoning = _split_inline_thinking(text)
            return ModelResponse(
                case_id=case_id, model=model,
                response_text=text,
                reasoning_text=reasoning,
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
                cost_usd=float(usage.get("total_cost", 0) or 0),
                latency_ms=latency,
                finish_reason=choice.get("finish_reason"),
            ), data

        except httpx.HTTPStatusError as e:
            last_error = f"HTTP {e.response.status_code}: {e.response.text[:200]}"
            if e.response.status_code >= 500:
                await asyncio.sleep(min(2 ** attempt * 3, 30))
                continue
            break
        except httpx.TransportError as e:
            last_error = str(e)
            await asyncio.sleep(min(2 ** attempt * 3, 30))
            continue

    return ModelResponse(
        case_id=case_id, model=model, response_text="",
        prompt_tokens=0, completion_tokens=0, total_tokens=0,
        cost_usd=0.0, latency_ms=0.0, error=last_error or "Unknown error",
    ), None


# ---------------------------------------------------------------------------
# OpenAI direct backend (Responses API with reasoning support)
# ---------------------------------------------------------------------------
async def _call_openai(
    api_key: str,
    model: str,
    messages: list[dict],
    case_id: str,
) -> tuple[ModelResponse, dict | None]:
    """Returns (ModelResponse, raw_api_response_dict). Uses Responses API with reasoning."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key)

    # Convert chat messages to Responses API input format
    resp_input = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            items = []
            for part in content:
                if part.get("type") == "image_url":
                    items.append({
                        "type": "input_image",
                        "image_url": part["image_url"]["url"],
                    })
                elif part.get("type") == "text":
                    items.append({
                        "type": "input_text",
                        "text": part["text"],
                    })
            resp_input.append({"role": msg["role"], "content": items})
        else:
            resp_input.append({"role": msg["role"], "content": content})

    last_error = None
    for attempt in range(5):
        t0 = time.monotonic()
        try:
            resp = await client.responses.create(
                model=model,
                input=resp_input,
                reasoning={"effort": "high", "summary": "auto"},
                max_output_tokens=12000,
            )
            latency = (time.monotonic() - t0) * 1000

            raw = resp.model_dump()

            # Extract response text and reasoning summaries from output
            text = ""
            reasoning_parts = []
            for item in resp.output:
                item_type = getattr(item, "type", None)
                if item_type == "message":
                    for c in item.content:
                        if getattr(c, "type", None) == "output_text":
                            text = c.text or ""
                elif item_type == "reasoning":
                    for s in (item.summary or []):
                        if hasattr(s, "text") and s.text:
                            reasoning_parts.append(s.text)

            reasoning_text = "\n\n".join(reasoning_parts) if reasoning_parts else None

            usage = resp.usage
            pt = usage.input_tokens if usage else 0
            ct = usage.output_tokens if usage else 0
            tt = pt + ct

            return ModelResponse(
                case_id=case_id, model=model,
                response_text=text,
                reasoning_text=reasoning_text,
                prompt_tokens=pt, completion_tokens=ct, total_tokens=tt,
                cost_usd=0.0,
                latency_ms=latency,
                finish_reason=getattr(resp, "status", "completed"),
            ), raw

        except Exception as e:
            last_error = str(e)
            if "rate" in last_error.lower() or "429" in last_error:
                wait = min(2 ** attempt * 5, 60)
                console.print(f"  [yellow]Rate limited on {case_id}, waiting {wait}s...[/yellow]")
                await asyncio.sleep(wait)
                continue
            if attempt < 4:
                await asyncio.sleep(min(2 ** attempt * 3, 30))
                continue
            break

    return ModelResponse(
        case_id=case_id, model=model, response_text="",
        prompt_tokens=0, completion_tokens=0, total_tokens=0,
        cost_usd=0.0, latency_ms=0.0, error=last_error or "Unknown error",
    ), None


# ---------------------------------------------------------------------------
# Gemini backend (via OpenAI-compatible endpoint)
# ---------------------------------------------------------------------------
async def _call_gemini(
    api_key: str,
    model: str,
    messages: list[dict],
    case_id: str,
) -> tuple[ModelResponse, dict | None]:
    """Returns (ModelResponse, raw_api_response_dict)."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    last_error = None
    for attempt in range(5):
        t0 = time.monotonic()
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=4096,
            )
            latency = (time.monotonic() - t0) * 1000

            # Dump the full SDK response to dict
            raw = resp.model_dump()

            msg = resp.choices[0].message
            text = msg.content or ""
            reasoning = getattr(msg, "reasoning_content", None)
            pt = resp.usage.prompt_tokens if resp.usage else 0
            ct = resp.usage.completion_tokens if resp.usage else 0
            tt = resp.usage.total_tokens if resp.usage else 0

            return ModelResponse(
                case_id=case_id, model=model,
                response_text=text,
                reasoning_text=reasoning,
                prompt_tokens=pt, completion_tokens=ct, total_tokens=tt,
                cost_usd=0.0,
                latency_ms=latency,
                finish_reason=resp.choices[0].finish_reason,
            ), raw

        except Exception as e:
            last_error = str(e)
            if "429" in last_error or "rate" in last_error.lower() or "quota" in last_error.lower():
                wait = min(2 ** attempt * 10, 60)
                console.print(f"  [yellow]Rate limited on {case_id}, waiting {wait}s...[/yellow]")
                await asyncio.sleep(wait)
                continue
            if attempt < 4:
                await asyncio.sleep(min(2 ** attempt * 3, 30))
                continue
            break

    return ModelResponse(
        case_id=case_id, model=model, response_text="",
        prompt_tokens=0, completion_tokens=0, total_tokens=0,
        cost_usd=0.0, latency_ms=0.0, error=last_error or "Unknown error",
    ), None


# ---------------------------------------------------------------------------
# Bedrock backend (AWS Converse API with vision)
# ---------------------------------------------------------------------------
def _bedrock_model_variants(model: str) -> list[str]:
    """Return model ID variants to rotate between (global. and us. prefixes have separate quotas)."""
    variants = [model]
    if model.startswith("global."):
        variants.append("us." + model[len("global."):])
    elif model.startswith("us."):
        variants.append("global." + model[len("us."):])
    return variants


async def _call_bedrock(
    model: str,
    case: "BenchmarkCase",
    images_dir: Path,
    case_id: str,
    region: str = "us-east-1",
    include_image: bool = True,
    include_grade: bool = True,
) -> tuple[ModelResponse, dict | None]:
    """Returns (ModelResponse, raw_api_response_dict).

    Automatically rotates between global. and us. inference profiles when
    hitting daily token quota limits, with exponential backoff up to 8 min.
    """
    import boto3
    from botocore.config import Config as BotoConfig

    def _sync_call() -> tuple[ModelResponse, dict | None]:
        client = boto3.client(
            "bedrock-runtime", region_name=region,
            config=BotoConfig(read_timeout=300, connect_timeout=10, retries={"max_attempts": 0}),
        )
        prompt = case.prompt if include_grade else _strip_grade(case.prompt)
        if include_image:
            img_bytes = (images_dir / case.image_path).read_bytes()
            content = [
                {"image": {"format": "png", "source": {"bytes": img_bytes}}},
                {"text": prompt},
            ]
        else:
            content = [{"text": prompt.replace(
                "[IMAGE ATTACHED]",
                "[NO IMAGE PROVIDED - reason from the data below]")}]
        messages = [{"role": "user", "content": content}]

        variants = _bedrock_model_variants(model)
        last_error = None
        # Up to 12 attempts with exponential backoff (covers both prefixes)
        for attempt in range(12):
            # Rotate between global. and us. prefixes
            current_model = variants[attempt % len(variants)]
            t0 = time.monotonic()
            try:
                resp = client.converse(
                    modelId=current_model,
                    messages=messages,
                    inferenceConfig={"maxTokens": 12000},
                    additionalModelRequestFields={
                        "thinking": {
                            "type": "enabled",
                            "budget_tokens": 8000,
                        }
                    },
                )
                latency = (time.monotonic() - t0) * 1000

                # Extract text and thinking from content blocks
                text = ""
                thinking_text = None
                content_blocks = resp["output"]["message"]["content"]
                raw_content = []
                for block in content_blocks:
                    if "text" in block:
                        text = block["text"]
                        raw_content.append({"type": "text", "text": text})
                    elif "reasoningContent" in block:
                        rc = block["reasoningContent"]
                        rt = rc.get("reasoningText", {})
                        if isinstance(rt, dict):
                            thinking_text = rt.get("text", "")
                        elif isinstance(rt, str):
                            thinking_text = rt
                        raw_content.append({"type": "thinking", "text": thinking_text or ""})

                usage = resp.get("usage", {})
                pt = usage.get("inputTokens", 0)
                ct = usage.get("outputTokens", 0)
                tt = pt + ct

                raw = {
                    "stopReason": resp.get("stopReason"),
                    "usage": usage,
                    "metrics": resp.get("metrics"),
                    "output": {"message": {"role": "assistant", "content": raw_content}},
                    "model": model,  # Store original model name for consistency
                }

                return ModelResponse(
                    case_id=case_id, model=model,
                    response_text=text,
                    reasoning_text=thinking_text if thinking_text else None,
                    prompt_tokens=pt, completion_tokens=ct, total_tokens=tt,
                    cost_usd=0.0,
                    latency_ms=latency,
                    finish_reason=resp.get("stopReason"),
                ), raw

            except Exception as e:
                last_error = str(e)
                is_throttle = "ThrottlingException" in last_error or "Too many requests" in last_error
                is_daily_quota = "Too many tokens per day" in last_error

                if is_daily_quota:
                    # Daily quota hit — try alternate prefix, exponential backoff up to 8 min
                    wait = min(2 ** (attempt // 2) * 60, 480)
                    alt = variants[(attempt + 1) % len(variants)]
                    console.print(
                        f"  [yellow]Daily quota hit on {current_model} for {case_id}, "
                        f"switching to {alt}, waiting {wait}s...[/yellow]"
                    )
                    time.sleep(wait)
                    continue
                elif is_throttle:
                    # Rate limit — short backoff
                    wait = min(2 ** attempt * 3, 60)
                    time.sleep(wait)
                    continue
                break

        latency = (time.monotonic() - t0) * 1000
        return ModelResponse(
            case_id=case_id, model=model, response_text="",
            prompt_tokens=0, completion_tokens=0, total_tokens=0,
            cost_usd=0.0, latency_ms=latency, error=last_error or "Unknown error",
        ), None

    return await asyncio.get_event_loop().run_in_executor(None, _sync_call)


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------
def _get_run_dir(base_dir: Path, model: str) -> Path:
    run_dir = base_dir / _slug(model)
    (run_dir / "responses").mkdir(parents=True, exist_ok=True)
    (run_dir / "raw").mkdir(parents=True, exist_ok=True)
    return run_dir


def _already_done(run_dir: Path, case_id: str, require_reasoning: bool = False) -> bool:
    resp_file = run_dir / "responses" / f"{case_id}.json"
    if not resp_file.exists():
        return False
    try:
        data = json.loads(resp_file.read_text())
        if data.get("error") is not None or data.get("response_text", "") == "":
            return False
        if require_reasoning and not data.get("reasoning_text"):
            return False
        return True
    except (json.JSONDecodeError, KeyError):
        return False


def _save_response(run_dir: Path, resp: ModelResponse, raw: dict | None = None) -> None:
    """Save parsed response + full raw API response."""
    out = run_dir / "responses" / f"{resp.case_id}.json"
    out.write_text(resp.model_dump_json(indent=2))
    if raw is not None:
        raw_out = run_dir / "raw" / f"{resp.case_id}.json"
        raw_out.write_text(json.dumps(raw, indent=2, default=str))


def _compute_cost(resp: ModelResponse, input_price: float, output_price: float) -> float:
    if resp.cost_usd > 0:
        return resp.cost_usd
    return (resp.prompt_tokens * input_price + resp.completion_tokens * output_price) / 1_000_000


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
async def run_model(
    api_key: str,
    model: str,
    cases_dir: Path,
    images_dir: Path,
    runs_dir: Path,
    split: str = "test",
    concurrency: int = 5,
    limit: Optional[int] = None,
    input_price: float = 0.0,
    output_price: float = 0.0,
    backend: str = "openrouter",
    force_rerun: bool = False,
    include_image: bool = True,
    include_grade: bool = True,
    run_suffix: str = "",
) -> Path:
    """Run inference for a model on all cases. Returns run_dir.

    include_image=False runs the text-only ablation. run_suffix keeps that
    ablation in its own run directory so it never overwrites the main results.
    """
    run_dir = _get_run_dir(runs_dir, model + run_suffix)

    case_files = sorted(cases_dir.glob("IVF-BENCH-*.json"))
    cases: list[BenchmarkCase] = []
    for cf in case_files:
        data = json.loads(cf.read_text())
        prompt = data.pop("prompt", "")
        c = BenchmarkCase(**data)
        c.__dict__["prompt"] = prompt
        cases.append(c)

    if limit:
        cases = cases[:limit]

    # Require reasoning_text for backends that support thinking
    require_reasoning = backend in ("bedrock", "openai")

    if force_rerun:
        pending = list(cases)
    else:
        pending = [c for c in cases if not _already_done(run_dir, c.case_id, require_reasoning=require_reasoning)]
    done_count = len(cases) - len(pending)

    if not pending:
        console.print(f"[green]All {len(cases)} cases already completed.[/green]")
        return run_dir

    console.print(
        f"Running [bold]{model}[/bold] via [cyan]{backend}[/cyan] on {len(pending)} cases "
        f"({done_count} already done, {concurrency} concurrent)"
    )

    sem = asyncio.Semaphore(concurrency)
    total_cost = 0.0
    completed = 0
    errors = 0

    async def process(case: BenchmarkCase) -> None:
        nonlocal total_cost, completed, errors
        async with sem:
            try:
                messages = _build_messages(case, images_dir, include_image=include_image, include_grade=include_grade)

                if backend == "bedrock":
                    resp, raw = await _call_bedrock(
                        model, case, images_dir, case.case_id,
                        include_image=include_image, include_grade=include_grade)
                elif backend == "openai":
                    resp, raw = await _call_openai(api_key, model, messages, case.case_id)
                elif backend == "gemini":
                    resp, raw = await _call_gemini(api_key, model, messages, case.case_id)
                else:
                    async with httpx.AsyncClient() as client:
                        resp, raw = await _call_openrouter(client, api_key, model, messages, case.case_id)

                if resp.error:
                    errors += 1
                    console.print(f"  [red]{case.case_id}: {resp.error}[/red]")
                else:
                    resp.cost_usd = _compute_cost(resp, input_price, output_price)
                    total_cost += resp.cost_usd
                    completed += 1

                _save_response(run_dir, resp, raw)
            except Exception as e:
                errors += 1
                console.print(f"  [red]{case.case_id}: Unhandled error: {e}[/red]")
                err_resp = ModelResponse(
                    case_id=case.case_id, model=model, response_text="",
                    prompt_tokens=0, completion_tokens=0, total_tokens=0,
                    cost_usd=0.0, latency_ms=0.0, error=str(e),
                )
                _save_response(run_dir, err_resp)
            progress.update(task, advance=1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"Inference: {model}", total=len(pending))
        await asyncio.gather(*[process(c) for c in pending])

    console.print(
        f"\n[bold green]Done:[/bold green] {completed} completed, {errors} errors, "
        f"${total_cost:.4f} total cost"
    )

    meta = {
        "model": model + run_suffix,
        "base_model": model,
        "include_image": include_image,
        "include_grade": include_grade,
        "model_slug": _slug(model + run_suffix),
        "split": split,
        "backend": backend,
        "total_cases": len(cases),
        "completed": done_count + completed,
        "failed": errors,
        "total_cost_usd": total_cost,
        "input_price_per_m": input_price,
        "output_price_per_m": output_price,
        # Sampling and budget come from the environment, so record what was
        # actually in force. Without this a finished run cannot say how it was
        # decoded, and a comparison between two runs cannot be checked.
        "sampling": {
            "temperature": float(os.environ.get("IVFBENCH_TEMPERATURE", "0.0")),
            "top_p": os.environ.get("IVFBENCH_TOP_P"),
            "top_k": os.environ.get("IVFBENCH_TOP_K"),
            "repetition_penalty": os.environ.get("IVFBENCH_REPETITION_PENALTY"),
            "max_tokens": int(os.environ.get("IVFBENCH_MAX_TOKENS", "4096")),
        },
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    return run_dir
