from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Callable

from billiard.exceptions import SoftTimeLimitExceeded
from langchain_openai import ChatOpenAI

from app.core.config import get_settings


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMCallResult:
    text: str | None
    model: str | None
    attempts: int
    latency_ms: int
    error_type: str | None = None
    status_code: int | None = None
    streamed: bool = False
    chunk_count: int = 0
    event_count: int = 0
    reasoning_chunk_count: int = 0
    first_content_latency_ms: int | None = None

    def metadata(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("text", None)
        data["succeeded"] = bool(self.text)
        data["degraded"] = not bool(self.text)
        return data


def _create_model(
    *,
    base_url: str,
    api_key: str,
    model: str,
    thinking_mode: str,
    reasoning_effort: str,
) -> ChatOpenAI | None:
    settings = get_settings()
    if not api_key or not model:
        return None
    kwargs = {
        "api_key": api_key,
        "model": model,
        "temperature": settings.llm_temperature,
        # Provider transport timeout is supplemented by explicit stream
        # first-content/idle/overall deadlines below.
        "timeout": min(settings.llm_timeout_seconds, 45),
        "max_retries": 0,
    }
    if base_url:
        kwargs["base_url"] = base_url
    if "deepseek" in (base_url or "").lower() or model.lower().startswith("deepseek-"):
        kwargs["extra_body"] = {"thinking": {"type": thinking_mode}}
        if thinking_mode == "enabled":
            kwargs["reasoning_effort"] = reasoning_effort
    return ChatOpenAI(**kwargs)


@lru_cache
def get_chat_models() -> tuple[ChatOpenAI, ...]:
    settings = get_settings()
    models = [
        _create_model(base_url=settings.llm_base_url, api_key=settings.llm_api_key, model=settings.llm_model, thinking_mode=settings.llm_thinking_mode, reasoning_effort=settings.llm_reasoning_effort),
        _create_model(base_url=settings.fallback_llm_base_url, api_key=settings.fallback_llm_api_key, model=settings.fallback_llm_model, thinking_mode=settings.llm_thinking_mode, reasoning_effort=settings.llm_reasoning_effort),
    ]
    return tuple(model for model in models if model is not None)


@lru_cache
def get_fast_chat_models() -> tuple[ChatOpenAI, ...]:
    """Models for routing and memory maintenance where hidden CoT adds latency."""
    settings = get_settings()
    models = [
        _create_model(base_url=settings.llm_base_url, api_key=settings.llm_api_key, model=settings.llm_model, thinking_mode=settings.router_thinking_mode, reasoning_effort="low"),
        _create_model(base_url=settings.fallback_llm_base_url, api_key=settings.fallback_llm_api_key, model=settings.fallback_llm_model, thinking_mode=settings.router_thinking_mode, reasoning_effort="low"),
    ]
    return tuple(model for model in models if model is not None)


def get_chat_model() -> ChatOpenAI | None:
    models = get_chat_models()
    return models[0] if models else None


def _model_name(model: ChatOpenAI) -> str:
    return str(getattr(model, "model_name", None) or getattr(model, "model", None) or "unknown")


def _status_code(error: Exception) -> int | None:
    value = getattr(error, "status_code", None)
    if isinstance(value, int):
        return value
    response = getattr(error, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def _is_transient(error: Exception) -> bool:
    status = _status_code(error)
    if status == 429 or (status is not None and status >= 500):
        return True
    error_name = type(error).__name__
    return "timeout" in error_name.lower() or error_name in {
        "APIConnectionError",
        "APITimeoutError",
        "ConnectError",
        "ReadTimeout",
        "TimeoutError",
    }


def _chunk_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        item_type = str(content.get("type") or "").lower()
        if any(marker in item_type for marker in ("reasoning", "thinking", "analysis")):
            return ""
        value = content.get("text")
        return value if isinstance(value, str) else ""
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                item_type = str(item.get("type") or "").lower()
                if any(marker in item_type for marker in ("reasoning", "thinking", "analysis")):
                    continue
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
        return "".join(parts)
    return "" if content is None else str(content)


def _chunk_has_reasoning_activity(chunk: Any) -> bool:
    """Detect provider reasoning signals without exposing chain-of-thought text."""
    for container_name in ("additional_kwargs", "response_metadata"):
        container = getattr(chunk, container_name, None) or {}
        if not isinstance(container, dict):
            continue
        for key in ("reasoning_content", "reasoning", "thinking", "analysis"):
            value = container.get(key)
            if value not in (None, "", [], {}):
                return True

    blocks = getattr(chunk, "content_blocks", None)
    if blocks is None:
        content = getattr(chunk, "content", None)
        blocks = content if isinstance(content, list) else []
    for block in blocks or []:
        if not isinstance(block, dict):
            continue
        block_type = str(block.get("type") or "").lower()
        if any(marker in block_type for marker in ("reasoning", "thinking", "analysis")):
            return True
    return False


class StreamDeadlineError(TimeoutError):
    def __init__(
        self,
        message: str,
        *,
        partial_text: str = "",
        content_chunks: int = 0,
        event_count: int = 0,
        reasoning_chunks: int = 0,
        first_content_latency_ms: int | None = None,
    ) -> None:
        super().__init__(message)
        self.partial_text = partial_text
        self.content_chunks = content_chunks
        self.event_count = event_count
        self.reasoning_chunks = reasoning_chunks
        self.first_content_latency_ms = first_content_latency_ms


class FirstContentTimeoutError(StreamDeadlineError):
    pass


class StreamIdleTimeoutError(StreamDeadlineError):
    pass


class StreamOverallTimeoutError(StreamDeadlineError):
    pass


async def _consume_async_stream(
    model: ChatOpenAI,
    prompt: str,
    *,
    on_text: Callable[[str], None] | None,
    on_status: Callable[[str, int], None] | None,
    first_content_timeout: float,
    idle_timeout: float,
    overall_timeout: float,
) -> tuple[str, int, int, int, int | None]:
    started = time.perf_counter()
    iterator = model.astream(prompt).__aiter__()
    parts: list[str] = []
    content_chunks = 0
    event_count = 0
    reasoning_chunks = 0
    first_content_latency_ms: int | None = None
    reasoning_detected = False
    last_content_activity = started

    def deadline_error(kind: type[StreamDeadlineError], message: str) -> StreamDeadlineError:
        return kind(
            message,
            partial_text="".join(parts),
            content_chunks=content_chunks,
            event_count=event_count,
            reasoning_chunks=reasoning_chunks,
            first_content_latency_ms=first_content_latency_ms,
        )

    try:
        while True:
            now = time.perf_counter()
            elapsed = now - started
            overall_remaining = overall_timeout - elapsed
            if overall_remaining <= 0:
                raise deadline_error(StreamOverallTimeoutError, "stream exceeded its overall deadline")
            # Enforce meaningful-activity deadlines even when a provider emits
            # a rapid sequence of empty heartbeat events.
            if first_content_latency_ms is None and not reasoning_detected and elapsed >= first_content_timeout:
                raise deadline_error(FirstContentTimeoutError, "model produced no visible content before deadline")
            if first_content_latency_ms is not None and now - last_content_activity >= idle_timeout:
                raise deadline_error(StreamIdleTimeoutError, "stream stopped producing content")
            next_chunk = asyncio.create_task(iterator.__anext__())
            try:
                while not next_chunk.done():
                    now = time.perf_counter()
                    elapsed = now - started
                    overall_remaining = overall_timeout - elapsed
                    if overall_remaining <= 0:
                        raise deadline_error(StreamOverallTimeoutError, "stream exceeded its overall deadline")
                    if first_content_latency_ms is None:
                        # Once the provider emits an explicit reasoning signal, the
                        # first-content deadline no longer applies. The stream may
                        # continue thinking until the overall hard limit.
                        if reasoning_detected:
                            phase_remaining = overall_remaining
                        else:
                            phase_remaining = first_content_timeout - elapsed
                            if phase_remaining <= 0:
                                raise deadline_error(FirstContentTimeoutError, "model produced no visible content before deadline")
                    else:
                        phase_remaining = idle_timeout - (now - last_content_activity)
                        if phase_remaining <= 0:
                            raise deadline_error(StreamIdleTimeoutError, "stream stopped producing content")
                    done, _ = await asyncio.wait(
                        {next_chunk},
                        timeout=min(2.0, phase_remaining, overall_remaining),
                    )
                    if not done and on_status:
                        phase = "reasoning" if reasoning_detected and first_content_latency_ms is None else "waiting"
                        on_status(phase, int(time.perf_counter() - started))
                chunk = next_chunk.result()
            except StopAsyncIteration:
                break
            finally:
                if not next_chunk.done():
                    next_chunk.cancel()
                    try:
                        await next_chunk
                    except BaseException:
                        pass
            event_count += 1
            text = _chunk_text(getattr(chunk, "content", chunk))
            reasoning = _chunk_has_reasoning_activity(chunk)
            if reasoning:
                reasoning_detected = True
                reasoning_chunks += 1
                if on_status:
                    on_status("reasoning", int(time.perf_counter() - started))
            if reasoning and not text:
                continue
            if not text:
                if on_status:
                    on_status("reasoning" if reasoning_detected else "connected", int(time.perf_counter() - started))
                continue
            if first_content_latency_ms is None:
                first_content_latency_ms = int((time.perf_counter() - started) * 1000)
            last_content_activity = time.perf_counter()
            parts.append(text)
            content_chunks += 1
            if on_text:
                on_text("".join(parts))
    finally:
        close = getattr(iterator, "aclose", None)
        if close:
            try:
                await close()
            except Exception:
                pass
    return "".join(parts).strip(), content_chunks, event_count, reasoning_chunks, first_content_latency_ms


def invoke_text_detailed(
    prompt: str,
    *,
    retry_transient: bool = False,
    fast: bool = False,
) -> LLMCallResult:
    started = time.perf_counter()
    models = get_fast_chat_models() if fast else get_chat_models()
    if not models:
        return LLMCallResult(None, None, 0, 0, "NO_MODEL_CONFIGURED", None)

    attempts = 0
    last_model: str | None = None
    last_error = "EMPTY_RESPONSE"
    last_status: int | None = None
    for model in models:
        last_model = _model_name(model)
        # Keep the whole call bounded: with an explicit fallback provider each
        # provider gets one attempt; without one, the primary may retry a
        # transient failure once. This caps remote attempts at two.
        max_attempts = 2 if retry_transient and len(models) == 1 else 1
        for attempt in range(max_attempts):
            attempts += 1
            try:
                result = model.invoke(prompt)
                content = str(result.content).strip()
                if content:
                    return LLMCallResult(
                        content,
                        last_model,
                        attempts,
                        int((time.perf_counter() - started) * 1000),
                    )
                last_error = "EMPTY_RESPONSE"
                last_status = None
                logger.warning("LLM returned an empty response model=%s attempt=%s", last_model, attempts)
                break
            except SoftTimeLimitExceeded:
                # This is a worker lifecycle signal, not a provider failure. Swallowing
                # it makes Celery tasks continue beyond their configured time limit.
                raise
            except Exception as error:
                last_error = type(error).__name__
                last_status = _status_code(error)
                logger.warning(
                    "LLM invocation failed model=%s attempt=%s error_type=%s status_code=%s",
                    last_model,
                    attempts,
                    last_error,
                    last_status,
                )
                if not retry_transient or not _is_transient(error) or attempt + 1 >= max_attempts:
                    break
                time.sleep(0.35)

    return LLMCallResult(
        None,
        last_model,
        attempts,
        int((time.perf_counter() - started) * 1000),
        last_error,
        last_status,
    )


def invoke_text_stream_detailed(
    prompt: str,
    *,
    on_text: Callable[[str], None] | None = None,
    on_status: Callable[[str, int], None] | None = None,
    retry_transient: bool = False,
) -> LLMCallResult:
    """Stream model output and report the accumulated text to ``on_text``.

    A retry is only attempted when no content has been emitted. Retrying after
    partial output would duplicate or visibly rewind a response in the UI.
    """
    started = time.perf_counter()
    models = get_chat_models()
    if not models:
        return LLMCallResult(None, None, 0, 0, "NO_MODEL_CONFIGURED", None, True, 0)

    attempts = 0
    last_model: str | None = None
    last_error = "EMPTY_RESPONSE"
    last_status: int | None = None
    total_chunks = 0
    total_events = 0
    total_reasoning_chunks = 0
    first_content_latency_ms: int | None = None
    settings = get_settings()
    global_deadline = started + settings.llm_stream_overall_timeout_seconds
    for model in models:
        last_model = _model_name(model)
        max_attempts = 2 if retry_transient and len(models) == 1 else 1
        for attempt in range(max_attempts):
            attempts += 1
            parts: list[str] = []
            chunk_count = 0
            try:
                remaining = global_deadline - time.perf_counter()
                if remaining <= 0:
                    raise StreamOverallTimeoutError("all model attempts exceeded the stream deadline")
                if hasattr(model, "astream"):
                    content, chunk_count, event_count, reasoning_count, first_latency = asyncio.run(
                        _consume_async_stream(
                            model,
                            prompt,
                            on_text=on_text,
                            on_status=on_status,
                            first_content_timeout=min(settings.llm_stream_first_content_timeout_seconds, remaining),
                            idle_timeout=min(settings.llm_stream_idle_timeout_seconds, remaining),
                            overall_timeout=remaining,
                        )
                    )
                    total_chunks += chunk_count
                    total_events += event_count
                    total_reasoning_chunks += reasoning_count
                    first_content_latency_ms = first_content_latency_ms or first_latency
                else:
                    for chunk in model.stream(prompt):
                        total_events += 1
                        text = _chunk_text(getattr(chunk, "content", chunk))
                        if not text:
                            continue
                        parts.append(text)
                        chunk_count += 1
                        total_chunks += 1
                        if first_content_latency_ms is None:
                            first_content_latency_ms = int((time.perf_counter() - started) * 1000)
                        if on_text:
                            on_text("".join(parts))
                    content = "".join(parts).strip()
                if content:
                    return LLMCallResult(
                        content,
                        last_model,
                        attempts,
                        int((time.perf_counter() - started) * 1000),
                        None,
                        None,
                        True,
                        total_chunks,
                        total_events,
                        total_reasoning_chunks,
                        first_content_latency_ms,
                    )
                last_error = "EMPTY_RESPONSE"
                last_status = None
                logger.warning("LLM stream returned no content model=%s attempt=%s", last_model, attempts)
                break
            except SoftTimeLimitExceeded:
                raise
            except StreamDeadlineError as error:
                total_chunks += error.content_chunks
                total_events += error.event_count
                total_reasoning_chunks += error.reasoning_chunks
                first_content_latency_ms = first_content_latency_ms or error.first_content_latency_ms
                last_error = type(error).__name__
                last_status = None
                logger.warning(
                    "LLM stream deadline model=%s attempt=%s chunks=%s events=%s reasoning_chunks=%s error_type=%s",
                    last_model, attempts, error.content_chunks, error.event_count,
                    error.reasoning_chunks, last_error,
                )
                if error.content_chunks:
                    return LLMCallResult(
                        None, last_model, attempts,
                        int((time.perf_counter() - started) * 1000),
                        last_error, None, True, total_chunks, total_events,
                        total_reasoning_chunks, first_content_latency_ms,
                    )
                if retry_transient and attempt + 1 < max_attempts and time.perf_counter() < global_deadline:
                    time.sleep(0.35)
                    continue
                break
            except Exception as error:
                last_error = type(error).__name__
                last_status = _status_code(error)
                logger.warning(
                    "LLM stream failed model=%s attempt=%s chunks=%s error_type=%s status_code=%s",
                    last_model,
                    attempts,
                    chunk_count,
                    last_error,
                    last_status,
                )
                if chunk_count:
                    return LLMCallResult(
                        None,
                        last_model,
                        attempts,
                        int((time.perf_counter() - started) * 1000),
                        last_error,
                        last_status,
                        True,
                        total_chunks,
                        total_events,
                        total_reasoning_chunks,
                        first_content_latency_ms,
                    )
                if retry_transient and _is_transient(error) and attempt + 1 < max_attempts:
                    time.sleep(0.35)
                    continue
                break

    return LLMCallResult(
        None,
        last_model,
        attempts,
        int((time.perf_counter() - started) * 1000),
        last_error,
        last_status,
        True,
        total_chunks,
        total_events,
        total_reasoning_chunks,
        first_content_latency_ms,
    )


def invoke_text(prompt: str) -> str | None:
    return invoke_text_detailed(prompt).text


def invoke_json(prompt: str) -> dict | None:
    text = invoke_text(prompt)
    if not text:
        return None
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        return json.loads(match.group(0)) if match else None
