from __future__ import annotations

import json
import re
from functools import lru_cache

from billiard.exceptions import SoftTimeLimitExceeded
from langchain_openai import ChatOpenAI

from app.core.config import get_settings


def _create_model(*, base_url: str, api_key: str, model: str) -> ChatOpenAI | None:
    settings = get_settings()
    if not api_key or not model:
        return None
    kwargs = {
        "api_key": api_key,
        "model": model,
        "temperature": settings.llm_temperature,
        # Chat/worker tasks have a 120-second soft limit. Keep each provider
        # attempt bounded and let the explicit fallback model provide the one
        # reliability retry; nested SDK retries can otherwise outlive the task.
        "timeout": min(settings.llm_timeout_seconds, 45),
        "max_retries": 0,
    }
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


@lru_cache
def get_chat_models() -> tuple[ChatOpenAI, ...]:
    settings = get_settings()
    models = [
        _create_model(base_url=settings.llm_base_url, api_key=settings.llm_api_key, model=settings.llm_model),
        _create_model(base_url=settings.fallback_llm_base_url, api_key=settings.fallback_llm_api_key, model=settings.fallback_llm_model),
    ]
    return tuple(model for model in models if model is not None)


def get_chat_model() -> ChatOpenAI | None:
    models = get_chat_models()
    return models[0] if models else None


def invoke_text(prompt: str) -> str | None:
    for model in get_chat_models():
        try:
            result = model.invoke(prompt)
            content = str(result.content).strip()
            if content:
                return content
        except SoftTimeLimitExceeded:
            # This is a worker lifecycle signal, not a provider failure. Swallowing
            # it makes Celery tasks continue beyond their configured time limit.
            raise
        except Exception:
            # The caller has a deterministic evidence-only fallback. Do not leak
            # provider errors or credentials into user-facing answers.
            continue
    return None


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
