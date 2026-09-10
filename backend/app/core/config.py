from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "金融研报智能分析系统"
    app_env: str = "development"
    app_secret_key: str = "development-only-secret-key-change-me"
    internal_service_token: str = "development-internal"
    database_url: str = "sqlite:///./finresearch.db"
    redis_url: str = "redis://localhost:6380/0"
    celery_broker_url: str = "redis://localhost:6380/1"
    celery_result_backend: str = "redis://localhost:6380/2"
    cors_origins: list[str] | str = ["http://localhost:5173"]
    storage_root: Path = Path("data")
    collector_url: str = "http://localhost:5001"
    collection_mode: str = "AUTO"

    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    llm_temperature: float = 0.1
    llm_thinking_mode: str = "enabled"
    llm_reasoning_effort: str = "low"
    router_thinking_mode: str = "disabled"
    llm_timeout_seconds: int = 45
    llm_max_retries: int = 0
    llm_stream_first_content_timeout_seconds: int = 180
    llm_stream_idle_timeout_seconds: int = 60
    llm_stream_overall_timeout_seconds: int = 480
    fallback_llm_base_url: str = ""
    fallback_llm_api_key: str = ""
    fallback_llm_model: str = ""
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_dimension: int = 512
    enable_local_embeddings: bool = False
    embedding_cache_dir: Path = Path("/root/.cache/huggingface/fastembed")
    vector_store_backend: str = "milvus"
    vector_store_fallback: bool = True
    milvus_uri: str = "http://localhost:19530"
    milvus_token: str = ""
    milvus_collection_name: str = "financial_chunks_v1"
    milvus_memory_collection_name: str = "chat_memory_v1"
    milvus_timeout_seconds: int = 30
    conversation_history_token_budget: int = 4000
    conversation_recent_turns: int = 4
    conversation_recall_limit: int = 2
    conversation_summary_trigger_turns: int = 10
    router_history_token_budget: int = 1200
    router_recent_turns: int = 3
    answer_history_token_budget: int = 2500
    answer_recent_turns: int = 4

    seed_admin_username: str = "admin"
    seed_admin_password: str = "Admin123!"
    seed_analyst_username: str = "analyst"
    seed_analyst_password: str = "Analyst123!"
    jwt_expire_minutes: int = 480

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value

    @field_validator("collection_mode")
    @classmethod
    def normalize_mode(cls, value: str) -> str:
        mode = value.upper()
        if mode not in {"LIVE", "SNAPSHOT", "AUTO"}:
            raise ValueError("COLLECTION_MODE must be LIVE, SNAPSHOT or AUTO")
        return mode

    @field_validator("vector_store_backend")
    @classmethod
    def normalize_vector_store_backend(cls, value: str) -> str:
        backend = value.lower()
        if backend not in {"faiss", "milvus"}:
            raise ValueError("VECTOR_STORE_BACKEND must be faiss or milvus")
        return backend

    @field_validator("llm_thinking_mode", "router_thinking_mode")
    @classmethod
    def normalize_thinking_mode(cls, value: str) -> str:
        mode = value.lower()
        if mode not in {"enabled", "disabled"}:
            raise ValueError("thinking mode must be enabled or disabled")
        return mode

    @field_validator("llm_reasoning_effort")
    @classmethod
    def normalize_reasoning_effort(cls, value: str) -> str:
        effort = value.lower()
        if effort not in {"low", "high", "max"}:
            raise ValueError("LLM_REASONING_EFFORT must be low, high or max")
        return effort

    @field_validator(
        "conversation_history_token_budget",
        "conversation_recent_turns",
        "conversation_recall_limit",
        "conversation_summary_trigger_turns",
        "router_history_token_budget",
        "router_recent_turns",
        "answer_history_token_budget",
        "answer_recent_turns",
        "llm_stream_first_content_timeout_seconds",
        "llm_stream_idle_timeout_seconds",
        "llm_stream_overall_timeout_seconds",
    )
    @classmethod
    def positive_memory_limits(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("conversation memory limits must be positive")
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
