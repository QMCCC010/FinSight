from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=128)


class UserOut(ORMModel):
    id: int
    username: str
    role: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


class CompanyOut(ORMModel):
    id: int
    stock_code: str
    name: str
    full_name: str | None
    exchange: str | None
    industry: str | None
    tracking_mode: str
    last_queried_at: datetime | None
    last_crawled_at: datetime | None
    last_indexed_at: datetime | None


class KnowledgeStatus(BaseModel):
    company: CompanyOut
    total_documents: int
    by_type: dict[str, int]
    indexed_chunks: int
    is_collecting: bool
    is_fresh: bool


class DocumentOut(ORMModel):
    id: int
    company_id: int
    document_type: str
    title: str
    source_name: str
    source_url: str
    acquisition_mode: str = "LIVE"
    author: str | None
    published_at: datetime | None
    summary: str | None
    parser_version: str | None = None
    parse_quality: float | None = None
    parse_warnings: list[str] = Field(default_factory=list)
    summary_method: str = "UNKNOWN"
    status: str
    error_stage: str | None
    error_message: str | None


class DocumentDetail(DocumentOut):
    parsed_text: str | None
    keywords: list[str]
    structured: dict[str, Any] = Field(default_factory=dict)
    evidences: list[dict[str, Any]] = Field(default_factory=list)


class Citation(BaseModel):
    document_id: int
    title: str
    source_type: str
    source_url: str
    published_at: datetime | None = None
    page: int | None = None
    quote: str


class CreateChatSession(BaseModel):
    title: str = Field(default="新对话", min_length=1, max_length=255)

    @field_validator("title")
    @classmethod
    def normalize_title(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("会话标题不能为空")
        return value


class ChatSessionOut(ORMModel):
    id: int
    title: str
    created_at: datetime
    updated_at: datetime


class CreateMessage(BaseModel):
    question: str = Field(min_length=2, max_length=2000)

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 2:
            raise ValueError("问题至少需要2个字符")
        return value


class ResolveCompanyRequest(BaseModel):
    stock_code: str = Field(pattern=r"^\d{6}$")


class ChatMessageOut(ORMModel):
    id: int
    session_id: int
    question: str | None
    answer: str | None
    status: str
    progress: int
    status_text: str | None
    company_id: int | None
    refresh_run_id: int | None
    refresh_status: str | None
    refresh_status_text: str | None
    refresh_requested_at: datetime | None
    refresh_completed_at: datetime | None
    data_as_of: datetime | None
    confidence: str | None
    citations: list[dict[str, Any]]
    missing_sources: list[str]
    clarification_candidates: list[dict[str, Any]] | None = None
    analysis_metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None
    created_at: datetime
    updated_at: datetime


class ReportComparisonRequest(BaseModel):
    stock_code: str = Field(pattern=r"^\d{6}$")
    comparison_mode: Literal["SELECTED", "CONSENSUS"] = "SELECTED"
    document_ids: list[int] | None = Field(default=None, min_length=2, max_length=10)
    limit: int = Field(default=5, ge=2, le=50)
    institutions: list[str] = Field(default_factory=list, max_length=30)
    normalized_ratings: list[Literal["POSITIVE", "SLIGHTLY_POSITIVE", "NEUTRAL", "SLIGHTLY_NEGATIVE", "NEGATIVE"]] = Field(default_factory=list, max_length=5)
    date_from: datetime | None = None
    date_to: datetime | None = None
    latest_per_institution: bool = True

    @model_validator(mode="after")
    def validate_comparison_scope(self):
        if self.comparison_mode == "SELECTED" and self.limit > 10:
            raise ValueError("精选对比最多分析10篇研报")
        if self.comparison_mode == "CONSENSUS" and self.document_ids:
            raise ValueError("全量共识模式使用筛选条件，不接受手动研报ID")
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("开始日期不能晚于结束日期")
        self.institutions = list(dict.fromkeys(value.strip() for value in self.institutions if value.strip()))
        return self


class CreateReportRequest(BaseModel):
    stock_code: str = Field(pattern=r"^\d{6}$")
    report_type: Literal["COMPANY_BRIEF", "REPORT_COMPARISON"]
    date_from: datetime | None = None
    date_to: datetime | None = None
    source_types: list[str] = Field(default_factory=list)
    document_ids: list[int] | None = Field(default=None, max_length=50)


class UpdateReportRequest(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=255)
    content_markdown: str | None = Field(default=None, min_length=1, max_length=500_000)


class RefreshReportMaterialsRequest(BaseModel):
    source_types: list[str] = Field(default_factory=list)


class ReportOut(ORMModel):
    id: int
    company_id: int
    report_type: str
    title: str
    status: str
    progress: int
    status_text: str | None
    error: str | None
    source_types: list[str]
    selected_document_ids: list[int]
    content_markdown: str
    citations: list[dict[str, Any]]
    completed_at: datetime | None
    created_at: datetime
    updated_at: datetime


class ReportVersionOut(ORMModel):
    id: int
    report_id: int
    version_number: int
    title: str
    content_markdown: str
    citations: list[dict[str, Any]]
    change_type: str
    created_at: datetime


class CrawlRunRequest(BaseModel):
    stock_codes: list[str] = Field(default_factory=list)
    source_types: list[str] = Field(default_factory=list)


class CrawlRunOut(ORMModel):
    id: int
    job_id: str
    company_id: int | None
    source_type: str | None
    source_types: list[str]
    trigger_type: str
    status: str
    stage: str | None
    progress: int
    discovered_count: int
    created_count: int
    duplicate_count: int
    failed_count: int
    missing_sources: list[str]
    source_stats: dict[str, Any] | None = None
    error: str | None
    created_at: datetime


class SourceOut(ORMModel):
    id: int
    name: str
    source_type: str
    adapter: str
    enabled: bool
    schedule: str
    config: dict[str, Any]
    last_attempt_at: datetime | None
    last_success_at: datetime | None
    last_duration_ms: int | None
    total_successes: int
    total_failures: int
    consecutive_failures: int
    last_error: str | None


class SourceUpdate(BaseModel):
    enabled: bool | None = None
    schedule: str | None = Field(default=None, max_length=64)


class TrackingUpdate(BaseModel):
    tracking_mode: Literal["PINNED", "INACTIVE"]


class FavoriteOut(BaseModel):
    company: CompanyOut
    created_at: datetime
