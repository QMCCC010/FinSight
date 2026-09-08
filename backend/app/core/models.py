from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, ForeignKey, Index, Integer, LargeBinary, String, Text, UniqueConstraint, func
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


LONG_TEXT = Text().with_variant(LONGTEXT(), "mysql")


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class User(Base, TimestampMixin):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(512))
    role: Mapped[str] = mapped_column(String(20), index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class Company(Base, TimestampMixin):
    __tablename__ = "companies"
    id: Mapped[int] = mapped_column(primary_key=True)
    stock_code: Mapped[str] = mapped_column(String(16), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(128), index=True)
    full_name: Mapped[str | None] = mapped_column(String(255))
    exchange: Mapped[str | None] = mapped_column(String(16))
    industry: Mapped[str | None] = mapped_column(String(128), index=True)
    aliases: Mapped[list[str]] = mapped_column(JSON, default=list)
    tracking_mode: Mapped[str] = mapped_column(String(20), default="INACTIVE", index=True)
    last_queried_at: Mapped[datetime | None] = mapped_column(DateTime)
    last_crawled_at: Mapped[datetime | None] = mapped_column(DateTime)
    last_indexed_at: Mapped[datetime | None] = mapped_column(DateTime)


class DataSource(Base, TimestampMixin):
    __tablename__ = "data_sources"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(128), unique=True)
    source_type: Mapped[str] = mapped_column(String(32), index=True)
    adapter: Mapped[str] = mapped_column(String(128))
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    schedule: Mapped[str] = mapped_column(String(64))
    config: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    last_attempt_at: Mapped[datetime | None] = mapped_column(DateTime)
    last_success_at: Mapped[datetime | None] = mapped_column(DateTime)
    last_duration_ms: Mapped[int | None] = mapped_column(Integer)
    total_successes: Mapped[int] = mapped_column(Integer, default=0)
    total_failures: Mapped[int] = mapped_column(Integer, default=0)
    consecutive_failures: Mapped[int] = mapped_column(Integer, default=0)
    last_error: Mapped[str | None] = mapped_column(Text)


class CrawlRun(Base, TimestampMixin):
    __tablename__ = "crawl_runs"
    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    company_id: Mapped[int | None] = mapped_column(ForeignKey("companies.id"), index=True)
    source_type: Mapped[str | None] = mapped_column(String(32), index=True)
    source_types: Mapped[list[str]] = mapped_column(JSON, default=list)
    trigger_type: Mapped[str] = mapped_column(String(20), index=True)
    requested_by_user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"))
    parent_message_id: Mapped[int | None] = mapped_column(Integer, index=True)
    status: Mapped[str] = mapped_column(String(20), default="QUEUED", index=True)
    stage: Mapped[str | None] = mapped_column(String(64))
    progress: Mapped[int] = mapped_column(Integer, default=0)
    discovered_count: Mapped[int] = mapped_column(Integer, default=0)
    created_count: Mapped[int] = mapped_column(Integer, default=0)
    duplicate_count: Mapped[int] = mapped_column(Integer, default=0)
    failed_count: Mapped[int] = mapped_column(Integer, default=0)
    missing_sources: Mapped[list[str]] = mapped_column(JSON, default=list)
    source_stats: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime)


class Document(Base, TimestampMixin):
    __tablename__ = "documents"
    __table_args__ = (UniqueConstraint("content_hash", name="uq_document_content_hash"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id"), index=True)
    source_id: Mapped[int | None] = mapped_column(ForeignKey("data_sources.id"))
    document_type: Mapped[str] = mapped_column(String(32), index=True)
    title: Mapped[str] = mapped_column(String(512))
    source_name: Mapped[str] = mapped_column(String(128))
    source_url: Mapped[str] = mapped_column(String(2048))
    acquisition_mode: Mapped[str] = mapped_column(String(16), default="LIVE", index=True)
    author: Mapped[str | None] = mapped_column(String(255))
    published_at: Mapped[datetime | None] = mapped_column(DateTime, index=True)
    report_period: Mapped[str | None] = mapped_column(String(64))
    raw_path: Mapped[str | None] = mapped_column(String(1024))
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    raw_text: Mapped[str | None] = mapped_column(LONG_TEXT)
    parsed_text: Mapped[str | None] = mapped_column(LONG_TEXT)
    summary: Mapped[str | None] = mapped_column(LONG_TEXT)
    keywords: Mapped[list[str]] = mapped_column(JSON, default=list)
    status: Mapped[str] = mapped_column(String(20), default="DISCOVERED", index=True)
    error_stage: Mapped[str | None] = mapped_column(String(64))
    error_message: Mapped[str | None] = mapped_column(Text)
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False)


class DocumentChunk(Base, TimestampMixin):
    __tablename__ = "document_chunks"
    __table_args__ = (UniqueConstraint("document_id", "chunk_index", name="uq_document_chunk"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id"), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    page_number: Mapped[int | None] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int | None] = mapped_column(Integer)
    vector_id: Mapped[int | None] = mapped_column(Integer, unique=True)


class FinancialMetric(Base, TimestampMixin):
    __tablename__ = "financial_metrics"
    id: Mapped[int] = mapped_column(primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id"), index=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    metric_name: Mapped[str] = mapped_column(String(64), index=True)
    raw_value: Mapped[str | None] = mapped_column(String(128))
    normalized_value: Mapped[float | None] = mapped_column(Float)
    unit: Mapped[str | None] = mapped_column(String(32))
    currency: Mapped[str | None] = mapped_column(String(16))
    period: Mapped[str | None] = mapped_column(String(64))
    yoy: Mapped[float | None] = mapped_column(Float)


class EarningsForecast(Base, TimestampMixin):
    __tablename__ = "earnings_forecasts"
    id: Mapped[int] = mapped_column(primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id"), index=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    institution: Mapped[str | None] = mapped_column(String(128))
    forecast_year: Mapped[int] = mapped_column(Integer, index=True)
    revenue: Mapped[float | None] = mapped_column(Float)
    net_profit: Mapped[float | None] = mapped_column(Float)
    eps: Mapped[float | None] = mapped_column(Float)
    unit: Mapped[str | None] = mapped_column(String(32))


class InvestmentRating(Base, TimestampMixin):
    __tablename__ = "investment_ratings"
    id: Mapped[int] = mapped_column(primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id"), index=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), unique=True)
    institution: Mapped[str | None] = mapped_column(String(128))
    analyst: Mapped[str | None] = mapped_column(String(128))
    original_rating: Mapped[str | None] = mapped_column(String(64))
    normalized_rating: Mapped[str | None] = mapped_column(String(32), index=True)
    previous_rating: Mapped[str | None] = mapped_column(String(64))
    target_price: Mapped[float | None] = mapped_column(Float)
    currency: Mapped[str | None] = mapped_column(String(16))


class Opinion(Base, TimestampMixin):
    __tablename__ = "opinions"
    id: Mapped[int] = mapped_column(primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id"), index=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    opinion_type: Mapped[str] = mapped_column(String(32), index=True)
    content: Mapped[str] = mapped_column(Text)
    sentiment: Mapped[str | None] = mapped_column(String(32))
    confidence: Mapped[float | None] = mapped_column(Float)


class RiskItem(Base, TimestampMixin):
    __tablename__ = "risk_items"
    id: Mapped[int] = mapped_column(primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id"), index=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    category: Mapped[str] = mapped_column(String(64), index=True)
    content: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float | None] = mapped_column(Float)


class SentimentResult(Base, TimestampMixin):
    __tablename__ = "sentiment_results"
    id: Mapped[int] = mapped_column(primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id"), index=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    sentiment: Mapped[str] = mapped_column(String(32), index=True)
    score: Mapped[float] = mapped_column(Float, default=0)
    reason: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[float | None] = mapped_column(Float)


class MarketPrice(Base, TimestampMixin):
    __tablename__ = "market_prices"
    __table_args__ = (UniqueConstraint("company_id", "trade_date", name="uq_market_price_company_date"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id"), index=True)
    trade_date: Mapped[datetime] = mapped_column(DateTime, index=True)
    open: Mapped[float | None] = mapped_column(Float)
    close: Mapped[float | None] = mapped_column(Float)
    high: Mapped[float | None] = mapped_column(Float)
    low: Mapped[float | None] = mapped_column(Float)
    volume: Mapped[float | None] = mapped_column(Float)
    amount: Mapped[float | None] = mapped_column(Float)
    change_pct: Mapped[float | None] = mapped_column(Float)
    source_name: Mapped[str] = mapped_column(String(64), default="东方财富")


class Evidence(Base, TimestampMixin):
    __tablename__ = "evidences"
    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), index=True)
    chunk_id: Mapped[int | None] = mapped_column(ForeignKey("document_chunks.id"))
    entity_type: Mapped[str] = mapped_column(String(64), index=True)
    entity_id: Mapped[int | None] = mapped_column(Integer)
    page_number: Mapped[int | None] = mapped_column(Integer)
    quote: Mapped[str] = mapped_column(Text)


class ChatSession(Base, TimestampMixin):
    __tablename__ = "chat_sessions"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    title: Mapped[str] = mapped_column(String(255), default="新对话")


class ChatMessage(Base, TimestampMixin):
    __tablename__ = "chat_messages"
    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("chat_sessions.id"), index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    role: Mapped[str] = mapped_column(String(16))
    question: Mapped[str | None] = mapped_column(LONG_TEXT)
    answer: Mapped[str | None] = mapped_column(LONG_TEXT)
    status: Mapped[str] = mapped_column(String(32), default="QUEUED", index=True)
    progress: Mapped[int] = mapped_column(Integer, default=0)
    status_text: Mapped[str | None] = mapped_column(String(255))
    company_id: Mapped[int | None] = mapped_column(ForeignKey("companies.id"))
    crawl_run_id: Mapped[int | None] = mapped_column(ForeignKey("crawl_runs.id", use_alter=True))
    refresh_run_id: Mapped[int | None] = mapped_column(ForeignKey("crawl_runs.id", use_alter=True), index=True)
    refresh_status: Mapped[str | None] = mapped_column(String(20), index=True)
    refresh_status_text: Mapped[str | None] = mapped_column(String(255))
    refresh_requested_at: Mapped[datetime | None] = mapped_column(DateTime)
    refresh_completed_at: Mapped[datetime | None] = mapped_column(DateTime)
    task_id: Mapped[str | None] = mapped_column(String(64), index=True)
    data_as_of: Mapped[datetime | None] = mapped_column(DateTime)
    confidence: Mapped[str | None] = mapped_column(String(16))
    citations: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    missing_sources: Mapped[list[str]] = mapped_column(JSON, default=list)
    clarification_candidates: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON, default=list)
    # User-visible trace metadata. This stores explainable decisions rather
    # than hidden model reasoning: entity, intent, retrieval and validation.
    analysis_metadata: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    error: Mapped[str | None] = mapped_column(Text)


class AgentCheckpoint(Base, TimestampMixin):
    __tablename__ = "agent_checkpoints"
    id: Mapped[int] = mapped_column(primary_key=True)
    message_id: Mapped[int] = mapped_column(ForeignKey("chat_messages.id"), unique=True)
    current_node: Mapped[str] = mapped_column(String(64))
    state: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class Favorite(Base, TimestampMixin):
    __tablename__ = "favorites"
    __table_args__ = (UniqueConstraint("user_id", "company_id", name="uq_user_favorite"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id"), index=True)


class GeneratedReport(Base, TimestampMixin):
    __tablename__ = "generated_reports"
    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    company_id: Mapped[int] = mapped_column(ForeignKey("companies.id"), index=True)
    report_type: Mapped[str] = mapped_column(String(64))
    title: Mapped[str] = mapped_column(String(255))
    date_from: Mapped[datetime | None] = mapped_column(DateTime)
    date_to: Mapped[datetime | None] = mapped_column(DateTime)
    status: Mapped[str] = mapped_column(String(20), default="QUEUED", index=True)
    progress: Mapped[int] = mapped_column(Integer, default=0)
    status_text: Mapped[str | None] = mapped_column(String(255))
    error: Mapped[str | None] = mapped_column(Text)
    source_types: Mapped[list[str]] = mapped_column(JSON, default=list)
    selected_document_ids: Mapped[list[int]] = mapped_column(JSON, default=list)
    content_markdown: Mapped[str] = mapped_column(LONG_TEXT, default="")
    citations: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime)
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False, index=True)


class ReportVersion(Base, TimestampMixin):
    __tablename__ = "report_versions"
    __table_args__ = (UniqueConstraint("report_id", "version_number", name="uq_report_version"),)
    id: Mapped[int] = mapped_column(primary_key=True)
    report_id: Mapped[int] = mapped_column(ForeignKey("generated_reports.id"), index=True)
    version_number: Mapped[int] = mapped_column(Integer)
    title: Mapped[str] = mapped_column(String(255))
    content_markdown: Mapped[str] = mapped_column(LONG_TEXT)
    citations: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    change_type: Mapped[str] = mapped_column(String(32), default="MANUAL")


Index("ix_documents_company_type_date", Document.company_id, Document.document_type, Document.published_at)
Index("ix_chunks_company_document", DocumentChunk.company_id, DocumentChunk.document_id)
Index("ix_market_prices_company_date", MarketPrice.company_id, MarketPrice.trade_date)
