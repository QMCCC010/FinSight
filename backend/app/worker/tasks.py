from __future__ import annotations

from datetime import datetime, timedelta
import logging
from time import perf_counter
from uuid import uuid4

from celery import shared_task
from billiard.exceptions import SoftTimeLimitExceeded
from redis import Redis
from sqlalchemy import delete, func, select

from app.ai.extraction import extract_document, persist_extraction
from app.ai.graph import run_message
from app.ai.index import index_document, rebuild_index, remove_document_from_index
from app.collector.adapters import ADAPTERS
from app.core.config import get_settings
from app.core.database import SessionLocal
from app.core.enums import DocumentStatus, MessageStatus, RunStatus, TrackingMode, TriggerType
from app.core.models import ChatMessage, Company, CrawlRun, DataSource, Document, DocumentChunk, GeneratedReport, MarketPrice, ReportVersion
from app.services.parser import chunk_pages, parse_stored
from app.services.storage import content_hash, save_raw
from app.services.crawl_runs import normalize_source_types, run_covers, run_source_types

settings = get_settings()
logger = logging.getLogger(__name__)
LIMITS = {"RESEARCH_REPORT": 10, "NEWS": 20, "ANNOUNCEMENT": 10, "SOCIAL": 30}


def _process_document(db, document: Document) -> None:
    try:
        text, pages = parse_stored(document.raw_path) if document.raw_path else (document.raw_text or "", [(1, document.raw_text or "")])
        if not text.strip():
            raise ValueError("解析正文为空")
        document.parsed_text = text
        document.status = DocumentStatus.PARSED
        db.commit()
        result = extract_document(document)
        persist_extraction(db, document, result)
        document.status = DocumentStatus.EXTRACTED
        db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document.id))
        db.flush()
        for index, (page, chunk) in enumerate(chunk_pages(pages)):
            db.add(DocumentChunk(document_id=document.id, company_id=document.company_id, chunk_index=index, page_number=page, content=chunk, token_count=len(chunk)))
        document.status = DocumentStatus.EXTRACTED
        document.error_stage = None
        document.error_message = None
        db.commit()
        index_document(db, document.id)
        document = db.get(Document, document.id)
        document.status = DocumentStatus.INDEXED
        db.commit()
    except Exception as exc:
        db.rollback()
        document = db.get(Document, document.id)
        document.status = DocumentStatus.FAILED
        document.error_stage = "PROCESSING"
        document.error_message = str(exc)[:2000]
        db.commit()
        raise


@shared_task(name="app.worker.tasks.process_document", bind=True, max_retries=3)
def process_document(self, document_id: int):
    try:
        with SessionLocal() as db:
            document = db.get(Document, document_id)
            if not document:
                return {"status": "missing"}
            _process_document(db, document)
            if settings.vector_store_backend == "faiss":
                rebuild_index(db)
            return {"status": "indexed", "document_id": document_id}
    except Exception as exc:
        raise self.retry(exc=exc, countdown=[5, 30, 120][min(self.request.retries, 2)])


@shared_task(name="app.worker.tasks.collect_company")
def collect_company(run_id: int, source_types: list[str] | None = None):
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    with SessionLocal() as db:
        run = db.get(CrawlRun, run_id)
        if not run:
            return {"status": "missing"}
        if run.status in (RunStatus.COMPLETED, RunStatus.PARTIAL, RunStatus.FAILED):
            return {"status": "stale_delivery_ignored", "run_status": str(run.status)}
        source_types = normalize_source_types(source_types) or run_source_types(run) or None
        company = db.get(Company, run.company_id)
        lock_key = f"crawl:lock:{company.stock_code}"
        try:
            # Keep the lock longer than Celery's hard limit so another worker
            # cannot enter during the final minute of a slow PDF/LLM pipeline.
            acquired = redis.set(lock_key, run.job_id, nx=True, ex=1200)
        except Exception:
            acquired = True
        if not acquired:
            active_job_id = redis.get(lock_key)
            active = db.scalar(select(CrawlRun).where(CrawlRun.job_id == active_job_id)) if active_job_id else None
            run.status = RunStatus.COMPLETED
            run.stage = "已复用同公司运行中的采集任务"
            run.progress = 100
            run.finished_at = datetime.now()
            foreground = list(db.scalars(select(ChatMessage).where(
                ChatMessage.crawl_run_id == run.id,
                ChatMessage.status == MessageStatus.COLLECTING,
            )).all())
            refreshes = list(db.scalars(select(ChatMessage).where(
                ChatMessage.refresh_run_id == run.id,
                ChatMessage.refresh_status.in_(["QUEUED", "RUNNING"]),
            )).all())
            if active and active.status in (RunStatus.QUEUED, RunStatus.RUNNING):
                for message in foreground:
                    message.crawl_run_id = active.id
                    message.status_text = "正在复用同公司运行中的采集任务"
                for message in refreshes:
                    message.refresh_run_id = active.id
                    message.refresh_status = "RUNNING"
                    message.refresh_status_text = "正在复用同公司运行中的后台更新"
            elif active and active.status in (RunStatus.COMPLETED, RunStatus.PARTIAL):
                for message in foreground:
                    message.missing_sources = active.missing_sources or []
                    message.status = MessageStatus.QUEUED
                    message.status_text = "采集完成，正在恢复原问题"
                for message in refreshes:
                    message.refresh_status = str(active.status)
                    message.refresh_status_text = "后台更新完成" if active.status == RunStatus.COMPLETED else "后台更新部分完成"
                    message.refresh_completed_at = datetime.now()
            else:
                for message in foreground:
                    message.status = MessageStatus.FAILED
                    message.progress = 100
                    message.status_text = "无法关联同公司的采集任务"
                    message.error = "采集锁状态异常，请重新执行。"
                for message in refreshes:
                    message.refresh_status = "FAILED"
                    message.refresh_status_text = "后台更新任务关联失败"
                    message.refresh_completed_at = datetime.now()
            db.commit()
            if active and active.status in (RunStatus.COMPLETED, RunStatus.PARTIAL):
                for message in foreground:
                    result = run_agent_message.apply_async(args=[message.id], queue="chat", priority=9, expires=300)
                    message.task_id = result.id
                db.commit()
            return {"status": "deduplicated"}
        run.status = RunStatus.RUNNING
        run.started_at = datetime.now()
        run.stage = "连接数据源"
        run.progress = 5
        for message in db.scalars(select(ChatMessage).where(
            ChatMessage.refresh_run_id == run.id,
            ChatMessage.refresh_status == "QUEUED",
        )).all():
            message.refresh_status = "RUNNING"
            message.refresh_status_text = "后台正在更新相关资料"
        db.commit()
        source_query = select(DataSource).where(DataSource.enabled.is_(True))
        if source_types:
            source_query = source_query.where(DataSource.source_type.in_(source_types))
        sources = list(db.scalars(source_query).all())
        missing: list[str] = []
        source_stats: dict[str, dict] = {}
        try:
            for source_index, source in enumerate(sources):
                source_started = perf_counter()
                source.last_attempt_at = datetime.now()
                run.stage = f"采集{source.name}"
                run.progress = 10 + int(source_index / max(1, len(sources)) * 60)
                db.commit()
                adapter_cls = ADAPTERS.get(source.source_type)
                if not adapter_cls:
                    missing.append(source.source_type)
                    continue
                try:
                    items = adapter_cls().collect(company.stock_code, company.name, LIMITS[source.source_type])
                    run.discovered_count += len(items)
                    if not items:
                        missing.append(source.source_type)
                    for item_index, item in enumerate(items, 1):
                        run.stage = f"{source.name}：处理 {item_index}/{len(items)}"
                        run.progress = min(75, 10 + int((source_index + item_index / max(1, len(items))) / max(1, len(sources)) * 60))
                        db.commit()
                        raw = item.raw_bytes if item.raw_bytes is not None else (item.raw_text or "")
                        digest = content_hash(f"{company.stock_code}|{item.source_url}|{content_hash(raw)}")
                        if db.scalar(select(Document.id).where(Document.content_hash == digest)):
                            run.duplicate_count += 1
                            continue
                        path = save_raw(company.stock_code, item.document_type, item.title, raw, item.extension)
                        document = Document(company_id=company.id, source_id=source.id, document_type=item.document_type, title=item.title, source_name=item.source_name, source_url=item.source_url, acquisition_mode="SNAPSHOT" if item.source_url.startswith("snapshot://") else "LIVE", author=item.author, published_at=item.published_at, raw_path=path, raw_text=item.raw_text, content_hash=digest, status=DocumentStatus.DOWNLOADED)
                        db.add(document)
                        db.commit()
                        try:
                            _process_document(db, document)
                            run.created_count += 1
                        except Exception:
                            run.failed_count += 1
                    source.last_success_at = datetime.now()
                    source.last_duration_ms = int((perf_counter() - source_started) * 1000)
                    source.total_successes = (source.total_successes or 0) + 1
                    source.consecutive_failures = 0
                    source.last_error = None
                    source_stats[source.source_type] = {
                        "status": "SUCCESS" if items else "EMPTY",
                        "received": len(items),
                        "duration_ms": source.last_duration_ms,
                    }
                    run.source_stats = source_stats
                    db.commit()
                except Exception as exc:
                    source.last_duration_ms = int((perf_counter() - source_started) * 1000)
                    source.total_failures = (source.total_failures or 0) + 1
                    source.consecutive_failures = (source.consecutive_failures or 0) + 1
                    source.last_error = str(exc)[:2000]
                    missing.append(source.source_type)
                    run.failed_count += 1
                    source_stats[source.source_type] = {
                        "status": "FAILED",
                        "received": 0,
                        "duration_ms": source.last_duration_ms,
                        "error": source.last_error,
                    }
                    run.source_stats = source_stats
                    db.commit()
            run.stage = "校验RAG索引"
            run.progress = 80
            db.commit()
            if settings.vector_store_backend == "faiss":
                rebuild_index(db)
            company.last_crawled_at = datetime.now()
            company.last_indexed_at = datetime.now()
            run.missing_sources = list(dict.fromkeys(missing))
            run.status = RunStatus.PARTIAL if missing else RunStatus.COMPLETED
            run.stage = "采集与索引完成"
            run.progress = 100
            run.finished_at = datetime.now()
            db.commit()
            refreshes = list(db.scalars(select(ChatMessage).where(
                ChatMessage.refresh_run_id == run.id,
                ChatMessage.refresh_status.in_(["QUEUED", "RUNNING"]),
            )).all())
            for message in refreshes:
                message.refresh_status = str(run.status)
                message.refresh_status_text = "后台更新完成" if run.status == RunStatus.COMPLETED else "后台更新部分完成"
                message.refresh_completed_at = datetime.now()
            indexed_count = db.scalar(select(func.count(Document.id)).where(
                Document.company_id == company.id,
                Document.status == DocumentStatus.INDEXED,
                Document.is_deleted.is_(False),
            )) or 0
            waiting = list(db.scalars(select(ChatMessage).where(
                ChatMessage.crawl_run_id == run.id,
                ChatMessage.status == MessageStatus.COLLECTING,
            )).all())
            for message in waiting:
                message.missing_sources = run.missing_sources
                if indexed_count:
                    message.status = MessageStatus.QUEUED
                    message.status_text = "采集完成，正在恢复原问题"
                else:
                    message.status = MessageStatus.FAILED
                    message.progress = 100
                    message.status_text = "未采集到可用资料"
                    message.error = "所有可用来源均未形成可索引证据，请稍后重试或检查数据源。"
            db.commit()
            for message in (item for item in waiting if item.status == MessageStatus.QUEUED):
                result = run_agent_message.apply_async(args=[message.id], queue="chat", priority=9, expires=300)
                message.task_id = result.id
            db.commit()
            return {"status": run.status, "created": run.created_count, "duplicates": run.duplicate_count, "missing_sources": run.missing_sources}
        except Exception as exc:
            db.rollback()
            run = db.get(CrawlRun, run_id)
            run.status = RunStatus.FAILED
            run.error = str(exc)[:2000]
            run.stage = "采集失败"
            run.progress = 100
            run.finished_at = datetime.now()
            waiting = list(db.scalars(select(ChatMessage).where(
                ChatMessage.crawl_run_id == run.id,
                ChatMessage.status == MessageStatus.COLLECTING,
            )).all())
            for message in waiting:
                message.status = MessageStatus.FAILED
                message.progress = 100
                message.status_text = "自动采集失败"
                message.error = "自动采集任务失败或超时，请点击重新执行。"
            refreshes = list(db.scalars(select(ChatMessage).where(
                ChatMessage.refresh_run_id == run.id,
                ChatMessage.refresh_status.in_(["QUEUED", "RUNNING"]),
            )).all())
            for message in refreshes:
                message.refresh_status = "FAILED"
                message.refresh_status_text = "后台更新失败，当前回答仍保留"
                message.refresh_completed_at = datetime.now()
            db.commit()
            return {"status": "FAILED", "error": str(exc)}
        finally:
            try:
                if redis.get(lock_key) == run.job_id:
                    redis.delete(lock_key)
            except Exception:
                pass


@shared_task(name="app.worker.tasks.rebuild_vector_index")
def rebuild_vector_index():
    with SessionLocal() as db:
        return {"indexed_chunks": rebuild_index(db)}


@shared_task(name="app.worker.tasks.delete_from_vector_index")
def delete_from_vector_index(document_id: int):
    remove_document_from_index(document_id)
    return {"document_id": document_id, "deleted": True}


@shared_task(name="app.worker.tasks.run_agent_message")
def run_agent_message(message_id: int):
    try:
        with SessionLocal() as db:
            message = db.get(ChatMessage, message_id)
            if not message or message.status == MessageStatus.CANCELLED:
                return {"status": "cancelled", "message_id": message_id}
        return run_message(message_id)
    except SoftTimeLimitExceeded:
        with SessionLocal() as db:
            message = db.get(ChatMessage, message_id)
            if message and message.status != MessageStatus.CANCELLED:
                message.status = MessageStatus.FAILED
                message.progress = 100
                message.status_text = "问答执行超时"
                message.error = "问答任务超过2分钟，请重试。"
                db.commit()
        return {"status": "FAILED", "error": "soft_time_limit"}
    except Exception as exc:
        logger.exception("Agent task failed for message %s", message_id)
        with SessionLocal() as db:
            message = db.get(ChatMessage, message_id)
            if message and message.status != MessageStatus.CANCELLED:
                message.status = MessageStatus.FAILED
                message.progress = 100
                message.status_text = "Agent执行失败"
                message.error = "智能分析任务执行失败，请稍后重试；如持续失败，请联系管理员查看服务日志。"
                db.commit()
        return {"status": "FAILED", "error": "agent_execution_failed"}


class ReportGenerationCancelled(Exception):
    pass


@shared_task(name="app.worker.tasks.generate_report")
def generate_report(report_id: int):
    try:
        with SessionLocal() as db:
            report = db.get(GeneratedReport, report_id)
            if not report or report.is_deleted:
                return {"status": "missing"}
            if report.status == "CANCELLED":
                return {"status": "cancelled"}
            company = db.get(Company, report.company_id)
            report.status = "RUNNING"
            report.progress = 5
            report.status_text = "正在准备资料"
            report.error = None
            db.commit()

            def update_progress(value: int, text: str) -> None:
                db.refresh(report)
                if report.status == "CANCELLED":
                    raise ReportGenerationCancelled()
                report.progress = value
                report.status_text = text
                db.commit()

            from app.ai.reporting import build_report

            content, citations = build_report(
                db,
                company,
                report.report_type,
                report.date_from,
                report.date_to,
                report.source_types or [],
                report.selected_document_ids or None,
                update_progress,
            )
            db.refresh(report)
            if report.status == "CANCELLED":
                return {"status": "cancelled"}
            report.content_markdown = content
            report.citations = citations
            report.status = "COMPLETED" if citations else "PARTIAL"
            report.progress = 100
            report.status_text = "报告生成完成" if citations else "报告已生成，但可引用资料不足"
            report.completed_at = datetime.now()
            latest = db.scalar(select(func.max(ReportVersion.version_number)).where(ReportVersion.report_id == report.id)) or 0
            db.add(ReportVersion(
                report_id=report.id,
                version_number=latest + 1,
                title=report.title,
                content_markdown=content,
                citations=citations,
                change_type="GENERATED",
            ))
            db.commit()
            return {"status": report.status, "report_id": report.id, "citations": len(citations)}
    except ReportGenerationCancelled:
        return {"status": "cancelled", "report_id": report_id}
    except SoftTimeLimitExceeded:
        with SessionLocal() as db:
            report = db.get(GeneratedReport, report_id)
            if report and report.status != "CANCELLED":
                report.status = "FAILED"
                report.progress = 100
                report.status_text = "报告生成超时"
                report.error = "报告生成超过允许时间，请缩小资料范围后重试。"
                db.commit()
        return {"status": "FAILED", "error": "soft_time_limit"}
    except Exception as exc:
        logger.exception("Report generation failed for report %s", report_id)
        with SessionLocal() as db:
            report = db.get(GeneratedReport, report_id)
            if report and report.status != "CANCELLED":
                report.status = "FAILED"
                report.progress = 100
                report.status_text = "报告生成失败"
                report.error = "报告生成失败，请调整资料范围后重试；如持续失败，请联系管理员查看服务日志。"
                db.commit()
        return {"status": "FAILED", "error": "report_generation_failed"}


@shared_task(name="app.worker.tasks.dispatch_tracked")
def dispatch_tracked(source_types: list[str]):
    cutoff = datetime.now() - timedelta(days=7)
    with SessionLocal() as db:
        companies = db.scalars(select(Company).where((Company.tracking_mode.in_([TrackingMode.SEED, TrackingMode.PINNED])) | ((Company.tracking_mode == TrackingMode.ON_DEMAND) & (Company.last_queried_at >= cutoff)))).all()
        count = 0
        skipped = 0
        requested = set(source_types)
        # Scheduled batches may legitimately wait while a PDF/LLM-heavy company
        # finishes. Keep deduplication across that window instead of creating a
        # fresh copy every 30 minutes.
        active_cutoff = datetime.now() - timedelta(hours=4)
        for company in companies:
            active_runs = list(db.scalars(select(CrawlRun).where(
                CrawlRun.company_id == company.id,
                CrawlRun.status.in_([RunStatus.QUEUED, RunStatus.RUNNING]),
                CrawlRun.updated_at >= active_cutoff,
            )).all())
            if any(
                run_covers(active, requested)
                for active in active_runs
            ):
                skipped += 1
                continue
            run = CrawlRun(
                job_id=str(uuid4()),
                company_id=company.id,
                source_type=source_types[0] if len(source_types) == 1 else None,
                source_types=normalize_source_types(source_types),
                trigger_type=TriggerType.SCHEDULED,
                status=RunStatus.QUEUED,
                stage="定时任务等待中",
            )
            db.add(run)
            db.commit()
            collect_company.apply_async(args=[run.id, source_types], queue="collection", priority=3)
            count += 1
        return {"dispatched": count, "skipped_active": skipped}


@shared_task(name="app.worker.tasks.expire_on_demand")
def expire_on_demand():
    cutoff = datetime.now() - timedelta(days=7)
    with SessionLocal() as db:
        companies = db.scalars(select(Company).where(Company.tracking_mode == TrackingMode.ON_DEMAND, Company.last_queried_at < cutoff)).all()
        for company in companies:
            company.tracking_mode = TrackingMode.INACTIVE
        db.commit()
        return {"expired": len(companies)}


@shared_task(name="app.worker.tasks.sync_company_master")
def sync_company_master():
    with SessionLocal() as db:
        from app.services.company_master import refresh_company_master
        try:
            return {"changed": refresh_company_master(db)}
        except Exception as exc:
            return {"changed": 0, "error": str(exc)}


def _number(value):
    try:
        if value is None or str(value).strip() in {"", "nan", "None", "-"}:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


@shared_task(name="app.worker.tasks.sync_market_history", bind=True, max_retries=2)
def sync_market_history(self, company_id: int):
    try:
        import akshare as ak

        with SessionLocal() as db:
            company = db.get(Company, company_id)
            if not company:
                return {"status": "missing"}
            end = datetime.now()
            start = end - timedelta(days=240)
            source_name = "东方财富"
            columns = {"date": "日期", "open": "开盘", "close": "收盘", "high": "最高", "low": "最低", "volume": "成交量", "amount": "成交额", "change_pct": "涨跌幅"}
            try:
                frame = ak.stock_zh_a_hist(
                    symbol=company.stock_code,
                    period="daily",
                    start_date=start.strftime("%Y%m%d"),
                    end_date=end.strftime("%Y%m%d"),
                    adjust="qfq",
                )
            except Exception:
                source_name = "新浪财经"
                symbol = ("sh" if company.stock_code.startswith(("5", "6", "9")) else "sz") + company.stock_code
                frame = ak.stock_zh_a_daily(
                    symbol=symbol,
                    start_date=start.strftime("%Y%m%d"),
                    end_date=end.strftime("%Y%m%d"),
                    adjust="qfq",
                )
                columns = {"date": "date", "open": "open", "close": "close", "high": "high", "low": "low", "volume": "volume", "amount": "amount", "change_pct": "change_pct"}
            if frame is None or frame.empty:
                return {"status": "empty", "stock_code": company.stock_code}
            first_date = frame.iloc[0][columns["date"]]
            earliest = datetime.combine(first_date, datetime.min.time()) if not isinstance(first_date, str) else datetime.strptime(first_date, "%Y-%m-%d")
            db.execute(delete(MarketPrice).where(MarketPrice.company_id == company.id, MarketPrice.trade_date >= earliest))
            for _, row in frame.iterrows():
                trade_date = row[columns["date"]]
                if isinstance(trade_date, str):
                    trade_date = datetime.strptime(trade_date, "%Y-%m-%d")
                elif not isinstance(trade_date, datetime):
                    trade_date = datetime.combine(trade_date, datetime.min.time())
                db.add(MarketPrice(
                    company_id=company.id,
                    trade_date=trade_date,
                    open=_number(row.get(columns["open"])), close=_number(row.get(columns["close"])),
                    high=_number(row.get(columns["high"])), low=_number(row.get(columns["low"])),
                    volume=_number(row.get(columns["volume"])), amount=_number(row.get(columns["amount"])),
                    change_pct=_number(row.get(columns["change_pct"])), source_name=source_name,
                ))
            db.commit()
            return {"status": "completed", "stock_code": company.stock_code, "rows": len(frame), "source": source_name}
    except Exception as exc:
        raise self.retry(exc=exc, countdown=60)


@shared_task(name="app.worker.tasks.dispatch_market_sync")
def dispatch_market_sync():
    cutoff = datetime.now() - timedelta(days=7)
    with SessionLocal() as db:
        companies = list(db.scalars(select(Company).where(
            (Company.tracking_mode.in_([TrackingMode.SEED, TrackingMode.PINNED])) |
            ((Company.tracking_mode == TrackingMode.ON_DEMAND) & (Company.last_queried_at >= cutoff))
        )).all())
        for company in companies:
            sync_market_history.apply_async(args=[company.id], queue="maintenance", priority=2)
        return {"dispatched": len(companies)}


@shared_task(name="app.worker.tasks.recover_stale_tasks")
def recover_stale_tasks():
    """Close abandoned UI states and collection locks after worker interruption."""
    now = datetime.now()
    chat_cutoff = now - timedelta(minutes=5)
    collection_cutoff = now - timedelta(minutes=30)
    queued_cutoff = now - timedelta(minutes=45)
    scheduled_queued_cutoff = now - timedelta(hours=4)
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    recovered_messages = 0
    recovered_runs = 0
    with SessionLocal() as db:
        messages = list(db.scalars(select(ChatMessage).where(
            ChatMessage.status.in_([
                MessageStatus.QUEUED,
                MessageStatus.RESOLVING_ENTITY,
                MessageStatus.PROCESSING,
                MessageStatus.ANSWERING,
            ]),
            ChatMessage.updated_at < chat_cutoff,
        )).all())
        collecting = list(db.scalars(select(ChatMessage).where(
            ChatMessage.status == MessageStatus.COLLECTING,
            ChatMessage.updated_at < queued_cutoff,
        )).all())
        for message in [*messages, *collecting]:
            message.status = MessageStatus.FAILED
            message.progress = 100
            message.status_text = "任务长时间无进展"
            message.error = "后台任务可能已中断，请点击重新执行。"
            recovered_messages += 1
        runs = list(db.scalars(select(CrawlRun).where(
            ((CrawlRun.status == RunStatus.RUNNING) & (CrawlRun.updated_at < collection_cutoff))
            | (
                (CrawlRun.status == RunStatus.QUEUED)
                & (CrawlRun.trigger_type != TriggerType.SCHEDULED)
                & (CrawlRun.updated_at < queued_cutoff)
            )
            | (
                (CrawlRun.status == RunStatus.QUEUED)
                & (CrawlRun.trigger_type == TriggerType.SCHEDULED)
                & (CrawlRun.updated_at < scheduled_queued_cutoff)
            )
        )).all())
        for run in runs:
            company = db.get(Company, run.company_id) if run.company_id else None
            run.status = RunStatus.FAILED
            run.progress = 100
            run.stage = "任务长时间无进展，已由看门狗终止"
            run.error = "worker_stale_timeout"
            run.finished_at = now
            if company:
                try:
                    redis.delete(f"crawl:lock:{company.stock_code}")
                except Exception:
                    pass
            for message in db.scalars(select(ChatMessage).where(
                ChatMessage.refresh_run_id == run.id,
                ChatMessage.refresh_status.in_(["QUEUED", "RUNNING"]),
            )).all():
                message.refresh_status = "FAILED"
                message.refresh_status_text = "后台更新长时间无进展，已终止"
                message.refresh_completed_at = now
            recovered_runs += 1
        db.commit()
    return {"messages": recovered_messages, "runs": recovered_runs}
