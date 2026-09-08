from datetime import UTC, datetime, timedelta
import logging
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user
from app.core.database import get_db
from app.core.enums import RunStatus, TrackingMode, TriggerType
from app.core.models import Company, CrawlRun, Document, GeneratedReport, ReportVersion, User
from app.core.schemas import CreateReportRequest, RefreshReportMaterialsRequest, ReportOut, ReportVersionOut, UpdateReportRequest


router = APIRouter(prefix="/reports", tags=["内容工作台"])
logger = logging.getLogger(__name__)
ALLOWED_SOURCES = {"RESEARCH_REPORT", "NEWS", "ANNOUNCEMENT", "SOCIAL"}
SOURCE_LABELS = {
    "RESEARCH_REPORT": "券商研报",
    "NEWS": "财经新闻",
    "ANNOUNCEMENT": "公司公告",
    "SOCIAL": "社交舆情",
}


def _company(db: Session, stock_code: str) -> Company:
    company = db.scalar(select(Company).where(Company.stock_code == stock_code))
    if not company:
        raise HTTPException(status_code=404, detail="未找到公司")
    return company


def _date_filter(query, date_from: datetime | None, date_to: datetime | None):
    if date_from:
        query = query.where(Document.published_at >= date_from)
    if date_to:
        query = query.where(Document.published_at <= date_to)
    return query


@router.get("/materials")
def report_materials(
    stock_code: str = Query(pattern=r"^\d{6}$"),
    report_type: str = Query(default="COMPANY_BRIEF", pattern="^(COMPANY_BRIEF|REPORT_COMPARISON)$"),
    source_types: list[str] = Query(default=[]),
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
) -> dict:
    company = _company(db, stock_code)
    invalid_sources = set(source_types) - ALLOWED_SOURCES
    if invalid_sources:
        raise HTTPException(status_code=422, detail=f"无效资料类型：{','.join(sorted(invalid_sources))}")
    effective_types = ["RESEARCH_REPORT"] if report_type == "REPORT_COMPARISON" else (source_types or list(ALLOWED_SOURCES))
    query = select(Document).where(
        Document.company_id == company.id,
        Document.is_deleted.is_(False),
        Document.document_type.in_(effective_types),
    )
    query = _date_filter(query, date_from, date_to)
    documents = list(db.scalars(query.order_by(Document.published_at.desc(), Document.id.desc()).limit(120)).all())
    counts = {source: 0 for source in ALLOWED_SOURCES}
    latest_by_type: dict[str, datetime | None] = {source: None for source in ALLOWED_SOURCES}
    usable = 0
    for document in documents:
        counts[document.document_type] = counts.get(document.document_type, 0) + 1
        if document.status == "INDEXED":
            usable += 1
        current = latest_by_type.get(document.document_type)
        if document.published_at and (current is None or document.published_at > current):
            latest_by_type[document.document_type] = document.published_at
    now = datetime.now(UTC).replace(tzinfo=None)
    freshness = {}
    for source in ALLOWED_SOURCES:
        latest = latest_by_type[source]
        threshold = timedelta(hours=2 if source in {"NEWS", "SOCIAL"} else 36)
        freshness[source] = {
            "label": SOURCE_LABELS[source],
            "count": counts[source],
            "latest_at": latest,
            "is_fresh": bool(latest and latest >= now - threshold),
        }
    warnings = []
    if not documents:
        warnings.append("当前筛选范围内没有资料，请调整条件或等待后台采集。")
    if report_type == "REPORT_COMPARISON" and len(documents) < 2:
        warnings.append("多研报观点对比至少需要两篇券商研报。")
    if any(item.status != "INDEXED" for item in documents):
        warnings.append("部分资料尚未完成解析或索引，生成时将优先使用已处理内容。")
    return {
        "company": {"id": company.id, "name": company.name, "stock_code": company.stock_code},
        "total": len(documents),
        "usable": usable,
        "last_crawled_at": company.last_crawled_at,
        "last_indexed_at": company.last_indexed_at,
        "is_collecting": bool(db.scalar(select(func.count(CrawlRun.id)).where(CrawlRun.company_id == company.id, CrawlRun.status.in_([RunStatus.QUEUED, RunStatus.RUNNING])))),
        "freshness": freshness,
        "warnings": warnings,
        "documents": [
            {
                "id": item.id,
                "title": item.title,
                "document_type": item.document_type,
                "source_name": item.source_name,
                "source_url": item.source_url,
                "published_at": item.published_at,
                "status": item.status,
                "summary": item.summary,
                "usable": item.status == "INDEXED" and bool(item.parsed_text),
            }
            for item in documents
        ],
    }


@router.post("/materials/{stock_code}/refresh", status_code=202)
def refresh_report_materials(
    stock_code: str,
    payload: RefreshReportMaterialsRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict:
    company = _company(db, stock_code)
    source_types = payload.source_types or list(ALLOWED_SOURCES)
    invalid_sources = set(source_types) - ALLOWED_SOURCES
    if invalid_sources:
        raise HTTPException(status_code=422, detail=f"无效资料类型：{','.join(sorted(invalid_sources))}")
    active = db.scalar(select(CrawlRun).where(
        CrawlRun.company_id == company.id,
        CrawlRun.status.in_([RunStatus.QUEUED, RunStatus.RUNNING]),
    ).order_by(CrawlRun.created_at.desc()))
    if active:
        return {"run_id": active.id, "status": active.status, "reused": True}
    run = CrawlRun(
        job_id=str(uuid4()),
        company_id=company.id,
        source_type=source_types[0] if len(source_types) == 1 else None,
        source_types=source_types,
        trigger_type=TriggerType.ON_DEMAND,
        requested_by_user_id=user.id,
        status=RunStatus.QUEUED,
        stage="内容工作台请求后台更新",
        progress=0,
    )
    if company.tracking_mode == TrackingMode.INACTIVE:
        company.tracking_mode = TrackingMode.ON_DEMAND
    company.last_queried_at = datetime.now()
    db.add(run)
    db.commit()
    from app.worker.tasks import collect_company

    collect_company.apply_async(args=[run.id, source_types], queue="collection", priority=5)
    return {"run_id": run.id, "status": run.status, "reused": False}


@router.post("", response_model=ReportOut, status_code=202)
def create_report(
    payload: CreateReportRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> GeneratedReport:
    company = _company(db, payload.stock_code)
    invalid_sources = set(payload.source_types) - ALLOWED_SOURCES
    if invalid_sources:
        raise HTTPException(status_code=422, detail=f"无效资料类型：{','.join(sorted(invalid_sources))}")
    if payload.date_from and payload.date_to and payload.date_from > payload.date_to:
        raise HTTPException(status_code=422, detail="资料开始时间不能晚于结束时间")
    source_types = ["RESEARCH_REPORT"] if payload.report_type == "REPORT_COMPARISON" else (payload.source_types or list(ALLOWED_SOURCES))
    if payload.document_ids:
        query = select(Document).where(
            Document.id.in_(payload.document_ids),
            Document.company_id == company.id,
            Document.is_deleted.is_(False),
            Document.document_type.in_(source_types),
        )
        query = _date_filter(query, payload.date_from, payload.date_to)
        selected = list(db.scalars(query).all())
        if len(selected) != len(set(payload.document_ids)):
            raise HTTPException(status_code=422, detail="部分所选资料不存在、不属于该公司或不符合筛选条件")
        if payload.report_type == "REPORT_COMPARISON" and len(selected) < 2:
            raise HTTPException(status_code=422, detail="多研报观点对比至少需要选择两篇研报")
    elif payload.report_type == "REPORT_COMPARISON":
        query = select(Document.id).where(
            Document.company_id == company.id,
            Document.document_type == "RESEARCH_REPORT",
            Document.is_deleted.is_(False),
        )
        query = _date_filter(query, payload.date_from, payload.date_to)
        payload.document_ids = list(db.scalars(query.order_by(Document.published_at.desc()).limit(5)).all())
        if len(payload.document_ids) < 2:
            raise HTTPException(status_code=422, detail="当前筛选范围内可比较研报不足两篇")
    title = f"{company.name} - {'公司研究简报' if payload.report_type == 'COMPANY_BRIEF' else '多研报观点对比'}"
    report = GeneratedReport(
        user_id=user.id,
        company_id=company.id,
        report_type=payload.report_type,
        title=title,
        date_from=payload.date_from,
        date_to=payload.date_to,
        source_types=source_types,
        selected_document_ids=payload.document_ids or [],
        status="QUEUED",
        progress=0,
        status_text="报告已进入生成队列",
        content_markdown="",
        citations=[],
    )
    db.add(report)
    db.commit()
    db.refresh(report)
    try:
        from app.worker.tasks import generate_report

        generate_report.apply_async(args=[report.id], queue="reports", priority=7, expires=900)
    except Exception as exc:
        logger.exception("Failed to enqueue generated report %s", report.id)
        report.status = "FAILED"
        report.progress = 100
        report.status_text = "无法提交报告任务"
        report.error = "报告任务队列暂时不可用，请稍后重试。"
        db.commit()
    return report


@router.get("", response_model=list[ReportOut])
def list_reports(db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> list[GeneratedReport]:
    return list(db.scalars(
        select(GeneratedReport)
        .where(GeneratedReport.user_id == user.id, GeneratedReport.is_deleted.is_(False))
        .order_by(GeneratedReport.created_at.desc())
    ).all())


def owned_report(db: Session, report_id: int, user: User) -> GeneratedReport:
    report = db.get(GeneratedReport, report_id)
    if not report or report.user_id != user.id or report.is_deleted:
        raise HTTPException(status_code=404, detail="报告不存在")
    return report


@router.get("/{report_id}", response_model=ReportOut)
def get_report(report_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> GeneratedReport:
    return owned_report(db, report_id, user)


@router.patch("/{report_id}", response_model=ReportOut)
def update_report(
    report_id: int,
    payload: UpdateReportRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> GeneratedReport:
    report = owned_report(db, report_id, user)
    if report.status not in {"COMPLETED", "PARTIAL"}:
        raise HTTPException(status_code=409, detail="报告尚未生成完成，暂不能编辑")
    if payload.title is not None:
        report.title = payload.title.strip()
    if payload.content_markdown is not None:
        report.content_markdown = payload.content_markdown
    latest = db.scalar(select(func.max(ReportVersion.version_number)).where(ReportVersion.report_id == report.id)) or 0
    db.add(ReportVersion(
        report_id=report.id,
        version_number=latest + 1,
        title=report.title,
        content_markdown=report.content_markdown,
        citations=report.citations,
        change_type="MANUAL",
    ))
    db.commit()
    db.refresh(report)
    return report


@router.get("/{report_id}/versions", response_model=list[ReportVersionOut])
def report_versions(report_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> list[ReportVersion]:
    report = owned_report(db, report_id, user)
    return list(db.scalars(select(ReportVersion).where(ReportVersion.report_id == report.id).order_by(ReportVersion.version_number.desc())).all())


@router.post("/{report_id}/cancel", response_model=ReportOut)
def cancel_report(report_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> GeneratedReport:
    report = owned_report(db, report_id, user)
    if report.status in {"QUEUED", "RUNNING"}:
        report.status = "CANCELLED"
        report.progress = 100
        report.status_text = "已停止报告生成"
        db.commit()
        db.refresh(report)
    return report


@router.post("/{report_id}/retry", response_model=ReportOut, status_code=202)
def retry_report(report_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> GeneratedReport:
    report = owned_report(db, report_id, user)
    if report.status not in {"FAILED", "CANCELLED"}:
        raise HTTPException(status_code=409, detail="只有失败或已停止的报告可以重试")
    report.status = "QUEUED"
    report.progress = 0
    report.status_text = "报告已重新进入生成队列"
    report.error = None
    db.commit()
    from app.worker.tasks import generate_report

    generate_report.apply_async(args=[report.id], queue="reports", priority=7, expires=900)
    db.refresh(report)
    return report


@router.delete("/{report_id}", status_code=204)
def delete_report(report_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> Response:
    report = owned_report(db, report_id, user)
    report.is_deleted = True
    db.commit()
    return Response(status_code=204)


@router.get("/{report_id}/download")
def download_report(report_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> Response:
    report = owned_report(db, report_id, user)
    if not report.content_markdown:
        raise HTTPException(status_code=409, detail="报告尚未生成完成")
    filename = f"report-{report.id}.md"
    return Response(report.content_markdown, media_type="text/markdown; charset=utf-8", headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@router.get("/{report_id}/export")
def export_report(
    report_id: int,
    format: str = Query(default="docx", pattern="^(docx|pdf)$"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Response:
    report = owned_report(db, report_id, user)
    if not report.content_markdown:
        raise HTTPException(status_code=409, detail="报告尚未生成完成")
    from app.services.report_export import export_docx, export_pdf

    if format == "pdf":
        content = export_pdf(report.title, report.content_markdown)
        media_type = "application/pdf"
    else:
        content = export_docx(report.title, report.content_markdown)
        media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    return Response(content, media_type=media_type, headers={"Content-Disposition": f'attachment; filename="report-{report.id}.{format}"'})
