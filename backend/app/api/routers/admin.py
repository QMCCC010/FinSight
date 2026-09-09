from uuid import uuid4

import httpx
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from app.api.dependencies import require_admin
from app.core.config import get_settings
from app.core.database import get_db
from app.core.enums import DocumentStatus, TriggerType
from app.core.models import Company, CrawlRun, DataSource, Document, User
from app.core.schemas import CrawlRunOut, CrawlRunRequest, SourceOut, SourceUpdate, TrackingUpdate

router = APIRouter(prefix="/admin", tags=["知识库管理"])


@router.get("/sources", response_model=list[SourceOut])
def list_sources(db: Session = Depends(get_db), _: User = Depends(require_admin)) -> list[DataSource]:
    return list(db.scalars(select(DataSource).order_by(DataSource.source_type)).all())


@router.patch("/sources/{source_id}", response_model=SourceOut)
def update_source(source_id: int, payload: SourceUpdate, db: Session = Depends(get_db), _: User = Depends(require_admin)) -> DataSource:
    source = db.get(DataSource, source_id)
    if not source:
        raise HTTPException(status_code=404, detail="数据源不存在")
    if payload.enabled is not None:
        source.enabled = payload.enabled
    if payload.schedule is not None:
        source.schedule = payload.schedule
    db.commit()
    return source


@router.post("/crawl-runs", response_model=list[CrawlRunOut])
def create_crawl_runs(payload: CrawlRunRequest, db: Session = Depends(get_db), user: User = Depends(require_admin)) -> list[CrawlRun]:
    settings = get_settings()
    try:
        response = httpx.post(
            f"{settings.collector_url}/internal/v1/crawl-runs",
            headers={"X-Internal-Token": settings.internal_service_token},
            json={"stock_codes": payload.stock_codes, "source_types": payload.source_types, "trigger_type": TriggerType.ADMIN},
            timeout=10,
        )
        response.raise_for_status()
        ids = [item["id"] for item in response.json()["runs"]]
        # Authentication already opened a REPEATABLE READ transaction. End it so
        # this session can see rows the collector committed in its transaction.
        db.rollback()
        return list(db.scalars(select(CrawlRun).where(CrawlRun.id.in_(ids)).order_by(CrawlRun.id)).all())
    except Exception:
        # Local developer fallback when only FastAPI and a Celery worker are running.
        pass
    companies_query = select(Company)
    if payload.stock_codes:
        companies_query = companies_query.where(Company.stock_code.in_(payload.stock_codes))
    else:
        companies_query = companies_query.where(Company.tracking_mode.in_(["SEED", "PINNED", "ON_DEMAND"]))
    companies = list(db.scalars(companies_query).all())
    runs = []
    from app.worker.tasks import collect_company
    for company in companies:
        run = CrawlRun(
            job_id=str(uuid4()),
            company_id=company.id,
            source_type=payload.source_types[0] if len(payload.source_types) == 1 else None,
            source_types=payload.source_types,
            trigger_type=TriggerType.ADMIN,
            requested_by_user_id=user.id,
            status="QUEUED",
            stage="等待采集",
        )
        db.add(run)
        db.flush()
        runs.append(run)
        collect_company.apply_async(args=[run.id, payload.source_types or None], queue="collection", priority=3)
    db.commit()
    return runs


@router.delete("/documents/{document_id}")
def soft_delete_document(document_id: int, db: Session = Depends(get_db), _: User = Depends(require_admin)) -> dict:
    document = db.get(Document, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="文档不存在")
    document.is_deleted = True
    db.commit()
    from app.worker.tasks import delete_from_vector_index
    task = delete_from_vector_index.apply_async(args=[document_id], queue="maintenance", priority=5)
    return {"document_id": document_id, "deleted": True, "index_job_id": task.id}


@router.get("/crawl-runs", response_model=list[CrawlRunOut])
def list_crawl_runs(db: Session = Depends(get_db), _: User = Depends(require_admin)) -> list[CrawlRun]:
    return list(db.scalars(select(CrawlRun).order_by(CrawlRun.id.desc()).limit(100)).all())


@router.get("/crawl-runs/{run_id}", response_model=CrawlRunOut)
def get_crawl_run(run_id: int, db: Session = Depends(get_db), _: User = Depends(require_admin)) -> CrawlRun:
    run = db.get(CrawlRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="任务不存在")
    return run


@router.post("/documents/{document_id}/reprocess")
def reprocess_document(document_id: int, db: Session = Depends(get_db), _: User = Depends(require_admin)) -> dict:
    document = db.get(Document, document_id)
    if not document:
        raise HTTPException(status_code=404, detail="文档不存在")
    document.status = DocumentStatus.DOWNLOADED
    document.error_stage = None
    document.error_message = None
    db.commit()
    from app.worker.tasks import process_document
    task = process_document.apply_async(args=[document.id], queue="collection", priority=3)
    return {"job_id": task.id, "document_id": document.id, "status": "QUEUED"}


@router.post("/documents/reprocess-batch")
def reprocess_documents_batch(
    failed_only: bool = False,
    limit: int = 100,
    document_type: str | None = None,
    source_name: str | None = None,
    legacy_only: bool = False,
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
) -> dict:
    query = select(Document).where(Document.is_deleted.is_(False))
    if failed_only:
        query = query.where(Document.status == DocumentStatus.FAILED)
    if document_type:
        query = query.where(Document.document_type == document_type)
    if source_name:
        query = query.where(Document.source_name == source_name)
    if legacy_only:
        query = query.where(or_(
            Document.raw_path.like("%.html"),
            Document.parsed_text.like("%{{%"),
        ))
    query = query.order_by(Document.id.desc()).limit(min(max(limit, 1), 500))
    documents = list(db.scalars(query).all())
    from app.worker.tasks import process_document
    job_ids = []
    for document in documents:
        document.status = DocumentStatus.DOWNLOADED
        document.error_stage = None
        document.error_message = None
        task = process_document.apply_async(args=[document.id], queue="collection", priority=3)
        job_ids.append(task.id)
    db.commit()
    return {"submitted": len(job_ids), "job_ids": job_ids}


@router.patch("/companies/{stock_code}/tracking")
def update_tracking(stock_code: str, payload: TrackingUpdate, db: Session = Depends(get_db), _: User = Depends(require_admin)) -> dict:
    company = db.scalar(select(Company).where(Company.stock_code == stock_code))
    if not company:
        raise HTTPException(status_code=404, detail="公司不存在")
    company.tracking_mode = payload.tracking_mode
    db.commit()
    return {"stock_code": stock_code, "tracking_mode": company.tracking_mode}


@router.post("/index/rebuild")
def rebuild_index(_: User = Depends(require_admin)) -> dict:
    from app.worker.tasks import rebuild_vector_index
    task = rebuild_vector_index.apply_async(queue="maintenance", priority=2)
    return {"job_id": task.id, "status": "QUEUED"}


@router.get("/index/status")
def index_status(_: User = Depends(require_admin)) -> dict:
    from app.ai.index import get_index_status
    return get_index_status()


@router.post("/market/sync")
def sync_market_prices(
    stock_codes: list[str] | None = None,
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
) -> dict:
    query = select(Company)
    if stock_codes:
        query = query.where(Company.stock_code.in_(stock_codes))
    else:
        query = query.where(Company.tracking_mode.in_(["SEED", "PINNED", "ON_DEMAND"]))
    companies = list(db.scalars(query).all())
    from app.worker.tasks import sync_market_history
    jobs = [sync_market_history.apply_async(args=[company.id], queue="maintenance", priority=2).id for company in companies]
    return {"submitted": len(jobs), "job_ids": jobs}
