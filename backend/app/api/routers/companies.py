from datetime import UTC, datetime, timedelta
from difflib import SequenceMatcher
import re

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import case, func, or_, select
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user
from app.core.database import get_db
from app.core.enums import DocumentStatus, RunStatus
from app.core.models import Company, CrawlRun, Document, DocumentChunk, EarningsForecast, Favorite, FinancialMetric, InvestmentRating, MarketPrice, SentimentResult, User
from app.core.schemas import CompanyOut, FavoriteOut, KnowledgeStatus

router = APIRouter(prefix="/companies", tags=["公司"])


@router.get("/favorites/me", response_model=list[FavoriteOut])
def list_favorites(db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> list[FavoriteOut]:
    rows = db.scalars(select(Favorite).where(Favorite.user_id == user.id).order_by(Favorite.created_at.desc())).all()
    return [FavoriteOut(company=CompanyOut.model_validate(db.get(Company, row.company_id)), created_at=row.created_at) for row in rows]


@router.post("/{stock_code}/favorite", status_code=201)
def add_favorite(stock_code: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> dict:
    company = _get_company(db, stock_code)
    existing = db.scalar(select(Favorite).where(Favorite.user_id == user.id, Favorite.company_id == company.id))
    if not existing:
        db.add(Favorite(user_id=user.id, company_id=company.id))
        db.commit()
    return {"stock_code": stock_code, "favorite": True}


@router.delete("/{stock_code}/favorite")
def remove_favorite(stock_code: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> dict:
    company = _get_company(db, stock_code)
    favorite = db.scalar(select(Favorite).where(Favorite.user_id == user.id, Favorite.company_id == company.id))
    if favorite:
        db.delete(favorite)
        db.commit()
    return {"stock_code": stock_code, "favorite": False}


@router.get("", response_model=list[CompanyOut])
def list_companies(
    tracked_only: bool = False,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
) -> list[Company]:
    query = select(Company).order_by(Company.tracking_mode, Company.stock_code)
    if tracked_only:
        query = query.where(Company.tracking_mode != "INACTIVE")
    return list(db.scalars(query).all())


@router.get("/search", response_model=list[CompanyOut])
def search_companies(
    q: str = Query(min_length=1, max_length=64),
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
) -> list[Company]:
    pattern = f"%{q.strip()}%"
    return list(db.scalars(select(Company).where(or_(Company.stock_code.like(pattern), Company.name.like(pattern), Company.full_name.like(pattern))).limit(20)).all())


@router.get("/dashboard/summary")
def dashboard_summary(db: Session = Depends(get_db), _: User = Depends(get_current_user)) -> dict:
    tracked = db.scalar(select(func.count(Company.id)).where(Company.tracking_mode != "INACTIVE")) or 0
    documents = db.scalar(select(func.count(Document.id)).where(Document.is_deleted.is_(False))) or 0
    live_documents = db.scalar(select(func.count(Document.id)).where(Document.is_deleted.is_(False), Document.acquisition_mode == "LIVE")) or 0
    indexed = db.scalar(select(func.count(Document.id)).where(Document.is_deleted.is_(False), Document.status == DocumentStatus.INDEXED)) or 0
    active_runs = db.scalar(select(func.count(CrawlRun.id)).where(CrawlRun.status.in_([RunStatus.QUEUED, RunStatus.RUNNING]))) or 0
    latest_document = db.scalar(select(func.max(Document.published_at)).where(Document.is_deleted.is_(False)))
    return {
        "tracked_companies": tracked,
        "documents": documents,
        "live_documents": live_documents,
        "snapshot_documents": max(0, documents - live_documents),
        "indexed_documents": indexed,
        "active_runs": active_runs,
        "latest_document_at": latest_document,
    }


def _get_company(db: Session, stock_code: str) -> Company:
    company = db.scalar(select(Company).where(Company.stock_code == stock_code))
    if not company:
        raise HTTPException(status_code=404, detail="未找到该A股公司")
    return company


@router.get("/{stock_code}/overview")
def company_overview(
    stock_code: str,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
) -> dict:
    company = _get_company(db, stock_code)
    counts = dict(db.execute(select(Document.document_type, func.count(Document.id)).where(Document.company_id == company.id, Document.is_deleted.is_(False)).group_by(Document.document_type)).all())
    latest = list(db.scalars(select(Document).where(Document.company_id == company.id, Document.is_deleted.is_(False)).order_by(Document.published_at.desc()).limit(10)).all())
    mode_counts = dict(db.execute(select(Document.acquisition_mode, func.count(Document.id)).where(Document.company_id == company.id, Document.is_deleted.is_(False)).group_by(Document.acquisition_mode)).all())
    latest_by_type = dict(db.execute(select(Document.document_type, func.max(Document.published_at)).where(Document.company_id == company.id, Document.is_deleted.is_(False)).group_by(Document.document_type)).all())
    return {
        "company": CompanyOut.model_validate(company),
        "document_counts": counts,
        "acquisition_counts": mode_counts,
        "latest_by_type": latest_by_type,
        "latest_documents": [
            {
                "id": d.id,
                "title": d.title,
                "document_type": d.document_type,
                "published_at": d.published_at,
                "summary": d.summary,
                "source_name": d.source_name,
                "source_url": d.source_url,
                "status": d.status,
                "acquisition_mode": d.acquisition_mode,
            }
            for d in latest
        ],
    }


def _event_similarity(left: str, right: str) -> float:
    def compact(value: str) -> str:
        return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]", "", value.lower())

    a, b = compact(left), compact(right)
    if not a or not b:
        return 0.0
    pairs_a = {a[index:index + 2] for index in range(max(1, len(a) - 1))}
    pairs_b = {b[index:index + 2] for index in range(max(1, len(b) - 1))}
    union = pairs_a | pairs_b
    return max(SequenceMatcher(None, a, b).ratio(), len(pairs_a & pairs_b) / len(union) if union else 0.0)


@router.get("/{stock_code}/events")
def company_events(
    stock_code: str,
    limit: int = Query(default=12, ge=1, le=30),
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
) -> list[dict]:
    company = _get_company(db, stock_code)
    documents = list(db.scalars(
        select(Document)
        .where(Document.company_id == company.id, Document.is_deleted.is_(False))
        .order_by(Document.published_at.desc(), Document.id.desc())
        .limit(120)
    ).all())
    clusters: list[list[Document]] = []
    for document in documents:
        target = None
        for cluster in clusters:
            representative = cluster[0]
            close_in_time = not document.published_at or not representative.published_at or abs((document.published_at - representative.published_at).days) <= 10
            if close_in_time and _event_similarity(document.title, representative.title) >= 0.48:
                target = cluster
                break
        if target is None:
            target = []
            clusters.append(target)
        target.append(document)
    result = []
    for cluster in clusters[:limit]:
        representative = cluster[0]
        result.append({
            "event_id": "-".join(str(item.id) for item in cluster[:8]),
            "title": representative.title,
            "date": representative.published_at,
            "summary": representative.summary or (representative.parsed_text or "")[:240],
            "document_count": len(cluster),
            "source_count": len({item.source_name for item in cluster}),
            "source_types": sorted({item.document_type for item in cluster}),
            "documents": [{"id": item.id, "title": item.title, "source_name": item.source_name, "source_url": item.source_url} for item in cluster[:10]],
        })
    return result


@router.get("/{stock_code}/research-metrics")
def company_research_metrics(
    stock_code: str,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
) -> dict:
    company = _get_company(db, stock_code)
    metrics = list(db.scalars(select(FinancialMetric).where(FinancialMetric.company_id == company.id).order_by(FinancialMetric.period, FinancialMetric.id)).all())
    forecasts = list(db.scalars(select(EarningsForecast).where(EarningsForecast.company_id == company.id).order_by(EarningsForecast.forecast_year, EarningsForecast.id)).all())
    ratings = list(db.execute(
        select(InvestmentRating, Document.published_at, Document.title)
        .join(Document, Document.id == InvestmentRating.document_id)
        .where(InvestmentRating.company_id == company.id, Document.is_deleted.is_(False))
        .order_by(Document.published_at)
    ).all())
    sentiments = list(db.execute(
        select(SentimentResult, Document.published_at, Document.document_type)
        .join(Document, Document.id == SentimentResult.document_id)
        .where(SentimentResult.company_id == company.id, Document.is_deleted.is_(False))
        .order_by(Document.published_at)
    ).all())
    return {
        "company": company.name,
        "financial_metrics": [{"name": item.metric_name, "period": item.period, "value": item.normalized_value, "raw_value": item.raw_value, "unit": item.unit, "yoy": item.yoy, "document_id": item.document_id} for item in metrics],
        "forecasts": [{"institution": item.institution, "year": item.forecast_year, "revenue": item.revenue, "net_profit": item.net_profit, "eps": item.eps, "unit": item.unit, "document_id": item.document_id} for item in forecasts],
        "ratings": [{"institution": item.institution, "rating": item.original_rating, "normalized_rating": item.normalized_rating, "target_price": item.target_price, "date": published_at, "title": title, "document_id": item.document_id} for item, published_at, title in ratings],
        "sentiments": [{"sentiment": item.sentiment, "score": item.score, "confidence": item.confidence, "date": published_at, "source_type": source_type, "document_id": item.document_id} for item, published_at, source_type in sentiments],
    }


@router.get("/{stock_code}/market-history")
def company_market_history(
    stock_code: str,
    limit: int = Query(default=120, ge=20, le=500),
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
) -> dict:
    company = _get_company(db, stock_code)
    rows = list(db.scalars(
        select(MarketPrice)
        .where(MarketPrice.company_id == company.id)
        .order_by(MarketPrice.trade_date.desc())
        .limit(limit)
    ).all())
    rows.reverse()
    return {
        "company": company.name,
        "status": "READY" if rows else "EMPTY",
        "data_as_of": rows[-1].trade_date if rows else None,
        "source_name": rows[-1].source_name if rows else None,
        "items": [{"date": item.trade_date, "open": item.open, "close": item.close, "high": item.high, "low": item.low, "volume": item.volume, "amount": item.amount, "change_pct": item.change_pct} for item in rows],
    }


@router.get("/{stock_code}/knowledge-status", response_model=KnowledgeStatus)
def knowledge_status(
    stock_code: str,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
) -> KnowledgeStatus:
    company = _get_company(db, stock_code)
    by_type = dict(db.execute(select(Document.document_type, func.count(Document.id)).where(Document.company_id == company.id, Document.is_deleted.is_(False)).group_by(Document.document_type)).all())
    chunks = db.scalar(select(func.count(DocumentChunk.id)).where(DocumentChunk.company_id == company.id)) or 0
    collecting = bool(db.scalar(select(func.count(CrawlRun.id)).where(CrawlRun.company_id == company.id, CrawlRun.status.in_([RunStatus.QUEUED, RunStatus.RUNNING]))))
    fresh = bool(company.last_crawled_at and company.last_crawled_at >= datetime.now(UTC).replace(tzinfo=None) - timedelta(hours=24))
    return KnowledgeStatus(company=CompanyOut.model_validate(company), total_documents=sum(by_type.values()), by_type=by_type, indexed_chunks=chunks, is_collecting=collecting, is_fresh=fresh)


@router.get("/{stock_code}/timeline")
def company_timeline(
    stock_code: str,
    limit: int = Query(default=30, ge=1, le=100),
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
) -> list[dict]:
    company = _get_company(db, stock_code)
    docs = db.scalars(select(Document).where(Document.company_id == company.id, Document.is_deleted.is_(False)).order_by(Document.published_at.desc()).limit(limit)).all()
    return [{"id": item.id, "date": item.published_at, "type": item.document_type, "title": item.title, "summary": item.summary, "source": item.source_name} for item in docs]
