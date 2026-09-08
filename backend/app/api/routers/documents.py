from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user
from app.core.database import get_db
from app.core.models import Company, Document, EarningsForecast, Evidence, FinancialMetric, InvestmentRating, Opinion, RiskItem, SentimentResult, User
from app.core.schemas import DocumentDetail, DocumentOut

router = APIRouter(prefix="/documents", tags=["文档"])


@router.get("", response_model=list[DocumentOut])
def list_documents(
    stock_code: str | None = None,
    document_type: str | None = None,
    status: str | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
) -> list[Document]:
    query = select(Document).where(Document.is_deleted.is_(False))
    if stock_code:
        company_id = db.scalar(select(Company.id).where(Company.stock_code == stock_code))
        query = query.where(Document.company_id == company_id)
    if document_type:
        query = query.where(Document.document_type == document_type)
    if status:
        query = query.where(Document.status == status)
    return list(db.scalars(query.order_by(Document.published_at.desc(), Document.id.desc()).limit(limit)).all())


@router.get("/{document_id}", response_model=DocumentDetail)
def document_detail(document_id: int, db: Session = Depends(get_db), _: User = Depends(get_current_user)) -> dict:
    document = db.get(Document, document_id)
    if not document or document.is_deleted:
        raise HTTPException(status_code=404, detail="文档不存在")
    rating = db.scalar(select(InvestmentRating).where(InvestmentRating.document_id == document.id))
    metrics = list(db.scalars(select(FinancialMetric).where(FinancialMetric.document_id == document.id)).all())
    forecasts = list(db.scalars(select(EarningsForecast).where(EarningsForecast.document_id == document.id).order_by(EarningsForecast.forecast_year)).all())
    opinions = list(db.scalars(select(Opinion).where(Opinion.document_id == document.id)).all())
    risks = list(db.scalars(select(RiskItem).where(RiskItem.document_id == document.id)).all())
    sentiments = list(db.scalars(select(SentimentResult).where(SentimentResult.document_id == document.id)).all())
    evidences = list(db.scalars(select(Evidence).where(Evidence.document_id == document.id).order_by(Evidence.id)).all())
    result = DocumentDetail.model_validate(document).model_dump()
    result["structured"] = {
        "rating": ({"institution": rating.institution, "analyst": rating.analyst, "rating": rating.original_rating, "normalized_rating": rating.normalized_rating, "previous_rating": rating.previous_rating, "target_price": rating.target_price, "currency": rating.currency} if rating else None),
        "financial_metrics": [{"name": item.metric_name, "raw_value": item.raw_value, "value": item.normalized_value, "unit": item.unit, "period": item.period, "yoy": item.yoy} for item in metrics],
        "forecasts": [{"institution": item.institution, "year": item.forecast_year, "revenue": item.revenue, "net_profit": item.net_profit, "eps": item.eps, "unit": item.unit} for item in forecasts],
        "opinions": [{"type": item.opinion_type, "content": item.content, "sentiment": item.sentiment, "confidence": item.confidence} for item in opinions],
        "risks": [{"category": item.category, "content": item.content, "confidence": item.confidence} for item in risks],
        "sentiments": [{"sentiment": item.sentiment, "score": item.score, "reason": item.reason, "confidence": item.confidence} for item in sentiments],
    }
    result["evidences"] = [{"entity_type": item.entity_type, "entity_id": item.entity_id, "page": item.page_number, "quote": item.quote} for item in evidences]
    return result
