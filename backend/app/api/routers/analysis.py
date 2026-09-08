from collections import Counter
from difflib import SequenceMatcher
import re

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user
from app.ai.extraction import clean_rating, normalize_rating, rule_based_extract
from app.core.database import get_db
from app.core.models import Company, Document, EarningsForecast, InvestmentRating, Opinion, RiskItem, SentimentResult, User
from app.core.schemas import ReportComparisonRequest

router = APIRouter(prefix="/analysis", tags=["分析"])

RATING_LABELS = {
    "POSITIVE": "积极",
    "SLIGHTLY_POSITIVE": "偏积极",
    "NEUTRAL": "中性",
    "SLIGHTLY_NEGATIVE": "偏谨慎",
    "NEGATIVE": "谨慎",
}


def _statement_similarity(left: str, right: str) -> float:
    def compact(value: str) -> str:
        return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]", "", value.lower())

    a, b = compact(left), compact(right)
    if not a or not b:
        return 0.0
    sequence_score = SequenceMatcher(None, a, b).ratio()
    a_pairs = {a[index : index + 2] for index in range(max(1, len(a) - 1))}
    b_pairs = {b[index : index + 2] for index in range(max(1, len(b) - 1))}
    union = a_pairs | b_pairs
    pair_score = len(a_pairs & b_pairs) / len(union) if union else 0.0
    return max(sequence_score, pair_score)


def _cluster_statements(points: list[dict], *, require_distinct_institutions: bool) -> list[str]:
    clusters: list[list[dict]] = []
    for point in points:
        content = re.sub(r"\s+", " ", point.get("content", "")).strip()
        if len(content) < 6:
            continue
        candidate = {**point, "content": content}
        best_cluster = None
        best_score = 0.0
        for cluster in clusters:
            score = max(_statement_similarity(content, member["content"]) for member in cluster)
            if score > best_score:
                best_cluster, best_score = cluster, score
        if best_cluster is not None and best_score >= 0.46:
            best_cluster.append(candidate)
        else:
            clusters.append([candidate])

    results: list[tuple[int, str]] = []
    for cluster in clusters:
        document_count = len({item["document_id"] for item in cluster})
        institutions = {item.get("institution") for item in cluster if item.get("institution")}
        support = len(institutions) if require_distinct_institutions else document_count
        if document_count < 2 or support < 2:
            continue
        representative = min((item["content"] for item in cluster), key=len)[:220]
        suffix = f"（{len(institutions)}家机构、{document_count}篇研报共同提及）" if institutions else f"（{document_count}篇研报共同提及）"
        results.append((support, representative + suffix))
    return [text for _, text in sorted(results, key=lambda item: item[0], reverse=True)[:5]]


def get_company(db: Session, stock_code: str) -> Company:
    company = db.scalar(select(Company).where(Company.stock_code == stock_code))
    if not company:
        raise HTTPException(status_code=404, detail="未找到公司")
    return company


@router.post("/report-comparison")
def compare_reports(
    payload: ReportComparisonRequest,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_user),
) -> dict:
    company = get_company(db, payload.stock_code)
    base_filter = (
        Document.company_id == company.id,
        Document.document_type == "RESEARCH_REPORT",
        Document.is_deleted.is_(False),
    )
    available_count = db.scalar(select(func.count(Document.id)).where(*base_filter)) or 0
    query = select(Document).where(*base_filter)
    if payload.document_ids:
        query = query.where(Document.id.in_(payload.document_ids))
    effective_limit = len(payload.document_ids) if payload.document_ids else payload.limit
    documents = list(db.scalars(query.order_by(Document.published_at.desc()).limit(effective_limit)).all())
    if payload.document_ids and len(documents) != len(set(payload.document_ids)):
        raise HTTPException(status_code=400, detail="部分研报不存在或不属于所选公司")

    rows = []
    positive_points: list[dict] = []
    risk_points: list[dict] = []
    bearish_points: list[str] = []
    rating_entries: list[tuple[str, str, str]] = []
    for document in documents:
        rating = db.scalar(select(InvestmentRating).where(InvestmentRating.document_id == document.id))
        forecasts = list(db.scalars(select(EarningsForecast).where(EarningsForecast.document_id == document.id).order_by(EarningsForecast.forecast_year)).all())
        opinions = list(db.scalars(select(Opinion).where(Opinion.document_id == document.id)).all())
        risks = list(db.scalars(select(RiskItem).where(RiskItem.document_id == document.id)).all())
        institution = rating.institution if rating and rating.institution else document.source_name
        valid_rating = clean_rating(rating.original_rating) if rating else None
        fallback = rule_based_extract(document) if not valid_rating or not opinions else None
        if not valid_rating and fallback:
            valid_rating = fallback.original_rating
        comparison_opinions = opinions or (fallback.opinions if fallback else [])
        normalized_rating = normalize_rating(valid_rating)
        if valid_rating and normalized_rating:
            rating_entries.append((institution, valid_rating, normalized_rating))
        for item in comparison_opinions:
            point = {"document_id": document.id, "institution": institution, "content": item.content}
            opinion_type = (getattr(item, "opinion_type", None) or getattr(item, "type", "")).upper()
            if opinion_type in {"BULL", "POSITIVE", "CATALYST"}:
                positive_points.append(point)
            elif opinion_type in {"BEAR", "NEGATIVE", "RISK"}:
                bearish_points.append(f"{institution}：{item.content[:220]}")
        risk_points.extend(
            {"document_id": document.id, "institution": institution, "content": item.content}
            for item in risks
        )
        rows.append({
            "document_id": document.id,
            "title": document.title,
            "institution": institution,
            "published_at": document.published_at,
            "rating": valid_rating,
            "normalized_rating": normalized_rating,
            "target_price": rating.target_price if rating else None,
            "forecasts": [{"year": f.forecast_year, "revenue": f.revenue, "net_profit": f.net_profit, "eps": f.eps, "unit": f.unit} for f in forecasts],
            "opinions": [item.content for item in comparison_opinions],
            "risks": [item.content for item in risks],
            "source_url": document.source_url,
        })
    if len(rows) < 2:
        return {
            "company": company.name,
            "status": "INSUFFICIENT",
            "message": "当前可比较研报不足两篇，请先同步研报数据。",
            "reports": rows,
            "available_count": available_count,
            "consensus": [],
            "common_risks": [],
            "differences": [],
            "consensus_message": "至少选择两篇研报后才能判断机构共识。",
            "common_risks_message": "至少选择两篇研报后才能判断共同风险。",
        }

    consensus = _cluster_statements(positive_points, require_distinct_institutions=True)
    common_risks = _cluster_statements(risk_points, require_distinct_institutions=False)
    rating_counts = Counter(item[2] for item in rating_entries)
    if rating_counts:
        top_rating, top_count = rating_counts.most_common(1)[0]
        top_institutions = {institution for institution, _, normalized in rating_entries if normalized == top_rating}
        if top_count >= 2 and len(top_institutions) >= 2:
            consensus.insert(0, f"评级倾向为{RATING_LABELS[top_rating]}（{len(top_institutions)}家机构采用相同方向）")
    differences = list(dict.fromkeys(bearish_points))[:5]
    if len(rating_counts) > 1:
        distribution = "、".join(f"{RATING_LABELS[key]}{count}篇" for key, count in rating_counts.most_common())
        differences.insert(0, f"评级方向存在分歧：{distribution}")

    return {
        "company": company.name,
        "status": "COMPLETED",
        "reports": rows,
        "available_count": available_count,
        "consensus": consensus[:5],
        "common_risks": common_risks,
        "differences": differences[:5],
        "consensus_message": "所选研报尚未抽取出至少两家机构共同支持的相似观点。" if not consensus else None,
        "common_risks_message": "所选研报尚未发现被至少两篇报告共同提及的相似风险。" if not common_risks else None,
        "summary": f"已从{available_count}篇可用研报中比较所选{len(rows)}篇；共识仅保留至少两篇报告支持的相似内容。",
    }


@router.get("/{stock_code}/sentiment")
def sentiment_summary(stock_code: str, db: Session = Depends(get_db), _: User = Depends(get_current_user)) -> dict:
    company = get_company(db, stock_code)
    rows = db.execute(select(SentimentResult.sentiment, func.count(SentimentResult.id)).where(SentimentResult.company_id == company.id).group_by(SentimentResult.sentiment)).all()
    return {"company": company.name, "distribution": dict(rows)}


@router.get("/{stock_code}/forecasts")
def forecast_summary(stock_code: str, db: Session = Depends(get_db), _: User = Depends(get_current_user)) -> dict:
    company = get_company(db, stock_code)
    forecasts = db.scalars(select(EarningsForecast).where(EarningsForecast.company_id == company.id).order_by(EarningsForecast.forecast_year)).all()
    return {"company": company.name, "items": [{"institution": f.institution, "year": f.forecast_year, "revenue": f.revenue, "net_profit": f.net_profit, "eps": f.eps, "unit": f.unit} for f in forecasts]}
