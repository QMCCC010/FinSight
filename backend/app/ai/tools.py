from __future__ import annotations

from langchain_core.tools import tool
from sqlalchemy import func, select

from app.ai.index import hybrid_search
from app.ai.reporting import build_report
from app.core.database import SessionLocal
from app.core.enums import DocumentStatus
from app.core.models import Company, Document, EarningsForecast, Evidence, FinancialMetric, InvestmentRating, MarketPrice, Opinion, RiskItem, SentimentResult


def _company(db, stock_code: str) -> Company:
    company = db.scalar(select(Company).where(Company.stock_code == stock_code))
    if not company:
        raise ValueError(f"未找到股票代码 {stock_code}")
    return company


@tool
def search_documents(query: str, stock_code: str) -> list[dict]:
    """在指定A股公司的共享金融知识库中执行混合检索。"""
    with SessionLocal() as db:
        return hybrid_search(db, query, _company(db, stock_code).id, limit=6)


@tool
def get_company_overview(stock_code: str) -> dict:
    """查询公司基本信息和四类文档数量。"""
    with SessionLocal() as db:
        company = _company(db, stock_code)
        counts = dict(db.execute(select(Document.document_type, func.count(Document.id)).where(
            Document.company_id == company.id,
            Document.status == DocumentStatus.INDEXED,
            Document.is_deleted.is_(False),
        ).group_by(Document.document_type)).all())
        return {"stock_code": company.stock_code, "name": company.name, "exchange": company.exchange, "industry": company.industry, "tracking_mode": company.tracking_mode, "listing_status": "A股基础名单已确认", "document_counts": counts}


@tool
def get_market_prices(stock_code: str) -> dict:
    """查询指定A股公司最近的日线行情；结果是结构化事实而非投资建议。"""
    with SessionLocal() as db:
        company = _company(db, stock_code)
        rows = list(db.scalars(
            select(MarketPrice)
            .where(MarketPrice.company_id == company.id)
            .order_by(MarketPrice.trade_date.desc())
            .limit(30)
        ).all())
        return {
            "source": rows[0].source_name if rows else None,
            "prices": [
                {
                    "trade_date": row.trade_date.date().isoformat(),
                    "open": row.open,
                    "close": row.close,
                    "high": row.high,
                    "low": row.low,
                    "volume": row.volume,
                    "amount": row.amount,
                    "change_pct": row.change_pct,
                }
                for row in rows
            ],
        }


@tool
def get_financial_metrics(stock_code: str) -> list[dict]:
    """查询公司已经抽取的实际财务指标，预测数据不包含在内。"""
    with SessionLocal() as db:
        company = _company(db, stock_code)
        rows = db.execute(
            select(FinancialMetric, Document)
            .join(Document, Document.id == FinancialMetric.document_id)
            .where(
                FinancialMetric.company_id == company.id,
                Document.status == DocumentStatus.INDEXED,
                Document.is_deleted.is_(False),
            )
            .order_by(Document.published_at.desc(), FinancialMetric.id.desc())
            .limit(20)
        ).all()
        result = []
        for row, document in rows:
            evidence = db.scalar(select(Evidence).where(
                Evidence.entity_type == "financial_metric",
                Evidence.entity_id == row.id,
            ).order_by(Evidence.id.desc()))
            result.append({
                "name": row.metric_name,
                "value": row.normalized_value,
                "raw_value": row.raw_value,
                "unit": row.unit,
                "period": row.period,
                "yoy": row.yoy,
                "document_id": document.id,
                "title": document.title,
                "source_type": document.document_type,
                "source_url": document.source_url,
                "published_at": document.published_at.isoformat() if document.published_at else None,
                "source_name": document.source_name,
                "evidence": evidence.quote if evidence else "",
                "page": evidence.page_number if evidence else None,
            })
        return result


@tool
def get_broker_forecasts(stock_code: str) -> dict:
    """查询券商盈利预测、评级与目标价。"""
    with SessionLocal() as db:
        company = _company(db, stock_code)
        forecasts = db.execute(
            select(EarningsForecast, Document)
            .join(Document, Document.id == EarningsForecast.document_id)
            .where(
                EarningsForecast.company_id == company.id,
                Document.document_type == "RESEARCH_REPORT",
                Document.status == DocumentStatus.INDEXED,
                Document.is_deleted.is_(False),
            )
            .order_by(Document.published_at.desc(), EarningsForecast.institution, EarningsForecast.forecast_year)
            .limit(80)
        ).all()
        ratings = db.execute(
            select(InvestmentRating, Document)
            .join(Document, Document.id == InvestmentRating.document_id)
            .where(
                InvestmentRating.company_id == company.id,
                Document.document_type == "RESEARCH_REPORT",
                Document.status == DocumentStatus.INDEXED,
                Document.is_deleted.is_(False),
            )
            .order_by(Document.published_at.desc(), InvestmentRating.id.desc())
            .limit(30)
        ).all()

        def provenance(document: Document, entity_type: str, entity_id: int) -> dict:
            evidence = db.scalar(
                select(Evidence)
                .where(Evidence.entity_type == entity_type, Evidence.entity_id == entity_id)
                .order_by(Evidence.id.desc())
            )
            return {
                "document_id": document.id,
                "title": document.title,
                "source_type": document.document_type,
                "source_url": document.source_url,
                "published_at": document.published_at.isoformat() if document.published_at else None,
                "source_name": document.source_name,
                "evidence": evidence.quote if evidence else "",
                "page": evidence.page_number if evidence else None,
            }

        return {
            "forecasts": [
                {
                    "institution": row.institution,
                    "year": row.forecast_year,
                    "revenue": row.revenue,
                    "net_profit": row.net_profit,
                    "eps": row.eps,
                    "unit": row.unit,
                    **provenance(document, "earnings_forecast", row.id),
                }
                for row, document in forecasts
            ],
            "ratings": [
                {
                    "institution": row.institution,
                    "rating": row.original_rating,
                    "normalized": row.normalized_rating,
                    "target_price": row.target_price,
                    **provenance(document, "investment_rating", row.id),
                }
                for row, document in ratings
            ],
        }


@tool
def compare_research_reports(stock_code: str) -> dict:
    """归纳同一公司的研报观点、风险和机构评级差异。"""
    with SessionLocal() as db:
        company = _company(db, stock_code)
        opinions = db.scalars(
            select(Opinion)
            .join(Document, Document.id == Opinion.document_id)
            .where(
                Opinion.company_id == company.id,
                Document.document_type == "RESEARCH_REPORT",
                Document.status == DocumentStatus.INDEXED,
                Document.is_deleted.is_(False),
            )
            .order_by(Document.published_at.desc(), Opinion.id.desc())
            .limit(30)
        ).all()
        risks = db.scalars(
            select(RiskItem)
            .join(Document, Document.id == RiskItem.document_id)
            .where(
                RiskItem.company_id == company.id,
                Document.document_type.in_(["RESEARCH_REPORT", "ANNOUNCEMENT"]),
                Document.status == DocumentStatus.INDEXED,
                Document.is_deleted.is_(False),
            )
            .order_by(Document.published_at.desc(), RiskItem.id.desc())
            .limit(50)
        ).all()
        def provenance(document_id: int) -> dict:
            document = db.get(Document, document_id)
            if not document:
                return {"document_id": document_id}
            return {
                "document_id": document.id,
                "title": document.title,
                "source_type": document.document_type,
                "source_url": document.source_url,
                "published_at": document.published_at.isoformat() if document.published_at else None,
                "source_name": document.source_name,
            }

        return {
            "opinions": [
                {"type": row.opinion_type, "content": row.content, "sentiment": row.sentiment, **provenance(row.document_id)}
                for row in opinions
            ],
            "risks": [
                {"category": row.category, "content": row.content, **provenance(row.document_id)}
                for row in risks
            ],
        }


def _sentiment(stock_code: str, document_type: str) -> dict:
    with SessionLocal() as db:
        company = _company(db, stock_code)
        rows = db.execute(
            select(SentimentResult.sentiment, func.count(SentimentResult.id), func.avg(SentimentResult.score))
            .join(Document, Document.id == SentimentResult.document_id)
            .where(
                SentimentResult.company_id == company.id,
                Document.document_type == document_type,
                Document.status == DocumentStatus.INDEXED,
                Document.is_deleted.is_(False),
            )
            .group_by(SentimentResult.sentiment)
        ).all()
        return {"distribution": {row[0]: row[1] for row in rows}, "average_scores": {row[0]: float(row[2]) for row in rows}}


@tool
def analyze_news_sentiment(stock_code: str) -> dict:
    """汇总指定公司的财经新闻情感结果。"""
    return _sentiment(stock_code, "NEWS")


@tool
def analyze_social_sentiment(stock_code: str) -> dict:
    """汇总指定公司的公开社交舆情；结果仅代表市场讨论。"""
    return _sentiment(stock_code, "SOCIAL")


@tool
def generate_research_brief(stock_code: str) -> str:
    """基于共享知识库生成带来源的公司研究简报。"""
    with SessionLocal() as db:
        company = _company(db, stock_code)
        content, _ = build_report(db, company, "COMPANY_BRIEF", None, None, [])
        return content


FINANCIAL_TOOLS = [search_documents, get_company_overview, get_market_prices, get_financial_metrics, get_broker_forecasts, compare_research_reports, analyze_news_sentiment, analyze_social_sentiment, generate_research_brief]
