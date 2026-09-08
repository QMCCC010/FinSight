from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import Any, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph
from sqlalchemy import func, select

from app.ai.index import hybrid_search
from app.ai.llm import invoke_text
from app.ai.broker_comparison import build_broker_citations, build_broker_comparison_answer
from app.ai.grounding import classify_research_intent, filter_unsupported_lines, intent_document_types, intent_label, is_professional_risk_item, validate_grounded_answer
from app.core.database import SessionLocal
from app.core.enums import DocumentStatus, MessageStatus, RunStatus, TrackingMode, TriggerType
from app.core.models import AgentCheckpoint, ChatMessage, Company, CrawlRun, Document, EarningsForecast, FinancialMetric, InvestmentRating, Opinion, RiskItem, SentimentResult
from app.services.crawl_runs import ALL_SOURCE_TYPES, run_covers


class FinancialAgentState(TypedDict, total=False):
    message_id: int
    question: str
    company_id: int | None
    stock_code: str | None
    company_name: str | None
    industry_name: str | None
    query_type: str
    knowledge_sufficient: bool
    knowledge_fresh: bool
    needs_collection: bool
    needs_refresh: bool
    refresh_source_types: list[str]
    knowledge_as_of: datetime | None
    collection_job_id: str | None
    refresh_job_id: str | None
    retrieved_documents: list[dict]
    tool_results: dict[str, Any]
    citations: list[dict]
    draft: str
    answer: str
    confidence: str
    missing_sources: list[str]
    errors: list[str]
    candidates: list[dict]
    intent: str
    research_intent: str
    retrieval_document_types: list[str]
    validation: dict[str, Any]


logger = logging.getLogger(__name__)


EXCHANGE_LABELS = {"SH": "上海证券交易所", "SZ": "深圳证券交易所", "BJ": "北京证券交易所"}


def _company_metadata(company: Company) -> dict[str, Any]:
    return {
        "id": company.id,
        "stock_code": company.stock_code,
        "name": company.name,
        "full_name": company.full_name,
        "exchange": company.exchange,
        "exchange_label": EXCHANGE_LABELS.get(company.exchange or "", company.exchange or "未知交易所"),
        "industry": company.industry,
        "listing_status": "A股基础名单已确认",
    }


def classify_refresh_sources(question: str) -> list[str]:
    """Select only the sources that can materially answer the user's question."""
    mappings = (
        ("RESEARCH_REPORT", ("研报", "机构", "评级", "目标价", "盈利预测", "预测", "观点", "风险")),
        ("NEWS", ("新闻", "资讯", "媒体", "事件")),
        ("ANNOUNCEMENT", ("公告", "披露", "年报", "半年报", "季报", "财报", "财务", "收入", "利润", "现金流", "eps")),
        ("SOCIAL", ("舆情", "情绪", "情感", "股吧", "市场讨论", "热度")),
    )
    compact = re.sub(r"\s+", "", question).lower()
    selected = [source_type for source_type, markers in mappings if any(marker in compact for marker in markers)]
    return selected or list(ALL_SOURCE_TYPES)


def _run_covers(run: CrawlRun, requested: list[str]) -> bool:
    return run_covers(run, requested)


def _update_message(message_id: int, **values: Any) -> bool:
    with SessionLocal() as db:
        message = db.get(ChatMessage, message_id)
        if not message or message.status == MessageStatus.CANCELLED:
            return False
        for key, value in values.items():
            setattr(message, key, value)
        db.commit()
        return True


def _update_metadata(message_id: int, **values: Any) -> None:
    with SessionLocal() as db:
        message = db.get(ChatMessage, message_id)
        if not message or message.status == MessageStatus.CANCELLED:
            return
        metadata = dict(message.analysis_metadata or {})
        metadata.update(values)
        message.analysis_metadata = metadata
        db.commit()


def classify_intent(question: str) -> str:
    compact = re.sub(r"\s+", "", question).lower()
    # System/help questions must be routed before the company-research
    # fallback. Match natural modal-verb variants, but require a system-like
    # subject so a company question such as “比亚迪汽车有哪些功能” is not
    # mistaken for a request to introduce the assistant.
    meta_markers = (
        "你是谁", "什么模型", "哪个模型", "介绍一下你", "介绍一下系统",
        "系统介绍", "系统功能", "你的能力", "怎么使用", "如何使用",
        "能问什么", "可以问什么",
    )
    meta_patterns = (
        r"(?:你|这个系统|本系统|系统|助手|finsight).{0,8}(?:能|能够|可以|支持).{0,8}(?:做什么|帮我做什么|提供什么|哪些功能|什么功能|问什么)",
        r"(?:你|这个系统|本系统|系统|助手|finsight).{0,8}(?:有|具备).{0,4}(?:什么|哪些).{0,2}(?:功能|能力)",
    )
    unsupported_markers = (
        "港股", "美股", "纳斯达克", "道琼斯", "标普500", "恒生指数", "腾讯", "腾讯控股",
        "阿里巴巴", "特斯拉", "苹果公司", "英伟达",
    )
    out_of_scope_markers = ("天气", "菜谱", "翻译", "写代码", "讲笑话", "旅游攻略")
    industry_markers = ("行业", "板块", "产业链", "赛道", "大盘", "市场整体")
    if any(marker in compact for marker in meta_markers) or any(
        re.search(pattern, compact) for pattern in meta_patterns
    ):
        return "SYSTEM_META"
    if any(marker in compact for marker in unsupported_markers):
        return "UNSUPPORTED_MARKET"
    if any(marker in compact for marker in out_of_scope_markers):
        return "OUT_OF_SCOPE"
    if any(marker in compact for marker in industry_markers):
        return "INDUSTRY_RESEARCH"
    return "COMPANY_RESEARCH"


def classify_question(state: FinancialAgentState) -> FinancialAgentState:
    _update_message(state["message_id"], status=MessageStatus.RESOLVING_ENTITY, progress=5, status_text="正在识别问题类型")
    intent = classify_intent(state["question"])
    if intent == "INDUSTRY_RESEARCH":
        with SessionLocal() as db:
            companies = db.scalars(select(Company)).all()
            if any(
                company.name in state["question"]
                or (company.full_name and company.full_name in state["question"])
                or company.stock_code in state["question"]
                for company in companies
            ):
                intent = "COMPANY_RESEARCH"
    research_intent = classify_research_intent(state["question"])
    _update_metadata(
        state["message_id"],
        scope_intent=intent,
        research_intent=research_intent,
        research_intent_label=intent_label(research_intent),
    )
    return {**state, "intent": intent, "research_intent": research_intent}


def answer_system_meta(state: FinancialAgentState) -> FinancialAgentState:
    from app.core.config import get_settings

    settings = get_settings()
    model_description = f"当前配置的对话模型是 `{settings.llm_model}`。" if settings.llm_model else "当前未配置远程对话模型，系统会使用规则抽取和证据拼接降级运行。"
    answer = (
        "我是 FinSight 金融研报智能分析助手，由 LangGraph 编排研究流程。"
        f"{model_description}\n\n"
        "我主要用于自动汇聚A股研报、新闻、公告和公开舆情，比较机构观点，并基于知识库证据回答公司研究问题。"
        "我不连接证券账户，不执行交易，输出也不构成投资建议。"
    )
    _update_message(
        state["message_id"],
        status=MessageStatus.COMPLETED,
        progress=100,
        status_text="已回答系统问题",
        answer=answer,
        citations=[],
        confidence="HIGH",
        data_as_of=datetime.now(),
        error=None,
    )
    _update_metadata(state["message_id"], route="SYSTEM_META", validation={"valid": True, "mode": "DIRECT"})
    return {**state, "answer": answer, "confidence": "HIGH"}


def _complete_direct_answer(state: FinancialAgentState, answer: str, status_text: str) -> FinancialAgentState:
    _update_message(
        state["message_id"],
        status=MessageStatus.COMPLETED,
        progress=100,
        status_text=status_text,
        answer=answer,
        citations=[],
        confidence="HIGH",
        data_as_of=datetime.now(),
        error=None,
    )
    return {**state, "answer": answer, "confidence": "HIGH"}


def answer_unsupported_market(state: FinancialAgentState) -> FinancialAgentState:
    return _complete_direct_answer(
        state,
        "当前版本的自动公司采集和结构化分析仅覆盖A股。你提到的标的可能属于港股、美股或其他市场，"
        "因此我不会把它误识别成A股公司。你可以改问明确的A股公司名称或6位股票代码。",
        "已说明当前市场覆盖范围",
    )


def answer_out_of_scope(state: FinancialAgentState) -> FinancialAgentState:
    return _complete_direct_answer(
        state,
        "这个问题不属于当前金融研报分析系统的工作范围。我可以帮助分析A股公司、行业、研报、公告、"
        "新闻、财务指标、盈利预测、评级、机构观点和风险，并为重要结论提供资料来源。",
        "已说明系统能力范围",
    )


def prepare_industry_research(state: FinancialAgentState) -> FinancialAgentState:
    question = state["question"]
    match = re.search(r"([\u4e00-\u9fffA-Za-z0-9]{2,16}?)(?:行业|板块|产业链|赛道)", question)
    industry_name = match.group(1) if match else "相关行业"
    _update_message(
        state["message_id"],
        status=MessageStatus.PROCESSING,
        progress=20,
        status_text=f"正在检索{industry_name}的跨公司资料",
    )
    _update_metadata(
        state["message_id"],
        resolved_subject={"type": "INDUSTRY", "name": industry_name},
        retrieval_scope={"company_filtered": False, "document_types": intent_document_types(state.get("research_intent"))},
    )
    return {**state, "industry_name": industry_name, "query_type": "INDUSTRY", "errors": []}


def _checkpoint(state: FinancialAgentState, node: str) -> None:
    serializable = {key: value for key, value in state.items() if key not in {"retrieved_documents"} or isinstance(value, list)}
    with SessionLocal() as db:
        checkpoint = db.scalar(select(AgentCheckpoint).where(AgentCheckpoint.message_id == state["message_id"]))
        if not checkpoint:
            checkpoint = AgentCheckpoint(message_id=state["message_id"], current_node=node, state=serializable)
            db.add(checkpoint)
        else:
            checkpoint.current_node = node
            checkpoint.state = serializable
        db.commit()


def resolve_entity(state: FinancialAgentState) -> FinancialAgentState:
    _update_message(state["message_id"], status=MessageStatus.RESOLVING_ENTITY, progress=10, status_text="正在识别公司与问题类型")
    question = state["question"]
    with SessionLocal() as db:
        message = db.get(ChatMessage, state["message_id"])
        if message and message.company_id:
            company = db.get(Company, message.company_id)
            candidates = [company] if company else []
        else:
            code_match = re.search(r"(?<!\d)(\d{6})(?!\d)", question)
            if code_match:
                company = db.scalar(select(Company).where(Company.stock_code == code_match.group(1)))
                candidates = [company] if company else []
            else:
                all_companies = db.scalars(select(Company)).all()
                candidates = [item for item in all_companies if item.name in question or (item.full_name and item.full_name in question) or any(alias in question for alias in (item.aliases or []) if len(alias) >= 2)]
            if not candidates:
                try:
                    company_count = db.scalar(select(func.count(Company.id))) or 0
                    if company_count < 100:
                        from app.services.company_master import refresh_company_master
                        refresh_company_master(db)
                        all_companies = db.scalars(select(Company)).all()
                        candidates = [item for item in all_companies if item.name in question or (item.full_name and item.full_name in question)]
                except Exception:
                    candidates = []
            # A follow-up such as “它的盈利预测呢” inherits the latest company in
            # the same conversation, but an explicit company/code always wins.
            if not candidates and message:
                previous = db.scalar(
                    select(ChatMessage)
                    .where(
                        ChatMessage.session_id == message.session_id,
                        ChatMessage.id < message.id,
                        ChatMessage.company_id.is_not(None),
                    )
                    .order_by(ChatMessage.id.desc())
                )
                if previous and previous.company_id:
                    company = db.get(Company, previous.company_id)
                    candidates = [company] if company else []
        unique = {item.id: item for item in candidates if item}
        candidates = list(unique.values())
        if len(candidates) > 1:
            data = [_company_metadata(item) for item in candidates[:8]]
            _update_message(state["message_id"], status=MessageStatus.NEEDS_CLARIFICATION, progress=10, status_text="检测到多个公司，请选择要分析的公司", clarification_candidates=data, error=None)
            _update_metadata(state["message_id"], entity_status="AMBIGUOUS", entity_candidates=data)
            return {**state, "candidates": data, "errors": ["company_ambiguous"]}
        if not candidates:
            _update_message(
                state["message_id"],
                status=MessageStatus.FAILED,
                progress=100,
                status_text="未能在A股基础名单中确认公司",
                clarification_candidates=[],
                error="请提供明确的A股公司名称或6位股票代码。未能确认只表示当前A股基础名单中未匹配到，不代表对该企业上市状态作出判断。",
            )
            _update_metadata(state["message_id"], entity_status="NOT_CONFIRMED")
            return {**state, "errors": ["company_not_found"]}
        company = candidates[0]
        company.last_queried_at = datetime.now()
        if company.tracking_mode == TrackingMode.INACTIVE:
            company.tracking_mode = TrackingMode.ON_DEMAND
        if message:
            message.company_id = company.id
            message.clarification_candidates = []
        db.commit()
        query_type = "LATEST" if any(word in question for word in ("最新", "最近", "近期", "今天", "现在", "目前", "当前", "近况", "截至")) else "GENERAL"
        resolved = _company_metadata(company)
        _update_metadata(
            state["message_id"],
            entity_status="CONFIRMED",
            resolved_subject={"type": "COMPANY", **resolved},
            query_recency=query_type,
        )
        return {**state, "company_id": company.id, "stock_code": company.stock_code, "company_name": company.name, "query_type": query_type, "errors": []}


def check_knowledge(state: FinancialAgentState) -> FinancialAgentState:
    _update_message(state["message_id"], status=MessageStatus.PROCESSING, progress=20, status_text="正在检查知识库完整性与时效性")
    with SessionLocal() as db:
        company = db.get(Company, state["company_id"])
        document_filter = (
            Document.company_id == company.id,
            Document.status == DocumentStatus.INDEXED,
            Document.is_deleted.is_(False),
        )
        count = db.scalar(select(func.count(Document.id)).where(*document_filter)) or 0
        knowledge_as_of = db.scalar(select(func.max(Document.published_at)).where(*document_filter)) or company.last_crawled_at
        now = datetime.now()
        source_types = classify_refresh_sources(state["question"])
        completed_runs = list(db.scalars(
            select(CrawlRun).where(
                CrawlRun.company_id == company.id,
                CrawlRun.status.in_([RunStatus.COMPLETED, RunStatus.PARTIAL]),
                CrawlRun.finished_at.is_not(None),
            ).order_by(CrawlRun.finished_at.desc()).limit(100)
        ).all())
        fresh = True
        for source_type in source_types:
            latest_run = next((run.finished_at for run in completed_runs if _run_covers(run, [source_type])), None)
            max_age = timedelta(minutes=30) if source_type in {"NEWS", "SOCIAL"} else timedelta(hours=24)
            if not latest_run or latest_run < now - max_age:
                fresh = False
                break
        needs_collection = count == 0
        needs_refresh = count > 0 and state.get("query_type") == "LATEST" and not fresh
        if needs_refresh:
            _update_message(
                state["message_id"],
                status_text="知识库已有资料，将先回答并在后台更新相关来源",
            )
        _update_metadata(
            state["message_id"],
            knowledge={
                "document_count": count,
                "sufficient": count > 0,
                "fresh": fresh,
                "as_of": knowledge_as_of.isoformat() if knowledge_as_of else None,
                "background_refresh_required": needs_refresh,
                "refresh_source_types": source_types,
            },
        )
        return {
            **state,
            "knowledge_sufficient": count > 0,
            "knowledge_fresh": fresh,
            "needs_collection": needs_collection,
            "needs_refresh": needs_refresh,
            "refresh_source_types": source_types,
            "knowledge_as_of": knowledge_as_of,
        }


def start_collection(state: FinancialAgentState) -> FinancialAgentState:
    with SessionLocal() as db:
        # A first-time company query may reuse work that is already executing, but
        # it must never sit behind an arbitrary old scheduled job.
        existing = db.scalar(
            select(CrawlRun)
            .where(
                CrawlRun.company_id == state["company_id"],
                (
                    (CrawlRun.status == RunStatus.RUNNING)
                    | (
                        (CrawlRun.status == RunStatus.QUEUED)
                        & (CrawlRun.trigger_type == TriggerType.ON_DEMAND)
                        & (CrawlRun.updated_at >= datetime.now() - timedelta(minutes=45))
                    )
                ),
            )
            .order_by(CrawlRun.id.desc())
        )
        if existing:
            run = existing
        else:
            run = CrawlRun(
                job_id=str(uuid4()),
                company_id=state["company_id"],
                source_type=None,
                source_types=list(ALL_SOURCE_TYPES),
                trigger_type=TriggerType.ON_DEMAND,
                parent_message_id=state["message_id"],
                status=RunStatus.QUEUED,
                stage="首次资料采集等待中",
            )
            db.add(run)
            db.flush()
        message = db.get(ChatMessage, state["message_id"])
        message.crawl_run_id = run.id
        message.status = MessageStatus.COLLECTING
        message.progress = 30
        message.status_text = f"正在为{state['company_name']}自动采集研报、新闻、公告和舆情"
        db.commit()
        if not existing:
            from app.worker.tasks import collect_company
            collect_company.apply_async(args=[run.id, ALL_SOURCE_TYPES], queue="collection_ondemand", priority=9)
        next_state = {**state, "collection_job_id": run.job_id}
        _checkpoint(next_state, "pause_for_collection")
        return next_state


def start_background_refresh(state: FinancialAgentState) -> FinancialAgentState:
    """Queue a freshness update without putting the answer into COLLECTING."""
    requested = state.get("refresh_source_types") or list(ALL_SOURCE_TYPES)
    now = datetime.now()
    with SessionLocal() as db:
        candidates = list(
            db.scalars(
                select(CrawlRun)
                .where(
                    CrawlRun.company_id == state["company_id"],
                    (
                        (CrawlRun.status == RunStatus.RUNNING)
                        | ((CrawlRun.status == RunStatus.QUEUED) & (CrawlRun.trigger_type == TriggerType.ON_DEMAND))
                    ),
                    CrawlRun.updated_at >= now - timedelta(minutes=45),
                )
                .order_by(CrawlRun.id.desc())
            ).all()
        )
        run = next((item for item in candidates if _run_covers(item, requested)), None)
        created = run is None
        if created:
            message = db.get(ChatMessage, state["message_id"])
            run = CrawlRun(
                job_id=str(uuid4()),
                company_id=state["company_id"],
                source_type=requested[0] if len(requested) == 1 else None,
                source_types=list(requested),
                trigger_type=TriggerType.ON_DEMAND,
                requested_by_user_id=message.user_id if message else None,
                parent_message_id=state["message_id"],
                status=RunStatus.QUEUED,
                stage="后台增量更新等待中",
            )
            db.add(run)
            db.flush()
        message = db.get(ChatMessage, state["message_id"])
        if message:
            message.refresh_run_id = run.id
            message.refresh_status = "RUNNING" if run.status == RunStatus.RUNNING else "QUEUED"
            message.refresh_status_text = "后台正在更新相关资料" if run.status == RunStatus.RUNNING else "相关资料已进入后台更新队列"
            message.refresh_requested_at = now
            message.refresh_completed_at = None
        db.commit()
        if created:
            try:
                from app.worker.tasks import collect_company
                collect_company.apply_async(args=[run.id, requested], queue="collection", priority=5)
            except Exception as exc:
                logger.exception("Failed to enqueue background refresh for message %s", state["message_id"])
                if message:
                    message.refresh_status = "FAILED"
                    message.refresh_status_text = "后台更新提交失败，当前回答不受影响"
                    message.refresh_completed_at = datetime.now()
                    db.commit()
        return {**state, "refresh_job_id": run.job_id}


def pause_for_collection(state: FinancialAgentState) -> FinancialAgentState:
    return state


def retrieve_context(state: FinancialAgentState) -> FinancialAgentState:
    research_intent = state.get("research_intent") or classify_research_intent(state["question"])
    document_types = intent_document_types(research_intent)
    label = intent_label(research_intent)
    _update_message(
        state["message_id"],
        status=MessageStatus.PROCESSING,
        progress=55,
        status_text=f"正在按“{label}”路径检索相关证据",
    )
    with SessionLocal() as db:
        items = hybrid_search(
            db,
            state["question"],
            state.get("company_id"),
            limit=8 if state.get("query_type") == "INDUSTRY" else 6,
            document_types=document_types,
        )
        fallback_used = False
        # A narrow intent can be correctly recognized while the corresponding
        # source has not arrived yet. Fall back to the same company's other
        # documents instead of silently searching a different company.
        if not items and set(document_types) != set(ALL_SOURCE_TYPES):
            items = hybrid_search(
                db,
                state["question"],
                state.get("company_id"),
                limit=8 if state.get("query_type") == "INDUSTRY" else 6,
            )
            fallback_used = bool(items)

        # Defence in depth for stale/misconfigured vector indexes: validate
        # Milvus/FAISS hits against MySQL ownership before they reach the LLM.
        document_ids = {int(item.get("document_id") or 0) for item in items}
        allowed_ids = set(document_ids)
        if state.get("company_id") is not None and document_ids:
            allowed_ids = set(db.scalars(
                select(Document.id).where(
                    Document.id.in_(document_ids),
                    Document.company_id == state["company_id"],
                    Document.status == DocumentStatus.INDEXED,
                    Document.is_deleted.is_(False),
                )
            ).all())
        items = [item for item in items if int(item.get("document_id") or 0) in allowed_ids]

    retrieval_scope = {
        "company_filtered": state.get("company_id") is not None,
        "company_id": state.get("company_id"),
        "document_types": document_types,
        "intent_fallback_used": fallback_used,
        "evidence_count": len(items),
    }
    retrieved_source_types = {item.get("source_type") for item in items}
    intent_missing_sources = [source_type for source_type in document_types if source_type not in retrieved_source_types]
    missing_sources = list(dict.fromkeys([*state.get("missing_sources", []), *intent_missing_sources]))
    _update_metadata(state["message_id"], retrieval_scope=retrieval_scope)
    return {
        **state,
        "research_intent": research_intent,
        "retrieval_document_types": document_types,
        "retrieved_documents": items,
        "citations": [{key: value for key, value in item.items() if key != "score"} for item in items],
        "missing_sources": missing_sources,
    }


def select_tools(state: FinancialAgentState) -> FinancialAgentState:
    question = state["question"]
    from app.ai.tools import analyze_news_sentiment, analyze_social_sentiment, compare_research_reports, get_broker_forecasts, get_company_overview, get_financial_metrics, get_market_prices
    stock_code = state.get("stock_code")
    if not stock_code:
        return {**state, "tool_results": {"scope": "INDUSTRY", "industry": state.get("industry_name")}}
    research_intent = state.get("research_intent") or classify_research_intent(question)
    result: dict[str, Any] = {"company": get_company_overview.invoke({"stock_code": stock_code})}
    if research_intent in {"MARKET_TREND", "OVERVIEW"}:
        result["market_prices"] = get_market_prices.invoke({"stock_code": stock_code})
    if research_intent in {"FINANCIAL", "OVERVIEW"}:
        result["financial_metrics"] = get_financial_metrics.invoke({"stock_code": stock_code})
    if research_intent in {"BROKER_RESEARCH", "RISK", "OVERVIEW"}:
        result["broker_data"] = get_broker_forecasts.invoke({"stock_code": stock_code})
        result["report_comparison"] = compare_research_reports.invoke({"stock_code": stock_code})
    if research_intent in {"NEWS_EVENTS", "SENTIMENT", "OVERVIEW"}:
        result["news_sentiment"] = analyze_news_sentiment.invoke({"stock_code": stock_code})
    if research_intent in {"SENTIMENT", "OVERVIEW"}:
        result["social_sentiment"] = analyze_social_sentiment.invoke({"stock_code": stock_code})
    risks = result.get("report_comparison", {}).get("risks", [])
    result["risks"] = [row["content"] for row in risks]

    citations = list(state.get("citations", []))
    if research_intent == "BROKER_RESEARCH":
        structured_citations = build_broker_citations(result.get("broker_data", {}))
        if structured_citations:
            # Use concise, provenance-preserving structured rows rather than
            # arbitrary PDF chunks (which often land on disclaimers).
            citations = structured_citations
    if research_intent == "RISK" and risks:
        # Structured risk items were extracted from a specific source document,
        # so they are cleaner evidence than arbitrary PDF chunks while retaining
        # full provenance. Prefer distinct risk statements, newest first.
        structured_citations = []
        seen_risks: set[str] = set()
        for row in risks:
            content = re.sub(r"\s+", " ", row.get("content") or "").strip()
            if (
                content in seen_risks
                or not row.get("document_id")
                or not is_professional_risk_item(content, row.get("source_type"))
            ):
                continue
            seen_risks.add(content)
            structured_citations.append({
                "document_id": row["document_id"],
                "title": row.get("title") or "研报风险提示",
                "source_type": row.get("source_type") or "RESEARCH_REPORT",
                "source_url": row.get("source_url") or "",
                "published_at": row.get("published_at"),
                "page": None,
                "quote": content,
                "source_name": row.get("source_name"),
            })
            if len(structured_citations) >= 6:
                break
        if structured_citations:
            citations = structured_citations
    prices = result.get("market_prices", {}).get("prices", [])
    if prices:
        latest = prices[0]
        source = result["market_prices"].get("source") or "公开行情数据"
        fields = []
        for label, key, unit in (("收盘价", "close", "元"), ("最高", "high", "元"), ("最低", "low", "元"), ("涨跌幅", "change_pct", "%")):
            if latest.get(key) is not None:
                fields.append(f"{label}{latest[key]}{unit}")
        quote = f"{state.get('company_name')} {latest['trade_date']} " + "，".join(fields) + "。"
        citations.append({
            "document_id": 0,
            "title": f"{state.get('company_name')}日线行情",
            "source_type": "MARKET_DATA",
            "source_url": "",
            "published_at": latest["trade_date"],
            "page": None,
            "quote": quote,
            "source_name": source,
        })
    _update_metadata(state["message_id"], selected_tools=sorted(result.keys()))
    return {**state, "tool_results": result, "citations": citations}


def analyze_evidence(state: FinancialAgentState) -> FinancialAgentState:
    citations = state.get("citations", [])
    if not citations:
        return {**state, "errors": [*state.get("errors", []), "no_evidence"]}
    snippets = "\n".join(f"[{index}] {item['quote']}" for index, item in enumerate(citations, 1))
    risks = state.get("tool_results", {}).get("risks", [])
    subject = state.get("company_name") or state.get("industry_name") or "研究主题"
    draft = f"检索到{len(citations)}条与{subject}相关的证据。\n{snippets}\n主要风险：" + ("；".join(risks[:5]) if risks else "当前结构化数据未提取到明确风险。")
    return {**state, "draft": draft}


def validate_answer(state: FinancialAgentState) -> FinancialAgentState:
    citations = state.get("citations", [])
    if not citations:
        return {**state, "errors": [*state.get("errors", []), "citation_required"]}
    source_types = {item["source_type"] for item in citations}
    confidence = "HIGH" if len(source_types) >= 2 and "ANNOUNCEMENT" in source_types else "MEDIUM" if len(citations) >= 2 else "LOW"
    return {**state, "confidence": confidence}


def _evidence_only_answer(state: FinancialAgentState) -> str:
    bullets = [f"- [{index}] {item['quote'][:260]}" for index, item in enumerate(state["citations"], 1)]
    answer = (
        "### 简明结论\n\n当前知识库检索到与本次问题相关的资料。"
        "为避免加入缺少原文支持的内容，以下按证据原文归纳。\n\n"
        "### 主要依据\n\n" + "\n".join(bullets)
    )
    answer += "\n\n### 风险与不确定性\n\n- 请结合上述原文证据中的相关提示判断；当前回答不补充无直接来源的判断。"
    answer += "\n\n> 仅供研究辅助，不构成投资建议。"
    return answer


def generate_answer(state: FinancialAgentState) -> FinancialAgentState:
    _update_message(state["message_id"], status=MessageStatus.ANSWERING, progress=80, status_text="正在基于证据生成回答")
    evidence = "\n".join(f"[{index}] {item['title']}：{item['quote']}" for index, item in enumerate(state["citations"], 1))
    cutoff = state.get("knowledge_as_of")
    freshness_instruction = (
        f"现有证据的资料截止时间约为{cutoff:%Y-%m-%d %H:%M}，后台更新正在进行。"
        "必须明确这是基于现有知识库的回答，不得把它表述为实时或最新完整结果。"
        if state.get("needs_refresh") and cutoff
        else ""
    )
    prompt = f"""你是金融研究辅助Agent。当前分析路径是“{intent_label(state.get('research_intent'))}”。仅根据证据回答，不补充未经证据支持的公司事实，不给出直接买卖或仓位建议。
必须区分已经发生的事实和机构预测。回答包含：简明结论、主要依据、机构观点或分歧、风险与不确定性，并用[1]格式引用。每个包含财务数字、价格、百分比或日期的事实句都必须引用能直接支持该数字的证据，而且数字不得自行换算。
{freshness_instruction}
问题：{state['question']}
结构化工具结果：{state.get('tool_results', {})}
证据：
{evidence}
"""
    broker_data = state.get("tool_results", {}).get("broker_data", {})
    structured_broker_answer = state.get("research_intent") == "BROKER_RESEARCH" and bool(build_broker_citations(broker_data))
    if structured_broker_answer:
        answer = build_broker_comparison_answer(state.get("company_name") or "该公司", broker_data, state["citations"])
        generated_by_llm = False
    else:
        answer = invoke_text(prompt)
        generated_by_llm = bool(answer)
    if not answer:
        answer = _evidence_only_answer(state)

    validation = validate_grounded_answer(answer, state["citations"])
    if not validation["valid"]:
        # Do not make a second remote-model call here. A repair request can
        # double tail latency and exceed the Celery task budget. Deterministic
        # evidence fallback is both safer and bounded.
        original_issues = validation["issues"]
        filtered_answer, filtered_validation = filter_unsupported_lines(answer, state["citations"])
        if generated_by_llm and filtered_validation["valid"] and filtered_validation["referenced_citation_count"] and len(filtered_answer) >= 80:
            answer = filtered_answer
            validation = {
                **filtered_validation,
                "mode": "FILTERED_LLM",
                "original_issue_count": len(original_issues),
            }
        else:
            answer = _evidence_only_answer(state)
            validation = {
                **validate_grounded_answer(answer, state["citations"]),
                "mode": "EVIDENCE_FALLBACK",
                "original_issue_count": len(original_issues),
            }
    else:
        validation = {
            **validation,
            "mode": "STRUCTURED_COMPARISON" if structured_broker_answer else "LLM" if generated_by_llm else "EVIDENCE_FALLBACK",
        }
    _update_metadata(state["message_id"], validation=validation)
    data_as_of = state.get("knowledge_as_of") or datetime.now()
    final_status = MessageStatus.PARTIAL if state.get("missing_sources") else MessageStatus.COMPLETED
    status_text = "分析完成，相关资料正在后台更新" if state.get("needs_refresh") else "分析完成"
    confidence = state.get("confidence", "LOW")
    if validation.get("mode") == "EVIDENCE_FALLBACK" and confidence == "HIGH":
        confidence = "MEDIUM"
    _update_message(state["message_id"], status=final_status, progress=100, status_text=status_text, answer=answer, citations=state["citations"], confidence=confidence, data_as_of=data_as_of, missing_sources=state.get("missing_sources", []), error=None)
    return {**state, "answer": answer, "validation": validation, "confidence": confidence}


def partial_answer(state: FinancialAgentState) -> FinancialAgentState:
    missing = "、".join(state.get("missing_sources", []))
    next_state = {**state, "draft": f"部分来源未成功获取：{missing}。\n" + state.get("draft", "")}
    return generate_answer(next_state)


def handle_failure(state: FinancialAgentState) -> FinancialAgentState:
    if "company_ambiguous" in state.get("errors", []) or "company_not_found" in state.get("errors", []):
        return state
    _update_message(state["message_id"], status=MessageStatus.FAILED, progress=100, status_text="没有足够证据生成回答", error="当前知识库资料不足，请稍后重试或由管理员检查数据源。")
    return state


def route_after_entity(state: FinancialAgentState) -> str:
    return "handle_failure" if state.get("errors") else "check_knowledge"


def route_after_classification(state: FinancialAgentState) -> str:
    return {
        "SYSTEM_META": "answer_system_meta",
        "UNSUPPORTED_MARKET": "answer_unsupported_market",
        "OUT_OF_SCOPE": "answer_out_of_scope",
        "INDUSTRY_RESEARCH": "prepare_industry_research",
    }.get(state.get("intent", "COMPANY_RESEARCH"), "resolve_entity")


def route_after_check(state: FinancialAgentState) -> str:
    if state.get("needs_collection"):
        return "start_collection"
    return "start_background_refresh" if state.get("needs_refresh") else "retrieve_context"


def route_after_retrieval(state: FinancialAgentState) -> str:
    return "select_tools" if state.get("retrieved_documents") else "handle_failure"


def route_after_analysis(state: FinancialAgentState) -> str:
    return "handle_failure" if state.get("errors") else "validate_answer"


def route_after_validation(state: FinancialAgentState) -> str:
    return "partial_answer" if state.get("missing_sources") else "generate_answer"


def build_graph():
    graph = StateGraph(FinancialAgentState)
    nodes = {
        "classify_question": classify_question,
        "answer_system_meta": answer_system_meta,
        "answer_unsupported_market": answer_unsupported_market,
        "answer_out_of_scope": answer_out_of_scope,
        "prepare_industry_research": prepare_industry_research,
        "resolve_entity": resolve_entity,
        "check_knowledge": check_knowledge,
        "start_collection": start_collection,
        "start_background_refresh": start_background_refresh,
        "pause_for_collection": pause_for_collection,
        "retrieve_context": retrieve_context,
        "select_tools": select_tools,
        "analyze_evidence": analyze_evidence,
        "validate_answer": validate_answer,
        "generate_answer": generate_answer,
        "partial_answer": partial_answer,
        "handle_failure": handle_failure,
    }
    for name, function in nodes.items():
        graph.add_node(name, function)
    graph.add_edge(START, "classify_question")
    graph.add_conditional_edges("classify_question", route_after_classification)
    graph.add_edge("answer_system_meta", END)
    graph.add_edge("answer_unsupported_market", END)
    graph.add_edge("answer_out_of_scope", END)
    graph.add_edge("prepare_industry_research", "retrieve_context")
    graph.add_conditional_edges("resolve_entity", route_after_entity)
    graph.add_conditional_edges("check_knowledge", route_after_check)
    graph.add_edge("start_collection", "pause_for_collection")
    graph.add_edge("start_background_refresh", "retrieve_context")
    graph.add_edge("pause_for_collection", END)
    graph.add_conditional_edges("retrieve_context", route_after_retrieval)
    graph.add_edge("select_tools", "analyze_evidence")
    graph.add_conditional_edges("analyze_evidence", route_after_analysis)
    graph.add_conditional_edges("validate_answer", route_after_validation)
    graph.add_edge("generate_answer", END)
    graph.add_edge("partial_answer", END)
    graph.add_edge("handle_failure", END)
    return graph.compile()


def run_message(message_id: int) -> dict:
    with SessionLocal() as db:
        message = db.get(ChatMessage, message_id)
        if not message:
            raise ValueError(f"message {message_id} not found")
        state: FinancialAgentState = {"message_id": message.id, "question": message.question or "", "company_id": message.company_id, "missing_sources": message.missing_sources or [], "errors": []}
    return build_graph().invoke(state)
