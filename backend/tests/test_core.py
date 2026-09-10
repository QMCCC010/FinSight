from datetime import datetime
import asyncio
from io import BytesIO
from types import SimpleNamespace
import time
import zipfile

import pymupdf as fitz
import pytest
from billiard.exceptions import SoftTimeLimitExceeded
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.ai.extraction import clean_rating, extract_document, normalize_rating, rule_based_extract
from app.ai.broker_comparison import build_broker_citations, build_broker_comparison_answer
from app.api.routers.analysis import _cluster_statements, _forecast_ranges
from app.api.routers.companies import _event_similarity
from app.ai.graph import _build_financial_citations, _build_market_citations, _citation_data_as_of, _compact_prompt_value, _company_hit_relevant, _matches_common_company_short_name, _retrieval_question, _risk_evidence_fallback, _tool_context_for_prompt, classify_intent, classify_refresh_sources, route_after_check, route_after_classification, route_after_entity, route_after_retrieval
from app.ai.grounding import classify_research_intent, contains_direct_trading_advice, filter_unsupported_lines, intent_document_types, is_professional_risk_item, validate_grounded_answer
from app.ai.llm import LLMCallResult
from app.ai.milvus_store import _filter_expression
from app.ai.conversation_memory import approximate_tokens, format_conversation_context, load_conversation_context
from app.ai.conversation_memory_store import _memory_filter, _truncate_utf8
from app.ai.router import route_request
from app.ai.reporting import _clusters, _metric_label, _safe_llm_summary
from app.collector.demo_data import demo_items
from app.collector.adapters.announcement import cninfo_pdf_url
from app.core.security import hash_password, verify_password
from app.core.database import Base
from app.core.models import ChatMessage, ChatSession, ConversationSummary, User
from app.core.schemas import CreateChatSession, CreateMessage, ReportComparisonRequest
from app.services.parser import assess_text_quality, chunk_pages, normalize_text, parse_pdf
from app.services.report_export import export_docx, export_pdf
from app.services.crawl_runs import normalize_source_types, run_covers, run_source_types


def test_conversation_token_estimate_counts_chinese_conservatively():
    assert approximate_tokens("你好abcde") == 4


def test_conversation_memory_filter_is_strictly_scoped():
    expression = _memory_filter(7, 19, [1, 2])
    assert "user_id == 7" in expression
    assert "session_id == 19" in expression
    assert "message_id not in [1, 2]" in expression


def test_conversation_memory_truncates_by_utf8_bytes():
    value = "中" * 4000
    truncated = _truncate_utf8(value, 8192)
    assert len(truncated.encode("utf-8")) <= 8192
    assert truncated


def test_formatted_conversation_contains_answers_and_invalidates_old_citations():
    context = {
        "strategy": "FULL_HISTORY",
        "summary": None,
        "recalled_turns": [],
        "recent_turns": [{"message_id": 1, "user": "第一问", "assistant": "第一答[1]"}],
    }
    rendered = format_conversation_context(context)
    assert "第一问" in rendered
    assert "第一答" in rendered
    assert "引用编号已失效" in rendered


def test_formatted_conversation_obeys_purpose_specific_budget():
    context = {
        "strategy": "SUMMARY_RECALL",
        "summary": {"summary": "旧摘要" * 1000, "discussed_topics": ["风险"] * 10},
        "recalled_turns": [],
        "recent_turns": [
            {"message_id": index, "user": f"第{index}问", "assistant": "很长的回答" * 1000}
            for index in range(1, 5)
        ],
    }
    rendered = format_conversation_context(
        context,
        token_budget=600,
        recent_turns=3,
        assistant_char_limit=240,
    )
    assert approximate_tokens(rendered) <= 600
    assert "第4问" in rendered


def _conversation_test_db() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def test_short_conversation_loads_all_questions_and_answers():
    with _conversation_test_db() as db:
        user = User(username="memory-user", password_hash="x", role="ANALYST")
        db.add(user)
        db.flush()
        session = ChatSession(user_id=user.id, title="memory")
        other_session = ChatSession(user_id=user.id, title="other")
        db.add_all([session, other_session])
        db.flush()
        db.add_all([
            ChatMessage(
                session_id=session.id, user_id=user.id, role="assistant",
                question="第一问", answer="第一答的第二项是乙", status="COMPLETED",
            ),
            ChatMessage(
                session_id=other_session.id, user_id=user.id, role="assistant",
                question="不应出现的问题", answer="不应出现的回答", status="COMPLETED",
            ),
        ])
        db.flush()
        current = ChatMessage(
            session_id=session.id, user_id=user.id, role="assistant",
            question="第二项是什么", status="QUEUED",
        )
        db.add(current)
        db.flush()

        context = load_conversation_context(db, current)
        rendered = format_conversation_context(context)
        assert context["strategy"] == "FULL_HISTORY"
        assert "第一问" in rendered and "第一答的第二项是乙" in rendered
        assert "不应出现" not in rendered


def test_long_conversation_uses_summary_recall_and_budget(monkeypatch):
    import app.ai.conversation_memory as memory_module
    import app.ai.conversation_memory_store as memory_store

    monkeypatch.setattr(memory_module, "get_settings", lambda: SimpleNamespace(
        conversation_history_token_budget=500,
        conversation_recent_turns=2,
        conversation_recall_limit=1,
        conversation_summary_trigger_turns=3,
    ))
    with _conversation_test_db() as db:
        user = User(username="long-memory-user", password_hash="x", role="ANALYST")
        db.add(user)
        db.flush()
        session = ChatSession(user_id=user.id, title="long-memory")
        db.add(session)
        db.flush()
        turns = []
        for index in range(6):
            turn = ChatMessage(
                session_id=session.id, user_id=user.id, role="assistant",
                question=f"第{index}个问题" + "风险" * 20,
                answer=f"第{index}个回答" + "分析" * 45,
                status="COMPLETED",
            )
            db.add(turn)
            turns.append(turn)
        db.flush()
        db.add(ConversationSummary(
            user_id=user.id, session_id=session.id,
            summary="此前讨论了公司风险和盈利预测。",
            active_entities=[], discussed_topics=["公司风险"],
            user_preferences=["简洁回答"], pending_questions=[],
            covered_until_message_id=turns[2].id,
        ))
        current = ChatMessage(
            session_id=session.id, user_id=user.id, role="assistant",
            question="之前的风险是什么", status="QUEUED",
        )
        db.add(current)
        db.flush()
        monkeypatch.setattr(
            memory_store,
            "search_turns",
            lambda *args, **kwargs: [{"message_id": turns[1].id}],
        )

        context = load_conversation_context(db, current)
        assert context["strategy"] == "SUMMARY_RECALL"
        assert context["summary"]["summary"]
        assert context["diagnostics"]["estimated_tokens"] <= 500
        assert context["recent_turns"][-1]["message_id"] == turns[-1].id


def test_password_hash_round_trip():
    encoded = hash_password("correct horse battery staple", iterations=1_000)
    assert verify_password("correct horse battery staple", encoded)
    assert not verify_password("wrong", encoded)


def test_demo_snapshot_covers_all_source_types():
    for source_type in ("RESEARCH_REPORT", "NEWS", "ANNOUNCEMENT", "SOCIAL"):
        items = demo_items("002594", "比亚迪", source_type)
        assert items
        assert all(item.document_type == source_type for item in items)
        assert all("课程演示" in item.title for item in items)


def test_rule_extraction_separates_forecast_and_rating():
    document = SimpleNamespace(
        title="课程演示研报",
        source_name="测试机构",
        parsed_text="评级：增持，目标价：42.00元。预计2027年营业收入1200亿元，归母净利润85亿元，EPS 2.10元。风险提示：需求不及预期、原材料价格波动。",
        raw_text=None,
    )
    result = rule_based_extract(document)
    assert result.original_rating == "增持"
    assert result.target_price == 42
    assert result.forecasts[0].year == 2027
    assert result.forecasts[0].net_profit == 85
    assert len(result.risks) == 2


def test_rating_mapping():
    assert normalize_rating("维持买入评级") == "POSITIVE"
    assert normalize_rating("增持") == "SLIGHTLY_POSITIVE"
    assert normalize_rating("中性") == "NEUTRAL"
    assert clean_rating("说明") is None
    assert normalize_rating("比率分析") is None


def test_rule_extraction_understands_common_research_headings():
    document = SimpleNamespace(
        title="公司点评",
        source_name="测试机构",
        parsed_text="投资评级：维持买入评级\n投资要点\n海外销量增长，成本控制改善，预计盈利能力逐步修复。\n风险提示：需求不及预期。",
        raw_text=None,
    )
    result = rule_based_extract(document)
    assert result.original_rating == "买入"
    assert result.opinions
    assert "海外销量增长" in result.opinions[0].content


def test_consensus_requires_two_distinct_reports_and_institutions():
    points = [
        {"document_id": 1, "institution": "机构A", "content": "海外销量持续增长，成本控制改善"},
        {"document_id": 2, "institution": "机构B", "content": "海外销量保持增长，成本控制持续改善"},
        {"document_id": 3, "institution": "机构C", "content": "国内竞争压力明显加大"},
    ]
    consensus = _cluster_statements(points, require_distinct_institutions=True)
    assert len(consensus) == 1
    assert "2家机构" in consensus[0]


def test_chunking_preserves_page_number_and_overlap():
    chunks = chunk_pages([(3, "测试文本" * 300)], size=100, overlap=20)
    assert len(chunks) > 2
    assert all(page == 3 for page, _ in chunks)
    assert all(len(text) <= 100 for _, text in chunks)


def test_cninfo_detail_url_resolves_to_canonical_pdf():
    detail = (
        "http://www.cninfo.com.cn/new/disclosure/detail?"
        "stockCode=688836&announcementId=1225544788&"
        "announcementTime=2026-09-03"
    )
    assert cninfo_pdf_url(detail) == (
        "https://static.cninfo.com.cn/finalpage/2026-09-03/1225544788.PDF"
    )


def test_template_shell_is_not_usable_document_text():
    template = """巨潮资讯网
{{month}}
{{stockName}}
{{isDownloading? '下载中' : '公告下载'}}"""
    quality = assess_text_quality(template)
    assert not quality.usable
    assert any("模板占位符" in warning for warning in quality.warnings)
    assert normalize_text(template) == ""


def test_soft_pdf_line_wraps_are_rejoined_without_flattening_lists():
    text = normalize_text("公司主营业务持续增\n长，盈利能力保持稳定。\n1、第一项\n2、第二项")
    assert "持续增长，盈利能力" in text
    assert "\n1、第一项\n2、第二项" in text


def test_pdf_parser_removes_repeated_margins_and_keeps_pages():
    document = fitz.open()
    for index in range(2):
        page = document.new_page()
        page.insert_text((72, 40), "REPEATED COMPANY HEADER")
        page.insert_text((72, 100), f"Substantive announcement paragraph {index + 1} " * 8)
        page.insert_text((280, 800), str(index + 1))
    text, pages = parse_pdf(document.tobytes())
    document.close()
    assert len(pages) == 2
    assert "Substantive announcement paragraph 1" in text
    assert "Substantive announcement paragraph 2" in text
    assert "REPEATED COMPANY HEADER" not in text


def test_fallback_summary_is_labeled_as_preview_and_skips_furniture():
    document = SimpleNamespace(
        document_type="ANNOUNCEMENT",
        title="利润分配公告",
        source_name="巨潮资讯",
        parsed_text=(
            "证券代码：000001\n"
            "本公司及董事会全体成员保证公告内容真实、准确、完整。\n"
            "公司拟以现有总股本为基数，每十股派发现金红利三元。"
        ),
        raw_text=None,
    )
    result = rule_based_extract(document)
    assert result.summary_method == "PREVIEW"
    assert "每十股派发现金红利三元" in result.summary
    assert "证券代码" not in result.summary


def test_milvus_filter_is_applied_before_hybrid_search():
    expression = _filter_expression(
        company_id=7,
        document_types=["ANNOUNCEMENT", "RESEARCH_REPORT"],
        date_from=datetime(2026, 1, 1),
        date_to=datetime(2026, 6, 30, 23, 59, 59),
    )
    assert "is_deleted == false" in expression
    assert "company_id == 7" in expression
    assert 'document_type in ["ANNOUNCEMENT", "RESEARCH_REPORT"]' in expression
    assert "published_ts >=" in expression
    assert "published_ts <=" in expression


def test_milvus_filter_can_restrict_an_industry_company_set():
    expression = _filter_expression(None, ["NEWS"], None, None, [3, 7, 9])
    assert "company_id in [3, 7, 9]" in expression


def test_chat_input_is_trimmed_and_bounded():
    assert CreateMessage(question="  宁德时代有什么风险？  ").question == "宁德时代有什么风险？"
    assert CreateChatSession(title="  我的   研究  ").title == "我的 研究"
    with pytest.raises(ValueError):
        CreateMessage(question="   ")
    with pytest.raises(ValueError):
        CreateMessage(question="问" * 2001)


def test_system_questions_bypass_company_resolution():
    assert classify_intent("你好，你是什么模型？") == "SYSTEM_META"
    assert classify_intent("你好，请问你是？") == "SYSTEM_META"
    assert classify_intent("你能做什么") == "SYSTEM_META"
    assert classify_intent("你好，请问你能够做什么？") == "SYSTEM_META"
    assert classify_intent("你可以帮我做什么？") == "SYSTEM_META"
    assert classify_intent("这个系统有哪些功能？") == "SYSTEM_META"
    assert classify_intent("这个系统可以问什么？") == "SYSTEM_META"
    assert classify_intent("这个系统能做什么？") == "SYSTEM_META"
    assert classify_intent("比亚迪最近有哪些风险") == "COMPANY_RESEARCH"
    assert classify_intent("比亚迪汽车有哪些功能？") == "COMPANY_RESEARCH"


def test_open_routes_do_not_default_to_company_research():
    assert classify_intent("ROE下降但净利润上升说明什么？") == "GENERAL_FINANCE"
    assert classify_intent("帮我生成一份公司研究简报") == "CONTENT_GENERATION"
    assert classify_intent("帮我看看这个") == "UNKNOWN"
    assert route_after_classification({"intent": "GENERAL_FINANCE"}) == "answer_general_finance"
    assert route_after_classification({"intent": "UNKNOWN"}) == "answer_open_fallback"
    assert route_after_entity({"errors": ["company_not_found"]}) == "await_clarification"


def test_low_confidence_question_uses_structured_llm_router(monkeypatch):
    response = LLMCallResult(
        '{"scope":"GENERAL_FINANCE","task_type":"EXPLAIN","requires_company":false,'
        '"requires_current_data":false,"requires_evidence":false,"is_follow_up":false,'
        '"company_mention":null,"stock_code":null,"confidence":0.88}',
        "test-model",
        1,
        12,
    )
    monkeypatch.setattr("app.ai.router.invoke_text_detailed", lambda _, **kwargs: response)
    decision, call = route_request("复合增长到底该怎么看？")
    assert decision.scope == "GENERAL_FINANCE"
    assert decision.source == "LLM"
    assert call is response


def test_router_failure_has_safe_unknown_and_follow_up_fallback(monkeypatch):
    failed = LLMCallResult(None, "test-model", 1, 15, "TimeoutError", None)
    monkeypatch.setattr("app.ai.router.invoke_text_detailed", lambda _, **kwargs: failed)
    decision, _ = route_request("帮我看看这个")
    assert decision.scope == "UNKNOWN"
    assert decision.source == "SAFE_FALLBACK"
    follow_up, _ = route_request("继续展开说说", active_company_name="贵州茅台")
    assert follow_up.scope == "COMPANY_RESEARCH"
    assert follow_up.is_follow_up


def test_explicit_unknown_company_does_not_inherit_previous_company(monkeypatch):
    response = LLMCallResult(
        '{"standalone_question":"宇树科技最近怎么样？","scope":"COMPANY_RESEARCH",'
        '"task_type":"RETRIEVE","requires_company":true,"requires_current_data":true,'
        '"requires_evidence":true,"is_follow_up":false,"company_mention":"宇树科技",'
        '"stock_code":null,"research_intent":"OVERVIEW","suggested_tools":["company"],'
        '"source_types":["NEWS"],"confidence":0.9}',
        "test-model", 1, 10,
    )
    monkeypatch.setattr("app.ai.router.invoke_text_detailed", lambda _, **kwargs: response)
    decision, _ = route_request("宇树科技最近怎么样？", active_company_name="贵州茅台")
    assert decision.scope == "COMPANY_RESEARCH"
    assert decision.company_mention == "宇树科技"
    assert decision.is_follow_up is False


def test_model_contextualizes_company_switch_and_inherits_previous_topic(monkeypatch):
    response = LLMCallResult(
        '{"standalone_question":"比亚迪最近舆情怎么样？","scope":"COMPANY_RESEARCH",'
        '"task_type":"舆情分析","requires_company":true,"requires_current_data":true,'
        '"requires_evidence":true,"is_follow_up":true,"company_mention":"比亚迪",'
        '"stock_code":null,"research_intent":"SENTIMENT",'
        '"suggested_tools":["company","news_sentiment","social_sentiment"],'
        '"source_types":["NEWS","SOCIAL"],"confidence":0.96}',
        "test-model", 1, 10,
    )
    captured = {}

    def fake_invoke(prompt, **kwargs):
        captured["prompt"] = prompt
        captured["fast"] = kwargs.get("fast")
        return response

    monkeypatch.setattr("app.ai.router.invoke_text_detailed", fake_invoke)
    decision, _ = route_request(
        "比亚迪呢？",
        active_company_name="贵州茅台",
        explicit_company_name="比亚迪",
        conversation_context='最近一轮：用户问“茅台最近舆情怎么样？”',
    )
    assert decision.standalone_question == "比亚迪最近舆情怎么样？"
    assert decision.task_type == "FOLLOW_UP"
    assert decision.research_intent == "SENTIMENT"
    assert decision.suggested_tools == ["company", "news_sentiment", "social_sentiment"]
    assert "茅台最近舆情怎么样" in captured["prompt"]
    assert captured["fast"] is True


def test_risk_fallback_groups_semantically_repeated_evidence():
    answer = _risk_evidence_fallback({
        "company_name": "贵州茅台",
        "citations": [
            {"quote": "经济复苏不及预期风险"},
            {"quote": "终端需求不及预期"},
            {"quote": "市场竞争加剧风险"},
        ],
    })
    assert "需求与宏观风险" in answer
    assert "市场竞争与价格风险" in answer
    assert "可归并为2类" in answer


def test_common_company_short_name_matching_is_conservative():
    assert _matches_common_company_short_name(SimpleNamespace(name="贵州茅台"), "茅台最近行情怎么样")
    assert _matches_common_company_short_name(SimpleNamespace(name="中国平安"), "平安最近有什么公告")
    assert not _matches_common_company_short_name(SimpleNamespace(name="比亚迪"), "汽车行业怎么样")


def test_question_intent_routes_are_explicit():
    assert classify_intent("新能源汽车行业最近有什么变化") == "INDUSTRY_RESEARCH"
    assert classify_intent("腾讯控股最近有什么新闻") == "UNSUPPORTED_MARKET"
    assert classify_intent("今天天气怎么样") == "OUT_OF_SCOPE"


def test_stale_existing_knowledge_answers_first_and_refreshes_in_background():
    assert route_after_check({"needs_collection": False, "needs_refresh": True}) == "start_background_refresh"
    assert route_after_check({"needs_collection": True, "needs_refresh": True}) == "start_collection"
    assert route_after_check({"needs_collection": False, "needs_refresh": False}) == "retrieve_context"


def test_structured_tools_can_survive_vector_retrieval_outage():
    assert route_after_retrieval({"retrieved_documents": [], "research_intent": "MARKET_TREND"}) == "select_tools"
    assert route_after_retrieval({"retrieved_documents": [], "research_intent": "FINANCIAL"}) == "select_tools"
    assert route_after_retrieval({"retrieved_documents": [], "research_intent": "OVERVIEW"}) == "handle_failure"


def test_financial_citations_keep_document_provenance_and_raw_value():
    citations = _build_financial_citations([{
        "document_id": 9,
        "title": "年度报告",
        "source_type": "ANNOUNCEMENT",
        "source_url": "https://example.com/report.pdf",
        "published_at": "2026-04-01T00:00:00",
        "source_name": "巨潮资讯",
        "name": "营业收入",
        "raw_value": "100",
        "unit": "亿元",
        "period": "2025年",
        "evidence": "2025年营业收入100亿元",
        "page": 3,
    }])
    assert citations[0]["document_id"] == 9
    assert "营业收入：100亿元" in citations[0]["quote"]
    assert citations[0]["page"] == 3


def test_background_refresh_only_requests_relevant_sources():
    assert classify_refresh_sources("比亚迪最近有哪些新闻，舆情怎么样？") == ["NEWS", "SOCIAL"]
    assert classify_refresh_sources("宁德时代目前的研报评级和盈利预测") == ["RESEARCH_REPORT"]
    assert classify_refresh_sources("赣锋锂业当前财务数据和公告") == ["ANNOUNCEMENT"]
    assert classify_refresh_sources("全面分析比亚迪") == ["RESEARCH_REPORT", "NEWS", "ANNOUNCEMENT", "SOCIAL"]
    assert classify_refresh_sources("比亚迪有哪些主要风险") == ["RESEARCH_REPORT", "ANNOUNCEMENT", "NEWS"]


def test_vague_follow_up_uses_recent_question_for_retrieval():
    query = _retrieval_question({
        "question": "那风险呢？",
        "company_name": "宁德时代",
        "conversation_questions": ["宁德时代最近的盈利预测有什么分歧？"],
    })
    assert "宁德时代" in query
    assert "盈利预测" in query
    assert "风险" in query


def test_explicit_short_company_question_does_not_inherit_unrelated_history():
    query = _retrieval_question({
        "question": "宁德时代最近行情怎么样？",
        "company_name": "宁德时代",
        "conversation_questions": ["什么是股票？"],
        "route_is_follow_up": False,
    })
    assert "什么是股票" not in query
    assert query == "宁德时代 宁德时代最近行情怎么样？"


def test_company_news_filter_rejects_incidental_single_mentions():
    incidental = {"source_type": "NEWS", "title": "机器人企业上市", "quote": "客户包括一汽、宁德时代、比亚迪和多家制造企业。"}
    focused = {"source_type": "NEWS", "title": "比亚迪发布投资者活动记录", "quote": "公司披露最新经营情况。"}
    repeated = {"source_type": "NEWS", "title": "新能源汽车动态", "quote": "比亚迪公布销量，比亚迪海外业务继续增长。"}
    assert not _company_hit_relevant(incidental, "比亚迪", "002594")
    assert _company_hit_relevant(focused, "比亚迪", "002594")
    assert _company_hit_relevant(repeated, "比亚迪", "002594")


def test_prompt_context_is_bounded_and_citation_date_is_evidence_based():
    compact = _compact_prompt_value({"rows": ["x" * 2000 for _ in range(30)]})
    assert len(compact["rows"]) == 12
    assert all(len(item) == 800 for item in compact["rows"])
    actual = _citation_data_as_of(
        [{"published_at": "2026-03-01T00:00:00"}, {"published_at": "2026-06-15T00:00:00"}],
        datetime(2026, 9, 1),
    )
    assert actual == datetime(2026, 6, 15)


def test_risk_prompt_omits_unrelated_heavy_tool_payloads():
    payload = _tool_context_for_prompt({
        "research_intent": "RISK",
        "tool_results": {
            "company": {"name": "宁德时代", "stock_code": "300750"},
            "financial_metrics": [{"name": "营业收入", "raw_value": "100"}] * 20,
            "market_prices": {"prices": [{"close": 100}] * 30},
            "risks": ["需求不及预期", "市场竞争加剧"],
            "social_sentiment": {"distribution": {"NEGATIVE": 1000}},
        },
    })
    assert payload["company"]["name"] == "宁德时代"
    assert payload["risks"] == ["需求不及预期", "市场竞争加剧"]
    assert "financial_metrics" not in payload
    assert "market_prices" not in payload
    assert "social_sentiment" not in payload


def test_research_intent_drives_source_specific_retrieval():
    assert classify_research_intent("比亚迪最近的股价走势和成交量如何") == "MARKET_TREND"
    assert classify_research_intent("宁德时代营收、利润和现金流怎么样") == "FINANCIAL"
    assert classify_research_intent("赣锋锂业各家券商评级有什么分歧") == "BROKER_RESEARCH"
    assert classify_research_intent("比亚迪最近有哪些新闻，舆情怎么样") == "SENTIMENT"
    assert classify_research_intent("比亚迪有哪些主要风险") == "RISK"
    assert intent_document_types("BROKER_RESEARCH") == ["RESEARCH_REPORT"]
    assert intent_document_types("SENTIMENT") == ["NEWS", "SOCIAL"]


def test_grounding_validation_accepts_supported_numbers_and_citations():
    citations = [{"title": "年度报告", "quote": "2025年营业收入100亿元，同比增长10%。"}]
    result = validate_grounded_answer("2025年营业收入100亿元，同比增长10%[1]。", citations)
    assert result["valid"]
    assert result["numeric_claim_count"] == 1
    assert result["supported_numeric_claim_count"] == 1


def test_grounding_validation_preserves_negative_number_signs():
    citations = [{"title": "阶段行情", "quote": "较5个交易日前变化-6.31%，当日涨跌幅-3.65%。"}]
    result = validate_grounded_answer("近5个交易日变化-6.31%，最新日跌幅-3.65%[1]。", citations)
    assert result["hard_valid"], result["hard_issues"]
    assert result["supported_numeric_claim_count"] == 1


def test_grounding_validation_rejects_uncited_or_invented_numbers():
    citations = [{"title": "年度报告", "quote": "2025年营业收入100亿元，同比增长10%。"}]
    uncited = validate_grounded_answer("2025年营业收入100亿元，同比增长10%。", citations)
    invented = validate_grounded_answer("2025年营业收入120亿元，同比增长10%[1]。", citations)
    invalid_reference = validate_grounded_answer("营业收入100亿元[2]。", citations)
    assert any(issue["code"] == "UNCITED_NUMBER" for issue in uncited["issues"])
    assert any(issue["code"] == "UNSUPPORTED_NUMBER" for issue in invented["issues"])
    assert any(issue["code"] == "INVALID_CITATION" for issue in invalid_reference["issues"])
    assert uncited["hard_valid"]
    assert invented["hard_valid"]
    assert not invalid_reference["hard_valid"]


def test_grounding_rejects_direct_trading_and_position_advice():
    citations = [{"title": "研报", "quote": "机构维持买入评级。"}]
    answer = "建议立即买入并把仓位提高到80%[1]。"
    result = validate_grounded_answer(answer, citations)
    assert contains_direct_trading_advice(answer)
    assert any(issue["code"] == "DIRECT_TRADING_ADVICE" for issue in result["issues"])
    filtered, filtered_result = filter_unsupported_lines(answer, citations)
    assert "立即买入" not in filtered
    assert not filtered_result["valid"]  # caller must fall back to cited evidence
    assert not contains_direct_trading_advice("国信证券维持买入评级[1]。")


def test_grounding_validation_requires_citations_for_material_factual_claims():
    citations = [{"title": "公司公告", "quote": "公司公告披露海外销量增长。"}]
    result = validate_grounded_answer("公司海外销量保持增长。", citations)
    assert any(issue["code"] == "UNCITED_FACT" for issue in result["issues"])
    assert result["hard_valid"]
    assert {issue["code"] for issue in result["soft_issues"]} == {"MISSING_CITATIONS", "UNCITED_FACT"}


def test_hard_only_filter_keeps_analysis_but_removes_trading_instruction():
    citations = [{"title": "行情", "quote": "最新收盘价100元。"}]
    answer = "从现有走势看，短期波动可能加大，估算波动区间约为10%。\n\n建议立即买入并满仓。\n\n> 不构成投资建议。"
    filtered, result = filter_unsupported_lines(answer, citations, hard_only=True)
    assert "短期波动可能加大" in filtered
    assert "10%" in filtered
    assert "立即买入" not in filtered
    assert result["hard_valid"]


def test_market_citations_cover_latest_and_period_statistics():
    prices = [
        {
            "trade_date": f"2026-08-{31 - index:02d}",
            "open": 100.0,
            "close": 100.0,
            "high": 105.0,
            "low": 95.0,
            "volume": 1000.0,
            "amount": 100000.0,
            "change_pct": 0.0,
        }
        for index in range(21)
    ]
    prices[5]["close"] = 110.0
    prices[20]["close"] = 80.0
    prices[10]["high"] = 130.0
    prices[15]["low"] = 70.0
    citations = _build_market_citations("测试公司", {"source": "测试行情源", "prices": prices})
    assert len(citations) == 2
    assert "最新收盘价100元" in citations[1]["quote"]
    assert "较5个交易日前" in citations[1]["quote"]
    assert "变化-9.09%" in citations[1]["quote"]
    assert "变化25.00%" in citations[1]["quote"]
    assert "最高价130元" in citations[1]["quote"]
    assert "最低价70元" in citations[1]["quote"]


def test_one_citation_can_support_multiple_sentences_in_the_same_bullet():
    citations = [{"title": "风险提示", "quote": "需求可能下降。原材料价格存在上涨风险。"}]
    result = validate_grounded_answer("- [1] 需求可能下降。原材料价格存在上涨风险。", citations)
    assert result["valid"]


def test_filtering_keeps_grounded_paragraphs_and_removes_unsupported_ones():
    citations = [{"title": "年度报告", "quote": "2025年营业收入100亿元，同比增长10%。"}]
    answer = "### 结论\n\n营业收入达到100亿元[1]。\n\n净利润达到999亿元。\n\n> 不构成投资建议。"
    filtered, result = filter_unsupported_lines(answer, citations)
    assert "营业收入达到100亿元[1]" in filtered
    assert "999亿元" not in filtered
    assert result["valid"]
    assert result["removed_line_count"] == 1


def test_filtering_removes_dangling_heading_and_keeps_disclaimer():
    citations = [{"title": "风险提示", "quote": "汽车销量不及预期。"}]
    answer = "简明结论：\n汽车销量不及预期[1]。\n\n风险与不确定性：\n无引用的新风险判断。"
    filtered, result = filter_unsupported_lines(answer, citations)
    assert not filtered.rstrip().endswith("风险与不确定性：")
    assert "不构成投资建议" in filtered
    assert result["valid"]


def test_professional_risk_gate_rejects_social_noise():
    assert is_professional_risk_item("汽车销量不及预期", "RESEARCH_REPORT")
    assert is_professional_risk_item("原材料价格波动风险", "ANNOUNCEMENT")
    assert not is_professional_risk_item("大把普通散户埋在里面", "SOCIAL")
    assert not is_professional_risk_item("妥妥的国民持仓股", "RESEARCH_REPORT")


def test_event_similarity_groups_repeated_reports_but_not_unrelated_topics():
    assert _event_similarity("比亚迪发布新车型，计划下月上市", "比亚迪新车型正式发布，下月上市") >= 0.48
    assert _event_similarity("比亚迪发布新车型", "宁德时代披露年度财务报告") < 0.48


def test_high_volume_sources_use_fast_rule_extraction(monkeypatch):
    document = SimpleNamespace(
        document_type="SOCIAL",
        parsed_text="市场讨论偏积极，同时提示需求波动风险。",
        raw_text=None,
        title="股吧讨论",
        source_name="测试来源",
    )
    monkeypatch.setattr("app.ai.extraction.invoke_json", lambda _: (_ for _ in ()).throw(AssertionError("不应调用LLM")))
    result = extract_document(document)
    assert result.summary


def test_llm_wrapper_does_not_swallow_worker_soft_timeout(monkeypatch):
    from app.ai import llm

    model = SimpleNamespace(invoke=lambda _: (_ for _ in ()).throw(SoftTimeLimitExceeded()))
    monkeypatch.setattr(llm, "get_chat_models", lambda: (model,))
    with pytest.raises(SoftTimeLimitExceeded):
        llm.invoke_text("test")


def test_llm_wrapper_retries_one_transient_failure_and_reports_diagnostics(monkeypatch):
    from app.ai import llm

    calls = {"count": 0}

    def invoke(_):
        calls["count"] += 1
        if calls["count"] == 1:
            raise TimeoutError("temporary")
        return SimpleNamespace(content="已恢复")

    model = SimpleNamespace(model_name="test-model", invoke=invoke)
    monkeypatch.setattr(llm, "get_chat_models", lambda: (model,))
    monkeypatch.setattr(llm.time, "sleep", lambda _: None)
    result = llm.invoke_text_detailed("test", retry_transient=True)
    assert result.text == "已恢复"
    assert result.attempts == 2
    assert result.metadata()["succeeded"] is True


def test_llm_stream_reports_incremental_accumulated_text(monkeypatch):
    from app.ai import llm

    model = SimpleNamespace(
        model_name="test-model",
        stream=lambda _: iter(
            [
                SimpleNamespace(content="你"),
                SimpleNamespace(content="好"),
                SimpleNamespace(content="，这是流式回答。"),
            ]
        ),
    )
    updates = []
    monkeypatch.setattr(llm, "get_chat_models", lambda: (model,))
    result = llm.invoke_text_stream_detailed("test", on_text=updates.append)

    assert updates == ["你", "你好", "你好，这是流式回答。"]
    assert result.text == "你好，这是流式回答。"
    assert result.streamed is True
    assert result.chunk_count == 3
    assert result.metadata()["succeeded"] is True


def test_llm_stream_retries_openai_timeout_before_first_chunk(monkeypatch):
    from app.ai import llm

    class OpenAITimeoutError(Exception):
        pass

    calls = {"count": 0}

    def stream(_):
        calls["count"] += 1
        if calls["count"] == 1:
            raise OpenAITimeoutError("temporary")
        return iter([SimpleNamespace(content="重试成功")])

    model = SimpleNamespace(model_name="test-model", stream=stream)
    monkeypatch.setattr(llm, "get_chat_models", lambda: (model,))
    monkeypatch.setattr(llm.time, "sleep", lambda _: None)
    result = llm.invoke_text_stream_detailed("test", retry_transient=True)

    assert result.text == "重试成功"
    assert result.attempts == 2


def test_async_stream_has_explicit_first_content_deadline(monkeypatch):
    from app.ai import llm

    class SlowAsyncModel:
        model_name = "slow-model"

        async def astream(self, _):
            await asyncio.sleep(0.05)
            yield SimpleNamespace(content="迟到的正文", additional_kwargs={})

    monkeypatch.setattr(llm, "get_chat_models", lambda: (SlowAsyncModel(),))
    monkeypatch.setattr(llm, "get_settings", lambda: SimpleNamespace(
        llm_stream_first_content_timeout_seconds=0.01,
        llm_stream_idle_timeout_seconds=0.01,
        llm_stream_overall_timeout_seconds=0.03,
    ))
    result = llm.invoke_text_stream_detailed("test")
    assert result.text is None
    assert result.error_type == "FirstContentTimeoutError"


def test_async_stream_reports_hidden_reasoning_without_exposing_it(monkeypatch):
    from app.ai import llm

    class ReasoningAsyncModel:
        model_name = "reasoning-model"

        async def astream(self, _):
            yield SimpleNamespace(content="", additional_kwargs={"reasoning_content": "内部推理"})
            yield SimpleNamespace(content="最终回答", additional_kwargs={})

    statuses = []
    updates = []
    monkeypatch.setattr(llm, "get_chat_models", lambda: (ReasoningAsyncModel(),))
    monkeypatch.setattr(llm, "get_settings", lambda: SimpleNamespace(
        llm_stream_first_content_timeout_seconds=1,
        llm_stream_idle_timeout_seconds=1,
        llm_stream_overall_timeout_seconds=2,
    ))
    result = llm.invoke_text_stream_detailed(
        "test", on_text=updates.append,
        on_status=lambda phase, elapsed: statuses.append((phase, elapsed)),
    )
    assert result.text == "最终回答"
    assert updates == ["最终回答"]
    assert statuses[0][0] == "reasoning"
    assert result.reasoning_chunk_count == 1
    assert result.chunk_count == 1


def test_async_stream_reasoning_suspends_first_content_deadline(monkeypatch):
    from app.ai import llm

    class LongReasoningAsyncModel:
        model_name = "reasoning-model"

        async def astream(self, _):
            yield SimpleNamespace(
                content=[],
                content_blocks=[{"type": "reasoning", "text": "不应展示的内部推理"}],
                additional_kwargs={},
            )
            await asyncio.sleep(0.04)
            yield SimpleNamespace(content="最终答案", content_blocks=[], additional_kwargs={})

    statuses = []
    updates = []
    monkeypatch.setattr(llm, "get_chat_models", lambda: (LongReasoningAsyncModel(),))
    monkeypatch.setattr(llm, "get_settings", lambda: SimpleNamespace(
        llm_stream_first_content_timeout_seconds=0.01,
        llm_stream_idle_timeout_seconds=0.01,
        llm_stream_overall_timeout_seconds=0.2,
    ))
    result = llm.invoke_text_stream_detailed(
        "test",
        on_text=updates.append,
        on_status=lambda phase, elapsed: statuses.append((phase, elapsed)),
    )

    assert result.text == "最终答案"
    assert updates == ["最终答案"]
    assert any(phase == "reasoning" for phase, _ in statuses)
    assert result.reasoning_chunk_count == 1
    assert "不应展示的内部推理" not in result.text


def test_async_stream_empty_heartbeats_do_not_count_as_reasoning(monkeypatch):
    from app.ai import llm

    class HeartbeatOnlyAsyncModel:
        model_name = "heartbeat-model"

        async def astream(self, _):
            until = time.perf_counter() + 0.04
            while time.perf_counter() < until:
                yield SimpleNamespace(content="", additional_kwargs={})
                await asyncio.sleep(0)
            yield SimpleNamespace(content="不应等到的答案", additional_kwargs={})

    monkeypatch.setattr(llm, "get_chat_models", lambda: (HeartbeatOnlyAsyncModel(),))
    monkeypatch.setattr(llm, "get_settings", lambda: SimpleNamespace(
        llm_stream_first_content_timeout_seconds=0.01,
        llm_stream_idle_timeout_seconds=0.01,
        llm_stream_overall_timeout_seconds=0.2,
    ))
    result = llm.invoke_text_stream_detailed("test")

    assert result.text is None
    assert result.error_type == "FirstContentTimeoutError"
    assert result.reasoning_chunk_count == 0


def test_social_conversation_uses_streaming_model_instead_of_fixed_answer(monkeypatch):
    from app.ai import graph

    captured = {"messages": [], "metadata": []}

    def fake_stream(prompt, *, on_text=None, on_status=None, retry_transient=False):
        captured["prompt"] = prompt
        assert retry_transient is False
        if on_text:
            on_text("这是模型生成的自然问候。")
        return LLMCallResult("这是模型生成的自然问候。", "test-model", 1, 10, streamed=True, chunk_count=1)

    def fake_update_message(message_id, **values):
        captured["messages"].append((message_id, values))
        return True

    monkeypatch.setattr(graph, "invoke_text_stream_detailed", fake_stream)
    monkeypatch.setattr(graph, "_update_message", fake_update_message)
    monkeypatch.setattr(graph, "_update_metadata", lambda message_id, **values: captured["metadata"].append((message_id, values)))
    monkeypatch.setattr(graph, "_assert_execution", lambda _: None)

    result = graph.answer_social_conversation({"message_id": 1, "question": "你好", "conversation_questions": []})
    assert result["answer"] == "这是模型生成的自然问候。"
    assert "不要使用固定话术" in captured["prompt"]
    assert captured["messages"][-1][1]["status"] == "COMPLETED"
    assert any(values.get("route") == "SOCIAL_CONVERSATION" for _, values in captured["metadata"])


def test_report_metric_names_are_presented_as_financial_labels():
    assert _metric_label("revenue") == "营业收入"
    assert _metric_label("gross_margin") == "毛利率"
    assert _metric_label("自定义指标") == "自定义指标"


def test_report_consensus_requires_multiple_documents():
    points = [
        {"document_id": 1, "content": "海外销量持续增长，盈利能力改善"},
        {"document_id": 2, "content": "海外销量保持增长，盈利能力持续改善"},
        {"document_id": 3, "content": "原材料价格存在波动"},
    ]
    clusters = _clusters(points)
    assert len(clusters) == 1
    assert {item["document_id"] for item in clusters[0]} == {1, 2}


def test_report_llm_summary_removes_other_company_bullets(monkeypatch):
    monkeypatch.setattr(
        "app.ai.reporting.invoke_text",
        lambda _: "- 安克创新目标价为154.17元[1]\n- 比亚迪目标价为125.28元[1]",
    )
    result = _safe_llm_summary("比亚迪", "核心研究摘要", [(1, "安克创新目标价154.17元；比亚迪目标价125.28元")])
    assert "安克创新目标价" not in result
    assert "比亚迪目标价" in result


def test_report_exports_generate_valid_office_and_pdf_files():
    markdown = """# 比亚迪公司研究简报

## 财务数据

| 指标 | 数值 | 来源 |
|---|---:|---|
| 营业收入 | 100亿元 | [1] |

- 示例结论 [1]
"""
    docx = export_docx("测试报告", markdown)
    pdf = export_pdf("测试报告", markdown)
    assert zipfile.is_zipfile(BytesIO(docx))
    assert pdf.startswith(b"%PDF-")


def test_crawl_run_source_types_support_four_sources_without_string_joining():
    requested = normalize_source_types(["RESEARCH_REPORT", "NEWS", "ANNOUNCEMENT", "SOCIAL"])
    run = SimpleNamespace(source_type=None, source_types=requested)
    assert requested == ["RESEARCH_REPORT", "NEWS", "ANNOUNCEMENT", "SOCIAL"]
    assert run_source_types(run) == requested
    assert run_covers(run, requested)


def test_crawl_run_source_types_remain_compatible_with_legacy_rows():
    legacy = SimpleNamespace(source_type="NEWS,SOCIAL", source_types=[])
    assert run_source_types(legacy) == ["NEWS", "SOCIAL"]
    assert run_covers(legacy, ["NEWS"])
    assert not run_covers(legacy, ["ANNOUNCEMENT"])


def test_broker_forecast_table_extraction_normalizes_common_pdf_layout():
    document = SimpleNamespace(
        document_type="RESEARCH_REPORT",
        title="半年报点评",
        source_name="测试证券",
        raw_text=None,
        parsed_text="""盈利预测和财务指标
2024
2025
2026E
2027E
营业收入(百万元)
362013
423702
617649
763004
营业收入增长率
归母净利润(百万元)
50745
72201
96351
120089
每股收益（元）
11.52
15.82
20.83
25.96
""",
    )
    result = rule_based_extract(document)
    assert [(row.year, row.revenue, row.net_profit, row.eps) for row in result.forecasts] == [
        (2026, 617649.0, 96351.0, 20.83),
        (2027, 763004.0, 120089.0, 25.96),
    ]


def test_news_and_social_posts_cannot_create_broker_ratings_or_forecasts():
    document = SimpleNamespace(
        document_type="SOCIAL",
        title="股吧讨论",
        source_name="公开讨论",
        raw_text=None,
        parsed_text="给予买入评级，预计2027年营业收入1200亿元，净利润85亿元。",
    )
    result = rule_based_extract(document)
    assert result.original_rating is None
    assert result.forecasts == []


def test_structured_broker_comparison_is_cited_and_reports_real_ranges():
    broker_data = {
        "ratings": [
            {"institution": "甲证券", "rating": "买入", "normalized": "POSITIVE", "target_price": None, "document_id": 1, "title": "甲研报", "source_url": "https://example.com/a.pdf", "published_at": "2026-07-31T00:00:00", "source_name": "甲证券", "evidence": "评级买入"},
            {"institution": "乙证券", "rating": "优于大市", "normalized": "SLIGHTLY_POSITIVE", "target_price": None, "document_id": 2, "title": "乙研报", "source_url": "https://example.com/b.pdf", "published_at": "2026-07-30T00:00:00", "source_name": "乙证券", "evidence": "评级优于大市"},
        ],
        "forecasts": [
            {"institution": "甲证券", "year": 2026, "revenue": 600000.0, "net_profit": 90000.0, "eps": 20.0, "unit": "百万元；EPS为元", "document_id": 1, "title": "甲研报", "source_url": "https://example.com/a.pdf", "published_at": "2026-07-31T00:00:00", "source_name": "甲证券", "evidence": "2026E预测"},
            {"institution": "乙证券", "year": 2026, "revenue": 650000.0, "net_profit": 95000.0, "eps": 22.0, "unit": "百万元；EPS为元", "document_id": 2, "title": "乙研报", "source_url": "https://example.com/b.pdf", "published_at": "2026-07-30T00:00:00", "source_name": "乙证券", "evidence": "2026E预测"},
        ],
    }
    citations = build_broker_citations(broker_data)
    answer = build_broker_comparison_answer("测试公司", broker_data, citations)
    validation = validate_grounded_answer(answer, citations)
    assert "6000.00亿元" in answer
    assert "6500.00亿元" in answer
    assert "| 2026E |" in answer
    assert "评级方向总体一致" in answer
    assert validation["valid"], validation["issues"]


def test_report_comparison_supports_ten_selected_and_fifty_consensus_reports():
    selected = ReportComparisonRequest(
        stock_code="300750",
        comparison_mode="SELECTED",
        document_ids=list(range(1, 11)),
        limit=10,
    )
    consensus = ReportComparisonRequest(
        stock_code="300750",
        comparison_mode="CONSENSUS",
        limit=50,
        institutions=["国信证券", "国信证券", " 山西证券 "],
    )
    assert len(selected.document_ids or []) == 10
    assert consensus.limit == 50
    assert consensus.institutions == ["国信证券", "山西证券"]
    with pytest.raises(ValueError):
        ReportComparisonRequest(stock_code="300750", comparison_mode="SELECTED", document_ids=list(range(1, 12)), limit=11)
    with pytest.raises(ValueError):
        ReportComparisonRequest(stock_code="300750", comparison_mode="CONSENSUS", document_ids=[1, 2])


def test_forecast_ranges_deduplicate_older_reports_from_same_institution():
    rows = [
        {"institution": "甲证券", "forecasts": [{"year": 2027, "revenue": 600000, "net_profit": 90000, "eps": 20, "unit": "百万元；EPS为元"}]},
        {"institution": "甲证券", "forecasts": [{"year": 2027, "revenue": 580000, "net_profit": 88000, "eps": 19, "unit": "百万元；EPS为元"}]},
        {"institution": "乙证券", "forecasts": [{"year": 2027, "revenue": 650000, "net_profit": 95000, "eps": 22, "unit": "百万元；EPS为元"}]},
    ]
    result = _forecast_ranges(rows)
    assert result[0]["institution_count"] == 2
    assert result[0]["revenue"]["min"] == 6000
    assert result[0]["revenue"]["max"] == 6500
