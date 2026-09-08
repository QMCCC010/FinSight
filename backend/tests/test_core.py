from datetime import datetime
from io import BytesIO
from types import SimpleNamespace
import zipfile

import pytest
from billiard.exceptions import SoftTimeLimitExceeded

from app.ai.extraction import clean_rating, extract_document, normalize_rating, rule_based_extract
from app.ai.broker_comparison import build_broker_citations, build_broker_comparison_answer
from app.api.routers.analysis import _cluster_statements
from app.api.routers.companies import _event_similarity
from app.ai.graph import classify_intent, classify_refresh_sources, route_after_check
from app.ai.grounding import classify_research_intent, filter_unsupported_lines, intent_document_types, is_professional_risk_item, validate_grounded_answer
from app.ai.milvus_store import _filter_expression
from app.ai.reporting import _clusters, _metric_label, _safe_llm_summary
from app.collector.demo_data import demo_items
from app.core.security import hash_password, verify_password
from app.services.parser import chunk_pages
from app.services.report_export import export_docx, export_pdf
from app.services.crawl_runs import normalize_source_types, run_covers, run_source_types


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


def test_system_questions_bypass_company_resolution():
    assert classify_intent("你好，你是什么模型？") == "SYSTEM_META"
    assert classify_intent("你能做什么") == "SYSTEM_META"
    assert classify_intent("你好，请问你能够做什么？") == "SYSTEM_META"
    assert classify_intent("你可以帮我做什么？") == "SYSTEM_META"
    assert classify_intent("这个系统有哪些功能？") == "SYSTEM_META"
    assert classify_intent("这个系统可以问什么？") == "SYSTEM_META"
    assert classify_intent("这个系统能做什么？") == "SYSTEM_META"
    assert classify_intent("比亚迪最近有哪些风险") == "COMPANY_RESEARCH"
    assert classify_intent("比亚迪汽车有哪些功能？") == "COMPANY_RESEARCH"


def test_question_intent_routes_are_explicit():
    assert classify_intent("新能源汽车行业最近有什么变化") == "INDUSTRY_RESEARCH"
    assert classify_intent("腾讯控股最近有什么新闻") == "UNSUPPORTED_MARKET"
    assert classify_intent("今天天气怎么样") == "OUT_OF_SCOPE"


def test_stale_existing_knowledge_answers_first_and_refreshes_in_background():
    assert route_after_check({"needs_collection": False, "needs_refresh": True}) == "start_background_refresh"
    assert route_after_check({"needs_collection": True, "needs_refresh": True}) == "start_collection"
    assert route_after_check({"needs_collection": False, "needs_refresh": False}) == "retrieve_context"


def test_background_refresh_only_requests_relevant_sources():
    assert classify_refresh_sources("比亚迪最近有哪些新闻，舆情怎么样？") == ["NEWS", "SOCIAL"]
    assert classify_refresh_sources("宁德时代目前的研报评级和盈利预测") == ["RESEARCH_REPORT"]
    assert classify_refresh_sources("赣锋锂业当前财务数据和公告") == ["ANNOUNCEMENT"]
    assert classify_refresh_sources("全面分析比亚迪") == ["RESEARCH_REPORT", "NEWS", "ANNOUNCEMENT", "SOCIAL"]


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


def test_grounding_validation_rejects_uncited_or_invented_numbers():
    citations = [{"title": "年度报告", "quote": "2025年营业收入100亿元，同比增长10%。"}]
    uncited = validate_grounded_answer("2025年营业收入100亿元，同比增长10%。", citations)
    invented = validate_grounded_answer("2025年营业收入120亿元，同比增长10%[1]。", citations)
    invalid_reference = validate_grounded_answer("营业收入100亿元[2]。", citations)
    assert any(issue["code"] == "UNCITED_NUMBER" for issue in uncited["issues"])
    assert any(issue["code"] == "UNSUPPORTED_NUMBER" for issue in invented["issues"])
    assert any(issue["code"] == "INVALID_CITATION" for issue in invalid_reference["issues"])


def test_grounding_validation_requires_citations_for_material_factual_claims():
    citations = [{"title": "公司公告", "quote": "公司公告披露海外销量增长。"}]
    result = validate_grounded_answer("公司海外销量保持增长。", citations)
    assert any(issue["code"] == "UNCITED_FACT" for issue in result["issues"])


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
