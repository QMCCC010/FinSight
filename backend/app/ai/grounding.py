from __future__ import annotations

import re
from typing import Any


# These are product intents, not unrestricted chain-of-thought. They are safe to
# expose in the UI and make retrieval/tool selection deterministic and testable.
RESEARCH_INTENTS: dict[str, dict[str, Any]] = {
    "MARKET_TREND": {
        "label": "行情与走势",
        "document_types": ["NEWS", "ANNOUNCEMENT"],
        "markers": ("行情", "股价", "涨跌", "走势", "成交量", "成交额", "技术面", "市值"),
    },
    "FINANCIAL": {
        "label": "财务与业绩",
        "document_types": ["ANNOUNCEMENT", "RESEARCH_REPORT"],
        "markers": ("财务", "财报", "营收", "收入", "利润", "净利润", "毛利率", "现金流", "eps", "业绩"),
    },
    "BROKER_RESEARCH": {
        "label": "研报与机构观点",
        "document_types": ["RESEARCH_REPORT"],
        "markers": ("研报", "机构", "券商", "评级", "目标价", "盈利预测", "预测", "观点", "共识", "分歧"),
    },
    "NEWS_EVENTS": {
        "label": "新闻与公司事件",
        "document_types": ["NEWS", "ANNOUNCEMENT"],
        "markers": ("新闻", "资讯", "事件", "公告", "披露", "发生了什么", "近况", "情况"),
    },
    "SENTIMENT": {
        "label": "新闻与舆情",
        "document_types": ["NEWS", "SOCIAL"],
        "markers": ("舆情", "情绪", "情感", "股吧", "讨论", "热度", "看多", "看空"),
    },
    "RISK": {
        "label": "风险分析",
        "document_types": ["RESEARCH_REPORT", "ANNOUNCEMENT", "NEWS"],
        "markers": ("风险", "不确定", "隐患", "挑战", "压力", "利空"),
    },
    "OVERVIEW": {
        "label": "综合研究",
        "document_types": ["RESEARCH_REPORT", "NEWS", "ANNOUNCEMENT", "SOCIAL"],
        "markers": (),
    },
}


def classify_research_intent(question: str) -> str:
    compact = re.sub(r"\s+", "", question).lower()
    scores: dict[str, int] = {}
    for intent, config in RESEARCH_INTENTS.items():
        if intent == "OVERVIEW":
            continue
        scores[intent] = sum(1 for marker in config["markers"] if marker in compact)
    highest = max(scores.values(), default=0)
    if highest == 0:
        return "OVERVIEW"
    # Stable declaration order resolves mixed questions. SENTIMENT is promoted
    # when explicitly requested because it otherwise overlaps NEWS_EVENTS.
    if scores.get("SENTIMENT", 0):
        return "SENTIMENT"
    return next(intent for intent in RESEARCH_INTENTS if scores.get(intent) == highest)


def intent_label(intent: str | None) -> str:
    return RESEARCH_INTENTS.get(intent or "OVERVIEW", RESEARCH_INTENTS["OVERVIEW"])["label"]


def intent_document_types(intent: str | None) -> list[str]:
    return list(RESEARCH_INTENTS.get(intent or "OVERVIEW", RESEARCH_INTENTS["OVERVIEW"])["document_types"])


def is_professional_risk_item(content: str, source_type: str | None) -> bool:
    normalized = re.sub(r"\s+", " ", content or "").strip()
    risk_markers = (
        "风险", "不及预期", "下滑", "下降", "波动", "竞争", "价格", "需求", "政策", "监管",
        "汇率", "原材料", "减值", "产能", "关税", "安全", "质量", "诉讼", "债务", "供应链",
    )
    low_quality_markers = ("散户", "持仓股", "大众情人", "这只票", "这6只票", "股吧")
    return (
        len(normalized) >= 8
        and source_type in {"RESEARCH_REPORT", "ANNOUNCEMENT"}
        and any(marker in normalized for marker in risk_markers)
        and not any(marker in normalized for marker in low_quality_markers)
    )


def _citation_references(text: str) -> list[int]:
    return [int(value) for value in re.findall(r"\[(\d+)]", text)]


def _number_tokens(text: str) -> list[str]:
    without_references = re.sub(r"\[\d+]", "", text)
    # Preserve the sign. Dropping it made a supported value such as ``-6.31%``
    # compare against ``6.31%`` and falsely trigger UNSUPPORTED_NUMBER.
    return [token.replace(",", "") for token in re.findall(r"(?<![A-Za-z0-9])[+-]?\d[\d,]*(?:\.\d+)?%?", without_references)]


def contains_direct_trading_advice(text: str) -> bool:
    """Detect imperative portfolio/trading instructions, not quoted ratings."""
    normalized = re.sub(r"\s+", "", text or "")
    if re.search(r"(?:券商|机构|分析师).{0,8}(?:给予|维持|评级为|建议).{0,4}(?:买入|卖出|增持|减持)", normalized):
        return False
    patterns = (
        r"(?:建议|应该|应当|务必|立即|马上|现在就)(?:你|投资者|用户)?(?:直接|立即|马上|考虑)?(?:买入|卖出|加仓|减仓|清仓|满仓|建仓)",
        r"(?:你|投资者|用户).{0,8}(?:可以买入|应买入|应卖出|加仓|减仓|清仓|满仓|建仓)",
        r"(?:仓位|持仓)(?:应|建议|控制在|提高到|降低到).{0,8}\d+(?:\.\d+)?%",
    )
    return any(re.search(pattern, normalized) for pattern in patterns)


def validate_grounded_answer(answer: str, citations: list[dict[str, Any]]) -> dict[str, Any]:
    """Check citation integrity and whether numeric claims are source-backed.

    This is intentionally deterministic. It does not claim to prove prose true;
    it prevents the two highest-risk presentation errors: invented citation
    numbers and uncited/unsupported financial figures.
    """
    issues: list[dict[str, Any]] = []
    references = _citation_references(answer)
    valid_numbers = set(range(1, len(citations) + 1))
    invalid_references = sorted(set(references) - valid_numbers)
    if invalid_references:
        issues.append({"code": "INVALID_CITATION", "references": invalid_references})
    if citations and not references:
        issues.append({"code": "MISSING_CITATIONS"})

    numeric_claims = 0
    supported_numeric_claims = 0
    factual_claims = 0
    cited_factual_claims = 0
    units = ("元", "亿", "万", "%", "倍", "股", "吨", "辆", "人", "家", "年", "月", "日")
    factual_markers = (
        "同比", "环比", "增长", "下降", "提升", "减少", "预计", "预测", "评级", "目标价",
        "营收", "收入", "利润", "现金流", "价格", "销量", "产能", "订单", "发布", "公告",
        "披露", "情绪", "舆情", "风险", "亏损", "扭亏", "创新高", "创新低",
    )
    # Citations in research writing conventionally support one paragraph or
    # list item, and are often placed at its beginning/end. Splitting again at
    # every Chinese full stop would falsely mark the remaining sentences in the
    # same cited bullet as uncited.
    segments = [item.strip() for item in answer.splitlines() if item.strip()]
    for segment in segments:
        if segment.startswith("#") or "不构成投资建议" in segment:
            continue
        segment_references = [number for number in _citation_references(segment) if number in valid_numbers]
        if contains_direct_trading_advice(segment):
            issues.append({"code": "DIRECT_TRADING_ADVICE", "text": segment[:180]})
        # Markdown table headers are labels, not factual claims. Data rows carry
        # their own citation in the final column and continue through validation.
        if segment.startswith("|") and not segment_references and not re.search(r"\d", segment):
            continue
        if len(re.sub(r"\[\d+]", "", segment)) >= 10 and any(marker in segment for marker in factual_markers):
            factual_claims += 1
            if segment_references:
                cited_factual_claims += 1
            else:
                issues.append({"code": "UNCITED_FACT", "text": segment[:180]})
        tokens = _number_tokens(segment)
        if not tokens or not any(unit in segment for unit in units):
            continue
        numeric_claims += 1
        cited = segment_references
        if not cited:
            issues.append({"code": "UNCITED_NUMBER", "text": segment[:180]})
            continue
        evidence_text = " ".join(
            f"{citations[number - 1].get('title', '')} {citations[number - 1].get('quote', '')}" for number in cited
        ).replace(",", "")
        unsupported = [token for token in tokens if token not in evidence_text]
        if unsupported:
            issues.append({"code": "UNSUPPORTED_NUMBER", "numbers": unsupported, "text": segment[:180]})
        else:
            supported_numeric_claims += 1

    # Keep only structural citation fraud and direct trading instructions as
    # hard blockers. Numeric/factual coverage is a quality warning: models may
    # legitimately calculate or infer values from several cited observations.
    hard_codes = {"INVALID_CITATION", "DIRECT_TRADING_ADVICE"}
    hard_issues = [issue for issue in issues if issue["code"] in hard_codes]
    soft_issues = [issue for issue in issues if issue["code"] not in hard_codes]
    return {
        "valid": not issues,
        # ``valid`` remains the strict quality signal for diagnostics. Runtime
        # routing uses ``hard_valid`` so an uncited analytical sentence is a
        # warning rather than a reason to discard the whole model answer.
        "hard_valid": not hard_issues,
        "issues": issues,
        "hard_issues": hard_issues,
        "soft_issues": soft_issues,
        "hard_issue_count": len(hard_issues),
        "soft_issue_count": len(soft_issues),
        "citation_count": len(citations),
        "referenced_citation_count": len(set(references) & valid_numbers),
        "numeric_claim_count": numeric_claims,
        "supported_numeric_claim_count": supported_numeric_claims,
        "factual_claim_count": factual_claims,
        "cited_factual_claim_count": cited_factual_claims,
    }


def filter_unsupported_lines(
    answer: str,
    citations: list[dict[str, Any]],
    *,
    hard_only: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Keep safe LLM paragraphs and remove only unsupported material claims."""
    blocking_codes = {"INVALID_CITATION", "DIRECT_TRADING_ADVICE"}
    if not hard_only:
        blocking_codes.update({"UNCITED_NUMBER", "UNSUPPORTED_NUMBER", "UNCITED_FACT"})
    kept: list[str] = []
    removed = 0
    for line in answer.splitlines():
        if not line.strip():
            kept.append(line)
            continue
        line_validation = validate_grounded_answer(line, citations)
        if any(issue["code"] in blocking_codes for issue in line_validation["issues"]):
            removed += 1
            continue
        kept.append(line)
    # Remove headings whose entire section was filtered out. In particular,
    # avoid leaving a dangling “风险与不确定性：” at the end of the answer.
    cleaned: list[str] = []
    for index, line in enumerate(kept):
        stripped = line.strip()
        is_heading = stripped.startswith("#") or (stripped.endswith(("：", ":")) and len(stripped) <= 30)
        if is_heading:
            next_content = next((item.strip() for item in kept[index + 1 :] if item.strip()), "")
            next_is_heading = next_content.startswith("#") or (next_content.endswith(("：", ":")) and len(next_content) <= 30)
            if not next_content or next_is_heading:
                continue
        cleaned.append(line)
    filtered = "\n".join(cleaned).strip()
    if filtered and "不构成投资建议" not in filtered:
        filtered += "\n\n> 仅供研究辅助，不构成投资建议。"
    validation = validate_grounded_answer(filtered, citations)
    validation["removed_line_count"] = removed
    return filtered, validation
