from __future__ import annotations

from collections import Counter
from datetime import datetime
from difflib import SequenceMatcher
import re
from typing import Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ai.index import hybrid_search
from app.ai.llm import invoke_text
from app.core.models import (
    Company,
    Document,
    EarningsForecast,
    FinancialMetric,
    InvestmentRating,
    Opinion,
    RiskItem,
    SentimentResult,
)


SOURCE_LABELS = {
    "RESEARCH_REPORT": "券商研报",
    "NEWS": "财经新闻",
    "ANNOUNCEMENT": "公司公告",
    "SOCIAL": "社交舆情",
}
RATING_LABELS = {
    "POSITIVE": "积极",
    "SLIGHTLY_POSITIVE": "偏积极",
    "NEUTRAL": "中性",
    "SLIGHTLY_NEGATIVE": "偏谨慎",
    "NEGATIVE": "谨慎",
}
METRIC_LABELS = {
    "revenue": "营业收入",
    "operating_revenue": "营业收入",
    "营业收入": "营业收入",
    "归母净利润": "归母净利润",
    "net_profit": "归母净利润",
    "net_profit_attributable": "归母净利润",
    "扣非净利润": "扣非净利润",
    "adjusted_net_profit": "扣非净利润",
    "eps": "每股收益（EPS）",
    "每股收益": "每股收益（EPS）",
    "gross_margin": "毛利率",
    "毛利率": "毛利率",
    "operating_cash_flow": "经营现金流",
    "经营现金流": "经营现金流",
}


class CitationRegistry:
    def __init__(self, documents: dict[int, Document]):
        self.documents = documents
        self.items: list[dict] = []
        self._keys: dict[tuple[int, str, int | None], int] = {}

    def add(self, document_id: int, quote: str, page: int | None = None) -> int | None:
        document = self.documents.get(document_id)
        clean_quote = re.sub(r"\s+", " ", quote or "").strip()[:700]
        if not document or not clean_quote:
            return None
        key = (document_id, clean_quote, page)
        if key in self._keys:
            return self._keys[key]
        number = len(self.items) + 1
        self._keys[key] = number
        self.items.append({
            "number": number,
            "document_id": document.id,
            "title": document.title,
            "source_type": document.document_type,
            "source_name": document.source_name,
            "source_url": document.source_url,
            "published_at": document.published_at.isoformat() if document.published_at else None,
            "page": page,
            "quote": clean_quote,
            "credibility": _credibility(document.document_type),
        })
        return number


def _credibility(source_type: str) -> str:
    return {
        "ANNOUNCEMENT": "一级事实来源",
        "RESEARCH_REPORT": "机构分析与预测",
        "NEWS": "事件报道",
        "SOCIAL": "低可信度情绪参考",
    }.get(source_type, "公开资料")


def _ref(number: int | None) -> str:
    return f" [{number}]" if number else ""


def _format_value(value: float | None, raw: str | None = None, unit: str | None = None) -> str:
    if raw:
        return raw
    if value is None:
        return "-"
    rendered = f"{value:,.4f}".rstrip("0").rstrip(".")
    return f"{rendered}{unit or ''}"


def _metric_label(name: str) -> str:
    key = (name or "").strip()
    return METRIC_LABELS.get(key, METRIC_LABELS.get(key.lower(), key or "未命名指标"))


def _statement_similarity(left: str, right: str) -> float:
    def compact(value: str) -> str:
        return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]", "", value.lower())

    a, b = compact(left), compact(right)
    if not a or not b:
        return 0.0
    a_pairs = {a[index:index + 2] for index in range(max(1, len(a) - 1))}
    b_pairs = {b[index:index + 2] for index in range(max(1, len(b) - 1))}
    union = a_pairs | b_pairs
    return max(SequenceMatcher(None, a, b).ratio(), len(a_pairs & b_pairs) / len(union) if union else 0.0)


def _clusters(points: list[dict], minimum_documents: int = 2) -> list[list[dict]]:
    clusters: list[list[dict]] = []
    for point in points:
        content = re.sub(r"\s+", " ", point.get("content", "")).strip()
        if len(content) < 6:
            continue
        candidate = {**point, "content": content}
        target = None
        best = 0.0
        for cluster in clusters:
            score = max(_statement_similarity(content, member["content"]) for member in cluster)
            if score > best:
                target, best = cluster, score
        if target is not None and best >= 0.46:
            target.append(candidate)
        else:
            clusters.append([candidate])
    return [cluster for cluster in clusters if len({item["document_id"] for item in cluster}) >= minimum_documents]


def _safe_llm_summary(company_name: str, heading: str, evidence: list[tuple[int, str]]) -> str | None:
    if not evidence:
        return None
    evidence_text = "\n".join(f"[{number}] {text[:500]}" for number, text in evidence[:10])
    prompt = f"""你是严谨的金融研究编辑。仅根据下列证据撰写“{company_name}”的“{heading}”。
要求：输出2至4条Markdown项目符号；只讨论{company_name}，忽略证据中顺带提及的其他公司；每条结论结尾必须有至少一个[n]引用；不得补充证据中没有的数字；区分事实、机构预测与舆情；不输出买卖指令。

证据：
{evidence_text}
"""
    result = invoke_text(prompt)
    if not result:
        return None
    allowed = {str(number) for number, _ in evidence}
    lines = [line.strip() for line in result.splitlines() if line.strip().startswith(("-", "*"))]
    lines = [line for line in lines if company_name in line or re.match(r"^[-*]\s*(该公司|公司|其|上述)", line)]
    if not lines:
        return None
    result = "\n".join(lines)
    references = re.findall(r"\[(\d+)\]", result)
    if not references or any(number not in allowed for number in references):
        return None
    allowed_numbers = set(re.findall(r"\d+(?:\.\d+)?", evidence_text)) | set(references)
    generated_numbers = set(re.findall(r"\d+(?:\.\d+)?", re.sub(r"\[\d+\]", "", result)))
    if not generated_numbers.issubset(allowed_numbers):
        return None
    return result.strip()


def _load_scope(
    db: Session,
    company: Company,
    date_from: datetime | None,
    date_to: datetime | None,
    source_types: list[str],
    document_ids: list[int] | None,
) -> list[Document]:
    query = select(Document).where(Document.company_id == company.id, Document.is_deleted.is_(False))
    if date_from:
        query = query.where(Document.published_at >= date_from)
    if date_to:
        query = query.where(Document.published_at <= date_to)
    if source_types:
        query = query.where(Document.document_type.in_(source_types))
    if document_ids:
        query = query.where(Document.id.in_(document_ids))
    return list(db.scalars(query.order_by(Document.published_at.desc(), Document.id.desc())).all())


def _evidence_for_scope(
    db: Session,
    company: Company,
    documents: list[Document],
    source_types: list[str],
    date_from: datetime | None,
    date_to: datetime | None,
) -> list[dict]:
    if not documents:
        return []
    allowed_ids = {item.id for item in documents}
    candidates = hybrid_search(
        db,
        "公司重要事件 财务表现 盈利预测 评级 核心观点 风险 情绪",
        company.id,
        limit=min(50, max(12, len(documents) * 3)),
        document_types=source_types or None,
        date_from=date_from,
        date_to=date_to,
    )
    return [item for item in candidates if item["document_id"] in allowed_ids][:12]


def _base_header(company: Company, title: str, documents: list[Document], date_from: datetime | None, date_to: datetime | None) -> list[str]:
    counts = Counter(item.document_type for item in documents)
    coverage = "、".join(f"{SOURCE_LABELS.get(key, key)}{value}篇" for key, value in counts.items()) or "无可用资料"
    latest = max((item.published_at for item in documents if item.published_at), default=None)
    return [
        f"# {company.name}{title}",
        "",
        f"> 股票代码：{company.stock_code}　生成时间：{datetime.now():%Y-%m-%d %H:%M}",
        f"> 资料范围：{date_from:%Y-%m-%d} 至 {date_to:%Y-%m-%d}" if date_from and date_to else f"> 资料范围：{date_from:%Y-%m-%d} 至不限" if date_from else f"> 资料范围：不限至 {date_to:%Y-%m-%d}" if date_to else "> 资料范围：不限",
        f"> 数据覆盖：{coverage}；最新资料：{latest:%Y-%m-%d %H:%M}" if latest else f"> 数据覆盖：{coverage}；最新资料：未知",
        "> 标注说明：财务实际值、机构预测和市场舆情分别展示；本报告仅供研究演示，不构成投资建议。",
        "",
    ]


def _company_brief(
    db: Session,
    company: Company,
    documents: list[Document],
    date_from: datetime | None,
    date_to: datetime | None,
    source_types: list[str],
    registry: CitationRegistry,
    progress: Callable[[int, str], None],
) -> list[str]:
    ids = [item.id for item in documents]
    if not ids:
        return _base_header(company, "公司研究简报", documents, date_from, date_to) + ["## 资料不足", "", "当前筛选范围内没有可用文档，系统没有生成无证据结论。"]
    progress(25, "整理实际财务数据")
    metrics = list(db.scalars(select(FinancialMetric).where(FinancialMetric.document_id.in_(ids)).order_by(FinancialMetric.period.desc(), FinancialMetric.id.desc()).limit(30)).all())
    forecasts = list(db.scalars(select(EarningsForecast).where(EarningsForecast.document_id.in_(ids)).order_by(EarningsForecast.forecast_year, EarningsForecast.id.desc()).limit(30)).all())
    ratings = list(db.scalars(select(InvestmentRating).where(InvestmentRating.document_id.in_(ids)).order_by(InvestmentRating.id.desc()).limit(20)).all())
    opinions = list(db.scalars(select(Opinion).where(Opinion.document_id.in_(ids)).order_by(Opinion.id.desc()).limit(30)).all())
    risks = list(db.scalars(select(RiskItem).where(RiskItem.document_id.in_(ids)).order_by(RiskItem.id.desc()).limit(20)).all())
    sentiments = list(db.scalars(select(SentimentResult).where(SentimentResult.document_id.in_(ids))).all())
    evidence = _evidence_for_scope(db, company, documents, source_types, date_from, date_to)
    evidence_refs: list[tuple[int, str]] = []
    for item in evidence:
        number = registry.add(item["document_id"], item["quote"], item.get("page"))
        if number:
            evidence_refs.append((number, item["quote"]))

    lines = _base_header(company, "公司研究简报", documents, date_from, date_to)
    progress(40, "生成研究摘要")
    summary = _safe_llm_summary(company.name, "核心研究摘要", evidence_refs)
    lines.extend(["## 核心研究摘要", ""])
    if summary:
        lines.append(summary)
    elif evidence_refs:
        focused = [item for item in evidence_refs if company.name in item[1]] or evidence_refs
        lines.extend(f"- {text[:220]} [{number}]" for number, text in focused[:3])
    else:
        lines.append("- 当前资料尚未形成可检索的正文证据。")

    lines.extend(["", "## 历史实际财务数据", ""])
    if metrics:
        lines.extend(["| 指标 | 报告期 | 实际值 | 同比 | 来源 |", "|---|---|---:|---:|---|"])
        seen: set[tuple[str, str | None]] = set()
        for item in metrics:
            key = (_metric_label(item.metric_name), item.period)
            if key in seen:
                continue
            seen.add(key)
            number = registry.add(item.document_id, f"{key[0]} {item.period or ''} {_format_value(item.normalized_value, item.raw_value, item.unit)}")
            yoy = f"{item.yoy:.2f}%" if item.yoy is not None else "-"
            lines.append(f"| {key[0]} | {item.period or '-'} | {_format_value(item.normalized_value, item.raw_value, item.unit)} | {yoy} |{_ref(number)} |")
            if len(seen) >= 12:
                break
    else:
        lines.append("当前筛选资料中未抽取到实际财务指标。")

    progress(55, "整理机构预测与评级")
    lines.extend(["", "## 机构盈利预测", ""])
    if forecasts:
        lines.extend(["| 机构 | 预测年度 | 预测收入 | 预测净利润 | 预测EPS | 来源 |", "|---|---:|---:|---:|---:|---|"])
        for item in forecasts[:20]:
            number = registry.add(item.document_id, f"{item.institution or '机构'}预测{item.forecast_year}年收入、净利润和EPS")
            lines.append(f"| {item.institution or '-'} | {item.forecast_year} | {_format_value(item.revenue, unit=item.unit)} | {_format_value(item.net_profit, unit=item.unit)} | {_format_value(item.eps)} |{_ref(number)} |")
    else:
        lines.append("当前筛选资料中未抽取到机构盈利预测。")

    lines.extend(["", "## 投资评级与目标价", ""])
    if ratings:
        lines.extend(["| 机构 | 原始评级 | 统一倾向 | 目标价 | 来源 |", "|---|---|---|---:|---|"])
        for item in ratings:
            number = registry.add(item.document_id, f"{item.institution or '机构'}评级{item.original_rating or '-'}，目标价{item.target_price or '-'}")
            normalized = RATING_LABELS.get(item.normalized_rating or "", item.normalized_rating or "-")
            lines.append(f"| {item.institution or '-'} | {item.original_rating or '-'} | {normalized} | {_format_value(item.target_price, unit=item.currency)} |{_ref(number)} |")
    else:
        lines.append("当前筛选资料中未抽取到有效投资评级。")

    lines.extend(["", "## 主要观点与事件", ""])
    if opinions:
        for item in opinions[:10]:
            number = registry.add(item.document_id, item.content)
            lines.append(f"- {item.content}{_ref(number)}")
    else:
        lines.append("- 当前筛选资料中未抽取到可验证观点。")

    if sentiments:
        distribution = Counter(item.sentiment for item in sentiments)
        rendered = "、".join(f"{key}{value}条" for key, value in distribution.most_common())
        lines.extend(["", "## 新闻与舆情观察", "", f"- 已分析情感记录：{rendered}。该结果只反映文本情绪，不等同于基本面判断。"])

    progress(70, "整理风险提示")
    lines.extend(["", "## 主要风险", ""])
    if risks:
        for item in risks[:10]:
            number = registry.add(item.document_id, item.content)
            lines.append(f"- **{item.category}**：{item.content}{_ref(number)}")
    else:
        lines.append("- 当前筛选资料中未抽取到有原文依据的风险项。")
    return lines


def _comparison_report(
    db: Session,
    company: Company,
    documents: list[Document],
    date_from: datetime | None,
    date_to: datetime | None,
    registry: CitationRegistry,
    progress: Callable[[int, str], None],
) -> list[str]:
    research = [item for item in documents if item.document_type == "RESEARCH_REPORT"]
    lines = _base_header(company, "多研报观点对比报告", research, date_from, date_to)
    if len(research) < 2:
        return lines + ["## 无法完成对比", "", "当前只选中了少于两篇券商研报。至少选择两篇研报后，系统才会判断共识与分歧。"]
    ids = [item.id for item in research]
    progress(30, "逐篇读取研报评级和预测")
    ratings = {item.document_id: item for item in db.scalars(select(InvestmentRating).where(InvestmentRating.document_id.in_(ids))).all()}
    forecasts = list(db.scalars(select(EarningsForecast).where(EarningsForecast.document_id.in_(ids)).order_by(EarningsForecast.forecast_year)).all())
    opinions = list(db.scalars(select(Opinion).where(Opinion.document_id.in_(ids))).all())
    risks = list(db.scalars(select(RiskItem).where(RiskItem.document_id.in_(ids))).all())
    doc_forecasts: dict[int, list[EarningsForecast]] = {item.id: [] for item in research}
    for forecast in forecasts:
        doc_forecasts.setdefault(forecast.document_id, []).append(forecast)

    lines.extend(["## 参与比较的研报", "", "| 日期 | 机构 | 研报 | 评级 | 目标价 | 来源 |", "|---|---|---|---|---:|---|"])
    for document in research:
        rating = ratings.get(document.id)
        institution = rating.institution if rating and rating.institution else document.source_name
        quote = f"{institution}发布《{document.title}》，评级{rating.original_rating if rating else '未抽取'}"
        number = registry.add(document.id, quote)
        lines.append(f"| {document.published_at:%Y-%m-%d} | {institution} | {document.title} | {rating.original_rating if rating and rating.original_rating else '-'} | {_format_value(rating.target_price if rating else None, unit=rating.currency if rating else None)} |{_ref(number)} |" if document.published_at else f"| - | {institution} | {document.title} | {rating.original_rating if rating and rating.original_rating else '-'} | {_format_value(rating.target_price if rating else None, unit=rating.currency if rating else None)} |{_ref(number)} |")

    progress(48, "对齐不同机构盈利预测")
    lines.extend(["", "## 盈利预测横向对比", ""])
    if forecasts:
        lines.extend(["| 机构 | 年度 | 收入预测 | 净利润预测 | EPS预测 | 来源 |", "|---|---:|---:|---:|---:|---|"])
        for document in research:
            rating = ratings.get(document.id)
            institution = rating.institution if rating and rating.institution else document.source_name
            for item in doc_forecasts.get(document.id, []):
                number = registry.add(document.id, f"{institution}对{item.forecast_year}年的盈利预测")
                lines.append(f"| {institution} | {item.forecast_year} | {_format_value(item.revenue, unit=item.unit)} | {_format_value(item.net_profit, unit=item.unit)} | {_format_value(item.eps)} |{_ref(number)} |")
    else:
        lines.append("所选研报尚未抽取到可横向比较的盈利预测。")

    progress(62, "识别机构共识与分歧")
    points = [{"document_id": item.document_id, "content": item.content} for item in opinions]
    consensus = _clusters(points)
    risk_clusters = _clusters([{"document_id": item.document_id, "content": item.content} for item in risks])
    lines.extend(["", "## 机构共识", ""])
    if consensus:
        for cluster in consensus[:5]:
            representative = min((item["content"] for item in cluster), key=len)[:260]
            refs = sorted({registry.add(item["document_id"], item["content"]) for item in cluster})
            suffix = "".join(_ref(number) for number in refs if number)
            lines.append(f"- {representative}（{len({item['document_id'] for item in cluster})}篇研报共同提及）{suffix}")
    else:
        lines.append("- 所选研报尚未形成至少两篇报告共同支持的相似观点；系统不会把单一机构观点当作共识。")

    rating_distribution = Counter((ratings.get(item.id).normalized_rating if ratings.get(item.id) else None) for item in research)
    rating_distribution.pop(None, None)
    consensus_doc_ids = {item["document_id"] for cluster in consensus for item in cluster}
    lines.extend(["", "## 主要分歧与独有观点", ""])
    if len(rating_distribution) > 1:
        rendered = "、".join(f"{RATING_LABELS.get(key, key)}{value}篇" for key, value in rating_distribution.most_common())
        lines.append(f"- 评级方向存在差异：{rendered}。")
    unique_points = [item for item in points if item["document_id"] not in consensus_doc_ids]
    for item in unique_points[:6]:
        document = registry.documents[item["document_id"]]
        number = registry.add(item["document_id"], item["content"])
        lines.append(f"- **{ratings.get(document.id).institution if ratings.get(document.id) and ratings.get(document.id).institution else document.source_name}**：{item['content']}{_ref(number)}")
    if len(rating_distribution) <= 1 and not unique_points:
        lines.append("- 当前抽取结果未发现明确分歧。")

    lines.extend(["", "## 共同风险与风险差异", ""])
    if risk_clusters:
        for cluster in risk_clusters[:5]:
            representative = min((item["content"] for item in cluster), key=len)[:260]
            refs = sorted({registry.add(item["document_id"], item["content"]) for item in cluster})
            lines.append(f"- {representative}{''.join(_ref(number) for number in refs if number)}")
    else:
        lines.append("- 暂未发现至少两篇研报共同提及的相似风险。")
    common_risk_doc_ids = {item["document_id"] for cluster in risk_clusters for item in cluster}
    for item in [risk for risk in risks if risk.document_id not in common_risk_doc_ids][:5]:
        document = registry.documents[item.document_id]
        number = registry.add(item.document_id, item.content)
        lines.append(f"- **{document.source_name}独有提示**：{item.content}{_ref(number)}")
    return lines


def _append_sources(lines: list[str], citations: list[dict]) -> None:
    lines.extend(["", "## 引用与证据", ""])
    if not citations:
        lines.append("当前筛选资料未形成可验证引用。")
        return
    for item in citations:
        page = f"，第{item['page']}页" if item.get("page") else ""
        date = f"，{item['published_at'][:10]}" if item.get("published_at") else ""
        lines.append(f"{item['number']}. {item['quote']}（{item['credibility']}；来源：[{item['title']}]({item['source_url']}){date}{page}）")


def build_report(
    db: Session,
    company: Company,
    report_type: str,
    date_from: datetime | None,
    date_to: datetime | None,
    source_types: list[str],
    document_ids: list[int] | None = None,
    progress: Callable[[int, str], None] | None = None,
) -> tuple[str, list[dict]]:
    update = progress or (lambda _value, _text: None)
    update(10, "检查资料范围")
    effective_types = ["RESEARCH_REPORT"] if report_type == "REPORT_COMPARISON" else source_types
    documents = _load_scope(db, company, date_from, date_to, effective_types, document_ids)
    if document_ids and len(documents) != len(set(document_ids)):
        raise ValueError("部分所选资料不存在、不属于该公司或不在筛选范围内")
    if report_type == "REPORT_COMPARISON" and len(documents) < 2:
        raise ValueError("多研报观点对比至少需要选择两篇券商研报")
    registry = CitationRegistry({item.id: item for item in documents})
    if report_type == "COMPANY_BRIEF":
        lines = _company_brief(db, company, documents, date_from, date_to, effective_types, registry, update)
    else:
        lines = _comparison_report(db, company, documents, date_from, date_to, registry, update)
    update(82, "校验引用与资料范围")
    _append_sources(lines, registry.items)
    lines.extend(["", "## 质量检查", "", f"- 已引用 {len(registry.items)} 条证据，全部关联到当前报告选定的原始资料。", "- 预测值已单独标记为机构预测；无证据字段保留为空或明确提示资料不足。"])
    update(92, "组装最终报告")
    return "\n".join(lines), registry.items
