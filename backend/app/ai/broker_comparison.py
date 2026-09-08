from __future__ import annotations

from collections import defaultdict
from typing import Any


def _date(value: str | None) -> str:
    return value[:10] if value else "日期未知"


def _amount_yi(value: float | None, unit: str | None) -> float | None:
    if value is None:
        return None
    if unit and "百万元" in unit:
        return value / 100
    return value


def _amount_text(value: float | None, unit: str | None) -> str:
    converted = _amount_yi(value, unit)
    return "—" if converted is None else f"{converted:.2f}亿元"


def _eps_text(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}元"


def build_broker_citations(broker_data: dict[str, Any]) -> list[dict[str, Any]]:
    """Build concise, source-linked citations from structured report rows."""
    rows_by_document: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in [*broker_data.get("ratings", []), *broker_data.get("forecasts", [])]:
        if row.get("document_id"):
            rows_by_document[int(row["document_id"])].append(row)

    citations: list[dict[str, Any]] = []
    for document_id, rows in sorted(
        rows_by_document.items(),
        key=lambda item: max((row.get("published_at") or "" for row in item[1]), default=""),
        reverse=True,
    ):
        first = rows[0]
        parts: list[str] = []
        rating = next((row for row in rows if "rating" in row), None)
        if rating and rating.get("rating"):
            parts.append(f"原始评级{rating['rating']}")
        if rating and rating.get("target_price") is not None:
            parts.append(f"目标价{rating['target_price']:g}元")
        forecasts = sorted((row for row in rows if "year" in row), key=lambda row: row["year"])
        for row in forecasts:
            metrics = []
            if row.get("revenue") is not None:
                metrics.append(f"营业收入{_amount_text(row['revenue'], row.get('unit'))}")
            if row.get("net_profit") is not None:
                metrics.append(f"归母净利润{_amount_text(row['net_profit'], row.get('unit'))}")
            if row.get("eps") is not None:
                metrics.append(f"EPS{_eps_text(row['eps'])}")
            if metrics:
                parts.append(f"{row['year']}年" + "、".join(metrics))
        evidence = "；".join(
            str(row.get("evidence") or "").strip()
            for row in rows
            if row.get("evidence")
        )
        quote = f"结构化研报数据（{_date(first.get('published_at'))}）：" + "；".join(parts)
        if evidence:
            quote += f"。抽取证据：{evidence[:500]}"
        citations.append({
            "document_id": document_id,
            "title": first.get("title") or "券商研报",
            "source_type": "RESEARCH_REPORT",
            "source_url": first.get("source_url") or "",
            "published_at": first.get("published_at"),
            "page": first.get("page"),
            "quote": quote[:1200],
            "source_name": first.get("source_name") or first.get("institution"),
        })
    return citations


def _references(rows: list[dict[str, Any]], citation_numbers: dict[int, int]) -> str:
    numbers = sorted({citation_numbers[int(row["document_id"])] for row in rows if row.get("document_id") in citation_numbers})
    return "".join(f"[{number}]" for number in numbers)


def _latest_per_institution(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in sorted(rows, key=lambda item: item.get("published_at") or "", reverse=True):
        institution = row.get("institution") or row.get("source_name") or "未知机构"
        latest.setdefault(institution, row)
    return list(latest.values())


def _rating_direction(value: str | None) -> str | None:
    if value in {"POSITIVE", "SLIGHTLY_POSITIVE"}:
        return "POSITIVE"
    if value in {"NEGATIVE", "SLIGHTLY_NEGATIVE"}:
        return "NEGATIVE"
    return value


def build_broker_comparison_answer(
    company_name: str,
    broker_data: dict[str, Any],
    citations: list[dict[str, Any]],
) -> str:
    """Create a deterministic comparison; never infer figures absent from rows."""
    citation_numbers = {int(item["document_id"]): index for index, item in enumerate(citations, 1) if item.get("document_id")}
    ratings = _latest_per_institution(broker_data.get("ratings", []))

    forecast_latest: dict[tuple[str, int], dict[str, Any]] = {}
    for row in sorted(broker_data.get("forecasts", []), key=lambda item: item.get("published_at") or "", reverse=True):
        institution = row.get("institution") or row.get("source_name") or "未知机构"
        forecast_latest.setdefault((institution, int(row["year"])), row)
    forecasts = sorted(forecast_latest.values(), key=lambda row: (int(row["year"]), row.get("institution") or ""))

    lines = [f"### {company_name}券商评级与盈利预测分歧", ""]
    if len({row.get("institution") for row in ratings}) >= 2:
        normalized = {_rating_direction(row.get("normalized")) for row in ratings if row.get("normalized")}
        original = {row.get("rating") for row in ratings if row.get("rating")}
        refs = _references(ratings, citation_numbers)
        if len(normalized) == 1:
            if len(original) > 1:
                lines.append(f"**简明结论：** 当前有效研报的评级方向总体一致、均偏积极，但原始措辞存在“{'、'.join(sorted(original))}”等口径差异；这主要反映券商评级体系不同，不等同于投资态度相反。{refs}")
            else:
                lines.append(f"**简明结论：** 当前有效研报的评级方向和原始评级基本一致，评级层面未见明显方向性分歧。{refs}")
        else:
            lines.append(f"**简明结论：** 当前有效研报在评级方向上存在分歧，需结合各券商自身评级定义解读。{refs}")
    elif ratings:
        lines.append(f"**简明结论：** 当前只有单一机构的有效评级样本，尚不足以判断机构间评级分歧。{_references(ratings, citation_numbers)}")
    else:
        lines.append("**简明结论：** 当前没有可核验的结构化券商评级，不能可靠判断评级分歧。")

    lines.extend(["", "#### 评级对比", "", "| 机构 | 原始评级 | 目标价 | 研报日期 | 来源 |", "| --- | --- | ---: | --- | --- |"])
    if ratings:
        for row in ratings:
            target = "—" if row.get("target_price") is None else f"{row['target_price']:g}元"
            lines.append(
                f"| {row.get('institution') or '未知机构'} | {row.get('rating') or '未披露'} | {target} | "
                f"{_date(row.get('published_at'))} | {_references([row], citation_numbers) or '—'} |"
            )
    else:
        lines.append("| 暂无有效数据 | — | — | — | — |")

    lines.extend(["", "#### 盈利预测对比", "", "单位：金额为亿元，EPS 为元/股。", "", "| 预测年度 | 机构 | 营业收入 | 归母净利润 | EPS | 来源 |", "| --- | --- | ---: | ---: | ---: | --- |"])
    if forecasts:
        for row in forecasts:
            lines.append(
                f"| {row['year']}E | {row.get('institution') or '未知机构'} | {_amount_text(row.get('revenue'), row.get('unit'))} | "
                f"{_amount_text(row.get('net_profit'), row.get('unit'))} | {_eps_text(row.get('eps'))} | "
                f"{_references([row], citation_numbers) or '—'} |"
            )
    else:
        lines.append("| 暂无有效数据 | — | — | — | — | — |")

    comparisons: list[str] = []
    by_year: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in forecasts:
        by_year[int(row["year"])].append(row)
    for year, rows in sorted(by_year.items()):
        if len({row.get("institution") for row in rows}) < 2:
            continue
        parts = []
        referenced_rows: list[dict[str, Any]] = []
        for key, label, formatter in (
            ("revenue", "营业收入", lambda row: _amount_text(row.get("revenue"), row.get("unit"))),
            ("net_profit", "归母净利润", lambda row: _amount_text(row.get("net_profit"), row.get("unit"))),
            ("eps", "EPS", lambda row: _eps_text(row.get("eps"))),
        ):
            available = [row for row in rows if row.get(key) is not None]
            if len({row.get("institution") for row in available}) < 2:
                continue
            low = min(available, key=lambda row: _amount_yi(row[key], row.get("unit")) if key != "eps" else row[key])
            high = max(available, key=lambda row: _amount_yi(row[key], row.get("unit")) if key != "eps" else row[key])
            parts.append(f"{label}从{formatter(low)}到{formatter(high)}")
            referenced_rows.extend([low, high])
        if parts:
            comparisons.append(f"- {year}年预测区间：" + "；".join(parts) + f"。{_references(referenced_rows, citation_numbers)}")

    lines.extend(["", "#### 分歧解读", ""])
    if comparisons:
        lines.extend(comparisons)
        lines.append("- 这些区间通常源于不同机构对出货量、价格、产品结构、成本与盈利能力等假设的差异；表中估计值不代表已经实现的业绩。" + _references(forecasts, citation_numbers))
    else:
        lines.append("- 当前至少两个机构的同年度可比数据不足，数值分歧暂不下结论。")
    lines.extend(["", "> 券商评级体系并不完全统一，盈利预测也会随新信息调整；仅供研究辅助，不构成投资建议。"])
    return "\n".join(lines)
