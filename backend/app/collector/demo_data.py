from __future__ import annotations

from datetime import datetime, timedelta

from app.collector.adapters.base import CollectedItem


def demo_items(stock_code: str, company_name: str, source_type: str) -> list[CollectedItem]:
    """Synthetic, clearly labelled fallback content for offline classroom demonstrations."""
    now = datetime.now().replace(microsecond=0)
    prefix = "课程演示快照（合成数据，非真实投资信息）"
    if source_type == "RESEARCH_REPORT":
        return [
            CollectedItem(
                title=f"{prefix}：{company_name}经营跟踪报告A",
                source_name="课程演示研究机构A",
                source_url=f"snapshot://research/{stock_code}/a",
                document_type=source_type,
                published_at=now - timedelta(days=2),
                raw_text=f"{company_name}（{stock_code}）示例研报。评级：增持，目标价：42.00元。预计2027年营业收入1200亿元，归母净利润85亿元，EPS 2.10元。核心观点：示例业务需求保持增长，成本控制改善。风险提示：行业竞争加剧、原材料价格波动、需求不及预期。本材料仅用于课程功能演示。",
            ),
            CollectedItem(
                title=f"{prefix}：{company_name}经营跟踪报告B",
                source_name="课程演示研究机构B",
                source_url=f"snapshot://research/{stock_code}/b",
                document_type=source_type,
                published_at=now - timedelta(days=5),
                raw_text=f"{company_name}（{stock_code}）示例研报。评级：中性，目标价：36.00元。预计2027年营业收入1080亿元，归母净利润70亿元，EPS 1.72元。谨慎观点：产品价格承压，资本开支可能影响现金流。风险提示：市场需求下滑、海外政策变化。本材料为合成数据。",
            ),
        ]
    if source_type == "NEWS":
        return [CollectedItem(title=f"{prefix}：{company_name}发布经营进展", source_name="课程演示财经新闻", source_url=f"snapshot://news/{stock_code}", document_type=source_type, published_at=now - timedelta(hours=6), raw_text=f"课程演示新闻：{company_name}发布阶段性经营进展。市场关注订单、成本和现金流变化。该内容为合成文本，不代表真实公司事件。")]
    if source_type == "ANNOUNCEMENT":
        return [CollectedItem(title=f"{prefix}：{company_name}风险提示公告", source_name="课程演示公告源", source_url=f"snapshot://announcement/{stock_code}", document_type=source_type, published_at=now - timedelta(days=1), raw_text=f"课程演示公告：{company_name}提示投资者关注行业竞争、原材料价格和宏观环境变化。本公告为合成数据，仅验证系统解析、抽取和引用功能。")]
    return [
        CollectedItem(title=f"{prefix}：讨论{index + 1}", source_name="课程演示舆情", source_url=f"snapshot://social/{stock_code}/{index + 1}", document_type="SOCIAL", published_at=now - timedelta(minutes=index * 8), author=f"演示用户{index + 1}", raw_text=text)
        for index, text in enumerate([
            f"看好{company_name}长期产品竞争力，但还需要观察下一季度销量。",
            f"担心{company_name}所在行业价格竞争加剧，短期保持谨慎。",
            f"暂时中性，等待{company_name}正式公告更多经营数据。",
        ])
    ]

