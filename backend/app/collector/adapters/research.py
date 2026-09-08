from __future__ import annotations

import httpx

from app.collector.adapters.base import CollectedItem, SnapshotCapableAdapter, parse_datetime


class ResearchReportAdapter(SnapshotCapableAdapter):
    source_type = "RESEARCH_REPORT"

    def collect_live(self, stock_code: str, company_name: str, limit: int) -> list[CollectedItem]:
        import akshare as ak

        frame = ak.stock_research_report_em(symbol=stock_code)
        items: list[CollectedItem] = []
        for _, row in frame.head(limit).iterrows():
            url = str(row.get("报告PDF链接") or "")
            if not url.startswith("http"):
                continue
            response = httpx.get(url, timeout=30, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            items.append(CollectedItem(
                title=str(row.get("报告名称") or f"{company_name}研究报告"),
                source_name=str(row.get("机构") or "东方财富研报"),
                source_url=url,
                document_type=self.source_type,
                published_at=parse_datetime(row.get("日期")),
                raw_bytes=response.content,
                extension="pdf",
                metadata={"rating": str(row.get("东财评级") or ""), "industry": str(row.get("行业") or "")},
            ))
        return items

