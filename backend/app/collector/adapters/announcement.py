from __future__ import annotations

from datetime import datetime, timedelta

import httpx

from app.collector.adapters.base import CollectedItem, SnapshotCapableAdapter, parse_datetime
from app.services.parser import parse_html


class AnnouncementAdapter(SnapshotCapableAdapter):
    source_type = "ANNOUNCEMENT"

    def collect_live(self, stock_code: str, company_name: str, limit: int) -> list[CollectedItem]:
        import akshare as ak

        end = datetime.now()
        start = end - timedelta(days=90)
        frame = ak.stock_zh_a_disclosure_report_cninfo(symbol=stock_code, market="沪深京", category="", start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"))
        items: list[CollectedItem] = []
        for _, row in frame.head(limit).iterrows():
            url = str(row.get("公告链接") or "")
            raw_text = str(row.get("公告标题") or "")
            raw_bytes = None
            extension = "txt"
            if url.startswith("http"):
                response = httpx.get(url, timeout=30, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0"})
                response.raise_for_status()
                if response.content.startswith(b"%PDF") or "pdf" in response.headers.get("content-type", "").lower():
                    raw_bytes, extension = response.content, "pdf"
                else:
                    raw_text, extension = parse_html(response.text), "html"
            items.append(CollectedItem(title=str(row.get("公告标题") or f"{company_name}公告"), source_name="巨潮资讯", source_url=url, document_type=self.source_type, published_at=parse_datetime(row.get("公告时间")), raw_text=raw_text, raw_bytes=raw_bytes, extension=extension))
        return items

