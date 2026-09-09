from __future__ import annotations

from datetime import datetime, timedelta
import re
from urllib.parse import parse_qs, urlparse

import httpx

from app.collector.adapters.base import CollectedItem, SnapshotCapableAdapter, parse_datetime


def cninfo_pdf_url(url: str, published_at: datetime | None = None) -> str:
    """Resolve an AKShare CNInfo detail page to its canonical announcement PDF."""
    if not url:
        raise ValueError("巨潮公告链接为空")
    parsed = urlparse(url)
    if parsed.path.lower().endswith(".pdf"):
        return url.replace("http://", "https://", 1)
    query = parse_qs(parsed.query)
    announcement_id = (query.get("announcementId") or query.get("announcementid") or [""])[0]
    raw_date = (query.get("announcementTime") or query.get("announcementtime") or [""])[0]
    date_match = re.search(r"20\d{2}[-/]\d{2}[-/]\d{2}", raw_date)
    date_value = date_match.group(0).replace("/", "-") if date_match else (
        published_at.strftime("%Y-%m-%d") if published_at else ""
    )
    if not announcement_id or not date_value:
        raise ValueError("无法从巨潮详情链接解析公告编号和日期")
    return f"https://static.cninfo.com.cn/finalpage/{date_value}/{announcement_id}.PDF"


def download_cninfo_pdf(url: str, published_at: datetime | None = None) -> tuple[bytes, str]:
    pdf_url = cninfo_pdf_url(url, published_at)
    response = httpx.get(
        pdf_url,
        timeout=30,
        follow_redirects=True,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://www.cninfo.com.cn/",
            "Accept": "application/pdf,*/*",
        },
    )
    response.raise_for_status()
    if not response.content.startswith(b"%PDF"):
        content_type = response.headers.get("content-type", "未知")
        raise ValueError(f"巨潮返回的不是PDF（Content-Type: {content_type}）")
    return response.content, str(response.url)


class AnnouncementAdapter(SnapshotCapableAdapter):
    source_type = "ANNOUNCEMENT"

    def collect_live(self, stock_code: str, company_name: str, limit: int) -> list[CollectedItem]:
        import akshare as ak

        end = datetime.now()
        start = end - timedelta(days=90)
        frame = ak.stock_zh_a_disclosure_report_cninfo(symbol=stock_code, market="沪深京", category="", start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"))
        items: list[CollectedItem] = []
        for _, row in frame.head(limit).iterrows():
            detail_url = str(row.get("公告链接") or "")
            published_at = parse_datetime(row.get("公告时间"))
            try:
                raw_bytes, pdf_url = download_cninfo_pdf(detail_url, published_at)
            except (httpx.HTTPError, ValueError):
                # Never ingest the JavaScript detail-page shell as announcement
                # text. A partial run is preferable to contaminating RAG.
                continue
            items.append(CollectedItem(
                title=str(row.get("公告标题") or f"{company_name}公告"),
                source_name="巨潮资讯",
                source_url=pdf_url,
                document_type=self.source_type,
                published_at=published_at,
                raw_text=None,
                raw_bytes=raw_bytes,
                extension="pdf",
                metadata={"detail_url": detail_url},
            ))
        if frame is not None and not frame.empty and not items:
            raise RuntimeError("巨潮公告列表存在，但没有下载到有效PDF")
        return items
