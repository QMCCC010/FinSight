from __future__ import annotations

from app.collector.adapters.base import CollectedItem, SnapshotCapableAdapter, parse_datetime


class NewsAdapter(SnapshotCapableAdapter):
    source_type = "NEWS"

    def collect_live(self, stock_code: str, company_name: str, limit: int) -> list[CollectedItem]:
        import akshare as ak

        frame = ak.stock_news_em(symbol=stock_code)
        items: list[CollectedItem] = []
        for _, row in frame.head(limit).iterrows():
            items.append(CollectedItem(
                title=str(row.get("新闻标题") or f"{company_name}相关新闻"),
                source_name=str(row.get("文章来源") or "东方财富新闻"),
                source_url=str(row.get("新闻链接") or ""),
                document_type=self.source_type,
                published_at=parse_datetime(row.get("发布时间")),
                raw_text=str(row.get("新闻内容") or row.get("新闻标题") or ""),
                extension="txt",
            ))
        return items

