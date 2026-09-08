from __future__ import annotations

import httpx
from bs4 import BeautifulSoup

from app.collector.adapters.base import CollectedItem, SnapshotCapableAdapter


class SocialAdapter(SnapshotCapableAdapter):
    source_type = "SOCIAL"

    def collect_live(self, stock_code: str, company_name: str, limit: int) -> list[CollectedItem]:
        url = f"https://mguba.eastmoney.com/mguba/list/{stock_code}"
        response = httpx.get(url, timeout=20, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        candidates = soup.select("article, li, .listitem, .articleh, .item")
        items: list[CollectedItem] = []
        seen: set[str] = set()
        for node in candidates:
            text = " ".join(node.get_text(" ", strip=True).split())
            if len(text) < 12 or text in seen:
                continue
            seen.add(text)
            link = node.find("a", href=True)
            href = link.get("href") if link else url
            if href and href.startswith("/"):
                href = "https://mguba.eastmoney.com" + href
            items.append(CollectedItem(title=text[:80], source_name="东方财富股吧", source_url=href or url, document_type=self.source_type, raw_text=text, extension="txt"))
            if len(items) >= limit:
                break
        return items

