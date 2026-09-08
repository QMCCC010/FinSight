from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Protocol

from app.core.config import get_settings
from app.services.storage import load_snapshot, save_snapshot


@dataclass
class CollectedItem:
    title: str
    source_name: str
    source_url: str
    document_type: str
    published_at: datetime | None = None
    author: str | None = None
    raw_text: str | None = None
    raw_bytes: bytes | None = None
    extension: str = "txt"
    metadata: dict | None = None

    def snapshot_dict(self) -> dict:
        value = asdict(self)
        value.pop("raw_bytes", None)
        if self.published_at:
            value["published_at"] = self.published_at.isoformat()
        return value

    @classmethod
    def from_snapshot(cls, value: dict) -> "CollectedItem":
        payload = dict(value)
        if payload.get("published_at"):
            payload["published_at"] = datetime.fromisoformat(payload["published_at"])
        return cls(**payload)


class SourceAdapter(Protocol):
    source_type: str

    def collect_live(self, stock_code: str, company_name: str, limit: int) -> list[CollectedItem]: ...


class SnapshotCapableAdapter:
    source_type = ""

    def collect(self, stock_code: str, company_name: str, limit: int) -> list[CollectedItem]:
        mode = get_settings().collection_mode
        if mode == "SNAPSHOT":
            return self.load_fallback(stock_code, company_name)[:limit]
        try:
            items = self.collect_live(stock_code, company_name, limit)
            if items:
                save_snapshot(stock_code, self.source_type, [item.snapshot_dict() for item in items])
                return items[:limit]
            raise RuntimeError("在线数据源返回空结果")
        except Exception:
            if mode != "AUTO":
                raise
            return self.load_fallback(stock_code, company_name)[:limit]

    def load_fallback(self, stock_code: str, company_name: str) -> list[CollectedItem]:
        stored = load_snapshot(stock_code, self.source_type)
        if stored:
            return [CollectedItem.from_snapshot(item) for item in stored]
        from app.collector.demo_data import demo_items
        return demo_items(stock_code, company_name, self.source_type)


def parse_datetime(value: object) -> datetime | None:
    if value is None or str(value) in {"", "NaT", "nan", "None"}:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("/", "-").replace("Z", "+00:00"))
        return dt.replace(tzinfo=None)
    except ValueError:
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y%m%d"):
            try:
                return datetime.strptime(str(value), fmt)
            except ValueError:
                continue
    return None

