from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from app.core.config import get_settings


def safe_name(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z._-]+", "-", value).strip("-")[:80] or "document"


def content_hash(content: bytes | str) -> str:
    raw = content.encode("utf-8", errors="ignore") if isinstance(content, str) else content
    return hashlib.sha256(raw).hexdigest()


def save_raw(stock_code: str, source_type: str, title: str, content: bytes | str, extension: str) -> str:
    settings = get_settings()
    raw = content.encode("utf-8", errors="ignore") if isinstance(content, str) else content
    digest = content_hash(raw)
    folder = settings.storage_root / "raw" / stock_code / source_type.lower()
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{digest[:16]}-{safe_name(title)}.{extension.lstrip('.')}"
    if not path.exists():
        path.write_bytes(raw)
    return str(path)


def save_snapshot(stock_code: str, source_type: str, payload: list[dict]) -> None:
    settings = get_settings()
    folder = settings.storage_root / "snapshots" / stock_code
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{source_type.lower()}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, default=str, indent=2), encoding="utf-8")


def load_snapshot(stock_code: str, source_type: str) -> list[dict]:
    path = get_settings().storage_root / "snapshots" / stock_code / f"{source_type.lower()}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))

