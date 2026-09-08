from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol


ALL_SOURCE_TYPES = ("RESEARCH_REPORT", "NEWS", "ANNOUNCEMENT", "SOCIAL")
VALID_SOURCE_TYPES = set(ALL_SOURCE_TYPES)


class CrawlRunLike(Protocol):
    source_type: str | None
    source_types: list[str] | None


def normalize_source_types(values: Iterable[str] | None) -> list[str]:
    if not values:
        return []
    requested = {str(value).strip().upper() for value in values if str(value).strip()}
    return [source_type for source_type in ALL_SOURCE_TYPES if source_type in requested]


def run_source_types(run: CrawlRunLike) -> list[str]:
    """Read the new JSON list and transparently support pre-migration rows."""
    current = normalize_source_types(getattr(run, "source_types", None))
    if current:
        return current
    legacy = getattr(run, "source_type", None)
    return normalize_source_types(legacy.split(",") if legacy else None)


def run_covers(run: CrawlRunLike, requested: Iterable[str]) -> bool:
    stored = run_source_types(run)
    # Historical NULL/empty rows represented an unrestricted all-source run.
    return not stored or set(normalize_source_types(requested)).issubset(stored)
