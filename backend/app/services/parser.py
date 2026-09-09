from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path
import re
import unicodedata

import pymupdf as fitz
from bs4 import BeautifulSoup


PARSER_VERSION = "document-cleaner-v3"
_PLACEHOLDER_RE = re.compile(r"\{\{.*?\}\}|\{%.*?%\}|<%.*?%>")
_PAGE_NUMBER_RE = re.compile(
    r"^(?:第\s*)?[—\-–·•\s]*\d{1,4}(?:\s*/\s*\d{1,4})?[—\-–·•\s]*(?:页)?$",
    re.IGNORECASE,
)
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ZERO_WIDTH_RE = re.compile("[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
_NOISE_LINE_RE = re.compile(
    r"^(?:巨潮资讯网|公告下载|下载中|收藏公告|已收藏|浏览|loading\.?\.?)$",
    re.IGNORECASE,
)
_STRUCTURED_LINE_RE = re.compile(
    r"^(?:#{1,6}\s+|\| |第[一二三四五六七八九十百\d]+[章节条款]"
    r"|[一二三四五六七八九十]+[、.]|\(?\d+\)?[、.])"
)


@dataclass(frozen=True)
class ParseQuality:
    score: float
    usable: bool
    warnings: list[str]
    character_count: int


def normalize_text(value: str) -> str:
    """Normalize extracted text without destroying financially meaningful signs."""
    # NFC fixes combining-character variants while preserving Chinese full-width
    # punctuation; NFKC would turn readable Chinese commas into ASCII commas.
    value = unicodedata.normalize("NFC", value or "")
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = value.replace("\u00ad", "").replace("\u00a0", " ")
    value = _ZERO_WIDTH_RE.sub("", value)
    value = _CONTROL_RE.sub("", value)
    lines: list[str] = []
    blank = False
    for raw_line in value.splitlines():
        line = re.sub(r"[ \t]+", " ", raw_line).strip()
        if not line:
            if lines and not blank:
                lines.append("")
            blank = True
            continue
        blank = False
        if _PLACEHOLDER_RE.fullmatch(line) or _NOISE_LINE_RE.fullmatch(line):
            continue
        # PDF generators occasionally emit every Chinese character separated by
        # spaces. Join only CJK-to-CJK gaps and preserve numeric/unit spacing.
        line = re.sub(r"(?<=[\u3400-\u9fff])\s+(?=[\u3400-\u9fff])", "", line)
        lines.append(line)
    joined: list[str] = []
    for line in lines:
        if not line:
            if joined and joined[-1]:
                joined.append("")
            continue
        previous = joined[-1] if joined else ""
        should_join = bool(
            previous
            and not previous.startswith("| ")
            and not line.startswith("| ")
            and not _STRUCTURED_LINE_RE.match(line)
            and not re.search(r"[。！？!?；;：:]$", previous)
            and not re.search(r"\.{4,}|…{2,}", previous)
        )
        if should_join:
            cjk_boundary = bool(
                re.search(r"[\u3400-\u9fff，。；：！？、）》】”’]$", previous)
                and re.match(r"^[\u3400-\u9fff（《【“‘]", line)
            )
            separator = "" if cjk_boundary else " "
            joined[-1] = previous + separator + line
        else:
            joined.append(line)
    return "\n".join(joined).strip()


def _line_signature(line: str) -> str:
    value = re.sub(r"\d+", "#", line.lower())
    return re.sub(r"\s+", "", value)[:160]


def _remove_repeated_margins(pages: list[tuple[int, list[str]]]) -> list[tuple[int, str]]:
    """Remove page numbers and headers/footers repeated across PDF pages."""
    if not pages:
        return []
    margin_candidates: Counter[str] = Counter()
    for _, lines in pages:
        nonempty = [line for line in lines if line]
        candidates = nonempty[:1] + nonempty[-1:]
        margin_candidates.update(set(_line_signature(line) for line in candidates))
    threshold = max(2, math.ceil(len(pages) * 0.6))
    repeated = {signature for signature, count in margin_candidates.items() if count >= threshold and len(signature) >= 4}
    cleaned_pages: list[tuple[int, str]] = []
    for page_number, lines in pages:
        nonempty_indexes = [index for index, line in enumerate(lines) if line]
        margin_indexes = set(nonempty_indexes[:1] + nonempty_indexes[-1:])
        kept: list[str] = []
        last = None
        for index, line in enumerate(lines):
            if _PAGE_NUMBER_RE.fullmatch(line):
                continue
            if index in margin_indexes and _line_signature(line) in repeated:
                continue
            if line == last:
                continue
            kept.append(line)
            last = line
        page_text = normalize_text("\n".join(kept))
        if page_text:
            cleaned_pages.append((page_number, page_text))
    return cleaned_pages


def _markdown_table(rows: list[list[str | None]]) -> str:
    normalized: list[list[str]] = []
    width = max((len(row) for row in rows), default=0)
    if width < 2 or len(rows) < 2:
        return ""
    for row in rows:
        cells = [normalize_text(str(cell or "")).replace("\n", " ").replace("|", "\\|") for cell in row]
        normalized.append((cells + [""] * width)[:width])
    if sum(bool(cell) for row in normalized for cell in row) < 4:
        return ""
    header = normalized[0]
    if not any(header):
        header = [f"列{index + 1}" for index in range(width)]
    return "\n".join([
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
        *("| " + " | ".join(row) + " |" for row in normalized[1:]),
    ])


def _overlap_ratio(left: fitz.Rect, right: fitz.Rect) -> float:
    intersection = left & right
    if intersection.is_empty or left.get_area() <= 0:
        return 0.0
    return intersection.get_area() / left.get_area()


def _page_items(page: fitz.Page) -> list[str]:
    """Extract positioned blocks and turn detected tables into Markdown."""
    table_items: list[tuple[fitz.Rect, str]] = []
    try:
        finder = page.find_tables()
        for table in finder.tables:
            markdown = _markdown_table(table.extract())
            if markdown:
                table_items.append((fitz.Rect(table.bbox), markdown))
    except (AttributeError, RuntimeError, ValueError, TypeError):
        # Some malformed PDFs cannot be analysed as tables; text extraction is
        # still useful and remains the deterministic fallback.
        table_items = []

    positioned: list[tuple[fitz.Rect, str, str]] = []
    for block in page.get_text("blocks", sort=True):
        rectangle = fitz.Rect(block[:4])
        if any(_overlap_ratio(rectangle, table_rect) >= 0.55 for table_rect, _ in table_items):
            continue
        text = normalize_text(str(block[4]))
        if text:
            positioned.append((rectangle, "text", text))
    positioned.extend((rectangle, "table", markdown) for rectangle, markdown in table_items)
    positioned.sort(key=lambda item: (round(item[0].y0 / 6), item[0].x0))

    merged: list[tuple[fitz.Rect, str, str]] = []
    for rectangle, kind, text in positioned:
        if merged:
            previous_rect, previous_kind, previous_text = merged[-1]
            line_height = max(previous_rect.height, rectangle.height, 1)
            is_wrapped_line = bool(
                kind == previous_kind == "text"
                and previous_rect.x1 >= page.rect.width * 0.76
                and rectangle.x0 <= previous_rect.x0 + 3
                and 0 <= rectangle.y0 - previous_rect.y1 <= line_height * 1.35
                and not _STRUCTURED_LINE_RE.match(text)
            )
            if is_wrapped_line:
                combined_rect = fitz.Rect(
                    min(previous_rect.x0, rectangle.x0),
                    previous_rect.y0,
                    rectangle.x1,
                    rectangle.y1,
                )
                merged[-1] = (
                    combined_rect,
                    previous_kind,
                    normalize_text(previous_text + "\n" + text),
                )
                continue
        merged.append((rectangle, kind, text))
    return [item[2] for item in merged]


def assess_text_quality(text: str) -> ParseQuality:
    normalized = normalize_text(text)
    compact = re.sub(r"\s+", "", normalized)
    warnings: list[str] = []
    score = 1.0
    placeholder_count = len(_PLACEHOLDER_RE.findall(text or ""))
    replacement_count = sum((text or "").count(mark) for mark in ("�", "□", "����"))
    readable_count = len(re.findall(r"[\u3400-\u9fffA-Za-z0-9]", compact))
    readable_ratio = readable_count / max(1, len(compact))

    if placeholder_count:
        warnings.append("检测到网页模板占位符，未获得公告正文")
        score -= 0.75
    if len(compact) < 80:
        warnings.append("可解析正文过短")
        score -= 0.55
    elif len(compact) < 300:
        warnings.append("可解析正文较短")
        score -= 0.15
    if replacement_count / max(1, len(compact)) > 0.01:
        warnings.append("特殊字体字符映射异常")
        score -= 0.35
    if compact and readable_ratio < 0.45:
        warnings.append("正文可读字符比例偏低")
        score -= 0.35
    score = round(max(0.0, min(1.0, score)), 3)
    usable = not placeholder_count and len(compact) >= 80 and readable_ratio >= 0.35 and score >= 0.35
    return ParseQuality(score=score, usable=usable, warnings=warnings, character_count=len(compact))


def parse_pdf(content: bytes) -> tuple[str, list[tuple[int, str]]]:
    if not content.startswith(b"%PDF"):
        raise ValueError("下载内容不是有效PDF")
    document = fitz.open(stream=content, filetype="pdf")
    raw_pages: list[tuple[int, list[str]]] = []
    for index, page in enumerate(document):
        items = _page_items(page)
        lines: list[str] = []
        for item in items:
            lines.extend(item.splitlines())
            lines.append("")
        raw_pages.append((index + 1, lines))
    pages = _remove_repeated_margins(raw_pages)
    joined = "\n\f\n".join(text for _, text in pages)
    return joined, pages


def parse_html(content: str | bytes) -> str:
    soup = BeautifulSoup(content, "html.parser")
    for element in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "template"]):
        element.decompose()
    candidates = soup.select("article, main, .article-content, .newsContent, .content, #ContentBody")
    root = max(candidates, key=lambda item: len(item.get_text()), default=soup)
    return normalize_text(root.get_text("\n"))


def parse_stored(path: str) -> tuple[str, list[tuple[int, str]]]:
    file_path = Path(path)
    content = file_path.read_bytes()
    if file_path.suffix.lower() == ".pdf" or content.startswith(b"%PDF"):
        return parse_pdf(content)
    text = content.decode("utf-8", errors="replace")
    if file_path.suffix.lower() in {".html", ".htm"} or b"<html" in content[:1000].lower():
        text = parse_html(text)
    else:
        text = normalize_text(text)
    return text, [(1, text)] if text else []


def split_pages(text: str) -> list[tuple[int, str]]:
    return [(index + 1, part.strip()) for index, part in enumerate(text.split("\n\f\n")) if part.strip()]


def chunk_pages(pages: list[tuple[int, str]], size: int = 700, overlap: int = 100) -> list[tuple[int, str]]:
    chunks: list[tuple[int, str]] = []
    step = max(1, size - overlap)
    for page, text in pages:
        clean = re.sub(r"[ \t]+", " ", normalize_text(text)).strip()
        # Preserve paragraph and Markdown-table boundaries for retrieval instead
        # of flattening an entire PDF page into one uninterrupted sentence.
        clean = re.sub(r"\n{3,}", "\n\n", clean)
        for start in range(0, len(clean), step):
            chunk = clean[start : start + size].strip()
            if chunk:
                chunks.append((page, chunk))
            if start + size >= len(clean):
                break
    return chunks
