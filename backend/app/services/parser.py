from __future__ import annotations

from pathlib import Path

import fitz
from bs4 import BeautifulSoup


def parse_pdf(content: bytes) -> tuple[str, list[tuple[int, str]]]:
    document = fitz.open(stream=content, filetype="pdf")
    pages: list[tuple[int, str]] = []
    for index, page in enumerate(document):
        text = page.get_text("text").strip()
        if text:
            pages.append((index + 1, text))
    joined = "\n\f\n".join(text for _, text in pages)
    return joined, pages


def parse_html(content: str | bytes) -> str:
    soup = BeautifulSoup(content, "html.parser")
    for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
        element.decompose()
    candidates = soup.select("article, main, .article-content, .newsContent, .content, #ContentBody")
    root = max(candidates, key=lambda item: len(item.get_text()), default=soup)
    lines = [" ".join(line.split()) for line in root.get_text("\n").splitlines()]
    return "\n".join(line for line in lines if len(line) > 1)


def parse_stored(path: str) -> tuple[str, list[tuple[int, str]]]:
    file_path = Path(path)
    content = file_path.read_bytes()
    if file_path.suffix.lower() == ".pdf":
        return parse_pdf(content)
    text = content.decode("utf-8", errors="ignore")
    if file_path.suffix.lower() in {".html", ".htm"}:
        text = parse_html(text)
    return text, [(1, text)]


def split_pages(text: str) -> list[tuple[int, str]]:
    return [(index + 1, part.strip()) for index, part in enumerate(text.split("\n\f\n")) if part.strip()]


def chunk_pages(pages: list[tuple[int, str]], size: int = 700, overlap: int = 100) -> list[tuple[int, str]]:
    chunks: list[tuple[int, str]] = []
    step = max(1, size - overlap)
    for page, text in pages:
        clean = " ".join(text.split())
        for start in range(0, len(clean), step):
            chunk = clean[start : start + size]
            if chunk:
                chunks.append((page, chunk))
            if start + size >= len(clean):
                break
    return chunks

