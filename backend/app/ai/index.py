from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from functools import lru_cache
from pathlib import Path

import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.models import Document, DocumentChunk


logger = logging.getLogger(__name__)


@lru_cache
def _local_model():
    settings = get_settings()
    if not settings.enable_local_embeddings:
        return None
    try:
        from fastembed import TextEmbedding
        return TextEmbedding(model_name=settings.embedding_model)
    except Exception:
        # The deterministic hash vector keeps offline classroom demos usable.
        return None


def _hash_embedding(text: str) -> np.ndarray:
    vector = np.zeros(get_settings().embedding_dimension, dtype="float32")
    clean = re.sub(r"\s+", "", text)
    tokens = [clean[index : index + 2] for index in range(max(1, len(clean) - 1))]
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8", errors="ignore")).digest()
        position = int.from_bytes(digest[:4], "big") % VECTOR_DIMENSION
        vector[position] += 1.0 if digest[4] % 2 else -1.0
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def _embedding_backend() -> str:
    return "fastembed" if _local_model() is not None else "hash"


def embed_texts(texts: list[str]) -> np.ndarray:
    model = _local_model()
    if model:
        vectors = np.asarray(list(model.embed(texts)), dtype="float32")
        if vectors.ndim != 2 or vectors.shape[1] != get_settings().embedding_dimension:
            actual_dimension = vectors.shape[1] if vectors.ndim == 2 else "未知"
            raise ValueError(
                f"嵌入模型维度{actual_dimension}"
                f"与配置维度{get_settings().embedding_dimension}不一致"
            )
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1e-12)
    return np.asarray([_hash_embedding(text) for text in texts], dtype="float32")


def index_paths() -> tuple[Path, Path]:
    folder = get_settings().storage_root / "index"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / "documents.faiss", folder / "documents.mapping.json"


def _metadata_path() -> Path:
    return get_settings().storage_root / "index" / "documents.meta.json"


def _get_faiss_index_status() -> dict:
    configured_backend = "fastembed" if get_settings().enable_local_embeddings else "hash"
    index_file, mapping_file = index_paths()
    metadata_file = _metadata_path()
    status = {
        "ready": index_file.exists() and mapping_file.exists(),
        "index_path": str(index_file),
        "metadata": None,
        "compatible": False,
    }
    if not status["ready"]:
        return status
    try:
        import faiss

        index = faiss.read_index(str(index_file))
        mapping = json.loads(mapping_file.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_file.read_text(encoding="utf-8")) if metadata_file.exists() else {}
        status["metadata"] = {**metadata, "actual_dimension": index.d, "actual_count": index.ntotal}
        status["compatible"] = (
            index.ntotal == len(mapping)
            and (not metadata or metadata.get("dimension") == index.d)
            and (not metadata.get("embedding_backend") or metadata.get("embedding_backend") == configured_backend)
            and (metadata.get("embedding_backend") != "fastembed" or metadata.get("embedding_model") == get_settings().embedding_model)
        )
    except Exception as exc:
        status["error"] = str(exc)
    return status


def _rebuild_faiss_index(db: Session) -> int:
    import faiss

    chunks = list(db.scalars(select(DocumentChunk).join(Document, Document.id == DocumentChunk.document_id).where(Document.is_deleted.is_(False)).order_by(DocumentChunk.id)).all())
    index_file, mapping_file = index_paths()
    metadata_file = _metadata_path()
    if not chunks:
        for path in (index_file, mapping_file, metadata_file):
            if path.exists():
                path.unlink()
        return 0
    vectors = embed_texts([chunk.content for chunk in chunks])
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    temp_index = index_file.with_suffix(".tmp.faiss")
    temp_mapping = mapping_file.with_suffix(".tmp.json")
    temp_metadata = metadata_file.with_suffix(".tmp.json")
    faiss.write_index(index, str(temp_index))
    temp_mapping.write_text(json.dumps([chunk.id for chunk in chunks]), encoding="utf-8")
    temp_metadata.write_text(
        json.dumps(
            {
                "version": 1,
                "dimension": int(vectors.shape[1]),
                "count": len(chunks),
                "embedding_model": get_settings().embedding_model,
                "local_embeddings_enabled": get_settings().enable_local_embeddings,
                "embedding_backend": _embedding_backend(),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    os.replace(temp_index, index_file)
    os.replace(temp_mapping, mapping_file)
    os.replace(temp_metadata, metadata_file)
    for vector_id, chunk in enumerate(chunks):
        chunk.vector_id = vector_id
    db.commit()
    return len(chunks)


def _vector_candidates(query: str, limit: int = 100) -> list[int]:
    import faiss

    index_file, mapping_file = index_paths()
    if not index_file.exists() or not mapping_file.exists():
        return []
    try:
        index = faiss.read_index(str(index_file))
        mapping = json.loads(mapping_file.read_text(encoding="utf-8"))
        metadata_file = _metadata_path()
        metadata = json.loads(metadata_file.read_text(encoding="utf-8")) if metadata_file.exists() else {}
        if metadata.get("embedding_backend") and metadata["embedding_backend"] != _embedding_backend():
            return []
        if metadata.get("embedding_backend") == "fastembed" and metadata.get("embedding_model") != get_settings().embedding_model:
            return []
        count = min(limit, len(mapping), index.ntotal)
        if count == 0:
            return []
        query_vector = embed_texts([query])
        # An embedding configuration may change while an old index is still on
        # disk. Never let FAISS assert and take down the request; BM25 remains a
        # safe fallback until the administrator rebuilds the index.
        if query_vector.ndim != 2 or query_vector.shape[1] != index.d:
            return []
        _, indices = index.search(query_vector, count)
        return [mapping[index] for index in indices[0] if 0 <= index < len(mapping)]
    except (OSError, ValueError, TypeError, json.JSONDecodeError, RuntimeError):
        return []


def _faiss_hybrid_search(
    db: Session,
    query: str,
    company_id: int | None,
    limit: int = 6,
    document_types: list[str] | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    company_ids: list[int] | None = None,
) -> list[dict]:
    chunk_query = select(DocumentChunk).join(Document, Document.id == DocumentChunk.document_id).where(
        Document.status == "INDEXED",
        Document.is_deleted.is_(False),
    )
    if company_id is not None:
        chunk_query = chunk_query.where(DocumentChunk.company_id == company_id)
    elif company_ids:
        chunk_query = chunk_query.where(DocumentChunk.company_id.in_(company_ids))
    if document_types:
        chunk_query = chunk_query.where(Document.document_type.in_(document_types))
    if date_from:
        chunk_query = chunk_query.where(Document.published_at >= date_from)
    if date_to:
        chunk_query = chunk_query.where(Document.published_at <= date_to)
    chunks = list(db.scalars(chunk_query).all())
    if not chunks:
        return []
    by_id = {chunk.id: chunk for chunk in chunks}
    vector_ids = [chunk_id for chunk_id in _vector_candidates(query) if chunk_id in by_id]
    corpus = [list(jieba.cut(chunk.content)) for chunk in chunks]
    bm25 = BM25Okapi(corpus)
    keyword_scores = bm25.get_scores(list(jieba.cut(query)))
    keyword_order = [chunks[index].id for index in np.argsort(keyword_scores)[::-1][: min(30, len(chunks))]]
    scores: dict[int, float] = {}
    for ranking in (vector_ids, keyword_order):
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (60 + rank + 1)
    source_weight = {"ANNOUNCEMENT": 1.20, "RESEARCH_REPORT": 1.12, "NEWS": 1.0, "SOCIAL": 0.86}
    ranked = []
    for chunk_id, score in scores.items():
        chunk = by_id[chunk_id]
        document = db.get(Document, chunk.document_id)
        score *= source_weight.get(document.document_type, 1.0)
        if document.published_at:
            age_days = max(0, (datetime.now() - document.published_at).days)
            score *= 1.0 + 0.12 / (1.0 + age_days / 30)
        ranked.append((chunk_id, score, document))
    result = []
    used_documents: set[int] = set()
    for chunk_id, score, document in sorted(ranked, key=lambda item: item[1], reverse=True):
        if document.id in used_documents and len(result) >= max(2, limit // 2):
            continue
        chunk = by_id[chunk_id]
        result.append({"chunk_id": chunk.id, "document_id": document.id, "title": document.title, "source_type": document.document_type, "source_url": document.source_url, "published_at": document.published_at.isoformat() if document.published_at else None, "page": chunk.page_number, "quote": chunk.content, "score": score})
        used_documents.add(document.id)
        if len(result) >= limit:
            break
    return result


def get_index_status() -> dict:
    settings = get_settings()
    if settings.vector_store_backend == "milvus":
        from app.ai.milvus_store import status

        result = status()
        if settings.vector_store_fallback:
            result["fallback"] = _get_faiss_index_status()
        return result
    result = _get_faiss_index_status()
    result["backend"] = "faiss"
    return result


def rebuild_index(db: Session) -> int:
    if get_settings().vector_store_backend == "milvus":
        from app.ai.milvus_store import rebuild

        return rebuild(db)
    return _rebuild_faiss_index(db)


def index_document(db: Session, document_id: int) -> int:
    if get_settings().vector_store_backend != "milvus":
        return 0
    from app.ai.milvus_store import upsert_document

    return upsert_document(db, document_id)


def remove_document_from_index(document_id: int) -> None:
    if get_settings().vector_store_backend == "milvus":
        from app.ai.milvus_store import delete_document

        delete_document(document_id)


def hybrid_search(
    db: Session,
    query: str,
    company_id: int | None,
    limit: int = 6,
    document_types: list[str] | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    company_ids: list[int] | None = None,
) -> list[dict]:
    settings = get_settings()
    if settings.vector_store_backend == "milvus":
        try:
            from app.ai.milvus_store import hybrid_search as milvus_hybrid_search

            return milvus_hybrid_search(
                query, company_id, limit, document_types, date_from, date_to, company_ids
            )
        except Exception as exc:
            if not settings.vector_store_fallback:
                raise
            logger.warning("Milvus retrieval failed; falling back to local FAISS/BM25: %s", exc)
    return _faiss_hybrid_search(
        db, query, company_id, limit, document_types, date_from, date_to, company_ids
    )
