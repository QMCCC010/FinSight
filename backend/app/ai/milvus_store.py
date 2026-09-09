from __future__ import annotations

from datetime import datetime
from functools import lru_cache
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import SessionLocal
from app.core.models import Company, Document, DocumentChunk


def _client_kwargs() -> dict[str, Any]:
    settings = get_settings()
    kwargs: dict[str, Any] = {"uri": settings.milvus_uri}
    if settings.milvus_token:
        kwargs["token"] = settings.milvus_token
    return kwargs


@lru_cache
def get_client():
    from pymilvus import MilvusClient

    return MilvusClient(**_client_kwargs())


def reset_client() -> None:
    client = get_client.cache_info()
    if client.currsize:
        try:
            get_client().close()
        except Exception:
            pass
    get_client.cache_clear()


def _collection_name() -> str:
    return get_settings().milvus_collection_name


def _embedding_dimension() -> int:
    return get_settings().embedding_dimension


def ensure_collection(*, recreate: bool = False) -> None:
    from pymilvus import DataType, Function, FunctionType

    client = get_client()
    name = _collection_name()
    if recreate and client.has_collection(name):
        client.drop_collection(name)
    if client.has_collection(name):
        client.load_collection(name)
        return

    dimension = _embedding_dimension()
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("chunk_id", DataType.INT64, is_primary=True)
    schema.add_field("document_id", DataType.INT64)
    schema.add_field("company_id", DataType.INT64)
    schema.add_field("stock_code", DataType.VARCHAR, max_length=32)
    schema.add_field("document_type", DataType.VARCHAR, max_length=64)
    schema.add_field("published_ts", DataType.INT64)
    schema.add_field("page_number", DataType.INT64)
    schema.add_field("title", DataType.VARCHAR, max_length=1024)
    schema.add_field("source_url", DataType.VARCHAR, max_length=4096)
    schema.add_field(
        "content",
        DataType.VARCHAR,
        max_length=8192,
        enable_analyzer=True,
        analyzer_params={"type": "chinese"},
    )
    schema.add_field("is_deleted", DataType.BOOL)
    schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=dimension)
    schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_function(
        Function(
            name="content_bm25",
            input_field_names=["content"],
            output_field_names=["sparse_vector"],
            function_type=FunctionType.BM25,
        )
    )

    indexes = client.prepare_index_params()
    indexes.add_index(
        field_name="dense_vector",
        index_type="AUTOINDEX",
        metric_type="COSINE",
    )
    indexes.add_index(
        field_name="sparse_vector",
        index_type="SPARSE_INVERTED_INDEX",
        metric_type="BM25",
        params={"inverted_index_algo": "DAAT_MAXSCORE", "bm25_k1": 1.2, "bm25_b": 0.75},
    )
    try:
        client.create_collection(
            collection_name=name,
            schema=schema,
            index_params=indexes,
            consistency_level="Strong",
        )
    except Exception:
        # API and workers can race during first startup. A collection created by
        # the other process is a successful outcome; other errors still surface.
        if not client.has_collection(name):
            raise


def _row(
    chunk: DocumentChunk, document: Document, stock_code: str, vector: list[float]
) -> dict[str, Any]:
    return {
        "chunk_id": chunk.id,
        "document_id": document.id,
        "company_id": document.company_id,
        "stock_code": stock_code,
        "document_type": document.document_type,
        "published_ts": int(document.published_at.timestamp()) if document.published_at else 0,
        "page_number": chunk.page_number or 0,
        "title": document.title[:1024],
        "source_url": document.source_url[:4096],
        "content": chunk.content[:8192],
        "is_deleted": bool(document.is_deleted),
        "dense_vector": vector,
    }


def upsert_document(db: Session, document_id: int) -> int:
    from app.ai.index import embed_texts

    ensure_collection()
    document = db.get(Document, document_id)
    if not document:
        return 0
    chunks = list(
        db.scalars(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index)
        ).all()
    )
    client = get_client()
    client.delete(collection_name=_collection_name(), filter=f"document_id == {int(document_id)}")
    if document.is_deleted or not chunks:
        client.flush(_collection_name())
        return 0
    company = db.get(Company, document.company_id)
    stock_code = company.stock_code if company else ""
    vectors = embed_texts([chunk.content for chunk in chunks])
    rows = [
        _row(chunk, document, stock_code, vectors[index].tolist())
        for index, chunk in enumerate(chunks)
    ]
    for start in range(0, len(rows), 128):
        client.upsert(collection_name=_collection_name(), data=rows[start : start + 128])
    client.flush(_collection_name())
    return len(chunks)


def delete_document(document_id: int) -> None:
    ensure_collection()
    client = get_client()
    client.delete(collection_name=_collection_name(), filter=f"document_id == {int(document_id)}")
    client.flush(_collection_name())


def rebuild(db: Session) -> int:
    ensure_collection(recreate=True)
    pairs = list(
        db.execute(
            select(DocumentChunk, Document)
            .join(Document, Document.id == DocumentChunk.document_id)
            .where(Document.is_deleted.is_(False))
            .order_by(DocumentChunk.id)
        ).all()
    )
    if not pairs:
        return 0
    from app.ai.index import embed_texts

    client = get_client()
    company_ids = {document.company_id for _, document in pairs}
    company_codes = {
        company.id: company.stock_code
        for company in db.scalars(select(Company).where(Company.id.in_(company_ids))).all()
    }
    total = 0
    for start in range(0, len(pairs), 128):
        batch = pairs[start : start + 128]
        vectors = embed_texts([chunk.content for chunk, _ in batch])
        rows = [
            _row(
                chunk, document, company_codes.get(document.company_id, ""),
                vectors[index].tolist(),
            )
            for index, (chunk, document) in enumerate(batch)
        ]
        client.insert(collection_name=_collection_name(), data=rows)
        total += len(batch)
    client.flush(_collection_name())
    return total


def _escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _filter_expression(
    company_id: int | None,
    document_types: list[str] | None,
    date_from: datetime | None,
    date_to: datetime | None,
    company_ids: list[int] | None = None,
) -> str:
    conditions = ["is_deleted == false"]
    if company_id is not None:
        conditions.append(f"company_id == {int(company_id)}")
    elif company_ids:
        values = ", ".join(str(int(item)) for item in company_ids)
        conditions.append(f"company_id in [{values}]")
    if document_types:
        values = ", ".join(f'"{_escape(item)}"' for item in document_types)
        conditions.append(f"document_type in [{values}]")
    if date_from:
        conditions.append(f"published_ts >= {int(date_from.timestamp())}")
    if date_to:
        conditions.append(f"published_ts <= {int(date_to.timestamp())}")
    return " and ".join(conditions)


def _hit_value(hit: Any, key: str, default: Any = None) -> Any:
    if isinstance(hit, dict):
        entity = hit.get("entity") or {}
        return entity.get(key, hit.get(key, default))
    entity = getattr(hit, "entity", None)
    if entity is not None:
        try:
            return entity.get(key, default)
        except AttributeError:
            pass
    return getattr(hit, key, default)


def hybrid_search(
    query: str,
    company_id: int | None,
    limit: int,
    document_types: list[str] | None,
    date_from: datetime | None,
    date_to: datetime | None,
    company_ids: list[int] | None = None,
) -> list[dict]:
    from pymilvus import AnnSearchRequest, RRFRanker
    from app.ai.index import embed_texts

    ensure_collection()
    expression = _filter_expression(company_id, document_types, date_from, date_to, company_ids)
    candidate_limit = max(20, limit * 4)
    vector = embed_texts([query])[0].tolist()
    requests = [
        AnnSearchRequest(
            data=[vector],
            anns_field="dense_vector",
            param={"metric_type": "COSINE", "params": {}},
            limit=candidate_limit,
            expr=expression,
        ),
        AnnSearchRequest(
            data=[query],
            anns_field="sparse_vector",
            param={"metric_type": "BM25", "params": {}},
            limit=candidate_limit,
            expr=expression,
        ),
    ]
    output_fields = [
        "chunk_id", "document_id", "document_type", "published_ts",
        "page_number", "title", "source_url", "content",
    ]
    response = get_client().hybrid_search(
        collection_name=_collection_name(),
        reqs=requests,
        ranker=RRFRanker(),
        limit=candidate_limit,
        output_fields=output_fields,
    )
    hits = response[0] if response else []
    # Reliability-aware reranking: social posts remain searchable evidence, but
    # cannot crowd out primary announcements and professional research merely
    # because they repeat query keywords more often.
    source_weight = {
        "ANNOUNCEMENT": 1.35,
        "RESEARCH_REPORT": 1.30,
        "NEWS": 1.0,
        "SOCIAL": 0.60,
    }
    ranked: list[tuple[float, dict]] = []
    for hit in hits:
        published_ts = int(_hit_value(hit, "published_ts", 0) or 0)
        published_at = datetime.fromtimestamp(published_ts) if published_ts else None
        source_type = str(_hit_value(hit, "document_type", ""))
        score = float(_hit_value(hit, "distance", 0.0) or 0.0) * source_weight.get(source_type, 1.0)
        if published_at:
            age_days = max(0, (datetime.now() - published_at).days)
            score *= 1.0 + 0.12 / (1.0 + age_days / 30)
        ranked.append((score, {
            "chunk_id": int(_hit_value(hit, "chunk_id", _hit_value(hit, "id", 0))),
            "document_id": int(_hit_value(hit, "document_id", 0)),
            "title": str(_hit_value(hit, "title", "")),
            "source_type": source_type,
            "source_url": str(_hit_value(hit, "source_url", "")),
            "published_at": published_at.isoformat() if published_at else None,
            "page": int(_hit_value(hit, "page_number", 0) or 0) or None,
            "quote": str(_hit_value(hit, "content", "")),
            "score": score,
        }))

    result: list[dict] = []
    used_documents: set[int] = set()
    for _, item in sorted(ranked, key=lambda pair: pair[0], reverse=True):
        if item["document_id"] in used_documents and len(result) >= max(2, limit // 2):
            continue
        result.append(item)
        used_documents.add(item["document_id"])
        if len(result) >= limit:
            break
    return result


def status() -> dict:
    settings = get_settings()
    result = {
        "backend": "milvus",
        "ready": False,
        "compatible": False,
        "collection": _collection_name(),
        "uri": settings.milvus_uri,
        "metadata": {
            "embedding_model": settings.embedding_model,
            "embedding_backend": "fastembed-configured" if settings.enable_local_embeddings else "hash",
        },
    }
    try:
        client = get_client()
        if not client.has_collection(_collection_name()):
            return result
        description = client.describe_collection(_collection_name())
        fields = description.get("fields", []) if isinstance(description, dict) else []
        dense = next((field for field in fields if field.get("name") == "dense_vector"), {})
        params = dense.get("params", {}) or {}
        dimension = int(params.get("dim", 0) or 0)
        stats = client.get_collection_stats(_collection_name())
        count = int(stats.get("row_count", 0) or 0)
        expected_dimension = _embedding_dimension()
        with SessionLocal() as db:
            database_count = db.scalar(
                select(func.count(DocumentChunk.id))
                .join(Document, Document.id == DocumentChunk.document_id)
                .where(Document.is_deleted.is_(False))
            ) or 0
        result["ready"] = True
        result["compatible"] = dimension == expected_dimension
        result["synchronized"] = count == database_count
        result["metadata"].update({
            "actual_dimension": dimension,
            "dimension": expected_dimension,
            "actual_count": count,
            "count": count,
            "database_count": database_count,
        })
    except Exception as exc:
        result["error"] = str(exc)
        reset_client()
    return result
