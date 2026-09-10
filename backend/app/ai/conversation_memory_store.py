from __future__ import annotations

from datetime import datetime
from typing import Any

from app.ai.index import embed_texts
from app.ai.milvus_store import get_client
from app.core.config import get_settings
from app.core.models import ChatMessage


def _collection_name() -> str:
    return get_settings().milvus_memory_collection_name


def ensure_memory_collection(*, recreate: bool = False) -> None:
    """Create a collection dedicated to conversation turns.

    Financial evidence and private conversation memory intentionally live in
    different collections. Every search below also applies user and session
    filters, so a semantically similar turn can never cross a conversation.
    """
    from pymilvus import DataType, Function, FunctionType

    client = get_client()
    name = _collection_name()
    if recreate and client.has_collection(name):
        client.drop_collection(name)
    if client.has_collection(name):
        client.load_collection(name)
        return

    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("message_id", DataType.INT64, is_primary=True)
    schema.add_field("user_id", DataType.INT64)
    schema.add_field("session_id", DataType.INT64)
    schema.add_field("company_id", DataType.INT64)
    schema.add_field("created_ts", DataType.INT64)
    schema.add_field(
        "content",
        DataType.VARCHAR,
        max_length=8192,
        enable_analyzer=True,
        analyzer_params={"type": "chinese"},
    )
    schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=get_settings().embedding_dimension)
    schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_function(
        Function(
            name="conversation_bm25",
            input_field_names=["content"],
            output_field_names=["sparse_vector"],
            function_type=FunctionType.BM25,
        )
    )

    indexes = client.prepare_index_params()
    indexes.add_index(field_name="dense_vector", index_type="AUTOINDEX", metric_type="COSINE")
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
        if not client.has_collection(name):
            raise


def _turn_text(message: ChatMessage) -> str:
    return _truncate_utf8(f"用户：{message.question or ''}\n助手：{message.answer or ''}", 8192)


def _truncate_utf8(value: str, max_bytes: int) -> str:
    """Respect Milvus VARCHAR's byte limit without splitting a code point."""
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    return encoded[:max_bytes].decode("utf-8", errors="ignore")


def upsert_turn(message: ChatMessage) -> None:
    if not message.question or not message.answer:
        return
    ensure_memory_collection()
    content = _turn_text(message)
    created_at = message.created_at or datetime.now()
    row = {
        "message_id": int(message.id),
        "user_id": int(message.user_id),
        "session_id": int(message.session_id),
        "company_id": int(message.company_id or 0),
        "created_ts": int(created_at.timestamp()),
        "content": content,
        "dense_vector": embed_texts([content])[0].tolist(),
    }
    client = get_client()
    client.upsert(collection_name=_collection_name(), data=[row])


def upsert_turns(messages: list[ChatMessage], *, batch_size: int = 64) -> int:
    """Bulk backfill completed turns with one flush instead of one per row."""
    eligible = [message for message in messages if message.question and message.answer]
    if not eligible:
        return 0
    ensure_memory_collection()
    client = get_client()
    for start in range(0, len(eligible), batch_size):
        batch = eligible[start:start + batch_size]
        contents = [_turn_text(message) for message in batch]
        vectors = embed_texts(contents)
        rows = []
        for index, message in enumerate(batch):
            created_at = message.created_at or datetime.now()
            rows.append({
                "message_id": int(message.id),
                "user_id": int(message.user_id),
                "session_id": int(message.session_id),
                "company_id": int(message.company_id or 0),
                "created_ts": int(created_at.timestamp()),
                "content": contents[index],
                "dense_vector": vectors[index].tolist(),
            })
        client.upsert(collection_name=_collection_name(), data=rows)
    client.flush(_collection_name())
    return len(eligible)


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


def _memory_filter(user_id: int, session_id: int, exclude_message_ids: list[int] | None = None) -> str:
    conditions = [f"user_id == {int(user_id)}", f"session_id == {int(session_id)}"]
    if exclude_message_ids:
        values = ", ".join(str(int(item)) for item in exclude_message_ids)
        conditions.append(f"message_id not in [{values}]")
    return " and ".join(conditions)


def search_turns(
    query: str,
    *,
    user_id: int,
    session_id: int,
    limit: int = 3,
    exclude_message_ids: list[int] | None = None,
) -> list[dict[str, Any]]:
    from pymilvus import AnnSearchRequest, RRFRanker

    if not query.strip() or limit <= 0:
        return []
    ensure_memory_collection()
    expression = _memory_filter(user_id, session_id, exclude_message_ids)
    candidate_limit = max(10, limit * 3)
    vector = embed_texts([query])[0].tolist()
    requests = [
        AnnSearchRequest(
            data=[vector], anns_field="dense_vector",
            param={"metric_type": "COSINE", "params": {}},
            limit=candidate_limit, expr=expression,
        ),
        AnnSearchRequest(
            data=[query], anns_field="sparse_vector",
            param={"metric_type": "BM25", "params": {}},
            limit=candidate_limit, expr=expression,
        ),
    ]
    response = get_client().hybrid_search(
        collection_name=_collection_name(),
        reqs=requests,
        ranker=RRFRanker(),
        limit=limit,
        output_fields=["message_id", "company_id", "created_ts", "content"],
    )
    hits = response[0] if response else []
    return [
        {
            "message_id": int(_hit_value(hit, "message_id", _hit_value(hit, "id", 0)) or 0),
            "company_id": int(_hit_value(hit, "company_id", 0) or 0) or None,
            "created_ts": int(_hit_value(hit, "created_ts", 0) or 0),
            "content": str(_hit_value(hit, "content", "")),
            "score": float(_hit_value(hit, "distance", 0.0) or 0.0),
        }
        for hit in hits
        if int(_hit_value(hit, "message_id", _hit_value(hit, "id", 0)) or 0)
    ]


def delete_session_memory(*, user_id: int, session_id: int) -> None:
    ensure_memory_collection()
    client = get_client()
    client.delete(collection_name=_collection_name(), filter=_memory_filter(user_id, session_id))
    client.flush(_collection_name())
