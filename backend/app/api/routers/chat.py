import asyncio
import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user
from app.core.database import SessionLocal, get_db
from app.core.enums import MessageStatus
from app.core.models import ChatMessage, ChatSession, Company, User
from app.core.schemas import ChatMessageOut, ChatSessionOut, CreateChatSession, CreateMessage, ResolveCompanyRequest

router = APIRouter(prefix="/chat", tags=["问答"])
logger = logging.getLogger(__name__)


def dispatch_message(message: ChatMessage, db: Session) -> None:
    from app.worker.tasks import run_agent_message
    result = run_agent_message.apply_async(args=[message.id], queue="chat", priority=9, expires=300)
    message.task_id = result.id
    db.commit()


@router.post("/sessions", response_model=ChatSessionOut)
def create_session(payload: CreateChatSession, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> ChatSession:
    session = ChatSession(user_id=user.id, title=payload.title)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


@router.get("/sessions", response_model=list[ChatSessionOut])
def list_sessions(db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> list[ChatSession]:
    return list(db.scalars(select(ChatSession).where(ChatSession.user_id == user.id).order_by(ChatSession.updated_at.desc())).all())


@router.get("/sessions/{session_id}/messages", response_model=list[ChatMessageOut])
def list_messages(session_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> list[ChatMessage]:
    session = db.get(ChatSession, session_id)
    if not session or session.user_id != user.id:
        raise HTTPException(status_code=404, detail="会话不存在")
    return list(db.scalars(select(ChatMessage).where(ChatMessage.session_id == session_id).order_by(ChatMessage.id)).all())


@router.get("/sessions/{session_id}/events")
async def message_events(
    session_id: int,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> StreamingResponse:
    session = db.get(ChatSession, session_id)
    if not session or session.user_id != user.id:
        raise HTTPException(status_code=404, detail="会话不存在")
    user_id = user.id

    async def stream():
        signature = None
        heartbeat = 0
        while not await request.is_disconnected():
            with SessionLocal() as stream_db:
                rows = list(stream_db.scalars(
                    select(ChatMessage)
                    .where(ChatMessage.session_id == session_id, ChatMessage.user_id == user_id)
                    .order_by(ChatMessage.id)
                ).all())
                payload = [ChatMessageOut.model_validate(row).model_dump(mode="json") for row in rows]
            current = tuple((
                item["id"], item["status"], item["progress"], item["updated_at"],
                item.get("answer"), item.get("refresh_status"), item.get("refresh_status_text"),
                json.dumps(item.get("analysis_metadata") or {}, ensure_ascii=False, sort_keys=True),
            ) for item in payload)
            if current != signature:
                signature = current
                yield f"event: messages\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
            heartbeat += 1
            if heartbeat >= 15:
                heartbeat = 0
                yield ": keep-alive\n\n"
            await asyncio.sleep(1)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/sessions/{session_id}/messages", response_model=ChatMessageOut, status_code=status.HTTP_202_ACCEPTED)
def create_message(
    session_id: int,
    payload: CreateMessage,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ChatMessage:
    session = db.get(ChatSession, session_id)
    if not session or session.user_id != user.id:
        raise HTTPException(status_code=404, detail="会话不存在")
    message = ChatMessage(session_id=session_id, user_id=user.id, role="assistant", question=payload.question, status=MessageStatus.QUEUED, status_text="问题已进入高优先级问答队列", clarification_candidates=[], analysis_metadata={})
    db.add(message)
    if session.title == "新对话":
        session.title = payload.question[:30]
    db.commit()
    db.refresh(message)
    try:
        dispatch_message(message, db)
    except Exception:
        logger.exception("Failed to enqueue chat message %s", message.id)
        # Allows the API to remain demonstrable without a running broker.
        message.status = MessageStatus.FAILED
        message.error = "任务队列暂时不可用，请稍后重试。"
        message.status_text = "任务队列不可用"
        db.commit()
    return message


@router.get("/messages/{message_id}", response_model=ChatMessageOut)
def get_message(message_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> ChatMessage:
    message = db.get(ChatMessage, message_id)
    if not message or message.user_id != user.id:
        raise HTTPException(status_code=404, detail="消息不存在")
    return message


@router.post("/messages/{message_id}/resolve-company", response_model=ChatMessageOut)
def resolve_company(
    message_id: int,
    payload: ResolveCompanyRequest,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> ChatMessage:
    message = db.get(ChatMessage, message_id)
    company = db.scalar(select(Company).where(Company.stock_code == payload.stock_code))
    if not message or message.user_id != user.id or not company:
        raise HTTPException(status_code=404, detail="消息或公司不存在")
    message.company_id = company.id
    message.status = MessageStatus.QUEUED
    message.status_text = "已确认公司，重新进入分析队列"
    message.clarification_candidates = []
    message.analysis_metadata = {}
    message.error = None
    db.commit()
    dispatch_message(message, db)
    return message


@router.post("/messages/{message_id}/cancel", response_model=ChatMessageOut)
def cancel_message(message_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> ChatMessage:
    message = db.get(ChatMessage, message_id)
    if not message or message.user_id != user.id:
        raise HTTPException(status_code=404, detail="消息不存在")
    if message.status in {MessageStatus.COMPLETED, MessageStatus.PARTIAL, MessageStatus.FAILED, MessageStatus.CANCELLED}:
        raise HTTPException(status_code=409, detail="该任务已经结束")
    if message.task_id:
        from app.worker.celery_app import celery
        celery.control.revoke(message.task_id, terminate=True, signal="SIGTERM")
    message.status = MessageStatus.CANCELLED
    message.progress = 100
    message.status_text = "已由用户停止"
    message.error = None
    db.commit()
    db.refresh(message)
    return message


@router.post("/messages/{message_id}/retry", response_model=ChatMessageOut)
def retry_message(message_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> ChatMessage:
    message = db.get(ChatMessage, message_id)
    if not message or message.user_id != user.id:
        raise HTTPException(status_code=404, detail="消息不存在")
    if message.status not in {MessageStatus.FAILED, MessageStatus.CANCELLED}:
        raise HTTPException(status_code=409, detail="只有失败或已停止的任务可以重新执行")
    message.status = MessageStatus.QUEUED
    message.progress = 0
    message.status_text = "问题已重新进入高优先级问答队列"
    message.answer = None
    message.citations = []
    message.missing_sources = []
    message.refresh_run_id = None
    message.refresh_status = None
    message.refresh_status_text = None
    message.refresh_requested_at = None
    message.refresh_completed_at = None
    message.clarification_candidates = []
    message.error = None
    db.commit()
    dispatch_message(message, db)
    db.refresh(message)
    return message
