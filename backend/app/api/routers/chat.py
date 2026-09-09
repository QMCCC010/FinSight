import asyncio
import json
import logging
from datetime import datetime, timedelta
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.dependencies import get_current_user
from app.core.database import SessionLocal, get_db
from app.core.enums import MessageStatus
from app.core.models import ChatMessage, ChatSession, Company, User
from app.core.schemas import ChatMessageOut, ChatSessionOut, CreateChatSession, CreateMessage, ResolveCompanyRequest

router = APIRouter(prefix="/chat", tags=["问答"])
logger = logging.getLogger(__name__)
ACTIVE_MESSAGE_STATUSES = (
    MessageStatus.QUEUED,
    MessageStatus.RESOLVING_ENTITY,
    MessageStatus.COLLECTING,
    MessageStatus.PROCESSING,
    MessageStatus.ANSWERING,
)
MAX_ACTIVE_MESSAGES_PER_USER = 3
MAX_SESSION_MESSAGES = 200


def dispatch_message(message: ChatMessage, db: Session) -> None:
    from app.worker.tasks import run_agent_message

    # Persist an execution token before publishing. This closes the race where
    # a fast worker starts before task_id is committed, and lets a retry make
    # an older still-running delivery harmless.
    execution_id = str(uuid4())
    message.task_id = execution_id
    db.commit()
    run_agent_message.apply_async(
        args=[message.id, execution_id],
        task_id=execution_id,
        queue="chat",
        priority=9,
        expires=300,
    )


def _mark_dispatch_failed(message: ChatMessage, db: Session) -> None:
    message.status = MessageStatus.FAILED
    message.progress = 100
    message.status_text = "任务队列不可用"
    message.error = "任务队列暂时不可用，请稍后重试。"
    message.task_id = None
    db.commit()


def _reset_message(message: ChatMessage, status_text: str) -> None:
    message.status = MessageStatus.QUEUED
    message.progress = 0
    message.status_text = status_text
    message.answer = None
    message.citations = []
    message.missing_sources = []
    message.data_as_of = None
    message.confidence = None
    message.crawl_run_id = None
    message.refresh_run_id = None
    message.refresh_status = None
    message.refresh_status_text = None
    message.refresh_requested_at = None
    message.refresh_completed_at = None
    message.clarification_candidates = []
    message.analysis_metadata = {}
    message.error = None
    message.task_id = None


def _recent_messages(db: Session, session_id: int, user_id: int) -> list[ChatMessage]:
    rows = list(db.scalars(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id, ChatMessage.user_id == user_id)
        .order_by(ChatMessage.id.desc())
        .limit(MAX_SESSION_MESSAGES)
    ).all())
    rows.reverse()
    return rows


@router.post("/sessions", response_model=ChatSessionOut)
def create_session(payload: CreateChatSession, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> ChatSession:
    session = ChatSession(user_id=user.id, title=payload.title)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


@router.get("/sessions", response_model=list[ChatSessionOut])
def list_sessions(db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> list[ChatSession]:
    return list(db.scalars(select(ChatSession).where(ChatSession.user_id == user.id).order_by(ChatSession.updated_at.desc()).limit(100)).all())


@router.get("/sessions/{session_id}/messages", response_model=list[ChatMessageOut])
def list_messages(session_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> list[ChatMessage]:
    session = db.get(ChatSession, session_id)
    if not session or session.user_id != user.id:
        raise HTTPException(status_code=404, detail="会话不存在")
    return _recent_messages(db, session_id, user.id)


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
    # Streaming responses otherwise keep this dependency-scoped connection
    # checked out for the whole lifetime of the browser tab.
    db.close()

    async def stream():
        signature = None
        heartbeat = 0
        while not await request.is_disconnected():
            with SessionLocal() as stream_db:
                current = stream_db.execute(
                    select(func.count(ChatMessage.id), func.max(ChatMessage.id), func.max(ChatMessage.updated_at)).where(
                        ChatMessage.session_id == session_id,
                        ChatMessage.user_id == user_id,
                    )
                ).one()
                if current != signature:
                    rows = _recent_messages(stream_db, session_id, user_id)
                    payload = [ChatMessageOut.model_validate(row).model_dump(mode="json") for row in rows]
                else:
                    payload = None
            if payload is not None:
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
    session = db.scalar(select(ChatSession).where(ChatSession.id == session_id).with_for_update())
    if not session or session.user_id != user.id:
        raise HTTPException(status_code=404, detail="会话不存在")
    db.scalar(select(User.id).where(User.id == user.id).with_for_update())
    active_count = db.scalar(select(func.count(ChatMessage.id)).where(
        ChatMessage.user_id == user.id,
        ChatMessage.status.in_(ACTIVE_MESSAGE_STATUSES),
    )) or 0
    if active_count >= MAX_ACTIVE_MESSAGES_PER_USER:
        raise HTTPException(status_code=429, detail="当前已有3个问答任务在运行，请等待完成或停止其中一个任务")
    duplicate = db.scalar(
        select(ChatMessage.id).where(
            ChatMessage.session_id == session_id,
            ChatMessage.user_id == user.id,
            ChatMessage.question == payload.question,
            ChatMessage.created_at >= datetime.now() - timedelta(seconds=5),
        ).limit(1)
    )
    if duplicate:
        raise HTTPException(status_code=409, detail="相同问题刚刚已经提交，请勿重复发送")
    message = ChatMessage(session_id=session_id, user_id=user.id, role="assistant", question=payload.question, status=MessageStatus.QUEUED, status_text="问题已进入高优先级问答队列", clarification_candidates=[], analysis_metadata={})
    db.add(message)
    if session.title == "新对话":
        session.title = payload.question[:30]
    session.updated_at = datetime.now()
    db.commit()
    db.refresh(message)
    try:
        dispatch_message(message, db)
    except Exception:
        logger.exception("Failed to enqueue chat message %s", message.id)
        _mark_dispatch_failed(message, db)
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
    message = db.scalar(select(ChatMessage).where(ChatMessage.id == message_id).with_for_update())
    if not message or message.user_id != user.id:
        raise HTTPException(status_code=404, detail="消息不存在")
    if message.status != MessageStatus.NEEDS_CLARIFICATION:
        raise HTTPException(status_code=409, detail="该消息当前不需要选择公司")
    allowed_codes = {str(item.get("stock_code")) for item in (message.clarification_candidates or [])}
    if payload.stock_code not in allowed_codes:
        raise HTTPException(status_code=409, detail="请选择系统给出的候选公司")
    company = db.scalar(select(Company).where(Company.stock_code == payload.stock_code))
    if not company:
        raise HTTPException(status_code=404, detail="消息或公司不存在")
    message.company_id = company.id
    _reset_message(message, "已确认公司，重新进入分析队列")
    db.commit()
    try:
        dispatch_message(message, db)
    except Exception:
        logger.exception("Failed to enqueue resolved chat message %s", message.id)
        _mark_dispatch_failed(message, db)
    db.refresh(message)
    return message


@router.post("/messages/{message_id}/cancel", response_model=ChatMessageOut)
def cancel_message(message_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> ChatMessage:
    message = db.scalar(select(ChatMessage).where(ChatMessage.id == message_id).with_for_update())
    if not message or message.user_id != user.id:
        raise HTTPException(status_code=404, detail="消息不存在")
    if message.status in {MessageStatus.COMPLETED, MessageStatus.PARTIAL, MessageStatus.FAILED, MessageStatus.CANCELLED}:
        raise HTTPException(status_code=409, detail="该任务已经结束")
    if message.task_id:
        from app.worker.celery_app import celery
        # Running tasks stop cooperatively at graph node boundaries. Killing the
        # prefork child here caused worker churn and occasionally lost unrelated
        # queued work.
        celery.control.revoke(message.task_id, terminate=False)
    message.status = MessageStatus.CANCELLED
    message.progress = 100
    message.status_text = "已由用户停止"
    message.error = None
    db.commit()
    db.refresh(message)
    return message


@router.post("/messages/{message_id}/retry", response_model=ChatMessageOut)
def retry_message(message_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> ChatMessage:
    message = db.scalar(select(ChatMessage).where(ChatMessage.id == message_id).with_for_update())
    if not message or message.user_id != user.id:
        raise HTTPException(status_code=404, detail="消息不存在")
    if message.status not in {MessageStatus.FAILED, MessageStatus.CANCELLED}:
        raise HTTPException(status_code=409, detail="只有失败或已停止的任务可以重新执行")
    if message.task_id:
        from app.worker.celery_app import celery
        celery.control.revoke(message.task_id, terminate=False)
    _reset_message(message, "问题已重新进入高优先级问答队列")
    db.commit()
    try:
        dispatch_message(message, db)
    except Exception:
        logger.exception("Failed to re-enqueue chat message %s", message.id)
        _mark_dispatch_failed(message, db)
    db.refresh(message)
    return message
