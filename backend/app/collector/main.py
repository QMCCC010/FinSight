from __future__ import annotations

from uuid import uuid4

from flask import Flask, jsonify, request
from sqlalchemy import select

from app.core.config import get_settings
from app.core.database import Base, SessionLocal, engine
from app.core.enums import TriggerType
from app.core.models import Company, CrawlRun
from app.core.seed import seed_database

app = Flask(__name__)
settings = get_settings()


def authorized() -> bool:
    return request.headers.get("X-Internal-Token") == settings.internal_service_token


@app.before_request
def require_internal_token():
    if request.path != "/health" and not authorized():
        return jsonify({"error": "unauthorized"}), 401


@app.get("/health")
def health():
    return jsonify({"status": "ok", "service": "collector"})


@app.post("/internal/v1/crawl-runs")
def create_runs():
    payload = request.get_json(silent=True) or {}
    stock_codes = payload.get("stock_codes") or []
    source_types = payload.get("source_types") or None
    trigger = payload.get("trigger_type") or TriggerType.ADMIN
    runs = []
    with SessionLocal() as db:
        query = select(Company)
        if stock_codes:
            query = query.where(Company.stock_code.in_(stock_codes))
        else:
            query = query.where(Company.tracking_mode.in_(["SEED", "PINNED", "ON_DEMAND"]))
        from app.worker.tasks import collect_company
        for company in db.scalars(query).all():
            run = CrawlRun(
                job_id=str(uuid4()),
                company_id=company.id,
                source_type=source_types[0] if source_types and len(source_types) == 1 else None,
                source_types=source_types or [],
                trigger_type=trigger,
                status="QUEUED",
                stage="等待采集",
            )
            db.add(run)
            db.flush()
            collect_company.apply_async(args=[run.id, source_types], queue="collection", priority=3)
            runs.append({"id": run.id, "job_id": run.job_id, "stock_code": company.stock_code})
        db.commit()
    return jsonify({"runs": runs}), 202


@app.get("/internal/v1/crawl-runs/<int:run_id>")
def get_run(run_id: int):
    with SessionLocal() as db:
        run = db.get(CrawlRun, run_id)
        if not run:
            return jsonify({"error": "not found"}), 404
        return jsonify({"id": run.id, "job_id": run.job_id, "status": run.status, "stage": run.stage, "progress": run.progress, "created_count": run.created_count, "duplicate_count": run.duplicate_count, "failed_count": run.failed_count})


@app.get("/internal/v1/sources/health")
def source_health():
    return jsonify({"mode": settings.collection_mode, "sources": ["RESEARCH_REPORT", "NEWS", "ANNOUNCEMENT", "SOCIAL"]})


with app.app_context():
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        seed_database(db)
