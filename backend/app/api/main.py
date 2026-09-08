from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from redis import Redis
from sqlalchemy import text

from app.api.routers import admin, analysis, auth, chat, companies, documents, reports
from app.core.config import get_settings
from app.core.database import SessionLocal
from app.core.seed import seed_database

settings = get_settings()


@asynccontextmanager
async def lifespan(_: FastAPI):
    with SessionLocal() as db:
        seed_database(db)
    yield


app = FastAPI(title=settings.app_name, version="0.2.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

for router in (auth.router, companies.router, documents.router, analysis.router, chat.router, reports.router, admin.router):
    app.include_router(router, prefix="/api/v1")


@app.get("/health")
def health() -> dict:
    checks = {"database": False, "redis": False}
    try:
        with SessionLocal() as db:
            db.execute(text("SELECT 1"))
            checks["database"] = True
    except Exception:
        pass
    try:
        checks["redis"] = bool(Redis.from_url(settings.redis_url).ping())
    except Exception:
        pass
    return {"status": "ok" if all(checks.values()) else "degraded", "service": "api", "version": app.version, "checks": checks}
