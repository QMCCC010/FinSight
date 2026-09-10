from celery import Celery
from celery.schedules import crontab

from app.core.config import get_settings

settings = get_settings()
celery = Celery("finresearch", broker=settings.celery_broker_url, backend=settings.celery_result_backend, include=["app.worker.tasks"])
celery.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Shanghai",
    enable_utc=True,
    task_track_started=True,
    task_default_queue="maintenance",
    task_create_missing_queues=True,
    worker_prefetch_multiplier=1,
    task_routes={
        "app.worker.tasks.run_agent_message": {"queue": "chat", "priority": 9},
        "app.worker.tasks.generate_report": {"queue": "reports", "priority": 7},
        "app.worker.tasks.collect_company": {"queue": "collection", "priority": 3},
        "app.worker.tasks.process_document": {"queue": "collection", "priority": 3},
        "app.worker.tasks.dispatch_tracked": {"queue": "maintenance", "priority": 2},
        "app.worker.tasks.expire_on_demand": {"queue": "maintenance", "priority": 2},
        "app.worker.tasks.sync_company_master": {"queue": "maintenance", "priority": 2},
        "app.worker.tasks.rebuild_vector_index": {"queue": "maintenance", "priority": 2},
        "app.worker.tasks.delete_from_vector_index": {"queue": "maintenance", "priority": 5},
        "app.worker.tasks.update_conversation_memory": {"queue": "chat_memory", "priority": 4},
        "app.worker.tasks.recover_stale_tasks": {"queue": "maintenance", "priority": 8},
        "app.worker.tasks.sync_market_history": {"queue": "maintenance", "priority": 2},
        "app.worker.tasks.dispatch_market_sync": {"queue": "maintenance", "priority": 2},
    },
    task_annotations={
        "app.worker.tasks.run_agent_message": {"soft_time_limit": 600, "time_limit": 630},
        "app.worker.tasks.generate_report": {"soft_time_limit": 240, "time_limit": 300},
        "app.worker.tasks.collect_company": {"soft_time_limit": 900, "time_limit": 960},
        "app.worker.tasks.process_document": {"soft_time_limit": 240, "time_limit": 300},
        "app.worker.tasks.update_conversation_memory": {"soft_time_limit": 120, "time_limit": 150},
    },
    beat_schedule={
        "daily-reports-announcements": {"task": "app.worker.tasks.dispatch_tracked", "schedule": crontab(hour=8, minute=0), "args": [["RESEARCH_REPORT", "ANNOUNCEMENT"]]},
        "frequent-news-social": {"task": "app.worker.tasks.dispatch_tracked", "schedule": crontab(minute="*/30"), "args": [["NEWS", "SOCIAL"]]},
        "expire-on-demand": {"task": "app.worker.tasks.expire_on_demand", "schedule": crontab(hour=0, minute=10)},
        "sync-company-master": {"task": "app.worker.tasks.sync_company_master", "schedule": crontab(hour=2, minute=0)},
        "recover-stale-tasks": {"task": "app.worker.tasks.recover_stale_tasks", "schedule": crontab(minute="*/5")},
        "daily-market-history": {"task": "app.worker.tasks.dispatch_market_sync", "schedule": crontab(hour=18, minute=10)},
    },
)
