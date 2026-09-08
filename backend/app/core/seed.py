from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.enums import DocumentType, TrackingMode, UserRole
from app.core.models import Company, DataSource, User
from app.core.security import hash_password


FALLBACK_COMPANIES = [
    {"stock_code": "002594", "name": "比亚迪", "full_name": "比亚迪股份有限公司", "exchange": "SZ", "industry": "新能源汽车", "tracking_mode": TrackingMode.SEED},
    {"stock_code": "300750", "name": "宁德时代", "full_name": "宁德时代新能源科技股份有限公司", "exchange": "SZ", "industry": "动力电池", "tracking_mode": TrackingMode.SEED},
    {"stock_code": "002460", "name": "赣锋锂业", "full_name": "江西赣锋锂业集团股份有限公司", "exchange": "SZ", "industry": "锂资源", "tracking_mode": TrackingMode.SEED},
    {"stock_code": "600519", "name": "贵州茅台", "full_name": "贵州茅台酒股份有限公司", "exchange": "SH", "industry": "白酒", "tracking_mode": TrackingMode.INACTIVE},
    {"stock_code": "000001", "name": "平安银行", "full_name": "平安银行股份有限公司", "exchange": "SZ", "industry": "银行", "tracking_mode": TrackingMode.INACTIVE},
    {"stock_code": "600030", "name": "中信证券", "full_name": "中信证券股份有限公司", "exchange": "SH", "industry": "证券", "tracking_mode": TrackingMode.INACTIVE},
    {"stock_code": "300059", "name": "东方财富", "full_name": "东方财富信息股份有限公司", "exchange": "SZ", "industry": "互联网金融", "tracking_mode": TrackingMode.INACTIVE},
]

SOURCES = [
    ("东方财富个股研报", DocumentType.RESEARCH_REPORT, "research_report", "0 8 * * *"),
    ("东方财富个股新闻", DocumentType.NEWS, "stock_news", "*/30 * * * *"),
    ("巨潮资讯公司公告", DocumentType.ANNOUNCEMENT, "cninfo_announcement", "0 8 * * *"),
    ("东方财富股吧舆情", DocumentType.SOCIAL, "eastmoney_social", "*/30 * * * *"),
]


def seed_database(db: Session) -> None:
    settings = get_settings()
    users = [
        (settings.seed_admin_username, settings.seed_admin_password, UserRole.ADMIN),
        (settings.seed_analyst_username, settings.seed_analyst_password, UserRole.ANALYST),
    ]
    for username, password, role in users:
        if not db.scalar(select(User).where(User.username == username)):
            db.add(User(username=username, password_hash=hash_password(password), role=role, is_active=True))

    for item in FALLBACK_COMPANIES:
        if not db.scalar(select(Company).where(Company.stock_code == item["stock_code"])):
            db.add(Company(**item, aliases=[item["name"], item["full_name"]]))

    for name, source_type, adapter, schedule in SOURCES:
        if not db.scalar(select(DataSource).where(DataSource.name == name)):
            db.add(DataSource(name=name, source_type=source_type, adapter=adapter, schedule=schedule, enabled=True))
    db.commit()

