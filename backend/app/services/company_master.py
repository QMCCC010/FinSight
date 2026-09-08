from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.enums import TrackingMode
from app.core.models import Company


def refresh_company_master(db: Session) -> int:
    import akshare as ak

    frame = ak.stock_info_a_code_name()
    changed = 0
    for _, row in frame.iterrows():
        code = str(row.get("code") or row.get("证券代码") or "").zfill(6)
        name = str(row.get("name") or row.get("证券简称") or "").strip()
        if not code or not name:
            continue
        company = db.scalar(select(Company).where(Company.stock_code == code))
        if not company:
            exchange = "SH" if code.startswith(("6", "9")) else "BJ" if code.startswith(("4", "8")) else "SZ"
            db.add(Company(stock_code=code, name=name, full_name=name, exchange=exchange, aliases=[name], tracking_mode=TrackingMode.INACTIVE))
            changed += 1
        elif company.name != name:
            company.name = name
            company.aliases = list(dict.fromkeys([*(company.aliases or []), name]))
            changed += 1
    db.commit()
    return changed

