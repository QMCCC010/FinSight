from datetime import datetime, timedelta

from app.core.enums import TrackingMode


def eligible_for_scheduled_update(mode: str, last_queried_at: datetime | None, now: datetime) -> bool:
    if mode in {TrackingMode.SEED, TrackingMode.PINNED}:
        return True
    return mode == TrackingMode.ON_DEMAND and bool(last_queried_at and last_queried_at >= now - timedelta(days=7))


def test_tracking_rules():
    now = datetime.now()
    assert eligible_for_scheduled_update(TrackingMode.SEED, None, now)
    assert eligible_for_scheduled_update(TrackingMode.PINNED, None, now)
    assert eligible_for_scheduled_update(TrackingMode.ON_DEMAND, now - timedelta(days=2), now)
    assert not eligible_for_scheduled_update(TrackingMode.ON_DEMAND, now - timedelta(days=8), now)
    assert not eligible_for_scheduled_update(TrackingMode.INACTIVE, now, now)

