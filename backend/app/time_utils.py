from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def app_now(timezone_name: str) -> datetime:
    try:
        timezone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        timezone = ZoneInfo("UTC")
    return datetime.now(timezone).replace(microsecond=0)


def parse_date_offset(base: datetime, offset: object) -> str:
    if isinstance(offset, bool):
        days = 0
    elif isinstance(offset, (int, float)):
        days = int(offset)
    else:
        days = 0
    days = max(-7, min(7, days))
    return base.date().fromordinal(base.date().toordinal() + days).isoformat()

