import statistics
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from .db import fetchall_dict, fetchone_dict, row_to_dict


@dataclass
class GarminMetric:
    metric_date: str
    steps: int | None = None
    sleep_hours: float | None = None
    sleep_score: float | None = None
    stress_avg: float | None = None
    stress_max: float | None = None
    body_battery_min: float | None = None
    body_battery_max: float | None = None
    body_battery_avg: float | None = None
    body_battery_end: float | None = None


@dataclass
class PendingLogin:
    client: Any
    mfa_context: Any
    tokenstore: str


_pending_logins: dict[str, PendingLogin] = {}
_garmin_factory: Callable[..., Any] | None = None


def set_garmin_factory(factory: Callable[..., Any] | None) -> None:
    global _garmin_factory
    _garmin_factory = factory


def _new_garmin(*args: Any, **kwargs: Any) -> Any:
    if _garmin_factory is not None:
        return _garmin_factory(*args, **kwargs)
    try:
        from garminconnect import Garmin
    except ImportError as exc:
        raise RuntimeError("garminconnect is not installed") from exc
    return Garmin(*args, **kwargs)


def tokenstore_exists(tokenstore: str) -> bool:
    path = Path(tokenstore)
    if path.is_dir():
        return any(path.iterdir())
    return path.exists()


def _ensure_tokenstore_parent(tokenstore: str) -> None:
    Path(tokenstore).expanduser().parent.mkdir(parents=True, exist_ok=True)


def _dump_tokens(client: Any, tokenstore: str) -> None:
    _ensure_tokenstore_parent(tokenstore)
    if hasattr(client, "client") and hasattr(client.client, "dump"):
        client.client.dump(tokenstore)
        return
    if hasattr(client, "dump"):
        client.dump(tokenstore)
        return
    raise RuntimeError("Garmin client cannot store tokens")


def _login_with_tokens(tokenstore: str) -> Any:
    client = _new_garmin()
    client.login(tokenstore)
    return client


def start_auth(email: str, password: str, tokenstore: str) -> dict[str, Any]:
    client = _new_garmin(email=email, password=password, is_cn=False, return_on_mfa=True)
    result = client.login()
    if isinstance(result, tuple) and result and result[0] == "needs_mfa":
        pending_id = str(uuid.uuid4())
        _pending_logins[pending_id] = PendingLogin(client=client, mfa_context=result[1], tokenstore=tokenstore)
        return {"connected": False, "mfa_required": True, "pending_id": pending_id}

    _dump_tokens(client, tokenstore)
    return {"connected": True, "mfa_required": False, "pending_id": None}


def finish_auth(pending_id: str, mfa_code: str) -> dict[str, Any]:
    pending = _pending_logins.get(pending_id)
    if pending is None:
        raise RuntimeError("No pending Garmin MFA login found")
    pending.client.resume_login(pending.mfa_context, mfa_code)
    _dump_tokens(pending.client, pending.tokenstore)
    _pending_logins.pop(pending_id, None)
    return {"connected": True, "mfa_required": False, "pending_id": None}


def _first_number(payload: Any, keys: tuple[str, ...]) -> float | None:
    if isinstance(payload, dict):
        for key in keys:
            value = payload.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
        for value in payload.values():
            found = _first_number(value, keys)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _first_number(value, keys)
            if found is not None:
                return found
    return None


def _numeric_values(payload: Any, low: float | None = None, high: float | None = None) -> list[float]:
    values: list[float] = []
    if isinstance(payload, dict):
        for value in payload.values():
            values.extend(_numeric_values(value, low, high))
    elif isinstance(payload, list):
        for value in payload:
            values.extend(_numeric_values(value, low, high))
    elif isinstance(payload, (int, float)) and not isinstance(payload, bool):
        numeric = float(payload)
        if low is not None and numeric < low:
            return values
        if high is not None and numeric > high:
            return values
        values.append(numeric)
    return values


def _round(value: float | None, places: int = 2) -> float | None:
    return round(value, places) if value is not None else None


def extract_steps(summary: dict[str, Any] | None) -> int | None:
    value = _first_number(summary or {}, ("totalSteps", "steps", "stepCount", "totalStepCount"))
    return int(value) if value is not None else None


def extract_sleep(sleep: dict[str, Any] | None) -> tuple[float | None, float | None]:
    payload = sleep or {}
    seconds = _first_number(
        payload,
        (
            "sleepTimeSeconds",
            "totalSleepSeconds",
            "durationInSeconds",
            "sleepDurationSeconds",
            "totalSleepTimeSeconds",
        ),
    )
    hours = seconds / 3600 if seconds is not None and seconds > 24 else seconds
    score = _first_number(
        payload,
        ("sleepScore", "overallSleepScore", "overallScore", "score", "value"),
    )
    return _round(hours), _round(score, 1)


def extract_stress(stress: dict[str, Any] | None) -> tuple[float | None, float | None]:
    payload = stress or {}
    avg = _first_number(payload, ("avgStressLevel", "averageStressLevel", "stressAvg", "avgStress"))
    max_value = _first_number(payload, ("maxStressLevel", "stressMax", "maxStress"))
    values = _numeric_values(payload, 0, 100)
    if avg is None and values:
        avg = statistics.fmean(values)
    if max_value is None and values:
        max_value = max(values)
    return _round(avg, 1), _round(max_value, 1)


def _body_battery_values(payload: Any) -> list[float]:
    values: list[float] = []
    if isinstance(payload, dict):
        for key in ("bodyBatteryValuesArray", "bodyBatteryValueArray", "bodyBatteryValues"):
            values.extend(_body_battery_values(payload.get(key)))
        for key in ("bodyBatteryValue", "bodyBatteryLevel", "bodyBattery"):
            value = payload.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 100:
                values.append(float(value))
        for key, value in payload.items():
            if key not in {
                "bodyBatteryValuesArray",
                "bodyBatteryValueArray",
                "bodyBatteryValues",
                "bodyBatteryValue",
                "bodyBatteryLevel",
                "bodyBattery",
                "bodyBatteryVersion",
            }:
                values.extend(_body_battery_values(value))
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, list) and len(item) >= 2:
                sample = item[1]
                if isinstance(sample, (int, float)) and not isinstance(sample, bool) and 0 <= sample <= 100:
                    values.append(float(sample))
                    continue
            values.extend(_body_battery_values(item))
    return values


def extract_body_battery(body_battery: Any) -> tuple[float | None, float | None, float | None, float | None]:
    values = _body_battery_values(body_battery)
    if not values:
        return None, None, None, None
    return _round(min(values), 1), _round(max(values), 1), _round(statistics.fmean(values), 1), _round(values[-1], 1)


def metric_from_responses(
    metric_date: str,
    summary: dict[str, Any] | None,
    sleep: dict[str, Any] | None,
    stress: dict[str, Any] | None,
    body_battery: Any,
) -> GarminMetric:
    sleep_hours, sleep_score = extract_sleep(sleep)
    stress_avg, stress_max = extract_stress(stress)
    bb_min, bb_max, bb_avg, bb_end = extract_body_battery(body_battery)
    return GarminMetric(
        metric_date=metric_date,
        steps=extract_steps(summary),
        sleep_hours=sleep_hours,
        sleep_score=sleep_score,
        stress_avg=stress_avg,
        stress_max=stress_max,
        body_battery_min=bb_min,
        body_battery_max=bb_max,
        body_battery_avg=bb_avg,
        body_battery_end=bb_end,
    )


def upsert_metric(conn: Any, metric: GarminMetric, synced_at: str) -> dict[str, Any]:
    conn.execute(
        """
        INSERT INTO garmin_daily_metrics (
            metric_date, steps, sleep_hours, sleep_score, stress_avg, stress_max,
            body_battery_min, body_battery_max, body_battery_avg, body_battery_end,
            synced_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(metric_date) DO UPDATE SET
            steps = excluded.steps,
            sleep_hours = excluded.sleep_hours,
            sleep_score = excluded.sleep_score,
            stress_avg = excluded.stress_avg,
            stress_max = excluded.stress_max,
            body_battery_min = excluded.body_battery_min,
            body_battery_max = excluded.body_battery_max,
            body_battery_avg = excluded.body_battery_avg,
            body_battery_end = excluded.body_battery_end,
            synced_at = excluded.synced_at
        """,
        (
            metric.metric_date,
            metric.steps,
            metric.sleep_hours,
            metric.sleep_score,
            metric.stress_avg,
            metric.stress_max,
            metric.body_battery_min,
            metric.body_battery_max,
            metric.body_battery_avg,
            metric.body_battery_end,
            synced_at,
        ),
    )
    return row_to_dict(conn.execute("SELECT * FROM garmin_daily_metrics WHERE metric_date = ?", (metric.metric_date,)).fetchone())


def set_sync_state(
    conn: Any,
    *,
    connected: bool | None = None,
    last_sync_at: str | None = None,
    last_error: str | None = None,
    last_success_start_date: str | None = None,
    last_success_end_date: str | None = None,
) -> None:
    current = get_sync_state(conn)
    conn.execute(
        """
        UPDATE garmin_sync_state
        SET connected = ?,
            last_sync_at = ?,
            last_error = ?,
            last_success_start_date = ?,
            last_success_end_date = ?
        WHERE id = 1
        """,
        (
            int(current["connected"] if connected is None else connected),
            current["last_sync_at"] if last_sync_at is None else last_sync_at,
            current["last_error"] if last_error is None else last_error,
            current["last_success_start_date"] if last_success_start_date is None else last_success_start_date,
            current["last_success_end_date"] if last_success_end_date is None else last_success_end_date,
        ),
    )


def get_sync_state(conn: Any) -> dict[str, Any]:
    state = fetchone_dict(conn, "SELECT * FROM garmin_sync_state WHERE id = 1", ())
    if state is None:
        conn.execute("INSERT OR IGNORE INTO garmin_sync_state (id, connected) VALUES (1, 0)")
        state = fetchone_dict(conn, "SELECT * FROM garmin_sync_state WHERE id = 1", ())
    return state or {
        "id": 1,
        "connected": False,
        "last_sync_at": None,
        "last_error": None,
        "last_success_start_date": None,
        "last_success_end_date": None,
    }


def status(conn: Any, tokenstore: str) -> dict[str, Any]:
    state = get_sync_state(conn)
    state["tokenstore_exists"] = tokenstore_exists(tokenstore)
    state["mfa_pending"] = bool(_pending_logins)
    return state


def test_connection(conn: Any, tokenstore: str, today: str, synced_at: str) -> dict[str, Any]:
    client = _login_with_tokens(tokenstore)
    summary = client.get_user_summary(today)
    steps = extract_steps(summary)
    set_sync_state(conn, connected=True, last_error="")
    conn.commit()
    return {"ok": True, "date": today, "steps": steps}


def sync_range(conn: Any, tokenstore: str, start_date: str, end_date: str, synced_at: str) -> dict[str, Any]:
    client = _login_with_tokens(tokenstore)
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    if start > end:
        raise ValueError("start_date cannot be after end_date")

    rows: list[dict[str, Any]] = []
    current = start
    while current <= end:
        day = current.isoformat()
        summary = _safe_call(lambda: client.get_user_summary(day))
        sleep = _safe_call(lambda: client.get_sleep_data(day))
        stress = _safe_call(lambda: client.get_all_day_stress(day))
        body_battery = _safe_call(lambda: client.get_body_battery(day, day))
        metric = metric_from_responses(day, summary, sleep, stress, body_battery)
        rows.append(upsert_metric(conn, metric, synced_at))
        current += timedelta(days=1)

    set_sync_state(
        conn,
        connected=True,
        last_sync_at=synced_at,
        last_error="",
        last_success_start_date=start_date,
        last_success_end_date=end_date,
    )
    conn.commit()
    return {"synced": len(rows), "start_date": start_date, "end_date": end_date, "metrics": rows}


def _safe_call(fetch: Callable[[], Any]) -> Any:
    try:
        return fetch()
    except Exception:
        return None


def metric_for_day(conn: Any, day: str) -> dict[str, Any] | None:
    return fetchone_dict(conn, "SELECT * FROM garmin_daily_metrics WHERE metric_date = ?", (day,))


def metrics_between(conn: Any, start_date: str, end_date: str) -> list[dict[str, Any]]:
    return fetchall_dict(
        conn,
        """
        SELECT * FROM garmin_daily_metrics
        WHERE metric_date BETWEEN ? AND ?
        ORDER BY metric_date
        """,
        (start_date, end_date),
    )


def summarize_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "avg_steps": _avg(row.get("steps") for row in rows),
        "avg_sleep_hours": _avg(row.get("sleep_hours") for row in rows),
        "avg_sleep_score": _avg(row.get("sleep_score") for row in rows),
        "avg_stress": _avg(row.get("stress_avg") for row in rows),
        "avg_body_battery": _avg(row.get("body_battery_avg") for row in rows),
        "days_with_data": sum(1 for row in rows if _row_has_data(row)),
    }


def _avg(values: Any) -> float | None:
    numbers = [float(value) for value in values if isinstance(value, (int, float)) and not isinstance(value, bool)]
    return _round(statistics.fmean(numbers), 1) if numbers else None


def _row_has_data(row: dict[str, Any]) -> bool:
    return any(row.get(key) is not None for key in (
        "steps",
        "sleep_hours",
        "sleep_score",
        "stress_avg",
        "body_battery_avg",
    ))
