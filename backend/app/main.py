import asyncio
import json
import logging
from datetime import date, datetime, time, timedelta
from typing import Any

import hmac

from fastapi import Cookie, Depends, FastAPI, HTTPException, Response, status
from pydantic import BaseModel

from .auth import COOKIE_NAME, clear_session_cookie, is_valid_session, set_session_cookie
from .classifier import classify_text
from .db import connect, fetchall_dict, fetchone_dict, init_db, row_to_dict
from .followups import answer_followup, create_followups, skip_followup
from . import garmin
from .parser import ParseResult, parse_entry
from .settings import Settings, get_settings
from .time_utils import app_now
from .triggers import analyze_trigger_patterns

app = FastAPI(title="Gut Check API")
logger = logging.getLogger(__name__)
_background_parse_tasks: set[asyncio.Task[None]] = set()
_background_garmin_tasks: set[asyncio.Task[None]] = set()
_garmin_nightly_task: asyncio.Task[None] | None = None


class LoginRequest(BaseModel):
    password: str


class RawLogRequest(BaseModel):
    raw_text: str


class FollowupAnswerRequest(BaseModel):
    answer_text: str


class GarminAuthStartRequest(BaseModel):
    email: str
    password: str


class GarminAuthFinishRequest(BaseModel):
    pending_id: str
    mfa_code: str


class GarminSyncRequest(BaseModel):
    days: int = 14


def settings_dep() -> Settings:
    return get_settings()


def get_conn(settings: Settings = Depends(settings_dep)):
    conn = connect(settings.database_path)
    try:
        yield conn
    finally:
        conn.close()


def protected(
    settings: Settings = Depends(settings_dep),
    cookie_value: str | None = Cookie(default=None, alias=COOKIE_NAME),
) -> None:
    if not is_valid_session(cookie_value, settings.session_secret):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")


@app.on_event("startup")
def startup() -> None:
    settings = get_settings()
    init_db(settings.database_path)
    _schedule_garmin_startup_sync(settings)
    _schedule_garmin_nightly_sync(settings)


@app.on_event("shutdown")
async def shutdown() -> None:
    global _garmin_nightly_task
    if _garmin_nightly_task is not None:
        _garmin_nightly_task.cancel()
        try:
            await _garmin_nightly_task
        except asyncio.CancelledError:
            pass
        _garmin_nightly_task = None


def _created_at_datetime(raw_created_at: str) -> datetime:
    return datetime.fromisoformat(raw_created_at)


def _public_log(conn, log_id: int) -> dict[str, Any]:
    log = fetchone_dict(conn, "SELECT * FROM raw_logs WHERE id = ?", (log_id,))
    if not log:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Log not found")
    log["events"] = fetchall_dict(
        conn,
        """
        SELECT * FROM events
        WHERE raw_log_id = ?
        ORDER BY event_date DESC, event_time DESC, id DESC
        """,
        (log_id,),
    )
    log["followups"] = fetchall_dict(
        conn,
        """
        SELECT * FROM follow_up_questions
        WHERE raw_log_id = ?
        ORDER BY id
        """,
        (log_id,),
    )
    return log


def _insert_events(conn, raw_log_id: int, parse_result: ParseResult) -> list[dict[str, Any]]:
    inserted: list[dict[str, Any]] = []
    for event in parse_result.events:
        cursor = conn.execute(
            """
            INSERT INTO events (
                raw_log_id, event_type, event_date, event_time,
                time_was_defaulted, notes, confidence, data_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                raw_log_id,
                event.event_type,
                event.event_date,
                event.event_time,
                1 if event.time_was_defaulted else 0,
                event.notes,
                event.confidence,
                json.dumps(event.data),
            ),
        )
        inserted.append(row_to_dict(conn.execute("SELECT * FROM events WHERE id = ?", (cursor.lastrowid,)).fetchone()))
    return inserted


def _update_log_after_parse(conn, raw_log_id: int, parse_result: ParseResult, model_name: str) -> None:
    conn.execute(
        """
        UPDATE raw_logs
        SET parser_status = ?,
            model_name = ?,
            parser_error = ?,
            parsed_json = ?,
            entry_classification = ?,
            classification_confidence = ?
        WHERE id = ?
        """,
        (
            parse_result.status,
            model_name,
            parse_result.parser_error,
            json.dumps(parse_result.parsed_json) if parse_result.parsed_json is not None else None,
            parse_result.classification,
            parse_result.confidence,
            raw_log_id,
        ),
    )


async def _parse_and_store(conn, log: dict[str, Any], settings: Settings) -> dict[str, Any]:
    logged_at = _created_at_datetime(log["created_at"])
    parse_result = await parse_entry(
        log["raw_text"],
        logged_at,
        settings.ollama_url,
        settings.ollama_model,
        settings.ollama_num_ctx,
        settings.ollama_num_predict,
        settings.ollama_timeout_seconds,
    )

    conn.execute("DELETE FROM follow_up_questions WHERE raw_log_id = ?", (log["id"],))
    conn.execute("DELETE FROM events WHERE raw_log_id = ?", (log["id"],))
    _update_log_after_parse(conn, log["id"], parse_result, settings.ollama_model)
    events = _insert_events(conn, log["id"], parse_result)
    followups = create_followups(
        conn,
        log["id"],
        log["raw_text"],
        events,
        parse_result.classification,
        app_now(settings.app_timezone).isoformat(),
    )
    conn.commit()

    public = _public_log(conn, log["id"])
    public["new_events"] = events
    public["new_followups"] = followups
    return public


async def _parse_log_in_background(log_id: int, settings: Settings) -> None:
    conn = connect(settings.database_path)
    try:
        log = fetchone_dict(conn, "SELECT * FROM raw_logs WHERE id = ?", (log_id,))
        if not log or log["parser_status"] != "pending":
            return
        await _parse_and_store(conn, log, settings)
    except Exception as exc:
        logger.exception("Background parse failed for raw log %s", log_id)
        conn.rollback()
        conn.execute(
            """
            UPDATE raw_logs
            SET parser_status = 'failed',
                parser_error = ?
            WHERE id = ?
              AND parser_status = 'pending'
            """,
            (f"Background parse failed: {exc.__class__.__name__}", log_id),
        )
        conn.commit()
    finally:
        conn.close()


def _schedule_parse(log_id: int, settings: Settings) -> None:
    task = asyncio.create_task(_parse_log_in_background(log_id, settings))
    _background_parse_tasks.add(task)
    task.add_done_callback(_background_parse_tasks.discard)


def _parse_garmin_sync_time(raw_time: str) -> time:
    try:
        hour_text, minute_text = raw_time.split(":", 1)
        parsed = time(hour=int(hour_text), minute=int(minute_text))
    except (TypeError, ValueError):
        logger.warning("Invalid GARMIN_SYNC_TIME=%r; using 03:15", raw_time)
        return time(hour=3, minute=15)
    return parsed


def _next_garmin_sync_after(now: datetime, raw_time: str) -> datetime:
    sync_time = _parse_garmin_sync_time(raw_time)
    next_run = now.replace(
        hour=sync_time.hour,
        minute=sync_time.minute,
        second=0,
        microsecond=0,
    )
    if next_run <= now:
        next_run += timedelta(days=1)
    return next_run


def _sync_recent_garmin(settings: Settings, reason: str = "startup") -> None:
    if not garmin.tokenstore_exists(settings.garmin_tokenstore):
        return
    conn = connect(settings.database_path)
    try:
        now = app_now(settings.app_timezone)
        days = max(1, min(settings.garmin_sync_days, 60))
        end = now.date()
        start = end - timedelta(days=days - 1)
        garmin.sync_range(
            conn,
            settings.garmin_tokenstore,
            start.isoformat(),
            end.isoformat(),
            now.isoformat(),
        )
    except Exception as exc:
        logger.warning("Garmin %s sync failed: %s", reason, exc)
        conn.rollback()
        garmin.set_sync_state(conn, connected=False, last_error=f"{exc.__class__.__name__}: {exc}")
        conn.commit()
    finally:
        conn.close()


def _schedule_garmin_startup_sync(settings: Settings) -> None:
    if not settings.garmin_auto_sync_enabled:
        return
    if not garmin.tokenstore_exists(settings.garmin_tokenstore):
        return
    try:
        task = asyncio.create_task(asyncio.to_thread(_sync_recent_garmin, settings, "startup"))
    except RuntimeError:
        return
    _background_garmin_tasks.add(task)
    task.add_done_callback(_background_garmin_tasks.discard)


async def _garmin_nightly_sync_loop(settings: Settings) -> None:
    while True:
        now = app_now(settings.app_timezone)
        next_run = _next_garmin_sync_after(now, settings.garmin_sync_time)
        await asyncio.sleep(max(1.0, (next_run - now).total_seconds()))
        await asyncio.to_thread(_sync_recent_garmin, settings, "nightly")


def _schedule_garmin_nightly_sync(settings: Settings) -> None:
    global _garmin_nightly_task
    if not settings.garmin_auto_sync_enabled:
        return
    if _garmin_nightly_task is not None and not _garmin_nightly_task.done():
        return
    try:
        _garmin_nightly_task = asyncio.create_task(_garmin_nightly_sync_loop(settings))
    except RuntimeError:
        _garmin_nightly_task = None


@app.get("/api/health")
def health(_auth: None = Depends(protected)) -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/auth/login")
def login(payload: LoginRequest, response: Response, settings: Settings = Depends(settings_dep)) -> dict[str, bool]:
    if not settings.app_password:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="APP_PASSWORD is not set")
    if not hmac.compare_digest(payload.password, settings.app_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid password")
    set_session_cookie(response, settings.session_secret)
    return {"authenticated": True}


@app.post("/api/auth/logout")
def logout(response: Response, _auth: None = Depends(protected)) -> dict[str, bool]:
    clear_session_cookie(response)
    return {"authenticated": False}


@app.get("/api/auth/me")
def me(_auth: None = Depends(protected)) -> dict[str, bool]:
    return {"authenticated": True}


@app.post("/api/logs")
async def create_log(
    payload: RawLogRequest,
    conn=Depends(get_conn),
    settings: Settings = Depends(settings_dep),
    _auth: None = Depends(protected),
) -> dict[str, Any]:
    raw_text = payload.raw_text.strip()
    if not raw_text:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Entry cannot be empty")

    created_at = app_now(settings.app_timezone).isoformat()
    classification, confidence, _scores = classify_text(raw_text)
    cursor = conn.execute(
        """
        INSERT INTO raw_logs (
            raw_text, created_at, parser_status, model_name, parser_error,
            parsed_json, entry_classification, classification_confidence
        )
        VALUES (?, ?, 'pending', ?, NULL, NULL, ?, ?)
        """,
        (raw_text, created_at, settings.ollama_model, classification, confidence),
    )
    log_id = cursor.lastrowid
    conn.commit()
    _schedule_parse(log_id, settings)
    public = _public_log(conn, log_id)
    public["new_events"] = []
    public["new_followups"] = []
    return public


@app.get("/api/logs/recent")
def recent_logs(conn=Depends(get_conn), _auth: None = Depends(protected)) -> list[dict[str, Any]]:
    return fetchall_dict(
        conn,
        """
        SELECT raw_logs.*,
               (SELECT COUNT(*) FROM events WHERE events.raw_log_id = raw_logs.id) AS event_count,
               (SELECT COUNT(*) FROM follow_up_questions
                WHERE follow_up_questions.raw_log_id = raw_logs.id
                  AND follow_up_questions.status = 'open') AS open_followup_count
        FROM raw_logs
        ORDER BY created_at DESC, id DESC
        LIMIT 50
        """,
    )


@app.get("/api/logs/{log_id}")
def get_log(log_id: int, conn=Depends(get_conn), _auth: None = Depends(protected)) -> dict[str, Any]:
    return _public_log(conn, log_id)


@app.post("/api/logs/{log_id}/reparse")
async def reparse_log(
    log_id: int,
    conn=Depends(get_conn),
    settings: Settings = Depends(settings_dep),
    _auth: None = Depends(protected),
) -> dict[str, Any]:
    log = fetchone_dict(conn, "SELECT * FROM raw_logs WHERE id = ?", (log_id,))
    if not log:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Log not found")
    conn.execute(
        "UPDATE raw_logs SET parser_status = 'pending', parser_error = NULL WHERE id = ?",
        (log_id,),
    )
    conn.commit()
    return await _parse_and_store(conn, log, settings)


@app.delete("/api/logs/{log_id}")
def delete_log(log_id: int, conn=Depends(get_conn), _auth: None = Depends(protected)) -> dict[str, bool]:
    cursor = conn.execute("DELETE FROM raw_logs WHERE id = ?", (log_id,))
    conn.commit()
    if cursor.rowcount == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Log not found")
    return {"deleted": True}


@app.get("/api/events")
def events_for_date(date: str, conn=Depends(get_conn), _auth: None = Depends(protected)) -> list[dict[str, Any]]:
    return fetchall_dict(
        conn,
        """
        SELECT events.*, raw_logs.entry_classification, raw_logs.parser_status
        FROM events
        JOIN raw_logs ON raw_logs.id = events.raw_log_id
        WHERE event_date = ?
        ORDER BY event_time DESC, events.id DESC
        """,
        (date,),
    )


@app.delete("/api/events/{event_id}")
def delete_event(event_id: int, conn=Depends(get_conn), _auth: None = Depends(protected)) -> dict[str, bool]:
    cursor = conn.execute("DELETE FROM events WHERE id = ?", (event_id,))
    conn.commit()
    if cursor.rowcount == 0:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Event not found")
    return {"deleted": True}


@app.get("/api/followups/open")
def open_followups(conn=Depends(get_conn), _auth: None = Depends(protected)) -> list[dict[str, Any]]:
    return fetchall_dict(
        conn,
        """
        SELECT follow_up_questions.*, raw_logs.raw_text
        FROM follow_up_questions
        JOIN raw_logs ON raw_logs.id = follow_up_questions.raw_log_id
        WHERE follow_up_questions.status = 'open'
        ORDER BY follow_up_questions.id
        LIMIT 20
        """,
    )


@app.post("/api/followups/{followup_id}/answer")
def answer_followup_endpoint(
    followup_id: int,
    payload: FollowupAnswerRequest,
    conn=Depends(get_conn),
    settings: Settings = Depends(settings_dep),
    _auth: None = Depends(protected),
) -> dict[str, Any]:
    updated = answer_followup(conn, followup_id, payload.answer_text, app_now(settings.app_timezone).isoformat())
    conn.commit()
    if not updated:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Follow-up not found")
    return row_to_dict(updated)


@app.post("/api/followups/{followup_id}/skip")
def skip_followup_endpoint(
    followup_id: int,
    conn=Depends(get_conn),
    settings: Settings = Depends(settings_dep),
    _auth: None = Depends(protected),
) -> dict[str, Any]:
    updated = skip_followup(conn, followup_id, app_now(settings.app_timezone).isoformat())
    conn.commit()
    if not updated:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Follow-up not found")
    return row_to_dict(updated)


def _day_payload(conn, day: str) -> dict[str, Any]:
    items = events_for_date(day, conn=conn, _auth=None)
    grouped = {"meal": [], "bowel_movement": [], "symptom": [], "context": []}
    for item in items:
        grouped.setdefault(item["event_type"], []).append(item)
    return {"date": day, "groups": grouped, "garmin": garmin.metric_for_day(conn, day)}


@app.get("/api/day/{day}")
def day_summary(day: str, conn=Depends(get_conn), _auth: None = Depends(protected)) -> dict[str, Any]:
    return _day_payload(conn, day)


def _event_is_high_symptom(event: dict[str, Any]) -> bool:
    data = event.get("data") or {}
    for key in ("urgency", "pain", "bloating", "gas"):
        value = data.get(key)
        if isinstance(value, (int, float)) and value >= 4:
            return True
    bristol = data.get("bristol")
    return isinstance(bristol, int) and bristol in (6, 7)


@app.get("/api/week/{start_date}")
def week_summary(start_date: str, conn=Depends(get_conn), _auth: None = Depends(protected)) -> dict[str, Any]:
    try:
        start = date.fromisoformat(start_date)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Use YYYY-MM-DD") from None

    end = start + timedelta(days=6)
    events = fetchall_dict(
        conn,
        """
        SELECT * FROM events
        WHERE event_date BETWEEN ? AND ?
        ORDER BY event_date, event_time, id
        """,
        (start.isoformat(), end.isoformat()),
    )
    bowel_movements = [event for event in events if event["event_type"] == "bowel_movement"]
    high_bms = [event for event in bowel_movements if _event_is_high_symptom(event)]
    symptoms = [event for event in events if event["event_type"] == "symptom"]

    possible_items: dict[str, int] = {}
    meal_events = [event for event in events if event["event_type"] == "meal"]
    for bad_event in high_bms:
        bad_dt = datetime.fromisoformat(f"{bad_event['event_date']}T{bad_event['event_time']}:00")
        for meal in meal_events:
            meal_dt = datetime.fromisoformat(f"{meal['event_date']}T{meal['event_time']}:00")
            delta_hours = (bad_dt - meal_dt).total_seconds() / 3600
            if 0 <= delta_hours <= 24:
                data = meal.get("data") or {}
                for item in data.get("foods", []) + data.get("drinks", []):
                    possible_items[item] = possible_items.get(item, 0) + 1

    repeated = [
        {"item": item, "count": count, "language": "possible; appeared before bad episodes"}
        for item, count in sorted(possible_items.items(), key=lambda pair: (-pair[1], pair[0]))
        if count >= 2
    ][:5]
    garmin_rows = garmin.metrics_between(conn, start.isoformat(), end.isoformat())
    return {
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "counts": {
            "bowel_movements": len(bowel_movements),
            "high_symptom_bowel_movements": len(high_bms),
            "symptom_entries": len(symptoms),
        },
        "possible_repeated_foods_or_drinks": repeated,
        "note": "insufficient data" if not repeated else "worth watching only; not confirmed and not causal",
        "garmin": {
            "days": garmin_rows,
            "averages": garmin.summarize_metrics(garmin_rows),
        },
    }


@app.get("/api/garmin/status")
def garmin_status(
    conn=Depends(get_conn),
    settings: Settings = Depends(settings_dep),
    _auth: None = Depends(protected),
) -> dict[str, Any]:
    status_payload = garmin.status(conn, settings.garmin_tokenstore)
    status_payload["auto_sync_enabled"] = settings.garmin_auto_sync_enabled
    status_payload["auto_sync_time"] = settings.garmin_sync_time
    status_payload["auto_sync_days"] = settings.garmin_sync_days
    status_payload["next_auto_sync_at"] = (
        _next_garmin_sync_after(app_now(settings.app_timezone), settings.garmin_sync_time).isoformat()
        if settings.garmin_auto_sync_enabled
        else None
    )
    return status_payload


@app.post("/api/garmin/auth/start")
def garmin_auth_start(
    payload: GarminAuthStartRequest,
    conn=Depends(get_conn),
    settings: Settings = Depends(settings_dep),
    _auth: None = Depends(protected),
) -> dict[str, Any]:
    email = payload.email.strip()
    password = payload.password
    if not email or not password:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Garmin email and password are required")
    try:
        result = garmin.start_auth(email, password, settings.garmin_tokenstore)
        garmin.set_sync_state(conn, connected=bool(result["connected"]), last_error="")
        conn.commit()
        return result
    except Exception as exc:
        conn.rollback()
        garmin.set_sync_state(conn, connected=False, last_error=f"{exc.__class__.__name__}: {exc}")
        conn.commit()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Garmin login failed: {exc}") from exc


@app.post("/api/garmin/auth/finish")
def garmin_auth_finish(
    payload: GarminAuthFinishRequest,
    conn=Depends(get_conn),
    _auth: None = Depends(protected),
) -> dict[str, Any]:
    if not payload.pending_id.strip() or not payload.mfa_code.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Pending login and MFA code are required")
    try:
        result = garmin.finish_auth(payload.pending_id.strip(), payload.mfa_code.strip())
        garmin.set_sync_state(conn, connected=True, last_error="")
        conn.commit()
        return result
    except Exception as exc:
        conn.rollback()
        garmin.set_sync_state(conn, connected=False, last_error=f"{exc.__class__.__name__}: {exc}")
        conn.commit()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Garmin MFA failed: {exc}") from exc


@app.post("/api/garmin/test")
def garmin_test(
    conn=Depends(get_conn),
    settings: Settings = Depends(settings_dep),
    _auth: None = Depends(protected),
) -> dict[str, Any]:
    try:
        return garmin.test_connection(
            conn,
            settings.garmin_tokenstore,
            app_now(settings.app_timezone).date().isoformat(),
            app_now(settings.app_timezone).isoformat(),
        )
    except Exception as exc:
        conn.rollback()
        garmin.set_sync_state(conn, connected=False, last_error=f"{exc.__class__.__name__}: {exc}")
        conn.commit()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Garmin test failed: {exc}") from exc


@app.post("/api/garmin/sync")
def garmin_sync(
    payload: GarminSyncRequest,
    conn=Depends(get_conn),
    settings: Settings = Depends(settings_dep),
    _auth: None = Depends(protected),
) -> dict[str, Any]:
    days = max(1, min(payload.days, 60))
    end = app_now(settings.app_timezone).date()
    start = end - timedelta(days=days - 1)
    try:
        return garmin.sync_range(
            conn,
            settings.garmin_tokenstore,
            start.isoformat(),
            end.isoformat(),
            app_now(settings.app_timezone).isoformat(),
        )
    except Exception as exc:
        conn.rollback()
        garmin.set_sync_state(conn, connected=False, last_error=f"{exc.__class__.__name__}: {exc}")
        conn.commit()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Garmin sync failed: {exc}") from exc


@app.get("/api/patterns")
def trigger_patterns(
    days: int = 60,
    conn=Depends(get_conn),
    settings: Settings = Depends(settings_dep),
    _auth: None = Depends(protected),
) -> dict[str, Any]:
    bounded_days = max(7, min(days, 180))
    end = app_now(settings.app_timezone).date()
    start = end - timedelta(days=bounded_days - 1)
    events = fetchall_dict(
        conn,
        """
        SELECT * FROM events
        WHERE event_date BETWEEN ? AND ?
        ORDER BY event_date, event_time, id
        """,
        (start.isoformat(), end.isoformat()),
    )
    payload = analyze_trigger_patterns(events, bounded_days)
    garmin_rows = garmin.metrics_between(conn, start.isoformat(), end.isoformat())
    payload["start_date"] = start.isoformat()
    payload["end_date"] = end.isoformat()
    payload["events"] = events
    payload["garmin"] = {
        "days": garmin_rows,
        "averages": garmin.summarize_metrics(garmin_rows),
    }
    return payload
