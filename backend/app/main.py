import asyncio
import json
import logging
from datetime import date, datetime, timedelta
from typing import Any

import hmac

from fastapi import Cookie, Depends, FastAPI, HTTPException, Response, status
from pydantic import BaseModel

from .auth import COOKIE_NAME, clear_session_cookie, is_valid_session, set_session_cookie
from .classifier import classify_text
from .db import connect, fetchall_dict, fetchone_dict, init_db, row_to_dict
from .followups import answer_followup, skip_followup
from .parser import ParseResult, parse_entry
from .settings import Settings, get_settings
from .time_utils import app_now

app = FastAPI(title="Gut Check API")
logger = logging.getLogger(__name__)
_background_parse_tasks: set[asyncio.Task[None]] = set()


class LoginRequest(BaseModel):
    password: str


class RawLogRequest(BaseModel):
    raw_text: str


class FollowupAnswerRequest(BaseModel):
    answer_text: str


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
    init_db(get_settings().database_path)


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
    conn.commit()

    public = _public_log(conn, log["id"])
    public["new_events"] = events
    public["new_followups"] = []
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
    return {"date": day, "groups": grouped}


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
    }
