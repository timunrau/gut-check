import json
import re
import sqlite3
from datetime import datetime
from typing import Any

VAGUE_MEAL_WORDS = {"takeout", "leftovers", "restaurant food", "snack", "meal"}


def _missing(data: dict[str, Any], key: str) -> bool:
    value = data.get(key)
    return value is None or value == [] or value == {}


def _raw_mentions(raw_text: str, words: list[str]) -> bool:
    lower = raw_text.lower()
    return any(word in lower for word in words)


def _choice_json(choices: list[str] | None) -> str | None:
    return json.dumps(choices) if choices else None


def questions_for_event(raw_text: str, event: dict[str, Any]) -> list[dict[str, Any]]:
    event_type = event["event_type"]
    data = event.get("data") or {}
    questions: list[dict[str, Any]] = []

    if event_type == "meal":
        if _missing(data, "portion"):
            questions.append(
                {
                    "question_text": "About how much was the portion?",
                    "field_target": "meal.portion",
                    "answer_type": "text",
                    "choices": None,
                }
            )
        item_text = " ".join(data.get("foods", []) + data.get("drinks", []))
        if any(word in raw_text.lower() or word in item_text for word in VAGUE_MEAL_WORDS):
            questions.append(
                {
                    "question_text": "What were the main ingredients?",
                    "field_target": "meal.ingredients",
                    "answer_type": "text",
                    "choices": None,
                }
            )

    elif event_type == "bowel_movement":
        if _missing(data, "bristol"):
            questions.append(
                {
                    "question_text": "What Bristol stool type was it?",
                    "field_target": "bowel_movement.bristol",
                    "answer_type": "choice",
                    "choices": ["1", "2", "3", "4", "5", "6", "7"],
                }
            )
        if _missing(data, "pain"):
            questions.append(
                {
                    "question_text": "Any pain, 1-5?",
                    "field_target": "bowel_movement.pain",
                    "answer_type": "number",
                    "choices": None,
                }
            )
        elif _missing(data, "bloating"):
            questions.append(
                {
                    "question_text": "Any bloating, 1-5?",
                    "field_target": "bowel_movement.bloating",
                    "answer_type": "number",
                    "choices": None,
                }
            )
        if _missing(data, "urgency") and not _raw_mentions(raw_text, ["urgent", "urgency", "rushed"]):
            questions.append(
                {
                    "question_text": "How urgent was it, 1-5?",
                    "field_target": "bowel_movement.urgency",
                    "answer_type": "number",
                    "choices": None,
                }
            )
        questions = questions[:2]

    elif event_type == "symptom":
        symptoms = data.get("symptoms") or []
        has_severity = any(symptom.get("severity") for symptom in symptoms if isinstance(symptom, dict))
        if not has_severity:
            questions.append(
                {
                    "question_text": "How severe was it, 1-5?",
                    "field_target": "symptom.severity",
                    "answer_type": "number",
                    "choices": None,
                }
            )

    elif event_type == "context":
        if _raw_mentions(raw_text, ["stress", "stressed"]) and _missing(data, "stress"):
            questions.append(
                {
                    "question_text": "How stressful was it, 1-5?",
                    "field_target": "context.stress",
                    "answer_type": "number",
                    "choices": None,
                }
            )
        if _raw_mentions(raw_text, ["sleep", "slept", "tired"]) and _missing(data, "sleep_hours"):
            questions.append(
                {
                    "question_text": "About how many hours did you sleep?",
                    "field_target": "context.sleep_hours",
                    "answer_type": "number",
                    "choices": None,
                }
            )
        if _raw_mentions(raw_text, ["med", "medication", "supplement"]) and not data.get("meds") and not data.get("supplements"):
            questions.append(
                {
                    "question_text": "What medication or supplement was it?",
                    "field_target": "context.medication_name",
                    "answer_type": "text",
                    "choices": None,
                }
            )

    return questions[:2]


def unknown_question() -> dict[str, Any]:
    return {
        "question_text": "What kind of entry is this?",
        "field_target": "unknown.type",
        "answer_type": "choice",
        "choices": ["Meal", "Poop", "Symptom", "Context", "Ignore"],
    }


def create_followups(
    conn: sqlite3.Connection,
    raw_log_id: int,
    raw_text: str,
    events: list[dict[str, Any]],
    classification: str,
    created_at: str,
) -> list[dict[str, Any]]:
    pending: list[dict[str, Any]] = []
    if classification == "unknown" and not events:
        item = unknown_question()
        item["event_id"] = None
        pending.append(item)
    else:
        for event in events:
            for item in questions_for_event(raw_text, event):
                item["event_id"] = event["id"]
                pending.append(item)
                if len(pending) >= 4:
                    break
            if len(pending) >= 4:
                break

    created = []
    for item in pending[:4]:
        cursor = conn.execute(
            """
            INSERT INTO follow_up_questions (
                raw_log_id, event_id, question_text, field_target, answer_type,
                choices_json, status, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, 'open', ?)
            """,
            (
                raw_log_id,
                item.get("event_id"),
                item["question_text"],
                item["field_target"],
                item["answer_type"],
                _choice_json(item.get("choices")),
                created_at,
            ),
        )
        created.append(
            {
                "id": cursor.lastrowid,
                "raw_log_id": raw_log_id,
                "event_id": item.get("event_id"),
                "question_text": item["question_text"],
                "field_target": item["field_target"],
                "answer_type": item["answer_type"],
                "choices": item.get("choices"),
                "status": "open",
                "answer_text": None,
                "created_at": created_at,
                "answered_at": None,
            }
        )
    return created


def _parse_first_number(answer: str) -> float | None:
    match = re.search(r"\d+(?:\.\d+)?", answer)
    return float(match.group(0)) if match else None


def _set_number(data: dict[str, Any], key: str, answer: str, low: int, high: int, integer: bool = True) -> bool:
    value = _parse_first_number(answer)
    if value is None or value < low or value > high:
        return False
    data[key] = int(value) if integer else value
    return True


def apply_followup_answer(conn: sqlite3.Connection, followup: dict[str, Any], answer_text: str) -> None:
    event_id = followup.get("event_id")
    if not event_id:
        return

    row = conn.execute("SELECT data_json FROM events WHERE id = ?", (event_id,)).fetchone()
    if not row:
        return
    try:
        data = json.loads(row["data_json"])
    except json.JSONDecodeError:
        data = {}

    target = followup["field_target"]
    changed = False
    if target == "meal.portion":
        data["portion"] = answer_text.strip() or None
        changed = bool(data["portion"])
    elif target == "bowel_movement.bristol":
        changed = _set_number(data, "bristol", answer_text, 1, 7)
    elif target == "bowel_movement.pain":
        changed = _set_number(data, "pain", answer_text, 1, 5)
    elif target == "bowel_movement.bloating":
        changed = _set_number(data, "bloating", answer_text, 1, 5)
    elif target == "bowel_movement.urgency":
        changed = _set_number(data, "urgency", answer_text, 1, 5)
    elif target == "symptom.severity":
        changed = _set_number(data, "severity", answer_text, 1, 5)
        symptoms = data.get("symptoms") or []
        if len(symptoms) == 1 and isinstance(symptoms[0], dict) and symptoms[0].get("severity") is None:
            value = data.pop("severity")
            symptoms[0]["severity"] = value
            data["symptoms"] = symptoms
    elif target == "context.stress":
        changed = _set_number(data, "stress", answer_text, 1, 5)
    elif target == "context.sleep_hours":
        changed = _set_number(data, "sleep_hours", answer_text, 0, 24, integer=False)
    elif target == "context.medication_name":
        name = answer_text.strip().lower()
        if name:
            data.setdefault("meds", []).append(name)
            changed = True

    if changed:
        conn.execute("UPDATE events SET data_json = ? WHERE id = ?", (json.dumps(data), event_id))


def answer_followup(conn: sqlite3.Connection, followup_id: int, answer_text: str, answered_at: str) -> dict[str, Any] | None:
    followup = conn.execute("SELECT * FROM follow_up_questions WHERE id = ?", (followup_id,)).fetchone()
    if not followup:
        return None
    followup_dict = dict(followup)
    apply_followup_answer(conn, followup_dict, answer_text)
    conn.execute(
        """
        UPDATE follow_up_questions
        SET status = 'answered', answer_text = ?, answered_at = ?
        WHERE id = ?
        """,
        (answer_text, answered_at, followup_id),
    )
    return dict(conn.execute("SELECT * FROM follow_up_questions WHERE id = ?", (followup_id,)).fetchone())


def skip_followup(conn: sqlite3.Connection, followup_id: int, answered_at: str) -> dict[str, Any] | None:
    followup = conn.execute("SELECT * FROM follow_up_questions WHERE id = ?", (followup_id,)).fetchone()
    if not followup:
        return None
    conn.execute(
        """
        UPDATE follow_up_questions
        SET status = 'skipped', answered_at = ?
        WHERE id = ?
        """,
        (answered_at, followup_id),
    )
    return dict(conn.execute("SELECT * FROM follow_up_questions WHERE id = ?", (followup_id,)).fetchone())

