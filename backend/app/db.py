import json
import sqlite3
from pathlib import Path
from typing import Any


def connect(database_path: str) -> sqlite3.Connection:
    Path(database_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(database_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(database_path: str) -> None:
    with connect(database_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS raw_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                raw_text TEXT NOT NULL,
                created_at TEXT NOT NULL,
                parser_status TEXT NOT NULL,
                model_name TEXT,
                parser_error TEXT,
                parsed_json TEXT,
                entry_classification TEXT NOT NULL,
                classification_confidence REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                raw_log_id INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                event_date TEXT NOT NULL,
                event_time TEXT NOT NULL,
                time_was_defaulted INTEGER NOT NULL,
                notes TEXT,
                confidence REAL NOT NULL,
                data_json TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY (raw_log_id) REFERENCES raw_logs(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS follow_up_questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                raw_log_id INTEGER NOT NULL,
                event_id INTEGER,
                question_text TEXT NOT NULL,
                field_target TEXT NOT NULL,
                answer_type TEXT NOT NULL,
                choices_json TEXT,
                status TEXT NOT NULL,
                answer_text TEXT,
                created_at TEXT NOT NULL,
                answered_at TEXT,
                FOREIGN KEY (raw_log_id) REFERENCES raw_logs(id) ON DELETE CASCADE,
                FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE
            );
            """
        )


def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    item = dict(row)
    for key in ("data_json", "parsed_json", "choices_json"):
        if key in item:
            value = item.pop(key)
            public_key = key.replace("_json", "")
            if value is None:
                item[public_key] = None
            else:
                try:
                    item[public_key] = json.loads(value)
                except json.JSONDecodeError:
                    item[public_key] = value
    if "time_was_defaulted" in item:
        item["time_was_defaulted"] = bool(item["time_was_defaulted"])
    return item


def fetchone_dict(conn: sqlite3.Connection, query: str, params: tuple[Any, ...]) -> dict[str, Any] | None:
    row = conn.execute(query, params).fetchone()
    return row_to_dict(row) if row else None


def fetchall_dict(conn: sqlite3.Connection, query: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    return [row_to_dict(row) for row in conn.execute(query, params).fetchall()]
