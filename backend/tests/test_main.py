import json
from datetime import datetime

from fastapi.testclient import TestClient

from app import garmin
from app import main
from app.db import connect, fetchone_dict, init_db
from app.settings import Settings


def test_create_log_returns_pending_and_schedules_background_parse(tmp_path, monkeypatch) -> None:
    database_path = str(tmp_path / "gutcheck.db")
    init_db(database_path)
    settings = Settings(
        app_password="test-password",
        session_secret="test-secret",
        database_path=database_path,
        ollama_url="http://ollama.test",
        ollama_model="test-model",
    )
    scheduled_log_ids: list[int] = []

    main.app.dependency_overrides[main.settings_dep] = lambda: settings
    main.app.dependency_overrides[main.protected] = lambda: None
    monkeypatch.setattr(main, "_schedule_parse", lambda log_id, _settings: scheduled_log_ids.append(log_id))

    try:
        response = TestClient(main.app).post("/api/logs", json={"raw_text": "I ate rice and chicken"})
    finally:
        main.app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert data["parser_status"] == "pending"
    assert data["events"] == []
    assert data["new_events"] == []
    assert scheduled_log_ids == [data["id"]]

    conn = connect(database_path)
    try:
        log = fetchone_dict(conn, "SELECT * FROM raw_logs WHERE id = ?", (data["id"],))
    finally:
        conn.close()

    assert log is not None
    assert log["raw_text"] == "I ate rice and chicken"
    assert log["parser_status"] == "pending"


def test_patterns_payload_includes_period_events_and_garmin_without_raw_logs(tmp_path, monkeypatch) -> None:
    database_path = str(tmp_path / "gutcheck.db")
    init_db(database_path)
    settings = Settings(
        app_password="test-password",
        session_secret="test-secret",
        database_path=database_path,
        ollama_url="http://ollama.test",
        ollama_model="test-model",
    )

    conn = connect(database_path)
    try:
        cursor = conn.execute(
            """
            INSERT INTO raw_logs (
                raw_text, created_at, parser_status, model_name, parser_error,
                parsed_json, entry_classification, classification_confidence
            )
            VALUES (?, ?, 'parsed', ?, NULL, ?, 'meal', 0.95)
            """,
            ("ate rice then had a bad BM", "2026-06-26T08:00:00", "test-model", json.dumps({"events": []})),
        )
        log_id = cursor.lastrowid
        conn.execute(
            """
            INSERT INTO events (
                raw_log_id, event_type, event_date, event_time,
                time_was_defaulted, notes, confidence, data_json
            )
            VALUES (?, 'meal', '2026-06-26', '08:00', 0, NULL, 0.9, ?)
            """,
            (log_id, json.dumps({"foods": ["rice"], "drinks": []})),
        )
        conn.execute(
            """
            INSERT INTO events (
                raw_log_id, event_type, event_date, event_time,
                time_was_defaulted, notes, confidence, data_json
            )
            VALUES (?, 'bowel_movement', '2026-06-26', '13:00', 0, NULL, 0.9, ?)
            """,
            (log_id, json.dumps({"bristol": 7, "urgency": 5})),
        )
        garmin.upsert_metric(
            conn,
            garmin.GarminMetric(metric_date="2026-06-26", steps=1234, sleep_hours=7.5, sleep_score=83),
            "2026-06-26T23:00:00",
        )
        conn.commit()
    finally:
        conn.close()

    main.app.dependency_overrides[main.settings_dep] = lambda: settings
    main.app.dependency_overrides[main.protected] = lambda: None
    monkeypatch.setattr(main, "app_now", lambda _timezone: datetime(2026, 6, 27, 12, 0, 0))

    try:
        response = TestClient(main.app).get("/api/patterns?days=30")
    finally:
        main.app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert data["start_date"] == "2026-05-29"
    assert data["end_date"] == "2026-06-27"
    assert [event["event_type"] for event in data["events"]] == ["meal", "bowel_movement"]
    assert "logs" not in data
    assert "raw_text" not in json.dumps(data)
    assert data["garmin"]["days"][0]["metric_date"] == "2026-06-26"
    assert data["garmin"]["days"][0]["steps"] == 1234
    assert data["garmin"]["averages"]["avg_steps"] == 1234.0
