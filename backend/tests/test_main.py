from fastapi.testclient import TestClient

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
