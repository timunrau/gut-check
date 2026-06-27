from pathlib import Path

from fastapi.testclient import TestClient

from app import garmin, main
from app.db import connect, fetchone_dict, init_db
from app.settings import Settings


class FakeTokenClient:
    def dump(self, tokenstore: str) -> None:
        path = Path(tokenstore)
        path.mkdir(parents=True, exist_ok=True)
        (path / "garmin_tokens.json").write_text("{}", encoding="utf-8")


class FakeGarmin:
    def __init__(self, email=None, password=None, is_cn=False, return_on_mfa=False, **_kwargs):
        self.email = email
        self.password = password
        self.return_on_mfa = return_on_mfa
        self.client = FakeTokenClient()

    def login(self, tokenstore=None):
        if tokenstore is not None:
            if not Path(tokenstore).exists():
                raise FileNotFoundError(tokenstore)
            return None
        if self.return_on_mfa and self.password == "mfa-password":
            return ("needs_mfa", {"ticket": "fake"})
        return None

    def resume_login(self, context, code):
        if context != {"ticket": "fake"}:
            raise RuntimeError("bad context")
        if code != "123456":
            raise RuntimeError("bad code")

    def get_user_summary(self, day):
        return {"calendarDate": day, "totalSteps": 4321}

    def get_sleep_data(self, _day):
        return {
            "dailySleepDTO": {"sleepTimeSeconds": 27000},
            "sleepScores": {"overall": {"value": 82}},
        }

    def get_all_day_stress(self, _day):
        return {"avgStressLevel": 31, "maxStressLevel": 72}

    def get_body_battery(self, _start, _end):
        return [
            {
                "bodyBatteryVersion": 1,
                "charged": 22,
                "drained": 31,
                "bodyBatteryValuesArray": [["2026-06-27T08:00:00", 45], ["2026-06-27T22:00:00", 68]],
            }
        ]


def override_app(tmp_path, monkeypatch):
    database_path = str(tmp_path / "gutcheck.db")
    tokenstore = str(tmp_path / "garmin_tokens")
    init_db(database_path)
    settings = Settings(
        app_password="test-password",
        session_secret="test-secret",
        database_path=database_path,
        ollama_url="http://ollama.test",
        ollama_model="test-model",
        garmin_tokenstore=tokenstore,
    )
    main.app.dependency_overrides[main.settings_dep] = lambda: settings
    main.app.dependency_overrides[main.protected] = lambda: None
    monkeypatch.setattr(main, "_schedule_parse", lambda _log_id, _settings: None)
    garmin.set_garmin_factory(FakeGarmin)
    return settings


def cleanup_app():
    main.app.dependency_overrides.clear()
    garmin.set_garmin_factory(None)


def test_garmin_extractors_handle_known_payload_shapes() -> None:
    assert garmin.extract_steps({"totalSteps": 123}) == 123
    assert garmin.extract_sleep({"dailySleepDTO": {"sleepTimeSeconds": 28800}, "sleepScores": {"overall": {"value": 91}}}) == (8.0, 91.0)
    assert garmin.extract_stress({"stressValuesArray": [["t1", 10], ["t2", 30], ["t3", -1]]}) == (20.0, 30.0)
    assert garmin.extract_body_battery({"bodyBatteryValuesArray": [["t1", 20], ["t2", 80]]}) == (20.0, 80.0, 50.0, 80.0)
    assert garmin.extract_body_battery({
        "bodyBatteryVersion": 1,
        "charged": 22,
        "drained": 31,
        "bodyBatteryValuesArray": [["t1", 40], ["t2", 65]],
    }) == (40.0, 65.0, 52.5, 65.0)


def test_garmin_metric_upsert_and_payloads(tmp_path) -> None:
    database_path = str(tmp_path / "gutcheck.db")
    init_db(database_path)
    conn = connect(database_path)
    try:
        row = garmin.upsert_metric(
            conn,
            garmin.GarminMetric(
                metric_date="2026-06-27",
                steps=1234,
                sleep_hours=7.5,
                sleep_score=83,
                stress_avg=22,
                body_battery_avg=61,
            ),
            "2026-06-27T12:00:00",
        )
        conn.commit()
        assert row["steps"] == 1234

        day = main._day_payload(conn, "2026-06-27")
        assert day["garmin"]["sleep_hours"] == 7.5

        week_rows = garmin.metrics_between(conn, "2026-06-24", "2026-06-30")
        assert garmin.summarize_metrics(week_rows)["avg_steps"] == 1234.0
        assert garmin.summarize_metrics(week_rows)["avg_sleep_score"] == 83.0
    finally:
        conn.close()


def test_garmin_auth_mfa_test_and_sync_endpoints(tmp_path, monkeypatch) -> None:
    settings = override_app(tmp_path, monkeypatch)
    client = TestClient(main.app)
    try:
        start = client.post("/api/garmin/auth/start", json={"email": "me@example.com", "password": "mfa-password"})
        assert start.status_code == 200
        start_data = start.json()
        assert start_data["mfa_required"] is True

        finish = client.post("/api/garmin/auth/finish", json={"pending_id": start_data["pending_id"], "mfa_code": "123456"})
        assert finish.status_code == 200
        assert Path(settings.garmin_tokenstore, "garmin_tokens.json").exists()

        test = client.post("/api/garmin/test")
        assert test.status_code == 200
        assert test.json()["steps"] == 4321

        sync = client.post("/api/garmin/sync", json={"days": 2})
        assert sync.status_code == 200
        assert sync.json()["synced"] == 2

        status = client.get("/api/garmin/status")
        assert status.status_code == 200
        assert status.json()["connected"] is True

        conn = connect(settings.database_path)
        try:
            row = fetchone_dict(conn, "SELECT * FROM garmin_daily_metrics ORDER BY metric_date DESC LIMIT 1", ())
        finally:
            conn.close()
        assert row is not None
        assert row["steps"] == 4321
        assert row["sleep_hours"] == 7.5
        assert row["sleep_score"] == 82.0
        assert row["stress_avg"] == 31.0
        assert row["body_battery_end"] == 68.0
    finally:
        cleanup_app()


def test_garmin_auth_without_mfa_stores_tokens(tmp_path, monkeypatch) -> None:
    settings = override_app(tmp_path, monkeypatch)
    client = TestClient(main.app)
    try:
        response = client.post("/api/garmin/auth/start", json={"email": "me@example.com", "password": "plain-password"})
        assert response.status_code == 200
        assert response.json()["connected"] is True
        assert Path(settings.garmin_tokenstore, "garmin_tokens.json").exists()
    finally:
        cleanup_app()
