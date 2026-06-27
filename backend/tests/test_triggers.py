from app.triggers import analyze_trigger_patterns


def _event(event_id: int, event_type: str, date: str, time: str, data: dict) -> dict:
    return {
        "id": event_id,
        "raw_log_id": event_id,
        "event_type": event_type,
        "event_date": date,
        "event_time": time,
        "time_was_defaulted": False,
        "notes": None,
        "confidence": 0.9,
        "data": data,
    }


def test_trigger_analysis_rewards_repeated_bad_outcomes_over_common_foods() -> None:
    events = [
        _event(1, "meal", "2026-06-01", "08:00", {"foods": ["milk", "rice"], "drinks": []}),
        _event(
            2,
            "symptom",
            "2026-06-01",
            "10:00",
            {"symptoms": [{"name": "lower stomach pain", "severity": 4, "location": "lower_stomach"}]},
        ),
        _event(3, "meal", "2026-06-02", "08:00", {"foods": ["milk"], "drinks": []}),
        _event(4, "bowel_movement", "2026-06-02", "18:00", {"bristol": 7}),
        _event(5, "meal", "2026-06-03", "08:00", {"foods": ["rice"], "drinks": []}),
        _event(6, "meal", "2026-06-04", "08:00", {"foods": ["milk"], "drinks": []}),
    ]

    result = analyze_trigger_patterns(events, days=60)

    candidates = {item["item"]: item for item in result["candidate_triggers"]}
    assert "milk" in candidates
    assert candidates["milk"]["bad_exposures"] == 2
    assert candidates["milk"]["tolerated_exposures"] == 1
    assert candidates["milk"]["strongest_outcome"] in {"lower stomach pain", "loose/watery BM"}
    assert "rice" not in candidates


def test_trigger_analysis_keeps_stomach_and_lower_stomach_pain_separate() -> None:
    events = [
        _event(1, "meal", "2026-06-01", "08:00", {"foods": ["milk"], "drinks": []}),
        _event(
            2,
            "symptom",
            "2026-06-01",
            "10:00",
            {"symptoms": [{"name": "stomach pain", "severity": 4, "location": "stomach"}]},
        ),
        _event(3, "meal", "2026-06-02", "08:00", {"foods": ["beans"], "drinks": []}),
        _event(
            4,
            "symptom",
            "2026-06-02",
            "10:00",
            {"symptoms": [{"name": "lower stomach pain", "severity": 4, "location": "lower_stomach"}]},
        ),
        _event(5, "meal", "2026-06-03", "08:00", {"foods": ["milk"], "drinks": []}),
        _event(
            6,
            "symptom",
            "2026-06-03",
            "10:00",
            {"symptoms": [{"name": "stomach pain", "severity": 4, "location": "stomach"}]},
        ),
        _event(7, "meal", "2026-06-04", "08:00", {"foods": ["beans"], "drinks": []}),
        _event(
            8,
            "symptom",
            "2026-06-04",
            "10:00",
            {"symptoms": [{"name": "lower stomach pain", "severity": 4, "location": "lower_stomach"}]},
        ),
    ]

    result = analyze_trigger_patterns(events, days=60)

    candidates = {item["item"]: item for item in result["candidate_triggers"]}
    assert candidates["milk"]["strongest_outcome"] == "stomach pain"
    assert candidates["beans"]["strongest_outcome"] == "lower stomach pain"
