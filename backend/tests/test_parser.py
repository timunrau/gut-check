from datetime import datetime

from app.parser import fallback_parse_result, validate_model_output


def test_empty_model_events_for_portion_food_entry_gets_meal_event() -> None:
    result = validate_model_output(
        "This morning I at half cup of all bran buds, a bannana, and a handful of almonds",
        {
            "entry_classification": "unknown",
            "classification_confidence": 0.2,
            "events": [],
        },
        datetime.fromisoformat("2026-06-26T23:27:32-05:00"),
    )

    assert result.status == "parsed"
    assert result.classification == "meal"
    assert result.parser_error is None
    assert len(result.events) == 1
    assert result.events[0].event_type == "meal"
    assert result.events[0].time_was_defaulted is True
    assert result.events[0].data["foods"] == ["all bran buds", "banana", "almonds"]
    assert result.events[0].data["drinks"] == []


def test_parser_fallback_for_portion_food_entry_is_not_failed() -> None:
    result = fallback_parse_result(
        "This morning I at half cup of all bran buds, a bannana, and a handful of almonds",
        datetime.fromisoformat("2026-06-26T23:27:32-05:00"),
        "No useful events found",
    )

    assert result.status == "parsed"
    assert result.classification == "meal"
    assert result.parser_error is None
    assert result.events[0].event_type == "meal"
    assert result.events[0].data["foods"] == ["all bran buds", "banana", "almonds"]
    assert result.events[0].data["drinks"] == []


def test_parser_fallback_cleans_foods_and_splits_drinks() -> None:
    result = fallback_parse_result(
        "I ate half a cup of all bran buds, with a cup of milk, a bananna, and some almonds",
        datetime.fromisoformat("2026-06-26T23:37:32-05:00"),
        "No useful events found",
    )

    assert result.status == "parsed"
    assert result.events[0].data["foods"] == ["all bran buds", "banana", "almonds"]
    assert result.events[0].data["drinks"] == ["milk"]


def test_model_output_is_cleaned_after_ai_parse() -> None:
    result = validate_model_output(
        "I ate a bananna and a cup of milk",
        {
            "entry_classification": "meal",
            "classification_confidence": 0.8,
            "events": [
                {
                    "type": "meal",
                    "time": None,
                    "date_offset": 0,
                    "foods": ["bananna", "with a cup of milk"],
                    "drinks": [],
                    "confidence": 0.7,
                }
            ],
        },
        datetime.fromisoformat("2026-06-26T23:37:32-05:00"),
    )

    assert result.status == "parsed"
    assert result.events[0].data["foods"] == ["banana"]
    assert result.events[0].data["drinks"] == ["milk"]


def test_model_drink_list_is_validated_against_known_beverages() -> None:
    result = validate_model_output(
        "I ate all bran buds, milk, banana, and almonds",
        {
            "entry_classification": "meal",
            "classification_confidence": 0.9,
            "events": [
                {
                    "type": "meal",
                    "time": None,
                    "date_offset": 0,
                    "foods": ["all bran buds", "banana"],
                    "drinks": ["milk", "almonds"],
                    "confidence": 0.8,
                }
            ],
        },
        datetime.fromisoformat("2026-06-26T23:42:32-05:00"),
    )

    assert result.events[0].data["foods"] == ["all bran buds", "banana", "almonds"]
    assert result.events[0].data["drinks"] == ["milk"]


def test_bowel_movement_text_keeps_stool_details_when_model_misses_fields() -> None:
    result = validate_model_output(
        "I pooped. It was soft with small chunks and stinky",
        {
            "entry_classification": "bowel_movement",
            "classification_confidence": 0.8,
            "events": [
                {
                    "type": "bowel_movement",
                    "time": None,
                    "date_offset": 0,
                    "confidence": 0.7,
                }
            ],
        },
        datetime.fromisoformat("2026-06-27T00:39:03-05:00"),
    )

    assert result.status == "parsed"
    assert result.events[0].event_type == "bowel_movement"
    assert result.events[0].data["bristol"] == 5
    assert result.events[0].data["stool_form"] == "soft pieces"
    assert result.events[0].data["odor"] == "strong"


def test_context_string_from_model_is_preserved_as_note() -> None:
    result = validate_model_output(
        "really stressful day today",
        {
            "entry_classification": "context",
            "classification_confidence": 0.9,
            "events": [
                {
                    "type": "context",
                    "time": None,
                    "date_offset": 0,
                    "confidence": 0.9,
                    "context": "really stressful day today",
                }
            ],
        },
        datetime.fromisoformat("2026-06-27T01:08:47-05:00"),
    )

    assert result.status == "parsed"
    assert result.classification == "context"
    assert result.events[0].notes == "really stressful day today"
    assert result.events[0].data["context"] == {"note": "really stressful day today"}


def test_unknown_model_event_fields_are_preserved_as_ai_extra() -> None:
    result = validate_model_output(
        "felt rushed at work and had stomach pressure after lunch",
        {
            "entry_classification": "mixed",
            "classification_confidence": 0.8,
            "summary": "work stress and stomach pressure after lunch",
            "events": [
                {
                    "type": "symptom",
                    "time": None,
                    "date_offset": 0,
                    "confidence": 0.75,
                    "symptoms": [{"name": "stomach pressure"}],
                    "possible_context": "rushed at work",
                    "timeline": {"after": "lunch"},
                }
            ],
        },
        datetime.fromisoformat("2026-06-27T12:15:00-05:00"),
    )

    assert result.status == "parsed"
    assert result.events[0].data["symptoms"] == [
        {"name": "stomach pressure", "severity": None, "location": "stomach"}
    ]
    assert result.events[0].data["ai_extra"] == {
        "possible_context": "rushed at work",
        "timeline": {"after": "lunch"},
    }


def test_textual_pain_field_becomes_symptom_name() -> None:
    result = validate_model_output(
        "my stomach hurts",
        {
            "entry_classification": "symptom",
            "classification_confidence": 1.0,
            "summary": "stomach pain",
            "events": [
                {
                    "type": "symptom",
                    "time": None,
                    "date_offset": 0,
                    "confidence": 1.0,
                    "pain": "stomach",
                }
            ],
        },
        datetime.fromisoformat("2026-06-27T00:58:21-05:00"),
    )

    assert result.status == "parsed"
    assert result.events[0].event_type == "symptom"
    assert result.events[0].data["symptoms"] == [
        {"name": "stomach pain", "severity": None, "location": "stomach"}
    ]


def test_symptom_fallback_keeps_raw_pain_description() -> None:
    result = fallback_parse_result(
        "my stomach hurts",
        datetime.fromisoformat("2026-06-27T00:58:21-05:00"),
        "No useful events found",
    )

    assert result.status == "parsed"
    assert result.classification == "symptom"
    assert result.events[0].data["symptoms"] == [
        {"name": "stomach pain", "severity": None, "location": "stomach"}
    ]


def test_stomach_and_lower_stomach_pain_are_split_by_location() -> None:
    result = validate_model_output(
        "stomach is hurting a tiny bit, and my gut is also",
        {
            "entry_classification": "symptom",
            "classification_confidence": 0.9,
            "summary": "stomach pain and gut discomfort",
            "events": [
                {
                    "type": "symptom",
                    "time": None,
                    "date_offset": 0,
                    "confidence": 0.9,
                    "pain": "tiny",
                    "notes": "gut is also",
                }
            ],
        },
        datetime.fromisoformat("2026-06-27T01:20:29-05:00"),
    )

    assert result.status == "parsed"
    assert result.events[0].data["symptoms"] == [
        {"name": "stomach pain", "severity": 1, "location": "stomach"},
        {"name": "lower stomach pain", "severity": 1, "location": "lower_stomach"},
    ]


def test_bowel_movement_fallback_infers_hard_stool() -> None:
    result = fallback_parse_result(
        "I took a huge shit right now. It was hard as a rock and girthy",
        datetime.fromisoformat("2026-06-27T00:24:53-05:00"),
        "No useful events found",
    )

    assert result.status == "parsed"
    assert result.classification == "bowel_movement"
    assert result.events[0].data["bristol"] == 1
    assert result.events[0].data["stool_form"] == "hard lumps"
    assert result.events[0].data["amount"] == "large"
