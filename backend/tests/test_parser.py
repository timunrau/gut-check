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
