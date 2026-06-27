from app.classifier import classify_text


def test_portion_food_entry_with_typo_classifies_as_meal() -> None:
    classification, confidence, scores = classify_text(
        "This morning I at half cup of all bran buds, a bannana, and a handful of almonds"
    )

    assert classification == "meal"
    assert confidence >= 0.8
    assert scores["meal"] >= 2
