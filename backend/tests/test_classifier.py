from app.classifier import classify_text


def test_portion_food_entry_with_typo_classifies_as_meal() -> None:
    classification, confidence, scores = classify_text(
        "This morning I at half cup of all bran buds, a bannana, and a handful of almonds"
    )

    assert classification == "meal"
    assert confidence >= 0.8
    assert scores["meal"] >= 2


def test_plain_language_stool_description_classifies_as_bowel_movement() -> None:
    classification, confidence, scores = classify_text(
        "I took a huge shit right now. It was hard as a rock and girthy"
    )

    assert classification == "bowel_movement"
    assert confidence >= 0.75
    assert scores["bowel_movement"] >= 2


def test_stomach_hurts_classifies_as_symptom() -> None:
    classification, confidence, scores = classify_text("my stomach hurts")

    assert classification == "symptom"
    assert confidence >= 0.75
    assert scores["symptom"] >= 2
