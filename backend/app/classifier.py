import re

VALID_CLASSIFICATIONS = {
    "meal",
    "bowel_movement",
    "symptom",
    "context",
    "mixed",
    "unknown",
}

KEYWORDS = {
    "meal": [
        "ate",
        "had",
        "breakfast",
        "lunch",
        "dinner",
        "snack",
        "coffee",
        "tea",
        "water",
        "soda",
        "milk",
        "food",
        "meal",
        "drank",
    ],
    "bowel_movement": [
        "poop",
        "pooped",
        "shit",
        "shat",
        "stool",
        "feces",
        "fecal",
        "bm",
        "bowel",
        "bathroom",
        "diarrhea",
        "loose",
        "watery",
        "mushy",
        "soft",
        "hard",
        "rock",
        "chunks",
        "chunky",
        "constipation",
        "constipated",
        "urgent",
        "urgency",
        "bristol",
    ],
    "symptom": [
        "cramp",
        "cramps",
        "cramping",
        "bloated",
        "bloating",
        "gas",
        "gassy",
        "pain",
        "hurt",
        "hurts",
        "ache",
        "aches",
        "stomach",
        "belly",
        "abdomen",
        "abdominal",
        "nausea",
        "reflux",
        "heartburn",
        "discomfort",
        "fatigue",
    ],
    "context": [
        "stress",
        "stressed",
        "sleep",
        "slept",
        "tired",
        "exercise",
        "walk",
        "workout",
        "run",
        "period",
        "hormones",
        "medication",
        "meds",
        "supplement",
        "sick",
        "travel",
    ],
}

MEAL_PORTION_PATTERN = re.compile(
    r"\b(?:(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten|half|quarter|\d+(?:\.\d+)?|\d+/\d+)\s+)?"
    r"(?:cups?|bowls?|plates?|servings?|handfuls?|slices?|pieces?|tbsp|tablespoons?|tsp|teaspoons?|ounces?|oz|grams?|ml|liters?|litres?)"
    r"\b"
)


def _keyword_hit(text: str, keyword: str) -> bool:
    if " " in keyword:
        return keyword in text
    return re.search(rf"\b{re.escape(keyword)}\b", text) is not None


def _meal_portion_hits(text: str) -> int:
    return len(MEAL_PORTION_PATTERN.findall(text))


def classify_text(raw_text: str) -> tuple[str, float, dict[str, int]]:
    text = raw_text.lower()
    scores = {
        category: sum(1 for keyword in keywords if _keyword_hit(text, keyword))
        for category, keywords in KEYWORDS.items()
    }
    scores["meal"] += _meal_portion_hits(text)
    matched = [category for category, score in scores.items() if score > 0]

    if not matched:
        return "unknown", 0.2, scores
    if len(matched) > 1:
        total = sum(scores.values())
        confidence = min(0.9, 0.55 + (total * 0.08))
        return "mixed", confidence, scores

    category = matched[0]
    confidence = min(0.95, 0.65 + (scores[category] * 0.1))
    return category, confidence, scores


def normalize_classification(value: object, fallback: str) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in VALID_CLASSIFICATIONS:
            if normalized == "unknown" and fallback != "unknown":
                return fallback
            return normalized
    return fallback
