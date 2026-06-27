import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx

from .classifier import classify_text, normalize_classification
from .time_utils import parse_date_offset

logger = logging.getLogger(__name__)

VALID_EVENT_TYPES = {"meal", "bowel_movement", "symptom", "context"}
LIST_FIELDS = ("foods", "drinks", "meds", "supplements")
SEVERITY_FIELDS = ("urgency", "pain", "bloating", "gas", "stress")
PAIN_LOCATIONS = {
    "stomach": "stomach",
    "upper_stomach": "stomach",
    "upper abdomen": "stomach",
    "upper_abdomen": "stomach",
    "gut": "lower_stomach",
    "lower gut": "lower_stomach",
    "lower_gut": "lower_stomach",
    "lower stomach": "lower_stomach",
    "lower_stomach": "lower_stomach",
    "lower abdomen": "lower_stomach",
    "lower_abdomen": "lower_stomach",
    "lower abdominal": "lower_stomach",
    "abdomen": "abdomen",
    "abdominal": "abdomen",
}
MODEL_EVENT_FIELDS = {
    "type",
    "time",
    "date_offset",
    "confidence",
    "foods",
    "drinks",
    "meds",
    "supplements",
    "portion",
    "bristol",
    "urgency",
    "pain",
    "bloating",
    "gas",
    "stress",
    "sleep_hours",
    "stool_form",
    "amount",
    "odor",
    "symptoms",
    "pain_location",
    "pain_locations",
    "context",
    "notes",
}
KNOWN_DRINKS = {
    "coffee",
    "tea",
    "water",
    "soda",
    "milk",
    "juice",
    "beer",
    "wine",
}
TYPO_CORRECTIONS = {
    "bananna": "banana",
    "bannana": "banana",
    "cupt": "cup",
}
MEAL_LEAD_PATTERN = re.compile(
    r"\b(?:this morning|this afternoon|this evening|tonight|today|yesterday|breakfast|lunch|dinner|snack)\b",
    re.IGNORECASE,
)
MEAL_VERB_PATTERN = re.compile(r"\bi\s+(?:ate|at|had|drank)\b|\b(?:ate|had|drank)\b", re.IGNORECASE)
MEAL_QUANTITY_PATTERN = re.compile(
    r"\b(?:(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten|half|quarter|\d+(?:\.\d+)?|\d+/\d+)\s+(?:a\s+)?)?"
    r"(?:cups?|bowls?|plates?|servings?|handfuls?|slices?|pieces?|tbsp|tablespoons?|tsp|teaspoons?|ounces?|oz|grams?|ml|liters?|litres?)"
    r"\b(?:\s+of\b)?",
    re.IGNORECASE,
)
BRISTOL_LABELS = {
    1: "separate hard lumps",
    2: "lumpy sausage",
    3: "cracked sausage",
    4: "smooth soft sausage",
    5: "soft blobs",
    6: "mushy loose stool",
    7: "watery stool",
}
BOWEL_FORM_PATTERNS = (
    (re.compile(r"\b(?:watery|water|liquid|diarrhea|diarrhoea)\b", re.IGNORECASE), 7, "watery"),
    (re.compile(r"\b(?:loose|mushy|sloppy|runny)\b", re.IGNORECASE), 6, "loose/mushy"),
    (re.compile(r"\b(?:soft|small chunks?|chunks?|chunky|pieces?)\b", re.IGNORECASE), 5, "soft pieces"),
    (re.compile(r"\b(?:smooth|normal|formed)\b", re.IGNORECASE), 4, "formed"),
    (re.compile(r"\b(?:cracked)\b", re.IGNORECASE), 3, "cracked"),
    (re.compile(r"\b(?:hard|rock|pellets?|rabbit)\b", re.IGNORECASE), 1, "hard lumps"),
    (re.compile(r"\b(?:lumpy|girthy|large)\b", re.IGNORECASE), 2, "lumpy/large"),
)
AMOUNT_PATTERNS = (
    (re.compile(r"\b(?:tiny amount|small amount|little bit)\b", re.IGNORECASE), "small"),
    (re.compile(r"\b(?:medium|moderate|normal amount)\b", re.IGNORECASE), "medium"),
    (re.compile(r"\b(?:large amount|big|huge|massive|a lot)\b", re.IGNORECASE), "large"),
)
ODOR_PATTERN = re.compile(r"\b(?:stinky|smelly|foul|strong smell|bad smell)\b", re.IGNORECASE)
PAIN_TEXT_PATTERN = re.compile(r"\b(?:hurt|hurts|hurting|pain|ache|aches|aching|cramp|cramps|cramping)\b", re.IGNORECASE)
STOMACH_LOCATION_PATTERN = re.compile(r"\b(?:stomach|upper stomach|upper abdomen|upper abdominal)\b", re.IGNORECASE)
LOWER_STOMACH_LOCATION_PATTERN = re.compile(
    r"\b(?:gut|lower gut|lower stomach|lower abdomen|lower abdominal|intestines?|intestinal)\b",
    re.IGNORECASE,
)
SMALL_SEVERITY_PATTERN = re.compile(r"\b(?:tiny|small|little|mild)\b", re.IGNORECASE)


class ParserError(Exception):
    pass


@dataclass
class ParsedEvent:
    event_type: str
    event_date: str
    event_time: str
    time_was_defaulted: bool
    notes: str | None
    confidence: float
    data: dict[str, Any]


@dataclass
class ParseResult:
    status: str
    classification: str
    confidence: float
    parsed_json: dict[str, Any] | None
    parser_error: str | None
    events: list[ParsedEvent]


SYSTEM_PROMPT = """You are an IBS log classifier and parser.

Classify the entry as one of:
meal, bowel_movement, symptom, context, mixed, unknown.

Rules:
* Return JSON only.
* Keep the JSON compact.
* Correct obvious spelling and dictation errors in structured fields.
* Do not invent facts.
* Do not assume meals caused symptoms.
* Do not assume symptoms are related to meals.
* If time is missing, return null. The backend will use logged time.
* Omit optional fields when unknown.
* Bristol must be 1-7 or null.
* Severity values must be 1-5 or null.
* Keep foods, drinks, meds, and supplements as clean lowercase names, without quantities or filler words.
* Put beverages in drinks, not foods.
* Add a short summary explaining what you understood from the entry.
* If useful text does not fit a structured field, put it in notes instead of dropping it.
* For pain symptoms, return symptoms as objects with name, location, and severity when known.
* Use location "stomach" for stomach/upper-stomach pain and "lower_stomach" for gut/lower-stomach/lower-abdomen pain.
* Do not give medical advice.
* Return only one JSON object.
"""


def build_prompt(raw_text: str) -> str:
    return f"""Parse this voice note into compact structured JSON.

Example:
Voice note: I at half cup of all bran buds, a bannana, and a cup of milk
JSON:
{{
  "entry_classification": "meal",
  "classification_confidence": 0.9,
  "summary": "meal with all bran buds, banana, and milk",
  "events": [
    {{
      "type": "meal",
      "time": null,
      "date_offset": 0,
      "foods": ["all bran buds", "banana"],
      "drinks": ["milk"],
      "portion": null,
      "confidence": 0.8
    }}
  ]
}}

Return this shape:
{{
  "entry_classification": "meal|bowel_movement|symptom|context|mixed|unknown",
  "classification_confidence": 0.0-1.0,
  "summary": "one short plain-language interpretation",
  "events": [
    {{
      "type": "meal|bowel_movement|symptom|context",
      "time": "HH:MM or null if missing",
      "date_offset": 0,
      "confidence": 0.0-1.0,
      "...": "only relevant optional fields"
    }}
  ]
}}

Optional event fields:
foods, drinks, meds, supplements, portion, bristol, urgency, pain, bloating, gas,
stress, sleep_hours, symptoms, pain_location, pain_locations, context, stool_form,
amount, odor, notes.

Voice note:
{raw_text}"""


async def call_ollama(
    raw_text: str,
    ollama_url: str,
    model: str,
    num_ctx: int,
    num_predict: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(raw_text)},
        ],
        "format": "json",
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_ctx": num_ctx, "num_predict": num_predict},
    }
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        response = await client.post(f"{ollama_url}/api/chat", json=payload)
        response.raise_for_status()
    data = response.json()
    content = data.get("message", {}).get("content", "")
    return load_json_object(content)


def load_json_object(content: str) -> dict[str, Any]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ParserError("Model returned malformed JSON") from None
        try:
            parsed = json.loads(content[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ParserError("Model returned malformed JSON") from exc
    if not isinstance(parsed, dict):
        raise ParserError("Model JSON was not an object")
    return parsed


def _none_if_empty(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    if isinstance(value, list):
        return [_none_if_empty(item) for item in value]
    if isinstance(value, dict):
        return {key: _none_if_empty(item) for key, item in value.items()}
    return value


def _safe_model_value(value: Any) -> Any:
    cleaned = _none_if_empty(value)
    if cleaned is None:
        return None
    if isinstance(cleaned, (str, int, float, bool)):
        return cleaned
    if isinstance(cleaned, list):
        values = [_safe_model_value(item) for item in cleaned]
        return [item for item in values if item is not None]
    if isinstance(cleaned, dict):
        values = {str(key): _safe_model_value(item) for key, item in cleaned.items()}
        return {key: item for key, item in values.items() if item is not None}
    return str(cleaned)


def _correct_typos(text: str) -> str:
    words = [TYPO_CORRECTIONS.get(word, word) for word in text.split()]
    return " ".join(words)


def _normalize_name(value: str) -> str | None:
    name = value.strip().lower()
    name = _correct_typos(name)
    name = MEAL_QUANTITY_PATTERN.sub(" ", name)
    name = re.sub(r"^(?:\s*(?:with|of|a|an|some|about)\s+)+", "", name)
    name = re.sub(r"\s+", " ", name).strip(" .,")
    return name or None


def _clean_names(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned = []
    seen = set()
    for item in value:
        if isinstance(item, str):
            name = _normalize_name(item)
            if name and name not in seen:
                cleaned.append(name)
                seen.add(name)
    return cleaned


def _split_known_drinks(data: dict[str, Any]) -> None:
    foods = []
    drinks = []
    drink_set = set()
    for item in data.get("foods") or []:
        if item in KNOWN_DRINKS:
            if item not in drink_set:
                drinks.append(item)
                drink_set.add(item)
        elif item not in foods:
            foods.append(item)
    for item in data.get("drinks") or []:
        if item in KNOWN_DRINKS:
            if item not in drink_set:
                drinks.append(item)
                drink_set.add(item)
        elif item not in foods:
            foods.append(item)
    data["foods"] = foods
    data["drinks"] = drinks


def _clamp_number(value: Any, low: float, high: float, integer: bool = True) -> int | float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number < low or number > high:
        return None
    return int(number) if integer else number


def _severity_value(value: Any) -> int | None:
    number = _clamp_number(value, 1, 5)
    if number is not None:
        return number
    if isinstance(value, bool):
        return 3 if value else None
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in {"yes", "y", "true", "present", "urgent", "severe"}:
            return 3
        if cleaned in {"no", "n", "false", "none", "absent"}:
            return None
    return None


def _confidence(value: Any) -> float:
    if isinstance(value, bool):
        return 0.7
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.7
    return max(0.0, min(1.0, number))


def _event_time(value: Any, logged_at) -> tuple[str, bool]:
    if isinstance(value, str) and re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", value.strip()):
        return value.strip(), False
    return logged_at.strftime("%H:%M"), True


def _normalize_symptom_location(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = _normalize_text_field(value)
    if not cleaned:
        return None
    return PAIN_LOCATIONS.get(cleaned, cleaned)


def _pain_severity_from_text(value: str) -> int | None:
    if SMALL_SEVERITY_PATTERN.search(value):
        return 1
    return None


def _pain_event_object(raw_event: dict[str, Any]) -> dict[str, Any]:
    pain = raw_event.get("pain")
    return pain if isinstance(pain, dict) else {}


def _symptom_display_name(location: str | None, fallback: str = "pain") -> str:
    if location == "stomach":
        return "stomach pain"
    if location == "lower_stomach":
        return "lower stomach pain"
    if location == "abdomen":
        return "abdominal pain"
    return fallback


def _normalize_symptoms(value: Any) -> list[dict[str, Any]]:
    normalized = []
    if not isinstance(value, list):
        return normalized
    for item in value:
        if isinstance(item, str):
            name = _normalize_name(item)
            if name:
                symptom = {"name": name, "severity": None}
                location = _infer_location_from_text(name)
                if location:
                    symptom["location"] = location
                normalized.append(symptom)
        elif isinstance(item, dict):
            name = item.get("name")
            location = _normalize_symptom_location(item.get("location") or item.get("area"))
            if isinstance(name, str) and name.strip():
                normalized_name = _normalize_name(name)
                if not normalized_name:
                    continue
                if location is None:
                    location = _infer_location_from_text(normalized_name)
                if location and normalized_name in {"pain", "hurt", "hurts", "ache", "aches", "cramp", "cramps"}:
                    normalized_name = _symptom_display_name(location)
                symptom = {
                    "name": normalized_name,
                    "severity": _clamp_number(item.get("severity"), 1, 5),
                }
                if location:
                    symptom["location"] = location
                normalized.append(symptom)
    return normalized


def _infer_location_from_text(value: str) -> str | None:
    if LOWER_STOMACH_LOCATION_PATTERN.search(value):
        return "lower_stomach"
    if STOMACH_LOCATION_PATTERN.search(value):
        return "stomach"
    return None


def _add_symptom(
    symptoms: list[dict[str, Any]],
    name: str,
    severity: int | None = None,
    location: str | None = None,
) -> None:
    normalized_name = _normalize_name(name)
    if not normalized_name:
        return
    normalized_location = location or _infer_location_from_text(normalized_name)
    if any(
        (
            symptom.get("name") == normalized_name
            or (
                normalized_location is not None
                and symptom.get("location") == normalized_location
                and "pain" in str(symptom.get("name") or "")
                and "pain" in normalized_name
            )
        )
        and symptom.get("location") == normalized_location
        for symptom in symptoms
    ):
        return
    symptom = {"name": normalized_name, "severity": severity}
    if normalized_location:
        symptom["location"] = normalized_location
    symptoms.append(symptom)


def _symptom_name_from_text_field(value: Any, label: str) -> str | None:
    if isinstance(value, dict):
        name = value.get("name") or value.get("type")
        if isinstance(name, str) and name.strip():
            cleaned_name = _normalize_text_field(name)
            if cleaned_name and (label in cleaned_name or PAIN_TEXT_PATTERN.search(cleaned_name)):
                return cleaned_name
            return f"{cleaned_name} {label}" if cleaned_name else None
        return label
    if not isinstance(value, str):
        return None
    cleaned = _normalize_text_field(value)
    if not cleaned:
        return None
    if label in cleaned or PAIN_TEXT_PATTERN.search(cleaned):
        return cleaned
    return f"{cleaned} {label}"


def _pain_locations_from_event(raw_event: dict[str, Any]) -> list[str]:
    values = raw_event.get("pain_locations")
    if values is None:
        values = raw_event.get("pain_location")
    pain = _pain_event_object(raw_event)
    if values is None:
        values = pain.get("location") or pain.get("area")
    if not isinstance(values, list):
        values = [values]
    locations = []
    for value in values:
        location = _normalize_symptom_location(value)
        if location and location not in locations:
            locations.append(location)
    return locations


def _pain_locations_from_text(raw_text: str) -> list[str]:
    if not PAIN_TEXT_PATTERN.search(raw_text):
        return []
    locations = []
    if STOMACH_LOCATION_PATTERN.search(raw_text):
        locations.append("stomach")
    if LOWER_STOMACH_LOCATION_PATTERN.search(raw_text):
        locations.append("lower_stomach")
    return locations


def _enrich_symptom(raw_text: str, raw_event: dict[str, Any], data: dict[str, Any]) -> None:
    symptoms = data.get("symptoms") or []
    pain = _pain_event_object(raw_event)
    pain_name = _symptom_name_from_text_field(raw_event.get("pain"), "pain")
    raw_severity = _severity_value(pain.get("severity")) or _pain_severity_from_text(raw_text)
    event_locations = _pain_locations_from_event(raw_event)
    text_locations = _pain_locations_from_text(raw_text)
    locations = event_locations or text_locations
    if pain_name and not locations:
        _add_symptom(symptoms, pain_name, raw_severity)
    for location in locations:
        _add_symptom(symptoms, _symptom_display_name(location, pain_name or "pain"), raw_severity, location)
    if not symptoms and PAIN_TEXT_PATTERN.search(raw_text):
        _add_symptom(symptoms, raw_text, raw_severity)
    data["symptoms"] = symptoms


def _normalize_context(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        note = _normalize_text_field(value)
        return {"note": note} if note else {}
    if not isinstance(value, dict):
        return {}
    return {
        str(key): _none_if_empty(item)
        for key, item in value.items()
        if _none_if_empty(item) is not None
    }


def _model_extra(raw_event: dict[str, Any]) -> dict[str, Any]:
    extra: dict[str, Any] = {}
    for key, value in raw_event.items():
        normalized_key = str(key).strip().lower()
        if not normalized_key or normalized_key in MODEL_EVENT_FIELDS:
            continue
        cleaned = _safe_model_value(value)
        if cleaned is not None and cleaned != [] and cleaned != {}:
            extra[normalized_key] = cleaned
    return extra


def _normalize_text_field(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = re.sub(r"\s+", " ", value.strip().lower())
    return cleaned.strip(" .,") or None


def _empty_event_data() -> dict[str, Any]:
    return {
        "foods": [],
        "drinks": [],
        "meds": [],
        "supplements": [],
        "portion": None,
        "bristol": None,
        "urgency": None,
        "pain": None,
        "bloating": None,
        "gas": None,
        "stress": None,
        "sleep_hours": None,
        "stool_form": None,
        "amount": None,
        "odor": None,
        "symptoms": [],
        "context": {},
    }


def _simple_meal_items(raw_text: str) -> tuple[list[str], list[str]]:
    text = _correct_typos(raw_text.lower())
    text = MEAL_LEAD_PATTERN.sub(" ", text)
    text = MEAL_VERB_PATTERN.sub(" ", text)
    foods: list[str] = []
    drinks: list[str] = []
    for part in re.split(r",|\band\b|&|\+", text):
        item = _normalize_name(part)
        if item and item not in {"i", "of"}:
            if item in KNOWN_DRINKS:
                drinks.append(item)
            else:
                foods.append(item)
    return foods[:8], drinks[:8]


def _heuristic_meal_event(raw_text: str, logged_at) -> ParsedEvent:
    data = _empty_event_data()
    data["foods"], data["drinks"] = _simple_meal_items(raw_text)
    return ParsedEvent(
        event_type="meal",
        event_date=logged_at.date().isoformat(),
        event_time=logged_at.strftime("%H:%M"),
        time_was_defaulted=True,
        notes=None if data["foods"] else raw_text,
        confidence=0.55,
        data=data,
    )


def _bowel_details_from_text(raw_text: str) -> dict[str, Any]:
    details: dict[str, Any] = {}
    for pattern, bristol, label in BOWEL_FORM_PATTERNS:
        if pattern.search(raw_text):
            details["bristol"] = bristol
            details["stool_form"] = label
            details["bristol_description"] = BRISTOL_LABELS[bristol]
            break
    for pattern, amount in AMOUNT_PATTERNS:
        if pattern.search(raw_text):
            details["amount"] = amount
            break
    if ODOR_PATTERN.search(raw_text):
        details["odor"] = "strong"
    return details


def _enrich_bowel_movement(raw_text: str, data: dict[str, Any]) -> None:
    inferred = _bowel_details_from_text(raw_text)
    for key, value in inferred.items():
        if data.get(key) is None:
            data[key] = value


def _heuristic_bowel_event(raw_text: str, logged_at) -> ParsedEvent:
    data = _empty_event_data()
    _enrich_bowel_movement(raw_text, data)
    return ParsedEvent(
        event_type="bowel_movement",
        event_date=logged_at.date().isoformat(),
        event_time=logged_at.strftime("%H:%M"),
        time_was_defaulted=True,
        notes=None if data.get("stool_form") else raw_text,
        confidence=0.45 if data.get("stool_form") else 0.35,
        data=data,
    )


def _heuristic_symptom_event(raw_text: str, logged_at) -> ParsedEvent:
    data = _empty_event_data()
    _enrich_symptom(raw_text, {}, data)
    return ParsedEvent(
        event_type="symptom",
        event_date=logged_at.date().isoformat(),
        event_time=logged_at.strftime("%H:%M"),
        time_was_defaulted=True,
        notes=None if data["symptoms"] else raw_text,
        confidence=0.4 if data["symptoms"] else 0.35,
        data=data,
    )


def validate_model_output(raw_text: str, model_json: dict[str, Any], logged_at) -> ParseResult:
    fallback_classification, fallback_confidence, _scores = classify_text(raw_text)
    classification = normalize_classification(model_json.get("entry_classification"), fallback_classification)
    confidence = _confidence(model_json.get("classification_confidence", fallback_confidence))
    if classification == fallback_classification and model_json.get("entry_classification") == "unknown":
        confidence = fallback_confidence

    raw_events = model_json.get("events", [])
    if not isinstance(raw_events, list):
        raw_events = []

    events: list[ParsedEvent] = []
    for raw_event in raw_events:
        if not isinstance(raw_event, dict):
            continue
        event_type = str(raw_event.get("type", "")).strip().lower()
        if event_type not in VALID_EVENT_TYPES:
            continue

        cleaned = _none_if_empty(raw_event)
        data: dict[str, Any] = {}
        for field in LIST_FIELDS:
            data[field] = _clean_names(cleaned.get(field))
        _split_known_drinks(data)
        data["portion"] = cleaned.get("portion") if isinstance(cleaned.get("portion"), str) else None
        data["bristol"] = _clamp_number(cleaned.get("bristol"), 1, 7)
        for field in SEVERITY_FIELDS:
            if field == "pain" and isinstance(cleaned.get(field), dict):
                data[field] = _severity_value(cleaned[field].get("severity"))
            else:
                data[field] = _severity_value(cleaned.get(field))
        data["sleep_hours"] = _clamp_number(cleaned.get("sleep_hours"), 0, 24, integer=False)
        data["stool_form"] = _normalize_text_field(cleaned.get("stool_form"))
        data["amount"] = _normalize_text_field(cleaned.get("amount"))
        data["odor"] = _normalize_text_field(cleaned.get("odor"))
        data["symptoms"] = _normalize_symptoms(cleaned.get("symptoms"))
        data["context"] = _normalize_context(cleaned.get("context"))
        extra = _model_extra(cleaned)
        if extra:
            data["ai_extra"] = extra
        if event_type == "bowel_movement":
            _enrich_bowel_movement(raw_text, data)
        elif event_type == "symptom":
            _enrich_symptom(raw_text, cleaned, data)

        event_time, defaulted = _event_time(cleaned.get("time"), logged_at)
        notes = cleaned.get("notes") if isinstance(cleaned.get("notes"), str) else None
        if notes is None and event_type == "context":
            context_note = data["context"].get("note")
            if isinstance(context_note, str):
                notes = context_note
        events.append(
            ParsedEvent(
                event_type=event_type,
                event_date=parse_date_offset(logged_at, cleaned.get("date_offset")),
                event_time=event_time,
                time_was_defaulted=defaulted,
                notes=notes,
                confidence=_confidence(cleaned.get("confidence")),
                data=data,
            )
        )

    if not events:
        if classification == "meal":
            events.append(_heuristic_meal_event(raw_text, logged_at))
        elif classification == "bowel_movement":
            events.append(_heuristic_bowel_event(raw_text, logged_at))
        elif classification == "symptom":
            events.append(_heuristic_symptom_event(raw_text, logged_at))
        else:
            raise ParserError("No useful events found")

    return ParseResult(
        status="parsed",
        classification=classification,
        confidence=confidence,
        parsed_json=model_json,
        parser_error=None,
        events=events,
    )


def fallback_parse_result(raw_text: str, logged_at, error: str) -> ParseResult:
    classification, confidence, _scores = classify_text(raw_text)
    events: list[ParsedEvent] = []
    if classification in VALID_EVENT_TYPES:
        if classification == "meal":
            events.append(_heuristic_meal_event(raw_text, logged_at))
        elif classification == "bowel_movement":
            events.append(_heuristic_bowel_event(raw_text, logged_at))
        elif classification == "symptom":
            events.append(_heuristic_symptom_event(raw_text, logged_at))
        else:
            events.append(
                ParsedEvent(
                    event_type=classification,
                    event_date=logged_at.date().isoformat(),
                    event_time=logged_at.strftime("%H:%M"),
                    time_was_defaulted=True,
                    notes=raw_text,
                    confidence=0.35,
                    data=_empty_event_data(),
                )
            )
        return ParseResult(
            status="parsed",
            classification=classification,
            confidence=confidence,
            parsed_json=None,
            parser_error=None,
            events=events,
        )
    return ParseResult(
        status="failed",
        classification=classification,
        confidence=confidence,
        parsed_json=None,
        parser_error=error,
        events=events,
    )


async def parse_entry(
    raw_text: str,
    logged_at,
    ollama_url: str,
    model: str,
    num_ctx: int = 4096,
    num_predict: int = 1024,
    timeout_seconds: float = 60.0,
) -> ParseResult:
    started_at = time.monotonic()
    try:
        model_json = await call_ollama(raw_text, ollama_url, model, num_ctx, num_predict, timeout_seconds)
        result = validate_model_output(raw_text, model_json, logged_at)
        logger.info(
            "ollama_parse_complete model=%s status=%s elapsed=%.2fs events=%s",
            model,
            result.status,
            time.monotonic() - started_at,
            len(result.events),
        )
        return result
    except (ParserError, httpx.HTTPError, json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            "ollama_parse_fallback model=%s elapsed=%.2fs error=%s",
            model,
            time.monotonic() - started_at,
            str(exc) or exc.__class__.__name__,
        )
        return fallback_parse_result(raw_text, logged_at, str(exc) or exc.__class__.__name__)
