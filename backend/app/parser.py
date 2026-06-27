import json
import re
from dataclasses import dataclass
from typing import Any

import httpx

from .classifier import classify_text, normalize_classification
from .time_utils import parse_date_offset

VALID_EVENT_TYPES = {"meal", "bowel_movement", "symptom", "context"}
LIST_FIELDS = ("foods", "drinks", "meds", "supplements")
SEVERITY_FIELDS = ("urgency", "pain", "bloating", "gas", "stress")
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
* Correct obvious spelling and dictation errors in structured fields.
* Do not invent facts.
* Do not assume meals caused symptoms.
* Do not assume symptoms are related to meals.
* If time is missing, return null. The backend will use logged time.
* Use null for unknown values.
* Bristol must be 1-7 or null.
* Severity values must be 1-5 or null.
* Keep foods, drinks, meds, and supplements as clean lowercase names, without quantities or filler words.
* Put beverages in drinks, not foods.
* Do not give medical advice.
* Return only one JSON object matching the requested shape.
"""


def build_prompt(raw_text: str) -> str:
    return f"""Parse this voice note into clean structured JSON.

Example:
Voice note: I at half cup of all bran buds, a bannana, and a cup of milk
JSON:
{{
  "entry_classification": "meal",
  "classification_confidence": 0.9,
  "events": [
    {{
      "type": "meal",
      "time": null,
      "date_offset": 0,
      "foods": ["all bran buds", "banana"],
      "drinks": ["milk"],
      "meds": [],
      "supplements": [],
      "portion": null,
      "bristol": null,
      "urgency": null,
      "pain": null,
      "bloating": null,
      "gas": null,
      "stress": null,
      "sleep_hours": null,
      "symptoms": [],
      "context": {{}},
      "notes": null,
      "confidence": 0.8
    }}
  ]
}}

Return:
{{
  "entry_classification": "...",
  "classification_confidence": 0.0,
  "events": [
    {{
      "type": "meal | bowel_movement | symptom | context",
      "time": "HH:MM or null",
      "date_offset": 0,
      "foods": [],
      "drinks": [],
      "meds": [],
      "supplements": [],
      "portion": null,
      "bristol": null,
      "urgency": null,
      "pain": null,
      "bloating": null,
      "gas": null,
      "stress": null,
      "sleep_hours": null,
      "symptoms": [],
      "context": {{}},
      "notes": null,
      "confidence": 0.7
    }}
  ]
}}

Voice note:
{raw_text}"""


async def call_ollama(raw_text: str, ollama_url: str, model: str, num_ctx: int) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(raw_text)},
        ],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0, "num_ctx": num_ctx},
    }
    async with httpx.AsyncClient(timeout=300.0) as client:
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


def _normalize_symptoms(value: Any) -> list[dict[str, Any]]:
    normalized = []
    if not isinstance(value, list):
        return normalized
    for item in value:
        if isinstance(item, str):
            name = _normalize_name(item)
            if name:
                normalized.append({"name": name, "severity": None})
        elif isinstance(item, dict):
            name = item.get("name")
            if isinstance(name, str) and name.strip():
                normalized_name = _normalize_name(name)
                if not normalized_name:
                    continue
                normalized.append(
                    {
                        "name": normalized_name,
                        "severity": _clamp_number(item.get("severity"), 1, 5),
                    }
                )
    return normalized


def _normalize_context(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): _none_if_empty(item)
        for key, item in value.items()
        if _none_if_empty(item) is not None
    }


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
            data[field] = _clamp_number(cleaned.get(field), 1, 5)
        data["sleep_hours"] = _clamp_number(cleaned.get("sleep_hours"), 0, 24, integer=False)
        data["symptoms"] = _normalize_symptoms(cleaned.get("symptoms"))
        data["context"] = _normalize_context(cleaned.get("context"))

        event_time, defaulted = _event_time(cleaned.get("time"), logged_at)
        events.append(
            ParsedEvent(
                event_type=event_type,
                event_date=parse_date_offset(logged_at, cleaned.get("date_offset")),
                event_time=event_time,
                time_was_defaulted=defaulted,
                notes=cleaned.get("notes") if isinstance(cleaned.get("notes"), str) else None,
                confidence=_confidence(cleaned.get("confidence")),
                data=data,
            )
        )

    if not events:
        if classification == "meal":
            events.append(_heuristic_meal_event(raw_text, logged_at))
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


async def parse_entry(raw_text: str, logged_at, ollama_url: str, model: str, num_ctx: int = 4096) -> ParseResult:
    try:
        model_json = await call_ollama(raw_text, ollama_url, model, num_ctx)
        return validate_model_output(raw_text, model_json, logged_at)
    except (ParserError, httpx.HTTPError, json.JSONDecodeError, ValueError) as exc:
        return fallback_parse_result(raw_text, logged_at, str(exc) or exc.__class__.__name__)
