from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

TIME_WINDOWS = (
    ("0-3h", 0, 3),
    ("3-12h", 3, 12),
    ("12-24h", 12, 24),
    ("24-48h", 24, 48),
)
PAIN_WORDS = ("pain", "cramp", "cramps", "cramping", "stomach", "gut", "ache")


def _event_dt(event: dict[str, Any]) -> datetime:
    return datetime.fromisoformat(f"{event['event_date']}T{event['event_time']}:00")


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _meal_items(event: dict[str, Any]) -> list[str]:
    data = event.get("data") or {}
    items = []
    seen = set()
    for item in data.get("foods", []) + data.get("drinks", []):
        if isinstance(item, str):
            normalized = item.strip().lower()
            if normalized and normalized not in seen:
                items.append(normalized)
                seen.add(normalized)
    return items


def _pain_outcome_label(symptom: dict[str, Any], name: str) -> str:
    location = str(symptom.get("location") or "").lower()
    if location == "stomach":
        return "stomach pain"
    if location == "lower_stomach":
        return "lower stomach pain"
    if "lower stomach" in name or "lower abdomen" in name or "lower gut" in name or "gut" in name:
        return "lower stomach pain"
    if "stomach" in name:
        return "stomach pain"
    return name or "pain"


def _outcome_labels(event: dict[str, Any]) -> list[str]:
    data = event.get("data") or {}
    labels = []
    if event["event_type"] == "bowel_movement":
        bristol = _number(data.get("bristol"))
        if bristol in (6, 7):
            labels.append("loose/watery BM")
        elif bristol in (1, 2):
            labels.append("hard BM")
        for key, label in (
            ("urgency", "urgent BM"),
            ("pain", "painful BM"),
            ("bloating", "bloating with BM"),
            ("gas", "gas with BM"),
        ):
            value = _number(data.get(key))
            if value is not None and value >= 4:
                labels.append(label)
    elif event["event_type"] == "symptom":
        symptoms = data.get("symptoms") or []
        for symptom in symptoms:
            if not isinstance(symptom, dict):
                continue
            name = str(symptom.get("name") or "").lower()
            severity = _number(symptom.get("severity"))
            if any(word in name for word in PAIN_WORDS) and (severity is None or severity >= 3):
                labels.append(_pain_outcome_label(symptom, name))
            elif severity is not None and severity >= 4 and name:
                labels.append(name)
    return labels


def _window_for(delta_hours: float) -> str | None:
    for label, low, high in TIME_WINDOWS:
        if low <= delta_hours < high:
            return label
    return None


def _confidence(exposures: int, bad_exposures: int, lift: float) -> str:
    if exposures >= 6 and bad_exposures >= 3 and lift >= 0.25:
        return "stronger"
    if exposures >= 4 and bad_exposures >= 2 and lift >= 0.15:
        return "medium"
    return "low"


def analyze_trigger_patterns(events: list[dict[str, Any]], days: int) -> dict[str, Any]:
    meal_events = [event for event in events if event["event_type"] == "meal" and _meal_items(event)]
    outcome_events = []
    for event in events:
        labels = _outcome_labels(event)
        if labels:
            outcome_events.append({**event, "outcome_labels": labels, "dt": _event_dt(event)})

    item_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "item": "",
            "exposures": 0,
            "bad_exposures": 0,
            "tolerated_exposures": 0,
            "outcomes": Counter(),
            "windows": Counter(),
            "evidence": [],
        }
    )
    total_exposures = 0
    total_bad_exposures = 0

    for meal in meal_events:
        meal_dt = _event_dt(meal)
        meal_bad_outcomes = []
        for outcome in outcome_events:
            delta_hours = (outcome["dt"] - meal_dt).total_seconds() / 3600
            window = _window_for(delta_hours)
            if window:
                meal_bad_outcomes.append((outcome, window, delta_hours))

        meal_had_bad_outcome = bool(meal_bad_outcomes)
        total_exposures += 1
        if meal_had_bad_outcome:
            total_bad_exposures += 1

        for item in _meal_items(meal):
            stats = item_stats[item]
            stats["item"] = item
            stats["exposures"] += 1
            if meal_had_bad_outcome:
                stats["bad_exposures"] += 1
                closest_outcome, window, delta_hours = sorted(meal_bad_outcomes, key=lambda pair: pair[2])[0]
                stats["windows"][window] += 1
                for label in closest_outcome["outcome_labels"]:
                    stats["outcomes"][label] += 1
                if len(stats["evidence"]) < 3:
                    stats["evidence"].append(
                        {
                            "meal_event_id": meal["id"],
                            "outcome_event_id": closest_outcome["id"],
                            "meal_at": meal_dt.isoformat(timespec="minutes"),
                            "outcome_at": closest_outcome["dt"].isoformat(timespec="minutes"),
                            "window": window,
                            "hours_after": round(delta_hours, 1),
                            "outcomes": closest_outcome["outcome_labels"],
                        }
                    )

    baseline_rate = (total_bad_exposures / total_exposures) if total_exposures else 0.0
    candidates = []
    for stats in item_stats.values():
        exposures = stats["exposures"]
        bad_exposures = stats["bad_exposures"]
        stats["tolerated_exposures"] = exposures - bad_exposures
        bad_rate = bad_exposures / exposures if exposures else 0.0
        lift = bad_rate - baseline_rate
        if exposures < 2 or bad_exposures < 1:
            continue
        if lift <= 0 and bad_exposures < 2:
            continue
        top_window = stats["windows"].most_common(1)
        top_outcome = stats["outcomes"].most_common(1)
        confidence = _confidence(exposures, bad_exposures, lift)
        candidates.append(
            {
                "item": stats["item"],
                "exposures": exposures,
                "bad_exposures": bad_exposures,
                "tolerated_exposures": stats["tolerated_exposures"],
                "bad_rate": round(bad_rate, 2),
                "baseline_bad_rate": round(baseline_rate, 2),
                "lift": round(lift, 2),
                "confidence": confidence,
                "strongest_window": top_window[0][0] if top_window else None,
                "strongest_outcome": top_outcome[0][0] if top_outcome else None,
                "evidence": stats["evidence"],
                "language": _candidate_language(stats["item"], bad_exposures, exposures, confidence, top_window, top_outcome),
            }
        )

    candidates.sort(
        key=lambda item: (
            {"stronger": 2, "medium": 1, "low": 0}[item["confidence"]],
            item["lift"],
            item["bad_exposures"],
            item["bad_rate"],
        ),
        reverse=True,
    )
    return {
        "days": days,
        "counts": {
            "meal_exposures": total_exposures,
            "bad_outcome_events": len(outcome_events),
            "baseline_bad_rate": round(baseline_rate, 2),
        },
        "candidate_triggers": candidates[:10],
        "summary": _summary(candidates, total_exposures, len(outcome_events)),
        "note": "candidate patterns only; not medical proof or diagnosis",
    }


def _candidate_language(
    item: str,
    bad_exposures: int,
    exposures: int,
    confidence: str,
    top_window: list[tuple[str, int]],
    top_outcome: list[tuple[str, int]],
) -> str:
    window = f" mostly within {top_window[0][0]}" if top_window else ""
    outcome = f" for {top_outcome[0][0]}" if top_outcome else ""
    return f"{item} is a {confidence} candidate{outcome}: {bad_exposures}/{exposures} exposures were followed by symptoms{window}."


def _summary(candidates: list[dict[str, Any]], meal_exposures: int, bad_outcomes: int) -> str:
    if meal_exposures < 5 or bad_outcomes < 2:
        return "Not enough clean data yet. Keep logging meals, symptoms, and bowel movements."
    if not candidates:
        return "No food stands out yet. The data has bad outcomes, but not a repeated food pattern."
    top = candidates[0]
    return f"Top candidate: {top['item']} ({top['bad_exposures']}/{top['exposures']} exposures followed by symptoms)."
