"""Parses constraint imports: manual form entries, timetable images (via LLM vision),
and Google Calendar OAuth events, normalizing them all into the calendar_events schema."""
from __future__ import annotations
from typing import Dict, List, Any
import base64
import json
import uuid


def manual_event(title: str, day: str, start: str, end: str, event_type: str) -> Dict[str, Any]:
    return {
        "id": f"ev_{uuid.uuid4().hex[:8]}",
        "title": title,
        "day": day,
        "start": start,
        "end": end,
        "type": event_type,  # "hard" or "soft"
    }


def _time_to_minutes(t: str) -> int:
    h, m = t.split(":")
    return int(h) * 60 + int(m)


def _minutes_to_time(m: int) -> str:
    return f"{m // 60:02d}:{m % 60:02d}"


# Commitments are assumed to live within this window; a bumped event that
# would run past the end of the day simply doesn't fit anywhere later that
# day, so resolve_event_time_conflict reports that instead of returning a
# nonsensical past-midnight slot.
DAY_END_HOUR = 23


def resolve_event_time_conflict(existing_events: List[Dict[str, Any]], day: str,
                                 start: str, end: str, exclude_id: str = None) -> Dict[str, Any]:
    """Checks a new event's (day, start, end) against the student's existing
    hard/soft commitments on that same day. If it overlaps one or more of
    them, the new event is pushed to start right after the latest
    conflicting commitment ends (keeping the same duration), and that push
    is repeated until the slot is clear of every other commitment or the day
    runs out. This is what makes "add two things to Monday evening" resolve
    into back-to-back events instead of both landing in the same window.

    Returns {"start": str, "end": str, "moved": bool, "bumped": [titles], "fits": bool}.
    When "fits" is False, the event (at its original duration) doesn't clear
    every conflict before the day ends — the caller should surface that
    instead of silently saving an overlapping or out-of-bounds event.
    """
    start_min = _time_to_minutes(start)
    end_min = _time_to_minutes(end)
    duration = end_min - start_min
    day_end = DAY_END_HOUR * 60

    day_events = sorted(
        (ev for ev in existing_events
         if ev.get("day") == day and ev.get("type") in ("hard", "soft") and ev.get("id") != exclude_id),
        key=lambda e: _time_to_minutes(e["start"]),
    )

    bumped: List[str] = []
    moved = False
    changed = True
    while changed:
        changed = False
        for ev in day_events:
            es, ee = _time_to_minutes(ev["start"]), _time_to_minutes(ev["end"])
            if start_min < ee and end_min > es:
                start_min = ee
                end_min = start_min + duration
                moved = True
                if ev["title"] not in bumped:
                    bumped.append(ev["title"])
                changed = True

    fits = end_min <= day_end
    return {
        "start": _minutes_to_time(start_min) if fits else start,
        "end": _minutes_to_time(end_min) if fits else end,
        "moved": moved,
        "bumped": bumped,
        "fits": fits,
    }


def parse_image_with_llm(llm_client, image_bytes: bytes) -> List[Dict[str, Any]]:
    """Uses the LLM's vision capability (if the configured Groq model supports it)
    to OCR + interpret a timetable screenshot into structured calendar events.
    Falls back to raising a clear error if the model can't process images."""
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    system_prompt = (
        "You extract a weekly class/activity timetable from an image. "
        "Return ONLY a JSON array (no prose, no markdown fences) of objects with keys: "
        "title (string), day (one of Monday..Sunday), start (HH:MM 24h), end (HH:MM 24h), "
        "type ('hard' for classes/labs/appointments that cannot move, 'soft' for optional "
        "clubs/activities that could be moved)."
    )
    try:
        response = llm_client.chat_vision(system_prompt, b64)
        cleaned = response.strip().strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
        events_raw = json.loads(cleaned)
    except Exception as exc:
        raise RuntimeError(
            "Couldn't parse the timetable image. Try manual entry instead, "
            f"or check your Groq vision model configuration. ({exc})"
        )
    events = []
    for e in events_raw:
        events.append(manual_event(e["title"], e["day"], e["start"], e["end"], e.get("type", "hard")))
    return events


def parse_google_calendar_events(raw_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalizes Google Calendar API event objects into JASSI's calendar_events schema.
    Expects each raw_event to have 'summary', 'start'.'dateTime', 'end'.'dateTime'."""
    import datetime
    out = []
    for ev in raw_events:
        try:
            start_dt = datetime.datetime.fromisoformat(ev["start"]["dateTime"])
            end_dt = datetime.datetime.fromisoformat(ev["end"]["dateTime"])
        except (KeyError, ValueError):
            continue
        day_name = start_dt.strftime("%A")
        out.append(manual_event(
            title=ev.get("summary", "Untitled event"),
            day=day_name,
            start=start_dt.strftime("%H:%M"),
            end=end_dt.strftime("%H:%M"),
            event_type="hard",
        ))
    return out
