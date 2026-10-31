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


def _normalize_timetable_time(value: Any) -> str:
    """Normalize common timetable formats (08:00AM, 8 AM, 14:30) to HH:MM."""
    import re
    raw = str(value or "").strip().upper().replace(".", "")
    match = re.fullmatch(r"(\d{1,2}):(\d{2})\s*(AM|PM)?", raw)
    if not match:
        match = re.fullmatch(r"(\d{1,2})\s*(AM|PM)", raw)
        if not match:
            raise ValueError(f"Unrecognized time '{value}'")
        hour, minute, suffix = int(match.group(1)), 0, match.group(2)
    else:
        hour, minute, suffix = int(match.group(1)), int(match.group(2)), match.group(3)
    if minute > 59 or hour > 24 or hour == 24 and minute:
        raise ValueError(f"Invalid time '{value}'")
    if suffix == "AM":
        hour = 0 if hour == 12 else hour
    elif suffix == "PM":
        hour = 12 if hour == 12 else hour + 12
    if hour > 23:
        raise ValueError(f"Invalid time '{value}'")
    return f"{hour:02d}:{minute:02d}"


def _parse_timetable_with_local_ocr(image_bytes: bytes) -> List[Dict[str, Any]]:
    """Fast local OCR for grid-style weekly timetables.

    Day-column headers provide the horizontal anchors. Each column is OCR'd
    independently so adjacent classes in different weekday columns cannot be
    merged into one text line. If Tesseract is unavailable or the layout cannot
    be recognized, the caller falls back to the configured vision model.
    """
    try:
        from io import BytesIO
        from PIL import Image
        import pytesseract
        from pytesseract import Output
        import re

        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        data = pytesseract.image_to_data(image, output_type=Output.DATAFRAME, config="--psm 6")
        data = data.dropna(subset=["text"])
        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        headers = {}
        for _, row in data.iterrows():
            token = str(row["text"]).strip().title()
            if token in day_names and float(row.get("conf", 0)) >= 35:
                headers[token] = (
                    float(row["left"]) + float(row["width"]) / 2,
                    float(row["top"]), float(row["height"]),
                )
        ordered = sorted(headers.items(), key=lambda pair: pair[1][0])
        if len(ordered) < 3:
            return []

        centers = [value[0] for _, value in ordered]
        boundaries = [max(0, centers[0] - (centers[1] - centers[0]) / 2)]
        boundaries.extend((a + b) / 2 for a, b in zip(centers, centers[1:]))
        boundaries.append(min(image.width, centers[-1] + (centers[-1] - centers[-2]) / 2))
        grid_top = int(max(value[1] + value[2] for _, value in ordered) + 25)

        time_range = re.compile(
            r"(\d{1,2}:\d{2}\s*[AP]M)\s*-\s*(\d{1,2}:\d{2}\s*[AP]M)", re.I
        )
        course_code = re.compile(r"\b[A-Z]{2,}\s*[A-Z]?\s*\d{2,4}\b", re.I)
        descriptors = {"lecture", "tutorial", "laboratory", "lab", "seminar", "recitation", "workshop"}
        events = []
        for column_index, (day, _) in enumerate(ordered):
            left, right = int(boundaries[column_index]), int(boundaries[column_index + 1])
            if right - left < 20:
                continue
            crop = image.crop((left, grid_top, right, image.height))
            scale = 2
            crop = crop.resize((crop.width * scale, crop.height * scale))
            text = pytesseract.image_to_string(crop, config="--psm 6")
            lines = [" ".join(line.split()) for line in text.splitlines() if line.strip()]
            for line_index, line in enumerate(lines):
                combined = line
                if re.search(r"\d{1,2}:\d{2}\s*[AP]M\s*-\s*$", line, re.I) and line_index + 1 < len(lines):
                    combined += " " + lines[line_index + 1]
                match = time_range.search(combined)
                if not match:
                    continue
                title_line = None
                for previous_index in range(line_index - 1, max(-1, line_index - 5), -1):
                    candidate = lines[previous_index]
                    if course_code.search(candidate):
                        title_line = candidate
                        if previous_index + 1 < line_index and lines[previous_index + 1].strip().lower() in descriptors:
                            title_line += " " + lines[previous_index + 1]
                        break
                if not title_line and line_index:
                    title_line = lines[line_index - 1]
                title_line = " ".join((title_line or f"{day} class").split())
                try:
                    start = _normalize_timetable_time(match.group(1))
                    end = _normalize_timetable_time(match.group(2))
                    if _time_to_minutes(end) <= _time_to_minutes(start):
                        continue
                except ValueError:
                    continue
                events.append(manual_event(title_line, day, start, end, "hard"))

        # Avoid accepting a partial/garbled OCR pass when the image contains a
        # weekly grid but Tesseract found almost no usable class blocks.
        unique = {}
        for event in events:
            key = (event["title"].casefold(), event["day"], event["start"], event["end"])
            unique[key] = event
        return list(unique.values())
    except Exception:
        # Tesseract is an optional fast path; the vision model remains the
        # fallback for non-grid layouts and deployments without the binary.
        return []


def parse_image_with_llm(llm_client, image_bytes: bytes) -> List[Dict[str, Any]]:
    """Extract recurring weekly commitments from a timetable screenshot.

    Prefer local grid OCR for speed and determinism; use Groq vision as a
    fallback for non-grid layouts. Output is normalized and validated before
    it can enter the student's calendar.
    """
    if not image_bytes:
        raise RuntimeError("The uploaded timetable image is empty. Please choose another image.")
    ocr_events = _parse_timetable_with_local_ocr(image_bytes)
    if ocr_events:
        return ocr_events
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    if image_bytes.startswith(b"\xff\xd8\xff"):
        mime_type = "image/jpeg"
    elif image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        mime_type = "image/webp"
    else:
        mime_type = "image/png"
    system_prompt = (
        "You are a meticulous university timetable OCR system. Read every visible "
        "class, lecture, tutorial, laboratory, and fixed activity block in the image. "
        "Return ONLY a JSON object with an 'events' array. Each event must contain "
        "title, day, start, end, type. Day must be the full weekday name Monday through "
        "Sunday. Times must be 24-hour HH:MM. Set type='hard' for classes, lectures, "
        "tutorials and labs; use 'soft' only for clearly optional activities. "
        "IMPORTANT: use the weekday column and the vertical time grid to determine each "
        "event's exact start/end; do not confuse row labels with event times. If a block "
        "prints its own time range, use that range. Extract separate adjacent blocks as "
        "separate events. Ignore empty cells, headers, date-picker controls and footer "
        "course metadata that is not placed on the weekly grid. If the image shows a "
        "dated example week, treat the displayed weekday pattern as a recurring weekly "
        "timetable (do not return a one-off date). Do not invent missing events or times. "
        "If a time cannot be read confidently, omit that event rather than guessing. "
        "Example: {\"events\":[{\"title\":\"CS F222 Tutorial\",\"day\":\"Monday\","
        "\"start\":\"08:00\",\"end\":\"09:00\",\"type\":\"hard\"}]}"
    )
    try:
        response = llm_client.chat_vision(system_prompt, b64, mime_type=mime_type)
        cleaned = str(response or "").strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
        payload = json.loads(cleaned)
        events_raw = payload.get("events", []) if isinstance(payload, dict) else payload
        if not isinstance(events_raw, list):
            raise ValueError("The vision response did not contain an events list.")
    except Exception as exc:
        raise RuntimeError(
            "Couldn't reliably read this timetable image. Check the Groq vision model "
            "setting or try a clearer, straight-on screenshot. Details: "
            f"{exc}"
        )

    valid_days = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}
    events, seen = [], set()
    for index, item in enumerate(events_raw):
        if not isinstance(item, dict):
            continue
        title = " ".join(str(item.get("title") or "").split())
        day = str(item.get("day") or "").strip().title()
        if day not in valid_days or not title:
            continue
        try:
            start = _normalize_timetable_time(item.get("start"))
            end = _normalize_timetable_time(item.get("end"))
            if _time_to_minutes(end) <= _time_to_minutes(start):
                continue
        except (TypeError, ValueError):
            continue
        event_type = str(item.get("type") or "hard").strip().lower()
        if event_type not in {"hard", "soft"}:
            event_type = "hard"
        key = (title.casefold(), day, start, end)
        if key in seen:
            continue
        seen.add(key)
        events.append(manual_event(title, day, start, end, event_type))
    if not events:
        raise RuntimeError(
            "No valid timetable blocks could be extracted. Make sure the screenshot "
            "shows the weekly grid, weekday labels, and readable start/end times."
        )
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
