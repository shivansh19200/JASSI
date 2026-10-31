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
    """Normalize common timetable time labels to 24-hour HH:MM.

    Accepts 08:00, 8:00 AM, 08:00AM, 8 AM, 0800, and common OCR punctuation
    variants. It intentionally contains no institution/course-specific rules.
    """
    import re
    raw = str(value or "").strip().upper()
    raw = raw.replace(".", ":").replace("O", "0").replace("I", "1")
    raw = re.sub(r"\s+", " ", raw)
    match = re.fullmatch(r"(\d{1,2})(?::(\d{1,2}))?\s*(AM|PM)?", raw)
    if not match:
        compact = re.fullmatch(r"(\d{2})(\d{2})", raw.replace(" ", ""))
        if compact:
            hour, minute, suffix = int(compact.group(1)), int(compact.group(2)), None
        else:
            raise ValueError(f"Unrecognized time '{value}'")
    else:
        hour = int(match.group(1))
        minute = int(match.group(2) or 0)
        suffix = match.group(3)
    if minute > 59 or hour > 24 or (hour == 24 and minute):
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
            r"(\d{1,2}[:.]\d{2}\s*(?:AM|PM)?)\s*(?:-|–|—|to)\s*"
            r"(\d{1,2}[:.]\d{2}\s*(?:AM|PM)?)", re.I
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
                if re.search(r"\d{1,2}[:.]\d{2}\s*(?:AM|PM)?\s*(?:-|–|—|to)\s*$", line, re.I) and line_index + 1 < len(lines):
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


def _extract_json_payload(raw: Any) -> Any:
    """Accept JSON arrays/objects even when a vision model wraps them in prose or fences."""
    import re
    if isinstance(raw, (dict, list)):
        return raw
    text = str(raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Find the first balanced JSON object/array without assuming a fixed wrapper.
    for opening, closing in (("{", "}"), ("[", "]")):
        first = text.find(opening)
        if first < 0:
            continue
        depth, in_string, escaped = 0, False, False
        for index in range(first, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == opening:
                depth += 1
            elif char == closing:
                depth -= 1
                if depth == 0:
                    return json.loads(text[first:index + 1])
    raise ValueError("The vision response did not contain valid JSON.")


def _event_time_fields(item: Dict[str, Any]):
    """Read common field-name and time-range variants from vision output."""
    import re
    start = item.get("start", item.get("start_time", item.get("from")))
    end = item.get("end", item.get("end_time", item.get("to")))
    if (start is None or end is None) and item.get("time_range"):
        parts = re.split(r"\s*(?:-|–|—|to)\s*", str(item["time_range"]), maxsplit=1, flags=re.I)
        if len(parts) == 2:
            start, end = parts
    return _normalize_timetable_time(start), _normalize_timetable_time(end)


def parse_image_with_llm(llm_client, image_bytes: bytes) -> List[Dict[str, Any]]:
    """Extract a weekly timetable without relying on a particular university layout.

    Groq vision is the primary extractor so different timetable layouts and
    time conventions are not tied to a brittle OCR template. Local OCR is only
    a fallback when vision fails or returns no valid events. Every candidate is
    normalized and validated before it can be imported.
    """
    if not image_bytes:
        raise RuntimeError("The uploaded timetable image is empty. Please choose another image.")

    # Keep local OCR available as a fallback, but do not trust it as the primary
    # parser: a different grid layout can produce a plausible-looking partial set.
    ocr_events = _parse_timetable_with_local_ocr(image_bytes)

    b64 = base64.b64encode(image_bytes).decode("utf-8")
    if image_bytes.startswith(b"\xff\xd8\xff"):
        mime_type = "image/jpeg"
    elif image_bytes.startswith(b"RIFF") and image_bytes[8:12] == b"WEBP":
        mime_type = "image/webp"
    else:
        mime_type = "image/png"

    system_prompt = (
        "You are a careful timetable image extraction system. Read the uploaded image "
        "as a human would, regardless of university, visual theme, grid style, camera "
        "angle, or time notation. Extract every visible scheduled block from the weekly "
        "calendar grid. Return a JSON object with an 'events' array only. Each item must "
        "have title, day, start, end, and type. Use full weekday names Monday through "
        "Sunday. Use 24-hour HH:MM times. Accept the printed time inside a block when "
        "available; otherwise infer the block's start/end from the vertical time axis "
        "and its cell boundaries. Do not mistake hour-axis labels for class events. "
        "Keep adjacent blocks separate. Ignore page headers, date controls, legends, "
        "and footer metadata outside the weekly grid. A displayed sample week represents "
        "a recurring weekly pattern. Classes, lectures, tutorials and labs are hard. "
        "Only clearly optional activities are soft. Do not invent unreadable titles or "
        "times; omit uncertain blocks. Do not assume any particular course-code format. "
        "Example shape: {\"events\":[{\"title\":\"Course tutorial\","
        "\"day\":\"Monday\",\"start\":\"08:00\","
        "\"end\":\"08:50\",\"type\":\"hard\"}]}"
    )
    try:
        response = llm_client.chat_vision(system_prompt, b64, mime_type=mime_type)
        payload = _extract_json_payload(response)
        events_raw = payload.get("events", payload.get("classes", [])) if isinstance(payload, dict) else payload
        if not isinstance(events_raw, list):
            raise ValueError("The vision response did not contain an events list.")
    except Exception as exc:
        if len(ocr_events) >= 5:
            return ocr_events
        raise RuntimeError(
            "Couldn't reliably read this timetable image. Try a sharper, straight-on "
            "image with the weekday headings and time labels visible. Details: "
            f"{exc}"
        )

    valid_days = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}
    day_aliases = {day[:3].lower(): day for day in valid_days}
    events, seen = [], set()
    for item in events_raw:
        if not isinstance(item, dict):
            continue
        title = " ".join(str(item.get("title", item.get("course", item.get("name", ""))) or "").split())
        day_raw = str(item.get("day", item.get("weekday", "")) or "").strip().title()
        day = day_raw if day_raw in valid_days else day_aliases.get(day_raw[:3].lower())
        if not title or not day:
            continue
        try:
            start, end = _event_time_fields(item)
            start_min, end_min = _time_to_minutes(start), _time_to_minutes(end)
            if end_min <= start_min:
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
        if len(ocr_events) >= 5:
            return ocr_events
        raise RuntimeError(
            "No valid timetable blocks could be extracted. Check that weekday headings "
            "and class times are visible, then try uploading a clearer image."
        )
    return events


def find_timetable_conflicts(imported_events: List[Dict[str, Any]],
                              existing_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return overlapping imported/existing weekday events using half-open intervals."""
    conflicts = []
    for imported in imported_events:
        try:
            new_start = _time_to_minutes(imported["start"])
            new_end = _time_to_minutes(imported["end"])
        except (KeyError, TypeError, ValueError):
            continue
        for existing in existing_events:
            if existing.get("day") != imported.get("day"):
                continue
            try:
                old_start = _time_to_minutes(existing["start"])
                old_end = _time_to_minutes(existing["end"])
            except (KeyError, TypeError, ValueError):
                continue
            if new_start < old_end and new_end > old_start:
                conflicts.append({"imported": imported, "existing": existing})
    return conflicts


def merge_timetable_import(existing_events: List[Dict[str, Any]],
                           imported_events: List[Dict[str, Any]],
                           override_conflicts: bool = False):
    """Merge imported fixed classes; optionally replace only overlapping old events.

    Default behavior preserves every existing commitment. If override is explicit,
    only existing events that overlap an imported class are removed.
    """
    conflicts = find_timetable_conflicts(imported_events, existing_events)
    if not override_conflicts:
        return list(existing_events) + list(imported_events), 0
    conflicting_ids = {
        conflict["existing"].get("id") for conflict in conflicts
        if conflict["existing"].get("id") is not None
    }
    conflicting_keys = {
        (e.get("day"), e.get("start"), e.get("end"), e.get("title"))
        for conflict in conflicts for e in [conflict["existing"]]
    }
    kept = [
        event for event in existing_events
        if event.get("id") not in conflicting_ids
        and (event.get("day"), event.get("start"), event.get("end"), event.get("title")) not in conflicting_keys
    ]
    return kept + list(imported_events), len(existing_events) - len(kept)


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
