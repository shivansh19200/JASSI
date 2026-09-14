"""Deterministic scheduling: constraint satisfaction for session placement."""
from __future__ import annotations
from typing import Dict, List, Any, Optional
import uuid

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DAY_START_HOUR = 6
DAY_END_HOUR = 23
SLOT_MINUTES = 30

# Windows the smart scheduler avoids placing sessions in (people are usually
# eating, not studying) unless literally nothing else is free.
LUNCH_WINDOW = (12 * 60, 13 * 60 + 30)
DINNER_WINDOW = (19 * 60 + 30, 21 * 60)


def _score_slot(start: int, end: int, pref_window: Optional[tuple], load_minutes: int) -> float:
    """Higher is better. Rewards matching the student's stated time-of-day
    preference, penalizes overlapping lunch/dinner, and penalizes days that
    already carry a lot of study load (so sessions spread out across the
    week instead of piling onto whichever day happens to be scanned first)."""
    score = 0.0
    if pref_window:
        pref_start, pref_end = pref_window
        span = max(1, end - start)
        overlap = max(0, min(end, pref_end) - max(start, pref_start))
        if overlap >= span:
            score += 100.0
        else:
            score += 100.0 * (overlap / span) - 15.0
    for ls, le in (LUNCH_WINDOW, DINNER_WINDOW):
        if start < le and end > ls:
            score -= 60.0
    score -= load_minutes / 20.0  # workload-balancing penalty
    return score


def _time_to_minutes(t: str) -> int:
    h, m = t.split(":")
    return int(h) * 60 + int(m)


def _minutes_to_time(m: int) -> str:
    return f"{m // 60:02d}:{m % 60:02d}"


def _build_busy_map(calendar_events: List[Dict[str, Any]], respect_soft: bool = True) -> Dict[str, List[tuple]]:
    busy: Dict[str, List[tuple]] = {d: [] for d in DAY_ORDER}
    for ev in calendar_events:
        if ev.get("type") == "soft" and not respect_soft:
            continue
        if ev.get("type") == "study_session":
            continue
        day = ev["day"]
        busy.setdefault(day, []).append((_time_to_minutes(ev["start"]), _time_to_minutes(ev["end"])))
    for d in busy:
        busy[d].sort()
    return busy


def _find_free_slots(busy_intervals: List[tuple], duration_mins: int) -> List[tuple]:
    slots = []
    cursor = DAY_START_HOUR * 60
    day_end = DAY_END_HOUR * 60
    intervals = sorted(busy_intervals)
    for start, end in intervals:
        if start - cursor >= duration_mins:
            slots.append((cursor, cursor + duration_mins))
        cursor = max(cursor, end)
    if day_end - cursor >= duration_mins:
        slots.append((cursor, cursor + duration_mins))
    return slots


def build_schedule(sessions_needed: List[Dict[str, Any]], calendar_events: List[Dict[str, Any]],
                    moved_soft_event_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    sessions_needed: list of {"topic": str, "duration_mins": int, "resource_id": str, "order": int}
      ordered by prerequisite priority (earlier = higher priority)
    moved_soft_event_ids: soft events the user agreed to move out of the way
    Returns {"scheduled": [...session dicts with day/start/end...], "unscheduled": [...]}
    """
    moved_soft_event_ids = moved_soft_event_ids or []
    active_events = [ev for ev in calendar_events if ev["id"] not in moved_soft_event_ids]
    busy = _build_busy_map(active_events, respect_soft=True)

    scheduled = []
    unscheduled = []

    # round-robin across days, filling earliest free slot per topic priority order
    day_cursor_idx = 0
    for item in sorted(sessions_needed, key=lambda x: x.get("order", 0)):
        placed = False
        attempts = 0
        idx = day_cursor_idx
        while attempts < 14 and not placed:
            day = DAY_ORDER[idx % 7]
            free = _find_free_slots(busy[day], item["duration_mins"])
            if free:
                start, end = free[0]
                session = {
                    "id": f"sess_{uuid.uuid4().hex[:8]}",
                    "topic": item["topic"],
                    "resource_id": item.get("resource_id"),
                    "resource_title": item.get("resource_title"),
                    "day": day,
                    "start": _minutes_to_time(start),
                    "end": _minutes_to_time(end),
                    "duration_mins": item["duration_mins"],
                    "status": "scheduled",
                }
                scheduled.append(session)
                busy[day].append((start, end))
                busy[day].sort()
                placed = True
            idx += 1
            attempts += 1
        if not placed:
            unscheduled.append(item)
        day_cursor_idx = (day_cursor_idx + 1) % 7

    return {"scheduled": scheduled, "unscheduled": unscheduled}


def _find_full_block_conflict(active_events: List[Dict[str, Any]], weekdays: List[str],
                               time_window: tuple) -> Optional[Dict[str, Any]]:
    """If a hard/soft commitment entirely covers the student's preferred study
    window on one of their allowed weekdays, that's worth flagging instead of
    silently scheduling around it — the student may want to move the
    commitment (if it's soft) rather than have JASSI quietly work around it."""
    pref_start, pref_end = time_window
    for ev in active_events:
        if ev.get("type") not in ("hard", "soft"):
            continue
        if ev.get("day") not in weekdays:
            continue
        es, ee = _time_to_minutes(ev["start"]), _time_to_minutes(ev["end"])
        if es <= pref_start and ee >= pref_end:
            return {"id": ev["id"], "title": ev["title"], "type": ev["type"], "day": ev["day"],
                    "start": ev["start"], "end": ev["end"]}
    return None


def build_schedule_dated(sessions_needed: List[Dict[str, Any]], calendar_events: List[Dict[str, Any]],
                          start_date: "datetime.date", deadline_days: int,
                          moved_soft_event_ids: Optional[List[str]] = None,
                          allowed_days: Optional[List[str]] = None,
                          existing_dated_sessions: Optional[List[Dict[str, Any]]] = None,
                          time_window: Optional[tuple] = None,
                          ignore_time_conflict: bool = False) -> Dict[str, Any]:
    """Date-aware scheduler: places sessions across the FULL deadline window
    (which may span several calendar weeks), not just a single generic
    Mon-Sun template. calendar_events (hard/soft) are still weekday-recurring
    templates and repeat every week in the window; study sessions get a real
    "date" in addition to "day" (their weekday, kept for the weekly views).

    allowed_days optionally restricts placement to the student's preferred
    weekdays (e.g. ["Saturday", "Sunday"] for "just weekends").

    time_window optionally restricts/biases placement to a (start_min, end_min)
    range the student prefers (e.g. evenings). Every candidate date+slot is
    scored (see `_score_slot`) and the single best one across the WHOLE
    allowed window is chosen for each session, instead of just taking the
    first free gap on the first day reached — this is what actually spreads
    sessions out (workload balance) and keeps them out of lunch/dinner.
    """
    import datetime as _dt
    moved_soft_event_ids = moved_soft_event_ids or []
    existing_dated_sessions = existing_dated_sessions or []
    active_events = [ev for ev in calendar_events if ev["id"] not in moved_soft_event_ids]
    weekday_busy = _build_busy_map(active_events, respect_soft=True)

    # busy[date_iso] = list of (start_min, end_min), seeded from the recurring
    # template for that date's weekday, plus any study sessions already placed
    # on that exact date (from a previous scheduling run).
    busy_by_date: Dict[str, List[tuple]] = {}
    daily_load: Dict[str, int] = {}
    for s in existing_dated_sessions:
        if s.get("date"):
            busy_by_date.setdefault(s["date"], []).append(
                (_time_to_minutes(s["start"]), _time_to_minutes(s["end"]))
            )
            daily_load[s["date"]] = daily_load.get(s["date"], 0) + s.get("duration_mins", 0)

    dates = []
    for i in range(max(1, deadline_days)):
        d = start_date + _dt.timedelta(days=i)
        weekday = DAY_ORDER[d.weekday()]
        if allowed_days and weekday not in allowed_days:
            continue
        dates.append((d, weekday))

    time_conflict = None
    if time_window and not ignore_time_conflict:
        weekdays_in_play = allowed_days or DAY_ORDER
        time_conflict = _find_full_block_conflict(active_events, weekdays_in_play, time_window)

    scheduled = []
    unscheduled = []

    for item in sorted(sessions_needed, key=lambda x: x.get("order", 0)):
        duration = item["duration_mins"]
        best = None  # (score, date_iso, weekday, start, end)
        for d, weekday in dates:
            date_iso = d.isoformat()
            busy_today = list(weekday_busy.get(weekday, [])) + busy_by_date.get(date_iso, [])
            busy_today.sort()
            for start, end in _find_free_slots(busy_today, duration):
                score = _score_slot(start, end, time_window, daily_load.get(date_iso, 0))
                if best is None or score > best[0]:
                    best = (score, date_iso, weekday, start, end)
        if best is None:
            unscheduled.append(item)
            continue
        _, date_iso, weekday, start, end = best
        session = {
            "id": f"sess_{uuid.uuid4().hex[:8]}",
            "topic": item["topic"],
            "resource_id": item.get("resource_id"),
            "resource_title": item.get("resource_title"),
            "date": date_iso,
            "day": weekday,
            "start": _minutes_to_time(start),
            "end": _minutes_to_time(end),
            "duration_mins": duration,
            "status": "scheduled",
        }
        scheduled.append(session)
        busy_by_date.setdefault(date_iso, []).append((start, end))
        daily_load[date_iso] = daily_load.get(date_iso, 0) + duration

    return {"scheduled": scheduled, "unscheduled": unscheduled, "time_conflict": time_conflict}


def reschedule_single_dated(session: Dict[str, Any], calendar_events: List[Dict[str, Any]],
                             all_current_sessions: List[Dict[str, Any]],
                             start_date: "datetime.date", horizon_days: int = 21,
                             exclude_date: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Date-aware version of reschedule_single: finds the next free dated slot
    within `horizon_days` of start_date, so a missed session gets a real
    future date instead of just "some day with this weekday name"."""
    import datetime as _dt
    weekday_busy = _build_busy_map(calendar_events, respect_soft=True)
    duration = session["duration_mins"]
    busy_by_date: Dict[str, List[tuple]] = {}
    for s in all_current_sessions:
        if s["id"] != session["id"] and s.get("date"):
            busy_by_date.setdefault(s["date"], []).append(
                (_time_to_minutes(s["start"]), _time_to_minutes(s["end"]))
            )

    for i in range(horizon_days):
        d = start_date + _dt.timedelta(days=i)
        date_iso = d.isoformat()
        if exclude_date and date_iso == exclude_date:
            continue
        weekday = DAY_ORDER[d.weekday()]
        busy_today = list(weekday_busy.get(weekday, [])) + busy_by_date.get(date_iso, [])
        busy_today.sort()
        free = _find_free_slots(busy_today, duration)
        if free:
            start, end = free[0]
            new_session = dict(session)
            new_session["date"] = date_iso
            new_session["day"] = weekday
            new_session["start"] = _minutes_to_time(start)
            new_session["end"] = _minutes_to_time(end)
            new_session["status"] = "scheduled"
            return new_session
    return None


def reschedule_single(session: Dict[str, Any], calendar_events: List[Dict[str, Any]],
                       all_current_sessions: List[Dict[str, Any]],
                       exclude_day: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Find the next available slot for one missed session."""
    other_study_events = [
        {"id": s["id"], "day": s["day"], "start": s["start"], "end": s["end"], "type": "study_session"}
        for s in all_current_sessions if s["id"] != session["id"]
    ]
    busy = _build_busy_map(calendar_events + other_study_events, respect_soft=True)
    duration = session["duration_mins"]
    idx = 0
    for _ in range(14):
        day = DAY_ORDER[idx % 7]
        idx += 1
        if exclude_day and day == exclude_day:
            continue
        free = _find_free_slots(busy[day], duration)
        if free:
            start, end = free[0]
            new_session = dict(session)
            new_session["day"] = day
            new_session["start"] = _minutes_to_time(start)
            new_session["end"] = _minutes_to_time(end)
            new_session["status"] = "scheduled"
            return new_session
    return None
