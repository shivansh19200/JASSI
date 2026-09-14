"""Deterministic feasibility math: required hours vs available hours."""
from __future__ import annotations
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
DAY_START_HOUR = 6
DAY_END_HOUR = 23

# Windows used for concrete, deterministic risk-flagging (see
# assess_schedule_risks) — kept separate from the scheduler's own
# LUNCH_WINDOW/DINNER_WINDOW constants so this module has no import
# dependency on app.tools.scheduler.
MEAL_WINDOWS = {"lunch": (12 * 60, 13 * 60 + 30), "dinner": (19 * 60 + 30, 21 * 60)}
# Gap (minutes) between two consecutive commitments below which they count
# as "back-to-back" for burnout purposes.
BACK_TO_BACK_GAP_MINUTES = 20
# Two or more back-to-back gaps in a single day is flagged as a burnout risk.
BACK_TO_BACK_RISK_THRESHOLD = 2
# A day already committed for this many hours (hard+soft only, before any
# studying is added) is flagged as having little slack left.
HEAVY_DAY_HOURS = 9.0


def load_topics(path: str = "data/topics.json") -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def topic_lookup(topics_data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {t["id"]: t for t in topics_data["topics"]}


def merge_dynamic_topics(topics_data: Dict[str, Any],
                          dynamic_topics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Returns a NEW topics dict with any LLM-synthesized topics (for requests
    outside the fixed catalog) layered on top, without mutating the original
    or the on-disk catalog. Dynamic topics carry a "dynamic": True flag so
    the UI can indicate they're AI-generated rather than curated."""
    if not dynamic_topics:
        return topics_data
    known_ids = {t["id"] for t in topics_data["topics"]}
    extra = [t for t in dynamic_topics if t["id"] not in known_ids]
    return {"topics": topics_data["topics"] + extra}


def required_hours_for_pr(topics: List[str], target_mastery: int,
                           current_mastery: Dict[str, int],
                           topics_data: Dict[str, Any]) -> float:
    """Estimate hours needed to bring each topic from current mastery to target."""
    lookup = topic_lookup(topics_data)
    total = 0.0
    for topic_id in topics:
        info = lookup.get(topic_id)
        if not info:
            continue
        base_hours = info["estimated_hours"]
        current = current_mastery.get(topic_id, 0)
        gap_ratio = max(0.0, (target_mastery - current)) / 100.0
        # scale estimated hours by how much of the mastery gap remains,
        # with a floor so even near-mastered topics get some review time
        total += base_hours * max(gap_ratio, 0.15) if current < target_mastery else 0.0
    return round(total, 2)


def _time_to_minutes(t: str) -> int:
    h, m = t.split(":")
    return int(h) * 60 + int(m)


def available_hours_in_window(calendar_events: List[Dict[str, Any]], deadline_days: int,
                               allowed_days: Optional[List[str]] = None) -> float:
    """Sum free hours across the deadline window, respecting hard AND soft
    constraints as currently-occupied (soft constraints are only freed up
    if the user explicitly chooses to move them during replanning).
    allowed_days optionally restricts studying to the student's preferred
    weekdays only (e.g. they said "just weekends")."""
    days_needed = min(deadline_days, 7)  # calendar currently modeled as a repeating week
    total_day_minutes = (DAY_END_HOUR - DAY_START_HOUR) * 60
    free_minutes = 0
    events_by_day: Dict[str, List[Dict[str, Any]]] = {}
    for ev in calendar_events:
        events_by_day.setdefault(ev["day"], []).append(ev)

    considered_days = DAY_ORDER[:days_needed] if deadline_days <= 7 else DAY_ORDER
    if allowed_days:
        considered_days = [d for d in considered_days if d in allowed_days]
    weeks = max(1, -(-deadline_days // 7))  # ceil division for multi-week windows

    for day in considered_days:
        occupied = 0
        for ev in events_by_day.get(day, []):
            occupied += _time_to_minutes(ev["end"]) - _time_to_minutes(ev["start"])
        free_minutes += max(0, total_day_minutes - occupied)

    total_free_minutes = free_minutes * weeks if deadline_days > 7 else free_minutes
    # cap realistic study time to ~35% of free time so we don't imply someone
    # will study every waking free minute
    return round((total_free_minutes * 0.35) / 60.0, 2)


def _day_commitment_blocks(calendar_events: List[Dict[str, Any]], day: str) -> List[tuple]:
    """(start_min, end_min, title) for hard/soft commitments on a given
    weekday, sorted by start time. study_session events are intentionally
    excluded — this describes the load the student is ALREADY carrying,
    not what JASSI itself is proposing to add."""
    blocks = []
    for ev in calendar_events:
        if ev.get("day") == day and ev.get("type") in ("hard", "soft"):
            blocks.append((_time_to_minutes(ev["start"]), _time_to_minutes(ev["end"]), ev.get("title", "Untitled")))
    return sorted(blocks)


def assess_schedule_risks(calendar_events: List[Dict[str, Any]], deadline_days: int,
                           allowed_days: Optional[List[str]] = None) -> List[str]:
    """Deterministic, concrete risk flags about the commitments the student
    ALREADY has — independent of (and a complement to) the required-vs-
    available hours math. This is what actually explains, in plain terms,
    why a plan might be rough to live with day to day: meals getting
    displaced, back-to-back blocks with no breathing room, days that are
    already nearly full before studying is even added. Deterministic on
    purpose, so the message is the same logic every time rather than
    whatever an LLM happens to phrase."""
    days_needed = min(deadline_days, 7) if deadline_days <= 7 else 7
    considered_days = DAY_ORDER[:days_needed]
    if allowed_days:
        considered_days = [d for d in considered_days if d in allowed_days]

    risks: List[str] = []
    for day in considered_days:
        blocks = _day_commitment_blocks(calendar_events, day)
        if not blocks:
            continue

        total_occupied = sum(e - s for s, e, _ in blocks)

        for meal_name, (ms, me) in MEAL_WINDOWS.items():
            if any(s <= ms and e >= me for s, e, _ in blocks):
                risks.append(
                    f"{day}: commitments run straight through {meal_name} — you may need to "
                    f"skip {meal_name} (or eat on the run) if a study session gets fit in around there."
                )

        tight_gaps = sum(
            1 for i in range(len(blocks) - 1)
            if 0 <= blocks[i + 1][0] - blocks[i][1] < BACK_TO_BACK_GAP_MINUTES
        )
        if tight_gaps >= BACK_TO_BACK_RISK_THRESHOLD:
            risks.append(
                f"{day} has {tight_gaps + 1} commitments back-to-back with almost no gap between "
                f"them — stacking study time on top of that raises real burnout risk."
            )

        if total_occupied / 60.0 >= HEAVY_DAY_HOURS:
            risks.append(
                f"{day} is already committed for about {round(total_occupied / 60.0, 1)}h before any "
                f"studying is added — there's very little slack left without cutting into rest or sleep."
            )

    return risks


def calculate_feasibility(topics: List[str], target_mastery: int, deadline_days: int,
                           current_mastery: Dict[str, int],
                           calendar_events: List[Dict[str, Any]],
                           topics_data: Dict[str, Any],
                           allowed_days: Optional[List[str]] = None) -> Dict[str, Any]:
    required = required_hours_for_pr(topics, target_mastery, current_mastery, topics_data)
    available = available_hours_in_window(calendar_events, deadline_days, allowed_days=allowed_days)
    shortfall = round(max(0.0, required - available), 2)
    feasible = shortfall <= 0.01
    risks = assess_schedule_risks(calendar_events, deadline_days, allowed_days=allowed_days)
    # Not just "infeasible" but so far off that presenting the usual 3
    # tradeoffs (move a few soft events / nudge the deadline / drop one
    # topic) would be misleading — none of them get you close on their own.
    # Flag this so the UI can reject the plan outright instead of offering
    # choices that look like a fix but aren't.
    reject = shortfall > 0 and (available <= 0.01 or shortfall > available * 2)
    return {
        "required_hours": required,
        "available_hours": available,
        "shortfall_hours": shortfall,
        "feasible": feasible,
        "risks": risks,
        "reject": reject,
    }


def soft_events_hours(calendar_events: List[Dict[str, Any]]) -> float:
    total = 0
    for ev in calendar_events:
        if ev.get("type") == "soft":
            total += _time_to_minutes(ev["end"]) - _time_to_minutes(ev["start"])
    return round(total / 60.0, 2)


def min_deadline_extension_days(shortfall_hours: float, avg_free_hours_per_day: float = 1.5) -> int:
    if avg_free_hours_per_day <= 0:
        return 999
    import math
    return max(1, math.ceil(shortfall_hours / avg_free_hours_per_day))
