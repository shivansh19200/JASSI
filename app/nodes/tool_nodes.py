"""Deterministic tool nodes for the LangGraph workflow: math, scheduling,
verification, and mastery updates. No LLM calls happen in this module."""
from __future__ import annotations
from typing import Dict, List, Any

from app.tools import feasibility, scheduler, verifier


def update_mastery(old_mastery: int, quiz_score: int) -> int:
    """Bayesian-style update: new = 0.7*score + 0.3*old, clamped to [0, 100].
    Used when the student actually took a quiz — a direct, high-confidence signal."""
    new_mastery = 0.7 * quiz_score + 0.3 * old_mastery
    return int(round(max(0, min(100, new_mastery))))


def update_mastery_from_completion(old_mastery: int, status: str, percent_done: int) -> int:
    """Lighter-touch update from a self-reported check-in (no quiz taken).
    This is the LLM's read of the student's free-text message (status +
    percent_done), not a test score, so it nudges mastery gently rather than
    re-anchoring it the way a quiz does:
      - "missed": small decay (nothing was practiced).
      - "completed"/"partial": moves with how much was actually done —
        0% done ~ -1, 100% done ~ +5.
    """
    if status == "missed":
        delta = -3
    else:
        delta = (percent_done / 100.0) * 6 - 1
    new_mastery = old_mastery + delta
    return int(round(max(0, min(100, new_mastery))))


def _load_merged_topics(state: Dict[str, Any]):
    topics_data = feasibility.load_topics(state["topics_path"])
    return feasibility.merge_dynamic_topics(topics_data, state.get("dynamic_topics", []))


def node_calculate_feasibility(state: Dict[str, Any]) -> Dict[str, Any]:
    topics_data = _load_merged_topics(state)
    result = feasibility.calculate_feasibility(
        topics=state["pr_topics"],
        target_mastery=state["pr_target_mastery"],
        deadline_days=state["pr_deadline_days"],
        current_mastery=state["current_mastery"],
        calendar_events=state["calendar_events"],
        topics_data=topics_data,
        allowed_days=state.get("preferred_days") or None,
    )
    state["feasibility_result"] = result
    return state


def node_build_schedule(state: Dict[str, Any]) -> Dict[str, Any]:
    import datetime
    sessions_needed = state["sessions_needed"]  # produced from the chosen resource path
    start_date = state.get("start_date") or datetime.date.today()
    time_window = None
    if state.get("preferred_time_start") is not None and state.get("preferred_time_end") is not None:
        time_window = (state["preferred_time_start"], state["preferred_time_end"])
    result = scheduler.build_schedule_dated(
        sessions_needed=sessions_needed,
        calendar_events=state["calendar_events"],
        start_date=start_date,
        deadline_days=state["pr_deadline_days"],
        moved_soft_event_ids=state.get("moved_soft_event_ids", []),
        allowed_days=state.get("preferred_days") or None,
        existing_dated_sessions=state.get("existing_sessions", []),
        time_window=time_window,
        ignore_time_conflict=state.get("ignore_time_conflict", False),
    )
    state["schedule_result"] = result
    return state


def node_verify_plan(state: Dict[str, Any]) -> Dict[str, Any]:
    topics_data = _load_merged_topics(state)
    all_sessions = state["schedule_result"]["scheduled"] + state.get("existing_sessions", [])
    result = verifier.verify_plan(
        sessions=all_sessions,
        calendar_events=state["calendar_events"],
        topics_data=topics_data,
        deadline_days=state["pr_deadline_days"],
    )
    state["verification_result"] = result
    return state


def node_update_mastery(state: Dict[str, Any]) -> Dict[str, Any]:
    """Always updates mastery on a check-in: a quiz score (if given) is the
    authoritative signal; otherwise we fall back to what the LLM parsed out
    of the student's free-text update (status + percent_done)."""
    topic = state["update_topic"]
    quiz_score = state.get("quiz_score")
    old = state["current_mastery"].get(topic, 0)
    parsed = state.get("parsed_update") or {}

    if quiz_score is not None:
        new = update_mastery(old, quiz_score)
        state["mastery_update_source"] = "quiz"
    else:
        status = parsed.get("status", "partial")
        percent_done = parsed.get("percent_done", 50)
        new = update_mastery_from_completion(old, status, percent_done)
        state["mastery_update_source"] = "self_report"

    state["current_mastery"][topic] = new
    state["mastery_delta"] = new - old
    return state
