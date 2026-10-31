"""Dependency-aware, whole-roadmap recovery for overdue learning sessions.

Recovery is deterministic and transactional: completed work is retained as
history, overdue work is converted into replacement tasks, and every remaining
active session is rescheduled together. The candidate roadmap is committed only
when all sessions fit and the verifier accepts the complete result.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple
import uuid

from app.tools import scheduler, verifier, feasibility

TERMINAL_STATUSES = {"completed", "skipped", "cancelled", "missed"}
ACTIVE_PLAN_STATUSES = {"active", "in_progress", "in progress"}


def _session_date(session: Dict[str, Any]):
    raw = session.get("date")
    if not raw:
        return None
    try:
        return date.fromisoformat(str(raw)[:10])
    except (TypeError, ValueError):
        return None


def _calendar_event(session: Dict[str, Any]) -> Dict[str, Any]:
    title = session.get("resource_title") or session.get("topic", "Study session").replace("_", " ").title()
    return {
        "id": session["id"],
        "title": f"{session.get('topic', 'Study').replace('_', ' ').title()} · {title}",
        "day": session["day"],
        "date": session.get("date"),
        "start": session["start"],
        "end": session["end"],
        "type": "study_session",
    }


def _topic_order(sessions: List[Dict[str, Any]], topics_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Stable topological ordering: prerequisites are scheduled before dependents."""
    lookup = feasibility.topic_lookup(topics_data)
    original_index = {id(s): i for i, s in enumerate(sessions)}
    by_topic: Dict[str, List[Dict[str, Any]]] = {}
    for session in sessions:
        by_topic.setdefault(session.get("topic", ""), []).append(session)

    # Keep sessions within each topic in their original chronological order.
    for topic_sessions in by_topic.values():
        topic_sessions.sort(key=lambda s: (
            _session_date(s) or date.max, s.get("start", "00:00"), original_index[id(s)]
        ))

    ordered_topics: List[str] = []
    visiting, visited = set(), set()

    def visit(topic: str):
        if topic in visited or topic in visiting:
            return
        visiting.add(topic)
        for prereq in lookup.get(topic, {}).get("prerequisites", []):
            if prereq in by_topic:
                visit(prereq)
        visiting.remove(topic)
        visited.add(topic)
        if topic in by_topic:
            ordered_topics.append(topic)

    # The original order is the stable tie-breaker for unrelated topics.
    for session in sorted(sessions, key=lambda s: original_index[id(s)]):
        visit(session.get("topic", ""))

    result: List[Dict[str, Any]] = []
    for topic in ordered_topics:
        result.extend(by_topic.get(topic, []))
    return result


def _remaining_deadline_days(plan: Dict[str, Any], today: date) -> int:
    """Return days left in the original deadline window; never silently extend it."""
    explicit = plan.get("deadline_date")
    if explicit:
        try:
            deadline = date.fromisoformat(str(explicit)[:10])
            return max(0, (deadline - today).days + 1)
        except (TypeError, ValueError):
            pass
    created = plan.get("created_at")
    try:
        created_date = date.fromisoformat(str(created)[:10]) if created else today
    except (TypeError, ValueError):
        created_date = today
    try:
        total_days = max(1, int(plan.get("deadline_days", 90) or 90))
    except (TypeError, ValueError):
        total_days = 90
    deadline = created_date + timedelta(days=total_days - 1)
    return max(0, (deadline - today).days + 1)


def _task_from_session(session: Dict[str, Any], order: int, recovery: bool = False) -> Dict[str, Any]:
    return {
        "topic": session.get("topic", "general_review"),
        "duration_mins": int(session.get("duration_mins", 60) or 60),
        "resource_id": session.get("resource_id"),
        "resource_title": session.get("resource_title"),
        "resource_url": session.get("resource_url"),
        "resources": session.get("resources", []),
        "order": order,
        "_source_session_id": session.get("id"),
        "_is_recovery": recovery,
    }


def _recover_plan(student: Dict[str, Any], plan: Dict[str, Any], today: date, force_session_ids: set[str] | None = None) -> List[Dict[str, Any]]:
    sessions = plan.setdefault("sessions", [])
    calendar = student.setdefault("calendar_events", [])
    force_session_ids = force_session_ids or set()
    overdue = sorted(
        [s for s in sessions
         if ((_session_date(s) is not None and _session_date(s) < today) or str(s.get("id")) in force_session_ids)
         and (str(s.get("status", "scheduled")).lower() not in TERMINAL_STATUSES or str(s.get("id")) in force_session_ids)
         and not s.get("replan_processed_at")
         and not s.get("replanned_to")],
        key=lambda s: (_session_date(s) or date.min, s.get("start", "00:00")),
    )
    if not overdue:
        return []

    now = datetime.now().isoformat(timespec="seconds")
    overdue_ids = {str(s.get("id")) for s in overdue}
    for old in overdue:
        old["status"] = "missed"
        old["missed_detected_at"] = now
        old["replan_reason"] = ("Learner reported the session missed" if str(old.get("id")) in force_session_ids else "Scheduled date passed without completion")
        old["replan_processed_at"] = now
    # A missed session is no longer a live calendar commitment, even if a
    # replacement cannot fit. Keep the historical session record in the plan.
    calendar[:] = [e for e in calendar if str(e.get("id")) not in overdue_ids]

    # Keep completed/skipped/cancelled/missed history intact. Rebuild every
    # remaining active task, including all overdue tasks as fresh replacements.
    active_sources = [
        s for s in sessions
        if str(s.get("status", "scheduled")).lower() not in TERMINAL_STATUSES
    ]
    task_sources: List[Tuple[Dict[str, Any], bool]] = [(s, False) for s in active_sources]
    task_sources.extend((s, True) for s in overdue)

    topics_path = plan.get("topics_path") or str(Path(__file__).resolve().parents[2] / "data" / "topics.json")
    try:
        topics_data = feasibility.merge_dynamic_topics(
            feasibility.load_topics(topics_path),
            plan.get("dynamic_topics") or student.get("dynamic_topics") or [],
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        message = f"Could not load topic dependencies for recovery: {exc}"
        for old in overdue:
            old["replan_failure"] = message
        return [{"plan_id": plan.get("id"), "original": old, "replacement": None,
                 "reason": message} for old in overdue]

    ordered_sources = _topic_order([s for s, _ in task_sources], topics_data)
    recovery_ids = {s.get("id") for s in overdue}
    tasks = [_task_from_session(s, i, s.get("id") in recovery_ids)
             for i, s in enumerate(ordered_sources)]

    # Other plans' active sessions are treated as occupied dated blocks. Current
    # plan's sessions are rebuilt from scratch, while hard/soft recurring events
    # remain the student's fixed and flexible commitments.
    other_plan_sessions = []
    for other_plan in student.get("planning_requests", []):
        if other_plan is plan or other_plan.get("status", "active") not in ACTIVE_PLAN_STATUSES:
            continue
        other_plan_sessions.extend(
            s for s in other_plan.get("sessions", [])
            if _session_date(s) is not None
            and _session_date(s) >= today
            and str(s.get("status", "scheduled")).lower() not in TERMINAL_STATUSES
        )

    remaining_days = _remaining_deadline_days(plan, today)
    if remaining_days < 1:
        message = "The plan's original deadline has passed; extend the deadline before JASSI can safely rebuild the remaining roadmap."
        for old in overdue:
            old["replan_failure"] = message
        return [{"plan_id": plan.get("id"), "original": old, "replacement": None,
                 "reason": message} for old in overdue]

    prefs = plan.get("study_preferences") or student.get("study_preferences") or {}
    try:
        preferred_time = None
        if plan.get("preferred_time_start") is not None and plan.get("preferred_time_end") is not None:
            preferred_time = (int(plan["preferred_time_start"]), int(plan["preferred_time_end"]))
        scheduled = scheduler.build_schedule_dated(
            sessions_needed=tasks,
            calendar_events=[e for e in calendar if e.get("type") != "study_session"],
            start_date=today,
            deadline_days=remaining_days,
            allowed_days=plan.get("preferred_days") or None,
            existing_dated_sessions=other_plan_sessions,
            time_window=preferred_time,
            max_daily_study_hours=float(prefs.get("max_daily_study_hours", 3.0)),
            wake_start=prefs.get("wake_start", "07:00"),
            sleep_start=prefs.get("sleep_start", "23:00"),
            lunch_start=prefs.get("lunch_start", "12:30"),
            dinner_start=prefs.get("dinner_start", "19:30"),
            break_minutes=int(prefs.get("break_minutes", 15)),
        )
        # The scheduler returns its task metadata except private mapping fields;
        # recover identity using stable task order/topic/duration/resource tuple.
        source_queues: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
        for task, (source, is_recovery) in zip(tasks, [(s, s.get("id") in recovery_ids) for s in ordered_sources]):
            key = (task["topic"], task["duration_mins"], task.get("resource_id"), task.get("resource_title"), task.get("resource_url"))
            source_queues.setdefault(key, []).append({"source": source, "recovery": is_recovery})
        candidates = []
        for placed in scheduled["scheduled"]:
            key = (placed.get("topic"), placed.get("duration_mins"), placed.get("resource_id"), placed.get("resource_title"), placed.get("resource_url"))
            source_info = source_queues[key].pop(0)
            source = source_info["source"]
            # Retain user/resource metadata that the scheduler does not need
            # to inspect (e.g. task descriptions and resource annotations).
            placed = {**source, **placed}
            placed["id"] = f"sess_{uuid.uuid4().hex[:10]}" if source_info["recovery"] else source["id"]
            if source_info["recovery"]:
                placed["rescheduled_from"] = source.get("id")
                placed["original_date"] = source.get("date")
                placed["replan_reason"] = "Automatically recovered as part of whole-roadmap replanning"
            else:
                placed["replanned_from_date"] = source.get("date")
            placed["status"] = "scheduled"
            placed["created_at"] = now
            candidates.append(placed)

        unscheduled = scheduled.get("unscheduled", [])
        candidate_all = candidates + other_plan_sessions
        candidate_calendar = [
            e for e in calendar
            if e.get("type") != "study_session" or str(e.get("id")) not in {s.get("id") for s in sessions}
        ] + [_calendar_event(s) for s in candidates]
        verification = verifier.verify_plan(
            sessions=candidate_all,
            calendar_events=candidate_calendar,
            topics_data=topics_data,
            deadline_days=remaining_days,
        )
        if unscheduled:
            verification["passed"] = False
            verification.setdefault("issues", []).append(
                f"{len(unscheduled)} remaining session(s) could not fit before the original deadline."
            )
    except (KeyError, TypeError, ValueError, OSError) as exc:
        candidates = []
        verification = {"passed": False, "issues": [f"Whole-roadmap recovery failed: {exc}"]}

    for old in overdue:
        old["replan_verification"] = verification

    if not verification.get("passed"):
        message = "The remaining roadmap was not changed because the complete candidate schedule failed verification: " + "; ".join(verification.get("issues", [])[:4])
        for old in overdue:
            old["replan_failure"] = message
        return [{"plan_id": plan.get("id"), "original": old, "replacement": None,
                 "reason": message, "verification": verification} for old in overdue]

    # Atomic commit: replace all active sessions for this plan, preserve history,
    # and rebuild only this plan's calendar blocks.
    historical = [s for s in sessions if str(s.get("status", "scheduled")).lower() in TERMINAL_STATUSES]
    plan["sessions"] = historical + candidates
    old_plan_ids = {s.get("id") for s in sessions}
    calendar[:] = [
        e for e in calendar
        if not (e.get("type") == "study_session" and str(e.get("id")) in old_plan_ids)
    ]
    calendar.extend(_calendar_event(s) for s in candidates)

    replacements = {s.get("rescheduled_from"): s for s in candidates if s.get("rescheduled_from")}
    for old in overdue:
        replacement = replacements.get(old.get("id"))
        if replacement:
            old["replanned_to"] = replacement["id"]
    plan["replan_count"] = int(plan.get("replan_count", 0) or 0) + len(overdue)
    plan["roadmap_replanned_at"] = now
    plan.setdefault("replan_history", []).append({
        "type": "whole_roadmap_recovery",
        "missed_session_ids": [s.get("id") for s in overdue],
        "replacement_ids": [s.get("replanned_to") for s in overdue],
        "sessions_rescheduled": len(candidates),
        "reason": "Rebuilt all remaining sessions in prerequisite order and verified the complete roadmap",
        "at": now,
    })

    changes = []
    for old in overdue:
        replacement = replacements.get(old.get("id"))
        changes.append({"plan_id": plan.get("id"), "original": old,
                        "replacement": replacement, "roadmap_sessions_rescheduled": len(candidates),
                        "verification": verification})
    return changes


def reconcile_overdue_sessions(student: Dict[str, Any], today: date | None = None,
                               force_session_ids: set[str] | None = None) -> List[Dict[str, Any]]:
    """Detect overdue sessions and transactionally replan each affected roadmap.

    Re-running this function is idempotent: original missed records are retained
    and carry a processed marker, while successful replacements are persisted.
    """
    today = today or date.today()
    changes: List[Dict[str, Any]] = []
    for plan in student.get("planning_requests", []):
        if plan.get("status", "active") not in ACTIVE_PLAN_STATUSES:
            continue
        plan_session_ids = {str(s.get("id")) for s in plan.get("sessions", [])}
        forced_for_plan = {str(sid) for sid in (force_session_ids or set()) if str(sid) in plan_session_ids}
        changes.extend(_recover_plan(student, plan, today, forced_for_plan))
    return changes
