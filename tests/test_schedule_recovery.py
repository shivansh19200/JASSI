from datetime import date

from app.tools.schedule_recovery import reconcile_overdue_sessions


def _student(sessions):
    return {
        "study_preferences": {"wake_start": "08:00", "sleep_start": "22:00", "max_daily_study_hours": 3},
        "calendar_events": [],
        "planning_requests": [{"id": "plan-1", "status": "active", "sessions": sessions}],
    }


def test_overdue_session_is_replaced_once_and_calendar_is_updated():
    student = _student([{
        "id": "old-1", "topic": "machine_learning", "resource_title": "NumPy",
        "date": "2026-10-30", "day": "Friday", "start": "09:00", "end": "10:00",
        "duration_mins": 60, "status": "scheduled",
    }])

    changes = reconcile_overdue_sessions(student, date(2026, 10, 31))
    plan = student["planning_requests"][0]
    old = plan["sessions"][0]
    replacement = changes[0]["replacement"]

    assert old["status"] == "missed"
    assert old["replanned_to"] == replacement["id"]
    assert replacement["id"] != old["id"]
    assert replacement["date"] >= "2026-10-31"
    assert replacement["start"] >= "08:00"
    assert any(event["id"] == replacement["id"] for event in student["calendar_events"])
    assert not any(event["id"] == old["id"] for event in student["calendar_events"])

    # Streamlit reruns must not create another replacement.
    assert reconcile_overdue_sessions(student, date(2026, 10, 31)) == []
    assert len(plan["sessions"]) == 2


def test_completed_and_future_sessions_are_not_replanned():
    student = _student([
        {"id": "done", "topic": "ml", "date": "2026-10-29", "day": "Thursday",
         "start": "09:00", "end": "10:00", "duration_mins": 60, "status": "completed"},
        {"id": "future", "topic": "ml", "date": "2026-11-01", "day": "Sunday",
         "start": "09:00", "end": "10:00", "duration_mins": 60, "status": "scheduled"},
    ])
    assert reconcile_overdue_sessions(student, date(2026, 10, 31)) == []
    assert len(student["planning_requests"][0]["sessions"]) == 2


def test_inactive_plan_is_not_modified():
    student = _student([{
        "id": "old", "topic": "ml", "date": "2026-10-20", "day": "Tuesday",
        "start": "09:00", "end": "10:00", "duration_mins": 60, "status": "scheduled",
    }])
    student["planning_requests"][0]["status"] = "archived"
    assert reconcile_overdue_sessions(student, date(2026, 10, 31)) == []
    assert student["planning_requests"][0]["sessions"][0]["status"] == "scheduled"


def test_recovery_rebuilds_the_whole_remaining_roadmap_in_dependency_order():
    student = _student([
        {"id": "missed-arrays", "topic": "arrays", "date": "2026-10-30", "day": "Friday",
         "start": "09:00", "end": "10:00", "duration_mins": 60, "status": "scheduled"},
        {"id": "stacks", "topic": "stacks_queues", "date": "2026-11-02", "day": "Monday",
         "start": "09:00", "end": "10:00", "duration_mins": 60, "status": "scheduled"},
        {"id": "trees", "topic": "trees", "date": "2026-11-03", "day": "Tuesday",
         "start": "09:00", "end": "10:00", "duration_mins": 60, "status": "scheduled"},
        {"id": "graphs", "topic": "graphs", "date": "2026-11-04", "day": "Wednesday",
         "start": "09:00", "end": "10:00", "duration_mins": 60, "status": "scheduled"},
        {"id": "already-done", "topic": "sorting", "date": "2026-10-28", "day": "Wednesday",
         "start": "11:00", "end": "12:00", "duration_mins": 60, "status": "completed"},
    ])
    plan = student["planning_requests"][0]
    plan.update({"created_at": "2026-10-20", "deadline_days": 30})

    changes = reconcile_overdue_sessions(student, date(2026, 10, 31))

    assert len(changes) == 1
    assert changes[0]["roadmap_sessions_rescheduled"] == 4
    assert changes[0]["verification"]["passed"] is True
    completed = next(s for s in plan["sessions"] if s["id"] == "already-done")
    assert completed["status"] == "completed"

    active = [s for s in plan["sessions"] if s["status"] == "scheduled"]
    by_topic = {s["topic"]: s for s in active}
    assert by_topic["arrays"]["rescheduled_from"] == "missed-arrays"
    assert by_topic["stacks_queues"]["id"] == "stacks"
    assert by_topic["trees"]["id"] == "trees"
    assert by_topic["graphs"]["id"] == "graphs"
    assert by_topic["arrays"]["date"] < by_topic["stacks_queues"]["date"]
    assert by_topic["stacks_queues"]["date"] < by_topic["trees"]["date"]
    assert by_topic["trees"]["date"] < by_topic["graphs"]["date"]
    assert len([e for e in student["calendar_events"] if e["type"] == "study_session"]) == 4

    # A Streamlit rerun must not rebuild the same roadmap again.
    assert reconcile_overdue_sessions(student, date(2026, 10, 31)) == []


def test_recovery_does_not_silently_extend_an_expired_deadline():
    student = _student([{
        "id": "old", "topic": "arrays", "date": "2026-10-20", "day": "Tuesday",
        "start": "09:00", "end": "10:00", "duration_mins": 60, "status": "scheduled",
    }, {
        "id": "future", "topic": "stacks_queues", "date": "2026-10-21", "day": "Wednesday",
        "start": "09:00", "end": "10:00", "duration_mins": 60, "status": "scheduled",
    }])
    plan = student["planning_requests"][0]
    plan.update({"created_at": "2026-10-01", "deadline_days": 7})

    changes = reconcile_overdue_sessions(student, date(2026, 10, 31))

    assert changes[0]["replacement"] is None
    assert "original deadline has passed" in changes[0]["reason"]
    assert plan["sessions"][1]["id"] == "future"
    assert plan["sessions"][1]["date"] == "2026-10-21"


def test_checkin_miss_forces_whole_roadmap_recovery_even_before_date_rollover():
    student = _student([{
        "id": "today-session", "topic": "arrays", "date": "2026-10-31", "day": "Saturday",
        "start": "09:00", "end": "10:00", "duration_mins": 60, "status": "missed",
    }, {
        "id": "next-session", "topic": "stacks_queues", "date": "2026-11-01", "day": "Sunday",
        "start": "09:00", "end": "10:00", "duration_mins": 60, "status": "scheduled",
    }])
    plan = student["planning_requests"][0]
    plan.update({"created_at": "2026-10-20", "deadline_days": 30})

    changes = reconcile_overdue_sessions(
        student, date(2026, 10, 31), force_session_ids={"today-session"}
    )

    assert len(changes) == 1
    assert changes[0]["replacement"] is not None
    assert changes[0]["replacement"]["rescheduled_from"] == "today-session"
    assert changes[0]["roadmap_sessions_rescheduled"] == 2
