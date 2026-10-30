import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import unittest
from app.tools import scheduler


class TestScheduler(unittest.TestCase):
    def setUp(self):
        self.calendar = [
            {"id": "e1", "title": "Class", "day": "Monday", "start": "09:00", "end": "10:30", "type": "hard"},
            {"id": "e2", "title": "Club", "day": "Thursday", "start": "17:00", "end": "18:30", "type": "soft"},
        ]
        self.sessions_needed = [
            {"topic": "trees", "duration_mins": 45, "resource_id": "r1", "resource_title": "Trees Concept", "order": 0},
            {"topic": "graphs", "duration_mins": 45, "resource_id": "r2", "resource_title": "Graphs Concept", "order": 1},
        ]

    def test_build_schedule_places_all_sessions(self):
        result = scheduler.build_schedule(self.sessions_needed, self.calendar)
        self.assertEqual(len(result["scheduled"]), 2)
        self.assertEqual(len(result["unscheduled"]), 0)

    def test_scheduled_sessions_avoid_hard_conflicts(self):
        result = scheduler.build_schedule(self.sessions_needed, self.calendar)
        for s in result["scheduled"]:
            if s["day"] == "Monday":
                self.assertFalse(s["start"] < "10:30" and s["end"] > "09:00")

    def test_scheduled_sessions_avoid_soft_conflicts_by_default(self):
        result = scheduler.build_schedule(self.sessions_needed, self.calendar)
        for s in result["scheduled"]:
            if s["day"] == "Thursday":
                self.assertFalse(s["start"] < "18:30" and s["end"] > "17:00")

    def test_moved_soft_events_free_up_slot(self):
        result = scheduler.build_schedule(self.sessions_needed, self.calendar, moved_soft_event_ids=["e2"])
        self.assertEqual(len(result["unscheduled"]), 0)

    def test_reschedule_single_finds_new_slot(self):
        session = {"id": "s1", "topic": "trees", "day": "Monday", "start": "11:00", "end": "11:45",
                   "duration_mins": 45, "resource_title": "Trees Concept"}
        new_session = scheduler.reschedule_single(session, self.calendar, [session], exclude_day="Monday")
        self.assertIsNotNone(new_session)
        self.assertNotEqual(new_session["day"], "Monday")

    def test_does_not_schedule_outside_waking_window(self):
        result = scheduler.build_schedule_dated(
            self.sessions_needed, self.calendar,
            start_date=__import__("datetime").date(2026, 11, 2), deadline_days=7,
            wake_start="09:00", sleep_start="18:00", max_daily_study_hours=2.0,
        )
        for session in result["scheduled"]:
            self.assertGreaterEqual(int(session["start"][:2]), 9)
            self.assertLessEqual(int(session["end"][:2]), 18)

    def test_does_not_schedule_during_meal_breaks(self):
        result = scheduler.build_schedule_dated(
            [{"topic": "trees", "duration_mins": 60, "order": 0}], [],
            start_date=__import__("datetime").date(2026, 11, 2), deadline_days=1,
            wake_start="07:00", sleep_start="23:00", lunch_start="12:00",
            dinner_start="19:00", meal_break_mins=60,
        )
        self.assertEqual(len(result["scheduled"]), 1)
        session = result["scheduled"][0]
        self.assertFalse(session["start"] < "13:00" and session["end"] > "12:00")
        self.assertFalse(session["start"] < "20:00" and session["end"] > "19:00")


if __name__ == "__main__":
    unittest.main()
