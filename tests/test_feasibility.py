import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import unittest
from datetime import date
from app.tools import feasibility


class TestFeasibility(unittest.TestCase):
    def setUp(self):
        self.topics_data = feasibility.load_topics(
            os.path.join(os.path.dirname(__file__), "..", "data", "topics.json"))
        self.calendar = [
            {"id": "e1", "title": "Class", "day": "Monday", "start": "09:00", "end": "10:30", "type": "hard"},
            {"id": "e2", "title": "Club", "day": "Thursday", "start": "17:00", "end": "18:30", "type": "soft"},
        ]

    def test_required_hours_scales_with_gap(self):
        low_gap = feasibility.required_hours_for_pr(["trees"], 60, {"trees": 55}, self.topics_data)
        high_gap = feasibility.required_hours_for_pr(["trees"], 90, {"trees": 20}, self.topics_data)
        self.assertLess(low_gap, high_gap)

    def test_required_hours_zero_when_already_mastered(self):
        result = feasibility.required_hours_for_pr(["trees"], 50, {"trees": 80}, self.topics_data)
        self.assertEqual(result, 0.0)

    def test_available_hours_nonnegative(self):
        hours = feasibility.available_hours_in_window(self.calendar, 7)
        self.assertGreaterEqual(hours, 0)

    def test_calculate_feasibility_shortfall(self):
        result = feasibility.calculate_feasibility(
            topics=["graphs"], target_mastery=95, deadline_days=1,
            current_mastery={"graphs": 0}, calendar_events=self.calendar,
            topics_data=self.topics_data,
        )
        self.assertGreaterEqual(result["shortfall_hours"], 0)
        self.assertIsInstance(result["feasible"], bool)

    def test_min_deadline_extension_positive(self):
        ext = feasibility.min_deadline_extension_days(5.0, avg_free_hours_per_day=1.0)
        self.assertEqual(ext, 5)

    def test_daily_study_threshold_caps_available_hours(self):
        low = feasibility.available_hours_in_window(self.calendar, 7, max_daily_study_hours=1.0)
        high = feasibility.available_hours_in_window(self.calendar, 7, max_daily_study_hours=5.0)
        self.assertLessEqual(low, 1.0 * 7)
        self.assertLess(low, high)

    def test_sleep_window_reduces_available_capacity(self):
        early_sleep = feasibility.available_hours_in_window(self.calendar, 7, sleep_start="20:00")
        late_sleep = feasibility.available_hours_in_window(self.calendar, 7, sleep_start="23:00")
        self.assertLessEqual(early_sleep, late_sleep)


    def test_no_preference_with_missing_deadline_does_not_mean_zero_capacity(self):
        hours = feasibility.available_hours_in_window(
            [], 0, allowed_days=None, max_daily_study_hours=3.0,
            start_date=date(2026, 10, 31),
        )
        self.assertGreater(hours, 0.0)

    def test_short_deadline_uses_actual_weekday_not_monday(self):
        # Saturday is free, while Monday is fully occupied. A one-day window
        # beginning Saturday must count Saturday's capacity.
        calendar = [
            {"id": "busy-mon", "title": "All-day commitment", "day": "Monday",
             "start": "07:00", "end": "23:00", "type": "hard"}
        ]
        hours = feasibility.available_hours_in_window(
            calendar, 1, allowed_days=None, max_daily_study_hours=3.0,
            start_date=date(2026, 10, 31),
        )
        self.assertGreater(hours, 0.0)

if __name__ == "__main__":
    unittest.main()
