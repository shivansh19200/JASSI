import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import unittest
from app.tools import feasibility, verifier


class TestVerifier(unittest.TestCase):
    def setUp(self):
        self.topics_data = feasibility.load_topics(
            os.path.join(os.path.dirname(__file__), "..", "data", "topics.json"))
        self.calendar = [
            {"id": "e1", "title": "Class", "day": "Monday", "start": "09:00", "end": "10:30", "type": "hard"},
        ]

    def test_passes_with_no_conflicts(self):
        sessions = [
            {"id": "s1", "topic": "trees", "day": "Monday", "start": "18:00", "end": "18:45"},
        ]
        result = verifier.verify_plan(sessions, self.calendar, self.topics_data, deadline_days=7)
        self.assertTrue(result["passed"])
        self.assertEqual(result["issues"], [])

    def test_detects_hard_constraint_conflict(self):
        sessions = [
            {"id": "s1", "topic": "trees", "day": "Monday", "start": "09:30", "end": "10:00"},
        ]
        result = verifier.verify_plan(sessions, self.calendar, self.topics_data, deadline_days=7)
        self.assertFalse(result["passed"])
        self.assertTrue(any("conflicts with hard constraint" in i for i in result["issues"]))

    def test_detects_session_overlap(self):
        sessions = [
            {"id": "s1", "topic": "trees", "day": "Tuesday", "start": "18:00", "end": "18:45"},
            {"id": "s2", "topic": "graphs", "day": "Tuesday", "start": "18:30", "end": "19:15"},
        ]
        result = verifier.verify_plan(sessions, self.calendar, self.topics_data, deadline_days=7)
        self.assertFalse(result["passed"])
        self.assertTrue(any("overlaps" in i for i in result["issues"]))

    def test_detects_prerequisite_violation(self):
        sessions = [
            {"id": "s1", "topic": "graphs", "day": "Monday", "start": "18:00", "end": "18:45"},
            {"id": "s2", "topic": "trees", "day": "Tuesday", "start": "18:00", "end": "18:45"},
        ]
        result = verifier.verify_plan(sessions, self.calendar, self.topics_data, deadline_days=7)
        self.assertFalse(result["passed"])
        self.assertTrue(any("Prerequisite violation" in i for i in result["issues"]))


if __name__ == "__main__":
    unittest.main()
