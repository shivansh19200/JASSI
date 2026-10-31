import json
import unittest
from app.tools.constraint_parser import parse_image_with_llm, _normalize_timetable_time


class FakeVisionClient:
    def __init__(self, response):
        self.response = response

    def chat_vision(self, system_prompt, image_b64, **kwargs):
        self.system_prompt = system_prompt
        return self.response


class TestTimetableImageParser(unittest.TestCase):
    def test_normalizes_am_pm_times(self):
        self.assertEqual(_normalize_timetable_time("8:00AM"), "08:00")
        self.assertEqual(_normalize_timetable_time("12:00PM"), "12:00")
        self.assertEqual(_normalize_timetable_time("12:00AM"), "00:00")
        self.assertEqual(_normalize_timetable_time("2:30 PM"), "14:30")

    def test_parses_and_validates_recurring_weekly_events(self):
        payload = {"events": [
            {"title": "CS F222 Tutorial", "day": "Monday", "start": "8:00AM", "end": "9:00AM", "type": "hard"},
            {"title": "CS F222 Tutorial", "day": "Monday", "start": "8:00AM", "end": "9:00AM", "type": "hard"},
            {"title": "CS F215 Lab", "day": "Wednesday", "start": "2:00PM", "end": "4:00PM", "type": "hard"},
            {"title": "Bad row", "day": "Funday", "start": "9:00AM", "end": "10:00AM", "type": "hard"},
            {"title": "Bad time", "day": "Friday", "start": "9:90AM", "end": "10:00AM", "type": "hard"},
        ]}
        client = FakeVisionClient(json.dumps(payload))
        events = parse_image_with_llm(client, b"fake-image")
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["start"], "08:00")
        self.assertEqual(events[1]["start"], "14:00")
        self.assertIn("recurring weekly", client.system_prompt)

    def test_rejects_empty_or_unreadable_timetable(self):
        with self.assertRaises(RuntimeError):
            parse_image_with_llm(FakeVisionClient('{"events": []}'), b"fake-image")


if __name__ == "__main__":
    unittest.main()
