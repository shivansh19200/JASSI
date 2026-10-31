import json
import unittest
from app.tools.constraint_parser import (
    parse_image_with_llm, _normalize_timetable_time,
    find_timetable_conflicts, merge_timetable_import,
)


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


    def test_normalizes_24_hour_and_ocr_punctuation_times(self):
        self.assertEqual(_normalize_timetable_time("08:00"), "08:00")
        self.assertEqual(_normalize_timetable_time("8.50"), "08:50")
        self.assertEqual(_normalize_timetable_time("0800"), "08:00")
        self.assertEqual(_normalize_timetable_time("6:30 PM"), "18:30")

    def test_normalizes_dotted_meridiem(self):
        self.assertEqual(_normalize_timetable_time("8.30 a.m."), "08:30")
        self.assertEqual(_normalize_timetable_time("2:15 p.m."), "14:15")

    def test_uses_populated_alias_when_canonical_time_field_is_empty(self):
        response = {"events": [{
            "title": "", "course": "Maths Tutorial",
            "day": None, "weekday": "Tue.",
            "start": None, "start_time": "9:00 AM",
            "end": "", "end_time": "10:00 AM",
        }]}
        events = parse_image_with_llm(FakeVisionClient(json.dumps(response)), b"fake-image")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["title"], "Maths Tutorial")
        self.assertEqual(events[0]["day"], "Tuesday")
        self.assertEqual(events[0]["start"], "09:00")
        self.assertEqual(events[0]["end"], "10:00")

    def test_fills_missing_time_fields_from_range(self):
        response = {"events": [{
            "title": "Physics Lab", "day": "Friday",
            "start": None, "end": None, "time_range": "2:00 p.m. to 4:00 p.m."
        }]}
        events = parse_image_with_llm(FakeVisionClient(json.dumps(response)), b"fake-image")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["start"], "14:00")
        self.assertEqual(events[0]["end"], "16:00")

    def test_accepts_wrapped_json_and_alternate_event_field_names(self):
        response = "Here are the events:\n```json\n{\"events\":[{\"course\":\"PHY F101 Tutorial\",\"weekday\":\"Mon\",\"time_range\":\"09:00 - 09:50\"}]}\n```"
        events = parse_image_with_llm(FakeVisionClient(response), b"fake-image")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["day"], "Monday")
        self.assertEqual(events[0]["start"], "09:00")
        self.assertEqual(events[0]["end"], "09:50")

    def test_import_preserves_conflicting_commitments_by_default(self):
        existing = [{"id": "old", "title": "Lab", "day": "Monday", "start": "09:00", "end": "10:00", "type": "hard"}]
        imported = [{"id": "new", "title": "Lecture", "day": "Monday", "start": "09:30", "end": "10:30", "type": "hard"}]
        self.assertEqual(len(find_timetable_conflicts(imported, existing)), 1)
        merged, replaced = merge_timetable_import(existing, imported, override_conflicts=False)
        self.assertEqual(len(merged), 2)
        self.assertEqual(replaced, 0)

    def test_import_override_replaces_only_overlapping_events(self):
        existing = [
            {"id": "old", "title": "Lab", "day": "Monday", "start": "09:00", "end": "10:00", "type": "hard"},
            {"id": "keep", "title": "Meeting", "day": "Tuesday", "start": "09:00", "end": "10:00", "type": "hard"},
        ]
        imported = [{"id": "new", "title": "Lecture", "day": "Monday", "start": "09:30", "end": "10:30", "type": "hard"}]
        merged, replaced = merge_timetable_import(existing, imported, override_conflicts=True)
        self.assertEqual(replaced, 1)
        self.assertEqual({event["id"] for event in merged}, {"keep", "new"})

    def test_rejects_empty_or_unreadable_timetable(self):
        with self.assertRaises(RuntimeError):
            parse_image_with_llm(FakeVisionClient('{"events": []}'), b"fake-image")


if __name__ == "__main__":
    unittest.main()
