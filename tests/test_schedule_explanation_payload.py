import unittest

from app.nodes.llm_nodes import generate_schedule_explanation


class FakeClient:
    def __init__(self):
        self.user_prompt = None

    def chat(self, system_prompt, user_prompt, temperature=0.7, json_mode=False):
        self.user_prompt = user_prompt
        return "A concise schedule overview."


class ScheduleExplanationPayloadTests(unittest.TestCase):
    def test_sends_compact_summary_not_full_resource_payload(self):
        client = FakeClient()
        sessions = [{
            "topic": "arrays", "day": "Monday", "start": "09:00",
            "duration_mins": 45, "resource_url": "https://example.com/" + "x" * 2000,
            "resources": [{"title": "resource" * 100, "url": "https://example.com"}],
        } for _ in range(100)]
        topics = {"topics": [{"id": "arrays", "name": "Arrays", "prerequisites": []}]}

        result = generate_schedule_explanation(client, sessions, topics)

        self.assertEqual(result, "A concise schedule overview.")
        self.assertLessEqual(len(client.user_prompt), 8000)
        self.assertIn('"total_sessions":100', client.user_prompt)
        self.assertIn('"total_hours":75.0', client.user_prompt)
        self.assertNotIn("resource_url", client.user_prompt)
        self.assertNotIn("example.com", client.user_prompt)


if __name__ == "__main__":
    unittest.main()
