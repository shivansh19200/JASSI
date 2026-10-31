import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app.nodes.tool_nodes import node_calculate_feasibility


class TestToolNodes(unittest.TestCase):
    def test_feasibility_node_ignores_scheduler_only_preferences(self):
        """Meal and break preferences belong to the scheduler, not feasibility math."""
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        state = {
            "topics_path": os.path.join(project_root, "data", "topics.json"),
            "dynamic_topics": [],
            "pr_topics": ["graphs"],
            "pr_target_mastery": 80,
            "pr_deadline_days": 7,
            "current_mastery": {"graphs": 20},
            "calendar_events": [],
            "preferred_days": None,
            "study_preferences": {
                "max_daily_study_hours": 3.0,
                "wake_start": "07:00",
                "sleep_start": "23:00",
                "lunch_start": "12:30",
                "dinner_start": "19:30",
                "break_minutes": 15,
            },
        }

        result = node_calculate_feasibility(state)

        self.assertIn("feasibility_result", result)
        self.assertIn("feasible", result["feasibility_result"])


if __name__ == "__main__":
    unittest.main()
