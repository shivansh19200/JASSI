"""
JASSI demo script — runs the 7-step end-to-end scenario from the terminal,
printing each LLM-generated explanation and tool-computed result so judges
can see the agentic behavior without needing the full dashboard running.

Usage:
    python demo_script.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from app.nodes.llm_nodes import GroqClient
from app.tools import feasibility as feasibility_tool
from app.vectorstore.resource_retriever import ResourceRetriever
from app.graph import build_planning_request_graph, build_daily_update_graph

TOPICS_PATH = "data/topics.json"
RESOURCES_PATH = "data/resources.json"


def banner(step, text):
    print(f"\n{'=' * 70}\nSTEP {step}: {text}\n{'=' * 70}")


def main():
    client = GroqClient()
    retriever = ResourceRetriever(RESOURCES_PATH)
    topics_data = feasibility_tool.load_topics(TOPICS_PATH)
    known_ids = [t["id"] for t in topics_data["topics"]]

    calendar_events = [
        {"id": "e1", "title": "DSA Lecture", "day": "Monday", "start": "09:00", "end": "10:30", "type": "hard"},
        {"id": "e2", "title": "Physics Lab", "day": "Tuesday", "start": "14:00", "end": "16:00", "type": "hard"},
        {"id": "e3", "title": "Dance Club", "day": "Monday", "start": "18:00", "end": "19:30", "type": "soft"},
        {"id": "e4", "title": "Coding Club", "day": "Thursday", "start": "17:00", "end": "18:30", "type": "soft"},
    ]
    current_mastery = {"trees": 55, "graphs": 25}

    banner(2, "Creating planning request via natural language")
    user_text = "I wanna finish trees and graphs in 10 days with 70% mastery"
    print(f"User: {user_text}")

    graph = build_planning_request_graph()
    state = {
        "client": client, "topics_path": TOPICS_PATH, "raw_text": user_text,
        "known_topic_ids": known_ids, "current_mastery": current_mastery,
        "calendar_events": calendar_events, "retriever": retriever,
        "existing_sessions": [], "moved_soft_event_ids": [],
    }
    result = graph.invoke(state)

    banner(3, "LLM-parsed structured PR")
    print(json.dumps({"topics": result["pr_topics"], "target_mastery": result["pr_target_mastery"],
                       "deadline_days": result["pr_deadline_days"]}, indent=2))

    banner(4, "Deterministic feasibility calculation")
    print(json.dumps(result["feasibility_result"], indent=2))
    print(f"\nLLM explanation: {result['feasibility_explanation']}")

    if not result["feasibility_result"]["feasible"]:
        banner(5, "LLM-generated Reality Check (trade-offs)")
        print(json.dumps(result["tradeoffs"], indent=2))
        print("\n[Demo] Selecting option A: keep deadline, move soft commitments")
        moved_ids = [e["id"] for e in calendar_events if e["type"] == "soft"]
        from app.nodes import tool_nodes as tn
        from app.graph import node_select_resources, node_schedule_explanation
        state["moved_soft_event_ids"] = moved_ids
        state = tn.node_calculate_feasibility(state)
        state = node_select_resources(state)
        state = tn.node_build_schedule(state)
        state = tn.node_verify_plan(state)
        state = node_schedule_explanation(state)
        result = state

    banner(6, "Vector DB semantic resource selection + schedule built")
    for s in result["schedule_result"]["scheduled"]:
        print(f"  {s['day']:9s} {s['start']}-{s['end']}  {s['topic']:8s}  {s['resource_title']}")

    banner(7, "Plan verification")
    print(json.dumps(result["verification_result"], indent=2))
    print(f"\nLLM schedule explanation: {result['schedule_explanation']}")

    banner(8, "Daily update: student misses a session")
    sessions = result["schedule_result"]["scheduled"]
    missed_session = sessions[0]
    update_graph = build_daily_update_graph()
    update_state = {
        "client": client, "topics_path": TOPICS_PATH,
        "raw_text": "didn't get to it today, was slammed with other work",
        "session": missed_session, "quiz_score": None,
        "update_topic": missed_session["topic"], "current_mastery": current_mastery,
        "calendar_events": calendar_events + [
            {"id": s["id"], "title": s["topic"], "day": s["day"], "start": s["start"],
             "end": s["end"], "type": "study_session"} for s in sessions if s["id"] != missed_session["id"]
        ],
        "existing_sessions": sessions, "quiz_history": [],
    }
    update_result = update_graph.invoke(update_state)
    print(f"Parsed update: {update_result['parsed_update']}")
    if update_result.get("rescheduled_session"):
        print(f"Rescheduled to: {update_result['rescheduled_session']['day']} "
              f"{update_result['rescheduled_session']['start']}")
        print(f"LLM explanation: {update_result['reschedule_explanation']}")

    banner(9, "Quiz score submitted -> mastery update + progress insight")
    update_state2 = dict(update_state)
    update_state2["raw_text"] = "just finished, felt pretty good about it"
    update_state2["quiz_score"] = 40
    update_state2["update_topic"] = "graphs"
    update_result2 = update_graph.invoke(update_state2)
    print(f"New mastery: {update_result2['current_mastery']}")
    print(f"LLM insight: {update_result2['progress_insight']}")

    print("\nDemo complete.")


if __name__ == "__main__":
    main()
