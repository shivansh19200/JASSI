"""LangGraph workflow definition for JASSI.

Two graphs are exposed:
  - planning_request_graph: parse -> feasibility -> (schedule | trade-offs)
  - daily_update_graph: parse update -> update mastery -> analyze -> replan -> verify -> notify

Both interleave LLM nodes (Groq) with deterministic tool nodes, matching the
hybrid architecture required by the spec: LLM handles perception/communication/
reasoning, Python handles math/logic/verification.
"""
from __future__ import annotations
from typing import Dict, Any, List, TypedDict, Optional
from langgraph.graph import StateGraph, END

from app.nodes import llm_nodes, tool_nodes
from app.tools import feasibility as feasibility_tool
from app.tools import scheduler as scheduler_tool
from app.vectorstore.resource_retriever import ResourceRetriever


class PlanningState(TypedDict, total=False):
    client: Any
    topics_path: str
    raw_text: str
    known_topic_ids: List[str]
    pr_topics: List[str]
    pr_target_mastery: int
    pr_deadline_days: int
    current_mastery: Dict[str, int]
    calendar_events: List[Dict[str, Any]]
    feasibility_result: Dict[str, Any]
    feasibility_explanation: str
    tradeoffs: Dict[str, Any]
    selected_tradeoff: Optional[str]
    moved_soft_event_ids: List[str]
    sessions_needed: List[Dict[str, Any]]
    schedule_result: Dict[str, Any]
    existing_sessions: List[Dict[str, Any]]
    verification_result: Dict[str, Any]
    schedule_explanation: str
    retriever: Any
    unmatched_reason: Optional[str]
    is_learning_request: bool
    preferred_days: List[str]
    dynamic_topics: List[Dict[str, Any]]
    resource_path_options: List[Dict[str, Any]]
    start_date: Any


def node_parse_pr(state: PlanningState) -> PlanningState:
    parsed = llm_nodes.parse_planning_request(state["client"], state["raw_text"], state["known_topic_ids"])
    state["pr_topics"] = [t for t in parsed.get("topics", []) if t in state["known_topic_ids"]]
    state["pr_target_mastery"] = int(parsed.get("target_mastery", 70))
    state["pr_deadline_days"] = int(parsed.get("deadline_days", 7))
    state["is_learning_request"] = bool(parsed.get("is_learning_request", True))
    state["preferred_days"] = [d for d in parsed.get("preferred_days", []) if d]
    # The model may still occasionally return an id outside the known list (or nothing at all).
    # Never silently fall back to a random/unrelated topic — treat that as "no match" too.
    if not state["pr_topics"]:
        state["unmatched_reason"] = parsed.get("unmatched_reason") or (
            "That doesn't match any topic JASSI's curated catalog currently has."
        )
    else:
        state["unmatched_reason"] = None
    return state


def node_synthesize_topic(state: PlanningState) -> PlanningState:
    """Runs when the request didn't match the curated catalog but does look
    like a genuine learning goal (e.g. "system design"). Rather than just
    refusing, asks the LLM to invent a lightweight topic definition so the
    rest of the pipeline (feasibility, scheduling, resource paths) can still
    build a real plan for it."""
    topics_data = feasibility_tool.load_topics(state["topics_path"])
    spec = llm_nodes.synthesize_dynamic_topic(state["client"], state["raw_text"], topics_data["topics"])
    topic_id = spec.get("id")
    if not topic_id:
        # The LLM itself agreed this isn't a learnable subject — keep the refusal.
        state["unmatched_reason"] = state.get("unmatched_reason") or (
            "That doesn't look like a learning goal JASSI can build a plan for."
        )
        return state

    topic_id = str(topic_id).lower().replace(" ", "_")
    known_ids = {t["id"] for t in topics_data["topics"]}
    valid_prereqs = [p for p in spec.get("prerequisites", []) if p in known_ids]
    new_topic = {
        "id": topic_id,
        "name": spec.get("name") or state["raw_text"][:40].title(),
        "estimated_hours": float(spec.get("estimated_hours") or 12),
        "prerequisites": valid_prereqs,
        "subtopics": spec.get("subtopics", []),
        "dynamic": True,
    }
    state["dynamic_topics"] = (state.get("dynamic_topics") or []) + [new_topic]
    state["pr_topics"] = [topic_id]
    state["current_mastery"].setdefault(topic_id, 0)
    state["unmatched_reason"] = None
    return state


def node_feasibility_explanation(state: PlanningState) -> PlanningState:
    state["feasibility_explanation"] = llm_nodes.generate_feasibility_explanation(
        state["client"], state["feasibility_result"], state["pr_topics"]
    )
    return state


def _prereq_depth(tid, lookup, seen=None):
    seen = seen or set()
    if tid in seen:
        return 0
    seen.add(tid)
    prereqs = lookup.get(tid, {}).get("prerequisites", [])
    if not prereqs:
        return 0
    return 1 + max((_prereq_depth(p, lookup, seen) for p in prereqs), default=0)


def node_generate_resource_paths(state: PlanningState) -> PlanningState:
    """Instead of silently auto-picking resources (the old behavior), this
    now asks the LLM for 3 alternative playlists PER TOPIC — each with pros
    and cons — so the student picks their own trajectory instead of JASSI
    deciding unilaterally. For catalog topics the paths are grounded in the
    real vector-retrieved candidates; for a dynamic (LLM-synthesized) topic
    there's no resource DB, so the paths are LLM-invented outlines."""
    topics_data = feasibility_tool.merge_dynamic_topics(
        feasibility_tool.load_topics(state["topics_path"]), state.get("dynamic_topics") or []
    )
    lookup = {t["id"]: t for t in topics_data["topics"]}
    retriever: ResourceRetriever = state["retriever"]

    ordered_topics = sorted(state["pr_topics"], key=lambda t: _prereq_depth(t, lookup))
    topic_choices = []
    for topic in ordered_topics:
        info = lookup.get(topic, {})
        is_dynamic = info.get("dynamic", False)
        candidate_resources = None
        if not is_dynamic:
            current = state["current_mastery"].get(topic, 0)
            candidates = retriever.similarity_search(f"{topic} concepts and practice", k=8, topic_filter=topic)
            candidate_resources = [
                {"id": r["id"], "title": r["title"], "duration_mins": r["duration_mins"],
                 "difficulty": r["difficulty"]} for r in candidates
            ] or None
        result = llm_nodes.suggest_resource_paths(
            state["client"], info.get("name", topic),
            subtopics=info.get("subtopics"), candidate_resources=candidate_resources,
        )
        topic_choices.append({
            "topic": topic, "topic_name": info.get("name", topic),
            "dynamic": is_dynamic, "paths": result.get("paths", []),
        })
    state["resource_path_options"] = topic_choices
    return state


def build_sessions_from_chosen_paths(chosen_paths: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """chosen_paths: {topic_id: path_dict} (the path the student picked for
    each topic). Converts them into the flat sessions_needed list the
    scheduler expects, preserving topic order as the outer ordering key."""
    sessions_needed = []
    for order, (topic, path) in enumerate(chosen_paths.items()):
        for sess in path.get("sessions", []):
            sessions_needed.append({
                "topic": topic,
                "resource_id": sess.get("resource_id"),
                "resource_title": sess.get("title", f"{topic} session"),
                "duration_mins": int(sess.get("duration_mins") or 30),
                "order": order,
            })
    return sessions_needed


def node_generate_tradeoffs(state: PlanningState) -> PlanningState:
    topics_data = feasibility_tool.load_topics(state["topics_path"])
    soft_events = [e for e in state["calendar_events"] if e.get("type") == "soft"]
    min_ext = feasibility_tool.min_deadline_extension_days(state["feasibility_result"]["shortfall_hours"])
    state["tradeoffs"] = llm_nodes.generate_tradeoffs(
        state["client"], state["feasibility_result"], state["pr_topics"], soft_events, min_ext
    )
    return state


def node_schedule_explanation(state: PlanningState) -> PlanningState:
    topics_data = feasibility_tool.merge_dynamic_topics(
        feasibility_tool.load_topics(state["topics_path"]), state.get("dynamic_topics") or []
    )
    state["schedule_explanation"] = llm_nodes.generate_schedule_explanation(
        state["client"], state["schedule_result"]["scheduled"], topics_data
    )
    return state


def route_after_feasibility(state: PlanningState) -> str:
    return "feasible" if state["feasibility_result"]["feasible"] else "infeasible"


def route_after_parse(state: PlanningState) -> str:
    if state.get("pr_topics"):
        return "matched"
    # Out-of-catalog but still looks like a real learning goal -> try to
    # synthesize a topic for it instead of refusing outright.
    return "synthesize" if state.get("is_learning_request", True) else "unmatched"


def route_after_synthesize(state: PlanningState) -> str:
    return "matched" if state.get("pr_topics") else "unmatched"


def node_noop(state: PlanningState) -> PlanningState:
    return state


def build_planning_request_graph():
    graph = StateGraph(PlanningState)
    graph.add_node("parse_planning_request", node_parse_pr)
    graph.add_node("synthesize_topic", node_synthesize_topic)
    graph.add_node("calculate_feasibility", tool_nodes.node_calculate_feasibility)
    graph.add_node("generate_feasibility_explanation", node_feasibility_explanation)
    graph.add_node("generate_resource_paths", node_generate_resource_paths)
    graph.add_node("generate_tradeoffs", node_generate_tradeoffs)
    graph.add_node("no_topic_match", node_noop)

    graph.set_entry_point("parse_planning_request")
    # If nothing in the curated catalog matched, try synthesizing a fresh
    # topic for it (as long as it looks like a genuine learning goal) instead
    # of refusing outright — only truly gives up if that also fails.
    graph.add_conditional_edges(
        "parse_planning_request",
        route_after_parse,
        {"matched": "calculate_feasibility", "synthesize": "synthesize_topic", "unmatched": "no_topic_match"},
    )
    graph.add_conditional_edges(
        "synthesize_topic",
        route_after_synthesize,
        {"matched": "calculate_feasibility", "unmatched": "no_topic_match"},
    )
    graph.add_edge("no_topic_match", END)
    graph.add_edge("calculate_feasibility", "generate_feasibility_explanation")
    # Feasible -> offer resource-path choices and stop; the app finalizes the
    # schedule once the student picks a path (see build_sessions_from_chosen_paths
    # + tool_nodes.node_build_schedule, called directly from the UI, mirroring
    # how a trade-off choice is resolved below).
    graph.add_conditional_edges(
        "generate_feasibility_explanation",
        route_after_feasibility,
        {"feasible": "generate_resource_paths", "infeasible": "generate_tradeoffs"},
    )
    graph.add_edge("generate_resource_paths", END)
    graph.add_edge("generate_tradeoffs", END)
    return graph.compile()


# ---------------------------------------------------------------------------
# Daily update graph
# ---------------------------------------------------------------------------
class UpdateState(TypedDict, total=False):
    client: Any
    topics_path: str
    raw_text: str
    session: Dict[str, Any]
    parsed_update: Dict[str, Any]
    quiz_score: Optional[int]
    update_topic: str
    current_mastery: Dict[str, int]
    mastery_delta: int
    calendar_events: List[Dict[str, Any]]
    existing_sessions: List[Dict[str, Any]]
    rescheduled_session: Optional[Dict[str, Any]]
    reschedule_explanation: str
    progress_insight: str
    quiz_history: List[Dict[str, Any]]


def node_parse_update(state: UpdateState) -> UpdateState:
    state["parsed_update"] = llm_nodes.parse_update(
        state["client"], state["raw_text"], [state["session"]["topic"]]
    )
    return state


def node_update_mastery_wrapper(state: UpdateState) -> UpdateState:
    return tool_nodes.node_update_mastery(state)


def node_reschedule_if_missed(state: UpdateState) -> UpdateState:
    """Reacts to what the LLM actually parsed out of the check-in, not just
    a binary missed/not-missed: a fully missed session gets rebooked in
    full, and a partial session below the completion threshold gets a
    shorter catch-up session for the remaining chunk of work."""
    parsed = state["parsed_update"]
    status = parsed.get("status")
    percent_done = parsed.get("percent_done", 0) or 0
    state["rescheduled_session"] = None

    if status == "missed":
        target_session = state["session"]
        exclude_day = state["session"]["day"]
        reason = "it's the next slot that doesn't clash with your other commitments"
    elif status == "partial" and percent_done < 70:
        remaining_pct = max(0, 100 - percent_done)
        original_duration = state["session"].get("duration_mins", 30)
        catchup_duration = max(15, round((original_duration * remaining_pct / 100) / 15) * 15)
        target_session = dict(state["session"])
        target_session["duration_mins"] = catchup_duration
        exclude_day = None
        reason = f"it's a short catch-up for the {remaining_pct}% of the topic left unfinished"
    else:
        return state

    if target_session.get("date"):
        import datetime
        new_session = scheduler_tool.reschedule_single_dated(
            target_session, state["calendar_events"], state["existing_sessions"],
            start_date=datetime.date.today(), exclude_date=target_session.get("date"),
        )
    else:
        new_session = scheduler_tool.reschedule_single(
            target_session, state["calendar_events"], state["existing_sessions"],
            exclude_day=exclude_day,
        )
    state["rescheduled_session"] = new_session
    if new_session:
        state["reschedule_explanation"] = llm_nodes.generate_reschedule_explanation(
            state["client"], state["session"], new_session, reason,
        )
    return state


def node_analyze_progress(state: UpdateState) -> UpdateState:
    struggle_note = (state.get("parsed_update") or {}).get("struggle_note")
    state["progress_insight"] = llm_nodes.analyze_progress_patterns(
        state["client"], state["current_mastery"], state.get("quiz_history", []),
        struggle_note=struggle_note,
    )
    return state


def build_daily_update_graph():
    graph = StateGraph(UpdateState)
    graph.add_node("parse_update", node_parse_update)
    graph.add_node("update_mastery", node_update_mastery_wrapper)
    graph.add_node("reschedule_if_missed", node_reschedule_if_missed)
    graph.add_node("analyze_progress", node_analyze_progress)

    graph.set_entry_point("parse_update")
    graph.add_edge("parse_update", "update_mastery")
    graph.add_edge("update_mastery", "reschedule_if_missed")
    graph.add_edge("reschedule_if_missed", "analyze_progress")
    graph.add_edge("analyze_progress", END)
    return graph.compile()
