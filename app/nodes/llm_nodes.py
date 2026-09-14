"""LLM-powered nodes (Groq API, mandatory). Each function here makes an actual
call to Groq for natural language understanding, generation, or reasoning —
none of these are mocked or stubbed. The GroqClient wrapper centralizes
error handling so a missing/invalid API key produces one clear message
instead of scattered exceptions across the app.
"""
from __future__ import annotations
from typing import Dict, List, Any, Optional
import json
import os

from dotenv import load_dotenv
load_dotenv()


class GroqClient:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if self.api_key:
            self.api_key = self.api_key.strip().strip('"').strip("'")
        # llama-3.3-70b-versatile was decommissioned by Groq on 2026-08-16;
        # openai/gpt-oss-120b is Groq's recommended replacement.
        self.model = model or os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
        if not self.api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to your .env file — see .env.example."
            )
        from groq import Groq
        self._client = Groq(api_key=self.api_key)

    def chat(self, system_prompt: str, user_prompt: str, temperature: float = 0.7,
              json_mode: bool = False) -> str:
        kwargs = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        completion = self._client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **kwargs,
        )
        return completion.choices[0].message.content

    def chat_vision(self, system_prompt: str, image_b64: str,
                     vision_model: Optional[str] = None) -> str:
        vision_model = vision_model or os.environ.get("GROQ_VISION_MODEL", "qwen/qwen3.6-27b")
        completion = self._client.chat.completions.create(
            model=vision_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract the timetable events from this image."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    ],
                },
            ],
        )
        return completion.choices[0].message.content


def _safe_json_parse(raw: str, fallback: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = raw.strip().strip("`")
    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return fallback


# ---------------------------------------------------------------------------
# 1. parse_planning_request
# ---------------------------------------------------------------------------
def parse_planning_request(client: GroqClient, user_text: str, known_topic_ids: List[str]) -> Dict[str, Any]:
    system = (
        "You convert a student's casual planning request into structured JSON. "
        f"The curated catalog topic ids are: {known_topic_ids}. "
        "Only include a topic id in \"topics\" if the student's request is genuinely, substantively "
        "about that same subject (e.g. 'trees' -> 'trees', 'graph theory' -> 'graphs', "
        "'sorting algos' -> 'sorting'). Loose wording about one of these topics still counts as a match. "
        "Do NOT force a match just because it's the closest available option — if the request is about "
        "a different subject entirely that isn't one of the catalog topics, leave \"topics\" empty. "
        "Separately, judge whether this is a genuine learning/study goal AT ALL (even if it's not in the "
        "catalog) as opposed to small talk, nonsense, or something unrelated to studying — set "
        "\"is_learning_request\" accordingly; this app can still build a plan for out-of-catalog subjects "
        "by generating fresh material for them. "
        "Also detect if the student stated a day-of-week preference (e.g. \"weekends only\", "
        "\"just Mon/Wed/Fri\") and return it as \"preferred_days\": a list of full weekday names "
        "(\"Monday\".. \"Sunday\"), or an empty list if they didn't say. "
        "Return ONLY a JSON object: {\"topics\": [topic_id,...], \"target_mastery\": int (0-100, "
        "default 70 if unspecified), \"deadline_days\": int, \"is_learning_request\": bool, "
        "\"preferred_days\": [string,...], "
        "\"unmatched_reason\": string or null (only set this when \"topics\" is empty because nothing "
        "in the catalog matched)}. No prose."
    )
    raw = client.chat(system, user_text, temperature=0.2, json_mode=True)
    fallback = {"topics": [], "target_mastery": 70, "deadline_days": 7, "is_learning_request": True,
                "preferred_days": [],
                "unmatched_reason": "Couldn't understand that request — please try rephrasing it."}
    return _safe_json_parse(raw, fallback)


# ---------------------------------------------------------------------------
# 1b. synthesize_dynamic_topic — for requests outside the fixed catalog.
# Instead of just refusing, JASSI asks the LLM to invent a lightweight topic
# definition (estimated hours, prerequisites, subtopic breakdown) so the same
# feasibility/scheduling/resource machinery can run on it.
# ---------------------------------------------------------------------------
def synthesize_dynamic_topic(client: GroqClient, user_text: str,
                              known_topics: List[Dict[str, Any]]) -> Dict[str, Any]:
    known_ids = [t["id"] for t in known_topics]
    system = (
        "A student asked to learn something that isn't in this app's fixed topic catalog. "
        "Invent a reasonable, lightweight topic definition (as JSON) so a study plan can still be "
        "built for it. "
        "If (and only if) the request genuinely isn't a learnable subject at all (e.g. small talk, "
        "nonsense), return the JSON {\"id\": null}. Otherwise return ONLY this JSON: "
        "{\"id\": short_snake_case_id, \"name\": \"Title Case Name\", "
        "\"estimated_hours\": number (realistic total hours an average learner needs to go from 0 to "
        "100 mastery), \"prerequisites\": [ids from the existing catalog that are genuinely foundational, "
        "usually empty], \"subtopics\": [4-6 short phrases breaking the topic into a logical learning "
        f"order, beginner to advanced]}}. Existing catalog ids (for prerequisite reference only): {known_ids}."
    )
    raw = client.chat(system, user_text, temperature=0.4, json_mode=True)
    fallback = {"id": None}
    return _safe_json_parse(raw, fallback)


# ---------------------------------------------------------------------------
# 1c. suggest_resource_paths — instead of silently auto-picking resources,
# offer the student 3 alternative playlists/trajectories with pros and cons
# (mirroring the trade-off "reality check" UX), grounded in the real curated
# resources when available, or LLM-invented ones for a dynamic topic.
# ---------------------------------------------------------------------------
def suggest_resource_paths(client: GroqClient, topic_name: str,
                            subtopics: Optional[List[str]] = None,
                            candidate_resources: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    has_catalog = bool(candidate_resources)
    context = {
        "topic": topic_name,
        "subtopics": subtopics or [],
        "candidate_resources": candidate_resources or [],
    }
    system = (
        "You are a study-resource curator. Propose exactly 3 alternative learning paths/playlists for "
        "this topic so the student can pick the one that fits them, each with honest pros and cons "
        "(e.g. faster but shallower, video-heavy vs reading-heavy, more practice vs more theory). "
        + (
            "Build each path ONLY from the given candidate_resources (reference them by their exact "
            "\"id\" and \"title\"), just choosing a different subset/order/emphasis per path — do not "
            "invent resources that aren't in the list."
            if has_catalog else
            "There is no existing resource database for this topic, so invent plausible session titles "
            "and realistic durations yourself (label them as suggestions, not links)."
        )
        + " Return ONLY JSON: {\"paths\": [{\"title\": short path name, \"pros\": [string,...], "
        "\"cons\": [string,...], \"sessions\": [{\"title\": string, \"duration_mins\": int, "
        "\"resource_id\": string or null}]}]} — exactly 3 entries in \"paths\"."
    )
    raw = client.chat(system, json.dumps(context), temperature=0.5, json_mode=True)
    fallback = {"paths": []}
    return _safe_json_parse(raw, fallback)


# ---------------------------------------------------------------------------
# 2. generate_feasibility_explanation
# ---------------------------------------------------------------------------
def generate_feasibility_explanation(client: GroqClient, feasibility_result: Dict[str, Any],
                                      topics: List[str]) -> str:
    risks = feasibility_result.get("risks") or []
    system = (
        "You are a warm, direct study planning assistant. Explain a feasibility calculation "
        "to a student in 2-3 short sentences. Be concrete: reference the actual hours, and if "
        "any concrete risks are given (e.g. a specific day skipping a meal, or back-to-back "
        "commitments with no break), name that specific consequence instead of a generic "
        "'you don't have enough time' line — the student should be able to picture exactly "
        "what would go wrong, not just see a number. Don't use markdown headers."
    )
    user = (
        f"Topics: {topics}. Required hours: {feasibility_result['required_hours']}. "
        f"Available hours: {feasibility_result['available_hours']}. "
        f"Feasible: {feasibility_result['feasible']}. "
        f"Shortfall: {feasibility_result['shortfall_hours']}. "
        f"Concrete risks already detected: {risks if risks else 'none'}."
    )
    return client.chat(system, user, temperature=0.6)


# ---------------------------------------------------------------------------
# 3. generate_tradeoffs (Reality Check)
# ---------------------------------------------------------------------------
def generate_tradeoffs(client: GroqClient, feasibility_result: Dict[str, Any],
                        topics: List[str], soft_events: List[Dict[str, Any]],
                        min_extension_days: int) -> Dict[str, Any]:
    system = (
        "You are an empathetic AI study planner. The student's plan doesn't fit their "
        "available time. Generate exactly 3 trade-off options as JSON. Return ONLY: "
        "{\"options\": [{\"id\": \"A\", \"title\": str, \"description\": str}, "
        "{\"id\": \"B\", \"title\": str, \"description\": str}, "
        "{\"id\": \"C\", \"title\": str, \"description\": str}]}. "
        "Option A = keep deadline, move soft commitments (name the actual soft events given). "
        "Option B = keep commitments, extend deadline (use the given minimum extension days). "
        "Option C = keep both, reduce topic scope (suggest deferring the least prerequisite-critical topic). "
        "Keep each description to one warm, specific sentence. No markdown."
    )
    user = json.dumps({
        "shortfall_hours": feasibility_result["shortfall_hours"],
        "topics": topics,
        "soft_events": [{"title": e["title"], "day": e["day"]} for e in soft_events],
        "min_extension_days": min_extension_days,
        "risks": feasibility_result.get("risks") or [],
    })
    raw = client.chat(system, user, temperature=0.7, json_mode=True)
    fallback = {"options": [
        {"id": "A", "title": "Keep deadline", "description": "Move soft commitments to free up time."},
        {"id": "B", "title": "Extend deadline", "description": f"Extend by {min_extension_days} days."},
        {"id": "C", "title": "Reduce scope", "description": "Defer one topic to focus on the rest."},
    ]}
    return _safe_json_parse(raw, fallback)


# ---------------------------------------------------------------------------
# 4. daily_checkin_dialogue
# ---------------------------------------------------------------------------
def daily_checkin_dialogue(client: GroqClient, sessions_today: List[Dict[str, Any]],
                            recent_notes: str = "") -> str:
    system = (
        "You are a friendly, casual study buddy AI checking in with a student in the evening. "
        "Write a short, warm 1-2 sentence check-in message referencing their actual scheduled "
        "session(s) today. If recent_notes mentions a struggle, acknowledge it naturally."
    )
    user = json.dumps({"sessions_today": sessions_today, "recent_notes": recent_notes})
    return client.chat(system, user, temperature=0.8)


# ---------------------------------------------------------------------------
# parse natural-language daily update into structured form
# ---------------------------------------------------------------------------
def parse_update(client: GroqClient, user_text: str, session_topics: List[str]) -> Dict[str, Any]:
    system = (
        "Convert a student's casual message about their study session into structured JSON. "
        f"Relevant topics today: {session_topics}. Return ONLY: "
        "{\"status\": \"completed\"|\"missed\"|\"partial\", \"percent_done\": int (0-100), "
        "\"struggle_note\": str or null}."
    )
    raw = client.chat(system, user_text, temperature=0.2, json_mode=True)
    fallback = {"status": "partial", "percent_done": 50, "struggle_note": None}
    return _safe_json_parse(raw, fallback)


# ---------------------------------------------------------------------------
# 5. analyze_progress_patterns
# ---------------------------------------------------------------------------
def analyze_progress_patterns(client: GroqClient, mastery_map: Dict[str, int],
                               quiz_history: List[Dict[str, Any]],
                               struggle_note: Optional[str] = None) -> str:
    system = (
        "You are a study-progress analyst AI. Look at mastery levels and recent quiz scores "
        "across topics and give one short, specific, encouraging-but-honest insight (2 sentences max) "
        "about a pattern you notice (e.g. relative pace between topics, an emerging strength or gap). "
        "If a struggle_note is given, directly address that specific difficulty rather than "
        "giving a generic comment."
    )
    user = json.dumps({
        "mastery_map": mastery_map,
        "quiz_history": quiz_history,
        "struggle_note": struggle_note,
    })
    return client.chat(system, user, temperature=0.6)


# ---------------------------------------------------------------------------
# 6. generate_schedule_explanation
# ---------------------------------------------------------------------------
def generate_schedule_explanation(client: GroqClient, scheduled_sessions: List[Dict[str, Any]],
                                   topics_data: Dict[str, Any]) -> str:
    system = (
        "You are a study planning AI explaining a generated schedule to a student. "
        "In 2-3 sentences, explain the overall shape of the plan and WHY topics were ordered "
        "the way they were (reference prerequisites where relevant). No markdown, no bullet lists."
    )
    user = json.dumps({"sessions": scheduled_sessions, "topics": topics_data["topics"]})
    return client.chat(system, user, temperature=0.6)


def generate_session_reasoning(client: GroqClient, session: Dict[str, Any],
                                reasons: List[str]) -> str:
    """Per-session hover explanation, combining deterministic reasons with natural phrasing."""
    system = (
        "Rewrite this list of scheduling reasons as one natural, friendly sentence explaining "
        "why this study session was placed here. Keep all the factual content."
    )
    user = json.dumps({"session": session, "reasons": reasons})
    return client.chat(system, user, temperature=0.5)


def generate_reschedule_explanation(client: GroqClient, old_session: Dict[str, Any],
                                     new_session: Dict[str, Any], reason: str) -> str:
    system = (
        "You are a study planning AI. In one short, friendly sentence, tell the student their "
        "missed session was moved, referencing the new day/time and the given reason."
    )
    user = json.dumps({"old": old_session, "new": new_session, "reason": reason})
    return client.chat(system, user, temperature=0.6)


def generate_verification_failure_message(client: GroqClient, issue: str,
                                           alternative_slot: Optional[Dict[str, Any]] = None) -> str:
    system = (
        "You are a study planning AI. Explain a scheduling conflict to the student in one "
        "friendly sentence, and if an alternative slot is given, offer it as a question."
    )
    user = json.dumps({"issue": issue, "alternative_slot": alternative_slot})
    return client.chat(system, user, temperature=0.5)
