"""Resolve a learner's course name to a useful resource URL.

A direct URL is kept as-is. For a course name, try a lightweight web search
(when available); if search is unavailable, provide a targeted YouTube search
URL rather than leaving the session without a link. Search-result URLs are
suggestions and should be confirmed by the learner before relying on them.
"""
from __future__ import annotations

from urllib.parse import quote_plus, urlparse
from typing import Optional


def _is_http_url(value: Optional[str]) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    parsed = urlparse(value.strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def resolve_resource_url(title: str, preferred_platform: str = "YouTube") -> tuple[str, str]:
    """Return (url, source_type). source_type is direct, search_result, or search_link."""
    title = (title or "").strip()
    if _is_http_url(title):
        return title, "direct"
    if not title:
        return "", "missing"

    platform = (preferred_platform or "").strip().lower()
    query = f'"{title}" course playlist'
    if platform in {"youtube", "yt"}:
        query += " site:youtube.com"
    elif platform == "coursera":
        query += " site:coursera.org"
    elif platform == "udemy":
        query += " site:udemy.com"

    # DDGS is optional at runtime; failure should never prevent planning.
    try:
        from ddgs import DDGS  # type: ignore
        results = list(DDGS().text(query, max_results=8))
        candidates = []
        title_terms = {w.lower() for w in title.split() if len(w) > 2}
        for result in results:
            url = result.get("href") or result.get("url") or ""
            if not _is_http_url(url):
                continue
            result_text = f"{result.get('title', '')} {result.get('body', '')}".lower()
            overlap = sum(1 for word in title_terms if word in result_text)
            domain = urlparse(url).netloc.lower()
            platform_match = (
                (platform in {"youtube", "yt"} and "youtube.com" in domain)
                or (platform == "coursera" and "coursera.org" in domain)
                or (platform == "udemy" and "udemy.com" in domain)
            )
            candidates.append((overlap + (3 if platform_match else 0), url))
        if candidates:
            candidates.sort(key=lambda item: item[0], reverse=True)
            return candidates[0][1], "search_result"
    except Exception:
        # Search is best-effort: network outages/rate limits must not break JASSI.
        pass

    # Useful, deterministic fallback. Label this as a search link in the UI;
    # do not pretend it is a verified canonical course URL.
    if platform == "coursera":
        return f"https://www.coursera.org/search?query={quote_plus(title)}", "search_link"
    if platform == "udemy":
        return f"https://www.udemy.com/courses/search/?q={quote_plus(title)}", "search_link"
    return f"https://www.youtube.com/results?search_query={quote_plus(title)}", "search_link"


def recommend_youtube_resources(topic: str, task: str, client=None, limit: int = 3) -> list[dict]:
    """Return only search-result videos with clear topic/task relevance.

    The LLM may suggest query wording, but never supplies URLs. Search results
    are filtered locally; if relevance cannot be established, return no video
    recommendations so the UI can show a focused YouTube search link instead.
    """
    import json
    import re

    def terms(value: str) -> set[str]:
        stop = {"the", "and", "for", "with", "from", "your", "into", "how", "what", "learn", "learning", "tutorial", "course", "guide", "part", "using", "based"}
        return {w for w in re.findall(r"[a-z0-9]+", (value or "").lower()) if len(w) > 2 and w not in stop}

    topic_terms, task_terms = terms(topic), terms(task)
    target_terms = topic_terms | task_terms
    if not target_terms:
        return []

    queries = []
    if client is not None:
        try:
            raw = client.chat(
                "Return JSON only: {\"queries\":[\"...\",\"...\",\"...\"]}. "
                "Create three focused YouTube search queries for the exact topic and study task. "
                "Do not provide URLs or unrelated broad recommendations.",
                f"Topic: {topic}\nExact study task: {task}\nKeep both concepts in each query.",
                temperature=0.2, json_mode=True,
            )
            parsed = json.loads(raw)
            queries = [str(q).strip() for q in parsed.get("queries", []) if str(q).strip()][:3]
        except Exception:
            queries = []
    if not queries:
        queries = [f"{topic} {task} tutorial", f"{task} in {topic} explained", f"{topic} {task} examples"]

    found, seen = [], set()
    try:
        from ddgs import DDGS  # type: ignore
        with DDGS() as ddgs:
            for query in queries:
                for item in ddgs.text(f"{query} site:youtube.com/watch OR site:youtu.be", max_results=8):
                    url = (item.get("href") or item.get("url") or "").strip()
                    parsed_url = urlparse(url)
                    host = parsed_url.netloc.lower()
                    if not (host.endswith("youtube.com") or host.endswith("youtu.be")):
                        continue
                    if (host.endswith("youtube.com") and "/watch" not in parsed_url.path) or url in seen:
                        continue
                    title = str(item.get("title") or "")
                    body = str(item.get("body") or "")
                    result_terms = terms(title + " " + body)
                    topic_hits = len(topic_terms & result_terms)
                    task_hits = len(task_terms & result_terms)
                    # Require evidence for the requested subject AND task. A
                    # generic YouTube hit must never be presented as relevant.
                    if topic_hits == 0 or (task_terms and task_hits == 0):
                        continue
                    if len(target_terms) >= 4 and (topic_hits + task_hits) < 2:
                        continue
                    seen.add(url)
                    found.append({"title": title or f"{task} — {topic}", "url": url, "platform": "YouTube"})
                    if len(found) >= limit:
                        return found
    except Exception:
        pass
    return found

