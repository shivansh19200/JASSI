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
    """Use Groq to suggest focused search queries, then resolve them to real YouTube results.

    URLs are taken from search results, never invented by the LLM. If Groq is unavailable,
    use a direct topic/task search query and return only URLs found by DDGS.
    """
    import json
    queries = []
    if client is not None:
        try:
            raw = client.chat(
                "You are a learning-resource researcher. Return JSON only: "
                '{"queries":["...","...","..."]}. Create exactly three specific YouTube search queries '
                "for high-quality tutorials that teach the requested task. Do not provide URLs, channel names, "
                "or claim that a particular video exists.",
                f"Topic: {topic}\nStudy task: {task}\nReturn three distinct focused YouTube search queries.",
                temperature=0.2, json_mode=True,
            )
            parsed = json.loads(raw)
            queries = [str(q).strip() for q in parsed.get("queries", []) if str(q).strip()][:3]
        except Exception:
            queries = []
    if not queries:
        queries = [f"{task} {topic} tutorial", f"{topic} {task} explained", f"{topic} {task} practice"]

    found, seen = [], set()
    try:
        from ddgs import DDGS  # type: ignore
        with DDGS() as ddgs:
            for query in queries:
                for item in ddgs.text(f"{query} site:youtube.com/watch OR site:youtu.be", max_results=5):
                    url = (item.get("href") or item.get("url") or "").strip()
                    parsed_url = urlparse(url)
                    host = parsed_url.netloc.lower()
                    if not (host.endswith("youtube.com") or host.endswith("youtu.be")):
                        continue
                    if "/watch" not in parsed_url.path and host.endswith("youtube.com"):
                        continue
                    if url in seen:
                        continue
                    seen.add(url)
                    found.append({"title": item.get("title") or f"{task} tutorial", "url": url, "platform": "YouTube"})
                    if len(found) >= limit:
                        return found
    except Exception:
        pass
    return found
