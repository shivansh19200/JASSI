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
