"""
News Search Module
Fetches recent news headlines related to a market question using DuckDuckGo.
No API key required.
"""

import logging
import re
import time
from typing import Optional

logger = logging.getLogger(__name__)


def _extract_search_keywords(question: str) -> str:
    """
    Extract meaningful search keywords from a prediction market question.
    Strips common prediction-market boilerplate to get the core topic.
    """
    # Remove common prefixes
    q = question.strip()
    for prefix in ["Will ", "Is ", "Does ", "Has ", "Are ", "Can ", "Should "]:
        if q.startswith(prefix):
            q = q[len(prefix):]
            break

    # Remove trailing question marks and dates like "before 2027?"
    q = re.sub(r'\?\s*$', '', q)
    q = re.sub(r'\b(before|by|in|on|end of)\s+\d{4}\b', '', q)
    q = re.sub(r'\bon \d{4}-\d{2}-\d{2}\b', '', q)

    # Trim to first ~8 words for a focused search
    words = q.split()
    return ' '.join(words[:8]).strip()


def fetch_news_context(question: str, max_results: int = 3) -> Optional[str]:
    """
    Search DuckDuckGo News for recent articles related to the market question.

    Args:
        question: The prediction market question text.
        max_results: Maximum number of headlines to return.

    Returns:
        Formatted string of recent news headlines with sources, or None if search fails.
    """
    try:
        from ddgs import DDGS
    except ImportError:
        logger.warning("ddgs not installed — pip install ddgs")
        return None

    keywords = _extract_search_keywords(question)
    if len(keywords) < 5:
        logger.debug(f"Keywords too short for news search: '{keywords}'")
        return None

    try:
        with DDGS() as ddgs:
            results = list(ddgs.news(keywords, max_results=max_results, safesearch="off"))

        if not results:
            logger.debug(f"No news results for: '{keywords}'")
            return None

        lines = [f"Recent news for: '{keywords}'"]
        for r in results[:max_results]:
            title = r.get("title", "").strip()
            source = r.get("source", "").strip()
            date = r.get("date", "").strip()[:10]  # Just the date part
            body = r.get("body", "").strip()[:150]  # Truncate long bodies
            if title:
                lines.append(f"  • [{date}] {title} ({source})")
                if body:
                    lines.append(f"    {body}")

        return '\n'.join(lines)

    except Exception as e:
        logger.warning(f"News search failed: {e}")
        return None


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.DEBUG)
    result = fetch_news_context("Will the U.S. invade Iran before 2027?")
    if result:
        print(result)
    else:
        print("No news found")
