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
    q = question.strip()

    # Remove common question prefixes
    for prefix in ["Will ", "Is ", "Does ", "Has ", "Are ", "Can ", "Should ", "Did "]:
        if q.startswith(prefix):
            q = q[len(prefix):]
            break

    # Remove trailing question marks
    q = re.sub(r'\?\s*$', '', q)

    # Remove date qualifiers like "before 2027", "by March 31", "in Q1 2026"
    q = re.sub(r'\b(before|by|in|on|end of|after)\s+(the\s+)?\d{4}\b', '', q, flags=re.IGNORECASE)
    q = re.sub(r'\bon \d{4}-\d{2}-\d{2}\b', '', q)
    q = re.sub(r'\bby\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d+\b', '', q, flags=re.IGNORECASE)
    q = re.sub(r'\bin\s+Q[1-4]\s+\d{4}\b', '', q, flags=re.IGNORECASE)

    # Remove percentage/number thresholds (keep the subject)
    q = re.sub(r'between \$[\d,]+ and \$[\d,]+', '', q)
    q = re.sub(r'(above|below|more than|less than|at least|greater than)\s+[\$\d,\.%]+', '', q, flags=re.IGNORECASE)

    # Clean up extra whitespace
    q = re.sub(r'\s+', ' ', q).strip()

    # If we stripped too much, fall back to first 8 words of original
    if len(q) < 8:
        words = question.split()
        q = ' '.join(words[:8]).strip()
        q = re.sub(r'\?\s*$', '', q)

    # Trim to first 10 words for a focused search
    words = q.split()
    return ' '.join(words[:10]).strip()


def fetch_news_context(question: str, max_results: int = 3) -> Optional[str]:
    """
    Search DuckDuckGo News for recent articles related to the market question.

    Args:
        question:    The prediction market question text.
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

    if len(keywords) < 4:
        logger.debug(f"Keywords too short for news search: '{keywords}'")
        return None

    logger.debug(f"News search: '{keywords}'")

    # Try up to 2 times with increasing timeout
    for attempt in range(2):
        try:
            with DDGS() as ddgs:
                results = list(
                    ddgs.news(
                        keywords,
                        max_results=max_results,
                        safesearch="off",
                    )
                )

            if not results:
                # Retry with a shorter, broader query
                if attempt == 0:
                    short_keywords = ' '.join(keywords.split()[:4])
                    logger.debug(f"No results, retrying with: '{short_keywords}'")
                    with DDGS() as ddgs:
                        results = list(
                            ddgs.news(
                                short_keywords,
                                max_results=max_results,
                                safesearch="off",
                            )
                        )

            if not results:
                logger.debug(f"No news results for: '{keywords}'")
                return None

            lines = [f"Recent news for: '{keywords}'"]
            for r in results[:max_results]:
                title  = r.get("title",  "").strip()
                source = r.get("source", "").strip()
                date   = r.get("date",   "").strip()[:10]
                body   = r.get("body",   "").strip()[:200]

                if title:
                    lines.append(f"  • [{date}] {title} ({source})")
                    if body:
                        lines.append(f"    {body}")

            return '\n'.join(lines)

        except Exception as e:
            err = str(e)
            if "Timeout" in err or "timeout" in err:
                if attempt == 0:
                    logger.debug(f"News timeout, retrying...")
                    time.sleep(1)
                    continue
                else:
                    logger.warning(f"News search timed out for: '{keywords}'")
            else:
                logger.warning(f"News search failed: {e}")
            return None

    return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    tests = [
        "Will Sweden qualify for the 2026 FIFA World Cup?",
        "Will OpenSea launch a token by December 31, 2026?",
        "Will Tesla deliver between 350000 and 375000 vehicles in Q1 2026?",
        "Will the US confirm that aliens exist before 2027?",
        "Will Alphabet be the largest company in the world by market cap on Dec 31?",
    ]

    for q in tests:
        print(f"\nQ: {q}")
        keywords = _extract_search_keywords(q)
        print(f"Keywords: '{keywords}'")
        result = fetch_news_context(q, max_results=2)
        if result:
            print(result)
        else:
            print("  → No results")
        print("─" * 60)