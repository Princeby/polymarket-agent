"""
Polymarket Gamma API Client
Fetches and filters active prediction markets from Polymarket's public API.
No authentication required for read operations.
"""

import json
import logging
import requests
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

GAMMA_API_BASE = "https://gamma-api.polymarket.com"


@dataclass
class Market:
    """Parsed prediction market data."""
    id: str
    question: str
    description: str
    yes_price: float
    no_price: float
    volume: float
    liquidity: float
    end_date: str
    active: bool
    slug: str
    clob_token_ids: list = field(default_factory=list)  # [yes_token_id, no_token_id]

    @property
    def implied_probability(self) -> float:
        """Market-implied probability of YES outcome."""
        return self.yes_price

    def __str__(self) -> str:
        return (
            f"[{self.id}] {self.question[:80]}\n"
            f"  YES: {self.yes_price:.1%} | NO: {self.no_price:.1%} | "
            f"Vol: ${self.volume:,.0f} | Liq: ${self.liquidity:,.0f} | "
            f"Ends: {self.end_date}"
        )


def _parse_market(raw: dict) -> Optional[Market]:
    """Parse a raw API response into a Market dataclass."""
    try:
        # outcomePrices is a JSON string like '["0.62", "0.38"]'
        prices = json.loads(raw.get("outcomePrices", "[]"))
        if len(prices) < 2:
            return None

        # clobTokenIds is a JSON string like '["<yes_token>", "<no_token>"]'
        raw_token_ids = raw.get("clobTokenIds", "[]")
        try:
            clob_token_ids = json.loads(raw_token_ids) if isinstance(raw_token_ids, str) else raw_token_ids
        except (json.JSONDecodeError, TypeError):
            clob_token_ids = []

        return Market(
            id=raw.get("id", ""),
            question=raw.get("question", ""),
            description=raw.get("description", "")[:500],  # truncate long descriptions
            yes_price=float(prices[0]),
            no_price=float(prices[1]),
            volume=float(raw.get("volumeNum", raw.get("volume", 0))),
            liquidity=float(raw.get("liquidityNum", raw.get("liquidity", 0))),
            end_date=raw.get("endDateIso", raw.get("endDate", "")),
            active=raw.get("active", False),
            slug=raw.get("slug", ""),
            clob_token_ids=clob_token_ids,
        )
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        logger.warning(f"Failed to parse market: {e}")
        return None


def get_active_markets(
    limit: int = 20,
    min_volume: float = 10000,
    min_liquidity: float = 1000,
) -> list[Market]:
    """
    Fetch active, open markets from Polymarket sorted by volume.

    Args:
        limit: Maximum number of markets to fetch from the API.
        min_volume: Minimum total volume (USD) to include a market.
        min_liquidity: Minimum current liquidity (USD) to include a market.

    Returns:
        List of Market objects that pass the filters.
    """
    url = f"{GAMMA_API_BASE}/markets"
    params = {
        "active": "true",
        "closed": "false",
        "limit": limit,
        "order": "volume",
        "ascending": "false",
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        raw_markets = response.json()
    except requests.RequestException as e:
        logger.error(f"Failed to fetch markets: {e}")
        return []
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Failed to parse API response: {e}")
        return []

    markets = []
    for raw in raw_markets:
        market = _parse_market(raw)
        if market is None:
            continue
        if not market.active:
            continue
        if market.volume < min_volume:
            continue
        if market.liquidity < min_liquidity:
            continue
        # Skip markets with extreme prices (already resolved or near-certain)
        if market.yes_price < 0.03 or market.yes_price > 0.97:
            continue
        # Skip sports spreads and O/U — model has no edge (50% win rate in backtest)
        q = market.question.lower()
        if q.startswith("spread:") or "o/u " in q or ": o/u " in q:
            logger.debug(f"Skipping sports spread: {market.question[:50]}")
            continue
        markets.append(market)

    logger.info(f"Fetched {len(markets)} markets (from {len(raw_markets)} raw)")
    return markets


def get_market_by_id(market_id: str) -> Optional[Market]:
    """Fetch a single market by its ID."""
    url = f"{GAMMA_API_BASE}/markets/{market_id}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return _parse_market(response.json())
    except requests.RequestException as e:
        logger.error(f"Failed to fetch market {market_id}: {e}")
        return None


def format_market_for_llm(market: Market, include_news: bool = True) -> str:
    """
    Format market data into a clean text block for the LLM prompt.
    Optionally includes recent news headlines for real-time context.
    """
    base = (
        f"Market Question: {market.question}\n"
        f"Current YES Price: {market.yes_price:.4f} "
        f"(implies {market.yes_price:.1%} probability of YES)\n"
        f"Current NO Price: {market.no_price:.4f}\n"
        f"Total Volume Traded: ${market.volume:,.0f}\n"
        f"Current Liquidity: ${market.liquidity:,.0f}\n"
        f"Resolution Date: {market.end_date}\n"
        f"Description: {market.description}"
    )

    if include_news:
        try:
            from src.news import fetch_news_context
            news = fetch_news_context(market.question, max_results=3)
            if news:
                base += f"\n\n--- Recent News Context ---\n{news}"
        except Exception as e:
            logger.debug(f"News fetch skipped: {e}")

    return base


if __name__ == "__main__":
    # Quick smoke test
    logging.basicConfig(level=logging.INFO)
    markets = get_active_markets(limit=5, min_volume=5000, min_liquidity=500)
    for m in markets:
        print(m)
        print(format_market_for_llm(m))
        print("---")