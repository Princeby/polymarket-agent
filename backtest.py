#!/usr/bin/env python3
"""
Backtester — Test Model Against Resolved Markets
─────────────────────────────────────────────────
Fetches historically resolved markets from Polymarket, runs them through
the LLM WITHOUT news context (to avoid data leakage), and measures:
  - Brier score (calibration)
  - Simulated P&L (profitability)
  - Win/loss rate
  - Calibration by probability bucket

Usage:
    python backtest.py                    # Default: 20 resolved markets
    python backtest.py --markets 50       # Test on 50 markets
    python backtest.py --min-volume 50000 # Only high-volume markets
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import requests
from dotenv import load_dotenv

from src.agent import get_backend, analyze_market, calculate_edge
from src.market import Market
from src.trader import kelly_stake

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
# Silence noisy libs
for lib in ["urllib3", "requests", "primp", "h2", "rustls", "hyper_util", "cookie_store", "ddgs"]:
    logging.getLogger(lib).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


GAMMA_API_BASE = "https://gamma-api.polymarket.com"


@dataclass
class BacktestResult:
    market_id: str
    question: str
    actual_outcome: str       # "YES" or "NO"
    model_probability: float  # model's P(YES)
    market_price_at_close: float  # final snapped price (1.0 or 0.0)
    edge: float
    direction: str            # "YES" or "NO" — what model would bet
    stake: float
    won: bool
    pnl: float


def fetch_resolved_markets(limit: int = 50, min_volume: float = 50000) -> list[dict]:
    """Fetch closed markets from Polymarket with clear outcomes."""
    logger.info(f"Fetching resolved markets (limit={limit}, min_vol=${min_volume:,.0f})...")
    url = f"{GAMMA_API_BASE}/markets"
    params = {
        "closed": "true",
        "limit": limit * 3,  # fetch extra since we filter
        "order": "volume",
        "ascending": "false",
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        raw = resp.json()
    except requests.RequestException as e:
        logger.error(f"API error: {e}")
        return []

    results = []
    for m in raw:
        # Parse outcome from snapped prices
        prices = json.loads(m.get("outcomePrices", "[]"))
        if len(prices) < 2:
            continue

        yes_price = float(prices[0])
        vol = float(m.get("volume", 0))

        # Must have clear outcome (price snapped to ~0 or ~1)
        if 0.05 < yes_price < 0.95:
            continue

        # Volume filter
        if vol < min_volume:
            continue

        # Skip trivially short questions
        q = m.get("question", "")
        if len(q) < 15:
            continue

        outcome = "YES" if yes_price >= 0.95 else "NO"

        results.append({
            "id": str(m.get("id", "")),
            "question": q,
            "description": m.get("description", "")[:500],
            "outcome": outcome,
            "volume": vol,
            "end_date": m.get("endDate", "")[:10],
        })

        if len(results) >= limit:
            break

    logger.info(f"Found {len(results)} resolved markets for backtesting")
    return results


def backtest_market(market_data: dict, backend, bankroll: float, edge_threshold: float) -> BacktestResult | None:
    """
    Run a single resolved market through the model WITHOUT news.
    Returns BacktestResult or None if analysis fails.
    """
    # Build a Market object (use 0.50/0.50 as "unknown" prices to not leak info)
    # The model should estimate probability purely from the question + description
    market = Market(
        id=market_data["id"],
        question=market_data["question"],
        description=market_data["description"],
        yes_price=0.50,  # Hide market price — force model to reason independently
        no_price=0.50,
        volume=market_data["volume"],
        liquidity=0,
        end_date=market_data["end_date"],
        active=False,
        slug="",
    )

    # Analyze WITHOUT news (include_news=False to prevent data leakage)
    analysis = analyze_market(market, backend, include_news=False)
    if analysis is None:
        return None

    # Calculate edge vs a 50/50 baseline (since we hid the market price)
    model_prob = analysis.estimated_probability
    actual_outcome = market_data["outcome"]
    actual_value = 1.0 if actual_outcome == "YES" else 0.0

    # Determine what the model would bet
    if model_prob > 0.50 + edge_threshold:
        direction = "YES"
        edge = model_prob - 0.50
    elif model_prob < 0.50 - edge_threshold:
        direction = "NO"
        edge = 0.50 - model_prob
    else:
        direction = "SKIP"
        edge = abs(model_prob - 0.50)

    # Kelly sizing (vs 50/50 baseline)
    stake = 0.0
    if direction != "SKIP":
        stake = kelly_stake(
            bankroll=bankroll,
            edge=edge,
            market_price=0.50,
            direction=direction,
        )

    # Did we win?
    won = False
    pnl = 0.0
    if stake > 0:
        if direction == "YES":
            won = actual_outcome == "YES"
        else:
            won = actual_outcome == "NO"

        if won:
            pnl = stake * (1 / 0.50 - 1)  # payout at 50/50 odds = 2x - stake = +stake
        else:
            pnl = -stake

    return BacktestResult(
        market_id=market_data["id"],
        question=market_data["question"][:70],
        actual_outcome=actual_outcome,
        model_probability=model_prob,
        market_price_at_close=1.0 if actual_outcome == "YES" else 0.0,
        edge=edge,
        direction=direction,
        stake=stake,
        won=won,
        pnl=pnl,
    )


def print_results(results: list[BacktestResult]):
    """Print detailed backtest results."""
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║               🧪 Backtest Results                              ║")
    print("║               News: DISABLED (no data leakage)                 ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    # ── Per-market results ───────────────────────────────────────────────
    bets = [r for r in results if r.direction != "SKIP"]
    skips = [r for r in results if r.direction == "SKIP"]

    print(f"\n  {'#':<4} {'Result':<8} {'Pred':>5} {'Actual':<6} {'Bet':<5} "
          f"{'Stake':>7} {'P&L':>8} {'Question'}")
    print(f"  {'─' * 85}")

    for i, r in enumerate(results, 1):
        if r.direction == "SKIP":
            icon = "⏭️"
            result_str = "skip"
            pnl_str = "   —"
        elif r.won:
            icon = "✅"
            result_str = "WIN"
            pnl_str = f"${r.pnl:+.2f}"
        else:
            icon = "❌"
            result_str = "LOSS"
            pnl_str = f"${r.pnl:+.2f}"

        print(f"  {i:<4} {icon} {result_str:<5} {r.model_probability:>4.0%}  "
              f"{r.actual_outcome:<6} {r.direction:<5} "
              f"${r.stake:>6.2f} {pnl_str:>8}  {r.question[:40]}")

    # ── Summary stats ────────────────────────────────────────────────────
    print(f"\n{'═' * 65}")

    # Brier score on ALL predictions
    brier_sum = sum((r.model_probability - (1.0 if r.actual_outcome == "YES" else 0.0)) ** 2
                    for r in results)
    brier = brier_sum / len(results)

    # Win/loss on actual bets
    wins = [r for r in bets if r.won]
    losses = [r for r in bets if not r.won]
    total_pnl = sum(r.pnl for r in bets)
    total_staked = sum(r.stake for r in bets)
    roi = (total_pnl / total_staked * 100) if total_staked > 0 else 0

    print(f"  Markets tested:    {len(results)}")
    print(f"  Bets placed:       {len(bets)} ({len(skips)} skipped)")
    print(f"  Win / Loss:        {len(wins)}W / {len(losses)}L "
          f"({len(wins)/len(bets)*100:.0f}% win rate)" if bets else "")
    print(f"  Total staked:      ${total_staked:.2f}")
    print(f"  Net P&L:           ${total_pnl:+.2f}")
    print(f"  ROI:               {roi:+.1f}%")

    print(f"\n  Brier Score:       {brier:.4f}")
    if brier < 0.15:
        print(f"  Assessment:        🏆 Excellent — model has real predictive power")
    elif brier < 0.25:
        print(f"  Assessment:        👍 Good — model adds signal over 50/50 baseline")
    elif brier < 0.30:
        print(f"  Assessment:        ⚠️  Marginal — close to no-skill baseline")
    else:
        print(f"  Assessment:        🚨 Poor — model is not well-calibrated")

    # ── Calibration buckets ──────────────────────────────────────────────
    print(f"\n  Calibration by prediction bucket:")
    print(f"  {'Bucket':<12} {'Count':<7} {'Model Avg':<11} {'Actual Rate':<12} {'Gap'}")
    print(f"  {'─'*50}")

    from collections import defaultdict
    buckets = defaultdict(lambda: {"n": 0, "sum_p": 0.0, "sum_y": 0.0})
    for r in results:
        b = int(r.model_probability * 10) * 10
        b = min(b, 90)
        buckets[b]["n"] += 1
        buckets[b]["sum_p"] += r.model_probability
        buckets[b]["sum_y"] += 1.0 if r.actual_outcome == "YES" else 0.0

    for b in sorted(buckets):
        d = buckets[b]
        avg_p = d["sum_p"] / d["n"]
        avg_y = d["sum_y"] / d["n"]
        gap = abs(avg_p - avg_y)
        gap_bar = "●" * min(int(gap * 20), 10)
        print(f"  {b}-{b+10}%      {d['n']:<7} {avg_p:<11.1%} {avg_y:<12.1%} {gap_bar}")

    # ── Verdict ──────────────────────────────────────────────────────────
    print(f"\n{'─' * 65}")
    if brier < 0.20 and roi > 0:
        print("  ✅ GREEN LIGHT — Model is calibrated AND profitable in backtest.")
        print("     Ready for cautious dry-run on live markets.")
    elif brier < 0.25:
        print("  🟡 YELLOW — Model shows some skill but needs more data.")
        print("     Continue dry-run and tune prompts.")
    else:
        print("  🔴 RED — Model needs significant improvement before live trading.")
        print("     Try a stronger model or better prompting strategy.")
    print()


def main():
    parser = argparse.ArgumentParser(description="Backtest model against resolved Polymarket data")
    parser.add_argument("--markets", type=int, default=20, help="Number of resolved markets to test")
    parser.add_argument("--min-volume", type=float, default=50000, help="Minimum volume filter")
    parser.add_argument("--edge-threshold", type=float, default=0.08,
                        help="Min edge to place bet (default 8%%)")
    parser.add_argument("--bankroll", type=float, default=100, help="Simulated bankroll")
    args = parser.parse_args()

    # Init LLM backend
    try:
        backend = get_backend()
        logger.info(f"Using backend: {backend.name}")
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)

    # Fetch resolved markets
    resolved = fetch_resolved_markets(limit=args.markets, min_volume=args.min_volume)
    if not resolved:
        print("No resolved markets found. Try lowering --min-volume.")
        return

    # Run backtest
    print(f"\n🧪 Backtesting {len(resolved)} markets (news DISABLED)...")
    print(f"   Backend: {backend.name}")
    print(f"   Edge threshold: {args.edge_threshold:.0%}")
    print(f"   Bankroll: ${args.bankroll:.0f}")

    results = []
    for i, m in enumerate(resolved, 1):
        logger.info(f"[{i}/{len(resolved)}] {m['question'][:60]}...")
        result = backtest_market(m, backend, args.bankroll, args.edge_threshold)
        if result:
            results.append(result)
            icon = "✅" if result.won else ("❌" if result.direction != "SKIP" else "⏭️")
            logger.info(f"  {icon} Pred: {result.model_probability:.0%} | "
                        f"Actual: {result.actual_outcome} | Bet: {result.direction}")
        else:
            logger.warning(f"  ⚠ Analysis failed — skipping")

    if not results:
        print("All analyses failed. Check your API key.")
        return

    # Save backtest results
    output_path = Path(__file__).parent / "data" / "backtest_results.json"
    output_path.write_text(json.dumps([asdict(r) for r in results], indent=2))
    logger.info(f"Results saved to {output_path}")

    # Print report
    print_results(results)


if __name__ == "__main__":
    main()
