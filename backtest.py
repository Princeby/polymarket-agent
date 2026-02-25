#!/usr/bin/env python3
"""
Backtester — Test Model Against Resolved Markets
─────────────────────────────────────────────────
Fetches historically resolved markets from Polymarket, runs them through
the LLM and measures:
  - Brier score (calibration)
  - Simulated P&L (profitability)
  - Win/loss rate
  - Calibration by probability bucket

News context is disabled by default (avoids data leakage on old markets).
For recently resolved markets (--days 7), enable news with --news to get
a realistic picture of live model performance.

Usage:
    python backtest.py                          # Default: 20 markets, no news
    python backtest.py --markets 50             # Test on 50 markets
    python backtest.py --news --days 7          # Recent markets WITH news (live sim)
    python backtest.py --news --days 3 --markets 30
    python backtest.py --min-volume 50000       # Only high-volume markets
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
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
    actual_outcome: str          # "YES" or "NO"
    model_probability: float     # model's P(YES)
    market_price_at_close: float # final snapped price (1.0 or 0.0)
    edge: float
    direction: str               # "YES" or "NO" — what model would bet
    stake: float
    won: bool
    pnl: float
    end_date: str
    news_used: bool


def fetch_resolved_markets(
    limit: int = 50,
    min_volume: float = 50000,
    max_days_old: int = None,  # None = no recency filter
) -> list[dict]:
    """
    Fetch closed markets from Polymarket with clear outcomes.

    Args:
        limit:        Max markets to return.
        min_volume:   Min total volume in USD.
        max_days_old: If set, only return markets resolved within this many days.
                      Use with --news to avoid data leakage on old markets.
    """
    recency_label = f"last {max_days_old}d" if max_days_old else "all time"
    logger.info(
        f"Fetching resolved markets "
        f"(limit={limit}, min_vol=${min_volume:,.0f}, recency={recency_label})..."
    )

    cutoff_dt = None
    if max_days_old is not None:
        cutoff_dt = datetime.now(timezone.utc) - timedelta(days=max_days_old)

    url = f"{GAMMA_API_BASE}/markets"
    # Fetch more than needed since we filter
    fetch_limit = max(limit * 5, 200)
    params = {
        "closed": "true",
        "limit": fetch_limit,
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
    skipped_recency = 0

    for m in raw:
        prices = json.loads(m.get("outcomePrices", "[]"))
        if len(prices) < 2:
            continue

        yes_price = float(prices[0])
        vol = float(m.get("volume", 0))

        # Must have clear outcome (price snapped to ~0 or ~1)
        if 0.05 < yes_price < 0.95:
            continue

        if vol < min_volume:
            continue

        q = m.get("question", "")
        if len(q) < 15:
            continue

        # ── Recency filter ──────────────────────────────────────────────────
        # Only apply when --days is set (for news-enabled runs).
        # This ensures the news fetched is genuinely useful context,
        # not a post-hoc report of an outcome from months ago.
        if cutoff_dt is not None:
            end_date_str = m.get("endDateIso") or m.get("endDate", "")
            if not end_date_str:
                skipped_recency += 1
                continue
            try:
                end_dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                # Normalize to UTC if the API returned a naive datetime
                if end_dt.tzinfo is None:
                    end_dt = end_dt.replace(tzinfo=timezone.utc)
                if end_dt < cutoff_dt:
                    skipped_recency += 1
                    continue
            except ValueError:
                skipped_recency += 1
                continue

        outcome = "YES" if yes_price >= 0.95 else "NO"

        results.append({
            "id": str(m.get("id", "")),
            "question": q,
            "description": m.get("description", "")[:500],
            "outcome": outcome,
            "volume": vol,
            "end_date": (m.get("endDateIso") or m.get("endDate", ""))[:10],
        })

        if len(results) >= limit:
            break

    if skipped_recency:
        logger.info(f"Skipped {skipped_recency} markets outside recency window")

    logger.info(f"Found {len(results)} resolved markets for backtesting")
    return results


def backtest_market(
    market_data: dict,
    backend,
    bankroll: float,
    edge_threshold: float,
    include_news: bool = False,
) -> BacktestResult | None:
    """
    Run a single resolved market through the model.

    We hide the market price (set to 0.50) so the model must reason
    independently — it can't just echo the crowd.

    When include_news=True, the model also gets recent DuckDuckGo headlines.
    This is valid for recently resolved markets (--days flag) since the
    news reflects genuinely contemporaneous information.
    """
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

    analysis = analyze_market(market, backend, include_news=include_news)
    if analysis is None:
        return None

    model_prob = analysis.estimated_probability
    actual_outcome = market_data["outcome"]

    # Determine what the model would bet (vs 50/50 baseline)
    if model_prob > 0.50 + edge_threshold:
        direction = "YES"
        edge = model_prob - 0.50
    elif model_prob < 0.50 - edge_threshold:
        direction = "NO"
        edge = 0.50 - model_prob
    else:
        direction = "SKIP"
        edge = abs(model_prob - 0.50)

    stake = 0.0
    if direction != "SKIP":
        stake = kelly_stake(
            bankroll=bankroll,
            edge=edge,
            market_price=0.50,
            direction=direction,
        )

    won = False
    pnl = 0.0
    if stake > 0:
        won = (direction == "YES" and actual_outcome == "YES") or \
              (direction == "NO"  and actual_outcome == "NO")
        pnl = stake * (1 / 0.50 - 1) if won else -stake

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
        end_date=market_data["end_date"],
        news_used=include_news,
    )


def print_results(results: list[BacktestResult], include_news: bool, max_days_old: int | None):
    """Print detailed backtest results."""
    news_label = "ENABLED ✓" if include_news else "DISABLED (no data leakage)"
    recency_label = f"last {max_days_old} days" if max_days_old else "all time"

    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║               🧪 Backtest Results                               ║")
    print(f"║  News:     {news_label:<53}║")
    print(f"║  Recency:  {recency_label:<53}║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    bets  = [r for r in results if r.direction != "SKIP"]
    skips = [r for r in results if r.direction == "SKIP"]

    print(f"\n  {'#':<4} {'Result':<8} {'Pred':>5} {'Actual':<6} {'Bet':<5} "
          f"{'Stake':>7} {'P&L':>8} {'Question'}")
    print(f"  {'─' * 85}")

    for i, r in enumerate(results, 1):
        if r.direction == "SKIP":
            icon, result_str, pnl_str = "⏭️", "skip", "   —"
        elif r.won:
            icon, result_str = "✅", "WIN"
            pnl_str = f"${r.pnl:+.2f}"
        else:
            icon, result_str = "❌", "LOSS"
            pnl_str = f"${r.pnl:+.2f}"

        print(f"  {i:<4} {icon} {result_str:<5} {r.model_probability:>4.0%}  "
              f"{r.actual_outcome:<6} {r.direction:<5} "
              f"${r.stake:>6.2f} {pnl_str:>8}  {r.question[:40]}")

    # ── Summary stats ────────────────────────────────────────────────────
    print(f"\n{'═' * 65}")

    brier_sum = sum(
        (r.model_probability - (1.0 if r.actual_outcome == "YES" else 0.0)) ** 2
        for r in results
    )
    brier = brier_sum / len(results)

    wins        = [r for r in bets if r.won]
    losses      = [r for r in bets if not r.won]
    total_pnl   = sum(r.pnl for r in bets)
    total_staked = sum(r.stake for r in bets)
    roi         = (total_pnl / total_staked * 100) if total_staked > 0 else 0

    print(f"  Markets tested:    {len(results)}")
    print(f"  Bets placed:       {len(bets)} ({len(skips)} skipped)")
    if bets:
        print(f"  Win / Loss:        {len(wins)}W / {len(losses)}L "
              f"({len(wins)/len(bets)*100:.0f}% win rate)")
    print(f"  Total staked:      ${total_staked:.2f}")
    print(f"  Net P&L:           ${total_pnl:+.2f}")
    print(f"  ROI:               {roi:+.1f}%")

    print(f"\n  Brier Score:       {brier:.4f}")
    if brier < 0.15:
        print("  Assessment:        🏆 Excellent — model has real predictive power")
    elif brier < 0.25:
        print("  Assessment:        👍 Good — model adds signal over 50/50 baseline")
    elif brier < 0.30:
        print("  Assessment:        ⚠️  Marginal — close to no-skill baseline")
    else:
        print("  Assessment:        🚨 Poor — model is not well-calibrated")

    # ── Calibration buckets ──────────────────────────────────────────────
    print(f"\n  Calibration by prediction bucket:")
    print(f"  {'Bucket':<12} {'Count':<7} {'Model Avg':<11} {'Actual Rate':<12} {'Gap'}")
    print(f"  {'─'*50}")

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

    # ── News impact note ─────────────────────────────────────────────────
    if include_news:
        print(f"\n  ℹ️  News was ENABLED. Run without --news on the same markets")
        print(f"     to measure the raw lift news context provides.")

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
    parser = argparse.ArgumentParser(
        description="Backtest model against resolved Polymarket data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python backtest.py                         # 20 markets, no news (safe baseline)
  python backtest.py --news --days 7         # Recent markets WITH news (live sim)
  python backtest.py --news --days 3 --markets 30
  python backtest.py --markets 50 --min-volume 25000

News + recency guidance:
  --news without --days   ⚠️  Risk of data leakage on old resolved markets.
                              The news may literally report the outcome.
  --news --days 7         ✅  Safe. News is contemporaneous with resolution.
  --news --days 30        ⚠️  Marginal. Some leakage risk for older markets.
        """
    )
    parser.add_argument("--markets",        type=int,   default=20,
                        help="Number of resolved markets to test (default: 20)")
    parser.add_argument("--min-volume",     type=float, default=50000,
                        help="Minimum volume filter (default: $50,000)")
    parser.add_argument("--edge-threshold", type=float, default=0.08,
                        help="Min edge to place bet (default: 8%%)")
    parser.add_argument("--bankroll",       type=float, default=100,
                        help="Simulated bankroll (default: $100)")
    parser.add_argument("--news",           action="store_true",
                        help="Enable DuckDuckGo news context for each market")
    parser.add_argument("--days",           type=int,   default=None,
                        help="Only test markets resolved within last N days. "
                             "Recommended when using --news to avoid data leakage.")
    args = parser.parse_args()

    # Warn if news is enabled without a recency filter
    if args.news and args.days is None:
        print("⚠️  WARNING: --news enabled without --days.")
        print("   Old resolved markets risk data leakage — news may describe the outcome.")
        print("   Consider: python backtest.py --news --days 7\n")

    # Init LLM backend
    try:
        backend = get_backend()
        logger.info(f"Using backend: {backend.name}")
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)

    # Fetch resolved markets
    resolved = fetch_resolved_markets(
        limit=args.markets,
        min_volume=args.min_volume,
        max_days_old=args.days,
    )

    if not resolved:
        msg = "No resolved markets found."
        if args.days:
            msg += f" Try increasing --days (currently {args.days}) or lowering --min-volume."
        else:
            msg += " Try lowering --min-volume."
        print(msg)
        return

    # Run backtest
    news_label = "ENABLED" if args.news else "DISABLED"
    days_label = f"last {args.days}d" if args.days else "all time"
    print(f"\n🧪 Backtesting {len(resolved)} markets...")
    print(f"   Backend:   {backend.name}")
    print(f"   News:      {news_label}")
    print(f"   Recency:   {days_label}")
    print(f"   Threshold: {args.edge_threshold:.0%}")
    print(f"   Bankroll:  ${args.bankroll:.0f}\n")

    results = []
    for i, m in enumerate(resolved, 1):
        logger.info(f"[{i}/{len(resolved)}] {m['question'][:60]}...")
        result = backtest_market(
            m, backend, args.bankroll, args.edge_threshold,
            include_news=args.news,
        )
        if result:
            results.append(result)
            icon = "✅" if result.won else ("❌" if result.direction != "SKIP" else "⏭️")
            logger.info(
                f"  {icon} Pred: {result.model_probability:.0%} | "
                f"Actual: {result.actual_outcome} | Bet: {result.direction}"
            )
        else:
            logger.warning("  ⚠ Analysis failed — skipping")

    if not results:
        print("All analyses failed. Check your API key.")
        return

    # Save results
    output_path = Path(__file__).parent / "data" / "backtest_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps([asdict(r) for r in results], indent=2))
    logger.info(f"Results saved to {output_path}")

    print_results(results, include_news=args.news, max_days_old=args.days)


if __name__ == "__main__":
    main()