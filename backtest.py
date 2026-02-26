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
    python backtest.py                    # Default: 30 resolved markets
    python backtest.py --markets 50       # Test on 50 markets
    python backtest.py --min-volume 50000 # Only high-volume markets
    python backtest.py --category politics # Filter to a category
"""

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
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

for lib in ["urllib3", "requests", "primp", "h2", "rustls",
            "hyper_util", "cookie_store", "ddgs"]:
    logging.getLogger(lib).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

GAMMA_API_BASE = "https://gamma-api.polymarket.com"

# ── Junk market patterns to exclude from backtesting ──────────────────────────
# These are markets the model has no edge on (pure coin flips or sports spreads)
JUNK_PATTERNS = [
    "up or down",           # 5-minute crypto candles — pure noise
    "spread:",              # sports point spreads
    "o/u ",                 # over/under totals
    ": o/u ",
    "vs.",                  # generic team matchup (no context for model)
]

JUNK_EXACT_PREFIXES = [
    "spread:",
]


def is_junk_market(question: str) -> bool:
    """Return True if this market is noise the model can't edge on."""
    q = question.lower().strip()
    for pattern in JUNK_PATTERNS:
        if pattern in q:
            return True
    for prefix in JUNK_EXACT_PREFIXES:
        if q.startswith(prefix):
            return True
    return False


@dataclass
class BacktestResult:
    market_id: str
    question: str
    category: str
    actual_outcome: str        # "YES" or "NO"
    model_probability: float   # model's P(YES)
    market_price_at_close: float
    edge: float
    direction: str             # "YES", "NO", or "SKIP"
    stake: float
    won: bool
    pnl: float


def fetch_resolved_markets(
    limit: int = 30,
    min_volume: float = 50000,
) -> list[dict]:
    """
    Fetch closed markets from Polymarket with clear outcomes.
    Filters out junk (coin-flip) markets that pollute calibration metrics.
    """
    logger.info(
        f"Fetching resolved markets "
        f"(limit={limit}, min_vol=${min_volume:,.0f}, recency=all time)..."
    )

    url = f"{GAMMA_API_BASE}/markets"
    fetched = []
    offset = 0
    batch = 100  # fetch in larger batches since we filter heavily

    while len(fetched) < limit:
        params = {
            "closed": "true",
            "limit": batch,
            "offset": offset,
            "order": "volume",
            "ascending": "false",
        }
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            raw = resp.json()
        except requests.RequestException as e:
            logger.error(f"API error: {e}")
            break

        if not raw:
            break

        for m in raw:
            prices = json.loads(m.get("outcomePrices", "[]"))
            if len(prices) < 2:
                continue

            yes_price = float(prices[0])
            vol = float(m.get("volume", 0))
            q = m.get("question", "")

            # Must have snapped to a clear outcome
            if 0.05 < yes_price < 0.95:
                continue

            # Volume filter
            if vol < min_volume:
                continue

            # Skip trivially short questions
            if len(q) < 15:
                continue

            # ── KEY FILTER: skip junk markets ─────────────────────────────
            if is_junk_market(q):
                logger.debug(f"Filtered junk: {q[:60]}")
                continue

            outcome = "YES" if yes_price >= 0.95 else "NO"
            fetched.append({
                "id": str(m.get("id", "")),
                "question": q,
                "description": m.get("description", "")[:500],
                "outcome": outcome,
                "volume": vol,
                "end_date": m.get("endDate", "")[:10],
            })

            if len(fetched) >= limit:
                break

        offset += batch
        if offset > 2000:  # safety cap
            break

    logger.info(f"Found {len(fetched)} resolved markets for backtesting")
    return fetched


def backtest_market(
    market_data: dict,
    backend,
    bankroll: float,
    edge_threshold: float,
) -> BacktestResult | None:
    """
    Run a single resolved market through the model WITHOUT news.
    Market price is hidden (set to 0.50) so the model must reason independently.
    """
    market = Market(
        id=market_data["id"],
        question=market_data["question"],
        description=market_data["description"],
        yes_price=0.50,   # hide actual price to prevent leakage
        no_price=0.50,
        volume=market_data["volume"],
        liquidity=0,
        end_date=market_data["end_date"],
        active=False,
        slug="",
    )

    analysis = analyze_market(market, backend, include_news=False)
    if analysis is None:
        return None

    model_prob = analysis.estimated_probability
    actual_outcome = market_data["outcome"]
    actual_value = 1.0 if actual_outcome == "YES" else 0.0

    # Bet direction vs hidden 50/50 baseline
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
    won = False
    pnl = 0.0

    if direction != "SKIP":
        stake = kelly_stake(
            bankroll=bankroll,
            edge=edge,
            market_price=0.50,
            direction=direction,
        )
        if stake > 0:
            won = (
                actual_outcome == "YES" if direction == "YES"
                else actual_outcome == "NO"
            )
            pnl = stake if won else -stake  # 50/50 pays 2x

    return BacktestResult(
        market_id=market_data["id"],
        question=market_data["question"][:70],
        category="general",
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
    print("║               🧪 Backtest Results                               ║")
    print("║  News:     DISABLED (no data leakage)                           ║")
    print("║  Junk:     FILTERED (no coin-flips or sports spreads)           ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    bets = [r for r in results if r.direction != "SKIP"]
    skips = [r for r in results if r.direction == "SKIP"]

    print(
        f"\n  {'#':<4} {'Result':<8} {'Pred':>5} {'Actual':<6} {'Bet':<5} "
        f"{'Stake':>7} {'P&L':>8} {'Question'}"
    )
    print(f"  {'─' * 85}")

    for i, r in enumerate(results, 1):
        if r.direction == "SKIP":
            icon, result_str, pnl_str = "⏭️", "skip", "  —"
        elif r.won:
            icon, result_str, pnl_str = "✅", "WIN", f"${r.pnl:+.2f}"
        else:
            icon, result_str, pnl_str = "❌", "LOSS", f"${r.pnl:+.2f}"

        print(
            f"  {i:<4} {icon} {result_str:<5} {r.model_probability:>4.0%} "
            f"{r.actual_outcome:<6} {r.direction:<5} "
            f"${r.stake:>6.2f} {pnl_str:>8} {r.question[:40]}"
        )

    # ── Summary stats ──────────────────────────────────────────────────────────
    print(f"\n{'═' * 65}")

    brier_sum = sum(
        (r.model_probability - (1.0 if r.actual_outcome == "YES" else 0.0)) ** 2
        for r in results
    )
    brier = brier_sum / len(results)

    wins = [r for r in bets if r.won]
    losses = [r for r in bets if not r.won]
    total_pnl = sum(r.pnl for r in bets)
    total_staked = sum(r.stake for r in bets)
    roi = (total_pnl / total_staked * 100) if total_staked > 0 else 0

    print(f"  Markets tested:    {len(results)}")
    print(f"  Bets placed:       {len(bets)} ({len(skips)} skipped)")
    if bets:
        print(
            f"  Win / Loss:        {len(wins)}W / {len(losses)}L "
            f"({len(wins)/len(bets)*100:.0f}% win rate)"
        )
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

    # ── Calibration buckets ────────────────────────────────────────────────────
    print(f"\n  Calibration by prediction bucket:")
    print(
        f"  {'Bucket':<12} {'Count':<7} {'Model Avg':<11} "
        f"{'Actual Rate':<12} {'Gap'}"
    )
    print(f"  {'─'*50}")

    buckets = defaultdict(lambda: {"n": 0, "sum_p": 0.0, "sum_y": 0.0})
    for r in results:
        b = min(int(r.model_probability * 10) * 10, 90)
        buckets[b]["n"] += 1
        buckets[b]["sum_p"] += r.model_probability
        buckets[b]["sum_y"] += 1.0 if r.actual_outcome == "YES" else 0.0

    for b in sorted(buckets):
        d = buckets[b]
        avg_p = d["sum_p"] / d["n"]
        avg_y = d["sum_y"] / d["n"]
        gap = abs(avg_p - avg_y)
        gap_bar = "●" * min(int(gap * 20), 10)
        print(
            f"  {b}-{b+10}%{'':<5} {d['n']:<7} {avg_p:<11.1%} "
            f"{avg_y:<12.1%} {gap_bar}"
        )

    # ── Verdict ────────────────────────────────────────────────────────────────
    print(f"\n{'─' * 65}")
    if brier < 0.20 and roi > 0:
        print("  ✅ GREEN LIGHT — Model is calibrated AND profitable in backtest.")
        print("     Ready for cautious dry-run on live markets.")
    elif brier < 0.25:
        print("  🟡 YELLOW — Model shows some skill but needs more data.")
        print("     Continue dry-run and tune prompts.")
    else:
        print("  🔴 RED — Model needs improvement before live trading.")
        print("     Try a stronger model or better prompting strategy.")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Backtest model against resolved Polymarket data"
    )
    parser.add_argument("--markets", type=int, default=30,
                        help="Number of resolved markets to test")
    parser.add_argument("--min-volume", type=float, default=50000,
                        help="Minimum volume filter")
    parser.add_argument("--edge-threshold", type=float, default=0.08,
                        help="Min edge to place bet (default 8%%)")
    parser.add_argument("--bankroll", type=float, default=100,
                        help="Simulated bankroll")
    args = parser.parse_args()

    try:
        backend = get_backend()
        logger.info(f"Using backend: {backend.name}")
    except ValueError as e:
        print(f"❌ {e}")
        sys.exit(1)

    resolved = fetch_resolved_markets(
        limit=args.markets,
        min_volume=args.min_volume,
    )

    if not resolved:
        print("No resolved markets found. Try lowering --min-volume.")
        return

    print(f"\n🧪 Backtesting {len(resolved)} markets (news DISABLED)...")
    print(f"   Backend:   {backend.name}")
    print(f"   News:      DISABLED")
    print(f"   Junk:      FILTERED (coin-flips, spreads, O/U removed)")
    print(f"   Threshold: {args.edge_threshold:.0%}")
    print(f"   Bankroll:  ${args.bankroll:.0f}")

    results = []
    for i, m in enumerate(resolved, 1):
        logger.info(f"[{i}/{len(resolved)}] {m['question'][:60]}...")
        result = backtest_market(m, backend, args.bankroll, args.edge_threshold)

        if result:
            results.append(result)
            icon = (
                "✅" if result.won else
                "❌" if result.direction != "SKIP" else "⏭️"
            )
            logger.info(
                f"  {icon} Pred: {result.model_probability:.0%} | "
                f"Actual: {result.actual_outcome} | Bet: {result.direction}"
            )
        else:
            logger.warning("  ⚠ Analysis failed — skipping")

        time.sleep(1.5)  # Groq rate limit

    if not results:
        print("All analyses failed. Check your API key.")
        return

    output_path = Path(__file__).parent / "data" / "backtest_results.json"
    output_path.write_text(json.dumps([asdict(r) for r in results], indent=2))
    logger.info(f"Results saved to {output_path}")

    print_results(results)


if __name__ == "__main__":
    main()