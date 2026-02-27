#!/usr/bin/env python3
"""
Stringent Categorical Backtester (Phase 1-3)
─────────────────────────────────────────────
Fetches resolved markets from Polymarket to rigorously validate the
model's edge before live trading.

Features:
- --days filter: only test markets resolved in the last N days
- --exclude-categories: skip specific categories
- --list-categories: print all valid category names
- Categorization (Politics, Weather, Crypto Token, etc.)
- Pagination to bypass API limits
- Simulated execution slippage penalty
- Strict success thresholds per category

Usage (run from project root OR backtests/ folder):
    python backtests/backtest_2.py --days 7 --markets 100
    python backtests/backtest_2.py --days 7 --exclude-categories "Crypto Price"
    python backtests/backtest_2.py --list-categories
"""

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

# ── Path fix: works whether run from root or from backtests/ ──────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
load_dotenv(_ROOT / ".env")
# ─────────────────────────────────────────────────────────────────────────────

from src.agent import get_backend, analyze_market
from src.market import Market
from src.trader import kelly_stake

for lib in ["urllib3", "requests", "primp", "h2", "rustls", "hyper_util", "cookie_store", "ddgs"]:
    logging.getLogger(lib).setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

GAMMA_API_BASE = "https://gamma-api.polymarket.com"

ALL_CATEGORIES = [
    "Crypto Price",
    "Crypto Token",
    "Politics/Geo",
    "Sports",
    "Esports",
    "Weather",
    "Speech/Social",
    "Soccer/Tennis",
    "Other",
]


@dataclass
class StrictBacktestResult:
    market_id: str
    question: str
    category: str
    actual_outcome: str
    model_probability: float
    direction: str
    stake: float
    won: bool
    pnl: float
    end_date: str


def categorize(q: str) -> str:
    q = q.lower()
    if any(w in q for w in ["bitcoin", "btc", "ethereum", "eth", "xrp", "solana", "crypto"]):
        return "Crypto Price"
    if any(w in q for w in ["fdv", "pump.fun", "superform", "aztec", "launch", "airdrop"]):
        return "Crypto Token"
    if any(w in q for w in ["trump", "pardon", "election", "democrat", "republican",
                             "primary", "vote", "cabinet", "president", "congress",
                             "nato", "israel", "saudi", "russia", "ukraine"]):
        return "Politics/Geo"
    if any(w in q for w in ["nba", "nfl", "spread:", "o/u", "vs.", "win on", "beat",
                             "baseball", "reds", "braves", "astros", "nationals",
                             "texans", "chargers", "knicks", "lakers", "76ers",
                             "raptors", "sabres", "warriors", "suns"]):
        return "Sports"
    if any(w in q for w in ["counter-strike", "dota", "esports", "bo3", "lol:"]):
        return "Esports"
    if any(w in q for w in ["temperature", "weather", "°f", "°c"]):
        return "Weather"
    if any(w in q for w in ["tweet", "musk post", 'say "', "tweets from"]):
        return "Speech/Social"
    if any(w in q for w in ["chelsea", "manchester", "palmeiras", "brugge", "lyonnais",
                             "marseille", "barcelona", "atletico", "villarreal", "nice",
                             "bratislava", "crystal palace", "aston villa", "swiatek",
                             "open:", "groningen", "rio open", "mexican open"]):
        return "Soccer/Tennis"
    return "Other"


def fetch_dataset(
    target_count: int,
    min_volume: float = 5000,
    max_days_old: int = None,
    exclude_categories: set = None,
) -> list[dict]:
    exclude_categories = exclude_categories or set()
    recency_label = f"last {max_days_old} day(s)" if max_days_old else "all time"

    logger.info(
        f"Fetching resolved markets "
        f"(target={target_count}, min_vol=${min_volume:,.0f}, recency={recency_label}"
        + (f", excluding={sorted(exclude_categories)}" if exclude_categories else "")
        + ")..."
    )

    cutoff_dt = None
    if max_days_old is not None:
        cutoff_dt = datetime.now(timezone.utc) - timedelta(days=max_days_old)

    results = []
    limit = 100
    offset = 0
    skipped_recency = 0
    skipped_category = 0
    pages_fetched = 0

    while len(results) < target_count:
        params = {
            "closed": "true",
            "limit": limit,
            "offset": offset,
            "order": "volume",
            "ascending": "false",
        }

        try:
            resp = requests.get(f"{GAMMA_API_BASE}/markets", params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logger.error(f"API Error: {e}")
            break

        if not data:
            logger.info("No more pages from API.")
            break

        pages_fetched += 1
        found_this_page = 0

        for m in data:
            prices = json.loads(m.get("outcomePrices", "[]"))
            if len(prices) < 2:
                continue

            yes_price = float(prices[0])
            vol = float(m.get("volume", 0))

            if 0.05 < yes_price < 0.95:
                continue
            if vol < min_volume:
                continue

            q = m.get("question", "")
            if len(q) < 15:
                continue

            if cutoff_dt is not None:
                end_date_str = m.get("endDateIso") or m.get("endDate", "")
                if not end_date_str:
                    skipped_recency += 1
                    continue
                try:
                    end_dt = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
                    if end_dt.tzinfo is None:
                        end_dt = end_dt.replace(tzinfo=timezone.utc)
                    if end_dt < cutoff_dt:
                        skipped_recency += 1
                        continue
                except ValueError:
                    skipped_recency += 1
                    continue

            category = categorize(q)
            if category in exclude_categories:
                skipped_category += 1
                continue

            outcome = "YES" if yes_price >= 0.95 else "NO"
            end_date = (m.get("endDateIso") or m.get("endDate", ""))[:10]

            results.append({
                "id": str(m.get("id", "")),
                "question": q,
                "description": m.get("description", "")[:500],
                "outcome": outcome,
                "volume": vol,
                "category": category,
                "end_date": end_date,
            })
            found_this_page += 1

            if len(results) >= target_count:
                break

        logger.info(
            f"  Page {pages_fetched} (offset={offset}): "
            f"+{found_this_page} valid | total={len(results)} | "
            f"recency_skip={skipped_recency} | cat_skip={skipped_category}"
        )

        if cutoff_dt is not None and found_this_page == 0 and pages_fetched > 1:
            logger.info("No recent markets on this page — stopping pagination.")
            break

        offset += limit
        time.sleep(0.8)

    if skipped_recency:
        logger.info(f"Total skipped (recency): {skipped_recency}")
    if skipped_category:
        logger.info(f"Total skipped (category): {skipped_category}")

    logger.info(f"Dataset ready: {len(results)} markets.")
    return results


def run_backtest_market(
    market_data: dict,
    backend,
    bankroll: float,
    edge_threshold: float,
    slippage: float,
) -> StrictBacktestResult | None:
    market = Market(
        id=market_data["id"],
        question=market_data["question"],
        description=market_data["description"],
        yes_price=0.50,
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

    exec_price_yes = 0.50 + slippage
    exec_price_no  = 0.50 + slippage

    if model_prob > exec_price_yes + edge_threshold:
        direction   = "YES"
        edge        = model_prob - exec_price_yes
        stake_price = exec_price_yes
    elif (1 - model_prob) > exec_price_no + edge_threshold:
        direction   = "NO"
        edge        = (1 - model_prob) - exec_price_no
        stake_price = exec_price_no
    else:
        direction   = "SKIP"
        edge        = 0.0
        stake_price = 0.50

    stake = 0.0
    won   = False
    pnl   = 0.0

    if direction != "SKIP":
        stake = kelly_stake(bankroll, edge, stake_price, direction, max_pct=0.10)
        won   = (direction == actual_outcome)
        pnl   = stake * (1 / stake_price - 1) if won else -stake

    return StrictBacktestResult(
        market_id=market_data["id"],
        question=market_data["question"][:70],
        category=market_data["category"],
        actual_outcome=actual_outcome,
        model_probability=model_prob,
        direction=direction,
        stake=stake,
        won=won,
        pnl=pnl,
        end_date=market_data["end_date"],
    )


def print_report(
    results: list[StrictBacktestResult],
    slippage: float,
    backend_name: str,
    max_days_old: int | None,
    excluded: set,
):
    recency_label = f"last {max_days_old} day(s)" if max_days_old else "all time"

    print("\n" + "═" * 72)
    print(f"  📊 BACKTEST RESULTS")
    print(f"  Model    : {backend_name}")
    print(f"  Recency  : {recency_label}  |  Slippage: {slippage:.1%}")
    if excluded:
        print(f"  Excluded : {', '.join(sorted(excluded))}")
    print("═" * 72)

    if not results:
        print("  No results to display.")
        return

    print(f"\n  {'#':<4} {'✓/✗':<3} {'Prob':>5} {'Act':<4} {'Bet':<5} {'P&L':>8}  {'Category':<14}  Question")
    print("  " + "─" * 90)

    for i, r in enumerate(results, 1):
        if r.direction == "SKIP":
            icon, pnl_str = "⏭", "      —"
        elif r.won:
            icon, pnl_str = "✅", f"${r.pnl:+7.2f}"
        else:
            icon, pnl_str = "❌", f"${r.pnl:+7.2f}"

        print(
            f"  {i:<4} {icon:<3} {r.model_probability:>4.0%}  "
            f"{r.actual_outcome:<4} {r.direction:<5} {pnl_str}  "
            f"{r.category:<14}  {r.question[:42]}"
        )

    bets  = [r for r in results if r.direction != "SKIP"]
    wins  = [r for r in bets if r.won]
    skips = [r for r in results if r.direction == "SKIP"]

    total_pnl    = sum(r.pnl for r in bets)
    total_staked = sum(r.stake for r in bets)
    roi          = (total_pnl / total_staked * 100) if total_staked > 0 else 0

    brier_sum = sum(
        (r.model_probability - (1.0 if r.actual_outcome == "YES" else 0.0)) ** 2
        for r in results
    )
    brier   = brier_sum / len(results)
    win_pct = len(wins) / len(bets) * 100 if bets else 0

    print(f"\n{'─' * 72}")
    print(f"  OVERALL")
    print(f"  Markets analyzed : {len(results)}  ({len(skips)} skipped, {len(bets)} bets)")
    print(f"  Win / Loss       : {len(wins)}W / {len(bets)-len(wins)}L  ({win_pct:.0f}% win rate)")
    print(f"  Total staked     : ${total_staked:.2f}")
    print(f"  Net P&L          : ${total_pnl:+.2f}")
    print(f"  ROI              : {roi:+.1f}%")
    print(f"  Brier Score      : {brier:.4f}  ", end="")
    if brier < 0.15:
        print("🏆 Excellent")
    elif brier < 0.20:
        print("👍 Good")
    elif brier < 0.25:
        print("⚠️  Baseline (marginal skill)")
    else:
        print("🚨 Poor — no better than random")

    print(f"\n  CATEGORY BREAKDOWN")
    print(f"  {'Category':<16} {'N':>4} {'Bets':>5} {'Win%':>6} {'P&L':>9} {'Brier':>7}  Verdict")
    print("  " + "─" * 72)

    cats = defaultdict(lambda: {"n": 0, "bets": 0, "wins": 0, "pnl": 0.0, "brier_sum": 0.0})
    for r in results:
        c = cats[r.category]
        c["n"] += 1
        c["brier_sum"] += (r.model_probability - (1.0 if r.actual_outcome == "YES" else 0.0)) ** 2
        if r.direction != "SKIP":
            c["bets"] += 1
            if r.won:
                c["wins"] += 1
            c["pnl"] += r.pnl

    for cat in sorted(cats, key=lambda x: -cats[x]["n"]):
        c = cats[cat]
        c_brier   = c["brier_sum"] / c["n"]
        c_win_pct = (c["wins"] / c["bets"] * 100) if c["bets"] > 0 else 0

        if c["n"] < 5:
            verdict = "🟡 Needs more data"
        elif c["bets"] == 0:
            verdict = "⏭  All skipped"
        elif c_win_pct >= 60 and c["pnl"] > 0 and c_brier < 0.22:
            verdict = "✅ Approved"
        elif c_win_pct >= 55 and c["pnl"] > 0:
            verdict = "⚠️  Borderline"
        else:
            verdict = "❌ Reject"

        print(
            f"  {cat:<16} {c['n']:>4} {c['bets']:>5} {c_win_pct:>5.0f}%  "
            f"${c['pnl']:>7.2f} {c_brier:>7.3f}  {verdict}"
        )

    print(f"\n{'═' * 72}")
    if brier < 0.18 and roi > 0 and win_pct >= 55:
        print("  ✅ GREEN LIGHT — Model is calibrated and profitable.")
        print("     Ready for cautious live trading.")
    elif brier < 0.23 or roi > 0:
        print("  🟡 YELLOW — Some skill detected, needs more data.")
        print("     Continue dry-run and monitor.")
    else:
        print("  🔴 RED — Model not performing well enough for live trading.")
        print("     Try a stronger model or refine prompts.")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Backtest model on recently resolved Polymarket markets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Category names (for --exclude-categories):
{chr(10).join('  ' + c for c in ALL_CATEGORIES)}

Examples:
  python backtests/backtest_2.py --days 7 --exclude-categories "Crypto Price"
  python backtests/backtest_2.py --days 30 --min-volume 50000 \\
      --exclude-categories "Crypto Price" "Esports"
  python backtests/backtest_2.py --list-categories
        """
    )
    parser.add_argument("--days",           type=int,   default=7)
    parser.add_argument("--markets",        type=int,   default=100)
    parser.add_argument("--min-volume",     type=float, default=5000)
    parser.add_argument("--slippage",       type=float, default=0.02)
    parser.add_argument("--edge-threshold", type=float, default=0.08)
    parser.add_argument("--bankroll",       type=float, default=1000)
    parser.add_argument("--no-days",        action="store_true")
    parser.add_argument("--exclude-categories", nargs="+", metavar="CATEGORY", default=[])
    parser.add_argument("--list-categories",    action="store_true")
    args = parser.parse_args()

    if args.list_categories:
        print("\n📋 Valid category names for --exclude-categories:\n")
        for cat in ALL_CATEGORIES:
            print(f'  "{cat}"')
        print(f'\nExample:')
        print(f'  python backtests/backtest_2.py --exclude-categories "Crypto Price" "Esports"\n')
        return

    excluded = set(args.exclude_categories)
    invalid  = excluded - set(ALL_CATEGORIES)
    if invalid:
        print(f"\n❌ Unknown categories: {invalid}")
        print(f"   Run --list-categories to see valid options.\n")
        return

    max_days = None if args.no_days else args.days

    try:
        backend = get_backend()
    except ValueError as e:
        print(f"❌ {e}")
        return

    print(f"\n🧪 Polymarket Backtest — Recently Resolved Markets")
    print(f"   Model     : {backend.name}")
    print(f"   Recency   : {'No filter' if max_days is None else f'Last {max_days} day(s)'}")
    print(f"   Markets   : up to {args.markets}")
    print(f"   Min volume: ${args.min_volume:,.0f}")
    print(f"   Slippage  : {args.slippage:.1%}")
    print(f"   Edge bar  : {args.edge_threshold:.1%} over slippage")
    if excluded:
        print(f"   Excluded  : {', '.join(sorted(excluded))}")
    print()

    markets = fetch_dataset(
        target_count=args.markets,
        min_volume=args.min_volume,
        max_days_old=max_days,
        exclude_categories=excluded,
    )

    if not markets:
        print("❌ No markets found. Try --no-days, lower --min-volume, or fewer exclusions.")
        return

    yes_n = sum(1 for m in markets if m["outcome"] == "YES")
    cat_counts = defaultdict(int)
    for m in markets:
        cat_counts[m["category"]] += 1

    print(f"  Test set  : {len(markets)} markets  ({yes_n} YES / {len(markets)-yes_n} NO)")
    print(f"  By category: {dict(sorted(cat_counts.items(), key=lambda x: -x[1]))}\n")

    results = []
    for i, m in enumerate(markets, 1):
        print(f"  [{i:>3}/{len(markets)}] {m['outcome']} | [{m['category']}] {m['question'][:55]}")
        res = run_backtest_market(m, backend, args.bankroll, args.edge_threshold, args.slippage)
        if res:
            icon = "✅" if res.won else ("❌" if res.direction != "SKIP" else "⏭ ")
            print(f"         {icon} Pred={res.model_probability:.0%} | Bet={res.direction} | P&L=${res.pnl:+.2f}")
            results.append(res)
        else:
            print("         ⚠️  Analysis failed")
        time.sleep(1.5)

    print_report(results, args.slippage, backend.name, max_days, excluded)


if __name__ == "__main__":
    main()