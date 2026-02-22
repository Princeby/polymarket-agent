#!/usr/bin/env python3
"""
Stringent Categorical Backtester (Phase 1-3)
─────────────────────────────────────────────
Fetches large datasets (500+ markets) from Polymarket to rigorously
validate the model's edge before live trading.

Features:
- Categorization (Politics, Weather, Crypto Token, etc.)
- Pagination to bypass API limits
- Simulated execution slippage penalty
- Strict success thresholds

Usage:
    python backtest_2.py --markets 500 --slippage 0.02
"""

import argparse
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass

import requests
from dotenv import load_dotenv

from src.agent import get_backend, analyze_market
from src.market import Market
from src.trader import kelly_stake

load_dotenv()

# Quiet noisy libraries
for lib in ["urllib3", "requests", "primp", "h2", "rustls", "hyper_util", "cookie_store", "ddgs"]:
    logging.getLogger(lib).setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

GAMMA_API_BASE = "https://gamma-api.polymarket.com"


@dataclass
class StrictBacktestResult:
    market_id: str
    question: str
    category: str
    actual_outcome: str       # "YES" or "NO"
    model_probability: float  # model's P(YES)
    direction: str            # "YES" or "NO"
    stake: float
    won: bool
    pnl: float


def categorize(q: str) -> str:
    """Categorize market based on question content."""
    q = q.lower()
    if any(w in q for w in ['bitcoin', 'btc', 'ethereum', 'eth', 'xrp', 'solana', 'crypto']):
        return 'Crypto Price'
    if any(w in q for w in ['fdv', 'pump.fun', 'superform', 'aztec', 'launch', 'airdrop']):
        return 'Crypto Token'
    if any(w in q for w in ['trump', 'pardon', 'election', 'democrat', 'republican', 'primary', 'vote', 'cabinet', 'president', 'congress', 'nato', 'israel', 'saudi', 'russia', 'ukraine']):
        return 'Politics/Geo'
    if any(w in q for w in ['nba', 'nfl', 'spread:', 'o/u', 'vs.', 'win on', 'beat', 'baseball', 'reds', 'braves', 'astros', 'nationals', 'texans', 'chargers', 'knicks', 'lakers', '76ers', 'raptors', 'sabres', 'warriors', 'suns']):
        return 'Sports'
    if any(w in q for w in ['counter-strike', 'dota', 'esports', 'bo3']):
        return 'Esports'
    if any(w in q for w in ['temperature', 'weather', '°f', '°c']):
        return 'Weather'
    if any(w in q for w in ['tweet', 'musk post', 'say "']):
        return 'Speech/Social'
    if any(w in q for w in ['chelsea', 'manchester', 'palmeiras', 'brugge', 'lyonnais', 'marseille', 'barcelona', 'atletico', 'villarreal', 'nice', 'bratislava', 'crystal palace', 'aston villa', 'swiatek', 'open', 'groningen']):
        return 'Soccer/Tennis'
    return 'Other'


def fetch_large_dataset(target_count: int, min_volume: float = 10000) -> list[dict]:
    """Paginate through Polymarket API to build a massive test set."""
    logger.info(f"Fetching {target_count} resolved markets (min_vol=${min_volume:,.0f})...")
    
    results = []
    limit = 100
    offset = 0

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
            break

        for m in data:
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

            outcome = "YES" if yes_price >= 0.95 else "NO"
            
            results.append({
                "id": str(m.get("id", "")),
                "question": m.get("question", ""),
                "description": m.get("description", "")[:500],
                "outcome": outcome,
                "volume": vol,
                "category": categorize(m.get("question", "")),
                "end_date": m.get("endDate", "")[:10]
            })

            if len(results) >= target_count:
                break
        
        offset += limit
        time.sleep(1) # Be gentle on API

    logger.info(f"Successfully fetched {len(results)} valid markets.")
    return results


def run_stringent_backtest(market_data: dict, backend, bankroll: float, edge_threshold: float, slippage: float) -> StrictBacktestResult | None:
    """Analyze market without news, applying a slippage penalty to the execution price."""
    # Build market
    market = Market(
        id=market_data["id"],
        question=market_data["question"],
        description=market_data["description"],
        yes_price=0.50,  # Base price for testing
        no_price=0.50,
        volume=market_data["volume"],
        liquidity=0,
        end_date=market_data["end_date"],
        active=False,
        slug="",
    )

    # Analyze WITHOUT news
    analysis = analyze_market(market, backend, include_news=False)
    if analysis is None:
        return None

    model_prob = analysis.estimated_probability
    actual_outcome = market_data["outcome"]
    
    # ── Slippage Simulation ──
    # If the model thinks P(YES) = 60%, but there's 2% slippage, the effective execution
    # price we'd get isn't 50¢, it's 52¢. We have to beat *that* price to have an edge.
    exec_price_yes = 0.50 + slippage
    exec_price_no = 0.50 + slippage

    if model_prob > exec_price_yes + edge_threshold:
        direction = "YES"
        edge = model_prob - exec_price_yes
        stake_price = exec_price_yes
    elif (1 - model_prob) > exec_price_no + edge_threshold:
        direction = "NO"
        edge = (1 - model_prob) - exec_price_no
        stake_price = exec_price_no
    else:
        direction = "SKIP"
        edge = 0.0
        stake_price = 0.50

    stake = 0.0
    won = False
    pnl = 0.0

    if direction != "SKIP":
        stake = kelly_stake(bankroll, edge, stake_price, direction, max_pct=0.10)
        
        if direction == "YES":
            won = actual_outcome == "YES"
        else:
            won = actual_outcome == "NO"

        if won:
            pnl = stake * (1 / stake_price - 1)
        else:
            pnl = -stake

    return StrictBacktestResult(
        market_id=market_data["id"],
        question=market_data["question"][:70],
        category=market_data["category"],
        actual_outcome=actual_outcome,
        model_probability=model_prob,
        direction=direction,
        stake=stake,
        won=won,
        pnl=pnl
    )


def print_stringent_report(results: list[StrictBacktestResult], slippage: float):
    print("\n" + "═" * 70)
    print(f"📉 STRINGENT CATEGORICAL BACKTEST REPORT (Slippage Penalty: {slippage:.1%})")
    print("═" * 70)

    # 1. Overall Performance
    bets = [r for r in results if r.direction != "SKIP"]
    wins = [r for r in bets if r.won]
    total_pnl = sum(r.pnl for r in bets)
    total_staked = sum(r.stake for r in bets)
    
    brier_sum = sum((r.model_probability - (1.0 if r.actual_outcome == "YES" else 0.0)) ** 2 for r in results)
    brier = brier_sum / len(results) if results else 0

    print(f"\n[OVERALL METRICS]")
    print(f"  Markets Analyzed:  {len(results)}")
    print(f"  Bets Placed:       {len(bets)} / {len(results)}")
    print(f"  Win / Loss:        {len(wins)}W / {len(bets)-len(wins)}L ({(len(wins)/len(bets)*100 if bets else 0):.1f}%)")
    print(f"  Net P&L:           ${total_pnl:+.2f}")
    print(f"  ROI:               {(total_pnl/total_staked*100 if total_staked else 0):+.1f}%")
    print(f"  Overall Brier:     {brier:.4f}")

    # 2. Category Breakdown
    print("\n[CATEGORY BREAKDOWN]")
    print(f"  {'Category':<20} {'N':>4} {'Bets':>5} {'Win%':>6} {'P&L':>8} {'Brier':>7} {'Verdict'}")
    print("  " + "-" * 75)

    cats = defaultdict(lambda: {'n': 0, 'bets': 0, 'wins': 0, 'pnl': 0.0, 'brier': 0.0})
    for r in results:
        c = cats[r.category]
        c['n'] += 1
        c['brier'] += (r.model_probability - (1.0 if r.actual_outcome == "YES" else 0.0)) ** 2
        if r.direction != "SKIP":
            c['bets'] += 1
            if r.won:
                c['wins'] += 1
            c['pnl'] += r.pnl

    for cat in sorted(cats, key=lambda x: -cats[x]['n']):
        c = cats[cat]
        c_brier = c['brier'] / c['n']
        c_win_pct = (c['wins'] / c['bets'] * 100) if c['bets'] > 0 else 0
        
        # Strict rules for live trading
        verdict = "❌ REJECT"
        if c['n'] >= 10 and c['bets'] >= 5 and c_win_pct >= 60 and c['pnl'] > 0 and c_brier < 0.22:
            verdict = "✅ APPROVED"
        elif c['n'] < 10:
            verdict = "🟡 NEEDS DATA"
        elif c_win_pct >= 55 and c['pnl'] > 0:
            verdict = "⚠️ BORDERLINE"

        print(f"  {cat:<20} {c['n']:>4} {c['bets']:>5} {c_win_pct:>5.0f}% ${c['pnl']:>7.2f} {c_brier:>7.3f}   {verdict}")

    print("\n" + "═" * 70)


def main():
    parser = argparse.ArgumentParser(description="Stringent pre-live validation")
    parser.add_argument("--markets", type=int, default=200, help="Markets to fetch via pagination")
    parser.add_argument("--min-volume", type=float, default=5000, help="Min volume filter")
    parser.add_argument("--slippage", type=float, default=0.02, help="Execution slippage penalty (e.g., 0.02 = 2%)")
    parser.add_argument("--edge-threshold", type=float, default=0.08, help="Min edge to bet")
    parser.add_argument("--bankroll", type=float, default=1000, help="Simulated bankroll for Kelly")
    args = parser.parse_args()

    backend = get_backend()
    markets = fetch_large_dataset(target_count=args.markets, min_volume=args.min_volume)
    
    if not markets:
        print("No markets found to test.")
        return

    print(f"\n🧪 Running strict backtest on {len(markets)} markets...")
    print(f"   Model:     {backend.name}")
    print(f"   Slippage:  {args.slippage:.1%} penalty per trade")
    print(f"   Threshold: {args.edge_threshold:.1%} min edge over slippage\n")

    results = []
    for i, m in enumerate(markets, 1):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(markets)}...")
            
        res = run_stringent_backtest(m, backend, args.bankroll, args.edge_threshold, args.slippage)
        if res:
            results.append(res)
        
        # Avoid Groq rate limits
        time.sleep(1.5)

    print_stringent_report(results, slippage=args.slippage)


if __name__ == "__main__":
    main()
