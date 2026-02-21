#!/usr/bin/env python3
"""
Auto-resolution & Calibration Script
─────────────────────────────────────
Checks Polymarket for resolved outcomes, updates the trade log,
and prints your Brier score calibration report.

Usage:
    python resolve.py              # Auto-resolve + show calibration
    python resolve.py --dry-run    # Show what would be resolved without writing
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

from src.trader import TradeLogger

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

GAMMA_API_BASE = "https://gamma-api.polymarket.com"


def check_resolution(market_id: str) -> dict:
    """
    Check a market's resolution status on Polymarket.

    Returns dict with:
        resolved: bool
        outcome: "YES" | "NO" | None
        closed: bool
        prices: current prices
    """
    url = f"{GAMMA_API_BASE}/markets/{market_id}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        closed = data.get("closed", False)
        active = data.get("active", True)
        resolution = data.get("resolution", None)

        # Parse outcome prices
        prices = json.loads(data.get("outcomePrices", "[]"))
        yes_price = float(prices[0]) if len(prices) > 0 else 0.5
        no_price = float(prices[1]) if len(prices) > 1 else 0.5

        # Determine outcome
        outcome = None
        is_resolved = False

        if resolution:
            # Explicit resolution field
            is_resolved = True
            outcome = resolution.upper() if resolution.lower() in ("yes", "no") else None
        elif closed and not active:
            # Market closed — check prices to infer outcome
            # Resolved markets typically snap to 1.00/0.00
            if yes_price >= 0.95:
                is_resolved = True
                outcome = "YES"
            elif no_price >= 0.95:
                is_resolved = True
                outcome = "NO"

        return {
            "resolved": is_resolved,
            "outcome": outcome,
            "closed": closed,
            "active": active,
            "yes_price": yes_price,
            "question": data.get("question", "")[:70],
        }
    except (requests.RequestException, json.JSONDecodeError, ValueError) as e:
        logger.warning(f"  Failed to check market {market_id}: {e}")
        return {"resolved": False, "outcome": None, "closed": False, "active": True}


def main():
    parser = argparse.ArgumentParser(description="Auto-resolve markets & check calibration")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be resolved without writing changes")
    args = parser.parse_args()

    trade_logger = TradeLogger()

    # ── Step 1: Find unresolved markets ──────────────────────────────────
    unresolved = trade_logger.get_unresolved_markets()
    if not unresolved:
        print("No unresolved markets to check.")
    else:
        print(f"\n🔍 Checking {len(unresolved)} unresolved markets against Polymarket...\n")

        resolved_count = 0
        still_open = 0

        for m in unresolved:
            mid = m["market_id"]
            status = check_resolution(mid)

            if status["resolved"] and status["outcome"]:
                resolved_count += 1
                symbol = "✅" if status["outcome"] == "YES" else "❌"
                print(f"  {symbol} [{status['outcome']}] {status['question']}")
                print(f"     Model predicted: {m['model_prob']:.1%} | Market ID: {mid}")

                if not args.dry_run:
                    trade_logger.resolve_market(mid, status["outcome"])
            elif status["closed"]:
                print(f"  🔒 Closed but unclear outcome: {status['question']}")
                still_open += 1
            else:
                still_open += 1

        print(f"\n  Resolved: {resolved_count} | Still open: {still_open}")
        if args.dry_run and resolved_count > 0:
            print("  (Dry run — no changes written. Remove --dry-run to save.)")

    # ── Step 2: Show calibration report ──────────────────────────────────
    cal = trade_logger.get_calibration()

    print("\n" + "═" * 55)
    print("📐 Calibration Report")
    print("═" * 55)

    if cal.get("num_resolved", 0) == 0:
        print(f"\n  {cal.get('message', 'No resolved data yet.')}")
        print("\n  💡 Tip: Markets need to expire and resolve before")
        print("     calibration can be calculated. Keep running the agent!")
        return

    brier = cal["brier_score"]
    interp = cal["brier_interpretation"]

    # Visual Brier score bar
    bar_len = 30
    pos = min(int(brier / 0.5 * bar_len), bar_len)
    bar = "█" * pos + "░" * (bar_len - pos)
    print(f"\n  Brier Score: {brier:.4f}  [{bar}]")
    print(f"  Assessment:  {interp}")
    print(f"  Markets:     {cal['num_resolved']}")

    print(f"\n  {'Bucket':<12} {'Count':<7} {'Forecast':<10} {'Actual':<10} {'Gap':<6}")
    print(f"  {'─' * 45}")
    for bucket, data in cal.get("calibration_buckets", {}).items():
        gap_bar = "●" * min(int(data["gap"] * 20), 10)
        print(f"  {bucket:<12} {data['count']:<7} {data['avg_forecast']:<10.1%} "
              f"{data['avg_actual']:<10.1%} {gap_bar}")

    # ── Step 3: Verdict ──────────────────────────────────────────────────
    print(f"\n{'─' * 55}")
    if brier < 0.10:
        print("  🏆 EXCELLENT — Your model significantly beats the market.")
        print("  ✅ GREEN LIGHT for cautious live trading.")
    elif brier < 0.20:
        print("  👍 GOOD — Your model adds meaningful signal.")
        print("  🟡 Consider live trading with small stakes.")
    elif brier < 0.26:
        print("  ⚠️  BASELINE — Model is no better than the market price.")
        print("  🔴 Do NOT go live. Tune prompts or try a stronger model.")
    else:
        print("  🚨 POOR — Model is worse than random guessing.")
        print("  🔴 Significant rework needed before any live trading.")

    # ── Step 4: Portfolio summary ────────────────────────────────────────
    summary = trade_logger.get_summary()
    print(f"\n{'─' * 55}")
    print("📊 Portfolio Summary")
    print(f"  Decisions: {summary['total_decisions']} | "
          f"Trades: {summary['trades_placed']} | "
          f"Staked: ${summary['total_staked_usd']:.2f}")
    print(f"  Avg Edge: {summary['avg_edge']:.1%} | "
          f"Resolved: {summary.get('markets_resolved', 0)}")
    print()


if __name__ == "__main__":
    main()
