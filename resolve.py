#!/usr/bin/env python3
"""
Auto-resolution & Calibration Script
──────────────────────────────────────
Checks Polymarket for resolved outcomes, updates the trade log,
and prints your Brier score calibration report.

Checks in order:
  1. Explicit `resolution` field on the market
  2. Price snapped to ≥0.95 or ≤0.05
  3. Market is closed/inactive AND end date has passed → flag for manual review

Usage:
    python resolve.py           # Auto-resolve + show calibration
    python resolve.py --dry-run # Show what would be resolved without writing
    python resolve.py --force   # Also show markets needing manual resolution
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
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

    Checks (in order of reliability):
      1. Explicit `resolution` field
      2. Price snapped to ≥0.95 (YES) or ≤0.05 (NO)
      3. Market closed + past end date → needs_manual_review

    Returns dict with:
      resolved:             bool
      outcome:              "YES" | "NO" | None
      needs_manual_review:  bool  (expired but outcome unclear)
      closed:               bool
      yes_price:            float
      question:             str
      end_date:             str
    """
    url = f"{GAMMA_API_BASE}/markets/{market_id}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, json.JSONDecodeError, ValueError) as e:
        logger.warning(f"  Failed to fetch market {market_id}: {e}")
        return {
            "resolved": False, "outcome": None,
            "needs_manual_review": False,
            "closed": False, "yes_price": 0.5,
            "question": "", "end_date": "",
        }

    closed   = data.get("closed", False)
    active   = data.get("active", True)
    resolution = data.get("resolution", None)
    end_date = data.get("endDateIso", data.get("endDate", ""))[:10]
    question = data.get("question", "")[:70]

    # Parse outcome prices
    try:
        prices = json.loads(data.get("outcomePrices", "[]"))
        yes_price = float(prices[0]) if len(prices) > 0 else 0.5
        no_price  = float(prices[1]) if len(prices) > 1 else 0.5
    except (json.JSONDecodeError, ValueError, IndexError):
        yes_price = no_price = 0.5

    outcome      = None
    is_resolved  = False
    needs_manual = False

    # ── Check 1: explicit resolution field ────────────────────────────────────
    if resolution:
        res_lower = resolution.lower().strip()
        if res_lower in ("yes", "1", "true"):
            is_resolved = True
            outcome = "YES"
        elif res_lower in ("no", "0", "false"):
            is_resolved = True
            outcome = "NO"

    # ── Check 2: price snapped to near 0 or 1 ────────────────────────────────
    if not is_resolved:
        if yes_price >= 0.95:
            is_resolved = True
            outcome = "YES"
        elif yes_price <= 0.05:
            is_resolved = True
            outcome = "NO"

    # ── Check 3: past end date but outcome still unclear ─────────────────────
    if not is_resolved and end_date:
        try:
            end_dt = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)
            now    = datetime.now(timezone.utc)
            if end_dt < now and (closed or not active):
                needs_manual = True
        except ValueError:
            pass

    return {
        "resolved":            is_resolved,
        "outcome":             outcome,
        "needs_manual_review": needs_manual,
        "closed":              closed,
        "active":              active,
        "yes_price":           yes_price,
        "question":            question,
        "end_date":            end_date,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Auto-resolve markets & check calibration"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be resolved without writing changes"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Also list markets that are expired but need manual resolution"
    )
    args = parser.parse_args()

    trade_logger = TradeLogger()

    # ── Step 1: Find unresolved markets ───────────────────────────────────────
    unresolved = trade_logger.get_unresolved_markets()

    if not unresolved:
        print("No unresolved markets to check.")
    else:
        print(f"\n🔍 Checking {len(unresolved)} unresolved markets...\n")

        auto_resolved  = []
        still_open     = []
        needs_manual   = []

        for m in unresolved:
            mid    = m["market_id"]
            status = check_resolution(mid)

            if status["resolved"] and status["outcome"]:
                auto_resolved.append((m, status))
            elif status["needs_manual_review"]:
                needs_manual.append((m, status))
            else:
                still_open.append((m, status))

        # ── Print auto-resolvable ──────────────────────────────────────────────
        if auto_resolved:
            print(f"  ✅ Auto-resolvable ({len(auto_resolved)}):")
            for m, status in auto_resolved:
                symbol = "✅" if status["outcome"] == "YES" else "❌"
                print(f"    {symbol} [{status['outcome']}] {status['question']}")
                print(f"       Model: {m['model_prob']:.0%}  |  "
                      f"Price: {status['yes_price']:.2f}  |  "
                      f"ID: {m['market_id']}")
                if not args.dry_run:
                    trade_logger.resolve_market(m["market_id"], status["outcome"])

            if args.dry_run:
                print(f"\n  (Dry run — remove --dry-run to save {len(auto_resolved)} resolutions)")
        else:
            print("  No markets auto-resolved yet — prices not snapped.")

        # ── Print needs manual review ──────────────────────────────────────────
        if needs_manual:
            print(f"\n  ⚠️  Expired but outcome unclear ({len(needs_manual)}) "
                  f"— resolve manually:")
            for m, status in needs_manual:
                print(f"    ID: {m['market_id']} | Ends: {status['end_date']} | "
                      f"Price: {status['yes_price']:.2f} | {status['question']}")
                print(f"    → python main.py --resolve {m['market_id']} YES|NO")

        # ── Print still open ───────────────────────────────────────────────────
        print(f"\n  ⏳ Still open: {len(still_open)} markets")

        print(f"\n  Summary: {len(auto_resolved)} resolved  |  "
              f"{len(needs_manual)} need manual  |  "
              f"{len(still_open)} still open")

    # ── Step 2: Calibration report ────────────────────────────────────────────
    cal = trade_logger.get_calibration()
    print("\n" + "═" * 55)
    print("📐 Calibration Report")
    print("═" * 55)

    if cal.get("num_resolved", 0) == 0:
        print(f"\n  {cal.get('message', 'No resolved data yet.')}")
        print("\n  💡 Markets need to expire and resolve before")
        print("     calibration can be calculated.")
        return

    brier  = cal["brier_score"]
    interp = cal["brier_interpretation"]

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

    # ── Step 3: Verdict ────────────────────────────────────────────────────────
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

    # ── Step 4: Portfolio summary ──────────────────────────────────────────────
    summary = trade_logger.get_summary()
    print(f"\n{'─' * 55}")
    print("📊 Portfolio Summary")
    print(f"  Decisions:      {summary['total_decisions']}")
    print(f"  Trades placed:  {summary['trades_placed']}")
    print(f"  Open positions: {summary.get('open_positions', 0)}")
    print(f"  Total staked:   ${summary['total_staked_usd']:.2f}")
    print(f"  Avg edge:       {summary['avg_edge']:.1%}")
    print(f"  Resolved:       {summary.get('markets_resolved', 0)}")
    print()


if __name__ == "__main__":
    main()