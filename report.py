#!/usr/bin/env python3
"""
Weekly Performance Report
─────────────────────────
Breaks down agent performance by week: bets placed, stakes,
model accuracy, P&L simulation, and trend analysis.

Usage:
    python report.py            # Full weekly breakdown
    python report.py --weeks 4  # Last 4 weeks only
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path(__file__).parent / "data"
TRADE_LOG = DATA_DIR / "trade_history.json"


def load_log() -> list[dict]:
    try:
        return json.loads(TRADE_LOG.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def week_key(timestamp_str: str) -> str:
    """Convert an ISO timestamp to a 'YYYY-WXX' week label."""
    dt = datetime.fromisoformat(timestamp_str)
    year, week, _ = dt.isocalendar()
    return f"{year}-W{week:02d}"


def week_start_date(week_str: str) -> str:
    """Convert 'YYYY-WXX' back to the Monday date for display."""
    year, w = week_str.split("-W")
    dt = datetime.strptime(f"{year} {w} 1", "%G %V %u")
    return dt.strftime("%b %d")


def compute_weekly_stats(log: list[dict]) -> dict:
    """Group trade log entries by week and compute stats."""
    weeks = defaultdict(lambda: {
        "decisions": 0,
        "bets": [],
        "skips": 0,
        "failures": 0,
        "total_stake": 0.0,
        "edges": [],
        "probabilities": [],
        "resolved": [],
        "directions": {"YES": 0, "NO": 0},
    })

    for entry in log:
        ts = entry.get("timestamp", "")
        if not ts:
            continue
        wk = week_key(ts)
        w = weeks[wk]
        w["decisions"] += 1

        action = entry.get("action_taken", "")
        stake = entry.get("stake_usd", 0)

        if "BET" in action:
            w["bets"].append(entry)
            w["total_stake"] += stake
            direction = entry.get("edge_direction", "")
            if direction in w["directions"]:
                w["directions"][direction] += 1
            if "abs_edge" in entry:
                w["edges"].append(entry["abs_edge"])
            if "model_probability" in entry:
                w["probabilities"].append(entry["model_probability"])
        elif "SKIP" in action:
            w["skips"] += 1
        elif "FAIL" in action:
            w["failures"] += 1

        # Track resolved outcomes
        if "outcome_value" in entry and "model_probability" in entry:
            w["resolved"].append({
                "forecast": entry["model_probability"],
                "actual": entry["outcome_value"],
                "stake": stake,
                "direction": entry.get("edge_direction", ""),
                "market_price": entry.get("market_yes_price", 0.5),
            })

    return dict(sorted(weeks.items()))


def simulate_pnl(resolved: list[dict]) -> dict:
    """
    Simulate P&L on resolved bets.
    YES bet wins if outcome=1, NO bet wins if outcome=0.
    Payout is (1/price_paid) * stake if win, lose stake if loss.
    """
    if not resolved:
        return {"wins": 0, "losses": 0, "pnl": 0.0, "roi": 0.0}

    wins = losses = 0
    total_pnl = 0.0
    total_staked = 0.0

    for r in resolved:
        stake = r["stake"]
        direction = r["direction"]
        actual = r["actual"]
        market_price = r["market_price"]

        if stake <= 0:
            continue

        total_staked += stake

        if direction == "YES":
            cost = market_price
            won = actual == 1.0
        else:
            cost = 1 - market_price
            won = actual == 0.0

        if won:
            profit = stake * (1 / cost - 1)  # payout minus cost
            total_pnl += profit
            wins += 1
        else:
            total_pnl -= stake
            losses += 1

    roi = (total_pnl / total_staked * 100) if total_staked > 0 else 0
    return {"wins": wins, "losses": losses, "pnl": round(total_pnl, 2), "roi": round(roi, 1)}


def brier_score(resolved: list[dict]) -> float | None:
    if not resolved:
        return None
    total = sum((r["forecast"] - r["actual"]) ** 2 for r in resolved)
    return round(total / len(resolved), 4)


def main():
    parser = argparse.ArgumentParser(description="Weekly performance report")
    parser.add_argument("--weeks", type=int, default=0, help="Show only last N weeks (0=all)")
    args = parser.parse_args()

    log = load_log()
    if not log:
        print("No trade history found. Run the agent first!")
        return

    weekly = compute_weekly_stats(log)

    if args.weeks > 0:
        keys = list(weekly.keys())[-args.weeks:]
        weekly = {k: weekly[k] for k in keys}

    # ── Header ───────────────────────────────────────────────────────────
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║              📊 Weekly Performance Report                   ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # ── Weekly Table ─────────────────────────────────────────────────────
    print(f"\n  {'Week':<10} {'Start':<8} {'Bets':<6} {'Skip':<6} {'Staked':>8} "
          f"{'Avg Edge':>9} {'YES/NO':<8} {'Brier':>7} {'P&L':>8}")
    print(f"  {'─'*72}")

    totals = {"bets": 0, "skips": 0, "stake": 0.0, "edges": [], "resolved": [], "decisions": 0}

    for wk, stats in weekly.items():
        n_bets = len(stats["bets"])
        avg_edge = sum(stats["edges"]) / len(stats["edges"]) if stats["edges"] else 0
        brier = brier_score(stats["resolved"])
        pnl_data = simulate_pnl(stats["resolved"])
        yes_no = f"{stats['directions']['YES']}/{stats['directions']['NO']}"
        start = week_start_date(wk)

        brier_str = f"{brier:.3f}" if brier is not None else "  —"
        pnl_str = f"${pnl_data['pnl']:+.2f}" if stats["resolved"] else "  —"

        print(f"  {wk:<10} {start:<8} {n_bets:<6} {stats['skips']:<6} "
              f"${stats['total_stake']:>7.2f} {avg_edge:>8.1%} {yes_no:<8} "
              f"{brier_str:>7} {pnl_str:>8}")

        totals["bets"] += n_bets
        totals["skips"] += stats["skips"]
        totals["stake"] += stats["total_stake"]
        totals["edges"].extend(stats["edges"])
        totals["resolved"].extend(stats["resolved"])
        totals["decisions"] += stats["decisions"]

    # ── Totals ───────────────────────────────────────────────────────────
    print(f"  {'─'*72}")
    total_avg_edge = sum(totals["edges"]) / len(totals["edges"]) if totals["edges"] else 0
    total_brier = brier_score(totals["resolved"])
    total_pnl = simulate_pnl(totals["resolved"])
    brier_str = f"{total_brier:.3f}" if total_brier is not None else "  —"
    pnl_str = f"${total_pnl['pnl']:+.2f}" if totals["resolved"] else "  —"

    print(f"  {'TOTAL':<10} {'':8} {totals['bets']:<6} {totals['skips']:<6} "
          f"${totals['stake']:>7.2f} {total_avg_edge:>8.1%} {'':8} "
          f"{brier_str:>7} {pnl_str:>8}")

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'─'*64}")
    print(f"  Total decisions:  {totals['decisions']}")
    print(f"  Total bets:       {totals['bets']}")
    print(f"  Total staked:     ${totals['stake']:.2f}")
    print(f"  Avg edge/bet:     {total_avg_edge:.1%}")

    if totals["resolved"]:
        print(f"\n  ── Resolved Bets ──")
        print(f"  Win/Loss:         {total_pnl['wins']}W / {total_pnl['losses']}L")
        print(f"  Net P&L:          ${total_pnl['pnl']:+.2f}")
        print(f"  ROI:              {total_pnl['roi']:+.1f}%")
        print(f"  Brier Score:      {total_brier:.4f}")

        if total_brier < 0.10:
            print(f"  Verdict:          🏆 Excellent — beating the market")
        elif total_brier < 0.20:
            print(f"  Verdict:          👍 Good signal — consider live trading")
        elif total_brier < 0.26:
            print(f"  Verdict:          ⚠️  Baseline — no better than market")
        else:
            print(f"  Verdict:          🚨 Poor — needs rework")
    else:
        print(f"\n  ⏳ No resolved bets yet. Markets need to expire first.")
        print(f"     Run: python resolve.py")

    print()


if __name__ == "__main__":
    main()
