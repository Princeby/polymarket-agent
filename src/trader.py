"""
Trade Execution & Risk Management
Position sizing (Kelly criterion), dry-run execution, trade logging,
portfolio summary, and open position deduplication.
"""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.market import Market
from src.agent import AnalysisResult

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
TRADE_LOG_PATH = DATA_DIR / "trade_history.json"


# ── Position Sizing ────────────────────────────────────────────────────────────

def kelly_stake(
    bankroll: float,
    edge: float,
    market_price: float,
    direction: str,
    fraction: float = 0.25,
    max_pct: float = 0.10,
) -> float:
    """
    Calculate position size using fractional Kelly criterion for binary markets.

    In a binary prediction market:
      - Buying YES at price p: you pay p, win (1 - p) profit if correct
      - Buying NO at price (1-p): you pay (1-p), win p profit if correct

    Kelly formula for binary bets:
        f* = edge / cost_per_contract

    Args:
        bankroll:    Current total bankroll in USD.
        edge:        Absolute edge between model and market probability.
        market_price: Current YES price on the market (0 to 1).
        direction:   "YES" or "NO" — which side we're betting.
        fraction:    Kelly fraction (0.25 = quarter-Kelly, much safer).
        max_pct:     Maximum percentage of bankroll per trade (hard cap).

    Returns:
        Dollar amount to stake.
    """
    if edge <= 0 or bankroll <= 0:
        return 0.0

    cost = market_price if direction == "YES" else 1 - market_price
    if cost <= 0 or cost >= 1:
        return 0.0

    kelly_full = edge / cost
    stake = bankroll * kelly_full * fraction
    stake = min(stake, bankroll * max_pct)

    if stake < 5.0:  # Polymarket minimum
        return 0.0

    return round(stake, 2)


# ── Trade Logging ──────────────────────────────────────────────────────────────

class TradeLogger:
    """Logs every trade decision (including skips) to a JSON file."""

    def __init__(self, log_path: Path = TRADE_LOG_PATH):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            self.log_path.write_text("[]")

    def _read_log(self) -> list[dict]:
        try:
            return json.loads(self.log_path.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    # ── Position Deduplication ─────────────────────────────────────────────────

    def get_open_position_ids(self) -> set:
        """
        Return the set of market IDs where we have an active (unresolved) bet.

        A position is 'open' when:
          - An entry with action_taken containing 'BET' exists for that market, AND
          - No entry for that market has a 'resolved_outcome' yet.

        This prevents the agent from doubling into the same market across cycles.
        """
        log = self._read_log()
        bet_markets: set[str] = set()
        resolved_markets: set[str] = set()

        for entry in log:
            mid = entry.get("market_id", "")
            if not mid:
                continue
            if "BET" in entry.get("action_taken", ""):
                bet_markets.add(mid)
            if "resolved_outcome" in entry:
                resolved_markets.add(mid)

        # Open = we bet on it, but it hasn't resolved yet
        return bet_markets - resolved_markets

    def is_open_position(self, market_id: str) -> bool:
        """Return True if we already hold an unresolved bet on this market."""
        return market_id in self.get_open_position_ids()

    # ── Core Logging ───────────────────────────────────────────────────────────

    def log_decision(
        self,
        market: Market,
        analysis: Optional[AnalysisResult],
        edge_info: Optional[dict],
        stake: float,
        action_taken: str,
    ) -> None:
        """
        Append a trade decision to the log.

        Args:
            market:       The market that was analyzed.
            analysis:     LLM analysis result (None if analysis failed).
            edge_info:    Edge calculation dict (None if analysis failed).
            stake:        Dollar amount staked (0 if skipped).
            action_taken: What happened — "DRY_RUN_BET_YES", "SKIPPED", etc.
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "market_id": market.id,
            "question": market.question,
            "market_yes_price": market.yes_price,
            "end_date": market.end_date,
        }

        if analysis:
            entry.update({
                "model_probability": analysis.estimated_probability,
                "confidence": analysis.confidence,
                "reasoning": analysis.reasoning,
                "key_factors": analysis.key_factors,
            })

        if edge_info:
            entry.update({
                "raw_edge": edge_info["raw_edge"],
                "abs_edge": edge_info["abs_edge"],
                "edge_direction": edge_info["direction"],
                "has_edge": edge_info["has_edge"],
            })

        entry.update({
            "stake_usd": stake,
            "action_taken": action_taken,
        })

        log = self._read_log()
        log.append(entry)
        self.log_path.write_text(json.dumps(log, indent=2))
        logger.info(f"Logged: {action_taken} | {market.question[:50]}... | ${stake:.2f}")

    def resolve_market(self, market_id: str, outcome: str) -> bool:
        """
        Mark a market as resolved with YES or NO outcome.
        Updates all log entries for that market_id.

        Args:
            market_id: The Polymarket market ID.
            outcome:   "YES" or "NO".

        Returns:
            True if any entries were updated.
        """
        outcome = outcome.upper()
        if outcome not in ("YES", "NO"):
            logger.error(f"Invalid outcome '{outcome}' — must be YES or NO")
            return False

        log = self._read_log()
        updated = 0
        for entry in log:
            if entry.get("market_id") == market_id:
                entry["resolved_outcome"] = outcome
                entry["outcome_value"] = 1.0 if outcome == "YES" else 0.0
                updated += 1

        if updated > 0:
            self.log_path.write_text(json.dumps(log, indent=2))
            logger.info(f"Resolved market {market_id} as {outcome} ({updated} entries)")
        else:
            logger.warning(f"No entries found for market {market_id}")

        return updated > 0

    def get_calibration(self) -> dict:
        """
        Calculate Brier score and calibration metrics on resolved markets.

        Brier score = mean of (forecast - outcome)^2
          - 0.0  = perfect calibration
          - 0.25 = coin-flip baseline (no skill)
          - Higher = worse than random

        Returns:
            Dict with brier_score, num_resolved, calibration buckets, etc.
        """
        log = self._read_log()
        resolved = [
            e for e in log
            if "outcome_value" in e and "model_probability" in e
        ]

        if not resolved:
            return {
                "num_resolved": 0,
                "message": "No resolved markets yet. Run for 2-4 weeks, then resolve markets.",
            }

        brier_sum = sum(
            (e["model_probability"] - e["outcome_value"]) ** 2
            for e in resolved
        )
        brier_score = brier_sum / len(resolved)

        buckets = {}
        for e in resolved:
            bucket = int(e["model_probability"] * 10) * 10
            bucket = min(bucket, 90)
            if bucket not in buckets:
                buckets[bucket] = {"count": 0, "sum_forecast": 0.0, "sum_actual": 0.0}
            buckets[bucket]["count"] += 1
            buckets[bucket]["sum_forecast"] += e["model_probability"]
            buckets[bucket]["sum_actual"] += e["outcome_value"]

        calibration_table = {}
        for b in sorted(buckets.keys()):
            n = buckets[b]["count"]
            avg_forecast = buckets[b]["sum_forecast"] / n
            avg_actual = buckets[b]["sum_actual"] / n
            calibration_table[f"{b}-{b+10}%"] = {
                "count": n,
                "avg_forecast": round(avg_forecast, 3),
                "avg_actual": round(avg_actual, 3),
                "gap": round(abs(avg_forecast - avg_actual), 3),
            }

        return {
            "brier_score": round(brier_score, 4),
            "brier_interpretation": (
                "Excellent" if brier_score < 0.1 else
                "Good" if brier_score < 0.2 else
                "Baseline (no skill)" if brier_score < 0.26 else
                "Poor — worse than random"
            ),
            "num_resolved": len(resolved),
            "calibration_buckets": calibration_table,
        }

    def get_unresolved_markets(self) -> list[dict]:
        """Return unique markets that have predictions but no resolution yet."""
        log = self._read_log()
        seen = set()
        unresolved = []
        for e in log:
            mid = e.get("market_id", "")
            if mid and mid not in seen and "outcome_value" not in e:
                if "model_probability" in e:
                    seen.add(mid)
                    unresolved.append({
                        "market_id": mid,
                        "question": e.get("question", "")[:70],
                        "model_prob": e.get("model_probability"),
                        "end_date": e.get("end_date", ""),
                    })
        return unresolved

    def get_summary(self) -> dict:
        """Return portfolio summary stats from the trade log."""
        log = self._read_log()
        if not log:
            return {"total_decisions": 0, "message": "No trades logged yet."}

        trades = [e for e in log if "BET" in e.get("action_taken", "")]
        skips = [e for e in log if "SKIP" in e.get("action_taken", "")]
        resolved = [e for e in log if "outcome_value" in e]

        total_staked = sum(e.get("stake_usd", 0) for e in trades)
        avg_edge = (
            sum(e.get("abs_edge", 0) for e in trades) / len(trades)
            if trades else 0
        )

        summary = {
            "total_decisions": len(log),
            "trades_placed": len(trades),
            "trades_skipped": len(skips),
            "open_positions": len(self.get_open_position_ids()),
            "total_staked_usd": round(total_staked, 2),
            "avg_edge": round(avg_edge, 4),
            "markets_resolved": len(set(e["market_id"] for e in resolved)),
            "first_trade": log[0]["timestamp"] if log else None,
            "last_trade": log[-1]["timestamp"] if log else None,
        }

        cal = self.get_calibration()
        if cal.get("brier_score") is not None:
            summary["brier_score"] = cal["brier_score"]
            summary["brier_interpretation"] = cal["brier_interpretation"]

        return summary


# ── Trade Execution ────────────────────────────────────────────────────────────

def execute_trade(
    market: Market,
    direction: str,
    stake: float,
    dry_run: bool = True,
) -> str:
    """
    Execute a trade on Polymarket (or log it in dry-run mode).

    Args:
        market:    The market to trade on.
        direction: "YES" or "NO".
        stake:     Dollar amount to stake.
        dry_run:   If True, only log what would happen. If False, place real order.

    Returns:
        String describing what happened.
    """
    if dry_run:
        msg = (
            f"[DRY RUN] Would BET {direction} on: {market.question[:60]}...\n"
            f"  Stake: ${stake:.2f} | "
            f"Price: {market.yes_price if direction == 'YES' else market.no_price:.4f}"
        )
        logger.info(msg)
        return f"DRY_RUN_BET_{direction}"

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # REAL EXECUTION — Uncomment when ready to trade with real money
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    #
    # from py_clob_client.client import ClobClient
    # from py_clob_client.clob_types import OrderArgs
    #
    # client = ClobClient(
    #     host="https://clob.polymarket.com",
    #     key=os.getenv("POLYMARKET_API_KEY"),
    #     chain_id=137,  # Polygon mainnet
    # )
    #
    # token_ids = json.loads(market.raw.get("clobTokenIds", "[]"))
    # token_id = token_ids[0] if direction == "YES" else token_ids[1]
    #
    # order = client.create_order(
    #     OrderArgs(
    #         token_id=token_id,
    #         price=market.yes_price if direction == "YES" else market.no_price,
    #         size=stake / (market.yes_price if direction == "YES" else market.no_price),
    #         side="BUY",
    #     )
    # )
    # result = client.post_order(order)
    # logger.info(f"Order placed: {result}")
    # return f"LIVE_BET_{direction}"
    #
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    logger.warning("Live trading is not yet implemented. Set DRY_RUN=true.")
    return "NOT_IMPLEMENTED"