"""
Polymarket Prediction Market Trading Agent
Main entry point — orchestrates the full agent loop.

Usage:
    python main.py                    # Run continuous loop
    python main.py --once             # Single pass, then exit
    python main.py --once --markets 3 # Analyze 3 markets, then exit
    python main.py --verbose          # Show full reasoning output
    python main.py --summary          # Print portfolio summary and exit
"""

import argparse
import logging
import os
import sys
import time

from dotenv import load_dotenv

from src.market import get_active_markets
from src.agent import get_backend, analyze_market, calculate_edge
from src.trader import kelly_stake, TradeLogger, execute_trade


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("primp").setLevel(logging.WARNING)
    logging.getLogger("h2").setLevel(logging.WARNING)
    logging.getLogger("rustls").setLevel(logging.WARNING)
    logging.getLogger("hyper_util").setLevel(logging.WARNING)
    logging.getLogger("cookie_store").setLevel(logging.WARNING)
    logging.getLogger("ddgs").setLevel(logging.WARNING)


def print_banner() -> None:
    print("""
╔══════════════════════════════════════════════════════╗
║       Polymarket Prediction Market Agent             ║
║       ─────────────────────────────────              ║
║       Discover → Reason → Size → Execute             ║
╚══════════════════════════════════════════════════════╝
""")


def run_cycle(
    backend,
    trade_logger: TradeLogger,
    max_markets: int,
    bankroll: float,
    dry_run: bool,
    verbose: bool,
    min_volume: float,
    min_liquidity: float,
) -> int:
    """
    Run one full cycle: fetch markets, analyze, decide, execute.
    Returns the number of trades placed (or would-be-placed in dry run).
    """
    logger = logging.getLogger(__name__)
    trades_placed = 0

    # ── 1. Fetch Markets ─────────────────────────────────────────────────
    logger.info("Fetching active markets from Polymarket...")
    markets = get_active_markets(
        limit=max_markets * 5,  # fetch extra, since we filter aggressively
        min_volume=min_volume,
        min_liquidity=min_liquidity,
    )

    if not markets:
        logger.warning("No markets found matching criteria.")
        return 0

    # Take only what we need
    markets = markets[:max_markets]
    logger.info(f"Analyzing {len(markets)} markets...")

    # ── 2. Analyze Each Market ───────────────────────────────────────────
    for i, market in enumerate(markets, 1):
        print(f"\n{'─' * 60}")
        print(f"Market {i}/{len(markets)}: {market.question[:70]}...")
        print(f"  Market price: YES={market.yes_price:.1%} NO={market.no_price:.1%}")

        # Get LLM analysis
        analysis = analyze_market(market, backend)
        if analysis is None:
            print("  ⚠ Analysis failed — skipping")
            trade_logger.log_decision(market, None, None, 0, "ANALYSIS_FAILED")
            continue

        # Calculate edge
        edge_info = calculate_edge(market, analysis)
        print(
            f"  Model estimate: {edge_info['model_probability']:.1%} | "
            f"Edge: {edge_info['raw_edge']:+.1%} ({edge_info['direction']})"
        )
        print(f"  Confidence: {analysis.confidence} | Action: {analysis.action}")

        if verbose:
            print(f"  Reasoning: {analysis.reasoning}")
            print(f"  Key factors: {', '.join(analysis.key_factors)}")

        # ── 3. Decide ────────────────────────────────────────────────────
        if not edge_info["has_edge"]:
            print(
                f"  ✗ Edge {edge_info['abs_edge']:.1%} below threshold "
                f"{edge_info['threshold']:.1%} — SKIP"
            )
            trade_logger.log_decision(market, analysis, edge_info, 0, "SKIPPED_LOW_EDGE")
            continue

        if analysis.confidence == "low":
            print("  ✗ Low confidence — SKIP")
            trade_logger.log_decision(market, analysis, edge_info, 0, "SKIPPED_LOW_CONFIDENCE")
            continue

        # ── 4. Size Position ─────────────────────────────────────────────
        stake = kelly_stake(
            bankroll=bankroll,
            edge=edge_info["abs_edge"],
            market_price=market.yes_price,
            direction=edge_info["direction"],
        )

        if stake <= 0:
            print("  ✗ Kelly sizing returned $0 — SKIP")
            trade_logger.log_decision(market, analysis, edge_info, 0, "SKIPPED_ZERO_STAKE")
            continue

        print(f"  ✓ Stake: ${stake:.2f} | Direction: {edge_info['direction']}")

        # ── 5. Execute ───────────────────────────────────────────────────
        action = execute_trade(market, edge_info["direction"], stake, dry_run=dry_run)
        trade_logger.log_decision(market, analysis, edge_info, stake, action)
        trades_placed += 1

    return trades_placed


def main() -> None:
    parser = argparse.ArgumentParser(description="Polymarket Prediction Market Agent")
    parser.add_argument("--once", action="store_true", help="Run a single pass then exit")
    parser.add_argument("--markets", type=int, default=10, help="Max markets to analyze per cycle")
    parser.add_argument("--verbose", action="store_true", help="Show full LLM reasoning")
    parser.add_argument("--summary", action="store_true", help="Print portfolio summary and exit")
    parser.add_argument("--calibration", action="store_true", help="Show Brier score and calibration report")
    parser.add_argument("--unresolved", action="store_true", help="List markets awaiting resolution")
    parser.add_argument("--resolve", nargs=2, metavar=("MARKET_ID", "OUTCOME"),
                        help="Resolve a market: --resolve 1234567 YES")
    args = parser.parse_args()

    # Load config
    load_dotenv()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # ── Resolve a market ─────────────────────────────────────────────────
    if args.resolve:
        trade_logger = TradeLogger()
        market_id, outcome = args.resolve
        success = trade_logger.resolve_market(market_id, outcome)
        if success:
            print(f"✅ Market {market_id} resolved as {outcome.upper()}")
        else:
            print(f"❌ No entries found for market ID {market_id}")
        return

    # ── Show unresolved markets ──────────────────────────────────────────
    if args.unresolved:
        trade_logger = TradeLogger()
        unresolved = trade_logger.get_unresolved_markets()
        if not unresolved:
            print("No unresolved markets with predictions.")
            return
        print(f"\n📋 Unresolved Markets ({len(unresolved)})")
        print("─" * 80)
        for m in unresolved:
            print(f"  ID: {m['market_id']} | Prob: {m['model_prob']:.1%} | "
                  f"Ends: {m['end_date']}")
            print(f"    {m['question']}")
        print(f"\nResolve with: python main.py --resolve <MARKET_ID> YES|NO")
        return

    # ── Show calibration report ──────────────────────────────────────────
    if args.calibration:
        trade_logger = TradeLogger()
        cal = trade_logger.get_calibration()
        print("\n📐 Calibration Report")
        print("─" * 50)
        if cal.get("num_resolved", 0) == 0:
            print(f"  {cal.get('message', 'No data')}")
            unresolved = trade_logger.get_unresolved_markets()
            if unresolved:
                print(f"\n  You have {len(unresolved)} markets awaiting resolution.")
                print(f"  Run: python main.py --unresolved")
            return
        print(f"  Brier Score:     {cal['brier_score']:.4f} ({cal['brier_interpretation']})")
        print(f"  Markets Scored:  {cal['num_resolved']}")
        print(f"\n  Calibration by bucket:")
        print(f"  {'Bucket':<12} {'Count':<7} {'Avg Forecast':<14} {'Avg Actual':<12} {'Gap':<6}")
        print(f"  {'─'*51}")
        for bucket, data in cal.get("calibration_buckets", {}).items():
            print(f"  {bucket:<12} {data['count']:<7} {data['avg_forecast']:<14.3f} "
                  f"{data['avg_actual']:<12.3f} {data['gap']:<6.3f}")
        return

    # Print summary and exit if requested
    if args.summary:
        trade_logger = TradeLogger()
        summary = trade_logger.get_summary()
        print("\n📊 Portfolio Summary")
        print("─" * 40)
        for k, v in summary.items():
            print(f"  {k}: {v}")
        return

    print_banner()

    # Config from .env
    dry_run = os.getenv("DRY_RUN", "true").lower() == "true"
    bankroll = float(os.getenv("BANKROLL", "100"))
    scan_interval = int(os.getenv("SCAN_INTERVAL_SECONDS", "300"))
    min_volume = float(os.getenv("MIN_VOLUME", "10000"))
    min_liquidity = float(os.getenv("MIN_LIQUIDITY", "1000"))

    mode_label = "🧪 DRY RUN" if dry_run else "🔴 LIVE TRADING"
    print(f"  Mode:     {mode_label}")
    print(f"  Bankroll: ${bankroll:.2f}")
    print(f"  Markets:  {args.markets} per cycle")
    print(f"  Interval: {scan_interval}s")

    # Init components
    try:
        backend = get_backend()
        print(f"  Backend:  {backend.name}")
    except ValueError as e:
        print(f"\n❌ {e}")
        sys.exit(1)

    trade_logger = TradeLogger()
    print()

    # ── Main Loop ────────────────────────────────────────────────────────
    cycle = 0
    while True:
        cycle += 1
        logger.info(f"═══ Cycle {cycle} ═══")

        try:
            trades = run_cycle(
                backend=backend,
                trade_logger=trade_logger,
                max_markets=args.markets,
                bankroll=bankroll,
                dry_run=dry_run,
                verbose=args.verbose,
                min_volume=min_volume,
                min_liquidity=min_liquidity,
            )
            logger.info(f"Cycle {cycle} complete — {trades} trade(s) placed")
        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.error(f"Cycle {cycle} failed: {e}", exc_info=True)

        if args.once:
            print("\n✅ Single pass complete. Exiting.")
            break

        logger.info(f"Sleeping {scan_interval}s until next cycle...")
        try:
            time.sleep(scan_interval)
        except KeyboardInterrupt:
            print("\n\n👋 Agent stopped. Run with --summary to see stats.")
            break


if __name__ == "__main__":
    main()
