"""
Polymarket Prediction Market Trading Agent
Main entry point — orchestrates the full agent loop.

Changes from v1:
  - Category filter: hard-blocks Politics/Geo and short-window crypto noise
  - Sports: soft-block (allowed but requires higher confidence + edge)
  - Pre-analysis filter runs BEFORE the LLM call to save API budget
  - Low-confidence bets now only blocked on high-variance categories
  - --no-news flag to disable news fetching (faster, for testing)
  - --block-sports flag to hard-block sports in addition to politics
  - Cycle summary now shows category breakdown of skips vs bets

Usage:
    python main.py                        # Continuous loop, dry run
    python main.py --once                 # Single pass
    python main.py --once --markets 5     # Single pass, 5 markets
    python main.py --verbose              # Show full LLM reasoning
    python main.py --once --no-news       # Disable news (faster, no signal boost)
    python main.py --summary              # Portfolio summary
    python main.py --calibration          # Brier score report
    python main.py --unresolved           # Markets awaiting resolution
    python main.py --resolve ID YES|NO    # Manually resolve a market
"""

import argparse
import logging
import os
import sys
import time
from collections import defaultdict

from dotenv import load_dotenv

from src.market import get_active_markets
from src.agent import get_backend, analyze_market, calculate_edge
from src.trader_two import kelly_stake, TradeLogger, execute_trade


# ── Logging ────────────────────────────────────────────────────────────────────

def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    for lib in ["urllib3", "requests", "primp", "h2", "rustls",
                "hyper_util", "cookie_store", "ddgs"]:
        logging.getLogger(lib).setLevel(logging.WARNING)


def print_banner() -> None:
    print("""
╔══════════════════════════════════════════════════════╗
║       Polymarket Prediction Market Agent             ║
║       ─────────────────────────────────              ║
║       Discover → Filter → Reason → Execute           ║
╚══════════════════════════════════════════════════════╝
""")


# ── Category Filter ────────────────────────────────────────────────────────────
#
# Based on 200-market backtest results:
#   Politics/Geo:  45% win, Brier 0.347  → HARD BLOCK
#   Short crypto:  ~50% win, Brier ~0.25 → HARD BLOCK (pure noise)
#   Sports:        62% win on 8 bets     → SOFT BLOCK (needs higher bar)
#
# The model fails on Politics because it applies stale training priors
# (e.g. "Russia won't advance", "deals are unlikely") with no real-time
# information. Even with news enabled, geopolitical prediction requires
# deep context the model doesn't reliably have.

# Hard blocked regardless of any flags — model is systematically wrong
HARD_BLOCK_PATTERNS = [
    # Short-window crypto Up/Down — pure random walk, zero model edge
    "up or down - ",        # "Bitcoin Up or Down - October 22, 12:00PM-4:00PM"
    "up or down – ",
    "up or down on ",       # "Ethereum Up or Down on January 3"
    # Sports spreads and O/U — model has ~50% win rate historically
    "spread:",
    " o/u ",
    ": o/u ",
    "o/u ",
]

# Politics/Geo — blocked by default, can enable with --allow-politics flag
POLITICS_PATTERNS = [
    "will russia ", "will ukraine", "capture territory",
    "will israel ", "will nato ", "will saudi",
    "normalize relations", "normalize relation",
    "will trump pardon", "margin of victory",
    "win new mexico", "win the primary", "win the election",
    "electoral college", "popular vote",
]

# Speech prediction markets — model has no real signal here
SPEECH_PATTERNS = [
    "will trump say ",
    "will biden say ",
    "will elon musk post ",
    "tweets from ",
    'say "',
    "will trump tweet",
    "will donald trump say",
]

# Sports — soft blocked by default (require --allow-sports to trade)
SPORTS_PATTERNS = [
    " vs. ",           # "Reds vs. Royals" style
    " vs ",
    "nba: ",
    "nfl: ",
    "will the suns ", "will the lakers ", "will the warriors ",
    "will the knicks ", "will the celtics ", "will the heat ",
    "will the 76ers", "will the raptors",
]


def categorize_market(question: str) -> str:
    """Quick category label for logging."""
    q = question.lower()
    if any(p in q for p in ["bitcoin", "btc", "ethereum", "eth", "xrp", "solana"]):
        if any(p in q for p in ["up or down", "above $", "below $", "between $"]):
            return "Crypto Price"
    if any(p in q for p in POLITICS_PATTERNS):
        return "Politics/Geo"
    if any(p in q for p in SPEECH_PATTERNS):
        return "Speech"
    if any(p in q for p in ["temperature", "°f", "°c", "highest temp", "weather"]):
        return "Weather"
    if any(p in q for p in ["nba", "nfl", "mlb", "spread:", " vs. "]):
        return "Sports"
    if any(p in q for p in ["counter-strike", "dota 2", "esports", "bo3", "ewc"]):
        return "Esports"
    return "Other"


def should_skip_market(
    question: str,
    allow_politics: bool = False,
    allow_sports: bool = False,
    allow_speech: bool = False,
) -> tuple[bool, str]:
    """
    Returns (should_skip, reason) based on category filters.

    Hard blocks always apply regardless of flags.
    Soft blocks (politics, sports, speech) can be overridden.

    Args:
        question:       Market question text.
        allow_politics: If True, don't block politics markets.
        allow_sports:   If True, don't block sports markets.
        allow_speech:   If True, don't block speech prediction markets.

    Returns:
        (True, reason_string) if market should be skipped.
        (False, "") if market should proceed to analysis.
    """
    q = question.lower()

    # Hard blocks — never trade these
    for pattern in HARD_BLOCK_PATTERNS:
        if pattern in q:
            return True, f"HARD_BLOCK:{pattern.strip().upper()}"

    # Politics — blocked by default
    if not allow_politics:
        for pattern in POLITICS_PATTERNS:
            if pattern in q:
                return True, "POLITICS_GEO"

    # Speech predictions — blocked by default
    if not allow_speech:
        for pattern in SPEECH_PATTERNS:
            if pattern in q:
                return True, "SPEECH_PREDICTION"

    # Sports — blocked by default
    if not allow_sports:
        for pattern in SPORTS_PATTERNS:
            if pattern in q:
                return True, "SPORTS"

    return False, ""


def requires_high_confidence(question: str) -> bool:
    """
    Returns True for categories where we require high (not just medium)
    confidence before placing a bet.

    Weather and niche "Other" markets are fine with medium confidence.
    Anything involving uncertain future events needs high confidence.
    """
    q = question.lower()
    uncertain_patterns = [
        "will ", "by ", "before ", "reach ", "hit ",
        "above ", "below ", "between ",
    ]
    # If it's a forward-looking binary question, require higher confidence
    return q.startswith("will ") or any(q.startswith(p) for p in ["will ", "is ", "does "])


# ── Main Cycle ─────────────────────────────────────────────────────────────────

def run_cycle(
    backend,
    trade_logger: TradeLogger,
    max_markets: int,
    bankroll: float,
    dry_run: bool,
    verbose: bool,
    min_volume: float,
    min_liquidity: float,
    allow_politics: bool,
    allow_sports: bool,
    allow_speech: bool,
    use_news: bool,
) -> dict:
    """
    Run one full cycle: fetch → pre-filter → analyze → size → execute.

    Returns a summary dict with counts for logging.
    """
    logger = logging.getLogger(__name__)
    counts = defaultdict(int)  # tracks skip reasons and bets

    # ── 1. Fetch Markets ───────────────────────────────────────────────────────
    logger.info("Fetching active markets from Polymarket...")
    markets = get_active_markets(
        limit=max_markets * 5,
        min_volume=min_volume,
        min_liquidity=min_liquidity,
    )

    if not markets:
        logger.warning("No markets found matching criteria.")
        return counts

    markets = markets[:max_markets]

    # ── 2. Load open positions once ────────────────────────────────────────────
    open_positions: set[str] = trade_logger.get_open_position_ids()
    if open_positions:
        logger.info(
            f"Open positions (will skip): "
            + ", ".join(list(open_positions)[:5])
            + (" ..." if len(open_positions) > 5 else "")
        )

    logger.info(f"Analyzing up to {len(markets)} markets...")

    # ── 3. Process Each Market ─────────────────────────────────────────────────
    for i, market in enumerate(markets, 1):
        print(f"\n{'─' * 62}")
        print(f"[{i}/{len(markets)}] {market.question[:68]}...")
        print(f"  Price: YES={market.yes_price:.1%}  NO={market.no_price:.1%}  "
              f"Vol=${market.volume:,.0f}")

        # ── Deduplication ──────────────────────────────────────────────────────
        if market.id in open_positions:
            print(f"  ⏭  Open position exists — SKIP")
            counts["skip_open_position"] += 1
            continue

        # ── Pre-analysis category filter ───────────────────────────────────────
        # This runs BEFORE the LLM call to save API budget on known-bad categories
        skip, reason = should_skip_market(
            market.question,
            allow_politics=allow_politics,
            allow_sports=allow_sports,
            allow_speech=allow_speech,
        )
        if skip:
            category = categorize_market(market.question)
            print(f"  ⊘  Category blocked [{category}] — SKIP")
            logger.debug(f"Blocked: {reason} | {market.question[:50]}")
            counts[f"skip_category_{reason}"] += 1
            trade_logger.log_decision(
                market, None, None, 0, f"SKIPPED_CATEGORY_{reason}"
            )
            continue

        # ── LLM Analysis ───────────────────────────────────────────────────────
        analysis = analyze_market(market, backend, include_news=use_news)
        if analysis is None:
            print("  ⚠  Analysis failed — SKIP")
            counts["skip_analysis_failed"] += 1
            trade_logger.log_decision(market, None, None, 0, "ANALYSIS_FAILED")
            continue

        # ── Edge Calculation ───────────────────────────────────────────────────
        edge_info = calculate_edge(market, analysis)
        print(
            f"  Model: {edge_info['model_probability']:.1%}  "
            f"Market: {edge_info['market_probability']:.1%}  "
            f"Edge: {edge_info['raw_edge']:+.1%} ({edge_info['direction']})"
        )
        print(f"  Confidence: {analysis.confidence}  |  Action: {analysis.action}")

        if verbose:
            print(f"  Reasoning: {analysis.reasoning}")
            if analysis.key_factors:
                print(f"  Factors: {', '.join(analysis.key_factors)}")

        # ── Edge Check ─────────────────────────────────────────────────────────
        if not edge_info["has_edge"]:
            print(
                f"  ✗ Edge {edge_info['abs_edge']:.1%} below "
                f"{edge_info['threshold']:.1%} threshold — SKIP"
            )
            counts["skip_low_edge"] += 1
            trade_logger.log_decision(
                market, analysis, edge_info, 0, "SKIPPED_LOW_EDGE"
            )
            continue

        # ── Confidence Check ───────────────────────────────────────────────────
        # Low confidence always skips.
        # Medium confidence skips on forward-looking binary questions
        # (where the model is more likely to be guessing).
        if analysis.confidence == "low":
            print("  ✗ Low confidence — SKIP")
            counts["skip_low_confidence"] += 1
            trade_logger.log_decision(
                market, analysis, edge_info, 0, "SKIPPED_LOW_CONFIDENCE"
            )
            continue

        if analysis.confidence == "medium" and requires_high_confidence(market.question):
            print("  ✗ Medium confidence on uncertain market — SKIP")
            counts["skip_medium_confidence"] += 1
            trade_logger.log_decision(
                market, analysis, edge_info, 0, "SKIPPED_MEDIUM_CONFIDENCE"
            )
            continue

        # ── Position Sizing ────────────────────────────────────────────────────
        stake = kelly_stake(
            bankroll=bankroll,
            edge=edge_info["abs_edge"],
            market_price=market.yes_price,
            direction=edge_info["direction"],
        )

        if stake <= 0:
            print("  ✗ Kelly sizing returned $0 — SKIP")
            counts["skip_zero_stake"] += 1
            trade_logger.log_decision(
                market, analysis, edge_info, 0, "SKIPPED_ZERO_STAKE"
            )
            continue

        print(f"  ✓ BET {edge_info['direction']}  |  Stake: ${stake:.2f}")

        # ── Execute ────────────────────────────────────────────────────────────
        action = execute_trade(
            market, edge_info["direction"], stake, dry_run=dry_run
        )
        trade_logger.log_decision(market, analysis, edge_info, stake, action)

        open_positions.add(market.id)
        counts["trades_placed"] += 1

    return counts


def print_cycle_summary(counts: dict, cycle: int) -> None:
    trades = counts.get("trades_placed", 0)
    cat_skips = sum(v for k, v in counts.items() if k.startswith("skip_category"))
    other_skips = sum(
        v for k, v in counts.items()
        if k.startswith("skip") and not k.startswith("skip_category")
    )
    logger = logging.getLogger(__name__)
    logger.info(
        f"Cycle {cycle} done — "
        f"bets: {trades}  "
        f"category_filtered: {cat_skips}  "
        f"other_skips: {other_skips}"
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Polymarket Prediction Market Agent")

    # Run modes
    parser.add_argument("--once",        action="store_true", help="Single pass then exit")
    parser.add_argument("--markets",     type=int, default=10, help="Max markets per cycle")
    parser.add_argument("--verbose",     action="store_true", help="Show full LLM reasoning")

    # Category controls
    parser.add_argument("--allow-politics", action="store_true",
                        help="Allow Politics/Geo markets (blocked by default — model performs poorly)")
    parser.add_argument("--allow-sports",   action="store_true",
                        help="Allow sports markets (blocked by default — insufficient edge)")
    parser.add_argument("--allow-speech",   action="store_true",
                        help="Allow speech prediction markets (blocked by default)")

    # News control
    parser.add_argument("--no-news", action="store_true",
                        help="Disable news fetching (faster cycles, loses real-time context)")

    # Info commands
    parser.add_argument("--summary",     action="store_true", help="Print portfolio summary and exit")
    parser.add_argument("--calibration", action="store_true", help="Show Brier score and calibration")
    parser.add_argument("--unresolved",  action="store_true", help="List markets awaiting resolution")
    parser.add_argument("--resolve",     nargs=2, metavar=("MARKET_ID", "OUTCOME"),
                        help="Resolve a market: --resolve <ID> YES|NO")
    parser.add_argument("--filters",     action="store_true",
                        help="Show active category filters and exit")

    args = parser.parse_args()

    load_dotenv()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # ── Info-only commands ─────────────────────────────────────────────────────

    if args.filters:
        print("\n📋 Active Category Filters")
        print("─" * 55)
        print("  HARD BLOCKED (always):")
        for p in HARD_BLOCK_PATTERNS:
            print(f"    • '{p.strip()}'")
        print(f"\n  SOFT BLOCKED by default (override with flags):")
        print(f"    Politics/Geo  (--allow-politics)  — Brier 0.347 in backtest")
        print(f"    Speech        (--allow-speech)    — model has no signal here")
        print(f"    Sports        (--allow-sports)    — 8-bet sample, needs more data")
        print(f"\n  NEWS: {'DISABLED (--no-news)' if args.no_news else 'ENABLED'}")
        print()
        return

    if args.resolve:
        trade_logger = TradeLogger()
        market_id, outcome = args.resolve
        if trade_logger.resolve_market(market_id, outcome):
            print(f"✅ Market {market_id} resolved as {outcome.upper()}")
        else:
            print(f"❌ No entries found for market ID {market_id}")
        return

    if args.unresolved:
        trade_logger = TradeLogger()
        unresolved = trade_logger.get_unresolved_markets()
        if not unresolved:
            print("No unresolved markets with predictions.")
            return
        print(f"\n📋 Unresolved Markets ({len(unresolved)})")
        print("─" * 80)
        for m in unresolved:
            print(
                f"  ID: {m['market_id']} | Prob: {m['model_prob']:.1%} | "
                f"Ends: {m['end_date']}"
            )
            print(f"  {m['question']}")
        print(f"\nResolve with: python main.py --resolve <ID> YES|NO")
        return

    if args.calibration:
        trade_logger = TradeLogger()
        cal = trade_logger.get_calibration()
        print("\n📐 Calibration Report")
        print("─" * 50)
        if cal.get("num_resolved", 0) == 0:
            print(f"  {cal.get('message', 'No data')}")
            unresolved = trade_logger.get_unresolved_markets()
            if unresolved:
                print(f"\n  {len(unresolved)} markets awaiting resolution.")
                print(f"  Run: python main.py --unresolved")
            return
        print(f"  Brier Score:    {cal['brier_score']:.4f} ({cal['brier_interpretation']})")
        print(f"  Markets Scored: {cal['num_resolved']}")
        print(f"\n  {'Bucket':<12} {'Count':<7} {'Forecast':<14} {'Actual':<12} {'Gap'}")
        print(f"  {'─'*51}")
        for bucket, data in cal.get("calibration_buckets", {}).items():
            bar = "●" * min(int(data["gap"] * 20), 10)
            print(
                f"  {bucket:<12} {data['count']:<7} {data['avg_forecast']:<14.1%} "
                f"{data['avg_actual']:<12.1%} {bar}"
            )
        return

    if args.summary:
        trade_logger = TradeLogger()
        summary = trade_logger.get_summary()
        print("\n📊 Portfolio Summary")
        print("─" * 40)
        for k, v in summary.items():
            print(f"  {k}: {v}")
        return

    # ── Agent startup ──────────────────────────────────────────────────────────

    print_banner()

    dry_run       = os.getenv("DRY_RUN", "true").lower() == "true"
    bankroll      = float(os.getenv("BANKROLL", "100"))
    scan_interval = int(os.getenv("SCAN_INTERVAL_SECONDS", "300"))
    min_volume    = float(os.getenv("MIN_VOLUME", "10000"))
    min_liquidity = float(os.getenv("MIN_LIQUIDITY", "1000"))
    use_news      = not args.no_news

    mode_label = "🧪 DRY RUN" if dry_run else "🔴 LIVE TRADING"
    print(f"  Mode:       {mode_label}")
    print(f"  Bankroll:   ${bankroll:.2f}")
    print(f"  Markets:    {args.markets} per cycle")
    print(f"  Interval:   {scan_interval}s")
    print(f"  News:       {'enabled ✓' if use_news else 'disabled (--no-news)'}")

    # Show which categories are active
    blocked = []
    if not args.allow_politics:
        blocked.append("Politics/Geo")
    if not args.allow_sports:
        blocked.append("Sports")
    if not args.allow_speech:
        blocked.append("Speech")
    blocked.append("Short-window crypto")  # always blocked
    blocked.append("Spreads/O/U")          # always blocked
    print(f"  Blocked:    {', '.join(blocked)}")

    try:
        backend = get_backend()
        print(f"  Model:      {backend.name}")
    except ValueError as e:
        print(f"\n❌ {e}")
        sys.exit(1)

    trade_logger = TradeLogger()
    open_count = len(trade_logger.get_open_position_ids())
    if open_count:
        print(f"  Positions:  {open_count} open (will be skipped)")

    print()

    # ── Main Loop ──────────────────────────────────────────────────────────────
    cycle = 0
    while True:
        cycle += 1
        logger.info(f"═══ Cycle {cycle} ═══")

        try:
            counts = run_cycle(
                backend=backend,
                trade_logger=trade_logger,
                max_markets=args.markets,
                bankroll=bankroll,
                dry_run=dry_run,
                verbose=args.verbose,
                min_volume=min_volume,
                min_liquidity=min_liquidity,
                allow_politics=args.allow_politics,
                allow_sports=args.allow_sports,
                allow_speech=args.allow_speech,
                use_news=use_news,
            )
            print_cycle_summary(counts, cycle)

        except KeyboardInterrupt:
            raise
        except Exception as e:
            logger.error(f"Cycle {cycle} failed: {e}", exc_info=True)

        if args.once:
            print("\n✅ Single pass complete.")
            print("   Run --summary to see portfolio stats.")
            print("   Run --calibration to see Brier score once markets resolve.")
            break

        logger.info(f"Sleeping {scan_interval}s until next cycle...")
        try:
            time.sleep(scan_interval)
        except KeyboardInterrupt:
            print("\n\n👋 Agent stopped.")
            print("   Run --summary for stats.")
            break


if __name__ == "__main__":
    main()