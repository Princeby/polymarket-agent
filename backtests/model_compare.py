#!/usr/bin/env python3
"""
Multi-Provider Model Comparison for Polymarket
───────────────────────────────────────────────
Compare Cerebras, Gemini, and Groq on the SAME resolved markets to find
which backend forecasts best before committing to production.

Usage (run from project root OR backtests/ folder):
    python backtests/model_compare.py                     # All 3 backends, 12 markets
    python backtests/model_compare.py --markets 20        # More markets
    python backtests/model_compare.py --backends groq gemini  # Subset only
"""

import argparse
import json
import logging
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests
from dotenv import load_dotenv

# ── Path fix: works whether run from root or from backtests/ ──────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
load_dotenv(_ROOT / ".env")
# ─────────────────────────────────────────────────────────────────────────────

from src.agent import (
    _create_backend,
    _BACKEND_REGISTRY,
    analyze_market,
    AnalysisResult,
    LLMBackend,
)
from src.market import Market

for lib in ["urllib3", "requests", "primp", "h2", "rustls", "hyper_util",
            "cookie_store", "ddgs"]:
    logging.getLogger(lib).setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

GAMMA_API_BASE = "https://gamma-api.polymarket.com"

# Backends to compare by default
DEFAULT_BACKENDS = ["cerebras", "gemini", "groq"]


# ── Data Fetching ──────────────────────────────────────────────────────────────

def fetch_test_markets(n: int = 12) -> list[dict]:
    """Fetch a balanced set of resolved markets (roughly equal YES/NO)."""
    params = {
        "closed": "true",
        "limit": 400,
        "order": "volume",
        "ascending": "false",
    }
    resp = requests.get(f"{GAMMA_API_BASE}/markets", params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    skip_patterns = [
        "up or down", "spread:", "o/u ", ": o/u ",
        'say "', "tweets from", "post 2", "post 1",
        "win the gold medal", "win the silver medal",
    ]

    yes_markets, no_markets = [], []
    target_each = (n + 1) // 2

    for m in data:
        prices = json.loads(m.get("outcomePrices", "[]"))
        if len(prices) < 2:
            continue
        yes_price = float(prices[0])
        vol = float(m.get("volume", 0))
        q = m.get("question", "").lower()

        if 0.05 < yes_price < 0.95:
            continue
        if vol < 5_000:
            continue
        if any(p in q for p in skip_patterns):
            continue
        if len(m.get("question", "")) < 20:
            continue

        outcome = "YES" if yes_price >= 0.95 else "NO"
        entry = {
            "id": str(m.get("id", "")),
            "question": m.get("question", ""),
            "description": m.get("description", "")[:400],
            "outcome": outcome,
            "volume": vol,
            "end_date": (m.get("endDateIso") or m.get("endDate", ""))[:10],
        }

        if outcome == "YES" and len(yes_markets) < target_each:
            yes_markets.append(entry)
        elif outcome == "NO" and len(no_markets) < target_each:
            no_markets.append(entry)

        if len(yes_markets) >= target_each and len(no_markets) >= target_each:
            break

    result = []
    for pair in zip(yes_markets[:target_each], no_markets[:target_each]):
        result.extend(pair)
    return result[:n]


# ── Per-Backend Stats ──────────────────────────────────────────────────────────

@dataclass
class ModelStats:
    backend_name: str
    n: int = 0
    failed: int = 0
    brier: float = 1.0
    bet_rate: float = 0.0
    prob_std: float = 0.0
    win_rate: float = 0.0
    total_bets: int = 0
    correct_bets: int = 0
    correct_direction: int = 0
    total_latency: float = 0.0
    all_probs: list = field(default_factory=list)
    market_results: list = field(default_factory=list)  # per-market detail


@dataclass
class MarketResult:
    question: str
    outcome: str
    probability: float
    direction: str
    correct: bool
    latency: float


# ── Analyze One Backend ───────────────────────────────────────────────────────

def analyze_with_backend(
    backend: LLMBackend,
    markets: list[dict],
    delay: float,
) -> ModelStats:
    """Run every market through a single backend and collect stats."""
    logger.info(f"\n  ── {backend.name} ──")
    s = ModelStats(backend_name=backend.name)
    brier_sum = 0.0
    bets = correct_bets = correct_dir = 0

    for i, m in enumerate(markets, 1):
        market = Market(
            id=m["id"],
            question=m["question"],
            description=m["description"],
            yes_price=0.50,
            no_price=0.50,
            volume=m["volume"],
            liquidity=0,
            end_date=m["end_date"],
            active=False,
            slug="",
        )

        t0 = time.time()
        analysis = analyze_market(market, backend, include_news=False)
        latency = time.time() - t0

        if analysis is None:
            s.failed += 1
            logger.warning(f"    [{i:>2}/{len(markets)}] FAILED  ({latency:.1f}s)")
            continue

        prob = analysis.estimated_probability
        actual_val = 1.0 if m["outcome"] == "YES" else 0.0
        brier_sum += (prob - actual_val) ** 2
        s.all_probs.append(prob)
        s.n += 1
        s.total_latency += latency

        if (prob > 0.50) == (m["outcome"] == "YES"):
            correct_dir += 1

        if prob > 0.60:
            direction = "YES"
            bets += 1
            if m["outcome"] == "YES":
                correct_bets += 1
        elif prob < 0.40:
            direction = "NO"
            bets += 1
            if m["outcome"] == "NO":
                correct_bets += 1
        else:
            direction = "SKIP"

        correct_call = (direction == m["outcome"]) or direction == "SKIP"
        icon = "✓" if correct_call else "✗"
        logger.info(
            f"    [{i:>2}/{len(markets)}] {icon} "
            f"Prob={prob:>3.0%} Actual={m['outcome']:<3} Bet={direction:<4} "
            f"({latency:.1f}s) | {m['question'][:50]}"
        )

        s.market_results.append(MarketResult(
            question=m["question"][:60],
            outcome=m["outcome"],
            probability=prob,
            direction=direction,
            correct=correct_call,
            latency=latency,
        ))

        time.sleep(delay)

    n = s.n
    s.brier = round(brier_sum / n, 4) if n > 0 else 1.0
    s.bet_rate = round(bets / n, 3) if n > 0 else 0.0
    s.prob_std = round(statistics.stdev(s.all_probs), 3) if len(s.all_probs) > 1 else 0.0
    s.total_bets = bets
    s.correct_bets = correct_bets
    s.win_rate = round(correct_bets / bets, 3) if bets > 0 else 0.0
    s.correct_direction = correct_dir
    return s


# ── Report ─────────────────────────────────────────────────────────────────────

def print_head_to_head(all_stats: list[ModelStats], markets: list[dict]):
    """Print a head-to-head grid showing each model's prediction per market."""
    valid = [s for s in all_stats if s.market_results]
    if not valid:
        return

    print(f"\n{'═' * 100}")
    print("📊  HEAD-TO-HEAD COMPARISON (per market)")
    print(f"{'═' * 100}")

    # Header
    header = f"  {'#':<3} {'Actual':<6} "
    for s in valid:
        short_name = s.backend_name.split("(")[0].strip()[:12]
        header += f" {short_name:^14}"
    header += f"  Question"
    print(header)
    print("  " + "─" * 96)

    max_rows = max(len(s.market_results) for s in valid)
    for i in range(max_rows):
        if i >= len(markets):
            break
        m = markets[i]
        row = f"  {i+1:<3} {m['outcome']:<6} "

        for s in valid:
            if i < len(s.market_results):
                mr = s.market_results[i]
                icon = "✓" if mr.correct else "✗"
                row += f" {icon} {mr.probability:>3.0%} → {mr.direction:<4} "
            else:
                row += f" {'FAIL':^14}"

        row += f"  {m['question'][:40]}"
        print(row)


def print_report(all_stats: list[ModelStats]):
    valid = [s for s in all_stats if s.n > 0]
    failed = [s for s in all_stats if s.n == 0]

    print(f"\n{'═' * 100}")
    print("🔬  MULTI-PROVIDER MODEL COMPARISON RESULTS")
    print(f"{'═' * 100}")

    if failed:
        print(f"\n  ❌ Complete failures (0 results):")
        for s in failed:
            print(f"     {s.backend_name}  (failed={s.failed})")

    if not valid:
        print("\n  No valid results.")
        return

    valid.sort(key=lambda x: x.brier)
    best = valid[0]

    print(
        f"\n  {'Backend':<30} {'Brier':>7} {'Dir%':>5} {'Bet%':>6} "
        f"{'P-Std':>7} {'W/L':>7} {'Avg-ms':>7}  Verdict"
    )
    print("  " + "─" * 96)

    for s in valid:
        dir_pct = s.correct_direction / s.n if s.n > 0 else 0
        wl = f"{s.correct_bets}/{s.total_bets}"
        avg_lat = (s.total_latency / s.n * 1000) if s.n > 0 else 0  # ms

        if s.brier < 0.18 and s.prob_std > 0.14 and s.bet_rate >= 0.20:
            verdict = "✅ EXCELLENT"
        elif s.brier < 0.22 and s.prob_std > 0.09:
            verdict = "✅ GOOD"
        elif s.prob_std < 0.06:
            verdict = "❌ Clusters near 50%"
        elif s.brier >= 0.25:
            verdict = "❌ No skill"
        else:
            verdict = "⚠️  Marginal"

        marker = "→" if s.backend_name == best.backend_name else " "
        print(
            f"  {marker} {s.backend_name:<28} {s.brier:>7.4f} {dir_pct:>4.0%} "
            f"{s.bet_rate:>5.0%} {s.prob_std:>7.3f} {wl:>7} {avg_lat:>6.0f}ms  {verdict}"
        )

    print(f"""
  Metrics guide:
    Brier  : 0.00=perfect  0.25=coin-flip  lower is better
    Dir%   : % of markets where model probability was on correct side of 50%
    Bet%   : % of markets where model had conviction to bet (prob >60% or <40%)
    P-Std  : Probability std dev. <0.06 = useless clustering near 50%
    W/L    : Won / total bets at the 60%/40% threshold
    Avg-ms : Average response latency in milliseconds""")

    print(f"\n{'─' * 100}")
    print(f"  🏆 Best: {best.backend_name}  (Brier={best.brier:.4f}, P-Std={best.prob_std:.3f})")

    if len(valid) > 1:
        worst = valid[-1]
        brier_diff = worst.brier - best.brier
        print(f"  📉 Gap : {brier_diff:+.4f} Brier between best and worst")

    print(f"\n  Next steps:")
    # Extract raw backend name from "Provider (model)" format
    best_raw = best.backend_name.split("(")[0].strip().lower()
    for name, cls in _BACKEND_REGISTRY.items():
        if name in best_raw.lower() or best_raw.lower() in name:
            best_raw = name
            break
    print(f"  1. Set primary:    LLM_BACKEND={best_raw}")
    if len(valid) > 1:
        second = valid[1]
        second_raw = second.backend_name.split("(")[0].strip().lower()
        for name, cls in _BACKEND_REGISTRY.items():
            if name in second_raw.lower() or second_raw.lower() in name:
                second_raw = name
                break
        print(f"  2. Set fallback:   LLM_FALLBACK={second_raw}")
    print(f"  3. Full backtest:  python backtests/backtest_2.py --markets 200")
    print("═" * 100 + "\n")


# ── Entrypoint ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare Cerebras, Gemini, and Groq on resolved Polymarket markets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available backends: {', '.join(_BACKEND_REGISTRY.keys())}

Examples:
  python backtests/model_compare.py                       # All 3 default backends
  python backtests/model_compare.py --markets 20          # More markets
  python backtests/model_compare.py --backends groq gemini  # Compare only 2
  python backtests/model_compare.py --delay 4             # Slower (rate-limit friendly)
        """
    )
    parser.add_argument("--markets",  type=int,   default=12,
                        help="Number of resolved markets to test (default: 12)")
    parser.add_argument("--backends", nargs="+",  default=DEFAULT_BACKENDS,
                        help=f"Backends to compare (default: {' '.join(DEFAULT_BACKENDS)})")
    parser.add_argument("--delay",    type=float, default=2.5,
                        help="Delay between API calls in seconds (default: 2.5)")
    args = parser.parse_args()

    # Validate backend names
    invalid = [b for b in args.backends if b.lower() not in _BACKEND_REGISTRY]
    if invalid:
        print(f"❌ Unknown backend(s): {invalid}")
        print(f"   Available: {', '.join(_BACKEND_REGISTRY.keys())}")
        return

    # Initialize backends (fail early if API keys are missing)
    backends: list[tuple[str, LLMBackend]] = []
    for name in args.backends:
        name = name.lower()
        try:
            backend = _create_backend(name)
            backends.append((name, backend))
            logger.info(f"✅ Initialized: {backend.name}")
        except ValueError as e:
            logger.warning(f"⚠️  Skipping {name}: {e}")

    if not backends:
        print("❌ No backends could be initialized. Check your API keys in .env")
        return

    total_calls = args.markets * len(backends)
    print(f"\n🔬 Multi-Provider Model Comparison")
    print(f"   Backends : {' vs '.join(b.name for _, b in backends)}")
    print(f"   Markets  : {args.markets}")
    print(f"   Total API calls: ~{total_calls}")
    print(f"   Est. time: ~{total_calls * (args.delay + 2) / 60:.0f} min\n")

    # Fetch markets once — same set for all backends
    markets = fetch_test_markets(args.markets)
    if not markets:
        print("No resolved markets found.")
        return

    yes_n = sum(1 for m in markets if m["outcome"] == "YES")
    print(f"  Test set: {len(markets)} markets  ({yes_n} YES / {len(markets)-yes_n} NO)\n")
    for m in markets:
        print(f"  [{m['outcome']}] {m['question'][:72]}")

    # Run each backend on the same markets
    all_stats = []
    for i, (name, backend) in enumerate(backends):
        stats = analyze_with_backend(backend, markets, delay=args.delay)
        all_stats.append(stats)

        if i < len(backends) - 1:
            pause = 5
            logger.info(f"\n  ⏳ Pausing {pause}s before next backend...\n")
            time.sleep(pause)

    # Print results
    print_head_to_head(all_stats, markets)
    print_report(all_stats)

    # Save results to JSON
    out_path = _ROOT / "data" / "model_comparison.json"
    out_path.parent.mkdir(exist_ok=True)
    results_json = []
    for s in all_stats:
        results_json.append({
            "backend": s.backend_name,
            "brier": s.brier,
            "direction_accuracy": s.correct_direction / s.n if s.n > 0 else 0,
            "bet_rate": s.bet_rate,
            "win_rate": s.win_rate,
            "prob_std": s.prob_std,
            "n": s.n,
            "failed": s.failed,
            "avg_latency_ms": (s.total_latency / s.n * 1000) if s.n > 0 else 0,
            "markets": [
                {
                    "question": mr.question,
                    "outcome": mr.outcome,
                    "probability": mr.probability,
                    "direction": mr.direction,
                    "correct": mr.correct,
                    "latency_s": round(mr.latency, 2),
                }
                for mr in s.market_results
            ],
        })

    with open(out_path, "w") as f:
        json.dump(results_json, f, indent=2)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()