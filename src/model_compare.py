#!/usr/bin/env python3
"""
Model Quality Comparison for Polymarket Backtesting v2
─────────────────────────────────────────────────────────
Fixes from v1:
  - qwen3-32b 400 errors: adds {"thinking": {"type": "disabled"}}
  - Balanced YES/NO test set to avoid direction bias
  - Directional accuracy (Dir%) metric
  - Better verdict thresholds based on P-Std

Budget reminder (Groq free tier):
  llama-3.1-8b-instant          → 14,400 RPD  (high budget, too weak)
  llama-3.3-70b-versatile       →  1,000 RPD
  meta-llama/llama-4-scout-*    →  1,000 RPD
  meta-llama/llama-4-maverick-* →  1,000 RPD
  qwen/qwen3-32b                →  1,000 RPD  (60 RPM)
  moonshotai/kimi-k2-instruct   →  1,000 RPD  (60 RPM)

Usage:
    python model_compare.py                  # Compare top 4 models, 12 markets each
    python model_compare.py --markets 20
    python model_compare.py --model llama-3.3-70b-versatile --markets 20
"""

import argparse
import json
import logging
import os
import statistics
import time
from dataclasses import dataclass, field

import requests
from dotenv import load_dotenv

load_dotenv()

for lib in ["urllib3", "requests", "primp"]:
    logging.getLogger(lib).setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

GAMMA_API_BASE = "https://gamma-api.polymarket.com"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

CANDIDATE_MODELS = [
    "llama-3.3-70b-versatile",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "qwen/qwen3-32b",
]

# These models require thinking to be explicitly disabled for json_object mode
THINKING_MODELS = {
    "qwen/qwen3-32b",
    "qwen/qwen3-32b-preview",
    "qwen/qwen3-14b",
}

SYSTEM_PROMPT = """You are an expert prediction market analyst and superforecaster.
Estimate the TRUE probability of prediction market events.

Reason systematically:
1. UNDERSTAND — What exactly is being asked? What are the resolution criteria?
2. BASE RATE — Historical frequency of similar events.
3. EVIDENCE — What adjusts the base rate up or down?
4. ESTIMATE — Final calibrated probability.

Rules:
- Be calibrated: 70% means it happens ~70% of the time
- Be DECISIVE: commit to a view when you have evidence
- Do NOT cluster near 50% — if you're uncertain, say 35% or 65%, not 48% or 52%
- Use the full range: strong NO = 5-20%, moderate NO = 25-40%, toss-up = 45-55%, moderate YES = 60-75%, strong YES = 80-95%
- A well-known favourite (e.g. Brazil vs Serbia in group stage) should be 75-85%

Respond ONLY with valid JSON:
{
  "reasoning": "2-4 sentences referencing specific evidence or base rates",
  "estimated_probability": 0.65,
  "confidence": "low | medium | high",
  "action": "BET_YES | BET_NO | SKIP",
  "key_factors": ["factor1", "factor2"]
}"""


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
            "question": m.get("question", ""),
            "description": m.get("description", "")[:400],
            "outcome": outcome,
            "volume": vol,
            "end_date": m.get("endDate", "")[:10],
        }

        if outcome == "YES" and len(yes_markets) < target_each:
            yes_markets.append(entry)
        elif outcome == "NO" and len(no_markets) < target_each:
            no_markets.append(entry)

        if len(yes_markets) >= target_each and len(no_markets) >= target_each:
            break

    # Interleave for a balanced set
    result = []
    for pair in zip(yes_markets[:target_each], no_markets[:target_each]):
        result.extend(pair)
    return result[:n]


def build_payload(model: str, question: str, description: str, end_date: str) -> dict:
    user_prompt = (
        f"Analyze this prediction market:\n\n"
        f"Market Question: {question}\n"
        f"Resolution Date: {end_date}\n"
        f"Description: {description}\n\n"
        f"Estimate the true probability. Respond ONLY with valid JSON."
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 600,
        "response_format": {"type": "json_object"},
    }

    # Fix for qwen3: requires thinking disabled to use json_object
    if model in THINKING_MODELS:
        payload["thinking"] = {"type": "disabled"}

    return payload


def query_model(model: str, question: str, description: str, end_date: str, api_key: str) -> dict | None:
    payload = build_payload(model, question, description, end_date)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for attempt in range(3):
        try:
            resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)

            if resp.status_code == 429:
                wait = float(resp.headers.get("retry-after", 10))
                logger.warning(f"    429 — waiting {wait:.0f}s")
                time.sleep(wait + 1)
                continue

            if resp.status_code == 400:
                try:
                    err_msg = resp.json().get("error", {}).get("message", resp.text[:150])
                except Exception:
                    err_msg = resp.text[:150]
                logger.warning(f"    400 Bad Request: {err_msg}")
                return None  # Not retryable

            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]

            # Strip markdown fences
            text = content.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()

            start, end = text.find("{"), text.rfind("}") + 1
            if start == -1:
                raise ValueError("No JSON object found in response")

            data = json.loads(text[start:end])
            prob = float(data.get("estimated_probability", 0.5))
            if prob > 1:
                prob /= 100.0

            return {
                "probability": max(0.01, min(0.99, prob)),
                "confidence": data.get("confidence", "?"),
                "action": data.get("action", "SKIP"),
                "reasoning": data.get("reasoning", "")[:150],
            }

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"    Parse error (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
        except requests.RequestException as e:
            logger.warning(f"    Request error (attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)

    return None


@dataclass
class ModelStats:
    model: str
    n: int = 0
    failed: int = 0
    brier: float = 1.0
    bet_rate: float = 0.0
    prob_std: float = 0.0
    win_rate: float = 0.0
    total_bets: int = 0
    correct_bets: int = 0
    correct_direction: int = 0
    all_probs: list = field(default_factory=list)


def analyze_model(model: str, markets: list[dict], api_key: str, delay: float) -> ModelStats:
    logger.info(f"\n  ── {model} ──")
    s = ModelStats(model=model)
    brier_sum = 0.0
    bets = correct_bets = correct_dir = 0

    for i, m in enumerate(markets, 1):
        result = query_model(model, m["question"], m["description"], m["end_date"], api_key)
        time.sleep(delay)

        if result is None:
            s.failed += 1
            logger.warning(f"    [{i:>2}/{len(markets)}] FAILED")
            continue

        prob = result["probability"]
        actual_val = 1.0 if m["outcome"] == "YES" else 0.0
        brier_sum += (prob - actual_val) ** 2
        s.all_probs.append(prob)
        s.n += 1

        # Directional accuracy
        if (prob > 0.50) == (m["outcome"] == "YES"):
            correct_dir += 1

        # Bet threshold: 60%/40%
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
            f"Prob={prob:>3.0%} Actual={m['outcome']:<3} Bet={direction:<4} | "
            f"{m['question'][:50]}"
        )

    n = s.n
    s.brier = round(brier_sum / n, 4) if n > 0 else 1.0
    s.bet_rate = round(bets / n, 3) if n > 0 else 0.0
    s.prob_std = round(statistics.stdev(s.all_probs), 3) if len(s.all_probs) > 1 else 0.0
    s.total_bets = bets
    s.correct_bets = correct_bets
    s.win_rate = round(correct_bets / bets, 3) if bets > 0 else 0.0
    s.correct_direction = correct_dir
    return s


def print_report(all_stats: list[ModelStats]):
    valid = [s for s in all_stats if s.n > 0]
    failed = [s for s in all_stats if s.n == 0]

    print("\n" + "═" * 86)
    print("🔬  MODEL COMPARISON RESULTS")
    print("═" * 86)

    if failed:
        print(f"\n  ❌ Complete failures (0 results):")
        for s in failed:
            print(f"     {s.model}")

    if not valid:
        print("\n  No valid results.")
        return

    valid.sort(key=lambda x: x.brier)
    best = valid[0]

    print(f"\n  {'Model':<50} {'Brier':>7} {'Dir%':>5} {'Bet%':>6} {'P-Std':>7} {'W/L':>7}  Verdict")
    print("  " + "─" * 86)

    for s in valid:
        dir_pct = s.correct_direction / s.n if s.n > 0 else 0
        wl = f"{s.correct_bets}/{s.total_bets}"

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

        marker = "→" if s.model == best.model else " "
        print(
            f"  {marker} {s.model:<48} {s.brier:>7.4f} {dir_pct:>4.0%} "
            f"{s.bet_rate:>5.0%} {s.prob_std:>7.3f} {wl:>7}  {verdict}"
        )

    print(f"""
  Metrics guide:
    Brier  : 0.00=perfect  0.25=coin-flip  lower is better
    Dir%   : % of markets where model probability was on correct side of 50%
    Bet%   : % of markets where model had conviction to bet (prob >60% or <40%)
    P-Std  : Probability std dev. <0.06 = useless clustering near 50%
    W/L    : Won / total bets at the 60%/40% threshold""")

    print(f"\n{'─' * 86}")
    print(f"  🏆 Best: {best.model}  (Brier={best.brier:.4f}, P-Std={best.prob_std:.3f})")
    print(f"\n  Next steps:")
    print(f"  1. Add to .env:  GROQ_MODEL={best.model}")
    print(f"  2. Full backtest (uses ~200 RPD):")
    print(f"     python backtest_honest.py --markets 200")
    print("═" * 86 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--markets", type=int,   default=12)
    parser.add_argument("--model",   type=str,   default=None)
    parser.add_argument("--delay",   type=float, default=2.5)
    args = parser.parse_args()

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        print("❌ GROQ_API_KEY not set in .env")
        return

    models = [args.model] if args.model else CANDIDATE_MODELS
    print(f"\n💡 Budget: ~{args.markets * len(models)} RPD  ({args.markets} markets × {len(models)} models)\n")

    markets = fetch_test_markets(args.markets)
    if not markets:
        print("No markets found.")
        return

    yes_n = sum(1 for m in markets if m["outcome"] == "YES")
    print(f"Test set: {len(markets)} markets  ({yes_n} YES / {len(markets)-yes_n} NO)\n")
    for m in markets:
        print(f"  [{m['outcome']}] {m['question'][:68]}")

    all_stats = []
    for i, model in enumerate(models):
        stats = analyze_model(model, markets, api_key, delay=args.delay)
        all_stats.append(stats)
        if i < len(models) - 1:
            time.sleep(3)

    print_report(all_stats)


if __name__ == "__main__":
    main()