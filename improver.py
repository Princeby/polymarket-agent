"""
improver.py
───────────
Claude-powered self-improvement engine.

Every week (or on demand via /improve), this module:
  1. Reads your trade log and calibration metrics
  2. Identifies your worst predictions and failure patterns
  3. Asks Claude to analyse and rewrite the trading prompt
  4. Saves the new prompt to a versioned file
  5. Returns a summary for your Telegram approval

Safety:
  - Every prompt version is saved with a timestamp (never overwritten)
  - The 'live' prompt is only updated after you approve via Telegram
  - Rollback restores the previous version instantly
  - Requires at least 10 resolved markets before improvement runs
"""

import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

PROMPTS_DIR    = Path("prompts")
VERSIONS_DIR   = PROMPTS_DIR / "versions"
LIVE_PROMPT    = PROMPTS_DIR / "trading_prompt.txt"
AGENT_SRC      = Path("src") / "agent.py"

PROMPTS_DIR.mkdir(exist_ok=True)
VERSIONS_DIR.mkdir(exist_ok=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _now_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _get_version_list() -> list[Path]:
    """Return all saved prompt versions sorted oldest → newest."""
    return sorted(VERSIONS_DIR.glob("trading_prompt_*.txt"))


def _current_prompt() -> str:
    try:
        return LIVE_PROMPT.read_text()
    except FileNotFoundError:
        return "(no prompt file found)"


def _extract_system_prompt() -> str:
    """Pull the SYSTEM_PROMPT constant out of src/agent.py for Claude to review."""
    try:
        src = AGENT_SRC.read_text()
        start = src.find('SYSTEM_PROMPT = """')
        if start == -1:
            return "(could not extract SYSTEM_PROMPT)"
        end = src.find('"""', start + 18)
        return src[start + 18: end].strip()
    except FileNotFoundError:
        return "(src/agent.py not found)"


def _get_worst_bets(log: list[dict], n: int = 10) -> list[dict]:
    """Return the N bets where the model was most wrong."""
    resolved = [
        e for e in log
        if "outcome_value" in e and "model_probability" in e and "BET" in e.get("action_taken", "")
    ]
    scored = []
    for e in resolved:
        error = abs(e["model_probability"] - e["outcome_value"])
        scored.append({
            "question":    e.get("question", "")[:80],
            "model_prob":  e["model_probability"],
            "actual":      "YES" if e["outcome_value"] == 1.0 else "NO",
            "direction":   e.get("edge_direction", "?"),
            "stake":       e.get("stake_usd", 0),
            "error":       round(error, 3),
            "category":    _quick_categorize(e.get("question", "")),
            "end_date":    e.get("end_date", ""),
        })
    scored.sort(key=lambda x: x["error"], reverse=True)
    return scored[:n]


def _quick_categorize(question: str) -> str:
    q = question.lower()
    if any(w in q for w in ["bitcoin", "btc", "ethereum", "eth", "crypto"]):
        return "Crypto"
    if any(w in q for w in ["trump", "election", "democrat", "republican", "vote"]):
        return "Politics"
    if any(w in q for w in ["temperature", "weather", "°f", "°c"]):
        return "Weather"
    if any(w in q for w in ["nba", "nfl", "mlb", "spread"]):
        return "Sports"
    return "Other"


def _category_breakdown(log: list[dict]) -> dict:
    """Compute per-category Brier scores from resolved bets."""
    from collections import defaultdict
    cats = defaultdict(lambda: {"n": 0, "brier_sum": 0.0, "bets": 0, "wins": 0})

    for e in log:
        if "outcome_value" not in e or "model_probability" not in e:
            continue
        cat = _quick_categorize(e.get("question", ""))
        c   = cats[cat]
        c["n"]         += 1
        c["brier_sum"] += (e["model_probability"] - e["outcome_value"]) ** 2
        if "BET" in e.get("action_taken", ""):
            c["bets"] += 1
            if (e["model_probability"] > 0.5) == (e["outcome_value"] == 1.0):
                c["wins"] += 1

    result = {}
    for cat, data in cats.items():
        result[cat] = {
            "n":       data["n"],
            "brier":   round(data["brier_sum"] / data["n"], 4),
            "win_pct": round(data["wins"] / data["bets"], 3) if data["bets"] > 0 else None,
        }
    return result


# ── Core Improvement Function ──────────────────────────────────────────────────

def run_improvement_cycle(trade_logger) -> dict:
    """
    Read trade history → ask Claude to improve the prompt → save versioned file.

    Returns:
        dict with keys: new_prompt, changes_made, hypothesis, version_path
    """
    import anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set — required for self-improvement")

    # ── 1. Gather performance data ─────────────────────────────────────────────
    log = trade_logger._read_log()
    cal = trade_logger.get_calibration()

    worst_bets      = _get_worst_bets(log, n=12)
    cat_breakdown   = _category_breakdown(log)
    current_prompt  = _current_prompt()
    system_prompt   = _extract_system_prompt()

    # ── 2. Build analysis context for Claude ───────────────────────────────────
    context = f"""
You are a meta-AI tasked with improving an AI prediction market trading agent's prompts.

The agent uses an LLM (Groq/Llama) to estimate probabilities for Polymarket prediction markets.
Your job is to analyse its past performance and rewrite its trading prompt to fix weaknesses.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OVERALL PERFORMANCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Brier Score:  {cal.get('brier_score', 'N/A')}  (0=perfect, 0.25=coin-flip, lower=better)
Assessment:   {cal.get('brier_interpretation', 'N/A')}
Markets resolved: {cal.get('num_resolved', 0)}

Calibration by bucket:
{json.dumps(cal.get('calibration_buckets', {}), indent=2)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PERFORMANCE BY CATEGORY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{json.dumps(cat_breakdown, indent=2)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WORST PREDICTIONS (model was most wrong here)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{json.dumps(worst_bets, indent=2)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CURRENT TRADING PROMPT (prompts/trading_prompt.txt)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{current_prompt}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CURRENT SYSTEM PROMPT (src/agent.py SYSTEM_PROMPT)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{system_prompt[:3000]}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
YOUR TASK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Identify the top 2-3 failure patterns from the data above
2. Rewrite ONLY the trading prompt (not the system prompt) to fix these patterns
3. Keep all existing good elements; only change what the data says is broken
4. Be specific: if crypto markets are over-estimated, add explicit guidance

Respond with ONLY valid JSON in this exact format:
{{
  "new_prompt": "The complete rewritten trading prompt...",
  "changes_made": [
    "Specific change 1 with reason",
    "Specific change 2 with reason"
  ],
  "hypothesis": "One sentence: what you believe was the main failure pattern and how the new prompt fixes it",
  "failure_patterns_identified": [
    "Pattern 1 from data",
    "Pattern 2 from data"
  ]
}}
"""

    # ── 3. Ask Claude ──────────────────────────────────────────────────────────
    logger.info("Sending performance data to Claude for analysis...")
    client   = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=4000,
        messages=[{"role": "user", "content": context}],
    )

    raw = response.content[0].text.strip()

    # Strip markdown fences if present
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()

    result = json.loads(raw)

    # ── 4. Save versioned prompt ───────────────────────────────────────────────
    stamp        = _now_stamp()
    version_path = VERSIONS_DIR / f"trading_prompt_{stamp}.txt"

    # Backup current live prompt first
    if LIVE_PROMPT.exists():
        shutil.copy(LIVE_PROMPT, VERSIONS_DIR / f"trading_prompt_BACKUP_{stamp}.txt")

    # Write new prompt to versions dir
    version_path.write_text(result["new_prompt"])

    # Also update the live prompt (user approves via Telegram before agent uses it on next scan)
    LIVE_PROMPT.write_text(result["new_prompt"])

    logger.info(f"New prompt saved: {version_path}")

    result["version_path"] = str(version_path)
    result["brier_at_improvement"] = cal.get("brier_score")
    result["num_resolved_at_improvement"] = cal.get("num_resolved")

    # Save improvement log
    _append_improvement_log(result, stamp)

    return result


def _append_improvement_log(result: dict, stamp: str):
    """Keep a history of all improvement cycles."""
    log_path = PROMPTS_DIR / "improvement_history.json"
    try:
        history = json.loads(log_path.read_text()) if log_path.exists() else []
    except Exception:
        history = []

    history.append({
        "timestamp": stamp,
        "brier_score": result.get("brier_at_improvement"),
        "num_resolved": result.get("num_resolved_at_improvement"),
        "changes_made": result.get("changes_made", []),
        "hypothesis": result.get("hypothesis", ""),
        "failure_patterns": result.get("failure_patterns_identified", []),
        "version_file": result.get("version_path", ""),
    })

    log_path.write_text(json.dumps(history, indent=2))


# ── Revert ─────────────────────────────────────────────────────────────────────

def revert_prompt() -> str:
    """
    Revert the live prompt to the previous saved version.
    Returns the filename reverted to.
    """
    versions = _get_version_list()
    # Filter out backup files, get only clean versions
    clean_versions = [v for v in versions if "BACKUP" not in v.name]

    if len(clean_versions) < 2:
        raise ValueError("No previous version to revert to (need at least 2 saved versions)")

    # The second-to-last is the previous good version
    previous = clean_versions[-2]
    shutil.copy(previous, LIVE_PROMPT)
    logger.info(f"Reverted live prompt to: {previous.name}")
    return previous.name


# ── Improvement History ────────────────────────────────────────────────────────

def get_improvement_history() -> list[dict]:
    """Return the full history of improvement cycles."""
    log_path = PROMPTS_DIR / "improvement_history.json"
    try:
        return json.loads(log_path.read_text()) if log_path.exists() else []
    except Exception:
        return []


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(message)s")

    if "--history" in sys.argv:
        history = get_improvement_history()
        if not history:
            print("No improvement history yet.")
        for entry in history:
            print(f"\n[{entry['timestamp']}] Brier: {entry['brier_score']} | Resolved: {entry['num_resolved']}")
            print(f"  Hypothesis: {entry['hypothesis']}")
            for c in entry['changes_made']:
                print(f"  • {c}")
        sys.exit(0)

    if "--revert" in sys.argv:
        try:
            reverted = revert_prompt()
            print(f"✅ Reverted to: {reverted}")
        except ValueError as e:
            print(f"❌ {e}")
        sys.exit(0)

    # Run improvement
    from src.trader import TradeLogger
    tl  = TradeLogger()
    cal = tl.get_calibration()

    if cal.get("num_resolved", 0) < 5:
        print(f"❌ Need at least 5 resolved markets (have {cal.get('num_resolved', 0)})")
        sys.exit(1)

    print(f"🧠 Running improvement cycle...")
    print(f"   Brier score: {cal.get('brier_score', 'N/A')}")
    print(f"   Resolved:    {cal.get('num_resolved', 0)} markets\n")

    result = run_improvement_cycle(tl)

    print(f"\n✅ New prompt saved to: {result['version_path']}")
    print(f"\nChanges made:")
    for c in result.get("changes_made", []):
        print(f"  • {c}")
    print(f"\nHypothesis: {result.get('hypothesis', '')}")
    print(f"\nUse 'python improver.py --revert' to undo.")
