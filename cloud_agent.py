"""
cloud_agent.py
──────────────
Cloud-hosted Polymarket agent with Telegram bot interface.
Runs on Railway using webhook mode (FastAPI + uvicorn).

Commands:
  /status        — Portfolio summary + open positions
  /scan          — Trigger immediate market scan
  /resolve       — List markets awaiting resolution
  /resolve ID YES|NO — Resolve a specific market
  /report        — Weekly P&L report
  /calibration   — Brier score & calibration table
  /pause         — Pause the agent
  /resume        — Resume the agent
  /improve       — Trigger Claude self-improvement cycle
  /approve       — Approve latest Claude prompt improvement
  /revert        — Revert to previous prompt
  /diff          — Show what Claude changed in last improvement
  /live on|off   — Enable/disable live trading (dangerous!)
  /help          — Show all commands
"""

import asyncio
import json
import logging
import os
import sys
import textwrap
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from telegram import (
    Bot,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
)

load_dotenv()

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
for lib in ["urllib3", "requests", "primp", "httpx", "apscheduler", "uvicorn.access"]:
    logging.getLogger(lib).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN       = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID     = os.getenv("TELEGRAM_CHAT_ID", "")
SCAN_INTERVAL_MIN    = int(os.getenv("SCAN_INTERVAL_MINUTES", "30"))
MAX_MARKETS_PER_SCAN = int(os.getenv("MAX_MARKETS_PER_SCAN", "10"))
BANKROLL             = float(os.getenv("BANKROLL", "100"))
MIN_VOLUME           = float(os.getenv("MIN_VOLUME", "10000"))
MIN_LIQUIDITY        = float(os.getenv("MIN_LIQUIDITY", "1000"))
# Railway injects PORT; fall back to 8080
PORT                 = int(os.getenv("PORT", "8080"))
# Full public URL of your Railway deployment, e.g. https://yourapp.railway.app
WEBHOOK_URL          = os.getenv("WEBHOOK_URL", "").rstrip("/")

DATA_DIR   = Path("data")
DATA_DIR.mkdir(exist_ok=True)
STATE_FILE = DATA_DIR / "agent_state.json"

# ── Agent State ────────────────────────────────────────────────────────────────

def load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {"paused": False, "dry_run": True, "pending_improvement": None}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))


state = load_state()

# ── Build Telegram Application (module-level so FastAPI lifespan can access it) ─
if not TELEGRAM_TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is not set — cannot start")

telegram_app: Application = Application.builder().token(TELEGRAM_TOKEN).build()


# ── Telegram helpers ───────────────────────────────────────────────────────────

async def notify(bot: Bot, text: str, parse_mode: str = "Markdown"):
    """Send a message to the owner's personal chat."""
    if not TELEGRAM_CHAT_ID:
        logger.warning("TELEGRAM_CHAT_ID not set — cannot send notification")
        return
    try:
        for chunk in _chunk_message(text):
            await bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=chunk,
                parse_mode=parse_mode,
            )
    except Exception as e:
        logger.error(f"Telegram send error: {e}")


def _chunk_message(text: str, limit: int = 4000) -> list[str]:
    if len(text) <= limit:
        return [text]
    lines = text.split("\n")
    chunks, current = [], ""
    for line in lines:
        if len(current) + len(line) + 1 > limit:
            chunks.append(current)
            current = line
        else:
            current += ("\n" if current else "") + line
    if current:
        chunks.append(current)
    return chunks


def _auth(update: Update) -> bool:
    """Only respond to the owner's chat."""
    if not TELEGRAM_CHAT_ID:
        return True
    return str(update.effective_chat.id) == str(TELEGRAM_CHAT_ID)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# ── Scheduled Jobs ─────────────────────────────────────────────────────────────

async def scan_job(bot: Bot):
    """Scheduled market scan — runs every SCAN_INTERVAL_MIN minutes."""
    if state.get("paused"):
        logger.info("Agent paused — skipping scan")
        return

    logger.info("⏰ Scheduled scan starting...")
    await notify(bot, "🔍 *Scheduled scan starting...*")

    try:
        from src.agent import get_backend, analyze_market, calculate_edge
        from src.market import get_active_markets
        from src.trader_two import kelly_stake, TradeLogger, execute_trade

        backend      = get_backend()
        trade_logger = TradeLogger()
        dry_run      = state.get("dry_run", True)

        markets = get_active_markets(
            limit=MAX_MARKETS_PER_SCAN * 5,
            min_volume=MIN_VOLUME,
            min_liquidity=MIN_LIQUIDITY,
        )[:MAX_MARKETS_PER_SCAN]

        open_positions = trade_logger.get_open_position_ids()
        bets_placed    = []
        skipped        = 0

        for market in markets:
            if market.id in open_positions:
                continue

            analysis = analyze_market(market, backend, include_news=True)
            if analysis is None:
                skipped += 1
                continue

            edge_info = calculate_edge(market, analysis)

            if not edge_info["has_edge"] or analysis.confidence == "low":
                skipped += 1
                trade_logger.log_decision(market, analysis, edge_info, 0, "SKIPPED_LOW_EDGE")
                continue

            stake = kelly_stake(
                bankroll=BANKROLL,
                edge=edge_info["abs_edge"],
                market_price=market.yes_price,
                direction=edge_info["direction"],
            )
            if stake <= 0:
                skipped += 1
                continue

            action = execute_trade(market, edge_info["direction"], stake, dry_run=dry_run)
            trade_logger.log_decision(market, analysis, edge_info, stake, action)
            open_positions.add(market.id)

            bets_placed.append({
                "question":   market.question[:55],
                "direction":  edge_info["direction"],
                "stake":      stake,
                "edge":       edge_info["abs_edge"],
                "prob":       analysis.estimated_probability,
                "confidence": analysis.confidence,
            })

        mode  = "🧪 DRY RUN" if dry_run else "🔴 LIVE"
        lines = [
            f"✅ *Scan Complete* `[{_now()}]` {mode}",
            "━━━━━━━━━━━━━━━━━━━━━━━━",
            f"Markets analyzed: `{len(markets)}`",
            f"Bets placed: `{len(bets_placed)}`  |  Skipped: `{skipped}`",
        ]

        if bets_placed:
            lines.append("")
            for b in bets_placed:
                icon = "🟢" if b["direction"] == "YES" else "🔴"
                lines.append(
                    f"{icon} *{b['direction']}* `${b['stake']:.2f}` "
                    f"| Edge `{b['edge']:.0%}` | Conf `{b['confidence']}`"
                )
                lines.append(f"  _{b['question']}_")
        else:
            lines.append("\n_No edge found — all markets skipped_")

        await notify(bot, "\n".join(lines))

    except Exception as e:
        logger.error(f"Scan job failed: {e}", exc_info=True)
        await notify(bot, f"❌ *Scan failed:* `{e}`")


async def resolve_job(bot: Bot):
    """Daily auto-resolution — runs at 9 am UTC."""
    logger.info("⏰ Daily resolution check starting...")
    await notify(bot, "🔎 *Daily resolution check...*")

    try:
        # FIX: was importing from src.trader (old file) — now uses src.trader_two consistently
        from src.trader_two import TradeLogger

        trade_logger = TradeLogger()
        unresolved   = trade_logger.get_unresolved_markets()

        if not unresolved:
            await notify(bot, "📭 No unresolved markets to check.")
            return

        auto_resolved = []
        needs_manual  = []

        for m in unresolved:
            status = _check_resolution(m["market_id"])
            if status["resolved"] and status["outcome"]:
                trade_logger.resolve_market(m["market_id"], status["outcome"])
                auto_resolved.append((m, status))
            elif status["needs_manual_review"]:
                needs_manual.append((m, status))

        lines = [f"📋 *Resolution Check* `[{_now()}]`", "━━━━━━━━━━━━━━━━━━━━━━━━"]

        if auto_resolved:
            lines.append(f"\n✅ *Auto-resolved ({len(auto_resolved)}):*")
            for m, s in auto_resolved:
                icon = "✅" if s["outcome"] == "YES" else "❌"
                lines.append(f"  {icon} `{s['outcome']}` — _{m['question'][:50]}_")

        if needs_manual:
            lines.append(f"\n⚠️ *Need manual resolution ({len(needs_manual)}):*")
            for m, s in needs_manual:
                lines.append(
                    f"  `/resolve {m['market_id']} YES` or `/resolve {m['market_id']} NO`"
                )
                lines.append(f"  _{m['question'][:50]}_")

        still_open = len(unresolved) - len(auto_resolved) - len(needs_manual)
        lines.append(
            f"\n_Auto: {len(auto_resolved)} | Manual needed: {len(needs_manual)} | Still open: {still_open}_"
        )
        await notify(bot, "\n".join(lines))

    except Exception as e:
        logger.error(f"Resolve job failed: {e}", exc_info=True)
        await notify(bot, f"❌ *Resolution check failed:* `{e}`")


async def improve_job(bot: Bot):
    """Weekly Claude self-improvement — runs Monday 8 am UTC."""
    from src.trader_two import TradeLogger

    trade_logger = TradeLogger()
    cal = trade_logger.get_calibration()

    if cal.get("num_resolved", 0) < 10:
        logger.info(
            f"Skipping improvement — only {cal.get('num_resolved', 0)} resolved markets (need 10+)"
        )
        return

    await notify(bot, "🧠 *Weekly improvement cycle starting...*\n_Analysing past performance..._")

    try:
        from improver import run_improvement_cycle

        result = run_improvement_cycle(trade_logger)

        s = load_state()
        s["pending_improvement"] = {
            "timestamp":    _now(),
            "changes":      result.get("changes_made", []),
            "hypothesis":   result.get("hypothesis", ""),
            "brier_before": cal.get("brier_score"),
        }
        save_state(s)

        keyboard = InlineKeyboardMarkup([[
            InlineKeyboardButton("✅ Approve", callback_data="approve_improvement"),
            InlineKeyboardButton("❌ Revert",  callback_data="revert_improvement"),
        ]])

        changes_text = "\n".join(f"• {c}" for c in result.get("changes_made", []))
        msg = (
            f"🧠 *Claude Improvement Ready*\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Brier before: `{cal.get('brier_score', '?'):.4f}`\n\n"
            f"*Changes proposed:*\n{changes_text}\n\n"
            f"*Hypothesis:* _{result.get('hypothesis', '')}_\n\n"
            f"Prompt saved to `prompts/versions/` — approve to deploy live."
        )

        if TELEGRAM_CHAT_ID:
            # FIX: was creating a new Bot() instead of reusing the one passed in
            await bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=msg,
                parse_mode="Markdown",
                reply_markup=keyboard,
            )

    except Exception as e:
        logger.error(f"Improvement job failed: {e}", exc_info=True)
        await notify(bot, f"❌ *Improvement cycle failed:* `{e}`")


# ── Command Handlers ───────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _auth(update):
        return
    s = load_state()
    mode   = "🔴 LIVE" if not s.get("dry_run", True) else "🧪 DRY RUN"
    paused = "⏸ PAUSED" if s.get("paused") else "▶️ RUNNING"
    await update.message.reply_text(
        f"👋 *Polymarket Agent Online*\n\n"
        f"Status: {paused}  |  Mode: {mode}\n\n"
        f"Use /help to see all commands.",
        parse_mode="Markdown",
    )


async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _auth(update):
        return
    text = textwrap.dedent("""
        📖 *Polymarket Agent Commands*
        ━━━━━━━━━━━━━━━━━━━━━━━━
        `/status`       — Portfolio summary
        `/scan`         — Run market scan now
        `/resolve`      — List unresolved markets
        `/resolve ID YES\\|NO` — Resolve a market
        `/report`       — Weekly P&L breakdown
        `/calibration`  — Brier score report
        `/pause`        — Pause the agent
        `/resume`       — Resume the agent
        `/improve`      — Trigger Claude improvement
        `/approve`      — Approve last improvement
        `/revert`       — Revert last improvement
        `/diff`         — Show prompt changes
        `/live on\\|off` — Toggle live trading ⚠️
        `/help`         — This message
    """).strip()
    await update.message.reply_text(text, parse_mode="Markdown")


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _auth(update):
        return
    try:
        from src.trader_two import TradeLogger

        tl      = TradeLogger()
        summary = tl.get_summary()
        s       = load_state()

        mode   = "🔴 LIVE" if not s.get("dry_run") else "🧪 DRY RUN"
        paused = "⏸ PAUSED" if s.get("paused") else "▶️ RUNNING"

        lines = [
            f"📊 *Portfolio Status* `[{_now()}]`",
            "━━━━━━━━━━━━━━━━━━━━━━━━",
            f"Mode: {mode}  |  {paused}",
            "",
            f"Total decisions: `{summary.get('total_decisions', 0)}`",
            f"Trades placed:  `{summary.get('trades_placed', 0)}`",
            f"Open positions: `{summary.get('open_positions', 0)}`",
            f"Total staked:   `${summary.get('total_staked_usd', 0):.2f}`",
            f"Avg edge:       `{summary.get('avg_edge', 0):.1%}`",
            f"Resolved:       `{summary.get('markets_resolved', 0)}`",
        ]

        if "brier_score" in summary:
            lines.append(
                f"Brier score:    `{summary['brier_score']:.4f}` "
                f"— _{summary.get('brier_interpretation', '')}_"
            )

        if summary.get("last_trade"):
            lines.append(f"\nLast trade: `{summary['last_trade'][:16]}`")

        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    except Exception as e:
        await update.message.reply_text(f"❌ Error: `{e}`", parse_mode="Markdown")


async def cmd_scan(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _auth(update):
        return
    if state.get("paused"):
        await update.message.reply_text("⏸ Agent is paused. Use /resume first.")
        return
    await update.message.reply_text("🔍 Triggering scan now...")
    asyncio.create_task(scan_job(context.bot))


async def cmd_resolve(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _auth(update):
        return
    from src.trader_two import TradeLogger

    tl   = TradeLogger()
    args = context.args

    if len(args) == 2:
        market_id, outcome = args[0], args[1].upper()
        if outcome not in ("YES", "NO"):
            await update.message.reply_text("❌ Outcome must be YES or NO")
            return
        success = tl.resolve_market(market_id, outcome)
        msg = (
            f"✅ Market `{market_id}` resolved as *{outcome}*"
            if success
            else f"❌ Market `{market_id}` not found in log"
        )
        await update.message.reply_text(msg, parse_mode="Markdown")
        return

    unresolved = tl.get_unresolved_markets()
    if not unresolved:
        await update.message.reply_text("✅ No markets awaiting resolution!")
        return

    lines = [f"⏳ *Unresolved Markets ({len(unresolved)})*", "━━━━━━━━━━━━━━━━━━━━━━━━"]
    for m in unresolved[:15]:
        lines.append(
            f"\n`{m['market_id']}`\n"
            f"_{m['question']}_\n"
            f"Prob: `{m['model_prob']:.0%}` | Ends: `{m['end_date']}`\n"
            f"→ `/resolve {m['market_id']} YES` or `NO`"
        )
    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def cmd_report(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _auth(update):
        return
    try:
        # FIX: avoid subprocess — call report logic directly
        import subprocess
        result = subprocess.run(
            [sys.executable, "report.py"],
            capture_output=True, text=True, timeout=30,
        )
        output = result.stdout or result.stderr or "No output"
        await update.message.reply_text(f"```\n{output[:3800]}\n```", parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"❌ Report failed: `{e}`", parse_mode="Markdown")


async def cmd_calibration(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _auth(update):
        return
    try:
        from src.trader_two import TradeLogger

        cal = TradeLogger().get_calibration()

        if cal.get("num_resolved", 0) == 0:
            await update.message.reply_text(
                "📐 *No resolved markets yet*\n\n"
                "_Run the agent for a few weeks, then use /resolve to score markets._",
                parse_mode="Markdown",
            )
            return

        brier   = cal["brier_score"]
        interp  = cal["brier_interpretation"]
        verdict = "🏆" if brier < 0.10 else "👍" if brier < 0.20 else "⚠️" if brier < 0.26 else "🚨"

        lines = [
            "📐 *Calibration Report*",
            "━━━━━━━━━━━━━━━━━━━━━━━━",
            f"Brier Score: `{brier:.4f}` {verdict}",
            f"Assessment:  _{interp}_",
            f"Markets:     `{cal['num_resolved']}`",
            "",
            f"{'Bucket':<12} {'N':>4} {'Forecast':>9} {'Actual':>8} {'Gap':>6}",
            f"`{'─'*44}`",
        ]
        for bucket, data in cal.get("calibration_buckets", {}).items():
            gap_bar = "●" * min(int(data["gap"] * 20), 8)
            lines.append(
                f"`{bucket:<12} {data['count']:>4} {data['avg_forecast']:>8.0%}"
                f" {data['avg_actual']:>8.0%}` {gap_bar}"
            )

        await update.message.reply_text("\n".join(lines), parse_mode="Markdown")

    except Exception as e:
        await update.message.reply_text(f"❌ Error: `{e}`", parse_mode="Markdown")


async def cmd_pause(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _auth(update):
        return
    s = load_state()
    s["paused"] = True
    save_state(s)
    state["paused"] = True
    await update.message.reply_text(
        "⏸ *Agent paused.* No scans will run.\nUse /resume to restart.",
        parse_mode="Markdown",
    )


async def cmd_resume(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _auth(update):
        return
    s = load_state()
    s["paused"] = False
    save_state(s)
    state["paused"] = False
    await update.message.reply_text(
        "▶️ *Agent resumed.* Scans will run on schedule.",
        parse_mode="Markdown",
    )


async def cmd_improve(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _auth(update):
        return
    await update.message.reply_text(
        "🧠 *Triggering Claude improvement cycle...*\n_This may take 30-60 seconds._",
        parse_mode="Markdown",
    )
    asyncio.create_task(improve_job(context.bot))


async def cmd_approve(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _auth(update):
        return
    s = load_state()
    if not s.get("pending_improvement"):
        await update.message.reply_text("ℹ️ No pending improvement to approve.")
        return
    s["pending_improvement"] = None
    save_state(s)
    await update.message.reply_text(
        "✅ *Improvement approved and deployed!*\n"
        "The agent will use the new prompt on the next scan.",
        parse_mode="Markdown",
    )


async def cmd_revert(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _auth(update):
        return
    try:
        from improver import revert_prompt

        reverted_to = revert_prompt()
        s = load_state()
        s["pending_improvement"] = None
        save_state(s)
        await update.message.reply_text(
            f"↩️ *Reverted to:* `{reverted_to}`",
            parse_mode="Markdown",
        )
    except Exception as e:
        await update.message.reply_text(f"❌ Revert failed: `{e}`", parse_mode="Markdown")


async def cmd_diff(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _auth(update):
        return
    s       = load_state()
    pending = s.get("pending_improvement")
    if not pending:
        await update.message.reply_text(
            "ℹ️ No pending improvement. Use /improve to generate one."
        )
        return

    changes = "\n".join(f"• {c}" for c in pending.get("changes", []))
    msg = (
        f"📝 *Pending Improvement* `[{pending.get('timestamp', '?')}]`\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Brier before: `{pending.get('brier_before', '?')}`\n\n"
        f"*Changes:*\n{changes}\n\n"
        f"*Hypothesis:* _{pending.get('hypothesis', '')}_\n\n"
        f"Use /approve or /revert"
    )
    await update.message.reply_text(msg, parse_mode="Markdown")


async def cmd_live(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not _auth(update):
        return
    args = context.args
    if not args or args[0].lower() not in ("on", "off"):
        s       = load_state()
        current = "OFF (dry run)" if s.get("dry_run", True) else "ON (live trading!)"
        await update.message.reply_text(
            f"⚙️ Live trading is currently: *{current}*\n\nUse `/live on` or `/live off`",
            parse_mode="Markdown",
        )
        return

    if args[0].lower() == "on":
        keyboard = InlineKeyboardMarkup([[
            InlineKeyboardButton("⚠️ YES, enable live trading", callback_data="live_confirm_on"),
            InlineKeyboardButton("Cancel", callback_data="live_cancel"),
        ]])
        await update.message.reply_text(
            "⚠️ *WARNING: This will place REAL orders with REAL money!*\n\nAre you sure?",
            parse_mode="Markdown",
            reply_markup=keyboard,
        )
    else:
        s = load_state()
        s["dry_run"] = True
        save_state(s)
        state["dry_run"] = True
        await update.message.reply_text(
            "🧪 *Live trading OFF.* Agent is back in dry-run mode.",
            parse_mode="Markdown",
        )


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    data  = query.data

    if data == "approve_improvement":
        s = load_state()
        s["pending_improvement"] = None
        save_state(s)
        await query.edit_message_text("✅ *Improvement approved and deployed!*", parse_mode="Markdown")

    elif data == "revert_improvement":
        try:
            from improver import revert_prompt

            reverted_to = revert_prompt()
            s = load_state()
            s["pending_improvement"] = None
            save_state(s)
            await query.edit_message_text(
                f"↩️ *Reverted to:* `{reverted_to}`", parse_mode="Markdown"
            )
        except Exception as e:
            await query.edit_message_text(f"❌ Revert failed: `{e}`", parse_mode="Markdown")

    elif data == "live_confirm_on":
        s = load_state()
        s["dry_run"] = False
        save_state(s)
        state["dry_run"] = False
        await query.edit_message_text(
            "🔴 *LIVE TRADING ENABLED.*\nAgent will place real orders. Use `/live off` to disable.",
            parse_mode="Markdown",
        )

    elif data == "live_cancel":
        await query.edit_message_text("✅ Cancelled. Agent remains in dry-run mode.")


# ── Register all handlers ──────────────────────────────────────────────────────

def _register_handlers(app: Application) -> None:
    app.add_handler(CommandHandler("start",       cmd_start))
    app.add_handler(CommandHandler("help",        cmd_help))
    app.add_handler(CommandHandler("status",      cmd_status))
    app.add_handler(CommandHandler("scan",        cmd_scan))
    app.add_handler(CommandHandler("resolve",     cmd_resolve))
    app.add_handler(CommandHandler("report",      cmd_report))
    app.add_handler(CommandHandler("calibration", cmd_calibration))
    app.add_handler(CommandHandler("pause",       cmd_pause))
    app.add_handler(CommandHandler("resume",      cmd_resume))
    app.add_handler(CommandHandler("improve",     cmd_improve))
    app.add_handler(CommandHandler("approve",     cmd_approve))
    app.add_handler(CommandHandler("revert",      cmd_revert))
    app.add_handler(CommandHandler("diff",        cmd_diff))
    app.add_handler(CommandHandler("live",        cmd_live))
    app.add_handler(CallbackQueryHandler(button_callback))


# ── Resolution helper ──────────────────────────────────────────────────────────

def _check_resolution(market_id: str) -> dict:
    import requests as _req

    try:
        resp = _req.get(
            f"https://gamma-api.polymarket.com/markets/{market_id}", timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return {"resolved": False, "outcome": None, "needs_manual_review": False}

    resolution = data.get("resolution")
    end_date   = data.get("endDateIso", data.get("endDate", ""))[:10]
    closed     = data.get("closed", False)
    active     = data.get("active", True)

    try:
        prices    = json.loads(data.get("outcomePrices", "[]"))
        yes_price = float(prices[0]) if prices else 0.5
    except Exception:
        yes_price = 0.5

    outcome, is_resolved, needs_manual = None, False, False

    if resolution:
        r = resolution.lower().strip()
        if r in ("yes", "1", "true"):
            is_resolved, outcome = True, "YES"
        elif r in ("no", "0", "false"):
            is_resolved, outcome = True, "NO"

    if not is_resolved:
        if yes_price >= 0.95:
            is_resolved, outcome = True, "YES"
        elif yes_price <= 0.05:
            is_resolved, outcome = True, "NO"

    if not is_resolved and end_date:
        try:
            end_dt = datetime.fromisoformat(end_date).replace(tzinfo=timezone.utc)
            if end_dt < datetime.now(timezone.utc) and (closed or not active):
                needs_manual = True
        except ValueError:
            pass

    return {
        "resolved":            is_resolved,
        "outcome":             outcome,
        "needs_manual_review": needs_manual,
        "yes_price":           yes_price,
        "question":            data.get("question", "")[:60],
        "end_date":            end_date,
    }


# ── FastAPI App with lifespan ──────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the Telegram app + scheduler on startup; clean up on shutdown."""
    _register_handlers(telegram_app)

    scheduler = AsyncIOScheduler(timezone="UTC")
    bot        = telegram_app.bot

    scheduler.add_job(
        scan_job, "interval", minutes=SCAN_INTERVAL_MIN,
        id="scan", args=[bot], max_instances=1,
    )
    scheduler.add_job(
        resolve_job, "cron", hour=9, minute=0,
        id="resolve", args=[bot],
    )
    scheduler.add_job(
        improve_job, "cron", day_of_week="mon", hour=8, minute=0,
        id="improve", args=[bot],
    )
    scheduler.start()
    logger.info(
        f"Scheduler started — scan every {SCAN_INTERVAL_MIN}min, "
        "resolve daily 09:00 UTC, improve weekly Mon 08:00 UTC"
    )

    await telegram_app.initialize()
    await telegram_app.start()

    if WEBHOOK_URL:
        webhook_path = f"{WEBHOOK_URL}/telegram-webhook"
        await telegram_app.bot.set_webhook(
            url=webhook_path,
            allowed_updates=["message", "callback_query"],
        )
        logger.info(f"Webhook registered: {webhook_path}")
        await notify(bot, f"🚀 *Agent started* (webhook mode)\n`{webhook_path}`")
    else:
        # Local dev: fall back to polling in a background task
        logger.info("No WEBHOOK_URL — starting polling (local dev mode)")
        asyncio.create_task(_run_polling())

    yield  # ← app is running

    # Shutdown
    scheduler.shutdown(wait=False)
    await telegram_app.stop()
    await telegram_app.shutdown()


async def _run_polling():
    """Background polling — only used locally when WEBHOOK_URL is not set."""
    await telegram_app.updater.start_polling(drop_pending_updates=True)


fastapi_app = FastAPI(lifespan=lifespan, title="Polymarket Agent")


# ── Routes ─────────────────────────────────────────────────────────────────────

@fastapi_app.get("/health")
async def health():
    """Health check endpoint — Railway pings this to verify the service is up."""
    s = load_state()
    return {
        "status":  "ok",
        "paused":  s.get("paused", False),
        "dry_run": s.get("dry_run", True),
        "time":    _now(),
    }


@fastapi_app.post("/telegram-webhook")
async def telegram_webhook(request: Request):
    """Receive Telegram updates via webhook and pass them to the bot."""
    try:
        data   = await request.json()
        update = Update.de_json(data, telegram_app.bot)
        await telegram_app.process_update(update)
        return Response(content="ok", status_code=200)
    except Exception as e:
        logger.error(f"Webhook handler error: {e}", exc_info=True)
        # Always return 200 to Telegram to prevent retries
        return Response(content="error", status_code=200)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "cloud_agent:fastapi_app",
        host="0.0.0.0",
        port=PORT,
        log_level="info",
    )