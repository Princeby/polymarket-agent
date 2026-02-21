#!/bin/bash
# Polymarket Agent Runner
# Called by launchd to run the agent on a schedule.

PROJ_DIR="/Users/ogbodoebuka/Desktop/Projects/polymarket-agent"
VENV="$PROJ_DIR/venv/bin/activate"
LOG_DIR="$PROJ_DIR/data/logs"

mkdir -p "$LOG_DIR"

source "$VENV"
cd "$PROJ_DIR"

TIMESTAMP=$(date +%Y-%m-%d_%H%M)

case "${1:-scan}" in
    scan)
        echo "[$TIMESTAMP] Running market scan..." >> "$LOG_DIR/agent.log"
        python main.py --once --markets 20 >> "$LOG_DIR/agent.log" 2>&1
        echo "[$TIMESTAMP] Scan complete." >> "$LOG_DIR/agent.log"
        echo "---" >> "$LOG_DIR/agent.log"
        ;;
    resolve)
        echo "[$TIMESTAMP] Running auto-resolution..." >> "$LOG_DIR/resolve.log"
        python resolve.py >> "$LOG_DIR/resolve.log" 2>&1
        echo "[$TIMESTAMP] Resolution complete." >> "$LOG_DIR/resolve.log"
        echo "---" >> "$LOG_DIR/resolve.log"
        ;;
esac
