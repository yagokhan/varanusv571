#!/usr/bin/env python3
"""
run_paper.py — Varanus Paper Trading Runner

Trains the XGBoost model on historical data, then starts a scheduler that
runs a full paper-trading cycle every 4h at candle close.

Usage
-----
# One-shot cycle (for testing):
    PYTHONPATH=/home/yagokhan python run_paper.py --once

# Dry-run (no state writes, print-to-console only, Telegram messages printed):
    PYTHONPATH=/home/yagokhan python run_paper.py --once --dry-run

# Continuous scheduler (runs at 00:05 / 04:05 / 08:05 / 12:05 / 16:05 / 20:05 UTC):
    PYTHONPATH=/home/yagokhan python run_paper.py

# Custom starting capital:
    PYTHONPATH=/home/yagokhan python run_paper.py --capital 10000

# Reset circuit breaker (after manual review):
    PYTHONPATH=/home/yagokhan python run_paper.py --reset-breaker

# Show current status:
    PYTHONPATH=/home/yagokhan python run_paper.py --status
"""

import argparse
import logging
import os
import signal
import sys
from pathlib import Path

# ── Ensure repo root is on path ───────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from apscheduler.schedulers.blocking import BlockingScheduler

_HERE = Path(__file__).parent

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR = _HERE / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-24s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "paper_trading.log", encoding="utf-8"),
    ],
)
# Quiet noisy third-party libraries
for _noisy in ("ccxt.base", "ccxt", "urllib3", "asyncio",
               "apscheduler", "xgboost"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

logger = logging.getLogger("run_paper")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Varanus Tier 2 Paper Trading Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run a single cycle then exit",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print alerts instead of sending; no state file writes",
    )
    parser.add_argument(
        "--reset-breaker", action="store_true",
        help="Reset the circuit breaker and exit",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Print current paper trading status and exit",
    )
    parser.add_argument(
        "--capital", type=float, default=5_000.0,
        help="Starting capital in USD (default: 5000). Ignored if state file exists.",
    )
    args = parser.parse_args()

    # Import after sys.path is set
    from varanus.paper_trader import PaperTrader

    logger.info(
        "Initialising Varanus Paper Trader | capital=$%.0f | dry_run=%s",
        args.capital, args.dry_run,
    )

    trader = PaperTrader(
        initial_capital=args.capital,
        dry_run=args.dry_run,
    )

    # ── Status display ─────────────────────────────────────────────────────────
    if args.status:
        trader.status()
        return

    # ── Circuit breaker reset ─────────────────────────────────────────────────
    if args.reset_breaker:
        trader.reset_breaker()
        logger.info("Circuit breaker reset. Exiting.")
        return

    # ── Train model on historical cache ───────────────────────────────────────
    logger.info("Training model on historical data (this takes ~60 s) ...")
    trader.train()
    logger.info("Model ready.")

    # ── Single cycle ──────────────────────────────────────────────────────────
    if args.once:
        logger.info("Running single cycle ...")
        result = trader.run_cycle()
        logger.info(
            "Cycle complete | opened=%d  closed=%d  halted=%s",
            len(result["opened"]), len(result["closed"]), result["halted"],
        )
        trader.status()
        return

    # ── Telegram heartbeat listener ───────────────────────────────────────────
    trader.start_listener()

    # ── Continuous scheduler ──────────────────────────────────────────────────
    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(
        trader.run_cycle,
        trigger="cron",
        hour="0,4,8,12,16,20",
        minute=5,           # 5 min after 4h candle close
        max_instances=1,    # Never run overlapping jobs
        coalesce=True,      # Skip missed firings, run once when back online
        id="paper_cycle",
    )

    def _graceful_exit(signum, frame):  # noqa: ANN001
        logger.info("Shutdown signal received — stopping scheduler ...")
        scheduler.shutdown(wait=False)
        sys.exit(0)

    signal.signal(signal.SIGINT,  _graceful_exit)
    signal.signal(signal.SIGTERM, _graceful_exit)

    logger.info(
        "Scheduler started — cycles at 00:05 04:05 08:05 12:05 16:05 20:05 UTC"
    )
    logger.info("Press Ctrl+C to stop.")
    scheduler.start()


if __name__ == "__main__":
    main()
