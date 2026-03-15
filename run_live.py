#!/usr/bin/env python3
"""
run_live.py — Varanus Live Trading Runner (Binance USDT-M Futures)

Trains the XGBoost model on historical data, then starts a scheduler that
runs a full live-trading cycle every 4h at candle close — placing real orders.

Usage
-----
# Testnet dry-run (no real orders, Binance testnet):
    python run_live.py --testnet --once --dry-run

# Testnet single cycle (real orders on testnet):
    python run_live.py --testnet --once

# Production single cycle (REAL MONEY):
    python run_live.py --once

# Production continuous scheduler (REAL MONEY):
    python run_live.py

# Emergency kill (close all positions immediately):
    python run_live.py --kill

# Reset circuit breaker:
    python run_live.py --reset-breaker

# Show current status:
    python run_live.py --status

# Custom starting capital:
    python run_live.py --capital 1000 --testnet --once
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

_HERE    = Path(__file__).parent
PID_FILE = _HERE / "logs" / "live_trader.pid"

# ── Logging (separate from paper trading) ──────────────────────────────────────
LOG_DIR = _HERE / "logs"
LOG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-24s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "live_trading.log", encoding="utf-8"),
    ],
)
for _noisy in ("ccxt.base", "ccxt", "urllib3", "asyncio",
               "apscheduler", "xgboost"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

logger = logging.getLogger("run_live")


def _acquire_pid_lock() -> None:
    """
    Write current PID to PID_FILE. Prevents duplicate live trading instances.
    Separate from paper_trader.pid — both can run simultaneously.
    """
    import atexit

    if PID_FILE.exists():
        try:
            existing_pid = int(PID_FILE.read_text().strip())
            os.kill(existing_pid, 0)
            print(
                f"ERROR: Another LIVE instance is already running (PID {existing_pid}).\n"
                f"       Stop it first, or delete {PID_FILE} if it is stale.",
                file=sys.stderr,
            )
            sys.exit(1)
        except (ProcessLookupError, ValueError):
            pass

    PID_FILE.write_text(str(os.getpid()))

    def _remove_pid():
        try:
            PID_FILE.unlink(missing_ok=True)
        except Exception:
            pass

    atexit.register(_remove_pid)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Varanus Live Trading Runner (Binance USDT-M Futures)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run a single cycle then exit",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print alerts instead of sending; no state file writes; no real orders",
    )
    parser.add_argument(
        "--testnet", action="store_true",
        help="Use Binance Futures testnet (no real money)",
    )
    parser.add_argument(
        "--kill", action="store_true",
        help="Emergency: close all positions and exit",
    )
    parser.add_argument(
        "--reset-breaker", action="store_true",
        help="Reset the circuit breaker / kill switch and exit",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Print current live trading status and exit",
    )
    parser.add_argument(
        "--capital", type=float, default=1_000.0,
        help="Starting capital in USD (default: 1000). Ignored if state file exists.",
    )
    args = parser.parse_args()

    # Import after sys.path is set
    from varanus.live_trader import LiveTrader

    mode = "TESTNET" if args.testnet else "PRODUCTION"
    logger.info(
        "Initialising Varanus Live Trader | mode=%s | capital=$%.0f | dry_run=%s",
        mode, args.capital, args.dry_run,
    )

    trader = LiveTrader(
        initial_capital=args.capital,
        dry_run=args.dry_run,
        testnet=args.testnet,
    )

    # ── Status display ─────────────────────────────────────────────────────
    if args.status:
        trader.status()
        return

    # ── Circuit breaker reset ──────────────────────────────────────────────
    if args.reset_breaker:
        trader.reset_breaker()
        logger.info("Circuit breaker and kill switch reset. Exiting.")
        return

    # ── Emergency kill ─────────────────────────────────────────────────────
    if args.kill:
        logger.critical("EMERGENCY KILL requested via CLI")
        trader._emergency_close_all("cli_kill_command")
        logger.info("All positions closed. Exiting.")
        return

    # ── PID lock ───────────────────────────────────────────────────────────
    if not args.once:
        _acquire_pid_lock()

    # ── Production safety confirmation ─────────────────────────────────────
    if not args.testnet and not args.dry_run and not args.once:
        bal = trader.client.get_balance()
        print("\n" + "=" * 60)
        print("  ⚠️  WARNING: LIVE TRADING WITH REAL MONEY  ⚠️")
        print("=" * 60)
        print(f"  Mode:            PRODUCTION")
        print(f"  Binance Balance: ${bal['total']:,.2f}")
        print(f"  Starting Capital: ${args.capital:,.2f}")
        print(f"  Schedule:        Every 4h (00:05 04:05 08:05 12:05 16:05 20:05 UTC)")
        print("=" * 60)
        confirm = input("  Type 'CONFIRM' to proceed: ").strip()
        if confirm != "CONFIRM":
            print("Aborted.")
            sys.exit(0)
        print()

    # ── Train model ────────────────────────────────────────────────────────
    logger.info("Training model on historical data (this takes ~60 s) ...")
    trader.train()
    logger.info("Model ready.")

    # ── Single cycle ───────────────────────────────────────────────────────
    if args.once:
        logger.info("Running single LIVE cycle ...")
        result = trader.run_cycle()
        logger.info(
            "Cycle complete | opened=%d  closed=%d  halted=%s",
            len(result["opened"]), len(result["closed"]), result["halted"],
        )
        trader.status()
        return

    # ── Telegram listener ──────────────────────────────────────────────────
    trader.start_listener()

    # ── Continuous scheduler ───────────────────────────────────────────────
    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(
        trader.run_cycle,
        trigger="cron",
        hour="0,4,8,12,16,20",
        minute=5,
        max_instances=1,
        coalesce=True,
        id="live_cycle",
    )

    def _graceful_exit(signum, frame):
        logger.info("Shutdown signal received — stopping live scheduler ...")
        scheduler.shutdown(wait=False)
        sys.exit(0)

    signal.signal(signal.SIGINT,  _graceful_exit)
    signal.signal(signal.SIGTERM, _graceful_exit)

    logger.info(
        "LIVE scheduler started — cycles at 00:05 04:05 08:05 12:05 16:05 20:05 UTC"
    )
    logger.info("Commands: /kill /resume confirm /positions /status (via Telegram)")
    logger.info("Press Ctrl+C to stop.")
    scheduler.start()


if __name__ == "__main__":
    main()
