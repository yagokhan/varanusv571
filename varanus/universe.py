"""
varanus/universe.py — Tier 2 Universe Definition.

Static asset list of 20 mid-cap crypto assets occupying the structural
bridge between BTC/ETH institutional liquidity and micro-cap speculation.
Dynamic exclusion rules filter the list at runtime based on live volume.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

# ── Static universe ────────────────────────────────────────────────────────────

TIER2_UNIVERSE: list[str] = [
    "ADA",
    "AVAX",
    "LINK",
    "DOT",
    "TRX",
    "NEAR",
    "UNI",
    "SUI",
    "ARB",
    "OP",
    "POL",
    "APT",
    "ATOM",
    "FIL",
    "ICP"
]

# High-Volatility Sub-Tier: wider barriers + reduced position size (0.75×)
HIGH_VOL_SUBTIER: list[str] = ["TAO", "ASTR", "KITE", "ICP"]

# Timeframe config
TIER2_QUOTE       = "USDT"
TIER2_TF          = "4h"           # Primary timeframe
TIER2_TF_HTF      = "1d"           # Higher-timeframe bias filter
TIER2_MIN_VOL_USD = 50_000_000     # 24h minimum volume gate (USD)

# ── Exclusion rules ───────────────────────────────────────────────────────────

EXCLUSION_RULES: dict = {
    # Suspend asset if 24h USD volume < $50M for 3+ consecutive days
    "min_volume_usd": 50_000_000,
    # Suppress signals ±2h around BTC/ETH options expiry (last Friday of month)
    "options_expiry_pause": True,
    # Position size multiplier for HIGH_VOL_SUBTIER assets
    "high_vol_size_scalar": 0.75,
}

# ── Public API ─────────────────────────────────────────────────────────────────

def get_symbols() -> list[str]:
    """Return the full Tier 2 universe as exchange symbol strings (BASE/USDT)."""
    return [f"{asset}/{TIER2_QUOTE}" for asset in TIER2_UNIVERSE]


def is_high_vol(asset: str) -> bool:
    """True if *asset* belongs to the high-volatility sub-tier."""
    return asset.upper() in HIGH_VOL_SUBTIER


def get_size_scalar(asset: str) -> float:
    """
    Position size scalar for *asset*.
    Returns 0.75 for HIGH_VOL_SUBTIER assets, 1.0 for all others.
    """
    return EXCLUSION_RULES["high_vol_size_scalar"] if is_high_vol(asset) else 1.0


def get_active_universe(volume_data: dict[str, float]) -> list[str]:
    """
    Filter TIER2_UNIVERSE by current 24h USD volume.

    Args:
        volume_data: mapping of {asset: 24h_volume_usd}, e.g. {"DOT": 120_000_000}.
                     Assets missing from the dict are treated as zero volume.

    Returns:
        Subset of TIER2_UNIVERSE whose volume meets TIER2_MIN_VOL_USD.
    """
    active = [
        asset for asset in TIER2_UNIVERSE
        if volume_data.get(asset, 0) >= EXCLUSION_RULES["min_volume_usd"]
    ]
    suspended = set(TIER2_UNIVERSE) - set(active)
    if suspended:
        logger.info(f"Volume-suspended assets ({len(suspended)}): {sorted(suspended)}")
    return active


def is_options_expiry_window(dt: Optional[datetime] = None) -> bool:
    """
    Return True if *dt* falls within ±2 hours of BTC/ETH options expiry.

    Options expire at 08:00 UTC on the last Friday of each month (Deribit/OKX).

    Args:
        dt: UTC datetime to check. Defaults to datetime.now(timezone.utc).
    """
    if not EXCLUSION_RULES["options_expiry_pause"]:
        return False

    if dt is None:
        dt = datetime.now(timezone.utc)

    expiry = _last_friday_of_month(dt.year, dt.month)
    expiry_utc = expiry.replace(hour=8, minute=0, second=0, microsecond=0)
    delta = abs((dt - expiry_utc).total_seconds())
    return delta <= 2 * 3600


def fetch_volumes(exchange=None) -> dict[str, float]:
    """
    Fetch current 24h USD quote volume for all Tier 2 assets from Binance.

    Args:
        exchange: optional pre-initialized ccxt.binance instance.
                  If None, a public (no-key) instance is created.

    Returns:
        dict mapping asset base currency → 24h volume in USD.
    """
    try:
        import ccxt
    except ImportError:
        raise RuntimeError("ccxt is required for fetch_volumes(). Install it in algo_env.")

    if exchange is None:
        exchange = ccxt.binance({"enableRateLimit": True})

    symbols = get_symbols()
    tickers = exchange.fetch_tickers(symbols)

    volumes: dict[str, float] = {}
    for asset in TIER2_UNIVERSE:
        sym = f"{asset}/{TIER2_QUOTE}"
        ticker = tickers.get(sym, {})
        volumes[asset] = float(ticker.get("quoteVolume") or 0.0)

    return volumes


# ── Internal helpers ───────────────────────────────────────────────────────────

def _last_friday_of_month(year: int, month: int) -> datetime:
    """Return the last Friday of *year*/*month* as a date-only datetime (UTC)."""
    # Start from the last day of the month and walk backwards
    if month == 12:
        next_month_first = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        next_month_first = datetime(year, month + 1, 1, tzinfo=timezone.utc)
    last_day = next_month_first - timedelta(days=1)
    # weekday(): Monday=0 … Friday=4 … Sunday=6
    days_since_friday = (last_day.weekday() - 4) % 7
    return last_day - timedelta(days=days_since_friday)
