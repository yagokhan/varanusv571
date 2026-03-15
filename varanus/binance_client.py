"""
varanus/binance_client.py — Binance USDT-M Futures API wrapper.

All exchange interaction is isolated here so the live trader never directly
calls Binance APIs. Provides a single point for error handling, retry logic,
and rate limiting.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import ccxt
import pandas as pd

logger = logging.getLogger(__name__)


class BinanceFuturesClient:
    """Low-level Binance USDT-M Futures client built on ccxt."""

    # ── Init ────────────────────────────────────────────────────────────────

    def __init__(self, api_key: str, api_secret: str, testnet: bool = False):
        self.exchange = ccxt.binanceusdm({
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,
            "options": {
                "defaultType": "future",
                "adjustForTimeDifference": True,
            },
        })
        if testnet:
            self.exchange.set_sandbox_mode(True)
            logger.info("Binance client initialised in TESTNET mode.")
        else:
            logger.info("Binance client initialised in PRODUCTION mode.")

        self.exchange.load_markets()
        self._market_cache: dict = {}
        for sym, mkt in self.exchange.markets.items():
            base = mkt.get("base", "")
            if mkt.get("linear") and mkt.get("settle") == "USDT":
                self._market_cache[base] = mkt

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _symbol(self, asset: str) -> str:
        return f"{asset}/USDT:USDT"

    def _retry(self, fn, *args, retries: int = 3, **kwargs):
        """Retry on transient network errors with exponential backoff."""
        for attempt in range(retries):
            try:
                return fn(*args, **kwargs)
            except (ccxt.NetworkError, ccxt.RequestTimeout) as exc:
                wait = 2 ** attempt
                logger.warning(
                    "Transient error (attempt %d/%d): %s — retrying in %ds",
                    attempt + 1, retries, exc, wait,
                )
                time.sleep(wait)
                if attempt == retries - 1:
                    raise
            except (ccxt.InsufficientFunds, ccxt.InvalidOrder):
                raise  # permanent — no retry

    # ── Account config ──────────────────────────────────────────────────────

    def set_leverage(self, asset: str, leverage: int) -> bool:
        try:
            self._retry(
                self.exchange.set_leverage, leverage, self._symbol(asset)
            )
            logger.debug("Leverage set: %s → %dx", asset, leverage)
            return True
        except Exception as exc:
            # Binance throws if already set — harmless
            if "No need to change" in str(exc) or "-4046" in str(exc):
                return True
            logger.error("set_leverage %s failed: %s", asset, exc)
            return False

    def set_margin_mode(self, asset: str, mode: str = "isolated") -> bool:
        try:
            self._retry(
                self.exchange.set_margin_mode, mode, self._symbol(asset)
            )
            logger.debug("Margin mode set: %s → %s", asset, mode)
            return True
        except Exception as exc:
            if "No need to change" in str(exc) or "-4046" in str(exc):
                return True
            logger.error("set_margin_mode %s failed: %s", asset, exc)
            return False

    # ── Order placement ─────────────────────────────────────────────────────

    def place_market_order(
        self,
        asset: str,
        side: str,
        quantity: float,
        reduce_only: bool = False,
    ) -> dict:
        """
        Place a market order.  side = 'buy' | 'sell'.
        Returns the ccxt order dict with id, status, average, filled, fee.
        """
        symbol = self._symbol(asset)
        rounded_qty = float(
            self.exchange.amount_to_precision(symbol, quantity)
        )
        logger.info(
            "MARKET ORDER  %s  %s  qty=%.8f (raw %.8f)  reduceOnly=%s",
            side.upper(), symbol, rounded_qty, quantity, reduce_only,
        )
        order = self._retry(
            self.exchange.create_order,
            symbol, "market", side, rounded_qty,
            params={"reduceOnly": reduce_only},
        )
        logger.info(
            "ORDER FILLED  id=%s  avg=%.8f  filled=%.8f  cost=%.4f",
            order.get("id"), order.get("average", 0),
            order.get("filled", 0), order.get("cost", 0),
        )
        return order

    def place_stop_market(
        self,
        asset: str,
        side: str,
        quantity: float,
        stop_price: float,
        reduce_only: bool = True,
    ) -> dict:
        """Place a server-side STOP_MARKET order (safety net SL)."""
        symbol = self._symbol(asset)
        rounded_qty = float(
            self.exchange.amount_to_precision(symbol, quantity)
        )
        rounded_stop = float(
            self.exchange.price_to_precision(symbol, stop_price)
        )
        logger.info(
            "STOP_MARKET  %s  %s  qty=%.8f  stop=%.8f",
            side.upper(), symbol, rounded_qty, rounded_stop,
        )
        order = self._retry(
            self.exchange.create_order,
            symbol, "STOP_MARKET", side, rounded_qty,
            params={
                "stopPrice": rounded_stop,
                "reduceOnly": reduce_only,
            },
        )
        logger.info("STOP_MARKET placed  id=%s", order.get("id"))
        return order

    def cancel_all_orders(self, asset: str) -> list:
        """Cancel all open orders for the asset."""
        try:
            result = self._retry(
                self.exchange.cancel_all_orders, self._symbol(asset)
            )
            logger.debug("Cancelled all orders for %s", asset)
            return result
        except Exception as exc:
            logger.warning("cancel_all_orders %s: %s", asset, exc)
            return []

    # ── Position & balance queries ──────────────────────────────────────────

    def get_position(self, asset: str) -> Optional[dict]:
        """Fetch the open position for asset. Returns None if flat."""
        positions = self._retry(
            self.exchange.fetch_positions, [self._symbol(asset)]
        )
        for p in positions:
            if abs(float(p.get("contracts", 0))) > 0:
                return p
        return None

    def get_open_orders(self, asset: str) -> list:
        return self._retry(
            self.exchange.fetch_open_orders, self._symbol(asset)
        )

    def get_balance(self) -> dict:
        """Return {'total': float, 'free': float, 'used': float} for USDT."""
        bal = self._retry(self.exchange.fetch_balance)
        usdt = bal.get("USDT", {})
        return {
            "total": float(usdt.get("total", 0)),
            "free":  float(usdt.get("free", 0)),
            "used":  float(usdt.get("used", 0)),
        }

    def get_ticker(self, asset: str) -> float:
        """Return the last mark price for the asset."""
        ticker = self._retry(
            self.exchange.fetch_ticker, self._symbol(asset)
        )
        return float(ticker["last"])

    def fetch_ohlcv(
        self, asset: str, timeframe: str = "4h", limit: int = 500,
    ) -> Optional[pd.DataFrame]:
        """Fetch OHLCV bars. Drops the last (still-forming) bar."""
        raw = self._retry(
            self.exchange.fetch_ohlcv,
            self._symbol(asset), timeframe, limit=limit,
        )
        if not raw:
            return None
        df = pd.DataFrame(
            raw, columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp").sort_index()
        df = df.iloc[:-1]   # drop still-forming bar
        return df if not df.empty else None

    # ── Quantity helpers ────────────────────────────────────────────────────

    def compute_quantity(self, asset: str, position_usd: float, price: float) -> float:
        """
        Convert a USD position size into the exchange-compliant quantity.
        Raises ValueError if below minimum notional ($5).
        """
        symbol = self._symbol(asset)
        raw_qty = position_usd / price
        rounded = float(self.exchange.amount_to_precision(symbol, raw_qty))

        # Minimum notional check
        notional = rounded * price
        if notional < 5.0:
            raise ValueError(
                f"{asset}: notional ${notional:.2f} below Binance minimum $5"
            )
        return rounded

    # ── Connectivity ────────────────────────────────────────────────────────

    def ping(self) -> bool:
        try:
            self._retry(self.exchange.fetch_time)
            return True
        except Exception as exc:
            logger.error("Binance ping failed: %s", exc)
            return False
