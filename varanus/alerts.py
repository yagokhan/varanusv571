import requests

# ── Entry alert ───────────────────────────────────────────────────────────────
ALERT_FORMAT = (
    "🦎 *VARANUS v5.7.1* | {asset} {direction} @ {confidence:.0%}\n"
    "Entry: {entry_price} | TP: {take_profit} | SL: {stop_loss}\n"
    "R:R {rr_ratio:.1f}x | Lev: {leverage}x | ATR: {atr_14:.4f}\n"
    "MSS: {mss} | FVG✓ | Sweep✓ | RVol: {rvol:.2f}x | RSI: {rsi:.1f}\n"
    "HTF: {htf_bias} | Pos: ${position_usd:.0f} | Port Lev: {port_lev:.2f}x"
)

REQUIRED_FIELDS = [
    "timestamp_utc", "asset", "direction", "confidence", "leverage",
    "entry_price", "take_profit", "stop_loss", "rr_ratio", "atr_14",
    "mss", "fvg_valid", "sweep_confirmed", "rvol", "rsi", "htf_bias",
    "position_usd", "port_lev",
]

# ── Exit alert ────────────────────────────────────────────────────────────────
EXIT_FORMAT = (
    "🔒 *VARANUS v5.7.1 EXIT* | {asset} {direction} {outcome}\n"
    "Entry: {entry_price:.4f} → Exit: {exit_price:.4f}\n"
    "PnL: {pnl_sign}${pnl_abs:.2f} ({pnl_sign}{pnl_pct:.2f}%) | Duration: {duration_h:.0f}h\n"
    "Outcome: {outcome_label}"
)

# ── Circuit breaker alert ─────────────────────────────────────────────────────
HALT_FORMAT = (
    "🚨 *VARANUS v5.7.1 — SIGNALS HALTED*\n"
    "Daily Loss: {daily_loss_pct:.1f}% | Drawdown: {drawdown_pct:.1f}%\n"
    "Current Equity: ${current_equity:.2f}\n"
    "Reason: {reason}"
)


def _post(msg: str, bot_token: str, chat_id: str) -> None:
    """Send a Markdown message via the Telegram Bot API. Fails silently."""
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"},
            timeout=5.0,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"[alerts] Warning: Telegram send failed: {e}")


def send_alert(trade: dict, bot_token: str, chat_id: str,
               dry_run: bool = False) -> None:
    """
    Validate and send an entry signal alert to Telegram.
    Raises ValueError if any REQUIRED_FIELDS are missing.
    In dry_run mode, prints the message instead of sending.
    """
    missing = [f for f in REQUIRED_FIELDS if f not in trade]
    if missing:
        raise ValueError(f"Alert missing fields: {missing}")

    msg = ALERT_FORMAT.format(**trade)

    if dry_run:
        print(f"[dry-run] Entry alert:\n{msg}\n")
        return

    _post(msg, bot_token, chat_id)


def send_exit_alert(trade: dict, bot_token: str, chat_id: str,
                    dry_run: bool = False) -> None:
    """
    Send a trade exit alert (TP / SL / time).
    Expects keys: asset, entry_price, exit_price, pnl_usd, outcome,
                  entry_ts, exit_ts.
    """
    import pandas as pd

    outcome = trade.get("outcome", "unknown")
    outcome_labels = {
        "tp":               "✅ Take-Profit Hit",
        "sl":               "❌ Stop-Loss Hit",
        "time":             "⏱ Time Barrier (Force Close)",
        "trailing_sl_hit":  "🔁 Trailing Stop Hit",
        "signal_decay":     "📉 Signal Decay Exit",
        "mss_invalidation": "⚡ MSS Invalidation Exit",
    }

    try:
        duration_h = (
            pd.to_datetime(trade["exit_ts"]) - pd.to_datetime(trade["entry_ts"])
        ).total_seconds() / 3600
    except Exception:
        duration_h = 0.0

    pnl = trade.get("pnl_usd", 0.0)
    position = trade.get("position_usd", 0.0)
    pnl_pct = abs(pnl) / position * 100 if position else 0.0
    direction = trade.get("direction", 1)
    dir_label = "LONG ↑" if direction == 1 else "SHORT ↓"
    msg = EXIT_FORMAT.format(
        asset         = trade.get("asset", "?"),
        direction     = dir_label,
        outcome       = outcome.upper(),
        entry_price   = float(trade.get("entry_price", 0)),
        exit_price    = float(trade.get("exit_price", 0)),
        pnl_sign      = "+" if pnl >= 0 else "-",
        pnl_abs       = abs(pnl),
        pnl_pct       = pnl_pct,
        duration_h    = duration_h,
        outcome_label = outcome_labels.get(outcome, outcome),
    )

    if dry_run:
        print(f"[dry-run] Exit alert:\n{msg}\n")
        return

    _post(msg, bot_token, chat_id)


def send_no_signal_alert(cycle_time: str, equity: float, daily_pct: float,
                         bot_token: str, chat_id: str,
                         dry_run: bool = False) -> None:
    """Send a no-signal notification at the end of each cycle."""
    msg = (
        f"🔍 *VARANUS v5.7.1 — No Signal*\n"
        f"Cycle: {cycle_time}\n"
        f"Scanned 15 assets — no setup above confidence threshold\n"
        f"Equity: ${equity:,.2f} | Daily: {daily_pct:+.1f}%"
    )
    if dry_run:
        print(f"[dry-run] No-signal alert:\n{msg}\n")
        return
    _post(msg, bot_token, chat_id)


def send_heartbeat_alert(state: dict, health: dict,
                         bot_token: str, chat_id: str,
                         next_cycle_mins: int = 0,
                         live_prices: dict[str, float] | None = None) -> None:
    """Send professional portfolio status in response to /status or heartbeat."""
    from datetime import datetime, timezone, timedelta

    if live_prices is None:
        live_prices = {}

    # ── Core data ────────────────────────────────────────────────────────────
    open_trades = state.get("open_trades", {})
    closed      = state.get("closed_trades", [])
    initial     = state.get("initial_capital", 0.0)
    peak        = state.get("peak_equity", initial)
    equity      = health["current_equity"]
    total_pnl   = equity - initial
    daily_pct   = health["daily_loss_pct"]
    dd_pct      = health["drawdown_pct"]

    # TSI timestamp (UTC+3)
    now_utc = datetime.now(timezone.utc)
    now_tsi = now_utc + timedelta(hours=3)
    tsi_str = now_tsi.strftime("%Y-%m-%d %H:%M TSI (UTC+3)")

    # Countdown to next cycle
    h, m      = divmod(next_cycle_mins, 60)
    countdown = f"{h}h {m}m" if h else f"{m}m"

    # ── Deployed & ROI (closed trades) ───────────────────────────────────────
    total_deployed = sum(t.get("position_usd", 0.0) for t in closed)
    total_pnl_closed = sum(t.get("pnl_usd", 0.0) for t in closed)
    roi_pct = total_pnl_closed / total_deployed * 100 if total_deployed else 0.0

    # ── Leveraged exposure (open trades) ─────────────────────────────────────
    total_exposure = sum(
        t.get("position_usd", 0.0) * t.get("leverage", 1.0)
        for t in open_trades.values()
    )
    exposure_pct = total_exposure / initial * 100 if initial else 0.0

    # ── Status icon ──────────────────────────────────────────────────────────
    status_icon = "🚨 HALTED" if state.get("halted") else "🟢 ACTIVE"

    # ── 8-Fold maintenance metrics ───────────────────────────────────────────
    last_train_str = state.get("last_training_date", "2026-02-28")
    try:
        last_train_dt = datetime.fromisoformat(last_train_str).replace(tzinfo=timezone.utc)
    except Exception:
        last_train_dt = datetime.strptime(last_train_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    next_fold_dt   = last_train_dt + timedelta(days=120)
    fold_remaining = next_fold_dt - now_utc
    fold_days      = max(fold_remaining.days, 0)

    fold_registry = state.get("fold_registry", [])
    if fold_registry:
        fold_range = f"{fold_registry[0]['start']} -> {fold_registry[-1]['end']}"
    else:
        fold_range = "N/A"

    # ATH age with tiered icons
    peak_eq_date = state.get("peak_equity_date", "")
    ath_age = 0
    if peak_eq_date:
        try:
            ath_dt  = datetime.fromisoformat(peak_eq_date).replace(tzinfo=timezone.utc)
            ath_age = (now_utc - ath_dt).days
        except Exception:
            pass
    if ath_age < 15:
        ath_icon = "🟢"
    elif ath_age <= 25:
        ath_icon = "🟡"
    else:
        ath_icon = "🔴"

    # Data sync status
    cache_ok = state.get("last_cache_ok", True)
    data_icon = "✅ Synced" if cache_ok else "⚠ Partial"

    # Maintenance / drift flags
    maint_flag = state.get("maintenance_required", False)
    drift_flag = state.get("performance_drift", False)
    if maint_flag:
        health_icon = "🔧 MAINTENANCE DUE"
    elif drift_flag:
        health_icon = "🔴 PERFORMANCE DRIFT"
    else:
        health_icon = "✅ Nominal"

    # ── Unrealized PnL from live prices ─────────────────────────────────────
    total_unrealized = 0.0
    open_pnl_map = {}
    for asset, t in open_trades.items():
        price_now = live_prices.get(asset)
        if price_now is None or not t.get("entry_price"):
            continue
        entry  = t["entry_price"]
        pos    = t.get("position_usd", 0.0)
        lev    = t.get("leverage", 1.0)
        dirn   = t.get("direction", 1)
        if dirn == 1:
            upnl = (price_now - entry) / entry * pos * lev
        else:
            upnl = (entry - price_now) / entry * pos * lev
        upnl_pct = upnl / pos * 100 if pos else 0.0
        open_pnl_map[asset] = {"upnl": upnl, "upnl_pct": upnl_pct, "price": price_now}
        total_unrealized += upnl

    # ── Build message ────────────────────────────────────────────────────────
    lines = [
        f"🐊 *VARANUS v5.7.1 The Golden Ratio*",
        f"_{tsi_str}_ | {status_icon}",
        f"",
        f"*— Portfolio —*",
        f"  Capital:    ${initial:>10,.2f}",
        f"  Equity:     ${equity:>10,.2f}",
        f"  Total PnL:  ${total_pnl:>+10,.2f}  ({total_pnl / initial * 100 if initial else 0:+.2f}%)",
        f"  Peak:       ${peak:>10,.2f}",
        f"  Deployed:   ${total_deployed:>10,.2f}",
        f"  ROI:        ${total_pnl_closed:>+10,.2f}  ({roi_pct:+.2f}% on deployed)",
        f"  Exposure:   ${total_exposure:>10,.2f}  ({exposure_pct:.1f}% of capital)",
        f"  Unreal PnL: ${total_unrealized:>+10,.2f}  {'📈' if total_unrealized >= 0 else '📉'}",
        f"",
        f"*— Risk —*",
        f"  Daily:      {daily_pct:+.1f}%  {'🚨' if daily_pct <= -5 else '⚠' if daily_pct <= -3 else '✅'}",
        f"  Drawdown:   {dd_pct:+.1f}%  {'🚨' if dd_pct <= -10 else '⚠' if dd_pct <= -5 else '✅'}",
        f"",
        f"*— 8-Fold Maintenance —*",
        f"  Window:     8-Fold Sliding ✅",
        f"  Folds:      {fold_range}",
        f"  Next Rot:   {next_fold_dt.strftime('%Y-%m-%d')}  ({fold_days}d left)",
        f"  ATH Age:    {ath_age}d / 30d  {ath_icon}",
        f"  Data:       {data_icon}",
        f"  Health:     {health_icon}",
        f"",
        f"*— Trades —*",
        f"  Open: {len(open_trades)}  |  Closed: {len(closed)}  |  Next: {countdown}",
    ]

    # ── Open positions ───────────────────────────────────────────────────────
    if open_trades:
        lines.append("")
        lines.append(f"*— Open Positions ({len(open_trades)}) —*")
        for asset, t in open_trades.items():
            direction = "LONG ↑" if t["direction"] == 1 else "SHORT ↓"
            lev       = t.get("leverage", 1.0)
            conf      = t.get("confidence", 0.0)
            pos       = t.get("position_usd", 0.0)
            entry     = t.get("entry_price", 0.0)
            tp        = t.get("take_profit", 0.0)
            sl        = t.get("stop_loss", 0.0)
            trail     = t.get("trail_active", False)
            try:
                entry_dt = datetime.fromisoformat(str(t.get("entry_ts", "")).replace("Z", "+00:00"))
                if entry_dt.tzinfo is None:
                    entry_dt = entry_dt.replace(tzinfo=timezone.utc)
                held_h = (now_utc - entry_dt).total_seconds() / 3600
                held_str = f"{held_h:.0f}h"
            except Exception:
                held_str = "?"
            tp_pct = (tp - entry) / entry * 100 if entry else 0
            sl_pct = (sl - entry) / entry * 100 if entry else 0
            exposure_t = pos * lev
            trail_str = " 🔁TRAIL" if trail else ""

            # Live price & unrealized PnL
            pnl_info = open_pnl_map.get(asset)
            if pnl_info:
                now_price = pnl_info["price"]
                upnl      = pnl_info["upnl"]
                upnl_pct  = pnl_info["upnl_pct"]
                price_chg = (now_price - entry) / entry * 100 if entry else 0
                pnl_icon  = "🟢" if upnl >= 0 else "🔴"
                price_line = (
                    f"    Now: {now_price:.4f} ({price_chg:+.2f}%) | "
                    f"uPnL: {pnl_icon} ${upnl:+.2f} ({upnl_pct:+.2f}%)"
                )
            else:
                price_line = "    Now: -- (price unavailable)"

            lines.append(
                f"  {asset} {direction} | {conf:.0%} conf | {lev:.0f}x | "
                f"${pos:,.0f} -> ${exposure_t:,.0f}{trail_str}"
            )
            lines.append(
                f"    E: {entry:.4f} | TP: {tp:.4f} ({tp_pct:+.1f}%) | "
                f"SL: {sl:.4f} ({sl_pct:+.1f}%) | {held_str}"
            )
            lines.append(price_line)

    # ── Closed trades summary ────────────────────────────────────────────────
    if closed:
        wins      = sum(1 for t in closed if t.get("pnl_usd", 0) > 0)
        losses    = len(closed) - wins
        tot_pnl_c = sum(t.get("pnl_usd", 0) for t in closed)
        win_rate  = wins / len(closed) * 100
        avg_win   = (sum(t["pnl_usd"] for t in closed if t.get("pnl_usd", 0) > 0) / wins) if wins else 0
        avg_loss  = (sum(t["pnl_usd"] for t in closed if t.get("pnl_usd", 0) <= 0) / losses) if losses else 0
        best      = max(closed, key=lambda t: t.get("pnl_usd", 0))
        worst     = min(closed, key=lambda t: t.get("pnl_usd", 0))
        outcomes  = {}
        for t in closed:
            o = t.get("outcome", "?")
            outcomes[o] = outcomes.get(o, 0) + 1
        exit_parts = []
        for k, v in sorted(outcomes.items(), key=lambda x: -x[1]):
            exit_parts.append(f"{k.upper().replace('_','-')}:{v}")
        exit_str = "  ".join(exit_parts)

        tot_deployed_c = sum(t.get("position_usd", 0.0) for t in closed)
        roi_c = tot_pnl_c / tot_deployed_c * 100 if tot_deployed_c else 0.0

        best_pos  = best.get("position_usd", 1.0)
        worst_pos = worst.get("position_usd", 1.0)
        best_roi  = best.get("pnl_usd", 0) / best_pos * 100 if best_pos else 0
        worst_roi = worst.get("pnl_usd", 0) / worst_pos * 100 if worst_pos else 0

        lines.append("")
        lines.append(f"*— Closed Trades ({len(closed)}) —*")
        lines.append(f"  W/L: {wins}W / {losses}L  ({win_rate:.0f}% WR)")
        lines.append(f"  Deployed: ${tot_deployed_c:,.2f}  PnL: ${tot_pnl_c:+,.2f}  ({roi_c:+.2f}%)")
        lines.append(f"  Avg Win: ${avg_win:+,.2f}  |  Avg Loss: ${avg_loss:+,.2f}")
        lines.append(f"  Best:  {best.get('asset','?')} ${best.get('pnl_usd',0):+,.2f} ({best_roi:+.1f}%)")
        lines.append(f"  Worst: {worst.get('asset','?')} ${worst.get('pnl_usd',0):+,.2f} ({worst_roi:+.1f}%)")
        lines.append(f"  Exits: {exit_str}")

    _post("\n".join(lines), bot_token, chat_id)


MAINTENANCE_FORMAT = (
    "🔧 *VARANUS v5.7.1 — MODEL MAINTENANCE REQUIRED*\n"
    "Trigger: {trigger}\n"
    "{details}\n"
    "Action: Prepare v5.7.2 candidate model."
)


def send_maintenance_alert(trigger: str, details: str,
                           bot_token: str, chat_id: str,
                           dry_run: bool = False) -> None:
    """Send a model maintenance / performance drift alert."""
    msg = MAINTENANCE_FORMAT.format(trigger=trigger, details=details)
    if dry_run:
        print(f"[dry-run] Maintenance alert:\n{msg}\n")
        return
    _post(msg, bot_token, chat_id)


def send_halt_alert(health: dict, bot_token: str, chat_id: str,
                    dry_run: bool = False) -> None:
    """
    Send a circuit-breaker halt notification.
    Expects the dict returned by risk.check_portfolio_health().
    """
    daily  = health.get("daily_loss_pct", 0.0)
    dd     = health.get("drawdown_pct", 0.0)
    equity = health.get("current_equity", 0.0)

    if daily <= -5.0 and dd <= -15.0:
        reason = "Daily loss AND peak drawdown limits both breached"
    elif daily <= -5.0:
        reason = f"Daily loss limit breached ({daily:.1f}%)"
    else:
        reason = f"Peak drawdown limit breached ({dd:.1f}%)"

    msg = HALT_FORMAT.format(
        daily_loss_pct = daily,
        drawdown_pct   = dd,
        current_equity = equity,
        reason         = reason,
    )

    if dry_run:
        print(f"[dry-run] Halt alert:\n{msg}\n")
        return

    _post(msg, bot_token, chat_id)
