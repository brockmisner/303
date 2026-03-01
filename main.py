#!/usr/bin/env python3
"""
PolyBot v2 — Polymarket 5-Min BTC Up/Down Quant Trading Bot
=============================================================

Key innovation: Uses Polymarket's own Chainlink RTDS feed as primary oracle,
eliminating the $20-60 basis error from using Binance/Coinbase prices.

Architecture:
  - chainlink_rtds.py  → Primary oracle (exact resolution feed)
  - oracle_engine.py   → Edge decision logic (cone, Z-gate, persistence)
  - main.py            → Bot orchestration, survival engine, execution

Price hierarchy:
  1. Chainlink RTDS (wss://ws-live-data.polymarket.com)  — PRIMARY
  2. Coinbase WebSocket (real-time trades)                 — sigma + fallback
  3. Binance WebSocket (fallback if Coinbase stale)        — secondary fallback
  4. Chainlink on-chain poll (Polygon)                     — emergency only

Edge pipeline (100 Hz):
  cone_p_and_z → edge calculation → Z gate → persistence → Kelly sizing → IOC/FOK
"""

import asyncio
import logging
import json
import time
import math
import os
import csv
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from collections import deque
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import sys
import threading

import requests
import websockets
import numpy as np
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

from py_clob_client.order_builder.constants import BUY, SELL
from web3 import Web3
from py_clob_client.client import ClobClient
from py_clob_client.clob_types import (
    OrderArgs, OrderType, ApiCreds,
    PartialCreateOrderOptions,
    BalanceAllowanceParams, AssetType,
)
from py_clob_client.constants import POLYGON

# Local modules
from oracle_engine import (
    LagAdaptiveZ, PersistState, decide_edge, ms_now,
    fee_per_share, cone_p_and_z,
    update_sigma_history, Z_TRAJ, reset_sniper_lock,
    confirm_sniper_fire,
    BASIS, CB_LEAD,
)
from chainlink_rtds import (
    RTDS, chainlink_rtds_task, STRIKE_CAPTURE,
)
from regime import RegimeClassifier, JumpDetector
from tail_risk import TailRiskGuard
from order_flow import FlowTracker, GoldskyAnalytics
from participation import ParticipationVelocity, PVConfig
from bipower_jump import BipowerJumpFilter, BipowerConfig
from momentum import MomentumEngine, MomentumConfig
from fill_prob import FillProbModel, ExpiryScaler
from sprt import SPRTValidator
from vwap_overlay import VWAPTracker
from execution_safety import validate_execution_edge, book_health_score
from endgame_manager import EndgameManager, EndgameConfig
from exit_manager import ExitManager, ExitConfig
from position_monitor import PositionMonitor, PositionMonitorConfig
from pnl_tracker import PnLTracker
from position import Position, Portfolio, Side
from inventory_layer import InventoryDecisionLayer, InventoryLayerConfig, DecisionType
from telemetry import TelemetryEmitter, TelemetryConfig
from adaptive_executor import L2Tracker, AdaptiveExecutor, MicroSnapshot, ExecutionResult


# ════════════════════════════════════════════════════════════════════════════
# INITIALIZATION
# ════════════════════════════════════════════════════════════════════════════

WINDOW_OPEN_MS:     int = 0
FIRST_REAL_BOOK_MS: int = 0

load_dotenv()
os.makedirs("logs", exist_ok=True)

# ── Logging ─────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fh = logging.FileHandler("logs/bot.log", encoding="utf-8")
fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(fh)
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(sh)


# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

CANDLE_API_URL      = "https://polymarket.com/api/chainlink-candles"
CRYPTO_PRICE_URL    = "https://polymarket.com/api/crypto/crypto-price"
POLYMARKET_TIMEZONE = ZoneInfo("America/New_York")
SIMULATION_MODE     = os.getenv("SIM_MODE", "true").lower() == "true"

MAX_SPEND_PER_ORDER_USD    = float(
    os.getenv("MAX_SPEND_PER_ORDER_USD", os.getenv("MAX_SPEND_USD", "999999.0"))
)
WINDOW_SPEND_CAP_USD       = float(os.getenv("WINDOW_SPEND_CAP_USD", "999999.0"))
POLY_BOOK_MAX_AGE_MS       = int(os.getenv("POLY_BOOK_MAX_AGE_MS", "15000"))
POLY_BOOK_RESEED_AHEAD_MS  = int(os.getenv("POLY_BOOK_RESEED_AHEAD_MS", "4000"))

# Oracle staleness thresholds
RTDS_FRESH_MS        = 45_000   # RTDS considered fresh if < 45s old
COINBASE_STALE_MS    = 5_000    # Binance takes over if Coinbase > 5s stale
BTC_FEED_MAX_AGE_MS  = 3_000    # skip brain tick if all BTC feeds > 3s old

# Vol regime gates — skip if market too quiet or too violent
SIGMA_MIN_TRADE = 0.0005   # relaxed from 0.0011 (observed range 0.0003-0.0009)
SIGMA_MAX_TRADE = 0.0060   # relaxed from 0.0042

# Dynamic-alpha exit policy:
# exit_target = max(entry + MIN_BUF, entry + alpha(T) * (fv - entry))
# alpha(T) = ALPHA_BASE * (T / T_entry)  — decays as time runs out
# This replaces the old static 20% profit requirement which was way too strict
EXIT_ALPHA_BASE = float(os.getenv("EXIT_ALPHA_BASE", "0.55"))
EXIT_MIN_BUF    = float(os.getenv("EXIT_MIN_BUF", "0.01"))    # always require at least 1c profit
EXIT_HARD_FLOOR_T = float(os.getenv("EXIT_HARD_FLOOR_T", "15"))  # below this T, allow any profitable exit


# ── Session PnL & Circuit Breaker ──────────────────────────────────────────
SESSION_PNL:              float = 0.0
CONSEC_LOSSES:            int   = 0
LAST_WINDOW_PNL:          float = 0.0
SESSION_LOSS_LIMIT:       float = float(os.getenv("SESSION_LOSS_LIMIT", "-50.0"))
MAX_CONSEC_LOSSES:        int   = int(os.getenv("MAX_CONSEC_LOSSES", "4"))
CIRCUIT_BREAKER_ACTIVE:   bool  = False
CIRCUIT_BREAKER_UNTIL:    float = 0.0
CIRCUIT_BREAKER_PAUSE_S:  int   = int(os.getenv("CIRCUIT_BREAKER_PAUSE_S", "1800"))

# Consecutive loss guard (RISK.md): halve Kelly for N trades after streak
LOSS_STREAK_KELLY_PENALTY: float = 1.0   # 1.0 = no penalty, 0.5 = halved
LOSS_STREAK_TRADES_LEFT:   int   = 0     # trades remaining under penalty

# ── Regime-aware risk budget multipliers ──────────────────────────────────
RISK_A_BASE = float(os.getenv("RISK_A_BASE", "0.35"))
RISK_B_BASE = float(os.getenv("RISK_B_BASE", "0.50"))
RISK_A_MIN  = float(os.getenv("RISK_A_MIN",  "0.15"))
RISK_A_MAX  = float(os.getenv("RISK_A_MAX",  "0.55"))
RISK_B_MIN  = float(os.getenv("RISK_B_MIN",  "0.20"))
RISK_B_MAX  = float(os.getenv("RISK_B_MAX",  "0.80"))

# ── Rolling equity / drawdown risk contraction ─────────────────────────
EQUITY_PEAK: float = 0.0
EQUITY_LAST: float = 0.0
DRAWDOWN:    float = 0.0
DD_START = float(os.getenv("DD_START", "0.03"))   # start contracting at 3% drawdown
DD_MAX   = float(os.getenv("DD_MAX",   "0.12"))   # max contraction at 12% drawdown
DD_FLOOR = float(os.getenv("DD_FLOOR", "0.35"))   # minimum multiplier at max drawdown

# ── Rolling Sharpe throttle (20-window) ──────────────────────────────
ROLL_SHARPE_N  = int(os.getenv("ROLL_SHARPE_N", "20"))
SHARPE_SOFT    = float(os.getenv("SHARPE_SOFT", "-0.15"))
SHARPE_HARD    = float(os.getenv("SHARPE_HARD", "-0.40"))
SHARPE_MULT_MIN = float(os.getenv("SHARPE_MULT_MIN", "0.25"))
SHARPE_MULT_MAX = float(os.getenv("SHARPE_MULT_MAX", "1.25"))

WINDOW_RETURNS: deque  = deque(maxlen=200)
WIN_REALIZED_START     = None
ROLL_SHARPE: float     = 0.0
SHARPE_MULT: float     = 1.0
SHARPE_PAUSED: bool    = False

# ── Hard stop day kill-switch ─────────────────────────────────────
DAY_LOSS_LIMIT = float(os.getenv("DAY_LOSS_LIMIT", "-150.0"))
DAY_DD_MAX     = float(os.getenv("DAY_DD_MAX", "0.10"))
DAY_PAUSE_SEC  = int(os.getenv("DAY_PAUSE_SEC", "86400"))
DAY_KILL_ACTIVE:    bool  = False
DAY_KILL_UNTIL_MS:  int   = 0
DAY_KEY:            str   = None
DAY_EQUITY_START:   float = 0.0
DAY_EQUITY_PEAK:    float = 0.0


def record_pnl(proceeds: float, cost: float) -> None:
    global SESSION_PNL, CONSEC_LOSSES, LAST_WINDOW_PNL
    global CIRCUIT_BREAKER_ACTIVE, CIRCUIT_BREAKER_UNTIL
    global LOSS_STREAK_KELLY_PENALTY, LOSS_STREAK_TRADES_LEFT

    pnl = proceeds - cost
    SESSION_PNL += pnl
    LAST_WINDOW_PNL = pnl

    if pnl < -0.01:
        CONSEC_LOSSES += 1
    elif pnl > 0.01:
        CONSEC_LOSSES = 0
        if LOSS_STREAK_TRADES_LEFT > 0:
            LOSS_STREAK_TRADES_LEFT -= 1
            if LOSS_STREAK_TRADES_LEFT == 0:
                LOSS_STREAK_KELLY_PENALTY = 1.0
                logger.info("LOSS_STREAK penalty expired — Kelly restored to 1.0×")

    logger.info(
        f"PNL: trade={pnl:+.2f} session={SESSION_PNL:+.2f} streak={CONSEC_LOSSES} "
        f"kelly_pen={LOSS_STREAK_KELLY_PENALTY:.2f}({LOSS_STREAK_TRADES_LEFT} left)"
    )

    # Consecutive loss guard: halve Kelly for 5 trades (softer than circuit breaker)
    if CONSEC_LOSSES >= 3 and LOSS_STREAK_TRADES_LEFT == 0:
        LOSS_STREAK_KELLY_PENALTY = 0.5
        LOSS_STREAK_TRADES_LEFT = 5
        logger.warning(
            f"LOSS_STREAK: {CONSEC_LOSSES} consecutive losses — "
            f"Kelly halved for next 5 trades"
        )

    if SESSION_PNL < SESSION_LOSS_LIMIT or CONSEC_LOSSES >= MAX_CONSEC_LOSSES:
        CIRCUIT_BREAKER_ACTIVE = True
        CIRCUIT_BREAKER_UNTIL  = ms_now() + CIRCUIT_BREAKER_PAUSE_S * 1000
        logger.critical(
            f"CIRCUIT BREAKER: PnL={SESSION_PNL:.2f} consec={CONSEC_LOSSES} "
            f"pausing {CIRCUIT_BREAKER_PAUSE_S}s"
        )


def get_available_usdc_balance() -> float:
    """Available USDC at the collateral level (no double-counting)."""
    try:
        if client is None:
            logger.warning("Balance check: client is None")
            return 0.0
        resp = client.get_balance_allowance(
            params=BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
        )
        balance   = float(resp.get("balance",   0) or 0) / 1e6  # USDC has 6 decimals
        allowance = float(resp.get("allowance", 0) or 0) / 1e6
        logger.info(f"BALANCE_CHECK: balance=${balance:.2f} allowance=${allowance:.2f}")
        if balance > 0 and allowance == 0:
            # Allowance field is contract-level, not user-available — use balance
            return balance
        return max(0.0, min(balance, allowance))
    except Exception as e:
        logger.warning(f"Balance fetch failed: {e}")
        return 0.0


def portfolio_liquidation_value(up_bid: float, dn_bid: float) -> float:
    """Conservative liquidation value (sell at bid, net of fees)."""
    up_qty = float(POS_UP.inventory)
    dn_qty = float(POS_DOWN.inventory)
    v_up = up_qty * max(0.0, float(up_bid) - fee_per_share(float(up_bid)))
    v_dn = dn_qty * max(0.0, float(dn_bid) - fee_per_share(float(dn_bid)))
    return float(v_up + v_dn)


def portfolio_gross_shares() -> float:
    return float(abs(POS_UP.inventory) + abs(POS_DOWN.inventory))


def portfolio_net_shares() -> float:
    # + means net UP, - means net DOWN
    return float(POS_UP.inventory - POS_DOWN.inventory)


def net_sell_after_fee(bid: float) -> float:
    b = float(bid)
    return max(0.0, b - fee_per_share(b))


# Flip escape hatch state: {side: first_breach_monotonic_ms} — hysteresis to avoid flicker
_FLIP_BREACH_TS: dict = {"UP": 0, "DOWN": 0}
_FLIP_P_THRESHOLD = 0.48        # below this = model says losing (with margin)
_FLIP_PERSIST_MS  = 200          # must persist for 200ms before firing

def _mono_ms() -> int:
    """Monotonic time in milliseconds — immune to NTP clock adjustments."""
    return int(time.monotonic() * 1000)


def early_sell_profit_gate(sec_remaining: float, bid: float, entry_price: float,
                           p_cone: float = 0.5, side: str = "UP",
                           T_entry: float = 0.0):
    """
    Dynamic-alpha exit gate (profit-taking exits only).

    Returns (allowed, net_bid, exit_target).

    Policy:
      p_side = side-adjusted probability
      edge = max(0, p_side - entry)           -- explicit clamp for fv < entry
      alpha(T) = ALPHA_BASE * (T / T_entry)   -- decays with time
      exit_target = max(entry + MIN_BUF, entry + alpha(T) * edge)

    Momentum flip escape hatch (with hysteresis):
      if p_side < 0.48 for >= 200ms AND net > entry + MIN_BUF → exit immediately

    Note: stop-loss exits (prob_stop_loss, trailing_stop, endgame_strong_loss)
    bypass this gate entirely — they're handled upstream.
    """
    net_bid = net_sell_after_fee(float(bid))
    entry = max(0.01, float(entry_price))

    # Fair value of this specific token (side-adjusted)
    p_side = p_cone if side == "UP" else (1.0 - p_cone)

    # Explicit edge clamp: if fv dropped below entry, edge is zero
    # (prevents nonsensical negative alpha targets)
    edge = max(0.0, p_side - entry)

    # Compute time-decaying alpha
    T = float(sec_remaining)
    T_e = float(T_entry) if T_entry and T_entry > 0 else 230.0  # default ~3:50
    alpha_time = EXIT_ALPHA_BASE * max(0.0, min(1.0, T / T_e))

    # Dynamic exit target
    exit_target = max(entry + EXIT_MIN_BUF, entry + alpha_time * edge)

    # ── Hard floor: very late in window, allow any profitable exit ──
    if T <= EXIT_HARD_FLOOR_T and net_bid > entry + EXIT_MIN_BUF:
        return True, net_bid, exit_target

    # ── Momentum flip escape hatch (with hysteresis) ──
    # Model says this side is now losing (p < 0.48) but we're still in profit.
    # Require persistence to avoid flicker: p must stay < 0.48 for 200ms.
    # Uses monotonic clock — immune to NTP adjustments.
    _now_mono = _mono_ms()
    if p_side < _FLIP_P_THRESHOLD and net_bid > entry + EXIT_MIN_BUF:
        if _FLIP_BREACH_TS[side] == 0:
            _FLIP_BREACH_TS[side] = _now_mono   # first breach: start timer
        elif _now_mono - _FLIP_BREACH_TS[side] >= _FLIP_PERSIST_MS:
            _FLIP_BREACH_TS[side] = 0            # reset after firing
            return True, net_bid, exit_target
    else:
        # p recovered above threshold or not profitable → reset
        _FLIP_BREACH_TS[side] = 0

    # ── Dynamic alpha gate ──
    return (net_bid >= exit_target), net_bid, exit_target


# ─────────────────────────────────────────────────────────────
# Risk Telemetry
# ─────────────────────────────────────────────────────────────

def log_risk_state(
    collateral: float,
    liq: float,
    risk_budget: float,
    gross_before: float,
    net_before: float,
    new_spend: float,
    gross_after: float,
    capped_size: int,
    limit_price: float,
    side: str,
):
    logger.info(
        "RISK_STATE | "
        f"collat=${collateral:.2f} "
        f"liq=${liq:.2f} "
        f"budget=${risk_budget:.2f} | "
        f"gross_before={gross_before:.2f} "
        f"net_before={net_before:.2f} | "
        f"new_spend=${new_spend:.2f} "
        f"gross_after=${gross_after:.2f} | "
        f"size={capped_size}@{limit_price:.2f} "
        f"side={side}"
    )


def settle_window():
    """
    Settlement reconciliation at window end.
    Determines winner (BTC close vs strike), credits winning shares at $1.00,
    debits losing shares at $0.00, and records PnL.
    Must be called BEFORE resetting FIFO lots.
    """
    up_qty = float(POS_UP.inventory)
    dn_qty = float(POS_DOWN.inventory)

    if up_qty <= 0 and dn_qty <= 0:
        logger.info("SETTLE | No inventory to settle.")
        return 0.0

    strike = float(STATE.open_price) if np.isfinite(STATE.open_price) else None
    if strike is None:
        logger.warning("SETTLE | No strike available — cannot determine winner. Writing off inventory as loss.")
        up_cost = up_qty * float(POS_UP.avg_entry_price) if up_qty > 0 else 0.0
        dn_cost = dn_qty * float(POS_DOWN.avg_entry_price) if dn_qty > 0 else 0.0
        total_cost = up_cost + dn_cost
        if total_cost > 0:
            record_pnl(proceeds=0.0, cost=total_cost)
        logger.info(f"SETTLE | UNKNOWN: wrote off ${total_cost:.2f} (no strike)")
        return -total_cost

    # Get close price — priority: 1) Official API  2) RTDS  3) Coinbase
    close_price = None
    close_src = "unknown"

    # 1) Try official crypto-price API (authoritative resolution price)
    try:
        _prev_start_unix = get_window_start_unix(offset=-1)
        _prev_start = datetime.fromtimestamp(_prev_start_unix, tz=POLYMARKET_TIMEZONE)
        _prev_end = _prev_start + timedelta(minutes=5)
        _settle_data = fetch_crypto_price(_prev_start, _prev_end)
        if _settle_data and _settle_data.get("completed", False) and "closePrice" in _settle_data:
            close_price = float(_settle_data["closePrice"])
            close_src = "official"
            # Also update strike from official source if we had a fuzzy capture
            if "openPrice" in _settle_data and STATE.strike_type != "OFFICIAL":
                _official_strike = float(_settle_data["openPrice"])
                logger.info(
                    f"SETTLE | Strike correction: {strike:.2f} → {_official_strike:.2f} "
                    f"(was {STATE.strike_type}, now OFFICIAL)"
                )
                strike = _official_strike
            logger.info(f"SETTLE | Official close=${close_price:.2f} (completed=True)")
    except Exception as e:
        logger.warning(f"SETTLE | crypto-price API failed: {e}")

    # 2) Fallback: RTDS oracle
    if close_price is None:
        if RTDS.price and RTDS.age_ms() < 30_000:
            close_price = float(RTDS.price)
            close_src = "rtds"

    # 3) Fallback: Coinbase
    if close_price is None:
        if STATE.btc_ts_ms > 0 and not np.isnan(STATE.btc_price) and STATE.btc_price > 0:
            close_price = float(STATE.btc_price)
            close_src = "coinbase"

    if close_price is None or close_price <= 0:
        logger.warning("SETTLE | No close price available — cannot determine winner.")
        up_cost = up_qty * float(POS_UP.avg_entry_price) if up_qty > 0 else 0.0
        dn_cost = dn_qty * float(POS_DOWN.avg_entry_price) if dn_qty > 0 else 0.0
        total_cost = up_cost + dn_cost
        if total_cost > 0:
            record_pnl(proceeds=0.0, cost=total_cost)
        logger.info(f"SETTLE | UNKNOWN: wrote off ${total_cost:.2f} (no close price)")
        return -total_cost

    # Determine winner
    up_wins = close_price >= strike  # UP = BTC went up from open

    # Calculate settlement PnL
    # Winning shares settle at $1.00, losing shares settle at $0.00
    up_entry_cost = up_qty * float(POS_UP.avg_entry_price) if up_qty > 0 else 0.0
    dn_entry_cost = dn_qty * float(POS_DOWN.avg_entry_price) if dn_qty > 0 else 0.0

    if up_wins:
        # UP shares pay $1.00 each, DOWN shares pay $0.00
        up_proceeds = up_qty * 1.0
        dn_proceeds = 0.0
        winner_str = "UP"
    else:
        # DOWN shares pay $1.00 each, UP shares pay $0.00
        up_proceeds = 0.0
        dn_proceeds = dn_qty * 1.0
        winner_str = "DOWN"

    total_proceeds = up_proceeds + dn_proceeds
    total_cost = up_entry_cost + dn_entry_cost
    settlement_pnl = total_proceeds - total_cost

    # Record via PnL system (updates SESSION_PNL, streaks, circuit breaker)
    if total_cost > 0 or total_proceeds > 0:
        record_pnl(proceeds=total_proceeds, cost=total_cost)

    logger.info(
        f"SETTLE | winner={winner_str} close=${close_price:.2f}({close_src}) strike=${strike:.2f} | "
        f"UP: {up_qty:.0f}sh@{POS_UP.avg_entry_price:.2f}→${up_proceeds:.2f} | "
        f"DN: {dn_qty:.0f}sh@{POS_DOWN.avg_entry_price:.2f}→${dn_proceeds:.2f} | "
        f"pnl={settlement_pnl:+.2f}"
    )

    # Record SPRT outcome for edge validation (Brier score vs market)
    if total_cost > 0:
        won = settlement_pnl > 0
        # Record per-side Brier observations
        for _sprt_side, _sprt_won in [("UP", up_wins), ("DOWN", not up_wins)]:
            _sp_model = ENTRY_P_CONE.get(_sprt_side)
            _sp_market = ENTRY_P_MARKET.get(_sprt_side)
            if _sp_model is not None and _sp_market is not None:
                # p_model/p_market are for the UP side; adjust for DOWN
                if _sprt_side == "DOWN":
                    _sp_model = 1.0 - _sp_model if ENTRY_P_CONE.get("DOWN") is None else _sp_model
                    _sp_market = 1.0 - _sp_market if ENTRY_P_MARKET.get("DOWN") is None else _sp_market
                SPRT.record(p_model=_sp_model, p_market=_sp_market, outcome=_sprt_won)
        logger.info(
            f"SPRT: recorded {'WIN' if won else 'LOSS'} → {SPRT.decision} "
            f"(lr={SPRT._log_lr:.3f} brier_m={SPRT.avg_brier_model:.4f} "
            f"brier_mkt={SPRT.avg_brier_market:.4f})"
        )

    return settlement_pnl


def regime_risk_multipliers():
    """
    Returns (a_mult, b_mult, why).
    a applies to collateral term, b applies to liquidation-credit term.
    HIGH_VOL / high flip_rate => shrink.  CALM + low flip => loosen.
    """
    label = getattr(REGIME, "label", "NORMAL")
    flip = float(getattr(REGIME, "flip_rate", 0.0) or 0.0)
    intensity = float(getattr(REGIME.vol_detector, "intensity", 0.0) or 0.0) if hasattr(REGIME, "vol_detector") else 0.0

    a_mult = 1.0
    b_mult = 1.0
    why = [label]

    # Flip-rate penalty
    if flip >= 0.10:
        a_mult *= 0.65; b_mult *= 0.70; why.append(f"flip>=0.10({flip:.2f})")
    elif flip >= 0.06:
        a_mult *= 0.80; b_mult *= 0.85; why.append(f"flip>=0.06({flip:.2f})")
    elif flip <= 0.02:
        a_mult *= 1.10; b_mult *= 1.05; why.append(f"flip<=0.02({flip:.2f})")

    # Vol regime penalty
    if label in ("HIGH_VOL", "VOL_EVENT"):
        a_mult *= 0.75; b_mult *= 0.80; why.append("vol_regime")
    elif label == "TRANSITION":
        a_mult *= 0.85; b_mult *= 0.90; why.append("transition")
    elif label == "CALM":
        a_mult *= 1.10; b_mult *= 1.05; why.append("calm")

    # Intensity penalty
    if intensity >= 1.0:
        a_mult *= 0.85; b_mult *= 0.90; why.append(f"intensity>=1.0({intensity:.2f})")

    return a_mult, b_mult, "|".join(why)


def drawdown_risk_multiplier(equity: float) -> float:
    """
    Returns a multiplier in [DD_FLOOR, 1.0] based on drawdown from peak.
    Contraction starts at DD_START and reaches DD_FLOOR at DD_MAX.
    """
    global EQUITY_PEAK, EQUITY_LAST, DRAWDOWN

    EQUITY_LAST = float(equity)
    if EQUITY_PEAK <= 0:
        EQUITY_PEAK = float(equity)
    if equity > EQUITY_PEAK:
        EQUITY_PEAK = float(equity)

    dd = 0.0
    if EQUITY_PEAK > 0:
        dd = max(0.0, (EQUITY_PEAK - equity) / EQUITY_PEAK)
    DRAWDOWN = dd

    if dd <= DD_START:
        return 1.0
    if dd >= DD_MAX:
        return float(DD_FLOOR)

    t = (dd - DD_START) / max(1e-9, (DD_MAX - DD_START))
    return float(1.0 - t * (1.0 - DD_FLOOR))


def _compute_roll_sharpe(rets) -> float:
    if len(rets) < 5:
        return 0.0
    m = sum(rets) / len(rets)
    var = sum((x - m) ** 2 for x in rets) / max(1, (len(rets) - 1))
    sd = math.sqrt(var)
    if sd < 1e-12:
        return 0.0
    return m / sd


def _sharpe_to_mult(sh: float) -> float:
    if sh >= 0.25:
        return SHARPE_MULT_MAX
    if sh >= 0.0:
        t = sh / 0.25
        return 1.0 + t * (SHARPE_MULT_MAX - 1.0)
    if sh >= SHARPE_SOFT:
        t = (sh - SHARPE_SOFT) / max(1e-9, (0.0 - SHARPE_SOFT))
        return 0.60 + t * (1.0 - 0.60)
    if sh <= SHARPE_HARD:
        return SHARPE_MULT_MIN
    t = (sh - SHARPE_HARD) / max(1e-9, (SHARPE_SOFT - SHARPE_HARD))
    return SHARPE_MULT_MIN + t * (0.60 - SHARPE_MULT_MIN)


def _day_key_now():
    return datetime.now(POLYMARKET_TIMEZONE).strftime("%Y-%m-%d")


def update_day_kill_switch(equity: float):
    global DAY_KEY, DAY_EQUITY_START, DAY_EQUITY_PEAK
    global DAY_KILL_ACTIVE, DAY_KILL_UNTIL_MS

    key = _day_key_now()
    if DAY_KEY != key:
        DAY_KEY = key
        DAY_EQUITY_START = float(equity)
        DAY_EQUITY_PEAK  = float(equity)
        DAY_KILL_ACTIVE = False
        DAY_KILL_UNTIL_MS = 0
        logger.info(f"DAY_RESET | key={DAY_KEY} start_eq={DAY_EQUITY_START:.2f}")

    if equity > DAY_EQUITY_PEAK:
        DAY_EQUITY_PEAK = float(equity)

    day_pnl = equity - DAY_EQUITY_START
    day_dd = 0.0
    if DAY_EQUITY_PEAK > 0:
        day_dd = max(0.0, (DAY_EQUITY_PEAK - equity) / DAY_EQUITY_PEAK)

    if (day_pnl <= DAY_LOSS_LIMIT) or (day_dd >= DAY_DD_MAX):
        if not DAY_KILL_ACTIVE:
            DAY_KILL_ACTIVE = True
            DAY_KILL_UNTIL_MS = ms_now() + DAY_PAUSE_SEC * 1000
            logger.critical(
                f"DAY_KILL | pnl={day_pnl:.2f} dd={day_dd*100:.1f}% "
                f"limit={DAY_LOSS_LIMIT:.2f}/{DAY_DD_MAX*100:.1f}% pause={DAY_PAUSE_SEC}s"
            )

    return day_pnl, day_dd


# ════════════════════════════════════════════════════════════════════════════
# STATE & DATA MODELS
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class BookSnapshot:
    token_id:    str
    best_bid:    float = 0.0
    best_ask:    float = 1.0
    spread:      float = 1.0
    imbalance:   float = 0.5
    bid_size:    float = 0.0
    ask_size:    float = 0.0
    source:      str   = "unknown"
    last_update: float = 0.0


class BookCache:
    def __init__(self):
        self._books: dict = {}
        self._lock  = threading.Lock()

    def update(self, token_id: str, snap: BookSnapshot) -> None:
        with self._lock:
            self._books[token_id] = snap

    def get(self, token_id: str) -> Optional[BookSnapshot]:
        with self._lock:
            return self._books.get(token_id)

    def delete(self, token_id: str) -> None:
        with self._lock:
            self._books.pop(token_id, None)

    def is_fresh(self, token_id: str, max_age_ms: float = 1500) -> bool:
        snap = self.get(token_id)
        if not snap:
            return False
        return (ms_now() - snap.last_update) < max_age_ms


BOOK_CACHE = BookCache()


class MarketState:
    # BTC price sources
    btc_price:     float = np.nan   # best available BTC price for sigma
    btc_ts_ms:     int   = 0        # timestamp of btc_price

    # RTDS oracle — primary trading price (from chainlink_rtds.py RTDS singleton)
    # Accessed via RTDS.price / RTDS.recv_ts_ms

    # Chainlink on-chain (emergency fallback)
    onchain_price: float = np.nan
    onchain_ts_ms: int   = 0

    # Market
    open_price:    float = np.nan
    sec_remaining: float = np.nan
    sigma_1m:      float = 0.0009
    sigma_slow:    float = 0.0003
    sigma_fast:    float = 0.0003
    sigma_w:       float = 0.5
    strike_type:   Optional[str] = None

    # WebSocket watchdog timestamps
    rtds_last_msg_ts:    float = 0.0
    poly_last_msg_ts:    float = 0.0
    trading_suspended:   bool  = False


STATE        = MarketState()
PERSIST      = PersistState()
LAG_ADAPTIVE = LagAdaptiveZ()
REGIME       = RegimeClassifier()
JUMP_DETECTOR = JumpDetector(dt_seconds=1.0)
TAIL_RISK    = TailRiskGuard()
FLOW         = FlowTracker()
PV_TRACKER   = ParticipationVelocity(PVConfig())
BIPOWER      = BipowerJumpFilter(BipowerConfig())
MOMENTUM     = MomentumEngine(MomentumConfig())
ENDGAME      = EndgameManager(EndgameConfig())     # legacy — kept for fallback
EXIT_MGR     = ExitManager(ExitConfig())           # legacy — kept for fallback
POS_MONITOR  = PositionMonitor(PositionMonitorConfig())
ADAPTIVE_EXEC: AdaptiveExecutor = None  # initialized after client is created
PNL_TRACKER  = PnLTracker()
FILL_PROB    = FillProbModel()
EXPIRY       = ExpiryScaler()
SPRT         = SPRTValidator()
VWAP         = VWAPTracker()
GOLDSKY      = GoldskyAnalytics()
L2_TRACKER   = L2Tracker()
LATEST_DEBUG: dict = {}



# ─────────────────────────────────────────────────────────────
# Directional Coverage Controller
# ─────────────────────────────────────────────────────────────

class DirectionalCoverageController:
    def __init__(self, target_up=0.30, target_dn=0.20, window=40):
        self.window = window
        self.target_up = target_up
        self.target_dn = target_dn
        self.up_hist = deque(maxlen=window)
        self.dn_hist = deque(maxlen=window)

    def record_window(self, traded_up: bool, traded_dn: bool):
        self.up_hist.append(1 if traded_up else 0)
        self.dn_hist.append(1 if traded_dn else 0)

    def coverage_up(self):
        if not self.up_hist:
            return 0.0
        return sum(self.up_hist) / len(self.up_hist)

    def coverage_dn(self):
        if not self.dn_hist:
            return 0.0
        return sum(self.dn_hist) / len(self.dn_hist)

    def z_bias(self, side: str):
        if side == "UP":
            coverage = self.coverage_up()
            target = self.target_up
        else:
            coverage = self.coverage_dn()
            target = self.target_dn
        error = coverage - target
        bias = error * 0.20
        return max(-0.10, min(0.15, bias))


COVERAGE = DirectionalCoverageController(target_up=0.30, target_dn=0.20, window=40)
TRADED_UP_THIS_WINDOW: bool = False
TRADED_DN_THIS_WINDOW: bool = False
TRADES_THIS_WINDOW: int = 0          # per-window trade counter
# SPRT Brier tracking: entry-time probabilities (model + market) per side
ENTRY_P_CONE: dict = {"UP": None, "DOWN": None}    # cone p at first fill
ENTRY_P_MARKET: dict = {"UP": None, "DOWN": None}  # market mid at first fill
ENTRY_T_SEC: dict = {"UP": None, "DOWN": None}      # seconds remaining at entry
MAX_TRADES_PER_WINDOW: int = 3       # cap: most edge in 1-2 bursts
WINDOW_BANKROLL: float = 0.0  # remaining per-window BUY budget (USD)

# Market tokens
MARKET_ID     = ""
UP_TOKEN_ID   = ""
DOWN_TOKEN_ID = ""

POLY_STATE = {}  # keyed by token_id (populated at runtime)

# Official Polymarket crypto-price data for current window
# Cached after first successful fetch; used for strike + settlement
WINDOW_CRYPTO_PRICE: Optional[dict] = None  # {"openPrice", "closePrice", "completed"}

# Lag detection
LAST_BTC_MOVE_TS: int   = 0
LAST_BTC_PRICE:   float = np.nan
LAST_BTC_DIR:     int   = 0

# Book management
BOOK_RECONNECT_FLAG:          bool = False
REST_SEED_INFLIGHT:           bool = False
LAST_REST_SEED_MS:            int  = 0
REST_SEED_COOLDOWN_MS:        int  = 5000
REST_SEED_EMPTY_COOLDOWN_MS:  int  = 30000
LAST_REST_SEED_HAD_QUOTES:    bool = False
LAST_STALE_BOOK_LOG_MS:       int  = 0
LAST_NON_WS_BOOK_LOG_MS:      int  = 0
LIQUIDITY_TIMEOUT_THIS_WINDOW: bool = False

# P-cone slope tracking (structural decay protection)
_P_CONE_HISTORY: deque = deque(maxlen=20)

# ── Confirmation-Based Execution Infrastructure ──────────────────────────
import hashlib

RECENT_FILLS: deque = deque(maxlen=2000)  # WS fill buffer (truth source)
PENDING_LOCKS: Dict[tuple, int] = {}      # (token_id, side) -> until_ms


def order_fingerprint(token_id: str, side: str, price: float, size: float) -> str:
    """Idempotency key for an order intent."""
    payload = {"t": token_id, "s": side, "p": round(price, 2), "q": int(size)}
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def lock_side(token_id: str, side: str, ttl_ms: int = 1200) -> None:
    """Lock a token+side to prevent duplicate orders."""
    PENDING_LOCKS[(token_id, side)] = ms_now() + ttl_ms


def is_locked(token_id: str, side: str) -> bool:
    """Check if a token+side is locked (pending order may be live)."""
    return ms_now() < PENDING_LOCKS.get((token_id, side), 0)


def find_fill_match(token_id: str, price: float, now_ms: int,
                    window_ms: int = 1200, price_tol: float = 0.02,
                    side: str = None, expected_size: float = 0.0) -> Optional[dict]:
    """Search RECENT_FILLS for a matching fill within time+price+side+size tolerance."""
    cutoff = now_ms - window_ms
    for f in reversed(RECENT_FILLS):
        if f["ts_ms"] < cutoff:
            break
        if f["token_id"] != token_id:
            continue
        if abs(f["price"] - price) > price_tol:
            continue
        # Side must match if provided
        if side and f.get("side") != side:
            continue
        # Size proximity: within 25% or exact int match
        if expected_size > 0:
            f_size = float(f.get("size", 0))
            if abs(f_size - expected_size) / max(1e-9, expected_size) > 0.25:
                continue
        return f
    return None


def dynamic_z_min(regime_label: str, sigma: float) -> float:
    """Regime-aware Z threshold. Monte Carlo sweet spot = 0.75."""
    if regime_label in ("ADVERSARIAL",):
        return 2.5  # effectively disables entries
    if regime_label == "HIGH_VOL":
        return 1.2
    if sigma > 0.0008:
        return 1.0
    return 0.75  # default: moderate participation


def flow_size_scale(flow: float) -> float:
    """Contra flow reduces size, never blocks. |flow|=0→1.0, =0.9→0.28."""
    return max(0.25, min(1.0, 1.0 - abs(flow) * 0.8))


# ── Position State ──────────────────────────────────────────────────────────

# ── Execution lock (replaces POSITION.locked) ─────────────────────────────
EXEC_LOCKED: bool = False
EXEC_LOCK_MS: int = 0

def exec_lock(ttl_ms: int = 1500):
    global EXEC_LOCKED, EXEC_LOCK_MS
    EXEC_LOCKED = True
    EXEC_LOCK_MS = ms_now() + ttl_ms

def exec_unlock():
    global EXEC_LOCKED
    EXEC_LOCKED = False

def exec_lock_expired() -> bool:
    return ms_now() >= EXEC_LOCK_MS

PENDING_FLIP: dict = {
    "active": False, "side": None, "token_id": None,
    "limit": None, "size": 0.0, "expires_ms": 0,
}

PENDING_FLIP_INTENT: dict = {
    "active": False, "buy_token_id": None, "buy_side": None,
    "buy_limit": None, "buy_size": 0.0, "expires_ms": 0,
}

LAST_FIRE_SNAPSHOT: dict = {}

# ── FIFO Lot Accounting (position.py) ──────────────────────────────────────
POS_UP   = Position("UP", lot_method="FIFO")
POS_DOWN = Position("DOWN", lot_method="FIFO")
PORTFOLIO = Portfolio(POS_UP, POS_DOWN)
IADL = InventoryDecisionLayer(InventoryLayerConfig(
    reversal_z=2.2,
    reversal_conf=0.88,
    reversal_cooldown_sec=30,
    late_t_sec=75,
    late_reversal_z=3.0,
    late_reversal_conf=0.93,
))
TELEMETRY = TelemetryEmitter(TelemetryConfig(jsonl_path="logs/pnl_telemetry.jsonl"))


# ─────────────────────────────────────────────────────────────
# Portfolio Position View (replaces PositionState everywhere)
# ─────────────────────────────────────────────────────────────

@dataclass
class PortfolioPos:
    up: float
    dn: float
    total: float
    dominant_side: Optional[str]      # "UP" or "DOWN"
    dominant_token_id: Optional[str]
    avg_cost_up: float
    avg_cost_dn: float


def get_portfolio_pos() -> PortfolioPos:
    up = float(POS_UP.inventory)
    dn = float(POS_DOWN.inventory)
    total = up + dn
    dom_side = None
    dom_tid = None
    if total > 0:
        dom_side = "UP" if up >= dn else "DOWN"
        dom_tid = UP_TOKEN_ID if dom_side == "UP" else DOWN_TOKEN_ID
    try:
        acu = float(POS_UP.avg_entry_incl_fee)  # true cost including entry fee
    except Exception:
        acu = 0.5
    try:
        acd = float(POS_DOWN.avg_entry_incl_fee)  # true cost including entry fee
    except Exception:
        acd = 0.5
    return PortfolioPos(
        up=up, dn=dn, total=total,
        dominant_side=dom_side, dominant_token_id=dom_tid,
        avg_cost_up=acu, avg_cost_dn=acd,
    )


# ── Dual-Timescale Convex Sigma (Elite Mode) ───────────────────────────────

class DualSigmaEstimator:
    """Blends slow (90s) and fast (8s) realized variance via convex weight.
    When impulse hits, sigma_fast spikes → w→1 → sigma_eff≈sigma_fast.
    During calm, w≈0.5 → sigma_eff stable."""

    def __init__(self, slow_window_s: float = 90.0, fast_window_s: float = 8.0):
        self.slow_window_s = slow_window_s
        self.fast_window_s = fast_window_s
        self.log_prices: deque = deque(maxlen=2000)
        self.timestamps: deque = deque(maxlen=2000)

    def add_price(self, price: float, ts_ms: int):
        if price > 0:
            self.log_prices.append(math.log(price))
            self.timestamps.append(ts_ms)

    def _rv(self, window_s: float) -> float:
        if len(self.log_prices) < 5 or len(self.timestamps) < 5:
            return 0.0
        cutoff = self.timestamps[-1] - int(window_s * 1000)
        # Find start index for this window
        arr = np.array(self.log_prices)
        ts_arr = np.array(self.timestamps)
        mask = ts_arr >= cutoff
        lps = arr[mask]
        if len(lps) < 3:
            return 0.0
        returns = np.diff(lps)
        if len(returns) < 2:
            return 0.0
        return float(np.sum(returns ** 2))

    def compute_sigma(self):
        """Returns (sigma_eff, sigma_slow, sigma_fast, w)."""
        if len(self.log_prices) < 5:
            return 0.0003, 0.0003, 0.0003, 0.5

        rv_slow = self._rv(self.slow_window_s)
        rv_fast = self._rv(self.fast_window_s)

        # Convert to per-minute log-vol
        sigma_slow = math.sqrt(max(0.0, rv_slow * (60.0 / self.slow_window_s)))
        sigma_fast = math.sqrt(max(0.0, rv_fast * (60.0 / self.fast_window_s)))

        # Convex weight: fast dominates when fast vol is high
        denom = sigma_fast + sigma_slow
        w = sigma_fast / denom if denom > 1e-10 else 0.5

        # Convex blend
        sigma_eff = sigma_slow + w * (sigma_fast - sigma_slow)

        # Hard floor / ceiling
        sigma_eff = max(0.00008, min(0.01, sigma_eff))

        return float(sigma_eff), float(sigma_slow), float(sigma_fast), float(w)

DUAL_SIGMA = DualSigmaEstimator(slow_window_s=90.0, fast_window_s=8.0)
SIGMA_HISTORY: deque = deque(maxlen=300)  # ~5 minutes at 1s updates

# ── FIRE Debounce ─────────────────────────────────────────────
MIN_FIRE_GAP_MS_IMPULSE = 700
MIN_FIRE_GAP_MS_DRIFT   = 1500
LAST_FIRE_TS: dict      = {}  # (token_id, order_side) -> ts_ms
ORDER_COOLDOWN: dict    = {}  # (token_id, order_side) -> until_ms


def fire_allowed(token_id: str, order_side: str, mode: str) -> bool:
    now = ms_now()
    key = (token_id, order_side)
    last = LAST_FIRE_TS.get(key, 0)
    gap = MIN_FIRE_GAP_MS_IMPULSE if mode == "impulse" else MIN_FIRE_GAP_MS_DRIFT
    if now - last < gap:
        return False
    LAST_FIRE_TS[key] = now
    return True

_LAST_SIGMA_TS:           float = 0.0
_SIGMA_UPDATE_INTERVAL_S: float = 1.0  # update every 1s (was 3s)


def _update_sigma(price: float, ts_ms: int) -> None:
    global _LAST_SIGMA_TS
    DUAL_SIGMA.add_price(price, ts_ms)

    now_s = ts_ms / 1000.0
    if now_s - _LAST_SIGMA_TS < _SIGMA_UPDATE_INTERVAL_S:
        return
    _LAST_SIGMA_TS = now_s

    sigma_eff, sigma_slow, sigma_fast, w = DUAL_SIGMA.compute_sigma()

    # ── Dynamic sigma floor ─────────────────────────────────────
    SIGMA_HISTORY.append(sigma_eff)
    if len(SIGMA_HISTORY) >= 30:
        dyn_floor = float(np.percentile(list(SIGMA_HISTORY), 20))
    else:
        dyn_floor = 0.00015

    # Regime inflation: if flip rate high, inflate floor
    _flip_rate = float(getattr(REGIME, "flip_rate", 0.0) or 0.0)
    if _flip_rate > 0.08:
        dyn_floor *= 1.5
    elif _flip_rate > 0.05:
        dyn_floor *= 1.25

    sigma_eff = max(sigma_eff, dyn_floor)
    if sigma_eff == dyn_floor:
        logger.debug(f"DYN_SIGMA_FLOOR applied: {dyn_floor:.6f}")

    STATE.sigma_1m = sigma_eff
    STATE.sigma_slow = sigma_slow
    STATE.sigma_fast = sigma_fast
    STATE.sigma_w = w
    update_sigma_history(sigma_eff)
    LATEST_DEBUG["sigma_floor"] = dyn_floor


# ════════════════════════════════════════════════════════════════════════════
# CHAINLINK ON-CHAIN (EMERGENCY FALLBACK)
# ════════════════════════════════════════════════════════════════════════════

AGGREGATOR_ADDR    = "0xc907E116054Ad103354f2D350FD2514433D57F6f"
CHAINLINK_DECIMALS = 8
AGGREGATOR_ABI     = [{
    "inputs": [], "name": "latestRoundData",
    "outputs": [
        {"name": "roundId", "type": "uint80"},
        {"name": "answer", "type": "int256"},
        {"name": "startedAt", "type": "uint256"},
        {"name": "updatedAt", "type": "uint256"},
        {"name": "answeredInRound", "type": "uint80"},
    ],
    "stateMutability": "view", "type": "function",
}]


def _build_w3() -> Web3:
    alchemy_key = os.getenv("ALCHEMY_API_KEY")
    rpc = (f"https://polygon-mainnet.g.alchemy.com/v2/{alchemy_key}"
           if alchemy_key else os.getenv("POLYGON_RPC", "https://polygon-rpc.com"))
    return Web3(Web3.HTTPProvider(rpc))


_W3 = _build_w3()
_CHAINLINK_CONTRACT = _W3.eth.contract(
    address=Web3.to_checksum_address(AGGREGATOR_ADDR), abi=AGGREGATOR_ABI,
)


def get_chainlink_btc_price() -> float:
    try:
        _, answer, _, _, _ = _CHAINLINK_CONTRACT.functions.latestRoundData().call()
        return answer / 10 ** CHAINLINK_DECIMALS
    except Exception as e:
        logger.error(f"Chainlink on-chain read error: {e}")
        return np.nan


async def chainlink_onchain_task(poll_interval_s: float = 30.0) -> None:
    """Emergency fallback: polls Chainlink on-chain every 30s."""
    loop = asyncio.get_running_loop()
    while True:
        try:
            price = await loop.run_in_executor(None, get_chainlink_btc_price)
            if not np.isnan(price):
                STATE.onchain_price = price
                STATE.onchain_ts_ms = ms_now()
        except Exception as e:
            logger.warning(f"On-chain oracle error: {e}")
        await asyncio.sleep(poll_interval_s)


# ════════════════════════════════════════════════════════════════════════════
# CLOB CLIENT
# ════════════════════════════════════════════════════════════════════════════

HOST           = "https://clob.polymarket.com"
PRIVATE_KEY    = os.getenv("CLOB_PRIVATE_KEY") or os.getenv("PK")
API_KEY        = os.getenv("CLOB_API_KEY")      or os.getenv("POLY_API_KEY")
API_SECRET     = os.getenv("CLOB_SECRET")       or os.getenv("POLY_SECRET")
API_PASSPHRASE = (os.getenv("CLOB_PASSPHRASE") or os.getenv("CLOB_PASS_PHRASE")
                  or os.getenv("POLY_PASSPHRASE"))
CLOB_FUNDER    = os.getenv("CLOB_FUNDER") or os.getenv("PROXY_FUNDER")

try:
    if not PRIVATE_KEY or PRIVATE_KEY == "YOUR_PRIVATE_KEY":
        raise ValueError("No private key in .env")

    if CLOB_FUNDER:
        h = CLOB_FUNDER[2:] if CLOB_FUNDER.startswith("0x") else CLOB_FUNDER
        if len(h) != 40:
            raise ValueError(f"CLOB_FUNDER must be 40 hex chars, got {len(h)}")

    clob_creds = None
    if API_KEY and API_SECRET and API_PASSPHRASE:
        clob_creds = ApiCreds(
            api_key=API_KEY, api_secret=API_SECRET, api_passphrase=API_PASSPHRASE,
        )

    client = ClobClient(
        host=HOST, key=PRIVATE_KEY, chain_id=137,
        signature_type=1, funder=CLOB_FUNDER, creds=clob_creds,  # 1 = EOA (creds derived for this type)
    )
    try:
        refreshed = client.create_or_derive_api_creds()
        if refreshed:
            client.set_api_creds(refreshed)
            logger.info("CLOB API creds refreshed.")
    except Exception as e:
        logger.warning(f"CLOB creds refresh failed: {e}")
except Exception as e:
    logger.error(f"Failed to init ClobClient: {e}")
    client = None

thread_pool:     ThreadPoolExecutor = ThreadPoolExecutor(max_workers=4)
execution_queue: asyncio.Queue      = asyncio.Queue()

# Initialize AdaptiveExecutor with client
if client is not None:
    ADAPTIVE_EXEC = AdaptiveExecutor(client)
    logger.info("AdaptiveExecutor initialized (maker routing available)")
else:
    logger.warning("AdaptiveExecutor NOT initialized (no client)")


# ════════════════════════════════════════════════════════════════════════════
# ORDER EXECUTION
# ════════════════════════════════════════════════════════════════════════════

def ensure_logs() -> None:
    path = "logs/trades.csv"
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow([
                "ts_ms", "side", "token_id", "limit_price", "size",
                "result", "order_id", "exec_ms", "retry_idx",
                "edge", "p_cone", "z", "sigma_1m",
                "oracle_source", "oracle_price",
                # ── Attribution columns (Phase 7) ──
                "execution_mode", "detail",
                "regime", "edge_target", "mode", "flow",
            ])


def _prime_tick_size_cache(token_id: str) -> None:
    if client is None:
        return
    try:
        client.get_tick_size(token_id)
        return
    except Exception:
        pass
    try:
        client.get_order_book(token_id)
        return
    except Exception:
        pass
    try:
        client._ClobClient__tick_sizes[token_id] = "0.01"
        client._ClobClient__tick_size_timestamps[token_id] = time.monotonic()
        logger.info(f"Injected fallback tick_size=0.01 for {token_id[:16]}…")
    except Exception as e:
        logger.error(f"tick_size inject failed: {e}")


def best_executable_price(
    signal_side: str, up_snap: BookSnapshot, dn_snap: BookSnapshot,
    tick_buffer: float = 0.02,
    allow_implied: bool = False,
) -> tuple:
    ILLIQUID = 0.50
    if signal_side == "UP":
        own, opp = up_snap, dn_snap
        own_tok, opp_tok = UP_TOKEN_ID, DOWN_TOKEN_ID
    else:
        own, opp = dn_snap, up_snap
        own_tok, opp_tok = DOWN_TOKEN_ID, UP_TOKEN_ID

    own_sp = own.best_ask - own.best_bid
    opp_sp = opp.best_ask - opp.best_bid

    # Direct route: BUY own token — only block if no real ask (default=1.0)
    if own.best_ask > 0 and own.best_ask < 1.0:
        d_exec = round(min(0.99, own.best_ask + tick_buffer), 2)
        direct = (d_exec, own_tok, "BUY", False, d_exec)
    else:
        direct = (float("inf"), own_tok, "BUY", False, 0.0)

    if not allow_implied:
        # Entry routing is BUY-only by default.
        return direct

    # Optional implied route: SELL opposite token.
    if (opp.best_bid > 0.02 and opp.best_bid < 0.98):
        o_exec = round(max(0.01, opp.best_bid - tick_buffer), 2)
        implied_eff = round(1.0 - o_exec, 4)
        implied = (implied_eff, opp_tok, "SELL", True, o_exec)
    else:
        implied = (float("inf"), opp_tok, "SELL", True, 0.0)

    return direct if direct[0] <= implied[0] else implied


# ─── 429 Rate Limit Protection ─────────────────────────────────────────────
RATE_LIMIT_UNTIL_MS: int = 0          # ms_now() when rate limit expires
RATE_LIMIT_COOLDOWN_MS: int = 2000    # 2 second cooldown after 429

def _check_rate_limit_response(resp, error_str: str = "") -> bool:
    """Check if response indicates 429. Returns True if rate limited."""
    global RATE_LIMIT_UNTIL_MS
    is_429 = False
    if isinstance(resp, dict):
        is_429 = resp.get("status", 0) == 429 or "429" in str(resp.get("errorMsg", ""))
    if "429" in error_str or "Too Many" in error_str:
        is_429 = True
    if is_429:
        RATE_LIMIT_UNTIL_MS = ms_now() + RATE_LIMIT_COOLDOWN_MS
        logger.error(f"RATE_LIMIT: 429 detected — suspending new orders until {RATE_LIMIT_COOLDOWN_MS}ms cooldown")
    return is_429

def is_rate_limited() -> bool:
    return ms_now() < RATE_LIMIT_UNTIL_MS


def _execute_ioc_sync(
    token_id: str, limit_price: float, size: float, retry_idx: int = 0
) -> Dict[str, Any]:
    """Clean IOC execution: create_and_post_order with OrderType.IOC.
    Always BUY side. Returns {ok, order_id, exec_ms, error}."""
    t0 = time.time()
    try:
        if client is None:
            return {"ok": False, "exec_ms": 0, "error": "No client", "retry_idx": retry_idx}

        args = OrderArgs(
            price=limit_price,
            size=size,
            side="BUY",
            token_id=token_id,
        )
        signed = client.create_order(
            args, options=PartialCreateOrderOptions(tick_size="0.01", neg_risk=False),
        )
        resp = client.post_order(signed, OrderType.FAK)
        exec_ms = int((time.time() - t0) * 1000)

        accepted = bool(resp) and resp.get("success", False)
        _check_rate_limit_response(resp)  # 429 detection

        # FAK success means accepted, not filled — verify actual fill
        ok = False
        oid = resp.get("orderID", "") if isinstance(resp, dict) else ""
        if accepted and oid:
            try:
                time.sleep(0.2)
                order_info = client.get_order(oid)
                if order_info:
                    matched = float(order_info.get("size_matched", 0) or 0)
                    ok = matched > 0
            except Exception as ve:
                logger.warning(f"IOC_VERIFY_ERR: {ve}")

        if ok:
            logger.info(
                f"IOC_HIT: {size}@{limit_price:.2f} exec_ms={exec_ms} "
                f"id={oid}"
            )
        else:
            err = resp.get("errorMsg", resp) if isinstance(resp, dict) else str(resp)
            logger.warning(f"IOC_MISS: exec_ms={exec_ms} accepted={accepted} err={err}")

        return {
            "ok": ok,
            "order_id": oid,
            "exec_ms": exec_ms,
            "error": "" if ok else str(resp.get("errorMsg", "") if isinstance(resp, dict) else ""),
            "retry_idx": retry_idx,
        }
    except Exception as e:
        exec_ms = int((time.time() - t0) * 1000)
        logger.error(f"IOC_CRASH: {e} exec_ms={exec_ms}")
        _check_rate_limit_response({}, str(e))  # 429 detection
        return {"ok": False, "order_id": "", "exec_ms": exec_ms,
                "error": str(e), "retry_idx": retry_idx}


def _execute_ioc_sync_side(
    token_id: str, limit_price: float, size: float,
    order_side: str, retry_idx: int = 0
) -> Dict[str, Any]:
    """IOC execution that supports BUY and SELL."""
    t0 = time.time()
    try:
        if client is None:
            return {"ok": False, "exec_ms": 0, "error": "No client", "retry_idx": retry_idx}

        side_const = BUY if order_side.upper() == "BUY" else SELL

        # Enforce CLOB $1 minimum
        import math as _m
        if size * limit_price < 1.0:
            size = max(size, _m.ceil(1.0 / max(0.01, limit_price)))

        args = OrderArgs(
            price=limit_price,
            size=size,
            side=side_const,
            token_id=token_id,
        )
        signed = client.create_order(
            args, options=PartialCreateOrderOptions(tick_size="0.01", neg_risk=False),
        )
        resp = client.post_order(signed, OrderType.FAK)
        _check_rate_limit_response(resp)
        exec_ms = int((time.time() - t0) * 1000)
        accepted = bool(resp) and resp.get("success", False)

        # FAK success means accepted, not filled — verify actual fill
        ok = False
        oid = resp.get("orderID", "") if isinstance(resp, dict) else ""
        if accepted and oid:
            try:
                time.sleep(0.2)
                order_info = client.get_order(oid)
                if order_info:
                    matched = float(order_info.get("size_matched", 0) or 0)
                    ok = matched > 0
            except Exception as ve:
                logger.warning(f"IOC_VERIFY_ERR({order_side}): {ve}")

        if ok:
            logger.info(f"IOC_HIT({order_side}): {size}@{limit_price:.2f} exec_ms={exec_ms}")
        else:
            err = resp.get("errorMsg", resp) if isinstance(resp, dict) else str(resp)
            logger.warning(f"IOC_MISS({order_side}): exec_ms={exec_ms} accepted={accepted} err={err}")

        return {
            "ok": ok,
            "order_id": oid,
            "exec_ms": exec_ms,
            "error": "" if ok else str(resp.get("errorMsg", "") if isinstance(resp, dict) else ""),
            "retry_idx": retry_idx,
        }
    except Exception as e:
        _check_rate_limit_response({}, str(e))
        exec_ms = int((time.time() - t0) * 1000)
        logger.error(f"IOC_CRASH({order_side}): {e} exec_ms={exec_ms}")
        return {"ok": False, "order_id": "", "exec_ms": exec_ms,
                "error": str(e), "retry_idx": retry_idx}


def _get_l2_top2(token_id: str):
    """
    Fetch top 2 levels via Polymarket REST book endpoint.
    Returns (bids, asks, ok) — ok=False on timeout/error (no stalling).
    """
    try:
        url = f"https://clob.polymarket.com/book?token_id={token_id}"
        r = requests.get(url, timeout=0.35)   # hard fast timeout
        r.raise_for_status()
        data = r.json()
        bids = [(float(x["price"]), float(x.get("size", 0.0))) for x in (data.get("bids") or [])[:2]]
        asks = [(float(x["price"]), float(x.get("size", 0.0))) for x in (data.get("asks") or [])[:2]]
        return bids, asks, True
    except Exception:
        return [], [], False


# ── Aggressive FOK Configuration ─────────────────────────────────
try:
    _fok_cap_env = float(os.getenv("FOK_HARD_CAP", "0.95"))
except Exception:
    _fok_cap_env = 0.86
FOK_HARD_CAP   = min(0.95, max(0.50, _fok_cap_env))  # BUY cap (configurable), bounded for safety
FOK_HARD_FLOOR = 0.01                                # Absolute min price for SELL
FOK_BUFFER     = 0.03                                # LEGACY fallback — use dynamic_fok_buffer()


def dynamic_fok_buffer(spread: float = 0.10, regime_label: str = "NORMAL") -> float:
    """
    Scale FOK buffer to match market conditions instead of wasting 3c on calm books.
    - Tight spread (1c) → buffer = 1c
    - Wide spread or HIGH_VOL → buffer up to 4c
    """
    base = max(0.01, min(spread, 0.02))          # never exceed spread, floor at 1c
    if regime_label in ("HIGH_VOL", "ADVERSARIAL", "VOL_EVENT"):
        base = min(0.04, base * 2.0)             # widen in volatile regimes
    return round(base, 2)


def _execute_fok_aggressive(
    token_id: str,
    order_side: str,
    target_size: float,
    fair_value: float = 0.50,
    max_slip: float = 0.06,
    signal_ask: float = 0.0,
    signal_bid: float = 0.0,
) -> Dict[str, Any]:
    """
    Aggressive FOK execution — one shot, fill-or-kill.

    Price = worst acceptable, NOT target:
      BUY:  min(ask + dyn_buf, fair_value + max_slip, FOK_HARD_CAP)
      SELL: max(bid - dyn_buf, fair_value - max_slip, FOK_HARD_FLOOR)
      where dyn_buf = dynamic_fok_buffer(spread, regime)

    Single atomic FOK — fills against resting liquidity at best
    available price or returns nothing. No maker ladder, no retries,
    no 2-level walk. Simple, reliable, no phantom orders.
    """
    global WINDOW_BANKROLL
    t0 = time.time()

    if client is None:
        return {"ok": False, "order_id": "", "exec_ms": 0,
                "used_limit": 0.0, "used_size": 0.0, "error": "No client",
                "state": "CONFIRMED_MISS"}

    order_side = order_side.upper()
    is_buy = (order_side == "BUY")

    # ── Lock check: prevent duplicate orders for same token+side ────
    if is_locked(token_id, order_side):
        logger.warning(f"FOK_SKIPPED({order_side}): side locked (pending order may be live)")
        return {"ok": False, "order_id": "", "exec_ms": 0,
                "used_limit": 0.0, "used_size": 0.0, "error": "SKIP_LOCKED",
                "state": "SKIP_LOCKED"}

    # ── Collateral preflight ────────────────────────────────────────
    cd_key = (token_id, order_side)

    # ── Fetch live book (with signal-time fallback for flicker) ─────
    snap = BOOK_CACHE.get(token_id)
    _used_signal_book = False

    if snap is None or (snap.best_bid == 0.0 and snap.best_ask == 1.0):
        # No live book at all — try signal-time fallback
        if signal_ask > 0.01 and signal_ask < 0.96:
            logger.info(
                f"FOK_FALLBACK({order_side}): no live book, using signal-time "
                f"bid={signal_bid:.2f} ask={signal_ask:.2f}"
            )
            bid, ask = float(signal_bid), float(signal_ask)
            _used_signal_book = True
        else:
            logger.warning(f"FOK_ABORT({order_side}): no valid book for {token_id[:12]}…")
            return {"ok": False, "order_id": "", "exec_ms": int((time.time()-t0)*1000),
                    "used_limit": 0.0, "used_size": 0.0, "error": "NO_BOOK_DATA",
                    "state": "CONFIRMED_MISS"}
    else:
        bid = float(snap.best_bid)
        ask = float(snap.best_ask)

    spread = max(0.0, ask - bid)

    # ── Book sanity: detect flicker (bid~0/ask~1) and fall back to signal-time ──
    _book_broken_buy = (is_buy and ask > 0.95 and spread > 0.90)
    _book_broken_sell = (not is_buy and bid < 0.05 and spread > 0.90)

    if (_book_broken_buy or _book_broken_sell) and not _used_signal_book:
        # Live book looks like a flicker tick — check if signal-time book was sane
        _sig_spread = signal_ask - signal_bid
        _sig_sane = (
            signal_ask > 0.01 and signal_ask < 0.96
            and signal_bid >= 0.0
            and _sig_spread < 0.90
        )
        if _sig_sane:
            logger.info(
                f"FOK_FLICKER_FALLBACK({order_side}): live book broken "
                f"(bid={bid:.2f} ask={ask:.2f}), using signal-time "
                f"bid={signal_bid:.2f} ask={signal_ask:.2f}"
            )
            bid, ask = float(signal_bid), float(signal_ask)
            spread = max(0.0, ask - bid)
            _used_signal_book = True
        else:
            # Both live and signal books are broken — abort
            logger.warning(
                f"FOK_ABORT({order_side}): broken book "
                f"(live bid={bid:.2f} ask={ask:.2f}, signal bid={signal_bid:.2f} ask={signal_ask:.2f})"
            )
            return {"ok": False, "order_id": "", "exec_ms": int((time.time()-t0)*1000),
                    "used_limit": 0.0, "used_size": 0.0,
                    "error": f"BOOK_SANITY ask={ask:.2f}",
                    "state": "CONFIRMED_MISS"}

    # ── Calculate worst-acceptable price ────────────────────────────
    size = float(max(0.0001, target_size))
    size = round(size, 6)

    _dyn_buf = dynamic_fok_buffer(spread, REGIME.label)

    if is_buy:
        # Worst price = ask + buffer, but never exceed FV cap or hard cap
        limit = ask + _dyn_buf
        fv_cap = fair_value + max_slip
        limit = min(limit, fv_cap, FOK_HARD_CAP)
        limit = round(min(0.99, max(0.01, limit)), 2)

        # Must still be marketable (>= ask)
        if limit < ask:
            logger.warning(
                f"FOK_ABORT({order_side}): limit={limit:.2f} < ask={ask:.2f} "
                f"(fv={fair_value:.2f} cap={FOK_HARD_CAP})"
            )
            return {"ok": False, "order_id": "", "exec_ms": int((time.time()-t0)*1000),
                    "used_limit": limit, "used_size": size,
                    "error": f"NON_MARKETABLE limit={limit:.2f}<ask={ask:.2f}",
                    "state": "CONFIRMED_MISS"}
    else:
        # Worst price = bid - buffer, but never go below FV floor or hard floor
        limit = bid - _dyn_buf
        fv_floor = fair_value - max_slip
        limit = max(limit, fv_floor, FOK_HARD_FLOOR)
        limit = round(max(0.01, min(0.99, limit)), 2)

        # Must still be marketable (<= bid)
        if limit > bid:
            logger.warning(
                f"FOK_ABORT({order_side}): limit={limit:.2f} > bid={bid:.2f} "
                f"(fv={fair_value:.2f} floor={FOK_HARD_FLOOR})"
            )
            return {"ok": False, "order_id": "", "exec_ms": int((time.time()-t0)*1000),
                    "used_limit": limit, "used_size": size,
                    "error": f"NON_MARKETABLE limit={limit:.2f}>bid={bid:.2f}",
                    "state": "CONFIRMED_MISS"}

    # ── Enforce CLOB $1 minimum notional ────────────────────────────
    if is_buy and (size * limit < 1.0):
        size = max(size, float(math.ceil(1.0 / max(0.01, limit))))

    # Preflight on actual submit limit, not fair value.
    try:
        _est_notional = float(size) * float(limit)
        _avail = max(0.0, float(WINDOW_BANKROLL))
        _buffer = 2.0
        if is_buy and _avail <= 0.0:
            logger.warning(f"FOK_PREFLIGHT({order_side}): window budget exhausted")
            ORDER_COOLDOWN[cd_key] = ms_now() + 3000
            return {"ok": False, "order_id": "", "exec_ms": 0,
                    "used_limit": limit, "used_size": float(size),
                    "error": "WINDOW_BUDGET_EXHAUSTED",
                    "state": "CONFIRMED_MISS"}
        if is_buy and _avail > 0 and _est_notional > (_avail - _buffer):
            logger.warning(
                f"FOK_PREFLIGHT({order_side}): notional=${_est_notional:.2f} > "
                f"avail=${_avail:.2f} — skipping"
            )
            ORDER_COOLDOWN[cd_key] = ms_now() + 3000
            return {"ok": False, "order_id": "", "exec_ms": 0,
                    "used_limit": limit, "used_size": float(size),
                    "error": "PREFLIGHT_BALANCE",
                    "state": "CONFIRMED_MISS"}
    except Exception:
        pass

    _prime_tick_size_cache(token_id)

    # ── Submit FAK (Fill-And-Kill) ─────────────────────────────────
    # FAK grabs whatever liquidity exists NOW and kills the rest.
    # Unlike FOK (all-or-nothing), FAK allows partial fills — dramatically
    # improving fill rate on thin Polymarket 5-min binary books.
    try:
        side_const = BUY if is_buy else SELL
        args = OrderArgs(price=limit, size=size, side=side_const, token_id=token_id)
        signed = client.create_order(
            args, options=PartialCreateOrderOptions(tick_size="0.01", neg_risk=False),
        )

        # Lock side BEFORE posting
        lock_side(token_id, order_side, ttl_ms=2000)

        resp = client.post_order(signed, OrderType.FAK)

        accepted = bool(resp) and resp.get("success", False)
        exec_ms = int((time.time() - t0) * 1000)
        oid = resp.get("orderID", "") if isinstance(resp, dict) else ""

        # FAK "success" means accepted, NOT filled — verify actual fill
        matched_size = 0.0
        fill_price = limit
        if accepted and oid:
            try:
                time.sleep(0.15)  # Brief wait for CLOB matching engine
                order_info = client.get_order(oid)
                if order_info:
                    matched_size = float(order_info.get("size_matched", 0) or 0)
                    # Use average fill price if available, else limit
                    _avg_price = order_info.get("associate_trades", [])
                    if _avg_price and isinstance(_avg_price, list) and len(_avg_price) > 0:
                        try:
                            _prices = [float(t.get("price", limit)) for t in _avg_price]
                            _sizes = [float(t.get("size", 1)) for t in _avg_price]
                            _total = sum(_sizes)
                            if _total > 0:
                                fill_price = sum(p * s for p, s in zip(_prices, _sizes)) / _total
                        except Exception:
                            pass
            except Exception as ve:
                logger.warning(f"FAK_VERIFY_ERR({order_side}): {ve}")
                # Conservative: if we can't verify, assume partial fill possible
                # Lock longer and report as pending
                lock_side(token_id, order_side, ttl_ms=3000)

        ok = matched_size > 0

        if ok:
            logger.info(
                f"FAK_HIT({order_side}): {matched_size}/{size}@{fill_price:.2f} "
                f"exec_ms={exec_ms} id={oid} ask={ask:.2f} bid={bid:.2f}"
                f"{' [PARTIAL]' if matched_size < size else ''}"
            )
            if is_buy:
                WINDOW_BANKROLL = max(0.0, WINDOW_BANKROLL - matched_size * fill_price)
            return {"ok": True, "order_id": oid,
                    "exec_ms": exec_ms, "used_limit": fill_price,
                    "used_size": matched_size, "error": "",
                    "state": "FILLED"}

        # No fill at all (no liquidity at our price)
        last_err = str(resp.get("errorMsg", resp) if isinstance(resp, dict) else resp)
        logger.info(
            f"FAK_MISS({order_side}): limit={limit:.2f} size={size} "
            f"ask={ask:.2f} bid={bid:.2f} accepted={accepted} err={last_err}"
        )
        return {"ok": False, "order_id": oid,
                "exec_ms": exec_ms, "used_limit": limit,
                "used_size": 0.0, "error": last_err or "NO_FILL",
                "state": "CONFIRMED_MISS"}

    except Exception as e:
        # ⚠️ NEVER RETRY — order may have been placed
        err_str = str(e).lower()
        exec_ms = int((time.time() - t0) * 1000)

        # Deterministic 4xx: order was NEVER placed
        if ("status_code=400" in err_str or "not enough balance" in err_str
                or "allowance" in err_str or "status_code=401" in err_str
                or "status_code=403" in err_str or "status_code=422" in err_str):
            logger.warning(
                f"FAK_REJECT_4xx({order_side}): {limit:.2f}x{size} err={e}"
            )
            lock_side(token_id, order_side, ttl_ms=800)
            if "not enough balance" in err_str or "allowance" in err_str:
                try:
                    # Re-sync allowance with CLOB backend (self-heal)
                    client.update_balance_allowance(
                        params=BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
                    )
                    logger.info("ALLOWANCE_RESYNC: refreshed USDC allowance after rejection")
                    _live_bal = float(get_available_usdc_balance())
                    WINDOW_BANKROLL = max(
                        0.0,
                        min(float(WINDOW_BANKROLL), _live_bal, WINDOW_SPEND_CAP_USD),
                    )
                except Exception:
                    WINDOW_BANKROLL = max(0.0, WINDOW_BANKROLL * 0.5)
            return {"ok": False, "order_id": "",
                    "exec_ms": exec_ms, "used_limit": limit,
                    "used_size": 0.0, "error": f"EXCEPTION: {e}",
                    "state": "CONFIRMED_MISS"}

        # Network/timeout/5xx: order MIGHT have been placed — conservative miss
        logger.error(
            f"FAK_EXCEPTION({order_side}): {limit:.2f}x{size} "
            f"err={e} — marking PENDING_ACK"
        )
        lock_side(token_id, order_side, ttl_ms=4000)
        return {"ok": False, "order_id": "",
                "exec_ms": exec_ms, "used_limit": limit,
                "used_size": 0.0, "error": f"EXCEPTION: {e}",
                "state": "PENDING_ACK"}


def execute_order_sync(payload: Dict[str, Any]) -> Dict[str, Any]:
    TICK_BUFFER   = 0.01
    t0            = time.time()
    token_id      = payload["token_id"]
    size          = float(payload["size"])
    original_edge = float(payload.get("edge", 0.0))
    retry_idx     = payload.get("retry_idx", 0)
    order_side_str = payload.get("order_side", "BUY")
    is_sell        = (order_side_str == "SELL")

    try:
        if client is None:
            return {"ok": False, "resp": {"error": "No client"},
                    "exec_ms": 0, "retry_idx": retry_idx}

        live_snap = BOOK_CACHE.get(token_id)

        if is_sell:
            signal_bid = round(payload["price"] + TICK_BUFFER, 2)
            if live_snap is not None:
                live_bid = live_snap.best_bid
                bid_slip = signal_bid - live_bid
                if bid_slip > 0.10:
                    return {"ok": False, "exec_ms": int((time.time()-t0)*1000),
                            "resp": {"error": f"EXIT_SLIPPAGE: {bid_slip:.2f}"},
                            "retry_idx": retry_idx}
                limit_price = round(live_bid - TICK_BUFFER, 2)
            else:
                limit_price = round(payload["price"], 2)
            limit_price = max(0.01, limit_price)
        else:
            signal_ask = round(payload["price"] - TICK_BUFFER, 2)
            if live_snap is not None:
                live_ask = live_snap.best_ask
                slippage = live_ask - signal_ask
                max_slip = max(0.03, original_edge)
                if slippage > max_slip:
                    return {"ok": False, "exec_ms": int((time.time()-t0)*1000),
                            "resp": {"error": f"EDGE_ERODED: slip={slippage:.2f}"},
                            "retry_idx": retry_idx}
                limit_price = round(live_ask + TICK_BUFFER, 2)
            else:
                limit_price = round(payload["price"], 2)
            limit_price = min(0.99, limit_price)

        _prime_tick_size_cache(token_id)

        side_const = SELL if is_sell else BUY
        args   = OrderArgs(price=limit_price, size=size, side=side_const, token_id=token_id)
        signed = client.create_order(
            args, options=PartialCreateOrderOptions(tick_size="0.01", neg_risk=False),
        )
        resp = client.post_order(signed, OrderType.FAK)

        exec_ms = int((time.time() - t0) * 1000)
        accepted = bool(resp) and resp.get("success", False)
        oid = resp.get("orderID", "") if isinstance(resp, dict) else ""

        # FAK verify: check actual fill
        matched_size = 0.0
        if accepted and oid:
            try:
                time.sleep(0.15)
                order_info = client.get_order(oid)
                if order_info:
                    matched_size = float(order_info.get("size_matched", 0) or 0)
            except Exception as ve:
                logger.warning(f"LEGACY_FAK_VERIFY_ERR: {ve}")

        ok = matched_size > 0
        if ok:
            d = "SELL" if is_sell else "BUY"
            logger.info(
                f"ORDER_{d}: {matched_size}/{size}@{limit_price:.4f} exec_ms={exec_ms}"
                f"{' [PARTIAL]' if matched_size < size else ''}"
            )
        return {"ok": ok, "resp": resp, "exec_ms": exec_ms,
                "retry_idx": retry_idx, "exec_price": limit_price,
                "matched_size": matched_size}

    except Exception as e:
        return {"ok": False, "resp": {"error": str(e)},
                "exec_ms": int((time.time()-t0)*1000), "retry_idx": retry_idx}


async def execution_loop() -> None:
    global TRADED_UP_THIS_WINDOW, TRADED_DN_THIS_WINDOW, TRADES_THIS_WINDOW, WINDOW_BANKROLL
    global ENTRY_P_CONE, ENTRY_P_MARKET, ENTRY_T_SEC
    ensure_logs()

    while True:
        payload = await execution_queue.get()
        action  = payload.get("action", "ORDER")

        exec_lock()

        # Determine oracle source for logging
        oracle_src = "rtds" if RTDS.is_fresh(RTDS_FRESH_MS) else "coinbase"
        oracle_px  = RTDS.price if RTDS.is_fresh(RTDS_FRESH_MS) else STATE.btc_price

        # ── Helper: map token_id → Portfolio Position ──────────────────────
        def pos_for_token(tid: str):
            if tid == UP_TOKEN_ID:
                return POS_UP
            if tid == DOWN_TOKEN_ID:
                return POS_DOWN
            return None

        # ── 3-Layer Execution Router ──────────────────────────────────────
        if action in ("ORDER", "PAIR_BUY", "PAIR_REBALANCE"):
            if client is None:
                logger.warning("No CLOB client — skipping order")
                exec_unlock()
                execution_queue.task_done()
                continue

            side = payload.get("side", "UP")
            token_id = payload["token_id"]
            order_side = payload.get("order_side", "BUY").upper()
            mode = str(payload.get("mode", "") or "").lower()
            limit_price = round(float(payload["price"]), 2)
            size = float(payload["size"])
            now_ms_exec = ms_now()
            cd_key = (token_id, order_side)
            if now_ms_exec < ORDER_COOLDOWN.get(cd_key, 0):
                logger.info(f"COOLDOWN_ACTIVE {cd_key}")
                exec_unlock()
                execution_queue.task_done()
                continue
            if size <= 0:
                logger.info(f"ORDER_SKIP: non-positive size={size} token={token_id[:12]}...")
                exec_unlock()
                execution_queue.task_done()
                continue

            # Hard per-order BUY cap ($-notional)
            if order_side == "BUY":
                _max_buy_qty = math.floor(
                    MAX_SPEND_PER_ORDER_USD / max(0.01, limit_price)
                )
                if _max_buy_qty < 1:
                    logger.warning(
                        f"ORDER_BLOCK: BUY cap too small for price={limit_price:.2f} "
                        f"(cap=${MAX_SPEND_PER_ORDER_USD:.2f})"
                    )
                    exec_unlock()
                    execution_queue.task_done()
                    continue
                if size > _max_buy_qty:
                    logger.info(
                        f"ORDER_CAP: BUY size {size:.0f}->{_max_buy_qty:.0f} "
                        f"at ${limit_price:.2f} (cap=${MAX_SPEND_PER_ORDER_USD:.2f})"
                    )
                    size = float(_max_buy_qty)
            else:
                # Never send a SELL larger than known local inventory.
                _pos_local = pos_for_token(token_id)
                _inv_local = float(_pos_local.inventory) if _pos_local is not None else 0.0
                if _inv_local <= 0.0:
                    logger.info(f"SELL_SKIP_NO_INV: token={token_id[:12]}...")
                    exec_unlock()
                    execution_queue.task_done()
                    continue
                if size > _inv_local:
                    logger.info(f"SELL_CAP_INV: {size:.6f}->{_inv_local:.6f}")
                    size = _inv_local

            loop = asyncio.get_running_loop()

            # Decide fair prob for the token being traded.
            p_cone = float(payload.get("p_cone", 0.5) or 0.5)
            z_val = float(payload.get("z", 0.0) or 0.0)

            # ── Compute fresh fair value from live oracle ─────────────
            edge_val = float(payload.get("edge", 0.0) or 0.0)
            if edge_val >= 0.10:
                max_slip = min(0.25, max(0.06, edge_val))
            else:
                max_slip = max(0.03, min(0.10, edge_val))

            _fresh_oracle = RTDS.price if RTDS.is_fresh(RTDS_FRESH_MS) else STATE.btc_price
            if (np.isfinite(_fresh_oracle) and _fresh_oracle > 0
                    and np.isfinite(STATE.open_price) and STATE.open_price > 0
                    and STATE.sec_remaining > 0 and STATE.sigma_1m > 0):
                from oracle_engine import cone_p_and_z
                p_cone, _ = cone_p_and_z(
                    _fresh_oracle, STATE.open_price, STATE.sec_remaining, STATE.sigma_1m
                )
            else:
                p_cone = float(payload.get("p_cone", 0.5) or 0.5)
            if token_id == UP_TOKEN_ID:
                token_fair_value = p_cone
                token_prob = p_cone
            elif token_id == DOWN_TOKEN_ID:
                token_fair_value = 1.0 - p_cone
                token_prob = 1.0 - p_cone
            else:
                token_fair_value = p_cone if side == "UP" else (1.0 - p_cone)
                token_prob = token_fair_value

            # Disable implied SELL for entry flow (entry should be BUY-only).
            _is_exit_mode = mode in ("exit", "trim", "hedge", "reduce", "close")
            if action == "ORDER" and order_side == "SELL" and not _is_exit_mode:
                logger.info("ENTRY_SELL_BLOCKED: implied SELL disabled for entry")
                ORDER_COOLDOWN[cd_key] = ms_now() + 2000
                with open("logs/trades.csv", "a", newline="") as f:
                    csv.writer(f).writerow([
                        ms_now(), side, token_id, limit_price, size,
                        "ENTRY_SELL_BLOCKED", "", 0, 0,
                        payload.get("edge", ""), payload.get("p_cone", ""),
                        payload.get("z", ""), payload.get("sigma_1m", ""),
                        oracle_src,
                        f"{oracle_px:.2f}" if np.isfinite(oracle_px) else "",
                        "FOK_AGGRESSIVE",
                        "implied_sell_entry_disabled",
                    ])
                exec_unlock()
                execution_queue.task_done()
                continue

            # ── Submission-time marketability pre-check ─────────────────
            # NOTE: The FOK function recalculates price with dynamic buffer,
            # so the signal's limit_price (= ask at signal time) will be stale.
            # Only block if the book has moved so far that even the FOK buffer
            # can't bridge the gap.  Otherwise let the FOK function handle it.
            snap = BOOK_CACHE.get(token_id)
            if snap:
                best_bid = float(snap.best_bid)
                best_ask = float(snap.best_ask)
                _book_flickering = (best_bid <= 0.02 and best_ask >= 0.98)
                _precheck_spread = max(0.0, best_ask - best_bid)
                _fok_buf = dynamic_fok_buffer(_precheck_spread, REGIME.label)
                if order_side == "BUY" and (limit_price + _fok_buf) < best_ask and not _book_flickering:
                    logger.warning(
                        f"NON_MARKETABLE BUY: limit={limit_price:.2f}+buf={_fok_buf} < ask={best_ask:.2f}"
                    )
                    ORDER_COOLDOWN[cd_key] = ms_now() + 2000
                    with open("logs/trades.csv", "a", newline="") as f:
                        csv.writer(f).writerow([
                            ms_now(), side, token_id, limit_price, size,
                            "NON_MARKETABLE", "", 0, 0,
                            payload.get("edge", ""), payload.get("p_cone", ""),
                            payload.get("z", ""), payload.get("sigma_1m", ""),
                            oracle_src,
                            f"{oracle_px:.2f}" if np.isfinite(oracle_px) else "",
                            "FOK_AGGRESSIVE",
                            f"limit={limit_price:.2f}+buf<ask={best_ask:.2f}",
                        ])
                    exec_unlock()
                    reset_sniper_lock()  # allow retry on same side
                    execution_queue.task_done()
                    continue
                if order_side == "SELL" and (limit_price - _fok_buf) > best_bid and not _book_flickering:
                    logger.warning(
                        f"NON_MARKETABLE SELL: limit={limit_price:.2f}-buf={_fok_buf} > bid={best_bid:.2f}"
                    )
                    ORDER_COOLDOWN[cd_key] = ms_now() + 2000
                    with open("logs/trades.csv", "a", newline="") as f:
                        csv.writer(f).writerow([
                            ms_now(), side, token_id, limit_price, size,
                            "NON_MARKETABLE", "", 0, 0,
                            payload.get("edge", ""), payload.get("p_cone", ""),
                            payload.get("z", ""), payload.get("sigma_1m", ""),
                            oracle_src,
                            f"{oracle_px:.2f}" if np.isfinite(oracle_px) else "",
                            "FOK_AGGRESSIVE",
                            f"limit={limit_price:.2f}-buf>bid={best_bid:.2f}",
                        ])
                    exec_unlock()
                    reset_sniper_lock()  # allow retry on same side
                    execution_queue.task_done()
                    continue

            # Use actual edge_target from signal, not zero — prevents executing
            # trades where edge has collapsed between signal and submission
            _min_edge_for_check = float(payload.get("edge_target", 0.0) or 0.0)
            # For exit/trim modes, edge_target is irrelevant — allow any edge
            if payload.get("mode") in ("exit", "trim", "hedge"):
                _min_edge_for_check = 0.0
            _edge_ok, _edge_now = validate_execution_edge(
                side=order_side,
                exec_price=limit_price,
                model_prob=token_prob,
                fee_fn=fee_per_share,
                min_edge=_min_edge_for_check,
            )
            if not _edge_ok:
                logger.warning(
                    f"EXEC_EDGE_BLOCK: side={order_side} "
                    f"limit={limit_price:.2f} token_prob={token_prob:.4f} edge_now={_edge_now:.4f}"
                )
                ORDER_COOLDOWN[cd_key] = ms_now() + 2000
                with open("logs/trades.csv", "a", newline="") as f:
                    csv.writer(f).writerow([
                        ms_now(), side, token_id, limit_price, size,
                        "EXEC_EDGE_BLOCK", "", 0, 0,
                        payload.get("edge", ""), payload.get("p_cone", ""),
                        payload.get("z", ""), payload.get("sigma_1m", ""),
                        oracle_src,
                        f"{oracle_px:.2f}" if np.isfinite(oracle_px) else "",
                        "FOK_AGGRESSIVE",
                        f"edge_now={_edge_now:.4f}",
                    ])
                exec_unlock()
                execution_queue.task_done()
                continue

            # ── Pre-order balance guard ──────────────────────────────
            order_cost = size * limit_price
            avail_bal = WINDOW_BANKROLL if WINDOW_BANKROLL > 0 else 0.0
            if order_side == "BUY" and avail_bal <= 0:
                logger.warning("WINDOW_BUDGET_EXHAUSTED: buy blocked for this 5m window")
                with open("logs/trades.csv", "a", newline="") as f:
                    csv.writer(f).writerow([
                        ms_now(), side, token_id, limit_price, size,
                        "WINDOW_BUDGET_EXHAUSTED", "", 0, 0,
                        payload.get("edge", ""), payload.get("p_cone", ""),
                        payload.get("z", ""), payload.get("sigma_1m", ""),
                        oracle_src,
                        f"{oracle_px:.2f}" if np.isfinite(oracle_px) else "",
                        "FOK_AGGRESSIVE",
                        "window_budget=$0.00",
                    ])
                exec_unlock()
                execution_queue.task_done()
                continue
            if avail_bal > 0 and order_cost > avail_bal:
                old_size = size
                size = max(1.0, float(math.floor(avail_bal / max(0.01, limit_price))))
                if size * limit_price > avail_bal or size < 1:
                    logger.warning(
                        f"BALANCE_BLOCK: need ${order_cost:.2f} but have ${avail_bal:.2f} — skipping"
                    )
                    ORDER_COOLDOWN[cd_key] = ms_now() + 2000
                    with open("logs/trades.csv", "a", newline="") as f:
                        csv.writer(f).writerow([
                            ms_now(), side, token_id, limit_price, old_size,
                            "INSUFFICIENT_BALANCE", "", 0, 0,
                            payload.get("edge", ""), payload.get("p_cone", ""),
                            payload.get("z", ""), payload.get("sigma_1m", ""),
                            oracle_src,
                            f"{oracle_px:.2f}" if np.isfinite(oracle_px) else "",
                            "FOK_AGGRESSIVE",
                            f"need=${order_cost:.2f} have=${avail_bal:.2f}",
                        ])
                    exec_unlock()
                    execution_queue.task_done()
                    continue
                logger.info(f"BALANCE_CAP: size {old_size}→{size} (${avail_bal:.2f} available)")

            # ── Aggressive FOK path (single shot, fill-or-kill) ───────
            # For sniper (mispricing) signals, enforce oracle_engine's price cap
            _mode_exec = str(payload.get("mode", "") or "").lower()
            _fok_fv = token_fair_value
            _fok_slip = max_slip
            if _mode_exec == "mispricing":
                # Sniper limit_price IS the price cap — clamp FOK to never exceed it
                sniper_cap = round(float(payload.get("price", 1.0)), 2)
                _fok_slip = max(0.01, sniper_cap - _fok_fv)
                logger.info(
                    f"SNIPER_FOK_CAP: fv={_fok_fv:.4f} slip={_fok_slip:.4f} "
                    f"sniper_limit={sniper_cap:.2f} (original_slip={max_slip:.2f})"
                )

            # ── Execution routing: AdaptiveExecutor (maker) vs FOK (taker) ──
            _exec_regime = str(payload.get("regime", "NORMAL"))
            _exec_snap = BOOK_CACHE.get(token_id)
            _exec_spread = float(_exec_snap.best_ask - _exec_snap.best_bid) if _exec_snap else 0.10
            _exec_T = float(STATE.sec_remaining)
            _use_adaptive = (
                ADAPTIVE_EXEC is not None
                and _exec_regime in ("CALM", "NORMAL")
                and order_side == "BUY"
                and _exec_T > 60.0
                and _exec_spread <= 0.06
                and _mode_exec not in ("exit", "trim", "hedge")
            )

            if _use_adaptive:
                # Build MicroSnapshot from live data
                _ms = MicroSnapshot()
                if _exec_snap:
                    _ms.best_bid = float(_exec_snap.best_bid)
                    _ms.best_ask = float(_exec_snap.best_ask)
                    _ms.best_bid_sz = float(getattr(_exec_snap, 'best_bid_size', 50.0))
                    _ms.best_ask_sz = float(getattr(_exec_snap, 'best_ask_size', 50.0))
                _ms.sigma = float(STATE.sigma_1m)
                _ms.trade_vel = float(getattr(FLOW, '_fills', deque()) and len(FLOW._fills) / 30.0 or 1.0)
                _ms.flow_bias = float(FLOW.imbalance(
                    up_price=_ms.best_bid if side == "UP" else 0.5,
                    down_price=_ms.best_bid if side == "DOWN" else 0.5,
                ) if FLOW else 0.0)
                _ms.strike_price = float(STATE.open_price) if np.isfinite(STATE.open_price) else 0.5
                _ms.current_btc_price = float(STATE.btc_price) if np.isfinite(STATE.btc_price) else 0.0
                _ms.net_position = float(POS_UP.inventory if side == "UP" else POS_DOWN.inventory)
                _ms.max_inventory = 200.0

                _p_cone = float(payload.get("p_cone", 0.5) or 0.5)
                _z_score = float(payload.get("z", 0.0) or 0.0)
                _deficit = 1.0  # full coverage desired

                logger.info(
                    f"EXEC_ROUTE: ADAPTIVE {side} {order_side} {size} "
                    f"regime={_exec_regime} spread={_exec_spread:.3f} T={_exec_T:.0f}s"
                )

                try:
                    _ae_result = await loop.run_in_executor(
                        thread_pool, ADAPTIVE_EXEC.execute,
                        side, token_id, _p_cone, _z_score,
                        size, _ms, _deficit, order_side,
                    )
                    # Convert ExecutionResult → fok_result dict
                    fok_result = {
                        "ok": _ae_result.filled,
                        "order_id": _ae_result.order_id,
                        "exec_ms": _ae_result.exec_ms,
                        "used_limit": _ae_result.fill_price,
                        "used_size": _ae_result.fill_size,
                        "error": _ae_result.error,
                        "state": "FILLED" if _ae_result.filled else "CONFIRMED_MISS",
                    }
                    _exec_method = f"ADAPTIVE_{_ae_result.method}"
                except Exception as e:
                    logger.warning(f"ADAPTIVE_EXEC failed: {e} — falling back to FOK")
                    _use_adaptive = False  # fallback below

            if not _use_adaptive:
                _exec_method = "FAK_AGGRESSIVE"
                logger.info(
                    f"EXEC_ROUTE: FAK {side} {order_side} {size}@worst "
                    f"fv={_fok_fv:.4f} slip={_fok_slip:.2f} cap={FOK_HARD_CAP}"
                )

                _sig_ask = float(payload.get("signal_ask", 0.0) or 0.0)
                _sig_bid = float(payload.get("signal_bid", 0.0) or 0.0)
                fok_result = await loop.run_in_executor(
                    thread_pool, _execute_fok_aggressive,
                    token_id, order_side, size, _fok_fv, _fok_slip,
                    _sig_ask, _sig_bid,
                )

            ok = fok_result["ok"]
            trade_state = fok_result.get("state", "FILLED" if ok else "CONFIRMED_MISS")
            if not ok:
                err_text = str(fok_result.get("error", ""))
                if (
                    trade_state in ("EXEC_EDGE_BLOCK", "INSUFFICIENT_BALANCE", "NON_MARKETABLE")
                    or "NON_MARKETABLE" in err_text
                ):
                    ORDER_COOLDOWN[cd_key] = ms_now() + 2000

            with open("logs/trades.csv", "a", newline="") as f:
                csv.writer(f).writerow([
                    ms_now(), side, token_id,
                    fok_result.get("used_limit", limit_price),
                    fok_result.get("used_size", size),
                    trade_state,
                    fok_result.get("order_id", ""),
                    fok_result["exec_ms"], 0,
                    payload.get("edge", ""), payload.get("p_cone", ""),
                    payload.get("z", ""), payload.get("sigma_1m", ""),
                    oracle_src,
                    f"{oracle_px:.2f}" if np.isfinite(oracle_px) else "",
                    # ── Attribution ──
                    _exec_method,
                    fok_result.get("error", ""),
                    payload.get("regime", ""),
                    payload.get("edge_target", ""),
                    payload.get("mode", "entry"),
                    payload.get("flow", ""),
                ])

            if ok:
                _fill_side = payload.get("side", "")
                if _fill_side == "UP":
                    TRADED_UP_THIS_WINDOW = True
                elif _fill_side == "DOWN":
                    TRADED_DN_THIS_WINDOW = True
                TRADES_THIS_WINDOW += 1

                # SPRT Brier: store entry-time probabilities (first fill only)
                _fill_mode = payload.get("mode", "entry")
                if _fill_mode not in ("exit", "trim", "hedge") and _fill_side in ("UP", "DOWN"):
                    if ENTRY_P_CONE.get(_fill_side) is None:
                        _p_c = float(payload.get("p_cone", 0.5) or 0.5)
                        ENTRY_P_CONE[_fill_side] = _p_c
                        ENTRY_T_SEC[_fill_side] = float(STATE.sec_remaining)
                        # Market midpoint = average of bid/ask for this token
                        _fill_snap = BOOK_CACHE.get(token_id)
                        if _fill_snap:
                            _mkt_mid = (float(_fill_snap.best_bid) + float(_fill_snap.best_ask)) / 2
                        else:
                            _mkt_mid = float(fok_result.get("used_limit", 0.5))
                        ENTRY_P_MARKET[_fill_side] = _mkt_mid

                _fill_price = float(fok_result.get("used_limit", limit_price))
                _fill_size  = float(fok_result.get("used_size", size))
                logger.info(
                    f"ORDER_FILLED({_exec_method}): {side} {order_side} "
                    f"{_fill_size}@{_fill_price:.2f} exec_ms={fok_result['exec_ms']}"
                )

                # FIFO portfolio accounting (truth)
                _fill_pos = pos_for_token(token_id)
                if _fill_pos is not None:
                    _actual_qty = _fill_size
                    if order_side == "SELL":
                        _actual_qty = min(_actual_qty, float(_fill_pos.inventory))
                        if _actual_qty <= 0:
                            logger.warning(f"SELL skipped: no FIFO inventory for {token_id}")
                        else:
                            _fill_pos.apply_fill(
                                side=Side.SELL,
                                qty=_actual_qty,
                                price=_fill_price,
                                ts_ms=ms_now(),
                                fee_per_share=fee_per_share(_fill_price),
                                meta={"path": _exec_method, "token_id": token_id, "order_side": order_side},
                            )
                    else:
                        _fill_pos.apply_fill(
                            side=Side.BUY,
                            qty=_actual_qty,
                            price=_fill_price,
                            ts_ms=ms_now(),
                            fee_per_share=fee_per_share(_fill_price),
                            meta={"path": _exec_method, "token_id": token_id, "order_side": order_side},
                        )

            exec_unlock()
            execution_queue.task_done()
            continue

        # ── Legacy FOK path for PAIR_BUY / PAIR_REBALANCE ────────────
        loop = asyncio.get_running_loop()
        fok_result = await loop.run_in_executor(thread_pool, execute_order_sync, payload)
        ok   = fok_result["ok"]
        resp = fok_result["resp"]

        with open("logs/trades.csv", "a", newline="") as f:
            csv.writer(f).writerow([
                ms_now(), payload.get("side"), payload["token_id"],
                payload["price"], payload["size"],
                "FILLED" if ok else "MISS",
                resp.get("orderID") if isinstance(resp, dict) else "",
                fok_result["exec_ms"], fok_result.get("retry_idx", 0),
                payload.get("edge", ""), payload.get("p_cone", ""),
                payload.get("z", ""), payload.get("sigma_1m", ""),
                oracle_src, f"{oracle_px:.2f}" if np.isfinite(oracle_px) else "",
            ])

        if action == "PAIR_BUY" and ok:
            filled = float(payload["size"])
            # FIFO accounting handled by pos_for_token in main ORDER path;
            # for legacy FOK, apply directly
            _pair_pos = pos_for_token(payload["token_id"])
            if _pair_pos is not None:
                _pair_pos.apply_fill(
                    side=Side.BUY, qty=filled,
                    price=float(payload["price"]),
                    ts_ms=ms_now(),
                    fee_per_share=fee_per_share(float(payload["price"])),
                    meta={"path": "PAIR_BUY", "token_id": payload["token_id"]},
                )
            logger.info(f"PAIR BUY → UP:{POS_UP.inventory:.0f} DN:{POS_DOWN.inventory:.0f}")

        elif action == "PAIR_REBALANCE" and ok:
            sold = float(payload["size"])
            _pair_pos = pos_for_token(payload["token_id"])
            _exec_price = float(fok_result.get("exec_price", payload["price"]))
            _fee = fee_per_share(_exec_price)
            if _pair_pos is not None:
                _pair_pos.apply_fill(
                    side=Side.SELL, qty=sold,
                    price=_exec_price,
                    ts_ms=ms_now(),
                    fee_per_share=_fee,
                    meta={"path": "PAIR_REBALANCE", "token_id": payload["token_id"]},
                )
            net_proceeds = sold * (_exec_price - _fee)
            _pp = get_portfolio_pos()
            ep = _pp.avg_cost_up if payload["side"] == "UP" else _pp.avg_cost_dn
            record_pnl(net_proceeds, sold * ep)
            logger.info(f"PAIR REBALANCE → UP:{POS_UP.inventory:.0f} DN:{POS_DOWN.inventory:.0f}")

        if not ok:
            err = str(resp.get("error") or resp) if isinstance(resp, dict) else str(resp)
            logger.warning(f"ORDER MISS: {action} {payload.get('side')} err={err}")

        exec_unlock()
        execution_queue.task_done()


# ════════════════════════════════════════════════════════════════════════════
# WEBSOCKET TASKS
# ════════════════════════════════════════════════════════════════════════════

def _process_book_update(msg: dict) -> bool:
    """
    Robust WebSocket book handler.
    - Preserves last known sizes
    - Updates sizes when provided
    - Removes fake FLOW.record_fill injections
    - Updates L2 tracker on real depth changes only
    Returns True if book was updated (for lag tracking).
    """
    updated = False

    try:
        tid = msg.get("asset_id")
        if not tid:
            # Some messages nest inside "price_changes"
            changes = msg.get("price_changes", [])
            if changes:
                for change in changes:
                    if _process_book_update(change):
                        updated = True
            return updated

        et = msg.get("event_type")

        # Ensure POLY_STATE entry exists (keyed by token_id)
        if tid not in POLY_STATE:
            POLY_STATE[tid] = {
                "bid": 0.0,
                "ask": 1.0,
                "bid_size": 0.0,
                "ask_size": 0.0,
                "last_update": 0,
                "source": "ws"
            }

        state = POLY_STATE[tid]
        now = ms_now()

        # ─────────────────────────────────────────────
        # BEST BID/ASK UPDATE
        # ─────────────────────────────────────────────
        if et == "best_bid_ask":

            bid = msg.get("bid_price") or msg.get("best_bid")
            ask = msg.get("ask_price") or msg.get("best_ask")

            if bid is not None:
                state["bid"] = float(bid)
            if ask is not None:
                state["ask"] = float(ask)

            # Size fields sometimes present
            bid_sz = msg.get("bid_size")
            ask_sz = msg.get("ask_size")

            if bid_sz is not None:
                state["bid_size"] = float(bid_sz)
            if ask_sz is not None:
                state["ask_size"] = float(ask_sz)

            state["last_update"] = now
            state["source"] = "ws"
            updated = True

        # ─────────────────────────────────────────────
        # FULL DEPTH SNAPSHOT
        # ─────────────────────────────────────────────
        elif et == "orderbook_snapshot" or (msg.get("bids") is not None or msg.get("asks") is not None):

            bids = msg.get("bids", [])
            asks = msg.get("asks", [])

            try:
                if bids:
                    state["bid"] = float(bids[0]["price"])
                    state["bid_size"] = float(bids[0].get("size", 0))
                if asks:
                    state["ask"] = float(asks[0]["price"])
                    state["ask_size"] = float(asks[0].get("size", 0))
            except (IndexError, KeyError, ValueError):
                pass

            state["last_update"] = now
            state["source"] = "ws"
            updated = True

        # ─────────────────────────────────────────────
        # DEPTH DELTA UPDATE (price_changes with side/price/size)
        # ─────────────────────────────────────────────
        elif et == "price_changes":

            for change in msg.get("price_changes", []):
                side = change.get("side")
                price = float(change.get("price", 0))
                size = float(change.get("size", 0))

                # Update top of book if level matches
                if side == "BUY" and price == state["bid"]:
                    if size == 0:
                        state["bid"] = 0.0
                        state["bid_size"] = 0.0
                    else:
                        state["bid_size"] = size
                elif side == "SELL" and price == state["ask"]:
                    if size == 0:
                        state["ask"] = 1.0
                        state["ask_size"] = 0.0
                    else:
                        state["ask_size"] = size

            # Also check for best_bid/best_ask in the message itself
            bb = msg.get("best_bid")
            ba = msg.get("best_ask")
            if bb is not None:
                state["bid"] = float(bb)
            if ba is not None:
                state["ask"] = float(ba)

            bs = msg.get("bid_size")
            as_ = msg.get("ask_size")
            if bs is not None:
                state["bid_size"] = float(bs)
            if as_ is not None:
                state["ask_size"] = float(as_)

            state["last_update"] = now
            state["source"] = "ws"
            updated = True

        # ─────────────────────────────────────────────
        # REAL TRADE EVENT (ONLY place FLOW update here)
        # ─────────────────────────────────────────────
        elif et == "last_trade_price":

            price = msg.get("price")
            if price is not None:
                # Determine aggressor from message if available
                side = msg.get("side")
                is_buy = True if side == "BUY" else False
                fill_price = float(price)
                fill_size = float(msg.get("size", 1.0))
                FLOW.record_fill(
                    token_id=tid,
                    price=fill_price,
                    size=fill_size,
                    ts_ms=now,
                    is_buy=is_buy,
                )
                # Feed confirmation buffer (truth source for reconciliation)
                RECENT_FILLS.append({
                    "ts_ms": now,
                    "token_id": tid,
                    "side": side or ("BUY" if is_buy else "SELL"),
                    "price": fill_price,
                    "size": fill_size,
                })

        # ─────────────────────────────────────────────
        # WRITE INTO BOOK_CACHE SNAPSHOT
        # ─────────────────────────────────────────────
        if updated and tid in (UP_TOKEN_ID, DOWN_TOKEN_ID):
            existing = BOOK_CACHE.get(tid)
            if existing:
                # Atomic update — avoid torn reads from unlocked field mutations
                BOOK_CACHE.update(tid, BookSnapshot(
                    token_id=tid,
                    best_bid=state["bid"],
                    best_ask=state["ask"],
                    spread=max(0.0, state["ask"] - state["bid"]),
                    bid_size=state["bid_size"],
                    ask_size=state["ask_size"],
                    source=state["source"],
                    last_update=state["last_update"],
                ))
            else:
                BOOK_CACHE.update(tid, BookSnapshot(
                    token_id=tid,
                    best_bid=state["bid"],
                    best_ask=state["ask"],
                    spread=max(0.0, state["ask"] - state["bid"]),
                    bid_size=state["bid_size"],
                    ask_size=state["ask_size"],
                    source=state["source"],
                    last_update=state["last_update"],
                ))

            # ─────────────────────────────────────────
            # UPDATE L2 TRACKER WITH REAL DEPTH CHANGES
            # ─────────────────────────────────────────
            L2_TRACKER.update(
                state["bid_size"],
                state["ask_size"],
                0.0,  # no fake trade vol injection
                0.0,
            )

    except Exception as e:
        logger.warning(f"WS_BOOK_ERROR: {e}")

    return updated


async def polymarket_book_task(poly_ws_url: str) -> None:
    global BOOK_RECONNECT_FLAG
    retry_delay = 1
    last_tokens = (None, None)

    while True:
        try:
            async with websockets.connect(
                poly_ws_url,
                ping_interval=None,    # disable client pings — Poly server sends its own
                ping_timeout=None,
                close_timeout=10,
                open_timeout=15,
            ) as ws:
                logger.info("Poly CLOB WS connected.")
                retry_delay = 1

                while True:
                    if BOOK_RECONNECT_FLAG:
                        us, ds = BOOK_CACHE.get(UP_TOKEN_ID), BOOK_CACHE.get(DOWN_TOKEN_ID)
                        if (us and ds and us.source == "ws" and ds.source == "ws" and
                                (ms_now()-us.last_update) < POLY_BOOK_MAX_AGE_MS and
                                (ms_now()-ds.last_update) < POLY_BOOK_MAX_AGE_MS):
                            BOOK_RECONNECT_FLAG = False
                        else:
                            BOOK_RECONNECT_FLAG = False
                            last_tokens = (None, None)
                            break

                    cur = (UP_TOKEN_ID, DOWN_TOKEN_ID)
                    if cur != last_tokens and cur[0] and cur[1]:
                        sub = {"type": "market", "assets_ids": list(cur),
                               "custom_feature_enabled": True}
                        await ws.send(json.dumps(sub))
                        logger.info(f"Subscribed: UP={cur[0][:12]}… DOWN={cur[1][:12]}…")
                        last_tokens = cur

                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                    except asyncio.TimeoutError:
                        # Check for total staleness — if no data for 60s, reconnect
                        up_s = BOOK_CACHE.get(UP_TOKEN_ID)
                        dn_s = BOOK_CACHE.get(DOWN_TOKEN_ID)
                        oldest_ws_age = max(
                            (ms_now() - up_s.last_update) if up_s and up_s.source == "ws" else 999999,
                            (ms_now() - dn_s.last_update) if dn_s and dn_s.source == "ws" else 999999,
                        ) if (up_s or dn_s) else 999999
                        if oldest_ws_age > 60_000:
                            logger.warning("Poly WS: no data for 60s, forcing reconnect")
                            break
                        continue
                    if not raw:
                        continue
                    try:
                        msg_data = json.loads(raw)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    for m in (msg_data if isinstance(msg_data, list) else [msg_data]):
                        if _process_book_update(m):
                            if LAST_BTC_MOVE_TS > 0 and LAST_BTC_DIR != 0:
                                lag = ms_now() - LAST_BTC_MOVE_TS
                                if 10 < lag < 5000:
                                    LAG_ADAPTIVE.add_lag(lag, direction=LAST_BTC_DIR)

        except Exception as e:
            logger.error(f"Poly WS error: {e}. Retry in {retry_delay}s")
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 60)
            last_tokens = (None, None)


async def coinbase_trade_task() -> None:
    """Secondary BTC price feed via Coinbase WebSocket."""
    global LAST_BTC_MOVE_TS, LAST_BTC_PRICE, LAST_BTC_DIR
    url = "wss://ws-feed.exchange.coinbase.com"
    payload = {"type": "subscribe", "product_ids": ["BTC-USD"],
               "channels": ["ticker", "matches"]}

    while True:
        try:
            async with websockets.connect(url, ping_interval=30, ping_timeout=30) as ws:
                await ws.send(json.dumps(payload))
                logger.info("Coinbase WS connected (fallback).")
                while True:
                    raw  = await ws.recv()
                    data = json.loads(raw)
                    if data.get("type") not in ("ticker", "match", "last_match"):
                        continue
                    if "price" not in data:
                        continue
                    price = float(data["price"])
                    ts_ms = ms_now()
                    STATE.btc_price = price
                    STATE.btc_ts_ms = ts_ms

                    # Feed basis tracker when RTDS is fresh
                    if RTDS.recv_ts_ms > 0 and RTDS.age_ms() < 5000:
                        BASIS.update(RTDS.price, price)

                    if not np.isnan(LAST_BTC_PRICE) and abs(price - LAST_BTC_PRICE) >= 8.0:
                        LAST_BTC_MOVE_TS = ts_ms
                        LAST_BTC_DIR = 1 if price > LAST_BTC_PRICE else -1
                    LAST_BTC_PRICE = price
                    _update_sigma(price, ts_ms)
                    TAIL_RISK.feed_price(price, ts_ms)
                    TAIL_RISK.feed_sigma(STATE.sigma_1m)
        except Exception as e:
            logger.error(f"Coinbase WS error: {e}")
            await asyncio.sleep(1)




# ════════════════════════════════════════════════════════════════════════════
# WINDOW TIMING & MARKET DISCOVERY
# ════════════════════════════════════════════════════════════════════════════

def get_window_start_unix(offset: int = 0) -> int:
    now = datetime.now(POLYMARKET_TIMEZONE)
    floored = (now.minute // 5) * 5
    ws_local = now.replace(minute=floored, second=0, microsecond=0)
    ws_local += timedelta(minutes=5 * offset)
    return int(ws_local.timestamp())


def get_current_window_times():
    start_unix = get_window_start_unix(0)
    ws = datetime.fromtimestamp(start_unix, tz=POLYMARKET_TIMEZONE)
    return ws, ws + timedelta(minutes=5)


def fetch_crypto_price(window_start_dt: datetime, window_end_dt: datetime) -> Optional[dict]:
    """
    Fetch official strike (openPrice) and resolution (closePrice) from Polymarket's
    crypto-price API.  Returns dict with 'openPrice', 'closePrice', 'completed'
    or None on failure.
    """
    start_utc = window_start_dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    end_utc = window_end_dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    params = {
        "symbol": "BTC",
        "eventStartTime": start_utc,
        "variant": "fiveminute",
        "endDate": end_utc,
    }
    try:
        resp = requests.get(CRYPTO_PRICE_URL, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if "openPrice" in data:
            return data
        return None
    except Exception as e:
        logger.error(f"crypto-price API error: {e}")
        return None


def fetch_price_from_candle_api(window_ts: int) -> Optional[float]:
    """Legacy fallback: fetch strike from chainlink-candles endpoint."""
    end_time_ms = (window_ts + 300) * 1000 - 1
    params = {"symbol": "BTC", "interval": "5m", "limit": 2, "endTime": end_time_ms}
    try:
        resp = requests.get(CANDLE_API_URL, params=params, timeout=5)
        resp.raise_for_status()
        candles = resp.json().get("candles", [])
        for c in candles:
            if c["time"] == window_ts:
                return float(c["open"])
        return None
    except Exception as e:
        logger.error(f"Candle API error: {e}")
        return None


def _auto_discover_market_sync() -> Optional[dict]:
    logger.info("Discovering active BTC 5-min market…")
    url    = "https://gamma-api.polymarket.com/events"
    params = {"active": "true", "closed": "false", "order": "id",
              "ascending": "true", "limit": 100, "series_id": 10684}
    now = datetime.now(timezone.utc)

    for offset_idx in range(5):
        if offset_idx > 0:
            params["offset"] = offset_idx * 100
        try:
            events = requests.get(url, params=params, timeout=15).json()
            if not events:
                break
        except Exception:
            break

        for event in events:
            if not event.get("ticker", "").startswith("btc-updown-5m"):
                continue
            market    = event.get("markets", [{}])[0]
            start_str = market.get("eventStartTime") or event.get("startTime")
            end_str   = market.get("endDate")
            if not start_str or not end_str:
                continue
            try:
                st = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
                et = datetime.fromisoformat(end_str.replace("Z", "+00:00"))
            except Exception:
                continue
            if st > now or et <= now:
                continue
            clob_raw = market.get("clobTokenIds", "[]")
            try:
                clob_ids = json.loads(clob_raw) if isinstance(clob_raw, str) else clob_raw
            except Exception:
                clob_ids = []
            if len(clob_ids) >= 2:
                logger.info(f"Discovered: {event.get('slug')}")
                return {"market_id": event.get("slug"), "up_token": str(clob_ids[0]),
                        "down_token": str(clob_ids[1]), "end_time": et}
    return None


# ════════════════════════════════════════════════════════════════════════════
# REST BOOK SEEDING
# ════════════════════════════════════════════════════════════════════════════

def _seed_book_from_rest() -> None:
    global LAST_REST_SEED_HAD_QUOTES, REST_SEED_INFLIGHT
    got_real = False
    for side, tid in [("UP", UP_TOKEN_ID), ("DOWN", DOWN_TOKEN_ID)]:
        if not tid:
            continue
        try:
            url  = f"https://clob.polymarket.com/book?token_id={tid}"
            resp = requests.get(url, timeout=3)
            resp.raise_for_status()
            data = resp.json()
            bids, asks = data.get("bids", []), data.get("asks", [])
            bid_r = float(bids[0]["price"]) if bids else None
            ask_r = float(asks[0]["price"]) if asks else None
            bid_sz = float(bids[0].get("size", 0)) if bids else 0.0
            ask_sz = float(asks[0].get("size", 0)) if asks else 0.0

            prev = BOOK_CACHE.get(tid)
            bb = bid_r if bid_r is not None else (prev.best_bid if prev else None) or POLY_STATE.get(tid, {}).get("bid")
            ba = ask_r if ask_r is not None else (prev.best_ask if prev else None) or POLY_STATE.get(tid, {}).get("ask")
            if bb is None or ba is None:
                continue
            bb, ba = max(0.0, min(1.0, float(bb))), max(0.0, min(1.0, float(ba)))
            if bb > ba:
                continue

            existing = BOOK_CACHE.get(tid)
            if existing and existing.source == "ws" and (ms_now()-existing.last_update) < POLY_BOOK_MAX_AGE_MS:
                continue

            # Write to token_id key (same as WS path) so both paths share state
            if tid not in POLY_STATE:
                POLY_STATE[tid] = {
                    "bid": 0.0, "ask": 1.0, "bid_size": 0.0, "ask_size": 0.0,
                    "last_update": 0, "source": "rest"
                }
            POLY_STATE[tid]["bid"], POLY_STATE[tid]["ask"] = bb, ba
            POLY_STATE[tid]["bid_size"], POLY_STATE[tid]["ask_size"] = bid_sz, ask_sz
            src = "rest" if (bid_r and ask_r) else "fallback"
            POLY_STATE[tid]["last_update"] = ms_now()
            POLY_STATE[tid]["source"] = src
            BOOK_CACHE.update(tid, BookSnapshot(
                token_id=tid, best_bid=bb, best_ask=ba, spread=ba-bb,
                bid_size=bid_sz, ask_size=ask_sz, source=src, last_update=ms_now()))
            if (ba - bb) < 0.95:
                got_real = True
        except Exception as e:
            logger.warning(f"REST SEED {side} failed: {e}")
    LAST_REST_SEED_HAD_QUOTES = got_real


def _seed_book_from_rest_with_reset() -> None:
    global REST_SEED_INFLIGHT
    try:
        _seed_book_from_rest()
    finally:
        REST_SEED_INFLIGHT = False


# ════════════════════════════════════════════════════════════════════════════
# MARKET CLOCK TASK
# ════════════════════════════════════════════════════════════════════════════

async def market_clock_task(end_time: datetime) -> None:
    global UP_TOKEN_ID, DOWN_TOKEN_ID, BOOK_RECONNECT_FLAG
    global WINDOW_OPEN_MS, FIRST_REAL_BOOK_MS, LIQUIDITY_TIMEOUT_THIS_WINDOW
    global LAST_REST_SEED_HAD_QUOTES, LAST_REST_SEED_MS
    global TRADED_UP_THIS_WINDOW, TRADED_DN_THIS_WINDOW, TRADES_THIS_WINDOW
    global WINDOW_BANKROLL
    global WINDOW_CRYPTO_PRICE

    last_api_check = 0.0
    last_ws, _ = get_current_window_times()

    while True:
        ws, we = get_current_window_times()
        now_et = datetime.now(POLYMARKET_TIMEZONE)
        STATE.sec_remaining = float((we - now_et).total_seconds())

        # ── Window turnover ────────────────────────────────────────────────
        if last_ws != ws:
            WINDOW_OPEN_MS, FIRST_REAL_BOOK_MS = ms_now(), 0
            PERSIST.side = PERSIST.start_ms = PERSIST.wait_ms = None
            LIQUIDITY_TIMEOUT_THIS_WINDOW = False
            LAST_REST_SEED_HAD_QUOTES = False
            LAST_REST_SEED_MS = ms_now()
            brain_loop._both_illiquid_since_ms = 0

            # ── CLOB-verified PnL: end-of-window snapshot (BEFORE settlement) ──
            _pp_settle = get_portfolio_pos()
            _up_snap_settle = BOOK_CACHE.get(UP_TOKEN_ID)
            _dn_snap_settle = BOOK_CACHE.get(DOWN_TOKEN_ID)
            _model_pnl_this_window = 0.0

            # ── Settlement reconciliation (BEFORE resetting inventory) ────
            if _pp_settle.total > 0:
                _model_pnl_this_window = settle_window()

            # ── CLOB-verified PnL: compare real vs model ──────────────────
            if client is not None:
                try:
                    PNL_TRACKER.on_window_end(
                        client=client,
                        up_token_id=UP_TOKEN_ID,
                        dn_token_id=DOWN_TOKEN_ID,
                        up_bid=float(_up_snap_settle.best_bid) if _up_snap_settle else 0.0,
                        dn_bid=float(_dn_snap_settle.best_bid) if _dn_snap_settle else 0.0,
                        model_pnl=_model_pnl_this_window,
                        fifo_up_inventory=float(POS_UP.inventory),
                        fifo_dn_inventory=float(POS_DOWN.inventory),
                    )
                except Exception as _pnl_err:
                    logger.warning(f"PNL_TRACKER_END_FAIL: {_pnl_err}")

            STRIKE_CAPTURE.reset()

            # Record previous window coverage before reset
            COVERAGE.record_window(
                traded_up=TRADED_UP_THIS_WINDOW,
                traded_dn=TRADED_DN_THIS_WINDOW,
            )
            logger.info(
                f"COVERAGE UPDATE | UP={COVERAGE.coverage_up():.2%} "
                f"DN={COVERAGE.coverage_dn():.2%}"
            )
            TRADED_UP_THIS_WINDOW = False
            TRADED_DN_THIS_WINDOW = False
            TRADES_THIS_WINDOW = 0
            ENTRY_P_CONE["UP"] = None
            ENTRY_P_CONE["DOWN"] = None
            ENTRY_P_MARKET["UP"] = None
            ENTRY_P_MARKET["DOWN"] = None
            ENTRY_T_SEC["UP"] = None
            ENTRY_T_SEC["DOWN"] = None
            _FLIP_BREACH_TS["UP"] = 0
            _FLIP_BREACH_TS["DOWN"] = 0

            # Reset Portfolio + IADL + PositionMonitor for new window
            POS_UP.__init__("UP", lot_method="FIFO")
            POS_DOWN.__init__("DOWN", lot_method="FIFO")
            IADL.window_bias = None
            IADL.last_flip_ts_ms = None
            POS_MONITOR.reset_window(ms_now())

            STATE.open_price  = np.nan
            STATE.strike_type = None
            WINDOW_CRYPTO_PRICE = None
            # Reset expiry sell flags so they can fire in the next window
            for _rst_side in ("UP", "DOWN"):
                _rst_flag = f"_expiry_sell_{_rst_side}"
                if hasattr(STATE, _rst_flag):
                    setattr(STATE, _rst_flag, False)

            # Reset Z trajectory filter (clear stale slope/EMA from previous window)
            Z_TRAJ.hist.clear()
            Z_TRAJ._expands_streak = 0
            Z_TRAJ.z_ema = 0.0
            Z_TRAJ._init = False
            Z_TRAJ.set_anticipation_mode(False)

            # Reset Coinbase lead indicator for new window
            CB_LEAD.reset()

            last_ws = ws

            # Snapshot bankroll at window start for consistent sizing
            try:
                WINDOW_BANKROLL = min(get_available_usdc_balance(), WINDOW_SPEND_CAP_USD)
                logger.info(
                    f"WINDOW_BANKROLL: ${WINDOW_BANKROLL:.2f} "
                    f"(window_cap=${WINDOW_SPEND_CAP_USD:.2f})"
                )
            except Exception:
                logger.warning("Failed to snapshot bankroll, using previous")

            # ── CLOB-verified PnL: start-of-window snapshot ──────────────
            if client is not None:
                try:
                    _u_s = BOOK_CACHE.get(UP_TOKEN_ID)
                    _d_s = BOOK_CACHE.get(DOWN_TOKEN_ID)
                    PNL_TRACKER.on_window_start(
                        client=client,
                        up_token_id=UP_TOKEN_ID,
                        dn_token_id=DOWN_TOKEN_ID,
                        up_bid=float(_u_s.best_bid) if _u_s else 0.0,
                        dn_bid=float(_d_s.best_bid) if _d_s else 0.0,
                        window_id=str(ws),
                    )
                except Exception as _pnl_err:
                    logger.warning(f"PNL_TRACKER_START_FAIL: {_pnl_err}")

            # ── Rolling Sharpe update at window rollover ──────────────────
            try:
                global WIN_REALIZED_START, ROLL_SHARPE, SHARPE_MULT, SHARPE_PAUSED
                cur_realized = float(SESSION_PNL)
                if WIN_REALIZED_START is None:
                    WIN_REALIZED_START = cur_realized
                win_ret = cur_realized - WIN_REALIZED_START
                WIN_REALIZED_START = cur_realized
                WINDOW_RETURNS.append(float(win_ret))
                recent = list(WINDOW_RETURNS)[-ROLL_SHARPE_N:]
                ROLL_SHARPE = _compute_roll_sharpe(recent)
                SHARPE_MULT = max(SHARPE_MULT_MIN, min(SHARPE_MULT_MAX, _sharpe_to_mult(ROLL_SHARPE)))
                SHARPE_PAUSED = (ROLL_SHARPE <= SHARPE_HARD)
                logger.info(
                    f"ROLL_SHARPE | n={len(recent)} ret={win_ret:+.2f} "
                    f"sh={ROLL_SHARPE:+.3f} mult={SHARPE_MULT:.2f} paused={SHARPE_PAUSED}"
                )
            except Exception as e:
                logger.warning(f"ROLL_SHARPE update failed: {e}")

            loop = asyncio.get_running_loop()
            info = await loop.run_in_executor(None, _auto_discover_market_sync)
            if info:
                new_up, new_dn = info["up_token"], info["down_token"]
                if new_up != UP_TOKEN_ID or new_dn != DOWN_TOKEN_ID:
                    BOOK_RECONNECT_FLAG = True
                    for old in (UP_TOKEN_ID, DOWN_TOKEN_ID):
                        if old: BOOK_CACHE.delete(old)
                    # Clear old token entries from POLY_STATE
                    for old_tid in (UP_TOKEN_ID, DOWN_TOKEN_ID):
                        if old_tid and old_tid in POLY_STATE:
                            del POLY_STATE[old_tid]
                    UP_TOKEN_ID, DOWN_TOKEN_ID = new_up, new_dn
                    logger.info(f"Tokens updated: UP={UP_TOKEN_ID[:16]}… DOWN={DOWN_TOKEN_ID[:16]}…")
                    try:
                        if client is not None:
                            # Refresh USDC (collateral) allowance every window
                            client.update_balance_allowance(
                                params=BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
                            )
                            # Refresh conditional token allowances for new tokens
                            client.update_balance_allowance(
                                params=BalanceAllowanceParams(
                                    asset_type=AssetType.CONDITIONAL, token_id=UP_TOKEN_ID
                                )
                            )
                            client.update_balance_allowance(
                                params=BalanceAllowanceParams(
                                    asset_type=AssetType.CONDITIONAL, token_id=DOWN_TOKEN_ID
                                )
                            )
                            logger.info("USDC + token allowances refreshed for new window.")
                    except Exception as _allow_err:
                        logger.warning(f"Allowance refresh failed: {_allow_err}")
                    # Sync flow tracker + analytics with new tokens
                    FLOW.set_tokens(UP_TOKEN_ID, DOWN_TOKEN_ID)
                    GOLDSKY.set_market(MARKET_ID, UP_TOKEN_ID, DOWN_TOKEN_ID)
                    await loop.run_in_executor(None, _seed_book_from_rest)
                    await loop.run_in_executor(None, _prime_tick_size_cache, UP_TOKEN_ID)
                    await loop.run_in_executor(None, _prime_tick_size_cache, DOWN_TOKEN_ID)
            else:
                logger.error("Market rediscovery failed")

        # ── Strike lock ────────────────────────────────────────────────────
        # Priority: 1) crypto-price API (authoritative)
        #           2) RTDS buffer (closest price to boundary)
        #           3) candle API (legacy fallback)
        if STATE.strike_type is None:
            unix_ts = int(ws.timestamp())
            now_ts = now_et.timestamp()
            should_poll = (
                (280.0 <= STATE.sec_remaining <= 300.0) or
                (STATE.sec_remaining < 280.0 and now_ts - last_api_check > 5.0)
            )

            # 1) Try crypto-price API first (official strike)
            if should_poll:
                last_api_check = now_ts
                loop = asyncio.get_running_loop()
                _cp_data = await loop.run_in_executor(
                    None, fetch_crypto_price, ws, we)
                if _cp_data and "openPrice" in _cp_data:
                    WINDOW_CRYPTO_PRICE = _cp_data
                    STATE.open_price = float(_cp_data["openPrice"])
                    STATE.strike_type = "OFFICIAL"
                    logger.info(
                        f"STRIKE LOCKED (OFFICIAL): {STATE.open_price:.2f} "
                        f"(completed={_cp_data.get('completed', False)})"
                    )

            # 2) Try RTDS strike capture
            if STATE.strike_type is None:
                rtds_strike = STRIKE_CAPTURE.try_capture(unix_ts)
                if rtds_strike:
                    STATE.open_price = rtds_strike
                    STATE.strike_type = "RTDS"
                    logger.info(f"STRIKE LOCKED (RTDS): {STATE.open_price:.2f}")

            # 3) Legacy candle API fallback
            if STATE.strike_type is None and should_poll:
                cp = await loop.run_in_executor(
                    None, fetch_price_from_candle_api, unix_ts)
                if cp:
                    STATE.open_price = cp
                    STATE.strike_type = "CANDLE"
                    logger.info(f"STRIKE LOCKED (CANDLE): {STATE.open_price:.2f}")

        await asyncio.sleep(1.0)


# ════════════════════════════════════════════════════════════════════════════
# ORACLE PRICE SELECTION (THE CRITICAL FUNCTION)
# ════════════════════════════════════════════════════════════════════════════

def get_best_oracle_price() -> tuple:
    """
    Returns (price, source_name, age_ms).

    Priority:
      1. RTDS (Chainlink Data Streams via Polymarket WS) — the resolution feed
      2. Coinbase (secondary — sigma computation + fallback)
      3. On-chain Chainlink (emergency)
    """
    now = ms_now()

    # 1) RTDS — THE resolution feed
    if RTDS.recv_ts_ms > 0:
        age = now - RTDS.recv_ts_ms
        if age < RTDS_FRESH_MS:
            return RTDS.price, "rtds", age

    # 2) Coinbase (secondary fallback) — apply basis correction if available
    if STATE.btc_ts_ms > 0:
        age = now - STATE.btc_ts_ms
        if age < BTC_FEED_MAX_AGE_MS * 2:
            if BASIS._init and BASIS.is_stable:
                return BASIS.corrected_price(STATE.btc_price), "coinbase_corrected", age
            return STATE.btc_price, "coinbase", age

    # 3) On-chain
    if STATE.onchain_ts_ms > 0:
        age = now - STATE.onchain_ts_ms
        if age < 120_000:
            return STATE.onchain_price, "onchain", age

    return np.nan, "none", 999_999


# ════════════════════════════════════════════════════════════════════════════
# SURVIVAL ENGINE
# ════════════════════════════════════════════════════════════════════════════

def target_delta(p_up, z, sec_remaining, total_inv):
    conf = min(1.0, abs(float(z)) / 1.25) if np.isfinite(z) else 0.0
    t = max(0.0, min(1.0, sec_remaining / 90.0))
    d = max(-1.0, min(1.0, (float(p_up) - 0.5) / 0.25))
    return 0.80 * conf * t * d * total_inv


def kelly_fraction(edge, p_up, total_inv):
    gross = float(max(1.0, total_inv))
    var = float(max(1e-6, p_up * (1.0 - p_up) * (gross ** 2)))
    return float(edge / var)


def survival_decide(*, side, p, z, sec_remaining, btc_price, open_price,
                    sigma_1m, up_snap, dn_snap, pos):
    """
    DEPRECATED: Use survival_decide_portfolio() instead.
    Called every tick while position active.
    Returns action dict: HOLD|TRIM|HEDGE|EXIT|PANIC|FLIP

    Upgraded from fast_boy_12: includes FLIP action, EV_trim
    decomposition, and full inventory-aware EV calculation.
    """
    p_now  = float(p)
    pos_p  = float(pos.entry_p) if np.isfinite(pos.entry_p) else p_now
    peak_p = float(pos.max_p) if np.isfinite(pos.max_p) else p_now

    p_drop_from_entry = p_now - pos_p
    p_drop_from_peak  = p_now - peak_p

    tau_min = max(sec_remaining / 60.0, 1e-6)
    sigma_T = float(sigma_1m * math.sqrt(tau_min))
    dist    = abs(math.log(btc_price / open_price)) if btc_price > 0 and open_price > 0 else float("inf")
    danger  = dist < 1.25 * sigma_T

    up_spread = up_snap.best_ask - up_snap.best_bid
    dn_spread = dn_snap.best_ask - dn_snap.best_bid

    A = float(pos.up_shares)
    B = float(pos.down_shares)
    q = float(pos.shares)
    total_inv = A + B
    delta_inv = A - B
    p_up = float(max(0.0, min(1.0, p_now)))

    d_tgt = target_delta(p_up, z, sec_remaining, total_inv)
    need = delta_inv - d_tgt

    # ── Fee-aware helpers ────────────────────────────────────────────────
    def _sell_val(bid): return max(0.0, bid - fee_per_share(bid))
    def _buy_cost(ask): return min(1.0, ask + fee_per_share(ask))

    p_win    = float(max(0.0, min(1.0, p_now if pos.side == "UP" else (1.0 - p_now))))
    own_snap = up_snap if pos.side == "UP" else dn_snap
    hdg_snap = dn_snap if pos.side == "UP" else up_snap
    bid_own  = float(own_snap.best_bid)
    ask_hdg  = float(hdg_snap.best_ask)
    bid_up, bid_dn = float(up_snap.best_bid), float(dn_snap.best_bid)

    # ── Inventory-aware EV calculations ──────────────────────────────────
    EV_hold = p_up * A + (1.0 - p_up) * B
    EV_exit = A * _sell_val(bid_up) + B * _sell_val(bid_dn)

    # Gamma penalty (digital gamma is VIOLENT ATM near expiry)
    # Nonlinear scaling: (30/T)^1.5 captures pin risk in last 60-90s
    atmness    = math.exp(-0.5 * float(z)**2) if np.isfinite(z) else 0.0
    gamma_pen  = q * 0.04 * atmness * (30.0 / max(sec_remaining, 1.0)) ** 1.5
    # Late-window gamma boost
    if sec_remaining < 45:
        gamma_pen *= 2.0
    elif sec_remaining < 75:
        gamma_pen *= 1.5

    # Theta penalty (penalizes holding when uncertainty is high)
    uncertainty = 4.0 * p_up * (1.0 - p_up)
    t_frac      = min(1.0, sec_remaining / 300.0)
    THETA_K     = 0.01
    theta_pen   = (A + B) * THETA_K * uncertainty * t_frac

    EV_hold_adj = EV_hold - gamma_pen - theta_pen

    # ── Kelly sizing ─────────────────────────────────────────────────────
    edge_hold = EV_hold_adj - EV_exit
    f_kelly = kelly_fraction(edge_hold, p_up, total_inv)
    KELLY_SCALE = 0.25
    f_used = max(0.0, min(1.0, KELLY_SCALE * f_kelly))
    h_base = min(abs(need), 25.0)
    h = float(max(1.0, math.floor(h_base * f_used)))

    # ── EV_hedge: buy opposite leg ───────────────────────────────────────
    EV_hedge = -1e9
    if np.isfinite(ask_hdg) and 0 < ask_hdg < 1:
        if pos.side == "UP":
            EV_hedge = p_up * A + (1-p_up) * (B + h) - h * _buy_cost(ask_hdg)
        else:
            EV_hedge = p_up * (A + h) + (1-p_up) * B - h * _buy_cost(ask_hdg)

    # ── EV_flip: sell all own, buy opposite ──────────────────────────────
    p_win_new     = 1.0 - p_win
    sell_proceeds = q * _sell_val(bid_own)
    buy_cost_flip = q * _buy_cost(ask_hdg)
    EV_flip       = sell_proceeds - buy_cost_flip + (q * p_win_new)
    if not np.isfinite(ask_hdg) or ask_hdg <= 0 or ask_hdg >= 1:
        EV_flip = -1e9
    elif f_used < 0.10:
        EV_flip = -1e9  # gated by Kelly minimum

    # ── EV_trim: realize k shares, optimize remainder ────────────────────
    k = float(max(1.0, math.floor(h)))
    if k >= q:
        k = float(max(1.0, q - 1.0))
    q_rem = q - k
    proceeds_trim = k * _sell_val(bid_own)
    A_rem = max(0.0, A - k) if pos.side == "UP" else A
    B_rem = max(0.0, B - k) if pos.side == "DOWN" else B
    EV_hold_rem = p_up * A_rem + (1-p_up) * B_rem
    theta_pen_rem = (A_rem + B_rem) * THETA_K * uncertainty * t_frac
    EV_hold_rem_adj = EV_hold_rem - (gamma_pen * q_rem / max(q, 1)) - theta_pen_rem
    EV_exit_rem = A_rem * _sell_val(bid_up) + B_rem * _sell_val(bid_dn)
    EV_trim = proceeds_trim + max(EV_hold_rem_adj, EV_exit_rem)

    # ── Adaptive EV_EPS ──────────────────────────────────────────────────
    EV_EPS_SHARE = max(0.005, min(0.04, 20.0 * sigma_T))
    EV_EPS = EV_EPS_SHARE * q

    ev_meta = {
        "ev_hold": round(EV_hold, 4), "ev_hold_adj": round(EV_hold_adj, 4),
        "ev_exit": round(EV_exit, 4), "ev_hedge": round(EV_hedge, 4),
        "ev_flip": round(EV_flip, 4), "ev_trim": round(EV_trim, 4),
        "p_win": round(p_win, 4), "h": h, "k": k,
        "gamma_pen": round(gamma_pen, 4), "theta_pen": round(theta_pen, 4),
        "inv_up": round(A, 1), "inv_dn": round(B, 1),
    }

    # ── Action helpers ───────────────────────────────────────────────────
    def _sell(reason, size=None, action="EXIT"):
        own_sp = own_snap.best_ask - own_snap.best_bid
        hdg_sp = hdg_snap.best_ask - hdg_snap.best_bid
        if own_sp <= 0.50 and own_snap.best_bid > 0:
            lim    = round(max(0.01, own_snap.best_bid - 0.01), 2)
            tok    = pos.token_id
            o_side = "SELL"
        elif hdg_sp <= 0.50 and hdg_snap.best_ask > 0:
            lim    = round(min(0.99, hdg_snap.best_ask + 0.01), 2)
            tok    = DOWN_TOKEN_ID if pos.side == "UP" else UP_TOKEN_ID
            o_side = "BUY"
        else:
            lim    = round(max(0.01, own_snap.best_bid - 0.01), 2)
            tok    = pos.token_id
            o_side = "SELL"
        return {"action": action, "reason": reason,
                "size": size if size is not None else q,
                "token_id": tok, "limit": lim,
                "order_side": o_side, **ev_meta}

    def _hedge(reason, frac=None):
        sz = float(max(1.0, math.floor(q * frac))) if frac else h
        lim = round(min(0.99, hdg_snap.best_ask + 0.01), 2)
        htok = DOWN_TOKEN_ID if pos.side == "UP" else UP_TOKEN_ID
        inv_d = {"up": 0.0, "down": sz} if pos.side == "UP" else {"up": sz, "down": 0.0}
        return {"action": "HEDGE", "reason": reason, "size": sz,
                "token_id": htok, "limit": lim, "order_side": "BUY",
                "inv_delta": inv_d, **ev_meta}

    def _trim(reason, size=None):
        lim = round(max(0.01, own_snap.best_bid - 0.01), 2)
        sz = float(size) if size else h
        inv_d = ({"up": -sz, "down": 0.0} if pos.side == "UP"
                 else {"up": 0.0, "down": -sz})
        return {"action": "TRIM", "reason": reason, "size": sz,
                "token_id": pos.token_id, "limit": lim, "order_side": "SELL",
                "inv_delta": inv_d, **ev_meta}

    def _panic(reason):
        return {"action": "PANIC", "reason": reason, "size": 0,
                "token_id": "", "limit": 0, "order_side": "SELL", **ev_meta}

    # ── Hard override rules ────────────────────────────────────────────────
    # Dynamic drawdown threshold
    if sec_remaining > 60:
        drawdown_threshold = -0.20
    elif sec_remaining > 30:
        drawdown_threshold = -0.15
    else:
        drawdown_threshold = -0.10

    # Pinning guard
    if sec_remaining < 90 and abs(z) < 0.90:
        if sec_remaining < 60:
            return _sell("PINNING_FORCE_EXIT")
        return _sell("PINNING_TRIM_75", size=q*0.75, action="TRIM")

    if sec_remaining < 60 and p_drop_from_peak < -0.12:
        return _sell("LATE_DRAWDOWN")

    if sec_remaining <= 8:
        if up_spread > 0.90 and dn_spread > 0.90:
            return _panic("last_8s_no_book")
        return _sell("last_8s")

    if up_spread > 0.60 and dn_spread > 0.60:
        own_sp = own_snap.best_ask - own_snap.best_bid
        hdg_sp = hdg_snap.best_ask - hdg_snap.best_bid
        if own_sp < 0.90:
            return _sell("panic_sell_own_book")
        if hdg_sp < 0.90:
            return _hedge("panic_hedge_opp_book", frac=1.0)
        return _panic("both_books_wide")

    crossed = (btc_price < open_price) if pos.side == "UP" else (btc_price > open_price)
    if crossed:
        if sec_remaining < 40:
            return _sell("crossed_strike_late")
        return _trim("crossed_strike_early")

    if p_drop_from_peak < drawdown_threshold:
        return _sell("p_drawdown")

    if sec_remaining < 20 and danger:
        return _hedge("danger_band_last20s", frac=1.0)

    if sec_remaining < 35 and danger and p_drop_from_entry < -0.10:
        return _trim("danger_band_p_slip_35s")

    # ── EV engine (5-way: hold, exit, hedge, trim, flip) ─────────────────
    best = max(EV_hold_adj, EV_exit, EV_hedge, EV_trim, EV_flip)

    if (best == EV_exit and
            (EV_exit - EV_hold_adj) > EV_EPS and
            (EV_exit - max(EV_hedge, EV_trim, EV_flip)) > EV_EPS):
        return _sell("ev_exit")

    if (best == EV_hedge and
            (EV_hedge - EV_hold_adj) > EV_EPS and
            (EV_hedge - max(EV_exit, EV_trim, EV_flip)) > EV_EPS):
        return _hedge("ev_hedge")

    if (best == EV_trim and
            (EV_trim - EV_hold_adj) > EV_EPS and
            (EV_trim - max(EV_exit, EV_hedge, EV_flip)) > EV_EPS):
        return _sell("ev_trim", size=k, action="TRIM")

    if (best == EV_flip and
            (EV_flip - EV_hold_adj) > EV_EPS and
            (EV_flip - max(EV_exit, EV_hedge, EV_trim)) > EV_EPS):
        flip_side  = "DOWN" if pos.side == "UP" else "UP"
        flip_token = DOWN_TOKEN_ID if pos.side == "UP" else UP_TOKEN_ID
        flip_limit = round(min(0.99, hdg_snap.best_ask + 0.01), 2)
        return {
            "action": "FLIP", "reason": "ev_flip", "size": q,
            "token_id": pos.token_id,
            "limit": round(max(0.01, own_snap.best_bid - 0.01), 2),
            "order_side": "SELL",
            "flip_side": flip_side, "flip_token_id": flip_token,
            "flip_limit": flip_limit, "flip_size": q,
            **ev_meta,
        }

    return {"action": "HOLD", "reason": "ev_hold", "size": 0,
            "token_id": "", "limit": 0, "order_side": "SELL", **ev_meta}


def survival_decide_portfolio(*, p_up: float, z: float, sec_remaining: float,
                              up_snap, dn_snap, pos: PortfolioPos):
    """
    Portfolio-native survival engine.
    Operates on real FIFO inventory (PortfolioPos).
    Returns dict with action, reason, token_id, order_side, size, limit.
    Actions: HOLD | EXIT | TRIM | HEDGE
    """
    A = float(pos.up)    # UP shares
    B = float(pos.dn)    # DOWN shares
    q = A + B
    if q <= 0:
        return {"action": "HOLD", "reason": "flat", "size": 0,
                "token_id": "", "limit": 0, "order_side": "SELL"}

    p_up = float(max(0.0, min(1.0, p_up)))

    # Fee-aware helpers
    def sell_val(bid): return max(0.0, float(bid) - fee_per_share(float(bid)))
    def buy_cost(ask): return min(1.0, float(ask) + fee_per_share(float(ask)))

    bid_up, ask_up = float(up_snap.best_bid), float(up_snap.best_ask)
    bid_dn, ask_dn = float(dn_snap.best_bid), float(dn_snap.best_ask)

    # Current liquidation value
    EV_exit = A * sell_val(bid_up) + B * sell_val(bid_dn)

    # Expected settlement value if held to expiry
    EV_hold = p_up * A + (1.0 - p_up) * B

    # Risk terms
    uncertainty = 4.0 * p_up * (1.0 - p_up)  # 0..1: maximal at p=0.5
    t_frac = min(1.0, max(0.0, sec_remaining / 300.0))
    theta_pen = q * 0.01 * uncertainty * t_frac

    # Gamma penalty: digital gamma is violent ATM near expiry
    atmness = math.exp(-0.5 * float(z)**2) if np.isfinite(z) else 0.0
    gamma_pen = q * 0.04 * atmness * (30.0 / max(sec_remaining, 1.0)) ** 1.5
    if sec_remaining < 45:
        gamma_pen *= 2.0
    elif sec_remaining < 75:
        gamma_pen *= 1.5

    EV_hold_adj = EV_hold - theta_pen - gamma_pen

    # Hard: last 8 seconds, flatten whatever exists
    if sec_remaining <= 8:
        if A > 0 and bid_up > 0.01:
            return {"action": "EXIT", "reason": "last_8s", "token_id": UP_TOKEN_ID,
                    "order_side": "SELL", "size": A,
                    "limit": round(max(0.01, bid_up - 0.01), 2)}
        if B > 0 and bid_dn > 0.01:
            return {"action": "EXIT", "reason": "last_8s", "token_id": DOWN_TOKEN_ID,
                    "order_side": "SELL", "size": B,
                    "limit": round(max(0.01, bid_dn - 0.01), 2)}
        return {"action": "HOLD", "reason": "last_8s_no_book", "size": 0,
                "token_id": "", "limit": 0, "order_side": "SELL"}

    # If exit clearly dominates hold, trim the larger leg
    EV_EPS = max(0.01, min(0.05, 0.5 * q * uncertainty))
    if (EV_exit - EV_hold_adj) > EV_EPS:
        if A >= B and A > 0 and bid_up > 0.01:
            ok_up, net_up, req_up = early_sell_profit_gate(
                sec_remaining=sec_remaining,
                bid=bid_up,
                entry_price=pos.avg_cost_up,
                p_cone=p_up,
                side="UP",
                T_entry=ENTRY_T_SEC.get("UP") or 230.0,
            )
            if ok_up:
                return {"action": "TRIM", "reason": "ev_exit_pref", "token_id": UP_TOKEN_ID,
                        "order_side": "SELL",
                        "size": max(1.0, math.floor(A * 0.6)),
                        "limit": round(max(0.01, bid_up - 0.01), 2)}
        if B > 0 and bid_dn > 0.01:
            ok_dn, net_dn, req_dn = early_sell_profit_gate(
                sec_remaining=sec_remaining,
                bid=bid_dn,
                entry_price=pos.avg_cost_dn,
                p_cone=p_up,
                side="DOWN",
                T_entry=ENTRY_T_SEC.get("DOWN") or 230.0,
            )
            if ok_dn:
                return {"action": "TRIM", "reason": "ev_exit_pref", "token_id": DOWN_TOKEN_ID,
                        "order_side": "SELL",
                        "size": max(1.0, math.floor(B * 0.6)),
                        "limit": round(max(0.01, bid_dn - 0.01), 2)}
        # Compute dynamic targets for logging
        _up_te = ENTRY_T_SEC.get("UP") or 230.0
        _dn_te = ENTRY_T_SEC.get("DOWN") or 230.0
        _, _u_net, _u_req = early_sell_profit_gate(sec_remaining, bid_up, pos.avg_cost_up,
                                                    p_up, "UP", _up_te)
        _, _d_net, _d_req = early_sell_profit_gate(sec_remaining, bid_dn, pos.avg_cost_dn,
                                                    p_up, "DOWN", _dn_te)
        if sec_remaining > EXIT_HARD_FLOOR_T:
            return {"action": "HOLD", "reason": "alpha_exit_gate", "size": 0,
                    "token_id": "", "limit": 0, "order_side": "SELL",
                    "up_net": round(_u_net, 4), "up_req": round(_u_req, 4),
                    "dn_net": round(_d_net, 4), "dn_req": round(_d_req, 4)}
        return {"action": "HOLD", "reason": "ev_exit_no_book", "size": 0,
                "token_id": "", "limit": 0, "order_side": "SELL"}

    # Hedge only when very uncertain near expiry (pin risk)
    if sec_remaining < 35 and abs(float(z)) < 0.9 and uncertainty > 0.85:
        if A > B and ask_dn < 0.99 and ask_dn > 0.01:
            h = max(1.0, math.floor((A - B) * 0.5))
            return {"action": "HEDGE", "reason": "pin_risk_balance",
                    "token_id": DOWN_TOKEN_ID, "order_side": "BUY",
                    "size": h, "limit": round(min(0.99, ask_dn + 0.01), 2)}
        if B > A and ask_up < 0.99 and ask_up > 0.01:
            h = max(1.0, math.floor((B - A) * 0.5))
            return {"action": "HEDGE", "reason": "pin_risk_balance",
                    "token_id": UP_TOKEN_ID, "order_side": "BUY",
                    "size": h, "limit": round(min(0.99, ask_up + 0.01), 2)}

    return {"action": "HOLD", "reason": "ev_hold", "size": 0,
            "token_id": "", "limit": 0, "order_side": "SELL",
            "ev_hold": round(EV_hold_adj, 4), "ev_exit": round(EV_exit, 4)}


def maybe_request_flip(ts_ms: int, p_cone: float, z: float, T_sec: float,
                       up_snap, dn_snap):
    """
    Portfolio-native flip: sell dominant leg first, then arm BUY intent.
    Returns (True, sell_orders) or (False, []).
    Gated by IADL reversal discipline.
    """
    total_inv = POS_UP.inventory + POS_DOWN.inventory
    if total_inv <= 0:
        return False, []

    cur_side = "UP" if POS_UP.inventory >= POS_DOWN.inventory else "DOWN"
    model_side = "UP" if p_cone >= 0.5 else "DOWN"

    if model_side == cur_side:
        return False, []

    # Gate through IADL (enforces reversal_z, dominance, cooldown)
    try:
        dec = IADL.gate(
            ts_ms=ts_ms,
            portfolio=PORTFOLIO,
            proposed_side=model_side,
            z=float(z),
            p_cone=float(p_cone),
            T_sec=float(T_sec),
            up_bid=float(up_snap.best_bid), up_ask=float(up_snap.best_ask),
            dn_bid=float(dn_snap.best_bid), dn_ask=float(dn_snap.best_ask),
            debug_in={"reason": "FLIP_CHECK"},
        )
        if dec.kind not in (DecisionType.CLOSE_THEN_REVERSE, DecisionType.UNWIND_STRADDLE):
            return False, []
    except Exception as e:
        logger.warning(f"IADL flip gate error: {e}")
        return False, []

    # Build sell plan: sell dominant leg
    if cur_side == "UP":
        sell_tid = UP_TOKEN_ID
        sell_qty = float(POS_UP.inventory)
        sell_bid = float(up_snap.best_bid)
    else:
        sell_tid = DOWN_TOKEN_ID
        sell_qty = float(POS_DOWN.inventory)
        sell_bid = float(dn_snap.best_bid)

    if sell_qty <= 0 or sell_bid <= 0.01:
        return False, []

    sell_entry = float(POS_UP.avg_entry_price) if cur_side == "UP" else float(POS_DOWN.avg_entry_price)
    flip_ok, net_bid, req_bid = early_sell_profit_gate(
        sec_remaining=T_sec,
        bid=sell_bid,
        entry_price=sell_entry,
    )
    if not flip_ok:
        logger.info(
            f"FLIP_BLOCKED_PROFIT_GATE: side={cur_side} T={T_sec:.0f}s "
            f"net_bid={net_bid:.3f} req={req_bid:.3f}"
        )
        return False, []

    sell_limit = round(max(0.01, sell_bid - 0.01), 2)

    # Determine buy leg
    if model_side == "UP":
        buy_tok = UP_TOKEN_ID
        buy_ask = float(up_snap.best_ask)
    else:
        buy_tok = DOWN_TOKEN_ID
        buy_ask = float(dn_snap.best_ask)

    buy_limit = round(min(0.99, buy_ask + 0.01), 2)

    # Arm pending intent
    PENDING_FLIP_INTENT.update({
        "active": True,
        "buy_token_id": buy_tok,
        "buy_side": model_side,
        "buy_limit": buy_limit,
        "buy_size": sell_qty,
        "expires_ms": ts_ms + 4000,
    })

    # Return sell order specification
    sell_order = {
        "token_side": cur_side,
        "token_id": sell_tid,
        "qty": sell_qty,
        "limit_price": sell_limit,
    }
    return True, [sell_order]


# ════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ════════════════════════════════════════════════════════════════════════════

async def dashboard_task() -> None:
    while True:
        _pp = get_portfolio_pos()
        src = LATEST_DEBUG  # always use live values (survival recomputes p/z each tick)

        p      = src.get("p_cone", np.nan)
        z_val  = src.get("z", np.nan)
        edge   = src.get("edge", 0.0)
        side   = src.get("side", "-")
        reason = src.get("reason", "Wait")

        # Oracle info
        ora_px, ora_src, ora_age = get_best_oracle_price()
        ora_s = f"${ora_px:.2f}({ora_src},{ora_age/1000:.0f}s)" if np.isfinite(ora_px) else "---"

        btc_s    = f"${STATE.btc_price:.2f}" if not np.isnan(STATE.btc_price) else "---"
        rtds_s   = f"${RTDS.price:.2f}" if RTDS.is_fresh(RTDS_FRESH_MS) else "---"
        strike_s = f"${STATE.open_price:.2f}" if not np.isnan(STATE.open_price) else "---"
        rem_s    = f"{STATE.sec_remaining:.0f}s" if not np.isnan(STATE.sec_remaining) else "---"

        if _pp.total > 0:
            pos_str = f"[INV:{_pp.dominant_side} U={_pp.up:.0f} D={_pp.dn:.0f}]"
            gate = "POSITION_OPEN"
        else:
            pos_str = ""
            gate = reason

        cb_str = " [CB:PAUSED]" if CIRCUIT_BREAKER_ACTIVE else ""

        try:
            sys.stdout.write(
                f"\r\033[K[Live] ORA:{ora_s} | CB:{btc_s} | CL:{rtds_s}"
                f" | K:{strike_s}({STATE.strike_type})"
                f" | T:{rem_s} | sig:{STATE.sigma_1m:.4f}(w={STATE.sigma_w:.2f})"
                f" | p:{p:.3f} z:{z_val:.2f} e:{edge:.4f}"
                f" | PnL:{SESSION_PNL:+.2f}{cb_str}"
                f" | ${WINDOW_BANKROLL:.0f}"
                f"({'U' + str(int(_pp.up)) if _pp.up > 0 else ''}"
                f"{'D' + str(int(_pp.dn)) if _pp.dn > 0 else ''})"
                f" => {side} {gate} {pos_str}"
            )
            sys.stdout.flush()
        except UnicodeEncodeError:
            pass
        await asyncio.sleep(1.0)


# ════════════════════════════════════════════════════════════════════════════
# BRAIN LOOP
# ════════════════════════════════════════════════════════════════════════════

async def brain_loop(eq: asyncio.Queue) -> None:
    global LATEST_DEBUG, LAST_FIRE_SNAPSHOT
    global REST_SEED_INFLIGHT, LAST_REST_SEED_MS
    global LAST_STALE_BOOK_LOG_MS, LAST_NON_WS_BOOK_LOG_MS
    global WINDOW_OPEN_MS, FIRST_REAL_BOOK_MS, LIQUIDITY_TIMEOUT_THIS_WINDOW
    global TRADED_UP_THIS_WINDOW, TRADED_DN_THIS_WINDOW, TRADES_THIS_WINDOW
    global LAST_REST_SEED_HAD_QUOTES, CIRCUIT_BREAKER_ACTIVE
    global DAY_KILL_ACTIVE

    logger.info("Brain loop operational.")

    _last_status_log_ms = 0
    _STATUS_LOG_INTERVAL_MS = 10_000  # log brain state every 10s

    while True:
        await asyncio.sleep(0.01)  # 100 Hz

        # ── Periodic status log (every 10s) ────────────────────────────────
        _now = ms_now()
        if _now - _last_status_log_ms >= _STATUS_LOG_INTERVAL_MS:
            _last_status_log_ms = _now
            _reason = LATEST_DEBUG.get("reason", "?")
            _o_src  = LATEST_DEBUG.get("oracle_source", "?")
            _rtds_age = RTDS.age_ms()
            _rtds_up  = RTDS.updates
            _fill_count = int(FLOW.trade_velocity(window_s=30.0) * 30)
            _up_snap  = BOOK_CACHE.get(UP_TOKEN_ID)
            _dn_snap  = BOOK_CACHE.get(DOWN_TOKEN_ID)
            _up_age   = (_now - _up_snap.last_update) if _up_snap else 999999
            _dn_age   = (_now - _dn_snap.last_update) if _dn_snap else 999999
            _up_src   = _up_snap.source if _up_snap else "none"
            _dn_src   = _dn_snap.source if _dn_snap else "none"
            _up_sp    = f"{_up_snap.best_bid:.2f}/{_up_snap.best_ask:.2f}" if _up_snap else "?/?"
            _dn_sp    = f"{_dn_snap.best_bid:.2f}/{_dn_snap.best_ask:.2f}" if _dn_snap else "?/?"
            logger.info(
                f"BRAIN: reason={_reason} | regime={REGIME.label} | "
                f"flow={FLOW.imbalance(up_price=_up_snap.best_bid if _up_snap else 0.5, down_price=_dn_snap.best_bid if _dn_snap else 0.5):.2f}({_fill_count}fills) | "
                f"oracle={_o_src}(age={_rtds_age}ms,n={_rtds_up}) | "
                f"btc_age={_now - STATE.btc_ts_ms}ms | "
                f"strike={STATE.open_price:.2f}({STATE.strike_type}) | "
                f"T={STATE.sec_remaining:.0f}s | sig={STATE.sigma_1m:.5f} | "
                f"UP={_up_sp}({_up_src},{_up_age}ms) DN={_dn_sp}({_dn_src},{_dn_age}ms) | "
                f"pos={'INV:' + ('UP' if POS_UP.inventory >= POS_DOWN.inventory else 'DOWN') if (POS_UP.inventory + POS_DOWN.inventory) > 0 else 'none'}"
            )

            # ── Lag sampling (append to JSONL every 10s when RTDS active) ──
            if "rtds" in _o_src.lower() and _rtds_age < 60_000:
                _p_cone = LATEST_DEBUG.get("p_cone", float('nan'))
                _z_val  = LATEST_DEBUG.get("z", float('nan'))
                _dir_str = "up" if LAST_BTC_DIR > 0 else ("down" if LAST_BTC_DIR < 0 else "neutral")
                try:
                    import json as _json2
                    _lag_row = {
                        "ts_ms": _now,
                        "lag_ms": _rtds_age,
                        "direction": _dir_str,
                        "p_cone": round(float(_p_cone), 4) if np.isfinite(_p_cone) else None,
                        "z": round(float(_z_val), 3) if np.isfinite(_z_val) else None,
                        "n_updates": _rtds_up,
                        "btc_age_ms": _now - STATE.btc_ts_ms,
                        "strike": round(STATE.open_price, 2) if np.isfinite(STATE.open_price) else None,
                        "sigma": round(STATE.sigma_1m, 6),
                        "T": round(STATE.sec_remaining, 1) if np.isfinite(STATE.sec_remaining) else None,
                    }
                    with open("logs/lag_samples.jsonl", "a") as _lf:
                        _lf.write(_json2.dumps(_lag_row) + "\n")
                    logger.info(
                        f"LAG_SAMPLE: dir={_dir_str} lag={_rtds_age}ms "
                        f"p={_p_cone:.2f} z={_z_val:.2f}" if np.isfinite(_p_cone) else
                        f"LAG_SAMPLE: dir={_dir_str} lag={_rtds_age}ms"
                    )
                except Exception:
                    pass

        # ── Circuit breaker ────────────────────────────────────────────────
        if CIRCUIT_BREAKER_ACTIVE:
            if ms_now() >= CIRCUIT_BREAKER_UNTIL:
                CIRCUIT_BREAKER_ACTIVE = False
                logger.info("Circuit breaker lifted.")
            else:
                LATEST_DEBUG = {"reason": "CIRCUIT_BREAKER"}
                continue

        # ── Get best oracle price ──────────────────────────────────────────
        oracle_px, oracle_src, oracle_age = get_best_oracle_price()

        LATEST_DEBUG = {
            "reason": "Wait", "p_cone": np.nan, "z": np.nan,
            "edge": 0.0, "side": "-", "oracle_source": oracle_src,
        }

        # ── Oracle diff logging (verify basis alignment) ──────────────────
        if not hasattr(brain_loop, "_last_diff_log_ms"):
            brain_loop._last_diff_log_ms = 0
        if (_now - brain_loop._last_diff_log_ms >= 30_000 and
                RTDS.is_fresh(60_000) and STATE.btc_ts_ms > 0 and
                not np.isnan(STATE.btc_price)):
            brain_loop._last_diff_log_ms = _now
            diff = RTDS.price - STATE.btc_price
            logger.info(
                f"ORACLE_DIFF: rtds=${RTDS.price:.2f} coinbase=${STATE.btc_price:.2f} "
                f"delta={diff:+.2f} | using={oracle_src}"
            )

        # Feed RTDS prices into strike capture
        if RTDS.recv_ts_ms > 0:
            STRIKE_CAPTURE.feed(RTDS.price, RTDS.recv_ts_ms)

        # Preview cone
        rp = getattr(brain_loop, '_last_rp', None)  # preserve across iterations
        if (np.isfinite(oracle_px) and not np.isnan(STATE.open_price) and
                STATE.sec_remaining > 0):
            # Feed regime classifier every tick
            _prev_oracle = LATEST_DEBUG.get("_prev_oracle_px", oracle_px)
            delta_sign = 1 if oracle_px > _prev_oracle else (-1 if oracle_px < _prev_oracle else 0)
            rp = REGIME.update(
                sigma_eff=STATE.sigma_1m,
                delta_sign=delta_sign,
                z=float(LATEST_DEBUG.get("z", 0.0)),
                sigma_fast=STATE.sigma_fast if hasattr(STATE, 'sigma_fast') else STATE.sigma_1m,
                sigma_slow=STATE.sigma_slow if hasattr(STATE, 'sigma_slow') else STATE.sigma_1m,
                spread=float(LATEST_DEBUG.get("spread_max", 0.01)),
            )
            LATEST_DEBUG["regime"] = REGIME.label

            # Feed momentum engine each tick.
            # Pull fresh book snaps locally here because the canonical `up_snap`/`dn_snap`
            # assignments happen later in the loop.
            _up_for_ofi = BOOK_CACHE.get(UP_TOKEN_ID)
            _dn_for_ofi = BOOK_CACHE.get(DOWN_TOKEN_ID)
            _ofi = (
                FLOW.imbalance(
                    up_price=_up_for_ofi.best_bid,
                    down_price=_dn_for_ofi.best_bid,
                )
                if (
                    _up_for_ofi
                    and _dn_for_ofi
                    and _up_for_ofi.best_bid > 0
                    and _dn_for_ofi.best_bid > 0
                )
                else 0.0
            )
            _mom_regime = MOMENTUM.update(
                ts_ms=ms_now(),
                z=float(LATEST_DEBUG.get("z", 0.0)),
                ofi=_ofi,
                price=oracle_px,
                jump_regime=BIPOWER.jump_regime,
            )
            LATEST_DEBUG.update(MOMENTUM.status_dict())

            # Feed VWAP overlay each tick
            VWAP.feed(oracle_px)
            LATEST_DEBUG.update(VWAP.status_dict())
            LATEST_DEBUG.update(SPRT.status_dict())

            # Feed jump detector with log-returns
            _prev_px = LATEST_DEBUG.get("_prev_oracle_px", oracle_px)
            if _prev_px > 0 and oracle_px > 0:
                r_t = math.log(oracle_px / _prev_px)
                _sigma_slow_j = STATE.sigma_slow if hasattr(STATE, 'sigma_slow') else STATE.sigma_1m
                is_jump = JUMP_DETECTOR.update(r_t, _sigma_slow_j)
                if is_jump:
                    LATEST_DEBUG["jump_detected"] = True
                # Feed bipower jump filter
                _bp_active = BIPOWER.feed(r_t)
                LATEST_DEBUG.update(BIPOWER.status_dict())
            LATEST_DEBUG["_prev_oracle_px"] = oracle_px

            p, z = cone_p_and_z(oracle_px, STATE.open_price,
                                STATE.sec_remaining, STATE.sigma_1m,
                                calibration_alpha=rp.calibration_alpha)
            LATEST_DEBUG.update({"p_cone": p, "z": z})
            brain_loop._last_rp = rp  # cache for use outside this block

        up_snap = BOOK_CACHE.get(UP_TOKEN_ID)
        dn_snap = BOOK_CACHE.get(DOWN_TOKEN_ID)

        # Auto-reseed stale books before signal evaluation
        if not up_snap or not BOOK_CACHE.is_fresh(UP_TOKEN_ID, 800):
            try:
                await asyncio.get_running_loop().run_in_executor(None, _seed_book_from_rest)
                up_snap = BOOK_CACHE.get(UP_TOKEN_ID)
            except Exception:
                pass
        if not dn_snap or not BOOK_CACHE.is_fresh(DOWN_TOKEN_ID, 800):
            try:
                await asyncio.get_running_loop().run_in_executor(None, _seed_book_from_rest)
                dn_snap = BOOK_CACHE.get(DOWN_TOKEN_ID)
            except Exception:
                pass

        if not up_snap or not dn_snap:
            LATEST_DEBUG["reason"] = "WAITING_FOR_BOOK"
            continue

        # Watchdog: suspend trading if all oracles dead
        if STATE.trading_suspended:
            LATEST_DEBUG["reason"] = "WS_SUSPENDED"
            continue

        # 429 Rate limit: suspend new orders during cooldown
        if is_rate_limited():
            LATEST_DEBUG["reason"] = "RATE_LIMITED"
            continue

        if np.isnan(oracle_px) or np.isnan(STATE.open_price) or np.isnan(STATE.sec_remaining):
            LATEST_DEBUG["reason"] = "MISSING_DATA"
            continue

        if STATE.strike_type not in ("OFFICIAL", "CANDLE", "RTDS"):
            LATEST_DEBUG["reason"] = "WAITING_FOR_STRIKE"
            continue

        # ── UNIFIED POSITION MONITOR: manages all exits (EV, stop, gamma, endgame) ──
        # (Replaces both EndgameManager and ExitManager)
        pass  # Position monitor evaluation is below in the survival engine

        # Preview edge
        p_preview = LATEST_DEBUG.get("p_cone", np.nan)
        if np.isfinite(p_preview):
            ua, da = up_snap.best_ask, dn_snap.best_ask
            if np.isfinite(ua) and np.isfinite(da):
                eu = p_preview - ua - fee_per_share(ua)
                ed = (1-p_preview) - da - fee_per_share(da)
                if eu >= ed:
                    LATEST_DEBUG.update({"side": "UP", "edge": float(eu)})
                else:
                    LATEST_DEBUG.update({"side": "DOWN", "edge": float(ed)})

        # ── Book staleness management ──────────────────────────────────────
        now_ms_val = ms_now()
        eff_max_age = (REST_SEED_EMPTY_COOLDOWN_MS + 5000
                       if not LAST_REST_SEED_HAD_QUOTES else POLY_BOOK_MAX_AGE_MS)
        up_age = now_ms_val - up_snap.last_update
        dn_age = now_ms_val - dn_snap.last_update
        oldest = max(up_age, dn_age)
        preseed_at = max(0, eff_max_age - POLY_BOOK_RESEED_AHEAD_MS)

        eff_cd = REST_SEED_COOLDOWN_MS if LAST_REST_SEED_HAD_QUOTES else REST_SEED_EMPTY_COOLDOWN_MS
        if oldest >= preseed_at and not REST_SEED_INFLIGHT and (now_ms_val - LAST_REST_SEED_MS >= eff_cd):
            REST_SEED_INFLIGHT = True
            LAST_REST_SEED_MS = now_ms_val
            await asyncio.get_running_loop().run_in_executor(None, _seed_book_from_rest_with_reset)

        if oldest >= eff_max_age:
            LATEST_DEBUG["reason"] = "STALE_POLY_BOOK"
            continue

        # Spread checks — also feed to tail risk guard
        up_sp = up_snap.best_ask - up_snap.best_bid
        dn_sp = dn_snap.best_ask - dn_snap.best_bid
        TAIL_RISK.feed_spread(max(up_sp, dn_sp))

        # ── Flicker guard: cache last good book, skip 0.01/0.99 ticks ─────
        # The Poly WS drops to bid=0.01/ask=0.99 between real quote updates.
        # Detect this pattern and reuse the last good snapshot.
        if not hasattr(brain_loop, "_last_good_up"):
            brain_loop._last_good_up = None
            brain_loop._last_good_dn = None

        _up_flicker = (up_snap.best_bid <= 0.02 and up_snap.best_ask >= 0.98)
        _dn_flicker = (dn_snap.best_bid <= 0.02 and dn_snap.best_ask >= 0.98)

        if not _up_flicker:
            brain_loop._last_good_up = up_snap
        elif brain_loop._last_good_up is not None:
            up_snap = brain_loop._last_good_up
            up_sp = up_snap.best_ask - up_snap.best_bid

        if not _dn_flicker:
            brain_loop._last_good_dn = dn_snap
        elif brain_loop._last_good_dn is not None:
            dn_snap = brain_loop._last_good_dn
            dn_sp = dn_snap.best_ask - dn_snap.best_bid

        # True empty book: no real quotes (defaults are bid=0.0, ask=1.0)
        up_truly_empty = (up_snap.best_bid <= 0 and up_snap.best_ask >= 1.0)
        dn_truly_empty = (dn_snap.best_bid <= 0 and dn_snap.best_ask >= 1.0)

        if up_truly_empty and dn_truly_empty:
            LATEST_DEBUG["reason"] = "EMPTY_BOOK"
            continue

        # Market already decided: one side at extreme price, no trade possible
        up_mid = (up_snap.best_bid + up_snap.best_ask) / 2
        dn_mid = (dn_snap.best_bid + dn_snap.best_ask) / 2
        if (up_mid >= 0.93 and dn_mid <= 0.07) or (dn_mid >= 0.93 and up_mid <= 0.07):
            LATEST_DEBUG["reason"] = "MARKET_DECIDED"
            continue

        if FIRST_REAL_BOOK_MS == 0 and WINDOW_OPEN_MS > 0:
            FIRST_REAL_BOOK_MS = now_ms_val

        # ── BTC feed freshness ─────────────────────────────────────────────
        # We need at least one price source reasonably fresh
        if oracle_age > BTC_FEED_MAX_AGE_MS * 5:
            LATEST_DEBUG["reason"] = "ALL_FEEDS_STALE"
            continue

        # ── Survival engine (portfolio-native) ──────────────────────────────
        pos = get_portfolio_pos()
        if pos.total > 0:
            # ── LIVE cone recomputation (never use stale LATEST_DEBUG) ──
            _surv_oracle = RTDS.price if RTDS.is_fresh(RTDS_FRESH_MS) else STATE.btc_price
            if (np.isfinite(_surv_oracle) and _surv_oracle > 0
                    and np.isfinite(STATE.open_price) and STATE.open_price > 0
                    and STATE.sec_remaining > 0 and STATE.sigma_1m > 0):
                _rp_surv = getattr(brain_loop, '_last_rp', None)
                _cal_alpha = _rp_surv.calibration_alpha if _rp_surv else 1.0
                p_surv, z_surv = cone_p_and_z(
                    _surv_oracle, STATE.open_price,
                    STATE.sec_remaining, STATE.sigma_1m,
                    calibration_alpha=_cal_alpha,
                )
            else:
                p_surv = float(LATEST_DEBUG.get("p_cone", 0.5) or 0.5)
                z_surv = float(LATEST_DEBUG.get("z", 0.0) or 0.0)
            # Update LATEST_DEBUG so dashboard always sees live model values
            LATEST_DEBUG["p_cone"] = p_surv
            LATEST_DEBUG["z"] = z_surv

            # ── UNIFIED POSITION MONITOR (replaces ExitManager + EndgameManager) ──
            # EV-driven exit decisions: prob stop, EV exit, gamma, trailing, endgame
            _u_snap = BOOK_CACHE.get(UP_TOKEN_ID)
            _d_snap = BOOK_CACHE.get(DOWN_TOKEN_ID)
            _regime_lbl = REGIME.label if REGIME else "NORMAL"

            for _pm_side, _pm_pos, _pm_tid, _pm_snap, _pm_entry in [
                ("UP",   POS_UP,   UP_TOKEN_ID,   _u_snap, pos.avg_cost_up),
                ("DOWN", POS_DOWN, DOWN_TOKEN_ID, _d_snap, pos.avg_cost_dn),
            ]:
                if _pm_pos.inventory <= 0 or _pm_snap is None or _pm_snap.best_bid <= 0.01:
                    continue

                _pm_bid = float(_pm_snap.best_bid)
                _pm_ask = float(_pm_snap.best_ask)
                _pm_spread = max(0.0, _pm_ask - _pm_bid)

                _pm_dec = POS_MONITOR.evaluate(
                    side=_pm_side,
                    entry_price=_pm_entry,
                    bid=_pm_bid,
                    p_cone=p_surv,
                    T_sec=STATE.sec_remaining,
                    sigma_1m=STATE.sigma_1m,
                    regime=_regime_lbl,
                    spread=_pm_spread,
                    now_ms=now_ms_val,
                )

                if _pm_dec.action == "SELL" and not SIMULATION_MODE and not EXEC_LOCKED:
                    # Risk-off reasons bypass the profit gate entirely —
                    # stop-losses NEED to sell even at a loss,
                    # gamma_danger is "profitable in coin-flip band, get out now"
                    _STOP_LOSS_REASONS = {"prob_stop_loss", "endgame_strong_loss",
                                          "trailing_stop", "endgame_gamma_exit",
                                          "gamma_danger"}
                    _bypass_gate = _pm_dec.reason in _STOP_LOSS_REASONS

                    if _bypass_gate:
                        _ok_sell = True
                        _net = net_sell_after_fee(_pm_bid)
                        _req = 0.0
                    else:
                        # Dynamic-alpha exit gate (profit-taking exits only)
                        _entry_T = ENTRY_T_SEC.get(_pm_side) or 230.0
                        _ok_sell, _net, _req = early_sell_profit_gate(
                            sec_remaining=STATE.sec_remaining,
                            bid=_pm_bid,
                            entry_price=_pm_entry,
                            p_cone=p_surv,
                            side=_pm_side,
                            T_entry=_entry_T,
                        )

                    if not _ok_sell:
                        # Rate-limit blocked log: 1 per 500ms per side (monotonic)
                        _block_log_key = f"_pm_block_log_{_pm_side}"
                        _last_block_log = getattr(brain_loop, _block_log_key, 0)
                        _mono_now = _mono_ms()
                        if _mono_now - _last_block_log >= 500:
                            setattr(brain_loop, _block_log_key, _mono_now)
                            _p_side = p_surv if _pm_side == "UP" else (1.0 - p_surv)
                            logger.info(
                                f"PM_EXIT_BLOCKED({_pm_side}): {_pm_dec.reason} "
                                f"T={STATE.sec_remaining:.0f}s net={_net:.3f} "
                                f"tgt={_req:.3f} p={_p_side:.3f}"
                            )
                    else:
                        logger.info(
                            f"PM_EXIT({_pm_side}): {_pm_dec.reason} "
                            f"hold_ev={_pm_dec.hold_ev:.3f} sell_ev={_pm_dec.sell_ev:.3f} "
                            f"profit={_pm_dec.profit:.3f}/sh ev_gap={_pm_dec.ev_gap:.3f} "
                            f"penalty={_pm_dec.penalty:.3f} "
                            f"entry={_pm_entry:.2f} bid={_pm_bid:.2f} regime={_regime_lbl}"
                        )
                        await eq.put({
                            "action": "ORDER", "side": _pm_side,
                            "token_id": _pm_tid,
                            "price": round(max(0.01, _pm_bid - 0.01), 2),
                            "size": float(_pm_pos.inventory),
                            "order_side": "SELL", "edge": 0.0,
                            "mode": "exit",
                            "p_cone": p_surv, "z": z_surv,
                            "sigma_1m": STATE.sigma_1m,
                            "regime": _regime_lbl,
                            "exit_reason": _pm_dec.reason,
                            "exit_hold_ev": round(_pm_dec.hold_ev, 4),
                            "exit_sell_ev": round(_pm_dec.sell_ev, 4),
                            "exit_ev_gap": round(_pm_dec.ev_gap, 4),
                        })
                        LATEST_DEBUG["reason"] = f"PM_{_pm_dec.reason}_{_pm_side}"
                        brain_loop._last_survival_ms = now_ms_val
                        break  # only one exit per tick

            # ── Flow reversal exit: contradicts dominant side ──────────────
            flow_now = FLOW.imbalance(
                up_price=_u_snap.best_bid if _u_snap else 0.5,
                down_price=_d_snap.best_bid if _d_snap else 0.5,
            )
            flow_reversal = (
                (pos.dominant_side == "UP" and flow_now < -0.70) or
                (pos.dominant_side == "DOWN" and flow_now > 0.70)
            )
            if flow_reversal and pos.total > 0:
                logger.warning(
                    f"FLOW REVERSAL: side={pos.dominant_side} but flow={flow_now:.2f} "
                    f"— triggering early exit"
                )
                LATEST_DEBUG["reason"] = "SURV_FLOW_REVERSAL"
                if not SIMULATION_MODE and not EXEC_LOCKED:
                    _fr_side = pos.dominant_side
                    _fr_tid = UP_TOKEN_ID if _fr_side == "UP" else DOWN_TOKEN_ID
                    _fr_inv = pos.up if _fr_side == "UP" else pos.dn
                    _fr_snap = _u_snap if _fr_side == "UP" else _d_snap
                    if _fr_snap and _fr_snap.best_bid > 0.01 and _fr_inv > 0:
                        await eq.put({
                            "action": "ORDER",
                            "side": _fr_side,
                            "token_id": _fr_tid,
                            "price": round(max(0.01, _fr_snap.best_bid - 0.01), 2),
                            "size": float(_fr_inv),
                            "order_side": "SELL",
                            "mode": "exit",
                            "edge": 0.0,
                            "p_cone": p_surv,
                            "z": z_surv,
                            "sigma_1m": STATE.sigma_1m,
                        })
                        brain_loop._last_survival_ms = now_ms_val
                        continue

            if not hasattr(brain_loop, "_last_survival_ms"):
                brain_loop._last_survival_ms = 0
            if now_ms_val - brain_loop._last_survival_ms < 900:
                LATEST_DEBUG["reason"] = "POSITION_OPEN"
                continue

            # Portfolio-native flip check (IADL-gated, two-leg)
            flip_ok, flip_sells = maybe_request_flip(
                ms_now(), p_surv, z_surv, STATE.sec_remaining, up_snap, dn_snap
            )
            if flip_ok and not SIMULATION_MODE:
                for op in flip_sells:
                    await eq.put({
                        "action": "ORDER",
                        "side": op["token_side"],
                        "token_id": op["token_id"],
                        "price": round(float(op["limit_price"]), 2),
                        "size": float(op["qty"]),
                        "order_side": "SELL",
                        "mode": "close",
                        "edge": 0.0,
                        "p_cone": p_surv,
                        "z": z_surv,
                        "sigma_1m": STATE.sigma_1m,
                    })
                LATEST_DEBUG["reason"] = "SURV_FLIP_SELL"
                brain_loop._last_survival_ms = now_ms_val
                continue

            sv = survival_decide_portfolio(
                p_up=p_surv,
                z=z_surv,
                sec_remaining=STATE.sec_remaining,
                up_snap=up_snap,
                dn_snap=dn_snap,
                pos=pos,
            )

            action = sv["action"]
            LATEST_DEBUG.update(sv)
            LATEST_DEBUG["reason"] = f"SURV_{action}"

            if action in ("EXIT", "TRIM", "HEDGE"):
                if not SIMULATION_MODE and sv.get("token_id") and sv.get("size", 0) > 0:
                    logger.info(f"SURVIVAL {action}: {sv['reason']} size={sv['size']}")
                    await eq.put({
                        "action": "ORDER",
                        "side": ("UP" if sv["token_id"] == UP_TOKEN_ID else "DOWN"),
                        "token_id": sv["token_id"],
                        "price": float(sv["limit"]),
                        "size": float(sv["size"]),
                        "order_side": sv["order_side"],
                        "mode": action.lower(),
                        "edge": 0.0,
                        "p_cone": p_surv,
                        "z": z_surv,
                        "sigma_1m": STATE.sigma_1m,
                    })

            brain_loop._last_survival_ms = now_ms_val
            continue

        # ── Pending portfolio-native flip BUY ────────────────────────────────
        if PENDING_FLIP_INTENT.get("active", False):
            if ms_now() > PENDING_FLIP_INTENT["expires_ms"]:
                logger.warning("PENDING_FLIP_INTENT expired — clearing.")
                PENDING_FLIP_INTENT["active"] = False
            elif abs(portfolio_net_shares()) < 1.0:
                # Net-flat: dominant leg has been sold, proceed with BUY
                if not SIMULATION_MODE:
                    await eq.put({
                        "action": "ORDER",
                        "side": PENDING_FLIP_INTENT["buy_side"],
                        "token_id": PENDING_FLIP_INTENT["buy_token_id"],
                        "price": float(PENDING_FLIP_INTENT["buy_limit"]),
                        "size": float(PENDING_FLIP_INTENT["buy_size"]),
                        "order_side": "BUY",
                        "edge": 0.0,
                        "p_cone": float(LATEST_DEBUG.get("p_cone", 0.5) or 0.5),
                        "z": float(LATEST_DEBUG.get("z", 0.0) or 0.0),
                        "sigma_1m": STATE.sigma_1m,

                    })
                logger.info(
                    f"FLIP_BUY: {PENDING_FLIP_INTENT['buy_side']} "
                    f"size={PENDING_FLIP_INTENT['buy_size']}"
                )
                PENDING_FLIP_INTENT["active"] = False

        # Legacy PENDING_FLIP backwards compat (drains naturally)
        if pos.total == 0 and PENDING_FLIP.get("active", False):
            if ms_now() > PENDING_FLIP["expires_ms"]:
                PENDING_FLIP["active"] = False
            elif (not SIMULATION_MODE and
                  PENDING_FLIP.get("token_id") and
                  PENDING_FLIP.get("size", 0) > 0):
                await eq.put({
                    "action": "ORDER", "side": PENDING_FLIP["side"],
                    "token_id": PENDING_FLIP["token_id"],
                    "price": float(PENDING_FLIP["limit"]),
                    "size": float(PENDING_FLIP["size"]),
                    "order_side": "BUY", "edge": 0.0,
                    "p_cone": LATEST_DEBUG.get("p_cone", np.nan),
                    "z": LATEST_DEBUG.get("z", np.nan),
                    "sigma_1m": STATE.sigma_1m,
                })
                PENDING_FLIP["active"] = False

        # ── Edge decision ──────────────────────────────────────────────────
        # Use RTDS price for edge if available, otherwise Coinbase
        btc_for_edge = oracle_px

        ILLIQUID = 0.50
        eff_up_bid, eff_up_ask = up_snap.best_bid, up_snap.best_ask
        eff_dn_bid, eff_dn_ask = dn_snap.best_bid, dn_snap.best_ask
        implied_up = implied_dn = False

        if up_sp > ILLIQUID and dn_sp <= ILLIQUID:
            eff_up_bid = round(1.0 - dn_snap.best_ask, 4)
            eff_up_ask = round(1.0 - dn_snap.best_bid, 4)
            implied_up = True
        elif dn_sp > ILLIQUID and up_sp <= ILLIQUID:
            eff_dn_bid = round(1.0 - up_snap.best_ask, 4)
            eff_dn_ask = round(1.0 - up_snap.best_bid, 4)
            implied_dn = True

        # ── Participation Velocity (compute before decide_edge) ──────────
        _tv_fast = float(FLOW.trade_velocity(window_s=5.0))
        _tv_slow = float(FLOW.trade_velocity(window_s=20.0))
        _pv = PV_TRACKER.update(
            ts_ms=ms_now(),
            z=float(LATEST_DEBUG.get("z", 0.0) or 0.0),
            T_sec=float(STATE.sec_remaining),
            up_bid=up_snap.best_bid, up_ask=up_snap.best_ask,
            up_bid_sz=up_snap.bid_size, up_ask_sz=up_snap.ask_size,
            dn_bid=dn_snap.best_bid, dn_ask=dn_snap.best_ask,
            dn_bid_sz=dn_snap.bid_size, dn_ask_sz=dn_snap.ask_size,
            tv_fast=_tv_fast,
            tv_slow=_tv_slow,
        )
        LATEST_DEBUG["pv"] = round(_pv["pv"], 3)
        LATEST_DEBUG["pa"] = round(_pv["pa"], 3)
        LATEST_DEBUG["qc"] = round(_pv["qc"], 3)
        LATEST_DEBUG["signal_quality"] = round(_pv["signal_quality"], 3)
        LATEST_DEBUG["pv_confirms"] = bool(_pv["participation_confirms"])
        LATEST_DEBUG["ttp_ms"] = _pv["ttp_ms"]
        LATEST_DEBUG["breakout"] = bool(_pv["breakout"])
        LATEST_DEBUG["trap"] = bool(_pv["trap"])

        # ── Coinbase Lead Indicator (hybrid oracle) ────────────────────
        _cb_lead_info = {"anticipation_active": False, "cb_z": 0.0,
                         "cb_z_velocity": 0.0, "direction_agrees": False}
        if (not np.isnan(STATE.btc_price) and STATE.btc_price > 0 and
                not np.isnan(STATE.open_price) and STATE.open_price > 0 and
                STATE.sec_remaining > 0 and STATE.sigma_1m > 0):
            _cb_z = CB_LEAD.compute_cb_z(
                STATE.btc_price, STATE.open_price,
                STATE.sec_remaining, STATE.sigma_1m,
            )
            _rtds_z = float(LATEST_DEBUG.get("z", 0.0) or 0.0)
            _cb_lead_info = CB_LEAD.update(
                cb_z=_cb_z, rtds_z=_rtds_z,
                sec_remaining=STATE.sec_remaining,
                basis_stable=BASIS.is_stable,
            )
            # Modulate Z_TRAJ confirm_ticks
            if _cb_lead_info["anticipation_active"] and _cb_lead_info["direction_agrees"]:
                Z_TRAJ.set_anticipation_mode(True, reduced_ticks=1)
            else:
                Z_TRAJ.set_anticipation_mode(False)

        LATEST_DEBUG.update({
            "cb_z": round(_cb_lead_info.get("cb_z", 0.0), 3),
            "cb_z_vel": round(_cb_lead_info.get("cb_z_velocity", 0.0), 4),
            "anticipation": bool(_cb_lead_info.get("anticipation_active", False)),
            "dir_agrees": bool(_cb_lead_info.get("direction_agrees", False)),
            "basis": round(BASIS.rolling_basis, 2),
            "basis_std": round(BASIS.basis_stdev, 2),
        })

        if _cb_lead_info.get("anticipation_active", False):
            logger.info(
                f"CB_LEAD: anticipation ACTIVE cb_z={_cb_lead_info['cb_z']:.3f} "
                f"vel={_cb_lead_info['cb_z_velocity']:.4f} "
                f"rtds_z={_rtds_z:.3f} agrees={_cb_lead_info['direction_agrees']} "
                f"basis={BASIS.rolling_basis:+.2f} basis_std={BASIS.basis_stdev:.2f}"
            )

        side, token, limit_price, size, debug = decide_edge(
            btc_price=btc_for_edge, open_price=STATE.open_price,
            sec_remaining=STATE.sec_remaining, sigma_1m=STATE.sigma_1m,
            up_bid=eff_up_bid, up_ask=eff_up_ask,
            down_bid=eff_dn_bid, down_ask=eff_dn_ask,
            lag_adaptive=LAG_ADAPTIVE, persist=PERSIST,
            regime_params=REGIME.current_params,
            flow_imbalance=FLOW.imbalance(
                up_price=eff_up_bid, down_price=eff_dn_bid),
            coverage_bias=COVERAGE.z_bias("UP"),
            coverage_bias_dn=COVERAGE.z_bias("DOWN"),
            sigma_fast=STATE.sigma_fast,
            sigma_slow=STATE.sigma_slow,
            flip_rate=float(getattr(REGIME, 'flip_rate', 0.0) or 0.0),
            pv_quality=float(_pv["signal_quality"]),
            pv_confirms=bool(_pv["participation_confirms"]),
            jump_regime=BIPOWER.jump_regime,
            oracle_age_ms=int(oracle_age),
            window_key=get_window_start_unix(),
            cb_lead_agrees=bool(_cb_lead_info.get("direction_agrees", False)),
            cb_anticipation=bool(_cb_lead_info.get("anticipation_active", False)),
            basis_shrinking=bool(BASIS.rolling_basis != 0 and abs(BASIS.rolling_basis) < abs(getattr(BASIS, '_prev_basis', BASIS.rolling_basis))),
        )

        debug["oracle_source"] = oracle_src
        if implied_up or implied_dn:
            debug["implied_side"] = "UP" if implied_up else "DN"
        LATEST_DEBUG.update(debug)

        # ── Coinbase direction agreement safety gate ──────────────────
        if (side is not None and
                _cb_lead_info.get("anticipation_active", False) and
                not _cb_lead_info.get("direction_agrees", False)):
            logger.info(
                f"CB_DIRECTION_DISAGREE: signal={side} blocked — "
                f"cb_z={_cb_lead_info['cb_z']:.3f} rtds_z={debug.get('z_ema', 0):.3f}"
            )
            debug["reason"] = "cb_direction_disagree"
            side = None
            LATEST_DEBUG.update(debug)

        # ── Flow imbalance hard gate: block adverse-selection entries ─────
        # If flow is heavily against our signal direction, we're likely
        # taking the wrong side of informed flow → block entry.
        _FLOW_ADVERSE_THRESHOLD = 0.55  # flow imbalance level to block
        if side is not None:
            _flow_c = float(debug.get("flow_centered", 0.0) or 0.0)
            _flow_adverse = (
                (side == "UP" and _flow_c < -_FLOW_ADVERSE_THRESHOLD) or
                (side == "DOWN" and _flow_c > _FLOW_ADVERSE_THRESHOLD)
            )
            if _flow_adverse:
                logger.info(
                    f"FLOW_ADVERSE_GATE: signal={side} blocked — "
                    f"flow={_flow_c:+.3f} threshold=±{_FLOW_ADVERSE_THRESHOLD}"
                )
                debug["reason"] = "flow_adverse_gate"
                side = None
                LATEST_DEBUG.update(debug)

        # ══════════════════════════════════════════════════════════════════
        # INVENTORY-AWARE DECISION LAYER (IADL)
        # ══════════════════════════════════════════════════════════════════
        if side is not None:
            _iadl_z = float(debug.get("z", 0.0) or 0.0)
            _iadl_p = float(debug.get("p_cone", 0.5) or 0.5)
            _iadl_dec = IADL.gate(
                ts_ms=ms_now(), portfolio=PORTFOLIO,
                proposed_side=side, z=_iadl_z, p_cone=_iadl_p,
                T_sec=STATE.sec_remaining,
                up_bid=eff_up_bid, up_ask=eff_up_ask,
                dn_bid=eff_dn_bid, dn_ask=eff_dn_ask,
                debug_in=debug,
            )
            debug["iadl"] = _iadl_dec.kind.value
            debug["iadl_reason"] = _iadl_dec.reason
            debug["iadl_size_mult"] = _iadl_dec.size_mult
            LATEST_DEBUG.update(debug)

            if _iadl_dec.kind == DecisionType.REJECT:
                logger.info(f"IADL_REJECT: {_iadl_dec.reason} side={side} bias={IADL.window_bias}")
                side = None

            elif _iadl_dec.kind in (DecisionType.CLOSE_THEN_REVERSE, DecisionType.UNWIND_STRADDLE):
                logger.info(
                    f"IADL_{_iadl_dec.kind.value}: {_iadl_dec.reason} "
                    f"plans={len(_iadl_dec.plan) if _iadl_dec.plan else 0}"
                )
                if _iadl_dec.plan and not SIMULATION_MODE:
                    for _op in _iadl_dec.plan:
                        _op_tok = UP_TOKEN_ID if _op.token_side == "UP" else DOWN_TOKEN_ID
                        await eq.put({
                            "action": "ORDER", "side": _op.token_side,
                            "token_id": _op_tok,
                            "price": round(float(_op.limit_price), 2),
                            "size": float(_op.qty),
                            "order_side": _op.action,
                            "edge": 0.0, "p_cone": _iadl_p,
                            "z": _iadl_z, "sigma_1m": STATE.sigma_1m,
                        })
                        logger.info(
                            f"IADL_PLAN: {_op.action} {_op.token_side} "
                            f"{_op.qty}@{_op.limit_price:.4f} reason={_op.reason}"
                        )
                side = None

            elif _iadl_dec.kind == DecisionType.FLATTEN:
                logger.info(f"IADL_FLATTEN: {_iadl_dec.reason}")
                if _iadl_dec.plan and not SIMULATION_MODE:
                    for _op in _iadl_dec.plan:
                        _op_tok = UP_TOKEN_ID if _op.token_side == "UP" else DOWN_TOKEN_ID
                        await eq.put({
                            "action": "ORDER", "side": _op.token_side,
                            "token_id": _op_tok,
                            "price": round(float(_op.limit_price), 2),
                            "size": float(_op.qty),
                            "order_side": _op.action, "edge": 0.0,
                        })
                side = None

        if side is None:
            _now_rej = time.time()
            if _now_rej - getattr(brain_loop, '_last_rej_log', 0) > 1.0:
                brain_loop._last_rej_log = _now_rej
                _z = debug.get('z')
                _zmin = debug.get('Z_MIN')
                _edge = debug.get('edge')
                _sp = debug.get('spread_max')
                _fl = debug.get('flow')
                logger.info(
                    f"TRINITY: reason={debug.get('reason')} "
                    f"in_impulse={debug.get('in_impulse')} "
                    f"imp_age={debug.get('impulse_age_ms')}ms "
                    f"lag_p50={debug.get('lag_p50_ms')}ms "
                    f"Z_MIN={f'{_zmin:.3f}' if _zmin is not None else 'N/A'} "
                    f"z={f'{_z:.3f}' if _z is not None else 'N/A'} "
                    f"edge={f'{_edge:.4f}' if _edge is not None else 'N/A'} "
                    f"spread={f'{_sp:.3f}' if _sp is not None else 'N/A'} "
                    f"flow={f'{_fl:.3f}' if _fl is not None else 'N/A'} "
                    f"T={STATE.sec_remaining:.0f}s "
                    f"regime={debug.get('regime')}"
                )

        # ── P-cone slope tracking (structural decay protection) ───────────
        _p = debug.get("p_cone")
        if _p is not None and np.isfinite(_p):
            _P_CONE_HISTORY.append(float(_p))
        _pcone_slope = 0.0
        if len(_P_CONE_HISTORY) >= 4:
            _pcone_slope = _P_CONE_HISTORY[-1] - _P_CONE_HISTORY[-4]
        debug["pcone_slope"] = round(_pcone_slope, 5)

        # ── Edge stats logger (1 Hz sampling, FIRE always logged) ─────────
        _reason = debug.get("reason", "?")
        if str(_reason).startswith("FIRE") or (now_ms_val - getattr(brain_loop, "_last_edge_log_ms", 0) >= 1000):
            brain_loop._last_edge_log_ms = now_ms_val
            try:
                import json as _json
                _row = {
                    "ts": now_ms_val,
                    "reason": _reason,
                    "edge": debug.get("edge"),
                    "z": debug.get("z"),
                    "p_cone": debug.get("p_cone"),
                    "Z_MIN": debug.get("Z_MIN"),
                    "min_ev": debug.get("min_ev"),
                    "side": debug.get("side"),
                    "spread_max": debug.get("spread_max"),
                    "cap": debug.get("cap"),
                    "flow": round(FLOW.imbalance(up_price=eff_up_bid, down_price=eff_dn_bid), 3),
                    "flow_01": debug.get("flow_01"),
                    "flow_boost": debug.get("flow_boost"),
                    "flow_confirms": debug.get("flow_confirms"),
                    "regime": debug.get("regime"),
                    "T": round(STATE.sec_remaining, 1),
                    "sigma": round(STATE.sigma_1m, 6),
                }
                with open("logs/edge_stats.jsonl", "a") as _f:
                    _f.write(_json.dumps(_row) + "\n")
            except Exception:
                pass

            # ── Per-window Z history (for historical analysis) ─────────
            try:
                import json as _json
                _strike = round(STATE.open_price, 2) if np.isfinite(STATE.open_price) else None
                # Window ID: detect new window when T jumps up
                if not hasattr(brain_loop, "_zh_window_id"):
                    brain_loop._zh_window_id = None
                    brain_loop._zh_last_T = 0
                _cur_T = STATE.sec_remaining
                if _cur_T > brain_loop._zh_last_T + 60:  # T jumped up → new window
                    brain_loop._zh_window_id = f"{_strike}_{now_ms_val}"
                brain_loop._zh_last_T = _cur_T

                _zh_row = {
                    "ts": now_ms_val,
                    "window_id": brain_loop._zh_window_id,
                    "strike": _strike,
                    "T": round(_cur_T, 1),
                    "z": debug.get("z"),
                    "z_ema": debug.get("z_ema"),
                    "edge": debug.get("edge"),
                    "edge_target": debug.get("edge_target"),
                    "p_cone": debug.get("p_cone"),
                    "spread_max": debug.get("spread_max"),
                    "side": debug.get("side"),
                    "reason": _reason,
                    "oracle_price": round(oracle_px, 2) if np.isfinite(oracle_px) else None,
                    "sigma": round(STATE.sigma_1m, 6),
                    "regime": debug.get("regime"),
                    "up_ask": round(eff_up_ask, 4),
                    "dn_ask": round(eff_dn_ask, 4),
                    "up_bid": round(eff_up_bid, 4),
                    "dn_bid": round(eff_dn_bid, 4),
                    "fired": str(_reason).startswith("FIRE"),
                }
                with open("logs/z_history.jsonl", "a") as _f:
                    _f.write(_json.dumps(_zh_row) + "\n")
            except Exception:
                pass

        # ── PnL telemetry (rate-limited → logs/pnl_telemetry.jsonl) ────────────
        try:
            TELEMETRY.maybe_emit(
                portfolio=PORTFOLIO,
                up_bid=up_snap.best_bid, up_ask=up_snap.best_ask,
                dn_bid=dn_snap.best_bid, dn_ask=dn_snap.best_ask,
                extra={
                    "z": debug.get("z"), "p_cone": debug.get("p_cone"),
                    "T": round(STATE.sec_remaining, 1),
                    "bias": IADL.window_bias,
                    "regime": REGIME.label,
                    "pos_state": {"up": POS_UP.inventory, "dn": POS_DOWN.inventory},
                    "fifo_state": {"up": POS_UP.inventory, "dn": POS_DOWN.inventory},
                },
            )
        except Exception:
            pass

        if not str(debug.get("reason", "")).startswith("FIRE"):
            # ── PAIR_BUY DISABLED — only FIRE_MISPRICING may open positions ──
            # (was: paired accumulation that bought 5-25 shares every tick)

            # ── Paired rebalance (SELL only — kept for inventory management) ─
            if (not EXEC_LOCKED and POS_UP.inventory > 0 and POS_DOWN.inventory > 0):
                total = POS_UP.inventory + POS_DOWN.inventory
                if total > 0:
                    imbal = max(POS_UP.inventory, POS_DOWN.inventory) / total
                    if imbal >= 0.65:
                        if POS_UP.inventory > POS_DOWN.inventory:
                            ex_side, ex_snap, ex_tok = "UP", up_snap, UP_TOKEN_ID
                            ex_qty = POS_UP.inventory - POS_DOWN.inventory
                        else:
                            ex_side, ex_snap, ex_tok = "DOWN", dn_snap, DOWN_TOKEN_ID
                            ex_qty = POS_DOWN.inventory - POS_UP.inventory
                        trim = max(1.0, math.floor(ex_qty * 0.5))
                        if ex_snap.best_bid > 0.01 and ex_snap.best_ask - ex_snap.best_bid < 0.50:
                            lim = round(max(0.01, ex_snap.best_bid - 0.01), 2)
                            logger.info(f"PAIR REBAL: sell {trim:.0f} {ex_side} @ {lim:.4f}")
                            await eq.put({
                                "action": "PAIR_REBALANCE", "side": ex_side,
                                "token_id": ex_tok, "price": lim, "size": trim,
                                "order_side": "SELL", "edge": 0.0,
                            })
            continue

        # ── FIRE cooldown: prevent 50+ fires/sec when execution fails ─────
        _fire_cd_ms = 500
        _fire_now = ms_now()
        if not hasattr(brain_loop, '_last_fire_ms'):
            brain_loop._last_fire_ms = 0
        if _fire_now - brain_loop._last_fire_ms < _fire_cd_ms:
            continue
        brain_loop._last_fire_ms = _fire_now

        # ── FIRE trace: log which downstream gate will apply ──────────────
        logger.info(
            f"FIRE_TRACE: side={side} T={STATE.sec_remaining:.0f}s "
            f"locked={EXEC_LOCKED} CB={CIRCUIT_BREAKER_ACTIVE} "
            f"edge={debug.get('edge', 0):.4f}"
        )

        # ── P-cone structural decay (advisory only) ───────────────────────
        if _pcone_slope < -0.10:
            logger.info(f"FIRE_WARN: pcone_slope={_pcone_slope:.4f} — structural decay")
        LATEST_DEBUG["pcone_slope"] = round(_pcone_slope, 4)

        # NOTE: Dynamic Z minimum is now handled inside decide_edge() — no double-gate

        # ── Burst score (advisory only) ────────────────────────────────────
        _spread_now = max(up_sp, dn_sp)
        _burst_sc = _spread_now * 3.0  # simplified diagnostic
        if _burst_sc > 4.0:
            logger.info(f"FIRE_WARN: burst_score={_burst_sc:.2f} — elevated")
        LATEST_DEBUG["burst_score"] = round(_burst_sc, 2)


        # ── Vol regime gate (relative to regime ref) ────────────────────────
        # No hard block — ratio-based: adjust Z requirement instead
        _sigma_ratio = REGIME.sigma_ratio
        if _sigma_ratio < 0.6:
            # Calm: allow but note for downstream Z bump
            LATEST_DEBUG["sigma_regime"] = "CALM_SIGMA"
        elif _sigma_ratio > 1.8:
            # Explosive: tighten survival, note for downstream
            LATEST_DEBUG["sigma_regime"] = "EXPLOSIVE_SIGMA"

        # Auto-release stuck lock after 5 seconds
        # Auto-release stuck lock after 5 seconds
        # ── RTDS required for entry — don't trade on fallback oracle ──────
        if not RTDS.is_fresh(RTDS_FRESH_MS):
            LATEST_DEBUG["reason"] = "NO_RTDS_NO_ENTRY"
            continue

        if EXEC_LOCKED:
            _lock_age = (ms_now() - EXEC_LOCK_MS) / 1000 if EXEC_LOCK_MS > 0 else 0
            if _lock_age > 5.0 or exec_lock_expired():
                logger.warning(f"LOCK_TIMEOUT: auto-releasing stuck lock after {_lock_age:.1f}s")
                exec_unlock()
            else:
                LATEST_DEBUG["reason"] = "ORDER_INFLIGHT"
                continue

        if CIRCUIT_BREAKER_ACTIVE:
            LATEST_DEBUG["reason"] = "CIRCUIT_BREAKER"
            logger.info("FIRE_GATE: blocked by CIRCUIT_BREAKER")
            continue

        if not side:
            logger.info(f"FIRE_GATE: blocked by side={side}")
            continue

        # ── Time window gate: first 60s and last 20s blocked ───────────────
        _T = STATE.sec_remaining
        if not np.isnan(_T):
            if _T > 240:
                LATEST_DEBUG["reason"] = "WINDOW_EARLY"
                continue
            if _T <= 20:
                LATEST_DEBUG["reason"] = "WINDOW_LATE"
                continue

        # ── Book-health gate: _execute_fok_aggressive handles its own book
        # sanity checks internally, so no pre-flight cooldown needed here.

        # ── Tail risk gate (advisory — decide_edge handles core risk) ──────
        tr = TAIL_RISK.check(
            ts_ms=now_ms_val,
            flip_rate=REGIME.flip_rate,
            sigma_eff=STATE.sigma_1m,
            spread=max(up_sp, dn_sp),
        )
        if not tr.allow_trade:
            logger.info(
                f"FIRE_WARN: tail_risk={tr.reason} — proceeding anyway "
                f"(edge={debug.get('edge', 0):.4f} z={debug.get('z', 0):.3f})"
            )
        LATEST_DEBUG["tail_risk"] = tr.reason if tr.reason else "OK"

        # Best execution route
        # Sniper route: respect oracle_engine limit_price/max_pay exactly
        if str(debug.get("reason", "")).startswith("FIRE_MISPRICING"):
            exec_token = UP_TOKEN_ID if side == "UP" else DOWN_TOKEN_ID
            exec_order_side = "BUY"
            exec_implied = False
            exec_limit = round(float(limit_price), 2)
            exec_price = exec_limit
        else:
            exec_price, exec_token, exec_order_side, exec_implied, exec_limit = \
                best_executable_price(side, up_snap, dn_snap, allow_implied=False)

        if exec_price == float("inf"):
            LATEST_DEBUG["reason"] = "BOTH_BOOKS_ILLIQUID"
            logger.info(
                f"FIRE_GATE: BOTH_BOOKS_ILLIQUID side={side} "
                f"UP={up_snap.best_bid:.2f}/{up_snap.best_ask:.2f}(sz={up_snap.ask_size}) "
                f"DN={dn_snap.best_bid:.2f}/{dn_snap.best_ask:.2f}(sz={dn_snap.ask_size}) "
                f"up_sp={up_sp:.3f} dn_sp={dn_sp:.3f}"
            )
            continue

        # ── Extreme price safety gate ─────────────────────────────────────
        # Effective token price < $0.05 or > $0.95 = near-zero EV trades
        # Use exec_price (effective price), NOT exec_limit (CLOB limit price)
        if exec_price < 0.05 or exec_price > 0.95:
            LATEST_DEBUG["reason"] = "EXTREME_PRICE"
            logger.info(
                f"FIRE_GATE: EXTREME_PRICE eff_price={exec_price:.2f} limit={exec_limit:.2f} "
                f"side={side} implied={exec_implied} p_cone={debug.get('p_cone', 0):.4f}"
            )
            exec_unlock()
            continue

        # ── FIRE passed ALL gates — order will submit ─────────────────────

        # ── Recompute effective edge for chosen execution route ─────────
        # decide_edge computes edge vs asks, but execution may route through
        # implied SELL of opposite token — recompute for actual route
        # Submit-time edge validation in execution_loop is canonical.


        # Sharpe pause gate (blocks new entries, not survival)
        if SHARPE_PAUSED:
            LATEST_DEBUG["reason"] = "SHARPE_PAUSE"
            continue

        # Day kill-switch gate (blocks new entries, not survival)
        if DAY_KILL_ACTIVE and ms_now() < DAY_KILL_UNTIL_MS:
            LATEST_DEBUG["reason"] = "DAY_KILL"
            continue
        elif DAY_KILL_ACTIVE and ms_now() >= DAY_KILL_UNTIL_MS:
            DAY_KILL_ACTIVE = False
            logger.info("DAY_KILL lifted.")

        logger.info(
            f"FIRE_CLEAR: side={side} T={STATE.sec_remaining:.0f}s "
            f"edge={debug.get('edge', 0):.4f} price={exec_limit:.4f} "
            f"bankroll=${WINDOW_BANKROLL:.2f} tail_risk={tr.reason or 'OK'}"
        )

        limit_price = exec_limit

        # ── Kelly entry sizing (with regime + tail risk multipliers) ───────
        edge_val = debug.get("edge", np.nan)
        p_fire   = debug.get("p_cone", np.nan)
        z_fire   = debug.get("z", np.nan)

        p_win = p_fire if side == "UP" else (1.0 - p_fire)
        var_e = p_win * (1.0 - p_win) + 1e-6
        f_kelly = edge_val / var_e if var_e > 0 else 0.0

        # Adaptive base: 0.50 (user-calibrated for $438 bankroll)
        _kelly_base = 0.50 * SHARPE_MULT
        if debug.get("flow_confirms"):
            _kelly_base *= 1.2
        if _sigma_ratio < 0.6:
            _kelly_base *= 0.8

        _kelly_raw = _kelly_base * f_kelly
        debug["kelly_pre_filters"] = round(_kelly_raw, 5)

        # ═══════════════════════════════════════════════════════════════
        # HIERARCHICAL KELLY GATING (4 layers, not multiplicative chaos)
        # Each layer has a floor to prevent collapse.
        # ═══════════════════════════════════════════════════════════════

        # ── Layer 1: STRUCTURAL REGIME DECISION ───────────────────────
        # Event/jump intensity factors
        _vol_intensity = REGIME.vol_detector.intensity
        _event_scale = 1.0 + min(1.5, _vol_intensity) if REGIME.vol_detector.event_active else 1.0
        _jump_scale = 1.0 + min(2.0, JUMP_DETECTOR.jump_intensity / 5.0) if JUMP_DETECTOR.lambda_hat > 0.02 else 1.0
        # Take worst-of regime multipliers (not all multiplied together)
        _regime_mult = min(
            REGIME.current_params.kelly_multiplier,
            BIPOWER.kelly_multiplier,
            MOMENTUM.kelly_multiplier,
        )
        _regime_mult = max(0.20, _regime_mult)  # floor: never below 20%
        # Event/jump scaling: take the larger effect, not multiply both
        _event_jump = max(_event_scale, _jump_scale) if _event_scale > 1.0 or _jump_scale > 1.0 else 1.0
        _regime_final = _regime_mult * min(2.0, _event_jump)

        _kelly_L1 = _kelly_raw * _regime_final
        debug["kelly_L1_regime"] = round(_kelly_L1, 5)

        # ── Layer 2: EXECUTION QUALITY ────────────────────────────────
        # Fill probability + spread tier (execution realism)
        _spread_for_fill = max(up_sp, dn_sp)
        _depth_for_fill = min(up_snap.bid_size + up_snap.ask_size, dn_snap.bid_size + dn_snap.ask_size)
        _ofi_agrees = bool(debug.get("flow_confirms", False))
        _tv_for_fill = float(FLOW.trade_velocity(window_s=5.0))
        _fp = FILL_PROB.estimate(_spread_for_fill, _depth_for_fill, _ofi_agrees, _tv_for_fill)
        _fill_kelly = max(0.30, _fp["fill_kelly"])  # floor: 30%
        debug["p_fill"] = _fp["p_fill"]
        debug["slippage"] = _fp["slippage"]
        debug["fill_kelly"] = _fill_kelly

        _late_kelly = debug.get("late_kelly_mult", 1.0)
        _spread_mult = debug.get("spread_size_mult", 1.0)
        _exec_mult = _fill_kelly * min(_spread_mult, _late_kelly)
        _exec_mult = max(0.25, _exec_mult)  # floor

        _kelly_L2 = _kelly_L1 * _exec_mult
        debug["kelly_L2_exec"] = round(_kelly_L2, 5)

        # ── Layer 3: RISK OVERLAY ─────────────────────────────────────
        # Take worst-of SPRT or loss_streak (NOT both — avoids double-punishment)
        _risk_sprt = SPRT.kelly_multiplier
        _risk_loss = LOSS_STREAK_KELLY_PENALTY
        _risk_mult = min(_risk_sprt, _risk_loss)  # worst-of, not product
        _risk_mult = max(0.25, _risk_mult)  # floor
        # Tail risk cap applies independently
        _risk_final = _risk_mult * tr.kelly_cap

        _kelly_L3 = _kelly_L2 * _risk_final
        debug["kelly_L3_risk"] = round(_kelly_L3, 5)

        # ── Layer 4: EXPIRY MODULATOR (last) ──────────────────────────
        _expiry_kelly = EXPIRY.kelly_mult(STATE.sec_remaining)
        debug.update(EXPIRY.status_dict(STATE.sec_remaining))

        _kelly_L4 = _kelly_L3 * _expiry_kelly

        # Final clamp with GLOBAL FLOOR: never below 3% of raw Kelly
        f_used = max(0.0, min(1.0, _kelly_L4))
        if _kelly_raw > 0:
            f_used = max(f_used, 0.03 * _kelly_raw)  # elasticity floor
        f_used = min(1.0, f_used)

        debug["kelly_post_filters"] = round(f_used, 5)
        debug["kelly_compression"] = round(f_used / max(1e-9, _kelly_raw), 3)

        # Liquidity tiering: adapt size to spread width (wide ≠ empty)
        spread_for_haircut = max(up_sp, dn_sp)
        if spread_for_haircut > 0.80:
            spread_haircut = 0.25   # extreme: maker-only territory
        elif spread_for_haircut > 0.40:
            spread_haircut = 0.50   # wide: reduced size
        elif spread_for_haircut > 0.20:
            spread_haircut = 0.70   # moderate: slight haircut
        else:
            spread_haircut = 1.0    # healthy: full size

        # Flow size scaling: contra flow → smaller, not blocked
        _flow_sz_scale = debug.get("flow_size_scale", 1.0)
        # Z magnitude scaling: stronger edge signal → slightly larger
        _z_scale = max(0.5, min(1.5, abs(debug.get("z", 0.0)) / 2.0))
        # Sigma ratio scaling: impulse vol → up to 1.5× size
        _sigma_ratio = STATE.sigma_fast / STATE.sigma_slow if STATE.sigma_slow > 1e-10 else 1.0
        _sigma_scale = 1.0 + 0.5 * min(1.0, max(0.0, _sigma_ratio - 1.0))

        max_shares  = math.floor(MAX_SPEND_PER_ORDER_USD / limit_price)
        min_shares  = max(1, math.ceil(1.0 / limit_price))  # CLOB minimum $1
        capped_size = max(min_shares, math.floor(
            max_shares * f_used * spread_haircut * _flow_sz_scale * _z_scale * _sigma_scale
        ))

        # Bankroll-based risk cap: use window snapshot (avoid REST in hot path)
        bankroll = max(0.0, float(WINDOW_BANKROLL))
        if bankroll > 0:
            capped_size = min(capped_size, math.floor(TAIL_RISK.clamp_size(
                capped_size * limit_price, bankroll) / limit_price))
            capped_size = max(min_shares, capped_size)

        # ── Regime-aware + drawdown-aware risk budget ──────────────────────
        collateral = bankroll
        liq = portfolio_liquidation_value(up_snap.best_bid, dn_snap.best_bid)

        a_mult, b_mult, why = regime_risk_multipliers()
        equity = float(collateral + liq + SESSION_PNL)
        dd_mult = drawdown_risk_multiplier(equity)
        day_pnl, day_dd = update_day_kill_switch(equity)
        logger.info(f"DAY_RISK | pnl={day_pnl:+.2f} dd={day_dd*100:.1f}% kill={DAY_KILL_ACTIVE}")

        a = max(RISK_A_MIN, min(RISK_A_MAX, RISK_A_BASE * a_mult * dd_mult))
        b = max(RISK_B_MIN, min(RISK_B_MAX, RISK_B_BASE * b_mult * dd_mult))
        RISK_BUDGET = a * collateral + b * liq

        logger.info(
            f"RISK_MULT | a={a:.3f} b={b:.3f} "
            f"regime=({why}) dd={DRAWDOWN*100:.1f}% dd_mult={dd_mult:.2f} "
            f"equity={equity:.2f} peak={EQUITY_PEAK:.2f}"
        )

        gross_before = liq
        net_before = portfolio_net_shares()
        new_spend = capped_size * limit_price
        gross_after = gross_before + new_spend
        clipped = False

        if gross_after > RISK_BUDGET:
            allowed = max(0.0, RISK_BUDGET - gross_before)
            capped_size = int(max(min_shares, math.floor(allowed / max(0.01, limit_price))))
            clipped = True

        # Gross / Net hard caps
        MAX_GROSS_SHARES = int(os.getenv("MAX_GROSS_SHARES", "90"))
        MAX_NET_SHARES   = int(os.getenv("MAX_NET_SHARES", "60"))

        gross_shares_after = portfolio_gross_shares() + capped_size
        net_shares_after = abs(portfolio_net_shares() + (capped_size if side == "UP" else -capped_size))

        if gross_shares_after > MAX_GROSS_SHARES:
            capped_size = max(min_shares, int(MAX_GROSS_SHARES - portfolio_gross_shares()))
            clipped = True

        if net_shares_after > MAX_NET_SHARES:
            cur_net = portfolio_net_shares()
            target_add = MAX_NET_SHARES - abs(cur_net)
            capped_size = max(min_shares, int(min(capped_size, target_add)))
            clipped = True

        new_spend = capped_size * limit_price
        gross_after = gross_before + new_spend

        # 🔥 Risk telemetry output
        log_risk_state(
            collateral=collateral,
            liq=liq,
            risk_budget=RISK_BUDGET,
            gross_before=gross_before,
            net_before=net_before,
            new_spend=new_spend,
            gross_after=gross_after,
            capped_size=capped_size,
            limit_price=limit_price,
            side=side,
        )
        if clipped:
            logger.info("RISK_CLIP_APPLIED")
        logger.info(
            f"RISK_WEIGHTS | a={a:.3f} b={b:.3f} "
            f"dd={DRAWDOWN*100:.1f}% dd_mult={dd_mult:.2f} "
            f"flip={getattr(REGIME,'flip_rate',0.0):.3f} regime={getattr(REGIME,'label','?')}"
        )

        actual_spend = capped_size * limit_price

        logger.info(
            f"SIZE: {capped_size} @ {limit_price:.4f} = ${actual_spend:.2f} "
            f"(Kelly={f_used:.2f}, regime={REGIME.label}, haircut={spread_haircut:.2f}, "
            f"flow_scale={_flow_sz_scale:.2f}, z_scale={_z_scale:.2f}, "
            f"sigma_scale={_sigma_scale:.2f}(r={_sigma_ratio:.2f}), "
            f"oracle={oracle_src})"
        )

        token_id_pos = UP_TOKEN_ID if side == "UP" else DOWN_TOKEN_ID

        # ── Hard no-trade zone: reject very low-|z| signals near p=0.50 ──
        # Only blocks extreme noise — edge validation already done upstream
        _T = STATE.sec_remaining
        if _T > 120:
            _z_floor = 0.40   # early: loose (sniper already validated)
        elif _T > 60:
            _z_floor = 0.60   # mid: moderate
        else:
            _z_floor = 0.85   # late: tighter
        if abs(z_fire) < _z_floor:
            LATEST_DEBUG["reason"] = f"NO_TRADE_LOW_Z(|z|={abs(z_fire):.2f}<{_z_floor})"
            logger.info(f"FIRE_BLOCK: NO_TRADE_LOW_Z |z|={abs(z_fire):.2f} < {_z_floor}")
            continue

        # ── Per-window trade cap: most edge in 1-2 bursts ────────────
        if TRADES_THIS_WINDOW >= MAX_TRADES_PER_WINDOW:
            LATEST_DEBUG["reason"] = f"WINDOW_TRADE_CAP({TRADES_THIS_WINDOW}/{MAX_TRADES_PER_WINDOW})"
            continue

        # Submit-time edge validation in execution_loop is canonical.

        # NOTE: inventory updates happen in execution_loop on confirmed fill
        LAST_FIRE_SNAPSHOT = dict(debug)
        # Coverage recording moved to execution_loop (on confirmed fill only)

        logger.info(
            f"SIGNAL FIRED: {side} @ {limit_price:.4f} size={capped_size} "
            f"(${actual_spend:.2f}) edge={edge_val:.4f} p={p_fire:.4f} "
            f"z={z_fire:.4f} sig={STATE.sigma_1m:.5f} oracle={oracle_src} "
            f"regime={REGIME.label}"
        )

        if SIMULATION_MODE:
            logger.info("SIM MODE: order suppressed.")
            continue

        # ── FIRE debounce: prevent spam on same token+side ───────────
        _mode = debug.get("mode", "regular")
        if not fire_allowed(exec_token, exec_order_side, _mode):
            LATEST_DEBUG["reason"] = "FIRE_DEBOUNCE"
            continue

        # ── PV trap block: refuse thin-liquidity fake breakouts ──────
        if _pv["trap"] and _pv["spread_norm"] > 1.4:
            LATEST_DEBUG["reason"] = "PV_TRAP_BLOCK"
            continue

        # ── Bipower jump regime: suppress mean-reversion when diffusion broken
        _fire_mode = debug.get("mode", "impulse")
        if BIPOWER.jump_regime and _fire_mode not in ("drift", "mispricing"):
            # In jump regime, only allow drift + mispricing (sniper is its own signal).
            # Impulse MR trades are suppressed — diffusion assumption broken.
            LATEST_DEBUG["reason"] = "BIPOWER_MR_SUPPRESS"
            LATEST_DEBUG["j_ema"] = round(BIPOWER.j_ema, 4)
            continue

        # ── VWAP overlay: skip MR if price inside fair value
        if _fire_mode not in ("drift", "mispricing") and not VWAP.mr_allowed():
            LATEST_DEBUG["reason"] = "VWAP_INSIDE_FAIR"
            LATEST_DEBUG["z_vwap"] = round(VWAP.z_score, 3)
            continue

        # ── Diagnostic: every gate state at submission time ────────────
        logger.critical(
            f"FIRE_TRIGGERED: side={side} T={STATE.sec_remaining:.0f}s "
            f"price={limit_price:.4f} size={capped_size} edge={edge_val:.4f} "
            f"p={p_fire:.4f} z={z_fire:.4f} | "
            f"exec_locked={EXEC_LOCKED} "
            f"sim={SIMULATION_MODE} regime={REGIME.label} | "
            f"book: up={up_snap.best_bid:.2f}/{up_snap.best_ask:.2f} "
            f"dn={dn_snap.best_bid:.2f}/{dn_snap.best_ask:.2f}"
        )

        _spread_tier = debug.get("spread_tier", "taker_allowed")
        _mode = debug.get("mode", "regular")

        # Compute flow at submission time for attribution logging
        _submit_flow = FLOW.imbalance(
            up_price=eff_up_bid, down_price=eff_dn_bid)

        # Capture signal-time book for exec_token (flicker-guarded)
        _signal_snap = up_snap if exec_token == UP_TOKEN_ID else dn_snap
        await eq.put({
            "action": "ORDER", "side": side, "token_id": exec_token,
            "price": limit_price, "size": capped_size,
            "order_side": exec_order_side, "edge": edge_val,
            "edge_target": float(debug.get("edge_target", 0.0) or 0.0),
            "p_cone": p_fire, "z": z_fire, "sigma_1m": STATE.sigma_1m,
            "spread_tier": _spread_tier,
            "mode": _mode,
            "regime": REGIME.label,
            "flow": round(_submit_flow, 3),
            "signal_type": debug.get("reason", ""),
            "signal_ask": round(float(_signal_snap.best_ask), 4),
            "signal_bid": round(float(_signal_snap.best_bid), 4),
        })
        logger.info(f"FIRE_SUBMIT: queued {side} {exec_order_side} {capped_size}@{limit_price:.4f} FOK_AGGRESSIVE spread={_spread_tier}")
        confirm_sniper_fire()  # cooldown starts only after order actually queued


# ════════════════════════════════════════════════════════════════════════════
# RTDS CALLBACK (feeds sigma + strike capture)
# ════════════════════════════════════════════════════════════════════════════

def _on_rtds_price(price: float, ts_ms: int) -> None:
    """Called by chainlink_rtds_task on every price update."""
    # Feed into strike capture
    STRIKE_CAPTURE.feed(price, ts_ms)
    # Feed tail risk oracle guard
    TAIL_RISK.feed_oracle(price, ts_ms)
    # Always feed RTDS into sigma — it's the authoritative source
    _update_sigma(price, ts_ms)
    # Feed basis tracker from RTDS side
    if STATE.btc_ts_ms > 0 and not np.isnan(STATE.btc_price):
        BASIS.update(price, STATE.btc_price)


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

async def main() -> None:
    global MARKET_ID, UP_TOKEN_ID, DOWN_TOKEN_ID, WINDOW_OPEN_MS, FIRST_REAL_BOOK_MS

    logger.info("=" * 60)
    logger.info("PolyBot v2 starting — Chainlink RTDS oracle mode")
    logger.info("=" * 60)

    loop = asyncio.get_running_loop()
    info = None
    for _disc_attempt in range(60):  # retry for up to ~10 minutes
        info = await loop.run_in_executor(None, _auto_discover_market_sync)
        if info:
            break
        _wait = 10
        logger.warning(f"No active 5-min market found. Retrying in {_wait}s… (attempt {_disc_attempt+1}/60)")
        await asyncio.sleep(_wait)
    if not info:
        logger.error("No active 5-min market found after 60 retries. Exiting.")
        return

    MARKET_ID     = info["market_id"]
    UP_TOKEN_ID   = info["up_token"]
    DOWN_TOKEN_ID = info["down_token"]
    WINDOW_OPEN_MS = ms_now()
    FIRST_REAL_BOOK_MS = 0
    # Ensure POLY_STATE entries exist for new token IDs
    for _tid in (UP_TOKEN_ID, DOWN_TOKEN_ID):
        if _tid not in POLY_STATE:
            POLY_STATE[_tid] = {
                "bid": 0.0, "ask": 1.0, "bid_size": 0.0, "ask_size": 0.0,
                "last_update": 0, "source": "init"
            }

    # Set token IDs on flow tracker and Goldsky analytics
    FLOW.set_tokens(UP_TOKEN_ID, DOWN_TOKEN_ID)
    GOLDSKY.set_market(MARKET_ID, UP_TOKEN_ID, DOWN_TOKEN_ID)

    await loop.run_in_executor(None, _seed_book_from_rest)
    logger.info("Book seeded from REST.")

    # Initialize WINDOW_BANKROLL at startup (prevents $0 bankroll blocking all trades)
    global WINDOW_BANKROLL
    _avail = await loop.run_in_executor(None, get_available_usdc_balance)
    WINDOW_BANKROLL = min(float(_avail), WINDOW_SPEND_CAP_USD)
    logger.info(
        f"WINDOW_BANKROLL initialized: ${WINDOW_BANKROLL:.2f} "
        f"(window_cap=${WINDOW_SPEND_CAP_USD:.2f}, per_order_cap=${MAX_SPEND_PER_ORDER_USD:.2f})"
    )

    await loop.run_in_executor(None, _prime_tick_size_cache, UP_TOKEN_ID)
    await loop.run_in_executor(None, _prime_tick_size_cache, DOWN_TOKEN_ID)
    logger.info("Tick-size cache primed.")

    # Approve USDC + token allowances for trading
    try:
        if client is not None:
            client.update_balance_allowance(
                params=BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
            )
            logger.info("USDC allowance approved.")
            for _tid in (UP_TOKEN_ID, DOWN_TOKEN_ID):
                client.update_balance_allowance(
                    params=BalanceAllowanceParams(
                        asset_type=AssetType.CONDITIONAL, token_id=_tid
                    )
                )
            logger.info("Token allowances approved.")
    except Exception as e:
        logger.warning(f"Allowance approval failed: {e} — may need manual approval")

    POLY_WS = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

    await asyncio.gather(
        # Oracle feeds (priority order)
        chainlink_rtds_task(on_price=_on_rtds_price),    # PRIMARY — resolution feed
        coinbase_trade_task(),                              # sigma + fallback

        chainlink_onchain_task(),                           # emergency on-chain

        # Polymarket book
        polymarket_book_task(POLY_WS),

        # Execution
        execution_loop(),

        # Market management
        market_clock_task(info["end_time"]),

        # Brain
        brain_loop(execution_queue),

        # Analytics
        GOLDSKY.poll_task(),                                # on-chain fill analytics

        # Monitoring
        watchdog_task(),                                    # WS health monitor

        # Dashboard
        dashboard_task(),
    )


async def watchdog_task():
    """Monitor WS feeds. Suspend trading if all oracles go silent > 10 seconds."""
    while True:
        await asyncio.sleep(1)
        now = time.time()
        rtds_age = RTDS.age_ms() / 1000.0
        cb_age_s = (ms_now() - STATE.btc_ts_ms) / 1000.0 if STATE.btc_ts_ms > 0 else 999.0

        all_stale = rtds_age > 10.0 and cb_age_s > 10.0
        if all_stale:
            if not STATE.trading_suspended:
                logger.error(f"WATCHDOG: All oracles stale (RTDS={rtds_age:.1f}s CB={cb_age_s:.1f}s) — SUSPENDING")
                STATE.trading_suspended = True
        elif STATE.trading_suspended:
            logger.info("WATCHDOG: Oracle recovered — resuming trading")
            STATE.trading_suspended = False


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
