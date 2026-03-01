"""
oracle_engine.py — Core edge-decision logic for Polymarket 5-min BTC markets.

Inputs:  BTC price (from Chainlink RTDS), strike, time remaining, Poly book.
Outputs: (side, token, limit_price, size, debug) per tick.

Key components:
  - Lag-adaptive Z gate (widens/tightens with measured Binance→Poly reaction lag)
  - Cone probability model (log-normal p/z for BTC finishing above/below strike)
  - Z shrinkage + σ floor + probability calibration (CALIBRATION.md)
  - Edge-adaptive ask cap ladder + persistence
  - Fee-conservative taker model
  - Regime-aware dynamic min_ev (REGIME.md)
"""

import time
import math
import os
from math import erfc
import logging
import numpy as np
from dataclasses import dataclass, field
from collections import deque
from typing import Optional, Tuple

from regime import RegimeClassifier, RegimeParams, dynamic_min_ev, NORMAL


# ─────────────────────────────────────────────────────────────────────────────
# Volatility-Adaptive Z Threshold
# ─────────────────────────────────────────────────────────────────────────────

_SIGMA_HISTORY: deque = deque(maxlen=3600)  # ~1h at 1Hz


def update_sigma_history(sigma_eff: float) -> None:
    """Feed effective sigma into rolling history for percentile ranking."""
    if sigma_eff > 0:
        _SIGMA_HISTORY.append(sigma_eff)


def percentile_rank(value: float, data: deque) -> float:
    """Percentile rank of *value* within *data*. Returns 0.5 when empty."""
    if not data:
        return 0.5
    sorted_data = sorted(data)
    count = sum(1 for x in sorted_data if x <= value)
    return count / len(sorted_data)


def adaptive_z_threshold(
    *,
    T_sec: float,
    sigma_fast: float,
    sigma_slow: float,
    sigma_eff: float,
    flip_rate: float,
    spread: float,
    book_healthy: bool,
) -> float:
    """
    Fully volatility-adaptive Z threshold.

    Returns dynamic Z_min based on volatility regime, flip pressure,
    book quality, and time-to-expiry.  No manual thresholds required.
    """
    # --- constants (stable structural params) ---
    Z_base = 0.95
    alpha = 0.45    # calm penalty
    beta = 0.55     # flip penalty
    gamma = 0.25    # no-breakout penalty

    eta = 0.35      # late-window gamma multiplier strength
    p_exp = 1.2
    T_ref = 60.0
    T_min = 15.0

    # --- volatility percentile ---
    update_sigma_history(sigma_eff)
    P_sigma = percentile_rank(sigma_eff, _SIGMA_HISTORY)

    # --- volatility ratio ---
    R_sigma = sigma_fast / sigma_slow if sigma_slow > 1e-12 else 1.0

    # --- normalized flip pressure ---
    F0 = 0.10
    F_n = min(1.0, flip_rate / F0)

    # --- book quality ---
    if not book_healthy or spread >= 0.50:
        Q_b = 0.01  # effectively disables trading
    elif spread < 0.03:
        Q_b = 1.0
    else:
        Q_b = max(0.2, 1.0 - spread * 10)

    # --- base adaptive threshold ---
    Z_min = (
        Z_base
        + alpha * (1.0 - P_sigma)
        + beta * F_n
        + gamma * max(0.0, 1.0 - R_sigma)
    )

    # --- smooth late-window escalation ---
    T_adj = max(T_sec, T_min)
    M_T = 1.0 + eta * (T_ref / T_adj) ** p_exp

    # --- apply time and book adjustments ---
    Z_min = Z_min * M_T * (1.0 / math.sqrt(Q_b))

    # --- soft floor / ceiling to prevent runaway extremes ---
    Z_min = max(0.85, min(2.5, Z_min))

    return Z_min


# ─────────────────────────────────────────────────────────────────────────────
# Z Trajectory Filter (EMA + slope)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ZTrajConfig:
    ema_alpha: float = 0.25         # smoothing for Z
    slope_lookback: int = 5         # seconds/ticks (1Hz)
    slope_eps: float = 0.20         # normalized threshold (not absolute z units)
    min_abs_z: float = 0.90         # minimum |Z| before we even care
    max_flip_rate: float = 0.08     # above this, treat as chop unless edge huge
    confirm_ticks: int = 3          # require N consecutive expanding ticks


class ZTrajectoryFilter:
    def __init__(self, cfg: ZTrajConfig = None):
        self.cfg = cfg or ZTrajConfig()
        self.z_ema = 0.0
        self._init = False
        self.hist: deque = deque(maxlen=60)
        self._expands_streak = 0
        self._effective_confirm_ticks = self.cfg.confirm_ticks

    def set_anticipation_mode(self, active: bool, reduced_ticks: int = 1) -> None:
        """Dynamically adjust confirm_ticks for Coinbase lead anticipation."""
        if active:
            self._effective_confirm_ticks = max(1, reduced_ticks)
        else:
            self._effective_confirm_ticks = self.cfg.confirm_ticks

    def apply_time_relaxation(self, T_sec: float) -> None:
        """Late-window persistence relaxation: momentum fires faster near expiry.
        T > 120s: no change (use normal confirm_ticks or anticipation ticks)
        T  60-120s: reduce by 1 tick
        T  30-60s: reduce to max(1, base-2)
        T < 30s: minimum 1 tick (fire ASAP)
        """
        base = self._effective_confirm_ticks
        if T_sec <= 30:
            self._effective_confirm_ticks = 1
        elif T_sec <= 60:
            self._effective_confirm_ticks = max(1, base - 2)
        elif T_sec <= 120:
            self._effective_confirm_ticks = max(1, base - 1)

    def update(self, z: float) -> dict:
        z = float(z)
        if not self._init:
            self.z_ema = z
            self._init = True
        else:
            a = self.cfg.ema_alpha
            self.z_ema = (1 - a) * self.z_ema + a * z

        self.hist.append(self.z_ema)

        lb = self.cfg.slope_lookback
        if len(self.hist) <= lb:
            dz = 0.0
        else:
            dz = self.hist[-1] - self.hist[-1 - lb]

        # ── Normalize slope by rolling z_ema stdev (scale-free) ──
        vals = list(self.hist)
        if len(vals) >= 10:
            mean = sum(vals) / len(vals)
            var = sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)
            z_std = math.sqrt(max(1e-9, var))
        else:
            z_std = 1.0
        dz_norm = dz / z_std

        direction = 1 if self.z_ema > 0 else -1
        expands = (dz_norm * direction) > self.cfg.slope_eps

        # ── 2-tick expansion confirmation ──
        if expands:
            self._expands_streak = min(10, self._expands_streak + 1)
        else:
            self._expands_streak = 0
        confirmed = self._expands_streak >= self._effective_confirm_ticks

        return {
            "z_ema": self.z_ema,
            "dz": dz,
            "z_std": z_std,
            "dz_norm": dz_norm,
            "expands": expands,
            "confirmed": confirmed,
            "direction": direction,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Basis Tracker (RTDS - Coinbase price delta)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BasisConfig:
    window_size: int = 120           # rolling window in samples (~2 min at 1Hz)
    stable_threshold: float = 15.0   # max stdev in $ to consider basis "stable"
    ema_alpha: float = 0.10          # smoothing for rolling basis mean


class BasisTracker:
    """Tracks rolling RTDS - Coinbase price delta for basis correction."""

    def __init__(self, cfg: BasisConfig = None):
        self.cfg = cfg or BasisConfig()
        self.deltas: deque = deque(maxlen=self.cfg.window_size)
        self.basis_ema: float = 0.0
        self._init: bool = False

    def update(self, rtds_price: float, cb_price: float) -> None:
        """Feed a paired price observation."""
        delta = rtds_price - cb_price
        self.deltas.append(delta)
        if not self._init:
            self.basis_ema = delta
            self._init = True
        else:
            a = self.cfg.ema_alpha
            self.basis_ema = (1 - a) * self.basis_ema + a * delta

    @property
    def rolling_basis(self) -> float:
        """Current EMA of RTDS - Coinbase delta."""
        return self.basis_ema

    @property
    def basis_stdev(self) -> float:
        """Standard deviation of recent basis deltas."""
        if len(self.deltas) < 10:
            return 999.0
        arr = list(self.deltas)
        mean = sum(arr) / len(arr)
        var = sum((x - mean) ** 2 for x in arr) / (len(arr) - 1)
        return math.sqrt(max(0, var))

    @property
    def is_stable(self) -> bool:
        """True if basis is stable enough to trust Coinbase lead."""
        return self.basis_stdev < self.cfg.stable_threshold

    def corrected_price(self, cb_price: float) -> float:
        """Approximate what RTDS would show given a Coinbase price."""
        return cb_price + self.basis_ema


BASIS = BasisTracker(BasisConfig())


# ─────────────────────────────────────────────────────────────────────────────
# Coinbase Lead Indicator (Hybrid Oracle)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CoinbaseLeadConfig:
    z_threshold: float = 0.50         # min |cb_z| to trigger anticipation
    move_threshold: float = 0.10      # min |delta_cb_z| per tick for fast move
    lookback_ticks: int = 3           # ticks to measure velocity over
    anticipation_confirm: int = 1     # reduced confirm_ticks in anticipation mode
    endgame_cutoff_sec: float = 30.0  # disable Coinbase lead below this T
    basis_stable_required: bool = True # require stable basis to trust lead


class CoinbaseLeadIndicator:
    """
    Parallel Z-score from Coinbase to detect fast moves before RTDS catches up.
    Does NOT replace RTDS for edge/p_token calculations.
    Only used to modulate Z_TRAJ confirm_ticks.
    """

    def __init__(self, cfg: CoinbaseLeadConfig = None):
        self.cfg = cfg or CoinbaseLeadConfig()
        self.cb_z_hist: deque = deque(maxlen=30)
        self.anticipation_active: bool = False
        self.anticipated_direction: int = 0
        self._last_cb_z: float = 0.0

    def compute_cb_z(
        self, cb_price: float, strike: float,
        sec_remaining: float, sigma_1m: float,
    ) -> float:
        """Compute Z-score from Coinbase price (same formula as cone_p_and_z)."""
        if sec_remaining <= 0 or sigma_1m <= 0 or strike <= 0 or cb_price <= 0:
            return 0.0
        sigma = max(sigma_1m, SIGMA_FLOOR)
        t_frac = sec_remaining / 60.0
        vol = sigma * math.sqrt(t_frac)
        if vol < 1e-12:
            return 0.0
        z_raw = math.log(cb_price / strike) / vol
        z_adj = z_raw / (1.0 + Z_SHRINK_LAMBDA * abs(z_raw))
        return z_adj

    def update(
        self, cb_z: float, rtds_z: float,
        sec_remaining: float, basis_stable: bool,
    ) -> dict:
        """
        Update state and determine if anticipation mode should be active.

        Returns dict with anticipation state, direction agreement, velocities.
        """
        self.cb_z_hist.append(cb_z)

        # Compute velocity of cb_z
        lb = min(self.cfg.lookback_ticks, len(self.cb_z_hist) - 1)
        if lb > 0:
            cb_z_velocity = cb_z - self.cb_z_hist[-1 - lb]
        else:
            cb_z_velocity = 0.0

        # Direction agreement: both feeds point same way
        if abs(rtds_z) > 0.1:
            direction_agrees = (cb_z > 0 and rtds_z > 0) or (cb_z < 0 and rtds_z < 0)
        else:
            direction_agrees = False

        self.anticipation_active = False
        self.anticipated_direction = 0

        # Disable in endgame
        if sec_remaining < self.cfg.endgame_cutoff_sec:
            self._last_cb_z = cb_z
            return {
                "anticipation_active": False,
                "anticipated_direction": 0,
                "cb_z": cb_z,
                "cb_z_velocity": cb_z_velocity,
                "direction_agrees": direction_agrees,
                "reason": "endgame_rtds_only",
            }

        # Require stable basis
        if self.cfg.basis_stable_required and not basis_stable:
            self._last_cb_z = cb_z
            return {
                "anticipation_active": False,
                "anticipated_direction": 0,
                "cb_z": cb_z,
                "cb_z_velocity": cb_z_velocity,
                "direction_agrees": direction_agrees,
                "reason": "basis_unstable",
            }

        # Check thresholds
        strong_z = abs(cb_z) >= self.cfg.z_threshold
        fast_move = abs(cb_z_velocity) >= self.cfg.move_threshold

        if strong_z and fast_move:
            self.anticipation_active = True
            self.anticipated_direction = 1 if cb_z > 0 else -1

        self._last_cb_z = cb_z

        return {
            "anticipation_active": self.anticipation_active,
            "anticipated_direction": self.anticipated_direction,
            "cb_z": cb_z,
            "cb_z_velocity": cb_z_velocity,
            "direction_agrees": direction_agrees,
            "reason": "anticipation" if self.anticipation_active else "normal",
        }

    def reset(self):
        """Reset for new window."""
        self.cb_z_hist.clear()
        self.anticipation_active = False
        self.anticipated_direction = 0
        self._last_cb_z = 0.0


CB_LEAD = CoinbaseLeadIndicator(CoinbaseLeadConfig())


# ─────────────────────────────────────────────────────────────────────────────
# Mispricing Sniper Signal
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MispricingConfig:
    # Required edge after fees
    edge_target_base: float = 0.020     # baseline minimum
    edge_target_late: float = 0.040     # late window stricter
    late_T_sec: float = 60.0

    # Price discipline
    max_pay_default: float = 0.85       # never pay above this in normal mode
    max_pay_late: float = 0.90          # slightly higher allowed late if EV huge

    # Additional cushion to avoid edge erosion at execution
    edge_buffer: float = 0.010

    # Require minimum probability dominance (avoid coinflip)
    # NOTE: p_cone already includes z-shrinkage + calibration anchoring,
    # so these thresholds are lower than if we gated on raw Φ(z).
    min_p: float = 0.58                 # baseline min_p (was 0.70 — double-shrunk)
    min_p_min: float = 0.55             # floor for dynamic min_p
    min_p_max: float = 0.72             # ceiling for dynamic min_p (calm markets)
    min_p_vol_boost: float = 0.10       # calm adds up to +0.10

    # When market is choppy, require bigger edge
    chop_edge_boost: float = 0.015

    # Dead book block
    dead_spread: float = 0.50

    # Kelly sizing (structural, not manual)
    kelly_fraction: float = 0.10        # 10% Kelly default
    kelly_max: float = 0.03             # cap: 3% bankroll per shot

    # Side-specific asymmetry knobs
    down_edge_mult: float = 1.10        # DOWN requires slightly more edge by default
    up_edge_mult: float = 1.00


def _phi(z: float) -> float:
    """Standard normal CDF using erfc (no scipy)."""
    return 0.5 * erfc(-z / math.sqrt(2.0))


def mispricing_sniper(
    *,
    side: str,                # "UP" or "DOWN" candidate side
    p_cone: float,            # calibrated cone probability (UP winning)
    z_ema: float,             # smoothed z (directional)
    dz: float,                # slope over lookback
    dz_norm: float,           # slope normalized by z_std
    z_std: float,             # rolling stdev of z_ema
    confirmed: bool,          # 2-tick expansion confirmation
    flip_rate: float,         # from RegimeClassifier
    T_sec: float,
    ask: float,               # ask price for the token we would BUY
    fee_fn,                   # fee_per_share(price)
    spread_max: float,
    sigma_eff: float = 0.0,   # effective sigma for vol-adjusted min_p
    oracle_age_ms: int = 0,   # age of oracle price in ms
    cb_lead_agrees: bool = False,   # CB Lead direction agrees with signal
    cb_anticipation: bool = False,  # CB Lead anticipation mode active
    basis_shrinking: bool = False,  # Coinbase-Poly basis converging
    edge_boost: float = 0.0,         # dynamic edge override from runtime calibration
    min_p_shift: float = 0.0,        # dynamic min_p shift from runtime calibration
    max_pay_shift: float = 0.0,      # dynamic max_pay shift from runtime calibration
    cfg: MispricingConfig = None,
) -> dict:
    """
    One-shot mispricing detector.

    Returns dict:
      ok (bool), p_token, cost_C, edge, max_pay, edge_target, reason,
      kelly_full, kelly_frac, min_p_dyn, P_sigma, dz_norm, z_std
    """
    cfg = cfg or MispricingConfig()

    # 0a) Stale oracle hard stop — never trade off stale data
    #     5s threshold for REST-polled Chainlink feed (~1s updates + HTTP latency)
    if oracle_age_ms > 5000:
        return {"ok": False, "reason": "stale_oracle", "oracle_age_ms": oracle_age_ms}

    # 0b) Dead book hard stop
    if spread_max >= cfg.dead_spread:
        return {"ok": False, "reason": "dead_book"}

    # ── Volatility-adjusted min_p ──
    P_sigma = percentile_rank(sigma_eff, _SIGMA_HISTORY) if _SIGMA_HISTORY else 0.5
    min_p_dyn = cfg.min_p + cfg.min_p_vol_boost * (1.0 - P_sigma)
    min_p_dyn += float(min_p_shift)
    min_p_dyn = min(cfg.min_p_max, max(cfg.min_p_min, min_p_dyn))

    # ── CB Lead agreement bonus: relax p threshold when Coinbase confirms ──
    # CB direction agrees + basis shrinking = strong cross-venue confirmation
    _cb_p_discount = 0.0
    if cb_lead_agrees and cb_anticipation:
        _cb_p_discount = 0.03                    # base discount for CB agreement
        if basis_shrinking:
            _cb_p_discount = 0.05                # stronger discount when basis converging
        min_p_dyn = max(cfg.min_p_min - 0.03, min_p_dyn - _cb_p_discount)

    # 1) Determine token win probability from cone model (already shrunk + calibrated)
    #    Using p_cone directly — NOT Φ(z_ema) which double-shrinks
    p_token = p_cone if side == "UP" else (1.0 - p_cone)

    # 2) Probability-driven direction gate: skip if model says ≤ coinflip
    if p_token <= 0.50:
        return {"ok": False, "reason": "p_wrong_side", "p_token": p_token}

    # 2b) Z momentum filter: z_ema must be expanding in our direction
    #     (not directional authority — p_cone decides that — just momentum confirmation)
    if not confirmed:
        return {"ok": False, "reason": "z_not_confirmed", "p_token": p_token,
                "dz": dz, "dz_norm": dz_norm}

    # 3) Fee-aware cost (compute early so p gate can see edge)
    fee = float(fee_fn(ask))
    C = float(ask + fee)

    # 4) Edge target (late window stricter)
    edge_target = cfg.edge_target_late if T_sec <= cfg.late_T_sec else cfg.edge_target_base
    edge_target *= (cfg.up_edge_mult if side == "UP" else cfg.down_edge_mult)
    edge_target += float(edge_boost)

    # 5) Soft p gate: only enforce probability dominance if edge isn't already strong
    #    Cheap asks (big edge) → let lower p through.
    #    Near-fair asks (thin edge) → demand higher p to avoid coinflip trades.
    _pre_edge = p_token - C
    if _pre_edge < edge_target:
        if p_token < min_p_dyn:
            return {"ok": False, "reason": "p_too_low", "p_token": p_token,
                    "min_p_dyn": min_p_dyn, "P_sigma": P_sigma,
                    "edge": _pre_edge, "edge_target": edge_target}

    # 6) Chop penalty: if flip_rate high, demand more edge
    if flip_rate > 0.08:
        edge_target += cfg.chop_edge_boost

    # 7) Dynamic price cap — scale with z strength
    #    Base 0.70, up to +0.10 when |z| is strong → allows 0.80-0.85 on conviction
    _z_scale = min(1.0, abs(z_ema) / 2.0)
    _dynamic_cap = 0.70 + 0.10 * _z_scale
    max_pay_regime = cfg.max_pay_late if T_sec <= cfg.late_T_sec else cfg.max_pay_default
    max_pay_regime = min(0.95, max(0.50, max_pay_regime + float(max_pay_shift)))
    max_pay = min(max_pay_regime, _dynamic_cap)
    max_pay = min(max_pay, max(0.01, p_token - cfg.edge_buffer))

    if ask > max_pay:
        return {
            "ok": False,
            "reason": "price_cap",
            "p_token": p_token,
            "ask": ask,
            "max_pay": max_pay,
        }

    # 8) Mispricing edge test
    edge = p_token - C
    if edge < edge_target:
        return {
            "ok": False,
            "reason": "edge_too_low",
            "p_token": p_token,
            "edge": edge,
            "edge_target": edge_target,
            "C": C,
        }

    # ── Kelly sizing ──
    denom = 1.0 - C
    kelly_full = 0.0
    if denom > 1e-9 and edge > 0:
        kelly_full = edge / denom
    kelly_frac = cfg.kelly_fraction * kelly_full
    kelly_frac = max(0.0, min(cfg.kelly_max, kelly_frac))

    return {
        "ok": True,
        "reason": "MISPRICING_OK",
        "p_token": p_token,
        "edge": edge,
        "edge_target": edge_target,
        "ask": ask,
        "C": C,
        "max_pay": max_pay,
        "kelly_full": kelly_full,
        "kelly_frac": kelly_frac,
        "min_p_dyn": min_p_dyn,
        "P_sigma": P_sigma,
        "dz_norm": dz_norm,
        "z_std": z_std,
    }


# Module-level singletons
Z_TRAJ = ZTrajectoryFilter(ZTrajConfig())
MIS_CFG = MispricingConfig()

# ── Sniper cooldown (one-shot lock) ─────────────────────────────────────
_LAST_SNIPER_FIRE_TS: int = 0
_SNIPER_COOLDOWN_MS: int = 3000

# ── Directional lock + micro-burst controls ──────────────────────────────
_LAST_SNIPER_SIDE: Optional[str] = None
_LAST_SNIPER_WINDOW: Optional[float] = None   # keyed by open_price
_Z_RESET_THRESHOLD: float = 0.6               # Z must collapse before flip
_MAX_SNIPER_FIRES_PER_SIDE: int = int(os.getenv("MAX_SNIPER_FIRES_PER_SIDE", "3"))
_MICRO_BURST_MIN_EDGE: float = float(os.getenv("MICRO_BURST_MIN_EDGE", "0.03"))
_MICRO_BURST_MIN_DEPTH: float = float(os.getenv("MICRO_BURST_MIN_DEPTH", "30.0"))
_WINDOW_SIDE_FIRE_COUNT: dict = {}


def confirm_sniper_fire() -> None:
    """Call AFTER order is actually placed (not on SIM suppress).
    Sets the cooldown timestamp so the engine won't re-fire for _SNIPER_COOLDOWN_MS."""
    global _LAST_SNIPER_FIRE_TS
    _LAST_SNIPER_FIRE_TS = ms_now()


def reset_sniper_lock() -> None:
    """Clear the one-shot sniper lock so the same side can fire again.
    Call this when a FOK order misses or is rejected (NON_MARKETABLE)."""
    global _LAST_SNIPER_SIDE, _LAST_SNIPER_WINDOW, _WINDOW_SIDE_FIRE_COUNT
    _LAST_SNIPER_SIDE = None
    _LAST_SNIPER_WINDOW = None
    _WINDOW_SIDE_FIRE_COUNT = {}


def ms_now() -> int:
    """Current time in milliseconds."""
    return int(time.time() * 1000)

# ─────────────────────────────────────────────────────────────────────────────
# Lag-Adaptive Z Gate
# ─────────────────────────────────────────────────────────────────────────────

class LagAdaptiveZ:
    """
    Directional lag-adaptive Z threshold + persistence.

    Tracks UP/DOWN lag distributions separately.
    Includes fast-tail detector and jackpot mode.

    - DOWN persistence multiplied by 0.65 (fear faster than greed)
    - Fast tail (p10 < 220ms): require edge >= 0.06 or skip
    - Jackpot mode: book-thinning detected → reduce persistence
    """

    PERSISTENCE_DOWN_MULT = 0.65  # DOWN fires faster
    FAST_TAIL_THRESHOLD_MS = 220  # p10 below this = apex predators
    FAST_TAIL_MIN_EDGE = 0.06     # minimum edge when fast tail detected

    def __init__(self, window: int = 500):
        self.lags_up: deque = deque(maxlen=window)
        self.lags_dn: deque = deque(maxlen=window)
        self.lags: deque = deque(maxlen=window)  # combined (backward compat)
        self.Z_BASE   = 0.65    # throughput baseline (tuned for 711ms median)
        self.LAG_REF  = 711.0   # measured p50 lag in ms
        self.K        = 0.0005  # sensitivity: 300ms→0.85, 711ms→0.65
        self.Z_MIN_LO = 0.55    # floor (slow day safety stop)
        self.Z_MIN_HI = 1.00    # ceiling (fast day brakes)

        # Jackpot mode state
        self._spread_buf: deque = deque(maxlen=200)
        self._flow_burst_count: int = 0
        self._jackpot_active: bool = False

    def add_lag(self, lag_ms: float, direction: int = 0) -> None:
        """Feed a lag measurement. direction: +1=UP, -1=DOWN, 0=unknown."""
        if lag_ms and lag_ms > 0:
            val = float(lag_ms)
            self.lags.append(val)
            if direction > 0:
                self.lags_up.append(val)
            elif direction < 0:
                self.lags_dn.append(val)

    def feed_spread(self, spread: float) -> None:
        """Feed current spread for jackpot detection."""
        self._spread_buf.append(spread)

    def feed_flow_burst(self, fill_count: int) -> None:
        """Feed fill count spike for jackpot detection."""
        self._flow_burst_count = fill_count

    def current_z(self, side: str = "UP") -> float:
        """Directional Z threshold based on per-side lag weather."""
        buf = self.lags_dn if side == "DOWN" else self.lags_up
        if not buf or len(buf) < 50:
            # Fall back to combined if directional insufficient
            if len(self.lags) >= 50:
                p50 = float(np.percentile(self.lags, 50))
                z = self.Z_BASE + self.K * (self.LAG_REF - p50)
                return float(max(self.Z_MIN_LO, min(self.Z_MIN_HI, z)))
            return self.Z_BASE
        p50 = float(np.percentile(buf, 50))
        z = self.Z_BASE + self.K * (self.LAG_REF - p50)
        return float(max(self.Z_MIN_LO, min(self.Z_MIN_HI, z)))

    def directional_z(self, side: str) -> float:
        """Alias for current_z(side) — backward compat."""
        return self.current_z(side)

    def persistence_mult(self, side: str) -> float:
        """Persistence multiplier for direction. DOWN=0.65, UP=1.0."""
        mult = self.PERSISTENCE_DOWN_MULT if side == "DOWN" else 1.0
        if self._jackpot_active:
            mult *= 0.5  # jackpot: halve persistence
        return mult

    def fast_tail_active(self, side: str) -> bool:
        """True if p10 lag for this direction is dangerously fast."""
        return self.p10_ms(side) < self.FAST_TAIL_THRESHOLD_MS

    def update_jackpot(self, spread: float, parity: float,
                       fill_count: int) -> bool:
        """Detect book-thinning 'jackpot' window. Returns True if active."""
        self.feed_spread(spread)
        signals = 0

        # 1) Spread jump relative to rolling median
        if len(self._spread_buf) >= 20:
            sorted_sp = sorted(self._spread_buf)
            median_sp = sorted_sp[len(sorted_sp) // 2]
            if spread > 1.5 * max(median_sp, 0.005):
                signals += 1

        # 2) Parity deviation
        if parity < 0.96 or parity > 1.06:
            signals += 1

        # 3) Flow burst then quiet
        if fill_count > 50:
            self._flow_burst_count = fill_count
        elif self._flow_burst_count > 50 and fill_count < 10:
            signals += 1
            self._flow_burst_count = 0

        self._jackpot_active = signals >= 2
        return self._jackpot_active

    def p10_ms(self, side: str = "UP") -> float:
        """10th percentile lag in ms for the given direction.
        Returns raw continuous value for institutional EV penalization.
        Used by AdaptiveExecutor for fill probability + regime classification."""
        buf = self.lags_dn if side == "DOWN" else self.lags_up
        if not buf:
            # Fall back to combined if directional empty
            if len(self.lags) >= 20:
                return float(np.percentile(self.lags, 10))
            return 9999.0  # conservative: assume high latency
        return float(np.percentile(buf, 10))

    def p50_ms(self, side: str = "UP") -> float:
        """50th percentile lag in ms (median). Used for lag-optimized gating."""
        buf = self.lags_dn if side == "DOWN" else self.lags_up
        if not buf:
            if len(self.lags) >= 20:
                return float(np.percentile(self.lags, 50))
            return self.LAG_REF
        return float(np.percentile(buf, 50))


# ─────────────────────────────────────────────────────────────────────────────
# Configuration Constants
# ─────────────────────────────────────────────────────────────────────────────

SPREAD_BLOCK  = 0.04                 # MC LOCKED: hard block above 4 cents
PARITY_LO, PARITY_HI = 0.98, 1.10   # up_ask + down_ask must be in this range
WINDOW_LO, WINDOW_HI = 20.0, 240.0  # first 60s blocked, last 20s blocked

EDGE_BASE     = 0.015   # MC LOCKED: proven base edge floor
EDGE_SPREAD_K = 0.8     # spread component: min_ev = 0.015 + 0.8 * spread
DOWN_EV_MULT  = 1.25    # DOWN needs 25% more edge
TICK_BUFFER   = 0.02    # 2 cent limit price buffer

# ── Calibration constants (CALIBRATION.md / TAIL_RISK.md) ──────────────────
SIGMA_FLOOR       = 0.00008   # prevents z → ∞ when vol collapses
Z_SHRINK_LAMBDA   = 0.07      # z_adj = z / (1 + λ|z|)  — compresses extremes
CALIBRATION_ALPHA = 0.93      # P_cal = α·Φ(z) + (1-α)·0.5  — default shrinkage
SIZE_SHARES = 50.0 # 5x test size (was 10.0)


# ─────────────────────────────────────────────────────────────────────────────
# Edge-Adaptive Ladders
# ─────────────────────────────────────────────────────────────────────────────

def ask_cap_for_edge(edge: float, z_ema: float = 0.0) -> float:
    """
    Max ask price we're willing to pay, based on edge + z strength.
    Dynamic cap: 0.70 + 0.10 * z_scale (allows up to 0.80 on conviction).
    """
    z_scale = min(1.0, abs(z_ema) / 2.0)
    dynamic_cap = 0.70 + 0.10 * z_scale
    if edge >= 0.12: return min(dynamic_cap, 0.85)
    if edge >= 0.08: return min(dynamic_cap, 0.80)
    if edge >= 0.05: return min(dynamic_cap, 0.75)
    if edge >= 0.03: return min(0.70, dynamic_cap)
    if edge >= 0.015: return 0.60
    return 0.0


def required_persistence(edge: float) -> float:
    """Seconds the signal must persist before firing. Lower edge → longer wait."""
    if edge >= 0.12: return 0.05
    if edge >= 0.08: return 0.15
    if edge >= 0.05: return 0.35
    return 0.70


def required_persistence_lag(edge: float) -> float:
    """Lag-optimized persistence ladder tuned to measured quantiles.
    t1 = 0.15 * p50 = 0.11s (big edge)
    t2 = 0.55 * p50 = 0.39s (mid edge)
    t3 = 0.85 * p75 = 0.82s (weak edge)"""
    if edge >= 0.08: return 0.11
    if edge >= 0.04: return 0.39
    return 0.82


def fee_per_share(price: float) -> float:
    """Polymarket fee per MATH.md: f = p(1-p) × r, r=0.0625.
    At p=0.50 → 0.0156, at p=0.95 → 0.0030, at p=0.05 → 0.0030."""
    FEE_RATE = 0.0625
    return price * (1.0 - price) * FEE_RATE


# ─────────────────────────────────────────────────────────────────────────────
# Cone probability model
# ─────────────────────────────────────────────────────────────────────────────

def cone_p_and_z(
    btc_price: float,
    open_price: float,
    sec_remaining: float,
    sigma_1m: float,
    calibration_alpha: float = CALIBRATION_ALPHA,
) -> Tuple[float, float]:
    """
    Log-normal cone probability + z-score for BTC finishing above strike.

    P(finish above strike) ≈ Φ(z_adj),  where:
        z_raw = ln(btc/strike) / (σ * √T)
        z_adj = z_raw / (1 + λ|z_raw|)    # shrinkage
        P_cal = α·Φ(z_adj) + (1-α)·0.5    # calibration anchor

    Returns (p_cone, z_adj).
    """
    if sec_remaining <= 0 or sigma_1m <= 0:
        return 0.5, 0.0

    sigma = max(sigma_1m, SIGMA_FLOOR)
    t_frac = sec_remaining / 60.0
    vol = sigma * math.sqrt(t_frac)

    if vol < 1e-12:
        return 0.5, 0.0

    z_raw = math.log(btc_price / open_price) / vol

    # z shrinkage: prevents extreme z values from dominating
    z_adj = z_raw / (1.0 + Z_SHRINK_LAMBDA * abs(z_raw))

    # Calibrated probability: anchor toward 0.5
    p_raw = 0.5 * erfc(-z_adj / math.sqrt(2.0))
    p_cal = calibration_alpha * p_raw + (1.0 - calibration_alpha) * 0.5

    return float(np.clip(p_cal, 0.01, 0.99)), float(z_adj)


def lag_adjusted_min_ev(spread_max: float, lag_p50_ms: float) -> float:
    """Lag-adjusted minimum EV: higher lag → need more edge to compensate."""
    base = EDGE_BASE + EDGE_SPREAD_K * spread_max
    lag_s = lag_p50_ms / 1000.0
    lag_penalty = 0.003 * max(0, lag_s - 0.30)
    return base + lag_penalty


# ─────────────────────────────────────────────────────────────────────────────
# Persistence State
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PersistState:
    side: Optional[str] = None
    start_ms: Optional[int] = None
    wait_ms: Optional[int] = None
    impulse_start_ms: Optional[int] = None


# NOTE: Module-level default; callers should pass their own persist instance to decide_edge()
persist = PersistState()


# ─────────────────────────────────────────────────────────────────────────────
# Main Decision Engine
# ─────────────────────────────────────────────────────────────────────────────

_log = logging.getLogger("polybot")


def decide_edge(
    *,
    btc_price: float,
    open_price: float,
    sec_remaining: float,
    sigma_1m: float,
    up_bid: float, up_ask: float,
    down_bid: float, down_ask: float,
    flow_imbalance: float,
    lag_adaptive: LagAdaptiveZ,
    persist: PersistState,
    regime_params: RegimeParams,
    coverage_bias: float = 0.0,
    coverage_bias_dn: float = 0.0,
    sigma_fast: float = 0.0,
    sigma_slow: float = 0.0,
    flip_rate: float = 0.0,
    window_key: int = 0,
    **kw,
) -> Tuple[Optional[str], Optional[str], Optional[float], float, dict]:
    """
    Core edge-decision function.
    debug_dict["reason"] == "FIRE" means we should trade.

    Returns (side, token, limit_price, size, debug).
    """
    now = int(time.time() * 1000)

    rp = regime_params
    sigma = sigma_1m
    # FlowTracker emits centered flow in [-1, 1]. Normalize once for
    # threshold logic that is defined on [0, 1].
    flow_centered = float(np.clip(float(flow_imbalance), -1.0, 1.0))
    flow_01 = 0.5 * (flow_centered + 1.0)

    # ── Dynamic calibration alpha: relax on high-conviction clean signals ──
    # Preliminary z to gauge signal strength (no shrinkage/calibration)
    _cal_alpha = rp.calibration_alpha
    if sec_remaining > 0 and sigma > 0:
        _pre_vol = max(sigma, SIGMA_FLOOR) * math.sqrt(sec_remaining / 60.0)
        if _pre_vol > 1e-12 and open_price > 0:
            _pre_z = abs(math.log(btc_price / open_price) / _pre_vol)
            _flow_clean = abs(flow_centered) < 0.30  # neutral flow = no adverse selection
            _regime_clean = rp.label in ("CALM", "NORMAL")
            if _pre_z > 1.5 and _flow_clean and _regime_clean:
                # Strong clean signal: trust model more, less 0.5 anchoring
                _cal_alpha = min(1.0, rp.calibration_alpha + 0.05)
            if _pre_z > 2.5 and _flow_clean and rp.label == "CALM":
                # Very strong CALM signal: near-full model trust
                _cal_alpha = min(1.0, rp.calibration_alpha + 0.07)

    # Compute cone probability
    p, z = cone_p_and_z(btc_price, open_price, sec_remaining, sigma,
                        calibration_alpha=_cal_alpha)

    debug: dict = {
        "z": z, "p_cone": p, "sigma": sigma,
        "regime": rp.label, "T": sec_remaining,
        "cal_alpha_used": _cal_alpha,
        "sigma_fast": sigma_fast if sigma_fast > 0 else sigma,
        "sigma_slow": sigma_slow if sigma_slow > 0 else sigma,
        "flow_centered": flow_centered,
        "flow_01": flow_01,
    }

    # 0) Oracle freshness — never evaluate when oracle is stale
    #    5s threshold for REST-polled Chainlink feed (~1s updates + HTTP latency)
    _oracle_age_ms = int(kw.get("oracle_age_ms", 0))
    if _oracle_age_ms > 5000:
        return None, None, None, 0.0, {"reason": "stale_oracle", "oracle_age_ms": _oracle_age_ms, **debug}

    # 1) Time window
    if sec_remaining < WINDOW_LO or sec_remaining > WINDOW_HI:
        return None, None, None, 0.0, {"reason": "time_window", **debug}

    # 2) Late-game Kelly multiplier
    late_kelly_mult = 1.0
    if sec_remaining < 60:
        late_kelly_mult = max(0.3, sec_remaining / 60.0)

    # 3) Spread + parity — tiered logic
    # Strict spread semantics: never use abs() here, because a crossed book
    # (ask < bid) is a data integrity issue, not a "tight spread".
    spread_up = max(0.0, float(up_ask) - float(up_bid))
    spread_dn = max(0.0, float(down_ask) - float(down_bid))
    spread_max = max(spread_up, spread_dn)

    # Crossed books imply stale/corrupt snapshots — skip this tick.
    if (float(up_ask) < float(up_bid)) or (float(down_ask) < float(down_bid)):
        persist.side = persist.start_ms = persist.wait_ms = None
        return None, None, None, 0.0, {
            "reason": "crossed_book",
            "spread_max": spread_max,
            **debug,
        }

    # ─────────────────────────────────────────────────────────────
    # Book sanity (must run BEFORE parity or sniper)
    # ─────────────────────────────────────────────────────────────

    # 1) Hard dead-book block — only when BOTH sides are dead
    #    (one-sided flickers are handled by implied pricing in main.py)
    spread_min = min(spread_up, spread_dn)
    if spread_min >= 0.50:
        return None, None, None, 0.0, {
            "reason": "dead_book",
            "spread_max": spread_max,
            "spread_min": spread_min,
            **debug,
        }

    # 1b) Extreme price block — both asks near 0.99 or both bids near 0.01
    #     Only trigger when spread_min also confirms genuine dead book (not WS flicker)
    if spread_min >= 0.40:
        if up_ask > 0.95 and down_ask > 0.95:
            return None, None, None, 0.0, {
                "reason": "dead_book_extreme",
                "up_ask": up_ask, "down_ask": down_ask,
                "spread_max": spread_max, "spread_min": spread_min,
                **debug,
            }
        if up_bid < 0.05 and down_bid < 0.05:
            return None, None, None, 0.0, {
                "reason": "dead_book_extreme",
                "up_bid": up_bid, "down_bid": down_bid,
                "spread_max": spread_max, "spread_min": spread_min,
                **debug,
            }

    # 2) Soft spread guard (do NOT over-block sniper)
    SOFT_SPREAD_BLOCK = 0.10  # was 0.04 — too strict

    if spread_max > SOFT_SPREAD_BLOCK:
        # only block if price dislocation not strong
        # allow sniper to still evaluate mispricing
        spread_flag = True
    else:
        spread_flag = False

    # 3) Parity check only if BOTH books look real
    book_real = (
        up_ask < 0.95 and down_ask < 0.95 and
        up_bid > 0.05 and down_bid > 0.05
    )

    if book_real:
        parity = up_ask + down_ask
        if not (0.98 <= parity <= 1.10):
            return None, None, None, 0.0, {
                "reason": "parity_out_of_range",
                "parity": parity,
                "spread_max": spread_max,
                **debug,
            }
    else:
        parity = up_ask + down_ask  # still compute for debug

    # Spread tiering for execution style
    if spread_max > 0.03:
        spread_tier = "maker_only"     # 3+ cents: post inside spread
        spread_size_mult = 0.40
    elif spread_max > 0.02:
        spread_tier = "maker_preferred" # 2-3 cents: maker-first, IOC fallback
        spread_size_mult = 0.60
    else:
        spread_tier = "taker_allowed"   # < 2 cents: full aggression
        spread_size_mult = 1.0

    debug.update({"spread_max": spread_max, "parity": parity,
                  "spread_tier": spread_tier, "spread_size_mult": spread_size_mult,
                  "spread_flag": spread_flag, "book_real": book_real,
                  "late_kelly_mult": late_kelly_mult})

    # 4) Jackpot detection
    _jackpot = lag_adaptive.update_jackpot(spread_max, parity, 0)

    # ═══════════════════════════════════════════════════════════════
    # MISPRICING SNIPER — one-shot dislocation entry
    # ═══════════════════════════════════════════════════════════════

    # NOTE: "both asks > 0.95" dead-book removed — main.py MARKET_DECIDED
    # gate handles this at line 3548, and implied book values caused false positives.

    # Clear stale persist state (sniper is stateless — no persistence filter)
    persist.side = persist.start_ms = persist.wait_ms = None
    persist.impulse_start_ms = None

    # Late-window persistence relaxation: let momentum fire faster near expiry
    Z_TRAJ.apply_time_relaxation(sec_remaining)

    # Z trajectory update (EMA + slope + normalized + confirmation)
    traj = Z_TRAJ.update(z)
    z_ema = traj["z_ema"]
    dz = traj["dz"]
    z_std = traj["z_std"]
    dz_norm = traj["dz_norm"]
    confirmed = traj["confirmed"]

    debug.update({
        "z_ema": z_ema, "dz": dz,
        "z_std": z_std, "dz_norm": dz_norm, "confirmed": confirmed,
        "p_up": _phi(z_ema),
        "flip_rate": float(flip_rate or 0.0),
    })

    # oracle_age_ms already extracted at top of decide_edge

    # Evaluate both sides, pick best edge
    cand = []
    for s in ("UP", "DOWN"):
        _ask = up_ask if s == "UP" else down_ask
        _side_l = str(s).upper()
        _edge_boost = float(kw.get("up_edge_boost", 0.0) if _side_l == "UP" else kw.get("down_edge_boost", 0.0))
        _min_p_shift = float(kw.get("up_min_p_shift", 0.0) if _side_l == "UP" else kw.get("down_min_p_shift", 0.0))
        _max_pay_shift = float(kw.get("up_max_pay_shift", 0.0) if _side_l == "UP" else kw.get("down_max_pay_shift", 0.0))

        res = mispricing_sniper(
            side=s,
            p_cone=p,
            z_ema=z_ema,
            dz=dz,
            dz_norm=dz_norm,
            z_std=z_std,
            confirmed=confirmed,
            flip_rate=float(flip_rate or 0.0),
            T_sec=float(sec_remaining),
            ask=float(_ask),
            fee_fn=fee_per_share,
            spread_max=float(spread_max),
            sigma_eff=float(sigma),
            oracle_age_ms=_oracle_age_ms,
            cb_lead_agrees=bool(kw.get("cb_lead_agrees", False)),
            cb_anticipation=bool(kw.get("cb_anticipation", False)),
            basis_shrinking=bool(kw.get("basis_shrinking", False)),
            edge_boost=_edge_boost,
            min_p_shift=_min_p_shift,
            max_pay_shift=_max_pay_shift,
            cfg=MIS_CFG,
        )
        res["side"] = s
        cand.append(res)

    best = None
    for r in cand:
        if r.get("ok"):
            if best is None or r["edge"] > best["edge"]:
                best = r

    if best is None:
        # No sniper opportunity — return best rejection reason
        _rej = cand[0] if cand else {"reason": "NO_SIGNAL"}
        debug.update({
            "reason": _rej.get("reason", "NO_SIGNAL"),
            **{k: v for k, v in _rej.items() if k != "ok"},
        })
        return None, None, None, 0.0, debug

    # ═══════════════════════════════════════════════════════════════
    # FIRE_MISPRICING — one-shot at full planned size
    # ═══════════════════════════════════════════════════════════════

    # Sniper cooldown: prevent repeated firing when Z stays valid
    global _LAST_SNIPER_FIRE_TS, _LAST_SNIPER_SIDE, _LAST_SNIPER_WINDOW
    _now_fire = ms_now()
    if _now_fire - _LAST_SNIPER_FIRE_TS < _SNIPER_COOLDOWN_MS:
        debug["reason"] = "sniper_cooldown"
        debug["cooldown_remaining_ms"] = _SNIPER_COOLDOWN_MS - (_now_fire - _LAST_SNIPER_FIRE_TS)
        return None, None, None, 0.0, debug

    side = best["side"]

    # ── Directional lock: one entry per side per window ──
    current_window = window_key if window_key is not None else int(open_price * 100)  # fallback
    _side_key = (current_window, side)
    _fire_count = int(_WINDOW_SIDE_FIRE_COUNT.get(_side_key, 0))
    _top_depth = float(kw.get("top_depth", 0.0) or 0.0)
    if _fire_count >= _MAX_SNIPER_FIRES_PER_SIDE:
        debug["reason"] = "micro_burst_side_cap"
        debug["side"] = side
        debug["side_fires"] = _fire_count
        return None, None, None, 0.0, debug
    if _fire_count > 0 and best["edge"] < _MICRO_BURST_MIN_EDGE:
        debug["reason"] = "micro_burst_requal_edge"
        debug["edge"] = best["edge"]
        debug["edge_need"] = _MICRO_BURST_MIN_EDGE
        return None, None, None, 0.0, debug
    if _fire_count > 0 and _top_depth < _MICRO_BURST_MIN_DEPTH:
        debug["reason"] = "micro_burst_requal_depth"
        debug["top_depth"] = _top_depth
        debug["depth_need"] = _MICRO_BURST_MIN_DEPTH
        return None, None, None, 0.0, debug

    # ── Z-reset guard: require Z collapse before flipping direction ──
    if (_LAST_SNIPER_SIDE is not None and
            _LAST_SNIPER_WINDOW == current_window and
            side != _LAST_SNIPER_SIDE and
            abs(z_ema) >= _Z_RESET_THRESHOLD):
        debug["reason"] = "no_reset_before_flip"
        debug["z_ema"] = z_ema
        return None, None, None, 0.0, debug

    # ── Passive-cross limit price (no +0.01 over-cross) ──
    limit_price = best["ask"]
    if limit_price > best["max_pay"]:
        return None, None, None, 0.0, {
            "reason": "price_cap_execution",
            "ask": best["ask"],
            "max_pay": best["max_pay"],
        }
    limit_price = min(0.99, max(0.01, round(limit_price, 2)))

    # Commit side/window lock (cooldown timestamp set by caller after order placed)
    _LAST_SNIPER_SIDE = side
    _LAST_SNIPER_WINDOW = current_window
    _WINDOW_SIDE_FIRE_COUNT[_side_key] = _fire_count + 1
    size = SIZE_SHARES

    # TRINITY diagnostic (v2: includes Kelly + vol-adjusted fields)
    _log.info(
        f"TRINITY: reason=FIRE_MISPRICING "
        f"side={side} p_token={best['p_token']:.4f} "
        f"edge={best['edge']:.4f} edge_target={best['edge_target']:.4f} "
        f"ask={best['ask']:.2f} max_pay={best['max_pay']:.2f} "
        f"z_ema={z_ema:.3f} dz={dz:.4f} dz_norm={dz_norm:.3f} z_std={z_std:.4f} "
        f"kelly_full={best.get('kelly_full', 0):.4f} kelly_frac={best.get('kelly_frac', 0):.4f} "
        f"min_p_dyn={best.get('min_p_dyn', 0):.3f} P_sigma={best.get('P_sigma', 0):.3f} "
        f"spread={spread_max:.3f} T={sec_remaining:.0f}s "
        f"flip_rate={float(flip_rate or 0.0):.3f} regime={rp.label}"
    )

    debug.update({
        "reason": "FIRE_MISPRICING",
        "mode": "mispricing",
        "side": side,
        "z_ema": z_ema,
        "dz": dz,
        "dz_norm": dz_norm,
        "z_std": z_std,
        "p_token": best["p_token"],
        "edge": best["edge"],
        "edge_target": best["edge_target"],
        "max_pay": best["max_pay"],
        "limit_price": limit_price,
        "kelly_full": best.get("kelly_full", 0.0),
        "kelly_frac": best.get("kelly_frac", 0.0),
        "min_p_dyn": best.get("min_p_dyn", 0.0),
        "P_sigma": best.get("P_sigma", 0.0),
        "side_fires": _WINDOW_SIDE_FIRE_COUNT.get(_side_key, 1),
        "max_side_fires": _MAX_SNIPER_FIRES_PER_SIDE,
    })
    return side, None, float(limit_price), float(size), debug
