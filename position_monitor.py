# position_monitor.py — Unified EV-driven position exit engine
"""
Subsumes ExitManager + EndgameManager into one p/EV-based decision system.

All exit decisions are probabilistic — no fixed price thresholds.
  hold_EV = p_side (probability this token settles at $1.00)
  sell_EV = bid - fee(bid)
  fee     = bid * (1-bid) * 0.0625

Decision priority (checked in order):
  1. PROB STOP-LOSS: p collapsed AND sell_EV >= hold_EV - penalty
  2. EV EXIT: profitable AND market overpaying vs model
  3. GAMMA DANGER: in uncertainty band AND profitable AND sell_EV adequate
  4. TRAILING STOP: in gamma band AND trail activated AND bid < HWM
  5. ENDGAME (T < 25s): strong loser → SELL; gamma+profitable → SELL

Key design properties:
  - Won't sell p=0.97 at bid=0.90 (hold_EV=0.97 > sell_EV)
  - Won't stop on temporary illiquidity (requires p collapse)
  - Trailing stop only in gamma zone (prevents churning strong winners)
  - Flicker guard: trailing stop requires persistence + spread sanity

Usage:
    PM = PositionMonitor()
    decision = PM.evaluate("UP", entry_price=0.15, bid=0.24,
                           p_cone=0.26, T_sec=180, sigma_1m=0.002,
                           regime="CALM", now_ms=ms_now())
    if decision.action == "SELL":
        # queue IOC sell order
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional
import math
import logging
import time

logger = logging.getLogger("position_monitor")

Side = Literal["UP", "DOWN"]
FEE_RATE = 0.0625  # Polymarket fee formula: f = p*(1-p)*r


def _fee(price: float) -> float:
    """Polymarket fee: price * (1 - price) * 0.0625"""
    return price * (1.0 - price) * FEE_RATE


@dataclass
class PositionMonitorConfig:
    # ── EV exit: market overpaying vs hold EV ──
    fee_buffer: float = 0.005
    min_profit_for_ev_exit: float = 0.02  # 2c minimum locked profit

    # ── Probabilistic stop-loss (p collapsed, not just low bid) ──
    p_cut_loss_early: float = 0.15   # at T > 60s
    p_cut_loss_late: float = 0.20    # at T < 30s
    stop_penalty_early: float = 0.04  # max EV haircut for stop
    stop_penalty_late: float = 0.02

    # ── Gamma danger (profitable in uncertainty band) ──
    gamma_band_mult: float = 2.0
    gamma_band_min: float = 0.20
    gamma_buffer: float = 0.01

    # ── Trailing stop (ONLY inside gamma band) ──
    trail_offset_early: float = 0.08   # wide trail at T > 60s
    trail_offset_late: float = 0.03    # tight trail at T < 30s
    trail_activation_profit: float = 0.03  # minimum profit to activate trailing

    # ── Trailing stop flicker guard ──
    trail_persistence_ms: int = 500    # bid must stay below trail for this long
    trail_min_spread: float = 0.08     # don't trail-stop on wide/illiquid books

    # ── Endgame (subsumes EndgameManager) ──
    endgame_T: float = 25.0
    endgame_strong_win: float = 0.85
    endgame_strong_loss: float = 0.15

    # ── Variance penalty by regime (base multipliers) ──
    variance_base: float = 0.02
    variance_gamma_k: float = 0.06
    # Regime multipliers for variance penalty
    regime_variance_mult: dict = field(default_factory=lambda: {
        "CALM": 0.8,         # less variance penalty in calm
        "NORMAL": 1.0,
        "TRANSITION": 1.2,
        "VOL_EVENT": 1.5,
        "HIGH_VOL": 1.8,
        "ADVERSARIAL": 2.0,  # max penalty in adversarial
    })

    # ── Controls ──
    min_eval_interval_ms: int = 100
    reentry_cooldown_ms: int = 5000
    min_bid: float = 0.02


@dataclass
class ExitDecision:
    action: str        # "SELL" or "HOLD"
    reason: str        # machine-readable reason code
    hold_ev: float     # p_side
    sell_ev: float     # bid - fee
    profit: float      # sell_ev - entry_price
    penalty: float     # variance/gamma penalty applied
    ev_gap: float      # hold_ev - sell_ev


class PositionMonitor:
    """
    Unified exit engine evaluating every tick whether to hold or sell.
    """

    def __init__(self, cfg: PositionMonitorConfig = None):
        self.cfg = cfg or PositionMonitorConfig()
        self._last_eval_ms: int = 0

        # Per-side trailing stop state
        self._trail_hwm: dict[str, float] = {"UP": 0.0, "DOWN": 0.0}
        self._trail_activated: dict[str, bool] = {"UP": False, "DOWN": False}

        # Flicker guard: timestamp when bid first dropped below trail level
        self._trail_breach_ts: dict[str, int] = {"UP": 0, "DOWN": 0}

        # Re-entry cooldown per side
        self._reentry_blocked_until: dict[str, int] = {"UP": 0, "DOWN": 0}

    def reset_window(self, now_ms: int = 0):
        """Call on window rollover to clear trailing state."""
        self._trail_hwm = {"UP": 0.0, "DOWN": 0.0}
        self._trail_activated = {"UP": False, "DOWN": False}
        self._trail_breach_ts = {"UP": 0, "DOWN": 0}
        self._reentry_blocked_until = {"UP": 0, "DOWN": 0}
        self._last_eval_ms = 0

    def is_reentry_blocked(self, side: str, now_ms: int) -> bool:
        """After an exit, block re-entry for cooldown period."""
        return now_ms < self._reentry_blocked_until.get(side, 0)

    def _gamma_band(self, sigma_1m: float, T_sec: float) -> tuple[float, float]:
        """Dynamic uncertainty band centered at 0.5."""
        sigma_T = sigma_1m * math.sqrt(max(0.01, T_sec) / 60.0)
        band = max(self.cfg.gamma_band_min,
                   self.cfg.gamma_band_mult * sigma_T)
        lower = max(0.0, 0.5 - band)
        upper = min(1.0, 0.5 + band)
        return lower, upper

    def _interpolate_by_time(self, T_sec: float,
                              early_val: float, late_val: float,
                              T_early: float = 120.0,
                              T_late: float = 30.0) -> float:
        """Linear interpolation between early/late values based on time remaining."""
        if T_sec >= T_early:
            return early_val
        if T_sec <= T_late:
            return late_val
        frac = (T_sec - T_late) / (T_early - T_late)
        return late_val + frac * (early_val - late_val)

    def _variance_penalty(self, p_side: float, T_sec: float,
                           sigma_1m: float, regime: str) -> float:
        """
        Compute variance penalty — how much EV haircut we accept
        to avoid binary settlement risk.
        """
        cfg = self.cfg
        regime_mult = cfg.regime_variance_mult.get(regime, 1.0)

        # Base penalty (always applied)
        penalty = cfg.variance_base * regime_mult

        # Uncertainty: 0 at p=0/1, 1 at p=0.5
        uncertainty = 4.0 * p_side * (1.0 - p_side)

        # Time urgency: increases as T → 0
        if T_sec < 45.0:
            time_urgency = 1.0 + (45.0 - T_sec) / 45.0
            penalty += cfg.variance_gamma_k * uncertainty * time_urgency * regime_mult
        if T_sec < 20.0:
            # Very close to expiry: extra gamma penalty
            penalty += cfg.variance_gamma_k * 0.5 * regime_mult

        # Near-certain outcomes (p > 0.88 or p < 0.12): reduce penalty
        if not (0.12 < p_side < 0.88):
            penalty *= 0.25

        return penalty

    def evaluate(
        self,
        side: str,           # "UP" or "DOWN"
        entry_price: float,  # avg cost basis
        bid: float,          # live bid for this token
        p_cone: float,       # model probability of UP winning
        T_sec: float,        # seconds remaining
        sigma_1m: float,     # 1-min realized vol
        regime: str = "NORMAL",
        spread: float = 0.10,  # current book spread
        now_ms: int = 0,
    ) -> ExitDecision:
        """
        Core evaluation: should we hold or sell this position?

        Returns ExitDecision with action, reason, and full EV metadata.
        """
        cfg = self.cfg
        if now_ms <= 0:
            now_ms = int(time.time() * 1000)

        # Throttle check
        if now_ms - self._last_eval_ms < cfg.min_eval_interval_ms:
            return ExitDecision("HOLD", "throttled", 0.0, 0.0, 0.0, 0.0, 0.0)
        self._last_eval_ms = now_ms

        # ── Core EV math ──
        p_side = p_cone if side == "UP" else (1.0 - p_cone)
        hold_ev = p_side

        # Bid sanity
        if bid < cfg.min_bid:
            return ExitDecision("HOLD", "bid_too_low", hold_ev, 0.0, 0.0, 0.0, hold_ev)

        price = min(0.99, max(0.01, bid))
        sell_ev = price - _fee(price)
        profit = sell_ev - entry_price
        ev_gap = hold_ev - sell_ev
        penalty = self._variance_penalty(p_side, T_sec, sigma_1m, regime)

        # ── Update trailing stop HWM ──
        if sell_ev > self._trail_hwm.get(side, 0.0):
            self._trail_hwm[side] = sell_ev
            self._trail_breach_ts[side] = 0  # reset breach on new HWM
        if profit >= cfg.trail_activation_profit:
            self._trail_activated[side] = True

        gamma_lo, gamma_hi = self._gamma_band(sigma_1m, T_sec)
        in_gamma = gamma_lo <= p_side <= gamma_hi

        # ═══════════════════════════════════════════════════════════
        # DECISION CASCADE (checked in priority order)
        # ═══════════════════════════════════════════════════════════

        # ── 1. PROBABILISTIC STOP-LOSS ──
        # p has collapsed → model says we're likely losing.
        # Only stop if sell_EV >= hold_EV - stop_penalty (don't sell
        # into thin air; requires p to collapse, not just low bid).
        p_cut = self._interpolate_by_time(
            T_sec, cfg.p_cut_loss_early, cfg.p_cut_loss_late,
            T_early=120.0, T_late=30.0)
        stop_pen = self._interpolate_by_time(
            T_sec, cfg.stop_penalty_early, cfg.stop_penalty_late,
            T_early=120.0, T_late=30.0)

        if p_side < p_cut and sell_ev >= (hold_ev - stop_pen):
            self._reentry_blocked_until[side] = now_ms + cfg.reentry_cooldown_ms
            return ExitDecision("SELL", "prob_stop_loss",
                                hold_ev, sell_ev, profit, stop_pen, ev_gap)

        # ── 2. EV EXIT ──
        # Market is paying us close to (or more than) our hold_EV.
        # Lock in guaranteed profit instead of gambling on settlement.
        if profit >= cfg.min_profit_for_ev_exit:
            if sell_ev >= (hold_ev - cfg.fee_buffer):
                self._reentry_blocked_until[side] = now_ms + cfg.reentry_cooldown_ms
                return ExitDecision("SELL", "ev_exit_lock_profit",
                                    hold_ev, sell_ev, profit, cfg.fee_buffer, ev_gap)

        # ── 3. GAMMA DANGER ──
        # Position is profitable AND we're in the high-uncertainty zone.
        # Take the guaranteed profit to avoid binary coin-flip risk.
        if in_gamma and profit > 0 and sell_ev >= (hold_ev - cfg.gamma_buffer):
            self._reentry_blocked_until[side] = now_ms + cfg.reentry_cooldown_ms
            return ExitDecision("SELL", "gamma_danger",
                                hold_ev, sell_ev, profit, cfg.gamma_buffer, ev_gap)

        # ── 4. TRAILING STOP (only in gamma band) ──
        # Prevents giving back gains in the uncertainty zone.
        # Flicker guard: bid must stay below trail level for
        # trail_persistence_ms AND book must not be illiquid.
        if in_gamma and self._trail_activated.get(side, False):
            trail_offset = self._interpolate_by_time(
                T_sec, cfg.trail_offset_early, cfg.trail_offset_late,
                T_early=120.0, T_late=30.0)

            hwm = self._trail_hwm.get(side, 0.0)
            trail_level = hwm - trail_offset

            if sell_ev < trail_level:
                # Bid is below trail level — check flicker guard
                breach_ts = self._trail_breach_ts.get(side, 0)
                if breach_ts == 0:
                    # First breach: record timestamp
                    self._trail_breach_ts[side] = now_ms
                elif (now_ms - breach_ts >= cfg.trail_persistence_ms
                      and spread <= cfg.trail_min_spread):
                    # Persistent breach + liquid book → fire trailing stop
                    self._reentry_blocked_until[side] = now_ms + cfg.reentry_cooldown_ms
                    return ExitDecision("SELL", "trailing_stop",
                                        hold_ev, sell_ev, profit, trail_offset, ev_gap)
            else:
                # Bid recovered above trail level → reset breach timer
                self._trail_breach_ts[side] = 0

        # ── 5. ENDGAME (T < 25s) ──
        if T_sec < cfg.endgame_T:
            # Strong winner: HOLD (let it settle at $1)
            if p_side >= cfg.endgame_strong_win:
                return ExitDecision("HOLD", "endgame_strong_win",
                                    hold_ev, sell_ev, profit, penalty, ev_gap)

            # Strong loser: SELL (salvage what we can)
            if p_side <= cfg.endgame_strong_loss:
                self._reentry_blocked_until[side] = now_ms + cfg.reentry_cooldown_ms
                return ExitDecision("SELL", "endgame_strong_loss",
                                    hold_ev, sell_ev, profit, penalty, ev_gap)

            # In gamma band + profitable → lock it in
            if in_gamma and profit > 0:
                self._reentry_blocked_until[side] = now_ms + cfg.reentry_cooldown_ms
                return ExitDecision("SELL", "endgame_gamma_exit",
                                    hold_ev, sell_ev, profit, penalty, ev_gap)

            # Outside gamma but EV says sell
            if ev_gap < cfg.fee_buffer:
                self._reentry_blocked_until[side] = now_ms + cfg.reentry_cooldown_ms
                return ExitDecision("SELL", "endgame_ev_exit",
                                    hold_ev, sell_ev, profit, penalty, ev_gap)

        # ── DEFAULT: HOLD ──
        return ExitDecision("HOLD", "hold_ev_superior",
                            hold_ev, sell_ev, profit, penalty, ev_gap)

    def status_dict(self, side: str) -> dict:
        """Return current monitoring state for debug logging."""
        return {
            "pm_trail_hwm": round(self._trail_hwm.get(side, 0.0), 4),
            "pm_trail_active": self._trail_activated.get(side, False),
            "pm_reentry_blocked": self._reentry_blocked_until.get(side, 0),
        }
