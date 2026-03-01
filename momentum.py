# momentum.py — Orthogonal Momentum Layer + Regime Arbitration
"""
Second alpha engine that activates when z stays extreme.
Works alongside mean-reversion (MR) — not a replacement.

Regime Arbitration:
    if jump regime → suppress MR (handled by bipower_jump)
    if strong trend → MOMENTUM mode
    else            → MR (default)

Momentum mode triggers when:
    |z| > threshold AND persistence > N seconds
    AND centered flow agrees  AND ADX > 25 (trend strength)
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import math
from typing import Optional


@dataclass
class MomentumConfig:
    # Activation thresholds
    z_mom_on: float = 1.8           # |z| must exceed this to consider momentum
    z_mom_persist_s: float = 3.0    # z must stay extreme this long
    adx_threshold: float = 25.0     # trend strength minimum
    ofi_agree_threshold: float = 0.6  # centered-flow agreement minimum (0-1)

    # Z threshold adjustments in momentum mode
    z_continuation_mult: float = 0.70  # lower Z for continuation trades
    z_reversion_mult: float = 1.50     # raise Z for counter-trend (harder to take MR)

    # Kelly multiplier in momentum mode
    kelly_mult_mom: float = 0.80    # slightly conservative until momentum proven

    # ADX computation
    adx_window: int = 30            # ticks for ADX computation
    adx_smoothing: int = 14         # EMA smoothing for ADX

    # Exit thresholds
    z_mom_off: float = 0.9          # |z| drops below this → exit momentum
    adx_exit: float = 18.0          # ADX drops below this → exit momentum

    # Transition timeout
    max_transition_cycles: int = 8  # force exit TRANSITION after this many consecutive cycles


class ADXCalculator:
    """
    Simplified ADX (Average Directional Index) for 1Hz tick data.
    Uses price deltas rather than high/low/close candles.
    """
    def __init__(self, period: int = 14):
        self.period = period
        self._prev_price: Optional[float] = None
        self._up_moves: deque = deque(maxlen=period * 2)
        self._dn_moves: deque = deque(maxlen=period * 2)
        self._dx_hist: deque = deque(maxlen=period)
        self.adx: float = 0.0
        self._plus_di: float = 0.0
        self._minus_di: float = 0.0

    def update(self, price: float) -> float:
        """Feed a price, return current ADX estimate."""
        if self._prev_price is None:
            self._prev_price = price
            return 0.0

        delta = price - self._prev_price
        self._prev_price = price

        up_move = max(0.0, delta)
        dn_move = max(0.0, -delta)

        self._up_moves.append(up_move)
        self._dn_moves.append(dn_move)

        if len(self._up_moves) < self.period:
            return 0.0

        # Smoothed directional indicators
        recent_up = list(self._up_moves)[-self.period:]
        recent_dn = list(self._dn_moves)[-self.period:]
        sum_up = sum(recent_up)
        sum_dn = sum(recent_dn)

        # True range proxy (sum of absolute moves)
        tr_sum = sum_up + sum_dn
        if tr_sum < 1e-12:
            return self.adx

        self._plus_di = 100.0 * sum_up / tr_sum
        self._minus_di = 100.0 * sum_dn / tr_sum

        di_sum = self._plus_di + self._minus_di
        if di_sum < 1e-9:
            dx = 0.0
        else:
            dx = 100.0 * abs(self._plus_di - self._minus_di) / di_sum

        self._dx_hist.append(dx)

        if len(self._dx_hist) >= 2:
            # EMA of DX
            alpha = 2.0 / (self.period + 1)
            self.adx = alpha * dx + (1.0 - alpha) * self.adx
        else:
            self.adx = dx

        return self.adx


class MomentumEngine:
    """
    Detects strong trend regime and provides regime arbitration.

    Call update() each tick with z, OFI, and price.
    Check .regime for current mode: 'MR', 'MOM', or 'TRANSITION'.
    """
    def __init__(self, cfg: Optional[MomentumConfig] = None):
        self.cfg = cfg or MomentumConfig()
        self.adx_calc = ADXCalculator(period=self.cfg.adx_smoothing)

        # Persistence tracking
        self._z_extreme_since_ms: Optional[int] = None
        self._z_extreme_dir: int = 0  # +1 or -1
        self._last_ts_ms: int = 0

        # State
        self.regime: str = "MR"         # 'MR', 'MOM', 'TRANSITION'
        self.mom_direction: int = 0     # +1 UP, -1 DOWN, 0 neutral
        self.adx: float = 0.0
        self.ofi_agreement: float = 0.0

        self._tick_count: int = 0
        self._transition_count: int = 0          # consecutive TRANSITION cycles
        self._last_non_transition: str = "MR"    # fallback when TRANSITION times out

    def update(
        self,
        *,
        ts_ms: int,
        z: float,
        ofi: float,            # centered flow imbalance [-1, 1], >0 = buy pressure
        price: float,          # oracle price for ADX
        jump_regime: bool = False,  # from bipower filter
    ) -> str:
        """
        Update state and return current regime: 'MR', 'MOM', or 'TRANSITION'.
        """
        self._tick_count += 1
        self._last_ts_ms = int(ts_ms)

        # Update ADX
        self.adx = self.adx_calc.update(price)

        z_abs = abs(float(z))
        z_dir = 1 if z > 0 else -1

        # Check OFI agreement with z direction (centered scale: [-1, 1]).
        ofi_centered = max(-1.0, min(1.0, float(ofi)))
        if ofi_centered > 0:
            ofi_dir = 1
        elif ofi_centered < 0:
            ofi_dir = -1
        else:
            ofi_dir = 0
        self.ofi_agreement = abs(ofi_centered)  # already normalized to [0, 1]
        ofi_agrees = (
            ofi_dir != 0
            and (ofi_dir == z_dir)
            and (self.ofi_agreement >= self.cfg.ofi_agree_threshold)
        )

        # Track z persistence
        if z_abs >= self.cfg.z_mom_on:
            if self._z_extreme_since_ms is None or z_dir != self._z_extreme_dir:
                self._z_extreme_since_ms = ts_ms
                self._z_extreme_dir = z_dir
        else:
            if z_abs < self.cfg.z_mom_off:
                self._z_extreme_since_ms = None
                self._z_extreme_dir = 0

        # Compute persistence duration
        persist_s = 0.0
        if self._z_extreme_since_ms is not None:
            persist_s = (ts_ms - self._z_extreme_since_ms) / 1000.0

        # Regime decision
        if jump_regime:
            # Bipower handles this — just note we're in transition
            self.regime = "TRANSITION"
            self.mom_direction = 0
        elif (z_abs >= self.cfg.z_mom_on
              and persist_s >= self.cfg.z_mom_persist_s
              and ofi_agrees
              and self.adx >= self.cfg.adx_threshold):
            self.regime = "MOM"
            self.mom_direction = z_dir
        elif self.regime == "MOM":
            # Check exit conditions
            if (z_abs < self.cfg.z_mom_off
                    or self.adx < self.cfg.adx_exit
                    or not ofi_agrees):
                self.regime = "TRANSITION"
                self.mom_direction = 0
        else:
            self.regime = "MR"
            self.mom_direction = 0

        # Transition → MR after 1 tick (when z has cooled)
        if self.regime == "TRANSITION" and not jump_regime:
            if z_abs < self.cfg.z_mom_off:
                self.regime = "MR"

        # Track consecutive TRANSITION cycles and enforce timeout
        if self.regime == "TRANSITION":
            self._transition_count += 1
            if self._transition_count >= self.cfg.max_transition_cycles:
                self.regime = self._last_non_transition
                self.mom_direction = 0
                self._transition_count = 0
        else:
            self._last_non_transition = self.regime
            self._transition_count = 0

        return self.regime

    def z_multiplier(self, is_continuation: bool) -> float:
        """
        Z threshold multiplier based on regime.
        - In MOM mode: continuation trades get lower Z, reversion gets higher Z.
        - In MR mode: no change (1.0).
        """
        if self.regime == "MOM":
            if is_continuation:
                return self.cfg.z_continuation_mult
            else:
                return self.cfg.z_reversion_mult
        return 1.0

    @property
    def kelly_multiplier(self) -> float:
        if self.regime == "MOM":
            return self.cfg.kelly_mult_mom
        return 1.0

    def status_dict(self) -> dict:
        if self._z_extreme_since_ms is None:
            z_persist_ms = 0
        else:
            z_persist_ms = max(0, int(self._last_ts_ms) - int(self._z_extreme_since_ms))
        return {
            "mom_regime": self.regime,
            "mom_dir": self.mom_direction,
            "adx": round(self.adx, 2),
            "ofi_agree": round(self.ofi_agreement, 3),
            "mom_kelly": round(self.kelly_multiplier, 3),
            "z_persist_ms": int(z_persist_ms),
        }
