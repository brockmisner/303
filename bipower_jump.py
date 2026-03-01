# bipower_jump.py — Realized Bipower Variation Jump Filter
"""
Detects structural jumps by comparing Realized Variance (RV) to
Bipower Variation (BV).  When J_t = (RV - BV) / RV exceeds threshold,
the continuous diffusion assumption is violated → mean-reversion
should be suppressed and Kelly reduced.

Usage:
    JF = BipowerJumpFilter()
    JF.feed(log_return)          # call with each tick's log-return
    if JF.jump_regime:
        # suppress MR, enable momentum, cut Kelly
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import math
from typing import Optional


@dataclass
class BipowerConfig:
    # Rolling window for RV/BV (number of returns)
    window: int = 120           # ~2 min at 1 Hz tick rate

    # Jump statistic threshold (J_t > this → jump regime)
    j_threshold: float = 0.40   # 40% of variance unexplained by diffusion

    # Hysteresis: once in jump regime, need J_t below this to exit
    j_exit: float = 0.20

    # Minimum observations before computing (avoid noisy early estimates)
    min_obs: int = 30

    # Kelly multiplier when jump regime active
    kelly_mult_jump: float = 0.35   # 65% reduction

    # Cooldown ticks after jump regime ends before full Kelly restored
    cooldown_ticks: int = 30

    # Smoothing: EMA alpha for jump statistic (reduces chatter)
    ema_alpha: float = 0.15


class BipowerJumpFilter:
    """
    Proactive jump detector using Realized Variance vs Bipower Variation.

    BV_t = (π/2) × Σ|r_i| × |r_{i-1}|    (consecutive absolute returns)
    RV_t = Σ r_i²                           (standard realized variance)
    J_t  = (RV_t - BV_t) / RV_t            (jump fraction)

    When J_t > threshold → jump regime active.
    """

    def __init__(self, cfg: Optional[BipowerConfig] = None):
        self.cfg = cfg or BipowerConfig()
        self._returns: deque = deque(maxlen=self.cfg.window)

        # State
        self.rv: float = 0.0
        self.bv: float = 0.0
        self.j_stat: float = 0.0
        self.j_ema: float = 0.0
        self.jump_regime: bool = False

        self._cooldown: int = 0
        self._tick_count: int = 0

    def feed(self, r_t: float) -> bool:
        """
        Feed a log-return.  Returns True if jump regime is active.
        """
        self._returns.append(float(r_t))
        self._tick_count += 1

        if len(self._returns) < self.cfg.min_obs:
            return False

        # Compute RV and BV over window
        returns = list(self._returns)
        n = len(returns)

        rv = sum(r * r for r in returns)
        bv = 0.0
        if n >= 2:
            bv = (math.pi / 2.0) * sum(
                abs(returns[i]) * abs(returns[i - 1])
                for i in range(1, n)
            )

        self.rv = rv
        self.bv = bv

        # Jump statistic
        if rv > 1e-15:
            j_raw = max(0.0, (rv - bv) / rv)
        else:
            j_raw = 0.0

        self.j_stat = j_raw

        # EMA smoothing
        alpha = self.cfg.ema_alpha
        self.j_ema = alpha * j_raw + (1.0 - alpha) * self.j_ema

        # Hysteresis regime detection
        if not self.jump_regime:
            if self.j_ema >= self.cfg.j_threshold:
                self.jump_regime = True
                self._cooldown = 0
        else:
            if self.j_ema < self.cfg.j_exit:
                self.jump_regime = False
                self._cooldown = self.cfg.cooldown_ticks

        # Cooldown decay
        if self._cooldown > 0:
            self._cooldown -= 1

        return self.jump_regime

    @property
    def kelly_multiplier(self) -> float:
        """
        Kelly multiplier: reduced during jump regime and cooldown.
        Returns value in [kelly_mult_jump, 1.0].
        """
        if self.jump_regime:
            return self.cfg.kelly_mult_jump
        if self._cooldown > 0:
            # Linear ramp from kelly_mult_jump back to 1.0 over cooldown
            frac = self._cooldown / max(1, self.cfg.cooldown_ticks)
            return self.cfg.kelly_mult_jump + (1.0 - self.cfg.kelly_mult_jump) * (1.0 - frac)
        return 1.0

    @property
    def in_cooldown(self) -> bool:
        return self._cooldown > 0

    def status_dict(self) -> dict:
        """Debug snapshot for LATEST_DEBUG / telemetry."""
        return {
            "rv": round(self.rv, 8),
            "bv": round(self.bv, 8),
            "j_stat": round(self.j_stat, 4),
            "j_ema": round(self.j_ema, 4),
            "jump_regime": self.jump_regime,
            "jump_kelly": round(self.kelly_multiplier, 3),
            "jump_cooldown": self._cooldown,
        }
