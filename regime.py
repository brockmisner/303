"""
regime.py — Volatility Regime Detection & Dynamic Parameter Control

Classifies market microstructure into four regimes and adjusts
trading parameters (min_ev, Kelly multiplier, z threshold) accordingly.

Regimes
-------
  CALM         σ_ratio < 0.7, flip_rate < 0.03, jump_intensity < 0.3
  NORMAL       0.7 ≤ σ_ratio ≤ 1.3, flip_rate < 0.08
  HIGH_VOL     σ_ratio > 1.3 OR jump_intensity > 0.5
  ADVERSARIAL  flip_rate > 0.10 OR sign flips > 3 in 20s

Persistence filter: regime only changes after ≥ PERSIST_TICKS consecutive
identical classifications, preventing oscillation.

Reference: agent_instructions/REGIME.md
"""

from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
from typing import Optional
import math
import time
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Regime Labels
# ─────────────────────────────────────────────────────────────────────────────

CALM        = "CALM"
NORMAL      = "NORMAL"
TRANSITION  = "TRANSITION"
VOL_EVENT   = "VOL_EVENT"
HIGH_VOL    = "HIGH_VOL"
ADVERSARIAL = "ADVERSARIAL"


# ─────────────────────────────────────────────────────────────────────────────
# Regime Parameters — what changes per regime
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RegimeParams:
    """Frozen parameter set for a given regime."""
    label:            str
    kelly_multiplier: float   # scale fractional Kelly
    min_ev_multiplier: float  # scale base min_ev
    min_z:            float   # minimum |z| to trade
    calibration_alpha: float  # probability shrinkage α (P = α·Φ(z) + (1-α)·0.5)
    size_multiplier:  float   # execution size haircut
    drift_enabled:    bool = True  # whether drift trading is allowed


REGIME_PARAMS = {
    CALM:        RegimeParams(CALM,        kelly_multiplier=1.2, min_ev_multiplier=0.8, min_z=0.55, calibration_alpha=0.95, size_multiplier=1.0, drift_enabled=True),
    NORMAL:      RegimeParams(NORMAL,      kelly_multiplier=1.0, min_ev_multiplier=1.0, min_z=0.65, calibration_alpha=0.90, size_multiplier=1.0, drift_enabled=True),
    TRANSITION:  RegimeParams(TRANSITION,  kelly_multiplier=0.9, min_ev_multiplier=1.0, min_z=0.75, calibration_alpha=0.90, size_multiplier=0.8, drift_enabled=False),
    VOL_EVENT:   RegimeParams(VOL_EVENT,   kelly_multiplier=0.7, min_ev_multiplier=1.2, min_z=0.85, calibration_alpha=0.92, size_multiplier=0.7, drift_enabled=False),
    HIGH_VOL:    RegimeParams(HIGH_VOL,    kelly_multiplier=0.7, min_ev_multiplier=1.25, min_z=0.95, calibration_alpha=0.85, size_multiplier=0.7, drift_enabled=False),
    ADVERSARIAL: RegimeParams(ADVERSARIAL, kelly_multiplier=0.5, min_ev_multiplier=2.0,  min_z=1.20, calibration_alpha=0.75, size_multiplier=0.5, drift_enabled=False),
}


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic min_ev formula
# ─────────────────────────────────────────────────────────────────────────────

def dynamic_min_ev(
    base_ev: float = 0.02,
    sigma_ratio: float = 1.0,
    flip_rate: float = 0.0,
    alpha: float = 0.5,
    beta: float = 2.0,
) -> float:
    """
    min_ev = base_ev × (1 + α·σ_ratio + β·flip_rate)

    Higher volatility or flip rate → higher EV bar to trade.
    """
    return base_ev * (1.0 + alpha * sigma_ratio + beta * flip_rate)


# ─────────────────────────────────────────────────────────────────────────────
# Volatility Event Detector
# ─────────────────────────────────────────────────────────────────────────────

class VolatilityEventDetector:
    """Percentile-adaptive volatility event detection.
    Uses rolling vol_ratio history to self-adapt thresholds.
    Exposes event_active (bool) and intensity (continuous 0→∞)."""

    def __init__(self, window: int = 600):
        self.history: deque = deque(maxlen=window)
        self.event_active: bool = False
        self.intensity: float = 0.0   # continuous: 0=calm, >1=event
        self._vol_ratio: float = 1.0

    def update(self, *, sigma_fast: float, sigma_slow: float, **_kw) -> str:
        """Returns regime label: CALM / TRANSITION / VOL_EVENT."""
        if sigma_slow <= 1e-9:
            return "CALM"

        vol_ratio = sigma_fast / sigma_slow
        self._vol_ratio = vol_ratio
        self.history.append(vol_ratio)

        if len(self.history) < 100:
            # Not enough data — use static fallback
            self.event_active = vol_ratio > 2.0
            self.intensity = max(0.0, vol_ratio - 1.0)
            if self.event_active:
                return "VOL_EVENT"
            return "CALM"

        # Adaptive percentile thresholds
        p50 = float(np.percentile(self.history, 50))
        p92 = float(np.percentile(self.history, 92))
        p97 = float(np.percentile(self.history, 97))

        # Continuous intensity: 0 at median, 1 at 92nd pctile, >1 above
        denom = max(p92 - p50, 0.01)
        self.intensity = max(0.0, (vol_ratio - p50) / denom)

        # State determination
        if vol_ratio > p97:
            self.event_active = True
            return "VOL_EVENT"      # extreme: top 3%
        elif vol_ratio > p92:
            self.event_active = True
            return "TRANSITION"     # elevated: top 8%
        else:
            self.event_active = False
            if vol_ratio < p50 * 0.8:
                return "CALM"
            return "NORMAL"


# ─────────────────────────────────────────────────────────────────────────────
# Jump Detector — Poisson jump-diffusion
# ─────────────────────────────────────────────────────────────────────────────

def _fast_norm_cdf(x: float) -> float:
    """Fast Φ(x) via erfc — avoids scipy import hang."""
    from math import erfc, sqrt
    return 0.5 * erfc(-x / sqrt(2.0))


class JumpDetector:
    """Poisson jump-diffusion detector.
    Flags structural jumps inconsistent with Gaussian diffusion."""

    def __init__(self, dt_seconds: float = 1.0, window: int = 600):
        self.dt = dt_seconds
        self.jump_intensity: float = 0.0  # latest |z| magnitude
        self.recent_jumps: deque = deque(maxlen=window)
        self.lambda_hat: float = 0.0       # empirical jump frequency

    def update(self, r_t: float, sigma_slow: float) -> bool:
        """Feed a return r_t and slow sigma. Returns True if jump detected."""
        if sigma_slow <= 1e-9:
            return False

        z = abs(r_t) / (sigma_slow * math.sqrt(self.dt))
        p_jump = 2.0 * (1.0 - _fast_norm_cdf(z))

        is_jump = p_jump < 0.005  # diffusion violation at 99.5%

        self.recent_jumps.append(int(is_jump))

        if len(self.recent_jumps) >= 50:
            self.lambda_hat = float(np.mean(self.recent_jumps))

        self.jump_intensity = z
        return is_jump


# ─────────────────────────────────────────────────────────────────────────────
# Regime Classifier
# ─────────────────────────────────────────────────────────────────────────────

class RegimeClassifier:
    """
    Classifies market microstructure regime every tick.

    Parameters
    ----------
    sigma_ref : float
        Long-run reference σ_eff (e.g. rolling median over session).
    persist_ticks : int
        Number of consecutive identical raw classifications before
        the regime actually changes (prevents oscillation).
    """

    def __init__(
        self,
        sigma_ref: float = 0.00035,
        persist_ticks: int = 3,
        sigma_ref_window: int = 500,
    ):
        self.sigma_ref = sigma_ref
        self.persist_ticks = persist_ticks

        # Persistence filter state
        self._current_regime: str = NORMAL
        self._candidate: str = NORMAL
        self._candidate_count: int = 0

        # Rolling σ_ref tracker (adaptive)
        self._sigma_buf: deque = deque(maxlen=sigma_ref_window)

        # Flip tracking
        self._sign_history: deque = deque(maxlen=100)
        self._last_sign: Optional[int] = None

        # Volatility event detector
        self.vol_detector = VolatilityEventDetector()

    # ── Public Interface ─────────────────────────────────────────────────

    def update(
        self,
        sigma_eff: float,
        delta_sign: int,          # sgn(S - K): +1 or -1
        sigma_jump: float = 0.0,  # jump component of σ
        z: float = 0.0,           # current z-score
        sigma_fast: float = 0.0,  # fast sigma for vol detection
        sigma_slow: float = 0.0,  # slow sigma for vol detection
        spread: float = 0.0,      # current spread
    ) -> RegimeParams:
        """
        Feed a new tick. Returns the active RegimeParams (possibly unchanged
        due to persistence filter).

        Vol event detector overlays on top of sigma classification:
        VOL_EVENT/TRANSITION override the base regime when active.
        """
        # Track σ for adaptive reference
        if sigma_eff > 0:
            self._sigma_buf.append(sigma_eff)
            if len(self._sigma_buf) >= 50:
                sorted_buf = sorted(self._sigma_buf)
                self.sigma_ref = sorted_buf[len(sorted_buf) // 2]  # median

        # Track sign flips
        if self._last_sign is not None and delta_sign != self._last_sign:
            self._sign_history.append(1)
        else:
            self._sign_history.append(0)
        self._last_sign = delta_sign

        # Compute regime inputs
        sigma_ratio = sigma_eff / max(self.sigma_ref, 1e-9)
        flip_rate = self._recent_flip_rate(window=30)
        flips_20t = sum(list(self._sign_history)[-20:])  # flips in last 20 ticks (not seconds)
        jump_intensity = (sigma_jump / max(sigma_eff, 1e-9)) if sigma_eff > 0 else 0.0

        # Raw classification (priority order: adversarial first)
        raw = self._classify(sigma_ratio, flip_rate, flips_20t, jump_intensity)

        # Volatility event overlay: overrides base regime when active
        vol_state = self.vol_detector.update(
            sigma_fast=sigma_fast if sigma_fast > 0 else sigma_eff,
            sigma_slow=sigma_slow if sigma_slow > 0 else sigma_eff,
        )
        if vol_state in ("VOL_EVENT", "TRANSITION"):
            raw = vol_state

        # Persistence filter
        if raw == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate = raw
            self._candidate_count = 1

        if self._candidate_count >= self.persist_ticks:
            self._current_regime = self._candidate

        return self.current_params

    @property
    def current_params(self) -> RegimeParams:
        return REGIME_PARAMS[self._current_regime]

    @property
    def label(self) -> str:
        return self._current_regime

    @property
    def sigma_ratio(self) -> float:
        """Current σ_eff / σ_ref (last value)."""
        if not self._sigma_buf:
            return 1.0
        return self._sigma_buf[-1] / max(self.sigma_ref, 1e-9)

    @property
    def flip_rate(self) -> float:
        return self._recent_flip_rate(window=30)

    # ── Internals ────────────────────────────────────────────────────────

    def _recent_flip_rate(self, window: int = 30) -> float:
        """Fraction of recent ticks that were sign flips."""
        recent = list(self._sign_history)[-window:]
        if not recent:
            return 0.0
        return sum(recent) / len(recent)

    @staticmethod
    def _classify(
        sigma_ratio: float,
        flip_rate: float,
        flips_20t: int,
        jump_intensity: float,
    ) -> str:
        # Adversarial: high flip rate or many flips in short window
        if flip_rate > 0.10 or flips_20t > 3:
            return ADVERSARIAL

        # High vol: σ expanding or jump-driven
        if sigma_ratio > 1.3 or jump_intensity > 0.5:
            return HIGH_VOL

        # Calm: low everything
        if sigma_ratio < 0.7 and flip_rate < 0.03 and jump_intensity < 0.3:
            return CALM

        return NORMAL
