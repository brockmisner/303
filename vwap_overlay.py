# vwap_overlay.py — VWAP Deviation Overlay for MR Filtering
"""
Computes Z_VWAP = (P - VWAP) / σ_VWAP.
Only allow MR trades when price deviates significantly from VWAP,
avoiding fading noise inside fair value.

Usage:
    V = VWAPTracker()
    V.feed(price, volume)
    z_vwap = V.z_score
    if abs(z_vwap) < 1.0:
        # inside fair value, skip MR trade
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import math
from typing import Optional


@dataclass
class VWAPConfig:
    # Rolling window for VWAP (ticks)
    window: int = 300        # ~5 min at 1Hz

    # Minimum observations before VWAP is valid
    min_obs: int = 20

    # Z-score threshold for MR filtering
    z_mr_threshold: float = 1.0   # |Z_VWAP| must exceed this for MR trades


class VWAPTracker:
    """
    Rolling VWAP with standard deviation for Z-score computation.

    Z_VWAP = (Price - VWAP) / σ_VWAP

    Positive Z_VWAP → price above VWAP (MR short candidate)
    Negative Z_VWAP → price below VWAP (MR long candidate)
    """

    def __init__(self, cfg: VWAPConfig = None):
        self.cfg = cfg or VWAPConfig()
        self._prices: deque = deque(maxlen=self.cfg.window)
        self._volumes: deque = deque(maxlen=self.cfg.window)
        self._pv: deque = deque(maxlen=self.cfg.window)  # price × volume

        self.vwap: float = 0.0
        self.sigma_vwap: float = 0.0
        self.z_score: float = 0.0
        self._valid: bool = False

    def feed(self, price: float, volume: float = 1.0):
        """
        Feed a price (and optional volume). Volume defaults to 1.0
        for tick-weighted VWAP if volume data isn't available.
        """
        price = float(price)
        volume = max(0.001, float(volume))

        self._prices.append(price)
        self._volumes.append(volume)
        self._pv.append(price * volume)

        n = len(self._prices)
        if n < self.cfg.min_obs:
            self._valid = False
            return

        self._valid = True

        # VWAP = Σ(P×V) / Σ(V)
        total_pv = sum(self._pv)
        total_v = sum(self._volumes)
        self.vwap = total_pv / max(1e-12, total_v)

        # VWAP standard deviation (volume-weighted)
        var_sum = 0.0
        for i in range(n):
            diff = self._prices[i] - self.vwap
            var_sum += self._volumes[i] * diff * diff
        variance = var_sum / max(1e-12, total_v)
        self.sigma_vwap = math.sqrt(max(1e-12, variance))

        # Z-score
        if self.sigma_vwap > 1e-9:
            self.z_score = (price - self.vwap) / self.sigma_vwap
        else:
            self.z_score = 0.0

    @property
    def valid(self) -> bool:
        return self._valid

    def mr_allowed(self) -> bool:
        """
        Returns True if price is sufficiently far from VWAP
        to justify a mean-reversion trade.
        """
        if not self._valid:
            return True  # no data → don't filter
        return abs(self.z_score) >= self.cfg.z_mr_threshold

    def status_dict(self) -> dict:
        return {
            "vwap": round(self.vwap, 2),
            "z_vwap": round(self.z_score, 3),
            "sigma_vwap": round(self.sigma_vwap, 4),
            "vwap_mr_ok": self.mr_allowed(),
        }
