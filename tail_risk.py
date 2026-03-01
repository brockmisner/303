"""
tail_risk.py — Tail Risk Defense Guards for 5-Minute Binary Markets

Implements 6 defense layers from TAIL_RISK.md:
  1. Jump detector with cooldown
  2. Oracle shock guard
  3. Spread explosion filter
  4. Flip avalanche guard
  5. Kelly tail cap
  6. Volatility kill switch

Each guard returns a TailRiskAction indicating whether trading should proceed.

Reference: agent_instructions/TAIL_RISK.md
"""

from __future__ import annotations
import time
import math
from dataclasses import dataclass
from collections import deque
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Action returned by guards
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TailRiskAction:
    """Result of a tail-risk check."""
    allow_trade: bool = True
    reason: str = ""
    kelly_cap: float = 1.0     # multiplier to further reduce Kelly (0..1)
    maker_only: bool = False   # force maker routing


# ─────────────────────────────────────────────────────────────────────────────
# Tail Risk Guard
# ─────────────────────────────────────────────────────────────────────────────

class TailRiskGuard:
    """
    Aggregates all tail-risk defense layers.

    Call check() every tick before deciding to trade.
    Call feed_price() on every BTC price update.
    Call feed_oracle() on every oracle update.
    """

    def __init__(
        self,
        # Jump detector
        jump_sigma_mult: float = 5.0,      # was 3.0 — too sensitive at 100Hz
        jump_cooldown_s: float = 1.5,      # was 5.0 — 5-min markets need fast recovery
        # Oracle shock
        oracle_shock_pct: float = 0.15,    # 0.15% = ~$150 at $100k
        oracle_freeze_s: float = 5.0,
        # Spread explosion
        spread_explosion_mult: float = 2.0,
        # Flip avalanche
        flip_avalanche_rate: float = 0.15,
        # Vol kill switch
        vol_kill_mult: float = 2.0,
        # Kelly tail cap
        kelly_tail_gamma: float = 0.35,
        max_risk_pct: float = 0.02,        # 2% of bankroll per trade
    ):
        # Config
        self._jump_mult = jump_sigma_mult
        self._jump_cooldown_ms = int(jump_cooldown_s * 1000)
        self._oracle_shock_pct = oracle_shock_pct
        self._oracle_freeze_ms = int(oracle_freeze_s * 1000)
        self._spread_mult = spread_explosion_mult
        self._flip_avalanche_rate = flip_avalanche_rate
        self._vol_kill_mult = vol_kill_mult
        self._kelly_gamma = kelly_tail_gamma
        self._max_risk_pct = max_risk_pct

        # State — jump detector
        self._price_buf: deque = deque(maxlen=500)
        self._jump_cooldown_until: int = 0
        self._jump_timestamps: deque = deque(maxlen=20)  # cascade tracker
        self._cascade_freeze_ms: int = 5_000              # was 10s → 5s for 5-min markets
        self._cascade_window_ms: int = 20_000

        # State — oracle shock
        self._last_oracle_price: float = 0.0
        self._oracle_freeze_until: int = 0

        # State — spread tracker
        self._spread_buf: deque = deque(maxlen=200)

        # State — vol kill
        self._sigma_buf: deque = deque(maxlen=300)

    # ── Price/Oracle feeds ───────────────────────────────────────────────

    def feed_price(self, price: float, ts_ms: int) -> None:
        """Feed every BTC price tick. Detects jumps and cascades."""
        self._price_buf.append((ts_ms, price))

        # Jump detection: |r| > 3 × rolling_σ
        if len(self._price_buf) >= 10:
            log_returns = []
            buf = list(self._price_buf)
            for i in range(1, min(50, len(buf))):
                if buf[i][1] > 0 and buf[i-1][1] > 0:
                    log_returns.append(math.log(buf[i][1] / buf[i-1][1]))
            if len(log_returns) >= 5:
                mean_sq = sum(r*r for r in log_returns) / len(log_returns)
                rolling_sigma = math.sqrt(mean_sq) if mean_sq > 0 else 1e-9
                latest_r = log_returns[-1] if log_returns else 0
                if abs(latest_r) > self._jump_mult * rolling_sigma:
                    # Record jump timestamp
                    self._jump_timestamps.append(ts_ms)

                    # Check for cascade: 2+ jumps within 20s window
                    cutoff = ts_ms - self._cascade_window_ms
                    recent_jumps = sum(1 for t in self._jump_timestamps if t >= cutoff)

                    new_until = ts_ms + (self._cascade_freeze_ms if recent_jumps >= 3 else self._jump_cooldown_ms)
                    self._jump_cooldown_until = max(self._jump_cooldown_until, new_until)

    def feed_oracle(self, price: float, ts_ms: int) -> None:
        """Feed every oracle price update. Detects shocks."""
        if self._last_oracle_price > 0:
            pct_change = abs(price - self._last_oracle_price) / self._last_oracle_price * 100
            if pct_change > self._oracle_shock_pct:
                self._oracle_freeze_until = ts_ms + self._oracle_freeze_ms
        self._last_oracle_price = price

    def feed_spread(self, spread: float) -> None:
        """Feed current best spread (max of up/down)."""
        self._spread_buf.append(spread)

    def feed_sigma(self, sigma_eff: float) -> None:
        """Feed current σ_eff for vol kill switch."""
        if sigma_eff > 0:
            self._sigma_buf.append(sigma_eff)

    # ── Main check ───────────────────────────────────────────────────────

    def check(
        self,
        ts_ms: int,
        flip_rate: float = 0.0,
        sigma_eff: float = 0.0,
        spread: float = 0.0,
    ) -> TailRiskAction:
        """
        Run all defense layers. Returns TailRiskAction.

        Call this before every trade decision. If allow_trade is False,
        do not enter any new position.
        """
        # Layer 1: Jump cooldown
        if ts_ms < self._jump_cooldown_until:
            remaining = (self._jump_cooldown_until - ts_ms) / 1000
            return TailRiskAction(
                allow_trade=False,
                reason=f"JUMP_COOLDOWN({remaining:.1f}s)",
            )

        # Layer 2: Oracle shock freeze
        if ts_ms < self._oracle_freeze_until:
            remaining = (self._oracle_freeze_until - ts_ms) / 1000
            return TailRiskAction(
                allow_trade=False,
                reason=f"ORACLE_SHOCK({remaining:.1f}s)",
            )

        # Layer 3: Spread explosion
        if len(self._spread_buf) >= 20:
            sorted_spreads = sorted(self._spread_buf)
            median_spread = sorted_spreads[len(sorted_spreads) // 2]
            if spread > self._spread_mult * max(median_spread, 0.005):
                return TailRiskAction(
                    allow_trade=True,  # allow but force maker only
                    reason="SPREAD_EXPLOSION",
                    maker_only=True,
                    kelly_cap=0.5,
                )

        # Layer 4: Flip avalanche
        if flip_rate > self._flip_avalanche_rate:
            return TailRiskAction(
                allow_trade=False,
                reason=f"FLIP_AVALANCHE({flip_rate:.2f})",
            )

        # Layer 5: Vol kill switch
        if len(self._sigma_buf) >= 30:
            sorted_sigma = sorted(self._sigma_buf)
            sigma_median = sorted_sigma[len(sorted_sigma) // 2]
            if sigma_eff > self._vol_kill_mult * sigma_median:
                return TailRiskAction(
                    allow_trade=False,
                    reason=f"VOL_KILL({sigma_eff:.6f}>{self._vol_kill_mult}×{sigma_median:.6f})",
                )

        # Layer 6: All layers passed — normal conditions, no tail cap
        kelly_cap = 1.0  # no reduction when all layers pass

        return TailRiskAction(
            allow_trade=True,
            reason="",
            kelly_cap=kelly_cap,
        )

    # ── Sizing helper ────────────────────────────────────────────────────

    def clamp_size(self, size: float, bankroll: float) -> float:
        """Hard cap: never risk more than max_risk_pct of bankroll."""
        max_size = bankroll * self._max_risk_pct
        return min(size, max_size) if bankroll > 0 else size
