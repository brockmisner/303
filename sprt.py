# sprt.py — Sequential Probability Ratio Test for Live Edge Validation
"""
Brier-score SPRT: tests whether our cone probability model predicts
outcomes better than the market midpoint.

Old approach (broken):
  Binary win/loss test: H0 win_rate=0.50 vs H1 win_rate=0.52.
  A bot buying at $0.15 with 20% win rate is hugely profitable
  but SPRT throttled Kelly because 20% < 50%.

New approach (correct):
  Compare our model's Brier score to the market's Brier score.
  Brier = (predicted_p - actual)^2.  Lower = better.
  If our Brier < market's Brier → we add value (H1).
  If our Brier >= market's Brier → no edge (H0).

Usage:
    S = SPRTValidator()
    S.record(p_model=0.26, p_market=0.15, outcome=True)
    kelly_mult = S.kelly_multiplier
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from collections import deque


@dataclass
class SPRTConfig:
    # Wald boundaries (log-likelihood)
    alpha: float = 0.05      # type I error -> upper bound
    beta: float = 0.10       # type II error -> lower bound

    # Kelly scaling
    kelly_floor: float = 0.30   # minimum Kelly multiplier when H0 accepted
    kelly_full: float = 1.0     # full Kelly when H1 accepted
    kelly_neutral: float = 0.70 # between boundaries

    # Rolling window for soft reset
    reset_window: int = 50

    # Brier epsilon: minimum meaningful difference
    brier_eps: float = 1e-9


@dataclass
class _Observation:
    """Single trade outcome for Brier SPRT."""
    p_model: float   # our cone probability at entry
    p_market: float  # market midpoint at entry
    outcome: float   # 1.0 if this side won, 0.0 if lost


class SPRTValidator:
    """
    Brier-score Sequential Probability Ratio Test.

    Compares our model's prediction accuracy (Brier score) against
    the market's prediction accuracy. If we're consistently more
    accurate → H1 (edge is real) → full Kelly.

    The log-likelihood ratio for each observation:
        brier_market = (p_market - actual)^2
        brier_model  = (p_model  - actual)^2
        lr_inc = log(brier_market / brier_model)

    Positive lr_inc → our model was more accurate on this trade.
    """

    def __init__(self, cfg: SPRTConfig = None):
        self.cfg = cfg or SPRTConfig()

        # Wald boundaries
        self._upper = math.log((1.0 - self.cfg.beta) / self.cfg.alpha)
        self._lower = math.log(self.cfg.beta / (1.0 - self.cfg.alpha))

        # State
        self._log_lr: float = 0.0   # cumulative log-likelihood ratio
        self._n: int = 0
        self._wins: int = 0
        self._observations: deque[_Observation] = deque(maxlen=self.cfg.reset_window)

        # Brier accumulators (for diagnostics)
        self._brier_model_sum: float = 0.0
        self._brier_market_sum: float = 0.0

        # Decision
        self._decision: str = "NEUTRAL"  # "H1", "H0", "NEUTRAL"

    def record(self, p_model: float, p_market: float, outcome: bool):
        """
        Record a trade outcome with probability context.

        Args:
            p_model:  our cone probability for this side at entry time
            p_market: market midpoint for this side at entry time
            outcome:  True if this side won at settlement
        """
        self._n += 1
        actual = 1.0 if outcome else 0.0
        if outcome:
            self._wins += 1

        obs = _Observation(p_model=p_model, p_market=p_market, outcome=actual)
        self._observations.append(obs)

        # Brier scores: lower = better calibration
        brier_model = (p_model - actual) ** 2
        brier_market = (p_market - actual) ** 2

        self._brier_model_sum += brier_model
        self._brier_market_sum += brier_market

        # Log-likelihood ratio update
        # Positive when model is more accurate (brier_market > brier_model)
        eps = self.cfg.brier_eps
        lr_inc = math.log(max(eps, brier_market) / max(eps, brier_model))

        # Clamp extreme individual updates to prevent single-trade domination
        lr_inc = max(-2.0, min(2.0, lr_inc))

        self._log_lr += lr_inc

        # Decision
        self._update_decision()

        # Soft reset on window boundary (rolling recalibration)
        if (len(self._observations) >= self.cfg.reset_window
                and self._n % self.cfg.reset_window == 0):
            self._soft_reset()

    def _update_decision(self):
        """Update H0/H1/NEUTRAL based on current log-LR."""
        if self._log_lr >= self._upper:
            self._decision = "H1"  # model has edge
        elif self._log_lr <= self._lower:
            self._decision = "H0"  # no edge
        else:
            self._decision = "NEUTRAL"

    def _soft_reset(self):
        """Recalculate log-LR from recent window observations."""
        eps = self.cfg.brier_eps
        self._log_lr = 0.0
        self._brier_model_sum = 0.0
        self._brier_market_sum = 0.0

        for obs in self._observations:
            bm = (obs.p_model - obs.outcome) ** 2
            bmkt = (obs.p_market - obs.outcome) ** 2
            self._brier_model_sum += bm
            self._brier_market_sum += bmkt
            lr_inc = math.log(max(eps, bmkt) / max(eps, bm))
            lr_inc = max(-2.0, min(2.0, lr_inc))
            self._log_lr += lr_inc

        self._update_decision()

    # ── Legacy compatibility: accept old-style record(won=True) ──
    def record_legacy(self, won: bool):
        """Backward-compatible record. Uses default p values."""
        self.record(p_model=0.52 if won else 0.48,
                    p_market=0.50, outcome=won)

    @property
    def kelly_multiplier(self) -> float:
        if self._decision == "H1":
            return self.cfg.kelly_full
        elif self._decision == "H0":
            return self.cfg.kelly_floor
        else:
            # Interpolate based on position between boundaries
            if self._upper - self._lower < 1e-9:
                return self.cfg.kelly_neutral
            frac = (self._log_lr - self._lower) / (self._upper - self._lower)
            frac = max(0.0, min(1.0, frac))
            return self.cfg.kelly_floor + frac * (self.cfg.kelly_full - self.cfg.kelly_floor)

    @property
    def win_rate(self) -> float:
        if not self._observations:
            return 0.5
        return sum(1 for o in self._observations if o.outcome > 0.5) / len(self._observations)

    @property
    def avg_brier_model(self) -> float:
        if self._n == 0:
            return 0.25  # naive baseline
        return self._brier_model_sum / self._n

    @property
    def avg_brier_market(self) -> float:
        if self._n == 0:
            return 0.25
        return self._brier_market_sum / self._n

    @property
    def decision(self) -> str:
        return self._decision

    def status_dict(self) -> dict:
        return {
            "sprt_decision": self._decision,
            "sprt_log_lr": round(self._log_lr, 4),
            "sprt_n": self._n,
            "sprt_win_rate": round(self.win_rate, 4),
            "sprt_kelly": round(self.kelly_multiplier, 3),
            "sprt_brier_model": round(self.avg_brier_model, 4),
            "sprt_brier_market": round(self.avg_brier_market, 4),
        }
