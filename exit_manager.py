# exit_manager.py — EV-Exit Manager for locked-in alpha extraction
"""
Instead of holding binary positions to settlement (0 or 1 outcome),
sells back into the bid when the market corrects to near fair value,
locking in guaranteed profit and bypassing end-of-window gamma risk.

Scenario: Bot buys UP token at 0.15 when p_cone=0.26 (11% edge).
  - Hold EV: 0.26 (26% chance of $1, 74% chance of $0)
  - 5 seconds later, bid recovers to 0.24
  - Sell EV: 0.24 * (1 - fee) = 0.238
  - With variance penalty: sell if sell_ev >= hold_ev - penalty
  - Locks in ~0.09/share guaranteed vs gambling on 26% hit rate

Usage:
    EXIT_MGR = ExitManager()
    decision = EXIT_MGR.evaluate(side, entry_price, current_bid, p_cone, T_sec, sigma_1m)
    if decision.action == "SELL":
        # queue IOC sell order
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import logging

logger = logging.getLogger("exit_manager")

FEE_RATE = 0.0625  # Polymarket: f = p(1-p) * r


def _fee(price: float) -> float:
    return price * (1.0 - price) * FEE_RATE


@dataclass
class ExitConfig:
    # Minimum profit (entry → bid) to even consider an exit
    min_profit_per_share: float = 0.02  # 2 cents minimum locked profit

    # Variance penalty: willing to accept this much haircut vs hold_ev
    # to eliminate binary settlement risk. Scales with gamma.
    variance_base: float = 0.02         # base penalty (always applied)
    variance_gamma_k: float = 0.06      # extra penalty near expiry + ATM

    # Time thresholds
    T_aggressive: float = 45.0          # below this, increase variance penalty
    T_very_aggressive: float = 20.0     # below this, maximize variance penalty

    # Probability band where exit is most valuable (uncertain zone)
    p_uncertain_lo: float = 0.12
    p_uncertain_hi: float = 0.88

    # Throttle: minimum ms between exit evaluations
    throttle_ms: int = 200

    # Don't exit if bid is clearly stale/broken
    min_bid: float = 0.02


@dataclass
class ExitDecision:
    action: str       # "SELL" or "HOLD"
    reason: str
    hold_ev: float
    sell_ev: float
    profit_per_share: float  # vs entry price
    variance_penalty: float


class ExitManager:

    def __init__(self, cfg: ExitConfig = None):
        self.cfg = cfg or ExitConfig()
        self._last_eval_ms: int = 0

    def should_throttle(self, now_ms: int) -> bool:
        if now_ms - self._last_eval_ms < self.cfg.throttle_ms:
            return True
        return False

    def evaluate(
        self,
        side: str,           # "UP" or "DOWN"
        entry_price: float,  # avg cost basis
        current_bid: float,  # live bid for this token
        p_cone: float,       # model probability of UP winning
        T_sec: float,        # seconds remaining
        sigma_1m: float,     # 1-min realized vol
        now_ms: int = 0,
    ) -> ExitDecision:
        """
        Core EV-exit decision: sell to bid vs hold to settlement.
        """
        cfg = self.cfg
        if now_ms > 0:
            self._last_eval_ms = now_ms

        # Token-specific probability
        p_token = p_cone if side == "UP" else (1.0 - p_cone)

        # Hold EV = probability this token settles at $1.00
        hold_ev = p_token

        # Sell EV = bid net of fees
        if current_bid < cfg.min_bid:
            return ExitDecision("HOLD", "bid_too_low", hold_ev, 0.0, 0.0, 0.0)

        price = min(0.99, max(0.01, current_bid))
        sell_ev = price - _fee(price)

        # Profit vs entry (must clear minimum to avoid churning)
        profit = sell_ev - entry_price
        if profit < cfg.min_profit_per_share:
            return ExitDecision("HOLD", "insufficient_profit", hold_ev, sell_ev, profit, 0.0)

        # Variance penalty: how much we're willing to leave on the table
        # to avoid binary settlement risk
        penalty = cfg.variance_base

        # Gamma scaling: penalty increases when near ATM and near expiry
        # p near 0.5 = max uncertainty, p near 0/1 = low uncertainty
        uncertainty = 4.0 * p_token * (1.0 - p_token)  # 0..1

        if T_sec < cfg.T_aggressive:
            # Approaching expiry: gamma risk is high
            time_urgency = 1.0 + (cfg.T_aggressive - T_sec) / cfg.T_aggressive
            penalty += cfg.variance_gamma_k * uncertainty * time_urgency
        if T_sec < cfg.T_very_aggressive:
            # Very close to expiry: maximize exit preference
            penalty += cfg.variance_gamma_k * 0.5

        # In the uncertain zone (p between 0.12 and 0.88), variance is real
        # Outside this zone (p > 0.88 or p < 0.12), settlement is near-certain
        if not (cfg.p_uncertain_lo < p_token < cfg.p_uncertain_hi):
            # Near-certain outcome: reduce penalty (let settlement play out)
            penalty *= 0.25

        # The exit rule: sell if guaranteed profit >= hold_ev minus penalty
        if sell_ev >= (hold_ev - penalty):
            return ExitDecision(
                "SELL", "ev_exit_lock_profit",
                hold_ev, sell_ev, profit, penalty,
            )

        return ExitDecision("HOLD", "hold_ev_superior", hold_ev, sell_ev, profit, penalty)
