# endgame_manager.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Side = Literal["UP", "DOWN"]


@dataclass
class EndgameConfig:
    strong_win_threshold: float = 0.85
    strong_loss_threshold: float = 0.15
    min_time_remaining: float = 25.0   # seconds
    fee_buffer: float = 0.003          # cushion vs spread/fees
    dynamic_band_mult: float = 2.0     # multiplier on sigma_T for gamma band
    min_uncertainty_band: float = 0.20 # floor band width


class EndgameManager:

    def __init__(self, config: EndgameConfig):
        self.cfg = config

    def _effective_p(self, side: Side, p_cone: float) -> float:
        return p_cone if side == "UP" else 1.0 - p_cone

    def _dynamic_band(self, sigma_T: float) -> tuple[float, float]:
        band = max(self.cfg.min_uncertainty_band,
                   self.cfg.dynamic_band_mult * sigma_T)
        lower = 0.5 - band
        upper = 0.5 + band
        return max(0.0, lower), min(1.0, upper)

    def decide(
        self,
        side: Side,
        p_cone: float,
        m_bid: float,
        sigma_T: float,
        time_remaining: float,
        is_profitable: bool,
    ) -> str:
        """
        Returns:
            "HOLD" or "SELL"
        """

        if time_remaining > self.cfg.min_time_remaining:
            return "HOLD"

        p = self._effective_p(side, p_cone)

        # 1️⃣ Strong winner
        if p >= self.cfg.strong_win_threshold:
            return "HOLD"

        # 2️⃣ Strong loser
        if p <= self.cfg.strong_loss_threshold:
            return "SELL"

        # 3️⃣ Dynamic gamma uncertainty zone
        lower, upper = self._dynamic_band(sigma_T)

        if lower <= p <= upper:
            # High gamma zone
            if is_profitable:
                return "SELL"
            else:
                return "HOLD"

        # 4️⃣ Outside uncertainty band → EV comparison
        edge_hold = p - m_bid

        if edge_hold > self.cfg.fee_buffer:
            return "HOLD"
        else:
            return "SELL"