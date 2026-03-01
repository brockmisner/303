# fill_prob.py — Execution Fill Probability Model + Time-to-Expiry Scaling
"""
Adjusts Kelly sizing by estimated fill probability and slippage,
and decays sizing as T → 0 (late window = more gamma = more chaos).

Usage:
    FP = FillProbModel()
    fp = FP.estimate(spread, depth, ofi_dir_agrees, trade_intensity)
    kelly_adj = fp.kelly_mult  # multiply into Kelly

    TE = ExpiryScaler()
    te_mult = TE.kelly_mult(T_sec)  # multiply into Kelly
"""
from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass
class FillProbConfig:
    # Spread impact on fill prob
    spread_decay_k: float = 15.0   # p_fill decays with spread: exp(-k * spread)

    # Depth impact: more depth → higher fill prob
    depth_ref: float = 200.0       # shares at top of book where p_fill saturates
    depth_weight: float = 0.30     # weight of depth component

    # OFI directional agreement bonus
    ofi_agree_bonus: float = 0.15  # p_fill += 0.15 when OFI agrees with trade direction

    # Trade intensity: higher intensity → higher fill prob (active book)
    intensity_ref: float = 5.0     # trades/sec where intensity component saturates
    intensity_weight: float = 0.20

    # Slippage model
    slippage_base: float = 0.005   # base slippage (half a cent)
    slippage_spread_k: float = 0.5 # slippage scales with spread

    # Floor/cap
    p_fill_floor: float = 0.10
    p_fill_cap: float = 0.95


@dataclass
class ExpiryConfig:
    # Time window bounds
    T_hi: float = 180.0    # start of window (full conviction)
    T_lo: float = 30.0     # end of tradeable window

    # Kelly multiplier range
    kelly_early: float = 1.0    # full Kelly at T_hi
    kelly_late: float = 0.35    # reduced Kelly at T_lo

    # Gamma scaling: σ_eff increases as T → 0
    gamma_boost_threshold: float = 60.0   # seconds where gamma boost kicks in
    gamma_boost_max: float = 1.8          # max σ multiplier for gamma


class FillProbModel:
    """
    Estimates fill probability and slippage for Kelly adjustment.
    f_adjusted = f_kelly × p_fill × (1 - slippage)
    """

    def __init__(self, cfg: FillProbConfig = None):
        self.cfg = cfg or FillProbConfig()

    def estimate(
        self,
        spread: float,
        depth: float,          # total shares at top of book
        ofi_agrees: bool,      # does OFI direction agree with trade?
        trade_intensity: float, # trades/sec
    ) -> dict:
        """
        Returns dict with p_fill, slippage, kelly_mult.
        """
        cfg = self.cfg

        # Spread component: tighter spread → higher fill
        p_spread = math.exp(-cfg.spread_decay_k * max(0.0, spread))

        # Depth component (raw, unweighted)
        raw_depth = min(1.0, depth / cfg.depth_ref)

        # Intensity component (raw, unweighted)
        raw_intensity = min(1.0, trade_intensity / cfg.intensity_ref)

        # Base fill probability (weighted combination)
        p_fill = (p_spread * (1.0 - cfg.depth_weight - cfg.intensity_weight)
                  + raw_depth * cfg.depth_weight
                  + raw_intensity * cfg.intensity_weight)
        p_fill = min(1.0, max(0.0, p_fill))

        # OFI bonus
        if ofi_agrees:
            p_fill += cfg.ofi_agree_bonus

        # Clamp
        p_fill = max(cfg.p_fill_floor, min(cfg.p_fill_cap, p_fill))

        # Slippage
        slippage = cfg.slippage_base + cfg.slippage_spread_k * spread
        slippage = min(0.50, slippage)  # cap at 50%

        # Kelly multiplier
        kelly_mult = p_fill * (1.0 - slippage)
        kelly_mult = max(0.05, min(1.0, kelly_mult))

        return {
            "p_fill": round(p_fill, 4),
            "slippage": round(slippage, 4),
            "fill_kelly": round(kelly_mult, 4),
        }


class ExpiryScaler:
    """
    Time-to-expiry drift adjustment.
    Binary options behave differently as T → 0:
    - More gamma (price sensitivity to BTC moves)
    - More chaos (smaller time for reversion)
    - Need smaller sizes

    μ_adjusted = μ × √T  (drift scaling)
    Kelly decays linearly from kelly_early to kelly_late.
    """

    def __init__(self, cfg: ExpiryConfig = None):
        self.cfg = cfg or ExpiryConfig()

    def kelly_mult(self, T_sec: float) -> float:
        """Kelly multiplier based on time to expiry."""
        cfg = self.cfg
        T = max(0.0, float(T_sec))

        if T >= cfg.T_hi:
            return cfg.kelly_early
        if T <= cfg.T_lo:
            return cfg.kelly_late

        # Linear interpolation
        frac = (T - cfg.T_lo) / max(1e-9, cfg.T_hi - cfg.T_lo)
        return cfg.kelly_late + frac * (cfg.kelly_early - cfg.kelly_late)

    def sigma_multiplier(self, T_sec: float) -> float:
        """
        Gamma-aware sigma scaling: σ effectively increases as T → 0.
        Use this to inflate σ_1m when computing cone probability near expiry.
        """
        cfg = self.cfg
        T = max(1.0, float(T_sec))

        if T >= cfg.gamma_boost_threshold:
            return 1.0

        # Boost σ as T shrinks below threshold
        frac = (cfg.gamma_boost_threshold - T) / cfg.gamma_boost_threshold
        return 1.0 + frac * (cfg.gamma_boost_max - 1.0)

    def drift_adjustment(self, T_sec: float) -> float:
        """μ_adjusted = μ × √T scaling factor."""
        return math.sqrt(max(0.01, float(T_sec)))

    def status_dict(self, T_sec: float) -> dict:
        return {
            "expiry_kelly": round(self.kelly_mult(T_sec), 3),
            "expiry_sigma_mult": round(self.sigma_multiplier(T_sec), 3),
        }
