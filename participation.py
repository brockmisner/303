# participation.py
from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from typing import Optional, Dict, Any
import math
import numpy as np


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


@dataclass
class PVConfig:
    # velocity windows
    w_fast_s: float = 5.0
    w_slow_s: float = 20.0

    # signal-quality thresholds
    breakout_q: float = 0.25
    trap_q: float = -0.25

    # time-to-participation (ms)
    ttp_breakout_ms: int = 800
    ttp_trap_ms: int = 2000

    # normalization refs (tune later)
    tv_ref: float = 6.0        # trades/sec where PV saturates
    cons_ref: float = 120.0    # shares/sec consumed where QC saturates
    depth_ref: float = 200.0   # top depth scale

    # spread normalization window
    spread_hist_len: int = 120   # ~2 min at 1Hz

    # impulse detection (using z)
    impulse_z_on: float = 1.20
    impulse_z_off: float = 0.70


class ParticipationVelocity:
    """
    Computes participation velocity and distinguishes breakout vs trap.

    Inputs:
      - trade_velocity(window_s) from your FlowTracker (trades/sec)
      - up/dn BookSnapshot with bid/ask + bid_size/ask_size
      - z (from cone) and T

    Outputs:
      - signal_quality in [-1, +1] approx
      - participation_confirms boolean (2-of-3 rule)
      - trap/breakout flags
      - time_to_participation_ms (from impulse onset)
    """
    def __init__(self, cfg: Optional[PVConfig] = None):
        self.cfg = cfg or PVConfig()

        self.spread_hist = deque(maxlen=self.cfg.spread_hist_len)

        # book change history for consumption
        self._last_ts_ms: Optional[int] = None
        self._last_up_sz: Optional[float] = None
        self._last_dn_sz: Optional[float] = None

        # impulse tracking
        self._impulse_active: bool = False
        self._impulse_ts_ms: Optional[int] = None
        self._impulse_dir: int = 0  # +1 for UP side z, -1 for DOWN side z

        # first participation after impulse
        self._participation_ts_ms: Optional[int] = None

    def _spread_norm(self, spread: float) -> float:
        self.spread_hist.append(float(spread))
        if len(self.spread_hist) < 20:
            return 1.0
        med = float(np.median(np.array(self.spread_hist)))
        return float(min(spread / max(1e-9, med), 10.0))

    def update(
        self,
        *,
        ts_ms: int,
        z: float,
        T_sec: float,
        up_bid: float, up_ask: float, up_bid_sz: float, up_ask_sz: float,
        dn_bid: float, dn_ask: float, dn_bid_sz: float, dn_ask_sz: float,
        tv_fast: float,   # trades/sec over fast window
        tv_slow: float,   # trades/sec over slow window
    ) -> Dict[str, Any]:
        # --- basic book stats
        up_sp = max(0.0, float(up_ask) - float(up_bid))
        dn_sp = max(0.0, float(dn_ask) - float(dn_bid))
        spread = max(up_sp, dn_sp)

        up_depth = max(0.0, float(up_bid_sz) + float(up_ask_sz))
        dn_depth = max(0.0, float(dn_bid_sz) + float(dn_ask_sz))
        depth = min(up_depth, dn_depth)

        spread_norm = self._spread_norm(spread)

        # --- consumption estimate (shares/sec) based on top-of-book size shrink
        cons_rate = 0.0
        if self._last_ts_ms is not None and ts_ms > self._last_ts_ms:
            dt = (ts_ms - self._last_ts_ms) / 1000.0
            if dt > 0:
                if self._last_up_sz is not None:
                    d_up = max(0.0, self._last_up_sz - up_depth)
                else:
                    d_up = 0.0
                if self._last_dn_sz is not None:
                    d_dn = max(0.0, self._last_dn_sz - dn_depth)
                else:
                    d_dn = 0.0
                cons_rate = float((d_up + d_dn) / dt)

        self._last_ts_ms = ts_ms
        self._last_up_sz = up_depth
        self._last_dn_sz = dn_depth

        # --- participation features
        tv_fast = float(tv_fast)
        tv_slow = float(tv_slow)
        if tv_slow < 0.1:
            accel = 0.0
        else:
            accel = (tv_fast - tv_slow) / tv_slow

        PV = clamp01(math.log1p(tv_fast) / math.log1p(self.cfg.tv_ref))
        PA = clamp01(accel)  # only positive accel matters
        QC = clamp01(cons_rate / self.cfg.cons_ref)
        SN = clamp01(1.0 - min(3.0, spread_norm) / 3.0)  # tight spread => high SN
        DEP = clamp01(depth / self.cfg.depth_ref)

        breakout_score = 0.35 * PV + 0.25 * PA + 0.20 * QC + 0.20 * SN
        trap_score     = 0.40 * (1.0 - SN) + 0.35 * (1.0 - DEP) + 0.25 * (1.0 - QC)
        signal_quality = float(max(-1.0, min(1.0, breakout_score - trap_score)))

        # --- 2-of-3 confirmation rule
        accel_confirm = (PA > 0.60)
        cons_confirm  = (QC > 0.45)
        spread_confirm = (spread_norm < 1.20)  # not widening too much
        participation_confirms = (int(accel_confirm) + int(cons_confirm) + int(spread_confirm)) >= 2

        # --- impulse / time-to-participation tracking (z-based)
        z_abs = abs(float(z))
        if (not self._impulse_active) and z_abs >= self.cfg.impulse_z_on:
            self._impulse_active = True
            self._impulse_ts_ms = ts_ms
            self._impulse_dir = 1 if z > 0 else -1
            self._participation_ts_ms = None

        if self._impulse_active and z_abs <= self.cfg.impulse_z_off:
            # reset once impulse fades
            self._impulse_active = False
            self._impulse_ts_ms = None
            self._impulse_dir = 0
            self._participation_ts_ms = None

        # First qualifying participation after impulse
        if self._impulse_active and self._impulse_ts_ms is not None and self._participation_ts_ms is None:
            if participation_confirms:
                self._participation_ts_ms = ts_ms

        ttp_ms = None
        if self._impulse_active and self._impulse_ts_ms is not None and self._participation_ts_ms is not None:
            ttp_ms = int(self._participation_ts_ms - self._impulse_ts_ms)

        breakout_flag = False
        trap_flag = False
        if ttp_ms is not None:
            breakout_flag = (ttp_ms <= self.cfg.ttp_breakout_ms) and (signal_quality >= self.cfg.breakout_q)
            trap_flag = (ttp_ms >= self.cfg.ttp_trap_ms) or (signal_quality <= self.cfg.trap_q)

        return {
            "pv": PV, "pa": PA, "qc": QC,
            "spread": spread, "spread_norm": spread_norm,
            "depth": depth, "signal_quality": signal_quality,
            "participation_confirms": participation_confirms,
            "accel_confirm": accel_confirm,
            "cons_confirm": cons_confirm,
            "spread_confirm": spread_confirm,
            "impulse_active": self._impulse_active,
            "ttp_ms": ttp_ms,
            "breakout": breakout_flag,
            "trap": trap_flag,
            "tv_fast": tv_fast,
            "tv_slow": tv_slow,
            "cons_rate": cons_rate,
            "T": float(T_sec),
        }
