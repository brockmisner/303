"""
adaptive_executor.py — Institutional Adaptive Microstructure Execution Engine

Drop-in module targeting 25% rolling market coverage through intelligent
maker→step→cross escalation with:
  - Edge decay model during retries
  - Adverse selection probability (incl. flow bias)
  - Latency-adjusted fill probability
  - Queue-based fill estimation
  - True L2 cancel measurement via delta decomposition (optional, fed from main.py)
  - Momentum burst regime detection (BURST)
  - Regime-aware tick stepping
  - EV-gated controlled crossing
  - Cross cooldown + loss-streak breaker
  - Gamma proximity + inventory-aware sizing

Interfaces:
  MicroSnapshot — built upstream (main.py) from BOOK_CACHE + STATE + FLOW (+ optional L2Tracker/position)
  clob_client   — py-clob-client instance supporting create_order/post_order/cancel/get_order

Notes:
  * This module is self-contained, but expects upstream wiring for:
      - snap.bid_trade_rate / snap.bid_cancel_rate / snap.bid_queue_ahead
      - snap.ask_trade_rate / snap.ask_cancel_rate / snap.ask_queue_ahead
      - snap.trade_vel, snap.flow_bias, snap.lag_p10_ms, snap.sigma
      - snap.current_btc_price, snap.strike_price, snap.net_position, snap.max_inventory
"""

import math
import time
import logging
from collections import deque
from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any

from py_clob_client.clob_types import OrderArgs, OrderType, PartialCreateOrderOptions
from py_clob_client.order_builder.constants import BUY, SELL

logger = logging.getLogger("polybot")

Side = Literal["UP", "DOWN"]


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


# ═════════════════════════════════════════════════════════════════════════════
# TRUE L2 CANCEL TRACKING (optional; instantiate/wire in main.py)
# ═════════════════════════════════════════════════════════════════════════════

class L2Tracker:
    """
    True L2 cancel *volume* via delta decomposition:
        cancels = max(0, (prev_size - curr_size) - traded_volume_at_level)

    Upstream should convert volumes to rates if desired (shares/sec),
    or you can compute rate from your own sampling interval.
    """
    def __init__(self):
        self.prev_bid_size: Optional[float] = None
        self.prev_ask_size: Optional[float] = None
        self.cancel_bid: float = 0.0
        self.cancel_ask: float = 0.0

    def update(
        self,
        bid_size: float,
        ask_size: float,
        trade_vol_bid: float,
        trade_vol_ask: float,
    ) -> None:
        if self.prev_bid_size is not None and self.prev_ask_size is not None:
            delta_bid = self.prev_bid_size - bid_size
            delta_ask = self.prev_ask_size - ask_size
            self.cancel_bid = max(0.0, delta_bid - trade_vol_bid)
            self.cancel_ask = max(0.0, delta_ask - trade_vol_ask)
        self.prev_bid_size = bid_size
        self.prev_ask_size = ask_size


# ═════════════════════════════════════════════════════════════════════════════
# MOMENTUM BURST DETECTOR (micro-panic before sigma catches up)
# ═════════════════════════════════════════════════════════════════════════════

class BurstDetector:
    """Detect micro-panic bursts from acceleration of trade velocity + imbalance + spread."""
    def __init__(self, threshold: float = 3.5):
        self.prev_trade_vel: float = 0.0
        self.prev_imbalance: float = 0.0
        self.threshold = threshold

    def burst_score(self, trade_vel: float, imbalance: float, spread: float) -> float:
        dv = trade_vel - self.prev_trade_vel
        di = imbalance - self.prev_imbalance
        score = abs(dv) * 0.4 + abs(di) * 2.0 + spread * 3.0
        self.prev_trade_vel = trade_vel
        self.prev_imbalance = imbalance
        return score

    def is_burst(self, trade_vel: float, imbalance: float, spread: float) -> bool:
        return self.burst_score(trade_vel, imbalance, spread) > self.threshold


# ═════════════════════════════════════════════════════════════════════════════
# GAMMA + INVENTORY HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def gamma_modifier(current_price: float, strike_price: float) -> float:
    """Gamma proxy: 1/(|price - strike| + ε). Higher near strike."""
    return 1.0 / (abs(current_price - strike_price) + 0.01)


class InventoryManager:
    """Track net position and compute inventory dampening."""
    def __init__(self, max_inventory: float = 100.0):
        self.max_inventory = max_inventory
        self.net_position = 0.0

    def update(self, net_position: float, max_inventory: Optional[float] = None) -> None:
        if max_inventory is not None:
            self.max_inventory = float(max_inventory)
        self.net_position = float(net_position)

    def inventory_ratio(self) -> float:
        return abs(self.net_position) / max(float(self.max_inventory), 1.0)

    def should_dampen(self) -> bool:
        return self.inventory_ratio() > 0.7

    def size_scale(self) -> float:
        return 0.6 if self.should_dampen() else 1.0

    def disable_cross(self) -> bool:
        return self.should_dampen()


# ═════════════════════════════════════════════════════════════════════════════
# COVERAGE CONTROLLER (UP/DOWN separate, rolling 40-window)
# ═════════════════════════════════════════════════════════════════════════════

class CoverageController:
    def __init__(self, window: int = 40, target: float = 0.25):
        self.target = target
        self.up: deque = deque(maxlen=window)
        self.down: deque = deque(maxlen=window)

    def record(self, side: Side, filled: bool):
        (self.up if side == "UP" else self.down).append(1 if filled else 0)

    def coverage(self, side: Side) -> float:
        arr = self.up if side == "UP" else self.down
        return (sum(arr) / len(arr)) if arr else 0.0

    def deficit(self, side: Side) -> float:
        return max(0.0, self.target - self.coverage(side))


# ═════════════════════════════════════════════════════════════════════════════
# REGIME CLASSIFIER (execution-specific)
# ═════════════════════════════════════════════════════════════════════════════

class ExecutionRegimeClassifier:
    """Execution-specific regime (finer-grained than the oracle regime)."""

    def classify(
        self,
        sigma: float,
        spread: float,
        lag_p10_ms: float,
        trade_vel: float = 0.0,
        burst_score: float = 0.0,
        burst_threshold: float = 3.5,
    ) -> str:
        # BURST gets priority (micro-panic): small maker only, no cross
        if burst_score > burst_threshold:
            return "BURST"
        if lag_p10_ms < 220:
            return "FAST_TAIL"
        if sigma > 0.0025 or spread > 0.08:
            return "PANIC"
        if trade_vel > 8.0 or spread > 0.04:
            return "ACTIVE"
        return "CALM"


# ═════════════════════════════════════════════════════════════════════════════
# MICRO PARAMS (calibrated for Polymarket 5-min BTC binary)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class MicroParams:
    tick_size: float = 0.01
    fee_per_share: float = 0.0     # Polymarket maker fee often 0; adjust if needed.

    # Maker fill horizon (s) — how long to wait before expected fill
    maker_horizon_s: float = 0.50

    # Latency penalty: p_fill *= exp(-k * lag_ms)
    latency_k: float = 0.0018

    # Adverse selection model weights
    adv_a_sigma: float = 35.0      # sigma contribution
    adv_b_imb: float = 0.45        # imbalance contribution
    adv_c_vel: float = 0.03        # trade velocity contribution
    adv_d_flow: float = 0.35       # flow bias contribution (momentum microbursts)
    adverse_impact: float = 0.015  # adverse impact in price units

    # Edge decay rate (per second) during retry waiting
    edge_decay_kappa: float = 1.0

    # Controlled cross gating
    deficit_cross_on: float = 0.05
    z_cross_threshold: float = 1.25
    cross_cooldown_signals: int = 2

    # Cross sizing bounds
    max_cross_frac_of_size: float = 1.0
    cross_size_scale: float = 3.0
    max_slippage_frac: float = 0.30

    # Fill probability sanity caps (stabilizes EV)
    pfill_floor: float = 0.02
    pfill_cap: float = 0.85

    # Burst detector
    burst_threshold: float = 3.5

    # Gamma / inventory penalties (tune from logs)
    gamma_scale_coef: float = 0.05   # scale = clamp(1 - gamma*coef, 0.3, 1.0)
    gamma_penalty_coef: float = 0.001  # penalty term multiplier
    inv_penalty_coef: float = 0.01     # penalty term multiplier


# ═════════════════════════════════════════════════════════════════════════════
# EXECUTION TOGGLES
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ExecutionToggles:
    allow_controlled_cross: bool = True


# ═════════════════════════════════════════════════════════════════════════════
# MICRO FEED SNAPSHOT
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class MicroSnapshot:
    """Book + microstructure snapshot passed from main.py."""
    best_bid: float = 0.0
    best_ask: float = 1.0
    best_bid_sz: float = 0.0
    best_ask_sz: float = 0.0
    imbalance: float = 0.0      # [-1, +1] signed bid/ask size imbalance
    trade_vel: float = 0.0      # trades/sec (CLOB prints ideally)
    sigma: float = 0.001
    lag_p10_ms: float = 500.0

    # Queue info at best levels (approximate from L2 unless you track per-level)
    bid_queue_ahead: float = 0.0
    bid_trade_rate: float = 1.0   # shares/sec traded at bid
    bid_cancel_rate: float = 0.5  # shares/sec cancelled at bid
    ask_queue_ahead: float = 0.0
    ask_trade_rate: float = 1.0
    ask_cancel_rate: float = 0.5
    flow_bias: float = 0.0       # (buy_prints - sell_prints) / total, [-1,+1]

    # Gamma/inventory context (optional but recommended)
    strike_price: float = 0.50
    current_btc_price: float = 0.0
    net_position: float = 0.0
    max_inventory: float = 100.0


# ═════════════════════════════════════════════════════════════════════════════
# EXECUTION RESULT
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ExecutionResult:
    filled: bool = False
    fill_price: float = 0.0
    fill_size: float = 0.0
    method: str = ""        # "maker", "step_N", "cross", "none"
    order_id: str = ""
    exec_ms: int = 0
    attempts: int = 0
    error: str = ""
    maker_ev: float = 0.0
    cross_ev: float = 0.0

    # Diagnostics (institutional telemetry)
    p_fill: float = 0.0
    p_adv: float = 0.0
    burst_score: float = 0.0
    gamma: float = 0.0
    gamma_scale: float = 1.0
    inv_ratio: float = 0.0
    size_final: int = 0


# ═════════════════════════════════════════════════════════════════════════════
# ADAPTIVE EXECUTOR — INSTITUTIONAL GRADE + REGIME/BURST/GAMMA/INVENTORY AWARE
# ═════════════════════════════════════════════════════════════════════════════

class AdaptiveExecutor:
    """
    Production-grade adaptive execution engine.

    Flow:
      1. Determine execution regime (incl. BURST)
      2. Apply gamma + inventory sizing modifiers
      3. Build maker price ladder (regime-dependent steps)
      4. Evaluate EV at each maker price (edge decay + fill prob + adverse selection)
      5. Place GTC at EV-positive prices, wait TTL (90-220ms)
      6. Step through ladder if unfilled
      7. If undercovered + EV-positive → controlled partial cross
         (disabled in BURST/FAST_TAIL/PANIC or inventory dampen)
      8. Record fill result to coverage controller

    Usage:
        executor = AdaptiveExecutor(clob_client)
        result = executor.execute(side, token_id, model_prob, z, size, snapshot, deficit)
    """

    def __init__(
        self,
        clob_client,
        params: Optional[MicroParams] = None,
        toggles: Optional[ExecutionToggles] = None,
    ):
        self.client = clob_client
        self.params = params or MicroParams()
        self.toggles = toggles or ExecutionToggles()

        self.coverage = CoverageController(window=40, target=0.25)
        self.exec_regime = ExecutionRegimeClassifier()
        self.burst = BurstDetector(threshold=self.params.burst_threshold)
        self.inventory = InventoryManager(max_inventory=100.0)

        self.cross_cooldown: int = 0
        self.consecutive_losses: int = 0

    # ─────────────────────────────────────────────────────────────────────
    # Main entry point (called from execution_loop via thread_pool)
    # ─────────────────────────────────────────────────────────────────────

    def execute(
        self,
        side: Side,
        token_id: str,
        model_prob: float,
        z_score: float,
        base_size: int,
        snap: MicroSnapshot,
        deficit: float,
        order_side: str = "BUY",
    ) -> ExecutionResult:
        """Full adaptive execution: maker ladder → controlled cross."""

        t0 = time.time()
        result = ExecutionResult()
        is_buy = (order_side.upper() == "BUY")

        spread = snap.best_ask - snap.best_bid

        # ── Burst detection ──────────────────────────────────────────────
        b_score = self.burst.burst_score(snap.trade_vel, snap.imbalance, spread)
        regime = self.exec_regime.classify(
            snap.sigma, spread, snap.lag_p10_ms, snap.trade_vel,
            burst_score=b_score, burst_threshold=self.params.burst_threshold,
        )

        # ── Inventory state ──────────────────────────────────────────────
        self.inventory.update(
            net_position=snap.net_position,
            max_inventory=snap.max_inventory,
        )
        inv_ratio = self.inventory.inventory_ratio()

        # ── Gamma scaling ────────────────────────────────────────────────
        gamma = gamma_modifier(snap.current_btc_price, snap.strike_price)
        gamma_scale = clamp(
            1.0 - gamma * self.params.gamma_scale_coef, 0.3, 1.0,
        )

        # ── Final size ───────────────────────────────────────────────────
        size_scale = gamma_scale * self.inventory.size_scale()
        import math as _m
        _min_sz = max(1, _m.ceil(1.0 / max(0.01, snap.best_ask if is_buy else snap.best_bid)))
        _raw = base_size * size_scale
        size_final = max(_min_sz, int(_raw)) if _m.isfinite(_raw) else _min_sz

        # Populate telemetry
        result.burst_score = b_score
        result.gamma = gamma
        result.gamma_scale = gamma_scale
        result.inv_ratio = inv_ratio
        result.size_final = size_final

        # ── Base edge ────────────────────────────────────────────────────
        if is_buy:
            edge0 = model_prob - snap.best_ask - self.params.fee_per_share
        else:
            edge0 = snap.best_ask - model_prob - self.params.fee_per_share

        if edge0 <= 0:
            result.error = "NO_EDGE"
            result.exec_ms = int((time.time() - t0) * 1000)
            logger.info(
                f"EXEC: {side} NO_EDGE edge0={edge0:.4f} "
                f"model={model_prob:.4f} bid={snap.best_bid:.2f} "
                f"ask={snap.best_ask:.2f}"
            )
            self.coverage.record(side, False)
            return result

        # ── Phase 1: Maker ladder ────────────────────────────────────────
        maker_prices = self._maker_price_ladder(
            snap.best_bid, snap.best_ask, spread, regime, deficit, is_buy,
        )
        aggression = clamp(deficit * 4.0, 0.0, 1.0)

        logger.info(
            f"EXEC_START: {side} {order_side} size={base_size}→{size_final} "
            f"bid={snap.best_bid:.2f} ask={snap.best_ask:.2f} "
            f"spread={spread:.2f} regime={regime} burst={b_score:.2f} "
            f"deficit={deficit:.2%} aggr={aggression:.2f} "
            f"gamma_scale={gamma_scale:.2f} inv={inv_ratio:.2f} "
            f"steps={len(maker_prices)}"
        )

        best_maker_ev = -1e9
        filled = False

        for step_idx, price in enumerate(maker_prices):
            price = round(price, 2)
            price = max(0.01, min(0.99, price))

            age_s = time.time() - t0
            maker_ev, p_fill, p_adv = self._maker_ev(
                model_prob=model_prob,
                price=price,
                snap=snap,
                age_s=age_s,
                is_buy=is_buy,
            )
            best_maker_ev = max(best_maker_ev, maker_ev)

            if maker_ev <= 0:
                logger.debug(
                    f"EXEC_STEP: step={step_idx} price={price:.2f} "
                    f"ev={maker_ev:.6f} — skipped (EV≤0)"
                )
                continue

            # Place GTC maker order with TTL
            oid, was_filled, matched_size = self._place_maker_with_ttl(
                side, token_id, price, size_final,
                deficit, regime, is_buy,
            )
            result.attempts += 1

            if was_filled:
                result.filled = True
                result.fill_price = price
                result.fill_size = matched_size if matched_size > 0 else size_final
                result.order_id = oid or ""
                result.method = (
                    "maker" if step_idx == 0 else f"step_{step_idx}"
                )
                result.maker_ev = maker_ev
                result.p_fill = p_fill
                result.p_adv = p_adv
                result.exec_ms = int((time.time() - t0) * 1000)

                logger.info(
                    f"EXEC_FILLED: {result.method} {side} {order_side} "
                    f"price={price:.2f} size={size_final} ev={maker_ev:.4f} "
                    f"p_fill={p_fill:.3f} p_adv={p_adv:.3f} "
                    f"exec_ms={result.exec_ms}"
                )
                self.coverage.record(side, True)
                self.consecutive_losses = 0
                return result

            logger.info(
                f"EXEC_STEP: step={step_idx} price={price:.2f} "
                f"ev={maker_ev:.4f} p_fill={p_fill:.3f} "
                f"p_adv={p_adv:.3f} unfilled → next"
            )

        # ── Phase 2: Controlled cross ────────────────────────────────────
        allow_cross = (
            self.toggles.allow_controlled_cross
            and not self.inventory.disable_cross()
            and regime not in ("BURST", "FAST_TAIL", "PANIC")
        )

        if allow_cross:
            cross_result = self._maybe_controlled_cross(
                side=side,
                token_id=token_id,
                model_prob=model_prob,
                z=z_score,
                snap=snap,
                regime=regime,
                deficit=deficit,
                base_size=size_final,
                maker_best_ev=best_maker_ev,
                is_buy=is_buy,
            )
            if cross_result:
                result.filled = True
                result.fill_price = (
                    snap.best_ask if is_buy else snap.best_bid
                )
                result.fill_size = cross_result["size"]
                result.order_id = cross_result.get("oid", "")
                result.method = "cross"
                result.cross_ev = cross_result.get("ev", 0.0)
                result.exec_ms = int((time.time() - t0) * 1000)
                logger.info(
                    f"EXEC_CROSS_FILLED: {side} "
                    f"price={result.fill_price:.2f} "
                    f"size={result.fill_size} "
                    f"ev={result.cross_ev:.4f} "
                    f"exec_ms={result.exec_ms}"
                )
                self.coverage.record(side, True)
                self.consecutive_losses = 0
                return result

        # ── No fill ──────────────────────────────────────────────────────
        result.exec_ms = int((time.time() - t0) * 1000)
        result.method = "none"
        result.maker_ev = best_maker_ev
        logger.info(
            f"EXEC_NOFILL: {side} attempts={result.attempts} "
            f"best_ev={best_maker_ev:.4f} exec_ms={result.exec_ms} "
            f"regime={regime} allow_cross={allow_cross}"
        )
        self.coverage.record(side, False)
        self.consecutive_losses += 1
        return result

    # ─────────────────────────────────────────────────────────────────────
    # Maker Price Ladder
    # ─────────────────────────────────────────────────────────────────────

    def _maker_price_ladder(
        self,
        best_bid: float,
        best_ask: float,
        spread: float,
        regime: str,
        deficit: float,
        is_buy: bool,
    ) -> list:
        """Build regime-dependent price ladder for maker orders."""

        # Regime policy table:
        #   CALM:      2 steps, cross allowed
        #   ACTIVE:    3 steps, cross limited (gated elsewhere)
        #   BURST:     1 step  (small maker only), cross disabled
        #   FAST_TAIL: 1 step,  cross disabled
        #   PANIC:     0 steps, cross disabled
        max_steps = {
            "CALM": 2,
            "ACTIVE": 3,
            "BURST": 1,
            "FAST_TAIL": 1,
            "PANIC": -1,
        }.get(regime, 2)

        if max_steps < 0:
            return []

        aggr = clamp(deficit * 4.0, 0.0, 1.0)
        # Extra step when heavily undercovered and regime allows
        if regime in ("CALM", "ACTIVE") and aggr > 0.75:
            max_steps += 1

        tick = self.params.tick_size
        if is_buy:
            prices = [best_bid + i * tick for i in range(max_steps + 1)]
            # Never cross the spread in maker mode
            prices = [p for p in prices if p < best_ask]
        else:
            prices = [best_bid + (i + 1) * tick for i in range(max_steps + 1)]
            prices = [p for p in prices if p < best_ask]

        # ── Queue jump: micro-improve first price to get ahead ─────────
        # Post 1 tick better than best bid/ask when spread allows, still maker
        if prices and spread >= 0.02:
            if is_buy:
                prices[0] = min(best_ask - tick, best_bid + tick)
            else:
                prices[0] = max(best_bid + tick, best_ask - tick)

        return prices

    # ─────────────────────────────────────────────────────────────────────
    # Maker EV Model
    # ─────────────────────────────────────────────────────────────────────

    def _maker_ev(
        self,
        model_prob: float,
        price: float,
        snap: MicroSnapshot,
        age_s: float,
        is_buy: bool,
    ) -> tuple:
        """
        Expected value of posting a maker order at `price`.

        EV = (p_fill × edge_decayed)
             - (p_adverse × impact)
             - gamma_penalty
             - inv_risk

        Returns (ev, p_fill, p_adv).
        """
        # Edge at this price
        if is_buy:
            edge0 = model_prob - price - self.params.fee_per_share
        else:
            edge0 = price - model_prob - self.params.fee_per_share
        if edge0 <= 0:
            return -1e9, 0.0, 0.0

        # Edge decays with time spent in the ladder
        edge = edge0 * math.exp(-self.params.edge_decay_kappa * age_s)

        # Queue-based fill probability
        tr = snap.bid_trade_rate if is_buy else snap.ask_trade_rate
        cr = snap.bid_cancel_rate if is_buy else snap.ask_cancel_rate
        q = snap.bid_queue_ahead if is_buy else snap.ask_queue_ahead
        cleared = (tr + cr) * self.params.maker_horizon_s
        p_fill = clamp(cleared / (q + 1e-9), 0.0, 1.0)

        # Latency penalty
        p_fill *= math.exp(
            -self.params.latency_k * max(0.0, snap.lag_p10_ms)
        )

        # Sanity caps — real queue survival rarely >85% in 350ms,
        # and true fill prob rarely <2% if at best bid/ask
        p_fill = clamp(p_fill, self.params.pfill_floor, self.params.pfill_cap)

        # Adverse selection probability (includes momentum microbursts)
        p_adv = clamp(
            snap.sigma * self.params.adv_a_sigma
            + abs(snap.imbalance) * self.params.adv_b_imb
            + snap.trade_vel * self.params.adv_c_vel
            + abs(snap.flow_bias) * self.params.adv_d_flow,
            0.0,
            0.6,
        )

        # Gamma penalty (higher near strike)
        gamma = gamma_modifier(snap.current_btc_price, snap.strike_price)
        gamma_penalty = (
            abs(snap.current_btc_price - snap.strike_price)
            * gamma * self.params.gamma_penalty_coef
        )

        # Inventory risk penalty
        inv_risk = (
            abs(snap.net_position / max(snap.max_inventory, 1.0))
            * self.params.inv_penalty_coef
        )

        ev = (
            (p_fill * edge)
            - (p_adv * self.params.adverse_impact)
            - gamma_penalty
            - inv_risk
        )
        return ev, p_fill, p_adv

    # ─────────────────────────────────────────────────────────────────────
    # TTL Maker Placement
    # ─────────────────────────────────────────────────────────────────────

    def _place_maker_with_ttl(
        self,
        side: Side,
        token_id: str,
        price: float,
        size: int,
        deficit: float,
        regime: str,
        is_buy: bool,
    ) -> tuple:
        """
        Place a GTC maker order and wait TTL before checking fill.
        TTL shrinks with coverage deficit (more aggressive when undercovered).

        Returns (order_id, filled_bool).
        """
        aggr = clamp(deficit * 4.0, 0.0, 1.0)

        base_ttl = {
            "CALM": 1500,
            # Note: classifier returns CALM/ACTIVE/BURST/FAST_TAIL/PANIC only
            "ACTIVE": 800,
            "BURST": 350,
            "FAST_TAIL": 350,
            "PANIC": 200,
        }.get(regime, 1000)

        # More aggressive when undercovered: shorten a bit, but don't kill it
        ttl_ms = int(base_ttl * (1.0 - 0.35 * aggr))
        ttl_ms = max(300, min(2000, ttl_ms))

        oid = self._clob_place_gtc(token_id, price, size, is_buy)
        if not oid:
            return None, False, 0.0

        time.sleep(ttl_ms / 1000.0)

        filled, matched = self._clob_check_filled(oid)
        if filled:
            return oid, True, matched

        self._clob_cancel(oid)
        # Re-check after cancel — order may have filled in the race window
        filled2, matched2 = self._clob_check_filled(oid)
        if filled2:
            logger.info(f"GTC_RACE_FILL: order {oid} filled between check and cancel, matched={matched2}")
            return oid, True, matched2
        return oid, False, 0.0

    # ─────────────────────────────────────────────────────────────────────
    # Controlled Cross
    # ─────────────────────────────────────────────────────────────────────

    def _maybe_controlled_cross(
        self,
        side: Side,
        token_id: str,
        model_prob: float,
        z: float,
        snap: MicroSnapshot,
        regime: str,
        deficit: float,
        base_size: int,
        maker_best_ev: float,
        is_buy: bool,
    ) -> Optional[Dict[str, Any]]:
        """
        Attempt a controlled cross if all preconditions met.
        Returns {size, oid, ev} on fill, else None.
        """
        if self.cross_cooldown > 0:
            return None
        if deficit < self.params.deficit_cross_on:
            return None
        if abs(z) < self.params.z_cross_threshold:
            return None
        if regime in ("BURST", "FAST_TAIL", "PANIC"):
            return None
        if self.consecutive_losses >= 2:
            return None
        if self.inventory.disable_cross():
            return None

        # Buy-cross uses best_ask; sell-cross uses best_bid
        cross_price = snap.best_ask if is_buy else snap.best_bid

        # Edge if crossing
        if is_buy:
            edge_cross = (
                model_prob - cross_price - self.params.fee_per_share
            )
        else:
            edge_cross = (
                cross_price - model_prob - self.params.fee_per_share
            )

        if edge_cross <= 0:
            return None

        # Impact proxy for crossing (half-spread)
        impact_cross = 0.5 * (snap.best_ask - snap.best_bid)

        # Cross adverse probability (tighter model than maker)
        p_adv_cross = clamp(
            snap.sigma * 25.0 + abs(snap.flow_bias) * 0.4,
            0.0,
            0.5,
        )

        ev_cross = (
            edge_cross
            - impact_cross
            - (p_adv_cross * 0.5 * self.params.adverse_impact)
        )

        # Only cross if it beats best maker EV
        if ev_cross <= maker_best_ev:
            return None

        # Slippage cap
        if is_buy:
            max_cross_price = model_prob - (
                (1.0 - self.params.max_slippage_frac) * edge_cross
            )
            if cross_price > max_cross_price:
                return None
        else:
            min_cross_price = model_prob + (
                (1.0 - self.params.max_slippage_frac) * edge_cross
            )
            if cross_price < min_cross_price:
                return None

        # Size scaled by deficit
        cross_mult = min(
            self.params.max_cross_frac_of_size,
            deficit * self.params.cross_size_scale,
        )
        import math as _m
        _min_cross = max(1, _m.ceil(1.0 / max(0.01, snap.best_ask if is_buy else snap.best_bid)))
        cross_size = max(_min_cross, int(base_size * cross_mult))

        logger.info(
            f"EXEC_CROSS: {side} price={cross_price:.2f} "
            f"size={cross_size} ev_cross={ev_cross:.4f} "
            f"ev_maker={maker_best_ev:.4f} deficit={deficit:.2%} "
            f"p_adv_cross={p_adv_cross:.3f}"
        )

        cross_price = round(cross_price, 2)
        cross_price = max(0.01, min(0.99, cross_price))

        # FOK taker order — is_buy is dynamic, not hardcoded
        oid = self._clob_place_fok(
            token_id, cross_price, cross_size, is_buy=is_buy,
        )
        if not oid:
            return None

        # Short wait for taker fill confirmation
        time.sleep(0.12)
        filled, matched = self._clob_check_filled(oid)
        if not filled:
            self._clob_cancel(oid)
            return None

        self.cross_cooldown = self.params.cross_cooldown_signals
        return {"size": matched if matched > 0 else cross_size, "oid": oid, "ev": ev_cross}

    # ─────────────────────────────────────────────────────────────────────
    # Window lifecycle
    # ─────────────────────────────────────────────────────────────────────

    def on_window_rollover(self):
        """Call at each window change to tick cooldowns."""
        if self.cross_cooldown > 0:
            self.cross_cooldown -= 1

    # ─────────────────────────────────────────────────────────────────────
    # CLOB Client Wrappers (py-clob-client API)
    # ─────────────────────────────────────────────────────────────────────

    def _clob_place_gtc(
        self, token_id: str, price: float, size: int, is_buy: bool
    ) -> Optional[str]:
        """Place a GTC maker limit order. Returns order_id or None."""
        try:
            side_const = BUY if is_buy else SELL
            args = OrderArgs(
                price=price, size=size, side=side_const, token_id=token_id
            )
            signed = self.client.create_order(
                args,
                options=PartialCreateOrderOptions(
                    tick_size="0.01", neg_risk=False
                ),
            )
            resp = self.client.post_order(signed, orderType=OrderType.GTC)
            if resp and resp.get("success"):
                oid = resp.get("orderID", "")
                logger.info(
                    f"GTC_PLACED: {oid} price={price:.2f} size={size} is_buy={is_buy}"
                )
                return oid
            else:
                err = (
                    resp.get("errorMsg", resp)
                    if isinstance(resp, dict) else resp
                )
                logger.warning(f"GTC_FAIL: {err}")
                return None
        except Exception as e:
            logger.warning(f"GTC_EXCEPTION: {e}")
            return None

    def _clob_place_fok(
        self, token_id: str, price: float, size: int, is_buy: bool
    ) -> Optional[str]:
        """Place a FAK taker order for crossing. Returns order_id or None.
        FAK (Fill-And-Kill) allows partial fills unlike FOK's all-or-nothing."""
        try:
            side_const = BUY if is_buy else SELL
            args = OrderArgs(
                price=price, size=size, side=side_const, token_id=token_id
            )
            signed = self.client.create_order(
                args,
                options=PartialCreateOrderOptions(
                    tick_size="0.01", neg_risk=False
                ),
            )
            resp = self.client.post_order(signed, orderType=OrderType.FAK)
            if resp and resp.get("success"):
                return resp.get("orderID", "")
            else:
                err = (
                    resp.get("errorMsg", resp)
                    if isinstance(resp, dict) else resp
                )
                logger.warning(f"FAK_FAIL: {err}")
                return None
        except Exception as e:
            logger.warning(f"FAK_EXCEPTION: {e}")
            return None

    def _clob_check_filled(self, order_id: str) -> tuple:
        """Check if an order is filled via get_order API.
        Returns (filled: bool, size_matched: float)."""
        try:
            resp = self.client.get_order(order_id)
            if isinstance(resp, dict):
                status = str(resp.get("status", "")).upper()
                size_matched = float(resp.get("size_matched", 0) or 0)
                original_size = float(
                    resp.get("original_size", 0)
                    or resp.get("size", 0)
                    or 0
                )
                if status in ("MATCHED", "FILLED"):
                    return True, size_matched if size_matched > 0 else original_size
                if size_matched > 0:
                    logger.info(f"PARTIAL_FILL: {size_matched}/{original_size} matched")
                    return True, size_matched
            return False, 0.0
        except Exception as e:
            logger.warning(f"CHECK_FILL_ERR: {e}")
            return False, 0.0

    def _clob_cancel(self, order_id: str):
        """Cancel an open order."""
        try:
            self.client.cancel(order_id=order_id)
        except Exception as e:
            logger.debug(f"CANCEL_ERR: {e}")
