
"""
inventory_layer.py

Inventory-aware decision layer for binary outcome markets (UP/DOWN).
Designed to sit between decide_edge() and execution, using Position/Portfolio
from position.py for inventory + PnL awareness.

Core goals:
- Prevent whipsaw: avoid opening the opposite side while you still hold the prior side
  unless a strong reversal threshold is met.
- Prevent accidental synthetic straddles (owning both sides in size) unless explicitly allowed.
- Provide explicit unwind plans when a straddle forms (e.g., due to reconciled fills).
- Provide a single "gate" function that returns an actionable Decision + optional order plan.

This module is independent and backward-compatible: you can call it without changing your model.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, Tuple
import math

from position import Portfolio, Position, Side  # expects /position.py in your repo


class DecisionType(str, Enum):
    REJECT = "REJECT"
    APPROVE = "APPROVE"
    APPROVE_REDUCED = "APPROVE_REDUCED"
    CLOSE_THEN_REVERSE = "CLOSE_THEN_REVERSE"
    UNWIND_STRADDLE = "UNWIND_STRADDLE"
    FLATTEN = "FLATTEN"


@dataclass
class OrderPlan:
    """A single suggested order (not executed here)."""
    token_side: str               # "UP" or "DOWN"
    action: str                   # "BUY" or "SELL"
    qty: float
    limit_price: float
    reason: str
    meta: Dict[str, Any]


@dataclass
class Decision:
    kind: DecisionType
    reason: str
    # If kind != REJECT, you may execute proposed_side with this size multiplier.
    size_mult: float = 1.0
    # Optional plan: close first, unwind straddle, flatten, etc.
    plan: Optional[Tuple[OrderPlan, ...]] = None
    debug: Optional[Dict[str, Any]] = None


@dataclass
class InventoryLayerConfig:
    # --- Window bias / reversal discipline ---
    reversal_z: float = 2.20           # require |z| >= this to flip against bias
    reversal_conf: float = 0.88        # require dominance >= this
    reversal_cooldown_sec: float = 30  # minimum time between bias flips (optional)

    # --- Straddle detection ---
    straddle_min_shares: float = 3.0   # both sides >= this triggers straddle detection
    straddle_unwind_frac: float = 1.0  # 1.0 = unwind full smaller leg, <1 unwind partial
    allow_straddle: bool = False       # default: never intentionally hold both

    # --- Risk caps (scale in) ---
    max_net_shares: float = 60.0       # net exposure cap per window
    max_gross_shares: float = 90.0     # gross exposure cap per window
    size_decay_power: float = 1.0      # 1.0 linear decay; >1 stronger decay

    # --- Late window discipline ---
    late_t_sec: float = 75.0           # below this, disallow flips (unless extremely strong)
    late_reversal_z: float = 3.00
    late_reversal_conf: float = 0.93

    # --- Pricing selection for unwind plans ---
    # For sells, we typically want to hit BID (conservative).
    # For buys, we typically pay ASK (conservative).
    use_mid_for_unrealized: bool = True


class InventoryDecisionLayer:
    """
    Maintains lightweight per-market state (window_bias, flip timestamps) and
    uses Portfolio inventory to decide whether to allow a proposed trade.
    """

    def __init__(self, cfg: Optional[InventoryLayerConfig] = None):
        self.cfg = cfg or InventoryLayerConfig()
        self.window_bias: Optional[str] = None  # "UP" or "DOWN"
        self.last_flip_ts_ms: Optional[int] = None

    @staticmethod
    def _dominance(p_cone: float) -> float:
        return max(p_cone, 1.0 - p_cone)

    def _bias_from_prob(self, p_cone: float) -> str:
        return "UP" if p_cone >= 0.5 else "DOWN"

    def _now_sec(self, ts_ms: int) -> float:
        return ts_ms / 1000.0

    def _cooldown_ok(self, ts_ms: int) -> bool:
        if self.last_flip_ts_ms is None:
            return True
        dt = self._now_sec(ts_ms) - self._now_sec(self.last_flip_ts_ms)
        return dt >= self.cfg.reversal_cooldown_sec

    def _inv(self, portfolio: Portfolio) -> Tuple[float, float]:
        up = portfolio.up.inventory
        dn = portfolio.down.inventory
        return up, dn

    def _gross(self, portfolio: Portfolio) -> float:
        up, dn = self._inv(portfolio)
        return abs(up) + abs(dn)

    def _net(self, portfolio: Portfolio) -> float:
        # + = UP net, - = DOWN net
        return portfolio.up.inventory - portfolio.down.inventory

    def _is_straddle(self, portfolio: Portfolio) -> bool:
        up, dn = self._inv(portfolio)
        return (up >= self.cfg.straddle_min_shares) and (dn >= self.cfg.straddle_min_shares)

    def _size_decay(self, portfolio: Portfolio) -> float:
        """Decay additional sizing as gross or net approaches caps."""
        gross = self._gross(portfolio)
        net = abs(self._net(portfolio))
        gross_ratio = min(1.0, gross / max(1e-9, self.cfg.max_gross_shares))
        net_ratio = min(1.0, net / max(1e-9, self.cfg.max_net_shares))
        ratio = max(gross_ratio, net_ratio)
        # decay factor in [0,1]
        factor = max(0.0, 1.0 - ratio)
        if self.cfg.size_decay_power != 1.0:
            factor = factor ** self.cfg.size_decay_power
        return factor

    def _unwind_plan(
        self,
        portfolio: Portfolio,
        up_bid: float, up_ask: float,
        dn_bid: float, dn_ask: float,
        ts_ms: int,
        reason: str
    ) -> Tuple[OrderPlan, ...]:
        """
        Unwind accidental straddle by selling the smaller leg (or a fraction).
        Conservative: sell at BID.
        """
        up, dn = self._inv(portfolio)
        if up <= 0 or dn <= 0:
            return tuple()

        # choose smaller leg to unwind
        if up <= dn:
            qty = up * self.cfg.straddle_unwind_frac
            return (OrderPlan(
                token_side="UP",
                action="SELL",
                qty=qty,
                limit_price=up_bid,
                reason=reason,
                meta={"ts_ms": ts_ms, "unwind": "smaller_leg", "leg": "UP"}
            ),)
        else:
            qty = dn * self.cfg.straddle_unwind_frac
            return (OrderPlan(
                token_side="DOWN",
                action="SELL",
                qty=qty,
                limit_price=dn_bid,
                reason=reason,
                meta={"ts_ms": ts_ms, "unwind": "smaller_leg", "leg": "DOWN"}
            ),)

    def _flatten_plan(
        self,
        portfolio: Portfolio,
        up_bid: float, dn_bid: float,
        ts_ms: int,
        reason: str
    ) -> Tuple[OrderPlan, ...]:
        up, dn = self._inv(portfolio)
        plans = []
        if up > 0:
            plans.append(OrderPlan("UP", "SELL", up, up_bid, reason, {"ts_ms": ts_ms}))
        if dn > 0:
            plans.append(OrderPlan("DOWN", "SELL", dn, dn_bid, reason, {"ts_ms": ts_ms}))
        return tuple(plans)

    def update_bias_if_flat(self, portfolio: Portfolio, proposed_side: str, ts_ms: int) -> None:
        """If no inventory, set bias on first entry."""
        if self._gross(portfolio) <= 1e-9 and self.window_bias is None:
            self.window_bias = proposed_side
            self.last_flip_ts_ms = ts_ms

    def gate(
        self,
        *,
        ts_ms: int,
        portfolio: Portfolio,
        proposed_side: str,               # "UP" or "DOWN" from decide_edge
        z: float,
        p_cone: float,
        T_sec: float,
        up_bid: float, up_ask: float,
        dn_bid: float, dn_ask: float,
        allow_flip: bool = True,
        debug_in: Optional[Dict[str, Any]] = None
    ) -> Decision:
        """
        Main entrypoint. Returns a Decision.
        Call this right before placing an order (FIRE), using the latest inventory.
        """
        dbg = dict(debug_in or {})
        dominance = self._dominance(p_cone)
        model_side = self._bias_from_prob(p_cone)

        up_inv, dn_inv = self._inv(portfolio)
        gross = self._gross(portfolio)
        net = self._net(portfolio)

        dbg.update({
            "iadl": True,
            "bias": self.window_bias,
            "model_side": model_side,
            "dominance": dominance,
            "inv_up": up_inv,
            "inv_dn": dn_inv,
            "inv_gross": gross,
            "inv_net": net,
            "T": T_sec,
        })

        # 0) Straddle detection (unintentional)
        if self._is_straddle(portfolio) and not self.cfg.allow_straddle:
            plan = self._unwind_plan(portfolio, up_bid, up_ask, dn_bid, dn_ask, ts_ms, "UNWIND_STRADDLE")
            return Decision(
                kind=DecisionType.UNWIND_STRADDLE,
                reason="UNWIND_STRADDLE",
                size_mult=0.0,
                plan=plan,
                debug=dbg
            )

        # 1) If flat, set bias on first entry
        if gross <= 1e-9:
            self.window_bias = proposed_side
            self.last_flip_ts_ms = ts_ms
            # allow entry, but apply size decay (will be 1.0 here)
            return Decision(
                kind=DecisionType.APPROVE,
                reason="SET_BIAS_FIRST_ENTRY",
                size_mult=1.0,
                plan=None,
                debug=dbg
            )

        # 2) Apply sizing decay near caps
        decay = self._size_decay(portfolio)
        dbg["size_decay"] = decay
        if decay <= 0.0:
            # We're at caps: reject any additional adds (but allow flatten/unwind elsewhere)
            return Decision(
                kind=DecisionType.REJECT,
                reason="INV_CAP_REJECT",
                size_mult=0.0,
                plan=None,
                debug=dbg
            )

        # 3) If same as bias, approve (possibly reduced)
        if self.window_bias is None:
            # If bias wasn't set (shouldn't happen if you set on first entry), set now.
            self.window_bias = "UP" if net >= 0 else "DOWN"

        if proposed_side == self.window_bias:
            kind = DecisionType.APPROVE if decay >= 0.999 else DecisionType.APPROVE_REDUCED
            return Decision(
                kind=kind,
                reason="ADD_WITH_BIAS",
                size_mult=decay,
                plan=None,
                debug=dbg
            )

        # 4) Opposite to bias -> reversal discipline
        if not allow_flip:
            return Decision(
                kind=DecisionType.REJECT,
                reason="FLIP_DISABLED",
                size_mult=0.0,
                plan=None,
                debug=dbg
            )

        # Late window tighter reversal rules
        if T_sec <= self.cfg.late_t_sec:
            req_z = self.cfg.late_reversal_z
            req_conf = self.cfg.late_reversal_conf
        else:
            req_z = self.cfg.reversal_z
            req_conf = self.cfg.reversal_conf

        dbg.update({"rev_req_z": req_z, "rev_req_conf": req_conf})

        # cooldown prevents flip-flop
        if not self._cooldown_ok(ts_ms):
            return Decision(
                kind=DecisionType.REJECT,
                reason="REV_COOLDOWN",
                size_mult=0.0,
                plan=None,
                debug=dbg
            )

        if abs(z) < req_z or dominance < req_conf:
            return Decision(
                kind=DecisionType.REJECT,
                reason="REV_NOT_STRONG_ENOUGH",
                size_mult=0.0,
                plan=None,
                debug=dbg
            )

        # 5) Close then reverse (recommended): flatten bias leg first
        # Determine which leg currently dominates inventory
        # If we are net UP and proposed is DOWN -> sell UP first, then allow DOWN buy next tick.
        # If we are net DOWN and proposed is UP -> sell DOWN first.
        if net > 0 and proposed_side == "DOWN":
            # sell some or all UP at bid
            qty = min(up_inv, abs(net))  # close net first
            plan = (OrderPlan("UP", "SELL", qty, up_bid, "CLOSE_UP_THEN_REVERSE", {"ts_ms": ts_ms}),)
            # flip bias now (so once closed, next entry is aligned)
            self.window_bias = proposed_side
            self.last_flip_ts_ms = ts_ms
            return Decision(
                kind=DecisionType.CLOSE_THEN_REVERSE,
                reason="CLOSE_UP_THEN_REVERSE",
                size_mult=0.0,
                plan=plan,
                debug=dbg
            )

        if net < 0 and proposed_side == "UP":
            qty = min(dn_inv, abs(net))
            plan = (OrderPlan("DOWN", "SELL", qty, dn_bid, "CLOSE_DN_THEN_REVERSE", {"ts_ms": ts_ms}),)
            self.window_bias = proposed_side
            self.last_flip_ts_ms = ts_ms
            return Decision(
                kind=DecisionType.CLOSE_THEN_REVERSE,
                reason="CLOSE_DN_THEN_REVERSE",
                size_mult=0.0,
                plan=plan,
                debug=dbg
            )

        # If net is ~0 but opposite bias is proposed, allow reduced (rare case)
        self.window_bias = proposed_side
        self.last_flip_ts_ms = ts_ms
        return Decision(
            kind=DecisionType.APPROVE_REDUCED,
            reason="REV_APPROVED_NET_FLAT",
            size_mult=min(decay, 0.5),
            plan=None,
            debug=dbg
        )
