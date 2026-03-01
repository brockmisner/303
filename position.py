"""
position.py

Inventory + PnL accounting for binary (0..1) outcome shares (e.g., Polymarket YES/NO or UP/DOWN).

- FIFO lot accounting by default (optionally average-cost)
- Realized/unrealized PnL in quote currency (e.g., USDC)
- Fee handling: pass fee_per_share at fill time (or 0.0)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, List, Optional, Dict
from collections import deque
import time


class Side(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass
class Fill:
    ts_ms: int
    side: Side
    qty: float          # shares, >0
    price: float        # 0..1
    fee: float = 0.0    # total fee (quote currency)
    meta: Dict[str, object] = field(default_factory=dict)


@dataclass
class Lot:
    qty: float
    price: float
    ts_ms: int
    fee: float = 0.0    # remaining entry fee allocated to remaining qty


class Position:
    """
    Accounting for a single outcome token (e.g., UP token).
    Instantiate two Positions (UP/DOWN) for a full market.
    """

    def __init__(self, symbol: str, *, lot_method: str = "FIFO", contract_multiplier: float = 1.0):
        self.symbol = symbol
        self.lot_method = lot_method.upper()
        if self.lot_method not in ("FIFO", "AVG"):
            raise ValueError("lot_method must be 'FIFO' or 'AVG'")
        self.contract_multiplier = float(contract_multiplier)

        self._lots: Deque[Lot] = deque()
        self._avg_qty: float = 0.0
        self._avg_price: float = 0.0
        self._avg_fee: float = 0.0

        self.realized_pnl: float = 0.0
        self.total_fees: float = 0.0
        self.entry_fees: float = 0.0
        self.exit_fees: float = 0.0

        self.fills: List[Fill] = []

    # ----- inventory properties -----

    @property
    def qty(self) -> float:
        if self.lot_method == "FIFO":
            return sum(l.qty for l in self._lots)
        return self._avg_qty

    @property
    def inventory(self) -> float:
        """Alias for qty — used by inventory_layer.py."""
        return self.qty

    @property
    def avg_entry_price(self) -> float:
        q = self.qty
        if q <= 0:
            return 0.0
        if self.lot_method == "FIFO":
            notional = sum(l.qty * l.price for l in self._lots)
            return notional / q
        return self._avg_price

    def _open_cost_ex_fee(self) -> float:
        if self.lot_method == "FIFO":
            return sum(l.qty * l.price for l in self._lots)
        return self._avg_qty * self._avg_price

    def _open_entry_fee(self) -> float:
        if self.lot_method == "FIFO":
            return sum(l.fee for l in self._lots)
        return self._avg_fee

    @property
    def avg_entry_incl_fee(self) -> float:
        q = self.qty
        if q <= 0:
            return 0.0
        return (self._open_cost_ex_fee() + self._open_entry_fee()) / q

    # ----- fills -----

    def apply_fill(
        self,
        *,
        side: Side,
        qty: float,
        price: float,
        ts_ms: Optional[int] = None,
        fee_per_share: float = 0.0,
        meta: Optional[Dict[str, object]] = None,
    ) -> Fill:
        if ts_ms is None:
            ts_ms = int(time.time() * 1000)
        if meta is None:
            meta = {}

        qty = float(qty)
        price = float(price)
        if qty <= 0:
            raise ValueError("qty must be > 0")
        if not (0.0 <= price <= 1.0):
            raise ValueError("price must be in [0,1]")

        fee_total = float(fee_per_share) * qty
        self.total_fees += fee_total

        f = Fill(ts_ms=ts_ms, side=side, qty=qty, price=price, fee=fee_total, meta=meta)
        self.fills.append(f)

        if side == Side.BUY:
            self.entry_fees += fee_total
            self._buy(qty, price, fee_total, ts_ms)
        else:
            self.exit_fees += fee_total
            self._sell(qty, price, fee_total)

        return f

    def _buy(self, qty: float, price: float, fee_total: float, ts_ms: int) -> None:
        if self.lot_method == "FIFO":
            self._lots.append(Lot(qty=qty, price=price, ts_ms=ts_ms, fee=fee_total))
            return

        prev_qty = self._avg_qty
        prev_cost = prev_qty * self._avg_price
        new_cost = prev_cost + qty * price
        new_qty = prev_qty + qty
        self._avg_qty = new_qty
        self._avg_price = (new_cost / new_qty) if new_qty > 0 else 0.0
        self._avg_fee += fee_total

    def _sell(self, qty: float, price: float, exit_fee_total: float) -> None:
        if qty > self.qty + 1e-9:
            raise ValueError(f"Sell qty {qty} exceeds open qty {self.qty} for {self.symbol}")

        realized_gross = 0.0
        released_entry_fee = 0.0

        if self.lot_method == "FIFO":
            remaining = qty
            while remaining > 1e-12:
                if not self._lots:
                    break
                lot = self._lots[0]
                take = min(lot.qty, remaining)

                fee_take = (lot.fee * (take / lot.qty)) if lot.qty > 0 else 0.0
                pnl_piece = (price - lot.price) * take * self.contract_multiplier

                realized_gross += pnl_piece
                released_entry_fee += fee_take

                lot.qty -= take
                lot.fee -= fee_take
                remaining -= take

                if lot.qty <= 1e-12:
                    self._lots.popleft()

        else:
            take = qty
            pnl_piece = (price - self._avg_price) * take * self.contract_multiplier
            realized_gross += pnl_piece

            fee_take = (self._avg_fee * (take / self._avg_qty)) if self._avg_qty > 0 else 0.0
            released_entry_fee += fee_take

            self._avg_qty -= take
            self._avg_fee -= fee_take
            if self._avg_qty <= 1e-12:
                self._avg_qty = 0.0
                self._avg_price = 0.0
                self._avg_fee = 0.0

        # Net realized subtracts both:
        # - exit fee paid now
        # - entry fee allocated to shares closed (entry fee is part of true cost)
        self.realized_pnl += realized_gross - exit_fee_total - released_entry_fee

    # ----- mark-to-market -----

    def unrealized_pnl(
        self,
        *,
        mid: Optional[float] = None,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        include_exit_fee_estimate: bool = False,
        exit_fee_per_share: float = 0.0,
        use: str = "MID",
    ) -> float:
        q = self.qty
        if q <= 0:
            return 0.0

        if mid is None:
            if bid is None or ask is None:
                raise ValueError("Provide mid or bid+ask")
            u = use.upper()
            if u == "MID":
                mid = 0.5 * (bid + ask)
            elif u == "BID":
                mid = bid
            elif u == "ASK":
                mid = ask
            else:
                raise ValueError("use must be MID/BID/ASK")

        mid = float(mid)
        if not (0.0 <= mid <= 1.0):
            raise ValueError("mark must be in [0,1]")

        basis = self._open_cost_ex_fee() + self._open_entry_fee()
        mkt = q * mid * self.contract_multiplier
        u = mkt - basis

        if include_exit_fee_estimate and exit_fee_per_share > 0:
            u -= q * float(exit_fee_per_share)

        return u

    def snapshot(self, *, mid: Optional[float] = None, bid: Optional[float] = None, ask: Optional[float] = None) -> Dict[str, float]:
        u = self.unrealized_pnl(mid=mid, bid=bid, ask=ask) if (mid is not None or (bid is not None and ask is not None)) else 0.0
        return {
            "qty": self.qty,
            "avg_entry": self.avg_entry_price,
            "avg_entry_incl_fee": self.avg_entry_incl_fee,
            "open_cost_ex_fee": self._open_cost_ex_fee(),
            "open_entry_fee": self._open_entry_fee(),
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": u,
            "total_pnl": self.realized_pnl + u,
            "total_fees": self.total_fees,
            "entry_fees": self.entry_fees,
            "exit_fees": self.exit_fees,
        }


class Portfolio:
    """Convenience wrapper for two-sided binary market (UP + DOWN)."""
    def __init__(self, up: Position, down: Position):
        self.up = up
        self.down = down

    def total_realized(self) -> float:
        return self.up.realized_pnl + self.down.realized_pnl

    def total_unrealized(self, up_mid: float, down_mid: float) -> float:
        return self.up.unrealized_pnl(mid=up_mid) + self.down.unrealized_pnl(mid=down_mid)

    def total_pnl(self, up_mid: float, down_mid: float) -> float:
        return self.total_realized() + self.total_unrealized(up_mid, down_mid)

    def net_delta_shares(self) -> float:
        return self.up.qty - self.down.qty

    def snapshot(self, up_mid: float, down_mid: float) -> Dict[str, object]:
        return {
            "UP": self.up.snapshot(mid=up_mid),
            "DOWN": self.down.snapshot(mid=down_mid),
            "net_delta_shares": self.net_delta_shares(),
            "total_realized": self.total_realized(),
            "total_unrealized": self.total_unrealized(up_mid, down_mid),
            "total_pnl": self.total_pnl(up_mid, down_mid),
        }
