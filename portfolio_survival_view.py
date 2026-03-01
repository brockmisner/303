# portfolio_survival_view.py
"""
Drop-in adapter: exposes Portfolio inventory as a flat SurvivalView
that the existing survival engine expects (same attribute names as
the legacy PositionState: side, token_id, up_shares, down_shares, etc.).

Usage:
    from portfolio_survival_view import build_survival_view
    sv = build_survival_view(PORTFOLIO, UP_TOKEN_ID, DOWN_TOKEN_ID, p_now, z_now)
    # sv.side, sv.shares, sv.entry_price, etc. — all available
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from position import Portfolio


@dataclass
class SurvivalView:
    side: Optional[str]          # "UP" or "DOWN" dominant inventory
    token_id: Optional[str]      # UP_TOKEN_ID or DOWN_TOKEN_ID
    up_shares: float
    down_shares: float
    shares: float
    entry_price: float           # avg cost proxy (dominant side)
    entry_p: float               # last p at "entry"
    entry_z: float               # last z at "entry"
    max_p: float
    min_p: float
    active: bool = False
    locked: bool = False
    entry_edge: float = 0.0
    entry_ts_ms: int = 0


def build_survival_view(
    portfolio: Portfolio,
    up_token_id: str,
    down_token_id: str,
    last_p: float,
    last_z: float,
) -> SurvivalView:
    up = float(portfolio.up.inventory)
    dn = float(portfolio.down.inventory)
    total = up + dn

    if total <= 0:
        return SurvivalView(
            side=None, token_id=None,
            up_shares=0.0, down_shares=0.0, shares=0.0,
            entry_price=0.0, entry_p=last_p, entry_z=last_z,
            max_p=last_p, min_p=last_p, active=False,
        )

    side = "UP" if up >= dn else "DOWN"
    token_id = up_token_id if side == "UP" else down_token_id

    # Entry price proxy: average cost of dominant side's FIFO lots
    dom_pos = portfolio.up if side == "UP" else portfolio.down
    avg_cost = dom_pos.avg_entry_price
    if avg_cost <= 0:
        avg_cost = 0.5  # fallback

    dom_shares = up if side == "UP" else dn

    return SurvivalView(
        side=side,
        token_id=token_id,
        up_shares=up,
        down_shares=dn,
        shares=dom_shares,
        entry_price=float(avg_cost),
        entry_p=last_p,
        entry_z=last_z,
        max_p=last_p,
        min_p=last_p,
        active=True,
    )
