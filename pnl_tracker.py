# pnl_tracker.py — CLOB-verified PnL tracking
"""
Tracks REAL PnL by snapshotting actual on-chain balances via the
Polymarket CLOB API at window start/end.

Three sources of truth, all from the CLOB:
  1. USDC balance  — get_balance_allowance(COLLATERAL)
  2. UP token holdings — get_balance_allowance(CONDITIONAL, up_token)
  3. DN token holdings — get_balance_allowance(CONDITIONAL, dn_token)

Real PnL per window = (end_equity - start_equity)
where equity = usdc + up_tokens * up_bid + dn_tokens * dn_bid

This catches everything the model-based PnL misses:
  - Phantom fills (bot thinks it owns tokens it doesn't)
  - Missed settlement credits
  - Fee discrepancies
  - Position drift from partial fills
"""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque
from typing import Optional
import time
import json
import logging

logger = logging.getLogger("pnl_tracker")


@dataclass
class BalanceSnapshot:
    ts_ms: int
    usdc: float
    up_tokens: float
    dn_tokens: float
    up_bid: float          # book bid at snapshot time
    dn_bid: float
    equity: float          # usdc + up_tokens*up_bid + dn_tokens*dn_bid
    source: str = "clob"


@dataclass
class WindowPnL:
    window_id: str
    start: BalanceSnapshot
    end: BalanceSnapshot
    real_pnl: float            # end.equity - start.equity (CLOB truth)
    model_pnl: float           # from settle_window() model
    drift: float               # real - model (should be ~0)
    up_token_drift: float      # actual UP tokens - FIFO UP inventory
    dn_token_drift: float      # actual DN tokens - FIFO DN inventory


class PnLTracker:

    def __init__(self, log_path: str = "logs/pnl_verified.jsonl"):
        self.log_path = log_path
        self._start_snap: Optional[BalanceSnapshot] = None
        self._window_id: str = ""
        self.session_real_pnl: float = 0.0
        self.session_model_pnl: float = 0.0
        self.window_history: deque = deque(maxlen=500)
        self.win_count: int = 0
        self.loss_count: int = 0

    def _fetch_balances(self, client, up_token_id: str, dn_token_id: str) -> dict:
        """Fetch USDC + token balances from CLOB. Returns raw dict."""
        from py_clob_client.clob_types import BalanceAllowanceParams, AssetType

        usdc_resp = client.get_balance_allowance(
            params=BalanceAllowanceParams(asset_type=AssetType.COLLATERAL)
        )
        usdc = float(usdc_resp.get("balance", 0) or 0) / 1e6

        up_resp = client.get_balance_allowance(
            params=BalanceAllowanceParams(
                asset_type=AssetType.CONDITIONAL,
                token_id=up_token_id,
            )
        )
        up_tokens = float(up_resp.get("balance", 0) or 0) / 1e6

        dn_resp = client.get_balance_allowance(
            params=BalanceAllowanceParams(
                asset_type=AssetType.CONDITIONAL,
                token_id=dn_token_id,
            )
        )
        dn_tokens = float(dn_resp.get("balance", 0) or 0) / 1e6

        return {"usdc": usdc, "up_tokens": up_tokens, "dn_tokens": dn_tokens}

    def snapshot(
        self,
        client,
        up_token_id: str,
        dn_token_id: str,
        up_bid: float,
        dn_bid: float,
    ) -> Optional[BalanceSnapshot]:
        """Take a balance snapshot from the CLOB."""
        try:
            bals = self._fetch_balances(client, up_token_id, dn_token_id)
            equity = (
                bals["usdc"]
                + bals["up_tokens"] * up_bid
                + bals["dn_tokens"] * dn_bid
            )
            snap = BalanceSnapshot(
                ts_ms=int(time.time() * 1000),
                usdc=bals["usdc"],
                up_tokens=bals["up_tokens"],
                dn_tokens=bals["dn_tokens"],
                up_bid=up_bid,
                dn_bid=dn_bid,
                equity=round(equity, 4),
            )
            logger.info(
                f"PNL_SNAP: usdc=${snap.usdc:.2f} "
                f"up={snap.up_tokens:.0f} dn={snap.dn_tokens:.0f} "
                f"equity=${snap.equity:.2f}"
            )
            return snap
        except Exception as e:
            logger.warning(f"PNL_SNAP_FAIL: {e}")
            return None

    def on_window_start(
        self,
        client,
        up_token_id: str,
        dn_token_id: str,
        up_bid: float,
        dn_bid: float,
        window_id: str = "",
    ):
        """Call at window start to record opening balances."""
        self._window_id = window_id
        self._start_snap = self.snapshot(
            client, up_token_id, dn_token_id, up_bid, dn_bid
        )

    def on_window_end(
        self,
        client,
        up_token_id: str,
        dn_token_id: str,
        up_bid: float,
        dn_bid: float,
        model_pnl: float,
        fifo_up_inventory: float,
        fifo_dn_inventory: float,
    ) -> Optional[WindowPnL]:
        """
        Call at window end (before reset).
        Compares CLOB reality to model assumptions.
        """
        if self._start_snap is None:
            logger.warning("PNL_VERIFY: no start snapshot — skipping")
            return None

        end_snap = self.snapshot(
            client, up_token_id, dn_token_id, up_bid, dn_bid
        )
        if end_snap is None:
            return None

        real_pnl = end_snap.equity - self._start_snap.equity
        drift = real_pnl - model_pnl
        up_drift = end_snap.up_tokens - fifo_up_inventory
        dn_drift = end_snap.dn_tokens - fifo_dn_inventory

        result = WindowPnL(
            window_id=self._window_id,
            start=self._start_snap,
            end=end_snap,
            real_pnl=round(real_pnl, 4),
            model_pnl=round(model_pnl, 4),
            drift=round(drift, 4),
            up_token_drift=round(up_drift, 2),
            dn_token_drift=round(dn_drift, 2),
        )

        # Update session totals
        self.session_real_pnl += real_pnl
        self.session_model_pnl += model_pnl
        if real_pnl > 0:
            self.win_count += 1
        elif real_pnl < 0:
            self.loss_count += 1
        self.window_history.append(result)

        # Log
        drift_flag = " ⚠️ DRIFT" if abs(drift) > 0.50 else ""
        pos_flag = ""
        if abs(up_drift) > 0.5 or abs(dn_drift) > 0.5:
            pos_flag = f" ⚠️ POS_DRIFT(up={up_drift:+.0f} dn={dn_drift:+.0f})"

        logger.info(
            f"PNL_VERIFY: real=${real_pnl:+.2f} model=${model_pnl:+.2f} "
            f"drift=${drift:+.2f}{drift_flag}{pos_flag} | "
            f"session: real=${self.session_real_pnl:+.2f} "
            f"model=${self.session_model_pnl:+.2f} "
            f"W/L={self.win_count}/{self.loss_count}"
        )

        if abs(drift) > 0.50:
            logger.warning(
                f"PNL_DRIFT_ALERT: ${drift:+.2f} discrepancy! "
                f"start_equity=${self._start_snap.equity:.2f} "
                f"end_equity=${end_snap.equity:.2f} "
                f"model_pnl=${model_pnl:+.2f}"
            )

        # Append to JSONL log
        try:
            with open(self.log_path, "a") as f:
                f.write(json.dumps({
                    "ts": end_snap.ts_ms,
                    "window": self._window_id,
                    "real_pnl": result.real_pnl,
                    "model_pnl": result.model_pnl,
                    "drift": result.drift,
                    "start_equity": self._start_snap.equity,
                    "end_equity": end_snap.equity,
                    "start_usdc": self._start_snap.usdc,
                    "end_usdc": end_snap.usdc,
                    "up_token_drift": result.up_token_drift,
                    "dn_token_drift": result.dn_token_drift,
                    "session_real": round(self.session_real_pnl, 4),
                    "session_model": round(self.session_model_pnl, 4),
                    "win_count": self.win_count,
                    "loss_count": self.loss_count,
                }) + "\n")
        except Exception as e:
            logger.warning(f"PNL_LOG_FAIL: {e}")

        return result
