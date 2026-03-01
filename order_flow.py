"""
order_flow.py — Dual-Layer Order Flow Monitor

Layer 1 (Real-Time): FlowTracker
  Captures trade fills from the CLOB WebSocket and computes:
  - Rolling imbalance (UP buy pressure vs DOWN buy pressure)
  - Volume-weighted average price (VWAP) per token
  - Net flow signal for directional alpha

Layer 2 (On-Chain): GoldskyAnalytics
  Polls Polymarket's orderbook-subgraph on Goldsky every 30s for
  orderFilledEvent entities. Stores to logs/fills.jsonl for
  post-session calibration analysis (Brier scores, fill quality).

Usage in main.py:
  FLOW = FlowTracker()
  GOLDSKY = GoldskyAnalytics(api_key=os.getenv("GOLDSKY_API_KEY"))

  # On each trade event from CLOB WS:
  FLOW.record_fill(token_id, price, size, ts_ms, is_buy=True)

  # In brain loop:
  flow = FLOW.imbalance(window_s=30)  # -1..+1
"""

from __future__ import annotations
import os
import json
import time
import math
import logging
import asyncio
from dataclasses import dataclass, field
from collections import deque
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1: Real-Time Flow Tracker (CLOB WebSocket fills)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Fill:
    """A single recorded fill."""
    token_id: str
    price: float
    size: float
    ts_ms: int
    is_buy: bool  # True = aggressive buy (lifts ask), False = aggressive sell


class FlowTracker:
    """
    Institutional binary flow model.

    Produces three signals:
      - structural_flow  : persistent directional pressure
      - burst_flow       : short-term aggression spike
      - normalized_flow  : adjusted for binary compression

    Final imbalance() returns normalized_flow.
    """

    def __init__(
        self,
        up_token_id: str = "",
        down_token_id: str = "",
        max_fills: int = 4000,
    ):
        self.up_token_id = up_token_id
        self.down_token_id = down_token_id
        self._fills: deque[Fill] = deque(maxlen=max_fills)

    def set_tokens(self, up_token_id: str, down_token_id: str) -> None:
        self.up_token_id = up_token_id
        self.down_token_id = down_token_id

    def record_fill(
        self,
        token_id: str,
        price: float,
        size: float,
        ts_ms: int,
        is_buy: bool = True,
    ) -> None:
        self._fills.append(Fill(
            token_id=token_id,
            price=price,
            size=size,
            ts_ms=ts_ms,
            is_buy=is_buy,
        ))

    # ─────────────────────────────────────────────────────

    def _decayed_fills(self, window_s: float, tau_s: float):
        now_ms = int(time.time() * 1000)
        cutoff = now_ms - int(window_s * 1000)
        tau_ms = tau_s * 1000.0

        for f in list(self._fills):
            if f.ts_ms < cutoff:
                continue
            age = max(0, now_ms - f.ts_ms)
            weight = math.exp(-age / tau_ms)
            yield f, weight

    # ─────────────────────────────────────────────────────
    # Structural Pressure
    # ─────────────────────────────────────────────────────

    def structural_flow(self, window_s: float = 30.0) -> float:
        """
        Persistent directional pressure.

        Bullish = UP buys + DOWN sells
        Bearish = DOWN buys + UP sells
        """
        bullish = 0.0
        bearish = 0.0

        # Tau scaled to window
        tau_s = max(4.0, window_s / 3.0)

        for f, w in self._decayed_fills(window_s, tau_s):
            vol = f.size * w

            if f.token_id == self.up_token_id:
                if f.is_buy:
                    bullish += vol
                else:
                    bearish += vol
            elif f.token_id == self.down_token_id:
                if f.is_buy:
                    bearish += vol
                else:
                    bullish += vol

        total = bullish + bearish
        if total < 0.001:
            return 0.0

        return 0.0 if total == 0 else (bullish - bearish) / total

    # ─────────────────────────────────────────────────────
    # Burst Detection (Short-term Aggression)
    # ─────────────────────────────────────────────────────

    def burst_flow(self, window_s: float = 5.0) -> float:
        """
        Short-term aggression spike.
        Much shorter decay.
        """
        bullish = 0.0
        bearish = 0.0

        tau_s = 2.5  # very short memory

        for f, w in self._decayed_fills(window_s, tau_s):
            vol = f.size * w

            # Keep directional semantics consistent with structural_flow:
            # UP buys / DOWN sells = bullish, DOWN buys / UP sells = bearish.
            if f.token_id == self.up_token_id:
                if f.is_buy:
                    bullish += vol
                else:
                    bearish += vol
            elif f.token_id == self.down_token_id:
                if f.is_buy:
                    bearish += vol
                else:
                    bullish += vol

        total = bullish + bearish
        if total < 0.001:
            return 0.0

        return 0.0 if total == 0 else (bullish - bearish) / total

    # ─────────────────────────────────────────────────────
    # Probability Compression Adjustment
    # ─────────────────────────────────────────────────────

    def normalized_flow(
        self,
        up_price: float,
        down_price: float,
        window_s: float = 30.0
    ) -> float:
        """
        Adjust structural flow for binary compression near 0/1.

        When a token trades near 0.99 or 0.01,
        raw directional pressure becomes mechanically skewed.
        We normalize by probability distance from 0.5.
        """
        raw = self.structural_flow(window_s)

        # Compression factor:
        # Near 0.5 → 1.0
        # Near extremes → reduced influence
        compression = 1.0 - abs(up_price - 0.5) * 1.8
        compression = max(0.3, compression)

        return raw * compression

    # ─────────────────────────────────────────────────────
    # Final Public Signal
    # ─────────────────────────────────────────────────────

    def imbalance(
        self,
        up_price: float = 0.5,
        down_price: float = 0.5,
        window_s: float = 30.0
    ) -> float:
        """
        Final institutional flow signal.

        Combines:
          - structural_flow (persistent pressure)
          - burst_flow (short aggression)
          - compression normalization
        """
        structural = self.normalized_flow(up_price, down_price, window_s)
        burst = self.burst_flow(min(window_s, 6.0))

        # Burst influences but does not dominate
        final = structural * 0.75 + burst * 0.25

        # Prevent saturation
        return max(-0.95, min(0.95, final))

    # ─────────────────────────────────────────────────────

    def trade_velocity(self, window_s: float = 10.0) -> float:
        fills = [f for f in self._fills
                 if f.ts_ms >= int(time.time() * 1000) - window_s * 1000]
        return len(fills) / window_s if window_s > 0 else 0.0

    def flow_bias(self, window_s: float = 10.0) -> float:
        """Directional flow bias for adverse selection model.
        Delegates to burst_flow for short-term aggression detection."""
        return self.burst_flow(window_s)

    def summary(self, up_price: float = 0.5, down_price: float = 0.5) -> dict:
        return {
            "structural": round(self.structural_flow(), 4),
            "burst": round(self.burst_flow(), 4),
            "normalized": round(self.normalized_flow(up_price, down_price), 4),
            "final_flow": round(self.imbalance(up_price, down_price), 4),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2: On-Chain Analytics (Goldsky Subgraph)
# ─────────────────────────────────────────────────────────────────────────────

# Polymarket orderbook subgraph on Goldsky
GOLDSKY_SUBGRAPH_URL = (
    "https://api.goldsky.com/api/public/"
    "project_cl6mb8i9h0003e201j6li0diw/"
    "subgraphs/orderbook-subgraph/0.0.1/gn"
)

# GraphQL query for recent fills
FILLS_QUERY = """
query RecentFills($market: String!, $first: Int!, $skip: Int!) {
  orderFilledEvents(
    where: { market: $market }
    orderBy: timestamp
    orderDirection: desc
    first: $first
    skip: $skip
  ) {
    id
    timestamp
    maker
    taker
    makerAssetId
    takerAssetId
    makerAmountFilled
    takerAmountFilled
    fee
    transactionHash
  }
}
"""

# Fallback: query by asset IDs (token IDs) if market ID doesn't work
FILLS_BY_ASSET_QUERY = """
query RecentFillsByAsset($assetId: String!, $first: Int!) {
  orderFilledEvents(
    where: { makerAssetId: $assetId }
    orderBy: timestamp
    orderDirection: desc
    first: $first
  ) {
    id
    timestamp
    maker
    taker
    makerAssetId
    takerAssetId
    makerAmountFilled
    takerAmountFilled
    fee
    transactionHash
  }
}
"""


class GoldskyAnalytics:
    """
    Polls Polymarket's Goldsky-hosted orderbook subgraph for
    on-chain fill events. Writes to logs/fills.jsonl.

    Used for:
    - Post-trade calibration (Brier score analysis)
    - Fill quality tracking (intended vs. actual fill prices)
    - On-chain verification of order execution
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        poll_interval_s: float = 30.0,
        subgraph_url: Optional[str] = None,
    ):
        self._api_key = api_key or os.getenv("GOLDSKY_API_KEY", "")
        self._poll_s = poll_interval_s
        self._url = subgraph_url or GOLDSKY_SUBGRAPH_URL
        self._last_seen_id: Optional[str] = None
        self._fills_path = "logs/fills.jsonl"
        self._market_id: str = ""
        self._up_token_id: str = ""
        self._down_token_id: str = ""
        self._total_fetched: int = 0

        os.makedirs("logs", exist_ok=True)

    def set_market(
        self, market_id: str, up_token_id: str, down_token_id: str
    ) -> None:
        """Set the current market context."""
        self._market_id = market_id
        self._up_token_id = up_token_id
        self._down_token_id = down_token_id

    async def poll_task(self) -> None:
        """
        Background task: polls Goldsky subgraph every poll_interval_s seconds.
        Writes new fills to logs/fills.jsonl.
        """
        if not self._api_key:
            logger.warning("GOLDSKY: No API key — on-chain analytics disabled.")
            return

        logger.info(f"GOLDSKY: Analytics polling started (every {self._poll_s}s)")

        while True:
            await asyncio.sleep(self._poll_s)
            try:
                await self._fetch_and_store()
            except Exception as e:
                logger.warning(f"GOLDSKY poll error: {e}")

    async def _fetch_and_store(self) -> None:
        """Fetch recent fills from Goldsky subgraph and store new ones."""
        import aiohttp

        headers = {
            "Content-Type": "application/json",
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        # Try querying by market, fall back to asset
        if self._market_id:
            payload = {
                "query": FILLS_QUERY,
                "variables": {
                    "market": self._market_id,
                    "first": 50,
                    "skip": 0,
                },
            }
        elif self._up_token_id:
            payload = {
                "query": FILLS_BY_ASSET_QUERY,
                "variables": {
                    "assetId": self._up_token_id,
                    "first": 50,
                },
            }
        else:
            return  # No market set yet

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._url, json=payload, headers=headers, timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status != 200:
                        logger.warning(f"GOLDSKY: HTTP {resp.status}")
                        return
                    data = await resp.json()
        except Exception as e:
            logger.warning(f"GOLDSKY fetch error: {e}")
            return

        fills = (
            data.get("data", {}).get("orderFilledEvents", [])
            if data.get("data") else []
        )

        if not fills:
            return

        new_count = 0
        with open(self._fills_path, "a") as f:
            for fill in fills:
                fill_id = fill.get("id", "")
                if fill_id == self._last_seen_id:
                    break

                # Enrich with token labels
                maker_asset = fill.get("makerAssetId", "")
                fill["token_label"] = (
                    "UP" if maker_asset == self._up_token_id
                    else "DOWN" if maker_asset == self._down_token_id
                    else "UNKNOWN"
                )

                f.write(json.dumps(fill) + "\n")
                new_count += 1

        if fills:
            self._last_seen_id = fills[0].get("id", "")

        self._total_fetched += new_count
        if new_count > 0:
            logger.info(
                f"GOLDSKY: {new_count} new on-chain fills "
                f"(total={self._total_fetched})"
            )
