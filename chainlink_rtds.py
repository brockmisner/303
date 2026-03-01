"""
chainlink_rtds.py — Real-Time Data Socket for Polymarket's Chainlink BTC/USD feed.

THIS IS THE SINGLE MOST IMPORTANT MODULE IN THE BOT.

Polymarket 5-min BTC markets resolve against Chainlink Data Streams BTC/USD.
Using Binance/Coinbase prices introduces a $20-60 basis error that destroys edge.

This module connects to Polymarket's public RTDS websocket and provides
the exact price the market will resolve against.

Endpoint: wss://ws-live-data.polymarket.com
Topic:    crypto_prices_chainlink
Symbol:   btc/usd
"""

import asyncio
import json
import time
import logging
from dataclasses import dataclass
from typing import Optional, Callable
from collections import deque

import websockets

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

logger = logging.getLogger(__name__)


def ms_now() -> int:
    return int(time.time() * 1000)


# ─────────────────────────────────────────────────────────────────────────────
# RTDS State
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RTDSState:
    price:         float = float('nan')
    timestamp_ms:  int   = 0
    recv_ts_ms:    int   = 0
    updates:       int   = 0
    reconnects:    int   = 0
    last_error:    str   = ""
    raw_messages:  int   = 0

    def is_fresh(self, max_age_ms: int = 45_000) -> bool:
        if self.recv_ts_ms == 0:
            return False
        return (ms_now() - self.recv_ts_ms) < max_age_ms

    def age_ms(self) -> int:
        if self.recv_ts_ms == 0:
            return 999_999
        return ms_now() - self.recv_ts_ms


RTDS = RTDSState()
_rtds_lock = asyncio.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# Subscription Variants (rotate on failure)
# ─────────────────────────────────────────────────────────────────────────────

RTDS_WS_URL = "wss://ws-live-data.polymarket.com"

SUBSCRIBE_VARIANTS = [
    # V0: Chainlink BTC/USD with JSON filter (per official docs)
    {
        "action": "subscribe",
        "subscriptions": [
            {
                "topic": "crypto_prices_chainlink",
                "type": "*",
                "filters": json.dumps({"symbol": "btc/usd"}),
            }
        ],
    },
    # V1: Chainlink ALL symbols (empty filters per official docs)
    {
        "action": "subscribe",
        "subscriptions": [
            {
                "topic": "crypto_prices_chainlink",
                "type": "*",
                "filters": "",
            }
        ],
    },
    # V2: Binance BTC/USDT (different topic, as documented)
    {
        "action": "subscribe",
        "subscriptions": [
            {
                "topic": "crypto_prices",
                "type": "update",
                "filters": "btcusdt",
            }
        ],
    },
]

WS_HEADERS = {
    "Origin": "https://polymarket.com",
    "User-Agent": "Mozilla/5.0 (compatible; PolyBot/2.0)",
}


# ─────────────────────────────────────────────────────────────────────────────
# Message Parsing
# ─────────────────────────────────────────────────────────────────────────────

def _extract_btc_price(d: dict) -> Optional[float]:
    """Extract BTC/USD price from a dict."""
    symbol = str(d.get("symbol", "")).lower().replace(" ", "").replace("_", "")
    is_btc = symbol in ("btc/usd", "btcusd", "btc-usd", "btc", "btcusdt")

    # If there's a symbol and it's not BTC, skip
    if "symbol" in d and not is_btc:
        return None

    # Prefer full_accuracy_value (higher decimal precision from Chainlink)
    full_val = d.get("full_accuracy_value")
    if full_val is not None:
        try:
            price = float(full_val)
            if 1000 < price < 1_000_000:
                return price
        except (ValueError, TypeError):
            pass

    for key in ("value", "price", "last_price", "mark_price"):
        val = d.get(key)
        if val is not None:
            try:
                price = float(val)
                if 1000 < price < 1_000_000:
                    return price
            except (ValueError, TypeError):
                continue
    return None


def _parse_rtds_message(raw: str) -> Optional[dict]:
    """
    Parse incoming RTDS message. Handles multiple payload shapes.
    Returns {"price": float, "ts_ms": int} or None.
    """
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return None

    items = data if isinstance(data, list) else [data]

    for item in items:
        if not isinstance(item, dict):
            continue

        # Shape A: {"topic": ..., "payload": {"symbol": "btc/usd", "value": 67234.50, ...}}
        if "payload" in item:
            payload = item["payload"]
            if isinstance(payload, dict):
                price = _extract_btc_price(payload)
                if price is not None:
                    try:
                        _ts = int(float(payload.get("timestamp", ms_now())))
                    except (ValueError, TypeError):
                        _ts = ms_now()
                    return {"price": price, "ts_ms": _ts}

        # Shape B: {"data": {...}} or {"data": [{...}]}
        if "data" in item:
            d = item["data"]
            if isinstance(d, dict):
                price = _extract_btc_price(d)
                if price is not None:
                    try:
                        _ts = int(float(d.get("timestamp", ms_now())))
                    except (ValueError, TypeError):
                        _ts = ms_now()
                    return {"price": price, "ts_ms": _ts}
            elif isinstance(d, list):
                for sub in d:
                    if isinstance(sub, dict):
                        price = _extract_btc_price(sub)
                        if price is not None:
                            try:
                                _ts = int(float(sub.get("timestamp", ms_now())))
                            except (ValueError, TypeError):
                                _ts = ms_now()
                            return {"price": price, "ts_ms": _ts}

        # Shape C: flat top-level
        price = _extract_btc_price(item)
        if price is not None:
            try:
                _ts = int(float(item.get("timestamp", ms_now())))
            except (ValueError, TypeError):
                _ts = ms_now()
            return {"price": price, "ts_ms": _ts}

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Chainlink REST Polling (fallback when WS is rate-limited)
# ─────────────────────────────────────────────────────────────────────────────

CHAINLINK_REST_URL = (
    "https://data.chain.link/api/query-timescale"
    "?query=LIVE_STREAM_REPORTS_QUERY"
    "&variables=%7B%22feedId%22%3A%220x00039d9e45394f473ab1f050a1b963e6b05351e52d71e507509ada0c95ed75b8%22%7D"
)


async def _chainlink_rest_poller(
    on_price: Optional[Callable[[float, int], None]] = None,
) -> None:
    """
    Poll Chainlink Data Streams REST API every ~1s.
    Runs in parallel with the WS task. Prices are 18-decimal fixed point.
    Only updates RTDS if the price is newer than what we already have.
    """
    if not HAS_AIOHTTP:
        logger.warning("RTDS_REST: aiohttp not installed — REST poller disabled")
        return

    logger.info("RTDS_REST: starting Chainlink REST poller")
    poll_interval = 1.0
    consecutive_errors = 0

    async with aiohttp.ClientSession() as session:
        while True:
            try:
                async with session.get(
                    CHAINLINK_REST_URL, timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status == 429:
                        poll_interval = min(poll_interval * 2, 30)
                        logger.warning(f"RTDS_REST: 429 rate limit, backoff {poll_interval:.0f}s")
                        await asyncio.sleep(poll_interval)
                        continue

                    if resp.status != 200:
                        consecutive_errors += 1
                        await asyncio.sleep(poll_interval)
                        continue

                    data = await resp.json()
                    nodes = (
                        data.get("data", {})
                        .get("liveStreamReports", {})
                        .get("nodes", [])
                    )
                    if not nodes:
                        await asyncio.sleep(poll_interval)
                        continue

                    # Latest node is first
                    node = nodes[0]
                    raw_price = node.get("price", "0")
                    ts_str = node.get("validFromTimestamp", "")

                    # Parse 18-decimal fixed point → float USD
                    price = float(raw_price) / 1e18
                    if not (1000 < price < 1_000_000):
                        await asyncio.sleep(poll_interval)
                        continue

                    # Parse ISO timestamp → ms
                    try:
                        from datetime import datetime, timezone
                        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                        ts_ms = int(dt.timestamp() * 1000)
                    except Exception:
                        ts_ms = ms_now()

                    recv = ms_now()

                    # Only update price if newer, but always refresh recv_ts_ms
                    async with _rtds_lock:
                        if ts_ms > RTDS.timestamp_ms:
                            RTDS.price = price
                            RTDS.timestamp_ms = ts_ms
                            RTDS.updates += 1

                            if on_price:
                                try:
                                    on_price(price, ts_ms)
                                except Exception:
                                    pass

                            if RTDS.updates <= 3 or RTDS.updates % 200 == 0:
                                logger.info(
                                    f"RTDS_REST: ${price:.2f} | total={RTDS.updates} | "
                                    f"age={RTDS.age_ms()}ms"
                                )

                        # Always mark as fresh — we confirmed the endpoint is alive
                        RTDS.recv_ts_ms = recv

                    consecutive_errors = 0
                    poll_interval = 1.0  # reset backoff on success

            except asyncio.CancelledError:
                raise
            except Exception as e:
                consecutive_errors += 1
                if consecutive_errors <= 3 or consecutive_errors % 50 == 0:
                    logger.error(f"RTDS_REST: {type(e).__name__}: {e}")
                poll_interval = min(poll_interval * 1.5, 15)

            await asyncio.sleep(poll_interval)


# ─────────────────────────────────────────────────────────────────────────────
# Main RTDS Task (WS + REST in parallel)
# ─────────────────────────────────────────────────────────────────────────────

async def chainlink_rtds_task(
    on_price: Optional[Callable[[float, int], None]] = None,
) -> None:
    """
    Runs both the WebSocket stream and REST poller in parallel.
    REST provides immediate data; WS provides lower-latency streaming once connected.
    """    
    await asyncio.gather(
        _chainlink_ws_task(on_price),
        _chainlink_rest_poller(on_price),
    )


async def _chainlink_ws_task(
    on_price: Optional[Callable[[float, int], None]] = None,
) -> None:
    """
    Persistent WebSocket connection to Polymarket's Chainlink RTDS feed.
    Rotates subscription variants on failure. Logs extensively for debugging.
    """
    reconnect_delay = 1
    variant_idx = 0

    while True:
        sub_msg = SUBSCRIBE_VARIANTS[variant_idx % len(SUBSCRIBE_VARIANTS)]
        v_num = variant_idx % len(SUBSCRIBE_VARIANTS)

        try:
            logger.info(f"RTDS: connecting to {RTDS_WS_URL} (variant V{v_num})…")

            async with websockets.connect(
                RTDS_WS_URL,
                ping_interval=None,   # disabled — we send text "PING" manually
                ping_timeout=None,
                close_timeout=10,
                additional_headers=WS_HEADERS,
            ) as ws:
                sub_json = json.dumps(sub_msg)
                await ws.send(sub_json)
                logger.info(f"RTDS: connected & subscribed V{v_num}: {sub_json[:200]}")
                reconnect_delay = 1
                RTDS.reconnects += 1

                msgs_this = 0
                prices_this = 0

                # Background task: send text "PING" every 5s (required by Polymarket)
                async def _ping_loop():
                    try:
                        while True:
                            await asyncio.sleep(5)
                            await ws.send("PING")
                    except Exception:
                        pass  # connection closed — let main loop handle it

                ping_task = asyncio.create_task(_ping_loop())

                try:
                    while True:
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=90.0)
                        except asyncio.TimeoutError:
                            # Only reconnect on silence if we never got prices
                            if prices_this == 0:
                                logger.warning(
                                    f"RTDS: 90s silence, 0 prices. Reconnecting…"
                                )
                                break
                            # Prices were flowing — just keep waiting
                            logger.info(
                                f"RTDS: recv timeout but {prices_this} prices received. Continuing…"
                            )
                            continue

                        # Skip PONG responses
                        if raw == "PONG":
                            continue

                        msgs_this += 1
                        RTDS.raw_messages += 1

                        # Log first 5 raw messages for debugging format
                        if msgs_this <= 5:
                            preview = raw[:500] if len(raw) > 500 else raw
                            logger.info(f"RTDS raw[{msgs_this}]: {preview}")

                        if msgs_this == 50:
                            logger.info(f"RTDS: 50 msgs, {prices_this} prices so far")

                        result = _parse_rtds_message(raw)
                        if result is None:
                            # Many msgs but zero prices → wrong subscription format
                            if msgs_this >= 100 and prices_this == 0:
                                logger.warning(
                                    f"RTDS: {msgs_this} msgs but 0 prices — "
                                    f"rotating to next variant"
                                )
                                variant_idx += 1
                                break
                            continue

                        price = result["price"]
                        ts_ms = result["ts_ms"]
                        recv  = ms_now()
                        prices_this += 1

                        # Warn if using Binance source — resolution mismatch risk
                        if v_num == 2:
                            logger.warning(
                                f"RTDS: Using Binance fallback ${price:.2f} "
                                f"— RESOLUTION MISMATCH RISK!"
                            )

                        async with _rtds_lock:
                            if ts_ms > RTDS.timestamp_ms:
                                RTDS.price = price
                                RTDS.timestamp_ms = ts_ms
                                RTDS.updates += 1
                            RTDS.recv_ts_ms = recv

                        if on_price:
                            try:
                                on_price(price, ts_ms)
                            except Exception as e:
                                logger.debug(f"RTDS on_price error: {e}")

                        if RTDS.updates <= 3 or RTDS.updates % 200 == 0:
                            logger.info(
                                f"RTDS: ${price:.2f} | total={RTDS.updates} | "
                                f"age={RTDS.age_ms()}ms | V{v_num}"
                            )
                finally:
                    ping_task.cancel()

        except websockets.exceptions.InvalidStatusCode as e:
            RTDS.last_error = f"HTTP {e.status_code}"
            if e.status_code == 429:
                # Rate limited: long exponential backoff starting at 30s
                reconnect_delay = max(reconnect_delay, 30)
                logger.error(
                    f"RTDS: HTTP 429 rate limited. Backing off {reconnect_delay}s"
                )
            else:
                logger.error(
                    f"RTDS: HTTP {e.status_code} rejection. "
                    f"Rotating variant, retry in {reconnect_delay}s"
                )
            variant_idx += 1
            if e.status_code in (403, 404, 401, 429):
                reconnect_delay = min(reconnect_delay * 2, 120)

        except websockets.exceptions.InvalidHandshake as e:
            RTDS.last_error = f"Handshake: {e}"
            # Check if it's a 429 in the handshake error message
            err_str = str(e)
            if "429" in err_str:
                reconnect_delay = max(reconnect_delay, 30)
                logger.error(f"RTDS: handshake 429 rate limit. Backoff {reconnect_delay}s")
                reconnect_delay = min(reconnect_delay * 2, 120)
            else:
                logger.error(f"RTDS: handshake failed: {e}. Retry in {reconnect_delay}s")
            variant_idx += 1

        except websockets.exceptions.ConnectionClosed as e:
            RTDS.last_error = f"Closed: {e.code}/{e.reason}"
            logger.warning(f"RTDS: closed (code={e.code}). Retry in {reconnect_delay}s")

        except ConnectionRefusedError:
            RTDS.last_error = "ConnectionRefused"
            logger.error(f"RTDS: connection refused. Retry in {reconnect_delay}s")

        except OSError as e:
            RTDS.last_error = f"OS: {e}"
            logger.error(f"RTDS: OS error: {e}. Retry in {reconnect_delay}s")

        except Exception as e:
            RTDS.last_error = f"{type(e).__name__}: {e}"
            logger.error(f"RTDS: {type(e).__name__}: {e}. Retry in {reconnect_delay}s")
            reconnect_delay = min(reconnect_delay * 2, 120)

        await asyncio.sleep(reconnect_delay)


# ─────────────────────────────────────────────────────────────────────────────
# Strike Capture
# ─────────────────────────────────────────────────────────────────────────────

class StrikeCapture:
    def __init__(self):
        self.buffer: deque = deque(maxlen=200)
        self.captured_strike: Optional[float] = None
        self.captured_window_ts: Optional[int] = None

    def feed(self, price: float, ts_ms: int) -> None:
        self.buffer.append((ts_ms, price))

    def try_capture(self, window_start_unix: int) -> Optional[float]:
        if self.captured_window_ts == window_start_unix and self.captured_strike is not None:
            return self.captured_strike

        boundary_ms = window_start_unix * 1000
        best_price = None
        best_dist = float('inf')

        for ts_ms, price in self.buffer:
            dist = abs(ts_ms - boundary_ms)
            if dist < best_dist:
                best_dist = dist
                best_price = price

        if best_price is not None and best_dist < 5000:
            self.captured_strike = best_price
            self.captured_window_ts = window_start_unix
            logger.info(
                f"STRIKE from RTDS: {best_price:.2f} (delta={best_dist}ms)"
            )
            return best_price
        return None

    def reset(self) -> None:
        self.captured_strike = None
        self.captured_window_ts = None
        self.buffer.clear()


STRIKE_CAPTURE = StrikeCapture()