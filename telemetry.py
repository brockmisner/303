
"""
telemetry.py

Real-time PnL telemetry helpers for dashboard/logging.

This module does NOT require a specific UI; it emits plain dicts you can:
- log as JSON lines
- append to a CSV
- pipe to a websocket dashboard
- print in your existing terminal TUI

Designed to work with Portfolio from position.py.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
import time
import json
from pathlib import Path

from position import Portfolio


@dataclass
class TelemetryConfig:
    emit_every_ms: int = 1000
    jsonl_path: Optional[str] = None  # e.g. "pnl_telemetry.jsonl"


class TelemetryEmitter:
    def __init__(self, cfg: Optional[TelemetryConfig] = None):
        self.cfg = cfg or TelemetryConfig()
        self._last_emit_ms = 0
        self._path = Path(self.cfg.jsonl_path) if self.cfg.jsonl_path else None
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _ms_now() -> int:
        return int(time.time() * 1000)

    def maybe_emit(
        self,
        *,
        portfolio: Portfolio,
        up_bid: float, up_ask: float,
        dn_bid: float, dn_ask: float,
        extra: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        now = self._ms_now()
        if now - self._last_emit_ms < self.cfg.emit_every_ms:
            return None
        self._last_emit_ms = now

        up_mid = 0.5 * (up_bid + up_ask)
        dn_mid = 0.5 * (dn_bid + dn_ask)

        snap = portfolio.snapshot(up_mid, dn_mid)
        payload = {
            "ts_ms": now,
            "pnl": snap,
            "quotes": {
                "UP": {"bid": up_bid, "ask": up_ask, "mid": up_mid},
                "DOWN": {"bid": dn_bid, "ask": dn_ask, "mid": dn_mid},
            },
        }
        if extra:
            payload.update(extra)

        if self._path:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")

        return payload
