# PolyBot v2 — SOP & Workflow Guide

This repository runs an automated strategy for **Polymarket 5-minute BTC Up/Down markets**.

This README is an operational SOP (Standard Operating Procedure) for:
- environment setup,
- safe startup,
- runbook workflows (SIM → canary → live),
- monitoring and recovery,
- change-management workflow.

---

## 1) System Overview

Core execution path:
1. **Market/oracle ingestion** (`chainlink_rtds.py`, Polymarket book WS in `main.py`)
2. **Edge decisioning** (`oracle_engine.py`)
3. **Risk/sizing overlays** (`main.py`, `tail_risk.py`, `sprt.py`, `position_monitor.py`)
4. **Execution routing** (`adaptive_executor.py` and aggressive FAK/FOK path in `main.py`)
5. **Telemetry/logging** (`logs/`, `telemetry.py`, `pnl_tracker.py`)

Primary entrypoint: `main.py`.

---

## 2) Prerequisites

- Python 3.10+
- Access credentials for Polymarket CLOB
- Network access to:
  - Polymarket APIs/WS
  - Gamma API
  - Chainlink RTDS endpoint

Recommended host:
- low-latency VPS,
- stable clock (NTP enabled),
- persistent disk for `logs/`.

---

## 3) Installation SOP

### 3.1 Clone + venv

```bash
git clone <repo-url>
cd 303
python -m venv .venv
source .venv/bin/activate
```

### 3.2 Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3.3 Environment variables

Create `.env` in repo root. Minimum pattern:

```bash
# Runtime mode
SIM_MODE=true

# CLOB / chain / auth (fill with real values)
PK=<private_key_or_signing_key>
CLOB_API_KEY=<api_key_if_used>
CLOB_API_SECRET=<api_secret_if_used>
CLOB_API_PASSPHRASE=<api_passphrase_if_used>
CHAIN_ID=137

# Risk caps (start conservative)
MAX_SPEND_PER_ORDER_USD=25
WINDOW_SPEND_CAP_USD=150
MAX_GROSS_SHARES=90
MAX_NET_SHARES=60

# Session/day controls
SESSION_LOSS_LIMIT=-50
MAX_CONSEC_LOSSES=4
DAY_LOSS_LIMIT=-150
DAY_DD_MAX=0.10

# EV calibration / attribution
CALIBRATION_LOG_FILE=logs/ev_calibration.csv
EXEC_LEAKAGE_WARN=0.30

# Micro-burst sniper controls
MAX_SNIPER_FIRES_PER_SIDE=3
MICRO_BURST_MIN_EDGE=0.03
MICRO_BURST_MIN_DEPTH=30
```

> SOP recommendation: first 24h in SIM mode with low spend caps.

---

## 4) Pre-Flight Checklist (Mandatory)

Before **every** run:

1. Activate venv.
2. Confirm `.env` exists and keys are loaded.
3. Confirm logs directory writable:
   ```bash
   mkdir -p logs
   ```
4. Syntax check:
   ```bash
   python -m py_compile main.py oracle_engine.py adaptive_executor.py position_monitor.py sprt.py
   ```
5. Ensure `SIM_MODE=true` for dry runs.
6. Start process and verify startup logs show:
   - market discovery success,
   - RTDS feed activity,
   - adaptive executor init (if client initialized),
   - no repeated exceptions.

---

## 5) Run Workflow (SIM → Canary → Live)

### Phase A — SIM burn-in

```bash
SIM_MODE=true python main.py
```

Observe for at least multiple windows:
- no crash loops,
- reasonable signal frequency,
- no runaway sizing,
- healthy book/oracle freshness behavior.

### Phase B — Live canary

Use very small notional caps:
- tiny `MAX_SPEND_PER_ORDER_USD`,
- strict `WINDOW_SPEND_CAP_USD`,
- keep all kill-switches enabled.

```bash
SIM_MODE=false python main.py
```

Canary criteria (example):
- stable fills,
- acceptable leakage,
- no frequent circuit/day-kill activations.

### Phase C — Controlled scale-up

Scale one parameter at a time:
1. per-order spend,
2. window cap,
3. (optional) micro-burst fire cap.

Never scale all three simultaneously.

---

## 6) Operational SOP During Run

Monitor these files/logs:
- `logs/bot.log`
- `logs/trades.csv`
- `logs/ev_calibration.csv`

Key runtime signals to watch:
- stale oracle / dead-book blocks,
- execution leakage warnings,
- SPRT status drift,
- circuit breaker/day kill triggers,
- repeated adaptive-exec fallback to aggressive taker path.

If any persistent anomaly appears:
1. stop process,
2. preserve logs,
3. reduce caps,
4. restart in SIM for diagnosis.

---

## 7) Emergency SOP (Kill / Recovery)

### Hard stop now
```bash
pkill -f "python main.py"
```

### Recovery flow
1. Stop bot.
2. Snapshot logs (`logs/` copy).
3. Identify cause (oracle staleness, marketability rejects, toxic-flow periods, etc.).
4. Restart in SIM mode first.
5. Resume live only after stable SIM behavior.

---

## 8) Change Management Workflow (Engineering SOP)

1. **Branch** from current mainline.
2. Implement smallest safe change.
3. Run syntax/static checks.
4. Run SIM smoke.
5. Review logs for regressions.
6. Commit with scoped message.
7. Open PR with:
   - what changed,
   - risk impact,
   - rollback plan,
   - validation commands + output summary.

Recommended PR checklist:
- [ ] No accidental cap removals
- [ ] Kill-switch logic untouched or intentionally reviewed
- [ ] No blocking I/O in hot path additions
- [ ] New env vars documented

---

## 9) Tuning Workflow (EV Improvement Loop)

Use rolling loop:
1. Collect data in `logs/ev_calibration.csv`.
2. Bucket by side / T / spread / lag / sigma ratio.
3. Refit threshold shifts (`CAL_*`) conservatively.
4. Deploy to SIM first, then canary.
5. Compare:
   - realized EV/share,
   - signal-to-fill leakage,
   - fill rate at EV-positive quotes,
   - drawdown behavior.

Do not tune more than 1–2 interacting knobs per deployment.

---

## 10) Common Issues & Fixes

### "No active 5-min market found"
- Check Gamma API/network reachability.
- Verify clock/timezone and retry logic in logs.

### Frequent `NON_MARKETABLE` / misses
- Market moved too fast; review spread + max slip controls.
- Ensure book freshness and routing conditions.

### Circuit breaker trips too often
- Reduce size caps.
- Tighten edge thresholds / micro-burst settings.
- Verify model-vs-market calibration and leakage.

### High leakage warnings
- Favor adaptive maker path where appropriate.
- Reduce taker aggressiveness in non-urgent regimes.
- Revisit max-pay and spread gating.

---

## 11) Useful Commands

```bash
# Syntax check
python -m py_compile main.py oracle_engine.py adaptive_executor.py position_monitor.py sprt.py

# Run sim
SIM_MODE=true python main.py

# Run live (after canary readiness)
SIM_MODE=false python main.py

# Tail logs
tail -f logs/bot.log

# Quick trade log peek
head -n 5 logs/trades.csv
```

---

## 12) Security Notes

- Never commit `.env`.
- Rotate keys after incident or host compromise.
- Restrict host access (SSH keys only, firewall rules, minimal services).

---

If you want, next step I can add:
1. `.env.example` template,
2. a `runbook.md` incident checklist,
3. a small `scripts/preflight.sh` to automate section 4.
