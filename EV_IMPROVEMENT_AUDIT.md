# EV Improvement Audit — Polymarket 5m BTC Up/Down

This audit focuses on **expected value per trade** and **edge capture rate** for the current strategy.

## Executive summary (highest-impact levers)

1. **Capture more of the same edge event (not just one shot):** right now the sniper path allows a single fire per side/window, which likely leaves positive EV on the table during persistent dislocations.
2. **Tighten signal quality with realized calibration, not static thresholds:** the system has excellent structure, but static `min_p` / edge gates should be continuously re-fit by spread, lag, and time bucket.
3. **Reduce execution EV leakage:** use adaptive maker/step routing as default in healthy books; reserve aggressive taker/FOK only when short-lived edge or late window justifies it.
4. **Reallocate risk to high-quality regimes:** current multi-layer Kelly stack can compress too hard in quality windows and still allow low-quality clips; use quality-conditioned floors/ceilings.

---

## What the code is already doing well

- Uses Chainlink RTDS as primary oracle and blocks stale oracles before firing.
- Uses calibrated cone model + z-trajectory confirmation + spread/dead-book checks.
- Has strong safety layers (tail risk, SPRT, drawdown/day kill, inventory controls).
- Includes adaptive execution infrastructure (maker ladder + EV-aware cross logic).

This means the biggest EV upside is **not** from adding more indicators; it is from improving:
- **entry selectivity calibration**,
- **fill quality and price paid**, and
- **how much size is deployed on truly high-quality shots**.

---

## Priority 1 — Multi-fire edge harvesting per window (drastic EV unlock)

### Current limiter
The sniper decision locks one side per window and blocks repeat firing for that side/window even if edge remains high and book replenishes.

### Why this matters
In short binary windows, dislocations often reappear in bursts. A one-shot policy captures only the first clip and misses follow-on positive EV.

### Recommended change
- Replace strict one-fire-per-side with a **budgeted micro-burst policy**:
  - Max `N` clips per side/window (e.g., 2–4),
  - Require edge re-qualification (e.g., edge stays above threshold + cooldown elapsed + fresh liquidity depth),
  - Decrease incremental size each additional clip unless edge improves.

### Suggested guardrails
- Keep total per-window spend cap unchanged initially.
- Enforce incremental fill-quality threshold (`p_fill`, spread tier, depth tier).
- Disable extra clips when flip-rate/chop is elevated.

---

## Priority 2 — Online calibration of edge thresholds (selection EV)

### Current limiter
`min_p`, edge targets, and late penalties are mostly static or rule-based.

### Why this matters
Selection EV decays quickly if thresholds are not conditioned on current microstructure (spread, lag weather, time-to-expiry, side asymmetry).

### Recommended change
Build a lightweight nightly recalibration job from production logs:

- Bucket by:
  - side (`UP`/`DOWN`),
  - `T` bucket (20–60, 60–120, 120–240),
  - spread bucket,
  - lag bucket (p50/p10),
  - sigma ratio bucket.
- For each bucket, estimate realized EV after fees/slippage.
- Learn bucket-specific `min_edge` and `max_pay` adjustments.

This should replace broad global constants with **state-dependent thresholds**.

---

## Priority 3 — Make execution routing EV-first by default

### Current limiter
The codebase supports adaptive maker/step/cross execution, but aggressive taker paths still consume meaningful flow in places where maker EV may dominate.

### Why this matters
On binaries, paying 1–3 cents extra is frequently the entire edge. Reducing price paid is usually the fastest EV improvement.

### Recommended change
- In healthy books and non-late windows, route via adaptive maker ladder first.
- Only escalate to taker when:
  - edge half-life is short,
  - residual EV after expected slippage remains above threshold,
  - and time decay is critical.
- Add strict post-trade attribution:
  - expected edge at signal,
  - actual paid edge at fill,
  - execution leakage (bps/share).

Set a KPI: **execution leakage < 30% of signal edge** median.

---

## Priority 4 — Re-shape Kelly compression to reward quality

### Current limiter
Many risk multipliers are sensibly defensive, but the stack can under-allocate top-decile opportunities and over-allocate marginal ones.

### Recommended change
Use quality-conditioned Kelly bands:

- Define a `quality_score` from edge percentile, spread tier, fill probability, lag regime.
- Apply:
  - higher Kelly floor for top-quality buckets,
  - lower Kelly ceiling for marginal buckets,
  - keep global drawdown/day kill unchanged.

This increases EV by concentrating risk where realized edge survives execution.

---

## Priority 5 — Side-specific asymmetry learning (UP vs DOWN)

### Observation
The strategy already has directional handling, but market microstructure and participant behavior are often asymmetric across UP/DOWN books.

### Recommended change
- Maintain separate calibrations for:
  - fill probability model parameters,
  - min-edge thresholds,
  - execution escalation timing,
  - cooldown and persistence.

Target: avoid applying a single policy to two structurally different books.

---

## 14-day rollout plan

### Week 1
1. Add EV attribution fields to every fire/fill/outcome event.
2. Implement micro-burst policy behind feature flag.
3. Route healthy-book entries through adaptive execution by default.

### Week 2
4. Run nightly threshold calibration job and emit config deltas.
5. Deploy side/time/spread/lag-conditioned thresholds.
6. Tune Kelly by quality score percentiles.

---

## Metrics to track (must improve)

- **Realized EV / share** (net fees/slippage), overall and by bucket.
- **Signal-to-fill leakage** = (signal edge − realized edge at fill).
- **Fill rate at EV-positive quotes**.
- **PnL per unit gross exposure**.
- **Top-decile quality bucket capital share** (should increase).

---

## Practical expectation

If implemented cleanly, the largest gains should come from:
1) better execution price discipline,
2) repeated capture of persistent dislocations, and
3) threshold calibration by market state.

Those three together are usually where the “drastic” EV step-change lives for high-frequency binary microstructure strategies.
