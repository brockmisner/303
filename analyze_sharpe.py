#!/usr/bin/env python3
"""
analyze_sharpe.py

Compute realized PnL Sharpe over last N completed 5-minute windows from
logs/pnl_telemetry.jsonl emitted by telemetry.py.

Usage:
  python analyze_sharpe.py --path logs/pnl_telemetry.jsonl --windows 50
"""

import argparse, json, math

def window_id(ts_ms: int) -> int:
    # 5-minute windows in UTC
    return int((ts_ms // 1000) // 300)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="logs/pnl_telemetry.jsonl")
    ap.add_argument("--windows", type=int, default=50)
    ap.add_argument("--min_window_seconds", type=float, default=240.0,
                    help="Minimum observed duration within a window to treat as complete.")
    args = ap.parse_args()

    by_win = {}
    with open(args.path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            ts = int(o.get("ts_ms", 0))
            if ts <= 0:
                continue
            pnl = o.get("pnl", {}) or {}
            realized = float(pnl.get("total_realized", 0.0))
            wid = window_id(ts)
            w = by_win.get(wid)
            if w is None:
                by_win[wid] = {"start_ts": ts, "end_ts": ts, "realized_start": realized, "realized_end": realized}
            else:
                w["end_ts"] = ts
                w["realized_end"] = realized

    wins = sorted(by_win.items(), key=lambda kv: kv[0])

    completed = []
    for wid, w in wins:
        dur = (w["end_ts"] - w["start_ts"]) / 1000.0
        if dur >= args.min_window_seconds:
            r = w["realized_end"] - w["realized_start"]
            completed.append((wid, r))

    sample = completed[-args.windows:]
    if len(sample) < 10:
        print(f"Only {len(sample)} completed windows found (need >= 10).")
        return

    rets = [r for _, r in sample]
    mean = sum(rets) / len(rets)
    var = sum((x - mean) ** 2 for x in rets) / (len(rets) - 1)
    stdev = math.sqrt(var)

    sharpe = (mean / stdev) if stdev > 1e-12 else 0.0
    # Annualize-ish for 5-min windows: ~105,120 windows/year
    ann = sharpe * math.sqrt(105120)

    win_rate = sum(1 for x in rets if x > 0) / len(rets)

    print(f"Windows analyzed: {len(rets)}")
    print(f"Mean PnL/window: {mean:+.4f} USD")
    print(f"Std  PnL/window: {stdev:.4f} USD")
    print(f"Win rate: {win_rate*100:.1f}%")
    print(f"Sharpe (per-window): {sharpe:.4f}")
    print(f"Sharpe (annualized-ish): {ann:.2f}")

    # Optional: show worst/best 5
    pairs = sorted(sample, key=lambda x: x[1])
    print("\nWorst 5 windows:")
    for wid, r in pairs[:5]:
        print(f"  wid={wid} pnl={r:+.2f}")
    print("\nBest 5 windows:")
    for wid, r in pairs[-5:]:
        print(f"  wid={wid} pnl={r:+.2f}")

if __name__ == "__main__":
    main()
