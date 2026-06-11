"""
Short TBM selectivity analysis — find a robust lever to clear 57% net WR.

Tests, PER FOLD (not just pooled) so we can judge OOS robustness:
  1. k-reduction: WR at k=1, k=2, k=3 (re-derived from stored top-3 trades)
  2. EV-percentile: WR keeping only top X% by EV within each fold
  3. p_tp ranking: is raw P(TP) a better conviction signal than EV?
  4. Volume at each setting (must stay usable, >= ~50/mo)

Reads: data/model_analysis/tbm_1h/test_trades_short.parquet
"""
import numpy as np
import pandas as pd

TRADES = 'data/model_analysis/tbm_1h/test_trades_short.parquet'
COST_10 = 0.0010

df = pd.read_parquet(TRADES)
print(f"Loaded {len(df):,} stored short trades")
print(f"Columns: {list(df.columns)}")
print(f"Folds: {sorted(df['fold'].unique())}")
print(f"realized_net is short P&L (positive = short won). label==1 => short TP.\n")

def wr(x):
    return float((x > 0).mean()) if len(x) else np.nan

def summarize(sub, label):
    n = len(sub)
    if n == 0:
        print(f"  {label:28s}  n=0")
        return
    w6  = wr(sub['realized_net'].values)
    w10 = wr(sub['realized_gross'].values - COST_10)
    exp = sub['realized_net'].mean() * 1e4
    # crude trades/month: 2 test months per fold, 5 folds = 10 months pooled
    print(f"  {label:28s}  n={n:5d}  WR@6={w6:5.1%}  WR@10={w10:5.1%}  exp={exp:+5.1f}bps")

# ── 1. k-reduction (pooled) ──────────────────────────────────────────────────
print("=" * 78)
print("1. K-REDUCTION (re-rank stored trades by EV within each DateTime)")
print("=" * 78)
print("POOLED:")
for k in [3, 2, 1]:
    topk = (df.sort_values('ev', ascending=False)
              .groupby(['fold', 'DateTime']).head(k))
    summarize(topk, f"k={k}")

print("\nPER-FOLD WR@6 (the robustness test):")
print(f"  {'fold':6s} {'k=3':>14s} {'k=2':>14s} {'k=1':>14s}")
for f in sorted(df['fold'].unique()):
    sub = df[df['fold'] == f]
    row = f"  {f:<6d}"
    for k in [3, 2, 1]:
        topk = (sub.sort_values('ev', ascending=False)
                   .groupby('DateTime').head(k))
        n = len(topk); w = wr(topk['realized_net'].values)
        row += f" {w:6.1%}(n={n:4d})"
    print(row)

# ── 2. EV-percentile threshold (within fold) ─────────────────────────────────
print("\n" + "=" * 78)
print("2. EV-PERCENTILE (keep top X% by EV within each fold, k=3 base)")
print("=" * 78)
for pct in [100, 75, 50, 30, 20, 10]:
    keep = []
    for f in sorted(df['fold'].unique()):
        sub = df[df['fold'] == f]
        if pct < 100:
            thr = np.percentile(sub['ev'].values, 100 - pct)
            sub = sub[sub['ev'] >= thr]
        keep.append(sub)
    allk = pd.concat(keep)
    summarize(allk, f"top {pct}% EV")

print("\nPER-FOLD WR@6 at top-30% and top-20% EV:")
print(f"  {'fold':6s} {'top30%':>16s} {'top20%':>16s} {'top10%':>16s}")
for f in sorted(df['fold'].unique()):
    sub = df[df['fold'] == f]
    row = f"  {f:<6d}"
    for pct in [30, 20, 10]:
        thr = np.percentile(sub['ev'].values, 100 - pct)
        s2 = sub[sub['ev'] >= thr]
        n = len(s2); w = wr(s2['realized_net'].values)
        row += f" {w:6.1%}(n={n:4d})"
    print(row)

# ── 3. p_tp ranking (alternative conviction) ─────────────────────────────────
print("\n" + "=" * 78)
print("3. P_TP RANKING (is raw P(short-TP) better than EV?)  k=1 by p_tp")
print("=" * 78)
print("POOLED:")
top1_ptp = df.sort_values('p_tp', ascending=False).groupby(['fold','DateTime']).head(1)
top1_ev  = df.sort_values('ev',   ascending=False).groupby(['fold','DateTime']).head(1)
summarize(top1_ptp, "k=1 by p_tp")
summarize(top1_ev,  "k=1 by ev")

print("\nPER-FOLD WR@6, k=1:")
print(f"  {'fold':6s} {'by p_tp':>16s} {'by ev':>16s}")
for f in sorted(df['fold'].unique()):
    sub = df[df['fold'] == f]
    a = sub.sort_values('p_tp', ascending=False).groupby('DateTime').head(1)
    b = sub.sort_values('ev',   ascending=False).groupby('DateTime').head(1)
    print(f"  {f:<6d} {wr(a['realized_net'].values):6.1%}(n={len(a):4d})  "
          f"{wr(b['realized_net'].values):6.1%}(n={len(b):4d})")

# ── 4. p_tp threshold (absolute conviction gate) ─────────────────────────────
print("\n" + "=" * 78)
print("4. P_TP ABSOLUTE GATE (keep trades with p_tp >= threshold, k=3 base)")
print("=" * 78)
print(f"  p_tp distribution: min={df['p_tp'].min():.3f}  med={df['p_tp'].median():.3f}  "
      f"p90={df['p_tp'].quantile(0.9):.3f}  max={df['p_tp'].max():.3f}")
for thr in [0.0, 0.15, 0.20, 0.25, 0.30, 0.35]:
    sub = df[df['p_tp'] >= thr]
    summarize(sub, f"p_tp>={thr:.2f}")

print("\nPER-FOLD WR@6 at p_tp>=0.25 and >=0.30:")
for f in sorted(df['fold'].unique()):
    sub = df[df['fold'] == f]
    s25 = sub[sub['p_tp'] >= 0.25]; s30 = sub[sub['p_tp'] >= 0.30]
    print(f"  fold {f}: p_tp>=0.25 {wr(s25['realized_net'].values):6.1%}(n={len(s25):4d})   "
          f"p_tp>=0.30 {wr(s30['realized_net'].values):6.1%}(n={len(s30):4d})")
