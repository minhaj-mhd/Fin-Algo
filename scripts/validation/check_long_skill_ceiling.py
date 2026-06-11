"""
Is there ANY exploitable long skill hiding at the top of the conviction
distribution?  (The short model's +12pp came through even with a permissive
filter; maybe the long model has skill only at its highest-conviction tail.)

Read-only on test_trades_long.parquet. If even the top-decile long trades are
~45% WR, long is unfixable with these features. If top-decile hits 55%+, we
can reach 57% by tightening selection (same lever that works for short).
"""
import numpy as np
import pandas as pd

df = pd.read_parquet('data/model_analysis/tbm_1h/test_trades_long.parquet')
COST_10 = 0.0010
def wr(x): return float((x > 0).mean()) if len(x) else np.nan

print(f"Long trades: {len(df):,}  folds: {sorted(df['fold'].unique())}\n")

print("=" * 70)
print("LONG WR by EV percentile band (higher EV = more confident long)")
print("=" * 70)
for lo, hi in [(0,100),(50,100),(70,100),(80,100),(90,100),(95,100),(99,100)]:
    th_lo = np.percentile(df['ev'], lo)
    sub = df[df['ev'] >= th_lo]
    print(f"  top {100-lo:3d}% EV : n={len(sub):5d}  WR@6={wr(sub['realized_net'].values):5.1%}  "
          f"WR@10={wr(sub['realized_gross'].values-COST_10):5.1%}  "
          f"exp={sub['realized_net'].mean()*1e4:+5.1f}bps")

print("\n" + "=" * 70)
print("LONG WR by p_tp percentile band (raw P(long-TP))")
print("=" * 70)
for lo in [0,50,70,80,90,95,99]:
    th = np.percentile(df['p_tp'], lo)
    sub = df[df['p_tp'] >= th]
    print(f"  top {100-lo:3d}% p_tp: n={len(sub):5d}  WR@6={wr(sub['realized_net'].values):5.1%}  "
          f"p_tp>={th:.3f}")

print("\n" + "=" * 70)
print("k=1 (single highest-EV long per timestamp) per fold")
print("=" * 70)
for f in sorted(df['fold'].unique()):
    sub = df[df['fold']==f]
    top1 = sub.sort_values('ev',ascending=False).groupby('DateTime').head(1)
    print(f"  fold {f}: k=1 WR@6={wr(top1['realized_net'].values):5.1%} (n={len(top1):4d})")
top1all = df.sort_values('ev',ascending=False).groupby(['fold','DateTime']).head(1)
print(f"  POOLED k=1 WR@6={wr(top1all['realized_net'].values):.1%} (n={len(top1all)})")
