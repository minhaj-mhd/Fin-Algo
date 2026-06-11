"""
Compute RAW vs NET win-rate for the TBM short model directly from saved trades,
and sanity-check the cost arithmetic in the harness.

For the short trades:
  realized_gross  should already be the SHORT gross P&L (harness negates long ret)
  realized_net    should be short_gross - COST
Verify: realized_net - realized_gross == -0.0006 (correct) or +0.0006 (BUG).
"""
import numpy as np
import pandas as pd

COST6, COST10 = 0.0006, 0.0010
def wr(x): return float((x > 0).mean()) if len(x) else np.nan

df = pd.read_parquet('data/model_analysis/tbm_1h/test_trades_short.parquet')
g   = df['realized_gross'].values    # short gross
net = df['realized_net'].values      # harness 'net'

diff = np.median(net - g) * 1e4
print(f"n trades: {len(df):,}")
print(f"median(realized_net - realized_gross) = {diff:+.2f} bps")
print(f"  -> should be -6.00 if cost subtracted correctly; +6.00 means cost SIGN BUG\n")

print("=" * 60)
print("WIN RATES (k=3, all stored trades)")
print("=" * 60)
print(f"  RAW WR        P(gross>0)        : {wr(g):.2%}")
print(f"  TRUE net@6bps P(gross-0.0006>0) : {wr(g - COST6):.2%}")
print(f"  TRUE net@10bps P(gross-0.0010>0): {wr(g - COST10):.2%}")
print(f"  (harness 'net@6' P(realized_net>0): {wr(net):.2%}  <- buggy if sign flipped)")
print(f"\n  RAW mean   : {g.mean()*1e4:+.2f} bps")
print(f"  TRUE net@6 mean : {(g-COST6).mean()*1e4:+.2f} bps")
print(f"  harness net mean: {net.mean()*1e4:+.2f} bps")

# k=1 by EV
print("\n" + "=" * 60)
print("WIN RATES (k=1 by EV per timestamp)")
print("=" * 60)
k1 = df.sort_values('ev', ascending=False).groupby(['fold','DateTime']).head(1)
g1 = k1['realized_gross'].values
print(f"  n={len(k1)}")
print(f"  RAW WR         : {wr(g1):.2%}")
print(f"  TRUE net@6bps  : {wr(g1 - COST6):.2%}")
print(f"  TRUE net@10bps : {wr(g1 - COST10):.2%}")

# Correct selection skill: TRUE net@6 selected vs unconditional
print("\n" + "=" * 60)
print("CORRECTED SELECTION SKILL (true net@6)")
print("=" * 60)
lab = pd.read_parquet('data/tbm_labels_1h.parquet')
gl = lab['realized_gross'].values          # long gross over all bars
short_gross_all = -gl
uncond_net6 = wr(short_gross_all - COST6)
sel_net6 = wr(g - COST6)
print(f"  unconditional short net@6 (all bars): {uncond_net6:.2%}")
print(f"  selected short net@6 (k=3 trades)   : {sel_net6:.2%}")
print(f"  TRUE selection skill                : {(sel_net6-uncond_net6)*100:+.2f} pp")

# per fold true net@6
print("\nPer-fold TRUE net@6 WR (k=3):")
for f in sorted(df['fold'].unique()):
    sub = df[df['fold']==f]
    gg = sub['realized_gross'].values
    print(f"  fold {f}: raw={wr(gg):.1%}  net@6={wr(gg-COST6):.1%}  net@10={wr(gg-COST10):.1%}")
