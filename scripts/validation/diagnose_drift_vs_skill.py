"""
Drift-vs-Skill decomposition for the TBM 1h labels.

Question: Is the short model's 56.5% WR genuine stock-selection skill, or is it
riding a persistent intraday DOWN-drift (which would also explain the long
model's 43.3% failure)?  This determines whether the long model is FIXABLE.

Method (uses only data/tbm_labels_1h.parquet — no model needed):
  1. Unconditional WR: if you traded EVERY bar long vs short, what WR?
     - If short-everything ~= 56% and long-everything ~= 43%, the models add
       ~no selection skill and the edge is pure drift.
  2. Drift: mean 1h forward (gross) return overall and per test-window.
  3. Cross-sectional skill: within each timestamp, demean returns. Does the
     spread of outcomes survive after removing the market's hourly move?
  4. Compare model-SELECTED WR (saved trades) vs unconditional WR = selection skill.
"""
import numpy as np
import pandas as pd

LABELS = 'data/tbm_labels_1h.parquet'
COST = 0.0006
COST_10 = 0.0010

df = pd.read_parquet(LABELS)
df['DateTime'] = pd.to_datetime(df['DateTime'])
df['YearMonth'] = df['DateTime'].dt.to_period('M').astype(str)
print(f"Loaded {len(df):,} labeled bars, {df['Ticker'].nunique()} tickers\n")

g = df['realized_gross'].values  # long gross return (entry->exit, fraction)

def wr(x):
    return float((x > 0).mean())

# ── 1. Unconditional WR (trade everything) ───────────────────────────────────
print("=" * 74)
print("1. UNCONDITIONAL WR  (trade EVERY bar — no model selection)")
print("=" * 74)
long_net6  = g - COST
short_net6 = -g - COST
long_net10  = g - COST_10
short_net10 = -g - COST_10
print(f"  LONG  @6bps: {wr(long_net6):6.2%}   @10bps: {wr(long_net10):6.2%}")
print(f"  SHORT @6bps: {wr(short_net6):6.2%}   @10bps: {wr(short_net10):6.2%}")
print(f"  (sum @6bps = {wr(long_net6)+wr(short_net6):.2%}  -- near 100% => pure drift)")
print(f"  Mean 1h drift (gross): {g.mean()*1e4:+.2f} bps")
print(f"  Median 1h drift:       {np.median(g)*1e4:+.2f} bps")

# ── 2. Drift per test window (the 5 WF test periods) ─────────────────────────
print("\n" + "=" * 74)
print("2. DRIFT + UNCONDITIONAL WR PER WF TEST WINDOW")
print("=" * 74)
test_windows = {
    1: ['2024-12','2025-01'],
    2: ['2025-04','2025-05'],
    3: ['2025-08','2025-09'],
    4: ['2025-12','2026-01'],
    5: ['2026-04','2026-05'],
}
print(f"  {'fold':5s} {'months':18s} {'drift_bps':>10s} {'L_WR@6':>8s} {'S_WR@6':>8s}")
for f, months in test_windows.items():
    sub = df[df['YearMonth'].isin(months)]
    gg = sub['realized_gross'].values
    print(f"  {f:<5d} {str(months):18s} {gg.mean()*1e4:>+9.2f} "
          f"{wr(gg-COST):>8.2%} {wr(-gg-COST):>8.2%}")

# ── 3. Cross-sectional (drift-neutral) skill ceiling ─────────────────────────
print("\n" + "=" * 74)
print("3. DRIFT-NEUTRAL VIEW  (demean returns per timestamp)")
print("=" * 74)
df['xs_mean'] = df.groupby('DateTime')['realized_gross'].transform('mean')
df['rel_ret'] = df['realized_gross'] - df['xs_mean']
rel = df['rel_ret'].values
# After demeaning, a long that beats the cross-section wins; cost still applies.
print(f"  Cross-sectional dispersion (std of rel_ret): {rel.std()*1e4:.1f} bps")
print(f"  If we could pick the top-half relative movers, long WR ceiling:")
# best-case: trade only bars with rel_ret > 0 (oracle) - sanity ceiling
print(f"    oracle long  (rel>0, net6): {wr(df[df['rel_ret']>0]['realized_gross'].values-COST):.1%} "
      f"(n={int((df['rel_ret']>0).sum()):,})")
print(f"    oracle short (rel<0, net6): {wr(-df[df['rel_ret']<0]['realized_gross'].values-COST):.1%} "
      f"(n={int((df['rel_ret']<0).sum()):,})")
print("  (oracle = perfect foresight; real model gets a fraction of this)")

# ── 4. Selection skill = model WR  minus  unconditional WR ───────────────────
print("\n" + "=" * 74)
print("4. SELECTION SKILL  (model-selected WR  vs  trade-everything WR)")
print("=" * 74)
import os
for side, uncond in [('short', wr(short_net6)), ('long', wr(long_net6))]:
    p = f'data/model_analysis/tbm_1h/test_trades_{side}.parquet'
    if not os.path.exists(p):
        continue
    t = pd.read_parquet(p)
    sel_wr = wr(t['realized_net'].values)
    print(f"  {side.upper():5s}: model-selected WR={sel_wr:.2%}  "
          f"unconditional WR={uncond:.2%}  "
          f"=> selection skill = {(sel_wr-uncond)*100:+.2f} pp")
print("\n  If selection skill ~= 0, the model adds nothing over a coin-flip")
print("  directional bet, and the WR is entirely drift. That is the disease.")
