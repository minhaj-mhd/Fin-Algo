"""
RESEARCH: Does daily-trend context create long selection skill?

Hypothesis: The short edge ("overbought -> reverts down") is regime-independent.
The long mirror ("oversold -> bounces up") only works in UPTRENDS. The 1h
features have no daily-scale trend context, so the model mixes "dip in uptrend"
(bounces) with "breakdown in downtrend" (keeps falling) -> net zero long skill.

Test (no model, pure stratification on labels):
  1. Build daily bars from 15m cache; compute daily trend indicators as of the
     PRIOR COMPLETE day (no lookahead).
  2. Attach prior-day daily context to each 1h long-label bar.
  3. Stratify LONG net WR by daily-trend regime, and by (daily-trend x intraday
     IBS oversold). If "oversold dip in daily uptrend" shows materially higher
     long WR than baseline (42%), the signal exists and justifies View E.

Reads: data/raw_upstox_cache_15min_3y/*.csv
       data/tbm_labels_1h.parquet
       data/tbm_feature_views/A_meanrev.parquet  (for IBS)
"""
import os, glob
import numpy as np
import pandas as pd

CACHE = 'data/raw_upstox_cache_15min_3y'
COST = 0.0006

def wr(x): return float((x > 0).mean()) if len(x) else np.nan

# ── 1. Build daily bars + prior-day daily trend per ticker ───────────────────
def daily_trend_for_ticker(path):
    df = pd.read_csv(path, parse_dates=['timestamp'])
    df['timestamp'] = (pd.to_datetime(df['timestamp'], utc=True)
                       .dt.tz_convert('Asia/Kolkata').dt.tz_localize(None))
    df['date'] = df['timestamp'].dt.date
    daily = df.groupby('date').agg(
        d_open=('open','first'), d_high=('high','max'),
        d_low=('low','min'),    d_close=('close','last'),
        d_vol=('volume','sum')).reset_index()
    daily = daily.sort_values('date').reset_index(drop=True)
    if len(daily) < 60:
        return None
    c = daily['d_close']
    # Indicators on the COMPLETE daily series
    sma20 = c.rolling(20, min_periods=20).mean()
    sma50 = c.rolling(50, min_periods=50).mean()
    roc10 = c / c.shift(10) - 1.0
    roc20 = c / c.shift(20) - 1.0
    # RSI14 (Wilder) on daily
    delta = c.diff()
    up = delta.clip(lower=0); dn = -delta.clip(upper=0)
    rs = up.ewm(alpha=1/14, min_periods=14).mean() / dn.ewm(alpha=1/14, min_periods=14).mean()
    rsi14 = 100 - 100/(1+rs)
    daily['dist_sma20'] = c/sma20 - 1.0
    daily['dist_sma50'] = c/sma50 - 1.0
    daily['roc10'] = roc10
    daily['roc20'] = roc20
    daily['rsi14_d'] = rsi14
    daily['sma20_slope'] = sma20 / sma20.shift(5) - 1.0
    daily['above_sma20'] = (c > sma20).astype(float)
    daily['above_sma50'] = (c > sma50).astype(float)
    # SHIFT by 1: row for date D holds indicators as of D-1 close (no lookahead)
    feat_cols = ['dist_sma20','dist_sma50','roc10','roc20','rsi14_d',
                 'sma20_slope','above_sma20','above_sma50']
    for col in feat_cols:
        daily[col] = daily[col].shift(1)
    daily['Ticker'] = os.path.basename(path).replace('.csv','')
    return daily[['date','Ticker'] + feat_cols]

print("Building daily trend per ticker ...")
paths = sorted(glob.glob(os.path.join(CACHE, '*.csv')))
parts = []
for i, p in enumerate(paths, 1):
    d = daily_trend_for_ticker(p)
    if d is not None:
        parts.append(d)
    if i % 40 == 0:
        print(f"  [{i}/{len(paths)}]")
daily_all = pd.concat(parts, ignore_index=True)
print(f"  daily trend rows: {len(daily_all):,}\n")

# ── 2. Join with long labels + IBS ───────────────────────────────────────────
lab = pd.read_parquet('data/tbm_labels_1h.parquet')
lab['DateTime'] = pd.to_datetime(lab['DateTime'])
lab['date'] = lab['DateTime'].dt.date
lab['YearMonth'] = lab['DateTime'].dt.to_period('M').astype(str)

va = pd.read_parquet('data/tbm_feature_views/A_meanrev.parquet')
va['DateTime'] = pd.to_datetime(va['DateTime'])
lab = lab.merge(va[['DateTime','Ticker','IBS','RSI_14']], on=['DateTime','Ticker'], how='left')

m = lab.merge(daily_all, on=['date','Ticker'], how='left')
m = m.dropna(subset=['dist_sma20','roc10'])
print(f"Joined long-label bars with daily context: {len(m):,} "
      f"({len(m)/len(lab)*100:.0f}% of labels)\n")

g = m['realized_gross'].values
m['long_net'] = g - COST   # long P&L net of cost

print("=" * 72)
print(f"BASELINE long WR (all bars w/ daily context): {wr(m['long_net'].values):.2%}")
print("=" * 72)

# ── 3a. Stratify by daily trend regime ───────────────────────────────────────
print("\n--- Long WR by DAILY trend regime ---")
for name, mask in [
    ('above daily SMA20', m['above_sma20']==1),
    ('below daily SMA20', m['above_sma20']==0),
    ('above daily SMA50', m['above_sma50']==1),
    ('below daily SMA50', m['above_sma50']==0),
    ('daily ROC10 > 0',   m['roc10']>0),
    ('daily ROC10 < 0',   m['roc10']<0),
    ('daily ROC10 > +3%',  m['roc10']>0.03),
    ('daily ROC10 < -3%',  m['roc10']<-0.03),
    ('sma20 rising',      m['sma20_slope']>0),
    ('sma20 falling',     m['sma20_slope']<0),
]:
    sub = m[mask]
    print(f"  {name:22s}: WR={wr(sub['long_net'].values):6.2%}  n={len(sub):7,d}  "
          f"exp={sub['long_net'].mean()*1e4:+5.1f}bps")

# ── 3b. The money test: oversold dip x daily uptrend ─────────────────────────
print("\n--- Long WR: intraday OVERSOLD (low IBS) x daily UPTREND ---")
ibs_lo = m['IBS'] <= m['IBS'].quantile(0.33)   # bottom third = closed near low = dip
ibs_hi = m['IBS'] >= m['IBS'].quantile(0.67)
up = m['above_sma20']==1
dn = m['above_sma20']==0
roc_up = m['roc10']>0.02

for name, mask in [
    ('oversold + above SMA20',        ibs_lo & up),
    ('oversold + above SMA20 + ROC>2%', ibs_lo & up & roc_up),
    ('oversold + below SMA20',        ibs_lo & dn),
    ('overbought + above SMA20',      ibs_hi & up),
    ('overbought + below SMA20',      ibs_hi & dn),
    ('oversold + sma20 rising',       ibs_lo & (m['sma20_slope']>0)),
    ('oversold + daily RSI 30-50',    ibs_lo & m['rsi14_d'].between(30,50)),
    ('oversold + daily RSI>55',       ibs_lo & (m['rsi14_d']>55)),
]:
    sub = m[mask]
    print(f"  {name:34s}: WR={wr(sub['long_net'].values):6.2%}  n={len(sub):7,d}  "
          f"exp={sub['long_net'].mean()*1e4:+5.1f}bps")

# ── 3c. Per-fold robustness of the best cell ─────────────────────────────────
print("\n--- Per-fold WR: oversold + above SMA20 + ROC10>2% (best-thesis cell) ---")
best = m[ibs_lo & up & roc_up]
windows = {1:['2024-12','2025-01'],2:['2025-04','2025-05'],3:['2025-08','2025-09'],
           4:['2025-12','2026-01'],5:['2026-04','2026-05']}
for f, months in windows.items():
    sub = best[best['YearMonth'].isin(months)]
    print(f"  fold {f}: WR={wr(sub['long_net'].values):6.2%}  n={len(sub):5,d}")
