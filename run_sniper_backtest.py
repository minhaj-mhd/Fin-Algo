"""
Sniper Strategy OOS Backtest — Jan–May 2026
Validates the research claims on data the model was never trained on.
Training cutoff: ~Jul 2025 (80% walk-forward split).
"""
import os, json, pickle, warnings
import numpy as np
import pandas as pd
import xgboost as xgb

warnings.filterwarnings('ignore')
os.chdir(r'c:\Users\loq\Desktop\Trading\finalgo')

COST_BPS     = 10
COST_PCT     = COST_BPS / 10000
OOS_PREFIXES = ['2026-01', '2026-02', '2026-03', '2026-04', '2026-05']

# Research thresholds from Complete Edge Catalog
TIER_A_S_HI =  0.100;  TIER_A_L_LO = -0.160
TIER_B_S_HI =  0.087
TIER_C_L_LO = -0.167
TIER_D_L_HI =  0.080;  TIER_D_S_LO = -0.200

# ── Load 2026 OOS data ───────────────────────────────────────────────────────
print('Loading 2026 OOS data from ranking_data_upstox_3y.csv ...')
chunks = []
for chunk in pd.read_csv('data/ranking_data_upstox_3y.csv', chunksize=150_000):
    mask = chunk['DateTime'].str.startswith(tuple(OOS_PREFIXES))
    if mask.any():
        chunks.append(chunk[mask])
df = pd.concat(chunks, ignore_index=True)
df['bar_time'] = df['DateTime'].str[11:16]
df['date']     = df['DateTime'].str[:10]
print(f'Rows: {len(df):,}  |  Tickers: {df["Ticker"].nunique()}  |  Range: {df["date"].min()} -> {df["date"].max()}')
print(f'Bar times: {sorted(df["bar_time"].unique())}')

# ── Score with v8_upstox_3y model (no Z-Score, raw features) ────────────────
print('\nScoring with v8_upstox_3y model...')
with open('models/v8_upstox_3y/metadata.json') as f:
    feature_cols = json.load(f)['features']

bst_long  = xgb.Booster()
bst_long.load_model('models/v8_upstox_3y/xgb_long_model.json')
bst_long.set_param({'device': 'cpu'})

bst_short = xgb.Booster()
bst_short.load_model('models/v8_upstox_3y/xgb_short_model.json')
bst_short.set_param({'device': 'cpu'})

# Confirm scaler status
scaler = None
if os.path.exists('models/scaler.pkl'):
    with open('models/scaler.pkl', 'rb') as sf:
        s = pickle.load(sf)
        if hasattr(s, 'scale_') and s.scale_ is not None:
            scaler = s
            print('  Scaler ACTIVE — applying transform')
        else:
            print('  Scaler is pass-through (scale_=None) — raw features used directly')

for c in [c for c in feature_cols if c not in df.columns]:
    df[c] = 0.0

X = np.nan_to_num(df[feature_cols].values)
if scaler:
    X = scaler.transform(X)

dmat = xgb.DMatrix(X, feature_names=feature_cols)
df['long_score']  = bst_long.predict(dmat)
df['short_score'] = bst_short.predict(dmat)

print(f'  long_score:  [{df["long_score"].min():.4f}, {df["long_score"].max():.4f}]  mean={df["long_score"].mean():.4f}')
print(f'  short_score: [{df["short_score"].min():.4f}, {df["short_score"].max():.4f}]  mean={df["short_score"].mean():.4f}')

# ── Signal counts ────────────────────────────────────────────────────────────
d14 = df[df['bar_time'] == '14:30'].copy()
print(f'\nTotal 14:30 bar rows: {len(d14):,}')
for name, mask in [
    ('Tier A (S>0.100 & L<-0.160)', (d14['short_score'] > TIER_A_S_HI) & (d14['long_score'] < TIER_A_L_LO)),
    ('Tier B (S>0.087)',             d14['short_score'] > TIER_B_S_HI),
    ('Tier C (L<-0.167)',            d14['long_score']  < TIER_C_L_LO),
    ('Tier D (L>0.080 & S<-0.200)', (d14['long_score']  > TIER_D_L_HI) & (d14['short_score'] < TIER_D_S_LO)),
]:
    n = mask.sum()
    print(f'  {name}: {n} signals  (~{n*12/5:.0f}/yr annualized)')

# ════════════════════════════════════════════════════════════════════════════
# PART A — Model Accuracy Check (14:30 bar own return = research methodology)
# ════════════════════════════════════════════════════════════════════════════
print('\n' + '='*70)
print('PART A — Model Accuracy at 14:30 Bar (mirrors research methodology)')
print('Uses completed bar features including IBS -> NOT tradeable live')
print('Validates whether model predictions are correct on this OOS period.')
print('='*70)

def model_accuracy(df_bar, signal_mask, side, label):
    hits = df_bar[signal_mask].copy()
    if len(hits) == 0:
        print(f'{label}: 0 signals')
        return None
    if side == 'SHORT':
        hits['win']     = hits['Return'] < 0
        hits['pnl_bps'] = -hits['Return'] * 10000
    else:
        hits['win']     = hits['Return'] > 0
        hits['pnl_bps'] =  hits['Return'] * 10000
    wr      = hits['win'].mean()
    avg_g   = hits['pnl_bps'].mean()
    net_bps = avg_g - COST_BPS
    n       = len(hits)
    print(f'{label}')
    print(f'  n={n} (~{n*12/5:.0f}/yr)  WR={wr:.1%}  AvgGross={avg_g:+.1f}bps  Net={net_bps:+.1f}bps')
    return (wr, net_bps, n)

ra = {}
ra['A'] = model_accuracy(d14,
    (d14['short_score'] > TIER_A_S_HI) & (d14['long_score'] < TIER_A_L_LO),
    'SHORT', 'Tier A  Dual-Lock Short  (S>0.100 & L<-0.160)')
ra['B'] = model_accuracy(d14,
    d14['short_score'] > TIER_B_S_HI,
    'SHORT', 'Tier B  Pure Short       (S>0.087)')
ra['C'] = model_accuracy(d14,
    d14['long_score']  < TIER_C_L_LO,
    'SHORT', 'Tier C  Inverted Long    (L<-0.167)')
ra['D'] = model_accuracy(d14,
    (d14['long_score'] > TIER_D_L_HI) & (d14['short_score'] < TIER_D_S_LO),
    'LONG',  'Tier D  Dual-Lock Long   (L>0.080 & S<-0.200)')

# ════════════════════════════════════════════════════════════════════════════
# PART B — Tradeable Backtest (13:30 signal -> enter 14:30 -> exit 15:30)
# ════════════════════════════════════════════════════════════════════════════
print('\n' + '='*70)
print('PART B — Tradeable Backtest (no lookahead)')
print('Signal: COMPLETED 13:30 bar  |  Entry: 13:30 close  |  Exit: 14:30 close')
print('Cost: 10 bps round-trip')
print('='*70)

d1330 = df[df['bar_time'] == '13:30'][
    ['date','Ticker','Close','long_score','short_score']
].copy()
d1330.columns = ['date','Ticker','entry_price','long_score','short_score']

d1430 = df[df['bar_time'] == '14:30'][
    ['date','Ticker','Close','Return']
].copy()
d1430.columns = ['date','Ticker','exit_price','exit_bar_return']

sim = d1330.merge(d1430, on=['date','Ticker'], how='inner')
print(f'Trade candidates (date x ticker): {len(sim):,}  across {sim["date"].nunique()} trading days\n')

def simulate_tier(sim_df, signal_mask, side, label):
    trades = sim_df[signal_mask].copy()
    if len(trades) == 0:
        print(f'{label}: 0 trades')
        return pd.DataFrame()
    if side == 'SHORT':
        trades['gross_ret'] = (trades['entry_price'] - trades['exit_price']) / trades['entry_price']
    else:
        trades['gross_ret'] = (trades['exit_price'] - trades['entry_price']) / trades['entry_price']
    trades['net_ret'] = trades['gross_ret'] - COST_PCT
    trades['is_win']  = trades['net_ret'] > 0
    n      = len(trades)
    wr     = trades['is_win'].mean()
    avg_g  = trades['gross_ret'].mean() * 10000
    avg_n  = trades['net_ret'].mean()   * 10000
    total  = trades['net_ret'].sum()    * 100
    ann_n  = n * 12 / 5
    cum    = trades['net_ret'].cumsum()
    mdd    = (cum - cum.cummax()).min() * 100
    wins   = trades[trades['net_ret'] > 0]['net_ret'].sum()
    losses = trades[trades['net_ret'] < 0]['net_ret'].sum()
    pf     = abs(wins / losses) if losses != 0 else float('inf')
    print(f'{label}')
    print(f'  Trades={n} (~{ann_n:.0f}/yr)  WR={wr:.1%}  AvgGross={avg_g:+.1f}bps  AvgNet={avg_n:+.1f}bps')
    print(f'  Total(5mo)={total:+.2f}%  MaxDD={mdd:.2f}%  ProfitFactor={pf:.2f}')
    return trades

ta = simulate_tier(sim, (sim['short_score'] > TIER_A_S_HI) & (sim['long_score'] < TIER_A_L_LO), 'SHORT', 'Tier A  Dual-Lock Short  (S>0.100 & L<-0.160)')
tb = simulate_tier(sim,  sim['short_score'] > TIER_B_S_HI,                                       'SHORT', 'Tier B  Pure Short       (S>0.087)')
tc = simulate_tier(sim,  sim['long_score']  < TIER_C_L_LO,                                       'SHORT', 'Tier C  Inverted Long    (L<-0.167)')
td = simulate_tier(sim, (sim['long_score']  > TIER_D_L_HI) & (sim['short_score'] < TIER_D_S_LO), 'LONG',  'Tier D  Dual-Lock Long   (L>0.080 & S<-0.200)')

# ── Monthly breakdown ────────────────────────────────────────────────────────
for tier_label, trades_df in [('Tier B', tb), ('Tier A', ta)]:
    if len(trades_df) > 0:
        print(f'\n--- {tier_label} Monthly Breakdown ---')
        trades_df['month'] = trades_df['date'].str[:7]
        mo = trades_df.groupby('month').agg(
            trades    =('net_ret', 'count'),
            wr        =('is_win',   lambda x: f'{x.mean():.1%}'),
            avg_net   =('net_ret',  lambda x: f'{x.mean()*10000:+.1f}bps'),
            total_pct =('net_ret',  lambda x: f'{x.sum()*100:+.2f}%')
        )
        print(mo.to_string())

# ── Final comparison table ───────────────────────────────────────────────────
print('\n' + '='*75)
print('FINAL SUMMARY — OOS Verification vs Research Claims')
print('Research period: Jul 2025 – May 2026  |  This test: Jan–May 2026')
print('='*75)
print(f'{"Tier":<6} {"Research WR":>12} {"PartA WR":>9} {"PartB WR":>9} {"PartB Net/trade":>16} {"PartB Trades":>12}')
print('-'*75)

claims  = {'A': ('74-76%', 'SHORT'), 'B': ('68%', 'SHORT'), 'C': ('62%', 'SHORT'), 'D': ('60%', 'LONG')}
pb_map  = {'A': ta, 'B': tb, 'C': tc, 'D': td}

for tier, (claim, side) in claims.items():
    a_r   = ra.get(tier)
    a_str = f'{a_r[0]:.1%}' if a_r else 'N/A'
    df_b  = pb_map[tier]
    if len(df_b) > 0:
        b_wr  = df_b['is_win'].mean()
        b_net = df_b['net_ret'].mean() * 10000
        b_n   = len(df_b)
        b_str  = f'{b_wr:.1%}'
        bn_str = f'{b_net:+.1f} bps'
        bnn    = f'{b_n} (~{b_n*12/5:.0f}/yr)'
    else:
        b_str = 'N/A'; bn_str = 'N/A'; bnn = '0'
    print(f'Tier {tier}  {claim:>12} {a_str:>9} {b_str:>9} {bn_str:>16} {bnn:>12}')

print()
print('Part A = Score 14:30 bar with own completed features (matches research, lookahead within bar)')
print('Part B = Tradeable: 13:30 signal -> enter at 13:30 close -> exit at 14:30 close, 10bps cost')

# ── Save results ─────────────────────────────────────────────────────────────
out = {}
for tier, df_b in pb_map.items():
    if len(df_b) > 0:
        out[f'tier_{tier}'] = {
            'trades': len(df_b),
            'win_rate': float(df_b['is_win'].mean()),
            'avg_net_bps': float(df_b['net_ret'].mean() * 10000),
            'total_return_pct': float(df_b['net_ret'].sum() * 100),
            'max_drawdown_pct': float((df_b['net_ret'].cumsum() - df_b['net_ret'].cumsum().cummax()).min() * 100),
        }

import json as j2
with open('data/sniper_oos_2026_results.json', 'w') as f:
    j2.dump({'period': 'Jan-May 2026', 'cost_bps': COST_BPS, 'results_part_b': out}, f, indent=2)
print('\nSaved: data/sniper_oos_2026_results.json')
