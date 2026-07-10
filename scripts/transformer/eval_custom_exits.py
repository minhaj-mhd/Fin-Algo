import os, sys, json
import numpy as np
import pandas as pd
sys.path.append(os.getcwd())

from scripts.transformer.daily_veto_walkforward import topk_picks

DATA_FILE = 'data/ranking_data_daily_macro_v2.csv'
P = 'data/daily_transformer_panel'

# Load the base CSV to get prices and tickers
df = pd.read_csv(DATA_FILE)
df['DateTime'] = pd.to_datetime(df['DateTime']).dt.normalize()
df = df.sort_values(['Ticker', 'DateTime']).reset_index(drop=True)

# Add 1-day, 2-day, and 3-day exit prices
df['Exit_1D_Price'] = df.groupby('Ticker')['Close'].shift(-1)
df['Exit_2D_Price'] = df.groupby('Ticker')['Close'].shift(-2)
df['Exit_3D_Price'] = df.groupby('Ticker')['Close'].shift(-3)

# Filter df for fast lookup
df_lookup = df.set_index(['DateTime', 'Ticker'])

# Load panel mapping
ts = np.load(f'{P}/ts_days.npy')
ts_dates = pd.to_datetime(ts).normalize()
with open(f'{P}/meta.json') as f:
    meta = json.load(f)
tickers = np.array(meta['tickers'])

# Load scores
v2_long = np.load(f'{P}/v2_long_score.npy')
v2_short = np.load(f'{P}/v2_short_score.npy')
oos_mask = np.load(f'{P}/v2_oos_mask.npy')

# Last month filter
last_month_mask = (ts_dates >= '2026-05-01')
oos_days = np.where(oos_mask & last_month_mask)[0]

trades = []

for t in oos_days:
    dt = ts_dates[t]
    
    # LONG PICKS
    valid_l = np.isfinite(v2_long[t])
    picks_l = topk_picks(v2_long[t], valid_l, 1)
    for p in picks_l:
        tk = tickers[p]
        try:
            row = df_lookup.loc[(dt, tk)]
            trades.append({'Date': dt.date(), 'Ticker': tk, 'Side': 'LONG', 'Entry': row['Close'],
                           'P1': row['Exit_1D_Price'], 'P2': row['Exit_2D_Price'], 'P3': row['Exit_3D_Price']})
        except KeyError:
            pass

    # SHORT PICKS
    valid_s = np.isfinite(v2_short[t])
    picks_s = topk_picks(v2_short[t], valid_s, 1)
    for p in picks_s:
        tk = tickers[p]
        try:
            row = df_lookup.loc[(dt, tk)]
            trades.append({'Date': dt.date(), 'Ticker': tk, 'Side': 'SHORT', 'Entry': row['Close'],
                           'P1': row['Exit_1D_Price'], 'P2': row['Exit_2D_Price'], 'P3': row['Exit_3D_Price']})
        except KeyError:
            pass

res = pd.DataFrame(trades)

# Returns computation in BPS
def get_ret(entry, exit, side):
    if side == 'LONG': return (exit / entry - 1) * 10000
    else: return (1 - exit / entry) * 10000

res['R1'] = res.apply(lambda x: get_ret(x['Entry'], x['P1'], x['Side']), axis=1)
res['R2'] = res.apply(lambda x: get_ret(x['Entry'], x['P2'], x['Side']), axis=1)
res['R3'] = res.apply(lambda x: get_ret(x['Entry'], x['P3'], x['Side']), axis=1)

# Base Holds
res['Ret_1D'] = res['R1']
res['Ret_3D'] = res['R3']

# Hybrid: Quit negative trades on day 1
res['Ret_Hybrid'] = np.where(res['R1'] < 0, res['R1'], res['R3'])

# Stop Loss Logic (evaluated at end of Day 1 and Day 2)
def apply_sl(row, sl_pct):
    sl_bps = -sl_pct * 100
    # Day 1 check
    if row['R1'] <= sl_bps:
        return row['R1']
    # Day 2 check
    if row['R2'] <= sl_bps:
        return row['R2']
    # If no SL hit, hold to Day 3
    return row['R3']

res['Ret_SL_2pct'] = res.apply(lambda x: apply_sl(x, 2), axis=1)
res['Ret_SL_3pct'] = res.apply(lambda x: apply_sl(x, 3), axis=1)
res['Ret_SL_5pct'] = res.apply(lambda x: apply_sl(x, 5), axis=1)

# Also apply SL to Hybrid
def apply_hybrid_sl(row, sl_pct):
    sl_bps = -sl_pct * 100
    # Quit negative on day 1 (hybrid rule)
    if row['R1'] < 0:
        return row['R1']
    # Day 2 SL check
    if row['R2'] <= sl_bps:
        return row['R2']
    return row['R3']

res['Ret_Hyb_SL_2pct'] = res.apply(lambda x: apply_hybrid_sl(x, 2), axis=1)

# Summarize Gross Returns
print(f"Total Trades Evaluated: {len(res)} ({len(res[res['Side']=='LONG'])} Long, {len(res[res['Side']=='SHORT'])} Short)")
print("Note: Values are GROSS average returns in bps (subtract 10 bps for net).")
summary = res.groupby('Side')[['Ret_1D', 'Ret_3D', 'Ret_Hybrid', 'Ret_SL_2pct', 'Ret_SL_3pct', 'Ret_SL_5pct', 'Ret_Hyb_SL_2pct']].mean().round(2)
print("\n--- GROSS AVERAGE RETURNS (bps) ---")
print(summary.to_string())

# Apply 10bps cost and show NET
net_summary = summary - 10
print("\n--- NET AVERAGE RETURNS (bps, cost=10bps) ---")
print(net_summary.to_string())
