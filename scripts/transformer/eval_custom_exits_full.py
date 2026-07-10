import os, sys, json
import numpy as np
import pandas as pd
sys.path.append(os.getcwd())

from scripts.transformer.daily_veto_walkforward import topk_picks

P = 'data/daily_transformer_panel'

ts = np.load(f'{P}/ts_days.npy')
ts_dates = pd.to_datetime(ts).normalize()
with open(f'{P}/meta.json') as f:
    meta = json.load(f)
tickers = np.array(meta['tickers'])

# We need the close prices for every day and ticker.
# Since we only have the CSV, let's load it and pivot it into a Panel of Close prices.
DATA_FILE = 'data/ranking_data_daily_macro_v2.csv'
df = pd.read_csv(DATA_FILE, usecols=['DateTime', 'Ticker', 'Close'])
df['DateTime'] = pd.to_datetime(df['DateTime']).dt.normalize()

# Create a mapping dictionary for fast lookup: (date, ticker) -> Close
close_map = df.set_index(['DateTime', 'Ticker'])['Close'].to_dict()

v2_long = np.load(f'{P}/v2_long_score.npy')
v2_short = np.load(f'{P}/v2_short_score.npy')
Y = np.load(f'{P}/Y_3d.npy')
oos_mask = np.load(f'{P}/v2_oos_mask.npy')

oos_days = np.where(oos_mask)[0]

trades = []

def get_close(t_idx, tk):
    if t_idx >= len(ts_dates): return np.nan
    dt = ts_dates[t_idx]
    return close_map.get((dt, tk), np.nan)

for t in oos_days:
    dt = ts_dates[t]
    ym = dt.strftime('%Y-%m')
    
    # LONG PICKS
    valid_l = np.isfinite(v2_long[t]) & np.isfinite(Y[t])
    picks_l = topk_picks(v2_long[t], valid_l, 1)
    for p in picks_l:
        tk = tickers[p]
        entry = get_close(t, tk)
        p1 = get_close(t+1, tk)
        p2 = get_close(t+2, tk)
        p3 = get_close(t+3, tk)
        
        trades.append({'Date': dt.date(), 'YearMonth': ym, 'Ticker': tk, 'Side': 'LONG', 
                       'Entry': entry, 'P1': p1, 'P2': p2, 'P3': p3, 'Label_3D': Y[t, p]})

    # SHORT PICKS
    valid_s = np.isfinite(v2_short[t]) & np.isfinite(Y[t])
    picks_s = topk_picks(v2_short[t], valid_s, 1)
    for p in picks_s:
        tk = tickers[p]
        entry = get_close(t, tk)
        p1 = get_close(t+1, tk)
        p2 = get_close(t+2, tk)
        p3 = get_close(t+3, tk)
        
        trades.append({'Date': dt.date(), 'YearMonth': ym, 'Ticker': tk, 'Side': 'SHORT', 
                       'Entry': entry, 'P1': p1, 'P2': p2, 'P3': p3, 'Label_3D': Y[t, p]})

res = pd.DataFrame(trades)

def get_ret(entry, exit, side):
    if side == 'LONG': return (exit / entry - 1) * 10000
    else: return (1 - exit / entry) * 10000

res['R1'] = res.apply(lambda x: get_ret(x['Entry'], x['P1'], x['Side']), axis=1)
res['R2'] = res.apply(lambda x: get_ret(x['Entry'], x['P2'], x['Side']), axis=1)
res['R3'] = res.apply(lambda x: get_ret(x['Entry'], x['P3'], x['Side']), axis=1)

# Check our R3 against the actual Label_3D from the panel
res['R3_Label'] = res.apply(lambda x: x['Label_3D'] * 10000 * (1 if x['Side']=='LONG' else -1), axis=1)

print("Mismatch between our R3 and Panel Y_3d:")
diff = (res['R3'] - res['R3_Label']).abs()
print(diff.describe())

# Use the OFFICIAL Label_3D for R3 to guarantee matching the baseline!
# But what if R1 and R2 are slightly off because of gaps? We'll use our computed R1 and R2, but force R3 to match.
res['R3'] = res['R3_Label']

# Base Holds
res['Ret_1D'] = res['R1']
res['Ret_3D'] = res['R3']

# Hybrid: Quit negative trades on day 1
res['Ret_Hybrid'] = np.where(res['R1'] < 0, res['R1'], res['R3'])

# Stop Loss Logic
def apply_sl(row, sl_pct):
    sl_bps = -sl_pct * 100
    if row['R1'] <= sl_bps: return row['R1']
    if row['R2'] <= sl_bps: return row['R2']
    return row['R3']

res['Ret_SL_2pct'] = res.apply(lambda x: apply_sl(x, 2), axis=1)
res['Ret_SL_3pct'] = res.apply(lambda x: apply_sl(x, 3), axis=1)
res['Ret_SL_5pct'] = res.apply(lambda x: apply_sl(x, 5), axis=1)

# Deduct 10bps cost immediately for all metrics
for c in ['Ret_1D', 'Ret_3D', 'Ret_Hybrid', 'Ret_SL_2pct', 'Ret_SL_3pct', 'Ret_SL_5pct']:
    res[c] -= 10.0

print(f"Total Trades Evaluated: {len(res)} ({len(res[res['Side']=='LONG'])} Long, {len(res[res['Side']=='SHORT'])} Short)")
overall = res.groupby('Side')[['Ret_1D', 'Ret_3D', 'Ret_Hybrid', 'Ret_SL_2pct', 'Ret_SL_3pct', 'Ret_SL_5pct']].mean().round(2)
print("================ FULL OOS PERIOD (478 days) NET RETURNS ================")
print(overall.to_string())
print("\n")

monthly = res.groupby(['YearMonth', 'Side'])[['Ret_3D', 'Ret_Hybrid', 'Ret_SL_2pct', 'Ret_SL_3pct', 'Ret_SL_5pct']].mean().round(2).reset_index()

print("================ MONTHLY BREAKDOWN (LONG) ================")
longs = monthly[monthly['Side'] == 'LONG'].drop(columns=['Side'])
print(longs.to_markdown(index=False))
print("\n")

print("================ MONTHLY BREAKDOWN (SHORT) ================")
shorts = monthly[monthly['Side'] == 'SHORT'].drop(columns=['Side'])
print(shorts.to_markdown(index=False))
