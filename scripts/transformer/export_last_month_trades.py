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

# Add 1-day and 3-day exit information
df['Exit_1D_Date'] = df.groupby('Ticker')['DateTime'].shift(-1)
df['Exit_1D_Price'] = df.groupby('Ticker')['Close'].shift(-1)
df['Exit_3D_Date'] = df.groupby('Ticker')['DateTime'].shift(-3)
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

K = 1

long_trades = []
short_trades = []

for t in oos_days:
    dt = ts_dates[t]
    
    # LONG PICKS
    valid_l = np.isfinite(v2_long[t])
    picks_l = topk_picks(v2_long[t], valid_l, K)
    for p in picks_l:
        tk = tickers[p]
        try:
            row = df_lookup.loc[(dt, tk)]
            long_trades.append({
                'Date': dt.date(),
                'Ticker': tk,
                'Side': 'LONG',
                'Entry_Price': row['Close'],
                'Exit_1D_Date': row['Exit_1D_Date'].date() if pd.notnull(row['Exit_1D_Date']) else 'N/A',
                'Exit_1D_Price': row['Exit_1D_Price'],
                'Exit_3D_Date': row['Exit_3D_Date'].date() if pd.notnull(row['Exit_3D_Date']) else 'N/A',
                'Exit_3D_Price': row['Exit_3D_Price']
            })
        except KeyError:
            pass

    # SHORT PICKS
    valid_s = np.isfinite(v2_short[t])
    picks_s = topk_picks(v2_short[t], valid_s, K)
    for p in picks_s:
        tk = tickers[p]
        try:
            row = df_lookup.loc[(dt, tk)]
            short_trades.append({
                'Date': dt.date(),
                'Ticker': tk,
                'Side': 'SHORT',
                'Entry_Price': row['Close'],
                'Exit_1D_Date': row['Exit_1D_Date'].date() if pd.notnull(row['Exit_1D_Date']) else 'N/A',
                'Exit_1D_Price': row['Exit_1D_Price'],
                'Exit_3D_Date': row['Exit_3D_Date'].date() if pd.notnull(row['Exit_3D_Date']) else 'N/A',
                'Exit_3D_Price': row['Exit_3D_Price']
            })
        except KeyError:
            pass

res_df = pd.DataFrame(long_trades + short_trades)
res_df = res_df.sort_values(['Date', 'Side'])

# Calculate actual percentages
res_df['1D_Return'] = np.where(res_df['Side'] == 'LONG', 
                               (res_df['Exit_1D_Price'] / res_df['Entry_Price'] - 1) * 1e4,
                               (1 - res_df['Exit_1D_Price'] / res_df['Entry_Price']) * 1e4)
res_df['3D_Return'] = np.where(res_df['Side'] == 'LONG', 
                               (res_df['Exit_3D_Price'] / res_df['Entry_Price'] - 1) * 1e4,
                               (1 - res_df['Exit_3D_Price'] / res_df['Entry_Price']) * 1e4)

print(res_df.to_markdown(index=False, floatfmt=".2f"))
