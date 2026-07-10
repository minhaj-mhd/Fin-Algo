import pandas as pd
import json
import xgboost as xgb
import numpy as np
from datetime import time, date, datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

COST_BPS = 6.0
NOTIONAL = 500000
BPS_TO_PNL = NOTIONAL / 10000

def main():
    print("Loading data...")
    nifty = pd.read_csv('data/raw_index_cache/nifty50_15m.csv')
    nifty['ts'] = pd.to_datetime(nifty['ts'])
    if nifty['ts'].dt.tz is not None: nifty['ts'] = nifty['ts'].dt.tz_localize(None)
    nifty = nifty.sort_values('ts').reset_index(drop=True)
    
    nifty['nifty_ret_2h'] = nifty['close'] / nifty['close'].shift(8) - 1
    nifty['date'] = nifty['ts'].dt.date
    daily_open = nifty.groupby('date')['open'].first().reset_index()
    daily_open.rename(columns={'open': 'daily_open'}, inplace=True)
    nifty = pd.merge(nifty, daily_open, on='date', how='left')
    nifty['nifty_intraday'] = nifty['close'] / nifty['daily_open'] - 1
    
    nifty_map = dict(zip(nifty['ts'], nifty['nifty_ret_2h']))
    nifty_intra_map = dict(zip(nifty['ts'], nifty['nifty_intraday']))

    df = pd.read_parquet('data/research/v20_rolling_1h/panel_backfilled.parquet')
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    if df['DateTime'].dt.tz is not None: df['DateTime'] = df['DateTime'].dt.tz_localize(None)
    
    # Filter for the exact requested range: 04 Aug 2025 -> 04 Jun 2026
    start_date = date(2025, 8, 4)
    end_date = date(2026, 6, 4)
    df = df[(df['DateTime'].dt.date >= start_date) & (df['DateTime'].dt.date <= end_date)]
    
    time_mask = (df['DateTime'].dt.time >= time(10, 15)) & (df['DateTime'].dt.time <= time(14, 15))
    df = df[time_mask]
    
    df['nifty_ret_2h'] = df['DateTime'].map(nifty_map)
    df['nifty_intraday'] = df['DateTime'].map(nifty_intra_map)
    df = df.dropna(subset=['nifty_ret_2h', 'nifty_intraday'])

    v20_feats = json.load(open('models/research/v20_rolling_1h/metadata.json'))['features']
    df = df.dropna(subset=v20_feats + ['Next_Hour_Return'])
    X = xgb.DMatrix(np.nan_to_num(df[v20_feats].values.astype(np.float32)), feature_names=v20_feats)

    bs = xgb.Booster()
    bs.load_model('models/research/v20_rolling_1h/xgb_short_model.json')
    bl = xgb.Booster()
    bl.load_model('models/research/v20_rolling_1h/xgb_long_model.json')

    df['ss'] = bs.predict(X)
    df['ls'] = bl.predict(X)

    ss_mean = df.groupby('DateTime')['ss'].transform('mean')
    ls_mean = df.groupby('DateTime')['ls'].transform('mean')
    df['short_conviction'] = (df['ss'] - ss_mean) - (df['ls'] - ls_mean)
    df['long_conviction'] = (df['ls'] - ls_mean) - (df['ss'] - ss_mean)

    print("Extracting trades...")
    shorts = []
    longs = []
    
    for ts, g in df.groupby('DateTime'):
        nifty_2h = g['nifty_ret_2h'].iloc[0]
        nifty_intraday = g['nifty_intraday'].iloc[0]
        
        # Shorts
        if nifty_2h <= 0.0025 or nifty_intraday > 0.0036:
            t = ts.time()
            if t < time(11, 30) or t > time(13, 0):
                cands = g[g['ss'] > 0.082].sort_values('short_conviction', ascending=False)
                if len(cands) > 0:
                    p = cands.iloc[0]
                    pnl = (-p['Next_Hour_Return'] * 10000 - COST_BPS) * BPS_TO_PNL
                    shorts.append({'ts': ts, 'pnl': pnl})
                    
        # Longs
        if nifty_2h > 0.0025 and nifty_intraday > 0.0020:
            cands = g.sort_values('long_conviction', ascending=False)
            if len(cands) > 0:
                p = cands.iloc[0]
                pnl = (p['Next_Hour_Return'] * 10000 - COST_BPS) * BPS_TO_PNL
                longs.append({'ts': ts, 'pnl': pnl})

    print(f"Longs: {len(longs)}, Shorts: {len(shorts)}")

    # Create master timeseries index
    all_ts = sorted(list(set(df['DateTime'].unique())))
    res = pd.DataFrame({'ts': all_ts})
    
    sdf = pd.DataFrame(shorts)
    ldf = pd.DataFrame(longs)
    
    res = pd.merge(res, sdf, on='ts', how='left').rename(columns={'pnl': 'short_pnl'}).fillna({'short_pnl': 0})
    res = pd.merge(res, ldf, on='ts', how='left').rename(columns={'pnl': 'long_pnl'}).fillna({'long_pnl': 0})
    
    res['combined_pnl'] = res['short_pnl'] + res['long_pnl']
    
    res['cum_short'] = res['short_pnl'].cumsum()
    res['cum_long'] = res['long_pnl'].cumsum()
    res['cum_combined'] = res['combined_pnl'].cumsum()
    
    res['dd_short'] = res['cum_short'] - res['cum_short'].cummax()
    res['dd_long'] = res['cum_long'] - res['cum_long'].cummax()
    res['dd_combined'] = res['cum_combined'] - res['cum_combined'].cummax()

    # Find Max DD Points
    max_dd_s = res.loc[res['dd_short'].idxmin()]
    max_dd_l = res.loc[res['dd_long'].idxmin()]
    max_dd_c = res.loc[res['dd_combined'].idxmin()]

    print("Generating Plotly Visualization...")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1, 
                        row_heights=[0.7, 0.3],
                        subplot_titles=("Cumulative P&L (₹)", "Drawdown (₹)"))

    # Top Chart: Cumulative P&L
    fig.add_trace(go.Scatter(x=res['ts'], y=res['cum_combined'], mode='lines', name='Combined', line=dict(color='#2ca02c', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=res['ts'], y=res['cum_short'], mode='lines', name='Short Leg', line=dict(color='#d62728', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=res['ts'], y=res['cum_long'], mode='lines', name='Long Leg', line=dict(color='#1f77b4', width=2)), row=1, col=1)

    # Bottom Chart: Drawdowns
    fig.add_trace(go.Scatter(x=res['ts'], y=res['dd_combined'], mode='lines', showlegend=False, line=dict(color='#2ca02c', width=1), fill='tozeroy'), row=2, col=1)
    fig.add_trace(go.Scatter(x=res['ts'], y=res['dd_short'], mode='lines', showlegend=False, line=dict(color='#d62728', width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=res['ts'], y=res['dd_long'], mode='lines', showlegend=False, line=dict(color='#1f77b4', width=1)), row=2, col=1)

    # Mark Max Drawdowns on Bottom Chart
    fig.add_annotation(x=max_dd_c['ts'], y=max_dd_c['dd_combined'], text=f"Combined: -₹{abs(max_dd_c['dd_combined']):,.0f}", showarrow=True, arrowhead=1, row=2, col=1)
    fig.add_annotation(x=max_dd_s['ts'], y=max_dd_s['dd_short'], text=f"Short: -₹{abs(max_dd_s['dd_short']):,.0f}", showarrow=True, arrowhead=1, ay=30, row=2, col=1)
    fig.add_annotation(x=max_dd_l['ts'], y=max_dd_l['dd_long'], text=f"Long: -₹{abs(max_dd_l['dd_long']):,.0f}", showarrow=True, arrowhead=1, ay=60, row=2, col=1)

    # Highlight Out-of-Sample gap slice (Feb 20 2026 to Mar 24 2026)
    fig.add_vrect(x0="2026-02-20", x1="2026-03-24", 
                  fillcolor="black", opacity=0.1, layer="below", line_width=0,
                  annotation_text="Hold-Out Slice (Gap)", annotation_position="top left", row=1, col=1)
    fig.add_vrect(x0="2026-02-20", x1="2026-03-24", 
                  fillcolor="black", opacity=0.1, layer="below", line_width=0, row=2, col=1)

    fig.update_layout(
        title="Strategy Performance: Long vs Short (₹5L/trade, Net of 6 bps)",
        hovermode="x unified",
        template="plotly_white",
        height=800
    )

    out_path = r'C:\Users\loq\.gemini\antigravity\brain\5da8b7a2-8ac8-435c-9e17-cf5e9031f4bc\interactive_dashboard.html'
    fig.write_html(out_path)
    print(f"Saved interactive dashboard to {out_path}")

if __name__ == '__main__':
    main()
