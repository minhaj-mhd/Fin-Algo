import pandas as pd
import json
import xgboost as xgb
import numpy as np
from datetime import time, date
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COST_BPS = 6.0
NOTIONAL = 500_000.0
STARTING_CAPITAL = 100_000.0

def main():
    print("Loading data for interactive dashboard...")
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
    df = df[df['DateTime'].dt.date >= date(2025, 8, 1)]
    time_mask = (df['DateTime'].dt.time >= time(10, 15)) & (df['DateTime'].dt.time <= time(14, 15))
    df = df[time_mask]
    
    df['nifty_ret_2h'] = df['DateTime'].map(nifty_map)
    df['nifty_intraday'] = df['DateTime'].map(nifty_intra_map)
    df = df.dropna(subset=['nifty_ret_2h', 'nifty_intraday'])

    v20_feats = json.load(open('models/research/v20_rolling_1h/metadata.json'))['features']
    df = df.dropna(subset=v20_feats + ['Next_Hour_Return'])
    X = xgb.DMatrix(np.nan_to_num(df[v20_feats].values.astype(np.float32)), feature_names=v20_feats)

    bs, bl = xgb.Booster(), xgb.Booster()
    bs.load_model('models/research/v20_rolling_1h/xgb_short_model.json')
    bl.load_model('models/research/v20_rolling_1h/xgb_long_model.json')

    df['ss'], df['ls'] = bs.predict(X), bl.predict(X)
    ss_mean = df.groupby('DateTime')['ss'].transform('mean')
    ls_mean = df.groupby('DateTime')['ls'].transform('mean')
    df['short_conviction'] = (df['ss'] - ss_mean) - (df['ls'] - ls_mean)
    df['long_conviction'] = (df['ls'] - ls_mean) - (df['ss'] - ss_mean)

    trades = []
    slot_free_at = pd.Timestamp.min
    
    for ts, g in df.groupby('DateTime'):
        if ts < slot_free_at: continue
        nifty_2h, nifty_intraday, t = g['nifty_ret_2h'].iloc[0], g['nifty_intraday'].iloc[0], ts.time()
        
        valid_short, valid_long = False, False
        best_short, best_long = None, None
        
        if nifty_2h <= 0.0025 or nifty_intraday > 0.0036:
            if t < time(11, 30) or t > time(13, 0):
                cands = g[g['ss'] > 0.082].sort_values('short_conviction', ascending=False)
                if len(cands) > 0: best_short, valid_short = cands.iloc[0], True
                    
        if nifty_2h > 0.0025 and nifty_intraday > 0.0020:
            cands = g.sort_values('long_conviction', ascending=False)
            if len(cands) > 0: best_long, valid_long = cands.iloc[0], True

        if valid_short and valid_long:
            trades.append((ts, 'SHORT', best_short['Ticker'], -best_short['Next_Hour_Return'] * 10000))
            slot_free_at = ts + pd.Timedelta(hours=1)
        elif valid_short:
            trades.append((ts, 'SHORT', best_short['Ticker'], -best_short['Next_Hour_Return'] * 10000))
            slot_free_at = ts + pd.Timedelta(hours=1)
        elif valid_long:
            trades.append((ts, 'LONG', best_long['Ticker'], best_long['Next_Hour_Return'] * 10000))
            slot_free_at = ts + pd.Timedelta(hours=1)

    td = pd.DataFrame(trades, columns=['ts', 'side', 'tk', 'gross_bps'])
    td['ts'] = pd.to_datetime(td['ts'])
    td = td.sort_values('ts').reset_index(drop=True)
    td['net_bps'] = td.gross_bps - COST_BPS
    td['pnl_rs'] = td.net_bps / 10000 * NOTIONAL
    td['cumulative_pnl'] = td['pnl_rs'].cumsum()
    td['equity'] = STARTING_CAPITAL + td['cumulative_pnl']
    td['peak_equity'] = td['equity'].cummax()
    td['drawdown_rs'] = td['equity'] - td['peak_equity']
    
    # Identify Max DD Point
    mdd_idx = td['drawdown_rs'].idxmin()
    mdd_val = td['drawdown_rs'].min()

    # Generate interactive Plotly HTML
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.08, row_heights=[0.7, 0.3])

    # 1. Equity Curve
    fig.add_trace(go.Scatter(
        x=td['ts'], y=td['equity'],
        mode='lines',
        name='Account Equity',
        line=dict(color='#00E676', width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 230, 118, 0.1)'
    ), row=1, col=1)

    # Max Drawdown Marker
    fig.add_trace(go.Scatter(
        x=[td['ts'].iloc[mdd_idx]], y=[td['equity'].iloc[mdd_idx]],
        mode='markers+text',
        name='Max Drawdown Point',
        marker=dict(color='red', size=10, symbol='x'),
        text=[f"Max DD: ₹{mdd_val:,.0f}"],
        textposition="bottom right",
        textfont=dict(color="red")
    ), row=1, col=1)

    # 2. Drawdown Curve
    fig.add_trace(go.Scatter(
        x=td['ts'], y=td['drawdown_rs'],
        mode='lines',
        name='Drawdown (₹)',
        line=dict(color='#FF1744', width=2),
        fill='tozeroy',
        fillcolor='rgba(255, 23, 68, 0.2)'
    ), row=2, col=1)

    fig.update_layout(
        title=dict(text="<b>1-Slot Portfolio Equity & Drawdown (Starting Capital: ₹1L)</b>", font=dict(size=20, color='white')),
        plot_bgcolor='#1E1E1E',
        paper_bgcolor='#121212',
        font=dict(color='white'),
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.update_yaxes(title_text="Account Equity (₹)", gridcolor='#333333', row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (₹)", gridcolor='#333333', row=2, col=1)
    fig.update_xaxes(gridcolor='#333333')

    out_path = r'C:\Users\loq\.gemini\antigravity\brain\5da8b7a2-8ac8-435c-9e17-cf5e9031f4bc\1slot_dashboard.html'
    fig.write_html(out_path, include_plotlyjs='cdn')
    print(f"Interactive dashboard generated at {out_path}")

if __name__ == '__main__':
    main()
