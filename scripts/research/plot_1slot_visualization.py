import pandas as pd
import json
import xgboost as xgb
import numpy as np
from datetime import time, date, datetime
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COST_BPS = 6.0
NOTIONAL = 500_000.0

def main():
    print("Loading data for visualization...")
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

    # Calculate cumulative series per leg
    all_dates = pd.date_range(td['ts'].min().date(), td['ts'].max().date(), freq='D')
    plot_df = pd.DataFrame({'date': all_dates})
    
    td['date'] = td['ts'].dt.date
    daily_pnl = td.groupby(['date', 'side'])['pnl_rs'].sum().unstack(fill_value=0).reset_index()
    daily_pnl['date'] = pd.to_datetime(daily_pnl['date'])
    
    plot_df = pd.merge(plot_df, daily_pnl, on='date', how='left').fillna(0)
    
    if 'SHORT' not in plot_df.columns: plot_df['SHORT'] = 0
    if 'LONG' not in plot_df.columns: plot_df['LONG'] = 0
    
    plot_df['Combined'] = plot_df['SHORT'] + plot_df['LONG']
    
    plot_df['cum_short'] = plot_df['SHORT'].cumsum()
    plot_df['cum_long'] = plot_df['LONG'].cumsum()
    plot_df['cum_combined'] = plot_df['Combined'].cumsum()
    
    def calc_dd(series):
        peak = series.cummax()
        return series - peak
        
    plot_df['dd_short'] = calc_dd(plot_df['cum_short'])
    plot_df['dd_long'] = calc_dd(plot_df['cum_long'])
    plot_df['dd_combined'] = calc_dd(plot_df['cum_combined'])

    # Find max DD points
    mdd_c_idx = plot_df['dd_combined'].idxmin()
    mdd_s_idx = plot_df['dd_short'].idxmin()
    mdd_l_idx = plot_df['dd_long'].idxmin()

    # Create Plot
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, row_heights=[0.7, 0.3])
    
    colors = {'Combined': '#00E676', 'Short': '#FF5252', 'Long': '#29B6F6'}
    
    # 1. Equity Curves
    fig.add_trace(go.Scatter(x=plot_df['date'], y=plot_df['cum_combined'], name='Combined P&L', line=dict(color=colors['Combined'], width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df['date'], y=plot_df['cum_short'], name='Short P&L', line=dict(color=colors['Short'], width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=plot_df['date'], y=plot_df['cum_long'], name='Long P&L', line=dict(color=colors['Long'], width=1.5)), row=1, col=1)
    
    # 2. Drawdown Curves
    fig.add_trace(go.Scatter(x=plot_df['date'], y=plot_df['dd_combined'], name='Combined DD', line=dict(color=colors['Combined'], width=2), fill='tozeroy'), row=2, col=1)
    fig.add_trace(go.Scatter(x=plot_df['date'], y=plot_df['dd_short'], name='Short DD', line=dict(color=colors['Short'], width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=plot_df['date'], y=plot_df['dd_long'], name='Long DD', line=dict(color=colors['Long'], width=1)), row=2, col=1)

    # Shading the gap slice
    gap_start = "2026-02-20"
    gap_end = "2026-03-24"
    for r in [1, 2]:
        fig.add_vrect(x0=gap_start, x1=gap_end, fillcolor="white", opacity=0.08, layer="below", line_width=0,
                      annotation_text="Backfilled Slice" if r==1 else "", annotation_position="top left", row=r, col=1)

    # Markers for max DD
    for leg, idx, y_col, c in [('Combined', mdd_c_idx, 'dd_combined', 'Combined'), ('Short', mdd_s_idx, 'dd_short', 'Short'), ('Long', mdd_l_idx, 'dd_long', 'Long')]:
        val = plot_df[y_col].iloc[idx]
        dt_val = plot_df['date'].iloc[idx]
        fig.add_trace(go.Scatter(x=[dt_val], y=[val], mode='markers+text', marker=dict(color=colors[c], size=8, symbol='x'),
                                 text=[f"Max {leg} DD: ₹{val:,.0f}"], textposition="bottom center", textfont=dict(color=colors[c]), showlegend=False), row=2, col=1)

    # Calculate final returns
    ret_c = plot_df['cum_combined'].iloc[-1]
    ret_s = plot_df['cum_short'].iloc[-1]
    ret_l = plot_df['cum_long'].iloc[-1]
    
    # Add annotation box
    fig.add_annotation(
        text=(f"<b>Total Returns (on ₹1L Base)</b><br>"
              f"Combined: +₹{ret_c:,.0f} (+{ret_c/1000:.1f}%)<br>"
              f"Short Leg: +₹{ret_s:,.0f} (+{ret_s/1000:.1f}%)<br>"
              f"Long Leg: +₹{ret_l:,.0f} (+{ret_l/1000:.1f}%)"),
        align='left', showarrow=False, xref='paper', yref='paper', x=0.02, y=0.95,
        bgcolor="rgba(0,0,0,0.7)", bordercolor="white", borderwidth=1, font=dict(size=13)
    )

    fig.update_layout(
        title=dict(text="<b>1-Slot Multi-Leg Strategy Performance (Aug '25 - Jun '26)</b>", font=dict(size=20, color='white')),
        plot_bgcolor='#1E1E1E', paper_bgcolor='#121212', font=dict(color='white'), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.update_yaxes(title_text="Cumulative P&L (₹)", gridcolor='#333333', row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (₹)", gridcolor='#333333', row=2, col=1)
    fig.update_xaxes(gridcolor='#333333')

    out_path = r'C:\Users\loq\.gemini\antigravity\brain\5da8b7a2-8ac8-435c-9e17-cf5e9031f4bc\1slot_interactive_dashboard.html'
    fig.write_html(out_path, include_plotlyjs='cdn')
    print(f"Interactive dashboard generated at {out_path}")

if __name__ == '__main__':
    main()
