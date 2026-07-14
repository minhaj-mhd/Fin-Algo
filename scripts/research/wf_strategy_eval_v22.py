import json, os, numpy as np, pandas as pd
import matplotlib.pyplot as plt

COST = 6.0 # 6 bps cost
oos = pd.read_parquet('scratch/wf_v22/oos_scored.parquet')
oos['DateTime'] = pd.to_datetime(oos['DateTime'])
oos['date'] = oos['DateTime'].dt.date
oos['year'] = oos['DateTime'].dt.year

# per-scan mean-centred convictions
oos['ss_m'] = oos.groupby('DateTime')['ss'].transform('mean')
oos['ls_m'] = oos.groupby('DateTime')['ls'].transform('mean')
oos['short_conv'] = (oos['ss'] - oos['ss_m']) - (oos['ls'] - oos['ls_m'])
oos['long_conv'] = (oos['ls'] - oos['ls_m']) - (oos['ss'] - oos['ss_m'])

win = oos.copy()
print(f"OOS scored: {len(oos):,} rows | trade-window {len(win):,} | span {oos['date'].min()}..{oos['date'].max()}")

def day_t(df, bps_col='net_bps'):
    dm = df.groupby('date')[bps_col].mean()
    if len(dm) < 2: return np.nan, len(dm)
    return dm.mean() / (dm.std(ddof=1) / np.sqrt(len(dm))), len(dm)

def book_stats(trades, label):
    if len(trades) == 0:
        return f"{label:28s} | 0 trades"
    t, nd = day_t(trades)
    net = trades['net_bps']
    return (f"{label:28s} | n {len(trades):4d} | WR {(net>0).mean():5.1%} | "
            f"net {net.mean():+6.2f}bps | t_day {t:+5.2f} | sum {net.sum():+8.0f}bps")

def topk_book(g_df, conv_col, side, k):
    rows = []
    for ts, g in g_df.groupby('DateTime'):
        gg = g.nlargest(k, conv_col)
        r = gg['Next_Hour_Return'].values
        bps = (-r if side == 'S' else r) * 10000 - COST
        for b in bps: rows.append({'date': gg['date'].iloc[0], 'datetime': ts, 'net_bps': b})
    return pd.DataFrame(rows)

out_text = []
out_text.append("=== V22 CORE ENGINE (retrained WF, NO gates) ===")
trades_dict = {}

for side, conv in [('S', 'short_conv'), ('L', 'long_conv')]:
    name = 'SHORT' if side == 'S' else 'LONG'
    for k in (1, 3):
        bk = topk_book(win, conv, side, k)
        stat_line = book_stats(bk, f"{name} top-{k} (all yrs)")
        print(stat_line)
        out_text.append(stat_line)
        if k == 1:
            trades_dict[name] = bk.groupby('datetime')['net_bps'].mean()
            
out_file = r'C:\Users\loq\.gemini\antigravity\brain\377526e7-2144-492d-aadd-efe9a4c24ef6\v22_metrics.txt'
with open(out_file, 'w') as f:
    f.write("\n".join(out_text))

# Plotting Equity Curve & Drawdowns for Top-1
plt.figure(figsize=(12, 10))

# LONG Equity
plt.subplot(2, 2, 1)
long_eq = trades_dict['LONG'].cumsum()
plt.plot(long_eq.index, long_eq.values, color='green')
plt.title("V22 LONG Top-1 Equity (bps)")
plt.grid(True, alpha=0.3)

# LONG Drawdown
plt.subplot(2, 2, 3)
long_peak = long_eq.cummax()
long_dd = long_eq - long_peak
plt.fill_between(long_dd.index, long_dd.values, 0, color='red', alpha=0.3)
plt.title("V22 LONG Drawdown (bps)")
plt.grid(True, alpha=0.3)

# SHORT Equity
plt.subplot(2, 2, 2)
short_eq = trades_dict['SHORT'].cumsum()
plt.plot(short_eq.index, short_eq.values, color='blue')
plt.title("V22 SHORT Top-1 Equity (bps)")
plt.grid(True, alpha=0.3)

# SHORT Drawdown
plt.subplot(2, 2, 4)
short_peak = short_eq.cummax()
short_dd = short_eq - short_peak
plt.fill_between(short_dd.index, short_dd.values, 0, color='red', alpha=0.3)
plt.title("V22 SHORT Drawdown (bps)")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plot_path = r'C:\Users\loq\.gemini\antigravity\brain\377526e7-2144-492d-aadd-efe9a4c24ef6\v22_equity_curve.png'
plt.savefig(plot_path)
print(f"Plot saved to {plot_path}")
