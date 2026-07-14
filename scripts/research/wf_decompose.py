"""
Decompose the retrained-WF gated strategy: is the edge the MODEL, or the GATE/beta/time?

Neg-controls on the identical gate machinery (per-leg flat net-bps + day-clustered t,
queue-independent so it's a clean edge measure):
  conv        : strategy pick (top-1 by conviction)          [baseline]
  rand_elig   : random name among gate-eligible (keeps ss floor for shorts) [model-selection off]
  rand_all    : random name among ALL names at the scan (ignore ss floor)   [model fully off]
  mean_beta   : mean net-bps of ALL eligible names (pure gate/beta of the moment)
Also: lunch-veto on/off for shorts, and gate-threshold perturbation (+/- the constants).
"""
import numpy as np, pandas as pd, datetime as dt

COST = 6.0
oos = pd.read_parquet('scratch/wf/oos_scored.parquet')
oos['DateTime'] = pd.to_datetime(oos['DateTime'])
oos['date'] = oos['DateTime'].dt.date
oos['ss_m'] = oos.groupby('DateTime')['ss'].transform('mean')
oos['ls_m'] = oos.groupby('DateTime')['ls'].transform('mean')
oos['short_conv'] = (oos['ss'] - oos['ss_m']) - (oos['ls'] - oos['ls_m'])
oos['long_conv'] = (oos['ls'] - oos['ls_m']) - (oos['ss'] - oos['ss_m'])
tmin, tmax = pd.to_datetime('10:15').time(), pd.to_datetime('14:15').time()
oos = oos[(oos['DateTime'].dt.time >= tmin) & (oos['DateTime'].dt.time <= tmax)].copy()

def gates_15m():
    n = pd.read_csv('data/raw_index_cache/nifty50_15m.csv'); n['ts'] = pd.to_datetime(n['ts']); n = n.sort_values('ts')
    n['n2h'] = n['close']/n['close'].shift(8)-1; n['d'] = n['ts'].dt.date
    n = n.merge(n.groupby('d')['open'].first().rename('dop'), on='d', how='left'); n['nin'] = n['close']/n['dop']-1
    return dict(zip(n['ts'], n['n2h'])), dict(zip(n['ts'], n['nin']))

def gates_1h(anchors):
    h = pd.read_csv('data/raw_index_cache/nifty500_1h.csv'); h['ts'] = pd.to_datetime(h['timestamp']); h = h.sort_values('ts')
    h['n2h'] = h['close']/h['close'].shift(2)-1; h['d'] = h['ts'].dt.date
    h = h.merge(h.groupby('d')['open'].first().rename('dop'), on='d', how='left'); h['nin'] = h['close']/h['dop']-1
    a = pd.DataFrame({'ts': sorted(anchors)}).sort_values('ts')
    m = pd.merge_asof(a, h[['ts','n2h','nin']], on='ts', direction='backward', tolerance=pd.Timedelta('90min'))
    return dict(zip(m['ts'], m['n2h'])), dict(zip(m['ts'], m['nin']))

def day_t(dfr):
    dm = dfr.groupby('date')['net_bps'].mean()
    if len(dm) < 2: return np.nan
    return dm.mean()/(dm.std(ddof=1)/np.sqrt(len(dm)))

def eval_leg(df, side, mode, lunch_veto=True, n2h_lo=None, n2h_hi=None, nin_lo=None, seeds=25):
    """Collect one trade per eligible scan under `mode`; return flat avg bps + t_day + n."""
    df = df.copy()
    df['n2h'] = df['DateTime'].map(n2h_map); df['nin'] = df['DateTime'].map(nin_map)
    df = df.dropna(subset=['n2h','nin'])
    picks_all = []  # for deterministic modes
    rand_scans = []  # (date, eligible net_bps array) for random modes
    lunch_a, lunch_b = pd.to_datetime('11:30').time(), pd.to_datetime('13:00').time()
    for ts, g in df.groupby('DateTime'):
        n2h = g['n2h'].iloc[0]; nin = g['nin'].iloc[0]; t = ts.time(); thr = g['ss_thr'].iloc[0]
        if side == 'S':
            if not (n2h <= 0.0025 or nin > 0.0036): continue
            if lunch_veto and (lunch_a <= t <= lunch_b): continue
            elig = g[g['ss'] > thr]
        else:
            if not (n2h > 0.0025 and nin > 0.0020): continue
            elig = g
        if len(elig) == 0: continue
        r = elig['Next_Hour_Return'].values
        bps = (-r if side == 'S' else r)*10000 - COST
        conv = elig['short_conv'].values if side == 'S' else elig['long_conv'].values
        allr = g['Next_Hour_Return'].values
        allbps = (-allr if side == 'S' else allr)*10000 - COST
        if mode == 'conv':
            picks_all.append({'date': g['date'].iloc[0], 'net_bps': bps[np.argmax(conv)]})
        elif mode == 'mean_beta':
            picks_all.append({'date': g['date'].iloc[0], 'net_bps': bps.mean()})
        elif mode == 'rand_elig':
            rand_scans.append((g['date'].iloc[0], bps))
        elif mode == 'rand_all':
            rand_scans.append((g['date'].iloc[0], allbps))
    if mode in ('conv', 'mean_beta'):
        d = pd.DataFrame(picks_all)
        return (d['net_bps'].mean(), day_t(d), len(d)) if len(d) else (np.nan, np.nan, 0)
    # random: average over seeds
    means, ts_list = [], []
    for s in range(seeds):
        rng = np.random.default_rng(s)
        rows = [{'date': dte, 'net_bps': arr[rng.integers(len(arr))]} for dte, arr in rand_scans]
        d = pd.DataFrame(rows); means.append(d['net_bps'].mean()); ts_list.append(day_t(d))
    return (float(np.mean(means)), float(np.mean(ts_list)), len(rand_scans))

def run_window(label, gate_maps, start=None):
    global n2h_map, nin_map
    n2h_map, nin_map = gate_maps
    df = oos if start is None else oos[oos['date'] >= start]
    print(f"\n{'='*88}\n{label}\n{'='*88}")
    print(f"{'leg / mode':32s} | {'avg net bps':>11s} | {'t_day':>6s} | {'n':>5s}")
    for side, nm in [('S','SHORT'), ('L','LONG')]:
        for mode in ['conv', 'rand_elig', 'rand_all', 'mean_beta']:
            b, t, n = eval_leg(df, side, mode)
            print(f"{nm+' '+mode:32s} | {b:+11.2f} | {t:+6.2f} | {n:5d}")
        if side == 'S':  # lunch veto OFF
            b, t, n = eval_leg(df, side, 'conv', lunch_veto=False)
            print(f"{'SHORT conv (NO lunch veto)':32s} | {b:+11.2f} | {t:+6.2f} | {n:5d}")
        print()

run_window("EXACT NIFTY50-15m gates, 2025-08+ (Test 2 window)", gates_15m(), dt.date(2025,8,1))
run_window("APPROX NIFTY500-1h gates, full 2024-2026 (Test 2b window)", gates_1h(oos['DateTime'].unique()))
