"""
Precompute v10's Top-5 long/short pick masks aligned to the transformer panel grid, saved as
data/transformer_panel/v10_pickmask_{short,long}.npy ([T,N] bool).

Kept torch-free and run as a standalone process: building this in-process inside train.py (after
`import torch`) segfaults on Windows via the duplicate OpenMP/MKL runtime clash. train.py's
build_v10_pickmask() just np.loads the output here.
"""
import sys
import numpy as np
import pandas as pd
import json

NPZ = 'data/model_analysis/v10_v18_independent/walkforward_preds.npz'
V3 = 'data/ranking_data_upstox_1h_v3_3y.csv'
PANEL = 'data/transformer_panel'
K = 5

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

z = np.load(NPZ, allow_pickle=True)
idx, q = z['idx'], z['q']
vv = pd.read_csv(V3, usecols=['DateTime', 'Ticker'])
dt_ns = pd.to_datetime(vv['DateTime']).values.astype('datetime64[ns]').astype('int64')[idx]
tk = vv['Ticker'].str.replace('.NS', '', regex=False).values[idx]
tickers = json.load(open(f'{PANEL}/meta.json'))['tickers']
tmap = {t: i for i, t in enumerate(tickers)}
ts1 = np.load(f'{PANEL}/ts_1h.npy')
T = len(ts1)
ts_to_t = {int(ts1[t]): t for t in range(T)}

for side, key in [('short', 'rs'), ('long', 'rl')]:
    rank = z[key]                                  # higher = stronger pick (matches argsort[-K:])
    mask = np.zeros((T, len(tickers)), dtype=bool)
    hits = 0
    for qid in np.unique(q):
        rows = np.where(q == qid)[0]
        for r in rows[np.argsort(rank[rows])[-K:]]:
            t = ts_to_t.get(int(dt_ns[r]))
            j = tmap.get(tk[r])
            if t is not None and j is not None:
                mask[t, j] = True
                hits += 1
    np.save(f'{PANEL}/v10_pickmask_{side}.npy', mask)
    print(f"saved v10_pickmask_{side}.npy: {hits:,} pick-cells over {int(mask.any(1).sum()):,} timestamps")
