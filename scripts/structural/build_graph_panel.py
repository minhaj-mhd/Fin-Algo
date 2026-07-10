"""
Build the LEVEL-GRAPH node panel, aligned 1:1 to the existing 1h decision grid
(data/transformer_panel_smc/ts_1h.npy) so the gated GCN trains/evaluates on an
IDENTICAL universe to the listwise transformer — a clean architecture comparison.

Per (decision_timestamp t, ticker n): up to K structural nodes (NOW + nearest
S/R / order-blocks / FVGs / round levels), causal, from scripts/structural/level_graph.

Outputs (data/graph_panel_smc/):
  nodes.npy (T1, N, K, NODE_DIM) float32   node_mask.npy (T1, N, K) bool   meta.json

Labels / macro / sectors / present-mask are reused from transformer_panel_smc at train
time (same grid), so this builder only emits the node tensors.

    python scripts/structural/build_graph_panel.py --limit 4   # smoke
    python scripts/structural/build_graph_panel.py             # full
"""
import os, sys, glob, json, argparse, warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings('ignore')
sys.path.append(os.getcwd())
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

import scripts.transformer.build_tensor_panel as B
from scripts.structural.level_graph import build_level_nodes, NODE_DIM, KMAX_DEFAULT

REF = 'data/transformer_panel_smc'      # grid reference (ts_1h, tickers)
OUT = 'data/graph_panel_smc'
K = KMAX_DEFAULT


def h1_frame(ohlc):
    h1 = ohlc[['Open', 'High', 'Low', 'Close', 'Volume']].resample(
        '1h', origin='start_day', offset='15min'
    ).agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'})
    h1 = h1.dropna(subset=['Open', 'Close'])
    h1 = h1[pd.Index(h1.index).strftime('%H:%M').isin(B.VALID_1H_TODS)]
    return h1 if len(h1) >= B.MIN_BARS else None


def main(limit=None):
    os.makedirs(OUT, exist_ok=True)
    meta_ref = json.load(open(f'{REF}/meta.json'))
    tickers = meta_ref['tickers']
    ts1 = np.load(f'{REF}/ts_1h.npy')                       # int64 ns, sorted unique
    tsmap = {int(t): i for i, t in enumerate(ts1)}
    tmap = {t: i for i, t in enumerate(tickers)}
    T1, N = len(ts1), len(tickers)
    print(f"grid: T1={T1}  N={N}  K={K}  NODE_DIM={NODE_DIM}")

    NODES = np.zeros((T1, N, K, NODE_DIM), np.float32)
    MASK = np.zeros((T1, N, K), bool)

    use = tickers[:limit] if limit else tickers
    filled = 0
    for tk in tqdm(use, desc='graphs'):
        ni = tmap[tk]
        ohlc = B.load_ticker_ohlcv(f'{B.CACHE_DIR}/{tk}.csv')
        if ohlc is None:
            continue
        h1 = h1_frame(ohlc)
        if h1 is None:
            continue
        nodes, mask = build_level_nodes(h1, K=K)
        ns = h1.index.values.astype('datetime64[ns]').astype('int64')
        rows = np.array([tsmap.get(int(x), -1) for x in ns])
        ok = rows >= 0
        NODES[rows[ok], ni] = nodes[ok]
        MASK[rows[ok], ni] = mask[ok]
        filled += int(ok.sum())

    print(f"filled {filled:,} (ticker,t) cells; present-NOW cells={int(MASK[:,:,0].sum()):,}")
    np.save(f'{OUT}/nodes.npy', NODES)
    np.save(f'{OUT}/node_mask.npy', MASK)
    meta = {'K': K, 'node_dim': NODE_DIM, 'ref_panel': REF, 'tickers': tickers,
            'n_tickers': N, 'shapes': {'nodes': list(NODES.shape)}, 'smoke_limit': limit,
            'node_layout': '[sdist,adist,is_above,age,strength,mit]+onehot7'
                           '(swing_hi,swing_lo,bull_ob,bear_ob,fvg,round,now)'}
    json.dump(meta, open(f'{OUT}/meta.json', 'w'), indent=2)
    print(f"SAVED -> {OUT}/  nodes {NODES.shape}  "
          f"avg live nodes/present-cell={MASK.sum()/max(MASK[:,:,0].sum(),1):.1f}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()
    main(limit=args.limit)
