"""
Market-structure LEVEL GRAPH extractor — lookahead-free, OHLCV-only.

For a single instrument, at every decision bar t it emits the set of *structural
objects* that are "live" in the chart at t — the things smart-money concepts say
orders cluster around:

  node types: SWING_HIGH, SWING_LOW (confirmed fractals = horizontal S/R),
              BULL_OB, BEAR_OB (order blocks), FVG (unfilled fair-value gap),
              ROUND (psychological round-number level), NOW (the query node = price).

Each node carries: signed ATR-distance from current price, |distance|, above/below
flag, age, strength (touch-count / zone size), mitigated flag, and a type one-hot.
A gated GCN (scripts/structural/gated_gcn.py) then message-passes over this graph so
the NOW node aggregates the multi-hop relationships between levels that the flat
scalar features in price_action.py cannot represent.

Output per ticker: nodes (T, K, D) padded to K slots + mask (T, K) bool. Node 0 is
always NOW. This is the per-instrument version of a graph net (intra-instrument price
structure), distinct from the inter-stock relation graph in build_relation_graph.py.

CAUSALITY CONTRACT (perturbation-tested in __main__): a node is only emitted once its
defining bars are <= t. Confirmed pivots use a release lag; OB/FVG tracked forward-only.
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.append(os.getcwd())
from scripts.features.price_action import _atr, _confirmed_pivots

# node-type ids
T_SWING_HIGH, T_SWING_LOW, T_BULL_OB, T_BEAR_OB, T_FVG, T_ROUND, T_NOW = range(7)
N_TYPES = 7
# feature layout: [sdist, adist, is_above, age, strength, mitigated] + one-hot(7)
NODE_DIM = 6 + N_TYPES
KMAX_DEFAULT = 24


def _round_levels(close, atr):
    step = 50.0 if close >= 1000 else (5.0 if close >= 100 else 0.5)
    base = np.round(close / step) * step
    return [base - step, base, base + step]


def build_level_nodes(df: pd.DataFrame, K: int = KMAX_DEFAULT, w: int = 2,
                      disp_mult: float = 1.5, atr_window: int = 14):
    """Return (nodes [T,K,NODE_DIM] float32, mask [T,K] bool) for one ticker's bar frame.
    Node 0 = NOW (always valid). Remaining slots = the K-1 structural nodes nearest in price
    (balanced above/below), all causal."""
    o = df["Open"].to_numpy(); h = df["High"].to_numpy()
    l = df["Low"].to_numpy(); c = df["Close"].to_numpy()
    atr = _atr(df["High"], df["Low"], df["Close"], atr_window).to_numpy()
    n = len(c)
    body = np.abs(c - o)

    hi_events, lo_events = _confirmed_pivots(df["High"], df["Low"], w=w)

    nodes = np.zeros((n, K, NODE_DIM), np.float32)
    mask = np.zeros((n, K), bool)

    # live structural objects: each = dict(level, type, born, strength, mitigated_flag, lo, hi)
    swing_hi: list[dict] = []   # confirmed swing highs
    swing_lo: list[dict] = []
    obs: list[dict] = []        # order blocks (zone lo/hi)
    fvgs: list[dict] = []       # unfilled fvg (zone lo/hi)
    jh = jl = 0
    KEEP = 40                    # cap live objects per family

    for t in range(n):
        a = atr[t] if atr[t] > 0 else (abs(c[t]) * 1e-3 + 1e-9)
        # release confirmed pivots visible by t
        while jh < len(hi_events) and hi_events[jh][0] <= t:
            lvl = hi_events[jh][1]
            swing_hi.append({'level': lvl, 'type': T_SWING_HIGH, 'born': hi_events[jh][0],
                             'strength': 1, 'mit': 0})
            jh += 1
        while jl < len(lo_events) and lo_events[jl][0] <= t:
            lvl = lo_events[jl][1]
            swing_lo.append({'level': lvl, 'type': T_SWING_LOW, 'born': lo_events[jl][0],
                             'strength': 1, 'mit': 0})
            jl += 1
        swing_hi = swing_hi[-KEEP:]; swing_lo = swing_lo[-KEEP:]
        # order blocks: impulsive bar at t whose prior bar is opposite colour
        if t >= 1 and body[t] >= disp_mult * a:
            if c[t] > o[t] and c[t - 1] < o[t - 1]:
                obs.append({'lo': l[t - 1], 'hi': h[t - 1], 'level': (l[t-1]+h[t-1])/2,
                            'type': T_BULL_OB, 'born': t, 'strength': body[t] / a, 'mit': 0})
            elif c[t] < o[t] and c[t - 1] > o[t - 1]:
                obs.append({'lo': l[t - 1], 'hi': h[t - 1], 'level': (l[t-1]+h[t-1])/2,
                            'type': T_BEAR_OB, 'born': t, 'strength': body[t] / a, 'mit': 0})
            obs = obs[-KEEP:]
        # FVG formed by t-2,t-1,t
        if t >= 2:
            if l[t] > h[t - 2]:
                fvgs.append({'lo': h[t - 2], 'hi': l[t], 'level': (h[t-2]+l[t])/2,
                             'type': T_FVG, 'born': t, 'strength': (l[t]-h[t-2])/a, 'mit': 0})
            elif h[t] < l[t - 2]:
                fvgs.append({'lo': h[t], 'hi': l[t - 2], 'level': (h[t]+l[t-2])/2,
                             'type': T_FVG, 'born': t, 'strength': (l[t-2]-h[t])/a, 'mit': 0})
            fvgs = fvgs[-KEEP:]
        # mitigation: mark objects the current bar has traded into (zone) or pivots touched
        for obj in obs + fvgs:
            if l[t] <= obj['hi'] and h[t] >= obj['lo']:
                obj['mit'] = 1
        for obj in swing_hi + swing_lo:
            if abs(c[t] - obj['level']) <= 0.25 * a:
                obj['strength'] += 1   # a touch strengthens the level

        # round levels (recomputed each bar, no state)
        round_objs = [{'level': rl, 'type': T_ROUND, 'born': t, 'strength': 1, 'mit': 0}
                      for rl in _round_levels(c[t], a)]

        # assemble candidate nodes, split above/below current price
        cand = swing_hi + swing_lo + obs + fvgs + round_objs
        above = sorted([x for x in cand if x['level'] >= c[t]], key=lambda x: x['level'] - c[t])
        below = sorted([x for x in cand if x['level'] < c[t]], key=lambda x: c[t] - x['level'])
        half = (K - 1) // 2
        chosen = above[:half] + below[:(K - 1 - half)]
        # NOW node (slot 0)
        nodes[t, 0] = _node_vec(0.0, 0.0, 0, t, t, 0, 0, T_NOW, a)
        mask[t, 0] = True
        for j, obj in enumerate(chosen[:K - 1], start=1):
            sdist = (obj['level'] - c[t]) / a
            nodes[t, j] = _node_vec(sdist, abs(sdist), int(obj['level'] >= c[t]),
                                    t - obj['born'], t, obj['strength'], obj['mit'], obj['type'], a)
            mask[t, j] = True
    return nodes, mask


def _node_vec(sdist, adist, is_above, age_bars, t, strength, mit, ntype, a):
    v = np.zeros(NODE_DIM, np.float32)
    v[0] = np.clip(sdist, -8, 8)
    v[1] = np.clip(adist, 0, 8)
    v[2] = float(is_above)
    v[3] = np.clip(age_bars / 100.0, 0, 3)
    v[4] = np.clip(strength / 5.0, 0, 4)
    v[5] = float(mit)
    v[6 + ntype] = 1.0
    return v


# ── smoke + causality test ──────────────────────────────────────────────────────
def _smoke_test():
    rng = np.random.default_rng(1)
    n = 600
    close = 1000 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    high = close * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.004, n)))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.002, n))
    df = pd.DataFrame({"Open": open_, "High": np.maximum.reduce([high, open_, close]),
                       "Low": np.minimum.reduce([low, open_, close]), "Close": close,
                       "Volume": rng.integers(1e4, 1e6, n)},
                      index=pd.date_range("2024-01-01", periods=n, freq="h"))
    nodes, mask = build_level_nodes(df)
    print(f"[smoke] nodes {nodes.shape} mask {mask.shape}  "
          f"avg live nodes/bar={mask.sum(1).mean():.1f} (incl NOW)  "
          f"finite={np.isfinite(nodes).all()}")
    # NOW node always present
    assert mask[:, 0].all(), "NOW node missing"
    # causality: corrupt bars > t0; nodes/mask at <= t0 must be unchanged
    t0 = 400
    df2 = df.astype(float).copy(); df2.iloc[t0 + 1:] = df2.iloc[t0 + 1:] * 1.5
    nodes2, mask2 = build_level_nodes(df2)
    dn = np.abs(nodes[:t0 + 1] - nodes2[:t0 + 1]).max()
    dm = (mask[:t0 + 1] != mask2[:t0 + 1]).sum()
    assert dn < 1e-6 and dm == 0, f"LOOKAHEAD LEAK nodes {dn} mask {dm}"
    print(f"[causality] OK — corrupting bars > {t0} left nodes/mask <= {t0} unchanged "
          f"(node max diff {dn:.2e}, mask diffs {dm})")


if __name__ == "__main__":
    import os, sys
    sys.path.append(os.getcwd())
    _smoke_test()
