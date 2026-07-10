"""
Reusable veto-evaluation machinery for the v20 XGB + BCE-transformer overlay.

Extracted from scripts/transformer/v20_bce_veto_walkforward.py so the SAME scoring + bootstrap
logic can run on ANY window — in particular the VAL window during Optuna tuning, while the TEST
window stays frozen for the single final confirmation.

Split of labour:
  * build_scored_window()   — load v20 panel, run XGB long/short, slice to a window. EXPENSIVE,
                              independent of the transformer, so the tuner caches it ONCE.
  * transformer_pup_table() — run a transformer to P(up) per (timestamp, ticker). Per-trial.
  * attach_pup()            — join P(up) onto the scored window.
  * veto_metrics()          — coverage-matched KEPT-vs-ALL Δnet with day-clustered bootstrap,
                              negative control, and a per-block stability floor.

Coverage-matched (not fixed-threshold) on purpose: weighted/focal/hybrid losses shift the P
distribution, so a fixed P>0.50 rule would give each loss a different coverage and make the
objective apples-to-oranges. Fixing coverage compares losses on RANKING, which is what a veto
actually uses. Calibration is a monotone transform → it does not change a coverage rule.

Exploratory only — no Gauntlet verdict, no registry stamp.
"""
import json
import numpy as np
import pandas as pd
import torch

V20_PANEL  = 'data/research/v20_rolling_1h/panel.parquet'
MODEL_LONG = 'models/research/v20_rolling_1h/xgb_long_model.json'
MODEL_SHORT= 'models/research/v20_rolling_1h/xgb_short_model.json'
XGB_META   = 'models/research/v20_rolling_1h/metadata.json'
COSTS      = [6.0, 10.0]


# ── XGB side: expensive, transformer-independent → cache once ────────────────
def load_xgb():
    import xgboost as xgb
    xl = xgb.XGBRanker(); xl.load_model(MODEL_LONG)
    xs = xgb.XGBRanker(); xs.load_model(MODEL_SHORT)
    feat = json.load(open(XGB_META))['features']
    return xl, xs, feat


def build_scored_window(start_ns, end_ns, xgb_long, xgb_short, feat_cols):
    """v20 panel sliced to [start_ns, end_ns], with xgb_long/short scores + join keys.
    Mirrors the exact conventions of v20_bce_veto_walkforward.py (NS-strip, ts*1000)."""
    df = pd.read_parquet(V20_PANEL)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    m = (df['DateTime'] >= pd.Timestamp(start_ns)) & (df['DateTime'] <= pd.Timestamp(end_ns))
    df = df[m].sort_values(['DateTime', 'Ticker']).reset_index(drop=True)
    X = df[feat_cols].to_numpy(dtype=np.float32)
    np.nan_to_num(X, nan=0.0, copy=False)
    df['xgb_long_score']  = xgb_long.predict(X)
    df['xgb_short_score'] = xgb_short.predict(X)
    df['ts_ns']  = df['DateTime'].astype(np.int64) * 1000          # panel is µs → ns
    df['Ticker'] = df['Ticker'].str.replace('.NS', '', regex=False)
    return df


# ── Transformer side: per-trial ─────────────────────────────────────────────
@torch.no_grad()
def transformer_pup_table(model, idx, d, sector_ids, tickers, device):
    """{(ts_ns, ticker): P(up)} for the model over decision timestamps `idx` of panel `d`."""
    from scripts.transformer.train import DecisionDataset
    ds = DecisionDataset(d, idx)
    model.eval()
    tab = {}
    for i in range(len(ds)):
        x1, x15, s1, s15, macro, ybin, y, present, valid = ds[i]
        tt = lambda a: torch.from_numpy(a[None]).to(device)
        with torch.autocast(device_type='cuda', enabled=(device == 'cuda')):
            logit = model(tt(x1), tt(x15), tt(s1), tt(s15), tt(macro),
                          sector_ids, ~tt(present.astype(np.bool_)))
        p = torch.sigmoid(logit.float())[0].cpu().numpy()
        tns = int(d['ts_1h'][idx[i]])
        for j in np.where(present)[0]:
            tab[(tns, tickers[j])] = float(p[j])
    return tab


def attach_pup(df_window, tab):
    rows = [(ts, tk, p) for (ts, tk), p in tab.items()]
    tab_df = pd.DataFrame(rows, columns=['ts_ns', 'Ticker', 'P_up'])
    return df_window.merge(tab_df, on=['ts_ns', 'Ticker'], how='left')


# ── Evaluation ───────────────────────────────────────────────────────────────
def _topk_picks(df, K, side, shuffle, rng):
    """Per-timestamp Top-K picks by the side's XGB score. Returns (days, side_return, P)."""
    score_col = 'xgb_long_score' if side == 'LONG' else 'xgb_short_score'
    days, ret, P = [], [], []
    for ts_val, grp in df.groupby('DateTime'):
        grp = grp.dropna(subset=['Next_Hour_Return', 'P_up'])
        if len(grp) < K + 1:
            continue
        y = grp['Next_Hour_Return'].to_numpy(float)
        if shuffle:
            y = rng.permutation(y)
        sc = grp[score_col].to_numpy(float)
        p  = grp['P_up'].to_numpy(float)
        day = pd.Timestamp(ts_val).normalize()
        for j in np.argsort(-sc)[:K]:
            days.append(day)
            ret.append(y[j] if side == 'LONG' else -y[j])   # SHORT pnl = -return
            P.append(p[j])
    return (np.array(days, dtype='datetime64[D]'), np.array(ret, float), np.array(P, float))


def coverage_keep(P, side, target_cov):
    """Boolean keep-mask achieving ~target_cov coverage by RANK (calibration-free).
    LONG keeps high P, SHORT keeps low P."""
    if len(P) == 0:
        return np.zeros(0, bool)
    if side == 'LONG':
        th = np.quantile(P, 1.0 - target_cov)
        return P > th
    th = np.quantile(P, target_cov)
    return P < th


def _uplift_boot(days, ret, keep, nb, rng):
    """Day-clustered bootstrap of Δnet = mean(ret[keep]) - mean(ret[all]) (cost cancels)."""
    if not keep.any() or not (~keep).any():
        return np.nan, np.nan, np.nan, np.nan
    uniq = np.unique(days)
    idx_by = {dd: np.where(days == dd)[0] for dd in uniq}
    boots = np.empty(nb)
    for b in range(nb):
        ii = np.concatenate([idx_by[dd] for dd in rng.choice(uniq, len(uniq), replace=True)])
        kk = keep[ii]
        boots[b] = (ret[ii][kk].mean() if kk.any() else 0.0) - ret[ii].mean()
    d0 = ret[keep].mean() - ret.mean()
    se = boots.std() + 1e-12
    return d0 * 1e4, d0 / se, np.percentile(boots, 2.5) * 1e4, np.percentile(boots, 97.5) * 1e4


def _keep_mask(P, side, keep_mode, target_cov, th):
    """coverage = keep top `target_cov` by rank (calibration-free, used for tuning);
    fixed = baseline-identical rule (LONG keep P>th, SHORT keep P<1-th)."""
    if keep_mode == 'fixed':
        return (P > th) if side == 'LONG' else (P < (1.0 - th))
    return coverage_keep(P, side, target_cov)


def veto_metrics(df, K, side, target_cov=0.65, keep_mode='coverage', th=0.5,
                 nb=2000, n_blocks=3, n_shuffle=1, seed=42):
    """Veto evaluation for one (K, side). Δnet (kept-all) with day-clustered t and CI, a
    per-block stability floor, and a negative control.

    keep_mode='coverage' (top target_cov by rank) for tuning; 'fixed' (P>th) for the
    baseline-identical comparison. n_shuffle>1 averages the control over many within-timestamp
    return shuffles — REQUIRED for an honest control, since a single shuffle is one noisy draw
    (we found it sat at a misleading +0.4 bps across tuning trials because they shared one seed)."""
    rng = np.random.default_rng(seed)
    days, ret, P = _topk_picks(df, K, side, shuffle=False, rng=rng)
    keep = _keep_mask(P, side, keep_mode, target_cov, th)

    du, dt, dlo, dhi = _uplift_boot(days, ret, keep, nb, rng)

    # negative control: shuffle returns within each timestamp → Δ must collapse to ~0.
    # Average over n_shuffle draws; report mean ± sd so a single noisy draw can't mislead.
    ncs = []
    for _ in range(max(1, n_shuffle)):
        nd, nr, nP = _topk_picks(df, K, side, shuffle=True, rng=rng)
        nk = _keep_mask(nP, side, keep_mode, target_cov, th)
        if nk.any() and len(nr):
            ncs.append((nr[nk].mean() - nr.mean()) * 1e4)
    nc = float(np.mean(ncs)) if ncs else np.nan
    nc_sd = float(np.std(ncs)) if len(ncs) > 1 else 0.0

    # stability floor: Δnet on each contiguous calendar block
    block_deltas = []
    if len(days):
        uniq = np.unique(days)
        for blk in np.array_split(uniq, n_blocks):
            mb = np.isin(days, blk)
            kb = keep[mb]
            if kb.any() and len(kb):
                block_deltas.append((ret[mb][kb].mean() - ret[mb].mean()) * 1e4)
    block_min = float(np.min(block_deltas)) if block_deltas else np.nan

    out = {
        'K': K, 'side': side, 'keep_mode': keep_mode, 'target_cov': target_cov, 'th': th,
        'n_picks': int(len(ret)), 'coverage': float(keep.mean()) if len(keep) else 0.0,
        'uplift_bps': float(du), 'uplift_t': float(dt),
        'uplift_CI_lo': float(dlo), 'uplift_CI_hi': float(dhi),
        'neg_ctrl_uplift': float(nc) if nc is not np.nan else None,
        'neg_ctrl_sd': nc_sd, 'n_shuffle': int(max(1, n_shuffle)),
        'adj_uplift_bps': float(du - nc) if nc is not np.nan else None,  # control-subtracted edge
        'block_min_uplift': block_min,
        'block_deltas': [float(x) for x in block_deltas],
    }
    for cost in COSTS:
        c = cost / 1e4
        out[f'kept_net_{int(cost)}'] = float((ret[keep] - c).mean() * 1e4) if keep.any() else None
        out[f'all_net_{int(cost)}']  = float((ret - c).mean() * 1e4) if len(ret) else None
    return out
