"""
Walk-forward veto evaluation: v20 XGBoost ranker + BCE DualRes Transformer veto.

Protocol:
  1. v20 XGB ranks every stock at every decision timestamp on the rolling-1h panel.
  2. BCE transformer (trained on transformer_panel_v20, 70/15/15 split) scores P(up) per stock.
  3. Veto rule: take a LONG pick only if P(up) > th ; SHORT pick only if P(up) < 1-th.
  4. Evaluate ONLY on the transformer's genuine OOS window (test split, ~Sep2025..Jun2026).
  5. Report KEPT vs VETOED vs ALL net @ 6 and 10 bps per side.
  6. Day-clustered bootstrap CI on uplift Δnet = kept_net - all_net (cost cancels).
  7. Negative control: shuffle returns within each timestamp → uplift must collapse to ~0.

Pre-registered success criterion:
  WIN  = uplift Δnet CI > 0 (lo > 0) AND v20+VETO net >= 1 bps AND vetoed < all.
  FAIL = any of those conditions not met → veto adds nothing, dead-end (no threshold sweep to pass).

Exploratory only — no Gauntlet verdict, no registry stamp.
"""
import os, sys, json
import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_1samp

sys.path.append(os.getcwd())
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

os.environ['TRANSFORMER_PANEL'] = 'data/transformer_panel_v20'
from scripts.transformer.train import load_panel, DecisionDataset, valid_decision_timestamps
from scripts.transformer.model import DualResCSTransformer

# ── Config ──────────────────────────────────────────────────────────────────
V20_PANEL  = 'data/research/v20_rolling_1h/panel.parquet'
MODEL_LONG = 'models/research/v20_rolling_1h/xgb_long_model.json'
MODEL_SHORT= 'models/research/v20_rolling_1h/xgb_short_model.json'
BCE_CKPT   = 'artifacts/dualres_transformer.pt'
COSTS      = [6.0, 10.0]                   # bps
KS         = [1, 3, 5]
THRESHOLDS = [0.50, 0.48, 0.45, 0.42]     # veto if P < 1-th (short) or < th (long)
NB         = 2000                           # bootstrap iterations
SEED       = 42
RNG        = np.random.default_rng(SEED)
device     = 'cuda' if torch.cuda.is_available() else 'cpu'


# ── Step 1: load transformer panel + determine OOS cutoff ───────────────────
print("Loading transformer panel …")
d    = load_panel()
ts_v = valid_decision_timestamps(d)
n    = len(ts_v)
i_te = int(n * 0.85)
EMBARGO = 30
te_idx  = ts_v[i_te + EMBARGO:]
cutoff_ns = int(d['ts_1h'][te_idx[0]])
print(f"  Transformer OOS starts : {pd.Timestamp(cutoff_ns)}  ({len(te_idx):,} timestamps)")

F    = d['meta']['n_features']
M    = d['macro'].shape[1]
n_sec= len(d['meta']['sectors'])
tickers_panel = d['meta']['tickers']          # transformer universe (172)
sector_ids = torch.from_numpy(d['sector_ids'].astype(np.int64)).to(device)


# ── Step 2: BCE transformer inference → {(ts_ns, ticker): P(up)} ────────────
print(f"\nInferring BCE transformer on {len(te_idx):,} OOS timestamps …")
model = DualResCSTransformer(
    F, M, n_sec,
    n_slots_1h=d['meta']['n_slots_1h'],
    n_slots_15m=d['meta']['n_slots_15m'],
    d_model=64,
).to(device)
model.load_state_dict(torch.load(BCE_CKPT, map_location=device))
model.eval()

ds  = DecisionDataset(d, te_idx)
tab = {}                                      # (ts_ns_int, ticker_str) → P(up) float
with torch.no_grad():
    for i in range(len(ds)):
        x1, x15, s1, s15, macro, ybin, y, present, valid = ds[i]
        tt = lambda a: torch.from_numpy(a[None]).to(device)
        with torch.autocast(device_type='cuda', enabled=(device == 'cuda')):
            logit = model(tt(x1), tt(x15), tt(s1), tt(s15), tt(macro),
                          sector_ids, ~tt(present.astype(np.bool_)))
        p = torch.sigmoid(logit.float())[0].cpu().numpy()
        tns = int(d['ts_1h'][te_idx[i]])          # nanoseconds
        for j in np.where(present)[0]:
            tab[(tns, tickers_panel[j])] = float(p[j])
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(te_idx)} …")

print(f"  transformer lookup table: {len(tab):,} (ts,ticker) entries")


# ── Step 3: v20 XGBoost scores on the OOS panel ─────────────────────────────
print("\nLoading v20 XGBoost models and panel …")
try:
    import xgboost as xgb
except ImportError:
    raise SystemExit("xgboost not installed — run: pip install xgboost")

xgb_long  = xgb.XGBRanker(); xgb_long.load_model(MODEL_LONG)
xgb_short = xgb.XGBRanker(); xgb_short.load_model(MODEL_SHORT)

print(f"  Loading panel parquet …")
df_all = pd.read_parquet(V20_PANEL)
df_all['DateTime'] = pd.to_datetime(df_all['DateTime'])

# Restrict to OOS window
df_oos = df_all[df_all['DateTime'] >= pd.Timestamp(cutoff_ns)].copy()
print(f"  OOS rows: {len(df_oos):,}  timestamps: {df_oos['DateTime'].nunique():,}  "
      f"span {df_oos['DateTime'].min()} .. {df_oos['DateTime'].max()}")

# Feature columns — use EXACT list the XGB was trained on (from metadata.json)
xgb_meta  = json.load(open('models/research/v20_rolling_1h/metadata.json'))
FEAT_COLS = xgb_meta['features']          # 86 features, correct order
print(f"  Feature columns for XGB: {len(FEAT_COLS)} (from model metadata)")

# Score per timestamp
df_oos = df_oos.sort_values(['DateTime', 'Ticker']).reset_index(drop=True)
X_oos  = df_oos[FEAT_COLS].to_numpy(dtype=np.float32)
np.nan_to_num(X_oos, nan=0.0, copy=False)

df_oos['xgb_long_score']  = xgb_long.predict(X_oos)
df_oos['xgb_short_score'] = xgb_short.predict(X_oos)

# Attach transformer P(up) — merge on (ts_ns_ns, ticker_no_ns)
# Build a lookup DataFrame from the transformer table (keys already nanoseconds, no .NS)
tab_rows = [(ts_ns, tk, p) for (ts_ns, tk), p in tab.items()]
tab_df   = pd.DataFrame(tab_rows, columns=['ts_ns', 'Ticker', 'P_up'])

# Panel DateTime is microsecond-precision → multiply by 1000 to get nanoseconds
df_oos['ts_ns'] = df_oos['DateTime'].astype(np.int64) * 1000
# Strip .NS suffix from panel tickers to match transformer universe
df_oos['Ticker'] = df_oos['Ticker'].str.replace('.NS', '', regex=False)

df_oos = df_oos.merge(tab_df, on=['ts_ns', 'Ticker'], how='left')

joined = df_oos['P_up'].notna().sum()
print(f"  P_up joined: {joined:,}/{len(df_oos):,} "
      f"({joined/len(df_oos)*100:.1f}%)")


# ── Step 4: per-timestamp Top-K evaluation ───────────────────────────────────
def eval_topk(df, K, th, shuffle=False):
    """Returns arrays: day (date), side_return, is_kept, for LONG and SHORT."""
    L_day, L_ret, L_kept = [], [], []
    S_day, S_ret, S_kept = [], [], []

    for ts_val, grp in df.groupby('DateTime'):
        grp = grp.dropna(subset=['Next_Hour_Return', 'P_up'])
        if len(grp) < K + 1:
            continue

        y = grp['Next_Hour_Return'].to_numpy(float)
        if shuffle:
            y = RNG.permutation(y)
        rl = grp['xgb_long_score'].to_numpy(float)
        rs = grp['xgb_short_score'].to_numpy(float)
        p  = grp['P_up'].to_numpy(float)
        day= pd.Timestamp(ts_val).normalize()

        # LONG: top-K by xgb_long_score
        long_idx = np.argsort(-rl)[:K]
        for j in long_idx:
            L_day.append(day); L_ret.append(y[j])
            L_kept.append(bool(p[j] > th))

        # SHORT: top-K by xgb_short_score (high score = strong short candidate)
        short_idx = np.argsort(-rs)[:K]
        for j in short_idx:
            S_day.append(day); S_ret.append(-y[j])      # flip sign → positive = good short
            S_kept.append(bool(p[j] < 1.0 - th))

    return (np.array(L_day, dtype='datetime64[D]'), np.array(L_ret, float), np.array(L_kept, bool),
            np.array(S_day, dtype='datetime64[D]'), np.array(S_ret, float), np.array(S_kept, bool))


def cluster_boot(days, vals, mask=None):
    """Day-clustered bootstrap of mean(vals[mask]): (mean, t, lo95, hi95) in bps."""
    if mask is not None:
        days, vals = days[mask], vals[mask]
    if len(vals) == 0:
        return np.nan, np.nan, np.nan, np.nan
    uniq = np.unique(days)
    by   = {d: vals[days == d] for d in uniq}
    boots = np.array([np.concatenate([by[d] for d in RNG.choice(uniq, len(uniq), replace=True)]).mean()
                      for _ in range(NB)])
    m  = vals.mean()
    se = boots.std() + 1e-12
    return m * 1e4, m / se, np.percentile(boots, 2.5) * 1e4, np.percentile(boots, 97.5) * 1e4


def uplift_boot(days, r, keep):
    """Day-clustered bootstrap of Δnet = mean(r[keep]) - mean(r[all]) (cost cancels)."""
    if not keep.any() or not (~keep).any():
        return np.nan, np.nan, np.nan, np.nan
    uniq   = np.unique(days)
    idx_by = {d: np.where(days == d)[0] for d in uniq}
    boots  = np.empty(NB)
    for b in range(NB):
        samp = RNG.choice(uniq, len(uniq), replace=True)
        ii   = np.concatenate([idx_by[d] for d in samp])
        kk   = keep[ii]
        boots[b] = (r[ii][kk].mean() if kk.any() else 0.0) - r[ii].mean()
    d0 = (r[keep].mean() if keep.any() else 0.0) - r.mean()
    se = boots.std() + 1e-12
    return d0 * 1e4, d0 / se, np.percentile(boots, 2.5) * 1e4, np.percentile(boots, 97.5) * 1e4


# ── Step 5: Main report ──────────────────────────────────────────────────────
print("\n" + "=" * 80)
print(f"V20 XGB + BCE TRANSFORMER VETO  —  Walk-Forward OOS")
print(f"OOS: {df_oos['DateTime'].min()} .. {df_oos['DateTime'].max()}")
print(f"Timestamps: {df_oos['DateTime'].nunique():,}   Joined P_up: {joined:,}")
print("=" * 80)

results = {}
for K in KS:
    print(f"\n{'#'*20}  Top-{K}  {'#'*20}")
    for th in THRESHOLDS:
        label = f"th={th:.2f}"
        print(f"\n  ── veto threshold {label} (LONG keep if P>{th:.2f}, SHORT keep if P<{1-th:.2f}) ──")
        ld, lr, lk, sd, sr, sk = eval_topk(df_oos, K, th)

        # negative control
        nld, nlr, nlk, nsd, nsr, nsk = eval_topk(df_oos, K, th, shuffle=True)

        for side, days, ret, keep, ndays, nret, nkeep in [
            ('LONG',  ld, lr, lk, nld, nlr, nlk),
            ('SHORT', sd, sr, sk, nsd, nsr, nsk),
        ]:
            n_all   = len(ret)
            n_kept  = keep.sum()
            n_veto  = (~keep).sum()
            cov     = n_kept / max(n_all, 1)

            for cost in COSTS:
                c = cost / 1e4
                all_net  = (ret - c).mean() * 1e4 if n_all > 0 else np.nan
                kp_mean, kp_t, kp_lo, kp_hi = cluster_boot(days, ret - c, mask=keep)
                vt_mean  = (ret[~keep] - c).mean() * 1e4 if n_veto > 0 else np.nan
                du, dt, dlo, dhi = uplift_boot(days, ret, keep)

                # negative control
                nc_all  = (nret - c).mean() * 1e4 if len(nret) > 0 else np.nan
                nc_kp   = ((nret[nkeep] - c).mean() * 1e4 if nkeep.any() else np.nan)
                nc_up   = (nc_kp - nc_all) if (nc_kp is not np.nan and nc_all is not np.nan) else np.nan

                win = (kp_lo is not np.nan and kp_lo > 0 and
                       kp_mean is not np.nan and kp_mean >= 1.0 and
                       vt_mean is not np.nan and vt_mean < all_net)
                flag = " ← WIN" if win else ""

                key = f"K{K}_{side}_{label}_@{int(cost)}bps"
                results[key] = dict(
                    K=K, side=side, th=th, cost_bps=cost,
                    n_all=int(n_all), n_kept=int(n_kept), coverage=float(cov),
                    all_net_bps=float(all_net) if all_net is not np.nan else None,
                    kept_net_bps=float(kp_mean) if kp_mean is not np.nan else None,
                    kept_t=float(kp_t), kept_CI_lo=float(kp_lo), kept_CI_hi=float(kp_hi),
                    vetoed_net_bps=float(vt_mean) if vt_mean is not np.nan else None,
                    uplift_bps=float(du), uplift_t=float(dt),
                    uplift_CI_lo=float(dlo), uplift_CI_hi=float(dhi),
                    neg_ctrl_uplift=float(nc_up) if nc_up is not np.nan else None,
                    win=win,
                )

                print(f"    {side} @{int(cost)}bps  n={n_all} kept={n_kept}({cov:.0%})")
                print(f"      ALL  net={all_net:+.2f}bps")
                print(f"      KEPT net={kp_mean:+.2f}bps  t={kp_t:+.2f}  CI[{kp_lo:+.1f},{kp_hi:+.1f}]{flag}")
                print(f"      VETO net={vt_mean:+.2f}bps")
                print(f"      Δnet={du:+.2f}bps  t={dt:+.2f}  CI[{dlo:+.1f},{dhi:+.1f}]")
                nc_flag = '[OK ~0]' if abs(nc_up) < 1.5 else '[!! leak?]'
                print(f"      neg-ctrl Δ={nc_up:+.2f}bps  {nc_flag}")

# ── Summary ──────────────────────────────────────────────────────────────────
wins = [k for k, v in results.items() if v.get('win')]
print("\n" + "=" * 80)
print("PRE-REGISTERED VERDICT")
print("=" * 80)
print(f"WIN conditions (Δnet CI>0 AND kept_net>=1bps AND vetoed<all): {len(wins)} hits")
for w in wins:
    v = results[w]
    print(f"  {w}: kept={v['kept_net_bps']:+.2f}  Δ={v['uplift_bps']:+.2f}  "
          f"CI[{v['uplift_CI_lo']:+.1f},{v['uplift_CI_hi']:+.1f}]  vetoed={v['vetoed_net_bps']:+.2f}")
if not wins:
    print("  NONE — veto adds no reliable edge. Dead-end.")
print("=" * 80)

os.makedirs('artifacts', exist_ok=True)
out_path = 'artifacts/v20_bce_veto_walkforward.json'
json.dump({'config': {'KS': KS, 'THRESHOLDS': THRESHOLDS, 'COSTS': COSTS, 'NB': NB,
                      'oos_start': str(pd.Timestamp(cutoff_ns)),
                      'oos_end': str(df_oos['DateTime'].max())},
           'results': results, 'wins': wins},
          open(out_path, 'w'), indent=2, default=float)
print(f"\nSaved → {out_path}")
