"""
Univariate feature IC scan — hunt for buried HIGH-|IC| signals, especially strongly NEGATIVE
ones (tradable when inverted; this universe mean-reverts so strength features predict reversal).

For each feature: per-timestamp cross-sectional Spearman vs forward return, then mean IC + t-stat
across timestamps. Ranks features by signed IC so the strongest negative (fade) and positive
(follow) univariate signals surface. Reuses the z-scored panels (fast, in-memory).

  python scripts/analysis/feature_ic_scan.py
Exploratory — NO Gauntlet verdict authority.
"""
import os, sys, json
import numpy as np
from scipy.stats import rankdata

sys.path.append(os.getcwd())


def scan(X, Y, feats, tag):
    T, N, F = X.shape
    ics = np.full((T, F), np.nan)
    for t in range(T):
        y = Y[t]
        m = np.isfinite(y) & np.isfinite(X[t]).all(1)
        if m.sum() < 20:
            continue
        ry = rankdata(y[m])
        ry = (ry - ry.mean()) / (ry.std() + 1e-12)
        Xt = X[t][m]                                   # (n,F)
        rX = np.apply_along_axis(rankdata, 0, Xt)      # rank each feature
        rX = (rX - rX.mean(0)) / (rX.std(0) + 1e-12)
        ics[t] = rX.T @ ry / len(ry)                   # per-feature IC
    mean = np.nanmean(ics, 0)
    se = np.nanstd(ics, 0) / np.sqrt(np.isfinite(ics).sum(0))
    tstat = mean / (se + 1e-12)
    order = np.argsort(mean)                           # ascending: most NEGATIVE first
    print(f"\n===== {tag}  (T={T}, F={F}) =====")
    print("  MOST NEGATIVE IC (tradable inverted = fade):")
    for i in order[:10]:
        print(f"    {feats[i]:<26} IC={mean[i]:+.4f}  t={tstat[i]:+6.1f}")
    print("  MOST POSITIVE IC (follow):")
    for i in order[::-1][:10]:
        print(f"    {feats[i]:<26} IC={mean[i]:+.4f}  t={tstat[i]:+6.1f}")
    # strongest by |IC|
    aord = np.argsort(-np.abs(mean))
    print("  STRONGEST |IC| overall:")
    for i in aord[:8]:
        print(f"    {feats[i]:<26} IC={mean[i]:+.4f}  t={tstat[i]:+6.1f}")
    return mean, tstat


def main():
    # 1h panel
    P = 'data/transformer_panel'
    meta = json.load(open(f'{P}/meta.json'))
    X = np.load(f'{P}/X_1h.npy'); Y = np.load(f'{P}/Y_ret.npy')
    scan(X, Y, meta['features'], '1h next-bar (transformer_panel)')
    del X, Y

    # daily panel (Y_1d and Y_3d)
    DP = 'data/daily_transformer_panel'
    if os.path.exists(f'{DP}/meta.json'):
        dm = json.load(open(f'{DP}/meta.json'))
        Xd = np.load(f'{DP}/X_daily.npy')
        feats = dm.get('features') or [f'f{i}' for i in range(Xd.shape[-1])]
        for lab in ['Y_1d', 'Y_3d']:
            if os.path.exists(f'{DP}/{lab}.npy'):
                Yd = np.load(f'{DP}/{lab}.npy')
                scan(Xd, Yd, feats, f'daily {lab} (daily_transformer_panel)')


if __name__ == '__main__':
    main()
