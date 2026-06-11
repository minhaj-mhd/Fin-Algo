"""
Deep numerical Prediction-Bucket + Calibration evaluation, parametrized for clean/native models.
  python scripts/analysis/eval_buckets_calibration_clean.py --model v10_native_1h
  python scripts/analysis/eval_buckets_calibration_clean.py --model v3_15min_clean
"""
import os, sys, json, argparse, warnings
import numpy as np, pandas as pd, xgboost as xgb
from scipy.stats import spearmanr, rankdata, ttest_1samp, mannwhitneyu
warnings.filterwarnings('ignore'); sys.path.append(os.getcwd())

CFG = {
    'v10_native_1h': dict(model_dir='models/v10_native_1h', data='data/ranking_data_upstox_1h_v3_3y.csv', ret='Next_Hour_Return', label='v10_native_1h'),
    'v3_15min_clean': dict(model_dir='models/v3_15min_clean', data='data/ranking_data_upstox_15min_3y_clean.csv', ret='Next_15Min_Return', label='v3_15min_clean'),
    'v9_clean_1h': dict(model_dir='models/v9_clean_1h', data='data/ranking_data_upstox_1h_3y_clean.csv', ret='Next_Hour_Return', label='v9_clean_1h'),
}
ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True, choices=list(CFG)); A = ap.parse_args(); C = CFG[A.model]
MODEL_DIR, DATA_FILE, RET, LABEL = C['model_dir'], C['data'], C['ret'], C['label']
OOS_MONTHS, NB, NC = 3, 10, 10
SEP = "=" * 72
def hdr(t): print(f"\n{SEP}\n  {t}\n{SEP}")

hdr(f"Loading {LABEL} & OOS data")
with open(f'{MODEL_DIR}/metadata.json') as f: meta = json.load(f)
fc = meta['features']
bl = xgb.Booster(); bl.load_model(f'{MODEL_DIR}/xgb_long_model.json')
bs = xgb.Booster(); bs.load_model(f'{MODEL_DIR}/xgb_short_model.json')
am = set()
for ch in pd.read_csv(DATA_FILE, usecols=['DateTime'], chunksize=500_000): am.update(ch['DateTime'].str[:7].unique())
oos = sorted(am)[-OOS_MONTHS:]; print(f"  OOS {oos[0]} -> {oos[-1]}")
chunks = []
for ch in pd.read_csv(DATA_FILE, chunksize=200_000):
    s = ch[ch['DateTime'].str[:7].isin(oos)]
    if len(s): chunks.append(s)
df = pd.concat(chunks, ignore_index=True); df['YearMonth'] = df['DateTime'].str[:7]
print(f"  Rows {len(df):,} | Queries {df['Query_ID'].nunique():,}")
X = df[fc].values.astype(float)
for ci in range(X.shape[1]):
    c = X[:, ci]; b = np.isnan(c) | np.isinf(c)
    if b.any(): X[b, ci] = float(np.nanmean(c[~b])) if (~b).any() else 0.0
df['long_score'] = bl.predict(xgb.DMatrix(X)); df['short_score'] = bs.predict(xgb.DMatrix(X))
mkt_mean = df[RET].mean()
print(f"  Market avg return {mkt_mean*100:+.5f}%")

hdr("SECTION 1 - PREDICTION BUCKET (Score Decile -> Return)")
for label, sc, inv, note in [("LONG", 'long_score', False, ""), ("SHORT", 'short_score', True, "  [negated return = short profit]")]:
    print(f"\n{'-'*72}\n  {label} MODEL{note}\n{'-'*72}")
    df['bk'] = pd.qcut(df[sc], NB, labels=False, duplicates='drop')
    rows = []
    for b in sorted(df['bk'].dropna().unique()):
        g = df[df['bk'] == b][RET]; vals = (-g.values if inv else g.values); n = len(vals)
        if n == 0: continue
        mean, med, std = vals.mean(), np.median(vals), vals.std(); sem = std/np.sqrt(n)
        t, p = ttest_1samp(vals, 0); sig = "***" if p < .001 else ("**" if p < .01 else ("*" if p < .05 else "   "))
        edge = mean - (-mkt_mean if inv else mkt_mean)
        rows.append(dict(b=int(b)+1, n=n, mean=mean, med=med, std=std, ci=1.96*sem, p=p, sig=sig, edge=edge, wr=(vals > 0).mean()))
    print(f"\n  {'D':>3} {'N':>7} {'Mean%':>10} {'Median%':>9} {'Std%':>8} {'95%CI':>9} {'vsMkt%':>9} {'Win%':>6} {'p':>8} Sig")
    for r in rows:
        print(f"  {r['b']:>3} {r['n']:>7,} {r['mean']*100:>+9.5f}% {r['med']*100:>+8.5f}% {r['std']*100:>7.4f}% "
              f"{r['ci']*100:>8.5f}% {r['edge']*100:>+8.5f}% {r['wr']:>5.1%} {r['p']:>8.5f} {r['sig']}")
    means = [r['mean'] for r in rows]; rho, rp = spearmanr(range(len(means)), means)
    d1, d10 = rows[0]['mean'], rows[-1]['mean']
    inv_ct = sum(1 for i in range(len(means)-1) if means[i] > means[i+1])
    print(f"\n  Bucket Spearman Rho: {rho:+.4f} (p={rp:.5f})")
    print(f"  D1 {d1*100:+.5f}% | D10 {d10*100:+.5f}% | spread {(d10-d1)*100:+.5f}%")
    print(f"  Monotonicity: {NB-inv_ct}/{NB} in order ({inv_ct} inversions)")
    print(f"  Stat-sig buckets (p<0.05): {sum(1 for r in rows if r['p']<.05)}/{len(rows)}")
    print(f"  Per-month bucket Rho:")
    for ym in oos:
        dm = df[df['YearMonth'] == ym].copy()
        if len(dm) < 500: continue
        dm['bk'] = pd.qcut(dm[sc], NB, labels=False, duplicates='drop')
        bm = [((-dm[dm['bk']==b][RET].values if inv else dm[dm['bk']==b][RET].values).mean()) for b in sorted(dm['bk'].dropna().unique())]
        rm, _ = spearmanr(range(len(bm)), bm); print(f"    {ym}: Rho {rm:+.4f}")

hdr("SECTION 2 - CALIBRATION (Predicted Rank Pct -> Realized Return)")
for label, sc, inv in [("LONG", 'long_score', False), ("SHORT", 'short_score', True)]:
    print(f"\n{'-'*72}\n  {label} MODEL\n{'-'*72}")
    pp, ar = [], []
    for qid, q in df.groupby('Query_ID'):
        if len(q) < 4: continue
        pp.extend((rankdata(q[sc].values, method='average')/len(q)).tolist())
        ar.extend((q[RET].values*(-1 if inv else 1)).tolist())
    pa, ra = np.array(pp), np.array(ar); bins = np.linspace(0, 1, NC+1); bi = np.digitize(pa, bins[1:-1]); gm = ra.mean()
    print(f"  Samples {len(pa):,}")
    print(f"\n  {'Bin':>4} {'Range':>12} {'N':>7} {'Mean%':>10} {'Median%':>9} {'vsGlobal%':>10} {'Win%':>7} Sig")
    bc, bmean = [], []
    for b in range(NC):
        m = bi == b
        if m.sum() < 50: continue
        v = ra[m]; mean = v.mean(); t, p = ttest_1samp(v, gm); sig = "***" if p < .001 else ("**" if p < .01 else ("*" if p < .05 else "   "))
        print(f"  {b+1:>4} [{bins[b]:.2f}-{bins[b+1]:.2f}] {m.sum():>7,} {mean*100:>+9.5f}% {np.median(v)*100:>+8.5f}% "
              f"{(mean-gm)*100:>+9.5f}% {(v>0).mean():>6.1%} {sig}")
        bc.append((bins[b]+bins[b+1])/2); bmean.append(mean)
    rc, pc = spearmanr(bc, bmean); z = np.polyfit(bc, bmean, 1)
    print(f"\n  Calibration Rho: {rc:+.4f} (p={pc:.6f}) | slope {z[0]*100:+.5f}%/unit | spread {(bmean[-1]-bmean[0])*100:+.5f}%")
    diag = ("EXCELLENT (>0.90)" if rc > .9 else "GOOD (>0.70)" if rc > .7 else "MODERATE (>0.50)" if rc > .5 else "WEAK (<0.50)")
    print(f"  Diagnosis: {diag}")
    tm, bm2 = pa >= .80, pa <= .20
    if tm.sum() > 100 and bm2.sum() > 100:
        u, pm = mannwhitneyu(ra[tm], ra[bm2], alternative='greater')
        print(f"  MWU Top20%>Bot20%: U={u:,.0f} p={pm:.2e} | top {ra[tm].mean()*100:+.5f}% vs bot {ra[bm2].mean()*100:+.5f}% (spread {(ra[tm].mean()-ra[bm2].mean())*100:+.5f}%)")
print(f"\n{SEP}")
