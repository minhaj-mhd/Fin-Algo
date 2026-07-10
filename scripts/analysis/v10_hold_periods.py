"""
Backtest the 1-hour v10 model with variable hold periods (2 hours and 3 hours).
Capped at 15:15 EOD exit.

We use the pre-calculated `Next_Hour_Return` from the dataset.
Since the dataset has bars for 09:15, 10:15, 11:15, 12:15, 13:15:
- R1(t) = Next_Hour_Return at t
- R2(t) = (1 + R1(t)) * (1 + R1(t+1h)) - 1, capped at end of day
- R3(t) = (1 + R1(t)) * (1 + R1(t+1h)) * (1 + R1(t+2h)) - 1, capped at end of day
"""
import os, sys, json, warnings, gc
import numpy as np, pandas as pd, xgboost as xgb
from scipy.stats import rankdata
warnings.filterwarnings('ignore')
sys.path.append(os.getcwd())

DIR, DATA, RET = 'models/v10_native_1h', 'data/ranking_data_upstox_1h_v3_3y.csv', 'Next_Hour_Return'
H_TEST, MIN_TRAIN, COST = 4, 18, 10/1e4
RNG = np.random.default_rng(0)

def load(p, ret):
    df = pd.concat([c for c in pd.read_csv(p, chunksize=200_000)], ignore_index=True)
    df['dt'] = pd.to_datetime(df['DateTime'])
    df['ym'] = df['DateTime'].str[:7]
    df['hour'] = df['dt'].dt.strftime('%H:%M')
    df['Date'] = df['dt'].dt.date
    df = df.dropna(subset=[ret]).reset_index(drop=True)
    
    # Sort to compute multi-hour returns
    df = df.sort_values(['Ticker', 'dt']).reset_index(drop=True)
    
    print("Computing multi-hour returns...")
    # Shift returns within the same Ticker and Date
    df['R1'] = df[RET]
    df['R2_next'] = df.groupby(['Ticker', 'Date'])['R1'].shift(-1).fillna(0)
    df['R3_next'] = df.groupby(['Ticker', 'Date'])['R1'].shift(-2).fillna(0)
    
    df['Ret_2h'] = (1 + df['R1']) * (1 + df['R2_next']) - 1
    df['Ret_3h'] = (1 + df['R1']) * (1 + df['R2_next']) * (1 + df['R3_next']) - 1
    
    return df

def Xmat(df, fe):
    X = df[fe].values.astype(float)
    for ci in range(X.shape[1]):
        c = X[:, ci]; b = np.isnan(c) | np.isinf(c)
        if b.any(): X[b, ci] = float(np.nanmean(c[~b])) if (~b).any() else 0.0
    return X

def stat(net, fold):
    n = len(net)
    if n < 20: return None
    t = net.mean()/(net.std()/np.sqrt(n)) if net.std() > 0 else 0
    bs = [net[RNG.integers(0,n,n)].mean()*1e4 for _ in range(1500)]
    fm = [net[fold==fi].mean() for fi in np.unique(fold)]
    sig = '***' if abs(t)>2.58 else ('**' if abs(t)>1.96 else ('*' if abs(t)>1.64 else ''))
    return dict(n=n, wr=(net>0).mean()*100, bps=net.mean()*1e4, ci=(np.percentile(bs,2.5),np.percentile(bs,97.5)), t=t, sig=sig, pos=sum(1 for x in fm if x>0), nf=len(fm))

print("Loading 1h v10 model feature spec...")
with open(f'{DIR}/metadata.json') as f: meta = json.load(f)
fe = meta['features']

print("Loading and preparing data...")
df = load(DATA, RET)
# We can load the saved models to evaluate, or do a walk-forward.
# The user wants "a backtest". If we do the walk-forward pooled OOS it gives the most realistic representation.
# We will do walk forward just like wf_1h_base.py
print("Prepping X...")
X = Xmat(df, fe)

params = meta['params']
params['tree_method'] = 'hist'
params['device'] = 'cuda'
def iranks(y, q, inv=False):
    out = np.zeros_like(y, dtype=int)
    for qid in np.unique(q):
        m = q == qid; v = -y[m] if inv else y[m]; out[m] = rankdata(v, method='ordinal')-1
    return out

def fitdm(X, y, q, inv):
    d = xgb.DMatrix(X, label=iranks(y, q, inv)); d.set_group(pd.Series(q).groupby(q).size().values); return d

months = sorted(df['ym'].unique())
folds = []; i = MIN_TRAIN + 1
while i + 1 <= len(months):
    folds.append((months[:i-1], months[i-1], months[i:i+H_TEST])); i += H_TEST
print(f"Total {len(folds)} folds OOS {folds[0][2][0]} -> {folds[-1][2][-1]}")

rows = []
for fi, (tr_m, val_m, te_m) in enumerate(folds, 1):
    tr = df['ym'].isin(tr_m).values; va = df['ym'].isin([val_m]).values; te = df['ym'].isin(te_m).values
    bl = xgb.train(params, fitdm(X[tr], df[RET].values[tr], df['Query_ID'].values[tr], False), 500,
                   evals=[(fitdm(X[va], df[RET].values[va], df['Query_ID'].values[va], False),'v')], early_stopping_rounds=50, verbose_eval=False)
    bs = xgb.train(params, fitdm(X[tr], df[RET].values[tr], df['Query_ID'].values[tr], True), 500,
                   evals=[(fitdm(X[va], df[RET].values[va], df['Query_ID'].values[va], True),'v')], early_stopping_rounds=50, verbose_eval=False)
    sub = df[te].copy(); sub['sL'] = bl.predict(xgb.DMatrix(X[te])); sub['sS'] = bs.predict(xgb.DMatrix(X[te]))
    sub['posL'] = sub.groupby('dt')['sL'].rank(ascending=False, method='first')
    sub['posS'] = sub.groupby('dt')['sS'].rank(ascending=False, method='first')
    sub['fold'] = fi
    rows.append(sub[['fold','ym','hour','R1','Ret_2h','Ret_3h','posL','posS']])
    print(f"  fold {fi}/{len(folds)} {te_m[0]}->{te_m[-1]} done")
    del sub; gc.collect()

P = pd.concat(rows, ignore_index=True)
print(f"\nCompleted Walk-Forward. Pooled OOS rows: {len(P):,}")

print("="*100)
print("  v10 1-Hour Backtest — Walk-Forward Pooled OOS, 10 bps Cost")
print("  Comparing 1-Hour vs 2-Hour vs 3-Hour Hold Periods (Capped at 15:15 EOD)")
print("="*100)

for d, sgn, poscol in [('LONG', 1, 'posL'), ('SHORT', -1, 'posS')]:
    print(f"\n>>> {d} SIDE (Top-3 Signals) <<<")
    print(f"  {'Hold':>5} {'N':>6} {'WR%':>6} {'NetBps':>8} {'95% CI':>16} {'t':>6} {'+folds':>7}")
    
    sub = P[P[poscol] <= 3]
    
    # 1 Hour
    net1 = sub['R1'].values*sgn - COST
    st1 = stat(net1, sub['fold'].values)
    if st1: print(f"  {'1h':>5} {st1['n']:>6} {st1['wr']:>5.1f}% {st1['bps']:>+7.1f} [{st1['ci'][0]:>+5.1f},{st1['ci'][1]:>+5.1f}] {st1['t']:>5.1f}{st1['sig']:<3} {st1['pos']:>2}/{st1['nf']:<2}")
    
    # 2 Hours
    net2 = sub['Ret_2h'].values*sgn - COST
    st2 = stat(net2, sub['fold'].values)
    if st2: print(f"  {'2h':>5} {st2['n']:>6} {st2['wr']:>5.1f}% {st2['bps']:>+7.1f} [{st2['ci'][0]:>+5.1f},{st2['ci'][1]:>+5.1f}] {st2['t']:>5.1f}{st2['sig']:<3} {st2['pos']:>2}/{st2['nf']:<2}")
    
    # 3 Hours
    net3 = sub['Ret_3h'].values*sgn - COST
    st3 = stat(net3, sub['fold'].values)
    if st3: print(f"  {'3h':>5} {st3['n']:>6} {st3['wr']:>5.1f}% {st3['bps']:>+7.1f} [{st3['ci'][0]:>+5.1f},{st3['ci'][1]:>+5.1f}] {st3['t']:>5.1f}{st3['sig']:<3} {st3['pos']:>2}/{st3['nf']:<2}")

print("\n" + "="*100 + "\n")
