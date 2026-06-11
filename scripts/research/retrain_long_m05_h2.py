"""
DEFINITIVE TEST of the user's geometry hypothesis: retrain the LONG model at the
most favorable geometry found by the scan (m=0.5 ATR, 2h horizon, intraday) and
measure real model selection skill via purged walk-forward.

If selection skill ~= baseline (+1.2pp), longs are conclusively dead across the
geometry axis too. If it jumps (e.g. +6pp), the geometry idea has legs.

SAFE: reads existing feature views READ-ONLY (features only). Writes nothing that
the short model uses. New labels are computed in-memory.
"""
import os, sys, glob
import numpy as np
import pandas as pd
from datetime import timedelta
from catboost import CatBoostClassifier, Pool

sys.path.append(os.getcwd())
from scripts.labeling.tbm_label_engine import (
    load_15m, build_signal_bars, add_atr, SIGNAL_CONFIG)

CACHE = 'data/raw_upstox_cache_15min_3y'
VIEWS = 'data/tbm_feature_views'
COST = 0.0006
M, HORIZON_H = 0.5, 2
META = ['DateTime','Ticker','label','realized_gross','realized_net',
        'entry_price','atr','R','weight','YearMonth']

def wr(x): return float((x>0).mean()) if len(x) else np.nan

# ── 1. Relabel long at (M, HORIZON_H), intraday, all fields ───────────────────
def relabel():
    paths = sorted(glob.glob(os.path.join(CACHE,'*.csv')))
    rows=[]
    for i,p in enumerate(paths,1):
        tkr=os.path.basename(p).replace('.csv','')
        df15=load_15m(tkr)
        if df15 is None or df15.empty: continue
        df1h=build_signal_bars(df15)
        if df1h.empty: continue
        df1h=add_atr(df1h)
        by_day={d:g for d,g in df15.groupby(df15.index.date)}
        for _,r in df1h.iterrows():
            d=r['date']; sig=r['signal_time']; ep=r['entry_price']; atr=r['atr']
            if d not in by_day or np.isnan(atr) or atr<=0: continue
            day15=by_day[d]
            ehm=SIGNAL_CONFIG[sig][0]; eh,em=int(ehm[:2]),int(ehm[3:])
            ets=pd.Timestamp(d.year,d.month,d.day,eh,em)
            if ets not in day15.index: continue
            R=M*atr; TP=ep+R; SL=ep-R
            fwd=day15[day15.index>ets]
            if len(fwd)<HORIZON_H*4: continue
            fwd=fwd.iloc[:HORIZON_H*4]
            lab,gross=2,np.nan
            for _,b in fwd.iterrows():
                hi,lo=float(b['High']),float(b['Low'])
                if hi>=TP and lo<=SL: lab,gross=0,(SL-ep)/ep; break
                if hi>=TP: lab,gross=1,(TP-ep)/ep; break
                if lo<=SL: lab,gross=0,(SL-ep)/ep; break
            if lab==2: gross=(float(fwd.iloc[-1]['Close'])-ep)/ep
            h,mn=int(sig[:2]),int(sig[3:])
            rows.append((pd.Timestamp(d.year,d.month,d.day,h,mn),tkr,lab,gross,ep,atr,R))
        if i%40==0: print(f"  relabel [{i}/{len(paths)}]")
    df=pd.DataFrame(rows,columns=['DateTime','Ticker','label','realized_gross',
                                  'entry_price','atr','R'])
    df['realized_net']=df['realized_gross']-COST
    n=df.groupby('DateTime')['Ticker'].transform('count')
    df['weight']=1.0/n
    df['YearMonth']=df['DateTime'].dt.to_period('M').astype(str)
    return df

print(f"Relabeling LONG at m={M}, horizon={HORIZON_H}h ...")
lab=relabel()
vc=lab['label'].value_counts(normalize=True).sort_index()
print(f"  {len(lab):,} bars | SL {vc.get(0,0):.1%} TP {vc.get(1,0):.1%} TO {vc.get(2,0):.1%}")
print(f"  unconditional long net WR: {wr(lab['realized_net'].values):.2%}\n")

# ── 2. Load features (READ-ONLY) from all 4 views, merge ──────────────────────
print("Loading features (read-only) ...")
feat=None
for v in ['A_meanrev','B_trend','C_vol','D_momentum']:
    dfv=pd.read_parquet(os.path.join(VIEWS,f'{v}.parquet'))
    dfv['DateTime']=pd.to_datetime(dfv['DateTime'])
    cols=[c for c in dfv.columns if c not in META]
    sub=dfv[['DateTime','Ticker']+cols]
    feat = sub if feat is None else feat.merge(sub,on=['DateTime','Ticker'],how='inner')
feat_cols=[c for c in feat.columns if c not in ('DateTime','Ticker')]
print(f"  {len(feat):,} rows x {len(feat_cols)} features")

m=lab.merge(feat,on=['DateTime','Ticker'],how='inner')
print(f"  merged: {len(m):,} rows\n")

X=m[feat_cols].values.astype(np.float64)
y=m['label'].values.astype(np.int32)
w=m['weight'].values.astype(np.float64)
ym=m['YearMonth'].values
dtv=m['DateTime'].values
net=m['realized_net'].values

# ── 3. Purged walk-forward, single CatBoost, top-K by P_TP ────────────────────
months=sorted(np.unique(ym))
MINTR,VAL,TE,STEP=18,4,2,4
folds=[]
i=MINTR
while i < len(months)-VAL-TE:
    folds.append((months[:i],months[i:i+VAL],months[i+VAL:i+VAL+TE]))
    i+=STEP
print(f"Folds: {len(folds)}\n")

CB=dict(iterations=500,learning_rate=0.03,depth=5,loss_function='MultiClass',
        classes_count=3,random_seed=42,task_type='GPU',devices='0',verbose=False)

all_sel=[]; uncond_all=[]
for fi,(trm,vam,tem) in enumerate(folds,1):
    trm,vam,tem=set(trm),set(vam),set(tem)
    tr=np.array([x in trm for x in ym]); te=np.array([x in tem for x in ym])
    tstart=pd.Timestamp(dtv[te].min())-timedelta(days=1)
    tr=tr & ~(pd.to_datetime(dtv)+timedelta(hours=HORIZON_H) > tstart)
    if tr.sum()<100 or te.sum()<20: continue
    Xtr=X[tr].copy(); Xte=X[te].copy()
    mu=np.nanmean(Xtr,axis=0); mu=np.where(np.isfinite(mu),mu,0.0)
    for ci in range(X.shape[1]):
        Xtr[~np.isfinite(Xtr[:,ci]),ci]=mu[ci]; Xte[~np.isfinite(Xte[:,ci]),ci]=mu[ci]
    cb=CatBoostClassifier(**CB); cb.fit(Pool(Xtr,label=y[tr],weight=w[tr]))
    proba=cb.predict_proba(Xte)
    p_tp=proba[:,1]
    dfte=m[te].copy(); dfte['p_tp']=p_tp
    topk=dfte.sort_values('p_tp',ascending=False).groupby('DateTime').head(3)
    sel=wr(topk['realized_net'].values); unc=wr(net[te])
    all_sel.append(topk['realized_net'].values); uncond_all.append(net[te])
    print(f"  fold {fi}: sel WR={sel:.1%}  uncond={unc:.1%}  skill={(sel-unc)*100:+.2f}pp  (n={len(topk)})")

sel_all=np.concatenate(all_sel); unc_pool=np.concatenate(uncond_all)
print("\n"+"="*60)
print(f"POOLED  m={M} H={HORIZON_H}h")
print(f"  selected long WR : {wr(sel_all):.2%}")
print(f"  unconditional WR : {wr(unc_pool):.2%}")
print(f"  SELECTION SKILL  : {(wr(sel_all)-wr(unc_pool))*100:+.2f} pp")
print(f"  (baseline 1h/1ATR skill was +1.22pp; >+6pp would mean geometry helps)")
print("="*60)
