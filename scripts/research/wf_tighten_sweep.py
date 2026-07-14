"""
Reproduce the "tighten the gates -> fewer trades -> higher %" lever on the CLEAN retrained
OOS panel, and show what it actually buys: headline compounded % vs the real beta-neutral alpha.
Sweeps the short ss floor and long gate, plus the exact S&P-500 dynamic-tightening rule.
"""
import numpy as np, pandas as pd, datetime as dt
COST = 6.0
oos = pd.read_parquet('scratch/wf/oos_scored.parquet')
oos['DateTime'] = pd.to_datetime(oos['DateTime']); oos['date'] = oos['DateTime'].dt.date
oos['ss_m'] = oos.groupby('DateTime')['ss'].transform('mean'); oos['ls_m'] = oos.groupby('DateTime')['ls'].transform('mean')
oos['short_conv'] = (oos['ss']-oos['ss_m'])-(oos['ls']-oos['ls_m']); oos['long_conv'] = (oos['ls']-oos['ls_m'])-(oos['ss']-oos['ss_m'])
oos['mkt_fwd'] = oos.groupby('DateTime')['Next_Hour_Return'].transform('mean')
tmin, tmax = pd.to_datetime('10:15').time(), pd.to_datetime('14:15').time()
oos = oos[(oos['DateTime'].dt.time>=tmin)&(oos['DateTime'].dt.time<=tmax)].copy()
oos = oos[oos['date'] >= dt.date(2025,8,1)]   # headline window (exact NIFTY50-15m gates exist here)

n = pd.read_csv('data/raw_index_cache/nifty50_15m.csv'); n['ts']=pd.to_datetime(n['ts']); n=n.sort_values('ts')
n['n2h']=n['close']/n['close'].shift(8)-1; n['d']=n['ts'].dt.date
n=n.merge(n.groupby('d')['open'].first().rename('dop'),on='d',how='left'); n['nin']=n['close']/n['dop']-1
oos['n2h']=oos['DateTime'].map(dict(zip(n['ts'],n['n2h']))); oos['nin']=oos['DateTime'].map(dict(zip(n['ts'],n['nin'])))
oos=oos.dropna(subset=['n2h','nin'])
sp = pd.read_parquet('data/raw_global_daily/SP500.parquet').sort_values('timestamp')
sp['sp_prev'] = sp['close'].pct_change().shift(1); sp['d']=sp['timestamp'].dt.date
oos['sp_prev'] = oos['date'].map(dict(zip(sp['d'], sp['sp_prev']))).fillna(0.0)

def day_t(d,col='net_bps'):
    dm=d.groupby('date')[col].mean(); return dm.mean()/(dm.std(ddof=1)/np.sqrt(len(dm))) if len(dm)>1 else np.nan

def run(short_thr_fn, long_gate, label):
    S,L=[],[]
    la,lb=pd.to_datetime('11:30').time(),pd.to_datetime('13:00').time()
    for ts,g in oos.groupby('DateTime'):
        n2h=g['n2h'].iloc[0]; nin=g['nin'].iloc[0]; t=ts.time(); mk=g['mkt_fwd'].iloc[0]
        thr=short_thr_fn(g['sp_prev'].iloc[0])
        if (n2h<=0.0025 or nin>0.0036) and (t<la or t>lb):
            e=g[g['ss']>thr]
            if len(e):
                p=e.sort_values('short_conv',ascending=False).iloc[0]; r=p['Next_Hour_Return']
                S.append({'ts':ts,'date':g['date'].iloc[0],'so':0,'net_bps':-r*10000-COST,'alpha_bps':(mk-r)*10000-COST})
        if n2h>long_gate and nin>0.0020:
            p=g.sort_values('long_conv',ascending=False).iloc[0]; r=p['Next_Hour_Return']
            L.append({'ts':ts,'date':g['date'].iloc[0],'so':1,'net_bps':r*10000-COST,'alpha_bps':(r-mk)*10000-COST})
    S=pd.DataFrame(S); L=pd.DataFrame(L)
    tr=pd.concat([S,L]).sort_values(['ts','so']).reset_index(drop=True)
    keep,active=[],pd.Timestamp('2000-01-01')
    for _,r in tr.iterrows():
        if r['ts']>=active: keep.append(r); active=r['ts']+pd.Timedelta(hours=1)
    ex=pd.DataFrame(keep); cap=100000.0
    for b in ex['net_bps']: cap+=cap*5*(b/10000)
    sa = S['alpha_bps'].mean() if len(S) else np.nan
    st = day_t(S,'alpha_bps') if len(S) else np.nan
    la_ = L['alpha_bps'].mean() if len(L) else np.nan
    print(f"{label:40s} | S {len(S):3d}/{ (S['net_bps'].mean() if len(S) else 0):+5.1f} L {len(L):3d}/{(L['net_bps'].mean() if len(L) else 0):+5.1f} "
          f"| trades {len(ex):3d} | flat {ex['net_bps'].sum()*50/1000:+5.0f}% | COMPND {cap/1e5-1:+7.0%} "
          f"| SHORT alpha {sa:+5.2f}(t{st:+.1f})")

print("Window 2025-08 -> 2026-06 (retrained OOS, exact NIFTY50-15m gates)")
print("="*140)
print("Legend: S n/avgbps  L n/avgbps | trades=post-queue | flat%=order-indep | COMPND=5x compounded | SHORT alpha=beta-neutral selection")
print("-"*140)
run(lambda sp: 0.082, 0.0025, "BASE (ss>0.082, long>0.25%)")
run(lambda sp: 0.090, 0.0025, "tighter short (ss>0.090)")
run(lambda sp: 0.110, 0.0025, "tighter short (ss>0.110)")
run(lambda sp: 0.082, 0.0040, "tighter long (long>0.40%)")
run(lambda sp: 0.090, 0.0040, "tighter both (ss>0.090 + long>0.40%)")
run(lambda sp: 0.110 if sp>0.005 else 0.082, 0.0025, "S&P-dynamic (ss->0.110 if SP prev>+0.5%)")
run(lambda sp: (0.110 if sp>0.005 else 0.090), 0.0040, "S&P-dynamic + tighter both (the '1600%' recipe)")
