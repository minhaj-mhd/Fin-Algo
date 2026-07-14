"""
Is anything left after removing market beta? Take the FULLY-walk-forward book (gates tuned
on past only) and re-measure every trade as SELECTION ALPHA vs the contemporaneous
cross-section mean forward return:
    long alpha  = (ret_pick - scan_mean_ret)
    short alpha = (scan_mean_ret - ret_pick)
If the surviving edge is beta, alpha collapses to ~0 (or below cost).
Reports RAW vs BETA-NEUTRAL for the fully-WF book, plus a market-neutral paired (L-S) book.
"""
import numpy as np, pandas as pd
COST = 6.0
oos = pd.read_parquet('scratch/wf/oos_scored.parquet')
oos['DateTime'] = pd.to_datetime(oos['DateTime']); oos['date'] = oos['DateTime'].dt.date
oos['ss_m'] = oos.groupby('DateTime')['ss'].transform('mean'); oos['ls_m'] = oos.groupby('DateTime')['ls'].transform('mean')
oos['short_conv'] = (oos['ss']-oos['ss_m'])-(oos['ls']-oos['ls_m']); oos['long_conv'] = (oos['ls']-oos['ls_m'])-(oos['ss']-oos['ss_m'])
oos['mkt_fwd'] = oos.groupby('DateTime')['Next_Hour_Return'].transform('mean')  # market 1h forward
tmin, tmax = pd.to_datetime('10:15').time(), pd.to_datetime('14:15').time()
oos = oos[(oos['DateTime'].dt.time>=tmin)&(oos['DateTime'].dt.time<=tmax)].copy()

h = pd.read_csv('data/raw_index_cache/nifty500_1h.csv'); h['ts']=pd.to_datetime(h['timestamp']); h=h.sort_values('ts')
h['n2h']=h['close']/h['close'].shift(2)-1; h['d']=h['ts'].dt.date
h=h.merge(h.groupby('d')['open'].first().rename('dop'),on='d',how='left'); h['nin']=h['close']/h['dop']-1
mm=pd.merge_asof(pd.DataFrame({'ts':sorted(oos['DateTime'].unique())}).sort_values('ts'),
                 h[['ts','n2h','nin']],on='ts',direction='backward',tolerance=pd.Timedelta('90min'))
oos['n2h']=oos['DateTime'].map(dict(zip(mm['ts'],mm['n2h']))); oos['nin']=oos['DateTime'].map(dict(zip(mm['ts'],mm['nin'])))
oos=oos.dropna(subset=['n2h','nin'])

S,L=[],[]
for ts,g in oos.groupby('DateTime'):
    n2h=g['n2h'].iloc[0]; nin=g['nin'].iloc[0]; t=ts.time(); dte=g['date'].iloc[0]; mk=g['mkt_fwd'].iloc[0]
    e=g[g['ss']>g['ss_thr'].iloc[0]]
    if len(e):
        p=e.sort_values('short_conv',ascending=False).iloc[0]; r=p['Next_Hour_Return']
        S.append({'ts':ts,'date':dte,'time':t,'n2h':n2h,'nin':nin,'net_bps':-r*10000-COST,'alpha_bps':(mk-r)*10000-COST})
    p=g.sort_values('long_conv',ascending=False).iloc[0]; r=p['Next_Hour_Return']
    L.append({'ts':ts,'date':dte,'time':t,'n2h':n2h,'nin':nin,'net_bps':r*10000-COST,'alpha_bps':(r-mk)*10000-COST})
SC=pd.DataFrame(S); LC=pd.DataFrame(L)
SC['q']=pd.PeriodIndex(SC['ts'],freq='Q'); LC['q']=pd.PeriodIndex(LC['ts'],freq='Q')

lunches={'none':('23:00','23:01'),'1130-1300':('11:30','13:00'),'1100-1330':('11:00','13:30')}
SHORT_GRID=[(a,b,ln) for a in [-0.001,0,0.0025,0.005,99] for b in [0.002,0.0036,0.006,99] for ln in lunches]
LONG_GRID=[(c,d) for c in [-99,0,0.0025,0.005] for d in [-99,0,0.002,0.004]]
def sm(df,cfg):
    a,b,ln=cfg; ls=pd.to_datetime(lunches[ln][0]).time(); le=pd.to_datetime(lunches[ln][1]).time()
    return ((df['n2h']<=a)|(df['nin']>b)) & ~((df['time']>=ls)&(df['time']<=le))
def lm(df,cfg): c,d=cfg; return (df['n2h']>c)&(df['nin']>d)
def best(df,grid,fn,min_n=30):
    bc,bt=None,-1e18
    for cfg in grid:
        s=df[fn(df,cfg)]
        if len(s)<min_n: continue
        if s['net_bps'].sum()>bt: bt=s['net_bps'].sum(); bc=cfg
    return bc if (bc is not None and bt>0) else None
def day_t(d,col):
    dm=d.groupby('date')[col].mean(); return dm.mean()/(dm.std(ddof=1)/np.sqrt(len(dm))) if len(dm)>1 else np.nan

graded=[q for q in sorted(SC['q'].unique()) if q>=pd.Period('2024Q3','Q')]
ws,wl=[],[]
for q in graded:
    cs=best(SC[SC['q']<q],SHORT_GRID,sm); cl=best(LC[LC['q']<q],LONG_GRID,lm)
    if cs is not None: ws.append(SC[SC['q']==q][sm(SC[SC['q']==q],cs)])
    if cl is not None: wl.append(LC[LC['q']==q][lm(LC[LC['q']==q],cl)])
ws=pd.concat(ws); wl=pd.concat(wl)

def line(df,name):
    return (f"  {name:12s} | n {len(df):4d} | RAW {df['net_bps'].mean():+6.2f}bps (t {day_t(df,'net_bps'):+.2f}) "
            f"| BETA-NEUTRAL(alpha) {df['alpha_bps'].mean():+6.2f}bps (t {day_t(df,'alpha_bps'):+.2f})")
print("="*96)
print("FULLY WALK-FORWARD book: RAW return vs BETA-NEUTRAL (selection alpha vs cross-section mean)")
print("="*96)
print(line(ws,'SHORT')); print(line(wl,'LONG'))
print("\n  Interpretation: RAW = directional (incl. index beta); BETA-NEUTRAL = pure stock-selection skill.")
# market drift over graded window (what beta paid)
mdays=oos.assign(q=pd.PeriodIndex(oos['DateTime'],freq='Q'))
mdays=mdays[mdays['q'].isin(graded)].groupby('date')['mkt_fwd'].mean()
print(f"  Market 1h-fwd drift over graded window: {mdays.mean()*10000:+.2f}bps/hr avg (the beta longs harvested).")

# --- neg-control: is the surviving short ALPHA the ss-floor (model) or just the weak regime? ---
print("\n  SHORT beta-neutral alpha neg-control (on the SAME fully-WF short-gate moments):")
short_ts=set(ws['ts'])
sub=oos[oos['DateTime'].isin(short_ts)].copy()
# conviction pick among floor (=ws, model), random among floor, random among ALL names
def alpha_book(mode,seeds=25):
    if mode=='conv': return ws['alpha_bps'].mean(), day_t(ws,'alpha_bps'), len(ws)
    vals=[]; tstats=[]
    for s in range(seeds):
        rng=np.random.default_rng(s); rows=[]
        for ts,g in sub.groupby('DateTime'):
            pool=g[g['ss']>g['ss_thr'].iloc[0]] if mode=='rand_floor' else g
            if len(pool)==0: continue
            p=pool.iloc[rng.integers(len(pool))]
            rows.append({'date':g['date'].iloc[0],'alpha_bps':(g['mkt_fwd'].iloc[0]-p['Next_Hour_Return'])*10000-COST})
        d=pd.DataFrame(rows); vals.append(d['alpha_bps'].mean()); tstats.append(day_t(d,'alpha_bps'))
    return float(np.mean(vals)), float(np.mean(tstats)), len(sub['DateTime'].unique())
for mode,nm in [('conv','conviction pick (model)'),('rand_floor','random above ss-floor (model filter, no rank)'),('rand_all','random ANY name (no model)')]:
    b,t,n=alpha_book(mode); print(f"    {nm:44s} | alpha {b:+6.2f}bps (t {t:+.2f})")
