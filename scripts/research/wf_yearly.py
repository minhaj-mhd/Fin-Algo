"""Per-year + compounded returns for the retrained-WF gated strategy (5x, start 1L)."""
import numpy as np, pandas as pd, datetime as dt
COST = 6.0
oos = pd.read_parquet('scratch/wf/oos_scored.parquet')
oos['DateTime'] = pd.to_datetime(oos['DateTime']); oos['date'] = oos['DateTime'].dt.date
oos['ss_m'] = oos.groupby('DateTime')['ss'].transform('mean'); oos['ls_m'] = oos.groupby('DateTime')['ls'].transform('mean')
oos['short_conv'] = (oos['ss']-oos['ss_m'])-(oos['ls']-oos['ls_m']); oos['long_conv'] = (oos['ls']-oos['ls_m'])-(oos['ss']-oos['ss_m'])
tmin, tmax = pd.to_datetime('10:15').time(), pd.to_datetime('14:15').time()
oos = oos[(oos['DateTime'].dt.time>=tmin)&(oos['DateTime'].dt.time<=tmax)].copy()

def g15():
    n=pd.read_csv('data/raw_index_cache/nifty50_15m.csv');n['ts']=pd.to_datetime(n['ts']);n=n.sort_values('ts')
    n['n2h']=n['close']/n['close'].shift(8)-1;n['d']=n['ts'].dt.date
    n=n.merge(n.groupby('d')['open'].first().rename('dop'),on='d',how='left');n['nin']=n['close']/n['dop']-1
    return dict(zip(n['ts'],n['n2h'])),dict(zip(n['ts'],n['nin']))
def g1h(anch):
    h=pd.read_csv('data/raw_index_cache/nifty500_1h.csv');h['ts']=pd.to_datetime(h['timestamp']);h=h.sort_values('ts')
    h['n2h']=h['close']/h['close'].shift(2)-1;h['d']=h['ts'].dt.date
    h=h.merge(h.groupby('d')['open'].first().rename('dop'),on='d',how='left');h['nin']=h['close']/h['dop']-1
    a=pd.DataFrame({'ts':sorted(anch)}).sort_values('ts')
    m=pd.merge_asof(a,h[['ts','n2h','nin']],on='ts',direction='backward',tolerance=pd.Timedelta('90min'))
    return dict(zip(m['ts'],m['n2h'])),dict(zip(m['ts'],m['nin']))

def build(df,n2h_map,nin_map,start=None):
    d=df.copy()
    if start: d=d[d['date']>=start]
    d['n2h']=d['DateTime'].map(n2h_map);d['nin']=d['DateTime'].map(nin_map);d=d.dropna(subset=['n2h','nin'])
    tr=[]
    for ts,g in d.groupby('DateTime'):
        n2h=g['n2h'].iloc[0];nin=g['nin'].iloc[0];t=ts.time();thr=g['ss_thr'].iloc[0]
        if (n2h<=0.0025 or nin>0.0036) and (t<pd.to_datetime('11:30').time() or t>pd.to_datetime('13:00').time()):
            c=g[g['ss']>thr].nlargest(1,'short_conv')
            if len(c): tr.append({'ts':ts,'so':0,'net_bps':-c['Next_Hour_Return'].iloc[0]*10000-COST})
        if n2h>0.0025 and nin>0.0020:
            c=g.nlargest(1,'long_conv')
            if len(c): tr.append({'ts':ts,'so':1,'net_bps':c['Next_Hour_Return'].iloc[0]*10000-COST})
    td=pd.DataFrame(tr).sort_values(['ts','so']).reset_index(drop=True)
    keep,active=[],pd.Timestamp('2000-01-01')
    for _,r in td.iterrows():
        if r['ts']>=active: keep.append(r);active=r['ts']+pd.Timedelta(hours=1)
    ex=pd.DataFrame(keep).reset_index(drop=True)
    cap=100000.0;caps=[]
    for b in ex['net_bps']:
        cap+=cap*5*(b/10000);caps.append(cap)
    ex['cap']=caps;ex['year']=ex['ts'].dt.year
    return ex

def report(ex,title):
    print(f"\n{'='*74}\n{title}\n{'='*74}")
    print(f"{'year':6s} | {'trades':>6s} | {'avg bps':>7s} | {'FLAT P&L (5x)':>14s} | {'compounded cap':>15s} | {'yr %':>7s}")
    prev=100000.0
    for yr in sorted(ex['year'].unique()):
        ey=ex[ex['year']==yr]
        flat=ey['net_bps'].sum()*50  # 5x on fixed 1L
        endcap=ey['cap'].iloc[-1]
        yrret=endcap/prev-1
        print(f"{yr:6d} | {len(ey):6d} | {ey['net_bps'].mean():+7.2f} | {flat:14,.0f} | {endcap:15,.0f} | {yrret:+7.1%}")
        prev=endcap
    tot=ex['cap'].iloc[-1]
    print(f"{'-'*74}")
    print(f"  FLAT (order-independent):  Rs {ex['net_bps'].sum()*50:,.0f}   (+{ex['net_bps'].sum()*50/100000:.0%} on 1L, 5x)")
    print(f"  COMPOUNDED final capital:  Rs {tot:,.0f}   (ROI {tot/100000-1:+.1%})")

report(build(oos,*g15(),start=dt.date(2025,8,1)), "EXACT NIFTY50-15m gates, 2025-08 -> 2026-06 (your claim window)")
report(build(oos,*g1h(oos['DateTime'].unique())), "APPROX NIFTY500-1h gates, full 2024-01 -> 2026-06")
