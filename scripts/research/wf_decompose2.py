"""
Final crux tests on the retrained-WF OOS panel:
 (1) SHORT ss>thr FLOOR ALONE (no nifty gate, no lunch veto) -> is the model's extreme
     short tail net-positive by itself, per year? (the only possibly-real model edge)
 (2) LONG mean_beta per year -> confirm it's bull-market beta, not alpha.
 (3) Gate fragility: perturb the lunch-veto window & the nifty threshold; show P&L swing.
"""
import numpy as np, pandas as pd, datetime as dt

COST = 6.0
oos = pd.read_parquet('scratch/wf/oos_scored.parquet')
oos['DateTime'] = pd.to_datetime(oos['DateTime']); oos['date'] = oos['DateTime'].dt.date
oos['year'] = oos['DateTime'].dt.year
oos['ss_m'] = oos.groupby('DateTime')['ss'].transform('mean')
oos['ls_m'] = oos.groupby('DateTime')['ls'].transform('mean')
oos['short_conv'] = (oos['ss']-oos['ss_m']) - (oos['ls']-oos['ls_m'])
oos['long_conv'] = (oos['ls']-oos['ls_m']) - (oos['ss']-oos['ss_m'])
tmin, tmax = pd.to_datetime('10:15').time(), pd.to_datetime('14:15').time()
oos = oos[(oos['DateTime'].dt.time>=tmin)&(oos['DateTime'].dt.time<=tmax)].copy()

def gates_1h(anchors):
    h = pd.read_csv('data/raw_index_cache/nifty500_1h.csv'); h['ts']=pd.to_datetime(h['timestamp']); h=h.sort_values('ts')
    h['n2h']=h['close']/h['close'].shift(2)-1; h['d']=h['ts'].dt.date
    h=h.merge(h.groupby('d')['open'].first().rename('dop'),on='d',how='left'); h['nin']=h['close']/h['dop']-1
    a=pd.DataFrame({'ts':sorted(anchors)}).sort_values('ts')
    m=pd.merge_asof(a,h[['ts','n2h','nin']],on='ts',direction='backward',tolerance=pd.Timedelta('90min'))
    return dict(zip(m['ts'],m['n2h'])), dict(zip(m['ts'],m['nin']))
n2h_map, nin_map = gates_1h(oos['DateTime'].unique())
oos['n2h']=oos['DateTime'].map(n2h_map); oos['nin']=oos['DateTime'].map(nin_map)

def day_t(d):
    dm=d.groupby('date')['net_bps'].mean()
    return dm.mean()/(dm.std(ddof=1)/np.sqrt(len(dm))) if len(dm)>1 else np.nan

# (1) SHORT ss-floor ALONE: top-1 by short_conv among ss>thr, every scan, NO gate/veto
print("="*80)
print("(1) SHORT ss>thr FLOOR ALONE (no nifty gate, no lunch veto) — model tail only")
print("="*80)
rows=[]
for ts,g in oos.groupby('DateTime'):
    e=g[g['ss']>g['ss_thr'].iloc[0]]
    if len(e)==0: continue
    r=e.sort_values('short_conv',ascending=False)['Next_Hour_Return'].iloc[0]
    rows.append({'date':g['date'].iloc[0],'year':ts.year,'net_bps':-r*10000-COST})
d=pd.DataFrame(rows)
print(f"  ALL 2024-2026 | n {len(d):4d} | net {d['net_bps'].mean():+6.2f}bps | t_day {day_t(d):+5.2f} | WR {(d['net_bps']>0).mean():.1%}")
for yr in sorted(d['year'].unique()):
    dy=d[d['year']==yr]; print(f"    {yr} | n {len(dy):4d} | net {dy['net_bps'].mean():+6.2f}bps | t_day {day_t(dy):+5.2f}")

# same but random among ss>thr (does conviction matter for the floor-only book?)
means=[]
for s in range(25):
    rng=np.random.default_rng(s); rr=[]
    for ts,g in oos.groupby('DateTime'):
        e=g[g['ss']>g['ss_thr'].iloc[0]]
        if len(e)==0: continue
        r=e['Next_Hour_Return'].values[rng.integers(len(e))]; rr.append(-r*10000-COST)
    means.append(np.mean(rr))
print(f"  (random among ss>thr, no gate): {np.mean(means):+6.2f}bps  [conviction adds {d['net_bps'].mean()-np.mean(means):+.2f}]")

# (2) LONG mean_beta per year (pure gate/beta)
print("\n"+"="*80)
print("(2) LONG gate mean-beta per year (random/mean stock on long-gate moments)")
print("="*80)
rows=[]
for ts,g in oos.groupby('DateTime'):
    n2h=g['n2h'].iloc[0]; nin=g['nin'].iloc[0]
    if pd.isna(n2h) or pd.isna(nin): continue
    if n2h>0.0025 and nin>0.0020:
        rows.append({'date':g['date'].iloc[0],'year':ts.year,'net_bps':(g['Next_Hour_Return'].mean())*10000-COST})
d=pd.DataFrame(rows)
print(f"  ALL | n {len(d):4d} | net {d['net_bps'].mean():+6.2f}bps | t_day {day_t(d):+5.2f}")
for yr in sorted(d['year'].unique()):
    dy=d[d['year']==yr]; print(f"    {yr} | n {len(dy):4d} | net {dy['net_bps'].mean():+6.2f}bps | t_day {day_t(dy):+5.2f}")

# (3) Gate fragility — perturb lunch window & nifty threshold (short conv book)
print("\n"+"="*80)
print("(3) SHORT gate fragility (avg net bps, full 2024-2026)")
print("="*80)
def short_book(lunch_start, lunch_end, n2h_cut, nin_cut):
    rows=[]
    for ts,g in oos.groupby('DateTime'):
        n2h=g['n2h'].iloc[0]; nin=g['nin'].iloc[0]; t=ts.time()
        if pd.isna(n2h) or pd.isna(nin): continue
        if not (n2h<=n2h_cut or nin>nin_cut): continue
        if lunch_start<=t<=lunch_end: continue
        e=g[g['ss']>g['ss_thr'].iloc[0]]
        if len(e)==0: continue
        r=e.sort_values('short_conv',ascending=False)['Next_Hour_Return'].iloc[0]
        rows.append({'date':g['date'].iloc[0],'net_bps':-r*10000-COST})
    d=pd.DataFrame(rows); return d['net_bps'].mean(), day_t(d), len(d)
base=pd.to_datetime
for name,(ls,le) in [("lunch 11:30-13:00",("11:30","13:00")),("lunch 11:00-13:30",("11:00","13:30")),
                     ("lunch 12:00-12:30",("12:00","12:30")),("lunch OFF",("23:00","23:01"))]:
    b,t,n=short_book(base(ls).time(),base(le).time(),0.0025,0.0036)
    print(f"  {name:22s} | net {b:+6.2f}bps | t_day {t:+5.2f} | n {n}")
for name,(n2c,nic) in [("nifty 0.0025/0.0036 (base)",(0.0025,0.0036)),("nifty 0.0015/0.0050",(0.0015,0.0050)),
                       ("nifty 0.0035/0.0025",(0.0035,0.0025)),("nifty OFF (any regime)",(9.9,9.9))]:
    b,t,n=short_book(base("11:30").time(),base("13:00").time(),n2c,nic)
    print(f"  {name:26s} | net {b:+6.2f}bps | t_day {t:+5.2f} | n {n}")
