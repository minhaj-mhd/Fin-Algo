"""
FULLY pre-registered walk-forward: retrained model (already OOS) + GATES tuned only on
PAST realized OOS data, then applied blind to the next quarter.

At each graded quarter q:
  tuning window = all OOS trade-candidates with date < q_start
  - short gate (a,b,lunch) chosen to MAXIMISE total short net-bps on the tuning window
    (min 30 tuning trades; else 'disable shorts' if every config is net-negative)
  - long gate (c,d) chosen likewise on the tuning window
  apply the winning configs BLIND to quarter q.
Burn-in: first 2 quarters (2024 H1) are tuning-only, not graded.
Gates use the NIFTY500-1h proxy so the full multi-year range is covered.
Compares FIXED-spec gates vs FULLY-WF gates on the identical graded window.
"""
import numpy as np, pandas as pd

COST = 6.0
oos = pd.read_parquet('scratch/wf/oos_scored.parquet')
oos['DateTime'] = pd.to_datetime(oos['DateTime']); oos['date'] = oos['DateTime'].dt.date
oos['ss_m'] = oos.groupby('DateTime')['ss'].transform('mean'); oos['ls_m'] = oos.groupby('DateTime')['ls'].transform('mean')
oos['short_conv'] = (oos['ss']-oos['ss_m'])-(oos['ls']-oos['ls_m']); oos['long_conv'] = (oos['ls']-oos['ls_m'])-(oos['ss']-oos['ss_m'])
tmin, tmax = pd.to_datetime('10:15').time(), pd.to_datetime('14:15').time()
oos = oos[(oos['DateTime'].dt.time>=tmin)&(oos['DateTime'].dt.time<=tmax)].copy()

h = pd.read_csv('data/raw_index_cache/nifty500_1h.csv'); h['ts']=pd.to_datetime(h['timestamp']); h=h.sort_values('ts')
h['n2h']=h['close']/h['close'].shift(2)-1; h['d']=h['ts'].dt.date
h=h.merge(h.groupby('d')['open'].first().rename('dop'),on='d',how='left'); h['nin']=h['close']/h['dop']-1
a_=pd.DataFrame({'ts':sorted(oos['DateTime'].unique())}).sort_values('ts')
m=pd.merge_asof(a_,h[['ts','n2h','nin']],on='ts',direction='backward',tolerance=pd.Timedelta('90min'))
n2h_map=dict(zip(m['ts'],m['n2h'])); nin_map=dict(zip(m['ts'],m['nin']))
oos['n2h']=oos['DateTime'].map(n2h_map); oos['nin']=oos['DateTime'].map(nin_map)
oos=oos.dropna(subset=['n2h','nin'])

# per-scan top-1 candidates (structural part, gate-independent)
def scan_candidates(df):
    S, L = [], []
    for ts,g in df.groupby('DateTime'):
        n2h=g['n2h'].iloc[0]; nin=g['nin'].iloc[0]; t=ts.time(); dte=g['date'].iloc[0]
        e=g[g['ss']>g['ss_thr'].iloc[0]]
        if len(e):
            r=e.sort_values('short_conv',ascending=False)['Next_Hour_Return'].iloc[0]
            S.append({'ts':ts,'date':dte,'time':t,'n2h':n2h,'nin':nin,'net_bps':-r*10000-COST})
        c=g.sort_values('long_conv',ascending=False).iloc[0]
        L.append({'ts':ts,'date':dte,'time':t,'n2h':n2h,'nin':nin,'net_bps':c['Next_Hour_Return']*10000-COST})
    return pd.DataFrame(S), pd.DataFrame(L)
SC, LC = scan_candidates(oos)
SC['q']=pd.PeriodIndex(SC['ts'],freq='Q'); LC['q']=pd.PeriodIndex(LC['ts'],freq='Q')

lunches = {'none':(pd.to_datetime('23:00').time(),pd.to_datetime('23:01').time()),
           '1130-1300':(pd.to_datetime('11:30').time(),pd.to_datetime('13:00').time()),
           '1100-1330':(pd.to_datetime('11:00').time(),pd.to_datetime('13:30').time())}
SHORT_GRID=[(a,b,ln) for a in [-0.001,0.0,0.0025,0.005,99] for b in [0.002,0.0036,0.006,99] for ln in lunches]
LONG_GRID=[(c,d) for c in [-99,0.0,0.0025,0.005] for d in [-99,0.0,0.002,0.004]]

def short_mask(df,cfg):
    a,b,ln=cfg; ls,le=lunches[ln]
    tt=df['time']
    return ((df['n2h']<=a)|(df['nin']>b)) & ~((tt>=ls)&(tt<=le))
def long_mask(df,cfg):
    c,d=cfg; return (df['n2h']>c)&(df['nin']>d)

def best_cfg(df, grid, mask_fn, min_n=30):
    best=None; best_tot=-1e18
    for cfg in grid:
        sel=df[mask_fn(df,cfg)]
        if len(sel)<min_n: continue
        tot=sel['net_bps'].sum()
        if tot>best_tot: best_tot=tot; best=cfg
    # disable leg if the best achievable total is still negative
    if best is None or best_tot<=0: return None
    return best

quarters=sorted(SC['q'].unique())
graded=[q for q in quarters if q>=pd.Period('2024Q3','Q')]
print(f"Quarters {quarters[0]}..{quarters[-1]} | burn-in <2024Q3 | graded {graded[0]}..{graded[-1]} ({len(graded)}q)")

def day_t(dfr):
    dm=dfr.groupby('date')['net_bps'].mean()
    return dm.mean()/(dm.std(ddof=1)/np.sqrt(len(dm))) if len(dm)>1 else np.nan

# ---- FULLY-WF: choose gates on past, trade next quarter ----
wf_short, wf_long, cfg_log = [], [], []
for q in graded:
    past_s=SC[SC['q']<q]; past_l=LC[LC['q']<q]
    cs=best_cfg(past_s,SHORT_GRID,short_mask); cl=best_cfg(past_l,LONG_GRID,long_mask)
    cur_s=SC[SC['q']==q]; cur_l=LC[LC['q']==q]
    if cs is not None: wf_short.append(cur_s[short_mask(cur_s,cs)])
    if cl is not None: wf_long.append(cur_l[long_mask(cur_l,cl)])
    cfg_log.append((str(q),cs,cl))
wf_short=pd.concat(wf_short) if wf_short else pd.DataFrame(columns=SC.columns)
wf_long=pd.concat(wf_long) if wf_long else pd.DataFrame(columns=LC.columns)

# ---- FIXED-spec gates on the SAME graded window ----
sc_g=SC[SC['q'].isin(graded)]; lc_g=LC[LC['q'].isin(graded)]
fix_short=sc_g[((sc_g['n2h']<=0.0025)|(sc_g['nin']>0.0036)) & ~((sc_g['time']>=pd.to_datetime('11:30').time())&(sc_g['time']<=pd.to_datetime('13:00').time()))]
fix_long=lc_g[(lc_g['n2h']>0.0025)&(lc_g['nin']>0.0020)]

def leg_line(df,name):
    if len(df)==0: return f"  {name:26s} | DISABLED / 0 trades"
    return f"  {name:26s} | n {len(df):4d} | net {df['net_bps'].mean():+6.2f}bps | t_day {day_t(df):+5.2f} | WR {(df['net_bps']>0).mean():.1%}"

def combined(short_df,long_df,title):
    print(f"\n{'='*78}\n{title}\n{'='*78}")
    print(leg_line(short_df,'SHORT'))
    print(leg_line(long_df,'LONG'))
    tr=pd.concat([short_df.assign(so=0),long_df.assign(so=1)]).sort_values(['ts','so']).reset_index(drop=True)
    if len(tr)==0: print("  no trades"); return
    keep,active=[],pd.Timestamp('2000-01-01')
    for _,r in tr.iterrows():
        if r['ts']>=active: keep.append(r); active=r['ts']+pd.Timedelta(hours=1)
    ex=pd.DataFrame(keep); cap=100000.0
    for b in ex['net_bps']: cap+=cap*5*(b/10000)
    flat=ex['net_bps'].sum()*50
    print(f"  COMBINED (queue): n {len(ex)} | flat {ex['net_bps'].mean():+.2f}bps/tr | flat P&L Rs {flat:,.0f} (+{flat/1000:.0f}% on 1L 5x) | compounded Rs {cap:,.0f} ({cap/1e5-1:+.0%})")
    ex['year']=ex['ts'].dt.year
    for yr in sorted(ex['year'].unique()):
        ey=ex[ex['year']==yr]; print(f"    {yr}: n {len(ey):3d} | net {ey['net_bps'].mean():+6.2f}bps | flat Rs {ey['net_bps'].sum()*50:,.0f}")

combined(fix_short,fix_long,"FIXED-spec gates (hindsight constants) — graded window 2024Q3-2026Q2")
combined(wf_short,wf_long,"FULLY WALK-FORWARD gates (tuned on past only) — same window")

print("\nChosen configs per quarter (short=(n2h<=a|nin>b,lunch), long=(n2h>c&nin>d)):")
for q,cs,cl in cfg_log: print(f"  {q}: short={cs}  long={cl}")
