"""
EXIT-RULE SWEEP (exploratory, no verdict authority) — reads the prebuilt panel,
no model retraining. Tests the "let winners run, cut losers" idea: make the
15m-conviction exit ASYMMETRIC so it stops clipping winners.

Consumes data/research/entry_exit/dualtf_trade_panel.csv.
In-hold path: sub_{60,75,90,105} (sub-period returns), own-side rank rk*_{60,75,90,105}.
Checkpoints after sub-period k=1,2,3 (no look-ahead: rank + return-so-far only);
exit realizes compounded return through k. Cost = one round-trip (10 bps) for all.

Rules:
  full_hold                  : hold the whole hour (baseline).
  conv<thr                   : exit if own-rank < thr (trigger-happy original).
  conv<thr & RED             : THE IDEA — exit only if conviction weak AND trade currently underwater.
  RED-only stop              : control — exit whenever underwater (no conviction).
  conv<thr & RED, else lock@G : exit if weak&red; once green, never exit (let runner run).
Reports mean_net, WR, and the winner/loser decomposition (clip vs save) so we see
whether asymmetry preserves the loss-saving while keeping winner profit.
"""
import numpy as np, pandas as pd
PANEL = 'data/research/entry_exit/dualtf_trade_panel.csv'
COST = 10/1e4
HOLD = [60, 75, 90, 105]

P = pd.read_csv(PANEL)
need = [f'sub_{m}' for m in HOLD]
P = P.dropna(subset=need).reset_index(drop=True)

def own_ranks(row):
    pre = 'rkL_' if row['dir'] == 'long' else 'rkS_'
    return [row[f'{pre}{m}'] for m in HOLD]

def sim(n, rk, sgn, thr, mode='conv'):
    """n,rk: length-4 in-hold sub-returns & own-side ranks. Return net (signed, -cost).
    mode: 'conv' exit if rank<thr | 'conv_red' exit if rank<thr AND underwater |
          'red' exit if underwater | 'conv_red_lock' conv_red but never exit once green."""
    ek = 4
    for k in (1, 2, 3):
        cum = (np.prod([1+x for x in n[:k]]) - 1) * sgn
        weak = rk[k] < thr
        red = cum < 0
        if mode == 'conv_red_lock' and cum > 0:      # seen green -> let runner run, stop checking
            break
        if mode == 'conv':                fire = weak
        elif mode in ('conv_red', 'conv_red_lock'): fire = weak and red
        elif mode == 'red':               fire = red
        else:                             fire = False
        if fire:
            ek = k; break
    return (np.prod([1+x for x in n[:ek]]) - 1) * sgn - COST

def report(label, net, fh):
    win = fh >= 0; los = fh < 0
    clip = (net[win].mean() - fh[win].mean()) * 1e4
    save = (net[los].mean() - fh[los].mean()) * 1e4
    print(f"  {label:<26} mean={net.mean()*1e4:>+6.1f}  WR={(net>0).mean()*100:>5.1f}%  "
          f"med={np.median(net)*1e4:>+5.1f} | winners {clip:>+6.1f}  losers {save:>+6.1f}")

for d, sgn in [('long', 1), ('short', -1)]:
    s = P[P['dir'] == d].reset_index(drop=True)
    N = np.array([[r[f'sub_{m}'] for m in HOLD] for _, r in s.iterrows()])
    RK = np.array([own_ranks(r) for _, r in s.iterrows()])
    fh = (np.prod(1+N, axis=1) - 1) * sgn - COST
    print(f"\n===== {d.upper()} (N={len(s)}) =====   [winners/losers cols = bps changed vs full-hold]")
    report('full_hold', fh, fh)
    for thr in (0.5, 0.35, 0.2):
        net = np.array([sim(N[i], RK[i], sgn, thr, 'conv') for i in range(len(s))])
        report(f'conv<{thr}', net, fh)
    print("  -- asymmetric: exit only when conviction weak AND currently underwater --")
    for thr in (0.5, 0.35, 0.2):
        net = np.array([sim(N[i], RK[i], sgn, thr, 'conv_red') for i in range(len(s))])
        report(f'conv<{thr} & RED', net, fh)
    net = np.array([sim(N[i], RK[i], sgn, 1.0, 'red') for i in range(len(s))])
    report('RED-only stop (ctrl)', net, fh)
    print("  -- let-winners-run: lock once green, else exit weak&red --")
    for thr in (0.5, 0.35):
        net = np.array([sim(N[i], RK[i], sgn, thr, 'conv_red_lock') for i in range(len(s))])
        report(f'conv<{thr}&RED,lock@green', net, fh)

print("\n" + "="*92)
print("  GOAL: a rule with mean > full_hold. Asymmetry should shrink the 'winners' clip")
print("  toward 0 while keeping the 'losers' save positive.")
print("="*92)
