"""
HONEST intrabar stop-loss sweep on the v21 ranker's actual Top-K trades.

Reads data/research/stop_research/trade_path_panel.parquet (per-leg 15-min CLOSE/LOW/HIGH path out
to 3h). Unlike the close-checkpoint sim, a fixed-% stop here triggers on the bar's intrabar LOW
(long) / HIGH (short) — the price a stop order would actually hit — and fills at exactly -level
(OPTIMISTIC: real fills slip worse on exactly the gap trades that form the fat tail, so a failure
here is robust). Cost = one round-trip (10bps) whether stopped or held.

For each horizon (1h/2h/3h) and side (long/short) we report, vs full-hold:
  - fixed-% stop sweep (the user's idea: cut only the big losers)
  - a TRAILING stop variant
  - winner-clip vs loser-save decomposition (a stop that saves losers but clips winners as much = wash)
  - RANDOM-BASKET control: identical stop on K random names. If the random book gains as much, the
    "profit" is mechanical tail-truncation of a negative-mean distribution (deleveraging), NOT alpha.
  - bootstrap CI on the best stop's mean net (is it even distinguishable from 0 / from full-hold?)

RESEARCH ONLY (AGENTS.md). Run: python scripts/analysis/stop_intrabar_sweep.py
"""
import numpy as np, pandas as pd
PANEL = 'data/research/stop_research/trade_path_panel.parquet'
COST = 10 / 1e4
LEVELS = [0.003, 0.005, 0.0075, 0.01, 0.015, 0.02]
TRAILS = [0.005, 0.0075, 0.01]
HOR = {'1h': 4, '2h': 8, '3h': 12}
rng = np.random.default_rng(42)

P = pd.read_parquet(PANEL)


def cols(prefix, kH):
    return [f'{prefix}_{k}' for k in range(1, kH + 1)]


def fixed_stop_net(sub, sgn, kH, level):
    """sub: DataFrame of legs. Returns (net[N], stopped[N]). Intrabar trigger, fill at -level."""
    rl = sub[cols('rl', kH)].values; rh = sub[cols('rh', kH)].values
    rc_close = sub[f'rc_{kH}'].values
    adverse = rl if sgn == 1 else rh
    trig = (adverse <= -level) if sgn == 1 else (adverse >= level)
    stopped = trig.any(axis=1)
    realized = np.where(stopped, -level, sgn * rc_close)
    return realized - COST, stopped


def trailing_stop_net(sub, sgn, kH, trail):
    """Trail from the running favorable peak by `trail`. Long: peak=run-max of high ret; exit if a
    later low falls trail below peak (fill at peak-trail). Short: mirror with lows/highs swapped."""
    rl = sub[cols('rl', kH)].values; rh = sub[cols('rh', kH)].values; rc_close = sub[f'rc_{kH}'].values
    N = len(sub); net = np.empty(N); stopped = np.zeros(N, bool)
    if sgn == 1:
        peak = np.maximum.accumulate(rh, axis=1)            # best favorable (high) so far
        trig = rl <= (peak - trail)                          # low retraced trail from peak
        fill = peak - trail
    else:
        trough = np.minimum.accumulate(rl, axis=1)           # best favorable (low) for a short
        trig = rh >= (trough + trail)
        fill = -(trough + trail)                             # signed short return at fill
    for i in range(N):
        w = np.where(trig[i])[0]
        if len(w):
            stopped[i] = True
            net[i] = (fill[i, w[0]] if sgn == 1 else fill[i, w[0]]) - COST
        else:
            net[i] = sgn * rc_close[i] - COST
    return net, stopped


def decomp(net, fh):
    win = fh >= 0; los = fh < 0
    clip = (net[win].mean() - fh[win].mean()) * 1e4 if win.any() else 0.0
    save = (net[los].mean() - fh[los].mean()) * 1e4 if los.any() else 0.0
    return clip, save


def boot_ci(net, n=2000):
    m = np.array([net[rng.integers(0, len(net), len(net))].mean() for _ in range(n)]) * 1e4
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def run_side(hname, kH, label, legs, sgn):
    sub = legs.dropna(subset=cols('rc', kH) + cols('rl', kH) + cols('rh', kH)).reset_index(drop=True)
    if len(sub) < 100:
        print(f"  [{label}] too few legs ({len(sub)})"); return
    fh = sgn * sub[f'rc_{kH}'].values - COST
    print(f"\n  [{hname} {label}] N={len(sub):,}  full_hold mean={fh.mean()*1e4:+.2f}bps "
          f"med={np.median(fh)*1e4:+.2f} WR={(fh>0).mean()*100:.1f}%")
    best = (None, fh.mean())
    for lv in LEVELS:
        net, st = fixed_stop_net(sub, sgn, kH, lv)
        clip, save = decomp(net, fh)
        tag = ''
        if net.mean() > best[1]:
            best = (f'fixed@{lv*100:.2f}%', net.mean()); tag = ' <-best'
        print(f"     fixed@-{lv*100:>4.2f}%  mean={net.mean()*1e4:>+6.2f}  WR={(net>0).mean()*100:>5.1f}%  "
              f"%stop={st.mean()*100:>5.1f}  clip{clip:>+6.1f} save{save:>+6.1f}{tag}")
    for tr in TRAILS:
        net, st = trailing_stop_net(sub, sgn, kH, tr)
        clip, save = decomp(net, fh)
        if net.mean() > best[1]:
            best = (f'trail@{tr*100:.2f}%', net.mean())
        print(f"     trail@{tr*100:>4.2f}%  mean={net.mean()*1e4:>+6.2f}  WR={(net>0).mean()*100:>5.1f}%  "
              f"%stop={st.mean()*100:>5.1f}  clip{clip:>+6.1f} save{save:>+6.1f}")
    return best, fh


def main():
    print("=" * 104)
    print("HONEST INTRABAR STOP SWEEP (fill at -level = optimistic; real fills slip worse on the tail)")
    print("  clip/save = bps changed vs full-hold for winners/losers;  %stop = fraction stopped")
    print("=" * 104)
    for hname, kH in HOR.items():
        print(f"\n################  HORIZON {hname}  ################")
        for side, sgn, src in [('LONG model', 1, 'long'), ('SHORT model', -1, 'short'),
                               ('LONG random', 1, 'rand'), ('SHORT random', -1, 'rand')]:
            legs = P[P['side'] == src]
            res = run_side(hname, kH, side, legs, sgn)

    # focused head-to-head at 1h: best model stop vs same stop on random, with bootstrap CI
    print("\n" + "=" * 104)
    print("ALPHA-vs-DELEVERAGING CHECK @1h: does the model book + stop beat the RANDOM book + same stop?")
    print("=" * 104)
    kH = 4
    for side, sgn in [('LONG', 1), ('SHORT', -1)]:
        m = P[P['side'] == ('long' if sgn == 1 else 'short')].dropna(subset=cols('rc', kH)+cols('rl', kH)+cols('rh', kH)).reset_index(drop=True)
        r = P[P['side'] == 'rand'].dropna(subset=cols('rc', kH)+cols('rl', kH)+cols('rh', kH)).reset_index(drop=True)
        for lv in (0.005, 0.0075, 0.01):
            nm, _ = fixed_stop_net(m, sgn, kH, lv)
            nr, _ = fixed_stop_net(r, sgn, kH, lv)
            cm = boot_ci(nm);
            print(f"  {side} fixed@-{lv*100:.2f}%:  MODEL mean={nm.mean()*1e4:+.2f}bps [95% CI {cm[0]:+.2f},{cm[1]:+.2f}]  "
                  f"| RANDOM mean={nr.mean()*1e4:+.2f}bps  | edge(model-rand)={(nm.mean()-nr.mean())*1e4:+.2f}bps")
    print("\nREAD: stop is real alpha only if MODEL+stop > RANDOM+stop (else it's just tail-truncation/"
          "deleveraging of a negative-mean book) AND the model CI clears 0 at a realistic stop width.")


if __name__ == '__main__':
    main()
