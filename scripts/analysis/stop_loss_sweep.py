"""
STOP-LOSS RESEARCH (exploratory, no verdict authority) — does cutting the loss tail make the
1h ranker book profitable? Reads the prebuilt dualtf_trade_panel (13k WF-OOS top-3 trades, both
sides) with the in-hold 15m path (sub_60,75,90,105) — no model retraining.

Two questions the user raised:
  Q1  PREMISE: is it true that "most trades make money but a few HUGE losses sink it"?  -> diagnose
      the full-hold per-trade net distribution (median, WR, tail mass, how much of the negative sum
      lives in the worst 5/10%).
  Q2  FIX: does a WIDE fixed-% stop (cut only the genuinely big losers, not every red) lift mean net
      above 0? Sweep stop levels; this is the gap prior research left (it only tested the 0% "RED-only"
      stop, the worst case under mean-reversion).

Stop sim: walk the in-hold cumulative SIGNED return at 15m checkpoints; if it breaches -level at
checkpoint k, exit there (fill at -level = proper stop-order semantics; FAVORABLE to the stop, so a
failure here is robust). Else hold to horizon. Cost = one round-trip (10bps) either way. Decompose
winner-clip vs loser-save (a stop that "saves losers" but clips winners by as much is a wash). Dumb
control: same stop on RANDOM picks, to see if any lift is real interaction or a distributional artifact.

Run: python scripts/analysis/stop_loss_sweep.py
"""
import numpy as np, pandas as pd
PANEL = 'data/research/entry_exit/dualtf_trade_panel.csv'
COST = 10 / 1e4
HOLD = [60, 75, 90, 105]                 # in-hold 15m sub-period checkpoints
LEVELS = [0.003, 0.005, 0.0075, 0.01, 0.015, 0.02]   # stop distances (0.3% .. 2%)
rng = np.random.default_rng(42)

P = pd.read_csv(PANEL)
P = P.dropna(subset=[f'sub_{m}' for m in HOLD]).reset_index(drop=True)


def cum_signed(N, sgn):
    """[T,4] sub-returns -> [T,4] cumulative SIGNED return at each checkpoint."""
    cp = np.cumprod(1 + N, axis=1) - 1.0
    return cp * sgn


def full_hold_net(N, sgn):
    return (np.prod(1 + N, axis=1) - 1.0) * sgn - COST


def stop_net(C, full, level):
    """C [T,4] cumulative signed return; full [T] full-hold signed gross (no cost).
    Exit at first checkpoint k<4 where C[:,k] <= -level (fill at -level); else full. Minus cost."""
    out = full.copy()
    stopped = np.zeros(len(C), bool)
    for k in range(3):                    # checkpoints 0,1,2 (60,75,90); k=3 (105) IS the horizon
        hit = (~stopped) & (C[:, k] <= -level)
        out[hit] = -level
        stopped |= hit
    return out - COST, stopped


def diagnose(net, sgn_label):
    pos = net[net > 0]; neg = net[net < 0]
    tot = net.sum()
    q = np.percentile(net, [1, 5, 10, 25, 50, 75, 90]) * 1e4
    worst10_mass = net[net <= np.percentile(net, 10)].sum()
    print(f"  [{sgn_label}] N={len(net):,}  mean={net.mean()*1e4:+.2f}bps  median={np.median(net)*1e4:+.2f}bps  "
          f"WR(net>0)={(net>0).mean()*100:.1f}%")
    print(f"     pctiles bps  p1={q[0]:+.0f} p5={q[1]:+.0f} p10={q[2]:+.0f} p25={q[3]:+.0f} "
          f"p50={q[4]:+.0f} p75={q[5]:+.0f} p90={q[6]:+.0f}")
    print(f"     pos-sum {pos.sum()*1e4:+.0f}bps over {len(pos)}  |  neg-sum {neg.sum()*1e4:+.0f}bps over {len(neg)}  "
          f"|  worst-10% contributes {worst10_mass/tot*100 if tot!=0 else float('nan'):.0f}% of net total")


def main():
    print("=" * 100)
    print("Q1  PREMISE CHECK — full-hold per-trade net distribution (10bps cost)")
    print("=" * 100)
    data = {}
    for d, sgn in [('long', 1), ('short', -1)]:
        s = P[P['dir'] == d].reset_index(drop=True)
        N = s[[f'sub_{m}' for m in HOLD]].values.astype(float)
        full = (np.prod(1 + N, axis=1) - 1.0) * sgn
        net = full - COST
        C = cum_signed(N, sgn)
        data[d] = (N, sgn, full, C, net)
        diagnose(net, d)

    print("\n" + "=" * 100)
    print("Q2  FIXED-% STOP SWEEP — mean net vs full-hold (fill at -level, favorable to the stop)")
    print("    winners/losers = bps changed vs full-hold; %stop = fraction stopped; ctrl = same stop on RANDOM picks")
    print("=" * 100)
    for d in ('long', 'short'):
        N, sgn, full, C, net_fh = data[d]
        win = net_fh >= 0; los = net_fh < 0
        print(f"\n===== {d.upper()} (N={len(N):,}) =====   full_hold mean={net_fh.mean()*1e4:+.2f}bps")
        # random control: shuffle the path<->none association by using random signs of equal stop freq
        for lv in LEVELS:
            net, stopped = stop_net(C, full, lv)
            clip = (net[win].mean() - net_fh[win].mean()) * 1e4
            save = (net[los].mean() - net_fh[los].mean()) * 1e4
            # dumb control: apply the SAME stop level to a randomly permuted path (breaks pick<->path link)
            perm = rng.permutation(len(C))
            net_c, _ = stop_net(C[perm], full[perm], lv)
            print(f"  stop@-{lv*100:>4.2f}%  mean={net.mean()*1e4:>+6.2f}bps  WR={(net>0).mean()*100:>5.1f}%  "
                  f"%stop={stopped.mean()*100:>5.1f}  | winners {clip:>+6.1f} losers {save:>+6.1f}  "
                  f"| ctrl(rand) mean={net_c.mean()*1e4:>+6.2f}")
        print(f"  {'(reference)':<12} full_hold mean={net_fh.mean()*1e4:+.2f}bps")

    print("\n" + "=" * 100)
    print("READ: a stop helps only if mean(stop) > mean(full_hold) AND losers-save > winners-clip.")
    print("Under mean-reversion, stops realize dips that would have bounced -> winners-clip ~ losers-save.")
    print("=" * 100)


if __name__ == '__main__':
    main()
