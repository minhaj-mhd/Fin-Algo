- 2026-07-11T22:32:38 | config=baseline_213 | holdout 2026-06-11..2026-07-10 | n=28 net=-39.34bps t=-2.51 | HYPOTHESIS: baseline 213-gate config as shipped; expect OOS degradation vs DEV
- 2026-07-11T23:46:47 | config=struct_plus_nifty2h | holdout 2026-06-11..2026-07-10 | n=45 net=-28.22bps t=-1.80 | HYPOTHESIS: Only DEV-surviving gate (Nifty-2h long uptrend, structural+this). PREDICT FAIL OOS: net<=0 and long<=0, given Apr/May DEV decay (-13/-10).
- 2026-07-12T10:52:44 | config=dyn_prob_floor_short | holdout 2026-06-11..2026-07-10 | n=20 net=-18.50bps t=-0.84 | HYPOTHESIS: Dynamic Probability Floor (short-only): base=99.92pct ss (0.0788 DEV-est), +0.028 tighten when SP500prev>+0.5% AND Nifty2h>=-0.10%. DEV +9.45bps t=1.05 but ranking-edge NEG (-0.80, pool-carving) => PREDICT FAIL OOS net<=0.
- 2026-07-12T10:52:58 | config=static_floor_short | holdout 2026-06-11..2026-07-10 | n=23 net=-18.93bps t=-0.96 | HYPOTHESIS: STATIC control for Dynamic Probability Floor: same base=99.92pct ss (0.0788) but NO macro tightening (short_dyn_thresh=null). Isolates OOS marginal value of the +0.028 penalty = H1(dyn) - H2(static). PREDICT FAIL OOS net<=0.

## 2026-07-13 — 3-tier examination of user-requested risk gates (DEV / PROXY-Jun / TRUE-Jul)
NOTE: informal multi-look via three_tier.py (proxy=Jun 1-30, true=Jul 1-10). Result NEGATIVE across the board — no spurious winner mined; but the Jun/Jul windows are now SPENT for the short-conviction-cap family (a truly-fresh confirm needs Aug+ data).
- baseline_213      DEV +26.36(t4.48,n206) | PROXY -16.11(n28) | TRUE -34.85(n12)  -> FAIL at proxy
- struct_plus_nifty2h DEV +5.30(t1.01,n348) | PROXY -20.58(n46) | TRUE +1.56(n16)   -> FAIL at proxy (consistent w/ prior sealed -28.22)
- sc_shortonly_nocap  DEV +39.84(t4.07,n106)| PROXY -12.74(n15) | TRUE -17.43(n4)   -> FAIL at proxy
- short_conv_cap sweep (0.125..0.20 on this feed's demeaned-score scale; 0.04 is off-scale, deletes all shorts): NO cap gives positive proxy w/ meaningful n; tighter caps overfit DEV & yield 0 true-OOS trades. Inverted-U cap does NOT port to the 1-slot engine.
VERDICT: every testable gate overfits DEV and dies at the proxy gate. Confirms the framework thesis — a rho~0.02 model can't be turned into edge by thresholds.
