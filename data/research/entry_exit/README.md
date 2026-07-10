# Dual-TF Entry/Exit Research (1h host × 15m conviction)

Research bundle for testing **entry, blocking, and exit** rules that combine the
1h ranker (host signal) with the 15m cross-sectional conviction ranker. All
results are **exploratory — no Gauntlet verdict authority** (per `AGENTS.md`).
Created 2026-06-11.

## TL;DR conclusion (so far)

The 15m conviction is a genuine cross-sectional **ranker** but adds **no deployable
post-cost edge** to 1h trade management:

- **Entry / blocking:** rank-trajectory (rising→take, falling→block) is *directionally
  correct* and *orthogonal to price momentum* (corr≈0) — a legit feature — but ~2 bps,
  not significant (p≈0.22/0.28), and the kept bucket stays net-negative.
- **Early exit:** conviction-exit *looks* like it cuts losses (losers −40→−23) but pays
  for the entire saving by clipping winners (−24.6). Net ≈ full-hold. **Full-hold is optimal.**
- **Asymmetric ("let winners run"):** protecting greens works mechanically (winner-clip
  −24.6→−2.9) but then stops cutting losers too → converges to full-hold. No variant beats it.
- **Conviction-momentum × price-direction (forward-return test):** the only *significant*
  effect (long p=0.046) but **INVERTED by mean reversion** — "looks strong" (FAV+/CONV+)
  has the worst forward return (−0.5), "looks weak" (FAV−/CONV−) the best (+3.0). Strength=
  exhaustion, weakness=oversold-bounce. Still <10 bps cost; not tradeable.
- **Fresh probes (`panel_edge_probes.py`):** (a) reversion is **depth-flat** (~2–3 bps
  regardless of dip size → "late dip entry" dead at −6.5/−8.2 net); (b) joint slope×level
  gate has **no positive cell** and flips the slope ordering → **entry-slope finding
  RETRACTED** (noise-level, not a stable feature); (c) ⭐ **reversion concentrates at EOD**:
  hour-13 FAV− remaining = +6.7 bps long (p=0.0037) / +8.9 bps short (p=0.0066), coherent
  ramp both sides (14:45–15:15 book-squaring window) — closest signal to the cost line in
  the price/volume family, but still < 10 bps binding cost.

**Root cause:** this universe mean-reverts at the 15m horizon, so every persistence/
momentum overlay fires the wrong way. The 15m features (IBS, Buy_Pressure) are mean-
reversion detectors → great cross-sectional ranking, useless as directional momentum.
**Next edge must come from NEW information** (options OI/PCR/IV, order-flow/depth), not
re-combinations of price + the price-based ranker.

## Primary asset

- **`dualtf_trade_panel.csv`** — 13,020 1h top-3 trades (WF OOS 2024-09→2026-06, both
  directions), each with full 15m context. **Use this to test any new rule in seconds —
  no model retraining.** Built by `scripts/analysis/build_dualtf_panel.py`.

  Columns: `fold, ym, dt1, ticker, dir, sL, sS, nhr` +
  `rkL_{0,15,30,45,60,75,90,105}`, `rkS_{...}` (cross-sectional pct ranks, pre-entry &
  in-hold), `sub_{60,75,90,105}` (in-hold 15m sub-period returns).
  Alignment: 1h bar closes at dt1+60 = ENTRY; `nhr` covers dt1+60..120; pre-entry bars
  (0–45) are strictly before entry → no look-ahead. Coverage ~90% full pre-entry & in-hold.

## Other candidate tables

- `rank_trajectory_candidates.csv` — entry study (slope, delta, entry rank, price-mom control, outcome).
- `early_exit_candidates.csv` — exit study (in-hold sub-returns n0–n3, in-hold ranks rk0–rk3).

## Scripts (in `scripts/analysis/`)

| Script | Question | Result file |
|---|---|---|
| `wf_rank_trajectory.py` | entry: does pre-entry rank slope gate 1h longs/shorts? | `results/rank_trajectory_2026-06-11.txt` |
| `wf_early_exit.py` | exit: does conviction decay during hold cut losses? | `results/early_exit_2026-06-11.txt` |
| `exit_rule_sweep.py` | exit: asymmetric "let winners run" rules (reads panel) | `results/exit_rule_sweep_2026-06-11.txt` |
| `exit_momentum_buckets.py` | exit: conviction-momentum × price-dir → remaining return | `results/exit_momentum_buckets_2026-06-11.txt` |
| `panel_edge_probes.py` | fresh probes: depth-reversion, slope×level gate, time-of-day | `results/panel_edge_probes_2026-06-11.txt` |

`build_dualtf_panel.py` regenerates the panel (retrains models per fold, ~minutes).
The other scripts that read the panel run in seconds.

## Guardrails (learned the hard way in this study)

1. **No look-ahead:** exit/entry decisions use only ranks + returns realized *before* the
   point being predicted. Forward return measured strictly after the decision bar.
2. **Always run a control:** price-momentum (entry) and dumb price-stop / RED-only (exit).
   If the 15m signal doesn't beat the dumb control, it adds nothing.
3. **Separation ≠ profit:** a rule that separates winners/losers in hindsight (e.g. "cuts
   losers −20 bps") is not tradeable unless the *kept* book is net-positive AND it doesn't
   clip winners by more than it saves. Always decompose winners-clip vs losers-save.
4. **Sign discipline:** for shorts, sign every return by `sgn=-1` on BOTH the held and the
   exited branch (a dropped sgn faked a +4.5 short result; fixed in `exit_momentum_buckets.py`).
5. **Costs:** 10 bps round-trip; early exit does not add cost (one round-trip either way).

## To certify anything here

A promising rule = a **pre-registered Gauntlet hypothesis** (one run, user-approved;
deflates future t-thresholds). Nothing in this bundle currently warrants a run — all
effects are sub-cost and/or inverted.
