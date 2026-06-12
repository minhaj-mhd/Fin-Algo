# 🔬 Dual-TF Entry/Exit Overlay Research (2026-06-11)

> [!warning] Verdict authority
> All results below are **exploratory** (scripts in `scripts/analysis/`) and hold **no verdict authority** per [[Validation Gauntlet Architecture|the Gauntlet protocol]]. No Gauntlet runs were consumed. Metrics here are ⚠️ UNVERIFIED in the registry sense; they are walk-forward OOS research numbers with t-stats, reproducible from the saved panel.

**Question:** can the 15m conviction ranker ([[Multi-Timeframe Models|v3_15min_clean]], FILTER_GRADE, Gauntlet run `20260610T113721Z-5f7d069f`) dictate **entry, blocking, or exit** for 1h top-3 host signals (v10_native_1h)?

**Answer: NO — full-hold is optimal; every overlay is sub-cost or inverted.** But the research produced a reusable trade panel, four clean negative results, one retraction, and one genuinely new structural finding (EOD-concentrated mean reversion, the closest any price/volume signal has come to the cost line).

## 🧰 Setup (shared by all studies)

- 6-fold purged walk-forward, OOS 2024-09 → 2026-06, models retrained per fold; cost 10 bps round-trip; ~11.7k–13k trade candidates both directions.
- Real cross-sectional percentile ranks (full-universe re-score each 15m bar) — **not** raw margin scores (see the single-ticker degeneracy finding in [[../../06. Context & Logs/Daily Logs/2026-06-11|today's Daily Log]]).
- Alignment: 1h bar (left-labeled, dt1) closes at dt1+60 = entry; `Next_Hour_Return` covers dt1+60..120; pre-entry 15m bars at dt1+{0,15,30,45}; in-hold sub-periods at dt1+{60,75,90,105}. No look-ahead anywhere.
- **Reusable asset:** `data/research/entry_exit/dualtf_trade_panel.csv` — 13,020 trades × full 15m context. Any new entry/exit rule tests in **seconds** (no retraining). Builder: `scripts/analysis/build_dualtf_panel.py`. Folder README documents schema + guardrails.

## 📉 Study 1 — Entry gating by pre-entry rank trajectory (`wf_rank_trajectory.py`)

Hypothesis: rising 15m rank slope over the 4 pre-entry candles → take the 1h shot; falling → block.

- LONG: THRIVING 43.3% WR / +4.2 gross vs DIMINISHING 40.4% / +2.2 → separation **+2.0 bps, p=0.22 ns**. SHORT: **+2.4 bps, p=0.28 ns**. THRIVING buckets still net-negative (−5.8 / −8.3).
- Orthogonal to price momentum (corr +0.08 / +0.00); the price-momentum control leaned the **wrong way** (mean reversion).
- **⚠️ RETRACTED as a feature (same day):** Probe B below crossed slope with pre-entry level — best long cell was *falling*+high (opposite ordering), 0/6 folds positive for rising+high. Two cuts of the same data disagree on sign ⇒ the ~2 bps slope effect is **noise-level, not a stable feature**.

## ✂️ Study 2 — Early exit on conviction decay (`wf_early_exit.py`)

Headline "cuts losers −40.3→−23.0 bps" is real but **paid for entirely by winner-clipping**:

| conv<0.5 exit | clips winners | saves losers | net vs full-hold |
|---|---|---|---|
| LONG | −24.6 | +17.3 | +0.5 |
| SHORT | −26.1 | +19.0 | −1.1 |

Clip:save ratio ≈ 1.4:1 ⇒ the signal is mildly **anti-selective**. It beats a dumb price-stop (which is the *worst* rule: −9.3/−9.9 — underwater trades recover), but cannot beat holding.

## 🛡️ Study 3 — Asymmetric "let winners run" (`exit_rule_sweep.py`)

Exit only when conviction weak **and** underwater (± lock-once-green). Winner-clip collapses (−24.6 → −2.9) but the loser-save collapses in lockstep (+17.3 → +2.0): every variant converges to full-hold (−8.0/−8.2). **Full-hold is the optimum of the entire rule family built from {15m conviction, current P&L}.** (Tight `conv<0.2` long at −7.3 is a counting artifact of the 60/40 loser-heavy book, not skill.)

## 🔄 Study 4 — Conviction-momentum × price-direction → remaining return (`exit_momentum_buckets.py`)

The correct selectivity test (forward remaining return +90..+120 per bucket) — and the only significant result, but **INVERTED**:

| bucket (long) | remaining |
|---|---|
| FAV+/CONV+ ("hold aggressively") | **−0.5 (worst)** |
| FAV−/CONV− ("exit candidate") | **+3.0 (best, t=+2.33)** |

Selectivity p=0.046 — "looks weak" bounces, "looks strong" exhausts. Decomposition: ~80% of the effect is the **price axis** (FAV− +2.35 vs FAV+ −0.92); conviction-momentum adds ~1 bp, inconsistent sign. Same mean-reversion fingerprint as the Study-1 control and the price-stop failure — mechanistic, since the 15m model's top features (IBS, Buy_Pressure) are mean-reversion detectors.

## 🌟 Study 5 — Fresh probes on the panel (`panel_edge_probes.py`)

- **Depth-conditioned reversion: FLAT.** Deepest dips (−74 bps) bounce +3.5, shallowest (−3 bps) +2.3 — a fixed ~2–3 bps refund, not depth-proportional. "Late dip entry" nets −6.5/−8.2 @10 bps. Dead.
- **Joint slope×level entry gate: no positive cell** (12 cells, best −0.3 ns); produced the Study-1 retraction.
- **⭐ TIME-OF-DAY (the new finding): reversion concentrates at EOD.** Remaining return for FAV− trades by 1h-entry hour: ramps from ~0 midday to **hour 13 = +6.7 bps LONG (p=0.0037) and +8.9 bps SHORT (p=0.0066)** — the remaining window ≈14:45–15:15, the EOD book-squaring zone. Coherent on both sides + monotonic ramp + matches the known EOD/IBS mechanism ⇒ likely real structure, **but 8.9 < 10 bps binding cost**. Closest any price/volume signal has come to the line; only plausible flip levers are execution-cost reduction (limit entries) or one new feature. A pre-registered Gauntlet hypothesis exists here but would pre-register as a FAIL at 10 bps — hold until the cost question is answered.

## 🧭 Synthesis

1. The residual price/volume edge on this universe has a precise shape: **small (~3 bps), depth-insensitive, EOD-concentrated (~7–9 bps gross at hour 13) mean reversion** on ranker-selected names. Everything else (conviction level/slope/momentum, P&L state, dip depth) is noise or inverted.
2. **Momentum/persistence framings are confirmed dead** intraday on this universe — four independent inversions. Any future intraday edge is mean-reversion-shaped.
3. Consistent with [[Meta-Veto Rectification Plan MV2|the closed meta-veto line]]: signal recombination cannot cross the cost hurdle; **new information** (options OI/PCR/IV, order-flow/depth) or lower friction is required.

## 📁 Artifacts

- Panel + candidates + per-study result logs: `data/research/entry_exit/` (README has schema + guardrails: always run dumb controls; separation ≠ profit — decompose clip vs save; sign shorts on both branches).
- Scripts: `scripts/analysis/{build_dualtf_panel, wf_rank_trajectory, wf_early_exit, exit_rule_sweep, exit_momentum_buckets, panel_edge_probes}.py`
- Two sign/logic bugs were caught and fixed mid-study (exit_rule_sweep `require_red` branch; exit_momentum_buckets short-side held-branch sgn) — both noted in the result files.
