---
title: "V20 Gates: Dev/Holdout Validation Framework & OOS Verdict"
type: research
status: concluded
updated: 2026-07-11
model: v20_rolling_1h
verdict: "⚠️ UNVERIFIED (no Gauntlet run) — GATE LAYER DEAD OOS. DEV-tuned gates fail a sealed 1-month holdout; the model itself generalises (rho~0.02), the gates overfit."
---

# V20 Gates: Dev/Holdout Validation Framework & OOS Verdict

## ⚠️ UNVERIFIED — exploratory framework, not a Gauntlet run
All numbers below come from `scripts/research/dev_holdout/` on the frozen model feed, not the
Validation Gauntlet. They hold no verdict authority for certification — but the holdout design
makes the **kill** (gates don't generalise) robust. This doc **supersedes the "CONFIRMED —
massive edge" verdict** in [[V20-15m-Regime-Gate-Sweep]], which was in-sample/hindsight-tuned.

## TL;DR
- The "+26 bps / 13× / 1-slot" backtest reproduces **to the rupee** — but reproducibility ≠ validity.
- **The model is a correctly held-out, stable input; the *gate layer* is what overfit.**
- Built a **develop-on-DEV / confirm-on-sealed-HOLDOUT** framework. Two pre-registered holdout
  looks, **both predicted, both FAILED**. No robust gate edge exists on this ρ≈0.02 signal.

## 1. The correction that reframes everything
Earlier this session it was wrongly asserted the 11-month window was *training* data. It is not.
`scripts/training/train_ranking_clean.py` does a **strict 80/20 temporal split**:

```
train = 2022-01 .. 2025-06   |   val = 2025-07   |   UNTOUCHED test = 2025-08 -> onward
```

The deployed `xgb_short_model.json` / `xgb_long_model.json` (the files the backtest loads) **never
saw Aug 2025 – Jun 2026**. So the "11-month OOS" label is correct *for the model*. And the model
**generalises** — rank-IC is stable across train → test → truly-unseen (barely any train/test gap):

| window | long ρ | short ρ |
|---|---|---|
| TRAIN 2024 (in-sample) | +0.0345 | +0.0324 |
| TRAIN 2025 H1 (in-sample) | +0.0286 | +0.0352 |
| TEST 11mo Aug25–Jun26 (held-out) | +0.0205 | +0.0195 |
| OOS Jun4–Jul10 (truly unseen) | +0.0187 | +0.0197 |

The model's ρ on the "collapse" window equals its ρ on the DEV window. **The model did not
degrade OOS.** What collapsed was the *strategy/gate layer* tuned to the DEV window's realisations.

## 2. The framework — `scripts/research/dev_holdout/`
- [build_feed.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/research/dev_holdout/build_feed.py) — scores the panel once with the frozen models → `data/research/dev_holdout/feed.parquet` (610,762 rows, 3,570 ts, 2025-08-01→2026-07-10). Offline + deterministic.
- [strategy.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/research/dev_holdout/strategy.py) — config-driven 1-slot gate engine (faithful generalisation of [temp_11m_combined.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/backtests/temp_11m_combined.py)); every threshold/toggle is a config key.
- [run.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/research/dev_holdout/run.py) — DEV runs free; HOLDOUT requires `--confirm --hypothesis`, appended to `HOLDOUT_LEDGER.md`, **single-use**.
- [ablation.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/research/dev_holdout/ablation.py) — leave-one-out / add-one / neg-control + per-month stability.

**Split:** DEV `2025-08-01 → 2026-06-11`, HOLDOUT `2026-06-11 → 2026-07-10` (~1 month; boundary is
the pre-registration knob `--holdout-start`). Both are model-OOS; the split is for *gate* development.
**Primary metric = net bps/trade + t-stat** (leverage-independent). The 5× compounding P&L is
reported but labelled leverage-dependent — the "13×" is that artifact, see [[5x_Compounding_Sequence_Risk]].

## 3. DEV ablation — where the "edge" actually comes from
| config | n | net bps | t | note |
|---|---|---|---|---|
| **structural min** (model + short 0.082 + longs, no gates) | 945 | **−2.02** | −0.9 | engine alone loses money |
| **FULL baseline** (all gates) | 218 | **+25.98** | +4.6 | all edge is gate-added |

Neg-control (random pick inside the same gated pool) on the baseline = **+21.48 bps** → the model's
**ranking adds only ~+4.5 bps**; the other +21.5 is gate/regime *carving* (the overfit-prone part).

**Leave-one-out (baseline − gate; ΔNet<0 ⇒ gate helps in-context):**
- `lunch_veto` **−20.5** (= "midday is noisy", a structural prior, not alpha)
- `long_nifty2h(>0.0025)` **−20.0** (index-timing / regime gate)
- `long_vwap_gate` −5.7 · `long_market_gate` −2.0 · `long_sp500_veto` −1.7 · `long_conv_cap` −1.4
- `short_dyn(0.110)` **+0.30** — the hindsight dynamic short threshold adds **negative** value.

**Add-one (structural + one gate; ΔNet>0 ⇒ helps standalone):** only `long_nifty2h` is meaningfully
positive (**+8.0**, 8/11 months) — but it is **negative in Feb/Apr/May** (decaying into the holdout).
`short_conv_cap(0.04)` **REJECTED** on DEV (−2.83, t−3.0, 1/11 months) — does **not** replicate the
live-shadow inverted-U (regime-specific), see [[Conviction Caps & Long-Side Filter (OOS 2026-07)]].

## 4. HOLDOUT confirmations — pre-registered, single-use, both FAILED
| hypothesis | DEV | HOLDOUT | predicted? |
|---|---|---|---|
| `baseline_213` (as shipped) | +25.98 (t+4.6, 10/11 mo) | **−39.34 (t−2.5)** | ✅ fail |
| `struct+nifty2h` (lone survivor) | +6.01 (t+1.2) | **−28.22 (t−1.8)**; long +6.09→**−3.74** | ✅ ≤0 |

Both logged in `HOLDOUT_LEDGER.md`. The holdout is now spent for these two hypotheses.

## 5. Lessons (durable)
1. **DEV stability is necessary, not sufficient.** The baseline was net-positive in **10/11 DEV
   months** and still failed the holdout — joint fitting fools per-month stability. Keep the
   holdout single-use; iterating against it = the original overfit, one level up.
2. **Rupees ≠ edge.** 5× geometric compounding turns a modest per-trade bps into a "13×" headline;
   it inverts to 0.5× OOS. Judge on net bps/trade + t.
3. **A ρ≈0.02 signal can't be turned into edge by thresholds.** Gates allocate/filter; they add no
   information. Remaining levers = **cost, sizing, or new data** — not more gates.
4. **Pre-register the direction before looking.** Both holdout calls were correct *predictions*, not
   post-hoc rationalisations — that is the difference between science and fishing.

## 6. Data-integrity issues found while building
- **panel.parquet was rebuilt mid-session** → now runs to 2026-07-10 with the Feb 20–Mar 24 gap
  **filled**. The audited doc's exact figures (213 / +26.59) were on the *old gapped* panel; the
  current panel gives 218 / +25.98. (Corrects the "fix in-place" note in the memory layer.)
- **`data/raw_index_cache/nifty50_15m.csv` is double-convention corrupted** (rows ≤ ~Jun 9 are IST
  wall-clock mislabelled `+0000`; rows ≥ ~Jun 10 are true UTC; Jun 8–9 hold both/duplicated).
  `build_feed.py` normalises both to naive IST — **the cache should be re-collected cleanly.**

## Links
- Supersedes: [[V20-15m-Regime-Gate-Sweep]] (its "CONFIRMED massive edge" verdict is in-sample).
- Related: [[v20_1Slot_Stress_Tests]] · [[V20 Rolling-1h Overlapping-Window Model]] · [[Conviction Caps & Long-Side Filter (OOS 2026-07)]] · [[5x_Compounding_Sequence_Risk]]
- Model split source: [train_ranking_clean.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/training/train_ranking_clean.py) (lines 229–247)
