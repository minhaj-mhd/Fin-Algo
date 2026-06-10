# 💬 Conversation Context: v8 Full Walk-Forward Reanalysis

## 📌 Metadata
- **Conversation ID**: c--Users-loq-Desktop-Trading-finalgo
- **Start Date**: 2026-06-10
- **Status**: 🔴 Concluded
- **Focus Area**: Model Suite (1-Hour Core Ranker Audit)

## 🎯 Objectives
- [x] Perform a full empirical reanalysis of `v8_upstox_3y` (the active 1H production ranker), not just reading static docs.
- [x] Run a genuine purged walk-forward retraining of v8's exact architecture.
- [x] Check raw vs net win rates, cost-sign correctness, and time-of-day alpha pocket claims.

## 💻 Active Code Files Modified
- [v8_walkforward.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/analysis/v8_walkforward.py) — new reproducibility script.
- Output: `data/model_analysis/v8_walkforward/walkforward.json`, `walkforward_preds.npz`.

## 📝 Compacted Session Log
- **Initial Analysis**: Read `models/v8_upstox_3y/metadata.json` and `feature_importance_report.md`. Confirmed v8 was evaluated on a single 80/20 chronological split (5168/1293 queries) where the test set doubled as the early-stopping eval set in `train_ranking_upstox.py` — a known leakage pattern flagged generically by the TBM postmortem ("re-audit v8/v10/15m").
- **Harness build**: Wrote `v8_walkforward.py`, mirroring `v10_v18_walkforward.py`'s rolling-fold structure but with v8's exact hyperparams (depth=5, rank:pairwise, 86 features) and a dedicated validation month per fold (no test-set leakage into early stopping). Used `ranking_data_upstox_1h_v3_3y.csv` (all 86 v8 features present, data through 2026-06).
- **Run**: 9 folds, 2023-08 → 2026-05, 320,931 OOS rows, ~85s total on CUDA.
- **Findings**:
  - Fold Spearman decays monotonically: long 0.044→0.012, short 0.037→0.004 (most recent fold near zero). Average ≈0.026/0.024, about half the static 0.046/0.049.
  - Top-1/Top-3 Long & Short all net-negative @6bps (-3.1 to -4.3bps, t=-2.0 to -4.2).
  - Cost-sign explicitly asserted correct — ruled out a TBM-style bug.
  - Last 12mo (2025-07+): raw edge near zero/negative (Top-3 Long raw -0.05bps).
  - Time-of-day: all 5 session bars (09:15-13:15) net-negative; no surviving pocket. Note: this dataset's session ends at 13:15 (different bar alignment than `ranking_data_upstox_3y.csv`'s :30 bars), so the previously-cited "14:30 IST" pocket is unverified, not contradicted.
- **Conclusion**: v8 demoted from "Active Production Champion" to KILLED, same class as TBM/v18/v19. Updated `Model Performance & Statistics.md` (matrix row + new detailed section) and `Current Context.md`.
- **Follow-up sanity check (user-requested)**: User suspected the rolling WF harness itself. Built `v8_static_70_10_20.py`: 38mo train / 5mo val / 11mo test (2025-08..2026-06, ~same window as original's 1,293-query test set), val≠test early stopping. Result: Spearman +0.0212/+0.0192 (matches WF, not 0.046/0.049); `best_iteration`=37/13 vs 500-round budget — proves the original's higher number came from selecting best_iteration against the test set. Top-K @6bps net-negative except Top-1 Short (+3.22bps, t=1.25, not significant, flips negative @10bps). **Confirms: not a harness bug.**
- **Pivot (user-requested)**: "Ignore net returns — which models have *real* raw signal we could filter on?" Computed raw Spearman directly from existing `data/model_analysis/v10_v18_independent/walkforward_preds.npz` (same 9-fold genuine WF) for v10-depth4. Result: ρ=+0.0273/+0.0251 avg (paired t-test vs 0: t=8.67/6.77, p<0.0001), decaying to +0.0129/+0.0049 in fold 9 — essentially identical to v8's +0.026/+0.024. Raw Top-3 win-rate z-tests vs 50%: v8 short z=5.4, v10 short z=7.3 (both p<0.0001, persists into last 12mo for v10), longs weaker/decayed. **Conclusion**: v8/v10 share one real-but-weak, decaying short-side raw signal — not tradeable standalone post-cost, but statistically real enough to use as a filter/feature. TBM/v18/v19 have zero such signal. `v2_15min_3y`/`v1_30min` still need this raw-signal check. Documented in new "Raw-Signal-Only Audit" section of `Model Performance & Statistics.md`.

## 🔗 Core Memory Links & Backlinks
- Linked Core Specs: [[02. Model Suite/Model Performance & Statistics]]
- Linked Core Specs: [[02. Model Suite/Training Data & Regime Requirements]]
- Related: [[06. Context & Logs/Daily Logs/2026-06-09|Daily Log 2026-06-09]] (TBM correction, origin of the re-audit mandate)
