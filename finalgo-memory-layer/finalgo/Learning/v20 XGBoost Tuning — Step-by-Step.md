---
title: "v20 Base Ranker — Optuna Hyperparameter Tuning (learning walkthrough)"
type: reference
status: 🟢 Active
updated: 2026-06-17
tags: [learning, automl, optuna, xgboost, v20]
---

# 📚 v20 Base Ranker — Optuna Tuning (plain-language walkthrough)

> Tuning the **v20 XGBoost ranker itself** (the model that actually picks the stocks), not the
> veto on top of it. Companion to [[BCE Optuna Tuning — Step-by-Step]].

---

## 1. Why tune the base ranker?

The veto can only make v20's basket *less bad* — if v20 is net-negative, the best a veto does is
trim the losers. The real lever is **v20 itself**. And here's the opening: v20's settings were never
actually tuned — they were copied from the older v10 recipe ("same recipe as v10, only the candle
grid differs"). XGBoost also trains in seconds, so unlike the transformer we can afford a big search.

**Honest expectation:** the data is the same information-limited 1h price/volume. Tuning will likely
squeeze a little more ranking skill, but probably won't flip a sub-cost ranker into a profitable one.
It's worth trying because it's cheap.

---

## 2. What we're optimising

- **Metric:** walk-forward **rank-IC** (Spearman) — how well the model's score orders stocks by
  next-hour return — averaged over 8 time folds, **floored by the worst fold** (so a config that's
  great on average but collapses in one period is penalised). Score = ½·average + ½·worst-fold.
- **Why rank-IC, not profit?** It's the stable thing the certification (Gauntlet) actually tracks,
  and it's far less noisy than post-cost profit (which is what made the veto results seed-fragile).

## 3. The 9 knobs being searched
learning rate (`eta`), tree depth (`max_depth`), row/feature sampling (`subsample`,
`colsample_bytree`), two regularisers (`alpha` L1, `reg_lambda` L2), split-conservativeness
(`min_child_weight`, `gamma`), and the ranking loss (`objective`: pairwise vs ndcg). Tree count is
auto-chosen per fold by early-stopping, so it isn't searched.

---

## 4. The rules (same discipline as everything here)

- 🚫 **Optuna can't certify anything.** Only the **Validation Gauntlet** issues a verdict. This search
  is exploratory / ⚠️ UNVERIFIED.
- 🚫 **Gauntlet runs are costly** — each one makes *all future* significance tests on this data
  stricter (a multiple-testing correction). So we tune freely *here*, pick **one** winner, and only
  then — with explicit approval and a written-down hypothesis — spend **one** Gauntlet run.
- 🚫 **Don't touch the certified model.** The search never overwrites `v20_rolling_1h`; it writes a
  study database and a best-params file only.
- ⚠️ The rolling panel **overlaps** (each hour reused 4×), which makes the rho numbers look more
  significant than they are. So these numbers only *rank* configs; the winner gets re-checked on the
  non-overlapping ":15" subset before any claim.

---

## 5. Progress Log

### Setup ✅
- Built [tune_v20_xgb_optuna.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/training/tune_v20_xgb_optuna.py), reusing v20's exact 8-fold walk-forward + labelling.
- **Sanity check passed:** re-running v20's own settings reproduced its certified skill
  (Lρ 0.031 / Sρ 0.032 ≈ certified 0.032/0.033) — so the harness is faithful.
- Timing: **~3 min/trial** (16 GPU model-fits each). The current v20 recipe is enqueued as trial 0
  so every tuned config is measured against it on identical folds. Baseline score = **+0.0234**.

### Search — DONE ✅ (57 trials)
**Tuning found a real, balanced improvement** (unlike the transformer, which found nothing):

| metric | raw v20 | tuned best (trial 39) | lift |
|---|---|---|---|
| long rho | 0.0314 | 0.0338 | +7.8% |
| short rho | 0.0324 | 0.0361 | +11.5% |
| worst-fold | 0.0150 | 0.0202 | +34.8% |
| composite | 0.0234 | 0.0276 | +17.7% |

- **Winning recipe:** `rank:ndcg`, depth 4, eta 0.21, lambda 5.2, min_child_weight 11, subsample 0.78
  — vs v20's `pairwise / depth 5 / eta 0.03`. A whole cluster of trials converged here → robust, not luck.
- **Takeaway:** the base ranker's inherited (untuned) v10 recipe was genuinely sub-optimal; a shallower
  `ndcg` model ranks better on both sides and is steadier across folds.

### :15 non-overlapping re-check — DONE ❌ (the "win" did not survive)
The headline lift was an illusion on two counts:

| on honest :15 grid | baseline | tuned | Δ |
|---|---|---|---|
| rank-IC long | 0.0256 | 0.0269 | +0.0013 |
| rank-IC short | 0.0287 | 0.0285 | −0.0002 |
| **Top-1 LONG gross** | **+5.12 bps** | **+3.56 bps** | **−1.56** |
| Top-3 LONG gross | +4.69 | +4.19 | −0.49 |

1. **Overlap inflation:** the "+8–11% rho" shrank to ~+5%/flat once the overlapping windows were
   removed — most of the apparent gain was the effective-N≈¼ inflation the panel warned about.
2. **Wrong objective:** we optimised *average* rank-IC, but trading uses the **top of the book** —
   and there the tuned (shallow `ndcg`) model is *worse* (Top-1 LONG gross 5.12 → 3.56). Spreading
   skill evenly across the cross-section ≠ being sharp at the extreme picks.
3. Every cell stays **net-negative** at 6/10 bps for both models.

**Verdict: NOT tradeable. No Gauntlet run** (nothing better to certify; top-of-book degraded).
Lesson: rank-IC is the wrong tuning target when you trade Top-K, and overlapping panels flatter
significance. Artifact: `artifacts/v20_tuned_15anchor_eval.json`.

### Follow-up: tuned directly on Top-K (the tradeable metric), with a held-out test — ALSO dead ❌
Re-ran Optuna with objective = **Top-3 gross** (= net minus constant cost), on the :15 grid, worst-fold
floored, **last 2 folds (2025-10..11, 2026-02..03) held out** as a frozen test
([tune_v20_topk_optuna.py](file:///c:/Users/loq/Desktop/Trading/finalgo/scripts/training/tune_v20_topk_optuna.py), `artifacts/optuna_v20_topk_best.json`).
- On the optimization folds the tuned config beat baseline (+0.67 bps) — **but it did NOT generalize**:
  on the held-out folds the tuned model is *worse* (Top-1 LONG −2.77 bps; the SHORT side collapsed,
  gross +4.83 → −1.74). The opt-fold lift was overfitting a noisy objective.
- **Conclusion: tuning v20 is conclusively dead** — across BOTH objectives (rank-IC, Top-K) and BOTH
  layers (veto, base ranker), tuning never beat baseline OOS.

### 🌟 The actual find — hiding in the BASELINE
On the recent held-out folds, **raw (un-tuned) v20's LONG top-of-book is net-positive at 6 bps**:
Top-1 LONG **+3.64**, Top-3 **+2.66** net@6. (The earlier 8-fold :15 average hid this — v20's LONG
edge is *stronger recently* than its long-run mean.) Caveats: only ~4 months; needs ~6 bps execution
(≈breakeven at 10 bps, consistent with its FILTER_GRADE-at-10bps certification); LONG only (short weak).
**Implication:** the lever is **execution economics** (can we trade raw v20-LONG at ~6 bps limit-order
cost, does the recent strength persist?) — a data/execution question, NOT model tuning.
