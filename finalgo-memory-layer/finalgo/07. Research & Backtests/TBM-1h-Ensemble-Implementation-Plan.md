# TBM 1-Hour Ensemble — Finalized Implementation Plan

**Owner:** signal-generation (1h layer)  **Status:** EXECUTED — BOTH SIDES KILLED  **Date completed:** 2026-06-09

> [!ERROR] **Final Outcome (2026-06-09) — BOTH SIDES KILLED:**
> - **Short TBM (A+B+C):** net WR @6bps **44.9%** (raw 50.7%), expectancy **−6.42 bps**, t=**−4.77**, **0/5** folds. KILLED. *The earlier "56.5% / t=4.48 / 5-of-5 near-miss" was a **cost-sign bug** — the harness added the 6bps cost to shorts instead of subtracting it. Fixed in `purged_wf_tbm.py`; WF re-run; result retracted.*
> - **Long TBM (A+B+C+D):** 43.3% net WR, 0/5 folds. KILLED. No selectable signal at any barrier/horizon (tested 6 ways incl. a barrier-geometry scan and a steelman retrain).
> - **Root cause:** With correct costs, both sides have ~50% raw WR and lose the round-trip cost; selection skill ~0 (short +0.63pp, long +1.22pp). At 1h scale, direction is not predictable from lagging price/volume features in this universe.
> - **Process lesson:** verify `median(net − gross) == −cost` per side and check RAW vs NET WR before trusting any headline.
> - **Full results:** [[08. Model Analysis/1-Hour Vanguard Model/TBM-1h-Ensemble-Results]]

This plan supersedes the draft "Triple Barrier Method: Research-Grade Architecture." It locks the
two barrier contradictions, aligns the design to the stated deliverable, and turns the single
CatBoost model into the decorrelated ensemble that is the actual goal.

---

## 0. The Contract & Locked Decisions

**Deliverable:** the 1h signal-generation layer outputs Long/Short candidates with **net win-rate ≥ 57%**
(after 6 bps round-trip cost) at **usable volume**. Downstream layers (daily gatekeeper, 15m edge,
2-layer Gemini veto) consume these candidates and handle final execution/risk.

> ⚠️ **Win-rate is the contract → barriers are SYMMETRIC.** With 57% net WR as the deliverable, a 2:1
> reward:risk geometry is mathematically incompatible (2:1 breaks even at 33% WR; 57% WR there ⇒
> profit factor ≈2.6, a leakage signature). Symmetric barriers make 50% the breakeven and 57% a
> meaningful, believable target (PF ≈1.33 gross). Asymmetric/EV-max is deferred to V2 as a separate
> *expectancy-ranked* variant — do not mix the two metrics.

| # | Decision | Value |
|---|----------|-------|
| D1 | Target metric | **Net win-rate ≥ 57%** @ 6 bps (also report @ 10 bps) |
| D2 | Barriers | **Symmetric, ATR-scaled**: TP = +m·ATR, SL = −m·ATR (m default 1.0, tuned in P0) |
| D3 | One-definition rule | Barriers used for **labeling = EV = live execution**. No second geometry anywhere. |
| D4 | ATR | 14-period ATR on **1h** bars; **cost floor**: require m·ATR ≥ 3× round-trip cost, else skip the bar |
| D5 | Horizon (vertical) | 1 hour = 4 × 15-min sub-bars (T+15, T+30, T+45, T+60) |
| D6 | Path resolution | Walk sub-bars sequentially; ambiguous (both barriers inside one sub-bar) ⇒ **assume STOP first** (conservative). **Never discard** — discarding is MNAR and biases toward calm regimes. |
| D7 | Cost | 6 bps primary (user real all-in), 10 bps reported alongside |
| D8 | Label space | CatBoost **MultiClass** {0=SL, 1=TP, 2=Timeout} + stored **realized net return** per sample |
| D9 | Sample weights | **Label-uniqueness / concurrency** (de Prado) — NOT ATR weighting (double-counts vol under D2) |
| D10 | Universe | **Gatekeeper-conditioned**: Long models train/eval on `long_eligible` names only; Short on `short_eligible`. Requires historical eligibility panel (see P0-T0). |
| D11 | Features | 3 **decorrelated views** (mean-reversion / trend / volatility). **Drop all time features** (`Hour`, `Time_To_Close`, `Is_Open_Hour`, `DayOfWeek`) — proven overfit clock (v18/v19). |
| D12 | Ensemble | Per-view CatBoost MultiClass → isotonic calibration → **stacked combiner (OOF)** → EV filter → Top-K |
| D13 | EV gate | `EV = P_TP·R − P_SL·R + P_TO·E[ret|TO] − Cost`; trade if **EV > τ**. τ is the **win-rate dial**. |
| D14 | Validation | **Purged + embargoed** walk-forward; OOF stacking; bootstrap CIs + t-stats on every headline |

**Honest prior (read this):** the audit established that the unconditional 1h tape has no edge after
costs (v8/v10/v18/v19 all net-negative or coin-flip OOS). This plan is the *best remaining shot* via
three changes that were never combined before: (a) path/cost-aware **TBM labels** instead of `>0bps`
direction, (b) the **gatekeeper-conditioned** distribution the layer actually trades, and (c) a
**decorrelated ensemble** instead of duplicate learners. It has explicit **kill-criteria** (§9). If
those trip, 1h directional is dead and the answer is new data sources, not more tuning.

---

## 1. Architecture

```
 daily gatekeeper ── long_eligible / short_eligible (per day) ─────────┐
                                                                        ▼
 1h bars (from 15m cache) ──► TBM LABEL ENGINE (P1)                conditioned
   + 4×15m sub-bar path        symmetric ATR barriers, stop-first      universe
                               labels {SL,TP,TO} + realized ret
                               + uniqueness weights
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
        ▼ (feature view A)               ▼ (view B)                       ▼ (view C)
   CatBoost MC  mean-reversion     CatBoost MC  trend             CatBoost MC  volatility
        │  isotonic calib               │  isotonic calib               │  isotonic calib
        └───────────────┬───────────────┴───────────────┬───────────────┘
                        ▼  (out-of-fold predictions only)
                 STACKED COMBINER  →  calibrated P_TP / P_SL / P_TO
                        ▼
                 EV FILTER   EV > τ   (τ tuned on validation → net WR ≥ 57%)
                        ▼
                 TOP-K RANKER (by EV)  →  Long / Short candidates  →  downstream pipeline
```

Two independent stacks: **Long** (TP above / SL below) and **Short** (mirrored barriers).

---

## 2. Phase 0 — Diagnostics & Go/No-Go Gates  *(do this BEFORE any training)*

Cheap, decisive. One afternoon. Each gate can kill or reshape the build.

| Task | Output | GATE |
|------|--------|------|
| **P0-T0** Generate **historical gatekeeper panel** by replaying gatekeeper logic over 3y daily data → `data/gatekeeper_panel_3y.parquet` (date × ticker × {long_elig, short_elig}) | eligibility panel | **G0:** panel reconstructable & deterministic. If gatekeeper isn't replayable historically, conditioned-universe training is blocked → fall back to full universe + flag. |
| **P0-T1** Ambiguity-rate analysis: % of resolved labels where both barriers fall in one 15m sub-bar, **bucketed by ATR/vol** | table + plot | **G1:** confirm stop-first rule (D6) is applied; quantify MNAR exposure. No hard threshold — informational, must be reported. |
| **P0-T2** Class balance of {SL,TP,TO} under D2 barriers at m∈{0.75,1.0,1.5}; pick **m** giving balanced TP/SL and TO < ~50% | m selection | **G2:** each class ≥ 10% (calibration needs positives). |
| **P0-T3** Cost-floor coverage: fraction of bars/names where m·ATR ≥ 3×cost | coverage % | **G3:** ≥ ~60% of conditioned-universe bars clear the floor, else raise m or horizon. |
| **P0-T4** Breakeven reconciliation: confirm symmetric ⇒ 50% breakeven; 57% ⇒ PF≈1.33 | sanity note | **G4:** metric/geometry coherent (auto-pass under D2). |
| **P0-T5** Decorrelation pre-check: quick single-view models, pairwise corr of P_TP | corr matrix | **G5:** view pairwise corr < ~0.7. If views are redundant, the ensemble adds nothing → rethink views. |

**Script:** `scripts/research/tbm_step0_diagnostic.py`

---

## 3. Phase 1 — TBM Label Engine

**Script:** `scripts/labeling/tbm_label_engine.py`  →  `data/tbm_labels_1h.parquet`

1. Build 1h bars per the **existing clean convention** (09:15-anchored IST, session-masked, drop the
   last tradeable bar so the vertical never crosses overnight — reuse `rebuild_aligned_datasets.py`
   conventions; see `reference_ranking_data_conventions`).
2. For each 1h signal bar at T with entry = close[T]:
   - `ATR = ATR14(1h)[T]`; `R = m·ATR`. If `R < 3×cost` → skip (D4).
   - TP = entry·(1+R), SL = entry·(1−R)  *(Long; mirror for Short)*.
   - Walk 15m sub-bars T+15…T+60 in order:
     - sub-bar High ≥ TP **and** Low ≤ SL ⇒ **SL** (stop-first, D6)
     - else High ≥ TP ⇒ **TP**; else Low ≤ SL ⇒ **SL**
     - first touch wins; record touch bar.
   - No touch by T+60 ⇒ **Timeout**, `realized_ret = close[T+60]/entry − 1`.
3. Store per sample: `label∈{0,1,2}`, `realized_gross_ret`, `realized_net_ret = gross − cost`,
   `ATR`, `R`, `side`, `ticker`, `datetime`, `t_touch`.
4. **Uniqueness weights (D9):** compute concurrency = # of other labels whose [T, t_touch] window
   overlaps; weight = 1/avg_concurrency (de Prado `getAvgUniqueness`). Persist `weight`.
5. Apply **D10** conditioning: tag each row with gatekeeper eligibility from `gatekeeper_panel_3y`.

---

## 4. Phase 2 — Feature Views & Base Learners

**Scripts:** `scripts/features/build_feature_views.py`, `scripts/training/train_tbm_base_learners.py`

**Views (D11, time features dropped):**
- **A — Mean-reversion:** IBS, IBS_3, Buy_Pressure, Upper/Lower_Shadow, VWAP_Dist, PercentB, Stoch_K/D, WPR_14, Dist_BB_*.
- **B — Trend:** Return, Log_Return, ROC_12, MOM_12_pct, PPO/PPO_Signal/PPO_Hist, Dist_SMA/EMA/HMA_*, Dist_Donchian_*, Vortex_±, Up/Down_Streak, Direction_Consistency_*.
- **C — Volatility/structure:** Keltner_Width, BB_Width, Donchian_Width, HL_Range, OC_Range, ATR-derived, RVOL, Volume_Zscore, Rolling_Skew/Kurt, Relative_Volatility, Market_Mean_Volatility.

Each view × side → **CatBoost MultiClass** (oblivious trees, depth 5, lr 0.03, GPU, eval_metric
`AUC`/`MultiClass`, early stopping on the validation slice). 6 base models total (3 views × 2 sides).

> **Leak fix carried from v19:** impute NaNs with **train-fold** statistics computed inside the WF
> loop, never global. Pass-through scaler (trees scale-invariant).

---

## 5. Phase 3 — Calibration & Stacked Combiner

**Scripts:** `scripts/training/calibrate_tbm.py`, `scripts/training/train_tbm_combiner.py`

1. **Isotonic calibration** of each base learner's class probabilities via
   `CalibratedClassifierCV(method='isotonic')` fit on a dedicated calibration slice (subset of train,
   never val/test). Renormalize the 3 calibrated probs to sum to 1.
2. **Stacking combiner:** logistic regression (or shallow GBM) trained on **out-of-fold** base
   predictions (9 features = 3 views × 3 classes; optionally + regime features from the gatekeeper).
   Output = final calibrated `P_TP / P_SL / P_TO`. OOF is mandatory — fitting the combiner on
   in-fold base preds re-introduces leakage.
3. Baseline to beat: a hand-built **unanimous AND-gate** (all views agree TP) — if the learned
   combiner can't beat the dumb gate on volume-at-57%, keep the gate.

---

## 6. Phase 4 — EV Execution Filter & WR Dial

**Script:** `scripts/execution/ev_filter.py`

- `EV = P_TP·R − P_SL·R + P_TO·E[ret|TO] − Cost`, with `R = m·ATR` (gross), `E[ret|TO]` estimated from
  **training** timeout realized returns (do not assume 0), `Cost` = round-trip (6 bps).
- Trade iff **EV > τ**. **τ is the win-rate dial:** sweep τ on the **validation** fold, pick the
  smallest τ that yields net WR ≥ 57%, **freeze it**, then read OOS. Report **volume retained at τ**
  — that number, not WR, is the quality of the ensemble (WR is trivially reachable by raising τ).
- Net WR definition: fraction of **executed** trades with `realized_net_ret > 0`.

---

## 7. Phase 5 — Purged Walk-Forward Validation

**Script:** `scripts/validation/purged_wf_tbm.py`

- Folds: min_train 18mo, val 4mo, test 2mo, step 4mo (reuse v19 fold scheme).
- **Purge:** drop any train sample whose label window [T, t_touch] overlaps the test block.
- **Embargo:** 1 trading day on **both** sides of each test block.
- Apply **uniqueness weights** in fit and in metric aggregation.
- All base learners + calibration + combiner + τ are fit **strictly within each fold's train/val**;
  test is touched once.
- **Metrics (business, per-fold + pooled, with bootstrap CI + t-stat):** Net WR, Net expectancy
  (bps/trade), Profit Factor, Avg trade return, **Volume (trades/month)**, Max DD. Report @6 and @10 bps.

---

## 8. Phase 6 — Capital Allocation (Top-K)

EV-passing candidates ranked by **EV** (V1 — simple, no extra model). Top-K per timestamp passed to
the downstream pipeline. *(LambdaMART secondary ranker is V2 scope — do not build in V1; EV-rank is
sufficient and avoids another fittable surface.)*

---

## 9. Acceptance Criteria & Kill-Criteria

**Definition of Done (V1 ships if ALL hold on pooled OOS):**
- Net WR ≥ **57%** @ 6 bps at frozen-on-validation τ.
- **Lower bootstrap CI of net expectancy > 0** (profitable, not just high WR).
- Volume ≥ **floor** (define with you; e.g., ≥ ~1 candidate/day across conditioned universe).
- **Reproducible:** from-scratch WF retrain reproduces the headline (the test v10 failed). No single
  fold/month drives the result.

**Kill-criteria (stop; escalate to new data sources, do not tune further):**
- At the max plausible τ, either net-expectancy lower CI ≤ 0 **or** volume-at-57%-WR below floor.
- Base views fail decorrelation (G5) — ensemble can't lift precision.
- Headline survives only in-sample (WF/from-scratch gap like v10) — provenance failure.

---

## 10. Module / Artifact Layout

```
scripts/research/tbm_step0_diagnostic.py
scripts/labeling/tbm_label_engine.py
scripts/features/build_feature_views.py
scripts/training/train_tbm_base_learners.py
scripts/training/calibrate_tbm.py
scripts/training/train_tbm_combiner.py
scripts/execution/ev_filter.py
scripts/validation/purged_wf_tbm.py
data/gatekeeper_panel_3y.parquet
data/tbm_labels_1h.parquet
data/tbm_feature_views/{A_meanrev,B_trend,C_vol}.parquet
models/tbm_1h_ensemble/{long,short}/{viewA,viewB,viewC}.cbm + calibrators + combiner.pkl + meta.json
data/model_analysis/tbm_1h/  (WF results, CIs, τ-sweep, volume-at-WR curves)
```

---

## 11. Build Order (dependency-correct)

1. **P0** diagnostics + gates (incl. historical gatekeeper panel) ← *start here; can kill the project cheaply*
2. **P1** label engine
3. **P2** feature views + base learners
4. **P3** calibration + combiner
5. **P4** EV filter + τ dial
6. **P5** purged WF + metrics  ← *the only numbers anyone is allowed to quote*
7. **P6** Top-K
8. Gate against §9 → ship or kill.

---

## 12. Deferred to V2 (explicitly out of scope for V1)
- ATR-asymmetric / EV-max variant (expectancy-ranked, different metric than 57% WR).
- LambdaMART capital allocator.
- Finer-than-15m first-touch (1-min refetch via existing `collect_upstox_15min_3y` source).
- Orthogonal public-data features (delivery%, PCR/OI, India VIX, FII/DII) as a 4th view.

## Backlinks
- `[[project-v18-v19-directional-deadend]]` — why `>0bps` direction failed; the prior this plan fights.
- `[[reference-ranking-data-conventions]]` — session-masking / overnight rules for the label engine.
- `[[project-v2-15min-3y-model]]` — the 15m edge + veto-value method this layer feeds.
