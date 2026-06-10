# 📊 Model Performance & Statistics

This document serves as the absolute technical ledger for all machine learning models in the **Vanguard High-Precision Intraday Trading Engine**. It details their structural setups, training parameters, out-of-sample Spearman correlations, feature dimensions, and historical simulation performance.

---

## 📈 Centralized Model Comparison Matrix

The Vanguard engine supports multiple model specialities. The matrix below contrasts all registered and active XGBoost ranking models:

| Model ID | Timeframe | Training Horizon | Features | Scaling Method | Long Spearman Rho | Short Spearman Rho | Precision @ 3 (Win Rate) | Primary Operational Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`daily_xgb`** | Daily | 5 Years (Upstox) | 165 | Scale-Invariant (Pass-Through) | `0.0204` | `0.0196` | Long: `50.8%`<br>Short: `50.6%` | **Macro Universe Gatekeeper** (DEAD / DEAD post-remediation; Run: `20260610T102743Z-5f7d069f`) |
| **`TBM_1h_short`** | 1-Hour | 3 Years (Upstox) | 79 (3 views) | CatBoost MultiClass | *N/A* | *N/A* | Short: `44.9%` net @6bps (raw 50.7%) | **KILLED — net-NEGATIVE (−6.42bps, t=−4.77). Earlier "56.5%/5-of-5/t=4.48" was a cost-sign bug, now fixed & retracted.** |
| **`TBM_1h_long`** | 1-Hour | 3 Years (Upstox) | 105 (4 views) | CatBoost MultiClass | *N/A* | *N/A* | Long: `43.3%` net @6bps | **KILLED; 0/5 folds; no selectable signal at any barrier/horizon (6 approaches tested)** |
| **`v18_random_forest_1h`**| 1-Hour | 3 Years (Upstox) | 86 | Pass-Through | *N/A (Classifier)* | *N/A* | Net Hit L: `40.0%`<br>S: `39.9%` | Pure Direction Classifier (>0bps) |
| **`v17_random_forest_1h`**| 1-Hour | 3 Years (Upstox) | 86 | Pass-Through | *N/A (Classifier)* | *N/A* | Net Hit L: **`46.2%`**<br>S: `43.4%` | **High-Recall Breakout Filter** |
| **`v16_binary_breakout_1h`**| 1-Hour | 3 Years (Upstox) | 86 | Pass-Through | *N/A (Classifier)* | *N/A* | Net Hit L: `34.0%`<br>S: `48.0%` | Gradient Boosting Breakout Filter |
| **`v15_lambdamart_es_1h`**| 1-Hour | 3 Years (Upstox) | 86 | Pass-Through | `N/A` | `N/A` | *N/A* | LambdaMART with ndcg@3 early stopping |
| **`v14_lambdamart_no_es`**| 1-Hour | 3 Years (Upstox) | 86 | Pass-Through | `N/A` | `N/A` | *N/A* | LambdaMART fixed 500 rounds |
| **`v13_ndcg_raw_1h`** | 1-Hour | 3 Years (Upstox) | 86 | Pass-Through | `N/A` | `N/A` | *N/A* | Raw NDCG@5 ranker |
| **`v12_lambdamart_1h`**| 1-Hour | 3 Years (Upstox) | 86 | Pass-Through | `N/A` | `N/A` | *N/A* | LambdaMART top-5 ranker |
| **`v11_utility_1h`** | 1-Hour | 3 Years (Upstox) | 86 | Pass-Through | `N/A` | `N/A` | *N/A* | Utility-target 1h ranker |
| **`v10_native_1h`** (depth-4) | 1-Hour | 3 Years (Upstox) | 86 | Pass-Through | **`+0.027` (decaying to `+0.013`)** | **`+0.025` (decaying to `+0.005`)** | Top-3 raw win L `51.6%`/S `54.9%`; Net @6bps: **−3.4 / −2.5 bps** | **Real but sub-cost raw signal — same class as v8** (FILTER_GRADE / FILTER_GRADE; Run: `20260610T093001Z-5f7d069f`) |
| **`v9_clean_1h`** | 1-Hour | 3 Years (Upstox) | 86 | Pass-Through | `N/A` | `N/A` | *N/A* | Cleaned dataset baseline |
| **`v8_upstox_3y`** | 1-Hour | 3 Years (Upstox) | 86 | StandardScaler (Pass-Through) | ~~`0.0461`~~ → **`0.026` (decaying to `0.012`)** | ~~`0.0490`~~ → **`0.024` (decaying to `0.004`)** | Top-3 Net @6bps: **−3.6 / −3.7 bps** | **DEMOTED — KILLED (net-negative under genuine WF)** (FILTER_GRADE / FILTER_GRADE; Run: `20260610T092549Z-5f7d069f`) |
| **`v4_regime`** | 1-Hour | 2 Years (yfinance) | 86 | **Scale-Free** (No Scaler) | **`0.0820`** | **`0.0850`** | *N/A* | Regime-Aware Multi-TF Specialist (Offline Benchmark) |
| **`v3_sector`** | 1-Hour | 2 Years (yfinance) | 95 | StandardScaler | `0.0558` | `0.0574` | Long: `55.4%`<br>Short: `57.2%` | Sector Momentum & VWAP Explorer (Archive) |
| **`v2_15min_3y`** | 15-Min | 3 Years (Upstox) | 86 | Scale-Invariant (Pass-Through) | `0.0608` | `0.0610` | Long: **`58.0%`**<br>Short: `57.0%` | **New High-Frequency Standard** (FILTER_GRADE / FILTER_GRADE; Run: `20260610T095723Z-5f7d069f`) |
| **`v1_15min`** | 15-Min | 1 Year (Upstox) | 95 | Scale-Invariant (Pass-Through) | `0.0571` | `0.0558` | Long: `58.9%`<br>Short: `57.4%` | Deprecated 1-Year Baseline |
| **`v1_30min`** | 30-Min | 1 Year (Upstox) | 95 | Scale-Invariant (Pass-Through) | `0.0449` | `0.0433` | Long: `57.1%`<br>Short: `54.9%` | Mid-frequency Volatility Specialist (Standalone backtest candidate) |
| **`v2_feature_fix`**| 1-Hour | 2 Years (yfinance) | 81 | StandardScaler | `0.0406` | `0.0409` | Long: `53.3%`<br>Short: `55.6%` | Fixed lag features and Nifty50 Regime (Archive) |
| **`v7_yfinance_2y`**| 1-Hour | 2 Years (yfinance) | 86 | StandardScaler | `0.0338` | `0.0357` | *N/A* | Legacy feature math validation (Archive) |
| **`v6_feat_fixed`** | 1-Hour | 60 Days (Upstox) | 86 | StandardScaler | `0.0268` | `0.0279` | *N/A* | Retrained debug model (Archive) |
| **`v1_yfinance`** | 1-Hour | 2 Years (yfinance) | 54 | StandardScaler | `0.0320` | `0.0350` | *N/A* | Original baseline production model (Archive) |

---

## 🎛️ Individual Model Profiles & Walk-Forward Valuations

### 1. Daily Macro Gatekeeper (`daily_xgb`)
Trained via rectified script `scripts/train_daily_xgboost_v2.py`. Evaluates trend regimes over daily bars and selects the top 40% long and short candidacies.

*   **Training Dataset Size**: 211,915 daily rows spanning May 2021 to May 2026.
*   **Evaluation Method**: 11-Fold Walk-Forward Temporal Validation (splits by calendar month to prevent data leakage).
*   **Target Label**: Next-day cross-sectional return ranks.
*   **Walk-Forward Performance Summary (Top 3 selections, post-remediation)**:
    *   **Average Long Spearman Rho**: `0.0204`
    *   **Average Short Spearman Rho**: `0.0196`
    *   **Average Top-3 Long Return**: `+11.74 bps` raw (`+1.74 bps` net @ 10bps)
    *   **Average Top-3 Short Return**: `-3.09 bps` raw (`-13.09 bps` net @ 10bps)
    *   **Average Combined Edge**: **`+8.65 bps`** raw combined return edge
    *   **Post-Remediation Verdict**: **DEAD** (Long) / **DEAD** (Short) - Run ID: `20260610T102743Z-5f7d069f`
*   **Production Model Setup**:
    *   **Train Set**: May 2021 to March 2026 (59 months).
    *   **Validation Set** (Used for early stopping): April 2026 to May 2026.
    *   **Model Iterations**: Long: `104` rounds | Short: `66` rounds.
*   **Top Predictor Features (Long)**:
    1.  `PPO_Hist` (Gain: `2.12`): Trend momentum histogram.
    2.  `Alpha_3D` (Gain: `1.84`): Short-term stock outperformance.
    3.  `Alpha_10D` (Gain: `1.79`): Long-term stock outperformance.
    4.  `RSI_7` (Gain: `1.47`): Fast momentum index.
    5.  `Relative_Return` (Gain: `1.36`): Daily relative return vs. market.
*   **Top Predictor Features (Short)**:
    1.  `Log_Return` (Gain: `4.59`): Log return over the last candle.
    2.  `Alpha_3D` (Gain: `3.65`): Stock relative drawdown.
    3.  `Return` (Gain: `3.01`): Raw simple return.
    4.  `Dist_SMA_5` (Gain: `2.12`): Fast moving average distance.

### 2. 1-Hour Intraday Specialist (`v8_upstox_3y`) - ACTIVE CHAMPION
Our main trading engine ranker. Injects technical, volume profile, and microstructure features.

*   **Training Dataset Size**: 3-Year historical Upstox hourly bars (falling back to a 90-day cross-sectional dataset of 63,619 rows, 296 train queries, and 74 test queries).
*   **Out-of-Sample Performance**:
    *   **Long Test Spearman Rho**: `0.0461`
    *   **Short Test Spearman Rho**: `0.0490`
*   **Top Features (Long)**:
    1.  `IBS` (Intraday Bar Position): Pinpoints candle close proximity to high/low.
    2.  `Buy_Pressure`: Relative volume-weighted buying dominance.
    3.  `Return_Accel`: Shift in price velocity over previous hours.
    4.  `Lower_Shadow`: Identifies bullish rejection zones.
*   **Scaler Requirement**: Fits standard scaler `models/scaler.pkl` with parameters `with_mean=False` and `with_std=False`. Because the parameters are off, it is mathematically a pass-through, preserving XGBoost's native **scale-invariance** while preventing scaler distortion!

### 3. 30-Minute Specialist (`v1_30min`)
Resampled specialist designed to filter mid-frequency noise.

*   **Training Dataset Size**: 541,143 resampled 30-minute rows spanning 1 year.
*   **Evaluation Method**: 3-Fold Walk-Forward Temporal Validation.
*   **Walk-Forward Performance Summary**:
    *   **Average Long Spearman Rho**: `0.0449`
    *   **Average Short Spearman Rho**: `0.0433`
    *   **Average Long Win Rate @ K=3**: **`57.1%`**
    *   **Average Short Win Rate @ K=3**: **`54.9%`**
    *   **Combined Edge**: **`+0.1404%`** per 30-min bar over market.
*   **Production Model Setup**:
    *   **Train Set**: May 2025 to April 2026.
    *   **Validation Set**: May 2026.
    *   **Best Iteration**: Long: `95` | Short: `92`.
*   **Top Predictor Features (Long)**:
    1.  `IBS` (Gain: `5.87`): Proximity to bar low.
    2.  `Is_Close_Hour` (Gain: `4.59`): Identifies EOD liquidity patterns.
    3.  `Buy_Pressure` (Gain: `3.79`): Microstructure order dominance.

### 4. 15-Minute Specialist (`v2_15min_3y`)
Our highest resolution timeframe model, completely upgraded to the 3-Year data mandate to neutralize regime overfitting.

*   **Training Dataset Size**: 3,190,598 rows of 15-minute bars spanning 3.5 years.
*   **Evaluation Method**: 6-Fold Walk-Forward Temporal Validation.
*   **Walk-Forward Performance Summary**:
    *   **Average Long Spearman Rho**: `0.0608`
    *   **Average Short Spearman Rho**: `0.0610`
    *   **Average Long Win Rate @ K=3**: **`58.0%`**
    *   **Average Short Win Rate @ K=3**: **`57.0%`**
    *   **Combined Edge**: **`+0.0938%`** per 15-min bar.
*   **Production Model Setup**:
    *   **Train Set**: Jan 2023 to May 2026.
    *   **Validation Set**: June 2026.
*   **Top Predictor Features (Long)**:
    1.  `IBS` (Gain: `52.67`): Strong mean-reversion signal indicating bar distribution.
    2.  `Buy_Pressure` (Gain: `47.23`): Immediate liquidity imbalance.
    3.  `Log_Return` (Gain: `33.33`): Relative volatility tracking.
    4.  `Lower_Shadow` (Gain: `15.40`): Rejection of low prices.

---

## ⚙️ Model Hyperparameters & Loss Configurations

Vanguard utilizes the **pairwise ranking** objective in XGBoost to optimize stock lists cross-sectionally rather than predicting raw asset returns (which are notoriously noisy and unstable).

### Pairwise Ranking Objective: `rank:pairwise`
*   **Concept**: Minimizes the number of out-of-order pairs in prediction lists compared to actual market returns. The model is trained to score stock $A$ higher than stock $B$ if return $R_A > R_B$:
    $$\mathcal{L} = - \sum_{i,j: R_i > R_j} \ln \sigma(\hat{y}_i - \hat{y}_j)$$
*   **Evaluation Metric**: Normalized Discounted Cumulative Gain (`ndcg@3` / `ndcg@5`). NDCG optimizes models specifically to surface the absolute strongest long/short candidates at the very top of our output rankings:
    $$NDCG@K = \frac{DCG@K}{IDCG@K}, \quad DCG@K = \sum_{i=1}^K \frac{2^{rel_i} - 1}{\log_2(i + 1)}$$

### Hyperparameter Register (Unified JSON Schema)

```json
{
  "objective": "rank:pairwise",
  "tree_method": "hist",
  "device": "cuda",
  "eta": 0.03,
  "max_depth": 5,
  "subsample": 0.8,
  "colsample_bytree": 0.8,
  "alpha": 1.0,
  "lambda": 2.0,
  "min_child_weight": 10,
  "eval_metric": "ndcg@3",
  "random_state": 42
}
```

> [!NOTE]
> For the **Daily Macro Gatekeeper (`daily_xgb`)**, L1 and L2 regularizations are programmatically doubled (`alpha = 2.0`, `lambda = 4.0`), learning rate is slowed (`eta = 0.01`), and nodes size is constrained (`min_child_weight = 40`) to prevent overfitting on noisy daily macro indicators.

---

## 📈 Out-of-Sample Evaluation Report (`models/eval_report.json`)

Vanguard maintains a centralized evaluation file (`models/eval_report.json`) to track live out-of-sample performance over test datasets.

*   **Overall Spearman Mean**: `0.2348` (High cross-sectional rank correlation)
*   **Top-1 Prediction Accuracy**: `9.17%` (Refers to stock outperforming the universe at the top-1 rank)
*   **Top-3 Overlap Index**: `15.90%`
*   **Mean Absolute Error (MAE)**: `0.0400`
*   **Root Mean Squared Error (RMSE)**: `0.0542`
*   **Financial Edges (Intraday)**:
    *   **Average Return (Top-1 Candidates)**: **`+0.3318%`** per trading window.
    *   **Average Return (Top-3 Candidates)**: **`+0.2475%`** per trading window.

---

## 🔄 Dual-Stage Conviction Ranking (V2.3 Architecture)

> [!IMPORTANT]
> **Retirement of the V1 Inversion Workaround:**
> 1. In Vanguard V1, a single model was trained, which exhibited a negative Spearman correlation of `-0.55` due to volume distribution anomalies. As a workaround, V1 negated raw predictions (`-predictions`) to capture mean-reversion.
> 2. In **Vanguard V2.3**, the inversion workaround has been completely retired.
> 3. Instead, the engine trains **separate, specialized Long and Short models** (e.g., `xgb_long_model.json` and `xgb_short_model.json`).
> 4. Conviction is calculated directly at runtime (e.g., `Long_Conviction = long_score - short_score`), and tickers are sorted in descending order (highest conviction first). The target labels are natively aligned during training, eliminating any need for raw output negations.

---

## 🧬 Hybrid Ensemble Frameworks (Rank + Veto)

Vanguard extensively backtested combining ranking models (LambdaMART) with binary breakout classifiers (Gradient Boosting / Random Forest) to artificially overcome the 10 bps statutory fee hurdle. The core theory was to use the Classifier to **veto** the Ranker's top choices if they lacked sufficient breakout probability.

### **V10 Ranker + V17 Random Forest Classifier**
*   **Architecture**: `v10` (LambdaMART Ranker) generates cross-sectional ranks, and `v17` (Random Forest Classifier, targeting >20bps) issues a veto if probability < 62%.
*   **Logic A (Rank then Veto)**: `v10` ranks the entire universe. The top K candidates are evaluated. If `v17` predicts < 62% breakout probability, the trade is vetoed.
*   **Logic B (Filter then Rank)**: `v17` filters the entire universe, passing only candidates > 62% probability. `v10` then ranks the survivors to pick the absolute best.
*   **Walk-Forward Outcome (8 Folds / 3 Years)**:
    *   **Logic A1 (Rank Top 1, then Veto)**
        *   **Long**: 51 trades | Raw: +0.34 bps | **Net: -9.66 bps** | Net Hitrate: 39.2%
        *   **Short**: 379 trades | Raw: +6.79 bps | **Net: -3.21 bps** | Net Hitrate: 51.2%
    *   **Logic A3 (Rank Top 3, then Veto)**
        *   **Long**: 139 trades | Raw: +12.31 bps | **Net: +2.31 bps** | Net Hitrate: 48.9%
        *   **Short**: 758 trades | Raw: +0.48 bps | **Net: -9.52 bps** | Net Hitrate: 47.6%
    *   **Logic B (Filter, then Rank)**
        *   **Long**: 182 trades | Raw: +4.66 bps | **Net: -5.34 bps** | Net Hitrate: 44.0%
        *   **Short**: 749 trades | Raw: +6.04 bps | **Net: -3.96 bps** | Net Hitrate: 49.0%
*   **Edges Discovered**: The only viable edge discovered was in **Logic A3 Long**, which successfully cleared the 10 bps statutory fee hurdle yielding a mathematical net profit of **+2.31 bps** across 139 trades. All short-side configurations and broader filter logic (Logic B) completely failed to clear the fee overhead.
*   **Conclusion**: The Random Forest's high-recall nature allows a healthy volume of trades to pass the filter (unlike the extremely restrictive `v16` Gradient Boosting model). However, its precision remains too weak to consistently clear the 10 bps fee on the short side.

### **V10 Ranker + V18 Pure Direction Classifier (>0 bps target)**
*   **Architecture**: `v10` (LambdaMART Ranker) generates cross-sectional ranks, and `v18` (Random Forest Classifier, targeting >0bps pure direction) issues a veto if probability < 52%.
*   **Inference Backtest (Untouched 12-Month Out-of-Sample / July 2025 - June 2026)**:
    *   **Logic A1 (Rank Top 1, then Veto)**
        *   **Long**: 339 trades | **Net: +10.2 bps** | Raw Win: 61.4%
        *   **Short**: 523 trades | **Net: +9.0 bps** | Raw Win: 66.5%
    *   **Logic A3 (Rank Top 3, then Veto)**
        *   **Long**: 1028 trades | **Net: +4.6 bps** | Raw Win: 59.0%
        *   **Short**: 1543 trades | **Net: +4.6 bps** | Raw Win: 63.0%
    *   **Logic B (Filter, then Rank)**
        *   **Long**: 465 trades | **Net: +6.1 bps** | Raw Win: 57.6%
        *   **Short**: 942 trades | **Net: +5.6 bps** | Raw Win: 62.5%
*   **True Portfolio Analytics (Logic A3 / 12-Months)**:
    *   **Base Allocation (20% Cash / No Leverage)**:
        *   **Cumulative Return**: +26.45%
        *   **Max Drawdown**: -3.68%
        *   **Annualized Sharpe Ratio**: 3.38
    *   **Aggressive Allocation (30% Cash / 5x Leverage)**:
        *   **Cumulative Return**: +396.06%
        *   **Max Drawdown**: -25.18%
        *   **Annualized Sharpe Ratio**: 3.44
*   **Conclusion**: Lowering the target threshold to `>0 bps` (pure direction) and dropping the veto threshold to `52%` completely solved the short-side bleeding observed in V17. The V18 framework acts as the ultimate filter, clearing the 10 bps fee mathematically across all configurations. The portfolio analytics prove the system is highly stable, generating a 3.38+ Sharpe Ratio that survives immense intraday leverage.

---

---

## 🧱 TBM 1-Hour Ensemble (2026-06-09)

A research-grade Triple Barrier Method ensemble was implemented to replace the directional classifiers (v18/v19) for the 1h signal-generation layer. See [[08. Model Analysis/1-Hour Vanguard Model/TBM-1h-Ensemble-Results]] for full fold-by-fold results.

### Design

- **Labeling:** `tbm_label_engine.py` assigns {SL=0, TP=1, Timeout=2} by walking 4 × 15m sub-bars after each signal bar. Symmetric ATR barriers (m=1.0), stop-first rule for ambiguous bars. 647,298 labeled rows.
- **Feature views:** 4 decorrelated views; time features dropped.
  - A: Mean-reversion (21 feats — IBS, Buy_Pressure, shadows, VWAP, oscillators)
  - B: Trend (31 feats — MA distances, momentum, relative return)
  - C: Volatility (29 feats — BB/Keltner width, volume, regime)
  - D: Momentum/breakout (26 feats — ADX, ORB, gap, RS — **long only**)
- **Per-view model:** CatBoost MultiClass (500 trees, depth 5, GPU) + isotonic calibration
- **Ensemble:** OOF stacked combiner (logistic regression on concatenated class probs)
- **EV filter:** `EV = P_TP·R − P_SL·R + P_TO·E[ret|TO] − Cost`; τ swept on val to hit ≥57% WR
- **Validation:** purged + embargoed walk-forward (18mo train / 4mo val / 2mo test / 4mo step)

### Key Results (CORRECTED — cost-sign bug fixed 2026-06-09)

| Side | Views | Net WR @6bps | Exp (bps) | t-stat | Folds+ | Status |
|---|---|---|---|---|---|---|
| Short | A+B+C | ~~56.5%~~ → **44.9%** | ~~+5.18~~ → **−6.42** | ~~+4.48~~ → **−4.77** | ~~5/5~~ → **0/5** | **KILLED (net-negative)** |
| Long  | A+B+C+D | 43.3% | −5.22 | −2.98 | 0/5 | KILLED |

> [!ERROR] The short "56.5% / t=4.48 / 5-of-5" was a **cost-sign bug** — the harness added the 6bps cost to shorts instead of subtracting it. True short: raw WR 50.7%, net @6bps 44.9%, −6.42 bps/trade, t=−4.77. Fixed in `purged_wf_tbm.py`; WF re-run. Both sides are net-negative.

### Critical Finding: no post-cost edge in either direction

With correct costs, both sides have raw WR ~50% (coin flip) and lose the 6 bps round-trip; selection skill is ~zero (short +0.63 pp, long +1.22 pp). At 1h scale, direction is not predictable from lagging price/volume features in this universe. Long was additionally tested across 6 approaches (drift removal, conviction tightening, View D momentum/breakout, daily-trend conditioning, a 6-config barrier-geometry scan, and a full retrain at the most favorable geometry) — all confirm no selectable long signal. The user's "smaller target / longer horizon" insight was mechanically correct (cut timeouts 86%→23%, lifted base rate 42%→46%) but did not create predictability.

### Next Steps

1. **Do NOT keep tuning TBM-on-price/volume** — it is exhausted. Any 1h directional edge needs new data (order-flow/L2, options flow, news/sentiment, event calendar).
2. **Process control:** always verify `median(net − gross) == −cost` per side and check RAW vs NET WR before trusting a headline. This bug was caught only by asking "what is the raw win-rate?"
3. ~~**Re-audit v8/v10/15m**~~ — **v8 done (2026-06-10), KILLED.** v10/15m still pending.

---

## 🔻 v8 1-Hour Walk-Forward Reanalysis (2026-06-10) — DEMOTED

The saved `v8_upstox_3y` artifact was evaluated on a single 80/20 chronological split where the **test set doubled as the early-stopping validation set** (`evals=[(dtrain,'train'),(dtest,'test')]` in `train_ranking_upstox.py`) — a model-selection leak. `scripts/analysis/v8_walkforward.py` retrained v8's exact architecture (rank:pairwise, depth=5, identical 86 features/hyperparams) in a genuine 9-fold rolling walk-forward (separate validation month, train-only NaN fills) on `ranking_data_upstox_1h_v3_3y.csv` — 320,931 OOS rows spanning 2023-08 → 2026-05.

### Fold-by-fold Spearman (genuinely OOS)

| Fold | Test period | Long ρ | Short ρ |
|---|---|---|---|
| 1 | 2023-08/09 | +0.0438 | +0.0372 |
| 2 | 2023-12/2024-01 | +0.0339 | +0.0357 |
| 3 | 2024-04/05 | +0.0237 | +0.0317 |
| 4 | 2024-08/09 | +0.0301 | +0.0201 |
| 5 | 2024-12/2025-01 | +0.0288 | +0.0288 |
| 6 | 2025-04/05 | +0.0254 | +0.0229 |
| 7 | 2025-08/09 | +0.0182 | +0.0227 |
| 8 | 2025-12/2026-01 | +0.0197 | +0.0176 |
| 9 | **2026-04/05** | **+0.0116** | **+0.0035** |

Average ρ ≈ **0.026 / 0.024** — about half the static metadata's `0.0461/0.0490` — and **monotonically decaying**, with the most recent fold near zero.

### Top-K Returns (genuinely OOS, full span)

| | n | Raw bps | Net @6bps | Raw Win | Net Win | t-stat |
|---|---|---|---|---|---|---|
| Top-1 Long | 1,867 | +2.90 | **−3.10** | 51.7% | 45.6% | −2.00 |
| Top-1 Short | 1,867 | +1.71 | **−4.29** | 55.1% | 50.0% | −1.87 |
| Top-3 Long | 5,601 | +2.40 | **−3.60** | 50.6% | 44.5% | −4.17 |
| Top-3 Short | 5,601 | +2.28 | **−3.72** | 53.6% | 48.7% | −3.11 |

Cost-sign explicitly asserted correct (`raw − net == cost`) — **this is not a TBM-style cost bug**, the raw edge itself is too thin. `eval_report.json`'s claimed +33bps/+25bps Top-1/Top-3 came from the same leaky single split.

### Last 12 months (2025-07+) — edge essentially gone

Top-1 Long raw **+0.31bps**, Top-3 Long raw **−0.05bps** (negative). All net-negative at 6bps.

### Time-of-day breakdown (Top-3, full OOS, @6bps)

All 5 session bars (09:15–13:15 in this dataset version) are net-negative for both Long and Short; best is 13:15 (Long −1.10bps, Short −1.36bps), still negative. **No surviving alpha pocket** — note this dataset's session ends at 13:15, so the previously-cited "14:30 IST" pocket (from `ranking_data_upstox_3y.csv`'s :30-aligned bars) cannot be directly re-checked here and should be treated as unverified.

> [!ERROR] **v8 has no deployable post-cost edge under genuine walk-forward, and its raw skill is decaying toward zero through 2025-2026.** It should no longer be treated as the "Active Production Champion." This mirrors the TBM and v18/v19 conclusions: single-split / leaky-early-stopping evaluations on this feature set systematically overstate edge.

### Corroboration: static 70/10/20 split (same test period as original, val≠test)

To rule out the rolling walk-forward harness itself as the cause, `scripts/analysis/v8_static_70_10_20.py` reproduces v8's original "one big split" regime — 38 months train (2022-01..2025-02), 5 months val (2025-03..2025-07) for early stopping, 11 months test (2025-08..2026-06, ~the same window as the original's 1,293-query test set) — but with **val ≠ test** (fixing the leak).

| | Long | Short |
|---|---|---|
| Test Spearman | **+0.0212** | **+0.0192** |
| best_iteration | 37 | 13 |

| Top-K @6bps | n | Raw bps | Net bps | Raw Win | t-stat |
|---|---|---|---|---|---|
| Top-1 Long | 1,030 | +1.00 | **−5.00** | 52.1% | −2.21 |
| Top-1 Short | 1,030 | +9.22 | **+3.22** (n.s.) | 57.8% | +1.25 |
| Top-3 Long | 3,090 | +2.49 | **−3.51** | 51.9% | −3.01 |
| Top-3 Short | 3,090 | +2.80 | **−3.20** | 54.3% | −2.16 |

Same answer as the rolling WF (~0.02 Spearman, net-negative/insignificant). With a genuinely held-out validation set, `best_iteration` drops to 37/13 (vs. 500-round budget) — the original's much higher reported Spearman came from selecting `best_iteration` against the test set itself. **This is not a walk-forward-harness artifact.**

### Next Steps
1. Re-audit `v10_native_1h` and `v2_15min_3y` with the same purged walk-forward methodology (same class of leak likely present — both were trained with the test set as eval set).
2. Do not deploy v8 as the live ranker until a from-scratch architecture with genuine walk-forward validation clears costs.
3. Reproducibility scripts: `scripts/analysis/v8_walkforward.py` (rolling WF), `scripts/analysis/v8_static_70_10_20.py` (static split corroboration). Outputs in `data/model_analysis/v8_walkforward/`.

---

## 🔍 Raw-Signal-Only Audit (2026-06-10): v8 vs v10-depth4 — "Is there *anything* to filter on?"

Setting aside net-of-cost profitability, both v8 (depth-5) and v10 (depth-4) were checked for **raw cross-sectional ranking skill** — i.e., is the predicted score genuinely correlated with next-hour return, regardless of whether 6bps costs survive? Computed directly from `data/model_analysis/v10_v18_independent/walkforward_preds.npz` (same 9-fold genuine WF, 320,931 OOS rows, 2023-08→2026-05).

| Fold | Test period | v10-d4 Long ρ | v10-d4 Short ρ |
|---|---|---|---|
| 1 | 2023-08/09 | +0.0446 | +0.0377 |
| 2 | 2023-12/2024-01 | +0.0378 | +0.0402 |
| 3 | 2024-04/05 | +0.0257 | +0.0301 |
| 4 | 2024-08/09 | +0.0290 | +0.0180 |
| 5 | 2024-12/2025-01 | +0.0274 | +0.0326 |
| 6 | 2025-04/05 | +0.0250 | +0.0188 |
| 7 | 2025-08/09 | +0.0181 | +0.0231 |
| 8 | 2025-12/2026-01 | +0.0255 | +0.0206 |
| 9 | **2026-04/05** | **+0.0129** | **+0.0049** |

Average ρ = **+0.0273 / +0.0251** (essentially identical to v8's +0.026/+0.024), and a paired t-test of the 9 fold-rhos against zero is **highly significant**: Long t=8.67 (p<0.0001), Short t=6.77 (p=0.0001). Same monotonic decay pattern as v8.

**Raw Top-3 win-rate significance (z-test vs 50%, n=5601):**
| | v8 raw win | v8 z | v10-d4 raw win | v10-d4 z |
|---|---|---|---|---|
| Long | 50.6% | 0.9 (n.s.) | 51.6% | **2.4** (p<0.05) |
| Short | 53.6% | **5.4** (p<0.0001) | 54.9% | **7.3** (p<0.0001) |

In the last 12 months (2025-07+), v10-d4's **short-side raw signal still holds** (raw win 54.4%, z=3.8) while long-side has decayed to noise (50.2%, z=0.1).

**Conclusion**: v8 and v10 are the **same underlying signal family** (depth 5 vs 4 on identical features/data) with genuinely real but weak (~ρ=0.025), decaying raw ranking skill — strongest and most persistent on the **short side**. Neither survives 6bps costs as a standalone Top-K strategy, but the short-side score is statistically real enough to be useful as a **filter/feature input** (e.g., a confirmation gate for a lower-cost execution venue, or one input into a meta-ensemble) rather than a standalone strategy. TBM and v18/v19 have **no** such signal (raw win ~50%, AUC~0.50) and are not useful even as filters. `v2_15min_3y` and `v1_30min` still need this same raw-signal-only check — their claimed Spearman (0.0608/0.0610 and 0.0449/0.0433) came from the same kind of evaluation that overstated v8 by ~2x.

---

## 👁️ Key Related Notes
*   See where these model files reside: [[02. Model Suite/Model Registry & File Structures|Model Registry & File Structures]].
*   See the multi-timeframe scanner boundary logic: [[02. Model Suite/Multi-Timeframe Models|Multi-Timeframe Models]].
*   Review our May 2026 backtest simulation returns: [[03. Trading Strategies/Strategy Catalog|Strategy Catalog]].
*   Full TBM fold-by-fold results: [[08. Model Analysis/1-Hour Vanguard Model/TBM-1h-Ensemble-Results]].
