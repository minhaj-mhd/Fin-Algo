# 📊 Model Performance & Statistics

This document serves as the absolute technical ledger for all machine learning models in the **Vanguard High-Precision Intraday Trading Engine**. It details their structural setups, training parameters, out-of-sample Spearman correlations, feature dimensions, and historical simulation performance.

---

## 📈 Centralized Model Comparison Matrix

The Vanguard engine supports multiple model specialities. The matrix below contrasts all registered and active XGBoost ranking models:

| Model ID | Timeframe | Training Horizon | Features | Scaling Method | Long Spearman Rho | Short Spearman Rho | Precision @ 3 (Win Rate) | Primary Operational Role |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`daily_xgb`** | Daily | 5 Years (Upstox) | 165 | Scale-Invariant (Pass-Through) | `0.0129` | `0.0217` | Long: `53.5%`<br>Short: `52.5%` | **Macro Universe Gatekeeper** (Filters top 40% trends) |
| **`v8_upstox_3y`** | 1-Hour | 3 Years (Upstox) | 86 | StandardScaler (Pass-Through) | `0.0461` | `0.0490` | *N/A* | **Active Core Intraday Specialist** (Active production model) |
| **`v4_regime`** | 1-Hour | 2 Years (yfinance) | 86 | **Scale-Free** (No Scaler) | **`0.0820`** | **`0.0850`** | *N/A* | Regime-Aware Multi-TF Specialist (Offline Benchmark) |
| **`v3_sector`** | 1-Hour | 2 Years (yfinance) | 95 | StandardScaler | `0.0558` | `0.0574` | Long: `55.4%`<br>Short: `57.2%` | Sector Momentum & VWAP Explorer (Archive) |
| **`v1_15min`** | 15-Min | 1 Year (Upstox) | 95 | Scale-Invariant (Pass-Through) | `0.0571` | `0.0558` | Long: **`58.9%`**<br>Short: **`57.4%`** | Execution-level Entry Sniper (Standalone backtest candidate) |
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
*   **Evaluation Method**: 4-Fold Walk-Forward Temporal Validation (splits by calendar month to prevent data leakage).
*   **Target Label**: Next-day cross-sectional return ranks.
*   **Walk-Forward Performance Summary (Top 3 selections)**:
    *   **Average Long Spearman Rho**: `0.0129`
    *   **Average Short Spearman Rho**: `0.0217`
    *   **Average Top-3 Long Return**: `+0.0639%` per day over market average
    *   **Average Top-3 Short Return**: `+0.1635%` per day over market average
    *   **Average Combined Edge**: **`+0.2274%`** daily return edge
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

### 4. 15-Minute Specialist (`v1_15min`)
Our highest resolution timeframe model, fitted on high-frequency microstructure indicators.

*   **Training Dataset Size**: 1,033,467 rows of 15-minute bars spanning 1 year.
*   **Evaluation Method**: 3-Fold Walk-Forward Temporal Validation.
*   **Walk-Forward Performance Summary**:
    *   **Average Long Spearman Rho**: `0.0571`
    *   **Average Short Spearman Rho**: `0.0558`
    *   **Average Long Win Rate @ K=3**: **`58.9%`** (Beats cross-sectional median)
    *   **Average Short Win Rate @ K=3**: **`57.4%`** (Falls below cross-sectional median)
    *   **Combined Edge**: **`+0.1127%`** per 15-min bar.
*   **Production Model Setup**:
    *   **Train Set**: May 2025 to April 2026.
    *   **Validation Set**: May 2026.
    *   **Best Iteration**: Long: `20` | Short: `59`.
*   **Top Predictor Features (Long)**:
    1.  `IBS` (Gain: `17.26`): Strong mean-reversion signal.
    2.  `Buy_Pressure` (Gain: `9.76`): Immediate liquidity imbalance.
    3.  `Lower_Shadow` (Gain: `4.18`): Rejection of low prices.
*   **Top Predictor Features (Short)**:
    1.  `IBS` (Gain: `13.89`): High proximity to the ceiling.
    2.  `OC_Range` (Gain: `11.28`): Large candle bodies indicate high expansion.
    3.  `Log_Return` (Gain: `8.21`): Immediate downward acceleration.

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

## 👁️ Key Related Notes
*   See where these model files reside: [[02. Model Suite/Model Registry & File Structures|Model Registry & File Structures]].
*   See the multi-timeframe scanner boundary logic: [[02. Model Suite/Multi-Timeframe Models|Multi-Timeframe Models]].
*   Review our May 2026 backtest simulation returns: [[03. Trading Strategies/Strategy Catalog|Strategy Catalog]].
