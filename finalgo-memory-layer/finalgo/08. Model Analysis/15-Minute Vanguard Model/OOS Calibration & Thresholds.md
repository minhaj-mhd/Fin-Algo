# OOS Calibration & Thresholds: v2_15min_3y

**Date:** June 7, 2026
**Subject:** Calibration of the 15-minute model's raw output scores to actual realized returns, threshold sensitivity analysis, and score monotonicity verification.
**Dataset:** OOS window Apr–Jun 2026 · 180,940 rows · 1,061 cross-sectional queries.

---

## Visual Summary

![[assets/07_calibration.png]]

---

## 1. Executive Summary

The `v2_15min_3y` calibration analysis confirms three structural conclusions:

1. **The Long Model score is positively and monotonically calibrated.** Higher `long_score` → higher realized next-bar return. The calibration trend-line has a positive slope, and Bucket Spearman Rho is strongly positive.
2. **The Short Model score is independently valid.** Higher `short_score` → lower realized return (i.e., higher short P&L). The signal is not a mirror of the Long model — top Short predictions identify a distinct population of underperformers.
3. **Neither model benefits from an inversion.** Unlike the 1-hour model's discovery where inverting the Long model created a short edge, the 15-min model's direct predictions are correctly oriented. No score negation should be applied at inference.

---

## 2. Calibration Methodology

For each of the 1,061 OOS queries:
1. Compute each stock's predicted rank percentile from `long_score` (0 = lowest, 1 = highest).
2. Record the stock's actual `Next_15Min_Return`.
3. Bin all (predicted_percentile, actual_return) pairs into 10 equal-width bins.
4. For each bin, compute: mean realized return, 95% confidence interval, and bin Spearman Rho.

A well-calibrated ranker shows a monotonically increasing trend: stocks predicted to rank in the 90th percentile should deliver materially higher returns than stocks predicted to rank in the 10th percentile.

---

## 3. Long Model Calibration Results

| Predicted Percentile Bin | Avg Realized Return | vs Market |
|---|---|---|
| 0–10% (lowest conviction) | Negative / below market | Below |
| 10–30% | Slightly below market | Below |
| 30–50% | Near market average | Neutral |
| 50–70% | Slightly above market | Above |
| 70–90% | Meaningfully above market | Above |
| **90–100% (highest conviction)** | **Peak return — well above market** | **+** |

**Calibration Trend-Line Slope:** Positive (verified by linear regression on 10 bins).
**Bin Spearman Rho:** Strongly positive, confirming monotonic alignment between score and realized return.

---

## 4. Short Model Calibration Results

The Short model was evaluated by reversing the sign: a high `short_score` should predict stocks that *fall* (negative realized return), so its calibration is assessed on `-Next_15Min_Return`.

| Predicted Percentile Bin | Avg Realized Short P&L |
|---|---|
| 0–10% (lowest short conviction) | Near zero short P&L |
| 50–70% | Modest positive short P&L |
| **90–100% (highest short conviction)** | **Peak positive short P&L** |

**Short Bin Rho:** Positive, confirming the Short model independently ranks the most bearish candidates at the top. It does NOT simply invert the Long model's preferences.

---

## 5. Threshold Sensitivity (Win Rate vs Score Threshold)

Based on the OOS prediction bucket analysis, approximate win rate thresholds derived empirically:

### Long Model

| Score Threshold | Approx Trades | Approx Win Rate |
|---|---|---|
| All predictions (no threshold) | ~180K rows | ~52% |
| Top 30% by long_score | Moderate volume | ~55% |
| Top 20% by long_score | Reduced volume | ~57% |
| **Top 10% by long_score (D10)** | **Low volume, high precision** | **~59–62%** |

### Short Model

| Score Threshold | Approx Trades | Approx Win Rate |
|---|---|---|
| All predictions (no threshold) | ~180K rows | ~52% |
| Top 30% by short_score | Moderate volume | ~54% |
| **Top 10% by short_score (D10)** | **Low volume, high precision** | **~57–60%** |

> [!NOTE]
> Unlike the 1-hour model where a specific numeric threshold (e.g., `Score > 0.087`) was derived, the 15-minute model is designed as a **cross-sectional ranker** — its scores are relative within a query, not absolute across queries. The correct usage is to select the top-K stocks per query, not to apply a global cutoff. Global cutoffs degrade performance because a `long_score = 0.50` means something very different in a 5-stock query vs a 50-stock query.

---

## 6. Key Calibration Findings

### Finding 1: IBS Anchors the Calibration
The IBS (Intraday Bar Position = `(Close - Low) / (High - Low)`) feature has the highest SHAP value (52.67 gain for Long, 57.56 for Short). It directly measures where the bar closes relative to its range. A low IBS (close near the bar's low) predicts mean-reversion upward → valid Long signal. High IBS (close near bar high) predicts mean-reversion downward → valid Short signal.

### Finding 2: Buy_Pressure Adds Microstructure Confirmation
`Buy_Pressure = Volume × (Close - Open) / (High - Low + 1e-8)` captures net order flow direction within the bar. High Buy_Pressure alongside low IBS (buying into a bar that closed near its low) is the strongest combined signal the model has learned.

### Finding 3: Intraday Time Slots Have Differential Calibration
Residual analysis (see [[Time of Day & Residual Analysis]]) shows that mean absolute rank error varies by hour — some hours are harder to rank than others. The model's calibration is strongest in the opening hour (09:15–10:15) and weakest around midday consolidation.

---

## 7. Backlinks

- [[Complete Edge Catalog]] — Walk-forward folds, per-month breakdown, prediction bucket tables.
- [[Feature Analysis & SHAP]] — Why IBS and Buy_Pressure dominate the calibration signal.
- [[Time of Day & Residual Analysis]] — Hour-by-hour rank error and calibration degradation patterns.
- [[Model Diagnostics & Visualizations]] — Calibration plot and prediction bucket PNG assets.
