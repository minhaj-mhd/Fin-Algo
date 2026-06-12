# Time of Day & Residual Analysis: v2_15min_3y

**Date:** June 7, 2026
**Subject:** Rank error distribution, predicted-vs-actual rank alignment, and hour-of-day calibration quality for the 15-minute model.
**Dataset:** OOS Apr–Jun 2026 · 180,940 rows · 1,061 queries.

---

## Visual Summary

![[assets/08_residual_analysis.png]]

---

## 1. Residual Definition

For each 15-min query (timestamp), each stock receives:
- **Predicted rank percentile** = `rank(long_score) / N` (normalized to 0–1)
- **True rank percentile** = `rank(Next_15Min_Return) / N`
- **Residual** = Predicted rank percentile − True rank percentile

A residual of 0 means the model ranked the stock perfectly. Positive residuals = model predicted a higher rank than the stock deserved (over-rated). Negative residuals = model predicted a lower rank than the stock deserved (under-rated).

---

## 2. Residual Distribution

### Long Model Residuals
- **Shape:** Near-symmetric bell curve centered close to 0.
- **Mean:** ≈ 0.000 (unbiased — no systematic over- or under-rating)
- **Std:** ≈ 0.35–0.40 (residuals spread across ~35–40% of the rank range)
- **Normal fit:** The distribution closely follows a fitted normal distribution, confirming no heavy tails or systematic bias.

### Short Model Residuals
- **Shape:** Also near-symmetric, centered at ≈ 0.000
- **Std:** Similar to Long model (≈ 0.35–0.40)
- **Independence:** The Short model's residuals are structurally similar to the Long model's, confirming both models make errors of similar magnitude in opposite directions (as expected from a dual-model architecture).

> [!NOTE]
> A residual std of ~0.38 in a uniform rank space means the model's per-stock rank is typically off by ±38 percentile points. This is expected and acceptable for a **ranker** (not a classifier). The value add comes from aggregate cross-sectional ordering, not from individual prediction precision.

---

## 3. Predicted Rank vs True Rank Scatter (Long Model)

A scatter plot of 200 queries × ~15 stocks each (≈3,000 sample points) shows:
- Positive Pearson correlation r ≈ 0.10–0.15 between predicted and true rank percentile.
- The scatter is wide and noisy at the individual level — this is expected.
- The aggregate (binned) trend confirms monotonic alignment: stocks predicted near rank 1.0 land materially higher on average than stocks predicted near rank 0.0.
- **This is the core principle:** Rankers do not need high per-stock accuracy — they need reliable sorting at the aggregate level.

---

## 4. Mean Absolute Rank Error by Hour (IST)

The residual analysis reveals that the model's calibration quality varies systematically by hour of day.

| Hour (IST) | Approximate MAE | Interpretation |
|---|---|---|
| **09:00 (Open)** | Low | Opening-hour microstructure is highly predictable — IBS + Buy_Pressure signals are strong |
| 10:00 | Low-Medium | Continuation of opening regime |
| 11:00–12:00 | Medium | Midday consolidation — noisier price action |
| **13:00** | Medium | Post-lunch re-pricing noise |
| 14:00 | Medium-Low | Pre-close momentum beginning to emerge |
| **15:00 (Close)** | Lowest | EOD institutional book-squaring is highly patterned — model's best hour |

**Color coding in the plot:** Green bars = below-median MAE (model is sharper here); Red bars = above-median MAE (model is noisier here).
**Mean global MAE:** ~0.28–0.32 (28–32% of rank range per bar).

### Key Observations

1. **Opening Hour (09:15–10:15) is the sharpest window.** The extreme IBS and Buy_Pressure values at the open — driven by overnight news and gap-open dynamics — give the model cleaner signals than any other period. Mean-reversion from gap-open bars is the model's most reliable prediction.

2. **Midday (11:30–13:15) has the highest rank error.** Choppy, low-volume consolidation candles generate ambiguous IBS and Buy_Pressure readings. The model's confidence is lowest here.

3. **EOD (15:00–15:30) is the second sharpest window.** Time-to-close becomes a dominant feature (Is_Close_Hour = 1), and institutional positioning creates directional clarity.

4. **This creates a natural deployment schedule:** The 15-min model's predictions should carry the most weight at the open and close, with reduced position sizing in the midday chop window.

---

## 5. Structural Comparison vs 30-Minute Model

| Metric | 15-Min Model | 30-Min Model |
|---|---|---|
| Primary alpha hour (Long) | Open (09:15) and EOD (15:00) | EOD only (15:15) |
| Primary alpha hour (Short) | Open + afternoon | 14:15 only (narrow) |
| Midday noise | High | High (same) |
| Short model standalone | Valid signal (57% WR) | Broken standalone |
| Residual distribution | Near-normal | Similar |

The 15-min model has more evenly distributed alpha across the trading day than the 30-min model, which is almost entirely concentrated in a single EOD slot. This makes the 15-min model more suitable for intraday execution across multiple sessions.

---

## 6. Recommended Execution Guidelines Based on Residual Analysis

Based on the hour-of-day MAE profile:

| Window | Model Confidence | Recommended Action |
|---|---|---|
| 09:15–10:15 | **High** | Deploy at full size. Opening bars have the cleanest IBS + Buy_Pressure signals. |
| 10:30–11:30 | Medium-High | Standard deployment. |
| 11:45–13:00 | **Low** | Reduce position size or skip entirely. Midday noise inflates rank errors. |
| 13:15–14:15 | Medium | Moderate deployment — pre-close momentum is building. |
| 14:30–15:15 | Medium-High | Standard to full deployment. |
| **15:15–15:30** | **High** | Deploy at full size. EOD book-squaring creates strongest directional signals. |

---

## 7. Backlinks

- [[Complete Edge Catalog]] — Per-month OOS and walk-forward folds.
- [[OOS Calibration & Thresholds]] — Score-to-return calibration and threshold tables.
- [[Feature Analysis & SHAP]] — Why opening-hour IBS and Buy_Pressure are the strongest signals.
- [[Model Diagnostics & Visualizations]] — Residual plot and hour-of-day MAE chart.
