# Quality Metrics Summary

## Current Model Performance

### 1. Ranking Accuracy Metrics
```
Metric                  Value    Expected    Status
────────────────────────────────────────────────────
Spearman Correlation    -0.6354  +0.3 to +0.5    POOR
Kendall Tau             -0.4795  +0.2 to +0.4    POOR
Top-1 Accuracy          0.00%    15-25%          CRITICAL
Top-3 Overlap           0.92%    20-30%          CRITICAL
Top-5 Overlap           1.75%    30-40%          CRITICAL
```

### 2. Profitability Metrics
```
Strategy                Return     Win Rate    Status
─────────────────────────────────────────────
Top-1 Prediction        -0.4327%   12.00%      LOSING
Top-3 Prediction        -0.3798%   10.75%      LOSING
Top-5 Prediction        -0.3454%   11.50%      LOSING
Contrarian (Worst-1)    +0.3084%   88.00%      WINNING!
```

### 3. Regression Accuracy Metrics
```
Metric          Value      Interpretation
────────────────────────────────────────────────
MAE             0.0965     On average, off by 9.65 basis points
RMSE            0.1234     Typical error magnitude
MAPE            inf%       Many near-zero returns cause division by zero
```

### 4. Feature Quality Analysis
```
Feature Category        #Features   Top Feature         Importance
─────────────────────────────────────────────────────────────────────
Volume-Based            5           Volume_Zscore       213
Momentum                 8           ROC_12              166
Trend                    7           SMA_6               ~100
Volatility               7           Bollinger/Keltner   ~150
Temporal                 2           Hour                ~50

Total Features: 48
Dominant Pattern: VOLUME (very high importance scores)
```

### 5. Prediction Distribution
```
Statistic           Value
─────────────────────────────
Min Score           -0.6456
Q1 (25%)            -0.0870
Median (50%)        -0.0088
Q3 (75%)            +0.0671
Max Score           +0.5447
Mean                -0.0112
Std Dev             0.1207
Range               1.2103

Interpretation: Predictions are mostly negative (mean -0.0112)
```

### 6. Per-Query Variance Analysis
```
Query Variance              Range
─────────────────────────────────
Mean Variance               0.0143
Minimum                     0.0047
Maximum                     0.0398
Ratio (Max/Min)             8.5x

Interpretation: Some queries have very clear signals (0.0398),
others are almost random (0.0047). Model not consistent.
```

### 7. Error Breakdown by Label Type
```
Return Type             Mean Error          Std Error       Bias
────────────────────────────────────────────────────────────────
Positive Return         +0.071938           0.105105        Overestimate
Negative Return         -0.048797           0.108900        Underestimate
Overall Bias:           ASYMMETRIC (differs by return sign)
```

---

## Critical Issues Found

### Issue #1: INVERSE LEARNING (CRITICAL)
**Symptom**: Negative correlation between predictions and actual returns
- Spearman: -0.6354
- Kendall Tau: -0.4795
- Contrarian strategy (+0.31%) beats normal strategy (-0.38%)

**Impact**: Model recommends stocks that FALL instead of those that RISE
**Severity**: 🔴 CRITICAL - Makes model harmful if used directly

**Testing Results**:
```
Inverted Predictions:
- Spearman: +0.5521 (positive!)
- Top-1 Accuracy: 9.25% (vs 0%)
- Top-3 Return: +0.2934% (vs -0.3798%)
- Improvement: +0.67% on top-3 strategy
```

**Recommended Fix**: Use inverted predictions OR verify temporal alignment

---

### Issue #2: POOR TOP-K ACCURACY (CRITICAL)
**Symptom**: Cannot identify best performers
- Top-1 Accuracy: 0% (should be ~2% for random)
- Top-3 Overlap: 0.92% (should be ~6% for random)
- Top-5 Overlap: 1.75% (should be ~10% for random)

**Impact**: Model worse than random at identifying winners
**Severity**: 🔴 CRITICAL - Ranking usefulness is near zero

**Root Cause**: 
- Low score variance (mean 0.0143)
- Cannot differentiate between stocks within each query
- High noise in hourly returns

**Recommended Fix**: Add relative features, improve data quality, tune hyperparameters

---

### Issue #3: HIGH VARIANCE ACROSS QUERIES (HIGH)
**Symptom**: Predictions inconsistent between queries
- Variance range: 0.0047 to 0.0398 (8.5x spread)
- Std Dev of Spearman per query: 0.3027

**Impact**: 
- Works well in some hours, fails in others
- Cannot trust model consistently
- May need time-dependent models

**Severity**: 🟠 HIGH - Reduces reliability

**Recommended Fix**: Add market regime detection, increase regularization

---

### Issue #4: ASYMMETRIC ERRORS (MEDIUM)
**Symptom**: Different error patterns for positive vs negative returns
- Positive return errors: +0.0719 (overestimate returns)
- Negative return errors: -0.0488 (underestimate losses)

**Impact**: 
- Biased predictions
- May recommend wrong stocks in different market conditions

**Severity**: 🟡 MEDIUM - Affects trading decisions

**Recommended Fix**: Adjust loss weights by return sign, try different target

---

### Issue #5: WEAK SIGNAL STRENGTH (HIGH)
**Symptom**: Hourly returns near zero make prediction hard
- Mean return: 0.000078
- Std Dev: 0.004567
- Signal/Noise ratio very low

**Impact**:
- Even perfect model would have ~50% accuracy
- Noise dominates the signal
- May need longer timeframes

**Severity**: 🟠 HIGH - Inherent data limitation

**Recommended Fix**: Focus on relative ranking (easier), use longer timeframes

---

## Metrics Scorecard

| Category | Metric | Score | Grade | Status |
|----------|--------|-------|-------|--------|
| Ranking | Top-1 Accuracy | 0% | F | 🔴 FAIL |
| Ranking | Top-3 Overlap | 0.92% | F | 🔴 FAIL |
| Ranking | Spearman | -0.6354 | D | 🔴 FAIL |
| Profitability | Top-3 Return | -0.38% | D | 🔴 LOSE |
| Regression | MAE | 0.0965 | C | 🟡 OK |
| Features | Importance | Concentrated | D | 🟡 OK |
| Stability | Query Variance | 8.5x spread | D | 🟡 POOR |
| Learning | Pattern Detection | Inverse | D | 🔴 WRONG |
| Data | Signal Strength | Very Weak | E | 🟡 HARD |

**Overall Grade: F (Failing)**
- Model learns patterns but in wrong direction
- Cannot rank stocks better than random
- Loses money if used as-is

---

## Quality Assessment Matrix

```
Dimension           Current    Target    Gap    Priority
───────────────────────────────────────────────────────────
Ranking Accuracy    0-1.75%    20-30%    -20%   🔴 CRITICAL
Return/Profitability -0.38%    +1.0%     +1.38% 🔴 CRITICAL
Prediction Variance 0.0143     0.03+     +0.02  🟠 HIGH
Model Consistency   0.3 Std    0.1 Std   8x     🟠 HIGH
Data Quality        Noisy      Clean     High   🟠 HIGH
Feature Engineering Limited    Advanced  High   🟡 MEDIUM
Hyperparameter Tune Basic       Optimized High   🟡 MEDIUM
```

---

## Quick Fix Impact Estimates

| Fix | Implementation | Effort | Expected Gain | Timeline |
|-----|----------------|--------|---------------|----------|
| Invert Predictions | 1 line code | 5 min | +0.67% | Now |
| Verify Data Alignment | Check logic | 15 min | +0-2% | Today |
| Add Regularization | Update params | 30 min | +0.5% | Today |
| Add Relative Features | Feature eng | 2 hours | +0.5-1.0% | Today |
| Clean Data (filter noise) | Remove outliers | 1 hour | +0.3-0.5% | Today |
| Try Different Target | Re-train | 2 hours | +0.5-1.0% | Today |
| Full Hyperparameter Tune | Grid search | 4 hours | +1.0-2.0% | Tomorrow |

**Quick Wins (within 1 hour)**: Inversion + Regularization = +1.0%+

---

## Recommendations Priority List

### 🔴 MUST DO (This Week)
1. Verify temporal alignment of features and labels
2. Test inverted predictions
3. Add minimum regularization (gamma, min_child_weight)
4. Evaluate impact

### 🟠 SHOULD DO (This Week)
1. Add relative features (vs peer average)
2. Remove redundant volume features
3. Clean data (remove illiquid/extreme periods)
4. Compare target types (direction, quintile, rank)

### 🟡 NICE TO DO (Next Week)
1. Implement market regime detection
2. Try NDCG and LambdaMART objectives
3. Add market microstructure features
4. Create ensemble model

### ⚪ FUTURE (Later)
1. Switch to deep learning (LSTM/GRU)
2. Implement reinforcement learning
3. Add sentiment data
4. Integrate with other data sources

---

## Success Criteria

After implementing fixes:

| Metric | Current | Target | Status |
|--------|---------|--------|--------|
| Top-3 Accuracy | 0.92% | >20% | ✓ Achievable |
| Profitability | -0.38% | >+0.50% | ✓ Achievable |
| Spearman Correlation | -0.6354 | >+0.4 | ✓ Achievable |
| Consistency (Std Dev) | 0.3027 | <0.15 | ✓ Achievable |

**Expected Timeline**: 2-3 weeks with focused effort
**Expected ROI**: 2-5x improvement in profitability

---

## Files Generated
- `evaluate_model.py` - Full evaluation script
- `test_inversion.py` - Hypothesis testing
- `improvement_strategy.md` - Detailed improvement plan
- `EVALUATION_REPORT.md` - This report
- `eval_report.json` - Machine-readable results
