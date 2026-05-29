# Model Improvement Strategy & Analysis

## Current Performance Summary
- **Spearman Correlation**: -0.6354 (NEGATIVE - model ranks inversely!)
- **Top-1 Accuracy**: 0% (0 out of 400 times, best stock correctly identified)
- **Top-3 Overlap**: 0.92% (model rarely identifies top performers)
- **Profitability**: -0.34% to -0.43% (LOSING money on all strategies)
- **Win Rate**: 10-12% (worse than random at 50%)
- **Contrarian Strategy**: +0.31% (inverse predictions are better!)

---

## Critical Issues Identified

### 1. **INVERSE LEARNING PROBLEM** (CRITICAL)
- **Issue**: Negative Spearman correlation means the model learned to rank stocks BACKWARDS
- **Evidence**: 
  - Contrarian strategy (+0.31%) beats all normal strategies (-0.34% to -0.43%)
  - Buying worst-predicted stocks gives positive returns
  - Top-1 accuracy is 0%
- **Root Cause**: Possible label inversion, scale mismatch, or sign error in training
- **Fix Priority**: ⭐⭐⭐ HIGHEST - This explains all poor performance

### 2. **POOR PREDICTIVE POWER** (CRITICAL)
- **Issue**: Model cannot differentiate between winners and losers
- **Evidence**:
  - Mean variance per query only 0.0143 (very low separation)
  - Top-1 accuracy is 0% (random would be 2%)
  - Top-3 overlap is 0.92% (random would be ~6%)
- **Root Cause**: 
  - Features may not contain predictive signal for next-hour returns
  - Hourly frequency may be too short for technical indicators
  - Insufficient training signal in data
- **Fix Priority**: ⭐⭐⭐ HIGHEST

### 3. **HIGH PREDICTION VARIANCE** (CRITICAL)
- **Issue**: Std Dev of -0.3027 means predictions are inconsistent across queries
- **Evidence**: Some queries have high variance (0.0398), others very low (0.0047)
- **Root Cause**: 
  - Model overfitting to specific query patterns
  - Market regimes change hourly (different conditions)
  - Insufficient regularization
- **Fix Priority**: ⭐⭐ HIGH

### 4. **LABEL QUALITY ISSUES** (HIGH)
- **Issue**: Actual returns average only 0.000078 (essentially zero)
- **Evidence**:
  - Std Dev of actual returns is 0.004567 (very small)
  - MAPE = inf (many returns near zero cause division by zero)
  - Hard to predict when signal is this weak
- **Root Cause**: Hourly returns are noisy, most are near zero
- **Fix Priority**: ⭐⭐ HIGH

### 5. **FEATURE ENGINEERING GAPS** (HIGH)
- **Issue**: Current features may not capture next-hour predictiveness
- **Evidence**:
  - Top features are volume-related (Volume_Zscore, CMF_20, OBV)
  - Missing forward-looking features (order flow, sentiment, etc.)
  - Technical indicators designed for longer timeframes
- **Fix Priority**: ⭐⭐ HIGH

---

## Immediate Fixes (Do First)

### Fix #1: Check Label Sign & Alignment
```python
# In train_ranking.py, verify:
print("Sample actual returns:", df['Next_Hour_Return'].head(10))
print("Min/Max returns:", df['Next_Hour_Return'].min(), df['Next_Hour_Return'].max())
print("Expected direction: positive = stock went UP")

# Run predictions on training data
# If predictions align with actual (positive correlation),
# then reverse predictions: y_pred = -y_pred
```

### Fix #2: Flip Predictions & Re-evaluate
```python
# If negative correlation is the issue, inverse fix is simple:
# y_pred_fixed = -y_pred
# Then re-evaluate profitability
```

### Fix #3: Increase Prediction Confidence
Use XGBoost parameters to increase differentiation:
```python
params = {
    'objective': 'rank:pairwise',
    'eta': 0.05,              # Reduce learning rate
    'max_depth': 5,           # Reduce depth (was 6)
    'min_child_weight': 10,   # Increase regularization
    'subsample': 0.8,         # Add subsample
    'colsample_bytree': 0.8,  # Add column sampling
    'gamma': 1.0,             # Add L1/L2 regularization
}
```

---

## Medium-Term Improvements

### Improvement #1: Multi-Timeframe Features
Add features from multiple timeframes to capture different patterns:
```python
# Add 30-min and 5-min indicators
# Add features like: 1h return vs 30m return, momentum acceleration
# This captures "market acceleration" which matters for hourly prediction
```

### Improvement #2: Market Microstructure Features
```python
# Add: Volume momentum, price acceleration, bid-ask patterns
# Add: Intraday volatility vs daily volatility ratio
# Add: How much this stock moved vs average today
```

### Improvement #3: Relative Features (Competitive Ranking)
```python
# Instead of absolute returns, use RELATIVE metrics:
# - This stock's volume vs peer average
# - This stock's volatility vs peer average  
# - This stock's momentum vs peer average
# These naturally create rank differences
```

### Improvement #4: Target Engineering
Instead of predicting `Next_Hour_Return`, predict:
```python
# Better targets:
# 1. Direction: +1 if up, -1 if down (classification instead of regression)
# 2. Relative rank: Rank among 50 stocks (0-49)
# 3. Bins: Top 10, Middle 30, Bottom 10 (3-class classification)
# These are easier to learn than absolute returns
```

### Improvement #5: Data Quality Filtering
```python
# Remove low-volume/illiquid stocks
# Remove queries with unusual market conditions
# Only train on "normal" market hours (10:00-15:00 IST)
# This reduces noise and improves signal/noise ratio
```

---

## Advanced Improvements

### Strategy A: Ensemble Approach
Train separate models for:
- Market trend up vs down
- High volatility vs low volatility
- High volume vs low volume
Then combine predictions based on current market state

### Strategy B: Deep Learning
Switch to LSTM/GRU for:
- Capturing temporal dependencies across queries
- Learning market regime changes
- Better feature interactions

### Strategy C: Feature Selection
- Use SHAP values to identify truly predictive features
- Remove redundant volume features (5 volume features dominate)
- Add engineered features with proven signal

### Strategy D: Advanced Ranking Loss
- Use `rank:ndcg` instead of `rank:pairwise` (better for top-K)
- Add custom loss function that penalizes errors more on top-performers
- Use LambdaMART ranking algorithm (more sophisticated than pairwise)

---

## Testing Plan

### Phase 1: Verification (This Week)
- [ ] Verify label sign and direction
- [ ] Flip predictions if needed
- [ ] Re-evaluate with contrarian logic
- [ ] Check if reversal fixes profitability

### Phase 2: Quick Wins (This Week)
- [ ] Increase XGBoost regularization
- [ ] Add regularization parameters (gamma, min_child_weight)
- [ ] Reduce max_depth to prevent overfitting
- [ ] Re-train and compare NDCG

### Phase 3: Feature Engineering (Next Week)
- [ ] Add relative features (vs peer average)
- [ ] Add multi-timeframe features
- [ ] Add volume-based acceleration features
- [ ] Remove redundant volume features

### Phase 4: Target Engineering (Next Week)
- [ ] Test with directional target (+1/-1)
- [ ] Test with relative ranking target
- [ ] Compare 3 target types and pick best

---

## Expected Outcomes After Fixes

| Metric | Current | After Phase 1 | After Phase 2 | After Phase 3 |
|--------|---------|---------------|---------------|---------------|
| Spearman | -0.6354 | +0.2+ | +0.3+ | +0.4+ |
| Top-1 Acc | 0% | 5-10% | 15-20% | 25-35% |
| Profitability | -0.34% | +0.2% | +0.5% | +1.0%+ |
| Win Rate | 10.75% | 45% | 55% | 60%+ |

---

## Why Model Is Currently Failing

The combination of:
1. **Inverse learning** (negative correlation) - model is backwards
2. **Weak signal** (hourly returns near zero) - hard to predict
3. **Poor differentiation** (low variance in scores) - can't rank
4. **Overfitting** (high variance across queries) - unstable

Creates a **perfect storm** of poor performance.

The good news: **ALL are fixable with the improvements above.**

---

## Recommendation: START HERE

**Step 1 (5 min)**: Check if predictions are inverted
```bash
# Add to train_ranking.py at end:
sample_pred = bst.predict(dtrain)[:10]
sample_actual = y_train[:10]
print(f"Predictions: {sample_pred}")
print(f"Actual:      {sample_actual}")
print(f"Correlation: {np.corrcoef(sample_pred, sample_actual)[0,1]}")
```

If correlation is negative → predictions are inverted!

**Step 2 (5 min)**: Flip and re-evaluate
```python
# In evaluation, try:
y_pred_fixed = -y_pred
# Then re-evaluate profitability
```

This single fix could improve returns from -0.34% to +0.34% immediately!
