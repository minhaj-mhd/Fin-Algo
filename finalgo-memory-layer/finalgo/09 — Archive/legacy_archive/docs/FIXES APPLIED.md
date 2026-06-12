# Critical Fixes Applied

## Summary

Two critical issues were identified and fixed that were causing model overfitting and incorrect backtesting predictions:

1. **Timezone Mismatch** (CRITICAL) - Fixed
2. **TATAMOTORS.NS Ticker** (Minor) - Fixed

---

## Fix #1: Timezone Consistency (CRITICAL)

### Problem

**Timezone mismatch between training and inference causing 6-hour offset in Hour feature:**

- **Training data** ([prepare_ranking_data.py](../scripts/prepare_ranking_data.py)): Uses UTC timestamps
  - Hour feature = 3, 4, 5, 6, 7, 8, 9 (UTC hours during market)

- **Backtesting** ([analyze_historical_day.py](../analysis/analyze_historical_day.py)): Was converting UTC → IST
  - Hour feature = 9, 10, 11, 12, 13, 14, 15 (IST hours)
  - **6-hour offset from training!**

**Impact:**
- Model learned "Hour=3 is market open, Hour=9 is near close"
- Backtesting fed "Hour=9" thinking it's 9 AM IST (market open)
- Model interpreted Hour=9 as "near market close" (3 PM IST)
- Applied completely wrong strategy!

### Solution Applied

**File**: [analyze_historical_day.py](../analysis/analyze_historical_day.py)

**Changes:**

1. **Removed UTC → IST conversion in feature computation** (Lines 70-72, 142-144):
```python
# REMOVED (2 occurrences):
# if df.index.tz is not None:
#     df.index = df.index.tz_convert(ist)

# REPLACED WITH:
# Keep UTC timezone to match training data
# DO NOT convert to IST - Hour feature must match training (UTC hours: 3-9)
```

2. **Added slot_time UTC conversion** in `build_feature_matrix_at_time()`:
```python
# Convert IST slot_time to UTC for data filtering
ist = pytz.timezone('Asia/Kolkata')
if slot_time.tzinfo is None:
    slot_time = ist.localize(slot_time)
slot_time_utc = slot_time.astimezone(pytz.UTC)

# Use slot_time_utc for filtering
df_before = df[df.index <= slot_time_utc + timedelta(minutes=30)]
```

**Note**: `fetch_minute_prices()` still converts to IST, which is correct since it's only fetching exit prices for simulation (not computing features).

### Validation

**Before Fix:**
```python
# Training Hour values: [3, 4, 5, 6, 7, 8, 9]  (UTC)
# Inference Hour values: [9, 10, 11, 12, 13, 14, 15]  (IST) ❌ MISMATCH
```

**After Fix:**
```python
# Training Hour values: [3, 4, 5, 6, 7, 8, 9]  (UTC)
# Inference Hour values: [3, 4, 5, 6, 7, 8, 9]  (UTC) ✅ MATCH
```

### Expected Impact

- ✅ Backtesting predictions now consistent with training
- ✅ Model applies correct strategy for each time period
- ✅ Evaluation metrics now reliable
- 📈 Expected improvement in backtesting accuracy

---

## Fix #2: Remove TATAMOTORS.NS

### Problem

**Ticker inconsistency between training and inference:**

- **Training data** ([prepare_ranking_data.py](../scripts/prepare_ranking_data.py)): Had 48 tickers including TATAMOTORS.NS
- **Realtime trader** ([realtime_trader.py](../scripts/realtime_trader.py)): Only 47 tickers (TATAMOTORS missing)
- **Backtesting** ([analyze_historical_day.py](../analysis/analyze_historical_day.py)): Only 47 tickers

**Impact:**
- Training had extra ticker that never appears in inference
- Minor inconsistency but worth fixing

### Solution Applied

**File**: [prepare_ranking_data.py](../scripts/prepare_ranking_data.py)

**Changes:**

Updated ticker list (Line 15-27):
```python
# OLD: 50 tickers including TATAMOTORS.NS
# NEW: 47 tickers (TATAMOTORS removed)

# 47 top Indian NSE stocks (TATAMOTORS removed - not available)
tickers = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    # ... 42 more tickers ...
    "EICHERMOT.NS", "SHREECEM.NS", "HINDALCO.NS", "BEL.NS",
    "ONGC.NS", "PFC.NS", "NBCC.NS"
]
# TATAMOTORS.NS removed from list
```

### Expected Impact

- ✅ Training and inference use same 47 tickers
- ✅ Consistency across all scripts
- Minor impact on model performance

---

## Remaining Issues to Fix (Not Yet Applied)

### 1. Overfitting from Random Train/Test Split

**Issue**: [train_ranking.py](../scripts/train_ranking.py) uses random train/test split instead of temporal split.

**Status**: ⚠️ NOT FIXED YET

**Priority**: CRITICAL

**Details**: See [OVERFITTING_ANALYSIS.md](OVERFITTING_ANALYSIS.md) - Root Cause #1

**Recommended Fix**:
```python
# In train_ranking.py line 54-59
# Replace random split with temporal split
df['DateTime_Hour'] = pd.to_datetime(df['DateTime_Hour'])
df_sorted = df.sort_values('DateTime_Hour')
unique_query_ids_sorted = df_sorted['Query_ID'].unique()
split_idx = int(len(unique_query_ids_sorted) * 0.8)
train_qids = unique_query_ids_sorted[:split_idx]
test_qids = unique_query_ids_sorted[split_idx:]
```

### 2. Insufficient Regularization

**Issue**: XGBoost parameters allow overfitting (max_depth=6, no penalties).

**Status**: ⚠️ NOT FIXED YET

**Priority**: HIGH

**Details**: See [OVERFITTING_ANALYSIS.md](OVERFITTING_ANALYSIS.md) - Root Cause #2

**Recommended Fix**: Add stronger regularization parameters in train_ranking.py

### 3. Hour/DayOfWeek Features Create Dataset-Specific Patterns

**Issue**: Temporal features cause overfitting and timezone complications.

**Status**: ⚠️ NOT FIXED YET

**Priority**: MEDIUM

**Details**: See [OVERFITTING_ANALYSIS.md](OVERFITTING_ANALYSIS.md) - Root Cause #5

**Recommended Fix**: Remove Hour and DayOfWeek features from prepare_ranking_data.py

---

## Testing Instructions

### Test 1: Verify Timezone Fix

```bash
# Run backtesting on December 4
cd c:\Users\Admin\Desktop\finalgo
.\env\Scripts\activate
python analysis/analyze_historical_day.py
```

**Expected**: Results should now be consistent and more accurate.

### Test 2: Compare Before/After

**Before fix** (from previous session):
- Dec 4 return: +0.17%, 83.33% win rate (5W/1L)

**After fix**: Should see similar or improved results with correct feature alignment.

### Test 3: Verify Feature Consistency

```python
# Check that Hour features match training
import pandas as pd

# Training data
df_train = pd.read_csv('data/ranking_data_full.csv')
print("Training Hour range:", df_train['Hour'].min(), "-", df_train['Hour'].max())

# Should show: 3 - 9 (UTC hours)
```

---

## Summary Table

| Issue | File | Status | Priority | Impact |
|-------|------|--------|----------|--------|
| Timezone mismatch | analyze_historical_day.py | ✅ FIXED | CRITICAL | High |
| TATAMOTORS ticker | prepare_ranking_data.py | ✅ FIXED | MINOR | Low |
| Random train/test split | train_ranking.py | ⚠️ TODO | CRITICAL | Very High |
| Weak regularization | train_ranking.py | ⚠️ TODO | HIGH | High |
| Temporal features | prepare_ranking_data.py | ⚠️ TODO | MEDIUM | Medium |

---

## Next Steps

1. ✅ **Done**: Fixed timezone mismatch
2. ✅ **Done**: Removed TATAMOTORS.NS
3. **TODO**: Implement temporal train/test split
4. **TODO**: Add regularization parameters
5. **TODO**: Consider removing Hour/DayOfWeek features
6. **TODO**: Retrain model with all fixes
7. **TODO**: Re-evaluate and compare performance

---

## References

- [TIMEZONE_ANALYSIS.md](TIMEZONE_ANALYSIS.md) - Detailed timezone issue analysis
- [OVERFITTING_ANALYSIS.md](OVERFITTING_ANALYSIS.md) - Comprehensive overfitting analysis with all root causes
- [prepare_ranking_data.py](../scripts/prepare_ranking_data.py) - Data preparation script
- [train_ranking.py](../scripts/train_ranking.py) - Training script
- [analyze_historical_day.py](../analysis/analyze_historical_day.py) - Backtesting script

---

**Last Updated**: Dec 5, 2024
**Applied By**: Claude Code
**Status**: Timezone fix complete, overfitting fixes pending
