import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import mean_squared_error, mean_absolute_error
import json

print("=" * 70)
print("XGBoost RANKING MODEL EVALUATION")
print("=" * 70)

# Load model, data, and scaler
print("\nLoading artifacts...")
# Load model, data, and scaler
print("\nLoading artifacts...")
df = pd.read_csv('data/ranking_data_full.csv') # Fixed path
bst = xgb.Booster()
bst.load_model('models/xgb_ranking_model.json') # Fixed path
scaler = pickle.load(open('models/scaler.pkl', 'rb')) # Fixed path

feature_cols = json.load(open('models/model_metadata.json'))['features'] # Fixed path
exclude_cols = ['DateTime', 'DateTime_Hour', 'Query_ID', 'Ticker', 
                'Open', 'High', 'Low', 'Close', 'Volume', 'Next_Hour_Return']

# Feature Engineering: Market Context (Mirroring train logic)
print("Adding Market Context Features...")
if 'Return' in df.columns:
    df['Market_Mean_Return'] = df.groupby('Query_ID')['Return'].transform('mean')
    df['Relative_Return'] = df['Return'] - df['Market_Mean_Return']
    
    if 'HL_Range' in df.columns:
         df['Market_Mean_Volatility'] = df.groupby('Query_ID')['HL_Range'].transform('mean')
         df['Relative_Volatility'] = df['HL_Range'] / (df['Market_Mean_Volatility'] + 1e-8)

# Temporal Split (Same as training)
unique_query_ids = np.sort(df['Query_ID'].unique())
split_idx = int(len(unique_query_ids) * 0.8)
test_qids = unique_query_ids[split_idx:]

print(f"Filtering for Test Set only (Queries {test_qids[0]} to {test_qids[-1]})...")
df = df[df['Query_ID'].isin(test_qids)].copy()

X = df[feature_cols].values

# Handle NaN/Inf
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        if np.isnan(X[i, j]) or np.isinf(X[i, j]):
            X[i, j] = 0.0

X_scaled = scaler.transform(X)

# Predictions
dmatrix = xgb.DMatrix(X_scaled)
y_pred = bst.predict(dmatrix)
y_actual = df['Next_Hour_Return'].values
query_ids = df['Query_ID'].values

print(f"Evaluated {len(y_pred)} predictions across {df['Query_ID'].nunique()} queries")

# ============================================================================
# 1. RANKING METRICS (Per Query)
# ============================================================================
print("\n" + "=" * 70)
print("1. RANKING METRICS (within each hourly query)")
print("=" * 70)

spearman_corrs = []
kendall_corrs = []
top_1_acc = []  # Did best performer get top score?
top_3_acc = []  # Did top 3 performers get top 3 scores?
top_5_acc = []  # Did top 5 performers get top 5 scores?

for qid in np.unique(query_ids):
    mask = query_ids == qid
    if mask.sum() < 2:
        continue
    
    pred_q = y_pred[mask]
    actual_q = y_actual[mask]
    
    # Spearman & Kendall
    sp_corr, _ = spearmanr(pred_q, actual_q)
    if not np.isnan(sp_corr):
        spearman_corrs.append(sp_corr)
    
    kt_corr, _ = kendalltau(pred_q, actual_q)
    if not np.isnan(kt_corr):
        kendall_corrs.append(kt_corr)
    
    # Top-K accuracy
    n = len(pred_q)
    actual_rank = np.argsort(np.argsort(-actual_q))
    pred_rank = np.argsort(np.argsort(-pred_q))
    
    if n >= 1:
        best_actual = np.where(actual_rank == 0)[0][0]
        best_pred_rank = pred_rank[best_actual]
        top_1_acc.append(1 if best_pred_rank == 0 else 0)
    
    if n >= 3:
        top_3_actual = np.where(actual_rank < 3)[0]
        top_3_pred = set(np.where(pred_rank < 3)[0])
        overlap_3 = len(set(top_3_actual) & top_3_pred)
        top_3_acc.append(overlap_3 / 3)
    
    if n >= 5:
        top_5_actual = np.where(actual_rank < 5)[0]
        top_5_pred = set(np.where(pred_rank < 5)[0])
        overlap_5 = len(set(top_5_actual) & top_5_pred)
        top_5_acc.append(overlap_5 / 5)

print(f"\nSpearman Correlation:")
print(f"  Mean:     {np.mean(spearman_corrs) if spearman_corrs else 0.0:.4f}")
print(f"  Std Dev:  {np.std(spearman_corrs) if spearman_corrs else 0.0:.4f}")
print(f"  Median:   {np.median(spearman_corrs) if spearman_corrs else 0.0:.4f}")
print(f"  Min/Max:  {np.min(spearman_corrs) if spearman_corrs else 0.0:.4f} / {np.max(spearman_corrs) if spearman_corrs else 0.0:.4f}")

print(f"\nKendall Tau Correlation:")
print(f"  Mean:     {np.mean(kendall_corrs) if kendall_corrs else 0.0:.4f}")
print(f"  Std Dev:  {np.std(kendall_corrs) if kendall_corrs else 0.0:.4f}")

print(f"\nTop-1 Accuracy (best stock ranked correctly):")
print(f"  {np.mean(top_1_acc)*100:.2f}%")

print(f"\nTop-3 Overlap (how many top-3 are correctly identified):")
print(f"  {np.mean(top_3_acc)*100:.2f}%")

print(f"\nTop-5 Overlap (how many top-5 are correctly identified):")
print(f"  {np.mean(top_5_acc)*100:.2f}%")

# ============================================================================
# 2. REGRESSION METRICS (Predicting actual returns)
# ============================================================================
print("\n" + "=" * 70)
print("2. REGRESSION METRICS (predicting absolute returns)")
print("=" * 70)

mae = mean_absolute_error(y_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
mape = np.mean(np.abs((y_actual - y_pred) / (np.abs(y_actual) + 1e-8)))

print(f"Mean Absolute Error (MAE):      {mae:.6f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.6f}")
print(f"Mean Absolute Percentage Error: {mape*100:.2f}%")

# Actual vs predicted stats
print(f"\nActual returns:    mean={y_actual.mean():.6f}, std={y_actual.std():.6f}")
print(f"Predicted scores:  mean={y_pred.mean():.4f}, std={y_pred.std():.4f}")

# ============================================================================
# 3. PROFITABILITY ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("3. PROFITABILITY ANALYSIS (if we trade based on rankings)")
print("=" * 70)

returns_if_bought_top1 = []
returns_if_bought_top3 = []
returns_if_bought_top5 = []
returns_if_bought_worst1 = []

for qid in np.unique(query_ids):
    mask = query_ids == qid
    if mask.sum() < 5:
        continue
    
    pred_q = y_pred[mask]
    actual_q = y_actual[mask]
    
    # Top-1 strategy
    top1_idx = np.argmax(pred_q)
    returns_if_bought_top1.append(actual_q[top1_idx])
    
    # Top-3 strategy
    top3_idxs = np.argsort(-pred_q)[:3]
    returns_if_bought_top3.append(np.mean(actual_q[top3_idxs]))
    
    # Top-5 strategy
    top5_idxs = np.argsort(-pred_q)[:5]
    returns_if_bought_top5.append(np.mean(actual_q[top5_idxs]))
    
    # Worst-1 strategy (contrarian)
    worst1_idx = np.argmin(pred_q)
    returns_if_bought_worst1.append(actual_q[worst1_idx])

print(f"\nBuy Top-1 Predicted Stock:")
print(f"  Avg Return:  {np.mean(returns_if_bought_top1)*100:.4f}%")
print(f"  Win Rate:    {sum(np.array(returns_if_bought_top1) > 0) / len(returns_if_bought_top1) * 100:.2f}%")

print(f"\nBuy Top-3 Predicted Stocks (equal weight):")
print(f"  Avg Return:  {np.mean(returns_if_bought_top3)*100:.4f}%")
print(f"  Win Rate:    {sum(np.array(returns_if_bought_top3) > 0) / len(returns_if_bought_top3) * 100:.2f}%")

print(f"\nBuy Top-5 Predicted Stocks (equal weight):")
print(f"  Avg Return:  {np.mean(returns_if_bought_top5)*100:.4f}%")
print(f"  Win Rate:    {sum(np.array(returns_if_bought_top5) > 0) / len(returns_if_bought_top5) * 100:.2f}%")

print(f"\nBuy Worst-1 (contrarian):")
print(f"  Avg Return:  {np.mean(returns_if_bought_worst1)*100:.4f}%")

# ============================================================================
# 4. FEATURE IMPORTANCE
# ============================================================================
print("\n" + "=" * 70)
print("4. FEATURE IMPORTANCE (XGBoost)")
print("=" * 70)

importance = bst.get_score(importance_type='weight')
sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

print("\nTop 15 Most Important Features:")
for i, (feat, score) in enumerate(sorted_importance[:15], 1):
    feat_name = feature_cols[int(feat.replace('f', ''))]
    print(f"  {i:2d}. {feat_name:20s} (score: {score})")

# ============================================================================
# 5. PREDICTION CONFIDENCE & DISTRIBUTION
# ============================================================================
print("\n" + "=" * 70)
print("5. PREDICTION CONFIDENCE & DISTRIBUTION")
print("=" * 70)

print(f"\nScore Distribution:")
print(f"  Min:   {np.min(y_pred):.4f}")
print(f"  Q1:    {np.percentile(y_pred, 25):.4f}")
print(f"  Median:{np.median(y_pred):.4f}")
print(f"  Q3:    {np.percentile(y_pred, 75):.4f}")
print(f"  Max:   {np.max(y_pred):.4f}")

# Score variance per query (confidence)
score_vars = []
for qid in np.unique(query_ids):
    mask = query_ids == qid
    score_vars.append(np.var(y_pred[mask]))

print(f"\nScore Variance per Query (model confidence):")
print(f"  Mean Variance: {np.mean(score_vars):.4f}")
print(f"  Min/Max:       {np.min(score_vars):.4f} / {np.max(score_vars):.4f}")

# ============================================================================
# 6. ERROR ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("6. ERROR ANALYSIS")
print("=" * 70)

errors = y_actual - y_pred
positive_return_errors = []
negative_return_errors = []

for i, actual in enumerate(y_actual):
    if actual > 0:
        positive_return_errors.append(errors[i])
    else:
        negative_return_errors.append(errors[i])

print(f"\nErrors on Positive Return Stocks:")
print(f"  Mean Error: {np.mean(positive_return_errors):.6f}")
print(f"  Std Error:  {np.std(positive_return_errors):.6f}")

print(f"\nErrors on Negative Return Stocks:")
print(f"  Mean Error: {np.mean(negative_return_errors):.6f}")
print(f"  Std Error:  {np.std(negative_return_errors):.6f}")

# ============================================================================
# 7. IMPROVEMENT RECOMMENDATIONS
# ============================================================================
print("\n" + "=" * 70)
print("7. IMPROVEMENT RECOMMENDATIONS")
print("=" * 70)

recommendations = []

# Check for overfitting
if np.mean(spearman_corrs) < 0:
    recommendations.append("- Negative Spearman correlation detected - model may be learning inverse patterns")

if np.mean(top_1_acc) < 0.35:
    recommendations.append("- Low top-1 accuracy (<35%) - consider tuning hyperparameters")

if np.std(spearman_corrs) > 0.15:
    recommendations.append("- High variance in predictions across queries - may need regularization")

if len(sorted_importance) < 10:
    recommendations.append("- Few features have importance - consider adding more features or checking data quality")

# Profitability check
avg_top3_return = np.mean(returns_if_bought_top3)
if avg_top3_return < 0.0001:
    recommendations.append("- Low profitability on top-3 strategy - model may need more predictive power")

# Data quality
if len(spearman_corrs) < len(np.unique(query_ids)) * 0.8:
    recommendations.append("- Many queries had NaN correlations - check data quality")

# Feature distribution
feat_stats = []
for col in feature_cols[:5]:
    col_data = df[col].values
    feat_stats.append(np.std(col_data) / (np.mean(np.abs(col_data)) + 1e-8))

if np.mean(feat_stats) > 5:
    recommendations.append("- High feature variance - consider additional normalization")

if recommendations:
    for rec in recommendations:
        print(rec)
else:
    print("Model performing well - minor optimizations possible")

# ============================================================================
# 8. SUMMARY REPORT
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY REPORT")
print("=" * 70)

summary = {
    'Spearman Correlation (mean)': f"{np.mean(spearman_corrs):.4f}",
    'Top-1 Accuracy': f"{np.mean(top_1_acc)*100:.2f}%",
    'Top-3 Overlap': f"{np.mean(top_3_acc)*100:.2f}%",
    'Top-5 Overlap': f"{np.mean(top_5_acc)*100:.2f}%",
    'Avg Return (Top-1)': f"{np.mean(returns_if_bought_top1)*100:.4f}%",
    'Avg Return (Top-3)': f"{np.mean(returns_if_bought_top3)*100:.4f}%",
    'Win Rate (Top-3)': f"{sum(np.array(returns_if_bought_top3) > 0) / len(returns_if_bought_top3) * 100:.2f}%",
    'MAE': f"{mae:.6f}",
    'RMSE': f"{rmse:.6f}",
}

for key, val in summary.items():
    print(f"{key:.<40} {val}")

# Calculate Win/Loss Stats
def calc_win_stats(returns):
    arr = np.array(returns)
    wins = sum(arr > 0)
    losses = sum(arr <= 0)
    win_rate = wins / len(arr) if len(arr) > 0 else 0
    wl_ratio = wins / losses if losses > 0 else float('inf')
    return wins, losses, win_rate, wl_ratio

w1, l1, wr1, wlr1 = calc_win_stats(returns_if_bought_top1)
w3, l3, wr3, wlr3 = calc_win_stats(returns_if_bought_top3)

print(f"\nWin/Loss Stats (Top-1):")
print(f"  Wins: {w1}, Losses: {l1}")
print(f"  Win Rate: {wr1*100:.2f}%")
print(f"  Win/Loss Ratio: {wlr1:.4f}")

# Save report
with open('eval_report.json', 'w') as f:
    json.dump({
        'spearman_mean': float(np.mean(spearman_corrs)),
        'top_1_accuracy': float(np.mean(top_1_acc)),
        'top_3_overlap': float(np.mean(top_3_acc)),
        'avg_return_top1': float(np.mean(returns_if_bought_top1)),
        'avg_return_top3': float(np.mean(returns_if_bought_top3)),
        'mae': float(mae),
        'rmse': float(rmse),
        'top1_win_rate': float(wr1),
        'top1_wl_ratio': float(wlr1),
        'top3_win_rate': float(wr3),
        'top3_wl_ratio': float(wlr3)
    }, f, indent=2)

print("\nReport saved to: eval_report.json")
print("=" * 70)
