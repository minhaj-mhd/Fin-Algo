import os
import re
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Set styles
plt.style.use('dark_background')
sns.set_palette("viridis")

DATA_PATH = "data/ranking_data_upstox_15min_3y.csv"
MODEL_PATH = "models/v2_15min_3y/xgb_long_model.json"
LOG_PATH = r"C:\Users\loq\.gemini\antigravity\brain\9a59d2ae-a358-4849-b110-0157afed9a87\.system_generated\tasks\task-142.log"
OUT_DIR = r"C:\Users\loq\.gemini\antigravity\brain\9a59d2ae-a358-4849-b110-0157afed9a87\15m_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading model...")
model = xgb.Booster()
model.load_model(MODEL_PATH)

print("Loading test data (2026)...")
df = pd.read_csv(DATA_PATH)
df['DateTime'] = pd.to_datetime(df['DateTime'])
test_df = df[df['DateTime'] >= '2026-01-01'].copy()
print(f"Test Set Rows: {len(test_df)}")

exclude_cols = {
    'DateTime', 'DateTime_15Min', 'Query_ID', 'Ticker', 'Next_15Min_Return',
    'Open', 'High', 'Low', 'Close', 'Volume',
    'Market_Mean_Return', 'Relative_Return',
    'Market_Mean_Volatility', 'Relative_Volatility',
    'Hour', 'DayOfWeek', 'Is_Open_Hour', 'Is_Close_Hour', 'Time_To_Close'
}
features = [c for c in test_df.columns if c not in exclude_cols]

X_test = test_df[features]
y_test = test_df['Next_15Min_Return']

print("Generating Predictions...")
dtest = xgb.DMatrix(X_test, label=y_test)
test_df['Prediction'] = model.predict(dtest)

# ---------------------------------------------------------
# 1 & 2. SHAP Plots (Sampled for speed)
# ---------------------------------------------------------
print("Generating SHAP Plots...")
X_sample = X_test.sample(10000, random_state=42)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_sample, show=False)
plt.savefig(f"{OUT_DIR}/shap_summary.png", bbox_inches='tight', dpi=300)
plt.close()

# Identify top feature for dependence plot
top_feature = X_sample.columns[np.argsort(np.abs(shap_values).mean(0))][-1]
plt.figure(figsize=(10, 6))
shap.dependence_plot(top_feature, shap_values, X_sample, show=False)
plt.savefig(f"{OUT_DIR}/shap_dependence.png", bbox_inches='tight', dpi=300)
plt.close()

# ---------------------------------------------------------
# 3. Learning Curve
# ---------------------------------------------------------
print("Parsing Learning Curve...")
try:
    with open(LOG_PATH, 'r') as f:
        log_text = f.read()
    
    prod_log = log_text.split("Training Production Long Model...")[-1]
    prod_log = prod_log.split("Training Production Short Model...")[0]
    
    train_scores = []
    val_scores = []
    iters = []
    for line in prod_log.split("\n"):
        match = re.search(r'\[(\d+)\]\s+train-ndcg@3:([\d.]+)\s+val-ndcg@3:([\d.]+)', line)
        if match:
            iters.append(int(match.group(1)))
            train_scores.append(float(match.group(2)))
            val_scores.append(float(match.group(3)))

    if iters:
        plt.figure(figsize=(10, 6))
        plt.plot(iters, train_scores, label='Train NDCG@3', color='cyan')
        plt.plot(iters, val_scores, label='Val NDCG@3', color='orange')
        plt.title('XGBoost Learning Curve (Production Long Model)')
        plt.xlabel('Boosting Round')
        plt.ylabel('NDCG@3')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(f"{OUT_DIR}/learning_curve.png", bbox_inches='tight', dpi=300)
        plt.close()
except Exception as e:
    print(f"Could not parse learning curve: {e}")

# ---------------------------------------------------------
# 4. Feature Importance
# ---------------------------------------------------------
print("Generating Feature Importance...")
plt.figure(figsize=(12, 10))
xgb.plot_importance(model, max_num_features=20, height=0.6, color='dodgerblue')
plt.title('Top 20 Feature Importances (Weight)')
plt.savefig(f"{OUT_DIR}/feature_importance.png", bbox_inches='tight', dpi=300)
plt.close()

# ---------------------------------------------------------
# 5. Prediction Bucket Analysis
# ---------------------------------------------------------
print("Generating Prediction Buckets...")
test_df['Bucket'] = pd.qcut(test_df['Prediction'], q=10, labels=False, duplicates='drop')
bucket_returns = test_df.groupby('Bucket')['Next_15Min_Return'].mean() * 10000 # in bps

plt.figure(figsize=(10, 6))
bucket_returns.plot(kind='bar', color='mediumseagreen')
plt.title('Mean Forward Return by Prediction Decile (Test Set)')
plt.xlabel('Prediction Decile (0=Lowest, 9=Highest)')
plt.ylabel('Mean Forward Return (bps)')
plt.axhline(0, color='white', linewidth=1)
plt.grid(alpha=0.3)
plt.savefig(f"{OUT_DIR}/prediction_buckets.png", bbox_inches='tight', dpi=300)
plt.close()

# ---------------------------------------------------------
# 6. Cumulative Return Curve (Top 3 vs Market)
# ---------------------------------------------------------
print("Generating Cumulative Returns...")
# Group by time and select top 3
top3_returns = test_df.sort_values(['DateTime', 'Prediction'], ascending=[True, False]) \
                      .groupby('DateTime').head(3) \
                      .groupby('DateTime')['Next_15Min_Return'].mean()

market_returns = test_df.groupby('DateTime')['Next_15Min_Return'].mean()

# Calculate cumulative geometric return
cum_top3 = (1 + top3_returns).cumprod()
cum_market = (1 + market_returns).cumprod()

plt.figure(figsize=(12, 6))
plt.plot(cum_top3.index, cum_top3.values, label='Top 3 Model Portfolio', color='lime')
plt.plot(cum_market.index, cum_market.values, label='Universe Average (Market)', color='grey', alpha=0.7)
plt.title('Cumulative Return: Model Top 3 vs Market (Test Period: 2026)')
plt.xlabel('Date')
plt.ylabel('Cumulative Return')
plt.legend()
plt.grid(alpha=0.3)
plt.savefig(f"{OUT_DIR}/cumulative_return.png", bbox_inches='tight', dpi=300)
plt.close()

# ---------------------------------------------------------
# 7. Calibration Plot
# ---------------------------------------------------------
print("Generating Calibration Plot...")
plt.figure(figsize=(10, 6))
plt.hexbin(test_df['Prediction'], test_df['Next_15Min_Return']*100, gridsize=50, cmap='inferno', bins='log')
plt.colorbar(label='log10(N)')
plt.title('Calibration Plot: Prediction Score vs Actual Return (%)')
plt.xlabel('XGBoost Prediction Score')
plt.ylabel('Actual Next 15M Return (%)')
plt.grid(alpha=0.2)
plt.savefig(f"{OUT_DIR}/calibration_plot.png", bbox_inches='tight', dpi=300)
plt.close()

# ---------------------------------------------------------
# 8. Residual Analysis
# ---------------------------------------------------------
print("Generating Residuals...")
test_df['Residual'] = (test_df['Next_15Min_Return'] - test_df['Prediction']) * 100
plt.figure(figsize=(10, 6))
sns.histplot(test_df['Residual'], bins=100, kde=True, color='purple')
plt.title('Residual Distribution (Actual - Predicted) %')
plt.xlabel('Residual %')
plt.xlim(-5, 5)
plt.grid(alpha=0.3)
plt.savefig(f"{OUT_DIR}/residual_analysis.png", bbox_inches='tight', dpi=300)
plt.close()

print("All analysis complete! Plots saved to artifacts/15m_analysis/")
