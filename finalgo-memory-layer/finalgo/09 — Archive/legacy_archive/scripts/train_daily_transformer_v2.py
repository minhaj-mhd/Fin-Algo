"""
Train a daily macro PyTorch Inverted Transformer (iTransformer) stock ranking model.
- Uses data/ranking_data_upstox_daily_5y_transformer.csv (~32 stationary, pruned features)
- Pre-compiles 20-day sequential tensors: input shape (N_stocks, 20_days, num_features)
- Implements iTransformer: treats features as tokens and attends across the feature dimension
- Optimizes using ListNet Listwise Ranking Loss to directly maximize cross-sectional ranking accuracy
- Employs Grouped-Day Batching (yields complete cross-sections of stocks per trading day)
- Employs Gradient Accumulation over 16 steps for stable deep learning convergence
- Validates using 4-fold Walk-Forward Validation (preventing leakage and tracking Spearman Rho/Edge)
- Saves best checkpoints, scaler, and metadata to models/daily_transformer_v2/
"""

import os
import sys
import pickle
import json
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from tqdm import tqdm

sys.path.append(os.getcwd())

# ========================================
# CONFIG
# ========================================
MODEL_VERSION = 'daily_transformer_v2'
MODEL_DIR = f'models/{MODEL_VERSION}'
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH  = f'{MODEL_DIR}/daily_transformer.pth'
META_PATH   = f'{MODEL_DIR}/metadata.json'
SCALER_PATH = f'{MODEL_DIR}/scaler.pkl'

DATA_FILE = 'data/ranking_data_upstox_daily_5y_transformer.csv'
LOOKBACK_LEN = 20
TEMPERATURE = 100.0  # ListNet return-scale factor (T=100.0 scales 1% return to 1.0)
GRAD_ACCUM_STEPS = 16  # Accumulate over 16 days (~3 business weeks) for massive gradient stability

print("=" * 70)
print("SOTA INVERTED TRANSFORMER + LISTNET RANKING PIPELINE")
print("Walk-Forward Validation on Grouped Daily Cross-Sections")
print("=" * 70)

if not os.path.exists(DATA_FILE):
    print(f"[FATAL] Data file not found: {DATA_FILE}")
    print("Please run scripts/rebuild_transformer_dataset.py first.")
    sys.exit(1)

print(f"Loading data from {DATA_FILE}...")
df = pd.read_csv(DATA_FILE)
print(f"Loaded {df.shape[0]:,} rows")

# Extract the YYYY-MM month string for temporal splits
df['YearMonth'] = df['DateTime'].str[:7]
unique_months = sorted(df['YearMonth'].unique())
print(f"Data spans {len(unique_months)} months: {unique_months[0]} to {unique_months[-1]}")

# ========================================
# FEATURE SELECTION
# ========================================
exclude_cols = ['DateTime', 'Query_ID', 'Ticker',
                'Open', 'High', 'Low', 'Close', 'Volume', 'Next_Day_Return', 'YearMonth']
feature_cols = [col for col in df.columns if col not in exclude_cols]

print(f"Features in use ({len(feature_cols)}): {feature_cols}")

# ========================================
# GROUPED DAY DATASET & PRE-COMPILATION
# ========================================
print("\nPre-compiling 20-day sequential tensors per stock...")
# Sort explicitly to guarantee correct time ordering
df = df.sort_values(['Ticker', 'DateTime']).reset_index(drop=True)

feature_indices = [df.columns.get_loc(col) for col in feature_cols]
return_idx = df.columns.get_loc('Next_Day_Return')
qid_idx = df.columns.get_loc('Query_ID')
ym_idx = df.columns.get_loc('YearMonth')
dt_idx = df.columns.get_loc('DateTime')

# day_data index -> {qid: {'X': [seq1, seq2, ...], 'y': [ret1, ret2, ...], 'tickers': [...], 'dates': [...], 'ym': 'YYYY-MM'}}
day_data = {}

for ticker, g in tqdm(df.groupby('Ticker'), desc="Sequence Gen"):
    vals = g.values
    if len(vals) < LOOKBACK_LEN:
        continue
    
    # Pre-extract values as arrays to bypass slow pandas indexing
    X_vals = vals[:, feature_indices].astype(np.float32)
    y_vals = vals[:, return_idx].astype(np.float32)
    q_vals = vals[:, qid_idx].astype(np.int64)
    ym_vals = vals[:, ym_idx]
    dt_vals = vals[:, dt_idx]
    
    # Fill any NaNs/Infs in raw features with 0
    X_vals = np.nan_to_num(X_vals, nan=0.0, posinf=0.0, neginf=0.0)

    # Compile lookback sequences
    for i in range(LOOKBACK_LEN - 1, len(vals)):
        seq = X_vals[i - (LOOKBACK_LEN - 1) : i + 1]  # shape: (20, num_features)
        target_ret = y_vals[i]
        qid = q_vals[i]
        dt = dt_vals[i]
        ym = ym_vals[i]
        
        if qid not in day_data:
            day_data[qid] = {'X': [], 'y': [], 'tickers': [], 'dates': [], 'ym': ym}
        
        day_data[qid]['X'].append(seq)
        day_data[qid]['y'].append(target_ret)
        day_data[qid]['tickers'].append(ticker)
        day_data[qid]['dates'].append(dt)

# Convert compiled day dictionary to a chronological list of days
sorted_qids = sorted(day_data.keys())
chronological_days = [day_data[q] for q in sorted_qids]

print(f"\nGrouped Day Compilation Complete:")
print(f"  Total chronological days: {len(chronological_days)}")
print(f"  Total cross-sectional samples: {sum(len(d['y']) for d in chronological_days):,}")

# ========================================
# PYTORCH GROUPED DATASET
# ========================================
class DailyGroupedDataset(Dataset):
    def __init__(self, days_list):
        self.days = []
        for day in days_list:
            X_tensor = torch.tensor(np.array(day['X'], dtype=np.float32), dtype=torch.float32)
            y_tensor = torch.tensor(np.array(day['y'], dtype=np.float32), dtype=torch.float32)
            self.days.append({
                'X': X_tensor,
                'y': y_tensor,
                'tickers': day['tickers'],
                'dates': day['dates']
            })
        
    def __len__(self):
        return len(self.days)
        
    def __getitem__(self, idx):
        day = self.days[idx]
        return day['X'], day['y'], day['tickers'], day['dates']


# ========================================
# SOTA iTRANSFORMER (INVERTED TRANSFORMER)
# ========================================
class InvertedTransformerRanker(nn.Module):
    def __init__(self, num_features, lookback_len=20, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.num_features = num_features
        self.lookback_len = lookback_len
        self.d_model = d_model
        
        # Shared Projection: Projects the entire 20-day path of each feature to a d_model token
        self.feature_projection = nn.Linear(lookback_len, d_model)
        
        # Learnable indicator identity embeddings
        self.feature_embed = nn.Parameter(torch.randn(num_features, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Deep Regressor Head
        self.regressor = nn.Sequential(
            nn.Linear(num_features * d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        # x shape: (N_stocks, lookback_len, num_features)
        
        # Rearrange to treat features as tokens (sequence dimension):
        # (N_stocks, num_features, lookback_len)
        x = x.transpose(1, 2)
        
        # Project 20-day history of each feature into a token:
        # (N_stocks, num_features, d_model)
        x = self.feature_projection(x)
        
        # Add feature identity embeddings
        x = x + self.feature_embed.unsqueeze(0)
        
        # Transformer Self-Attention (Attention computed ACROSS features!)
        # (N_stocks, num_features, d_model)
        x = self.transformer_encoder(x)
        
        # Flatten all feature tokens
        # (N_stocks, num_features * d_model)
        x_flat = x.reshape(x.size(0), -1)
        
        # Regressor score
        out = self.regressor(x_flat)
        return out.squeeze(-1)

# ========================================
# LISTNET LISTWISE RANKING LOSS
# ========================================
def listnet_loss(y_pred, y_true, temp=100.0):
    """
    y_pred: [N_stocks] predicted ranking scores
    y_true: [N_stocks] ground truth next-day returns
    temp: temperature factor to sharpen return softmax
    """
    if len(y_pred) <= 1:
        return torch.tensor(0.0, device=y_pred.device, requires_grad=True)
        
    # Scale returns (e.g. 0.01 return becomes 1.0) for sharp target probabilities
    y_true_scaled = y_true * temp
    
    # Target distribution (Softmax over ground-truth returns)
    target_dist = F.softmax(y_true_scaled, dim=0)
    
    # Model prediction log-probabilities
    pred_log_soft = F.log_softmax(y_pred, dim=0)
    
    # Cross-Entropy Loss
    loss = -torch.sum(target_dist * pred_log_soft)
    return loss

# ========================================
# GPU DETECTION
# ========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device selected: {device}")

# ========================================
# METRIC EVALUATOR
# ========================================
def evaluate_ranking_performance(df_eval, long_scores, short_scores):
    df_sub = df_eval.copy()
    df_sub['long_score'] = long_scores
    df_sub['short_score'] = short_scores
    
    unique_qids = df_sub['Query_ID'].unique()
    
    # Spearman Rhos
    long_rhos = []
    short_rhos = []
    for qid in unique_qids:
        q_df = df_sub[df_sub['Query_ID'] == qid]
        if len(q_df) > 1:
            # Long
            rho_l, _ = spearmanr(q_df['long_score'].values, q_df['Next_Day_Return'].values)
            if not np.isnan(rho_l):
                long_rhos.append(rho_l)
            # Short (invert targets)
            rho_s, _ = spearmanr(q_df['short_score'].values, -q_df['Next_Day_Return'].values)
            if not np.isnan(rho_s):
                short_rhos.append(rho_s)
                
    long_rho = np.mean(long_rhos) if long_rhos else 0.0
    short_rho = np.mean(short_rhos) if short_rhos else 0.0
    
    # Top-K Win Rates (K = 1, 3, 5)
    topk_list = [1, 3, 5]
    long_win_rates = {}
    short_win_rates = {}
    
    for k in topk_list:
        long_hits = 0
        long_total = 0
        short_hits = 0
        short_total = 0
        
        for qid in unique_qids:
            q_df = df_sub[df_sub['Query_ID'] == qid]
            if len(q_df) < k + 1:
                continue
                
            actual_returns = q_df['Next_Day_Return'].values
            median_return = np.median(actual_returns)
            
            # Long (pick high score)
            top_long_idx = np.argsort(q_df['long_score'].values)[::-1][:k]
            long_hits += (actual_returns[top_long_idx] > median_return).sum()
            long_total += k
            
            # Short (pick high score in inverted)
            top_short_idx = np.argsort(q_df['short_score'].values)[::-1][:k]
            short_hits += (actual_returns[top_short_idx] < median_return).sum()
            short_total += k
            
        long_win_rates[k] = long_hits / long_total if long_total > 0 else 0.0
        short_win_rates[k] = short_hits / short_total if short_total > 0 else 0.0
        
    # Expected Returns & Edges at Top 3
    top3_long_returns = []
    top3_short_returns = []
    random_returns = []
    
    for qid in unique_qids:
        q_df = df_sub[df_sub['Query_ID'] == qid]
        if len(q_df) < 4:
            continue
            
        actual = q_df['Next_Day_Return'].values
        
        top3_long_idx = np.argsort(q_df['long_score'].values)[::-1][:3]
        top3_short_idx = np.argsort(q_df['short_score'].values)[::-1][:3]
        
        top3_long_returns.append(actual[top3_long_idx].mean())
        top3_short_returns.append(-actual[top3_short_idx].mean())
        random_returns.append(actual.mean())
        
    if top3_long_returns:
        avg_long = np.mean(top3_long_returns)
        avg_short = np.mean(top3_short_returns)
        avg_rand = np.mean(random_returns)
        
        long_edge = avg_long - avg_rand
        short_edge = avg_short - (-avg_rand)
        combined_edge = avg_long + avg_short
    else:
        avg_long, avg_short, avg_rand = 0.0, 0.0, 0.0
        long_edge, short_edge, combined_edge = 0.0, 0.0, 0.0
        
    return {
        'long_rho': long_rho,
        'short_rho': short_rho,
        'long_win_rates': long_win_rates,
        'short_win_rates': short_win_rates,
        'avg_long_return': avg_long,
        'avg_short_return': avg_short,
        'avg_market_return': avg_rand,
        'long_edge': long_edge,
        'short_edge': short_edge,
        'combined_edge': combined_edge
    }

# ========================================
# WALK-FORWARD FOLDS CONFIG
# ========================================
folds_config = [
    {
        'fold': 1,
        'train_months': unique_months[:30],
        'val_months': unique_months[30:36],
        'test_months': unique_months[36:42]
    },
    {
        'fold': 2,
        'train_months': unique_months[6:36],
        'val_months': unique_months[36:42],
        'test_months': unique_months[42:48]
    },
    {
        'fold': 3,
        'train_months': unique_months[12:42],
        'val_months': unique_months[42:48],
        'test_months': unique_months[48:54]
    },
    {
        'fold': 4,
        'train_months': unique_months[18:48],
        'val_months': unique_months[48:54],
        'test_months': unique_months[54:]
    }
]

walk_forward_results = []

# ========================================
# MODEL TRAINING FUNCTION
# ========================================
def train_model(train_loader, val_loader, num_features, epochs=100, patience=10, fold_num="Prod"):
    model = InvertedTransformerRanker(
        num_features=num_features,
        lookback_len=LOOKBACK_LEN,
        d_model=64,
        nhead=4,
        num_layers=2,
        dropout=0.1
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = 9999.0
    best_state = None
    patience_counter = 0
    
    print(f"\n  [TRAINING START] {fold_num} | Features: {num_features} | Accumulation Steps: {GRAD_ACCUM_STEPS}")
    print(f"  {'Epoch':<10} | {'Train Loss':<12} | {'Val Loss':<12} | {'LR':<10} | {'Status':<15}")
    print(f"  {'-'*65}")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()
        
        for step, (X_batch, y_batch, _, _) in enumerate(train_loader):
            # Shape from DataLoader: (1, N_stocks, 20, M) -> Squeeze batch dim
            X_batch = X_batch.squeeze(0).to(device)
            y_batch = y_batch.squeeze(0).to(device)
            
            if len(y_batch) <= 1:
                continue
                
            pred = model(X_batch)
            loss = listnet_loss(pred, y_batch, temp=TEMPERATURE)
            
            # Scale loss for gradient accumulation
            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()
            
            train_loss += loss.item() * GRAD_ACCUM_STEPS
            
            # Step optimizer every GRAD_ACCUM_STEPS daily batches
            if (step + 1) % GRAD_ACCUM_STEPS == 0 or (step + 1) == len(train_loader):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                
        train_loss /= len(train_loader)
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        # Validation Evaluation (Loss-based for 10x GPU speed)
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for X_val_batch, y_val_batch, _, _ in val_loader:
                X_val_batch = X_val_batch.squeeze(0).to(device)
                y_val_batch = y_val_batch.squeeze(0).to(device)
                
                if len(y_val_batch) <= 1:
                    continue
                    
                pred = model(X_val_batch)
                loss = listnet_loss(pred, y_val_batch, temp=TEMPERATURE)
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        
        status_str = ""
        # Early Stopping check (Minimize Validation ListNet Loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
            status_str = "★ New Best"
        else:
            patience_counter += 1
            status_str = f"Patience {patience_counter}/{patience}"
            
        print(f"  Epoch {epoch+1:<4}/{epochs:<3} | {train_loss:<12.5f} | {val_loss:<12.5f} | {current_lr:<10.6f} | {status_str:<15}")
        sys.stdout.flush()  # Force unbuffered terminal output
        
        if patience_counter >= patience:
            print(f"  [EARLY STOP] Fold {fold_num} stopped at Epoch {epoch+1}. Best Val Loss: {best_val_loss:.5f}")
            sys.stdout.flush()
            break
            
    # Load best model weights
    model.load_state_dict(best_state)
    return model

# ========================================
# RUN WALK-FORWARD VALIDATION
# ========================================
print("\n" + "=" * 70)
print("RUNNING WALK-FORWARD VALIDATION ON INVERTED TRANSFORMER")
print("=" * 70)

for cfg in folds_config:
    fold_idx = cfg['fold']
    tr_m, val_m, te_m = cfg['train_months'], cfg['val_months'], cfg['test_months']
    
    print(f"\n--- FOLD {fold_idx} ---")
    print(f"  Train: {tr_m[0]} -> {tr_m[-1]}")
    print(f"  Val:   {val_m[0]} -> {val_m[-1]}")
    print(f"  Test:  {te_m[0]} -> {te_m[-1]}")
    
    # Split daily cross-sections
    tr_days = [d for d in chronological_days if d['ym'] in tr_m]
    val_days = [d for d in chronological_days if d['ym'] in val_m]
    te_days = [d for d in chronological_days if d['ym'] in te_m]
    
    print(f"  Days: Train={len(tr_days)}, Val={len(val_days)}, Test={len(te_days)}")
    
    train_dataset = DailyGroupedDataset(tr_days)
    val_dataset = DailyGroupedDataset(val_days)
    test_dataset = DailyGroupedDataset(te_days)
    
    # Grouped DataLoader: batch_size=1 represents 1 trading day cross-section
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    print(f"  Training iTransformer sequence model for Fold {fold_idx}...")
    model = train_model(train_loader, val_loader, len(feature_cols), epochs=80, patience=10, fold_num=f"Fold {fold_idx}")
    
    # Predict on test set
    model.eval()
    test_preds = []
    test_targets = []
    test_qids = []
    test_tickers = []
    test_dates = []
    
    with torch.no_grad():
        for test_idx, (X_te_batch, y_te_batch, tickers, dates) in enumerate(test_loader):
            X_te_batch = X_te_batch.squeeze(0).to(device)
            y_te_batch = y_te_batch.squeeze(0)
            
            if len(y_te_batch) <= 1:
                continue
                
            pred = model(X_te_batch)
            
            test_preds.extend(pred.cpu().numpy())
            test_targets.extend(y_te_batch.numpy())
            test_qids.extend([test_idx] * len(y_te_batch))
            test_tickers.extend(tickers)
            test_dates.extend(dates)
            
    df_te_eval = pd.DataFrame({
        'DateTime': test_dates,
        'Query_ID': test_qids,
        'Ticker': test_tickers,
        'Next_Day_Return': test_targets
    })
    
    long_scores = np.array(test_preds)
    short_scores = -long_scores
    
    metrics = evaluate_ranking_performance(df_te_eval, long_scores, short_scores)
    metrics['fold'] = fold_idx
    walk_forward_results.append(metrics)
    
    # Print fold summary
    print(f"  [FOLD SUMMARY]")
    print(f"    Long Rho : {metrics['long_rho']:.4f} | Short Rho: {metrics['short_rho']:.4f}")
    print(f"    Long Win Rate @ 3 : {metrics['long_win_rates'][3]:.1%} | Short @ 3: {metrics['short_win_rates'][3]:.1%}")
    print(f"    Long Return Edge  : {metrics['long_edge']*100:+.4f}% | Short Edge: {metrics['short_edge']*100:+.4f}% per day")
    print(f"    Combined Edge     : {metrics['combined_edge']*100:+.4f}% per day")

# ========================================
# REPORT AGGREGATE WF RESULTS
# ========================================
avg_long_rho = np.mean([r['long_rho'] for r in walk_forward_results])
avg_short_rho = np.mean([r['short_rho'] for r in walk_forward_results])

avg_long_wr_1 = np.mean([r['long_win_rates'][1] for r in walk_forward_results])
avg_long_wr_3 = np.mean([r['long_win_rates'][3] for r in walk_forward_results])
avg_long_wr_5 = np.mean([r['long_win_rates'][5] for r in walk_forward_results])

avg_short_wr_1 = np.mean([r['short_win_rates'][1] for r in walk_forward_results])
avg_short_wr_3 = np.mean([r['short_win_rates'][3] for r in walk_forward_results])
avg_short_wr_5 = np.mean([r['short_win_rates'][5] for r in walk_forward_results])

avg_long_ret = np.mean([r['avg_long_return'] for r in walk_forward_results])
avg_short_ret = np.mean([r['avg_short_return'] for r in walk_forward_results])
avg_market_ret = np.mean([r['avg_market_return'] for r in walk_forward_results])

avg_long_edge = np.mean([r['long_edge'] for r in walk_forward_results])
avg_short_edge = np.mean([r['short_edge'] for r in walk_forward_results])
avg_combined_edge = np.mean([r['combined_edge'] for r in walk_forward_results])

print("\n" + "=" * 70)
print("AGGREGATE WALK-FORWARD VALIDATION RESULTS")
print("=" * 70)
print(f"Averaged over {len(folds_config)} temporal test folds:")
print(f"  Average Spearman Rho:")
print(f"    Long Model  : {avg_long_rho:.4f}")
print(f"    Short Model : {avg_short_rho:.4f}")
print(f"  Average Win Rates (Beats/Falls Below Median):")
print(f"    Long Model  : K=1: {avg_long_wr_1:.1%}, K=3: {avg_long_wr_3:.1%}, K=5: {avg_long_wr_5:.1%}")
print(f"    Short Model : K=1: {avg_short_wr_1:.1%}, K=3: {avg_short_wr_3:.1%}, K=5: {avg_short_wr_5:.1%}")
print(f"  Average Top-3 Expected Returns & Edges:")
print(f"    Top-3 Long Selections  Avg Return: {avg_long_ret*100:+.4f}% per day")
print(f"    Top-3 Short Selections Avg Return: {avg_short_ret*100:+.4f}% per day")
print(f"    Market (Random) Pick   Avg Return: {avg_market_ret*100:+.4f}% per day")
print(f"    Long Edge over Market  : {avg_long_edge*100:+.4f}% per day")
print(f"    Short Edge over Market : {avg_short_edge*100:+.4f}% per day")
print(f"    Combined Long/Short Edge: {avg_combined_edge*100:+.4f}% per day")
print("=" * 70)

# ========================================
# TRAIN PRODUCTION MODEL
# ========================================
print("\n" + "=" * 70)
print("TRAINING FINAL PRODUCTION MODEL ON ALL DATA")
print("=" * 70)

# Dump scaler (using empty StandardScaler to match pipeline structure)
scaler = StandardScaler(with_mean=False, with_std=False)
with open(SCALER_PATH, 'wb') as f:
    pickle.dump(scaler, f)

# Train on all months except the last 2, validation on the last 2 for early stopping
prod_train_months = unique_months[:-2]
prod_val_months = unique_months[-2:]

print(f"  Production Train: {prod_train_months[0]} -> {prod_train_months[-1]}")
print(f"  Production Val:   {prod_val_months[0]} -> {prod_val_months[-1]}")

prod_tr_days = [d for d in chronological_days if d['ym'] in prod_train_months]
prod_val_days = [d for d in chronological_days if d['ym'] in prod_val_months]

prod_train_dataset = DailyGroupedDataset(prod_tr_days)
prod_val_dataset = DailyGroupedDataset(prod_val_days)

prod_train_loader = DataLoader(prod_train_dataset, batch_size=1, shuffle=True)
prod_val_loader = DataLoader(prod_val_dataset, batch_size=1, shuffle=False)

print("  Training Production sequence model...")
prod_model = train_model(prod_train_loader, prod_val_loader, len(feature_cols), epochs=80, patience=10, fold_num="Production Model")

# Save checkpoint
torch.save(prod_model.state_dict(), MODEL_PATH)
print(f"    Saved Production Model Checkpoint: {MODEL_PATH}")

# ========================================
# SAVE METADATA
# ========================================
metadata = {
    'features': feature_cols,
    'num_features': len(feature_cols),
    'data_source': 'upstox_5y_daily_transformer_v2',
    'data_file': DATA_FILE,
    'total_days': len(chronological_days),
    'total_sequences': sum(len(d['y']) for d in chronological_days),
    'walk_forward_summary': {
        'avg_long_spearman': float(avg_long_rho),
        'avg_short_spearman': float(avg_short_rho),
        'avg_long_win_rate_k3': float(avg_long_wr_3),
        'avg_short_win_rate_k3': float(avg_short_wr_3),
        'avg_long_return_edge_k3': float(avg_long_edge),
        'avg_short_return_edge_k3': float(avg_short_edge),
        'avg_combined_edge_k3': float(avg_combined_edge),
    },
    'walk_forward_folds': [
        {
            'fold': int(r['fold']),
            'long_rho': float(r['long_rho']),
            'short_rho': float(r['short_rho']),
            'long_win_rates': {str(k): float(v) for k, v in r['long_win_rates'].items()},
            'short_win_rates': {str(k): float(v) for k, v in r['short_win_rates'].items()},
            'long_edge': float(r['long_edge']),
            'short_edge': float(r['short_edge']),
            'combined_edge': float(r['combined_edge']),
        }
        for r in walk_forward_results
    ],
    'production_training': {
        'train_months': prod_train_months,
        'val_months': prod_val_months,
        'lookback_len': LOOKBACK_LEN,
        'd_model': 64,
        'nhead': 4,
        'num_layers': 2,
    },
    'trained_at': datetime.now().isoformat(),
}

with open(META_PATH, 'w') as f:
    json.dump(metadata, f, indent=2)

print("\n" + "=" * 70)
print("DAILY MACRO INVERTED TRANSFORMER TRAINING COMPLETE")
print(f"  Model Checkpoint : {MODEL_PATH}")
print(f"  Metadata         : {META_PATH}")
print(f"  Scaler           : {SCALER_PATH}")
print("=" * 70)
print()
