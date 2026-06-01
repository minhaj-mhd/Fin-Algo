"""
Train a daily macro PyTorch Temporal Transformer ranking model.
- Loads data/ranking_data_upstox_daily_5y.csv
- Compiles 20-day sequence tensors: input shape (batch_size, 20_days, num_features)
- Defines TemporalTransformerRanker with Multi-Head Self-Attention
- Implements 4-fold Walk-Forward Validation to prevent regime overfitting and leakage
- Evaluates Spearman Rho and Top-K metrics (Top 1, 3, 5 selections)
- Saves best checkpoints, walk-forward metadata, and scaler to models/daily_transformer/
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
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, rankdata
from tqdm import tqdm

sys.path.append(os.getcwd())

# ========================================
# CONFIG
# ========================================
MODEL_VERSION = 'daily_transformer'
MODEL_DIR = f'models/{MODEL_VERSION}'
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH  = f'{MODEL_DIR}/daily_transformer.pth'
META_PATH   = f'{MODEL_DIR}/metadata.json'
SCALER_PATH = f'{MODEL_DIR}/scaler.pkl'

DATA_FILE = 'data/ranking_data_upstox_daily_5y_transformer.csv'

print("=" * 60)
print("DAILY MACRO TEMPORAL TRANSFORMER RANKER PIPELINE")
print("Walk-Forward Validation with sequence-based deep learning")
print("=" * 60)

if not os.path.exists(DATA_FILE):
    print(f"[FATAL] Data file not found: {DATA_FILE}")
    print("Please run scripts/collect_upstox_daily_5y.py first.")
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

print(f"Features: {len(feature_cols)}")
print(f"Samples: {df.shape[0]:,}")

# ========================================
# COMPILE 10-DAY TIME SERIES SEQUENCES
# ========================================
print("\nCompiling 10-day sequential tensors per stock...")
# Sort explicitly to guarantee correct time ordering
df = df.sort_values(['Ticker', 'DateTime']).reset_index(drop=True)

feature_indices = [df.columns.get_loc(col) for col in feature_cols]
return_idx = df.columns.get_loc('Next_Day_Return')
qid_idx = df.columns.get_loc('Query_ID')
ym_idx = df.columns.get_loc('YearMonth')
dt_idx = df.columns.get_loc('DateTime')

X_seqs = []
y_seqs = []
qid_seqs = []
ym_seqs = []
ticker_seqs = []
dt_seqs = []

# Loop through each stock's history and extract sequences
for ticker, g in tqdm(df.groupby('Ticker'), desc="Sequence Gen"):
    vals = g.values
    if len(vals) < 10:
        continue
    
    # Pre-extract values as arrays to bypass slow pandas indexing
    X_vals = vals[:, feature_indices].astype(np.float32)
    y_vals = vals[:, return_idx].astype(np.float32)
    q_vals = vals[:, qid_idx].astype(np.int64)
    ym_vals = vals[:, ym_idx]
    dt_vals = vals[:, dt_idx]
    
    # Fill any NaNs/Infs in raw features with 0
    nan_mask = np.isnan(X_vals) | np.isinf(X_vals)
    if nan_mask.any():
        X_vals[nan_mask] = 0.0

    for i in range(9, len(vals)):
        X_seqs.append(X_vals[i-9 : i+1])
        y_seqs.append(y_vals[i])
        qid_seqs.append(q_vals[i])
        ym_seqs.append(ym_vals[i])
        ticker_seqs.append(ticker)
        dt_seqs.append(dt_vals[i])

X_seqs = np.array(X_seqs, dtype=np.float32)
y_seqs = np.array(y_seqs, dtype=np.float32)
qid_seqs = np.array(qid_seqs, dtype=np.int64)
ym_seqs = np.array(ym_seqs)
ticker_seqs = np.array(ticker_seqs)
dt_seqs = np.array(dt_seqs)

print(f"Sequences compiled: {X_seqs.shape[0]:,} samples")
print(f"Sequence shape    : {X_seqs.shape}")

# Create evaluation DataFrame for computing Spearman and Top-K metrics
df_eval_all = pd.DataFrame({
    'DateTime': dt_seqs,
    'Query_ID': qid_seqs,
    'Ticker': ticker_seqs,
    'Next_Day_Return': y_seqs,
    'YearMonth': ym_seqs
})

# ========================================
# PYTORCH DATASET & DATALOADER
# ========================================
class DailySequenceDataset(Dataset):
    def __init__(self, X, y, qids):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.qids = torch.tensor(qids, dtype=torch.long)
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.qids[idx]

# ========================================
# TRANSFORMATION MODEL DEFINITIONS
# ========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TemporalTransformerRanker(nn.Module):
    def __init__(self, input_dim, d_model=32, nhead=2, num_layers=1, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=100)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.regressor = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        
        # Mean pooling across the temporal dimension
        x_mean = x.mean(dim=1)
        
        out = self.regressor(x_mean)
        return out.squeeze(-1)

# ========================================
# GPU DETECTION
# ========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device selected: {device}")

# ========================================
# METRIC EVALUATORS
# ========================================
def compute_spearman_rho(df_eval, score_col, invert=False):
    rhos = []
    for qid in df_eval['Query_ID'].unique():
        q_df = df_eval[df_eval['Query_ID'] == qid]
        if len(q_df) > 1:
            y_eval = -q_df['Next_Day_Return'].values if invert else q_df['Next_Day_Return'].values
            pred = q_df[score_col].values
            rho, _ = spearmanr(pred, y_eval)
            if not np.isnan(rho):
                rhos.append(rho)
    return np.mean(rhos) if rhos else 0.0

def evaluate_ranking_performance(df_eval, long_scores, short_scores):
    df_sub = df_eval.copy()
    df_sub['long_score'] = long_scores
    df_sub['short_score'] = short_scores
    
    unique_qids = df_sub['Query_ID'].unique()
    
    # Spearman Rhos
    long_rho = compute_spearman_rho(df_sub, 'long_score', invert=False)
    short_rho = compute_spearman_rho(df_sub, 'short_score', invert=True)
    
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
            
            # Long
            long_sc = q_df['long_score'].values
            top_long_idx = np.argsort(long_sc)[::-1][:k]
            long_hits += (actual_returns[top_long_idx] > median_return).sum()
            long_total += k
            
            # Short
            short_sc = q_df['short_score'].values
            top_short_idx = np.argsort(short_sc)[::-1][:k]
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
        long_sc = q_df['long_score'].values
        short_sc = q_df['short_score'].values
        
        top3_long_idx = np.argsort(long_sc)[::-1][:3]
        top3_short_idx = np.argsort(short_sc)[::-1][:3]
        
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
def train_model(train_loader, val_loader, df_val, num_features, epochs=100, patience=10):
    model = TemporalTransformerRanker(input_dim=num_features).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.0001)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_rho = -999.0
    best_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch, _ in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * X_batch.size(0)
            
        train_loss /= len(train_loader.dataset)
        scheduler.step()
        
        # Validation evaluation
        model.eval()
        val_preds = []
        with torch.no_grad():
            for X_batch, _, _ in val_loader:
                X_batch = X_batch.to(device)
                pred = model(X_batch)
                val_preds.extend(pred.cpu().numpy())
                
        # Spearman correlation on val
        df_val_eval = df_val.copy()
        df_val_eval['pred_score'] = val_preds
        val_rho = compute_spearman_rho(df_val_eval, 'pred_score', invert=False)
        
        if val_rho > best_val_rho:
            best_val_rho = val_rho
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            break
            
    # Load best model
    model.load_state_dict(best_state)
    return model

# ========================================
# RUN WALK-FORWARD VALIDATION
# ========================================
print("\n" + "=" * 60)
print("RUNNING WALK-FORWARD VALIDATION ON DEEP SEQUENCES")
print("=" * 60)

for cfg in folds_config:
    fold_idx = cfg['fold']
    tr_m, val_m, te_m = cfg['train_months'], cfg['val_months'], cfg['test_months']
    
    print(f"\n--- FOLD {fold_idx} ---")
    print(f"  Train: {tr_m[0]} -> {tr_m[-1]}")
    print(f"  Val:   {val_m[0]} -> {val_m[-1]}")
    print(f"  Test:  {te_m[0]} -> {te_m[-1]}")
    
    # Split indexes
    tr_mask = np.isin(ym_seqs, tr_m)
    val_mask = np.isin(ym_seqs, val_m)
    te_mask = np.isin(ym_seqs, te_m)
    
    X_tr, y_tr, qids_tr = X_seqs[tr_mask], y_seqs[tr_mask], qid_seqs[tr_mask]
    X_val, y_val, qids_val = X_seqs[val_mask], y_seqs[val_mask], qid_seqs[val_mask]
    X_te, y_te, qids_te = X_seqs[te_mask], y_seqs[te_mask], qid_seqs[te_mask]
    
    df_val = df_eval_all[val_mask].copy()
    df_te = df_eval_all[te_mask].copy()
    
    print(f"  Data sizes: Train={X_tr.shape[0]:,}, Val={X_val.shape[0]:,}, Test={X_te.shape[0]:,}")
    
    train_dataset = DailySequenceDataset(X_tr, y_tr, qids_tr)
    val_dataset = DailySequenceDataset(X_val, y_val, qids_val)
    test_dataset = DailySequenceDataset(X_te, y_te, qids_te)
    
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    
    # Train separate models for Long and Short (or use single model predicting return, then sorting both ways)
    # To mirror XGBoost, we can train one highly generic sequence regressor to predict return, 
    # then evaluate it in both Long (sorting high to low) and Short (sorting low to high) configurations.
    print("  Training Temporal sequence model...")
    model = train_model(train_loader, val_loader, df_val, len(feature_cols), epochs=80, patience=10)
    
    # Predict on test set
    model.eval()
    test_preds = []
    with torch.no_grad():
        for X_batch, _, _ in test_loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch)
            test_preds.extend(pred.cpu().numpy())
            
    # For short, we invert predictions
    long_scores = np.array(test_preds)
    short_scores = -long_scores
    
    metrics = evaluate_ranking_performance(df_te, long_scores, short_scores)
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

print("\n" + "=" * 60)
print("AGGREGATE WALK-FORWARD VALIDATION RESULTS")
print("=" * 60)
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
print("=" * 60)

# ========================================
# TRAIN PRODUCTION MODEL ON ALL HISTORICAL DATA
# ========================================
print("\n" + "=" * 60)
print("TRAINING FINAL PRODUCTION MODEL")
print("=" * 60)

scaler = StandardScaler(with_mean=False, with_std=False)
with open(SCALER_PATH, 'wb') as f:
    pickle.dump(scaler, f)

prod_train_months = unique_months[:-2]
prod_val_months = unique_months[-2:]

print(f"  Production Train: {prod_train_months[0]} -> {prod_train_months[-1]}")
print(f"  Production Val:   {prod_val_months[0]} -> {prod_val_months[-1]}")

prod_tr_mask = np.isin(ym_seqs, prod_train_months)
prod_val_mask = np.isin(ym_seqs, prod_val_months)

X_prod_tr, y_prod_tr, qids_prod_tr = X_seqs[prod_tr_mask], y_seqs[prod_tr_mask], qid_seqs[prod_tr_mask]
X_prod_val, y_prod_val, qids_prod_val = X_seqs[prod_val_mask], y_seqs[prod_val_mask], qid_seqs[prod_val_mask]

df_prod_val = df_eval_all[prod_val_mask].copy()

prod_train_dataset = DailySequenceDataset(X_prod_tr, y_prod_tr, qids_prod_tr)
prod_val_dataset = DailySequenceDataset(X_prod_val, y_prod_val, qids_prod_val)

prod_train_loader = DataLoader(prod_train_dataset, batch_size=512, shuffle=True)
prod_val_loader = DataLoader(prod_val_dataset, batch_size=1024, shuffle=False)

print("  Training Production sequence model...")
prod_model = train_model(prod_train_loader, prod_val_loader, df_prod_val, len(feature_cols), epochs=80, patience=10)

# Save the PyTorch model checkpoint
torch.save(prod_model.state_dict(), MODEL_PATH)
print(f"    Saved Production Model Checkpoint: {MODEL_PATH}")

# ========================================
# SAVE METADATA
# ========================================
metadata = {
    'features': feature_cols,
    'num_features': len(feature_cols),
    'data_source': 'upstox_5y_daily',
    'data_file': DATA_FILE,
    'total_sequences': int(X_seqs.shape[0]),
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
        'd_model': 32,
        'nhead': 2,
        'num_layers': 1,
    },
    'trained_at': datetime.now().isoformat(),
}

with open(META_PATH, 'w') as f:
    json.dump(metadata, f, indent=2)

print("\n" + "=" * 60)
print("DAILY MACRO DEEP TRANSFORMER TRAINING COMPLETE")
print(f"  Model Checkpoint : {MODEL_PATH}")
print(f"  Metadata         : {META_PATH}")
print(f"  Scaler           : {SCALER_PATH}")
print("=" * 60)
print()
