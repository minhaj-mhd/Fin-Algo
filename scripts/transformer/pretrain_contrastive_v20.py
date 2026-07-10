"""
Pre-train the Dual-Resolution Cross-Sectional Transformer using InfoNCE contrastive learning
and smoothness regularization.
"""
import os, sys, json, argparse, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.append(os.getcwd())
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from scripts.transformer.model import DualResCSTransformer
from scripts.transformer.multitask_utils_v20 import (
    build_contrastive_masks,
    masked_infonce_loss,
    compute_smoothness_loss
)

P = 'data/transformer_panel_v20'
L1, L2 = 30, 60
EMBARGO = 30
SEED = 42

# Define the wrapper subclass that can return embeddings
class ContrastiveTransformer(DualResCSTransformer):
    def forward(self, x1h, x15m, slot1h, slot15m, macro, sector_ids, pad_mask, return_embeddings=False):
        B, N, L1, F = x1h.shape
        L2 = x15m.size(2)
        se1 = self.slot_emb_1h(slot1h).unsqueeze(1).expand(B, N, L1, -1).reshape(B * N, L1, -1)
        se2 = self.slot_emb_15m(slot15m).unsqueeze(1).expand(B, N, L2, -1).reshape(B * N, L2, -1)
        h1 = self.enc_1h(x1h.reshape(B * N, L1, F), se1).reshape(B, N, -1)
        h2 = self.enc_15m(x15m.reshape(B * N, L2, F), se2).reshape(B, N, -1)
        tok = self.token_proj(torch.cat([h1, h2], dim=-1))            # (B,N,d)
        tok = tok + self.sector_emb(sector_ids).unsqueeze(0)          # broadcast sector
        tok = tok + self.macro_mlp(macro).unsqueeze(1)                # broadcast macro context
        z = self.cross(tok, src_key_padding_mask=pad_mask)           # cross-sectional attention (B,N,d)
        if return_embeddings:
            return z
        return self.head(z).squeeze(-1)                               # (B,N)

def load_panel():
    d = {}
    for k in ['X_1h', 'Y_ret', 'slot_1h', 'end15', 'ts_1h', 'date_idx', 'macro',
              'slot_15m', 'sector_ids', 'ts_15m']:
        d[k] = np.load(f'{P}/{k}.npy')
    d['X_15m'] = np.load(f'{P}/X_15m.npy')
    d['meta'] = json.load(open(f'{P}/meta.json'))
    # Load regime labels
    d['regimes'] = np.load(f'{P}/regime_labels.npy')
    return d

class PretrainDataset(Dataset):
    def __init__(self, d, t_indices):
        self.X1, self.X15 = d['X_1h'], d['X_15m']
        self.s1, self.s15, self.end15 = d['slot_1h'], d['slot_15m'], d['end15']
        self.macro, self.date_idx = d['macro'], d['date_idx']
        self.ts_1h = d['ts_1h']
        self.regimes = d['regimes']
        self.t_idx = t_indices

    def __len__(self):
        return len(self.t_idx)

    def __getitem__(self, i):
        t = int(self.t_idx[i])
        e = int(self.end15[t])
        x1 = np.nan_to_num(self.X1[t - L1 + 1:t + 1])
        x15 = np.nan_to_num(self.X15[e - L2 + 1:e + 1])
        x1 = np.transpose(x1, (1, 0, 2))
        x15 = np.transpose(x15, (1, 0, 2))
        s1 = self.s1[t - L1 + 1:t + 1].astype(np.int64)
        s15 = self.s15[e - L2 + 1:e + 1].astype(np.int64)
        macro = np.nan_to_num(self.macro[int(self.date_idx[t])])
        present = np.isfinite(self.X1[t, :, 0])
        reg = self.regimes[t]
        
        # Grid index of timestamp (its index in ts_1h)
        grid_idx = t
        
        return (x1.astype(np.float32), x15.astype(np.float32), s1, s15,
                macro.astype(np.float32), present.astype(np.bool_), 
                np.full(172, self.ts_1h[t], dtype=np.int64),
                np.full(172, reg, dtype=np.int32),
                np.full(172, grid_idx, dtype=np.int64))

def collate(batch):
    x1, x15, s1, s15, macro, present, ts, regimes, grid_idx = zip(*batch)
    f = lambda a: torch.from_numpy(np.stack(a))
    return (f(x1), f(x15), f(s1), f(s15), f(macro), f(present), f(ts), f(regimes), f(grid_idx))

def chrono_split(ts, embargo=EMBARGO):
    n = len(ts)
    i_tr, i_va = int(n * 0.70), int(n * 0.85)
    return ts[:i_tr], ts[i_tr + embargo:i_va], ts[i_va + embargo:]

def valid_decision_timestamps(d):
    T = d['X_1h'].shape[0]
    finite_label = np.isfinite(d['Y_ret']).sum(axis=1) > 0
    ok = np.zeros(T, dtype=bool)
    for t in range(T):
        if t < L1 - 1 or not finite_label[t]:
            continue
        e = int(d['end15'][t])
        if e < L2 - 1 or d['date_idx'][t] < 0:
            continue
        ok[t] = True
    return np.where(ok)[0]

def compute_stock_correlation(X_15m, end15, train_t_indices, return_idx):
    """
    Computes stock-to-stock Pearson correlation matrix using 15m returns during the training period.
    """
    # Extract corresponding 15m indices
    train_end15 = end15[train_t_indices].astype(np.int32)
    # Gather 15m return slices
    ret_15m = X_15m[train_end15, :, return_idx] # (T_train, N)
    ret_15m_clean = np.nan_to_num(ret_15m, nan=0.0)
    corr = np.corrcoef(ret_15m_clean.T)
    corr = np.nan_to_num(corr, nan=0.0)
    return corr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--temp', type=float, default=0.07)
    parser.add_argument('--lam', type=float, default=0.01) # smoothness weight
    args = parser.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading panel data...")
    d = load_panel()
    
    valid_t = valid_decision_timestamps(d)
    print(f"Total valid decision timestamps: {len(valid_t)}")
    
    # Chronological split on valid timestamps
    train_t, val_t, test_t = chrono_split(valid_t)
    print(f"Splits: Train {len(train_t)}, Val {len(val_t)}, Test {len(test_t)}")
    
    # Compute stock correlation matrix on train split
    meta = d['meta']
    return_idx = meta['features'].index('Return')
    print("Computing stock-to-stock correlation matrix on training split...")
    corr_matrix_np = compute_stock_correlation(d['X_15m'], d['end15'], train_t, return_idx)
    corr_matrix = torch.from_numpy(corr_matrix_np).float().to(device)
    
    # Setup datasets
    train_dataset = PretrainDataset(d, train_t)
    val_dataset = PretrainDataset(d, val_t)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate, drop_last=False)
    
    # Setup model
    n_feat = len(meta['features'])
    n_macro = len(meta['macro_cols'])
    n_sectors = len(meta['sectors'])
    
    # sector ids mapping
    sector_ids = torch.from_numpy(d['sector_ids']).long().to(device)
    
    model = ContrastiveTransformer(
        n_feat=n_feat,
        n_macro=n_macro,
        n_sectors=n_sectors,
        n_slots_1h=meta['n_slots_1h'],
        n_slots_15m=meta['n_slots_15m'],
        d_model=64,
        t_layers=2,
        c_layers=2,
        nhead=4,
        dropout=0.1
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_val_loss = float('inf')
    ckpt_path = 'encoder_pretrained_v20.ckpt'
    
    print("\nStarting contrastive pre-training...")
    for epoch in range(args.epochs):
        model.train()
        train_loss_sum = 0.0
        train_contrast_sum = 0.0
        train_smooth_sum = 0.0
        n_batches = 0
        
        t0 = time.time()
        for batch in train_loader:
            x1h, x15m, slot1h, slot15m, macro, present, ts, regimes, grid_idx = batch
            
            x1h = x1h.to(device)
            x15m = x15m.to(device)
            slot1h = slot1h.to(device)
            slot15m = slot15m.to(device)
            macro = macro.to(device)
            present = present.to(device)
            
            # Forward pass: get embeddings
            # pad_mask = ~present
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                embeddings = model(x1h, x15m, slot1h, slot15m, macro, sector_ids, ~present, return_embeddings=True) # (B, N, d)
                
            # Flatten across batch and tickers
            B, N, D = embeddings.shape
            emb_flat = embeddings.reshape(B * N, D)
            present_flat = present.reshape(B * N)
            ts_flat = ts.reshape(B * N).to(device)
            regimes_flat = regimes.reshape(B * N).to(device)
            grid_idx_flat = grid_idx.reshape(B * N).to(device)
            
            # Ticker index for each sample
            tickers_flat = torch.arange(N, device=device).repeat(B)
            # Sector ID for each sample
            sectors_flat = sector_ids.repeat(B)
            
            # Filter only present samples
            emb_flat = emb_flat[present_flat]
            ts_flat = ts_flat[present_flat]
            tickers_flat = tickers_flat[present_flat]
            sectors_flat = sectors_flat[present_flat]
            regimes_flat = regimes_flat[present_flat]
            grid_idx_flat = grid_idx_flat[present_flat]
            
            if len(emb_flat) < 2:
                continue
                
            # Build contrastive masks
            pos_mask, neg_mask = build_contrastive_masks(
                ts_flat, tickers_flat, sectors_flat, regimes_flat, corr_matrix, grid_idx_flat, device
            )
            
            # Calculate losses
            loss_contrastive = masked_infonce_loss(emb_flat, pos_mask, neg_mask, temp=args.temp)
            
            # Smoothness loss: consecutive same-stock pairs
            # pos_same_stock_t1 is the first component of pos_mask
            same_stock = (tickers_flat.unsqueeze(1) == tickers_flat.unsqueeze(0))
            consecutive_t = (torch.abs(grid_idx_flat.unsqueeze(1) - grid_idx_flat.unsqueeze(0)) == 1)
            consecutive_mask = same_stock & consecutive_t & (~torch.eye(len(emb_flat), dtype=torch.bool, device=device))
            
            loss_smooth = compute_smoothness_loss(emb_flat, consecutive_mask)
            
            loss = loss_contrastive + args.lam * loss_smooth
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item()
            train_contrast_sum += loss_contrastive.item()
            train_smooth_sum += loss_smooth.item()
            n_batches += 1
            
        scheduler.step()
        
        # Validation epoch
        model.eval()
        val_loss_sum = 0.0
        val_contrast_sum = 0.0
        val_smooth_sum = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                x1h, x15m, slot1h, slot15m, macro, present, ts, regimes, grid_idx = batch
                
                x1h = x1h.to(device)
                x15m = x15m.to(device)
                slot1h = slot1h.to(device)
                slot15m = slot15m.to(device)
                macro = macro.to(device)
                present = present.to(device)
                
                embeddings = model(x1h, x15m, slot1h, slot15m, macro, sector_ids, ~present, return_embeddings=True)
                
                B, N, D = embeddings.shape
                emb_flat = embeddings.reshape(B * N, D)
                present_flat = present.reshape(B * N)
                ts_flat = ts.reshape(B * N).to(device)
                regimes_flat = regimes.reshape(B * N).to(device)
                grid_idx_flat = grid_idx.reshape(B * N).to(device)
                
                tickers_flat = torch.arange(N, device=device).repeat(B)
                sectors_flat = sector_ids.repeat(B)
                
                emb_flat = emb_flat[present_flat]
                ts_flat = ts_flat[present_flat]
                tickers_flat = tickers_flat[present_flat]
                sectors_flat = sectors_flat[present_flat]
                regimes_flat = regimes_flat[present_flat]
                grid_idx_flat = grid_idx_flat[present_flat]
                
                if len(emb_flat) < 2:
                    continue
                    
                pos_mask, neg_mask = build_contrastive_masks(
                    ts_flat, tickers_flat, sectors_flat, regimes_flat, corr_matrix, grid_idx_flat, device
                )
                
                loss_contrastive = masked_infonce_loss(emb_flat, pos_mask, neg_mask, temp=args.temp)
                
                same_stock = (tickers_flat.unsqueeze(1) == tickers_flat.unsqueeze(0))
                consecutive_t = (torch.abs(grid_idx_flat.unsqueeze(1) - grid_idx_flat.unsqueeze(0)) == 1)
                consecutive_mask = same_stock & consecutive_t & (~torch.eye(len(emb_flat), dtype=torch.bool, device=device))
                
                loss_smooth = compute_smoothness_loss(emb_flat, consecutive_mask)
                
                loss = loss_contrastive + args.lam * loss_smooth
                
                val_loss_sum += loss.item()
                val_contrast_sum += loss_contrastive.item()
                val_smooth_sum += loss_smooth.item()
                val_batches += 1
                
        avg_train = train_loss_sum / n_batches if n_batches else 0.0
        avg_val = val_loss_sum / val_batches if val_batches else 0.0
        
        print(f"Epoch {epoch+1:02d}/{args.epochs} | Train Loss: {avg_train:.4f} (InfoNCE: {train_contrast_sum/n_batches:.4f}, Smooth: {train_smooth_sum/n_batches:.4f}) | "
              f"Val Loss: {avg_val:.4f} (InfoNCE: {val_contrast_sum/val_batches:.4f}, Smooth: {val_smooth_sum/val_batches:.4f}) | Time: {time.time()-t0:.1f}s")
              
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), ckpt_path)
            print(f"  --> Saved new best checkpoint to {ckpt_path}")

    print(f"\nContrastive pre-training finished. Best validation loss: {best_val_loss:.4f}")

if __name__ == '__main__':
    main()
