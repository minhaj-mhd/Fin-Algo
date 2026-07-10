"""
Extract embeddings for the full panel using the pre-trained frozen encoder checkpoint.
Saves the extracted embeddings as embeddings_v20.npy in data/transformer_panel_v20/.
"""
import os, sys, json, argparse, time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.getcwd())
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from scripts.transformer.model import DualResCSTransformer
from scripts.transformer.pretrain_contrastive_v20 import PretrainDataset, collate, load_panel

P = 'data/transformer_panel_v20'
L1, L2 = 30, 60

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default='encoder_pretrained_v20.ckpt')
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if not os.path.exists(args.ckpt):
        print(f"Error: Checkpoint {args.ckpt} not found. Please run pretrain_contrastive_v20.py first.")
        sys.exit(1)
        
    print("Loading panel data...")
    d = load_panel()
    
    meta = d['meta']
    T = d['X_1h'].shape[0]
    N = d['X_1h'].shape[1]
    
    # Identify which timestamps are embeddable
    # We need t >= L1 - 1 and end15[t] >= L2 - 1 and date_idx[t] >= 0
    embeddable_mask = np.zeros(T, dtype=bool)
    for t in range(T):
        if t < L1 - 1:
            continue
        e = int(d['end15'][t])
        if e < L2 - 1 or d['date_idx'][t] < 0:
            continue
        embeddable_mask[t] = True
        
    embeddable_indices = np.where(embeddable_mask)[0]
    print(f"Total timestamps in panel: {T}")
    print(f"Embeddable timestamps: {len(embeddable_indices)}")
    
    # Setup model
    n_feat = len(meta['features'])
    n_macro = len(meta['macro_cols'])
    n_sectors = len(meta['sectors'])
    
    sector_ids = torch.from_numpy(d['sector_ids']).long().to(device)
    
    model = DualResCSTransformer(
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
    )
    
    # Load state dict
    state_dict = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Setup dataset/dataloader for embeddable indices
    dataset = PretrainDataset(d, embeddable_indices)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate, drop_last=False)
    
    # Placeholder for all embeddings (T, N, 64)
    all_embeddings = np.zeros((T, N, 64), dtype=np.float32)
    
    print("\nExtracting embeddings...")
    t0 = time.time()
    
    # Subclass forward pass to return embeddings
    def get_embeddings_forward(model, x1h, x15m, slot1h, slot15m, macro, sector_ids, pad_mask):
        B, N, L1, F = x1h.shape
        L2 = x15m.size(2)
        se1 = model.slot_emb_1h(slot1h).unsqueeze(1).expand(B, N, L1, -1).reshape(B * N, L1, -1)
        se2 = model.slot_emb_15m(slot15m).unsqueeze(1).expand(B, N, L2, -1).reshape(B * N, L2, -1)
        h1 = model.enc_1h(x1h.reshape(B * N, L1, F), se1).reshape(B, N, -1)
        h2 = model.enc_15m(x15m.reshape(B * N, L2, F), se2).reshape(B, N, -1)
        tok = model.token_proj(torch.cat([h1, h2], dim=-1))
        tok = tok + model.sector_emb(sector_ids).unsqueeze(0)
        tok = tok + model.macro_mlp(macro).unsqueeze(1)
        z = model.cross(tok, src_key_padding_mask=pad_mask)
        return z
        
    with torch.no_grad():
        for i, batch in enumerate(loader):
            x1h, x15m, slot1h, slot15m, macro, present, ts, regimes, grid_idx = batch
            
            x1h = x1h.to(device)
            x15m = x15m.to(device)
            slot1h = slot1h.to(device)
            slot15m = slot15m.to(device)
            macro = macro.to(device)
            present = present.to(device)
            
            z = get_embeddings_forward(model, x1h, x15m, slot1h, slot15m, macro, sector_ids, ~present)
            
            # Map back to all_embeddings using the grid_idx
            z_np = z.cpu().numpy() # (B, N, 64)
            grid_idx_np = grid_idx.cpu().numpy() # (B, N)
            
            for b in range(z_np.shape[0]):
                t_val = int(grid_idx_np[b, 0])
                # Only keep embeddings for present tickers, other tickers remain 0
                pres = present[b].cpu().numpy()
                all_embeddings[t_val, pres] = z_np[b, pres]
                
            if (i + 1) % 20 == 0:
                print(f"Processed {i+1} batches... Time: {time.time()-t0:.1f}s")
                
    # Save embeddings
    out_path = f'{P}/embeddings_v20.npy'
    np.save(out_path, all_embeddings)
    print(f"\nSaved embeddings to {out_path} with shape {all_embeddings.shape}")
    print(f"Total time: {time.time()-t0:.1f}s")
    print("=" * 70)

if __name__ == '__main__':
    main()
