"""
Run pre-retrieval diagnostics on the extracted embeddings:
1. Linear probe (Ridge regression) Spearman Rho comparison (Probe A vs B vs C)
2. Distance-vs-outcome slope and p-value check
"""
import os, sys, json
import numpy as np

sys.path.append(os.getcwd())
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

from scripts.transformer.pretrain_contrastive_v20 import valid_decision_timestamps, load_panel, chrono_split
from scripts.transformer.multitask_utils_v20 import train_eval_linear_probes, distance_vs_outcome_diagnostic

P = 'data/transformer_panel_v20'

def main():
    print("=" * 70)
    print("RUNNING PRE-RETRIEVAL DIAGNOSTICS")
    print("=" * 70)
    
    # Load panel data
    print("Loading panel data...")
    d = load_panel()
    X1 = d['X_1h']
    Y_ret = d['Y_ret']
    
    # Load extracted embeddings
    print("Loading extracted embeddings...")
    embeddings_path = f'{P}/embeddings_v20.npy'
    if not os.path.exists(embeddings_path):
        print(f"Error: {embeddings_path} not found. Extract embeddings first.")
        sys.exit(1)
    embeddings = np.load(embeddings_path)
    
    # Valid decision timestamps
    valid_t = valid_decision_timestamps(d)
    train_t, val_t, test_t = chrono_split(valid_t)
    
    print(f"Train split decision timestamps: {len(train_t)}")
    print(f"Val split decision timestamps:   {len(val_t)}")
    
    # Generate valid mask (T, N)
    T, N = Y_ret.shape
    valid_mask = np.zeros((T, N), dtype=bool)
    for t in valid_t:
        present = np.isfinite(X1[t, :, 0])
        valid_mask[t] = present & np.isfinite(Y_ret[t])
        
    # 1. Linear Probe Evaluation
    print("\nTraining and evaluating linear probes...")
    rho_a, rho_b, rho_c = train_eval_linear_probes(
        X1, embeddings, Y_ret, valid_mask, train_t, val_t, ridge_alpha=1.0
    )
    
    delta_rho = rho_c - rho_a
    print(f"Probe A (Hand-crafted only) Rho: {rho_a:+.6f}")
    print(f"Probe B (Embedding only) Rho:    {rho_b:+.6f}")
    print(f"Probe C (Both) Rho:             {rho_c:+.6f}")
    print(f"--> Delta Rho (C - A):          {delta_rho:+.6f}")
    
    # 2. Distance-vs-Outcome Diagnostic
    print("\nRunning distance-vs-outcome geometric regression...")
    slope, p_val = distance_vs_outcome_diagnostic(
        embeddings, Y_ret, valid_mask, train_t, val_t, sample_size=1000
    )
    
    print(f"Distance-vs-Outcome Slope:       {slope:+.6f}")
    print(f"Regression p-value:              {p_val:.4e}")
    
    # Check acceptance criteria
    print("\nAcceptance Criteria Check:")
    criteria_met = True
    
    if delta_rho >= 0.001:
        print("  [PASS] Linear-probe Delta Rho >= 0.001")
    else:
        print("  [FAIL] Linear-probe Delta Rho < 0.001 (not significant information addition)")
        criteria_met = False
        
    if slope > 0 and p_val < 0.05:
        print("  [PASS] Distance-vs-outcome slope is positive and statistically significant (p < 0.05)")
    else:
        print("  [FAIL] Distance-vs-outcome slope is not positive or not statistically significant")
        criteria_met = False
        
    if criteria_met:
        print("\nSUMMARY: Pre-retrieval diagnostics PASSED. Ready for retrieval layer.")
    else:
        print("\nSUMMARY: Pre-retrieval diagnostics FAILED. Retrieval might not yield alpha uplift.")
        
    print("=" * 70)

if __name__ == '__main__':
    main()
