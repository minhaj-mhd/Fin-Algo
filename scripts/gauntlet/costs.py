import numpy as np

def verify_cost_invariants(raw_returns: np.ndarray, net_returns: np.ndarray, cost: float):
    """
    Verifies that cost calculation is mathematically consistent and correct.
    Catches cost-sign bugs (like the historical TBM Short bug where cost was added instead of subtracted).
    """
    if len(raw_returns) == 0:
        return
        
    raw = np.asarray(raw_returns, dtype=float)
    net = np.asarray(net_returns, dtype=float)
    
    # Invariant 1: Cost must reduce return (net < raw)
    if cost > 0:
        assert (net < raw).all(), "Cost invariant violation: net returns must be strictly less than raw returns"
        
    # Invariant 2: Elementwise difference must equal exactly -cost
    diffs = net - raw
    assert np.allclose(diffs, -cost, atol=1e-9), f"Cost invariant violation: net - raw difference does not match -{cost}"
    
    # Invariant 3: Median difference is exactly -cost
    median_diff = np.median(diffs)
    assert np.isclose(median_diff, -cost, atol=1e-9), f"Cost invariant violation: median difference {median_diff} != -{cost}"
    
    # Invariant 4: Net bps is raw bps minus cost bps
    raw_bps = float(raw.mean()) * 10000
    net_bps = float(net.mean()) * 10000
    cost_bps = cost * 10000
    assert np.isclose(net_bps, raw_bps - cost_bps, atol=1e-9), f"Cost invariant violation: net bps {net_bps} != raw bps {raw_bps} - cost bps {cost_bps}"
