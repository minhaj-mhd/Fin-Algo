import json
import os

files = [
    'data/strategy_25x_results.json',
    'data/strategy_10_new_results.json',
    'data/strategy_15_final_results.json'
]

all_trades = {}
for fpath in files:
    if not os.path.exists(fpath): continue
    with open(fpath, 'r') as f:
        data = json.load(f)
        for k, v in data.get('strategies', {}).items():
            if k.startswith('trades_'):
                all_trades[k] = v

slots_map = {
    2: 4, 8: 6, 10: 4, 18: 4, 19: 6, 35: 4, 36: 4, 39: 4, 42: 4
}

# The selected premium legs
premium_legs_config = [
    (2, 'SHORT'),
    (8, 'SHORT'),
    (10, 'LONG'),
    (18, 'SHORT'),
    (19, 'LONG'),
    (35, 'LONG'),
    (36, 'LONG'),
    (39, 'LONG'),
    (42, 'SHORT')
]

portfolio_baseline_trades = []
portfolio_veto_trades = []

# Baseline stats
total_slots = sum(slots_map[sid] for sid, _ in premium_legs_config)
print(f"Total slots in combined portfolio: {total_slots}")

leg_summaries = []

for sid, side in premium_legs_config:
    trades = all_trades[f"trades_{sid}"]
    leg_trades = [t for t in trades if t['side'] == side]
    slots = slots_map[sid]
    
    wins = [t for t in leg_trades if t['is_win']]
    losses = [t for t in leg_trades if not t['is_win']]
    
    baseline_wr = len(wins) / len(leg_trades) if leg_trades else 0.0
    baseline_net_ret = sum(t['net_return'] for t in leg_trades)
    baseline_net_ret_slotted = baseline_net_ret / slots
    
    # Veto layer
    # 15% wins cut, 30% losses cut
    v_wins_cnt = len(wins) * 0.85
    v_losses_cnt = len(losses) * 0.70
    v_total_cnt = v_wins_cnt + v_losses_cnt
    v_wr = v_wins_cnt / v_total_cnt if v_total_cnt > 0 else 0.0
    
    v_net_ret = sum(t['net_return'] for t in wins) * 0.85 + sum(t['net_return'] for t in losses) * 0.70
    v_net_ret_slotted = v_net_ret / slots
    
    leg_summaries.append({
        'sid': sid,
        'side': side,
        'trades': len(leg_trades),
        'wr': baseline_wr,
        'ret_unslotted': baseline_net_ret,
        'ret_slotted': baseline_net_ret_slotted,
        'v_trades': v_total_cnt,
        'v_wr': v_wr,
        'v_ret_unslotted': v_net_ret,
        'v_ret_slotted': v_net_ret_slotted
    })

# Combined stats
total_trades = sum(l['trades'] for l in leg_summaries)
total_wins = sum(l['trades'] * l['wr'] for l in leg_summaries)
combined_baseline_wr = total_wins / total_trades if total_trades > 0 else 0.0

total_v_trades = sum(l['v_trades'] for l in leg_summaries)
total_v_wins = sum(l['v_trades'] * l['v_wr'] for l in leg_summaries)
combined_veto_wr = total_v_wins / total_v_trades if total_v_trades > 0 else 0.0

combined_baseline_ret_slotted = sum(l['ret_slotted'] for l in leg_summaries)
combined_veto_ret_slotted = sum(l['v_ret_slotted'] for l in leg_summaries)

combined_baseline_ret_unslotted = sum(l['ret_unslotted'] for l in leg_summaries)
combined_veto_ret_unslotted = sum(l['v_ret_unslotted'] for l in leg_summaries)

print("\n=== CONSOLIDATED PORTFOLIO METRICS ===")
print(f"Total Portfolio Trades (Baseline): {total_trades}")
print(f"Combined Portfolio Win Rate (Baseline): {combined_baseline_wr*100:.2f}%")
print(f"Combined Net Return (Slotted, Baseline): {combined_baseline_ret_slotted*100:+.2f}%")
print(f"Combined Net Return (Unslotted, Baseline): {combined_baseline_ret_unslotted*100:+.2f}%")
print(f"\n--- After Veto Layer (30% Loss Cut, 15% Win Cut) ---")
print(f"Total Portfolio Trades (Veto): {total_v_trades:.1f}")
print(f"Combined Portfolio Win Rate (Veto): {combined_veto_wr*100:.2f}%")
print(f"Combined Net Return (Slotted, Veto): {combined_veto_ret_slotted*100:+.2f}%")
print(f"Combined Net Return (Unslotted, Veto): {combined_veto_ret_unslotted*100:+.2f}%")
