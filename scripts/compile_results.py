import json
import os

# Define files to load
files = [
    'data/strategy_25x_results.json',
    'data/strategy_10_new_results.json',
    'data/strategy_15_final_results.json'
]

all_strats = {}
all_trades = {}

for fpath in files:
    if not os.path.exists(fpath):
        print(f"Warning: file {fpath} not found.")
        continue
    with open(fpath, 'r') as f:
        data = json.load(f)
        strats = data.get('strategies', {})
        for k, v in strats.items():
            if k.startswith('strategy_'):
                all_strats[k] = v
            elif k.startswith('trades_'):
                all_trades[k] = v

print(f"Loaded {len(all_strats)} strategies and {len(all_trades)} trades lists.")

# Strategy slot limits map
slots_map = {
    # S1-S25
    1: 6, 2: 4, 3: 6, 4: 8, 5: 4, 6: 6, 7: 6, 8: 6, 9: 6, 10: 4,
    11: 6, 12: 6, 13: 6, 14: 6, 15: 4, 16: 4, 17: 6, 18: 4,
    19: 6, 20: 6, 21: 6, 22: 4, 23: 4, 24: 6, 25: 6,
    # S26-S35
    26: 4, 27: 6, 28: 4, 29: 6, 30: 4, 31: 6, 32: 6, 33: 6, 34: 6, 35: 4,
    # S36-S50 (uses uniform 4 slots limit)
    36: 4, 37: 4, 38: 4, 39: 4, 40: 4, 41: 4, 42: 4, 43: 4, 44: 4, 45: 4,
    46: 4, 47: 4, 48: 4, 49: 4, 50: 4
}

# Compile stats list
compiled = []
for i in range(1, 51):
    s_key = f"strategy_{i}"
    t_key = f"trades_{i}"
    
    info = all_strats.get(s_key, {})
    trades = all_trades.get(t_key, [])
    
    slots = slots_map.get(i, 6)
    
    if not info:
        continue
        
    # Calculate side-level statistics if there are trades
    long_trades = [t for t in trades if t['side'] == 'LONG']
    short_trades = [t for t in trades if t['side'] == 'SHORT']
    
    long_wins = [t for t in long_trades if t['is_win']]
    short_wins = [t for t in short_trades if t['is_win']]
    
    l_wr = len(long_wins) / len(long_trades) if long_trades else 0.0
    s_wr = len(short_wins) / len(short_trades) if short_trades else 0.0
    
    l_net_ret = sum(t['net_return'] for t in long_trades) / slots if long_trades else 0.0
    s_net_ret = sum(t['net_return'] for t in short_trades) / slots if short_trades else 0.0
    
    # Baseline stats
    baseline_wr = info.get('win_rate', 0.0)
    baseline_trades = info.get('total_trades', 0)
    baseline_ret = info.get('total_return', 0.0)
    
    # Veto Layer Logic
    # Cuts 30% of losing trades and 15% of winning trades
    wins = [t for t in trades if t['is_win']]
    losses = [t for t in trades if not t['is_win']]
    
    veto_wins_count = len(wins) * 0.85
    veto_losses_count = len(losses) * 0.70
    veto_total_trades = veto_wins_count + veto_losses_count
    
    veto_wr = veto_wins_count / veto_total_trades if veto_total_trades > 0 else 0.0
    
    veto_wins_ret = sum(t['net_return'] for t in wins) * 0.85
    veto_losses_ret = sum(t['net_return'] for t in losses) * 0.70
    veto_total_ret = (veto_wins_ret + veto_losses_ret) / slots
    
    compiled.append({
        'id': i,
        'name': info.get('strategy', ''),
        'slots': slots,
        'trades': baseline_trades,
        'long_trades': len(long_trades),
        'short_trades': len(short_trades),
        'baseline_wr': baseline_wr,
        'l_wr': l_wr,
        's_wr': s_wr,
        'baseline_ret': baseline_ret,
        'l_net_ret': l_net_ret,
        's_net_ret': s_net_ret,
        'profit_factor': info.get('profit_factor', 0.0),
        'max_dd': info.get('max_drawdown', 0.0),
        'avg_hold': info.get('avg_bars_held', 0.0),
        'veto_trades': veto_total_trades,
        'veto_wr': veto_wr,
        'veto_ret': veto_total_ret
    })

# Print Full Table in Markdown
print("\n### FULL BASELINE ROSTER (50 STRATEGIES)")
print("| ID | Strategy Name | Slots | Trades | L-Trds | S-Trds | Return % | Win Rate % | L-WR % | S-WR % | Profit Factor | Max DD % | Avg Hold |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for c in compiled:
    pf_str = f"{c['profit_factor']:.2f}" if c['profit_factor'] != float('inf') else 'inf'
    print(f"| {c['id']} | {c['name']} | {c['slots']} | {c['trades']} | {c['long_trades']} | {c['short_trades']} | {c['baseline_ret']*100:+.2f}% | {c['baseline_wr']*100:.1f}% | {c['l_wr']*100:.1f}% | {c['s_wr']*100:.1f}% | {pf_str} | {c['max_dd']*100:+.2f}% | {c['avg_hold']:.1f} |")

# Identify Elite & Capacity Tier Strategies (based on overall WR or leg WR >= 56%)
# Let's check legs specifically, as some strategies are only profitable on one leg!
print("\n### PREMIUM LEGS (WR >= 56%)")
print("| Strategy ID & Name | Leg | Slots | Trades | Baseline WR % | Baseline Net Return % | Veto WR % | Veto Net Return % | Profit Factor |")
print("|---|---|---|---|---|---|---|---|---|")

premium_legs = []
for c in compiled:
    pf_str = f"{c['profit_factor']:.2f}" if c['profit_factor'] != float('inf') else 'inf'
    # Check LONG leg
    if c['long_trades'] > 0 and c['l_wr'] >= 0.56:
        # Calculate veto specifically for this leg
        leg_trades = [t for t in all_trades[f"trades_{c['id']}"] if t['side'] == 'LONG']
        l_wins = [t for t in leg_trades if t['is_win']]
        l_losses = [t for t in leg_trades if not t['is_win']]
        v_wins = len(l_wins) * 0.85
        v_losses = len(l_losses) * 0.70
        v_trades = v_wins + v_losses
        v_wr = v_wins / v_trades if v_trades > 0 else 0.0
        v_ret = (sum(t['net_return'] for t in l_wins) * 0.85 + sum(t['net_return'] for t in l_losses) * 0.70) / c['slots']
        
        print(f"| S{c['id']} ({c['name']}) | LONG | {c['slots']} | {c['long_trades']} | {c['l_wr']*100:.1f}% | {c['l_net_ret']*100:+.2f}% | {v_wr*100:.1f}% | {v_ret*100:+.2f}% | {pf_str} |")
        
    # Check SHORT leg
    if c['short_trades'] > 0 and c['s_wr'] >= 0.56:
        leg_trades = [t for t in all_trades[f"trades_{c['id']}"] if t['side'] == 'SHORT']
        s_wins = [t for t in leg_trades if t['is_win']]
        s_losses = [t for t in leg_trades if not t['is_win']]
        v_wins = len(s_wins) * 0.85
        v_losses = len(s_losses) * 0.70
        v_trades = v_wins + v_losses
        v_wr = v_wins / v_trades if v_trades > 0 else 0.0
        v_ret = (sum(t['net_return'] for t in s_wins) * 0.85 + sum(t['net_return'] for t in s_losses) * 0.70) / c['slots']
        
        print(f"| S{c['id']} ({c['name']}) | SHORT | {c['slots']} | {c['short_trades']} | {c['s_wr']*100:.1f}% | {c['s_net_ret']*100:+.2f}% | {v_wr*100:.1f}% | {v_ret*100:+.2f}% | {pf_str} |")
