import trailing_stop_experiment as t
import json
import os

print("Running S19 and S23...")
t.simulate_strategy(19, 'S19')
t.simulate_strategy(23, 'S23')

print("Saving quick results...")
with open('../data/trailing_quick.json', 'w') as f:
    json.dump({
        'strategies': {
            'trades_19': t.results['trades_19'],
            'trades_23': t.results['trades_23']
        }
    }, f)

print("Calculating returns...")
for s in [19, 23]:
    trades = t.results[f'trades_{s}']
    total_net = sum(tr['net_return'] for tr in trades) * 100
    wins = len([tr for tr in trades if tr['is_win']])
    win_rate = (wins / len(trades) * 100) if trades else 0
    print(f"S{s}: Trades={len(trades)}, WinRate={win_rate:.1f}%, TotalNet={total_net:+.2f}%")
