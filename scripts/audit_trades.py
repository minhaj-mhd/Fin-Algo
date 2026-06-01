"""Deep audit of trade-level P&L across all premium strategies."""
import json
import statistics

with open('data/premium_trades_results.json') as f:
    data = json.load(f)

strats = data['strategies']

premium_ids = [2, 12, 18, 19, 23, 33, 35]

for sid in premium_ids:
    info = strats.get(f'strategy_{sid}', {})
    trades = strats.get(f'trades_{sid}', [])
    
    if not trades:
        print(f"\n{'='*70}")
        print(f"S{sid}: NO TRADE DATA AVAILABLE")
        continue
    
    print(f"\n{'='*70}")
    print(f"S{sid} ({info.get('strategy','?')}) — {len(trades)} trades")
    print(f"{'='*70}")
    
    gross_rets = [t['gross_return']*100 for t in trades]
    net_rets = [t['net_return']*100 for t in trades]
    
    wins = [t for t in trades if t['is_win']]
    losses = [t for t in trades if not t['is_win']]
    
    win_gross = [t['gross_return']*100 for t in wins]
    loss_gross = [t['gross_return']*100 for t in losses]
    win_net = [t['net_return']*100 for t in wins]
    loss_net = [t['net_return']*100 for t in losses]
    
    print(f"  Win Rate: {len(wins)}/{len(trades)} = {len(wins)/len(trades)*100:.1f}%")
    print(f"  Total Gross Return: {sum(gross_rets):+.4f}%")
    print(f"  Total Net Return:   {sum(net_rets):+.4f}%")
    print(f"  Transaction Cost Drag: {sum(gross_rets)-sum(net_rets):+.4f}% ({(sum(gross_rets)-sum(net_rets))/len(trades)*100:.2f} bps/trade)")
    
    if win_gross:
        print(f"\n  WINNERS ({len(wins)}):")
        print(f"    Avg gross: {statistics.mean(win_gross):+.4f}%")
        print(f"    Avg net:   {statistics.mean(win_net):+.4f}%")
        print(f"    Max win:   {max(win_gross):+.4f}%")
        print(f"    Min win:   {min(win_gross):+.4f}%")
    
    if loss_gross:
        print(f"\n  LOSERS ({len(losses)}):")
        print(f"    Avg gross: {statistics.mean(loss_gross):+.4f}%")
        print(f"    Avg net:   {statistics.mean(loss_net):+.4f}%")
        print(f"    Max loss:  {min(loss_gross):+.4f}%")
        print(f"    Min loss:  {max(loss_gross):+.4f}%")
    
    # Exit reason breakdown
    print(f"\n  EXIT REASON BREAKDOWN:")
    reasons = {}
    for t in trades:
        r = t.get('exit_reason', 'UNKNOWN')
        if r not in reasons:
            reasons[r] = {'count': 0, 'gross_pnl': 0, 'net_pnl': 0, 'wins': 0}
        reasons[r]['count'] += 1
        reasons[r]['gross_pnl'] += t['gross_return']*100
        reasons[r]['net_pnl'] += t['net_return']*100
        if t['is_win']:
            reasons[r]['wins'] += 1
    
    for r, v in sorted(reasons.items(), key=lambda x: -x[1]['count']):
        wr = v['wins']/v['count']*100 if v['count'] > 0 else 0
        print(f"    {r:20s}: {v['count']:3d} trades, WR={wr:5.1f}%, gross={v['gross_pnl']:+.4f}%, net={v['net_pnl']:+.4f}%")
    
    # Side breakdown
    longs = [t for t in trades if t['side'] == 'LONG']
    shorts = [t for t in trades if t['side'] == 'SHORT']
    if longs:
        l_wins = sum(1 for t in longs if t['is_win'])
        l_ret = sum(t['net_return']*100 for t in longs)
        print(f"\n  LONG leg:  {len(longs)} trades, WR={l_wins/len(longs)*100:.1f}%, net={l_ret:+.4f}%")
    if shorts:
        s_wins = sum(1 for t in shorts if t['is_win'])
        s_ret = sum(t['net_return']*100 for t in shorts)
        print(f"  SHORT leg: {len(shorts)} trades, WR={s_wins/len(shorts)*100:.1f}%, net={s_ret:+.4f}%")
    
    # Bars held analysis
    bars = [t['bars_held'] for t in trades]
    print(f"\n  HOLD DURATION: avg={statistics.mean(bars):.1f} bars, min={min(bars)}, max={max(bars)}")
    
    # Win rate by bars held
    by_bars = {}
    for t in trades:
        b = t['bars_held']
        if b not in by_bars:
            by_bars[b] = {'count': 0, 'wins': 0, 'pnl': 0}
        by_bars[b]['count'] += 1
        if t['is_win']:
            by_bars[b]['wins'] += 1
        by_bars[b]['pnl'] += t['net_return']*100
    
    print(f"  BY BARS HELD:")
    for b in sorted(by_bars.keys()):
        v = by_bars[b]
        wr = v['wins']/v['count']*100
        print(f"    {b} bars: {v['count']:3d} trades, WR={wr:5.1f}%, pnl={v['pnl']:+.4f}%")

# Summary: what if we extended hold periods?
print(f"\n{'='*70}")
print("CRITICAL ANALYSIS: STOP LOSS vs TIME EXPIRY IMPACT")
print(f"{'='*70}")
total_sl_loss = 0
total_sl_count = 0
total_time_loss = 0
total_time_count = 0
for sid in premium_ids:
    trades = strats.get(f'trades_{sid}', [])
    for t in trades:
        if t['exit_reason'] == 'STOP_LOSS':
            total_sl_count += 1
            total_sl_loss += t['net_return']*100
        elif t['exit_reason'] == 'TIME_EXPIRY':
            total_time_count += 1
            total_time_loss += t['net_return']*100

print(f"STOP_LOSS exits: {total_sl_count} trades, total P&L={total_sl_loss:+.4f}%")
print(f"TIME_EXPIRY exits: {total_time_count} trades, total P&L={total_time_loss:+.4f}%")
print(f"Avg STOP_LOSS P&L: {total_sl_loss/total_sl_count if total_sl_count else 0:+.4f}%")
print(f"Avg TIME_EXPIRY P&L: {total_time_loss/total_time_count if total_time_count else 0:+.4f}%")
