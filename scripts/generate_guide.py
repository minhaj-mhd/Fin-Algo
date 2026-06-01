import json
import os

def generate():
    results_path = "data/strategy_35x_results.json"
    if not os.path.exists(results_path):
        print(f"Waiting for {results_path}...")
        return
        
    with open(results_path, 'r') as f:
        data = json.load(f)
        
    strategies = data.get('strategies', {})
    
    # We will hardcode the strategy names and descriptions
    descriptions = {
        1: ("Daily Macro Gatekeeper", "Enters trades on the 15M chart only when the Daily prediction strongly agrees (e.g. Daily Rank is extremely high/low)."),
        2: ("Short-Side Specialist", "Looks exclusively for short setups on weak tickers during downtrends, targeting specific breakdown points."),
        3: ("Timeframe Divergence Fade", "Fades short-term 15M signals if they heavily contradict the 1H and Daily macro trends."),
        4: ("Score Momentum Scalper", "Scalps tight intraday moves when the raw XGBoost score is accelerating across 3 consecutive 15M bars."),
        5: ("Power Hour Sniper", "Operates only after 13:45, aligning with the Daily macro bias for the late-day volume surge."),
        6: ("Market-Neutral Pairs", "Takes simultaneous long and short positions on highly correlated tickers to capture relative alpha regardless of market direction."),
        7: ("Volatility Regime Switcher", "Dynamically adjusts the trade threshold and stop-loss widths based on the rolling 30-day market volatility percentile."),
        8: ("Opening Range Breakout + Confirmation", "Waits for the 45-minute opening range to establish, then enters on breakouts confirmed by 1H conviction scores."),
        9: ("Conviction Spread Z-Score", "Trades when a stock's conviction score deviates wildly from the market average (Z-Score > 2.0)."),
        10: ("Quad-Timeframe Unanimous", "The strictest trend filter. Requires the 15M, 30M, 1H, and Daily timeframes to *all* rank the stock in the top percentiles."),
        11: ("Trend Pullback Entry (Buy-the-Dip)", "Buys stocks with strong 1H and Daily ranks when they experience a temporary pullback on the 15M chart."),
        12: ("High-Velocity Breakout (Speed Scanner)", "Enters when the conviction score jumps massively over a single 15M bar, acting as a speed scanner."),
        13: ("Midday Momentum Extension", "Operates strictly between 11:30 and 13:30, catching stocks re-accelerating after morning volume dies down."),
        14: ("Volume-Weighted Conviction Leader", "Weighs the AI conviction score heavily against recent volume anomalies to find institutional footprints."),
        15: ("Overextended Conviction Fade", "Fades extreme 15M convictions that have persisted for too long without price follow-through."),
        16: ("Opening Gap Fade", "Fades morning gaps that contradict the overarching Daily/1H model predictions."),
        17: ("Bollinger-Band Conviction Divergence", "Triggers when price hits the upper/lower Bollinger Bands but the AI model ranks the stock poorly (exhaustion)."),
        18: ("Volatility Expansion (ATR Breakout)", "Triggers in high volatility regimes when a stock's ATR expands massively alongside strong conviction."),
        19: ("Low-Volatility Grind (IBS Reversion)", "Only trades when market volatility is low. Uses Internal Bar Strength to buy stocks closing near their lows in slow grinding markets."),
        20: ("Mean-Reverting Conviction Spread", "Looks for temporary disconnects in the conviction spread and trades the reversion to the mean."),
        21: ("Conviction Persistence Anchor", "Requires a stock to hold the Top 3 rank for at least an hour before entering, confirming true institutional interest."),
        22: ("Momentum Spike & Hold", "Catches severe momentum spikes and holds them aggressively, allowing winners to run."),
        23: ("Pre-Close Power Hour Reversal", "Looks for exhausted trends to reverse right before the closing bell (14:15 onward) as day traders square off."),
        24: ("Blended Conviction Ensemble", "Uses a weighted formula combining Daily (10%), 1H (20%), 30M (30%), and 15M (50%) convictions."),
        25: ("Triple-Timeframe Momentum Grid", "Requires the 15M, 30M, and 1H convictions to all be rising simultaneously (Rising Momentum)."),
        26: ("The Morning Gap Reversal", "Buys massive morning gap downs (gap < -1.0%) if the 1H and Daily models are extremely bullish."),
        27: ("Consecutive Conviction Acceleration", "Buys when the 15-minute conviction score increases for 3 consecutive bars."),
        28: ("Midday Volatility Squeeze", "Buys during midday low-volatility regimes when the Internal Bar Strength is perfectly balanced, anticipating a breakout."),
        29: ("Contrarian High-Vol Fade", "Shorts stocks that break the upper Bollinger Band during chaotic high-volatility market regimes."),
        30: ("Macro Alignment Scalper", "Shorts stocks when the 15M model is bullish but the overarching 1H and Daily models are extremely bearish."),
        31: ("Extreme Z-Score Reversion", "Fades the AI model when a stock's conviction score is a massive statistical outlier (Z-Score > 2.5)."),
        32: ("The Persistent Anchor", "Buys at 13:00 if a stock has remained Rank 1 for the entire preceding hour."),
        33: ("EOD Retail Liquidity Trap", "Operates after 14:15. Shorts stocks closing near the highs of the day if the AI model detects spiking short conviction."),
        34: ("Triple-Timeframe Laggard", "Buys the dip when the 15M rank falls, but the 30M and 1H macro ranks remain exceptionally strong."),
        35: ("Volatility Contraction Breakout", "Buys highly ranked stocks when their 14-period ATR is extremely compressed, catching the kinetic energy release.")
    }

    md = []
    md.append("# 🚀 Ultimate Strategy Guide: 35-Engine Trading Suite\n")
    md.append("This guide serves as your master playbook to understand the underlying logic, timeframes, and up-to-date realistic performance of all 35 intraday trading strategies running inside our automated engine.\n")
    md.append("The system uses a **15-Minute Base Timeframe**, applies predictive machine-learning (XGBoost) models to score each ticker's conviction, and executes using a rigorous **1.0% Intrabar Stop Loss** and historically-optimized holding periods.\n")
    md.append("## 📊 35-Strategy Performance Table\n")
    
    headers = ["ID", "Strategy Name", "Total Trds", "L-Trds", "S-Trds", "Return %", "Win Rate %", "L-WR %", "S-WR %", "Profit Factor", "Max DD %", "Avg Hold"]
    md.append(f"| {' | '.join(headers)} |")
    md.append(f"|{'-|-'.join(['-' * len(h) for h in headers])}|")
    
    # Build rows
    for i in range(1, 36):
        res = strategies.get(f"strategy_{i}")
        if not res:
            continue
        
        name = descriptions[i][0]
        trades = res['total_trades']
        l_trds = res['long_trades']
        s_trds = res['short_trades']
        pnl_str = f"{res['total_return']*100:+.2f}%"
        wr_str = f"{res['win_rate']*100:.1f}%"
        l_wr_str = f"{res['long_win_rate']*100:.1f}%" if l_trds > 0 else "N/A"
        s_wr_str = f"{res['short_win_rate']*100:.1f}%" if s_trds > 0 else "N/A"
        pf_str = f"{res['profit_factor']:.2f}" if res['profit_factor'] != float('inf') else "inf"
        dd_str = f"{res['max_drawdown']*100:.2f}%"
        hold_str = f"{res['avg_bars_held']:.1f}"
        
        row = f"| {i} | {name} | {trades} | {l_trds} | {s_trds} | **{pnl_str}** | {wr_str} | {l_wr_str} | {s_wr_str} | {pf_str} | {dd_str} | {hold_str} |"
        md.append(row)
        
    md.append("\n---\n")
    md.append("## 🧠 Strategy Logic & Descriptions\n")
    
    for i in range(1, 36):
        name, desc = descriptions[i]
        res = strategies.get(f"strategy_{i}")
        if not res: continue
        
        pnl_str = f"{res['total_return']*100:+.2f}%"
        wr_str = f"{res['win_rate']*100:.1f}%"
        pf_str = f"{res['profit_factor']:.2f}" if res['profit_factor'] != float('inf') else "inf"
        
        md.append(f"### Strategy {i}: {name}")
        md.append(f"*   **Logic:** {desc}")
        md.append(f"*   **Performance:** {pnl_str} | Win Rate: {wr_str} | Profit Factor: {pf_str}")
        md.append("")
        
    md.append("---\n")
    md.append("## 💡 How to Use This Guide")
    md.append("When reviewing your terminal logs or daily trading activity, reference this sheet to understand *why* the bot took a trade.")
    md.append("**Pro Tip:** To deploy capital efficiently, filter the engine to only run the Top 5 performing strategies (e.g., S19, S23, S24, S33). This provides intense, focused profitability while minimizing the chaotic drawdowns of the mean-reversion engines!")
    
    out_path = "C:/Users/loq/.gemini/antigravity/brain/1d4a79a3-38f3-4f4a-9d62-717b834a41df/ultimate_strategy_guide.md"
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(md))
        
    print(f"Successfully generated {out_path}")

if __name__ == '__main__':
    generate()
