import pandas as pd
import numpy as np
from datetime import datetime

class StrategyFilters:
    def __init__(self):
        self.opening_ranges = {}
        self.rolling_vol = {}

    def _update_opening_range(self, ticker, date_str, current_time, df_live):
        # We need historical 15m candles to calculate ORB (09:15 to 09:45)
        # Assuming df_live is a history of the day for the ticker.
        # In this live implementation, we will use a simplified ORB based on the first hour if available.
        if date_str not in self.opening_ranges:
            self.opening_ranges[date_str] = {}
        
        if ticker not in self.opening_ranges[date_str]:
            # This is a placeholder for real ORB logic which requires historical bars
            # In live, we will use the High/Low of the day so far if time is < 10:15
            self.opening_ranges[date_str][ticker] = {
                'or_high': df_live.get('High', 0),
                'or_low': df_live.get('Low', 0)
            }

    def apply_filters(self, scores_df, current_time_str, date_str):
        """
        Applies the 9 Elite/Capacity strategies to the current ranked tickers.
        Returns a list of dicts: [{'ticker': '...', 'side': 'LONG/SHORT', 'strategy_id': X, 'conviction': Y}]
        """
        scores_df = scores_df.copy()
        if 'Long_Rank' not in scores_df.columns and 'Long_Conviction' in scores_df.columns:
            scores_df['Long_Rank'] = scores_df['Long_Conviction'].rank(ascending=False, method='min')
        if 'Short_Rank' not in scores_df.columns and 'Short_Conviction' in scores_df.columns:
            scores_df['Short_Rank'] = scores_df['Short_Conviction'].rank(ascending=False, method='min')

        active_signals = []
        
        # We need the tickers sorted by rank for the strategies
        if 'Long_Rank' in scores_df.columns:
            top_longs = scores_df.sort_values('Long_Rank').head(10).to_dict('records')
            top_shorts = scores_df.sort_values('Short_Rank').head(10).to_dict('records')
        else:
            top_longs = scores_df.sort_values('Long_Conviction', ascending=False).head(10).to_dict('records')
            top_shorts = scores_df.sort_values('Short_Conviction', ascending=False).head(10).to_dict('records')

        # S2: Short-Side Specialist
        for row in top_shorts:
            if row.get('Short_Rank', 99) <= 5:
                active_signals.append({
                    'ticker': row['ticker'], 'side': 'SHORT', 'strategy_id': 2,
                    'conviction': row.get('Short_Conviction', 0.0), 'raw_score': row.get('short_score', 0.0)
                })

        # S8: Opening Range Breakout (SHORT)
        if current_time_str >= "10:00":
            for row in top_shorts:
                if row.get('Short_Rank', 99) <= 5:
                    # In live, if close is near day low
                    if row.get('Close', 1) <= row.get('Low', 0) * 1.002:
                        active_signals.append({
                            'ticker': row['ticker'], 'side': 'SHORT', 'strategy_id': 8,
                            'conviction': row.get('Short_Conviction', 0.0), 'raw_score': row.get('short_score', 0.0)
                        })

        # S10: Quad-Timeframe Unanimous (LONG)
        for row in top_longs:
            if row.get('Long_Rank', 99) <= 3:
                # We assume if it's top 3 on 15m, it satisfies live S10 proxy
                active_signals.append({
                    'ticker': row['ticker'], 'side': 'LONG', 'strategy_id': 10,
                    'conviction': row.get('Long_Conviction', 0.0), 'raw_score': row.get('long_score', 0.0)
                })

        # S18: Volatility Expansion (SHORT)
        for row in top_shorts:
            if row.get('Short_Rank', 99) <= 3 and row.get('atr_14_pct', 0) > 0.015:
                active_signals.append({
                    'ticker': row['ticker'], 'side': 'SHORT', 'strategy_id': 18,
                    'conviction': row.get('Short_Conviction', 0.0), 'raw_score': row.get('short_score', 0.0)
                })

        # S19: Low-Vol Grind (LONG)
        for row in top_longs:
            if row.get('Long_Rank', 99) <= 5 and row.get('ibs', 1.0) < 0.15:
                active_signals.append({
                    'ticker': row['ticker'], 'side': 'LONG', 'strategy_id': 19,
                    'conviction': row.get('Long_Conviction', 0.0), 'raw_score': row.get('long_score', 0.0)
                })

        # S35: Volatility Contraction (LONG)
        for row in top_longs:
            if row.get('Long_Rank', 99) <= 5 and row.get('atr_14_pct', 1.0) < 0.002:
                active_signals.append({
                    'ticker': row['ticker'], 'side': 'LONG', 'strategy_id': 35,
                    'conviction': row.get('Long_Conviction', 0.0), 'raw_score': row.get('long_score', 0.0)
                })

        # S36: The Opening Drive (LONG)
        if current_time_str <= "10:30":
            for row in top_longs:
                if row.get('Long_Rank', 99) <= 10 and row.get('gap_pct', 0) > 0.003:
                    active_signals.append({
                        'ticker': row['ticker'], 'side': 'LONG', 'strategy_id': 36,
                        'conviction': row.get('Long_Conviction', 0.0), 'raw_score': row.get('long_score', 0.0)
                    })

        # S39: The VWAP Pinch (LONG)
        for row in top_longs:
            ma20 = row.get('ma20', row.get('Close', 1))
            close = row.get('Close', 1)
            if row.get('Long_Rank', 99) <= 3 and row.get('atr_14_pct', 1.0) < 0.002:
                if abs(close - ma20) / ma20 < 0.002:
                    active_signals.append({
                        'ticker': row['ticker'], 'side': 'LONG', 'strategy_id': 39,
                        'conviction': row.get('Long_Conviction', 0.0), 'raw_score': row.get('long_score', 0.0)
                    })

        # S42: Trend Exhaustion Trap (SHORT)
        for row in top_shorts:
            if row.get('Short_Rank', 99) <= 10 and row.get('ibs', 1.0) < 0.3:
                active_signals.append({
                    'ticker': row['ticker'], 'side': 'SHORT', 'strategy_id': 42,
                    'conviction': row.get('Short_Conviction', 0.0), 'raw_score': row.get('short_score', 0.0)
                })

        # Remove duplicates (keep highest conviction if multiple strategies trigger same ticker/side)
        final_signals = {}
        for sig in active_signals:
            key = (sig['ticker'], sig['side'])
            if key not in final_signals:
                final_signals[key] = sig
            else:
                if sig['conviction'] > final_signals[key]['conviction']:
                    final_signals[key] = sig
                    
        return list(final_signals.values())
