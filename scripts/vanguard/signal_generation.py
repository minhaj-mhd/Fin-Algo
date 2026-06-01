import os
import pandas as pd
from datetime import datetime
from scripts.vanguard import config
from scripts.terminal_utils import log

class SignalGenerator:
    def __init__(self, strategy_filters):
        self.strategy_filters = strategy_filters

    def generate_candidate_signals(
        self,
        scores_df,
        long_eligible,
        short_eligible,
        is_in_cooldown_fn,
        is_veto_cooldown_fn,
        min_conviction=config.MIN_CONVICTION,
        min_raw_score=config.MIN_RAW_SCORE
    ):
        """Processes live model scores and applies dual pipelines (Pure AI + Structural Strategy)
        to return merged, ranked, and ensemble-annotated candidate signals.
        """
        if scores_df.empty:
            return pd.DataFrame()

        now_str = datetime.now().strftime("%H:%M")
        date_str = datetime.now().strftime("%Y-%m-%d")

        # 1. Pipeline 1: Pure AI Signals
        ai_signals = []
        for side in ["LONG", "SHORT"]:
            eligible_tickers = long_eligible if side == "LONG" else short_eligible
            
            eligible_mask = scores_df['ticker'].apply(
                lambda x: x in eligible_tickers
                and not (is_in_cooldown_fn(x) or is_veto_cooldown_fn(x))
            )
            eligible_df = scores_df[eligible_mask]
            
            if eligible_df.empty:
                continue
                
            conv_col = "Long_Conviction" if side == "LONG" else "Short_Conviction"
            raw_col = "long_score" if side == "LONG" else "short_score"
            rank_col_name = "Long_Rank" if side == "LONG" else "Short_Rank"
            
            # Top 2 Hybrid (Net) Candidates that meet min_conviction
            top_net = eligible_df[eligible_df[conv_col] >= min_conviction].sort_values(rank_col_name, ascending=True).head(2)
            
            # Top 2 Pure Directional Candidates that meet min_raw_score (excluding top_net)
            eligible_raw_df = eligible_df[~eligible_df['ticker'].isin(top_net['ticker'])]
            top_raw = eligible_raw_df[eligible_raw_df[raw_col] >= min_raw_score].sort_values(raw_col, ascending=False).head(2)
            
            for _, row in top_net.iterrows():
                ai_signals.append({
                    'ticker': row['ticker'],
                    'side': side,
                    'conviction': float(row[conv_col]),
                    'raw_score': float(row[raw_col]),
                    'strategy_id': None,
                    'source': 'AI_Net'
                })
            for _, row in top_raw.iterrows():
                ai_signals.append({
                    'ticker': row['ticker'],
                    'side': side,
                    'conviction': float(row[conv_col]),
                    'raw_score': float(row[raw_col]),
                    'strategy_id': None,
                    'source': 'AI_Raw'
                })

        # 2. Pipeline 2: Structural Strategy Signals
        strategy_eligible_mask = scores_df['ticker'].apply(
            lambda x: (x in long_eligible or x in short_eligible)
            and not (is_in_cooldown_fn(x) or is_veto_cooldown_fn(x))
        )
        base_eligible_df = scores_df[strategy_eligible_mask]
        
        strategy_signals_raw = self.strategy_filters.apply_filters(base_eligible_df, now_str, date_str)
        
        strategy_signals = []
        for sig in strategy_signals_raw:
            sig_side = sig['side']
            sig_ticker = sig['ticker']
            if sig_side == "LONG" and sig_ticker not in long_eligible:
                continue
            if sig_side == "SHORT" and sig_ticker not in short_eligible:
                continue
            strategy_signals.append(sig)

        # 3. The Merge & Ensemble Identification
        merged_signals = []
        
        # Add all strategy signals first
        for sig in strategy_signals:
            sig_ticker = sig['ticker']
            sig_side = sig['side']
            
            row_dict = scores_df[scores_df['ticker'] == sig_ticker].iloc[0].to_dict()
            row_dict.update({
                'side': sig_side,
                'conviction': float(sig['conviction']),
                'raw_score': float(sig['raw_score']),
                'strategy_id': sig['strategy_id'],
                'source': f"Strategy_S{sig['strategy_id']}",
                'is_ensemble': False
            })
            merged_signals.append(row_dict)
            
        # Add AI signals and check for overlap (Ensemble)
        for sig in ai_signals:
            sig_ticker = sig['ticker']
            sig_side = sig['side']
            
            existing = next((m for m in merged_signals if m['ticker'] == sig_ticker and m['side'] == sig_side), None)
            
            if existing is not None:
                existing['is_ensemble'] = True
                existing['source'] = f"Ensemble_S{existing['strategy_id']}+{sig['source']}"
                log(f"[ENSEMBLE MATCH] Ticker {sig_ticker} ({sig_side}) triggered by both Strategy S{existing['strategy_id']} and {sig['source']}!")
            else:
                row_dict = scores_df[scores_df['ticker'] == sig_ticker].iloc[0].to_dict()
                row_dict.update({
                    'side': sig_side,
                    'conviction': float(sig['conviction']),
                    'raw_score': float(sig['raw_score']),
                    'strategy_id': None,
                    'source': sig['source'],
                    'is_ensemble': False
                })
                merged_signals.append(row_dict)
                
        return pd.DataFrame(merged_signals) if merged_signals else pd.DataFrame()
