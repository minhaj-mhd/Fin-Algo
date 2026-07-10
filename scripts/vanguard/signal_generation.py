import os
import pandas as pd
from datetime import datetime
from scripts.vanguard import config
from scripts.terminal_utils import log

class SignalGenerator:
    def __init__(self, strategy_filters=None):
        # Pipeline 2 (Structural Strategies S1-S50) is retired. ``strategy_filters``
        # is accepted for backward compatibility but is no longer used: the live
        # engine generates signals exclusively from Pipeline 1 (Pure AI Signals).
        self.strategy_filters = strategy_filters

    def generate_candidate_signals(
        self,
        scores_df,
        long_eligible,
        short_eligible,
        is_in_cooldown_fn,
        is_veto_cooldown_fn,
        min_conviction=0.0,
        min_raw_score=0.0,
        entry_top_k=2
    ):
        """Processes live model scores via Pipeline 1 (Pure AI Signals) only.

        Pipeline 2 (Structural Strategies S1-S50) has been retired; the merge /
        ensemble-overlap stage is therefore gone and every returned signal is a
        pure AI signal with ``strategy_id=None`` and ``is_ensemble=False``.
        """
        if scores_df.empty:
            return pd.DataFrame()

        # ========================================================
        # Pipeline 1: Pure AI Signals
        # ========================================================
        signals = []
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

            if getattr(config, "SIGNAL_RAW_SCORE_ONLY", False):
                # 2026-07-05: single top-K pick per side by RAW model score only. Research showed
                # raw beats the conviction (long-short) rank on longs (+0.6 vs -1.7 bps) and ties
                # on shorts; drops the AI_Net path so exactly `entry_top_k` picks/side are emitted.
                top_raw = eligible_df.sort_values(raw_col, ascending=False).head(entry_top_k)
                source_sets = (("AI_Raw", top_raw),)
            else:
                # Top-K Hybrid (Net) Candidates sorted by rank (scale-free)
                top_net = eligible_df.sort_values(rank_col_name, ascending=True).head(entry_top_k)
                # Top-K Pure Directional Candidates sorted by raw score (excluding top_net)
                eligible_raw_df = eligible_df[~eligible_df['ticker'].isin(top_net['ticker'])]
                top_raw = eligible_raw_df.sort_values(raw_col, ascending=False).head(entry_top_k)
                source_sets = (("AI_Net", top_net), ("AI_Raw", top_raw))

            for source, candidates in source_sets:
                for _, row in candidates.iterrows():
                    row_dict = row.to_dict()
                    row_dict.update({
                        'side': side,
                        'conviction': float(row[conv_col]),
                        'raw_score': float(row[raw_col]),
                        'strategy_id': None,
                        'source': source,
                        'size_multiplier': 1.0,
                        'is_ensemble': False
                    })
                    signals.append(row_dict)

        return pd.DataFrame(signals) if signals else pd.DataFrame()
