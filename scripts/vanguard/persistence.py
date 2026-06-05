import os
import json
from scripts.vanguard import config
from scripts.database_manager import log_trade as db_log_trade

def update_markdown_ledger(trade):
    """Appends trade metrics to the physical Markdown ledger file."""
    ledger_path = "data/VANGUARD_DEMO_LEDGER.md"
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(ledger_path):
        with open(ledger_path, "w", encoding="utf-8") as f:
            f.write(
                "# VANGUARD ELITE COMMAND CENTER LEDGER\n\n| Timestamp | Ticker | Side | Qty | Entry | Exit | Net ₹ | Final % | Status | Comment |\n| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
            )
    with open(ledger_path, "a", encoding="utf-8") as f:
        qty = trade.get("quantity", 1)
        net_amt = trade.get("net_pnl_amt", 0)
        f.write(
            f"| {trade['timestamp'][:16]} | {trade['ticker']} | {trade['side']} | {qty} | {trade['entry_price']:.2f} | {trade.get('exit_price', 0):.2f} | ₹{net_amt:.2f} | {trade['final_profit_pct']:.2f}% | {trade['status']} | {trade.get('comment', '')} |\n"
        )

def log_trade(trade):
    """Saves a trade record to the SQLite database and logs to the Markdown ledger if concluded."""
    db_log_trade(trade)
    if trade["status"] in ["CLOSED", "TAKE_PROFIT", "STOP_LOSS", "VETOED_EXPIRED"]:
        update_markdown_ledger(trade)

def save_latest_scores(scores_df, long_eligible, short_eligible):
    """Saves the latest scoring matrix for UI dashboard consumption."""
    if scores_df.empty:
        return
        
    # Safely ensure all multi-tf scores exist in the dataframe
    for col in ["score_15m", "score_30m", "score_1d"]:
        if col not in scores_df.columns:
            scores_df[col] = 0.0

    dashboard_cols = [
        "ticker", "Close", "long_score", "short_score", "Long_Conviction",
        "Short_Conviction", "Long_Rank", "Short_Rank", "dv_raw", "rvol_raw",
        "dist_52h_model", "dist_52h_actual", "daily_long_eligible", "daily_short_eligible",
        "score_15m", "score_30m", "score_1d"
    ]
    
    # Safely compute flag mappings
    scores_df = scores_df.assign(
        daily_long_eligible=scores_df["ticker"].apply(lambda x: x in long_eligible),
        daily_short_eligible=scores_df["ticker"].apply(lambda x: x in short_eligible)
    ).copy()
    
    latest_data = (
        scores_df[dashboard_cols]
        .fillna(0)
        .sort_values("Long_Rank")
        .to_dict(orient="records")
    )
    
    os.makedirs(os.path.dirname(config.LATEST_SCORES_FILE), exist_ok=True)
    with open(config.LATEST_SCORES_FILE, "w") as f:
        json.dump(latest_data, f)
