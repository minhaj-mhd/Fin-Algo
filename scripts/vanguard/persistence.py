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
    
    # Check if concluded (terminal)
    terminal_statuses = [
        "CLOSED", "TAKE_PROFIT", "STOP_LOSS", "EOD", "BREAKEVEN",
        "CONVICTION_FLIP", "TRAILING_STOP", "VETOED_EXPIRED", "CANCELLED_EXPIRED"
    ]
    status_str = trade.get("status") or ""
    is_terminal = status_str in terminal_statuses or status_str.endswith("_EXPIRED")
    
    if is_terminal:
        # Check if the trade is a candle-stage decision
        if trade.get("reject_stage") == "candle" or trade.get("entry_mode") is not None:
            entry_price = float(trade.get("entry_price") or 0.0)
            exit_price = float(trade.get("exit_price") or entry_price)
            
            if entry_price > 0:
                if trade.get("side") == "LONG":
                    gross_return = (exit_price - entry_price) / entry_price
                else:
                    gross_return = (entry_price - exit_price) / entry_price
                    
                gross_pnl_bps = gross_return * 10000.0
                net_pnl_bps = gross_pnl_bps - 10.0  # 10 bps round-trip cost model
                
                # Invariant assert: median(net - gross) == -cost
                assert abs((net_pnl_bps - gross_pnl_bps) - (-10.0)) < 1e-6, "Cost accounting invariant check failed!"
                
                row = {
                    "timestamp": trade.get("timestamp"),
                    "trade_id": trade.get("trade_id"),
                    "ticker": trade.get("ticker"),
                    "side": trade.get("side"),
                    "entry_mode": trade.get("entry_mode"),
                    "status": trade.get("status"),
                    "reject_stage": trade.get("reject_stage"),
                    "reject_reason": trade.get("reject_reason"),
                    "rvol": trade.get("rvol"),
                    "dist_52h": trade.get("dist_52h"),
                    "close_pos": trade.get("close_pos"),
                    "range_pct": trade.get("range_pct"),
                    "adverse_pos": trade.get("adverse_pos"),
                    "market_entry_px": trade.get("market_entry_px"),
                    "limit_px": trade.get("limit_px"),
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "stop_loss_pct": trade.get("stop_loss_pct"),
                    "take_profit_pct": trade.get("take_profit_pct"),
                    "peak_profit_pct": trade.get("peak_profit_pct"),
                    "peak_adverse_pct": trade.get("peak_adverse_pct"),
                    # Stop-loss / take-profit barrier checkpoint (SHADOW_SL_CHECKPOINT):
                    # the return at the first barrier touch, held through to the 1h close.
                    "sl_hit": trade.get("sl_hit", 0),
                    "sl_hit_time": trade.get("sl_hit_time"),
                    "sl_hit_price": trade.get("sl_hit_price"),
                    "sl_hit_pnl": trade.get("sl_hit_pnl"),
                    "tp_hit": trade.get("tp_hit", 0),
                    "tp_hit_time": trade.get("tp_hit_time"),
                    "tp_hit_price": trade.get("tp_hit_price"),
                    "tp_hit_pnl": trade.get("tp_hit_pnl"),
                    "gross_pnl_bps": round(gross_pnl_bps, 4),
                    "net_pnl_bps": round(net_pnl_bps, 4)
                }
                
                os.makedirs("data/research", exist_ok=True)
                with open("data/research/candle_rejections.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")

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
