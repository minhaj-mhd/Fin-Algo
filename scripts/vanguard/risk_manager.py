import os
import json
from datetime import datetime
from scripts.vanguard import config
from scripts.terminal_utils import log

class RiskManager:
    def __init__(self, initial_capital=config.INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.virtual_capital = initial_capital
        self.day_start_capital = initial_capital
        self.used_margin = 0.0
        self.realized_charges = 0.0
        self.entry_top_k = getattr(config, "ENTRY_TOP_K", 5)
        self.hold_percentile = getattr(config, "HOLD_PERCENTILE", 0.95)
        self.stats_file = config.STATS_FILE
        self._load_virtual_stats()

    def _load_virtual_stats(self):
        """Restores full capital state from JSON. Called once on startup."""
        if os.path.exists(self.stats_file):
            try:
                with open(self.stats_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.initial_capital = data.get("initial_capital", config.INITIAL_CAPITAL)
                    self.virtual_capital = data.get("virtual_capital", config.INITIAL_CAPITAL)
                    self.realized_charges = data.get("realized_charges", 0.0)
                    self.day_start_capital = self.virtual_capital
                    log(f"[RESTORE] Capital: Rs{self.virtual_capital:.2f} | Charges: Rs{self.realized_charges:.2f}")
            except Exception as e:
                log(f"[WARN] Could not load stats file: {e}")

    def calculate_exit_charges(self, sell_value):
        """Full regulatory charge stack for Indian equity intraday."""
        brokerage = config.BROKERAGE_PER_ORDER  # ₹10 flat
        stt = sell_value * config.STT_RATE      # 0.025% on sell
        txn_charges = sell_value * 0.0000345   # NSE transaction (0.00345%)
        gst = (brokerage + txn_charges) * 0.18  # 18% GST
        sebi_fee = sell_value * 0.000001       # ₹10 per crore
        stamp_duty = 0.0                       # Only on buy-side, absorbed
        return brokerage + stt + txn_charges + gst + sebi_fee + stamp_duty

    def calculate_trade_quantity(self, price, stop_loss_pct=0.50):
        """Risk-parity sizing: every SL hit costs exactly RISK_PER_TRADE % of capital."""
        RISK_PER_TRADE = 0.005  # 0.5% of capital risked per trade
        risk_amount = self.day_start_capital * RISK_PER_TRADE
        sl_distance = price * (stop_loss_pct / 100)
        
        if sl_distance <= 0:
            sl_distance = price * 0.005  # fallback
        
        ideal_qty = int(risk_amount / sl_distance)
        max_slot_capital = (self.day_start_capital / config.MAX_TRADE_SLOTS) * config.MARGIN_MULTIPLIER
        max_qty = int(max_slot_capital / price)
        
        qty = max(1, min(ideal_qty, max_qty))
        return qty

    def recompute_used_margin(self, active_shadow_trades):
        """Recomputes margin usage across all active, pending, or open trades."""
        margin = 0.0
        for t in active_shadow_trades:
            if t["status"] in ["OPEN", "PENDING_ENTRY"]:
                qty = t.get("quantity", 1)
                entry = t.get("entry_price", 0)
                m = t.get("margin_used")
                if not m or m == 0:
                    m = (qty * entry) / config.MARGIN_MULTIPLIER
                margin += m
        self.used_margin = margin
        return margin

    def update_upstox_stats(self, active_shadow_trades):
        """Saves a full virtual portfolio snapshot to JSON and SQLite for persistence."""
        self.recompute_used_margin(active_shadow_trades)

        unrealized_pnl_inr = 0.0
        open_positions = []
        pending_positions = []
        now = datetime.now()

        for t in active_shadow_trades:
            if t["status"] == "PENDING_ENTRY":
                pending_positions.append({
                    "ticker":              t["ticker"],
                    "side":                t["side"],
                    "quantity":            t.get("quantity", 0),
                    "entry_price":         round(t.get("entry_price", 0), 2),
                    "entry_time":          t["timestamp"],
                    "one_hour_prob":       t.get("one_hour_prob", "N/A"),
                    "comment":             t.get("comment", ""),
                    "status":              "PENDING_ENTRY",
                    "strategy_id":         t.get("strategy_id"),
                    "tech_score":          float(t.get("tech_score") or 0.0) if t.get("tech_score") is not None else None,
                    "long_score":          float(t.get("long_score") or 0.0) if t.get("long_score") is not None else None,
                    "short_score":         float(t.get("short_score") or 0.0) if t.get("short_score") is not None else None,
                })
                continue

            if t["status"] != "OPEN":
                continue

            qty         = t.get("quantity", 1)
            entry       = t.get("entry_price", 0)
            current     = t.get("exit_price") or entry
            side        = t.get("side", "LONG")
            margin_used = t.get("margin_used") or ((qty * entry) / config.MARGIN_MULTIPLIER)

            if side == "LONG":
                gross_pnl_inr = (current - entry) * qty
            else:
                gross_pnl_inr = (entry - current) * qty

            pnl_pct = (gross_pnl_inr / (entry * qty) * 100) if entry > 0 else 0.0

            try:
                ext_pending  = t.get("extension_pending", False)
                ext_exit_str = t.get("extended_exit_time")
                if ext_pending and ext_exit_str:
                    effective_exit_dt = datetime.fromisoformat(ext_exit_str)
                else:
                    effective_exit_dt = datetime.fromisoformat(t["exit_time"])
                time_left = max(0, int((effective_exit_dt - now).total_seconds() / 60))
            except Exception:
                time_left = 0

            unrealized_pnl_inr += gross_pnl_inr
            ext_count = t.get("extension_count", 0)
            open_positions.append({
                "ticker":              t["ticker"],
                "side":                side,
                "quantity":            qty,
                "entry_price":         round(entry, 2),
                "current_price":       round(current, 2),
                "trade_value":         round(qty * entry, 2),
                "margin_used":         round(margin_used, 2),
                "unrealized_pnl_inr":  round(gross_pnl_inr, 2),
                "unrealized_pnl_pct":  round(pnl_pct, 4),
                "peak_profit_pct":     round(t.get("peak_profit_pct", 0), 4),
                "peak_adverse_pct":    round(t.get("peak_adverse_pct", 0), 4),
                "entry_time":          t["timestamp"],
                "exit_time":           t["exit_time"],
                "time_left_min":       time_left,
                "one_hour_prob":       t.get("one_hour_prob", "N/A"),
                "comment":             t.get("comment", ""),
                "strategy_id":         t.get("strategy_id"),
                "extension_count":     ext_count,
                "extension_pending":   t.get("extension_pending", False),
                "extended_exit_time":  t.get("extended_exit_time"),
                "tech_score":          float(t.get("tech_score") or 0.0) if t.get("tech_score") is not None else None,
                "long_score":          float(t.get("long_score") or 0.0) if t.get("long_score") is not None else None,
                "short_score":         float(t.get("short_score") or 0.0) if t.get("short_score") is not None else None,
            })

        today_pnl_inr = 0.0
        try:
            from scripts.database_manager import get_today_realized_pnl
            today_pnl_inr = get_today_realized_pnl()
        except Exception:
            pass

        total_pnl_inr = self.virtual_capital - self.initial_capital
        total_pnl_pct = (total_pnl_inr / self.initial_capital * 100) if self.initial_capital > 0 else 0.0

        stats = {
            "initial_capital":        self.initial_capital,
            "virtual_capital":        round(self.virtual_capital, 2),
            "used_margin":            round(self.used_margin, 2),
            "available_margin":       round(self.virtual_capital - self.used_margin, 2),
            "realized_charges":       round(self.realized_charges, 2),
            "open_positions_count":   len(open_positions),
            "unrealized_pnl_inr":     round(unrealized_pnl_inr, 2),
            "day_realized_pnl_inr":   round(today_pnl_inr, 2),
            "total_pnl_inr":          round(total_pnl_inr, 2),
            "total_pnl_pct":          round(total_pnl_pct, 4),
            "positions":              open_positions,
            "pending_positions":      pending_positions,
            "timestamp":              now.isoformat(),
        }
        with open(self.stats_file, "w") as f:
            json.dump(stats, f, indent=4)

        try:
            from scripts.database_manager import log_system_stats
            log_system_stats(stats)
        except Exception:
            pass
