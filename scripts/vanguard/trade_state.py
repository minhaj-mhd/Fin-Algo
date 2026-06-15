from datetime import datetime, timedelta
from scripts.vanguard import config

class TradeStateManager:
    @staticmethod
    def check_pending_entry_expiry(trade, now):
        """Returns True if the pending entry has expired based on standard 15-minute boundaries."""
        pending_since = datetime.fromisoformat(trade.get("pending_since") or trade["timestamp"])
        minute = pending_since.minute
        next_15 = ((minute // 15) + 1) * 15
        next_boundary = pending_since.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=next_15)
        
        # Expiry is 5 minutes past the next 15-minute boundary
        return now > next_boundary + timedelta(minutes=5)

    @staticmethod
    def check_candle_confirmation(trade, candle):
        """Checks if the completed 15-min candle matches the direction of our trade."""
        if candle is None:
            return False
        
        if trade["side"] == "LONG":
            return candle["close"] > candle["open"]
        elif trade["side"] == "SHORT":
            return candle["close"] < candle["open"]
        return False

    @staticmethod
    def check_candle_direction(side, candle):
        """Checks if the completed 15-min candle matches the direction on the spot.
        For LONG: bullish (close > open) OR close is in the upper part of the range (>= 60% of total length from low).
        For SHORT: bearish (close < open) OR close is in the lower part of the range (<= 40% of total length from low).
        """
        if candle is None:
            return False
        
        o = candle.get("open")
        h = candle.get("high")
        l = candle.get("low")
        c = candle.get("close")
        
        if o is None or h is None or l is None or c is None:
            return False
            
        length = h - l
        if length <= 1e-8:
            return False
            
        pos = (c - l) / length
        
        if side == "LONG":
            return (c > o) or (pos >= 0.60)
        elif side == "SHORT":
            return (c < o) or (pos <= 0.40)
        return False

    @staticmethod
    def check_live_candle_not_reversing(side, candle):
        """Guards against entering into a violent immediate reversal on the live
        (still-forming) 1-minute candle.

        Returns True when it is SAFE to enter (the live candle is NOT reversing
        heavily against `side`), and False only when the candle is strongly
        against the trade. A heavy reversal is a bar moving against us (bearish
        for LONG / bullish for SHORT) that also closes in the far end of its
        range — mirroring the 0.40/0.60 bar-position thresholds used by
        check_candle_direction. Missing or degenerate candles are treated as
        non-reversing (True) so a thin live bar never blocks an otherwise
        confirmed entry.
        """
        if not candle:
            return True

        o = candle.get("open")
        h = candle.get("high")
        l = candle.get("low")
        c = candle.get("close")

        if o is None or h is None or l is None or c is None:
            return True

        length = h - l
        if length <= 1e-8:
            return True

        pos = (c - l) / length

        if side == "LONG":
            # Strong red bar closing near its low → reversing against the long.
            return not (c < o and pos <= 0.40)
        elif side == "SHORT":
            # Strong green bar closing near its high → reversing against the short.
            return not (c > o and pos >= 0.60)
        return True

    @staticmethod
    def evaluate_open_trade_exit(trade, price, pnl, now):
        """Evaluates whether an open trade has met exit criteria (SL, BE, Trailing Stop, Time Expiry).
        Returns a tuple: (should_exit, exit_status, exit_note)
        """
        # 1. Stop Loss Check
        is_stop_loss_hit = pnl <= -trade.get("stop_loss_pct", 0.50)

        # 2. Breakeven & Trailing Stop Activations
        sl_pct = trade.get("stop_loss_pct", 0.50)
        
        # We activate in-memory trackers on the trade dictionary itself
        if not trade.get("breakeven_locked") and pnl >= sl_pct:
            trade["breakeven_locked"] = True
            print(f"[BREAKEVEN] {trade['ticker']} locked at entry. P&L={pnl:.2f}%")

        if not trade.get("trailing_active") and pnl >= (sl_pct * 2.0):
            trade["trailing_active"] = True
            print(f"[TRAILING] {trade['ticker']} trailing stop activated. P&L={pnl:.2f}%")

        # 3. Trailing/Breakeven Exit Triggers
        is_trailing_exit = False
        is_breakeven_exit = False
        
        if trade.get("trailing_active"):
            trailing_stop_level = trade["peak_profit_pct"] - sl_pct
            if pnl <= trailing_stop_level:
                is_trailing_exit = True
        elif trade.get("breakeven_locked"):
            if pnl <= 0.0:
                is_breakeven_exit = True

        # 4. Take Profit Check
        is_take_profit_hit = pnl >= trade.get("take_profit_pct", 1.00)
        if is_take_profit_hit or is_trailing_exit:
            if is_trailing_exit:
                tp_note = f" | Trailing Stop @ {pnl:.2f}% (peak {trade['peak_profit_pct']:.2f}%)"
            else:
                tp_pct_used = trade.get("take_profit_pct", 1.00)
                tp_note = f" | TP Hit @ {tp_pct_used:.2f}%"
            return True, "TAKE_PROFIT", tp_note

        # 5. Time Expiry / Market Close Checks
        raw_time_expiry = now >= datetime.fromisoformat(trade["exit_time"])
        is_time_expiry = raw_time_expiry and not trade.get("extension_pending", False)
        is_market_close = now.strftime("%H:%M") >= "15:15"

        if is_breakeven_exit:
            be_note = f" | Breakeven Exit (peak {trade.get('peak_profit_pct', 0):.2f}%)"
            return True, "CLOSED", be_note
        elif is_stop_loss_hit:
            ext_used = trade.get("extension_count", 0)
            sl_pct_used = trade.get("stop_loss_pct", 0.50)
            sl_note = f" | SL Hit @ {sl_pct_used:.2f}%" + (f" (after {ext_used} ext)" if ext_used else "")
            return True, "STOP_LOSS", sl_note
        elif is_time_expiry:
            return True, "CLOSED", " | Expiry Exit"
        elif is_market_close:
            return True, "CLOSED", " | EOD Hard-Close Exit"

        return False, None, None
