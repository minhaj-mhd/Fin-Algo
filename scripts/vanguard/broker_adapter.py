import os
from scripts.upstox_broker import UpstoxSandboxBroker
from scripts.terminal_utils import log
from scripts.vanguard import config

class BrokerAdapter:
    def __init__(self, sandbox=config.SANDBOX_MODE):
        self.sandbox = sandbox
        # For now we use Sandbox Broker as specified in the monolith
        self.broker = UpstoxSandboxBroker()
        log(f"[BROKER] Adapter initialized in {'SANDBOX/PAPER' if self.sandbox else 'PRODUCTION/LIVE'} mode.")

    def get_live_price(self, ticker):
        """Fetches the live LTP of a ticker."""
        return self.broker.get_live_price(ticker)

    def get_instrument_key(self, ticker):
        """Resolves the exchange instrument key for Upstox."""
        return self.broker.get_instrument_key(ticker)

    def attach_websocket(self, ws_manager):
        """Attaches real-time WebSocket cache to the broker."""
        if hasattr(self.broker, 'attach_websocket'):
            self.broker.attach_websocket(ws_manager)

    def get_recent_candles(self, ticker, interval='1minute', count=120):
        """Fetches historical candles."""
        return self.broker.get_recent_candles(ticker, interval=interval, count=count)

    def place_order(self, ticker, side, quantity, price, stop_loss):
        """Places a buy/sell order with Stop Loss."""
        try:
            upstox_order = self.broker.place_order(ticker, side, quantity=quantity, price=price, stop_loss=stop_loss)
            
            # Normalize response to a standard dictionary with order_id
            order_id = "SANDBOX-ERROR"
            if upstox_order:
                if hasattr(upstox_order, "data") and upstox_order.data:
                    order_id = getattr(upstox_order.data, "order_id", "SANDBOX-SUCCESS")
                elif isinstance(upstox_order, dict) and "data" in upstox_order:
                    order_id = upstox_order["data"].get("order_id", "SANDBOX-SUCCESS")
                else:
                    order_id = "SANDBOX-SUCCESS"

            return {
                "success": order_id != "SANDBOX-ERROR",
                "order_id": order_id,
                "raw_response": str(upstox_order)
            }
        except Exception as e:
            log(f"[BROKER-ERROR] Order execution failed for {ticker}: {e}")
            return {
                "success": False,
                "order_id": "SANDBOX-ERROR",
                "error": str(e)
            }
