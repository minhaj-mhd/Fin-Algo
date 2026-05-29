"""
╔══════════════════════════════════════════════════════════════╗
║          UPSTOX SANDBOX — DEBUG & MONITOR CONSOLE           ║
║  Run in a separate terminal. No engine needed.              ║
║  Usage:  python scripts/upstox_debug.py                     ║
╚══════════════════════════════════════════════════════════════╝

UPSTOX SANDBOX — CONFIRMED WORKING ENDPOINTS (live-tested)
──────────────────────────────────────────────────────────
 SANDBOX token  → api-sandbox.upstox.com
   ✔  place_order       LIMIT / MARKET / SL / SL-M
   ✔  cancel_order      cancel any open/pending order by ID
   ✔  modify_order      change price, qty, trigger on pending order
   ✔  get_order_book    list all today's orders
   ✘  exit_positions    NOT available in sandbox
   ✘  get_trade_history NOT available in sandbox
   ✘  funds/margin      NOT available in sandbox
   ✘  positions         NOT available in sandbox

 ANALYTICS token → api.upstox.com (live, READ-ONLY)
   ✔  Market quotes / LTP
   ✔  Historical candles
   ✔  Instrument search
   ✘  account/order endpoints  (read-only token = UDAPI100067)
"""

import os, sys, json, time
from datetime import datetime

# ── Colour helpers ────────────────────────────────────────────────────────────
os.system("")
R   = "\033[91m";  G  = "\033[92m";  Y  = "\033[93m"
B   = "\033[94m";  C  = "\033[96m";  W  = "\033[97m"
DIM = "\033[90m";  BOLD = "\033[1m"; RST = "\033[0m"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dotenv import load_dotenv
load_dotenv()

import upstox_client
from upstox_client.rest import ApiException

# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def ts():      return datetime.now().strftime("%H:%M:%S")
def ok(m):     print(f"{G}  \u2714  {m}{RST}")
def warn(m):   print(f"{Y}  \u26a0  {m}{RST}")
def err(m):    print(f"{R}  \u2718  {m}{RST}")
def info(m):   print(f"{DIM}     {m}{RST}")
def skip(m):   print(f"{B}  \u2192  {m}{RST}")
def label(k, v, color=W): print(f"  {DIM}{k:<32}{RST}{color}{v}{RST}")

def header(title):
    line = "\u2500" * 62
    print(f"\n{C}{line}{RST}")
    print(f"{BOLD}{C}  {title}{RST}")
    print(f"{C}{line}{RST}")

def pretty_json(obj):
    try:
        if hasattr(obj, "to_dict"):
            obj = obj.to_dict()
        raw = json.dumps(obj, indent=4, default=str)
        lines = []
        for line in raw.splitlines():
            if '": ' in line:
                k, _, v = line.partition('": ')
                lines.append(f'{DIM}{k}":{RST} {W}{v}{RST}')
            else:
                lines.append(f"{DIM}{line}{RST}")
        return "\n".join(lines)
    except Exception as e:
        return f"{Y}(Cannot serialise: {e}){RST}\n{str(obj)}"

def section_dump(title, obj):
    print(f"\n{DIM}  {title}:{RST}")
    print(pretty_json(obj))

def api_call(fn, *args, label_str=""):
    """Execute fn(*args), print timing + raw error body. Returns (resp, ok_flag)."""
    t0 = time.perf_counter()
    try:
        resp = fn(*args)
        ms   = int((time.perf_counter() - t0) * 1000)
        ok(f"{label_str}  [{ms} ms]")
        return resp, True
    except ApiException as e:
        ms = int((time.perf_counter() - t0) * 1000)
        err(f"ApiException [{ms} ms]:  HTTP {e.status} \u2014 {e.reason}")
        body = (e.body[:500] if isinstance(e.body, str) else e.body.decode()[:500]) if e.body else "empty"
        try:
            parsed = json.loads(body)
            for er in parsed.get("errors", []):
                info(f"Code    : {er.get('errorCode','?')}")
                info(f"Message : {er.get('message','?')}")
        except Exception:
            info(f"Body    : {body}")
        return None, False
    except Exception as e:
        ms = int((time.perf_counter() - t0) * 1000)
        err(f"Error [{ms} ms]: {e}")
        return None, False

# ─────────────────────────────────────────────────────────────────────────────
# Client builders
# ─────────────────────────────────────────────────────────────────────────────

SANDBOX_TOKEN   = os.getenv("UPSTOX_SANDBOX_ACCESS_TOKEN",  "")
ANALYTICS_TOKEN = os.getenv("UPSTOX_ANALYTICS_ACCESS_TOKEN", "")

def build_sandbox_client():
    """
    Sandbox client — only order endpoints.
    Only adds paths the sandbox server actually implements.
    """
    cfg = upstox_client.Configuration(sandbox=True)
    cfg.access_token = SANDBOX_TOKEN
    # NOTE: we intentionally do NOT add margin/positions to sandbox_urls.
    # Those paths are not implemented on api-sandbox.upstox.com and return 404.
    return upstox_client.ApiClient(cfg)

def build_analytics_client():
    """Analytics (read-only live) client — market data only."""
    cfg = upstox_client.Configuration(sandbox=False)
    cfg.sandbox = False
    cfg.host    = "https://api.upstox.com"
    cfg.access_token = ANALYTICS_TOKEN
    return upstox_client.ApiClient(cfg)

# ─────────────────────────────────────────────────────────────────────────────
# Test functions
# ─────────────────────────────────────────────────────────────────────────────

def check_tokens():
    header("TOKEN STATUS CHECK")
    for name, val in [("SANDBOX token",   SANDBOX_TOKEN),
                      ("ANALYTICS token", ANALYTICS_TOKEN)]:
        if val:
            ok(f"{name} loaded")
            info(f"Preview : {val[:40]}...")
            # Decode header+payload (no verification, just for debug)
            try:
                import base64
                parts = val.split(".")
                pad   = parts[1] + "=="
                decoded = json.loads(base64.urlsafe_b64decode(pad))
                exp_ts  = decoded.get("exp", 0)
                exp_dt  = datetime.fromtimestamp(exp_ts).strftime("%Y-%m-%d %H:%M")
                now_ts  = datetime.now().timestamp()
                if now_ts > exp_ts:
                    warn(f"Token EXPIRED  (exp: {exp_dt})")
                else:
                    days_left = int((exp_ts - now_ts) / 86400)
                    ok(f"Token VALID  \u2014 expires {exp_dt}  ({days_left}d left)")
                info(f"Subject : {decoded.get('sub','?')}")
                info(f"isPlus  : {decoded.get('isPlus','?')}")
            except Exception as ex:
                warn(f"Could not decode JWT: {ex}")
        else:
            err(f"{name} NOT FOUND in .env")


def test_orderbook(sandbox_client):
    header("SANDBOX \u2014 ORDER BOOK  (/v2/order/retrieve-all)")
    api  = upstox_client.OrderApi(sandbox_client)
    resp, ok_flag = api_call(api.get_order_book, "2.0", label_str="Order book fetch")
    if not ok_flag:
        return
    data = getattr(resp, "data", []) or []
    label("Orders in book", str(len(data)))
    label("Status",         getattr(resp, "status", "?"), G)

    if data:
        print(f"\n{DIM}  Last 3 orders:{RST}")
        for o in data[:3]:
            d = o if isinstance(o, dict) else (o.to_dict() if hasattr(o, "to_dict") else {})
            print(f"  {W}{d.get('trading_symbol','?'):<12}{RST}"
                  f"  {d.get('transaction_type','?'):<5}"
                  f"  qty={d.get('quantity','?')}"
                  f"  status={d.get('status','?')}"
                  f"  price={d.get('price','?')}")
    section_dump("Full Response", resp)


def test_margins_note():
    """
    Funds & Margin cannot be fetched in sandbox mode.
    Explain clearly what tokens would be needed.
    """
    header("SANDBOX \u2014 FUNDS & MARGIN  (diagnostic)")

    warn("This endpoint is NOT implemented on the Upstox sandbox server.")
    print(f"""
{DIM}  What actually happens:{RST}
  \u2022 The sandbox server ({C}api-sandbox.upstox.com{RST}) only handles ORDER operations.
  \u2022 Margin / positions data lives on the live server ({C}api.upstox.com{RST}).
  \u2022 The ANALYTICS token is read-only \u2014 it cannot access account endpoints.
  \u2022 To read real margin you need a {Y}full-access live token{RST} (login flow token).

{DIM}  In the engine:{RST}
  The engine tracks margin {G}internally{RST} via virtual_capital / used_margin
  in {W}upstox_stats.json{RST} \u2014 it does {Y}not{RST} rely on this API call.
""")

    # Show what the engine's own stats file says instead
    stats_path = os.path.join(os.getcwd(), "upstox_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        ok("Loaded engine's virtual portfolio snapshot (upstox_stats.json)")
        label("Virtual Capital",     f"\u20b9{stats.get('virtual_capital',0):,.2f}", G)
        label("Available Margin",    f"\u20b9{stats.get('available_margin',0):,.2f}", G)
        label("Used Margin",         f"\u20b9{stats.get('used_margin',0):,.2f}", Y)
        label("Unrealized P&L",      f"\u20b9{stats.get('unrealized_pnl_inr',0):,.2f}")
        label("Open Positions",      str(stats.get('open_positions_count', 0)))
        label("Snapshot time",       stats.get('timestamp','?'), DIM)
    else:
        warn("upstox_stats.json not found \u2014 engine may not be running")


def test_positions_note(sandbox_client):
    """
    Positions cannot be read directly from sandbox.
    Read from engine's stats + order book instead.
    """
    header("SANDBOX \u2014 POSITIONS  (diagnostic)")
    warn("Short-term positions API is NOT available on the sandbox server.")
    print(f"""
{DIM}  Reason:{RST}
  \u2022 {C}api-sandbox.upstox.com{RST} does not implement {W}/v2/portfolio/short-term-positions{RST}.
  \u2022 Adding it to sandbox_urls routes requests to a 404.

{DIM}  Workarounds available in this console:{RST}
  \u2022 {W}[2]{RST}  Order book \u2014 shows all sandbox orders placed (placed/pending/rejected)
  \u2022 {W}[s]{RST}  Engine snapshot \u2014 reads upstox_stats.json for virtual open positions
""")

    stats_path = os.path.join(os.getcwd(), "upstox_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path) as f:
            stats = json.load(f)
        positions = stats.get("positions", [])
        ok(f"Engine virtual positions  ({len(positions)} open)")
        if positions:
            print()
            fmt = f"  {{:<12}} {{:<6}} {{:>8}} {{:>12}} {{:>10}} {{:>8}}"
            print(fmt.format("Ticker","Side","Qty","Entry \u20b9","Unrealised","P&L %"))
            print(f"  {DIM}{'-'*64}{RST}")
            for p in positions:
                pct = p.get("unrealized_pnl_pct", 0)
                col = G if pct >= 0 else R
                print(fmt.format(
                    p.get("ticker","?").replace(".NS",""),
                    p.get("side","?"),
                    str(p.get("quantity","?")),
                    f"\u20b9{p.get('entry_price',0):.2f}",
                    f"\u20b9{p.get('unrealized_pnl_inr',0):.2f}",
                    f"{col}{pct:+.2f}%{RST}",
                ))
        else:
            info("No open positions in engine snapshot")


def test_live_price(data_client, ticker="RELIANCE.NS"):
    header(f"ANALYTICS \u2014 LIVE MARKET QUOTE  ({ticker})")
    cache_path = os.path.join(os.path.dirname(__file__), "instrument_cache.json")
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)
    symbol = ticker.replace(".NS", "")
    ikey   = cache.get(symbol, f"NSE_EQ|{symbol}")
    label("Instrument key", ikey)

    api  = upstox_client.MarketQuoteApi(data_client)
    resp, ok_flag = api_call(api.get_full_market_quote, ikey, "2.0",
                             label_str="Market quote fetch")
    if not ok_flag:
        return

    data = getattr(resp, "data", {}) or {}
    price = None
    if ikey in data:
        price = data[ikey].last_price
    elif data:
        price = list(data.values())[0].last_price

    if price:
        label("LTP", f"\u20b9{price:.2f}", G)
    else:
        warn("Could not extract LTP")
    section_dump("Full Quote Response", resp)


def test_instrument_search(data_client, ticker="RELIANCE.NS"):
    header(f"ANALYTICS \u2014 INSTRUMENT SEARCH  ({ticker})")
    symbol = ticker.replace(".NS", "")
    api  = upstox_client.InstrumentsApi(data_client)
    resp, ok_flag = api_call(api.search_instrument, symbol,
                             label_str="Instrument search")
    if not ok_flag:
        return

    data = getattr(resp, "data", []) or []
    label("Results", str(len(data)))
    if data:
        print(f"\n{DIM}  Top matches:{RST}")
        for i, inst in enumerate(data[:5]):
            d = inst if isinstance(inst, dict) else (inst.to_dict() if hasattr(inst, "to_dict") else {})
            exact = d.get("trading_symbol","?") == symbol
            marker = f"{G} \u2190 EXACT MATCH{RST}" if exact else ""
            print(f"  [{i}]  {W}{d.get('trading_symbol','?'):<15}{RST}"
                  f"  key={d.get('instrument_key','?')}"
                  f"  seg={d.get('segment','?')}"
                  f"{marker}")
    section_dump("Full Response", resp)


def test_place_order(sandbox_client, data_client, ticker="RELIANCE.NS",
                     side="LONG", qty=1):
    header(f"SANDBOX \u2014 PLACE ORDER  ({side} {qty}x {ticker})")

    cache_path = os.path.join(os.path.dirname(__file__), "instrument_cache.json")
    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cache = json.load(f)
    symbol = ticker.replace(".NS", "")
    ikey   = cache.get(symbol, f"NSE_EQ|{symbol}")

    # Fetch live price
    price = 100.0
    try:
        api  = upstox_client.MarketQuoteApi(data_client)
        resp = api.get_full_market_quote(ikey, "2.0")
        data = getattr(resp, "data", {}) or {}
        if ikey in data:
            price = data[ikey].last_price
        elif data:
            price = list(data.values())[0].last_price
    except Exception as e:
        warn(f"Price fetch failed ({e}). Using \u20b9{price:.2f} as dummy")

    label("Instrument key", ikey)
    label("Limit price    ", f"\u20b9{price:.2f}")

    order_body = {
        "quantity":           qty,
        "product":            "I",
        "validity":           "DAY",
        "price":              round(price, 2),
        "tag":                "VANGUARD_DEBUG",
        "instrument_token":   ikey,
        "order_type":         "LIMIT",
        "transaction_type":   "BUY" if side == "LONG" else "SELL",
        "disclosed_quantity": 0,
        "trigger_price":      0.0,
        "is_amo":             False,
    }
    print(f"\n{DIM}  Request body:{RST}")
    print(pretty_json(order_body))

    api   = upstox_client.OrderApi(sandbox_client)
    resp, ok_flag = api_call(api.place_order, order_body, "2.0",
                             label_str="Place order")
    if ok_flag:
        section_dump("Place Order Response", resp)
        d = resp.to_dict() if hasattr(resp, "to_dict") else {}
        oid = (d.get("data") or {}).get("order_id", "")
        if oid:
            ok(f"Order ID: {oid}  (use 'c {oid}' to cancel it)")


def _load_cache():
    cache_path = os.path.join(os.path.dirname(__file__), "instrument_cache.json")
    try:
        with open(cache_path) as f:
            return json.load(f)
    except Exception:
        return {}


def _get_ikey(ticker):
    symbol = ticker.replace(".NS", "")
    return _load_cache().get(symbol, f"NSE_EQ|{symbol}")


def _get_ltp(data_client, ikey):
    """Fetch LTP from analytics; return float or None."""
    try:
        api  = upstox_client.MarketQuoteApi(data_client)
        resp = api.get_full_market_quote(ikey, "2.0")
        data = getattr(resp, "data", {}) or {}
        if ikey in data:
            return data[ikey].last_price
        if data:
            return list(data.values())[0].last_price
    except Exception:
        pass
    return None


def test_cancel_order(sandbox_client, order_id):
    """Cancel a single order by its ID."""
    header(f"SANDBOX — CANCEL ORDER  (id={order_id})")
    api  = upstox_client.OrderApi(sandbox_client)
    resp, ok_flag = api_call(api.cancel_order, order_id, "2.0",
                             label_str="Cancel order")
    if ok_flag:
        section_dump("Cancel Response", resp)


def test_cancel_all(sandbox_client):
    """Fetch order book and cancel every open/pending order."""
    header("SANDBOX — CANCEL ALL PENDING ORDERS")
    api = upstox_client.OrderApi(sandbox_client)

    # Step 1: fetch book
    resp, ok_flag = api_call(api.get_order_book, "2.0", label_str="Fetch order book")
    if not ok_flag:
        return

    orders  = getattr(resp, "data", []) or []
    pending = []
    for o in orders:
        d = o.to_dict() if hasattr(o, "to_dict") else o
        if "pending" in str(d.get("status", "")).lower():
            pending.append(d)

    if not pending:
        info("No pending orders to cancel.")
        return

    label("Pending orders found", str(len(pending)), Y)
    confirm = input(f"\n{Y}  Cancel all {len(pending)} pending orders? [y/N]: {RST}").strip().lower()
    if confirm != "y":
        info("Cancelled.")
        return

    for d in pending:
        oid = d.get("order_id", "")
        sym = d.get("trading_symbol", "?")
        print(f"  {DIM}Cancelling {sym} ({oid})...{RST}", end=" ")
        try:
            r = api.cancel_order(oid, "2.0")
            print(f"{G}✔ {r.status}{RST}")
        except Exception as e:
            print(f"{R}✘ {str(e)[:120]}{RST}")


def test_modify_order(sandbox_client):
    """
    Interactively modify a pending order — change price or trigger (stop-loss).
    Shows the order book first so the user can pick an order ID.
    """
    header("SANDBOX — MODIFY ORDER  (price / stop-loss update)")
    api = upstox_client.OrderApi(sandbox_client)

    # Show pending orders
    resp, ok_flag = api_call(api.get_order_book, "2.0", label_str="Fetch order book")
    if not ok_flag:
        return

    orders  = getattr(resp, "data", []) or []
    pending = []
    for o in orders:
        d = o.to_dict() if hasattr(o, "to_dict") else o
        if "pending" in str(d.get("status", "")).lower():
            pending.append(d)

    if not pending:
        info("No pending orders available to modify.")
        return

    print(f"\n{DIM}  Pending orders:{RST}")
    fmt = "  {:<26} {:<12} {:<5} {:<6} {:>10} {:>10}"
    print(fmt.format("order_id", "symbol", "txn", "type", "price", "trigger"))
    print(f"  {DIM}{'-'*75}{RST}")
    for d in pending:
        print(fmt.format(
            d.get("order_id", "?"),
            d.get("trading_symbol", "?"),
            d.get("transaction_type", "?"),
            d.get("order_type", "?"),
            str(d.get("price", "?")),
            str(d.get("trigger_price", "?")),
        ))

    oid = input(f"\n{C}  Enter order_id to modify: {RST}").strip()
    match = next((d for d in pending if d.get("order_id") == oid), None)
    if not match:
        err(f"Order ID '{oid}' not found in pending orders.")
        return

    cur_price   = float(match.get("price", 0))
    cur_trigger = float(match.get("trigger_price", 0))
    cur_qty     = int(match.get("quantity", 1))
    otype       = match.get("order_type", "LIMIT")

    label("Current price  ", f"₹{cur_price:.2f}")
    label("Current trigger", f"₹{cur_trigger:.2f}")
    label("Order type     ", otype)

    new_price_str   = input(f"  {DIM}New price   (Enter to keep ₹{cur_price:.2f}): {RST}").strip()
    new_trigger_str = input(f"  {DIM}New trigger (Enter to keep ₹{cur_trigger:.2f}): {RST}").strip()
    new_qty_str     = input(f"  {DIM}New qty     (Enter to keep {cur_qty}): {RST}").strip()

    new_price   = float(new_price_str)   if new_price_str   else cur_price
    new_trigger = float(new_trigger_str) if new_trigger_str else cur_trigger
    new_qty     = int(new_qty_str)       if new_qty_str     else cur_qty

    modify_body = {
        "order_id":           oid,
        "quantity":           new_qty,
        "validity":           "DAY",
        "price":              round(new_price, 2),
        "order_type":         otype,
        "disclosed_quantity": 0,
        "trigger_price":      round(new_trigger, 2),
    }
    print(f"\n{DIM}  Modify body:{RST}")
    print(pretty_json(modify_body))

    resp2, ok_flag2 = api_call(api.modify_order, modify_body, "2.0",
                               label_str="Modify order")
    if ok_flag2:
        section_dump("Modify Response", resp2)


def test_place_sl_order(sandbox_client, data_client,
                        ticker="RELIANCE.NS", side="LONG", qty=1,
                        trigger_offset_pct=1.0):
    """
    Place a SL-M (Stop-Loss Market) order.
    trigger_price is set at current_price ± offset% away from current price
    so the sandbox accepts it without immediately triggering.
    """
    header(f"SANDBOX — PLACE SL-M ORDER  ({side} {qty}x {ticker})")
    ikey  = _get_ikey(ticker)
    price = _get_ltp(data_client, ikey)
    if price is None:
        price = 100.0
        warn(f"Could not fetch LTP. Using ₹{price:.2f} as dummy.")

    # For LONG: SL fires if price drops → trigger below market
    # For SHORT: SL fires if price rises → trigger above market
    if side == "LONG":
        trigger = round(price * (1 - trigger_offset_pct / 100), 2)
    else:
        trigger = round(price * (1 + trigger_offset_pct / 100), 2)

    label("Instrument key ", ikey)
    label("LTP            ", f"₹{price:.2f}")
    label("Trigger price  ", f"₹{trigger:.2f}  ({trigger_offset_pct}% away)")

    order_body = {
        "quantity":           qty,
        "product":            "I",
        "validity":           "DAY",
        "price":              0.0,          # 0 for SL-M (market fill on trigger)
        "tag":                "VANGUARD_DEBUG_SLM",
        "instrument_token":   ikey,
        "order_type":         "SL-M",
        "transaction_type":   "SELL" if side == "LONG" else "BUY",  # SL exits the position
        "disclosed_quantity": 0,
        "trigger_price":      trigger,
        "is_amo":             False,
    }
    print(f"\n{DIM}  Request body:{RST}")
    print(pretty_json(order_body))

    api  = upstox_client.OrderApi(sandbox_client)
    resp, ok_flag = api_call(api.place_order, order_body, "2.0",
                             label_str="Place SL-M order")
    if ok_flag:
        section_dump("SL-M Order Response", resp)
        d   = resp.to_dict() if hasattr(resp, "to_dict") else {}
        oid = (d.get("data") or {}).get("order_id", "")
        if oid:
            ok(f"Order ID: {oid}  (use 'c {oid}' to cancel it)")


def test_market_sell(sandbox_client, data_client,
                     ticker="RELIANCE.NS", qty=1):
    """Place a MARKET SELL order (immediate execution at best available price)."""
    header(f"SANDBOX — PLACE MARKET SELL  ({qty}x {ticker})")
    ikey  = _get_ikey(ticker)
    price = _get_ltp(data_client, ikey)
    if price:
        label("Current LTP", f"₹{price:.2f}", G)
    else:
        warn("Could not fetch LTP — order will still be placed at market.")

    order_body = {
        "quantity":           qty,
        "product":            "I",
        "validity":           "DAY",
        "price":              0.0,     # 0 = market order
        "tag":                "VANGUARD_DEBUG_MKT",
        "instrument_token":   ikey,
        "order_type":         "MARKET",
        "transaction_type":   "SELL",
        "disclosed_quantity": 0,
        "trigger_price":      0.0,
        "is_amo":             False,
    }
    print(f"\n{DIM}  Request body:{RST}")
    print(pretty_json(order_body))

    api  = upstox_client.OrderApi(sandbox_client)
    resp, ok_flag = api_call(api.place_order, order_body, "2.0",
                             label_str="Place MARKET SELL")
    if ok_flag:
        section_dump("Market Sell Response", resp)
        d   = resp.to_dict() if hasattr(resp, "to_dict") else {}
        oid = (d.get("data") or {}).get("order_id", "")
        if oid:
            ok(f"Order ID: {oid}")


def show_engine_snapshot():
    header("ENGINE VIRTUAL PORTFOLIO SNAPSHOT  (upstox_stats.json)")
    stats_path = os.path.join(os.getcwd(), "upstox_stats.json")
    if not os.path.exists(stats_path):
        warn("upstox_stats.json not found \u2014 is the signal engine running?")
        return

    with open(stats_path) as f:
        stats = json.load(f)

    ok(f"Snapshot loaded  (age: {_age(stats.get('timestamp',''))})")
    print()
    label("Virtual Capital",    f"\u20b9{stats.get('virtual_capital',0):,.2f}", G)
    label("Available Margin",   f"\u20b9{stats.get('available_margin',0):,.2f}", G)
    label("Used Margin",        f"\u20b9{stats.get('used_margin',0):,.2f}", Y)
    label("Unrealized P&L",     f"\u20b9{stats.get('unrealized_pnl_inr',0):+,.2f}")
    label("Today Realized",     f"\u20b9{stats.get('day_realized_pnl_inr',0):+,.2f}")
    label("Total P&L",          f"\u20b9{stats.get('total_pnl_inr',0):+,.2f}  ({stats.get('total_pnl_pct',0):+.4f}%)")
    label("Open positions",     str(stats.get('open_positions_count', 0)))
    label("Realized charges",   f"\u20b9{stats.get('realized_charges',0):,.2f}", R)

    positions = stats.get("positions", [])
    pending   = stats.get("pending_positions", [])

    if positions:
        print(f"\n{C}  Open Positions:{RST}")
        fmt = f"  {{:<12}} {{:<6}} {{:>6}} {{:>10}} {{:>12}} {{:>8}}"
        print(fmt.format("Ticker","Side","Qty","Entry","Unrealised","PnL%"))
        print(f"  {DIM}{'-'*60}{RST}")
        for p in positions:
            pct = p.get("unrealized_pnl_pct", 0)
            col = G if pct >= 0 else R
            print(fmt.format(
                p.get("ticker","?").replace(".NS",""),
                p.get("side","?"),
                str(p.get("quantity","?")),
                f"\u20b9{p.get('entry_price',0):.2f}",
                f"\u20b9{p.get('unrealized_pnl_inr',0):.2f}",
                f"{col}{pct:+.2f}%{RST}",
            ))

    if pending:
        print(f"\n{Y}  Pending Confirmations:{RST}")
        for p in pending:
            print(f"  {W}{p.get('ticker','?').replace('.NS',''):<12}{RST}"
                  f"  {p.get('side','?'):<6}"
                  f"  \u20b9{p.get('entry_price',0):.2f}")

    if not positions and not pending:
        info("No open or pending positions")


def _age(iso_str):
    try:
        dt  = datetime.fromisoformat(iso_str)
        sec = int((datetime.now() - dt).total_seconds())
        if sec < 60:  return f"{sec}s ago"
        if sec < 3600: return f"{sec//60}m ago"
        return f"{sec//3600}h ago"
    except Exception:
        return "unknown"


def run_all(sandbox_client, data_client):
    check_tokens()
    test_orderbook(sandbox_client)
    test_margins_note()
    test_positions_note(sandbox_client)
    test_live_price(data_client, "RELIANCE.NS")
    test_instrument_search(data_client, "RELIANCE.NS")
    show_engine_snapshot()

# ─────────────────────────────────────────────────────────────────────────────
# Interactive menu
# ─────────────────────────────────────────────────────────────────────────────

MENU = f"""
{C}{BOLD}╬═══════════════════════════════════════════════════════════╬
   UPSTOX SANDBOX — DEBUG CONSOLE
╬═══════════════════════════════════════════════════════════╬{RST}

{DIM}  ── SANDBOX ORDER MANAGEMENT (all confirmed working) ──{RST}
{W}  [2]{RST}  Order book        {G}✔ list all today's orders{RST}
{W}  [7]{RST}  Place LIMIT order {G}✔  e.g. 7 RELIANCE LONG 1{RST}
{W}  [9]{RST}  Place SL-M order  {G}✔  e.g. 9 TCS LONG 1{RST}   (stop-loss market)
{W}  [ms]{RST} Market SELL       {G}✔  e.g. ms INFY 2{RST}       (immediate fill)
{W}  [m]{RST}  Modify order      {G}✔  interactive — change price/trigger/qty{RST}
{W}  [c]{RST}  Cancel order      {G}✔  e.g. c 260521232159126{RST}
{W}  [ca]{RST} Cancel ALL        {G}✔  cancel every pending order{RST}

{DIM}  ── MARKET DATA (analytics token) ──{RST}
{W}  [5]{RST}  Live market quote            e.g. 5 INFY
{W}  [6]{RST}  Instrument key search        e.g. 6 HDFCBANK

{DIM}  ── DIAGNOSTICS ──{RST}
{W}  [1]{RST}  Token status + expiry check
{W}  [3]{RST}  Funds & margin  {Y}(not in sandbox → shows engine snapshot){RST}
{W}  [4]{RST}  Positions       {Y}(not in sandbox → shows engine snapshot){RST}
{W}  [s]{RST}  Engine virtual snapshot  (reads upstox_stats.json)
{W}  [8]{RST}  Run ALL tests
{W}  [0]{RST}  Exit

{DIM}  Ticker: NSE symbol only, e.g.  7 RELIANCE LONG 1  |  9 TCS SHORT 2  |  ms INFY 1{RST}
"""

def parse_args(parts):
    ticker = "RELIANCE.NS"; side = "LONG"; qty = 1
    if len(parts) > 1:
        raw = parts[1].upper()
        ticker = raw if raw.endswith(".NS") else raw + ".NS"
    if len(parts) > 2:
        s = parts[2].upper()
        side = "SHORT" if s in ("SHORT","SELL") else "LONG"
    if len(parts) > 3:
        try: qty = int(parts[3])
        except ValueError: pass
    return ticker, side, qty

def main():
    print(MENU)
    sandbox_client  = build_sandbox_client()
    data_client     = build_analytics_client()

    while True:
        try:
            raw = input(f"\n{C}  >{RST} ").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{DIM}  Bye!{RST}\n"); break
        if not raw: continue
        parts  = raw.split()
        choice = parts[0].lower()

        if   choice == "0": print(f"\n{DIM}  Bye!{RST}\n"); break
        elif choice == "1": check_tokens()
        elif choice == "2": test_orderbook(sandbox_client)
        elif choice == "3": test_margins_note()
        elif choice == "4": test_positions_note(sandbox_client)
        elif choice == "5":
            ticker, _, _ = parse_args(parts)
            test_live_price(data_client, ticker)
        elif choice == "6":
            ticker, _, _ = parse_args(parts)
            test_instrument_search(data_client, ticker)
        elif choice == "7":
            ticker, side, qty = parse_args(parts)
            confirm = input(
                f"\n{Y}  Place REAL sandbox LIMIT order: {side} {qty}x {ticker}. Continue? [y/N]: {RST}"
            ).strip().lower()
            if confirm == "y":
                test_place_order(sandbox_client, data_client, ticker, side, qty)
            else:
                info("Cancelled.")
        elif choice == "9":
            ticker, side, qty = parse_args(parts)
            confirm = input(
                f"\n{Y}  Place SL-M order: {side} {qty}x {ticker}. Continue? [y/N]: {RST}"
            ).strip().lower()
            if confirm == "y":
                test_place_sl_order(sandbox_client, data_client, ticker, side, qty)
            else:
                info("Cancelled.")
        elif choice == "ms":
            # ms TICKER QTY
            ticker = "RELIANCE.NS"; qty = 1
            if len(parts) > 1:
                raw = parts[1].upper()
                ticker = raw if raw.endswith(".NS") else raw + ".NS"
            if len(parts) > 2:
                try: qty = int(parts[2])
                except ValueError: pass
            confirm = input(
                f"\n{Y}  Place MARKET SELL: {qty}x {ticker}. Continue? [y/N]: {RST}"
            ).strip().lower()
            if confirm == "y":
                test_market_sell(sandbox_client, data_client, ticker, qty)
            else:
                info("Cancelled.")
        elif choice == "m":
            # distinguish between 'm' (menu) and 'ms' (market sell) — 'ms' already handled
            print(MENU)
        elif choice == "c":
            if len(parts) > 1:
                test_cancel_order(sandbox_client, parts[1])
            else:
                oid = input(f"  {C}Order ID to cancel: {RST}").strip()
                if oid:
                    test_cancel_order(sandbox_client, oid)
        elif choice == "ca":
            test_cancel_all(sandbox_client)
        elif choice == "mo" or choice == "mod":
            test_modify_order(sandbox_client)
        elif choice == "s": show_engine_snapshot()
        elif choice == "8": run_all(sandbox_client, data_client)
        else: warn(f"Unknown option '{choice}'. Type '0' to exit or press Enter to re-show menu.")

if __name__ == "__main__":
    main()
