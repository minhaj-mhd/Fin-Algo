import os
import sys
import json
from datetime import datetime
from flask import Flask, jsonify, render_template

# Add project root to path before importing local modules
sys.path.append(os.getcwd())

from scripts.database_manager import (
    get_trades_by_status, get_performance_stats, get_trades_for_ticker, 
    get_ticker_performance, get_detailed_performance, get_portfolio_summary
)
from scripts.ticker_intelligence import get_ticker_analysis
from scripts.tickers import TICKERS
from scripts.terminal_utils import log

# Configuration
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../templates'))
log(f"[INFO] Using template directory: {template_dir}")
app = Flask(__name__, template_folder=template_dir)

@app.route('/')
def index():
    return render_template('vanguard_v2.html')

@app.route('/api/tickers')
def get_tickers():
    return jsonify([t.replace('.NS', '') for t in TICKERS])

@app.route('/api/upstox/stats')
def get_upstox_stats():
    """Returns full portfolio state: capital, margins, open positions, today's P&L."""
    stats_path = os.path.join(os.getcwd(), 'upstox_stats.json')
    
    # Load base stats from the engine's persisted JSON
    if os.path.exists(stats_path):
        with open(stats_path, 'r') as f:
            stats = json.load(f)
    else:
        stats = {
            "initial_capital":      99517.68,
            "virtual_capital":      99517.68,
            "used_margin":          0.0,
            "available_margin":     99517.68,
            "realized_charges":     0.0,
            "open_positions_count": 0,
            "unrealized_pnl_inr":   0.0,
            "day_realized_pnl_inr": 0.0,
            "total_pnl_inr":        0.0,
            "total_pnl_pct":        0.0,
            "positions":            [],
            "timestamp":            datetime.now().isoformat(),
        }

    # Merge with live DB portfolio summary (wins, losses, today's closed trades)
    try:
        summary = get_portfolio_summary()
        stats['today_summary'] = summary
    except Exception as e:
        log(f"[WARN] Portfolio summary error: {e}")
        stats['today_summary'] = {
            'total_trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0.0,
            'net_pnl_inr': 0.0, 'best_trade_inr': 0.0, 'worst_trade_inr': 0.0,
            'avg_pnl_inr': 0.0, 'today_trades': []
        }

    return jsonify(stats)

@app.route('/performance')
def performance_page():
    return render_template('performance.html')

@app.route('/api/performance_data')
def performance_data():
    try:
        data = get_detailed_performance()
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/upstox/orders')
def get_upstox_orders():
    try:
        from scripts.upstox_broker import UpstoxSandboxBroker
        broker = UpstoxSandboxBroker()
        orders = broker.get_order_book()
        
        # Handle Upstox SDK response objects
        if hasattr(orders, 'to_dict'):
            return jsonify(orders.to_dict())
        return jsonify(orders)
    except Exception as e:
        log(f"[ERROR] Order Book API Error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/vanguard_status')
def get_vanguard_status():
    try:
        # Separate Open, Closed, and Vetoed trades for the UI
        open_trades = get_trades_by_status("OPEN", 20)
        pending_trades = get_trades_by_status("PENDING_ENTRY", 20)
        closed_trades = get_trades_by_status(["CLOSED", "STOP_LOSS", "TAKE_PROFIT"], 50)
        
        # Filter vetoed trades to remove overlaps with open trades
        raw_vetoed = get_trades_by_status("VETOED", 20) + get_trades_by_status("VETOED_EXPIRED", 50)
        open_tickers = {t['ticker'] for t in open_trades}
        vetoed_trades = sorted([t for t in raw_vetoed if not (t['status'] == 'VETOED' and t['ticker'] in open_tickers)], 
                               key=lambda x: x['timestamp'], reverse=True)
        stats = get_performance_stats()
        
        return jsonify({
            'open_trades': open_trades,
            'pending_trades': pending_trades,
            'closed_trades': closed_trades,
            'vetoed_trades': vetoed_trades,
            'stats': stats,
            'system_health': 'ONLINE'
        })
    except Exception as e:
        log(f"[ERROR] Dashboard Data Error: {e}")
        return jsonify({'error': str(e), 'system_health': 'ERROR'})

@app.route('/live_scores')
def get_live_scores():
    try:
        if os.path.exists('data/latest_scores.json'):
            with open('data/latest_scores.json', 'r') as f:
                scores = json.load(f)
            return jsonify({'scores': scores})
        return jsonify({'scores': []})
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/ticker/<symbol>')
def ticker_detail(symbol):
    try:
        # Sanitize symbol
        symbol = symbol.upper()
        if not symbol.endswith('.NS') and '.' not in symbol:
            symbol += '.NS'
            
        # 1. Fetch trade history
        history = get_trades_for_ticker(symbol, 20)
        
        # 2. Fetch performance stats
        perf = get_ticker_performance(symbol)
        
        # 3. Retrieve actual live price
        live_price = None
        if os.path.exists('data/latest_scores.json'):
            try:
                with open('data/latest_scores.json', 'r') as f:
                    scores = json.load(f)
                ticker_data = next((s for s in scores if s['ticker'] == symbol), None)
                if ticker_data:
                    live_price = ticker_data.get('Close')
            except Exception as e:
                log(f"[WARN] Failed to load live price from scores: {e}")
                
        if live_price is None:
            try:
                from scripts.upstox_broker import UpstoxSandboxBroker
                broker = UpstoxSandboxBroker()
                live_price = broker.get_live_price(symbol)
            except Exception as e:
                log(f"[WARN] Failed to fetch live price from broker for {symbol}: {e}")
        
        return render_template('ticker_detail.html', 
                             ticker=symbol, 
                             history=history, 
                             perf=perf,
                             live_price=live_price)
    except Exception as e:
        log(f"[ERROR] Ticker Detail Error: {e}")
        return f"Error loading detail page for {symbol}: {e}", 500

@app.route('/api/intelligence/<symbol>')
def get_intelligence(symbol):
    try:
        symbol = symbol.upper()
        # Basic history check to determine bias
        history = get_trades_for_ticker(symbol, 1)
        bias = history[0].get('side', 'NEUTRAL') if history else 'NEUTRAL'
        
        intelligence = get_ticker_analysis(symbol, bias)
        return jsonify(intelligence)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model_snapshot')
def api_model_snapshot():
    """Returns current XGBoost model sentiment derived from conviction scores."""
    try:
        from scripts.market_tracker import get_model_snapshot
        return jsonify(get_model_snapshot())
    except Exception as e:
        log(f"[ERROR] Model Snapshot API Error: {e}")
        return jsonify({'error': str(e), 'overall_sentiment': 'UNAVAILABLE'}), 500

@app.route('/api/market_snapshot')
def api_market_snapshot():
    """Returns NIFTY 50 index data and market sentiment (cached 5 min)."""
    try:
        from scripts.market_tracker import get_market_snapshot
        return jsonify(get_market_snapshot())
    except Exception as e:
        log(f"[ERROR] Market Snapshot API Error: {e}")
        return jsonify({'error': str(e), 'market_sentiment': 'UNAVAILABLE'}), 500

if __name__ == '__main__':
    if not os.path.exists('data'): os.makedirs('data')
    log("============================================================")
    log("VANGUARD INTELLIGENCE DASHBOARD: http://127.0.0.1:5001")
    log("============================================================")
    app.run(debug=False, port=5001, host='0.0.0.0', threaded=True)
