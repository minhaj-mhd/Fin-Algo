import sqlite3
import os
from datetime import datetime, timedelta

DB_PATH = 'data/vanguard_trades.db'
START_DATE_FILTER = '2026-05-21'

def init_db():
    """Initializes the SQLite database with the required schema."""
    os.makedirs('data', exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    cursor = conn.cursor()
    
    # Create trades table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id TEXT UNIQUE,
            timestamp TEXT,
            ticker TEXT,
            side TEXT,
            tech_score REAL,
            nlp_sentiment REAL,
            entry_price REAL,
            exit_price REAL,
            peak_price REAL,
            peak_profit_pct REAL,
            final_profit_pct REAL,
            exit_time TEXT,
            status TEXT,
            comment TEXT,
            one_hour_prob TEXT,
            quantity INTEGER,
            net_pnl_amt REAL,
            margin_used REAL,
            tv_sentiment TEXT,
            pending_since TEXT,
            extension_count INTEGER,
            extended_exit_time TEXT,
            extension_pending INTEGER,
            stop_loss_pct REAL,
            take_profit_pct REAL,
            trailing_active INTEGER,
            breakeven_locked INTEGER,
            buy_brokerage REAL,
            long_score REAL,
            short_score REAL,
            strategy_id INTEGER
        )
    ''')
    
    # Check for missing column and add if necessary
    try:
        cursor.execute('ALTER TABLE trades ADD COLUMN tv_sentiment TEXT')
    except sqlite3.OperationalError:
        pass # Already exists
    
    # Create system_stats table for historical tracking
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS system_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            virtual_capital REAL,
            used_margin REAL,
            realized_charges REAL
        )
    ''')
    
    conn.commit()
    
    # Migration: Add new columns if they don't exist
    for col, col_type in [
        ('one_hour_prob', 'TEXT'),
        ('quantity', 'INTEGER'),
        ('net_pnl_amt', 'REAL'),
        ('margin_used', 'REAL'),
        ('buy_brokerage', 'REAL'),
        ('exit_price', 'REAL'),
        ('pending_since', 'TEXT'),
        ('extension_count', 'INTEGER'),
        ('extended_exit_time', 'TEXT'),
        ('extension_pending', 'INTEGER'),
        ('stop_loss_pct', 'REAL'),
        ('take_profit_pct', 'REAL'),
        ('trailing_active', 'INTEGER'),
        ('breakeven_locked', 'INTEGER'),
        ('long_score', 'REAL'),
        ('short_score', 'REAL'),
        ('strategy_id', 'INTEGER'),
    ]:
        try:
            cursor.execute(f'ALTER TABLE trades ADD COLUMN {col} {col_type}')
            conn.commit()
        except sqlite3.OperationalError:
            pass # Column already exists
            
    conn.close()
    print(f"Database initialized at {DB_PATH}")

def log_trade(trade_data):
    """Inserts a new trade or updates an existing one."""
    conn = sqlite3.connect(DB_PATH, timeout=10)
    cursor = conn.cursor()
    
    # Use REPLACE to update if the trade_id already exists (for status updates)
    cursor.execute('''
        INSERT OR REPLACE INTO trades (
            trade_id, timestamp, ticker, side, tech_score, nlp_sentiment, 
            entry_price, exit_price, peak_price, peak_profit_pct, 
            final_profit_pct, exit_time, status, comment, one_hour_prob,
            quantity, net_pnl_amt, margin_used, tv_sentiment,
            pending_since, extension_count, extended_exit_time, extension_pending,
            stop_loss_pct, take_profit_pct, trailing_active, breakeven_locked, buy_brokerage,
            long_score, short_score, strategy_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        trade_data.get('trade_id'),
        trade_data.get('timestamp'),
        trade_data.get('ticker'),
        trade_data.get('side'),
        trade_data.get('tech_score'),
        trade_data.get('nlp_sentiment'),
        trade_data.get('entry_price'),
        trade_data.get('exit_price'),
        trade_data.get('peak_price'),
        trade_data.get('peak_profit_pct'),
        trade_data.get('final_profit_pct'),
        trade_data.get('exit_time'),
        trade_data.get('status'),
        trade_data.get('comment'),
        trade_data.get('one_hour_prob'),
        trade_data.get('quantity'),
        trade_data.get('net_pnl_amt'),
        trade_data.get('margin_used'),
        trade_data.get('tv_sentiment'),
        trade_data.get('pending_since'),
        trade_data.get('extension_count'),
        trade_data.get('extended_exit_time'),
        1 if trade_data.get('extension_pending') else 0,
        trade_data.get('stop_loss_pct'),
        trade_data.get('take_profit_pct'),
        1 if trade_data.get('trailing_active') else 0,
        1 if trade_data.get('breakeven_locked') else 0,
        trade_data.get('buy_brokerage'),
        trade_data.get('long_score'),
        trade_data.get('short_score'),
        trade_data.get('strategy_id')
    ))
    
    conn.commit()
    conn.close()

def get_recent_trades(limit=50):
    """Fetches the most recent trades for the dashboard."""
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row # Return as dictionary-like objects
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM trades ORDER BY timestamp DESC LIMIT ?', (limit,))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows

def get_performance_stats():
    """Calculates granular performance metrics from the SQLite DB."""
    conn = sqlite3.connect(DB_PATH, timeout=10)
    cursor = conn.cursor()
    
    # 1. TOTAL STATS (filtered from START_DATE_FILTER onwards)
    cursor.execute('SELECT SUM(final_profit_pct), COUNT(*) FROM trades WHERE status IN ("CLOSED", "STOP_LOSS", "TAKE_PROFIT") AND trade_id LIKE "T-%" AND timestamp >= ?', (START_DATE_FILTER,))
    row = cursor.fetchone()
    total_alpha = row[0] or 0.0
    total_trades = row[1]
    
    # 2. DAILY STATS (Today)
    today_str = datetime.now().strftime('%Y-%m-%d')
    
    # Daily Total
    cursor.execute('SELECT SUM(final_profit_pct), COUNT(*) FROM trades WHERE status IN ("CLOSED", "STOP_LOSS", "TAKE_PROFIT") AND trade_id LIKE "T-%" AND timestamp LIKE ?', (f'{today_str}%',))
    d_row = cursor.fetchone()
    daily_alpha = d_row[0] or 0.0
    daily_trades = d_row[1]
    
    # Daily Breakdown (Long vs Short)
    cursor.execute('SELECT SUM(final_profit_pct) FROM trades WHERE status IN ("CLOSED", "STOP_LOSS", "TAKE_PROFIT") AND side="LONG" AND trade_id LIKE "T-%" AND timestamp LIKE ?', (f'{today_str}%',))
    daily_long = cursor.fetchone()[0] or 0.0
    cursor.execute('SELECT SUM(final_profit_pct) FROM trades WHERE status IN ("CLOSED", "STOP_LOSS", "TAKE_PROFIT") AND side="SHORT" AND trade_id LIKE "T-%" AND timestamp LIKE ?', (f'{today_str}%',))
    daily_short = cursor.fetchone()[0] or 0.0

    # 3. WEEKLY STATS (From last Monday, but never before START_DATE_FILTER)
    # Get current date and find last Monday
    now = datetime.now()
    monday = now - timedelta(days=now.weekday())
    monday_str = monday.strftime('%Y-%m-%d')
    # Clamp: weekly window cannot go before the system baseline date
    weekly_start = max(monday_str, START_DATE_FILTER)
    
    # Weekly Total
    cursor.execute('SELECT SUM(final_profit_pct), COUNT(*) FROM trades WHERE status IN ("CLOSED", "STOP_LOSS", "TAKE_PROFIT") AND trade_id LIKE "T-%" AND timestamp >= ?', (weekly_start,))
    w_row = cursor.fetchone()
    weekly_alpha = w_row[0] or 0.0
    weekly_trades = w_row[1]
    
    # Weekly Breakdown (Long vs Short)
    cursor.execute('SELECT SUM(final_profit_pct) FROM trades WHERE status IN ("CLOSED", "STOP_LOSS", "TAKE_PROFIT") AND side="LONG" AND trade_id LIKE "T-%" AND timestamp >= ?', (weekly_start,))
    weekly_long = cursor.fetchone()[0] or 0.0
    cursor.execute('SELECT SUM(final_profit_pct) FROM trades WHERE status IN ("CLOSED", "STOP_LOSS", "TAKE_PROFIT") AND side="SHORT" AND trade_id LIKE "T-%" AND timestamp >= ?', (weekly_start,))
    weekly_short = cursor.fetchone()[0] or 0.0
    
    # AI VETOED STATS (filtered from START_DATE_FILTER onwards)
    # Total Vetoed
    cursor.execute('SELECT SUM(final_profit_pct - 0.06), COUNT(*) FROM trades WHERE (status="VETOED" OR status="VETOED_EXPIRED") AND timestamp >= ?', (START_DATE_FILTER,))
    v_row = cursor.fetchone()
    total_vetoed_alpha = v_row[0] or 0.0
    total_vetoed_count = v_row[1]
    
    # Daily Vetoed
    cursor.execute('SELECT SUM(final_profit_pct - 0.06), COUNT(*) FROM trades WHERE (status="VETOED" OR status="VETOED_EXPIRED") AND timestamp LIKE ?', (f'{today_str}%',))
    dv_row = cursor.fetchone()
    daily_vetoed_alpha = dv_row[0] or 0.0
    daily_vetoed_count = dv_row[1]
    
    # S1 Daily Vetoed
    cursor.execute('SELECT SUM(final_profit_pct - 0.06), COUNT(*) FROM trades WHERE (status="VETOED" OR status="VETOED_EXPIRED") AND comment LIKE "%[S1-%" AND timestamp LIKE ?', (f'{today_str}%',))
    dv_s1_row = cursor.fetchone()
    s1_vetoed_alpha = dv_s1_row[0] or 0.0
    s1_vetoed_count = dv_s1_row[1]

    # S2 Daily Vetoed
    cursor.execute('SELECT SUM(final_profit_pct - 0.06), COUNT(*) FROM trades WHERE (status="VETOED" OR status="VETOED_EXPIRED") AND comment LIKE "%[S2-%" AND timestamp LIKE ?', (f'{today_str}%',))
    dv_s2_row = cursor.fetchone()
    s2_vetoed_alpha = dv_s2_row[0] or 0.0
    s2_vetoed_count = dv_s2_row[1]
    
    # Weekly Vetoed (clamped to baseline start)
    cursor.execute('SELECT SUM(final_profit_pct - 0.06), COUNT(*) FROM trades WHERE (status="VETOED" OR status="VETOED_EXPIRED") AND timestamp >= ?', (weekly_start,))
    wv_row = cursor.fetchone()
    weekly_vetoed_alpha = wv_row[0] or 0.0
    weekly_vetoed_count = wv_row[1]

    conn.close()
    
    return {
        'total': {'alpha': round(total_alpha, 2), 'count': total_trades},
        'daily': {
            'alpha': round(daily_alpha, 2), 
            'long': round(daily_long, 2), 
            'short': round(daily_short, 2),
            'count': daily_trades
        },
        'weekly': {
            'alpha': round(weekly_alpha, 2), 
            'long': round(weekly_long, 2), 
            'short': round(weekly_short, 2),
            'count': weekly_trades
        },
        'vetoed': {
            'total': {'alpha': round(total_vetoed_alpha, 2), 'count': total_vetoed_count},
            'daily': {'alpha': round(daily_vetoed_alpha, 2), 'count': daily_vetoed_count},
            'daily_s1': {'alpha': round(s1_vetoed_alpha, 2), 'count': s1_vetoed_count},
            'daily_s2': {'alpha': round(s2_vetoed_alpha, 2), 'count': s2_vetoed_count},
            'weekly': {'alpha': round(weekly_vetoed_alpha, 2), 'count': weekly_vetoed_count}
        }
    }

def get_trades_by_status(status="OPEN", limit=50):
    """Fetches trades filtered by status (can be a string or list)."""
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    if isinstance(status, list):
        query = f'SELECT * FROM trades WHERE status IN ({",".join(["?"] * len(status))}) ORDER BY timestamp DESC LIMIT ?'
        cursor.execute(query, status + [limit])
    else:
        cursor.execute('SELECT * FROM trades WHERE status = ? ORDER BY timestamp DESC LIMIT ?', (status, limit))
        
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows

def get_trades_by_strategy(strategy_id, limit=200):
    """Fetches trades (taken and vetoed) filtered by strategy_id."""
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM trades WHERE strategy_id = ? ORDER BY timestamp DESC LIMIT ?', (strategy_id, limit))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_trades_for_ticker(ticker, limit=50):
    """Fetches all trades taken for a specific ticker."""
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM trades WHERE ticker = ? ORDER BY timestamp DESC LIMIT ?', (ticker, limit))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows

def get_ticker_performance(ticker):
    """Calculates specific performance metrics for a single ticker."""
    conn = sqlite3.connect(DB_PATH, timeout=10)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT 
            COUNT(*), 
            SUM(final_profit_pct), 
            AVG(final_profit_pct),
            MAX(final_profit_pct),
            MIN(final_profit_pct)
        FROM trades 
        WHERE ticker = ? AND status IN ("CLOSED", "STOP_LOSS", "TAKE_PROFIT") AND timestamp >= ?
    ''', (ticker, START_DATE_FILTER))
    
    row = cursor.fetchone()
    conn.close()
    
    return {
        'count': row[0] or 0,
        'total_alpha': round(row[1] or 0.0, 2),
        'avg_pnl': round(row[2] or 0.0, 2),
        'best_trade': round(row[3] or 0.0, 2),
        'worst_trade': round(row[4] or 0.0, 2)
    }

def get_detailed_performance():
    """Aggregates performance by month, week, and day."""
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # 1. Monthly Aggregation
    cursor.execute('''
        SELECT 
            strftime('%Y-%m', timestamp) as period,
            SUM(final_profit_pct) as total_pnl,
            COUNT(*) as trade_count,
            SUM(CASE WHEN final_profit_pct > 0 THEN 1 ELSE 0 END) as wins
        FROM trades 
        WHERE status IN ("CLOSED", "STOP_LOSS", "TAKE_PROFIT") AND timestamp >= ?
        GROUP BY period
        ORDER BY period DESC
    ''', (START_DATE_FILTER,))
    monthly = [dict(row) for row in cursor.fetchall()]
    
    # 2. Weekly Aggregation
    cursor.execute('''
        SELECT 
            strftime('%Y-W%W', timestamp) as period,
            SUM(final_profit_pct) as total_pnl,
            COUNT(*) as trade_count,
            SUM(CASE WHEN final_profit_pct > 0 THEN 1 ELSE 0 END) as wins
        FROM trades 
        WHERE status IN ("CLOSED", "STOP_LOSS", "TAKE_PROFIT") AND timestamp >= ?
        GROUP BY period
        ORDER BY period DESC
    ''', (START_DATE_FILTER,))
    weekly = [dict(row) for row in cursor.fetchall()]
    
    # 3. Daily Aggregation
    cursor.execute('''
        SELECT 
            strftime('%Y-%m-%d', timestamp) as period,
            SUM(final_profit_pct) as total_pnl,
            COUNT(*) as trade_count,
            SUM(CASE WHEN final_profit_pct > 0 THEN 1 ELSE 0 END) as wins
        FROM trades 
        WHERE status IN ("CLOSED", "STOP_LOSS", "TAKE_PROFIT") AND timestamp >= ?
        GROUP BY period
        ORDER BY period DESC
    ''', (START_DATE_FILTER,))
    daily = [dict(row) for row in cursor.fetchall()]
    
    # 4. Cumulative Equity Curve Data
    cursor.execute('''
        SELECT timestamp, final_profit_pct
        FROM trades 
        WHERE status IN ("CLOSED", "STOP_LOSS", "TAKE_PROFIT") AND timestamp >= ?
        ORDER BY timestamp ASC
    ''', (START_DATE_FILTER,))
    raw_trades = cursor.fetchall()
    
    equity_curve = []
    current_cumulative = 0.0
    for row in raw_trades:
        current_cumulative += row[1]
        equity_curve.append({
            't': row[0],
            'y': round(current_cumulative, 2)
        })
        
    conn.close()
    
    return {
        'monthly': monthly,
        'weekly': weekly,
        'daily': daily,
        'equity_curve': equity_curve
    }

def log_system_stats(stats):
    """Records a snapshot of the system's financial state."""
    conn = sqlite3.connect(DB_PATH, timeout=10)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO system_stats (timestamp, virtual_capital, used_margin, realized_charges)
        VALUES (?, ?, ?, ?)
    ''', (
        stats.get('timestamp', datetime.now().isoformat()),
        stats.get('virtual_capital'),
        stats.get('used_margin'),
        stats.get('realized_charges')
    ))
    conn.commit()
    conn.close()


def get_today_realized_pnl():
    """Returns today's total realized net P&L in INR (sum of net_pnl_amt for CLOSED/STOP_LOSS trades)."""
    conn = sqlite3.connect(DB_PATH, timeout=10)
    cursor = conn.cursor()
    today_str = datetime.now().strftime('%Y-%m-%d')
    cursor.execute('''
        SELECT COALESCE(SUM(net_pnl_amt), 0.0)
        FROM trades
        WHERE status IN ("CLOSED", "STOP_LOSS", "TAKE_PROFIT")
          AND trade_id LIKE "T-%"
          AND timestamp LIKE ?
    ''', (f'{today_str}%',))
    result = cursor.fetchone()[0]
    conn.close()
    return float(result or 0.0)


def get_portfolio_summary():
    """Returns a full portfolio summary for today: wins, losses, gross/net P&L, charges."""
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    today_str = datetime.now().strftime('%Y-%m-%d')

    # Today's closed trades
    cursor.execute('''
        SELECT
            COUNT(*)                                             AS total_trades,
            SUM(CASE WHEN net_pnl_amt > 0 THEN 1 ELSE 0 END)   AS wins,
            SUM(CASE WHEN net_pnl_amt <= 0 THEN 1 ELSE 0 END)  AS losses,
            COALESCE(SUM(net_pnl_amt), 0.0)                     AS net_pnl_inr,
            COALESCE(MAX(net_pnl_amt), 0.0)                     AS best_trade_inr,
            COALESCE(MIN(net_pnl_amt), 0.0)                     AS worst_trade_inr,
            COALESCE(AVG(net_pnl_amt), 0.0)                     AS avg_pnl_inr
        FROM trades
        WHERE status IN ("CLOSED", "STOP_LOSS", "TAKE_PROFIT")
          AND trade_id LIKE "T-%"
          AND timestamp LIKE ?
    ''', (f'{today_str}%',))
    row = dict(cursor.fetchone())

    # Today's total charges calculated as: gross_pnl - net_pnl
    cursor.execute('''
        SELECT COALESCE(SUM(
            CASE 
                WHEN side = 'LONG' THEN (exit_price - entry_price) * quantity - net_pnl_amt
                ELSE (entry_price - exit_price) * quantity - net_pnl_amt
            END
        ), 0.0)
        FROM trades
        WHERE status IN ("CLOSED", "STOP_LOSS", "TAKE_PROFIT")
          AND trade_id LIKE "T-%"
          AND timestamp LIKE ?
    ''', (f'{today_str}%',))
    today_charges = float(cursor.fetchone()[0] or 0.0)

    # Fetch today's closed trades detail
    cursor.execute('''
        SELECT ticker, side, quantity, entry_price, exit_price,
               final_profit_pct, net_pnl_amt, status, timestamp, comment, one_hour_prob,
               tech_score, long_score, short_score, strategy_id
        FROM trades
        WHERE status IN ("CLOSED", "STOP_LOSS", "TAKE_PROFIT")
          AND trade_id LIKE "T-%"
          AND timestamp LIKE ?
        ORDER BY timestamp DESC
        LIMIT 50
    ''', (f'{today_str}%',))
    today_trades = [dict(r) for r in cursor.fetchall()]

    conn.close()
    wins = int(row.get('wins') or 0)
    total = int(row.get('total_trades') or 0)
    return {
        'total_trades':   total,
        'wins':           wins,
        'losses':         int(row.get('losses') or 0),
        'win_rate':       round(wins / total * 100, 1) if total > 0 else 0.0,
        'net_pnl_inr':    round(float(row.get('net_pnl_inr') or 0.0), 2),
        'best_trade_inr': round(float(row.get('best_trade_inr') or 0.0), 2),
        'worst_trade_inr':round(float(row.get('worst_trade_inr') or 0.0), 2),
        'avg_pnl_inr':    round(float(row.get('avg_pnl_inr') or 0.0), 2),
        'today_charges_inr': round(today_charges, 2),
        'today_trades':   today_trades,
    }


def get_strategy_performance():
    """Aggregates performance metrics grouped by strategy_id."""
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT 
            COALESCE(strategy_id, -1) as strategy_id,
            COUNT(*) as trade_count,
            SUM(CASE WHEN final_profit_pct > 0 THEN 1 ELSE 0 END) as wins,
            SUM(final_profit_pct) as total_alpha,
            MAX(final_profit_pct) as best_trade,
            MIN(final_profit_pct) as worst_trade,
            AVG(final_profit_pct) as avg_pnl
        FROM trades
        WHERE status IN ("CLOSED", "STOP_LOSS", "TAKE_PROFIT") AND timestamp >= ?
        GROUP BY strategy_id
        ORDER BY strategy_id ASC
    ''', (START_DATE_FILTER,))
    
    strategies = [dict(row) for row in cursor.fetchall()]
    
    # Also fetch recent strategy trades
    cursor.execute('''
        SELECT ticker, side, entry_price, exit_price, final_profit_pct, status, timestamp, strategy_id, comment
        FROM trades
        WHERE status IN ("CLOSED", "STOP_LOSS", "TAKE_PROFIT") AND timestamp >= ? AND strategy_id IS NOT NULL
        ORDER BY timestamp DESC
        LIMIT 100
    ''', (START_DATE_FILTER,))
    
    recent_strategy_trades = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    
    return {
        'stats': strategies,
        'recent_trades': recent_strategy_trades
    }

if __name__ == '__main__':
    init_db()
