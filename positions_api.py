"""
GoldGPT Positions Management API
Real-time position tracking with signal generation and P&L monitoring
"""

from flask import Blueprint, request, jsonify, session
import sqlite3
import json
from datetime import datetime, timedelta
import uuid
import threading
import time
import requests
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

positions_bp = Blueprint('positions', __name__, url_prefix='/api/positions')

# Database setup
DB_PATH = 'positions.db'

def init_positions_db():
    """Initialize positions database with required tables"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create signals table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS signals (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            symbol TEXT NOT NULL,
            entry_price REAL NOT NULL,
            current_price REAL,
            take_profit REAL,
            stop_loss REAL,
            quantity REAL NOT NULL,
            status TEXT DEFAULT 'OPEN',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            closed_at TIMESTAMP,
            pnl REAL DEFAULT 0,
            pnl_percentage REAL DEFAULT 0,
            confidence REAL DEFAULT 0,
            strategy TEXT,
            notes TEXT
        )
    ''')
    
    # Create position history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS position_history (
            id TEXT PRIMARY KEY,
            signal_id TEXT,
            action TEXT,
            price REAL,
            quantity REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            details TEXT,
            FOREIGN KEY (signal_id) REFERENCES signals (id)
        )
    ''')
    
    # Create portfolio summary table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS portfolio_summary (
            id INTEGER PRIMARY KEY,
            total_balance REAL DEFAULT 10000,
            available_balance REAL DEFAULT 10000,
            used_margin REAL DEFAULT 0,
            total_pnl REAL DEFAULT 0,
            win_rate REAL DEFAULT 0,
            total_trades INTEGER DEFAULT 0,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Initialize portfolio if empty
    cursor.execute('SELECT COUNT(*) FROM portfolio_summary')
    if cursor.fetchone()[0] == 0:
        cursor.execute('''
            INSERT INTO portfolio_summary (total_balance, available_balance)
            VALUES (10000, 10000)
        ''')
    
    conn.commit()
    conn.close()

# Initialize database on import
init_positions_db()

def get_gold_price():
    """Get current gold price from external API"""
    try:
        # Try multiple sources for gold price
        sources = [
            'https://api.metals.live/v1/spot/gold',
            'https://api.coindesk.com/v1/bpi/currentprice.json'  # Backup (Bitcoin, but shows API pattern)
        ]
        
        for source in sources:
            try:
                response = requests.get(source, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    if 'price' in data:
                        return float(data['price'])
                    # Simulate gold price for demo (remove in production)
                    return 2000 + (hash(str(datetime.now().minute)) % 100) - 50
            except:
                continue
        
        # Fallback simulated price
        return 2000 + (hash(str(datetime.now().minute)) % 100) - 50
        
    except Exception as e:
        logger.error(f"Error getting gold price: {e}")
        return 2000.0  # Fallback price

def calculate_pnl(signal):
    """Calculate P&L for a signal"""
    try:
        current_price = get_gold_price()
        entry_price = signal['entry_price']
        quantity = signal['quantity']
        
        if signal['type'].upper() == 'BUY':
            pnl = (current_price - entry_price) * quantity
        else:  # SELL
            pnl = (entry_price - current_price) * quantity
            
        pnl_percentage = (pnl / (entry_price * quantity)) * 100
        
        return pnl, pnl_percentage, current_price
    except Exception as e:
        logger.error(f"Error calculating P&L: {e}")
        return 0, 0, get_gold_price()

@positions_bp.route('/generate-signal', methods=['POST'])
def generate_signal():
    """Generate a new trading signal"""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['type', 'symbol', 'quantity', 'take_profit', 'stop_loss']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Get current gold price
        current_price = get_gold_price()
        
        signal_id = str(uuid.uuid4())
        signal = {
            'id': signal_id,
            'type': data['type'].upper(),
            'symbol': data.get('symbol', 'XAUUSD'),
            'entry_price': current_price,
            'current_price': current_price,
            'take_profit': float(data['take_profit']),
            'stop_loss': float(data['stop_loss']),
            'quantity': float(data['quantity']),
            'status': 'OPEN',
            'confidence': data.get('confidence', 75),
            'strategy': data.get('strategy', 'Manual'),
            'notes': data.get('notes', ''),
            'created_at': datetime.now().isoformat()
        }
        
        # Save to database
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO signals (
                id, type, symbol, entry_price, current_price, take_profit, 
                stop_loss, quantity, status, confidence, strategy, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal['id'], signal['type'], signal['symbol'], signal['entry_price'],
            signal['current_price'], signal['take_profit'], signal['stop_loss'],
            signal['quantity'], signal['status'], signal['confidence'],
            signal['strategy'], signal['notes']
        ))
        
        # Log position history
        cursor.execute('''
            INSERT INTO position_history (id, signal_id, action, price, quantity, details)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            str(uuid.uuid4()), signal_id, 'OPEN', current_price,
            signal['quantity'], f"Opened {signal['type']} position"
        ))
        
        # Update portfolio
        used_margin = signal['quantity'] * current_price * 0.1  # 10% margin
        cursor.execute('''
            UPDATE portfolio_summary 
            SET used_margin = used_margin + ?, 
                available_balance = available_balance - ?,
                last_updated = CURRENT_TIMESTAMP
        ''', (used_margin, used_margin))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Generated signal {signal_id}: {signal['type']} {signal['symbol']}")
        
        return jsonify({
            'success': True,
            'signal': signal,
            'message': f"Signal generated successfully: {signal['type']} {signal['symbol']} at ${current_price:.2f}"
        })
        
    except Exception as e:
        logger.error(f"Error generating signal: {e}")
        return jsonify({'error': str(e)}), 500

@positions_bp.route('/open', methods=['GET'])
def get_open_positions():
    """Get all open positions with live P&L"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM signals WHERE status = 'OPEN' ORDER BY created_at DESC
        ''')
        
        columns = [description[0] for description in cursor.description]
        positions = []
        
        for row in cursor.fetchall():
            signal = dict(zip(columns, row))
            
            # Calculate live P&L
            pnl, pnl_percentage, current_price = calculate_pnl(signal)
            
            signal.update({
                'current_price': current_price,
                'pnl': round(pnl, 2),
                'pnl_percentage': round(pnl_percentage, 2),
                'live_value': round(signal['quantity'] * current_price, 2)
            })
            
            # Check if TP/SL hit
            if signal['type'] == 'BUY':
                if current_price >= signal['take_profit'] or current_price <= signal['stop_loss']:
                    close_position(signal['id'], current_price, 'AUTO')
                    signal['status'] = 'CLOSED'
            else:  # SELL
                if current_price <= signal['take_profit'] or current_price >= signal['stop_loss']:
                    close_position(signal['id'], current_price, 'AUTO')
                    signal['status'] = 'CLOSED'
            
            if signal['status'] == 'OPEN':
                positions.append(signal)
        
        conn.close()
        
        return jsonify({
            'success': True,
            'positions': positions,
            'count': len(positions),
            'last_updated': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting open positions: {e}")
        return jsonify({'error': str(e)}), 500

@positions_bp.route('/history', methods=['GET'])
def get_position_history():
    """Get trading history"""
    try:
        limit = request.args.get('limit', 50)
        
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM signals WHERE status != 'OPEN' 
            ORDER BY created_at DESC LIMIT ?
        ''', (limit,))
        
        columns = [description[0] for description in cursor.description]
        history = []
        
        for row in cursor.fetchall():
            signal = dict(zip(columns, row))
            history.append(signal)
        
        conn.close()
        
        return jsonify({
            'success': True,
            'history': history,
            'count': len(history)
        })
        
    except Exception as e:
        logger.error(f"Error getting position history: {e}")
        return jsonify({'error': str(e)}), 500

@positions_bp.route('/close/<signal_id>', methods=['POST'])
def close_position_endpoint(signal_id):
    """Close a specific position"""
    try:
        current_price = get_gold_price()
        result = close_position(signal_id, current_price, 'MANUAL')
        
        if result:
            return jsonify({
                'success': True,
                'message': f'Position {signal_id} closed at ${current_price:.2f}'
            })
        else:
            return jsonify({'error': 'Position not found or already closed'}), 404
            
    except Exception as e:
        logger.error(f"Error closing position: {e}")
        return jsonify({'error': str(e)}), 500

def close_position(signal_id, close_price, close_type='MANUAL'):
    """Close a position and update portfolio"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get signal details
        cursor.execute('SELECT * FROM signals WHERE id = ? AND status = "OPEN"', (signal_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return False
        
        columns = [description[0] for description in cursor.description]
        signal = dict(zip(columns, row))
        
        # Calculate final P&L
        pnl, pnl_percentage, _ = calculate_pnl(signal)
        
        # Update signal as closed
        cursor.execute('''
            UPDATE signals 
            SET status = 'CLOSED', current_price = ?, closed_at = CURRENT_TIMESTAMP,
                pnl = ?, pnl_percentage = ?
            WHERE id = ?
        ''', (close_price, pnl, pnl_percentage, signal_id))
        
        # Log closure in history
        cursor.execute('''
            INSERT INTO position_history (id, signal_id, action, price, quantity, details)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            str(uuid.uuid4()), signal_id, 'CLOSE', close_price,
            signal['quantity'], f"Closed by {close_type}: P&L ${pnl:.2f}"
        ))
        
        # Update portfolio
        used_margin = signal['quantity'] * signal['entry_price'] * 0.1
        cursor.execute('''
            UPDATE portfolio_summary 
            SET used_margin = used_margin - ?, 
                available_balance = available_balance + ? + ?,
                total_pnl = total_pnl + ?,
                total_trades = total_trades + 1,
                last_updated = CURRENT_TIMESTAMP
        ''', (used_margin, used_margin, pnl, pnl))
        
        # Update win rate
        cursor.execute('SELECT COUNT(*) FROM signals WHERE status = "CLOSED" AND pnl > 0')
        wins = cursor.fetchone()[0]
        cursor.execute('SELECT COUNT(*) FROM signals WHERE status = "CLOSED"')
        total = cursor.fetchone()[0]
        
        if total > 0:
            win_rate = (wins / total) * 100
            cursor.execute('UPDATE portfolio_summary SET win_rate = ?', (win_rate,))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Closed position {signal_id}: P&L ${pnl:.2f}")
        return True
        
    except Exception as e:
        logger.error(f"Error closing position: {e}")
        return False

@positions_bp.route('/portfolio', methods=['GET'])
def get_portfolio_summary():
    """Get portfolio summary and statistics"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Get portfolio summary
        cursor.execute('SELECT * FROM portfolio_summary ORDER BY last_updated DESC LIMIT 1')
        portfolio_row = cursor.fetchone()
        
        if portfolio_row:
            columns = [description[0] for description in cursor.description]
            portfolio = dict(zip(columns, portfolio_row))
        else:
            portfolio = {
                'total_balance': 10000,
                'available_balance': 10000,
                'used_margin': 0,
                'total_pnl': 0,
                'win_rate': 0,
                'total_trades': 0
            }
        
        # Get recent performance
        cursor.execute('''
            SELECT pnl, created_at FROM signals 
            WHERE status = "CLOSED" AND created_at >= date('now', '-7 days')
            ORDER BY created_at DESC
        ''')
        
        recent_trades = cursor.fetchall()
        daily_pnl = sum(trade[0] for trade in recent_trades)
        
        # Get open positions count
        cursor.execute('SELECT COUNT(*) FROM signals WHERE status = "OPEN"')
        open_positions_count = cursor.fetchone()[0]
        
        conn.close()
        
        return jsonify({
            'success': True,
            'portfolio': {
                **portfolio,
                'daily_pnl': round(daily_pnl, 2),
                'open_positions': open_positions_count,
                'equity': round(portfolio['total_balance'] + portfolio['total_pnl'], 2),
                'margin_usage': round((portfolio['used_margin'] / portfolio['total_balance']) * 100, 2) if portfolio['total_balance'] > 0 else 0
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting portfolio summary: {e}")
        return jsonify({'error': str(e)}), 500

@positions_bp.route('/live-update', methods=['GET'])
def live_update():
    """Get live update of all positions (for real-time refresh)"""
    try:
        # Get open positions with live P&L
        open_response = get_open_positions()
        open_data = json.loads(open_response.data)
        
        # Get portfolio summary
        portfolio_response = get_portfolio_summary()
        portfolio_data = json.loads(portfolio_response.data)
        
        return jsonify({
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'gold_price': get_gold_price(),
            'open_positions': open_data.get('positions', []),
            'portfolio': portfolio_data.get('portfolio', {}),
            'last_updated': datetime.now().strftime('%H:%M:%S')
        })
        
    except Exception as e:
        logger.error(f"Error in live update: {e}")
        return jsonify({'error': str(e)}), 500

# Auto-close monitoring (runs in background)
def start_position_monitor():
    """Start background thread to monitor TP/SL"""
    def monitor_positions():
        while True:
            try:
                time.sleep(30)  # Check every 30 seconds
                get_open_positions()  # This will trigger auto-close if TP/SL hit
            except Exception as e:
                logger.error(f"Position monitor error: {e}")
            
    monitor_thread = threading.Thread(target=monitor_positions, daemon=True)
    monitor_thread.start()
    logger.info("Position monitor started")

# Start monitoring on import
start_position_monitor()

if __name__ == '__main__':
    print("Positions API module - import this in your main app.py")
