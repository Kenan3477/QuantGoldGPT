"""
Enhanced Signal Tracker System
Comprehensive signal lifecycle management with live P&L tracking and auto-closing
"""

import sqlite3
import logging
import time
import json
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import requests
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignalTracker:
    """Enhanced signal tracker with live P&L monitoring and auto-closing"""
    
    def __init__(self, db_path: str = "enhanced_signal_tracking.db"):
        self.db_path = db_path
        self.gold_api_url = "https://api.gold-api.com/price/XAU"
        self._init_database()
        logger.info("✅ Enhanced Signal Tracker initialized")
    
    def _init_database(self):
        """Initialize enhanced signal tracking database"""
        with self._get_db_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tracked_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id TEXT UNIQUE NOT NULL,
                    signal_type TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    current_price REAL,
                    take_profit REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    risk_amount REAL NOT NULL,
                    current_pnl REAL DEFAULT 0,
                    status TEXT DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    closed_at TIMESTAMP,
                    close_reason TEXT,
                    macro_indicators TEXT,
                    confidence_score REAL DEFAULT 0.75,
                    max_profit REAL DEFAULT 0,
                    max_drawdown REAL DEFAULT 0
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS signal_updates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id TEXT NOT NULL,
                    price REAL NOT NULL,
                    pnl REAL NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (signal_id) REFERENCES tracked_signals (signal_id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS signal_statistics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    total_signals INTEGER DEFAULT 0,
                    active_signals INTEGER DEFAULT 0,
                    winning_signals INTEGER DEFAULT 0,
                    losing_signals INTEGER DEFAULT 0,
                    total_pnl REAL DEFAULT 0,
                    win_rate REAL DEFAULT 0,
                    avg_win REAL DEFAULT 0,
                    avg_loss REAL DEFAULT 0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Insert initial statistics row if not exists
            conn.execute("""
                INSERT OR IGNORE INTO signal_statistics 
                (id, total_signals, active_signals) 
                VALUES (1, 0, 0)
            """)
            
            conn.commit()
            logger.info("✅ Enhanced signal tracking database initialized")
    
    @contextmanager
    def _get_db_connection(self):
        """Context manager for database connections"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"❌ Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def add_signal(self, signal_data: Dict) -> str:
        """Add a new signal to tracking system"""
        try:
            signal_id = f"signal_{int(time.time() * 1000)}"
            
            with self._get_db_connection() as conn:
                conn.execute("""
                    INSERT INTO tracked_signals 
                    (signal_id, signal_type, entry_price, current_price, take_profit, 
                     stop_loss, risk_amount, macro_indicators, confidence_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    signal_id,
                    signal_data.get('signal_type', 'long'),
                    signal_data.get('entry_price', 0),
                    signal_data.get('entry_price', 0),  # Initial current price
                    signal_data.get('take_profit', 0),
                    signal_data.get('stop_loss', 0),
                    signal_data.get('risk_amount', 100),
                    json.dumps(signal_data.get('macro_indicators', {})),
                    signal_data.get('confidence_score', 0.75)
                ))
                
                # Update statistics
                conn.execute("""
                    UPDATE signal_statistics 
                    SET total_signals = total_signals + 1,
                        active_signals = active_signals + 1,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = 1
                """)
                
                conn.commit()
                logger.info(f"✅ Signal {signal_id} added to tracking system")
                return signal_id
                
        except Exception as e:
            logger.error(f"❌ Error adding signal: {e}")
            return None
    
    def get_current_gold_price(self) -> float:
        """Fetch current gold price from API"""
        try:
            response = requests.get(self.gold_api_url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                price = float(data.get('price', 0))
                logger.debug(f"💰 Current gold price: ${price}")
                return price
            else:
                logger.warning(f"⚠️ Gold API returned status {response.status_code}")
                return 0
        except Exception as e:
            logger.error(f"❌ Error fetching gold price: {e}")
            return 0
    
    def calculate_pnl(self, signal_type: str, entry_price: float, current_price: float, risk_amount: float) -> float:
        """Calculate P&L for a signal"""
        try:
            if signal_type.lower() == 'long':
                price_diff = current_price - entry_price
            else:  # short
                price_diff = entry_price - current_price
            
            # Calculate P&L as percentage of risk amount
            pnl_percentage = (price_diff / entry_price) * 100
            pnl = (pnl_percentage / 100) * risk_amount
            
            return round(pnl, 2)
        except Exception as e:
            logger.error(f"❌ Error calculating P&L: {e}")
            return 0
    
    def update_signals(self) -> int:
        """Update all active signals with current market data"""
        try:
            current_price = self.get_current_gold_price()
            if current_price <= 0:
                logger.warning("⚠️ Unable to fetch current price, skipping update")
                return 0
            
            updated_count = 0
            signals_to_close = []
            
            with self._get_db_connection() as conn:
                # Get all active signals
                active_signals = conn.execute("""
                    SELECT * FROM tracked_signals WHERE status = 'active'
                """).fetchall()
                
                for signal in active_signals:
                    # Calculate current P&L
                    current_pnl = self.calculate_pnl(
                        signal['signal_type'],
                        signal['entry_price'],
                        current_price,
                        signal['risk_amount']
                    )
                    
                    # Update max profit and max drawdown
                    max_profit = max(signal['max_profit'], current_pnl)
                    max_drawdown = min(signal['max_drawdown'], current_pnl)
                    
                    # Check if signal should be closed
                    close_reason = None
                    if signal['signal_type'].lower() == 'long':
                        if current_price >= signal['take_profit']:
                            close_reason = "take_profit_hit"
                        elif current_price <= signal['stop_loss']:
                            close_reason = "stop_loss_hit"
                    else:  # short
                        if current_price <= signal['take_profit']:
                            close_reason = "take_profit_hit"
                        elif current_price >= signal['stop_loss']:
                            close_reason = "stop_loss_hit"
                    
                    # Update signal
                    conn.execute("""
                        UPDATE tracked_signals 
                        SET current_price = ?, current_pnl = ?, max_profit = ?, max_drawdown = ?
                        WHERE signal_id = ?
                    """, (current_price, current_pnl, max_profit, max_drawdown, signal['signal_id']))
                    
                    # Add price update record
                    conn.execute("""
                        INSERT INTO signal_updates (signal_id, price, pnl)
                        VALUES (?, ?, ?)
                    """, (signal['signal_id'], current_price, current_pnl))
                    
                    updated_count += 1
                    
                    # Mark for closing if needed
                    if close_reason:
                        signals_to_close.append((signal['signal_id'], close_reason, current_pnl))
                
                conn.commit()
            
            # Close signals that hit TP/SL
            for signal_id, reason, final_pnl in signals_to_close:
                self.close_signal(signal_id, reason, final_pnl)
                logger.info(f"🎯 Signal {signal_id} auto-closed: {reason} (P&L: ${final_pnl})")
            
            logger.debug(f"📊 Updated {updated_count} signals, closed {len(signals_to_close)} signals")
            return updated_count
            
        except Exception as e:
            logger.error(f"❌ Error updating signals: {e}")
            return 0
    
    def close_signal(self, signal_id: str, reason: str, final_pnl: float = None):
        """Close a signal and update statistics"""
        try:
            with self._get_db_connection() as conn:
                # Get signal details
                signal = conn.execute("""
                    SELECT * FROM tracked_signals WHERE signal_id = ? AND status = 'active'
                """, (signal_id,)).fetchone()
                
                if not signal:
                    logger.warning(f"⚠️ Signal {signal_id} not found or already closed")
                    return
                
                # Use provided P&L or calculate current
                if final_pnl is None:
                    final_pnl = signal['current_pnl']
                
                # Close the signal
                conn.execute("""
                    UPDATE tracked_signals 
                    SET status = 'closed', 
                        closed_at = CURRENT_TIMESTAMP,
                        close_reason = ?,
                        current_pnl = ?
                    WHERE signal_id = ?
                """, (reason, final_pnl, signal_id))
                
                # Update statistics
                is_winning = final_pnl > 0
                conn.execute("""
                    UPDATE signal_statistics 
                    SET active_signals = active_signals - 1,
                        winning_signals = winning_signals + ?,
                        losing_signals = losing_signals + ?,
                        total_pnl = total_pnl + ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = 1
                """, (1 if is_winning else 0, 0 if is_winning else 1, final_pnl))
                
                # Recalculate win rate
                stats = conn.execute("SELECT * FROM signal_statistics WHERE id = 1").fetchone()
                total_closed = stats['winning_signals'] + stats['losing_signals']
                if total_closed > 0:
                    win_rate = (stats['winning_signals'] / total_closed) * 100
                    avg_win = 0
                    avg_loss = 0
                    
                    if stats['winning_signals'] > 0:
                        winning_pnl = conn.execute("""
                            SELECT AVG(current_pnl) FROM tracked_signals 
                            WHERE status = 'closed' AND current_pnl > 0
                        """).fetchone()[0] or 0
                        avg_win = winning_pnl
                    
                    if stats['losing_signals'] > 0:
                        losing_pnl = conn.execute("""
                            SELECT AVG(current_pnl) FROM tracked_signals 
                            WHERE status = 'closed' AND current_pnl <= 0
                        """).fetchone()[0] or 0
                        avg_loss = abs(losing_pnl)
                    
                    conn.execute("""
                        UPDATE signal_statistics 
                        SET win_rate = ?, avg_win = ?, avg_loss = ?
                        WHERE id = 1
                    """, (win_rate, avg_win, avg_loss))
                
                conn.commit()
                logger.info(f"✅ Signal {signal_id} closed: {reason} (Final P&L: ${final_pnl})")
                
        except Exception as e:
            logger.error(f"❌ Error closing signal {signal_id}: {e}")
    
    def get_active_signals(self) -> List[Dict]:
        """Get all active signals with current P&L"""
        try:
            with self._get_db_connection() as conn:
                signals = conn.execute("""
                    SELECT * FROM tracked_signals 
                    WHERE status = 'active'
                    ORDER BY created_at DESC
                """).fetchall()
                
                result = []
                for signal in signals:
                    signal_dict = dict(signal)
                    # Parse macro indicators
                    try:
                        signal_dict['macro_indicators'] = json.loads(signal_dict['macro_indicators'] or '{}')
                    except:
                        signal_dict['macro_indicators'] = {}
                    
                    result.append(signal_dict)
                
                return result
                
        except Exception as e:
            logger.error(f"❌ Error getting active signals: {e}")
            return []
    
    def get_statistics(self) -> Dict:
        """Get comprehensive trading statistics"""
        try:
            with self._get_db_connection() as conn:
                stats = conn.execute("SELECT * FROM signal_statistics WHERE id = 1").fetchone()
                
                if stats:
                    return {
                        'total_signals': stats['total_signals'],
                        'active_signals': stats['active_signals'],
                        'winning_signals': stats['winning_signals'],
                        'losing_signals': stats['losing_signals'],
                        'total_pnl': round(stats['total_pnl'], 2),
                        'win_rate': round(stats['win_rate'], 1),
                        'avg_win': round(stats['avg_win'], 2),
                        'avg_loss': round(stats['avg_loss'], 2),
                        'profit_factor': round(stats['avg_win'] / max(stats['avg_loss'], 0.01), 2) if stats['avg_loss'] > 0 else 0,
                        'updated_at': stats['updated_at']
                    }
                else:
                    return {
                        'total_signals': 0,
                        'active_signals': 0,
                        'winning_signals': 0,
                        'losing_signals': 0,
                        'total_pnl': 0,
                        'win_rate': 0,
                        'avg_win': 0,
                        'avg_loss': 0,
                        'profit_factor': 0,
                        'updated_at': datetime.now().isoformat()
                    }
                    
        except Exception as e:
            logger.error(f"❌ Error getting statistics: {e}")
            return {}
    
    def get_signal_history(self, limit: int = 50) -> List[Dict]:
        """Get recent signal history"""
        try:
            with self._get_db_connection() as conn:
                signals = conn.execute("""
                    SELECT * FROM tracked_signals 
                    ORDER BY created_at DESC 
                    LIMIT ?
                """, (limit,)).fetchall()
                
                result = []
                for signal in signals:
                    signal_dict = dict(signal)
                    try:
                        signal_dict['macro_indicators'] = json.loads(signal_dict['macro_indicators'] or '{}')
                    except:
                        signal_dict['macro_indicators'] = {}
                    
                    result.append(signal_dict)
                
                return result
                
        except Exception as e:
            logger.error(f"❌ Error getting signal history: {e}")
            return []

# Initialize global tracker instance
signal_tracker = SignalTracker()

if __name__ == "__main__":
    # Test the system
    tracker = SignalTracker()
    
    # Add a test signal
    test_signal = {
        'signal_type': 'long',
        'entry_price': 3380.0,
        'take_profit': 3400.0,
        'stop_loss': 3370.0,
        'risk_amount': 100,
        'macro_indicators': {
            'rsi': 65,
            'macd': 'bullish',
            'sentiment': 'positive'
        },
        'confidence_score': 0.85
    }
    
    signal_id = tracker.add_signal(test_signal)
    print(f"Added test signal: {signal_id}")
    
    # Update signals
    updated = tracker.update_signals()
    print(f"Updated {updated} signals")
    
    # Get statistics
    stats = tracker.get_statistics()
    print(f"Statistics: {stats}")
    
    # Get active signals
    active = tracker.get_active_signals()
    print(f"Active signals: {len(active)}")
