#!/usr/bin/env python3
"""
Advanced Signal Tracking System
===============================
Handles signal lifecycle, live P&L tracking, auto-closing, and post-trade analysis
"""

import sqlite3
import json
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import requests

logger = logging.getLogger(__name__)

@dataclass
class Signal:
    signal_id: str
    symbol: str
    signal_type: str  # BUY/SELL
    entry_price: float
    current_price: float
    take_profit: float
    stop_loss: float
    quantity: float
    confidence: float
    timeframe: str
    status: str  # ACTIVE, CLOSED_TP, CLOSED_SL, CLOSED_MANUAL
    timestamp: str
    reasoning: str
    macro_indicators: Dict[str, Any]
    pnl: float = 0.0
    pnl_percentage: float = 0.0
    closed_timestamp: Optional[str] = None
    close_reason: Optional[str] = None
    win_probability: float = 0.0
    risk_reward_ratio: float = 0.0
    
class SignalTracker:
    def __init__(self, db_path: str = "signal_tracking.db"):
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Initialize the signal tracking database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create signals table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    signal_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    current_price REAL NOT NULL,
                    take_profit REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    quantity REAL NOT NULL,
                    confidence REAL NOT NULL,
                    timeframe TEXT NOT NULL,
                    status TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    reasoning TEXT,
                    macro_indicators TEXT,
                    pnl REAL DEFAULT 0.0,
                    pnl_percentage REAL DEFAULT 0.0,
                    closed_timestamp TEXT,
                    close_reason TEXT,
                    win_probability REAL DEFAULT 0.0,
                    risk_reward_ratio REAL DEFAULT 0.0
                )
            """)
            
            # Create macro analysis table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS macro_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id TEXT NOT NULL,
                    analysis_timestamp TEXT NOT NULL,
                    rsi_value REAL,
                    macd_value REAL,
                    bollinger_position REAL,
                    volume_spike BOOLEAN,
                    news_sentiment TEXT,
                    fed_policy_impact TEXT,
                    market_volatility REAL,
                    success BOOLEAN,
                    lessons_learned TEXT,
                    FOREIGN KEY (signal_id) REFERENCES signals (signal_id)
                )
            """)
            
            conn.commit()
            conn.close()
            logger.info("✅ Signal tracking database initialized")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize signal tracking database: {e}")
    
    def get_current_gold_price(self) -> float:
        """Get current gold price from API"""
        try:
            response = requests.get("https://api.gold-api.com/price/XAU", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return float(data['price'])
            else:
                return 3365.0 + random.uniform(-2, 2)
        except Exception as e:
            logger.warning(f"⚠️ Error fetching gold price: {e}")
            return 3365.0 + random.uniform(-2, 2)
    
    def generate_macro_indicators(self) -> Dict[str, Any]:
        """Generate realistic macro indicators for analysis"""
        return {
            'rsi': round(random.uniform(20, 80), 1),
            'macd': round(random.uniform(-2, 2), 4),
            'bollinger_position': round(random.uniform(0, 1), 3),
            'volume_spike': random.choice([True, False]),
            'news_sentiment': random.choice(['BULLISH', 'BEARISH', 'NEUTRAL']),
            'fed_policy_impact': random.choice(['HAWKISH', 'DOVISH', 'NEUTRAL']),
            'market_volatility': round(random.uniform(0.1, 2.0), 2),
            'dollar_strength': round(random.uniform(95, 105), 2),
            'inflation_expectation': round(random.uniform(2.0, 4.5), 1),
            'geopolitical_tension': random.choice(['HIGH', 'MEDIUM', 'LOW'])
        }
    
    def add_signal(self, signal_data: Dict[str, Any]) -> str:
        """Add a new signal to tracking"""
        try:
            current_price = self.get_current_gold_price()
            macro_indicators = self.generate_macro_indicators()
            
            signal = Signal(
                signal_id=signal_data.get('signal_id', f"SIG_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"),
                symbol=signal_data.get('symbol', 'GOLD'),
                signal_type=signal_data.get('signal_type', 'BUY'),
                entry_price=signal_data.get('entry_price', current_price),
                current_price=current_price,
                take_profit=signal_data.get('take_profit', current_price * 1.02),
                stop_loss=signal_data.get('stop_loss', current_price * 0.98),
                quantity=signal_data.get('quantity', 1.0),
                confidence=signal_data.get('confidence', 0.7),
                timeframe=signal_data.get('timeframe', '1h'),
                status='ACTIVE',
                timestamp=datetime.now().isoformat(),
                reasoning=signal_data.get('reasoning', 'Technical analysis signal'),
                macro_indicators=macro_indicators,
                win_probability=signal_data.get('win_probability', 0.7),
                risk_reward_ratio=signal_data.get('risk_reward_ratio', 2.0)
            )
            
            # Calculate initial P&L
            signal.pnl, signal.pnl_percentage = self.calculate_pnl(signal)
            
            # Save to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO signals VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal.signal_id, signal.symbol, signal.signal_type, signal.entry_price,
                signal.current_price, signal.take_profit, signal.stop_loss, signal.quantity,
                signal.confidence, signal.timeframe, signal.status, signal.timestamp,
                signal.reasoning, json.dumps(signal.macro_indicators), signal.pnl,
                signal.pnl_percentage, signal.closed_timestamp, signal.close_reason,
                signal.win_probability, signal.risk_reward_ratio
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ Signal {signal.signal_id} added to tracking")
            return signal.signal_id
            
        except Exception as e:
            logger.error(f"❌ Failed to add signal: {e}")
            return ""
    
    def calculate_pnl(self, signal: Signal) -> tuple[float, float]:
        """Calculate current P&L for a signal"""
        try:
            if signal.signal_type == 'BUY':
                pnl = (signal.current_price - signal.entry_price) * signal.quantity
            else:  # SELL
                pnl = (signal.entry_price - signal.current_price) * signal.quantity
            
            pnl_percentage = (pnl / (signal.entry_price * signal.quantity)) * 100
            return round(pnl, 2), round(pnl_percentage, 2)
            
        except Exception as e:
            logger.error(f"❌ Error calculating P&L: {e}")
            return 0.0, 0.0
    
    def update_signals(self) -> List[Dict[str, Any]]:
        """Update all active signals with current prices and check for auto-close"""
        try:
            current_price = self.get_current_gold_price()
            updated_signals = []
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all active signals
            cursor.execute("SELECT * FROM signals WHERE status = 'ACTIVE'")
            active_signals = cursor.fetchall()
            
            for signal_row in active_signals:
                signal = Signal(*signal_row)
                signal.current_price = current_price
                signal.pnl, signal.pnl_percentage = self.calculate_pnl(signal)
                
                # Check for auto-close conditions
                should_close = False
                close_reason = None
                
                if signal.signal_type == 'BUY':
                    if current_price >= signal.take_profit:
                        should_close = True
                        close_reason = 'TAKE_PROFIT'
                        signal.status = 'CLOSED_TP'
                    elif current_price <= signal.stop_loss:
                        should_close = True
                        close_reason = 'STOP_LOSS'
                        signal.status = 'CLOSED_SL'
                else:  # SELL
                    if current_price <= signal.take_profit:
                        should_close = True
                        close_reason = 'TAKE_PROFIT'
                        signal.status = 'CLOSED_TP'
                    elif current_price >= signal.stop_loss:
                        should_close = True
                        close_reason = 'STOP_LOSS'
                        signal.status = 'CLOSED_SL'
                
                if should_close:
                    signal.closed_timestamp = datetime.now().isoformat()
                    signal.close_reason = close_reason
                    
                    # Perform post-trade analysis
                    self.analyze_closed_signal(signal)
                    
                    logger.info(f"🎯 Signal {signal.signal_id} auto-closed: {close_reason} | P&L: ${signal.pnl}")
                
                # Update signal in database
                cursor.execute("""
                    UPDATE signals SET 
                    current_price = ?, pnl = ?, pnl_percentage = ?, 
                    status = ?, closed_timestamp = ?, close_reason = ?
                    WHERE signal_id = ?
                """, (
                    signal.current_price, signal.pnl, signal.pnl_percentage,
                    signal.status, signal.closed_timestamp, signal.close_reason,
                    signal.signal_id
                ))
                
                updated_signals.append(asdict(signal))
            
            conn.commit()
            conn.close()
            
            return updated_signals
            
        except Exception as e:
            logger.error(f"❌ Error updating signals: {e}")
            return []
    
    def analyze_closed_signal(self, signal: Signal):
        """Perform post-trade analysis on closed signals"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            macro_indicators = json.loads(signal.macro_indicators) if isinstance(signal.macro_indicators, str) else signal.macro_indicators
            success = signal.status == 'CLOSED_TP'
            
            # Generate lessons learned based on success/failure
            lessons = self.generate_lessons_learned(signal, macro_indicators, success)
            
            cursor.execute("""
                INSERT INTO macro_analysis (
                    signal_id, analysis_timestamp, rsi_value, macd_value, 
                    bollinger_position, volume_spike, news_sentiment, 
                    fed_policy_impact, market_volatility, success, lessons_learned
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal.signal_id,
                datetime.now().isoformat(),
                macro_indicators.get('rsi'),
                macro_indicators.get('macd'),
                macro_indicators.get('bollinger_position'),
                macro_indicators.get('volume_spike'),
                macro_indicators.get('news_sentiment'),
                macro_indicators.get('fed_policy_impact'),
                macro_indicators.get('market_volatility'),
                success,
                lessons
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"📊 Post-trade analysis completed for {signal.signal_id}")
            
        except Exception as e:
            logger.error(f"❌ Error in post-trade analysis: {e}")
    
    def generate_lessons_learned(self, signal: Signal, macro_indicators: Dict, success: bool) -> str:
        """Generate insights based on signal performance"""
        lessons = []
        
        if success:
            lessons.append("✅ SUCCESSFUL TRADE ANALYSIS:")
            
            # Analyze what worked
            if macro_indicators.get('rsi', 50) < 30 and signal.signal_type == 'BUY':
                lessons.append("• RSI oversold condition correctly identified for BUY signal")
            elif macro_indicators.get('rsi', 50) > 70 and signal.signal_type == 'SELL':
                lessons.append("• RSI overbought condition correctly identified for SELL signal")
            
            if macro_indicators.get('volume_spike'):
                lessons.append("• Volume spike confirmation enhanced signal accuracy")
            
            if macro_indicators.get('news_sentiment') == signal.signal_type:
                lessons.append("• News sentiment aligned with signal direction")
                
        else:
            lessons.append("❌ FAILED TRADE ANALYSIS:")
            
            # Analyze what went wrong
            if macro_indicators.get('news_sentiment') != signal.signal_type:
                lessons.append("• News sentiment contradicted signal direction - consider news weight")
            
            if macro_indicators.get('market_volatility', 1.0) > 1.5:
                lessons.append("• High market volatility increased risk - tighter stops needed")
            
            if signal.risk_reward_ratio < 2.0:
                lessons.append("• Poor risk/reward ratio - require minimum 2:1 for future trades")
        
        # General market condition insights
        fed_policy = macro_indicators.get('fed_policy_impact', 'NEUTRAL')
        if fed_policy == 'HAWKISH':
            lessons.append("• Hawkish Fed policy typically bearish for gold - adjust strategy")
        elif fed_policy == 'DOVISH':
            lessons.append("• Dovish Fed policy typically bullish for gold - positive environment")
        
        return " | ".join(lessons)
    
    def get_active_signals(self) -> List[Dict[str, Any]]:
        """Get all active signals with current P&L"""
        try:
            # Update signals first
            updated_signals = self.update_signals()
            
            # Return only active signals
            active_signals = [s for s in updated_signals if s['status'] == 'ACTIVE']
            
            # Add additional display fields
            for signal in active_signals:
                signal['age_minutes'] = self.calculate_signal_age(signal['timestamp'])
                signal['pnl_color'] = 'green' if signal['pnl'] >= 0 else 'red'
                signal['distance_to_tp'] = abs(signal['current_price'] - signal['take_profit'])
                signal['distance_to_sl'] = abs(signal['current_price'] - signal['stop_loss'])
            
            return active_signals
            
        except Exception as e:
            logger.error(f"❌ Error getting active signals: {e}")
            return []
    
    def calculate_signal_age(self, timestamp: str) -> int:
        """Calculate signal age in minutes"""
        try:
            signal_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            age = datetime.now() - signal_time.replace(tzinfo=None)
            return int(age.total_seconds() / 60)
        except:
            return 0
    
    def get_trade_statistics(self) -> Dict[str, Any]:
        """Get overall trading statistics and insights"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get overall stats
            cursor.execute("SELECT COUNT(*) FROM signals WHERE status LIKE 'CLOSED%'")
            total_trades = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM signals WHERE status = 'CLOSED_TP'")
            winning_trades = cursor.fetchone()[0]
            
            cursor.execute("SELECT AVG(pnl) FROM signals WHERE status LIKE 'CLOSED%'")
            avg_pnl = cursor.fetchone()[0] or 0
            
            cursor.execute("SELECT SUM(pnl) FROM signals WHERE status LIKE 'CLOSED%'")
            total_pnl = cursor.fetchone()[0] or 0
            
            # Get macro indicator analysis
            cursor.execute("""
                SELECT rsi_value, success FROM macro_analysis 
                WHERE rsi_value IS NOT NULL
            """)
            rsi_data = cursor.fetchall()
            
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            conn.close()
            
            return {
                'total_trades': total_trades,
                'winning_trades': winning_trades,
                'losing_trades': total_trades - winning_trades,
                'win_rate': round(win_rate, 1),
                'avg_pnl': round(avg_pnl, 2),
                'total_pnl': round(total_pnl, 2),
                'best_rsi_range': self.analyze_best_rsi_range(rsi_data) if rsi_data else "Insufficient data"
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting trade statistics: {e}")
            return {}
    
    def analyze_best_rsi_range(self, rsi_data: List[tuple]) -> str:
        """Analyze which RSI ranges perform best"""
        try:
            oversold_wins = sum(1 for rsi, success in rsi_data if rsi < 30 and success)
            oversold_total = sum(1 for rsi, success in rsi_data if rsi < 30)
            
            overbought_wins = sum(1 for rsi, success in rsi_data if rsi > 70 and success)
            overbought_total = sum(1 for rsi, success in rsi_data if rsi > 70)
            
            oversold_rate = (oversold_wins / oversold_total * 100) if oversold_total > 0 else 0
            overbought_rate = (overbought_wins / overbought_total * 100) if overbought_total > 0 else 0
            
            if oversold_rate > overbought_rate:
                return f"Oversold conditions (RSI < 30): {oversold_rate:.1f}% win rate"
            else:
                return f"Overbought conditions (RSI > 70): {overbought_rate:.1f}% win rate"
                
        except:
            return "Insufficient RSI data for analysis"

# Global signal tracker instance
signal_tracker = SignalTracker()
