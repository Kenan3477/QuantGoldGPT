"""
QuantGold Signal Memory System - Main Signal Brain
=================================================
Comprehensive signal storage and learning system that tracks every signal
from generation to outcome for continuous AI improvement.
"""

import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import uuid

@dataclass
class SignalData:
    """Complete signal data structure for memory storage"""
    signal_id: str
    timestamp: str
    price_at_generation: float
    entry_price: float
    stop_loss: float
    take_profit: float
    signal_type: str  # BULLISH, BEARISH, NEUTRAL
    confidence_score: float
    
    # Analysis components that determined the signal
    candlestick_patterns: List[Dict]
    macro_indicators: Dict[str, Any]
    news_articles: List[Dict]
    technical_indicators: Dict[str, Any]
    sentiment_data: Dict[str, Any]
    
    # Outcome tracking
    status: str  # ACTIVE, CLOSED_WIN, CLOSED_LOSS, EXPIRED
    close_timestamp: Optional[str] = None
    close_price: Optional[float] = None
    profit_loss: Optional[float] = None
    duration_minutes: Optional[int] = None
    close_reason: Optional[str] = None  # TP_HIT, SL_HIT, EXPIRED, MANUAL
    
    # Learning metrics
    accuracy_score: Optional[float] = None
    pattern_effectiveness: Optional[Dict] = None
    macro_correlation: Optional[Dict] = None

class SignalMemorySystem:
    """Main Signal Brain - Stores and analyzes all signal data for learning"""
    
    def __init__(self, db_path: str = "quantgold_signal_memory.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.init_database()
    
    def init_database(self):
        """Initialize the signal memory database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Main signals table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS signal_memory (
                        signal_id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        price_at_generation REAL NOT NULL,
                        entry_price REAL NOT NULL,
                        stop_loss REAL NOT NULL,
                        take_profit REAL NOT NULL,
                        signal_type TEXT NOT NULL,
                        confidence_score REAL NOT NULL,
                        
                        -- Analysis components (JSON stored)
                        candlestick_patterns TEXT NOT NULL,
                        macro_indicators TEXT NOT NULL,
                        news_articles TEXT NOT NULL,
                        technical_indicators TEXT NOT NULL,
                        sentiment_data TEXT NOT NULL,
                        
                        -- Outcome tracking
                        status TEXT DEFAULT 'ACTIVE',
                        close_timestamp TEXT,
                        close_price REAL,
                        profit_loss REAL,
                        duration_minutes INTEGER,
                        close_reason TEXT,
                        
                        -- Learning metrics
                        accuracy_score REAL,
                        pattern_effectiveness TEXT,
                        macro_correlation TEXT,
                        
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Pattern effectiveness tracking
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS pattern_performance (
                        pattern_name TEXT PRIMARY KEY,
                        total_signals INTEGER DEFAULT 0,
                        wins INTEGER DEFAULT 0,
                        losses INTEGER DEFAULT 0,
                        win_rate REAL DEFAULT 0.0,
                        avg_profit REAL DEFAULT 0.0,
                        avg_loss REAL DEFAULT 0.0,
                        confidence_correlation REAL DEFAULT 0.0,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Macro indicator effectiveness
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS macro_performance (
                        indicator_name TEXT PRIMARY KEY,
                        bullish_accuracy REAL DEFAULT 0.0,
                        bearish_accuracy REAL DEFAULT 0.0,
                        total_signals INTEGER DEFAULT 0,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # News sentiment tracking
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS news_performance (
                        news_sentiment TEXT PRIMARY KEY,
                        signal_accuracy REAL DEFAULT 0.0,
                        total_signals INTEGER DEFAULT 0,
                        avg_impact REAL DEFAULT 0.0,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.commit()
                self.logger.info("✅ Signal Memory System database initialized successfully")
                
        except Exception as e:
            self.logger.error(f"❌ Error initializing signal memory database: {e}")
    
    def store_signal(self, signal_data: SignalData) -> bool:
        """Store a new signal in the memory system"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO signal_memory (
                        signal_id, timestamp, price_at_generation, entry_price,
                        stop_loss, take_profit, signal_type, confidence_score,
                        candlestick_patterns, macro_indicators, news_articles,
                        technical_indicators, sentiment_data, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    signal_data.signal_id,
                    signal_data.timestamp,
                    signal_data.price_at_generation,
                    signal_data.entry_price,
                    signal_data.stop_loss,
                    signal_data.take_profit,
                    signal_data.signal_type,
                    signal_data.confidence_score,
                    json.dumps(signal_data.candlestick_patterns),
                    json.dumps(signal_data.macro_indicators),
                    json.dumps(signal_data.news_articles),
                    json.dumps(signal_data.technical_indicators),
                    json.dumps(signal_data.sentiment_data),
                    signal_data.status
                ))
                
                conn.commit()
                self.logger.info(f"🧠 Signal {signal_data.signal_id} stored in memory system")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Error storing signal in memory: {e}")
            return False
    
    def update_signal_outcome(self, signal_id: str, close_price: float, 
                            close_reason: str) -> bool:
        """Update signal with its outcome when closed"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get signal data
                cursor.execute('''
                    SELECT entry_price, stop_loss, take_profit, timestamp, signal_type
                    FROM signal_memory WHERE signal_id = ?
                ''', (signal_id,))
                
                result = cursor.fetchone()
                if not result:
                    self.logger.error(f"Signal {signal_id} not found in memory")
                    return False
                
                entry_price, stop_loss, take_profit, timestamp, signal_type = result
                
                # Calculate outcome
                if signal_type == "BULLISH":
                    profit_loss = close_price - entry_price
                    status = "CLOSED_WIN" if close_price >= take_profit else "CLOSED_LOSS"
                else:  # BEARISH
                    profit_loss = entry_price - close_price
                    status = "CLOSED_WIN" if close_price <= take_profit else "CLOSED_LOSS"
                
                # Calculate duration
                start_time = datetime.fromisoformat(timestamp)
                close_time = datetime.now()
                duration_minutes = int((close_time - start_time).total_seconds() / 60)
                
                # Calculate accuracy score
                if signal_type == "BULLISH":
                    accuracy_score = min(100, max(0, (close_price - entry_price) / (take_profit - entry_price) * 100))
                else:
                    accuracy_score = min(100, max(0, (entry_price - close_price) / (entry_price - take_profit) * 100))
                
                # Update signal
                cursor.execute('''
                    UPDATE signal_memory SET
                        status = ?, close_timestamp = ?, close_price = ?,
                        profit_loss = ?, duration_minutes = ?, close_reason = ?,
                        accuracy_score = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE signal_id = ?
                ''', (
                    status, close_time.isoformat(), close_price,
                    profit_loss, duration_minutes, close_reason,
                    accuracy_score, signal_id
                ))
                
                conn.commit()
                
                # Update learning metrics
                self._update_learning_metrics(signal_id)
                
                self.logger.info(f"🎯 Signal {signal_id} outcome updated: {status} (P&L: {profit_loss:.2f})")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Error updating signal outcome: {e}")
            return False
    
    def _update_learning_metrics(self, signal_id: str):
        """Update pattern and macro performance metrics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get signal data
                cursor.execute('''
                    SELECT candlestick_patterns, macro_indicators, status, profit_loss, confidence_score
                    FROM signal_memory WHERE signal_id = ?
                ''', (signal_id,))
                
                result = cursor.fetchone()
                if not result:
                    return
                
                patterns_json, macro_json, status, profit_loss, confidence = result
                patterns = json.loads(patterns_json)
                macro_data = json.loads(macro_json)
                
                is_win = status == "CLOSED_WIN"
                
                # Update pattern performance
                for pattern in patterns:
                    pattern_name = pattern.get('name', 'Unknown')
                    cursor.execute('''
                        INSERT OR REPLACE INTO pattern_performance (
                            pattern_name, total_signals, wins, losses, win_rate, avg_profit
                        ) VALUES (
                            ?, 
                            COALESCE((SELECT total_signals FROM pattern_performance WHERE pattern_name = ?), 0) + 1,
                            COALESCE((SELECT wins FROM pattern_performance WHERE pattern_name = ?), 0) + ?,
                            COALESCE((SELECT losses FROM pattern_performance WHERE pattern_name = ?), 0) + ?,
                            0.0, 0.0
                        )
                    ''', (pattern_name, pattern_name, pattern_name, 1 if is_win else 0, 
                         pattern_name, 0 if is_win else 1))
                
                # Update macro indicator performance
                for indicator, value in macro_data.items():
                    cursor.execute('''
                        INSERT OR REPLACE INTO macro_performance (
                            indicator_name, total_signals
                        ) VALUES (
                            ?, 
                            COALESCE((SELECT total_signals FROM macro_performance WHERE indicator_name = ?), 0) + 1
                        )
                    ''', (indicator, indicator))
                
                # Recalculate win rates
                cursor.execute('''
                    UPDATE pattern_performance 
                    SET win_rate = CAST(wins AS REAL) / total_signals * 100
                    WHERE total_signals > 0
                ''')
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"❌ Error updating learning metrics: {e}")
    
    def get_active_signals(self) -> List[Dict]:
        """Get all active signals for monitoring"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM signal_memory 
                    WHERE status = 'ACTIVE'
                    ORDER BY timestamp DESC
                ''')
                
                columns = [description[0] for description in cursor.description]
                signals = []
                
                for row in cursor.fetchall():
                    signal_dict = dict(zip(columns, row))
                    # Parse JSON fields
                    signal_dict['candlestick_patterns'] = json.loads(signal_dict['candlestick_patterns'])
                    signal_dict['macro_indicators'] = json.loads(signal_dict['macro_indicators'])
                    signal_dict['news_articles'] = json.loads(signal_dict['news_articles'])
                    signal_dict['technical_indicators'] = json.loads(signal_dict['technical_indicators'])
                    signal_dict['sentiment_data'] = json.loads(signal_dict['sentiment_data'])
                    signals.append(signal_dict)
                
                return signals
                
        except Exception as e:
            self.logger.error(f"❌ Error getting active signals: {e}")
            return []
    
    def clear_all_signals(self) -> bool:
        """Clear all signals from the database - USE WITH CAUTION"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM signal_memory")
                conn.commit()
                
                self.logger.info("🗑️ All signals cleared from database")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ Error clearing signals: {e}")
            return False

    def get_signals_count(self) -> int:
        """Get total number of signals in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM signal_memory")
                result = cursor.fetchone()
                return result[0] if result else 0
        except Exception as e:
            self.logger.error(f"❌ Error counting signals: {e}")
            return 0

    def get_learning_insights(self) -> Dict[str, Any]:
        """Get comprehensive learning insights from stored signals"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                insights = {}
                
                # Overall performance
                cursor.execute('''
                    SELECT 
                        COUNT(*) as total_signals,
                        SUM(CASE WHEN status = 'CLOSED_WIN' THEN 1 ELSE 0 END) as wins,
                        SUM(CASE WHEN status = 'CLOSED_LOSS' THEN 1 ELSE 0 END) as losses,
                        AVG(CASE WHEN status IN ('CLOSED_WIN', 'CLOSED_LOSS') THEN profit_loss END) as avg_pnl,
                        AVG(CASE WHEN status IN ('CLOSED_WIN', 'CLOSED_LOSS') THEN duration_minutes END) as avg_duration
                    FROM signal_memory
                ''')
                
                overall = cursor.fetchone()
                if overall and overall[0] > 0:
                    total, wins, losses, avg_pnl, avg_duration = overall
                    win_rate = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
                    
                    insights['overall'] = {
                        'total_signals': total,
                        'wins': wins,
                        'losses': losses,
                        'win_rate': win_rate,
                        'avg_pnl': avg_pnl or 0,
                        'avg_duration_minutes': avg_duration or 0
                    }
                
                # Pattern performance
                cursor.execute('''
                    SELECT pattern_name, total_signals, wins, losses, win_rate
                    FROM pattern_performance
                    ORDER BY win_rate DESC
                    LIMIT 10
                ''')
                
                insights['best_patterns'] = [
                    {
                        'pattern': row[0],
                        'total_signals': row[1],
                        'wins': row[2],
                        'losses': row[3],
                        'win_rate': row[4]
                    } for row in cursor.fetchall()
                ]
                
                # Recent performance trend
                cursor.execute('''
                    SELECT 
                        DATE(timestamp) as date,
                        COUNT(*) as signals,
                        SUM(CASE WHEN status = 'CLOSED_WIN' THEN 1 ELSE 0 END) as wins,
                        AVG(profit_loss) as avg_pnl
                    FROM signal_memory
                    WHERE timestamp >= date('now', '-30 days')
                    AND status IN ('CLOSED_WIN', 'CLOSED_LOSS')
                    GROUP BY DATE(timestamp)
                    ORDER BY date DESC
                    LIMIT 30
                ''')
                
                insights['recent_performance'] = [
                    {
                        'date': row[0],
                        'signals': row[1],
                        'wins': row[2],
                        'win_rate': (row[2] / row[1] * 100) if row[1] > 0 else 0,
                        'avg_pnl': row[3] or 0
                    } for row in cursor.fetchall()
                ]
                
                return insights
                
        except Exception as e:
            self.logger.error(f"❌ Error getting learning insights: {e}")
            return {}
    
    def get_pattern_effectiveness(self) -> Dict[str, float]:
        """Get effectiveness scores for each pattern"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT pattern_name, win_rate 
                    FROM pattern_performance
                    WHERE total_signals >= 5
                    ORDER BY win_rate DESC
                ''')
                
                return {row[0]: row[1] for row in cursor.fetchall()}
                
        except Exception as e:
            self.logger.error(f"❌ Error getting pattern effectiveness: {e}")
            return {}
    
    def optimize_strategy_weights(self) -> Dict[str, float]:
        """Calculate optimal strategy weights based on historical performance"""
        try:
            insights = self.get_learning_insights()
            pattern_effectiveness = self.get_pattern_effectiveness()
            
            if not insights.get('overall'):
                return {'technical': 0.25, 'sentiment': 0.25, 'macro': 0.25, 'pattern': 0.25}
            
            # Calculate weights based on performance
            overall_win_rate = insights['overall']['win_rate']
            
            # Pattern weight based on best patterns effectiveness
            best_patterns = insights.get('best_patterns', [])
            pattern_weight = 0.25
            if best_patterns:
                avg_pattern_win_rate = sum(p['win_rate'] for p in best_patterns[:5]) / len(best_patterns[:5])
                pattern_weight = min(0.5, max(0.1, avg_pattern_win_rate / 100 * 0.5))
            
            # Technical weight (inverse correlation with pattern weight)
            technical_weight = min(0.5, max(0.1, 0.5 - pattern_weight))
            
            # Distribute remaining weight between sentiment and macro
            remaining = 1.0 - pattern_weight - technical_weight
            sentiment_weight = remaining * 0.6  # Favor sentiment slightly
            macro_weight = remaining * 0.4
            
            optimized_weights = {
                'technical': round(technical_weight, 3),
                'sentiment': round(sentiment_weight, 3),
                'macro': round(macro_weight, 3),
                'pattern': round(pattern_weight, 3)
            }
            
            self.logger.info(f"🎯 Optimized strategy weights: {optimized_weights}")
            return optimized_weights
            
        except Exception as e:
            self.logger.error(f"❌ Error optimizing strategy weights: {e}")
            return {'technical': 0.25, 'sentiment': 0.25, 'macro': 0.25, 'pattern': 0.25}

def create_signal_data(signal_type: str, confidence: float, price: float,
                      entry: float, sl: float, tp: float,
                      patterns: List[Dict], macro: Dict, news: List[Dict],
                      technical: Dict, sentiment: Dict) -> SignalData:
    """Helper function to create SignalData object"""
    
    signal_id = f"QG_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
    
    return SignalData(
        signal_id=signal_id,
        timestamp=datetime.now().isoformat(),
        price_at_generation=price,
        entry_price=entry,
        stop_loss=sl,
        take_profit=tp,
        signal_type=signal_type,
        confidence_score=confidence,
        candlestick_patterns=patterns,
        macro_indicators=macro,
        news_articles=news,
        technical_indicators=technical,
        sentiment_data=sentiment,
        status="ACTIVE"
    )

# Example usage and testing
if __name__ == "__main__":
    # Initialize system
    memory_system = SignalMemorySystem()
    
    # Example signal data
    example_patterns = [
        {"name": "Doji", "confidence": 75, "timeframe": "1H"},
        {"name": "Hammer", "confidence": 60, "timeframe": "15M"}
    ]
    
    example_macro = {
        "DXY": -0.3,
        "10Y_YIELD": 4.2,
        "INFLATION": 2.4,
        "FED_SENTIMENT": "DOVISH"
    }
    
    example_news = [
        {"headline": "Fed signals rate cuts", "sentiment": "BULLISH", "impact": 8.5},
        {"headline": "Gold demand rising", "sentiment": "BULLISH", "impact": 7.2}
    ]
    
    example_technical = {
        "RSI": 65,
        "MACD": "BULLISH_CROSSOVER",
        "SUPPORT": 3642,
        "RESISTANCE": 3665
    }
    
    example_sentiment = {
        "fear_greed": 52,
        "market_mood": "RISK_ON",
        "buyer_strength": 65
    }
    
    # Create and store signal
    signal = create_signal_data(
        signal_type="BULLISH",
        confidence=76.5,
        price=3648.90,
        entry=3650.00,
        sl=3635.00,
        tp=3680.00,
        patterns=example_patterns,
        macro=example_macro,
        news=example_news,
        technical=example_technical,
        sentiment=example_sentiment
    )
    
    memory_system.store_signal(signal)
    print(f"✅ Example signal {signal.signal_id} stored successfully!")
