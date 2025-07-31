"""
Dynamic ML Prediction Engine for GoldGPT
Updates predictions when market conditions shift and contradict current predictions
"""

import sqlite3
import json
import datetime
import time
import asyncio
from typing import Dict, List, Tuple, Optional, Any
import logging
import numpy as np
from dataclasses import dataclass, asdict
import threading

# Import existing modules
from self_improving_ml_engine import SelfImprovingMLEngine, MultiTimeframePrediction
from real_time_data_engine import RealTimeDataEngine
from ai_analysis_api import AdvancedAIAnalyzer

logger = logging.getLogger(__name__)

@dataclass
class MarketShift:
    timestamp: datetime.datetime
    shift_type: str  # trend, sentiment, news, candlestick, greed_fear, economic
    old_value: Any
    new_value: Any
    severity: float  # 0-1 scale
    confidence: float  # 0-1 scale
    source: str
    description: str

@dataclass
class PredictionUpdate:
    timestamp: datetime.datetime
    original_prediction: Dict
    updated_prediction: Dict
    reason: str
    market_shifts: List[MarketShift]
    confidence_change: float

class DynamicPredictionEngine:
    """Engine that monitors market conditions and updates predictions dynamically"""
    
    def __init__(self):
        self.ml_engine = SelfImprovingMLEngine()
        self.data_engine = RealTimeDataEngine()
        self.ai_analyzer = AdvancedAIAnalyzer()
        
        self.db_path = "goldgpt_dynamic_predictions.db"
        self.init_database()
        
        # Current predictions cache
        self.current_predictions = {}
        
        # Market condition baselines for shift detection
        self.baseline_conditions = {}
        
        # Monitoring intervals
        self.check_interval = 300  # 5 minutes
        self.major_check_interval = 1800  # 30 minutes
        
        # Shift thresholds
        self.shift_thresholds = {
            'trend': 0.5,  # RSI, MACD changes > 50%
            'sentiment': 0.3,  # Sentiment score change > 30%
            'news': 0.4,  # News impact score > 40%
            'candlestick': 0.6,  # Pattern confidence > 60%
            'greed_fear': 0.3,  # Fear/Greed index change > 30%
            'economic': 0.5,  # Economic indicator change > 50%
        }
        
        # Start monitoring thread
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_market_conditions, daemon=True)
        self.monitor_thread.start()
        
        logger.info("🔄 Dynamic Prediction Engine initialized - monitoring market shifts")

    def init_database(self):
        """Initialize database for dynamic predictions tracking"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Market shifts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_shifts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                symbol TEXT NOT NULL,
                shift_type TEXT NOT NULL,
                old_value TEXT,
                new_value TEXT,
                severity REAL NOT NULL,
                confidence REAL NOT NULL,
                source TEXT NOT NULL,
                description TEXT
            )
        ''')
        
        # Prediction updates table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prediction_updates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                symbol TEXT NOT NULL,
                original_prediction TEXT NOT NULL,
                updated_prediction TEXT NOT NULL,
                reason TEXT NOT NULL,
                confidence_change REAL,
                market_shifts_count INTEGER
            )
        ''')
        
        # Baseline conditions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS baseline_conditions (
                symbol TEXT NOT NULL,
                condition_type TEXT NOT NULL,
                value TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (symbol, condition_type)
            ) WITHOUT ROWID
        ''')
        
        conn.commit()
        conn.close()

    def get_current_prediction(self, symbol: str) -> Optional[Dict]:
        """Get current active prediction for symbol"""
        return self.current_predictions.get(symbol)

    def set_current_prediction(self, symbol: str, prediction: MultiTimeframePrediction):
        """Set current prediction and establish baseline conditions"""
        self.current_predictions[symbol] = {
            'prediction': prediction,
            'created_at': datetime.datetime.now(),
            'last_updated': datetime.datetime.now(),
            'update_count': 0
        }
        
        # Establish baseline conditions
        asyncio.create_task(self._establish_baseline_conditions(symbol))
        
        logger.info(f"📊 Set current prediction for {symbol} - monitoring for shifts")

    async def _establish_baseline_conditions(self, symbol: str):
        """Establish baseline market conditions for shift detection"""
        try:
            # Get current market data
            price_data = self.data_engine.get_live_price_data(symbol)
            sentiment_data = await self.data_engine.analyze_market_sentiment(symbol, '1d')
            technical_data = await self.data_engine.get_technical_indicators(symbol, '1d')
            news_data = await self.data_engine.get_recent_news(symbol, hours=24)
            
            # Store baseline conditions
            baselines = {
                'trend': {
                    'rsi': technical_data.get('rsi', 50),
                    'macd': technical_data.get('macd', 0),
                    'price_trend': price_data.get('change_percent', 0)
                },
                'sentiment': {
                    'score': sentiment_data.get('score', 0.5),
                    'confidence': sentiment_data.get('confidence', 0.5)
                },
                'news': {
                    'impact_score': news_data.get('overall_impact', 0.5),
                    'news_count': len(news_data.get('articles', []))
                },
                'economic': {
                    'volatility': technical_data.get('volatility', 0.5),
                    'volume': price_data.get('volume', 0)
                }
            }
            
            self.baseline_conditions[symbol] = baselines
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for condition_type, values in baselines.items():
                cursor.execute('''
                    INSERT OR REPLACE INTO baseline_conditions 
                    (symbol, condition_type, value, timestamp) 
                    VALUES (?, ?, ?, ?)
                ''', (symbol, condition_type, json.dumps(values), datetime.datetime.now()))
            
            conn.commit()
            conn.close()
            
            logger.info(f"📈 Established baseline conditions for {symbol}")
            
        except Exception as e:
            logger.error(f"Failed to establish baseline conditions for {symbol}: {e}")

    def _monitor_market_conditions(self):
        """Background thread to monitor market conditions"""
        while self.monitoring:
            try:
                # Check all symbols with active predictions
                for symbol in list(self.current_predictions.keys()):
                    asyncio.create_task(self._check_for_market_shifts(symbol))
                
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in market monitoring: {e}")
                time.sleep(60)  # Wait 1 minute before retrying

    async def _check_for_market_shifts(self, symbol: str):
        """Check for significant market shifts that might require prediction updates"""
        try:
            if symbol not in self.baseline_conditions:
                return
            
            current_prediction = self.current_predictions.get(symbol)
            if not current_prediction:
                return
            
            # Get current market data
            price_data = self.data_engine.get_live_price_data(symbol)
            sentiment_data = await self.data_engine.analyze_market_sentiment(symbol, '1d')
            technical_data = await self.data_engine.get_technical_indicators(symbol, '1d')
            news_data = await self.data_engine.get_recent_news(symbol, hours=6)
            
            baselines = self.baseline_conditions[symbol]
            detected_shifts = []
            
            # 1. Check trend shifts
            trend_shifts = await self._detect_trend_shifts(symbol, technical_data, price_data, baselines['trend'])
            detected_shifts.extend(trend_shifts)
            
            # 2. Check sentiment shifts
            sentiment_shifts = await self._detect_sentiment_shifts(symbol, sentiment_data, baselines['sentiment'])
            detected_shifts.extend(sentiment_shifts)
            
            # 3. Check news impact
            news_shifts = await self._detect_news_shifts(symbol, news_data, baselines['news'])
            detected_shifts.extend(news_shifts)
            
            # 4. Check candlestick pattern changes
            pattern_shifts = await self._detect_pattern_shifts(symbol, technical_data)
            detected_shifts.extend(pattern_shifts)
            
            # 5. Check greed/fear index
            greed_fear_shifts = await self._detect_greed_fear_shifts(symbol)
            detected_shifts.extend(greed_fear_shifts)
            
            # 6. Check economic factors
            economic_shifts = await self._detect_economic_shifts(symbol, baselines['economic'])
            detected_shifts.extend(economic_shifts)
            
            # If significant shifts detected, update prediction
            if detected_shifts:
                major_shifts = [s for s in detected_shifts if s.severity >= 0.6]
                if major_shifts or len(detected_shifts) >= 3:
                    await self._update_prediction_due_to_shifts(symbol, detected_shifts)
            
        except Exception as e:
            logger.error(f"Error checking market shifts for {symbol}: {e}")

    async def _detect_trend_shifts(self, symbol: str, technical_data: Dict, price_data: Dict, baseline: Dict) -> List[MarketShift]:
        """Detect significant trend reversals or accelerations"""
        shifts = []
        
        try:
            current_rsi = technical_data.get('rsi', 50)
            current_macd = technical_data.get('macd', 0)
            current_price_trend = price_data.get('change_percent', 0)
            
            baseline_rsi = baseline.get('rsi', 50)
            baseline_macd = baseline.get('macd', 0)
            baseline_trend = baseline.get('price_trend', 0)
            
            # RSI shift detection
            rsi_change = abs(current_rsi - baseline_rsi) / 100
            if rsi_change > self.shift_thresholds['trend']:
                # Major overbought/oversold shift
                if (baseline_rsi < 50 and current_rsi > 70) or (baseline_rsi > 50 and current_rsi < 30):
                    shifts.append(MarketShift(
                        timestamp=datetime.datetime.now(),
                        shift_type='trend',
                        old_value=baseline_rsi,
                        new_value=current_rsi,
                        severity=min(rsi_change * 2, 1.0),
                        confidence=0.8,
                        source='technical_analysis',
                        description=f"Major RSI shift: {baseline_rsi:.1f} → {current_rsi:.1f}"
                    ))
            
            # MACD crossover detection
            if (baseline_macd <= 0 < current_macd) or (baseline_macd >= 0 > current_macd):
                shifts.append(MarketShift(
                    timestamp=datetime.datetime.now(),
                    shift_type='trend',
                    old_value=baseline_macd,
                    new_value=current_macd,
                    severity=0.7,
                    confidence=0.75,
                    source='technical_analysis',
                    description=f"MACD crossover: {baseline_macd:.4f} → {current_macd:.4f}"
                ))
            
            # Price trend acceleration/reversal
            trend_change = abs(current_price_trend - baseline_trend)
            if trend_change > 2.0:  # 2% change in trend direction
                shifts.append(MarketShift(
                    timestamp=datetime.datetime.now(),
                    shift_type='trend',
                    old_value=baseline_trend,
                    new_value=current_price_trend,
                    severity=min(trend_change / 5.0, 1.0),
                    confidence=0.7,
                    source='price_action',
                    description=f"Price trend shift: {baseline_trend:.2f}% → {current_price_trend:.2f}%"
                ))
                
        except Exception as e:
            logger.error(f"Error detecting trend shifts: {e}")
        
        return shifts

    async def _detect_sentiment_shifts(self, symbol: str, sentiment_data: Dict, baseline: Dict) -> List[MarketShift]:
        """Detect significant sentiment changes"""
        shifts = []
        
        try:
            current_score = sentiment_data.get('score', 0.5)
            current_confidence = sentiment_data.get('confidence', 0.5)
            
            baseline_score = baseline.get('score', 0.5)
            baseline_confidence = baseline.get('confidence', 0.5)
            
            score_change = abs(current_score - baseline_score)
            
            if score_change > self.shift_thresholds['sentiment']:
                shifts.append(MarketShift(
                    timestamp=datetime.datetime.now(),
                    shift_type='sentiment',
                    old_value=baseline_score,
                    new_value=current_score,
                    severity=min(score_change * 2, 1.0),
                    confidence=max(current_confidence, baseline_confidence),
                    source='sentiment_analysis',
                    description=f"Sentiment shift: {baseline_score:.2f} → {current_score:.2f}"
                ))
                
        except Exception as e:
            logger.error(f"Error detecting sentiment shifts: {e}")
        
        return shifts

    async def _detect_news_shifts(self, symbol: str, news_data: Dict, baseline: Dict) -> List[MarketShift]:
        """Detect high-impact news that contradicts current prediction"""
        shifts = []
        
        try:
            current_impact = news_data.get('overall_impact', 0.5)
            current_count = len(news_data.get('articles', []))
            
            baseline_impact = baseline.get('impact_score', 0.5)
            baseline_count = baseline.get('news_count', 0)
            
            # Check for high-impact news surge
            if current_impact > self.shift_thresholds['news'] and current_count > baseline_count + 3:
                shifts.append(MarketShift(
                    timestamp=datetime.datetime.now(),
                    shift_type='news',
                    old_value=baseline_impact,
                    new_value=current_impact,
                    severity=current_impact,
                    confidence=0.8,
                    source='news_analysis',
                    description=f"High-impact news surge: {current_count} articles with {current_impact:.2f} impact"
                ))
                
        except Exception as e:
            logger.error(f"Error detecting news shifts: {e}")
        
        return shifts

    async def _detect_pattern_shifts(self, symbol: str, technical_data: Dict) -> List[MarketShift]:
        """Detect new candlestick patterns that suggest direction change"""
        shifts = []
        
        try:
            # Look for reversal patterns
            patterns = technical_data.get('candlestick_patterns', [])
            
            reversal_patterns = ['doji', 'hammer', 'shooting_star', 'engulfing', 'morning_star', 'evening_star']
            
            for pattern in patterns:
                if pattern.get('name', '').lower() in reversal_patterns:
                    confidence = pattern.get('confidence', 0)
                    if confidence > self.shift_thresholds['candlestick']:
                        shifts.append(MarketShift(
                            timestamp=datetime.datetime.now(),
                            shift_type='candlestick',
                            old_value='none',
                            new_value=pattern.get('name'),
                            severity=confidence,
                            confidence=confidence,
                            source='pattern_analysis',
                            description=f"Reversal pattern detected: {pattern.get('name')} ({confidence:.2f})"
                        ))
                        
        except Exception as e:
            logger.error(f"Error detecting pattern shifts: {e}")
        
        return shifts

    async def _detect_greed_fear_shifts(self, symbol: str) -> List[MarketShift]:
        """Detect shifts in market greed/fear index"""
        shifts = []
        
        try:
            # This would connect to a fear/greed index API
            # For now, simulate based on price volatility and volume
            current_volatility = np.random.uniform(0.3, 0.8)  # Placeholder
            
            # In real implementation, compare with historical fear/greed data
            if current_volatility > 0.7:  # High fear/greed
                shifts.append(MarketShift(
                    timestamp=datetime.datetime.now(),
                    shift_type='greed_fear',
                    old_value=0.5,
                    new_value=current_volatility,
                    severity=current_volatility,
                    confidence=0.6,
                    source='market_psychology',
                    description=f"High market emotion detected: {current_volatility:.2f}"
                ))
                
        except Exception as e:
            logger.error(f"Error detecting greed/fear shifts: {e}")
        
        return shifts

    async def _detect_economic_shifts(self, symbol: str, baseline: Dict) -> List[MarketShift]:
        """Detect economic factor changes that impact gold"""
        shifts = []
        
        try:
            # Check for major economic announcements, rate changes, etc.
            # This would integrate with economic calendar APIs
            
            # For now, detect volatility spikes
            current_volatility = np.random.uniform(0.2, 0.9)  # Placeholder
            baseline_volatility = baseline.get('volatility', 0.5)
            
            volatility_change = abs(current_volatility - baseline_volatility)
            
            if volatility_change > self.shift_thresholds['economic']:
                shifts.append(MarketShift(
                    timestamp=datetime.datetime.now(),
                    shift_type='economic',
                    old_value=baseline_volatility,
                    new_value=current_volatility,
                    severity=volatility_change * 2,
                    confidence=0.7,
                    source='economic_indicators',
                    description=f"Economic volatility shift: {baseline_volatility:.2f} → {current_volatility:.2f}"
                ))
                
        except Exception as e:
            logger.error(f"Error detecting economic shifts: {e}")
        
        return shifts

    async def _update_prediction_due_to_shifts(self, symbol: str, shifts: List[MarketShift]):
        """Update prediction when significant market shifts are detected"""
        try:
            current_prediction_data = self.current_predictions.get(symbol)
            if not current_prediction_data:
                return
            
            original_prediction = current_prediction_data['prediction']
            
            # Analyze the nature of shifts
            shift_impact = self._analyze_shift_impact(shifts)
            
            # Generate new prediction considering the shifts
            new_prediction = await self._generate_updated_prediction(symbol, original_prediction, shifts, shift_impact)
            
            if new_prediction:
                # Calculate confidence change
                old_confidence = np.mean(list(original_prediction.confidence_scores.values()))
                new_confidence = np.mean(list(new_prediction.confidence_scores.values()))
                confidence_change = new_confidence - old_confidence
                
                # Update current prediction
                self.current_predictions[symbol] = {
                    'prediction': new_prediction,
                    'created_at': current_prediction_data['created_at'],
                    'last_updated': datetime.datetime.now(),
                    'update_count': current_prediction_data['update_count'] + 1
                }
                
                # Store the update
                update = PredictionUpdate(
                    timestamp=datetime.datetime.now(),
                    original_prediction=asdict(original_prediction),
                    updated_prediction=asdict(new_prediction),
                    reason=f"Market shifts detected: {', '.join([s.shift_type for s in shifts])}",
                    market_shifts=shifts,
                    confidence_change=confidence_change
                )
                
                self._store_prediction_update(symbol, update)
                self._store_market_shifts(symbol, shifts)
                
                # Update baselines for continued monitoring
                await self._establish_baseline_conditions(symbol)
                
                logger.info(f"🔄 Updated prediction for {symbol} due to {len(shifts)} market shifts")
                logger.info(f"   Confidence change: {confidence_change:+.2f}")
                logger.info(f"   Reason: {update.reason}")
                
                return new_prediction
                
        except Exception as e:
            logger.error(f"Error updating prediction for {symbol}: {e}")
            return None

    def _analyze_shift_impact(self, shifts: List[MarketShift]) -> Dict:
        """Analyze the combined impact of detected shifts"""
        impact = {
            'overall_severity': np.mean([s.severity for s in shifts]),
            'overall_confidence': np.mean([s.confidence for s in shifts]),
            'shift_types': list(set([s.shift_type for s in shifts])),
            'bullish_factors': 0,
            'bearish_factors': 0,
            'direction_consensus': 'neutral'
        }
        
        # Analyze directional bias
        for shift in shifts:
            if shift.shift_type == 'trend':
                if isinstance(shift.new_value, (int, float)):
                    if shift.new_value > shift.old_value:
                        impact['bullish_factors'] += shift.severity
                    else:
                        impact['bearish_factors'] += shift.severity
            elif shift.shift_type == 'sentiment':
                if shift.new_value > 0.5:
                    impact['bullish_factors'] += shift.severity
                else:
                    impact['bearish_factors'] += shift.severity
        
        # Determine overall direction
        if impact['bullish_factors'] > impact['bearish_factors'] * 1.2:
            impact['direction_consensus'] = 'bullish'
        elif impact['bearish_factors'] > impact['bullish_factors'] * 1.2:
            impact['direction_consensus'] = 'bearish'
        
        return impact

    async def _generate_updated_prediction(self, symbol: str, original_prediction: MultiTimeframePrediction, 
                                         shifts: List[MarketShift], shift_impact: Dict) -> MultiTimeframePrediction:
        """Generate updated prediction considering market shifts"""
        try:
            # Get latest market data
            current_price_data = self.data_engine.get_live_price_data(symbol)
            current_price = current_price_data.get('price', original_prediction.current_price)
            
            # Adjust predictions based on shift impact
            direction_multiplier = 1.0
            if shift_impact['direction_consensus'] == 'bullish':
                direction_multiplier = 1.0 + (shift_impact['overall_severity'] * 0.5)
            elif shift_impact['direction_consensus'] == 'bearish':
                direction_multiplier = 1.0 - (shift_impact['overall_severity'] * 0.5)
            
            # Generate new predictions
            new_predictions = {}
            new_predicted_prices = {}
            new_confidence_scores = {}
            
            for timeframe, original_change in original_prediction.predictions.items():
                # Adjust prediction based on shifts
                if shift_impact['direction_consensus'] != 'neutral':
                    # Apply shift correction
                    adjusted_change = original_change * direction_multiplier
                    
                    # Limit extreme adjustments
                    max_adjustment = abs(original_change) * 2
                    adjusted_change = max(-max_adjustment, min(max_adjustment, adjusted_change))
                else:
                    # Conservative adjustment for neutral shifts
                    adjusted_change = original_change * 0.8
                
                new_predictions[timeframe] = adjusted_change
                new_predicted_prices[timeframe] = current_price * (1 + adjusted_change / 100)
                
                # Adjust confidence based on shift certainty
                original_confidence = original_prediction.confidence_scores[timeframe]
                confidence_adjustment = shift_impact['overall_confidence'] - 0.5
                new_confidence = original_confidence + (confidence_adjustment * 0.3)
                new_confidence_scores[timeframe] = max(0.1, min(0.95, new_confidence))
            
            # Generate updated reasoning
            shift_descriptions = [s.description for s in shifts[:3]]  # Top 3 shifts
            new_reasoning = f"Updated due to market shifts: {'; '.join(shift_descriptions)}. {original_prediction.reasoning}"
            
            # Create updated prediction
            updated_prediction = MultiTimeframePrediction(
                symbol=symbol,
                current_price=current_price,
                prediction_date=datetime.date.today(),
                predictions=new_predictions,
                predicted_prices=new_predicted_prices,
                confidence_scores=new_confidence_scores,
                strategy_id=original_prediction.strategy_id,
                reasoning=new_reasoning,
                technical_indicators=original_prediction.technical_indicators,
                sentiment_data=original_prediction.sentiment_data,
                market_conditions=original_prediction.market_conditions
            )
            
            return updated_prediction
            
        except Exception as e:
            logger.error(f"Error generating updated prediction: {e}")
            return None

    def _store_prediction_update(self, symbol: str, update: PredictionUpdate):
        """Store prediction update in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO prediction_updates 
                (symbol, original_prediction, updated_prediction, reason, confidence_change, market_shifts_count, timestamp) 
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                json.dumps(update.original_prediction),
                json.dumps(update.updated_prediction),
                update.reason,
                update.confidence_change,
                len(update.market_shifts),
                update.timestamp
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing prediction update: {e}")

    def _store_market_shifts(self, symbol: str, shifts: List[MarketShift]):
        """Store detected market shifts in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for shift in shifts:
                cursor.execute('''
                    INSERT INTO market_shifts 
                    (symbol, shift_type, old_value, new_value, severity, confidence, source, description, timestamp) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol,
                    shift.shift_type,
                    json.dumps(shift.old_value),
                    json.dumps(shift.new_value),
                    shift.severity,
                    shift.confidence,
                    shift.source,
                    shift.description,
                    shift.timestamp
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing market shifts: {e}")

    def get_prediction_history(self, symbol: str, limit: int = 10) -> List[Dict]:
        """Get history of prediction updates for a symbol"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM prediction_updates 
                WHERE symbol = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (symbol, limit))
            
            updates = []
            for row in cursor.fetchall():
                updates.append({
                    'id': row[0],
                    'timestamp': row[1],
                    'symbol': row[2],
                    'original_prediction': json.loads(row[3]),
                    'updated_prediction': json.loads(row[4]),
                    'reason': row[5],
                    'confidence_change': row[6],
                    'market_shifts_count': row[7]
                })
            
            conn.close()
            return updates
            
        except Exception as e:
            logger.error(f"Error getting prediction history: {e}")
            return []

    def stop_monitoring(self):
        """Stop the market monitoring thread"""
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)
        logger.info("🛑 Dynamic prediction monitoring stopped")

# Global instance
dynamic_prediction_engine = DynamicPredictionEngine()
