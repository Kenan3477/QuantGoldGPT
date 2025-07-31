"""
Self-Improving Daily ML Prediction Engine
Makes one comprehensive prediction every 24 hours, tracks accuracy, and evolves strategy
"""

import sqlite3
import json
import datetime
from typing import Dict, List, Tuple, Optional
import logging
import numpy as np
from dataclasses import dataclass
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MultiTimeframePrediction:
    symbol: str
    current_price: float
    prediction_date: datetime.date
    
    # Predictions for different timeframes
    predictions: Dict[str, float]  # {'1h': 0.05, '4h': 0.12, etc}
    predicted_prices: Dict[str, float]
    confidence_scores: Dict[str, float]
    
    # Strategy information
    strategy_id: int
    reasoning: str
    technical_indicators: Dict
    sentiment_data: Dict
    market_conditions: Dict

@dataclass
class PredictionValidation:
    prediction_id: int
    timeframe: str
    actual_price: float
    predicted_price: float
    predicted_change: float
    actual_change: float
    direction_correct: bool
    performance_score: float

class SelfImprovingMLEngine:
    def __init__(self, db_path: str = "goldgpt_ml_learning.db"):
        self.db_path = db_path
        self.init_database()
        self.strategies = self.load_strategies()
        self.current_strategy_id = self.get_best_strategy_id()
        logger.info(f"🎯 Self-improving ML engine initialized with strategy {self.current_strategy_id}")
        
    def load_strategies(self) -> Dict:
        """Load available strategies from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, strategy_name, strategy_description, accuracy_rate, is_active
            FROM strategy_performance 
            WHERE is_active = 1
            ORDER BY accuracy_rate DESC
        ''')
        
        strategies = {}
        for row in cursor.fetchall():
            strategies[row[0]] = {
                'id': row[0],
                'strategy_name': row[1],
                'strategy_description': row[2],
                'accuracy_rate': row[3],
                'is_active': row[4]
            }
        
        conn.close()
        return strategies
    
    def get_best_strategy_id(self) -> int:
        """Get the currently best performing strategy"""
        if not self.strategies:
            return 1  # Default to first strategy
        
        # Return strategy with highest accuracy
        best_strategy = max(self.strategies.values(), key=lambda s: s['accuracy_rate'])
        return best_strategy['id']
    
    def get_strategy(self, strategy_id: int) -> Dict:
        """Get strategy by ID"""
        return self.strategies.get(strategy_id, list(self.strategies.values())[0])
    
    def has_prediction_for_date(self, date: datetime.date, symbol: str) -> bool:
        """Check if prediction exists for given date"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT COUNT(*) FROM daily_predictions 
            WHERE prediction_date = ? AND symbol = ?
        ''', (str(date), symbol))
        
        count = cursor.fetchone()[0]
        conn.close()
        
        return count > 0
    
    def get_prediction_for_date(self, date: datetime.date, symbol: str) -> Optional[MultiTimeframePrediction]:
        """Get existing prediction for date"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, prediction_date, symbol, current_price,
                   prediction_1h, prediction_4h, prediction_1d, prediction_3d, prediction_7d,
                   predicted_price_1h, predicted_price_4h, predicted_price_1d, predicted_price_3d, predicted_price_7d,
                   confidence_1h, confidence_4h, confidence_1d, confidence_3d, confidence_7d,
                   strategy_id, reasoning, technical_indicators, sentiment_data, market_conditions
            FROM daily_predictions 
            WHERE prediction_date = ? AND symbol = ?
        ''', (str(date), symbol))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        # Convert database row back to MultiTimeframePrediction object
        predictions = {
            '1h': row[4], '4h': row[5], '1d': row[6], '3d': row[7], '7d': row[8]
        }
        predicted_prices = {
            '1h': row[9], '4h': row[10], '1d': row[11], '3d': row[12], '7d': row[13]
        }
        confidence_scores = {
            '1h': row[14], '4h': row[15], '1d': row[16], '3d': row[17], '7d': row[18]
        }
        
        prediction = MultiTimeframePrediction(
            symbol=row[2],
            current_price=row[3],
            prediction_date=datetime.datetime.strptime(str(row[1]), '%Y-%m-%d').date(),
            predictions=predictions,
            predicted_prices=predicted_prices,
            confidence_scores=confidence_scores,
            strategy_id=row[19],
            reasoning=row[20] or "Mock reasoning",
            technical_indicators=json.loads(row[21]) if row[21] else {},
            sentiment_data=json.loads(row[22]) if row[22] else {},
            market_conditions=json.loads(row[23]) if row[23] else {}
        )
        
        # Add the ID to the prediction object
        prediction.id = row[0]
        return prediction
    
    def store_daily_prediction(self, prediction: MultiTimeframePrediction) -> int:
        """Store prediction in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO daily_predictions (
                prediction_date, symbol, current_price,
                prediction_1h, prediction_4h, prediction_1d, prediction_3d, prediction_7d,
                predicted_price_1h, predicted_price_4h, predicted_price_1d, predicted_price_3d, predicted_price_7d,
                confidence_1h, confidence_4h, confidence_1d, confidence_3d, confidence_7d,
                strategy_id, reasoning, technical_indicators, sentiment_data, market_conditions
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            str(prediction.prediction_date),
            prediction.symbol,
            prediction.current_price,
            prediction.predictions['1h'],
            prediction.predictions['4h'], 
            prediction.predictions['1d'],
            prediction.predictions['3d'],
            prediction.predictions['7d'],
            prediction.predicted_prices['1h'],
            prediction.predicted_prices['4h'],
            prediction.predicted_prices['1d'],
            prediction.predicted_prices['3d'],
            prediction.predicted_prices['7d'],
            prediction.confidence_scores['1h'],
            prediction.confidence_scores['4h'],
            prediction.confidence_scores['1d'],
            prediction.confidence_scores['3d'],
            prediction.confidence_scores['7d'],
            prediction.strategy_id,
            prediction.reasoning,
            json.dumps(prediction.technical_indicators),
            json.dumps(prediction.sentiment_data),
            json.dumps(prediction.market_conditions)
        ))
        
        prediction_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return prediction_id
    
    def generate_reasoning(self, strategy: Dict, technical_data: Dict, sentiment_data: Dict) -> str:
        """Generate human-readable reasoning for the prediction"""
        strategy_name = strategy['strategy_name']
        
        reasoning_parts = [f"Using {strategy_name} strategy."]
        
        # Add technical analysis insights
        if technical_data:
            rsi = technical_data.get('rsi', 50)
            if rsi > 70:
                reasoning_parts.append("RSI indicates overbought conditions.")
            elif rsi < 30:
                reasoning_parts.append("RSI indicates oversold conditions.")
                
            macd = technical_data.get('macd', 0)
            if macd > 0:
                reasoning_parts.append("MACD shows bullish momentum.")
            else:
                reasoning_parts.append("MACD shows bearish momentum.")
        
        # Add sentiment insights
        if sentiment_data:
            sentiment_score = sentiment_data.get('sentiment_score', 0.5)
            if sentiment_score > 0.6:
                reasoning_parts.append("Market sentiment is positive.")
            elif sentiment_score < 0.4:
                reasoning_parts.append("Market sentiment is negative.")
        
        return " ".join(reasoning_parts)
    
    def assess_market_conditions(self, technical_data: Dict, sentiment_data: Dict) -> Dict:
        """Assess overall market conditions"""
        conditions = {
            'volatility': 'medium',
            'trend': 'neutral',
            'sentiment': 'neutral',
            'risk_level': 'medium'
        }
        
        # Simple assessment based on available data
        if technical_data:
            rsi = technical_data.get('rsi', 50)
            if rsi > 70 or rsi < 30:
                conditions['volatility'] = 'high'
                conditions['risk_level'] = 'high'
        
        if sentiment_data:
            sentiment_score = sentiment_data.get('sentiment_score', 0.5)
            if sentiment_score > 0.6:
                conditions['sentiment'] = 'positive'
                conditions['trend'] = 'bullish'
            elif sentiment_score < 0.4:
                conditions['sentiment'] = 'negative'  
                conditions['trend'] = 'bearish'
        
        return conditions
    
    def generate_mock_predictions(self, symbol: str, current_price: float, strategy: Dict) -> MultiTimeframePrediction:
        """Generate mock predictions when ML modules are not available"""
        import random
        
        # Generate realistic predictions based on strategy
        base_change = random.uniform(-2, 2)  # Base change percentage
        
        predictions = {}
        predicted_prices = {}
        confidence_scores = {}
        
        timeframes = ['1h', '4h', '1d', '3d', '7d']
        multipliers = {'1h': 0.3, '4h': 0.6, '1d': 1.0, '3d': 1.8, '7d': 2.5}
        
        for tf in timeframes:
            change = base_change * multipliers[tf] * random.uniform(0.8, 1.2)
            price = current_price * (1 + change / 100)
            confidence = random.uniform(0.6, 0.9)
            
            predictions[tf] = change
            predicted_prices[tf] = price
            confidence_scores[tf] = confidence
        
        return MultiTimeframePrediction(
            symbol=symbol,
            current_price=current_price,
            prediction_date=datetime.date.today(),
            predictions=predictions,
            predicted_prices=predicted_prices,
            confidence_scores=confidence_scores,
            strategy_id=strategy['id'],
            reasoning=f"Mock prediction using {strategy['strategy_name']} strategy",
            technical_indicators={'rsi': 50, 'macd': 0},
            sentiment_data={'sentiment_score': 0.5},
            market_conditions={'volatility': 'medium', 'trend': 'neutral'}
        )
        
    def init_database(self):
        """Initialize the learning database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.executescript('''
        CREATE TABLE IF NOT EXISTS daily_predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_date DATE NOT NULL UNIQUE,
            symbol VARCHAR(10) NOT NULL,
            current_price DECIMAL(10,2) NOT NULL,
            
            prediction_1h DECIMAL(5,3) NOT NULL,
            prediction_4h DECIMAL(5,3) NOT NULL,
            prediction_1d DECIMAL(5,3) NOT NULL,
            prediction_3d DECIMAL(5,3) NOT NULL,
            prediction_7d DECIMAL(5,3) NOT NULL,
            
            predicted_price_1h DECIMAL(10,2) NOT NULL,
            predicted_price_4h DECIMAL(10,2) NOT NULL,
            predicted_price_1d DECIMAL(10,2) NOT NULL,
            predicted_price_3d DECIMAL(10,2) NOT NULL,
            predicted_price_7d DECIMAL(10,2) NOT NULL,
            
            confidence_1h DECIMAL(3,3) NOT NULL,
            confidence_4h DECIMAL(3,3) NOT NULL,
            confidence_1d DECIMAL(3,3) NOT NULL,
            confidence_3d DECIMAL(3,3) NOT NULL,
            confidence_7d DECIMAL(3,3) NOT NULL,
            
            strategy_id INTEGER NOT NULL,
            reasoning TEXT NOT NULL,
            technical_indicators TEXT NOT NULL,
            sentiment_data TEXT NOT NULL,
            market_conditions TEXT NOT NULL,
            
            status VARCHAR(20) DEFAULT 'pending',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS prediction_validation (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_id INTEGER NOT NULL,
            timeframe VARCHAR(5) NOT NULL,
            validation_date TIMESTAMP NOT NULL,
            actual_price DECIMAL(10,2) NOT NULL,
            predicted_price DECIMAL(10,2) NOT NULL,
            predicted_change DECIMAL(5,3) NOT NULL,
            actual_change DECIMAL(5,3) NOT NULL,
            price_accuracy DECIMAL(5,3) NOT NULL,
            direction_correct BOOLEAN NOT NULL,
            confidence_score DECIMAL(3,3) NOT NULL,
            error_margin DECIMAL(5,3) NOT NULL,
            performance_score DECIMAL(3,3) NOT NULL,
            
            FOREIGN KEY (prediction_id) REFERENCES daily_predictions(id)
        );
        
        CREATE TABLE IF NOT EXISTS strategy_performance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_name VARCHAR(100) NOT NULL UNIQUE,
            strategy_description TEXT NOT NULL,
            total_predictions INTEGER DEFAULT 0,
            correct_predictions INTEGER DEFAULT 0,
            accuracy_rate DECIMAL(5,3) DEFAULT 0,
            avg_confidence DECIMAL(3,3) DEFAULT 0,
            avg_error_margin DECIMAL(5,3) DEFAULT 0,
            accuracy_1h DECIMAL(5,3) DEFAULT 0,
            accuracy_4h DECIMAL(5,3) DEFAULT 0,
            accuracy_1d DECIMAL(5,3) DEFAULT 0,
            accuracy_3d DECIMAL(5,3) DEFAULT 0,
            accuracy_7d DECIMAL(5,3) DEFAULT 0,
            is_active BOOLEAN DEFAULT 1,
            last_used TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE TABLE IF NOT EXISTS learning_insights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            insight_type VARCHAR(50) NOT NULL,
            insight_data TEXT NOT NULL,
            confidence DECIMAL(3,3) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        ''')
        
        # Initialize base strategies if empty
        cursor.execute("SELECT COUNT(*) FROM strategy_performance")
        if cursor.fetchone()[0] == 0:
            self.init_base_strategies(cursor)
            
        conn.commit()
        conn.close()
        logger.info("✅ Self-improving ML database initialized")
    
    def init_base_strategies(self, cursor):
        """Initialize the base ML strategies"""
        base_strategies = [
            ("Technical Heavy", "Focus on RSI, MACD, Bollinger Bands, moving averages"),
            ("Sentiment Heavy", "Focus on news sentiment, social media, fear/greed index"),
            ("Pattern Recognition", "Focus on chart patterns, support/resistance levels"),
            ("Hybrid Ensemble", "Balanced combination of technical, sentiment, and patterns"),
            ("Momentum Following", "Trend continuation and momentum indicators"),
            ("Mean Reversion", "Contrarian approach, oversold/overbought conditions"),
            ("Volume Analysis", "Focus on volume patterns and price-volume relationships"),
            ("Volatility Based", "Focus on volatility indicators and market uncertainty")
        ]
        
        for name, description in base_strategies:
            cursor.execute('''
                INSERT INTO strategy_performance (strategy_name, strategy_description)
                VALUES (?, ?)
            ''', (name, description))
            
        logger.info("🎯 Initialized 8 base ML strategies")
    
    def generate_daily_prediction(self, symbol: str = "XAUUSD") -> MultiTimeframePrediction:
        """Generate the single daily multi-timeframe prediction"""
        
        # Check if we already have a prediction for today
        today = datetime.date.today()
        if self.has_prediction_for_date(today, symbol):
            logger.info(f"📅 Prediction already exists for {today}")
            return self.get_prediction_for_date(today, symbol)
            
        logger.info(f"🎯 Generating daily prediction for {symbol} on {today}")
        
        try:
            # Get current market data
            current_price = self.get_current_price(symbol)
            
            # Use current best strategy
            strategy_id = self.current_strategy_id
            strategy = self.get_strategy(strategy_id)
            
            # Generate predictions for all timeframes
            predictions = self.generate_multi_timeframe_predictions(
                symbol, current_price, strategy
            )
            
            # Store the prediction
            prediction_id = self.store_daily_prediction(predictions)
            predictions.id = prediction_id
            
            logger.info(f"✅ Daily prediction generated and stored (ID: {prediction_id})")
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Failed to generate daily prediction: {e}")
            raise
    
    def generate_multi_timeframe_predictions(self, symbol: str, current_price: float, strategy: Dict) -> MultiTimeframePrediction:
        """Generate predictions for multiple timeframes using the selected strategy"""
        
        # Import the actual ML prediction modules
        try:
            from intelligent_ml_predictor import get_intelligent_ml_predictions
            from technical_analysis import get_technical_analysis
            from enhanced_news_analyzer import get_news_sentiment
        except ImportError:
            logger.warning("⚠️ ML modules not available, using mock data")
            return self.generate_mock_predictions(symbol, current_price, strategy)
        
        # Get comprehensive market data
        technical_data = get_technical_analysis(symbol)
        sentiment_data = get_news_sentiment(symbol)
        
        # Apply strategy-specific weighting
        predictions = {}
        predicted_prices = {}
        confidence_scores = {}
        
        timeframes = ['1h', '4h', '1d', '3d', '7d']
        
        for timeframe in timeframes:
            # Generate prediction based on strategy
            pred_change, pred_price, confidence = self.apply_strategy_prediction(
                strategy, current_price, technical_data, sentiment_data, timeframe
            )
            
            predictions[timeframe] = pred_change
            predicted_prices[timeframe] = pred_price
            confidence_scores[timeframe] = confidence
        
        return MultiTimeframePrediction(
            symbol=symbol,
            current_price=current_price,
            prediction_date=datetime.date.today(),
            predictions=predictions,
            predicted_prices=predicted_prices,
            confidence_scores=confidence_scores,
            strategy_id=strategy['id'],
            reasoning=self.generate_reasoning(strategy, technical_data, sentiment_data),
            technical_indicators=technical_data,
            sentiment_data=sentiment_data,
            market_conditions=self.assess_market_conditions(technical_data, sentiment_data)
        )
    
    def apply_strategy_prediction(self, strategy: Dict, current_price: float, 
                                technical_data: Dict, sentiment_data: Dict, 
                                timeframe: str) -> Tuple[float, float, float]:
        """Apply strategy-specific logic to generate a prediction"""
        
        strategy_name = strategy['strategy_name']
        
        # Base prediction from intelligent ML
        try:
            from intelligent_ml_predictor import get_intelligent_ml_predictions
            base_prediction = get_intelligent_ml_predictions('XAUUSD')
            
            if base_prediction and 'predictions' in base_prediction:
                # Find matching timeframe or use closest
                base_pred = None
                for pred in base_prediction['predictions']:
                    if pred.get('timeframe') == timeframe:
                        base_pred = pred
                        break
                
                if base_pred:
                    base_change = base_pred.get('change_percent', 0)
                    base_confidence = base_pred.get('confidence', 0.5)
                else:
                    base_change = 0
                    base_confidence = 0.5
            else:
                base_change = 0
                base_confidence = 0.5
                
        except Exception:
            base_change = 0
            base_confidence = 0.5
        
        # Apply strategy-specific modifications
        if strategy_name == "Technical Heavy":
            # Weight technical indicators more heavily
            rsi = technical_data.get('rsi', 50)
            if rsi > 70:
                base_change *= 0.7  # Reduce bullish prediction
            elif rsi < 30:
                base_change *= 1.3  # Increase bullish potential
                
            confidence = min(0.95, base_confidence * 1.2)
            
        elif strategy_name == "Sentiment Heavy":
            # Weight sentiment more heavily
            sentiment_score = sentiment_data.get('sentiment_score', 0.5)
            if sentiment_score > 0.6:
                base_change *= 1.2
            elif sentiment_score < 0.4:
                base_change *= 0.8
                
            confidence = min(0.95, base_confidence * 1.1)
            
        elif strategy_name == "Pattern Recognition":
            # Focus on chart patterns
            pattern = technical_data.get('pattern', 'none')
            if 'bullish' in pattern.lower():
                base_change = abs(base_change) if base_change != 0 else 0.1
            elif 'bearish' in pattern.lower():
                base_change = -abs(base_change) if base_change != 0 else -0.1
                
            confidence = base_confidence
            
        elif strategy_name == "Mean Reversion":
            # Contrarian approach
            base_change *= -0.8  # Reverse the prediction partially
            confidence = min(0.9, base_confidence * 0.9)
            
        elif strategy_name == "Momentum Following":
            # Follow trends strongly
            if base_change > 0:
                base_change *= 1.3
            else:
                base_change *= 1.3
            confidence = min(0.95, base_confidence * 1.1)
            
        else:  # Hybrid or other strategies
            confidence = base_confidence
        
        # Adjust for timeframe
        timeframe_multipliers = {'1h': 0.3, '4h': 0.6, '1d': 1.0, '3d': 1.8, '7d': 2.5}
        base_change *= timeframe_multipliers.get(timeframe, 1.0)
        
        # Calculate predicted price
        predicted_price = current_price * (1 + base_change / 100)
        
        return base_change, predicted_price, confidence
    
    def validate_predictions(self, symbol: str = "XAUUSD"):
        """Check and validate previous predictions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get predictions that need validation
        cursor.execute('''
            SELECT id, prediction_date, current_price, 
                   prediction_1h, prediction_4h, prediction_1d, prediction_3d, prediction_7d,
                   predicted_price_1h, predicted_price_4h, predicted_price_1d, predicted_price_3d, predicted_price_7d,
                   confidence_1h, confidence_4h, confidence_1d, confidence_3d, confidence_7d,
                   strategy_id
            FROM daily_predictions 
            WHERE symbol = ? AND status = 'pending'
            ORDER BY prediction_date DESC
        ''', (symbol,))
        
        predictions_to_validate = cursor.fetchall()
        
        for pred in predictions_to_validate:
            self.validate_single_prediction(pred, cursor)
        
        conn.commit()
        conn.close()
        
        # Update strategy performance
        self.update_strategy_performance()
        
        # Check if strategy needs to change
        self.evaluate_strategy_change()
        
    def validate_single_prediction(self, prediction_data, cursor):
        """Validate a single prediction against actual market data"""
        (pred_id, pred_date, orig_price, 
         pred_1h, pred_4h, pred_1d, pred_3d, pred_7d,
         price_1h, price_4h, price_1d, price_3d, price_7d,
         conf_1h, conf_4h, conf_1d, conf_3d, conf_7d,
         strategy_id) = prediction_data
        
        pred_date = datetime.datetime.strptime(pred_date, '%Y-%m-%d').date()
        current_time = datetime.datetime.now()
        
        # Define validation timepoints
        timeframes = {
            '1h': (pred_1h, price_1h, conf_1h, datetime.timedelta(hours=1)),
            '4h': (pred_4h, price_4h, conf_4h, datetime.timedelta(hours=4)),
            '1d': (pred_1d, price_1d, conf_1d, datetime.timedelta(days=1)),
            '3d': (pred_3d, price_3d, conf_3d, datetime.timedelta(days=3)),
            '7d': (pred_7d, price_7d, conf_7d, datetime.timedelta(days=7))
        }
        
        validations_made = 0
        
        for timeframe, (pred_change, pred_price, confidence, time_delta) in timeframes.items():
            target_time = datetime.datetime.combine(pred_date, datetime.time()) + time_delta
            
            # Only validate if enough time has passed
            if current_time >= target_time:
                # Check if already validated
                cursor.execute('''
                    SELECT COUNT(*) FROM prediction_validation 
                    WHERE prediction_id = ? AND timeframe = ?
                ''', (pred_id, timeframe))
                
                if cursor.fetchone()[0] == 0:
                    # Get actual price at validation time
                    actual_price = self.get_historical_price(target_time)
                    
                    if actual_price:
                        # Calculate validation metrics
                        actual_change = ((actual_price - orig_price) / orig_price) * 100
                        price_accuracy = 1 - abs(actual_price - pred_price) / orig_price
                        direction_correct = (pred_change > 0) == (actual_change > 0)
                        error_margin = abs(pred_change - actual_change)
                        
                        # Performance score (weighted combination)
                        performance_score = (
                            (0.4 * price_accuracy) + 
                            (0.3 * (1 if direction_correct else 0)) +
                            (0.3 * (1 - min(error_margin / 5, 1)))  # Error margin penalty
                        ) * confidence
                        
                        # Store validation
                        cursor.execute('''
                            INSERT INTO prediction_validation 
                            (prediction_id, timeframe, validation_date, actual_price, predicted_price,
                             predicted_change, actual_change, price_accuracy, direction_correct,
                             confidence_score, error_margin, performance_score)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (pred_id, timeframe, target_time, actual_price, pred_price,
                              pred_change, actual_change, price_accuracy, direction_correct,
                              confidence, error_margin, performance_score))
                        
                        validations_made += 1
                        logger.info(f"✅ Validated {timeframe} prediction: {performance_score:.3f} score")
        
        # Update prediction status if all timeframes validated
        cursor.execute('''
            SELECT COUNT(*) FROM prediction_validation WHERE prediction_id = ?
        ''', (pred_id,))
        
        total_validations = cursor.fetchone()[0]
        if total_validations >= 5:  # All timeframes validated
            cursor.execute('''
                UPDATE daily_predictions SET status = 'completed' WHERE id = ?
            ''', (pred_id,))
        elif total_validations > 0:
            cursor.execute('''
                UPDATE daily_predictions SET status = 'partial' WHERE id = ?
            ''', (pred_id,))
    
    def get_current_price(self, symbol: str) -> float:
        """Get current price for symbol"""
        # Mock implementation - replace with real data source
        import random
        return 3350.0 + random.uniform(-50, 50)
    
    def get_historical_price(self, target_time: datetime.datetime) -> Optional[float]:
        """Get historical price at specific time"""
        # Mock implementation - replace with real historical data
        import random
        return 3350.0 + random.uniform(-100, 100)
    
    def update_strategy_performance(self):
        """Update strategy performance metrics based on validation results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get validation results grouped by strategy
        cursor.execute('''
            SELECT 
                dp.strategy_id,
                COUNT(pv.id) as total_validations,
                AVG(pv.performance_score) as avg_performance,
                AVG(pv.confidence_score) as avg_confidence,
                AVG(pv.error_margin) as avg_error,
                SUM(CASE WHEN pv.direction_correct = 1 THEN 1 ELSE 0 END) as correct_predictions
            FROM daily_predictions dp
            JOIN prediction_validation pv ON dp.id = pv.prediction_id
            GROUP BY dp.strategy_id
        ''')
        
        results = cursor.fetchall()
        
        for result in results:
            strategy_id, total, avg_perf, avg_conf, avg_error, correct = result
            accuracy_rate = correct / total if total > 0 else 0
            
            # Update strategy performance
            cursor.execute('''
                UPDATE strategy_performance 
                SET total_predictions = ?,
                    correct_predictions = ?,
                    accuracy_rate = ?,
                    avg_confidence = ?,
                    avg_error_margin = ?,
                    last_used = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (total, correct, accuracy_rate, avg_conf, avg_error, strategy_id))
        
        conn.commit()
        conn.close()
        logger.info("✅ Strategy performance updated")
    
    def evaluate_strategy_change(self):
        """Evaluate if strategy should be changed based on performance"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get recent performance of current strategy
        cursor.execute('''
            SELECT accuracy_rate, total_predictions
            FROM strategy_performance 
            WHERE id = ?
        ''', (self.current_strategy_id,))
        
        result = cursor.fetchone()
        
        if result:
            current_accuracy, total_predictions = result
            
            # Change strategy if accuracy is below 60% and we have at least 5 predictions
            if total_predictions >= 5 and current_accuracy < 0.6:
                # Find best performing strategy
                cursor.execute('''
                    SELECT id, accuracy_rate
                    FROM strategy_performance 
                    WHERE is_active = 1 AND id != ?
                    ORDER BY accuracy_rate DESC
                    LIMIT 1
                ''', (self.current_strategy_id,))
                
                best_strategy = cursor.fetchone()
                
                if best_strategy and best_strategy[1] > current_accuracy:
                    old_strategy_id = self.current_strategy_id
                    self.current_strategy_id = best_strategy[0]
                    
                    logger.info(f"🔄 Strategy changed from {old_strategy_id} to {self.current_strategy_id}")
                    logger.info(f"   Old accuracy: {current_accuracy:.1%}")
                    logger.info(f"   New accuracy: {best_strategy[1]:.1%}")
        
        conn.close()
    
    def get_strategy_performance_report(self) -> Dict:
        """Get comprehensive strategy performance report"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                id, strategy_name, strategy_description,
                total_predictions, correct_predictions, accuracy_rate,
                avg_confidence, avg_error_margin,
                accuracy_1h, accuracy_4h, accuracy_1d, accuracy_3d, accuracy_7d,
                is_active, last_used
            FROM strategy_performance
            ORDER BY accuracy_rate DESC
        ''')
        
        strategies = []
        for row in cursor.fetchall():
            strategies.append({
                'id': row[0],
                'name': row[1],
                'description': row[2],
                'total_predictions': row[3],
                'correct_predictions': row[4],
                'accuracy_rate': row[5],
                'avg_confidence': row[6],
                'avg_error_margin': row[7],
                'timeframe_accuracy': {
                    '1h': row[8],
                    '4h': row[9],
                    '1d': row[10],
                    '3d': row[11],
                    '7d': row[12]
                },
                'is_active': row[13],
                'last_used': row[14]
            })
        
        conn.close()
        
        return {
            'strategies': strategies,
            'current_strategy_id': self.current_strategy_id,
            'total_strategies': len(strategies),
            'generated_at': datetime.datetime.now().isoformat()
        }
    
    def should_change_strategy(self) -> bool:
        """Determine if strategy should be changed"""
        # Simple rule: change if current strategy has accuracy < 60% with at least 5 predictions
        current_strategy = self.strategies.get(self.current_strategy_id)
        if not current_strategy:
            return True
            
        return (current_strategy.get('accuracy_rate', 0) < 0.6 and 
                current_strategy.get('total_predictions', 0) >= 5)
    
    def select_best_strategy(self) -> int:
        """Select the best performing strategy"""
        if not self.strategies:
            return 1
            
        best_strategy = max(self.strategies.values(), key=lambda s: s.get('accuracy_rate', 0))
        return best_strategy['id']
    
    def get_performance_dashboard(self) -> Dict:
        """Get performance dashboard data for API"""
        performance_report = self.get_strategy_performance_report()
        
        # Get recent predictions
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                prediction_date, symbol, current_price,
                prediction_1d, predicted_price_1d, confidence_1d,
                strategy_id, status
            FROM daily_predictions
            ORDER BY prediction_date DESC
            LIMIT 10
        ''')
        
        recent_predictions = []
        for row in cursor.fetchall():
            recent_predictions.append({
                'date': row[0],
                'symbol': row[1],
                'current_price': row[2],
                'prediction_1d': row[3],
                'predicted_price_1d': row[4],
                'confidence_1d': row[5],
                'strategy_id': row[6],
                'status': row[7]
            })
        
        conn.close()
        
        return {
            'success': True,
            'performance_report': performance_report,
            'recent_predictions': recent_predictions,
            'system_status': {
                'current_strategy': self.current_strategy_id,
                'total_strategies': len(self.strategies),
                'database_path': self.db_path
            },
            'generated_at': datetime.datetime.now().isoformat()
        }

# Usage example
if __name__ == "__main__":
    engine = SelfImprovingMLEngine()
    
    # Generate today's prediction
    prediction = engine.generate_daily_prediction("XAUUSD")
    print(f"📊 Daily Prediction Generated:")
    print(f"1H: {prediction.predictions['1h']:.1f}% (${prediction.predicted_prices['1h']:.2f})")
    print(f"4H: {prediction.predictions['4h']:.1f}% (${prediction.predicted_prices['4h']:.2f})")
    print(f"1D: {prediction.predictions['1d']:.1f}% (${prediction.predicted_prices['1d']:.2f})")
    
    # Validate previous predictions
    engine.validate_predictions("XAUUSD")
