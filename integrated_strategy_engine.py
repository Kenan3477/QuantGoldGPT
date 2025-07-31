#!/usr/bin/env python3
"""
GoldGPT Integrated Strategy Engine
Combines ML predictions, AI analysis, signal generation, and advanced backtesting
"""

import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json
import logging

# Import existing components
try:
    from advanced_backtesting_framework import (
        AdvancedBacktestEngine, AdvancedHistoricalDataManager,
        StrategyOptimizer, BacktestVisualization, Trade, OHLCV, BacktestResult
    )
    BACKTESTING_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Backtesting framework not available: {e}")
    BACKTESTING_AVAILABLE = False

try:
    from dual_ml_prediction_system import DualMLPredictionSystem
    ML_SYSTEM_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ ML system not available: {e}")
    ML_SYSTEM_AVAILABLE = False

try:
    from enhanced_signal_generator import EnhancedAISignalGenerator
    SIGNAL_GENERATOR_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Signal generator not available: {e}")
    SIGNAL_GENERATOR_AVAILABLE = False

try:
    from ai_analysis_api import get_ai_analysis_sync
    AI_ANALYSIS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ AI analysis not available: {e}")
    AI_ANALYSIS_AVAILABLE = False

try:
    from data_pipeline_core import data_pipeline
    DATA_PIPELINE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Data pipeline not available: {e}")
    DATA_PIPELINE_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class IntegratedSignal:
    """Unified signal combining all analysis sources"""
    timestamp: datetime
    symbol: str
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0.0 to 1.0
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    timeframe: str = "1h"
    
    # Source contributions
    ml_prediction: Optional[Dict] = None
    ai_analysis: Optional[Dict] = None
    technical_signals: Optional[Dict] = None
    news_sentiment: Optional[Dict] = None
    
    # Risk management
    position_size: float = 0.01  # 1% of portfolio
    risk_reward_ratio: float = 2.0
    
    # Metadata
    strategy_name: str = "integrated"
    notes: str = ""

class IntegratedStrategyEngine:
    """
    Main strategy engine that combines all GoldGPT components
    """
    
    def __init__(self):
        self.db_path = "goldgpt_integrated_strategies.db"
        self.initialize_database()
        
        # Initialize components
        self.ml_system = None
        self.signal_generator = None
        self.backtest_engine = None
        self.data_manager = None
        
        self._initialize_components()
        
        # Strategy configurations (lowered confidence thresholds for more trades)
        self.strategies = {
            "ml_momentum": {
                "ml_weight": 0.4,
                "ai_weight": 0.3,
                "technical_weight": 0.2,
                "sentiment_weight": 0.1,
                "min_confidence": 0.45  # Lowered from 0.65
            },
            "conservative": {
                "ml_weight": 0.3,
                "ai_weight": 0.4,
                "technical_weight": 0.25,
                "sentiment_weight": 0.05,
                "min_confidence": 0.55  # Lowered from 0.75
            },
            "aggressive": {
                "ml_weight": 0.5,
                "ai_weight": 0.2,
                "technical_weight": 0.2,
                "sentiment_weight": 0.1,
                "min_confidence": 0.35  # Lowered from 0.55
            }
        }
        
        logger.info("✅ Integrated Strategy Engine initialized")

    def _initialize_components(self):
        """Initialize all available components"""
        if ML_SYSTEM_AVAILABLE:
            try:
                self.ml_system = DualMLPredictionSystem()
                logger.info("✅ ML system initialized")
            except Exception as e:
                logger.error(f"❌ ML system initialization failed: {e}")

        if SIGNAL_GENERATOR_AVAILABLE:
            try:
                self.signal_generator = EnhancedAISignalGenerator()
                logger.info("✅ Signal generator initialized")
            except Exception as e:
                logger.error(f"❌ Signal generator initialization failed: {e}")

        if BACKTESTING_AVAILABLE:
            try:
                self.data_manager = AdvancedHistoricalDataManager()
                self.backtest_engine = AdvancedBacktestEngine(
                    initial_capital=10000
                    # Note: commission and slippage are handled in strategy logic
                )
                logger.info("✅ Backtesting engine initialized")
            except Exception as e:
                logger.error(f"❌ Backtesting initialization failed: {e}")
                self.backtest_engine = None

    def initialize_database(self):
        """Initialize integrated strategy database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Integrated signals table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS integrated_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                confidence REAL NOT NULL,
                entry_price REAL NOT NULL,
                stop_loss REAL,
                take_profit REAL,
                timeframe TEXT DEFAULT '1h',
                ml_prediction TEXT,
                ai_analysis TEXT,
                technical_signals TEXT,
                news_sentiment TEXT,
                position_size REAL DEFAULT 0.01,
                risk_reward_ratio REAL DEFAULT 2.0,
                strategy_name TEXT DEFAULT 'integrated',
                notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Strategy performance tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategy_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                total_return REAL,
                win_rate REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                total_trades INTEGER,
                avg_trade_duration REAL,
                last_updated TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Backtest results storage
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS backtest_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                strategy_name TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                start_date TEXT NOT NULL,
                end_date TEXT NOT NULL,
                initial_capital REAL,
                final_capital REAL,
                total_return REAL,
                total_return_percent REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                total_trades INTEGER,
                win_rate REAL,
                performance_data TEXT,
                parameters TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("✅ Integrated strategy database initialized")

    async def generate_integrated_signal(self, symbol: str = "XAU", timeframe: str = "1h") -> Optional[IntegratedSignal]:
        """
        Generate unified trading signal combining all sources
        """
        try:
            current_price = await self._get_current_price(symbol)
            if not current_price:
                return None

            # Collect signals from all sources
            ml_prediction = await self._get_ml_prediction(symbol, timeframe)
            ai_analysis = await self._get_ai_analysis(symbol)
            technical_signals = await self._get_technical_signals(symbol, timeframe)
            news_sentiment = await self._get_news_sentiment(symbol)

            # Apply strategy logic
            for strategy_name, config in self.strategies.items():
                signal = self._combine_signals(
                    ml_prediction, ai_analysis, technical_signals, news_sentiment,
                    config, current_price, symbol, timeframe, strategy_name
                )
                
                if signal and signal.confidence >= config["min_confidence"]:
                    # Store signal
                    self._store_signal(signal)
                    logger.info(f"✅ Generated {strategy_name} signal: {signal.signal_type} with {signal.confidence:.2f} confidence")
                    return signal

            return None

        except Exception as e:
            logger.error(f"❌ Error generating integrated signal: {e}")
            return None

    def _combine_signals(self, ml_pred, ai_analysis, tech_signals, sentiment, 
                        config, price, symbol, timeframe, strategy_name) -> Optional[IntegratedSignal]:
        """Combine signals using weighted scoring"""
        try:
            # Calculate weighted scores
            ml_score = self._score_ml_prediction(ml_pred) * config["ml_weight"]
            ai_score = self._score_ai_analysis(ai_analysis) * config["ai_weight"]
            tech_score = self._score_technical_signals(tech_signals) * config["technical_weight"]
            sentiment_score = self._score_sentiment(sentiment) * config["sentiment_weight"]

            # Combined score (-1 to +1)
            total_score = ml_score + ai_score + tech_score + sentiment_score
            confidence = abs(total_score)

            # Determine signal type
            if total_score > 0.1:
                signal_type = "BUY"
            elif total_score < -0.1:
                signal_type = "SELL"
            else:
                signal_type = "HOLD"

            if signal_type == "HOLD":
                return None

            # Calculate risk management levels
            stop_loss, take_profit = self._calculate_risk_levels(
                price, signal_type, tech_signals
            )

            return IntegratedSignal(
                timestamp=datetime.now(),
                symbol=symbol,
                signal_type=signal_type,
                confidence=confidence,
                entry_price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                timeframe=timeframe,
                ml_prediction=ml_pred,
                ai_analysis=ai_analysis,
                technical_signals=tech_signals,
                news_sentiment=sentiment,
                strategy_name=strategy_name
            )

        except Exception as e:
            logger.error(f"❌ Error combining signals: {e}")
            return None

    def _score_ml_prediction(self, ml_pred) -> float:
        """Score ML prediction (-1 to +1)"""
        if not ml_pred:
            return 0.0
        
        try:
            # Extract confidence and direction from ML prediction
            confidence = ml_pred.get('confidence', 0.5)
            prediction = ml_pred.get('prediction', 'HOLD')
            
            if prediction == 'BUY':
                return confidence
            elif prediction == 'SELL':
                return -confidence
            return 0.0
        except:
            return 0.0

    def _score_ai_analysis(self, ai_analysis) -> float:
        """Score AI analysis (-1 to +1)"""
        if not ai_analysis:
            return 0.0
        
        try:
            # Extract sentiment and strength from AI analysis
            sentiment = ai_analysis.get('sentiment', 'NEUTRAL')
            strength = ai_analysis.get('confidence', 0.5)
            
            if sentiment in ['BULLISH', 'STRONG_BUY']:
                return strength
            elif sentiment in ['BEARISH', 'STRONG_SELL']:
                return -strength
            return 0.0
        except:
            return 0.0

    def _score_technical_signals(self, tech_signals) -> float:
        """Score technical analysis (-1 to +1)"""
        if not tech_signals:
            return 0.0
        
        try:
            # Combine multiple technical indicators
            rsi_score = self._score_rsi(tech_signals.get('rsi'))
            macd_score = self._score_macd(tech_signals.get('macd'))
            bb_score = self._score_bollinger_bands(tech_signals.get('bollinger'))
            
            return (rsi_score + macd_score + bb_score) / 3
        except:
            return 0.0

    def _score_sentiment(self, sentiment) -> float:
        """Score news sentiment (-1 to +1)"""
        if not sentiment:
            return 0.0
        
        try:
            score = sentiment.get('compound', 0.0)
            return max(-1.0, min(1.0, score))
        except:
            return 0.0

    def _calculate_risk_levels(self, price: float, signal_type: str, tech_signals: Dict) -> Tuple[float, float]:
        """Calculate stop loss and take profit levels"""
        try:
            # Base risk percentage
            risk_percent = 0.02  # 2%
            reward_percent = 0.04  # 4% (2:1 risk-reward)

            # Adjust based on volatility
            if tech_signals and 'atr' in tech_signals:
                atr = tech_signals['atr']
                risk_percent = max(0.01, min(0.03, atr / price * 2))
                reward_percent = risk_percent * 2

            if signal_type == "BUY":
                stop_loss = price * (1 - risk_percent)
                take_profit = price * (1 + reward_percent)
            else:  # SELL
                stop_loss = price * (1 + risk_percent)
                take_profit = price * (1 - reward_percent)

            return round(stop_loss, 2), round(take_profit, 2)

        except Exception as e:
            logger.error(f"❌ Error calculating risk levels: {e}")
            # Fallback to simple percentage
            if signal_type == "BUY":
                return round(price * 0.98, 2), round(price * 1.04, 2)
            else:
                return round(price * 1.02, 2), round(price * 0.96, 2)

    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price using GoldAPI via data pipeline"""
        try:
            # Primary: Use data pipeline GoldAPI for consistency
            if DATA_PIPELINE_AVAILABLE:
                from data_pipeline_core import DataType
                data = await data_pipeline.get_unified_data(symbol, DataType.PRICE)
                if data and 'price' in data:
                    price = float(data['price'])
                    logger.info(f"📡 Current {symbol} price from GoldAPI: ${price:.2f}")
                    return price
            
            # Secondary: Direct price storage manager (already uses GoldAPI)
            from price_storage_manager import get_current_gold_price
            price = get_current_gold_price()
            if price:
                logger.info(f"📋 Current {symbol} price from storage: ${price:.2f}")
                return price
                
            # Fallback
            logger.warning("⚠️ Using fallback price")
            return 3400.0
            
        except Exception as e:
            logger.error(f"❌ Error getting current price: {e}")
            return 3400.0

    async def _get_ml_prediction(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """Get ML prediction using GoldAPI-based dual ML system"""
        try:
            if self.ml_system:
                # Use the new GoldAPI-based dual predictions
                if hasattr(self.ml_system, 'get_dual_predictions'):
                    predictions = await self.ml_system.get_dual_predictions(symbol)
                    if predictions and predictions.get('success'):
                        # Extract the best prediction from the engines
                        engines = predictions.get('engines', [])
                        if engines:
                            # Use the first engine's prediction (Enhanced ML Engine)
                            engine = engines[0]
                            preds = engine.get('predictions', [])
                            if preds:
                                # Find prediction matching the timeframe or use first one
                                best_pred = None
                                for pred in preds:
                                    if pred.get('timeframe') == timeframe:
                                        best_pred = pred
                                        break
                                if not best_pred:
                                    best_pred = preds[0]
                                
                                return {
                                    'prediction': best_pred.get('direction', 'BUY'),
                                    'confidence': best_pred.get('confidence', 0.65),
                                    'predicted_price': best_pred.get('predicted_price', 0),
                                    'current_price': predictions.get('current_price', 0),
                                    'change_percent': best_pred.get('change_percent', 0),
                                    'timeframe': best_pred.get('timeframe', timeframe),
                                    'source': 'goldapi_dual_ml',
                                    'engine_count': len(engines)
                                }
                
                # Fallback methods
                elif hasattr(self.ml_system, 'get_predictions'):
                    predictions = await self.ml_system.get_predictions(symbol, timeframe)
                elif hasattr(self.ml_system, 'get_prediction'):
                    predictions = await self.ml_system.get_prediction(symbol, timeframe)
                elif hasattr(self.ml_system, 'generate_predictions'):
                    predictions = await self.ml_system.generate_predictions(symbol, timeframe)
                else:
                    # Fallback: create mock prediction with current GoldAPI price
                    current_price = await self._get_current_price(symbol)
                    predictions = {
                        'prediction': 'BUY',
                        'confidence': 0.65,
                        'predicted_price': current_price * 1.01,
                        'current_price': current_price,
                        'source': 'goldapi_fallback'
                    }
                return predictions
                
        except Exception as e:
            logger.error(f"❌ Error getting ML prediction: {e}")
        return None

    async def _get_ai_analysis(self, symbol: str) -> Optional[Dict]:
        """Get AI analysis"""
        try:
            if AI_ANALYSIS_AVAILABLE:
                # Run synchronous function in executor since it's not async
                import asyncio
                loop = asyncio.get_event_loop()
                analysis = await loop.run_in_executor(None, get_ai_analysis_sync, symbol)
                return analysis
        except Exception as e:
            logger.error(f"❌ Error getting AI analysis: {e}")
        return None

    async def _get_technical_signals(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """Get technical analysis signals"""
        try:
            if self.signal_generator:
                # Try different method names based on what's available
                if hasattr(self.signal_generator, 'force_generate_signal'):
                    # For testing, use force generation to bypass time intervals
                    signal = self.signal_generator.force_generate_signal()
                    if signal:
                        return {
                            'signal_type': signal.get('signal_type', 'NONE'),
                            'confidence': signal.get('confidence', 0.5),
                            'entry_price': signal.get('entry_price', 0),
                            'target_price': signal.get('target_price', 0),
                            'stop_loss': signal.get('stop_loss', 0),
                            'analysis_summary': signal.get('analysis_summary', '')
                        }
                elif hasattr(self.signal_generator, 'generate_enhanced_signal'):
                    signal = self.signal_generator.generate_enhanced_signal()
                    if signal:
                        return {
                            'signal_type': signal.get('signal_type', 'NONE'),
                            'confidence': signal.get('confidence', 0.5),
                            'entry_price': signal.get('entry_price', 0),
                            'target_price': signal.get('target_price', 0),
                            'stop_loss': signal.get('stop_loss', 0),
                            'analysis_summary': signal.get('analysis_summary', '')
                        }
                elif hasattr(self.signal_generator, 'generate_signals'):
                    signals = await self.signal_generator.generate_signals()
                elif hasattr(self.signal_generator, 'get_signals'):
                    signals = await self.signal_generator.get_signals()
                elif hasattr(self.signal_generator, 'generate_ai_signals'):
                    signals = await self.signal_generator.generate_ai_signals()
                else:
                    # Fallback: create mock technical signals
                    signals = {
                        'rsi': 55.0,
                        'macd': {'macd': 0.5, 'signal': 0.3, 'histogram': 0.2},
                        'bollinger': {'upper': 2060, 'lower': 1980, 'price': 2020},
                        'atr': 15.0,
                        'signal': 'BUY',
                        'confidence': 0.6
                    }
                return signals
        except Exception as e:
            logger.error(f"❌ Error getting technical signals: {e}")
        return None

    async def _get_news_sentiment(self, symbol: str) -> Optional[Dict]:
        """Get news sentiment analysis"""
        try:
            from news_aggregator import get_latest_news
            news = get_latest_news()
            if news:
                # Simple sentiment calculation
                return {"compound": 0.0, "source": "news"}
        except Exception as e:
            logger.error(f"❌ Error getting news sentiment: {e}")
        return None

    def _store_signal(self, signal: IntegratedSignal):
        """Store signal in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO integrated_signals 
                (timestamp, symbol, signal_type, confidence, entry_price, stop_loss, 
                 take_profit, timeframe, ml_prediction, ai_analysis, technical_signals, 
                 news_sentiment, position_size, risk_reward_ratio, strategy_name, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal.timestamp.isoformat(),
                signal.symbol,
                signal.signal_type,
                signal.confidence,
                signal.entry_price,
                signal.stop_loss,
                signal.take_profit,
                signal.timeframe,
                json.dumps(signal.ml_prediction) if signal.ml_prediction else None,
                json.dumps(signal.ai_analysis) if signal.ai_analysis else None,
                json.dumps(signal.technical_signals) if signal.technical_signals else None,
                json.dumps(signal.news_sentiment) if signal.news_sentiment else None,
                signal.position_size,
                signal.risk_reward_ratio,
                signal.strategy_name,
                signal.notes
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Error storing signal: {e}")

    def run_strategy_backtest(self, strategy_name: str, timeframe: str = "1h", 
                            days: int = 30) -> Optional[BacktestResult]:
        """Run backtest for integrated strategy"""
        try:
            if not BACKTESTING_AVAILABLE or not self.backtest_engine:
                logger.error("❌ Backtesting not available")
                return None

            # Setup backtest timeframe
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            # Create strategy function
            strategy_func = self._create_strategy_function(strategy_name)
            
            # Run backtest using the framework's built-in data generation
            result = self.backtest_engine.run_backtest(
                strategy_func=strategy_func,
                symbol="XAU",
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                strategy_id=strategy_name
            )

            # Store results
            if result:
                self._store_backtest_result(result)
                logger.info(f"✅ Backtest completed for {strategy_name}: {result.total_return_percent:.2f}% return")
            
            return result

        except Exception as e:
            logger.error(f"❌ Error running strategy backtest: {e}")
            return None

    def _create_strategy_function(self, strategy_name: str):
        """Create strategy function for backtesting"""
        config = self.strategies.get(strategy_name, self.strategies["ml_momentum"])
        
        def integrated_strategy(data, current_idx):
            """Integrated strategy logic for backtesting"""
            try:
                # Ensure we have enough data
                if current_idx < 50 or current_idx >= len(data):
                    return None

                # Get current bar - access the row properly
                if hasattr(data, 'iloc'):
                    current_bar = data.iloc[current_idx]
                else:
                    current_bar = data[current_idx]
                
                # Extract price from OHLCV object if needed
                if hasattr(current_bar, 'close'):
                    price = current_bar.close
                elif isinstance(current_bar, dict):
                    price = current_bar.get('close', current_bar.get('price', 3400.0))
                elif hasattr(current_bar, '__getitem__'):
                    try:
                        price = current_bar['close']
                    except (KeyError, TypeError):
                        price = 3400.0  # Fallback
                else:
                    price = 3400.0  # Fallback

                # Simulate signal generation
                mock_signal = self._generate_mock_signal(data, current_idx, config)
                
                if mock_signal['confidence'] >= config['min_confidence']:
                    if mock_signal['signal'] == 'BUY':
                        return {
                            'action': 'BUY',
                            'quantity': 1.0,
                            'stop_loss': price * 0.98,
                            'take_profit': price * 1.04
                        }
                    elif mock_signal['signal'] == 'SELL':
                        return {
                            'action': 'SELL',
                            'quantity': 1.0,
                            'stop_loss': price * 1.02,
                            'take_profit': price * 0.96
                        }

                return None

            except Exception as e:
                logger.error(f"❌ Strategy error: {e}")
                return None

        return integrated_strategy

    def _generate_mock_signal(self, data, current_idx, config):
        """Generate mock signal for backtesting"""
        try:
            # Simple technical analysis for mock signal
            current_bar = data.iloc[current_idx]
            
            # More aggressive RSI signal (expanded ranges)
            rsi = current_bar.get('rsi', 50)
            if rsi < 40:
                rsi_signal = 0.8  # Strong buy signal
            elif rsi > 60:
                rsi_signal = -0.8  # Strong sell signal
            else:
                rsi_signal = (50 - rsi) * 0.02  # Gradual signal based on RSI deviation
            
            # Moving average signal (more sensitive)
            sma_20 = current_bar.get('sma_20', current_bar['close'])
            sma_50 = current_bar.get('sma_50', current_bar['close'])
            ma_signal = 0.6 if sma_20 > sma_50 else -0.6
            
            # Price momentum signal (new addition)
            if current_idx >= 5:
                price_5_ago = data.iloc[current_idx - 5]['close']
                momentum = (current_bar['close'] - price_5_ago) / price_5_ago
                momentum_signal = max(-0.8, min(0.8, momentum * 20))  # Scale momentum
            else:
                momentum_signal = 0
            
            # Volume signal (if available)
            volume_signal = 0
            if 'volume' in current_bar and current_idx >= 10:
                avg_volume = data.iloc[current_idx-10:current_idx]['volume'].mean()
                if current_bar['volume'] > avg_volume * 1.5:
                    volume_signal = 0.3  # Volume confirmation
            
            # Combine signals with weights
            combined_score = (
                rsi_signal * 0.3 + 
                ma_signal * 0.25 + 
                momentum_signal * 0.35 + 
                volume_signal * 0.1
            )
            
            # More aggressive confidence calculation
            confidence = min(0.9, abs(combined_score) * 1.2)  # Higher confidence
            
            # Lower thresholds for signal generation
            signal = 'BUY' if combined_score > 0.05 else ('SELL' if combined_score < -0.05 else 'HOLD')
            
            return {
                'signal': signal,
                'confidence': confidence
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating mock signal: {e}")
            return {'signal': 'HOLD', 'confidence': 0.0}

    def _store_backtest_result(self, result: BacktestResult):
        """Store backtest result in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO backtest_results 
                (strategy_name, timeframe, start_date, end_date, initial_capital, 
                 final_capital, total_return, total_return_percent, sharpe_ratio, 
                 max_drawdown, total_trades, win_rate, performance_data, parameters)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.strategy_id,
                result.timeframe,
                result.start_date.isoformat(),
                result.end_date.isoformat(),
                result.initial_capital,
                result.final_capital,
                result.total_return,
                result.total_return_percent,
                result.performance_metrics.get('sharpe_ratio', 0),
                result.performance_metrics.get('max_drawdown', 0),
                len(result.trades),
                result.trade_analysis.get('win_rate', 0),
                json.dumps(result.performance_metrics),
                json.dumps(self.strategies.get(result.strategy_id, {}))
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Error storing backtest result: {e}")

    def get_strategy_performance(self, strategy_name: str = None) -> Dict:
        """Get strategy performance summary"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if strategy_name:
                cursor.execute('''
                    SELECT * FROM backtest_results 
                    WHERE strategy_name = ? 
                    ORDER BY created_at DESC LIMIT 10
                ''', (strategy_name,))
            else:
                cursor.execute('''
                    SELECT * FROM backtest_results 
                    ORDER BY created_at DESC LIMIT 20
                ''')
            
            results = cursor.fetchall()
            conn.close()
            
            if not results:
                return {"error": "No backtest results found"}
            
            # Process results
            performance_data = []
            for row in results:
                performance_data.append({
                    "strategy_name": row[1],
                    "timeframe": row[2],
                    "total_return_percent": row[7],
                    "sharpe_ratio": row[8],
                    "max_drawdown": row[9],
                    "total_trades": row[10],
                    "win_rate": row[11],
                    "created_at": row[14]
                })
            
            return {
                "results": performance_data,
                "summary": self._calculate_performance_summary(performance_data)
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting strategy performance: {e}")
            return {"error": str(e)}

    def _calculate_performance_summary(self, data: List[Dict]) -> Dict:
        """Calculate performance summary statistics"""
        try:
            if not data:
                return {}
            
            returns = [item["total_return_percent"] for item in data]
            sharpe_ratios = [item["sharpe_ratio"] for item in data]
            
            return {
                "avg_return": np.mean(returns),
                "best_return": max(returns),
                "worst_return": min(returns),
                "avg_sharpe": np.mean(sharpe_ratios),
                "total_backtests": len(data),
                "win_rate": len([r for r in returns if r > 0]) / len(returns) * 100
            }
            
        except Exception as e:
            logger.error(f"❌ Error calculating performance summary: {e}")
            return {}

    def optimize_strategy(self, strategy_name: str, timeframe: str = "1h") -> Dict:
        """Optimize strategy parameters using genetic algorithm"""
        try:
            if not BACKTESTING_AVAILABLE or not self.backtest_engine:
                return {"error": "Backtesting not available"}

            # Setup optimization timeframe
            end_date = datetime.now()
            start_date = end_date - timedelta(days=60)

            # Define parameters to optimize
            from advanced_backtesting_framework import StrategyParameters
            parameters = [
                StrategyParameters("ml_weight", 0.1, 0.6, 0.05, 0.4, "float"),
                StrategyParameters("ai_weight", 0.1, 0.6, 0.05, 0.3, "float"),
                StrategyParameters("min_confidence", 0.5, 0.9, 0.05, 0.65, "float")
            ]

            # Run optimization
            optimizer = StrategyOptimizer()
            strategy_func = self._create_optimizable_strategy_function()
            
            result = optimizer.optimize_strategy(
                strategy_func=strategy_func,
                optimization_params=parameters,
                backtest_engine=self.backtest_engine,
                symbol="XAU",
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                optimization_metric="sharpe_ratio"
            )

            logger.info(f"✅ Strategy optimization completed for {strategy_name}")
            return {
                "success": True,
                "best_parameters": result.best_parameters,
                "best_fitness": result.best_fitness,
                "optimization_data": result.convergence_data
            }

        except Exception as e:
            logger.error(f"❌ Error optimizing strategy: {e}")
            return {"error": str(e)}

    def _create_optimizable_strategy_function(self):
        """Create strategy function that accepts parameters for optimization"""
        def optimizable_strategy(data, current_idx, **params):
            try:
                if current_idx < 50:
                    return None

                current_bar = data.iloc[current_idx]
                price = current_bar['close']

                # Use optimized parameters
                ml_weight = params.get('ml_weight', 0.4)
                ai_weight = params.get('ai_weight', 0.3)
                min_confidence = params.get('min_confidence', 0.65)

                # Generate signal with optimized weights
                mock_signal = self._generate_parameterized_signal(
                    data, current_idx, ml_weight, ai_weight
                )
                
                if mock_signal['confidence'] >= min_confidence:
                    if mock_signal['signal'] == 'BUY':
                        return {
                            'action': 'BUY',
                            'quantity': 1.0,
                            'stop_loss': price * 0.98,
                            'take_profit': price * 1.04
                        }
                    elif mock_signal['signal'] == 'SELL':
                        return {
                            'action': 'SELL',
                            'quantity': 1.0,
                            'stop_loss': price * 1.02,
                            'take_profit': price * 0.96
                        }

                return None

            except Exception as e:
                return None

        return optimizable_strategy

    def _generate_parameterized_signal(self, data, current_idx, ml_weight, ai_weight):
        """Generate signal with specific parameter weights"""
        try:
            current_bar = data.iloc[current_idx]
            
            # Enhanced ML signal (RSI-based)
            rsi = current_bar.get('rsi', 50)
            if rsi < 35:
                ml_signal = 0.8
            elif rsi > 65:
                ml_signal = -0.8
            else:
                ml_signal = (50 - rsi) * 0.02
            
            # Enhanced AI signal (MACD + momentum)
            macd = current_bar.get('macd', 0)
            if current_idx >= 3:
                price_change = (current_bar['close'] - data.iloc[current_idx-3]['close']) / data.iloc[current_idx-3]['close']
                ai_signal = (1 if macd > 0 else -1) * 0.6 + price_change * 10
                ai_signal = max(-0.8, min(0.8, ai_signal))
            else:
                ai_signal = 1 if macd > 0 else -1
            
            # Combine with weights (normalize to prevent over-weighting)
            total_weight = ml_weight + ai_weight
            if total_weight > 0:
                combined_score = (ml_signal * ml_weight + ai_signal * ai_weight) / total_weight
            else:
                combined_score = 0
                
            confidence = min(0.9, abs(combined_score) * 1.1)
            
            signal = 'BUY' if combined_score > 0.05 else ('SELL' if combined_score < -0.05 else 'HOLD')
            
            return {
                'signal': signal,
                'confidence': confidence
            }
            
        except Exception as e:
            return {'signal': 'HOLD', 'confidence': 0.0}

    def get_recent_signals(self, limit: int = 10) -> List[Dict]:
        """Get recent integrated signals"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM integrated_signals 
                ORDER BY timestamp DESC LIMIT ?
            ''', (limit,))
            
            rows = cursor.fetchall()
            conn.close()
            
            signals = []
            for row in rows:
                signals.append({
                    "id": row[0],
                    "timestamp": row[1],
                    "symbol": row[2],
                    "signal_type": row[3],
                    "confidence": row[4],
                    "entry_price": row[5],
                    "stop_loss": row[6],
                    "take_profit": row[7],
                    "timeframe": row[8],
                    "strategy_name": row[15],
                    "notes": row[16]
                })
            
            return signals
            
        except Exception as e:
            logger.error(f"❌ Error getting recent signals: {e}")
            return []
    
    async def force_generate_signal(self, symbol: str, timeframe: str) -> Optional[IntegratedSignal]:
        """Force generate a signal for testing purposes"""
        if self.signal_generator and hasattr(self.signal_generator, 'force_generate_signal'):
            # Force signal generation to bypass time intervals
            signal_data = self.signal_generator.force_generate_signal()
            if signal_data:
                return IntegratedSignal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    signal_type=signal_data.get('signal_type', 'BUY'),
                    confidence=signal_data.get('confidence', 0.7),
                    entry_price=signal_data.get('entry_price', 0),
                    stop_loss=signal_data.get('stop_loss', 0),
                    take_profit=signal_data.get('target_price', 0),
                    timeframe=timeframe,
                    strategy_name="forced_test"
                )
        
        # Fallback: generate a test signal with current GoldAPI price
        try:
            current_price = await self._get_current_price(symbol)  # Uses GoldAPI via data pipeline
            
            if current_price:
                logger.info(f"📡 Force signal using GoldAPI price: ${current_price:.2f}")
                return IntegratedSignal(
                    timestamp=datetime.now(),
                    symbol=symbol,
                    signal_type="BUY",
                    confidence=0.6,
                    entry_price=current_price,
                    take_profit=current_price * 1.02,  # 2% target
                    stop_loss=current_price * 0.99,    # 1% stop loss
                    timeframe=timeframe,
                    strategy_name="goldapi_fallback_test"
                )
        except Exception as e:
            logger.error(f"Error in force signal generation: {e}")
        
        return None

# Global instance
integrated_strategy_engine = IntegratedStrategyEngine()

# Helper functions for scoring
def _score_rsi(rsi_value):
    """Score RSI indicator"""
    if not rsi_value:
        return 0.0
    if rsi_value < 30:
        return 0.7  # Oversold - bullish
    elif rsi_value > 70:
        return -0.7  # Overbought - bearish
    return 0.0

def _score_macd(macd_data):
    """Score MACD indicator"""
    if not macd_data:
        return 0.0
    try:
        macd_line = macd_data.get('macd', 0)
        signal_line = macd_data.get('signal', 0)
        histogram = macd_data.get('histogram', 0)
        
        if macd_line > signal_line and histogram > 0:
            return 0.6  # Bullish
        elif macd_line < signal_line and histogram < 0:
            return -0.6  # Bearish
        return 0.0
    except:
        return 0.0

def _score_bollinger_bands(bb_data):
    """Score Bollinger Bands position"""
    if not bb_data:
        return 0.0
    try:
        price = bb_data.get('price', 0)
        upper = bb_data.get('upper', 0)
        lower = bb_data.get('lower', 0)
        
        if price < lower:
            return 0.5  # Oversold
        elif price > upper:
            return -0.5  # Overbought
        return 0.0
    except:
        return 0.0

if __name__ == "__main__":
    import asyncio
    
    async def test_integrated_engine():
        """Test the integrated strategy engine"""
        print("🧪 Testing Integrated Strategy Engine...")
        
        engine = IntegratedStrategyEngine()
        
        # Test signal generation
        signal = await engine.generate_integrated_signal("XAU", "1h")
        if signal:
            print(f"✅ Generated signal: {signal.signal_type} with {signal.confidence:.2f} confidence")
        
        # Test backtesting
        result = engine.run_strategy_backtest("ml_momentum", "1h", 30)
        if result:
            print(f"✅ Backtest completed: {result.total_return_percent:.2f}% return")
        
        # Test optimization
        opt_result = engine.optimize_strategy("ml_momentum", "1h")
        if opt_result.get("success"):
            print(f"✅ Optimization completed: {opt_result['best_fitness']:.3f} fitness")
        
        print("✅ Integrated Strategy Engine test completed!")
    
    asyncio.run(test_integrated_engine())
