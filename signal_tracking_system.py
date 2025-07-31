#!/usr/bin/env python3
"""
Advanced Signal Tracking and Learning System
Monitors live P&L, automatically marks wins/losses, and learns from outcomes
"""
import sqlite3
import json
import logging
import asyncio
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from price_storage_manager import get_current_gold_price
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SignalPerformance:
    """Signal performance metrics"""
    signal_id: int
    entry_price: float
    current_price: float
    current_pnl: float
    current_pnl_pct: float
    max_favorable: float
    max_adverse: float
    duration_minutes: int
    status: str
    technical_factors: Dict[str, Any]

class SignalTrackingSystem:
    """Advanced signal tracking and learning system"""
    
    def __init__(self):
        self.db_path = 'goldgpt_enhanced_signals.db'
        self.learning_model_path = 'signal_learning_model.pkl'
        self.scaler_path = 'signal_scaler.pkl'
        self.monitoring_active = False
        self.monitoring_thread = None
        self.check_interval = 30  # seconds
        self.learning_model = None
        self.feature_scaler = None
        self._initialize_learning_model()
        
    def _initialize_learning_model(self):
        """Initialize or load the ML learning model"""
        try:
            if os.path.exists(self.learning_model_path) and os.path.exists(self.scaler_path):
                with open(self.learning_model_path, 'rb') as f:
                    self.learning_model = pickle.load(f)
                with open(self.scaler_path, 'rb') as f:
                    self.feature_scaler = pickle.load(f)
                logger.info("✅ Loaded existing learning model")
            else:
                # Create new model
                self.learning_model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42,
                    class_weight='balanced'
                )
                self.feature_scaler = StandardScaler()
                logger.info("📊 Created new learning model")
        except Exception as e:
            logger.error(f"Error initializing learning model: {e}")
            self.learning_model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.feature_scaler = StandardScaler()

    def start_monitoring(self):
        """Start the signal monitoring service"""
        if self.monitoring_active:
            logger.info("Signal monitoring already active")
            return
            
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("🚀 Signal tracking system started")

    def stop_monitoring(self):
        """Stop the signal monitoring service"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("⏹️ Signal tracking system stopped")

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                self._check_active_signals()
                self._update_daily_performance()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Wait longer on error

    def _check_active_signals(self):
        """Check all active signals for TP/SL hits"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get all active signals
            cursor.execute("""
                SELECT id, signal_type, entry_price, target_price, stop_loss, 
                       timestamp, factors_json, max_favorable_price, max_adverse_price,
                       confidence, risk_reward_ratio
                FROM enhanced_signals 
                WHERE status = 'active'
            """)
            
            active_signals = cursor.fetchall()
            current_price = get_current_gold_price()
            
            if not current_price or current_price <= 0:
                return
                
            for signal in active_signals:
                signal_id, signal_type, entry_price, target_price, stop_loss, \
                timestamp, factors_json, max_fav, max_adv, confidence, rr_ratio = signal
                
                # Update max favorable/adverse prices
                if max_fav is None:
                    max_fav = current_price
                if max_adv is None:
                    max_adv = current_price
                    
                if signal_type == 'buy':
                    max_fav = max(max_fav, current_price)
                    max_adv = min(max_adv, current_price)
                    
                    # Check for TP/SL hit
                    if current_price >= target_price:
                        self._close_signal(signal_id, current_price, 'TP_HIT', 'WIN')
                    elif current_price <= stop_loss:
                        self._close_signal(signal_id, current_price, 'SL_HIT', 'LOSS')
                    else:
                        self._update_signal_tracking(signal_id, current_price, max_fav, max_adv)
                        
                else:  # sell signal
                    max_fav = min(max_fav, current_price)
                    max_adv = max(max_adv, current_price)
                    
                    # Check for TP/SL hit
                    if current_price <= target_price:
                        self._close_signal(signal_id, current_price, 'TP_HIT', 'WIN')
                    elif current_price >= stop_loss:
                        self._close_signal(signal_id, current_price, 'SL_HIT', 'LOSS')
                    else:
                        self._update_signal_tracking(signal_id, current_price, max_fav, max_adv)
                        
        except Exception as e:
            logger.error(f"Error checking active signals: {e}")
        finally:
            conn.close()

    def _update_signal_tracking(self, signal_id: int, current_price: float, max_fav: float, max_adv: float):
        """Update signal tracking data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE enhanced_signals 
                SET current_price = ?, max_favorable_price = ?, max_adverse_price = ?
                WHERE id = ?
            """, (current_price, max_fav, max_adv, signal_id))
            conn.commit()
        except Exception as e:
            logger.error(f"Error updating signal tracking: {e}")
        finally:
            conn.close()

    def _close_signal(self, signal_id: int, exit_price: float, exit_reason: str, outcome: str):
        """Close a signal and analyze performance"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get signal details
            cursor.execute("""
                SELECT signal_type, entry_price, target_price, stop_loss, timestamp, 
                       factors_json, confidence, risk_reward_ratio
                FROM enhanced_signals WHERE id = ?
            """, (signal_id,))
            
            signal_data = cursor.fetchone()
            if not signal_data:
                return
                
            signal_type, entry_price, target_price, stop_loss, timestamp, \
            factors_json, confidence, rr_ratio = signal_data
            
            # Calculate P&L
            if signal_type == 'buy':
                pnl = exit_price - entry_price
            else:
                pnl = entry_price - exit_price
                
            pnl_pct = (pnl / entry_price) * 100
            
            # Calculate duration
            entry_time = datetime.fromisoformat(timestamp)
            duration_minutes = int((datetime.now() - entry_time).total_seconds() / 60)
            
            # Update signal record
            cursor.execute("""
                UPDATE enhanced_signals 
                SET status = 'closed', exit_price = ?, exit_reason = ?, 
                    profit_loss = ?, profit_loss_pct = ?, exit_timestamp = ?,
                    duration_minutes = ?
                WHERE id = ?
            """, (exit_price, exit_reason, pnl, pnl_pct, datetime.now().isoformat(), 
                  duration_minutes, signal_id))
            
            conn.commit()
            
            # Log the result
            logger.info(f"📊 Signal {signal_id} closed: {outcome} | P&L: ${pnl:.2f} ({pnl_pct:.2f}%) | Duration: {duration_minutes}min")
            
            # Learn from the outcome
            self._learn_from_signal_outcome(signal_id, outcome, pnl, factors_json, confidence, rr_ratio)
            
        except Exception as e:
            logger.error(f"Error closing signal: {e}")
        finally:
            conn.close()

    def _learn_from_signal_outcome(self, signal_id: int, outcome: str, pnl: float, 
                                 factors_json: str, confidence: float, rr_ratio: float):
        """Learn from signal outcome and update strategy"""
        try:
            factors = json.loads(factors_json) if factors_json else {}
            
            # Extract features for learning
            features = self._extract_learning_features(factors, confidence, rr_ratio)
            
            # Store learning data
            self._store_learning_metrics(signal_id, outcome, pnl, features)
            
            # Update learning model if we have enough data
            self._update_learning_model()
            
        except Exception as e:
            logger.error(f"Error learning from signal outcome: {e}")

    def _extract_learning_features(self, factors: Dict[str, Any], confidence: float, rr_ratio: float) -> List[float]:
        """Extract numerical features from signal factors for ML learning"""
        features = []
        
        try:
            # Market condition features
            technical = factors.get('technical', {})
            features.extend([
                technical.get('rsi', 50),
                technical.get('macd_signal', 0),
                technical.get('bb_position', 0.5),
                technical.get('volume_ratio', 1.0),
                confidence,
                rr_ratio
            ])
            
            # Sentiment features
            sentiment = factors.get('sentiment', {})
            features.extend([
                sentiment.get('news_sentiment', 0),
                sentiment.get('market_fear_greed', 50)
            ])
            
            # Volatility features
            features.append(factors.get('volatility', {}).get('current_volatility', 0.02))
            
            # Trend features
            trend = factors.get('trend', {})
            features.extend([
                trend.get('short_trend', 0),
                trend.get('medium_trend', 0),
                trend.get('long_trend', 0)
            ])
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            # Return default features if extraction fails
            features = [50, 0, 0.5, 1.0, confidence, rr_ratio, 0, 50, 0.02, 0, 0, 0]
            
        return features

    def _store_learning_metrics(self, signal_id: int, outcome: str, pnl: float, features: List[float]):
        """Store learning metrics in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Store each feature with its contribution score
            feature_names = [
                'rsi', 'macd_signal', 'bb_position', 'volume_ratio', 'confidence', 'rr_ratio',
                'news_sentiment', 'market_fear_greed', 'volatility', 'short_trend', 'medium_trend', 'long_trend'
            ]
            
            success_weight = 1.0 if outcome == 'WIN' else -1.0
            
            for i, (feature_name, feature_value) in enumerate(zip(feature_names, features)):
                # Calculate contribution score based on outcome and feature value
                contribution_score = self._calculate_feature_contribution(feature_name, feature_value, outcome, pnl)
                
                cursor.execute("""
                    INSERT INTO signal_learning_metrics 
                    (signal_id, factor_type, factor_value, contribution_score, success_weight, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (signal_id, feature_name, feature_value, contribution_score, success_weight, datetime.now().isoformat()))
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Error storing learning metrics: {e}")
        finally:
            conn.close()

    def _calculate_feature_contribution(self, feature_name: str, feature_value: float, 
                                      outcome: str, pnl: float) -> float:
        """Calculate how much a feature contributed to the signal outcome"""
        try:
            # Base contribution score
            base_score = abs(pnl) / 100  # Normalize by typical gold price movement
            
            # Feature-specific adjustments
            if feature_name == 'confidence':
                # High confidence should correlate with success
                if outcome == 'WIN':
                    return base_score * (feature_value / 100)
                else:
                    return -base_score * (feature_value / 100)
                    
            elif feature_name in ['rsi']:
                # RSI extreme values (oversold/overbought) should indicate reversals
                rsi_extreme = min(abs(feature_value - 30), abs(feature_value - 70))
                contribution = base_score * (rsi_extreme / 20)
                return contribution if outcome == 'WIN' else -contribution
                
            elif feature_name == 'rr_ratio':
                # Higher risk-reward should be rewarded if successful
                contribution = base_score * min(feature_value / 2, 1.0)
                return contribution if outcome == 'WIN' else -contribution
                
            else:
                # Default contribution calculation
                return base_score if outcome == 'WIN' else -base_score
                
        except Exception as e:
            logger.error(f"Error calculating feature contribution: {e}")
            return 0.0

    def _update_learning_model(self):
        """Update the ML model with recent learning data"""
        try:
            # Only update if we have sufficient data
            if not self._has_sufficient_learning_data():
                return
                
            # Prepare training data
            X, y = self._prepare_training_data()
            
            if len(X) < 10:  # Need minimum samples
                return
                
            # Scale features
            X_scaled = self.feature_scaler.fit_transform(X)
            
            # Train model
            self.learning_model.fit(X_scaled, y)
            
            # Save updated model
            self._save_learning_model()
            
            logger.info(f"🧠 Updated learning model with {len(X)} samples")
            
        except Exception as e:
            logger.error(f"Error updating learning model: {e}")

    def _has_sufficient_learning_data(self) -> bool:
        """Check if we have enough data to update the learning model"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT COUNT(*) FROM enhanced_signals WHERE status = 'closed'")
            closed_signals = cursor.fetchone()[0]
            return closed_signals >= 10  # Minimum signals needed
        except Exception as e:
            logger.error(f"Error checking learning data: {e}")
            return False
        finally:
            conn.close()

    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from historical signals"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        X, y = [], []
        
        try:
            # Get closed signals with their outcomes
            cursor.execute("""
                SELECT factors_json, confidence, risk_reward_ratio, profit_loss
                FROM enhanced_signals 
                WHERE status = 'closed' AND factors_json IS NOT NULL
                ORDER BY exit_timestamp DESC
                LIMIT 100
            """)
            
            signals = cursor.fetchall()
            
            for factors_json, confidence, rr_ratio, pnl in signals:
                factors = json.loads(factors_json) if factors_json else {}
                features = self._extract_learning_features(factors, confidence, rr_ratio)
                
                # Binary classification: profit or loss
                outcome = 1 if pnl > 0 else 0
                
                X.append(features)
                y.append(outcome)
                
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
        finally:
            conn.close()
            
        return np.array(X), np.array(y)

    def _save_learning_model(self):
        """Save the trained model and scaler"""
        try:
            with open(self.learning_model_path, 'wb') as f:
                pickle.dump(self.learning_model, f)
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.feature_scaler, f)
        except Exception as e:
            logger.error(f"Error saving learning model: {e}")

    def predict_signal_success_probability(self, factors: Dict[str, Any], 
                                         confidence: float, rr_ratio: float) -> float:
        """Predict the probability of signal success"""
        try:
            if self.learning_model is None or self.feature_scaler is None:
                return 0.5  # Default neutral probability
                
            # Check if scaler is fitted
            if not hasattr(self.feature_scaler, 'scale_'):
                return 0.5  # Scaler not fitted yet
                
            features = self._extract_learning_features(factors, confidence, rr_ratio)
            features_scaled = self.feature_scaler.transform([features])
            
            # Get probability of success (class 1)
            probabilities = self.learning_model.predict_proba(features_scaled)
            return probabilities[0][1] if len(probabilities[0]) > 1 else 0.5
            
        except Exception as e:
            logger.error(f"Error predicting signal success: {e}")
            return 0.5

    def get_strategy_performance_insights(self) -> Dict[str, Any]:
        """Get insights about which strategies are working best"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        insights = {
            'total_signals': 0,
            'win_rate': 0.0,
            'avg_profit': 0.0,
            'best_factors': [],
            'worst_factors': [],
            'recommendations': []
        }
        
        try:
            # Basic performance metrics
            cursor.execute("""
                SELECT COUNT(*), AVG(profit_loss), AVG(CASE WHEN profit_loss > 0 THEN 1.0 ELSE 0.0 END)
                FROM enhanced_signals WHERE status = 'closed'
            """)
            result = cursor.fetchone()
            if result[0] > 0:
                insights['total_signals'] = result[0]
                insights['avg_profit'] = result[1]
                insights['win_rate'] = result[2] * 100
            
            # Factor analysis
            cursor.execute("""
                SELECT factor_type, AVG(contribution_score), COUNT(*)
                FROM signal_learning_metrics
                GROUP BY factor_type
                ORDER BY AVG(contribution_score) DESC
            """)
            
            factor_analysis = cursor.fetchall()
            insights['best_factors'] = [f[0] for f in factor_analysis[:3]]
            insights['worst_factors'] = [f[0] for f in factor_analysis[-3:]]
            
            # Generate recommendations
            insights['recommendations'] = self._generate_strategy_recommendations(insights)
            
        except Exception as e:
            logger.error(f"Error getting performance insights: {e}")
        finally:
            conn.close()
            
        return insights

    def _generate_strategy_recommendations(self, insights: Dict[str, Any]) -> List[str]:
        """Generate strategy recommendations based on performance"""
        recommendations = []
        
        win_rate = insights.get('win_rate', 0)
        avg_profit = insights.get('avg_profit', 0)
        
        if win_rate < 40:
            recommendations.append("Win rate is low - consider increasing signal confidence threshold")
        elif win_rate > 70:
            recommendations.append("Excellent win rate - consider slightly lowering confidence threshold to capture more opportunities")
            
        if avg_profit < 5:
            recommendations.append("Average profit is low - consider adjusting risk-reward ratios")
            
        best_factors = insights.get('best_factors', [])
        if 'confidence' in best_factors:
            recommendations.append("High confidence signals are performing well - maintain strict confidence criteria")
            
        return recommendations

    def _update_daily_performance(self):
        """Update daily performance statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            today = datetime.now().date().isoformat()
            
            # Get today's performance
            cursor.execute("""
                SELECT COUNT(*), AVG(profit_loss), 
                       AVG(CASE WHEN profit_loss > 0 THEN 1.0 ELSE 0.0 END),
                       MAX(profit_loss), MIN(profit_loss), AVG(duration_minutes)
                FROM enhanced_signals 
                WHERE status = 'closed' AND DATE(exit_timestamp) = ?
            """, (today,))
            
            result = cursor.fetchone()
            if result[0] > 0:
                total, avg_pnl, win_rate, best_profit, worst_loss, avg_duration = result
                
                # Update or insert daily performance
                cursor.execute("""
                    INSERT OR REPLACE INTO performance_history
                    (date, total_signals, successful_signals, success_rate, total_profit_loss,
                     avg_profit_loss, best_signal_profit, worst_signal_loss, avg_duration_minutes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (today, total, int(total * win_rate), win_rate * 100, 
                      total * avg_pnl, avg_pnl, best_profit or 0, worst_loss or 0, avg_duration or 0))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error updating daily performance: {e}")
        finally:
            conn.close()

# Global instance
signal_tracking_system = SignalTrackingSystem()
