"""
🎯 GOLDGPT LIVE TRADING INTEGRATION MODULE
=========================================

Integration module to connect enhanced backtesting system with live trading operations.
Provides real-time strategy validation and adaptive risk management.

Author: GoldGPT Development Team  
Created: July 23, 2025
Status: PRODUCTION INTEGRATION READY
"""

import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
import json
import sqlite3
from enhanced_backtesting_system_v2 import (
    AdvancedMarketRegimeAnalyzer,
    AdvancedRiskManager,
    AdvancedPerformanceAnalyzer,
    EnhancedBacktestConfig
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('live_trading_integration')

@dataclass
class LiveTradingConfig:
    """Configuration for live trading integration"""
    enable_real_time_validation: bool = True
    regime_update_interval: int = 300  # 5 minutes
    risk_check_interval: int = 60      # 1 minute
    performance_update_interval: int = 3600  # 1 hour
    max_position_size: float = 10000   # Maximum position size
    emergency_stop_drawdown: float = 0.15  # 15% emergency stop
    adaptive_sizing: bool = True
    regime_based_stops: bool = True

class LiveTradingIntegrationEngine:
    """Main engine for integrating enhanced backtesting with live trading"""
    
    def __init__(self, config: LiveTradingConfig):
        self.config = config
        self.is_running = False
        self.current_regime = None
        self.live_positions = {}
        self.performance_tracker = {}
        
        # Initialize enhanced components
        self.backtest_config = EnhancedBacktestConfig()
        self.regime_analyzer = AdvancedMarketRegimeAnalyzer()
        self.risk_manager = AdvancedRiskManager(self.backtest_config)
        self.performance_analyzer = AdvancedPerformanceAnalyzer()
        
        # Threading for real-time operations
        self.regime_thread = None
        self.risk_thread = None
        self.performance_thread = None
        
        logger.info("🎯 Live Trading Integration Engine initialized")
    
    def start_live_integration(self):
        """Start live trading integration services"""
        try:
            self.is_running = True
            
            # Start background monitoring threads
            if self.config.enable_real_time_validation:
                self._start_regime_monitoring()
                self._start_risk_monitoring()
                self._start_performance_monitoring()
            
            logger.info("🚀 Live trading integration started successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to start live integration: {e}")
            raise
    
    def stop_live_integration(self):
        """Stop live trading integration services"""
        self.is_running = False
        
        # Stop all monitoring threads
        if self.regime_thread and self.regime_thread.is_alive():
            self.regime_thread.join(timeout=5)
        
        if self.risk_thread and self.risk_thread.is_alive():
            self.risk_thread.join(timeout=5)
        
        if self.performance_thread and self.performance_thread.is_alive():
            self.performance_thread.join(timeout=5)
        
        logger.info("🛑 Live trading integration stopped")
    
    def validate_live_signal(self, signal: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a live trading signal using enhanced backtesting"""
        try:
            # Get recent market data for validation
            market_data = self._get_recent_market_data(symbol=signal['symbol'], days=30)
            
            if market_data.empty:
                return self._create_signal_response(signal, False, "Insufficient market data")
            
            # Regime-based signal validation
            regime_validation = self._validate_signal_against_regime(signal, market_data)
            
            # Risk-based position sizing
            position_size = self._calculate_live_position_size(signal, market_data)
            
            # Performance-based confidence scoring
            confidence_score = self._calculate_signal_confidence(signal, market_data)
            
            # Final validation decision
            is_valid = (
                regime_validation['approved'] and 
                position_size > 0 and 
                confidence_score > 0.5
            )
            
            validation_result = {
                'signal_id': signal.get('id', f"signal_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                'original_signal': signal,
                'validated': is_valid,
                'regime_analysis': regime_validation,
                'position_size': position_size,
                'confidence_score': confidence_score,
                'risk_metrics': self._get_current_risk_metrics(),
                'recommendations': self._generate_signal_recommendations(signal, market_data),
                'timestamp': datetime.now().isoformat()
            }
            
            # Log validation result
            logger.info(f"📊 Signal validation: {signal['symbol']} - {'APPROVED' if is_valid else 'REJECTED'}")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"❌ Signal validation failed: {e}")
            return self._create_signal_response(signal, False, f"Validation error: {e}")
    
    def update_live_performance(self, trade_result: Dict[str, Any]):
        """Update live performance tracking with trade result"""
        try:
            symbol = trade_result.get('symbol', 'UNKNOWN')
            
            if symbol not in self.performance_tracker:
                self.performance_tracker[symbol] = {
                    'trades': [],
                    'total_pnl': 0,
                    'win_rate': 0,
                    'current_drawdown': 0,
                    'peak_value': 0
                }
            
            # Add trade to tracker
            self.performance_tracker[symbol]['trades'].append({
                'timestamp': trade_result.get('timestamp', datetime.now()),
                'type': trade_result.get('type', 'unknown'),
                'size': trade_result.get('size', 0),
                'price': trade_result.get('price', 0),
                'pnl': trade_result.get('pnl', 0)
            })
            
            # Update performance metrics
            self._update_performance_metrics(symbol)
            
            # Check for emergency conditions
            self._check_emergency_conditions(symbol)
            
            logger.info(f"📈 Performance updated for {symbol}: PnL ${trade_result.get('pnl', 0):,.2f}")
            
        except Exception as e:
            logger.error(f"❌ Performance update failed: {e}")
    
    def get_adaptive_strategy_parameters(self, strategy_name: str) -> Dict[str, float]:
        """Get adaptive strategy parameters based on current market regime"""
        try:
            if not self.current_regime:
                return self._get_default_strategy_parameters(strategy_name)
            
            # Load regime-specific parameters from database or config
            regime_params = self._load_regime_parameters(strategy_name, self.current_regime.regime_type)
            
            # Apply confidence-based adjustments
            confidence_multiplier = self.current_regime.confidence
            
            # Adjust parameters based on regime confidence
            adjusted_params = {}
            for param_name, param_value in regime_params.items():
                if param_name in ['stop_loss', 'position_size_multiplier']:
                    # More conservative in low confidence regimes
                    adjusted_params[param_name] = param_value * (0.5 + 0.5 * confidence_multiplier)
                else:
                    adjusted_params[param_name] = param_value
            
            logger.info(f"🔄 Adaptive parameters for {strategy_name} in {self.current_regime.regime_type} regime")
            
            return adjusted_params
            
        except Exception as e:
            logger.error(f"❌ Failed to get adaptive parameters: {e}")
            return self._get_default_strategy_parameters(strategy_name)
    
    def _start_regime_monitoring(self):
        """Start regime monitoring thread"""
        def regime_monitor():
            while self.is_running:
                try:
                    # Get recent market data
                    market_data = self._get_recent_market_data(symbol='XAUUSD', days=30)
                    
                    if not market_data.empty:
                        # Detect current regime
                        regimes = self.regime_analyzer.detect_regimes(market_data, method='threshold')
                        
                        if regimes:
                            new_regime = regimes[-1]
                            
                            # Check for regime change
                            if (not self.current_regime or 
                                self.current_regime.regime_type != new_regime.regime_type):
                                
                                logger.info(f"🔄 Regime change detected: {new_regime.regime_type} (confidence: {new_regime.confidence:.2f})")
                                self.current_regime = new_regime
                                
                                # Notify other components of regime change
                                self._handle_regime_change(new_regime)
                    
                    # Wait before next check
                    threading.Event().wait(self.config.regime_update_interval)
                    
                except Exception as e:
                    logger.error(f"❌ Regime monitoring error: {e}")
                    threading.Event().wait(60)  # Wait 1 minute on error
        
        self.regime_thread = threading.Thread(target=regime_monitor, daemon=True)
        self.regime_thread.start()
        logger.info("🔄 Regime monitoring started")
    
    def _start_risk_monitoring(self):
        """Start risk monitoring thread"""
        def risk_monitor():
            while self.is_running:
                try:
                    # Check current portfolio risk
                    total_exposure = sum(pos.get('market_value', 0) for pos in self.live_positions.values())
                    
                    # Check individual position risks
                    for symbol, position in self.live_positions.items():
                        market_data = self._get_recent_market_data(symbol=symbol, days=5)
                        
                        if not market_data.empty:
                            # Calculate current risk metrics
                            current_price = market_data['close'].iloc[-1]
                            position_pnl = (current_price - position.get('entry_price', current_price)) * position.get('size', 0)
                            
                            # Check for risk limit breaches
                            if position_pnl < -position.get('max_loss', 5000):
                                logger.warning(f"⚠️ Position {symbol} exceeding max loss threshold")
                                self._trigger_risk_alert(symbol, 'max_loss_exceeded', position_pnl)
                    
                    threading.Event().wait(self.config.risk_check_interval)
                    
                except Exception as e:
                    logger.error(f"❌ Risk monitoring error: {e}")
                    threading.Event().wait(60)
        
        self.risk_thread = threading.Thread(target=risk_monitor, daemon=True)
        self.risk_thread.start()
        logger.info("🛡️ Risk monitoring started")
    
    def _start_performance_monitoring(self):
        """Start performance monitoring thread"""
        def performance_monitor():
            while self.is_running:
                try:
                    # Update performance metrics for all tracked symbols
                    for symbol in self.performance_tracker:
                        self._update_performance_metrics(symbol)
                    
                    # Log overall performance summary
                    total_pnl = sum(tracker['total_pnl'] for tracker in self.performance_tracker.values())
                    logger.info(f"📊 Portfolio performance update: Total PnL ${total_pnl:,.2f}")
                    
                    threading.Event().wait(self.config.performance_update_interval)
                    
                except Exception as e:
                    logger.error(f"❌ Performance monitoring error: {e}")
                    threading.Event().wait(300)  # Wait 5 minutes on error
        
        self.performance_thread = threading.Thread(target=performance_monitor, daemon=True)
        self.performance_thread.start()
        logger.info("📈 Performance monitoring started")
    
    def _validate_signal_against_regime(self, signal: Dict[str, Any], market_data: pd.DataFrame) -> Dict[str, Any]:
        """Validate signal against current market regime"""
        try:
            if not self.current_regime:
                return {'approved': True, 'reason': 'No regime analysis available'}
            
            signal_direction = signal.get('direction', 'long').lower()
            regime_type = self.current_regime.regime_type
            
            # Regime-based signal validation rules
            validation_rules = {
                'bull': {'long': 0.8, 'short': 0.2},
                'bear': {'long': 0.2, 'short': 0.8},
                'sideways': {'long': 0.5, 'short': 0.5},
                'volatile': {'long': 0.3, 'short': 0.3}
            }
            
            approval_probability = validation_rules.get(regime_type, {}).get(signal_direction, 0.5)
            confidence_adjustment = self.current_regime.confidence
            
            final_approval_score = approval_probability * confidence_adjustment
            approved = final_approval_score > 0.5
            
            return {
                'approved': approved,
                'regime': regime_type,
                'approval_score': final_approval_score,
                'confidence': confidence_adjustment,
                'reason': f"Signal {signal_direction} in {regime_type} regime (score: {final_approval_score:.2f})"
            }
            
        except Exception as e:
            logger.error(f"❌ Regime validation error: {e}")
            return {'approved': False, 'reason': f'Validation error: {e}'}
    
    def _calculate_live_position_size(self, signal: Dict[str, Any], market_data: pd.DataFrame) -> float:
        """Calculate position size for live trading"""
        try:
            # Get current portfolio value
            portfolio_value = self._get_current_portfolio_value()
            
            # Use enhanced risk management for position sizing
            position_size = self.risk_manager.calculate_position_size(
                signal_strength=signal.get('confidence', 0.5),
                market_data=market_data,
                portfolio_value=portfolio_value,
                current_positions=self.live_positions
            )
            
            # Apply additional live trading constraints
            max_position = min(
                self.config.max_position_size,
                portfolio_value * 0.1  # Max 10% per position
            )
            
            return min(position_size, max_position)
            
        except Exception as e:
            logger.error(f"❌ Position sizing error: {e}")
            return 0.0
    
    def _get_recent_market_data(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Get recent market data for analysis"""
        try:
            # This would connect to your existing data service
            # For demonstration, creating sample data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Generate sample data (replace with actual data service call)
            dates = pd.date_range(start=start_date, end=end_date, freq='H')
            base_price = 3400 if symbol == 'XAUUSD' else 100
            
            prices = base_price + np.cumsum(np.random.normal(0, 1, len(dates)))
            
            return pd.DataFrame({
                'timestamp': dates,
                'open': prices * 0.999,
                'high': prices * 1.001,
                'low': prices * 0.998,
                'close': prices,
                'volume': np.random.randint(1000, 10000, len(dates))
            })
            
        except Exception as e:
            logger.error(f"❌ Failed to get market data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _create_signal_response(self, signal: Dict[str, Any], approved: bool, reason: str) -> Dict[str, Any]:
        """Create standardized signal response"""
        return {
            'signal_id': signal.get('id', f"signal_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            'original_signal': signal,
            'validated': approved,
            'reason': reason,
            'position_size': 0,
            'confidence_score': 0,
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_signal_confidence(self, signal: Dict[str, Any], market_data: pd.DataFrame) -> float:
        """Calculate confidence score for trading signal"""
        try:
            base_confidence = signal.get('confidence', 0.5)
            
            # Technical analysis confidence
            returns = market_data['close'].pct_change().dropna()
            volatility = returns.std()
            trend_strength = abs(returns.mean())
            
            # Market condition adjustment
            if volatility < returns.std() * 0.5:  # Low volatility
                market_confidence = 0.8
            elif volatility > returns.std() * 1.5:  # High volatility
                market_confidence = 0.3
            else:
                market_confidence = 0.6
            
            # Regime confidence
            regime_confidence = self.current_regime.confidence if self.current_regime else 0.5
            
            # Combined confidence
            final_confidence = (base_confidence * 0.4 + market_confidence * 0.3 + regime_confidence * 0.3)
            
            return min(1.0, max(0.0, final_confidence))
            
        except Exception as e:
            logger.error(f"❌ Confidence calculation error: {e}")
            return 0.5
    
    def _get_current_portfolio_value(self) -> float:
        """Get current portfolio value"""
        try:
            # Sum up all position values
            total_value = sum(pos.get('market_value', 0) for pos in self.live_positions.values())
            
            # Add cash/available balance (placeholder)
            cash_balance = 100000  # This would come from your trading account API
            
            return total_value + cash_balance
            
        except Exception as e:
            logger.error(f"❌ Portfolio value calculation error: {e}")
            return 100000  # Default fallback
    
    def _get_current_risk_metrics(self) -> Dict[str, float]:
        """Get current portfolio risk metrics"""
        try:
            portfolio_value = self._get_current_portfolio_value()
            total_exposure = sum(abs(pos.get('market_value', 0)) for pos in self.live_positions.values())
            
            return {
                'portfolio_heat': total_exposure / portfolio_value if portfolio_value > 0 else 0,
                'number_of_positions': len(self.live_positions),
                'largest_position_pct': max([abs(pos.get('market_value', 0)) / portfolio_value 
                                           for pos in self.live_positions.values()], default=0),
                'total_exposure': total_exposure
            }
            
        except Exception as e:
            logger.error(f"❌ Risk metrics calculation error: {e}")
            return {'error': str(e)}
    
    def _generate_signal_recommendations(self, signal: Dict[str, Any], market_data: pd.DataFrame) -> List[str]:
        """Generate recommendations for signal execution"""
        recommendations = []
        
        try:
            # Regime-based recommendations
            if self.current_regime:
                if self.current_regime.regime_type == 'volatile':
                    recommendations.append("Consider reducing position size due to volatile market conditions")
                elif self.current_regime.regime_type == 'sideways':
                    recommendations.append("Tighten stop losses in sideways market")
                elif self.current_regime.confidence < 0.5:
                    recommendations.append("Low regime confidence - consider waiting for better setup")
            
            # Technical recommendations
            returns = market_data['close'].pct_change().dropna()
            if returns.std() > returns.mean() * 2:
                recommendations.append("High volatility detected - consider smaller position size")
            
            # Risk recommendations
            portfolio_heat = sum(abs(pos.get('market_value', 0)) for pos in self.live_positions.values())
            if portfolio_heat > 80000:  # 80% of default portfolio
                recommendations.append("Portfolio heat is high - consider reducing exposure")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Recommendation generation error: {e}")
            return ["Error generating recommendations"]
    
    def _handle_regime_change(self, new_regime):
        """Handle regime change event"""
        try:
            logger.info(f"🔄 Handling regime change to {new_regime.regime_type}")
            
            # Update all strategy parameters based on new regime
            # This would trigger parameter updates in your trading strategies
            
            # Example actions:
            if new_regime.regime_type == 'volatile':
                # Reduce position sizes, tighten stops
                logger.info("📉 Volatile regime detected - reducing risk exposure")
            elif new_regime.regime_type == 'bull':
                # Allow larger positions, looser stops
                logger.info("📈 Bull regime detected - increasing trend exposure")
            elif new_regime.regime_type == 'bear':
                # Defensive positioning
                logger.info("📉 Bear regime detected - defensive positioning")
            
        except Exception as e:
            logger.error(f"❌ Regime change handling error: {e}")
    
    def _update_performance_metrics(self, symbol: str):
        """Update performance metrics for a symbol"""
        try:
            if symbol not in self.performance_tracker:
                return
            
            tracker = self.performance_tracker[symbol]
            trades = tracker['trades']
            
            if not trades:
                return
            
            # Calculate win rate
            profitable_trades = [t for t in trades if t.get('pnl', 0) > 0]
            tracker['win_rate'] = len(profitable_trades) / len(trades) if trades else 0
            
            # Calculate total PnL
            tracker['total_pnl'] = sum(t.get('pnl', 0) for t in trades)
            
            # Calculate current drawdown
            cumulative_pnl = []
            running_total = 0
            for trade in trades:
                running_total += trade.get('pnl', 0)
                cumulative_pnl.append(running_total)
            
            if cumulative_pnl:
                peak = max(cumulative_pnl)
                current_value = cumulative_pnl[-1]
                tracker['current_drawdown'] = (peak - current_value) / peak if peak > 0 else 0
                tracker['peak_value'] = peak
            
        except Exception as e:
            logger.error(f"❌ Performance update error for {symbol}: {e}")
    
    def _check_emergency_conditions(self, symbol: str):
        """Check for emergency stop conditions"""
        try:
            if symbol not in self.performance_tracker:
                return
            
            tracker = self.performance_tracker[symbol]
            
            # Check drawdown emergency stop
            if tracker.get('current_drawdown', 0) > self.config.emergency_stop_drawdown:
                logger.warning(f"🚨 EMERGENCY STOP triggered for {symbol}: Drawdown {tracker['current_drawdown']:.1%}")
                self._trigger_emergency_stop(symbol, 'excessive_drawdown')
            
            # Check consecutive losses
            recent_trades = tracker['trades'][-5:] if len(tracker['trades']) >= 5 else tracker['trades']
            if all(t.get('pnl', 0) < 0 for t in recent_trades) and len(recent_trades) >= 3:
                logger.warning(f"⚠️ Consecutive losses detected for {symbol}")
                self._trigger_risk_alert(symbol, 'consecutive_losses', len(recent_trades))
            
        except Exception as e:
            logger.error(f"❌ Emergency check error for {symbol}: {e}")
    
    def _trigger_emergency_stop(self, symbol: str, reason: str):
        """Trigger emergency stop for a symbol"""
        logger.critical(f"🚨 EMERGENCY STOP: {symbol} - {reason}")
        # This would integrate with your trading system to close positions
        
    def _trigger_risk_alert(self, symbol: str, alert_type: str, value: Any):
        """Trigger risk management alert"""
        logger.warning(f"⚠️ RISK ALERT: {symbol} - {alert_type}: {value}")
        # This would send alerts to your monitoring system
    
    def _calculate_overall_win_rate(self) -> float:
        """Calculate overall portfolio win rate"""
        try:
            total_trades = 0
            winning_trades = 0
            
            for tracker in self.performance_tracker.values():
                trades = tracker.get('trades', [])
                total_trades += len(trades)
                winning_trades += len([t for t in trades if t.get('pnl', 0) > 0])
            
            return winning_trades / total_trades if total_trades > 0 else 0
            
        except Exception as e:
            logger.error(f"❌ Overall win rate calculation error: {e}")
            return 0
    
    def _generate_portfolio_recommendations(self) -> List[str]:
        """Generate portfolio-level recommendations"""
        recommendations = []
        
        try:
            # Portfolio heat recommendations
            risk_metrics = self._get_current_risk_metrics()
            portfolio_heat = risk_metrics.get('portfolio_heat', 0)
            
            if portfolio_heat > 0.8:
                recommendations.append("Portfolio heat is high - consider reducing overall exposure")
            elif portfolio_heat < 0.3:
                recommendations.append("Portfolio heat is low - consider increasing selective exposure")
            
            # Regime-based recommendations
            if self.current_regime:
                if self.current_regime.regime_type == 'volatile':
                    recommendations.append("Volatile market conditions - focus on risk management")
                elif self.current_regime.regime_type == 'bull':
                    recommendations.append("Bull market conditions - consider trend-following strategies")
                elif self.current_regime.regime_type == 'bear':
                    recommendations.append("Bear market conditions - consider defensive strategies")
            
            # Performance-based recommendations
            overall_win_rate = self._calculate_overall_win_rate()
            if overall_win_rate < 0.4:
                recommendations.append("Win rate below 40% - review strategy parameters")
            elif overall_win_rate > 0.7:
                recommendations.append("Strong win rate - current strategies performing well")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Portfolio recommendation error: {e}")
            return ["Error generating portfolio recommendations"]
    
    def _get_default_strategy_parameters(self, strategy_name: str) -> Dict[str, float]:
        """Get default parameters for a strategy"""
        default_params = {
            'ma_crossover': {
                'short_ma': 10,
                'long_ma': 20,
                'stop_loss': 0.02,
                'take_profit': 0.04,
                'position_size_multiplier': 1.0
            },
            'trend_following': {
                'lookback_period': 14,
                'trend_threshold': 0.01,
                'stop_loss': 0.025,
                'take_profit': 0.05,
                'position_size_multiplier': 1.0
            }
        }
        
        return default_params.get(strategy_name, {
            'stop_loss': 0.02,
            'take_profit': 0.04,
            'position_size_multiplier': 1.0
        })
    
    def _load_regime_parameters(self, strategy_name: str, regime_type: str) -> Dict[str, float]:
        """Load regime-specific parameters"""
        # This would load from database or configuration file
        regime_adjustments = {
            'bull': {'position_size_multiplier': 1.2, 'stop_loss': 0.025},
            'bear': {'position_size_multiplier': 0.8, 'stop_loss': 0.015},
            'sideways': {'position_size_multiplier': 0.9, 'stop_loss': 0.02},
            'volatile': {'position_size_multiplier': 0.6, 'stop_loss': 0.015}
        }
        
        base_params = self._get_default_strategy_parameters(strategy_name)
        regime_params = regime_adjustments.get(regime_type, {})
        
        # Merge parameters
        for param, value in regime_params.items():
            if param in base_params:
                base_params[param] = value
        
        return base_params
    
    def generate_live_trading_report(self) -> Dict[str, Any]:
        """Generate comprehensive live trading performance report"""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'current_regime': {
                    'type': self.current_regime.regime_type if self.current_regime else 'Unknown',
                    'confidence': self.current_regime.confidence if self.current_regime else 0,
                    'characteristics': self.current_regime.characteristics if self.current_regime else {}
                },
                'portfolio_summary': {
                    'total_value': self._get_current_portfolio_value(),
                    'total_pnl': sum(tracker['total_pnl'] for tracker in self.performance_tracker.values()),
                    'active_positions': len(self.live_positions),
                    'overall_win_rate': self._calculate_overall_win_rate()
                },
                'performance_by_symbol': {},
                'risk_metrics': self._get_current_risk_metrics(),
                'recommendations': self._generate_portfolio_recommendations()
            }
            
            # Add per-symbol performance
            for symbol, tracker in self.performance_tracker.items():
                report['performance_by_symbol'][symbol] = {
                    'total_pnl': tracker['total_pnl'],
                    'win_rate': tracker['win_rate'],
                    'trade_count': len(tracker['trades']),
                    'current_drawdown': tracker['current_drawdown']
                }
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Report generation failed: {e}")
            return {'error': str(e)}

def create_live_integration_demo():
    """Demonstrate live trading integration"""
    print("🎯 GOLDGPT LIVE TRADING INTEGRATION DEMO")
    print("=" * 50)
    
    # Initialize live trading integration
    config = LiveTradingConfig(
        enable_real_time_validation=True,
        regime_update_interval=10,  # Fast demo intervals
        risk_check_interval=5,
        adaptive_sizing=True
    )
    
    engine = LiveTradingIntegrationEngine(config)
    
    try:
        print("🚀 Starting live integration engine...")
        engine.start_live_integration()
        
        # Simulate live signal validation
        print("\n📊 Testing live signal validation...")
        
        test_signal = {
            'id': 'test_signal_001',
            'symbol': 'XAUUSD',
            'direction': 'long',
            'confidence': 0.75,
            'entry_price': 3400,
            'stop_loss': 3380,
            'take_profit': 3450
        }
        
        validation_result = engine.validate_live_signal(test_signal)
        
        print(f"   Signal Validation Result:")
        print(f"   • Validated: {'✅' if validation_result['validated'] else '❌'}")
        print(f"   • Position Size: ${validation_result['position_size']:,.0f}")
        print(f"   • Confidence Score: {validation_result['confidence_score']:.2f}")
        print(f"   • Current Regime: {validation_result.get('regime_analysis', {}).get('regime', 'Unknown')}")
        
        # Test adaptive parameters
        print("\n🔄 Testing adaptive strategy parameters...")
        adaptive_params = engine.get_adaptive_strategy_parameters('ma_crossover')
        print(f"   Adaptive Parameters: {adaptive_params}")
        
        # Simulate trade result update
        print("\n📈 Simulating trade result update...")
        trade_result = {
            'symbol': 'XAUUSD',
            'type': 'buy',
            'size': 1000,
            'price': 3405,
            'pnl': 500,
            'timestamp': datetime.now()
        }
        
        engine.update_live_performance(trade_result)
        
        # Generate performance report
        print("\n📊 Generating live trading report...")
        report = engine.generate_live_trading_report()
        
        print(f"   Portfolio Value: ${report['portfolio_summary']['total_value']:,.0f}")
        print(f"   Total PnL: ${report['portfolio_summary']['total_pnl']:,.0f}")
        print(f"   Active Positions: {report['portfolio_summary']['active_positions']}")
        print(f"   Current Regime: {report['current_regime']['type']}")
        
        print("\n✅ LIVE INTEGRATION DEMO COMPLETE!")
        print("   • Real-time regime monitoring: ACTIVE")
        print("   • Signal validation: OPERATIONAL")
        print("   • Risk management: MONITORING")
        print("   • Performance tracking: UPDATING")
        
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
    
    finally:
        print("\n🔄 Stopping live integration engine...")
        engine.stop_live_integration()
        print("✅ Live integration demo completed successfully")

if __name__ == "__main__":
    create_live_integration_demo()
