"""
🏆 ADVANCED BACKTESTING SYSTEM FOR GOLDGPT
==========================================

Professional backtesting framework with walk-forward optimization,
Monte Carlo simulation, and comprehensive risk management.

Author: GoldGPT AI System
Created: July 23, 2025
"""

import numpy as np
import pandas as pd
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
import threading
import asyncio
import json
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('advanced_backtester')

@dataclass
class BacktestConfig:
    """Backtesting configuration parameters"""
    initial_capital: float = 100000.0
    commission_rate: float = 0.0002  # 0.02%
    slippage_bps: float = 0.5  # 0.5 basis points
    max_positions: int = 10
    risk_free_rate: float = 0.02  # 2% annual
    confidence_level: float = 0.95  # For VaR calculations
    lookback_window: int = 252  # Trading days for volatility calc
    rebalance_frequency: str = 'daily'  # daily, weekly, monthly

@dataclass
class Trade:
    """Individual trade record"""
    entry_time: datetime
    exit_time: Optional[datetime]
    symbol: str
    direction: str  # 'long' or 'short'
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    commission: float
    slippage: float
    pnl: Optional[float]
    strategy: str
    confidence: float
    trade_id: str

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    win_rate: float
    profit_factor: float
    information_ratio: float
    tracking_error: float
    var_95: float
    cvar_95: float
    recovery_factor: float
    average_win: float
    average_loss: float
    trades_count: int
    winning_trades: int
    losing_trades: int

@dataclass
class MarketRegime:
    """Market regime classification"""
    period_start: datetime
    period_end: datetime
    regime_type: str  # 'bull', 'bear', 'sideways'
    volatility_regime: str  # 'low', 'medium', 'high'
    trend_strength: float
    volatility_level: float
    regime_confidence: float

class AdvancedBacktester:
    """
    Professional backtesting system with advanced features:
    - Walk-forward optimization
    - Monte Carlo simulation
    - Out-of-sample testing
    - Comprehensive risk management
    """
    
    def __init__(self, config: BacktestConfig = None, db_path: str = "goldgpt_backtesting.db"):
        self.config = config or BacktestConfig()
        self.db_path = db_path
        self.trades: List[Trade] = []
        self.portfolio_values: List[float] = []
        self.timestamps: List[datetime] = []
        self.positions: Dict[str, float] = {}
        self.cash = self.config.initial_capital
        self.market_data: pd.DataFrame = None
        self.strategies: Dict[str, Any] = {}
        self.risk_manager = RiskManagement(self.config)
        self.performance_calculator = PerformanceMetrics()
        self.regime_analyzer = MarketRegimeAnalysis()
        self.init_database()
        logger.info("🏆 Advanced Backtester initialized")
    
    def init_database(self):
        """Initialize backtesting database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create trades table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS backtest_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT UNIQUE NOT NULL,
                    entry_time DATETIME NOT NULL,
                    exit_time DATETIME,
                    symbol TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    quantity REAL NOT NULL,
                    commission REAL NOT NULL,
                    slippage REAL NOT NULL,
                    pnl REAL,
                    strategy TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create backtest results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS backtest_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    backtest_name TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    start_date DATE NOT NULL,
                    end_date DATE NOT NULL,
                    initial_capital REAL NOT NULL,
                    final_capital REAL NOT NULL,
                    total_return REAL NOT NULL,
                    sharpe_ratio REAL NOT NULL,
                    max_drawdown REAL NOT NULL,
                    win_rate REAL NOT NULL,
                    trades_count INTEGER NOT NULL,
                    performance_metrics TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create walk forward results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS walk_forward_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    optimization_id TEXT NOT NULL,
                    period_start DATE NOT NULL,
                    period_end DATE NOT NULL,
                    in_sample_return REAL NOT NULL,
                    out_sample_return REAL NOT NULL,
                    parameter_set TEXT NOT NULL,
                    performance_metrics TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create monte carlo results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS monte_carlo_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    simulation_id TEXT NOT NULL,
                    simulation_number INTEGER NOT NULL,
                    final_return REAL NOT NULL,
                    max_drawdown REAL NOT NULL,
                    sharpe_ratio REAL NOT NULL,
                    var_95 REAL NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("✅ Backtesting database initialized")
            
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
    
    def load_market_data(self, data: pd.DataFrame):
        """Load market data for backtesting"""
        self.market_data = data.copy()
        self.market_data = self.market_data.sort_values('timestamp')
        logger.info(f"📊 Loaded {len(self.market_data)} data points")
    
    def add_strategy(self, name: str, strategy_func: callable, parameters: Dict[str, Any]):
        """Add a trading strategy to backtest"""
        self.strategies[name] = {
            'function': strategy_func,
            'parameters': parameters
        }
        logger.info(f"➕ Added strategy: {name}")
    
    def run_backtest(self, strategy_name: str, start_date: datetime = None, 
                     end_date: datetime = None) -> PerformanceMetrics:
        """Run a complete backtest for a strategy"""
        try:
            if strategy_name not in self.strategies:
                raise ValueError(f"Strategy {strategy_name} not found")
            
            # Reset state
            self._reset_state()
            
            # Filter data by date range
            if start_date and end_date:
                mask = (self.market_data['timestamp'] >= start_date) & \
                       (self.market_data['timestamp'] <= end_date)
                data = self.market_data[mask].copy()
            else:
                data = self.market_data.copy()
            
            strategy = self.strategies[strategy_name]
            
            logger.info(f"🚀 Starting backtest for {strategy_name}")
            
            # Main backtesting loop
            for i, row in data.iterrows():
                current_time = row['timestamp']
                current_price = row['close']
                
                # Generate strategy signals
                signals = strategy['function'](
                    data.iloc[:i+1], 
                    **strategy['parameters']
                )
                
                # Process signals
                if signals:
                    for signal in signals:
                        self._process_signal(signal, current_time, current_price, strategy_name)
                
                # Update portfolio value
                portfolio_value = self._calculate_portfolio_value(current_price)
                self.portfolio_values.append(portfolio_value)
                self.timestamps.append(current_time)
                
                # Risk management checks
                self.risk_manager.check_risk_limits(
                    portfolio_value, self.positions, current_price
                )
            
            # Close all open positions
            self._close_all_positions(data.iloc[-1]['close'], data.iloc[-1]['timestamp'])
            
            # Calculate performance metrics
            performance = self._calculate_performance_metrics()
            
            # Store results
            self._store_backtest_results(strategy_name, performance, start_date, end_date)
            
            logger.info(f"✅ Backtest completed. Return: {performance.total_return:.2%}")
            return performance
            
        except Exception as e:
            logger.error(f"❌ Backtest failed: {e}")
            raise
    
    def walk_forward_optimization(self, strategy_name: str, 
                                parameter_grid: Dict[str, List], 
                                optimization_window: int = 252,
                                validation_window: int = 63) -> Dict[str, Any]:
        """
        Walk-forward optimization to prevent overfitting
        """
        try:
            logger.info(f"🚶 Starting walk-forward optimization for {strategy_name}")
            
            if len(self.market_data) < optimization_window + validation_window:
                raise ValueError("Insufficient data for walk-forward optimization")
            
            results = []
            optimization_id = f"wfo_{strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Generate parameter combinations
            param_combinations = self._generate_parameter_combinations(parameter_grid)
            
            start_idx = 0
            while start_idx + optimization_window + validation_window <= len(self.market_data):
                # Define in-sample and out-of-sample periods
                opt_end_idx = start_idx + optimization_window
                val_end_idx = opt_end_idx + validation_window
                
                in_sample_data = self.market_data.iloc[start_idx:opt_end_idx]
                out_sample_data = self.market_data.iloc[opt_end_idx:val_end_idx]
                
                period_start = in_sample_data.iloc[0]['timestamp']
                period_end = out_sample_data.iloc[-1]['timestamp']
                
                logger.info(f"📊 Optimizing period: {period_start} to {period_end}")
                
                # Optimize parameters on in-sample data
                best_params, best_performance = self._optimize_parameters(
                    strategy_name, param_combinations, in_sample_data
                )
                
                # Test on out-of-sample data
                out_sample_performance = self._test_out_of_sample(
                    strategy_name, best_params, out_sample_data
                )
                
                # Store results
                period_result = {
                    'optimization_id': optimization_id,
                    'period_start': period_start,
                    'period_end': period_end,
                    'best_parameters': best_params,
                    'in_sample_return': best_performance.total_return,
                    'out_sample_return': out_sample_performance.total_return,
                    'in_sample_sharpe': best_performance.sharpe_ratio,
                    'out_sample_sharpe': out_sample_performance.sharpe_ratio,
                    'in_sample_metrics': best_performance,
                    'out_sample_metrics': out_sample_performance
                }
                
                results.append(period_result)
                
                # Store in database
                self._store_walk_forward_result(period_result)
                
                # Move to next period
                start_idx += validation_window
            
            # Analyze overall walk-forward results
            wf_analysis = self._analyze_walk_forward_results(results)
            
            logger.info(f"✅ Walk-forward optimization completed. Avg OOS return: {wf_analysis['avg_oos_return']:.2%}")
            return wf_analysis
            
        except Exception as e:
            logger.error(f"❌ Walk-forward optimization failed: {e}")
            raise
    
    def monte_carlo_simulation(self, strategy_name: str, num_simulations: int = 1000,
                             confidence_levels: List[float] = [0.05, 0.95]) -> Dict[str, Any]:
        """
        Monte Carlo simulation for robustness testing
        """
        try:
            logger.info(f"🎲 Starting Monte Carlo simulation for {strategy_name} ({num_simulations} runs)")
            
            if strategy_name not in self.strategies:
                raise ValueError(f"Strategy {strategy_name} not found")
            
            simulation_id = f"mc_{strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            results = []
            
            original_data = self.market_data.copy()
            
            for sim_num in range(num_simulations):
                if sim_num % 100 == 0:
                    logger.info(f"📊 Running simulation {sim_num + 1}/{num_simulations}")
                
                # Generate randomized price path
                randomized_data = self._generate_randomized_path(original_data)
                
                # Run backtest on randomized data
                self.market_data = randomized_data
                performance = self.run_backtest(strategy_name)
                
                sim_result = {
                    'simulation_id': simulation_id,
                    'simulation_number': sim_num + 1,
                    'final_return': performance.total_return,
                    'max_drawdown': performance.max_drawdown,
                    'sharpe_ratio': performance.sharpe_ratio,
                    'var_95': performance.var_95,
                    'volatility': performance.volatility,
                    'win_rate': performance.win_rate
                }
                
                results.append(sim_result)
                self._store_monte_carlo_result(sim_result)
            
            # Restore original data
            self.market_data = original_data
            
            # Analyze simulation results
            mc_analysis = self._analyze_monte_carlo_results(results, confidence_levels)
            
            logger.info(f"✅ Monte Carlo simulation completed. Expected return: {mc_analysis['expected_return']:.2%}")
            return mc_analysis
            
        except Exception as e:
            logger.error(f"❌ Monte Carlo simulation failed: {e}")
            raise
    
    def _reset_state(self):
        """Reset backtester state"""
        self.trades = []
        self.portfolio_values = []
        self.timestamps = []
        self.positions = {}
        self.cash = self.config.initial_capital
    
    def _process_signal(self, signal: Dict[str, Any], current_time: datetime, 
                       current_price: float, strategy_name: str):
        """Process a trading signal"""
        try:
            symbol = signal.get('symbol', 'XAUUSD')
            direction = signal.get('direction', 'long')
            confidence = signal.get('confidence', 0.5)
            
            # Calculate position size using risk management
            position_size = self.risk_manager.calculate_position_size(
                self.cash + sum(self.positions.values()) * current_price,
                current_price,
                confidence
            )
            
            if position_size <= 0:
                return
            
            # Calculate costs
            commission = position_size * current_price * self.config.commission_rate
            slippage = position_size * current_price * (self.config.slippage_bps / 10000)
            total_cost = position_size * current_price + commission + slippage
            
            if total_cost > self.cash:
                return  # Insufficient funds
            
            # Execute trade
            trade_id = f"{strategy_name}_{current_time.strftime('%Y%m%d_%H%M%S')}_{len(self.trades)}"
            
            trade = Trade(
                entry_time=current_time,
                exit_time=None,
                symbol=symbol,
                direction=direction,
                entry_price=current_price + (slippage / position_size),
                exit_price=None,
                quantity=position_size,
                commission=commission,
                slippage=slippage,
                pnl=None,
                strategy=strategy_name,
                confidence=confidence,
                trade_id=trade_id
            )
            
            self.trades.append(trade)
            
            # Update positions and cash
            if direction == 'long':
                self.positions[symbol] = self.positions.get(symbol, 0) + position_size
            else:
                self.positions[symbol] = self.positions.get(symbol, 0) - position_size
            
            self.cash -= total_cost
            
        except Exception as e:
            logger.error(f"❌ Signal processing failed: {e}")
    
    def _calculate_portfolio_value(self, current_price: float) -> float:
        """Calculate current portfolio value"""
        position_value = sum(pos * current_price for pos in self.positions.values())
        return self.cash + position_value
    
    def _close_all_positions(self, final_price: float, final_time: datetime):
        """Close all open positions at the end of backtest"""
        for trade in self.trades:
            if trade.exit_time is None:
                # Close the trade
                trade.exit_time = final_time
                trade.exit_price = final_price
                
                # Calculate P&L
                if trade.direction == 'long':
                    trade.pnl = (trade.exit_price - trade.entry_price) * trade.quantity - trade.commission
                else:
                    trade.pnl = (trade.entry_price - trade.exit_price) * trade.quantity - trade.commission
        
        # Clear positions and add cash
        total_position_value = sum(pos * final_price for pos in self.positions.values())
        self.cash += total_position_value
        self.positions = {}
    
    def _calculate_performance_metrics(self) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        try:
            if not self.portfolio_values or not self.trades:
                return PerformanceMetrics(
                    total_return=0.0, annualized_return=0.0, volatility=0.0,
                    sharpe_ratio=0.0, sortino_ratio=0.0, calmar_ratio=0.0,
                    max_drawdown=0.0, max_drawdown_duration=0, win_rate=0.0,
                    profit_factor=0.0, information_ratio=0.0, tracking_error=0.0,
                    var_95=0.0, cvar_95=0.0, recovery_factor=0.0,
                    average_win=0.0, average_loss=0.0, trades_count=0,
                    winning_trades=0, losing_trades=0
                )
            
            # Convert to numpy arrays for calculations
            portfolio_values = np.array(self.portfolio_values)
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            
            # Basic metrics
            total_return = (portfolio_values[-1] - self.config.initial_capital) / self.config.initial_capital
            
            # Annualized return
            days = len(portfolio_values)
            annualized_return = (1 + total_return) ** (252 / days) - 1
            
            # Volatility
            volatility = np.std(returns) * np.sqrt(252)
            
            # Sharpe ratio
            excess_returns = returns - (self.config.risk_free_rate / 252)
            sharpe_ratio = np.mean(excess_returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            
            # Sortino ratio
            negative_returns = returns[returns < 0]
            downside_volatility = np.std(negative_returns) * np.sqrt(252) if len(negative_returns) > 0 else 0.001
            sortino_ratio = (annualized_return - self.config.risk_free_rate) / downside_volatility
            
            # Maximum drawdown
            peak = np.maximum.accumulate(portfolio_values)
            drawdown = (portfolio_values - peak) / peak
            max_drawdown = -np.min(drawdown)
            
            # Drawdown duration
            max_drawdown_duration = self._calculate_max_drawdown_duration(drawdown)
            
            # Calmar ratio
            calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0
            
            # Trade-based metrics
            completed_trades = [t for t in self.trades if t.pnl is not None]
            winning_trades = [t for t in completed_trades if t.pnl > 0]
            losing_trades = [t for t in completed_trades if t.pnl < 0]
            
            win_rate = len(winning_trades) / len(completed_trades) if completed_trades else 0
            
            total_wins = sum(t.pnl for t in winning_trades)
            total_losses = abs(sum(t.pnl for t in losing_trades))
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
            
            average_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
            average_loss = np.mean([abs(t.pnl) for t in losing_trades]) if losing_trades else 0
            
            # VaR and CVaR
            var_95 = np.percentile(returns, 5)
            cvar_95 = np.mean(returns[returns <= var_95])
            
            # Recovery factor
            recovery_factor = total_return / max_drawdown if max_drawdown > 0 else 0
            
            # Information ratio and tracking error (vs risk-free rate)
            benchmark_returns = np.full_like(returns, self.config.risk_free_rate / 252)
            excess_returns_vs_benchmark = returns - benchmark_returns
            tracking_error = np.std(excess_returns_vs_benchmark) * np.sqrt(252)
            information_ratio = np.mean(excess_returns_vs_benchmark) / np.std(excess_returns_vs_benchmark) * np.sqrt(252) if np.std(excess_returns_vs_benchmark) > 0 else 0
            
            return PerformanceMetrics(
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                max_drawdown=max_drawdown,
                max_drawdown_duration=max_drawdown_duration,
                win_rate=win_rate,
                profit_factor=profit_factor,
                information_ratio=information_ratio,
                tracking_error=tracking_error,
                var_95=var_95,
                cvar_95=cvar_95,
                recovery_factor=recovery_factor,
                average_win=average_win,
                average_loss=average_loss,
                trades_count=len(completed_trades),
                winning_trades=len(winning_trades),
                losing_trades=len(losing_trades)
            )
            
        except Exception as e:
            logger.error(f"❌ Performance calculation failed: {e}")
            raise
    
    def _calculate_max_drawdown_duration(self, drawdown: np.ndarray) -> int:
        """Calculate maximum drawdown duration in days"""
        duration = 0
        max_duration = 0
        
        for dd in drawdown:
            if dd < 0:
                duration += 1
                max_duration = max(max_duration, duration)
            else:
                duration = 0
        
        return max_duration
    
    def _generate_parameter_combinations(self, parameter_grid: Dict[str, List]) -> List[Dict[str, Any]]:
        """Generate all parameter combinations for optimization"""
        import itertools
        
        keys = list(parameter_grid.keys())
        values = list(parameter_grid.values())
        
        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations
    
    def _optimize_parameters(self, strategy_name: str, param_combinations: List[Dict], 
                           data: pd.DataFrame) -> Tuple[Dict[str, Any], PerformanceMetrics]:
        """Optimize parameters on in-sample data"""
        best_params = None
        best_performance = None
        best_sharpe = -float('inf')
        
        original_data = self.market_data
        self.market_data = data
        
        for params in param_combinations:
            try:
                # Update strategy parameters
                self.strategies[strategy_name]['parameters'] = params
                
                # Run backtest
                performance = self.run_backtest(strategy_name)
                
                # Check if this is the best performance
                if performance.sharpe_ratio > best_sharpe:
                    best_sharpe = performance.sharpe_ratio
                    best_params = params.copy()
                    best_performance = performance
                    
            except Exception as e:
                logger.warning(f"⚠️ Parameter combination failed: {params}, error: {e}")
                continue
        
        self.market_data = original_data
        return best_params, best_performance
    
    def _test_out_of_sample(self, strategy_name: str, best_params: Dict[str, Any], 
                           data: pd.DataFrame) -> PerformanceMetrics:
        """Test optimized parameters on out-of-sample data"""
        original_data = self.market_data
        self.market_data = data
        
        # Set optimized parameters
        self.strategies[strategy_name]['parameters'] = best_params
        
        # Run backtest
        performance = self.run_backtest(strategy_name)
        
        self.market_data = original_data
        return performance
    
    def _generate_randomized_path(self, original_data: pd.DataFrame) -> pd.DataFrame:
        """Generate randomized price path for Monte Carlo simulation"""
        data = original_data.copy()
        
        # Calculate returns
        data['returns'] = data['close'].pct_change().fillna(0)
        
        # Bootstrap returns (sampling with replacement)
        randomized_returns = np.random.choice(
            data['returns'].dropna().values, 
            size=len(data), 
            replace=True
        )
        
        # Generate new price path
        new_prices = [data['close'].iloc[0]]
        for ret in randomized_returns[1:]:
            new_price = new_prices[-1] * (1 + ret)
            new_prices.append(new_price)
        
        # Update data with new prices
        data['close'] = new_prices
        data['high'] = data['close'] * (1 + np.random.uniform(0, 0.01, len(data)))
        data['low'] = data['close'] * (1 - np.random.uniform(0, 0.01, len(data)))
        data['open'] = data['close'].shift(1).fillna(data['close'].iloc[0])
        
        return data
    
    def _analyze_walk_forward_results(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze walk-forward optimization results"""
        if not results:
            return {}
        
        in_sample_returns = [r['in_sample_return'] for r in results]
        out_sample_returns = [r['out_sample_return'] for r in results]
        
        analysis = {
            'periods_count': len(results),
            'avg_is_return': np.mean(in_sample_returns),
            'avg_oos_return': np.mean(out_sample_returns),
            'is_oos_correlation': np.corrcoef(in_sample_returns, out_sample_returns)[0, 1],
            'oos_consistency': len([r for r in out_sample_returns if r > 0]) / len(out_sample_returns),
            'degradation_factor': np.mean(out_sample_returns) / np.mean(in_sample_returns) if np.mean(in_sample_returns) != 0 else 0,
            'oos_volatility': np.std(out_sample_returns),
            'worst_oos_period': min(out_sample_returns),
            'best_oos_period': max(out_sample_returns),
            'detailed_results': results
        }
        
        return analysis
    
    def _analyze_monte_carlo_results(self, results: List[Dict], 
                                   confidence_levels: List[float]) -> Dict[str, Any]:
        """Analyze Monte Carlo simulation results"""
        returns = [r['final_return'] for r in results]
        drawdowns = [r['max_drawdown'] for r in results]
        sharpe_ratios = [r['sharpe_ratio'] for r in results]
        
        analysis = {
            'simulations_count': len(results),
            'expected_return': np.mean(returns),
            'return_volatility': np.std(returns),
            'return_percentiles': {
                f'p{int(level*100)}': np.percentile(returns, level*100) 
                for level in confidence_levels
            },
            'positive_outcomes': len([r for r in returns if r > 0]) / len(returns),
            'expected_drawdown': np.mean(drawdowns),
            'worst_case_drawdown': max(drawdowns),
            'expected_sharpe': np.mean(sharpe_ratios),
            'sharpe_stability': np.std(sharpe_ratios),
            'tail_risk': {
                'var_95': np.percentile(returns, 5),
                'cvar_95': np.mean([r for r in returns if r <= np.percentile(returns, 5)])
            },
            'detailed_results': results
        }
        
        return analysis
    
    def _store_backtest_results(self, strategy_name: str, performance: PerformanceMetrics,
                               start_date: datetime = None, end_date: datetime = None):
        """Store backtest results in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO backtest_results (
                    backtest_name, strategy_name, start_date, end_date,
                    initial_capital, final_capital, total_return, sharpe_ratio,
                    max_drawdown, win_rate, trades_count, performance_metrics
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                strategy_name,
                start_date or self.timestamps[0] if self.timestamps else datetime.now(),
                end_date or self.timestamps[-1] if self.timestamps else datetime.now(),
                self.config.initial_capital,
                self.portfolio_values[-1] if self.portfolio_values else self.config.initial_capital,
                performance.total_return,
                performance.sharpe_ratio,
                performance.max_drawdown,
                performance.win_rate,
                performance.trades_count,
                json.dumps(asdict(performance))
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Failed to store backtest results: {e}")
    
    def _store_walk_forward_result(self, result: Dict[str, Any]):
        """Store walk-forward optimization result"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO walk_forward_results (
                    optimization_id, period_start, period_end,
                    in_sample_return, out_sample_return, parameter_set,
                    performance_metrics
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                result['optimization_id'],
                result['period_start'],
                result['period_end'],
                result['in_sample_return'],
                result['out_sample_return'],
                json.dumps(result['best_parameters']),
                json.dumps({
                    'in_sample': asdict(result['in_sample_metrics']),
                    'out_sample': asdict(result['out_sample_metrics'])
                })
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Failed to store walk-forward result: {e}")
    
    def _store_monte_carlo_result(self, result: Dict[str, Any]):
        """Store Monte Carlo simulation result"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO monte_carlo_results (
                    simulation_id, simulation_number, final_return,
                    max_drawdown, sharpe_ratio, var_95
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                result['simulation_id'],
                result['simulation_number'],
                result['final_return'],
                result['max_drawdown'],
                result['sharpe_ratio'],
                result['var_95']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Failed to store Monte Carlo result: {e}")


class RiskManagement:
    """
    Advanced risk management system for backtesting
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.max_portfolio_risk = 0.02  # 2% portfolio risk per trade
        self.max_correlation_exposure = 0.6  # 60% max correlation exposure
        self.volatility_lookback = 20  # Days for volatility calculation
        logger.info("🛡️ Risk Management initialized")
    
    def calculate_position_size(self, portfolio_value: float, price: float, 
                               confidence: float, volatility: float = None) -> float:
        """Calculate position size based on volatility and confidence"""
        try:
            # Base position size (Kelly criterion adaptation)
            base_risk = self.max_portfolio_risk * confidence
            
            # Adjust for volatility if provided
            if volatility:
                volatility_adjustment = min(1.0, 0.2 / volatility)  # Reduce size for high volatility
                base_risk *= volatility_adjustment
            
            # Calculate position size
            risk_amount = portfolio_value * base_risk
            position_size = risk_amount / price
            
            # Maximum position size check (no more than 10% of portfolio)
            max_position_value = portfolio_value * 0.1
            max_position_size = max_position_value / price
            
            return min(position_size, max_position_size)
            
        except Exception as e:
            logger.error(f"❌ Position sizing failed: {e}")
            return 0.0
    
    def check_risk_limits(self, portfolio_value: float, positions: Dict[str, float], 
                         current_price: float):
        """Check various risk limits"""
        try:
            # Maximum drawdown check
            if portfolio_value < self.config.initial_capital * 0.8:  # 20% max drawdown
                logger.warning("⚠️ Maximum drawdown limit reached")
                return False
            
            # Concentration risk check
            total_exposure = sum(abs(pos) * current_price for pos in positions.values())
            if total_exposure > portfolio_value * 0.8:  # 80% max exposure
                logger.warning("⚠️ Concentration risk limit exceeded")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Risk limit check failed: {e}")
            return False
    
    def calculate_var(self, returns: np.ndarray, confidence_level: float = 0.95) -> float:
        """Calculate Value at Risk"""
        try:
            return np.percentile(returns, (1 - confidence_level) * 100)
        except Exception as e:
            logger.error(f"❌ VaR calculation failed: {e}")
            return 0.0
    
    def calculate_correlation_exposure(self, positions: Dict[str, float]) -> float:
        """Calculate correlation exposure across positions"""
        # Simplified correlation calculation
        # In practice, this would use actual correlation matrices
        if not positions:
            return 0.0
        
        total_exposure = sum(abs(pos) for pos in positions.values())
        return min(1.0, total_exposure / len(positions))


class MarketRegimeAnalysis:
    """
    Market regime analysis for adaptive strategy selection
    """
    
    def __init__(self):
        self.regime_lookback = 60  # Days for regime analysis
        self.volatility_threshold_low = 0.15
        self.volatility_threshold_high = 0.35
        logger.info("📊 Market Regime Analysis initialized")
    
    def identify_market_regime(self, price_data: pd.DataFrame) -> MarketRegime:
        """Identify current market regime"""
        try:
            if len(price_data) < self.regime_lookback:
                return MarketRegime(
                    period_start=price_data.iloc[0]['timestamp'],
                    period_end=price_data.iloc[-1]['timestamp'],
                    regime_type='sideways',
                    volatility_regime='medium',
                    trend_strength=0.0,
                    volatility_level=0.2,
                    regime_confidence=0.5
                )
            
            # Calculate returns
            returns = price_data['close'].pct_change().dropna()
            
            # Trend analysis
            recent_data = price_data.tail(self.regime_lookback)
            price_change = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
            
            # Volatility analysis
            volatility = returns.std() * np.sqrt(252)
            
            # Determine trend regime
            if price_change > 0.1:  # 10% up
                trend_regime = 'bull'
                trend_strength = min(1.0, abs(price_change))
            elif price_change < -0.1:  # 10% down
                trend_regime = 'bear'
                trend_strength = min(1.0, abs(price_change))
            else:
                trend_regime = 'sideways'
                trend_strength = 1.0 - min(1.0, abs(price_change) * 10)  # Inverse for sideways
            
            # Determine volatility regime
            if volatility < self.volatility_threshold_low:
                vol_regime = 'low'
            elif volatility > self.volatility_threshold_high:
                vol_regime = 'high'
            else:
                vol_regime = 'medium'
            
            # Calculate confidence based on consistency of signals
            trend_consistency = self._calculate_trend_consistency(recent_data)
            regime_confidence = trend_consistency
            
            return MarketRegime(
                period_start=recent_data.iloc[0]['timestamp'],
                period_end=recent_data.iloc[-1]['timestamp'],
                regime_type=trend_regime,
                volatility_regime=vol_regime,
                trend_strength=trend_strength,
                volatility_level=volatility,
                regime_confidence=regime_confidence
            )
            
        except Exception as e:
            logger.error(f"❌ Market regime identification failed: {e}")
            return MarketRegime(
                period_start=datetime.now(),
                period_end=datetime.now(),
                regime_type='unknown',
                volatility_regime='medium',
                trend_strength=0.0,
                volatility_level=0.2,
                regime_confidence=0.0
            )
    
    def _calculate_trend_consistency(self, data: pd.DataFrame) -> float:
        """Calculate trend consistency score"""
        try:
            # Calculate moving averages
            short_ma = data['close'].rolling(10).mean()
            long_ma = data['close'].rolling(30).mean()
            
            # Count consistent trend signals
            trend_signals = short_ma > long_ma
            consistency = len(trend_signals[trend_signals == trend_signals.iloc[-1]]) / len(trend_signals)
            
            return consistency
            
        except Exception as e:
            logger.error(f"❌ Trend consistency calculation failed: {e}")
            return 0.5
    
    def analyze_strategy_performance_by_regime(self, backtest_results: List[Dict],
                                             regime_data: List[MarketRegime]) -> Dict[str, Any]:
        """Analyze strategy performance by market regime"""
        try:
            regime_performance = {
                'bull': {'returns': [], 'trades': []},
                'bear': {'returns': [], 'trades': []},
                'sideways': {'returns': [], 'trades': []},
                'low_vol': {'returns': [], 'trades': []},
                'medium_vol': {'returns': [], 'trades': []},
                'high_vol': {'returns': [], 'trades': []}
            }
            
            # Group results by regime
            for result in backtest_results:
                for regime in regime_data:
                    if (result['start_date'] >= regime.period_start and 
                        result['end_date'] <= regime.period_end):
                        
                        # Trend regime
                        regime_performance[regime.regime_type]['returns'].append(result['total_return'])
                        regime_performance[regime.regime_type]['trades'].append(result['trades_count'])
                        
                        # Volatility regime
                        vol_key = f"{regime.volatility_regime}_vol"
                        regime_performance[vol_key]['returns'].append(result['total_return'])
                        regime_performance[vol_key]['trades'].append(result['trades_count'])
            
            # Calculate statistics for each regime
            regime_stats = {}
            for regime_type, data in regime_performance.items():
                if data['returns']:
                    regime_stats[regime_type] = {
                        'avg_return': np.mean(data['returns']),
                        'volatility': np.std(data['returns']),
                        'success_rate': len([r for r in data['returns'] if r > 0]) / len(data['returns']),
                        'avg_trades': np.mean(data['trades']),
                        'periods_count': len(data['returns'])
                    }
            
            return regime_stats
            
        except Exception as e:
            logger.error(f"❌ Regime performance analysis failed: {e}")
            return {}


if __name__ == "__main__":
    print("🏆 Advanced Backtesting System for GoldGPT")
    print("=" * 50)
    
    # Example usage
    config = BacktestConfig(
        initial_capital=100000.0,
        commission_rate=0.0002,
        slippage_bps=0.5
    )
    
    backtester = AdvancedBacktester(config)
    print("✅ Advanced Backtester initialized")
    
    # Example strategy
    def example_strategy(data, short_window=10, long_window=30):
        """Example moving average crossover strategy"""
        if len(data) < long_window:
            return []
        
        short_ma = data['close'].rolling(short_window).mean().iloc[-1]
        long_ma = data['close'].rolling(long_window).mean().iloc[-1]
        prev_short_ma = data['close'].rolling(short_window).mean().iloc[-2]
        prev_long_ma = data['close'].rolling(long_window).mean().iloc[-2]
        
        signals = []
        
        # Bullish crossover
        if short_ma > long_ma and prev_short_ma <= prev_long_ma:
            signals.append({
                'symbol': 'XAUUSD',
                'direction': 'long',
                'confidence': 0.7
            })
        
        # Bearish crossover
        elif short_ma < long_ma and prev_short_ma >= prev_long_ma:
            signals.append({
                'symbol': 'XAUUSD',
                'direction': 'short',
                'confidence': 0.7
            })
        
        return signals
    
    # Add strategy
    backtester.add_strategy('ma_crossover', example_strategy, {'short_window': 10, 'long_window': 30})
    
    print("\n🚀 Advanced Backtesting System ready for:")
    print("  • Walk-forward optimization")
    print("  • Monte Carlo simulation")
    print("  • Comprehensive risk management")
    print("  • Market regime analysis")
    print("  • Professional performance metrics")
