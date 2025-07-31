#!/usr/bin/env python3
"""
Advanced Backtesting Framework for GoldGPT ML Prediction System
Simulates historical market conditions to validate and improve prediction models
"""

import logging
import json
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import sqlite3
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
import statistics

from prediction_tracker import PredictionTracker, PredictionRecord
from learning_engine import LearningEngine

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BacktestConfig:
    """Configuration for backtesting runs"""
    start_date: datetime
    end_date: datetime
    symbols: List[str]
    timeframes: List[str]
    strategies: List[str]
    initial_capital: float = 10000.0
    max_risk_per_trade: float = 0.02  # 2% risk per trade
    commission_per_trade: float = 2.0
    slippage_pips: float = 1.0
    market_hours_only: bool = True
    exclude_news_events: bool = False

@dataclass
class BacktestResult:
    """Results from a backtest run"""
    config: BacktestConfig
    start_date: datetime
    end_date: datetime
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    total_pnl_percent: float
    max_drawdown: float
    max_drawdown_percent: float
    sharpe_ratio: float
    sortino_ratio: float
    profit_factor: float
    average_win: float
    average_loss: float
    largest_win: float
    largest_loss: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    trades: List[Dict[str, Any]]
    equity_curve: List[Dict[str, Any]]
    monthly_returns: Dict[str, float]
    strategy_performance: Dict[str, Dict[str, Any]]

class HistoricalDataManager:
    """Manages historical market data for backtesting"""
    
    def __init__(self, cache_dir: str = "backtest_data_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.data_cache = {}
    
    async def get_historical_data(self, symbol: str, start_date: datetime, 
                                end_date: datetime, interval: str = '1h') -> Optional[pd.DataFrame]:
        """
        Get historical OHLCV data for backtesting
        
        Args:
            symbol: Trading symbol (e.g., 'GC=F' for gold futures)
            start_date: Start date for data
            end_date: End date for data
            interval: Data interval (1m, 5m, 15m, 30m, 1h, 1d)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            cache_key = f"{symbol}_{interval}_{start_date.date()}_{end_date.date()}"
            
            # Check cache first
            if cache_key in self.data_cache:
                logger.info(f"Using cached data for {cache_key}")
                return self.data_cache[cache_key]
            
            cache_file = self.cache_dir / f"{cache_key}.csv"
            
            if cache_file.exists():
                logger.info(f"Loading data from cache file: {cache_file}")
                df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
                self.data_cache[cache_key] = df
                return df
            
            # Fetch from Yahoo Finance
            logger.info(f"Fetching historical data for {symbol} from {start_date} to {end_date}")
            
            # Convert symbol for yfinance (XAUUSD -> GC=F)
            yf_symbol = self._convert_symbol_for_yfinance(symbol)
            
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if df.empty:
                logger.warning(f"No data found for {symbol}")
                return None
            
            # Save to cache
            df.to_csv(cache_file)
            self.data_cache[cache_key] = df
            
            logger.info(f"Fetched {len(df)} records for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to get historical data for {symbol}: {e}")
            return None
    
    def _convert_symbol_for_yfinance(self, symbol: str) -> str:
        """Convert trading symbols to Yahoo Finance format"""
        symbol_mapping = {
            'XAUUSD': 'GC=F',  # Gold futures
            'EURUSD': 'EURUSD=X',
            'GBPUSD': 'GBPUSD=X',
            'USDJPY': 'USDJPY=X',
            'BTCUSD': 'BTC-USD',
            'ETHUSD': 'ETH-USD'
        }
        return symbol_mapping.get(symbol, symbol)
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for historical data"""
        try:
            # Simple Moving Averages
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
            
            # Exponential Moving Averages
            df['EMA_12'] = df['Close'].ewm(span=12).mean()
            df['EMA_26'] = df['Close'].ewm(span=26).mean()
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            
            # Bollinger Bands
            bb_period = 20
            bb_std = 2
            df['BB_Middle'] = df['Close'].rolling(window=bb_period).mean()
            bb_std_val = df['Close'].rolling(window=bb_period).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std_val * bb_std)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std_val * bb_std)
            df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
            
            # Average True Range (ATR)
            high_low = df['High'] - df['Low']
            high_close = np.abs(df['High'] - df['Close'].shift())
            low_close = np.abs(df['Low'] - df['Close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df['ATR'] = true_range.rolling(window=14).mean()
            
            # Stochastic
            low_14 = df['Low'].rolling(window=14).min()
            high_14 = df['High'].rolling(window=14).max()
            df['Stochastic_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
            df['Stochastic_D'] = df['Stochastic_K'].rolling(window=3).mean()
            
            # Volume indicators (if volume data available)
            if 'Volume' in df.columns:
                df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
                df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to calculate technical indicators: {e}")
            return df

class BacktestEngine:
    """
    Advanced backtesting engine for ML prediction strategies
    
    Features:
    - Historical market data simulation
    - Realistic trade execution with slippage and commission
    - Multiple strategy backtesting
    - Comprehensive performance metrics
    - Risk management simulation
    - Monte Carlo analysis
    """
    
    def __init__(self, prediction_tracker: PredictionTracker, 
                 data_manager: HistoricalDataManager):
        self.tracker = prediction_tracker
        self.data_manager = data_manager
        self.results_cache = {}
        
    async def run_backtest(self, config: BacktestConfig, 
                          prediction_strategies: Dict[str, Any]) -> BacktestResult:
        """
        Run a comprehensive backtest
        
        Args:
            config: Backtest configuration
            prediction_strategies: Dictionary of strategy functions
            
        Returns:
            BacktestResult with comprehensive metrics
        """
        try:
            logger.info(f"Starting backtest from {config.start_date} to {config.end_date}")
            
            # Initialize tracking variables
            trades = []
            equity_curve = []
            current_capital = config.initial_capital
            peak_capital = config.initial_capital
            max_drawdown = 0.0
            open_positions = []
            
            # Get historical data for all symbols
            historical_data = {}
            for symbol in config.symbols:
                for timeframe in config.timeframes:
                    interval = self._timeframe_to_interval(timeframe)
                    data = await self.data_manager.get_historical_data(
                        symbol, config.start_date, config.end_date, interval
                    )
                    
                    if data is not None:
                        data = self.data_manager.calculate_technical_indicators(data)
                        historical_data[f"{symbol}_{timeframe}"] = data
            
            if not historical_data:
                raise ValueError("No historical data available for backtesting")
            
            # Simulate trading day by day
            current_date = config.start_date
            daily_pnl = []
            
            while current_date <= config.end_date:
                try:
                    # Process each symbol/timeframe combination
                    for symbol in config.symbols:
                        for timeframe in config.timeframes:
                            data_key = f"{symbol}_{timeframe}"
                            
                            if data_key not in historical_data:
                                continue
                            
                            df = historical_data[data_key]
                            
                            # Get market data for current date
                            current_data = self._get_market_data_for_date(df, current_date)
                            
                            if current_data is None:
                                continue
                            
                            # Generate predictions for each strategy
                            for strategy_name in config.strategies:
                                if strategy_name in prediction_strategies:
                                    strategy_func = prediction_strategies[strategy_name]
                                    
                                    # Generate prediction
                                    prediction = await self._generate_backtest_prediction(
                                        strategy_func, current_data, symbol, timeframe, strategy_name
                                    )
                                    
                                    if prediction:
                                        # Simulate trade execution
                                        trade_result = await self._simulate_trade_execution(
                                            prediction, current_data, config
                                        )
                                        
                                        if trade_result:
                                            trades.append(trade_result)
                                            current_capital += trade_result['pnl']
                                            
                                            # Update drawdown
                                            if current_capital > peak_capital:
                                                peak_capital = current_capital
                                            
                                            current_drawdown = (peak_capital - current_capital) / peak_capital
                                            max_drawdown = max(max_drawdown, current_drawdown)
                    
                    # Record daily equity
                    equity_curve.append({
                        'date': current_date.isoformat(),
                        'equity': current_capital,
                        'drawdown': (peak_capital - current_capital) / peak_capital if peak_capital > 0 else 0
                    })
                    
                    # Move to next day
                    current_date += timedelta(days=1)
                    
                except Exception as e:
                    logger.warning(f"Error processing {current_date}: {e}")
                    current_date += timedelta(days=1)
                    continue
            
            # Calculate performance metrics
            result = self._calculate_backtest_metrics(
                config, trades, equity_curve, config.initial_capital, current_capital, max_drawdown
            )
            
            logger.info(f"Backtest completed: {result.total_trades} trades, {result.win_rate:.1f}% win rate")
            
            return result
            
        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            raise
    
    def _timeframe_to_interval(self, timeframe: str) -> str:
        """Convert timeframe to Yahoo Finance interval"""
        mapping = {
            '1M': '1m',
            '5M': '5m',
            '15M': '15m',
            '30M': '30m',
            '1H': '1h',
            '4H': '4h',
            '1D': '1d'
        }
        return mapping.get(timeframe, '1h')
    
    def _get_market_data_for_date(self, df: pd.DataFrame, date: datetime) -> Optional[Dict[str, Any]]:
        """Get market data for a specific date"""
        try:
            # Find the closest date in the data
            date_str = date.strftime('%Y-%m-%d')
            
            # Get data for the date (or closest available)
            available_dates = df.index.date
            target_date = date.date()
            
            # Find closest date
            closest_date = min(available_dates, key=lambda x: abs((x - target_date).days))
            
            if abs((closest_date - target_date).days) > 7:  # More than a week difference
                return None
            
            # Get the row for that date
            mask = df.index.date == closest_date
            data_row = df[mask].iloc[-1] if any(mask) else None
            
            if data_row is None:
                return None
            
            return {
                'timestamp': closest_date,
                'open': data_row['Open'],
                'high': data_row['High'],
                'low': data_row['Low'],
                'close': data_row['Close'],
                'volume': data_row.get('Volume', 0),
                'sma_20': data_row.get('SMA_20', 0),
                'sma_50': data_row.get('SMA_50', 0),
                'rsi': data_row.get('RSI', 50),
                'macd': data_row.get('MACD', 0),
                'macd_signal': data_row.get('MACD_Signal', 0),
                'bb_upper': data_row.get('BB_Upper', data_row['Close']),
                'bb_lower': data_row.get('BB_Lower', data_row['Close']),
                'atr': data_row.get('ATR', 1.0),
                'stochastic_k': data_row.get('Stochastic_K', 50),
                'stochastic_d': data_row.get('Stochastic_D', 50)
            }
            
        except Exception as e:
            logger.warning(f"Failed to get market data for {date}: {e}")
            return None
    
    async def _generate_backtest_prediction(self, strategy_func: callable, market_data: Dict[str, Any],
                                          symbol: str, timeframe: str, strategy_name: str) -> Optional[Dict[str, Any]]:
        """Generate a prediction using a strategy function"""
        try:
            # Call the strategy function with market data
            prediction = await strategy_func(market_data, symbol, timeframe)
            
            if prediction and isinstance(prediction, dict):
                return {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'strategy': strategy_name,
                    'direction': prediction.get('direction', 'neutral'),
                    'confidence': prediction.get('confidence', 0.5),
                    'entry_price': market_data['close'],
                    'stop_loss': prediction.get('stop_loss'),
                    'take_profit': prediction.get('take_profit'),
                    'timestamp': market_data['timestamp']
                }
            
            return None
            
        except Exception as e:
            logger.warning(f"Strategy {strategy_name} failed: {e}")
            return None
    
    async def _simulate_trade_execution(self, prediction: Dict[str, Any], 
                                      market_data: Dict[str, Any], 
                                      config: BacktestConfig) -> Optional[Dict[str, Any]]:
        """Simulate realistic trade execution with slippage and commission"""
        try:
            if prediction['direction'] == 'neutral':
                return None
            
            entry_price = prediction['entry_price']
            
            # Apply slippage
            slippage = config.slippage_pips * 0.01  # Convert pips to price units
            if prediction['direction'] == 'bullish':
                actual_entry_price = entry_price + slippage
            else:
                actual_entry_price = entry_price - slippage
            
            # Calculate position size based on risk management
            risk_amount = config.initial_capital * config.max_risk_per_trade
            stop_loss_distance = abs(actual_entry_price - prediction.get('stop_loss', actual_entry_price * 0.99))
            
            if stop_loss_distance > 0:
                position_size = risk_amount / stop_loss_distance
            else:
                position_size = config.initial_capital * 0.01  # 1% of capital as fallback
            
            # Simulate holding period (simplified - hold for timeframe duration)
            holding_hours = self._get_holding_hours(prediction['timeframe'])
            
            # Simulate exit (simplified - assume we hit target or stop)
            exit_price = self._simulate_exit_price(
                actual_entry_price, prediction, market_data, holding_hours
            )
            
            # Calculate P&L
            if prediction['direction'] == 'bullish':
                pnl = (exit_price - actual_entry_price) * position_size
            else:
                pnl = (actual_entry_price - exit_price) * position_size
            
            # Apply commission
            pnl -= (2 * config.commission_per_trade)  # Entry and exit commission
            
            # Determine if trade was a winner
            is_winner = pnl > 0
            
            return {
                'timestamp': prediction['timestamp'],
                'symbol': prediction['symbol'],
                'timeframe': prediction['timeframe'],
                'strategy': prediction['strategy'],
                'direction': prediction['direction'],
                'confidence': prediction['confidence'],
                'entry_price': actual_entry_price,
                'exit_price': exit_price,
                'position_size': position_size,
                'pnl': pnl,
                'pnl_percent': (pnl / config.initial_capital) * 100,
                'is_winner': is_winner,
                'holding_hours': holding_hours
            }
            
        except Exception as e:
            logger.warning(f"Trade simulation failed: {e}")
            return None
    
    def _get_holding_hours(self, timeframe: str) -> int:
        """Get typical holding period for a timeframe"""
        mapping = {
            '1M': 1,
            '5M': 5,
            '15M': 15,
            '30M': 30,
            '1H': 1,
            '4H': 4,
            '1D': 24
        }
        return mapping.get(timeframe, 1)
    
    def _simulate_exit_price(self, entry_price: float, prediction: Dict[str, Any], 
                           market_data: Dict[str, Any], holding_hours: int) -> float:
        """Simulate exit price (simplified model)"""
        # Simplified: randomly determine if we hit TP, SL, or exit at market
        import random
        
        take_profit = prediction.get('take_profit')
        stop_loss = prediction.get('stop_loss')
        
        # Random outcome based on confidence
        confidence = prediction['confidence']
        hit_tp_probability = confidence * 0.7  # Higher confidence = higher TP probability
        
        random_outcome = random.random()
        
        if take_profit and random_outcome < hit_tp_probability:
            return take_profit
        elif stop_loss and random_outcome < hit_tp_probability + 0.3:
            return stop_loss
        else:
            # Exit at market price with some random movement
            volatility = market_data.get('atr', 1.0)
            price_movement = np.random.normal(0, volatility * 0.5)
            return entry_price + price_movement
    
    def _calculate_backtest_metrics(self, config: BacktestConfig, trades: List[Dict[str, Any]], 
                                   equity_curve: List[Dict[str, Any]], 
                                   initial_capital: float, final_capital: float,
                                   max_drawdown: float) -> BacktestResult:
        """Calculate comprehensive backtest performance metrics"""
        
        if not trades:
            # Return empty result if no trades
            return BacktestResult(
                config=config,
                start_date=config.start_date,
                end_date=config.end_date,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                total_pnl=0.0,
                total_pnl_percent=0.0,
                max_drawdown=0.0,
                max_drawdown_percent=0.0,
                sharpe_ratio=0.0,
                sortino_ratio=0.0,
                profit_factor=0.0,
                average_win=0.0,
                average_loss=0.0,
                largest_win=0.0,
                largest_loss=0.0,
                max_consecutive_wins=0,
                max_consecutive_losses=0,
                trades=[],
                equity_curve=[],
                monthly_returns={},
                strategy_performance={}
            )
        
        # Basic metrics
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t['is_winner'])
        losing_trades = total_trades - winning_trades
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # P&L metrics
        total_pnl = sum(t['pnl'] for t in trades)
        total_pnl_percent = ((final_capital - initial_capital) / initial_capital) * 100
        
        # Win/Loss analysis
        wins = [t['pnl'] for t in trades if t['is_winner']]
        losses = [t['pnl'] for t in trades if not t['is_winner']]
        
        average_win = statistics.mean(wins) if wins else 0
        average_loss = statistics.mean(losses) if losses else 0
        largest_win = max(wins) if wins else 0
        largest_loss = min(losses) if losses else 0
        
        # Profit factor
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Consecutive wins/losses
        max_consecutive_wins = self._calculate_max_consecutive(trades, True)
        max_consecutive_losses = self._calculate_max_consecutive(trades, False)
        
        # Risk metrics
        daily_returns = self._calculate_daily_returns(equity_curve)
        sharpe_ratio = self._calculate_sharpe_ratio(daily_returns)
        sortino_ratio = self._calculate_sortino_ratio(daily_returns)
        
        # Monthly returns
        monthly_returns = self._calculate_monthly_returns(equity_curve)
        
        # Strategy performance breakdown
        strategy_performance = self._calculate_strategy_performance(trades)
        
        return BacktestResult(
            config=config,
            start_date=config.start_date,
            end_date=config.end_date,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=round(win_rate, 2),
            total_pnl=round(total_pnl, 2),
            total_pnl_percent=round(total_pnl_percent, 2),
            max_drawdown=round(max_drawdown * initial_capital, 2),
            max_drawdown_percent=round(max_drawdown * 100, 2),
            sharpe_ratio=round(sharpe_ratio, 3),
            sortino_ratio=round(sortino_ratio, 3),
            profit_factor=round(profit_factor, 2),
            average_win=round(average_win, 2),
            average_loss=round(average_loss, 2),
            largest_win=round(largest_win, 2),
            largest_loss=round(largest_loss, 2),
            max_consecutive_wins=max_consecutive_wins,
            max_consecutive_losses=max_consecutive_losses,
            trades=trades,
            equity_curve=equity_curve,
            monthly_returns=monthly_returns,
            strategy_performance=strategy_performance
        )
    
    def _calculate_max_consecutive(self, trades: List[Dict[str, Any]], winners: bool) -> int:
        """Calculate maximum consecutive wins or losses"""
        max_consecutive = 0
        current_consecutive = 0
        
        for trade in trades:
            if (trade['is_winner'] and winners) or (not trade['is_winner'] and not winners):
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _calculate_daily_returns(self, equity_curve: List[Dict[str, Any]]) -> List[float]:
        """Calculate daily returns from equity curve"""
        if len(equity_curve) < 2:
            return []
        
        returns = []
        for i in range(1, len(equity_curve)):
            prev_equity = equity_curve[i-1]['equity']
            curr_equity = equity_curve[i]['equity']
            
            if prev_equity > 0:
                daily_return = (curr_equity - prev_equity) / prev_equity
                returns.append(daily_return)
        
        return returns
    
    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if not returns or len(returns) < 2:
            return 0.0
        
        mean_return = statistics.mean(returns)
        std_return = statistics.stdev(returns)
        
        if std_return == 0:
            return 0.0
        
        # Annualized Sharpe ratio (assuming daily returns)
        excess_return = mean_return * 252 - risk_free_rate  # 252 trading days
        volatility = std_return * np.sqrt(252)
        
        return excess_return / volatility if volatility > 0 else 0.0
    
    def _calculate_sortino_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio (focuses on downside risk)"""
        if not returns or len(returns) < 2:
            return 0.0
        
        mean_return = statistics.mean(returns)
        negative_returns = [r for r in returns if r < 0]
        
        if not negative_returns:
            return float('inf')  # No downside risk
        
        downside_std = statistics.stdev(negative_returns)
        
        # Annualized Sortino ratio
        excess_return = mean_return * 252 - risk_free_rate
        downside_volatility = downside_std * np.sqrt(252)
        
        return excess_return / downside_volatility if downside_volatility > 0 else 0.0
    
    def _calculate_monthly_returns(self, equity_curve: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate monthly returns"""
        if not equity_curve:
            return {}
        
        monthly_returns = {}
        current_month = None
        month_start_equity = None
        
        for point in equity_curve:
            date = datetime.fromisoformat(point['date'])
            month_key = date.strftime('%Y-%m')
            
            if current_month != month_key:
                if current_month and month_start_equity:
                    # Calculate return for previous month
                    prev_point = equity_curve[equity_curve.index(point) - 1]
                    month_return = ((prev_point['equity'] - month_start_equity) / month_start_equity) * 100
                    monthly_returns[current_month] = round(month_return, 2)
                
                current_month = month_key
                month_start_equity = point['equity']
        
        # Handle last month
        if current_month and month_start_equity and equity_curve:
            last_equity = equity_curve[-1]['equity']
            month_return = ((last_equity - month_start_equity) / month_start_equity) * 100
            monthly_returns[current_month] = round(month_return, 2)
        
        return monthly_returns
    
    def _calculate_strategy_performance(self, trades: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Calculate performance metrics by strategy"""
        strategy_stats = {}
        
        for trade in trades:
            strategy = trade['strategy']
            
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {
                    'trades': [],
                    'wins': 0,
                    'losses': 0,
                    'total_pnl': 0
                }
            
            strategy_stats[strategy]['trades'].append(trade)
            strategy_stats[strategy]['total_pnl'] += trade['pnl']
            
            if trade['is_winner']:
                strategy_stats[strategy]['wins'] += 1
            else:
                strategy_stats[strategy]['losses'] += 1
        
        # Calculate metrics for each strategy
        performance = {}
        for strategy, stats in strategy_stats.items():
            total_trades = len(stats['trades'])
            win_rate = (stats['wins'] / total_trades) * 100 if total_trades > 0 else 0
            
            performance[strategy] = {
                'total_trades': total_trades,
                'wins': stats['wins'],
                'losses': stats['losses'],
                'win_rate': round(win_rate, 2),
                'total_pnl': round(stats['total_pnl'], 2),
                'avg_pnl_per_trade': round(stats['total_pnl'] / total_trades, 2) if total_trades > 0 else 0
            }
        
        return performance
    
    async def run_monte_carlo_analysis(self, config: BacktestConfig, 
                                     prediction_strategies: Dict[str, Any],
                                     num_simulations: int = 100) -> Dict[str, Any]:
        """Run Monte Carlo analysis for robustness testing"""
        try:
            logger.info(f"Starting Monte Carlo analysis with {num_simulations} simulations")
            
            simulation_results = []
            
            for i in range(num_simulations):
                # Randomize some parameters for each simulation
                modified_config = self._randomize_config(config)
                
                # Run backtest
                result = await self.run_backtest(modified_config, prediction_strategies)
                
                simulation_results.append({
                    'simulation': i + 1,
                    'total_pnl_percent': result.total_pnl_percent,
                    'win_rate': result.win_rate,
                    'max_drawdown_percent': result.max_drawdown_percent,
                    'sharpe_ratio': result.sharpe_ratio,
                    'total_trades': result.total_trades
                })
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Completed {i + 1}/{num_simulations} simulations")
            
            # Analyze results
            pnl_results = [r['total_pnl_percent'] for r in simulation_results]
            win_rates = [r['win_rate'] for r in simulation_results]
            drawdowns = [r['max_drawdown_percent'] for r in simulation_results]
            sharpe_ratios = [r['sharpe_ratio'] for r in simulation_results]
            
            analysis = {
                'num_simulations': num_simulations,
                'total_pnl_percent': {
                    'mean': round(statistics.mean(pnl_results), 2),
                    'median': round(statistics.median(pnl_results), 2),
                    'std': round(statistics.stdev(pnl_results) if len(pnl_results) > 1 else 0, 2),
                    'min': round(min(pnl_results), 2),
                    'max': round(max(pnl_results), 2),
                    'percentile_5': round(np.percentile(pnl_results, 5), 2),
                    'percentile_95': round(np.percentile(pnl_results, 95), 2)
                },
                'win_rate': {
                    'mean': round(statistics.mean(win_rates), 2),
                    'median': round(statistics.median(win_rates), 2),
                    'std': round(statistics.stdev(win_rates) if len(win_rates) > 1 else 0, 2),
                    'min': round(min(win_rates), 2),
                    'max': round(max(win_rates), 2)
                },
                'max_drawdown_percent': {
                    'mean': round(statistics.mean(drawdowns), 2),
                    'median': round(statistics.median(drawdowns), 2),
                    'std': round(statistics.stdev(drawdowns) if len(drawdowns) > 1 else 0, 2),
                    'min': round(min(drawdowns), 2),
                    'max': round(max(drawdowns), 2)
                },
                'sharpe_ratio': {
                    'mean': round(statistics.mean(sharpe_ratios), 2),
                    'median': round(statistics.median(sharpe_ratios), 2),
                    'std': round(statistics.stdev(sharpe_ratios) if len(sharpe_ratios) > 1 else 0, 2),
                    'min': round(min(sharpe_ratios), 2),
                    'max': round(max(sharpe_ratios), 2)
                },
                'probability_of_profit': len([r for r in pnl_results if r > 0]) / num_simulations * 100,
                'simulation_results': simulation_results
            }
            
            logger.info(f"Monte Carlo analysis completed. Mean P&L: {analysis['total_pnl_percent']['mean']:.2f}%")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Monte Carlo analysis failed: {e}")
            return {'error': str(e)}
    
    def _randomize_config(self, base_config: BacktestConfig) -> BacktestConfig:
        """Create a randomized version of the config for Monte Carlo"""
        import copy
        import random
        
        config = copy.deepcopy(base_config)
        
        # Randomize some parameters within reasonable ranges
        config.max_risk_per_trade *= random.uniform(0.5, 1.5)
        config.commission_per_trade *= random.uniform(0.8, 1.2)
        config.slippage_pips *= random.uniform(0.5, 2.0)
        
        return config

# Factory functions and example usage
def create_backtest_engine(prediction_tracker: PredictionTracker) -> BacktestEngine:
    """Factory function to create a BacktestEngine"""
    data_manager = HistoricalDataManager()
    return BacktestEngine(prediction_tracker, data_manager)

async def example_strategy_function(market_data: Dict[str, Any], symbol: str, timeframe: str) -> Dict[str, Any]:
    """Example strategy function for backtesting"""
    # Simple RSI strategy
    rsi = market_data.get('rsi', 50)
    current_price = market_data['close']
    
    if rsi < 30:  # Oversold
        return {
            'direction': 'bullish',
            'confidence': 0.7,
            'stop_loss': current_price * 0.98,
            'take_profit': current_price * 1.02
        }
    elif rsi > 70:  # Overbought
        return {
            'direction': 'bearish',
            'confidence': 0.7,
            'stop_loss': current_price * 1.02,
            'take_profit': current_price * 0.98
        }
    
    return {'direction': 'neutral'}

if __name__ == "__main__":
    # Example usage
    async def test_backtesting():
        from prediction_tracker import PredictionTracker
        
        tracker = PredictionTracker()
        engine = create_backtest_engine(tracker)
        
        # Configure backtest
        config = BacktestConfig(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            symbols=['XAUUSD'],
            timeframes=['1H'],
            strategies=['rsi_strategy'],
            initial_capital=10000.0
        )
        
        # Define strategies
        strategies = {
            'rsi_strategy': example_strategy_function
        }
        
        # Run backtest
        result = await engine.run_backtest(config, strategies)
        
        print(f"Backtest Results:")
        print(f"Total Trades: {result.total_trades}")
        print(f"Win Rate: {result.win_rate}%")
        print(f"Total P&L: {result.total_pnl_percent:.2f}%")
        print(f"Max Drawdown: {result.max_drawdown_percent:.2f}%")
        print(f"Sharpe Ratio: {result.sharpe_ratio:.3f}")
    
    # Run the test
    asyncio.run(test_backtesting())
