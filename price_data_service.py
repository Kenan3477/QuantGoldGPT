#!/usr/bin/env python3
"""
GoldGPT Price Data Service
Specialized service for OHLCV data with real-time and historical endpoints
"""

import asyncio
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import aiohttp
import pandas as pd
import numpy as np
from data_pipeline_core import data_pipeline, DataType, DataSourceTier

logger = logging.getLogger(__name__)

@dataclass
class OHLCVData:
    """OHLCV data structure"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    source: str
    confidence: float

@dataclass
class PriceAlert:
    """Price alert configuration"""
    symbol: str
    target_price: float
    direction: str  # 'above' or 'below'
    active: bool
    created_at: datetime

class PriceDataService:
    """Advanced price data service with multiple timeframes and sources"""
    
    def __init__(self, db_path: str = "goldgpt_price_data.db"):
        self.db_path = db_path
        self.price_cache = {}
        self.active_alerts = []
        self.supported_timeframes = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M']
        self.initialize_database()
    
    def initialize_database(self):
        """Initialize price data database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # OHLCV data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS ohlcv_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                timeframe TEXT NOT NULL,
                open_price REAL NOT NULL,
                high_price REAL NOT NULL,
                low_price REAL NOT NULL,
                close_price REAL NOT NULL,
                volume REAL DEFAULT 0,
                source TEXT NOT NULL,
                confidence REAL DEFAULT 1.0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timestamp, timeframe, source)
            )
        ''')
        
        # Real-time price table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS realtime_prices (
                symbol TEXT PRIMARY KEY,
                current_price REAL NOT NULL,
                bid REAL,
                ask REAL,
                spread REAL,
                change_24h REAL,
                change_percent_24h REAL,
                high_24h REAL,
                low_24h REAL,
                volume_24h REAL,
                market_cap REAL,
                last_update DATETIME NOT NULL,
                source TEXT NOT NULL,
                confidence REAL DEFAULT 1.0
            )
        ''')
        
        # Price alerts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                target_price REAL NOT NULL,
                direction TEXT NOT NULL,
                active BOOLEAN DEFAULT 1,
                triggered BOOLEAN DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                triggered_at DATETIME
            )
        ''')
        
        # Price history aggregations
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_aggregations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                period_start DATETIME NOT NULL,
                period_end DATETIME NOT NULL,
                avg_price REAL,
                min_price REAL,
                max_price REAL,
                volatility REAL,
                trade_count INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timeframe, period_start)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("✅ Price data database initialized")
    
    async def get_realtime_price(self, symbol: str = 'XAU') -> Optional[Dict]:
        """Get real-time price from best available source"""
        try:
            # Try to get from unified pipeline first
            data = await data_pipeline.get_unified_data(symbol, DataType.PRICE)
            
            if data and not data.get('is_simulated', False):
                price_info = {
                    'symbol': symbol,
                    'price': data['price'],
                    'timestamp': data['timestamp'],
                    'source': data['source'],
                    'confidence': data['confidence'],
                    'bid': data.get('bid'),
                    'ask': data.get('ask'),
                    'spread': data.get('spread'),
                    'change_24h': data.get('change_24h'),
                    'change_percent_24h': data.get('change_percent_24h')
                }
                
                # Store in realtime table
                self.store_realtime_price(price_info)
                return price_info
            
            # Fallback to database if available
            return self.get_latest_stored_price(symbol)
            
        except Exception as e:
            logger.error(f"Error getting realtime price: {e}")
            return self.get_latest_stored_price(symbol)
    
    def store_realtime_price(self, price_info: Dict):
        """Store real-time price in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO realtime_prices 
            (symbol, current_price, bid, ask, spread, change_24h, change_percent_24h,
             high_24h, low_24h, volume_24h, last_update, source, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            price_info['symbol'],
            price_info['price'],
            price_info.get('bid'),
            price_info.get('ask'),
            price_info.get('spread'),
            price_info.get('change_24h'),
            price_info.get('change_percent_24h'),
            price_info.get('high_24h'),
            price_info.get('low_24h'),
            price_info.get('volume_24h'),
            price_info['timestamp'],
            price_info['source'],
            price_info['confidence']
        ))
        
        conn.commit()
        conn.close()
    
    def get_latest_stored_price(self, symbol: str) -> Optional[Dict]:
        """Get latest stored price from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT current_price, bid, ask, spread, change_24h, change_percent_24h,
                   last_update, source, confidence
            FROM realtime_prices 
            WHERE symbol = ?
        ''', (symbol,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'symbol': symbol,
                'price': result[0],
                'bid': result[1],
                'ask': result[2],
                'spread': result[3],
                'change_24h': result[4],
                'change_percent_24h': result[5],
                'timestamp': result[6],
                'source': result[7],
                'confidence': result[8],
                'is_cached': True
            }
        
        return None
    
    async def get_historical_data(self, symbol: str = 'XAU', timeframe: str = '1h', 
                                 limit: int = 100, start_date: datetime = None) -> List[Dict]:
        """Get historical OHLCV data"""
        
        # Check if we have data in database first
        stored_data = self.get_stored_historical_data(symbol, timeframe, limit, start_date)
        
        if len(stored_data) >= limit:
            logger.info(f"📊 Using {len(stored_data)} stored historical points")
            return stored_data
        
        # Need to fetch more data
        logger.info(f"🔄 Fetching historical data for {symbol} {timeframe}")
        
        # Try multiple sources for historical data
        sources_data = await self.fetch_historical_from_sources(symbol, timeframe, limit, start_date)
        
        # Merge and store the data
        if sources_data:
            self.store_historical_data(sources_data, symbol, timeframe)
            return sources_data
        
        # Return what we have in storage
        return stored_data
    
    def get_stored_historical_data(self, symbol: str, timeframe: str, limit: int, start_date: datetime = None) -> List[Dict]:
        """Get historical data from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = '''
            SELECT timestamp, open_price, high_price, low_price, close_price, 
                   volume, source, confidence
            FROM ohlcv_data 
            WHERE symbol = ? AND timeframe = ?
        '''
        params = [symbol, timeframe]
        
        if start_date:
            query += ' AND timestamp >= ?'
            params.append(start_date.isoformat())
        
        query += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        
        historical_data = []
        for row in results:
            historical_data.append({
                'timestamp': row[0],
                'open': row[1],
                'high': row[2],
                'low': row[3],
                'close': row[4],
                'volume': row[5],
                'source': row[6],
                'confidence': row[7]
            })
        
        return list(reversed(historical_data))  # Return chronological order
    
    async def fetch_historical_from_sources(self, symbol: str, timeframe: str, limit: int, start_date: datetime = None) -> List[Dict]:
        """Fetch historical data from multiple sources"""
        historical_data = []
        
        # Try Yahoo Finance first (good for historical data)
        yahoo_data = await self.fetch_yahoo_historical(symbol, timeframe, limit, start_date)
        if yahoo_data:
            historical_data.extend(yahoo_data)
        
        # Try Alpha Vantage for additional data
        av_data = await self.fetch_alpha_vantage_historical(symbol, timeframe, limit, start_date)
        if av_data:
            historical_data.extend(av_data)
        
        # Generate synthetic data if needed (for demo purposes)
        if len(historical_data) < limit // 2:
            synthetic_data = self.generate_synthetic_historical_data(symbol, timeframe, limit, start_date)
            historical_data.extend(synthetic_data)
        
        # Remove duplicates and sort
        seen_timestamps = set()
        unique_data = []
        for item in historical_data:
            if item['timestamp'] not in seen_timestamps:
                seen_timestamps.add(item['timestamp'])
                unique_data.append(item)
        
        unique_data.sort(key=lambda x: x['timestamp'])
        return unique_data[:limit]
    
    async def fetch_yahoo_historical(self, symbol: str, timeframe: str, limit: int, start_date: datetime = None) -> List[Dict]:
        """Fetch historical data from Yahoo Finance"""
        try:
            # Map timeframes to Yahoo intervals
            interval_map = {
                '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
                '1h': '1h', '4h': '4h', '1d': '1d', '1w': '1wk', '1M': '1mo'
            }
            
            interval = interval_map.get(timeframe, '1h')
            
            # Calculate period
            if not start_date:
                if timeframe in ['1m', '5m']:
                    period = '7d'
                elif timeframe in ['15m', '30m', '1h']:
                    period = '60d'
                else:
                    period = '2y'
            else:
                period = 'max'
            
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/GC=F"
            params = {
                'interval': interval,
                'period1': int((start_date or datetime.now() - timedelta(days=365)).timestamp()),
                'period2': int(datetime.now().timestamp())
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if 'chart' in data and data['chart']['result']:
                            result = data['chart']['result'][0]
                            timestamps = result['timestamp']
                            quotes = result['indicators']['quote'][0]
                            
                            historical_data = []
                            for i, timestamp in enumerate(timestamps):
                                if (quotes['open'][i] is not None and 
                                    quotes['high'][i] is not None and
                                    quotes['low'][i] is not None and
                                    quotes['close'][i] is not None):
                                    
                                    historical_data.append({
                                        'timestamp': datetime.fromtimestamp(timestamp).isoformat(),
                                        'open': quotes['open'][i],
                                        'high': quotes['high'][i],
                                        'low': quotes['low'][i],
                                        'close': quotes['close'][i],
                                        'volume': quotes['volume'][i] or 0,
                                        'source': 'yahoo_finance',
                                        'confidence': 0.85
                                    })
                            
                            return historical_data[-limit:]
            
        except Exception as e:
            logger.error(f"Yahoo historical data error: {e}")
        
        return []
    
    async def fetch_alpha_vantage_historical(self, symbol: str, timeframe: str, limit: int, start_date: datetime = None) -> List[Dict]:
        """Fetch historical data from Alpha Vantage"""
        try:
            # Map timeframes to Alpha Vantage functions
            if timeframe in ['1m', '5m', '15m', '30m', '1h']:
                function = 'TIME_SERIES_INTRADAY'
                interval = timeframe
            elif timeframe == '1d':
                function = 'TIME_SERIES_DAILY'
                interval = None
            else:
                function = 'TIME_SERIES_WEEKLY'
                interval = None
            
            params = {
                'function': function,
                'symbol': 'GLD',  # Gold ETF
                'apikey': 'demo',
                'outputsize': 'full'
            }
            
            if interval:
                params['interval'] = interval
            
            url = "https://www.alphavantage.co/query"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Find time series data
                        time_series_key = None
                        for key in data.keys():
                            if 'Time Series' in key:
                                time_series_key = key
                                break
                        
                        if time_series_key and time_series_key in data:
                            time_series = data[time_series_key]
                            historical_data = []
                            
                            for timestamp_str, values in time_series.items():
                                historical_data.append({
                                    'timestamp': timestamp_str,
                                    'open': float(values.get('1. open', 0)),
                                    'high': float(values.get('2. high', 0)),
                                    'low': float(values.get('3. low', 0)),
                                    'close': float(values.get('4. close', 0)),
                                    'volume': float(values.get('5. volume', 0)),
                                    'source': 'alpha_vantage',
                                    'confidence': 0.90
                                })
                            
                            # Convert ETF prices to approximate gold prices
                            for item in historical_data:
                                multiplier = 10.0  # Rough GLD to gold conversion
                                item['open'] *= multiplier
                                item['high'] *= multiplier
                                item['low'] *= multiplier
                                item['close'] *= multiplier
                            
                            return historical_data[-limit:]
            
        except Exception as e:
            logger.error(f"Alpha Vantage historical data error: {e}")
        
        return []
    
    def generate_synthetic_historical_data(self, symbol: str, timeframe: str, limit: int, start_date: datetime = None) -> List[Dict]:
        """Generate synthetic historical data for testing"""
        if not start_date:
            start_date = datetime.now() - timedelta(days=30)
        
        # Timeframe to minutes mapping
        timeframe_minutes = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '4h': 240, '1d': 1440, '1w': 10080, '1M': 43200
        }
        
        minutes = timeframe_minutes.get(timeframe, 60)
        synthetic_data = []
        
        current_time = start_date
        base_price = 3400.0
        
        for i in range(limit):
            # Generate realistic OHLCV data with some randomness
            variation = np.random.normal(0, 10)  # Random walk
            base_price += variation
            
            # Ensure reasonable price range
            base_price = max(3000, min(4000, base_price))
            
            open_price = base_price
            high_price = open_price + abs(np.random.normal(0, 15))
            low_price = open_price - abs(np.random.normal(0, 15))
            close_price = open_price + np.random.normal(0, 8)
            
            # Ensure OHLC logic
            high_price = max(high_price, open_price, close_price)
            low_price = min(low_price, open_price, close_price)
            
            synthetic_data.append({
                'timestamp': current_time.isoformat(),
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': int(np.random.exponential(1000)),
                'source': 'synthetic',
                'confidence': 0.50
            })
            
            current_time += timedelta(minutes=minutes)
            base_price = close_price
        
        return synthetic_data
    
    def store_historical_data(self, historical_data: List[Dict], symbol: str, timeframe: str):
        """Store historical data in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for item in historical_data:
            cursor.execute('''
                INSERT OR REPLACE INTO ohlcv_data 
                (symbol, timestamp, timeframe, open_price, high_price, low_price, 
                 close_price, volume, source, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                item['timestamp'],
                timeframe,
                item['open'],
                item['high'],
                item['low'],
                item['close'],
                item['volume'],
                item['source'],
                item['confidence']
            ))
        
        conn.commit()
        conn.close()
        logger.info(f"📝 Stored {len(historical_data)} historical data points")
    
    def calculate_price_metrics(self, symbol: str = 'XAU', period_hours: int = 24) -> Dict:
        """Calculate price metrics over specified period"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        start_time = (datetime.now() - timedelta(hours=period_hours)).isoformat()
        
        cursor.execute('''
            SELECT close_price, timestamp FROM ohlcv_data 
            WHERE symbol = ? AND timestamp >= ?
            ORDER BY timestamp ASC
        ''', (symbol, start_time))
        
        prices = cursor.fetchall()
        conn.close()
        
        if not prices:
            return {'error': 'No data available for metrics calculation'}
        
        price_values = [p[0] for p in prices]
        
        # Calculate metrics
        current_price = price_values[-1]
        start_price = price_values[0]
        high_price = max(price_values)
        low_price = min(price_values)
        
        change = current_price - start_price
        change_percent = (change / start_price) * 100
        
        # Calculate volatility (standard deviation)
        volatility = np.std(price_values) if len(price_values) > 1 else 0
        
        # Calculate moving averages
        ma_20 = np.mean(price_values[-20:]) if len(price_values) >= 20 else current_price
        ma_50 = np.mean(price_values[-50:]) if len(price_values) >= 50 else current_price
        
        return {
            'symbol': symbol,
            'current_price': current_price,
            'period_hours': period_hours,
            'change': round(change, 2),
            'change_percent': round(change_percent, 3),
            'high': high_price,
            'low': low_price,
            'volatility': round(volatility, 2),
            'ma_20': round(ma_20, 2),
            'ma_50': round(ma_50, 2),
            'data_points': len(prices),
            'last_update': datetime.now().isoformat()
        }
    
    def add_price_alert(self, symbol: str, target_price: float, direction: str) -> int:
        """Add price alert"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO price_alerts (symbol, target_price, direction)
            VALUES (?, ?, ?)
        ''', (symbol, target_price, direction))
        
        alert_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        logger.info(f"🔔 Added price alert: {symbol} {direction} ${target_price}")
        return alert_id
    
    def check_price_alerts(self, current_price: float, symbol: str = 'XAU') -> List[Dict]:
        """Check and trigger price alerts"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, target_price, direction FROM price_alerts 
            WHERE symbol = ? AND active = 1 AND triggered = 0
        ''', (symbol,))
        
        alerts = cursor.fetchall()
        triggered_alerts = []
        
        for alert_id, target_price, direction in alerts:
            should_trigger = False
            
            if direction == 'above' and current_price >= target_price:
                should_trigger = True
            elif direction == 'below' and current_price <= target_price:
                should_trigger = True
            
            if should_trigger:
                cursor.execute('''
                    UPDATE price_alerts 
                    SET triggered = 1, triggered_at = ?
                    WHERE id = ?
                ''', (datetime.now().isoformat(), alert_id))
                
                triggered_alerts.append({
                    'alert_id': alert_id,
                    'symbol': symbol,
                    'target_price': target_price,
                    'current_price': current_price,
                    'direction': direction,
                    'triggered_at': datetime.now().isoformat()
                })
        
        conn.commit()
        conn.close()
        
        if triggered_alerts:
            logger.info(f"🚨 Triggered {len(triggered_alerts)} price alerts")
        
        return triggered_alerts

# Global instance
price_service = PriceDataService()

if __name__ == "__main__":
    # Test the price service
    async def test_price_service():
        print("🧪 Testing Price Data Service...")
        
        # Test real-time price
        price = await price_service.get_realtime_price('XAU')
        print(f"💰 Real-time price: {price}")
        
        # Test historical data
        historical = await price_service.get_historical_data('XAU', '1h', 24)
        print(f"📊 Historical data points: {len(historical)}")
        
        # Test price metrics
        metrics = price_service.calculate_price_metrics('XAU', 24)
        print(f"📈 Price metrics: {metrics}")
        
        # Test price alert
        if price:
            alert_id = price_service.add_price_alert('XAU', price['price'] + 10, 'above')
            print(f"🔔 Added price alert: {alert_id}")
    
    asyncio.run(test_price_service())
