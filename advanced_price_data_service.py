#!/usr/bin/env python3
"""
GoldGPT Advanced Price Data Service
Advanced OHLCV data management with real-time and historical endpoints
"""

import asyncio
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import aiohttp
import time
import numpy as np
from data_pipeline_core import DataPipelineCore, DataType, DataSourceTier

logger = logging.getLogger(__name__)

@dataclass
class OHLCVData:
    """OHLCV (Open, High, Low, Close, Volume) data structure"""
    symbol: str
    timestamp: datetime
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    volume: int
    source: str
    timeframe: str  # 1m, 5m, 15m, 1h, 4h, 1d
    confidence: float

@dataclass
class PriceAlert:
    """Price alert configuration"""
    symbol: str
    price_target: float
    direction: str  # 'above' or 'below'
    active: bool
    created_at: datetime

class AdvancedPriceDataService:
    """Advanced price data service with OHLCV support"""
    
    def __init__(self, pipeline: DataPipelineCore, db_path: str = "goldgpt_advanced_price_data.db"):
        self.pipeline = pipeline
        self.db_path = db_path
        self.active_subscriptions = set()
        self.price_alerts = []
        self.real_time_callbacks = []
        
        # Timeframe configurations
        self.timeframes = {
            '1m': 60,       # 1 minute
            '5m': 300,      # 5 minutes
            '15m': 900,     # 15 minutes
            '1h': 3600,     # 1 hour
            '4h': 14400,    # 4 hours
            '1d': 86400     # 1 day
        }
        
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
                volume INTEGER DEFAULT 0,
                source TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timestamp, timeframe, source)
            )
        ''')
        
        # Real-time price tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS real_time_prices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                price REAL NOT NULL,
                bid REAL,
                ask REAL,
                spread REAL,
                volume INTEGER DEFAULT 0,
                source TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Price alerts
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                price_target REAL NOT NULL,
                direction TEXT NOT NULL,
                active BOOLEAN DEFAULT 1,
                triggered_at DATETIME,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Price statistics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS price_statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                date DATE NOT NULL,
                avg_price REAL,
                min_price REAL,
                max_price REAL,
                volatility REAL,
                volume_avg INTEGER,
                price_change_pct REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, timeframe, date)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("✅ Advanced price data database initialized")
    
    async def get_real_time_price(self, symbol: str) -> Optional[Dict]:
        """Get current real-time price with bid/ask spread"""
        try:
            # Try multiple sources for best price
            data = await self.pipeline.get_unified_data(symbol, DataType.PRICE)
            
            if data:
                price = data.get('price', 0)
                
                # Calculate synthetic bid/ask spread (0.1% typical for gold)
                spread = price * 0.001
                bid = price - (spread / 2)
                ask = price + (spread / 2)
                
                price_data = {
                    'symbol': symbol,
                    'price': price,
                    'bid': bid,
                    'ask': ask,
                    'spread': spread,
                    'timestamp': data.get('timestamp', datetime.now().isoformat()),
                    'source': data.get('source', 'unknown'),
                    'confidence': data.get('confidence', 0.5)
                }
                
                # Store in database
                await self.store_real_time_price(price_data)
                
                # Check price alerts
                await self.check_price_alerts(symbol, price)
                
                # Notify subscribers
                await self.notify_price_update(symbol, price_data)
                
                return price_data
                
        except Exception as e:
            logger.error(f"Error getting real-time price for {symbol}: {e}")
        
        return None
    
    async def store_real_time_price(self, price_data: Dict):
        """Store real-time price in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO real_time_prices 
            (symbol, price, bid, ask, spread, source, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            price_data['symbol'],
            price_data['price'],
            price_data['bid'],
            price_data['ask'],
            price_data['spread'],
            price_data['source'],
            price_data['timestamp']
        ))
        
        conn.commit()
        conn.close()
    
    async def get_historical_ohlcv(self, symbol: str, timeframe: str, 
                                   start_time: datetime, end_time: datetime,
                                   limit: int = 1000) -> List[OHLCVData]:
        """Get historical OHLCV data for specified timeframe"""
        
        # First check database
        stored_data = await self.get_stored_ohlcv(symbol, timeframe, start_time, end_time)
        
        # If we have recent data, return it
        if stored_data and len(stored_data) > 0:
            latest_timestamp = max(d.timestamp for d in stored_data)
            if datetime.now() - latest_timestamp < timedelta(hours=1):
                return stored_data[:limit]
        
        # Fetch fresh data from external sources
        fresh_data = await self.fetch_external_ohlcv(symbol, timeframe, start_time, end_time)
        
        if fresh_data:
            # Store in database
            await self.store_ohlcv_data(fresh_data)
            return fresh_data[:limit]
        
        # Return stored data as fallback
        return stored_data[:limit] if stored_data else []
    
    async def get_stored_ohlcv(self, symbol: str, timeframe: str, 
                               start_time: datetime, end_time: datetime) -> List[OHLCVData]:
        """Get OHLCV data from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT symbol, timestamp, timeframe, open_price, high_price, 
                   low_price, close_price, volume, source, confidence
            FROM ohlcv_data
            WHERE symbol = ? AND timeframe = ? 
            AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp DESC
        ''', (symbol, timeframe, start_time.isoformat(), end_time.isoformat()))
        
        results = []
        for row in cursor.fetchall():
            results.append(OHLCVData(
                symbol=row[0],
                timestamp=datetime.fromisoformat(row[1]),
                timeframe=row[2],
                open_price=row[3],
                high_price=row[4],
                low_price=row[5],
                close_price=row[6],
                volume=row[7],
                source=row[8],
                confidence=row[9]
            ))
        
        conn.close()
        return results
    
    async def fetch_external_ohlcv(self, symbol: str, timeframe: str, 
                                   start_time: datetime, end_time: datetime) -> List[OHLCVData]:
        """Fetch OHLCV data from external sources"""
        try:
            # For demonstration, generate synthetic OHLCV data
            # In production, this would fetch from Alpha Vantage, Yahoo Finance, etc.
            
            interval_seconds = self.timeframes.get(timeframe, 3600)
            current_time = start_time
            ohlcv_data = []
            
            base_price = 3400  # Base gold price
            
            while current_time <= end_time and len(ohlcv_data) < 500:
                # Generate realistic OHLCV data
                price_variation = np.random.normal(0, 5)  # Small random variation
                
                open_price = base_price + price_variation
                high_variation = abs(np.random.normal(0, 8))
                low_variation = abs(np.random.normal(0, 8))
                
                high_price = open_price + high_variation
                low_price = open_price - low_variation
                
                close_variation = np.random.normal(0, 3)
                close_price = open_price + close_variation
                
                # Ensure OHLC logic is maintained
                high_price = max(high_price, open_price, close_price)
                low_price = min(low_price, open_price, close_price)
                
                volume = int(np.random.normal(1000, 300))  # Synthetic volume
                
                ohlcv_data.append(OHLCVData(
                    symbol=symbol,
                    timestamp=current_time,
                    timeframe=timeframe,
                    open_price=round(open_price, 2),
                    high_price=round(high_price, 2),
                    low_price=round(low_price, 2),
                    close_price=round(close_price, 2),
                    volume=max(volume, 0),
                    source='synthetic_generation',
                    confidence=0.75
                ))
                
                current_time += timedelta(seconds=interval_seconds)
                base_price = close_price  # Next candle starts where this one ended
            
            return ohlcv_data
            
        except Exception as e:
            logger.error(f"Error fetching external OHLCV data: {e}")
            return []
    
    async def store_ohlcv_data(self, ohlcv_data: List[OHLCVData]):
        """Store OHLCV data in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for data in ohlcv_data:
            cursor.execute('''
                INSERT OR IGNORE INTO ohlcv_data
                (symbol, timestamp, timeframe, open_price, high_price, 
                 low_price, close_price, volume, source, confidence)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data.symbol,
                data.timestamp.isoformat(),
                data.timeframe,
                data.open_price,
                data.high_price,
                data.low_price,
                data.close_price,
                data.volume,
                data.source,
                data.confidence
            ))
        
        conn.commit()
        conn.close()
        logger.info(f"✅ Stored {len(ohlcv_data)} OHLCV records")
    
    async def get_market_summary(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive market summary"""
        try:
            # Current price
            current_data = await self.get_real_time_price(symbol)
            
            # Historical data for comparison
            end_time = datetime.now()
            start_time = end_time - timedelta(days=1)
            
            daily_data = await self.get_historical_ohlcv(symbol, '1h', start_time, end_time)
            
            if not daily_data:
                return {'error': 'No market data available'}
            
            # Calculate 24h change
            prices_24h = [d.close_price for d in daily_data]
            price_change_24h = 0
            price_change_pct_24h = 0
            
            if len(prices_24h) >= 2:
                price_change_24h = prices_24h[0] - prices_24h[-1]
                price_change_pct_24h = (price_change_24h / prices_24h[-1]) * 100
            
            # Calculate volatility
            volatility_24h = np.std(prices_24h) if len(prices_24h) > 1 else 0
            
            summary = {
                'symbol': symbol,
                'current_price': current_data['price'] if current_data else 0,
                'bid': current_data['bid'] if current_data else 0,
                'ask': current_data['ask'] if current_data else 0,
                'spread': current_data['spread'] if current_data else 0,
                'price_change_24h': round(price_change_24h, 2),
                'price_change_pct_24h': round(price_change_pct_24h, 3),
                'volatility_24h': round(volatility_24h, 2),
                'high_24h': max(prices_24h) if prices_24h else 0,
                'low_24h': min(prices_24h) if prices_24h else 0,
                'volume_24h': sum(d.volume for d in daily_data),
                'last_updated': datetime.now().isoformat(),
                'data_points': len(daily_data)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating market summary: {e}")
            return {'error': str(e)}

# Global instance
advanced_price_service = AdvancedPriceDataService(DataPipelineCore())

if __name__ == "__main__":
    async def test_advanced_price_service():
        print("🧪 Testing Advanced Price Data Service...")
        
        # Test real-time price
        price_data = await advanced_price_service.get_real_time_price('XAU')
        print(f"💰 Real-time price: {price_data}")
        
        # Test market summary
        summary = await advanced_price_service.get_market_summary('XAU')
        print(f"📊 Market summary: {summary}")
        
        # Test historical data
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=6)
        ohlcv = await advanced_price_service.get_historical_ohlcv('XAU', '1h', start_time, end_time)
        print(f"📉 Historical OHLCV: {len(ohlcv)} records")
    
    asyncio.run(test_advanced_price_service())
