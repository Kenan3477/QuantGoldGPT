#!/usr/bin/env python3
"""
Price Storage Manager - Centralized Real-Time Price Management
Stores and manages all price tick data with fallback capabilities and candlestick data for technical analysis
"""

import sqlite3
import requests
import logging
import time
from datetime import datetime, timedelta
import json
from typing import Dict, Optional, List
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PriceStorageManager:
    """Centralized price management with storage and fallback capabilities"""
    
    def __init__(self, db_path: str = "goldgpt_prices.db"):
        self.db_path = db_path
        self.init_database()
        self.cache = {}
        self.cache_timeout = 30  # 30 seconds cache
        
    def init_database(self):
        """Initialize price storage database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create price_ticks table with comprehensive data
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS price_ticks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    price REAL NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    source TEXT NOT NULL,
                    high_24h REAL,
                    low_24h REAL,
                    change_24h REAL,
                    change_percent REAL,
                    volume REAL,
                    market_cap REAL,
                    api_response TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create candlestick data table for technical analysis
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS candlestick_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    open_price REAL NOT NULL,
                    high_price REAL NOT NULL,
                    low_price REAL NOT NULL,
                    close_price REAL NOT NULL,
                    volume REAL DEFAULT 0,
                    tick_count INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timeframe, timestamp)
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("✅ Price storage database initialized")
            
        except Exception as e:
            logger.error(f"❌ Database initialization failed: {e}")
    
    def store_price(self, symbol: str, price: float, source: str = "unknown", **kwargs) -> bool:
        """Store a price tick in the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO price_ticks (
                    symbol, price, source, high_24h, low_24h, 
                    change_24h, change_percent, volume, market_cap, api_response
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                price,
                source,
                kwargs.get('high_24h'),
                kwargs.get('low_24h'),
                kwargs.get('change_24h'),
                kwargs.get('change_percent'),
                kwargs.get('volume'),
                kwargs.get('market_cap'),
                json.dumps(kwargs.get('raw_response', {}))
            ))
            
            conn.commit()
            conn.close()
            
            # Update cache
            self.cache[symbol] = {
                'price': price,
                'timestamp': time.time(),
                'source': source,
                **kwargs
            }
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to store price: {e}")
            return False
    
    def get_latest_price(self, symbol: str, max_age_seconds: int = 300) -> Optional[Dict]:
        """Get the latest price from cache or database"""
        try:
            # Check cache first
            if symbol in self.cache:
                cache_age = time.time() - self.cache[symbol]['timestamp']
                if cache_age < self.cache_timeout:
                    logger.info(f"📋 Using cached price for {symbol}")
                    return self.cache[symbol]
            
            # Fetch from database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT price, timestamp, source, high_24h, low_24h, change_24h, change_percent
                FROM price_ticks 
                WHERE symbol = ? AND timestamp > datetime('now', '-{} seconds')
                ORDER BY timestamp DESC 
                LIMIT 1
            '''.format(max_age_seconds), (symbol,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                result = {
                    'price': row[0],
                    'timestamp': row[1],
                    'source': row[2],
                    'high_24h': row[3],
                    'low_24h': row[4],
                    'change_24h': row[5],
                    'change_percent': row[6]
                }
                
                # Update cache
                self.cache[symbol] = result
                return result
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get latest price: {e}")
            return None
    
    def fetch_and_store_price(self, symbol: str = "XAUUSD") -> Optional[float]:
        """Fetch current price from Gold API and store it"""
        try:
            # Call the cleanup process before fetching new data
            self.cleanup_old_records()
            
            # Fetch from Gold API
            url = "https://api.gold-api.com/price/XAU"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                current_price = float(data.get('price', 0))
                
                if current_price > 0:
                    # Store in database
                    self.store_price(
                        symbol=symbol,
                        price=current_price,
                        source="gold_api",
                        high_24h=data.get('high_24h'),
                        low_24h=data.get('low_24h'),
                        change_24h=data.get('change_24h'),
                        change_percent=data.get('change_percent'),
                        raw_response=data
                    )
                    
                    logger.info(f"✅ Real-time {symbol} price: ${current_price}")
                    return current_price
                    
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to fetch and store price: {e}")
            return None
    
    def cleanup_old_records(self):
        """Archive old records and create candlestick data for technical analysis"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create candlestick data table if it doesn't exist
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS candlestick_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    open_price REAL NOT NULL,
                    high_price REAL NOT NULL,
                    low_price REAL NOT NULL,
                    close_price REAL NOT NULL,
                    volume REAL DEFAULT 0,
                    tick_count INTEGER DEFAULT 0,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, timeframe, timestamp)
                )
            ''')
            
            # Convert old tick data to 1-minute candlesticks instead of deleting
            cursor.execute('''
                INSERT OR REPLACE INTO candlestick_data (
                    symbol, timeframe, timestamp, open_price, high_price, 
                    low_price, close_price, volume, tick_count
                )
                SELECT 
                    symbol,
                    '1m' as timeframe,
                    datetime(timestamp, '-' || strftime('%S', timestamp) || ' seconds') as candle_time,
                    (SELECT price FROM price_ticks p2 WHERE p2.symbol = p1.symbol 
                     AND datetime(p2.timestamp, '-' || strftime('%S', p2.timestamp) || ' seconds') = 
                         datetime(p1.timestamp, '-' || strftime('%S', p1.timestamp) || ' seconds')
                     ORDER BY p2.timestamp ASC LIMIT 1) as open_price,
                    MAX(price) as high_price,
                    MIN(price) as low_price,
                    (SELECT price FROM price_ticks p3 WHERE p3.symbol = p1.symbol 
                     AND datetime(p3.timestamp, '-' || strftime('%S', p3.timestamp) || ' seconds') = 
                         datetime(p1.timestamp, '-' || strftime('%S', p1.timestamp) || ' seconds')
                     ORDER BY p3.timestamp DESC LIMIT 1) as close_price,
                    COUNT(*) * 100 as volume,
                    COUNT(*) as tick_count
                FROM price_ticks p1
                WHERE timestamp < datetime('now', '-1 hour')
                GROUP BY symbol, datetime(timestamp, '-' || strftime('%S', timestamp) || ' seconds')
            ''')
            
            candlesticks_created = cursor.rowcount
            
            # Now remove old tick data (older than 1 hour) since we've archived it as candlesticks
            cursor.execute('''
                DELETE FROM price_ticks 
                WHERE timestamp < datetime('now', '-1 hour')
            ''')
            
            deleted = cursor.rowcount
            conn.commit()
            conn.close()
            
            if candlesticks_created > 0:
                logger.info(f"📊 Created {candlesticks_created} candlestick records from historical data")
            if deleted > 0:
                logger.info(f"🗄️ Archived {deleted} old price ticks (converted to candlestick data)")
                
        except Exception as e:
            logger.error(f"❌ Failed to process historical data: {e}")
    
    def get_stats(self) -> Dict:
        """Get storage statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT COUNT(*) FROM price_ticks')
            total_records = cursor.fetchone()[0]
            
            cursor.execute('''
                SELECT symbol, COUNT(*), MAX(timestamp) as latest
                FROM price_ticks 
                GROUP BY symbol
            ''')
            
            symbols = cursor.fetchall()
            conn.close()
            
            return {
                'total_records': total_records,
                'symbols': [
                    {
                        'symbol': row[0],
                        'count': row[1],
                        'latest': row[2]
                    } for row in symbols
                ]
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get stats: {e}")
            return {'total_records': 0, 'symbols': []}

    def get_candlestick_data(self, symbol: str, timeframe: str = '1m', limit: int = 100) -> List[Dict]:
        """Get candlestick data for technical analysis"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT timestamp, open_price, high_price, low_price, close_price, volume, tick_count
                FROM candlestick_data 
                WHERE symbol = ? AND timeframe = ?
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (symbol, timeframe, limit))
            
            rows = cursor.fetchall()
            conn.close()
            
            return [
                {
                    'timestamp': row[0],
                    'open': row[1],
                    'high': row[2],
                    'low': row[3],
                    'close': row[4],
                    'volume': row[5],
                    'tick_count': row[6]
                } for row in rows
            ]
            
        except Exception as e:
            logger.error(f"❌ Failed to get candlestick data: {e}")
            return []

    def get_historical_prices(self, symbol: str, hours: int = 24) -> List[Dict]:
        """Get historical price ticks for analysis"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT price, timestamp, high_24h, low_24h, change_24h, change_percent
                FROM price_ticks 
                WHERE symbol = ? AND timestamp > datetime('now', '-{} hours')
                ORDER BY timestamp ASC
            '''.format(hours), (symbol,))
            
            rows = cursor.fetchall()
            conn.close()
            
            return [
                {
                    'price': row[0],
                    'timestamp': row[1],
                    'high_24h': row[2],
                    'low_24h': row[3],
                    'change_24h': row[4],
                    'change_percent': row[5]
                } for row in rows
            ]
            
        except Exception as e:
            logger.error(f"❌ Failed to get historical prices: {e}")
            return []

# Global instance
price_manager = PriceStorageManager()

def get_current_gold_price():
    """Get current gold price - API function for backward compatibility"""
    try:
        # Try to get from cache/database first
        latest = price_manager.get_latest_price("XAUUSD")
        if latest:
            return latest['price']
        
        # Fetch fresh price
        price = price_manager.fetch_and_store_price("XAUUSD")
        if price:
            return price
            
        # Ultimate fallback
        return 3429.50
        
    except Exception as e:
        logger.error(f"❌ Error getting current gold price: {e}")
        return 3429.50

def store_gold_price(price: float, source: str = "manual"):
    """Store a gold price - API function for external use"""
    return price_manager.store_price("XAUUSD", price, source)

def get_price_stats():
    """Get price storage statistics"""
    return price_manager.get_stats()

def get_historical_prices(symbol: str = "XAUUSD", hours: int = 24):
    """Get historical prices - API function for backward compatibility"""
    return price_manager.get_historical_prices(symbol, hours)
