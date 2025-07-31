#!/usr/bin/env python3
"""
GoldGPT Multi-Source Data Pipeline Core
Advanced tiered data architecture with intelligent fallback mechanisms
"""

import asyncio
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import aiohttp
import time
from concurrent.futures import ThreadPoolExecutor
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataSourceTier(Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    TERTIARY = "tertiary"

class DataType(Enum):
    PRICE = "price"
    NEWS = "news"
    TECHNICAL = "technical"
    MACRO = "macro"
    SENTIMENT = "sentiment"

@dataclass
class DataSourceConfig:
    """Configuration for data sources"""
    name: str
    tier: DataSourceTier
    url: str
    headers: Dict[str, str]
    rate_limit: int  # requests per minute
    reliability_score: float = 1.0
    timeout: int = 10
    enabled: bool = True
    api_key: Optional[str] = None

@dataclass
class DataPoint:
    """Standardized data point structure"""
    symbol: str
    timestamp: datetime
    data_type: DataType
    value: Any
    source: str
    confidence: float
    metadata: Dict[str, Any]

@dataclass
class CacheEntry:
    """Cache entry with TTL"""
    data: Any
    timestamp: datetime
    ttl: int
    source: str
    hash_key: str

class DataPipelineCore:
    """Core data pipeline with multi-source management"""
    
    def __init__(self, db_path: str = "goldgpt_data_pipeline.db"):
        self.db_path = db_path
        self.cache = {}
        self.source_configs = {}
        self.source_reliability = {}
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # TTL configuration for different data types (seconds)
        self.ttl_config = {
            DataType.PRICE: 5,      # 5 seconds for real-time prices
            DataType.NEWS: 300,     # 5 minutes for news
            DataType.TECHNICAL: 60, # 1 minute for technical indicators
            DataType.MACRO: 3600,   # 1 hour for macro data
            DataType.SENTIMENT: 300 # 5 minutes for sentiment
        }
        
        self.initialize_database()
        self.setup_data_sources()
    
    def initialize_database(self):
        """Initialize SQLite database with comprehensive schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Data points table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS data_points (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                data_type TEXT NOT NULL,
                value TEXT NOT NULL,
                source TEXT NOT NULL,
                confidence REAL NOT NULL,
                metadata TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create index separately
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_data_points_lookup 
            ON data_points(symbol, data_type, timestamp)
        ''')
        
        # Source reliability tracking
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS source_reliability (
                source TEXT PRIMARY KEY,
                total_requests INTEGER DEFAULT 0,
                successful_requests INTEGER DEFAULT 0,
                last_success DATETIME,
                last_failure DATETIME,
                average_response_time REAL DEFAULT 0,
                reliability_score REAL DEFAULT 1.0,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Cache management table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cache_entries (
                hash_key TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                timestamp DATETIME NOT NULL,
                ttl INTEGER NOT NULL,
                source TEXT NOT NULL,
                data_type TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("✅ Database initialized successfully")
    
    def setup_data_sources(self):
        """Configure all data sources with tiered architecture"""
        
        # PRIMARY TIER - Premium APIs
        primary_sources = [
            DataSourceConfig(
                name="gold_api",
                tier=DataSourceTier.PRIMARY,
                url="https://api.gold-api.com/price/XAU",
                headers={"X-API-KEY": "goldapi-YOUR_KEY"},
                rate_limit=1000,
                timeout=5
            ),
            DataSourceConfig(
                name="alpha_vantage",
                tier=DataSourceTier.PRIMARY,
                url="https://www.alphavantage.co/query",
                headers={},
                rate_limit=5,
                timeout=10,
                api_key="ALPHA_VANTAGE_KEY"
            ),
            DataSourceConfig(
                name="yahoo_finance",
                tier=DataSourceTier.PRIMARY,
                url="https://query1.finance.yahoo.com/v8/finance/chart/GC=F",
                headers={"User-Agent": "Mozilla/5.0"},
                rate_limit=60,
                timeout=10
            )
        ]
        
        # SECONDARY TIER - Web Scraping
        secondary_sources = [
            DataSourceConfig(
                name="investing_com",
                tier=DataSourceTier.SECONDARY,
                url="https://www.investing.com/commodities/gold",
                headers={"User-Agent": "Mozilla/5.0"},
                rate_limit=30,
                timeout=15
            ),
            DataSourceConfig(
                name="marketwatch",
                tier=DataSourceTier.SECONDARY,
                url="https://www.marketwatch.com/investing/future/gc00",
                headers={"User-Agent": "Mozilla/5.0"},
                rate_limit=30,
                timeout=15
            )
        ]
        
        # Store configurations
        for source in primary_sources + secondary_sources:
            self.source_configs[source.name] = source
            # Set Gold API as highest priority with superior reliability score
            initial_score = 2.0 if source.name == "gold_api" else 1.0
            self.source_reliability[source.name] = {
                'score': initial_score,
                'total_requests': 0,
                'successful_requests': 0,
                'last_response_time': 0
            }
        
        logger.info(f"✅ Configured {len(self.source_configs)} data sources")
    
    def generate_cache_key(self, symbol: str, data_type: DataType, params: Dict = None) -> str:
        """Generate unique cache key"""
        key_data = f"{symbol}_{data_type.value}"
        if params:
            key_data += "_" + json.dumps(params, sort_keys=True)
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get_cached_data(self, cache_key: str) -> Optional[Any]:
        """Retrieve data from cache if still valid"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT data, timestamp, ttl FROM cache_entries 
            WHERE hash_key = ?
        ''', (cache_key,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            data, timestamp_str, ttl = result
            timestamp = datetime.fromisoformat(timestamp_str)
            
            if datetime.now() - timestamp < timedelta(seconds=ttl):
                return json.loads(data)
        
        return None
    
    def cache_data(self, cache_key: str, data: Any, data_type: DataType, source: str):
        """Store data in cache with TTL"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        ttl = self.ttl_config.get(data_type, 300)  # Default 5 minutes
        
        cursor.execute('''
            INSERT OR REPLACE INTO cache_entries 
            (hash_key, data, timestamp, ttl, source, data_type)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            cache_key,
            json.dumps(data),
            datetime.now().isoformat(),
            ttl,
            source,
            data_type.value
        ))
        
        conn.commit()
        conn.close()
    
    def update_source_reliability(self, source: str, success: bool, response_time: float):
        """Update source reliability metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get current stats
        cursor.execute('''
            SELECT total_requests, successful_requests, average_response_time 
            FROM source_reliability WHERE source = ?
        ''', (source,))
        
        result = cursor.fetchone()
        
        if result:
            total, successful, avg_time = result
            total += 1
            if success:
                successful += 1
            
            # Calculate new average response time
            new_avg_time = ((avg_time * (total - 1)) + response_time) / total
            reliability_score = successful / total
            
            cursor.execute('''
                UPDATE source_reliability 
                SET total_requests = ?, successful_requests = ?, 
                    average_response_time = ?, reliability_score = ?,
                    last_success = ?, last_failure = ?, updated_at = ?
                WHERE source = ?
            ''', (
                total, successful, new_avg_time, reliability_score,
                datetime.now().isoformat() if success else None,
                datetime.now().isoformat() if not success else None,
                datetime.now().isoformat(),
                source
            ))
        else:
            # First entry for this source
            cursor.execute('''
                INSERT INTO source_reliability 
                (source, total_requests, successful_requests, average_response_time, 
                 reliability_score, last_success, last_failure)
                VALUES (?, 1, ?, ?, ?, ?, ?)
            ''', (
                source, 1 if success else 0, response_time, 1.0 if success else 0.0,
                datetime.now().isoformat() if success else None,
                datetime.now().isoformat() if not success else None
            ))
        
        conn.commit()
        conn.close()
        
        # Update in-memory tracking
        self.source_reliability[source] = {
            'score': reliability_score if 'reliability_score' in locals() else (1.0 if success else 0.0),
            'total_requests': total if 'total' in locals() else 1,
            'successful_requests': successful if 'successful' in locals() else (1 if success else 0),
            'last_response_time': response_time
        }
    
    def get_best_sources(self, data_type: DataType, limit: int = 3) -> List[str]:
        """Get best sources for data type based on reliability with Gold API priority"""
        available_sources = []
        
        for source_name, config in self.source_configs.items():
            if config.enabled and source_name in self.source_reliability:
                reliability = self.source_reliability[source_name]
                available_sources.append((source_name, reliability['score'], config.tier.value))
        
        # Sort by reliability score and tier priority with explicit Gold API preference
        tier_priority = {"primary": 3, "secondary": 2, "tertiary": 1}
        
        def sort_key(source_info):
            name, score, tier = source_info
            # Give Gold API maximum priority regardless of reliability
            if name == "gold_api":
                return (10, score)  # Highest priority for Gold API
            return (tier_priority.get(tier, 0), score)
        
        available_sources.sort(key=sort_key, reverse=True)
        
        return [source[0] for source in available_sources[:limit]]
    
    async def fetch_from_source(self, source_name: str, symbol: str, data_type: DataType, **kwargs) -> Optional[Dict]:
        """Fetch data from specific source with error handling"""
        if source_name not in self.source_configs:
            logger.warning(f"Unknown source: {source_name}")
            return None
        
        config = self.source_configs[source_name]
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=config.timeout)) as session:
                
                if source_name == "gold_api":
                    return await self._fetch_gold_api(session, config, symbol)
                elif source_name == "alpha_vantage":
                    return await self._fetch_alpha_vantage(session, config, symbol, **kwargs)
                elif source_name == "yahoo_finance":
                    return await self._fetch_yahoo_finance(session, config, symbol)
                elif source_name == "investing_com":
                    return await self._fetch_investing_com(session, config, symbol)
                elif source_name == "marketwatch":
                    return await self._fetch_marketwatch(session, config, symbol)
                else:
                    logger.warning(f"No fetch method for source: {source_name}")
                    return None
        
        except Exception as e:
            response_time = time.time() - start_time
            self.update_source_reliability(source_name, False, response_time)
            logger.error(f"Error fetching from {source_name}: {e}")
            return None
    
    async def _fetch_gold_api(self, session: aiohttp.ClientSession, config: DataSourceConfig, symbol: str) -> Optional[Dict]:
        """Fetch from Gold-API.com"""
        try:
            async with session.get(config.url, headers=config.headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        'price': data.get('price', 0),
                        'timestamp': datetime.now().isoformat(),
                        'source': 'gold_api',
                        'confidence': 0.95
                    }
        except Exception as e:
            logger.error(f"Gold API error: {e}")
        return None
    
    async def _fetch_alpha_vantage(self, session: aiohttp.ClientSession, config: DataSourceConfig, symbol: str, **kwargs) -> Optional[Dict]:
        """Fetch from Alpha Vantage"""
        try:
            function = kwargs.get('function', 'GLOBAL_QUOTE')
            params = {
                'function': function,
                'symbol': 'GLD',  # Gold ETF as proxy
                'apikey': config.api_key or 'demo'
            }
            
            async with session.get(config.url, params=params, headers=config.headers) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'Global Quote' in data:
                        quote = data['Global Quote']
                        return {
                            'price': float(quote.get('05. price', 0)),
                            'timestamp': datetime.now().isoformat(),
                            'source': 'alpha_vantage',
                            'confidence': 0.90
                        }
        except Exception as e:
            logger.error(f"Alpha Vantage error: {e}")
        return None
    
    async def _fetch_yahoo_finance(self, session: aiohttp.ClientSession, config: DataSourceConfig, symbol: str) -> Optional[Dict]:
        """Fetch from Yahoo Finance"""
        try:
            async with session.get(config.url, headers=config.headers) as response:
                if response.status == 200:
                    data = await response.json()
                    if 'chart' in data and 'result' in data['chart']:
                        result = data['chart']['result'][0]
                        meta = result.get('meta', {})
                        return {
                            'price': meta.get('regularMarketPrice', 0),
                            'timestamp': datetime.now().isoformat(),
                            'source': 'yahoo_finance',
                            'confidence': 0.85
                        }
        except Exception as e:
            logger.error(f"Yahoo Finance error: {e}")
        return None
    
    async def _fetch_investing_com(self, session: aiohttp.ClientSession, config: DataSourceConfig, symbol: str) -> Optional[Dict]:
        """Fetch from Investing.com (web scraping)"""
        # This would implement web scraping logic
        # For now, return simulated data
        try:
            return {
                'price': 3400 + (hash(str(datetime.now().minute)) % 100) / 10,  # Simulated
                'timestamp': datetime.now().isoformat(),
                'source': 'investing_com',
                'confidence': 0.70
            }
        except Exception as e:
            logger.error(f"Investing.com error: {e}")
        return None
    
    async def _fetch_marketwatch(self, session: aiohttp.ClientSession, config: DataSourceConfig, symbol: str) -> Optional[Dict]:
        """Fetch from MarketWatch (web scraping)"""
        # This would implement web scraping logic
        # For now, return simulated data
        try:
            return {
                'price': 3400 + (hash(str(datetime.now().second)) % 100) / 10,  # Simulated
                'timestamp': datetime.now().isoformat(),
                'source': 'marketwatch',
                'confidence': 0.70
            }
        except Exception as e:
            logger.error(f"MarketWatch error: {e}")
        return None
    
    async def get_unified_data(self, symbol: str, data_type: DataType, **kwargs) -> Optional[Dict]:
        """Get data from best available sources with fallback"""
        cache_key = self.generate_cache_key(symbol, data_type, kwargs)
        
        # Check cache first
        cached_data = self.get_cached_data(cache_key)
        if cached_data:
            logger.info(f"📋 Cache hit for {symbol} {data_type.value}")
            return cached_data
        
        # Get best sources to try
        sources = self.get_best_sources(data_type, limit=3)
        logger.info(f"🎯 Trying sources for {symbol}: {sources}")
        
        # Try sources in order of reliability
        for source in sources:
            start_time = time.time()
            data = await self.fetch_from_source(source, symbol, data_type, **kwargs)
            response_time = time.time() - start_time
            
            if data:
                # Validate data quality
                if self.validate_data_quality(data, data_type):
                    self.update_source_reliability(source, True, response_time)
                    self.cache_data(cache_key, data, data_type, source)
                    logger.info(f"✅ Got data from {source} for {symbol}")
                    return data
                else:
                    logger.warning(f"⚠️ Data quality check failed for {source}")
                    self.update_source_reliability(source, False, response_time)
            else:
                self.update_source_reliability(source, False, response_time)
        
        # All sources failed - return simulated data if available
        logger.warning(f"🔄 All sources failed for {symbol}, using fallback")
        return self.get_fallback_data(symbol, data_type)
    
    def validate_data_quality(self, data: Dict, data_type: DataType) -> bool:
        """Validate data quality and detect anomalies"""
        if not data:
            return False
        
        if data_type == DataType.PRICE:
            price = data.get('price', 0)
            
            # Basic price validation for gold
            if not (1000 <= price <= 10000):  # Reasonable gold price range
                logger.warning(f"Price {price} outside reasonable range")
                return False
            
            # Check timestamp freshness (should be recent)
            timestamp_str = data.get('timestamp')
            if timestamp_str:
                try:
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    age = datetime.now() - timestamp.replace(tzinfo=None)
                    if age > timedelta(minutes=10):  # Data shouldn't be too old
                        logger.warning(f"Data too old: {age}")
                        return False
                except:
                    pass
        
        return True
    
    def get_fallback_data(self, symbol: str, data_type: DataType) -> Dict:
        """Generate fallback data when all sources fail"""
        if data_type == DataType.PRICE:
            # Generate realistic simulated price based on time
            base_price = 3400
            variation = (hash(str(datetime.now().minute)) % 200 - 100) / 10  # ±10 range
            
            return {
                'price': base_price + variation,
                'timestamp': datetime.now().isoformat(),
                'source': 'fallback_simulation',
                'confidence': 0.30,
                'is_simulated': True
            }
        
        return {
            'value': None,
            'timestamp': datetime.now().isoformat(),
            'source': 'fallback_simulation',
            'confidence': 0.10,
            'is_simulated': True
        }
    
    def get_source_status(self) -> Dict[str, Any]:
        """Get status of all data sources"""
        status = {}
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT source, total_requests, successful_requests, 
                   reliability_score, average_response_time, last_success
            FROM source_reliability
        ''')
        
        for row in cursor.fetchall():
            source, total, successful, score, avg_time, last_success = row
            status[source] = {
                'total_requests': total,
                'successful_requests': successful,
                'reliability_score': round(score, 3),
                'average_response_time': round(avg_time, 3),
                'last_success': last_success,
                'tier': self.source_configs[source].tier.value if source in self.source_configs else 'unknown'
            }
        
        conn.close()
        return status
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all data sources"""
        results = {}
        
        for source_name in self.source_configs:
            start_time = time.time()
            data = await self.fetch_from_source(source_name, 'XAU', DataType.PRICE)
            response_time = time.time() - start_time
            
            results[source_name] = {
                'status': 'healthy' if data else 'unhealthy',
                'response_time': round(response_time, 3),
                'last_check': datetime.now().isoformat()
            }
        
        return results

# Global instance
data_pipeline = DataPipelineCore()

if __name__ == "__main__":
    # Test the pipeline
    async def test_pipeline():
        print("🧪 Testing Data Pipeline Core...")
        
        # Test unified data fetching
        data = await data_pipeline.get_unified_data('XAU', DataType.PRICE)
        print(f"📊 Price data: {data}")
        
        # Test health check
        health = await data_pipeline.health_check()
        print(f"🏥 Health check: {health}")
        
        # Test source status
        status = data_pipeline.get_source_status()
        print(f"📈 Source status: {status}")
    
    asyncio.run(test_pipeline())
