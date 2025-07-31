#!/usr/bin/env python3
"""
Emergency Cache System for GoldGPT - CPU Optimization
====================================================

High-performance caching layer to prevent repeated ML calculations
and reduce CPU usage to optimal levels.
"""

import functools
import time
import json
import os
import hashlib
import pickle
import logging
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Optional
import sqlite3
import threading

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedCacheManager:
    """Professional caching system with TTL, memory management, and performance monitoring"""
    
    def __init__(self, cache_dir: str = "cache", max_memory_mb: int = 500):
        self.cache_dir = cache_dir
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.memory_cache = {}
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'memory_usage': 0,
            'disk_usage': 0
        }
        self.lock = threading.RLock()
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize SQLite for cache metadata
        self.db_path = os.path.join(cache_dir, "cache_metadata.db")
        self._init_cache_db()
        
        # Start background cleanup thread
        self._start_cleanup_thread()
        
        logger.info(f"🚀 Advanced Cache Manager initialized - Max memory: {max_memory_mb}MB")
    
    def _init_cache_db(self):
        """Initialize cache metadata database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS cache_entries (
                cache_key TEXT PRIMARY KEY,
                file_path TEXT,
                created_at TIMESTAMP,
                last_accessed TIMESTAMP,
                ttl_seconds INTEGER,
                size_bytes INTEGER,
                access_count INTEGER DEFAULT 0
            )
        """)
        
        conn.commit()
        conn.close()
    
    def _start_cleanup_thread(self):
        """Start background thread for cache cleanup"""
        def cleanup_worker():
            while True:
                try:
                    self._cleanup_expired_entries()
                    self._manage_memory_usage()
                    time.sleep(300)  # Run every 5 minutes
                except Exception as e:
                    logger.error(f"Cache cleanup error: {e}")
                    time.sleep(60)  # Retry in 1 minute on error
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
    
    def _cleanup_expired_entries(self):
        """Remove expired cache entries"""
        with self.lock:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Find expired entries
            cursor.execute("""
                SELECT cache_key, file_path FROM cache_entries 
                WHERE datetime(created_at, '+' || ttl_seconds || ' seconds') < datetime('now')
            """)
            
            expired_entries = cursor.fetchall()
            
            for cache_key, file_path in expired_entries:
                # Remove from memory cache
                if cache_key in self.memory_cache:
                    del self.memory_cache[cache_key]
                
                # Remove file
                if file_path and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except OSError:
                        pass
                
                # Remove from database
                cursor.execute("DELETE FROM cache_entries WHERE cache_key = ?", (cache_key,))
            
            conn.commit()
            conn.close()
            
            if expired_entries:
                logger.info(f"🧹 Cleaned up {len(expired_entries)} expired cache entries")
    
    def _manage_memory_usage(self):
        """Manage memory cache size"""
        with self.lock:
            current_memory = sum(len(pickle.dumps(value)) for value in self.memory_cache.values())
            
            if current_memory > self.max_memory_bytes:
                # Remove least recently used items
                sorted_keys = sorted(
                    self.memory_cache.keys(),
                    key=lambda k: self._get_last_accessed(k)
                )
                
                removed_count = 0
                for key in sorted_keys:
                    if current_memory <= self.max_memory_bytes * 0.8:  # Target 80% of max
                        break
                    
                    value_size = len(pickle.dumps(self.memory_cache[key]))
                    del self.memory_cache[key]
                    current_memory -= value_size
                    removed_count += 1
                
                logger.info(f"💾 Memory cache cleanup: removed {removed_count} entries")
    
    def _get_last_accessed(self, cache_key: str) -> datetime:
        """Get last accessed time for cache entry"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT last_accessed FROM cache_entries WHERE cache_key = ?", (cache_key,))
        result = cursor.fetchone()
        
        conn.close()
        
        if result:
            return datetime.fromisoformat(result[0])
        return datetime.min
    
    def _update_access_time(self, cache_key: str):
        """Update last accessed time for cache entry"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE cache_entries 
            SET last_accessed = datetime('now'), access_count = access_count + 1
            WHERE cache_key = ?
        """, (cache_key,))
        
        conn.commit()
        conn.close()
    
    def get(self, cache_key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            # Check memory cache first
            if cache_key in self.memory_cache:
                self._update_access_time(cache_key)
                self.cache_stats['hits'] += 1
                logger.debug(f"📦 Memory cache hit: {cache_key}")
                return self.memory_cache[cache_key]
            
            # Check disk cache
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT file_path, ttl_seconds, created_at FROM cache_entries 
                WHERE cache_key = ?
            """, (cache_key,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                file_path, ttl_seconds, created_at = result
                created_time = datetime.fromisoformat(created_at)
                
                # Check if expired
                if datetime.now() - created_time > timedelta(seconds=ttl_seconds):
                    self._remove_cache_entry(cache_key)
                    self.cache_stats['misses'] += 1
                    return None
                
                # Load from disk
                if file_path and os.path.exists(file_path):
                    try:
                        with open(file_path, 'rb') as f:
                            value = pickle.load(f)
                        
                        # Store in memory cache for faster access
                        self.memory_cache[cache_key] = value
                        self._update_access_time(cache_key)
                        self.cache_stats['hits'] += 1
                        logger.debug(f"💿 Disk cache hit: {cache_key}")
                        return value
                    except Exception as e:
                        logger.error(f"Error loading cache file {file_path}: {e}")
                        self._remove_cache_entry(cache_key)
            
            self.cache_stats['misses'] += 1
            return None
    
    def set(self, cache_key: str, value: Any, ttl_seconds: int = 300):
        """Set value in cache"""
        with self.lock:
            # Store in memory cache
            self.memory_cache[cache_key] = value
            
            # Store in disk cache for persistence
            file_path = os.path.join(self.cache_dir, f"{hashlib.md5(cache_key.encode()).hexdigest()}.pkl")
            
            try:
                with open(file_path, 'wb') as f:
                    pickle.dump(value, f)
                
                size_bytes = os.path.getsize(file_path)
                
                # Update database
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO cache_entries 
                    (cache_key, file_path, created_at, last_accessed, ttl_seconds, size_bytes)
                    VALUES (?, ?, datetime('now'), datetime('now'), ?, ?)
                """, (cache_key, file_path, ttl_seconds, size_bytes))
                
                conn.commit()
                conn.close()
                
                logger.debug(f"💾 Cached: {cache_key} (TTL: {ttl_seconds}s)")
                
            except Exception as e:
                logger.error(f"Error saving cache file: {e}")
    
    def _remove_cache_entry(self, cache_key: str):
        """Remove cache entry from memory, disk, and database"""
        # Remove from memory
        if cache_key in self.memory_cache:
            del self.memory_cache[cache_key]
        
        # Remove from database and get file path
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT file_path FROM cache_entries WHERE cache_key = ?", (cache_key,))
        result = cursor.fetchone()
        
        if result and result[0]:
            file_path = result[0]
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    pass
        
        cursor.execute("DELETE FROM cache_entries WHERE cache_key = ?", (cache_key,))
        conn.commit()
        conn.close()
    
    def clear_all(self):
        """Clear all cache entries"""
        with self.lock:
            # Clear memory cache
            self.memory_cache.clear()
            
            # Clear database and files
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT file_path FROM cache_entries")
            file_paths = cursor.fetchall()
            
            for (file_path,) in file_paths:
                if file_path and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except OSError:
                        pass
            
            cursor.execute("DELETE FROM cache_entries")
            conn.commit()
            conn.close()
            
            logger.info("🧹 All cache entries cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        with self.lock:
            total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
            hit_rate = (self.cache_stats['hits'] / total_requests * 100) if total_requests > 0 else 0
            
            # Calculate memory usage
            memory_usage = sum(len(pickle.dumps(value)) for value in self.memory_cache.values())
            
            # Calculate disk usage
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT SUM(size_bytes) FROM cache_entries")
            disk_usage = cursor.fetchone()[0] or 0
            cursor.execute("SELECT COUNT(*) FROM cache_entries")
            entry_count = cursor.fetchone()[0] or 0
            conn.close()
            
            return {
                'hit_rate': f"{hit_rate:.1f}%",
                'total_requests': total_requests,
                'hits': self.cache_stats['hits'],
                'misses': self.cache_stats['misses'],
                'memory_usage_mb': f"{memory_usage / 1024 / 1024:.1f}",
                'disk_usage_mb': f"{disk_usage / 1024 / 1024:.1f}",
                'entry_count': entry_count,
                'memory_entries': len(self.memory_cache)
            }

# Global cache manager instance
cache_manager = AdvancedCacheManager()

def cached_prediction(ttl_seconds: int = 300, use_memory: bool = True):
    """
    Advanced cache decorator for ML predictions and heavy computations
    
    Args:
        ttl_seconds: Time to live in seconds
        use_memory: Whether to use memory cache for faster access
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            key_data = {
                'function': func.__name__,
                'args': str(args),
                'kwargs': str(sorted(kwargs.items())),
                'timestamp': int(time.time() / ttl_seconds) * ttl_seconds  # Round to TTL interval
            }
            cache_key = hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
            
            # Try to get from cache
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                logger.info(f"📦 Cache hit for {func.__name__}")
                return cached_result
            
            # Generate fresh result
            logger.info(f"🔄 Cache miss for {func.__name__} - generating fresh result")
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Cache the result
            cache_manager.set(cache_key, result, ttl_seconds)
            
            logger.info(f"✅ {func.__name__} completed in {execution_time:.3f}s and cached")
            return result
        
        return wrapper
    return decorator

def smart_cache(ttl_seconds: int = 300, condition_func: Optional[Callable] = None):
    """
    Smart caching that can conditionally cache based on system state
    
    Args:
        ttl_seconds: Time to live in seconds
        condition_func: Optional function to determine if caching should be used
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Check condition if provided
            if condition_func and not condition_func():
                logger.debug(f"⏭️ Skipping cache for {func.__name__} due to condition")
                return func(*args, **kwargs)
            
            # Use regular caching
            return cached_prediction(ttl_seconds)(func)(*args, **kwargs)
        
        return wrapper
    return decorator

def get_cache_stats() -> Dict[str, Any]:
    """Get comprehensive cache statistics"""
    return cache_manager.get_stats()

def clear_cache():
    """Clear all cache entries"""
    cache_manager.clear_all()

def warmup_cache():
    """Warmup cache with commonly requested data"""
    logger.info("🔥 Starting cache warmup...")
    
    try:
        # Import here to avoid circular imports
        from ml_prediction_api import get_multi_timeframe_predictions
        from price_storage_manager import get_current_gold_price
        
        # Warmup common predictions
        timeframes = ['1H', '4H', '1D']
        for timeframe in timeframes:
            try:
                get_multi_timeframe_predictions(symbol="XAUUSD", timeframe=timeframe)
                time.sleep(0.1)  # Small delay to prevent overwhelming the system
            except Exception as e:
                logger.error(f"Warmup error for {timeframe}: {e}")
        
        # Warmup current price
        try:
            get_current_gold_price()
        except Exception as e:
            logger.error(f"Warmup error for current price: {e}")
        
        logger.info("✅ Cache warmup completed")
        
    except ImportError as e:
        logger.warning(f"Cache warmup skipped due to import error: {e}")

if __name__ == "__main__":
    # Test the cache system
    @cached_prediction(ttl_seconds=60)
    def test_function(x, y):
        time.sleep(2)  # Simulate heavy computation
        return x + y
    
    print("Testing cache system...")
    
    # First call - should be slow
    start = time.time()
    result1 = test_function(1, 2)
    time1 = time.time() - start
    print(f"First call: {result1} in {time1:.3f}s")
    
    # Second call - should be fast (cached)
    start = time.time()
    result2 = test_function(1, 2)
    time2 = time.time() - start
    print(f"Second call: {result2} in {time2:.3f}s")
    
    # Show cache stats
    stats = get_cache_stats()
    print(f"Cache stats: {stats}")
