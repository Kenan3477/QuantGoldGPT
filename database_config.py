"""
Database Configuration for GoldGPT
Handles both SQLite (development) and PostgreSQL (production/Railway)
"""

import os
import sqlite3
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)

def get_database_connection():
    """Get database connection based on environment"""
    database_url = os.environ.get('DATABASE_URL')
    
    if database_url and database_url.startswith('postgresql'):
        # Railway PostgreSQL
        try:
            import psycopg2
            from urllib.parse import urlparse
            
            result = urlparse(database_url)
            connection = psycopg2.connect(
                database=result.path[1:],
                user=result.username,
                password=result.password,
                host=result.hostname,
                port=result.port
            )
            logger.info("✅ Connected to PostgreSQL database (Railway)")
            return connection
        except ImportError:
            logger.error("❌ psycopg2 not installed, falling back to SQLite")
            return sqlite3.connect('goldgpt_data.db')
        except Exception as e:
            logger.error(f"❌ PostgreSQL connection failed: {e}, falling back to SQLite")
            return sqlite3.connect('goldgpt_data.db')
    else:
        # Local SQLite
        logger.info("✅ Connected to SQLite database (Local)")
        return sqlite3.connect('goldgpt_data.db')

def init_database():
    """Initialize database tables for both SQLite and PostgreSQL"""
    try:
        conn = get_database_connection()
        cursor = conn.cursor()
        
        database_url = os.environ.get('DATABASE_URL', '')
        is_postgresql = database_url.startswith('postgresql')
        
        # AI Signals table
        if is_postgresql:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ai_signals (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    signal VARCHAR(20) NOT NULL,
                    confidence DECIMAL(5,2) NOT NULL,
                    current_price DECIMAL(10,2) NOT NULL,
                    target_price DECIMAL(10,2),
                    reasoning TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    technical_data JSONB,
                    timeframe VARCHAR(10) DEFAULT '1H'
                )
            ''')
        else:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ai_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    current_price REAL NOT NULL,
                    target_price REAL,
                    reasoning TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    technical_data TEXT,
                    timeframe TEXT DEFAULT '1H'
                )
            ''')
        
        # ML Predictions table
        if is_postgresql:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ml_predictions (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(10) NOT NULL,
                    timeframe VARCHAR(10) NOT NULL,
                    target_price DECIMAL(10,2),
                    confidence DECIMAL(3,2),
                    direction VARCHAR(10),
                    reasoning TEXT,
                    technical_indicators JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    accuracy_tracked BOOLEAN DEFAULT FALSE
                )
            ''')
        else:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ml_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    target_price REAL,
                    confidence REAL,
                    direction TEXT,
                    reasoning TEXT,
                    technical_indicators TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    accuracy_tracked INTEGER DEFAULT 0
                )
            ''')
        
        # Price Data table
        if is_postgresql:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS price_data (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(10) NOT NULL,
                    price DECIMAL(10,2) NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    source VARCHAR(50),
                    volume BIGINT,
                    change_24h DECIMAL(10,2)
                )
            ''')
        else:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS price_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    price REAL NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    source TEXT,
                    volume INTEGER,
                    change_24h REAL
                )
            ''')
        
        # News Data table
        if is_postgresql:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS news_data (
                    id SERIAL PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT,
                    source VARCHAR(100),
                    url TEXT,
                    sentiment_score DECIMAL(3,2),
                    impact_score DECIMAL(3,2),
                    published_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
        else:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS news_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    content TEXT,
                    source TEXT,
                    url TEXT,
                    sentiment_score REAL,
                    impact_score REAL,
                    published_at TIMESTAMP,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
        
        # Create indexes for better performance
        if is_postgresql:
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ai_signals_symbol ON ai_signals(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ai_signals_timestamp ON ai_signals(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ml_predictions_symbol ON ml_predictions(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_price_data_symbol ON price_data(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_price_data_timestamp ON price_data(timestamp)')
        else:
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ai_signals_symbol ON ai_signals(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ai_signals_timestamp ON ai_signals(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ml_predictions_symbol ON ml_predictions(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_price_data_symbol ON price_data(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_price_data_timestamp ON price_data(timestamp)')
        
        conn.commit()
        conn.close()
        
        db_type = "PostgreSQL" if is_postgresql else "SQLite"
        logger.info(f"✅ Database ({db_type}) initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        raise e

def execute_query(query, params=None, fetch=False):
    """Execute a database query with proper error handling"""
    try:
        conn = get_database_connection()
        cursor = conn.cursor()
        
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        
        if fetch:
            if fetch == 'one':
                result = cursor.fetchone()
            else:
                result = cursor.fetchall()
        else:
            result = None
            
        conn.commit()
        conn.close()
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Database query failed: {e}")
        logger.error(f"Query: {query}")
        logger.error(f"Params: {params}")
        raise e

def get_table_info():
    """Get information about database tables"""
    try:
        conn = get_database_connection()
        cursor = conn.cursor()
        
        database_url = os.environ.get('DATABASE_URL', '')
        is_postgresql = database_url.startswith('postgresql')
        
        if is_postgresql:
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
        else:
            cursor.execute("""
                SELECT name 
                FROM sqlite_master 
                WHERE type='table'
            """)
        
        tables = cursor.fetchall()
        conn.close()
        
        return [table[0] for table in tables]
        
    except Exception as e:
        logger.error(f"❌ Failed to get table info: {e}")
        return []

if __name__ == "__main__":
    # Test database connection and initialization
    print("🔧 Testing database configuration...")
    
    try:
        init_database()
        tables = get_table_info()
        print(f"✅ Database test successful. Tables: {tables}")
    except Exception as e:
        print(f"❌ Database test failed: {e}")
