"""
Configuration file for GoldGPT Web Application
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Base configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'goldgpt-secret-key-2025'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Database
    DATABASE_URL = os.environ.get('DATABASE_URL') or 'sqlite:///goldgpt.db'
    
    # Redis for caching (optional)
    REDIS_URL = os.environ.get('REDIS_URL') or 'redis://localhost:6379'
    
    # API Keys (add your own)
    ALPHA_VANTAGE_API_KEY = os.environ.get('ALPHA_VANTAGE_API_KEY')
    FINNHUB_API_KEY = os.environ.get('FINNHUB_API_KEY')
    POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY')
    
    # Trading settings
    DEFAULT_LEVERAGE = 1
    MAX_TRADES_PER_DAY = 50
    RISK_PERCENTAGE = 2.0  # Max 2% risk per trade
    
    # AI/ML settings
    ML_MODEL_PATH = os.environ.get('ML_MODEL_PATH') or './models/'
    CONFIDENCE_THRESHOLD = 0.7
    
    # Real-time data settings
    PRICE_UPDATE_INTERVAL = 5  # seconds
    MAX_HISTORICAL_BARS = 1000
    
    # Notification settings
    TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')
    
    # Security
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY') or 'jwt-secret-key'
    JWT_ACCESS_TOKEN_EXPIRES = 3600  # 1 hour
    
    # Feature flags
    ENABLE_REAL_TRADING = os.environ.get('ENABLE_REAL_TRADING', 'False').lower() == 'true'
    ENABLE_ADVANCED_AI = os.environ.get('ENABLE_ADVANCED_AI', 'True').lower() == 'true'
    ENABLE_NOTIFICATIONS = os.environ.get('ENABLE_NOTIFICATIONS', 'True').lower() == 'true'

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///goldgpt_dev.db'

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///goldgpt_prod.db'

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///goldgpt_test.db'

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

# Constants for data_fetcher.py
POLYGON_API_KEY = os.environ.get('POLYGON_API_KEY')
FRED_API_KEY = os.environ.get('FRED_API_KEY') 
DB_PATH = os.environ.get('DATABASE_PATH') or 'goldgpt.db'
