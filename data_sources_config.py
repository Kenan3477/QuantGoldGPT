#!/usr/bin/env python3
"""
Data Sources Configuration for GoldGPT
Centralizes all API keys, endpoints, and data source configurations
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class APIConfig:
    """Configuration for a single API"""
    name: str
    base_url: str
    api_key: Optional[str] = None
    rate_limit: int = 60  # requests per minute
    timeout: int = 10
    retry_attempts: int = 3
    enabled: bool = True

@dataclass
class DataSourceConfig:
    """Configuration for a data source with multiple APIs"""
    source_type: str
    primary_api: APIConfig
    fallback_apis: List[APIConfig]
    cache_ttl: int
    priority: int

class DataSourcesConfig:
    """Centralized configuration for all data sources"""
    
    def __init__(self):
        self._load_environment_variables()
        self._initialize_configurations()
    
    def _load_environment_variables(self):
        """Load API keys from environment variables"""
        self.api_keys = {
            'ALPHA_VANTAGE_API_KEY': os.getenv('ALPHA_VANTAGE_API_KEY'),
            'FRED_API_KEY': os.getenv('FRED_API_KEY'),
            'NEWSAPI_KEY': os.getenv('NEWSAPI_KEY'),
            'QUANDL_API_KEY': os.getenv('QUANDL_API_KEY'),
            'TRADING_ECONOMICS_KEY': os.getenv('TRADING_ECONOMICS_KEY'),
            'WORLD_BANK_API_KEY': os.getenv('WORLD_BANK_API_KEY'),
            'YAHOO_FINANCE_KEY': os.getenv('YAHOO_FINANCE_KEY'),
            'FINNHUB_API_KEY': os.getenv('FINNHUB_API_KEY'),
            'IEX_CLOUD_KEY': os.getenv('IEX_CLOUD_KEY')
        }
    
    def _initialize_configurations(self):
        """Initialize all data source configurations"""
        
        # Price Data Sources
        self.price_data_sources = [
            DataSourceConfig(
                source_type="price_data",
                primary_api=APIConfig(
                    name="Gold API",
                    base_url="https://api.gold-api.com",
                    rate_limit=100,
                    cache_ttl=60
                ),
                fallback_apis=[
                    APIConfig(
                        name="Alpha Vantage",
                        base_url="https://www.alphavantage.co/query",
                        api_key=self.api_keys.get('ALPHA_VANTAGE_API_KEY'),
                        rate_limit=5,
                        enabled=bool(self.api_keys.get('ALPHA_VANTAGE_API_KEY'))
                    ),
                    APIConfig(
                        name="Yahoo Finance",
                        base_url="https://query1.finance.yahoo.com/v8/finance/chart",
                        rate_limit=2000
                    ),
                    APIConfig(
                        name="IEX Cloud",
                        base_url="https://cloud.iexapis.com/stable",
                        api_key=self.api_keys.get('IEX_CLOUD_KEY'),
                        rate_limit=100,
                        enabled=bool(self.api_keys.get('IEX_CLOUD_KEY'))
                    )
                ],
                cache_ttl=60,
                priority=1
            )
        ]
        
        # News Data Sources
        self.news_data_sources = [
            DataSourceConfig(
                source_type="news_data",
                primary_api=APIConfig(
                    name="NewsAPI",
                    base_url="https://newsapi.org/v2",
                    api_key=self.api_keys.get('NEWSAPI_KEY'),
                    rate_limit=1000,
                    enabled=bool(self.api_keys.get('NEWSAPI_KEY'))
                ),
                fallback_apis=[
                    APIConfig(
                        name="Finnhub News",
                        base_url="https://finnhub.io/api/v1",
                        api_key=self.api_keys.get('FINNHUB_API_KEY'),
                        rate_limit=60,
                        enabled=bool(self.api_keys.get('FINNHUB_API_KEY'))
                    ),
                    APIConfig(
                        name="MarketWatch Scraper",
                        base_url="https://www.marketwatch.com",
                        rate_limit=30
                    ),
                    APIConfig(
                        name="Reuters Scraper",
                        base_url="https://www.reuters.com",
                        rate_limit=30
                    ),
                    APIConfig(
                        name="Investing.com Scraper",
                        base_url="https://www.investing.com",
                        rate_limit=30
                    )
                ],
                cache_ttl=3600,  # 1 hour
                priority=2
            )
        ]
        
        # Economic Data Sources
        self.economic_data_sources = [
            DataSourceConfig(
                source_type="economic_data",
                primary_api=APIConfig(
                    name="FRED API",
                    base_url="https://api.stlouisfed.org/fred",
                    api_key=self.api_keys.get('FRED_API_KEY'),
                    rate_limit=120,
                    enabled=bool(self.api_keys.get('FRED_API_KEY'))
                ),
                fallback_apis=[
                    APIConfig(
                        name="World Bank API",
                        base_url="https://api.worldbank.org/v2",
                        rate_limit=120
                    ),
                    APIConfig(
                        name="Trading Economics API",
                        base_url="https://api.tradingeconomics.com",
                        api_key=self.api_keys.get('TRADING_ECONOMICS_KEY'),
                        rate_limit=100,
                        enabled=bool(self.api_keys.get('TRADING_ECONOMICS_KEY'))
                    ),
                    APIConfig(
                        name="Quandl API",
                        base_url="https://www.quandl.com/api/v3",
                        api_key=self.api_keys.get('QUANDL_API_KEY'),
                        rate_limit=50,
                        enabled=bool(self.api_keys.get('QUANDL_API_KEY'))
                    )
                ],
                cache_ttl=14400,  # 4 hours
                priority=3
            )
        ]
    
    def get_price_sources(self) -> List[DataSourceConfig]:
        """Get all configured price data sources"""
        return self.price_data_sources
    
    def get_news_sources(self) -> List[DataSourceConfig]:
        """Get all configured news data sources"""
        return self.news_data_sources
    
    def get_economic_sources(self) -> List[DataSourceConfig]:
        """Get all configured economic data sources"""
        return self.economic_data_sources
    
    def get_all_sources(self) -> List[DataSourceConfig]:
        """Get all configured data sources"""
        return (self.price_data_sources + 
                self.news_data_sources + 
                self.economic_data_sources)
    
    def get_enabled_sources(self) -> List[DataSourceConfig]:
        """Get only enabled data sources"""
        enabled_sources = []
        for sources in [self.price_data_sources, self.news_data_sources, self.economic_data_sources]:
            for source in sources:
                if source.primary_api.enabled:
                    enabled_sources.append(source)
                else:
                    # Check if any fallback APIs are enabled
                    enabled_fallbacks = [api for api in source.fallback_apis if api.enabled]
                    if enabled_fallbacks:
                        # Create a new config with enabled fallbacks only
                        enabled_source = DataSourceConfig(
                            source_type=source.source_type,
                            primary_api=enabled_fallbacks[0],  # Use first enabled fallback as primary
                            fallback_apis=enabled_fallbacks[1:],
                            cache_ttl=source.cache_ttl,
                            priority=source.priority
                        )
                        enabled_sources.append(enabled_source)
        
        return sorted(enabled_sources, key=lambda x: x.priority)

# News Sources Configuration
NEWS_SOURCES = {
    'financial_news': [
        {
            'name': 'MarketWatch',
            'url': 'https://www.marketwatch.com/topics/gold',
            'selectors': {
                'articles': 'article.article--wrap',
                'title': 'h3.article__headline',
                'content': 'div.article__body',
                'time': 'time.article__timestamp'
            },
            'rate_limit': 30
        },
        {
            'name': 'Reuters Business',
            'url': 'https://www.reuters.com/business/commodities/',
            'selectors': {
                'articles': 'article[data-testid="MediaStoryCard"]',
                'title': 'a[data-testid="Heading"]',
                'content': 'p[data-testid="Body"]',
                'time': 'time'
            },
            'rate_limit': 30
        },
        {
            'name': 'Investing.com Gold News',
            'url': 'https://www.investing.com/news/commodities-news',
            'selectors': {
                'articles': 'article.js-article-item',
                'title': 'a.title',
                'content': 'p',
                'time': 'time'
            },
            'rate_limit': 30
        },
        {
            'name': 'Bloomberg Commodities',
            'url': 'https://www.bloomberg.com/markets/commodities',
            'selectors': {
                'articles': 'article',
                'title': 'h3',
                'content': 'p',
                'time': 'time'
            },
            'rate_limit': 20
        }
    ],
    'central_bank_news': [
        {
            'name': 'Federal Reserve',
            'url': 'https://www.federalreserve.gov/newsevents/pressreleases.htm',
            'selectors': {
                'articles': 'div.row',
                'title': 'h3',
                'content': 'div.col-md-8',
                'time': 'time'
            },
            'rate_limit': 10
        },
        {
            'name': 'ECB Press',
            'url': 'https://www.ecb.europa.eu/press/html/index.en.html',
            'selectors': {
                'articles': 'article',
                'title': 'h3',
                'content': 'p',
                'time': 'time'
            },
            'rate_limit': 10
        }
    ]
}

# Economic Indicators Configuration
ECONOMIC_INDICATORS = {
    'primary_indicators': [
        {
            'name': 'US_CPI',
            'fred_series': 'CPIAUCSL',
            'description': 'Consumer Price Index for All Urban Consumers',
            'impact': 'high',
            'frequency': 'monthly',
            'relevance': 0.9
        },
        {
            'name': 'US_UNEMPLOYMENT',
            'fred_series': 'UNRATE',
            'description': 'Unemployment Rate',
            'impact': 'medium',
            'frequency': 'monthly',
            'relevance': 0.7
        },
        {
            'name': 'US_GDP_GROWTH',
            'fred_series': 'A191RL1Q225SBEA',
            'description': 'Real GDP Growth Rate',
            'impact': 'high',
            'frequency': 'quarterly',
            'relevance': 0.8
        },
        {
            'name': 'US_INTEREST_RATE',
            'fred_series': 'FEDFUNDS',
            'description': 'Federal Funds Rate',
            'impact': 'high',
            'frequency': 'monthly',
            'relevance': 0.95
        },
        {
            'name': 'US_DOLLAR_INDEX',
            'fred_series': 'DTWEXBGS',
            'description': 'Trade Weighted US Dollar Index',
            'impact': 'high',
            'frequency': 'daily',
            'relevance': 0.9
        }
    ],
    'secondary_indicators': [
        {
            'name': 'US_INDUSTRIAL_PRODUCTION',
            'fred_series': 'INDPRO',
            'description': 'Industrial Production Index',
            'impact': 'medium',
            'frequency': 'monthly',
            'relevance': 0.6
        },
        {
            'name': 'US_RETAIL_SALES',
            'fred_series': 'RSAFS',
            'description': 'Advance Retail Sales',
            'impact': 'medium',
            'frequency': 'monthly',
            'relevance': 0.5
        },
        {
            'name': 'US_HOUSING_STARTS',
            'fred_series': 'HOUST',
            'description': 'Housing Starts',
            'impact': 'low',
            'frequency': 'monthly',
            'relevance': 0.4
        }
    ]
}

# Technical Analysis Configuration
TECHNICAL_INDICATORS = {
    'trend_indicators': [
        {'name': 'SMA_20', 'period': 20, 'type': 'simple_moving_average'},
        {'name': 'SMA_50', 'period': 50, 'type': 'simple_moving_average'},
        {'name': 'EMA_20', 'period': 20, 'type': 'exponential_moving_average'},
        {'name': 'EMA_50', 'period': 50, 'type': 'exponential_moving_average'},
        {'name': 'MACD', 'fast': 12, 'slow': 26, 'signal': 9, 'type': 'macd'}
    ],
    'momentum_indicators': [
        {'name': 'RSI_14', 'period': 14, 'type': 'rsi'},
        {'name': 'RSI_21', 'period': 21, 'type': 'rsi'},
        {'name': 'STOCH_K', 'k_period': 14, 'type': 'stochastic'},
        {'name': 'WILLIAMS_R', 'period': 14, 'type': 'williams_r'}
    ],
    'volatility_indicators': [
        {'name': 'BOLLINGER_BANDS', 'period': 20, 'std_dev': 2, 'type': 'bollinger_bands'},
        {'name': 'ATR_14', 'period': 14, 'type': 'average_true_range'},
        {'name': 'KELTNER_CHANNELS', 'period': 20, 'multiplier': 2, 'type': 'keltner_channels'}
    ],
    'volume_indicators': [
        {'name': 'VOLUME_SMA', 'period': 20, 'type': 'volume_sma'},
        {'name': 'OBV', 'type': 'on_balance_volume'},
        {'name': 'VOLUME_RATE', 'type': 'volume_rate_of_change'}
    ]
}

# Cache TTL Configuration (in seconds)
CACHE_TTL = {
    'price_data': {
        '1m': 60,       # 1 minute
        '5m': 300,      # 5 minutes  
        '15m': 900,     # 15 minutes
        '1h': 3600,     # 1 hour
        '4h': 14400,    # 4 hours
        '1d': 86400     # 1 day
    },
    'news_data': 3600,          # 1 hour
    'economic_data': 14400,     # 4 hours
    'technical_indicators': 300, # 5 minutes
    'sentiment_data': 1800,     # 30 minutes
    'market_data': 300          # 5 minutes
}

# Rate Limiting Configuration
RATE_LIMITS = {
    'gold_api': {'requests': 100, 'period': 3600},     # 100 requests per hour
    'alpha_vantage': {'requests': 5, 'period': 60},     # 5 requests per minute
    'newsapi': {'requests': 1000, 'period': 86400},     # 1000 requests per day
    'fred_api': {'requests': 120, 'period': 3600},      # 120 requests per hour
    'world_bank': {'requests': 120, 'period': 3600},    # 120 requests per hour
    'scraping': {'requests': 30, 'period': 3600},       # 30 requests per hour for scraping
    'finnhub': {'requests': 60, 'period': 60}           # 60 requests per minute
}

# Feature Engineering Configuration
FEATURE_CONFIG = {
    'price_features': [
        'current_price', 'price_change', 'price_change_percent',
        'daily_high', 'daily_low', 'daily_range', 'daily_range_percent',
        'volume', 'volume_change', 'volume_ratio',
        'volatility_1d', 'volatility_7d', 'volatility_30d',
        'momentum_1d', 'momentum_7d', 'momentum_30d'
    ],
    'technical_features': [
        'sma_20', 'sma_50', 'ema_20', 'ema_50',
        'rsi_14', 'rsi_21', 'macd_line', 'macd_signal', 'macd_histogram',
        'bollinger_upper', 'bollinger_middle', 'bollinger_lower',
        'bollinger_position', 'atr_14', 'stoch_k', 'stoch_d', 'williams_r'
    ],
    'sentiment_features': [
        'news_sentiment_1h', 'news_sentiment_4h', 'news_sentiment_24h',
        'news_count_1h', 'news_count_4h', 'news_count_24h',
        'news_relevance_avg', 'sentiment_change_1h', 'sentiment_change_4h'
    ],
    'economic_features': [
        'usd_index', 'interest_rates', 'inflation_rate', 'unemployment_rate',
        'gdp_growth', 'vix_index', 'bond_yields_10y', 'bond_yields_2y',
        'oil_price', 'crypto_correlation'
    ],
    'time_features': [
        'hour_of_day', 'day_of_week', 'day_of_month', 'month_of_year',
        'is_weekend', 'is_market_open', 'is_london_session', 'is_ny_session',
        'is_asian_session', 'time_to_market_open', 'time_to_market_close'
    ]
}

# Global configuration instance
config = DataSourcesConfig()

# Export key configurations
__all__ = [
    'DataSourcesConfig', 'APIConfig', 'DataSourceConfig',
    'NEWS_SOURCES', 'ECONOMIC_INDICATORS', 'TECHNICAL_INDICATORS',
    'CACHE_TTL', 'RATE_LIMITS', 'FEATURE_CONFIG', 'config'
]
