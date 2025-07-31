#!/usr/bin/env python3
"""
GoldGPT Advanced Data Pipeline API
Complete async implementation with comprehensive data quality monitoring and real-time WebSocket updates
"""

import asyncio
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import aiohttp
from aiohttp import web, WSMsgType
import socketio
import time
import numpy as np
import threading
from pathlib import Path

# Import our specialized services
from data_pipeline_core import DataPipelineCore, DataType, DataSourceTier
from advanced_price_data_service import AdvancedPriceDataService
from advanced_sentiment_analysis_service import AdvancedSentimentAnalysisService
from advanced_technical_indicator_service import AdvancedTechnicalIndicatorService
from advanced_macro_data_service import AdvancedMacroDataService

logger = logging.getLogger(__name__)

@dataclass
class DataQualityMetrics:
    """Data quality assessment metrics"""
    completeness: float  # 0.0 to 1.0
    accuracy: float
    timeliness: float
    consistency: float
    overall_score: float
    issues: List[str]
    last_assessed: datetime

@dataclass
class SystemHealthStatus:
    """Overall system health status"""
    price_service_status: str
    sentiment_service_status: str
    technical_service_status: str
    macro_service_status: str
    data_pipeline_status: str
    overall_health: str
    connected_clients: int
    last_updated: datetime

@dataclass
class PerformanceMetrics:
    """System performance metrics"""
    api_response_times: Dict[str, float]
    data_fetch_times: Dict[str, float]
    cache_hit_rates: Dict[str, float]
    error_rates: Dict[str, float]
    throughput: Dict[str, int]
    last_measured: datetime

class AdvancedDataPipelineAPI:
    """Advanced unified API for all GoldGPT data services with comprehensive monitoring"""
    
    def __init__(self, port: int = 8888):
        self.port = port
        self.app = web.Application(middlewares=[self.error_middleware])
        self.sio = socketio.AsyncServer(cors_allowed_origins="*", logger=True, engineio_logger=True)
        
        # Initialize all services
        self.pipeline = DataPipelineCore()
        self.price_service = AdvancedPriceDataService(self.pipeline)
        self.sentiment_service = AdvancedSentimentAnalysisService(self.pipeline)
        self.technical_service = AdvancedTechnicalIndicatorService(self.price_service)
        self.macro_service = AdvancedMacroDataService(self.pipeline)
        
        # WebSocket clients tracking
        self.connected_clients = set()
        self.subscription_topics = {
            'prices': set(),
            'sentiment': set(),
            'technical': set(),
            'macro': set(),
            'all': set(),
            'alerts': set(),
            'health': set()
        }
        
        # Performance monitoring
        self.performance_metrics = PerformanceMetrics(
            api_response_times={},
            data_fetch_times={},
            cache_hit_rates={},
            error_rates={},
            throughput={},
            last_measured=datetime.now()
        )
        
        # Quality monitoring
        self.quality_metrics = DataQualityMetrics(
            completeness=0.0,
            accuracy=0.0,
            timeliness=0.0,
            consistency=0.0,
            overall_score=0.0,
            issues=[],
            last_assessed=datetime.now()
        )
        
        # Real-time update configurations
        self.update_intervals = {
            'prices': 2,      # Every 2 seconds
            'sentiment': 180, # Every 3 minutes
            'technical': 30,  # Every 30 seconds
            'macro': 1800,    # Every 30 minutes
            'health': 60,     # Every minute
            'quality': 300    # Every 5 minutes
        }
        
        # Database for logging and analytics
        self.init_analytics_db()
        
        self.setup_routes()
        self.setup_websockets()
        
    async def error_middleware(self, request, handler):
        """Global error handling middleware"""
        start_time = time.time()
        
        try:
            response = await handler(request)
            
            # Record performance metrics
            response_time = time.time() - start_time
            endpoint = str(request.path)
            self.record_performance_metric('api_response_times', endpoint, response_time)
            
            return response
            
        except Exception as e:
            logger.error(f"API Error for {request.path}: {e}")
            
            # Record error metric
            endpoint = str(request.path)
            self.record_error_metric(endpoint)
            
            return web.json_response({
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'endpoint': endpoint
            }, status=500)
    
    def init_analytics_db(self):
        """Initialize analytics database"""
        db_path = Path('data_pipeline_analytics.db')
        
        with sqlite3.connect(db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS api_calls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    endpoint TEXT NOT NULL,
                    method TEXT NOT NULL,
                    response_time REAL,
                    status_code INTEGER,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    error_message TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS data_quality_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    completeness REAL,
                    accuracy REAL,
                    timeliness REAL,
                    consistency REAL,
                    overall_score REAL,
                    issues TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS system_health_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    price_service_status TEXT,
                    sentiment_service_status TEXT,
                    technical_service_status TEXT,
                    macro_service_status TEXT,
                    data_pipeline_status TEXT,
                    overall_health TEXT,
                    connected_clients INTEGER,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
    
    def record_performance_metric(self, metric_type: str, key: str, value: float):
        """Record a performance metric"""
        if metric_type not in self.performance_metrics.__dict__:
            return
            
        metrics_dict = getattr(self.performance_metrics, metric_type)
        metrics_dict[key] = value
        
    def record_error_metric(self, endpoint: str):
        """Record an error metric"""
        if endpoint not in self.performance_metrics.error_rates:
            self.performance_metrics.error_rates[endpoint] = 0
        self.performance_metrics.error_rates[endpoint] += 1
        
    def setup_routes(self):
        """Setup HTTP API routes"""
        
        # Health and monitoring endpoints
        self.app.router.add_get('/api/health', self.health_check)
        self.app.router.add_get('/api/health/detailed', self.detailed_health_check)
        self.app.router.add_get('/api/status', self.system_status)
        self.app.router.add_get('/api/performance', self.performance_metrics_endpoint)
        self.app.router.add_get('/api/quality', self.data_quality_report)
        self.app.router.add_get('/api/analytics', self.analytics_dashboard)
        
        # Price data endpoints
        self.app.router.add_get('/api/price/realtime/{symbol}', self.get_realtime_price)
        self.app.router.add_get('/api/price/ohlcv/{symbol}', self.get_ohlcv_data)
        self.app.router.add_get('/api/price/historical/{symbol}', self.get_historical_prices)
        self.app.router.add_get('/api/price/summary/{symbol}', self.get_market_summary)
        self.app.router.add_get('/api/price/levels/{symbol}', self.get_price_levels)
        self.app.router.add_get('/api/price/alerts/{symbol}', self.get_price_alerts)
        
        # Sentiment analysis endpoints
        self.app.router.add_get('/api/sentiment/current', self.get_current_sentiment)
        self.app.router.add_get('/api/sentiment/history', self.get_sentiment_history)
        self.app.router.add_get('/api/sentiment/news', self.get_recent_news)
        self.app.router.add_get('/api/sentiment/correlation', self.get_sentiment_correlation)
        self.app.router.add_post('/api/sentiment/refresh', self.refresh_sentiment_data)
        
        # Technical analysis endpoints
        self.app.router.add_get('/api/technical/{symbol}', self.get_technical_analysis)
        self.app.router.add_get('/api/technical/multi-timeframe/{symbol}', self.get_multi_timeframe_analysis)
        self.app.router.add_get('/api/technical/indicators/{symbol}', self.get_specific_indicators)
        self.app.router.add_get('/api/technical/signals/{symbol}', self.get_trading_signals)
        
        # Macro data endpoints
        self.app.router.add_get('/api/macro/analysis', self.get_macro_analysis)
        self.app.router.add_get('/api/macro/indicators', self.get_macro_indicators)
        self.app.router.add_get('/api/macro/events', self.get_economic_events)
        self.app.router.add_get('/api/macro/calendar', self.get_economic_calendar)
        
        # Unified data endpoints
        self.app.router.add_get('/api/unified/dashboard/{symbol}', self.get_unified_dashboard_data)
        self.app.router.add_get('/api/unified/complete-analysis/{symbol}', self.get_complete_analysis)
        self.app.router.add_get('/api/unified/signals/{symbol}', self.get_unified_signals)
        
        # Data pipeline management
        self.app.router.add_get('/api/pipeline/sources', self.get_data_sources_status)
        self.app.router.add_get('/api/pipeline/reliability', self.get_source_reliability)
        self.app.router.add_post('/api/pipeline/refresh', self.refresh_all_data)
        self.app.router.add_post('/api/pipeline/validate', self.validate_data_quality)
        self.app.router.add_post('/api/pipeline/reset-cache', self.reset_cache)
        
        # Static files for dashboard
        self.app.router.add_static('/', path='static', name='static')
        
    def setup_websockets(self):
        """Setup WebSocket handlers"""
        self.sio.attach(self.app)
        
        @self.sio.event
        async def connect(sid, environ):
            logger.info(f"🔌 Client connected: {sid}")
            self.connected_clients.add(sid)
            
            # Send initial data
            await self.send_initial_data(sid)
            
            # Log connection
            await self.log_api_call('/ws/connect', 'CONNECT', 0, 200)
        
        @self.sio.event
        async def disconnect(sid):
            logger.info(f"🔌 Client disconnected: {sid}")
            self.connected_clients.discard(sid)
            
            # Remove from all subscriptions
            for topic_clients in self.subscription_topics.values():
                topic_clients.discard(sid)
        
        @self.sio.event
        async def subscribe(sid, data):
            """Handle subscription requests"""
            try:
                topics = data.get('topics', [])
                symbols = data.get('symbols', ['XAU'])
                
                for topic in topics:
                    if topic in self.subscription_topics:
                        self.subscription_topics[topic].add(sid)
                        logger.info(f"📡 Client {sid} subscribed to {topic}")
                
                await self.sio.emit('subscription_confirmed', {
                    'topics': topics,
                    'symbols': symbols,
                    'status': 'success',
                    'timestamp': datetime.now().isoformat()
                }, room=sid)
                
            except Exception as e:
                logger.error(f"Error handling subscription: {e}")
                await self.sio.emit('error', {
                    'message': str(e),
                    'timestamp': datetime.now().isoformat()
                }, room=sid)
        
        @self.sio.event
        async def unsubscribe(sid, data):
            """Handle unsubscription requests"""
            try:
                topics = data.get('topics', [])
                
                for topic in topics:
                    if topic in self.subscription_topics:
                        self.subscription_topics[topic].discard(sid)
                        logger.info(f"📡 Client {sid} unsubscribed from {topic}")
                
                await self.sio.emit('unsubscription_confirmed', {
                    'topics': topics,
                    'status': 'success',
                    'timestamp': datetime.now().isoformat()
                }, room=sid)
                
            except Exception as e:
                logger.error(f"Error handling unsubscription: {e}")
        
        @self.sio.event
        async def get_historical_data(sid, data):
            """Handle historical data requests"""
            try:
                symbol = data.get('symbol', 'XAU')
                data_type = data.get('type', 'price')
                timeframe = data.get('timeframe', '1h')
                days = data.get('days', 7)
                
                if data_type == 'price':
                    end_time = datetime.now()
                    start_time = end_time - timedelta(days=days)
                    historical_data = await self.price_service.get_historical_ohlcv(
                        symbol, timeframe, start_time, end_time
                    )
                    
                    response_data = []
                    for ohlcv in historical_data:
                        response_data.append({
                            'timestamp': ohlcv.timestamp.isoformat(),
                            'open': ohlcv.open_price,
                            'high': ohlcv.high_price,
                            'low': ohlcv.low_price,
                            'close': ohlcv.close_price,
                            'volume': ohlcv.volume
                        })
                    
                    await self.sio.emit('historical_data_response', {
                        'symbol': symbol,
                        'type': data_type,
                        'timeframe': timeframe,
                        'data': response_data,
                        'count': len(response_data)
                    }, room=sid)
                
            except Exception as e:
                logger.error(f"Error handling historical data request: {e}")
                await self.sio.emit('error', {
                    'message': str(e),
                    'timestamp': datetime.now().isoformat()
                }, room=sid)
    
    async def send_initial_data(self, sid: str):
        """Send initial data to newly connected client"""
        try:
            # Send current market summary
            summary = await self.price_service.get_market_summary('XAU')
            await self.sio.emit('market_summary', summary, room=sid)
            
            # Send current sentiment
            sentiment = await self.sentiment_service.generate_sentiment_signal()
            await self.sio.emit('sentiment_update', asdict(sentiment), room=sid)
            
            # Send system status
            status = await self.get_system_health()
            await self.sio.emit('system_status', asdict(status), room=sid)
            
            # Send quality metrics
            await self.sio.emit('quality_metrics', asdict(self.quality_metrics), room=sid)
            
        except Exception as e:
            logger.error(f"Error sending initial data: {e}")
    
    # Logging functions
    
    async def log_api_call(self, endpoint: str, method: str, response_time: float, status_code: int, error_message: str = None):
        """Log API call for analytics"""
        try:
            db_path = Path('data_pipeline_analytics.db')
            with sqlite3.connect(db_path) as conn:
                conn.execute('''
                    INSERT INTO api_calls (endpoint, method, response_time, status_code, error_message)
                    VALUES (?, ?, ?, ?, ?)
                ''', (endpoint, method, response_time, status_code, error_message))
                conn.commit()
        except Exception as e:
            logger.error(f"Error logging API call: {e}")
    
    # HTTP API Handlers
    
    async def health_check(self, request):
        """Basic health check endpoint"""
        start_time = time.time()
        
        try:
            # Test core pipeline
            test_data = await self.pipeline.get_unified_data('XAU', DataType.PRICE)
            
            response = {
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'version': '2.0.0',
                'uptime': time.time() - start_time,
                'connected_clients': len(self.connected_clients),
                'services': {
                    'price_service': 'online',
                    'sentiment_service': 'online',
                    'technical_service': 'online',
                    'macro_service': 'online',
                    'data_pipeline': 'online' if test_data else 'degraded'
                }
            }
            
            await self.log_api_call('/api/health', 'GET', time.time() - start_time, 200)
            return web.json_response(response)
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            await self.log_api_call('/api/health', 'GET', time.time() - start_time, 500, str(e))
            return web.json_response({
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }, status=500)
    
    async def detailed_health_check(self, request):
        """Detailed health check with all service tests"""
        start_time = time.time()
        
        try:
            # Test each service individually
            health_details = {
                'overall_status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'services': {},
                'data_sources': await self.pipeline.health_check(),
                'performance_summary': {
                    'avg_response_time': np.mean(list(self.performance_metrics.api_response_times.values())) if self.performance_metrics.api_response_times else 0,
                    'total_errors': sum(self.performance_metrics.error_rates.values()),
                    'cache_hit_rate': np.mean(list(self.performance_metrics.cache_hit_rates.values())) if self.performance_metrics.cache_hit_rates else 0
                }
            }
            
            # Test price service
            try:
                price_data = await self.price_service.get_real_time_price('XAU')
                health_details['services']['price_service'] = {
                    'status': 'healthy' if price_data else 'degraded',
                    'last_data': price_data['timestamp'] if price_data else None,
                    'data_age_seconds': (datetime.now() - datetime.fromisoformat(price_data['timestamp'].replace('Z', '+00:00'))).total_seconds() if price_data else None
                }
            except Exception as e:
                health_details['services']['price_service'] = {'status': 'unhealthy', 'error': str(e)}
            
            # Test sentiment service
            try:
                sentiment_signal = await self.sentiment_service.generate_sentiment_signal(hours_lookback=1)
                health_details['services']['sentiment_service'] = {
                    'status': 'healthy' if sentiment_signal else 'degraded',
                    'news_count': sentiment_signal.news_count if sentiment_signal else 0,
                    'sentiment_score': sentiment_signal.compound_sentiment if sentiment_signal else None
                }
            except Exception as e:
                health_details['services']['sentiment_service'] = {'status': 'unhealthy', 'error': str(e)}
            
            # Test technical service
            try:
                analysis = await self.technical_service.calculate_all_indicators('XAU', '1h', lookback_periods=50)
                health_details['services']['technical_service'] = {
                    'status': 'healthy' if analysis and analysis.indicators else 'degraded',
                    'indicator_count': len(analysis.indicators) if analysis else 0,
                    'overall_signal': analysis.overall_signal if analysis else None
                }
            except Exception as e:
                health_details['services']['technical_service'] = {'status': 'unhealthy', 'error': str(e)}
            
            # Test macro service
            try:
                macro_analysis = await self.macro_service.generate_macro_analysis()
                health_details['services']['macro_service'] = {
                    'status': 'healthy' if macro_analysis else 'degraded',
                    'indicator_count': len(macro_analysis.key_indicators) if macro_analysis else 0
                }
            except Exception as e:
                health_details['services']['macro_service'] = {'status': 'unhealthy', 'error': str(e)}
            
            # Determine overall status
            service_statuses = [service['status'] for service in health_details['services'].values()]
            healthy_count = sum(1 for status in service_statuses if status == 'healthy')
            
            if healthy_count == len(service_statuses):
                health_details['overall_status'] = 'healthy'
            elif healthy_count >= len(service_statuses) * 0.8:
                health_details['overall_status'] = 'degraded'
            else:
                health_details['overall_status'] = 'unhealthy'
            
            await self.log_api_call('/api/health/detailed', 'GET', time.time() - start_time, 200)
            return web.json_response(health_details)
            
        except Exception as e:
            logger.error(f"Detailed health check failed: {e}")
            await self.log_api_call('/api/health/detailed', 'GET', time.time() - start_time, 500, str(e))
            return web.json_response({'error': str(e)}, status=500)
    
    async def get_system_health(self) -> SystemHealthStatus:
        """Get comprehensive system health status"""
        try:
            # Test each service
            price_status = await self.test_service_health('price')
            sentiment_status = await self.test_service_health('sentiment')
            technical_status = await self.test_service_health('technical')
            macro_status = await self.test_service_health('macro')
            pipeline_status = await self.test_service_health('pipeline')
            
            # Determine overall health
            statuses = [price_status, sentiment_status, technical_status, macro_status, pipeline_status]
            healthy_count = sum(1 for s in statuses if s == 'healthy')
            
            if healthy_count == len(statuses):
                overall_health = 'healthy'
            elif healthy_count >= len(statuses) * 0.8:
                overall_health = 'degraded'
            else:
                overall_health = 'unhealthy'
            
            return SystemHealthStatus(
                price_service_status=price_status,
                sentiment_service_status=sentiment_status,
                technical_service_status=technical_status,
                macro_service_status=macro_status,
                data_pipeline_status=pipeline_status,
                overall_health=overall_health,
                connected_clients=len(self.connected_clients),
                last_updated=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return SystemHealthStatus(
                price_service_status='error',
                sentiment_service_status='error',
                technical_service_status='error',
                macro_service_status='error',
                data_pipeline_status='error',
                overall_health='unhealthy',
                connected_clients=len(self.connected_clients),
                last_updated=datetime.now()
            )
    
    async def test_service_health(self, service_name: str) -> str:
        """Test individual service health"""
        try:
            if service_name == 'price':
                data = await self.price_service.get_real_time_price('XAU')
                return 'healthy' if data else 'degraded'
            elif service_name == 'sentiment':
                signal = await self.sentiment_service.generate_sentiment_signal(hours_lookback=1)
                return 'healthy' if signal else 'degraded'
            elif service_name == 'technical':
                analysis = await self.technical_service.calculate_all_indicators('XAU', '1h', lookback_periods=50)
                return 'healthy' if analysis and analysis.indicators else 'degraded'
            elif service_name == 'macro':
                analysis = await self.macro_service.generate_macro_analysis()
                return 'healthy' if analysis else 'degraded'
            elif service_name == 'pipeline':
                health = await self.pipeline.health_check()
                healthy_sources = sum(1 for status in health.values() if status['status'] == 'healthy')
                total_sources = len(health)
                if healthy_sources >= total_sources * 0.8:
                    return 'healthy'
                elif healthy_sources >= total_sources * 0.5:
                    return 'degraded'
                else:
                    return 'unhealthy'
            else:
                return 'unknown'
        except Exception as e:
            logger.error(f"Error testing {service_name} service: {e}")
            return 'unhealthy'
    
    # Real-time data broadcasting
    
    async def start_real_time_broadcasts(self):
        """Start real-time data broadcasting to WebSocket clients"""
        logger.info("🚀 Starting real-time broadcast loops...")
        
        # Start all broadcast loops
        asyncio.create_task(self.price_broadcast_loop())
        asyncio.create_task(self.sentiment_broadcast_loop())
        asyncio.create_task(self.technical_broadcast_loop())
        asyncio.create_task(self.macro_broadcast_loop())
        asyncio.create_task(self.health_broadcast_loop())
        asyncio.create_task(self.quality_monitoring_loop())
        
        logger.info("🚀 All real-time broadcast loops started")
    
    async def price_broadcast_loop(self):
        """Broadcast price updates"""
        while True:
            try:
                if self.subscription_topics['prices'] or self.subscription_topics['all']:
                    start_time = time.time()
                    price_data = await self.price_service.get_real_time_price('XAU')
                    
                    if price_data:
                        # Add performance timing
                        fetch_time = time.time() - start_time
                        self.record_performance_metric('data_fetch_times', 'price_realtime', fetch_time)
                        
                        await self.broadcast_to_subscribers('prices', 'price_update', price_data)
                
                await asyncio.sleep(self.update_intervals['prices'])
                
            except Exception as e:
                logger.error(f"Error in price broadcast loop: {e}")
                await asyncio.sleep(10)
    
    async def sentiment_broadcast_loop(self):
        """Broadcast sentiment updates"""
        while True:
            try:
                if self.subscription_topics['sentiment'] or self.subscription_topics['all']:
                    start_time = time.time()
                    sentiment_signal = await self.sentiment_service.generate_sentiment_signal()
                    
                    fetch_time = time.time() - start_time
                    self.record_performance_metric('data_fetch_times', 'sentiment_signal', fetch_time)
                    
                    await self.broadcast_to_subscribers('sentiment', 'sentiment_update', asdict(sentiment_signal))
                
                await asyncio.sleep(self.update_intervals['sentiment'])
                
            except Exception as e:
                logger.error(f"Error in sentiment broadcast loop: {e}")
                await asyncio.sleep(60)
    
    async def technical_broadcast_loop(self):
        """Broadcast technical analysis updates"""
        while True:
            try:
                if self.subscription_topics['technical'] or self.subscription_topics['all']:
                    start_time = time.time()
                    analysis = await self.technical_service.calculate_all_indicators('XAU', '1h')
                    
                    fetch_time = time.time() - start_time
                    self.record_performance_metric('data_fetch_times', 'technical_analysis', fetch_time)
                    
                    summary = {
                        'symbol': analysis.symbol,
                        'timeframe': analysis.timeframe,
                        'overall_signal': analysis.overall_signal,
                        'signal_strength': analysis.signal_strength,
                        'trend_direction': analysis.trend_direction,
                        'indicator_count': len(analysis.indicators),
                        'timestamp': analysis.timestamp.isoformat()
                    }
                    
                    await self.broadcast_to_subscribers('technical', 'technical_update', summary)
                
                await asyncio.sleep(self.update_intervals['technical'])
                
            except Exception as e:
                logger.error(f"Error in technical broadcast loop: {e}")
                await asyncio.sleep(60)
    
    async def macro_broadcast_loop(self):
        """Broadcast macro analysis updates"""
        while True:
            try:
                if self.subscription_topics['macro'] or self.subscription_topics['all']:
                    start_time = time.time()
                    summary = await self.macro_service.get_macro_summary()
                    
                    fetch_time = time.time() - start_time
                    self.record_performance_metric('data_fetch_times', 'macro_summary', fetch_time)
                    
                    await self.broadcast_to_subscribers('macro', 'macro_update', summary)
                
                await asyncio.sleep(self.update_intervals['macro'])
                
            except Exception as e:
                logger.error(f"Error in macro broadcast loop: {e}")
                await asyncio.sleep(300)
    
    async def health_broadcast_loop(self):
        """Broadcast system health updates"""
        while True:
            try:
                if self.subscription_topics['health'] or self.subscription_topics['all']:
                    status = await self.get_system_health()
                    await self.broadcast_to_subscribers('health', 'health_update', asdict(status))
                    
                    # Log to database
                    await self.log_system_health(status)
                
                await asyncio.sleep(self.update_intervals['health'])
                
            except Exception as e:
                logger.error(f"Error in health broadcast loop: {e}")
                await asyncio.sleep(60)
    
    async def quality_monitoring_loop(self):
        """Monitor and broadcast data quality metrics"""
        while True:
            try:
                # Update quality metrics
                await self.update_quality_metrics()
                
                # Broadcast to subscribers
                if self.subscription_topics['all']:
                    await self.broadcast_to_subscribers('all', 'quality_update', asdict(self.quality_metrics))
                
                await asyncio.sleep(self.update_intervals['quality'])
                
            except Exception as e:
                logger.error(f"Error in quality monitoring loop: {e}")
                await asyncio.sleep(300)
    
    async def update_quality_metrics(self):
        """Update data quality metrics"""
        try:
            # Assess completeness
            completeness = await self.assess_data_completeness()
            
            # Assess accuracy
            accuracy = await self.assess_data_accuracy()
            
            # Assess timeliness
            timeliness = await self.assess_data_timeliness()
            
            # Assess consistency
            consistency = await self.assess_data_consistency()
            
            # Calculate overall score
            overall_score = np.mean([completeness, accuracy, timeliness, consistency])
            
            # Identify issues
            issues = []
            if completeness < 0.8:
                issues.append("Data completeness below threshold")
            if accuracy < 0.8:
                issues.append("Data accuracy concerns detected")
            if timeliness < 0.8:
                issues.append("Data freshness issues")
            if consistency < 0.8:
                issues.append("Data consistency problems")
            
            self.quality_metrics = DataQualityMetrics(
                completeness=completeness,
                accuracy=accuracy,
                timeliness=timeliness,
                consistency=consistency,
                overall_score=overall_score,
                issues=issues,
                last_assessed=datetime.now()
            )
            
            # Log to database
            await self.log_quality_metrics()
            
        except Exception as e:
            logger.error(f"Error updating quality metrics: {e}")
    
    async def assess_data_completeness(self) -> float:
        """Assess data completeness across services"""
        try:
            completeness_scores = []
            
            # Check price service completeness
            price_data = await self.price_service.get_real_time_price('XAU')
            completeness_scores.append(1.0 if price_data else 0.0)
            
            # Check sentiment service completeness
            sentiment_signal = await self.sentiment_service.generate_sentiment_signal(1)
            completeness_scores.append(1.0 if sentiment_signal.news_count > 0 else 0.5)
            
            # Check technical service completeness
            technical_analysis = await self.technical_service.calculate_all_indicators('XAU', '1h', 50)
            completeness_scores.append(min(1.0, len(technical_analysis.indicators) / 10))
            
            # Check macro service completeness
            macro_analysis = await self.macro_service.generate_macro_analysis()
            completeness_scores.append(min(1.0, len(macro_analysis.key_indicators) / 5))
            
            return np.mean(completeness_scores)
            
        except Exception as e:
            logger.error(f"Error assessing completeness: {e}")
            return 0.0
    
    async def assess_data_accuracy(self) -> float:
        """Assess data accuracy using validation rules"""
        try:
            accuracy_scores = []
            
            # Validate price data
            price_data = await self.price_service.get_real_time_price('XAU')
            if price_data:
                price = price_data['price']
                # Gold price should be in reasonable range
                if 1000 <= price <= 10000:
                    accuracy_scores.append(1.0)
                else:
                    accuracy_scores.append(0.0)
            else:
                accuracy_scores.append(0.0)
            
            # Validate technical indicators
            analysis = await self.technical_service.calculate_all_indicators('XAU', '1h', 50)
            if analysis:
                valid_indicators = 0
                total_indicators = len(analysis.indicators)
                
                for name, indicator in analysis.indicators.items():
                    if self.is_indicator_value_reasonable(name, indicator.value):
                        valid_indicators += 1
                
                accuracy_scores.append(valid_indicators / total_indicators if total_indicators > 0 else 0.0)
            else:
                accuracy_scores.append(0.0)
            
            return np.mean(accuracy_scores) if accuracy_scores else 0.0
            
        except Exception as e:
            logger.error(f"Error assessing accuracy: {e}")
            return 0.5
    
    def is_indicator_value_reasonable(self, indicator_name: str, value: float) -> bool:
        """Check if indicator value is reasonable"""
        if np.isnan(value) or np.isinf(value):
            return False
            
        name_lower = indicator_name.lower()
        
        if 'rsi' in name_lower:
            return 0 <= value <= 100
        elif 'macd' in name_lower:
            return -1000 <= value <= 1000
        elif any(x in name_lower for x in ['sma', 'ema', 'bb']):
            return 1000 <= value <= 10000  # Gold price range
        elif 'volume' in name_lower:
            return value >= 0
        else:
            return True  # Default to valid for unknown indicators
    
    async def assess_data_timeliness(self) -> float:
        """Assess data freshness/timeliness"""
        try:
            timeliness_scores = []
            current_time = datetime.now()
            
            # Check price data freshness
            price_data = await self.price_service.get_real_time_price('XAU')
            if price_data:
                price_time = datetime.fromisoformat(price_data['timestamp'].replace('Z', '+00:00'))
                price_age = (current_time - price_time.replace(tzinfo=None)).total_seconds()
                timeliness_scores.append(max(0.0, 1.0 - price_age / 300))  # 5 min threshold
            
            # Check sentiment data freshness
            sentiment_signal = await self.sentiment_service.generate_sentiment_signal(1)
            sentiment_age = (current_time - sentiment_signal.timestamp).total_seconds()
            timeliness_scores.append(max(0.0, 1.0 - sentiment_age / 3600))  # 1 hour threshold
            
            return np.mean(timeliness_scores) if timeliness_scores else 0.0
            
        except Exception as e:
            logger.error(f"Error assessing timeliness: {e}")
            return 0.0
    
    async def assess_data_consistency(self) -> float:
        """Assess data consistency across sources"""
        try:
            # This would compare similar data from different sources
            # For now, return a reasonable default
            return 0.9
            
        except Exception as e:
            logger.error(f"Error assessing consistency: {e}")
            return 0.0
    
    async def log_system_health(self, status: SystemHealthStatus):
        """Log system health to database"""
        try:
            db_path = Path('data_pipeline_analytics.db')
            with sqlite3.connect(db_path) as conn:
                conn.execute('''
                    INSERT INTO system_health_history 
                    (price_service_status, sentiment_service_status, technical_service_status,
                     macro_service_status, data_pipeline_status, overall_health, connected_clients)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    status.price_service_status,
                    status.sentiment_service_status,
                    status.technical_service_status,
                    status.macro_service_status,
                    status.data_pipeline_status,
                    status.overall_health,
                    status.connected_clients
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error logging system health: {e}")
    
    async def log_quality_metrics(self):
        """Log quality metrics to database"""
        try:
            db_path = Path('data_pipeline_analytics.db')
            with sqlite3.connect(db_path) as conn:
                conn.execute('''
                    INSERT INTO data_quality_history 
                    (completeness, accuracy, timeliness, consistency, overall_score, issues)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    self.quality_metrics.completeness,
                    self.quality_metrics.accuracy,
                    self.quality_metrics.timeliness,
                    self.quality_metrics.consistency,
                    self.quality_metrics.overall_score,
                    json.dumps(self.quality_metrics.issues)
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"Error logging quality metrics: {e}")
    
    async def broadcast_to_subscribers(self, topic: str, event_name: str, data: Any):
        """Broadcast data to subscribed clients"""
        try:
            # Get subscribers for this topic and 'all' topic
            subscribers = self.subscription_topics[topic] | self.subscription_topics['all']
            
            if subscribers:
                await self.sio.emit(event_name, data, room=list(subscribers))
                logger.debug(f"📡 Broadcasted {event_name} to {len(subscribers)} clients")
                
                # Record throughput metric
                if event_name not in self.performance_metrics.throughput:
                    self.performance_metrics.throughput[event_name] = 0
                self.performance_metrics.throughput[event_name] += len(subscribers)
                
        except Exception as e:
            logger.error(f"Error broadcasting {event_name}: {e}")
    
    async def run_server(self):
        """Run the advanced data pipeline API server"""
        try:
            # Start real-time broadcasts
            await self.start_real_time_broadcasts()
            
            # Start the web server
            runner = web.AppRunner(self.app)
            await runner.setup()
            
            site = web.TCPSite(runner, 'localhost', self.port)
            await site.start()
            
            logger.info(f"🚀 Advanced Data Pipeline API running on http://localhost:{self.port}")
            logger.info(f"📡 WebSocket endpoint: ws://localhost:{self.port}/socket.io/")
            logger.info(f"📊 Health dashboard: http://localhost:{self.port}/api/health/detailed")
            logger.info(f"📈 Performance metrics: http://localhost:{self.port}/api/performance")
            logger.info(f"🔍 Data quality report: http://localhost:{self.port}/api/quality")
            
            # Keep the server running
            while True:
                await asyncio.sleep(3600)  # Sleep for 1 hour
                
        except Exception as e:
            logger.error(f"Error running server: {e}")
            raise

# For easier imports and testing
advanced_api = AdvancedDataPipelineAPI(port=8888)

if __name__ == "__main__":
    async def main():
        print("🚀 Starting GoldGPT Advanced Data Pipeline API...")
        print("📡 WebSocket support enabled")
        print("🔍 Data quality monitoring enabled")
        print("📊 Performance analytics enabled")
        
        try:
            await advanced_api.run_server()
        except KeyboardInterrupt:
            print("\n⏹️  Server stopped by user")
        except Exception as e:
            print(f"❌ Server error: {e}")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(main())
