#!/usr/bin/env python3
"""
GoldGPT Data Pipeline Web Integration
Integrates the advanced data pipeline core into the web application's ML and signal systems
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from flask import Flask, jsonify, request
from flask_socketio import SocketIO
import threading
import time

# Import our data pipeline core
from data_pipeline_core import data_pipeline, DataType, DataPoint

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebDataPipelineIntegration:
    """Integration layer between data pipeline core and web application"""
    
    def __init__(self, app: Flask = None, socketio: SocketIO = None):
        self.app = app
        self.socketio = socketio
        self.pipeline = data_pipeline
        self.is_streaming = False
        self.streaming_thread = None
        self.last_price_update = None
        
        # WebSocket streaming configuration
        self.stream_config = {
            'price_interval': 1,     # seconds between price updates
            'status_interval': 30,   # seconds between status updates
            'health_check_interval': 60  # seconds between health checks
        }
        
        if app:
            self.register_routes(app)
    
    def register_routes(self, app: Flask):
        """Register data pipeline routes with Flask app"""
        
        @app.route('/api/data-pipeline/price/<symbol>')
        async def get_unified_price(symbol):
            """Get unified price data from best available sources"""
            try:
                data = await self.pipeline.get_unified_data(symbol, DataType.PRICE)
                if data:
                    return jsonify({
                        'success': True,
                        'symbol': symbol,
                        'price': data.get('price'),
                        'timestamp': data.get('timestamp'),
                        'source': data.get('source'),
                        'confidence': data.get('confidence'),
                        'is_simulated': data.get('is_simulated', False)
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': 'No data available from any source'
                    }), 503
                    
            except Exception as e:
                logger.error(f"Error getting unified price for {symbol}: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @app.route('/api/data-pipeline/sources/status')
        def get_sources_status():
            """Get status of all data sources"""
            try:
                status = self.pipeline.get_source_status()
                return jsonify({
                    'success': True,
                    'sources': status,
                    'total_sources': len(status),
                    'healthy_sources': sum(1 for s in status.values() if s.get('reliability_score', 0) > 0.5)
                })
            except Exception as e:
                logger.error(f"Error getting sources status: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @app.route('/api/data-pipeline/health-check')
        async def perform_health_check():
            """Perform health check on all data sources"""
            try:
                health = await self.pipeline.health_check()
                return jsonify({
                    'success': True,
                    'health_check': health,
                    'timestamp': datetime.now().isoformat(),
                    'healthy_count': sum(1 for h in health.values() if h.get('status') == 'healthy')
                })
            except Exception as e:
                logger.error(f"Error performing health check: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @app.route('/api/data-pipeline/stream/start', methods=['POST'])
        def start_price_streaming():
            """Start real-time price streaming via WebSocket"""
            try:
                if not self.is_streaming:
                    self.start_streaming()
                    return jsonify({
                        'success': True,
                        'message': 'Price streaming started',
                        'config': self.stream_config
                    })
                else:
                    return jsonify({
                        'success': True,
                        'message': 'Price streaming already active'
                    })
            except Exception as e:
                logger.error(f"Error starting price streaming: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @app.route('/api/data-pipeline/stream/stop', methods=['POST'])
        def stop_price_streaming():
            """Stop real-time price streaming"""
            try:
                self.stop_streaming()
                return jsonify({
                    'success': True,
                    'message': 'Price streaming stopped'
                })
            except Exception as e:
                logger.error(f"Error stopping price streaming: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
    
    async def get_enhanced_price_data(self, symbol: str = 'XAU') -> Dict[str, Any]:
        """Get enhanced price data with multiple source validation"""
        try:
            # Get unified data from pipeline
            data = await self.pipeline.get_unified_data(symbol, DataType.PRICE)
            
            if data:
                # Enhance with additional metadata
                enhanced_data = {
                    'symbol': symbol,
                    'price': data.get('price'),
                    'timestamp': data.get('timestamp'),
                    'source': data.get('source'),
                    'confidence': data.get('confidence'),
                    'is_simulated': data.get('is_simulated', False),
                    'quality_score': self.calculate_data_quality_score(data),
                    'source_tier': self.get_source_tier(data.get('source')),
                    'last_update': datetime.now().isoformat()
                }
                
                # Add price validation flags
                enhanced_data.update(self.validate_price_data(data))
                
                return enhanced_data
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting enhanced price data: {e}")
            return None
    
    def calculate_data_quality_score(self, data: Dict[str, Any]) -> float:
        """Calculate overall data quality score"""
        score = 0.0
        
        # Base confidence score
        confidence = data.get('confidence', 0)
        score += confidence * 0.4
        
        # Source reliability
        source = data.get('source', '')
        if source in self.pipeline.source_reliability:
            reliability = self.pipeline.source_reliability[source].get('score', 0)
            score += reliability * 0.3
        
        # Freshness score (how recent is the data)
        timestamp_str = data.get('timestamp')
        if timestamp_str:
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                age = datetime.now() - timestamp.replace(tzinfo=None)
                freshness = max(0, 1 - (age.total_seconds() / 300))  # 5 minute decay
                score += freshness * 0.2
            except:
                pass
        
        # Simulation penalty
        if data.get('is_simulated', False):
            score *= 0.5
        
        # Price validity
        price = data.get('price', 0)
        if 1000 <= price <= 10000:  # Reasonable gold price range
            score += 0.1
        
        return min(1.0, score)
    
    def get_source_tier(self, source: str) -> str:
        """Get source tier information"""
        if source in self.pipeline.source_configs:
            return self.pipeline.source_configs[source].tier.value
        return 'unknown'
    
    def validate_price_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate price data and return validation flags"""
        validations = {
            'price_valid': False,
            'timestamp_valid': False,
            'source_reliable': False,
            'within_range': False
        }
        
        # Price validation
        price = data.get('price', 0)
        if price and price > 0:
            validations['price_valid'] = True
            
            # Range validation for gold
            if 1000 <= price <= 10000:
                validations['within_range'] = True
        
        # Timestamp validation
        timestamp_str = data.get('timestamp')
        if timestamp_str:
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                age = datetime.now() - timestamp.replace(tzinfo=None)
                if age < timedelta(minutes=10):
                    validations['timestamp_valid'] = True
            except:
                pass
        
        # Source reliability
        source = data.get('source', '')
        if source in self.pipeline.source_reliability:
            reliability = self.pipeline.source_reliability[source].get('score', 0)
            if reliability > 0.7:
                validations['source_reliable'] = True
        
        return validations
    
    def start_streaming(self):
        """Start real-time data streaming via WebSocket"""
        if self.socketio and not self.is_streaming:
            self.is_streaming = True
            self.streaming_thread = threading.Thread(target=self._streaming_worker)
            self.streaming_thread.daemon = True
            self.streaming_thread.start()
            logger.info("✅ Started data pipeline streaming")
    
    def stop_streaming(self):
        """Stop real-time data streaming"""
        self.is_streaming = False
        if self.streaming_thread:
            self.streaming_thread.join(timeout=5)
        logger.info("🛑 Stopped data pipeline streaming")
    
    def _streaming_worker(self):
        """Worker thread for real-time data streaming"""
        logger.info("🚀 Data pipeline streaming worker started")
        
        last_price_update = 0
        last_status_update = 0
        last_health_check = 0
        
        while self.is_streaming:
            try:
                current_time = time.time()
                
                # Price updates
                if current_time - last_price_update >= self.stream_config['price_interval']:
                    asyncio.run(self._emit_price_update())
                    last_price_update = current_time
                
                # Status updates
                if current_time - last_status_update >= self.stream_config['status_interval']:
                    self._emit_status_update()
                    last_status_update = current_time
                
                # Health checks
                if current_time - last_health_check >= self.stream_config['health_check_interval']:
                    asyncio.run(self._emit_health_check())
                    last_health_check = current_time
                
                time.sleep(0.5)  # Small sleep to prevent CPU overload
                
            except Exception as e:
                logger.error(f"Error in streaming worker: {e}")
                time.sleep(5)  # Longer sleep on error
    
    async def _emit_price_update(self):
        """Emit price update via WebSocket"""
        try:
            enhanced_data = await self.get_enhanced_price_data('XAU')
            if enhanced_data and self.socketio:
                self.socketio.emit('price_update', enhanced_data)
                self.last_price_update = enhanced_data
        except Exception as e:
            logger.error(f"Error emitting price update: {e}")
    
    def _emit_status_update(self):
        """Emit source status update via WebSocket"""
        try:
            status = self.pipeline.get_source_status()
            if status and self.socketio:
                self.socketio.emit('sources_status', {
                    'sources': status,
                    'timestamp': datetime.now().isoformat(),
                    'summary': {
                        'total': len(status),
                        'healthy': sum(1 for s in status.values() if s.get('reliability_score', 0) > 0.5),
                        'average_reliability': sum(s.get('reliability_score', 0) for s in status.values()) / len(status) if status else 0
                    }
                })
        except Exception as e:
            logger.error(f"Error emitting status update: {e}")
    
    async def _emit_health_check(self):
        """Emit health check results via WebSocket"""
        try:
            health = await self.pipeline.health_check()
            if health and self.socketio:
                self.socketio.emit('health_check', {
                    'health': health,
                    'timestamp': datetime.now().isoformat(),
                    'summary': {
                        'total': len(health),
                        'healthy': sum(1 for h in health.values() if h.get('status') == 'healthy')
                    }
                })
        except Exception as e:
            logger.error(f"Error emitting health check: {e}")

# Integration functions for existing web app
def replace_price_fetching_functions():
    """Replace existing price fetching functions with data pipeline integration"""
    
    async def fetch_live_gold_price_enhanced():
        """Enhanced version of fetch_live_gold_price using data pipeline"""
        try:
            integration = WebDataPipelineIntegration()
            data = await integration.get_enhanced_price_data('XAU')
            
            if data:
                return {
                    'price': data['price'],
                    'status': 'success',
                    'timestamp': data['timestamp'],
                    'source': data['source'],
                    'confidence': data['confidence'],
                    'quality_score': data['quality_score'],
                    'validations': {
                        'price_valid': data['price_valid'],
                        'timestamp_valid': data['timestamp_valid'],
                        'source_reliable': data['source_reliable'],
                        'within_range': data['within_range']
                    }
                }
            else:
                # Fallback to existing logic
                return {
                    'price': 3400.0,
                    'status': 'fallback',
                    'timestamp': datetime.now().isoformat(),
                    'source': 'fallback',
                    'confidence': 0.1
                }
                
        except Exception as e:
            logger.error(f"Enhanced price fetching failed: {e}")
            return {
                'price': 3400.0,
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'source': 'error_fallback',
                'confidence': 0.1
            }
    
    return fetch_live_gold_price_enhanced

def integrate_with_ml_engine():
    """Integration function for ML engine to use enhanced data pipeline"""
    
    async def get_ml_data_source(symbol: str, data_type: str):
        """Get data for ML engine from enhanced pipeline"""
        try:
            # Map data types
            type_mapping = {
                'price': DataType.PRICE,
                'technical': DataType.TECHNICAL,
                'sentiment': DataType.SENTIMENT,
                'macro': DataType.MACRO,
                'news': DataType.NEWS
            }
            
            dt = type_mapping.get(data_type, DataType.PRICE)
            data = await data_pipeline.get_unified_data(symbol, dt)
            
            return data
            
        except Exception as e:
            logger.error(f"ML data source error: {e}")
            return None
    
    return get_ml_data_source

def integrate_with_signal_generator():
    """Integration function for signal generator to use enhanced data pipeline"""
    
    async def get_signal_data_source(symbol: str, lookback_minutes: int = 60):
        """Get comprehensive data for signal generation"""
        try:
            # Get current price with high confidence
            price_data = await data_pipeline.get_unified_data(symbol, DataType.PRICE)
            
            # Get technical analysis data
            technical_data = await data_pipeline.get_unified_data(symbol, DataType.TECHNICAL)
            
            # Get sentiment data
            sentiment_data = await data_pipeline.get_unified_data(symbol, DataType.SENTIMENT)
            
            # Get macro data
            macro_data = await data_pipeline.get_unified_data(symbol, DataType.MACRO)
            
            # Combine all data sources
            combined_data = {
                'price': price_data,
                'technical': technical_data,
                'sentiment': sentiment_data,
                'macro': macro_data,
                'timestamp': datetime.now().isoformat(),
                'data_quality': {
                    'price_confidence': price_data.get('confidence', 0) if price_data else 0,
                    'technical_confidence': technical_data.get('confidence', 0) if technical_data else 0,
                    'sentiment_confidence': sentiment_data.get('confidence', 0) if sentiment_data else 0,
                    'macro_confidence': macro_data.get('confidence', 0) if macro_data else 0
                }
            }
            
            return combined_data
            
        except Exception as e:
            logger.error(f"Signal data source error: {e}")
            return None
    
    return get_signal_data_source

# Global integration instance
web_data_integration = None

def initialize_web_integration(app: Flask, socketio: SocketIO):
    """Initialize web integration with Flask app and SocketIO"""
    global web_data_integration
    web_data_integration = WebDataPipelineIntegration(app, socketio)
    logger.info("✅ Data pipeline web integration initialized")
    return web_data_integration

if __name__ == "__main__":
    # Test the integration
    async def test_integration():
        print("🧪 Testing Data Pipeline Web Integration...")
        
        integration = WebDataPipelineIntegration()
        
        # Test enhanced price data
        data = await integration.get_enhanced_price_data('XAU')
        print(f"📊 Enhanced price data: {data}")
        
        # Test data quality calculation
        if data:
            quality = integration.calculate_data_quality_score(data)
            print(f"🎯 Data quality score: {quality}")
    
    asyncio.run(test_integration())
