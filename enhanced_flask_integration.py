"""
Enhanced Flask Integration for Robust Data System
Replaces existing data fetching with the new multi-source system
"""

import asyncio
from flask import Flask, jsonify, request
from datetime import datetime
import logging
import json
from typing import Dict, Any, Optional
import sys
import os

# Import the robust data system
try:
    from robust_data_system import unified_data_provider, DataSource, cleanup_data_cache
    ROBUST_DATA_AVAILABLE = True
    print("✅ Robust Data System loaded successfully")
except ImportError as e:
    print(f"⚠️ Robust Data System not available: {e}")
    ROBUST_DATA_AVAILABLE = False

logger = logging.getLogger(__name__)

class FlaskDataIntegration:
    """Integration layer between Flask and the robust data system"""
    
    def __init__(self, app: Flask):
        self.app = app
        self.data_provider = unified_data_provider if ROBUST_DATA_AVAILABLE else None
        self.setup_routes()
        
        # Start background cleanup task
        if ROBUST_DATA_AVAILABLE:
            self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background tasks for cache cleanup"""
        try:
            import threading
            
            def run_cleanup():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(cleanup_data_cache())
            
            cleanup_thread = threading.Thread(target=run_cleanup, daemon=True)
            cleanup_thread.start()
            logger.info("Background cache cleanup started")
            
        except Exception as e:
            logger.warning(f"Failed to start background tasks: {e}")
    
    def setup_routes(self):
        """Setup enhanced API routes"""
        
        @self.app.route('/api/enhanced/price/<symbol>')
        async def get_enhanced_price(symbol):
            """Enhanced price endpoint with multiple sources"""
            try:
                if not ROBUST_DATA_AVAILABLE:
                    return jsonify({
                        'success': False,
                        'error': 'Robust data system not available',
                        'fallback': True
                    }), 503
                
                # Get price data asynchronously
                price_data = await self.data_provider.get_price_data(symbol.upper())
                
                return jsonify({
                    'success': True,
                    'data': {
                        'symbol': price_data.symbol,
                        'price': price_data.price,
                        'bid': price_data.bid,
                        'ask': price_data.ask,
                        'spread': price_data.spread,
                        'change': price_data.change,
                        'change_percent': price_data.change_percent,
                        'volume': price_data.volume,
                        'high_24h': price_data.high_24h,
                        'low_24h': price_data.low_24h,
                        'source': price_data.source.value,
                        'timestamp': price_data.timestamp.isoformat()
                    },
                    'metadata': {
                        'cache_hit': False,  # Would be determined by checking cache
                        'response_time_ms': 0  # Would be measured
                    }
                })
                
            except Exception as e:
                logger.error(f"Enhanced price fetch failed for {symbol}: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'symbol': symbol
                }), 500
        
        @self.app.route('/api/enhanced/sentiment/<symbol>')
        async def get_enhanced_sentiment(symbol):
            """Enhanced sentiment analysis endpoint"""
            try:
                if not ROBUST_DATA_AVAILABLE:
                    return jsonify({
                        'success': False,
                        'error': 'Robust data system not available'
                    }), 503
                
                timeframe = request.args.get('timeframe', '1d')
                
                sentiment_data = await self.data_provider.get_sentiment_data(
                    symbol.upper(), timeframe
                )
                
                return jsonify({
                    'success': True,
                    'data': {
                        'symbol': sentiment_data.symbol,
                        'sentiment_score': sentiment_data.sentiment_score,
                        'sentiment_label': sentiment_data.sentiment_label,
                        'confidence': sentiment_data.confidence,
                        'sources_count': sentiment_data.sources_count,
                        'timeframe': sentiment_data.timeframe,
                        'timestamp': sentiment_data.timestamp.isoformat(),
                        'articles_sample': sentiment_data.news_articles[:3]  # First 3 articles
                    },
                    'metadata': {
                        'total_articles': len(sentiment_data.news_articles),
                        'analysis_method': 'nlp_and_keywords'
                    }
                })
                
            except Exception as e:
                logger.error(f"Enhanced sentiment analysis failed for {symbol}: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'symbol': symbol
                }), 500
        
        @self.app.route('/api/enhanced/technical/<symbol>')
        async def get_enhanced_technical(symbol):
            """Enhanced technical analysis endpoint"""
            try:
                if not ROBUST_DATA_AVAILABLE:
                    return jsonify({
                        'success': False,
                        'error': 'Robust data system not available'
                    }), 503
                
                timeframe = request.args.get('timeframe', '1H')
                
                technical_data = await self.data_provider.get_technical_data(
                    symbol.upper(), timeframe
                )
                
                return jsonify({
                    'success': True,
                    'data': {
                        'symbol': technical_data.symbol,
                        'timeframe': technical_data.analysis_timeframe,
                        'indicators': technical_data.indicators,
                        'source': technical_data.source.value,
                        'timestamp': technical_data.timestamp.isoformat()
                    },
                    'metadata': {
                        'calculation_method': 'real_time',
                        'data_points_used': 100  # Could be dynamic
                    }
                })
                
            except Exception as e:
                logger.error(f"Enhanced technical analysis failed for {symbol}: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'symbol': symbol
                }), 500
        
        @self.app.route('/api/enhanced/comprehensive/<symbol>')
        async def get_comprehensive_data(symbol):
            """Get all data types for a symbol"""
            try:
                if not ROBUST_DATA_AVAILABLE:
                    return jsonify({
                        'success': False,
                        'error': 'Robust data system not available'
                    }), 503
                
                comprehensive_data = await self.data_provider.get_comprehensive_data(
                    symbol.upper()
                )
                
                return jsonify({
                    'success': True,
                    'data': comprehensive_data,
                    'metadata': {
                        'fetch_time': datetime.now().isoformat(),
                        'data_sources': {
                            'price': comprehensive_data['price']['source'],
                            'sentiment': 'news_analysis',
                            'technical': comprehensive_data['technical']['source']
                        }
                    }
                })
                
            except Exception as e:
                logger.error(f"Comprehensive data fetch failed for {symbol}: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'symbol': symbol
                }), 500
        
        @self.app.route('/api/enhanced/watchlist')
        async def get_enhanced_watchlist():
            """Get enhanced data for multiple symbols"""
            try:
                if not ROBUST_DATA_AVAILABLE:
                    return jsonify({
                        'success': False,
                        'error': 'Robust data system not available'
                    }), 503
                
                symbols = request.args.get('symbols', 'XAUUSD,EURUSD,GBPUSD,USDJPY').split(',')
                symbols = [s.strip().upper() for s in symbols]
                
                results = []
                
                # Fetch data for all symbols concurrently
                tasks = [self.data_provider.get_price_data(symbol) for symbol in symbols]
                price_data_list = await asyncio.gather(*tasks, return_exceptions=True)
                
                for i, symbol in enumerate(symbols):
                    price_data = price_data_list[i]
                    
                    if isinstance(price_data, Exception):
                        results.append({
                            'symbol': symbol,
                            'error': str(price_data),
                            'success': False
                        })
                    else:
                        results.append({
                            'symbol': symbol,
                            'price': price_data.price,
                            'change': price_data.change,
                            'change_percent': price_data.change_percent,
                            'bid': price_data.bid,
                            'ask': price_data.ask,
                            'source': price_data.source.value,
                            'timestamp': price_data.timestamp.isoformat(),
                            'success': True
                        })
                
                return jsonify({
                    'success': True,
                    'data': results,
                    'metadata': {
                        'symbols_requested': len(symbols),
                        'symbols_successful': sum(1 for r in results if r.get('success')),
                        'fetch_time': datetime.now().isoformat()
                    }
                })
                
            except Exception as e:
                logger.error(f"Enhanced watchlist fetch failed: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/enhanced/status')
        def get_enhanced_status():
            """Get status of the enhanced data system"""
            try:
                if not ROBUST_DATA_AVAILABLE:
                    return jsonify({
                        'success': False,
                        'robust_data_available': False,
                        'error': 'Robust data system not loaded'
                    })
                
                # Get provider statistics
                provider_stats = self.data_provider.get_provider_stats()
                
                return jsonify({
                    'success': True,
                    'robust_data_available': True,
                    'provider_statistics': provider_stats,
                    'cache_manager': {
                        'available': True,
                        'database': 'SQLite'
                    },
                    'rate_limiter': {
                        'active': True,
                        'respectful_scraping': True
                    },
                    'capabilities': {
                        'price_data': ['api_primary', 'web_scraping', 'simulated'],
                        'sentiment_analysis': ['news_scraping', 'nlp_analysis', 'simulated'],
                        'technical_indicators': ['real_calculation', 'simulated'],
                        'automatic_fallback': True,
                        'caching': True,
                        'rate_limiting': True
                    },
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Status check failed: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/enhanced/cache/cleanup', methods=['POST'])
        def cleanup_cache():
            """Manual cache cleanup endpoint"""
            try:
                if not ROBUST_DATA_AVAILABLE:
                    return jsonify({
                        'success': False,
                        'error': 'Robust data system not available'
                    }), 503
                
                self.data_provider.cleanup_cache()
                
                return jsonify({
                    'success': True,
                    'message': 'Cache cleanup completed',
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Cache cleanup failed: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500

# Synchronous wrapper functions for use with existing Flask routes
def get_price_data_sync(symbol: str) -> Dict[str, Any]:
    """Synchronous wrapper for price data"""
    if not ROBUST_DATA_AVAILABLE:
        return {
            'success': False,
            'error': 'Robust data system not available'
        }
    
    try:
        # Create a new event loop in a thread-safe way
        import threading
        
        result = {}
        error = None
        
        def run_async():
            nonlocal result, error
            try:
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    price_data = new_loop.run_until_complete(
                        unified_data_provider.get_price_data(symbol.upper())
                    )
                    
                    result = {
                        'success': True,
                        'price': price_data.price,
                        'change': price_data.change,
                        'change_percent': price_data.change_percent,
                        'bid': price_data.bid,
                        'ask': price_data.ask,
                        'volume': price_data.volume,
                        'source': price_data.source.value,
                        'timestamp': price_data.timestamp.isoformat()
                    }
                finally:
                    new_loop.close()
            except Exception as e:
                error = str(e)
        
        thread = threading.Thread(target=run_async)
        thread.start()
        thread.join(timeout=10)  # 10 second timeout
        
        if error:
            raise Exception(error)
        
        if not result:
            raise Exception("Request timed out")
            
        return result
            
    except Exception as e:
        logger.error(f"Sync price data fetch failed for {symbol}: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def get_sentiment_data_sync(symbol: str, timeframe: str = '1d') -> Dict[str, Any]:
    """Synchronous wrapper for sentiment data"""
    if not ROBUST_DATA_AVAILABLE:
        return {
            'success': False,
            'error': 'Robust data system not available'
        }
    
    try:
        import threading
        
        result = {}
        error = None
        
        def run_async():
            nonlocal result, error
            try:
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    sentiment_data = new_loop.run_until_complete(
                        unified_data_provider.get_sentiment_data(symbol.upper(), timeframe)
                    )
                    
                    result = {
                        'success': True,
                        'sentiment_score': sentiment_data.sentiment_score,
                        'sentiment_label': sentiment_data.sentiment_label,
                        'confidence': sentiment_data.confidence,
                        'sources_count': sentiment_data.sources_count,
                        'timeframe': sentiment_data.timeframe,
                        'timestamp': sentiment_data.timestamp.isoformat()
                    }
                finally:
                    new_loop.close()
            except Exception as e:
                error = str(e)
        
        thread = threading.Thread(target=run_async)
        thread.start()
        thread.join(timeout=10)
        
        if error:
            raise Exception(error)
            
        if not result:
            raise Exception("Request timed out")
            
        return result
            
    except Exception as e:
        logger.error(f"Sync sentiment analysis failed for {symbol}: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def get_technical_data_sync(symbol: str, timeframe: str = '1H') -> Dict[str, Any]:
    """Synchronous wrapper for technical data"""
    if not ROBUST_DATA_AVAILABLE:
        return {
            'success': False,
            'error': 'Robust data system not available'
        }
    
    try:
        import threading
        
        result = {}
        error = None
        
        def run_async():
            nonlocal result, error
            try:
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    technical_data = new_loop.run_until_complete(
                        unified_data_provider.get_technical_data(symbol.upper(), timeframe)
                    )
                    
                    result = {
                        'success': True,
                        'indicators': technical_data.indicators,
                        'timeframe': technical_data.analysis_timeframe,
                        'source': technical_data.source.value,
                        'timestamp': technical_data.timestamp.isoformat()
                    }
                finally:
                    new_loop.close()
            except Exception as e:
                error = str(e)
        
        thread = threading.Thread(target=run_async)
        thread.start()
        thread.join(timeout=10)
        
        if error:
            raise Exception(error)
            
        if not result:
            raise Exception("Request timed out")
            
        return result
            
    except Exception as e:
        logger.error(f"Sync technical analysis failed for {symbol}: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def get_comprehensive_data_sync(symbol: str) -> Dict[str, Any]:
    """Synchronous wrapper for comprehensive data"""
    if not ROBUST_DATA_AVAILABLE:
        return {
            'success': False,
            'error': 'Robust data system not available'
        }
    
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            comprehensive_data = loop.run_until_complete(
                unified_data_provider.get_comprehensive_data(symbol.upper())
            )
            
            return {
                'success': True,
                'data': comprehensive_data
            }
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Sync comprehensive data fetch failed for {symbol}: {e}")
        return {
            'success': False,
            'error': str(e)
        }

# Enhancement for existing routes
class RoutesEnhancer:
    """Enhances existing Flask routes with robust data system"""
    
    @staticmethod
    def enhance_price_route(app: Flask):
        """Enhance existing price routes"""
        
        @app.route('/api/price/<symbol>')
        def enhanced_price_route(symbol):
            """Enhanced version of existing price route"""
            try:
                # Try robust data system first
                result = get_price_data_sync(symbol)
                
                if result['success']:
                    return jsonify({
                        'success': True,
                        'symbol': symbol.upper(),
                        'price': result['price'],
                        'change': result['change'],
                        'change_percent': result['change_percent'],
                        'bid': result['bid'],
                        'ask': result['ask'],
                        'volume': result['volume'],
                        'source': result['source'],
                        'timestamp': result['timestamp'],
                        'enhanced': True
                    })
                else:
                    # Fallback to original implementation would go here
                    return jsonify({
                        'success': False,
                        'error': result['error'],
                        'fallback_needed': True
                    }), 503
                    
            except Exception as e:
                logger.error(f"Enhanced price route failed for {symbol}: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
    
    @staticmethod
    def enhance_sentiment_route(app: Flask):
        """Enhance existing sentiment routes"""
        
        @app.route('/api/sentiment/<symbol>')
        def enhanced_sentiment_route(symbol):
            """Enhanced version of existing sentiment route"""
            try:
                timeframe = request.args.get('timeframe', '1d')
                result = get_sentiment_data_sync(symbol, timeframe)
                
                if result['success']:
                    return jsonify({
                        'success': True,
                        'symbol': symbol.upper(),
                        'sentiment': {
                            'score': result['sentiment_score'],
                            'label': result['sentiment_label'],
                            'confidence': result['confidence'],
                            'sources_count': result['sources_count'],
                            'timeframe': result['timeframe']
                        },
                        'timestamp': result['timestamp'],
                        'enhanced': True
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': result['error'],
                        'fallback_needed': True
                    }), 503
                    
            except Exception as e:
                logger.error(f"Enhanced sentiment route failed for {symbol}: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
    
    @staticmethod
    def enhance_technical_route(app: Flask):
        """Enhance existing technical analysis routes"""
        
        @app.route('/api/technical/<symbol>')
        def enhanced_technical_route(symbol):
            """Enhanced version of existing technical route"""
            try:
                timeframe = request.args.get('timeframe', '1H')
                result = get_technical_data_sync(symbol, timeframe)
                
                if result['success']:
                    return jsonify({
                        'success': True,
                        'symbol': symbol.upper(),
                        'technical': result['indicators'],
                        'timeframe': result['timeframe'],
                        'source': result['source'],
                        'timestamp': result['timestamp'],
                        'enhanced': True
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': result['error'],
                        'fallback_needed': True
                    }), 503
                    
            except Exception as e:
                logger.error(f"Enhanced technical route failed for {symbol}: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500

def setup_enhanced_routes(app: Flask):
    """Setup all enhanced routes"""
    try:
        # Initialize the integration
        integration = FlaskDataIntegration(app)
        
        # Enhance existing routes
        enhancer = RoutesEnhancer()
        # Note: These would replace existing routes in your app.py
        # enhancer.enhance_price_route(app)
        # enhancer.enhance_sentiment_route(app)
        # enhancer.enhance_technical_route(app)
        
        logger.info("Enhanced data routes setup completed")
        return True
        
    except Exception as e:
        logger.error(f"Failed to setup enhanced routes: {e}")
        return False

# Export for easy integration
__all__ = [
    'FlaskDataIntegration',
    'RoutesEnhancer', 
    'setup_enhanced_routes',
    'get_price_data_sync',
    'get_sentiment_data_sync',
    'get_technical_data_sync',
    'get_comprehensive_data_sync',
    'ROBUST_DATA_AVAILABLE'
]
