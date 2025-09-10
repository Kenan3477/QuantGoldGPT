"""
REAL-TIME Live Pattern Detection API
===================================
Enhanced real-time candlestick pattern scanning with exact timestamps
"""

from flask import jsonify
from datetime import datetime, timedelta
import logging
from real_pattern_detection import get_real_candlestick_patterns, format_patterns_for_api

logger = logging.getLogger(__name__)

def get_live_patterns_enhanced():
    """
    REAL-TIME live candlestick pattern detection with exact timestamps
    
    Features:
    - Multi-source data scanning (Yahoo Finance, Gold API, Alpha Vantage)
    - Exact timestamp tracking for each pattern
    - Market effect analysis and strength scoring
    - Real-time freshness scoring
    - Live vs historical pattern classification
    """
    try:
        print("🔴 LIVE PATTERN SCAN STARTED!")
        logger.info("🔴 LIVE PATTERN ENDPOINT CALLED - Starting real-time scan...")
        
        # Get current gold price first for context
        current_price = 3540.0  # Default fallback
        try:
            # This would connect to the gold price API in the main app
            import requests
            response = requests.get('https://api.gold-api.com/price/XAU', timeout=3)
            if response.status_code == 200:
                current_price = float(response.json().get('price', current_price))
                logger.info(f"💰 Current gold price: ${current_price}")
        except Exception as e:
            logger.warning(f"⚠️ Price fetch warning: {e}")
        
        # Get REAL-TIME patterns from live market data
        try:
            logger.info("🔍 Initiating live market scan...")
            patterns = get_real_candlestick_patterns()
            formatted_patterns = format_patterns_for_api()
            
            logger.info(f"✅ LIVE SCAN COMPLETE: {len(patterns)} raw patterns, {len(formatted_patterns)} formatted")
            
            # Calculate live metrics
            live_pattern_count = len([p for p in patterns if p.get('is_live', False)])
            high_confidence_count = len([p for p in formatted_patterns if float(p['confidence'].replace('%', '')) > 80])
            
            # Enhanced response with comprehensive live data metrics
            response_data = {
                'success': True,
                'scan_timestamp': datetime.now().isoformat(),
                'current_patterns': formatted_patterns,
                'pattern_count': len(formatted_patterns),
                'live_pattern_count': live_pattern_count,
                'high_confidence_count': high_confidence_count,
                'current_price': current_price,
                'data_source': 'LIVE_MARKET_SCAN',
                'scan_status': 'COMPLETED',
                'scan_quality': 'HIGH' if len(formatted_patterns) > 0 else 'MEDIUM',
                'most_recent_pattern': formatted_patterns[0] if formatted_patterns else None,
                'market_activity': 'HIGH' if live_pattern_count > 2 else ('MEDIUM' if live_pattern_count > 0 else 'LOW'),
                'last_updated': datetime.now().isoformat(),
                'next_scan_in': '60s',
                'data_freshness': 'LIVE'
            }
            
            # Log key pattern details for monitoring
            if formatted_patterns:
                logger.info(f"📊 PATTERN BREAKDOWN:")
                logger.info(f"   📈 Total: {len(formatted_patterns)} | Live: {live_pattern_count} | High Conf: {high_confidence_count}")
                
                for i, pattern in enumerate(formatted_patterns[:3]):
                    logger.info(f"🎯 #{i+1}: {pattern['pattern']} | {pattern['confidence']} | {pattern['time_ago']} | {pattern['urgency']} | {pattern['signal']}")
            else:
                logger.info("📊 No active patterns detected in current market scan")
            
            return jsonify(response_data)
            
        except Exception as e:
            logger.error(f"❌ Live pattern scan failed: {e}")
            import traceback
            logger.error(f"🔍 Full error trace: {traceback.format_exc()}")
            
            return jsonify({
                'success': False,
                'error': f"Live scan failed: {str(e)}",
                'current_patterns': [],
                'current_price': current_price,
                'data_source': 'SCAN_ERROR',
                'scan_status': 'FAILED',
                'last_updated': datetime.now().isoformat(),
                'retry_in': '30s'
            })
        
    except Exception as e:
        logger.error(f"❌ Critical pattern endpoint error: {e}")
        return jsonify({
            'success': False, 
            'error': str(e),
            'current_patterns': [],
            'data_source': 'CRITICAL_ERROR',
            'scan_status': 'CRITICAL_FAILURE',
            'last_updated': datetime.now().isoformat()
        })

def get_pattern_analysis_summary():
    """Get summary of pattern detection performance and statistics"""
    try:
        patterns = get_real_candlestick_patterns()
        
        if not patterns:
            return {
                'total_patterns': 0,
                'pattern_types': {},
                'confidence_distribution': {},
                'signal_distribution': {},
                'time_distribution': {},
                'status': 'NO_PATTERNS'
            }
        
        # Analyze pattern distribution
        pattern_types = {}
        confidence_levels = {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0}
        signal_types = {'BULLISH': 0, 'BEARISH': 0, 'NEUTRAL': 0}
        time_ranges = {'LIVE': 0, 'RECENT': 0, 'OLDER': 0}
        
        current_time = datetime.now()
        
        for pattern in patterns:
            # Pattern type count
            pattern_name = pattern.get('name', 'Unknown')
            pattern_types[pattern_name] = pattern_types.get(pattern_name, 0) + 1
            
            # Confidence level
            confidence = pattern.get('confidence', 0)
            if confidence >= 85:
                confidence_levels['HIGH'] += 1
            elif confidence >= 65:
                confidence_levels['MEDIUM'] += 1
            else:
                confidence_levels['LOW'] += 1
            
            # Signal type
            signal = pattern.get('signal', 'NEUTRAL')
            signal_types[signal] = signal_types.get(signal, 0) + 1
            
            # Time classification
            time_diff = current_time - pattern.get('timestamp', current_time)
            minutes_ago = time_diff.total_seconds() / 60
            
            if minutes_ago <= 5:
                time_ranges['LIVE'] += 1
            elif minutes_ago <= 15:
                time_ranges['RECENT'] += 1
            else:
                time_ranges['OLDER'] += 1
        
        return {
            'total_patterns': len(patterns),
            'pattern_types': pattern_types,
            'confidence_distribution': confidence_levels,
            'signal_distribution': signal_types,
            'time_distribution': time_ranges,
            'analysis_timestamp': current_time.isoformat(),
            'status': 'ANALYZED'
        }
        
    except Exception as e:
        logger.error(f"❌ Pattern analysis failed: {e}")
        return {
            'total_patterns': 0,
            'error': str(e),
            'status': 'ERROR'
        }
