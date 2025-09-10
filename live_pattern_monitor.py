"""
Real-Time Pattern Monitoring Service
===================================
Continuous background monitoring of candlestick patterns with live updates
"""

import threading
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from real_pattern_detection import RealCandlestickDetector
import json

logger = logging.getLogger(__name__)

class LivePatternMonitor:
    """Continuous real-time pattern monitoring service"""
    
    def __init__(self, update_interval: int = 60):
        self.detector = RealCandlestickDetector()
        self.update_interval = update_interval  # seconds
        self.monitoring = False
        self.monitor_thread = None
        self.latest_patterns = []
        self.pattern_history = []
        self.last_update = None
        self.pattern_alerts = []
        
        # Pattern significance thresholds
        self.significance_thresholds = {
            'confidence': 80,  # Minimum confidence for significant patterns
            'freshness': 5,    # Minutes for "live" classification
            'volume_spike': 1.5  # Volume multiplier for significance
        }
        
        logger.info("🔄 Live Pattern Monitor initialized")
    
    def start_monitoring(self):
        """Start continuous pattern monitoring"""
        if self.monitoring:
            logger.warning("⚠️ Monitor already running")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("✅ Live pattern monitoring started")
    
    def stop_monitoring(self):
        """Stop pattern monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("🛑 Live pattern monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop - runs continuously"""
        logger.info("🔄 Starting continuous pattern monitoring loop...")
        
        while self.monitoring:
            try:
                scan_start = datetime.now()
                logger.info(f"🔍 LIVE SCAN #{len(self.pattern_history) + 1} - {scan_start.strftime('%H:%M:%S')}")
                
                # Perform pattern detection
                patterns = self.detector.detect_all_patterns()
                
                if patterns:
                    # Filter for significant patterns only
                    significant_patterns = self._filter_significant_patterns(patterns)
                    
                    # Check for new high-impact patterns
                    new_alerts = self._check_for_alerts(significant_patterns)
                    
                    # Update state
                    self.latest_patterns = significant_patterns
                    self.last_update = scan_start
                    self.pattern_history.append({
                        'timestamp': scan_start,
                        'pattern_count': len(patterns),
                        'significant_count': len(significant_patterns),
                        'new_alerts': len(new_alerts)
                    })
                    
                    # Log results
                    scan_duration = (datetime.now() - scan_start).total_seconds()
                    logger.info(f"✅ SCAN COMPLETE: {len(patterns)} total, {len(significant_patterns)} significant ({scan_duration:.1f}s)")
                    
                    if significant_patterns:
                        for i, pattern in enumerate(significant_patterns[:3]):
                            urgency = "🔴" if pattern.get('urgency') == 'HIGH' else ("🟡" if pattern.get('urgency') == 'MEDIUM' else "🟢")
                            logger.info(f"   {urgency} {pattern['name']}: {pattern['confidence']:.1f}% | {pattern['signal']} | {pattern.get('time_ago', 'now')}")
                    
                    if new_alerts:
                        logger.warning(f"🚨 {len(new_alerts)} NEW HIGH-IMPACT PATTERNS DETECTED!")
                        for alert in new_alerts:
                            logger.warning(f"   🚨 {alert['pattern']} - {alert['confidence']}% | {alert['signal']} | {alert['market_effect']}")
                else:
                    logger.info("📊 No patterns detected in current scan")
                
                # Cleanup old history (keep last 100 scans)
                if len(self.pattern_history) > 100:
                    self.pattern_history = self.pattern_history[-100:]
                
                # Wait for next scan
                time.sleep(self.update_interval)
                
            except Exception as e:
                logger.error(f"❌ Monitoring loop error: {e}")
                time.sleep(30)  # Wait 30s before retry on error
    
    def _filter_significant_patterns(self, patterns: List[Dict]) -> List[Dict]:
        """Filter patterns for significance based on confidence, freshness, etc."""
        significant = []
        current_time = datetime.now()
        
        for pattern in patterns:
            # Check confidence threshold
            confidence = pattern.get('confidence', 0)
            if confidence < self.significance_thresholds['confidence']:
                continue
            
            # Check freshness (how recent the pattern is)
            time_diff = current_time - pattern.get('timestamp', current_time)
            minutes_ago = time_diff.total_seconds() / 60
            
            # Mark as significant if high confidence OR very fresh
            is_significant = (
                confidence >= self.significance_thresholds['confidence'] or
                minutes_ago <= self.significance_thresholds['freshness']
            )
            
            if is_significant:
                # Add significance scoring
                pattern['significance_score'] = self._calculate_significance_score(pattern, minutes_ago)
                significant.append(pattern)
        
        # Sort by significance score
        significant.sort(key=lambda x: x.get('significance_score', 0), reverse=True)
        return significant
    
    def _calculate_significance_score(self, pattern: Dict, minutes_ago: float) -> float:
        """Calculate overall significance score for a pattern"""
        confidence = pattern.get('confidence', 0)
        
        # Base score from confidence
        score = confidence
        
        # Freshness bonus (more recent = higher score)
        freshness_bonus = max(0, 20 - (minutes_ago * 2))  # Up to 20 points for very fresh patterns
        score += freshness_bonus
        
        # Pattern type bonuses
        pattern_bonuses = {
            'Dragonfly Doji': 10,
            'Gravestone Doji': 10,
            'Bullish Engulfing': 15,
            'Bearish Engulfing': 15,
            'Hammer': 8,
            'Shooting Star': 8
        }
        
        pattern_name = pattern.get('name', '')
        score += pattern_bonuses.get(pattern_name, 0)
        
        # Signal strength bonus
        strength = pattern.get('strength', 'MEDIUM')
        strength_bonuses = {'VERY_STRONG': 15, 'STRONG': 10, 'MEDIUM': 5, 'WEAK': 0}
        score += strength_bonuses.get(strength, 0)
        
        return min(100, score)  # Cap at 100
    
    def _check_for_alerts(self, patterns: List[Dict]) -> List[Dict]:
        """Check for patterns that should trigger alerts"""
        new_alerts = []
        current_time = datetime.now()
        
        for pattern in patterns:
            # High-impact criteria
            is_high_impact = (
                pattern.get('confidence', 0) >= 90 or
                pattern.get('significance_score', 0) >= 85 or
                pattern.get('strength') == 'VERY_STRONG'
            )
            
            if is_high_impact:
                alert = {
                    'alert_id': f"ALERT_{current_time.strftime('%H%M%S')}_{pattern.get('name', 'UNKNOWN').replace(' ', '_')}",
                    'timestamp': current_time,
                    'pattern': pattern.get('name', 'Unknown'),
                    'confidence': pattern.get('confidence', 0),
                    'signal': pattern.get('signal', 'NEUTRAL'),
                    'market_effect': pattern.get('market_effect', 'UNKNOWN'),
                    'significance_score': pattern.get('significance_score', 0),
                    'urgency': 'CRITICAL' if pattern.get('confidence', 0) >= 95 else 'HIGH',
                    'candle_timestamp': pattern.get('timestamp'),
                    'price_at_detection': pattern.get('candle_data', {}).get('close', 0)
                }
                
                new_alerts.append(alert)
                self.pattern_alerts.append(alert)
        
        # Clean old alerts (keep last 50)
        if len(self.pattern_alerts) > 50:
            self.pattern_alerts = self.pattern_alerts[-50:]
        
        return new_alerts
    
    def get_live_status(self) -> Dict:
        """Get current monitoring status and latest patterns"""
        current_time = datetime.now()
        
        status = {
            'monitoring_active': self.monitoring,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'update_interval': self.update_interval,
            'total_scans': len(self.pattern_history),
            'current_patterns': len(self.latest_patterns),
            'recent_alerts': len([a for a in self.pattern_alerts if (current_time - a['timestamp']).total_seconds() < 3600]),
            'monitoring_uptime': (current_time - self.pattern_history[0]['timestamp']).total_seconds() if self.pattern_history else 0,
            'status': 'ACTIVE' if self.monitoring else 'STOPPED'
        }
        
        return status
    
    def get_latest_patterns(self, limit: int = 10) -> List[Dict]:
        """Get most recent significant patterns"""
        return self.latest_patterns[:limit]
    
    def get_recent_alerts(self, minutes: int = 60) -> List[Dict]:
        """Get alerts from the last N minutes"""
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [alert for alert in self.pattern_alerts if alert['timestamp'] >= cutoff_time]

# Global monitor instance
live_pattern_monitor = LivePatternMonitor(update_interval=60)  # Update every minute

def start_live_monitoring():
    """Start the global live pattern monitor"""
    live_pattern_monitor.start_monitoring()

def stop_live_monitoring():
    """Stop the global live pattern monitor"""
    live_pattern_monitor.stop_monitoring()

def get_live_monitor_status():
    """Get current live monitoring status"""
    return live_pattern_monitor.get_live_status()

def get_monitored_patterns(limit: int = 10):
    """Get latest patterns from live monitor"""
    return live_pattern_monitor.get_latest_patterns(limit)

def get_pattern_alerts(minutes: int = 60):
    """Get recent pattern alerts"""
    return live_pattern_monitor.get_recent_alerts(minutes)
