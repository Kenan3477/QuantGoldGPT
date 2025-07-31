#!/usr/bin/env python3
"""
Enhanced Signal Monitoring Service
Automatically monitors active signals for TP/SL hits and learns from outcomes
"""
import threading
import time
import logging
from datetime import datetime, timedelta
from enhanced_signal_generator import enhanced_signal_generator
from price_storage_manager import get_current_gold_price

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignalMonitoringService:
    """Background service to monitor signals and update their status"""
    
    def __init__(self):
        self.is_running = False
        self.monitor_thread = None
        self.check_interval = 30  # Check every 30 seconds
        self.last_price_check = None
        
    def start_monitoring(self):
        """Start the background monitoring service"""
        if self.is_running:
            logger.warning("Signal monitoring service already running")
            return
            
        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("🚀 Enhanced Signal Monitoring Service started")
        
    def stop_monitoring(self):
        """Stop the background monitoring service"""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
            
        logger.info("⏹️ Enhanced Signal Monitoring Service stopped")
        
    def _monitoring_loop(self):
        """Main monitoring loop"""
        logger.info("🔄 Signal monitoring loop started")
        
        while self.is_running:
            try:
                # Monitor signals for TP/SL hits
                monitoring_result = enhanced_signal_generator.monitor_active_signals()
                
                if monitoring_result and not monitoring_result.get('error'):
                    self._process_monitoring_result(monitoring_result)
                    
                # Update last check time
                self.last_price_check = datetime.now()
                
            except Exception as e:
                logger.error(f"Error in signal monitoring loop: {e}")
                
            # Wait before next check
            time.sleep(self.check_interval)
            
    def _process_monitoring_result(self, result):
        """Process monitoring results and log important events"""
        try:
            current_price = result.get('current_price')
            active_count = result.get('active_signals', 0)
            updates = result.get('updates', [])
            closed_signals = result.get('closed_signals', [])
            
            # Log current status
            if current_price:
                logger.debug(f"💰 Current gold price: ${current_price:.2f} | Active signals: {active_count}")
                
            # Log signal closures
            for closed_signal in closed_signals:
                self._log_signal_closure(closed_signal)
                
            # Log significant price updates if any
            if updates:
                logger.debug(f"📊 Updated {len(updates)} active signals with current prices")
                
        except Exception as e:
            logger.error(f"Error processing monitoring result: {e}")
            
    def _log_signal_closure(self, closed_signal):
        """Log details when a signal closes"""
        try:
            signal_id = closed_signal.get('id')
            signal_type = closed_signal.get('type', 'Unknown').upper()
            exit_reason = closed_signal.get('exit_reason', 'unknown')
            profit_loss = closed_signal.get('profit_loss', 0)
            profit_loss_pct = closed_signal.get('profit_loss_pct', 0)
            entry_price = closed_signal.get('entry_price', 0)
            exit_price = closed_signal.get('exit_price', 0)
            duration_minutes = closed_signal.get('duration_minutes', 0)
            
            # Determine if profitable
            is_profitable = profit_loss > 0
            result_emoji = "🎉" if is_profitable else "📉"
            result_text = "PROFIT" if is_profitable else "LOSS"
            
            # Create detailed log message
            duration_text = f"{duration_minutes:.1f} minutes" if duration_minutes < 60 else f"{duration_minutes/60:.1f} hours"
            
            log_message = (
                f"{result_emoji} Signal #{signal_id} CLOSED - {result_text}\n"
                f"   Type: {signal_type}\n"
                f"   Reason: {exit_reason.upper()}\n"
                f"   Entry: ${entry_price:.2f} → Exit: ${exit_price:.2f}\n"
                f"   P&L: ${profit_loss:.2f} ({profit_loss_pct:+.2f}%)\n"
                f"   Duration: {duration_text}"
            )
            
            if is_profitable:
                logger.info(log_message)
            else:
                logger.warning(log_message)
                
        except Exception as e:
            logger.error(f"Error logging signal closure: {e}")
            
    def get_service_status(self):
        """Get current status of the monitoring service"""
        return {
            'is_running': self.is_running,
            'last_check': self.last_price_check.isoformat() if self.last_price_check else None,
            'check_interval_seconds': self.check_interval,
            'thread_alive': self.monitor_thread.is_alive() if self.monitor_thread else False
        }
        
    def force_check_now(self):
        """Force an immediate check of all signals"""
        try:
            logger.info("🔍 Forcing immediate signal check...")
            monitoring_result = enhanced_signal_generator.monitor_active_signals()
            
            if monitoring_result and not monitoring_result.get('error'):
                self._process_monitoring_result(monitoring_result)
                logger.info("✅ Forced signal check completed")
                return monitoring_result
            else:
                logger.error(f"Failed forced signal check: {monitoring_result.get('error', 'Unknown error')}")
                return None
                
        except Exception as e:
            logger.error(f"Error in forced signal check: {e}")
            return None

# Global service instance
signal_monitoring_service = SignalMonitoringService()

def start_signal_monitoring():
    """Start the global signal monitoring service"""
    signal_monitoring_service.start_monitoring()

def stop_signal_monitoring():
    """Stop the global signal monitoring service"""
    signal_monitoring_service.stop_monitoring()

def get_monitoring_status():
    """Get the current monitoring service status"""
    return signal_monitoring_service.get_service_status()

def force_signal_check():
    """Force an immediate signal check"""
    return signal_monitoring_service.force_check_now()

if __name__ == "__main__":
    # Test the monitoring service
    print("🧪 Testing Enhanced Signal Monitoring Service...")
    
    try:
        # Start monitoring
        start_signal_monitoring()
        
        # Run for a short time to test
        print("🔄 Monitoring for 60 seconds...")
        time.sleep(60)
        
        # Force a check
        print("🔍 Forcing immediate check...")
        result = force_signal_check()
        print(f"Check result: {result}")
        
        # Get status
        status = get_monitoring_status()
        print(f"Service status: {status}")
        
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    finally:
        # Stop monitoring
        stop_signal_monitoring()
        print("✅ Test completed")
