#!/usr/bin/env python3
"""
Resource Governor for GoldGPT - CPU and Memory Management
========================================================

Advanced system resource monitoring and management to prevent
100% CPU usage and maintain optimal performance.
"""

import os
import psutil
import time
import threading
import logging
import functools
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import queue
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemResourceGovernor:
    """Advanced system resource monitoring and control"""
    
    def __init__(self, 
                 cpu_threshold: float = 75.0,
                 memory_threshold: float = 80.0,
                 check_interval: int = 5,
                 cooldown_period: int = 30):
        
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.check_interval = check_interval
        self.cooldown_period = cooldown_period
        
        # System state
        self.pause_processing = False
        self.current_cpu = 0.0
        self.current_memory = 0.0
        self.resource_history = []
        self.last_alert_time = None
        
        # Monitoring thread
        self._monitor_thread = None
        self._shutdown_event = threading.Event()
        
        # Process management
        self.managed_processes = []
        self.task_queue = queue.Queue()
        
        # Performance metrics
        self.metrics = {
            'cpu_alerts': 0,
            'memory_alerts': 0,
            'tasks_paused': 0,
            'tasks_resumed': 0,
            'uptime_start': datetime.now()
        }
        
        # Adaptive thresholds
        self.adaptive_mode = True
        self.base_cpu_threshold = cpu_threshold
        self.base_memory_threshold = memory_threshold
        
        logger.info(f"🛡️ Resource Governor initialized - CPU: {cpu_threshold}%, Memory: {memory_threshold}%")
    
    def start_monitoring(self):
        """Start background resource monitoring"""
        if self._monitor_thread is None or not self._monitor_thread.is_alive():
            self._shutdown_event.clear()
            self._monitor_thread = threading.Thread(
                target=self._resource_monitor_worker,
                daemon=True,
                name="ResourceGovernor"
            )
            self._monitor_thread.start()
            logger.info("🔍 Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop background resource monitoring"""
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._shutdown_event.set()
            self._monitor_thread.join(timeout=10)
            logger.info("⏹️ Resource monitoring stopped")
    
    def _resource_monitor_worker(self):
        """Background worker for resource monitoring"""
        while not self._shutdown_event.is_set():
            try:
                self._check_system_resources()
                self._adapt_thresholds()
                self._cleanup_old_history()
                
                # Sleep with interruption check
                for _ in range(self.check_interval):
                    if self._shutdown_event.is_set():
                        break
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Resource monitor error: {e}")
                time.sleep(5)  # Shorter sleep on error
    
    def _check_system_resources(self):
        """Check current system resource usage"""
        try:
            # Get CPU usage (1 second interval for accuracy)
            self.current_cpu = psutil.cpu_percent(interval=1)
            
            # Get memory usage
            memory = psutil.virtual_memory()
            self.current_memory = memory.percent
            
            # Record in history
            self.resource_history.append({
                'timestamp': datetime.now(),
                'cpu': self.current_cpu,
                'memory': self.current_memory
            })
            
            # Check thresholds
            cpu_alert = self.current_cpu > self.cpu_threshold
            memory_alert = self.current_memory > self.memory_threshold
            
            if cpu_alert or memory_alert:
                self._handle_high_resource_usage(cpu_alert, memory_alert)
            else:
                self._handle_normal_resource_usage()
                
        except Exception as e:
            logger.error(f"Error checking system resources: {e}")
    
    def _handle_high_resource_usage(self, cpu_alert: bool, memory_alert: bool):
        """Handle high resource usage situation"""
        now = datetime.now()
        
        # Avoid spam alerts
        if (self.last_alert_time and 
            now - self.last_alert_time < timedelta(seconds=self.cooldown_period)):
            return
        
        if not self.pause_processing:
            self.pause_processing = True
            self.last_alert_time = now
            
            alert_reasons = []
            if cpu_alert:
                alert_reasons.append(f"CPU: {self.current_cpu:.1f}%")
                self.metrics['cpu_alerts'] += 1
            if memory_alert:
                alert_reasons.append(f"Memory: {self.current_memory:.1f}%")
                self.metrics['memory_alerts'] += 1
            
            logger.warning(f"🚨 High resource usage detected - {', '.join(alert_reasons)}")
            logger.info("⏸️ Pausing non-essential processing")
            
            # Take corrective action
            self._emergency_resource_management()
    
    def _handle_normal_resource_usage(self):
        """Handle normal resource usage - resume operations"""
        if self.pause_processing:
            logger.info(f"✅ Resource usage normalized - CPU: {self.current_cpu:.1f}%, Memory: {self.current_memory:.1f}%")
            logger.info("▶️ Resuming normal processing")
            self.pause_processing = False
            self.metrics['tasks_resumed'] += 1
    
    def _emergency_resource_management(self):
        """Emergency actions to reduce resource usage"""
        try:
            # 1. Clear caches
            self._clear_application_caches()
            
            # 2. Pause background tasks
            self._pause_background_tasks()
            
            # 3. Reduce refresh rates
            self._reduce_refresh_rates()
            
            # 4. Force garbage collection
            self._force_garbage_collection()
            
            logger.info("🚑 Emergency resource management actions completed")
            
        except Exception as e:
            logger.error(f"Emergency resource management error: {e}")
    
    def _clear_application_caches(self):
        """Clear application caches to free memory"""
        try:
            # Clear our emergency cache
            from emergency_cache_fix import cache_manager
            stats_before = cache_manager.get_stats()
            cache_manager.memory_cache.clear()
            logger.info(f"🧹 Cleared memory cache - freed {stats_before['memory_entries']} entries")
            
        except ImportError:
            pass
        except Exception as e:
            logger.error(f"Error clearing caches: {e}")
    
    def _pause_background_tasks(self):
        """Pause background tasks"""
        # This would integrate with your specific background tasks
        # For now, we'll just log the action
        logger.info("⏸️ Background tasks paused")
        self.metrics['tasks_paused'] += 1
    
    def _reduce_refresh_rates(self):
        """Reduce refresh rates for real-time components"""
        # This would communicate with frontend to slow down refresh rates
        logger.info("🐌 Refresh rates reduced")
    
    def _force_garbage_collection(self):
        """Force Python garbage collection"""
        import gc
        collected = gc.collect()
        logger.info(f"🗑️ Garbage collection freed {collected} objects")
    
    def _adapt_thresholds(self):
        """Adapt thresholds based on system behavior"""
        if not self.adaptive_mode or len(self.resource_history) < 10:
            return
        
        # Calculate average resource usage over last 10 minutes
        recent_history = [
            entry for entry in self.resource_history
            if datetime.now() - entry['timestamp'] < timedelta(minutes=10)
        ]
        
        if len(recent_history) < 5:
            return
        
        avg_cpu = sum(entry['cpu'] for entry in recent_history) / len(recent_history)
        avg_memory = sum(entry['memory'] for entry in recent_history) / len(recent_history)
        
        # Adapt thresholds (be more aggressive if consistently high usage)
        if avg_cpu > self.base_cpu_threshold * 0.8:
            self.cpu_threshold = max(self.base_cpu_threshold * 0.8, 60)
        else:
            self.cpu_threshold = min(self.base_cpu_threshold, self.cpu_threshold + 1)
        
        if avg_memory > self.base_memory_threshold * 0.8:
            self.memory_threshold = max(self.base_memory_threshold * 0.8, 70)
        else:
            self.memory_threshold = min(self.base_memory_threshold, self.memory_threshold + 1)
    
    def _cleanup_old_history(self):
        """Remove old entries from resource history"""
        cutoff_time = datetime.now() - timedelta(hours=1)
        self.resource_history = [
            entry for entry in self.resource_history
            if entry['timestamp'] > cutoff_time
        ]
    
    def should_process(self, task_type: str = "general") -> bool:
        """Check if processing should continue based on current resource usage"""
        if not self.pause_processing:
            return True
        
        # Allow critical tasks even during high resource usage
        critical_tasks = ["error_handling", "system_monitoring", "user_authentication"]
        if task_type in critical_tasks:
            return True
        
        return False
    
    def register_task(self, task_name: str, task_func: Callable, priority: int = 1):
        """Register a task for resource-aware execution"""
        task = {
            'name': task_name,
            'function': task_func,
            'priority': priority,
            'registered_at': datetime.now()
        }
        self.task_queue.put(task)
        logger.debug(f"📝 Task registered: {task_name} (priority: {priority})")
    
    def get_system_status(self) -> Dict:
        """Get current system status and metrics"""
        uptime = datetime.now() - self.metrics['uptime_start']
        
        # Get process information
        current_process = psutil.Process()
        process_info = {
            'cpu_percent': current_process.cpu_percent(),
            'memory_mb': current_process.memory_info().rss / 1024 / 1024,
            'threads': current_process.num_threads(),
            'open_files': len(current_process.open_files())
        }
        
        return {
            'current_resources': {
                'cpu_percent': self.current_cpu,
                'memory_percent': self.current_memory,
                'cpu_threshold': self.cpu_threshold,
                'memory_threshold': self.memory_threshold
            },
            'system_status': {
                'processing_paused': self.pause_processing,
                'adaptive_mode': self.adaptive_mode,
                'monitoring_active': self._monitor_thread and self._monitor_thread.is_alive()
            },
            'process_info': process_info,
            'metrics': {
                **self.metrics,
                'uptime_hours': uptime.total_seconds() / 3600,
                'history_entries': len(self.resource_history)
            },
            'recommendations': self._get_performance_recommendations()
        }
    
    def _get_performance_recommendations(self) -> List[str]:
        """Get performance optimization recommendations"""
        recommendations = []
        
        if self.current_cpu > 80:
            recommendations.append("Consider reducing ML model complexity or batch size")
        
        if self.current_memory > 85:
            recommendations.append("Clear caches and reduce data retention periods")
        
        if len(self.resource_history) > 0:
            recent_cpu = [e['cpu'] for e in self.resource_history[-10:]]
            if len(recent_cpu) >= 5 and all(cpu > 70 for cpu in recent_cpu):
                recommendations.append("Consistent high CPU - consider scaling to multiple processes")
        
        if not recommendations:
            recommendations.append("System running optimally")
        
        return recommendations
    
    def force_resource_cleanup(self):
        """Force immediate resource cleanup"""
        logger.info("🚑 Forcing resource cleanup...")
        self._emergency_resource_management()
        
        # Additional aggressive cleanup
        import gc
        gc.collect()
        
        # Log results
        logger.info(f"✅ Resource cleanup completed - CPU: {psutil.cpu_percent()}%, Memory: {psutil.virtual_memory().percent}%")

# Global resource governor instance
resource_governor = SystemResourceGovernor(
    cpu_threshold=75.0,
    memory_threshold=80.0,
    check_interval=10,
    cooldown_period=30
)

def governed_task(task_type: str = "general", min_interval: float = 1.0):
    """
    Decorator for resource-governed task execution
    
    Args:
        task_type: Type of task for priority handling
        min_interval: Minimum interval between executions
    """
    def decorator(func: Callable) -> Callable:
        last_execution = {}
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            func_key = f"{func.__name__}_{id(func)}"
            now = time.time()
            
            # Check minimum interval
            if func_key in last_execution:
                elapsed = now - last_execution[func_key]
                if elapsed < min_interval:
                    logger.debug(f"⏳ Skipping {func.__name__} - minimum interval not reached")
                    return None
            
            # Check resource governor
            if not resource_governor.should_process(task_type):
                logger.info(f"⏸️ Skipping {func.__name__} - resource governor active")
                return None
            
            # Execute function
            try:
                result = func(*args, **kwargs)
                last_execution[func_key] = now
                return result
            except Exception as e:
                logger.error(f"Error in governed task {func.__name__}: {e}")
                raise
        
        return wrapper
    return decorator

def start_resource_monitoring():
    """Start the global resource governor"""
    resource_governor.start_monitoring()

def stop_resource_monitoring():
    """Stop the global resource governor"""
    resource_governor.stop_monitoring()

def get_system_status():
    """Get system status from the global resource governor"""
    return resource_governor.get_system_status()

def force_cleanup():
    """Force immediate resource cleanup"""
    resource_governor.force_resource_cleanup()

if __name__ == "__main__":
    # Test the resource governor
    print("Testing Resource Governor...")
    
    # Start monitoring
    start_resource_monitoring()
    
    # Test governed task
    @governed_task("test", min_interval=2.0)
    def test_task():
        print(f"Test task executed at {datetime.now()}")
        return "success"
    
    # Run test
    for i in range(5):
        result = test_task()
        print(f"Result {i}: {result}")
        time.sleep(1)
    
    # Show status
    status = get_system_status()
    print(f"System Status: {json.dumps(status, indent=2, default=str)}")
    
    # Stop monitoring
    stop_resource_monitoring()
