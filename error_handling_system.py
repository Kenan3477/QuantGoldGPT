#!/usr/bin/env python3
"""
Standardized Error Handling System for GoldGPT
Provides consistent error responses, logging, and debugging across all components
"""

import logging
import traceback
import sys
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass, asdict
from flask import jsonify, request
import functools

# Configure comprehensive logging
def setup_logging():
    """Setup comprehensive logging configuration"""
    
    # Create formatters for different log levels
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-20s | %(funcName)-20s | Line:%(lineno)-4d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler for general output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler for detailed debugging
    try:
        file_handler = logging.FileHandler('goldgpt_debug.log', mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Could not create log file: {e}")
    
    # Error file handler for errors only
    try:
        error_handler = logging.FileHandler('goldgpt_errors.log', mode='a', encoding='utf-8')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(error_handler)
    except Exception as e:
        print(f"Warning: Could not create error log file: {e}")
    
    return root_logger

class ErrorType(Enum):
    """Standardized error types"""
    VALIDATION_ERROR = "validation_error"
    DATA_PIPELINE_ERROR = "data_pipeline_error"
    ML_PREDICTION_ERROR = "ml_prediction_error"
    API_ERROR = "api_error"
    WEBSOCKET_ERROR = "websocket_error"
    DATABASE_ERROR = "database_error"
    EXTERNAL_SERVICE_ERROR = "external_service_error"
    AUTHENTICATION_ERROR = "authentication_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    INTERNAL_ERROR = "internal_error"
    CONFIGURATION_ERROR = "configuration_error"

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class StandardError:
    """Standardized error response structure"""
    error_type: ErrorType
    message: str
    severity: ErrorSeverity
    timestamp: str
    request_id: Optional[str] = None
    component: Optional[str] = None
    function_name: Optional[str] = None
    line_number: Optional[int] = None
    stack_trace: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    user_message: Optional[str] = None
    suggested_action: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON response"""
        error_dict = {
            'success': False,
            'error': {
                'type': self.error_type.value,
                'message': self.message,
                'severity': self.severity.value,
                'timestamp': self.timestamp,
                'user_message': self.user_message or self.message,
                'suggested_action': self.suggested_action
            }
        }
        
        # Add optional fields if present
        if self.request_id:
            error_dict['error']['request_id'] = self.request_id
        if self.component:
            error_dict['error']['component'] = self.component
        if self.context:
            error_dict['error']['context'] = self.context
            
        # Include debug info only in development
        if logging.getLogger().level == logging.DEBUG:
            if self.function_name:
                error_dict['error']['function'] = self.function_name
            if self.line_number:
                error_dict['error']['line'] = self.line_number
            if self.stack_trace:
                error_dict['error']['stack_trace'] = self.stack_trace
                
        return error_dict

class ErrorHandler:
    """Centralized error handling system"""
    
    def __init__(self, component_name: str):
        self.component_name = component_name
        self.logger = logging.getLogger(component_name)
        
    def create_error(self,
                    error_type: ErrorType,
                    message: str,
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                    exception: Optional[Exception] = None,
                    context: Optional[Dict[str, Any]] = None,
                    user_message: Optional[str] = None,
                    suggested_action: Optional[str] = None) -> StandardError:
        """Create a standardized error object"""
        
        # Get call stack info
        frame = sys._getframe(1)
        function_name = frame.f_code.co_name
        line_number = frame.f_lineno
        
        # Generate request ID if available
        request_id = None
        try:
            if hasattr(request, 'id'):
                request_id = request.id
            elif hasattr(request, 'environ'):
                request_id = id(request)
        except:
            pass
            
        # Get stack trace if exception provided
        stack_trace = None
        if exception:
            stack_trace = ''.join(traceback.format_exception(
                type(exception), exception, exception.__traceback__
            ))
            
        # Create error object
        error = StandardError(
            error_type=error_type,
            message=message,
            severity=severity,
            timestamp=datetime.now(timezone.utc).isoformat(),
            request_id=str(request_id) if request_id else None,
            component=self.component_name,
            function_name=function_name,
            line_number=line_number,
            stack_trace=stack_trace,
            context=context,
            user_message=user_message,
            suggested_action=suggested_action
        )
        
        # Log the error
        self._log_error(error, exception)
        
        return error
    
    def _log_error(self, error: StandardError, exception: Optional[Exception] = None):
        """Log the error with appropriate level"""
        
        log_message = f"{error.error_type.value.upper()}: {error.message}"
        
        if error.context:
            log_message += f" | Context: {error.context}"
            
        if error.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message, exc_info=exception)
        elif error.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message, exc_info=exception)
        elif error.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message, exc_info=exception)
        else:
            self.logger.info(log_message, exc_info=exception)
    
    def handle_ml_prediction_error(self, 
                                  exception: Exception,
                                  symbol: str = None,
                                  timeframe: str = None) -> StandardError:
        """Handle ML prediction specific errors"""
        
        context = {}
        if symbol:
            context['symbol'] = symbol
        if timeframe:
            context['timeframe'] = timeframe
            
        user_message = "Unable to generate ML predictions at this time"
        suggested_action = "Please try again in a few moments or check system status"
        
        return self.create_error(
            error_type=ErrorType.ML_PREDICTION_ERROR,
            message=f"ML prediction failed: {str(exception)}",
            severity=ErrorSeverity.HIGH,
            exception=exception,
            context=context,
            user_message=user_message,
            suggested_action=suggested_action
        )
    
    def handle_data_pipeline_error(self,
                                  exception: Exception,
                                  source: str = None) -> StandardError:
        """Handle data pipeline specific errors"""
        
        context = {}
        if source:
            context['data_source'] = source
            
        user_message = "Unable to fetch current market data"
        suggested_action = "Price data may be delayed. Check your connection or try again later"
        
        return self.create_error(
            error_type=ErrorType.DATA_PIPELINE_ERROR,
            message=f"Data pipeline error: {str(exception)}",
            severity=ErrorSeverity.HIGH,
            exception=exception,
            context=context,
            user_message=user_message,
            suggested_action=suggested_action
        )
    
    def handle_api_error(self,
                        exception: Exception,
                        endpoint: str = None,
                        method: str = None) -> StandardError:
        """Handle API specific errors"""
        
        context = {}
        if endpoint:
            context['endpoint'] = endpoint
        if method:
            context['method'] = method
            
        user_message = "API request failed"
        suggested_action = "Please check your request and try again"
        
        return self.create_error(
            error_type=ErrorType.API_ERROR,
            message=f"API error: {str(exception)}",
            severity=ErrorSeverity.MEDIUM,
            exception=exception,
            context=context,
            user_message=user_message,
            suggested_action=suggested_action
        )
    
    def handle_websocket_error(self,
                              exception: Exception,
                              event_type: str = None) -> StandardError:
        """Handle WebSocket specific errors"""
        
        context = {}
        if event_type:
            context['event_type'] = event_type
            
        user_message = "Real-time connection error"
        suggested_action = "Refresh the page to reconnect to live updates"
        
        return self.create_error(
            error_type=ErrorType.WEBSOCKET_ERROR,
            message=f"WebSocket error: {str(exception)}",
            severity=ErrorSeverity.MEDIUM,
            exception=exception,
            context=context,
            user_message=user_message,
            suggested_action=suggested_action
        )

def error_handler_decorator(component_name: str):
    """Decorator for automatic error handling"""
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            error_handler = ErrorHandler(component_name)
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error = error_handler.create_error(
                    error_type=ErrorType.INTERNAL_ERROR,
                    message=f"Unexpected error in {func.__name__}: {str(e)}",
                    severity=ErrorSeverity.HIGH,
                    exception=e
                )
                
                # Return appropriate response based on context
                try:
                    # If this is a Flask route, return JSON response
                    from flask import jsonify
                    return jsonify(error.to_dict()), 500
                except:
                    # Otherwise, raise the original exception
                    raise e
                    
        return wrapper
    return decorator

def setup_flask_error_handlers(app):
    """Setup Flask-specific error handlers"""
    
    @app.errorhandler(404)
    def handle_404(error):
        error_handler = ErrorHandler('flask_app')
        std_error = error_handler.create_error(
            error_type=ErrorType.API_ERROR,
            message="Endpoint not found",
            severity=ErrorSeverity.LOW,
            user_message="The requested page or API endpoint was not found",
            suggested_action="Please check the URL and try again"
        )
        return jsonify(std_error.to_dict()), 404
    
    @app.errorhandler(500)
    def handle_500(error):
        error_handler = ErrorHandler('flask_app')
        std_error = error_handler.create_error(
            error_type=ErrorType.INTERNAL_ERROR,
            message="Internal server error",
            severity=ErrorSeverity.CRITICAL,
            exception=error,
            user_message="An unexpected error occurred",
            suggested_action="Please try again later or contact support"
        )
        return jsonify(std_error.to_dict()), 500
    
    @app.errorhandler(Exception)
    def handle_exception(error):
        error_handler = ErrorHandler('flask_app')
        std_error = error_handler.create_error(
            error_type=ErrorType.INTERNAL_ERROR,
            message=f"Unhandled exception: {str(error)}",
            severity=ErrorSeverity.CRITICAL,
            exception=error,
            user_message="An unexpected error occurred",
            suggested_action="Please try again later"
        )
        return jsonify(std_error.to_dict()), 500

# Initialize logging on module import
logger = setup_logging()

# Example usage
if __name__ == "__main__":
    # Test the error handling system
    error_handler = ErrorHandler('test_component')
    
    try:
        raise ValueError("This is a test error")
    except Exception as e:
        error = error_handler.handle_ml_prediction_error(e, 'XAUUSD', '1H')
        print("Error Response:", error.to_dict())
        
    print("Error handling system test completed!")
