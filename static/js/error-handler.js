/**
 * Frontend Error Handling System for GoldGPT
 * Provides consistent error handling, user notifications, and debugging
 */

class ErrorHandler {
    constructor() {
        this.errorLog = [];
        this.maxLogSize = 100;
        this.setupGlobalErrorHandler();
        this.setupUnhandledRejectionHandler();
    }

    /**
     * Error types matching backend
     */
    static ErrorTypes = {
        VALIDATION_ERROR: 'validation_error',
        DATA_PIPELINE_ERROR: 'data_pipeline_error',
        ML_PREDICTION_ERROR: 'ml_prediction_error',
        API_ERROR: 'api_error',
        WEBSOCKET_ERROR: 'websocket_error',
        NETWORK_ERROR: 'network_error',
        UI_ERROR: 'ui_error',
        CHART_ERROR: 'chart_error',
        AUTHENTICATION_ERROR: 'authentication_error',
        TIMEOUT_ERROR: 'timeout_error'
    };

    /**
     * Error severity levels
     */
    static Severity = {
        LOW: 'low',
        MEDIUM: 'medium',
        HIGH: 'high',
        CRITICAL: 'critical'
    };

    /**
     * Setup global error handler for uncaught errors
     */
    setupGlobalErrorHandler() {
        window.addEventListener('error', (event) => {
            this.handleError({
                type: ErrorHandler.ErrorTypes.UI_ERROR,
                message: event.message,
                severity: ErrorHandler.Severity.HIGH,
                source: event.filename,
                line: event.lineno,
                column: event.colno,
                stack: event.error?.stack,
                timestamp: new Date().toISOString()
            });
        });
    }

    /**
     * Setup handler for unhandled promise rejections
     */
    setupUnhandledRejectionHandler() {
        window.addEventListener('unhandledrejection', (event) => {
            this.handleError({
                type: ErrorHandler.ErrorTypes.API_ERROR,
                message: `Unhandled promise rejection: ${event.reason}`,
                severity: ErrorHandler.Severity.HIGH,
                stack: event.reason?.stack,
                timestamp: new Date().toISOString()
            });
        });
    }

    /**
     * Main error handling method
     */
    handleError(errorInfo) {
        // Add to error log
        this.addToErrorLog(errorInfo);

        // Log to console with appropriate level
        this.logToConsole(errorInfo);

        // Show user notification if appropriate
        if (this.shouldShowUserNotification(errorInfo)) {
            this.showUserNotification(errorInfo);
        }

        // Send to backend for logging (if configured)
        this.reportToBackend(errorInfo);

        return errorInfo;
    }

    /**
     * Handle API response errors
     */
    handleApiError(response, requestInfo = {}) {
        let errorInfo = {
            type: ErrorHandler.ErrorTypes.API_ERROR,
            severity: ErrorHandler.Severity.MEDIUM,
            timestamp: new Date().toISOString(),
            context: requestInfo
        };

        if (response.ok) {
            return null; // No error
        }

        // Determine error details based on status
        switch (response.status) {
            case 400:
                errorInfo.message = 'Bad request - please check your input';
                errorInfo.userMessage = 'Invalid request. Please check your input and try again.';
                break;
            case 401:
                errorInfo.type = ErrorHandler.ErrorTypes.AUTHENTICATION_ERROR;
                errorInfo.message = 'Authentication required';
                errorInfo.userMessage = 'Please log in to continue.';
                errorInfo.severity = ErrorHandler.Severity.HIGH;
                break;
            case 403:
                errorInfo.message = 'Access forbidden';
                errorInfo.userMessage = 'You do not have permission to access this resource.';
                errorInfo.severity = ErrorHandler.Severity.HIGH;
                break;
            case 404:
                errorInfo.message = 'Resource not found';
                errorInfo.userMessage = 'The requested resource was not found.';
                errorInfo.severity = ErrorHandler.Severity.LOW;
                break;
            case 429:
                errorInfo.type = ErrorHandler.ErrorTypes.TIMEOUT_ERROR;
                errorInfo.message = 'Rate limit exceeded';
                errorInfo.userMessage = 'Too many requests. Please wait and try again.';
                errorInfo.severity = ErrorHandler.Severity.MEDIUM;
                break;
            case 500:
                errorInfo.message = 'Internal server error';
                errorInfo.userMessage = 'A server error occurred. Please try again later.';
                errorInfo.severity = ErrorHandler.Severity.HIGH;
                break;
            case 502:
            case 503:
            case 504:
                errorInfo.type = ErrorHandler.ErrorTypes.NETWORK_ERROR;
                errorInfo.message = 'Service unavailable';
                errorInfo.userMessage = 'Service is temporarily unavailable. Please try again later.';
                errorInfo.severity = ErrorHandler.Severity.HIGH;
                break;
            default:
                errorInfo.message = `HTTP ${response.status}: ${response.statusText}`;
                errorInfo.userMessage = 'An unexpected error occurred. Please try again.';
                errorInfo.severity = ErrorHandler.Severity.MEDIUM;
        }

        errorInfo.status = response.status;
        errorInfo.statusText = response.statusText;

        return this.handleError(errorInfo);
    }

    /**
     * Handle ML prediction specific errors
     */
    handleMlPredictionError(error, symbol = null, timeframe = null) {
        const errorInfo = {
            type: ErrorHandler.ErrorTypes.ML_PREDICTION_ERROR,
            message: `ML prediction failed: ${error.message || error}`,
            severity: ErrorHandler.Severity.HIGH,
            userMessage: 'Unable to generate predictions at this time.',
            suggestedAction: 'Please try again in a few moments.',
            context: { symbol, timeframe },
            timestamp: new Date().toISOString(),
            stack: error.stack
        };

        return this.handleError(errorInfo);
    }

    /**
     * Handle WebSocket connection errors
     */
    handleWebSocketError(error, eventType = null) {
        const errorInfo = {
            type: ErrorHandler.ErrorTypes.WEBSOCKET_ERROR,
            message: `WebSocket error: ${error.message || error}`,
            severity: ErrorHandler.Severity.MEDIUM,
            userMessage: 'Real-time connection lost.',
            suggestedAction: 'Refresh the page to reconnect.',
            context: { eventType },
            timestamp: new Date().toISOString(),
            stack: error.stack
        };

        return this.handleError(errorInfo);
    }

    /**
     * Handle chart/visualization errors
     */
    handleChartError(error, chartType = null) {
        const errorInfo = {
            type: ErrorHandler.ErrorTypes.CHART_ERROR,
            message: `Chart error: ${error.message || error}`,
            severity: ErrorHandler.Severity.MEDIUM,
            userMessage: 'Unable to display chart.',
            suggestedAction: 'Please refresh the page.',
            context: { chartType },
            timestamp: new Date().toISOString(),
            stack: error.stack
        };

        return this.handleError(errorInfo);
    }

    /**
     * Handle network/fetch errors
     */
    handleNetworkError(error, url = null) {
        const errorInfo = {
            type: ErrorHandler.ErrorTypes.NETWORK_ERROR,
            message: `Network error: ${error.message || error}`,
            severity: ErrorHandler.Severity.HIGH,
            userMessage: 'Connection error. Please check your internet connection.',
            suggestedAction: 'Please try again or refresh the page.',
            context: { url },
            timestamp: new Date().toISOString(),
            stack: error.stack
        };

        return this.handleError(errorInfo);
    }

    /**
     * Add error to internal log
     */
    addToErrorLog(errorInfo) {
        this.errorLog.unshift({
            ...errorInfo,
            id: Date.now() + Math.random()
        });

        // Maintain log size
        if (this.errorLog.length > this.maxLogSize) {
            this.errorLog = this.errorLog.slice(0, this.maxLogSize);
        }
    }

    /**
     * Log to browser console with appropriate level
     */
    logToConsole(errorInfo) {
        const logMessage = `[${errorInfo.type}] ${errorInfo.message}`;
        const logData = {
            ...errorInfo,
            userAgent: navigator.userAgent,
            url: window.location.href
        };

        switch (errorInfo.severity) {
            case ErrorHandler.Severity.CRITICAL:
            case ErrorHandler.Severity.HIGH:
                console.error(logMessage, logData);
                break;
            case ErrorHandler.Severity.MEDIUM:
                console.warn(logMessage, logData);
                break;
            case ErrorHandler.Severity.LOW:
                console.info(logMessage, logData);
                break;
            default:
                console.log(logMessage, logData);
        }
    }

    /**
     * Determine if user should see a notification
     */
    shouldShowUserNotification(errorInfo) {
        // Don't show for low severity errors
        if (errorInfo.severity === ErrorHandler.Severity.LOW) {
            return false;
        }

        // Don't show duplicate notifications within short time
        const recentErrors = this.errorLog.slice(0, 5);
        const duplicateRecent = recentErrors.some(err => 
            err.type === errorInfo.type && 
            err.message === errorInfo.message &&
            (Date.now() - new Date(err.timestamp).getTime()) < 5000
        );

        return !duplicateRecent;
    }

    /**
     * Show user notification
     */
    showUserNotification(errorInfo) {
        const message = errorInfo.userMessage || errorInfo.message;
        const action = errorInfo.suggestedAction;

        // Try to use existing notification system
        if (typeof showNotification === 'function') {
            showNotification(message, 'error', action);
        } else if (typeof showToast === 'function') {
            showToast(message, 'error');
        } else {
            // Fallback to custom notification
            this.showFallbackNotification(message, errorInfo.severity, action);
        }
    }

    /**
     * Fallback notification system
     */
    showFallbackNotification(message, severity, action = null) {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `error-notification severity-${severity}`;
        notification.innerHTML = `
            <div class="error-content">
                <span class="error-message">${message}</span>
                ${action ? `<span class="error-action">${action}</span>` : ''}
                <button class="error-close">&times;</button>
            </div>
        `;

        // Add styles if not already present
        if (!document.getElementById('error-notification-styles')) {
            const styles = document.createElement('style');
            styles.id = 'error-notification-styles';
            styles.textContent = `
                .error-notification {
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    max-width: 400px;
                    padding: 15px;
                    border-radius: 5px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    z-index: 10000;
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    animation: slideIn 0.3s ease-out;
                }
                .error-notification.severity-low { background: #e3f2fd; border-left: 4px solid #2196f3; }
                .error-notification.severity-medium { background: #fff3e0; border-left: 4px solid #ff9800; }
                .error-notification.severity-high { background: #ffebee; border-left: 4px solid #f44336; }
                .error-notification.severity-critical { background: #f3e5f5; border-left: 4px solid #9c27b0; }
                .error-content { display: flex; flex-direction: column; gap: 5px; }
                .error-message { font-weight: 500; }
                .error-action { font-size: 0.9em; opacity: 0.8; }
                .error-close { 
                    position: absolute; top: 10px; right: 10px; 
                    background: none; border: none; font-size: 20px; 
                    cursor: pointer; opacity: 0.7; 
                }
                .error-close:hover { opacity: 1; }
                @keyframes slideIn {
                    from { transform: translateX(100%); opacity: 0; }
                    to { transform: translateX(0); opacity: 1; }
                }
            `;
            document.head.appendChild(styles);
        }

        // Add to page
        document.body.appendChild(notification);

        // Add close handler
        const closeBtn = notification.querySelector('.error-close');
        closeBtn.addEventListener('click', () => {
            notification.remove();
        });

        // Auto-remove after delay
        const delay = severity === 'critical' ? 10000 : 5000;
        setTimeout(() => {
            if (notification.parentNode) {
                notification.remove();
            }
        }, delay);
    }

    /**
     * Report error to backend for centralized logging
     */
    async reportToBackend(errorInfo) {
        try {
            // Only report significant errors to avoid spam
            if (errorInfo.severity === ErrorHandler.Severity.LOW) {
                return;
            }

            const reportData = {
                ...errorInfo,
                userAgent: navigator.userAgent,
                url: window.location.href,
                timestamp: new Date().toISOString(),
                sessionId: this.getSessionId()
            };

            // Send to error reporting endpoint
            await fetch('/api/error-report', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(reportData)
            });
        } catch (e) {
            // Don't throw errors from error reporting
            console.warn('Failed to report error to backend:', e);
        }
    }

    /**
     * Get or generate session ID
     */
    getSessionId() {
        let sessionId = sessionStorage.getItem('error_session_id');
        if (!sessionId) {
            sessionId = Date.now() + '-' + Math.random().toString(36).substr(2, 9);
            sessionStorage.setItem('error_session_id', sessionId);
        }
        return sessionId;
    }

    /**
     * Get error log for debugging
     */
    getErrorLog() {
        return [...this.errorLog];
    }

    /**
     * Clear error log
     */
    clearErrorLog() {
        this.errorLog = [];
    }

    /**
     * Export error log for support
     */
    exportErrorLog() {
        const exportData = {
            timestamp: new Date().toISOString(),
            userAgent: navigator.userAgent,
            url: window.location.href,
            errors: this.errorLog
        };

        const blob = new Blob([JSON.stringify(exportData, null, 2)], {
            type: 'application/json'
        });

        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `goldgpt-error-log-${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);
    }
}

/**
 * Enhanced fetch wrapper with error handling
 */
async function safeFetch(url, options = {}) {
    const errorHandler = window.globalErrorHandler || new ErrorHandler();
    
    try {
        // Add timeout if not specified
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), options.timeout || 30000);
        
        const response = await fetch(url, {
            ...options,
            signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        // Handle HTTP errors
        if (!response.ok) {
            const error = errorHandler.handleApiError(response, { url, options });
            throw new Error(error?.userMessage || `HTTP ${response.status}`);
        }
        
        return response;
        
    } catch (error) {
        if (error.name === 'AbortError') {
            errorHandler.handleError({
                type: ErrorHandler.ErrorTypes.TIMEOUT_ERROR,
                message: `Request timeout: ${url}`,
                severity: ErrorHandler.Severity.MEDIUM,
                userMessage: 'Request timed out. Please try again.',
                context: { url, timeout: options.timeout || 30000 }
            });
        } else {
            errorHandler.handleNetworkError(error, url);
        }
        throw error;
    }
}

/**
 * Enhanced async function wrapper
 */
function withErrorHandling(asyncFunction, errorType = ErrorHandler.ErrorTypes.UI_ERROR) {
    return async function(...args) {
        const errorHandler = window.globalErrorHandler || new ErrorHandler();
        
        try {
            return await asyncFunction.apply(this, args);
        } catch (error) {
            errorHandler.handleError({
                type: errorType,
                message: `Error in ${asyncFunction.name}: ${error.message}`,
                severity: ErrorHandler.Severity.MEDIUM,
                stack: error.stack,
                context: { functionName: asyncFunction.name, args }
            });
            throw error;
        }
    };
}

// Initialize global error handler
window.globalErrorHandler = new ErrorHandler();

// Make classes available globally
window.ErrorHandler = ErrorHandler;
window.safeFetch = safeFetch;
window.withErrorHandling = withErrorHandling;

console.log('Frontend error handling system initialized');
