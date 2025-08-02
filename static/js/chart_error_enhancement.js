/**
 * Chart Error Enhancement System
 * Adds robust error handling to existing chart integrations
 */

// Enhanced error handling for auto-integration
if (window.goldGPTChartAutoIntegration) {
    const originalIntegration = window.goldGPTChartAutoIntegration;
    
    // Add error styling
    const errorStyles = document.createElement('style');
    errorStyles.id = 'chart-error-enhancement-styles';
    errorStyles.textContent = `
        .chart-error-overlay {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(255, 0, 0, 0.1);
            border: 2px solid rgba(255, 0, 0, 0.3);
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 1000;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        
        .chart-error-content {
            text-align: center;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            max-width: 300px;
        }
        
        .chart-error-icon {
            font-size: 48px;
            margin-bottom: 15px;
        }
        
        .chart-error-title {
            color: #ff4444;
            font-weight: bold;
            margin-bottom: 10px;
            font-size: 16px;
        }
        
        .chart-error-message {
            color: #666;
            margin-bottom: 15px;
            font-size: 14px;
            line-height: 1.4;
        }
        
        .chart-error-actions {
            display: flex;
            gap: 10px;
            justify-content: center;
        }
        
        .chart-error-btn {
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            transition: all 0.2s ease;
        }
        
        .chart-error-retry {
            background: #4CAF50;
            color: white;
        }
        
        .chart-error-retry:hover {
            background: #45a049;
        }
        
        .chart-error-details {
            background: #2196F3;
            color: white;
        }
        
        .chart-error-details:hover {
            background: #1976D2;
        }
        
        .chart-error-dismiss {
            background: #666;
            color: white;
        }
        
        .chart-error-dismiss:hover {
            background: #555;
        }
        
        .chart-recovery-indicator {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(255, 255, 255, 0.95);
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            z-index: 1001;
        }
        
        .chart-recovery-spinner {
            width: 32px;
            height: 32px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #2196F3;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .global-chart-error {
            position: fixed;
            top: 20px;
            right: 20px;
            max-width: 400px;
            background: white;
            border: 2px solid #ff4444;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            z-index: 10000;
            padding: 20px;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        
        .global-error-title {
            color: #ff4444;
            font-weight: bold;
            margin-bottom: 10px;
            font-size: 18px;
        }
        
        .global-error-message {
            color: #666;
            margin-bottom: 15px;
            font-size: 14px;
        }
        
        .error-details-toggle {
            background: none;
            border: none;
            color: #2196F3;
            cursor: pointer;
            text-decoration: underline;
            font-size: 12px;
            margin-bottom: 15px;
        }
        
        .error-details {
            background: #f5f5f5;
            padding: 10px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 11px;
            margin-bottom: 15px;
            max-height: 150px;
            overflow-y: auto;
            display: none;
        }
        
        .global-error-actions {
            display: flex;
            gap: 10px;
        }
        
        .global-error-btn {
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
        }
        
        .error-retry-all {
            background: #4CAF50;
            color: white;
        }
        
        .error-dismiss {
            background: #666;
            color: white;
        }
    `;
    document.head.appendChild(errorStyles);

    // Enhanced error handling functions
    window.ChartErrorHandler = {
        /**
         * Show error overlay on chart container
         */
        showChartError(containerId, error, options = {}) {
            const container = document.getElementById(containerId);
            if (!container) return;

            // Remove existing error overlays
            const existingOverlay = container.querySelector('.chart-error-overlay');
            if (existingOverlay) {
                existingOverlay.remove();
            }

            const overlay = document.createElement('div');
            overlay.className = 'chart-error-overlay';
            
            const errorCode = error.code || 'UNKNOWN_ERROR';
            const userFriendlyMessage = this.getUserFriendlyMessage(errorCode, error.message);
            
            overlay.innerHTML = `
                <div class="chart-error-content">
                    <div class="chart-error-icon">${this.getErrorIcon(errorCode)}</div>
                    <div class="chart-error-title">Chart Error</div>
                    <div class="chart-error-message">${userFriendlyMessage}</div>
                    <div class="chart-error-actions">
                        ${error.recoverable !== false ? `<button class="chart-error-btn chart-error-retry" onclick="ChartErrorHandler.retryChart('${containerId}')">Retry</button>` : ''}
                        <button class="chart-error-btn chart-error-details" onclick="ChartErrorHandler.showErrorDetails('${containerId}', '${errorCode}')">Details</button>
                        <button class="chart-error-btn chart-error-dismiss" onclick="ChartErrorHandler.dismissError('${containerId}')">Dismiss</button>
                    </div>
                </div>
            `;

            // Position container relative if not already
            if (getComputedStyle(container).position === 'static') {
                container.style.position = 'relative';
            }

            container.appendChild(overlay);
            
            console.error(`Chart Error [${containerId}]:`, error);
        },

        /**
         * Show recovery indicator
         */
        showRecoveryIndicator(containerId, message = 'Attempting recovery...') {
            const container = document.getElementById(containerId);
            if (!container) return;

            const indicator = document.createElement('div');
            indicator.className = 'chart-recovery-indicator';
            indicator.innerHTML = `
                <div class="chart-recovery-spinner"></div>
                <div>${message}</div>
            `;

            container.appendChild(indicator);
            return indicator;
        },

        /**
         * Hide recovery indicator
         */
        hideRecoveryIndicator(containerId) {
            const container = document.getElementById(containerId);
            if (!container) return;

            const indicator = container.querySelector('.chart-recovery-indicator');
            if (indicator) {
                indicator.remove();
            }
        },

        /**
         * Retry chart initialization
         */
        async retryChart(containerId) {
            const indicator = this.showRecoveryIndicator(containerId, 'Retrying chart...');
            
            try {
                // Use factory retry if available
                if (window.UnifiedChartManagerFactory && typeof window.UnifiedChartManagerFactory.retryChart === 'function') {
                    await window.UnifiedChartManagerFactory.retryChart(containerId);
                } else {
                    // Fallback: recreate chart through auto-integration
                    if (window.goldGPTChartAutoIntegration) {
                        await window.goldGPTChartAutoIntegration.destroyChart(containerId);
                        
                        // Find original container and config
                        const container = document.getElementById(containerId);
                        if (container) {
                            // Try to find matching config
                            for (const [selector, config] of Object.entries(CHART_AUTO_CONFIG.containers)) {
                                if (container.matches(selector)) {
                                    await window.goldGPTChartAutoIntegration.createAutoChart(containerId, config);
                                    break;
                                }
                            }
                        }
                    }
                }

                this.dismissError(containerId);
                this.showSuccessMessage(containerId, 'Chart recovered successfully!');
                
            } catch (retryError) {
                console.error(`Retry failed for ${containerId}:`, retryError);
                this.showChartError(containerId, {
                    code: 'RETRY_FAILED',
                    message: `Retry failed: ${retryError.message}`,
                    recoverable: true
                });
            } finally {
                this.hideRecoveryIndicator(containerId);
            }
        },

        /**
         * Show success message
         */
        showSuccessMessage(containerId, message) {
            const container = document.getElementById(containerId);
            if (!container) return;

            const successDiv = document.createElement('div');
            successDiv.className = 'chart-success-message';
            successDiv.style.cssText = `
                position: absolute;
                top: 10px;
                left: 10px;
                background: #4CAF50;
                color: white;
                padding: 8px 12px;
                border-radius: 4px;
                font-size: 12px;
                z-index: 1000;
                animation: fadeInOut 3s ease-out forwards;
            `;
            successDiv.textContent = message;

            // Add fadeInOut animation
            if (!document.querySelector('#success-animation-style')) {
                const style = document.createElement('style');
                style.id = 'success-animation-style';
                style.textContent = `
                    @keyframes fadeInOut {
                        0% { opacity: 0; transform: translateY(-10px); }
                        20% { opacity: 1; transform: translateY(0); }
                        80% { opacity: 1; transform: translateY(0); }
                        100% { opacity: 0; transform: translateY(-10px); }
                    }
                `;
                document.head.appendChild(style);
            }

            container.appendChild(successDiv);

            // Remove after animation
            setTimeout(() => {
                if (container.contains(successDiv)) {
                    successDiv.remove();
                }
            }, 3000);
        },

        /**
         * Dismiss error overlay
         */
        dismissError(containerId) {
            const container = document.getElementById(containerId);
            if (!container) return;

            const overlay = container.querySelector('.chart-error-overlay');
            if (overlay) {
                overlay.remove();
            }
        },

        /**
         * Show error details in modal
         */
        showErrorDetails(containerId, errorCode) {
            const modal = document.createElement('div');
            modal.className = 'error-details-modal';
            modal.style.cssText = `
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(0,0,0,0.5);
                display: flex;
                align-items: center;
                justify-content: center;
                z-index: 10001;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            `;

            const modalContent = document.createElement('div');
            modalContent.style.cssText = `
                background: white;
                padding: 30px;
                border-radius: 8px;
                max-width: 600px;
                max-height: 80vh;
                overflow-y: auto;
                position: relative;
            `;

            const debugInfo = this.gatherDebugInfo(containerId);
            
            modalContent.innerHTML = `
                <h3 style="margin-top: 0; color: #ff4444;">Chart Error Details</h3>
                <p><strong>Container:</strong> ${containerId}</p>
                <p><strong>Error Code:</strong> ${errorCode}</p>
                
                <h4>Debug Information:</h4>
                <pre style="background: #f5f5f5; padding: 15px; border-radius: 4px; overflow-x: auto; font-size: 11px;">${JSON.stringify(debugInfo, null, 2)}</pre>
                
                <h4>Troubleshooting Steps:</h4>
                <ul style="font-size: 14px; line-height: 1.5;">
                    ${this.getTroubleshootingSteps(errorCode).map(step => `<li>${step}</li>`).join('')}
                </ul>
                
                <div style="text-align: center; margin-top: 20px;">
                    <button onclick="this.parentElement.parentElement.parentElement.remove()" style="background: #2196F3; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer;">Close</button>
                </div>
            `;

            modal.appendChild(modalContent);
            document.body.appendChild(modal);

            // Close on background click
            modal.addEventListener('click', (e) => {
                if (e.target === modal) {
                    modal.remove();
                }
            });
        },

        /**
         * Show global error notification
         */
        showGlobalError(title, message, details = null) {
            // Remove existing global errors
            const existing = document.querySelector('.global-chart-error');
            if (existing) {
                existing.remove();
            }

            const errorDiv = document.createElement('div');
            errorDiv.className = 'global-chart-error';
            
            const detailsHtml = details ? `
                <button class="error-details-toggle" onclick="this.nextElementSibling.style.display = this.nextElementSibling.style.display === 'none' ? 'block' : 'none'">Show Details</button>
                <div class="error-details">${typeof details === 'object' ? JSON.stringify(details, null, 2) : details}</div>
            ` : '';

            errorDiv.innerHTML = `
                <div class="global-error-title">${title}</div>
                <div class="global-error-message">${message}</div>
                ${detailsHtml}
                <div class="global-error-actions">
                    <button class="global-error-btn error-retry-all" onclick="ChartErrorHandler.retryAllCharts()">Retry All</button>
                    <button class="global-error-btn error-dismiss" onclick="this.parentElement.parentElement.remove()">Dismiss</button>
                </div>
            `;

            document.body.appendChild(errorDiv);

            // Auto-dismiss after 30 seconds
            setTimeout(() => {
                if (document.body.contains(errorDiv)) {
                    errorDiv.remove();
                }
            }, 30000);
        },

        /**
         * Retry all failed charts
         */
        async retryAllCharts() {
            const containers = document.querySelectorAll('.chart-error-overlay');
            const retryPromises = Array.from(containers).map(overlay => {
                const container = overlay.closest('[id]');
                if (container) {
                    return this.retryChart(container.id);
                }
            }).filter(Boolean);

            try {
                await Promise.all(retryPromises);
                this.showGlobalError('✅ Recovery Complete', 'All charts have been successfully recovered.', null);
            } catch (error) {
                this.showGlobalError('❌ Recovery Failed', 'Some charts could not be recovered. Check individual chart errors for details.', error);
            }
        },

        /**
         * Get user-friendly error message
         */
        getUserFriendlyMessage(errorCode, originalMessage) {
            const messages = {
                'CONTAINER_NOT_FOUND': 'Chart container could not be found on the page.',
                'LIBRARY_NOT_AVAILABLE': 'Chart library is not loaded. Please refresh the page.',
                'LIBRARY_INITIALIZATION_FAILED': 'Failed to initialize the charting library.',
                'DATA_LOAD_FAILED': 'Could not load chart data. Check your connection.',
                'WEBSOCKET_DISCONNECTED': 'Real-time connection lost. Charts may not update.',
                'NETWORK_ERROR': 'Network error occurred. Please check your connection.',
                'CHART_RENDER_FAILED': 'Chart failed to render properly.',
                'NO_LIBRARY_AVAILABLE': 'No charting libraries are available.',
                'UNKNOWN_ERROR': 'An unexpected error occurred.',
                'RETRY_FAILED': 'Chart recovery attempt failed.'
            };

            return messages[errorCode] || originalMessage || 'An unknown error occurred.';
        },

        /**
         * Get error icon
         */
        getErrorIcon(errorCode) {
            const icons = {
                'CONTAINER_NOT_FOUND': '📭',
                'LIBRARY_NOT_AVAILABLE': '📚',
                'LIBRARY_INITIALIZATION_FAILED': '🔧',
                'DATA_LOAD_FAILED': '📡',
                'WEBSOCKET_DISCONNECTED': '🔌',
                'NETWORK_ERROR': '🌐',
                'CHART_RENDER_FAILED': '📊',
                'NO_LIBRARY_AVAILABLE': '❌',
                'RETRY_FAILED': '🔄'
            };

            return icons[errorCode] || '⚠️';
        },

        /**
         * Get troubleshooting steps
         */
        getTroubleshootingSteps(errorCode) {
            const steps = {
                'CONTAINER_NOT_FOUND': [
                    'Verify the chart container element exists in the DOM',
                    'Check if the container ID is correct',
                    'Ensure the container is not removed by other scripts'
                ],
                'LIBRARY_NOT_AVAILABLE': [
                    'Refresh the page to reload chart libraries',
                    'Check browser console for script loading errors',
                    'Verify CDN connections are working'
                ],
                'DATA_LOAD_FAILED': [
                    'Check your internet connection',
                    'Verify API endpoints are accessible',
                    'Check browser console for network errors',
                    'Try refreshing the chart data'
                ],
                'WEBSOCKET_DISCONNECTED': [
                    'Check your internet connection',
                    'WebSocket server may be temporarily unavailable',
                    'Real-time updates will resume when connection is restored'
                ],
                'DEFAULT': [
                    'Try refreshing the page',
                    'Check browser console for additional errors',
                    'Contact support if the problem persists'
                ]
            };

            return steps[errorCode] || steps['DEFAULT'];
        },

        /**
         * Gather debug information
         */
        gatherDebugInfo(containerId) {
            return {
                containerId,
                timestamp: new Date().toISOString(),
                userAgent: navigator.userAgent,
                containerExists: !!document.getElementById(containerId),
                availableLibraries: {
                    lightweightCharts: typeof LightweightCharts !== 'undefined',
                    chartjs: typeof Chart !== 'undefined',
                    tradingview: typeof TradingView !== 'undefined'
                },
                factoryStatus: window.UnifiedChartManagerFactory ? 
                    (typeof window.UnifiedChartManagerFactory.getFactoryStatus === 'function' ? 
                        window.UnifiedChartManagerFactory.getFactoryStatus() : 'Factory available') : 
                    'Factory not available',
                autoIntegrationStatus: window.goldGPTChartAutoIntegration ? 
                    window.goldGPTChartAutoIntegration.getIntegrationStatus() : 
                    'Auto-integration not available'
            };
        }
    };

    // Enhanced integration error handling
    if (window.goldGPTChartAutoIntegration) {
        const originalCreateAutoChart = window.goldGPTChartAutoIntegration.createAutoChart;
        
        window.goldGPTChartAutoIntegration.createAutoChart = async function(containerId, config) {
            try {
                return await originalCreateAutoChart.call(this, containerId, config);
            } catch (error) {
                console.error(`Auto-integration error for ${containerId}:`, error);
                
                // Show error overlay
                window.ChartErrorHandler.showChartError(containerId, {
                    code: error.code || 'AUTO_INTEGRATION_FAILED',
                    message: error.message || 'Auto-integration failed',
                    recoverable: true
                });
                
                throw error;
            }
        };
    }

    console.log('🛡️ Chart Error Enhancement System loaded');
    console.log('💡 Use ChartErrorHandler for advanced error management');
}

// Global error event listeners
document.addEventListener('unifiedChartFactory:chartError', (event) => {
    const { containerId, error } = event.detail;
    window.ChartErrorHandler.showChartError(containerId, error);
});

document.addEventListener('unifiedChartFactory:factoryError', (event) => {
    const { error } = event.detail;
    window.ChartErrorHandler.showGlobalError(
        '🚨 Chart System Error',
        'The chart system encountered an error. Some charts may not work properly.',
        error
    );
});

// Window error handler for unhandled chart-related errors
window.addEventListener('error', (event) => {
    if (event.error && (
        event.error.message.includes('chart') ||
        event.error.message.includes('Chart') ||
        event.error.stack?.includes('UnifiedChartManager') ||
        event.error.stack?.includes('LightweightCharts') ||
        event.error.stack?.includes('TradingView')
    )) {
        console.error('Unhandled chart error detected:', event.error);
        
        window.ChartErrorHandler.showGlobalError(
            '🚨 Unhandled Chart Error',
            'An unexpected chart error occurred. This may affect chart functionality.',
            {
                message: event.error.message,
                filename: event.filename,
                lineno: event.lineno,
                colno: event.colno
            }
        );
    }
});

console.log('🛡️ Chart Error Enhancement System initialized');
