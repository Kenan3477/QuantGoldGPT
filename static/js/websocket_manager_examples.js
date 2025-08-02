/**
 * WebSocketManager Usage Examples
 * Demonstrates how to use the WebSocketManager in your GoldGPT application
 */

// Example 1: Basic Setup and Connection
function basicWebSocketSetup() {
    // Get the default WebSocketManager instance
    const wsManager = window.WebSocketManagerFactory.getInstance('default', {
        debug: true,
        reconnectAttempts: 5,
        heartbeatInterval: 30000
    });

    // Subscribe to connection state changes
    wsManager.subscribe('connectionStateChanged', (data) => {
        console.log('Connection state changed:', data.state);
        updateConnectionUI(data.state);
    });

    // Subscribe to price updates
    wsManager.subscribe('priceUpdate', (data) => {
        console.log('New price:', data.price);
        updatePriceDisplay(data);
    });

    // Subscribe to AI analysis
    wsManager.subscribe('aiAnalysis', (data) => {
        console.log('AI Analysis:', data.signal, data.confidence);
        updateAIAnalysisDisplay(data);
    });

    // Subscribe to portfolio updates
    wsManager.subscribe('portfolioUpdate', (data) => {
        console.log('Portfolio update:', data.total_value);
        updatePortfolioDisplay(data);
    });

    // Connect to server
    wsManager.connect().then(() => {
        console.log('WebSocket connected successfully');
    }).catch(error => {
        console.error('WebSocket connection failed:', error);
    });
}

// Example 2: Advanced Event Handling with Unsubscribe
function advancedEventHandling() {
    const wsManager = window.WebSocketManager;

    // Subscribe with unsubscribe capability
    const unsubscribePrices = wsManager.subscribe('priceUpdate', (data) => {
        updateRealTimePriceChart(data);
    });

    const unsubscribeAI = wsManager.subscribe('aiAnalysis', (data) => {
        updateTradingSignals(data);
    });

    // Later, unsubscribe when component unmounts or page changes
    // unsubscribePrices();
    // unsubscribeAI();
}

// Example 3: Manual Request Functions
function manualDataRequests() {
    const wsManager = window.WebSocketManager;

    // Request immediate price update
    document.getElementById('refresh-price').addEventListener('click', () => {
        wsManager.requestPriceUpdate();
    });

    // Request AI analysis
    document.getElementById('refresh-ai').addEventListener('click', () => {
        wsManager.requestAIAnalysis();
    });

    // Request portfolio update
    document.getElementById('refresh-portfolio').addEventListener('click', () => {
        wsManager.requestPortfolioUpdate();
    });
}

// Example 4: Connection Status Monitoring
function setupConnectionMonitoring() {
    const wsManager = window.WebSocketManager;

    // Update UI based on connection state
    wsManager.subscribe('connectionStateChanged', (data) => {
        const statusElement = document.querySelector('.connection-status');
        if (statusElement) {
            statusElement.className = `connection-status ${data.state}`;
            statusElement.textContent = data.state.charAt(0).toUpperCase() + data.state.slice(1);
        }

        // Handle different states
        switch (data.state) {
            case 'connected':
                showNotification('Connected to real-time updates', 'success');
                break;
            case 'authenticated':
                showNotification('Authenticated successfully', 'success');
                break;
            case 'disconnected':
                showNotification('Disconnected from server', 'warning');
                break;
            case 'reconnecting':
                showNotification(`Reconnecting... (attempt ${data.attempt})`, 'info');
                break;
            case 'error':
                showNotification('Connection error occurred', 'error');
                break;
        }
    });

    // Monitor connection health
    setInterval(() => {
        const status = wsManager.getStatus();
        const stats = wsManager.getStats();
        
        // Update connection info display
        updateConnectionStats({
            state: status.state,
            uptime: stats.uptime,
            messagesReceived: stats.messagesReceived,
            messagesSent: stats.messagesSent,
            totalReconnects: stats.totalReconnects
        });
    }, 5000);
}

// Example 5: Error Handling and Recovery
function setupErrorHandling() {
    const wsManager = window.WebSocketManager;

    wsManager.subscribe('error', (data) => {
        console.error('WebSocket error:', data);
        
        // Handle different error types
        switch (data.type) {
            case 'connection':
                showNotification('Connection error: ' + data.error, 'error');
                break;
            case 'authentication':
                showNotification('Authentication failed', 'error');
                // Maybe redirect to login?
                break;
            case 'rate_limit':
                showNotification('Rate limit exceeded, please slow down', 'warning');
                break;
            default:
                showNotification('Unknown error occurred', 'error');
        }
    });

    // Handle authentication events
    wsManager.subscribe('authenticated', (data) => {
        console.log('Authentication successful');
        showNotification('Real-time updates enabled', 'success');
    });

    wsManager.subscribe('disconnected', (data) => {
        console.log('Disconnected:', data.reason);
        // Could show reconnection countdown
    });
}

// Example 6: Dashboard Integration
class GoldGPTDashboard {
    constructor() {
        this.wsManager = window.WebSocketManager;
        this.setupWebSocketIntegration();
    }

    setupWebSocketIntegration() {
        // Price updates
        this.wsManager.subscribe('priceUpdate', (data) => {
            this.updatePriceElements(data);
            this.updatePriceChart(data);
            this.checkPriceAlerts(data);
        });

        // AI analysis updates
        this.wsManager.subscribe('aiAnalysis', (data) => {
            this.updateAISignal(data.signal);
            this.updateConfidenceScore(data.confidence);
            this.updateRecommendations(data.recommendations);
        });

        // Portfolio updates
        this.wsManager.subscribe('portfolioUpdate', (data) => {
            this.updatePortfolioValue(data.total_value);
            this.updateDailyPnL(data.daily_pnl);
            this.updatePositions(data.positions);
        });

        // Connection status
        this.wsManager.subscribe('connectionStateChanged', (data) => {
            this.updateConnectionIndicator(data.state);
        });
    }

    updatePriceElements(data) {
        // Update price display
        const priceElement = document.querySelector('.gold-price');
        if (priceElement) {
            priceElement.textContent = `$${data.price}`;
            priceElement.classList.toggle('price-up', data.change > 0);
            priceElement.classList.toggle('price-down', data.change < 0);
        }

        // Update change indicator
        const changeElement = document.querySelector('.price-change');
        if (changeElement) {
            const sign = data.change >= 0 ? '+' : '';
            changeElement.textContent = `${sign}${data.change} (${data.change_percent}%)`;
            changeElement.className = `price-change ${data.change >= 0 ? 'positive' : 'negative'}`;
        }

        // Update timestamp
        const timestampElement = document.querySelector('.price-timestamp');
        if (timestampElement) {
            timestampElement.textContent = new Date(data.timestamp).toLocaleTimeString();
        }
    }

    updatePriceChart(data) {
        // Add to real-time chart
        if (window.goldPriceChart) {
            window.goldPriceChart.data.labels.push(new Date(data.timestamp));
            window.goldPriceChart.data.datasets[0].data.push(data.price);

            // Keep only last 100 points
            if (window.goldPriceChart.data.labels.length > 100) {
                window.goldPriceChart.data.labels.shift();
                window.goldPriceChart.data.datasets[0].data.shift();
            }

            window.goldPriceChart.update('none');
        }
    }

    updateAISignal(signal) {
        const signalElement = document.querySelector('.ai-signal');
        if (signalElement) {
            signalElement.textContent = signal.toUpperCase();
            signalElement.className = `ai-signal ${signal.toLowerCase()}`;
        }
    }

    updateConfidenceScore(confidence) {
        const confidenceElement = document.querySelector('.ai-confidence');
        if (confidenceElement) {
            confidenceElement.textContent = `${(confidence * 100).toFixed(1)}%`;
        }

        // Update confidence bar
        const confidenceBar = document.querySelector('.confidence-bar');
        if (confidenceBar) {
            confidenceBar.style.width = `${confidence * 100}%`;
        }
    }

    updateConnectionIndicator(state) {
        const indicator = document.querySelector('.connection-status');
        if (indicator) {
            indicator.className = `connection-status ${state}`;
            indicator.textContent = state.charAt(0).toUpperCase() + state.slice(1);
        }
    }

    // Manual refresh methods
    refreshPrice() {
        this.wsManager.requestPriceUpdate();
    }

    refreshAI() {
        this.wsManager.requestAIAnalysis();
    }

    refreshPortfolio() {
        this.wsManager.requestPortfolioUpdate();
    }
}

// Example 7: Connection Health Monitoring
function setupHealthMonitoring() {
    const wsManager = window.WebSocketManager;
    
    // Create health monitor
    const healthMonitor = {
        checkInterval: 30000, // 30 seconds
        timer: null,
        
        start() {
            this.timer = setInterval(() => {
                const status = wsManager.getStatus();
                const stats = wsManager.getStats();
                
                // Log health info
                console.log('WebSocket Health Check:', {
                    state: status.state,
                    connected: status.connected,
                    uptime: stats.uptime,
                    errors: stats.totalErrors,
                    reconnects: stats.totalReconnects
                });
                
                // Check for issues
                if (stats.totalErrors > 10) {
                    console.warn('High error count detected');
                }
                
                if (stats.totalReconnects > 5) {
                    console.warn('Frequent reconnections detected');
                }
                
                // Update health display
                this.updateHealthDisplay(status, stats);
                
            }, this.checkInterval);
        },
        
        stop() {
            if (this.timer) {
                clearInterval(this.timer);
                this.timer = null;
            }
        },
        
        updateHealthDisplay(status, stats) {
            const healthElement = document.querySelector('.connection-health');
            if (healthElement) {
                const uptime = Math.floor(stats.uptime / 1000);
                healthElement.innerHTML = `
                    <div>Status: ${status.state}</div>
                    <div>Uptime: ${uptime}s</div>
                    <div>Messages: ↓${stats.messagesReceived} ↑${stats.messagesSent}</div>
                    <div>Errors: ${stats.totalErrors}</div>
                `;
            }
        }
    };
    
    // Start monitoring when connected
    wsManager.subscribe('connectionStateChanged', (data) => {
        if (data.state === 'authenticated') {
            healthMonitor.start();
        } else if (data.state === 'disconnected') {
            healthMonitor.stop();
        }
    });
}

// Helper functions for UI updates
function updateConnectionUI(state) {
    document.body.classList.remove('ws-connected', 'ws-disconnected', 'ws-reconnecting');
    document.body.classList.add(`ws-${state}`);
}

function showNotification(message, type = 'info') {
    // Simple notification system
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.textContent = message;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.remove();
    }, 5000);
}

function updateConnectionStats(stats) {
    const statsElement = document.querySelector('.connection-stats');
    if (statsElement) {
        statsElement.innerHTML = `
            <div>State: ${stats.state}</div>
            <div>Uptime: ${Math.floor(stats.uptime / 1000)}s</div>
            <div>Received: ${stats.messagesReceived}</div>
            <div>Sent: ${stats.messagesSent}</div>
            <div>Reconnects: ${stats.totalReconnects}</div>
        `;
    }
}

// Auto-initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 WebSocketManager examples loaded');
    
    // Uncomment to auto-start basic setup
    // basicWebSocketSetup();
    
    // Initialize dashboard if elements exist
    if (document.querySelector('.gold-price')) {
        const dashboard = new GoldGPTDashboard();
        window.goldGPTDashboard = dashboard;
        console.log('📊 GoldGPT Dashboard WebSocket integration ready');
    }
    
    // Setup health monitoring
    setupHealthMonitoring();
    
    console.log('💡 Use WebSocketManager examples in browser console:');
    console.log('   - basicWebSocketSetup()');
    console.log('   - manualDataRequests()');
    console.log('   - setupConnectionMonitoring()');
});
