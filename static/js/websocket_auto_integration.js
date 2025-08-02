/**
 * WebSocketManager Auto-Integration for GoldGPT Dashboard
 * This script automatically sets up WebSocket connectivity on dashboard pages
 */

// Auto-initialization when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 GoldGPT WebSocket Auto-Integration starting...');

    // Wait for all scripts to load
    setTimeout(() => {
        initializeWebSocketManager();
    }, 1000);
});

function initializeWebSocketManager() {
    try {
        // Check if WebSocketManager is available
        if (typeof window.WebSocketManagerFactory === 'undefined') {
            console.warn('WebSocketManager not available, falling back to basic mode');
            initializeBasicWebSocket();
            return;
        }

        console.log('📡 Initializing enhanced WebSocketManager...');

        // Get or create WebSocket manager instance
        const wsManager = window.WebSocketManagerFactory.getInstance('goldgpt-dashboard', {
            debug: true,
            reconnectAttempts: 10,
            heartbeatInterval: 30000,
            reconnectDelay: 2000,
            maxReconnectDelay: 30000
        });

        // Make it globally accessible
        window.wsManager = wsManager;

        // Set up event subscriptions
        setupWebSocketEvents(wsManager);

        // Connect to server
        wsManager.connect().then(() => {
            console.log('✅ WebSocket connected successfully');
            showConnectionStatus('connected');
        }).catch(error => {
            console.error('❌ WebSocket connection failed:', error);
            showConnectionStatus('failed');
        });

    } catch (error) {
        console.error('❌ Error initializing WebSocketManager:', error);
        initializeBasicWebSocket();
    }
}

function setupWebSocketEvents(wsManager) {
    // Connection state changes
    wsManager.subscribe('connectionStateChanged', (data) => {
        console.log(`🔄 Connection state: ${data.state}`);
        updateConnectionIndicator(data.state);
        
        // Show user notifications
        switch (data.state) {
            case 'connected':
                showNotification('Connected to real-time updates', 'success');
                break;
            case 'authenticated':
                showNotification('Enhanced real-time features enabled', 'success');
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

    // Price updates - Enhanced 2-second updates
    wsManager.subscribe('priceUpdate', (data) => {
        console.log('💰 Price update:', data.price);
        updatePriceDisplay(data);
        updatePriceChart(data);
    });

    // AI analysis updates
    wsManager.subscribe('aiAnalysis', (data) => {
        console.log('🤖 AI Analysis:', data.signal);
        updateAIAnalysisDisplay(data);
    });

    // Portfolio updates
    wsManager.subscribe('portfolioUpdate', (data) => {
        console.log('💼 Portfolio update:', data.total_value);
        updatePortfolioDisplay(data);
    });

    // Error handling
    wsManager.subscribe('error', (data) => {
        console.error('❌ WebSocket error:', data);
        showNotification('Connection error: ' + (data.error || 'Unknown error'), 'error');
    });
}

function updateConnectionIndicator(state) {
    // Update connection status indicator
    const indicators = document.querySelectorAll('.connection-status');
    indicators.forEach(indicator => {
        indicator.className = `connection-status ${state}`;
        
        let displayText = state.charAt(0).toUpperCase() + state.slice(1);
        if (state === 'authenticated') {
            displayText = 'Live';
        } else if (state === 'reconnecting') {
            displayText = 'Reconnecting...';
        }
        
        indicator.textContent = displayText;
    });

    // Update status dots
    const statusDots = document.querySelectorAll('.status-dot');
    statusDots.forEach(dot => {
        dot.className = `status-dot ${state}`;
    });
}

function updatePriceDisplay(data) {
    try {
        // Update main price display
        const priceElements = document.querySelectorAll('.gold-price, .price-value, [data-price="gold"]');
        priceElements.forEach(element => {
            if (element) {
                element.textContent = `$${data.price.toFixed(2)}`;
                
                // Add price movement animation
                element.classList.remove('price-up', 'price-down');
                if (data.change > 0) {
                    element.classList.add('price-up');
                } else if (data.change < 0) {
                    element.classList.add('price-down');
                }
            }
        });

        // Update price change indicators
        const changeElements = document.querySelectorAll('.price-change, .change-value, [data-change="gold"]');
        changeElements.forEach(element => {
            if (element) {
                const sign = data.change >= 0 ? '+' : '';
                element.textContent = `${sign}${data.change.toFixed(2)} (${data.change_percent.toFixed(2)}%)`;
                element.className = `price-change ${data.change >= 0 ? 'positive' : 'negative'}`;
            }
        });

        // Update timestamps
        const timestampElements = document.querySelectorAll('.price-timestamp, .last-updated, [data-timestamp]');
        timestampElements.forEach(element => {
            if (element) {
                element.textContent = new Date(data.timestamp).toLocaleTimeString();
            }
        });

        // Update high/low if available
        if (data.high) {
            const highElements = document.querySelectorAll('.price-high, [data-high]');
            highElements.forEach(element => {
                if (element) element.textContent = `$${data.high.toFixed(2)}`;
            });
        }

        if (data.low) {
            const lowElements = document.querySelectorAll('.price-low, [data-low]');
            lowElements.forEach(element => {
                if (element) element.textContent = `$${data.low.toFixed(2)}`;
            });
        }

    } catch (error) {
        console.error('Error updating price display:', error);
    }
}

function updatePriceChart(data) {
    try {
        // Update Chart.js charts
        if (window.goldPriceChart && window.goldPriceChart.data) {
            const chart = window.goldPriceChart;
            const time = new Date(data.timestamp);
            
            chart.data.labels.push(time);
            chart.data.datasets[0].data.push(data.price);
            
            // Keep only last 50 points for performance
            if (chart.data.labels.length > 50) {
                chart.data.labels.shift();
                chart.data.datasets[0].data.shift();
            }
            
            chart.update('none'); // No animation for real-time updates
        }

        // Update TradingView widget if available
        if (window.tradingViewWidget && window.tradingViewWidget.setSymbol) {
            // TradingView handles its own updates
        }

        // Update ApexCharts if available
        if (window.priceChart && window.priceChart.updateSeries) {
            window.priceChart.appendData([{
                data: [data.price]
            }]);
        }

    } catch (error) {
        console.error('Error updating price chart:', error);
    }
}

function updateAIAnalysisDisplay(data) {
    try {
        // Update AI signal
        const signalElements = document.querySelectorAll('.ai-signal, .signal-value, [data-ai-signal]');
        signalElements.forEach(element => {
            if (element) {
                element.textContent = data.signal.toUpperCase();
                element.className = `ai-signal ${data.signal.toLowerCase()}`;
            }
        });

        // Update confidence score
        const confidenceElements = document.querySelectorAll('.ai-confidence, .confidence-value, [data-confidence]');
        confidenceElements.forEach(element => {
            if (element) {
                element.textContent = `${(data.confidence * 100).toFixed(1)}%`;
            }
        });

        // Update confidence bars
        const confidenceBars = document.querySelectorAll('.confidence-bar, .confidence-progress');
        confidenceBars.forEach(bar => {
            if (bar) {
                bar.style.width = `${data.confidence * 100}%`;
                
                // Color based on confidence level
                if (data.confidence > 0.8) {
                    bar.style.backgroundColor = '#10b981'; // High confidence - green
                } else if (data.confidence > 0.6) {
                    bar.style.backgroundColor = '#f59e0b'; // Medium confidence - yellow
                } else {
                    bar.style.backgroundColor = '#ef4444'; // Low confidence - red
                }
            }
        });

        // Update recommendations if available
        if (data.recommendations && data.recommendations.length > 0) {
            const recommendationElements = document.querySelectorAll('.ai-recommendations, [data-recommendations]');
            recommendationElements.forEach(element => {
                if (element) {
                    element.innerHTML = data.recommendations.map(rec => 
                        `<div class="recommendation ${rec.type}">${rec.message}</div>`
                    ).join('');
                }
            });
        }

    } catch (error) {
        console.error('Error updating AI analysis display:', error);
    }
}

function updatePortfolioDisplay(data) {
    try {
        // Update total portfolio value
        const valueElements = document.querySelectorAll('.portfolio-value, .total-value, [data-portfolio-value]');
        valueElements.forEach(element => {
            if (element) {
                element.textContent = `$${data.total_value.toLocaleString()}`;
            }
        });

        // Update daily P&L
        const pnlElements = document.querySelectorAll('.portfolio-pnl, .daily-pnl, [data-daily-pnl]');
        pnlElements.forEach(element => {
            if (element) {
                const sign = data.daily_pnl >= 0 ? '+' : '';
                element.textContent = `${sign}$${data.daily_pnl.toLocaleString()}`;
                element.className = `portfolio-pnl ${data.daily_pnl >= 0 ? 'positive' : 'negative'}`;
            }
        });

        // Update positions if available
        if (data.positions && Array.isArray(data.positions)) {
            const positionsElements = document.querySelectorAll('.positions-list, [data-positions]');
            positionsElements.forEach(element => {
                if (element) {
                    element.innerHTML = data.positions.map(pos => `
                        <div class="position-item">
                            <span class="symbol">${pos.symbol}</span>
                            <span class="quantity">${pos.quantity}</span>
                            <span class="value">$${pos.value.toFixed(2)}</span>
                        </div>
                    `).join('');
                }
            });
        }

    } catch (error) {
        console.error('Error updating portfolio display:', error);
    }
}

function showConnectionStatus(status) {
    const statusText = {
        'connected': 'Connected to real-time data',
        'failed': 'Connection failed',
        'disconnected': 'Disconnected'
    };

    console.log(`🔗 Connection status: ${statusText[status] || status}`);
}

function showNotification(message, type = 'info') {
    // Try to use existing notification system
    if (typeof window.showNotification === 'function') {
        window.showNotification(message, type);
        return;
    }

    // Fallback notification system
    console.log(`📢 [${type.toUpperCase()}] ${message}`);
    
    // Create simple toast notification
    const notification = document.createElement('div');
    notification.className = `ws-notification ws-${type}`;
    notification.textContent = message;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 12px 16px;
        border-radius: 6px;
        color: white;
        font-size: 14px;
        z-index: 10000;
        max-width: 300px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        background: ${type === 'success' ? '#10b981' : type === 'error' ? '#ef4444' : type === 'warning' ? '#f59e0b' : '#3b82f6'};
    `;
    
    document.body.appendChild(notification);
    
    // Remove after 5 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.parentNode.removeChild(notification);
        }
    }, 5000);
}

function initializeBasicWebSocket() {
    console.log('📡 Initializing basic Socket.IO connection...');
    
    try {
        if (typeof io === 'undefined') {
            console.warn('Socket.IO not available');
            return;
        }

        const socket = io();
        
        socket.on('connect', () => {
            console.log('✅ Basic Socket.IO connected');
            showConnectionStatus('connected');
            updateConnectionIndicator('connected');
        });

        socket.on('disconnect', () => {
            console.log('❌ Basic Socket.IO disconnected');
            showConnectionStatus('disconnected');
            updateConnectionIndicator('disconnected');
        });

        socket.on('price_update', updatePriceDisplay);
        socket.on('ai_analysis', updateAIAnalysisDisplay);
        socket.on('portfolio_update', updatePortfolioDisplay);

        // Make basic socket globally available
        window.basicSocket = socket;

    } catch (error) {
        console.error('❌ Error initializing basic Socket.IO:', error);
    }
}

// Utility functions for manual requests
window.requestPriceUpdate = function() {
    if (window.wsManager && window.wsManager.isReady()) {
        window.wsManager.requestPriceUpdate();
    } else if (window.basicSocket) {
        window.basicSocket.emit('request_price_update');
    }
};

window.requestAIAnalysis = function() {
    if (window.wsManager && window.wsManager.isReady()) {
        window.wsManager.requestAIAnalysis();
    } else if (window.basicSocket) {
        window.basicSocket.emit('request_ai_analysis');
    }
};

window.requestPortfolioUpdate = function() {
    if (window.wsManager && window.wsManager.isReady()) {
        window.wsManager.requestPortfolioUpdate();
    } else if (window.basicSocket) {
        window.basicSocket.emit('request_portfolio_update');
    }
};

// Debug function to check WebSocket status
window.checkWebSocketStatus = function() {
    if (window.wsManager) {
        const status = window.wsManager.getStatus();
        const stats = window.wsManager.getStats();
        console.log('📊 WebSocket Status:', status);
        console.log('📈 WebSocket Stats:', stats);
        return { status, stats };
    } else if (window.basicSocket) {
        console.log('📡 Basic Socket.IO connected:', window.basicSocket.connected);
        return { connected: window.basicSocket.connected };
    } else {
        console.log('❌ No WebSocket connection available');
        return { connected: false };
    }
};

console.log('🚀 GoldGPT WebSocket Auto-Integration loaded');
console.log('💡 Available functions: requestPriceUpdate(), requestAIAnalysis(), requestPortfolioUpdate(), checkWebSocketStatus()');
