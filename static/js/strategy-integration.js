/**
 * Integrated Strategy Dashboard Client
 * Connects strategy engine to main GoldGPT dashboard
 */

class StrategyIntegration {
    constructor() {
        this.isInitialized = false;
        this.currentSignal = null;
        this.performanceData = null;
        this.init();
    }

    init() {
        // Add strategy navigation to main dashboard
        this.addStrategyNavigation();
        
        // Initialize strategy widgets
        this.initializeStrategyWidgets();
        
        // Setup WebSocket integration
        this.setupWebSocketHandlers();
        
        // Auto-update strategy data
        this.startAutoUpdate();
        
        this.isInitialized = true;
        console.log('✅ Strategy Integration initialized');
    }

    addStrategyNavigation() {
        // Add strategy link to main navigation
        const navElement = document.querySelector('.nav-links, .navbar-nav, .main-nav');
        if (navElement) {
            const strategyNavItem = document.createElement('li');
            strategyNavItem.className = 'nav-item';
            strategyNavItem.innerHTML = `
                <a class="nav-link" href="/strategy/" target="_blank">
                    <i class="fas fa-brain me-2"></i>
                    <span>Strategy Engine</span>
                </a>
            `;
            navElement.appendChild(strategyNavItem);
        }

        // Add strategy quick access to dashboard
        const dashboardContent = document.querySelector('.dashboard-content, .main-content');
        if (dashboardContent) {
            const strategyWidget = document.createElement('div');
            strategyWidget.className = 'strategy-widget-container';
            strategyWidget.innerHTML = this.createStrategyWidget();
            
            // Insert after existing widgets
            const existingWidgets = dashboardContent.querySelector('.row, .dashboard-row');
            if (existingWidgets) {
                existingWidgets.appendChild(strategyWidget);
            } else {
                dashboardContent.appendChild(strategyWidget);
            }
        }
    }

    createStrategyWidget() {
        return `
            <div class="col-lg-6 col-xl-4 mb-4">
                <div class="card strategy-quick-card h-100">
                    <div class="card-header d-flex justify-content-between align-items-center">
                        <h6 class="card-title mb-0">
                            <i class="fas fa-brain me-2 text-warning"></i>
                            Strategy Engine
                        </h6>
                        <div class="strategy-status">
                            <span class="status-dot status-loading"></span>
                        </div>
                    </div>
                    <div class="card-body">
                        <div id="currentStrategySignal" class="current-signal">
                            <div class="text-center text-muted">
                                <i class="fas fa-chart-line fa-2x mb-2"></i>
                                <p class="small">Loading strategy data...</p>
                            </div>
                        </div>
                        
                        <div class="strategy-metrics mt-3">
                            <div class="row text-center">
                                <div class="col-6">
                                    <div class="metric-small">
                                        <div class="metric-value" id="strategyWinRate">--</div>
                                        <div class="metric-label">Win Rate</div>
                                    </div>
                                </div>
                                <div class="col-6">
                                    <div class="metric-small">
                                        <div class="metric-value" id="strategyReturn">--</div>
                                        <div class="metric-label">Return</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="card-footer">
                        <div class="d-flex gap-2">
                            <button class="btn btn-sm btn-warning flex-fill" onclick="strategyIntegration.generateQuickSignal()">
                                <i class="fas fa-signal me-1"></i>Generate
                            </button>
                            <button class="btn btn-sm btn-outline-secondary flex-fill" onclick="window.open('/strategy/', '_blank')">
                                <i class="fas fa-external-link-alt me-1"></i>Full Dashboard
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    initializeStrategyWidgets() {
        // Add strategy styles
        this.addStrategyStyles();
        
        // Load initial data
        this.loadStrategyStatus();
        this.loadRecentSignal();
        this.loadPerformanceMetrics();
    }

    addStrategyStyles() {
        const style = document.createElement('style');
        style.textContent = `
            .strategy-quick-card {
                border: 1px solid rgba(255, 193, 7, 0.3);
                background: linear-gradient(135deg, rgba(255, 193, 7, 0.05) 0%, rgba(0, 0, 0, 0.1) 100%);
                transition: all 0.3s ease;
            }

            .strategy-quick-card:hover {
                border-color: rgba(255, 193, 7, 0.6);
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(255, 193, 7, 0.2);
            }

            .status-dot {
                width: 8px;
                height: 8px;
                border-radius: 50%;
                display: inline-block;
            }

            .status-loading {
                background: #6c757d;
                animation: pulse 1.5s ease-in-out infinite;
            }

            .status-active {
                background: #28a745;
            }

            .status-error {
                background: #dc3545;
            }

            .current-signal {
                min-height: 80px;
                display: flex;
                align-items: center;
                justify-content: center;
                border-radius: 8px;
                background: rgba(255, 255, 255, 0.02);
                border: 1px solid rgba(255, 255, 255, 0.1);
            }

            .signal-active {
                border-color: #28a745;
                background: rgba(40, 167, 69, 0.1);
            }

            .signal-buy {
                border-left: 4px solid #28a745;
            }

            .signal-sell {
                border-left: 4px solid #dc3545;
            }

            .metric-small {
                padding: 0.5rem;
            }

            .metric-small .metric-value {
                font-size: 1.2rem;
                font-weight: bold;
                color: #ffc107;
            }

            .metric-small .metric-label {
                font-size: 0.75rem;
                color: #6c757d;
                text-transform: uppercase;
            }

            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.5; }
            }

            .strategy-widget-container {
                animation: fadeIn 0.5s ease-in-out;
            }

            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
        `;
        document.head.appendChild(style);
    }

    async loadStrategyStatus() {
        try {
            const statusDot = document.querySelector('.status-dot');
            if (statusDot) {
                statusDot.className = 'status-dot status-active';
            }
        } catch (error) {
            console.error('Error loading strategy status:', error);
            const statusDot = document.querySelector('.status-dot');
            if (statusDot) {
                statusDot.className = 'status-dot status-error';
            }
        }
    }

    async loadRecentSignal() {
        try {
            const response = await fetch('/strategy/api/signals/recent?limit=1');
            const data = await response.json();
            
            if (data.success && data.signals.length > 0) {
                this.displayQuickSignal(data.signals[0]);
            } else {
                this.displayNoSignal();
            }
        } catch (error) {
            console.error('Error loading recent signal:', error);
            this.displayNoSignal();
        }
    }

    displayQuickSignal(signal) {
        const signalElement = document.getElementById('currentStrategySignal');
        if (!signalElement) return;

        const signalClass = signal.signal_type.toLowerCase();
        const confidence = (signal.confidence * 100).toFixed(0);
        
        signalElement.className = `current-signal signal-active signal-${signalClass}`;
        signalElement.innerHTML = `
            <div class="text-center">
                <div class="d-flex align-items-center justify-content-center mb-2">
                    <span class="badge ${signalClass === 'buy' ? 'bg-success' : signalClass === 'sell' ? 'bg-danger' : 'bg-warning'} me-2">
                        ${signal.signal_type}
                    </span>
                    <small class="text-muted">${confidence}% conf.</small>
                </div>
                <div class="small">
                    <div><strong>$${signal.entry_price}</strong></div>
                    <div class="text-muted">${signal.strategy_name}</div>
                </div>
            </div>
        `;

        this.currentSignal = signal;
    }

    displayNoSignal() {
        const signalElement = document.getElementById('currentStrategySignal');
        if (!signalElement) return;

        signalElement.className = 'current-signal';
        signalElement.innerHTML = `
            <div class="text-center text-muted">
                <i class="fas fa-chart-line fa-2x mb-2"></i>
                <p class="small">No active signals</p>
            </div>
        `;
    }

    async loadPerformanceMetrics() {
        try {
            const response = await fetch('/strategy/api/performance');
            const data = await response.json();
            
            if (data.success && data.performance.summary) {
                const summary = data.performance.summary;
                
                document.getElementById('strategyWinRate').textContent = `${summary.win_rate?.toFixed(0) || '--'}%`;
                document.getElementById('strategyReturn').textContent = `${summary.avg_return > 0 ? '+' : ''}${summary.avg_return?.toFixed(1) || '--'}%`;
            }
        } catch (error) {
            console.error('Error loading performance metrics:', error);
        }
    }

    async generateQuickSignal() {
        const button = event.target;
        const originalHTML = button.innerHTML;
        
        button.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>Generating...';
        button.disabled = true;

        try {
            const response = await fetch('/strategy/api/signals/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    symbol: 'XAU',
                    timeframe: '1h'
                })
            });

            const data = await response.json();
            
            if (data.success) {
                this.displayQuickSignal(data.signal);
                this.showNotification('New signal generated!', 'success');
                
                // Update main dashboard if it has signal displays
                this.broadcastSignalUpdate(data.signal);
            } else {
                this.showNotification(data.message || 'No signal generated', 'warning');
            }
        } catch (error) {
            console.error('Error generating signal:', error);
            this.showNotification('Error generating signal', 'error');
        } finally {
            button.innerHTML = originalHTML;
            button.disabled = false;
        }
    }

    broadcastSignalUpdate(signal) {
        // Broadcast to other components
        const event = new CustomEvent('strategySignalUpdate', {
            detail: { signal }
        });
        document.dispatchEvent(event);

        // Update trade signal manager if available
        if (window.tradeSignalManager) {
            window.tradeSignalManager.addSignal(signal);
        }

        // Update position manager if available
        if (window.positionManager) {
            window.positionManager.notifyNewSignal(signal);
        }
    }

    setupWebSocketHandlers() {
        // Listen for WebSocket updates if available
        if (window.socket) {
            window.socket.on('strategy_signal_update', (data) => {
                if (data.signal) {
                    this.displayQuickSignal(data.signal);
                }
            });

            window.socket.on('strategy_performance_update', (data) => {
                if (data.metrics) {
                    this.updatePerformanceDisplay(data.metrics);
                }
            });
        }
    }

    updatePerformanceDisplay(metrics) {
        if (metrics.win_rate !== undefined) {
            document.getElementById('strategyWinRate').textContent = `${metrics.win_rate.toFixed(0)}%`;
        }
        if (metrics.avg_return !== undefined) {
            document.getElementById('strategyReturn').textContent = `${metrics.avg_return > 0 ? '+' : ''}${metrics.avg_return.toFixed(1)}%`;
        }
    }

    startAutoUpdate() {
        // Update strategy data every 30 seconds
        setInterval(() => {
            this.loadRecentSignal();
            this.loadPerformanceMetrics();
        }, 30000);
    }

    showNotification(message, type) {
        // Integration with existing notification system
        if (window.notificationManager) {
            window.notificationManager.show(message, type);
        } else if (window.showNotification) {
            window.showNotification(message, type);
        } else {
            console.log(`${type.toUpperCase()}: ${message}`);
        }
    }

    // Public API for other components
    getCurrentSignal() {
        return this.currentSignal;
    }

    getPerformanceData() {
        return this.performanceData;
    }

    openFullDashboard() {
        window.open('/strategy/', '_blank');
    }

    // Integration with existing dashboard systems
    integrateWithPortfolio() {
        // Add strategy signals to portfolio calculations
        document.addEventListener('portfolioUpdate', (event) => {
            if (this.currentSignal) {
                event.detail.strategySignal = this.currentSignal;
            }
        });
    }

    integrateWithAlerts() {
        // Add strategy alerts to main alert system
        document.addEventListener('strategySignalUpdate', (event) => {
            const signal = event.detail.signal;
            if (signal.confidence > 0.7) {
                this.showNotification(
                    `High confidence ${signal.signal_type} signal generated!`,
                    'info'
                );
            }
        });
    }
}

// Auto-initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    if (!window.strategyIntegration) {
        window.strategyIntegration = new StrategyIntegration();
        
        // Setup integrations
        setTimeout(() => {
            window.strategyIntegration.integrateWithPortfolio();
            window.strategyIntegration.integrateWithAlerts();
        }, 1000);
    }
});

// Export for global access
window.StrategyIntegration = StrategyIntegration;
