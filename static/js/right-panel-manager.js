/**
 * Right Panel Components Manager
 * Handles Live Order Book, Fear/Greed Index, and ML AI Overview
 */

class RightPanelManager {
    constructor() {
        this.updateIntervals = {
            orderBook: null,
            fearGreed: null,
            mlOverview: null
        };
        this.componentIds = {
            orderBook: 'order-book',
            fearGreed: 'fear-greed-index',
            mlOverview: 'ml-overview'
        };
        this.eventCleanup = [];
        
        this.init();
    }
    
    init() {
        console.log('🎛️ Initializing Right Panel Components...');
        
        // Setup connection manager integration
        this.setupConnectionManagerIntegration();
        
        // Start loading all components
        this.loadOrderBook();
        this.loadFearGreedIndex();
        this.loadMLOverview();
        
        // Set up refresh intervals
        this.setupRefreshIntervals();
    }
    
    setupRefreshIntervals() {
        // Order book updates every 5 seconds
        this.updateIntervals.orderBook = setInterval(() => {
            this.loadOrderBook();
        }, 5000);
        
        // Fear & Greed updates every 15 minutes
        this.updateIntervals.fearGreed = setInterval(() => {
            this.loadFearGreedIndex();
        }, 15 * 60 * 1000);
        
        // ML Overview updates every 5 minutes
        this.updateIntervals.mlOverview = setInterval(() => {
            this.loadMLOverview();
        }, 5 * 60 * 1000);
    }
    
    async loadOrderBook() {
        try {
            const data = await window.connectionManager?.request('/api/order-book') || 
                          await fetch('/api/order-book').then(r => r.json());
            
            if (data.success) {
                this.renderOrderBook(data.data);
            } else {
                this.renderOrderBookError(data.error || 'Failed to load order book');
            }
        } catch (error) {
            console.error('Error loading order book:', error);
            
            // Handle error through connection manager
            if (window.connectionManager) {
                window.connectionManager.handleError(this.componentIds.orderBook, error);
            } else {
                this.renderOrderBookError('Connection error loading order book');
            }
        }
    }
    
    renderOrderBook(data) {
        const container = document.getElementById('order-book-content');
        if (!container) return;
        
        const { bids, asks, current_price, spread } = data;
        
        // Sort asks ascending (lowest first), bids descending (highest first)
        asks.sort((a, b) => a.price - b.price);
        bids.sort((a, b) => b.price - a.price);
        
        // Limit to top 8 levels each
        const topAsks = asks.slice(0, 8).reverse(); // Reverse so highest ask is at top
        const topBids = bids.slice(0, 8);
        
        const html = `
            <div class="order-book-section order-book-asks">
                <div class="order-book-section-title">Asks (Sell Orders)</div>
                ${topAsks.map(ask => `
                    <div class="order-book-row">
                        <div class="order-book-price">${ask.price}</div>
                        <div class="order-book-size">${ask.size}</div>
                        <div class="order-book-total">${ask.total.toFixed(0)}</div>
                    </div>
                `).join('')}
            </div>
            
            <div class="order-book-current-price">
                <div style="font-size: 16px;">${current_price}</div>
                <div class="order-book-spread">Spread: $${spread}</div>
            </div>
            
            <div class="order-book-section order-book-bids">
                <div class="order-book-section-title">Bids (Buy Orders)</div>
                ${topBids.map(bid => `
                    <div class="order-book-row">
                        <div class="order-book-price">${bid.price}</div>
                        <div class="order-book-size">${bid.size}</div>
                        <div class="order-book-total">${bid.total.toFixed(0)}</div>
                    </div>
                `).join('')}
            </div>
        `;
        
        container.innerHTML = html;
    }
    
    renderOrderBookError(error) {
        const container = document.getElementById('order-book-content');
        if (!container) return;
        
        container.innerHTML = `
            <div class="component-loading">
                <i class="fas fa-exclamation-triangle" style="color: var(--danger);"></i>
                <div class="component-loading-text">Failed to load order book</div>
                <div class="component-loading-subtext">${error}</div>
            </div>
        `;
    }
    
    async loadFearGreedIndex() {
        try {
            const response = await fetch('/api/fear-greed-index');
            const data = await response.json();
            
            if (data.success) {
                this.renderFearGreedIndex(data);
            } else {
                console.error('Fear & Greed API error:', data.error);
                this.renderFearGreedError(data.error);
            }
        } catch (error) {
            console.error('Failed to load Fear & Greed index:', error);
            this.renderFearGreedError(error?.message || 'Unknown error occurred');
        }
    }
    
    renderFearGreedIndex(data) {
        const container = document.getElementById('fear-greed-content');
        if (!container) return;
        
        const { index, level, color, trend, components, gold_factors } = data;
        
        // Calculate needle rotation (0-180 degrees for semicircle)
        const needleRotation = (index / 100) * 180 - 90;
        
        const html = `
            <div class="fear-greed-gauge">
                <div class="fear-greed-semicircle"></div>
                <div class="fear-greed-needle" style="transform: translateX(-50%) rotate(${needleRotation}deg);"></div>
            </div>
            
            <div class="fear-greed-value" style="color: ${color};">${index}</div>
            <div class="fear-greed-level" style="color: ${color};">${level}</div>
            
            <div class="fear-greed-description">
                Current market sentiment shows <strong>${level.toLowerCase()}</strong> conditions.
                Gold trend: <span style="color: ${this.getTrendColor(trend)};">${trend.toUpperCase()}</span>
            </div>
            
            <div class="fear-greed-components">
                <div style="font-size: 11px; font-weight: 600; color: var(--text-secondary); margin-bottom: 8px; text-transform: uppercase;">
                    Key Factors
                </div>
                ${Object.entries(gold_factors).slice(0, 4).map(([key, value]) => `
                    <div class="fear-greed-component">
                        <span class="fear-greed-component-name">${this.formatFactorName(key)}</span>
                        <span class="fear-greed-component-value" style="color: ${this.getFactorColor(value)};">${value}</span>
                    </div>
                `).join('')}
            </div>
            
            <div class="fear-greed-last-updated">
                Last updated: ${this.formatTime(data.last_updated)}
            </div>
        `;
        
        container.innerHTML = html;
    }
    
    renderFearGreedError(error) {
        const container = document.getElementById('fear-greed-content');
        if (!container) return;
        
        container.innerHTML = `
            <div class="component-loading">
                <i class="fas fa-exclamation-triangle" style="color: var(--danger);"></i>
                <div class="component-loading-text">Failed to load Fear & Greed</div>
                <div class="component-loading-subtext">${error}</div>
            </div>
        `;
    }
    
    async loadMLOverview() {
        try {
            const response = await fetch('/api/ml-gold-overview');
            const data = await response.json();
            
            if (data.success) {
                this.renderMLOverview(data);
            } else {
                console.error('ML Overview API error:', data.error);
                this.renderMLOverviewError(data.error);
            }
        } catch (error) {
            console.error('Failed to load ML overview:', error);
            this.renderMLOverviewError(error?.message || 'Unknown error occurred');
        }
    }
    
    renderMLOverview(data) {
        const container = document.getElementById('ml-overview-content');
        if (!container) return;
        
        const { overall_assessment, ml_signals, key_levels, market_conditions, ai_insights } = data;
        
        const assessmentClass = `ml-assessment-${overall_assessment.signal.toLowerCase()}`;
        const assessmentIcon = this.getAssessmentIcon(overall_assessment.signal);
        
        const html = `
            <div class="ml-assessment ${assessmentClass}">
                <div class="ml-assessment-icon">${assessmentIcon}</div>
                <div class="ml-assessment-details">
                    <h4>${overall_assessment.signal} OUTLOOK</h4>
                    <div class="ml-assessment-confidence">
                        Confidence: ${(overall_assessment.confidence * 100).toFixed(0)}% • ${overall_assessment.timeframe}
                    </div>
                </div>
            </div>
            
            <div class="ml-signals-grid">
                <div class="ml-signal">
                    <div class="ml-signal-name">LSTM Neural</div>
                    <div class="ml-signal-value" style="color: ${this.getSignalColor(ml_signals.lstm_prediction)}">${ml_signals.lstm_prediction}</div>
                    <div class="ml-signal-confidence">${(ml_signals.lstm_confidence * 100).toFixed(0)}%</div>
                </div>
                <div class="ml-signal">
                    <div class="ml-signal-name">Random Forest</div>
                    <div class="ml-signal-value" style="color: ${this.getSignalColor(ml_signals.random_forest)}">${ml_signals.random_forest}</div>
                    <div class="ml-signal-confidence">${(ml_signals.rf_confidence * 100).toFixed(0)}%</div>
                </div>
                <div class="ml-signal">
                    <div class="ml-signal-name">SVM Trend</div>
                    <div class="ml-signal-value" style="color: ${this.getSignalColor(ml_signals.svm_trend)}">${ml_signals.svm_trend}</div>
                    <div class="ml-signal-confidence">${(ml_signals.svm_confidence * 100).toFixed(0)}%</div>
                </div>
                <div class="ml-signal">
                    <div class="ml-signal-name">Momentum</div>
                    <div class="ml-signal-value" style="color: ${this.getSignalColor(market_conditions.momentum)}">${market_conditions.momentum}</div>
                    <div class="ml-signal-confidence">Live</div>
                </div>
            </div>
            
            <div class="ml-key-levels">
                <h5>Key Levels</h5>
                <div class="ml-levels-grid">
                    <div class="ml-level">
                        <span class="ml-level-name">Resistance</span>
                        <span class="ml-level-value">$${key_levels.resistance_1}</span>
                    </div>
                    <div class="ml-level">
                        <span class="ml-level-name">Support</span>
                        <span class="ml-level-value">$${key_levels.support_1}</span>
                    </div>
                    <div class="ml-level">
                        <span class="ml-level-name">Target</span>
                        <span class="ml-level-value">$${key_levels.target_price}</span>
                    </div>
                    <div class="ml-level">
                        <span class="ml-level-name">Stop Loss</span>
                        <span class="ml-level-value">$${key_levels.stop_loss}</span>
                    </div>
                </div>
            </div>
            
            <div class="ml-insights">
                <h5>AI Insights</h5>
                ${ai_insights.map(insight => `
                    <div class="ml-insight">${insight}</div>
                `).join('')}
            </div>
            
            <div class="ml-last-updated">
                Updated: ${this.formatTime(data.last_updated)} • Next: ${this.formatTime(data.next_update)}
            </div>
        `;
        
        container.innerHTML = html;
    }
    
    renderMLOverviewError(error) {
        const container = document.getElementById('ml-overview-content');
        if (!container) return;
        
        container.innerHTML = `
            <div class="component-loading">
                <i class="fas fa-exclamation-triangle" style="color: var(--danger);"></i>
                <div class="component-loading-text">Failed to load ML Overview</div>
                <div class="component-loading-subtext">${error}</div>
            </div>
        `;
    }
    
    // Helper methods
    getTrendColor(trend) {
        const colors = {
            'bullish': 'var(--success)',
            'bearish': 'var(--danger)',
            'neutral': 'var(--warning)'
        };
        return colors[trend] || 'var(--text-secondary)';
    }
    
    getFactorColor(value) {
        if (value >= 70) return 'var(--success)';
        if (value >= 50) return 'var(--warning)';
        if (value >= 30) return 'var(--text-secondary)';
        return 'var(--danger)';
    }
    
    formatFactorName(key) {
        return key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }
    
    getAssessmentIcon(signal) {
        const icons = {
            'BULLISH': '<i class="fas fa-arrow-trend-up"></i>',
            'BEARISH': '<i class="fas fa-arrow-trend-down"></i>',
            'NEUTRAL': '<i class="fas fa-minus"></i>'
        };
        return icons[signal] || '<i class="fas fa-question"></i>';
    }
    
    getSignalColor(signal) {
        const bullishSignals = ['BULLISH', 'BUY', 'UPTREND', 'STRONG_BULLISH', 'STRONG_UPTREND'];
        const bearishSignals = ['BEARISH', 'SELL', 'DOWNTREND', 'STRONG_BEARISH', 'STRONG_DOWNTREND'];
        
        if (bullishSignals.includes(signal)) return 'var(--success)';
        if (bearishSignals.includes(signal)) return 'var(--danger)';
        return 'var(--warning)';
    }
    
    formatTime(timestamp) {
        try {
            const date = new Date(timestamp);
            return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        } catch (error) {
            return 'Unknown';
        }
    }
    
    // Manual refresh methods
    refreshOrderBook() {
        this.loadOrderBook();
    }
    
    refreshFearGreed() {
        this.loadFearGreedIndex();
    }
    
    refreshMLOverview() {
        this.loadMLOverview();
    }
    
    /**
     * Setup connection manager integration
     */
    setupConnectionManagerIntegration() {
        if (!window.connectionManager) {
            console.warn('⚠️ Connection Manager not available for right panel manager');
            return;
        }
        
        // Add data-component attributes for loading/error states
        const containers = [
            { selector: '#order-book-content', id: this.componentIds.orderBook },
            { selector: '#fear-greed-content', id: this.componentIds.fearGreed },
            { selector: '#ml-overview-content', id: this.componentIds.mlOverview }
        ];
        
        containers.forEach(({ selector, id }) => {
            const container = document.querySelector(selector);
            if (container) {
                container.setAttribute('data-component', id);
            }
        });
        
        // Setup retry handlers
        const retryCleanup = window.connectionManager.on('retry', (data) => {
            const componentId = data.componentId;
            
            if (componentId === this.componentIds.orderBook) {
                console.log('🔄 Retrying order book load...');
                this.loadOrderBook();
            } else if (componentId === this.componentIds.fearGreed) {
                console.log('🔄 Retrying fear/greed index load...');
                this.loadFearGreedIndex();
            } else if (componentId === this.componentIds.mlOverview) {
                console.log('🔄 Retrying ML overview load...');
                this.loadMLOverview();
            }
        });
        this.eventCleanup.push(retryCleanup);
        
        // Setup WebSocket market data updates
        const marketDataCleanup = window.connectionManager.on('market_data_update', (data) => {
            console.log('📊 Real-time market data update received:', data);
            // Refresh components that depend on market data
            this.loadOrderBook();
            this.loadMLOverview();
        });
        this.eventCleanup.push(marketDataCleanup);
        
        console.log('✅ Connection Manager integration setup for right panel');
    }
    
    /**
     * Cleanup all event listeners and resources
     */
    cleanup() {
        console.log('🧹 Cleaning up Right Panel Manager...');
        
        // Clear all intervals
        Object.values(this.updateIntervals).forEach(interval => {
            if (interval) clearInterval(interval);
        });
        
        // Clean up event listeners
        this.eventCleanup.forEach(cleanup => cleanup());
        this.eventCleanup = [];
        
        // Clean up from connection manager
        if (window.connectionManager) {
            window.connectionManager.offContext(this);
        }
        
        console.log('✅ Right Panel Manager cleaned up');
    }
    
    // Cleanup
    destroy() {
        this.cleanup();
        console.log('🎛️ Right Panel Components destroyed');
    }
}

// Create global instance for component loader
window.rightPanelManager = new RightPanelManager();

// Add init method for component loader compatibility
window.rightPanelManager.init = function() {
    console.log('🎛️ Initializing Right Panel Components...');
    
    // Setup connection manager integration
    this.setupConnectionManagerIntegration();
    
    // Start loading all components
    this.loadOrderBook();
    this.loadFearGreedIndex();
    this.loadMLOverview();
    
    // Set up refresh intervals
    this.setupRefreshIntervals();
    
    return Promise.resolve();
};

// Auto-initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    console.log('🎛️ Right Panel Components initialized');
});

// Export for manual control
window.RightPanelManager = RightPanelManager;

console.log('🎛️ Right Panel Manager loaded successfully');
