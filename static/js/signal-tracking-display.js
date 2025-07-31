/**
 * Signal Tracking Dashboard Integration
 * Displays live P&L tracking, win/loss monitoring, and ML learning progress
 */

class SignalTrackingDisplay {
    constructor() {
        this.updateInterval = 10000; // 10 seconds
        this.trackingActive = false;
        this.trackingTimer = null;
        this.init();
    }

    init() {
        this.createTrackingUI();
        this.startTracking();
    }

    createTrackingUI() {
        // Find or create the tracking container
        let trackingContainer = document.getElementById('signal-tracking-container');
        if (!trackingContainer) {
            trackingContainer = document.createElement('div');
            trackingContainer.id = 'signal-tracking-container';
            trackingContainer.className = 'signal-tracking-dashboard';
            
            // Insert after the main dashboard content
            const mainContent = document.querySelector('.dashboard-content') || document.body;
            mainContent.appendChild(trackingContainer);
        }

        trackingContainer.innerHTML = `
            <div class="tracking-header">
                <h3><i class="fas fa-chart-line"></i> Live Signal Tracking</h3>
                <div class="tracking-status">
                    <span class="status-dot" id="tracking-status-dot"></span>
                    <span id="tracking-status-text">Initializing...</span>
                </div>
            </div>
            
            <div class="tracking-grid">
                <!-- Active Signals Section -->
                <div class="tracking-panel active-signals-panel">
                    <div class="panel-header">
                        <h4><i class="fas fa-broadcast-tower"></i> Active Signals</h4>
                        <div class="signal-counters">
                            <span class="counter winning" id="winning-signals">0</span>
                            <span class="counter losing" id="losing-signals">0</span>
                        </div>
                    </div>
                    <div class="signals-list" id="active-signals-list">
                        <div class="loading-message">Loading active signals...</div>
                    </div>
                </div>

                <!-- Performance Insights Section -->
                <div class="tracking-panel performance-panel">
                    <div class="panel-header">
                        <h4><i class="fas fa-trophy"></i> Performance Insights</h4>
                    </div>
                    <div class="performance-stats" id="performance-stats">
                        <div class="stat-row">
                            <span class="stat-label">Win Rate:</span>
                            <span class="stat-value" id="win-rate">0%</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Total Signals:</span>
                            <span class="stat-value" id="total-signals">0</span>
                        </div>
                        <div class="stat-row">
                            <span class="stat-label">Avg Profit:</span>
                            <span class="stat-value" id="avg-profit">$0.00</span>
                        </div>
                    </div>
                    <div class="recommendations" id="strategy-recommendations"></div>
                </div>

                <!-- ML Learning Section -->
                <div class="tracking-panel learning-panel">
                    <div class="panel-header">
                        <h4><i class="fas fa-brain"></i> ML Learning Progress</h4>
                    </div>
                    <div class="learning-status" id="learning-status">
                        <div class="learning-progress">
                            <div class="progress-bar">
                                <div class="progress-fill" id="learning-progress-fill"></div>
                            </div>
                            <div class="progress-text" id="learning-progress-text">Initializing...</div>
                        </div>
                    </div>
                    <div class="top-factors" id="top-factors"></div>
                </div>
            </div>
        `;

        this.injectTrackingStyles();
    }

    injectTrackingStyles() {
        if (document.getElementById('signal-tracking-styles')) return;

        const styles = document.createElement('style');
        styles.id = 'signal-tracking-styles';
        styles.textContent = `
            .signal-tracking-dashboard {
                background: rgba(20, 25, 35, 0.95);
                border-radius: 12px;
                padding: 20px;
                margin: 20px 0;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
                border: 1px solid rgba(255, 215, 0, 0.1);
            }

            .tracking-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 25px;
                padding-bottom: 15px;
                border-bottom: 1px solid rgba(255, 215, 0, 0.2);
            }

            .tracking-header h3 {
                color: #FFD700;
                font-size: 1.4em;
                font-weight: 600;
                margin: 0;
            }

            .tracking-header i {
                margin-right: 8px;
            }

            .tracking-status {
                display: flex;
                align-items: center;
                gap: 8px;
            }

            .status-dot {
                width: 12px;
                height: 12px;
                border-radius: 50%;
                background: #4CAF50;
                animation: pulse 2s infinite;
            }

            .status-dot.inactive {
                background: #f44336;
                animation: none;
            }

            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.5; }
                100% { opacity: 1; }
            }

            .tracking-grid {
                display: grid;
                grid-template-columns: 1fr 1fr 1fr;
                gap: 20px;
            }

            @media (max-width: 1200px) {
                .tracking-grid {
                    grid-template-columns: 1fr;
                    gap: 15px;
                }
            }

            .tracking-panel {
                background: rgba(30, 35, 45, 0.8);
                border-radius: 8px;
                padding: 15px;
                border: 1px solid rgba(255, 215, 0, 0.1);
            }

            .panel-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
                padding-bottom: 10px;
                border-bottom: 1px solid rgba(255, 215, 0, 0.1);
            }

            .panel-header h4 {
                color: #FFD700;
                font-size: 1.1em;
                margin: 0;
            }

            .signal-counters {
                display: flex;
                gap: 10px;
            }

            .counter {
                padding: 4px 8px;
                border-radius: 12px;
                font-weight: bold;
                font-size: 0.9em;
            }

            .counter.winning {
                background: rgba(76, 175, 80, 0.2);
                color: #4CAF50;
                border: 1px solid #4CAF50;
            }

            .counter.losing {
                background: rgba(244, 67, 54, 0.2);
                color: #f44336;
                border: 1px solid #f44336;
            }

            .signals-list {
                max-height: 200px;
                overflow-y: auto;
            }

            .signal-item {
                background: rgba(40, 45, 55, 0.6);
                border-radius: 6px;
                padding: 10px;
                margin-bottom: 8px;
                border-left: 4px solid #FFD700;
            }

            .signal-item.winning {
                border-left-color: #4CAF50;
                background: rgba(76, 175, 80, 0.1);
            }

            .signal-item.losing {
                border-left-color: #f44336;
                background: rgba(244, 67, 54, 0.1);
            }

            .signal-info {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 5px;
            }

            .signal-type {
                font-weight: bold;
                color: #FFD700;
                text-transform: uppercase;
            }

            .signal-pnl {
                font-weight: bold;
                font-size: 0.9em;
            }

            .signal-pnl.positive {
                color: #4CAF50;
            }

            .signal-pnl.negative {
                color: #f44336;
            }

            .signal-details {
                font-size: 0.85em;
                color: #ccc;
            }

            .performance-stats {
                margin-bottom: 15px;
            }

            .stat-row {
                display: flex;
                justify-content: space-between;
                margin-bottom: 8px;
                padding: 5px 0;
            }

            .stat-label {
                color: #ccc;
                font-size: 0.9em;
            }

            .stat-value {
                color: #FFD700;
                font-weight: bold;
            }

            .recommendations {
                background: rgba(255, 215, 0, 0.05);
                border-radius: 6px;
                padding: 10px;
                border: 1px solid rgba(255, 215, 0, 0.1);
            }

            .recommendation {
                font-size: 0.85em;
                color: #FFD700;
                margin-bottom: 5px;
                padding-left: 15px;
                position: relative;
            }

            .recommendation:before {
                content: "💡";
                position: absolute;
                left: 0;
            }

            .learning-progress {
                margin-bottom: 15px;
            }

            .progress-bar {
                background: rgba(40, 45, 55, 0.6);
                height: 20px;
                border-radius: 10px;
                overflow: hidden;
                margin-bottom: 8px;
            }

            .progress-fill {
                background: linear-gradient(90deg, #FFD700, #FFA500);
                height: 100%;
                width: 0%;
                transition: width 0.3s ease;
            }

            .progress-text {
                font-size: 0.9em;
                color: #ccc;
                text-align: center;
            }

            .top-factors {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
            }

            .factor-tag {
                background: rgba(255, 215, 0, 0.1);
                color: #FFD700;
                padding: 4px 8px;
                border-radius: 12px;
                font-size: 0.8em;
                border: 1px solid rgba(255, 215, 0, 0.2);
            }

            .loading-message {
                text-align: center;
                color: #ccc;
                padding: 20px;
                font-style: italic;
            }

            .error-message {
                text-align: center;
                color: #f44336;
                padding: 20px;
                font-style: italic;
            }
        `;
        document.head.appendChild(styles);
    }

    startTracking() {
        if (this.trackingActive) return;
        
        this.trackingActive = true;
        this.updateTrackingData();
        
        this.trackingTimer = setInterval(() => {
            this.updateTrackingData();
        }, this.updateInterval);

        console.log('🎯 Signal tracking display started');
    }

    stopTracking() {
        this.trackingActive = false;
        if (this.trackingTimer) {
            clearInterval(this.trackingTimer);
            this.trackingTimer = null;
        }
        
        const statusDot = document.getElementById('tracking-status-dot');
        const statusText = document.getElementById('tracking-status-text');
        
        if (statusDot) statusDot.classList.add('inactive');
        if (statusText) statusText.textContent = 'Inactive';
    }

    async updateTrackingData() {
        try {
            // Update status indicator
            const statusDot = document.getElementById('tracking-status-dot');
            const statusText = document.getElementById('tracking-status-text');
            
            if (statusDot) statusDot.classList.remove('inactive');
            if (statusText) statusText.textContent = 'Active';

            // Fetch all tracking data in parallel
            const [statusData, performanceData, learningData] = await Promise.all([
                this.fetchData('/api/signal-tracking/status'),
                this.fetchData('/api/signal-tracking/performance-insights'),
                this.fetchData('/api/signal-tracking/learning-progress')
            ]);

            // Update displays
            if (statusData.success) {
                this.updateActiveSignals(statusData);
            }

            if (performanceData.success) {
                this.updatePerformanceInsights(performanceData.insights);
            }

            if (learningData.success) {
                this.updateLearningProgress(learningData.learning);
            }

        } catch (error) {
            console.error('Error updating tracking data:', error);
            this.showError('Failed to update tracking data');
        }
    }

    async fetchData(endpoint) {
        const response = await fetch(endpoint);
        return await response.json();
    }

    updateActiveSignals(data) {
        const winningEl = document.getElementById('winning-signals');
        const losingEl = document.getElementById('losing-signals');
        const listEl = document.getElementById('active-signals-list');

        if (winningEl) winningEl.textContent = data.winning_signals || 0;
        if (losingEl) losingEl.textContent = data.losing_signals || 0;

        if (listEl) {
            if (data.active_signals && data.active_signals.length > 0) {
                listEl.innerHTML = data.active_signals.map(signal => this.createSignalElement(signal)).join('');
            } else {
                listEl.innerHTML = '<div class="loading-message">No active signals</div>';
            }
        }
    }

    createSignalElement(signal) {
        const pnlClass = signal.current_pnl_pct > 0 ? 'positive' : signal.current_pnl_pct < 0 ? 'negative' : '';
        const itemClass = signal.status.toLowerCase();
        
        return `
            <div class="signal-item ${itemClass}">
                <div class="signal-info">
                    <span class="signal-type">${signal.type}</span>
                    <span class="signal-pnl ${pnlClass}">${signal.current_pnl_pct > 0 ? '+' : ''}${signal.current_pnl_pct}%</span>
                </div>
                <div class="signal-details">
                    Entry: $${signal.entry_price} | Current: $${signal.current_price} | TP: $${signal.target_price} | SL: $${signal.stop_loss}
                </div>
                <div class="signal-details">
                    Confidence: ${signal.confidence}% | Age: ${this.getSignalAge(signal.timestamp)}
                </div>
            </div>
        `;
    }

    updatePerformanceInsights(insights) {
        const winRateEl = document.getElementById('win-rate');
        const totalSignalsEl = document.getElementById('total-signals');
        const avgProfitEl = document.getElementById('avg-profit');
        const recommendationsEl = document.getElementById('strategy-recommendations');

        if (winRateEl) winRateEl.textContent = `${insights.win_rate || 0}%`;
        if (totalSignalsEl) totalSignalsEl.textContent = insights.total_signals || 0;
        if (avgProfitEl) avgProfitEl.textContent = `$${(insights.avg_profit || 0).toFixed(2)}`;

        if (recommendationsEl && insights.recommendations) {
            recommendationsEl.innerHTML = insights.recommendations
                .map(rec => `<div class="recommendation">${rec}</div>`)
                .join('');
        }
    }

    updateLearningProgress(learning) {
        const progressFillEl = document.getElementById('learning-progress-fill');
        const progressTextEl = document.getElementById('learning-progress-text');
        const topFactorsEl = document.getElementById('top-factors');

        if (learning.learning_enabled) {
            const progress = learning.model_ready ? 100 : Math.min((learning.total_learning_samples / 10) * 100, 100);
            
            if (progressFillEl) progressFillEl.style.width = `${progress}%`;
            if (progressTextEl) progressTextEl.textContent = learning.learning_status || 'Learning...';

            if (topFactorsEl && learning.top_performing_factors) {
                topFactorsEl.innerHTML = learning.top_performing_factors
                    .map(factor => `<div class="factor-tag">${factor.factor}: ${factor.score}</div>`)
                    .join('');
            }
        } else {
            if (progressTextEl) progressTextEl.textContent = 'Learning disabled';
            if (topFactorsEl) topFactorsEl.innerHTML = '<div class="loading-message">Learning not active</div>';
        }
    }

    getSignalAge(timestamp) {
        const now = new Date();
        const signalTime = new Date(timestamp);
        const diffMs = now - signalTime;
        const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
        const diffMinutes = Math.floor((diffMs % (1000 * 60 * 60)) / (1000 * 60));
        
        if (diffHours > 0) {
            return `${diffHours}h ${diffMinutes}m`;
        }
        return `${diffMinutes}m`;
    }

    showError(message) {
        const containers = ['active-signals-list', 'performance-stats', 'learning-status'];
        containers.forEach(containerId => {
            const el = document.getElementById(containerId);
            if (el) {
                el.innerHTML = `<div class="error-message">${message}</div>`;
            }
        });
    }

    // Force update for testing
    forceUpdate() {
        this.updateTrackingData();
    }

    // Manual signal check for testing
    async forceSignalCheck() {
        try {
            const response = await fetch('/api/signal-tracking/force-check', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            const data = await response.json();
            
            if (data.success) {
                console.log('✅ Forced signal check completed');
                setTimeout(() => this.updateTrackingData(), 1000);
            } else {
                console.error('❌ Force check failed:', data.error);
            }
        } catch (error) {
            console.error('Error forcing signal check:', error);
        }
    }
}

// Initialize tracking display when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    if (typeof window.signalTrackingDisplay === 'undefined') {
        window.signalTrackingDisplay = new SignalTrackingDisplay();
        console.log('🚀 Signal Tracking Display initialized');
    }
});

// Global access for debugging
window.SignalTrackingDisplay = SignalTrackingDisplay;
