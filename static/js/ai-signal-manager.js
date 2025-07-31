/**
 * AI Trade Signal Generator - Frontend Integration
 * Sophisticated signal display for high-ROI trading recommendations
 */

class AITradeSignalManager {
    constructor() {
        this.signals = [];
        this.openSignals = [];
        this.stats = {};
        this.isLoading = false;
        this.updateInterval = null;
        
        this.initializeSignalDisplay();
        this.startAutoUpdate();
    }

    initializeSignalDisplay() {
        // Create signal container in portfolio section
        this.createSignalContainer();
        this.bindEventListeners();
    }

    createSignalContainer() {
        // Find portfolio section or create one
        let portfolioSection = document.querySelector('.portfolio-section');
        
        if (!portfolioSection) {
            // Create portfolio section if it doesn't exist
            portfolioSection = document.createElement('div');
            portfolioSection.className = 'portfolio-section';
            portfolioSection.innerHTML = `
                <div class="portfolio-header">
                    <h2><i class="fas fa-briefcase"></i> Portfolio & AI Signals</h2>
                </div>
            `;
            
            // Insert into main dashboard
            const mainDashboard = document.querySelector('.main-dashboard') || document.body;
            mainDashboard.appendChild(portfolioSection);
        }

        // Create AI signals container
        const signalContainer = document.createElement('div');
        signalContainer.id = 'ai-signal-container';
        signalContainer.className = 'ai-signal-container';
        signalContainer.innerHTML = this.getSignalHTML();
        
        portfolioSection.appendChild(signalContainer);
    }

    getSignalHTML() {
        return `
            <div class="ai-signals-panel">
                <!-- Signal Header -->
                <div class="signal-header">
                    <div class="signal-title">
                        <h3><i class="fas fa-robot"></i> AI Trade Signals</h3>
                        <span class="signal-subtitle">High-ROI Trading Recommendations</span>
                    </div>
                    <div class="signal-controls">
                        <button id="generate-signal-btn" class="btn-generate">
                            <i class="fas fa-cog"></i> Generate Signal
                        </button>
                        <button id="refresh-signals-btn" class="btn-refresh">
                            <i class="fas fa-sync"></i> Refresh
                        </button>
                    </div>
                </div>

                <!-- Latest Signal Card -->
                <div id="latest-signal-card" class="latest-signal-card">
                    <div class="signal-loading">
                        <i class="fas fa-spinner fa-spin"></i> Loading latest signal...
                    </div>
                </div>

                <!-- Performance Stats -->
                <div class="signal-stats-grid">
                    <div class="stat-card win-rate">
                        <div class="stat-icon"><i class="fas fa-trophy"></i></div>
                        <div class="stat-content">
                            <div class="stat-value" id="win-rate-value">--</div>
                            <div class="stat-label">Win Rate</div>
                        </div>
                    </div>
                    <div class="stat-card total-signals">
                        <div class="stat-icon"><i class="fas fa-chart-line"></i></div>
                        <div class="stat-content">
                            <div class="stat-value" id="total-signals-value">--</div>
                            <div class="stat-label">Total Signals</div>
                        </div>
                    </div>
                    <div class="stat-card profit-factor">
                        <div class="stat-icon"><i class="fas fa-dollar-sign"></i></div>
                        <div class="stat-content">
                            <div class="stat-value" id="profit-factor-value">--</div>
                            <div class="stat-label">Profit Factor</div>
                        </div>
                    </div>
                    <div class="stat-card total-return">
                        <div class="stat-icon"><i class="fas fa-percentage"></i></div>
                        <div class="stat-content">
                            <div class="stat-value" id="total-return-value">--</div>
                            <div class="stat-label">Total Return</div>
                        </div>
                    </div>
                </div>

                <!-- Open Signals -->
                <div class="open-signals-section">
                    <h4><i class="fas fa-bullseye"></i> Open Signals</h4>
                    <div id="open-signals-list" class="open-signals-list">
                        <!-- Open signals will be populated here -->
                    </div>
                </div>
            </div>
        `;
    }

    bindEventListeners() {
        // Generate new signal
        document.addEventListener('click', (e) => {
            if (e.target.closest('#generate-signal-btn')) {
                this.generateNewSignal();
            }
            
            if (e.target.closest('#refresh-signals-btn')) {
                this.refreshAllData();
            }
        });
    }

    async generateNewSignal() {
        const btn = document.getElementById('generate-signal-btn');
        if (!btn || this.isLoading) return;

        try {
            this.setLoadingState(true);
            btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating...';
            
            const response = await fetch('/api/signals/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const data = await response.json();
            
            if (data.success && data.signal) {
                this.displayLatestSignal(data.signal);
                this.showNotification('New signal generated successfully!', 'success');
                
                // Refresh open signals and stats
                await this.loadOpenSignals();
                await this.loadStats();
            } else {
                this.showNotification(data.message || 'No new signal generated', 'info');
            }
            
        } catch (error) {
            console.error('Error generating signal:', error);
            this.showNotification('Error generating signal', 'error');
        } finally {
            this.setLoadingState(false);
            btn.innerHTML = '<i class="fas fa-cog"></i> Generate Signal';
        }
    }

    async refreshAllData() {
        try {
            this.setLoadingState(true);
            
            // Load all data in parallel
            await Promise.all([
                this.loadPortfolioData(),
                this.loadOpenSignals(),
                this.loadStats()
            ]);
            
            this.showNotification('Data refreshed successfully', 'success');
            
        } catch (error) {
            console.error('Error refreshing data:', error);
            this.showNotification('Error refreshing data', 'error');
        } finally {
            this.setLoadingState(false);
        }
    }

    async loadPortfolioData() {
        try {
            const response = await fetch('/api/signals/portfolio');
            const data = await response.json();
            
            if (data.success) {
                const portfolio = data.portfolio;
                
                // Display latest signal
                if (portfolio.latest_signal) {
                    this.displayLatestSignal(portfolio.latest_signal);
                }
                
                // Update stats
                if (portfolio.statistics) {
                    this.updateStatsDisplay(portfolio.statistics);
                }
                
                // Update open signals
                if (portfolio.open_signals) {
                    this.displayOpenSignals(portfolio.open_signals);
                }
            }
            
        } catch (error) {
            console.error('Error loading portfolio data:', error);
        }
    }

    async loadOpenSignals() {
        try {
            const response = await fetch('/api/signals/open');
            const data = await response.json();
            
            if (data.success) {
                this.openSignals = data.signals;
                this.displayOpenSignals(this.openSignals);
            }
            
        } catch (error) {
            console.error('Error loading open signals:', error);
        }
    }

    async loadStats() {
        try {
            const response = await fetch('/api/signals/stats');
            const data = await response.json();
            
            if (data.success) {
                this.stats = data.stats;
                this.updateStatsDisplay(this.stats);
            }
            
        } catch (error) {
            console.error('Error loading stats:', error);
        }
    }

    displayLatestSignal(signal) {
        const container = document.getElementById('latest-signal-card');
        if (!container) return;

        const signalTypeClass = signal.type === 'buy' ? 'signal-buy' : 'signal-sell';
        const signalIcon = signal.type === 'buy' ? 'fa-arrow-up' : 'fa-arrow-down';
        const riskColor = this.getRiskColor(signal.confidence);
        
        container.innerHTML = `
            <div class="latest-signal ${signalTypeClass}">
                <div class="signal-main">
                    <div class="signal-type">
                        <i class="fas ${signalIcon}"></i>
                        <span class="signal-action">${signal.type.toUpperCase()}</span>
                        <span class="signal-confidence" style="color: ${riskColor}">
                            ${signal.confidence}% confidence
                        </span>
                    </div>
                    
                    <div class="signal-price-info">
                        <div class="price-row">
                            <span class="price-label">Entry:</span>
                            <span class="price-value">$${signal.entry_price}</span>
                        </div>
                        <div class="price-row">
                            <span class="price-label">Target:</span>
                            <span class="price-value target">$${signal.target_price}</span>
                        </div>
                        <div class="price-row">
                            <span class="price-label">Stop Loss:</span>
                            <span class="price-value stop">$${signal.stop_loss}</span>
                        </div>
                        <div class="price-row">
                            <span class="price-label">Risk/Reward:</span>
                            <span class="price-value rr">1:${signal.risk_reward}</span>
                        </div>
                    </div>
                </div>
                
                <div class="signal-summary">
                    <p class="signal-reasoning">${signal.summary}</p>
                    <div class="signal-meta">
                        <span class="signal-timeframe">${signal.timeframe}</span>
                        <span class="signal-timestamp">${this.formatTimestamp(signal.timestamp)}</span>
                    </div>
                </div>
            </div>
        `;
    }

    displayOpenSignals(signals) {
        const container = document.getElementById('open-signals-list');
        if (!container) return;

        if (!signals || signals.length === 0) {
            container.innerHTML = `
                <div class="no-signals">
                    <i class="fas fa-chart-line"></i>
                    <p>No open signals</p>
                    <small>Generate your first signal to get started</small>
                </div>
            `;
            return;
        }

        container.innerHTML = signals.map(signal => `
            <div class="open-signal-item ${signal.signal_type}">
                <div class="signal-header-row">
                    <span class="signal-type-badge ${signal.signal_type}">
                        ${signal.signal_type.toUpperCase()}
                    </span>
                    <span class="signal-confidence">${signal.confidence}%</span>
                </div>
                
                <div class="signal-prices">
                    <div class="price-item">
                        <label>Entry</label>
                        <value>$${signal.entry_price}</value>
                    </div>
                    <div class="price-item">
                        <label>Target</label>
                        <value>$${signal.target_price}</value>
                    </div>
                    <div class="price-item">
                        <label>Stop</label>
                        <value>$${signal.stop_loss}</value>
                    </div>
                </div>
                
                <div class="signal-summary-text">
                    ${signal.analysis_summary}
                </div>
                
                <div class="signal-footer">
                    <span class="timeframe">${signal.timeframe}</span>
                    <span class="timestamp">${this.formatTimestamp(signal.timestamp)}</span>
                </div>
            </div>
        `).join('');
    }

    updateStatsDisplay(stats) {
        // Update win rate
        const winRateEl = document.getElementById('win-rate-value');
        if (winRateEl) {
            winRateEl.textContent = `${stats.win_rate.toFixed(1)}%`;
            winRateEl.className = 'stat-value ' + (stats.win_rate > 60 ? 'positive' : stats.win_rate > 40 ? 'neutral' : 'negative');
        }

        // Update total signals
        const totalSignalsEl = document.getElementById('total-signals-value');
        if (totalSignalsEl) {
            totalSignalsEl.textContent = stats.total_signals;
        }

        // Update profit factor
        const profitFactorEl = document.getElementById('profit-factor-value');
        if (profitFactorEl) {
            profitFactorEl.textContent = stats.profit_factor.toFixed(2);
            profitFactorEl.className = 'stat-value ' + (stats.profit_factor > 1.5 ? 'positive' : stats.profit_factor > 1 ? 'neutral' : 'negative');
        }

        // Update total return
        const totalReturnEl = document.getElementById('total-return-value');
        if (totalReturnEl) {
            totalReturnEl.textContent = `${stats.total_return > 0 ? '+' : ''}${stats.total_return.toFixed(2)}%`;
            totalReturnEl.className = 'stat-value ' + (stats.total_return > 0 ? 'positive' : stats.total_return < 0 ? 'negative' : 'neutral');
        }
    }

    getRiskColor(confidence) {
        if (confidence >= 80) return '#10b981'; // Green
        if (confidence >= 60) return '#f59e0b'; // Yellow
        return '#ef4444'; // Red
    }

    formatTimestamp(timestamp) {
        const date = new Date(timestamp);
        return date.toLocaleString('en-US', {
            hour: '2-digit',
            minute: '2-digit',
            month: 'short',
            day: 'numeric'
        });
    }

    setLoadingState(loading) {
        this.isLoading = loading;
        
        // Add/remove loading class to container
        const container = document.getElementById('ai-signal-container');
        if (container) {
            if (loading) {
                container.classList.add('loading');
            } else {
                container.classList.remove('loading');
            }
        }
    }

    showNotification(message, type = 'info') {
        // Use existing notification system if available
        if (window.notificationManager) {
            window.notificationManager.show(message, type);
        } else {
            // Fallback notification
            console.log(`${type.toUpperCase()}: ${message}`);
            
            // Simple toast notification
            const toast = document.createElement('div');
            toast.className = `toast toast-${type}`;
            toast.textContent = message;
            toast.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                padding: 10px 20px;
                background: ${type === 'success' ? '#10b981' : type === 'error' ? '#ef4444' : '#3b82f6'};
                color: white;
                border-radius: 5px;
                z-index: 10000;
                opacity: 0;
                transition: opacity 0.3s;
            `;
            
            document.body.appendChild(toast);
            
            // Fade in
            setTimeout(() => toast.style.opacity = '1', 100);
            
            // Fade out and remove
            setTimeout(() => {
                toast.style.opacity = '0';
                setTimeout(() => document.body.removeChild(toast), 300);
            }, 3000);
        }
    }

    startAutoUpdate() {
        // Update signals every 5 minutes
        this.updateInterval = setInterval(() => {
            this.refreshAllData();
        }, 5 * 60 * 1000);
    }

    stopAutoUpdate() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
    }

    // Initialize on page load
    init() {
        // Load initial data
        this.loadPortfolioData();
        
        console.log('🤖 AI Trade Signal Manager initialized');
    }
}

// Global instance
window.aiTradeSignalManager = new AITradeSignalManager();

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.aiTradeSignalManager.init();
});

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AITradeSignalManager;
}
