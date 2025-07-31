/**
 * Trade Signal Manager - Handles AI trade signal display and portfolio integration
 */

class TradeSignalManager {
    constructor() {
        this.signalContainer = null;
        this.signalUpdateInterval = 60000; // Check for new signals every minute
        this.openSignals = [];
        this.currentSignal = null;
        this.stats = {};
        this.initialized = false;
        this.currentGoldPrice = 3350.70; // Default fallback
    }

    init(containerId = 'trade-signals-container') {
        if (this.initialized) return;
        
        console.log('🎯 Initializing Trade Signal Manager...');
        
        this.signalContainer = document.getElementById(containerId);
        if (!this.signalContainer) {
            console.error(`Container #${containerId} not found for trade signals`);
            return;
        }
        
        // Create UI structure
        this.createUIStructure();
        
        // Load initial signals
        this.fetchSignals();
        
        // Setup refresh interval
        this.refreshInterval = setInterval(() => this.fetchSignals(), this.signalUpdateInterval);
        
        this.initialized = true;
        console.log('✅ Trade Signal Manager initialized successfully');
    }

    destroy() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
        }
        this.initialized = false;
    }
    
    createUIStructure() {
        this.signalContainer.innerHTML = `
            <div class="trade-signals-header">
                <div class="signals-title">
                    <h3><i class="fas fa-robot"></i> AI Trade Signal Generator</h3>
                    <div class="signals-subtitle">High-ROI Trading Signals</div>
                </div>
                <div class="signal-stats" id="signal-stats-display">
                    <div class="stat-item">
                        <span class="stat-label">Win Rate</span>
                        <span class="stat-value" id="win-rate-stat">-</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Profit Factor</span>
                        <span class="stat-value" id="profit-factor-stat">-</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Total Signals</span>
                        <span class="stat-value" id="total-signals-stat">-</span>
                    </div>
                </div>
            </div>
            
            <div class="current-signal-panel" id="current-signal-panel">
                <div class="loading-indicator">
                    <i class="fas fa-spinner fa-spin"></i> 
                    <span>Analyzing market conditions...</span>
                </div>
            </div>
            
            <div class="open-signals-container" id="open-signals-container">
                <!-- Open signals will be inserted here -->
            </div>
            
            <div class="signal-actions-bar">
                <button id="generate-new-signal" class="btn btn-primary">
                    <i class="fas fa-magic"></i> Generate New Signal
                </button>
                <button id="refresh-signals" class="btn btn-outline">
                    <i class="fas fa-sync-alt"></i> Refresh
                </button>
                <button id="toggle-signal-history" class="btn btn-outline">
                    <i class="fas fa-history"></i> Signal History
                </button>
            </div>
            
            <div class="signal-history-container" id="signal-history-container" style="display:none;">
                <!-- Signal history will be inserted here -->
            </div>
        `;
        
        // Add event listeners
        this.setupEventListeners();
    }

    setupEventListeners() {
        // History toggle
        document.getElementById('toggle-signal-history').addEventListener('click', () => this.toggleSignalHistory());
        
        // Refresh signals
        document.getElementById('refresh-signals').addEventListener('click', () => this.fetchSignals());
        
        // Generate new signal
        document.getElementById('generate-new-signal').addEventListener('click', () => this.generateNewSignal());
    }
    
    toggleSignalHistory() {
        const historyContainer = document.getElementById('signal-history-container');
        const toggleButton = document.getElementById('toggle-signal-history');
        
        if (historyContainer.style.display === 'none') {
            historyContainer.style.display = 'block';
            toggleButton.innerHTML = '<i class="fas fa-chevron-up"></i> Hide History';
            this.fetchSignalHistory();
        } else {
            historyContainer.style.display = 'none';
            toggleButton.innerHTML = '<i class="fas fa-history"></i> Signal History';
        }
    }

    async generateNewSignal() {
        const button = document.getElementById('generate-new-signal');
        const originalText = button.innerHTML;
        
        // Show loading state
        button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating...';
        button.disabled = true;
        
        try {
            const response = await fetch('/api/signals/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            if (!response.ok) {
                throw new Error(`Failed to generate signal: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.success) {
                // Refresh all signals to show the new one
                await this.fetchSignals();
                this.showNotification('✅ New signal generated successfully!', 'success');
            } else {
                throw new Error(data.error || 'Failed to generate signal');
            }
            
        } catch (error) {
            console.error('Error generating signal:', error);
            this.showNotification('❌ Failed to generate signal', 'error');
        } finally {
            // Restore button state
            button.innerHTML = originalText;
            button.disabled = false;
        }
    }
    
    async fetchSignals() {
        try {
            console.log('📡 Fetching trade signals...');
            const response = await fetch('/api/trade-signals');
            
            if (!response.ok) {
                throw new Error(`Failed to fetch signals: ${response.status}`);
            }
            
            const data = await response.json();
            
            if (data.success) {
                // Update properties
                this.currentSignal = data.current_signal;
                this.openSignals = data.open_signals || [];
                this.stats = data.statistics || {};
                
                console.log('✅ Signals fetched successfully:', {
                    currentSignal: this.currentSignal?.signal_type,
                    openSignals: this.openSignals.length,
                    stats: this.stats
                });
                
                // Update UI
                this.updateSignalDisplay();
                this.updateStatsDisplay();
                this.updateOpenSignalsDisplay();
            } else {
                throw new Error(data.error || 'Unknown error');
            }
            
        } catch (error) {
            console.error('Error fetching trade signals:', error);
            this.showErrorState(error.message);
        }
    }
    
    async fetchSignalHistory() {
        const historyContainer = document.getElementById('signal-history-container');
        
        historyContainer.innerHTML = `
            <div class="signal-history-header">
                <h4><i class="fas fa-history"></i> Signal History</h4>
            </div>
            <div class="loading-indicator">
                <i class="fas fa-spinner fa-spin"></i> Loading history...
            </div>
        `;
        
        try {
            // Try to fetch real history (this endpoint may not exist yet)
            const response = await fetch('/api/signals/history');
            
            if (response.ok) {
                const data = await response.json();
                this.displaySignalHistory(data.signals || []);
            } else {
                // Fallback to mock data
                this.displayMockSignalHistory();
            }
        } catch (error) {
            console.log('Using mock signal history');
            this.displayMockSignalHistory();
        }
    }

    displaySignalHistory(signals) {
        const historyContainer = document.getElementById('signal-history-container');
        
        if (!signals || signals.length === 0) {
            historyContainer.innerHTML = `
                <div class="signal-history-header">
                    <h4><i class="fas fa-history"></i> Signal History</h4>
                </div>
                <div class="no-history">
                    <i class="fas fa-info-circle"></i>
                    <p>No signal history available yet</p>
                </div>
            `;
            return;
        }

        let html = `
            <div class="signal-history-header">
                <h4><i class="fas fa-history"></i> Signal History</h4>
            </div>
            <div class="signal-history-list">
        `;

        signals.forEach(signal => {
            const isProfit = signal.profit_loss > 0;
            const statusClass = isProfit ? 'success' : 'failure';
            const profitDisplay = isProfit ? `+${signal.profit_loss.toFixed(2)}%` : `${signal.profit_loss.toFixed(2)}%`;

            html += `
                <div class="signal-history-item ${statusClass}">
                    <div class="signal-type">${signal.signal_type.toUpperCase()}</div>
                    <div class="signal-details">
                        <div>Entry: $${parseFloat(signal.entry_price).toFixed(2)}</div>
                        <div>Exit: $${parseFloat(signal.exit_price).toFixed(2)}</div>
                        <div class="${isProfit ? 'profit' : 'loss'}">${profitDisplay}</div>
                    </div>
                    <div class="signal-date">${this.formatTimestamp(signal.exit_timestamp)}</div>
                </div>
            `;
        });

        html += '</div>';
        historyContainer.innerHTML = html;
    }

    displayMockSignalHistory() {
        const historyContainer = document.getElementById('signal-history-container');
        
        historyContainer.innerHTML = `
            <div class="signal-history-header">
                <h4><i class="fas fa-history"></i> Signal History</h4>
                <div class="history-note">Sample signals for demonstration</div>
            </div>
            <div class="signal-history-list">
                <div class="signal-history-item success">
                    <div class="signal-type">BUY</div>
                    <div class="signal-details">
                        <div>Entry: $3325.40</div>
                        <div>Exit: $3345.80</div>
                        <div class="profit">+0.61%</div>
                    </div>
                    <div class="signal-date">July 18, 2025</div>
                </div>
                <div class="signal-history-item failure">
                    <div class="signal-type">SELL</div>
                    <div class="signal-details">
                        <div>Entry: $3340.15</div>
                        <div>Exit: $3350.25</div>
                        <div class="loss">-0.30%</div>
                    </div>
                    <div class="signal-date">July 17, 2025</div>
                </div>
                <div class="signal-history-item success">
                    <div class="signal-type">BUY</div>
                    <div class="signal-details">
                        <div>Entry: $3310.75</div>
                        <div>Exit: $3338.90</div>
                        <div class="profit">+0.85%</div>
                    </div>
                    <div class="signal-date">July 15, 2025</div>
                </div>
                <div class="signal-history-item success">
                    <div class="signal-type">BUY</div>
                    <div class="signal-details">
                        <div>Entry: $3298.20</div>
                        <div>Exit: $3321.45</div>
                        <div class="profit">+0.70%</div>
                    </div>
                    <div class="signal-date">July 12, 2025</div>
                </div>
            </div>
        `;
    }
    
    updateSignalDisplay() {
        const panel = document.getElementById('current-signal-panel');
        
        if (!this.currentSignal) {
            panel.innerHTML = `
                <div class="no-signal">
                    <i class="fas fa-search"></i>
                    <div class="no-signal-content">
                        <h4>No Active Signal</h4>
                        <p>The AI is waiting for optimal market conditions to generate a high-probability trade signal.</p>
                        <p class="wait-text">Market analysis is ongoing...</p>
                    </div>
                    <button class="btn btn-primary" onclick="tradeSignalManager.generateNewSignal()">
                        <i class="fas fa-magic"></i> Generate Signal
                    </button>
                </div>
            `;
            return;
        }
        
        const signal = this.currentSignal;
        const signalClass = signal.signal_type === 'BUY' ? 'bullish' : 'bearish';
        const signalIcon = signal.signal_type === 'BUY' ? 'arrow-up' : 'arrow-down';
        const formattedEntry = parseFloat(signal.entry_price).toFixed(2);
        const formattedTarget = parseFloat(signal.target_price).toFixed(2);
        const formattedStop = parseFloat(signal.stop_loss).toFixed(2);
        const riskReward = ((signal.target_price - signal.entry_price) / (signal.entry_price - signal.stop_loss)).toFixed(2);
        
        panel.innerHTML = `
            <div class="signal ${signalClass}">
                <div class="signal-header">
                    <div class="signal-type-section">
                        <div class="signal-type">
                            <i class="fas fa-${signalIcon}"></i> 
                            <span>${signal.signal_type}</span>
                        </div>
                        <div class="signal-symbol">XAUUSD</div>
                    </div>
                    <div class="signal-confidence">
                        <div class="confidence-meter">
                            <div class="confidence-fill" style="width: ${signal.confidence}%"></div>
                        </div>
                        <span class="confidence-text">${signal.confidence.toFixed(1)}% Confidence</span>
                    </div>
                </div>
                
                <div class="signal-body">
                    <div class="price-levels">
                        <div class="level entry">
                            <span class="level-label">Entry</span>
                            <span class="level-value">$${formattedEntry}</span>
                        </div>
                        <div class="level target">
                            <span class="level-label">Target</span>
                            <span class="level-value">$${formattedTarget}</span>
                        </div>
                        <div class="level stop">
                            <span class="level-label">Stop Loss</span>
                            <span class="level-value">$${formattedStop}</span>
                        </div>
                    </div>
                    
                    <div class="risk-reward">
                        <div class="rr-ratio">1:${riskReward}</div>
                        <span class="rr-label">Risk/Reward</span>
                    </div>
                </div>
                
                <div class="signal-footer">
                    <div class="signal-analysis">
                        <p>${signal.analysis || 'Multi-factor analysis indicates a high-probability trading opportunity.'}</p>
                    </div>
                    <div class="signal-time">
                        <i class="fas fa-clock"></i> ${this.formatTimestamp(signal.timestamp)}
                    </div>
                </div>
                
                <div class="signal-actions">
                    <button class="btn btn-primary btn-trade" onclick="window.openTradeModal('${signal.signal_type}', ${signal.entry_price}, ${signal.target_price}, ${signal.stop_loss})">
                        <i class="fas fa-plus-circle"></i> Add to Portfolio
                    </button>
                    <button class="btn btn-outline btn-analyze" onclick="tradeSignalManager.showSignalDetails('${signal.id || 'current'}')">
                        <i class="fas fa-chart-area"></i> View Analysis
                    </button>
                </div>
            </div>
        `;
    }
    
    updateStatsDisplay() {
        const winRateEl = document.getElementById('win-rate-stat');
        const profitFactorEl = document.getElementById('profit-factor-stat');
        const totalSignalsEl = document.getElementById('total-signals-stat');
        
        if (this.stats && this.stats.total_signals > 0) {
            winRateEl.textContent = `${this.stats.win_rate.toFixed(1)}%`;
            profitFactorEl.textContent = this.stats.profit_factor.toFixed(2);
            totalSignalsEl.textContent = this.stats.total_signals;
            
            // Add color coding
            winRateEl.className = 'stat-value';
            profitFactorEl.className = 'stat-value';
            
            if (this.stats.win_rate >= 60) {
                winRateEl.classList.add('positive');
            } else if (this.stats.win_rate < 45) {
                winRateEl.classList.add('negative');
            }
            
            if (this.stats.profit_factor >= 1.5) {
                profitFactorEl.classList.add('positive');
            } else if (this.stats.profit_factor < 1) {
                profitFactorEl.classList.add('negative');
            }
        } else {
            winRateEl.textContent = 'N/A';
            profitFactorEl.textContent = 'N/A';
            totalSignalsEl.textContent = '0';
        }
    }
    
    updateOpenSignalsDisplay() {
        const container = document.getElementById('open-signals-container');
        
        if (!this.openSignals || this.openSignals.length === 0) {
            container.innerHTML = `
                <div class="open-signals-section">
                    <div class="open-signals-header">
                        <h4><i class="fas fa-layer-group"></i> Open Positions</h4>
                    </div>
                    <div class="no-open-signals">
                        <i class="fas fa-info-circle"></i>
                        <p>No open signals currently active</p>
                    </div>
                </div>
            `;
            return;
        }
        
        let html = `
            <div class="open-signals-section">
                <div class="open-signals-header">
                    <h4><i class="fas fa-layer-group"></i> Open Positions (${this.openSignals.length})</h4>
                </div>
                <div class="open-signals-list">
        `;
        
        this.openSignals.forEach(signal => {
            const signalClass = signal.signal_type === 'BUY' ? 'bullish' : 'bearish';
            const progress = this.calculateTradeProgress(signal);
            
            html += `
                <div class="open-signal-item ${signalClass}">
                    <div class="signal-info">
                        <div class="signal-type">${signal.signal_type}</div>
                        <div class="signal-entry">Entry: $${parseFloat(signal.entry_price).toFixed(2)}</div>
                        <div class="signal-confidence">${signal.confidence.toFixed(1)}%</div>
                    </div>
                    <div class="signal-progress">
                        <div class="progress-bar">
                            <div class="progress-fill ${progress.class}" style="width: ${progress.percent}%"></div>
                        </div>
                        <div class="progress-labels">
                            <span class="stop-label">SL: $${parseFloat(signal.stop_loss).toFixed(2)}</span>
                            <span class="current-label ${progress.class}">$${parseFloat(progress.currentPrice).toFixed(2)}</span>
                            <span class="target-label">TP: $${parseFloat(signal.target_price).toFixed(2)}</span>
                        </div>
                        <div class="progress-pnl ${progress.class}">
                            ${progress.pnl >= 0 ? '+' : ''}${progress.pnl.toFixed(2)}%
                        </div>
                    </div>
                    <div class="signal-actions">
                        <button class="btn btn-sm btn-outline" onclick="tradeSignalManager.closeSignal('${signal.id}')">
                            <i class="fas fa-times"></i> Close
                        </button>
                    </div>
                </div>
            `;
        });
        
        html += '</div></div>';
        container.innerHTML = html;
    }
    
    calculateTradeProgress(signal) {
        // Get current price from global variable or live price feed
        const currentPrice = this.getCurrentGoldPrice();
        
        let percent, progressClass, pnl;
        
        if (signal.signal_type === 'BUY') {
            const totalRange = signal.target_price - signal.stop_loss;
            const currentProgress = currentPrice - signal.stop_loss;
            percent = Math.min(Math.max((currentProgress / totalRange) * 100, 0), 100);
            
            // Calculate P&L percentage
            pnl = ((currentPrice - signal.entry_price) / signal.entry_price) * 100;
            
            if (currentPrice > signal.entry_price) {
                progressClass = 'profit';
            } else if (currentPrice < signal.entry_price) {
                progressClass = 'loss';
            } else {
                progressClass = 'neutral';
            }
        } else { // SELL
            const totalRange = signal.stop_loss - signal.target_price;
            const currentProgress = signal.stop_loss - currentPrice;
            percent = Math.min(Math.max((currentProgress / totalRange) * 100, 0), 100);
            
            // Calculate P&L percentage for short position
            pnl = ((signal.entry_price - currentPrice) / signal.entry_price) * 100;
            
            if (currentPrice < signal.entry_price) {
                progressClass = 'profit';
            } else if (currentPrice > signal.entry_price) {
                progressClass = 'loss';
            } else {
                progressClass = 'neutral';
            }
        }
        
        return {
            percent: percent,
            class: progressClass,
            currentPrice: currentPrice,
            pnl: pnl
        };
    }

    getCurrentGoldPrice() {
        // Try to get current price from various global sources
        if (window.liveGoldPrice) {
            return window.liveGoldPrice;
        }
        if (window.currentGoldPrice) {
            return window.currentGoldPrice;
        }
        if (typeof get_current_gold_price === 'function') {
            return get_current_gold_price();
        }
        
        // Fallback to stored price
        return this.currentGoldPrice;
    }

    async closeSignal(signalId) {
        if (!signalId) return;
        
        try {
            const response = await fetch(`/api/signals/close/${signalId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            if (response.ok) {
                this.showNotification('✅ Signal closed successfully', 'success');
                this.fetchSignals(); // Refresh the display
            } else {
                throw new Error('Failed to close signal');
            }
        } catch (error) {
            console.error('Error closing signal:', error);
            this.showNotification('❌ Failed to close signal', 'error');
        }
    }

    showSignalDetails(signalId) {
        // This would show a detailed analysis modal
        console.log('Showing details for signal:', signalId);
        this.showNotification('📊 Signal analysis details coming soon!', 'info');
    }
    
    showErrorState(errorMessage) {
        const panel = document.getElementById('current-signal-panel');
        panel.innerHTML = `
            <div class="error-state">
                <i class="fas fa-exclamation-triangle"></i>
                <div class="error-content">
                    <h4>Unable to fetch trade signals</h4>
                    <p>${errorMessage || 'Please check your connection and try again.'}</p>
                </div>
                <button class="btn btn-outline" onclick="tradeSignalManager.fetchSignals()">
                    <i class="fas fa-sync"></i> Retry
                </button>
            </div>
        `;
    }

    showNotification(message, type = 'info') {
        // Create a notification element
        const notification = document.createElement('div');
        notification.className = `trade-signal-notification ${type}`;
        notification.innerHTML = `
            <span>${message}</span>
            <button onclick="this.parentElement.remove()" class="notification-close">
                <i class="fas fa-times"></i>
            </button>
        `;
        
        // Add to page
        document.body.appendChild(notification);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            if (notification.parentElement) {
                notification.remove();
            }
        }, 5000);
    }
    
    formatTimestamp(timestamp) {
        if (!timestamp) return 'Just now';
        
        const date = new Date(timestamp);
        const now = new Date();
        const diffMs = now - date;
        const diffMins = Math.floor(diffMs / 60000);
        
        if (diffMins < 1) return 'Just now';
        if (diffMins < 60) return `${diffMins}m ago`;
        if (diffMins < 1440) return `${Math.floor(diffMins / 60)}h ago`;
        
        return date.toLocaleDateString();
    }
}

// Initialize the trade signal manager
const tradeSignalManager = new TradeSignalManager();

// Global function to open trade modal (to be implemented in portfolio.js)
window.openTradeModal = function(type, entry, target, stop) {
    console.log('Opening trade modal with signal:', { type, entry, target, stop });
    
    // Try to integrate with existing portfolio manager
    if (window.portfolioManager && typeof window.portfolioManager.createTradeFromSignal === 'function') {
        window.portfolioManager.createTradeFromSignal(type, entry, target, stop);
    } else if (window.addTradeFromSignal && typeof window.addTradeFromSignal === 'function') {
        window.addTradeFromSignal(type, entry, target, stop);
    } else {
        // Fallback notification
        tradeSignalManager.showNotification(
            `📊 Trade Signal: ${type} Gold at $${entry} (Target: $${target}, Stop: $${stop})`,
            'info'
        );
    }
};

// Auto-initialize when document is ready
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Trade Signal Manager ready for initialization');
    
    // Wait for portfolio section to be ready
    setTimeout(() => {
        const container = document.getElementById('trade-signals-container');
        if (container) {
            tradeSignalManager.init();
        } else {
            console.log('⏳ Trade signals container not found, will retry...');
            // Retry in case the container is created dynamically
            setTimeout(() => {
                tradeSignalManager.init();
            }, 2000);
        }
    }, 1000);
});

// Export for global access
window.tradeSignalManager = tradeSignalManager;
