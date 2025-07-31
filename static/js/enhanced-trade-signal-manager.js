/**
 * Enhanced Trade Signal Manager with TP/SL Monitoring and Learning
 */

class EnhancedTradeSignalManager {
    constructor() {
        this.signalContainer = null;
        this.monitoringInterval = 30000; // Monitor every 30 seconds
        this.activeSignals = [];
        this.signalHistory = [];
        this.performance = {};
        this.isMonitoring = false;
        this.initialized = false;
        this.currentGoldPrice = 3350.70;
    }

    init(containerId = 'trade-signals-container') {
        if (this.initialized) return;
        
        console.log('🚀 Initializing Enhanced Trade Signal Manager...');
        
        this.signalContainer = document.getElementById(containerId);
        if (!this.signalContainer) {
            console.error(`Container #${containerId} not found for enhanced trade signals`);
            return;
        }
        
        // Create enhanced UI structure
        this.createEnhancedUIStructure();
        
        // Load initial data
        this.loadInitialData();
        
        // Start monitoring
        this.startSignalMonitoring();
        
        this.initialized = true;
        console.log('✅ Enhanced Trade Signal Manager initialized successfully');
    }

    async loadInitialData() {
        await Promise.all([
            this.fetchActiveSignals(),
            this.fetchPerformanceData(),
            this.fetchSignalHistory()
        ]);
        
        this.updateAllDisplays();
    }
    
    createEnhancedUIStructure() {
        this.signalContainer.innerHTML = `
            <div class="enhanced-signals-header">
                <div class="signals-title">
                    <h3><i class="fas fa-brain"></i> AI Trade Signals with TP/SL</h3>
                    <p class="signals-subtitle">Advanced signals with automated monitoring</p>
                </div>
                <div class="signals-controls">
                    <button class="btn-generate-signal" onclick="enhancedSignalManager.generateNewSignal()">
                        <i class="fas fa-magic"></i> Generate Signal
                    </button>
                    <div class="monitoring-status ${this.isMonitoring ? 'active' : 'inactive'}">
                        <i class="fas fa-radar"></i>
                        <span class="status-text">${this.isMonitoring ? 'Monitoring Active' : 'Monitoring Off'}</span>
                    </div>
                </div>
            </div>

            <div class="enhanced-signal-stats">
                <!-- Performance stats will be loaded here -->
            </div>

            <div class="enhanced-signals-content">
                <div class="active-signals-section">
                    <h4><i class="fas fa-chart-line"></i> Active Signals</h4>
                    <div class="active-signals-list">
                        <!-- Active signals will be loaded here -->
                    </div>
                </div>

                <div class="signal-history-section">
                    <h4><i class="fas fa-history"></i> Recent History</h4>
                    <div class="signal-history-list">
                        <!-- Signal history will be loaded here -->
                    </div>
                </div>
            </div>
        `;
    }

    async fetchActiveSignals() {
        try {
            const response = await fetch('/api/enhanced-signals/active');
            const data = await response.json();
            
            if (data.success) {
                this.activeSignals = data.active_signals || [];
                console.log(`📊 Loaded ${this.activeSignals.length} active enhanced signals`);
                return true;
            } else {
                console.error('Failed to fetch active signals:', data.error);
                return false;
            }
        } catch (error) {
            console.error('Error fetching active signals:', error);
            return false;
        }
    }

    async fetchPerformanceData() {
        try {
            const response = await fetch('/api/enhanced-signals/performance');
            const data = await response.json();
            
            if (data.success) {
                this.performance = data.performance || {};
                console.log('📈 Enhanced performance data loaded');
                return true;
            } else {
                console.error('Failed to fetch performance data:', data.error);
                return false;
            }
        } catch (error) {
            console.error('Error fetching performance data:', error);
            return false;
        }
    }

    async fetchSignalHistory() {
        try {
            const response = await fetch('/api/enhanced-signals/history?limit=10');
            const data = await response.json();
            
            if (data.success) {
                this.signalHistory = data.signal_history || [];
                console.log(`📚 Loaded ${this.signalHistory.length} historical signals`);
                return true;
            } else {
                console.error('Failed to fetch signal history:', data.error);
                return false;
            }
        } catch (error) {
            console.error('Error fetching signal history:', error);
            return false;
        }
    }

    async generateNewSignal() {
        try {
            console.log('🎯 Generating new enhanced signal...');
            
            // Show loading state
            this.showGeneratingState();
            
            const response = await fetch('/api/enhanced-signals/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const data = await response.json();
            
            if (data.success && data.signal) {
                console.log('✅ New enhanced signal generated:', data.signal);
                
                // Add to active signals
                this.activeSignals.unshift(data.signal);
                
                // Update displays
                this.updateActiveSignalsDisplay();
                
                // Show success notification
                this.showNotification(`🎯 New ${data.signal.signal_type.toUpperCase()} signal generated with ${data.signal.confidence}% confidence!`, 'success');
                
                return data.signal;
            } else {
                console.log('ℹ️ No enhanced signal generated:', data.message);
                this.showNotification(data.message || 'No signal generated - waiting for better market conditions', 'info');
                return null;
            }
        } catch (error) {
            console.error('Error generating enhanced signal:', error);
            this.showNotification('Error generating signal - please try again', 'error');
            return null;
        } finally {
            this.hideGeneratingState();
        }
    }

    async monitorSignalsUpdate() {
        try {
            const response = await fetch('/api/enhanced-signals/monitor');
            const data = await response.json();
            
            if (data.success) {
                const monitoring = data.monitoring;
                
                // Update current price
                if (monitoring.current_price) {
                    this.currentGoldPrice = monitoring.current_price;
                }
                
                // Handle closed signals
                if (monitoring.closed_signals && monitoring.closed_signals.length > 0) {
                    for (const closedSignal of monitoring.closed_signals) {
                        this.handleSignalClosed(closedSignal);
                    }
                    
                    // Refresh performance data after signals close
                    await this.fetchPerformanceData();
                    this.updatePerformanceDisplay();
                }
                
                // Update active signals with current prices
                if (monitoring.updates && monitoring.updates.length > 0) {
                    this.updateSignalPrices(monitoring.updates);
                }
                
                // Update displays if there were changes
                if (monitoring.closed_signals?.length > 0 || monitoring.updates?.length > 0) {
                    this.updateActiveSignalsDisplay();
                }
                
            } else {
                console.error('Enhanced monitoring error:', data.error);
            }
        } catch (error) {
            console.error('Error in enhanced signal monitoring:', error);
        }
    }

    handleSignalClosed(closedSignal) {
        console.log('🔔 Enhanced signal closed:', closedSignal);
        
        // Remove from active signals
        this.activeSignals = this.activeSignals.filter(s => s.id !== closedSignal.id);
        
        // Add to history
        this.signalHistory.unshift({
            ...closedSignal,
            exit_timestamp: new Date().toISOString()
        });
        
        // Show notification with detailed info
        const isProfit = closedSignal.profit_loss > 0;
        const emoji = isProfit ? '🎉' : '📉';
        const message = `${emoji} ${closedSignal.type?.toUpperCase() || 'Signal'} ${closedSignal.exit_reason}: ${isProfit ? '+' : ''}$${closedSignal.profit_loss.toFixed(2)} (${isProfit ? '+' : ''}${closedSignal.profit_loss_pct.toFixed(2)}%)`;
        
        this.showNotification(message, isProfit ? 'success' : 'warning');
        
        // Update history display
        this.updateHistoryDisplay();
    }

    updateSignalPrices(updates) {
        for (const update of updates) {
            const signal = this.activeSignals.find(s => s.id === update.id);
            if (signal) {
                signal.current_price = update.current_price;
                signal.unrealized_pnl = update.unrealized_pnl;
                signal.unrealized_pnl_pct = (update.unrealized_pnl / signal.entry_price) * 100;
            }
        }
    }

    startSignalMonitoring() {
        if (this.isMonitoring) return;
        
        this.isMonitoring = true;
        this.monitoringTimer = setInterval(() => {
            this.monitorSignalsUpdate();
        }, this.monitoringInterval);
        
        this.updateMonitoringStatus();
        console.log('🔄 Enhanced signal monitoring started');
    }

    stopSignalMonitoring() {
        if (this.monitoringTimer) {
            clearInterval(this.monitoringTimer);
            this.monitoringTimer = null;
        }
        this.isMonitoring = false;
        this.updateMonitoringStatus();
        console.log('⏹️ Enhanced signal monitoring stopped');
    }

    updateMonitoringStatus() {
        const statusElement = this.signalContainer.querySelector('.monitoring-status');
        if (statusElement) {
            statusElement.className = `monitoring-status ${this.isMonitoring ? 'active' : 'inactive'}`;
            statusElement.querySelector('.status-text').textContent = this.isMonitoring ? 'Monitoring Active' : 'Monitoring Off';
        }
    }

    updateAllDisplays() {
        this.updatePerformanceDisplay();
        this.updateActiveSignalsDisplay();
        this.updateHistoryDisplay();
    }

    updatePerformanceDisplay() {
        const statsContainer = this.signalContainer.querySelector('.enhanced-signal-stats');
        if (!statsContainer) return;
        
        const perf = this.performance;
        
        statsContainer.innerHTML = `
            <div class="stat-item">
                <div class="stat-label">Success Rate</div>
                <div class="stat-value ${(perf.success_rate || 0) >= 60 ? 'positive' : 'negative'}">
                    ${(perf.success_rate || 0).toFixed(1)}%
                </div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Total Signals</div>
                <div class="stat-value">${perf.total_signals || 0}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Avg P&L</div>
                <div class="stat-value ${(perf.avg_profit_loss_pct || 0) >= 0 ? 'positive' : 'negative'}">
                    ${(perf.avg_profit_loss_pct || 0).toFixed(2)}%
                </div>
            </div>
            <div class="stat-item">
                <div class="stat-label">Best Trade</div>
                <div class="stat-value positive">${(perf.best_profit_pct || 0).toFixed(2)}%</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">7-Day Performance</div>
                <div class="stat-value ${(perf.recent_7_days?.success_rate || 0) >= 60 ? 'positive' : 'negative'}">
                    ${perf.recent_7_days?.signals || 0} signals (${(perf.recent_7_days?.success_rate || 0).toFixed(1)}%)
                </div>
            </div>
        `;
    }

    updateActiveSignalsDisplay() {
        const container = this.signalContainer.querySelector('.active-signals-list');
        if (!container) return;
        
        if (this.activeSignals.length === 0) {
            container.innerHTML = `
                <div class="no-signals">
                    <div class="no-signals-icon">📊</div>
                    <p>No active signals</p>
                    <button class="btn-generate-signal" onclick="enhancedSignalManager.generateNewSignal()">
                        <i class="fas fa-magic"></i> Generate New Signal
                    </button>
                </div>
            `;
            return;
        }
        
        container.innerHTML = this.activeSignals.map(signal => this.createEnhancedSignalCard(signal)).join('');
    }

    createEnhancedSignalCard(signal) {
        const currentPrice = signal.current_price || this.currentGoldPrice;
        const unrealizedPnl = signal.unrealized_pnl || 0;
        const unrealizedPnlPct = signal.unrealized_pnl_pct || 0;
        
        const isProfit = unrealizedPnl > 0;
        const pnlClass = isProfit ? 'positive' : 'negative';
        
        // Calculate progress towards TP/SL
        let progress = 0;
        if (signal.signal_type === 'buy') {
            const totalDistance = signal.target_price - signal.stop_loss;
            const currentDistance = currentPrice - signal.stop_loss;
            progress = Math.max(0, Math.min(100, (currentDistance / totalDistance) * 100));
        } else {
            const totalDistance = signal.stop_loss - signal.target_price;
            const currentDistance = signal.stop_loss - currentPrice;
            progress = Math.max(0, Math.min(100, (currentDistance / totalDistance) * 100));
        }
        
        const timeAgo = this.getTimeAgo(signal.timestamp);
        
        return `
            <div class="enhanced-signal-card ${signal.signal_type}">
                <div class="signal-card-header">
                    <div class="signal-type-badge ${signal.signal_type}">
                        <i class="fas fa-arrow-${signal.signal_type === 'buy' ? 'up' : 'down'}"></i>
                        ${signal.signal_type.toUpperCase()}
                    </div>
                    <div class="signal-confidence">
                        <i class="fas fa-brain"></i>
                        ${signal.confidence.toFixed(1)}%
                    </div>
                </div>
                
                <div class="signal-prices-grid">
                    <div class="price-item">
                        <label>Entry</label>
                        <span class="price">$${signal.entry_price.toFixed(2)}</span>
                    </div>
                    <div class="price-item">
                        <label>Current</label>
                        <span class="price current">$${currentPrice.toFixed(2)}</span>
                    </div>
                    <div class="price-item">
                        <label>Take Profit</label>
                        <span class="price target">$${signal.target_price.toFixed(2)}</span>
                    </div>
                    <div class="price-item">
                        <label>Stop Loss</label>
                        <span class="price stop">$${signal.stop_loss.toFixed(2)}</span>
                    </div>
                </div>
                
                <div class="signal-progress-section">
                    <div class="progress-info">
                        <span>Progress to TP</span>
                        <span class="progress-percent">${progress.toFixed(1)}%</span>
                    </div>
                    <div class="progress-bar">
                        <div class="progress-fill ${signal.signal_type}" style="width: ${progress}%"></div>
                    </div>
                </div>
                
                <div class="signal-pnl-section">
                    <div class="pnl-display ${pnlClass}">
                        <div class="pnl-amount">
                            ${isProfit ? '+' : ''}$${unrealizedPnl.toFixed(2)}
                        </div>
                        <div class="pnl-percent">
                            (${isProfit ? '+' : ''}${unrealizedPnlPct.toFixed(2)}%)
                        </div>
                    </div>
                    <div class="risk-reward-info">
                        <span>R:R ${signal.risk_reward_ratio.toFixed(1)}:1</span>
                    </div>
                </div>
                
                <div class="signal-meta">
                    <div class="signal-time">
                        <i class="fas fa-clock"></i>
                        ${timeAgo}
                    </div>
                </div>
                
                <div class="signal-actions">
                    <button class="btn-analyze" onclick="enhancedSignalManager.showEnhancedAnalysis(${signal.id})">
                        <i class="fas fa-chart-line"></i> Analysis
                    </button>
                    <button class="btn-add-portfolio" onclick="enhancedSignalManager.addToPortfolio(${signal.id})">
                        <i class="fas fa-plus"></i> Portfolio
                    </button>
                </div>
            </div>
        `;
    }

    updateHistoryDisplay() {
        const container = this.signalContainer.querySelector('.signal-history-list');
        if (!container) return;
        
        if (this.signalHistory.length === 0) {
            container.innerHTML = '<div class="no-history">No closed signals yet</div>';
            return;
        }
        
        const recentHistory = this.signalHistory.slice(0, 5);
        container.innerHTML = recentHistory.map(signal => {
            const isSuccess = signal.success || signal.exit_reason === 'take_profit';
            const successClass = isSuccess ? 'success' : 'failure';
            
            return `
                <div class="history-item ${successClass}">
                    <div class="history-type">
                        <i class="fas fa-arrow-${signal.signal_type === 'buy' ? 'up' : 'down'}"></i>
                        ${signal.signal_type?.toUpperCase() || 'SIGNAL'}
                    </div>
                    <div class="history-result">
                        <span class="result-badge ${successClass}">
                            ${isSuccess ? 'TP' : 'SL'}
                        </span>
                    </div>
                    <div class="history-pnl ${isSuccess ? 'positive' : 'negative'}">
                        ${isSuccess ? '+' : ''}${(signal.profit_loss_pct || 0).toFixed(2)}%
                    </div>
                    <div class="history-time">
                        ${this.getTimeAgo(signal.exit_timestamp)}
                    </div>
                </div>
            `;
        }).join('');
    }

    showEnhancedAnalysis(signalId) {
        const signal = this.activeSignals.find(s => s.id === signalId);
        if (!signal) return;
        
        // Create enhanced analysis modal
        const modal = document.createElement('div');
        modal.className = 'enhanced-signal-modal';
        modal.innerHTML = `
            <div class="modal-backdrop" onclick="this.parentElement.remove()"></div>
            <div class="modal-content">
                <div class="modal-header">
                    <h3><i class="fas fa-brain"></i> Enhanced Signal Analysis #${signalId}</h3>
                    <button class="close-btn" onclick="this.closest('.enhanced-signal-modal').remove()">
                        <i class="fas fa-times"></i>
                    </button>
                </div>
                <div class="modal-body">
                    <div class="analysis-section">
                        <h4>Signal Details</h4>
                        <div class="signal-details-grid">
                            <div class="detail-item">
                                <label>Type:</label>
                                <span class="${signal.signal_type}">${signal.signal_type.toUpperCase()}</span>
                            </div>
                            <div class="detail-item">
                                <label>Confidence:</label>
                                <span>${signal.confidence.toFixed(1)}%</span>
                            </div>
                            <div class="detail-item">
                                <label>Entry Price:</label>
                                <span>$${signal.entry_price.toFixed(2)}</span>
                            </div>
                            <div class="detail-item">
                                <label>Take Profit:</label>
                                <span>$${signal.target_price.toFixed(2)}</span>
                            </div>
                            <div class="detail-item">
                                <label>Stop Loss:</label>
                                <span>$${signal.stop_loss.toFixed(2)}</span>
                            </div>
                            <div class="detail-item">
                                <label>Risk:Reward:</label>
                                <span>${signal.risk_reward_ratio.toFixed(1)}:1</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="analysis-section">
                        <h4>AI Analysis Summary</h4>
                        <div class="analysis-text">
                            ${signal.analysis_summary ? signal.analysis_summary.replace(/\n/g, '<br>') : 'No detailed analysis available'}
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
    }

    addToPortfolio(signalId) {
        const signal = this.activeSignals.find(s => s.id === signalId);
        if (!signal) return;
        
        // Add signal to portfolio (integrate with existing portfolio system)
        console.log('Adding enhanced signal to portfolio:', signal);
        
        // Show success notification
        this.showNotification(`📊 Signal #${signalId} added to portfolio tracking`, 'success');
        
        // You can integrate this with your existing portfolio management system
    }

    showGeneratingState() {
        const generateBtns = this.signalContainer.querySelectorAll('.btn-generate-signal');
        generateBtns.forEach(btn => {
            btn.disabled = true;
            btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
        });
    }

    hideGeneratingState() {
        const generateBtns = this.signalContainer.querySelectorAll('.btn-generate-signal');
        generateBtns.forEach(btn => {
            btn.disabled = false;
            btn.innerHTML = '<i class="fas fa-magic"></i> Generate Signal';
        });
    }

    getTimeAgo(timestamp) {
        const now = new Date();
        const time = new Date(timestamp);
        const diffMs = now - time;
        
        const diffMins = Math.floor(diffMs / 60000);
        const diffHours = Math.floor(diffMs / 3600000);
        const diffDays = Math.floor(diffMs / 86400000);
        
        if (diffMins < 1) return 'Just now';
        if (diffMins < 60) return `${diffMins}m ago`;
        if (diffHours < 24) return `${diffHours}h ago`;
        return `${diffDays}d ago`;
    }

    showNotification(message, type = 'info') {
        // Use existing notification system if available
        if (window.notificationManager) {
            window.notificationManager.show({
                title: 'Enhanced Trade Signals',
                message: message,
                type: type,
                duration: 6000
            });
        } else {
            console.log(`[${type.toUpperCase()}] ${message}`);
            
            // Fallback simple notification
            const notification = document.createElement('div');
            notification.className = `simple-notification ${type}`;
            notification.textContent = message;
            notification.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                background: ${type === 'success' ? '#4CAF50' : type === 'error' ? '#f44336' : '#2196F3'};
                color: white;
                padding: 12px 24px;
                border-radius: 4px;
                z-index: 10000;
                max-width: 300px;
            `;
            
            document.body.appendChild(notification);
            
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 5000);
        }
    }

    destroy() {
        if (this.monitoringTimer) {
            clearInterval(this.monitoringTimer);
        }
        this.initialized = false;
        console.log('🔄 Enhanced Trade Signal Manager destroyed');
    }
}

// Global instance
window.enhancedSignalManager = new EnhancedTradeSignalManager();
