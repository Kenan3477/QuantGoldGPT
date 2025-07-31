/**
 * Dual ML Engine Prediction Display with Accuracy Tracking
 * Shows predictions from both engines with their track records
 */

class DualMLPredictionDisplay {
    constructor() {
        this.apiUrl = '/api/ml-predictions/dual';
        this.accuracyApiUrl = '/api/ml-accuracy/stats';
        this.updateInterval = 60000; // 1 minute
        this.isLoading = false;
        this.currentData = null;
        
        this.init();
    }
    
    init() {
        console.log('🔄 Initializing Dual ML Prediction Display...');
        this.createPredictionPanels();
        this.loadInitialData();
        this.setupAutoRefresh();
    }
    
    createPredictionPanels() {
        // Find existing ML prediction container or create one
        let container = document.querySelector('#ml-predictions') ||
                       document.querySelector('.ml-predictions-container') ||
                       document.querySelector('#ml-predictions-container') ||
                       document.querySelector('.predictions-panel') ||
                       document.querySelector('.dashboard-grid');
        
        console.log('🔍 Container search result:', container);
        console.log('🔍 Available elements with ml-predictions:', document.querySelectorAll('[id*="ml-predictions"], [class*="ml-predictions"]'));
        
        if (!container) {
            console.warn('⚠️ No container found for ML predictions, will create one');
            const dashboardGrid = document.querySelector('.dashboard-grid');
            if (dashboardGrid) {
                container = document.createElement('div');
                container.className = 'dashboard-card slide-up ml-predictions-container';
                container.id = 'ml-predictions';
                dashboardGrid.appendChild(container);
                console.log('✅ Created new ML predictions container');
            } else {
                console.error('❌ Cannot find dashboard grid to insert ML predictions');
                return;
            }
        }
        
        // Create dual prediction display
        container.innerHTML = `
            <div class="dual-ml-predictions">
                <div class="dual-ml-header">
                    <h3><i class="fas fa-robot"></i> AI Prediction Engines</h3>
                    <div class="ml-status" id="ml-status">
                        <span class="status-dot loading"></span>
                        <span>Loading predictions...</span>
                    </div>
                </div>
                
                <div class="accuracy-summary" id="accuracy-summary">
                    <div class="accuracy-stats">
                        <div class="stat-item">
                            <span class="stat-label">Best Performer</span>
                            <span class="stat-value" id="best-performer">-</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Total Predictions</span>
                            <span class="stat-value" id="total-predictions">0</span>
                        </div>
                        <div class="stat-item">
                            <span class="stat-label">Last Updated</span>
                            <span class="stat-value" id="last-updated">-</span>
                        </div>
                    </div>
                </div>
                
                <div class="engines-container" id="engines-container">
                    <!-- Engine predictions will be inserted here -->
                </div>
                
                <div class="comparison-panel" id="comparison-panel">
                    <!-- Comparison data will be inserted here -->
                </div>
            </div>
        `;
        
        this.injectStyles();
    }
    
    async loadInitialData() {
        console.log('📊 Loading initial dual ML prediction data...');
        await this.refreshPredictions();
    }
    
    async refreshPredictions() {
        if (this.isLoading) return;
        
        this.isLoading = true;
        this.updateStatus('loading', 'Fetching predictions...');
        
        try {
            // Get dual predictions
            const response = await fetch(this.apiUrl);
            const data = await response.json();
            
            if (data.success) {
                this.currentData = data;
                this.displayPredictions(data);
                this.updateStatus('success', 'Predictions updated');
                
                // Also load accuracy stats
                await this.loadAccuracyStats();
            } else {
                throw new Error(data.error || 'Failed to load predictions');
            }
            
        } catch (error) {
            console.error('❌ Failed to load dual predictions:', error);
            this.updateStatus('error', 'Failed to load predictions');
            this.displayError(error.message);
        } finally {
            this.isLoading = false;
        }
    }
    
    async loadAccuracyStats() {
        try {
            const response = await fetch(this.accuracyApiUrl);
            const data = await response.json();
            
            console.log('📊 Accuracy stats loaded:', data);
            
            if (data.success) {
                this.displayAccuracyStats(data.stats);
            }
        } catch (error) {
            console.error('❌ Failed to load accuracy stats:', error);
        }
    }
    
    displayPredictions(data) {
        const container = document.getElementById('engines-container');
        if (!container) return;
        
        container.innerHTML = '';
        
        // Display each engine's predictions
        data.engines.forEach((engine, index) => {
            const enginePanel = this.createEnginePanel(engine, index);
            container.appendChild(enginePanel);
        });
        
        // Display comparison if available
        if (data.comparison) {
            this.displayComparison(data.comparison);
        }
        
        // Update current price display
        this.updateCurrentPrice(data.current_price);
    }
    
    createEnginePanel(engine, index) {
        const panel = document.createElement('div');
        panel.className = `engine-panel ${engine.status}`;
        
        const statusIcon = engine.status === 'active' ? '✅' : '❌';
        const accuracyBadge = this.getAccuracyBadge(engine.name);
        
        panel.innerHTML = `
            <div class="engine-header">
                <div class="engine-info">
                    <h4>${statusIcon} ${engine.display_name}</h4>
                    ${accuracyBadge}
                </div>
                <div class="engine-metrics">
                    <span class="confidence">Confidence: ${(engine.confidence_avg * 100).toFixed(1)}%</span>
                    <span class="data-quality">${engine.data_quality}</span>
                </div>
            </div>
            
            <div class="predictions-grid">
                ${engine.status === 'active' ? 
                    engine.predictions.map(pred => this.createPredictionCard(pred)).join('') :
                    `<div class="error-message">${engine.error || 'Engine unavailable'}</div>`
                }
            </div>
        `;
        
        return panel;
    }
    
    createPredictionCard(prediction) {
        const directionClass = prediction.direction.toLowerCase();
        const changeSign = prediction.change_percent >= 0 ? '+' : '';
        const confidenceWidth = (prediction.confidence || 0.5) * 100;
        
        return `
            <div class="prediction-card ${directionClass}">
                <div class="prediction-header">
                    <span class="timeframe">${prediction.timeframe}</span>
                    <span class="direction ${directionClass}">${prediction.direction.toUpperCase()}</span>
                </div>
                <div class="prediction-data">
                    <div class="price-prediction">
                        <span class="predicted-price">$${prediction.predicted_price.toFixed(2)}</span>
                        <span class="change-percent ${directionClass}">
                            ${changeSign}${prediction.change_percent.toFixed(3)}%
                        </span>
                    </div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${confidenceWidth}%"></div>
                        <span class="confidence-text">${(prediction.confidence * 100).toFixed(0)}%</span>
                    </div>
                </div>
            </div>
        `;
    }
    
    getAccuracyBadge(engineName) {
        if (!this.accuracyData || !this.accuracyData.engines) return '';
        
        const engineStats = this.accuracyData.engines.find(e => e.name === engineName);
        if (!engineStats) return '';
        
        const accuracy = engineStats.overall_accuracy;
        const badge = engineStats.badge;
        
        return `
            <div class="accuracy-badge" style="background: ${badge.color}">
                ${badge.icon} ${accuracy.toFixed(1)}% ${badge.label}
            </div>
        `;
    }
    
    displayAccuracyStats(stats) {
        this.accuracyData = stats;
        
        console.log('📈 Displaying accuracy stats:', stats);
        
        // Update summary stats
        const bestPerformerEl = document.getElementById('best-performer');
        const totalPredictionsEl = document.getElementById('total-predictions');
        const lastUpdatedEl = document.getElementById('last-updated');
        
        if (bestPerformerEl) {
            bestPerformerEl.textContent = stats.best_performer ? 
                stats.best_performer.replace('_', ' ').toUpperCase() : 'N/A';
        }
        
        if (totalPredictionsEl) {
            totalPredictionsEl.textContent = stats.total_predictions || 0;
        }
        
        if (lastUpdatedEl) {
            lastUpdatedEl.textContent = stats.last_updated ? 
                new Date(stats.last_updated).toLocaleTimeString() : '-';
        }
        
        // Refresh the engine panels to include accuracy badges
        if (this.currentData) {
            this.displayPredictions(this.currentData);
        }
    }
    
    displayComparison(comparison) {
        const container = document.getElementById('comparison-panel');
        if (!container) return;
        
        const agreements = comparison.agreement || {};
        const conflicts = comparison.conflict_areas || [];
        
        container.innerHTML = `
            <div class="comparison-header">
                <h4><i class="fas fa-chart-line"></i> Engine Comparison</h4>
            </div>
            
            <div class="comparison-grid">
                ${Object.entries(agreements).map(([timeframe, data]) => `
                    <div class="comparison-card">
                        <div class="comparison-timeframe">${timeframe}</div>
                        <div class="agreement-status ${data.direction_agreement ? 'agree' : 'disagree'}">
                            ${data.direction_agreement ? '✅ Agree' : '⚠️ Disagree'}
                        </div>
                        <div class="consensus-info">
                            Direction: ${data.consensus_direction.toUpperCase()}<br>
                            Avg Change: ${data.avg_change.toFixed(2)}%<br>
                            Confidence: ${data.confidence_level.toUpperCase()}
                        </div>
                    </div>
                `).join('')}
            </div>
            
            ${conflicts.length > 0 ? `
                <div class="conflicts-section">
                    <h5>⚠️ Prediction Conflicts</h5>
                    ${conflicts.map(conflict => `
                        <div class="conflict-item">
                            <strong>${conflict.timeframe}:</strong> ${conflict.details}
                        </div>
                    `).join('')}
                </div>
            ` : ''}
        `;
    }
    
    updateCurrentPrice(price) {
        // Update price displays throughout the dashboard
        const priceElements = document.querySelectorAll('.current-price, .gold-price, .live-price');
        priceElements.forEach(el => {
            if (el) {
                el.textContent = `$${price.toFixed(2)}`;
            }
        });
    }
    
    updateStatus(type, message) {
        const statusEl = document.getElementById('ml-status');
        if (!statusEl) return;
        
        const dot = statusEl.querySelector('.status-dot');
        const text = statusEl.querySelector('span:last-child');
        
        dot.className = `status-dot ${type}`;
        text.textContent = message;
    }
    
    displayError(message) {
        const container = document.getElementById('engines-container');
        if (!container) return;
        
        container.innerHTML = `
            <div class="error-panel">
                <i class="fas fa-exclamation-triangle"></i>
                <h4>Prediction Error</h4>
                <p>${message}</p>
                <button onclick="window.dualMLDisplay.refreshPredictions()" class="retry-btn">
                    <i class="fas fa-refresh"></i> Retry
                </button>
            </div>
        `;
    }
    
    setupAutoRefresh() {
        setInterval(() => {
            if (!document.hidden) {
                this.refreshPredictions();
            }
        }, this.updateInterval);
        
        // Also refresh when page becomes visible
        document.addEventListener('visibilitychange', () => {
            if (!document.hidden && !this.isLoading) {
                this.refreshPredictions();
            }
        });
    }
    
    injectStyles() {
        if (document.getElementById('dual-ml-styles')) return;
        
        const styles = document.createElement('style');
        styles.id = 'dual-ml-styles';
        styles.textContent = `
            .dual-ml-predictions {
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                border-radius: 12px;
                padding: 20px;
                margin: 15px 0;
                border: 1px solid #2d3748;
            }
            
            .dual-ml-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 20px;
                padding-bottom: 15px;
                border-bottom: 1px solid #2d3748;
            }
            
            .dual-ml-header h3 {
                color: #00d4aa;
                margin: 0;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            
            .ml-status {
                display: flex;
                align-items: center;
                gap: 8px;
                color: #a0aec0;
                font-size: 0.9em;
            }
            
            .status-dot {
                width: 8px;
                height: 8px;
                border-radius: 50%;
                animation: pulse 2s infinite;
            }
            
            .status-dot.loading { background: #fbbf24; }
            .status-dot.success { background: #10b981; }
            .status-dot.error { background: #ef4444; }
            
            .accuracy-summary {
                background: rgba(0, 212, 170, 0.1);
                border-radius: 8px;
                padding: 15px;
                margin-bottom: 20px;
                border: 1px solid rgba(0, 212, 170, 0.2);
            }
            
            .accuracy-stats {
                display: flex;
                justify-content: space-around;
                text-align: center;
            }
            
            .stat-item {
                display: flex;
                flex-direction: column;
                gap: 5px;
            }
            
            .stat-label {
                color: #a0aec0;
                font-size: 0.8em;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            
            .stat-value {
                color: #00d4aa;
                font-weight: bold;
                font-size: 1.1em;
            }
            
            .engines-container {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                gap: 20px;
                margin-bottom: 20px;
            }
            
            .engine-panel {
                background: rgba(255, 255, 255, 0.05);
                border-radius: 10px;
                padding: 18px;
                border: 1px solid #2d3748;
                transition: all 0.3s ease;
            }
            
            .engine-panel:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(0, 212, 170, 0.15);
            }
            
            .engine-panel.error {
                border-color: #ef4444;
                background: rgba(239, 68, 68, 0.1);
            }
            
            .engine-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
                padding-bottom: 10px;
                border-bottom: 1px solid #2d3748;
            }
            
            .engine-info h4 {
                color: #e2e8f0;
                margin: 0;
                font-size: 1.1em;
            }
            
            .accuracy-badge {
                display: inline-flex;
                align-items: center;
                gap: 5px;
                padding: 4px 8px;
                border-radius: 12px;
                font-size: 0.8em;
                font-weight: bold;
                color: white;
                margin-top: 5px;
            }
            
            .engine-metrics {
                display: flex;
                flex-direction: column;
                align-items: flex-end;
                gap: 5px;
                font-size: 0.9em;
                color: #a0aec0;
            }
            
            .predictions-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
                gap: 12px;
            }
            
            .prediction-card {
                background: rgba(255, 255, 255, 0.08);
                border-radius: 8px;
                padding: 12px;
                border: 1px solid #2d3748;
                transition: all 0.3s ease;
            }
            
            .prediction-card.bullish {
                border-left: 3px solid #10b981;
            }
            
            .prediction-card.bearish {
                border-left: 3px solid #ef4444;
            }
            
            .prediction-card.neutral {
                border-left: 3px solid #fbbf24;
            }
            
            .prediction-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 8px;
            }
            
            .timeframe {
                font-weight: bold;
                color: #e2e8f0;
                font-size: 0.9em;
            }
            
            .direction {
                font-size: 0.8em;
                font-weight: bold;
                padding: 2px 6px;
                border-radius: 4px;
            }
            
            .direction.bullish { background: rgba(16, 185, 129, 0.2); color: #10b981; }
            .direction.bearish { background: rgba(239, 68, 68, 0.2); color: #ef4444; }
            .direction.neutral { background: rgba(251, 191, 36, 0.2); color: #fbbf24; }
            
            .prediction-data {
                text-align: center;
            }
            
            .price-prediction {
                margin-bottom: 8px;
            }
            
            .predicted-price {
                display: block;
                font-size: 1.1em;
                font-weight: bold;
                color: #e2e8f0;
                margin-bottom: 2px;
            }
            
            .change-percent {
                font-size: 0.9em;
                font-weight: bold;
            }
            
            .change-percent.bullish { color: #10b981; }
            .change-percent.bearish { color: #ef4444; }
            .change-percent.neutral { color: #fbbf24; }
            
            .confidence-bar {
                position: relative;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 10px;
                height: 6px;
                margin-top: 8px;
            }
            
            .confidence-fill {
                background: linear-gradient(90deg, #00d4aa, #0ea5e9);
                border-radius: 10px;
                height: 100%;
                transition: width 0.5s ease;
            }
            
            .confidence-text {
                position: absolute;
                top: -20px;
                right: 0;
                font-size: 0.7em;
                color: #a0aec0;
            }
            
            .comparison-panel {
                background: rgba(255, 255, 255, 0.05);
                border-radius: 10px;
                padding: 18px;
                border: 1px solid #2d3748;
            }
            
            .comparison-header h4 {
                color: #00d4aa;
                margin: 0 0 15px 0;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            
            .comparison-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
                margin-bottom: 15px;
            }
            
            .comparison-card {
                background: rgba(255, 255, 255, 0.05);
                border-radius: 8px;
                padding: 12px;
                border: 1px solid #2d3748;
            }
            
            .comparison-timeframe {
                font-weight: bold;
                color: #e2e8f0;
                margin-bottom: 8px;
            }
            
            .agreement-status {
                margin-bottom: 8px;
                font-weight: bold;
            }
            
            .agreement-status.agree { color: #10b981; }
            .agreement-status.disagree { color: #ef4444; }
            
            .consensus-info {
                font-size: 0.8em;
                color: #a0aec0;
                line-height: 1.4;
            }
            
            .conflicts-section {
                border-top: 1px solid #2d3748;
                padding-top: 15px;
            }
            
            .conflicts-section h5 {
                color: #fbbf24;
                margin: 0 0 10px 0;
            }
            
            .conflict-item {
                background: rgba(251, 191, 36, 0.1);
                border: 1px solid rgba(251, 191, 36, 0.2);
                border-radius: 6px;
                padding: 8px;
                margin-bottom: 8px;
                font-size: 0.9em;
                color: #e2e8f0;
            }
            
            .error-panel {
                text-align: center;
                padding: 30px;
                color: #ef4444;
            }
            
            .error-panel i {
                font-size: 2em;
                margin-bottom: 15px;
            }
            
            .retry-btn {
                background: #00d4aa;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 6px;
                cursor: pointer;
                margin-top: 15px;
                transition: background 0.3s ease;
            }
            
            .retry-btn:hover {
                background: #00b894;
            }
            
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.5; }
            }
            
            @media (max-width: 768px) {
                .engines-container {
                    grid-template-columns: 1fr;
                }
                
                .accuracy-stats {
                    flex-direction: column;
                    gap: 10px;
                }
                
                .comparison-grid {
                    grid-template-columns: 1fr;
                }
            }
        `;
        
        document.head.appendChild(styles);
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    console.log('🎯 DOM loaded, initializing Dual ML Prediction Display...');
    
    // Check if container exists
    const container = document.querySelector('#ml-predictions') || 
                      document.querySelector('.ml-predictions-container');
    console.log('🔍 Found container:', container);
    
    if (container) {
        console.log('✅ Container found, creating DualMLPredictionDisplay...');
        window.dualMLDisplay = new DualMLPredictionDisplay();
    } else {
        console.warn('⚠️ No ML predictions container found in DOM');
        // Try again after a short delay
        setTimeout(() => {
            console.log('🔄 Retrying container search...');
            const retryContainer = document.querySelector('#ml-predictions') || 
                                   document.querySelector('.ml-predictions-container');
            if (retryContainer) {
                console.log('✅ Container found on retry, creating DualMLPredictionDisplay...');
                window.dualMLDisplay = new DualMLPredictionDisplay();
            } else {
                console.error('❌ ML predictions container still not found after retry');
            }
        }, 2000);
    }
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DualMLPredictionDisplay;
}
