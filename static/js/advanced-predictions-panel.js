/**
 * Advanced ML Predictions Panel
 * Comprehensive prediction display with confidence visualization and strategy breakdown
 */

class AdvancedPredictionsPanel {
    constructor() {
        this.predictions = new Map();
        this.timeframes = ['1M', '5M', '15M', '1H', '4H', '1D'];
        this.strategies = ['ml_momentum', 'conservative', 'aggressive'];
        this.conflictThreshold = 0.3;
        this.updateInterval = 30000; // 30 seconds
        
        this.init();
    }

    init() {
        this.createPanelStructure();
        this.loadPredictions();
        this.setupEventListeners();
        this.startAutoUpdate();
        
        console.log('✅ Advanced Predictions Panel initialized');
    }

    createPanelStructure() {
        const container = document.getElementById('predictions-panel');
        if (!container) return;

        container.innerHTML = `
            <div class="predictions-header">
                <div class="header-content">
                    <h3><i class="fas fa-brain"></i> AI Predictions</h3>
                    <div class="predictions-controls">
                        <button class="control-btn refresh-btn" data-tooltip="Refresh Predictions">
                            <i class="fas fa-sync-alt"></i>
                        </button>
                        <button class="control-btn settings-btn" data-tooltip="Prediction Settings">
                            <i class="fas fa-cog"></i>
                        </button>
                        <div class="confidence-filter">
                            <label>Min Confidence:</label>
                            <input type="range" id="confidence-slider" min="0" max="100" value="60">
                            <span id="confidence-value">60%</span>
                        </div>
                    </div>
                </div>
                <div class="predictions-status">
                    <div class="status-indicator">
                        <span class="status-dot live"></span>
                        <span>Live Predictions</span>
                    </div>
                    <div class="last-update">
                        <span>Updated: <span id="last-update-time">--:--</span></span>
                    </div>
                </div>
            </div>

            <div class="predictions-content">
                <!-- Multi-timeframe predictions grid -->
                <div class="timeframe-predictions">
                    <div class="timeframes-header">
                        <div class="timeframe-tabs">
                            ${this.timeframes.map(tf => `
                                <button class="timeframe-tab ${tf === '1H' ? 'active' : ''}" 
                                        data-timeframe="${tf}">${tf}</button>
                            `).join('')}
                        </div>
                        <div class="view-options">
                            <button class="view-btn grid-view active" data-view="grid">
                                <i class="fas fa-th"></i>
                            </button>
                            <button class="view-btn list-view" data-view="list">
                                <i class="fas fa-list"></i>
                            </button>
                        </div>
                    </div>

                    <div class="predictions-grid" id="predictions-grid">
                        ${this.timeframes.map(tf => this.createTimeframePredictionCard(tf)).join('')}
                    </div>
                </div>

                <!-- Strategy breakdown section -->
                <div class="strategy-breakdown">
                    <h4><i class="fas fa-chart-pie"></i> Strategy Contributions</h4>
                    <div class="strategy-cards">
                        ${this.strategies.map(strategy => this.createStrategyCard(strategy)).join('')}
                    </div>
                </div>

                <!-- Conflict analysis section -->
                <div class="conflict-analysis">
                    <h4><i class="fas fa-exclamation-triangle"></i> Signal Conflicts & Resolutions</h4>
                    <div class="conflicts-container" id="conflicts-container">
                        <div class="no-conflicts">
                            <i class="fas fa-check-circle"></i>
                            <span>No conflicting signals detected</span>
                        </div>
                    </div>
                </div>

                <!-- Probability distributions -->
                <div class="probability-section">
                    <h4><i class="fas fa-chart-area"></i> Confidence Distributions</h4>
                    <div class="probability-charts" id="probability-charts">
                        <!-- Confidence distribution charts will be rendered here -->
                    </div>
                </div>
            </div>

            <!-- Prediction details modal -->
            <div class="prediction-modal" id="prediction-modal">
                <div class="modal-content">
                    <div class="modal-header">
                        <h4>Prediction Details</h4>
                        <button class="close-modal">&times;</button>
                    </div>
                    <div class="modal-body" id="modal-body">
                        <!-- Detailed prediction analysis -->
                    </div>
                </div>
            </div>
        `;
    }

    createTimeframePredictionCard(timeframe) {
        return `
            <div class="prediction-card" data-timeframe="${timeframe}">
                <div class="card-header">
                    <h5>${timeframe}</h5>
                    <div class="card-status">
                        <span class="status-indicator loading"></span>
                    </div>
                </div>
                <div class="card-content">
                    <div class="prediction-summary">
                        <div class="prediction-direction">
                            <span class="direction-icon">⏳</span>
                            <span class="direction-text">Loading...</span>
                        </div>
                        <div class="confidence-display">
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: 0%"></div>
                            </div>
                            <span class="confidence-text">0%</span>
                        </div>
                    </div>
                    
                    <div class="prediction-details">
                        <div class="price-target">
                            <label>Target Price:</label>
                            <span class="target-value">$0.00</span>
                        </div>
                        <div class="time-horizon">
                            <label>Time Horizon:</label>
                            <span class="horizon-value">--</span>
                        </div>
                        <div class="risk-level">
                            <label>Risk Level:</label>
                            <span class="risk-value">--</span>
                        </div>
                    </div>

                    <div class="prediction-metrics">
                        <div class="metric-item">
                            <span class="metric-label">Accuracy</span>
                            <span class="metric-value">--</span>
                        </div>
                        <div class="metric-item">
                            <span class="metric-label">Success Rate</span>
                            <span class="metric-value">--</span>
                        </div>
                    </div>
                </div>
                <div class="card-footer">
                    <button class="detail-btn" data-timeframe="${timeframe}">
                        View Details
                    </button>
                </div>
            </div>
        `;
    }

    createStrategyCard(strategy) {
        const strategyNames = {
            ml_momentum: 'ML Momentum',
            conservative: 'Conservative',
            aggressive: 'Aggressive'
        };

        return `
            <div class="strategy-card" data-strategy="${strategy}">
                <div class="strategy-header">
                    <h5>${strategyNames[strategy]}</h5>
                    <div class="strategy-status">
                        <span class="status-dot"></span>
                    </div>
                </div>
                <div class="strategy-content">
                    <div class="contribution-chart" id="contribution-${strategy}">
                        <!-- Chart will be rendered here -->
                    </div>
                    <div class="strategy-metrics">
                        <div class="metric">
                            <label>Weight:</label>
                            <span class="weight-value">0%</span>
                        </div>
                        <div class="metric">
                            <label>Confidence:</label>
                            <span class="confidence-value">0%</span>
                        </div>
                        <div class="metric">
                            <label>Signal:</label>
                            <span class="signal-value">HOLD</span>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    setupEventListeners() {
        // Timeframe tabs
        document.querySelectorAll('.timeframe-tab').forEach(tab => {
            tab.addEventListener('click', (e) => {
                this.switchTimeframe(e.target.dataset.timeframe);
            });
        });

        // View options
        document.querySelectorAll('.view-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.switchView(e.target.closest('.view-btn').dataset.view);
            });
        });

        // Refresh button
        document.querySelector('.refresh-btn')?.addEventListener('click', () => {
            this.loadPredictions();
        });

        // Confidence filter
        const confidenceSlider = document.getElementById('confidence-slider');
        const confidenceValue = document.getElementById('confidence-value');
        
        confidenceSlider?.addEventListener('input', (e) => {
            const value = e.target.value;
            confidenceValue.textContent = `${value}%`;
            this.filterByConfidence(value);
        });

        // Prediction details
        document.querySelectorAll('.detail-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                this.showPredictionDetails(e.target.dataset.timeframe);
            });
        });

        // Modal close
        document.querySelector('.close-modal')?.addEventListener('click', () => {
            this.closePredictionModal();
        });

        // Click outside modal to close
        document.getElementById('prediction-modal')?.addEventListener('click', (e) => {
            if (e.target.id === 'prediction-modal') {
                this.closePredictionModal();
            }
        });
    }

    async loadPredictions() {
        try {
            this.showLoadingState();
            
            // Load predictions for all timeframes
            const promises = this.timeframes.map(tf => this.fetchPrediction(tf));
            const results = await Promise.allSettled(promises);
            
            results.forEach((result, index) => {
                const timeframe = this.timeframes[index];
                if (result.status === 'fulfilled') {
                    this.updatePredictionCard(timeframe, result.value);
                } else {
                    this.showPredictionError(timeframe, result.reason);
                }
            });

            // Load strategy contributions
            await this.loadStrategyContributions();
            
            // Analyze conflicts
            this.analyzeConflicts();
            
            // Update probability distributions
            this.updateProbabilityCharts();
            
            this.updateLastUpdateTime();
            
        } catch (error) {
            console.error('Error loading predictions:', error);
            this.showErrorState();
        }
    }

    async fetchPrediction(timeframe) {
        const response = await fetch(`/api/ml/predictions/${timeframe}`);
        if (!response.ok) {
            throw new Error(`Failed to fetch ${timeframe} prediction`);
        }
        return await response.json();
    }

    updatePredictionCard(timeframe, prediction) {
        const card = document.querySelector(`[data-timeframe="${timeframe}"]`);
        if (!card || !prediction) return;

        // Store prediction for analysis
        this.predictions.set(timeframe, prediction);

        // Update direction
        const directionIcon = card.querySelector('.direction-icon');
        const directionText = card.querySelector('.direction-text');
        const directionIcons = {
            'BUY': '📈',
            'SELL': '📉',
            'HOLD': '➡️'
        };
        
        directionIcon.textContent = directionIcons[prediction.direction] || '❓';
        directionText.textContent = prediction.direction || 'UNKNOWN';
        directionText.className = `direction-text ${prediction.direction?.toLowerCase() || 'neutral'}`;

        // Update confidence
        const confidence = Math.round(prediction.confidence * 100);
        const confidenceFill = card.querySelector('.confidence-fill');
        const confidenceText = card.querySelector('.confidence-text');
        
        confidenceFill.style.width = `${confidence}%`;
        confidenceFill.className = `confidence-fill ${this.getConfidenceClass(confidence)}`;
        confidenceText.textContent = `${confidence}%`;

        // Update details
        card.querySelector('.target-value').textContent = `$${prediction.targetPrice?.toFixed(2) || '0.00'}`;
        card.querySelector('.horizon-value').textContent = prediction.timeHorizon || '--';
        card.querySelector('.risk-value').textContent = prediction.riskLevel || '--';

        // Update metrics
        const metrics = card.querySelectorAll('.metric-value');
        if (metrics.length >= 2) {
            metrics[0].textContent = `${Math.round(prediction.accuracy * 100)}%`;
            metrics[1].textContent = `${Math.round(prediction.successRate * 100)}%`;
        }

        // Update status
        const statusIndicator = card.querySelector('.status-indicator');
        statusIndicator.className = `status-indicator ${prediction.status || 'active'}`;
    }

    getConfidenceClass(confidence) {
        if (confidence >= 80) return 'high';
        if (confidence >= 60) return 'medium';
        if (confidence >= 40) return 'low';
        return 'very-low';
    }

    async loadStrategyContributions() {
        try {
            const response = await fetch('/api/strategy/contributions');
            if (!response.ok) return;
            
            const contributions = await response.json();
            
            this.strategies.forEach(strategy => {
                this.updateStrategyCard(strategy, contributions[strategy]);
            });
            
        } catch (error) {
            console.error('Error loading strategy contributions:', error);
        }
    }

    updateStrategyCard(strategy, data) {
        const card = document.querySelector(`[data-strategy="${strategy}"]`);
        if (!card || !data) return;

        // Update metrics
        card.querySelector('.weight-value').textContent = `${Math.round(data.weight * 100)}%`;
        card.querySelector('.confidence-value').textContent = `${Math.round(data.confidence * 100)}%`;
        card.querySelector('.signal-value').textContent = data.signal || 'HOLD';

        // Update status
        const statusDot = card.querySelector('.status-dot');
        statusDot.className = `status-dot ${data.status || 'active'}`;

        // Render contribution chart
        this.renderContributionChart(`contribution-${strategy}`, data);
    }

    renderContributionChart(containerId, data) {
        const container = document.getElementById(containerId);
        if (!container || !data.contributions) return;

        // Create simple donut chart for contributions
        const chartData = data.contributions;
        const total = Object.values(chartData).reduce((sum, val) => sum + val, 0);
        
        let html = '<div class="contribution-donut">';
        let startAngle = 0;
        
        Object.entries(chartData).forEach(([source, value]) => {
            const percentage = (value / total) * 100;
            const angle = (value / total) * 360;
            
            html += `
                <div class="contribution-segment" 
                     style="--start-angle: ${startAngle}deg; --end-angle: ${startAngle + angle}deg"
                     data-tooltip="${source}: ${percentage.toFixed(1)}%">
                </div>
            `;
            
            startAngle += angle;
        });
        
        html += `
            <div class="chart-center">
                <span class="total-value">${Math.round(data.confidence * 100)}%</span>
            </div>
        </div>`;
        
        container.innerHTML = html;
    }

    analyzeConflicts() {
        const conflicts = [];
        const timeframePredictions = Array.from(this.predictions.entries());
        
        // Check for conflicts between adjacent timeframes
        for (let i = 0; i < timeframePredictions.length - 1; i++) {
            const [tf1, pred1] = timeframePredictions[i];
            const [tf2, pred2] = timeframePredictions[i + 1];
            
            if (this.hasConflict(pred1, pred2)) {
                conflicts.push({
                    timeframes: [tf1, tf2],
                    predictions: [pred1, pred2],
                    severity: this.calculateConflictSeverity(pred1, pred2),
                    resolution: this.suggestResolution(pred1, pred2)
                });
            }
        }
        
        this.displayConflicts(conflicts);
    }

    hasConflict(pred1, pred2) {
        if (!pred1 || !pred2) return false;
        
        // Check direction conflicts
        if (pred1.direction !== pred2.direction && 
            pred1.direction !== 'HOLD' && pred2.direction !== 'HOLD') {
            return true;
        }
        
        // Check confidence conflicts
        const confidenceDiff = Math.abs(pred1.confidence - pred2.confidence);
        return confidenceDiff > this.conflictThreshold;
    }

    calculateConflictSeverity(pred1, pred2) {
        const directionConflict = (pred1.direction !== pred2.direction) ? 0.5 : 0;
        const confidenceConflict = Math.abs(pred1.confidence - pred2.confidence) * 0.5;
        
        return Math.min(directionConflict + confidenceConflict, 1.0);
    }

    suggestResolution(pred1, pred2) {
        // Simple resolution logic
        if (pred1.confidence > pred2.confidence) {
            return `Follow ${pred1.direction} signal with ${Math.round(pred1.confidence * 100)}% confidence`;
        } else {
            return `Follow ${pred2.direction} signal with ${Math.round(pred2.confidence * 100)}% confidence`;
        }
    }

    displayConflicts(conflicts) {
        const container = document.getElementById('conflicts-container');
        if (!container) return;

        if (conflicts.length === 0) {
            container.innerHTML = `
                <div class="no-conflicts">
                    <i class="fas fa-check-circle"></i>
                    <span>No conflicting signals detected</span>
                </div>
            `;
            return;
        }

        container.innerHTML = conflicts.map(conflict => `
            <div class="conflict-item severity-${this.getSeverityClass(conflict.severity)}">
                <div class="conflict-header">
                    <i class="fas fa-exclamation-triangle"></i>
                    <span>Conflict: ${conflict.timeframes.join(' vs ')}</span>
                    <span class="severity-badge">${this.getSeverityLabel(conflict.severity)}</span>
                </div>
                <div class="conflict-details">
                    <div class="conflicting-predictions">
                        ${conflict.predictions.map((pred, i) => `
                            <div class="prediction-summary">
                                <span class="timeframe">${conflict.timeframes[i]}</span>
                                <span class="direction ${pred.direction?.toLowerCase()}">${pred.direction}</span>
                                <span class="confidence">${Math.round(pred.confidence * 100)}%</span>
                            </div>
                        `).join(' vs ')}
                    </div>
                    <div class="resolution-suggestion">
                        <strong>Suggested Resolution:</strong> ${conflict.resolution}
                    </div>
                </div>
            </div>
        `).join('');
    }

    getSeverityClass(severity) {
        if (severity >= 0.7) return 'high';
        if (severity >= 0.4) return 'medium';
        return 'low';
    }

    getSeverityLabel(severity) {
        if (severity >= 0.7) return 'High';
        if (severity >= 0.4) return 'Medium';
        return 'Low';
    }

    updateProbabilityCharts() {
        const container = document.getElementById('probability-charts');
        if (!container) return;

        // Create confidence distribution chart
        const confidenceData = Array.from(this.predictions.values()).map(pred => pred.confidence);
        
        container.innerHTML = `
            <div class="probability-chart">
                <h5>Confidence Distribution</h5>
                <div class="distribution-bars">
                    ${this.createDistributionBars(confidenceData)}
                </div>
            </div>
            <div class="probability-stats">
                <div class="stat-item">
                    <label>Average Confidence:</label>
                    <span>${this.calculateAverage(confidenceData)}%</span>
                </div>
                <div class="stat-item">
                    <label>High Confidence Signals:</label>
                    <span>${confidenceData.filter(c => c >= 0.8).length}</span>
                </div>
            </div>
        `;
    }

    createDistributionBars(data) {
        const bins = 10;
        const binSize = 1.0 / bins;
        const distribution = new Array(bins).fill(0);
        
        data.forEach(value => {
            const binIndex = Math.min(Math.floor(value / binSize), bins - 1);
            distribution[binIndex]++;
        });
        
        const maxCount = Math.max(...distribution);
        
        return distribution.map((count, i) => {
            const height = maxCount > 0 ? (count / maxCount) * 100 : 0;
            const rangeStart = i * binSize * 100;
            const rangeEnd = (i + 1) * binSize * 100;
            
            return `
                <div class="distribution-bar" 
                     style="height: ${height}%"
                     data-tooltip="${rangeStart.toFixed(0)}-${rangeEnd.toFixed(0)}%: ${count} predictions">
                </div>
            `;
        }).join('');
    }

    calculateAverage(data) {
        if (data.length === 0) return 0;
        const sum = data.reduce((acc, val) => acc + val, 0);
        return Math.round((sum / data.length) * 100);
    }

    switchTimeframe(timeframe) {
        // Update active tab
        document.querySelectorAll('.timeframe-tab').forEach(tab => {
            tab.classList.toggle('active', tab.dataset.timeframe === timeframe);
        });

        // Highlight corresponding prediction card
        document.querySelectorAll('.prediction-card').forEach(card => {
            card.classList.toggle('highlighted', card.dataset.timeframe === timeframe);
        });
    }

    switchView(view) {
        // Update active view button
        document.querySelectorAll('.view-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.view === view);
        });

        // Update grid layout
        const grid = document.getElementById('predictions-grid');
        grid.className = `predictions-grid ${view}-view`;
    }

    filterByConfidence(minConfidence) {
        const threshold = minConfidence / 100;
        
        document.querySelectorAll('.prediction-card').forEach(card => {
            const timeframe = card.dataset.timeframe;
            const prediction = this.predictions.get(timeframe);
            
            if (prediction) {
                const show = prediction.confidence >= threshold;
                card.style.display = show ? 'block' : 'none';
            }
        });
    }

    showPredictionDetails(timeframe) {
        const prediction = this.predictions.get(timeframe);
        if (!prediction) return;

        const modal = document.getElementById('prediction-modal');
        const modalBody = document.getElementById('modal-body');
        
        modalBody.innerHTML = `
            <div class="prediction-detail-content">
                <div class="detail-header">
                    <h4>${timeframe} Prediction Details</h4>
                    <div class="prediction-badge ${prediction.direction?.toLowerCase()}">
                        ${prediction.direction}
                    </div>
                </div>
                
                <div class="detail-sections">
                    <div class="detail-section">
                        <h5>Core Prediction</h5>
                        <div class="detail-grid">
                            <div class="detail-item">
                                <label>Direction:</label>
                                <span>${prediction.direction}</span>
                            </div>
                            <div class="detail-item">
                                <label>Confidence:</label>
                                <span>${Math.round(prediction.confidence * 100)}%</span>
                            </div>
                            <div class="detail-item">
                                <label>Target Price:</label>
                                <span>$${prediction.targetPrice?.toFixed(2)}</span>
                            </div>
                            <div class="detail-item">
                                <label>Stop Loss:</label>
                                <span>$${prediction.stopLoss?.toFixed(2)}</span>
                            </div>
                        </div>
                    </div>
                    
                    <div class="detail-section">
                        <h5>Model Performance</h5>
                        <div class="performance-metrics">
                            <div class="metric">
                                <label>Accuracy:</label>
                                <div class="metric-bar">
                                    <div class="metric-fill" style="width: ${prediction.accuracy * 100}%"></div>
                                </div>
                                <span>${Math.round(prediction.accuracy * 100)}%</span>
                            </div>
                            <div class="metric">
                                <label>Success Rate:</label>
                                <div class="metric-bar">
                                    <div class="metric-fill" style="width: ${prediction.successRate * 100}%"></div>
                                </div>
                                <span>${Math.round(prediction.successRate * 100)}%</span>
                            </div>
                        </div>
                    </div>

                    <div class="detail-section">
                        <h5>Contributing Factors</h5>
                        <div class="factors-list">
                            ${(prediction.factors || []).map(factor => `
                                <div class="factor-item">
                                    <span class="factor-name">${factor.name}</span>
                                    <span class="factor-weight">${Math.round(factor.weight * 100)}%</span>
                                    <span class="factor-impact ${factor.impact}">${factor.impact}</span>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        modal.classList.add('active');
    }

    closePredictionModal() {
        const modal = document.getElementById('prediction-modal');
        modal.classList.remove('active');
    }

    showLoadingState() {
        document.querySelectorAll('.prediction-card').forEach(card => {
            const statusIndicator = card.querySelector('.status-indicator');
            statusIndicator.className = 'status-indicator loading';
        });
    }

    showErrorState() {
        document.querySelectorAll('.prediction-card').forEach(card => {
            const statusIndicator = card.querySelector('.status-indicator');
            statusIndicator.className = 'status-indicator error';
        });
    }

    showPredictionError(timeframe, error) {
        const card = document.querySelector(`[data-timeframe="${timeframe}"]`);
        if (!card) return;

        const content = card.querySelector('.card-content');
        content.innerHTML = `
            <div class="prediction-error">
                <i class="fas fa-exclamation-triangle"></i>
                <p>Failed to load prediction</p>
                <button onclick="window.predictionsPanel.loadPredictions()">Retry</button>
            </div>
        `;
    }

    updateLastUpdateTime() {
        const element = document.getElementById('last-update-time');
        if (element) {
            element.textContent = new Date().toLocaleTimeString();
        }
    }

    startAutoUpdate() {
        setInterval(() => {
            this.loadPredictions();
        }, this.updateInterval);
    }

    // Performance optimization
    optimizePerformance() {
        // Implement virtual scrolling for large datasets
        // Use requestAnimationFrame for smooth animations
        // Debounce rapid updates
    }

    dispose() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
        console.log('✅ Advanced Predictions Panel disposed');
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.predictionsPanel = new AdvancedPredictionsPanel();
});
