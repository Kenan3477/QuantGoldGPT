/**
 * Enhanced ML Dashboard Controller
 * Handles real-time data loading and display for the Advanced ML Dashboard
 */

class EnhancedMLDashboard {
    constructor() {
        this.refreshInterval = null;
        this.lastUpdateTime = null;
        this.isLoading = false;
        this.retryCount = 0;
        this.maxRetries = 3;
        
        this.init();
    }
    
    async init() {
        console.log('🚀 Initializing Enhanced ML Dashboard...');
        
        // Load initial data
        await this.loadAllData();
        
        // Set up auto-refresh every 60 seconds
        this.startAutoRefresh();
        
        // Set up event listeners
        this.setupEventListeners();
        
        console.log('✅ Enhanced ML Dashboard initialized successfully');
    }
    
    async loadAllData() {
        if (this.isLoading) return;
        
        this.isLoading = true;
        this.updateLoadingStates(true);
        
        try {
            // Load data in parallel for better performance
            const [
                predictions,
                featureImportance,
                accuracyMetrics,
                modelStats,
                marketContext,
                comprehensiveAnalysis
            ] = await Promise.all([
                this.fetchPredictions(),
                this.fetchFeatureImportance(),
                this.fetchAccuracyMetrics(),
                this.fetchModelStats(),
                this.fetchMarketContext(),
                this.fetchComprehensiveAnalysis()
            ]);
            
            // Update UI with loaded data
            this.updatePredictionsDisplay(predictions);
            this.updateFeatureImportanceChart(featureImportance);
            this.updateAccuracyMetrics(accuracyMetrics);
            this.updateModelStats(modelStats);
            this.updateMarketContext(marketContext);
            this.updateComprehensiveAnalysis(comprehensiveAnalysis);
            
            this.lastUpdateTime = new Date();
            this.updateTimestamp();
            this.retryCount = 0;
            
        } catch (error) {
            console.error('❌ Error loading ML dashboard data:', error);
            this.handleLoadError();
        } finally {
            this.isLoading = false;
            this.updateLoadingStates(false);
        }
    }
    
    async fetchPredictions() {
        const response = await fetch('/api/ml-dashboard/predictions');
        if (!response.ok) throw new Error(`Predictions API error: ${response.status}`);
        return await response.json();
    }
    
    async fetchFeatureImportance() {
        const response = await fetch('/api/ml-dashboard/feature-importance');
        if (!response.ok) throw new Error(`Feature importance API error: ${response.status}`);
        return await response.json();
    }
    
    async fetchAccuracyMetrics() {
        const response = await fetch('/api/ml-dashboard/accuracy-metrics?timeframe=7d');
        if (!response.ok) throw new Error(`Accuracy metrics API error: ${response.status}`);
        return await response.json();
    }
    
    async fetchModelStats() {
        const response = await fetch('/api/ml-dashboard/model-stats');
        if (!response.ok) throw new Error(`Model stats API error: ${response.status}`);
        return await response.json();
    }
    
    async fetchMarketContext() {
        const response = await fetch('/api/market-context');
        if (!response.ok) throw new Error(`Market context API error: ${response.status}`);
        return await response.json();
    }
    
    async fetchComprehensiveAnalysis() {
        const response = await fetch('/api/ml-dashboard/comprehensive-analysis');
        if (!response.ok) throw new Error(`Comprehensive analysis API error: ${response.status}`);
        return await response.json();
    }
    
    updatePredictionsDisplay(data) {
        if (!data || !data.success || !data.predictions) {
            console.warn('Invalid predictions data received');
            return;
        }
        
        const predictions = data.predictions;
        const timeframes = ['15m', '1h', '4h', '24h'];
        
        timeframes.forEach(tf => {
            const prediction = predictions[tf];
            if (!prediction) return;
            
            // Update target price
            const priceElement = document.querySelector(`#prediction-${tf} .value`);
            if (priceElement) {
                priceElement.textContent = `$${prediction.target.toFixed(2)}`;
                priceElement.removeAttribute('data-placeholder');
            }
            
            // Update change percentage
            const changeElement = document.querySelector(`#prediction-${tf} .change`);
            if (changeElement) {
                const change = prediction.change_percent;
                changeElement.textContent = `${change > 0 ? '+' : ''}${change.toFixed(2)}%`;
                changeElement.className = `change ${change > 0 ? 'positive' : change < 0 ? 'negative' : 'neutral'}`;
                changeElement.removeAttribute('data-placeholder');
            }
            
            // Update confidence
            const confidenceBar = document.getElementById(`confidence-${tf}`);
            const confidenceText = document.getElementById(`confidence-text-${tf}`);
            if (confidenceBar && confidenceText) {
                const confidence = Math.round(prediction.confidence * 100);
                confidenceBar.style.width = `${confidence}%`;
                confidenceText.textContent = `${confidence}%`;
                confidenceText.removeAttribute('data-placeholder');
            }
            
            // Update direction
            const directionElement = document.querySelector(`#direction-${tf} span`);
            if (directionElement) {
                directionElement.textContent = prediction.direction.charAt(0).toUpperCase() + prediction.direction.slice(1);
                directionElement.className = `direction-${prediction.direction}`;
                directionElement.removeAttribute('data-placeholder');
            }
            
            // Update status indicator
            const statusElement = document.getElementById(`status-${tf}`);
            if (statusElement) {
                statusElement.innerHTML = '<i class="fas fa-check-circle" style="color: #4CAF50;"></i>';
            }
        });
        
        console.log('✅ Predictions display updated');
    }
    
    updateFeatureImportanceChart(data) {
        if (!data || !data.success || !data.features) {
            console.warn('Invalid feature importance data received');
            return;
        }
        
        const canvas = document.getElementById('feature-importance-chart');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        
        // Destroy existing chart if it exists
        if (window.featureChart) {
            window.featureChart.destroy();
        }
        
        const features = data.features;
        const labels = Object.keys(features);
        const values = Object.values(features).map(v => v * 100); // Convert to percentages
        
        window.featureChart = new Chart(ctx, {
            type: 'horizontalBar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Feature Importance (%)',
                    data: values,
                    backgroundColor: [
                        'rgba(54, 162, 235, 0.8)',
                        'rgba(255, 99, 132, 0.8)',
                        'rgba(255, 206, 86, 0.8)',
                        'rgba(75, 192, 192, 0.8)',
                        'rgba(153, 102, 255, 0.8)',
                        'rgba(255, 159, 64, 0.8)'
                    ],
                    borderColor: [
                        'rgba(54, 162, 235, 1)',
                        'rgba(255, 99, 132, 1)',
                        'rgba(255, 206, 86, 1)',
                        'rgba(75, 192, 192, 1)',
                        'rgba(153, 102, 255, 1)',
                        'rgba(255, 159, 64, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        beginAtZero: true,
                        max: Math.max(...values) * 1.1,
                        ticks: {
                            callback: function(value) {
                                return value.toFixed(1) + '%';
                            },
                            color: '#fff'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        }
                    },
                    y: {
                        ticks: {
                            color: '#fff'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        }
                    }
                }
            }
        });
        
        console.log('✅ Feature importance chart updated');
    }
    
    updateAccuracyMetrics(data) {
        if (!data || !data.success || !data.metrics) {
            console.warn('Invalid accuracy metrics data received');
            return;
        }
        
        const metrics = data.metrics;
        
        // Update metric values and bars
        Object.keys(metrics).forEach(metric => {
            const value = Math.round(metrics[metric] * 100);
            
            // Update value display
            const valueElement = document.querySelector(`#${metric.replace('_', '-')} .value`);
            if (valueElement) {
                valueElement.textContent = `${value}%`;
            }
            
            // Update progress bar
            const barElement = document.getElementById(`${metric.replace('_', '-')}-bar`);
            if (barElement) {
                barElement.style.width = `${value}%`;
            }
            
            // Update trend (simulate trend for now)
            const trendElement = document.getElementById(`${metric.replace('_', '-')}-trend`);
            if (trendElement) {
                const trend = Math.random() > 0.5 ? 'up' : 'down';
                const trendIcon = trend === 'up' ? 'fa-arrow-up' : 'fa-arrow-down';
                const trendClass = trend === 'up' ? 'positive' : 'negative';
                const trendText = trend === 'up' ? '+2.1%' : '-1.3%';
                
                trendElement.innerHTML = `<i class="fas ${trendIcon}"></i><span class="${trendClass}">${trendText}</span>`;
            }
        });
        
        // Update accuracy trend chart
        if (data.trend_data && data.trend_data.length > 0) {
            this.updateAccuracyTrendChart(data.trend_data);
        }
        
        console.log('✅ Accuracy metrics updated');
    }
    
    updateAccuracyTrendChart(trendData) {
        const canvas = document.getElementById('accuracy-trend-chart');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        
        // Destroy existing chart if it exists
        if (window.accuracyTrendChart) {
            window.accuracyTrendChart.destroy();
        }
        
        window.accuracyTrendChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: trendData.map(d => d.date),
                datasets: [{
                    label: 'Accuracy %',
                    data: trendData.map(d => d.accuracy * 100),
                    borderColor: 'rgba(54, 162, 235, 1)',
                    backgroundColor: 'rgba(54, 162, 235, 0.1)',
                    borderWidth: 2,
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        ticks: {
                            color: '#fff'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        }
                    },
                    y: {
                        beginAtZero: false,
                        min: 60,
                        max: 100,
                        ticks: {
                            callback: function(value) {
                                return value + '%';
                            },
                            color: '#fff'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        }
                    }
                }
            }
        });
    }
    
    updateModelStats(data) {
        if (!data || !data.success || !data.stats) {
            console.warn('Invalid model stats data received');
            return;
        }
        
        const stats = data.stats;
        
        // Update each stat
        Object.keys(stats).forEach(stat => {
            const element = document.getElementById(stat.replace('_', '-'));
            if (element) {
                let value = stats[stat];
                
                // Format the value based on the stat type
                if (stat === 'avg_response_time') {
                    value = `${value}ms`;
                } else if (stat === 'model_health') {
                    // Update health indicator
                    const indicator = document.getElementById('health-indicator');
                    const text = document.getElementById('health-text');
                    if (indicator && text) {
                        indicator.className = `health-indicator ${value.toLowerCase()}`;
                        text.textContent = value;
                        text.removeAttribute('data-placeholder');
                    }
                    return;
                }
                
                element.textContent = value;
            }
        });
        
        console.log('✅ Model stats updated');
    }
    
    updateMarketContext(data) {
        if (!data || !data.success || !data.context) {
            console.warn('Invalid market context data received');
            return;
        }
        
        const context = data.context;
        
        // Update volatility index
        const volatilityElement = document.getElementById('volatilityIndex');
        if (volatilityElement) {
            volatilityElement.textContent = context.volatility_index + '%';
        }
        
        // Update sentiment score
        const sentimentElement = document.getElementById('sentimentScore');
        if (sentimentElement) {
            sentimentElement.textContent = Math.round(context.sentiment_score * 100) + '%';
        }
        
        // Update market regime
        const regimeIndicator = document.getElementById('regimeIndicator');
        const regimeName = document.getElementById('regimeName');
        const regimeConfidence = document.getElementById('regimeConfidence');
        
        if (regimeIndicator && regimeName && regimeConfidence) {
            regimeIndicator.textContent = context.market_regime.indicator;
            regimeName.textContent = context.market_regime.name;
            regimeConfidence.textContent = Math.round(context.market_regime.confidence * 100) + '%';
        }
        
        // Update key levels
        const levelsList = document.getElementById('levelsList');
        if (levelsList) {
            const levels = context.key_levels;
            levelsList.innerHTML = `
                <div class="level-item support">
                    <span class="level-label">Support:</span>
                    <span class="level-values">
                        $${levels.support[0]} | $${levels.support[1]}
                    </span>
                </div>
                <div class="level-item resistance">
                    <span class="level-label">Resistance:</span>
                    <span class="level-values">
                        $${levels.resistance[0]} | $${levels.resistance[1]}
                    </span>
                </div>
            `;
        }
        
        console.log('✅ Market context updated');
    }
    
    updateComprehensiveAnalysis(data) {
        if (!data || !data.success) {
            console.warn('Invalid comprehensive analysis data received');
            return;
        }
        
        // This could be used for additional dashboard elements
        console.log('✅ Comprehensive analysis loaded:', data);
    }
    
    updateLoadingStates(isLoading) {
        // Update loading indicators across the dashboard
        const loadingElements = document.querySelectorAll('[data-placeholder="true"]');
        loadingElements.forEach(element => {
            if (isLoading) {
                element.classList.add('loading');
            } else {
                element.classList.remove('loading');
            }
        });
        
        // Update spinner states
        const spinners = document.querySelectorAll('.fa-spinner');
        spinners.forEach(spinner => {
            if (!isLoading) {
                spinner.style.display = 'none';
            }
        });
    }
    
    updateTimestamp() {
        const timestampElement = document.getElementById('predictions-update-time');
        if (timestampElement && this.lastUpdateTime) {
            timestampElement.textContent = `Last updated: ${this.lastUpdateTime.toLocaleTimeString()}`;
        }
    }
    
    handleLoadError() {
        this.retryCount++;
        
        if (this.retryCount < this.maxRetries) {
            console.log(`🔄 Retrying data load (${this.retryCount}/${this.maxRetries})...`);
            setTimeout(() => this.loadAllData(), 2000 * this.retryCount);
        } else {
            console.error('❌ Max retries reached. Data loading failed.');
            this.showErrorMessage();
        }
    }
    
    showErrorMessage() {
        // Show user-friendly error message
        const errorElements = document.querySelectorAll('[data-placeholder="true"]');
        errorElements.forEach(element => {
            if (element.getAttribute('data-type') === 'price') {
                element.textContent = 'Error';
            } else if (element.getAttribute('data-type') === 'confidence') {
                element.textContent = '--';
            }
        });
    }
    
    startAutoRefresh() {
        // Clear existing interval
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
        }
        
        // Set up new interval (60 seconds)
        this.refreshInterval = setInterval(() => {
            console.log('🔄 Auto-refreshing ML dashboard data...');
            this.loadAllData();
        }, 60000);
        
        console.log('⏰ Auto-refresh enabled (60 seconds)');
    }
    
    setupEventListeners() {
        // Manual refresh button
        const refreshBtn = document.querySelector('button[onclick="MLDashboard.refreshPredictions()"]');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', (e) => {
                e.preventDefault();
                this.loadAllData();
            });
        }
        
        // Accuracy timeframe selector
        const timeframeSelector = document.getElementById('accuracy-timeframe');
        if (timeframeSelector) {
            timeframeSelector.addEventListener('change', (e) => {
                this.updateAccuracyMetrics(e.target.value);
            });
        }
    }
    
    // Public method for manual refresh
    async refreshPredictions() {
        console.log('🔄 Manual refresh triggered');
        await this.loadAllData();
    }
    
    // Cleanup method
    destroy() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
        }
        
        // Destroy charts
        if (window.featureChart) {
            window.featureChart.destroy();
        }
        if (window.accuracyTrendChart) {
            window.accuracyTrendChart.destroy();
        }
    }
}

// Initialize the dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.enhancedMLDashboard = new EnhancedMLDashboard();
});

// Global functions for backward compatibility
window.MLDashboard = {
    refreshPredictions: () => {
        if (window.enhancedMLDashboard) {
            window.enhancedMLDashboard.refreshPredictions();
        }
    },
    updateAccuracyMetrics: (timeframe) => {
        if (window.enhancedMLDashboard) {
            window.enhancedMLDashboard.fetchAccuracyMetrics(`?timeframe=${timeframe}`)
                .then(data => window.enhancedMLDashboard.updateAccuracyMetrics(data));
        }
    }
};

window.refreshMLPredictions = () => {
    if (window.enhancedMLDashboard) {
        window.enhancedMLDashboard.refreshPredictions();
    }
};

console.log('📊 Enhanced ML Dashboard controller loaded');
            timeframes: timeframes
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        return response.json();
    })
    .then(data => {
        console.log('✅ Enhanced ML Predictions loaded:', data);
        
        if (data.success) {
            // Update prediction cards with comprehensive data
            updateEnhancedPredictionCards(data.predictions);
            
            // Update overall market bias
            updateOverallMarketBias(data.overall_bias);
            
            // Update comprehensive analysis sections
            if (data.comprehensive_analysis) {
                updateTechnicalAnalysisSection(data.comprehensive_analysis.technical);
                updateSentimentAnalysisSection(data.comprehensive_analysis.sentiment);
                updateEconomicAnalysisSection(data.comprehensive_analysis.economic);
                updatePatternAnalysisSection(data.comprehensive_analysis.patterns);
            }
            
            // Update confidence factors
            if (data.confidence_factors) {
                updateConfidenceFactors(data.confidence_factors);
            }
        }
    })
    .catch(error => {
        console.error('❌ Enhanced ML Predictions failed:', error);
        // Fallback to basic ML predictions
        loadBasicMLPredictions();
    });
}

/**
 * Load basic ML predictions as fallback
 */
function loadBasicMLPredictions() {
    console.log('🔄 Loading Basic ML Predictions (Fallback)...');
    
    fetch('/api/ml-predictions')
        .then(response => response.json())
        .then(data => {
            console.log('✅ Basic ML Predictions loaded:', data);
            if (data.success && data.predictions) {
                updateBasicPredictionCards(data.predictions);
            }
        })
        .catch(error => {
            console.error('❌ All ML predictions failed:', error);
            showMLDataUnavailable();
        });
}

/**
 * Update enhanced prediction cards with comprehensive data
 */
function updateEnhancedPredictionCards(predictions) {
    console.log('📊 Updating Enhanced Prediction Cards...');
    
    predictions.forEach(prediction => {
        const timeframe = prediction.timeframe;
        
        // Find prediction card for this timeframe
        const card = document.querySelector(`[data-type="ml-prediction-${timeframe}"]`) ||
                    document.querySelector(`[data-timeframe="${timeframe}"]`) ||
                    document.querySelector(`.prediction-card:contains("${timeframe.toUpperCase()}")`);
        
        if (card) {
            updatePredictionCardContent(card, prediction);
        } else {
            // Create new prediction card if not found
            createPredictionCard(prediction);
        }
    });
}

/**
 * Update prediction card content with comprehensive data
 */
function updatePredictionCardContent(card, prediction) {
    try {
        // Update direction indicator
        const directionEl = card.querySelector('.prediction-direction') || 
                          card.querySelector('.direction') ||
                          card.querySelector('[data-field="direction"]');
        if (directionEl) {
            directionEl.textContent = prediction.direction.toUpperCase();
            directionEl.className = `prediction-direction ${prediction.direction.toLowerCase()}`;
        }
        
        // Update confidence score
        const confidenceEl = card.querySelector('.confidence-score') ||
                           card.querySelector('.confidence') ||
                           card.querySelector('[data-field="confidence"]');
        if (confidenceEl) {
            confidenceEl.textContent = `${prediction.confidence}%`;
        }
        
        // Update target price
        const targetEl = card.querySelector('.target-price') ||
                        card.querySelector('.price-target') ||
                        card.querySelector('[data-field="target"]');
        if (targetEl) {
            targetEl.textContent = `$${prediction.target_price}`;
        }
        
        // Update price change
        const changeEl = card.querySelector('.price-change') ||
                        card.querySelector('[data-field="change"]');
        if (changeEl) {
            const changePercent = prediction.price_change_percent;
            const changeSign = changePercent >= 0 ? '+' : '';
            changeEl.textContent = `${changeSign}${changePercent}%`;
            changeEl.className = `price-change ${changePercent >= 0 ? 'positive' : 'negative'}`;
        }
        
        // Update timeframe
        const timeframeEl = card.querySelector('.timeframe') ||
                          card.querySelector('[data-field="timeframe"]');
        if (timeframeEl) {
            timeframeEl.textContent = prediction.timeframe.toUpperCase();
        }
        
        // Update support/resistance levels
        const supportEl = card.querySelector('.support-level') ||
                         card.querySelector('[data-field="support"]');
        if (supportEl) {
            supportEl.textContent = `$${prediction.support_level}`;
        }
        
        const resistanceEl = card.querySelector('.resistance-level') ||
                           card.querySelector('[data-field="resistance"]');
        if (resistanceEl) {
            resistanceEl.textContent = `$${prediction.resistance_level}`;
        }
        
        // Update stop loss and take profit
        const stopLossEl = card.querySelector('.stop-loss') ||
                          card.querySelector('[data-field="stop-loss"]');
        if (stopLossEl) {
            stopLossEl.textContent = `$${prediction.stop_loss}`;
        }
        
        const takeProfitEl = card.querySelector('.take-profit') ||
                           card.querySelector('[data-field="take-profit"]');
        if (takeProfitEl) {
            takeProfitEl.textContent = `$${prediction.take_profit}`;
        }
        
        // Add visual styling based on direction
        card.classList.remove('bullish', 'bearish', 'neutral');
        card.classList.add(prediction.direction.toLowerCase());
        
        console.log(`✅ Updated ${prediction.timeframe} prediction card`);
        
    } catch (error) {
        console.error(`❌ Error updating prediction card for ${prediction.timeframe}:`, error);
    }
}

/**
 * Update overall market bias display
 */
function updateOverallMarketBias(bias) {
    console.log('📈 Updating Overall Market Bias...', bias);
    
    try {
        // Find bias display elements
        const biasDirection = document.querySelector('.market-bias-direction') ||
                            document.querySelector('[data-field="bias-direction"]');
        if (biasDirection) {
            biasDirection.textContent = bias.direction.toUpperCase();
            biasDirection.className = `market-bias-direction ${bias.direction.toLowerCase()}`;
        }
        
        const biasStrength = document.querySelector('.market-bias-strength') ||
                           document.querySelector('[data-field="bias-strength"]');
        if (biasStrength) {
            biasStrength.textContent = bias.strength.toUpperCase();
        }
        
        const biasScore = document.querySelector('.market-bias-score') ||
                         document.querySelector('[data-field="bias-score"]');
        if (biasScore) {
            biasScore.textContent = bias.score;
        }
        
        // Update factor contributions
        if (bias.factors) {
            const factorEls = {
                technical: document.querySelector('.factor-technical') || document.querySelector('[data-field="factor-technical"]'),
                sentiment: document.querySelector('.factor-sentiment') || document.querySelector('[data-field="factor-sentiment"]'),
                economic: document.querySelector('.factor-economic') || document.querySelector('[data-field="factor-economic"]'),
                pattern: document.querySelector('.factor-pattern') || document.querySelector('[data-field="factor-pattern"]')
            };
            
            Object.keys(factorEls).forEach(factor => {
                if (factorEls[factor] && bias.factors[factor] !== undefined) {
                    factorEls[factor].textContent = bias.factors[factor];
                }
            });
        }
        
    } catch (error) {
        console.error('❌ Error updating market bias:', error);
    }
}

/**
 * Update technical analysis section
 */
function updateTechnicalAnalysisSection(technical) {
    console.log('📊 Updating Technical Analysis...', technical);
    
    try {
        // Update RSI
        const rsiEl = document.querySelector('.rsi-value') || document.querySelector('[data-field="rsi"]');
        if (rsiEl) {
            rsiEl.textContent = technical.rsi;
        }
        
        // Update MACD
        const macdEl = document.querySelector('.macd-value') || document.querySelector('[data-field="macd"]');
        if (macdEl) {
            macdEl.textContent = technical.macd;
        }
        
        // Update trend direction
        const trendEl = document.querySelector('.trend-direction') || document.querySelector('[data-field="trend"]');
        if (trendEl) {
            trendEl.textContent = technical.trend_direction.toUpperCase();
            trendEl.className = `trend-direction ${technical.trend_direction.toLowerCase()}`;
        }
        
        // Update support/resistance
        const supportEl = document.querySelector('.global-support') || document.querySelector('[data-field="global-support"]');
        if (supportEl) {
            supportEl.textContent = `$${technical.support_level}`;
        }
        
        const resistanceEl = document.querySelector('.global-resistance') || document.querySelector('[data-field="global-resistance"]');
        if (resistanceEl) {
            resistanceEl.textContent = `$${technical.resistance_level}`;
        }
        
        // Update Bollinger Bands position
        const bbEl = document.querySelector('.bollinger-position') || document.querySelector('[data-field="bollinger"]');
        if (bbEl) {
            bbEl.textContent = technical.bollinger_position.toUpperCase();
        }
        
        // Update volatility
        const volatilityEl = document.querySelector('.volatility-value') || document.querySelector('[data-field="volatility"]');
        if (volatilityEl) {
            volatilityEl.textContent = `${(technical.volatility * 100).toFixed(1)}%`;
        }
        
    } catch (error) {
        console.error('❌ Error updating technical analysis:', error);
    }
}

/**
 * Update sentiment analysis section
 */
function updateSentimentAnalysisSection(sentiment) {
    console.log('💭 Updating Sentiment Analysis...', sentiment);
    
    try {
        // Update Fear & Greed Index
        const fearGreedEl = document.querySelector('.fear-greed-index') || document.querySelector('[data-field="fear-greed"]');
        if (fearGreedEl) {
            fearGreedEl.textContent = sentiment.fear_greed_index;
        }
        
        // Update news sentiment
        const newsEl = document.querySelector('.news-sentiment') || document.querySelector('[data-field="news-sentiment"]');
        if (newsEl) {
            newsEl.textContent = `${(sentiment.news_sentiment * 100).toFixed(0)}%`;
        }
        
        // Update social sentiment
        const socialEl = document.querySelector('.social-sentiment') || document.querySelector('[data-field="social-sentiment"]');
        if (socialEl) {
            socialEl.textContent = `${(sentiment.social_sentiment * 100).toFixed(0)}%`;
        }
        
        // Update institutional flow
        const institutionalEl = document.querySelector('.institutional-flow') || document.querySelector('[data-field="institutional"]');
        if (institutionalEl) {
            institutionalEl.textContent = sentiment.institutional_flow.replace('_', ' ').toUpperCase();
        }
        
        // Update market mood
        const moodEl = document.querySelector('.market-mood') || document.querySelector('[data-field="market-mood"]');
        if (moodEl) {
            moodEl.textContent = sentiment.market_mood.toUpperCase();
            moodEl.className = `market-mood ${sentiment.market_mood.toLowerCase()}`;
        }
        
    } catch (error) {
        console.error('❌ Error updating sentiment analysis:', error);
    }
}

/**
 * Update economic analysis section
 */
function updateEconomicAnalysisSection(economic) {
    console.log('💰 Updating Economic Analysis...', economic);
    
    try {
        // Update Dollar Index
        const dxyEl = document.querySelector('.dollar-index') || document.querySelector('[data-field="dxy"]');
        if (dxyEl) {
            dxyEl.textContent = economic.dollar_index;
        }
        
        // Update Federal Rate
        const rateEl = document.querySelector('.federal-rate') || document.querySelector('[data-field="fed-rate"]');
        if (rateEl) {
            rateEl.textContent = `${economic.federal_rate}%`;
        }
        
        // Update Inflation
        const inflationEl = document.querySelector('.inflation-rate') || document.querySelector('[data-field="inflation"]');
        if (inflationEl) {
            inflationEl.textContent = `${economic.inflation_cpi}%`;
        }
        
        // Update economic uncertainty
        const uncertaintyEl = document.querySelector('.economic-uncertainty') || document.querySelector('[data-field="uncertainty"]');
        if (uncertaintyEl) {
            uncertaintyEl.textContent = economic.economic_uncertainty;
        }
        
        // Update central bank stance
        const cbEl = document.querySelector('.central-bank-stance') || document.querySelector('[data-field="cb-stance"]');
        if (cbEl) {
            cbEl.textContent = economic.central_bank_stance.toUpperCase();
            cbEl.className = `cb-stance ${economic.central_bank_stance.toLowerCase()}`;
        }
        
    } catch (error) {
        console.error('❌ Error updating economic analysis:', error);
    }
}

/**
 * Update pattern analysis section
 */
function updatePatternAnalysisSection(patterns) {
    console.log('📈 Updating Pattern Analysis...', patterns);
    
    try {
        // Update detected pattern
        const patternEl = document.querySelector('.detected-pattern') || document.querySelector('[data-field="pattern"]');
        if (patternEl) {
            patternEl.textContent = patterns.detected_pattern.replace('_', ' ').toUpperCase();
        }
        
        // Update pattern signal
        const signalEl = document.querySelector('.pattern-signal') || document.querySelector('[data-field="pattern-signal"]');
        if (signalEl) {
            signalEl.textContent = patterns.pattern_signal.toUpperCase();
            signalEl.className = `pattern-signal ${patterns.pattern_signal.toLowerCase()}`;
        }
        
        // Update pattern strength
        const strengthEl = document.querySelector('.pattern-strength') || document.querySelector('[data-field="pattern-strength"]');
        if (strengthEl) {
            strengthEl.textContent = `${(patterns.pattern_strength * 100).toFixed(0)}%`;
        }
        
        // Update reliability score
        const reliabilityEl = document.querySelector('.pattern-reliability') || document.querySelector('[data-field="pattern-reliability"]');
        if (reliabilityEl) {
            reliabilityEl.textContent = `${(patterns.reliability_score * 100).toFixed(0)}%`;
        }
        
    } catch (error) {
        console.error('❌ Error updating pattern analysis:', error);
    }
}

/**
 * Load detailed market analysis
 */
function loadMarketAnalysis() {
    console.log('📊 Loading Detailed Market Analysis...');
    
    fetch('/api/market-analysis')
        .then(response => response.json())
        .then(data => {
            console.log('✅ Market Analysis loaded:', data);
            
            if (data.success) {
                // Update current price display
                updateCurrentPrice(data.current_price);
                
                // Update detailed analysis sections
                if (data.analysis) {
                    updateTechnicalAnalysisSection(data.analysis.technical_indicators);
                    updateSentimentAnalysisSection(data.analysis.market_sentiment);
                    updateEconomicAnalysisSection(data.analysis.economic_factors);
                    updatePatternAnalysisSection(data.analysis.candlestick_patterns);
                }
                
                // Update overall assessment
                if (data.overall_assessment) {
                    updateOverallMarketBias(data.overall_assessment);
                }
            }
        })
        .catch(error => {
            console.error('❌ Market Analysis failed:', error);
        });
}

/**
 * Update current price display
 */
function updateCurrentPrice(price) {
    const priceElements = document.querySelectorAll('.current-price, [data-field="current-price"]');
    priceElements.forEach(el => {
        if (el) {
            el.textContent = `$${price}`;
        }
    });
}

/**
 * Refresh comprehensive analysis
 */
function refreshComprehensiveAnalysis() {
    console.log('🔄 Refreshing Comprehensive Analysis...');
    loadEnhancedMLPredictions();
    loadMarketAnalysis();
}

/**
 * Initialize real-time updates
 */
function initializeRealTimeUpdates() {
    console.log('⚡ Initializing Real-time Updates...');
    
    // Set up WebSocket connection for real-time data if available
    if (window.socket) {
        window.socket.on('ml_prediction_update', (data) => {
            console.log('🔔 Real-time ML prediction update:', data);
            updateEnhancedPredictionCards(data.predictions);
        });
        
        window.socket.on('market_analysis_update', (data) => {
            console.log('🔔 Real-time market analysis update:', data);
            loadMarketAnalysis();
        });
    }
}

/**
 * Show ML data unavailable message
 */
function showMLDataUnavailable() {
    console.log('⚠️ Showing ML data unavailable message');
    
    const mlCards = document.querySelectorAll('.prediction-card, [data-type*="ml-prediction"]');
    mlCards.forEach(card => {
        const loadingEl = card.querySelector('.loading-state') || card;
        if (loadingEl) {
            loadingEl.innerHTML = `
                <div class="ml-unavailable">
                    <span class="warning-icon">⚠️</span>
                    <span>ML Analysis Temporarily Unavailable</span>
                </div>
            `;
        }
    });
}

/**
 * Update basic prediction cards (fallback)
 */
function updateBasicPredictionCards(predictions) {
    console.log('📊 Updating Basic Prediction Cards (Fallback)...');
    
    predictions.forEach(prediction => {
        const card = document.querySelector(`[data-type="ml-prediction-${prediction.timeframe}"]`);
        if (card) {
            updatePredictionCardContent(card, prediction);
        }
    });
}

/**
 * Create new prediction card if not found
 */
function createPredictionCard(prediction) {
    console.log(`📋 Creating new prediction card for ${prediction.timeframe}`);
    // Implementation for creating new card would go here
    // This is a placeholder for dynamic card creation
}

console.log('✅ Enhanced ML Dashboard Controller loaded successfully');

// Auto-initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Initializing Enhanced ML Dashboard...');
    
    try {
        // Initialize the enhanced ML dashboard
        if (typeof EnhancedMLDashboard !== 'undefined') {
            window.enhancedMLDashboard = new EnhancedMLDashboard();
            console.log('✅ Enhanced ML Dashboard initialized successfully');
        } else {
            console.error('❌ EnhancedMLDashboard class not found');
        }
    } catch (error) {
        console.error('❌ Failed to initialize Enhanced ML Dashboard:', error);
    }
});

// Initialize immediately if DOM is already loaded
if (document.readyState === 'loading') {
    // DOM is still loading, wait for DOMContentLoaded
    console.log('⏳ Waiting for DOM to load...');
} else {
    // DOM is already loaded, initialize immediately
    console.log('🚀 DOM already loaded, initializing Enhanced ML Dashboard...');
    
    try {
        if (typeof EnhancedMLDashboard !== 'undefined') {
            window.enhancedMLDashboard = new EnhancedMLDashboard();
            console.log('✅ Enhanced ML Dashboard initialized successfully');
        } else {
            console.error('❌ EnhancedMLDashboard class not found');
        }
    } catch (error) {
        console.error('❌ Failed to initialize Enhanced ML Dashboard:', error);
    }
}
