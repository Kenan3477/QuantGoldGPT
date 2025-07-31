/**
 * Advanced AI Analysis Center - Real-time AI Trading Analysis
 * Integrates with GoldGPT backend for comprehensive market analysis
 */

class AdvancedAIAnalysisCenter {
    constructor() {
        this.currentSymbol = 'XAUUSD';
        this.analysisData = null;
        this.updateInterval = null;
        this.autoRefreshEnabled = true;
        this.refreshRate = 30000; // 30 seconds
        
        // Analysis components
        this.technicalAnalysis = null;
        this.sentimentAnalysis = null;
        this.mlPredictions = null;
        this.overallRecommendation = null;
        
        // UI elements
        this.analysisContainer = null;
        this.loadingStates = new Map();
        
        this.init();
    }
    
    /**
     * Initialize the AI Analysis Center
     */
    init() {
        console.log('🧠 Initializing Advanced AI Analysis Center...');
        
        // Create analysis UI if it doesn't exist
        this.createAnalysisUI();
        
        // Set up event listeners
        this.setupEventListeners();
        
        // Load initial analysis
        this.loadComprehensiveAnalysis();
        
        // Start auto-refresh
        this.startAutoRefresh();
        
        console.log('✅ AI Analysis Center initialized');
    }
    
    /**
     * Create the AI Analysis UI components
     */
    createAnalysisUI() {
        // Find or create the analysis container
        this.analysisContainer = document.getElementById('ai-analysis-center');
        
        if (!this.analysisContainer) {
            console.log('📊 Creating AI Analysis Center UI...');
            this.createAnalysisCenterHTML();
        }
        
        // Cache UI elements
        this.cacheUIElements();
    }
    
    /**
     * Create the HTML structure for AI Analysis Center
     */
    createAnalysisCenterHTML() {
        const analysisHTML = `
            <div id="ai-analysis-center" class="ai-analysis-center">
                <!-- AI Analysis Header -->
                <div class="analysis-header">
                    <div class="analysis-title">
                        <i class="fas fa-brain"></i>
                        <h2>AI Analysis Center</h2>
                        <div class="analysis-status" id="analysis-status">
                            <div class="status-dot loading"></div>
                            <span>Analyzing...</span>
                        </div>
                    </div>
                    <div class="analysis-controls">
                        <div class="symbol-selector">
                            <select id="analysis-symbol-select">
                                <option value="XAUUSD">Gold (XAU/USD)</option>
                                <option value="XAGUSD">Silver (XAG/USD)</option>
                                <option value="EURUSD">EUR/USD</option>
                                <option value="GBPUSD">GBP/USD</option>
                                <option value="BTCUSD">Bitcoin</option>
                            </select>
                        </div>
                        <button id="refresh-analysis-btn" class="btn btn-primary">
                            <i class="fas fa-sync-alt"></i>
                            Refresh
                        </button>
                        <button id="auto-refresh-toggle" class="btn btn-secondary active">
                            <i class="fas fa-play"></i>
                            Auto
                        </button>
                    </div>
                </div>
                
                <!-- Overall Recommendation -->
                <div class="overall-recommendation" id="overall-recommendation">
                    <div class="recommendation-card">
                        <div class="recommendation-header">
                            <h3>Overall Recommendation</h3>
                            <div class="confidence-score" id="confidence-score">85%</div>
                        </div>
                        <div class="recommendation-content">
                            <div class="recommendation-signal" id="recommendation-signal">
                                <span class="signal-badge buy">BUY</span>
                                <span class="signal-strength">Strong Signal</span>
                            </div>
                            <div class="recommendation-reasoning" id="recommendation-reasoning">
                                <ul>
                                    <li>Technical indicators show bullish momentum</li>
                                    <li>Market sentiment is positive</li>
                                    <li>ML models predict upward movement</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Analysis Grid -->
                <div class="analysis-grid">
                    <!-- Technical Analysis Panel -->
                    <div class="analysis-panel technical-panel">
                        <div class="panel-header">
                            <h3><i class="fas fa-chart-line"></i> Technical Analysis</h3>
                            <div class="panel-status" id="technical-status">
                                <div class="loading-spinner"></div>
                            </div>
                        </div>
                        <div class="panel-content" id="technical-content">
                            <div class="indicators-grid" id="technical-indicators">
                                <!-- Technical indicators will be populated here -->
                            </div>
                        </div>
                    </div>
                    
                    <!-- Sentiment Analysis Panel -->
                    <div class="analysis-panel sentiment-panel">
                        <div class="panel-header">
                            <h3><i class="fas fa-heart"></i> Market Sentiment</h3>
                            <div class="panel-status" id="sentiment-status">
                                <div class="loading-spinner"></div>
                            </div>
                        </div>
                        <div class="panel-content" id="sentiment-content">
                            <div class="sentiment-overview" id="sentiment-overview">
                                <!-- Sentiment data will be populated here -->
                            </div>
                        </div>
                    </div>
                    
                    <!-- ML Predictions Panel -->
                    <div class="analysis-panel ml-panel">
                        <div class="panel-header">
                            <h3><i class="fas fa-robot"></i> ML Predictions</h3>
                            <div class="panel-status" id="ml-status">
                                <div class="loading-spinner"></div>
                            </div>
                        </div>
                        <div class="panel-content" id="ml-content">
                            <div class="predictions-grid" id="ml-predictions">
                                <!-- ML predictions will be populated here -->
                            </div>
                        </div>
                    </div>
                    
                    <!-- Market News Panel -->
                    <div class="analysis-panel news-panel">
                        <div class="panel-header">
                            <h3><i class="fas fa-newspaper"></i> Market News</h3>
                            <div class="panel-status" id="news-status">
                                <div class="loading-spinner"></div>
                            </div>
                        </div>
                        <div class="panel-content" id="news-content">
                            <div class="news-list" id="market-news">
                                <!-- Market news will be populated here -->
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // Find appropriate container and insert
        const contentArea = document.querySelector('.content') || document.body;
        const tempDiv = document.createElement('div');
        tempDiv.innerHTML = analysisHTML;
        contentArea.appendChild(tempDiv.firstElementChild);
    }
    
    /**
     * Cache UI elements for better performance
     */
    cacheUIElements() {
        this.elements = {
            analysisStatus: document.getElementById('analysis-status'),
            symbolSelect: document.getElementById('analysis-symbol-select'),
            refreshBtn: document.getElementById('refresh-analysis-btn'),
            autoRefreshToggle: document.getElementById('auto-refresh-toggle'),
            overallRecommendation: document.getElementById('overall-recommendation'),
            confidenceScore: document.getElementById('confidence-score'),
            recommendationSignal: document.getElementById('recommendation-signal'),
            recommendationReasoning: document.getElementById('recommendation-reasoning'),
            technicalContent: document.getElementById('technical-content'),
            technicalIndicators: document.getElementById('technical-indicators'),
            sentimentContent: document.getElementById('sentiment-content'),
            sentimentOverview: document.getElementById('sentiment-overview'),
            mlContent: document.getElementById('ml-content'),
            mlPredictions: document.getElementById('ml-predictions'),
            newsContent: document.getElementById('news-content'),
            marketNews: document.getElementById('market-news')
        };
    }
    
    /**
     * Set up event listeners
     */
    setupEventListeners() {
        // Symbol selection
        if (this.elements.symbolSelect) {
            this.elements.symbolSelect.addEventListener('change', (e) => {
                this.currentSymbol = e.target.value;
                this.loadComprehensiveAnalysis();
            });
        }
        
        // Manual refresh
        if (this.elements.refreshBtn) {
            this.elements.refreshBtn.addEventListener('click', () => {
                this.loadComprehensiveAnalysis();
            });
        }
        
        // Auto-refresh toggle
        if (this.elements.autoRefreshToggle) {
            this.elements.autoRefreshToggle.addEventListener('click', () => {
                this.toggleAutoRefresh();
            });
        }
        
        // Socket.IO event listeners
        if (typeof socket !== 'undefined') {
            socket.on('ai_analysis_update', (data) => {
                this.handleAnalysisUpdate(data);
            });
            
            socket.on('technical_analysis_update', (data) => {
                this.updateTechnicalAnalysis(data.technical_signals);
            });
            
            socket.on('sentiment_analysis_update', (data) => {
                this.updateSentimentAnalysis(data.sentiment);
            });
        }
    }
    
    /**
     * Load comprehensive AI analysis
     */
    async loadComprehensiveAnalysis() {
        try {
            console.log(`🔄 Loading AI analysis for ${this.currentSymbol}...`);
            
            // Update status
            this.updateAnalysisStatus('loading', 'Analyzing market data...');
            
            // Show loading states
            this.showLoadingStates();
            
            // Fetch comprehensive analysis
            const response = await fetch(`/api/ai-analysis/${this.currentSymbol}`);
            const result = await response.json();
            
            if (result.success) {
                this.analysisData = result.data;
                this.renderAnalysisData();
                this.updateAnalysisStatus('success', 'Analysis complete');
                console.log('✅ AI analysis loaded successfully');
            } else {
                throw new Error(result.error || 'Analysis failed');
            }
            
        } catch (error) {
            console.error('❌ Error loading AI analysis:', error);
            this.updateAnalysisStatus('error', 'Analysis failed');
            this.showErrorState(error.message);
        }
    }
    
    /**
     * Render the complete analysis data
     */
    renderAnalysisData() {
        if (!this.analysisData) return;
        
        // Update overall recommendation
        this.updateOverallRecommendation();
        
        // Update individual panels
        this.updateTechnicalAnalysis(this.analysisData.technical_signals);
        this.updateSentimentAnalysis(this.analysisData.sentiment);
        this.updateMLPredictions(this.analysisData.ml_predictions);
        this.updateMarketNews();
        
        // Hide loading states
        this.hideLoadingStates();
    }
    
    /**
     * Update overall recommendation display
     */
    updateOverallRecommendation() {
        if (!this.analysisData) return;
        
        const { overall_recommendation, confidence_score, reasoning } = this.analysisData;
        
        // Update confidence score
        if (this.elements.confidenceScore) {
            this.elements.confidenceScore.textContent = `${Math.round(confidence_score * 100)}%`;
        }
        
        // Update recommendation signal
        if (this.elements.recommendationSignal) {
            const signalClass = overall_recommendation.toLowerCase();
            const signalStrength = confidence_score > 0.8 ? 'Strong' : 
                                 confidence_score > 0.6 ? 'Moderate' : 'Weak';
            
            this.elements.recommendationSignal.innerHTML = `
                <span class="signal-badge ${signalClass}">${overall_recommendation}</span>
                <span class="signal-strength">${signalStrength} Signal</span>
            `;
        }
        
        // Update reasoning
        if (this.elements.recommendationReasoning && reasoning) {
            const reasoningHTML = reasoning.map(reason => `<li>${reason}</li>`).join('');
            this.elements.recommendationReasoning.innerHTML = `<ul>${reasoningHTML}</ul>`;
        }
    }
    
    /**
     * Update technical analysis display
     */
    updateTechnicalAnalysis(technicalSignals) {
        if (!this.elements.technicalIndicators || !technicalSignals) return;
        
        const indicatorsHTML = technicalSignals.map(signal => {
            const signalClass = signal.signal.toLowerCase();
            const strengthPercent = Math.round(signal.strength * 100);
            
            return `
                <div class="indicator-card">
                    <div class="indicator-header">
                        <span class="indicator-name">${signal.indicator}</span>
                        <span class="indicator-timeframe">${signal.timeframe}</span>
                    </div>
                    <div class="indicator-value">${this.formatIndicatorValue(signal.value)}</div>
                    <div class="indicator-signal">
                        <span class="signal-badge ${signalClass}">${signal.signal}</span>
                        <div class="strength-bar">
                            <div class="strength-fill ${signalClass}" style="width: ${strengthPercent}%"></div>
                        </div>
                    </div>
                </div>
            `;
        }).join('');
        
        this.elements.technicalIndicators.innerHTML = indicatorsHTML;
    }
    
    /**
     * Update sentiment analysis display
     */
    updateSentimentAnalysis(sentimentData) {
        if (!this.elements.sentimentOverview || !sentimentData) return;
        
        const overallPercent = Math.round(sentimentData.overall_score * 100);
        const newsPercent = Math.round(sentimentData.news_sentiment * 100);
        const socialPercent = Math.round(sentimentData.social_sentiment * 100);
        
        const sentimentHTML = `
            <div class="sentiment-gauge">
                <div class="gauge-container">
                    <div class="gauge-chart" data-value="${overallPercent}">
                        <div class="gauge-value">${overallPercent}%</div>
                        <div class="gauge-label">Overall Sentiment</div>
                    </div>
                </div>
            </div>
            
            <div class="sentiment-breakdown">
                <div class="sentiment-item">
                    <span class="sentiment-label">News Sentiment</span>
                    <div class="sentiment-bar">
                        <div class="sentiment-fill positive" style="width: ${newsPercent}%"></div>
                    </div>
                    <span class="sentiment-value">${newsPercent}%</span>
                </div>
                
                <div class="sentiment-item">
                    <span class="sentiment-label">Social Sentiment</span>
                    <div class="sentiment-bar">
                        <div class="sentiment-fill positive" style="width: ${socialPercent}%"></div>
                    </div>
                    <span class="sentiment-value">${socialPercent}%</span>
                </div>
                
                <div class="sentiment-item">
                    <span class="sentiment-label">Fear & Greed Index</span>
                    <div class="sentiment-bar">
                        <div class="sentiment-fill neutral" style="width: ${sentimentData.fear_greed_index}%"></div>
                    </div>
                    <span class="sentiment-value">${sentimentData.fear_greed_index}</span>
                </div>
            </div>
            
            <div class="sentiment-sources">
                <span class="sources-count">${sentimentData.sources_count} sources analyzed</span>
            </div>
        `;
        
        this.elements.sentimentOverview.innerHTML = sentimentHTML;
        
        // Initialize gauge animation
        this.animateGauge();
    }
    
    /**
     * Update ML predictions display
     */
    updateMLPredictions(predictions) {
        if (!this.elements.mlPredictions || !predictions) return;
        
        const predictionsHTML = predictions.map(prediction => {
            const changePercent = ((prediction.predicted_price - this.analysisData.current_price) / 
                                 this.analysisData.current_price * 100);
            const changeClass = changePercent >= 0 ? 'positive' : 'negative';
            const confidencePercent = Math.round(prediction.confidence * 100);
            
            return `
                <div class="prediction-card">
                    <div class="prediction-header">
                        <span class="prediction-timeframe">${prediction.timeframe}</span>
                        <span class="prediction-confidence">${confidencePercent}%</span>
                    </div>
                    <div class="prediction-price">
                        $${prediction.predicted_price.toFixed(2)}
                    </div>
                    <div class="prediction-change ${changeClass}">
                        <i class="fas fa-arrow-${changePercent >= 0 ? 'up' : 'down'}"></i>
                        ${changePercent >= 0 ? '+' : ''}${changePercent.toFixed(2)}%
                    </div>
                    <div class="prediction-direction">
                        <span class="direction-badge ${prediction.direction.toLowerCase()}">${prediction.direction}</span>
                        <span class="probability">${Math.round(prediction.probability * 100)}%</span>
                    </div>
                </div>
            `;
        }).join('');
        
        this.elements.mlPredictions.innerHTML = predictionsHTML;
    }
    
    /**
     * Update market news display
     */
    updateMarketNews() {
        if (!this.elements.marketNews) return;
        
        // Mock news data - in real implementation, this would come from news API
        const mockNews = [
            {
                title: "Fed signals potential rate cuts in Q2 2024",
                impact: "high",
                time: "2 min ago",
                sentiment: "positive"
            },
            {
                title: "Gold reaches new monthly high amid inflation concerns",
                impact: "medium",
                time: "15 min ago",
                sentiment: "positive"
            },
            {
                title: "China's manufacturing PMI beats expectations",
                impact: "low",
                time: "1 hour ago",
                sentiment: "neutral"
            }
        ];
        
        const newsHTML = mockNews.map(news => `
            <div class="news-item">
                <div class="news-header">
                    <span class="news-impact ${news.impact}">${news.impact.toUpperCase()}</span>
                    <span class="news-time">${news.time}</span>
                </div>
                <div class="news-title">${news.title}</div>
                <div class="news-sentiment ${news.sentiment}">${news.sentiment}</div>
            </div>
        `).join('');
        
        this.elements.marketNews.innerHTML = newsHTML;
    }
    
    /**
     * Handle real-time analysis updates from WebSocket
     */
    handleAnalysisUpdate(data) {
        if (data.success && data.symbol === this.currentSymbol) {
            this.analysisData = data.data;
            this.renderAnalysisData();
        }
    }
    
    /**
     * Update analysis status
     */
    updateAnalysisStatus(status, message) {
        if (!this.elements.analysisStatus) return;
        
        const statusDot = this.elements.analysisStatus.querySelector('.status-dot');
        const statusText = this.elements.analysisStatus.querySelector('span');
        
        if (statusDot) {
            statusDot.className = `status-dot ${status}`;
        }
        
        if (statusText) {
            statusText.textContent = message;
        }
    }
    
    /**
     * Show loading states for all panels
     */
    showLoadingStates() {
        const panels = ['technical', 'sentiment', 'ml', 'news'];
        panels.forEach(panel => {
            const statusElement = document.getElementById(`${panel}-status`);
            if (statusElement) {
                statusElement.innerHTML = '<div class="loading-spinner"></div>';
            }
        });
    }
    
    /**
     * Hide loading states for all panels
     */
    hideLoadingStates() {
        const panels = ['technical', 'sentiment', 'ml', 'news'];
        panels.forEach(panel => {
            const statusElement = document.getElementById(`${panel}-status`);
            if (statusElement) {
                statusElement.innerHTML = '<i class="fas fa-check-circle success"></i>';
            }
        });
    }
    
    /**
     * Show error state
     */
    showErrorState(message) {
        const errorHTML = `
            <div class="error-state">
                <i class="fas fa-exclamation-triangle"></i>
                <p>Error loading analysis: ${message}</p>
                <button onclick="aiAnalysisCenter.loadComprehensiveAnalysis()" class="btn btn-primary">
                    Try Again
                </button>
            </div>
        `;
        
        if (this.analysisContainer) {
            this.analysisContainer.innerHTML = errorHTML;
        }
    }
    
    /**
     * Start auto-refresh
     */
    startAutoRefresh() {
        if (this.autoRefreshEnabled && !this.updateInterval) {
            this.updateInterval = setInterval(() => {
                this.loadComprehensiveAnalysis();
            }, this.refreshRate);
            
            console.log(`🔄 Auto-refresh started (${this.refreshRate/1000}s interval)`);
        }
    }
    
    /**
     * Stop auto-refresh
     */
    stopAutoRefresh() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
            console.log('⏹️ Auto-refresh stopped');
        }
    }
    
    /**
     * Toggle auto-refresh
     */
    toggleAutoRefresh() {
        this.autoRefreshEnabled = !this.autoRefreshEnabled;
        
        if (this.autoRefreshEnabled) {
            this.startAutoRefresh();
            this.elements.autoRefreshToggle.classList.add('active');
            this.elements.autoRefreshToggle.innerHTML = '<i class="fas fa-pause"></i> Auto';
        } else {
            this.stopAutoRefresh();
            this.elements.autoRefreshToggle.classList.remove('active');
            this.elements.autoRefreshToggle.innerHTML = '<i class="fas fa-play"></i> Auto';
        }
    }
    
    /**
     * Format indicator values for display
     */
    formatIndicatorValue(value) {
        if (typeof value === 'number') {
            return value.toFixed(2);
        }
        return value;
    }
    
    /**
     * Animate sentiment gauge
     */
    animateGauge() {
        const gauges = document.querySelectorAll('.gauge-chart');
        gauges.forEach(gauge => {
            const value = parseInt(gauge.dataset.value);
            // Add gauge animation logic here
        });
    }
    
    /**
     * Cleanup resources
     */
    destroy() {
        this.stopAutoRefresh();
        
        // Remove event listeners
        if (this.elements.symbolSelect) {
            this.elements.symbolSelect.removeEventListener('change', this.loadComprehensiveAnalysis);
        }
        
        console.log('🧠 AI Analysis Center destroyed');
    }
}

// Global instance
let aiAnalysisCenter = null;

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    // Initialize AI Analysis Center if container exists or create it
    aiAnalysisCenter = new AdvancedAIAnalysisCenter();
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AdvancedAIAnalysisCenter;
}
