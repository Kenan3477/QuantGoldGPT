/**
 * Enhanced AI Analysis Panel
 * Integrates advanced AI systems from bot modules
 */

class EnhancedAIAnalysisPanel {
    constructor() {
        this.analysisData = {};
        this.confidence = 0;
        this.updateInterval = 30000; // 30 seconds
        this.analysisTimer = null;
        
        this.initializePanel();
        this.startAutoAnalysis();
    }
    
    initializePanel() {
        this.createAIPanel();
        this.loadAIAnalysis();
    }
    
    createAIPanel() {
        const existingPanel = document.querySelector('.ai-analysis-panel');
        if (existingPanel) {
            existingPanel.remove();
        }
        
        const panel = document.createElement('div');
        panel.className = 'dashboard-card ai-analysis-panel enhanced';
        panel.innerHTML = `
            <div class="card-header">
                <div class="card-title">
                    <i class="fas fa-brain"></i>
                    Enhanced AI Analysis
                </div>
                <div class="ai-controls">
                    <button class="btn-small" onclick="enhancedAI.runFullAnalysis()" title="Run Full Analysis">
                        <i class="fas fa-magic"></i>
                    </button>
                    <button class="btn-small" onclick="enhancedAI.refreshAnalysis()" title="Refresh">
                        <i class="fas fa-sync-alt"></i>
                    </button>
                    <div class="ai-status" id="ai-status">
                        <i class="fas fa-circle"></i>
                        <span>Analyzing...</span>
                    </div>
                </div>
            </div>
            <div class="card-content">
                <!-- AI Confidence Meter -->
                <div class="confidence-section">
                    <div class="confidence-circle">
                        <svg class="confidence-progress" viewBox="0 0 36 36">
                            <path class="confidence-bg" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="var(--border-secondary)" stroke-width="2"/>
                            <path class="confidence-fill" id="confidence-path" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="var(--success)" stroke-width="2" stroke-dasharray="0, 100"/>
                        </svg>
                        <div class="confidence-text">
                            <div class="confidence-value" id="ai-confidence-value">--</div>
                            <div class="confidence-label">AI Confidence</div>
                        </div>
                    </div>
                    <div class="overall-recommendation">
                        <div class="recommendation-text" id="overall-recommendation">ANALYZING...</div>
                        <div class="recommendation-reasoning" id="recommendation-reasoning">
                            Advanced AI systems are analyzing market conditions...
                        </div>
                    </div>
                </div>
                
                <!-- AI Analysis Components -->
                <div class="analysis-components">
                    <div class="component-grid">
                        <!-- Technical Analysis AI -->
                        <div class="analysis-component technical">
                            <div class="component-header">
                                <i class="fas fa-chart-line"></i>
                                <span>Technical AI</span>
                                <div class="component-score" id="technical-score">--</div>
                            </div>
                            <div class="component-details">
                                <div class="detail-item">
                                    <span>Signal</span>
                                    <span class="signal-badge" id="technical-signal">--</span>
                                </div>
                                <div class="detail-item">
                                    <span>Patterns</span>
                                    <span id="technical-patterns">--</span>
                                </div>
                                <div class="detail-item">
                                    <span>Momentum</span>
                                    <span id="technical-momentum">--</span>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Sentiment Analysis AI -->
                        <div class="analysis-component sentiment">
                            <div class="component-header">
                                <i class="fas fa-heart"></i>
                                <span>Sentiment AI</span>
                                <div class="component-score" id="sentiment-score">--</div>
                            </div>
                            <div class="component-details">
                                <div class="detail-item">
                                    <span>Market Mood</span>
                                    <span class="signal-badge" id="sentiment-mood">--</span>
                                </div>
                                <div class="detail-item">
                                    <span>News Impact</span>
                                    <span id="sentiment-news">--</span>
                                </div>
                                <div class="detail-item">
                                    <span>Social Sentiment</span>
                                    <span id="sentiment-social">--</span>
                                </div>
                            </div>
                        </div>
                        
                        <!-- ML Prediction AI -->
                        <div class="analysis-component ml-prediction">
                            <div class="component-header">
                                <i class="fas fa-robot"></i>
                                <span>ML Predictor</span>
                                <div class="component-score" id="ml-score">--</div>
                            </div>
                            <div class="component-details">
                                <div class="detail-item">
                                    <span>Next 1H</span>
                                    <span class="prediction-value" id="ml-1h">--</span>
                                </div>
                                <div class="detail-item">
                                    <span>Next 4H</span>
                                    <span class="prediction-value" id="ml-4h">--</span>
                                </div>
                                <div class="detail-item">
                                    <span>Next 1D</span>
                                    <span class="prediction-value" id="ml-1d">--</span>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Pattern Recognition AI -->
                        <div class="analysis-component pattern-recognition">
                            <div class="component-header">
                                <i class="fas fa-search"></i>
                                <span>Pattern AI</span>
                                <div class="component-score" id="pattern-score">--</div>
                            </div>
                            <div class="component-details">
                                <div class="detail-item">
                                    <span>Active Patterns</span>
                                    <span id="active-patterns-count">--</span>
                                </div>
                                <div class="detail-item">
                                    <span>Success Rate</span>
                                    <span id="pattern-success-rate">--</span>
                                </div>
                                <div class="detail-item">
                                    <span>Strength</span>
                                    <span id="pattern-strength">--</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Advanced Insights -->
                <div class="advanced-insights">
                    <div class="section-title">
                        <i class="fas fa-lightbulb"></i>
                        AI Insights
                    </div>
                    <div class="insights-list" id="ai-insights">
                        <!-- AI insights will be populated here -->
                    </div>
                </div>
                
                <!-- Risk Assessment -->
                <div class="risk-assessment">
                    <div class="section-title">
                        <i class="fas fa-shield-alt"></i>
                        Risk Assessment
                    </div>
                    <div class="risk-grid">
                        <div class="risk-item">
                            <div class="risk-label">Market Risk</div>
                            <div class="risk-level" id="market-risk">--</div>
                        </div>
                        <div class="risk-item">
                            <div class="risk-label">Volatility Risk</div>
                            <div class="risk-level" id="volatility-risk">--</div>
                        </div>
                        <div class="risk-item">
                            <div class="risk-label">Liquidity Risk</div>
                            <div class="risk-level" id="liquidity-risk">--</div>
                        </div>
                        <div class="risk-item">
                            <div class="risk-label">Sentiment Risk</div>
                            <div class="risk-level" id="sentiment-risk">--</div>
                        </div>
                    </div>
                </div>
                
                <!-- Trade Suggestions -->
                <div class="trade-suggestions">
                    <div class="section-title">
                        <i class="fas fa-target"></i>
                        AI Trade Suggestions
                    </div>
                    <div class="suggestions-list" id="trade-suggestions">
                        <!-- Trade suggestions will be populated here -->
                    </div>
                </div>
            </div>
        `;
        
        // Replace existing AI panel or insert into dashboard
        const existingAIPanel = document.querySelector('.dashboard-card:has(.card-title:contains("AI Market Analysis"))');
        if (existingAIPanel) {
            existingAIPanel.replaceWith(panel);
        } else {
            const dashboardGrid = document.querySelector('.dashboard-grid');
            if (dashboardGrid) {
                dashboardGrid.appendChild(panel);
            }
        }
    }
    
    async loadAIAnalysis() {
        try {
            this.updateStatus('analyzing', 'Running AI analysis...');
            
            // Fetch comprehensive AI analysis
            const response = await fetch('/api/comprehensive_analysis/XAUUSD');
            const data = await response.json();
            
            if (data.error) {
                throw new Error(data.error);
            }
            
            this.analysisData = data;
            this.updateAllComponents();
            this.updateStatus('complete', 'Analysis complete');
            
        } catch (error) {
            console.error('Error loading AI analysis:', error);
            this.updateStatus('error', 'Analysis failed');
            this.loadMockAnalysis();
        }
    }
    
    async runFullAnalysis() {
        try {
            this.updateStatus('analyzing', 'Running comprehensive analysis...');
            
            // Run multiple AI analysis endpoints in parallel
            const [technical, sentiment, patterns, mlPrediction] = await Promise.all([
                fetch('/api/gold/technical').then(r => r.json()).catch(() => ({})),
                fetch('/api/gold/sentiment').then(r => r.json()).catch(() => ({})),
                fetch('/api/gold/patterns').then(r => r.json()).catch(() => ({})),
                fetch('/api/gold/ml_prediction').then(r => r.json()).catch(() => ({}))
            ]);
            
            // Combine all analysis results
            this.analysisData = {
                technical_analysis: technical,
                sentiment_analysis: sentiment,
                pattern_analysis: patterns,
                ml_prediction: mlPrediction,
                overall_confidence: this.calculateOverallConfidence(technical, sentiment, patterns, mlPrediction),
                overall_recommendation: this.generateOverallRecommendation(technical, sentiment, patterns, mlPrediction)
            };
            
            this.updateAllComponents();
            this.updateStatus('complete', 'Full analysis complete');
            
        } catch (error) {
            console.error('Error in full analysis:', error);
            this.updateStatus('error', 'Analysis failed');
        }
    }
    
    updateAllComponents() {
        this.updateConfidenceMeter();
        this.updateOverallRecommendation();
        this.updateTechnicalComponent();
        this.updateSentimentComponent();
        this.updateMLComponent();
        this.updatePatternComponent();
        this.updateInsights();
        this.updateRiskAssessment();
        this.updateTradeSuggestions();
    }
    
    updateConfidenceMeter() {
        const confidence = this.analysisData.overall_confidence || 0.5;
        const confidencePercent = Math.round(confidence * 100);
        
        const confidenceValue = document.getElementById('ai-confidence-value');
        const confidencePath = document.getElementById('confidence-path');
        
        if (confidenceValue) {
            confidenceValue.textContent = `${confidencePercent}%`;
        }
        
        if (confidencePath) {
            const circumference = 100;
            const offset = circumference - (confidence * circumference);
            confidencePath.style.strokeDasharray = `${confidence * circumference}, ${circumference}`;
            
            // Update color based on confidence level
            if (confidence >= 0.8) {
                confidencePath.style.stroke = 'var(--success)';
            } else if (confidence >= 0.6) {
                confidencePath.style.stroke = 'var(--warning)';
            } else {
                confidencePath.style.stroke = 'var(--danger)';
            }
        }
    }
    
    updateOverallRecommendation() {
        const recommendation = this.analysisData.overall_recommendation || 'HOLD';
        const reasoning = this.analysisData.reasoning || 'Analysis in progress...';
        
        const recommendationEl = document.getElementById('overall-recommendation');
        const reasoningEl = document.getElementById('recommendation-reasoning');
        
        if (recommendationEl) {
            recommendationEl.textContent = recommendation;
            recommendationEl.className = `recommendation-text ${recommendation.toLowerCase().replace('_', '-')}`;
        }
        
        if (reasoningEl) {
            reasoningEl.textContent = reasoning;
        }
    }
    
    updateTechnicalComponent() {
        const technical = this.analysisData.technical_analysis || {};
        
        this.updateComponentScore('technical-score', technical.confidence || 0.5);
        this.updateElement('technical-signal', technical.signal || 'NEUTRAL');
        this.updateElement('technical-patterns', `${technical.patterns_detected || 0} detected`);
        this.updateElement('technical-momentum', technical.momentum || 'NEUTRAL');
    }
    
    updateSentimentComponent() {
        const sentiment = this.analysisData.sentiment_analysis || {};
        
        this.updateComponentScore('sentiment-score', sentiment.confidence || 0.5);
        this.updateElement('sentiment-mood', sentiment.overall_sentiment || 'NEUTRAL');
        this.updateElement('sentiment-news', sentiment.news_impact || 'LOW');
        this.updateElement('sentiment-social', sentiment.social_sentiment || 'NEUTRAL');
    }
    
    updateMLComponent() {
        const ml = this.analysisData.ml_prediction || {};
        
        this.updateComponentScore('ml-score', ml.confidence || 0.5);
        this.updateElement('ml-1h', this.formatPrediction(ml.prediction_1h));
        this.updateElement('ml-4h', this.formatPrediction(ml.prediction_4h));
        this.updateElement('ml-1d', this.formatPrediction(ml.prediction_1d));
    }
    
    updatePatternComponent() {
        const patterns = this.analysisData.pattern_analysis || {};
        
        this.updateComponentScore('pattern-score', patterns.confidence || 0.5);
        this.updateElement('active-patterns-count', patterns.active_patterns || 0);
        this.updateElement('pattern-success-rate', `${patterns.success_rate || 0}%`);
        this.updateElement('pattern-strength', patterns.strength || 'WEAK');
    }
    
    updateInsights() {
        const insights = this.analysisData.key_insights || this.generateDefaultInsights();
        const insightsEl = document.getElementById('ai-insights');
        
        if (insightsEl) {
            insightsEl.innerHTML = insights.map(insight => `
                <div class="insight-item">
                    <div class="insight-icon">
                        <i class="fas fa-${this.getInsightIcon(insight.type)}"></i>
                    </div>
                    <div class="insight-content">
                        <div class="insight-title">${insight.title}</div>
                        <div class="insight-description">${insight.description}</div>
                    </div>
                    <div class="insight-impact ${insight.impact.toLowerCase()}">
                        ${insight.impact}
                    </div>
                </div>
            `).join('');
        }
    }
    
    updateRiskAssessment() {
        const risk = this.analysisData.risk_assessment || {};
        
        this.updateRiskLevel('market-risk', risk.market_risk || 'MEDIUM');
        this.updateRiskLevel('volatility-risk', risk.volatility_risk || 'MEDIUM');
        this.updateRiskLevel('liquidity-risk', risk.liquidity_risk || 'LOW');
        this.updateRiskLevel('sentiment-risk', risk.sentiment_risk || 'MEDIUM');
    }
    
    updateTradeSuggestions() {
        const suggestions = this.analysisData.trade_suggestions || this.generateDefaultSuggestions();
        const suggestionsEl = document.getElementById('trade-suggestions');
        
        if (suggestionsEl) {
            suggestionsEl.innerHTML = suggestions.map(suggestion => `
                <div class="suggestion-item ${suggestion.type.toLowerCase()}">
                    <div class="suggestion-header">
                        <div class="suggestion-type">${suggestion.type}</div>
                        <div class="suggestion-confidence">${suggestion.confidence}%</div>
                    </div>
                    <div class="suggestion-details">
                        <div class="suggestion-action">${suggestion.action}</div>
                        <div class="suggestion-target">Target: ${suggestion.target}</div>
                        <div class="suggestion-stop">Stop: ${suggestion.stop_loss}</div>
                    </div>
                </div>
            `).join('');
        }
    }
    
    // Helper methods
    updateComponentScore(elementId, score) {
        const element = document.getElementById(elementId);
        if (element) {
            const percentage = Math.round(score * 100);
            element.textContent = `${percentage}%`;
            element.className = `component-score ${this.getScoreClass(score)}`;
        }
    }
    
    updateElement(elementId, value) {
        const element = document.getElementById(elementId);
        if (element) {
            element.textContent = value;
        }
    }
    
    updateRiskLevel(elementId, level) {
        const element = document.getElementById(elementId);
        if (element) {
            element.textContent = level;
            element.className = `risk-level ${level.toLowerCase()}`;
        }
    }
    
    formatPrediction(prediction) {
        if (!prediction && prediction !== 0) return '--';
        const sign = prediction >= 0 ? '+' : '';
        return `${sign}${prediction.toFixed(2)}%`;
    }
    
    getScoreClass(score) {
        if (score >= 0.8) return 'excellent';
        if (score >= 0.6) return 'good';
        if (score >= 0.4) return 'fair';
        return 'poor';
    }
    
    getInsightIcon(type) {
        const icons = {
            'technical': 'chart-line',
            'fundamental': 'building',
            'sentiment': 'heart',
            'risk': 'exclamation-triangle',
            'opportunity': 'star'
        };
        return icons[type] || 'info-circle';
    }
    
    generateDefaultInsights() {
        return [
            {
                type: 'technical',
                title: 'Technical Divergence Detected',
                description: 'Price action showing divergence with momentum indicators',
                impact: 'HIGH'
            },
            {
                type: 'sentiment',
                title: 'Positive Sentiment Shift',
                description: 'Market sentiment improving based on recent news flow',
                impact: 'MEDIUM'
            },
            {
                type: 'risk',
                title: 'Volatility Expansion Expected',
                description: 'Multiple indicators suggest increased volatility ahead',
                impact: 'MEDIUM'
            }
        ];
    }
    
    generateDefaultSuggestions() {
        return [
            {
                type: 'BUY',
                action: 'Long position recommended',
                target: '$2,100.00',
                stop_loss: '$2,070.00',
                confidence: 75
            },
            {
                type: 'WAIT',
                action: 'Wait for better entry',
                target: '$2,090.00',
                stop_loss: '$2,080.00',
                confidence: 60
            }
        ];
    }
    
    calculateOverallConfidence(technical, sentiment, patterns, ml) {
        // Simple averaging for now - could be more sophisticated
        const scores = [
            technical.confidence || 0.5,
            sentiment.confidence || 0.5,
            patterns.confidence || 0.5,
            ml.confidence || 0.5
        ];
        return scores.reduce((a, b) => a + b) / scores.length;
    }
    
    generateOverallRecommendation(technical, sentiment, patterns, ml) {
        // Simple logic for generating recommendation
        const signals = [
            technical.signal,
            sentiment.overall_sentiment,
            patterns.signal,
            ml.signal
        ].filter(signal => signal);
        
        const buyCount = signals.filter(s => s === 'BUY' || s === 'BULLISH').length;
        const sellCount = signals.filter(s => s === 'SELL' || s === 'BEARISH').length;
        
        if (buyCount > sellCount + 1) return 'STRONG BUY';
        if (buyCount > sellCount) return 'BUY';
        if (sellCount > buyCount + 1) return 'STRONG SELL';
        if (sellCount > buyCount) return 'SELL';
        return 'HOLD';
    }
    
    loadMockAnalysis() {
        this.analysisData = {
            overall_confidence: 0.75,
            overall_recommendation: 'BUY',
            reasoning: 'Multiple AI systems indicate bullish momentum with strong technical setup',
            technical_analysis: {
                signal: 'BUY',
                confidence: 0.8,
                patterns_detected: 3,
                momentum: 'STRONG'
            },
            sentiment_analysis: {
                overall_sentiment: 'BULLISH',
                confidence: 0.7,
                news_impact: 'HIGH',
                social_sentiment: 'POSITIVE'
            },
            ml_prediction: {
                prediction_1h: 0.5,
                prediction_4h: 1.2,
                prediction_1d: 2.1,
                confidence: 0.75
            },
            pattern_analysis: {
                active_patterns: 2,
                success_rate: 78,
                strength: 'STRONG',
                confidence: 0.8
            },
            risk_assessment: {
                market_risk: 'MEDIUM',
                volatility_risk: 'HIGH',
                liquidity_risk: 'LOW',
                sentiment_risk: 'LOW'
            }
        };
        
        this.updateAllComponents();
    }
    
    updateStatus(status, message) {
        const statusEl = document.getElementById('ai-status');
        if (statusEl) {
            statusEl.className = `ai-status ${status}`;
            statusEl.querySelector('span').textContent = message;
        }
    }
    
    async refreshAnalysis() {
        await this.loadAIAnalysis();
    }
    
    startAutoAnalysis() {
        this.analysisTimer = setInterval(() => {
            this.loadAIAnalysis();
        }, this.updateInterval);
    }
    
    stopAutoAnalysis() {
        if (this.analysisTimer) {
            clearInterval(this.analysisTimer);
            this.analysisTimer = null;
        }
    }
    
    destroy() {
        this.stopAutoAnalysis();
    }
}

// Create global instance for component loader
window.aiAnalysisCenter = new EnhancedAIAnalysisPanel();

// Add init method for component loader compatibility
window.aiAnalysisCenter.init = function() {
    this.initializePanel();
    this.startAutoAnalysis();
    return Promise.resolve();
};

// Legacy global reference
let enhancedAI = window.aiAnalysisCenter;

// Auto-initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    if (!window.aiAnalysisCenter.analysisTimer) {
        window.aiAnalysisCenter.init();
    }
});

console.log('🧠 Enhanced AI Analysis Panel loaded successfully');
