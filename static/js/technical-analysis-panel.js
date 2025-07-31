/**
 * Technical Analysis Panel Component for Advanced Dashboard
 * Integrates with backend technical analysis and pattern detection
 */

class TechnicalAnalysisPanel {
    constructor() {
        this.indicators = new Map();
        this.patterns = [];
        this.currentTimeframe = '1h';
        this.supportLevels = [];
        this.resistanceLevels = [];
        
        this.initializePanel();
    }
    
    initializePanel() {
        this.createTechnicalPanel();
        this.setupIndicatorControls();
        this.loadTechnicalData();
    }
    
    createTechnicalPanel() {
        const panel = document.createElement('div');
        panel.className = 'dashboard-card technical-analysis-panel';
        panel.innerHTML = `
            <div class="card-header">
                <div class="card-title">
                    <i class="fas fa-chart-area"></i>
                    Technical Analysis
                </div>
                <div class="ta-controls">
                    <button class="btn-small" onclick="technicalPanel.refreshAnalysis()">
                        <i class="fas fa-sync-alt"></i>
                    </button>
                    <button class="btn-small" onclick="technicalPanel.toggleSettings()">
                        <i class="fas fa-cog"></i>
                    </button>
                </div>
            </div>
            <div class="card-content">
                <!-- Technical Indicators Grid -->
                <div class="indicators-grid">
                    <div class="indicator-card rsi-card">
                        <div class="indicator-header">
                            <span class="indicator-name">RSI (14)</span>
                            <span class="indicator-value" id="rsi-value">--</span>
                        </div>
                        <div class="indicator-visual">
                            <div class="rsi-gauge" id="rsi-gauge"></div>
                        </div>
                        <div class="indicator-signal" id="rsi-signal">NEUTRAL</div>
                    </div>
                    
                    <div class="indicator-card macd-card">
                        <div class="indicator-header">
                            <span class="indicator-name">MACD</span>
                            <span class="indicator-value" id="macd-value">--</span>
                        </div>
                        <div class="indicator-visual">
                            <canvas id="macd-mini-chart" width="100" height="40"></canvas>
                        </div>
                        <div class="indicator-signal" id="macd-signal">NEUTRAL</div>
                    </div>
                    
                    <div class="indicator-card bb-card">
                        <div class="indicator-header">
                            <span class="indicator-name">Bollinger Bands</span>
                            <span class="indicator-value" id="bb-position">--</span>
                        </div>
                        <div class="indicator-visual">
                            <div class="bb-position-indicator" id="bb-visual"></div>
                        </div>
                        <div class="indicator-signal" id="bb-signal">NEUTRAL</div>
                    </div>
                    
                    <div class="indicator-card trend-card">
                        <div class="indicator-header">
                            <span class="indicator-name">Trend Analysis</span>
                            <span class="indicator-value" id="trend-direction">--</span>
                        </div>
                        <div class="indicator-visual">
                            <div class="trend-strength-bar" id="trend-strength"></div>
                        </div>
                        <div class="indicator-signal" id="trend-signal">NEUTRAL</div>
                    </div>
                </div>
                
                <!-- Pattern Detection Section -->
                <div class="patterns-section">
                    <div class="section-title">
                        <i class="fas fa-eye"></i>
                        Detected Patterns
                        <span class="pattern-count" id="pattern-count">0</span>
                    </div>
                    <div class="patterns-list" id="patterns-list">
                        <!-- Patterns will be populated here -->
                    </div>
                </div>
                
                <!-- Support/Resistance Levels -->
                <div class="levels-section">
                    <div class="section-title">
                        <i class="fas fa-layer-group"></i>
                        Key Levels
                    </div>
                    <div class="levels-grid">
                        <div class="resistance-levels">
                            <h4>Resistance</h4>
                            <div id="resistance-list">
                                <!-- Resistance levels -->
                            </div>
                        </div>
                        <div class="support-levels">
                            <h4>Support</h4>
                            <div id="support-list">
                                <!-- Support levels -->
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Overall Technical Score -->
                <div class="technical-score-section">
                    <div class="score-circle">
                        <div class="score-value" id="technical-score">--</div>
                        <div class="score-label">Technical Score</div>
                    </div>
                    <div class="score-breakdown">
                        <div class="score-item">
                            <span>Momentum</span>
                            <div class="score-bar">
                                <div class="score-fill" id="momentum-score"></div>
                            </div>
                        </div>
                        <div class="score-item">
                            <span>Trend</span>
                            <div class="score-bar">
                                <div class="score-fill" id="trend-score"></div>
                            </div>
                        </div>
                        <div class="score-item">
                            <span>Volume</span>
                            <div class="score-bar">
                                <div class="score-fill" id="volume-score"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        // Insert panel into dashboard
        const dashboardGrid = document.querySelector('.dashboard-grid');
        dashboardGrid.appendChild(panel);
    }
    
    async loadTechnicalData() {
        try {
            const [technicalData, patternData, comprehensiveData] = await Promise.all([
                fetch(`/api/technical_analysis/XAUUSD?timeframe=${this.currentTimeframe}`).then(r => r.json()),
                fetch(`/api/pattern_detection/XAUUSD?timeframe=${this.currentTimeframe}`).then(r => r.json()),
                fetch(`/api/comprehensive_ta/XAUUSD?timeframe=${this.currentTimeframe}`).then(r => r.json())
            ]);
            
            this.updateIndicators(technicalData.indicators);
            this.updatePatterns(patternData.patterns_found);
            this.updateLevels(technicalData.indicators.support_levels, technicalData.indicators.resistance_levels);
            this.updateTechnicalScore(comprehensiveData.overall_recommendation);
            
        } catch (error) {
            console.error('Error loading technical data:', error);
        }
    }
    
    updateIndicators(indicators) {
        // Update RSI
        const rsiValue = indicators.rsi || 50;
        document.getElementById('rsi-value').textContent = rsiValue.toFixed(1);
        document.getElementById('rsi-signal').textContent = this.getRSISignal(rsiValue);
        document.getElementById('rsi-signal').className = 'indicator-signal ' + this.getRSISignal(rsiValue).toLowerCase();
        this.updateRSIGauge(rsiValue);
        
        // Update MACD
        const macdValue = indicators.macd || 0;
        document.getElementById('macd-value').textContent = macdValue.toFixed(4);
        document.getElementById('macd-signal').textContent = macdValue > 0 ? 'BULLISH' : 'BEARISH';
        document.getElementById('macd-signal').className = 'indicator-signal ' + (macdValue > 0 ? 'bullish' : 'bearish');
        
        // Update Bollinger Bands
        const bbPosition = indicators.bollinger_position || 'MIDDLE';
        document.getElementById('bb-position').textContent = bbPosition;
        document.getElementById('bb-signal').textContent = this.getBBSignal(bbPosition);
        document.getElementById('bb-signal').className = 'indicator-signal ' + this.getBBSignal(bbPosition).toLowerCase();
        
        // Update Trend
        const trendDirection = indicators.trend || 'NEUTRAL';
        document.getElementById('trend-direction').textContent = trendDirection;
        document.getElementById('trend-signal').textContent = trendDirection;
        document.getElementById('trend-signal').className = 'indicator-signal ' + trendDirection.toLowerCase();
    }
    
    updatePatterns(patterns) {
        const patternsList = document.getElementById('patterns-list');
        const patternCount = document.getElementById('pattern-count');
        
        patternCount.textContent = patterns.length;
        
        if (patterns.length === 0) {
            patternsList.innerHTML = '<div class="no-patterns">No patterns detected</div>';
            return;
        }
        
        patternsList.innerHTML = patterns.map(pattern => `
            <div class="pattern-item ${pattern.signal.toLowerCase()}">
                <div class="pattern-info">
                    <div class="pattern-name">${pattern.name}</div>
                    <div class="pattern-meta">
                        <span class="pattern-confidence">${(pattern.confidence * 100).toFixed(0)}%</span>
                        <span class="pattern-time">${this.formatTime(pattern.timestamp)}</span>
                    </div>
                </div>
                <div class="pattern-signal ${pattern.signal.toLowerCase()}">
                    ${pattern.signal}
                </div>
            </div>
        `).join('');
    }
    
    updateLevels(supportLevels, resistanceLevels) {
        const supportList = document.getElementById('support-list');
        const resistanceList = document.getElementById('resistance-list');
        
        supportList.innerHTML = supportLevels.map(level => `
            <div class="level-item support">
                <span class="level-price">$${level.toFixed(2)}</span>
                <span class="level-strength">Strong</span>
            </div>
        `).join('');
        
        resistanceList.innerHTML = resistanceLevels.map(level => `
            <div class="level-item resistance">
                <span class="level-price">$${level.toFixed(2)}</span>
                <span class="level-strength">Strong</span>
            </div>
        `).join('');
    }
    
    updateTechnicalScore(recommendation) {
        const scoreElement = document.getElementById('technical-score');
        const confidence = recommendation.confidence || 0.5;
        const score = Math.round(confidence * 100);
        
        scoreElement.textContent = score;
        scoreElement.className = 'score-value ' + this.getScoreColor(score);
        
        // Update score breakdown
        document.getElementById('momentum-score').style.width = `${score}%`;
        document.getElementById('trend-score').style.width = `${score * 0.9}%`;
        document.getElementById('volume-score').style.width = `${score * 1.1}%`;
    }
    
    // Helper methods
    getRSISignal(rsi) {
        if (rsi < 30) return 'OVERSOLD';
        if (rsi > 70) return 'OVERBOUGHT';
        return 'NEUTRAL';
    }
    
    getBBSignal(position) {
        if (position === 'BELOW_LOWER') return 'OVERSOLD';
        if (position === 'ABOVE_UPPER') return 'OVERBOUGHT';
        return 'NEUTRAL';
    }
    
    getScoreColor(score) {
        if (score >= 80) return 'excellent';
        if (score >= 60) return 'good';
        if (score >= 40) return 'neutral';
        return 'poor';
    }
    
    updateRSIGauge(value) {
        const gauge = document.getElementById('rsi-gauge');
        const angle = (value / 100) * 180;
        gauge.style.background = `conic-gradient(
            var(--danger) 0deg 54deg,
            var(--warning) 54deg 126deg,
            var(--success) 126deg 180deg
        )`;
    }
    
    formatTime(timestamp) {
        return new Date(timestamp).toLocaleTimeString();
    }
    
    async refreshAnalysis() {
        await this.loadTechnicalData();
        this.showNotification('Technical analysis refreshed', 'success');
    }
    
    showNotification(message, type) {
        // Implement notification system
        console.log(`${type}: ${message}`);
    }
}

// Initialize technical analysis panel
let technicalPanel;
document.addEventListener('DOMContentLoaded', () => {
    technicalPanel = new TechnicalAnalysisPanel();
});
