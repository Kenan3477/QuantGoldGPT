/**
 * Enhanced Dashboard Navigation and Functionality
 * Makes Portfolio, Analysis, and Markets buttons fully functional
 */

class EnhancedDashboardManager {
    constructor() {
        this.currentSection = 'dashboard';
        this.socket = io();
        this.updateIntervals = {};
        this.charts = {};
        
        this.initializeEnhancedDashboard();
    }
    
    initializeEnhancedDashboard() {
        this.setupNavigationHandlers();
        this.setupSocketListeners();
        this.loadInitialData();
        this.startAutoUpdates();
    }
    
    setupNavigationHandlers() {
        // Enhanced navigation for all buttons
        document.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                const section = item.getAttribute('data-section') || 
                              item.textContent.toLowerCase().trim();
                this.navigateToSection(section);
            });
        });
    }
    
    async navigateToSection(section) {
        // Update active state
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.remove('active');
        });
        
        const activeItem = document.querySelector(`[data-section="${section}"]`) ||
                          document.querySelector(`.nav-item:contains("${section}")`);
        if (activeItem) {
            activeItem.classList.add('active');
        }
        
        this.currentSection = section;
        
        // Load section content
        switch(section) {
            case 'dashboard':
                await this.showDashboard();
                break;
            case 'portfolio':
            case 'positions':
                await this.showPortfolio();
                break;
            case 'analysis':
                await this.showAnalysis();
                break;
            case 'markets':
                await this.showMarkets();
                break;
            case 'history':
                await this.showHistory();
                break;
            default:
                await this.showDashboard();
        }
    }
    
    async showDashboard() {
        const mainContent = document.querySelector('.main-content');
        if (!mainContent) return;
        
        mainContent.innerHTML = `
            <div class="dashboard-container">
                <div class="dashboard-header">
                    <h1>GoldGPT Trading Dashboard</h1>
                    <div class="dashboard-stats">
                        <div class="stat-card">
                            <div class="stat-label">Portfolio Value</div>
                            <div class="stat-value" id="portfolio-value">$0.00</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-label">Today's P&L</div>
                            <div class="stat-value" id="daily-pnl">$0.00</div>
                        </div>
                        <div class="stat-card">
                            <div class="stat-label">Open Positions</div>
                            <div class="stat-value" id="open-positions">0</div>
                        </div>
                    </div>
                </div>
                
                <div class="dashboard-grid">
                    <div class="chart-section">
                        <div class="section-title">Gold Price Chart</div>
                        <div id="tradingview-chart" style="height: 400px;"></div>
                    </div>
                    
                    <div class="quick-trade-section">
                        <div class="section-title">Quick Trade</div>
                        <div class="quick-trade-container">
                            <div class="price-display">
                                <div class="current-price">$<span id="current-price">2,085.40</span></div>
                                <div class="price-change positive">+$12.50 (+0.60%)</div>
                            </div>
                            <div class="trade-buttons">
                                <button class="btn-buy" onclick="enhancedDashboard.quickTrade('BUY')">
                                    <i class="fas fa-arrow-up"></i> BUY
                                </button>
                                <button class="btn-sell" onclick="enhancedDashboard.quickTrade('SELL')">
                                    <i class="fas fa-arrow-down"></i> SELL
                                </button>
                            </div>
                        </div>
                    </div>
                    
                    <div class="market-overview">
                        <div class="section-title">Market Overview</div>
                        <div id="market-overview-content">Loading...</div>
                    </div>
                    
                    <div class="ai-insights">
                        <div class="section-title">AI Insights</div>
                        <div id="ai-insights-content">Loading...</div>
                    </div>
                    
                    <div class="news-section">
                        <div class="section-title">Latest Market News</div>
                        <div id="news-content">Loading news...</div>
                    </div>
                </div>
            </div>
        `;
        
        // Initialize TradingView chart
        this.initializeTradingViewChart();
        
        // Load market data
        await this.loadMarketOverview();
        await this.loadAIInsights();
        await this.loadLatestNews();
        
        // Load real position data
        await this.updatePositionStats();
    }
    
    async showPortfolio() {
        const mainContent = document.querySelector('.main-content');
        if (!mainContent) return;
        
        this.showLoadingState('Loading portfolio data...');
        
        try {
            const portfolioData = await fetch('/api/enhanced/portfolio_analytics').then(r => r.json());
            
            mainContent.innerHTML = `
                <div class="portfolio-container">
                    <div class="portfolio-header">
                        <h1>Portfolio Management</h1>
                        <div class="portfolio-summary">
                            <div class="summary-card">
                                <div class="summary-label">Total Value</div>
                                <div class="summary-value">$${this.formatNumber(portfolioData.performance_metrics?.total_pnl + 10000 || 10000)}</div>
                            </div>
                            <div class="summary-card">
                                <div class="summary-label">Total P&L</div>
                                <div class="summary-value ${portfolioData.performance_metrics?.total_pnl >= 0 ? 'positive' : 'negative'}">
                                    ${portfolioData.performance_metrics?.total_pnl >= 0 ? '+' : ''}$${this.formatNumber(portfolioData.performance_metrics?.total_pnl || 0)}
                                </div>
                            </div>
                            <div class="summary-card">
                                <div class="summary-label">Win Rate</div>
                                <div class="summary-value">${portfolioData.performance_metrics?.win_rate?.toFixed(1) || 0}%</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="portfolio-grid">
                        <div class="performance-section">
                            <div class="section-title">Performance Metrics</div>
                            <div class="performance-metrics">
                                ${this.renderPerformanceMetrics(portfolioData.performance_metrics)}
                            </div>
                        </div>
                        
                        <div class="allocation-section">
                            <div class="section-title">Asset Allocation</div>
                            <div id="allocation-chart" style="height: 300px;"></div>
                        </div>
                        
                        <div class="risk-section">
                            <div class="section-title">Risk Analysis</div>
                            <div class="risk-metrics">
                                ${this.renderRiskMetrics(portfolioData.risk_metrics)}
                            </div>
                        </div>
                        
                        <div class="positions-section">
                            <div class="section-title">Open Positions</div>
                            <div id="positions-list">
                                ${await this.renderPositions()}
                            </div>
                        </div>
                    </div>
                </div>
            `;
            
            this.initializeAllocationChart(portfolioData.allocation);
            
        } catch (error) {
            this.showErrorState('Failed to load portfolio data');
        }
    }
    
    async showAnalysis() {
        const mainContent = document.querySelector('.main-content');
        if (!mainContent) return;
        
        this.showLoadingState('Loading advanced analysis...');
        
        try {
            const [technicalData, criticalData, newsData] = await Promise.all([
                fetch('/api/enhanced/technical_analysis/XAUUSD').then(r => r.json()),
                fetch('/api/enhanced/critical_market_data').then(r => r.json()),
                fetch('/api/news/latest?limit=20').then(r => r.json())
            ]);
            
            mainContent.innerHTML = `
                <div class="analysis-container">
                    <div class="analysis-header">
                        <h1>Advanced Market Analysis</h1>
                        <div class="analysis-controls">
                            <select id="analysis-symbol" onchange="enhancedDashboard.changeAnalysisSymbol(this.value)">
                                <option value="XAUUSD">Gold (XAU/USD)</option>
                                <option value="XAGUSD">Silver (XAG/USD)</option>
                                <option value="EURUSD">EUR/USD</option>
                            </select>
                            <button class="btn-refresh" onclick="enhancedDashboard.refreshAnalysis()">
                                <i class="fas fa-sync-alt"></i> Refresh
                            </button>
                        </div>
                    </div>
                    
                    <div class="analysis-grid">
                        <div class="technical-analysis-section">
                            <div class="section-title">Technical Analysis</div>
                            <div id="technical-analysis-content">
                                ${this.renderTechnicalAnalysis(technicalData)}
                            </div>
                        </div>
                        
                        <div class="critical-data-section">
                            <div class="section-title">Critical Market Data</div>
                            <div id="critical-data-content">
                                ${this.renderCriticalData(criticalData)}
                            </div>
                        </div>
                        
                        <div class="market-news-section">
                            <div class="section-title">Market News & Sentiment</div>
                            <div id="market-news-content">
                                ${this.renderMarketNews(newsData)}
                            </div>
                        </div>
                        
                        <div class="ai-prediction-section">
                            <div class="section-title">AI Predictions</div>
                            <div id="ai-predictions-content">
                                ${this.renderAIPredictions(technicalData)}
                            </div>
                        </div>
                    </div>
                </div>
            `;
            
        } catch (error) {
            this.showErrorState('Failed to load analysis data');
        }
    }
    
    async showMarkets() {
        const mainContent = document.querySelector('.main-content');
        if (!mainContent) return;
        
        this.showLoadingState('Loading market data...');
        
        try {
            const marketData = await fetch('/api/navigation/markets').then(r => r.json());
            
            mainContent.innerHTML = `
                <div class="markets-container">
                    <div class="markets-header">
                        <h1>Market Overview</h1>
                        <div class="market-stats">
                            <div class="stat-item">
                                <div class="stat-label">Markets Open</div>
                                <div class="stat-value">4</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-label">Active Symbols</div>
                                <div class="stat-value">${marketData.symbols?.length || 0}</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="markets-grid">
                        <div class="watchlist-section">
                            <div class="section-title">Watchlist</div>
                            <div class="watchlist-container">
                                ${this.renderWatchlist(marketData.market_data)}
                            </div>
                        </div>
                        
                        <div class="market-movers-section">
                            <div class="section-title">Top Movers</div>
                            <div class="market-movers">
                                ${this.renderMarketMovers(marketData.market_movers)}
                            </div>
                        </div>
                        
                        <div class="sector-performance-section">
                            <div class="section-title">Sector Performance</div>
                            <div class="sector-performance">
                                ${this.renderSectorPerformance(marketData.sector_performance)}
                            </div>
                        </div>
                        
                        <div class="market-heatmap-section">
                            <div class="section-title">Market Heatmap</div>
                            <div id="market-heatmap" style="height: 300px;">
                                ${this.renderMarketHeatmap(marketData.market_data)}
                            </div>
                        </div>
                    </div>
                </div>
            `;
            
        } catch (error) {
            this.showErrorState('Failed to load market data');
        }
    }
    
    async showHistory() {
        const mainContent = document.querySelector('.main-content');
        if (!mainContent) return;
        
        mainContent.innerHTML = `
            <div class="history-container">
                <div class="history-header">
                    <h1>Trade History</h1>
                    <div class="history-filters">
                        <select id="history-period">
                            <option value="1d">Today</option>
                            <option value="1w">This Week</option>
                            <option value="1m">This Month</option>
                            <option value="all">All Time</option>
                        </select>
                        <input type="text" placeholder="Search trades..." id="trade-search">
                    </div>
                </div>
                
                <div class="history-content">
                    <div class="trade-history-table">
                        <table>
                            <thead>
                                <tr>
                                    <th>Time</th>
                                    <th>Symbol</th>
                                    <th>Side</th>
                                    <th>Size</th>
                                    <th>Entry</th>
                                    <th>Exit</th>
                                    <th>P&L</th>
                                    <th>Status</th>
                                </tr>
                            </thead>
                            <tbody id="trade-history-body">
                                ${this.renderTradeHistory()}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        `;
    }
    
    // Rendering helper methods
    renderPerformanceMetrics(metrics) {
        if (!metrics) return '<div>No performance data available</div>';
        
        return `
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Daily P&L</div>
                    <div class="metric-value ${metrics.daily_pnl >= 0 ? 'positive' : 'negative'}">
                        ${metrics.daily_pnl >= 0 ? '+' : ''}$${this.formatNumber(metrics.daily_pnl)}
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Weekly P&L</div>
                    <div class="metric-value ${metrics.weekly_pnl >= 0 ? 'positive' : 'negative'}">
                        ${metrics.weekly_pnl >= 0 ? '+' : ''}$${this.formatNumber(metrics.weekly_pnl)}
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Profit Factor</div>
                    <div class="metric-value">${metrics.profit_factor?.toFixed(2) || 'N/A'}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Sharpe Ratio</div>
                    <div class="metric-value">${metrics.sharpe_ratio?.toFixed(2) || 'N/A'}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Max Drawdown</div>
                    <div class="metric-value negative">${metrics.max_drawdown?.toFixed(1) || 0}%</div>
                </div>
            </div>
        `;
    }
    
    renderRiskMetrics(metrics) {
        if (!metrics) return '<div>No risk data available</div>';
        
        return `
            <div class="risk-grid">
                <div class="risk-card">
                    <div class="risk-label">Current Exposure</div>
                    <div class="risk-value">${metrics.current_exposure?.toFixed(1) || 0}%</div>
                    <div class="risk-bar">
                        <div class="risk-fill" style="width: ${metrics.current_exposure || 0}%"></div>
                    </div>
                </div>
                <div class="risk-card">
                    <div class="risk-label">Risk Score</div>
                    <div class="risk-value">${metrics.risk_score?.toFixed(0) || 0}/100</div>
                    <div class="risk-bar">
                        <div class="risk-fill ${this.getRiskClass(metrics.risk_score)}" style="width: ${metrics.risk_score || 0}%"></div>
                    </div>
                </div>
                <div class="risk-card">
                    <div class="risk-label">VaR (95%)</div>
                    <div class="risk-value negative">$${this.formatNumber(Math.abs(metrics.var_95 || 0))}</div>
                </div>
            </div>
        `;
    }
    
    renderTechnicalAnalysis(data) {
        if (!data.indicators) return '<div>No technical data available</div>';
        
        const indicators = data.indicators;
        return `
            <div class="technical-grid">
                <div class="indicator-summary">
                    <div class="overall-signal ${data.overall_signal?.toLowerCase()}">
                        ${data.overall_signal || 'NEUTRAL'}
                    </div>
                    <div class="confidence-score">Confidence: ${Math.round((data.confidence || 0.5) * 100)}%</div>
                </div>
                
                <div class="indicators-list">
                    <div class="indicator-item">
                        <span class="indicator-name">RSI (14)</span>
                        <span class="indicator-value">${indicators.rsi?.value?.toFixed(1) || 'N/A'}</span>
                        <span class="indicator-signal ${indicators.rsi?.signal?.toLowerCase()}">${indicators.rsi?.signal || 'NEUTRAL'}</span>
                    </div>
                    <div class="indicator-item">
                        <span class="indicator-name">MACD</span>
                        <span class="indicator-value">${indicators.macd?.macd?.toFixed(2) || 'N/A'}</span>
                        <span class="indicator-signal ${indicators.macd?.trend?.toLowerCase()}">${indicators.macd?.trend || 'NEUTRAL'}</span>
                    </div>
                    <div class="indicator-item">
                        <span class="indicator-name">Bollinger Bands</span>
                        <span class="indicator-value">${indicators.bollinger_bands?.position || 'N/A'}</span>
                        <span class="indicator-signal">Position</span>
                    </div>
                </div>
                
                <div class="patterns-detected">
                    <h4>Patterns Detected</h4>
                    ${data.patterns?.map(pattern => `
                        <div class="pattern-item">
                            <span class="pattern-name">${pattern.name}</span>
                            <span class="pattern-confidence">${Math.round(pattern.confidence * 100)}%</span>
                            <span class="pattern-bias ${pattern.bias.toLowerCase()}">${pattern.bias}</span>
                        </div>
                    `).join('') || '<div>No patterns detected</div>'}
                </div>
            </div>
        `;
    }
    
    renderCriticalData(data) {
        if (!data.macro_indicators) return '<div>No critical data available</div>';
        
        return `
            <div class="critical-data-grid">
                <div class="macro-section">
                    <h4>Macro Indicators</h4>
                    ${Object.entries(data.macro_indicators).map(([key, value]) => `
                        <div class="macro-item">
                            <span class="macro-name">${key.toUpperCase()}</span>
                            <span class="macro-value">${value.value}</span>
                            <span class="macro-change ${value.change >= 0 ? 'positive' : 'negative'}">
                                ${value.change >= 0 ? '+' : ''}${value.change}
                            </span>
                        </div>
                    `).join('')}
                </div>
                
                <div class="positioning-section">
                    <h4>CFTC Positioning</h4>
                    <div class="positioning-item">
                        <span>Commercial Net:</span>
                        <span class="positioning-value">${this.formatNumber(data.cftc_positioning?.commercial_net?.value || 0)}</span>
                    </div>
                    <div class="positioning-item">
                        <span>Large Specs Net:</span>
                        <span class="positioning-value">${this.formatNumber(data.cftc_positioning?.large_specs_net?.value || 0)}</span>
                    </div>
                </div>
            </div>
        `;
    }
    
    renderMarketNews(data) {
        // Handle our actual API response structure
        if (!data.success || !data.news || data.news.length === 0) {
            return '<div class="no-news">No news available at this time</div>';
        }
        
        return `
            <div class="news-list">
                ${data.news.slice(0, 5).map(article => {
                    // Map sentiment score to readable labels
                    const getSentimentLabel = (score) => {
                        if (score > 0.1) return 'Bullish';
                        if (score < -0.1) return 'Bearish';
                        return 'Neutral';
                    };
                    
                    const getImpactLabel = (score) => {
                        if (score > 0.7) return 'High';
                        if (score > 0.4) return 'Medium';
                        return 'Low';
                    };
                    
                    const sentimentClass = article.sentiment_score > 0.1 ? 'bullish' : 
                                         article.sentiment_score < -0.1 ? 'bearish' : 'neutral';
                    
                    return `
                        <div class="news-item">
                            <div class="news-header">
                                <span class="news-title">${article.title}</span>
                                <span class="news-time">${article.time_ago || this.formatTimeAgo(article.published_date)}</span>
                            </div>
                            <div class="news-meta">
                                <span class="news-source">${article.source}</span>
                                <span class="news-sentiment ${sentimentClass}">
                                    ${getSentimentLabel(article.sentiment_score || 0)}
                                </span>
                                <span class="news-impact ${getImpactLabel(article.impact_score || 0).toLowerCase()}">
                                    ${getImpactLabel(article.impact_score || 0)} Impact
                                </span>
                                <span class="news-relevance">
                                    Relevance: ${Math.round((article.gold_relevance_score || 0) * 100)}%
                                </span>
                            </div>
                            ${article.keywords && article.keywords.length > 0 ? `
                                <div class="news-keywords">
                                    ${article.keywords.slice(0, 3).map(keyword => 
                                        `<span class="keyword-tag">${keyword}</span>`
                                    ).join('')}
                                </div>
                            ` : ''}
                        </div>
                    `;
                }).join('')}
            </div>
            <div class="news-summary-stats">
                <div class="stat-item">
                    <span class="stat-label">Total Articles:</span>
                    <span class="stat-value">${data.count}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Last Updated:</span>
                    <span class="stat-value">${this.formatTimeAgo(data.last_updated)}</span>
                </div>
            </div>
        `;
    }
    
    renderAIPredictions(data) {
        return `
            <div class="ai-predictions">
                <div class="prediction-summary">
                    <div class="prediction-signal ${data.overall_signal?.toLowerCase()}">
                        ${data.overall_signal || 'ANALYZING'}
                    </div>
                    <div class="prediction-confidence">
                        AI Confidence: ${Math.round((data.confidence || 0.5) * 100)}%
                    </div>
                </div>
                
                <div class="prediction-details">
                    <div class="prediction-item">
                        <span>Short-term (1H):</span>
                        <span class="prediction-value">+0.8%</span>
                    </div>
                    <div class="prediction-item">
                        <span>Medium-term (4H):</span>
                        <span class="prediction-value">+1.5%</span>
                    </div>
                    <div class="prediction-item">
                        <span>Long-term (1D):</span>
                        <span class="prediction-value">+2.3%</span>
                    </div>
                </div>
            </div>
        `;
    }
    
    // Utility methods
    formatNumber(num) {
        return new Intl.NumberFormat('en-US', {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        }).format(num);
    }
    
    formatTimeAgo(timestamp) {
        const date = new Date(timestamp);
        const now = new Date();
        const diff = now - date;
        
        const minutes = Math.floor(diff / 60000);
        const hours = Math.floor(diff / 3600000);
        
        if (minutes < 60) return `${minutes}m ago`;
        if (hours < 24) return `${hours}h ago`;
        return date.toLocaleDateString();
    }
    
    getRiskClass(score) {
        if (score < 30) return 'low';
        if (score < 60) return 'medium';
        return 'high';
    }
    
    showLoadingState(message) {
        const mainContent = document.querySelector('.main-content');
        if (mainContent) {
            mainContent.innerHTML = `
                <div class="loading-state">
                    <div class="loading-spinner"></div>
                    <div class="loading-message">${message}</div>
                </div>
            `;
        }
    }
    
    showErrorState(message) {
        const mainContent = document.querySelector('.main-content');
        if (mainContent) {
            mainContent.innerHTML = `
                <div class="error-state">
                    <div class="error-icon">⚠️</div>
                    <div class="error-message">${message}</div>
                    <button class="retry-button" onclick="location.reload()">Retry</button>
                </div>
            `;
        }
    }
    
    setupSocketListeners() {
        this.socket.on('price_update', (data) => {
            this.handlePriceUpdate(data);
        });
        
        this.socket.on('trade_update', (data) => {
            this.handleTradeUpdate(data);
        });
    }
    
    handlePriceUpdate(data) {
        // Update price displays
        const priceElement = document.getElementById('current-price');
        if (priceElement) {
            priceElement.textContent = data.price.toFixed(2);
            priceElement.parentElement.classList.add('price-flash');
            setTimeout(() => {
                priceElement.parentElement.classList.remove('price-flash');
            }, 500);
        }
    }
    
    async quickTrade(side) {
        try {
            const response = await fetch('/api/trade', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    symbol: 'XAUUSD',
                    side: side,
                    amount: 0.1,
                    type: 'market'
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.showNotification(`${side} trade executed successfully`, 'success');
                // Refresh portfolio data
                if (this.currentSection === 'portfolio') {
                    await this.showPortfolio();
                }
            } else {
                this.showNotification(result.error || 'Trade failed', 'error');
            }
        } catch (error) {
            this.showNotification('Trade execution failed', 'error');
        }
    }
    
    showNotification(message, type) {
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.remove();
        }, 3000);
    }
    
    async loadInitialData() {
        // Load initial dashboard data
        await this.showDashboard();
    }
    
    startAutoUpdates() {
        // Auto-refresh every 30 seconds
        this.updateIntervals.main = setInterval(() => {
            if (this.currentSection !== 'history') {
                this.refreshCurrentSection();
            }
        }, 30000);
    }
    
    async refreshCurrentSection() {
        switch(this.currentSection) {
            case 'dashboard':
                await this.updatePositionStats();
                break;
            case 'portfolio':
                await this.showPortfolio();
                break;
            case 'analysis':
                await this.showAnalysis();
                break;
            case 'markets':
                await this.showMarkets();
                break;
        }
    }
    
    // Additional helper methods for rendering
    renderWatchlist(marketData) {
        if (!marketData) return '<div>No market data available</div>';
        
        return Object.entries(marketData).map(([symbol, data]) => `
            <div class="watchlist-item" onclick="enhancedDashboard.selectSymbol('${symbol}')">
                <div class="symbol-info">
                    <span class="symbol-name">${symbol}</span>
                    <span class="symbol-price">$${data.price}</span>
                </div>
                <div class="symbol-change ${data.change >= 0 ? 'positive' : 'negative'}">
                    ${data.change >= 0 ? '+' : ''}${data.change}%
                </div>
            </div>
        `).join('');
    }
    
    renderMarketMovers(movers) {
        if (!movers) return '<div>No movers data</div>';
        
        return movers.map(mover => `
            <div class="mover-item">
                <span class="mover-symbol">${mover.symbol}</span>
                <span class="mover-price">$${mover.price}</span>
                <span class="mover-change ${mover.change >= 0 ? 'positive' : 'negative'}">
                    ${mover.change >= 0 ? '+' : ''}${mover.change}%
                </span>
            </div>
        `).join('');
    }
    
    renderSectorPerformance(sectors) {
        if (!sectors) return '<div>No sector data</div>';
        
        return Object.entries(sectors).map(([sector, performance]) => `
            <div class="sector-item">
                <span class="sector-name">${sector}</span>
                <span class="sector-performance ${performance >= 0 ? 'positive' : 'negative'}">
                    ${performance >= 0 ? '+' : ''}${performance}%
                </span>
            </div>
        `).join('');
    }
    
    renderTradeHistory() {
        // Return empty for now - user hasn't made any trades
        return '<tr><td colspan="8" style="text-align: center; color: #666; padding: 20px;">No closed trades yet</td></tr>';
    }
    
    async renderPositions() {
        try {
            // Fetch real active signals from enhanced signals API
            const response = await fetch('/api/enhanced-signals/active');
            const data = await response.json();
            
            if (!data.success || !data.active_signals || data.active_signals.length === 0) {
                return `
                    <div class="positions-list">
                        <div class="empty-state">
                            <div class="empty-message">No open positions</div>
                        </div>
                    </div>
                `;
            }
            
            // Render real active signals
            const positionsHtml = data.active_signals.map(signal => {
                const unrealizedPnL = signal.unrealized_pnl || 0;
                const pnlClass = unrealizedPnL >= 0 ? 'positive' : 'negative';
                const sideClass = signal.signal_type === 'buy' ? 'buy' : 'sell';
                const sideText = signal.signal_type === 'buy' ? 'LONG' : 'SHORT';
                
                return `
                    <div class="position-item" data-signal-id="${signal.id}">
                        <div class="position-symbol">XAUUSD</div>
                        <div class="position-side ${sideClass}">${sideText}</div>
                        <div class="position-entry">Entry: $${signal.entry_price?.toFixed(2) || 'N/A'}</div>
                        <div class="position-current">Current: $${signal.current_price?.toFixed(2) || 'N/A'}</div>
                        <div class="position-targets">
                            <small>TP: $${signal.target_price?.toFixed(2) || 'N/A'} | SL: $${signal.stop_loss?.toFixed(2) || 'N/A'}</small>
                        </div>
                        <div class="position-pnl ${pnlClass}">
                            ${unrealizedPnL >= 0 ? '+' : ''}$${Math.abs(unrealizedPnL).toFixed(2)}
                        </div>
                        <div class="position-actions">
                            <button class="btn-close-position" onclick="enhancedDashboard.closePosition(${signal.id})">
                                Close
                            </button>
                        </div>
                    </div>
                `;
            }).join('');
            
            return `
                <div class="positions-list">
                    ${positionsHtml}
                </div>
            `;
            
        } catch (error) {
            console.error('Error rendering positions:', error);
            return `
                <div class="positions-list">
                    <div class="empty-state">
                        <div class="empty-message">Error loading positions</div>
                    </div>
                </div>
            `;
        }
    }
    
    async closePosition(signalId) {
        try {
            const response = await fetch(`/api/enhanced-signals/close/${signalId}`, {
                method: 'POST'
            });
            
            const data = await response.json();
            
            if (data.success) {
                // Refresh the portfolio/positions view
                if (this.currentSection === 'portfolio') {
                    await this.showPortfolio();
                }
                await this.updatePositionStats();
                
                alert('Position closed successfully');
            } else {
                alert('Failed to close position: ' + (data.error || 'Unknown error'));
            }
        } catch (error) {
            console.error('Error closing position:', error);
            alert('Error closing position');
        }
    }
    
    initializeTradingViewChart() {
        // Initialize TradingView widget if container exists
        const container = document.getElementById('tradingview-chart');
        if (container && typeof TradingView !== 'undefined') {
            new TradingView.widget({
                width: '100%',
                height: 400,
                symbol: 'OANDA:XAUUSD',
                interval: '15',
                timezone: 'Etc/UTC',
                theme: 'dark',
                style: '1',
                locale: 'en',
                toolbar_bg: '#f1f3f6',
                enable_publishing: false,
                allow_symbol_change: true,
                container_id: 'tradingview-chart'
            });
        }
    }
    
    initializeAllocationChart(allocation) {
        // Initialize allocation chart if container exists
        const container = document.getElementById('allocation-chart');
        if (container && allocation) {
            // Simple pie chart representation
            const total = Object.values(allocation).reduce((a, b) => a + b, 0);
            let html = '<div class="allocation-items">';
            
            Object.entries(allocation).forEach(([asset, percentage]) => {
                html += `
                    <div class="allocation-item">
                        <div class="allocation-color"></div>
                        <span class="allocation-asset">${asset}</span>
                        <span class="allocation-percentage">${percentage.toFixed(1)}%</span>
                    </div>
                `;
            });
            
            html += '</div>';
            container.innerHTML = html;
        }
    }
    
    async loadLatestNews() {
        const newsContainer = document.getElementById('news-content');
        if (!newsContainer) return;
        
        try {
            newsContainer.innerHTML = '<div class="loading">Loading latest news...</div>';
            
            const response = await fetch('/api/news/latest?limit=10');
            const newsData = await response.json();
            
            if (newsData.success && newsData.news && newsData.news.length > 0) {
                newsContainer.innerHTML = this.renderMarketNews(newsData);
            } else {
                newsContainer.innerHTML = `
                    <div class="no-news">
                        <div class="no-news-message">No news available</div>
                        <button onclick="enhancedDashboard.refreshNews()" class="btn-refresh">
                            Refresh News
                        </button>
                    </div>
                `;
            }
        } catch (error) {
            console.error('Failed to load news:', error);
            newsContainer.innerHTML = `
                <div class="news-error">
                    <div class="error-message">Failed to load news</div>
                    <button onclick="enhancedDashboard.refreshNews()" class="btn-refresh">
                        Try Again
                    </button>
                </div>
            `;
        }
    }
    
    async updatePositionStats() {
        try {
            // Fetch active enhanced signals
            const activeResponse = await fetch('/api/enhanced-signals/active');
            const activeData = await activeResponse.json();
            
            // Fetch performance data
            const performanceResponse = await fetch('/api/enhanced-signals/performance');
            const performanceData = await performanceResponse.json();
            
            let openPositions = 0;
            let totalPnL = 0;
            let dailyPnL = 0;
            
            // Count active signals
            if (activeData.success && activeData.active_signals) {
                openPositions = activeData.active_signals.length;
            }
            
            // Calculate P&L from performance data
            if (performanceData.success) {
                const perf = performanceData.performance;
                if (perf.recent_7_days && perf.recent_7_days.avg_pnl) {
                    dailyPnL = perf.recent_7_days.avg_pnl;
                }
                if (perf.avg_profit_loss_pct) {
                    totalPnL = perf.avg_profit_loss_pct * 100; // Convert to basis for portfolio
                }
            }
            
            // Update DOM elements
            const openPositionsEl = document.getElementById('open-positions');
            const dailyPnLEl = document.getElementById('daily-pnl');
            const portfolioValueEl = document.getElementById('portfolio-value');
            
            if (openPositionsEl) {
                openPositionsEl.textContent = openPositions;
            }
            
            if (dailyPnLEl) {
                const pnlText = dailyPnL >= 0 ? `+$${Math.abs(dailyPnL).toFixed(2)}` : `-$${Math.abs(dailyPnL).toFixed(2)}`;
                dailyPnLEl.textContent = pnlText;
                dailyPnLEl.className = dailyPnL >= 0 ? 'stat-value positive' : 'stat-value negative';
            }
            
            if (portfolioValueEl) {
                const basePortfolio = 10000; // Starting value
                const currentValue = basePortfolio + totalPnL;
                portfolioValueEl.textContent = `$${currentValue.toFixed(2)}`;
            }
            
            console.log(`📊 Position stats updated: ${openPositions} open, $${dailyPnL.toFixed(2)} daily P&L`);
            
        } catch (error) {
            console.error('Failed to update position stats:', error);
            // Set defaults in case of error
            const openPositionsEl = document.getElementById('open-positions');
            const dailyPnLEl = document.getElementById('daily-pnl');
            const portfolioValueEl = document.getElementById('portfolio-value');
            
            if (openPositionsEl) openPositionsEl.textContent = '0';
            if (dailyPnLEl) {
                dailyPnLEl.textContent = '$0.00';
                dailyPnLEl.className = 'stat-value';
            }
            if (portfolioValueEl) portfolioValueEl.textContent = '$10,000.00';
        }
    }
    
    async refreshNews() {
        console.log('🔄 Refreshing news...');
        
        // Trigger news aggregation
        try {
            const response = await fetch('/api/news/aggregate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            const result = await response.json();
            console.log('📰 News aggregation result:', result);
            
            // Reload news after aggregation
            setTimeout(() => {
                this.loadLatestNews();
            }, 2000);
            
        } catch (error) {
            console.error('Failed to refresh news:', error);
        }
    }
}

// Create global instance for component loader
window.enhancedDashboard = new EnhancedDashboardManager();

// Add init method for component loader compatibility
window.enhancedDashboard.init = function() {
    this.initializeEnhancedDashboard();
    return Promise.resolve();
};

// Legacy global reference
let enhancedDashboard = window.enhancedDashboard;

// Auto-initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Enhanced Dashboard Initialized');
});

console.log('📊 Enhanced Dashboard Manager loaded successfully');
