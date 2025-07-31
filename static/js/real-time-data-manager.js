/**
 * Real-Time Data Manager for GoldGPT Frontend
 * Replaces all hardcoded data with live API calls
 */

class RealTimeDataManager {
    constructor() {
        this.apiBase = '/api/realtime';
        this.updateIntervals = new Map();
        this.cache = new Map();
        this.cacheTimeout = 30000; // 30 seconds
        
        // Data update frequencies (in milliseconds)
        this.updateFrequencies = {
            price: 5000,        // 5 seconds
            sentiment: 30000,   // 30 seconds
            technical: 60000,   // 1 minute
            watchlist: 10000,   // 10 seconds
            news: 300000        // 5 minutes
        };
        
        // Event listeners for data updates
        this.listeners = new Map();
        
        console.log('🔄 Real-Time Data Manager initialized');
    }
    
    /**
     * Start real-time data updates
     */
    async startRealTimeUpdates() {
        console.log('🚀 Starting real-time data updates...');
        
        try {
            // Start price updates for watchlist
            this.startPriceUpdates();
            
            // Start sentiment updates
            this.startSentimentUpdates();
            
            // Start technical analysis updates
            this.startTechnicalUpdates();
            
            // Start news updates
            this.startNewsUpdates();
            
            console.log('✅ All real-time updates started');
            
        } catch (error) {
            console.error('❌ Failed to start real-time updates:', error);
        }
    }
    
    /**
     * Stop all real-time updates
     */
    stopRealTimeUpdates() {
        console.log('⏹️ Stopping real-time data updates...');
        
        this.updateIntervals.forEach((intervalId, key) => {
            clearInterval(intervalId);
        });
        this.updateIntervals.clear();
        
        console.log('✅ All real-time updates stopped');
    }
    
    /**
     * Start price updates for watchlist symbols
     */
    startPriceUpdates() {
        const updatePrices = async () => {
            try {
                const response = await fetch(`${this.apiBase}/watchlist`);
                const result = await response.json();
                
                if (result.success && result.data) {
                    this.cache.set('watchlist_prices', {
                        data: result.data,
                        timestamp: Date.now()
                    });
                    
                    // Emit price update event
                    this.emitEvent('priceUpdate', result.data);
                    
                    // Update UI elements
                    this.updatePriceDisplay(result.data);
                }
            } catch (error) {
                console.warn('Price update failed:', error);
            }
        };
        
        // Initial update
        updatePrices();
        
        // Schedule periodic updates
        const intervalId = setInterval(updatePrices, this.updateFrequencies.price);
        this.updateIntervals.set('prices', intervalId);
    }
    
    /**
     * Start sentiment analysis updates
     */
    startSentimentUpdates() {
        const updateSentiment = async () => {
            try {
                const symbol = window.app?.currentSymbol || 'XAUUSD';
                const response = await fetch(`${this.apiBase}/sentiment/${symbol}`);
                const result = await response.json();
                
                if (result.success && result.data) {
                    this.cache.set('sentiment_data', {
                        data: result.data,
                        timestamp: Date.now()
                    });
                    
                    // Emit sentiment update event
                    this.emitEvent('sentimentUpdate', result.data);
                    
                    // Update sentiment display
                    this.updateSentimentDisplay(result.data);
                }
            } catch (error) {
                console.warn('Sentiment update failed:', error);
            }
        };
        
        // Initial update
        updateSentiment();
        
        // Schedule periodic updates
        const intervalId = setInterval(updateSentiment, this.updateFrequencies.sentiment);
        this.updateIntervals.set('sentiment', intervalId);
    }
    
    /**
     * Start technical analysis updates
     */
    startTechnicalUpdates() {
        const updateTechnical = async () => {
            try {
                const symbol = window.app?.currentSymbol || 'XAUUSD';
                const response = await fetch(`${this.apiBase}/technical/${symbol}`);
                const result = await response.json();
                
                if (result.success && result.data) {
                    this.cache.set('technical_data', {
                        data: result.data,
                        timestamp: Date.now()
                    });
                    
                    // Emit technical update event
                    this.emitEvent('technicalUpdate', result.data);
                    
                    // Update technical indicators display
                    this.updateTechnicalDisplay(result.data);
                }
            } catch (error) {
                console.warn('Technical update failed:', error);
            }
        };
        
        // Initial update
        updateTechnical();
        
        // Schedule periodic updates
        const intervalId = setInterval(updateTechnical, this.updateFrequencies.technical);
        this.updateIntervals.set('technical', intervalId);
    }
    
    /**
     * Start news updates
     */
    startNewsUpdates() {
        const updateNews = async () => {
            try {
                const response = await fetch('/api/news/latest?limit=10');
                const result = await response.json();
                
                if (result.success && result.news) {
                    this.cache.set('news_data', {
                        data: result.news,
                        timestamp: Date.now()
                    });
                    
                    // Emit news update event
                    this.emitEvent('newsUpdate', result.news);
                    
                    // Update news display
                    this.updateNewsDisplay(result.news);
                }
            } catch (error) {
                console.warn('News update failed:', error);
            }
        };
        
        // Initial update
        updateNews();
        
        // Schedule periodic updates
        const intervalId = setInterval(updateNews, this.updateFrequencies.news);
        this.updateIntervals.set('news', intervalId);
    }
    
    /**
     * Get cached data or fetch from API
     */
    async getData(type, symbol = null) {
        const cacheKey = symbol ? `${type}_${symbol}` : type;
        
        // Check cache first
        const cached = this.cache.get(cacheKey);
        if (cached && (Date.now() - cached.timestamp) < this.cacheTimeout) {
            return cached.data;
        }
        
        // Fetch from API
        try {
            let url = `${this.apiBase}/${type}`;
            if (symbol) {
                url += `/${symbol}`;
            }
            
            const response = await fetch(url);
            const result = await response.json();
            
            if (result.success) {
                this.cache.set(cacheKey, {
                    data: result.data,
                    timestamp: Date.now()
                });
                return result.data;
            }
        } catch (error) {
            console.warn(`Failed to fetch ${type} data:`, error);
        }
        
        return null;
    }
    
    /**
     * Update price display in UI
     */
    updatePriceDisplay(priceData) {
        priceData.forEach(symbolData => {
            const symbol = symbolData.symbol;
            
            // Update main price display
            const priceElement = document.getElementById('current-price');
            if (priceElement && (window.app?.currentSymbol === symbol || symbol === 'XAUUSD')) {
                priceElement.textContent = `$${symbolData.price.toFixed(2)}`;
                
                // Update price change
                const changeElement = document.getElementById('price-change');
                if (changeElement) {
                    const change = symbolData.change || 0;
                    const changePercent = symbolData.change_percent || 0;
                    
                    changeElement.textContent = `${change >= 0 ? '+' : ''}${change.toFixed(2)} (${changePercent.toFixed(2)}%)`;
                    changeElement.className = `price-change ${change >= 0 ? 'positive' : 'negative'}`;
                }
            }
            
            // Update watchlist items
            const watchlistItem = document.querySelector(`[data-symbol="${symbol}"]`);
            if (watchlistItem) {
                const priceSpan = watchlistItem.querySelector('.watchlist-price');
                const changeSpan = watchlistItem.querySelector('.watchlist-change');
                
                if (priceSpan) {
                    priceSpan.textContent = `$${symbolData.price.toFixed(2)}`;
                }
                
                if (changeSpan) {
                    const change = symbolData.change || 0;
                    const changePercent = symbolData.change_percent || 0;
                    
                    changeSpan.textContent = `${change >= 0 ? '+' : ''}${changePercent.toFixed(2)}%`;
                    changeSpan.className = `watchlist-change ${change >= 0 ? 'positive' : 'negative'}`;
                }
            }
        });
    }
    
    /**
     * Update sentiment display in UI
     */
    updateSentimentDisplay(sentimentData) {
        // Update overall sentiment
        const overallSentimentElement = document.getElementById('overall-sentiment');
        if (overallSentimentElement && sentimentData.overall) {
            const sentiment = sentimentData.overall.sentiment;
            const confidence = (sentimentData.overall.confidence * 100).toFixed(0);
            
            overallSentimentElement.textContent = sentiment.toUpperCase();
            overallSentimentElement.className = `sentiment ${sentiment}`;
            
            // Update confidence
            const confidenceElement = document.getElementById('sentiment-confidence');
            if (confidenceElement) {
                confidenceElement.textContent = `${confidence}%`;
            }
        }
        
        // Update timeframe sentiments
        if (sentimentData.timeframes) {
            Object.entries(sentimentData.timeframes).forEach(([timeframe, data]) => {
                const elementId = `sentiment-${timeframe}`;
                const element = document.getElementById(elementId);
                
                if (element) {
                    element.textContent = data.sentiment.toUpperCase();
                    element.className = `sentiment-indicator ${data.sentiment}`;
                }
                
                // Update confidence bars
                const confidenceBar = document.getElementById(`confidence-${timeframe}`);
                if (confidenceBar) {
                    const confidencePercent = (data.confidence * 100).toFixed(0);
                    confidenceBar.style.width = `${confidencePercent}%`;
                    confidenceBar.setAttribute('data-confidence', confidencePercent);
                }
            });
        }
    }
    
    /**
     * Update technical indicators display
     */
    updateTechnicalDisplay(technicalData) {
        // Update RSI
        if (technicalData.rsi) {
            const rsiValue = document.getElementById('rsi-value');
            const rsiSignal = document.getElementById('rsi-signal');
            
            if (rsiValue) {
                rsiValue.textContent = technicalData.rsi.value.toFixed(2);
            }
            
            if (rsiSignal) {
                rsiSignal.textContent = technicalData.rsi.signal.toUpperCase();
                rsiSignal.className = `signal ${technicalData.rsi.signal}`;
            }
        }
        
        // Update MACD
        if (technicalData.macd) {
            const macdValue = document.getElementById('macd-value');
            const macdSignal = document.getElementById('macd-signal');
            
            if (macdValue) {
                macdValue.textContent = technicalData.macd.value.toFixed(4);
            }
            
            if (macdSignal) {
                macdSignal.textContent = technicalData.macd.signal.toUpperCase();
                macdSignal.className = `signal ${technicalData.macd.signal}`;
            }
        }
        
        // Update Bollinger Bands
        if (technicalData.bollinger_bands) {
            const bbUpper = document.getElementById('bb-upper');
            const bbMiddle = document.getElementById('bb-middle');
            const bbLower = document.getElementById('bb-lower');
            
            if (bbUpper) bbUpper.textContent = `$${technicalData.bollinger_bands.upper.toFixed(2)}`;
            if (bbMiddle) bbMiddle.textContent = `$${technicalData.bollinger_bands.middle.toFixed(2)}`;
            if (bbLower) bbLower.textContent = `$${technicalData.bollinger_bands.lower.toFixed(2)}`;
        }
        
        // Update Moving Averages
        if (technicalData.moving_averages) {
            const ma20 = document.getElementById('ma20-value');
            const ma50 = document.getElementById('ma50-value');
            const trend = document.getElementById('ma-trend');
            
            if (ma20) ma20.textContent = `$${technicalData.moving_averages.ma20.toFixed(2)}`;
            if (ma50) ma50.textContent = `$${technicalData.moving_averages.ma50.toFixed(2)}`;
            if (trend) {
                trend.textContent = technicalData.moving_averages.trend.toUpperCase();
                trend.className = `trend ${technicalData.moving_averages.trend}`;
            }
        }
    }
    
    /**
     * Update news display
     */
    updateNewsDisplay(newsData) {
        const newsContainer = document.getElementById('news-articles');
        if (!newsContainer) return;
        
        // Clear existing news
        newsContainer.innerHTML = '';
        
        // Add new articles
        newsData.forEach(article => {
            const articleElement = document.createElement('div');
            articleElement.className = 'news-article';
            
            const sentimentClass = this.getSentimentClass(article.sentiment_score);
            
            articleElement.innerHTML = `
                <div class="news-header">
                    <span class="news-source">${article.source}</span>
                    <span class="news-time">${article.time_ago}</span>
                </div>
                <h4 class="news-title">${article.title}</h4>
                <div class="news-metrics">
                    <span class="sentiment-indicator ${sentimentClass}">
                        ${this.getSentimentLabel(article.sentiment_score)}
                    </span>
                    <span class="impact-score">
                        Impact: ${(article.impact_score * 100).toFixed(0)}%
                    </span>
                </div>
            `;
            
            newsContainer.appendChild(articleElement);
        });
    }
    
    /**
     * Get sentiment CSS class from score
     */
    getSentimentClass(score) {
        if (score > 0.2) return 'bullish';
        if (score < -0.2) return 'bearish';
        return 'neutral';
    }
    
    /**
     * Get sentiment label from score
     */
    getSentimentLabel(score) {
        if (score > 0.2) return 'Bullish';
        if (score < -0.2) return 'Bearish';
        return 'Neutral';
    }
    
    /**
     * Add event listener for data updates
     */
    on(event, callback) {
        if (!this.listeners.has(event)) {
            this.listeners.set(event, []);
        }
        this.listeners.get(event).push(callback);
    }
    
    /**
     * Remove event listener
     */
    off(event, callback) {
        if (this.listeners.has(event)) {
            const listeners = this.listeners.get(event);
            const index = listeners.indexOf(callback);
            if (index > -1) {
                listeners.splice(index, 1);
            }
        }
    }
    
    /**
     * Emit event to listeners
     */
    emitEvent(event, data) {
        if (this.listeners.has(event)) {
            this.listeners.get(event).forEach(callback => {
                try {
                    callback(data);
                } catch (error) {
                    console.error(`Event listener error for ${event}:`, error);
                }
            });
        }
    }
    
    /**
     * Replace hardcoded portfolio data with real data
     */
    async updatePortfolioData() {
        try {
            const response = await fetch('/api/portfolio');
            const result = await response.json();
            
            if (result.success && window.app) {
                // Replace hardcoded portfolio data
                window.app.portfolioData = result.portfolio;
                
                // Update portfolio display
                this.updatePortfolioDisplay(result.portfolio);
                
                // Emit portfolio update event
                this.emitEvent('portfolioUpdate', result.portfolio);
            }
        } catch (error) {
            console.warn('Portfolio update failed:', error);
        }
    }
    
    /**
     * Update portfolio display in UI
     */
    updatePortfolioDisplay(portfolioData) {
        // Update account balance
        const balanceElement = document.getElementById('account-balance');
        if (balanceElement && portfolioData.account) {
            balanceElement.textContent = `$${portfolioData.account.balance.toFixed(2)}`;
        }
        
        // Update equity
        const equityElement = document.getElementById('account-equity');
        if (equityElement && portfolioData.account) {
            equityElement.textContent = `$${portfolioData.account.equity.toFixed(2)}`;
        }
        
        // Update profit/loss
        const plElement = document.getElementById('account-pl');
        if (plElement && portfolioData.account) {
            const pl = portfolioData.account.profit_loss || 0;
            plElement.textContent = `${pl >= 0 ? '+' : ''}$${pl.toFixed(2)}`;
            plElement.className = `account-pl ${pl >= 0 ? 'positive' : 'negative'}`;
        }
        
        // Update open positions
        const positionsContainer = document.getElementById('open-positions');
        if (positionsContainer && portfolioData.trades) {
            positionsContainer.innerHTML = '';
            
            portfolioData.trades.forEach(trade => {
                const tradeElement = document.createElement('div');
                tradeElement.className = 'position-item';
                
                const pl = trade.profit_loss || 0;
                
                tradeElement.innerHTML = `
                    <div class="position-symbol">${trade.symbol}</div>
                    <div class="position-type">${trade.type.toUpperCase()}</div>
                    <div class="position-amount">${trade.amount}</div>
                    <div class="position-pl ${pl >= 0 ? 'positive' : 'negative'}">
                        ${pl >= 0 ? '+' : ''}$${pl.toFixed(2)}
                    </div>
                    <button class="close-position-btn" data-trade-id="${trade.id}">
                        ✕
                    </button>
                `;
                
                positionsContainer.appendChild(tradeElement);
            });
        }
    }
    
    /**
     * Initialize real-time data manager
     */
    async initialize() {
        console.log('🔄 Initializing Real-Time Data Manager...');
        
        try {
            // Check system status
            const statusResponse = await fetch(`${this.apiBase}/status`);
            const statusResult = await statusResponse.json();
            
            if (statusResult.success) {
                console.log('📊 Real-time data status:', statusResult.status);
                
                // Update initial data
                await this.updatePortfolioData();
                
                // Start real-time updates
                await this.startRealTimeUpdates();
                
                console.log('✅ Real-Time Data Manager initialized successfully');
                return true;
            } else {
                console.warn('⚠️ Real-time data system not fully operational');
                return false;
            }
        } catch (error) {
            console.error('❌ Failed to initialize Real-Time Data Manager:', error);
            return false;
        }
    }
}

// Create global instance
window.realTimeDataManager = new RealTimeDataManager();

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', async () => {
    console.log('🚀 Starting Real-Time Data Manager initialization...');
    
    // Wait a bit for other systems to initialize
    setTimeout(async () => {
        const initialized = await window.realTimeDataManager.initialize();
        
        if (initialized) {
            console.log('✅ Real-time data replacement system active');
        } else {
            console.warn('⚠️ Falling back to existing data systems');
        }
    }, 2000);
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (window.realTimeDataManager) {
        window.realTimeDataManager.stopRealTimeUpdates();
    }
});

// Export for manual control and testing
window.RealTimeDataManager = RealTimeDataManager;

console.log('📊 Real-Time Data Manager module loaded');
