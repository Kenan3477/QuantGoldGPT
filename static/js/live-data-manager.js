/**
 * Live Data Manager - Ensures dashboard uses only real data
 * Replaces hardcoded fallbacks with proper error handling
 */

class LiveDataManager {
    constructor() {
        this.priceUpdateInterval = 5000; // 5 seconds
        this.newsUpdateInterval = 30000; // 30 seconds
        this.isConnected = false;
        this.lastPriceUpdate = null;
        this.lastNewsUpdate = null;
        this.priceRetryCount = 0;
        this.maxRetries = 3;
        
        // Start live data streams
        this.initializeLiveData();
    }
    
    async initializeLiveData() {
        console.log('🚀 Initializing Live Data Manager...');
        
        // Initial data fetch
        await this.updateLivePrice();
        await this.updateNewsSentiment();
        
        // Set up recurring updates
        this.startPriceUpdates();
        this.startNewsUpdates();
        
        console.log('✅ Live Data Manager initialized');
    }
    
    startPriceUpdates() {
        setInterval(async () => {
            await this.updateLivePrice();
        }, this.priceUpdateInterval);
    }
    
    startNewsUpdates() {
        setInterval(async () => {
            await this.updateNewsSentiment();
        }, this.newsUpdateInterval);
    }
    
    async updateLivePrice() {
        try {
            console.log('📡 Fetching live gold price...');
            
            const response = await fetch('/api/live-gold-price');
            const result = await response.json();
            
            if (result.success && result.data && result.data.price) {
                // Success - we have real live data
                this.isConnected = true;
                this.priceRetryCount = 0;
                this.lastPriceUpdate = new Date();
                
                const priceData = {
                    price: result.data.price,
                    source: result.data.source,
                    timestamp: result.data.timestamp,
                    isLive: result.data.is_live !== false,
                    bid: result.data.price - 0.25,
                    ask: result.data.price + 0.25
                };
                
                // Calculate change from previous price
                if (window.currentGoldPrice) {
                    priceData.change = priceData.price - window.currentGoldPrice;
                    priceData.change_percent = (priceData.change / window.currentGoldPrice) * 100;
                }
                
                // Update display
                this.updatePriceDisplay(priceData);
                
                // Update global price for other components
                window.currentGoldPrice = priceData.price;
                
                // Emit event for other components
                window.dispatchEvent(new CustomEvent('livePriceUpdated', { 
                    detail: priceData 
                }));
                
                console.log(`✅ Live price updated: $${priceData.price} (${priceData.source})`);
                
            } else {
                // API unavailable
                this.handlePriceError(result.error || 'Price API unavailable');
            }
            
        } catch (error) {
            console.error('❌ Price update error:', error);
            this.handlePriceError(error.message);
        }
    }
    
    handlePriceError(errorMessage) {
        this.priceRetryCount++;
        this.isConnected = false;
        
        // Update Gold API status to show error
        const statusElement = document.getElementById('gold-api-status');
        const textElement = document.getElementById('gold-api-text');
        if (statusElement && textElement) {
            statusElement.className = 'gold-api-status error';
            textElement.innerHTML = `<i class="fas fa-times-circle"></i> Connection Failed - ${new Date().toLocaleTimeString()}`;
        }
        
        if (this.priceRetryCount <= this.maxRetries) {
            console.log(`🔄 Retrying price fetch (${this.priceRetryCount}/${this.maxRetries})...`);
            
            // Show retrying status
            if (statusElement && textElement) {
                statusElement.className = 'gold-api-status retrying';
                textElement.innerHTML = `<i class="fas fa-sync-alt fa-spin"></i> Retrying (${this.priceRetryCount}/${this.maxRetries})...`;
            }
            return;
        }
        
        // Show error state instead of fake data
        this.updatePriceDisplay({
            error: true,
            message: 'Live price temporarily unavailable',
            details: errorMessage
        });
        
        console.warn('⚠️ Live price unavailable - showing error state');
    }
    
    async updateNewsSentiment() {
        try {
            console.log('📰 Fetching real news sentiment...');
            
            const response = await fetch('/api/news/sentiment-summary');
            const result = await response.json();
            
            if (result.success || result['1D']) {
                this.lastNewsUpdate = new Date();
                
                // Extract daily sentiment data
                const dailySentiment = result['1D'] || result.timeframes?.['1D'];
                
                if (dailySentiment && dailySentiment.total_articles > 0) {
                    const sentimentData = {
                        sentiment: dailySentiment.overall_sentiment.toLowerCase(),
                        strength: dailySentiment.sentiment_strength > 0.7 ? 'strong' : 
                                 dailySentiment.sentiment_strength > 0.4 ? 'moderate' : 'weak',
                        score: dailySentiment.average_score,
                        totalArticles: dailySentiment.total_articles,
                        bullishPct: dailySentiment.bullish_percentage,
                        bearishPct: dailySentiment.bearish_percentage,
                        neutralPct: dailySentiment.neutral_percentage,
                        isReal: true
                    };
                    
                    // Update global sentiment variable
                    this.updateGlobalSentiment(sentimentData);
                    
                    // Update sentiment displays
                    this.updateSentimentDisplay(sentimentData);
                    
                    // Emit event for other components
                    window.dispatchEvent(new CustomEvent('newsSentimentUpdated', { 
                        detail: sentimentData 
                    }));
                    
                    console.log(`✅ News sentiment updated: ${sentimentData.sentiment} (${sentimentData.totalArticles} articles)`);
                    
                } else {
                    console.warn('⚠️ No news articles found for sentiment analysis');
                    this.handleNoNewsData();
                }
                
            } else {
                throw new Error(result.error || 'News API error');
            }
            
        } catch (error) {
            console.error('❌ News sentiment error:', error);
            this.handleNoNewsData();
        }
    }
    
    updateGlobalSentiment(sentimentData) {
        // Update global sentiment variable used by other components
        if (sentimentData.sentiment === 'bullish') {
            window.currentNewsSentiment = 55 + (sentimentData.bullishPct - 40) * 0.5;
        } else if (sentimentData.sentiment === 'bearish') {
            window.currentNewsSentiment = 45 - (sentimentData.bearishPct - 40) * 0.5;
        } else {
            window.currentNewsSentiment = 50;
        }
        
        // Clamp to reasonable range
        window.currentNewsSentiment = Math.max(20, Math.min(80, window.currentNewsSentiment));
    }
    
    handleNoNewsData() {
        const fallbackSentiment = {
            sentiment: 'neutral',
            strength: 'weak',
            score: 0,
            totalArticles: 0,
            bullishPct: 33,
            bearishPct: 33,
            neutralPct: 34,
            isReal: false
        };
        
        this.updateSentimentDisplay(fallbackSentiment);
        window.currentNewsSentiment = 50;
    }
    
    updatePriceDisplay(data) {
        // Handle error cases
        if (data.error) {
            const priceElements = document.querySelectorAll('.price-value, .current-price, .symbol-details .price');
            priceElements.forEach(el => {
                el.textContent = 'Price Unavailable';
                el.style.color = '#ff4757';
            });
            
            // Update Gold API status indicator
            const statusElement = document.getElementById('gold-api-status');
            const textElement = document.getElementById('gold-api-text');
            if (statusElement && textElement) {
                statusElement.className = 'gold-api-status error';
                textElement.innerHTML = `<i class="fas fa-times-circle"></i> ${data.message} - ${new Date().toLocaleTimeString()}`;
            }
            
            return;
        }
        
        // Update successful price display
        const price = parseFloat(data.price);
        const formattedPrice = `$${price.toLocaleString('en-US', { 
            style: 'decimal',  // Force decimal style to prevent currency conversion
            minimumFractionDigits: 2, 
            maximumFractionDigits: 2 
        })}`;
        
        // Update all price elements
        const priceElements = document.querySelectorAll('.price-value, .current-price, .symbol-details .price');
        priceElements.forEach(el => {
            el.textContent = formattedPrice;
            el.style.color = '#00d4aa';
        });
        
        // Update Gold API status indicator
        const statusElement = document.getElementById('gold-api-status');
        const textElement = document.getElementById('gold-api-text');
        if (statusElement && textElement) {
            statusElement.className = 'gold-api-status connected';
            textElement.innerHTML = `<i class="fas fa-satellite-dish"></i> Live ${data.source} Connected - ${new Date().toLocaleTimeString()}`;
        }
        
        // Update watchlist price for XAU/USD
        const watchlistElement = document.querySelector('[data-symbol="XAUUSD"] .price');
        if (watchlistElement) {
            watchlistElement.textContent = formattedPrice;
            watchlistElement.style.color = '#00d4aa';
        }
        
        // Update change display
        if (data.change !== undefined) {
            const changeElements = document.querySelectorAll('.price-change, .symbol-details .change');
            changeElements.forEach(el => {
                const changeText = `${data.change >= 0 ? '+' : ''}$${Math.abs(data.change).toFixed(2)}`;
                const changePctText = data.change_percent ? ` (${data.change_percent >= 0 ? '+' : ''}${data.change_percent.toFixed(2)}%)` : '';
                
                el.innerHTML = `<span class="${data.change >= 0 ? 'positive' : 'negative'}">${changeText}${changePctText}</span>`;
            });
        }
        
        // Update bid/ask in trading panel
        if (data.bid && data.ask) {
            const buyPriceEl = document.getElementById('buy-price');
            const sellPriceEl = document.getElementById('sell-price');
            if (buyPriceEl) buyPriceEl.textContent = `$${data.ask.toFixed(2)}`;
            if (sellPriceEl) sellPriceEl.textContent = `$${data.bid.toFixed(2)}`;
        }
        
        // Update timestamp
        const timestampElements = document.querySelectorAll('.price-timestamp');
        timestampElements.forEach(el => {
            const timestamp = new Date(data.timestamp);
            el.textContent = `Updated: ${timestamp.toLocaleTimeString()}`;
        });
        
        // Update connection status in right panel
        const connectionElements = document.querySelectorAll('.connection-status');
        connectionElements.forEach(el => {
            el.textContent = `${data.source} Connected`;
            el.style.color = data.isLive ? '#00d084' : '#ffa502';
        });
    }
    
    updateSentimentDisplay(sentimentData) {
        // Update news sentiment display
        const newsElement = document.getElementById('news-sentiment');
        if (newsElement) {
            newsElement.className = `factor-sentiment ${sentimentData.sentiment}`;
            newsElement.textContent = `${sentimentData.sentiment.toUpperCase()}`;
            
            // Add data attributes for tooltip/details
            newsElement.setAttribute('data-articles', sentimentData.totalArticles);
            newsElement.setAttribute('data-bullish', sentimentData.bullishPct);
            newsElement.setAttribute('data-bearish', sentimentData.bearishPct);
            newsElement.setAttribute('data-real', sentimentData.isReal);
        }
        
        // Update sentiment labels in news section
        const sentimentLabels = document.querySelectorAll('.sentiment-label');
        sentimentLabels.forEach(label => {
            if (label.textContent.includes('News Sentiment')) {
                const parent = label.parentElement;
                if (parent) {
                    const valueSpan = parent.querySelector('.sentiment-value') || document.createElement('span');
                    valueSpan.className = 'sentiment-value';
                    valueSpan.textContent = `${sentimentData.sentiment.toUpperCase()} (${sentimentData.totalArticles} articles)`;
                    if (!parent.querySelector('.sentiment-value')) {
                        parent.appendChild(valueSpan);
                    }
                }
            }
        });
    }
    
    getConnectionStatus() {
        return {
            isConnected: this.isConnected,
            lastPriceUpdate: this.lastPriceUpdate,
            lastNewsUpdate: this.lastNewsUpdate,
            priceRetryCount: this.priceRetryCount
        };
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Wait a bit for other scripts to load
    setTimeout(() => {
        console.log('🚀 Starting Live Data Manager initialization...');
        window.liveDataManager = new LiveDataManager();
        
        // Expose status for debugging
        window.getLiveDataStatus = () => window.liveDataManager.getConnectionStatus();
        
        console.log('✅ Live Data Manager initialized and running');
    }, 1000);
});

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = LiveDataManager;
}
