/**
 * GoldGPT Live Price Integration
 * Fetches live gold prices from Gold-API.com and updates all price displays
 */

class GoldAPILivePriceFetcher {
    constructor() {
        this.isRunning = false;
        this.updateInterval = null;
        this.lastPrice = null;
        this.apiEndpoint = '/api/live-gold-price';
        this.updateFrequency = 1500; // 1.5 seconds for smooth updates
        
        console.log('🏆 GoldAPILivePriceFetcher initialized');
        this.init();
    }

    async init() {
        console.log('🚀 Starting Gold-API live price integration...');
        
        // Start fetching immediately
        await this.fetchAndUpdatePrice();
        
        // Start the update loop
        this.startUpdates();
        
        // Update price source indicators
        this.updatePriceSourceIndicators();
    }

    async fetchAndUpdatePrice() {
        try {
            console.log('📡 Fetching live gold price from Gold-API...');
            
            const response = await fetch(this.apiEndpoint);
            if (!response.ok) {
                throw new Error(`API response not ok: ${response.status}`);
            }
            
            const data = await response.json();
            console.log('✅ Gold-API response:', data);
            
            // Handle nested data structure
            const priceData = data.data || data;
            const price = priceData.price || data.price;
            
            if (data.success && price && price > 0) {
                await this.updateAllPriceDisplays(price, data);
                console.log(`💰 Live gold price updated: $${price}`);
            } else {
                console.warn('⚠️ Invalid price data received:', data);
            }
            
        } catch (error) {
            console.error('❌ Error fetching live gold price:', error);
            this.handleFetchError(error);
        }
    }

    async updateAllPriceDisplays(currentPrice, priceData = {}) {
        if (!currentPrice || currentPrice <= 0) {
            console.warn('Invalid price data, skipping update');
            return;
        }

        const isUp = this.lastPrice ? currentPrice > this.lastPrice : true;
        const change = this.lastPrice ? currentPrice - this.lastPrice : 0;
        const changePercent = this.lastPrice ? (change / this.lastPrice) * 100 : 0;
        
        const formattedPrice = `$${currentPrice.toFixed(2)}`;
        const formattedChange = this.lastPrice ? 
            `${change >= 0 ? '+' : ''}$${change.toFixed(2)} (${changePercent >= 0 ? '+' : ''}${changePercent.toFixed(2)}%)` : 
            'Live Data';

        console.log(`🔄 Updating displays - Price: ${formattedPrice}, Change: ${formattedChange}`);

        // Update main current price display (handle multiple elements with same ID)
        const priceElements = document.querySelectorAll('#current-price, [id="current-price"]');
        console.log(`🎯 Found ${priceElements.length} price elements to update`);
        priceElements.forEach((el, index) => {
            if (el) {
                console.log(`💰 Updating price element ${index + 1}: ${formattedPrice}`);
                el.textContent = formattedPrice;
                el.classList.add('price-flash');
                setTimeout(() => el.classList.remove('price-flash'), 500);
            }
        });

        // Update price change displays (handle multiple elements)
        const changeElements = document.querySelectorAll('#price-change, [id="price-change"]');
        console.log(`📊 Found ${changeElements.length} change elements to update`);
        changeElements.forEach((el, index) => {
            if (el) {
                console.log(`📈 Updating change element ${index + 1}: ${formattedChange}`);
                el.innerHTML = `<span class="${isUp ? 'positive' : 'negative'}">${formattedChange}</span>`;
            }
        });

        // Update watchlist XAU/USD entries
        const watchlistPrices = document.querySelectorAll('.watchlist-item[data-symbol="XAUUSD"] .price-value');
        watchlistPrices.forEach(el => {
            if (el) {
                el.textContent = formattedPrice;
                el.classList.add('price-flash');
                setTimeout(() => el.classList.remove('price-flash'), 500);
            }
        });

        // Update watchlist change percentages
        const watchlistChanges = document.querySelectorAll('.watchlist-item[data-symbol="XAUUSD"] .price-change');
        watchlistChanges.forEach(el => {
            if (el && this.lastPrice) {
                el.textContent = `${changePercent >= 0 ? '+' : ''}${changePercent.toFixed(2)}%`;
                el.className = `price-change ${isUp ? 'positive' : 'negative'}`;
            }
        });

        // Update trading button prices (with small spread)
        const spread = currentPrice * 0.0001; // 0.01% spread
        const buyPriceElements = document.querySelectorAll('#buy-price');
        const sellPriceElements = document.querySelectorAll('#sell-price');
        
        buyPriceElements.forEach(el => {
            if (el) el.textContent = `$${(currentPrice + spread).toFixed(2)}`;
        });
        
        sellPriceElements.forEach(el => {
            if (el) el.textContent = `$${(currentPrice - spread).toFixed(2)}`;
        });

        // Update all price source indicators
        this.updatePriceSourceIndicators(true);

        // Store the last price for comparison
        this.lastPrice = currentPrice;

        // Flash price change effect
        this.addPriceFlashEffect(isUp);
    }

    updatePriceSourceIndicators(isLive = false) {
        // Update all price source indicators (handle multiple elements)
        const sourceElements = document.querySelectorAll('#price-source, [id="price-source"]');
        console.log(`📡 Found ${sourceElements.length} source elements to update`);
        sourceElements.forEach((el, index) => {
            if (el) {
                console.log(`🔗 Updating source element ${index + 1}`);
                el.innerHTML = `<i class="fas fa-satellite-dish"></i> Live from Gold-API.com ${isLive ? '🟢' : '🔄'}`;
                el.style.color = isLive ? '#00d084' : '#ffa502';
            }
        });

        // Update Gold API status indicator
        const statusIndicator = document.getElementById('gold-api-status');
        if (statusIndicator) {
            const statusText = document.getElementById('gold-api-text');
            if (statusText) {
                statusText.textContent = isLive ? 'Gold API Connected ✅' : 'Connecting to Gold API...';
            }
            statusIndicator.style.background = isLive ? 
                'linear-gradient(135deg, #00d084, #00a86b)' : 
                'linear-gradient(135deg, #ffa502, #ff9500)';
        }
    }

    addPriceFlashEffect(isUp) {
        // Add subtle flash effect to price displays
        const priceElements = document.querySelectorAll('.price-value, #current-price');
        priceElements.forEach(el => {
            if (el) {
                el.style.background = isUp ? 
                    'linear-gradient(90deg, rgba(0, 208, 132, 0.2), transparent)' :
                    'linear-gradient(90deg, rgba(255, 71, 87, 0.2), transparent)';
                setTimeout(() => {
                    el.style.background = 'transparent';
                }, 300);
            }
        });
    }

    startUpdates() {
        if (this.isRunning) {
            console.log('⚠️ Price updates already running');
            return;
        }

        this.isRunning = true;
        console.log(`🔄 Starting price updates every ${this.updateFrequency}ms`);
        
        this.updateInterval = setInterval(async () => {
            await this.fetchAndUpdatePrice();
        }, this.updateFrequency);
    }

    stopUpdates() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
        this.isRunning = false;
        console.log('🛑 Price updates stopped');
    }

    handleFetchError(error) {
        console.error('API fetch error:', error);
        
        // Update status indicators to show error
        const sourceElements = document.querySelectorAll('#price-source');
        sourceElements.forEach(el => {
            if (el) {
                el.innerHTML = '<i class="fas fa-exclamation-triangle"></i> API Error - Retrying...';
                el.style.color = '#ff4757';
            }
        });
    }

    // Public method to manually refresh price
    async refreshPrice() {
        console.log('🔄 Manual price refresh requested');
        await this.fetchAndUpdatePrice();
    }

    // Cleanup method
    destroy() {
        this.stopUpdates();
        console.log('🧹 GoldAPILivePriceFetcher destroyed');
    }
}

// Create global instance for component loader
window.goldApiLivePriceFetcher = new GoldAPILivePriceFetcher();

// Add init method for component loader compatibility
window.goldApiLivePriceFetcher.init = function() {
    return this.init();
};

// Legacy global reference
window.goldAPIFetcher = window.goldApiLivePriceFetcher;

// Auto-initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    console.log('🎯 DOM ready, initializing Gold-API live price fetcher...');
    console.log('🔍 Current URL:', window.location.href);
    console.log('🔍 Available elements with current-price:', document.querySelectorAll('#current-price, [id="current-price"]').length);
    
    // Add global refresh function
    window.refreshGoldPrice = () => {
        if (window.goldApiLivePriceFetcher) {
            window.goldApiLivePriceFetcher.refreshPrice();
        }
    };
    
    // Add console helper
    console.log('💡 Use refreshGoldPrice() to manually update the price');
    console.log('💡 Use window.goldApiLivePriceFetcher to access the fetcher instance');
});

// Handle page unload
window.addEventListener('beforeunload', () => {
    if (window.goldApiLivePriceFetcher) {
        window.goldApiLivePriceFetcher.destroy();
    }
});

console.log('💰 Gold API Live Price Fetcher loaded successfully');
