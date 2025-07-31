/**
 * GoldGPT Market Data Configuration & Management System
 * Centralizes all market symbols, pricing, and data fetching logic
 */

// =====================================
// CENTRAL CONFIGURATION OBJECT
// =====================================

const GoldGPTConfig = {
    // Symbol mappings and metadata
    SYMBOLS: {
        'XAUUSD': {
            badge: 'GOLD',
            name: 'Gold Spot',
            decimals: 2,
            basePrice: null, // Will be fetched from real-time API
            currency: 'USD',
            category: 'precious_metals',
            fetchEndpoint: '/api/live-gold-price'
        },
        'XAGUSD': {
            badge: 'SILVER',
            name: 'Silver Spot',
            decimals: 2,
            basePrice: 31.25,
            currency: 'USD',
            category: 'precious_metals',
            fetchEndpoint: '/api/price/XAGUSD'
        },
        'EURUSD': {
            badge: 'EUR',
            name: 'Euro US Dollar',
            decimals: 4,
            basePrice: 1.0875,
            currency: 'USD',
            category: 'forex',
            fetchEndpoint: '/api/price/EURUSD'
        },
        'GBPUSD': {
            badge: 'GBP',
            name: 'British Pound US Dollar',
            decimals: 4,
            basePrice: null, // Will be fetched from real-time API
            currency: 'USD',
            category: 'forex',
            fetchEndpoint: '/api/price/GBPUSD'
        },
        'USDJPY': {
            badge: 'JPY',
            name: 'US Dollar Japanese Yen',
            decimals: 2,
            basePrice: 148.50,
            currency: 'JPY',
            category: 'forex',
            fetchEndpoint: '/api/price/USDJPY'
        },
        'BTCUSD': {
            badge: 'BTC',
            name: 'Bitcoin',
            decimals: 2,
            basePrice: 43250.00,
            currency: 'USD',
            category: 'crypto',
            fetchEndpoint: '/api/price/BTCUSD'
        },
        'SPY': {
            badge: 'SPY',
            name: 'SPDR S&P 500 ETF',
            decimals: 2,
            basePrice: 445.50,
            currency: 'USD',
            category: 'etf',
            fetchEndpoint: '/api/price/SPY'
        },
        'QQQ': {
            badge: 'QQQ',
            name: 'Invesco QQQ Trust',
            decimals: 2,
            basePrice: 375.25,
            currency: 'USD',
            category: 'etf',
            fetchEndpoint: '/api/price/QQQ'
        }
    },

    // Timeframe mappings for TradingView
    TIMEFRAMES: {
        '1m': '1',
        '5m': '5',
        '15m': '15',
        '1h': '60',
        '4h': '240',
        '1d': 'D',
        '1w': 'W'
    },

    // Cache settings
    CACHE: {
        EXPIRY_MS: 30000, // 30 seconds
        MAX_RETRIES: 3,
        RETRY_DELAY_MS: 1000
    },

    // Update intervals (milliseconds)
    UPDATE_INTERVALS: {
        REAL_TIME: 2000,     // 2 seconds for real-time updates
        NORMAL: 30000,       // 30 seconds for normal updates
        BACKGROUND: 300000   // 5 minutes for background data
    },

    // API endpoints
    ENDPOINTS: {
        LIVE_GOLD: '/api/live-gold-price',
        PRICE_GENERIC: '/api/price/',
        CHART_DATA: '/api/chart/data/',
        COMPREHENSIVE_ANALYSIS: '/api/comprehensive-analysis/'
    }
};

// =====================================
// PRICE CACHE & DATA MANAGER
// =====================================

class MarketDataManager {
    constructor() {
        this.cache = new Map();
        this.subscribers = new Map();
        this.updateIntervals = new Map();
        this.isInitialized = false;
        this.retryCount = new Map();
        
        console.log('🏦 Market Data Manager initialized');
    }

    /**
     * Initialize the market data system
     */
    async init() {
        if (this.isInitialized) return;
        
        console.log('🚀 Initializing Market Data Manager...');
        
        // Load initial prices for all symbols
        await this.loadInitialPrices();
        
        // Start real-time updates for key symbols
        this.startRealTimeUpdates();
        
        this.isInitialized = true;
        console.log('✅ Market Data Manager ready');
    }

    /**
     * Load initial prices for all configured symbols
     */
    async loadInitialPrices() {
        const symbols = Object.keys(GoldGPTConfig.SYMBOLS);
        console.log(`📊 Loading initial prices for ${symbols.length} symbols...`);
        
        const promises = symbols.map(symbol => this.fetchPrice(symbol, true));
        const results = await Promise.allSettled(promises);
        
        let loaded = 0;
        results.forEach((result, index) => {
            if (result.status === 'fulfilled') {
                loaded++;
            } else {
                console.warn(`⚠️ Failed to load initial price for ${symbols[index]}:`, result.reason);
            }
        });
        
        console.log(`✅ Loaded initial prices for ${loaded}/${symbols.length} symbols`);
    }

    /**
     * Fetch price for a specific symbol
     */
    async fetchPrice(symbol, isInitialLoad = false) {
        const config = GoldGPTConfig.SYMBOLS[symbol];
        if (!config) {
            throw new Error(`Unknown symbol: ${symbol}`);
        }

        // Check cache first (unless initial load)
        if (!isInitialLoad && this.isCacheValid(symbol)) {
            return this.cache.get(symbol);
        }

        try {
            const response = await fetch(config.fetchEndpoint);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const data = await response.json();
            const priceData = this.processPriceData(symbol, data);
            
            // Cache the data
            this.updateCache(symbol, priceData);
            
            // Notify subscribers
            this.notifySubscribers(symbol, priceData);
            
            // Reset retry count on success
            this.retryCount.set(symbol, 0);
            
            return priceData;
            
        } catch (error) {
            console.error(`❌ Error fetching price for ${symbol}:`, error);
            
            // Implement retry logic
            const retries = this.retryCount.get(symbol) || 0;
            if (retries < GoldGPTConfig.CACHE.MAX_RETRIES) {
                this.retryCount.set(symbol, retries + 1);
                setTimeout(() => this.fetchPrice(symbol), GoldGPTConfig.CACHE.RETRY_DELAY_MS);
            }
            
            // Return fallback data
            return this.getFallbackPrice(symbol);
        }
    }

    /**
     * Process raw price data into standardized format
     */
    processPriceData(symbol, rawData) {
        const config = GoldGPTConfig.SYMBOLS[symbol];
        const timestamp = new Date();
        
        // Handle different API response formats
        let price = config.basePrice;
        let change = 0;
        let changePercent = 0;
        
        if (rawData.price !== undefined) {
            price = parseFloat(rawData.price);
        } else if (rawData.current_price !== undefined) {
            price = parseFloat(rawData.current_price);
        } else if (rawData.close !== undefined) {
            price = parseFloat(rawData.close);
        }
        
        if (rawData.change !== undefined) {
            change = parseFloat(rawData.change);
        }
        
        if (rawData.change_percent !== undefined) {
            changePercent = parseFloat(rawData.change_percent);
        } else if (change && price) {
            changePercent = ((change / (price - change)) * 100);
        }
        
        return {
            symbol,
            price: this.formatPrice(price, config.decimals),
            change: this.formatPrice(change, config.decimals),
            changePercent: Math.round(changePercent * 100) / 100,
            timestamp,
            currency: config.currency,
            isUp: change >= 0,
            raw: rawData
        };
    }

    /**
     * Get fallback price when API fails
     */
    getFallbackPrice(symbol) {
        const config = GoldGPTConfig.SYMBOLS[symbol];
        const cached = this.cache.get(symbol);
        
        // Return cached data if available, otherwise base price
        if (cached) {
            return { ...cached, isFallback: true };
        }
        
        return {
            symbol,
            price: config.basePrice,
            change: 0,
            changePercent: 0,
            timestamp: new Date(),
            currency: config.currency,
            isUp: false,
            isFallback: true
        };
    }

    /**
     * Format price according to symbol configuration
     */
    formatPrice(price, decimals = 2) {
        if (price === null || price === undefined || isNaN(price)) {
            return 0;
        }
        return Math.round(price * Math.pow(10, decimals)) / Math.pow(10, decimals);
    }

    /**
     * Format price for display
     */
    formatPriceDisplay(price, symbol) {
        const config = GoldGPTConfig.SYMBOLS[symbol];
        if (!config) return price.toString();
        
        const formatted = this.formatPrice(price, config.decimals);
        
        // Force USD formatting to prevent locale issues
        if (formatted >= 1000) {
            return formatted.toLocaleString('en-US', {
                style: 'decimal',  // Force decimal style, not currency
                minimumFractionDigits: config.decimals,
                maximumFractionDigits: config.decimals
            });
        }
        
        return formatted.toFixed(config.decimals);
    }

    /**
     * Check if cached data is still valid
     */
    isCacheValid(symbol) {
        const cached = this.cache.get(symbol);
        if (!cached) return false;
        
        const age = Date.now() - cached.timestamp.getTime();
        return age < GoldGPTConfig.CACHE.EXPIRY_MS;
    }

    /**
     * Update cache with new data
     */
    updateCache(symbol, data) {
        this.cache.set(symbol, data);
        
        // Emit cache update event
        if (typeof window !== 'undefined') {
            window.dispatchEvent(new CustomEvent('marketDataUpdated', {
                detail: { symbol, data }
            }));
        }
    }

    /**
     * Subscribe to price updates for a symbol
     */
    subscribe(symbol, callback) {
        if (!this.subscribers.has(symbol)) {
            this.subscribers.set(symbol, new Set());
        }
        this.subscribers.get(symbol).add(callback);
        
        // Immediately call with cached data if available
        const cached = this.cache.get(symbol);
        if (cached) {
            callback(cached);
        }
    }

    /**
     * Unsubscribe from price updates
     */
    unsubscribe(symbol, callback) {
        const subscribers = this.subscribers.get(symbol);
        if (subscribers) {
            subscribers.delete(callback);
        }
    }

    /**
     * Notify all subscribers of price updates
     */
    notifySubscribers(symbol, data) {
        const subscribers = this.subscribers.get(symbol);
        if (subscribers) {
            subscribers.forEach(callback => {
                try {
                    callback(data);
                } catch (error) {
                    console.error(`Error in subscriber callback for ${symbol}:`, error);
                }
            });
        }
    }

    /**
     * Start real-time updates for important symbols
     */
    startRealTimeUpdates() {
        const primarySymbols = ['XAUUSD', 'BTCUSD', 'EURUSD'];
        
        primarySymbols.forEach(symbol => {
            const interval = setInterval(() => {
                this.fetchPrice(symbol);
            }, GoldGPTConfig.UPDATE_INTERVALS.REAL_TIME);
            
            this.updateIntervals.set(symbol, interval);
        });
        
        console.log(`🔄 Started real-time updates for ${primarySymbols.length} symbols`);
    }

    /**
     * Get current price for a symbol
     */
    getCurrentPrice(symbol) {
        const cached = this.cache.get(symbol);
        if (cached && this.isCacheValid(symbol)) {
            return cached;
        }
        
        // Return fallback if no valid cache
        return this.getFallbackPrice(symbol);
    }

    /**
     * Get symbol configuration
     */
    getSymbolConfig(symbol) {
        return GoldGPTConfig.SYMBOLS[symbol] || null;
    }

    /**
     * Get all symbols in a category
     */
    getSymbolsByCategory(category) {
        return Object.entries(GoldGPTConfig.SYMBOLS)
            .filter(([_, config]) => config.category === category)
            .map(([symbol, _]) => symbol);
    }

    /**
     * Cleanup resources
     */
    destroy() {
        // Clear all intervals
        this.updateIntervals.forEach(interval => clearInterval(interval));
        this.updateIntervals.clear();
        
        // Clear cache and subscribers
        this.cache.clear();
        this.subscribers.clear();
        
        console.log('🧹 Market Data Manager destroyed');
    }
}

// =====================================
// HELPER FUNCTIONS (Legacy Support)
// =====================================

/**
 * Get symbol badge (legacy function)
 */
function getSymbolBadge(symbol) {
    const config = GoldGPTConfig.SYMBOLS[symbol];
    return config ? config.badge : symbol;
}

/**
 * Get symbol name (legacy function)
 */
function getSymbolName(symbol) {
    const config = GoldGPTConfig.SYMBOLS[symbol];
    return config ? config.name : symbol;
}

/**
 * Convert timeframe to TradingView format (legacy function)
 */
function convertTimeframeToTradingView(timeframe) {
    return GoldGPTConfig.TIMEFRAMES[timeframe] || '60';
}

// =====================================
// GLOBAL INITIALIZATION
// =====================================

// Create global market data manager instance
window.marketDataManager = new MarketDataManager();

// Auto-initialize when DOM is ready
document.addEventListener('DOMContentLoaded', async () => {
    console.log('🚀 Initializing Market Data Manager...');
    await window.marketDataManager.init();
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (window.marketDataManager) {
        window.marketDataManager.destroy();
    }
});

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { GoldGPTConfig, MarketDataManager };
}

console.log('📊 Market Data Configuration loaded successfully');
