/**
 * Price Display Manager for GoldGPT
 * Handles all price display updates across the dashboard
 */

class PriceDisplayManager {
    constructor() {
        this.displayElements = new Map();
        this.animationQueue = [];
        this.isProcessingAnimations = false;
        this.componentId = 'price-display';
        this.eventCleanup = [];
        this.updateInterval = null;
        
        console.log('💰 Price Display Manager initialized');
    }

    /**
     * Initialize price displays across the dashboard
     */
    async init() {
        console.log('🎯 Initializing Price Display Manager...');
        
        // Register all price display elements
        this.registerDisplayElements();
        
        // Setup connection manager integration
        this.setupConnectionManagerIntegration();
        
        // Subscribe to market data updates
        this.subscribeToMarketData();
        
        // Set up periodic display refresh
        this.setupPeriodicRefresh();
        
        // Load initial prices
        await this.loadInitialPrices();
        
        console.log('✅ Price Display Manager ready');
    }

    /**
     * Register all price display elements in the DOM
     */
    registerDisplayElements() {
        const selectors = [
            // Watchlist prices
            '#watchlist-xauusd-price',
            '#watchlist-xagusd-price', 
            '#watchlist-eurusd-price',
            '#watchlist-gbpusd-price',
            '#watchlist-usdjpy-price',
            '#watchlist-btcusd-price',
            
            // Chart current prices
            '#current-price',
            '#chart-current-price',
            '#gold-current-price',
            
            // Header price displays
            '.gold-price-display',
            '.current-gold-price',
            
            // Dashboard main price
            '.main-price-display'
        ];

        selectors.forEach(selector => {
            const elements = document.querySelectorAll(selector);
            elements.forEach(element => {
                const symbol = this.extractSymbolFromElement(element);
                if (symbol) {
                    if (!this.displayElements.has(symbol)) {
                        this.displayElements.set(symbol, []);
                    }
                    this.displayElements.get(symbol).push({
                        element,
                        selector,
                        type: this.getDisplayType(selector)
                    });
                }
            });
        });

        console.log(`📝 Registered ${this.displayElements.size} symbol displays`);
    }

    /**
     * Extract symbol from element ID or data attributes
     */
    extractSymbolFromElement(element) {
        // Try to get symbol from ID
        if (element.id) {
            const match = element.id.match(/watchlist-(\w+)-price/);
            if (match) {
                return match[1].toUpperCase();
            }
        }

        // Try data attribute
        if (element.dataset.symbol) {
            return element.dataset.symbol.toUpperCase();
        }

        // Default to XAUUSD for generic price displays
        return 'XAUUSD';
    }

    /**
     * Determine display type based on selector
     */
    getDisplayType(selector) {
        if (selector.includes('watchlist')) return 'watchlist';
        if (selector.includes('chart')) return 'chart';
        if (selector.includes('current')) return 'current';
        return 'general';
    }

    /**
     * Subscribe to market data updates
     */
    subscribeToMarketData() {
        // Subscribe to each symbol we have displays for
        this.displayElements.forEach((displays, symbol) => {
            if (window.marketDataManager) {
                window.marketDataManager.subscribe(symbol, (data) => {
                    this.updateDisplaysForSymbol(symbol, data);
                });
            }
        });

        // Also listen for custom market data events
        window.addEventListener('marketDataUpdated', (event) => {
            const { symbol, data } = event.detail;
            this.updateDisplaysForSymbol(symbol, data);
        });
    }

    /**
     * Load initial prices for all registered symbols
     */
    async loadInitialPrices() {
        const symbols = Array.from(this.displayElements.keys());
        
        for (const symbol of symbols) {
            try {
                // Set loading state
                if (window.connectionManager) {
                    window.connectionManager.setLoading(`${this.componentId}-${symbol}`, true);
                }
                
                let priceData;
                if (window.marketDataManager) {
                    priceData = await window.marketDataManager.fetchPrice(symbol, true);
                } else {
                    // Fallback to direct API call
                    priceData = await window.connectionManager?.request(`/api/price/${symbol}`) || 
                                await fetch(`/api/price/${symbol}`).then(r => r.json());
                }
                
                this.updateDisplaysForSymbol(symbol, priceData);
                
                // Clear loading state
                if (window.connectionManager) {
                    window.connectionManager.setLoading(`${this.componentId}-${symbol}`, false);
                }
                
            } catch (error) {
                console.error(`Error loading price for ${symbol}:`, error);
                
                // Handle error through connection manager
                if (window.connectionManager) {
                    window.connectionManager.handleError(`${this.componentId}-${symbol}`, error);
                }
                
                // Use fallback data
                this.updateDisplaysWithFallback(symbol);
            }
        }
    }

    /**
     * Update all displays for a specific symbol
     */
    updateDisplaysForSymbol(symbol, priceData) {
        const displays = this.displayElements.get(symbol);
        if (!displays) return;

        displays.forEach(({ element, type }) => {
            this.updateSingleDisplay(element, priceData, type);
        });
    }

    /**
     * Update a single price display element
     */
    updateSingleDisplay(element, priceData, type) {
        if (!element || !priceData) return;

        try {
            const { price, change, changePercent, isUp, symbol } = priceData;
            
            // Format price based on symbol configuration
            const formattedPrice = window.marketDataManager 
                ? window.marketDataManager.formatPriceDisplay(price, symbol)
                : price.toFixed(2);

            // Update price text
            const priceText = this.formatPriceForDisplay(formattedPrice, symbol);
            
            // Store old price for animation
            const oldPrice = parseFloat(element.dataset.lastPrice || 0);
            element.dataset.lastPrice = price;

            // Update the display based on type
            switch (type) {
                case 'watchlist':
                    this.updateWatchlistDisplay(element, priceText, change, changePercent, isUp);
                    break;
                case 'chart':
                    this.updateChartDisplay(element, priceText, isUp);
                    break;
                case 'current':
                    this.updateCurrentPriceDisplay(element, priceText, isUp);
                    break;
                default:
                    this.updateGeneralDisplay(element, priceText, isUp);
            }

            // Add price movement animation
            if (oldPrice > 0 && oldPrice !== price) {
                this.addPriceAnimation(element, price > oldPrice);
            }

            // Remove loading state
            element.classList.remove('loading', 'price-loading');

        } catch (error) {
            console.error('Error updating display:', error);
        }
    }

    /**
     * Update watchlist display format
     */
    updateWatchlistDisplay(element, priceText, change, changePercent, isUp) {
        element.textContent = priceText;
        
        // Update change display if it exists
        const symbolMatch = element.id.match(/watchlist-(\w+)-price/);
        if (symbolMatch) {
            const changeElement = document.querySelector(`#watchlist-${symbolMatch[1]}-change`);
            if (changeElement) {
                const changeText = `${isUp ? '+' : ''}${changePercent.toFixed(2)}%`;
                changeElement.textContent = changeText;
                changeElement.className = `price-change ${isUp ? 'positive' : 'negative'}`;
            }
        }
    }

    /**
     * Update chart display format
     */
    updateChartDisplay(element, priceText, isUp) {
        element.textContent = priceText;
        element.className = `chart-price ${isUp ? 'price-up' : 'price-down'}`;
    }

    /**
     * Update current price display format
     */
    updateCurrentPriceDisplay(element, priceText, isUp) {
        element.textContent = priceText;
        
        // Add currency symbol if not present
        if (!priceText.includes('$')) {
            element.textContent = `$${priceText}`;
        }
    }

    /**
     * Update general display format
     */
    updateGeneralDisplay(element, priceText, isUp) {
        element.textContent = priceText;
        
        // Add basic price movement class
        element.classList.remove('price-up', 'price-down');
        element.classList.add(isUp ? 'price-up' : 'price-down');
    }

    /**
     * Format price for display with appropriate currency symbols
     */
    formatPriceForDisplay(price, symbol) {
        const config = window.marketDataManager?.getSymbolConfig(symbol);
        
        if (config && config.currency === 'USD') {
            return `$${price}`;
        }
        
        return price.toString();
    }

    /**
     * Add price movement animation
     */
    addPriceAnimation(element, isUp) {
        // Remove existing animation classes
        element.classList.remove('price-flash-up', 'price-flash-down');
        
        // Add new animation class
        const animationClass = isUp ? 'price-flash-up' : 'price-flash-down';
        element.classList.add(animationClass);
        
        // Remove animation class after animation completes
        setTimeout(() => {
            element.classList.remove(animationClass);
        }, 1000);
    }

    /**
     * Update displays with fallback data when API fails
     */
    updateDisplaysWithFallback(symbol) {
        const config = window.marketDataManager?.getSymbolConfig(symbol);
        if (!config) return;

        const fallbackData = {
            symbol,
            price: config.basePrice,
            change: 0,
            changePercent: 0,
            isUp: false,
            isFallback: true
        };

        this.updateDisplaysForSymbol(symbol, fallbackData);
    }

    /**
     * Set up periodic refresh of all displays
     */
    setupPeriodicRefresh() {
        // Refresh displays every 30 seconds to ensure consistency
        setInterval(() => {
            this.refreshAllDisplays();
        }, 30000);
    }

    /**
     * Refresh all price displays
     */
    async refreshAllDisplays() {
        const symbols = Array.from(this.displayElements.keys());
        
        for (const symbol of symbols) {
            try {
                if (window.marketDataManager) {
                    const priceData = window.marketDataManager.getCurrentPrice(symbol);
                    this.updateDisplaysForSymbol(symbol, priceData);
                }
            } catch (error) {
                console.error(`Error refreshing display for ${symbol}:`, error);
            }
        }
    }

    /**
     * Manually update a specific symbol display
     */
    async updateSymbol(symbol) {
        try {
            if (window.marketDataManager) {
                const priceData = await window.marketDataManager.fetchPrice(symbol);
                this.updateDisplaysForSymbol(symbol, priceData);
                return priceData;
            }
        } catch (error) {
            console.error(`Error updating ${symbol}:`, error);
            this.updateDisplaysWithFallback(symbol);
        }
    }

    /**
     * Add a new display element for a symbol
     */
    registerNewDisplay(element, symbol, type = 'general') {
        symbol = symbol.toUpperCase();
        
        if (!this.displayElements.has(symbol)) {
            this.displayElements.set(symbol, []);
            
            // Subscribe to market data for new symbol
            if (window.marketDataManager) {
                window.marketDataManager.subscribe(symbol, (data) => {
                    this.updateDisplaysForSymbol(symbol, data);
                });
            }
        }

        this.displayElements.get(symbol).push({
            element,
            selector: element.id || element.className,
            type
        });

        // Update immediately with current data
        if (window.marketDataManager) {
            const currentData = window.marketDataManager.getCurrentPrice(symbol);
            this.updateSingleDisplay(element, currentData, type);
        }
    }

    /**
     * Get current display status
     */
    getDisplayStatus() {
        const status = {
            totalSymbols: this.displayElements.size,
            totalDisplays: 0,
            symbolBreakdown: {}
        };

        this.displayElements.forEach((displays, symbol) => {
            status.totalDisplays += displays.length;
            status.symbolBreakdown[symbol] = displays.length;
        });

        return status;
    }

    /**
     * Cleanup resources
     */
    destroy() {
        // Clear display elements
        this.displayElements.clear();
        
        // Clear animation queue
        this.animationQueue = [];
        
        console.log('🧹 Price Display Manager destroyed');
    }
}

// =====================================
// CSS ANIMATIONS (Injected Styles)
// =====================================

function injectPriceAnimationStyles() {
    if (document.getElementById('price-animation-styles')) return;

    const styles = `
        <style id="price-animation-styles">
            .price-flash-up {
                animation: priceFlashUp 1s ease-out;
            }
            
            .price-flash-down {
                animation: priceFlashDown 1s ease-out;
            }
            
            @keyframes priceFlashUp {
                0% { background-color: rgba(0, 208, 132, 0.3); }
                100% { background-color: transparent; }
            }
            
            @keyframes priceFlashDown {
                0% { background-color: rgba(255, 71, 87, 0.3); }
                100% { background-color: transparent; }
            }
            
            .price-up {
                color: var(--success, #00d084) !important;
            }
            
            .price-down {
                color: var(--danger, #ff4757) !important;
            }
            
            .price-loading {
                opacity: 0.6;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
                background-size: 200% 100%;
                animation: loading-shimmer 1.5s infinite;
            }
            
            @keyframes loading-shimmer {
                0% { background-position: -200% 0; }
                100% { background-position: 200% 0; }
            }
        </style>
    `;
    
    document.head.insertAdjacentHTML('beforeend', styles);
}

/**
 * Setup connection manager integration
 */
PriceDisplayManager.prototype.setupConnectionManagerIntegration = function() {
    if (!window.connectionManager) {
        console.warn('⚠️ Connection Manager not available for price display manager');
        return;
    }
    
    // Setup retry handler
    const retryCleanup = window.connectionManager.on('retry', (data) => {
        if (data.componentId.startsWith(this.componentId)) {
            const symbol = data.componentId.split('-')[2];
            console.log(`🔄 Retrying price load for ${symbol}...`);
            this.loadInitialPrices();
        }
    });
    this.eventCleanup.push(retryCleanup);
    
    // Setup WebSocket price updates
    const priceUpdateCleanup = window.connectionManager.on('price_update', (data) => {
        console.log('💰 Real-time price update received:', data);
        this.updateDisplaysForSymbol(data.symbol, data);
    });
    this.eventCleanup.push(priceUpdateCleanup);
    
    console.log('✅ Connection Manager integration setup for price display');
};

/**
 * Cleanup all event listeners and resources
 */
PriceDisplayManager.prototype.cleanup = function() {
    console.log('🧹 Cleaning up Price Display Manager...');
    
    // Clear update interval
    if (this.updateInterval) {
        clearInterval(this.updateInterval);
    }
    
    // Clean up event listeners
    this.eventCleanup.forEach(cleanup => cleanup());
    this.eventCleanup = [];
    
    // Clean up from connection manager
    if (window.connectionManager) {
        window.connectionManager.offContext(this);
    }
    
    console.log('✅ Price Display Manager cleaned up');
};

// =====================================
// GLOBAL INITIALIZATION
// =====================================

// Create global price display manager
window.priceDisplayManager = new PriceDisplayManager();

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', async () => {
    // Inject animation styles
    injectPriceAnimationStyles();
    
    // Wait for market data manager to be ready
    if (window.marketDataManager) {
        await window.priceDisplayManager.init();
    } else {
        // Wait for market data manager and then initialize
        const checkMarketData = setInterval(() => {
            if (window.marketDataManager) {
                clearInterval(checkMarketData);
                window.priceDisplayManager.init();
            }
        }, 100);
    }
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (window.priceDisplayManager) {
        window.priceDisplayManager.destroy();
    }
});

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PriceDisplayManager;
}

console.log('💰 Price Display Manager loaded successfully');
