/**
 * TradingView Widget Gold Price Scraper
 * Extracts live gold price data directly from TradingView widget DOM elements
 */

class TradingViewGoldScraper {
    constructor() {
        this.currentPrice = null;
        this.lastUpdate = null;
        this.observers = [];
        this.scrapingInterval = null;
        this.widgetContainer = null;
        this.priceSelectors = [
            // Common TradingView price selectors
            '[data-field="last_price"]',
            '.tv-symbol-price-quote__value',
            '.js-symbol-last',
            '.tv-ticker__price',
            '[class*="price"]',
            '[class*="last"]',
            '[data-symbol-last-price]',
            // Widget-specific selectors
            'iframe[id*="tradingview"]',
            '.tradingview-widget-container',
            // Price display elements
            'span[class*="price"]:not(:empty)',
            'div[class*="price"]:not(:empty)',
        ];
        
        console.log('🔍 TradingView Gold Scraper initialized');
    }
    
    // Initialize scraping for embedded TradingView widgets
    init() {
        console.log('🚀 Starting TradingView Gold Price Scraper...');
        
        // Wait for widgets to load
        setTimeout(() => {
            this.findTradingViewWidgets();
            this.startScraping();
            this.setupMutationObserver();
        }, 3000);
    }
    
    // Find TradingView widget containers
    findTradingViewWidgets() {
        const containers = [
            ...document.querySelectorAll('.tradingview-widget-container'),
            ...document.querySelectorAll('[id*="tradingview"]'),
            ...document.querySelectorAll('iframe[src*="tradingview"]'),
            ...document.querySelectorAll('iframe[src*="widget"]')
        ];
        
        console.log('🔍 Found TradingView containers:', containers.length);
        
        containers.forEach((container, index) => {
            console.log(`📊 Widget ${index + 1}:`, {
                tagName: container.tagName,
                id: container.id,
                className: container.className,
                src: container.src || 'N/A'
            });
        });
        
        if (containers.length > 0) {
            this.widgetContainer = containers[0];
            console.log('✅ Primary widget container set');
        }
        
        return containers;
    }
    
    // Scrape price from widget DOM
    scrapePriceFromDOM() {
        let foundPrice = null;
        let foundSelector = null;
        
        // Try different selectors
        for (const selector of this.priceSelectors) {
            const elements = document.querySelectorAll(selector);
            
            for (const element of elements) {
                const text = element.textContent || element.innerText || '';
                const cleanText = text.trim().replace(/[^\d.,]/g, '');
                
                // Look for gold price patterns (3000-4000 range)
                const priceMatch = cleanText.match(/(\d{1,2}[,.]?\d{3}[.,]\d{2})/);
                if (priceMatch) {
                    const price = parseFloat(priceMatch[1].replace(/,/g, ''));
                    if (price >= 2000 && price <= 5000) {
                        foundPrice = price;
                        foundSelector = selector;
                        console.log('💰 Found gold price:', price, 'from selector:', selector);
                        break;
                    }
                }
            }
            
            if (foundPrice) break;
        }
        
        return { price: foundPrice, selector: foundSelector };
    }
    
    // Scrape from iframe content (if accessible)
    scrapePriceFromIframe() {
        const iframes = document.querySelectorAll('iframe[src*="tradingview"], iframe[id*="tradingview"]');
        
        for (const iframe of iframes) {
            try {
                // Note: This will likely fail due to CORS restrictions
                const iframeDoc = iframe.contentDocument || iframe.contentWindow.document;
                if (iframeDoc) {
                    const priceElements = iframeDoc.querySelectorAll('[class*="price"], [data-field*="price"]');
                    
                    for (const element of priceElements) {
                        const text = element.textContent || '';
                        const priceMatch = text.match(/(\d{1,2}[,.]?\d{3}[.,]\d{2})/);
                        if (priceMatch) {
                            const price = parseFloat(priceMatch[1].replace(/,/g, ''));
                            if (price >= 2000 && price <= 5000) {
                                console.log('💰 Found iframe price:', price);
                                return price;
                            }
                        }
                    }
                }
            } catch (error) {
                console.log('🚫 Iframe access blocked (CORS):', error.message);
            }
        }
        
        return null;
    }
    
    // Advanced scraping with text analysis
    advancedPriceScraping() {
        // Look for any element containing gold price patterns
        const allElements = document.querySelectorAll('*');
        const pricePatterns = [
            /XAU.*?(\d{1,2}[,.]?\d{3}[.,]\d{2})/i,
            /GOLD.*?(\d{1,2}[,.]?\d{3}[.,]\d{2})/i,
            /(\d{1,2}[,.]?\d{3}[.,]\d{2}).*?XAU/i,
            /(\d{1,2}[,.]?\d{3}[.,]\d{2}).*?GOLD/i,
            /\$(\d{1,2}[,.]?\d{3}[.,]\d{2})/
        ];
        
        for (const element of allElements) {
            const text = element.textContent || element.innerText || '';
            if (text.length > 200) continue; // Skip large text blocks
            
            for (const pattern of pricePatterns) {
                const match = text.match(pattern);
                if (match) {
                    const price = parseFloat(match[1].replace(/,/g, ''));
                    if (price >= 2000 && price <= 5000) {
                        console.log('🎯 Advanced scraping found:', price, 'in text:', text.substring(0, 100));
                        return price;
                    }
                }
            }
        }
        
        return null;
    }
    
    // Main scraping function
    scrapeGoldPrice() {
        console.log('🔍 Scraping gold price from TradingView widget...');
        
        // Method 1: DOM scraping
        const domResult = this.scrapePriceFromDOM();
        if (domResult.price) {
            this.updatePrice(domResult.price, 'DOM Scraping');
            return domResult.price;
        }
        
        // Method 2: Iframe scraping (usually blocked)
        const iframePrice = this.scrapePriceFromIframe();
        if (iframePrice) {
            this.updatePrice(iframePrice, 'Iframe Scraping');
            return iframePrice;
        }
        
        // Method 3: Advanced text analysis
        const advancedPrice = this.advancedPriceScraping();
        if (advancedPrice) {
            this.updatePrice(advancedPrice, 'Advanced Scraping');
            return advancedPrice;
        }
        
        console.log('⚠️ No gold price found in current scraping attempt');
        return null;
    }
    
    // Update price with validation
    updatePrice(newPrice, source) {
        if (newPrice && newPrice !== this.currentPrice) {
            const change = this.currentPrice ? newPrice - this.currentPrice : 0;
            const changePercent = this.currentPrice ? ((change / this.currentPrice) * 100).toFixed(2) : 0;
            
            this.currentPrice = newPrice;
            this.lastUpdate = new Date();
            
            console.log('📈 Gold Price Update:', {
                price: `$${newPrice.toFixed(2)}`,
                change: `${change >= 0 ? '+' : ''}${change.toFixed(2)}`,
                changePercent: `${changePercent}%`,
                source: source,
                timestamp: this.lastUpdate.toISOString()
            });
            
            // Emit custom event
            const event = new CustomEvent('goldPriceUpdate', {
                detail: {
                    price: newPrice,
                    change: change,
                    changePercent: changePercent,
                    source: source,
                    timestamp: this.lastUpdate
                }
            });
            document.dispatchEvent(event);
            
            // Update UI if element exists
            this.updateUI(newPrice);
        }
    }
    
    // Update UI elements
    updateUI(price) {
        const priceElements = [
            document.getElementById('watchlist-xauusd-price'),
            document.querySelector('[data-symbol="XAUUSD"] .price-value'),
            ...document.querySelectorAll('.gold-price, .xau-price, [class*="gold-price"]')
        ];
        
        priceElements.forEach(element => {
            if (element) {
                element.textContent = `$${price.toFixed(2)}`;
                element.style.color = '#FFD700';
                element.style.backgroundColor = 'rgba(255, 215, 0, 0.1)';
                element.style.border = '1px solid #FFD700';
                element.style.padding = '2px 6px';
                element.style.borderRadius = '4px';
                
                console.log('🎨 Updated UI element:', element);
            }
        });
    }
    
    // Setup mutation observer to watch for widget changes
    setupMutationObserver() {
        const observer = new MutationObserver((mutations) => {
            let shouldScrape = false;
            
            mutations.forEach((mutation) => {
                if (mutation.type === 'childList' || mutation.type === 'characterData') {
                    // Check if any changed node contains price-related content
                    const nodes = [...mutation.addedNodes, ...mutation.removedNodes];
                    nodes.forEach(node => {
                        if (node.textContent && /\d{3,4}[.,]\d{2}/.test(node.textContent)) {
                            shouldScrape = true;
                        }
                    });
                }
            });
            
            if (shouldScrape) {
                console.log('🔄 DOM changes detected, re-scraping...');
                setTimeout(() => this.scrapeGoldPrice(), 100);
            }
        });
        
        // Observe the entire document for changes
        observer.observe(document.body, {
            childList: true,
            subtree: true,
            characterData: true
        });
        
        console.log('👀 Mutation observer set up for widget changes');
    }
    
    // Start periodic scraping
    startScraping() {
        // Initial scrape
        this.scrapeGoldPrice();
        
        // Periodic scraping every 5 seconds
        this.scrapingInterval = setInterval(() => {
            this.scrapeGoldPrice();
        }, 5000);
        
        console.log('⏰ Periodic scraping started (5 second intervals)');
    }
    
    // Stop scraping
    stopScraping() {
        if (this.scrapingInterval) {
            clearInterval(this.scrapingInterval);
            this.scrapingInterval = null;
            console.log('⏹️ Scraping stopped');
        }
    }
    
    // Get current price
    getCurrentPrice() {
        return {
            price: this.currentPrice,
            lastUpdate: this.lastUpdate,
            formatted: this.currentPrice ? `$${this.currentPrice.toFixed(2)}` : null
        };
    }
    
    // Manual trigger
    manualScrape() {
        console.log('🔧 Manual scraping triggered');
        return this.scrapeGoldPrice();
    }
}

// Global instance
window.tvGoldScraper = new TradingViewGoldScraper();

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.tvGoldScraper.init();
    });
} else {
    window.tvGoldScraper.init();
}

// Listen for gold price updates
document.addEventListener('goldPriceUpdate', (event) => {
    console.log('🎉 Gold Price Update Event:', event.detail);
});

// Debug utilities
window.debugTVScraper = {
    getCurrentPrice: () => window.tvGoldScraper.getCurrentPrice(),
    manualScrape: () => window.tvGoldScraper.manualScrape(),
    stopScraping: () => window.tvGoldScraper.stopScraping(),
    startScraping: () => window.tvGoldScraper.startScraping(),
    findWidgets: () => window.tvGoldScraper.findTradingViewWidgets()
};

console.log('🎯 TradingView Gold Scraper loaded! Use debugTVScraper for manual control.');
