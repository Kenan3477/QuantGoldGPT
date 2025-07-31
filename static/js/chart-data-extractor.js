/**
 * TradingView Chart Data Extractor for GoldGPT
 * Extracts real-time data from TradingView widgets displayed in the dashboard
 * Allows your bot to access the same data displayed on the charts
 */

class TradingViewDataExtractor {
    constructor() {
        this.isExtracting = false;
        this.extractedData = null;
        this.lastUpdateTime = null;
        this.subscribers = [];
        
        console.log('🔍 TradingView Data Extractor initialized');
        this.init();
    }
    
    init() {
        // Monitor for TradingView widgets
        this.monitorTradingViewWidgets();
        
        // Set up periodic data extraction
        this.startPeriodicExtraction();
    }
    
    /**
     * Monitor for TradingView widgets being loaded
     */
    monitorTradingViewWidgets() {
        // Watch for TradingView iframe loading
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                mutation.addedNodes.forEach((node) => {
                    if (node.nodeType === 1) { // Element node
                        // Check if it's a TradingView iframe
                        const tvIframes = node.querySelectorAll ? 
                            node.querySelectorAll('iframe[src*="tradingview"]') : [];
                        
                        if (tvIframes.length > 0 || (node.src && node.src.includes('tradingview'))) {
                            console.log('📊 TradingView widget detected, setting up data extraction');
                            this.setupWidgetDataExtraction(node);
                        }
                    }
                });
            });
        });
        
        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
        
        // Check for existing widgets
        this.checkExistingWidgets();
    }
    
    /**
     * Check for existing TradingView widgets
     */
    checkExistingWidgets() {
        const existingIframes = document.querySelectorAll('iframe[src*="tradingview"]');
        existingIframes.forEach(iframe => {
            this.setupWidgetDataExtraction(iframe);
        });
        
        // Also check for TradingView divs that might contain widgets
        const tvContainers = document.querySelectorAll('[id*="tradingview"], [class*="tradingview"]');
        if (tvContainers.length > 0) {
            console.log(`🎯 Found ${tvContainers.length} potential TradingView containers`);
        }
    }
    
    /**
     * Set up data extraction from a TradingView widget
     */
    setupWidgetDataExtraction(widget) {
        try {
            // Listen for postMessage from TradingView iframe
            window.addEventListener('message', (event) => {
                if (event.origin.includes('tradingview.com')) {
                    this.handleTradingViewMessage(event.data);
                }
            });
            
            console.log('✅ TradingView widget data extraction setup complete');
        } catch (error) {
            console.error('❌ Error setting up widget extraction:', error);
        }
    }
    
    /**
     * Handle messages from TradingView widgets
     */
    handleTradingViewMessage(data) {
        try {
            // TradingView sends various message types
            if (data && typeof data === 'object') {
                if (data.name === 'quote-data' || data.type === 'quote') {
                    this.extractQuoteData(data);
                } else if (data.name === 'chart-data' || data.type === 'chart') {
                    this.extractChartData(data);
                }
            }
        } catch (error) {
            console.error('❌ Error handling TradingView message:', error);
        }
    }
    
    /**
     * Extract quote/price data
     */
    extractQuoteData(data) {
        try {
            const extractedQuote = {
                symbol: data.symbol || 'UNKNOWN',
                price: data.last || data.price || data.close,
                bid: data.bid,
                ask: data.ask,
                volume: data.volume,
                change: data.change,
                changePercent: data.changePercent || data.change_percent,
                timestamp: Date.now(),
                source: 'TradingView'
            };
            
            this.updateExtractedData('quote', extractedQuote);
            console.log('📈 Quote data extracted:', extractedQuote);
        } catch (error) {
            console.error('❌ Error extracting quote data:', error);
        }
    }
    
    /**
     * Extract chart OHLCV data
     */
    extractChartData(data) {
        try {
            if (data.bars || data.candles || data.ohlcv) {
                const chartData = {
                    symbol: data.symbol || 'UNKNOWN',
                    timeframe: data.timeframe || data.interval,
                    bars: data.bars || data.candles || data.ohlcv,
                    timestamp: Date.now(),
                    source: 'TradingView'
                };
                
                this.updateExtractedData('chart', chartData);
                console.log('📊 Chart data extracted:', chartData);
            }
        } catch (error) {
            console.error('❌ Error extracting chart data:', error);
        }
    }
    
    /**
     * Alternative method: Extract data from DOM elements
     */
    extractFromDOM() {
        try {
            const extractedData = {};
            
            // Look for price displays in the page
            const priceSelectors = [
                '[data-symbol-price]',
                '.tv-symbol-price-quote__value',
                '.js-symbol-last',
                '.price-value',
                '#current-price'
            ];
            
            priceSelectors.forEach(selector => {
                const elements = document.querySelectorAll(selector);
                elements.forEach(el => {
                    const price = this.parsePrice(el.textContent);
                    if (price) {
                        extractedData.price = price;
                        extractedData.source = 'DOM';
                        extractedData.timestamp = Date.now();
                    }
                });
            });
            
            // Look for volume, change, etc.
            const volumeEl = document.querySelector('[data-symbol-volume], .tv-symbol-price-quote__volume');
            if (volumeEl) {
                extractedData.volume = this.parseNumber(volumeEl.textContent);
            }
            
            const changeEl = document.querySelector('[data-symbol-change], .tv-symbol-price-quote__change');
            if (changeEl) {
                extractedData.change = this.parseNumber(changeEl.textContent);
            }
            
            if (Object.keys(extractedData).length > 0) {
                this.updateExtractedData('dom', extractedData);
                console.log('🏷️ DOM data extracted:', extractedData);
            }
            
            return extractedData;
        } catch (error) {
            console.error('❌ Error extracting from DOM:', error);
            return null;
        }
    }
    
    /**
     * Parse price from text content
     */
    parsePrice(text) {
        if (!text) return null;
        
        // Remove currency symbols, commas, spaces
        const cleaned = text.replace(/[$€£¥,\s]/g, '');
        const number = parseFloat(cleaned);
        
        return isNaN(number) ? null : number;
    }
    
    /**
     * Parse any number from text
     */
    parseNumber(text) {
        if (!text) return null;
        
        const cleaned = text.replace(/[,%\s]/g, '');
        const number = parseFloat(cleaned);
        
        return isNaN(number) ? null : number;
    }
    
    /**
     * Update extracted data and notify subscribers
     */
    updateExtractedData(type, data) {
        if (!this.extractedData) this.extractedData = {};
        
        this.extractedData[type] = data;
        this.lastUpdateTime = Date.now();
        
        // Notify subscribers
        this.notifySubscribers(type, data);
        
        // Send to backend
        this.sendToBackend(type, data);
    }
    
    /**
     * Send extracted data to GoldGPT backend
     */
    async sendToBackend(type, data) {
        try {
            const response = await fetch('/api/chart/extracted-data', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    type: type,
                    data: data,
                    extractedAt: Date.now()
                })
            });
            
            if (response.ok) {
                console.log('✅ Extracted data sent to backend');
            }
        } catch (error) {
            console.error('❌ Error sending data to backend:', error);
        }
    }
    
    /**
     * Subscribe to data updates
     */
    subscribe(callback) {
        this.subscribers.push(callback);
        
        // Send current data if available
        if (this.extractedData) {
            callback(this.extractedData);
        }
    }
    
    /**
     * Notify all subscribers of data updates
     */
    notifySubscribers(type, data) {
        this.subscribers.forEach(callback => {
            try {
                callback({ type, data, timestamp: this.lastUpdateTime });
            } catch (error) {
                console.error('❌ Error notifying subscriber:', error);
            }
        });
    }
    
    /**
     * Start periodic data extraction
     */
    startPeriodicExtraction() {
        // Extract DOM data every 5 seconds
        setInterval(() => {
            this.extractFromDOM();
        }, 5000);
        
        console.log('🔄 Periodic data extraction started');
    }
    
    /**
     * Get current extracted data
     */
    getCurrentData() {
        return {
            data: this.extractedData,
            lastUpdate: this.lastUpdateTime,
            isActive: this.isExtracting
        };
    }
    
    /**
     * Export data for bot consumption
     */
    exportForBot() {
        const exportData = {
            timestamp: Date.now(),
            lastUpdate: this.lastUpdateTime,
            extractedData: this.extractedData
        };
        
        // Create downloadable JSON
        const blob = new Blob([JSON.stringify(exportData, null, 2)], {
            type: 'application/json'
        });
        
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `tradingview-data-${Date.now()}.json`;
        a.click();
        
        URL.revokeObjectURL(url);
        
        console.log('📥 Data exported for bot consumption');
    }
}

// Initialize the extractor
window.tradingViewExtractor = new TradingViewDataExtractor();

// Export for global access
window.TradingViewDataExtractor = TradingViewDataExtractor;

// Auto-start extraction when page loads
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 TradingView Data Extractor ready');
    
    // Expose global functions for bot access
    window.extractTradingViewData = () => window.tradingViewExtractor.getCurrentData();
    window.exportTradingViewData = () => window.tradingViewExtractor.exportForBot();
});

console.log('📊 TradingView Chart Data Extractor loaded');
