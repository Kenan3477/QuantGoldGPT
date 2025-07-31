// Streamlined TradingView Chart Manager
// This module handles ONLY the visible TradingView widget and its controls

class TradingViewChartManager {
    constructor() {
        this.widget = null;
        this.currentSymbol = 'XAUUSD';
        this.currentTimeframe = '1h';
        this.isInitialized = false;
        this.initializationAttempts = 0;
        this.maxInitializationAttempts = 10;
        this.componentId = 'tradingview-chart';
        this.eventCleanup = [];
        
        console.log('📊 TradingView Chart Manager initialized');
    }

    async initialize() {
        if (this.isInitialized) return;

        try {
            // Set loading state
            if (window.connectionManager) {
                window.connectionManager.setLoading(this.componentId, true);
            }

            // Wait for TradingView library to load
            await this.waitForTradingView();
            
            // Initialize the widget
            this.createWidget();
            
            // Setup event listeners for controls
            this.setupControlEventListeners();
            
            // Setup retry handler
            this.setupRetryHandler();
            
            this.isInitialized = true;
            console.log('✅ TradingView Chart Manager ready');
            
            // Clear loading state
            if (window.connectionManager) {
                window.connectionManager.setLoading(this.componentId, false);
            }
            
            // Notify other systems that the chart is ready
            window.dispatchEvent(new CustomEvent('tradingview-chart-ready', {
                detail: { manager: this }
            }));
            
        } catch (error) {
            console.error('❌ Failed to initialize TradingView Chart Manager:', error);
            
            // Handle error through connection manager
            if (window.connectionManager) {
                window.connectionManager.handleError(this.componentId, error);
            }
            
            this.scheduleRetry();
        }
    }

    async waitForTradingView() {
        return new Promise((resolve, reject) => {
            const checkTradingView = () => {
                if (window.TradingView && typeof window.TradingView.widget === 'function') {
                    resolve();
                } else if (this.initializationAttempts >= this.maxInitializationAttempts) {
                    reject(new Error('TradingView library failed to load'));
                } else {
                    this.initializationAttempts++;
                    setTimeout(checkTradingView, 500);
                }
            };
            
            checkTradingView();
        });
    }

    createWidget() {
        const container = document.getElementById('tradingview-chart');
        if (!container) {
            throw new Error('TradingView chart container not found');
        }

        // Clear any existing content
        container.innerHTML = '';

        console.log('🚀 Creating TradingView widget...');

        this.widget = new TradingView.widget({
            "autosize": true,
            "symbol": this.getTradingViewSymbol(this.currentSymbol),
            "interval": this.convertTimeframeToTradingView(this.currentTimeframe),
            "timezone": "Etc/UTC",
            "theme": "dark",
            "style": "1",
            "locale": "en",
            "toolbar_bg": "#1a1a1a",
            "enable_publishing": false,
            "allow_symbol_change": false,
            "container_id": "tradingview-chart",
            "width": "100%",
            "height": "100%",
            "hide_top_toolbar": false,
            "hide_legend": false,
            "hide_side_toolbar": false,
            "save_image": false,
            "details": true,
            "hotlist": false,
            "calendar": false,
            "news": [],
            "studies": [
                "Volume@tv-basicstudies",
                "RSI@tv-basicstudies"
            ],
            "backgroundColor": "#1a1a1a",
            "gridColor": "rgba(255,255,255,0.1)",
            "loading_screen": {
                "backgroundColor": "#1a1a1a",
                "foregroundColor": "#0088ff"
            },
            "overrides": {
                "paneProperties.background": "#1a1a1a",
                "paneProperties.backgroundType": "solid",
                "scalesProperties.textColor": "#ffffff",
                "scalesProperties.backgroundColor": "#2a2a2a"
            }
        });

        console.log('✅ TradingView widget created successfully');

        // Store reference globally for other components
        window.mainTVWidget = this.widget;
    }

    setupControlEventListeners() {
        // Timeframe controls
        this.setupTimeframeControls();
        
        // Symbol switching controls
        this.setupSymbolControls();
        
        // Indicator controls
        this.setupIndicatorControls();
    }

    setupTimeframeControls() {
        const timeframeButtons = document.querySelectorAll('.timeframe-btn');
        
        timeframeButtons.forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                const timeframe = btn.dataset.timeframe;
                if (timeframe) {
                    this.changeTimeframe(timeframe);
                }
            });
        });

        console.log(`📊 Setup ${timeframeButtons.length} timeframe controls`);
    }

    setupSymbolControls() {
        const symbolButtons = document.querySelectorAll('.watchlist-item');
        
        symbolButtons.forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                const symbol = btn.dataset.symbol;
                if (symbol) {
                    this.changeSymbol(symbol);
                }
            });
        });

        console.log(`📊 Setup ${symbolButtons.length} symbol controls`);
    }

    setupIndicatorControls() {
        const indicatorButtons = document.querySelectorAll('.indicator-btn');
        
        indicatorButtons.forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.preventDefault();
                const indicator = btn.dataset.indicator;
                if (indicator) {
                    this.toggleIndicator(indicator, btn);
                }
            });
        });

        console.log(`📊 Setup ${indicatorButtons.length} indicator controls`);
    }

    changeTimeframe(timeframe) {
        if (!this.widget || !this.isInitialized) {
            console.warn('⚠️ TradingView widget not ready for timeframe change');
            return;
        }

        console.log(`📊 Changing timeframe to ${timeframe}`);

        try {
            // Update UI state
            document.querySelectorAll('.timeframe-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            
            const activeBtn = document.querySelector(`[data-timeframe="${timeframe}"]`);
            if (activeBtn) {
                activeBtn.classList.add('active');
            }

            // Convert timeframe and update widget
            const tvTimeframe = this.convertTimeframeToTradingView(timeframe);
            
            this.widget.chart().setResolution(tvTimeframe, () => {
                console.log(`✅ TradingView timeframe changed to ${timeframe}`);
                this.currentTimeframe = timeframe;
                
                // Show notification
                this.showNotification(`📊 Switched to ${timeframe.toUpperCase()} timeframe`, 'success');
                
                // Notify other systems about timeframe change
                window.dispatchEvent(new CustomEvent('timeframe-changed', {
                    detail: { timeframe, tvTimeframe }
                }));
            });

        } catch (error) {
            console.error('❌ Error changing timeframe:', error);
            this.showNotification('❌ Failed to change timeframe', 'error');
        }
    }

    changeSymbol(symbol) {
        if (!this.widget || !this.isInitialized) {
            console.warn('⚠️ TradingView widget not ready for symbol change');
            return;
        }

        console.log(`🔄 Changing symbol to ${symbol}`);

        try {
            // Update UI state
            document.querySelectorAll('.watchlist-item').forEach(item => {
                item.classList.remove('active');
            });
            
            const activeItem = document.querySelector(`[data-symbol="${symbol}"]`);
            if (activeItem) {
                activeItem.classList.add('active');
            }

            // Update symbol display
            const symbolBadge = document.querySelector('.symbol-badge');
            const symbolDetails = document.querySelector('.symbol-details h3');
            
            if (symbolBadge) symbolBadge.textContent = this.getSymbolBadge(symbol);
            if (symbolDetails) symbolDetails.textContent = this.getSymbolDisplayName(symbol);

            // Convert symbol and update widget
            const tvSymbol = this.getTradingViewSymbol(symbol);
            
            this.widget.chart().setSymbol(tvSymbol, () => {
                console.log(`✅ TradingView symbol changed to ${symbol}`);
                this.currentSymbol = symbol;
                
                // Show notification
                this.showNotification(`🔄 Switched to ${this.getSymbolDisplayName(symbol)}`, 'info');
                
                // Notify other systems about symbol change
                window.dispatchEvent(new CustomEvent('symbol-changed', {
                    detail: { symbol, tvSymbol }
                }));
            });

        } catch (error) {
            console.error('❌ Error changing symbol:', error);
            this.showNotification('❌ Failed to change symbol', 'error');
        }
    }

    toggleIndicator(indicator, buttonElement) {
        if (!this.widget || !this.isInitialized) {
            console.warn('⚠️ TradingView widget not ready for indicator toggle');
            return;
        }

        console.log(`📈 Toggling ${indicator} indicator`);

        try {
            const isActive = buttonElement.classList.contains('active');
            
            if (isActive) {
                // Remove indicator
                buttonElement.classList.remove('active');
                this.showNotification(`📉 ${indicator.toUpperCase()} indicator hidden`, 'info');
            } else {
                // Add indicator
                buttonElement.classList.add('active');
                
                // Map indicator names to TradingView study names
                const studyMap = {
                    'volume': 'Volume@tv-basicstudies',
                    'rsi': 'RSI@tv-basicstudies',
                    'macd': 'MACD@tv-basicstudies',
                    'bb': 'BB@tv-basicstudies'
                };
                
                const studyName = studyMap[indicator];
                if (studyName) {
                    this.widget.chart().createStudy(studyName, false, false);
                }
                
                this.showNotification(`📈 ${indicator.toUpperCase()} indicator shown`, 'success');
            }

            // Notify other systems about indicator change
            window.dispatchEvent(new CustomEvent('indicator-toggled', {
                detail: { indicator, active: !isActive }
            }));

        } catch (error) {
            console.error('❌ Error toggling indicator:', error);
            this.showNotification('❌ Failed to toggle indicator', 'error');
        }
    }

    // Helper methods
    getTradingViewSymbol(symbol) {
        const symbolMap = {
            'XAUUSD': 'OANDA:XAUUSD',
            'XAGUSD': 'OANDA:XAGUSD',
            'EURUSD': 'OANDA:EURUSD',
            'GBPUSD': 'OANDA:GBPUSD',
            'USDJPY': 'OANDA:USDJPY',
            'BTCUSD': 'COINBASE:BTCUSD',
            'SPY': 'AMEX:SPY',
            'QQQ': 'NASDAQ:QQQ'
        };
        return symbolMap[symbol] || `OANDA:${symbol}`;
    }

    convertTimeframeToTradingView(timeframe) {
        const timeframeMap = {
            '1m': '1',
            '5m': '5',
            '15m': '15',
            '1h': '60',
            '4h': '240',
            '1d': 'D',
            '1w': 'W'
        };
        return timeframeMap[timeframe] || '60';
    }

    getSymbolDisplayName(symbol) {
        const nameMap = {
            'XAUUSD': 'XAU/USD - Gold Spot',
            'XAGUSD': 'XAG/USD - Silver Spot',
            'EURUSD': 'EUR/USD - Euro Dollar',
            'GBPUSD': 'GBP/USD - British Pound',
            'USDJPY': 'USD/JPY - Dollar Yen',
            'BTCUSD': 'BTC/USD - Bitcoin',
            'SPY': 'SPY - S&P 500 ETF',
            'QQQ': 'QQQ - NASDAQ 100 ETF'
        };
        return nameMap[symbol] || symbol;
    }

    getSymbolBadge(symbol) {
        const badgeMap = {
            'XAUUSD': 'GOLD',
            'XAGUSD': 'SILVER',
            'EURUSD': 'EUR',
            'GBPUSD': 'GBP',
            'USDJPY': 'JPY',
            'BTCUSD': 'BTC',
            'SPY': 'SPY',
            'QQQ': 'QQQ'
        };
        return badgeMap[symbol] || symbol;
    }

    showNotification(message, type = 'info') {
        // Create and show notification
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = message;
        notification.style.cssText = `
            position: fixed;
            top: 80px;
            right: 20px;
            background: var(--bg-secondary, #2a2a2a);
            color: var(--text-primary, #ffffff);
            padding: 12px 20px;
            border-radius: 8px;
            border-left: 4px solid ${this.getNotificationColor(type)};
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            z-index: 1000;
            animation: slideInRight 0.3s ease;
            font-size: 14px;
            max-width: 300px;
        `;

        // Add CSS animation if not already present
        if (!document.getElementById('notification-styles')) {
            const style = document.createElement('style');
            style.id = 'notification-styles';
            style.textContent = `
                @keyframes slideInRight {
                    from { transform: translateX(100%); opacity: 0; }
                    to { transform: translateX(0); opacity: 1; }
                }
                @keyframes slideOutRight {
                    from { transform: translateX(0); opacity: 1; }
                    to { transform: translateX(100%); opacity: 0; }
                }
            `;
            document.head.appendChild(style);
        }

        document.body.appendChild(notification);

        // Remove after 3 seconds
        setTimeout(() => {
            notification.style.animation = 'slideOutRight 0.3s ease';
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }

    getNotificationColor(type) {
        const colors = {
            success: '#00ff88',
            error: '#ff4444',
            warning: '#ffaa00',
            info: '#0088ff'
        };
        return colors[type] || colors.info;
    }

    scheduleRetry() {
        if (this.initializationAttempts < this.maxInitializationAttempts) {
            console.log(`🔄 Retrying TradingView initialization (${this.initializationAttempts}/${this.maxInitializationAttempts})...`);
            setTimeout(() => this.initialize(), 2000);
        } else {
            console.error('❌ Max initialization attempts reached for TradingView Chart Manager');
            this.showNotification('❌ Failed to load TradingView chart', 'error');
        }
    }

    // Public API methods
    getCurrentSymbol() {
        return this.currentSymbol;
    }

    getCurrentTimeframe() {
        return this.currentTimeframe;
    }

    getWidget() {
        return this.widget;
    }

    isReady() {
        return this.isInitialized && this.widget !== null;
    }

    /**
     * Setup retry handler for connection manager
     */
    setupRetryHandler() {
        if (window.connectionManager) {
            const cleanup = window.connectionManager.on('retry', (data) => {
                if (data.componentId === this.componentId) {
                    console.log('🔄 Retrying TradingView chart initialization...');
                    this.initialize();
                }
            });
            this.eventCleanup.push(cleanup);
        }
    }

    /**
     * Cleanup all event listeners and resources
     */
    cleanup() {
        console.log('🧹 Cleaning up TradingView Chart Manager...');
        
        // Clean up event listeners
        this.eventCleanup.forEach(cleanup => cleanup());
        this.eventCleanup = [];
        
        // Clean up from connection manager
        if (window.connectionManager) {
            window.connectionManager.offContext(this);
        }
        
        this.destroy();
    }

    destroy() {
        if (this.widget) {
            try {
                this.widget.remove();
            } catch (error) {
                console.warn('Warning during widget removal:', error);
            }
        }
        
        this.widget = null;
        this.isInitialized = false;
        console.log('🗑️ TradingView Chart Manager destroyed');
    }
}

// Create global instance for component loader
window.tradingViewChartManager = new TradingViewChartManager();

// Add init method for component loader compatibility
window.tradingViewChartManager.init = function() {
    return this.initialize();
};

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        // Delay initialization slightly to ensure TradingView library is loaded
        setTimeout(() => {
            window.tradingViewChartManager.initialize();
        }, 1000);
    });
} else {
    setTimeout(() => {
        window.tradingViewChartManager.initialize();
    }, 1000);
}

console.log('📊 TradingView Chart Manager loaded successfully');
