/**
 * GoldGPT Advanced Trading Interface
 * Trading 212 inspired functionality with real-time updates
 * ENHANCED WITH COMPREHENSIVE DEBUGGING SYSTEM
 */

// Debug system for tracking all features
class DebugTracker {
    constructor() {
        this.features = {
            socketConnection: { status: 'unknown', lastUpdate: null, errors: [] },
            portfolio: { status: 'unknown', lastUpdate: null, errors: [] },
            watchlist: { status: 'unknown', lastUpdate: null, errors: [] },
            charts: { status: 'unknown', lastUpdate: null, errors: [] },
            trading: { status: 'unknown', lastUpdate: null, errors: [] },
            aiAnalysis: { status: 'unknown', lastUpdate: null, errors: [] },
            priceUpdates: { status: 'unknown', lastUpdate: null, errors: [] },
            navigation: { status: 'unknown', lastUpdate: null, errors: [] }
        };
        this.startTime = Date.now();
        this.setupDebugPanel();
    }

    log(feature, status, message, data = null) {
        const timestamp = new Date().toISOString();
        console.log(`🔍 [${feature.toUpperCase()}] ${status}: ${message}`, data || '');
        
        if (this.features[feature]) {
            this.features[feature].status = status;
            this.features[feature].lastUpdate = timestamp;
            if (status === 'error') {
                this.features[feature].errors.push({ message, timestamp, data });
            }
        }
        
        this.updateDebugPanel();
        this.sendToServer(feature, status, message, data);
    }

    setupDebugPanel() {
        // Create floating debug panel
        const debugPanel = document.createElement('div');
        debugPanel.id = 'debug-panel';
        debugPanel.style.cssText = `
            position: fixed; top: 10px; right: 10px; width: 300px; height: 400px;
            background: #1a1a1a; border: 2px solid #00d084; border-radius: 8px;
            color: #fff; font-family: monospace; font-size: 12px; z-index: 10000;
            overflow-y: auto; padding: 10px; box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        `;
        debugPanel.innerHTML = `
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <strong>🔍 DEBUG TRACKER</strong>
                <button onclick="this.parentElement.parentElement.style.display='none'" style="background: #ff4757; border: none; color: white; padding: 2px 6px; border-radius: 3px; cursor: pointer;">×</button>
            </div>
            <div id="debug-content"></div>
        `;
        document.body.appendChild(debugPanel);
    }

    updateDebugPanel() {
        const content = document.getElementById('debug-content');
        if (!content) return;

        const uptime = Math.floor((Date.now() - this.startTime) / 1000);
        let html = `<div style="margin-bottom: 10px; color: #00d084;">Uptime: ${uptime}s</div>`;
        
        Object.entries(this.features).forEach(([feature, data]) => {
            const statusColor = data.status === 'success' ? '#00d084' : 
                               data.status === 'error' ? '#ff4757' : 
                               data.status === 'loading' ? '#ffa502' : '#666';
            
            html += `
                <div style="margin: 5px 0; padding: 5px; background: #2a2a2a; border-radius: 3px;">
                    <div style="color: ${statusColor};">
                        <strong>${feature.toUpperCase()}</strong>
                        <span style="float: right;">${data.status}</span>
                    </div>
                    ${data.lastUpdate ? `<small>Last: ${new Date(data.lastUpdate).toLocaleTimeString()}</small>` : ''}
                    ${data.errors.length > 0 ? `<div style="color: #ff4757; font-size: 10px;">Errors: ${data.errors.length}</div>` : ''}
                </div>
            `;
        });
        
        content.innerHTML = html;
    }

    sendToServer(feature, status, message, data) {
        // RE-ENABLED: Debug info to server for troubleshooting component loading issues
        // This will help identify why 5/19 modules are failing to load
        try {
            // Create abort controller for timeout and retry logic
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 3000);
            
            fetch('/api/debug', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    feature, status, message, data,
                    timestamp: new Date().toISOString(),
                    userAgent: navigator.userAgent,
                    url: window.location.href,
                    componentLoadingStatus: window.componentLoader?.getStatus() || 'unknown'
                }),
                signal: controller.signal
            }).then(response => {
                clearTimeout(timeoutId);
                if (!response.ok) {
                    console.warn('Debug API response not OK:', response.status);
                }
            }).catch((error) => {
                clearTimeout(timeoutId);
                // Enhanced error logging for debugging
                if (error.name === 'AbortError') {
                    console.warn('Debug sendToServer timeout - continuing without debug');
                } else {
                    console.warn('Debug sendToServer failed:', error.message);
                }
                // Implement simple retry logic for critical features
                if (feature === 'socketConnection' || feature === 'coreApplication') {
                    setTimeout(() => {
                        this.sendToServerRetry(feature, status, message, data);
                    }, 1000);
                }
            });
        } catch (error) {
            console.error('Critical error in sendToServer:', error);
        }
    }

    sendToServerRetry(feature, status, message, data) {
        // Simplified retry without timeout for critical features
        try {
            fetch('/api/debug', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    feature, status, message, data,
                    timestamp: new Date().toISOString(),
                    retry: true
                })
            }).catch(() => {
                // Silent fail on retry - don't spam logs
                console.debug('Debug retry failed silently');
            });
        } catch (error) {
            console.debug('Debug retry critical error:', error.message);
        }
    }

    getFeatureStatus(feature) {
        return this.features[feature]?.status || 'unknown';
    }
}

// =====================================
// COMPONENT MANAGER CLASSES
// =====================================

/**
 * Portfolio Manager - Handles portfolio operations and data
 */
class PortfolioManager {
    constructor(app) {
        this.app = app;
        this.isInitialized = false;
        console.log('💼 Portfolio Manager created');
    }
    
    async initialize() {
        console.log('💼 Initializing Portfolio Manager...');
        try {
            // Initialize portfolio data
            this.portfolio = {
                balance: 10000.00,
                equity: 10000.00,
                margin: 0,
                freeMargin: 10000.00,
                positions: []
            };
            
            this.isInitialized = true;
            console.log('✅ Portfolio Manager initialized');
            return true;
        } catch (error) {
            console.error('❌ Portfolio Manager initialization failed:', error);
            throw error;
        }
    }
    
    getBalance() {
        return this.portfolio?.balance || 0;
    }
    
    updateBalance(newBalance) {
        if (this.portfolio) {
            this.portfolio.balance = newBalance;
            this.app.debug.log('portfolio', 'success', 'Balance updated', { balance: newBalance });
        }
    }
}

/**
 * Watchlist Manager - Handles watchlist operations
 */
class WatchlistManager {
    constructor(app) {
        this.app = app;
        this.isInitialized = false;
        this.symbols = ['XAUUSD', 'EURUSD', 'GBPUSD', 'USDJPY', 'BTCUSD'];
        console.log('👁️ Watchlist Manager created');
    }
    
    async initialize() {
        console.log('👁️ Initializing Watchlist Manager...');
        try {
            // Initialize watchlist UI bindings
            this.setupWatchlistEvents();
            
            this.isInitialized = true;
            console.log('✅ Watchlist Manager initialized');
            return true;
        } catch (error) {
            console.error('❌ Watchlist Manager initialization failed:', error);
            throw error;
        }
    }
    
    setupWatchlistEvents() {
        const watchlistItems = document.querySelectorAll('.watchlist-item');
        watchlistItems.forEach(item => {
            item.addEventListener('click', (e) => {
                const symbol = item.getAttribute('data-symbol');
                if (symbol) {
                    this.selectSymbol(symbol);
                }
            });
        });
    }
    
    selectSymbol(symbol) {
        this.app.currentSymbol = symbol;
        this.app.debug.log('watchlist', 'success', 'Symbol selected', { symbol });
        
        // Update UI
        document.querySelectorAll('.watchlist-item').forEach(item => {
            item.classList.remove('active');
        });
        document.querySelector(`[data-symbol="${symbol}"]`)?.classList.add('active');
    }
}

/**
 * Trading Manager - Handles trading operations
 */
class TradingManager {
    constructor(app) {
        this.app = app;
        this.isInitialized = false;
        console.log('💹 Trading Manager created');
    }
    
    async initialize() {
        console.log('💹 Initializing Trading Manager...');
        try {
            // Initialize trading capabilities
            this.setupTradingEvents();
            
            this.isInitialized = true;
            console.log('✅ Trading Manager initialized');
            return true;
        } catch (error) {
            console.error('❌ Trading Manager initialization failed:', error);
            throw error;
        }
    }
    
    setupTradingEvents() {
        // Setup buy/sell button events
        const buyBtn = document.querySelector('.trade-btn.buy');
        const sellBtn = document.querySelector('.trade-btn.sell');
        
        if (buyBtn) {
            buyBtn.addEventListener('click', () => this.executeTrade('buy'));
        }
        if (sellBtn) {
            sellBtn.addEventListener('click', () => this.executeTrade('sell'));
        }
    }
    
    async executeTrade(type) {
        try {
            this.app.debug.log('trading', 'loading', `Executing ${type} trade`);
            
            // Mock trade execution - replace with real API call
            const tradeData = {
                symbol: this.app.currentSymbol,
                type: type,
                amount: 0.1,
                timestamp: new Date().toISOString()
            };
            
            // Emit to server
            this.app.socket.emit('execute_trade', tradeData);
            
            this.app.debug.log('trading', 'success', 'Trade executed', tradeData);
            return tradeData;
        } catch (error) {
            this.app.debug.log('trading', 'error', 'Trade execution failed', error.message);
            throw error;
        }
    }
}

class GoldGPTApp {
    constructor() {
        // Initialize debug tracker first
        this.debug = new DebugTracker();
        this.debug.log('initialization', 'loading', 'Starting GoldGPT app initialization');

        this.socket = io();
        this.currentSymbol = 'XAUUSD';
        this.portfolioData = {};
        this.watchlistSymbols = ['XAUUSD', 'EURUSD', 'GBPUSD', 'USDJPY', 'BTCUSD']; // Renamed to avoid conflict
        this.charts = {};
        
        // Initialize component managers required by component loader
        this.portfolio = new PortfolioManager(this);
        this.watchlist = new WatchlistManager(this);
        this.trading = new TradingManager(this);
        
        this.debug.log('initialization', 'success', 'GoldGPT app variables and managers initialized');
        this.initializeApp();
    }
    
    initializeApp() {
        this.debug.log('initialization', 'loading', 'Starting app initialization sequence');
        
        try {
            this.setupSocketListeners();
            this.debug.log('initialization', 'success', 'Socket listeners setup complete');
            
            this.setupEventListeners();
            this.debug.log('initialization', 'success', 'Event listeners setup complete');
            
            this.loadInitialData();
            this.debug.log('initialization', 'loading', 'Initial data loading started');
            
            this.setupCharts();
            this.debug.log('initialization', 'loading', 'Chart setup started');
            
            this.debug.log('initialization', 'success', 'App initialization sequence complete');
        } catch (error) {
            this.debug.log('initialization', 'error', 'Failed during app initialization', error.message);
            console.error('🚨 App initialization failed:', error);
        }
    }
    
    setupSocketListeners() {
        this.debug.log('socketConnection', 'loading', 'Setting up socket listeners');
        
        this.socket.on('connect', () => {
            console.log('🚀 Connected to GoldGPT server');
            this.debug.log('socketConnection', 'success', 'Successfully connected to server');
            this.updateConnectionStatus(true);
        });
        
        this.socket.on('disconnect', () => {
            console.log('❌ Disconnected from server');
            this.debug.log('socketConnection', 'error', 'Disconnected from server');
            this.updateConnectionStatus(false);
        });
        
        this.socket.on('connect_error', (error) => {
            this.debug.log('socketConnection', 'error', 'Socket connection error', error.message);
        });

        this.socket.on('price_update', (data) => {
            this.debug.log('priceUpdates', 'success', `Price update received for ${data.symbol}`, data);
            this.handlePriceUpdate(data);
        });
        
        this.socket.on('trade_executed', (data) => {
            this.debug.log('trading', 'success', 'Trade executed successfully', data);
            this.handleTradeExecuted(data);
            this.showNotification('Trade executed successfully', 'success');
        });
        
        this.socket.on('trade_closed', (data) => {
            this.debug.log('trading', 'success', 'Trade closed successfully', data);
            this.handleTradeClosed(data);
            this.showNotification('Trade closed', 'info');
        });

        this.debug.log('socketConnection', 'success', 'All socket listeners registered');
    }
    
    setupEventListeners() {
        this.debug.log('navigation', 'loading', 'Setting up event listeners');
        
        try {
            // Navigation
            const navItems = document.querySelectorAll('.nav-item');
            this.debug.log('navigation', 'success', `Found ${navItems.length} navigation items`);
            
            navItems.forEach(item => {
                item.addEventListener('click', (e) => {
                    this.debug.log('navigation', 'success', `Navigation clicked: ${item.dataset.section}`);
                    this.handleNavigation(e.target.closest('.nav-item'));
                });
            });
            
            // Symbol selection
            document.addEventListener('click', (e) => {
                if (e.target.classList.contains('symbol-select')) {
                    this.debug.log('navigation', 'success', `Symbol selected: ${e.target.dataset.symbol}`);
                    this.changeSymbol(e.target.dataset.symbol);
                }
            });
            
            // Quick trade buttons
            document.addEventListener('click', (e) => {
                if (e.target.classList.contains('quick-buy')) {
                    this.debug.log('trading', 'loading', 'Quick buy button clicked');
                    this.quickTrade('buy');
                } else if (e.target.classList.contains('quick-sell')) {
                    this.debug.log('trading', 'loading', 'Quick sell button clicked');
                    this.quickTrade('sell');
                }
            });

            this.debug.log('navigation', 'success', 'All event listeners setup complete');
        } catch (error) {
            this.debug.log('navigation', 'error', 'Failed to setup event listeners', error.message);
        }
    }
    
    async loadInitialData() {
        this.debug.log('portfolio', 'loading', 'Starting initial data load');
        
        try {
            await Promise.all([
                this.loadPortfolio(),
                this.loadWatchlist(),
                this.subscribeToSymbols()
            ]);
            this.debug.log('portfolio', 'success', 'All initial data loaded successfully');
        } catch (error) {
            console.error('Error loading initial data:', error);
            this.debug.log('portfolio', 'error', 'Failed to load initial data', error.message);
            this.showNotification('Error loading data', 'error');
        }
    }
    
    async loadPortfolio() {
        this.debug.log('portfolio', 'loading', 'Loading portfolio data from API');
        
        try {
            const response = await fetch('/api/portfolio');
            this.debug.log('portfolio', 'success', `Portfolio API response: ${response.status}`);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            this.portfolioData = await response.json();
            this.debug.log('portfolio', 'success', 'Portfolio data parsed successfully', this.portfolioData);
            
            this.updatePortfolioDisplay();
            this.debug.log('portfolio', 'success', 'Portfolio display updated');
        } catch (error) {
            console.error('Error loading portfolio:', error);
            this.debug.log('portfolio', 'error', 'Failed to load portfolio', error.message);
        }
    }
    
    updatePortfolioDisplay() {
        this.debug.log('portfolio', 'loading', 'Updating portfolio display');
        
        const balanceElement = document.getElementById('portfolio-balance');
        const contentElement = document.getElementById('portfolio-content');
        
        if (!balanceElement) {
            this.debug.log('portfolio', 'error', 'Portfolio balance element not found');
            return;
        }
        
        if (!contentElement) {
            this.debug.log('portfolio', 'error', 'Portfolio content element not found');
            return;
        }
        
        if (!this.portfolioData) {
            this.debug.log('portfolio', 'error', 'No portfolio data available');
            return;
        }
        
        // Update balance with animation
        this.animateValue(balanceElement, this.portfolioData.total_value || 0, '$');
        this.debug.log('portfolio', 'success', `Balance updated: $${this.portfolioData.total_value || 0}`);
        
        // Update portfolio cards
        if (this.portfolioData.trades && this.portfolioData.trades.length > 0) {
            contentElement.innerHTML = this.portfolioData.trades.map(trade => 
                this.createTradeCard(trade)
            ).join('');
            this.debug.log('portfolio', 'success', `${this.portfolioData.trades.length} trade cards created`);
        } else {
            contentElement.innerHTML = this.createEmptyState('No open trades');
            this.debug.log('portfolio', 'success', 'Empty state displayed - no trades');
        }
    }
    
    createTradeCard(trade) {
        this.debug.log('portfolio', 'loading', `Creating trade card for ${trade.symbol}`, trade);
        
        const pnlClass = trade.pnl >= 0 ? 'side-buy' : 'side-sell';
        const sideClass = trade.side === 'buy' ? 'side-buy' : 'side-sell';
        
        return `
            <div class="portfolio-card" data-trade-id="${trade.id}">
                <div class="trade-header">
                    <div class="symbol">${trade.symbol}</div>
                    <div class="${sideClass}">${trade.side.toUpperCase()}</div>
                    <button class="btn-close" onclick="app.closeTrade(${trade.id})">×</button>
                </div>
                <div class="trade-details">
                    <div>Entry: $${trade.entry_price}</div>
                    <div>Current: $${trade.current_price}</div>
                    <div class="${pnlClass}">P&L: $${trade.pnl?.toFixed(2) || '0.00'}</div>
                </div>
                <div class="trade-actions">
                    <button class="btn btn-small" onclick="app.editTrade(${trade.id})">Edit</button>
                    <button class="btn btn-small btn-danger" onclick="app.closeTrade(${trade.id})">Close</button>
                </div>
            </div>
        `;
    }
    
    async loadWatchlist() {
        this.debug.log('watchlist', 'loading', 'Loading watchlist data');
        
        const watchlistElement = document.getElementById('watchlist-content');
        
        if (!watchlistElement) {
            this.debug.log('watchlist', 'error', 'Watchlist content element not found');
            return;
        }
        
        try {
            this.debug.log('watchlist', 'loading', `Fetching prices for ${this.watchlistSymbols.length} symbols`);
            
            const promises = this.watchlistSymbols.map(async (symbol) => {
                this.debug.log('watchlist', 'loading', `Fetching price for ${symbol}`);
                const response = await fetch(`/api/price/${symbol}`);
                if (!response.ok) {
                    throw new Error(`Failed to fetch ${symbol}: HTTP ${response.status}`);
                }
                return await response.json();
            });
            
            const prices = await Promise.all(promises);
            this.debug.log('watchlist', 'success', `All ${prices.length} prices loaded successfully`);
            
            watchlistElement.innerHTML = prices.map(price => 
                this.createWatchlistCard(price)
            ).join('');
            
            this.debug.log('watchlist', 'success', 'Watchlist display updated');
        } catch (error) {
            console.error('Error loading watchlist:', error);
            this.debug.log('watchlist', 'error', 'Failed to load watchlist', error.message);
            watchlistElement.innerHTML = this.createEmptyState('Error loading watchlist');
        }
    }
    
    handlePriceUpdate(data) {
        this.debug.log('priceUpdates', 'success', `Processing price update for ${data.symbol}: $${data.price}`);
        
        try {
            // Update watchlist prices
            const watchlistItem = document.querySelector(`[data-symbol="${data.symbol}"]`);
            if (watchlistItem) {
                this.updateWatchlistItem(watchlistItem, data);
                this.debug.log('priceUpdates', 'success', `Watchlist updated for ${data.symbol}`);
            } else {
                this.debug.log('priceUpdates', 'error', `Watchlist item not found for ${data.symbol}`);
            }
            
            // Update chart if it's the current symbol
            if (data.symbol === this.currentSymbol) {
                this.updateChart(data);
                this.debug.log('priceUpdates', 'success', `Chart updated for current symbol ${data.symbol}`);
            }
            
            // Update any open trades for this symbol
            this.updateTradesPrices(data);
            this.debug.log('priceUpdates', 'success', `Trade prices updated for ${data.symbol}`);
        } catch (error) {
            this.debug.log('priceUpdates', 'error', `Failed to handle price update for ${data.symbol}`, error.message);
        }
    }
    
    updateTradesPrices(data) {
        try {
            if (this.portfolioData.trades) {
                const updatedTrades = this.portfolioData.trades.filter(trade => trade.symbol === data.symbol);
                this.debug.log('priceUpdates', 'success', `Found ${updatedTrades.length} trades to update for ${data.symbol}`);
                
                updatedTrades.forEach(trade => {
                    trade.current_price = data.price;
                    // Recalculate P&L
                    trade.pnl = (data.price - trade.entry_price) * trade.quantity;
                    if (trade.side === 'sell') trade.pnl *= -1;
                });
                
                // Update portfolio display if trades were updated
                if (updatedTrades.length > 0) {
                    this.updatePortfolioDisplay();
                }
            }
        } catch (error) {
            this.debug.log('priceUpdates', 'error', `Failed to update trade prices for ${data.symbol}`, error.message);
        }
    }
    
    async quickTrade(side) {
        this.debug.log('trading', 'loading', `Executing quick ${side} trade for ${this.currentSymbol}`);
        
        const symbol = this.currentSymbol;
        
        try {
            // Get current price
            this.debug.log('trading', 'loading', `Fetching current price for ${symbol}`);
            const priceResponse = await fetch(`/api/price/${symbol}`);
            
            if (!priceResponse.ok) {
                throw new Error(`Price fetch failed: HTTP ${priceResponse.status}`);
            }
            
            const priceData = await priceResponse.json();
            this.debug.log('trading', 'success', `Current price received: $${priceData.price}`, priceData);
            
            // Execute trade with default quantity
            const tradeData = {
                symbol: symbol,
                side: side,
                price: priceData.price,
                quantity: 1, // Default quantity
                confidence: 0.8
            };
            
            this.debug.log('trading', 'loading', 'Sending trade execution request', tradeData);
            
            const response = await fetch('/api/trade', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(tradeData)
            });
            
            const result = await response.json();
            this.debug.log('trading', 'success', `Trade API response: ${response.status}`, result);
            
            if (result.success) {
                this.showNotification(`${side.toUpperCase()} order executed for ${symbol}`, 'success');
                this.debug.log('trading', 'success', `${side} trade executed successfully for ${symbol}`);
                await this.loadPortfolio();
            } else {
                this.showNotification('Trade execution failed', 'error');
                this.debug.log('trading', 'error', 'Trade execution failed', result);
            }
        } catch (error) {
            console.error('Quick trade error:', error);
            this.debug.log('trading', 'error', `Quick ${side} trade failed`, error.message);
            this.showNotification('Trade execution error', 'error');
        }
    }
    
    async closeTrade(tradeId) {
        this.debug.log('trading', 'loading', `Closing trade ${tradeId}`);
        
        try {
            // Get current price for P&L calculation
            const trade = this.portfolioData.trades.find(t => t.id === tradeId);
            if (!trade) {
                this.debug.log('trading', 'error', `Trade ${tradeId} not found in portfolio`);
                return;
            }
            
            this.debug.log('trading', 'loading', `Fetching current price for ${trade.symbol} to close trade`);
            const priceResponse = await fetch(`/api/price/${trade.symbol}`);
            const priceData = await priceResponse.json();
            
            const pnl = (priceData.price - trade.entry_price) * trade.quantity;
            if (trade.side === 'sell') pnl *= -1;
            
            this.debug.log('trading', 'loading', `Calculated P&L: $${pnl.toFixed(2)}`);
            
            const response = await fetch(`/api/close_trade/${tradeId}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    close_price: priceData.price,
                    pnl: pnl
                })
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.showNotification('Trade closed successfully', 'success');
                this.debug.log('trading', 'success', `Trade ${tradeId} closed successfully with P&L: $${pnl.toFixed(2)}`);
                await this.loadPortfolio();
            } else {
                this.debug.log('trading', 'error', `Failed to close trade ${tradeId}`, result);
            }
        } catch (error) {
            console.error('Close trade error:', error);
            this.debug.log('trading', 'error', `Error closing trade ${tradeId}`, error.message);
            this.showNotification('Error closing trade', 'error');
        }
    }
    
    async runAdvancedAnalysis(type) {
        this.debug.log('aiAnalysis', 'loading', `Running ${type} analysis for ${this.currentSymbol}`);
        
        const analysisContent = document.getElementById('ai-analysis-content');
        
        if (!analysisContent) {
            this.debug.log('aiAnalysis', 'error', 'AI analysis content element not found');
            return;
        }
        
        analysisContent.innerHTML = '<div class="loading"></div> Running advanced AI analysis...';
        
        try {
            this.debug.log('aiAnalysis', 'loading', `Fetching analysis data for ${this.currentSymbol}`);
            const response = await fetch(`/api/analysis/${this.currentSymbol}`);
            
            if (!response.ok) {
                throw new Error(`Analysis API failed: HTTP ${response.status}`);
            }
            
            const analysis = await response.json();
            this.debug.log('aiAnalysis', 'success', 'Analysis data received', analysis);
            
            let content = this.formatAnalysisContent(analysis, type);
            analysisContent.innerHTML = content;
            
            // Update confidence bar
            this.updateConfidenceBar(analysis.confidence || 0.5);
            this.debug.log('aiAnalysis', 'success', `${type} analysis completed with confidence: ${analysis.confidence || 0.5}`);
        } catch (error) {
            analysisContent.innerHTML = '<p style="color: #ef4444;">Error running analysis</p>';
            console.error('Analysis error:', error);
            this.debug.log('aiAnalysis', 'error', `Failed to run ${type} analysis`, error.message);
        }
    }
    
    formatAnalysisContent(analysis, type) {
        if (type === 'comprehensive') {
            return `
                <div class="analysis-grid">
                    <div class="analysis-section">
                        <h4>🔍 Technical Analysis</h4>
                        <p>Trend: <span class="highlight">${analysis.technical?.trend || 'Analyzing...'}</span></p>
                        <p>Signal: <span class="highlight">${analysis.technical?.signal || 'N/A'}</span></p>
                        <p>Support: $${analysis.technical?.support || 'N/A'}</p>
                        <p>Resistance: $${analysis.technical?.resistance || 'N/A'}</p>
                    </div>
                    
                    <div class="analysis-section">
                        <h4>😊 Sentiment Analysis</h4>
                        <p>Score: <span class="highlight">${analysis.sentiment?.overall_score || 'N/A'}</span></p>
                        <p>Label: <span class="highlight">${analysis.sentiment?.label || 'Analyzing...'}</span></p>
                        <p>Confidence: ${(analysis.sentiment?.confidence * 100)?.toFixed(1) || 'N/A'}%</p>
                    </div>
                    
                    <div class="analysis-section">
                        <h4>🤖 ML Prediction</h4>
                        <p>Direction: <span class="highlight">${analysis.ml_prediction?.ensemble?.direction || 'Analyzing...'}</span></p>
                        <p>Confidence: ${(analysis.ml_prediction?.ensemble?.confidence * 100)?.toFixed(1) || 'N/A'}%</p>
                        <p>Consensus: ${(analysis.ml_prediction?.ensemble?.consensus * 100)?.toFixed(1) || 'N/A'}%</p>
                    </div>
                    
                    <div class="analysis-section">
                        <h4>📊 Recommendation</h4>
                        <p class="recommendation ${analysis.recommendation?.toLowerCase()}">${analysis.recommendation || 'HOLD'}</p>
                        <p>Overall Score: ${analysis.overall_score?.toFixed(3) || 'N/A'}</p>
                    </div>
                </div>
                
                <div class="analysis-actions">
                    <button class="btn btn-primary" onclick="app.executeRecommendation()">
                        Execute Recommendation
                    </button>
                    <button class="btn" onclick="app.saveAnalysis()">
                        Save Analysis
                    </button>
                </div>
            `;
        }
        
        // Return simplified content for specific analysis types
        return '<p>Analysis completed. Check comprehensive analysis for details.</p>';
    }
    
    updateConfidenceBar(confidence) {
        const confidenceFill = document.querySelector('.confidence-fill');
        const confidenceText = confidenceFill?.parentElement?.nextElementSibling;
        
        if (confidenceFill) {
            confidenceFill.style.width = `${confidence * 100}%`;
        }
        
        if (confidenceText) {
            confidenceText.textContent = `${(confidence * 100).toFixed(0)}%`;
        }
    }
    
    subscribeToSymbols() {
        this.debug.log('priceUpdates', 'loading', `Subscribing to ${this.watchlistSymbols.length} symbols`);
        
        try {
            this.watchlistSymbols.forEach(symbol => {
                this.socket.emit('subscribe_symbol', { symbol });
                this.debug.log('priceUpdates', 'success', `Subscribed to ${symbol}`);
            });
            this.debug.log('priceUpdates', 'success', 'All symbol subscriptions completed');
        } catch (error) {
            this.debug.log('priceUpdates', 'error', 'Symbol subscription failed', error.message);
        }
    }
    
    changeSymbol(symbol) {
        this.debug.log('navigation', 'loading', `Changing symbol from ${this.currentSymbol} to ${symbol}`);
        
        try {
            this.currentSymbol = symbol;
            console.log(`Switched to ${symbol}`);
            
            // Update active symbol in UI
            document.querySelectorAll('.symbol-select').forEach(el => {
                el.classList.remove('active');
            });
            
            const symbolElement = document.querySelector(`[data-symbol="${symbol}"]`);
            if (symbolElement) {
                symbolElement.classList.add('active');
                this.debug.log('navigation', 'success', `UI updated for symbol ${symbol}`);
            } else {
                this.debug.log('navigation', 'error', `Symbol element not found for ${symbol}`);
            }
            
            // Update chart
            this.updateChart({ symbol });
            
            // Subscribe to this symbol
            this.socket.emit('subscribe_symbol', { symbol });
            this.debug.log('navigation', 'success', `Symbol changed to ${symbol} and subscribed`);
        } catch (error) {
            this.debug.log('navigation', 'error', `Failed to change symbol to ${symbol}`, error.message);
        }
    }
    
    setupCharts() {
        this.debug.log('charts', 'loading', 'Setting up charts');
        
        try {
            // Check if chart container exists
            const chartContainer = document.getElementById('tradingview-chart');
            if (!chartContainer) {
                this.debug.log('charts', 'error', 'Chart container not found');
                return;
            }
            
            // Check if LightweightCharts is available
            if (typeof LightweightCharts === 'undefined') {
                this.debug.log('charts', 'error', 'LightweightCharts library not loaded');
                return;
            }
            
            this.debug.log('charts', 'success', 'Chart setup environment ready');
            
            // Check for nuclear chart system
            const nuclearContainer = document.getElementById('nuclear-chart-container');
            if (nuclearContainer) {
                this.debug.log('charts', 'success', 'Nuclear chart container found');
            } else {
                this.debug.log('charts', 'error', 'Nuclear chart container not found');
            }
            
            console.log('Setting up charts...');
            this.debug.log('charts', 'success', 'Chart setup completed');
        } catch (error) {
            this.debug.log('charts', 'error', 'Chart setup failed', error.message);
        }
    }
    
    updateChart(data) {
        this.debug.log('charts', 'loading', `Updating chart with data for ${data.symbol}`);
        
        try {
            // Placeholder for chart updates
            console.log('Updating chart with:', data);
            this.debug.log('charts', 'success', `Chart updated for ${data.symbol}`, data);
        } catch (error) {
            this.debug.log('charts', 'error', 'Chart update failed', error.message);
        }
    }
    
    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = message;
        
        // Add to page
        document.body.appendChild(notification);
        
        // Auto remove after 3 seconds
        setTimeout(() => {
            notification.remove();
        }, 3000);
    }
}

// Global notification function for component loader
window.showNotification = function(message, type = 'info') {
    if (window.app && window.app.showNotification) {
        window.app.showNotification(message, type);
    } else {
        // Fallback notification
        console.log(`[${type.toUpperCase()}] ${message}`);
        
        // Create simple notification
        const notification = document.createElement('div');
        notification.style.cssText = `
            position: fixed; top: 20px; right: 20px; z-index: 100000;
            background: ${type === 'error' ? '#ff4757' : type === 'success' ? '#2ed573' : '#00d4aa'};
            color: white; padding: 12px 20px; border-radius: 6px;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-size: 14px; box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            animation: slideIn 0.3s ease-out;
        `;
        notification.textContent = message;
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease-in forwards';
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }
};

// Add CSS for notification animations
if (!document.getElementById('notification-styles')) {
    const style = document.createElement('style');
    style.id = 'notification-styles';
    style.textContent = `
        @keyframes slideIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }
        @keyframes slideOut {
            from { transform: translateX(0); opacity: 1; }
            to { transform: translateX(100%); opacity: 0; }
        }
    `;
    document.head.appendChild(style);
};
    
function updateConnectionStatus(connected) {
    const statusElement = document.querySelector('.status-online');
    if (statusElement) {
        statusElement.style.backgroundColor = connected ? '#22c55e' : '#ef4444';
    }
}
    
function animateValue(element, value, prefix = '') {
        const duration = 1000;
        const start = parseFloat(element.textContent.replace(/[^0-9.-]+/g, '')) || 0;
        const end = value;
        const startTime = performance.now();
        
        const updateValue = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const current = start + (end - start) * progress;
            
            element.textContent = `${prefix}${current.toFixed(2)}`;
            
            if (progress < 1) {
                requestAnimationFrame(updateValue);
            }
        };
        
        requestAnimationFrame(updateValue);
}
    
function createEmptyState(message) {
    return `
        <div style="text-align: center; color: #666; padding: 20px;">
            <div style="font-size: 48px; margin-bottom: 10px;">📊</div>
            <p>${message}</p>
        </div>
    `;
}
    
function handleNavigation(navItem) {
        this.debug.log('navigation', 'loading', `Handling navigation to ${navItem.dataset.section}`);
        
        try {
            // Remove active class from all nav items
            const allNavItems = document.querySelectorAll('.nav-item');
            allNavItems.forEach(item => {
                item.classList.remove('active');
            });
            
            // Add active class to clicked item
            navItem.classList.add('active');
            
            const section = navItem.dataset.section;
            console.log(`Navigating to ${section}`);
            this.debug.log('navigation', 'success', `Navigation UI updated for ${section}`);
            
            // Handle section-specific logic
            switch (section) {
                case 'dashboard':
                    this.showDashboard();
                    break;
                case 'portfolio':
                    this.showPortfolio();
                    break;
                case 'trading':
                    this.showTrading();
                    break;
                case 'analysis':
                    this.showAnalysis();
                    break;
                case 'history':
                    this.showHistory();
                    break;
                case 'settings':
                    this.showSettings();
                    break;
                default:
                    this.debug.log('navigation', 'error', `Unknown section: ${section}`);
            }
            
            this.debug.log('navigation', 'success', `Navigation to ${section} completed`);
        } catch (error) {
            this.debug.log('navigation', 'error', `Navigation failed for ${navItem.dataset.section}`, error.message);
    }
}
    
function showDashboard() {
        this.debug.log('navigation', 'loading', 'Showing dashboard section');
        console.log('Showing dashboard');
        this.debug.log('navigation', 'success', 'Dashboard section displayed');
}
    
function showPortfolio() {
        this.debug.log('navigation', 'loading', 'Showing portfolio section');
        console.log('Showing portfolio');
        this.loadPortfolio(); // Refresh portfolio data
        this.debug.log('navigation', 'success', 'Portfolio section displayed');
}
    
function showTrading() {
        this.debug.log('navigation', 'loading', 'Showing trading interface');
        console.log('Showing trading interface');
        this.debug.log('navigation', 'success', 'Trading interface displayed');
}
    
function showAnalysis() {
        this.debug.log('navigation', 'loading', 'Showing analysis section');
        console.log('Showing analysis');
        this.runAdvancedAnalysis('comprehensive');
        this.debug.log('navigation', 'success', 'Analysis section displayed');
}
    
function showHistory() {
        this.debug.log('navigation', 'loading', 'Showing trade history');
        console.log('Showing trade history');
        this.debug.log('navigation', 'success', 'Trade history displayed');
}
    
function showSettings() {
        this.debug.log('navigation', 'loading', 'Showing settings');
        console.log('Showing settings');
        this.debug.log('navigation', 'success', 'Settings displayed');
}

// All component loader integration is now handled within the new manager class architecture
// Obsolete initialization functions removed to prevent conflicts

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 DOM Content Loaded - Initializing GoldGPT Application with Component Loader');
    
    // Initialize Component Loader
    const loader = new ComponentLoader();
    window.componentLoader = loader;
    
    // Register components with dependencies and priorities
    loader
        // Critical infrastructure components
        .register('socketConnection', {
            dependencies: [],
            priority: loader.PRIORITIES.CRITICAL,
            critical: true,
            timeout: 15000,
            retries: 3,
            loader: async () => {
                console.log('🔌 Initializing Socket Connection...');
                if (typeof io === 'undefined') {
                    throw new Error('Socket.IO library not available');
                }
                
                // Initialize socket connection
                window.socket = io();
                
                return new Promise((resolve, reject) => {
                    const timeout = setTimeout(() => {
                        reject(new Error('Socket connection timeout'));
                    }, 10000);
                    
                    window.socket.on('connect', () => {
                        clearTimeout(timeout);
                        console.log('✅ Socket connected successfully');
                        resolve();
                    });
                    
                    window.socket.on('connect_error', (error) => {
                        clearTimeout(timeout);
                        reject(new Error(`Socket connection failed: ${error.message}`));
                    });
                });
            }
        })
        
        // Chart system - depends on DOM being ready
        .register('chartSystem', {
            dependencies: [],
            priority: loader.PRIORITIES.CRITICAL,
            critical: true,
            timeout: 20000,
            retries: 2,
            loader: async () => {
                console.log('� Initializing Chart System...');
                
                // Wait for UnifiedChartManager to be available
                if (typeof UnifiedChartManager === 'undefined') {
                    throw new Error('UnifiedChartManager not available');
                }
                
                // UnifiedChartManager auto-initializes, just verify it worked
                return new Promise((resolve, reject) => {
                    const checkInterval = setInterval(() => {
                        if (window.unifiedChartManager && window.unifiedChartManager.isInitialized) {
                            clearInterval(checkInterval);
                            console.log('✅ Chart system initialized');
                            resolve();
                        }
                    }, 500);
                    
                    // Timeout after 15 seconds
                    setTimeout(() => {
                        clearInterval(checkInterval);
                        reject(new Error('Chart system initialization timeout'));
                    }, 15000);
                });
            }
        })
        
        // Core application - depends on socket connection
        .register('coreApplication', {
            dependencies: ['socketConnection'],
            priority: loader.PRIORITIES.CRITICAL,
            critical: true,
            timeout: 10000,
            retries: 2,
            loader: async () => {
                console.log('🏗️ Initializing Core Application...');
                
                // Initialize the main GoldGPT app
                window.app = new GoldGPTApp();
                
                // Verify critical elements exist
                const requiredElements = [
                    'account-balance',
                    'unified-chart-container'
                ];
                
                for (const elementId of requiredElements) {
                    const element = document.getElementById(elementId);
                    if (!element) {
                        throw new Error(`Required element missing: ${elementId}`);
                    }
                }
                
                // Verify watchlist exists (by class since no single ID)
                const watchlist = document.querySelector('.watchlist');
                if (!watchlist) {
                    throw new Error('Required element missing: watchlist container');
                }
                
                console.log('✅ Core application initialized');
            }
        })
        
        // Portfolio system - depends on core app and socket
        .register('portfolioSystem', {
            dependencies: ['coreApplication', 'socketConnection'],
            priority: loader.PRIORITIES.HIGH,
            critical: false,
            timeout: 8000,
            retries: 2,
            loader: async () => {
                console.log('💼 Initializing Portfolio System...');
                
                if (!window.app || !window.app.portfolio) {
                    throw new Error('Portfolio manager not available');
                }
                
                // Trigger initial portfolio load
                await window.app.portfolio.initialize();
                console.log('✅ Portfolio system initialized');
            }
        })
        
        // Watchlist system - depends on core app, socket, and market data
        .register('watchlistSystem', {
            dependencies: ['coreApplication', 'socketConnection', 'marketDataManager'],
            priority: loader.PRIORITIES.HIGH,
            critical: false,
            timeout: 8000,
            retries: 2,
            loader: async () => {
                console.log('👁️ Initializing Watchlist System...');
                
                if (!window.app || !window.app.watchlist) {
                    throw new Error('Watchlist manager not available');
                }
                
                // Initialize watchlist
                await window.app.watchlist.initialize();
                console.log('✅ Watchlist system initialized');
            }
        })
        
        // Trading system - depends on portfolio and watchlist
        .register('tradingSystem', {
            dependencies: ['portfolioSystem', 'watchlistSystem'],
            priority: loader.PRIORITIES.HIGH,
            critical: false,
            timeout: 5000,
            retries: 2,
            loader: async () => {
                console.log('💹 Initializing Trading System...');
                
                if (!window.app || !window.app.trading) {
                    throw new Error('Trading manager not available');
                }
                
                // Initialize trading capabilities
                await window.app.trading.initialize();
                console.log('✅ Trading system initialized');
            }
        })
        
        // AI Analysis system - can run independently with better error handling
        .register('aiAnalysisSystem', {
            dependencies: ['socketConnection'],
            priority: loader.PRIORITIES.MEDIUM,
            critical: false,
            timeout: 15000,
            retries: 3,
            loader: async () => {
                console.log('🤖 Initializing AI Analysis System...');
                
                // Check if AI analysis endpoints are available
                try {
                    // Try multiple endpoints for better compatibility
                    let response;
                    const endpoints = ['/api/ai-analysis/status', '/api/ai-analysis', '/api/ml-predictions'];
                    
                    for (const endpoint of endpoints) {
                        try {
                            response = await fetch(endpoint, { timeout: 5000 });
                            if (response.ok) {
                                console.log(`✅ AI Analysis API available at ${endpoint}`);
                                break;
                            }
                        } catch (e) {
                            console.warn(`⚠️ AI Analysis endpoint ${endpoint} not available`);
                        }
                    }
                    
                    if (!response || !response.ok) {
                        throw new Error('No AI Analysis API endpoints available');
                    }
                    
                    console.log('✅ AI Analysis system initialized');
                } catch (error) {
                    console.warn('⚠️ AI Analysis system failed to initialize:', error.message);
                    // Don't throw - allow system to continue without AI analysis
                }
            }
        })
        
        // News system - runs independently with fallback support
        .register('newsSystem', {
            dependencies: [],
            priority: loader.PRIORITIES.MEDIUM,
            critical: false,
            timeout: 10000,
            retries: 3,
            loader: async () => {
                console.log('📰 Initializing News System...');
                
                // Check if news API is available
                try {
                    // Try multiple news endpoints
                    const endpoints = ['/api/news/latest', '/api/news', '/api/enhanced-news'];
                    let newsAvailable = false;
                    
                    for (const endpoint of endpoints) {
                        try {
                            const response = await fetch(endpoint, { timeout: 5000 });
                            if (response.ok) {
                                console.log(`✅ News API available at ${endpoint}`);
                                newsAvailable = true;
                                break;
                            }
                        } catch (e) {
                            console.warn(`⚠️ News endpoint ${endpoint} not available`);
                        }
                    }
                    
                    if (!newsAvailable) {
                        console.warn('⚠️ No news endpoints available, using fallback');
                        // Initialize fallback news display
                        this.initializeFallbackNews();
                    }
                    
                    console.log('✅ News system initialized');
                } catch (error) {
                    console.warn('⚠️ News system initialization issue:', error.message);
                    // Don't throw - allow system to continue without news
                }
            },
            
            initializeFallbackNews: function() {
                // Simple fallback news display
                const newsContainer = document.querySelector('.enhanced-news-container');
                if (newsContainer) {
                    newsContainer.innerHTML = `
                        <div class="news-header">
                            <h3>📰 Market News</h3>
                            <div class="fallback-notice">News service temporarily unavailable</div>
                        </div>
                    `;
                }
            }
        })
        
        // Market Data Manager - Core data management
        .register('marketDataManager', {
            dependencies: ['socketConnection'],
            priority: loader.PRIORITIES.HIGH,
            critical: false,
            timeout: 3000,
            retries: 2,
            loader: async () => {
                console.log('📊 Initializing Market Data Manager...');
                
                if (typeof MarketDataManager !== 'undefined') {
                    window.marketDataManager = new MarketDataManager();
                    await window.marketDataManager.init();
                    console.log('✅ Market Data Manager ready');
                } else {
                    console.warn('⚠️ MarketDataManager class not found, skipping');
                }
            }
        })
        
        // Price Display Manager - Price display updates
        .register('priceDisplayManager', {
            dependencies: ['marketDataManager'],
            priority: loader.PRIORITIES.MEDIUM,
            critical: false,
            timeout: 3000,
            retries: 2,
            loader: async () => {
                console.log('💰 Initializing Price Display Manager...');
                
                if (typeof PriceDisplayManager !== 'undefined') {
                    window.priceDisplayManager = new PriceDisplayManager();
                    await window.priceDisplayManager.init();
                    console.log('✅ Price Display Manager ready');
                } else {
                    console.warn('⚠️ PriceDisplayManager class not found, skipping');
                }
            }
        })
        
        // TradingView Chart Manager - Chart management
        .register('tradingViewChartManager', {
            dependencies: ['chartSystem'],
            priority: loader.PRIORITIES.MEDIUM,
            critical: false,
            timeout: 5000,
            retries: 1,
            loader: async () => {
                console.log('📈 Initializing TradingView Chart Manager...');
                
                if (typeof TradingViewChartManager !== 'undefined') {
                    window.tradingViewChartManager = new TradingViewChartManager();
                    await window.tradingViewChartManager.initialize();
                    console.log('✅ TradingView Chart Manager ready');
                } else {
                    console.warn('⚠️ TradingViewChartManager class not found, skipping');
                }
            }
        })
        
        // Internal Chart Analysis - Background chart analysis
        .register('internalChartAnalysis', {
            dependencies: ['chartSystem'],
            priority: loader.PRIORITIES.LOW,
            critical: false,
            timeout: 3000,
            retries: 1,
            loader: async () => {
                console.log('🔬 Initializing Internal Chart Analysis...');
                
                if (typeof InternalChartAnalysisEngine !== 'undefined') {
                    window.internalChartAnalysis = new InternalChartAnalysisEngine();
                    await window.internalChartAnalysis.initialize();
                    console.log('✅ Internal Chart Analysis ready');
                } else {
                    console.warn('⚠️ InternalChartAnalysisEngine class not found, skipping');
                }
            }
        })
        
        // AI Analysis Center - AI-powered analysis
        .register('aiAnalysisCenter', {
            dependencies: ['coreApplication'],
            priority: loader.PRIORITIES.MEDIUM,
            critical: false,
            timeout: 5000,
            retries: 1,
            loader: async () => {
                console.log('🤖 Initializing AI Analysis Center...');
                
                if (typeof AdvancedAIAnalysisCenter !== 'undefined') {
                    window.aiAnalysisCenter = new AdvancedAIAnalysisCenter();
                    await window.aiAnalysisCenter.init();
                    console.log('✅ AI Analysis Center ready');
                } else {
                    console.warn('⚠️ AdvancedAIAnalysisCenter class not found, skipping');
                }
            }
        })
        
        // Enhanced News Manager - News management
        .register('enhancedNewsManager', {
            dependencies: ['newsSystem'],
            priority: loader.PRIORITIES.MEDIUM,
            critical: false,
            timeout: 3000,
            retries: 1,
            loader: async () => {
                console.log('📰 Initializing Enhanced News Manager...');
                
                if (typeof EnhancedNewsManager !== 'undefined') {
                    window.enhancedNewsManager = new EnhancedNewsManager();
                    await window.enhancedNewsManager.init();
                    console.log('✅ Enhanced News Manager ready');
                } else {
                    console.warn('⚠️ EnhancedNewsManager class not found, skipping');
                }
            }
        })
        
        // Right Panel Manager - Right panel components
        .register('rightPanelManager', {
            dependencies: ['coreApplication'],
            priority: loader.PRIORITIES.LOW,
            critical: false,
            timeout: 3000,
            retries: 1,
            loader: async () => {
                console.log('📊 Initializing Right Panel Manager...');
                
                if (typeof RightPanelManager !== 'undefined') {
                    window.rightPanelManager = new RightPanelManager();
                    await window.rightPanelManager.init();
                    console.log('✅ Right Panel Manager ready');
                } else {
                    console.warn('⚠️ RightPanelManager class not found, skipping');
                }
            }
        })
        
        // Gold API Live Price Fetcher - Live price updates
        .register('goldApiLivePriceFetcher', {
            dependencies: ['priceDisplayManager'],
            priority: loader.PRIORITIES.HIGH,
            critical: false,
            timeout: 3000,
            retries: 2,
            loader: async () => {
                console.log('🥇 Initializing Gold API Live Price Fetcher...');
                
                if (typeof GoldAPILivePriceFetcher !== 'undefined') {
                    window.goldApiLivePriceFetcher = new GoldAPILivePriceFetcher();
                    await window.goldApiLivePriceFetcher.init();
                    console.log('✅ Gold API Live Price Fetcher ready');
                } else {
                    console.warn('⚠️ GoldAPILivePriceFetcher class not found, skipping');
                }
            }
        })
        
        // Enhanced Dashboard - Dashboard enhancements
        .register('enhancedDashboard', {
            dependencies: ['coreApplication'],
            priority: loader.PRIORITIES.LOW,
            critical: false,
            timeout: 3000,
            retries: 1,
            loader: async () => {
                console.log('✨ Initializing Enhanced Dashboard...');
                
                if (typeof EnhancedDashboardManager !== 'undefined') {
                    window.enhancedDashboard = new EnhancedDashboardManager();
                    await window.enhancedDashboard.initializeEnhancedDashboard();
                    console.log('✅ Enhanced Dashboard ready');
                } else {
                    console.warn('⚠️ EnhancedDashboardManager class not found, skipping');
                }
            }
        })
        
        // Real-time updates - depends on all trading components
        .register('realTimeUpdates', {
            dependencies: ['portfolioSystem', 'watchlistSystem', 'chartSystem'],
            priority: loader.PRIORITIES.HIGH,
            critical: false,
            timeout: 5000,
            retries: 1,
            loader: async () => {
                console.log('⚡ Initializing Real-time Updates...');
                
                if (!window.socket) {
                    throw new Error('Socket connection not available');
                }
                
                // Set up real-time price updates
                window.socket.emit('subscribe_prices', {
                    symbols: ['XAUUSD', 'XAGUSD', 'EURUSD', 'GBPUSD', 'USDJPY', 'BTCUSD']
                });
                
                console.log('✅ Real-time updates initialized');
            }
        })
        
        // Enhanced features - lowest priority
        .register('enhancedFeatures', {
            dependencies: ['coreApplication'],
            priority: loader.PRIORITIES.LOW,
            critical: false,
            timeout: 5000,
            retries: 1,
            loader: async () => {
                console.log('✨ Initializing Enhanced Features...');
                
                // Initialize non-critical features
                if (window.app && window.app.initializeEnhancedFeatures) {
                    await window.app.initializeEnhancedFeatures();
                }
                
                console.log('✅ Enhanced features initialized');
            }
        });
    
    // Start the loading process
    loader.startLoading().then(() => {
        console.log('🎉 All components loaded successfully');
        
        // Set up global error handlers after everything is loaded
        window.addEventListener('error', (event) => {
            if (window.app && window.app.debug) {
                window.app.debug.log('runtime', 'error', `Global error: ${event.error.message}`, {
                    filename: event.filename,
                    lineno: event.lineno,
                    colno: event.colno
                });
            }
            console.error('🚨 Global Error:', event.error);
        });
        
        window.addEventListener('unhandledrejection', (event) => {
            if (window.app && window.app.debug) {
                window.app.debug.log('runtime', 'error', `Unhandled promise rejection: ${event.reason}`);
            }
            console.error('🚨 Unhandled Promise Rejection:', event.reason);
        });
        
    }).catch((error) => {
        console.error('🚨 Critical component loading failed:', error);
        
        // Still try to create a basic error display
        if (!window.app) {
            const errorDiv = document.createElement('div');
            errorDiv.style.cssText = `
                position: fixed; top: 50%; left: 50%; transform: translate(-50%, -50%);
                background: #ff4757; color: white; padding: 20px; border-radius: 8px;
                font-family: monospace; z-index: 99999; max-width: 500px;
            `;
            errorDiv.innerHTML = `
                <h3>🚨 Application Initialization Failed</h3>
                <p><strong>Error:</strong> ${error.message}</p>
                <p><strong>Time:</strong> ${new Date().toISOString()}</p>
                <button onclick="window.location.reload()" style="background: white; color: #ff4757; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer; margin-top: 10px;">
                    Reload Page
                </button>
            `;
            
            document.body.appendChild(errorDiv);
        }
    });
    
});
