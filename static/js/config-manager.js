/**
 * GoldGPT Configuration Manager
 * Centralized configuration system following Trading 212's modular approach
 */

class ConfigManager {
    constructor() {
        this.config = {
            // Default configuration enhanced for gold trading
            symbols: {
                primary: 'XAUUSD',
                watchlist: ['XAUUSD', 'XAGUSD', 'EURUSD', 'GBPUSD', 'USDJPY', 'BTCUSD', 'DXY'],
                mappings: {
                    'XAUUSD': { name: 'Gold', icon: '🥇', category: 'commodity', priority: 1 },
                    'XAGUSD': { name: 'Silver', icon: '🥈', category: 'commodity', priority: 2 },
                    'DXY': { name: 'Dollar Index', icon: '💵', category: 'index', priority: 3 },
                    'EURUSD': { name: 'Euro/USD', icon: '💶', category: 'forex', priority: 4 },
                    'GBPUSD': { name: 'GBP/USD', icon: '💷', category: 'forex', priority: 5 },
                    'USDJPY': { name: 'USD/JPY', icon: '💴', category: 'forex', priority: 6 },
                    'BTCUSD': { name: 'Bitcoin', icon: '₿', category: 'crypto', priority: 7 }
                },
                correlations: {
                    'XAUUSD': ['DXY', 'EURUSD', 'GBPUSD'],
                    'XAGUSD': ['XAUUSD', 'DXY']
                }
            },
            dataSources: {
                primary: 'gold-api',
                fallback: 'local-simulation',
                endpoints: {
                    'gold-api': 'https://api.gold-api.com',
                    'local-api': '/api',
                    'websocket': 'ws://localhost:5000'
                },
                updateIntervals: {
                    price: 2000,        // 2 seconds for gold trading
                    news: 60000,        // 1 minute
                    analysis: 300000,   // 5 minutes
                    macro: 3600000      // 1 hour
                },
                timeout: 10000,
                retryAttempts: 3
            },
            charts: {
                defaultTimeframe: '1h',
                availableTimeframes: ['1m', '5m', '15m', '1h', '4h', '1d', '1w'],
                indicators: {
                    'ma': { name: 'Moving Average', enabled: true, params: { period: 20 } },
                    'ema': { name: 'EMA', enabled: true, params: { period: 50 } },
                    'rsi': { name: 'RSI', enabled: true, params: { period: 14 } },
                    'macd': { name: 'MACD', enabled: true, params: { fast: 12, slow: 26, signal: 9 } },
                    'bb': { name: 'Bollinger Bands', enabled: true, params: { period: 20, stdDev: 2 } },
                    'stoch': { name: 'Stochastic', enabled: false, params: { k: 14, d: 3 } },
                    'atr': { name: 'ATR', enabled: false, params: { period: 14 } }
                },
                tradingViewConfig: {
                    theme: 'dark',
                    toolbar_bg: '#1a1a1a',
                    studies_overrides: {},
                    overrides: {
                        'mainSeriesProperties.candleStyle.upColor': '#00ff88',
                        'mainSeriesProperties.candleStyle.downColor': '#ff4444',
                        'mainSeriesProperties.candleStyle.borderUpColor': '#00ff88',
                        'mainSeriesProperties.candleStyle.borderDownColor': '#ff4444'
                    }
                }
            },
            // Enhanced gold trading configuration
            goldTrading: {
                enabled: true,
                symbols: {
                    primary: 'XAUUSD',
                    secondary: ['XAGUSD', 'DXY', 'EURUSD', 'GBPUSD'],
                    displayNames: {
                        'XAUUSD': 'Gold/USD',
                        'XAGUSD': 'Silver/USD',
                        'DXY': 'Dollar Index',
                        'EURUSD': 'Euro/USD',
                        'GBPUSD': 'Pound/USD'
                    }
                },
                trading: {
                    defaultLotSize: 0.01,
                    maxLotSize: 1.0,
                    minLotSize: 0.01,
                    stopLossPercent: 1.0,
                    takeProfitPercent: 2.0,
                    riskPerTrade: 2.0,
                    maxDailyRisk: 6.0,
                    maxOpenPositions: 3,
                    tradingHours: {
                        start: '00:00',
                        end: '23:59',
                        timezone: 'UTC'
                    },
                    riskManagement: {
                        maxDrawdown: 10.0,
                        stopTradingDrawdown: 15.0,
                        dailyLossLimit: 5.0,
                        consecutiveLossLimit: 5,
                        requireConfirmation: true
                    }
                },
                analysis: {
                    technicalTimeframes: ['1m', '5m', '15m', '1h', '4h', '1d'],
                    defaultTimeframe: '1h',
                    indicators: {
                        primary: ['RSI', 'MACD', 'SMA_20', 'EMA_50'],
                        secondary: ['BB', 'Stoch', 'ADX', 'ATR']
                    },
                    sentimentSources: ['news', 'social', 'cot'],
                    mlModelRefreshInterval: 300000 // 5 minutes
                }
            },
            notifications: {
                enabled: true,
                position: 'top-right',
                duration: 6000,
                maxNotifications: 5,
                sound: {
                    enabled: true,
                    volume: 0.7,
                    types: {
                        price: 'trading212-price.mp3',
                        trade: 'trading212-trade.mp3',
                        alert: 'trading212-alert.mp3',
                        success: 'trading212-success.mp3',
                        error: 'trading212-error.mp3'
                    }
                },
                types: {
                    price: { enabled: true, priority: 'high', threshold: 0.5 },
                    trades: { enabled: true, priority: 'high', sound: true },
                    system: { enabled: true, priority: 'medium', sound: false },
                    ai: { enabled: true, priority: 'medium', sound: false },
                    news: { enabled: true, priority: 'low', sound: false },
                    analysis: { enabled: true, priority: 'medium', sound: false }
                },
                thresholds: {
                    goldPriceChange: 0.5,      // 0.5% change
                    volumeSpike: 2.0,          // 2x average volume
                    sentimentShift: 0.3,       // 30% sentiment change
                    newsImpact: 'medium'       // medium or higher impact
                }
            },
            websocket: {
                enabled: true,
                reconnectInterval: 2000,
                maxReconnectAttempts: 10,
                heartbeatInterval: 25000,
                fallbackToPolling: true,
                pollingInterval: 3000,
                compression: true,
                bufferSize: 1024,
                endpoints: {
                    price: '/ws/price',
                    trades: '/ws/trades',
                    news: '/ws/news',
                    analysis: '/ws/analysis'
                }
            },
            ui: {
                theme: 'trading212-dark',
                animations: true,
                compactMode: false,
                refreshInterval: 2000,
                loadingTimeout: 8000,
                tradingView: {
                    theme: 'dark',
                    style: 'trading212',
                    autosize: true,
                    height: 500,
                    toolbar_bg: '#1e1e1e',
                    allow_symbol_change: false,
                    details: false,
                    hotlist: false,
                    calendar: false
                },
                dashboard: {
                    layout: 'advanced',
                    panels: {
                        pricePanel: { enabled: true, size: 'large' },
                        chartPanel: { enabled: true, size: 'xlarge' },
                        aiPanel: { enabled: true, size: 'medium' },
                        newsPanel: { enabled: true, size: 'small' },
                        portfolioPanel: { enabled: true, size: 'medium' }
                    }
                }
            },
            trading: {
                enabled: false,
                demoMode: true,
                confirmation: {
                    required: true,
                    doubleClick: false,
                    timeDelay: 2000
                },
                riskManagement: {
                    maxPositionSize: 1000,
                    stopLossRequired: true,
                    takeProfitRequired: false,
                    riskPerTrade: 2.0,
                    maxDailyLoss: 5.0,
                    positionSizing: 'fixed'
                },
                execution: {
                    slippage: 0.1,
                    timeout: 30000,
                    retries: 3
                }
            },
            // Additional gold trading specific configurations
            goldSpecific: {
                marketHours: {
                    london: { open: '08:00', close: '17:00', timezone: 'GMT' },
                    newyork: { open: '13:00', close: '22:00', timezone: 'GMT' },
                    asia: { open: '00:00', close: '09:00', timezone: 'GMT' }
                },
                news: {
                    sources: ['fed', 'ecb', 'boe', 'treasury', 'inflation', 'employment'],
                    keywords: ['gold', 'inflation', 'fed', 'interest rates', 'dollar', 'recession'],
                    refreshInterval: 300000,
                    importance: 'medium'
                },
                correlations: {
                    enabled: true,
                    instruments: ['DXY', 'EURUSD', 'GBPUSD', 'TNX', 'SPX500'],
                    updateInterval: 60000
                }
            }
        };
        
        this.isLoaded = false;
        this.loadPromise = null;
        this.eventListeners = new Set();
        this.updateQueue = [];
        
        console.log('⚙️ ConfigManager initialized');
    }

    /**
     * Initialize configuration system with gold trading optimizations
     */
    async initialize() {
        if (this.loadPromise) {
            return this.loadPromise;
        }

        this.loadPromise = this._loadConfiguration();
        
        // Apply gold trading optimizations
        this._optimizeForGoldTrading();
        
        // Setup real-time config updates
        this._setupConfigWatcher();
        
        return this.loadPromise;
    }

    /**
     * Apply gold trading specific optimizations
     * @private
     */
    _optimizeForGoldTrading() {
        try {
            // Optimize update intervals for gold market volatility
            this.config.dataSources.updateIntervals.price = 1500; // 1.5 seconds for high volatility
            
            // Enable key indicators for gold analysis
            this.config.charts.indicators.rsi.enabled = true;
            this.config.charts.indicators.macd.enabled = true;
            this.config.charts.indicators.bb.enabled = true;
            
            // Set conservative risk parameters for gold trading
            this.config.goldTrading.trading.riskPerTrade = Math.min(
                this.config.goldTrading.trading.riskPerTrade, 
                2.0
            );
            
            // Prioritize gold symbol
            this.config.symbols.watchlist = this.config.symbols.watchlist.sort((a, b) => {
                const priorities = this.config.symbols.mappings;
                return (priorities[a]?.priority || 999) - (priorities[b]?.priority || 999);
            });
            
            console.log('🥇 Gold trading optimizations applied');
            
        } catch (error) {
            console.warn('⚠️ Gold trading optimization failed:', error);
        }
    }

    /**
     * Setup configuration file watcher for real-time updates
     * @private
     */
    _setupConfigWatcher() {
        // Listen for external config changes
        if (window.addEventListener) {
            window.addEventListener('storage', (e) => {
                if (e.key === 'goldgpt_config') {
                    this._reloadConfiguration();
                }
            });
        }
    }

    /**
     * Get optimized configuration for specific trading session
     * @param {string} session - Trading session ('london', 'newyork', 'asia')
     * @returns {Object} Session-optimized configuration
     */
    getSessionConfig(session = 'london') {
        const sessionConfig = { ...this.config };
        
        // Adjust update intervals based on session volatility
        const sessionMultipliers = {
            london: 1.0,    // Normal speed
            newyork: 0.8,   // Faster updates
            asia: 1.5       // Slower updates
        };
        
        const multiplier = sessionMultipliers[session] || 1.0;
        sessionConfig.dataSources.updateIntervals.price *= multiplier;
        sessionConfig.websocket.heartbeatInterval *= multiplier;
        
        return sessionConfig;
    }

    /**
     * Get risk management configuration for current market conditions
     * @param {Object} marketConditions - Current market volatility, trend, etc.
     * @returns {Object} Risk-adjusted configuration
     */
    getRiskAdjustedConfig(marketConditions = {}) {
        const config = { ...this.config };
        const { volatility = 'medium', trend = 'neutral', volume = 'normal' } = marketConditions;
        
        // Adjust risk parameters based on market conditions
        if (volatility === 'high') {
            config.goldTrading.trading.riskPerTrade *= 0.7;
            config.goldTrading.trading.stopLossPercent *= 0.8;
            config.dataSources.updateIntervals.price = 1000; // 1 second
        } else if (volatility === 'low') {
            config.goldTrading.trading.riskPerTrade *= 1.2;
            config.dataSources.updateIntervals.price = 3000; // 3 seconds
        }
        
        // Adjust based on trend strength
        if (trend === 'strong') {
            config.goldTrading.trading.takeProfitPercent *= 1.5;
        }
        
        return config;
    }

    /**
     * Load configuration from server with fallbacks
     */
    async _loadConfiguration() {
        try {
            console.log('📡 Loading configuration from server...');
            
            // Try to load from server
            const serverConfig = await this._fetchServerConfig();
            
            if (serverConfig) {
                this.config = this._mergeConfigs(this.config, serverConfig);
                console.log('✅ Server configuration loaded');
            }
            
            // Load user preferences from localStorage
            const userPrefs = this._loadUserPreferences();
            if (userPrefs) {
                this.config = this._mergeConfigs(this.config, userPrefs);
                console.log('✅ User preferences loaded');
            }
            
            this.isLoaded = true;
            this._notifyListeners('config-loaded', this.config);
            
            return this.config;
            
        } catch (error) {
            console.warn('⚠️ Failed to load server configuration, using defaults:', error);
            
            // Load cached configuration if available
            const cachedConfig = this._loadCachedConfig();
            if (cachedConfig) {
                this.config = this._mergeConfigs(this.config, cachedConfig);
                console.log('✅ Cached configuration loaded');
            }
            
            this.isLoaded = true;
            this._notifyListeners('config-loaded', this.config);
            
            return this.config;
        }
    }

    /**
     * Fetch configuration from server
     */
    async _fetchServerConfig() {
        try {
            const response = await fetch('/api/config', {
                method: 'GET',
                headers: { 'Content-Type': 'application/json' },
                timeout: 5000
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            const serverConfig = await response.json();
            
            // Cache the server configuration
            this._cacheConfig(serverConfig);
            
            return serverConfig;
            
        } catch (error) {
            console.error('❌ Failed to fetch server config:', error);
            return null;
        }
    }

    /**
     * Deep merge configuration objects
     */
    _mergeConfigs(target, source) {
        const result = { ...target };
        
        for (const key in source) {
            if (source.hasOwnProperty(key)) {
                if (typeof source[key] === 'object' && source[key] !== null && !Array.isArray(source[key])) {
                    result[key] = this._mergeConfigs(result[key] || {}, source[key]);
                } else {
                    result[key] = source[key];
                }
            }
        }
        
        return result;
    }

    /**
     * Load user preferences from localStorage
     */
    _loadUserPreferences() {
        try {
            const prefs = localStorage.getItem('goldgpt-user-preferences');
            return prefs ? JSON.parse(prefs) : null;
        } catch (error) {
            console.warn('⚠️ Failed to load user preferences:', error);
            return null;
        }
    }

    /**
     * Save user preferences to localStorage
     */
    _saveUserPreferences(prefs) {
        try {
            localStorage.setItem('goldgpt-user-preferences', JSON.stringify(prefs));
        } catch (error) {
            console.warn('⚠️ Failed to save user preferences:', error);
        }
    }

    /**
     * Load cached configuration
     */
    _loadCachedConfig() {
        try {
            const cached = localStorage.getItem('goldgpt-cached-config');
            return cached ? JSON.parse(cached) : null;
        } catch (error) {
            console.warn('⚠️ Failed to load cached config:', error);
            return null;
        }
    }

    /**
     * Cache configuration to localStorage
     */
    _cacheConfig(config) {
        try {
            localStorage.setItem('goldgpt-cached-config', JSON.stringify(config));
        } catch (error) {
            console.warn('⚠️ Failed to cache config:', error);
        }
    }

    /**
     * Get configuration value by path
     */
    get(path, defaultValue = null) {
        if (!this.isLoaded) {
            console.warn('⚠️ Configuration not loaded yet');
            return defaultValue;
        }

        const keys = path.split('.');
        let current = this.config;
        
        for (const key of keys) {
            if (current && typeof current === 'object' && key in current) {
                current = current[key];
            } else {
                return defaultValue;
            }
        }
        
        return current;
    }

    /**
     * Set configuration value by path
     */
    set(path, value, saveToUserPrefs = true) {
        const keys = path.split('.');
        let current = this.config;
        
        // Navigate to the parent object
        for (let i = 0; i < keys.length - 1; i++) {
            const key = keys[i];
            if (!(key in current) || typeof current[key] !== 'object') {
                current[key] = {};
            }
            current = current[key];
        }
        
        // Set the value
        const lastKey = keys[keys.length - 1];
        const oldValue = current[lastKey];
        current[lastKey] = value;
        
        // Save to user preferences if requested
        if (saveToUserPrefs) {
            this._saveUserPreferences(this.config);
        }
        
        // Notify listeners
        this._notifyListeners('config-changed', {
            path,
            value,
            oldValue
        });
        
        console.log(`⚙️ Config updated: ${path} = ${JSON.stringify(value)}`);
    }

    /**
     * Get all symbols configuration
     */
    getSymbols() {
        return this.get('symbols', {});
    }

    /**
     * Get symbol configuration by symbol
     */
    getSymbol(symbol) {
        return this.get(`symbols.mappings.${symbol}`, null);
    }

    /**
     * Get data source configuration
     */
    getDataSource(name = null) {
        if (name) {
            return this.get(`dataSources.endpoints.${name}`, null);
        }
        return this.get('dataSources', {});
    }

    /**
     * Get chart configuration
     */
    getChartConfig() {
        return this.get('charts', {});
    }

    /**
     * Get notification settings
     */
    getNotificationSettings() {
        return this.get('notifications', {});
    }

    /**
     * Get WebSocket configuration
     */
    getWebSocketConfig() {
        return this.get('websocket', {});
    }

    /**
     * Get UI configuration
     */
    getUIConfig() {
        return this.get('ui', {});
    }

    /**
     * Update configuration from server
     */
    async refresh() {
        try {
            console.log('🔄 Refreshing configuration...');
            
            const serverConfig = await this._fetchServerConfig();
            
            if (serverConfig) {
                const oldConfig = { ...this.config };
                this.config = this._mergeConfigs(this.config, serverConfig);
                
                this._notifyListeners('config-refreshed', {
                    oldConfig,
                    newConfig: this.config
                });
                
                console.log('✅ Configuration refreshed');
            }
            
        } catch (error) {
            console.error('❌ Failed to refresh configuration:', error);
            throw error;
        }
    }

    /**
     * Add event listener
     */
    on(event, callback) {
        const listener = { event, callback };
        this.eventListeners.add(listener);
        
        return () => {
            this.eventListeners.delete(listener);
        };
    }

    /**
     * Notify event listeners
     */
    _notifyListeners(event, data) {
        for (const listener of this.eventListeners) {
            if (listener.event === event) {
                try {
                    listener.callback(data);
                } catch (error) {
                    console.error('❌ Error in config listener:', error);
                }
            }
        }
    }

    /**
     * Wait for configuration to be loaded
     */
    async waitForConfig() {
        if (this.isLoaded) {
            return this.config;
        }
        
        if (this.loadPromise) {
            return this.loadPromise;
        }
        
        return this.initialize();
    }

    /**
     * Validate configuration
     */
    validate() {
        const errors = [];
        
        // Validate symbols
        if (!this.config.symbols || !this.config.symbols.watchlist) {
            errors.push('Missing symbols configuration');
        }
        
        // Validate data sources
        if (!this.config.dataSources || !this.config.dataSources.endpoints) {
            errors.push('Missing data sources configuration');
        }
        
        // Validate WebSocket config
        if (!this.config.websocket) {
            errors.push('Missing WebSocket configuration');
        }
        
        return {
            isValid: errors.length === 0,
            errors
        };
    }

    /**
     * Get configuration summary for debugging
     */
    getDebugInfo() {
        return {
            isLoaded: this.isLoaded,
            symbolCount: Object.keys(this.config.symbols?.mappings || {}).length,
            dataSourceCount: Object.keys(this.config.dataSources?.endpoints || {}).length,
            listenersCount: this.eventListeners.size,
            validation: this.validate()
        };
    }

    /**
     * Export configuration for backup
     */
    export() {
        return {
            config: this.config,
            timestamp: new Date().toISOString(),
            version: '1.0.0'
        };
    }

    /**
     * Import configuration from backup
     */
    import(backupData) {
        if (!backupData || !backupData.config) {
            throw new Error('Invalid backup data');
        }
        
        this.config = this._mergeConfigs(this.config, backupData.config);
        this._saveUserPreferences(this.config);
        
        this._notifyListeners('config-imported', backupData);
        
        console.log('📥 Configuration imported successfully');
    }

    /**
     * Reset configuration to defaults
     */
    reset() {
        // Clear cached data
        localStorage.removeItem('goldgpt-user-preferences');
        localStorage.removeItem('goldgpt-cached-config');
        
        // Reload from server
        this.isLoaded = false;
        this.loadPromise = null;
        
        this.initialize();
        
        console.log('🔄 Configuration reset to defaults');
    }

    /**
     * Update configuration values
     * @param {Object} updates - Configuration updates to apply
     * @param {boolean} save - Whether to save to localStorage
     */
    updateConfig(updates, save = true) {
        try {
            // Deep merge the updates with current config
            this.config = this._mergeConfigs(this.config, updates);
            
            // Save to localStorage if requested
            if (save) {
                this._saveUserPreferences(updates);
            }
            
            // Notify listeners of config changes
            this._notifyListeners('config-updated', { updates, config: this.config });
            
            console.log('✅ Configuration updated:', updates);
            
            return this.config;
            
        } catch (error) {
            console.error('❌ Failed to update configuration:', error);
            throw error;
        }
    }

    /**
     * Get current configuration
     * @param {string} path - Optional path to specific config value (e.g., 'dataSources.updateIntervals.price')
     * @returns {*} Configuration value or entire config
     */
    getConfig(path = null) {
        if (!path) {
            return this.config;
        }
        
        // Navigate to nested config value
        return path.split('.').reduce((obj, key) => obj && obj[key], this.config);
    }

    /**
     * Set a specific configuration value
     * @param {string} path - Path to the config value (e.g., 'dataSources.updateIntervals.price')
     * @param {*} value - New value to set
     * @param {boolean} save - Whether to save to localStorage
     */
    setConfig(path, value, save = true) {
        try {
            const pathArray = path.split('.');
            const lastKey = pathArray.pop();
            
            // Navigate to parent object
            const parent = pathArray.reduce((obj, key) => {
                if (!obj[key]) obj[key] = {};
                return obj[key];
            }, this.config);
            
            parent[lastKey] = value;
            
            // Save if requested
            if (save) {
                this._saveUserPreferences({ [path]: value });
            }
            
            // Notify listeners
            this._notifyListeners('config-updated', { 
                updates: { [path]: value }, 
                config: this.config 
            });
            
            console.log(`✅ Configuration set: ${path} = ${value}`);
            
        } catch (error) {
            console.error(`❌ Failed to set configuration ${path}:`, error);
            throw error;
        }
    }

    /**
     * Cleanup resources
     */
    cleanup() {
        this.eventListeners.clear();
        this.updateQueue = [];
        
        console.log('🧹 ConfigManager cleaned up');
    }
}

// Create global instance
window.configManager = new ConfigManager();

// Auto-initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.configManager.initialize().catch(error => {
        console.error('❌ Failed to initialize configuration:', error);
    });
});

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ConfigManager;
}

console.log('⚙️ Configuration Manager loaded successfully');
