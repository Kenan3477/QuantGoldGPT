/**
 * GoldGPT Startup Manager
 * Consolidates all initialization logic to prevent race conditions
 * Ensures proper loading order for all components
 */

class StartupManager {
    constructor() {
        this.initialized = false;
        this.startTime = Date.now();
        this.initializationQueue = [];
        this.completedSteps = new Set();
        this.failedSteps = new Set();
        
        // Initialization steps in order
        this.initSteps = [
            { name: 'dom', priority: 1, handler: () => this.waitForDOM() },
            { name: 'config', priority: 2, handler: () => this.initializeConfig() },
            { name: 'connection', priority: 3, handler: () => this.initializeConnection() },
            { name: 'components', priority: 4, handler: () => this.initializeComponents() },
            { name: 'dataFeeds', priority: 5, handler: () => this.initializeDataFeeds() },
            { name: 'charts', priority: 6, handler: () => this.initializeCharts() },
            { name: 'finalSetup', priority: 7, handler: () => this.finalizeSetup() }
        ];
        
        console.log('🚀 StartupManager initialized');
    }
    
    async initialize() {
        if (this.initialized) {
            console.warn('⚠️ StartupManager already initialized');
            return;
        }
        
        this.initialized = true;
        console.log('🔥 Starting GoldGPT application...');
        
        try {
            for (const step of this.initSteps) {
                await this.executeStep(step);
            }
            
            this.onStartupComplete();
        } catch (error) {
            console.error('❌ Startup failed:', error);
            this.onStartupFailed(error);
        }
    }
    
    async executeStep(step) {
        try {
            console.log(`🔄 Executing step: ${step.name}`);
            await step.handler();
            this.completedSteps.add(step.name);
            console.log(`✅ Step completed: ${step.name}`);
        } catch (error) {
            console.error(`❌ Step failed: ${step.name}`, error);
            this.failedSteps.add(step.name);
            throw error;
        }
    }
    
    async waitForDOM() {
        return new Promise((resolve) => {
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', resolve);
            } else {
                resolve();
            }
        });
    }
    
    async initializeConfig() {
        console.log('⚙️ Initializing configuration...');
        
        // Initialize market data manager
        if (window.marketDataManager) {
            window.marketDataManager.initialize();
        }
        
        // Initialize global configuration
        window.goldGPTConfig = {
            debug: true,
            apiEndpoints: {
                goldAPI: '/api/live-gold-price',
                news: '/api/news/latest',
                analysis: '/api/comprehensive-analysis'
            },
            updateIntervals: {
                price: 2000,
                news: 300000,
                analysis: 30000
            }
        };
        
        console.log('✅ Configuration initialized');
    }
    
    async initializeConnection() {
        console.log('🔌 Initializing WebSocket connection...');
        
        // Initialize Socket.IO if not already done
        if (typeof socket === 'undefined') {
            window.socket = io();
        }
        
        // Wait for connection
        return new Promise((resolve, reject) => {
            const timeout = setTimeout(() => {
                reject(new Error('Socket connection timeout'));
            }, 10000);
            
            socket.on('connect', () => {
                clearTimeout(timeout);
                console.log('✅ Socket connected');
                resolve();
            });
            
            socket.on('connect_error', (error) => {
                clearTimeout(timeout);
                reject(error);
            });
        });
    }
    
    async initializeComponents() {
        console.log('🔧 Initializing components...');
        
        // Initialize component loader
        if (window.ComponentLoader) {
            window.componentLoader = new ComponentLoader();
            await window.componentLoader.initialize();
        }
        
        // Initialize notification manager
        if (window.notificationManager) {
            window.notificationManager.initialize();
        }
        
        // Initialize connection manager
        if (window.connectionManager) {
            window.connectionManager.initialize();
        }
        
        console.log('✅ Components initialized');
    }
    
    async initializeDataFeeds() {
        console.log('📊 Initializing data feeds...');
        
        // Initialize price display manager
        if (window.priceDisplayManager) {
            window.priceDisplayManager.initialize();
        }
        
        // Initialize live data manager
        if (window.liveDataManager) {
            await window.liveDataManager.initialize();
        }
        
        // Initialize news manager
        if (window.enhancedNewsManager) {
            await window.enhancedNewsManager.initialize();
        }
        
        console.log('✅ Data feeds initialized');
    }
    
    async initializeCharts() {
        console.log('📈 Initializing charts...');
        
        // Initialize unified chart manager
        if (window.unifiedChartManager) {
            await window.unifiedChartManager.initialize();
        }
        
        // Initialize TradingView chart manager
        if (window.tradingViewChartManager) {
            await window.tradingViewChartManager.initialize();
        }
        
        console.log('✅ Charts initialized');
    }
    
    async finalizeSetup() {
        console.log('🎯 Finalizing setup...');
        
        // Initialize AI analysis
        if (window.aiAnalysisCenter) {
            await window.aiAnalysisCenter.initialize();
        }
        
        // Initialize right panel components
        if (window.rightPanelManager) {
            window.rightPanelManager.initialize();
        }
        
        // Initialize ML predictions
        if (window.goldMLPredictionManager) {
            await window.goldMLPredictionManager.initialize();
        }
        
        // Start periodic updates
        this.startPeriodicUpdates();
        
        console.log('✅ Setup finalized');
    }
    
    startPeriodicUpdates() {
        console.log('🔄 Starting periodic updates...');
        
        // Price updates every 2 seconds
        setInterval(() => {
            if (window.liveDataManager) {
                window.liveDataManager.updatePrices();
            }
        }, 2000);
        
        // News updates every 5 minutes
        setInterval(() => {
            if (window.enhancedNewsManager) {
                window.enhancedNewsManager.refreshNews();
            }
        }, 300000);
        
        // AI analysis updates every 30 seconds
        setInterval(() => {
            if (window.aiAnalysisCenter) {
                window.aiAnalysisCenter.refreshAnalysis();
            }
        }, 30000);
    }
    
    onStartupComplete() {
        const duration = Date.now() - this.startTime;
        console.log(`🎉 GoldGPT startup completed in ${duration}ms`);
        
        // Show success notification
        if (window.notificationManager) {
            window.notificationManager.showSuccess(
                'GoldGPT Pro Loaded',
                `All systems operational (${duration}ms)`,
                { duration: 3000 }
            );
        }
        
        // Update status indicators
        this.updateStatusIndicators(true);
        
        // Dispatch startup complete event
        document.dispatchEvent(new CustomEvent('goldgpt:startup:complete', {
            detail: { duration, completedSteps: Array.from(this.completedSteps) }
        }));
    }
    
    onStartupFailed(error) {
        const duration = Date.now() - this.startTime;
        console.error(`💥 GoldGPT startup failed after ${duration}ms:`, error);
        
        // Show error notification
        if (window.notificationManager) {
            window.notificationManager.showError(
                'Startup Failed',
                'Some components failed to load. Check console for details.',
                { persist: true }
            );
        }
        
        // Update status indicators
        this.updateStatusIndicators(false);
        
        // Dispatch startup failed event
        document.dispatchEvent(new CustomEvent('goldgpt:startup:failed', {
            detail: { error, duration, failedSteps: Array.from(this.failedSteps) }
        }));
    }
    
    updateStatusIndicators(success) {
        const statusDots = document.querySelectorAll('.status-dot');
        statusDots.forEach(dot => {
            dot.style.background = success ? 'var(--success)' : 'var(--danger)';
        });
        
        const connectionStatus = document.querySelector('.connection-status span');
        if (connectionStatus) {
            connectionStatus.textContent = success ? 'Live' : 'Error';
        }
    }
    
    getStatus() {
        return {
            initialized: this.initialized,
            duration: Date.now() - this.startTime,
            completedSteps: Array.from(this.completedSteps),
            failedSteps: Array.from(this.failedSteps),
            totalSteps: this.initSteps.length
        };
    }
}

// Initialize startup manager
window.startupManager = new StartupManager();

// Auto-start when script loads
window.startupManager.initialize().catch(error => {
    console.error('💥 Critical startup error:', error);
});

console.log('🚀 Startup Manager loaded');
