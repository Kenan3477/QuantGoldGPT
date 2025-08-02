/**
 * WebSocketManager - Advanced WebSocket Connection Manager for GoldGPT
 * 
 * Features:
 * - Automatic connection establishment to Flask-SocketIO server
 * - Connection status management (connected/disconnected/reconnecting)
 * - Event listeners for price_update, ai_analysis, and portfolio_update
 * - Exponential backoff reconnection logic
 * - Event subscription system for main application
 * - Rate limiting and error handling
 * - Authentication management
 */

class WebSocketManager {
    constructor(options = {}) {
        this.options = {
            serverUrl: options.serverUrl || window.location.origin,
            reconnectAttempts: options.reconnectAttempts || 10,
            reconnectDelay: options.reconnectDelay || 1000,
            maxReconnectDelay: options.maxReconnectDelay || 30000,
            reconnectBackoff: options.reconnectBackoff || 1.5,
            heartbeatInterval: options.heartbeatInterval || 30000,
            debug: options.debug || false,
            ...options
        };

        // Connection state
        this.connectionState = 'disconnected'; // disconnected, connecting, connected, authenticated, reconnecting, error
        this.wsClient = null;
        this.reconnectAttempt = 0;
        this.reconnectTimer = null;
        this.heartbeatTimer = null;
        this.lastHeartbeat = null;

        // Event subscription system
        this.eventSubscribers = {
            connectionStateChanged: [],
            priceUpdate: [],
            aiAnalysis: [],
            portfolioUpdate: [],
            error: [],
            authenticated: [],
            disconnected: []
        };

        // Connection statistics
        this.stats = {
            connectTime: null,
            totalReconnects: 0,
            totalErrors: 0,
            lastError: null,
            messagesReceived: 0,
            messagesSent: 0
        };

        this.log('WebSocketManager initialized', this.options);
    }

    /**
     * Initialize and establish WebSocket connection
     */
    async connect() {
        if (this.connectionState === 'connecting' || this.connectionState === 'connected') {
            this.log('Connection already in progress or established');
            return;
        }

        this.setConnectionState('connecting');
        this.log('Establishing WebSocket connection...');

        try {
            // Attempt to connect with error recovery
            await this.errorHandler.executeWithRecovery(async () => {
                await this.attemptConnection();
            }, 'connection');

            this.stats.connectTime = new Date();
            this.recoveryAttempts = 0; // Reset on successful connection
            this.log('WebSocket connection established successfully');

        } catch (error) {
            this.handleConnectionError(error);
            throw error;
        }
    }

    /**
     * Attempt WebSocket connection with library fallback
     */
    async attemptConnection() {
        const connectionStrategies = [
            () => this.connectEnhancedClient(),
            () => this.connectBasicSocketIO(),
            () => this.connectPollingFallback()
        ];

        let lastError = null;

        for (let i = 0; i < connectionStrategies.length; i++) {
            try {
                this.log(`Attempting connection strategy ${i + 1}/${connectionStrategies.length}`);
                await connectionStrategies[i]();
                this.log(`Connection strategy ${i + 1} successful`);
                return;
            } catch (error) {
                lastError = error;
                this.log(`Connection strategy ${i + 1} failed:`, error.message);
                
                // Show user-friendly message for first strategy failure
                if (i === 0) {
                    this.errorHandler.showUserMessage(
                        'Attempting alternative connection method...',
                        'warning'
                    );
                }
            }
        }

        // All strategies failed
        throw new Error(`All connection strategies failed. Last error: ${lastError?.message || 'Unknown error'}`);
    }

    /**
     * Connect using enhanced WebSocket client
     */
    async connectEnhancedClient() {
        if (typeof GoldGPTWebSocketClient === 'undefined') {
            throw new Error('Enhanced WebSocket client not available');
        }

        this.wsClient = new GoldGPTWebSocketClient(this.options.serverUrl, {
            reconnectAttempts: this.options.reconnectAttempts,
            reconnectDelay: this.options.reconnectDelay,
            maxReconnectDelay: this.options.maxReconnectDelay,
            reconnectBackoff: this.options.reconnectBackoff
        });

        this.setupEnhancedClientEvents();
        await this.wsClient.connect();
    }

    /**
     * Connect using basic Socket.IO
     */
    async connectBasicSocketIO() {
        if (typeof io === 'undefined') {
            throw new Error('Socket.IO library not available');
        }

        this.log('Using basic Socket.IO fallback');
        this.wsClient = io(this.options.serverUrl, {
            transports: ['websocket', 'polling'],
            upgrade: true,
            timeout: 20000,
            forceNew: true
        });

        this.setupBasicClientEvents();
        
        // Wait for connection
        return new Promise((resolve, reject) => {
            const timeout = setTimeout(() => {
                reject(new Error('Socket.IO connection timeout'));
            }, 20000);

            this.wsClient.on('connect', () => {
                clearTimeout(timeout);
                resolve();
            });

            this.wsClient.on('connect_error', (error) => {
                clearTimeout(timeout);
                reject(error);
            });
        });
    }

    /**
     * Connect using polling-only fallback
     */
    async connectPollingFallback() {
        if (typeof io === 'undefined') {
            throw new Error('Socket.IO library not available for polling fallback');
        }

        this.log('Using polling-only fallback');
        this.fallbackMode = true;
        
        this.wsClient = io(this.options.serverUrl, {
            transports: ['polling'], // Polling only
            upgrade: false,
            timeout: 30000,
            forceNew: true
        });

        this.setupBasicClientEvents();
        
        // Wait for connection
        return new Promise((resolve, reject) => {
            const timeout = setTimeout(() => {
                reject(new Error('Polling connection timeout'));
            }, 30000);

            this.wsClient.on('connect', () => {
                clearTimeout(timeout);
                this.errorHandler.showUserMessage(
                    'Connected in compatibility mode',
                    'info'
                );
                resolve();
            });

            this.wsClient.on('connect_error', (error) => {
                clearTimeout(timeout);
                reject(error);
            });
        });
    }

    /**
     * Set up event handlers for enhanced WebSocket client
     */
    setupEnhancedClientEvents() {
        this.wsClient.on('connected', (data) => {
            this.setConnectionState('connected');
            this.log('Enhanced WebSocket connected', data);
            this.notifySubscribers('connectionStateChanged', {
                state: 'connected',
                features: data.features || [],
                updateIntervals: data.update_intervals || {}
            });
        });

        this.wsClient.on('authenticated', (data) => {
            this.setConnectionState('authenticated');
            this.log('WebSocket authenticated', data);
            this.notifySubscribers('authenticated', data);
            this.startHeartbeat();
        });

        this.wsClient.on('disconnected', (data) => {
            this.setConnectionState('disconnected');
            this.log('WebSocket disconnected', data);
            this.stopHeartbeat();
            this.notifySubscribers('disconnected', data);
        });

        this.wsClient.on('reconnecting', (data) => {
            this.setConnectionState('reconnecting');
            this.stats.totalReconnects++;
            this.log(`WebSocket reconnecting (attempt ${data.attempt})`, data);
            this.notifySubscribers('connectionStateChanged', {
                state: 'reconnecting',
                attempt: data.attempt,
                delay: data.delay
            });
        });

        this.wsClient.on('reconnectFailed', (data) => {
            this.setConnectionState('error');
            this.log('WebSocket reconnection failed', data);
            this.handleReconnectionFailure();
        });

        this.wsClient.on('priceUpdate', (data) => {
            this.stats.messagesReceived++;
            this.log('Price update received', data);
            this.notifySubscribers('priceUpdate', data);
        });

        this.wsClient.on('aiAnalysis', (data) => {
            this.stats.messagesReceived++;
            this.log('AI analysis received', data);
            this.notifySubscribers('aiAnalysis', data);
        });

        this.wsClient.on('portfolioUpdate', (data) => {
            this.stats.messagesReceived++;
            this.log('Portfolio update received', data);
            this.notifySubscribers('portfolioUpdate', data);
        });

        this.wsClient.on('error', (data) => {
            this.stats.totalErrors++;
            this.stats.lastError = data;
            this.log('WebSocket error', data);
            this.notifySubscribers('error', data);
        });
    }

    /**
     * Set up event handlers for basic Socket.IO client
     */
    setupBasicClientEvents() {
        this.wsClient.on('connect', () => {
            this.setConnectionState('connected');
            this.log('Basic Socket.IO connected');
            this.notifySubscribers('connectionStateChanged', {
                state: 'connected',
                features: ['basic'],
                updateIntervals: {}
            });
            this.startHeartbeat();
        });

        this.wsClient.on('disconnect', (reason) => {
            this.setConnectionState('disconnected');
            this.log('Basic Socket.IO disconnected:', reason);
            this.stopHeartbeat();
            this.notifySubscribers('disconnected', { reason });

            // Auto-reconnect for basic client
            if (reason !== 'io client disconnect') {
                this.scheduleReconnect();
            }
        });

        this.wsClient.on('connect_error', (error) => {
            this.handleConnectionError(error);
        });

        // Basic event handlers
        this.wsClient.on('price_update', (data) => {
            this.stats.messagesReceived++;
            this.log('Price update received (basic)', data);
            this.notifySubscribers('priceUpdate', data);
        });

        this.wsClient.on('ai_analysis', (data) => {
            this.stats.messagesReceived++;
            this.log('AI analysis received (basic)', data);
            this.notifySubscribers('aiAnalysis', data);
        });

        this.wsClient.on('portfolio_update', (data) => {
            this.stats.messagesReceived++;
            this.log('Portfolio update received (basic)', data);
            this.notifySubscribers('portfolioUpdate', data);
        });
    }

    /**
     * Handle connection errors
     */
    handleConnectionError(error) {
        this.stats.totalErrors++;
        this.stats.lastError = error;
        this.setConnectionState('error');
        this.log('Connection error:', error);
        this.notifySubscribers('error', { type: 'connection', error: error.message });
        this.scheduleReconnect();
    }

    /**
     * Handle reconnection failure
     */
    handleReconnectionFailure() {
        this.log('All reconnection attempts failed');
        this.notifySubscribers('connectionStateChanged', {
            state: 'failed',
            totalAttempts: this.options.reconnectAttempts
        });
    }

    /**
     * Schedule reconnection with exponential backoff
     */
    scheduleReconnect() {
        if (this.reconnectAttempt >= this.options.reconnectAttempts) {
            this.handleReconnectionFailure();
            return;
        }

        const delay = Math.min(
            this.options.reconnectDelay * Math.pow(this.options.reconnectBackoff, this.reconnectAttempt),
            this.options.maxReconnectDelay
        );

        this.reconnectAttempt++;
        this.setConnectionState('reconnecting');

        this.log(`Scheduling reconnection in ${delay}ms (attempt ${this.reconnectAttempt}/${this.options.reconnectAttempts})`);

        this.reconnectTimer = setTimeout(() => {
            this.log(`Attempting reconnection (${this.reconnectAttempt}/${this.options.reconnectAttempts})`);
            this.connect().catch(error => {
                this.log('Reconnection attempt failed:', error);
            });
        }, delay);
    }

    /**
     * Set connection state and notify subscribers
     */
    setConnectionState(newState) {
        const oldState = this.connectionState;
        this.connectionState = newState;

        if (oldState !== newState) {
            this.log(`Connection state changed: ${oldState} -> ${newState}`);
            this.notifySubscribers('connectionStateChanged', {
                state: newState,
                previousState: oldState,
                timestamp: new Date()
            });
        }

        // Reset reconnect attempts on successful connection
        if (newState === 'connected' || newState === 'authenticated') {
            this.reconnectAttempt = 0;
            if (this.reconnectTimer) {
                clearTimeout(this.reconnectTimer);
                this.reconnectTimer = null;
            }
        }
    }

    /**
     * Start heartbeat mechanism
     */
    startHeartbeat() {
        this.stopHeartbeat(); // Clear any existing heartbeat

        this.heartbeatTimer = setInterval(() => {
            if (this.connectionState === 'connected' || this.connectionState === 'authenticated') {
                this.sendHeartbeat();
            }
        }, this.options.heartbeatInterval);

        this.log('Heartbeat started');
    }

    /**
     * Stop heartbeat mechanism
     */
    stopHeartbeat() {
        if (this.heartbeatTimer) {
            clearInterval(this.heartbeatTimer);
            this.heartbeatTimer = null;
            this.log('Heartbeat stopped');
        }
    }

    /**
     * Send heartbeat ping
     */
    sendHeartbeat() {
        try {
            if (this.wsClient && typeof this.wsClient.emit === 'function') {
                this.wsClient.emit('ping', { timestamp: Date.now() });
                this.lastHeartbeat = new Date();
                this.stats.messagesSent++;
                this.log('Heartbeat sent');
            }
        } catch (error) {
            this.log('Heartbeat failed:', error);
        }
    }

    /**
     * Subscribe to WebSocket events
     */
    subscribe(eventType, callback) {
        if (!this.eventSubscribers[eventType]) {
            this.eventSubscribers[eventType] = [];
        }

        this.eventSubscribers[eventType].push(callback);
        this.log(`Subscribed to ${eventType} events`);

        // Return unsubscribe function
        return () => {
            const index = this.eventSubscribers[eventType].indexOf(callback);
            if (index > -1) {
                this.eventSubscribers[eventType].splice(index, 1);
                this.log(`Unsubscribed from ${eventType} events`);
            }
        };
    }

    /**
     * Notify all subscribers of an event
     */
    notifySubscribers(eventType, data) {
        if (this.eventSubscribers[eventType]) {
            this.eventSubscribers[eventType].forEach(callback => {
                try {
                    callback(data);
                } catch (error) {
                    this.log(`Error in ${eventType} subscriber:`, error);
                }
            });
        }
    }

    /**
     * Request immediate price update
     */
    requestPriceUpdate() {
        if (this.wsClient && this.connectionState === 'authenticated') {
            if (typeof this.wsClient.requestPriceUpdate === 'function') {
                this.wsClient.requestPriceUpdate();
            } else {
                this.wsClient.emit('request_price_update');
            }
            this.stats.messagesSent++;
            this.log('Price update requested');
        } else {
            this.log('Cannot request price update - not authenticated');
        }
    }

    /**
     * Request AI analysis update
     */
    requestAIAnalysis() {
        if (this.wsClient && this.connectionState === 'authenticated') {
            if (typeof this.wsClient.requestAIAnalysis === 'function') {
                this.wsClient.requestAIAnalysis();
            } else {
                this.wsClient.emit('request_ai_analysis');
            }
            this.stats.messagesSent++;
            this.log('AI analysis requested');
        } else {
            this.log('Cannot request AI analysis - not authenticated');
        }
    }

    /**
     * Request portfolio update
     */
    requestPortfolioUpdate() {
        if (this.wsClient && this.connectionState === 'authenticated') {
            if (typeof this.wsClient.requestPortfolioUpdate === 'function') {
                this.wsClient.requestPortfolioUpdate();
            } else {
                this.wsClient.emit('request_portfolio_update');
            }
            this.stats.messagesSent++;
            this.log('Portfolio update requested');
        } else {
            this.log('Cannot request portfolio update - not authenticated');
        }
    }

    /**
     * Disconnect from WebSocket server
     */
    disconnect() {
        this.log('Disconnecting WebSocket...');
        
        this.stopHeartbeat();
        
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }

        if (this.wsClient) {
            if (typeof this.wsClient.disconnect === 'function') {
                this.wsClient.disconnect();
            } else if (typeof this.wsClient.close === 'function') {
                this.wsClient.close();
            }
            this.wsClient = null;
        }

        this.setConnectionState('disconnected');
        this.log('WebSocket disconnected manually');
    }

    /**
     * Get current connection status
     */
    getStatus() {
        return {
            state: this.connectionState,
            connected: this.connectionState === 'connected' || this.connectionState === 'authenticated',
            authenticated: this.connectionState === 'authenticated',
            reconnectAttempt: this.reconnectAttempt,
            stats: { ...this.stats },
            lastHeartbeat: this.lastHeartbeat,
            serverUrl: this.options.serverUrl
        };
    }

    /**
     * Get connection statistics
     */
    getStats() {
        return {
            ...this.stats,
            connectionState: this.connectionState,
            uptime: this.stats.connectTime ? Date.now() - this.stats.connectTime.getTime() : 0,
            reconnectAttempts: this.reconnectAttempt,
            lastHeartbeat: this.lastHeartbeat
        };
    }

    /**
     * Check if WebSocket is ready for communication
     */
    isReady() {
        return this.connectionState === 'authenticated' || 
               (this.connectionState === 'connected' && !this.wsClient?.isAuthenticated);
    }

    /**
     * Enable/disable debug logging
     */
    setDebug(enabled) {
        this.options.debug = enabled;
        this.log(`Debug logging ${enabled ? 'enabled' : 'disabled'}`);
    }

    /**
     * Internal logging method
     */
    log(message, data = null) {
        if (this.options.debug) {
            const timestamp = new Date().toLocaleTimeString();
            if (data) {
                console.log(`[${timestamp}] WebSocketManager: ${message}`, data);
            } else {
                console.log(`[${timestamp}] WebSocketManager: ${message}`);
            }
        }
    }
}

/**
 * WebSocketManagerFactory - Factory for creating and managing WebSocket instances
 */
class WebSocketManagerFactory {
    constructor() {
        this.instances = new Map();
        this.defaultOptions = {
            debug: true,
            reconnectAttempts: 10,
            reconnectDelay: 1000,
            maxReconnectDelay: 30000,
            reconnectBackoff: 1.5,
            heartbeatInterval: 30000
        };
    }

    /**
     * Create or get WebSocket manager instance
     */
    getInstance(name = 'default', options = {}) {
        if (!this.instances.has(name)) {
            const mergedOptions = { ...this.defaultOptions, ...options };
            const manager = new WebSocketManager(mergedOptions);
            this.instances.set(name, manager);
            console.log(`WebSocketManager instance '${name}' created`);
        }

        return this.instances.get(name);
    }

    /**
     * Remove WebSocket manager instance
     */
    removeInstance(name) {
        const manager = this.instances.get(name);
        if (manager) {
            manager.disconnect();
            this.instances.delete(name);
            console.log(`WebSocketManager instance '${name}' removed`);
        }
    }

    /**
     * Get all active instances
     */
    getAllInstances() {
        return Array.from(this.instances.entries()).map(([name, manager]) => ({
            name,
            status: manager.getStatus()
        }));
    }

    /**
     * Disconnect all instances
     */
    disconnectAll() {
        this.instances.forEach((manager, name) => {
            console.log(`Disconnecting WebSocketManager instance '${name}'`);
            manager.disconnect();
        });
        this.instances.clear();
    }
}

// Global factory instance
window.WebSocketManagerFactory = window.WebSocketManagerFactory || new WebSocketManagerFactory();

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { WebSocketManager, WebSocketManagerFactory };
}

// Auto-initialize default instance when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 WebSocketManager system ready');
    
    // Create default instance but don't auto-connect
    // Let the application decide when to connect
    const defaultManager = window.WebSocketManagerFactory.getInstance('default');
    
    // Make it globally accessible
    window.WebSocketManager = defaultManager;
    
    console.log('📡 Default WebSocketManager instance available as window.WebSocketManager');
    console.log('💡 Use WebSocketManager.connect() to establish connection');
});
