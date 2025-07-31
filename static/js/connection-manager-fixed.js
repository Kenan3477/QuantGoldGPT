/**
 * GoldGPT Enhanced Connection Manager
 * Advanced WebSocket & HTTP Management with Trading 212-style reliability
 */

class EnhancedConnectionManager {
    constructor() {
        this.socket = null;
        this.isConnected = false;
        this.isReconnecting = false;
        this.connectionAttempts = 0;
        this.maxReconnectAttempts = 10;
        this.reconnectDelay = 1000;
        this.maxReconnectDelay = 30000;
        this.heartbeatInterval = null;
        this.connectionTimeout = 5000;
        this.config = null;
        
        // Enhanced debugging for gold trading
        this.debugMode = true;
        this.connectionMetrics = {
            totalConnections: 0,
            totalDisconnections: 0,
            totalReconnections: 0,
            avgConnectionTime: 0,
            lastConnectionTime: null,
            dataReceived: 0,
            errorsCount: 0
        };
        
        // Gold trading specific connection priorities
        this.goldTradingEndpoints = new Map([
            ['price', { priority: 1, timeout: 2000, retries: 5 }],
            ['trades', { priority: 2, timeout: 3000, retries: 3 }],
            ['analysis', { priority: 3, timeout: 5000, retries: 2 }],
            ['news', { priority: 4, timeout: 10000, retries: 1 }]
        ]);
        
        // Event management
        this.eventListeners = new Map();
        this.subscriptions = new Set();
        this.eventBuffer = [];
        this.bufferEnabled = true;
        
        // State management
        this.loadingStates = new Map();
        this.errorStates = new Map();
        this.retryTimers = new Map();
        
        // Connection callbacks
        this.onConnected = [];
        this.onDisconnected = [];
        this.onError = [];
        this.onReconnecting = [];
        
        // Fallback polling with gold trading optimization
        this.pollingEnabled = false;
        this.pollingInterval = null;
        this.pollingEndpoints = new Map();
        this.goldPricePollingInterval = 2000; // 2 seconds for gold price updates
        
        console.log('🔌 Enhanced Connection Manager initialized for Gold Trading');
        this._logDebugInfo('Constructor completed', 'info');
    }

    /**
     * Enhanced debug logging specifically for gold trading scenarios
     * @private
     */
    _logDebugInfo(message, level = 'info', data = null) {
        if (!this.debugMode) return;
        
        const timestamp = new Date().toISOString();
        const logEntry = {
            timestamp,
            level,
            message,
            data,
            metrics: { ...this.connectionMetrics }
        };
        
        const logSymbols = {
            info: '💙',
            warn: '⚠️',
            error: '❌',
            success: '✅',
            gold: '🥇'
        };
        
        console.log(
            `${logSymbols[level] || '📡'} [ConnectionManager] ${timestamp}: ${message}`,
            data ? data : ''
        );
        
        // Store debug info for dashboard display
        if (window.GoldGPTDebugger) {
            window.GoldGPTDebugger.addLog('ConnectionManager', logEntry);
        }
    }

    /**
     * Initialize the connection manager with gold trading optimizations
     */
    async initialize() {
        try {
            this._logDebugInfo('Starting initialization...', 'info');
            
            // Wait for configuration to be loaded
            await this._waitForConfig();
            
            // Apply gold trading specific configurations
            this._applyGoldTradingConfig();
            
            // Setup connection monitoring
            this._setupConnectionMonitoring();
            
            // Initialize connection based on priority
            if (this.config?.websocket?.enabled) {
                this._logDebugInfo('WebSocket enabled, attempting connection', 'info');
                await this.connect();
            } else {
                this._logDebugInfo('WebSocket disabled, enabling polling fallback', 'warn');
                this._enablePolling();
            }
            
            this._logDebugInfo('Connection Manager initialization completed', 'success');
            
        } catch (error) {
            this._logDebugInfo('Initialization failed', 'error', error);
            console.error('❌ Connection Manager initialization failed:', error);
            
            // Fallback to polling mode
            this._enablePolling();
        }
    }

    /**
     * Apply gold trading specific connection configurations
     * @private
     */
    _applyGoldTradingConfig() {
        if (!this.config) return;
        
        try {
            // Optimize connection settings for gold trading
            if (this.config.goldTrading?.enabled) {
                this.connectionTimeout = 3000; // Faster timeout for gold trading
                this.reconnectDelay = 500;     // Faster reconnection
                this.goldPricePollingInterval = 1500; // More frequent gold price updates
                
                this._logDebugInfo('Applied gold trading connection optimizations', 'gold');
            }
            
            // Configure WebSocket settings from config
            if (this.config.websocket) {
                this.maxReconnectAttempts = this.config.websocket.maxReconnectAttempts || 10;
                this.reconnectDelay = this.config.websocket.reconnectInterval || 2000;
                this.heartbeatInterval = this.config.websocket.heartbeatInterval || 25000;
                
                if (this.config.websocket.fallbackToPolling) {
                    this.pollingEnabled = true;
                    this.goldPricePollingInterval = this.config.websocket.pollingInterval || 3000;
                }
            }
            
            this._logDebugInfo('Configuration applied successfully', 'success', {
                timeout: this.connectionTimeout,
                reconnectDelay: this.reconnectDelay,
                maxAttempts: this.maxReconnectAttempts
            });
            
        } catch (error) {
            this._logDebugInfo('Failed to apply gold trading config', 'error', error);
        }
    }

    /**
     * Setup connection health monitoring
     * @private
     */
    _setupConnectionMonitoring() {
        // Monitor connection health every 30 seconds
        setInterval(() => {
            this._checkConnectionHealth();
        }, 30000);
        
        // Monitor gold price data flow
        if (this.config?.goldTrading?.enabled) {
            this._setupGoldPriceMonitoring();
        }
    }

    /**
     * Setup gold price specific monitoring
     * @private
     */
    _setupGoldPriceMonitoring() {
        let lastGoldUpdate = Date.now();
        
        this.on('gold_price_update', () => {
            lastGoldUpdate = Date.now();
        });
        
        // Check if gold price updates are flowing
        setInterval(() => {
            const timeSinceLastUpdate = Date.now() - lastGoldUpdate;
            const threshold = this.goldPricePollingInterval * 3; // 3x polling interval
            
            if (timeSinceLastUpdate > threshold) {
                this._logDebugInfo(
                    `Gold price data stale (${Math.round(timeSinceLastUpdate/1000)}s)`,
                    'warn'
                );
                
                // Trigger reconnection if WebSocket is supposed to be connected
                if (this.isConnected && !this.pollingEnabled) {
                    this._forceReconnect('Gold price data stale');
                }
            }
        }, this.goldPricePollingInterval * 2);
    }

    /**
     * Wait for configuration manager to be ready
     */
    async _waitForConfig() {
        return new Promise((resolve) => {
            const checkConfig = () => {
                if (window.configManager && window.configManager.isLoaded) {
                    resolve();
                } else {
                    setTimeout(checkConfig, 100);
                }
            };
            checkConfig();
        });
    }

    /**
     * Establish WebSocket connection with enhanced retry logic
     */
    async connect() {
        if (this.isConnected || this.isReconnecting) {
            return;
        }

        return new Promise((resolve, reject) => {
            try {
                this.isReconnecting = true;
                this.updateConnectionStatus('connecting');
                
                const wsUrl = this.config?.endpoints?.websocket || 'ws://localhost:5000';
                
                // Initialize Socket.IO connection
                this.socket = io(wsUrl, {
                    timeout: this.connectionTimeout,
                    reconnection: false, // We handle reconnection manually
                    forceNew: true
                });

                // Connection success
                this.socket.on('connect', () => {
                    console.log('🔗 WebSocket connected successfully');
                    this.isConnected = true;
                    this.isReconnecting = false;
                    this.connectionAttempts = 0;
                    this.reconnectDelay = 1000;
                    
                    this.updateConnectionStatus('connected');
                    this.notifyCallbacks(this.onConnected);
                    
                    // Process buffered events
                    this.processBufferedEvents();
                    
                    // Stop polling if it was enabled
                    this.stopPolling();
                    
                    resolve();
                });

                // Connection error
                this.socket.on('connect_error', (error) => {
                    console.error('🔴 WebSocket connection error:', error);
                    this.isConnected = false;
                    this.isReconnecting = false;
                    this.handleConnectionError(error);
                    reject(error);
                });

                // Disconnection
                this.socket.on('disconnect', (reason) => {
                    console.warn('⚠️ WebSocket disconnected:', reason);
                    this.isConnected = false;
                    this.isReconnecting = false;
                    this.updateConnectionStatus('disconnected');
                    this.notifyCallbacks(this.onDisconnected, reason);
                    
                    // Start fallback polling if enabled
                    if (this.config?.fallbackToPolling) {
                        this.startPolling();
                    }
                    
                    // Attempt reconnection if not intentional
                    if (reason !== 'io client disconnect') {
                        this.scheduleReconnect();
                    }
                });

                // Setup default event handlers
                this.setupDefaultEventHandlers();

            } catch (error) {
                console.error('❌ Connection setup failed:', error);
                this.isReconnecting = false;
                reject(error);
            }
        });
    }

    /**
     * Handle connection errors with exponential backoff
     */
    handleConnectionError(error) {
        this.connectionAttempts++;
        
        if (this.connectionAttempts <= this.maxReconnectAttempts) {
            console.log(`🔄 Retrying connection (${this.connectionAttempts}/${this.maxReconnectAttempts})`);
            this.scheduleReconnect();
        } else {
            console.error('❌ Max reconnection attempts reached');
            this.updateConnectionStatus('failed');
            this.notifyCallbacks(this.onError, error);
            
            // Start polling as last resort
            if (this.config?.fallbackToPolling) {
                this.startPolling();
            }
        }
    }

    /**
     * Schedule reconnection with exponential backoff
     */
    scheduleReconnect() {
        if (this.isReconnecting) {
            return;
        }
        
        this.updateConnectionStatus('reconnecting');
        this.notifyCallbacks(this.onReconnecting);
        
        setTimeout(() => {
            this.connect().catch(error => {
                console.error('🔴 Reconnection failed:', error);
            });
        }, this.reconnectDelay);
        
        // Exponential backoff with jitter
        const jitter = Math.random() * 1000;
        this.reconnectDelay = Math.min(this.reconnectDelay * 2 + jitter, this.maxReconnectDelay);
    }

    /**
     * Start fallback polling when WebSocket is unavailable
     */
    startPolling() {
        if (this.pollingEnabled) {
            return;
        }
        
        console.log('🔄 Starting fallback polling...');
        this.pollingEnabled = true;
        
        const pollInterval = this.config?.pollingInterval || 5000;
        
        this.pollingInterval = setInterval(() => {
            this.pollEndpoints();
        }, pollInterval);
        
        // Initial poll
        this.pollEndpoints();
    }

    /**
     * Stop fallback polling
     */
    stopPolling() {
        if (!this.pollingEnabled) {
            return;
        }
        
        console.log('⏹️ Stopping fallback polling...');
        this.pollingEnabled = false;
        
        if (this.pollingInterval) {
            clearInterval(this.pollingInterval);
            this.pollingInterval = null;
        }
    }

    /**
     * Poll registered endpoints when WebSocket is unavailable
     */
    async pollEndpoints() {
        for (const [endpoint, lastData] of this.pollingEndpoints) {
            try {
                const response = await fetch(endpoint, {
                    method: 'GET',
                    headers: { 'Content-Type': 'application/json' },
                    timeout: 3000
                });
                
                if (response.ok) {
                    const data = await response.json();
                    
                    // Only emit if data has changed
                    if (JSON.stringify(data) !== JSON.stringify(lastData)) {
                        this.pollingEndpoints.set(endpoint, data);
                        this.emit('polling_data', { endpoint, data });
                    }
                }
                
            } catch (error) {
                console.error(`❌ Polling error for ${endpoint}:`, error);
            }
        }
    }

    /**
     * Register endpoint for polling fallback
     */
    registerPollingEndpoint(endpoint) {
        this.pollingEndpoints.set(endpoint, null);
        console.log(`📡 Registered polling endpoint: ${endpoint}`);
    }

    /**
     * Setup default WebSocket event handlers
     */
    setupDefaultEventHandlers() {
        if (!this.socket) return;

        // Price updates
        this.socket.on('price_update', (data) => {
            this.clearError('price_update');
            this.emit('price_update', data);
        });

        // News updates
        this.socket.on('news_update', (data) => {
            this.clearError('news_update');
            this.emit('news_update', data);
        });

        // AI analysis updates
        this.socket.on('ai_analysis_update', (data) => {
            this.clearError('ai_analysis');
            this.emit('ai_analysis_update', data);
        });

        // Market data updates
        this.socket.on('market_data_update', (data) => {
            this.clearError('market_data');
            this.emit('market_data_update', data);
        });

        // Trade execution updates
        this.socket.on('trade_executed', (data) => {
            this.clearError('trade_execution');
            this.emit('trade_executed', data);
        });

        // System notifications
        this.socket.on('system_notification', (data) => {
            this.emit('system_notification', data);
        });

        // Heartbeat response
        this.socket.on('pong', (data) => {
            console.log('💓 Heartbeat received');
        });

        // Error handling
        this.socket.on('error', (error) => {
            console.error('🔴 WebSocket error:', error);
            this.handleError('websocket', error);
        });
    }

    /**
     * Start heartbeat to monitor connection health
     */
    startHeartbeat() {
        const heartbeatInterval = this.config?.heartbeatInterval || 30000;
        
        this.heartbeatInterval = setInterval(() => {
            if (this.isConnected && this.socket) {
                this.socket.emit('ping', { timestamp: Date.now() });
            }
        }, heartbeatInterval);
    }

    /**
     * Stop heartbeat
     */
    stopHeartbeat() {
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
            this.heartbeatInterval = null;
        }
    }

    /**
     * Process buffered events after reconnection
     */
    processBufferedEvents() {
        if (this.eventBuffer.length === 0) {
            return;
        }
        
        console.log(`📦 Processing ${this.eventBuffer.length} buffered events`);
        
        for (const event of this.eventBuffer) {
            this.emit(event.name, event.data);
        }
        
        this.eventBuffer = [];
    }

    /**
     * Emit event to all listeners with buffering support
     */
    emit(eventName, data) {
        // Buffer events if not connected and buffering is enabled
        if (!this.isConnected && this.bufferEnabled) {
            this.eventBuffer.push({ name: eventName, data, timestamp: Date.now() });
            
            // Limit buffer size
            if (this.eventBuffer.length > 100) {
                this.eventBuffer.shift();
            }
            
            return;
        }
        
        const listeners = this.eventListeners.get(eventName) || [];
        listeners.forEach(callback => {
            try {
                callback(data);
            } catch (error) {
                console.error(`❌ Error in event listener for ${eventName}:`, error);
            }
        });
    }

    /**
     * Add event listener with cleanup tracking
     */
    on(eventName, callback, context = null) {
        if (!this.eventListeners.has(eventName)) {
            this.eventListeners.set(eventName, []);
        }
        
        const listenerInfo = { callback, context };
        this.eventListeners.get(eventName).push(listenerInfo);
        
        // Return cleanup function
        return () => this.off(eventName, callback);
    }

    /**
     * Remove event listener
     */
    off(eventName, callback) {
        const listeners = this.eventListeners.get(eventName);
        if (listeners) {
            const index = listeners.findIndex(l => l.callback === callback);
            if (index !== -1) {
                listeners.splice(index, 1);
            }
        }
    }

    /**
     * Remove all event listeners for a context
     */
    offContext(context) {
        this.eventListeners.forEach((listeners, eventName) => {
            const filteredListeners = listeners.filter(l => l.context !== context);
            this.eventListeners.set(eventName, filteredListeners);
        });
    }

    /**
     * Send data via WebSocket with fallback to HTTP
     */
    async send(eventName, data = {}) {
        if (this.isConnected && this.socket) {
            return new Promise((resolve, reject) => {
                const timeout = setTimeout(() => {
                    reject(new Error('WebSocket send timeout'));
                }, this.connectionTimeout);

                try {
                    this.socket.emit(eventName, data, (response) => {
                        clearTimeout(timeout);
                        if (response && response.error) {
                            reject(new Error(response.error));
                        } else {
                            resolve(response || {});
                        }
                    });
                } catch (error) {
                    clearTimeout(timeout);
                    reject(error);
                }
            });
        } else {
            // Fallback to HTTP request
            return this.request(`/api/ws/${eventName}`, {
                method: 'POST',
                body: JSON.stringify(data)
            });
        }
    }

    /**
     * Make HTTP request with enhanced error handling
     */
    async request(url, options = {}) {
        const requestId = this.generateRequestId();
        
        try {
            this.setLoading(requestId, true);
            
            const defaultOptions = {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                    'X-Request-ID': requestId
                },
                timeout: 10000
            };

            const mergedOptions = { ...defaultOptions, ...options };
            
            // Add timeout wrapper
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), mergedOptions.timeout);
            mergedOptions.signal = controller.signal;

            const response = await fetch(url, mergedOptions);
            clearTimeout(timeoutId);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            this.clearError(requestId);
            return data;

        } catch (error) {
            this.handleError(requestId, error);
            throw error;
        } finally {
            this.setLoading(requestId, false);
        }
    }

    /**
     * Request with retry logic
     */
    async requestWithRetry(url, options = {}, maxRetries = 3) {
        let lastError;
        
        for (let attempt = 1; attempt <= maxRetries; attempt++) {
            try {
                return await this.request(url, options);
            } catch (error) {
                lastError = error;
                
                if (attempt < maxRetries) {
                    const delay = Math.pow(2, attempt) * 1000; // Exponential backoff
                    console.log(`🔄 Request failed, retrying in ${delay}ms (attempt ${attempt}/${maxRetries})`);
                    await new Promise(resolve => setTimeout(resolve, delay));
                }
            }
        }
        
        throw lastError;
    }

    /**
     * Subscribe to real-time updates
     */
    subscribe(eventName, callback, context = null) {
        const subscription = {
            eventName,
            callback,
            context,
            id: this.generateRequestId()
        };
        
        this.subscriptions.add(subscription);
        this.on(eventName, callback, context);
        
        // Send subscription request if connected
        if (this.isConnected) {
            this.send('subscribe', { eventName }).catch(error => {
                console.error(`❌ Subscription failed for ${eventName}:`, error);
            });
        }
        
        return subscription.id;
    }

    /**
     * Unsubscribe from real-time updates
     */
    unsubscribe(subscriptionId) {
        const subscription = [...this.subscriptions].find(s => s.id === subscriptionId);
        if (subscription) {
            this.subscriptions.delete(subscription);
            this.off(subscription.eventName, subscription.callback);
            
            // Send unsubscription request if connected
            if (this.isConnected) {
                this.send('unsubscribe', { eventName: subscription.eventName }).catch(error => {
                    console.error(`❌ Unsubscription failed for ${subscription.eventName}:`, error);
                });
            }
        }
    }

    /**
     * Set loading state for a component
     */
    setLoading(key, loading) {
        this.loadingStates.set(key, loading);
        this.updateLoadingIndicator(key, loading);
        this.emit('loading_state_changed', { key, loading });
    }

    /**
     * Check if any component is loading
     */
    isLoading(key = null) {
        if (key) {
            return this.loadingStates.get(key) || false;
        }
        return [...this.loadingStates.values()].some(loading => loading);
    }

    /**
     * Handle error with retry options
     */
    handleError(key, error, retryCallback = null) {
        const errorInfo = {
            key,
            error,
            timestamp: Date.now(),
            retryCallback
        };
        
        this.errorStates.set(key, errorInfo);
        this.updateErrorIndicator(key, error);
        this.emit('error_occurred', errorInfo);
        
        // Log error details
        console.error(`❌ Error in ${key}:`, error);
        
        // Show error notification
        if (window.notificationManager) {
            window.notificationManager.showError(
                `Error in ${key}`,
                error.message || 'An unexpected error occurred',
                retryCallback ? { retry: retryCallback } : undefined
            );
        }
    }

    /**
     * Clear error state
     */
    clearError(key) {
        this.errorStates.delete(key);
        this.updateErrorIndicator(key, null);
        this.emit('error_cleared', { key });
    }

    /**
     * Retry a failed operation
     */
    retry(key) {
        const errorInfo = this.errorStates.get(key);
        if (errorInfo && errorInfo.retryCallback) {
            this.clearError(key);
            errorInfo.retryCallback();
        }
    }

    /**
     * Setup global error handling
     */
    setupErrorHandling() {
        // Handle unhandled promise rejections
        window.addEventListener('unhandledrejection', (event) => {
            console.error('🔴 Unhandled promise rejection:', event.reason);
            this.handleError('global', event.reason);
        });

        // Handle JavaScript errors
        window.addEventListener('error', (event) => {
            const error = event.error || event.message || 'Unknown JavaScript error';
            console.error('🔴 JavaScript error:', error);
            this.handleError('global', error);
        });

        // Handle network errors
        window.addEventListener('offline', () => {
            console.warn('🔴 Network offline');
            this.handleError('network', new Error('Network connection lost'));
        });

        window.addEventListener('online', () => {
            console.log('🟢 Network online');
            this.clearError('network');
            
            // Attempt reconnection
            if (!this.isConnected) {
                this.connect();
            }
        });
    }

    /**
     * Create connection status indicator
     */
    createConnectionStatusIndicator() {
        const indicator = document.createElement('div');
        indicator.id = 'connection-status';
        indicator.className = 'connection-status';
        indicator.innerHTML = `
            <div class="connection-icon">
                <div class="connection-dot"></div>
            </div>
            <span class="connection-text">Connecting...</span>
        `;
        
        document.body.appendChild(indicator);
        
        // Add click handler for connection details
        indicator.addEventListener('click', () => {
            this.showConnectionDetails();
        });
    }

    /**
     * Update connection status indicator
     */
    updateConnectionStatus(status) {
        const indicator = document.getElementById('connection-status');
        if (!indicator) return;
        
        const dot = indicator.querySelector('.connection-dot');
        const text = indicator.querySelector('.connection-text');
        
        indicator.className = `connection-status ${status}`;
        
        switch (status) {
            case 'connected':
                dot.className = 'connection-dot connected';
                text.textContent = 'Connected';
                break;
            case 'connecting':
                dot.className = 'connection-dot connecting';
                text.textContent = 'Connecting...';
                break;
            case 'reconnecting':
                dot.className = 'connection-dot reconnecting';
                text.textContent = 'Reconnecting...';
                break;
            case 'disconnected':
                dot.className = 'connection-dot disconnected';
                text.textContent = 'Disconnected';
                break;
            case 'failed':
                dot.className = 'connection-dot failed';
                text.textContent = 'Connection Failed';
                break;
        }
    }

    /**
     * Show connection details modal
     */
    showConnectionDetails() {
        const modal = document.createElement('div');
        modal.className = 'connection-modal';
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h3>Connection Details</h3>
                    <button class="close-button">&times;</button>
                </div>
                <div class="modal-body">
                    <div class="connection-info">
                        <div class="info-item">
                            <label>Status:</label>
                            <span class="status-value">${this.isConnected ? 'Connected' : 'Disconnected'}</span>
                        </div>
                        <div class="info-item">
                            <label>Connection Attempts:</label>
                            <span>${this.connectionAttempts}</span>
                        </div>
                        <div class="info-item">
                            <label>Polling Enabled:</label>
                            <span>${this.pollingEnabled ? 'Yes' : 'No'}</span>
                        </div>
                        <div class="info-item">
                            <label>Active Subscriptions:</label>
                            <span>${this.subscriptions.size}</span>
                        </div>
                        <div class="info-item">
                            <label>Error States:</label>
                            <span>${this.errorStates.size}</span>
                        </div>
                    </div>
                    <div class="connection-actions">
                        <button class="btn-primary" onclick="window.connectionManager.reconnect()">
                            Reconnect
                        </button>
                        <button class="btn-secondary" onclick="window.connectionManager.clearAllErrors()">
                            Clear Errors
                        </button>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        
        // Close modal handlers
        modal.querySelector('.close-button').addEventListener('click', () => {
            modal.remove();
        });
        
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.remove();
            }
        });
    }

    /**
     * Update loading indicator for a component
     */
    updateLoadingIndicator(key, loading) {
        const element = document.querySelector(`[data-loading-key="${key}"]`);
        if (element) {
            if (loading) {
                element.classList.add('loading');
                element.setAttribute('aria-busy', 'true');
            } else {
                element.classList.remove('loading');
                element.setAttribute('aria-busy', 'false');
            }
        }
    }

    /**
     * Update error indicator for a component
     */
    updateErrorIndicator(key, error) {
        const element = document.querySelector(`[data-error-key="${key}"]`);
        if (element) {
            if (error) {
                element.classList.add('error');
                element.setAttribute('aria-invalid', 'true');
                element.title = error.message || 'An error occurred';
            } else {
                element.classList.remove('error');
                element.setAttribute('aria-invalid', 'false');
                element.title = '';
            }
        }
    }

    /**
     * Inject loading and error styles
     */
    injectLoadingStyles() {
        const style = document.createElement('style');
        style.textContent = `
            /* Connection Status Indicator */
            .connection-status {
                position: fixed;
                top: 10px;
                right: 10px;
                background: rgba(0, 0, 0, 0.8);
                color: white;
                padding: 8px 12px;
                border-radius: 20px;
                font-size: 12px;
                display: flex;
                align-items: center;
                gap: 8px;
                z-index: 10000;
                cursor: pointer;
                transition: all 0.3s ease;
            }

            .connection-status:hover {
                background: rgba(0, 0, 0, 0.9);
            }

            .connection-dot {
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: #666;
                transition: all 0.3s ease;
            }

            .connection-dot.connected {
                background: #4CAF50;
                box-shadow: 0 0 10px rgba(76, 175, 80, 0.5);
            }

            .connection-dot.connecting {
                background: #FF9800;
                animation: pulse 1s infinite;
            }

            .connection-dot.reconnecting {
                background: #2196F3;
                animation: pulse 1s infinite;
            }

            .connection-dot.disconnected {
                background: #F44336;
            }

            .connection-dot.failed {
                background: #F44336;
                animation: flash 0.5s infinite;
            }

            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.5; }
            }

            @keyframes flash {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.3; }
            }

            /* Loading States */
            .loading {
                position: relative;
                pointer-events: none;
                opacity: 0.7;
            }

            .loading::before {
                content: '';
                position: absolute;
                top: 50%;
                left: 50%;
                width: 20px;
                height: 20px;
                margin: -10px 0 0 -10px;
                border: 2px solid #f3f3f3;
                border-top: 2px solid #3498db;
                border-radius: 50%;
                animation: spin 1s linear infinite;
                z-index: 1000;
            }

            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }

            /* Error States */
            .error {
                border: 2px solid #f44336 !important;
                background-color: rgba(244, 67, 54, 0.1) !important;
            }

            .error::after {
                content: '⚠️';
                position: absolute;
                top: 5px;
                right: 5px;
                font-size: 14px;
                color: #f44336;
            }

            /* Connection Modal */
            .connection-modal {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.5);
                display: flex;
                justify-content: center;
                align-items: center;
                z-index: 10001;
            }

            .modal-content {
                background: white;
                border-radius: 8px;
                width: 90%;
                max-width: 500px;
                max-height: 80vh;
                overflow-y: auto;
            }

            .modal-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 20px;
                border-bottom: 1px solid #eee;
            }

            .modal-header h3 {
                margin: 0;
                font-size: 18px;
            }

            .close-button {
                background: none;
                border: none;
                font-size: 24px;
                cursor: pointer;
                color: #666;
            }

            .close-button:hover {
                color: #000;
            }

            .modal-body {
                padding: 20px;
            }

            .connection-info {
                margin-bottom: 20px;
            }

            .info-item {
                display: flex;
                justify-content: space-between;
                margin-bottom: 10px;
                padding: 8px 0;
                border-bottom: 1px solid #f0f0f0;
            }

            .info-item label {
                font-weight: 500;
                color: #333;
            }

            .status-value {
                font-weight: 500;
                padding: 2px 8px;
                border-radius: 4px;
                font-size: 12px;
            }

            .connection-actions {
                display: flex;
                gap: 10px;
                justify-content: flex-end;
            }

            .btn-primary, .btn-secondary {
                padding: 8px 16px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 14px;
                transition: all 0.3s ease;
            }

            .btn-primary {
                background: #3498db;
                color: white;
            }

            .btn-primary:hover {
                background: #2980b9;
            }

            .btn-secondary {
                background: #95a5a6;
                color: white;
            }

            .btn-secondary:hover {
                background: #7f8c8d;
            }

            /* Responsive Design */
            @media (max-width: 768px) {
                .connection-status {
                    top: 5px;
                    right: 5px;
                    font-size: 11px;
                    padding: 6px 10px;
                }

                .modal-content {
                    width: 95%;
                    margin: 10px;
                }

                .connection-actions {
                    flex-direction: column;
                }
            }
        `;
        
        document.head.appendChild(style);
    }

    /**
     * Notify callbacks for connection events
     */
    notifyCallbacks(callbacks, data = null) {
        callbacks.forEach(callback => {
            try {
                callback(data);
            } catch (error) {
                console.error('❌ Error in connection callback:', error);
            }
        });
    }

    /**
     * Generate unique request ID
     */
    generateRequestId() {
        return `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    /**
     * Reconnect to WebSocket
     */
    async reconnect() {
        console.log('🔄 Manual reconnection initiated');
        
        if (this.socket) {
            this.socket.disconnect();
        }
        
        this.isConnected = false;
        this.connectionAttempts = 0;
        this.reconnectDelay = 1000;
        
        try {
            await this.connect();
            console.log('✅ Manual reconnection successful');
        } catch (error) {
            console.error('❌ Manual reconnection failed:', error);
        }
    }

    /**
     * Clear all error states
     */
    clearAllErrors() {
        console.log('🧹 Clearing all error states');
        
        this.errorStates.clear();
        
        // Clear visual error indicators
        document.querySelectorAll('.error').forEach(element => {
            element.classList.remove('error');
            element.setAttribute('aria-invalid', 'false');
            element.title = '';
        });
        
        this.emit('all_errors_cleared');
    }

    /**
     * Get connection statistics
     */
    getConnectionStats() {
        return {
            isConnected: this.isConnected,
            connectionAttempts: this.connectionAttempts,
            pollingEnabled: this.pollingEnabled,
            subscriptions: this.subscriptions.size,
            errorStates: this.errorStates.size,
            loadingStates: this.loadingStates.size,
            eventBuffer: this.eventBuffer.length
        };
    }

    /**
     * Disconnect and cleanup
     */
    disconnect() {
        console.log('🔌 Disconnecting connection manager');
        
        this.isConnected = false;
        this.stopHeartbeat();
        this.stopPolling();
        
        if (this.socket) {
            this.socket.disconnect();
            this.socket = null;
        }
        
        // Clear all timers
        this.retryTimers.forEach(timer => clearTimeout(timer));
        this.retryTimers.clear();
        
        // Clear all states
        this.loadingStates.clear();
        this.errorStates.clear();
        this.eventBuffer = [];
        
        // Remove connection indicator
        const indicator = document.getElementById('connection-status');
        if (indicator) {
            indicator.remove();
        }
        
        console.log('✅ Connection manager disconnected');
    }

    /**
     * Cleanup all resources
     */
    cleanup() {
        console.log('🧹 Cleaning up connection manager');
        
        this.disconnect();
        
        // Remove all event listeners
        this.eventListeners.clear();
        this.subscriptions.clear();
        
        // Remove connection callbacks
        this.onConnected = [];
        this.onDisconnected = [];
        this.onError = [];
        this.onReconnecting = [];
        
        console.log('✅ Connection manager cleanup complete');
    }
}

// Initialize global connection manager
window.connectionManager = new EnhancedConnectionManager();

// Auto-initialize the connection manager
document.addEventListener('DOMContentLoaded', () => {
    if (window.connectionManager && typeof window.connectionManager.initialize === 'function') {
        window.connectionManager.initialize().catch(error => {
            console.error('❌ Failed to initialize connection manager:', error);
        });
    }
});

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = EnhancedConnectionManager;
}

console.log('🔌 Connection Manager loaded successfully');
