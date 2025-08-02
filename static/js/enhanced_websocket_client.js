/**
 * Enhanced WebSocket Client for GoldGPT
 * Includes authentication, error handling, and automatic reconnection
 */

class GoldGPTWebSocketClient {
    constructor(serverUrl = null, options = {}) {
        this.serverUrl = serverUrl || window.location.origin;
        this.socket = null;
        this.isConnected = false;
        this.isAuthenticated = false;
        this.authToken = null;
        this.clientId = null;
        
        // Configuration options
        this.options = {
            reconnectAttempts: 5,
            reconnectDelay: 1000,
            maxReconnectDelay: 30000,
            reconnectBackoff: 1.5,
            pingInterval: 30000,
            authRetries: 3,
            ...options
        };
        
        // State management
        this.reconnectAttempt = 0;
        this.reconnectTimer = null;
        this.pingTimer = null;
        this.authRetryCount = 0;
        
        // Event handlers
        this.eventHandlers = {
            connected: [],
            disconnected: [],
            authenticated: [],
            authFailed: [],
            priceUpdate: [],
            aiAnalysis: [],
            portfolioUpdate: [],
            error: [],
            reconnecting: [],
            reconnectFailed: []
        };
        
        // Rate limiting
        this.requestCounts = {};
        this.rateLimitWindow = 60000; // 1 minute
        
        console.log('🚀 GoldGPT WebSocket Client initialized');
    }
    
    /**
     * Connect to WebSocket server
     */
    async connect() {
        try {
            console.log(`🔌 Connecting to ${this.serverUrl}...`);
            
            // Initialize Socket.IO connection
            this.socket = io(this.serverUrl, {
                transports: ['websocket', 'polling'],
                upgrade: true,
                rememberUpgrade: true,
                timeout: 20000,
                forceNew: true
            });
            
            this.setupEventHandlers();
            
            return new Promise((resolve, reject) => {
                this.socket.on('connected', (data) => {
                    this.handleConnected(data);
                    resolve(data);
                });
                
                this.socket.on('connect_error', (error) => {
                    console.error('❌ Connection error:', error);
                    reject(error);
                });
                
                // Timeout fallback
                setTimeout(() => {
                    if (!this.isConnected) {
                        reject(new Error('Connection timeout'));
                    }
                }, 10000);
            });
            
        } catch (error) {
            console.error('❌ Connect error:', error);
            this.handleConnectionError(error);
            throw error;
        }
    }
    
    /**
     * Setup Socket.IO event handlers
     */
    setupEventHandlers() {
        // Connection events
        this.socket.on('connected', (data) => this.handleConnected(data));
        this.socket.on('disconnect', (reason) => this.handleDisconnected(reason));
        this.socket.on('connect_error', (error) => this.handleConnectionError(error));
        
        // Authentication events
        this.socket.on('authenticated', (data) => this.handleAuthenticated(data));
        this.socket.on('auth_failed', (data) => this.handleAuthFailed(data));
        
        // Data update events
        this.socket.on('price_update', (data) => this.handlePriceUpdate(data));
        this.socket.on('ai_analysis', (data) => this.handleAIAnalysis(data));
        this.socket.on('portfolio_update', (data) => this.handlePortfolioUpdate(data));
        
        // Error and status events
        this.socket.on('error', (data) => this.handleError(data));
        this.socket.on('server_ping', (data) => this.handleServerPing(data));
        this.socket.on('pong', (data) => this.handlePong(data));
        
        // Room events
        this.socket.on('room_joined', (data) => console.log('📡 Joined room:', data.room));
        this.socket.on('room_left', (data) => console.log('📡 Left room:', data.room));
    }
    
    /**
     * Handle successful connection
     */
    handleConnected(data) {
        console.log('✅ Connected to GoldGPT WebSocket server');
        console.log('📡 Server features:', data.features);
        console.log('⏱️ Update intervals:', data.update_intervals);
        
        this.isConnected = true;
        this.authToken = data.auth_token;
        this.clientId = data.client_id;
        this.reconnectAttempt = 0;
        
        // Start authentication
        this.authenticate();
        
        // Start ping mechanism
        this.startPing();
        
        this.triggerEvent('connected', data);
    }
    
    /**
     * Handle disconnection
     */
    handleDisconnected(reason) {
        console.warn(`🔌 Disconnected: ${reason}`);
        
        this.isConnected = false;
        this.isAuthenticated = false;
        
        this.stopPing();
        this.triggerEvent('disconnected', { reason });
        
        // Auto-reconnect unless manually disconnected
        if (reason !== 'io client disconnect') {
            this.scheduleReconnect();
        }
    }
    
    /**
     * Handle connection errors
     */
    handleConnectionError(error) {
        console.error('❌ Connection error:', error);
        
        this.isConnected = false;
        this.isAuthenticated = false;
        
        this.triggerEvent('error', { type: 'connection', error: error.message });
        this.scheduleReconnect();
    }
    
    /**
     * Authenticate with server
     */
    async authenticate() {
        if (!this.authToken) {
            console.error('❌ No auth token available');
            return;
        }
        
        try {
            console.log('🔐 Authenticating...');
            this.socket.emit('authenticate', { token: this.authToken });
            
        } catch (error) {
            console.error('❌ Authentication error:', error);
            this.handleAuthFailed({ message: error.message });
        }
    }
    
    /**
     * Handle successful authentication
     */
    handleAuthenticated(data) {
        console.log('✅ Authenticated successfully');
        
        this.isAuthenticated = true;
        this.authRetryCount = 0;
        
        // Join default rooms for updates
        this.joinRoom('prices');
        this.joinRoom('ai_analysis');
        this.joinRoom('portfolio');
        
        this.triggerEvent('authenticated', data);
    }
    
    /**
     * Handle authentication failure
     */
    handleAuthFailed(data) {
        console.error('🔐 Authentication failed:', data.message);
        
        this.isAuthenticated = false;
        this.authRetryCount++;
        
        if (this.authRetryCount < this.options.authRetries) {
            console.log(`🔄 Retrying authentication (${this.authRetryCount}/${this.options.authRetries})...`);
            setTimeout(() => this.authenticate(), 2000);
        } else {
            console.error('❌ Max authentication retries exceeded');
            this.triggerEvent('authFailed', data);
        }
    }
    
    /**
     * Handle price updates
     */
    handlePriceUpdate(data) {
        console.debug('📈 Price update received:', data.price);
        this.triggerEvent('priceUpdate', data);
    }
    
    /**
     * Handle AI analysis updates
     */
    handleAIAnalysis(data) {
        console.debug('🤖 AI analysis received:', data.signal);
        this.triggerEvent('aiAnalysis', data);
    }
    
    /**
     * Handle portfolio updates
     */
    handlePortfolioUpdate(data) {
        console.debug('💼 Portfolio update received:', data.total_value);
        this.triggerEvent('portfolioUpdate', data);
    }
    
    /**
     * Handle errors
     */
    handleError(data) {
        console.error(`❌ Server error [${data.code}]: ${data.message}`);
        this.triggerEvent('error', data);
    }
    
    /**
     * Handle server ping
     */
    handleServerPing(data) {
        console.debug('🏓 Server ping received');
        // Respond with pong
        this.socket.emit('ping');
    }
    
    /**
     * Handle pong response
     */
    handlePong(data) {
        const latency = Date.now() - this.lastPingTime;
        console.debug(`🏓 Pong received - Latency: ${latency}ms`);
    }
    
    /**
     * Start ping mechanism
     */
    startPing() {
        this.pingTimer = setInterval(() => {
            if (this.isConnected) {
                this.lastPingTime = Date.now();
                this.socket.emit('ping');
            }
        }, this.options.pingInterval);
    }
    
    /**
     * Stop ping mechanism
     */
    stopPing() {
        if (this.pingTimer) {
            clearInterval(this.pingTimer);
            this.pingTimer = null;
        }
    }
    
    /**
     * Schedule reconnection attempt
     */
    scheduleReconnect() {
        if (this.reconnectAttempt >= this.options.reconnectAttempts) {
            console.error('❌ Max reconnection attempts exceeded');
            this.triggerEvent('reconnectFailed', { attempts: this.reconnectAttempt });
            return;
        }
        
        this.reconnectAttempt++;
        const delay = Math.min(
            this.options.reconnectDelay * Math.pow(this.options.reconnectBackoff, this.reconnectAttempt - 1),
            this.options.maxReconnectDelay
        );
        
        console.log(`🔄 Reconnecting in ${delay}ms (attempt ${this.reconnectAttempt}/${this.options.reconnectAttempts})...`);
        this.triggerEvent('reconnecting', { attempt: this.reconnectAttempt, delay });
        
        this.reconnectTimer = setTimeout(() => {
            this.connect().catch(error => {
                console.error('❌ Reconnection failed:', error);
                this.scheduleReconnect();
            });
        }, delay);
    }
    
    /**
     * Join a room for targeted updates
     */
    joinRoom(room) {
        if (!this.isAuthenticated) {
            console.warn('⚠️ Cannot join room - not authenticated');
            return;
        }
        
        console.log(`📡 Joining room: ${room}`);
        this.socket.emit('join_room', { room });
    }
    
    /**
     * Leave a room
     */
    leaveRoom(room) {
        if (!this.isAuthenticated) {
            console.warn('⚠️ Cannot leave room - not authenticated');
            return;
        }
        
        console.log(`📡 Leaving room: ${room}`);
        this.socket.emit('leave_room', { room });
    }
    
    /**
     * Request immediate price update
     */
    requestPriceUpdate() {
        if (!this.checkRateLimit('requestPriceUpdate', 30)) return;
        
        if (this.isAuthenticated) {
            this.socket.emit('request_price_update');
        } else {
            console.warn('⚠️ Cannot request price update - not authenticated');
        }
    }
    
    /**
     * Request AI analysis update
     */
    requestAIAnalysis() {
        if (!this.checkRateLimit('requestAIAnalysis', 10)) return;
        
        if (this.isAuthenticated) {
            this.socket.emit('request_ai_analysis');
        } else {
            console.warn('⚠️ Cannot request AI analysis - not authenticated');
        }
    }
    
    /**
     * Request portfolio update
     */
    requestPortfolioUpdate() {
        if (!this.checkRateLimit('requestPortfolioUpdate', 20)) return;
        
        if (this.isAuthenticated) {
            this.socket.emit('request_portfolio_update');
        } else {
            console.warn('⚠️ Cannot request portfolio update - not authenticated');
        }
    }
    
    /**
     * Rate limiting check
     */
    checkRateLimit(action, maxRequests) {
        const now = Date.now();
        const windowStart = now - this.rateLimitWindow;
        
        if (!this.requestCounts[action]) {
            this.requestCounts[action] = [];
        }
        
        // Clean old requests
        this.requestCounts[action] = this.requestCounts[action].filter(time => time > windowStart);
        
        if (this.requestCounts[action].length >= maxRequests) {
            console.warn(`⚠️ Rate limit exceeded for ${action}`);
            return false;
        }
        
        this.requestCounts[action].push(now);
        return true;
    }
    
    /**
     * Add event listener
     */
    on(event, handler) {
        if (this.eventHandlers[event]) {
            this.eventHandlers[event].push(handler);
        } else {
            console.warn(`⚠️ Unknown event: ${event}`);
        }
    }
    
    /**
     * Remove event listener
     */
    off(event, handler) {
        if (this.eventHandlers[event]) {
            const index = this.eventHandlers[event].indexOf(handler);
            if (index > -1) {
                this.eventHandlers[event].splice(index, 1);
            }
        }
    }
    
    /**
     * Trigger event handlers
     */
    triggerEvent(event, data) {
        if (this.eventHandlers[event]) {
            this.eventHandlers[event].forEach(handler => {
                try {
                    handler(data);
                } catch (error) {
                    console.error(`❌ Error in ${event} handler:`, error);
                }
            });
        }
    }
    
    /**
     * Disconnect from server
     */
    disconnect() {
        console.log('🔌 Disconnecting...');
        
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }
        
        this.stopPing();
        
        if (this.socket) {
            this.socket.disconnect();
            this.socket = null;
        }
        
        this.isConnected = false;
        this.isAuthenticated = false;
    }
    
    /**
     * Get connection status
     */
    getStatus() {
        return {
            connected: this.isConnected,
            authenticated: this.isAuthenticated,
            clientId: this.clientId,
            reconnectAttempt: this.reconnectAttempt,
            serverUrl: this.serverUrl
        };
    }
}

// Usage example and integration
class GoldGPTDashboardIntegration {
    constructor() {
        this.wsClient = new GoldGPTWebSocketClient();
        this.setupEventHandlers();
    }
    
    async initialize() {
        try {
            console.log('🚀 Initializing GoldGPT WebSocket integration...');
            await this.wsClient.connect();
            console.log('✅ WebSocket integration ready');
        } catch (error) {
            console.error('❌ Failed to initialize WebSocket:', error);
            this.showConnectionError(error);
        }
    }
    
    setupEventHandlers() {
        // Connection status
        this.wsClient.on('connected', (data) => {
            this.updateConnectionStatus('connected');
            this.showNotification('Connected to real-time updates', 'success');
        });
        
        this.wsClient.on('disconnected', (data) => {
            this.updateConnectionStatus('disconnected');
            this.showNotification('Disconnected from real-time updates', 'warning');
        });
        
        this.wsClient.on('authenticated', (data) => {
            this.updateConnectionStatus('authenticated');
            this.showNotification('Authentication successful', 'success');
        });
        
        this.wsClient.on('reconnecting', (data) => {
            this.updateConnectionStatus('reconnecting');
            this.showNotification(`Reconnecting... (attempt ${data.attempt})`, 'info');
        });
        
        // Data updates
        this.wsClient.on('priceUpdate', (data) => {
            this.updatePrice(data);
        });
        
        this.wsClient.on('aiAnalysis', (data) => {
            this.updateAIAnalysis(data);
        });
        
        this.wsClient.on('portfolioUpdate', (data) => {
            this.updatePortfolio(data);
        });
        
        // Error handling
        this.wsClient.on('error', (data) => {
            this.handleError(data);
        });
    }
    
    updateConnectionStatus(status) {
        const indicator = document.querySelector('.connection-status');
        if (indicator) {
            indicator.className = `connection-status ${status}`;
            indicator.textContent = status.charAt(0).toUpperCase() + status.slice(1);
        }
    }
    
    updatePrice(data) {
        // Update price display
        const priceElement = document.querySelector('.gold-price');
        if (priceElement) {
            priceElement.textContent = `$${data.price}`;
        }
        
        // Update change indicators
        const changeElement = document.querySelector('.price-change');
        if (changeElement) {
            changeElement.textContent = `${data.change >= 0 ? '+' : ''}${data.change} (${data.change_percent}%)`;
            changeElement.className = `price-change ${data.change >= 0 ? 'positive' : 'negative'}`;
        }
        
        // Update timestamp
        const timestampElement = document.querySelector('.price-timestamp');
        if (timestampElement) {
            timestampElement.textContent = new Date(data.timestamp).toLocaleTimeString();
        }
    }
    
    updateAIAnalysis(data) {
        // Update AI signal
        const signalElement = document.querySelector('.ai-signal');
        if (signalElement) {
            signalElement.textContent = data.signal.toUpperCase();
            signalElement.className = `ai-signal ${data.signal}`;
        }
        
        // Update confidence
        const confidenceElement = document.querySelector('.ai-confidence');
        if (confidenceElement) {
            confidenceElement.textContent = `${(data.confidence * 100).toFixed(1)}%`;
        }
    }
    
    updatePortfolio(data) {
        // Update total value
        const valueElement = document.querySelector('.portfolio-value');
        if (valueElement) {
            valueElement.textContent = `$${data.total_value.toLocaleString()}`;
        }
        
        // Update P&L
        const pnlElement = document.querySelector('.portfolio-pnl');
        if (pnlElement) {
            pnlElement.textContent = `${data.daily_pnl >= 0 ? '+' : ''}$${data.daily_pnl.toLocaleString()}`;
            pnlElement.className = `portfolio-pnl ${data.daily_pnl >= 0 ? 'positive' : 'negative'}`;
        }
    }
    
    showNotification(message, type = 'info') {
        // Create or update notification
        console.log(`📢 [${type.toUpperCase()}] ${message}`);
        
        // You can implement toast notifications here
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.remove();
        }, 5000);
    }
    
    handleError(data) {
        console.error(`❌ WebSocket error [${data.code}]: ${data.message}`);
        this.showNotification(`Error: ${data.message}`, 'error');
    }
    
    showConnectionError(error) {
        this.showNotification(`Connection failed: ${error.message}`, 'error');
    }
}

// Auto-initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.goldGPTWebSocket = new GoldGPTDashboardIntegration();
    window.goldGPTWebSocket.initialize();
});

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { GoldGPTWebSocketClient, GoldGPTDashboardIntegration };
}
