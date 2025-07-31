/**
 * GoldGPT Connection Manager Integration Demo
 * Shows how to properly integrate components with the connection manager
 */

// Example component that demonstrates proper integration
class ExampleComponent {
    constructor() {
        this.componentId = 'example-component';
        this.eventCleanup = [];
        this.data = null;
        this.isInitialized = false;
        
        this.init();
    }
    
    async init() {
        console.log('📋 Initializing Example Component...');
        
        // Wait for connection manager to be available
        await this.waitForConnectionManager();
        
        // Setup connection manager integration
        this.setupConnectionManagerIntegration();
        
        // Load initial data
        await this.loadData();
        
        this.isInitialized = true;
        console.log('✅ Example Component initialized');
    }
    
    async waitForConnectionManager() {
        return new Promise((resolve) => {
            const checkConnectionManager = () => {
                if (window.connectionManager) {
                    resolve();
                } else {
                    setTimeout(checkConnectionManager, 100);
                }
            };
            checkConnectionManager();
        });
    }
    
    setupConnectionManagerIntegration() {
        // Add data-component attribute to containers
        const containers = document.querySelectorAll('.example-container');
        containers.forEach(container => {
            container.setAttribute('data-component', this.componentId);
        });
        
        // Setup retry handler
        const retryCleanup = window.connectionManager.on('retry', (data) => {
            if (data.componentId === this.componentId) {
                console.log('🔄 Retrying example component...');
                this.loadData();
            }
        });
        this.eventCleanup.push(retryCleanup);
        
        // Setup WebSocket event handlers
        const dataUpdateCleanup = window.connectionManager.on('data_update', (data) => {
            console.log('📊 Real-time data update:', data);
            this.handleDataUpdate(data);
        });
        this.eventCleanup.push(dataUpdateCleanup);
        
        // Setup connection state handlers
        const connectionStateCleanup = window.connectionManager.on('connection_state_change', (state) => {
            console.log('🔌 Connection state changed:', state);
            this.handleConnectionStateChange(state);
        });
        this.eventCleanup.push(connectionStateCleanup);
    }
    
    async loadData() {
        try {
            // Use connection manager for HTTP requests
            const data = await window.connectionManager.request('/api/example-data', {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            this.data = data;
            this.renderData(data);
            
        } catch (error) {
            console.error('Error loading example data:', error);
            // Error handling is automatically managed by connection manager
        }
    }
    
    renderData(data) {
        const containers = document.querySelectorAll(`[data-component="${this.componentId}"]`);
        containers.forEach(container => {
            container.innerHTML = `
                <div class="example-data">
                    <h3>Example Data</h3>
                    <pre>${JSON.stringify(data, null, 2)}</pre>
                </div>
            `;
        });
    }
    
    handleDataUpdate(data) {
        // Handle real-time data updates
        this.data = { ...this.data, ...data };
        this.renderData(this.data);
    }
    
    handleConnectionStateChange(state) {
        const containers = document.querySelectorAll(`[data-component="${this.componentId}"]`);
        
        containers.forEach(container => {
            // Add visual indicators for connection state
            container.classList.remove('connected', 'disconnected', 'reconnecting');
            container.classList.add(state);
        });
    }
    
    cleanup() {
        console.log('🧹 Cleaning up Example Component...');
        
        // Clean up event listeners
        this.eventCleanup.forEach(cleanup => cleanup());
        this.eventCleanup = [];
        
        // Clean up from connection manager
        if (window.connectionManager) {
            window.connectionManager.offContext(this);
        }
        
        console.log('✅ Example Component cleaned up');
    }
}

// Connection Manager Usage Examples
class ConnectionManagerExamples {
    static async demonstrateHTTPRequests() {
        console.log('🌐 Demonstrating HTTP Requests with Connection Manager');
        
        try {
            // GET request
            const getData = await window.connectionManager.request('/api/data');
            console.log('GET response:', getData);
            
            // POST request
            const postData = await window.connectionManager.request('/api/data', {
                method: 'POST',
                body: JSON.stringify({ key: 'value' })
            });
            console.log('POST response:', postData);
            
        } catch (error) {
            console.error('HTTP request error:', error);
        }
    }
    
    static demonstrateWebSocketCommunication() {
        console.log('🔌 Demonstrating WebSocket Communication');
        
        // Listen for events
        const cleanup1 = window.connectionManager.on('price_update', (data) => {
            console.log('💰 Price update:', data);
        });
        
        const cleanup2 = window.connectionManager.on('news_update', (data) => {
            console.log('📰 News update:', data);
        });
        
        // Send WebSocket message
        window.connectionManager.send('subscribe', {
            symbols: ['XAUUSD', 'EURUSD'],
            dataTypes: ['price', 'news']
        }).then(response => {
            console.log('✅ Subscription successful:', response);
        }).catch(error => {
            console.error('❌ Subscription failed:', error);
        });
        
        // Clean up after 30 seconds
        setTimeout(() => {
            cleanup1();
            cleanup2();
            console.log('🧹 WebSocket listeners cleaned up');
        }, 30000);
    }
    
    static demonstrateLoadingStates() {
        console.log('⏳ Demonstrating Loading States');
        
        // Show loading state
        window.connectionManager.setLoading('demo-component', true);
        
        // Simulate async operation
        setTimeout(() => {
            window.connectionManager.setLoading('demo-component', false);
            console.log('✅ Loading state cleared');
        }, 2000);
    }
    
    static demonstrateErrorHandling() {
        console.log('❌ Demonstrating Error Handling');
        
        // Simulate error
        const error = new Error('This is a demo error');
        window.connectionManager.handleError('demo-component', error);
        
        // Clear error after 5 seconds
        setTimeout(() => {
            window.connectionManager.clearError('demo-component');
            console.log('✅ Error state cleared');
        }, 5000);
    }
    
    static demonstrateConnectionCallbacks() {
        console.log('🔗 Demonstrating Connection Callbacks');
        
        // Add connection callbacks
        window.connectionManager.addConnectionCallback('connected', () => {
            console.log('✅ Connected to server');
        });
        
        window.connectionManager.addConnectionCallback('disconnected', (reason) => {
            console.log('⚠️ Disconnected from server:', reason);
        });
        
        window.connectionManager.addConnectionCallback('reconnecting', () => {
            console.log('🔄 Reconnecting to server...');
        });
        
        window.connectionManager.addConnectionCallback('error', (error) => {
            console.error('❌ Connection error:', error);
        });
    }
    
    static getConnectionStats() {
        console.log('📊 Connection Manager Statistics');
        const stats = window.connectionManager.getStats();
        console.table(stats);
        return stats;
    }
}

// Integration helper functions
window.GoldGPTConnectionHelpers = {
    /**
     * Create a properly integrated component
     */
    createComponent(componentId, initFn, cleanupFn) {
        const component = {
            componentId,
            eventCleanup: [],
            
            async init() {
                await this.waitForConnectionManager();
                this.setupConnectionManagerIntegration();
                await initFn.call(this);
            },
            
            async waitForConnectionManager() {
                return new Promise((resolve) => {
                    const check = () => {
                        if (window.connectionManager) {
                            resolve();
                        } else {
                            setTimeout(check, 100);
                        }
                    };
                    check();
                });
            },
            
            setupConnectionManagerIntegration() {
                // Add data-component attribute
                const containers = document.querySelectorAll(`[data-component="${this.componentId}"]`);
                if (containers.length === 0) {
                    console.warn(`⚠️ No containers found for component ${this.componentId}`);
                }
                
                // Setup retry handler
                const retryCleanup = window.connectionManager.on('retry', (data) => {
                    if (data.componentId === this.componentId) {
                        console.log(`🔄 Retrying component ${this.componentId}...`);
                        this.init();
                    }
                });
                this.eventCleanup.push(retryCleanup);
            },
            
            cleanup() {
                this.eventCleanup.forEach(cleanup => cleanup());
                this.eventCleanup = [];
                
                if (window.connectionManager) {
                    window.connectionManager.offContext(this);
                }
                
                if (cleanupFn) {
                    cleanupFn.call(this);
                }
            }
        };
        
        return component;
    },
    
    /**
     * Wait for connection manager to be ready
     */
    async waitForConnectionManager() {
        return new Promise((resolve) => {
            const check = () => {
                if (window.connectionManager && window.connectionManager.isConnected) {
                    resolve();
                } else {
                    setTimeout(check, 100);
                }
            };
            check();
        });
    },
    
    /**
     * Safely make HTTP requests
     */
    async safeRequest(url, options = {}) {
        try {
            if (window.connectionManager) {
                return await window.connectionManager.request(url, options);
            } else {
                // Fallback to regular fetch
                const response = await fetch(url, options);
                return await response.json();
            }
        } catch (error) {
            console.error('Request failed:', error);
            throw error;
        }
    },
    
    /**
     * Safely emit WebSocket events
     */
    async safeEmit(eventName, data = {}) {
        try {
            if (window.connectionManager && window.connectionManager.isConnected) {
                return await window.connectionManager.send(eventName, data);
            } else {
                throw new Error('WebSocket not connected');
            }
        } catch (error) {
            console.error('WebSocket emit failed:', error);
            throw error;
        }
    }
};

// Auto-run examples when page loads
document.addEventListener('DOMContentLoaded', () => {
    // Wait for connection manager to be ready
    setTimeout(() => {
        if (window.connectionManager) {
            console.log('🚀 Running Connection Manager Integration Examples...');
            
            // Run examples
            ConnectionManagerExamples.demonstrateConnectionCallbacks();
            ConnectionManagerExamples.getConnectionStats();
            
            // Create example component
            const exampleComponent = new ExampleComponent();
            
            // Store reference for cleanup
            window.exampleComponent = exampleComponent;
        }
    }, 2000);
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (window.exampleComponent) {
        window.exampleComponent.cleanup();
    }
});

console.log('🔧 Connection Manager Integration Demo loaded');
