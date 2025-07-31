/**
 * GoldGPT Sequential Component Loader
 * Ensures proper initialization order and eliminates race conditions
 * Follows Trading 212's design principles for smooth startup experience
 */

class ComponentLoader {
    constructor() {
        this.components = new Map();
        this.loadedComponents = new Set();
        this.failedComponents = new Set();
        this.loadingQueue = [];
        this.isLoading = false;
        this.startTime = Date.now();
        
        // Component states
        this.STATES = {
            PENDING: 'pending',
            LOADING: 'loading',
            LOADED: 'loaded',
            FAILED: 'failed',
            SKIPPED: 'skipped'
        };
        
        // Priority levels
        this.PRIORITIES = {
            CRITICAL: 1,    // Essential for basic functionality
            HIGH: 2,        // Important features
            MEDIUM: 3,      // Enhanced features
            LOW: 4          // Optional features
        };
        
        this.initializeUI();
        console.log('🔧 Component Loader initialized');
    }

    /**
     * Register a component with its dependencies and configuration
     */
    register(name, config) {
        const component = {
            name,
            dependencies: config.dependencies || [],
            priority: config.priority || this.PRIORITIES.MEDIUM,
            loader: config.loader,
            timeout: config.timeout || 10000,
            retries: config.retries || 2,
            critical: config.critical || false,
            state: this.STATES.PENDING,
            error: null,
            loadTime: null,
            retryCount: 0
        };
        
        this.components.set(name, component);
        console.log(`📦 Registered component: ${name} (Priority: ${component.priority})`);
        return this;
    }

    /**
     * Start the sequential loading process
     */
    async startLoading() {
        if (this.isLoading) {
            console.warn('⚠️ Loading already in progress');
            return;
        }

        this.isLoading = true;
        console.log('🚀 Starting sequential component loading...');
        
        try {
            this.showLoadingUI();
            this.buildLoadingQueue();
            await this.processLoadingQueue();
            this.completeLoading();
        } catch (error) {
            console.error('❌ Critical error during component loading:', error);
            this.handleCriticalError(error);
        }
    }

    /**
     * Build the loading queue based on dependencies and priorities
     */
    buildLoadingQueue() {
        const queue = [];
        const processed = new Set();
        const visiting = new Set();

        // Topological sort with priority consideration
        const visit = (componentName) => {
            if (visiting.has(componentName)) {
                throw new Error(`Circular dependency detected: ${componentName}`);
            }
            if (processed.has(componentName)) {
                return;
            }

            const component = this.components.get(componentName);
            if (!component) {
                throw new Error(`Unknown component: ${componentName}`);
            }

            visiting.add(componentName);

            // Process dependencies first
            for (const dep of component.dependencies) {
                visit(dep);
            }

            visiting.delete(componentName);
            processed.add(componentName);
            queue.push(component);
        };

        // Visit all components
        for (const componentName of this.components.keys()) {
            if (!processed.has(componentName)) {
                visit(componentName);
            }
        }

        // Sort by priority within dependency constraints
        this.loadingQueue = queue.sort((a, b) => a.priority - b.priority);
        
        console.log('📋 Loading queue built:', this.loadingQueue.map(c => `${c.name}(P${c.priority})`));
    }

    /**
     * Process the loading queue sequentially
     */
    async processLoadingQueue() {
        const totalComponents = this.loadingQueue.length;
        let loadedCount = 0;

        for (const component of this.loadingQueue) {
            try {
                this.updateLoadingProgress(component.name, loadedCount, totalComponents);
                await this.loadComponent(component);
                loadedCount++;
            } catch (error) {
                console.error(`❌ Failed to load component ${component.name}:`, error);
                
                if (component.critical) {
                    throw new Error(`Critical component failed: ${component.name}`);
                }
                
                // Continue with non-critical components
                component.state = this.STATES.FAILED;
                component.error = error.message;
                this.failedComponents.add(component.name);
            }
        }
    }

    /**
     * Load a single component with timeout and retry logic
     */
    async loadComponent(component) {
        component.state = this.STATES.LOADING;
        const startTime = Date.now();

        while (component.retryCount <= component.retries) {
            try {
                console.log(`🔄 Loading ${component.name} (attempt ${component.retryCount + 1})`);
                
                await Promise.race([
                    component.loader(),
                    this.createTimeout(component.timeout, component.name)
                ]);

                component.state = this.STATES.LOADED;
                component.loadTime = Date.now() - startTime;
                this.loadedComponents.add(component.name);
                
                console.log(`✅ ${component.name} loaded successfully in ${component.loadTime}ms`);
                return;

            } catch (error) {
                component.retryCount++;
                
                if (component.retryCount <= component.retries) {
                    console.warn(`⚠️ ${component.name} failed, retrying... (${component.retryCount}/${component.retries})`);
                    await this.delay(1000 * component.retryCount); // Exponential backoff
                } else {
                    throw error;
                }
            }
        }
    }

    /**
     * Create a timeout promise
     */
    createTimeout(ms, componentName) {
        return new Promise((_, reject) => {
            setTimeout(() => reject(new Error(`Timeout loading ${componentName}`)), ms);
        });
    }

    /**
     * Delay utility
     */
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    /**
     * Initialize the loading UI
     */
    initializeUI() {
        // Create loading overlay
        const overlay = document.createElement('div');
        overlay.id = 'component-loader-overlay';
        overlay.innerHTML = `
            <div class="loader-container">
                <div class="loader-header">
                    <div class="loader-logo">
                        <i class="fas fa-coins"></i>
                        <span>GoldGPT</span>
                    </div>
                    <div class="loader-subtitle">Advanced AI Trading Platform</div>
                </div>
                
                <div class="loader-progress">
                    <div class="progress-bar">
                        <div class="progress-fill" id="progress-fill"></div>
                    </div>
                    <div class="progress-text" id="progress-text">Initializing...</div>
                </div>
                
                <div class="loader-status">
                    <div class="status-item" id="current-component">
                        <i class="fas fa-cog fa-spin"></i>
                        <span>Preparing components...</span>
                    </div>
                </div>
                
                <div class="loader-details" id="loader-details">
                    <div class="detail-list" id="component-list"></div>
                </div>
            </div>
        `;
        
        document.body.appendChild(overlay);
    }

    /**
     * Show the loading UI
     */
    showLoadingUI() {
        const overlay = document.getElementById('component-loader-overlay');
        if (overlay) {
            overlay.style.display = 'flex';
            document.body.style.overflow = 'hidden';
        }
    }

    /**
     * Update loading progress
     */
    updateLoadingProgress(componentName, loaded, total) {
        const progressPercent = Math.round((loaded / total) * 100);
        
        // Update progress bar
        const progressFill = document.getElementById('progress-fill');
        const progressText = document.getElementById('progress-text');
        const currentComponent = document.getElementById('current-component');
        
        if (progressFill) {
            progressFill.style.width = `${progressPercent}%`;
        }
        
        if (progressText) {
            progressText.textContent = `Loading ${componentName}... (${loaded}/${total})`;
        }
        
        if (currentComponent) {
            currentComponent.innerHTML = `
                <i class="fas fa-cog fa-spin"></i>
                <span>Loading ${componentName}...</span>
            `;
        }
        
        // Update component list
        this.updateComponentList();
    }

    /**
     * Update the component status list
     */
    updateComponentList() {
        const componentList = document.getElementById('component-list');
        if (!componentList) return;
        
        componentList.innerHTML = '';
        
        for (const [name, component] of this.components) {
            const item = document.createElement('div');
            item.className = 'component-status-item';
            
            let icon, status;
            switch (component.state) {
                case this.STATES.LOADED:
                    icon = 'fas fa-check-circle';
                    status = 'loaded';
                    break;
                case this.STATES.LOADING:
                    icon = 'fas fa-spinner fa-spin';
                    status = 'loading';
                    break;
                case this.STATES.FAILED:
                    icon = 'fas fa-exclamation-circle';
                    status = 'failed';
                    break;
                default:
                    icon = 'fas fa-clock';
                    status = 'pending';
            }
            
            item.innerHTML = `
                <i class="${icon}"></i>
                <span>${name}</span>
                <small class="status-${status}">${component.state}</small>
            `;
            
            componentList.appendChild(item);
        }
    }

    /**
     * Complete the loading process
     */
    completeLoading() {
        const totalTime = Date.now() - this.startTime;
        console.log(`🎉 Component loading completed in ${totalTime}ms`);
        console.log(`✅ Loaded: ${this.loadedComponents.size}`);
        console.log(`❌ Failed: ${this.failedComponents.size}`);
        
        // Update UI for completion
        setTimeout(() => {
            this.hideLoadingUI();
            this.showCompletionMessage();
        }, 500);
        
        this.isLoading = false;
        
        // Emit completion event
        window.dispatchEvent(new CustomEvent('componentsLoaded', {
            detail: {
                loaded: Array.from(this.loadedComponents),
                failed: Array.from(this.failedComponents),
                totalTime
            }
        }));
    }

    /**
     * Hide the loading UI
     */
    hideLoadingUI() {
        const overlay = document.getElementById('component-loader-overlay');
        if (overlay) {
            overlay.style.opacity = '0';
            setTimeout(() => {
                overlay.style.display = 'none';
                document.body.style.overflow = '';
            }, 300);
        }
    }

    /**
     * Show completion message
     */
    showCompletionMessage() {
        if (this.failedComponents.size === 0) {
            this.showNotification('🎉 GoldGPT loaded successfully!', 'success');
        } else {
            this.showNotification(
                `⚠️ GoldGPT loaded with ${this.failedComponents.size} component(s) failed`,
                'warning'
            );
        }
    }

    /**
     * Handle critical errors
     */
    handleCriticalError(error) {
        console.error('🚨 Critical loading error:', error);
        
        const overlay = document.getElementById('component-loader-overlay');
        if (overlay) {
            overlay.innerHTML = `
                <div class="loader-container error">
                    <div class="error-icon">
                        <i class="fas fa-exclamation-triangle"></i>
                    </div>
                    <h2>Loading Failed</h2>
                    <p>A critical component failed to load:</p>
                    <code>${error.message}</code>
                    <button onclick="location.reload()" class="retry-btn">
                        <i class="fas fa-refresh"></i>
                        Retry
                    </button>
                </div>
            `;
        }
    }

    /**
     * Utility method for notifications
     */
    showNotification(message, type = 'info') {
        // This will be implemented by the main app
        if (window.showNotification) {
            window.showNotification(message, type);
        } else {
            console.log(`[${type.toUpperCase()}] ${message}`);
        }
    }

    /**
     * Get component status
     */
    getComponentStatus(name) {
        const component = this.components.get(name);
        return component ? component.state : null;
    }

    /**
     * Check if all critical components are loaded
     */
    areCriticalComponentsLoaded() {
        for (const [name, component] of this.components) {
            if (component.critical && component.state !== this.STATES.LOADED) {
                return false;
            }
        }
        return true;
    }

    /**
     * Get loading statistics
     */
    getStats() {
        return {
            total: this.components.size,
            loaded: this.loadedComponents.size,
            failed: this.failedComponents.size,
            pending: this.components.size - this.loadedComponents.size - this.failedComponents.size,
            totalTime: Date.now() - this.startTime
        };
    }
}

// Export for global use
window.ComponentLoader = ComponentLoader;
