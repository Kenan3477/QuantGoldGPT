/**
 * Performance Optimization Module for Advanced ML Dashboard
 * ========================================================
 * 
 * Advanced caching, lazy loading, and performance monitoring features
 */

class DashboardPerformanceOptimizer {
    constructor(dashboard) {
        this.dashboard = dashboard;
        this.cache = new Map();
        this.performanceMetrics = {
            apiCalls: 0,
            cacheHits: 0,
            averageResponseTime: 0,
            errorRate: 0
        };
        this.observers = [];
        
        this.init();
    }

    init() {
        this.setupCaching();
        this.setupLazyLoading();
        this.setupPerformanceMonitoring();
        this.setupMemoryManagement();
    }

    setupCaching() {
        // API response caching with TTL
        this.apiCache = new Map();
        this.cacheTTL = new Map();
        
        // Cache configurations
        this.cacheConfig = {
            'predictions': 30000,      // 30 seconds
            'performance': 60000,      // 1 minute
            'market-analysis': 45000,  // 45 seconds
            'learning-data': 120000    // 2 minutes
        };
    }

    async cachedApiCall(endpoint, options = {}) {
        const cacheKey = `${endpoint}_${JSON.stringify(options)}`;
        const now = Date.now();
        
        // Check cache
        if (this.apiCache.has(cacheKey)) {
            const cacheTime = this.cacheTTL.get(cacheKey);
            const ttl = this.cacheConfig[endpoint.split('/').pop()] || 30000;
            
            if (now - cacheTime < ttl) {
                this.performanceMetrics.cacheHits++;
                return this.apiCache.get(cacheKey);
            } else {
                // Cache expired
                this.apiCache.delete(cacheKey);
                this.cacheTTL.delete(cacheKey);
            }
        }
        
        // Make API call
        const startTime = performance.now();
        this.performanceMetrics.apiCalls++;
        
        try {
            const response = await fetch(`/api/advanced-ml${endpoint}`, {
                method: options.method || 'GET',
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                },
                body: options.body ? JSON.stringify(options.body) : undefined
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            const data = await response.json();
            const endTime = performance.now();
            
            // Update performance metrics
            const responseTime = endTime - startTime;
            this.updateResponseTime(responseTime);
            
            // Cache successful response
            this.apiCache.set(cacheKey, data);
            this.cacheTTL.set(cacheKey, now);
            
            return data;
            
        } catch (error) {
            this.performanceMetrics.errorRate++;
            throw error;
        }
    }

    setupLazyLoading() {
        // Intersection Observer for lazy loading charts
        this.chartObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const chartId = entry.target.id;
                    this.loadChartData(chartId);
                    this.chartObserver.unobserve(entry.target);
                }
            });
        }, {
            rootMargin: '100px' // Load 100px before visible
        });

        // Observe chart containers
        document.querySelectorAll('.chart-container').forEach(container => {
            this.chartObserver.observe(container);
        });
    }

    setupPerformanceMonitoring() {
        // Monitor performance metrics
        this.performanceInterval = setInterval(() => {
            this.collectPerformanceMetrics();
            this.optimizeMemoryUsage();
        }, 30000); // Every 30 seconds

        // Monitor DOM mutations for optimization opportunities
        this.mutationObserver = new MutationObserver((mutations) => {
            this.optimizeDOMMutations(mutations);
        });

        this.mutationObserver.observe(document.body, {
            childList: true,
            subtree: true,
            attributes: false
        });
    }

    setupMemoryManagement() {
        // Clear old cache entries periodically
        setInterval(() => {
            this.cleanupCache();
        }, 300000); // Every 5 minutes

        // Monitor memory usage
        if ('memory' in performance) {
            setInterval(() => {
                this.monitorMemoryUsage();
            }, 60000); // Every minute
        }
    }

    async loadChartData(chartId) {
        const chartConfig = {
            'predictionChart': '/predictions',
            'accuracyChart': '/performance',
            'strategyChart': '/feature-importance',
            'featureChart': '/learning-data'
        };

        const endpoint = chartConfig[chartId];
        if (endpoint) {
            try {
                const data = await this.cachedApiCall(endpoint);
                this.dashboard.updateChart(chartId, data);
            } catch (error) {
                console.error(`Failed to load chart data for ${chartId}:`, error);
            }
        }
    }

    collectPerformanceMetrics() {
        // Collect browser performance metrics
        const perfEntries = performance.getEntriesByType('navigation');
        if (perfEntries.length > 0) {
            const navEntry = perfEntries[0];
            this.performanceMetrics.pageLoadTime = navEntry.loadEventEnd - navEntry.fetchStart;
            this.performanceMetrics.domContentLoaded = navEntry.domContentLoadedEventEnd - navEntry.fetchStart;
        }

        // Collect custom metrics
        this.performanceMetrics.cacheHitRate = 
            this.performanceMetrics.apiCalls > 0 
                ? (this.performanceMetrics.cacheHits / this.performanceMetrics.apiCalls * 100).toFixed(1)
                : 0;

        // Update performance display
        this.updatePerformanceDisplay();
    }

    updateResponseTime(responseTime) {
        const currentCount = this.performanceMetrics.responseTimeCount || 0;
        const currentTotal = this.performanceMetrics.totalResponseTime || 0;
        
        this.performanceMetrics.responseTimeCount = currentCount + 1;
        this.performanceMetrics.totalResponseTime = currentTotal + responseTime;
        this.performanceMetrics.averageResponseTime = 
            this.performanceMetrics.totalResponseTime / this.performanceMetrics.responseTimeCount;
    }

    updatePerformanceDisplay() {
        // Update performance indicators in the UI
        const performanceContainer = document.getElementById('performanceMetrics');
        if (performanceContainer) {
            performanceContainer.innerHTML = `
                <div class="metric-item">
                    <span class="metric-label">API Calls</span>
                    <span class="metric-value">${this.performanceMetrics.apiCalls}</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Cache Hit Rate</span>
                    <span class="metric-value">${this.performanceMetrics.cacheHitRate}%</span>
                </div>
                <div class="metric-item">
                    <span class="metric-label">Avg Response</span>
                    <span class="metric-value">${this.performanceMetrics.averageResponseTime.toFixed(0)}ms</span>
                </div>
            `;
        }
    }

    optimizeDOMMutations(mutations) {
        // Batch DOM updates to avoid layout thrashing
        const updates = [];
        
        mutations.forEach(mutation => {
            if (mutation.type === 'childList') {
                mutation.addedNodes.forEach(node => {
                    if (node.nodeType === Node.ELEMENT_NODE) {
                        updates.push(() => this.optimizeElement(node));
                    }
                });
            }
        });

        // Execute updates in batches
        if (updates.length > 0) {
            requestAnimationFrame(() => {
                updates.forEach(update => update());
            });
        }
    }

    optimizeElement(element) {
        // Add performance optimizations to dynamically added elements
        if (element.classList.contains('chart-container')) {
            this.chartObserver.observe(element);
        }

        // Lazy load images
        const images = element.querySelectorAll('img[data-src]');
        images.forEach(img => this.lazyLoadImage(img));
    }

    lazyLoadImage(img) {
        const imageObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const lazyImg = entry.target;
                    lazyImg.src = lazyImg.dataset.src;
                    lazyImg.classList.remove('lazy');
                    imageObserver.unobserve(lazyImg);
                }
            });
        });

        imageObserver.observe(img);
    }

    cleanupCache() {
        const now = Date.now();
        
        // Clean API cache
        for (const [key, timestamp] of this.cacheTTL.entries()) {
            const endpoint = key.split('_')[0].split('/').pop();
            const ttl = this.cacheConfig[endpoint] || 30000;
            
            if (now - timestamp > ttl * 2) { // Double TTL for cleanup
                this.apiCache.delete(key);
                this.cacheTTL.delete(key);
            }
        }

        console.log(`🧹 Cache cleanup: ${this.apiCache.size} entries remaining`);
    }

    monitorMemoryUsage() {
        if ('memory' in performance) {
            const memInfo = performance.memory;
            const memUsage = {
                used: Math.round(memInfo.usedJSHeapSize / 1048576), // MB
                total: Math.round(memInfo.totalJSHeapSize / 1048576), // MB
                limit: Math.round(memInfo.jsHeapSizeLimit / 1048576) // MB
            };

            console.log(`💾 Memory usage: ${memUsage.used}MB / ${memUsage.limit}MB`);

            // Trigger cleanup if memory usage is high
            if (memUsage.used / memUsage.limit > 0.8) {
                this.emergencyCleanup();
            }
        }
    }

    emergencyCleanup() {
        console.warn('🚨 High memory usage detected, performing emergency cleanup');
        
        // Clear all caches
        this.apiCache.clear();
        this.cacheTTL.clear();
        
        // Clear old chart data
        Object.values(this.dashboard.charts).forEach(chart => {
            if (chart && chart.data && chart.data.datasets) {
                chart.data.datasets.forEach(dataset => {
                    if (dataset.data && dataset.data.length > 100) {
                        dataset.data = dataset.data.slice(-50); // Keep only last 50 points
                    }
                });
                chart.update();
            }
        });
        
        // Force garbage collection if available
        if (window.gc) {
            window.gc();
        }
    }

    // Public API methods
    preloadCriticalData() {
        const criticalEndpoints = ['/predictions', '/performance'];
        return Promise.all(
            criticalEndpoints.map(endpoint => this.cachedApiCall(endpoint))
        );
    }

    invalidateCache(pattern) {
        for (const key of this.apiCache.keys()) {
            if (key.includes(pattern)) {
                this.apiCache.delete(key);
                this.cacheTTL.delete(key);
            }
        }
    }

    getPerformanceReport() {
        return {
            ...this.performanceMetrics,
            cacheSize: this.apiCache.size,
            memoryUsage: 'memory' in performance ? {
                used: Math.round(performance.memory.usedJSHeapSize / 1048576),
                total: Math.round(performance.memory.totalJSHeapSize / 1048576)
            } : null
        };
    }

    destroy() {
        // Cleanup observers and intervals
        if (this.chartObserver) this.chartObserver.disconnect();
        if (this.mutationObserver) this.mutationObserver.disconnect();
        if (this.performanceInterval) clearInterval(this.performanceInterval);
        
        // Clear caches
        this.apiCache.clear();
        this.cacheTTL.clear();
    }
}

// Export for use with the main dashboard
if (typeof window !== 'undefined') {
    window.DashboardPerformanceOptimizer = DashboardPerformanceOptimizer;
}
