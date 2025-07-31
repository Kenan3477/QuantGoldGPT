/**
 * ====================================================================
 *                      UNIFIED NEWS MANAGER
 * ====================================================================
 * 
 * Consolidates multiple competing news loading functions into a single,
 * reliable system with proper fallback logic and race condition prevention.
 * 
 * Replaces:
 * - loadNewsDirectly() function from dashboard template
 * - forceInitializeNews() function from dashboard template  
 * - EnhancedNewsManager class functionality
 * 
 * Features:
 * - Single initialization point to prevent race conditions
 * - Proper fallback hierarchy for reliability
 * - Memory leak prevention with cleanup methods
 * - Trading 212-inspired modular architecture
 * - Event listener management for proper cleanup
 */

class UnifiedNewsManager {
    constructor() {
        // Core properties
        this.newsContainer = null;
        this.isInitialized = false;
        this.isLoading = false;
        this.initializationAttempts = 0;
        this.maxInitializationAttempts = 3;
        
        // State management
        this.lastUpdate = null;
        this.refreshInterval = null;
        this.eventCleanup = [];
        this.componentId = 'unified-news-manager';
        
        // Configuration
        this.config = {
            apiEndpoint: '/api/news/enhanced',
            refreshIntervalMs: 5 * 60 * 1000, // 5 minutes
            maxArticles: 20,
            retryDelay: 2000,
            fallbackDelay: 1000
        };
        
        // Initialization with proper timing
        this.initialize();
    }
    
    /**
     * Master initialization method with fallback timing
     * Prevents race conditions with DOM loading and other managers
     */
    initialize() {
        console.log(`🚀 ${this.componentId}: Starting unified news manager initialization...`);
        
        // Check if already initialized to prevent duplicates
        if (this.isInitialized) {
            console.log(`⚠️ ${this.componentId}: Already initialized, skipping...`);
            return;
        }
        
        // Handle different DOM states
        if (document.readyState === 'loading') {
            // DOM still loading - wait for DOMContentLoaded
            this.addEventListenerWithCleanup(document, 'DOMContentLoaded', () => {
                this.attemptInitialization();
            });
        } else if (document.readyState === 'interactive') {
            // DOM loaded but resources may still be loading - slight delay
            setTimeout(() => this.attemptInitialization(), 100);
        } else {
            // DOM and resources fully loaded - immediate initialization
            this.attemptInitialization();
        }
    }
    
    /**
     * Attempts initialization with retry logic
     */
    async attemptInitialization() {
        try {
            this.initializationAttempts++;
            console.log(`🔍 ${this.componentId}: Initialization attempt ${this.initializationAttempts}/${this.maxInitializationAttempts}`);
            
            // Find news container with fallback selectors
            if (!this.findNewsContainer()) {
                if (this.initializationAttempts < this.maxInitializationAttempts) {
                    console.log(`⏳ ${this.componentId}: Container not found, retrying in ${this.config.retryDelay}ms...`);
                    setTimeout(() => this.attemptInitialization(), this.config.retryDelay);
                    return;
                } else {
                    console.error(`❌ ${this.componentId}: Container not found after ${this.maxInitializationAttempts} attempts`);
                    return;
                }
            }
            
            // Setup news container and load content
            this.setupNewsContainer();
            await this.loadNews();
            this.setupAutoRefresh();
            this.isInitialized = true;
            
            console.log(`✅ ${this.componentId}: Successfully initialized after ${this.initializationAttempts} attempts`);
            
        } catch (error) {
            console.error(`❌ ${this.componentId}: Initialization error:`, error);
            
            if (this.initializationAttempts < this.maxInitializationAttempts) {
                setTimeout(() => this.attemptInitialization(), this.config.retryDelay);
            }
        }
    }
    
    /**
     * Finds news container using multiple fallback selectors
     * Handles various container naming conventions
     */
    findNewsContainer() {
        // Primary selectors (in order of preference)
        const selectors = [
            '#enhanced-news-container',
            '#news-container', 
            '.enhanced-news-container',
            '.news-container',
            '.right-column .news-section',
            '.right-panel .news-section',
            '#news-section',
            '.news-panel',
            '[data-component=\"news\"]'
        ];
        
        for (const selector of selectors) {
            const container = document.querySelector(selector);
            if (container) {
                this.newsContainer = container;
                console.log(`📍 ${this.componentId}: Container found using selector: ${selector}`);
                return true;
            }
        }
        
        // Last resort: create container if parent exists
        const parentSelectors = ['.right-column', '.right-panel', '.sidebar', '.news-section'];
        for (const parentSelector of parentSelectors) {
            const parent = document.querySelector(parentSelector);
            if (parent) {
                this.newsContainer = this.createNewsContainer(parent);
                console.log(`📍 ${this.componentId}: Container created in parent: ${parentSelector}`);
                return true;
            }
        }
        
        console.error(`❌ ${this.componentId}: No suitable container or parent found`);
        console.log('Available elements:', document.querySelectorAll('[id*=\"news\"], [class*=\"news\"], [class*=\"panel\"]'));
        return false;
    }
    
    /**
     * Creates a news container if none exists
     */
    createNewsContainer(parent) {
        const container = document.createElement('div');
        container.id = 'enhanced-news-container';
        container.className = 'enhanced-news-container unified-news-container';
        parent.appendChild(container);
        return container;
    }
    
    /**
     * Sets up the news container with proper structure and controls
     */
    setupNewsContainer() {
        // Ensure container has proper CSS classes
        if (!this.newsContainer.classList.contains('unified-news-container')) {
            this.newsContainer.classList.add('unified-news-container');
        }
        
        // Check if we need to add header controls
        let header = this.newsContainer.querySelector('.news-header');
        if (!header) {
            header = document.createElement('div');
            header.className = 'news-header';
            header.innerHTML = this.getHeaderHTML();
            this.newsContainer.insertBefore(header, this.newsContainer.firstChild);
        }
        
        // Add content container if not present
        let contentContainer = this.newsContainer.querySelector('.news-content');
        if (!contentContainer) {
            contentContainer = document.createElement('div');
            contentContainer.className = 'news-content';
            this.newsContainer.appendChild(contentContainer);
        }
    }
    
    /**
     * Gets the header HTML with refresh controls
     */
    getHeaderHTML() {
        return `
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 15px; border-bottom: 1px solid #333; padding-bottom: 10px;">
                <h3 style="margin: 0; color: #fff; font-size: 16px;">
                    <i class="fas fa-newspaper" style="color: #0088ff; margin-right: 8px;"></i>
                    Market News & Analysis
                </h3>
                <div class="news-controls" style="display: flex; align-items: center; gap: 10px;">
                    <button class="unified-refresh-btn" style="background: #0066cc; color: white; border: none; border-radius: 4px; padding: 6px 12px; cursor: pointer; font-size: 12px; display: flex; align-items: center; gap: 5px;">
                        <i class="fas fa-sync-alt"></i> Refresh
                    </button>
                    <span class="last-update-indicator" style="color: #888; font-size: 11px;">Loading...</span>
                </div>
            </div>
        `;
    }
    
    /**
     * Main news loading method with comprehensive fallback logic
     */
    async loadNews() {
        // Prevent concurrent loading
        if (this.isLoading) {
            console.log(`⏳ ${this.componentId}: Already loading, skipping request`);
            return;
        }
        
        this.isLoading = true;
        console.log(`🔄 ${this.componentId}: Loading news...`);
        
        try {
            // Show loading state
            this.showLoadingState();
            
            // Primary loading method: Enhanced API
            const newsData = await this.loadFromEnhancedAPI();
            
            if (newsData && newsData.success && newsData.articles && newsData.articles.length > 0) {
                console.log(`✅ ${this.componentId}: Loaded ${newsData.articles.length} articles from enhanced API`);
                this.displayNews(newsData.articles);
                this.updateLastRefreshTime();
                return;
            }
            
            // Fallback 1: Try EnhancedNewsManager if available
            if (await this.tryEnhancedNewsManager()) {
                console.log(`✅ ${this.componentId}: Fallback to EnhancedNewsManager successful`);
                return;
            }
            
            // Fallback 2: Basic API endpoint
            const basicNewsData = await this.loadFromBasicAPI();
            if (basicNewsData && basicNewsData.articles && basicNewsData.articles.length > 0) {
                console.log(`✅ ${this.componentId}: Loaded ${basicNewsData.articles.length} articles from basic API`);
                this.displayNews(basicNewsData.articles);
                this.updateLastRefreshTime();
                return;
            }
            
            // Fallback 3: Static fallback content
            console.log(`⚠️ ${this.componentId}: All API methods failed, showing fallback content`);
            this.showFallbackContent();
            
        } catch (error) {
            console.error(`❌ ${this.componentId}: Error loading news:`, error);
            this.showErrorState(error.message);
        } finally {
            this.isLoading = false;
        }
    }
    
    /**
     * Loads news from the enhanced API endpoint
     */
    async loadFromEnhancedAPI() {
        try {
            const response = await fetch(`${this.config.apiEndpoint}?limit=${this.config.maxArticles}`, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                },
                timeout: 10000 // 10 second timeout
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error(`❌ ${this.componentId}: Enhanced API failed:`, error);
            return null;
        }
    }
    
    /**
     * Tries to use the existing EnhancedNewsManager if available
     */
    async tryEnhancedNewsManager() {
        try {
            // Check if EnhancedNewsManager class exists and can be instantiated
            if (typeof EnhancedNewsManager !== 'undefined') {
                console.log(`🔄 ${this.componentId}: Attempting EnhancedNewsManager fallback...`);
                
                // Create temporary instance to get data
                const tempManager = new EnhancedNewsManager();
                if (tempManager && typeof tempManager.loadEnhancedNews === 'function') {
                    await tempManager.loadEnhancedNews();
                    return true;
                }
            }
            return false;
        } catch (error) {
            console.error(`❌ ${this.componentId}: EnhancedNewsManager fallback failed:`, error);
            return false;
        }
    }
    
    /**
     * Loads news from basic API endpoint (fallback)
     */
    async loadFromBasicAPI() {
        try {
            const response = await fetch('/api/news?limit=10');
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            console.error(`❌ ${this.componentId}: Basic API failed:`, error);
            return null;
        }
    }
    
    /**
     * Displays news articles in the container
     */
    displayNews(articles) {
        const contentContainer = this.newsContainer.querySelector('.news-content');
        if (!contentContainer) {
            console.error(`❌ ${this.componentId}: Content container not found`);
            return;
        }
        
        const articlesHTML = articles.map((article, index) => this.getArticleHTML(article, index)).join('');
        
        contentContainer.innerHTML = `
            <div style="max-height: 450px; overflow-y: auto; padding-right: 5px;">
                ${articlesHTML}
            </div>
        `;
        
        console.log(`📰 ${this.componentId}: Displayed ${articles.length} articles`);
    }
    
    /**
     * Generates HTML for a single article
     */
    getArticleHTML(article, index) {
        const sentiment = (article.sentiment_label || 'neutral').toLowerCase();
        const sentimentColor = this.getSentimentColor(sentiment);
        const confidence = Math.round((article.confidence_score || 0.5) * 100);
        const confidenceColor = confidence >= 70 ? '#00ff88' : confidence >= 40 ? '#ffaa00' : '#ff4444';
        
        return `
            <div class="news-article" style="background: #222; border-radius: 6px; padding: 12px; margin-bottom: 12px; border-left: 3px solid ${sentimentColor}; transition: background-color 0.2s ease;">
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                    <span style="background: ${sentimentColor}; color: white; padding: 3px 8px; border-radius: 12px; font-size: 10px; font-weight: bold;">
                        ${sentiment.toUpperCase()}
                    </span>
                    <span style="color: #888; font-size: 11px;">${article.source || 'Market News'}</span>
                    <span style="color: #666; font-size: 10px;">${article.time_ago || 'Recent'}</span>
                </div>
                <div style="color: #fff; font-size: 13px; font-weight: 500; margin-bottom: 6px; line-height: 1.3;">
                    ${article.title || 'Market Update'}
                </div>
                <div style="color: #aaa; font-size: 11px; line-height: 1.4; margin-bottom: 8px;">
                    ${(article.content || '').substring(0, 150)}${(article.content || '').length > 150 ? '...' : ''}
                </div>
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="color: #888; font-size: 9px;">Confidence:</span>
                        <div style="width: 60px; height: 4px; background: #333; border-radius: 2px; overflow: hidden;">
                            <div style="height: 100%; width: ${confidence}%; background: ${confidenceColor}; border-radius: 2px; transition: width 0.3s ease;"></div>
                        </div>
                        <span style="color: #ccc; font-size: 9px;">${confidence}%</span>
                    </div>
                    ${article.url ? `<a href="${article.url}" target="_blank" style="color: #0088ff; font-size: 10px; text-decoration: none;">Read More</a>` : ''}
                </div>
            </div>
        `;
    }
    
    /**
     * Gets the color for sentiment indicators
     */
    getSentimentColor(sentiment) {
        switch (sentiment) {
            case 'bullish': return '#00ff88';
            case 'bearish': return '#ff4444';
            case 'positive': return '#00ff88';
            case 'negative': return '#ff4444';
            case 'neutral': return '#ffaa00';
            default: return '#666';
        }
    }
    
    /**
     * Shows loading state in the news container
     */
    showLoadingState() {
        const contentContainer = this.newsContainer.querySelector('.news-content') || this.newsContainer;
        contentContainer.innerHTML = `
            <div style="text-align: center; padding: 40px; color: #fff;">
                <i class="fas fa-spinner fa-spin" style="font-size: 24px; color: #0088ff; margin-bottom: 12px; display: block;"></i>
                <span style="font-size: 14px;">Loading market news...</span>
            </div>
        `;
    }
    
    /**
     * Shows error state with retry option
     */
    showErrorState(errorMessage) {
        const contentContainer = this.newsContainer.querySelector('.news-content') || this.newsContainer;
        contentContainer.innerHTML = `
            <div style="text-align: center; padding: 40px; color: #ff4444;">
                <i class="fas fa-exclamation-triangle" style="font-size: 24px; margin-bottom: 12px; display: block;"></i>
                <div style="margin-bottom: 15px;">Error loading news</div>
                <div style="font-size: 12px; color: #888; margin-bottom: 15px;">${errorMessage}</div>
                <button onclick="unifiedNewsManager.loadNews()" style="background: #0066cc; color: white; border: none; border-radius: 4px; padding: 8px 16px; cursor: pointer; font-size: 12px;">
                    <i class="fas fa-retry"></i> Try Again
                </button>
            </div>
        `;
    }
    
    /**
     * Shows fallback content when all loading methods fail
     */
    showFallbackContent() {
        const contentContainer = this.newsContainer.querySelector('.news-content') || this.newsContainer;
        contentContainer.innerHTML = `
            <div style="padding: 20px; color: #fff; text-align: center;">
                <i class="fas fa-newspaper" style="font-size: 24px; color: #888; margin-bottom: 12px; display: block;"></i>
                <div style="margin-bottom: 10px; color: #ccc;">News service temporarily unavailable</div>
                <div style="font-size: 12px; color: #888; margin-bottom: 15px;">Please check your connection and try again</div>
                <button onclick="unifiedNewsManager.loadNews()" style="background: #0066cc; color: white; border: none; border-radius: 4px; padding: 8px 16px; cursor: pointer; font-size: 12px;">
                    <i class="fas fa-sync-alt"></i> Retry
                </button>
            </div>
        `;
    }
    
    /**
     * Sets up auto-refresh functionality
     */
    setupAutoRefresh() {
        // Clear existing interval
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
        }
        
        // Setup refresh button click handler
        const refreshBtn = this.newsContainer.querySelector('.unified-refresh-btn');
        if (refreshBtn) {
            this.addEventListenerWithCleanup(refreshBtn, 'click', (e) => {
                e.preventDefault();
                this.loadNews();
            });
        }
        
        // Setup auto-refresh interval
        this.refreshInterval = setInterval(() => {
            if (!this.isLoading) {
                console.log(`🔄 ${this.componentId}: Auto-refreshing news...`);
                this.loadNews();
            }
        }, this.config.refreshIntervalMs);
        
        console.log(`⏰ ${this.componentId}: Auto-refresh set to ${this.config.refreshIntervalMs / 1000}s intervals`);
    }
    
    /**
     * Updates the last refresh time indicator
     */
    updateLastRefreshTime() {
        this.lastUpdate = new Date();
        const timeString = this.lastUpdate.toLocaleTimeString();
        
        const indicator = this.newsContainer.querySelector('.last-update-indicator');
        if (indicator) {
            indicator.textContent = `Updated: ${timeString}`;
        }
    }
    
    /**
     * Adds event listener with automatic cleanup tracking
     */
    addEventListenerWithCleanup(element, event, handler) {
        element.addEventListener(event, handler);
        this.eventCleanup.push(() => element.removeEventListener(event, handler));
    }
    
    /**
     * Cleanup method to prevent memory leaks
     * Call this when destroying the component
     */
    cleanup() {
        console.log(`🧹 ${this.componentId}: Cleaning up...`);
        
        // Clear refresh interval
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
            this.refreshInterval = null;
        }
        
        // Remove all event listeners
        this.eventCleanup.forEach(cleanup => {
            try {
                cleanup();
            } catch (error) {
                console.error(`❌ ${this.componentId}: Error during event cleanup:`, error);
            }
        });
        this.eventCleanup = [];
        
        // Reset state
        this.isInitialized = false;
        this.isLoading = false;
        this.newsContainer = null;
        
        console.log(`✅ ${this.componentId}: Cleanup completed`);
    }
    
    /**
     * Static method to check if unified news manager should be used
     * This helps prevent conflicts with existing news managers
     */
    static shouldInitialize() {
        // Check if already initialized
        if (window.unifiedNewsManager && window.unifiedNewsManager.isInitialized) {
            return false;
        }
        
        // Check for news containers
        const containers = document.querySelectorAll('#enhanced-news-container, .news-container, [data-component=\"news\"]');
        return containers.length > 0;
    }
    
    /**
     * Static factory method for safe initialization
     */
    static initialize() {
        if (UnifiedNewsManager.shouldInitialize()) {
            console.log('🚀 Initializing Unified News Manager...');
            window.unifiedNewsManager = new UnifiedNewsManager();
            return window.unifiedNewsManager;
        } else {
            console.log('⏭️ Unified News Manager initialization skipped (not needed or already exists)');
            return null;
        }
    }
}

// Auto-initialize when script loads (can be overridden)
if (typeof window !== 'undefined') {
    // Make available globally
    window.UnifiedNewsManager = UnifiedNewsManager;
    
    // Initialize on DOM ready or immediately if DOM is already loaded
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => {
            UnifiedNewsManager.initialize();
        });
    } else {
        // DOM already loaded, initialize with slight delay to allow other scripts
        setTimeout(() => {
            UnifiedNewsManager.initialize();
        }, 100);
    }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = UnifiedNewsManager;
}
