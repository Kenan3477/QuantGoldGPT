/**
 * Enhanced News Display Manager
 * Handles fetching and displaying news with sentiment analysis and price correlation
 */

class EnhancedNewsManager {
    constructor() {
        this.newsContainer = null;
        this.isLoading = false;
        this.refreshInterval = null;
        this.lastUpdate = null;
        this.componentId = 'enhanced-news';
        this.eventCleanup = [];
        
        this.init();
    }
    
    init() {
        // Wait for DOM to load
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.initializeNewsDisplay());
        } else {
            this.initializeNewsDisplay();
        }
    }
    
    initializeNewsDisplay() {
        console.log('🔍 Enhanced News Manager: Initializing...');
        
        // Find or create news container
        this.newsContainer = document.getElementById('enhanced-news-container');
        
        if (!this.newsContainer) {
            // Try alternative selectors
            this.newsContainer = document.querySelector('.enhanced-news-container, .news-container, .right-column, #news-section');
            console.log('📍 Enhanced News: Container found via alternative selector:', this.newsContainer);
        } else {
            console.log('📍 Enhanced News: Container found by ID:', this.newsContainer);
        }
        
        if (!this.newsContainer) {
            console.error('❌ Enhanced News: Container not found! Available containers:', 
                document.querySelectorAll('[id*="news"], [class*="news"], [class*="panel"]'));
            return;
        }
        
        console.log('✅ Enhanced News: Container found, setting up...');
        
        // Initialize display
        this.setupNewsContainer();
        this.setupConnectionManagerIntegration();
        this.loadEnhancedNews();
        
        // Set up auto-refresh every 5 minutes
        this.refreshInterval = setInterval(() => {
            this.loadEnhancedNews();
        }, 5 * 60 * 1000);
        
        console.log('✅ Enhanced News Manager initialized successfully');
    }
    
    setupNewsContainer() {
        // Add CSS classes if not present
        if (!this.newsContainer.classList.contains('enhanced-news-container')) {
            this.newsContainer.classList.add('enhanced-news-container');
        }
        
        // Add header if not present
        let header = this.newsContainer.querySelector('.news-header');
        if (!header) {
            header = document.createElement('div');
            header.className = 'news-header';
            header.innerHTML = `
                <h3>Market News & Analysis</h3>
                <div class="news-controls">
                    <button class="refresh-btn" onclick="enhancedNewsManager.loadEnhancedNews()">
                        <i class="fas fa-sync-alt"></i>
                    </button>
                    <span class="last-update">Updated: Never</span>
                </div>
            `;
            this.newsContainer.insertBefore(header, this.newsContainer.firstChild);
        }
    }
    
    async loadEnhancedNews() {
        if (this.isLoading) return;
        
        this.isLoading = true;
        
        try {
            // Use connection manager for consistent loading/error handling
            const articlesData = await window.connectionManager?.request('/api/news/enhanced?limit=20') || 
                                  await fetch('/api/news/enhanced?limit=20').then(r => r.json());
            
            const summaryData = await window.connectionManager?.request('/api/news/sentiment-summary') || 
                                await fetch('/api/news/sentiment-summary').then(r => r.json());
            
            if (articlesData.success && articlesData.articles) {
                this.displayNewsWithSummary(articlesData.articles, summaryData);
                this.updateLastUpdated(articlesData.last_updated);
                
                if (articlesData.fallback) {
                    this.showFallbackNotice();
                }
            } else {
                this.showErrorState('Failed to load news articles');
            }
            
        } catch (error) {
            console.error('Error loading enhanced news:', error);
            
            // Handle error through connection manager
            if (window.connectionManager) {
                window.connectionManager.handleError(this.componentId, error);
            } else {
                this.showErrorState('Network error loading news');
            }
        } finally {
            this.isLoading = false;
        }
    }
    
    displayNewsWithSummary(articles, summaryData) {
        console.log('📰 Displaying news with summary. Articles:', articles?.length, 'Summary:', summaryData);
        
        // Find or create articles container
        let articlesContainer = this.newsContainer.querySelector('.news-articles');
        if (!articlesContainer) {
            articlesContainer = document.createElement('div');
            articlesContainer.className = 'news-articles';
            this.newsContainer.appendChild(articlesContainer);
            console.log('📦 Created new articles container');
        }
        
        // Clear existing articles
        articlesContainer.innerHTML = '';
        console.log('🧹 Cleared existing articles');
        
        // Create sentiment summary using API data
        const sentimentSummary = this.createSentimentSummaryFromAPI(summaryData);
        articlesContainer.appendChild(sentimentSummary);
        console.log('📊 Added sentiment summary');
        
        // Add articles header
        const articlesHeader = document.createElement('div');
        articlesHeader.className = 'articles-header';
        articlesHeader.innerHTML = `
            <h4><i class="fas fa-list"></i> Latest News Articles (${articles?.length || 0})</h4>
            <div class="articles-stats">
                <span class="bullish-articles">${articles?.filter(a => (a.sentiment_label || '').toLowerCase().includes('bullish')).length || 0} Bullish</span>
                <span class="neutral-articles">${articles?.filter(a => (a.sentiment_label || '').toLowerCase() === 'neutral').length || 0} Neutral</span>
                <span class="bearish-articles">${articles?.filter(a => (a.sentiment_label || '').toLowerCase().includes('bearish')).length || 0} Bearish</span>
            </div>
        `;
        articlesContainer.appendChild(articlesHeader);
        console.log('📋 Added articles header');
        
        // Display articles
        if (articles && articles.length > 0) {
            console.log('📰 Processing', articles.length, 'articles...');
            
            articles.forEach((article, index) => {
                try {
                    const articleElement = this.createArticleElement(article, index);
                    articlesContainer.appendChild(articleElement);
                    console.log(`✅ Added article ${index + 1}:`, article.title?.substring(0, 50) + '...');
                } catch (error) {
                    console.error(`❌ Error creating article ${index + 1}:`, error, article);
                }
            });
            
            console.log(`✅ Successfully displayed ${articles.length} news articles with comprehensive sentiment analysis`);
        } else {
            console.log('⚠️ No articles to display, showing fallback message');
            
            // Show no articles message
            const noArticlesDiv = document.createElement('div');
            noArticlesDiv.className = 'no-articles';
            noArticlesDiv.innerHTML = `
                <div class="no-articles-content">
                    <i class="fas fa-newspaper"></i>
                    <p>No articles available at the moment</p>
                    <button onclick="enhancedNewsManager.loadEnhancedNews()" class="retry-btn">
                        <i class="fas fa-sync-alt"></i> Retry
                    </button>
                </div>
            `;
            articlesContainer.appendChild(noArticlesDiv);
        }
    }
    
    displayNewsArticles(articles) {
        // Find or create articles container
        let articlesContainer = this.newsContainer.querySelector('.news-articles');
        if (!articlesContainer) {
            articlesContainer = document.createElement('div');
            articlesContainer.className = 'news-articles';
            this.newsContainer.appendChild(articlesContainer);
        }
        
        // Clear existing articles
        articlesContainer.innerHTML = '';
        
        // Create sentiment summary first
        const sentimentSummary = this.createSentimentSummary(articles);
        articlesContainer.appendChild(sentimentSummary);
        
        // Display articles
        articles.forEach((article, index) => {
            const articleElement = this.createArticleElement(article, index);
            articlesContainer.appendChild(articleElement);
        });
        
        console.log(`Displayed ${articles.length} news articles with sentiment analysis`);
    }
    
    createSentimentSummaryFromAPI(summaryData) {
        const summaryDiv = document.createElement('div');
        summaryDiv.className = 'sentiment-summary-panel';
        
        if (!summaryData || !summaryData.success) {
            summaryDiv.innerHTML = `
                <div class="summary-header">
                    <h4><i class="fas fa-chart-line"></i> Market Sentiment Analysis</h4>
                    <div class="overall-sentiment sentiment-neutral">LOADING...</div>
                </div>
                <div class="sentiment-error">
                    <i class="fas fa-exclamation-triangle"></i>
                    <span>Unable to load sentiment summary</span>
                </div>
            `;
            return summaryDiv;
        }
        
        const timeframes = summaryData.timeframes;
        const marketOutlook = summaryData.market_outlook || 'NEUTRAL';
        
        summaryDiv.innerHTML = `
            <div class="summary-header">
                <h4><i class="fas fa-chart-line"></i> Market Sentiment Analysis</h4>
                <div class="overall-sentiment ${this.getOverallSentimentClassFromAPI(marketOutlook)}">
                    ${marketOutlook}
                </div>
            </div>
            
            <div class="market-stats">
                <div class="stat-item">
                    <span class="stat-label">Total Articles:</span>
                    <span class="stat-value">${summaryData.total_articles_analyzed || 0}</span>
                </div>
                <div class="stat-item">
                    <span class="stat-label">Sentiment Score:</span>
                    <span class="stat-value ${summaryData.overall_sentiment_score >= 0 ? 'positive' : 'negative'}">
                        ${summaryData.overall_sentiment_score >= 0 ? '+' : ''}${(summaryData.overall_sentiment_score || 0).toFixed(2)}
                    </span>
                </div>
            </div>
            
            <div class="timeframe-grid">
                ${Object.keys(timeframes).map(period => {
                    const data = timeframes[period];
                    return `
                        <div class="timeframe-item">
                            <div class="timeframe-label">${period}</div>
                            <div class="sentiment-bar">
                                <div class="sentiment-fill ${this.getSentimentClassFromAPI(data.overall_sentiment)}" 
                                     style="width: ${Math.max(data.sentiment_strength * 100, 10)}%"></div>
                            </div>
                            <div class="sentiment-details">
                                <span class="sentiment-value">${data.overall_sentiment}</span>
                                <span class="article-count">(${data.total_articles} articles)</span>
                            </div>
                            <div class="sentiment-breakdown">
                                <span class="bullish-count">↗ ${data.bullish_count}</span>
                                <span class="neutral-count">→ ${data.neutral_count}</span>
                                <span class="bearish-count">↘ ${data.bearish_count}</span>
                            </div>
                        </div>
                    `;
                }).join('')}
            </div>
            
            <div class="sentiment-insights">
                <div class="insight-item">
                    <i class="fas fa-arrow-up text-success"></i>
                    <span>Bullish Signals: ${this.getTotalBullish(timeframes)}</span>
                </div>
                <div class="insight-item">
                    <i class="fas fa-minus text-neutral"></i>
                    <span>Neutral: ${this.getTotalNeutral(timeframes)}</span>
                </div>
                <div class="insight-item">
                    <i class="fas fa-arrow-down text-danger"></i>
                    <span>Bearish Signals: ${this.getTotalBearish(timeframes)}</span>
                </div>
            </div>
        `;
        
        return summaryDiv;
    }
    
    getOverallSentimentClassFromAPI(sentiment) {
        if (sentiment === 'BULLISH') return 'sentiment-bullish';
        if (sentiment === 'BEARISH') return 'sentiment-bearish';
        return 'sentiment-neutral';
    }
    
    getSentimentClassFromAPI(sentiment) {
        if (sentiment === 'BULLISH') return 'sentiment-bullish';
        if (sentiment === 'BEARISH') return 'sentiment-bearish';
        return 'sentiment-neutral';
    }
    
    getTotalBullish(timeframes) {
        return Object.values(timeframes).reduce((sum, tf) => sum + tf.bullish_count, 0);
    }
    
    getTotalNeutral(timeframes) {
        return Object.values(timeframes).reduce((sum, tf) => sum + tf.neutral_count, 0);
    }
    
    getTotalBearish(timeframes) {
        return Object.values(timeframes).reduce((sum, tf) => sum + tf.bearish_count, 0);
    }

    createSentimentSummary(articles) {
        const summaryDiv = document.createElement('div');
        summaryDiv.className = 'sentiment-summary-panel';
        
        // Calculate timeframe-based sentiment
        const timeframes = this.calculateTimeframeSentiment(articles);
        
        summaryDiv.innerHTML = `
            <div class="summary-header">
                <h4><i class="fas fa-chart-line"></i> Market Sentiment Analysis</h4>
                <div class="overall-sentiment ${this.getOverallSentimentClass(timeframes['1H'])}">
                    ${this.getOverallSentimentText(timeframes['1H'])}
                </div>
            </div>
            
            <div class="timeframe-grid">
                <div class="timeframe-item">
                    <div class="timeframe-label">1H</div>
                    <div class="sentiment-bar">
                        <div class="sentiment-fill ${this.getSentimentClass(timeframes['1H'].sentiment)}" 
                             style="width: ${Math.abs(timeframes['1H'].score) * 100}%"></div>
                    </div>
                    <div class="sentiment-details">
                        <span class="sentiment-value">${timeframes['1H'].score >= 0 ? '+' : ''}${timeframes['1H'].score.toFixed(1)}</span>
                        <span class="article-count">(${timeframes['1H'].count} articles)</span>
                    </div>
                </div>
                
                <div class="timeframe-item">
                    <div class="timeframe-label">4H</div>
                    <div class="sentiment-bar">
                        <div class="sentiment-fill ${this.getSentimentClass(timeframes['4H'].sentiment)}" 
                             style="width: ${Math.abs(timeframes['4H'].score) * 100}%"></div>
                    </div>
                    <div class="sentiment-details">
                        <span class="sentiment-value">${timeframes['4H'].score >= 0 ? '+' : ''}${timeframes['4H'].score.toFixed(1)}</span>
                        <span class="article-count">(${timeframes['4H'].count} articles)</span>
                    </div>
                </div>
                
                <div class="timeframe-item">
                    <div class="timeframe-label">1D</div>
                    <div class="sentiment-bar">
                        <div class="sentiment-fill ${this.getSentimentClass(timeframes['1D'].sentiment)}" 
                             style="width: ${Math.abs(timeframes['1D'].score) * 100}%"></div>
                    </div>
                    <div class="sentiment-details">
                        <span class="sentiment-value">${timeframes['1D'].score >= 0 ? '+' : ''}${timeframes['1D'].score.toFixed(1)}</span>
                        <span class="article-count">(${timeframes['1D'].count} articles)</span>
                    </div>
                </div>
                
                <div class="timeframe-item">
                    <div class="timeframe-label">1W</div>
                    <div class="sentiment-bar">
                        <div class="sentiment-fill ${this.getSentimentClass(timeframes['1W'].sentiment)}" 
                             style="width: ${Math.abs(timeframes['1W'].score) * 100}%"></div>
                    </div>
                    <div class="sentiment-details">
                        <span class="sentiment-value">${timeframes['1W'].score >= 0 ? '+' : ''}${timeframes['1W'].score.toFixed(1)}</span>
                        <span class="article-count">(${timeframes['1W'].count} articles)</span>
                    </div>
                </div>
                
                <div class="timeframe-item">
                    <div class="timeframe-label">1M</div>
                    <div class="sentiment-bar">
                        <div class="sentiment-fill ${this.getSentimentClass(timeframes['1M'].sentiment)}" 
                             style="width: ${Math.abs(timeframes['1M'].score) * 100}%"></div>
                    </div>
                    <div class="sentiment-details">
                        <span class="sentiment-value">${timeframes['1M'].score >= 0 ? '+' : ''}${timeframes['1M'].score.toFixed(1)}</span>
                        <span class="article-count">(${timeframes['1M'].count} articles)</span>
                    </div>
                </div>
            </div>
            
            <div class="sentiment-insights">
                <div class="insight-item">
                    <i class="fas fa-arrow-up text-success"></i>
                    <span>Bullish: ${timeframes.summary.bullish} articles</span>
                </div>
                <div class="insight-item">
                    <i class="fas fa-minus text-neutral"></i>
                    <span>Neutral: ${timeframes.summary.neutral} articles</span>
                </div>
                <div class="insight-item">
                    <i class="fas fa-arrow-down text-danger"></i>
                    <span>Bearish: ${timeframes.summary.bearish} articles</span>
                </div>
            </div>
        `;
        
        return summaryDiv;
    }
    
    calculateTimeframeSentiment(articles) {
        const now = new Date();
        const timeframes = {
            '1H': { cutoff: new Date(now.getTime() - 60 * 60 * 1000), score: 0, count: 0, sentiment: 'neutral' },
            '4H': { cutoff: new Date(now.getTime() - 4 * 60 * 60 * 1000), score: 0, count: 0, sentiment: 'neutral' },
            '1D': { cutoff: new Date(now.getTime() - 24 * 60 * 60 * 1000), score: 0, count: 0, sentiment: 'neutral' },
            '1W': { cutoff: new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000), score: 0, count: 0, sentiment: 'neutral' },
            '1M': { cutoff: new Date(now.getTime() - 30 * 24 * 60 * 60 * 1000), score: 0, count: 0, sentiment: 'neutral' }
        };
        
        const summary = { bullish: 0, neutral: 0, bearish: 0 };
        
        articles.forEach(article => {
            const publishedDate = new Date(article.published_at || article.time_ago || Date.now());
            const sentimentScore = article.sentiment_score || 0;
            const sentimentLabel = (article.sentiment_label || 'neutral').toLowerCase();
            
            // Update summary counts
            if (sentimentLabel.includes('bullish') || sentimentLabel.includes('positive')) {
                summary.bullish++;
            } else if (sentimentLabel.includes('bearish') || sentimentLabel.includes('negative')) {
                summary.bearish++;
            } else {
                summary.neutral++;
            }
            
            // Calculate timeframe scores
            Object.keys(timeframes).forEach(period => {
                if (publishedDate >= timeframes[period].cutoff) {
                    timeframes[period].score += sentimentScore;
                    timeframes[period].count++;
                }
            });
        });
        
        // Calculate average scores and determine sentiment labels
        Object.keys(timeframes).forEach(period => {
            if (timeframes[period].count > 0) {
                timeframes[period].score = timeframes[period].score / timeframes[period].count;
                
                if (timeframes[period].score > 0.2) {
                    timeframes[period].sentiment = 'bullish';
                } else if (timeframes[period].score < -0.2) {
                    timeframes[period].sentiment = 'bearish';
                } else {
                    timeframes[period].sentiment = 'neutral';
                }
            }
        });
        
        timeframes.summary = summary;
        return timeframes;
    }
    
    getOverallSentimentClass(timeframe) {
        if (timeframe.score > 0.2) return 'sentiment-bullish';
        if (timeframe.score < -0.2) return 'sentiment-bearish';
        return 'sentiment-neutral';
    }
    
    getOverallSentimentText(timeframe) {
        if (timeframe.score > 0.2) return 'BULLISH TREND';
        if (timeframe.score < -0.2) return 'BEARISH TREND';
        return 'NEUTRAL MARKET';
    }

    createArticleElement(article, index) {
        const articleDiv = document.createElement('div');
        articleDiv.className = 'news-article';
        articleDiv.setAttribute('data-index', index);
        
        // Determine sentiment styling
        const sentimentClass = this.getSentimentClass(article.sentiment_label);
        const sentimentIcon = this.getSentimentIcon(article.sentiment_label);
        const priceChangeColor = this.getPriceChangeColor(article.price_change_1h || 0);
        
        // Format price changes
        const priceChange1h = this.formatPriceChange(article.price_change_1h);
        const priceChange4h = this.formatPriceChange(article.price_change_4h);
        const priceChange24h = this.formatPriceChange(article.price_change_24h);
        
        // Confidence level
        const confidenceLevel = this.getConfidenceLevel(article.confidence_score || 0.1);
        
        articleDiv.innerHTML = `
            <div class=\"article-header\">
                <div class=\"sentiment-indicator ${sentimentClass}\">
                    ${sentimentIcon}
                    <span class=\"sentiment-label\">${(article.sentiment_label || 'neutral').toUpperCase()}</span>
                </div>
                <div class=\"article-time\">${article.time_ago || 'Unknown'}</div>
            </div>
            
            <div class=\"article-content\">
                <h4 class=\"article-title\">${this.escapeHtml(article.title)}</h4>
                <p class=\"article-summary\">${this.escapeHtml(this.truncateText(article.content, 120))}</p>
                <div class=\"article-source\">${this.escapeHtml(article.source)}</div>
            </div>
            
            <div class=\"price-impact\">
                <div class=\"price-changes\">
                    <div class=\"price-change\" style=\"color: ${priceChangeColor}\">
                        <span class=\"label\">1H:</span>
                        <span class=\"value\">${priceChange1h}</span>
                    </div>
                    <div class=\"price-change\" style=\"color: ${this.getPriceChangeColor(article.price_change_4h || 0)}\">
                        <span class=\"label\">4H:</span>
                        <span class=\"value\">${priceChange4h}</span>
                    </div>
                    <div class=\"price-change\" style=\"color: ${this.getPriceChangeColor(article.price_change_24h || 0)}\">
                        <span class=\"label\">24H:</span>
                        <span class=\"value\">${priceChange24h}</span>
                    </div>
                </div>
                <div class=\"confidence-meter\">
                    <span class=\"confidence-label\">Confidence:</span>
                    <div class=\"confidence-bar\">
                        <div class=\"confidence-fill ${confidenceLevel}\" style=\"width: ${(article.confidence_score || 0.1) * 100}%\"></div>
                    </div>
                    <span class=\"confidence-value\">${Math.round((article.confidence_score || 0.1) * 100)}%</span>
                </div>
            </div>
            
            <div class=\"article-actions\">
                <button class=\"read-more-btn\" onclick=\"enhancedNewsManager.openArticle('${article.url || '#'}')\" ${!article.url ? 'disabled' : ''}>
                    Read More
                </button>
                <button class=\"analyze-btn\" onclick=\"enhancedNewsManager.analyzeArticle(${index})\">
                    Deep Analysis
                </button>
            </div>
        `;
        
        return articleDiv;
    }
    
    getSentimentClass(sentiment) {
        switch (sentiment?.toLowerCase()) {
            case 'bullish':
            case 'positive':
                return 'sentiment-bullish';
            case 'bearish':
            case 'negative':
                return 'sentiment-bearish';
            default:
                return 'sentiment-neutral';
        }
    }
    
    getSentimentIcon(sentiment) {
        switch (sentiment?.toLowerCase()) {
            case 'bullish':
            case 'positive':
                return '<i class=\"fas fa-arrow-up\"></i>';
            case 'bearish':
            case 'negative':
                return '<i class=\"fas fa-arrow-down\"></i>';
            default:
                return '<i class=\"fas fa-minus\"></i>';
        }
    }
    
    getPriceChangeColor(change) {
        if (!change || change === 0) return '#666';
        return change > 0 ? '#00ff88' : '#ff4444';
    }
    
    formatPriceChange(change) {
        if (!change || change === 0) return '0.0%';
        const sign = change > 0 ? '+' : '';
        return `${sign}${change.toFixed(1)}%`;
    }
    
    getConfidenceLevel(confidence) {
        if (confidence >= 0.8) return 'confidence-high';
        if (confidence >= 0.5) return 'confidence-medium';
        return 'confidence-low';
    }
    
    truncateText(text, maxLength) {
        if (!text || text.length <= maxLength) return text || '';
        return text.substring(0, maxLength) + '...';
    }
    
    escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    openArticle(url) {
        if (url && url !== '#') {
            window.open(url, '_blank');
        }
    }
    
    async analyzeArticle(index) {
        // Get article from the displayed articles
        const articleElement = this.newsContainer.querySelector(`[data-index=\"${index}\"]`);
        if (!articleElement) return;
        
        const title = articleElement.querySelector('.article-title').textContent;
        const content = articleElement.querySelector('.article-summary').textContent;
        
        try {
            const response = await fetch('/api/news/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    title: title,
                    content: content
                })
            });
            
            const data = await response.json();
            
            if (data.success) {
                this.showAnalysisModal(data.prediction);
            } else {
                alert('Analysis failed: ' + data.error);
            }
            
        } catch (error) {
            console.error('Analysis error:', error);
            alert('Failed to analyze article');
        }
    }
    
    showAnalysisModal(prediction) {
        // Create modal for detailed analysis
        const modal = document.createElement('div');
        modal.className = 'analysis-modal';
        modal.innerHTML = `
            <div class=\"modal-content\">
                <div class=\"modal-header\">
                    <h3>Deep Analysis Results</h3>
                    <button class=\"close-btn\" onclick=\"this.closest('.analysis-modal').remove()\">&times;</button>
                </div>
                <div class=\"modal-body\">
                    <div class=\"prediction-result\">
                        <div class=\"predicted-direction ${prediction.direction === 'up' ? 'bullish' : prediction.direction === 'down' ? 'bearish' : 'neutral'}\">
                            <strong>Predicted Direction:</strong> ${prediction.direction?.toUpperCase() || 'NEUTRAL'}
                        </div>
                        <div class=\"predicted-magnitude\">
                            <strong>Expected Impact:</strong> ${prediction.magnitude?.toFixed(2) || '0.00'}%
                        </div>
                        <div class=\"prediction-confidence\">
                            <strong>Confidence:</strong> ${Math.round((prediction.confidence || 0) * 100)}%
                        </div>
                    </div>
                    <div class=\"similar-events\">
                        <h4>Similar Historical Events:</h4>
                        <div class=\"historical-matches\">
                            ${prediction.similar_events?.map(event => `
                                <div class=\"historical-event\">
                                    <strong>${event.headline}</strong><br>
                                    <small>Impact: ${event.price_change > 0 ? '+' : ''}${event.price_change.toFixed(2)}% | ${event.date}</small>
                                </div>
                            `).join('') || '<p>No similar historical events found.</p>'}
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        
        // Close modal when clicking outside
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.remove();
            }
        });
    }
    
    showLoadingState() {
        const refreshBtn = this.newsContainer.querySelector('.refresh-btn i');
        if (refreshBtn) {
            refreshBtn.classList.add('fa-spin');
        }
        
        // Add loading overlay
        let loadingOverlay = this.newsContainer.querySelector('.loading-overlay');
        if (!loadingOverlay) {
            loadingOverlay = document.createElement('div');
            loadingOverlay.className = 'loading-overlay';
            loadingOverlay.innerHTML = '<div class=\"loading-spinner\"><i class=\"fas fa-spinner fa-spin\"></i> Loading news...</div>';
            this.newsContainer.appendChild(loadingOverlay);
        }
        loadingOverlay.style.display = 'flex';
    }
    
    hideLoadingState() {
        const refreshBtn = this.newsContainer.querySelector('.refresh-btn i');
        if (refreshBtn) {
            refreshBtn.classList.remove('fa-spin');
        }
        
        const loadingOverlay = this.newsContainer.querySelector('.loading-overlay');
        if (loadingOverlay) {
            loadingOverlay.style.display = 'none';
        }
    }
    
    showErrorState(message) {
        const articlesContainer = this.newsContainer.querySelector('.news-articles');
        if (articlesContainer) {
            articlesContainer.innerHTML = `
                <div class=\"error-state\">
                    <i class=\"fas fa-exclamation-triangle\"></i>
                    <p>${message}</p>
                    <button onclick=\"enhancedNewsManager.loadEnhancedNews()\">Try Again</button>
                </div>
            `;
        }
    }
    
    showFallbackNotice() {
        const header = this.newsContainer.querySelector('.news-header');
        if (header && !header.querySelector('.fallback-notice')) {
            const notice = document.createElement('div');
            notice.className = 'fallback-notice';
            notice.innerHTML = '<i class=\"fas fa-info-circle\"></i> Using sample data (live feeds unavailable)';
            header.appendChild(notice);
            
            // Remove notice after 5 seconds
            setTimeout(() => notice.remove(), 5000);
        }
    }
    
    updateLastUpdated(timestamp) {
        const lastUpdateSpan = this.newsContainer.querySelector('.last-update');
        if (lastUpdateSpan && timestamp) {
            const date = new Date(timestamp);
            lastUpdateSpan.textContent = `Updated: ${date.toLocaleTimeString()}`;
        }
    }
    
    /**
     * Setup connection manager integration
     */
    setupConnectionManagerIntegration() {
        if (!window.connectionManager) {
            console.warn('⚠️ Connection Manager not available for news manager');
            return;
        }
        
        // Add data-component attribute for loading/error states
        this.newsContainer.setAttribute('data-component', this.componentId);
        
        // Setup retry handler
        const retryCleanup = window.connectionManager.on('retry', (data) => {
            if (data.componentId === this.componentId) {
                console.log('🔄 Retrying news load...');
                this.loadEnhancedNews();
            }
        });
        this.eventCleanup.push(retryCleanup);
        
        // Setup WebSocket news updates
        const newsUpdateCleanup = window.connectionManager.on('news_update', (data) => {
            console.log('📰 Real-time news update received:', data);
            this.loadEnhancedNews();
        });
        this.eventCleanup.push(newsUpdateCleanup);
        
        console.log('✅ Connection Manager integration setup for news');
    }
    
    /**
     * Cleanup all event listeners and resources
     */
    cleanup() {
        console.log('🧹 Cleaning up Enhanced News Manager...');
        
        // Clear refresh interval
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
        }
        
        // Clean up event listeners
        this.eventCleanup.forEach(cleanup => cleanup());
        this.eventCleanup = [];
        
        // Clean up from connection manager
        if (window.connectionManager) {
            window.connectionManager.offContext(this);
        }
        
        console.log('✅ Enhanced News Manager cleaned up');
    }
    
    destroy() {
        this.cleanup();
    }
}

// Create global instance for component loader
window.enhancedNewsManager = new EnhancedNewsManager();

// Add init method for component loader compatibility
window.enhancedNewsManager.init = function() {
    this.initializeNewsDisplay();
    return Promise.resolve();
};

// Legacy global reference
let enhancedNewsManager = window.enhancedNewsManager;

// Auto-initialize when script loads
if (typeof window !== 'undefined') {
    // Cleanup on page unload
    window.addEventListener('beforeunload', () => {
        if (window.enhancedNewsManager) {
            window.enhancedNewsManager.destroy();
        }
    });
}

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = EnhancedNewsManager;
}

console.log('📰 Enhanced News Manager loaded successfully');
