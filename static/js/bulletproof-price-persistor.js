// 🛡️ BULLETPROOF GOLD PRICE PERSISTOR
// Ensures the live gold price stays displayed and prevents any overriding

(function() {
    console.log('🛡️ BULLETPROOF PRICE PERSISTOR: Starting...');
    
    let lastKnownPrice = null;
    let priceElement = null;
    let isInitialized = false;
    
    // Price protection object
    const PricePersistor = {
        init: function() {
            priceElement = document.getElementById('watchlist-xauusd-price');
            if (!priceElement) {
                console.log('🛡️ Price element not found, retrying...');
                setTimeout(this.init.bind(this), 500);
                return;
            }
            
            console.log('🛡️ Price element found:', priceElement);
            this.setupSocketConnection();
            this.setupMutationObserver();
            this.setupPeriodicCheck();
            isInitialized = true;
        },
        
        setupSocketConnection: function() {
            const socket = io();
            
            // Listen for any price updates
            socket.on('price_update', (data) => {
                if (data.symbol === 'XAUUSD' && data.price) {
                    this.updatePrice(data.price);
                }
            });
            
            socket.on('gold_price_update', (data) => {
                if (data.price) {
                    this.updatePrice(data.price);
                }
            });
            
            console.log('🛡️ Socket listeners attached');
        },
        
        updatePrice: function(newPrice) {
            const formattedPrice = `$${parseFloat(newPrice).toLocaleString('en-US', {
                minimumFractionDigits: 2,
                maximumFractionDigits: 2
            })}`;
            
            lastKnownPrice = formattedPrice;
            
            if (priceElement) {
                priceElement.textContent = formattedPrice;
                priceElement.style.color = '#00ff88';
                priceElement.style.backgroundColor = 'rgba(0, 255, 136, 0.1)';
                priceElement.style.fontWeight = 'bold';
                priceElement.style.padding = '2px 6px';
                priceElement.style.borderRadius = '4px';
                priceElement.style.border = '1px solid #00ff88';
                
                console.log('🛡️ PRICE UPDATED & PROTECTED:', formattedPrice);
            }
        },
        
        setupMutationObserver: function() {
            if (!priceElement) return;
            
            const observer = new MutationObserver((mutations) => {
                mutations.forEach((mutation) => {
                    if (mutation.type === 'childList' || mutation.type === 'characterData') {
                        const currentText = priceElement.textContent;
                        
                        // If price got changed to something we don't want, restore it
                        if (lastKnownPrice && currentText !== lastKnownPrice && 
                            (currentText.includes('3350.70') || currentText.includes('3341.93') || 
                             currentText.includes('Loading') || currentText === '--' || 
                             currentText.includes('2065.50'))) {
                            
                            console.log('🛡️ PROTECTING: Restoring price from', currentText, 'to', lastKnownPrice);
                            priceElement.textContent = lastKnownPrice;
                            
                            // Re-apply protection styles
                            priceElement.style.color = '#00ff88';
                            priceElement.style.backgroundColor = 'rgba(0, 255, 136, 0.1)';
                            priceElement.style.fontWeight = 'bold';
                            priceElement.style.padding = '2px 6px';
                            priceElement.style.borderRadius = '4px';
                            priceElement.style.border = '1px solid #00ff88';
                        }
                    }
                });
            });
            
            observer.observe(priceElement, {
                childList: true,
                subtree: true,
                characterData: true
            });
            
            console.log('🛡️ Mutation observer active - protecting price element');
        },
        
        setupPeriodicCheck: function() {
            setInterval(() => {
                if (!priceElement || !lastKnownPrice) return;
                
                const currentText = priceElement.textContent;
                
                // Periodic protection check
                if (currentText.includes('3350.70') || currentText.includes('3341.93') || 
                    currentText.includes('Loading') || currentText === '--' || 
                    currentText.includes('2065.50')) {
                    
                    console.log('🛡️ PERIODIC PROTECTION: Restoring price from', currentText, 'to', lastKnownPrice);
                    priceElement.textContent = lastKnownPrice;
                    
                    // Re-apply protection styles
                    priceElement.style.color = '#00ff88';
                    priceElement.style.backgroundColor = 'rgba(0, 255, 136, 0.1)';
                    priceElement.style.fontWeight = 'bold';
                    priceElement.style.padding = '2px 6px';
                    priceElement.style.borderRadius = '4px';
                    priceElement.style.border = '1px solid #00ff88';
                }
            }, 2000); // Check every 2 seconds
        },
        
        // Force fetch latest price from backend
        fetchLatestPrice: function() {
            fetch('/api/gold-price')
                .then(response => response.json())
                .then(data => {
                    if (data.price) {
                        this.updatePrice(data.price);
                    }
                })
                .catch(error => {
                    console.log('🛡️ Failed to fetch latest price:', error);
                });
        }
    };
    
    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', () => PricePersistor.init());
    } else {
        PricePersistor.init();
    }
    
    // Fetch latest price immediately and every 5 seconds
    setTimeout(() => {
        PricePersistor.fetchLatestPrice();
        setInterval(() => PricePersistor.fetchLatestPrice(), 5000);
    }, 2000);
    
    // Global access for debugging
    window.PricePersistor = PricePersistor;
    
    console.log('🛡️ BULLETPROOF PRICE PERSISTOR: Ready');
})();
