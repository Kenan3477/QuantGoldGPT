// Emergency price fix - place this at the very end of the HTML file
console.log('🔧 Emergency price fix starting...');

// Force update the price display immediately
function forceUpdatePrice() {
    fetch('/api/live-gold-price')
        .then(response => response.json())
        .then(result => {
            console.log('🔧 Emergency price fetch result:', result);
            
            if (result.success && result.data && result.data.price) {
                const price = result.data.price;
                const formattedPrice = `$${price.toLocaleString('en-US', { 
                    style: 'decimal',  // Force decimal style to prevent currency conversion
                    minimumFractionDigits: 2, 
                    maximumFractionDigits: 2 
                })}`;
                
                // Update all price elements
                const priceElements = document.querySelectorAll('.price-value, .current-price, .symbol-details .price');
                priceElements.forEach(el => {
                    el.textContent = formattedPrice;
                    el.style.color = '#00d4aa';
                    el.classList.remove('loading'); // Remove any loading classes
                });
                
                // Update Gold API status - FORCE to connected state
                const statusElement = document.getElementById('gold-api-status');
                const textElement = document.getElementById('gold-api-text');
                if (statusElement && textElement) {
                    statusElement.className = 'gold-api-status connected';
                    textElement.textContent = `Live Gold API Connected - ${new Date().toLocaleTimeString()}`;
                }
                
                // Update watchlist
                const watchlistElement = document.querySelector('[data-symbol="XAUUSD"] .price-value');
                if (watchlistElement) {
                    watchlistElement.textContent = formattedPrice;
                    watchlistElement.style.color = '#00d4aa';
                    watchlistElement.classList.remove('loading');
                }
                
                console.log('✅ Emergency price fix applied:', formattedPrice);
            } else {
                console.error('❌ Emergency price fix failed:', result);
            }
        })
        .catch(error => {
            console.error('❌ Emergency price fetch error:', error);
        });
}

// Run immediately and then every 5 seconds
forceUpdatePrice();
setInterval(forceUpdatePrice, 5000);

// Also remove any spinning elements immediately
document.addEventListener('DOMContentLoaded', function() {
    // Remove all fa-spin classes
    const spinningElements = document.querySelectorAll('.fa-spin');
    spinningElements.forEach(el => el.classList.remove('fa-spin'));
    
    // Remove all loading classes
    const loadingElements = document.querySelectorAll('.loading');
    loadingElements.forEach(el => el.classList.remove('loading'));
});

console.log('🔧 Emergency price fix initialized - ALL SPINNING DISABLED');
forceUpdatePrice();
setInterval(forceUpdatePrice, 5000);

console.log('🔧 Emergency price fix initialized');
