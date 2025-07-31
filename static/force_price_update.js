// 🔥 BRUTE FORCE PRICE UPDATER - BYPASSES ALL CACHING
(function() {
    console.log('🔥 BRUTE FORCE: Starting aggressive price updater...');
    
    function bruteForceUpdate() {
        // 1. Direct DOM manipulation
        const element = document.getElementById('watchlist-xauusd-price');
        if (element) {
            // Force clear any cached content
            element.innerHTML = '';
            element.textContent = '';
            
            // Get live price from backend
            fetch('/api/gold/price?' + Date.now(), {
                method: 'GET',
                headers: {
                    'Cache-Control': 'no-cache, no-store, must-revalidate',
                    'Pragma': 'no-cache',
                    'Expires': '0'
                }
            })
            .then(response => response.json())
            .then(data => {
                if (data.success && data.price) {
                    const livePrice = `$${data.price.toLocaleString('en-US', {
                        minimumFractionDigits: 2,
                        maximumFractionDigits: 2
                    })}`;
                    
                    // BRUTE FORCE UPDATE
                    element.textContent = livePrice;
                    element.innerHTML = livePrice;
                    element.innerText = livePrice;
                    element.style.color = '#ff0000';
                    element.style.backgroundColor = '#ffff00';
                    element.style.fontWeight = 'bold';
                    element.style.fontSize = '16px';
                    
                    console.log('🔥 BRUTE FORCE: Updated to', livePrice);
                    
                    // Store in multiple places
                    window.currentGoldPrice = data.price;
                    localStorage.setItem('forceGoldPrice', livePrice);
                    sessionStorage.setItem('forceGoldPrice', livePrice);
                } else {
                    console.error('🔥 BRUTE FORCE: API failed', data);
                }
            })
            .catch(error => {
                console.error('🔥 BRUTE FORCE: Error', error);
            });
        }
    }
    
    // Execute immediately on load
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', bruteForceUpdate);
    } else {
        bruteForceUpdate();
    }
    
    // Execute every 5 seconds aggressively
    setInterval(bruteForceUpdate, 5000);
    
    // Also execute on any page interaction
    document.addEventListener('click', bruteForceUpdate);
    document.addEventListener('scroll', bruteForceUpdate);
    
    console.log('🔥 BRUTE FORCE: Price updater active');
})();
