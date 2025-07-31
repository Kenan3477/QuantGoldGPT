// GoldGPT Professional Dashboard Service Worker
// Provides offline capabilities and caching for PWA functionality

const CACHE_NAME = 'goldgpt-dashboard-v1.0.0';
const STATIC_CACHE_NAME = 'goldgpt-static-v1.0.0';
const API_CACHE_NAME = 'goldgpt-api-v1.0.0';

// Files to cache for offline access
const STATIC_FILES = [
    '/dashboard',
    '/static/css/unified-dashboard.css',
    '/static/css/chart-system.css',
    '/static/css/predictions-panel.css',
    '/static/css/market-context.css',
    '/static/css/correlation-analyzer.css',
    '/static/css/responsive-themes.css',
    '/static/js/unified-chart-system.js',
    '/static/js/advanced-predictions-panel.js',
    '/static/js/timeframe-correlation-analyzer.js',
    '/static/js/real-time-market-context.js',
    '/static/manifest.json'
];

// API endpoints to cache with shorter TTL
const API_ENDPOINTS = [
    '/api/chart-data/',
    '/api/predictions',
    '/api/market-context',
    '/api/correlation',
    '/api/portfolio-summary'
];

// Install event - cache static files
self.addEventListener('install', (event) => {
    console.log('🔧 Service Worker: Installing...');
    
    event.waitUntil(
        Promise.all([
            // Cache static files
            caches.open(STATIC_CACHE_NAME).then((cache) => {
                console.log('📁 Service Worker: Caching static files');
                return cache.addAll(STATIC_FILES.map(file => new Request(file, { cache: 'no-cache' })));
            }),
            // Initialize API cache
            caches.open(API_CACHE_NAME)
        ]).then(() => {
            console.log('✅ Service Worker: Installation complete');
            // Force activation of new service worker
            return self.skipWaiting();
        }).catch((error) => {
            console.error('❌ Service Worker: Installation failed', error);
        })
    );
});

// Activate event - clean up old caches
self.addEventListener('activate', (event) => {
    console.log('🚀 Service Worker: Activating...');
    
    event.waitUntil(
        caches.keys().then((cacheNames) => {
            const deletePromises = cacheNames
                .filter((cacheName) => {
                    // Delete old caches that don't match current version
                    return cacheName !== STATIC_CACHE_NAME && 
                           cacheName !== API_CACHE_NAME &&
                           cacheName.startsWith('goldgpt-');
                })
                .map((cacheName) => {
                    console.log('🗑️ Service Worker: Deleting old cache', cacheName);
                    return caches.delete(cacheName);
                });
            
            return Promise.all(deletePromises);
        }).then(() => {
            console.log('✅ Service Worker: Activation complete');
            // Take control of all clients immediately
            return self.clients.claim();
        }).catch((error) => {
            console.error('❌ Service Worker: Activation failed', error);
        })
    );
});

// Fetch event - implement caching strategies
self.addEventListener('fetch', (event) => {
    const request = event.request;
    const url = new URL(request.url);
    
    // Skip non-GET requests and chrome-extension requests
    if (request.method !== 'GET' || url.protocol === 'chrome-extension:') {
        return;
    }
    
    // Different caching strategies based on request type
    if (isStaticFile(url.pathname)) {
        // Static files: Cache first with fallback to network
        event.respondWith(cacheFirstStrategy(request, STATIC_CACHE_NAME));
    } else if (isAPIRequest(url.pathname)) {
        // API requests: Network first with cache fallback
        event.respondWith(networkFirstStrategy(request, API_CACHE_NAME));
    } else if (isDashboardRequest(url.pathname)) {
        // Dashboard: Network first with cache fallback
        event.respondWith(networkFirstStrategy(request, STATIC_CACHE_NAME));
    } else {
        // Everything else: Network only
        event.respondWith(fetch(request));
    }
});

// Cache first strategy - for static files
async function cacheFirstStrategy(request, cacheName) {
    try {
        const cachedResponse = await caches.match(request);
        if (cachedResponse) {
            console.log('📁 Service Worker: Serving from cache', request.url);
            return cachedResponse;
        }
        
        console.log('🌐 Service Worker: Fetching from network', request.url);
        const networkResponse = await fetch(request);
        
        if (networkResponse.ok) {
            const cache = await caches.open(cacheName);
            cache.put(request, networkResponse.clone());
        }
        
        return networkResponse;
    } catch (error) {
        console.error('❌ Service Worker: Cache first strategy failed', error);
        
        // Return offline fallback if available
        if (request.destination === 'document') {
            return caches.match('/dashboard');
        }
        
        throw error;
    }
}

// Network first strategy - for API requests and dynamic content
async function networkFirstStrategy(request, cacheName) {
    try {
        console.log('🌐 Service Worker: Trying network first', request.url);
        const networkResponse = await fetch(request);
        
        if (networkResponse.ok) {
            // Cache successful responses
            const cache = await caches.open(cacheName);
            cache.put(request, networkResponse.clone());
            console.log('💾 Service Worker: Cached network response', request.url);
        }
        
        return networkResponse;
    } catch (error) {
        console.log('📁 Service Worker: Network failed, trying cache', request.url);
        
        const cachedResponse = await caches.match(request);
        if (cachedResponse) {
            console.log('📁 Service Worker: Serving stale data from cache', request.url);
            return cachedResponse;
        }
        
        console.error('❌ Service Worker: Network first strategy failed', error);
        
        // Return offline fallback for API requests
        if (isAPIRequest(new URL(request.url).pathname)) {
            return new Response(JSON.stringify({
                error: 'Offline',
                message: 'No network connection available',
                cached: false
            }), {
                status: 503,
                headers: { 'Content-Type': 'application/json' }
            });
        }
        
        throw error;
    }
}

// Helper functions
function isStaticFile(pathname) {
    return pathname.startsWith('/static/') || 
           pathname.endsWith('.css') || 
           pathname.endsWith('.js') || 
           pathname.endsWith('.json') ||
           pathname.endsWith('.png') ||
           pathname.endsWith('.jpg') ||
           pathname.endsWith('.svg') ||
           pathname.endsWith('.ico');
}

function isAPIRequest(pathname) {
    return pathname.startsWith('/api/');
}

function isDashboardRequest(pathname) {
    return pathname === '/dashboard' || pathname === '/';
}

// Background sync for when connection is restored
self.addEventListener('sync', (event) => {
    console.log('🔄 Service Worker: Background sync triggered', event.tag);
    
    if (event.tag === 'refresh-data') {
        event.waitUntil(refreshCachedData());
    }
});

// Refresh cached data when connection is restored
async function refreshCachedData() {
    try {
        console.log('🔄 Service Worker: Refreshing cached data...');
        
        const cache = await caches.open(API_CACHE_NAME);
        const requests = await cache.keys();
        
        // Refresh all cached API data
        const refreshPromises = requests.map(async (request) => {
            try {
                const response = await fetch(request);
                if (response.ok) {
                    await cache.put(request, response);
                    console.log('✅ Service Worker: Refreshed cached data for', request.url);
                }
            } catch (error) {
                console.warn('⚠️ Service Worker: Failed to refresh', request.url, error);
            }
        });
        
        await Promise.all(refreshPromises);
        console.log('✅ Service Worker: Data refresh complete');
        
    } catch (error) {
        console.error('❌ Service Worker: Data refresh failed', error);
    }
}

// Handle push notifications (if implemented later)
self.addEventListener('push', (event) => {
    if (!event.data) return;
    
    try {
        const data = event.data.json();
        const options = {
            body: data.body || 'New trading signal available',
            icon: '/static/icons/icon-192x192.png',
            badge: '/static/icons/badge-72x72.png',
            tag: data.tag || 'goldgpt-notification',
            requireInteraction: data.urgent || false,
            actions: [
                {
                    action: 'open',
                    title: 'Open Dashboard'
                },
                {
                    action: 'dismiss',
                    title: 'Dismiss'
                }
            ],
            data: {
                url: data.url || '/dashboard'
            }
        };
        
        event.waitUntil(
            self.registration.showNotification(data.title || 'GoldGPT', options)
        );
    } catch (error) {
        console.error('❌ Service Worker: Push notification error', error);
    }
});

// Handle notification clicks
self.addEventListener('notificationclick', (event) => {
    event.notification.close();
    
    if (event.action === 'open' || !event.action) {
        const urlToOpen = event.notification.data?.url || '/dashboard';
        
        event.waitUntil(
            clients.matchAll({ type: 'window', includeUncontrolled: true })
                .then((clientList) => {
                    // Check if dashboard is already open
                    for (const client of clientList) {
                        if (client.url.includes('/dashboard') && 'focus' in client) {
                            return client.focus();
                        }
                    }
                    
                    // Open new dashboard window
                    if (clients.openWindow) {
                        return clients.openWindow(urlToOpen);
                    }
                })
        );
    }
});

// Message handling from main thread
self.addEventListener('message', (event) => {
    const { type, payload } = event.data;
    
    switch (type) {
        case 'SKIP_WAITING':
            self.skipWaiting();
            break;
            
        case 'GET_VERSION':
            event.ports[0].postMessage({ version: CACHE_NAME });
            break;
            
        case 'CLEAR_CACHE':
            clearAllCaches().then(() => {
                event.ports[0].postMessage({ success: true });
            }).catch((error) => {
                event.ports[0].postMessage({ success: false, error: error.message });
            });
            break;
            
        case 'CACHE_URLS':
            if (payload && payload.urls) {
                cacheUrls(payload.urls).then(() => {
                    event.ports[0].postMessage({ success: true });
                }).catch((error) => {
                    event.ports[0].postMessage({ success: false, error: error.message });
                });
            }
            break;
            
        default:
            console.warn('🤷 Service Worker: Unknown message type', type);
    }
});

// Clear all caches
async function clearAllCaches() {
    const cacheNames = await caches.keys();
    const deletePromises = cacheNames
        .filter(name => name.startsWith('goldgpt-'))
        .map(name => caches.delete(name));
    
    await Promise.all(deletePromises);
    console.log('🗑️ Service Worker: All caches cleared');
}

// Cache specific URLs
async function cacheUrls(urls) {
    const cache = await caches.open(STATIC_CACHE_NAME);
    await cache.addAll(urls);
    console.log('💾 Service Worker: URLs cached', urls);
}

console.log('🚀 Service Worker: Script loaded');
