#!/usr/bin/env python3
"""
Enhanced SocketIO Installation and Testing Script for GoldGPT
"""

import subprocess
import sys
import os

def install_dependencies():
    """Install required dependencies for enhanced SocketIO"""
    dependencies = [
        'PyJWT>=2.8.0',
        'python-socketio>=5.10.0',
        'flask-socketio>=5.3.0'
    ]
    
    print("Installing enhanced SocketIO dependencies...")
    
    for dep in dependencies:
        try:
            print(f"Installing {dep}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', dep])
            print(f"[OK] {dep} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Failed to install {dep}: {e}")
            return False
    
    print("[OK] All dependencies installed successfully!")
    return True

def test_integration():
    """Test the enhanced SocketIO integration"""
    print("\nTesting Enhanced SocketIO Integration...")
    print("=" * 50)
    
    try:
        # Test basic imports
        print("Testing imports...")
        
        import jwt
        print("[OK] JWT authentication available")
        
        from flask_socketio import SocketIO
        print("[OK] Flask-SocketIO available")
        
        # Test enhanced modules
        try:
            from enhanced_socketio_server import GoldGPTSocketIOServer
            print("[OK] Enhanced SocketIO server module available")
        except ImportError as e:
            print(f"[ERROR] Enhanced SocketIO server module missing: {e}")
            return False
        
        try:
            from enhanced_socketio_integration import integrate_enhanced_socketio
            print("[OK] Enhanced SocketIO integration module available")
        except ImportError as e:
            print(f"[ERROR] Enhanced SocketIO integration module missing: {e}")
            return False
        
        # Test server initialization
        print("\nTesting server initialization...")
        from flask import Flask
        
        test_app = Flask(__name__)
        test_app.config['SECRET_KEY'] = 'test-secret-key'
        
        try:
            server = GoldGPTSocketIOServer(test_app)
            print("[OK] Enhanced SocketIO server can be initialized")
            
            # Test token generation
            token = server.generate_auth_token('test_client')
            print("[OK] JWT token generation working")
            
            # Test token verification
            client_id = server.verify_auth_token(token)
            if client_id == 'test_client':
                print("[OK] JWT token verification working")
            else:
                print("[ERROR] JWT token verification failed")
                return False
                
        except Exception as e:
            print(f"[ERROR] Server initialization failed: {e}")
            return False
        
        print("\n[OK] All tests passed! Enhanced SocketIO is ready to use.")
        return True
        
    except ImportError as e:
        print(f"[ERROR] Import error: {e}")
        print("Try running: pip install PyJWT flask-socketio")
        return False
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        return False

def create_integration_guide():
    """Create a simple integration guide"""
    guide_content = '''
# GoldGPT Enhanced SocketIO Integration Guide

## Files Created:
1. `enhanced_socketio_server.py` - Core enhanced SocketIO server
2. `enhanced_socketio_integration.py` - Integration utilities  
3. `static/js/enhanced_websocket_client.js` - Client-side JavaScript
4. `socketio_integration_patch.py` - Integration examples

## Quick Integration Steps:

### 1. Install Dependencies (if needed):
```bash
pip install PyJWT flask-socketio
```

### 2. Add to your existing app.py:
```python
# Add after your existing imports
try:
    from enhanced_socketio_integration import (
        integrate_enhanced_socketio, 
        add_enhanced_websocket_routes
    )
    ENHANCED_AVAILABLE = True
except ImportError:
    ENHANCED_AVAILABLE = False

# Add after your socketio = SocketIO(app, ...) line
if ENHANCED_AVAILABLE:
    enhanced_server = integrate_enhanced_socketio(app, socketio)
    add_enhanced_websocket_routes(app)
```

### 3. Update your templates to include enhanced client:
```html
<script src="https://cdn.socket.io/4.0.0/socket.io.min.js"></script>
<script src="{{ url_for('static', filename='js/enhanced_websocket_client.js') }}"></script>
```

### 4. Test the integration:
- Visit `/websocket-test` for interactive testing
- Visit `/websocket-docs` for API documentation
- Check `/api/websocket/stats` for server statistics

## Enhanced Features:

### Real-time Updates:
- Price updates every 2 seconds (vs 15 seconds)
- AI analysis every 30 seconds
- Portfolio updates every 10 seconds

### Security:
- JWT authentication for all WebSocket connections
- Rate limiting to prevent abuse
- Secure token-based session management

### Reliability:
- Automatic reconnection with exponential backoff
- Connection health monitoring
- Comprehensive error handling

### Monitoring:
- Real-time connection statistics
- Rate limiting tracking
- Performance metrics

### Advanced Features:
- Room-based subscriptions for targeted updates
- Background task management
- Multiple API fallbacks for price data

## API Endpoints:

- `GET /api/websocket/stats` - Server statistics
- `GET /api/websocket/auth` - Get auth token
- `GET /api/websocket/test` - Test broadcast
- `GET /websocket-test` - Interactive test page
- `GET /websocket-docs` - Full API documentation

## Client Usage Example:

```javascript
const wsClient = new GoldGPTWebSocketClient();

wsClient.on('connected', (data) => {
    console.log('Connected with features:', data.features);
});

wsClient.on('priceUpdate', (data) => {
    console.log('Gold price:', data.price);
    // Update your UI
});

await wsClient.connect();
```

## Troubleshooting:

1. **Import errors**: Run `pip install PyJWT flask-socketio`
2. **Connection issues**: Check firewall and port settings
3. **Authentication failures**: Verify JWT secret key configuration
4. **Rate limiting**: Reduce request frequency or increase limits

## Support:
- Test page: `/websocket-test`
- Documentation: `/websocket-docs`
- Statistics: `/api/websocket/stats`
'''
    
    with open('ENHANCED_SOCKETIO_GUIDE.md', 'w', encoding='utf-8') as f:
        f.write(guide_content)
    
    print("Integration guide created: ENHANCED_SOCKETIO_GUIDE.md")

def main():
    """Main installation and testing function"""
    print("GoldGPT Enhanced SocketIO Setup")
    print("=" * 40)
    
    # Check if we're in the right directory
    if not os.path.exists('app.py'):
        print("Warning: app.py not found in current directory")
        print("Make sure you're running this from your GoldGPT project root")
    
    # Install dependencies
    print("\n1. Installing dependencies...")
    if not install_dependencies():
        print("Dependency installation failed")
        return
    
    # Test integration
    print("\n2. Testing integration...")
    if not test_integration():
        print("Integration test failed")
        return
    
    # Create guide
    print("\n3. Creating integration guide...")
    create_integration_guide()
    
    print("\n" + "=" * 50)
    print("Enhanced SocketIO setup completed successfully!")
    print("\nNext steps:")
    print("1. Read ENHANCED_SOCKETIO_GUIDE.md for integration instructions")
    print("2. Add the integration code to your app.py")
    print("3. Test with /websocket-test endpoint")
    print("4. Check documentation at /websocket-docs")
    
    print("\nEnhanced Features Ready:")
    print("- 2-second price updates")
    print("- JWT authentication") 
    print("- Automatic reconnection")
    print("- Rate limiting")
    print("- Comprehensive error handling")
    print("- Real-time monitoring")

if __name__ == '__main__':
    main()
