#!/usr/bin/env python3
"""
Check registered routes in the GoldGPT application
"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.getcwd())

try:
    from app import app
    
    print("\n=== All Registered Routes ===")
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append((str(rule.methods), str(rule)))
    
    # Sort routes for better readability
    routes.sort(key=lambda x: x[1])
    
    print("\n=== API Routes ===")
    api_routes = [r for r in routes if '/api/' in r[1]]
    for methods, route in api_routes:
        print(f"{methods:<25} {route}")
    
    print(f"\nFound {len(api_routes)} API routes")
    
    print("\n=== ML Related Routes ===")
    ml_routes = [r for r in routes if 'ml' in r[1].lower()]
    for methods, route in ml_routes:
        print(f"{methods:<25} {route}")
    
    print(f"\nFound {len(ml_routes)} ML-related routes")
    
    print(f"\nTotal routes: {len(routes)}")
    
except Exception as e:
    print(f"Error loading app: {e}")
    import traceback
    traceback.print_exc()
