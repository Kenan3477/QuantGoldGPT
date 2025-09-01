#!/usr/bin/env python3
"""
Check Flask routes registration
"""
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("🔍 Checking Flask routes...")

try:
    # Try to import the Flask app without running it
    print("Importing Flask app...")
    
    # First, let's see if we can import Flask at all
    from flask import Flask
    print("✅ Flask imported successfully")
    
    # Now try to access the main app
    print("Trying to import main app...")
    import app as main_app
    print(f"✅ Main app imported: {type(main_app.app)}")
    
    # Check registered routes
    print("\n📋 Registered routes:")
    for rule in main_app.app.url_map.iter_rules():
        print(f"  {rule.rule} -> {rule.endpoint}")
        
    # Check specifically for our problem routes
    problem_routes = ['/api/live-gold-price', '/api/signals/active']
    print(f"\n🔍 Checking for problem routes: {problem_routes}")
    
    registered_routes = [rule.rule for rule in main_app.app.url_map.iter_rules()]
    for route in problem_routes:
        if route in registered_routes:
            print(f"  ✅ {route} is registered")
        else:
            print(f"  ❌ {route} is NOT registered")
            
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
