#!/usr/bin/env python3
"""
Simple Flask route checker
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from app import app
    print("🔍 Checking Flask Routes:")
    for rule in app.url_map.iter_rules():
        if '/api/' in str(rule):
            print(f"  {rule}")
except Exception as e:
    print(f"❌ Error importing app: {e}")
