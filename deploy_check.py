#!/usr/bin/env python3
"""
Railway Deployment Automation Script for QuantGoldGPT
Helps prepare and validate deployment configuration
"""

import os
import json
import sys
import subprocess
from pathlib import Path

def check_deployment_files():
    """Check if all required deployment files exist"""
    required_files = [
        'railway.json',
        'Procfile', 
        'wsgi.py',
        'requirements.txt',
        'runtime.txt',
        '.env.production'
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing deployment files: {missing_files}")
        return False
    
    print("✅ All deployment files present")
    return True

def validate_requirements():
    """Validate requirements.txt has essential packages"""
    try:
        with open('requirements.txt', 'r') as f:
            requirements = f.read()
        
        essential_packages = [
            'Flask',
            'gunicorn', 
            'eventlet',
            'psycopg2-binary',
            'requests',
            'numpy',
            'pandas'
        ]
        
        missing_packages = []
        for package in essential_packages:
            if package not in requirements:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"⚠️ Missing packages in requirements.txt: {missing_packages}")
            return False
        
        print("✅ All essential packages in requirements.txt")
        return True
        
    except FileNotFoundError:
        print("❌ requirements.txt not found")
        return False

def check_railway_config():
    """Validate railway.json configuration"""
    try:
        with open('railway.json', 'r') as f:
            config = json.load(f)
        
        if 'deploy' not in config:
            print("❌ Missing 'deploy' section in railway.json")
            return False
        
        if 'startCommand' not in config['deploy']:
            print("❌ Missing 'startCommand' in railway.json")
            return False
        
        print("✅ Railway configuration valid")
        return True
        
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"❌ Railway config error: {e}")
        return False

def generate_env_template():
    """Generate environment variables template"""
    env_vars = {
        "# Flask Configuration": "",
        "SECRET_KEY": "your-super-secret-key-here-minimum-32-characters",
        "FLASK_ENV": "production",
        "FLASK_DEBUG": "False",
        "": "",
        "# ML Dashboard Features": "",
        "ML_DASHBOARD_ENABLED": "True",
        "ENHANCED_SOCKETIO_ENABLED": "True", 
        "REAL_TIME_UPDATES": "True",
        " ": "",
        "# API Configuration": "",
        "GOLD_API_ENABLED": "True",
        "GOLD_API_URL": "https://api.gold-api.com/price/XAU",
        "  ": "",
        "# Performance Settings": "",
        "AUTO_REFRESH_INTERVAL": "60",
        "PARALLEL_DATA_LOADING": "True",
        "CACHING_ENABLED": "True",
        "   ": "",
        "# Logging": "",
        "LOG_LEVEL": "INFO",
        "PRODUCTION_LOGGING": "True"
    }
    
    print("\n📋 Environment Variables to Set in Railway:")
    print("=" * 50)
    for key, value in env_vars.items():
        if key.startswith("#"):
            print(f"\n{key}")
        elif key.strip() == "":
            continue
        else:
            print(f"{key}={value}")
    print("=" * 50)

def check_git_status():
    """Check if changes are committed"""
    try:
        result = subprocess.run(['git', 'status', '--porcelain'], 
                              capture_output=True, text=True)
        
        if result.stdout.strip():
            print("⚠️ You have uncommitted changes:")
            print(result.stdout)
            print("💡 Commit your changes before deploying to Railway")
            return False
        
        print("✅ Git repository is clean")
        return True
        
    except FileNotFoundError:
        print("⚠️ Git not found - make sure your code is version controlled")
        return False

def main():
    """Main deployment validation"""
    print("🚀 QuantGoldGPT Railway Deployment Checker")
    print("=" * 50)
    
    checks = [
        ("Deployment Files", check_deployment_files),
        ("Requirements.txt", validate_requirements),
        ("Railway Config", check_railway_config),
        ("Git Status", check_git_status)
    ]
    
    all_passed = True
    for check_name, check_func in checks:
        print(f"\n📋 Checking {check_name}...")
        if not check_func():
            all_passed = False
    
    print(f"\n{'='*50}")
    
    if all_passed:
        print("✅ Ready for Railway Deployment!")
        print("\n🚀 Next Steps:")
        print("1. Go to https://railway.app")
        print("2. Connect your GitHub repository")
        print("3. Select 'Deploy from GitHub'") 
        print("4. Choose your QuantGoldGPT repository")
        print("5. Add PostgreSQL database")
        print("6. Set environment variables (see below)")
        print("7. Deploy!")
        
        generate_env_template()
        
    else:
        print("❌ Some checks failed - fix issues before deploying")
        sys.exit(1)

if __name__ == "__main__":
    main()
