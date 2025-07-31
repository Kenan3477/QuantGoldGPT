# Simple fix for app.py dashboard function
def fix_dashboard_function():
    # Read the file
    with open('app.py', 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    # Find and replace the problematic section
    # Look for the pattern: return render_template... followed by HTML junk... followed by except
    import re
    
    # Pattern to match from return statement to except clause
    pattern = r'(return render_template\(\'dashboard_advanced\.html\'\))\s+.*?(\s+except Exception as e:)'
    
    # Replacement: just the return statement followed by proper except
    replacement = r'\1\n    \2'
    
    # Apply the fix
    fixed_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    # Write back
    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(fixed_content)
    
    print("Dashboard function fixed!")

fix_dashboard_function()
