#!/usr/bin/env python3
"""
Quick fix for Flask app indentation errors
"""

def fix_app_indentation():
    """Fix the indentation issues in app.py"""
    
    with open('app.py', 'r') as f:
        lines = f.readlines()
    
    # Find problematic lines and comment them out
    fixed_lines = []
    in_orphaned_block = False
    
    for i, line in enumerate(lines):
        # Check if this is one of the orphaned lines
        if 'Run the async function' in line or 'result = loop.run_until_complete' in line:
            in_orphaned_block = True
            
        # Check if we're back to a proper route
        if line.startswith('@app.route') and in_orphaned_block:
            in_orphaned_block = False
            
        # If we're in the orphaned block, comment out the line
        if in_orphaned_block and line.strip():
            fixed_lines.append('# ORPHANED CODE: ' + line)
        else:
            fixed_lines.append(line)
    
    # Write back the fixed file
    with open('app.py', 'w') as f:
        f.writelines(fixed_lines)
    
    print("✅ Fixed indentation errors in app.py")

if __name__ == "__main__":
    fix_app_indentation()
