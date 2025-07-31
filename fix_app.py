#!/usr/bin/env python3
"""
Fix the corrupted app.py file by removing invalid HTML sections
"""

def fix_app_py():
    try:
        # Read the corrupted file
        with open('app.py', 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # Find and fix the dashboard function
        fixed_lines = []
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # If we find the problematic return statement
            if 'return render_template(\'dashboard_advanced.html\')' in line:
                # Add the return statement
                fixed_lines.append(line)
                i += 1
                
                # Skip all invalid HTML until we find the except clause
                while i < len(lines) and 'except Exception as e:' not in lines[i]:
                    i += 1
                
                # Add the except clause when found
                if i < len(lines):
                    fixed_lines.append('    except Exception as e:\n')
                    i += 1
            else:
                fixed_lines.append(line)
                i += 1
        
        # Write the fixed file
        with open('app_fixed.py', 'w', encoding='utf-8') as f:
            f.writelines(fixed_lines)
        
        print("✅ Fixed app.py saved as app_fixed.py")
        return True
        
    except Exception as e:
        print(f"❌ Error fixing app.py: {e}")
        return False

if __name__ == "__main__":
    fix_app_py()
