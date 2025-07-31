#!/usr/bin/env python3
"""Script to fix the corrupted dashboard HTML file by removing everything after the proper HTML ending."""

import os

def fix_html_file():
    file_path = r"c:\Users\kenne\Downloads\ml-model-training\bot_modules\GoldGPT\templates\dashboard_advanced.html"
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find the proper ending line
    end_line = None
    for i, line in enumerate(lines):
        if line.strip() == '</html>':
            end_line = i + 1
            break
    
    if end_line:
        print(f"Found proper HTML ending at line {end_line}")
        print(f"Total lines in file: {len(lines)}")
        print(f"Lines to remove: {len(lines) - end_line}")
        
        # Keep only lines up to and including </html>
        clean_lines = lines[:end_line]
        
        # Write the cleaned content back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(clean_lines)
        
        print(f"✅ File cleaned successfully! Reduced from {len(lines)} to {len(clean_lines)} lines.")
    else:
        print("❌ Could not find proper HTML ending")

if __name__ == "__main__":
    fix_html_file()
