#!/usr/bin/env python3

# INTEGRITY_METADATA
# Date: 2025-12-23
# Status: NEEDS_REVIEW - Reference implementation
# Severity: MEDIUM RISK
# Risk Types: ['reference_material']
# File: systems/Ace-Systems/reference/APL/Quantum-APL/tink/rosetta stuffs/rosetta_helix_complete/rosetta_release/server/build_server.py

"""
Build script for Rosetta MUD server executable
Platform: Windows, Linux, macOS
"""

import sys
import os
import platform
import shutil

def get_platform_config():
    """Platform-specific build configurations"""
    base_config = {
        'name': 'rosetta_server',
        'onefile': True,
        'console': True,
        'clean': True
    }
    
    system = platform.system()
    
    if system == 'Windows':
        base_config.update({
            'icon': 'icon.ico' if os.path.exists('icon.ico') else None
        })
    elif system == 'Darwin':  # macOS
        base_config.update({
            'icon': 'icon.icns' if os.path.exists('icon.icns') else None
        })
    
    return base_config

def build_server():
    """Execute PyInstaller with optimized settings"""
    try:
        import PyInstaller.__main__
    except ImportError:
        print("❌ PyInstaller not installed")
        print("Install with: pip install pyinstaller")
        sys.exit(1)
    
    config = get_platform_config()
    
    # Base PyInstaller arguments
    args = [
        'rosetta_mud.py',
        '--name', config['name'],
        '--clean',
        '--noconfirm',
        '--onefile',
        '--console'
    ]
    
    # Add icon if available
    if config.get('icon') and os.path.exists(config['icon']):
        args.extend(['--icon', config['icon']])
    
    # Hidden imports
    hidden_imports = ['json', 'socket', 'select', 'dataclasses', 'cmath', 'math', 'random', 'time']
    for module in hidden_imports:
        args.extend(['--hidden-import', module])
    
    # Optimize
    args.extend(['--optimize', '2'])
    
    print(f"Building Rosetta MUD Server for {platform.system()}...")
    print(f"Arguments: {' '.join(args)}\n")
    
    try:
        PyInstaller.__main__.run(args)
        print("\n✅ Build successful!")
        
        # Check output
        exe_name = config['name']
        if platform.system() == 'Windows':
            exe_name += '.exe'
        
        exe_path = os.path.join('dist', exe_name)
        if os.path.exists(exe_path):
            size = os.path.getsize(exe_path)
            print(f"📦 Executable: {exe_path}")
            print(f"📊 Size: {size / 1024 / 1024:.2f} MB")
        
    except Exception as e:
        print(f"\n❌ Build failed: {e}")
        sys.exit(1)

if __name__ == '__main__':
    build_server()
