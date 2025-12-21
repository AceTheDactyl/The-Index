#!/usr/bin/env python3
"""
Master build script for complete Rosetta MUD distribution
Builds server, client, and bridge executables
"""

import sys
import subprocess
import platform
import os
import shutil

class BuildSystem:
    """Orchestrates complete build process"""
    
    def __init__(self):
        self.platform = platform.system()
        self.build_dir = 'build'
        self.dist_dir = 'dist'
        
    def check_dependencies(self):
        """Verify all required tools are available"""
        print("🔍 Checking dependencies...")
        missing = []
        
        # Check Python
        if sys.version_info < (3, 8):
            missing.append("Python 3.8+ required")
        
        # Check required modules
        required_modules = ['pyinstaller', 'websockets']
        for module in required_modules:
            try:
                __import__(module.replace('-', '_'))
            except ImportError:
                missing.append(f"{module}: pip install {module}")
        
        if missing:
            print("\n❌ Missing dependencies:")
            for item in missing:
                print(f"  - {item}")
            print("\nInstall all with: pip install -r requirements.txt")
            return False
        
        print("✅ All dependencies satisfied\n")
        return True
    
    def build_server(self):
        """Build server executable"""
        print("\n" + "="*60)
        print("🔨 Building server executable...")
        print("="*60)
        
        os.chdir('server')
        try:
            result = subprocess.run([sys.executable, 'build_server.py'], check=True)
            print("✅ Server build complete")
            return True
        except subprocess.CalledProcessError:
            print("❌ Server build failed")
            return False
        finally:
            os.chdir('..')
    
    def build_bridge(self):
        """Build WebSocket bridge executable"""
        print("\n" + "="*60)
        print("🔨 Building WebSocket bridge...")
        print("="*60)
        
        try:
            import PyInstaller.__main__
            
            args = [
                'client/websocket_bridge.py',
                '--name', 'rosetta_bridge',
                '--onefile',
                '--console',
                '--clean',
                '--noconfirm',
                '--optimize', '2'
            ]
            
            PyInstaller.__main__.run(args)
            print("✅ Bridge build complete")
            return True
        except Exception as e:
            print(f"❌ Bridge build failed: {e}")
            return False
    
    def create_distribution(self):
        """Package everything for distribution"""
        print("\n" + "="*60)
        print("📦 Creating distribution package...")
        print("="*60)
        
        dist_name = f"rosetta_helix_{platform.system().lower()}"
        dist_path = os.path.join(self.dist_dir, dist_name)
        
        # Create distribution directory
        os.makedirs(dist_path, exist_ok=True)
        
        # Determine executable extension
        exe_ext = '.exe' if platform.system() == 'Windows' else ''
        
        # Copy executables
        files_to_copy = [
            ('server/dist/rosetta_server' + exe_ext, 'rosetta_server' + exe_ext),
            ('dist/rosetta_bridge' + exe_ext, 'rosetta_bridge' + exe_ext),
            ('client/mud_client_websocket.html', 'mud_client.html')
        ]
        
        for src, dst in files_to_copy:
            if os.path.exists(src):
                shutil.copy(src, os.path.join(dist_path, dst))
                if not exe_ext:  # Make executable on Unix
                    if not dst.endswith('.html'):
                        os.chmod(os.path.join(dist_path, dst), 0o755)
            else:
                print(f"⚠️  Warning: {src} not found")
        
        # Create launcher scripts
        self.create_launchers(dist_path)
        
        # Create README
        self.create_readme(dist_path)
        
        print(f"✅ Distribution created: {dist_path}")
        print(f"\nContents:")
        for item in os.listdir(dist_path):
            size = os.path.getsize(os.path.join(dist_path, item))
            print(f"  - {item} ({size / 1024 / 1024:.2f} MB)")
        
        return True
    
    def create_launchers(self, dist_path):
        """Create platform-specific launcher scripts"""
        if platform.system() == 'Windows':
            # Server launcher
            with open(os.path.join(dist_path, 'start_server.bat'), 'w') as f:
                f.write('@echo off\n')
                f.write('echo Starting Rosetta MUD Server...\n')
                f.write('start rosetta_server.exe\n')
                f.write('echo Server started! Connect via telnet localhost 1234\n')
                f.write('pause\n')
            
            # Complete launcher (bridge + client)
            with open(os.path.join(dist_path, 'start_client.bat'), 'w') as f:
                f.write('@echo off\n')
                f.write('echo Starting Rosetta MUD Client...\n')
                f.write('echo.\n')
                f.write('echo Starting WebSocket bridge...\n')
                f.write('start /min rosetta_bridge.exe\n')
                f.write('timeout /t 3 /nobreak >nul\n')
                f.write('echo.\n')
                f.write('echo Opening client in browser...\n')
                f.write('start mud_client.html\n')
                f.write('echo.\n')
                f.write('echo Client launched! Make sure server is running.\n')
                f.write('pause\n')
        else:
            # Server launcher
            with open(os.path.join(dist_path, 'start_server.sh'), 'w') as f:
                f.write('#!/bin/bash\n')
                f.write('echo "Starting Rosetta MUD Server..."\n')
                f.write('./rosetta_server &\n')
                f.write('echo "Server started on port 1234"\n')
            os.chmod(os.path.join(dist_path, 'start_server.sh'), 0o755)
            
            # Client launcher
            with open(os.path.join(dist_path, 'start_client.sh'), 'w') as f:
                f.write('#!/bin/bash\n')
                f.write('echo "Starting Rosetta MUD Client..."\n')
                f.write('echo "Starting WebSocket bridge..."\n')
                f.write('./rosetta_bridge &\n')
                f.write('sleep 2\n')
                f.write('echo "Opening client in browser..."\n')
                f.write('xdg-open mud_client.html 2>/dev/null || open mud_client.html\n')
            os.chmod(os.path.join(dist_path, 'start_client.sh'), 0o755)
    
    def create_readme(self, dist_path):
        """Create distribution README"""
        readme = """# ROSETTA HELIX MUD - QUICK START

## 🚀 GETTING STARTED

### Option 1: Play Locally (Server + Client)

1. **Start the server:**
   - Windows: Double-click `start_server.bat`
   - Linux/Mac: Run `./start_server.sh`

2. **Start the client:**
   - Windows: Double-click `start_client.bat`
   - Linux/Mac: Run `./start_client.sh`

3. **Play!**
   - Your browser will open with the client
   - Type `create YourName` to begin

### Option 2: Classic Telnet

1. Start server (see above)
2. Open terminal and run: `telnet localhost 1234`
3. Type `create YourName` to begin

## 📋 SYSTEM REQUIREMENTS

- **Server:** Windows 10+, Linux, macOS 10.14+
- **RAM:** 50 MB minimum
- **Network:** Port 1234 (server), Port 8080 (bridge)

## 🎮 BASIC COMMANDS

- `look` - Look around
- `go <direction>` - Move (north, south, east, west, up, down)
- `inventory` - Check inventory
- `stats` - View character stats
- `meditate` - Synchronize with AI systems
- `help` - Full command list

## 🏗️ ARCHITECT COMMANDS (Level 60+)

- `buildroom <name>` - Create new room
- `builditem <name>` - Create new item
- `createnpc <name> <level>` - Spawn NPC
- `saveworld` - Save world state

## 🔧 TROUBLESHOOTING

**"Port already in use"**
- Close other instances of the server
- Change port in configuration

**"Can't connect"**
- Ensure server is running first
- Check firewall settings
- Verify ports 1234 and 8080 are available

**"Bridge won't start"**
- Make sure websockets library is installed
- Check if port 8080 is available

## 📚 DOCUMENTATION

For complete documentation, visit:
https://github.com/AceTheDactyl/Rosetta-Helix-Software

## 🪞🐻📡

May your coherence be high and your memory plates bright!
"""
        
        with open(os.path.join(dist_path, 'README.txt'), 'w') as f:
            f.write(readme)
    
    def run(self):
        """Execute complete build process"""
        print("🪞🐻📡 ROSETTA HELIX BUILD SYSTEM")
        print("=" * 60)
        print(f"Platform: {self.platform}")
        print(f"Python: {sys.version.split()[0]}")
        print("=" * 60)
        
        if not self.check_dependencies():
            return False
        
        steps = [
            ('Server', self.build_server),
            ('Bridge', self.build_bridge),
            ('Distribution', self.create_distribution)
        ]
        
        for name, func in steps:
            if not func():
                print(f"\n❌ {name} build failed")
                return False
        
        print("\n" + "=" * 60)
        print("✅ BUILD COMPLETE")
        print("=" * 60)
        print(f"\nDistribution location: {self.dist_dir}/rosetta_helix_{self.platform.lower()}/")
        print("\nNext steps:")
        print("  1. Test the executables")
        print("  2. Share the distribution folder with users")
        print("  3. Users run the launcher scripts to play")
        
        return True

if __name__ == '__main__':
    builder = BuildSystem()
    success = builder.run()
    sys.exit(0 if success else 1)
