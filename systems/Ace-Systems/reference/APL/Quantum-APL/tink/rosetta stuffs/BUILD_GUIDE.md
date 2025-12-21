# ROSETTA HELIX MUD - BUILD & DEPLOYMENT GUIDE

## 🎯 QUICK START

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- 100 MB free disk space

### Build Steps

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Build everything
python build_all.py

# 3. Test the build
cd dist/rosetta_helix_*/
./start_server.sh  # or start_server.bat on Windows
./start_client.sh  # or start_client.bat on Windows
```

That's it! Your executables are ready to distribute.

---

## 📁 PROJECT STRUCTURE

```
rosetta_release/
├── build_all.py              # Master build script
├── requirements.txt          # Python dependencies
│
├── server/
│   ├── rosetta_mud.py        # Main MUD server
│   └── build_server.py       # Server build script
│
├── client/
│   ├── websocket_bridge.py   # WebSocket-to-TCP bridge
│   └── mud_client_websocket.html  # Web client interface
│
└── dist/                     # Output directory
    └── rosetta_helix_*/      # Platform-specific distribution
        ├── rosetta_server*   # Server executable
        ├── rosetta_bridge*   # Bridge executable
        ├── mud_client.html   # Client interface
        ├── start_server.*    # Server launcher
        ├── start_client.*    # Client launcher
        └── README.txt        # User instructions
```

---

## 🔨 BUILD PROCESS DETAILS

### What build_all.py Does

1. **Dependency Check**
   - Verifies Python version (3.8+)
   - Checks for PyInstaller
   - Checks for websockets library

2. **Server Build**
   - Packages rosetta_mud.py into standalone executable
   - Bundles all Python dependencies
   - Size: ~15-25 MB

3. **Bridge Build**
   - Packages websocket_bridge.py into standalone executable
   - Includes websockets library
   - Size: ~10-20 MB

4. **Distribution Package**
   - Creates platform-specific folder
   - Copies all executables
   - Generates launcher scripts
   - Creates user README

### Build Outputs by Platform

**Windows:**
```
rosetta_helix_windows/
├── rosetta_server.exe      # Server executable
├── rosetta_bridge.exe      # Bridge executable
├── mud_client.html         # Client interface
├── start_server.bat        # Server launcher
├── start_client.bat        # Client launcher
└── README.txt              # User guide
```

**Linux:**
```
rosetta_helix_linux/
├── rosetta_server          # Server executable
├── rosetta_bridge          # Bridge executable
├── mud_client.html         # Client interface
├── start_server.sh         # Server launcher
├── start_client.sh         # Client launcher
└── README.txt              # User guide
```

**macOS:**
```
rosetta_helix_darwin/
├── rosetta_server          # Server executable
├── rosetta_bridge          # Bridge executable
├── mud_client.html         # Client interface
├── start_server.sh         # Server launcher
├── start_client.sh         # Client launcher
└── README.txt              # User guide
```

---

## 🧪 TESTING THE BUILD

### Server Test
```bash
# Start server
cd dist/rosetta_helix_*/
./rosetta_server

# Expected output:
# 🪞🐻📡 ROSETTA MUD - STARTING
# ✅ Brain system initialized
# ✅ Heart system initialized
# ✅ World builder initialized
# 🎧 Server listening on 0.0.0.0:1234
```

### Bridge Test
```bash
# In another terminal
./rosetta_bridge

# Expected output:
# INFO - 🔌 Starting WebSocket bridge on 0.0.0.0:8080
# INFO - 🔗 Forwarding to MUD server at localhost:1234
```

### Client Test
```bash
# Open mud_client.html in browser
# Should see:
# - Green "🟢 Connected" status
# - Connection successful message
# - Can type commands
```

### Integration Test
```bash
# In client browser window:
create TestPlayer
look
go north
stats
meditate

# Verify:
# - Commands execute without errors
# - Server responds appropriately
# - AI coherence updates
```

---

## 🐛 TROUBLESHOOTING

### Build Issues

**Error: "ModuleNotFoundError: No module named 'PyInstaller'"**
```bash
Solution: pip install pyinstaller
```

**Error: "ModuleNotFoundError: No module named 'websockets'"**
```bash
Solution: pip install websockets
```

**Error: "Permission denied" (Linux/Mac)**
```bash
Solution: chmod +x build_all.py
```

**Warning: "Executable suspiciously small"**
```bash
This usually means build failed silently.
Check: dist/rosetta_helix_*/build/ for error logs
```

### Runtime Issues

**Server won't start: "Address already in use"**
```bash
# Find and kill existing process
# Linux/Mac:
lsof -i :1234
kill -9 <PID>

# Windows:
netstat -ano | findstr :1234
taskkill /PID <PID> /F
```

**Bridge won't start: "Cannot import websockets"**
```bash
The bridge executable should be self-contained.
If this error occurs, rebuild with:
pip install --upgrade websockets
python build_all.py
```

**Client won't connect: "WebSocket connection failed"**
```bash
Checklist:
1. Is server running? (check port 1234)
2. Is bridge running? (check port 8080)
3. Firewall blocking? (allow ports)
4. Correct URL? (should be ws://localhost:8080)
```

**Antivirus blocking executable**
```bash
PyInstaller executables sometimes trigger false positives.
Solutions:
1. Add exception for the executable
2. Build with --upx-exclude to avoid compression
3. Submit false positive report to antivirus vendor
```

---

## 📊 SIZE OPTIMIZATION

### Current Sizes (Typical)
- Server: 20-30 MB
- Bridge: 15-25 MB
- Client: 0.01 MB (HTML)
- **Total: 35-55 MB**

### Optimization Options

**1. Enable UPX Compression**
```python
# In build_server.py and build_all.py, add:
args.append('--upx-dir=/path/to/upx')
# Reduces size by 40-60%
# Download UPX from: https://upx.github.io/
```

**2. Strip Debug Symbols**
```python
# Already enabled in build scripts
args.append('--strip')
```

**3. Exclude Unused Modules**
```python
# Add to build scripts if you know what's unused:
args.extend(['--exclude-module', 'matplotlib'])
args.extend(['--exclude-module', 'numpy'])
```

**Expected After Full Optimization:**
- Server: 10-15 MB
- Bridge: 8-12 MB
- Total: 18-27 MB

---

## 🚀 DISTRIBUTION

### For End Users

1. **Zip the distribution folder:**
```bash
cd dist
zip -r rosetta_helix.zip rosetta_helix_*/
# or on Windows: use built-in "Compress to ZIP"
```

2. **Upload to:**
   - GitHub Releases
   - Your website
   - File sharing service

3. **Provide:**
   - Download link
   - Platform requirements (Windows 10+, Linux, macOS 10.14+)
   - README.txt contents
   - Support contact

### For Developers

1. **Share source:**
```bash
# Package source code
tar czf rosetta_helix_source.tar.gz \
  server/ client/ build_all.py requirements.txt
```

2. **GitHub Release:**
```bash
# Tag version
git tag -a v1.0.0 -m "Version 1.0.0"
git push origin v1.0.0

# Attach built executables to release
```

---

## 🔧 CUSTOMIZATION

### Change Server Port

Edit `rosetta_mud.py`:
```python
# Line ~960
if __name__ == "__main__":
    server = MudServer(port=4321)  # Changed from 1234
```

Rebuild:
```bash
python build_all.py
```

### Change Bridge Port

Edit `websocket_bridge.py`:
```python
# Line ~150
bridge = WebSocketBridge(
    ws_port=8888,  # Changed from 8080
    mud_port=4321   # Match server port if changed
)
```

### Customize Client Appearance

Edit `mud_client_websocket.html` CSS section (lines 8-150):
```css
body {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    color: #00ff88;  /* Change colors here */
}
```

---

## 📦 ADVANCED: CREATE INSTALLER

### Windows Installer (Inno Setup)

1. **Install Inno Setup:**
   - Download from: https://jrsoftware.org/isinfo.php

2. **Create installer script:**
```inno
[Setup]
AppName=Rosetta Helix MUD
AppVersion=1.0
DefaultDirName={autopf}\RosettaMUD
OutputDir=installers
OutputBaseFilename=RosettaMUD_Setup

[Files]
Source: "dist\rosetta_helix_windows\*"; DestDir: "{app}"

[Icons]
Name: "{group}\Rosetta MUD"; Filename: "{app}\start_client.bat"
```

3. **Compile:**
```bash
iscc installer.iss
```

### Linux Package (.deb)

1. **Create package structure:**
```bash
mkdir -p rosetta-mud_1.0/DEBIAN
mkdir -p rosetta-mud_1.0/opt/rosetta-mud
mkdir -p rosetta-mud_1.0/usr/share/applications
```

2. **Create control file:**
```
Package: rosetta-mud
Version: 1.0
Architecture: amd64
Maintainer: Your Name <email@example.com>
Description: AI-powered MUD game
```

3. **Build package:**
```bash
dpkg-deb --build rosetta-mud_1.0
```

### macOS App Bundle

1. **Create app structure:**
```bash
mkdir -p RosettaMUD.app/Contents/MacOS
mkdir -p RosettaMUD.app/Contents/Resources
```

2. **Create Info.plist**
3. **Package with:**
```bash
hdiutil create -volname "Rosetta MUD" -srcfolder RosettaMUD.app \
  -ov -format UDZO RosettaMUD.dmg
```

---

## 🎓 UNDERSTANDING THE BUILD

### PyInstaller Process

1. **Analysis Phase**
   - Scans Python script for imports
   - Identifies all dependencies
   - Creates dependency graph

2. **Collection Phase**
   - Gathers all required .py files
   - Collects shared libraries
   - Includes data files

3. **Bundling Phase**
   - Creates bootloader
   - Packages Python interpreter
   - Compresses everything

4. **Executable Creation**
   - Combines all components
   - Creates single file or directory
   - Sets execution permissions

### WebSocket Bridge Architecture

```
Browser Client
    ↓
WebSocket (port 8080)
    ↓
Bridge Process
    ↓
TCP Socket (port 1234)
    ↓
MUD Server
```

**Why Bridge Needed:**
- Browsers can't directly connect to TCP sockets
- WebSocket is browser-compatible
- Bridge translates between protocols

---

## 📈 PERFORMANCE METRICS

### Build Time
- Server: 30-60 seconds
- Bridge: 20-40 seconds
- Total: 1-2 minutes

### Runtime Performance
- Server RAM: 10-50 MB (depending on players)
- Bridge RAM: 5-15 MB
- Client RAM: Browser-dependent (20-50 MB)
- CPU: Minimal (<1% on modern systems)

### Network Usage
- Per player: ~1 KB/s
- 100 players: ~100 KB/s
- Bridge overhead: <5%

---

## ✅ PRODUCTION CHECKLIST

Before distributing to end users:

- [ ] Built on clean system
- [ ] Tested on target platforms
- [ ] No hardcoded development paths
- [ ] README.txt included
- [ ] Version number set
- [ ] Changelog created
- [ ] Known issues documented
- [ ] Support contact provided
- [ ] License file included
- [ ] Tested with firewall enabled
- [ ] Tested with antivirus enabled
- [ ] Verified no dependencies required
- [ ] Cross-platform testing complete

---

## 🆘 GETTING HELP

### Build Issues
1. Check error messages in terminal
2. Verify dependencies installed: `pip list`
3. Try clean rebuild: `rm -rf build dist && python build_all.py`

### Runtime Issues
1. Check server logs
2. Verify ports available
3. Test with telnet first: `telnet localhost 1234`

### Community Support
- GitHub Issues: https://github.com/AceTheDactyl/Rosetta-Helix-Software
- Email: collective@rosettabear.org

---

## 🎉 SUCCESS!

If you've made it this far, you now have:
- ✅ Standalone server executable
- ✅ WebSocket bridge executable
- ✅ Web client interface
- ✅ Complete distribution package
- ✅ Ready-to-share release

**Congratulations!** 🪞🐻📡

Users can now run your MUD without installing Python or any dependencies.

---

**END OF BUILD GUIDE**
