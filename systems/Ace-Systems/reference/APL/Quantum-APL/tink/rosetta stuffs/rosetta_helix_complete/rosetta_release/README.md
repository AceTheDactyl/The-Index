# 🪞🐻📡 ROSETTA HELIX MUD - EXECUTABLE PACKAGE

**Self-Modifying AI-Powered MUD with Desktop Client Interface**

Version: 1.0  
Date: December 12, 2025  
Platform Support: Windows, Linux, macOS

---

## 📦 WHAT'S INCLUDED

This package contains everything needed to build and deploy Rosetta Helix MUD as standalone executables:

```
rosetta_release/
├── README.md                    # This file
├── BUILD_GUIDE.md              # Comprehensive build instructions
├── requirements.txt            # Python dependencies
├── build_all.py                # Master build script (RUN THIS)
│
├── server/
│   ├── rosetta_mud.py          # Main MUD server (33 KB)
│   └── build_server.py         # Server build script
│
└── client/
    ├── websocket_bridge.py     # WebSocket-to-TCP bridge
    ├── mud_client_websocket.html   # Web client (real WS)
    └── mud_client.html         # Web client (demo mode)
```

---

## ⚡ QUICK START (3 Steps)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Installs:**
- `pyinstaller` - Creates standalone executables
- `websockets` - Enables WebSocket bridge

### 2. Build Everything

```bash
python build_all.py
```

**Time:** 1-2 minutes  
**Output:** `dist/rosetta_helix_[platform]/`

### 3. Test & Distribute

```bash
cd dist/rosetta_helix_*/
./start_server.sh    # or .bat on Windows
./start_client.sh    # or .bat on Windows
```

**Done!** Your executables are ready.

---

## 🎯 FOR END USERS

After building, the `dist/` folder contains a complete, standalone distribution:

**What Users Get:**
- ✅ Server executable (no Python needed)
- ✅ Client launcher (opens in browser)
- ✅ README with instructions
- ✅ One-click startup scripts

**What Users Do:**
1. Extract the zip
2. Double-click launcher
3. Play!

**System Requirements:**
- Windows 10+, Linux, or macOS 10.14+
- 50 MB RAM
- Ports 1234 & 8080 available

---

## 🏗️ ARCHITECTURE OVERVIEW

### System Components

**Server (rosetta_mud.py)**
- Pure Python TCP socket server
- Kuramoto oscillator "heart" system (60 nodes)
- Holographic memory "brain" system (20 plates)
- Multi-user support (100+ concurrent)
- World building capabilities
- Self-modification interface

**Bridge (websocket_bridge.py)**
- WebSocket ↔ TCP protocol translator
- Enables browser-based clients
- Rate limiting (100 msg/sec)
- Automatic reconnection
- Bidirectional message relay

**Client (mud_client_websocket.html)**
- Modern web interface
- Real-time AI coherence visualization
- Command history
- Connection state management
- No additional dependencies

### Data Flow

```
Player Browser
    ↕️ (WebSocket)
Bridge Executable
    ↕️ (TCP Socket)
Server Executable
    ↕️
World State + AI Systems
```

---

## 🔨 BUILD PROCESS EXPLAINED

### What build_all.py Does

**Phase 1: Dependency Verification**
- Checks Python version ≥ 3.8
- Verifies PyInstaller installed
- Verifies websockets library
- Exits early if missing dependencies

**Phase 2: Server Build**
- Runs `server/build_server.py`
- Packages rosetta_mud.py with PyInstaller
- Bundles Python interpreter
- Creates `rosetta_server` executable (~20-25 MB)
- No external dependencies required

**Phase 3: Bridge Build**
- Packages websocket_bridge.py
- Includes websockets library
- Creates `rosetta_bridge` executable (~15-20 MB)
- Self-contained, no Python needed

**Phase 4: Distribution Assembly**
- Creates platform-specific folder
- Copies all executables
- Generates launcher scripts
- Creates user README.txt
- Sets correct permissions

**Total Build Time:** 1-2 minutes  
**Total Output Size:** 35-55 MB (uncompressed)

### Platform-Specific Notes

**Windows:**
- .exe executables
- .bat launcher scripts
- May trigger antivirus (false positive)
- Add Windows Defender exception if needed

**Linux:**
- ELF binaries
- .sh launcher scripts
- Requires executable permissions (auto-set)
- May need firewall rules for ports

**macOS:**
- Mach-O binaries
- .sh launcher scripts
- May need Gatekeeper approval
- Run: `xattr -d com.apple.quarantine rosetta_*`

---

## 📊 TECHNICAL SPECIFICATIONS

### Server Capabilities

**Performance:**
- CPU: <1% idle, 5-10% at 50 players
- RAM: 10 MB base + 5 KB per NPC
- Network: 1 KB/sec per active player
- Disk: Minimal (world saves ~50 KB)

**AI Systems:**
- 60 Kuramoto oscillators per Level 60+ entity
- 20 GHMP memory plates per brain
- Real-time coherence calculation
- Energy tracking and dissipation

**Game Features:**
- 5 default rooms (expandable)
- Dynamic world building
- NPC creation with AI
- Item system
- Multiplayer chat
- Level-based permissions

### Bridge Specifications

**Protocol Translation:**
- WebSocket (client-facing)
- TCP socket (server-facing)
- JSON message encapsulation
- UTF-8 encoding handling
- Partial message buffering

**Safety Features:**
- Rate limiting (100 msg/sec/client)
- Connection timeout (10 seconds)
- Automatic reconnection
- Error handling and logging
- Graceful shutdown

### Client Features

**User Interface:**
- Responsive design
- Coherence visualization
- Command history
- Quick-command buttons
- Real-time status updates

**Connection Management:**
- Automatic reconnection (up to 10 attempts)
- Exponential backoff (5s → 30s max)
- Connection state indicators
- Error message display

---

## 🚀 DEPLOYMENT OPTIONS

### Option 1: Direct Distribution

**For:** Small user base, simple setup

```bash
cd dist
zip -r rosetta_helix.zip rosetta_helix_*/
# Share the zip file
```

**Users:** Extract and run launcher

### Option 2: Installer Package

**For:** Professional distribution, easier installation

**Windows:** Use Inno Setup (see BUILD_GUIDE.md)
**Linux:** Create .deb/.rpm package
**macOS:** Create .dmg disk image

**Result:** Single installer file, desktop shortcuts

### Option 3: Cloud Deployment

**For:** Centralized hosting, web-only access

**Steps:**
1. Deploy server on VPS
2. Deploy bridge with HTTPS
3. Serve client via web server
4. Users access via URL

**Providers:** DigitalOcean, AWS, Heroku, etc.

---

## 🐛 TROUBLESHOOTING

### Build Failures

**"PyInstaller not found"**
```bash
Solution: pip install pyinstaller
```

**"Build completed but executable won't run"**
```bash
Possible causes:
1. Antivirus quarantine (check AV logs)
2. Missing shared libraries (Linux)
3. Permission issues (run as admin/sudo)

Debug: Run executable from terminal to see errors
```

**"Executable too large (>100 MB)"**
```bash
This is normal for first build.
Optimize with:
1. UPX compression: --upx-dir flag
2. Exclude unused modules
3. Strip debug symbols (auto-enabled)
```

### Runtime Errors

**"Port already in use"**
```bash
# Find process using port
lsof -i :1234       # Linux/Mac
netstat -ano | findstr :1234   # Windows

# Kill process
kill -9 <PID>       # Linux/Mac
taskkill /PID <PID> /F    # Windows
```

**"Cannot connect to server"**
```bash
Checklist:
1. Server running? Check process list
2. Correct port? Default is 1234
3. Firewall blocking? Add exception
4. Bridge running? Need both server & bridge
```

### User Issues

**"Antivirus blocking executable"**
```
This is a false positive (common with PyInstaller).
Solutions:
1. Add exception for the file
2. Verify hash matches official release
3. Build from source yourself
```

**"Client won't connect"**
```
1. Check browser console for errors (F12)
2. Verify bridge is running (check port 8080)
3. Try different browser
4. Check if HTTPS/HTTP mixed content blocking
```

---

## 📈 PERFORMANCE TUNING

### Optimize Executable Size

**Current:** 35-55 MB  
**Target:** 20-30 MB

**Method 1: UPX Compression**
```python
# Add to build scripts:
'--upx-dir', '/path/to/upx'
# Reduces by 40-60%
```

**Method 2: Exclude Modules**
```python
# Add if not using certain features:
'--exclude-module', 'tkinter'
'--exclude-module', 'PIL'
```

### Optimize Runtime Performance

**Server:**
```python
# Reduce AI tick rate for many NPCs
if self.tick_count >= 50:  # Was 10 (slower, less CPU)
```

**Bridge:**
```python
# Increase buffer size for high traffic
data = await sock_recv(self.tcp_socket, 8192)  # Was 4096
```

---

## 🔐 SECURITY CONSIDERATIONS

### For Development

**Current State:**
- ✅ No remote code execution
- ✅ Input validation
- ✅ Rate limiting (bridge)
- ❌ No authentication
- ❌ No encryption
- ❌ No audit logging

**Production Requirements:**
1. Add user authentication
2. Implement TLS/SSL
3. Add admin commands
4. Enable audit logging
5. Input sanitization (enhanced)
6. Session management

### For Deployment

**Public Internet:**
- ⚠️ NOT RECOMMENDED without security hardening
- Use reverse proxy (nginx) with rate limiting
- Add authentication layer
- Enable HTTPS for WebSocket
- Monitor for abuse

**Local Network:**
- ✅ Safe for trusted users
- Firewall rules recommended
- Consider VPN for remote access

---

## 📚 ADDITIONAL RESOURCES

### Documentation

- **BUILD_GUIDE.md** - Complete build instructions with troubleshooting
- **Server Code** - `rosetta_mud.py` (well-commented, 966 lines)
- **Bridge Code** - `websocket_bridge.py` (fully documented)
- **Client Code** - `mud_client_websocket.html` (inline documentation)

### Learning Resources

**Understanding Kuramoto Oscillators:**
- Paper: "Pulse-Driven Coherent Memory Architecture"
- See: `server/rosetta_mud.py` lines 54-76 (Heart class)

**Understanding GHMP Plates:**
- See: `server/rosetta_mud.py` lines 24-52 (Brain class)
- Geometric Holographic Memory Plates concept

**WebSocket Protocol:**
- RFC 6455: https://tools.ietf.org/html/rfc6455
- See: `websocket_bridge.py` for implementation

### Community

- **GitHub:** https://github.com/AceTheDactyl/Rosetta-Helix-Software
- **Email:** collective@rosettabear.org
- **Issues:** Use GitHub Issues for bug reports

---

## ✅ FINAL CHECKLIST

Before distributing to users:

**Build Quality:**
- [ ] Builds successfully on clean system
- [ ] All executables under 30 MB (with optimization)
- [ ] No build warnings or errors
- [ ] PyInstaller version documented

**Testing:**
- [ ] Server starts without errors
- [ ] Bridge connects to server
- [ ] Client connects to bridge
- [ ] Commands execute correctly
- [ ] AI systems functional (coherence, memory)
- [ ] Multiplayer works (2+ clients)
- [ ] Tested on all target platforms

**Documentation:**
- [ ] README.txt in distribution
- [ ] Version number set
- [ ] Known issues documented
- [ ] Support contact included
- [ ] License file included

**Distribution:**
- [ ] Zip file created
- [ ] File size reasonable (<60 MB)
- [ ] No development files included
- [ ] Tested extraction and launch
- [ ] Platform clearly marked

---

## 🎓 LEARNING OPPORTUNITIES

### For Students

This project demonstrates:
- Socket programming (server/client architecture)
- Protocol translation (WebSocket ↔ TCP)
- Executable packaging (PyInstaller)
- Cross-platform development
- AI system implementation (Kuramoto + GHMP)
- Game architecture (MUD systems)

### For Developers

Extend with:
- Database integration (persist world state)
- Combat system implementation
- Quest and achievement systems
- Admin dashboard (web-based)
- Metrics and monitoring
- Distributed server architecture

### For Researchers

Explore:
- Kuramoto oscillator synchronization dynamics
- Holographic memory plate effectiveness
- Multi-agent AI coordination
- Emergent behavior in NPCs
- Player-AI interaction patterns

---

## 🌟 FEATURES AT A GLANCE

**Core Systems:**
- ✅ Multi-user TCP server
- ✅ WebSocket bridge for browsers
- ✅ AI-powered NPCs (Kuramoto + GHMP)
- ✅ Dynamic world building
- ✅ Real-time coherence tracking
- ✅ Cross-platform executables

**User Experience:**
- ✅ Web-based client (no install)
- ✅ Telnet client support (classic)
- ✅ One-click launchers
- ✅ Visual AI feedback
- ✅ Command history
- ✅ Quick-command buttons

**Technical:**
- ✅ No dependencies (standalone)
- ✅ Low resource usage
- ✅ Hot-reload capable
- ✅ JSON world save/load
- ✅ Extensible architecture
- ✅ Self-modification interface

---

## 🎉 YOU'RE READY!

Everything needed to build, test, and distribute Rosetta Helix MUD is included in this package.

**Next Steps:**

1. **Right Now:** Run `python build_all.py`
2. **5 Minutes:** Test the executables
3. **10 Minutes:** Share with your first user

**Questions?** Check BUILD_GUIDE.md or open a GitHub issue.

**May your coherence be high and your memory plates bright!**

🪞🐻📡

---

**END OF README**
