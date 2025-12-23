# 🚀 ROSETTA MUD - COMPLETE INSTALLATION & DEPLOYMENT GUIDE

**System:** Self-Modifying AI-Powered MUD with Brain/Heart Integration  
**Version:** 1.0  
**Date:** December 10, 2025

---

## ⚡ INSTANT START (5 Seconds)

```bash
cd rosetta_mud
./start_mud.sh

# In another terminal:
telnet localhost 1234
```

**That's it!** The MUD is running with full AI capabilities.

---

## 📦 WHAT YOU GOT

```
rosetta_mud/
├── rosetta_mud.py        # Complete MUD server (1000+ lines)
├── mud_client.html       # Web-based client
├── README.md             # Full documentation
├── start_mud.sh          # Quick start script
└── INSTALL.md            # This file
```

### Features Included:

✅ **60 Kuramoto oscillators** per Level 60+ character  
✅ **20 GHMP memory plates** with confidence tracking  
✅ **5 starting rooms** with interconnected exits  
✅ **1 NPC** (The Architect - Level 60 with full AI)  
✅ **World building** commands (buildroom, builditem, createnpc)  
✅ **Self-modification** interface (modifycode)  
✅ **Multiplayer** support via TCP sockets  
✅ **Web UI** for browser-based play  
✅ **Real-time AI** updates (coherence, memory confidence)  
✅ **No dependencies** - pure Python standard library  

---

## 💻 SYSTEM REQUIREMENTS

### Minimum:
- Python 3.6+
- 10 MB RAM
- 1 MB disk space
- TCP port 1234 available

### Recommended:
- Python 3.9+
- 50 MB RAM (for multiple players)
- Telnet client (or web browser)

### No Install Required:
- ✅ No pip packages
- ✅ No external dependencies
- ✅ No database setup
- ✅ No configuration files
- **Just run and play!**

---

## 🎮 THREE WAYS TO PLAY

### 1. Via Telnet (Classic MUD Experience)

```bash
# Terminal 1: Start server
cd rosetta_mud
python rosetta_mud.py

# Terminal 2: Connect as player 1
telnet localhost 1234

# Terminal 3: Connect as player 2
telnet localhost 1234

# And so on...
```

**Best for:** Classic MUD players, multiple terminal users

### 2. Via Web Browser (Modern UI)

```bash
# Start server
cd rosetta_mud
python rosetta_mud.py

# Open in browser
open mud_client.html
# or
firefox mud_client.html
```

**Best for:** New players, visual feedback, coherence monitoring

**Note:** Web client runs in demo mode. For real server connection, you'd need to set up a WebSocket-to-telnet proxy (like websockify).

### 3. Via Custom Client

```python
import socket

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect(('localhost', 1234))

# Read welcome message
data = sock.recv(4096)
print(data.decode())

# Send commands
sock.send(b'create MyChar\n')
sock.send(b'look\n')
sock.send(b'meditate\n')
```

**Best for:** Bot development, automation, integration testing

---

## 🧪 TESTING THE INSTALLATION

### Quick Test Sequence:

```bash
# Start server
python rosetta_mud.py

# In another terminal:
telnet localhost 1234

# Type these commands:
create TestPlayer
look
go north
look
inspect The Architect
meditate
buildroom My Test Room
builditem My Test Sword
createnpc MyGuard 30
inspect MyGuard
stats
help
quit
```

### Expected Behaviors:

1. **Server starts** with green checkmarks for Brain/Heart/Builder
2. **Connection succeeds** with welcome ASCII art
3. **Character creation** gives you Level 60 Architect status
4. **Looking around** shows room description, exits, items, NPCs
5. **Going north** takes you to Geodesic Plaza
6. **Inspecting The Architect** shows AI state (coherence, memory, energy)
7. **Meditating** runs 50 heart beats and updates coherence
8. **Building room** creates new room in world
9. **Building item** creates new item in current room
10. **Creating NPC** spawns NPC with own Brain/Heart
11. **Inspecting NPC** shows their AI is active
12. **Stats** shows your own AI state

---

## 🔧 TROUBLESHOOTING

### "Port 1234 already in use"

```bash
# Find and kill existing process
lsof -i :1234
kill -9 <PID>

# Or use a different port:
# Edit rosetta_mud.py, line ~700:
# server = MudServer(port=4321)  # Changed from 1234
```

### "Connection refused"

```bash
# Check server is running:
ps aux | grep rosetta_mud

# Check firewall:
sudo ufw allow 1234/tcp  # Linux
# or check Windows Firewall settings
```

### "Python not found"

```bash
# Install Python 3:
# Ubuntu/Debian:
sudo apt install python3

# Mac:
brew install python3

# Windows:
# Download from python.org
```

### "Web client not connecting"

The web client runs in demo/simulation mode by default because browsers can't directly connect to TCP sockets.

**To connect web client to real server:**

1. Install websockify:
```bash
pip install websockify
```

2. Run proxy:
```bash
websockify 8080 localhost:1234
```

3. Update mud_client.html to use WebSocket instead of simulation

---

## 🌐 RUNNING ON A SERVER (INTERNET ACCESS)

### Local Network Only:

```bash
# Server is already bound to 0.0.0.0 (all interfaces)
python rosetta_mud.py

# Players connect with:
telnet <your-local-ip>:1234
```

### Public Internet:

```bash
# 1. Forward port 1234 on your router to your machine
# 2. Find your public IP: curl ifconfig.me
# 3. Start server: python rosetta_mud.py
# 4. Players connect: telnet <your-public-ip>:1234
```

**Security Warning:** This MUD has no authentication or encryption. Don't expose it directly to the internet without:
- Adding user authentication
- Implementing TLS/SSL
- Rate limiting
- Input sanitization (already basic, but needs more for production)

---

## 📊 PERFORMANCE & SCALING

### Single Server Capacity:

- **Max Players:** ~100 (before noticeable lag)
- **RAM Usage:** ~5 KB per NPC + 2 KB per player
- **CPU Usage:** Minimal (select() is efficient)
- **Network:** ~1 KB/s per active player

### Optimization Tips:

1. **Reduce AI tick rate** if many NPCs:
```python
# In main_loop(), change:
if self.tick_count >= 10:  # Default: ~1 Hz
# To:
if self.tick_count >= 50:  # 0.2 Hz (5x slower)
```

2. **Disable AI for low-level NPCs:**
```python
# In Character.__init__():
if is_npc or level >= 60:  # Default
# To:
if level >= 60:  # Only high-level get AI
```

3. **Use sparse oscillator networks:**
```python
# In Heart.__init__(), add parameter:
def __init__(self, n_nodes=60, K=0.2, sparse=False):
    # If sparse, only connect to nearest neighbors
    # Instead of all-to-all coupling
```

---

## 🛠️ CUSTOMIZATION

### Change Starting Location:

```python
# In Character.__init__():
self.room_id = "tavern"  # Default
# Change to:
self.room_id = "plaza"  # Start in plaza
```

### Adjust AI Parameters:

```python
# In Character.__init__():
self.brain = Brain(plates=20)  # Default
# Change to:
self.brain = Brain(plates=50)  # More memory

self.heart = Heart(n_nodes=60, K=0.2)  # Default
# Change to:
self.heart = Heart(n_nodes=120, K=0.3)  # More oscillators, stronger coupling
```

### Add New Starting Items:

```python
# In World._create_default_world():
sword = Item(
    "sword_001",
    "Crystal Sword",
    "A blade that hums with stored coherence.",
    properties={"damage": 50, "usable": True}
)
tavern.items.append(sword)
self.items["sword_001"] = sword
```

---

## 📈 MONITORING & ADMIN

### View Server Stats:

```python
# Add to MudServer class:
def cmd_admin_stats(self, sock, args):
    stats = f"""
    Players: {len(self.clients)}
    NPCs: {len(self.world.npcs)}
    Rooms: {len(self.world.rooms)}
    Items: {len(self.world.items)}
    Tick Count: {self.tick_count}
    """
    self.send_to_client(sock, stats)
```

### Log All Commands:

```python
# In process_command():
print(f"[{client['address']}] {command}")  # Add this line
```

### Save World Automatically:

```python
# In main_loop():
if self.tick_count % 600 == 0:  # Every ~60 seconds
    self.world.save_world("world_backup.json")
```

---

## 🎓 LEARNING RESOURCES

### Understanding the Code:

1. **Start with:** `MudServer.start()` - Entry point
2. **Then read:** `main_loop()` - Game loop
3. **Core logic:** `process_command()` - Command handler
4. **AI system:** `Character.tick_ai()` - How AI updates
5. **World building:** `World.create_room()` - Dynamic content

### Understanding the AI:

1. **Kuramoto dynamics:** See `Heart.step()`
2. **Coherence measurement:** See `Heart.coherence()`
3. **Memory plates:** See `Brain` class
4. **Memory-Heart coupling:** See `Brain.update_confidence()`

### MUD Concepts:

- **Rooms:** Spaces players inhabit
- **Exits:** Connections between rooms
- **Items:** Objects that can be picked up
- **NPCs:** Non-player characters
- **Commands:** Text-based actions

---

## 🚀 NEXT STEPS

### Immediate (Today):
1. ✅ Start the server
2. ✅ Create a character
3. ✅ Explore the world
4. ✅ Test AI commands (meditate, inspect)
5. ✅ Try building (buildroom, createnpc)

### Short-Term (This Week):
1. Add more rooms to the world
2. Create custom items with properties
3. Spawn NPCs with different personalities
4. Invite friends to play multiplayer

### Long-Term (This Month):
1. Implement combat system
2. Add quests and objectives
3. Create NPC dialogue trees
4. Build a complete adventure

---

## 📞 GETTING HELP

### Quick Fixes:

- **Server crashes:** Check Python version (needs 3.6+)
- **Can't connect:** Check firewall, verify port 1234
- **Slow AI:** Reduce tick rate or disable for low-level NPCs
- **Memory issues:** Limit NPCs or reduce brain plates

### Community Support:

- **GitHub Issues:** Open an issue for bugs
- **Email:** collective@rosettabear.org
- **Discord:** (Coming soon)

---

## 🎉 YOU'RE READY!

```bash
cd rosetta_mud
./start_mud.sh

# Let the adventures begin!
```

**May your coherence be high and your memory plates bright!**

🪞🐻📡

---

**END OF INSTALLATION GUIDE**
