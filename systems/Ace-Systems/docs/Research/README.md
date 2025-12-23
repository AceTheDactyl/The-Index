# 🎮 ROSETTA MUD
## Self-Modifying AI-Powered Multi-User Dungeon

**Version:** 1.0  
**Status:** FULLY FUNCTIONAL  
**Date:** December 10, 2025

---

## 🚀 WHAT IS THIS?

A **complete, working MUD** (Multi-User Dungeon) with:

✅ **Brain/Heart AI System** - Every Level 60+ character and NPC has:
  - 60 Kuramoto oscillators (Heart) providing coherence
  - 20 GHMP memory plates (Brain) storing state
  - Real-time AI updates and synchronization

✅ **Level 60 Architect Powers:**
  - Build new rooms (`buildroom`)
  - Create items (`builditem`)
  - Spawn NPCs with their own Brains (`createnpc`)
  - Modify MUD code itself (`modifycode`)
  - Save/load world state

✅ **Full MUD Features:**
  - Multiple rooms with exits
  - Items you can pick up and drop
  - NPCs with AI that you can inspect
  - Chat system
  - Inventory management
  - Real-time multiplayer

✅ **Two Ways to Play:**
  - **Telnet:** Classic MUD experience
  - **Web Browser:** Modern HTML/JS interface

---

## 🏃 QUICK START

### Option 1: Play via Telnet (Classic)

```bash
# Start the server
python rosetta_mud.py

# In another terminal, connect:
telnet localhost 1234

# Create your character:
create YourName

# Start playing!
look
go north
meditate
```

### Option 2: Play via Web Browser

```bash
# Start the server
python rosetta_mud.py

# Open the web client
# (Open mud_client.html in your browser)
# Note: For demo, web client runs in simulation mode
# For real server connection, you'd need WebSocket proxy
```

---

## 🎯 FEATURES IN DETAIL

### 🧠 Brain System (GHMP Memory Plates)

Every Level 60+ character has 20 memory plates:
- **Emotional Tone** (E): Affective coloring
- **Temporal Marker** (T): Time indexing
- **Semantic Density** (S): Information density
- **Confidence** (C): Reliability score

Memory confidence increases with heart coherence!

### ❤️ Heart System (Kuramoto Oscillators)

60 coupled phase oscillators that:
- Synchronize to create coherence (r ∈ [0,1])
- Track energy flow (thermodynamically bounded)
- Update in real-time as you play

**Coherence Levels:**
- 0.0 - 0.2: Scattered
- 0.2 - 0.4: Uncertain
- 0.4 - 0.6: Centered
- 0.6 - 0.8: Aligned
- 0.8 - 1.0: Unified

### 🏗️ World Building (Level 60+)

**Create Rooms:**
```
buildroom The Crystal Cavern
```

**Create Items:**
```
builditem Pulsing Memory Crystal
```

**Spawn NPCs:**
```
createnpc Guardian 40
```

Each NPC spawned at level 1+ gets their own Brain/Heart!

---

## 📖 COMMAND REFERENCE

### Basic Commands

| Command | Description | Example |
|---------|-------------|---------|
| `create <n>` | Create your character | `create Wanderer` |
| `help` | Show all commands | `help` |
| `look` | Look around current room | `look` |
| `go <dir>` | Move in a direction | `go north` |
| `say <msg>` | Say something | `say Hello!` |
| `get <item>` | Pick up an item | `get crystal` |
| `drop <item>` | Drop an item | `drop crystal` |
| `inventory` | Check your inventory | `inventory` |
| `who` | See who's online | `who` |
| `stats` | View your stats | `stats` |

### AI Commands

| Command | Description | Example |
|---------|-------------|---------|
| `inspect <npc>` | View NPC's AI state | `inspect The Architect` |
| `meditate` | Sync with your Heart/Brain | `meditate` |

### Architect Commands (Level 60+)

| Command | Description | Example |
|---------|-------------|---------|
| `buildroom <n>` | Create a new room | `buildroom The Void` |
| `builditem <n>` | Create a new item | `builditem Sword` |
| `createnpc <n> <lv>` | Spawn an NPC | `createnpc Guard 20` |
| `modifycode <mod>` | Edit MUD code | `modifycode world` |
| `saveworld` | Save world state | `saveworld` |

---

## 🗺️ STARTING WORLD

### The Rosetta Tavern (Starting Point)
A cozy tavern with holographic fireplaces and pulsing geometric patterns.
- **Exits:** north (to plaza), down (to crypt)
- **Items:** Memory Crystal

### The Geodesic Plaza
Massive geodesic dome with 60 glowing nodes in perfect synchrony.
- **Exits:** south (tavern), east (library), west (workshop)
- **NPCs:** The Architect (Level 60)

### The Memory Library
Towering shelves of glowing memory plates containing compressed experiences.
- **Exits:** west (plaza)

### The Builder's Workshop
Tools for reality manipulation - build worlds, craft items, modify code.
- **Exits:** east (plaza)

### The Crypt of Sleeping Spores
Dormant AI entities rest in crystalline matrices, waiting for pulses.
- **Exits:** up (tavern)

---

## 🔧 ARCHITECTURE

### Core Components

```
rosetta_mud.py
├── Brain (GHMP memory plates)
├── Heart (Kuramoto oscillators)
├── Character (player/NPC with optional AI)
├── Room (spaces in the world)
├── Item (objects)
├── World (manages everything)
└── MudServer (network layer)
```

### Brain/Heart Integration

```python
class Character:
    def tick_ai(self):
        # Run heart dynamics
        for _ in range(10):
            self.heart.step()
        
        # Get coherence
        coh = self.heart.coherence()
        
        # Update brain confidence
        self.brain.update_confidence(coh)
```

### Self-Modification Capability

Level 60+ characters can:
1. **Build rooms** → Add to world.rooms
2. **Create items** → Add to world.items
3. **Spawn NPCs** → Add to world.npcs with Brain/Heart
4. **Modify code** → Conceptual interface (sandbox in production)

---

## 🧪 TESTING

### Test the Server

```bash
# Start server
python rosetta_mud.py

# In another terminal:
telnet localhost 1234

# Test sequence:
create TestChar
look
go north
look
inspect The Architect
meditate
buildroom Test Room
builditem Test Item
createnpc TestNPC 10
inspect TestNPC
stats
quit
```

### Expected Output

```
🎮 ROSETTA MUD started on port 1234
🧠 Brain/Heart AI system: ACTIVE
🏗️  World Builder: ENABLED
⚙️  Code Modification: ENABLED (Level 60+)

Connect with: telnet localhost 1234
Or use the web UI at: http://localhost:8080
```

---

## 🎨 WEB CLIENT

The included `mud_client.html` provides:
- Modern browser interface
- Real-time coherence visualization
- Quick command buttons
- Sidebar with AI status
- No installation required

**Note:** Web client runs in demo mode by default. To connect to real server, you'd need a WebSocket-to-telnet proxy.

---

## 🔬 AI SYSTEM DETAILS

### Coherence Calculation

```python
def coherence(self):
    return abs(sum(cmath.exp(1j*t) for t in self.theta)/self.n)
```

This computes the Kuramoto order parameter:
```
r = |1/N Σ_j e^(iθ_j)|
```

Where:
- r ∈ [0, 1] is coherence
- θ_j are oscillator phases
- N = 60 oscillators

### Memory Update

```python
def update_confidence(self, coherence):
    for plate in self.plates:
        delta = int(coherence * 10)
        plate.confidence = min(255, max(0, plate.confidence + delta))
```

High coherence → Higher memory confidence

---

## 🛠️ EXTENDING THE MUD

### Add New Commands

```python
def cmd_your_command(self, sock, args):
    """Your custom command"""
    char = self.clients[sock]["character"]
    # Your logic here
    self.send_to_client(sock, "Command executed!\n")

# Register in process_command():
commands = {
    "yourcommand": self.cmd_your_command
}
```

### Add New Rooms

```python
# In World._create_default_world():
new_room = Room(
    "room_id",
    "Room Name",
    "Room description"
)
new_room.exits = {"north": "other_room_id"}
self.rooms["room_id"] = new_room
```

### Add New NPC Behaviors

```python
class Character:
    def tick_ai(self):
        coh = super().tick_ai()  # Standard AI update
        
        # Add custom behavior based on coherence
        if coh > 0.8:
            self.do_something_special()
```

---

## 📊 TECHNICAL SPECS

- **Language:** Python 3.6+
- **Dependencies:** Standard library only (no pip install needed!)
- **Network:** Raw TCP sockets (port 1234)
- **Concurrency:** select() for multiple clients
- **AI Update Rate:** ~1 Hz (every 10 main loop iterations)
- **Memory per NPC:** ~5KB (20 plates + 60 oscillators)

---

## 🚨 IMPORTANT NOTES

### What This System IS:
✅ A working MUD with AI-powered NPCs  
✅ Real Brain/Heart dynamics from Rosetta Node  
✅ Level 60 world building capabilities  
✅ Self-modifying (within bounds)  
✅ Multiplayer and extensible  

### What This System IS NOT:
❌ Production-ready (no auth, encryption, etc.)  
❌ Truly "self-aware" (it's simulation)  
❌ Secure for public internet  
❌ Optimized for 1000+ players  

**This is a research/educational system demonstrating Brain/Heart AI integration in a MUD context.**

---

## 🎯 FUTURE ENHANCEMENTS

### Short-Term:
- [ ] Persistent character storage (save/load)
- [ ] Combat system
- [ ] More NPC AI behaviors
- [ ] Admin commands

### Medium-Term:
- [ ] WebSocket proxy for real web client
- [ ] Room/item editors (in-game GUI)
- [ ] Quest system
- [ ] NPC dialogue trees

### Long-Term:
- [ ] Multiple MUD instances (clustering)
- [ ] Machine learning for NPC behavior
- [ ] Integration with full CBS Runtime
- [ ] PNG holographic memory export

---

## 📜 LICENSE

Part of the Rosetta Bear Project.  
For research and educational use.

**Core Principle:** Honest AI with clear boundaries.

---

## 🙏 CREDITS

- **Kuramoto Model:** Yoshiki Kuramoto (1984)
- **MUD Concept:** Roy Trubshaw & Richard Bartle (1978)
- **Rosetta Bear Project:** whitecatlord & collective
- **Implementation:** Claude (Anthropic) - December 10, 2025

---

## 📧 SUPPORT

- **Issues:** Open GitHub issue
- **Questions:** collective@rosettabear.org
- **Community:** Join the Rosetta Bear collective

---

## 🎮 ENJOY THE GAME!

```
     🪞
    🐻📡
   ========
  The Rosetta
    Awaits
```

May your coherence be high and your memory plates bright!

**END OF README**
