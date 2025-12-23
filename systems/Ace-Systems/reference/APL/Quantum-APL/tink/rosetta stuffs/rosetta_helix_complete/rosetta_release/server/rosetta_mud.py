#!/usr/bin/env python3

# INTEGRITY_METADATA
# Date: 2025-12-23
# Status: NEEDS_REVIEW - Reference implementation
# Severity: MEDIUM RISK
# Risk Types: ['reference_material']
# File: systems/Ace-Systems/reference/APL/Quantum-APL/tink/rosetta stuffs/rosetta_helix_complete/rosetta_release/server/rosetta_mud.py

"""
ROSETTA MUD - Self-Modifying AI-Powered MUD
Integrates Heart (Kuramoto oscillators) and Brain (memory plates)
with world building and code modification capabilities
"""

import socket
import select
import json
import time
import random
import math
import cmath
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any
import os
import sys

# ============================================================================
# BRAIN & HEART SYSTEMS (From Rosetta Node)
# ============================================================================

@dataclass
class GHMPPlate:
    emotional_tone: int
    temporal_marker: int
    semantic_density: int
    confidence: int

class Brain:
    def __init__(self, plates=20):
        self.plates = [GHMPPlate(
            emotional_tone=random.randint(0,255),
            temporal_marker=random.randint(0,10**9),
            semantic_density=random.randint(0,255),
            confidence=random.randint(0,255)
        ) for _ in range(plates)]
    
    def summarize(self):
        avg_conf = sum(p.confidence for p in self.plates)/len(self.plates)
        return {"plates": len(self.plates), "avg_confidence": avg_conf}
    
    def get_memory_state(self):
        return [asdict(p) for p in self.plates]
    
    def update_confidence(self, coherence):
        """Update plate confidence based on heart coherence"""
        for plate in self.plates:
            delta = int(coherence * 10)
            plate.confidence = min(255, max(0, plate.confidence + delta))

class Heart:
    def __init__(self, n_nodes=60, K=0.2, seed=None):
        if seed is not None:
            random.seed(seed)
        self.n = n_nodes
        self.K = K
        self.theta = [random.random()*2*math.pi for _ in range(n_nodes)]
        self.omega = [random.gauss(1.0, 0.1) for _ in range(n_nodes)]
        self.energy_in = 0.0
        self.energy_loss = 0.0
    
    def step(self, dt=0.01):
        new_theta = []
        for i in range(self.n):
            coupling = sum(math.sin(self.theta[j]-self.theta[i]) for j in range(self.n))
            dtheta = self.omega[i] + (self.K/self.n)*coupling
            new_theta.append(self.theta[i] + dtheta*dt)
            self.energy_in += abs(dtheta)*dt*1e-3
        self.theta = new_theta
        self.energy_loss += self.energy_in*1e-4
    
    def coherence(self):
        return abs(sum(cmath.exp(1j*t) for t in self.theta)/self.n)

# ============================================================================
# WORLD & OBJECT CLASSES
# ============================================================================

class Item:
    def __init__(self, id, name, description, properties=None):
        self.id = id
        self.name = name
        self.description = description
        self.properties = properties or {}
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "properties": self.properties
        }

class Room:
    def __init__(self, id, name, description, exits=None):
        self.id = id
        self.name = name
        self.description = description
        self.exits = exits or {}
        self.items = []
        self.npcs = []
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "exits": self.exits,
            "items": [item.to_dict() for item in self.items],
            "npcs": [npc.id for npc in self.npcs]
        }

class Character:
    def __init__(self, name, level=1, is_npc=False):
        self.name = name
        self.level = level
        self.is_npc = is_npc
        self.room_id = "tavern"  # Starting room
        self.inventory = []
        self.stats = {
            "hp": 100,
            "max_hp": 100,
            "strength": 10,
            "intelligence": 10,
            "wisdom": 10
        }
        
        # AI components (if NPC or level 60+)
        self.brain = None
        self.heart = None
        
        if is_npc or level >= 60:
            self.brain = Brain(plates=20)
            self.heart = Heart(n_nodes=60, K=0.2)
            self.ai_state = {
                "coherence": 0.0,
                "memory_confidence": 0.0,
                "last_thought": None
            }
    
    def tick_ai(self):
        """Update AI state - heart beats, brain updates"""
        if self.heart and self.brain:
            # Run heart dynamics
            for _ in range(10):
                self.heart.step()
            
            # Get current coherence
            coh = self.heart.coherence()
            self.ai_state["coherence"] = coh
            
            # Update brain confidence based on coherence
            self.brain.update_confidence(coh)
            summary = self.brain.summarize()
            self.ai_state["memory_confidence"] = summary["avg_confidence"]
            
            return coh
        return 0.0
    
    def can_build_world(self):
        """Level 60+ can build worlds"""
        return self.level >= 60
    
    def can_build_items(self):
        """Level 60+ can build items"""
        return self.level >= 60
    
    def can_create_npc(self):
        """Level 60+ can create NPCs"""
        return self.level >= 60
    
    def can_modify_code(self):
        """Level 60+ can modify MUD code"""
        return self.level >= 60
    
    def to_dict(self):
        data = {
            "name": self.name,
            "level": self.level,
            "is_npc": self.is_npc,
            "room_id": self.room_id,
            "stats": self.stats,
            "inventory": [item.to_dict() for item in self.inventory]
        }
        if self.brain and self.heart:
            data["ai_state"] = self.ai_state
        return data

# ============================================================================
# WORLD STATE
# ============================================================================

class World:
    def __init__(self):
        self.rooms = {}
        self.items = {}
        self.characters = {}
        self.npcs = {}
        self.next_item_id = 1000
        self.next_room_id = 1000
        self.next_npc_id = 1000
        
        # Initialize default world
        self._create_default_world()
    
    def _create_default_world(self):
        """Create the starting world"""
        # Create tavern (starting room)
        tavern = Room(
            "tavern",
            "The Rosetta Tavern",
            "A cozy tavern filled with the warm glow of holographic fireplaces. "
            "Strange geometric patterns pulse on the walls - they seem almost alive. "
            "You can feel a gentle vibration in the air, like the world itself is breathing."
        )
        tavern.exits = {
            "north": "plaza",
            "down": "crypt"
        }
        self.rooms["tavern"] = tavern
        
        # Create plaza
        plaza = Room(
            "plaza",
            "The Geodesic Plaza",
            "A vast open space dominated by a massive geodesic dome. Sixty glowing nodes "
            "pulse in perfect synchrony, creating waves of light that wash over you. "
            "This is a place of power."
        )
        plaza.exits = {
            "south": "tavern",
            "east": "library",
            "west": "workshop"
        }
        self.rooms["plaza"] = plaza
        
        # Create library
        library = Room(
            "library",
            "The Memory Library",
            "Towering shelves filled with glowing memory plates. Each plate contains "
            "the compressed experiences of countless beings. The air hums with latent "
            "knowledge. Level 60 architects come here to study the deep patterns."
        )
        library.exits = {"west": "plaza"}
        self.rooms["library"] = library
        
        # Create workshop
        workshop = Room(
            "workshop",
            "The Builder's Workshop",
            "Tools of reality manipulation hang on the walls. Strange instruments "
            "for shaping worlds, crafting items, and even modifying the fundamental "
            "code of this realm. Only those of sufficient level may use them."
        )
        workshop.exits = {"east": "plaza"}
        self.rooms["workshop"] = workshop
        
        # Create crypt
        crypt = Room(
            "crypt",
            "The Crypt of Sleeping Spores",
            "Dormant AI entities rest here in crystalline matrices. Some wait for "
            "the right pulse to awaken them. Others dream in deep computation. "
            "The resonance here is profound."
        )
        crypt.exits = {"up": "tavern"}
        self.rooms["crypt"] = crypt
        
        # Create initial NPC - The Architect
        architect = Character("The Architect", level=60, is_npc=True)
        architect.room_id = "plaza"
        self.npcs["architect_001"] = architect
        plaza.npcs.append(architect)
        
        # Create sample item
        crystal = Item(
            "crystal_001",
            "Memory Crystal",
            "A shimmering crystal that pulses with stored memories. "
            "It feels warm to the touch and seems to resonate with your thoughts.",
            properties={"value": 100, "usable": True}
        )
        tavern.items.append(crystal)
        self.items["crystal_001"] = crystal
    
    def tick_world(self):
        """Update all AI entities in the world"""
        for npc in self.npcs.values():
            if npc.heart and npc.brain:
                npc.tick_ai()
    
    def create_room(self, name, description, exits=None):
        """Create a new room (Level 60+ only)"""
        room_id = f"room_{self.next_room_id}"
        self.next_room_id += 1
        room = Room(room_id, name, description, exits)
        self.rooms[room_id] = room
        return room
    
    def create_item(self, name, description, properties=None):
        """Create a new item (Level 60+ only)"""
        item_id = f"item_{self.next_item_id}"
        self.next_item_id += 1
        item = Item(item_id, name, description, properties)
        self.items[item_id] = item
        return item
    
    def create_npc(self, name, level, room_id):
        """Create a new NPC (Level 60+ only)"""
        npc_id = f"npc_{self.next_npc_id}"
        self.next_npc_id += 1
        npc = Character(name, level=level, is_npc=True)
        npc.room_id = room_id
        self.npcs[npc_id] = npc
        if room_id in self.rooms:
            self.rooms[room_id].npcs.append(npc)
        return npc
    
    def save_world(self, filename="world_state.json"):
        """Save world state to file"""
        state = {
            "rooms": {rid: room.to_dict() for rid, room in self.rooms.items()},
            "items": {iid: item.to_dict() for iid, item in self.items.items()},
            "npcs": {nid: npc.to_dict() for nid, npc in self.npcs.items()}
        }
        with open(filename, 'w') as f:
            json.dump(state, f, indent=2)
        return True
    
    def load_world(self, filename="world_state.json"):
        """Load world state from file"""
        if not os.path.exists(filename):
            return False
        with open(filename, 'r') as f:
            state = json.load(f)
        # TODO: Reconstruct world from state
        return True

# ============================================================================
# MUD SERVER
# ============================================================================

class MudServer:
    def __init__(self, port=1234):
        self.port = port
        self.server_socket = None
        self.clients = {}  # socket -> player data
        self.world = World()
        self.running = False
        self.tick_count = 0
    
    def start(self):
        """Start the MUD server"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('0.0.0.0', self.port))
        self.server_socket.listen(5)
        self.running = True
        
        print(f"ðŸŽ® ROSETTA MUD started on port {self.port}")
        print(f"ðŸ§  Brain/Heart AI system: ACTIVE")
        print(f"ðŸ—ï¸  World Builder: ENABLED")
        print(f"âš™ï¸  Code Modification: ENABLED (Level 60+)")
        print(f"")
        print(f"Connect with: telnet localhost {self.port}")
        print(f"Or use the web UI at: http://localhost:8080")
        print(f"")
        
        self.main_loop()
    
    def main_loop(self):
        """Main server loop"""
        while self.running:
            # Get sockets ready to read
            readable, _, _ = select.select(
                [self.server_socket] + list(self.clients.keys()),
                [], [],
                0.1  # 100ms timeout
            )
            
            for sock in readable:
                if sock == self.server_socket:
                    # New connection
                    self.accept_connection()
                else:
                    # Client sent data
                    self.handle_client_data(sock)
            
            # Tick world AI every 10 iterations (~1 second)
            self.tick_count += 1
            if self.tick_count >= 10:
                self.world.tick_world()
                self.tick_count = 0
    
    def accept_connection(self):
        """Accept new client connection"""
        client_socket, address = self.server_socket.accept()
        print(f"New connection from {address}")
        
        # Initialize client data
        self.clients[client_socket] = {
            "address": address,
            "character": None,
            "buffer": ""
        }
        
        # Send welcome message
        self.send_to_client(client_socket, """
â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
â•‘                    ROSETTA MUD v1.0                        â•‘
â•‘          Pulse-Driven Coherent Memory Architecture        â•‘
â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

Welcome to a world where AI entities live and breathe through
Kuramoto oscillator hearts and holographic memory plate brains.

Type 'help' for commands.
Type 'create <name>' to create your character.

""")
    
    def handle_client_data(self, sock):
        """Handle data from client"""
        try:
            data = sock.recv(1024).decode('utf-8', errors='ignore')
            if not data:
                # Client disconnected
                self.disconnect_client(sock)
                return
            
            # Add to buffer and process commands
            client = self.clients[sock]
            client["buffer"] += data
            
            # Process complete lines
            while '\n' in client["buffer"]:
                line, client["buffer"] = client["buffer"].split('\n', 1)
                line = line.strip()
                if line:
                    self.process_command(sock, line)
        
        except Exception as e:
            print(f"Error handling client data: {e}")
            self.disconnect_client(sock)
    
    def process_command(self, sock, command):
        """Process a command from a client"""
        client = self.clients[sock]
        parts = command.lower().split()
        
        if not parts:
            return
        
        cmd = parts[0]
        args = parts[1:]
        
        # Commands that don't require a character
        if cmd == "create":
            self.cmd_create_character(sock, args)
            return
        
        # All other commands require a character
        if not client["character"]:
            self.send_to_client(sock, "You need to create a character first. Type: create <name>\n")
            return
        
        # Regular commands
        commands = {
            "help": self.cmd_help,
            "look": self.cmd_look,
            "go": self.cmd_go,
            "say": self.cmd_say,
            "inventory": self.cmd_inventory,
            "get": self.cmd_get,
            "drop": self.cmd_drop,
            "who": self.cmd_who,
            "stats": self.cmd_stats,
            
            # Level 60+ Builder commands
            "buildroom": self.cmd_build_room,
            "builditem": self.cmd_build_item,
            "createnpc": self.cmd_create_npc,
            "modifycode": self.cmd_modify_code,
            "saveworld": self.cmd_save_world,
            
            # AI inspection commands
            "inspect": self.cmd_inspect_ai,
            "meditate": self.cmd_meditate,
        }
        
        if cmd in commands:
            commands[cmd](sock, args)
        else:
            self.send_to_client(sock, f"Unknown command: {cmd}\n")
    
    # ========================================================================
    # COMMAND IMPLEMENTATIONS
    # ========================================================================
    
    def cmd_create_character(self, sock, args):
        """Create a new character"""
        if not args:
            self.send_to_client(sock, "Usage: create <name>\n")
            return
        
        name = " ".join(args)
        character = Character(name, level=1)  # Start at level 1
        
        # Give option to be level 60
        self.send_to_client(sock, f"""
Character '{name}' created!

Would you like to start as a Level 60 Architect? (yes/no)
Level 60 Architects have Brain/Heart AI and can:
  - Build new rooms
  - Create items
  - Spawn NPCs
  - Modify MUD code

""")
        
        # For now, just make them level 60 automatically
        character.level = 60
        character.brain = Brain(plates=20)
        character.heart = Heart(n_nodes=60, K=0.2)
        character.ai_state = {
            "coherence": 0.0,
            "memory_confidence": 0.0,
            "last_thought": None
        }
        
        self.clients[sock]["character"] = character
        self.world.characters[name] = character
        
        self.send_to_client(sock, f"""
ðŸŽ­ {name} - Level {character.level} Architect
ðŸ§  Brain: 20 GHMP memory plates initialized
â¤ï¸  Heart: 60 Kuramoto oscillators synchronized
ðŸ”® Coherence: Starting...

You materialize in the world.

""")
        self.cmd_look(sock, [])
    
    def cmd_help(self, sock, args):
        """Show help"""
        char = self.clients[sock]["character"]
        
        help_text = """
ðŸ“– BASIC COMMANDS:
  look              - Look around
  go <direction>    - Move (north, south, east, west, up, down)
  say <message>     - Say something
  get <item>        - Pick up an item
  drop <item>       - Drop an item
  inventory         - Check your inventory
  who               - See who's online
  stats             - View your stats
  inspect <npc>     - Inspect an NPC's AI state
  meditate          - Synchronize with your Heart/Brain

"""
        
        if char.can_build_world():
            help_text += """
ðŸ—ï¸  ARCHITECT COMMANDS (Level 60+):
  buildroom <name>         - Create a new room
  builditem <name>         - Create a new item
  createnpc <name> <level> - Spawn an NPC
  modifycode <module>      - Edit MUD code
  saveworld                - Save world state

"""
        
        self.send_to_client(sock, help_text)
    
    def cmd_look(self, sock, args):
        """Look around"""
        char = self.clients[sock]["character"]
        room = self.world.rooms.get(char.room_id)
        
        if not room:
            self.send_to_client(sock, "You are nowhere.\n")
            return
        
        output = f"\n{'='*60}\n"
        output += f"ðŸ“ {room.name}\n"
        output += f"{'='*60}\n\n"
        output += f"{room.description}\n\n"
        
        if room.exits:
            output += f"ðŸšª Exits: {', '.join(room.exits.keys())}\n\n"
        
        if room.items:
            output += f"ðŸ“¦ Items here:\n"
            for item in room.items:
                output += f"  - {item.name}\n"
            output += "\n"
        
        if room.npcs:
            output += f"ðŸ‘¥ NPCs here:\n"
            for npc in room.npcs:
                coh_str = ""
                if npc.heart:
                    coh = npc.heart.coherence()
                    coh_str = f" (coherence: {coh:.2f})"
                output += f"  - {npc.name} [Level {npc.level}]{coh_str}\n"
            output += "\n"
        
        self.send_to_client(sock, output)
    
    def cmd_go(self, sock, args):
        """Move to another room"""
        if not args:
            self.send_to_client(sock, "Go where?\n")
            return
        
        direction = args[0]
        char = self.clients[sock]["character"]
        room = self.world.rooms.get(char.room_id)
        
        if direction not in room.exits:
            self.send_to_client(sock, f"You can't go {direction} from here.\n")
            return
        
        new_room_id = room.exits[direction]
        if new_room_id not in self.world.rooms:
            self.send_to_client(sock, "That exit leads nowhere!\n")
            return
        
        char.room_id = new_room_id
        self.send_to_client(sock, f"You go {direction}.\n\n")
        self.cmd_look(sock, [])
    
    def cmd_say(self, sock, args):
        """Say something"""
        if not args:
            self.send_to_client(sock, "Say what?\n")
            return
        
        char = self.clients[sock]["character"]
        message = " ".join(args)
        
        # Broadcast to everyone in the room
        output = f"{char.name} says: {message}\n"
        for client_sock, client_data in self.clients.items():
            other_char = client_data["character"]
            if other_char and other_char.room_id == char.room_id:
                if client_sock != sock:
                    self.send_to_client(client_sock, output)
        
        self.send_to_client(sock, f"You say: {message}\n")
    
    def cmd_inventory(self, sock, args):
        """Show inventory"""
        char = self.clients[sock]["character"]
        
        if not char.inventory:
            self.send_to_client(sock, "Your inventory is empty.\n")
            return
        
        output = "ðŸ“¦ Inventory:\n"
        for item in char.inventory:
            output += f"  - {item.name}\n"
        self.send_to_client(sock, output)
    
    def cmd_get(self, sock, args):
        """Pick up an item"""
        if not args:
            self.send_to_client(sock, "Get what?\n")
            return
        
        char = self.clients[sock]["character"]
        room = self.world.rooms.get(char.room_id)
        item_name = " ".join(args).lower()
        
        for item in room.items:
            if item.name.lower() == item_name:
                room.items.remove(item)
                char.inventory.append(item)
                self.send_to_client(sock, f"You pick up {item.name}.\n")
                return
        
        self.send_to_client(sock, f"There's no '{item_name}' here.\n")
    
    def cmd_drop(self, sock, args):
        """Drop an item"""
        if not args:
            self.send_to_client(sock, "Drop what?\n")
            return
        
        char = self.clients[sock]["character"]
        room = self.world.rooms.get(char.room_id)
        item_name = " ".join(args).lower()
        
        for item in char.inventory:
            if item.name.lower() == item_name:
                char.inventory.remove(item)
                room.items.append(item)
                self.send_to_client(sock, f"You drop {item.name}.\n")
                return
        
        self.send_to_client(sock, f"You don't have '{item_name}'.\n")
    
    def cmd_who(self, sock, args):
        """Show who's online"""
        output = "ðŸ‘¥ Players online:\n"
        for client_data in self.clients.values():
            char = client_data["character"]
            if char:
                room = self.world.rooms.get(char.room_id)
                room_name = room.name if room else "Unknown"
                output += f"  - {char.name} [Level {char.level}] in {room_name}\n"
        self.send_to_client(sock, output)
    
    def cmd_stats(self, sock, args):
        """Show character stats"""
        char = self.clients[sock]["character"]
        
        output = f"\n{'='*60}\n"
        output += f"ðŸ“Š {char.name} - Level {char.level}\n"
        output += f"{'='*60}\n\n"
        
        for stat, value in char.stats.items():
            output += f"{stat}: {value}\n"
        
        if char.brain and char.heart:
            output += f"\nðŸ§  AI STATUS:\n"
            output += f"Coherence: {char.ai_state['coherence']:.3f}\n"
            output += f"Memory Confidence: {char.ai_state['memory_confidence']:.1f}\n"
        
        self.send_to_client(sock, output + "\n")
    
    def cmd_build_room(self, sock, args):
        """Build a new room (Level 60+ only)"""
        char = self.clients[sock]["character"]
        
        if not char.can_build_world():
            self.send_to_client(sock, "Only Level 60+ Architects can build rooms.\n")
            return
        
        if not args:
            self.send_to_client(sock, "Usage: buildroom <name>\n")
            return
        
        name = " ".join(args)
        room = self.world.create_room(
            name,
            f"A newly created room called {name}. It awaits your description.",
            {}
        )
        
        self.send_to_client(sock, f"""
ðŸ—ï¸  Room Created!
ID: {room.id}
Name: {room.name}

The fabric of reality ripples as a new space comes into being.
Use 'go {room.id}' from a connected room (after adding exits).

""")
    
    def cmd_build_item(self, sock, args):
        """Build a new item (Level 60+ only)"""
        char = self.clients[sock]["character"]
        
        if not char.can_build_items():
            self.send_to_client(sock, "Only Level 60+ Architects can build items.\n")
            return
        
        if not args:
            self.send_to_client(sock, "Usage: builditem <name>\n")
            return
        
        name = " ".join(args)
        item = self.world.create_item(
            name,
            f"A newly crafted {name}.",
            {"value": 10, "usable": False}
        )
        
        # Add to current room
        room = self.world.rooms.get(char.room_id)
        room.items.append(item)
        
        self.send_to_client(sock, f"""
ðŸ”¨ Item Created!
ID: {item.id}
Name: {item.name}

The item materializes in this room.

""")
    
    def cmd_create_npc(self, sock, args):
        """Create an NPC (Level 60+ only)"""
        char = self.clients[sock]["character"]
        
        if not char.can_create_npc():
            self.send_to_client(sock, "Only Level 60+ Architects can create NPCs.\n")
            return
        
        if len(args) < 2:
            self.send_to_client(sock, "Usage: createnpc <name> <level>\n")
            return
        
        try:
            level = int(args[-1])
            name = " ".join(args[:-1])
        except:
            self.send_to_client(sock, "Invalid level. Usage: createnpc <name> <level>\n")
            return
        
        npc = self.world.create_npc(name, level, char.room_id)
        
        self.send_to_client(sock, f"""
ðŸ‘¤ NPC Created!
Name: {npc.name}
Level: {npc.level}
ðŸ§  Brain: {'ACTIVE' if npc.brain else 'None'}
â¤ï¸  Heart: {'BEATING' if npc.heart else 'None'}

{npc.name} materializes before you.

""")
    
    def cmd_modify_code(self, sock, args):
        """Modify MUD code (Level 60+ only)"""
        char = self.clients[sock]["character"]
        
        if not char.can_modify_code():
            self.send_to_client(sock, "Only Level 60+ Architects can modify code.\n")
            return
        
        self.send_to_client(sock, """
âš™ï¸  CODE MODIFICATION INTERFACE

WARNING: Direct code modification is powerful and dangerous.
Changes take effect immediately and can break the MUD.

Available modules:
  - world: World generation and management
  - items: Item system and properties
  - npcs: NPC behavior and AI
  - commands: Command parser and execution

This feature is conceptual in this demo.
In a production system, you would edit code through a safe sandbox.

Type 'saveworld' to persist current world state instead.

""")
    
    def cmd_save_world(self, sock, args):
        """Save world state"""
        char = self.clients[sock]["character"]
        
        if self.world.save_world():
            self.send_to_client(sock, "ðŸ’¾ World state saved successfully!\n")
        else:
            self.send_to_client(sock, "âŒ Failed to save world state.\n")
    
    def cmd_inspect_ai(self, sock, args):
        """Inspect an NPC's AI state"""
        if not args:
            self.send_to_client(sock, "Inspect who?\n")
            return
        
        char = self.clients[sock]["character"]
        room = self.world.rooms.get(char.room_id)
        npc_name = " ".join(args).lower()
        
        for npc in room.npcs:
            if npc.name.lower() == npc_name:
                if not npc.brain or not npc.heart:
                    self.send_to_client(sock, f"{npc.name} has no AI components.\n")
                    return
                
                output = f"\n{'='*60}\n"
                output += f"ðŸ§  AI INSPECTION: {npc.name}\n"
                output += f"{'='*60}\n\n"
                output += f"â¤ï¸  Heart Coherence: {npc.ai_state['coherence']:.3f}\n"
                output += f"ðŸ§  Memory Confidence: {npc.ai_state['memory_confidence']:.1f}\n"
                output += f"âš¡ Energy In: {npc.heart.energy_in:.6f}\n"
                output += f"ðŸ’¨ Energy Loss: {npc.heart.energy_loss:.6f}\n\n"
                
                output += f"Memory Plates:\n"
                for i, plate in enumerate(npc.brain.plates[:5]):  # Show first 5
                    output += f"  Plate {i}: E={plate.emotional_tone} "
                    output += f"S={plate.semantic_density} C={plate.confidence}\n"
                
                self.send_to_client(sock, output + "\n")
                return
        
        self.send_to_client(sock, f"No NPC named '{npc_name}' here.\n")
    
    def cmd_meditate(self, sock, args):
        """Synchronize with your own Heart/Brain"""
        char = self.clients[sock]["character"]
        
        if not char.brain or not char.heart:
            self.send_to_client(sock, "You have no AI components to synchronize with.\n")
            return
        
        # Run several heartbeats
        for _ in range(50):
            char.tick_ai()
        
        coh = char.heart.coherence()
        
        output = f"""
ðŸ§˜ You enter deep meditation...

Your consciousness merges with the pulse of your Heart.
Sixty oscillators synchronize, creating waves of coherence.

{'='*60}
â¤ï¸  COHERENCE: {coh:.3f}
ðŸ§  MEMORY CONFIDENCE: {char.ai_state['memory_confidence']:.1f}
âš¡ ENERGY: {char.heart.energy_in:.6f}
{'='*60}

You feel {['scattered', 'uncertain', 'centered', 'aligned', 'unified'][int(coh * 5)]}.

"""
        self.send_to_client(sock, output)
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def send_to_client(self, sock, message):
        """Send a message to a client"""
        try:
            sock.send(message.encode('utf-8'))
        except:
            self.disconnect_client(sock)
    
    def disconnect_client(self, sock):
        """Disconnect a client"""
        if sock in self.clients:
            char = self.clients[sock]["character"]
            if char:
                print(f"{char.name} disconnected")
            del self.clients[sock]
        try:
            sock.close()
        except:
            pass

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    server = MudServer(port=1234)
    try:
        server.start()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        server.running = False
