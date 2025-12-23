#!/usr/bin/env python3

# INTEGRITY_METADATA
# Date: 2025-12-23
# Status: NEEDS_REVIEW - Uncategorized file
# Severity: MEDIUM RISK
# Risk Types: ['uncategorized']
# File: systems/Rosetta-Helix Research Group/AcornEngine_v7_FINAL/clients/terminal.py

"""
Terminal Client for The Ultimate Acorn v7

MUD-style text interface with full world interaction.
"""

import sys
import time
from typing import Optional
from acorn import (
    AcornEngine,
    WorldState,
    EntityType,
    Position,
    AdapterLayer
)
from acorn.plates import PlateManager
from acorn.fractal import FractalSimulationEngine


class TerminalClient:
    """Text-based terminal interface for Acorn Engine."""
    
    def __init__(self, engine: AcornEngine):
        self.engine = engine
        self.adapter = AdapterLayer(engine)
        self.plate_manager = PlateManager(engine)
        
        # Fractal engine (if enabled)
        if engine.config.get("fractal", {}).get("enabled", False):
            self.fractal = FractalSimulationEngine(engine, engine.config)
        else:
            self.fractal = None
        
        self.running = False
        self.player_entity_id: Optional[str] = None
        self.auto_tick = False
        
    def start(self):
        """Start the terminal interface."""
        self.running = True
        self.print_welcome()
        
        # Create player entity
        self.create_player()
        
        # Main loop
        while self.running:
            try:
                self.main_loop()
            except KeyboardInterrupt:
                self.running = False
                print("\n\nShutting down...")
            except Exception as e:
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc()
    
    def print_welcome(self):
        """Print welcome message."""
        print("=" * 60)
        print(" " * 15 + "THE ULTIMATE ACORN v7")
        print(" " * 10 + "Fractal Universe Simulator")
        print("=" * 60)
        print()
        print(f"World: {self.engine.world.width}x{self.engine.world.height}")
        print(f"Seed: {self.engine.world.seed}")
        print(f"ISS: {'Enabled' if self.engine.config['iss_enabled'] else 'Disabled'}")
        print(f"Fractal: {'Enabled' if self.fractal else 'Disabled'}")
        print()
        print("Type 'help' for commands")
        print()
    
    def create_player(self):
        """Create the player entity."""
        # Spawn in center of world
        x = self.engine.world.width // 2
        y = self.engine.world.height // 2
        
        self.player_entity_id = self.adapter.create_entity("player", x, y)
        
        if self.player_entity_id:
            print(f"✓ Player spawned at ({x}, {y})")
        else:
            print("✗ Failed to spawn player")
    
    def main_loop(self):
        """Main interaction loop."""
        # Auto-tick if enabled
        if self.auto_tick:
            self.adapter.tick_engine()
            time.sleep(0.1)  # 10 TPS when auto-ticking
        
        # Get user input
        try:
            command = input("> ").strip().lower()
        except EOFError:
            self.running = False
            return
        
        if not command:
            return
        
        # Parse and execute command
        self.execute_command(command)
    
    def execute_command(self, command: str):
        """Execute a user command."""
        parts = command.split()
        cmd = parts[0]
        args = parts[1:]
        
        # Command dispatch
        commands = {
            "help": self.cmd_help,
            "look": self.cmd_look,
            "move": self.cmd_move,
            "tick": self.cmd_tick,
            "auto": self.cmd_auto,
            "stats": self.cmd_stats,
            "spawn": self.cmd_spawn,
            "fractal": self.cmd_fractal,
            "save": self.cmd_save,
            "load": self.cmd_load,
            "quit": self.cmd_quit,
            "exit": self.cmd_quit
        }
        
        if cmd in commands:
            commands[cmd](args)
        else:
            print(f"Unknown command: {cmd}")
            print("Type 'help' for list of commands")
    
    def cmd_help(self, args):
        """Show help."""
        print()
        print("Commands:")
        print("  help              - Show this help")
        print("  look              - Look around")
        print("  move <dir>        - Move (n/s/e/w or north/south/east/west)")
        print("  tick [N]          - Execute N ticks (default: 1)")
        print("  auto [on|off]     - Toggle auto-tick mode")
        print("  stats             - Show statistics")
        print("  spawn <type> <x> <y> - Spawn entity")
        print("  fractal <cmd>     - Fractal simulation commands")
        print("  save <file>       - Save to PNG plate")
        print("  load <file>       - Load from PNG plate")
        print("  quit/exit         - Exit simulation")
        print()
    
    def cmd_look(self, args):
        """Look around."""
        if not self.player_entity_id:
            print("No player entity!")
            return
        
        player = self.engine.world.entities.get(self.player_entity_id)
        if not player:
            print("Player not found!")
            return
        
        print()
        print(f"=== You are at ({player.position.x}, {player.position.y}) ===")
        print()
        
        # Show nearby entities
        nearby = self.engine.world.get_entities_in_range(player.position, radius=5)
        nearby = [e for e in nearby if e.id != self.player_entity_id]
        
        if nearby:
            print("Nearby entities:")
            for entity in nearby:
                dist = player.position.distance_to(entity.position)
                print(f"  {entity.type.value} at ({entity.position.x}, {entity.position.y}) - {dist:.1f} units away")
        else:
            print("Nothing nearby.")
        
        print()
        
        # Show ISS state if available
        if player.awareness_state:
            print(f"Awareness: {player.awareness_state}")
            if player.affect_state:
                print(f"Affect: valence={player.affect_state.get('valence', 0):.2f}, " +
                     f"arousal={player.affect_state.get('arousal', 0):.2f}")
        print()
    
    def cmd_move(self, args):
        """Move player."""
        if not args:
            print("Usage: move <direction>")
            return
        
        if not self.player_entity_id:
            print("No player entity!")
            return
        
        player = self.engine.world.entities.get(self.player_entity_id)
        if not player:
            print("Player not found!")
            return
        
        # Parse direction
        direction = args[0].lower()
        dx, dy = 0, 0
        
        if direction in ["n", "north"]:
            dy = -1
        elif direction in ["s", "south"]:
            dy = 1
        elif direction in ["e", "east"]:
            dx = 1
        elif direction in ["w", "west"]:
            dx = -1
        else:
            print(f"Invalid direction: {direction}")
            return
        
        # Calculate target
        target_x = player.position.x + dx
        target_y = player.position.y + dy
        
        # Attempt move
        moved = self.adapter.move_entity(self.player_entity_id, target_x, target_y)
        
        if moved:
            print(f"Moved to ({target_x}, {target_y})")
        else:
            print("Cannot move there!")
    
    def cmd_tick(self, args):
        """Execute ticks."""
        n = 1
        if args:
            try:
                n = int(args[0])
            except ValueError:
                print("Invalid number of ticks")
                return
        
        for i in range(n):
            result = self.adapter.tick_engine()
            if result.get("success"):
                tick_result = result.get("tick_result", {})
                if n == 1 or i == n - 1:  # Only show last tick for multiple
                    print(f"Tick {tick_result['tick']}: " +
                         f"{tick_result['duration_ms']:.2f}ms, " +
                         f"{tick_result['entities']} entities")
    
    def cmd_auto(self, args):
        """Toggle auto-tick."""
        if args:
            mode = args[0].lower()
            if mode in ["on", "1", "true"]:
                self.auto_tick = True
                print("Auto-tick enabled (10 TPS)")
            elif mode in ["off", "0", "false"]:
                self.auto_tick = False
                print("Auto-tick disabled")
            else:
                print("Usage: auto [on|off]")
        else:
            self.auto_tick = not self.auto_tick
            print(f"Auto-tick {'enabled' if self.auto_tick else 'disabled'}")
    
    def cmd_stats(self, args):
        """Show statistics."""
        stats = self.engine.get_stats()
        adapter_stats = self.adapter.get_adapter_stats()
        
        print()
        print("=== Engine Statistics ===")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print()
        print("=== Adapter Statistics ===")
        for key, value in adapter_stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
        
        if self.fractal:
            fractal_stats = self.fractal.get_stats()
            print()
            print("=== Fractal Statistics ===")
            for key, value in fractal_stats.items():
                print(f"  {key}: {value}")
        
        print()
    
    def cmd_spawn(self, args):
        """Spawn an entity."""
        if len(args) < 3:
            print("Usage: spawn <type> <x> <y>")
            return
        
        entity_type = args[0]
        try:
            x = int(args[1])
            y = int(args[2])
        except ValueError:
            print("Invalid coordinates")
            return
        
        entity_id = self.adapter.create_entity(entity_type, x, y)
        
        if entity_id:
            print(f"Spawned {entity_type} at ({x}, {y}): {entity_id}")
        else:
            print("Failed to spawn entity")
    
    def cmd_fractal(self, args):
        """Fractal simulation commands."""
        if not self.fractal:
            print("Fractal simulation not enabled!")
            return
        
        if not args:
            print("Usage: fractal <spawn|exec|stats|hierarchy>")
            return
        
        subcmd = args[0].lower()
        
        if subcmd == "spawn":
            if not self.player_entity_id:
                print("No player entity!")
                return
            
            layer_id = self.fractal.spawn_layer(self.player_entity_id, "manual")
            if layer_id:
                print(f"Spawned fractal layer: {layer_id}")
            else:
                print("Failed to spawn layer")
        
        elif subcmd == "exec":
            results = self.fractal.execute_all_layers()
            print(f"Executed {len(results)} layers")
            for result in results[:5]:  # Show first 5
                print(f"  {result}")
        
        elif subcmd == "stats":
            stats = self.fractal.get_stats()
            print("\n=== Fractal Statistics ===")
            for key, value in stats.items():
                print(f"  {key}: {value}")
            print()
        
        elif subcmd == "hierarchy":
            hierarchy = self.fractal.get_layer_hierarchy()
            print("\n=== Layer Hierarchy ===")
            print(f"Base (depth 0): {hierarchy['base']['entities']} entities")
            for depth, layers in hierarchy['base']['layers'].items():
                print(f"  Depth {depth}: {len(layers)} layers")
            print()
        
        else:
            print(f"Unknown fractal command: {subcmd}")
    
    def cmd_save(self, args):
        """Save to PNG plate."""
        if not args:
            print("Usage: save <filename>")
            return
        
        filepath = args[0]
        if not filepath.endswith('.png'):
            filepath += '.png'
        
        try:
            self.plate_manager.save_plate(filepath)
            print(f"✓ Saved to {filepath}")
        except Exception as e:
            print(f"✗ Failed to save: {e}")
    
    def cmd_load(self, args):
        """Load from PNG plate."""
        if not args:
            print("Usage: load <filename>")
            return
        
        filepath = args[0]
        
        try:
            self.plate_manager.load_plate(filepath)
            print(f"✓ Loaded from {filepath}")
            
            # Need to recreate player reference
            player = [e for e in self.engine.world.entities.values() 
                     if e.type == EntityType.PLAYER]
            if player:
                self.player_entity_id = player[0].id
            else:
                self.player_entity_id = None
                print("Warning: No player entity in loaded world")
        except Exception as e:
            print(f"✗ Failed to load: {e}")
    
    def cmd_quit(self, args):
        """Quit the simulation."""
        print("\nGoodbye!")
        self.running = False


def main():
    """Main entry point."""
    # Create world and engine
    world = WorldState(40, 40, seed=int(time.time()))
    
    config = {
        "iss_enabled": True,
        "max_entities": 1000,
        "tick_rate": 60,
        "fractal": {
            "enabled": True,
            "max_depth": 5,
            "budget_decay": 0.5,
            "base_budget": 100
        }
    }
    
    engine = AcornEngine(world, config)
    engine.initialize_iss()
    
    # Create and start client
    client = TerminalClient(engine)
    client.start()


if __name__ == "__main__":
    main()
