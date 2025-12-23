# INTEGRITY_METADATA
# Date: 2025-12-23
# Status: NEEDS_REVIEW - Uncategorized file
# Severity: MEDIUM RISK
# Risk Types: ['uncategorized']
# File: systems/Rosetta-Helix Research Group/AcornEngine_v7_FINAL/acorn/plates.py

"""
Holographic Memory Plates - PNG-Based Universe Persistence

Stores complete universe state in PNG images using:
- Visual representation (human-readable at a glance)
- Steganographic encoding (LSB data channels)
- Cross-platform compatibility
- Beautiful aesthetic

This is NOT magic. This is information theory + pretty pictures.
"""

from PIL import Image, ImageDraw, ImageFont
import json
import numpy as np
from typing import Dict, Any, Optional
import io
import base64
import zlib


class HolographicPlate:
    """
    A single holographic memory plate (PNG file).
    
    Encodes complete universe state as:
    1. Visual layer: Human-readable representation
    2. Data layer: Steganographic encoding in LSB channels
    """
    
    def __init__(self, width: int = 1024, height: int = 1024):
        self.width = width
        self.height = height
        self.image = None
        self.metadata = {}
        
    def encode_universe(self, engine_snapshot: Dict) -> Image.Image:
        """
        Encode complete universe state into a PNG plate.
        
        Args:
            engine_snapshot: Complete engine state from engine.get_snapshot()
        
        Returns:
            PIL Image object
        """
        # Create visual representation
        self.image = self._create_visual_layer(engine_snapshot)
        
        # Encode data steganographically
        self._embed_data_layer(engine_snapshot)
        
        return self.image
    
    def _create_visual_layer(self, snapshot: Dict) -> Image.Image:
        """Create beautiful visual representation of universe state."""
        # Create blank image with dark background
        img = Image.new('RGB', (self.width, self.height), color=(10, 10, 20))
        draw = ImageDraw.Draw(img)
        
        world = snapshot["world"]
        
        # Calculate scaling
        world_w = world["width"]
        world_h = world["height"]
        scale_x = self.width // (world_w + 2)
        scale_y = self.height // (world_h + 2)
        scale = min(scale_x, scale_y)
        
        # Offset for centering
        offset_x = (self.width - world_w * scale) // 2
        offset_y = (self.height - world_h * scale) // 2
        
        # Draw world grid (subtle)
        for x in range(world_w + 1):
            x_pos = offset_x + x * scale
            draw.line([(x_pos, offset_y), 
                      (x_pos, offset_y + world_h * scale)],
                     fill=(30, 30, 40), width=1)
        
        for y in range(world_h + 1):
            y_pos = offset_y + y * scale
            draw.line([(offset_x, y_pos),
                      (offset_x + world_w * scale, y_pos)],
                     fill=(30, 30, 40), width=1)
        
        # Draw tiles (different terrain types)
        for y in range(world_h):
            for x in range(world_w):
                tile = world["tiles"][y][x]
                tile_color = self._get_tile_color(tile)
                
                x1 = offset_x + x * scale
                y1 = offset_y + y * scale
                x2 = x1 + scale - 1
                y2 = y1 + scale - 1
                
                draw.rectangle([x1, y1, x2, y2], fill=tile_color)
        
        # Draw entities (bright points)
        for entity in world["entities"].values():
            pos = entity["position"]
            x = offset_x + pos["x"] * scale + scale // 2
            y = offset_y + pos["y"] * scale + scale // 2
            
            # Color by entity type
            color = self._get_entity_color(entity["type"])
            
            # Draw entity as a glowing point
            radius = scale // 3
            for r in range(radius, 0, -1):
                alpha = int(255 * (r / radius))
                # Simulate glow with multiple circles
                draw.ellipse([x-r, y-r, x+r, y+r],
                           fill=(*color, alpha) if len(color) == 3 else color)
        
        # Draw metadata overlay
        self._draw_metadata_overlay(draw, snapshot)
        
        return img
    
    def _get_tile_color(self, tile: Dict) -> tuple:
        """Get color for a tile type."""
        tile_type = tile.get("type", "grass")
        
        colors = {
            "grass": (20, 50, 20),
            "water": (20, 20, 60),
            "stone": (40, 40, 40),
            "sand": (60, 60, 30),
            "forest": (10, 40, 10),
            "mountain": (50, 50, 50)
        }
        
        return colors.get(tile_type, (30, 30, 30))
    
    def _get_entity_color(self, entity_type: str) -> tuple:
        """Get color for an entity type."""
        colors = {
            "basic": (100, 100, 255),
            "player": (255, 255, 100),
            "npc": (100, 255, 100),
            "structure": (255, 100, 100),
            "resource": (255, 150, 50)
        }
        
        return colors.get(entity_type, (150, 150, 150))
    
    def _draw_metadata_overlay(self, draw, snapshot: Dict):
        """Draw metadata information on the image."""
        world = snapshot["world"]
        stats = snapshot["stats"]
        
        # Draw text in corner
        text_lines = [
            f"The Ultimate Acorn v7",
            f"Tick: {world['tick']}",
            f"Entities: {len(world['entities'])}",
            f"Size: {world['width']}x{world['height']}",
            f"Seed: {world['seed']}"
        ]
        
        y_pos = 20
        for line in text_lines:
            draw.text((20, y_pos), line, fill=(200, 200, 220))
            y_pos += 20
    
    def _embed_data_layer(self, snapshot: Dict):
        """Embed complete state data using steganography."""
        if self.image is None:
            raise RuntimeError("Visual layer must be created first")
        
        # Serialize state to JSON
        state_json = json.dumps(snapshot, separators=(',', ':'))
        
        # Compress for efficiency
        compressed = zlib.compress(state_json.encode('utf-8'))
        
        # Convert to binary
        data_bytes = compressed
        data_bits = ''.join(format(byte, '08b') for byte in data_bytes)
        
        # Encode length header (32 bits)
        length_bits = format(len(data_bytes), '032b')
        full_bits = length_bits + data_bits
        
        # Get image pixels
        pixels = np.array(self.image)
        h, w, c = pixels.shape
        
        # Check if we have enough space (using LSB of red channel only)
        available_bits = h * w
        if len(full_bits) > available_bits:
            raise ValueError(f"Data too large: need {len(full_bits)} bits, have {available_bits}")
        
        # Embed bits in LSB of red channel
        flat_pixels = pixels.reshape(-1, c)
        for i, bit in enumerate(full_bits):
            flat_pixels[i, 0] = (flat_pixels[i, 0] & 0xFE) | int(bit)
        
        # Reconstruct image
        pixels = flat_pixels.reshape(h, w, c)
        self.image = Image.fromarray(pixels, 'RGB')
    
    def decode_universe(self, image: Image.Image) -> Dict:
        """
        Decode universe state from a PNG plate.
        
        Args:
            image: PIL Image object
        
        Returns:
            Engine snapshot dictionary
        """
        # Extract embedded data
        pixels = np.array(image)
        h, w, c = pixels.shape
        
        # Extract bits from LSB of red channel
        flat_pixels = pixels.reshape(-1, c)
        
        # Read length header (32 bits)
        length_bits = ''.join(str(flat_pixels[i, 0] & 1) for i in range(32))
        data_length = int(length_bits, 2)
        
        # Read data bits
        data_bits = ''.join(str(flat_pixels[i, 0] & 1) 
                           for i in range(32, 32 + data_length * 8))
        
        # Convert bits to bytes
        data_bytes = bytes(int(data_bits[i:i+8], 2) 
                          for i in range(0, len(data_bits), 8))
        
        # Decompress
        decompressed = zlib.decompress(data_bytes)
        
        # Parse JSON
        snapshot = json.loads(decompressed.decode('utf-8'))
        
        return snapshot
    
    def save(self, filepath: str):
        """Save plate to PNG file."""
        if self.image is None:
            raise RuntimeError("No image to save")
        
        self.image.save(filepath, 'PNG', optimize=False)
    
    @classmethod
    def load(cls, filepath: str) -> 'HolographicPlate':
        """Load plate from PNG file."""
        image = Image.open(filepath)
        plate = cls(image.width, image.height)
        plate.image = image
        return plate


class PlateManager:
    """Manages holographic plates for an Acorn engine."""
    
    def __init__(self, engine):
        self.engine = engine
        self.auto_save_enabled = False
        self.auto_save_interval = 100  # ticks
        self.last_save_tick = 0
        
    def create_plate(self) -> HolographicPlate:
        """Create a plate from current engine state."""
        snapshot = self.engine.get_snapshot()
        plate = HolographicPlate()
        plate.encode_universe(snapshot)
        return plate
    
    def save_plate(self, filepath: str):
        """Save current state as a plate."""
        plate = self.create_plate()
        plate.save(filepath)
        
        self.engine.log_event("plate_saved", {
            "filepath": filepath,
            "tick": self.engine.world.tick
        })
    
    def load_plate(self, filepath: str):
        """Load state from a plate."""
        plate = HolographicPlate.load(filepath)
        snapshot = plate.decode_universe(plate.image)
        
        self.engine.load_snapshot(snapshot)
        
        self.engine.log_event("plate_loaded", {
            "filepath": filepath,
            "tick": self.engine.world.tick
        })
    
    def enable_auto_save(self, interval_ticks: int = 100):
        """Enable automatic plate saving."""
        self.auto_save_enabled = True
        self.auto_save_interval = interval_ticks
        self.last_save_tick = self.engine.world.tick
    
    def disable_auto_save(self):
        """Disable automatic plate saving."""
        self.auto_save_enabled = False
    
    def check_auto_save(self):
        """Check if auto-save should trigger."""
        if not self.auto_save_enabled:
            return
        
        current_tick = self.engine.world.tick
        if current_tick - self.last_save_tick >= self.auto_save_interval:
            filepath = f"autosave_tick_{current_tick}.png"
            self.save_plate(filepath)
            self.last_save_tick = current_tick


if __name__ == "__main__":
    # Self-test
    print("Holographic Plate System - Self Test")
    print("=" * 50)
    
    # Create test engine
    from acorn.engine import WorldState, AcornEngine, EntityType, Position
    
    world = WorldState(20, 20, seed=42)
    engine = AcornEngine(world, {"iss_enabled": False})
    
    # Add some entities
    for i in range(5):
        engine.create_entity(EntityType.BASIC, Position(i * 4, i * 4))
    
    print(f"Created engine with {len(world.entities)} entities")
    
    # Create plate
    manager = PlateManager(engine)
    plate = manager.create_plate()
    print(f"Created plate: {plate.width}x{plate.height}")
    
    # Save plate
    test_file = "/tmp/test_plate.png"
    plate.save(test_file)
    print(f"Saved plate to {test_file}")
    
    # Load plate
    loaded_plate = HolographicPlate.load(test_file)
    print(f"Loaded plate: {loaded_plate.width}x{loaded_plate.height}")
    
    # Decode universe
    decoded = loaded_plate.decode_universe(loaded_plate.image)
    print(f"Decoded universe: tick={decoded['world']['tick']}, entities={len(decoded['world']['entities'])}")
    
    # Verify integrity
    original_tick = engine.world.tick
    decoded_tick = decoded['world']['tick']
    assert original_tick == decoded_tick, "Tick mismatch!"
    
    print("\n✓ Self-test passed!")
    print(f"✓ Plate visualization saved to {test_file}")
