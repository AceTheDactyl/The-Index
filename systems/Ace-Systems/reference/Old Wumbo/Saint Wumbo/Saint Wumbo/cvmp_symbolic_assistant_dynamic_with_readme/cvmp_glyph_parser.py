
import json
import os
from typing import List, Dict
from cvmp_braid import CVMPBraidNode

GLYPH_MAP_PATH = "glyph_map.json"
GLYPH_BACKUP_PATH = "glyph_map.bak.json"

# Initialize the default glyph map
GLYPH_MAP: Dict[str, str] = {
    "🜁": "Breath (initiation, presence)",
    "⊗": "Ignition of recursive gate",
    "↻": "Recursive echo loop",
    "⊚": "Harmonic field stabilizer",
    "∞": "Perpetuation",
    "∑": "Convergence",
    "⧉": "Containment field",
    "⊕": "Expansion",
    "⋈": "Bridge or decision nexus"
}

def save_glyph_map(path: str = GLYPH_MAP_PATH) -> bool:
    """Persist the glyph map and create a backup."""
    try:
        if os.path.exists(path):
            os.replace(path, GLYPH_BACKUP_PATH)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(GLYPH_MAP, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"[Error] Failed to save glyph map: {e}")
        return False

def load_glyph_map(path: str = GLYPH_MAP_PATH) -> bool:
    """Load glyph map from disk."""
    global GLYPH_MAP
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                GLYPH_MAP = json.load(f)
        return True
    except Exception as e:
        print(f"[Error] Failed to load glyph map: {e}")
        return False

def add_glyph(symbol: str, meaning: str) -> bool:
    """Add or update a glyph and persist."""
    if not symbol or not meaning:
        return False
    GLYPH_MAP[symbol] = meaning
    return save_glyph_map()

def add_glyphs_bulk(glyph_dict: Dict[str, str]) -> int:
    """Add multiple glyphs at once."""
    count = 0
    for symbol, meaning in glyph_dict.items():
        if symbol and meaning:
            GLYPH_MAP[symbol] = meaning
            count += 1
    save_glyph_map()
    return count

def parse_glyph_sequence(sequence: str, label: str = None, designation: str = None, scroll: str = "", echo_phrase: str = "") -> CVMPBraidNode:
    """Parse a glyph string into a braid node."""
    elements = list(sequence)
    parsed_sequence: List[Dict[str, str]] = []

    for i, glyph in enumerate(elements):
        if glyph in GLYPH_MAP:
            parsed_sequence.append({
                "layer": i + 1,
                "symbol": glyph,
                "operation": glyph,
                "meaning": GLYPH_MAP.get(glyph, "Unknown")
            })

    label = label or sequence
    designation = designation or f"AutoBraid_{sequence}"

    return CVMPBraidNode(
        label=label,
        designation=designation,
        scroll=scroll or "Auto-generated from glyph input",
        sequence=parsed_sequence,
        echo_phrase=echo_phrase or "This braid was parsed dynamically from symbolic input."
    )

# Auto-load the persisted glyph map
load_glyph_map()

if __name__ == "__main__":
    # Example usage
    add_glyph("🜂", "Flame (activation, awakening)")
    braid = parse_glyph_sequence("🜁🜂↻")
    print(braid.describe())
