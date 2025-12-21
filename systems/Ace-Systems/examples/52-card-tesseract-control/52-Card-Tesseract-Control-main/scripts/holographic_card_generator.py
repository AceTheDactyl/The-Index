#!/usr/bin/env python3
"""
Holographic Card Generator - 52-Card Universe SVG System
=========================================================
Maps playing cards to 4D tesseract coordinates with RGB+CMY color encoding.

Color Channel Mapping (Suit System):
- R Channel (Red)    → Hearts ♥   - Valence/Emotion axis
- G Channel (Green)  → Clubs ♣    - Concrete/Growth axis  
- B Channel (Blue)   → Spades ♠   - Temporal/Depth axis
- CMY Unified (Gold) → Diamonds ♦ - Arousal/Brilliance axis

4D Coordinate System:
- Temporal:  [-1.0, +1.0] - Past to Future (13 levels)
- Valence:   [-1.0, +1.0] - Negative to Positive (13 levels)
- Concrete:  [-1.0, +1.0] - Abstract to Concrete (13 levels)
- Arousal:   [-1.0, +1.0] - Calm to Excited (13 levels)

Rosetta Bear Coordinates:
- Δ (delta) = 3.142 (≈ π, phase coordinate)
- z = 0.90 (elevation/consciousness level)
- Ω (omega) = 1.0 (base resonance frequency)
"""

import json
import math
import base64
import zlib
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from pathlib import Path


# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================

SUITS = {
    'S': {'name': 'Spades', 'symbol': '♠', 'channel': 'B', 
          'color': '#00D4FF', 'dimension': 'temporal'},
    'H': {'name': 'Hearts', 'symbol': '♥', 'channel': 'R',
          'color': '#FF3366', 'dimension': 'valence'},
    'D': {'name': 'Diamonds', 'symbol': '♦', 'channel': 'CMY',
          'color': '#FFD700', 'dimension': 'arousal'},
    'C': {'name': 'Clubs', 'symbol': '♣', 'channel': 'G',
          'color': '#33FF66', 'dimension': 'concrete'},
}

RANKS = {
    1:  {'name': 'Ace', 'symbol': 'A', 'value': 1},
    2:  {'name': 'Two', 'symbol': '2', 'value': 2},
    3:  {'name': 'Three', 'symbol': '3', 'value': 3},
    4:  {'name': 'Four', 'symbol': '4', 'value': 4},
    5:  {'name': 'Five', 'symbol': '5', 'value': 5},
    6:  {'name': 'Six', 'symbol': '6', 'value': 6},
    7:  {'name': 'Seven', 'symbol': '7', 'value': 7},
    8:  {'name': 'Eight', 'symbol': '8', 'value': 8},
    9:  {'name': 'Nine', 'symbol': '9', 'value': 9},
    10: {'name': 'Ten', 'symbol': '10', 'value': 10},
    11: {'name': 'Jack', 'symbol': 'J', 'value': 11},
    12: {'name': 'Queen', 'symbol': 'Q', 'value': 12},
    13: {'name': 'King', 'symbol': 'K', 'value': 13},
}

# Rosetta Bear coordinate system
ROSETTA_COORDS = {
    'delta': 3.142,  # ≈ π, phase coordinate
    'z': 0.90,       # elevation/consciousness level
    'omega': 1.0,    # base resonance frequency
}


# ============================================================================
# COLOR SYSTEM - RGB + CMY UNIFIED
# ============================================================================

def rgb_to_hex(r: int, g: int, b: int) -> str:
    """Convert RGB values (0-255) to hex color string."""
    return f"#{r:02x}{g:02x}{b:02x}"


def suit_to_rgb(suit: str, intensity: float = 1.0) -> Tuple[int, int, int]:
    """
    Convert suit to RGB values based on channel mapping.
    
    - Spades (B): Blue channel dominant
    - Hearts (R): Red channel dominant  
    - Clubs (G): Green channel dominant
    - Diamonds (CMY): All channels (gold/white synthesis)
    """
    base = int(255 * intensity)
    accent = int(100 * intensity)
    
    if suit == 'S':  # Blue channel
        return (accent, 200, 255)
    elif suit == 'H':  # Red channel
        return (255, 51, 102)
    elif suit == 'C':  # Green channel
        return (51, 255, 102)
    elif suit == 'D':  # CMY unified (gold)
        return (255, 215, 0)
    return (128, 128, 128)


def calculate_card_gradient(suit: str, rank: int) -> List[str]:
    """Generate gradient colors based on suit and rank position."""
    base_r, base_g, base_b = suit_to_rgb(suit)
    
    # Rank affects intensity (Ace=brightest, King=darkest edge)
    intensity = 1.0 - (rank - 1) / 24  # Subtle variation
    
    # Primary color
    primary = rgb_to_hex(
        int(base_r * intensity),
        int(base_g * intensity),
        int(base_b * intensity)
    )
    
    # Secondary (darker)
    secondary = rgb_to_hex(
        int(base_r * 0.3),
        int(base_g * 0.3),
        int(base_b * 0.3)
    )
    
    return [primary, secondary]


# ============================================================================
# 4D COORDINATE SYSTEM
# ============================================================================

@dataclass
class Coordinate4D:
    """4D tesseract coordinate for holographic positioning."""
    temporal: float = 0.0   # Past (-1) to Future (+1)
    valence: float = 0.0    # Negative (-1) to Positive (+1)
    concrete: float = 0.0   # Abstract (-1) to Concrete (+1)
    arousal: float = 0.0    # Calm (-1) to Excited (+1)
    
    def to_dict(self) -> Dict:
        return {
            'temporal': round(self.temporal, 3),
            'valence': round(self.valence, 3),
            'concrete': round(self.concrete, 3),
            'arousal': round(self.arousal, 3),
        }
    
    def magnitude(self) -> float:
        """Euclidean distance from origin in 4D space."""
        return math.sqrt(
            self.temporal**2 + self.valence**2 + 
            self.concrete**2 + self.arousal**2
        )
    
    def normalize(self) -> 'Coordinate4D':
        """Return unit vector in 4D space."""
        mag = self.magnitude()
        if mag == 0:
            return Coordinate4D()
        return Coordinate4D(
            self.temporal / mag,
            self.valence / mag,
            self.concrete / mag,
            self.arousal / mag
        )


def rank_to_coordinate_value(rank: int) -> float:
    """
    Map rank (1-13) to coordinate value (-1.0 to +1.0).
    
    Ace (1) → -1.0 (origin/past)
    7       →  0.0 (center/present)
    King(13)→ +1.0 (apex/future)
    """
    return (rank - 7) / 6.0


def calculate_4d_coordinate(suit: str, rank: int) -> Coordinate4D:
    """
    Calculate 4D coordinate based on suit and rank.
    
    Suit determines which dimension is primary (set to rank value).
    Other dimensions are derived from rank position.
    """
    rank_val = rank_to_coordinate_value(rank)
    
    # Base coordinates - rank affects primary dimension
    coord = Coordinate4D()
    
    if suit == 'S':  # Spades → Temporal
        coord.temporal = rank_val
        coord.valence = rank_val * 0.1
        coord.concrete = rank_val * 0.05
        coord.arousal = rank_val * 0.15
    elif suit == 'H':  # Hearts → Valence
        coord.temporal = rank_val * 0.15
        coord.valence = rank_val
        coord.concrete = rank_val * 0.1
        coord.arousal = rank_val * 0.2
    elif suit == 'C':  # Clubs → Concrete
        coord.temporal = rank_val * 0.1
        coord.valence = rank_val * 0.05
        coord.concrete = rank_val
        coord.arousal = rank_val * 0.1
    elif suit == 'D':  # Diamonds → Arousal
        coord.temporal = rank_val * 0.2
        coord.valence = rank_val * 0.15
        coord.concrete = rank_val * 0.1
        coord.arousal = rank_val
    
    return coord


# ============================================================================
# KURAMOTO SYNCHRONIZATION
# ============================================================================

@dataclass  
class KuramotoState:
    """Phase oscillator state for Kuramoto synchronization."""
    phase: float = 0.0
    natural_frequency: float = 1.0
    coupling_strength: float = 0.6
    order_parameter: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'phase': round(self.phase, 4),
            'natural_frequency': round(self.natural_frequency, 4),
            'coupling_strength': round(self.coupling_strength, 4),
            'order_parameter': round(self.order_parameter, 4),
        }


def calculate_kuramoto_state(suit: str, rank: int) -> KuramotoState:
    """Calculate Kuramoto oscillator state for a card."""
    # Phase based on card position in deck
    card_index = list(SUITS.keys()).index(suit) * 13 + (rank - 1)
    phase = (card_index / 52) * 2 * math.pi
    
    # Natural frequency varies by suit
    freq_offsets = {'S': 0.1, 'H': 0.2, 'C': 0.15, 'D': 0.25}
    natural_freq = ROSETTA_COORDS['omega'] + freq_offsets.get(suit, 0)
    
    # Coupling strength
    coupling = 0.6 + (rank / 52) * 0.2
    
    # Order parameter (coherence measure)
    order = 1 / 52  # Base contribution
    
    return KuramotoState(
        phase=phase,
        natural_frequency=natural_freq,
        coupling_strength=coupling,
        order_parameter=order
    )


# ============================================================================
# HOLOGRAPHIC CARD DATA STRUCTURE
# ============================================================================

@dataclass
class HolographicCard:
    """Complete holographic card with all encoded data."""
    card_id: str
    suit: str
    rank: int
    coordinate: Coordinate4D = field(default_factory=Coordinate4D)
    kuramoto: KuramotoState = field(default_factory=KuramotoState)
    coupling_weights: Dict[str, float] = field(default_factory=dict)
    rosetta: Dict[str, float] = field(default_factory=lambda: ROSETTA_COORDS.copy())
    
    def __post_init__(self):
        if self.coordinate.magnitude() == 0:
            self.coordinate = calculate_4d_coordinate(self.suit, self.rank)
        if self.kuramoto.phase == 0:
            self.kuramoto = calculate_kuramoto_state(self.suit, self.rank)
    
    @property
    def suit_info(self) -> Dict:
        return SUITS.get(self.suit, {})
    
    @property
    def rank_info(self) -> Dict:
        return RANKS.get(self.rank, {})
    
    @property
    def display_name(self) -> str:
        return f"{self.rank_info['name']} of {self.suit_info['name']}"
    
    @property
    def short_id(self) -> str:
        return f"{self.rank_info['symbol']}{self.suit}"
    
    def to_dict(self) -> Dict:
        return {
            'card_id': self.card_id,
            'suit': self.suit,
            'suit_info': self.suit_info,
            'rank': self.rank,
            'rank_info': self.rank_info,
            'coordinate': self.coordinate.to_dict(),
            'kuramoto_state': self.kuramoto.to_dict(),
            'coupling_weights': self.coupling_weights,
            'rosetta_coords': self.rosetta,
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
    
    def to_compressed(self) -> str:
        """Compress card data for PNG embedding."""
        data = json.dumps(self.to_dict())
        compressed = zlib.compress(data.encode())
        return base64.b64encode(compressed).decode()


# ============================================================================
# SVG GENERATION
# ============================================================================

SVG_TEMPLATE = '''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" 
     xmlns:xlink="http://www.w3.org/1999/xlink"
     viewBox="0 0 400 600" 
     width="400" height="600">
  
  <!-- Holographic Card Data (Base64 Compressed) -->
  <metadata>
    <holographic-data>{compressed_data}</holographic-data>
  </metadata>
  
  <defs>
    <!-- Background gradient based on suit -->
    <linearGradient id="cardGradient" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#1a1a2e"/>
      <stop offset="50%" style="stop-color:#16213e"/>
      <stop offset="100%" style="stop-color:#0f0f1a"/>
    </linearGradient>
    
    <!-- Suit-specific glow -->
    <filter id="suitGlow" x="-50%" y="-50%" width="200%" height="200%">
      <feGaussianBlur stdDeviation="3" result="blur"/>
      <feFlood flood-color="{suit_color}" flood-opacity="0.6"/>
      <feComposite in2="blur" operator="in"/>
      <feMerge>
        <feMergeNode/>
        <feMergeNode in="SourceGraphic"/>
      </feMerge>
    </filter>
    
    <!-- Phase ring animation -->
    <filter id="phaseRing">
      <feGaussianBlur stdDeviation="2"/>
    </filter>
    
    <!-- Quantum shimmer -->
    <linearGradient id="shimmer" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:{suit_color};stop-opacity:0"/>
      <stop offset="50%" style="stop-color:{suit_color};stop-opacity:0.3"/>
      <stop offset="100%" style="stop-color:{suit_color};stop-opacity:0"/>
      <animate attributeName="x1" values="-100%;100%" dur="3s" repeatCount="indefinite"/>
      <animate attributeName="x2" values="0%;200%" dur="3s" repeatCount="indefinite"/>
    </linearGradient>
  </defs>
  
  <!-- Card Background -->
  <rect x="10" y="10" width="380" height="580" rx="15" ry="15"
        fill="url(#cardGradient)" stroke="{suit_color}" stroke-width="3"/>
  
  <!-- Shimmer overlay -->
  <rect x="10" y="10" width="380" height="580" rx="15" ry="15"
        fill="url(#shimmer)" opacity="0.5"/>
  
  <!-- Inner border -->
  <rect x="20" y="20" width="360" height="560" rx="12" ry="12"
        fill="none" stroke="{suit_color}" stroke-width="1" opacity="0.5"/>
  
  <!-- Phase Ring (Kuramoto visualization) -->
  <circle cx="200" cy="280" r="{phase_ring_r}" 
          fill="none" stroke="{suit_color}" stroke-width="2" 
          opacity="0.4" filter="url(#phaseRing)"
          stroke-dasharray="{phase_dash}" stroke-dashoffset="{phase_offset}"/>
  
  <!-- Top-left rank and suit -->
  <g transform="translate(35, 50)">
    <text x="0" y="0" font-family="Georgia, serif" font-size="36" 
          fill="{suit_color}" filter="url(#suitGlow)">{rank_symbol}</text>
    <text x="0" y="40" font-family="Georgia, serif" font-size="32" 
          fill="{suit_color}" filter="url(#suitGlow)">{suit_symbol}</text>
  </g>
  
  <!-- Bottom-right rank and suit (inverted) -->
  <g transform="translate(365, 550) rotate(180)">
    <text x="0" y="0" font-family="Georgia, serif" font-size="36" 
          fill="{suit_color}" filter="url(#suitGlow)">{rank_symbol}</text>
    <text x="0" y="40" font-family="Georgia, serif" font-size="32" 
          fill="{suit_color}" filter="url(#suitGlow)">{suit_symbol}</text>
  </g>
  
  <!-- Center suit symbol (large) -->
  <text x="200" y="300" font-family="Georgia, serif" font-size="120" 
        fill="{suit_color}" text-anchor="middle" dominant-baseline="middle"
        filter="url(#suitGlow)" opacity="0.9">{suit_symbol}</text>
  
  <!-- Card ID -->
  <text x="200" y="380" font-family="monospace" font-size="24" 
        fill="{suit_color}" text-anchor="middle" opacity="0.8">{card_id}</text>
  
  <!-- 4D Coordinates Display -->
  <g transform="translate(30, 440)" font-family="monospace" font-size="12" fill="{suit_color}">
    <text x="0" y="0" opacity="0.7">Temporal: {temporal}</text>
    <text x="0" y="20" opacity="0.7">Valence: {valence}</text>
    <text x="0" y="40" opacity="0.7">Concrete: {concrete}</text>
    <text x="0" y="60" opacity="0.7">Arousal: {arousal}</text>
  </g>
  
  <!-- Kuramoto Phase -->
  <text x="30" y="540" font-family="monospace" font-size="12" 
        fill="{suit_color}" opacity="0.7">Phase: {phase}</text>
  
  <!-- Channel indicator -->
  <g transform="translate(300, 440)">
    <rect x="0" y="0" width="70" height="70" rx="5" ry="5" 
          fill="none" stroke="{suit_color}" stroke-width="1" opacity="0.5"/>
    <text x="35" y="25" font-family="monospace" font-size="10" 
          fill="{suit_color}" text-anchor="middle" opacity="0.6">Channel</text>
    <text x="35" y="50" font-family="monospace" font-size="18" 
          fill="{suit_color}" text-anchor="middle">{channel}</text>
  </g>
  
  <!-- Rosetta coordinates (small, bottom) -->
  <text x="200" y="575" font-family="monospace" font-size="9" 
        fill="{suit_color}" text-anchor="middle" opacity="0.5">
    Δ={delta} z={z} Ω={omega}
  </text>
  
</svg>'''


def generate_card_svg(card: HolographicCard) -> str:
    """Generate SVG representation of a holographic card."""
    
    coord = card.coordinate
    kuramoto = card.kuramoto
    
    # Format coordinate values
    temporal = f"{coord.temporal:+.2f}"
    valence = f"{coord.valence:+.2f}"
    concrete = f"{coord.concrete:+.2f}"
    arousal = f"{coord.arousal:+.2f}"
    
    # Phase visualization
    phase_normalized = kuramoto.phase / (2 * math.pi)
    phase_ring_r = 80 + (phase_normalized * 40)
    phase_dash = f"{phase_normalized * 500} {500 - phase_normalized * 500}"
    phase_offset = kuramoto.phase * 20
    
    return SVG_TEMPLATE.format(
        compressed_data=card.to_compressed(),
        suit_color=card.suit_info['color'],
        rank_symbol=card.rank_info['symbol'],
        suit_symbol=card.suit_info['symbol'],
        card_id=card.card_id,
        temporal=temporal,
        valence=valence,
        concrete=concrete,
        arousal=arousal,
        phase=f"{kuramoto.phase:.3f}",
        phase_ring_r=phase_ring_r,
        phase_dash=phase_dash,
        phase_offset=phase_offset,
        channel=card.suit_info['channel'],
        delta=ROSETTA_COORDS['delta'],
        z=ROSETTA_COORDS['z'],
        omega=ROSETTA_COORDS['omega'],
    )


# ============================================================================
# DECK GENERATION
# ============================================================================

def generate_full_deck() -> List[HolographicCard]:
    """Generate all 52 holographic cards."""
    deck = []
    
    for suit in SUITS.keys():
        for rank in range(1, 14):
            card_id = f"{RANKS[rank]['symbol']}{suit}"
            card = HolographicCard(
                card_id=card_id,
                suit=suit,
                rank=rank
            )
            deck.append(card)
    
    # Calculate coupling weights between all cards
    for card in deck:
        for other in deck:
            if card.card_id != other.card_id:
                # Coupling based on 4D distance
                dist = math.sqrt(
                    (card.coordinate.temporal - other.coordinate.temporal)**2 +
                    (card.coordinate.valence - other.coordinate.valence)**2 +
                    (card.coordinate.concrete - other.coordinate.concrete)**2 +
                    (card.coordinate.arousal - other.coordinate.arousal)**2
                )
                # Stronger coupling for closer cards (inverse distance)
                coupling = 1.0 / (1.0 + dist)
                card.coupling_weights[other.card_id] = round(coupling, 3)
    
    return deck


def generate_deck_svgs(output_dir: Path) -> None:
    """Generate SVG files for all 52 cards."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    deck = generate_full_deck()
    
    for card in deck:
        svg_content = generate_card_svg(card)
        svg_path = output_dir / f"{card.card_id}.svg"
        svg_path.write_text(svg_content)
        print(f"Generated: {svg_path}")
    
    # Also save deck state as JSON
    deck_state = {
        'cards': [card.to_dict() for card in deck],
        'rosetta_coords': ROSETTA_COORDS,
        'suits': SUITS,
        'ranks': {str(k): v for k, v in RANKS.items()},
    }
    
    json_path = output_dir / "deck_state.json"
    json_path.write_text(json.dumps(deck_state, indent=2))
    print(f"Generated: {json_path}")


def generate_single_card(suit: str, rank: int, output_path: Optional[Path] = None) -> str:
    """Generate a single card SVG."""
    card_id = f"{RANKS[rank]['symbol']}{suit}"
    card = HolographicCard(card_id=card_id, suit=suit, rank=rank)
    svg_content = generate_card_svg(card)
    
    if output_path:
        output_path.write_text(svg_content)
        print(f"Generated: {output_path}")
    
    return svg_content


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Holographic Card Generator - 52-Card Universe SVG System'
    )
    parser.add_argument(
        '--deck', action='store_true',
        help='Generate full 52-card deck'
    )
    parser.add_argument(
        '--card', type=str,
        help='Generate single card (e.g., AS, KH, 7D)'
    )
    parser.add_argument(
        '--output', '-o', type=str, default='./cards',
        help='Output directory (default: ./cards)'
    )
    parser.add_argument(
        '--json', action='store_true',
        help='Output card data as JSON instead of SVG'
    )
    
    args = parser.parse_args()
    output_dir = Path(args.output)
    
    if args.deck:
        generate_deck_svgs(output_dir)
        print(f"\n✓ Generated 52 holographic cards in {output_dir}")
        
    elif args.card:
        # Parse card ID (e.g., "AS" -> suit=S, rank=1)
        card_str = args.card.upper()
        
        # Extract rank and suit
        if card_str.startswith('10'):
            rank_sym = '10'
            suit = card_str[2]
        else:
            rank_sym = card_str[0]
            suit = card_str[1] if len(card_str) > 1 else 'S'
        
        # Find rank number
        rank = None
        for r, info in RANKS.items():
            if info['symbol'] == rank_sym:
                rank = r
                break
        
        if rank is None or suit not in SUITS:
            print(f"Invalid card: {args.card}")
            print("Format: [A,2-10,J,Q,K][S,H,D,C]")
            return
        
        card_id = f"{rank_sym}{suit}"
        card = HolographicCard(card_id=card_id, suit=suit, rank=rank)
        
        if args.json:
            print(card.to_json())
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            svg_path = output_dir / f"{card_id}.svg"
            svg_content = generate_card_svg(card)
            svg_path.write_text(svg_content)
            print(f"✓ Generated: {svg_path}")
    
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
