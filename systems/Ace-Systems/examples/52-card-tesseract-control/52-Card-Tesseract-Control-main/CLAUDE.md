# CLAUDE.md - Holographic Card Game Development Instructions

## Project Overview

This repository implements **Quantum Resonance**, a deck-building card game using a 52-card holographic universe. Each card encodes 4D coordinates, Kuramoto phase states, and 2,652 coupling relationships that create emergent gameplay through mathematical structure.

**Core Design Principle**: The game mechanics ARE the mathematics. No arbitrary rules - scoring, combos, and interactions derive directly from the underlying coordinate system.

---

## Repository Structure

```
holographic-cards/
├── CLAUDE.md                    # THIS FILE - Development instructions
├── RULES.md                     # Complete game rulebook
├── scripts/
│   ├── holographic_card_generator.py   # Card SVG generation
│   ├── game_engine.py                  # Core game logic
│   ├── faction_abilities.py            # Asymmetric faction powers
│   ├── scoring.py                      # Mathematical scoring system
│   ├── deck_validator.py               # Deck legality checker
│   └── ai_opponent.py                  # AI player logic
├── data/
│   ├── deck_state.json                 # All 52 cards with full data
│   ├── coupling_matrix.json            # 52x52 coupling weights
│   ├── faction_cards.json              # Faction card assignments
│   └── prebuilt_decks/                 # Starter deck configurations
├── assets/cards/                       # 52 SVG card files
├── tests/
│   ├── test_scoring.py
│   ├── test_factions.py
│   ├── test_game_flow.py
│   └── test_deck_validity.py
├── docs/
│   ├── GAME_DESIGN.md                  # Design philosophy
│   ├── FACTION_GUIDE.md                # Faction strategy guide
│   ├── MATHEMATICAL_FOUNDATION.md      # Technical details
│   └── API_REFERENCE.md                # Engine API documentation
└── web/                                # Optional web interface
    ├── index.html
    ├── game.js
    └── style.css
```

---

## Development Commands

### Card Generation
```bash
# Generate single card
python scripts/holographic_card_generator.py --card AS --output assets/cards

# Generate full deck
python scripts/holographic_card_generator.py --deck --output assets/cards

# Export card data as JSON
python scripts/holographic_card_generator.py --card AS --json
```

### Game Engine
```bash
# Run game simulation
python scripts/game_engine.py --players 2 --faction1 spades --faction2 hearts

# Validate deck
python scripts/deck_validator.py --deck data/prebuilt_decks/spades_tempo.json

# Run AI match
python scripts/ai_opponent.py --difficulty hard --faction clubs
```

### Testing
```bash
# Run all tests
python -m pytest tests/ -v

# Test specific module
python -m pytest tests/test_scoring.py -v
```

---

## Core Mathematical Systems

### 1. 4D Coordinate System

Every card has coordinates in 4D tesseract space:

```python
@dataclass
class Coordinate4D:
    temporal: float   # [-1.0, +1.0] Past to Future
    valence: float    # [-1.0, +1.0] Negative to Positive emotion
    concrete: float   # [-1.0, +1.0] Abstract to Concrete
    arousal: float    # [-1.0, +1.0] Calm to Excited
```

**Suit-Dimension Mapping:**
| Suit | Primary Dimension | Channel |
|------|-------------------|---------|
| ♠ Spades | Temporal | Blue (B) |
| ♥ Hearts | Valence | Red (R) |
| ♦ Diamonds | Arousal | CMY (Gold) |
| ♣ Clubs | Concrete | Green (G) |

**Rank-Value Mapping:**
```python
def rank_to_value(rank: int) -> float:
    """Rank 1-13 maps to [-1.0, +1.0]"""
    return (rank - 7) / 6.0

# Results:
# Ace (1)  → -1.000
# 7        →  0.000
# King(13) → +1.000
```

### 2. Coupling Network

2,652 weighted edges connect all card pairs:

```python
def calculate_coupling(card_a: Card, card_b: Card) -> float:
    """Coupling strength based on 4D distance (inverse)."""
    distance = euclidean_4d(card_a.coordinate, card_b.coordinate)
    return 1.0 / (1.0 + distance)
```

**Usage in gameplay:**
- Coupling > 0.7 = Strong bond (combo eligible)
- Coupling 0.4-0.7 = Moderate (chain eligible)
- Coupling < 0.4 = Weak (no synergy)

### 3. Kuramoto Phase Synchronization

Each card has an oscillator phase:

```python
@dataclass
class KuramotoState:
    phase: float              # [0, 2π] Current phase angle
    natural_frequency: float  # Base oscillation rate
    coupling_strength: float  # How strongly it pulls others
```

**Coherence Calculation:**
```python
def calculate_coherence(cards: List[Card]) -> float:
    """Order parameter R ∈ [0, 1]. Higher = more synchronized."""
    n = len(cards)
    if n == 0:
        return 0.0
    
    sum_cos = sum(math.cos(c.kuramoto.phase) for c in cards)
    sum_sin = sum(math.sin(c.kuramoto.phase) for c in cards)
    
    return math.sqrt(sum_cos**2 + sum_sin**2) / n
```

---

## Game Rules Implementation

### Win Condition
First player to reach **100 Resonance Points** wins.

### Turn Structure
```python
class TurnPhase(Enum):
    DRAW = "draw"           # Draw 2 cards
    MAIN = "main"           # Play cards, activate abilities
    RESONANCE = "resonance" # Calculate phase bonuses
    DISCARD = "discard"     # Hand limit enforcement (7 cards)
    END = "end"             # Cleanup, pass turn
```

### Scoring System

Points are calculated from three sources:

```python
def calculate_turn_score(played_cards: List[Card], 
                         faction: Faction,
                         field_state: FieldState) -> int:
    """Total points = Base + Cluster + Resonance + Faction"""
    
    base = sum(card.rank for card in played_cards)
    
    cluster_bonus = calculate_cluster_bonus(played_cards)
    resonance_bonus = calculate_resonance_bonus(played_cards)
    faction_bonus = faction.calculate_bonus(played_cards, field_state)
    
    return base + cluster_bonus + resonance_bonus + faction_bonus
```

#### Cluster Bonus (4D Proximity)
```python
def calculate_cluster_bonus(cards: List[Card]) -> int:
    """Bonus for tight 4D clustering."""
    if len(cards) < 2:
        return 0
    
    # Calculate centroid
    centroid = average_coordinate(cards)
    
    # Calculate average distance from centroid
    avg_distance = mean([
        euclidean_4d(card.coordinate, centroid) 
        for card in cards
    ])
    
    # Tighter cluster = higher bonus
    # Max bonus at avg_distance = 0, zero bonus at avg_distance >= 1.0
    if avg_distance >= 1.0:
        return 0
    
    return int((1.0 - avg_distance) * 20 * len(cards))
```

#### Resonance Bonus (Phase Coherence)
```python
def calculate_resonance_bonus(cards: List[Card]) -> int:
    """Bonus for phase-aligned cards."""
    coherence = calculate_coherence(cards)
    
    # Coherence thresholds
    if coherence >= 0.9:
        return 30  # Perfect resonance
    elif coherence >= 0.7:
        return 20  # Strong resonance
    elif coherence >= 0.5:
        return 10  # Moderate resonance
    else:
        return 0
```

#### Chain Bonus (Coupling Network)
```python
def calculate_chain_bonus(cards: List[Card]) -> int:
    """Bonus for playing cards with strong coupling paths."""
    if len(cards) < 2:
        return 0
    
    # Check if cards form a valid chain
    total_coupling = 0
    sorted_cards = sort_by_coupling_path(cards)
    
    for i in range(len(sorted_cards) - 1):
        coupling = get_coupling(sorted_cards[i], sorted_cards[i+1])
        if coupling < 0.4:  # Chain broken
            return total_coupling
        total_coupling += int(coupling * 10)
    
    # Unbroken chain bonus
    if len(cards) >= 3:
        total_coupling += 15
    
    return total_coupling
```

---

## Faction Implementation

### Faction Base Class
```python
@dataclass
class Faction:
    name: str
    suit: str
    primary_dimension: str
    passive_ability: str
    active_abilities: List[Ability]
    
    def calculate_bonus(self, cards: List[Card], field: FieldState) -> int:
        """Override in subclass."""
        raise NotImplementedError
    
    def can_activate(self, ability: Ability, game_state: GameState) -> bool:
        """Check if ability can be used."""
        raise NotImplementedError
```

### ♠ Spades Faction - Temporal Manipulation

**Theme**: Control time, see the future, reorder fate

**Passive - Temporal Echo**: 
When you play a Spades card, you may look at the top card of your deck.

**Active Abilities**:

```python
class SpadesFaction(Faction):
    def __init__(self):
        super().__init__(
            name="Temporal Weavers",
            suit="S",
            primary_dimension="temporal",
            passive_ability="temporal_echo",
            active_abilities=[
                Ability("rewind", cost=2, cooldown=2),
                Ability("foresight", cost=1, cooldown=1),
                Ability("time_lock", cost=3, cooldown=3),
            ]
        )
    
    def rewind(self, game_state: GameState) -> GameState:
        """Return last played card to hand. Cost: 2 cards from hand."""
        last_played = game_state.field.pop()
        game_state.current_player.hand.append(last_played)
        return game_state
    
    def foresight(self, game_state: GameState) -> List[Card]:
        """Look at top 3 cards of deck, reorder them."""
        top_3 = game_state.deck[:3]
        # Player chooses order (UI handles this)
        return top_3
    
    def time_lock(self, game_state: GameState, target_card: Card):
        """Target card cannot be played next turn."""
        game_state.locked_cards.append(target_card)
        game_state.lock_duration[target_card.id] = 1
    
    def calculate_bonus(self, cards: List[Card], field: FieldState) -> int:
        """Bonus for temporal axis alignment."""
        spades_cards = [c for c in cards if c.suit == 'S']
        if len(spades_cards) >= 2:
            # Bonus for temporal spread (controlling more of the axis)
            temporal_range = max(c.coordinate.temporal for c in spades_cards) - \
                           min(c.coordinate.temporal for c in spades_cards)
            return int(temporal_range * 15)
        return 0
```

### ♥ Hearts Faction - Valence Control

**Theme**: Emotional manipulation, buff/debuff, healing

**Passive - Empathic Bond**: 
+2 points for each Hearts card an opponent plays.

**Active Abilities**:

```python
class HeartsFaction(Faction):
    def __init__(self):
        super().__init__(
            name="Valence Shapers",
            suit="H",
            primary_dimension="valence",
            passive_ability="empathic_bond",
            active_abilities=[
                Ability("inspire", cost=1, cooldown=1),
                Ability("drain", cost=2, cooldown=2),
                Ability("emotional_surge", cost=3, cooldown=3),
            ]
        )
    
    def inspire(self, game_state: GameState, target_cards: List[Card]) -> None:
        """Shift target cards +0.2 on valence axis for scoring."""
        for card in target_cards:
            card.temp_valence_modifier = +0.2
    
    def drain(self, game_state: GameState, opponent: Player) -> int:
        """Steal points equal to opponent's last turn resonance bonus."""
        stolen = opponent.last_resonance_bonus
        opponent.score -= stolen
        game_state.current_player.score += stolen
        return stolen
    
    def emotional_surge(self, game_state: GameState) -> None:
        """Double valence contribution to cluster bonus this turn."""
        game_state.valence_multiplier = 2.0
    
    def calculate_bonus(self, cards: List[Card], field: FieldState) -> int:
        """Bonus for positive valence stacking."""
        hearts_cards = [c for c in cards if c.suit == 'H']
        positive_sum = sum(
            max(0, c.coordinate.valence) for c in hearts_cards
        )
        return int(positive_sum * 10)
```

### ♦ Diamonds Faction - Arousal Spikes

**Theme**: Burst damage, high risk/reward, volatility

**Passive - Brilliance**: 
When you play 3+ cards in one turn, gain +5 points.

**Active Abilities**:

```python
class DiamondsFaction(Faction):
    def __init__(self):
        super().__init__(
            name="Radiant Catalysts",
            suit="D",
            primary_dimension="arousal",
            passive_ability="brilliance",
            active_abilities=[
                Ability("flare", cost=1, cooldown=0),
                Ability("overcharge", cost=2, cooldown=2),
                Ability("supernova", cost=4, cooldown=4),
            ]
        )
    
    def flare(self, game_state: GameState, card: Card) -> int:
        """Discard a card to deal damage equal to its arousal * 5."""
        damage = abs(card.coordinate.arousal) * 5
        game_state.opponent.score -= int(damage)
        game_state.current_player.discard.append(card)
        return int(damage)
    
    def overcharge(self, game_state: GameState) -> None:
        """Next card played has triple arousal contribution."""
        game_state.arousal_multiplier = 3.0
        game_state.arousal_multiplier_count = 1  # Only next card
    
    def supernova(self, game_state: GameState) -> int:
        """Discard entire hand. Score = sum of all arousal values * 10."""
        hand = game_state.current_player.hand
        total_arousal = sum(abs(c.coordinate.arousal) for c in hand)
        score = int(total_arousal * 10)
        
        game_state.current_player.discard.extend(hand)
        game_state.current_player.hand = []
        game_state.current_player.score += score
        
        return score
    
    def calculate_bonus(self, cards: List[Card], field: FieldState) -> int:
        """Bonus for high arousal extremes."""
        diamond_cards = [c for c in cards if c.suit == 'D']
        extreme_bonus = sum(
            5 if abs(c.coordinate.arousal) > 0.7 else 0 
            for c in diamond_cards
        )
        return extreme_bonus
```

### ♣ Clubs Faction - Concrete Grounding

**Theme**: Stability, defense, resource accumulation

**Passive - Rooted**: 
Your cards cannot be moved or locked by opponent abilities.

**Active Abilities**:

```python
class ClubsFaction(Faction):
    def __init__(self):
        super().__init__(
            name="Foundation Builders",
            suit="C",
            primary_dimension="concrete",
            passive_ability="rooted",
            active_abilities=[
                Ability("fortify", cost=1, cooldown=1),
                Ability("growth", cost=2, cooldown=2),
                Ability("unshakeable", cost=3, cooldown=3),
            ]
        )
    
    def fortify(self, game_state: GameState, cards: List[Card]) -> None:
        """Selected cards gain +0.3 concrete modifier (better clustering)."""
        for card in cards:
            card.temp_concrete_modifier = +0.3
    
    def growth(self, game_state: GameState) -> None:
        """Draw 2 additional cards this turn."""
        game_state.current_player.draw(2)
    
    def unshakeable(self, game_state: GameState) -> None:
        """This turn, your score cannot decrease from opponent effects."""
        game_state.score_protected = True
    
    def calculate_bonus(self, cards: List[Card], field: FieldState) -> int:
        """Bonus for concrete value stacking (cumulative presence)."""
        clubs_cards = [c for c in cards if c.suit == 'C']
        
        # Count cards already on field from previous turns
        field_clubs = [c for c in field.all_cards if c.suit == 'C']
        
        # Synergy bonus for each clubs card already present
        synergy = len(field_clubs) * 3 * len(clubs_cards)
        
        return synergy
```

---

## Deck Building Rules

### Deck Composition
```python
DECK_RULES = {
    "min_cards": 20,
    "max_cards": 30,
    "max_copies": 2,        # Max 2 copies of any card (except faction)
    "faction_minimum": 8,   # At least 8 cards from chosen faction suit
    "faction_maximum": 15,  # No more than 15 from faction suit
}

def validate_deck(deck: List[Card], faction: Faction) -> Tuple[bool, List[str]]:
    """Validate deck legality. Returns (valid, list of violations)."""
    violations = []
    
    # Size check
    if len(deck) < DECK_RULES["min_cards"]:
        violations.append(f"Deck too small: {len(deck)} < {DECK_RULES['min_cards']}")
    if len(deck) > DECK_RULES["max_cards"]:
        violations.append(f"Deck too large: {len(deck)} > {DECK_RULES['max_cards']}")
    
    # Copy limit check
    from collections import Counter
    card_counts = Counter(c.card_id for c in deck)
    for card_id, count in card_counts.items():
        if count > DECK_RULES["max_copies"]:
            violations.append(f"Too many copies of {card_id}: {count}")
    
    # Faction card check
    faction_cards = [c for c in deck if c.suit == faction.suit]
    if len(faction_cards) < DECK_RULES["faction_minimum"]:
        violations.append(f"Too few faction cards: {len(faction_cards)}")
    if len(faction_cards) > DECK_RULES["faction_maximum"]:
        violations.append(f"Too many faction cards: {len(faction_cards)}")
    
    return (len(violations) == 0, violations)
```

### Deck Archetypes

```python
# Example prebuilt deck configurations

SPADES_TEMPO_DECK = {
    "name": "Temporal Assault",
    "faction": "spades",
    "strategy": "Fast plays with temporal manipulation",
    "cards": [
        # Core Spades (10)
        "AS", "AS", "2S", "3S", "4S", "5S", "6S", "7S", "JS", "QS",
        # Support - Low cost for speed (8)
        "2H", "3H", "2C", "3C", "2D", "3D", "4D", "5D",
        # Finishers (2)
        "KS", "KH"
    ],
    "key_combos": [
        ["AS", "2S", "3S"],  # Temporal chain
        ["7S", "7H", "7D"],  # Center alignment
    ]
}

HEARTS_CONTROL_DECK = {
    "name": "Emotional Tide",
    "faction": "hearts",
    "strategy": "Drain opponent while building valence stacks",
    "cards": [
        # Core Hearts (12)
        "AH", "AH", "3H", "4H", "5H", "6H", "7H", "8H", "9H", "JH", "QH", "KH",
        # Neutral support (8)
        "7S", "7C", "7D",  # Centers for clustering
        "AS", "AC", "AD",  # Origins for spread control
        "KS", "KC"         # Apex for range
    ],
    "key_combos": [
        ["JH", "QH", "KH"],  # High valence stack
        ["7H", "7S", "7C", "7D"],  # Perfect center cluster
    ]
}

DIAMONDS_BURST_DECK = {
    "name": "Solar Flare",
    "faction": "diamonds",
    "strategy": "High variance burst damage",
    "cards": [
        # Core Diamonds - focus on extremes (11)
        "AD", "AD", "2D", "3D", "JD", "JD", "QD", "QD", "KD", "KD", "KD",
        # High arousal support (9)
        "KH", "KS", "KC",  # All kings for apex
        "AS", "AH", "AC",  # All aces for origin
        "JH", "JS", "JC"   # Jacks for mid-high
    ],
    "key_combos": [
        ["KD", "KH", "KS", "KC"],  # All kings burst
        ["AD", "JD", "QD", "KD"],  # Diamond chain
    ]
}

CLUBS_FORTRESS_DECK = {
    "name": "Growing Bastion",
    "faction": "clubs",
    "strategy": "Slow build, unassailable late game",
    "cards": [
        # Core Clubs (13)
        "AC", "AC", "2C", "3C", "4C", "5C", "6C", "7C", "8C", "9C", "10C", "JC", "QC",
        # Defensive support (7)
        "7S", "7H", "7D",  # Centers for stability
        "5S", "5H", "5D", "6S"  # Mid-range for clustering
    ],
    "key_combos": [
        ["5C", "6C", "7C", "8C"],  # Tight concrete chain
        ["AC", "2C", "3C"],  # Origin cluster
    ]
}
```

---

## Game Engine Architecture

### State Management
```python
@dataclass
class GameState:
    players: List[Player]
    current_player_idx: int
    deck: List[Card]
    discard: List[Card]
    field: FieldState
    turn_number: int
    phase: TurnPhase
    
    # Temporary modifiers (clear at end of turn)
    valence_multiplier: float = 1.0
    arousal_multiplier: float = 1.0
    arousal_multiplier_count: int = 0
    score_protected: bool = False
    locked_cards: List[Card] = field(default_factory=list)
    
    def advance_phase(self) -> None:
        phases = list(TurnPhase)
        current_idx = phases.index(self.phase)
        self.phase = phases[(current_idx + 1) % len(phases)]
    
    def end_turn(self) -> None:
        """Clean up and pass to next player."""
        self.valence_multiplier = 1.0
        self.arousal_multiplier = 1.0
        self.score_protected = False
        self.current_player_idx = (self.current_player_idx + 1) % len(self.players)
        self.turn_number += 1
        self.phase = TurnPhase.DRAW

@dataclass
class Player:
    name: str
    faction: Faction
    hand: List[Card]
    deck: List[Card]
    discard: List[Card]
    score: int = 0
    ability_cooldowns: Dict[str, int] = field(default_factory=dict)
    last_resonance_bonus: int = 0
```

### Event System
```python
class GameEvent(Enum):
    CARD_PLAYED = "card_played"
    CARD_DRAWN = "card_drawn"
    ABILITY_ACTIVATED = "ability_activated"
    SCORE_CHANGED = "score_changed"
    TURN_ENDED = "turn_ended"
    GAME_ENDED = "game_ended"

class EventHandler:
    def __init__(self):
        self.listeners: Dict[GameEvent, List[Callable]] = defaultdict(list)
    
    def subscribe(self, event: GameEvent, callback: Callable) -> None:
        self.listeners[event].append(callback)
    
    def emit(self, event: GameEvent, data: Dict) -> None:
        for callback in self.listeners[event]:
            callback(data)
```

---

## Testing Requirements

### Unit Tests
```python
# tests/test_scoring.py

def test_cluster_bonus_perfect_overlap():
    """Cards at same coordinate should give max bonus."""
    cards = [create_card("7S"), create_card("7S")]  # Same card twice
    bonus = calculate_cluster_bonus(cards)
    assert bonus == 40  # Max for 2 cards

def test_cluster_bonus_opposite_corners():
    """Cards at opposite tesseract corners should give zero."""
    cards = [create_card("AS"), create_card("KH")]  # Origin vs apex
    bonus = calculate_cluster_bonus(cards)
    assert bonus == 0

def test_resonance_bonus_aligned_phases():
    """Cards with same phase should give max resonance."""
    cards = [create_card("AS"), create_card("2S"), create_card("3S")]
    # These have sequential phases, should be somewhat aligned
    bonus = calculate_resonance_bonus(cards)
    assert bonus >= 10

def test_chain_bonus_strong_coupling():
    """Cards with high coupling should chain."""
    cards = [create_card("AS"), create_card("2S")]  # Adjacent, same suit
    bonus = calculate_chain_bonus(cards)
    assert bonus > 0
```

### Integration Tests
```python
# tests/test_game_flow.py

def test_full_turn_sequence():
    """Complete turn should progress through all phases."""
    game = GameEngine()
    game.setup(["Alice", "Bob"], ["spades", "hearts"])
    
    assert game.state.phase == TurnPhase.DRAW
    game.execute_draw()
    assert len(game.current_player.hand) == 7  # 5 start + 2 draw
    
    game.advance_phase()
    assert game.state.phase == TurnPhase.MAIN
    
    # Play cards
    cards_to_play = game.current_player.hand[:3]
    game.play_cards(cards_to_play)
    
    game.advance_phase()
    assert game.state.phase == TurnPhase.RESONANCE
    
    score_before = game.current_player.score
    game.calculate_resonance()
    # Score should have changed
    assert game.current_player.score >= score_before
```

### Faction Tests
```python
# tests/test_factions.py

def test_spades_rewind():
    """Rewind should return last played card to hand."""
    game = setup_game_with_faction("spades")
    card = game.current_player.hand[0]
    game.play_cards([card])
    
    hand_size_before = len(game.current_player.hand)
    game.activate_ability("rewind")
    
    assert len(game.current_player.hand) == hand_size_before + 1
    assert card in game.current_player.hand

def test_diamonds_supernova():
    """Supernova should discard hand and score based on arousal."""
    game = setup_game_with_faction("diamonds")
    game.current_player.hand = [
        create_card("KD"),  # arousal = 1.0
        create_card("AD"),  # arousal = -1.0
    ]
    
    score_before = game.current_player.score
    game.activate_ability("supernova")
    
    assert len(game.current_player.hand) == 0
    assert game.current_player.score > score_before
```

---

## Development Workflow

### Adding New Features

1. **Update Data Models** (if needed)
   - Modify `holographic_card_generator.py` for card structure changes
   - Regenerate `deck_state.json` and all SVGs

2. **Implement Logic**
   - Add to appropriate module (`scoring.py`, `faction_abilities.py`, etc.)
   - Follow existing patterns for consistency

3. **Write Tests First**
   - Create test cases before implementation
   - Cover edge cases and interactions

4. **Update Documentation**
   - Add to `RULES.md` for player-facing changes
   - Update `API_REFERENCE.md` for code changes

5. **Validate Deck Interactions**
   - Run `deck_validator.py` against all prebuilt decks
   - Test new cards/abilities don't break existing combos

### Debugging Commands
```bash
# Dump card data for inspection
python -c "from scripts.holographic_card_generator import *; print(HolographicCard('AS', 'S', 1).to_json())"

# Check coupling between two cards
python -c "from scripts.game_engine import *; print(get_coupling('AS', 'KH'))"

# Simulate 100 games for balance testing
python scripts/ai_opponent.py --simulate 100 --faction1 spades --faction2 diamonds
```

---

## Code Style Guidelines

- Use type hints everywhere
- Dataclasses for all data structures
- Pure functions where possible (no side effects)
- Document mathematical formulas in docstrings
- Keep game logic separate from UI/IO

---

## Current Development Status

- [x] Card generation system
- [x] 4D coordinate mapping
- [x] Coupling weight calculation
- [x] Kuramoto phase states
- [ ] Game engine core loop
- [ ] Faction ability implementations
- [ ] Scoring system
- [ ] Deck validator
- [ ] AI opponent
- [ ] Web interface
- [ ] Comprehensive tests

---

## Key Formulas Reference

```
Euclidean 4D Distance:
d = √[(t₁-t₂)² + (v₁-v₂)² + (c₁-c₂)² + (a₁-a₂)²]

Coupling Weight:
w = 1 / (1 + d)

Kuramoto Order Parameter:
R = (1/N) × √[(Σcos(θᵢ))² + (Σsin(θᵢ))²]

Cluster Bonus:
B = (1 - avg_distance) × 20 × n_cards  [if avg_distance < 1]

Rank to Coordinate:
value = (rank - 7) / 6
```

---

*Last Updated: Session Active*
*Tails to 7D, Acorns Ready*
