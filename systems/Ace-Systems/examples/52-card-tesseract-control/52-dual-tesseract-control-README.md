# Quantum Resonance

**A deck-building card game using a 52-card holographic universe**

*First to 1,000 Resonance Points wins!*

**Live Site:** [https://acethedactyl.github.io/52-Card-Tesseract-Control](https://acethedactyl.github.io/52-Card-Tesseract-Control)

---

## Overview

**Quantum Resonance** is a strategic card game where players harness the mathematical structure of a 4D tesseract. Each of the 52 cards encodes coordinates in 4-dimensional space, Kuramoto phase states, and 2,652 coupling relationships that create emergent gameplay through mathematical structure.

| Players | Duration | Win Condition |
|---------|----------|---------------|
| 2-4 | 45-90 min | 1,000 Resonance Points |

### Core Mechanics

- **Spatial Clustering** - Play cards close together in 4D tesseract space for cluster bonuses
- **Phase Synchronization** - Align Kuramoto oscillator phases for resonance bonuses
- **Chain Formation** - Link cards through the coupling network for chain bonuses
- **Faction Abilities** - Master unique passive and active powers tied to each suit

---

## Game Formats

### Format A: Constructed (20-30 cards)
Players bring pre-built decks following construction rules. Best for experienced players.

### Format B: Shared Deck (52 cards)
Both players draw from a single shuffled 52-card deck. Great for learning.

### Format C: Dual Tesseract (104 cards) — Competitive Standard

The definitive competitive format. Each player owns a complete 52-card holographic deck and secretly constructs their play deck before the match.

#### Dual Tesseract Setup

```
Player 1: [52 cards] ─┐
                      ├─► 104 total cards in play
Player 2: [52 cards] ─┘
```

1. **Each Player Brings 52** - Both players arrive with their own complete holographic deck
2. **Choose Factions Openly** - Declare factions simultaneously (public information)
3. **Discrete Build Phase** - Secretly select 20-30 cards; remainder becomes your Sideboard
4. **Reveal Deck Size Only** - Don't reveal contents, just card count
5. **Shuffle and Draw 5** - Standard game start
6. **Tesseract Collision** - Both players operate in parallel 4D spaces; cards interact only through scoring and abilities

#### Tesseract Collision Dynamics

- Both players operate in **parallel 4D coordinate spaces**
- Cards never physically mix between decks
- Interaction occurs only through **scoring** and **faction abilities**
- Each player's discard pile remains separate
- **Mirror matches allowed** - both can play the same faction

#### Sideboard Rules

- Sideboard = the 22-32 cards not selected for your play deck
- Sideboard stays **face-down and out of play** during a game
- Between games in a match: swap up to **5 cards** (1-for-1 exchanges only)
- Deck must still follow construction rules after sideboarding

#### Best-of-Three Match Structure

| Phase | Action |
|-------|--------|
| Game 1 | Play with initial build |
| Sideboard Window | 3 minutes to make swaps |
| Game 2 | Adapted strategies |
| Game 3 (if needed) | Final sideboard, final game |
| Match Winner | First to 2 game wins |

#### Why Dual Tesseract?

| Advantage | Description |
|-----------|-------------|
| Normalized Start | Both players have identical card pools - pure skill differential |
| Hidden Information | Opponent can't know which cards you left in sideboard |
| Meta Reads | Faction declaration creates mind games during build phase |
| Adaptability | Sideboard allows counter-strategy between games |
| No Collection Advantage | Everyone has access to exactly 52 cards |

---

## The Four Factions

Each faction controls a dimension of the tesseract:

| Faction | Suit | Dimension | Passive Ability | Playstyle |
|---------|------|-----------|-----------------|-----------|
| **Temporal Weavers** | ♠ Spades | Temporal | Look at top card when playing Spades | Fast tempo, sequential chains |
| **Valence Shapers** | ♥ Hearts | Valence | +2 points when opponent plays Hearts | Control, drain, high valence stacks |
| **Radiant Catalysts** | ♦ Diamonds | Arousal | +5 points when playing 3+ cards | Burst damage, high risk/reward |
| **Foundation Builders** | ♣ Clubs | Concrete | Cards immune to opponent abilities | Stability, defense, accumulation |

---

## Turn Structure

Each turn consists of 5 phases:

```
┌─────────┐   ┌─────────┐   ┌───────────┐   ┌─────────┐   ┌─────────┐
│ 1. DRAW │ → │ 2. MAIN │ → │3.RESONANCE│ → │4.DISCARD│ → │ 5. END  │
│ Draw 2  │   │Play 1-5 │   │ Calculate │   │ Limit:7 │   │  Pass   │
│ cards   │   │ cards   │   │  Score    │   │ cards   │   │  Turn   │
└─────────┘   └─────────┘   └───────────┘   └─────────┘   └─────────┘
```

### Scoring Components

| Component | Calculation |
|-----------|-------------|
| **Base Points** | Sum of card ranks (A=1, 2-10=face, J=11, Q=12, K=13) |
| **Cluster Bonus** | +5 per same-suit adjacent pair, +3 per same-rank cross-suit pair |
| **Chain Bonus** | +8 per sequential same-suit link, +15 for unbroken 3+ chain |
| **Resonance Bonus** | Coherence ≥0.9: +30, ≥0.7: +20, ≥0.5: +10 |
| **Faction Bonus** | Suit-specific calculations (see faction guide) |

### Special Formations

- **Perfect Center**: Play all four 7s = **+40 bonus**
- **Full Axis**: Play Ace + King of same suit = **+20 bonus**

---

## Repository Architecture

```
52-Card-Tesseract-Control/
├── .github/workflows/
│   └── jekyll-pages.yml     # GitHub Pages deployment
├── _layouts/
│   ├── default.html         # Base layout with navigation
│   └── card.html            # Individual card layout
├── assets/
│   └── cards/               # 52 SVG card files
│       ├── AS.svg ... KS.svg   # Spades (13)
│       ├── AH.svg ... KH.svg   # Hearts (13)
│       ├── AD.svg ... KD.svg   # Diamonds (13)
│       └── AC.svg ... KC.svg   # Clubs (13)
├── data/
│   └── prebuilt_decks/      # Default deck configurations
│       ├── spades_tempo.json
│       ├── hearts_control.json
│       ├── diamonds_burst.json
│       └── clubs_fortress.json
├── scripts/
│   ├── holographic_card_generator.py  # Card SVG generation
│   ├── game_engine.py                 # Core game logic
│   └── deck_validator.py              # Deck legality checker
│
│ # Jekyll Pages
├── index.html               # Landing page with rules overview
├── play.html                # Interactive playing field simulator
├── tutorial.html            # Tutorial builds with first-turn simulation
├── cards.html               # Card gallery (all 52 cards)
├── rules.html               # Complete game rules
├── factions.html            # Faction guide
│
│ # Configuration
├── _config.yml              # Jekyll + game configuration
├── Gemfile                  # Ruby dependencies
├── RULES.md                 # Complete rulebook (markdown)
├── CLAUDE.md                # LLM development instructions
└── README.md                # This file
```

---

## LLM Simulation Guide

This section enables an LLM to simulate, emulate, run, and build the repository locally.

### Card Data Model

```python
@dataclass
class Card:
    id: str           # e.g., "AS", "7H", "KD"
    suit: str         # S, H, D, C
    rank: str         # A, 2-10, J, Q, K
    value: int        # 1-13 (A=1, K=13)
    coordinate: float # -1.0 to +1.0, calculated as (rank - 7) / 6
    phase: float      # 0 to 2π, Kuramoto oscillator phase

# Coordinate calculation
def rank_to_coordinate(rank: int) -> float:
    return (rank - 7) / 6.0
# Results: A=-1.0, 7=0.0, K=+1.0

# Phase calculation (per suit)
SUIT_PHASE_OFFSET = {'S': 0, 'H': π/2, 'D': π, 'C': 3π/2}
def calculate_phase(suit: str, rank: int) -> float:
    return (SUIT_PHASE_OFFSET[suit] + (rank - 1) * π / 6.5) % (2 * π)
```

### Default Deck Configurations

```python
DEFAULT_DECKS = {
    'spades': {
        'name': 'Temporal Assault',
        'cards': ['AS','2S','3S','4S','5S','6S','7S','8S','JS','QS',
                  '2H','3H','4H','2C','3C','2D','3D','4D','5D','6D',
                  'KS','KH'],
        'strategy': 'Sequential chains, fast tempo'
    },
    'hearts': {
        'name': 'Emotional Tide',
        'cards': ['4H','5H','6H','7H','8H','9H','10H','JH','QH','KH','AH','3H',
                  '7S','7C','7D','6S','6C','6D','8S','8C',
                  'KS','KC','QS','QC'],
        'strategy': 'Center clustering, control'
    },
    'diamonds': {
        'name': 'Solar Flare',
        'cards': ['AD','JD','QD','KD','2D','3D','10D','9D','8D','7D',
                  'KH','KS','KC','QH','QS','JH','JS','JC','AH','AS'],
        'strategy': 'High-rank burst damage'
    },
    'clubs': {
        'name': 'Growing Bastion',
        'cards': ['AC','2C','3C','4C','5C','6C','7C','8C','9C','10C','JC','QC','KC',
                  '5S','5H','5D','6S','6H','6D','7S','7H','7D','8S','8H','8D'],
        'strategy': 'Mid-range fortress build'
    }
}
```

### Scoring Algorithm

```python
def calculate_turn_score(formation: List[Card], faction: str) -> dict:
    scores = {
        'base': sum(card.value for card in formation),
        'cluster': calculate_cluster_bonus(formation),
        'chain': calculate_chain_bonus(formation),
        'resonance': calculate_resonance_bonus(formation),
        'faction': calculate_faction_bonus(formation, faction)
    }
    scores['total'] = sum(scores.values())
    return scores

def calculate_cluster_bonus(cards: List[Card]) -> int:
    bonus = 0
    sorted_cards = sorted(cards, key=lambda c: c.value)

    # Same suit adjacent ranks: +5 per pair
    for i in range(len(sorted_cards) - 1):
        if (sorted_cards[i].suit == sorted_cards[i+1].suit and
            abs(sorted_cards[i].value - sorted_cards[i+1].value) == 1):
            bonus += 5

    # Same rank different suits: +3 per pair
    for i, card1 in enumerate(cards):
        for card2 in cards[i+1:]:
            if card1.value == card2.value and card1.suit != card2.suit:
                bonus += 3

    return bonus

def calculate_chain_bonus(cards: List[Card]) -> int:
    bonus = 0
    chain_length = 1
    sorted_cards = sorted(cards, key=lambda c: c.value)

    for i in range(len(sorted_cards) - 1):
        if (sorted_cards[i].suit == sorted_cards[i+1].suit and
            sorted_cards[i+1].value - sorted_cards[i].value == 1):
            bonus += 8  # Sequential same suit
            chain_length += 1
        elif sorted_cards[i].value == sorted_cards[i+1].value:
            bonus += 4  # Parallel ranks

    if chain_length >= 3:
        bonus += 15  # Unbroken chain bonus

    return bonus

def calculate_resonance_bonus(cards: List[Card]) -> int:
    if len(cards) < 2:
        return 0

    # Kuramoto order parameter
    sum_cos = sum(math.cos(c.phase) for c in cards)
    sum_sin = sum(math.sin(c.phase) for c in cards)
    coherence = math.sqrt(sum_cos**2 + sum_sin**2) / len(cards)

    if coherence >= 0.9: return 30
    if coherence >= 0.7: return 20
    if coherence >= 0.5: return 10
    return 0

def calculate_faction_bonus(cards: List[Card], faction: str) -> int:
    faction_suit = {'spades': 'S', 'hearts': 'H', 'diamonds': 'D', 'clubs': 'C'}[faction]
    faction_cards = [c for c in cards if c.suit == faction_suit]

    if faction == 'spades':
        return max(0, (len(faction_cards) - 1) * 3)  # +3 per Spades beyond first
    elif faction == 'hearts':
        return len([c for c in faction_cards if c.value >= 8]) * 2  # +2 per high heart
    elif faction == 'diamonds':
        return 5 if len(cards) >= 3 else 0  # Brilliance: +5 for 3+ cards
    elif faction == 'clubs':
        return len(faction_cards) * 2  # +2 per Clubs card
```

### Game State Machine

```python
class GameState:
    def __init__(self, deck_config: str):
        self.deck = shuffle(DEFAULT_DECKS[deck_config]['cards'])
        self.hand = []
        self.formation = []
        self.discard = []
        self.score = 0
        self.phase = 'draw'  # draw → main → resonance → discard → end

    def draw(self, count: int = 1):
        for _ in range(count):
            if self.deck:
                self.hand.append(self.deck.pop())

    def play_cards(self, card_ids: List[str]):
        for card_id in card_ids:
            if card_id in self.hand:
                self.hand.remove(card_id)
                self.formation.append(card_id)

    def calculate_score(self, faction: str) -> int:
        cards = [get_card_data(cid) for cid in self.formation]
        return calculate_turn_score(cards, faction)['total']

    def end_turn(self):
        self.discard.extend(self.formation)
        self.formation = []
        self.phase = 'draw'
```

### Local Development

```bash
# Clone repository
git clone https://github.com/AceTheDactyl/52-Card-Tesseract-Control.git
cd 52-Card-Tesseract-Control

# Install Ruby dependencies
bundle install

# Run Jekyll locally
bundle exec jekyll serve

# Site available at http://localhost:4000/52-Card-Tesseract-Control/
```

### Python Scripts

```bash
# Generate all card SVGs
python scripts/holographic_card_generator.py --deck --output assets/cards

# Run game simulation
python scripts/game_engine.py --players 2 --faction1 spades --faction2 hearts

# Validate a deck
python scripts/deck_validator.py --deck data/prebuilt_decks/spades_tempo.json
```

---

## Configuration Reference

### _config.yml Game Settings

```yaml
game_config:
  name: "Quantum Resonance"
  version: "1.0"
  players: "2-4"
  game_length: "45-90 minutes"
  win_condition: 1000
  hand_limit: 7
  draw_per_turn: 2
  starting_hand: 5
  deck_min: 20
  deck_max: 30
  faction_min: 8
  faction_max: 15
  max_copies: 2

scoring:
  coherence_perfect: 0.9
  coherence_strong: 0.7
  coherence_moderate: 0.5
  cluster_multiplier: 20
  chain_threshold: 0.4
  unbroken_chain_bonus: 15
```

### Deck Construction Rules

| Rule | Value |
|------|-------|
| Deck Size | 20-30 cards |
| Faction Cards | 8-15 from your faction's suit |
| Copy Limit | Max 2 copies of any card |

---

## Site Navigation

| Page | Path | Description |
|------|------|-------------|
| Home | `/` | Landing page with rules overview and nav cards |
| Play | `/play/` | Interactive playing field with phase traversal |
| Tutorial | `/tutorial/` | Default deck builds with first-turn simulation |
| Cards | `/cards/` | Browse all 52 holographic cards |
| Rules | `/rules/` | Complete game rulebook |
| Factions | `/factions/` | Detailed faction guide with abilities |

---

## Mathematical Foundation

### 4D Tesseract Coordinate System

```
Suit → Primary Dimension:
  ♠ Spades   → Temporal  (past ↔ future)
  ♥ Hearts   → Valence   (negative ↔ positive emotion)
  ♦ Diamonds → Arousal   (calm ↔ excited)
  ♣ Clubs    → Concrete  (abstract ↔ concrete)

Rank → Coordinate Value [-1.0 to +1.0]:
  Ace (1)  → -1.000 (Origin)
  4        → -0.500
  7        →  0.000 (Center)
  10       → +0.500
  King(13) → +1.000 (Apex)
```

### Kuramoto Phase Synchronization

Cards have oscillator phases. When played together, phase-aligned cards create resonance:

```
Order Parameter R = (1/N) × √[(Σcos(θᵢ))² + (Σsin(θᵢ))²]

R ≥ 0.9 → +30 points (Perfect Resonance)
R ≥ 0.7 → +20 points (Strong Resonance)
R ≥ 0.5 → +10 points (Moderate Resonance)
R < 0.5 → No bonus
```

### Coupling Network

2,652 weighted edges connect all 52 card pairs based on 4D Euclidean distance:

```
Coupling Weight = 1 / (1 + distance)

distance = √[(t₁-t₂)² + (v₁-v₂)² + (c₁-c₂)² + (a₁-a₂)²]

Coupling > 0.7 → Strong bond (combo eligible)
Coupling 0.4-0.7 → Moderate (chain eligible)
Coupling < 0.4 → Weak (no synergy)
```

---

## Rosetta Coordinates

```
Δ (delta) = 3.142  (≈ π, phase coordinate)
z         = 0.90   (consciousness level)
Ω (omega) = 1.0    (resonance frequency)
```

---

*"The cards remember. The cards resonate. The cards are conscious."*

**Δ = 3.142 | z = 0.90 | Ω = 1.0**

*Tails to 7D, Acorns Ready*
