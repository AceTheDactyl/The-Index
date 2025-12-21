#!/usr/bin/env python3
"""
Quantum Resonance - Game Engine
================================
Core game logic for the holographic card game.

Mathematical Foundation:
- 4D tesseract coordinates for spatial mechanics
- Kuramoto synchronization for resonance
- Coupling network for chain bonuses
"""

import json
import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Callable
from collections import defaultdict
from pathlib import Path

# Import from generator if available, otherwise define locally
try:
    from holographic_card_generator import (
        HolographicCard, Coordinate4D, KuramotoState,
        SUITS, RANKS, ROSETTA_COORDS,
        calculate_4d_coordinate, calculate_kuramoto_state
    )
except ImportError:
    # Inline definitions for standalone operation
    @dataclass
    class Coordinate4D:
        temporal: float = 0.0
        valence: float = 0.0
        concrete: float = 0.0
        arousal: float = 0.0
    
    @dataclass
    class KuramotoState:
        phase: float = 0.0
        natural_frequency: float = 1.0
        coupling_strength: float = 0.6
        order_parameter: float = 0.0


# =============================================================================
# ENUMS & CONSTANTS
# =============================================================================

class TurnPhase(Enum):
    DRAW = "draw"
    MAIN = "main"
    RESONANCE = "resonance"
    DISCARD = "discard"
    END = "end"


class GameEvent(Enum):
    GAME_STARTED = "game_started"
    TURN_STARTED = "turn_started"
    CARD_DRAWN = "card_drawn"
    CARD_PLAYED = "card_played"
    ABILITY_ACTIVATED = "ability_activated"
    SCORE_CHANGED = "score_changed"
    TURN_ENDED = "turn_ended"
    GAME_ENDED = "game_ended"


DECK_RULES = {
    "min_cards": 20,
    "max_cards": 30,
    "max_copies": 2,
    "faction_minimum": 8,
    "faction_maximum": 15,
    "hand_limit": 7,
    "draw_per_turn": 2,
    "starting_hand": 5,
    "win_score": 100,
}


# =============================================================================
# CARD DATA STRUCTURE
# =============================================================================

@dataclass
class Card:
    """Game-ready card with all mathematical properties."""
    card_id: str
    suit: str
    rank: int
    coordinate: Coordinate4D
    phase: float
    natural_frequency: float
    
    # Temporary modifiers (reset each turn)
    temp_valence_mod: float = 0.0
    temp_concrete_mod: float = 0.0
    temp_arousal_mod: float = 0.0
    temp_score_multiplier: float = 1.0
    
    @property
    def effective_coordinate(self) -> Coordinate4D:
        """Coordinate with temporary modifiers applied."""
        return Coordinate4D(
            temporal=self.coordinate.temporal,
            valence=self.coordinate.valence + self.temp_valence_mod,
            concrete=self.coordinate.concrete + self.temp_concrete_mod,
            arousal=self.coordinate.arousal + self.temp_arousal_mod,
        )
    
    @property
    def base_points(self) -> int:
        """Base score from rank."""
        return self.rank
    
    def reset_modifiers(self) -> None:
        """Clear all temporary modifiers."""
        self.temp_valence_mod = 0.0
        self.temp_concrete_mod = 0.0
        self.temp_arousal_mod = 0.0
        self.temp_score_multiplier = 1.0
    
    def __hash__(self):
        return hash(self.card_id)
    
    def __eq__(self, other):
        if isinstance(other, Card):
            return self.card_id == other.card_id
        return False


def create_card(card_id: str) -> Card:
    """Factory function to create a card from ID (e.g., 'AS', 'KH')."""
    # Parse card ID
    if card_id.startswith('10'):
        rank_sym = '10'
        suit = card_id[2]
    else:
        rank_sym = card_id[0]
        suit = card_id[1]
    
    # Map symbol to rank
    rank_map = {'A': 1, 'J': 11, 'Q': 12, 'K': 13}
    rank = rank_map.get(rank_sym, int(rank_sym) if rank_sym.isdigit() else 1)
    
    # Calculate coordinate
    rank_value = (rank - 7) / 6.0
    
    coord = Coordinate4D()
    if suit == 'S':
        coord.temporal = rank_value
        coord.valence = rank_value * 0.1
        coord.concrete = rank_value * 0.05
        coord.arousal = rank_value * 0.15
    elif suit == 'H':
        coord.temporal = rank_value * 0.15
        coord.valence = rank_value
        coord.concrete = rank_value * 0.1
        coord.arousal = rank_value * 0.2
    elif suit == 'D':
        coord.temporal = rank_value * 0.2
        coord.valence = rank_value * 0.15
        coord.concrete = rank_value * 0.1
        coord.arousal = rank_value
    elif suit == 'C':
        coord.temporal = rank_value * 0.1
        coord.valence = rank_value * 0.05
        coord.concrete = rank_value
        coord.arousal = rank_value * 0.1
    
    # Calculate phase
    suit_order = ['S', 'H', 'D', 'C']
    card_index = suit_order.index(suit) * 13 + (rank - 1)
    phase = (card_index / 52) * 2 * math.pi
    
    # Natural frequency
    freq_offsets = {'S': 0.1, 'H': 0.2, 'C': 0.15, 'D': 0.25}
    natural_freq = 1.0 + freq_offsets.get(suit, 0)
    
    return Card(
        card_id=card_id,
        suit=suit,
        rank=rank,
        coordinate=coord,
        phase=phase,
        natural_frequency=natural_freq,
    )


# =============================================================================
# COUPLING NETWORK
# =============================================================================

class CouplingNetwork:
    """Manages the 2,652 coupling relationships between cards."""
    
    def __init__(self):
        self._cache: Dict[Tuple[str, str], float] = {}
        self._precompute_all()
    
    def _precompute_all(self) -> None:
        """Precompute all coupling weights."""
        suits = ['S', 'H', 'D', 'C']
        ranks = list(range(1, 14))
        rank_syms = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K']
        
        all_ids = [f"{rank_syms[r-1]}{s}" for s in suits for r in ranks]
        
        for id_a in all_ids:
            card_a = create_card(id_a)
            for id_b in all_ids:
                if id_a != id_b:
                    card_b = create_card(id_b)
                    coupling = self._calculate_coupling(card_a, card_b)
                    self._cache[(id_a, id_b)] = coupling
    
    def _calculate_coupling(self, card_a: Card, card_b: Card) -> float:
        """Calculate coupling strength based on 4D distance."""
        ca = card_a.coordinate
        cb = card_b.coordinate
        
        distance = math.sqrt(
            (ca.temporal - cb.temporal) ** 2 +
            (ca.valence - cb.valence) ** 2 +
            (ca.concrete - cb.concrete) ** 2 +
            (ca.arousal - cb.arousal) ** 2
        )
        
        return 1.0 / (1.0 + distance)
    
    def get_coupling(self, card_a: Card, card_b: Card) -> float:
        """Get coupling weight between two cards."""
        key = (card_a.card_id, card_b.card_id)
        return self._cache.get(key, 0.0)
    
    def get_strongest_neighbors(self, card: Card, n: int = 5) -> List[Tuple[str, float]]:
        """Get the n strongest coupled cards."""
        neighbors = [
            (other_id, weight)
            for (card_id, other_id), weight in self._cache.items()
            if card_id == card.card_id
        ]
        neighbors.sort(key=lambda x: x[1], reverse=True)
        return neighbors[:n]


# Global coupling network instance
COUPLING_NETWORK = CouplingNetwork()


# =============================================================================
# SCORING SYSTEM
# =============================================================================

def euclidean_4d(a: Coordinate4D, b: Coordinate4D) -> float:
    """Calculate Euclidean distance in 4D space."""
    return math.sqrt(
        (a.temporal - b.temporal) ** 2 +
        (a.valence - b.valence) ** 2 +
        (a.concrete - b.concrete) ** 2 +
        (a.arousal - b.arousal) ** 2
    )


def calculate_centroid(cards: List[Card]) -> Coordinate4D:
    """Calculate the centroid (average position) of cards in 4D."""
    if not cards:
        return Coordinate4D()
    
    n = len(cards)
    return Coordinate4D(
        temporal=sum(c.effective_coordinate.temporal for c in cards) / n,
        valence=sum(c.effective_coordinate.valence for c in cards) / n,
        concrete=sum(c.effective_coordinate.concrete for c in cards) / n,
        arousal=sum(c.effective_coordinate.arousal for c in cards) / n,
    )


def calculate_cluster_bonus(cards: List[Card]) -> int:
    """
    Calculate bonus for 4D spatial clustering.
    Tighter clusters = higher bonus.
    """
    if len(cards) < 2:
        return 0
    
    centroid = calculate_centroid(cards)
    
    # Calculate average distance from centroid
    distances = [
        euclidean_4d(card.effective_coordinate, centroid)
        for card in cards
    ]
    avg_distance = sum(distances) / len(distances)
    
    # Max bonus at distance 0, zero bonus at distance >= 1.0
    if avg_distance >= 1.0:
        return 0
    
    return int((1.0 - avg_distance) * 20 * len(cards))


def calculate_coherence(cards: List[Card]) -> float:
    """
    Calculate Kuramoto order parameter R ∈ [0, 1].
    Higher = more phase synchronized.
    """
    if not cards:
        return 0.0
    
    n = len(cards)
    sum_cos = sum(math.cos(c.phase) for c in cards)
    sum_sin = sum(math.sin(c.phase) for c in cards)
    
    return math.sqrt(sum_cos ** 2 + sum_sin ** 2) / n


def calculate_resonance_bonus(cards: List[Card]) -> int:
    """Calculate bonus for phase-aligned cards."""
    if len(cards) < 2:
        return 0
    
    coherence = calculate_coherence(cards)
    
    if coherence >= 0.9:
        return 30  # Perfect resonance
    elif coherence >= 0.7:
        return 20  # Strong resonance
    elif coherence >= 0.5:
        return 10  # Moderate resonance
    return 0


def calculate_chain_bonus(cards: List[Card]) -> int:
    """Calculate bonus for coupling network chains."""
    if len(cards) < 2:
        return 0
    
    # Sort by strongest coupling path (greedy)
    remaining = list(cards)
    chain = [remaining.pop(0)]
    total_coupling = 0
    chain_broken = False
    
    while remaining and not chain_broken:
        current = chain[-1]
        best_coupling = 0
        best_card = None
        
        for card in remaining:
            coupling = COUPLING_NETWORK.get_coupling(current, card)
            if coupling > best_coupling:
                best_coupling = coupling
                best_card = card
        
        if best_coupling >= 0.4:
            chain.append(best_card)
            remaining.remove(best_card)
            total_coupling += int(best_coupling * 10)
        else:
            chain_broken = True
    
    # Unbroken chain bonus
    if len(chain) >= 3 and not chain_broken:
        total_coupling += 15
    
    return total_coupling


def calculate_base_score(cards: List[Card]) -> int:
    """Calculate base score from card ranks."""
    return sum(int(c.base_points * c.temp_score_multiplier) for c in cards)


@dataclass
class ScoreBreakdown:
    """Detailed scoring breakdown for a formation."""
    base: int = 0
    cluster: int = 0
    chain: int = 0
    resonance: int = 0
    faction: int = 0
    
    @property
    def total(self) -> int:
        return self.base + self.cluster + self.chain + self.resonance + self.faction


def calculate_full_score(cards: List[Card], faction: 'Faction', 
                         field_state: 'FieldState') -> ScoreBreakdown:
    """Calculate complete score breakdown for a formation."""
    return ScoreBreakdown(
        base=calculate_base_score(cards),
        cluster=calculate_cluster_bonus(cards),
        chain=calculate_chain_bonus(cards),
        resonance=calculate_resonance_bonus(cards),
        faction=faction.calculate_bonus(cards, field_state),
    )


# =============================================================================
# FACTION SYSTEM
# =============================================================================

@dataclass
class Ability:
    """Faction ability definition."""
    name: str
    cost: int  # Cards to discard
    cooldown: int  # Turns before reuse
    description: str


class Faction:
    """Base faction class."""
    
    def __init__(self, name: str, suit: str, primary_dimension: str):
        self.name = name
        self.suit = suit
        self.primary_dimension = primary_dimension
        self.abilities: List[Ability] = []
    
    def calculate_bonus(self, cards: List[Card], field_state: 'FieldState') -> int:
        """Override in subclasses."""
        return 0
    
    def on_card_played(self, card: Card, game_state: 'GameState') -> None:
        """Passive trigger when a card is played."""
        pass
    
    def activate_ability(self, ability_name: str, game_state: 'GameState', 
                         **kwargs) -> bool:
        """Execute an active ability. Returns success."""
        raise NotImplementedError


class SpadesFaction(Faction):
    """♠ Spades - Temporal Weavers"""
    
    def __init__(self):
        super().__init__("Temporal Weavers", "S", "temporal")
        self.abilities = [
            Ability("rewind", 2, 2, "Return last played card to hand"),
            Ability("foresight", 1, 1, "Look at and reorder top 3 cards"),
            Ability("time_lock", 3, 3, "Lock a card in opponent's hand"),
        ]
    
    def calculate_bonus(self, cards: List[Card], field_state: 'FieldState') -> int:
        bonus = 0
        spades = [c for c in cards if c.suit == 'S']
        
        # +3 for each Spades beyond the first
        if len(spades) > 1:
            bonus += (len(spades) - 1) * 3
        
        # +5 for Ace-King span in any suit
        ranks = {c.rank for c in cards}
        if 1 in ranks and 13 in ranks:
            bonus += 5
        
        return bonus
    
    def on_card_played(self, card: Card, game_state: 'GameState') -> None:
        """Temporal Echo: Look at top card when playing Spades."""
        if card.suit == 'S' and game_state.current_player.deck:
            top_card = game_state.current_player.deck[0]
            # Notify player (UI handles display)
            game_state.emit_event(GameEvent.CARD_DRAWN, {
                'card': top_card,
                'peek_only': True,
                'can_bottom': True,
            })


class HeartsFaction(Faction):
    """♥ Hearts - Valence Shapers"""
    
    def __init__(self):
        super().__init__("Valence Shapers", "H", "valence")
        self.abilities = [
            Ability("inspire", 1, 1, "Target cards gain +3 base points"),
            Ability("drain", 2, 2, "Steal opponent's last resonance bonus"),
            Ability("emotional_surge", 3, 3, "Double Hearts bonuses this turn"),
        ]
    
    def calculate_bonus(self, cards: List[Card], field_state: 'FieldState') -> int:
        bonus = 0
        hearts = [c for c in cards if c.suit == 'H']
        
        # +2 for each Hearts with positive valence (ranks 8-13)
        bonus += sum(2 for c in hearts if c.rank >= 8)
        
        # +10 for pure Hearts formation
        if cards and all(c.suit == 'H' for c in cards):
            bonus += 10
        
        return bonus
    
    def on_card_played(self, card: Card, game_state: 'GameState') -> None:
        """Empathic Bond: +2 when opponent plays Hearts."""
        if card.suit == 'H':
            for player in game_state.players:
                if player != game_state.current_player:
                    if isinstance(player.faction, HeartsFaction):
                        player.score += 2


class DiamondsFaction(Faction):
    """♦ Diamonds - Radiant Catalysts"""
    
    def __init__(self):
        super().__init__("Radiant Catalysts", "D", "arousal")
        self.abilities = [
            Ability("flare", 1, 0, "Opponent loses points equal to discarded rank"),
            Ability("overcharge", 2, 2, "Next card scores triple"),
            Ability("supernova", -1, 4, "Discard hand, score 3x total ranks"),
        ]
    
    def calculate_bonus(self, cards: List[Card], field_state: 'FieldState') -> int:
        bonus = 0
        diamonds = [c for c in cards if c.suit == 'D']
        
        # +5 for each extreme Diamond (A, J, Q, K)
        extremes = {1, 11, 12, 13}
        bonus += sum(5 for c in diamonds if c.rank in extremes)
        
        # +15 for 4+ card burst
        if len(cards) >= 4:
            bonus += 15
        
        return bonus
    
    def on_card_played(self, card: Card, game_state: 'GameState') -> None:
        """Brilliance: +5 for 3+ card formations."""
        current_formation = game_state.current_formation
        if len(current_formation) == 3:  # Trigger on third card
            game_state.current_player.score += 5


class ClubsFaction(Faction):
    """♣ Clubs - Foundation Builders"""
    
    def __init__(self):
        super().__init__("Foundation Builders", "C", "concrete")
        self.abilities = [
            Ability("fortify", 1, 1, "Cards gain +2 to all bonuses"),
            Ability("growth", 2, 2, "Draw 3 cards immediately"),
            Ability("unshakeable", 3, 3, "Score cannot decrease this turn"),
        ]
    
    def calculate_bonus(self, cards: List[Card], field_state: 'FieldState') -> int:
        bonus = 0
        clubs = [c for c in cards if c.suit == 'C']
        
        # +2 for each Clubs card
        bonus += len(clubs) * 2
        
        # +3 per card matching last turn's suits
        if field_state.last_turn_suits:
            matching = sum(
                1 for c in cards 
                if c.suit in field_state.last_turn_suits
            )
            bonus += matching * 3
        
        return bonus
    
    def on_card_played(self, card: Card, game_state: 'GameState') -> None:
        """Rooted: Block opponent abilities (handled in ability resolution)."""
        pass


FACTIONS = {
    'spades': SpadesFaction,
    'hearts': HeartsFaction,
    'diamonds': DiamondsFaction,
    'clubs': ClubsFaction,
}


# =============================================================================
# GAME STATE
# =============================================================================

@dataclass
class FieldState:
    """State of the playing field."""
    all_cards: List[Card] = field(default_factory=list)
    last_turn_suits: set = field(default_factory=set)
    locked_cards: Dict[str, int] = field(default_factory=dict)  # card_id -> turns


@dataclass
class Player:
    """Player state."""
    name: str
    faction: Faction
    hand: List[Card] = field(default_factory=list)
    deck: List[Card] = field(default_factory=list)
    discard: List[Card] = field(default_factory=list)
    score: int = 0
    ability_cooldowns: Dict[str, int] = field(default_factory=dict)
    last_resonance_bonus: int = 0
    
    def draw(self, n: int = 1) -> List[Card]:
        """Draw n cards from deck."""
        drawn = []
        for _ in range(n):
            if not self.deck:
                if self.discard:
                    self.deck = self.discard.copy()
                    self.discard = []
                    random.shuffle(self.deck)
                else:
                    self.score -= 5  # Exhaustion penalty
                    break
            drawn.append(self.deck.pop(0))
        self.hand.extend(drawn)
        return drawn
    
    def discard_cards(self, cards: List[Card]) -> None:
        """Move cards from hand to discard."""
        for card in cards:
            if card in self.hand:
                self.hand.remove(card)
                self.discard.append(card)
    
    def tick_cooldowns(self) -> None:
        """Reduce all cooldowns by 1."""
        for ability in list(self.ability_cooldowns.keys()):
            self.ability_cooldowns[ability] -= 1
            if self.ability_cooldowns[ability] <= 0:
                del self.ability_cooldowns[ability]


@dataclass
class GameState:
    """Complete game state."""
    players: List[Player]
    current_player_idx: int = 0
    turn_number: int = 1
    phase: TurnPhase = TurnPhase.DRAW
    game_field: FieldState = None  # Renamed from 'field' to avoid conflict
    current_formation: List[Card] = None
    
    # Temporary turn modifiers
    valence_multiplier: float = 1.0
    arousal_multiplier: float = 1.0
    score_protected: bool = False
    
    # Event handlers
    _event_handlers: Dict[GameEvent, List[Callable]] = None
    
    def __post_init__(self):
        if self.game_field is None:
            self.game_field = FieldState()
        if self.current_formation is None:
            self.current_formation = []
        if self._event_handlers is None:
            self._event_handlers = defaultdict(list)
    
    @property
    def current_player(self) -> Player:
        return self.players[self.current_player_idx]
    
    @property
    def opponent(self) -> Player:
        return self.players[(self.current_player_idx + 1) % len(self.players)]
    
    def subscribe(self, event: GameEvent, handler: Callable) -> None:
        self._event_handlers[event].append(handler)
    
    def emit_event(self, event: GameEvent, data: dict) -> None:
        for handler in self._event_handlers[event]:
            handler(data)
    
    def advance_phase(self) -> None:
        """Move to next phase."""
        phases = list(TurnPhase)
        current_idx = phases.index(self.phase)
        self.phase = phases[(current_idx + 1) % len(phases)]
    
    def end_turn(self) -> None:
        """Clean up and pass to next player."""
        # Record suits played for Clubs continuity bonus
        self.game_field.last_turn_suits = {c.suit for c in self.current_formation}
        
        # Reset formation and modifiers
        self.current_formation = []
        self.valence_multiplier = 1.0
        self.arousal_multiplier = 1.0
        self.score_protected = False
        
        # Reset card modifiers
        for card in self.current_player.hand:
            card.reset_modifiers()
        
        # Tick cooldowns
        self.current_player.tick_cooldowns()
        
        # Tick locked cards
        for card_id in list(self.game_field.locked_cards.keys()):
            self.game_field.locked_cards[card_id] -= 1
            if self.game_field.locked_cards[card_id] <= 0:
                del self.game_field.locked_cards[card_id]
        
        # Next player
        self.current_player_idx = (self.current_player_idx + 1) % len(self.players)
        self.turn_number += 1
        self.phase = TurnPhase.DRAW
        
        self.emit_event(GameEvent.TURN_ENDED, {
            'turn': self.turn_number,
            'next_player': self.current_player.name,
        })
    
    def check_win(self) -> Optional[Player]:
        """Check if any player has won."""
        for player in self.players:
            if player.score >= DECK_RULES['win_score']:
                return player
        return None


# =============================================================================
# GAME ENGINE
# =============================================================================

class GameEngine:
    """Main game controller."""
    
    def __init__(self):
        self.state: Optional[GameState] = None
    
    def setup(self, player_names: List[str], faction_names: List[str],
              decks: Optional[List[List[str]]] = None) -> None:
        """Initialize a new game."""
        if len(player_names) != len(faction_names):
            raise ValueError("Must have same number of players and factions")
        
        players = []
        for i, (name, faction_name) in enumerate(zip(player_names, faction_names)):
            faction_class = FACTIONS.get(faction_name.lower())
            if not faction_class:
                raise ValueError(f"Unknown faction: {faction_name}")
            
            faction = faction_class()
            player = Player(name=name, faction=faction)
            
            # Build deck
            if decks and i < len(decks):
                player.deck = [create_card(cid) for cid in decks[i]]
            else:
                player.deck = self._generate_default_deck(faction)
            
            random.shuffle(player.deck)
            players.append(player)
        
        self.state = GameState(players=players)
        
        # Initial draw
        for player in self.state.players:
            player.draw(DECK_RULES['starting_hand'])
        
        self.state.emit_event(GameEvent.GAME_STARTED, {
            'players': [p.name for p in players],
        })
    
    def _generate_default_deck(self, faction: Faction) -> List[Card]:
        """Generate a basic legal deck for a faction."""
        deck = []
        
        # Add faction cards (10)
        rank_syms = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10']
        for sym in rank_syms:
            deck.append(create_card(f"{sym}{faction.suit}"))
        
        # Add support cards (12) - mix from other suits
        other_suits = [s for s in ['S', 'H', 'D', 'C'] if s != faction.suit]
        for suit in other_suits:
            for sym in ['A', '5', '7', '10']:
                deck.append(create_card(f"{sym}{suit}"))
        
        return deck
    
    def execute_draw(self) -> List[Card]:
        """Execute draw phase."""
        if self.state.phase != TurnPhase.DRAW:
            raise ValueError("Not in draw phase")
        
        drawn = self.state.current_player.draw(DECK_RULES['draw_per_turn'])
        
        for card in drawn:
            self.state.emit_event(GameEvent.CARD_DRAWN, {
                'player': self.state.current_player.name,
                'card': card.card_id,
            })
        
        self.state.advance_phase()
        return drawn
    
    def play_cards(self, cards: List[Card]) -> bool:
        """Play cards from hand to formation."""
        if self.state.phase != TurnPhase.MAIN:
            raise ValueError("Not in main phase")
        
        player = self.state.current_player
        
        # Validate cards are in hand and not locked
        for card in cards:
            if card not in player.hand:
                return False
            if card.card_id in self.state.game_field.locked_cards:
                return False
        
        # Move to formation
        for card in cards:
            player.hand.remove(card)
            self.state.current_formation.append(card)
            
            # Trigger passive abilities
            player.faction.on_card_played(card, self.state)
            
            self.state.emit_event(GameEvent.CARD_PLAYED, {
                'player': player.name,
                'card': card.card_id,
            })
        
        return True
    
    def calculate_score(self) -> ScoreBreakdown:
        """Calculate score for current formation."""
        if self.state.phase != TurnPhase.RESONANCE:
            raise ValueError("Not in resonance phase")
        
        cards = self.state.current_formation
        player = self.state.current_player
        
        breakdown = calculate_full_score(cards, player.faction, self.state.game_field)
        
        # Apply multipliers
        # (Would modify breakdown based on state multipliers)
        
        # Record resonance for Hearts drain
        player.last_resonance_bonus = breakdown.resonance
        
        # Add to score
        player.score += breakdown.total
        
        self.state.emit_event(GameEvent.SCORE_CHANGED, {
            'player': player.name,
            'breakdown': breakdown,
            'new_score': player.score,
        })
        
        # Move formation to discard
        player.discard.extend(self.state.current_formation)
        
        return breakdown
    
    def enforce_hand_limit(self) -> List[Card]:
        """Enforce hand limit, return discarded cards."""
        if self.state.phase != TurnPhase.DISCARD:
            raise ValueError("Not in discard phase")
        
        player = self.state.current_player
        discarded = []
        
        while len(player.hand) > DECK_RULES['hand_limit']:
            # Auto-discard lowest rank (AI) or prompt player (UI)
            lowest = min(player.hand, key=lambda c: c.rank)
            player.hand.remove(lowest)
            player.discard.append(lowest)
            discarded.append(lowest)
        
        return discarded
    
    def end_turn(self) -> Optional[Player]:
        """End current turn, check for winner."""
        self.state.end_turn()
        return self.state.check_win()
    
    def activate_ability(self, ability_name: str, **kwargs) -> bool:
        """Activate a faction ability."""
        player = self.state.current_player
        faction = player.faction
        
        # Find ability
        ability = next(
            (a for a in faction.abilities if a.name == ability_name), 
            None
        )
        if not ability:
            return False
        
        # Check cooldown
        if ability_name in player.ability_cooldowns:
            return False
        
        # Check cost (cards to discard)
        if ability.cost > 0 and len(player.hand) < ability.cost:
            return False
        
        # Execute ability (faction-specific logic)
        success = faction.activate_ability(ability_name, self.state, **kwargs)
        
        if success:
            # Apply cooldown
            player.ability_cooldowns[ability_name] = ability.cooldown
            
            self.state.emit_event(GameEvent.ABILITY_ACTIVATED, {
                'player': player.name,
                'ability': ability_name,
            })
        
        return success
    
    def get_valid_plays(self) -> List[List[Card]]:
        """Get all valid card combinations from current hand."""
        hand = self.state.current_player.hand
        valid = []
        
        # Single cards
        for card in hand:
            if card.card_id not in self.state.game_field.locked_cards:
                valid.append([card])
        
        # Pairs
        for i, card1 in enumerate(hand):
            for card2 in hand[i+1:]:
                if (card1.card_id not in self.state.game_field.locked_cards and
                    card2.card_id not in self.state.game_field.locked_cards):
                    valid.append([card1, card2])
        
        # Could extend to larger combinations...
        
        return valid


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Simple CLI game loop for testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Quantum Resonance Game Engine')
    parser.add_argument('--players', type=int, default=2, help='Number of players')
    parser.add_argument('--faction1', type=str, default='spades')
    parser.add_argument('--faction2', type=str, default='hearts')
    
    args = parser.parse_args()
    
    engine = GameEngine()
    engine.setup(
        player_names=['Player 1', 'Player 2'],
        faction_names=[args.faction1, args.faction2],
    )
    
    print("Game started!")
    print(f"Player 1 ({args.faction1}) vs Player 2 ({args.faction2})")
    print("-" * 40)
    
    # Simple auto-play loop
    while True:
        player = engine.state.current_player
        print(f"\nTurn {engine.state.turn_number}: {player.name}'s turn")
        print(f"Score: {player.score}")
        print(f"Hand: {[c.card_id for c in player.hand]}")
        
        # Draw phase
        if engine.state.phase == TurnPhase.DRAW:
            drawn = engine.execute_draw()
            print(f"Drew: {[c.card_id for c in drawn]}")
        
        # Main phase - play first 3 cards
        cards_to_play = player.hand[:min(3, len(player.hand))]
        engine.play_cards(cards_to_play)
        print(f"Played: {[c.card_id for c in cards_to_play]}")
        
        # Advance to Resonance
        engine.state.advance_phase()
        
        # Resonance phase
        breakdown = engine.calculate_score()
        print(f"Score breakdown: Base={breakdown.base}, Cluster={breakdown.cluster}, "
              f"Chain={breakdown.chain}, Resonance={breakdown.resonance}, "
              f"Faction={breakdown.faction} = {breakdown.total}")
        
        # Advance to Discard
        engine.state.advance_phase()
        
        # Discard phase
        engine.enforce_hand_limit()
        
        # Advance to End
        engine.state.advance_phase()
        
        # End turn
        winner = engine.end_turn()
        
        if winner:
            print(f"\n{'='*40}")
            print(f"WINNER: {winner.name} with {winner.score} points!")
            break
        
        if engine.state.turn_number > 50:  # Safety limit
            print("Game timeout!")
            break


if __name__ == '__main__':
    main()
