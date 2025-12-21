#!/usr/bin/env python3
"""
Deck Validator - Ensures deck legality for Quantum Resonance
=============================================================
Validates deck composition against game rules.
"""

import json
import sys
from collections import Counter
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
from pathlib import Path


DECK_RULES = {
    "min_cards": 20,
    "max_cards": 30,
    "max_copies": 2,
    "faction_minimum": 8,
    "faction_maximum": 15,
}

VALID_SUITS = {'S', 'H', 'D', 'C'}
VALID_RANKS = {'A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K'}

FACTION_SUITS = {
    'spades': 'S',
    'hearts': 'H',
    'diamonds': 'D',
    'clubs': 'C',
}


@dataclass
class ValidationResult:
    """Result of deck validation."""
    valid: bool
    violations: List[str]
    warnings: List[str]
    stats: Dict
    
    def __str__(self) -> str:
        lines = []
        if self.valid:
            lines.append("✓ Deck is VALID")
        else:
            lines.append("✗ Deck is INVALID")
        
        if self.violations:
            lines.append("\nViolations:")
            for v in self.violations:
                lines.append(f"  ✗ {v}")
        
        if self.warnings:
            lines.append("\nWarnings:")
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")
        
        lines.append("\nDeck Statistics:")
        for key, value in self.stats.items():
            lines.append(f"  {key}: {value}")
        
        return "\n".join(lines)


def parse_card_id(card_id: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse card ID into (rank, suit). Returns (None, None) if invalid."""
    card_id = card_id.upper().strip()
    
    if len(card_id) < 2:
        return None, None
    
    # Handle 10 specially
    if card_id.startswith('10'):
        rank = '10'
        suit = card_id[2:3]
    else:
        rank = card_id[0]
        suit = card_id[1:2]
    
    if rank not in VALID_RANKS:
        return None, None
    if suit not in VALID_SUITS:
        return None, None
    
    return rank, suit


def validate_deck(card_ids: List[str], faction: str) -> ValidationResult:
    """
    Validate a deck against game rules.
    
    Args:
        card_ids: List of card IDs (e.g., ['AS', '2S', 'KH', ...])
        faction: Faction name (e.g., 'spades', 'hearts', 'diamonds', 'clubs')
    
    Returns:
        ValidationResult with validity status, violations, warnings, and stats
    """
    violations = []
    warnings = []
    stats = {}
    
    # Normalize faction
    faction = faction.lower()
    if faction not in FACTION_SUITS:
        violations.append(f"Unknown faction: {faction}")
        return ValidationResult(False, violations, warnings, stats)
    
    faction_suit = FACTION_SUITS[faction]
    
    # Parse all cards
    parsed_cards = []
    invalid_ids = []
    
    for card_id in card_ids:
        rank, suit = parse_card_id(card_id)
        if rank is None:
            invalid_ids.append(card_id)
        else:
            parsed_cards.append((rank, suit, card_id.upper()))
    
    if invalid_ids:
        violations.append(f"Invalid card IDs: {invalid_ids}")
    
    # Count cards
    total_cards = len(parsed_cards)
    stats['total_cards'] = total_cards
    
    # Check deck size
    if total_cards < DECK_RULES['min_cards']:
        violations.append(
            f"Deck too small: {total_cards} cards (minimum: {DECK_RULES['min_cards']})"
        )
    if total_cards > DECK_RULES['max_cards']:
        violations.append(
            f"Deck too large: {total_cards} cards (maximum: {DECK_RULES['max_cards']})"
        )
    
    # Check copy limits
    card_counts = Counter(card[2] for card in parsed_cards)
    over_limit = [
        (card_id, count) 
        for card_id, count in card_counts.items() 
        if count > DECK_RULES['max_copies']
    ]
    if over_limit:
        for card_id, count in over_limit:
            violations.append(
                f"Too many copies of {card_id}: {count} (maximum: {DECK_RULES['max_copies']})"
            )
    
    # Check faction card requirements
    faction_cards = [c for c in parsed_cards if c[1] == faction_suit]
    faction_count = len(faction_cards)
    stats['faction_cards'] = faction_count
    stats['faction_suit'] = faction_suit
    
    if faction_count < DECK_RULES['faction_minimum']:
        violations.append(
            f"Too few {faction} cards: {faction_count} (minimum: {DECK_RULES['faction_minimum']})"
        )
    if faction_count > DECK_RULES['faction_maximum']:
        violations.append(
            f"Too many {faction} cards: {faction_count} (maximum: {DECK_RULES['faction_maximum']})"
        )
    
    # Suit distribution
    suit_counts = Counter(card[1] for card in parsed_cards)
    stats['suit_distribution'] = dict(suit_counts)
    
    # Rank distribution
    rank_counts = Counter(card[0] for card in parsed_cards)
    stats['rank_distribution'] = dict(rank_counts)
    
    # Calculate deck metrics for strategy analysis
    # Cluster potential (cards in same suit)
    max_suit_count = max(suit_counts.values()) if suit_counts else 0
    stats['cluster_potential'] = f"{max_suit_count}/{total_cards} ({max_suit_count/total_cards*100:.1f}%)" if total_cards > 0 else "N/A"
    
    # Rank spread
    if parsed_cards:
        rank_values = []
        rank_map = {'A': 1, 'J': 11, 'Q': 12, 'K': 13, 
                    '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, 
                    '7': 7, '8': 8, '9': 9, '10': 10}
        for rank, _, _ in parsed_cards:
            rank_values.append(rank_map.get(rank, 7))  # Default to 7 if unknown
        
        stats['rank_range'] = f"{min(rank_values)} - {max(rank_values)}"
        stats['avg_rank'] = f"{sum(rank_values)/len(rank_values):.2f}"
    
    # Warnings for suboptimal builds
    if faction_count == DECK_RULES['faction_minimum']:
        warnings.append("Running minimum faction cards - consider adding more for synergy")
    
    if max_suit_count < 8:
        warnings.append("Low suit concentration - cluster bonuses may be weak")
    
    # Check for 7s (center cards)
    sevens = [c for c in parsed_cards if c[0] == '7']
    if len(sevens) == 0:
        warnings.append("No 7s in deck - missing center cluster potential")
    stats['center_cards'] = len(sevens)
    
    # Check for extremes (Aces and Kings)
    extremes = [c for c in parsed_cards if c[0] in ('A', 'K')]
    stats['extreme_cards'] = len(extremes)
    
    valid = len(violations) == 0
    
    return ValidationResult(valid, violations, warnings, stats)


def load_deck_file(filepath: str) -> Tuple[List[str], str]:
    """Load deck from JSON file."""
    path = Path(filepath)
    
    if not path.exists():
        raise FileNotFoundError(f"Deck file not found: {filepath}")
    
    with open(path) as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        cards = data.get('cards', [])
        faction = data.get('faction', '')
    else:
        raise ValueError("Deck file must be a JSON object with 'cards' and 'faction' keys")
    
    return cards, faction


def analyze_deck_combos(card_ids: List[str], faction: str) -> Dict:
    """Analyze potential combos in a deck."""
    combos = {
        'chains': [],
        'clusters': [],
        'special': [],
    }
    
    # Parse cards
    cards = []
    rank_map = {'A': 1, 'J': 11, 'Q': 12, 'K': 13,
                '2': 2, '3': 3, '4': 4, '5': 5, '6': 6,
                '7': 7, '8': 8, '9': 9, '10': 10}
    
    for card_id in card_ids:
        rank, suit = parse_card_id(card_id)
        if rank:
            rank_val = rank_map.get(rank, 7)
            cards.append({'id': card_id.upper(), 'rank': rank, 'rank_val': rank_val, 'suit': suit})
    
    # Find sequential chains (same suit, sequential ranks)
    for suit in VALID_SUITS:
        suit_cards = sorted([c for c in cards if c['suit'] == suit], key=lambda x: x['rank_val'])
        
        # Find runs of 3+
        if len(suit_cards) >= 3:
            current_run = [suit_cards[0]]
            for i in range(1, len(suit_cards)):
                if suit_cards[i]['rank_val'] == suit_cards[i-1]['rank_val'] + 1:
                    current_run.append(suit_cards[i])
                else:
                    if len(current_run) >= 3:
                        combos['chains'].append({
                            'cards': [c['id'] for c in current_run],
                            'type': 'sequential',
                            'suit': suit,
                        })
                    current_run = [suit_cards[i]]
            
            if len(current_run) >= 3:
                combos['chains'].append({
                    'cards': [c['id'] for c in current_run],
                    'type': 'sequential',
                    'suit': suit,
                })
    
    # Find parallel combos (same rank, different suits)
    from collections import defaultdict
    by_rank = defaultdict(list)
    for c in cards:
        by_rank[c['rank']].append(c)
    
    for rank, rank_cards in by_rank.items():
        if len(rank_cards) >= 3:
            combos['clusters'].append({
                'cards': [c['id'] for c in rank_cards],
                'type': 'parallel',
                'rank': rank,
            })
    
    # Check for special combos
    # All 7s
    sevens = [c for c in cards if c['rank'] == '7']
    if len(sevens) == 4:
        combos['special'].append({
            'name': 'Perfect Center',
            'cards': [c['id'] for c in sevens],
            'bonus': '+40 points',
        })
    
    # Ace-King of same suit
    for suit in VALID_SUITS:
        has_ace = any(c['rank'] == 'A' and c['suit'] == suit for c in cards)
        has_king = any(c['rank'] == 'K' and c['suit'] == suit for c in cards)
        if has_ace and has_king:
            combos['special'].append({
                'name': 'Full Axis',
                'cards': [f"A{suit}", f"K{suit}"],
                'bonus': '+20 points',
            })
    
    # All Kings
    kings = [c for c in cards if c['rank'] == 'K']
    if len(kings) >= 3:
        combos['special'].append({
            'name': 'Apex Alignment',
            'cards': [c['id'] for c in kings],
            'bonus': 'High burst potential',
        })
    
    return combos


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Validate Quantum Resonance deck'
    )
    parser.add_argument(
        '--deck', '-d', type=str,
        help='Path to deck JSON file'
    )
    parser.add_argument(
        '--cards', '-c', type=str, nargs='+',
        help='Card IDs (e.g., AS 2S KH 7D)'
    )
    parser.add_argument(
        '--faction', '-f', type=str, required=True,
        help='Faction (spades, hearts, diamonds, clubs)'
    )
    parser.add_argument(
        '--analyze', '-a', action='store_true',
        help='Analyze potential combos'
    )
    
    args = parser.parse_args()
    
    # Load cards
    if args.deck:
        cards, faction = load_deck_file(args.deck)
        if args.faction:
            faction = args.faction
    elif args.cards:
        cards = args.cards
        faction = args.faction
    else:
        parser.error("Must provide --deck or --cards")
        return
    
    # Validate
    result = validate_deck(cards, faction)
    print(result)
    
    # Analyze combos if requested
    if args.analyze:
        print("\n" + "="*40)
        print("COMBO ANALYSIS")
        print("="*40)
        
        combos = analyze_deck_combos(cards, faction)
        
        if combos['chains']:
            print("\nSequential Chains:")
            for combo in combos['chains']:
                print(f"  {combo['suit']}: {' → '.join(combo['cards'])}")
        
        if combos['clusters']:
            print("\nParallel Clusters:")
            for combo in combos['clusters']:
                print(f"  Rank {combo['rank']}: {', '.join(combo['cards'])}")
        
        if combos['special']:
            print("\nSpecial Combos:")
            for combo in combos['special']:
                print(f"  {combo['name']}: {', '.join(combo['cards'])} ({combo['bonus']})")
        
        if not any(combos.values()):
            print("\nNo significant combos detected. Consider rebuilding.")
    
    # Exit code
    sys.exit(0 if result.valid else 1)


if __name__ == '__main__':
    main()
