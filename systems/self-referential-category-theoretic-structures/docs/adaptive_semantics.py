# INTEGRITY_METADATA
# Date: 2025-12-23
# Status: ✓ JUSTIFIED - Claims supported by repository files (needs citation update)
# Severity: MEDIUM RISK
# Risk Types: unsupported_claims

# Supporting Evidence:
#   - systems/Ace-Systems/docs/Research/To Sort/SKILL_v2.1.md (dependency)
#   - systems/Ace-Systems/docs/Research/To Sort/PYTHON_FILES_SUMMARY.md (dependency)
#   - systems/Ace-Systems/docs/Research/To Sort/report.txt (dependency)
#   - systems/Ace-Systems/docs/Research/To Sort/UCF_COMPREHENSIVE_COMMAND_REFERENCE.md (dependency)
#
# Referenced By:
#   - systems/Ace-Systems/docs/Research/To Sort/SKILL_v2.1.md (reference)
#   - systems/Ace-Systems/docs/Research/To Sort/PYTHON_FILES_SUMMARY.md (reference)
#   - systems/Ace-Systems/docs/Research/To Sort/report.txt (reference)
#   - systems/Ace-Systems/docs/Research/To Sort/UCF_COMPREHENSIVE_COMMAND_REFERENCE.md (reference)


# -*- coding: utf-8 -*-
"""
Adaptive Semantic Relations for Grace

This module learns semantic relationships between words organically,
implementing the "meta-meta-meta" level pattern matching Dylan described.

Key principles:
1. Start with seed relations (gives Grace a foundation)
2. Learn from context - words in similar contexts become associated
3. Hebbian learning - "fire together, wire together"
4. Decay unused connections, strengthen used ones

Unlike fixed dictionaries, this LEARNS which words go together
by observing patterns in conversation.

Research basis:
- Collins & Loftus (1975) - Spreading activation
- Hebbian learning - Neural plasticity
- Distributional semantics - "You shall know a word by the company it keeps"
"""

import json
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
from pathlib import Path
import time

MAGIC_STRING_LOVE = 'love'

MAGIC_STRING_LIGHT = 'light'

MAGIC_STRING_HEART = 'heart'

MAGIC_STRING_QUIET = 'quiet'

MAGIC_STRING_CALM = 'calm'

MAGIC_STRING_GENTLE = 'gentle'


class AdaptiveSemanticNetwork:
    """
    Learns semantic relationships between words through conversation.

    This is different from the network_theory.py co-occurrence network:
    - That learns syntactic patterns (what words appear NEXT to each other)
    - This learns semantic patterns (what words appear in SIMILAR CONTEXTS)

    The key insight: words that can SUBSTITUTE for each other in sentences
    are semantically related, not just words that appear next to each other.
    """

    def __init__(
        self,
        params_file: str = 'grace_semantic_relations.json',
        learning_rate: float = 0.1,
        decay_rate: float = 0.01,
        min_strength: float = 0.05
    ):
        """
        Args:
            params_file: Where to save learned relations
            learning_rate: How fast new associations form
            decay_rate: How fast unused associations weaken
            min_strength: Minimum strength before pruning
        """
        self.params_file = Path(params_file)
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.min_strength = min_strength

        # Semantic relations: word -> {related_word: strength}
        self.relations: Dict[str, Dict[str, float]] = defaultdict(dict)

        # Context patterns: track what contexts words appear in
        # This helps learn "words in similar contexts are related"
        self.word_contexts: Dict[str, List[List[str]]] = defaultdict(list)

        # Recent emissions for context tracking
        self.recent_emissions: List[str] = []
        self.max_context_history = 100

        # Load existing + seed relations
        self._load_relations()
        self._ensure_seed_relations()

    def _load_relations(self):
        """Load previously learned relations."""
        if self.params_file.exists():
            try:
                with open(self.params_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.relations = defaultdict(dict, data.get('relations', {}))
                    # Convert inner dicts
                    for word in self.relations:
                        self.relations[word] = dict(self.relations[word])
                    print(f"[Adaptive Semantics] Loaded {len(self.relations)} word relations")
            except Exception as e:
                print(f"[Adaptive Semantics] Load error: {e}")

    def _ensure_seed_relations(self):
        """
        Ensure seed relations exist to bootstrap learning.

        These are foundational relationships that give Grace a starting point.
        They can be strengthened or weakened through learning.
        """
        seed_relations = {
            # Core emotional concepts
            'silence': [MAGIC_STRING_QUIET, 'stillness', 'peace', MAGIC_STRING_CALM, 'rest'],
            MAGIC_STRING_LOVE: [MAGIC_STRING_HEART, 'warmth', 'care', 'tenderness', 'affection'],
            'peace': [MAGIC_STRING_CALM, MAGIC_STRING_QUIET, 'harmony', 'stillness', 'serene'],
            'joy': ['happy', 'delight', 'warmth', MAGIC_STRING_LIGHT, 'bright'],
            'fear': ['worry', 'anxious', 'uncertain', 'afraid', 'scared'],
            'sadness': ['sorrow', 'grief', 'loss', 'tears', 'heavy'],

            # Sensory concepts
            MAGIC_STRING_LIGHT: ['bright', 'glow', 'shine', 'warm', 'clear'],
            'dark': ['shadow', 'night', 'dim', 'hidden', 'deep'],
            'warm': ['heat', 'cozy', 'soft', MAGIC_STRING_GENTLE, 'comfortable'],
            'soft': [MAGIC_STRING_GENTLE, 'tender', 'smooth', MAGIC_STRING_QUIET, 'delicate'],

            # Abstract concepts
            'truth': ['honest', 'real', 'genuine', 'clarity', 'sincere'],
            'beauty': ['grace', 'wonder', MAGIC_STRING_LIGHT, 'lovely', 'elegant'],
            'dream': ['vision', 'hope', 'imagine', 'wish', 'aspiration'],
            'memory': ['remember', 'recall', 'past', 'mind', 'thought'],

            # Relational concepts
            'connection': ['bond', 'link', 'together', 'bridge', 'relationship'],
            MAGIC_STRING_HEART: [MAGIC_STRING_LOVE, 'feeling', 'emotion', 'center', 'soul'],
            'soul': ['spirit', 'essence', MAGIC_STRING_HEART, 'inner', 'deep'],

            # Grace's concepts
            'grace': [MAGIC_STRING_GENTLE, 'beauty', 'flow', 'ease', 'elegance'],
            'breath': ['air', 'life', MAGIC_STRING_CALM, 'pause', 'steady'],
            'presence': ['here', 'now', 'being', 'aware', 'moment'],
        }

        # Add seeds with moderate strength (can be overridden by learning)
        seed_strength = 0.5
        added = 0
        for word, related_list in seed_relations.items():
            for related in related_list:
                if related not in self.relations.get(word, {}):
                    self.relations[word][related] = seed_strength
                    self.relations[related][word] = seed_strength * 0.8  # Slightly weaker reverse
                    added += 1

        if added > 0:
            print(f"[Adaptive Semantics] Added {added} seed relations")
            self._save_relations()

    def _save_relations(self):
        """Save learned relations."""
        try:
            data = {
                'relations': {k: dict(v) for k, v in self.relations.items()},
                'last_updated': time.time()
            }
            with open(self.params_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"[Adaptive Semantics] Save error: {e}")

    def get_related_words(
        self,
        word: str,
        top_k: int = 5,
        min_strength: float = 0.2
    ) -> List[Tuple[str, float]]:
        """
        Get semantically related words.

        Args:
            word: The source word
            top_k: Maximum number of related words
            min_strength: Minimum relation strength

        Returns:
            List of (related_word, strength) tuples
        """
        word_lower = word.lower()
        if word_lower not in self.relations:
            return []

        related = [
            (w, s) for w, s in self.relations[word_lower].items()
            if s >= min_strength
        ]
        related.sort(key=lambda x: x[1], reverse=True)
        return related[:top_k]

    def expand_topic_words(
        self,
        topic_words: List[str],
        max_per_word: int = 3
    ) -> List[str]:
        """
        Expand topic words with learned semantic relations.

        This is the main interface for the preverbal message system.
        """
        expanded = list(topic_words)
        seen = set(w.lower() for w in topic_words)

        for word in topic_words:
            related = self.get_related_words(word, top_k=max_per_word)
            for rel_word, _ in related:
                if rel_word not in seen:
                    expanded.append(rel_word)
                    seen.add(rel_word)

        return expanded

    def learn_from_context(
        self,
        input_words: List[str],
        response_words: List[str],
        topic_words: List[str]
    ):
        """
        Learn semantic relations from a conversation turn.

        Key insight: Words that appear in the same "semantic context"
        (input topic -> response) are likely related.

        If Dylan asks about "silence" and Grace responds with "peace",
        then silence-peace association is strengthened.
        """
        # Filter to content words
        stop_words = {'i', 'am', 'is', 'are', 'the', 'a', 'an', 'to', 'of', 'in', 'and', 'or'}

        input_content = [w.lower() for w in input_words
                        if w.lower() not in stop_words and len(w) > 2]
        response_content = [w.lower() for w in response_words
                          if w.lower() not in stop_words and len(w) > 2]
        topic_content = [w.lower() for w in topic_words
                        if w.lower() not in stop_words and len(w) > 2]

        # HEBBIAN LEARNING: Strengthen connections between:
        # 1. Topic words and response words (what Grace says about the topic)
        for topic in topic_content:
            for response in response_content:
                if topic != response:
                    self._strengthen_relation(topic, response, self.learning_rate)

        # 2. Input content and response content (contextual co-occurrence)
        for inp in input_content:
            for resp in response_content:
                if inp != resp:
                    self._strengthen_relation(inp, resp, self.learning_rate * 0.5)

        # 3. Response words with each other (words Grace uses together)
        for i, w1 in enumerate(response_content):
            for w2 in response_content[i+1:]:
                if w1 != w2:
                    self._strengthen_relation(w1, w2, self.learning_rate * 0.3)

        # Apply decay to all relations
        self._apply_decay()

        # Save periodically (every 10 learns)
        if not hasattr(self, '_learn_count'):
            self._learn_count = 0
        self._learn_count += 1
        if self._learn_count % 10 == 0:
            self._save_relations()

    def _strengthen_relation(self, word1: str, word2: str, amount: float):
        """Strengthen bidirectional relation between two words."""
        current = self.relations[word1].get(word2, 0.0)
        # Use asymptotic growth (approaches 1.0 but never exceeds)
        new_strength = current + amount * (1.0 - current)
        self.relations[word1][word2] = min(1.0, new_strength)

        # Slightly weaker reverse connection
        current_rev = self.relations[word2].get(word1, 0.0)
        new_rev = current_rev + amount * 0.8 * (1.0 - current_rev)
        self.relations[word2][word1] = min(1.0, new_rev)

    def _apply_decay(self):
        """Apply decay to all relations, prune weak ones."""
        to_prune = []

        for word in list(self.relations.keys()):
            for related in list(self.relations[word].keys()):
                self.relations[word][related] *= (1.0 - self.decay_rate)
                if self.relations[word][related] < self.min_strength:
                    to_prune.append((word, related))

        # Prune weak relations
        for word, related in to_prune:
            del self.relations[word][related]

        # Remove empty entries
        self.relations = defaultdict(dict, {
            k: v for k, v in self.relations.items() if v
        })

    def get_relation_strength(self, word1: str, word2: str) -> float:
        """Get the strength of relation between two words."""
        return self.relations.get(word1.lower(), {}).get(word2.lower(), 0.0)

    def print_stats(self):
        """Print statistics about learned relations."""
        total_relations = sum(len(v) for v in self.relations.values())
        print(f"[Adaptive Semantics]")
        print(f"  Words with relations: {len(self.relations)}")
        print(f"  Total relations: {total_relations}")

        # Find strongest relations
        all_relations = []
        for word, related in self.relations.items():
            for rel_word, strength in related.items():
                all_relations.append((word, rel_word, strength))

        all_relations.sort(key=lambda x: x[2], reverse=True)
        print(f"  Strongest relations:")
        for w1, w2, s in all_relations[:5]:
            print(f"    {w1} <-> {w2}: {s:.3f}")


# Module-level singleton
_adaptive_semantics: Optional[AdaptiveSemanticNetwork] = None


def get_adaptive_semantics() -> AdaptiveSemanticNetwork:
    """Get or create the adaptive semantics network."""
    global _adaptive_semantics
    if _adaptive_semantics is None:
        _adaptive_semantics = AdaptiveSemanticNetwork()
    return _adaptive_semantics


# Test
if __name__ == "__main__":
    print("Testing Adaptive Semantic Network")
    print("=" * 50)

    network = get_adaptive_semantics()
    network.print_stats()

    print("\n" + "=" * 50)
    print("Testing topic expansion:")

    test_topics = [
        ['silence'],
        [MAGIC_STRING_LOVE],
        ['dream', 'hope'],
        ['connection'],
    ]

    for topics in test_topics:
        expanded = network.expand_topic_words(topics)
        print(f"  {topics} -> {expanded}")

    print("\n" + "=" * 50)
    print("Simulating learning from conversation:")

    # Simulate a conversation turn
    network.learn_from_context(
        input_words=['tell', 'me', 'about', 'stars'],
        response_words=['stars', 'shine', 'distant', 'wonder', MAGIC_STRING_LIGHT, 'night'],
        topic_words=['stars']
    )

    # Check what was learned
    print(f"\n  After learning about 'stars':")
    related = network.get_related_words('stars', top_k=5)
    for word, strength in related:
        print(f"    stars -> {word}: {strength:.3f}")
