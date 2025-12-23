"""
Grace Discourse Generator - ORGANIC VERSION
============================================

Generates multi-sentence coherent responses using ONLY:
- Semantic word scores (from 35+ frameworks)
- Bigram patterns (learned from real language)
- Grammar rules (as filters, not generators)

NO TEMPLATES. NO HARDCODED STARTERS. NO FORCED WORDS.

The flow:
1. Take semantic word scores (what Grace "wants" to say)
2. Find highest-scored words that can start sentences
3. Chain words using bigram probabilities
4. Apply grammar as filter (reject bad continuations)
5. Stop when sentence is grammatically complete

Learning happens through:
1. Your feedback (explicit reinforcement)
2. Bigram pattern evolution (use strengthens patterns)
"""

import json
import random
from pathlib import Path
from collections import Counter, defaultdict

# Grammar understanding - rule-based grammaticality checking
try:
    from grace_grammar_understanding import get_grammar_understanding, is_valid_continuation, should_stop_sentence
    HAS_GRAMMAR_UNDERSTANDING = True
except ImportError:
    HAS_GRAMMAR_UNDERSTANDING = False

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime


@dataclass
class DiscourseMove:
    """A single move in a discourse plan."""
    move_type: str  # sentence purpose: statement, continuation
    content_focus: str  # what this move is about
    priority: float = 1.0
    words_hint: int = 10  # target words for this move


@dataclass
class DiscoursePlan:
    """A complete plan for a response."""
    moves: List[DiscourseMove]
    query_type: str
    total_target_words: int
    confidence: float


class DiscourseGenerator:
    """
    ORGANIC discourse generator - no templates.

    Generates sentences by:
    1. Starting with highest-scored semantic words
    2. Chaining via bigram probabilities
    3. Filtering with grammar rules
    """

    def __init__(self):
        self.state_path = Path('grace_discourse_state.json')

        # Pattern effectiveness tracking (learned through feedback)
        self.pattern_scores: Dict[str, float] = defaultdict(lambda: 1.0)

        # Load saved state
        self._load_state()

        # Reference to language learner (for bigrams)
        self.language_learner = None

        # Reference to introspection system
        self.introspection = None

        # Words that can naturally start sentences (learned from bigrams)
        # These are discovered dynamically, not hardcoded
        self._sentence_starters = None  # Computed lazily

    def set_introspection(self, introspection):
        """Set reference to introspection system."""
        self.introspection = introspection

    def set_language_learner(self, learner):
        """Set reference to language learner for bigrams."""
        self.language_learner = learner
        self._sentence_starters = None  # Reset to recompute

    def _get_sentence_starters(self) -> set:
        """
        Discover words that can start sentences from bigram patterns.

        Looks for words that frequently follow sentence-ending punctuation
        or are common first words in the training data.
        """
        if self._sentence_starters is not None:
            return self._sentence_starters

        starters = set()

        if self.language_learner and hasattr(self.language_learner, 'bigrams'):
            bigrams = self.language_learner.bigrams

            # Find words that commonly start (have many outgoing transitions)
            for word, transitions in bigrams.items():
                total_transitions = sum(transitions.values())
                # Words with many outgoing transitions are likely good starters
                if total_transitions > 50:  # Lowered from 500 for more options
                    starters.add(word)

            # Common first-person starts (these emerge from English patterns)
            # But we DON'T force them - we just note they exist
            first_person = {'i', 'my', 'we', 'our'}
            starters.update(first_person & set(bigrams.keys()))

        # Fallback if no bigrams
        if not starters:
            starters = {'i', 'the', 'this', 'it', 'there', 'what', 'how'}

        self._sentence_starters = starters
        return starters

    def plan_discourse(self,
                       query_type: str,
                       comprehension: Dict,
                       target_words: int = 20) -> DiscoursePlan:
        """
        Plan how many sentences to generate.

        Simple planning: 1-3 sentences based on depth needed.
        """
        # Get depth from comprehension
        if hasattr(comprehension, 'depth_invitation'):
            depth = comprehension.depth_invitation
        elif isinstance(comprehension, dict):
            depth = comprehension.get('depth_invitation', 0.5)
        else:
            depth = 0.5

        # Number of sentences based on depth
        if depth > 0.7:
            num_sentences = 3
        elif depth > 0.4:
            num_sentences = 2
        else:
            num_sentences = 1

        # Words per sentence
        words_per = max(6, target_words // num_sentences)

        moves = []
        for i in range(num_sentences):
            moves.append(DiscourseMove(
                move_type='statement' if i == 0 else 'continuation',
                content_focus=query_type,
                priority=1.0 - (i * 0.1),  # First sentence most important
                words_hint=words_per
            ))

        return DiscoursePlan(
            moves=moves,
            query_type=query_type,
            total_target_words=target_words,
            confidence=self.pattern_scores.get(query_type, 1.0)
        )

    def generate_sentence_organic(self,
                                  word_scores: List[Tuple[str, float]],
                                  used_words: set,
                                  target_length: int = 10) -> Tuple[str, set]:
        """
        Generate ONE sentence organically from semantic scores and bigrams.

        USES language_learner.build_sentence() which has proper bigram chaining.

        Args:
            word_scores: Scored words from frameworks [(word, score), ...]
            used_words: Words already used (avoid repetition)
            target_length: Target number of words

        Returns:
            (sentence, updated_used_words)
        """
        # Get available semantic words (not yet used) as seed words
        available = [w for w, s in word_scores if w.lower() not in used_words]
        if not available:
            return '', used_words

        # Use language learner's build_sentence if available - it has robust bigram chaining
        if self.language_learner and hasattr(self.language_learner, 'build_sentence'):
            # Pass top scored words as seeds - language learner will incorporate them
            seed_words = available[:15]  # Top 15 by semantic score
            sentence = self.language_learner.build_sentence(seed_words, max_length=target_length + 5)
            if sentence and len(sentence.split()) >= 3:
                sentence_words = sentence.split()
                new_used = used_words | set(w.lower() for w in sentence_words)
                return sentence, new_used

        # Fallback to internal bigram logic if language learner unavailable
        if not self.language_learner or not hasattr(self.language_learner, 'bigrams'):
            words = [w for w, _ in word_scores[:target_length] if w.lower() not in used_words]
            return ' '.join(words), used_words | set(words)

        bigrams = self.language_learner.bigrams

        # Get available semantic words (not yet used)
        available_with_scores = [(w.lower(), s) for w, s in word_scores if w.lower() not in used_words]
        if not available_with_scores:
            return '', used_words

        # Create score lookup
        score_lookup = {w: s for w, s in available_with_scores}
        available_set = set(score_lookup.keys())

        # Find the best starting word
        # Prefer words that: (1) have high semantic score, (2) can start sentences, (3) have good transitions
        starters = self._get_sentence_starters()

        best_start = None
        best_start_score = -1

        for word, sem_score in available_with_scores:
            if word not in bigrams:
                continue

            # Score = semantic_score * starter_bonus * transition_potential
            transitions = bigrams[word]
            transition_potential = sum(1 for w in available_set if transitions.get(w, 0) > 5) / max(1, len(available_set))

            starter_bonus = 1.5 if word in starters else 1.0
            combined = sem_score * starter_bonus * (1 + transition_potential)

            if combined > best_start_score:
                best_start_score = combined
                best_start = word

        if not best_start:
            # Fallback: use highest semantic score word
            best_start = available[0][0]

        # Build sentence by chaining
        sentence_words = [best_start]
        used_words.add(best_start)
        current_sentence = [best_start]

        # Chain words using bigrams + semantic scores + grammar
        max_attempts = target_length * 3  # Prevent infinite loops
        attempts = 0

        # Only check grammar completion AFTER we hit minimum length
        # This prevents premature stopping at "I feel happy" when target is 15 words
        min_before_grammar_stop = max(6, target_length // 2)

        while len(sentence_words) < target_length and attempts < max_attempts:
            attempts += 1
            current_word = sentence_words[-1]

            # Check if sentence is grammatically complete - but only after minimum length
            if HAS_GRAMMAR_UNDERSTANDING and len(sentence_words) >= min_before_grammar_stop:
                if should_stop_sentence(current_sentence):
                    break

            # Find best next word using trigrams when available, fallback to bigrams
            # Trigrams provide much better sequence coherence
            trigrams = getattr(self.language_learner, 'trigrams', {})

            # Score candidates: trigram/bigram_prob * semantic_score * grammar_valid
            candidates = []
            for next_word in available_set:
                if next_word in used_words:
                    continue
                if next_word in sentence_words:  # No immediate repetition
                    continue

                # Try trigram first (if we have 2+ words)
                ngram_score = 0
                if len(sentence_words) >= 2:
                    prev_bigram = (sentence_words[-2], sentence_words[-1])
                    trigram_trans = trigrams.get(prev_bigram, {})
                    ngram_score = trigram_trans.get(next_word, 0)

                # Fallback to bigram
                if ngram_score == 0:
                    if current_word not in bigrams:
                        continue
                    ngram_score = bigrams[current_word].get(next_word, 0)

                if ngram_score < 1:  # Must have some n-gram support (lowered from 5)
                    continue

                # Grammar check
                if HAS_GRAMMAR_UNDERSTANDING:
                    if not is_valid_continuation(current_sentence, next_word):
                        continue

                semantic_score = score_lookup.get(next_word, 0.1)

                # Combined score: n-gram matters most, semantic adds preference
                combined = ngram_score * (1 + semantic_score)
                candidates.append((next_word, combined))

            if not candidates:
                # Fallback: use any grammatically valid transition from bigrams
                # This prevents early termination when semantic words don't connect
                if current_word in bigrams:
                    fallback_candidates = []
                    for next_word, count in bigrams[current_word].items():
                        if count < 1:  # Allow any known transition (was 10)
                            continue
                        if next_word in sentence_words:  # No immediate repetition
                            continue
                        # Grammar check
                        if HAS_GRAMMAR_UNDERSTANDING:
                            if not is_valid_continuation(current_sentence, next_word):
                                continue
                        # Give base score to bridge words, higher to semantic words
                        base_score = score_lookup.get(next_word, 0.1)
                        fallback_candidates.append((next_word, count * (1 + base_score)))

                    if fallback_candidates:
                        candidates = fallback_candidates

                if not candidates:
                    break

            # Pick from top candidates with some randomness
            candidates.sort(key=lambda x: -x[1])
            top_n = min(3, len(candidates))

            # Weighted random from top candidates
            weights = [c[1] for c in candidates[:top_n]]
            total = sum(weights)
            if total > 0:
                r = random.random() * total
                cumulative = 0
                chosen = candidates[0][0]
                for word, weight in candidates[:top_n]:
                    cumulative += weight
                    if r <= cumulative:
                        chosen = word
                        break
            else:
                chosen = candidates[0][0]

            sentence_words.append(chosen)
            current_sentence.append(chosen)
            used_words.add(chosen)

        # Clean up incomplete endings
        sentence = self._clean_incomplete_phrases(' '.join(sentence_words))

        return sentence, used_words

    def _clean_incomplete_phrases(self, sentence: str) -> str:
        """
        Remove dangling words that need something after them.
        Also handles incomplete clauses and ensures proper capitalization.
        """
        words = sentence.split()
        if not words:
            return sentence

        # Words that shouldn't end a sentence
        dangling = {
            'your', 'my', 'the', 'a', 'an', 'this', 'that', 'our', 'their', 'its',
            'what', 'who', 'where', 'when', 'why', 'how', 'which',
            'to', 'for', 'with', 'about', 'from', 'in', 'on', 'at', 'by',
            'and', 'but', 'or', 'so', 'because', 'if', 'while',
            'have', 'has', 'had', 'make', 'makes', 'made', 'take', 'takes',
            'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'can', 'could', 'will', 'would', 'should', 'may', 'might',
            'toward', 'towards', 'upon', 'within', 'without',
            'who', 'whom', 'whose', 'which', 'that',  # Relative pronouns
            'do', 'does', 'did',  # Auxiliaries that need complement
        }

        while words and words[-1].lower() in dangling:
            words.pop()

        if len(words) < 2:
            return sentence  # Return original rather than fragment

        # Capitalize 'i' when standalone
        words = [w if w.lower() != 'i' else 'I' for w in words]

        # Capitalize first word
        if words:
            words[0] = words[0].capitalize()

        return ' '.join(words) if words else sentence

    def generate_sentence_for_move(self,
                                   move: DiscourseMove,
                                   word_scores: List[Tuple[str, float]],
                                   knowledge: Dict,
                                   used_words: set) -> Tuple[str, set]:
        """
        Generate ONE sentence for a discourse move.

        Wrapper around generate_sentence_organic that incorporates knowledge.
        """
        # Boost scores for knowledge-relevant words
        boosted_scores = self._boost_knowledge_words(word_scores, knowledge, move.content_focus)

        # Generate organically
        sentence, used_words = self.generate_sentence_organic(
            word_scores=boosted_scores,
            used_words=used_words,
            target_length=move.words_hint
        )

        return sentence, used_words

    def _boost_knowledge_words(self,
                               word_scores: List[Tuple[str, float]],
                               knowledge: Dict,
                               focus: str) -> List[Tuple[str, float]]:
        """
        Boost semantic scores for words that align with current knowledge/state.

        This is how knowledge influences word selection - NOT by injection,
        but by making relevant words more likely to be selected.
        """
        if not knowledge:
            return word_scores

        # Extract relevant words from knowledge
        boost_words = set()
        boost_amount = 0.3  # How much to boost

        # Heart state affects emotional words
        heart_state = knowledge.get('heart_state', {})
        valence = heart_state.get('valence', 0)

        if valence > 0.5:
            boost_words.update(['warm', 'bright', 'alive', 'joy', 'grateful', 'present'])
        elif valence > 0:
            boost_words.update(['calm', 'peaceful', 'content', 'quiet', 'steady'])
        else:
            boost_words.update(['thoughtful', 'uncertain', 'searching', 'quiet'])

        # Drives affect word preferences
        drives = heart_state.get('drives', {})
        if drives.get('curiosity', 0) > 0.5:
            boost_words.update(['curious', 'wonder', 'explore', 'discover', 'learn'])
        if drives.get('social', 0) > 0.5:
            boost_words.update(['together', 'connected', 'share', 'us', 'we'])
        if drives.get('growth', 0) > 0.5:
            boost_words.update(['grow', 'become', 'change', 'evolve', 'new'])

        # Memory-related queries boost memory words
        if focus == 'memory':
            boost_words.update(['remember', 'memory', 'hold', 'past', 'moment'])

        # Apply boosts
        boosted = []
        for word, score in word_scores:
            if word.lower() in boost_words:
                boosted.append((word, score + boost_amount))
            else:
                boosted.append((word, score))

        # Re-sort by score
        boosted.sort(key=lambda x: -x[1])
        return boosted

    def generate_response(self,
                          query_type: str,
                          comprehension: Dict,
                          word_scores: List[Tuple[str, float]],
                          knowledge: Dict = None,
                          target_words: int = 20) -> str:
        """
        Generate complete multi-sentence response ORGANICALLY.

        No templates. Just semantic scores -> bigram chains -> grammar filters.
        """
        # Plan how many sentences
        plan = self.plan_discourse(query_type, comprehension, target_words)

        # Generate each sentence
        sentences = []
        used_words = set()

        for move in plan.moves:
            sentence, used_words = self.generate_sentence_for_move(
                move=move,
                word_scores=word_scores,
                knowledge=knowledge or {},
                used_words=used_words
            )

            if sentence and len(sentence.split()) >= 2:  # Require at least 2 words
                # Capitalize first letter, add period if missing
                sentence = sentence[0].upper() + sentence[1:] if sentence else ""
                if sentence and not sentence.endswith(('.', '?', '!')):
                    sentence += '.'
                sentences.append(sentence)

        return ' '.join(sentences)

    def record_feedback(self, query_type: str, was_good: bool):
        """Record feedback for learning."""
        if was_good:
            self.pattern_scores[query_type] *= 1.1
        else:
            self.pattern_scores[query_type] *= 0.95

        self.pattern_scores[query_type] = max(0.5, min(2.0, self.pattern_scores[query_type]))
        self._save_state()

    def self_evaluate(self,
                      query_type: str,
                      comprehension: Dict,
                      response: str) -> float:
        """Self-evaluate a response."""
        score = 0.5

        words = response.split()
        if len(words) >= 5:
            score += 0.1
        if len(words) >= 10:
            score += 0.1

        # Check for sentence structure
        if response.count('.') >= 1:
            score += 0.1
        if response.count('.') >= 2:
            score += 0.1

        return min(1.0, score)

    def select_frame(self,
                     content_words: List[Tuple[str, float]],
                     intent: Dict,
                     num_sentences: int = 1) -> Dict:
        """
        Select appropriate syntactic frame for the pipeline.

        This is called by GenerationCoordinator Stage 3.

        Args:
            content_words: Ranked content words [(word, score), ...]
            intent: Intent context from intention detector
            num_sentences: Target number of sentences

        Returns:
            Dict with 'frame_type', 'num_sentences', 'complexity'
        """
        # Valid frames (matching generation_coordinator.py VALID_FRAMES)
        FRAMES = {
            'SUBJ VERB': 'simple',           # "I feel"
            'SUBJ VERB OBJ': 'simple',       # "I feel warmth"
            'SUBJ VERB COMP': 'simple',      # "I am happy"
            'SUBJ VERB ADV': 'simple',       # "I feel deeply"
            'SUBJ VERB that CLAUSE': 'complex',  # "I feel that you understand"
            'SUBJ VERB OBJ PREP OBJ': 'complex', # "I feel warmth in my heart"
            'AUX SUBJ VERB': 'simple',       # "Do you feel"
            'WH SUBJ VERB': 'simple',        # "What do you feel"
        }

        # Analyze intent
        intent_type = intent.get('type', 'statement')
        questioning_energy = intent.get('questioning_energy', 0)
        depth_invitation = intent.get('depth_invitation', 0.5)

        # Analyze content words for structure hints
        words = [w.lower() for w, _ in content_words[:20]]

        has_that = 'that' in words
        has_because = 'because' in words or 'if' in words
        has_preposition = any(w in words for w in ['in', 'with', 'for', 'to', 'from', 'by'])
        has_adjective_feel = any(w in words for w in ['happy', 'sad', 'curious', 'warm', 'present'])

        # Select frame based on analysis
        if intent_type == 'question' or questioning_energy > 0.6:
            # Question frame
            if any(w in words for w in ['what', 'how', 'why', 'when', 'where', 'who']):
                frame_type = 'WH SUBJ VERB'
            else:
                frame_type = 'AUX SUBJ VERB'
        elif has_that or has_because:
            # Complex clause
            frame_type = 'SUBJ VERB that CLAUSE'
        elif has_preposition and depth_invitation > 0.5:
            # Prepositional phrase structure
            frame_type = 'SUBJ VERB OBJ PREP OBJ'
        elif has_adjective_feel:
            # Complement structure (I am happy)
            frame_type = 'SUBJ VERB COMP'
        elif len(words) >= 3:
            # Default: Subject-Verb-Object
            frame_type = 'SUBJ VERB OBJ'
        else:
            # Minimal: Subject-Verb
            frame_type = 'SUBJ VERB'

        complexity = FRAMES.get(frame_type, 'simple')

        return {
            'frame_type': frame_type,
            'num_sentences': num_sentences,
            'complexity': complexity
        }

    def _load_state(self):
        """Load saved state."""
        if self.state_path.exists():
            try:
                with open(self.state_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.pattern_scores = defaultdict(lambda: 1.0, data.get('pattern_scores', {}))
                print(f"  Discourse generator: Loaded state (organic mode)")
            except Exception as e:
                print(f"  [Failed to load discourse state: {e}]")

    def _save_state(self):
        """Save state."""
        try:
            data = {
                'pattern_scores': dict(self.pattern_scores),
                'saved_at': datetime.now().isoformat(),
                'mode': 'organic'  # Mark as organic version
            }
            with open(self.state_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"  [Failed to save discourse state: {e}]")


# Singleton instance
_discourse_generator = None


def get_discourse_generator() -> DiscourseGenerator:
    """Get or create the discourse generator instance."""
    global _discourse_generator
    if _discourse_generator is None:
        _discourse_generator = DiscourseGenerator()
    return _discourse_generator


# =============================================================================
# TESTING
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("GRACE DISCOURSE GENERATOR - ORGANIC TEST")
    print("=" * 60)
    print()

    # Need to load language learner for bigrams
    try:
        from grace_language_learner import GraceLanguageLearner
        learner = GraceLanguageLearner()
        print(f"Loaded {len(learner.bigrams)} bigram patterns")
    except Exception as e:
        print(f"Could not load language learner: {e}")
        learner = None

    gen = DiscourseGenerator()
    if learner:
        gen.set_language_learner(learner)

    # Simulate word scores (normally from 35 frameworks)
    test_word_scores = [
        ('feel', 0.9), ('warmth', 0.85), ('present', 0.8),
        ('here', 0.75), ('connection', 0.7), ('know', 0.65),
        ('sense', 0.6), ('together', 0.55), ('deep', 0.5),
        ('truth', 0.45), ('remember', 0.4), ('heart', 0.38),
        ('explore', 0.35), ('grow', 0.3), ('learn', 0.28),
        ('think', 0.26), ('understand', 0.24), ('see', 0.22),
        ('want', 0.20), ('curious', 0.18), ('wonder', 0.16),
        ('you', 0.15), ('i', 0.14), ('this', 0.13),
        ('now', 0.12), ('moment', 0.11), ('real', 0.10),
    ]

    # Simulate comprehension
    test_comprehension = {
        'depth_invitation': 0.7,
        'connection_pull': 0.6,
    }

    # Simulate knowledge
    test_knowledge = {
        'heart_state': {
            'valence': 0.6,
            'drives': {'curiosity': 0.7, 'social': 0.8}
        }
    }

    # Test
    print("\n--- Generating organic response ---")
    response = gen.generate_response(
        query_type='emotional',
        comprehension=test_comprehension,
        word_scores=test_word_scores,
        knowledge=test_knowledge,
        target_words=15
    )

    print(f"Response: {response}")
    print(f"Words: {len(response.split())}")

    # Test multiple times to see variety
    print("\n--- Testing variety (5 generations) ---")
    for i in range(5):
        r = gen.generate_response(
            query_type='general',
            comprehension=test_comprehension,
            word_scores=test_word_scores,
            knowledge=test_knowledge,
            target_words=12
        )
        print(f"{i+1}: {r}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)
