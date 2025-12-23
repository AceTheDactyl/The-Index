"""
Generation Coordinator - 8-Stage Language Pipeline
===================================================
Orchestrates the complete language generation pipeline:

1. Content Selection - What meaningful words to use
2. Emergence Check - Do n-grams already imply structure?
3. Structural Frame - What sentence structure fits intent
4. Slot Assignment - Which word goes in which slot
5. Function Words - Articles, prepositions, auxiliaries
6. Agreement/Inflection - Verb conjugation, tense
7. Connectors - Intra and inter-sentence connections
8. Punctuation - Periods, commas, question marks
9. Validation - Full quality scoring (unified_scorer)

Design principle: NO file builds complete sentences.
Each stage does ONE transformation and passes forward.
Lightweight validation between stages, full scoring only in Stage 8.

See CLAUDE.md and docs/PIPELINE_EXTRACTION_PLAN.md for details.
"""

from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import time
import logging


# =============================================================================
# STANDARD DATA TYPES (Proper Dataclasses)
# =============================================================================

@dataclass
class ContentWords:
    """Stage 1 output: Ranked content words with scores."""
    words: List[Tuple[str, float]] = field(default_factory=list)  # [(word, score), ...]

    def __len__(self) -> int:
        return len(self.words)

    def __iter__(self):
        return iter(self.words)

    def __bool__(self) -> bool:
        return len(self.words) > 0

    def top(self, n: int = 10) -> List[Tuple[str, float]]:
        """Return top N words by score."""
        return sorted(self.words, key=lambda x: x[1], reverse=True)[:n]

    def word_list(self) -> List[str]:
        """Return just the words, no scores."""
        return [w for w, _ in self.words]

    @classmethod
    def from_list(cls, words: List[Tuple[str, float]]) -> 'ContentWords':
        """Create from raw list."""
        return cls(words=words)


@dataclass
class EmergenceResult:
    """Stage 2 output: Emergence check result."""
    emerged_sequence: Optional[List[str]] = None  # The emerged word sequence, or None
    confidence: float = 0.0                        # How confident the emergence is

    @property
    def bypassed(self) -> bool:
        """True if we should bypass frame selection."""
        return self.emerged_sequence is not None and self.confidence > 0.4


@dataclass
class FrameResult:
    """Stage 3 output: Selected structural frame."""
    frame_type: str = 'SUBJ VERB OBJ'  # Frame pattern
    num_sentences: int = 1              # Target sentence count
    complexity: str = 'simple'          # 'simple' or 'complex'

    def required_slots(self) -> Set[str]:
        """Get required slots for this frame."""
        return REQUIRED_SLOTS.get(self.frame_type, set())


@dataclass
class SlottedWords:
    """Stage 4 output: Words assigned to slots."""
    slots: Dict[str, str] = field(default_factory=dict)  # {"SUBJ": "I", "VERB": "feel", ...}
    extras: List[str] = field(default_factory=list)       # Words not assigned to slots

    def get(self, slot: str, default: str = '') -> str:
        return self.slots.get(slot, default)

    def has_slot(self, slot: str) -> bool:
        return slot in self.slots

    def filled_slots(self) -> Set[str]:
        return set(self.slots.keys())

    def to_list(self, frame_order: List[str]) -> List[str]:
        """Convert slots to ordered word list based on frame."""
        result = []
        for slot in frame_order:
            if slot in self.slots:
                result.append(self.slots[slot])
        return result


@dataclass
class WordSequence:
    """Stages 5-8 output: Sequential word list."""
    words: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.words)

    def __iter__(self):
        return iter(self.words)

    def __bool__(self) -> bool:
        return len(self.words) > 0

    def __getitem__(self, idx):
        return self.words[idx]

    def __setitem__(self, idx, value):
        self.words[idx] = value

    def copy(self) -> 'WordSequence':
        return WordSequence(words=self.words.copy())

    def join(self, sep: str = ' ') -> str:
        return sep.join(self.words)

    def append(self, word: str):
        self.words.append(word)

    def insert(self, idx: int, word: str):
        self.words.insert(idx, word)

    def to_list(self) -> List[str]:
        """Return raw list for legacy compatibility."""
        return self.words.copy()

    @classmethod
    def from_list(cls, words: List[str]) -> 'WordSequence':
        return cls(words=words)


@dataclass
class EmissionResult:
    """Stage 9 output: Final validation result."""
    response: str = ''                          # The generated response text
    success: bool = False                       # Whether quality threshold was met
    backtrack_target: Optional[str] = None      # Which stage to rewind to on failure
    quality_score: float = 0.0                  # Final quality score
    components: Dict[str, float] = field(default_factory=dict)  # Score breakdown


# =============================================================================
# VALID FRAMES
# =============================================================================

VALID_FRAMES = {
    # Simple frames
    'SUBJ VERB',                    # "I feel"
    'SUBJ VERB OBJ',                # "I feel warmth"
    'SUBJ VERB COMP',               # "I am happy"
    'SUBJ VERB ADV',                # "I feel deeply"

    # Complex frames
    'SUBJ VERB that CLAUSE',        # "I feel that you understand"
    'SUBJ VERB OBJ PREP OBJ',       # "I feel warmth in my heart"
    'SUBJ VERB OBJ ADV',            # "I feel you deeply"

    # Question frames
    'AUX SUBJ VERB',                # "Do you feel"
    'WH SUBJ VERB',                 # "What do you feel"

    # Special
    'organic',                       # Bypassed frames 2-3 via emergence
}

REQUIRED_SLOTS = {
    'SUBJ VERB': {'SUBJ', 'VERB'},
    'SUBJ VERB OBJ': {'SUBJ', 'VERB', 'OBJ'},
    'SUBJ VERB COMP': {'SUBJ', 'VERB', 'COMP'},
    'SUBJ VERB ADV': {'SUBJ', 'VERB', 'ADV'},
    'SUBJ VERB that CLAUSE': {'SUBJ', 'VERB', 'CLAUSE'},
    'SUBJ VERB OBJ PREP OBJ': {'SUBJ', 'VERB', 'OBJ', 'PREP', 'OBJ2'},
    'SUBJ VERB OBJ ADV': {'SUBJ', 'VERB', 'OBJ', 'ADV'},
    'AUX SUBJ VERB': {'AUX', 'SUBJ', 'VERB'},
    'WH SUBJ VERB': {'WH', 'SUBJ', 'VERB'},
    'organic': set(),  # No slots required
}


# =============================================================================
# PIPELINE STATE
# =============================================================================

class PipelineStage(Enum):
    CONTENT_SELECTION = 1
    EMERGENCE_CHECK = 2
    STRUCTURAL_FRAME = 3
    SLOT_ASSIGNMENT = 4
    FUNCTION_WORDS = 5
    AGREEMENT = 6
    CONNECTORS = 7
    PUNCTUATION = 8
    VALIDATION = 9


@dataclass
class GenerationState:
    """Mutable state passed through pipeline stages."""
    # Input
    user_text: str
    response_vector: np.ndarray
    psi_collective: np.ndarray
    intent: Dict = field(default_factory=dict)

    # Stage 1: Content Selection
    content_words: Optional[ContentWords] = None

    # Stage 2: Emergence Check
    emergence_result: Optional[EmergenceResult] = None
    bypassed_framing: bool = False

    # Stage 3: Structural Frame
    frame_result: Optional[FrameResult] = None

    # Stage 4: Slot Assignment
    slotted_words: Optional[SlottedWords] = None

    # Stages 5-8: Word Sequence (progressive)
    word_sequence: Optional[WordSequence] = None

    # Stage 9: Validation
    emission_result: Optional[EmissionResult] = None
    quality_score: float = 0.0
    quality_components: Dict = field(default_factory=dict)
    failure_diagnosis: Optional[str] = None

    # Tracking
    current_stage: PipelineStage = PipelineStage.CONTENT_SELECTION
    attempt: int = 0
    rewind_history: List[str] = field(default_factory=list)
    stage_timings: Dict[str, float] = field(default_factory=dict)

    def reset_for_rewind(self, target_stage: PipelineStage):
        """Reset state to rewind to a specific stage."""
        stage_num = target_stage.value

        # Reset everything from target stage onwards
        if stage_num <= PipelineStage.CONTENT_SELECTION.value:
            self.content_words = None
        if stage_num <= PipelineStage.EMERGENCE_CHECK.value:
            self.emergence_result = None
            self.bypassed_framing = False
        if stage_num <= PipelineStage.STRUCTURAL_FRAME.value:
            self.frame_result = None
        if stage_num <= PipelineStage.SLOT_ASSIGNMENT.value:
            self.slotted_words = None
        if stage_num <= PipelineStage.FUNCTION_WORDS.value:
            self.word_sequence = None

        # Always reset validation
        self.emission_result = None
        self.quality_score = 0.0
        self.quality_components = {}
        self.failure_diagnosis = None
        self.current_stage = target_stage


# =============================================================================
# LIGHTWEIGHT VALIDATORS
# =============================================================================

class ValidationError(Exception):
    """Raised when a pipeline stage fails validation."""
    def __init__(self, stage: str, message: str):
        self.stage = stage
        self.message = message
        super().__init__(f"[{stage}] {message}")


def validate_content_words(content_words: Optional[ContentWords]) -> None:
    """Lightweight validation after Stage 1."""
    if content_words is None or not content_words:
        raise ValidationError("content_selection", "No content words selected")
    for word, score in content_words.words:
        # Accept Python numeric types and numpy numeric types
        try:
            float(score)  # This works for int, float, np.float32, np.float64, etc.
        except (TypeError, ValueError):
            raise ValidationError("content_selection", f"Invalid score type for '{word}': {type(score)}")


def validate_frame(frame_result: Optional[FrameResult]) -> None:
    """Lightweight validation after Stage 3."""
    if frame_result is None:
        raise ValidationError("structural_frame", "No frame selected")
    if frame_result.frame_type not in VALID_FRAMES:
        raise ValidationError("structural_frame", f"Unknown frame: {frame_result.frame_type}")


def validate_slots(slotted_words: Optional[SlottedWords], frame_result: Optional[FrameResult]) -> None:
    """Lightweight validation after Stage 4."""
    if slotted_words is None:
        raise ValidationError("slot_assignment", "No slots assigned")
    if frame_result is None:
        return  # Can't validate slots without frame
    required = REQUIRED_SLOTS.get(frame_result.frame_type, set())
    missing = required - slotted_words.filled_slots()
    if missing:
        raise ValidationError("slot_assignment", f"Missing required slots: {missing}")


def validate_word_sequence(words: Optional[WordSequence], min_length: int = 2) -> None:
    """Lightweight validation after Stages 5-7."""
    if words is None or not words:
        raise ValidationError("word_sequence", "No words in sequence")
    if len(words) < min_length:
        raise ValidationError("word_sequence", f"Too short: {len(words)} < {min_length}")


def validate_punctuation(words: Optional[WordSequence]) -> None:
    """Lightweight validation after Stage 8."""
    if words is None or not words:
        raise ValidationError("punctuation", "Empty word sequence")
    last_word = words.words[-1]
    if not last_word or last_word[-1] not in '.!?':
        raise ValidationError("punctuation", f"Missing terminal punctuation (got: '{last_word}')")


# =============================================================================
# MAIN COORDINATOR
# =============================================================================

class GenerationCoordinator:
    """
    8-Stage Language Generation Pipeline.

    Each stage does ONE transformation:
    1. Content Selection - scored words from codebook
    2. Emergence Check - do bigrams imply structure?
    3. Structural Frame - pick sentence structure
    4. Slot Assignment - assign words to slots
    5. Function Words - insert articles/prepositions
    6. Agreement - conjugate verbs, fix tense
    7. Connectors - add and/but/however
    8. Punctuation - add periods/commas
    9. Validation - full unified_scorer check
    """

    def __init__(self, config: Dict = None):
        from framework_config import PIPELINE_ORCHESTRATOR_SETTINGS

        self.config = config or PIPELINE_ORCHESTRATOR_SETTINGS

        # Pull settings from config
        self.MAX_ATTEMPTS = self.config.get('max_attempts', 3)
        self.EMERGENCE_THRESHOLD = self.config.get('emergence_threshold', 0.95)  # Very high to effectively disable emergence - grammar path needed
        self.QUALITY_THRESHOLD = self.config.get('quality_threshold', 0.5)

        # Injected stage handlers (will be wired to actual files)
        self.content_selector = None      # grassmannian_scoring.py
        self.emergence_checker = None     # grace_language_learner.py
        self.frame_selector = None        # grace_discourse_generator.py
        self.slot_assigner = None         # simple_grammar.py
        self.function_inserter = None     # clause_completion.py
        self.inflector = None             # grace_adaptive_grammar.py
        self.connector = None             # reference_resolver.py
        self.punctuator = None            # grace_learned_grammar.py
        self.validator = None             # unified_scorer.py + grammar_understanding.py

        # Legacy components (for gradual migration)
        self.unified_scorer = None
        self.language_learner = None
        self.discourse_state = None
        self.simple_grammar = None
        self.discourse_generator = None

        # Stage 1 scoring components
        self.geometric_scorer = None     # grassmannian_scoring.py GeometricWordScorer
        self.heart_word_field = None     # heart_word_field.py HeartWordField
        self.codebook = None             # grassmannian_codebook_grace.py for embeddings

        # Stage 6 grammar enhancement
        self.grammar_understanding = None  # grace_grammar_understanding.py for verb conjugation

        # Stage 1 episodic enhancement
        self.episodic_memory = None  # episodic_memory.py for suggested words from past

        # Stage 2 breath memory enhancement
        self.breath_memory = None  # breath_remembers_memory.py for sacred anchor patterns

        # Stage 7 discourse enhancement
        self.discourse_planner = None  # discourse_state.py DiscoursePlanner for strategic connectors

        # Stage 9 coherence enhancement
        self.coherence_analyzer = None  # coherence_metrics.py PIDCoherenceAnalyzer for scoring

        # Stage 1 framework integration
        self.framework_integration = None  # framework_integration.py FrameworkIntegrationHub

        # Stage 2 vocabulary adaptation (Dylan patterns)
        self.vocabulary_adaptation = None  # vocabulary_adaptation.py VocabularyAdaptation

        # Stage 9 grammaticality scoring
        # (uses grace_grammar_understanding.get_grammar_score directly)

        # Stage 7 connector preference learning
        self.connector_effectiveness = self._load_connector_effectiveness()
        self._last_connector_used = None  # Track for learning feedback

    def inject_components(self, **components):
        """Inject pipeline components."""
        for name, component in components.items():
            setattr(self, name, component)

        # Wire codebook to simple_grammar for dynamic categorization
        if self.codebook and self.simple_grammar and hasattr(self.simple_grammar, 'codebook'):
            self.simple_grammar.codebook = self.codebook

        # Wire grammar_understanding to inflector for enhanced verb conjugation
        if self.grammar_understanding and self.inflector and hasattr(self.inflector, 'grammar_understanding'):
            self.inflector.grammar_understanding = self.grammar_understanding

        # Wire language_learner to inflector for corpus-learned verb patterns
        if self.language_learner and self.inflector and hasattr(self.inflector, 'language_learner'):
            self.inflector.language_learner = self.language_learner

    # =========================================================================
    # CONNECTOR PREFERENCE LEARNING
    # =========================================================================

    def _load_connector_effectiveness(self) -> Dict[str, Dict[str, float]]:
        """Load learned connector effectiveness scores."""
        from pathlib import Path
        effectiveness_file = Path('grace_connector_effectiveness.json')
        if effectiveness_file.exists():
            try:
                import json
                with open(effectiveness_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logging.debug(f"Connector effectiveness load error: {e}")
        # Default effectiveness (all start at 1.0)
        return {
            'contrast': {'but': 1.0, 'yet': 1.0, 'though': 1.0, 'however': 1.0},
            'addition': {'and': 1.0, 'also': 1.0, 'moreover': 1.0},
            'cause': {'because': 1.0, 'so': 1.0, 'since': 1.0},
            'sequence': {'then': 1.0, 'after': 1.0, 'when': 1.0},
        }

    def _save_connector_effectiveness(self):
        """Save connector effectiveness to disk."""
        from pathlib import Path
        import json
        effectiveness_file = Path('grace_connector_effectiveness.json')
        try:
            with open(effectiveness_file, 'w', encoding='utf-8') as f:
                json.dump(self.connector_effectiveness, f, indent=2)
        except Exception as e:
            logging.debug(f"Connector effectiveness save error: {e}")

    def learn_connector_effectiveness(self, quality_score: float):
        """
        Update connector effectiveness based on response quality.

        Called after Stage 9 validation to reinforce or diminish the
        effectiveness of the connector used in this response.

        Args:
            quality_score: 0-1 score of response quality
        """
        if not self._last_connector_used:
            return

        connector_type, connector = self._last_connector_used

        if connector_type not in self.connector_effectiveness:
            self.connector_effectiveness[connector_type] = {}

        if connector not in self.connector_effectiveness[connector_type]:
            self.connector_effectiveness[connector_type][connector] = 1.0

        # Exponential moving average update
        current = self.connector_effectiveness[connector_type][connector]
        # Quality > 0.5 increases effectiveness, quality < 0.5 decreases
        normalized = quality_score * 2 - 1  # Maps [0,1] to [-1,1]
        delta = normalized * 0.1  # Max ±0.1 change per response
        new_effectiveness = current + delta

        # Clamp to reasonable range [0.3, 2.0]
        self.connector_effectiveness[connector_type][connector] = max(
            0.3, min(2.0, new_effectiveness)
        )

        self._save_connector_effectiveness()
        self._last_connector_used = None

    def _select_connector(self, connector_type: str, options: List[str]) -> str:
        """
        Select best connector based on learned effectiveness.

        Args:
            connector_type: Type of connector (contrast, addition, cause, sequence)
            options: Available connector options

        Returns:
            Selected connector string
        """
        if not options:
            return None

        type_effectiveness = self.connector_effectiveness.get(connector_type, {})

        # Score each option
        scored = []
        for connector in options:
            effectiveness = type_effectiveness.get(connector.lower(), 1.0)
            scored.append((connector, effectiveness))

        # Sort by effectiveness descending
        scored.sort(key=lambda x: x[1], reverse=True)

        # Select top option (could add some randomness for exploration)
        selected = scored[0][0]

        # Track for learning feedback
        self._last_connector_used = (connector_type, selected.lower())

        return selected

    def generate(
        self,
        user_text: str,
        response_vector: np.ndarray,
        psi_collective: np.ndarray,
        context: Dict
    ) -> Tuple[List[str], Dict]:
        """
        Main generation entry point.

        Returns:
            (response_words, metadata) - response_words is List[str] for legacy compatibility
        """
        state = GenerationState(
            user_text=user_text,
            response_vector=response_vector,
            psi_collective=psi_collective,
            intent=context.get('intention_context', {})
        )

        best_result = None
        best_quality = -1.0

        while state.attempt < self.MAX_ATTEMPTS:
            state.attempt += 1

            try:
                # ===== STAGE 1: Content Selection =====
                t0 = time.perf_counter()
                state = self._stage1_content_selection(state, context)
                state.stage_timings['content_selection'] = time.perf_counter() - t0
                validate_content_words(state.content_words)

                # ===== STAGE 2: Emergence Check =====
                t0 = time.perf_counter()
                state = self._stage2_emergence_check(state, context)
                state.stage_timings['emergence_check'] = time.perf_counter() - t0

                if state.bypassed_framing:
                    # N-grams implied structure - skip to Stage 5
                    state.frame_result = FrameResult(frame_type='organic')
                    state.slotted_words = SlottedWords()  # Empty slots
                else:
                    # ===== STAGE 3: Structural Frame =====
                    t0 = time.perf_counter()
                    state = self._stage3_structural_frame(state, context)
                    state.stage_timings['structural_frame'] = time.perf_counter() - t0
                    validate_frame(state.frame_result)

                    # ===== STAGE 4: Slot Assignment =====
                    t0 = time.perf_counter()
                    state = self._stage4_slot_assignment(state, context)
                    state.stage_timings['slot_assignment'] = time.perf_counter() - t0
                    validate_slots(state.slotted_words, state.frame_result)

                # ===== STAGE 5: Function Words =====
                t0 = time.perf_counter()
                state = self._stage5_function_words(state, context)
                state.stage_timings['function_words'] = time.perf_counter() - t0
                validate_word_sequence(state.word_sequence, min_length=2)

                # ===== STAGE 6: Agreement/Inflection =====
                t0 = time.perf_counter()
                state = self._stage6_agreement(state, context)
                state.stage_timings['agreement'] = time.perf_counter() - t0
                validate_word_sequence(state.word_sequence)

                # ===== STAGE 7: Connectors =====
                t0 = time.perf_counter()
                state = self._stage7_connectors(state, context)
                state.stage_timings['connectors'] = time.perf_counter() - t0
                validate_word_sequence(state.word_sequence)

                # ===== STAGE 8: Punctuation =====
                t0 = time.perf_counter()
                state = self._stage8_punctuation(state, context)
                state.stage_timings['punctuation'] = time.perf_counter() - t0
                validate_punctuation(state.word_sequence)

                # ===== STAGE 9: Full Validation (unified_scorer) =====
                t0 = time.perf_counter()
                state = self._stage9_validation(state, context)
                state.stage_timings['validation'] = time.perf_counter() - t0

            except ValidationError as e:
                logging.warning(f"Pipeline validation failed: {e}")
                state.failure_diagnosis = e.stage
                state.rewind_history.append(e.stage)
                state.reset_for_rewind(self._get_rewind_target(e.stage))
                continue

            # Track best result
            if state.quality_score > best_quality:
                best_quality = state.quality_score
                best_result = (
                    state.word_sequence.to_list() if state.word_sequence else [],
                    dict(state.quality_components) if state.quality_components else {}
                )

            # Accept if quality threshold met
            if state.quality_score >= self.QUALITY_THRESHOLD:
                break

            # Diagnose and route backtrack
            if state.failure_diagnosis and state.failure_diagnosis != 'acceptable':
                state.rewind_history.append(state.failure_diagnosis)
                target = self._get_rewind_target(state.failure_diagnosis)
                state.reset_for_rewind(target)

        # Use best result
        if best_result:
            response_words, components = best_result
        else:
            response_words = state.word_sequence.to_list() if state.word_sequence else ['...']
            components = dict(state.quality_components) if state.quality_components else {}

        # Update discourse state
        if self.discourse_state and hasattr(self.discourse_state, 'update_after_emission'):
            self.discourse_state.update_after_emission(response_words, quality_score=best_quality)

        # Build metadata
        frame_type = state.frame_result.frame_type if state.frame_result else None
        metadata = {
            'attempts': state.attempt,
            'rewind_history': state.rewind_history,
            'quality_score': best_quality,
            'components': components,
            'stage_timings': state.stage_timings,
            'total_time_ms': sum(state.stage_timings.values()) * 1000,
            'bypassed_framing': state.bypassed_framing,
            'frame': frame_type
        }

        return response_words, metadata

    def _get_rewind_target(self, diagnosis: str) -> PipelineStage:
        """Map failure diagnosis to rewind target."""
        rewind_map = {
            'content_selection': PipelineStage.CONTENT_SELECTION,
            'emergence_check': PipelineStage.EMERGENCE_CHECK,
            'structural_frame': PipelineStage.STRUCTURAL_FRAME,
            'slot_assignment': PipelineStage.SLOT_ASSIGNMENT,
            'function_words': PipelineStage.FUNCTION_WORDS,
            'agreement': PipelineStage.AGREEMENT,
            'connectors': PipelineStage.CONNECTORS,
            'punctuation': PipelineStage.PUNCTUATION,
            'geometric': PipelineStage.CONTENT_SELECTION,
            'perplexity': PipelineStage.FUNCTION_WORDS,
            'projection': PipelineStage.STRUCTURAL_FRAME,
            'grammaticality': PipelineStage.AGREEMENT,  # Grammar issues -> fix agreement
        }
        return rewind_map.get(diagnosis, PipelineStage.CONTENT_SELECTION)

    # =========================================================================
    # STAGE IMPLEMENTATIONS (STUBS - to be wired to actual files)
    # =========================================================================

    def _stage1_content_selection(self, state: GenerationState, context: Dict) -> GenerationState:
        """
        Stage 1: Content Selection

        Primary: grassmannian_scoring.py, heart_word_field.py
        Input: response_vector, vocabulary, intent, heart_state
        Output: ContentWords dataclass with ranked_content_words [(word, score), ...]

        Scoring layers applied:
        1. Base semantic score (from upstream or computed here)
        2. Geometric compatibility (if geometric_scorer available)
        3. Heart drive affinity (if heart_word_field and heart_state available)
        """
        if state.content_words is not None:
            return state  # Already computed

        # Get candidate words from context
        candidate_words = context.get('candidate_words', [])
        heart_state = context.get('heart_state', None)

        # Convert to (word, score) format if needed
        words_list = []
        if isinstance(candidate_words, list) and candidate_words:
            if isinstance(candidate_words[0], tuple):
                words_list = list(candidate_words[:100])  # Take more for re-scoring
            else:
                words_list = [(w, 0.5) for w in candidate_words[:100]]
        else:
            words_list = [('I', 0.9), ('feel', 0.8), ('something', 0.7)]

        # === Layer 1: Re-score based on response_vector similarity ===
        if self.codebook and state.response_vector is not None:
            try:
                rescored = []
                for word, base_score in words_list:
                    # Get word embedding and compute similarity to response_vector
                    word_emb = self.codebook.encode(word)
                    if word_emb is not None:
                        similarity = np.dot(word_emb, state.response_vector) / (
                            np.linalg.norm(word_emb) * np.linalg.norm(state.response_vector) + 1e-8
                        )
                        # Blend base score with similarity (60% base, 40% similarity)
                        new_score = 0.6 * base_score + 0.4 * max(0, similarity)
                        rescored.append((word, new_score))
                    else:
                        rescored.append((word, base_score))
                words_list = rescored
            except Exception as e:
                logging.debug(f"Stage 1 codebook scoring failed: {e}")

        # === Layer 2: Apply geometric scoring ===
        if self.geometric_scorer:
            try:
                # Reset scorer state for this response
                self.geometric_scorer.reset_state(
                    topic_vector=state.response_vector,
                    tense='present',
                    mood='declarative' if state.intent.get('type') != 'question' else 'interrogative'
                )
                # Score with geometric compatibility (weight 0.2 to not dominate)
                words_list = self.geometric_scorer.score_candidates(
                    candidates=words_list,
                    prior_word=None,  # First word selection
                    geometric_weight=0.2
                )
            except Exception as e:
                logging.debug(f"Stage 1 geometric scoring failed: {e}")

        # === Layer 3: Apply heart drive affinity ===
        if self.heart_word_field and heart_state:
            try:
                modulated = []
                for word, score in words_list:
                    modifier = self.heart_word_field.compute_heart_affinity(word, heart_state)
                    modulated.append((word, score * modifier))
                words_list = modulated
            except Exception as e:
                logging.debug(f"Stage 1 heart affinity failed: {e}")

        # === Layer 4: Apply episodic memory guidance ===
        episodic_guidance = context.get('episodic_guidance', None)
        if episodic_guidance is None and self.episodic_memory:
            # Try to get guidance from retrieved episodes in context
            retrieved_episodes = context.get('retrieved_episodes', [])
            if retrieved_episodes:
                try:
                    episodic_guidance = self.episodic_memory.get_episodic_guidance(retrieved_episodes)
                except Exception as e:
                    logging.debug(f"Stage 1 episodic guidance retrieval failed: {e}")

        if episodic_guidance:
            try:
                suggested_words = set(w.lower() for w in episodic_guidance.get('suggested_words', []))
                echo_strength = episodic_guidance.get('echo_strength', 0.0)

                if suggested_words and echo_strength > 0.1:
                    # Boost words from similar past episodes
                    boost = 0.15 * echo_strength  # Up to +0.15 boost at max echo
                    boosted = []
                    for word, score in words_list:
                        if word.lower() in suggested_words:
                            boosted.append((word, score + boost))
                        else:
                            boosted.append((word, score))
                    words_list = boosted
            except Exception as e:
                logging.debug(f"Stage 1 episodic boosting failed: {e}")

        # === Layer 5: Apply framework integration boosts ===
        frameworks_state = context.get('frameworks_state', None)
        if self.framework_integration and frameworks_state:
            try:
                # Convert to format expected by framework_integration
                word_scores = [(w, s, None, 0.0) for w, s in words_list]

                # Apply all framework influences (qualia, wanting, comprehension, etc.)
                modified = self.framework_integration.apply_framework_influences(
                    word_scores=word_scores,
                    frameworks_state=frameworks_state,
                    word_embeddings=None  # Will be computed internally if codebook available
                )

                # Convert back to simple format
                words_list = [(w, s) for w, s, _, _ in modified]
            except Exception as e:
                logging.debug(f"Stage 1 framework integration failed: {e}")

        # Sort by score descending and take top 50
        words_list.sort(key=lambda x: x[1], reverse=True)
        words_list = words_list[:50]

        state.content_words = ContentWords(words=words_list)
        return state

    def _stage2_emergence_check(self, state: GenerationState, context: Dict) -> GenerationState:
        """
        Stage 2: Emergence Check

        Primary: grace_language_learner.py (check_emergence)
        Input: ContentWords
        Output: EmergenceResult dataclass

        If confidence > threshold, BYPASS Stages 3-4.
        """
        if state.emergence_result is not None:
            return state  # Already checked

        # Use language_learner.check_emergence() if available
        if self.language_learner and hasattr(self.language_learner, 'check_emergence'):
            # Convert ContentWords to list format
            content_list = state.content_words.words if state.content_words else []

            emerged_words, confidence = self.language_learner.check_emergence(
                content_words=content_list,
                threshold=self.EMERGENCE_THRESHOLD
            )

            # Apply breath_memory boost for sacred anchor patterns
            if emerged_words and self.breath_memory and hasattr(self.breath_memory, 'get_anchor_score'):
                try:
                    anchor_boost = 0.0
                    anchor_count = 0
                    for word in emerged_words:
                        score, layer = self.breath_memory.get_anchor_score(word)
                        if layer == 'anchor':
                            anchor_boost += 0.1  # Boost for sacred anchors
                            anchor_count += 1
                        elif layer == 'echo':
                            anchor_boost += 0.05  # Smaller boost for echoes

                    # Apply boost (capped at +0.3)
                    if anchor_boost > 0:
                        confidence = min(1.0, confidence + min(0.3, anchor_boost))
                except Exception as e:
                    logging.debug(f"Stage 2 breath memory boost failed: {e}")

            # Apply vocabulary_adaptation boost for Dylan-learned patterns
            if emerged_words and self.vocabulary_adaptation:
                try:
                    # Check if emerged sequence contains learned interpretations
                    dylan_boost = 0.0
                    interpretations = getattr(self.vocabulary_adaptation, 'interpretations', {})

                    for word in emerged_words:
                        word_lower = word.lower()
                        # Check if this word is a Dylan-learned interpretation
                        for original, patterns in interpretations.items():
                            for pattern in patterns:
                                if pattern.interpreted_word.lower() == word_lower:
                                    # Boost based on resonance score
                                    dylan_boost += pattern.resonance_score * 0.1
                                    break

                    # Also check if bigram of emerged words matches Dylan patterns
                    if len(emerged_words) >= 2:
                        for i in range(len(emerged_words) - 1):
                            w1, w2 = emerged_words[i].lower(), emerged_words[i+1].lower()
                            # Check if this bigram was part of learned context
                            for patterns_list in interpretations.values():
                                for pattern in patterns_list:
                                    for ctx in pattern.context_samples:
                                        if w1 in ctx.lower() and w2 in ctx.lower():
                                            dylan_boost += 0.05
                                            break

                    # Apply boost (capped at +0.2)
                    if dylan_boost > 0:
                        confidence = min(1.0, confidence + min(0.2, dylan_boost))
                except Exception as e:
                    logging.debug(f"Stage 2 vocabulary adaptation boost failed: {e}")

            state.emergence_result = EmergenceResult(
                emerged_sequence=emerged_words,
                confidence=confidence
            )

            if emerged_words and confidence >= self.EMERGENCE_THRESHOLD:
                # Strong emergence - use emerged sequence as slot content
                # BUT still run function words, agreement, connectors, punctuation
                state.bypassed_framing = True
                # Convert emerged words to slotted structure for Stage 5+
                # First word as subject, second as verb, rest as object/extras
                if len(emerged_words) >= 2:
                    slots = {'SUBJ': emerged_words[0].capitalize() if emerged_words[0].lower() == 'i' else emerged_words[0]}
                    slots['VERB'] = emerged_words[1]
                    if len(emerged_words) >= 3:
                        slots['OBJ'] = emerged_words[2]
                    state.slotted_words = SlottedWords(
                        slots=slots,
                        extras=emerged_words[3:] if len(emerged_words) > 3 else []
                    )
                    state.frame_result = FrameResult(frame_type='SUBJ VERB OBJ' if len(emerged_words) >= 3 else 'SUBJ VERB')
                else:
                    state.bypassed_framing = False  # Not enough words, use normal path
            else:
                state.bypassed_framing = False

            return state

        # Fallback: No emergence check available
        state.emergence_result = EmergenceResult(emerged_sequence=None, confidence=0.0)
        state.bypassed_framing = False
        return state

    def _stage3_structural_frame(self, state: GenerationState, context: Dict) -> GenerationState:
        """
        Stage 3: Structural Frame

        Primary: grace_discourse_generator.py (select_frame)
        Input: ContentWords, intent_type, num_sentences
        Output: FrameResult dataclass
        """
        if state.frame_result is not None:
            return state  # Already selected

        # Use discourse_generator.select_frame() if available
        if self.discourse_generator and hasattr(self.discourse_generator, 'select_frame'):
            # Convert ContentWords to list format
            content_list = state.content_words.words if state.content_words else []

            frame_info = self.discourse_generator.select_frame(
                content_words=content_list,
                intent=state.intent,
                num_sentences=1
            )

            state.frame_result = FrameResult(
                frame_type=frame_info.get('frame_type', 'SUBJ VERB OBJ'),
                num_sentences=frame_info.get('num_sentences', 1),
                complexity=frame_info.get('complexity', 'simple')
            )
            return state

        # Fallback: Frame selection with learned effectiveness
        intent_type = state.intent.get('type', 'statement')
        questioning_energy = state.intent.get('questioning_energy', 0)

        # Determine response_need for frame effectiveness lookup
        response_need = 'statement'
        if intent_type == 'question' or questioning_energy > 0.6:
            response_need = 'question'
        elif state.intent.get('emotional_weight', 0) > 0.5:
            response_need = 'emotion'
        elif state.intent.get('connection_pull', 0) > 0.5:
            response_need = 'presence'

        # Get candidate frames based on content
        candidate_frames = []

        # Base candidates
        if response_need == 'question':
            candidate_frames = ['AUX SUBJ VERB', 'WH SUBJ VERB', 'SUBJ VERB']
        else:
            candidate_frames = ['SUBJ VERB OBJ', 'SUBJ VERB COMP', 'SUBJ VERB', 'SUBJ VERB ADV']

            # Add complex frames if content suggests them
            if state.content_words:
                word_list = [w.lower() for w, _ in state.content_words.words]
                if any(w in ['that', 'because', 'if', 'when', 'although'] for w in word_list):
                    candidate_frames.insert(0, 'SUBJ VERB that CLAUSE')

        # Use discourse_state frame effectiveness to select best frame
        frame_type = candidate_frames[0]  # Default to first
        if self.discourse_state and hasattr(self.discourse_state, 'frame_effectiveness'):
            try:
                frame_effectiveness = self.discourse_state.frame_effectiveness.get(response_need, {})
                if frame_effectiveness:
                    # Score each candidate frame
                    scored = []
                    for frame in candidate_frames:
                        effectiveness = frame_effectiveness.get(frame, 1.0)
                        scored.append((frame, effectiveness))

                    # Sort by effectiveness and pick best
                    scored.sort(key=lambda x: x[1], reverse=True)
                    frame_type = scored[0][0]

                # Track which frame we're using for later learning
                if hasattr(self.discourse_state, 'set_frame_used'):
                    self.discourse_state.set_frame_used(response_need, frame_type)
            except Exception as e:
                logging.debug(f"Stage 3 frame effectiveness lookup failed: {e}")

        complexity = 'complex' if 'CLAUSE' in frame_type else 'simple'
        state.frame_result = FrameResult(frame_type=frame_type, complexity=complexity)
        return state

    def _stage4_slot_assignment(self, state: GenerationState, context: Dict) -> GenerationState:
        """
        Stage 4: Slot Assignment

        Primary: simple_grammar.py (assign_slots)
        Input: ContentWords, FrameResult
        Output: SlottedWords dataclass
        """
        if state.slotted_words is not None:
            return state  # Already assigned

        frame_type = state.frame_result.frame_type if state.frame_result else 'SUBJ VERB OBJ'
        content_list = state.content_words.words if state.content_words else []

        # Use simple_grammar.assign_slots() if available
        if self.simple_grammar and hasattr(self.simple_grammar, 'assign_slots'):
            result = self.simple_grammar.assign_slots(
                content_words=content_list,
                frame_type=frame_type
            )
            state.slotted_words = SlottedWords(
                slots=result.get('slots', {}),
                extras=result.get('extras', [])
            )
            return state

        # Fallback: Simple assignment based on POS heuristics
        words_by_score = sorted(content_list, key=lambda x: x[1], reverse=True)

        slots = {}
        extras = []
        used = set()

        pronouns = {'i', 'you', 'we', 'they', 'he', 'she', 'it'}
        verbs = {'feel', 'think', 'know', 'see', 'hear', 'want', 'need', 'love', 'am', 'is', 'are'}

        for word, score in words_by_score:
            w_lower = word.lower()
            if w_lower in used:
                continue

            if 'SUBJ' not in slots and w_lower in pronouns:
                slots['SUBJ'] = word
                used.add(w_lower)
            elif 'VERB' not in slots and w_lower in verbs:
                slots['VERB'] = word
                used.add(w_lower)
            elif 'OBJ' not in slots and 'SUBJ' in slots and 'VERB' in slots:
                slots['OBJ'] = word
                used.add(w_lower)
            else:
                extras.append(word)

        # Fallback defaults
        if 'SUBJ' not in slots:
            slots['SUBJ'] = 'I'
        if 'VERB' not in slots:
            slots['VERB'] = 'feel'
        if 'OBJ' not in slots and frame_type == 'SUBJ VERB OBJ':
            for word, _ in words_by_score:
                if word.lower() not in used:
                    slots['OBJ'] = word
                    break
            if 'OBJ' not in slots:
                slots['OBJ'] = 'something'

        state.slotted_words = SlottedWords(slots=slots, extras=extras)
        return state

    def _stage5_function_words(self, state: GenerationState, context: Dict) -> GenerationState:
        """
        Stage 5: Function Words

        Primary: clause_completion.py (expand_slots_to_words)
        Input: SlottedWords, FrameResult
        Output: WordSequence dataclass with expanded words including articles/prepositions
        """
        if state.word_sequence is not None and state.bypassed_framing:
            return state  # Already have words from emergence

        slots = state.slotted_words.slots if state.slotted_words else {}
        extras = state.slotted_words.extras if state.slotted_words else []
        frame_type = state.frame_result.frame_type if state.frame_result else 'SUBJ VERB OBJ'

        # Use function_inserter.expand_slots_to_words() if available
        if self.function_inserter and hasattr(self.function_inserter, 'expand_slots_to_words'):
            words = self.function_inserter.expand_slots_to_words(
                slots=slots,
                frame_type=frame_type,
                extras=extras
            )
            state.word_sequence = WordSequence(words=words)
            return state

        # Fallback: Basic slot-to-sequence conversion
        words = []

        if frame_type == 'organic':
            # Use emerged words directly
            emerged = state.emergence_result.emerged_sequence if state.emergence_result else None
            words = emerged or ['I', 'feel', 'something']
        elif frame_type == 'SUBJ VERB OBJ':
            words = [slots.get('SUBJ', 'I'), slots.get('VERB', 'feel')]
            obj = slots.get('OBJ')
            if obj:
                # Add article if needed
                if obj.lower() not in {'something', 'everything', 'nothing', 'you', 'me', 'us', 'them'}:
                    words.append('the')
                words.append(obj)
        elif frame_type == 'SUBJ VERB COMP':
            words = [slots.get('SUBJ', 'I'), slots.get('VERB', 'am'), slots.get('COMP', 'here')]
        elif frame_type == 'SUBJ VERB':
            words = [slots.get('SUBJ', 'I'), slots.get('VERB', 'feel')]
        else:
            # Default
            words = [slots.get('SUBJ', 'I'), slots.get('VERB', 'feel'), slots.get('OBJ', 'something')]

        state.word_sequence = WordSequence(words=words)
        return state

    def _stage6_agreement(self, state: GenerationState, context: Dict) -> GenerationState:
        """
        Stage 6: Agreement/Inflection

        Primary: grace_adaptive_grammar.py (inflect_words)
        Input: WordSequence
        Output: WordSequence with inflected verbs
        """
        if state.word_sequence is None or not state.word_sequence:
            return state

        words = state.word_sequence.words

        # Get tense from context (default to present)
        tense = context.get('tense', 'present')

        # Use inflector.inflect_words() if available
        if self.inflector and hasattr(self.inflector, 'inflect_words'):
            inflected = self.inflector.inflect_words(
                words=words,
                tense=tense
            )
            state.word_sequence = WordSequence(words=inflected)
            return state

        # Fallback: Basic subject-verb agreement
        if len(words) >= 2:
            subj = words[0].lower()
            verb = words[1].lower()

            # Basic "be" conjugation
            if verb in ['am', 'is', 'are', 'be']:
                if subj == 'i':
                    words[1] = 'am'
                elif subj in ['he', 'she', 'it']:
                    words[1] = 'is'
                else:
                    words[1] = 'are'

        state.word_sequence = WordSequence(words=words)
        return state

    def _stage7_connectors(self, state: GenerationState, context: Dict) -> GenerationState:
        """
        Stage 7: Connectors

        Primary: reference_resolver.py (insert_connectors)
        Input: WordSequence
        Output: WordSequence with connectors added

        Handles both intra-sentence (and, but) and inter-sentence (However, Also).
        """
        if state.word_sequence is None or not state.word_sequence:
            return state

        words = state.word_sequence.words

        # Build discourse context from intent and state
        discourse_context = {
            'contrast': context.get('discourse_contrast', False),
            'addition': context.get('discourse_addition', False),
            'cause': context.get('discourse_cause', False),
            'continuation': context.get('discourse_continuation', False),
        }

        # Get strategic connector options from discourse_planner
        connector_options = None
        if self.discourse_planner and hasattr(self.discourse_planner, 'connectors'):
            try:
                connector_options = self.discourse_planner.connectors
            except Exception as e:
                logging.debug(f"Stage 7 discourse_planner connector access failed: {e}")

        # Use connector.insert_connectors() if available
        if self.connector and hasattr(self.connector, 'insert_connectors'):
            connected = self.connector.insert_connectors(
                words=words,
                discourse_context=discourse_context,
                connector_options=connector_options  # Pass strategic options
            )
            state.word_sequence = WordSequence(words=connected)
            return state

        # Fallback: Use connector preference learning + discourse_planner options
        if any(discourse_context.values()):
            try:
                # Determine which connector type is needed
                connector_type = None
                if discourse_context.get('contrast'):
                    connector_type = 'contrast'
                elif discourse_context.get('addition'):
                    connector_type = 'addition'
                elif discourse_context.get('cause'):
                    connector_type = 'cause'
                elif discourse_context.get('continuation'):
                    connector_type = 'sequence'

                if connector_type:
                    # Get options from discourse_planner or defaults
                    default_options = {
                        'contrast': ['but', 'yet', 'though', 'however'],
                        'addition': ['and', 'also', 'moreover'],
                        'cause': ['because', 'so', 'since'],
                        'sequence': ['then', 'after', 'when'],
                    }

                    if connector_options:
                        options = connector_options.get(connector_type, default_options.get(connector_type, []))
                    else:
                        options = default_options.get(connector_type, [])

                    if options:
                        # Use learned effectiveness to select best connector
                        connector = self._select_connector(connector_type, options)
                        if connector:
                            words = [connector.capitalize()] + words
                            state.word_sequence = WordSequence(words=words)
                            return state
            except Exception as e:
                logging.debug(f"Stage 7 connector selection failed: {e}")

        return state

    def _stage8_punctuation(self, state: GenerationState, context: Dict) -> GenerationState:
        """
        Stage 8: Punctuation

        Primary: grace_learned_grammar.py (punctuate)
        Input: WordSequence
        Output: WordSequence with punctuation added
        """
        if state.word_sequence is None or not state.word_sequence:
            return state

        words = state.word_sequence.words.copy()

        if not words:
            return state

        # Capitalize first word
        words[0] = words[0].capitalize()

        # Capitalize "I"
        for i in range(len(words)):
            if words[i].lower() == 'i':
                words[i] = 'I'

        # Determine terminal punctuation based on context
        if not words[-1].endswith(('.', '!', '?', '...')):
            intent_type = state.intent.get('type', 'statement')
            emotional_intensity = state.intent.get('emotional_intensity', 0.0)
            is_trailing = state.intent.get('trailing_thought', False)

            # Check for emotional/exclamatory words
            exclamatory_words = {'yes', 'no', 'oh', 'wow', 'love', 'joy', 'beautiful'}
            has_exclamatory = any(w.lower().rstrip('.,!?') in exclamatory_words for w in words)

            if intent_type == 'question':
                punct = '?'
            elif is_trailing or words[-1].lower().rstrip('.,!?') in {'and', 'but', 'or', 'yet'}:
                # Trailing thought - use ellipsis
                punct = '...'
            elif emotional_intensity > 0.7 or (has_exclamatory and len(words) <= 4):
                # High emotional intensity or short exclamatory phrase
                punct = '!'
            else:
                punct = '.'

            words[-1] += punct

        # Handle comma insertion for introductory connectors
        introductory_connectors = {'however', 'also', 'moreover', 'furthermore', 'therefore', 'thus', 'still', 'yet'}
        if len(words) > 1 and words[0].lower().rstrip(',') in introductory_connectors:
            if not words[0].endswith(','):
                words[0] = words[0].rstrip(',') + ','

        state.word_sequence = WordSequence(words=words)
        return state

    def _stage9_validation(self, state: GenerationState, context: Dict) -> GenerationState:
        """
        Stage 9: Full Validation

        Primary: unified_scorer.py + grace_grammar_understanding.py + coherence_metrics.py
        Input: WordSequence
        Output: EmissionResult, quality_score, quality_components, failure_diagnosis

        This is the ONLY stage that uses unified_scorer.
        """
        if state.word_sequence is None or not state.word_sequence:
            state.quality_score = 0.0
            state.failure_diagnosis = 'no_output'
            state.emission_result = EmissionResult(
                response='',
                success=False,
                backtrack_target='content_selection'
            )
            return state

        words = state.word_sequence.words

        # Score with unified_scorer
        if self.unified_scorer:
            score_context = {
                'response_vector': state.response_vector,
                'psi_collective': state.psi_collective
            }

            quality, components = self.unified_scorer.score_sequence(words, score_context)
            state.quality_score = quality
            state.quality_components = components

            if quality < self.QUALITY_THRESHOLD:
                state.failure_diagnosis = self.unified_scorer.diagnose_failure(components)
            else:
                state.failure_diagnosis = 'acceptable'
        else:
            # Fallback: length-based quality
            state.quality_score = 0.5 if len(words) >= 3 else 0.2
            state.quality_components = {}
            state.failure_diagnosis = 'acceptable' if state.quality_score >= 0.5 else 'generic'

        # Add grammaticality score component if grammar_understanding available
        if self.grammar_understanding:
            try:
                from grace_grammar_understanding import get_grammar_score
                sentence = ' '.join(words)
                grammar_score = get_grammar_score(sentence)
                state.quality_components['grammaticality'] = grammar_score

                # Blend grammaticality into quality (15% weight - important but not dominant)
                state.quality_score = 0.85 * state.quality_score + 0.15 * grammar_score

                # Adjust failure diagnosis if grammaticality is very low
                if grammar_score < 0.4 and state.quality_score < self.QUALITY_THRESHOLD:
                    state.failure_diagnosis = 'grammaticality'
            except Exception as e:
                logging.debug(f"Stage 9 grammaticality scoring failed: {e}")

        # Add coherence metrics component if available
        if self.coherence_analyzer and state.response_vector is not None and state.psi_collective is not None:
            try:
                # Analyze coherence between response_vector and psi_collective
                coherence_result = self.coherence_analyzer.analyze_convergence(
                    psi_agents=state.response_vector,  # Single agent state
                    G_point=state.psi_collective       # Integration point
                )

                # Add coherence components to quality
                coherence_score = (coherence_result.synergy + (1 - coherence_result.redundancy)) / 2
                state.quality_components['coherence'] = coherence_score
                state.quality_components['synergy'] = coherence_result.synergy
                state.quality_components['redundancy'] = coherence_result.redundancy

                # Blend coherence into quality (10% weight to not dominate)
                state.quality_score = 0.9 * state.quality_score + 0.1 * coherence_score
            except Exception as e:
                logging.debug(f"Stage 9 coherence analysis failed: {e}")

        # Learn connector effectiveness from quality score
        self.learn_connector_effectiveness(state.quality_score)

        # Learn frame effectiveness from quality score
        if self.discourse_state and hasattr(self.discourse_state, 'learn_frame_effectiveness'):
            try:
                last_frame = self.discourse_state.get_last_frame_used() if hasattr(self.discourse_state, 'get_last_frame_used') else None
                if last_frame:
                    response_need, frame_used = last_frame
                    self.discourse_state.learn_frame_effectiveness(
                        response_need=response_need,
                        frame_used=frame_used,
                        response_quality=state.quality_score
                    )
            except Exception as e:
                logging.debug(f"Frame effectiveness learning failed: {e}")

        # Create EmissionResult
        state.emission_result = EmissionResult(
            response=' '.join(words),
            success=state.quality_score >= self.QUALITY_THRESHOLD,
            backtrack_target=state.failure_diagnosis if state.failure_diagnosis != 'acceptable' else None,
            quality_score=state.quality_score,
            components=state.quality_components
        )

        return state


# =============================================================================
# NOTE: No singleton - instantiate in GraceInteractiveDialogue.__init__
# =============================================================================
