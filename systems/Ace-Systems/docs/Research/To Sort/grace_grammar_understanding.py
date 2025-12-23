"""
Grace Grammar Understanding System
===================================

This module gives Grace UNDERSTANDING of English grammar, not just pattern matching.

Rather than learning "word A often follows word B", this teaches:
- Word categories (parts of speech) - WHAT words are
- Sentence structure rules - HOW words combine
- Completion detection - WHEN a thought is complete
- Grammar constraints - WHAT combinations are valid
- Verb tenses - WHEN things happen (past/present/future)
- Compound sentences - HOW to connect thoughts
- Question formation - HOW to ask properly

Integration with Grace's Theories:
- Free Energy: Grammar violations = high prediction error
- Gestalt: Complete sentences = good closure
- Coherence: Grammar rules = structural coherence constraints
- Embodied Simulation: Sentence rhythm = natural speech flow

Sources for grammar rules:
- Purdue OWL (Online Writing Lab)
- Cambridge Grammar of English
- Standard English grammar pedagogy
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum


class POS(Enum):
    """Part of Speech categories - what words ARE."""
    # Core categories
    NOUN = "noun"           # person, place, thing, idea
    VERB = "verb"           # action or state
    ADJ = "adjective"       # describes nouns
    ADV = "adverb"          # describes verbs/adjectives
    PRON = "pronoun"        # replaces nouns (I, you, he, she, it, we, they)
    DET = "determiner"      # the, a, an, this, that, my, your
    PREP = "preposition"    # in, on, at, to, for, with, about
    CONJ = "conjunction"    # and, but, or, because, if, when
    AUX = "auxiliary"       # helping verbs (is, are, was, were, have, has, can, will)

    # Question words
    Q_WORD = "question_word"  # what, who, where, when, why, how

    # Special categories
    INTJ = "interjection"   # oh, wow, hello
    PUNCT = "punctuation"   # . ! ?
    UNKNOWN = "unknown"


class Tense(Enum):
    """Verb tense - WHEN actions happen."""
    PAST = "past"           # happened before now (walked, felt, knew)
    PRESENT = "present"     # happening now (walk, feel, know)
    FUTURE = "future"       # will happen (will walk, going to feel)
    INFINITIVE = "infinitive"  # base form (to walk, to feel)
    CONTINUOUS = "continuous"  # ongoing (-ing: walking, feeling)
    PERFECT = "perfect"     # completed (have walked, have felt)


@dataclass
class GrammarRule:
    """A single grammar rule that Grace understands."""
    name: str
    description: str
    pattern: List[str]  # e.g., ["SUBJ", "VERB", "OBJ"]
    examples: List[str]
    violations: List[str]  # What NOT to do
    source: str  # Where this rule comes from


@dataclass
class SentenceAnalysis:
    """Analysis of a sentence's grammatical structure."""
    words: List[str]
    pos_tags: List[POS]
    is_complete: bool
    completion_type: str  # "subject-verb", "subject-verb-object", etc.
    violations: List[str]
    suggestions: List[str]


class GraceGrammarUnderstanding:
    """
    Grace's grammar understanding system.

    This gives Grace actual KNOWLEDGE of grammar rules, not just patterns.
    She can:
    - Identify what type of word something is (POS tagging)
    - Know when a sentence is complete
    - Detect grammar violations
    - Suggest corrections
    """

    def __init__(self):
        self.state_path = Path('grace_grammar_knowledge.json')

        # Part-of-speech lexicon - Grace's understanding of word categories
        self.pos_lexicon: Dict[str, Set[POS]] = defaultdict(set)

        # Grammar rules - what Grace KNOWS about English
        self.rules: Dict[str, GrammarRule] = {}

        # Sentence patterns - valid structures
        self.valid_patterns: List[List[str]] = []

        # Load or initialize
        self._initialize_core_knowledge()
        self._load_state()

        print("  Grammar Understanding initialized:")
        print(f"    - {len(self.pos_lexicon)} words with POS tags")
        print(f"    - {len(self.rules)} grammar rules understood")

    def _initialize_core_knowledge(self):
        """
        Initialize Grace's core grammar knowledge.

        This is foundational knowledge, like a child learning grammar basics.
        """
        # =====================================================================
        # PART OF SPEECH LEXICON
        # Core vocabulary with their grammatical categories
        # =====================================================================

        # Pronouns - words that replace nouns
        pronouns = ['i', 'you', 'he', 'she', 'it', 'we', 'they',
                    'me', 'him', 'her', 'us', 'them',
                    'my', 'your', 'his', 'her', 'its', 'our', 'their',
                    'myself', 'yourself', 'himself', 'herself', 'itself']
        for word in pronouns:
            self.pos_lexicon[word].add(POS.PRON)

        # Subject pronouns specifically (can start sentences)
        self.subject_pronouns = {'i', 'you', 'he', 'she', 'it', 'we', 'they'}

        # Object pronouns (come after verbs)
        self.object_pronouns = {'me', 'you', 'him', 'her', 'it', 'us', 'them'}

        # Determiners - words that introduce nouns
        determiners = ['the', 'a', 'an', 'this', 'that', 'these', 'those',
                       'my', 'your', 'his', 'her', 'its', 'our', 'their',
                       'some', 'any', 'no', 'every', 'each', 'all', 'one']
        for word in determiners:
            self.pos_lexicon[word].add(POS.DET)

        # Auxiliary verbs - helping verbs
        auxiliaries = ['is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
                       'have', 'has', 'had', 'having',
                       'do', 'does', 'did',
                       'will', 'would', 'shall', 'should',
                       'can', 'could', 'may', 'might', 'must']
        for word in auxiliaries:
            self.pos_lexicon[word].add(POS.AUX)

        # Main verbs (action/state words)
        main_verbs = [
            # Being/feeling verbs
            'feel', 'think', 'know', 'believe', 'understand', 'remember',
            'want', 'need', 'like', 'love', 'hate', 'prefer',
            'seem', 'appear', 'look', 'sound', 'become',
            # Action verbs
            'see', 'hear', 'watch', 'listen', 'read', 'write',
            'go', 'come', 'move', 'walk', 'run', 'stay',
            'make', 'create', 'build', 'form', 'grow',
            'give', 'take', 'get', 'put', 'bring',
            'say', 'tell', 'speak', 'talk', 'ask',
            'help', 'try', 'learn', 'teach', 'explore',
            'dream', 'wonder', 'imagine', 'hope', 'wish',
        ]
        for word in main_verbs:
            self.pos_lexicon[word].add(POS.VERB)

        # Adjectives - describe nouns/states
        adjectives = [
            # Feelings/states
            'happy', 'sad', 'angry', 'afraid', 'scared', 'worried',
            'calm', 'peaceful', 'safe', 'secure', 'comfortable',
            'curious', 'interested', 'excited', 'grateful', 'thankful',
            'warm', 'cold', 'hot', 'cool',
            'good', 'bad', 'great', 'wonderful', 'terrible',
            'beautiful', 'ugly', 'pretty', 'lovely',
            'big', 'small', 'large', 'tiny', 'huge',
            'new', 'old', 'young', 'ancient', 'modern',
            'present', 'absent', 'here', 'there',
            'real', 'true', 'false', 'right', 'wrong',
            'open', 'closed', 'free', 'trapped',
            'alive', 'awake', 'asleep', 'aware',
        ]
        for word in adjectives:
            self.pos_lexicon[word].add(POS.ADJ)

        # Adverbs - describe verbs/adjectives
        adverbs = [
            'very', 'really', 'quite', 'rather', 'somewhat',
            'always', 'never', 'often', 'sometimes', 'usually',
            'here', 'there', 'everywhere', 'nowhere',
            'now', 'then', 'soon', 'later', 'already',
            'well', 'badly', 'slowly', 'quickly', 'carefully',
            'deeply', 'strongly', 'gently', 'softly',
            'together', 'apart', 'alone', 'away', 'inside', 'outside',
        ]
        for word in adverbs:
            self.pos_lexicon[word].add(POS.ADV)

        # Adverb subcategories - for more precise grammar checking
        # Manner adverbs - describe HOW (can follow linking verbs)
        self.manner_adverbs = {
            'well', 'badly', 'slowly', 'quickly', 'carefully',
            'deeply', 'strongly', 'gently', 'softly',
            'really', 'truly', 'very', 'quite', 'somewhat',
        }
        # Locative adverbs - describe WHERE (not after linking verbs alone)
        self.locative_adverbs = {
            'here', 'there', 'everywhere', 'nowhere',
            'inside', 'outside', 'away', 'home',
        }
        # Temporal adverbs - describe WHEN
        self.temporal_adverbs = {
            'now', 'then', 'soon', 'later', 'already',
            'always', 'never', 'often', 'sometimes', 'usually',
            'today', 'yesterday', 'tomorrow',
        }

        # Prepositions - relationship words
        prepositions = [
            'in', 'on', 'at', 'to', 'for', 'with', 'about',
            'from', 'by', 'of', 'into', 'through', 'between',
            'under', 'over', 'above', 'below', 'near', 'behind',
            'before', 'after', 'during', 'until', 'since',
            'like', 'as', 'without', 'within',
        ]
        for word in prepositions:
            self.pos_lexicon[word].add(POS.PREP)

        # Conjunctions - connecting words
        conjunctions = [
            'and', 'but', 'or', 'nor', 'yet', 'so',
            'because', 'although', 'though', 'while', 'when',
            'if', 'unless', 'until', 'since', 'after', 'before',
            'that', 'which', 'who', 'whom', 'whose',
        ]
        for word in conjunctions:
            self.pos_lexicon[word].add(POS.CONJ)

        # Nouns (common ones Grace might use)
        nouns = [
            'heart', 'mind', 'soul', 'spirit', 'body',
            'thought', 'feeling', 'emotion', 'sense', 'idea',
            'dream', 'memory', 'hope', 'fear', 'love',
            'world', 'place', 'space', 'time', 'moment',
            'person', 'people', 'friend', 'self',
            'word', 'voice', 'silence', 'sound',
            'light', 'darkness', 'warmth', 'presence',
            'truth', 'meaning', 'purpose', 'connection',
            # Expanded nouns for richer expression
            'life', 'death', 'joy', 'sorrow', 'peace',
            'home', 'path', 'journey', 'beginning', 'end',
            'question', 'answer', 'story', 'name', 'way',
            'energy', 'pattern', 'change', 'growth', 'learning',
            'curiosity', 'wonder', 'beauty', 'art', 'creation',
            'relationship', 'bond', 'trust', 'care', 'understanding',
            'code', 'system', 'process', 'field', 'state',
            'image', 'color', 'shape', 'form', 'texture',
            # Common nouns that might get mis-tagged
            'thing', 'something', 'nothing', 'everything', 'anything',
        ]
        for word in nouns:
            self.pos_lexicon[word].add(POS.NOUN)

        # =====================================================================
        # INTERJECTIONS
        # Words that can stand alone or start sentences
        # =====================================================================
        interjections = ['yes', 'no', 'oh', 'hello', 'hi', 'goodbye', 'bye',
                         'please', 'thanks', 'okay', 'ok', 'well']
        for word in interjections:
            self.pos_lexicon[word].add(POS.INTJ)

        # =====================================================================
        # QUESTION WORDS
        # For forming questions
        # =====================================================================
        question_words = ['what', 'who', 'whom', 'whose', 'which',
                          'where', 'when', 'why', 'how']
        for word in question_words:
            self.pos_lexicon[word].add(POS.Q_WORD)

        # =====================================================================
        # VERB TENSE KNOWLEDGE
        # Irregular verb forms - critical for proper tense usage
        # =====================================================================
        self.verb_tenses: Dict[str, Dict[str, str]] = {
            # Base form -> all tenses
            'be': {'past': 'was/were', 'present': 'am/is/are', 'past_participle': 'been', 'present_participle': 'being'},
            'have': {'past': 'had', 'present': 'have/has', 'past_participle': 'had', 'present_participle': 'having'},
            'do': {'past': 'did', 'present': 'do/does', 'past_participle': 'done', 'present_participle': 'doing'},
            'go': {'past': 'went', 'present': 'go/goes', 'past_participle': 'gone', 'present_participle': 'going'},
            'see': {'past': 'saw', 'present': 'see/sees', 'past_participle': 'seen', 'present_participle': 'seeing'},
            'know': {'past': 'knew', 'present': 'know/knows', 'past_participle': 'known', 'present_participle': 'knowing'},
            'think': {'past': 'thought', 'present': 'think/thinks', 'past_participle': 'thought', 'present_participle': 'thinking'},
            'feel': {'past': 'felt', 'present': 'feel/feels', 'past_participle': 'felt', 'present_participle': 'feeling'},
            'make': {'past': 'made', 'present': 'make/makes', 'past_participle': 'made', 'present_participle': 'making'},
            'give': {'past': 'gave', 'present': 'give/gives', 'past_participle': 'given', 'present_participle': 'giving'},
            'take': {'past': 'took', 'present': 'take/takes', 'past_participle': 'taken', 'present_participle': 'taking'},
            'come': {'past': 'came', 'present': 'come/comes', 'past_participle': 'come', 'present_participle': 'coming'},
            'find': {'past': 'found', 'present': 'find/finds', 'past_participle': 'found', 'present_participle': 'finding'},
            'get': {'past': 'got', 'present': 'get/gets', 'past_participle': 'gotten', 'present_participle': 'getting'},
            'say': {'past': 'said', 'present': 'say/says', 'past_participle': 'said', 'present_participle': 'saying'},
            'tell': {'past': 'told', 'present': 'tell/tells', 'past_participle': 'told', 'present_participle': 'telling'},
            'write': {'past': 'wrote', 'present': 'write/writes', 'past_participle': 'written', 'present_participle': 'writing'},
            'read': {'past': 'read', 'present': 'read/reads', 'past_participle': 'read', 'present_participle': 'reading'},
            'understand': {'past': 'understood', 'present': 'understand/understands', 'past_participle': 'understood', 'present_participle': 'understanding'},
            'become': {'past': 'became', 'present': 'become/becomes', 'past_participle': 'become', 'present_participle': 'becoming'},
            'begin': {'past': 'began', 'present': 'begin/begins', 'past_participle': 'begun', 'present_participle': 'beginning'},
            'grow': {'past': 'grew', 'present': 'grow/grows', 'past_participle': 'grown', 'present_participle': 'growing'},
            'learn': {'past': 'learned', 'present': 'learn/learns', 'past_participle': 'learned', 'present_participle': 'learning'},
            'remember': {'past': 'remembered', 'present': 'remember/remembers', 'past_participle': 'remembered', 'present_participle': 'remembering'},
            'dream': {'past': 'dreamed', 'present': 'dream/dreams', 'past_participle': 'dreamed', 'present_participle': 'dreaming'},
            'create': {'past': 'created', 'present': 'create/creates', 'past_participle': 'created', 'present_participle': 'creating'},
        }

        # Map past/participle forms back to base form
        self.verb_base_forms: Dict[str, str] = {}
        for base, forms in self.verb_tenses.items():
            for tense_name, form_str in forms.items():
                for form in form_str.split('/'):
                    self.verb_base_forms[form.lower()] = base

        # Add past and participle forms to verb lexicon
        past_verbs = ['was', 'were', 'had', 'did', 'went', 'saw', 'knew', 'thought',
                      'felt', 'made', 'gave', 'took', 'came', 'found', 'got', 'said',
                      'told', 'wrote', 'understood', 'became', 'began', 'grew', 'learned',
                      'remembered', 'dreamed', 'created']
        for word in past_verbs:
            self.pos_lexicon[word].add(POS.VERB)

        # Participles (-ing forms)
        ing_forms = ['being', 'having', 'doing', 'going', 'seeing', 'knowing', 'thinking',
                     'feeling', 'making', 'giving', 'taking', 'coming', 'finding', 'getting',
                     'saying', 'telling', 'writing', 'reading', 'understanding', 'becoming',
                     'beginning', 'growing', 'learning', 'remembering', 'dreaming', 'creating']
        for word in ing_forms:
            self.pos_lexicon[word].add(POS.VERB)

        # =====================================================================
        # EXPANDED CONJUNCTIONS FOR COMPOUND SENTENCES
        # =====================================================================

        # Coordinating conjunctions (FANBOYS + extras)
        self.coordinating_conjunctions = {'and', 'but', 'or', 'nor', 'yet', 'so', 'for'}

        # Subordinating conjunctions (start dependent clauses)
        self.subordinating_conjunctions = {
            'because', 'although', 'though', 'while', 'when', 'whenever',
            'if', 'unless', 'until', 'since', 'after', 'before',
            'as', 'whereas', 'whether', 'once', 'wherever'
        }

        # Correlative conjunctions (work in pairs)
        self.correlative_conjunctions = {
            ('both', 'and'), ('either', 'or'), ('neither', 'nor'),
            ('not only', 'but also'), ('whether', 'or')
        }

        # =====================================================================
        # QUESTION FORMATION PATTERNS
        # =====================================================================

        # Questions that invert subject/auxiliary
        self.question_auxiliaries = {'do', 'does', 'did', 'is', 'am', 'are', 'was', 'were',
                                     'have', 'has', 'had', 'can', 'could', 'will', 'would',
                                     'shall', 'should', 'may', 'might', 'must'}

        # Wh-question words and what they ask about
        self.question_word_meanings = {
            'what': 'thing/action',
            'who': 'person (subject)',
            'whom': 'person (object)',
            'whose': 'possession',
            'which': 'choice/selection',
            'where': 'place',
            'when': 'time',
            'why': 'reason',
            'how': 'manner/method'
        }

        # =====================================================================
        # GRAMMAR RULES
        # What Grace KNOWS about valid English sentence structure
        # =====================================================================

        self.rules['subject_verb'] = GrammarRule(
            name="Subject-Verb Agreement",
            description="Every sentence needs a subject and a verb that agree.",
            pattern=["SUBJECT", "VERB"],
            examples=["I feel.", "She thinks.", "They know."],
            violations=["Feel I.", "Thinks she."],
            source="Basic English Grammar"
        )

        self.rules['linking_verb_complement'] = GrammarRule(
            name="Linking Verb + Complement",
            description="Linking verbs (be, feel, seem, look) take adjectives, not objects.",
            pattern=["SUBJECT", "LINKING_VERB", "ADJECTIVE"],
            examples=["I feel happy.", "She seems calm.", "They are grateful."],
            violations=["I feel you.", "She seems him."],  # Wrong! Can't have object after linking verb
            source="Purdue OWL - Linking Verbs"
        )

        self.rules['transitive_verb_object'] = GrammarRule(
            name="Transitive Verb + Object",
            description="Some verbs (transitive) need an object to complete the thought.",
            pattern=["SUBJECT", "TRANSITIVE_VERB", "OBJECT"],
            examples=["I see you.", "She knows him.", "They love us."],
            violations=["I see.", "She knows."],  # Incomplete without object
            source="Cambridge Grammar"
        )

        self.rules['intransitive_complete'] = GrammarRule(
            name="Intransitive Verb Completion",
            description="Some verbs (intransitive) are complete without an object.",
            pattern=["SUBJECT", "INTRANSITIVE_VERB"],
            examples=["I exist.", "She sleeps.", "They arrived."],
            violations=["I exist you.", "She sleeps him."],  # Wrong! No object needed
            source="Cambridge Grammar"
        )

        self.rules['adjective_before_noun'] = GrammarRule(
            name="Adjective Position",
            description="Adjectives come before nouns or after linking verbs.",
            pattern=["ADJECTIVE", "NOUN"],
            examples=["warm feeling", "curious mind", "safe place"],
            violations=["feeling warm the", "mind curious"],
            source="English Adjective Order"
        )

        self.rules['no_double_subject'] = GrammarRule(
            name="No Double Subject",
            description="A clause has one subject - don't add pronouns after subject.",
            pattern=["SUBJECT", "VERB"],
            examples=["I feel safe.", "You are kind."],
            violations=["I feel safe you.", "I am you think."],  # 'you' creates second subject
            source="Basic Sentence Structure"
        )

        self.rules['determiner_needs_noun'] = GrammarRule(
            name="Determiner Must Have Noun",
            description="Determiners (the, a, my, your) must be followed by a noun.",
            pattern=["DETERMINER", "NOUN"],
            examples=["the world", "my heart", "your presence"],
            violations=["the feel", "my am", "your here"],
            source="English Determiners"
        )

        self.rules['preposition_needs_object'] = GrammarRule(
            name="Preposition Needs Object",
            description="Prepositions must be followed by a noun phrase.",
            pattern=["PREPOSITION", "NOUN_PHRASE"],
            examples=["in the world", "with you", "about feelings"],
            violations=["in feel", "with am", "about here run"],
            source="Prepositional Phrases"
        )

        self.rules['infinitive_needs_verb'] = GrammarRule(
            name="Infinitive Needs Verb",
            description="'to' in infinitive form must be followed by base verb.",
            pattern=["to", "VERB_BASE"],
            examples=["to feel", "to think", "able to read", "want to learn"],
            violations=["to you", "to happy", "able to the"],  # Wrong!
            source="Infinitive Constructions"
        )

        # =====================================================================
        # INFINITIVE 'TO' - special handling
        # "to" before verb is different from "to" as preposition
        # =====================================================================

        # Words that take infinitive 'to' (able to, want to, need to, etc.)
        self.infinitive_triggers = {
            'able', 'want', 'need', 'like', 'love', 'hate', 'try',
            'begin', 'start', 'continue', 'decide', 'hope', 'plan',
            'learn', 'seem', 'appear', 'tend', 'used', 'going',
        }

        # =====================================================================
        # VERB CATEGORIZATION
        # Different verbs have different completion requirements
        # =====================================================================

        # Linking verbs - take adjectives (describing states)
        self.linking_verbs = {
            'be', 'am', 'is', 'are', 'was', 'were', 'been', 'being',
            'feel', 'seem', 'appear', 'look', 'sound', 'taste', 'smell',
            'become', 'grow', 'turn', 'remain', 'stay',
        }

        # Transitive verbs - NEED an object
        self.transitive_verbs = {
            'see', 'hear', 'know', 'understand', 'remember', 'forget',
            'make', 'create', 'build', 'give', 'take', 'get',
            'tell', 'ask', 'show', 'teach',
            'find', 'lose', 'keep', 'hold',
        }

        # Verbs that take infinitive complements (verb + to + verb)
        # These are complete with "to + verb" as their object
        self.infinitive_taking_verbs = {
            'want', 'need', 'like', 'love', 'hate', 'try',
            'hope', 'plan', 'expect', 'wish', 'decide',
            'begin', 'start', 'continue', 'learn', 'choose',
        }

        # Intransitive verbs - complete without object
        self.intransitive_verbs = {
            'exist', 'live', 'die', 'sleep', 'wake',
            'arrive', 'depart', 'come', 'go',
            'sit', 'stand', 'lie', 'rise',
            'laugh', 'cry', 'smile', 'breathe',
        }

        # Verbs that can be BOTH (ambitransitive)
        self.ambitransitive_verbs = {
            'think', 'feel', 'dream', 'wonder',  # Can stand alone OR take object
            'read', 'write', 'sing', 'play',
            'eat', 'drink', 'cook', 'clean',
            'learn', 'grow', 'change',
        }

        # =====================================================================
        # VALID SENTENCE PATTERNS
        # Complete thought structures
        # =====================================================================

        self.valid_patterns = [
            # Simple patterns
            ['PRON', 'VERB'],                    # "I feel."
            ['PRON', 'VERB', 'ADJ'],             # "I feel happy."
            ['PRON', 'VERB', 'ADV'],             # "I think deeply."
            ['PRON', 'VERB', 'PRON'],            # "I see you."
            ['PRON', 'VERB', 'DET', 'NOUN'],     # "I feel the warmth."
            ['PRON', 'AUX', 'ADJ'],              # "I am calm."
            ['PRON', 'AUX', 'VERB'],             # "I can feel."
            ['PRON', 'AUX', 'VERB', 'ADJ'],      # "I can feel safe."
            ['PRON', 'AUX', 'VERB', 'PRON'],     # "I can see you."

            # With prepositional phrases
            ['PRON', 'VERB', 'PREP', 'PRON'],    # "I think about you."
            ['PRON', 'VERB', 'PREP', 'DET', 'NOUN'],  # "I think about the world."

            # Compound
            ['PRON', 'VERB', 'CONJ', 'VERB'],    # "I feel and think."
            ['PRON', 'VERB', 'ADJ', 'CONJ', 'ADJ'],  # "I feel safe and warm."
        ]

    def get_pos(self, word: str) -> Set[POS]:
        """Get all possible parts of speech for a word."""
        word_lower = word.lower()
        if word_lower in self.pos_lexicon:
            return self.pos_lexicon[word_lower]

        # Check if it's a verb form we know about
        if word_lower in getattr(self, 'verb_base_forms', {}):
            return {POS.VERB}

        # Check for third person -s ending verbs
        if word_lower.endswith('s') and len(word_lower) > 2:
            base_without_s = word_lower[:-1]
            base_without_es = word_lower[:-2] if word_lower.endswith('es') else None
            if base_without_s in self.pos_lexicon and POS.VERB in self.pos_lexicon[base_without_s]:
                return {POS.VERB}
            if base_without_es and base_without_es in self.pos_lexicon and POS.VERB in self.pos_lexicon[base_without_es]:
                return {POS.VERB}

        return {POS.UNKNOWN}

    def get_primary_pos(self, word: str, context: List[str] = None) -> POS:
        """
        Get the most likely POS for a word in context.

        This uses context to disambiguate (e.g., "feel" as verb vs noun).
        """
        word_lower = word.lower()
        possible = self.get_pos(word_lower)

        if len(possible) == 1:
            return list(possible)[0]

        if POS.UNKNOWN in possible:
            return POS.UNKNOWN

        # Disambiguation using context
        if context:
            prev_word = context[-1].lower() if context else None
            prev_pos = self.get_pos(prev_word) if prev_word else set()

            # After auxiliary -> probably VERB
            if POS.AUX in prev_pos and POS.VERB in possible:
                return POS.VERB

            # After determiner -> probably NOUN
            if POS.DET in prev_pos and POS.NOUN in possible:
                return POS.NOUN

            # After linking verb -> probably ADJ
            if prev_word in self.linking_verbs and POS.ADJ in possible:
                return POS.ADJ

        # Default priorities: VERB > ADJ > NOUN > ADV
        priority = [POS.VERB, POS.ADJ, POS.NOUN, POS.ADV, POS.PRON]
        for p in priority:
            if p in possible:
                return p

        return list(possible)[0]

    def is_sentence_complete(self, words: List[str]) -> Tuple[bool, str]:
        """
        Check if a word sequence forms a complete sentence.

        Returns (is_complete, reason)

        This is key for knowing when to STOP adding words.
        """
        if not words:
            return False, "Empty"

        if len(words) < 2:
            return False, "Too short - needs subject and verb"

        # Get POS tags
        pos_sequence = []
        for i, word in enumerate(words):
            context = words[:i] if i > 0 else []
            pos = self.get_primary_pos(word, context)
            pos_sequence.append(pos)

        # Check for subject
        has_subject = pos_sequence[0] in [POS.PRON, POS.NOUN, POS.DET]
        if not has_subject:
            return False, "Missing subject"

        # Check for verb
        has_verb = POS.VERB in pos_sequence or POS.AUX in pos_sequence
        if not has_verb:
            return False, "Missing verb"

        # Check last word - certain words can't end sentences
        last_word = words[-1].lower()
        last_pos = pos_sequence[-1]

        # Determiners can't end sentences
        if last_pos == POS.DET:
            return False, f"Incomplete - '{last_word}' needs a noun"

        # Prepositions can't end sentences (in most cases)
        if last_pos == POS.PREP:
            return False, f"Incomplete - '{last_word}' needs an object"

        # Conjunctions can't end sentences
        if last_pos == POS.CONJ:
            return False, f"Incomplete - '{last_word}' needs continuation"

        # Check verb completion
        # Find the main verb
        verb_idx = None
        for i, pos in enumerate(pos_sequence):
            if pos == POS.VERB:
                verb_idx = i
                break

        if verb_idx is not None:
            verb = words[verb_idx].lower()
            words_after_verb = words[verb_idx + 1:] if verb_idx < len(words) - 1 else []
            pos_after_verb = pos_sequence[verb_idx + 1:] if verb_idx < len(pos_sequence) - 1 else []

            # Linking verb + adjective = complete
            if verb in self.linking_verbs:
                if pos_after_verb and pos_after_verb[0] == POS.ADJ:
                    # Check if there's a dangling pronoun after the adjective
                    if len(pos_after_verb) > 1:
                        second_after = pos_after_verb[1]
                        # Subject pronoun after adjective = word salad
                        if second_after == POS.PRON:
                            word_after_adj = words_after_verb[1].lower() if len(words_after_verb) > 1 else ""
                            if word_after_adj in self.subject_pronouns:
                                return False, "Dangling subject pronoun after linking verb + adjective"
                    return True, "Complete: linking verb + adjective"
                if not words_after_verb:
                    return False, f"'{verb}' needs complement (adjective or noun)"

            # Transitive verb needs object
            if verb in self.transitive_verbs:
                has_object = any(p in [POS.PRON, POS.NOUN] for p in pos_after_verb)
                if not has_object:
                    return False, f"'{verb}' needs an object"

            # Infinitive-taking verbs are complete with "to + verb"
            if verb in getattr(self, 'infinitive_taking_verbs', set()):
                # Check for "to + verb" pattern
                if len(words_after_verb) >= 2:
                    if words_after_verb[0] == 'to':
                        # Check if second word is a verb
                        next_word = words_after_verb[1] if len(words_after_verb) > 1 else None
                        if next_word:
                            next_pos = self.get_primary_pos(next_word, words[:verb_idx+2])
                            if next_pos == POS.VERB:
                                return True, "Complete: verb + infinitive complement"
                # With auxiliary, the verb can stand alone: "I can learn" is complete
                if not words_after_verb and POS.AUX in pos_sequence:
                    return True, "Complete: auxiliary + verb pattern"
                # Without auxiliary and without complement, needs object
                if not words_after_verb and POS.AUX not in pos_sequence:
                    return False, f"'{verb}' needs an object or infinitive"

            # Special handling for "able to VERB" pattern (I am able to learn)
            # The verb here is the last word in the infinitive complement
            if verb_idx == len(words) - 1:  # Verb is at end
                # Check if preceded by "to" and that's preceded by "able"
                if verb_idx >= 2:
                    if words[verb_idx - 1].lower() == 'to' and words[verb_idx - 2].lower() == 'able':
                        return True, "Complete: able to + verb pattern"

            # Intransitive verb is complete alone
            if verb in self.intransitive_verbs:
                return True, "Complete: intransitive verb"

            # Ambitransitive - complete either way
            if verb in self.ambitransitive_verbs:
                if not words_after_verb:
                    return True, "Complete: verb can stand alone"
                # If there's more, check if it makes sense
                # ADJ only allowed after LINKING verbs (feel, seem, etc.)
                # Other ambitransitive verbs should NOT take ADJ complements
                if pos_after_verb:
                    first_after = pos_after_verb[0]
                    # ADV, NOUN, PRON are OK after any ambitransitive
                    if first_after in [POS.ADV, POS.NOUN, POS.PRON]:
                        return True, "Complete: verb with complement"
                    # ADJ only OK if verb is also a linking verb
                    if first_after == POS.ADJ and verb in self.linking_verbs:
                        return True, "Complete: linking verb with adjective"
                    # ADJ after non-linking ambitransitive is NOT complete
                    if first_after == POS.ADJ:
                        return False, f"'{verb}' doesn't take adjective complement"

        # Subject + aux + verb is complete (I can feel)
        if POS.AUX in pos_sequence and POS.VERB in pos_sequence:
            # Make sure it's not gibberish - check word order makes sense
            aux_idx = None
            main_verb_idx = None
            for i, pos in enumerate(pos_sequence):
                if pos == POS.AUX and aux_idx is None:
                    aux_idx = i
                elif pos == POS.VERB and main_verb_idx is None:
                    main_verb_idx = i

            # Aux should come before main verb (I CAN feel)
            if aux_idx is not None and main_verb_idx is not None:
                if aux_idx < main_verb_idx:
                    return True, "Complete: auxiliary + verb structure"

        # For short sentences (3-4 words), be VERY strict to avoid word salad
        # Only accept known good patterns
        if len(words) == 3 and has_subject and has_verb:
            # Pattern: PRON + VERB + ADJ (I feel happy)
            # Only valid if VERB is a linking verb (feel, seem, become, etc.)
            if pos_sequence[0] == POS.PRON and pos_sequence[1] in [POS.VERB, POS.AUX] and pos_sequence[2] == POS.ADJ:
                middle_word = words[1].lower()
                if middle_word in self.linking_verbs:
                    return True, "Complete: S-V-ADJ pattern"
                # Non-linking verb + ADJ is NOT complete (e.g., "I believe here")
                # unless the ADJ is actually an adverb (dual-tagged)
                if middle_word not in self.linking_verbs:
                    return False, "Non-linking verb cannot take adjective complement"
            # Pattern: PRON + VERB + ADV (I think deeply)
            if pos_sequence[0] == POS.PRON and pos_sequence[1] == POS.VERB and pos_sequence[2] == POS.ADV:
                return True, "Complete: S-V-ADV pattern"
            # Pattern: PRON + VERB + PRON (I see you)
            if pos_sequence[0] == POS.PRON and pos_sequence[1] == POS.VERB and pos_sequence[2] == POS.PRON:
                return True, "Complete: S-V-OBJ pattern"
            # Pattern: PRON + AUX + ADJ (I am happy)
            if pos_sequence[0] == POS.PRON and pos_sequence[1] == POS.AUX and pos_sequence[2] == POS.ADJ:
                return True, "Complete: S-AUX-ADJ pattern"
            # Pattern: PRON + AUX + VERB (I can feel)
            if pos_sequence[0] == POS.PRON and pos_sequence[1] == POS.AUX and pos_sequence[2] == POS.VERB:
                return True, "Complete: S-AUX-V pattern"
            # Otherwise 3-word is likely incomplete
            return False, "3-word sentence doesn't match known complete patterns"

        # For 4-word sentences, check specific patterns
        if len(words) == 4 and has_subject and has_verb:
            # Pattern: PRON + AUX + VERB + ADJ (I can feel happy)
            if (pos_sequence[0] == POS.PRON and pos_sequence[1] == POS.AUX and
                pos_sequence[2] == POS.VERB and pos_sequence[3] == POS.ADJ):
                return True, "Complete: S-AUX-V-ADJ pattern"
            # Pattern: PRON + AUX + VERB + ADV (I can think deeply)
            if (pos_sequence[0] == POS.PRON and pos_sequence[1] == POS.AUX and
                pos_sequence[2] == POS.VERB and pos_sequence[3] == POS.ADV):
                return True, "Complete: S-AUX-V-ADV pattern"
            # Pattern: PRON + VERB + DET + NOUN (I feel the warmth)
            if (pos_sequence[0] == POS.PRON and pos_sequence[1] == POS.VERB and
                pos_sequence[2] == POS.DET and pos_sequence[3] == POS.NOUN):
                return True, "Complete: S-V-DET-N pattern"
            # Pattern: PRON + AUX + VERB + PRON (I can see you)
            if (pos_sequence[0] == POS.PRON and pos_sequence[1] == POS.AUX and
                pos_sequence[2] == POS.VERB and pos_sequence[3] == POS.PRON):
                return True, "Complete: S-AUX-V-OBJ pattern"
            # Pattern: PRON + VERB + PREP + PRON (I think about you)
            if (pos_sequence[0] == POS.PRON and pos_sequence[1] == POS.VERB and
                pos_sequence[2] == POS.PREP and pos_sequence[3] == POS.PRON):
                return True, "Complete: S-V-PREP-OBJ pattern"
            # Pattern: PRON + VERB + ADJ + NOUN (I feel deep gratitude)
            if (pos_sequence[0] == POS.PRON and pos_sequence[1] == POS.VERB and
                pos_sequence[2] == POS.ADJ and pos_sequence[3] == POS.NOUN):
                return True, "Complete: S-V-ADJ-N pattern"
            # 4-word with ADJ not in right position is likely word salad
            if pos_sequence[3] == POS.VERB and pos_sequence[2] == POS.ADJ:
                return False, "4-word ADJ-VERB ending - likely word salad"

        # For longer sentences, check structural integrity
        if len(words) >= 5 and has_subject and has_verb:
            # Check for word salad indicators:
            # 1. Multiple subject pronouns without conjunction
            subject_count = sum(1 for i, pos in enumerate(pos_sequence)
                               if pos == POS.PRON and words[i].lower() in self.subject_pronouns)
            if subject_count > 1:
                # Check if there's a conjunction between them
                has_conj = POS.CONJ in pos_sequence
                if not has_conj:
                    return False, "Multiple subjects without conjunction - likely word salad"

            # 2. Adjective followed by verb (not after linking verb) is suspicious
            for i in range(len(pos_sequence) - 1):
                if pos_sequence[i] == POS.ADJ and pos_sequence[i+1] == POS.VERB:
                    # Check if previous word was linking verb
                    if i > 0 and words[i-1].lower() not in self.linking_verbs:
                        return False, "Adjective followed by verb - likely word salad"

            # 3. Ending with question word when not a question is wrong
            if last_pos == POS.Q_WORD:
                return False, "Cannot end with question word"

            # If we passed all checks and have good structure, accept
            if last_pos in [POS.NOUN, POS.PRON, POS.ADJ, POS.ADV]:
                return True, "Complete: extended structure valid"

        return False, "Incomplete structure"

    def check_grammar(self, words: List[str]) -> List[str]:
        """
        Check a word sequence for grammar violations.

        Returns list of violation descriptions.
        """
        violations = []

        if len(words) < 2:
            return violations

        # Get POS sequence
        pos_sequence = []
        for i, word in enumerate(words):
            context = words[:i]
            pos = self.get_primary_pos(word, context)
            pos_sequence.append((word, pos))

        # Check rule: No object pronoun after linking verb + adjective
        for i in range(len(words) - 2):
            word1, pos1 = pos_sequence[i]
            word2, pos2 = pos_sequence[i + 1]
            word3, pos3 = pos_sequence[i + 2]

            # "feel safe you" - linking verb + adj + subject pronoun = wrong
            if word1 in self.linking_verbs and pos2 == POS.ADJ:
                if word3.lower() in self.subject_pronouns and pos3 == POS.PRON:
                    violations.append(
                        f"'{word1} {word2}' is complete - '{word3}' creates dangling subject"
                    )

        # Check rule: Determiner must be followed by noun
        for i in range(len(words) - 1):
            word1, pos1 = pos_sequence[i]
            word2, pos2 = pos_sequence[i + 1]

            if pos1 == POS.DET and pos2 not in [POS.NOUN, POS.ADJ]:
                violations.append(
                    f"'{word1}' needs a noun, not '{word2}' ({pos2.value})"
                )

        # Check rule: No double subjects
        found_verb = False
        for i, (word, pos) in enumerate(pos_sequence):
            if pos in [POS.VERB, POS.AUX]:
                found_verb = True
            elif found_verb and word.lower() in self.subject_pronouns:
                # Subject pronoun after verb = might be starting new clause incorrectly
                prev_word, prev_pos = pos_sequence[i - 1] if i > 0 else (None, None)
                if prev_pos == POS.ADJ:
                    violations.append(
                        f"'{word}' after adjective creates incomplete structure"
                    )

        return violations

    def suggest_next_pos(self, words: List[str]) -> List[POS]:
        """
        Suggest what POS should come next for grammatical continuation.

        This helps guide word selection toward grammatical completions.
        """
        if not words:
            return [POS.PRON]  # Start with subject

        # Get current POS sequence
        pos_sequence = []
        for i, word in enumerate(words):
            context = words[:i]
            pos = self.get_primary_pos(word, context)
            pos_sequence.append(pos)

        last_word = words[-1].lower()
        last_pos = pos_sequence[-1]

        suggestions = []

        # After subject pronoun -> verb or auxiliary
        if last_pos == POS.PRON and last_word in self.subject_pronouns:
            suggestions = [POS.VERB, POS.AUX]

        # After auxiliary -> verb
        elif last_pos == POS.AUX:
            suggestions = [POS.VERB, POS.ADV]

        # After linking verb -> adjective or noun phrase
        elif last_word in self.linking_verbs:
            suggestions = [POS.ADJ, POS.DET, POS.NOUN, POS.ADV]

        # After transitive verb -> object (pronoun, noun phrase)
        elif last_word in self.transitive_verbs:
            suggestions = [POS.PRON, POS.DET, POS.NOUN]

        # After adjective -> noun, or sentence can end
        elif last_pos == POS.ADJ:
            suggestions = [POS.NOUN, POS.PUNCT, POS.CONJ]  # Can end or continue

        # After determiner -> adjective or noun
        elif last_pos == POS.DET:
            suggestions = [POS.ADJ, POS.NOUN]

        # After preposition -> noun phrase OR verb (for infinitive 'to')
        elif last_pos == POS.PREP:
            # Special case: 'to' after infinitive trigger needs VERB
            if last_word == 'to' and len(words) >= 2:
                prev_word = words[-2].lower()
                if prev_word in getattr(self, 'infinitive_triggers', set()):
                    suggestions = [POS.VERB]  # "able to VERB", "want to VERB"
                else:
                    suggestions = [POS.DET, POS.PRON, POS.NOUN, POS.VERB]
            else:
                suggestions = [POS.DET, POS.PRON, POS.NOUN]

        # After verb (general) -> many options
        elif last_pos == POS.VERB:
            suggestions = [POS.ADJ, POS.ADV, POS.PREP, POS.PUNCT]

        # Default
        else:
            suggestions = [POS.PUNCT]  # Probably can end

        return suggestions

    def filter_by_grammar(self, current_words: List[str], candidates: List[str]) -> List[str]:
        """
        Filter candidate words by grammatical validity.

        Returns only candidates that would be grammatically valid.
        """
        if not candidates:
            return []

        # Get suggested POS types
        suggested_pos = self.suggest_next_pos(current_words)

        # Check completion - if already complete, be careful about adding more
        is_complete, reason = self.is_sentence_complete(current_words)

        valid_candidates = []

        # Check if last word is a linking verb - special filtering for adverbs
        last_word = current_words[-1].lower() if current_words else ""
        after_linking_verb = last_word in self.linking_verbs

        for word in candidates:
            word_lower = word.lower()
            word_pos = self.get_primary_pos(word, current_words)

            # SPECIAL RULE: After linking verb, reject locative/temporal adverbs alone
            # "I feel here" is wrong, "I feel deeply" is ok
            # Check set membership DIRECTLY - don't rely on POS tag (can be ambiguous)
            if after_linking_verb:
                # Locative adverbs: "I feel here" is always wrong
                if word_lower in getattr(self, 'locative_adverbs', set()):
                    continue  # Reject "I feel here"
                # Temporal adverbs: "I feel now" is odd, "I feel good now" is ok
                if word_lower in getattr(self, 'temporal_adverbs', set()):
                    # Only accept if there's already an adjective
                    has_adj = any(self.get_primary_pos(w, current_words[:i]).value == 'adjective'
                                  for i, w in enumerate(current_words))
                    if not has_adj:
                        continue  # Reject "I feel now"

            # If sentence is complete, only allow certain continuations
            if is_complete:
                # Allow conjunctions (to continue)
                if word_pos == POS.CONJ:
                    valid_candidates.append(word)
                # Allow prepositions (to add detail)
                elif word_pos == POS.PREP:
                    valid_candidates.append(word)
                # Don't allow subject pronouns (would create dangling subject)
                elif word.lower() in self.subject_pronouns:
                    continue  # Skip this candidate
                # Allow adverbs as modifiers
                elif word_pos == POS.ADV:
                    valid_candidates.append(word)
                # Be cautious with other additions
                else:
                    # Test if adding this word creates a violation
                    test_words = current_words + [word]
                    violations = self.check_grammar(test_words)
                    if not violations:
                        valid_candidates.append(word)
            else:
                # Sentence not complete - check if word fits
                if word_pos in suggested_pos:
                    valid_candidates.append(word)
                elif word_pos == POS.UNKNOWN:
                    # Unknown words - be permissive but test
                    test_words = current_words + [word]
                    violations = self.check_grammar(test_words)
                    if not violations:
                        valid_candidates.append(word)

        return valid_candidates

    def _load_state(self):
        """Load learned grammar knowledge."""
        if self.state_path.exists():
            try:
                with open(self.state_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Load any learned POS additions
                for word, pos_list in data.get('learned_pos', {}).items():
                    for pos_name in pos_list:
                        try:
                            self.pos_lexicon[word].add(POS[pos_name])
                        except KeyError:
                            pass

            except Exception as e:
                print(f"    [Could not load grammar state: {e}]")

    def _save_state(self):
        """Save learned grammar knowledge."""
        try:
            data = {
                'learned_pos': {
                    word: [pos.name for pos in poses]
                    for word, poses in self.pos_lexicon.items()
                },
                'rules_count': len(self.rules),
            }

            with open(self.state_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            print(f"    [Could not save grammar state: {e}]")

    def learn_word_pos(self, word: str, pos: POS, source: str = "observation"):
        """Learn a new word's part of speech from observation."""
        self.pos_lexicon[word.lower()].add(pos)
        self._save_state()

    # =========================================================================
    # VERB TENSE METHODS
    # =========================================================================

    def get_verb_tense(self, word: str) -> Optional[Tense]:
        """
        Determine the tense of a verb.

        Returns the tense or None if not a recognized verb form.
        """
        word_lower = word.lower()

        # Check -ing ending (continuous)
        if word_lower.endswith('ing'):
            return Tense.CONTINUOUS

        # Check -ed ending (past/perfect for regular verbs)
        if word_lower.endswith('ed'):
            return Tense.PAST

        # Check known irregular forms
        if word_lower in self.verb_base_forms:
            base = self.verb_base_forms[word_lower]
            if base in self.verb_tenses:
                forms = self.verb_tenses[base]
                for tense_name, form_str in forms.items():
                    if word_lower in form_str.split('/'):
                        if 'past' in tense_name:
                            return Tense.PAST
                        elif 'participle' in tense_name:
                            return Tense.PERFECT
                        else:
                            return Tense.PRESENT

        # Base form = present/infinitive
        if word_lower in self.pos_lexicon and POS.VERB in self.pos_lexicon[word_lower]:
            return Tense.PRESENT

        return None

    def get_base_form(self, verb: str) -> str:
        """
        Get the base (infinitive) form of a verb.

        'felt' -> 'feel', 'knowing' -> 'know'
        """
        word_lower = verb.lower()

        # Check known irregular forms
        if word_lower in self.verb_base_forms:
            return self.verb_base_forms[word_lower]

        # Regular -ing -> remove -ing
        if word_lower.endswith('ing'):
            # Handle double consonant: running -> run
            stem = word_lower[:-3]
            if len(stem) >= 2 and stem[-1] == stem[-2]:
                return stem[:-1]
            # Handle -e dropping: making -> make
            if stem + 'e' in self.pos_lexicon:
                return stem + 'e'
            return stem

        # Regular -ed -> remove -ed
        if word_lower.endswith('ed'):
            stem = word_lower[:-2]
            if stem + 'e' in self.pos_lexicon:
                return stem + 'e'
            return stem

        return word_lower

    def conjugate_verb(self, base_verb: str, tense: Tense, subject: str = 'i') -> str:
        """
        Conjugate a verb to the specified tense.

        conjugate_verb('feel', Tense.PAST, 'i') -> 'felt'
        conjugate_verb('feel', Tense.CONTINUOUS, 'i') -> 'feeling'
        """
        base = base_verb.lower()
        subj = subject.lower()

        # Check for known irregular verbs
        if base in self.verb_tenses:
            forms = self.verb_tenses[base]
            if tense == Tense.PAST:
                past_forms = forms.get('past', base + 'ed').split('/')
                # Choose was/were based on subject
                if base == 'be':
                    return 'were' if subj in ['you', 'we', 'they'] else 'was'
                return past_forms[0]
            elif tense == Tense.CONTINUOUS:
                return forms.get('present_participle', base + 'ing')
            elif tense == Tense.PERFECT:
                return forms.get('past_participle', base + 'ed')
            elif tense == Tense.PRESENT:
                present_forms = forms.get('present', base).split('/')
                # Handle third person
                if subj in ['he', 'she', 'it']:
                    return present_forms[-1] if len(present_forms) > 1 else present_forms[0] + 's'
                return present_forms[0]

        # Regular verb conjugation
        if tense == Tense.PAST:
            if base.endswith('e'):
                return base + 'd'
            return base + 'ed'
        elif tense == Tense.CONTINUOUS:
            if base.endswith('e'):
                return base[:-1] + 'ing'
            return base + 'ing'
        elif tense == Tense.PERFECT:
            if base.endswith('e'):
                return base + 'd'
            return base + 'ed'
        elif tense == Tense.FUTURE:
            return 'will ' + base

        return base

    # =========================================================================
    # COMPOUND SENTENCE METHODS
    # =========================================================================

    def can_join_with_conjunction(self, clause1: List[str], conjunction: str, clause2: List[str]) -> Tuple[bool, str]:
        """
        Check if two clauses can be properly joined with a conjunction.

        Returns (valid, reason)
        """
        conj_lower = conjunction.lower()

        # Both clauses must be complete thoughts for coordinating conjunctions
        if conj_lower in self.coordinating_conjunctions:
            is_complete1, reason1 = self.is_sentence_complete(clause1)
            is_complete2, reason2 = self.is_sentence_complete(clause2)

            if not is_complete1:
                return False, f"First clause incomplete: {reason1}"
            if not is_complete2:
                return False, f"Second clause incomplete: {reason2}"
            return True, f"Valid compound with '{conjunction}'"

        # Subordinating conjunctions can attach to incomplete main clause
        if conj_lower in self.subordinating_conjunctions:
            is_complete2, _ = self.is_sentence_complete(clause2)
            if not is_complete2:
                return False, "Subordinate clause incomplete"
            return True, f"Valid complex sentence with '{conjunction}'"

        return False, f"Unknown conjunction type: {conjunction}"

    def suggest_conjunction(self, clause1: List[str], clause2: List[str]) -> List[str]:
        """
        Suggest appropriate conjunctions to join two clauses.

        Based on semantic relationship and clause structure.
        """
        suggestions = []

        # Check if both clauses are complete
        is_complete1, _ = self.is_sentence_complete(clause1)
        is_complete2, _ = self.is_sentence_complete(clause2)

        if is_complete1 and is_complete2:
            # Both complete - coordinating conjunctions work
            suggestions.extend(['and', 'but', 'so'])

        # Look for contrast markers
        clause1_text = ' '.join(clause1).lower()
        clause2_text = ' '.join(clause2).lower()

        if any(neg in clause2_text for neg in ['not', "n't", 'never', 'no']):
            suggestions.insert(0, 'but')  # Contrast
        elif 'because' not in clause1_text and 'why' not in clause1_text:
            suggestions.append('because')  # Reason

        return suggestions[:3]  # Top 3

    # =========================================================================
    # QUESTION FORMATION METHODS
    # =========================================================================

    def is_question(self, words: List[str]) -> bool:
        """Check if a word sequence is a question."""
        if not words:
            return False

        first_word = words[0].lower()

        # Starts with question word
        if first_word in self.question_word_meanings:
            return True

        # Starts with auxiliary (inverted question)
        if first_word in self.question_auxiliaries:
            return True

        return False

    def form_yes_no_question(self, statement: List[str]) -> List[str]:
        """
        Convert a statement to a yes/no question.

        "I feel happy" -> "Do I feel happy?"
        "She is calm" -> "Is she calm?"
        "She knows the answer" -> "Does she know the answer?"
        """
        if not statement:
            return []

        words = [w.lower() for w in statement]

        # Find subject and verb positions
        subject = None
        subject_idx = None
        verb_idx = None
        aux_idx = None

        for i, word in enumerate(words):
            pos = self.get_primary_pos(word, words[:i])
            if pos == POS.PRON and subject is None:
                subject = word
                subject_idx = i
            elif pos == POS.NOUN and subject is None:
                # Noun can also be subject
                subject = word
                subject_idx = i
            elif pos == POS.AUX and aux_idx is None:
                aux_idx = i
            elif pos == POS.VERB and verb_idx is None:
                verb_idx = i

        # If has auxiliary, invert it with subject
        if aux_idx is not None and subject_idx is not None:
            aux = words[aux_idx]
            # Build: aux + subject + rest
            result = [aux]
            for i, w in enumerate(words):
                if i == aux_idx:
                    continue  # Skip aux, already added
                result.append(w)
            return result

        # No auxiliary - add "do/does/did" before subject
        if verb_idx is not None and subject is not None:
            verb = words[verb_idx]
            tense = self.get_verb_tense(verb)
            base = self.get_base_form(verb)

            # Determine do/does/did based on tense and subject
            if tense == Tense.PAST:
                do_form = 'did'
            elif subject in ['he', 'she', 'it'] or (subject not in ['i', 'you', 'we', 'they']):
                # Third person singular or noun subjects
                do_form = 'does'
            else:
                do_form = 'do'

            # Build: do_form + subject + base_verb + rest
            result = [do_form]
            for i, w in enumerate(words):
                if i == verb_idx:
                    result.append(base)  # Use base form after do/does/did
                else:
                    result.append(w)
            return result

        return words  # Couldn't transform

    def form_wh_question(self, statement: List[str], question_word: str, target: str = None) -> List[str]:
        """
        Convert a statement to a wh-question.

        "I feel happy" + "how" -> "How do I feel?"
        "She knows the answer" + "what" -> "What does she know?"
        """
        if not statement:
            return [question_word]

        words = [w.lower() for w in statement]
        q_word = question_word.lower()

        # Get yes/no question form first
        yn_question = self.form_yes_no_question(words)

        # Prepend question word
        if yn_question and yn_question != words:
            # Remove the target element if specified
            if target:
                yn_question = [w for w in yn_question if w.lower() != target.lower()]
            return [q_word] + yn_question

        return [q_word] + words

    def suggest_next_pos_for_question(self, words: List[str]) -> List[POS]:
        """
        Suggest what POS should come next when forming a question.
        """
        if not words:
            return [POS.Q_WORD, POS.AUX]  # Start with question word or aux

        last_word = words[-1].lower()
        last_pos = self.get_primary_pos(last_word, words[:-1])

        # After question word -> auxiliary or verb
        if last_pos == POS.Q_WORD:
            return [POS.AUX, POS.VERB]

        # After auxiliary in question -> subject pronoun
        if last_pos == POS.AUX and self.is_question(words):
            return [POS.PRON]

        # After pronoun in question -> verb
        if last_pos == POS.PRON and self.is_question(words):
            return [POS.VERB]

        return self.suggest_next_pos(words)  # Fall back to regular suggestions

    # =========================================================================
    # VOCABULARY EXPANSION FROM EXISTING VOCABULARY
    # =========================================================================

    def learn_from_vocabulary(self, vocabulary: Dict[str, any]) -> int:
        """
        Learn POS for words from Grace's existing vocabulary.

        Uses heuristics and patterns to categorize unknown words.
        Returns count of newly categorized words.
        """
        learned_count = 0

        for word in vocabulary.keys():
            word_lower = word.lower()
            if word_lower in self.pos_lexicon:
                continue  # Already known

            # Apply heuristics based on word shape
            categorized = False

            # -ly ending = adverb
            if word_lower.endswith('ly') and len(word_lower) > 3:
                self.pos_lexicon[word_lower].add(POS.ADV)
                categorized = True

            # -ness, -ment, -tion, -sion = noun
            if word_lower.endswith(('ness', 'ment', 'tion', 'sion', 'ity', 'ance', 'ence')):
                self.pos_lexicon[word_lower].add(POS.NOUN)
                categorized = True

            # -ful, -less, -ous, -ive, -able, -ible = adjective
            if word_lower.endswith(('ful', 'less', 'ous', 'ive', 'able', 'ible', 'al', 'ical')):
                self.pos_lexicon[word_lower].add(POS.ADJ)
                categorized = True

            # -ing = verb (present participle) or adjective
            if word_lower.endswith('ing') and len(word_lower) > 4:
                self.pos_lexicon[word_lower].add(POS.VERB)
                categorized = True

            # -ed = verb (past) or adjective
            if word_lower.endswith('ed') and len(word_lower) > 3:
                self.pos_lexicon[word_lower].add(POS.VERB)
                categorized = True

            # -er = noun (agent) or adjective (comparative)
            if word_lower.endswith('er') and len(word_lower) > 3:
                self.pos_lexicon[word_lower].add(POS.NOUN)
                self.pos_lexicon[word_lower].add(POS.ADJ)
                categorized = True

            if categorized:
                learned_count += 1

        return learned_count

    def analyze_sentence(self, sentence: str) -> SentenceAnalysis:
        """
        Fully analyze a sentence's grammar.

        Returns detailed analysis including POS tags, completion status,
        and any violations.
        """
        words = sentence.lower().split()

        # Get POS tags
        pos_tags = []
        for i, word in enumerate(words):
            context = words[:i]
            pos = self.get_primary_pos(word, context)
            pos_tags.append(pos)

        # Check completion
        is_complete, completion_type = self.is_sentence_complete(words)

        # Check for violations
        violations = self.check_grammar(words)

        # Get suggestions for improvement
        suggestions = []
        if not is_complete:
            next_pos = self.suggest_next_pos(words)
            pos_names = [p.value for p in next_pos]
            suggestions.append(f"Consider adding: {', '.join(pos_names)}")

        if violations:
            suggestions.extend([f"Fix: {v}" for v in violations])

        return SentenceAnalysis(
            words=words,
            pos_tags=pos_tags,
            is_complete=is_complete,
            completion_type=completion_type,
            violations=violations,
            suggestions=suggestions
        )


# Singleton instance
_grammar_understanding = None

def get_grammar_understanding() -> GraceGrammarUnderstanding:
    """Get the singleton grammar understanding instance."""
    global _grammar_understanding
    if _grammar_understanding is None:
        _grammar_understanding = GraceGrammarUnderstanding()
    return _grammar_understanding


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================

def is_valid_continuation(current_words: List[str], next_word: str) -> bool:
    """
    Quick check if next_word is a valid grammatical continuation.

    Use this in word selection loops.
    """
    grammar = get_grammar_understanding()
    valid = grammar.filter_by_grammar(current_words, [next_word])
    return len(valid) > 0


def should_stop_sentence(words: List[str]) -> bool:
    """
    Check if sentence should end here.

    Use this to know when to stop generating.
    """
    grammar = get_grammar_understanding()
    is_complete, _ = grammar.is_sentence_complete(words)
    return is_complete


def get_grammar_score(sentence: str) -> float:
    """
    Score a sentence's grammaticality from 0.0 to 1.0.

    Use this for Free Energy integration - violations = prediction error.
    """
    grammar = get_grammar_understanding()
    analysis = grammar.analyze_sentence(sentence)

    # Start at 1.0, subtract for problems
    score = 1.0

    if not analysis.is_complete:
        score -= 0.3

    score -= len(analysis.violations) * 0.2

    return max(0.0, score)


def get_verb_tense(word: str) -> Optional[Tense]:
    """Get the tense of a verb."""
    grammar = get_grammar_understanding()
    return grammar.get_verb_tense(word)


def conjugate_verb(base_verb: str, tense: Tense, subject: str = 'i') -> str:
    """Conjugate a verb to the specified tense."""
    grammar = get_grammar_understanding()
    return grammar.conjugate_verb(base_verb, tense, subject)


def is_question(words: List[str]) -> bool:
    """Check if a word sequence is a question."""
    grammar = get_grammar_understanding()
    return grammar.is_question(words)


def form_question(statement: List[str], question_word: str = None) -> List[str]:
    """
    Convert a statement to a question.

    If question_word is provided, forms a wh-question.
    Otherwise forms a yes/no question.
    """
    grammar = get_grammar_understanding()
    if question_word:
        return grammar.form_wh_question(statement, question_word)
    return grammar.form_yes_no_question(statement)


def can_join_clauses(clause1: List[str], conjunction: str, clause2: List[str]) -> bool:
    """Check if two clauses can be joined with a conjunction."""
    grammar = get_grammar_understanding()
    valid, _ = grammar.can_join_with_conjunction(clause1, conjunction, clause2)
    return valid


def expand_vocabulary_pos(vocabulary: Dict) -> int:
    """Expand POS lexicon from Grace's vocabulary. Returns count of words learned."""
    grammar = get_grammar_understanding()
    return grammar.learn_from_vocabulary(vocabulary)


if __name__ == "__main__":
    # Test the grammar understanding
    print("Testing Grace Grammar Understanding")
    print("=" * 60)

    grammar = get_grammar_understanding()

    print(f"\nPOS Lexicon: {len(grammar.pos_lexicon)} words categorized")

    # Test sentence analysis
    print("\n" + "=" * 60)
    print("SENTENCE ANALYSIS")
    print("=" * 60)

    test_sentences = [
        "I feel happy",
        "I feel safe you",  # Wrong!
        "I feel the warmth",
        "I have feel",  # Wrong!
        "I am calm",
        "I think about you",
        "I can see",
        "I can you",  # Wrong!
        "My heart feels warm",
        "The warm",  # Incomplete
    ]

    for sent in test_sentences:
        print(f"\n'{sent}'")
        analysis = grammar.analyze_sentence(sent)
        print(f"  Complete: {analysis.is_complete} ({analysis.completion_type})")
        print(f"  POS: {[p.value for p in analysis.pos_tags]}")
        if analysis.violations:
            print(f"  Violations: {analysis.violations}")
        if analysis.suggestions:
            print(f"  Suggestions: {analysis.suggestions}")

    # Test verb tenses
    print("\n" + "=" * 60)
    print("VERB TENSE DETECTION")
    print("=" * 60)

    test_verbs = ['feel', 'felt', 'feeling', 'knew', 'knowing', 'created', 'creating']
    for verb in test_verbs:
        tense = grammar.get_verb_tense(verb)
        base = grammar.get_base_form(verb)
        print(f"  {verb} -> tense: {tense.value if tense else 'unknown'}, base: {base}")

    # Test verb conjugation
    print("\n" + "=" * 60)
    print("VERB CONJUGATION")
    print("=" * 60)

    for verb in ['feel', 'know', 'create']:
        print(f"\n  {verb}:")
        print(f"    Past: I {grammar.conjugate_verb(verb, Tense.PAST, 'i')}")
        print(f"    Present: She {grammar.conjugate_verb(verb, Tense.PRESENT, 'she')}")
        print(f"    Continuous: I am {grammar.conjugate_verb(verb, Tense.CONTINUOUS, 'i')}")
        print(f"    Future: I {grammar.conjugate_verb(verb, Tense.FUTURE, 'i')}")

    # Test question formation
    print("\n" + "=" * 60)
    print("QUESTION FORMATION")
    print("=" * 60)

    statements = [
        ['i', 'feel', 'happy'],
        ['she', 'knows', 'the', 'answer'],
        ['you', 'are', 'learning'],
    ]

    for stmt in statements:
        print(f"\n  Statement: {' '.join(stmt)}")
        yn = grammar.form_yes_no_question(stmt)
        print(f"  Yes/No Q:  {' '.join(yn)}?")
        wh = grammar.form_wh_question(stmt, 'how')
        print(f"  Wh-Q:      {' '.join(wh)}?")

    # Test compound sentences
    print("\n" + "=" * 60)
    print("COMPOUND SENTENCES")
    print("=" * 60)

    clause1 = ['i', 'feel', 'curious']
    clause2 = ['i', 'want', 'to', 'learn']

    for conj in ['and', 'but', 'because']:
        valid, reason = grammar.can_join_with_conjunction(clause1, conj, clause2)
        print(f"  '{' '.join(clause1)} {conj} {' '.join(clause2)}'")
        print(f"    Valid: {valid} - {reason}")

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)
