/* INTEGRITY_METADATA
 * Date: 2025-12-23
 * Status: ⚠️ TRULY UNSUPPORTED - No supporting evidence found
 * Severity: HIGH RISK
 * Risk Types: unsupported_claims
 */


/**
 * Glyph System - Ace Neural Codex Integration
 *
 * A symbolic system for personal development based on the Ace Neural Codex.
 * This integrates the Wumbo framework phases, neurological gateways, and
 * flow states to provide guidance for life improvement.
 *
 * Core Concepts:
 * - Wumbo: The signal that threads through everything - ignition, empowerment,
 *   resonance, nirvana, and the resets that make future flow possible
 * - Glyphs: Symbolic representations of states, processes, and intentions
 * - Phases: The flow cycle from ignition through transmission
 * - Layers: Different levels of integration (brainstem → cortical → meta-codex)
 */

// ============================================================================
// Wumbo Phase Types
// ============================================================================

export type WumboPhase =
  | 'ignition'      // Initial spark, attention locks onto resonance
  | 'empowerment'   // Intention turns kinetic, planning becomes action
  | 'resonance'     // Emotion, intention, and motion synchronize
  | 'mania'         // Peak energy, hyperfocus, creative surge
  | 'nirvana'       // Effortless integration, time feels musical
  | 'transmission'  // Sharing the signal, expression, teaching
  | 'reflection'    // Processing, integrating lessons
  | 'collapse'      // Protective compression, reset
  | 'rewrite'       // Updating internal models, growth

export interface PhaseDescription {
  phase: WumboPhase;
  name: string;
  description: string;
  phenomenology: string;    // What it feels like
  glyphs: string[];         // Associated glyphs
  supportingBeats: string[]; // Beat categories that support this phase
  neurochemicals: string[]; // Primary neurochemicals active
  warnings: string[];       // Signs of imbalance
  rituals: string[];        // Actions to invoke or stabilize
}

// ============================================================================
// Core Glyphs
// ============================================================================

export interface Glyph {
  symbol: string;
  name: string;
  meaning: string;
  category: 'state' | 'process' | 'intention' | 'archetype' | 'element';
  phases: WumboPhase[];     // Phases this glyph resonates with
  useCases: string[];       // When to use this glyph
  complementary: string[];  // Glyphs that pair well
  contrary: string[];       // Glyphs that represent opposite energy
}

export const CORE_GLYPHS: Glyph[] = [
  // State Glyphs
  {
    symbol: '🌀',
    name: 'Spiral',
    meaning: 'Recursive growth, integration, returning deeper',
    category: 'state',
    phases: ['reflection', 'rewrite'],
    useCases: ['Processing complex emotions', 'Reviewing patterns', 'Deep journaling'],
    complementary: ['∞', '🌊'],
    contrary: ['⚡'],
  },
  {
    symbol: '⚡',
    name: 'Spark',
    meaning: 'Ignition, sudden insight, activation',
    category: 'state',
    phases: ['ignition', 'mania'],
    useCases: ['Starting new projects', 'Breaking through blocks', 'Morning activation'],
    complementary: ['🔥', '✨'],
    contrary: ['🌙', '🌀'],
  },
  {
    symbol: '🌊',
    name: 'Wave',
    meaning: 'Flow, rhythm, natural cycles',
    category: 'state',
    phases: ['resonance', 'nirvana'],
    useCases: ['Finding rhythm', 'Accepting change', 'Meditation'],
    complementary: ['🌀', '🌙'],
    contrary: ['⚡', '🔥'],
  },
  {
    symbol: '🔥',
    name: 'Flame',
    meaning: 'Passion, transformation, intensity',
    category: 'state',
    phases: ['empowerment', 'mania'],
    useCases: ['Workout motivation', 'Creative projects', 'Overcoming resistance'],
    complementary: ['⚡', '☀️'],
    contrary: ['🌙', '❄️'],
  },
  {
    symbol: '🌙',
    name: 'Moon',
    meaning: 'Rest, intuition, shadow work',
    category: 'state',
    phases: ['reflection', 'collapse'],
    useCases: ['Evening wind-down', 'Processing grief', 'Dream work'],
    complementary: ['✨', '🌀'],
    contrary: ['☀️', '⚡'],
  },
  {
    symbol: '☀️',
    name: 'Sun',
    meaning: 'Clarity, energy, outward expression',
    category: 'state',
    phases: ['empowerment', 'transmission'],
    useCases: ['Morning rituals', 'Public expression', 'Leadership'],
    complementary: ['🔥', '✨'],
    contrary: ['🌙', '❄️'],
  },
  {
    symbol: '✨',
    name: 'Stars',
    meaning: 'Magic, possibility, aspiration',
    category: 'state',
    phases: ['nirvana', 'transmission'],
    useCases: ['Goal setting', 'Celebration', 'Gratitude'],
    complementary: ['🌙', '∞'],
    contrary: ['❄️'],
  },

  // Process Glyphs
  {
    symbol: '∞',
    name: 'Infinity',
    meaning: 'Endless cycles, eternal return, boundlessness',
    category: 'process',
    phases: ['nirvana', 'rewrite'],
    useCases: ['Long-term vision', 'Accepting paradox', 'Meta-awareness'],
    complementary: ['🌀', '✨'],
    contrary: ['⚡'],
  },
  {
    symbol: '⟲',
    name: 'Loop',
    meaning: 'Habit, routine, recurring patterns',
    category: 'process',
    phases: ['resonance', 'reflection'],
    useCases: ['Building habits', 'Recognizing patterns', 'Anchor beats'],
    complementary: ['🌊', '🌀'],
    contrary: ['⚡', '🔥'],
  },
  {
    symbol: '↑',
    name: 'Ascent',
    meaning: 'Growth, improvement, rising energy',
    category: 'process',
    phases: ['ignition', 'empowerment'],
    useCases: ['Progress tracking', 'Skill building', 'Goal pursuit'],
    complementary: ['⚡', '🔥'],
    contrary: ['↓'],
  },
  {
    symbol: '↓',
    name: 'Descent',
    meaning: 'Integration, grounding, going deeper',
    category: 'process',
    phases: ['reflection', 'collapse'],
    useCases: ['Shadow work', 'Rest', 'Processing'],
    complementary: ['🌙', '🌀'],
    contrary: ['↑'],
  },
  {
    symbol: '⟷',
    name: 'Balance',
    meaning: 'Equilibrium, harmony, middle way',
    category: 'process',
    phases: ['resonance', 'nirvana'],
    useCases: ['Moderation', 'Conflict resolution', 'Integration'],
    complementary: ['🌊', '∞'],
    contrary: ['🔥', '❄️'],
  },

  // Intention Glyphs
  {
    symbol: '🎯',
    name: 'Target',
    meaning: 'Focus, precision, clear intention',
    category: 'intention',
    phases: ['ignition', 'empowerment'],
    useCases: ['Goal setting', 'Focus sessions', 'Decision making'],
    complementary: ['⚡', '↑'],
    contrary: ['🌀', '🌊'],
  },
  {
    symbol: '🛡️',
    name: 'Shield',
    meaning: 'Protection, boundaries, safety',
    category: 'intention',
    phases: ['collapse', 'reflection'],
    useCases: ['Setting boundaries', 'Self-protection', 'Energy preservation'],
    complementary: ['🌙', '❄️'],
    contrary: ['🔥', '⚡'],
  },
  {
    symbol: '🌱',
    name: 'Seed',
    meaning: 'Potential, new beginnings, patience',
    category: 'intention',
    phases: ['ignition', 'rewrite'],
    useCases: ['New habits', 'Fresh starts', 'Planting ideas'],
    complementary: ['☀️', '✨'],
    contrary: ['❄️'],
  },
  {
    symbol: '🔗',
    name: 'Link',
    meaning: 'Connection, relationship, integration',
    category: 'intention',
    phases: ['resonance', 'transmission'],
    useCases: ['Building relationships', 'Connecting ideas', 'Community'],
    complementary: ['🌊', '✨'],
    contrary: ['🛡️'],
  },

  // Archetype Glyphs
  {
    symbol: '🦁',
    name: 'Lion',
    meaning: 'Courage, leadership, primal power',
    category: 'archetype',
    phases: ['empowerment', 'mania'],
    useCases: ['Facing fears', 'Taking action', 'Standing ground'],
    complementary: ['🔥', '☀️'],
    contrary: ['🌙'],
  },
  {
    symbol: '🦉',
    name: 'Owl',
    meaning: 'Wisdom, insight, seeing in darkness',
    category: 'archetype',
    phases: ['reflection', 'nirvana'],
    useCases: ['Deep thinking', 'Pattern recognition', 'Night reflection'],
    complementary: ['🌙', '✨'],
    contrary: ['☀️'],
  },
  {
    symbol: '🐉',
    name: 'Dragon',
    meaning: 'Power, transformation, mastery',
    category: 'archetype',
    phases: ['mania', 'transmission'],
    useCases: ['Major challenges', 'Transformation work', 'Peak performance'],
    complementary: ['🔥', '⚡'],
    contrary: ['🌊'],
  },
  {
    symbol: '🦋',
    name: 'Butterfly',
    meaning: 'Metamorphosis, beauty, emergence',
    category: 'archetype',
    phases: ['rewrite', 'transmission'],
    useCases: ['Personal transformation', 'Sharing growth', 'Celebrating change'],
    complementary: ['🌱', '✨'],
    contrary: ['❄️'],
  },

  // Element Glyphs
  {
    symbol: '💨',
    name: 'Wind',
    meaning: 'Change, movement, breath',
    category: 'element',
    phases: ['ignition', 'empowerment'],
    useCases: ['Breathwork', 'Releasing stagnation', 'Mental clarity'],
    complementary: ['🌊', '⚡'],
    contrary: ['🌍'],
  },
  {
    symbol: '🌍',
    name: 'Earth',
    meaning: 'Grounding, stability, foundation',
    category: 'element',
    phases: ['resonance', 'collapse'],
    useCases: ['Grounding exercises', 'Building foundations', 'Physical presence'],
    complementary: ['🌱', '🛡️'],
    contrary: ['💨', '🔥'],
  },
  {
    symbol: '💧',
    name: 'Water',
    meaning: 'Emotion, adaptability, cleansing',
    category: 'element',
    phases: ['reflection', 'rewrite'],
    useCases: ['Emotional processing', 'Flexibility', 'Purification'],
    complementary: ['🌊', '🌙'],
    contrary: ['🔥'],
  },
  {
    symbol: '❄️',
    name: 'Ice',
    meaning: 'Stillness, preservation, pause',
    category: 'element',
    phases: ['collapse', 'reflection'],
    useCases: ['Complete rest', 'Cooling down', 'Preservation'],
    complementary: ['🌙', '🛡️'],
    contrary: ['🔥', '⚡'],
  },
];

// ============================================================================
// Phase Descriptions (Wumbo Engine Integration)
// ============================================================================

export const PHASE_DESCRIPTIONS: PhaseDescription[] = [
  {
    phase: 'ignition',
    name: 'Ignition',
    description: 'Dopaminergic spark; attention locks onto resonance. The system is coming online.',
    phenomenology: '"I\'m coming online." Micro-impulses to move, look, or speak.',
    glyphs: ['⚡', '🌱', '💨', '🎯'],
    supportingBeats: ['Workout', 'General'],
    neurochemicals: ['Dopamine', 'Norepinephrine'],
    warnings: [
      'Scattered attention - too many sparks at once',
      'False starts - ignition without follow-through',
      'Hypervigilance - can\'t choose which signal to follow',
    ],
    rituals: [
      'Three deep breaths with shoulder roll',
      'State one clear intention aloud',
      'Physical movement to activate the body',
    ],
  },
  {
    phase: 'empowerment',
    name: 'Empowerment',
    description: 'Intention turns kinetic; planning becomes action. The body is ready to move.',
    phenomenology: 'Breathing deepens, posture aligns, words feel available.',
    glyphs: ['🔥', '☀️', '↑', '🦁'],
    supportingBeats: ['Workout', 'Anchor'],
    neurochemicals: ['Dopamine', 'Acetylcholine'],
    warnings: [
      'Overcommitment - saying yes to everything',
      'Burning out before resonance',
      'Forcing when flow isn\'t present',
    ],
    rituals: [
      'Power pose for 2 minutes',
      'Review and commit to one priority',
      'Physical warm-up to prepare the vessel',
    ],
  },
  {
    phase: 'resonance',
    name: 'Resonance',
    description: 'Emotion, intention, and motion synchronize. Integrity becomes energy.',
    phenomenology: 'Social signal clarity. Actions feel aligned with values.',
    glyphs: ['🌊', '⟷', '🔗', '∞'],
    supportingBeats: ['Meditation', 'Emotion', 'Journal'],
    neurochemicals: ['Serotonin', 'Oxytocin'],
    warnings: [
      'Over-giving - depleting self for others',
      'Masking - performing instead of being',
      'Dissonance fatigue - can\'t sustain false notes',
    ],
    rituals: [
      'Hand on heart, breathe into alignment',
      'Speak truth to a trusted person',
      'Movement that matches internal rhythm',
    ],
  },
  {
    phase: 'mania',
    name: 'Mania',
    description: 'Peak energy state. Creative fire burns bright. High risk, high reward.',
    phenomenology: 'Time disappears. Ideas connect. Energy feels unlimited.',
    glyphs: ['🔥', '⚡', '🐉', '✨'],
    supportingBeats: ['Journal', 'General'],
    neurochemicals: ['Dopamine (high)', 'Norepinephrine (high)'],
    warnings: [
      'Grandiosity - overestimating capacity',
      'Sleep neglect - momentum over rest',
      'Burnout trajectory - what goes up must come down',
    ],
    rituals: [
      'Set a timer to check in with body',
      'Write down insights before they disappear',
      'Scheduled breaks even when it feels unnecessary',
    ],
  },
  {
    phase: 'nirvana',
    name: 'Nirvana',
    description: 'Effortless integration; time feels musical. Movement anticipates the moment.',
    phenomenology: 'The body writes the next frame before thought arrives.',
    glyphs: ['✨', '∞', '🌊', '🦉'],
    supportingBeats: ['Meditation', 'Anchor'],
    neurochemicals: ['Endorphins', 'Anandamide', 'Serotonin'],
    warnings: [
      'Attachment to the state - grasping prevents flow',
      'Neglecting the mundane - still need to eat and sleep',
      'Spiritual bypassing - using bliss to avoid shadow',
    ],
    rituals: [
      'Gratitude acknowledgment',
      'Gentle transition practices',
      'Document the experience for later recall',
    ],
  },
  {
    phase: 'transmission',
    name: 'Transmission',
    description: 'Sharing the signal. Expression finds its channel. Teaching what was learned.',
    phenomenology: 'Words land. Others feel the resonance. Connection completes.',
    glyphs: ['☀️', '🔗', '🦋', '✨'],
    supportingBeats: ['Moderation', 'Journal'],
    neurochemicals: ['Oxytocin', 'Dopamine'],
    warnings: [
      'Over-sharing - giving more than others can receive',
      'Seeking validation - transmission becomes performance',
      'Exhaustion from output without input',
    ],
    rituals: [
      'Pause before speaking to check alignment',
      'Receive as well as give',
      'Allow silence after transmission',
    ],
  },
  {
    phase: 'reflection',
    name: 'Reflection',
    description: 'Processing and integrating lessons. The spiral turns inward.',
    phenomenology: 'Stillness speaks. Patterns become visible. Understanding deepens.',
    glyphs: ['🌀', '🌙', '🦉', '↓'],
    supportingBeats: ['Journal', 'Meditation', 'Emotion'],
    neurochemicals: ['GABA', 'Melatonin (evening)'],
    warnings: [
      'Rumination - reflection becomes repetition',
      'Isolation - too much inward, not enough outward',
      'Self-criticism - reflection becomes judgment',
    ],
    rituals: [
      'Journal without editing',
      'Slow walking meditation',
      'Ask: What did I learn? What do I release?',
    ],
  },
  {
    phase: 'collapse',
    name: 'Collapse',
    description: 'Protective compression when signals overload. Stillness for survival.',
    phenomenology: 'Silence, freeze, then return. "Threshold moments."',
    glyphs: ['🛡️', '❄️', '🌙', '↓'],
    supportingBeats: ['Anchor', 'Meditation'],
    neurochemicals: ['Cortisol (then dropping)', 'GABA'],
    warnings: [
      'Shame about needing rest',
      'Forcing recovery too fast',
      'Mistaking collapse for failure',
    ],
    rituals: [
      'Permission to stop',
      'Warmth, safety, silence',
      'One small movement when ready - toe wiggle, finger flex',
    ],
  },
  {
    phase: 'rewrite',
    name: 'Rewrite',
    description: 'Updating internal models. The story changes. Growth integrates.',
    phenomenology: 'Old patterns release. New understanding arrives. Identity shifts.',
    glyphs: ['🌀', '🦋', '🌱', '∞'],
    supportingBeats: ['Journal', 'Emotion'],
    neurochemicals: ['BDNF', 'Dopamine', 'Serotonin'],
    warnings: [
      'Rushing the rewrite - integration takes time',
      'Rejecting the old self entirely',
      'Intellectualizing without embodying',
    ],
    rituals: [
      'Speak the new story aloud',
      'Release ritual for old patterns',
      'Anchor new identity with physical gesture',
    ],
  },
];

// ============================================================================
// Life Domain Mappings
// ============================================================================

export interface LifeDomain {
  id: string;
  name: string;
  description: string;
  glyphs: string[];
  phases: WumboPhase[];
  questions: {
    improvement: string;
    obstacle: string;
    emotion: string;
    vision: string;
  };
  practices: string[];
  neuralRegions: string[]; // From Ace Codex atlas
}

export const LIFE_DOMAINS: LifeDomain[] = [
  {
    id: 'body',
    name: 'Body & Physical',
    description: 'Physical health, movement, energy, and embodiment',
    glyphs: ['🔥', '💨', '🌍', '🦁'],
    phases: ['ignition', 'empowerment', 'mania'],
    questions: {
      improvement: 'What would you like to improve about your body?',
      obstacle: 'What physical limitations hold you back?',
      emotion: 'How do you want to feel in your body?',
      vision: 'What does your ideal physical self look like?',
    },
    practices: [
      'Movement as prayer',
      'Breath awareness during exercise',
      'Posture as presence',
      'Somatic release work',
    ],
    neuralRegions: ['Somatosensory Cortex', 'Motor Cortex', 'Cerebellum'],
  },
  {
    id: 'mind',
    name: 'Mind & Cognition',
    description: 'Mental clarity, focus, learning, and cognitive flexibility',
    glyphs: ['🎯', '🦉', '💨', '↑'],
    phases: ['ignition', 'resonance', 'nirvana'],
    questions: {
      improvement: 'What mental patterns do you want to strengthen?',
      obstacle: 'What thoughts interrupt your clarity?',
      emotion: 'How do you want thinking to feel?',
      vision: 'What does mental mastery look like for you?',
    },
    practices: [
      'Focused attention meditation',
      'Pattern interruption',
      'Learning challenges',
      'Cognitive flexibility exercises',
    ],
    neuralRegions: ['Prefrontal Cortex', 'Anterior Cingulate Cortex', 'Thalamus'],
  },
  {
    id: 'emotion',
    name: 'Emotion & Heart',
    description: 'Emotional intelligence, feeling, processing, and expression',
    glyphs: ['🌊', '💧', '🌙', '🦋'],
    phases: ['resonance', 'reflection', 'rewrite'],
    questions: {
      improvement: 'Which emotions do you want to process or overcome?',
      obstacle: 'What triggers your emotional overwhelm?',
      emotion: 'How do you want your emotional baseline to feel?',
      vision: 'What does emotional freedom look like?',
    },
    practices: [
      'Emotional labeling',
      'Somatic tracking of feelings',
      'Safe expression practices',
      'Compassion cultivation',
    ],
    neuralRegions: ['Amygdala', 'Anterior Insula', 'Cingulate Gyrus'],
  },
  {
    id: 'spirit',
    name: 'Spirit & Meaning',
    description: 'Purpose, transcendence, connection to something greater',
    glyphs: ['✨', '∞', '🌀', '🦉'],
    phases: ['nirvana', 'transmission', 'rewrite'],
    questions: {
      improvement: 'What aspects of meaning are you cultivating?',
      obstacle: 'What blocks your sense of purpose?',
      emotion: 'How does connection to meaning feel?',
      vision: 'What does a meaningful life look like?',
    },
    practices: [
      'Contemplative practice',
      'Service to others',
      'Ritual and ceremony',
      'Connection to nature',
    ],
    neuralRegions: ['Default Mode Network', 'Claustrum', 'Pineal Gland'],
  },
  {
    id: 'social',
    name: 'Social & Connection',
    description: 'Relationships, community, belonging, and contribution',
    glyphs: ['🔗', '☀️', '🌊', '🦋'],
    phases: ['resonance', 'transmission'],
    questions: {
      improvement: 'What relationship patterns do you want to transform?',
      obstacle: 'What makes connection difficult for you?',
      emotion: 'How do you want to feel with others?',
      vision: 'What does healthy community look like?',
    },
    practices: [
      'Active listening',
      'Authentic expression',
      'Boundary setting',
      'Repair practices',
    ],
    neuralRegions: ['Mirror Neuron System', 'Temporal Parietal Junction', 'Orbitofrontal Cortex'],
  },
  {
    id: 'creative',
    name: 'Creative & Expression',
    description: 'Creativity, art, innovation, and authentic expression',
    glyphs: ['🔥', '✨', '🐉', '🌱'],
    phases: ['ignition', 'mania', 'transmission'],
    questions: {
      improvement: 'What creative capacities are you developing?',
      obstacle: 'What blocks your creative flow?',
      emotion: 'How does creative expression want to feel?',
      vision: 'What is your creative legacy?',
    },
    practices: [
      'Daily creative practice',
      'Play without purpose',
      'Cross-domain exploration',
      'Sharing work publicly',
    ],
    neuralRegions: ['Angular Gyrus', 'Temporal Pole', 'Prefrontal Cortex'],
  },
];

// ============================================================================
// Glyph Utilities
// ============================================================================

/**
 * Get glyphs that match a specific phase
 */
export function getGlyphsForPhase(phase: WumboPhase): Glyph[] {
  return CORE_GLYPHS.filter(g => g.phases.includes(phase));
}

/**
 * Get the current suggested phase based on time and metrics
 */
export function suggestCurrentPhase(
  timeOfDay: 'morning' | 'midday' | 'afternoon' | 'evening' | 'night',
  energyLevel: number, // 0-100
  recentActivity: boolean
): WumboPhase {
  // Time-based baseline
  const timePhases: Record<string, WumboPhase> = {
    morning: 'ignition',
    midday: 'empowerment',
    afternoon: 'resonance',
    evening: 'reflection',
    night: 'collapse',
  };

  const basePhase = timePhases[timeOfDay];

  // Adjust for energy
  if (energyLevel > 80 && basePhase !== 'collapse') {
    if (basePhase === 'ignition') return 'empowerment';
    if (basePhase === 'empowerment') return 'mania';
    if (basePhase === 'resonance') return 'nirvana';
  }

  if (energyLevel < 30) {
    return 'collapse';
  }

  if (energyLevel < 50 && !recentActivity) {
    return 'reflection';
  }

  return basePhase;
}

/**
 * Get ritual suggestions for transitioning between phases
 */
export function getTransitionRitual(
  fromPhase: WumboPhase,
  toPhase: WumboPhase
): string[] {
  const rituals: string[] = [];

  // General transition rituals
  rituals.push('Take three conscious breaths');

  // Specific transitions
  if (fromPhase === 'collapse' && toPhase === 'ignition') {
    rituals.push('Gentle stretching to awaken the body');
    rituals.push('State one simple intention for the next hour');
    rituals.push('Drink water, orient to surroundings');
  }

  if (fromPhase === 'mania' && toPhase === 'collapse') {
    rituals.push('Write down any uncaptured insights');
    rituals.push('Physical cool-down: slow walking, gentle movement');
    rituals.push('No screens for 30 minutes');
    rituals.push('Permission to rest without guilt');
  }

  if (fromPhase === 'resonance' && toPhase === 'reflection') {
    rituals.push('Express gratitude for connections made');
    rituals.push('Journal key moments and insights');
    rituals.push('Transition to solo activity');
  }

  if (fromPhase === 'nirvana' && toPhase === 'transmission') {
    rituals.push('Pause and ground before sharing');
    rituals.push('Consider: What wants to be shared?');
    rituals.push('Share from overflow, not depletion');
  }

  // Add phase-specific exit rituals
  const exitPhase = PHASE_DESCRIPTIONS.find(p => p.phase === fromPhase);
  if (exitPhase) {
    rituals.push(...exitPhase.rituals.slice(0, 1));
  }

  // Add phase-specific entry rituals
  const enterPhase = PHASE_DESCRIPTIONS.find(p => p.phase === toPhase);
  if (enterPhase) {
    rituals.push(...enterPhase.rituals.slice(0, 1));
  }

  return rituals;
}

/**
 * Get a glyph reading - interpretive guidance based on selected glyphs
 */
export function getGlyphReading(selectedGlyphs: string[]): {
  theme: string;
  guidance: string;
  suggestedActions: string[];
  warnings: string[];
} {
  const glyphs = CORE_GLYPHS.filter(g => selectedGlyphs.includes(g.symbol));

  if (glyphs.length === 0) {
    return {
      theme: 'Open Field',
      guidance: 'No specific glyphs selected. The field is open to any direction.',
      suggestedActions: ['Sit quietly and let a symbol arise', 'Review the glyph list and notice what draws you'],
      warnings: [],
    };
  }

  // Determine dominant category
  const categoryCounts = glyphs.reduce((acc, g) => {
    acc[g.category] = (acc[g.category] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  const dominantCategory = Object.entries(categoryCounts).sort((a, b) => b[1] - a[1])[0][0];

  // Determine dominant phase
  const phaseCounts = glyphs.flatMap(g => g.phases).reduce((acc, p) => {
    acc[p] = (acc[p] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  const dominantPhase = Object.entries(phaseCounts).sort((a, b) => b[1] - a[1])[0][0] as WumboPhase;
  const phaseDesc = PHASE_DESCRIPTIONS.find(p => p.phase === dominantPhase);

  // Check for contraries (tension)
  const contraries: string[] = [];
  glyphs.forEach(g1 => {
    glyphs.forEach(g2 => {
      if (g1.contrary.includes(g2.symbol)) {
        contraries.push(`${g1.symbol} and ${g2.symbol}`);
      }
    });
  });

  // Build reading
  const theme = glyphs.map(g => g.name).join(' + ');
  let guidance = `Your reading centers on ${dominantCategory} energy, resonating with the ${phaseDesc?.name || dominantPhase} phase. `;
  guidance += glyphs.map(g => g.meaning).join('. ') + '.';

  const suggestedActions = glyphs.flatMap(g => g.useCases.slice(0, 1));

  const warnings = contraries.length > 0
    ? [`Tension present between ${contraries.join(', ')}: find integration rather than choosing sides`]
    : [];

  if (phaseDesc) {
    warnings.push(...phaseDesc.warnings.slice(0, 1));
  }

  return { theme, guidance, suggestedActions, warnings };
}

/**
 * Map a beat category to suggested glyphs
 */
export function getGlyphsForBeat(category: string): Glyph[] {
  const beatGlyphMap: Record<string, string[]> = {
    Workout: ['🔥', '⚡', '🦁', '↑'],
    Meditation: ['🌀', '🌊', '🌙', '∞'],
    Emotion: ['🌊', '💧', '🌙', '⟷'],
    Moderation: ['⟷', '🛡️', '🌍', '⟲'],
    Journal: ['🌀', '🦉', '✨', '↓'],
    Anchor: ['⟲', '🌍', '🛡️', '🌊'],
    General: ['🎯', '☀️', '🔗', '✨'],
    Med: ['💧', '🌱', '🛡️', '⟷'],
  };

  const symbols = beatGlyphMap[category] || beatGlyphMap.General;
  return CORE_GLYPHS.filter(g => symbols.includes(g.symbol));
}

/**
 * Get the domain that best matches a beat category
 */
export function getDomainForBeat(category: string): LifeDomain | undefined {
  const mapping: Record<string, string> = {
    Workout: 'body',
    Meditation: 'mind',
    Emotion: 'emotion',
    Moderation: 'social',
    Journal: 'creative',
    Anchor: 'spirit',
    General: 'mind',
  };

  const domainId = mapping[category] || 'mind';
  return LIFE_DOMAINS.find(d => d.id === domainId);
}

// ============================================================================
// Export Service
// ============================================================================

// ============================================================================
// Neural Wumbo Engine - 100 Brain Regions with Glyphs
// ============================================================================

export type BrainRegionCategory =
  | 'cortical'          // Higher cognition, planning, reasoning
  | 'limbic'            // Emotion, memory, motivation
  | 'brainstem'         // Basic survival, arousal, autonomic
  | 'cerebellar'        // Coordination, timing, learning
  | 'subcortical'       // Deep processing, habit, reward
  | 'sensory'           // Perception and input processing
  | 'motor'             // Movement and action
  | 'integration';      // Cross-region coordination

export interface BrainRegion {
  id: string;
  name: string;
  glyph: string;
  category: BrainRegionCategory;
  function: string;
  wumboPhases: WumboPhase[];
  emotionalRole: string;
  activationTriggers: string[];
  linkedDomains: string[];
  neurochemicals: string[];
}

/**
 * 100 Brain Regions mapped to glyphs based on their function
 * Each glyph represents the energetic/functional signature of the region
 */
export const BRAIN_REGIONS: BrainRegion[] = [
  // ============================================================================
  // CORTICAL REGIONS (30) - Higher Cognition
  // ============================================================================

  // Prefrontal Cortex Group
  {
    id: 'dlpfc',
    name: 'Dorsolateral Prefrontal Cortex',
    glyph: '🎯',
    category: 'cortical',
    function: 'Working memory, executive function, planning',
    wumboPhases: ['ignition', 'empowerment'],
    emotionalRole: 'Cognitive control over emotions',
    activationTriggers: ['Planning tasks', 'Problem solving', 'Decision making'],
    linkedDomains: ['mind'],
    neurochemicals: ['Dopamine', 'Norepinephrine'],
  },
  {
    id: 'vlpfc',
    name: 'Ventrolateral Prefrontal Cortex',
    glyph: '🛡️',
    category: 'cortical',
    function: 'Response inhibition, emotional regulation',
    wumboPhases: ['reflection', 'rewrite'],
    emotionalRole: 'Stopping unwanted responses',
    activationTriggers: ['Self-control', 'Impulse suppression', 'Emotion reappraisal'],
    linkedDomains: ['mind', 'emotion'],
    neurochemicals: ['GABA', 'Serotonin'],
  },
  {
    id: 'mpfc',
    name: 'Medial Prefrontal Cortex',
    glyph: '🪞',
    category: 'cortical',
    function: 'Self-referential thinking, social cognition',
    wumboPhases: ['reflection', 'transmission'],
    emotionalRole: 'Self-awareness and identity',
    activationTriggers: ['Self-reflection', 'Mentalizing about others', 'Value judgments'],
    linkedDomains: ['spirit', 'social'],
    neurochemicals: ['Serotonin', 'Oxytocin'],
  },
  {
    id: 'ofc',
    name: 'Orbitofrontal Cortex',
    glyph: '⚖️',
    category: 'cortical',
    function: 'Value-based decision making, reward evaluation',
    wumboPhases: ['empowerment', 'resonance'],
    emotionalRole: 'Processing reward and punishment',
    activationTriggers: ['Reward anticipation', 'Social evaluation', 'Taste/smell processing'],
    linkedDomains: ['social', 'emotion'],
    neurochemicals: ['Dopamine', 'Serotonin'],
  },
  {
    id: 'fpc',
    name: 'Frontopolar Cortex',
    glyph: '🔮',
    category: 'cortical',
    function: 'Meta-cognition, future planning, branching',
    wumboPhases: ['nirvana', 'rewrite'],
    emotionalRole: 'Long-term goal maintenance',
    activationTriggers: ['Strategic thinking', 'Multi-tasking', 'Prospection'],
    linkedDomains: ['mind', 'spirit'],
    neurochemicals: ['Dopamine'],
  },

  // Anterior Cingulate Group
  {
    id: 'dacc',
    name: 'Dorsal Anterior Cingulate Cortex',
    glyph: '⚠️',
    category: 'cortical',
    function: 'Conflict monitoring, error detection',
    wumboPhases: ['ignition', 'empowerment'],
    emotionalRole: 'Detecting when effort is needed',
    activationTriggers: ['Mistakes', 'Conflicting information', 'Effortful tasks'],
    linkedDomains: ['mind'],
    neurochemicals: ['Norepinephrine', 'Dopamine'],
  },
  {
    id: 'racc',
    name: 'Rostral Anterior Cingulate Cortex',
    glyph: '💗',
    category: 'cortical',
    function: 'Emotional conflict resolution, empathy',
    wumboPhases: ['resonance', 'transmission'],
    emotionalRole: 'Emotional awareness and regulation',
    activationTriggers: ['Emotional decisions', 'Empathic responses', 'Moral judgments'],
    linkedDomains: ['emotion', 'social'],
    neurochemicals: ['Serotonin', 'Oxytocin'],
  },
  {
    id: 'sgacc',
    name: 'Subgenual Anterior Cingulate',
    glyph: '🌧️',
    category: 'cortical',
    function: 'Mood regulation, autonomic control',
    wumboPhases: ['collapse', 'reflection'],
    emotionalRole: 'Sadness processing, mood baseline',
    activationTriggers: ['Sadness', 'Depression symptoms', 'Emotional pain'],
    linkedDomains: ['emotion'],
    neurochemicals: ['Serotonin', 'GABA'],
  },

  // Parietal Regions
  {
    id: 'spl',
    name: 'Superior Parietal Lobule',
    glyph: '🧭',
    category: 'cortical',
    function: 'Spatial awareness, attention shifting',
    wumboPhases: ['ignition', 'resonance'],
    emotionalRole: 'Body-space awareness',
    activationTriggers: ['Navigation', 'Visual attention', 'Reaching movements'],
    linkedDomains: ['body', 'mind'],
    neurochemicals: ['Acetylcholine', 'Norepinephrine'],
  },
  {
    id: 'ipl',
    name: 'Inferior Parietal Lobule',
    glyph: '🔢',
    category: 'cortical',
    function: 'Mathematical processing, tool use',
    wumboPhases: ['empowerment', 'mania'],
    emotionalRole: 'Sense of agency',
    activationTriggers: ['Calculations', 'Tool manipulation', 'Action understanding'],
    linkedDomains: ['mind', 'creative'],
    neurochemicals: ['Dopamine', 'Glutamate'],
  },
  {
    id: 'tpj',
    name: 'Temporoparietal Junction',
    glyph: '👁️‍🗨️',
    category: 'cortical',
    function: 'Theory of mind, self-other distinction',
    wumboPhases: ['transmission', 'resonance'],
    emotionalRole: 'Understanding others\' mental states',
    activationTriggers: ['Social interaction', 'Perspective taking', 'Moral reasoning'],
    linkedDomains: ['social', 'spirit'],
    neurochemicals: ['Oxytocin', 'Serotonin'],
  },
  {
    id: 'precuneus',
    name: 'Precuneus',
    glyph: '🌌',
    category: 'cortical',
    function: 'Self-consciousness, episodic memory retrieval',
    wumboPhases: ['reflection', 'nirvana'],
    emotionalRole: 'Autobiographical self',
    activationTriggers: ['Memory recall', 'Self-reflection', 'Mental imagery'],
    linkedDomains: ['spirit', 'mind'],
    neurochemicals: ['Acetylcholine', 'Serotonin'],
  },

  // Temporal Regions
  {
    id: 'stg',
    name: 'Superior Temporal Gyrus',
    glyph: '👂',
    category: 'cortical',
    function: 'Auditory processing, language comprehension',
    wumboPhases: ['resonance', 'transmission'],
    emotionalRole: 'Processing emotional tone of voice',
    activationTriggers: ['Listening', 'Music', 'Speech comprehension'],
    linkedDomains: ['social', 'creative'],
    neurochemicals: ['Glutamate', 'GABA'],
  },
  {
    id: 'mtg',
    name: 'Middle Temporal Gyrus',
    glyph: '📖',
    category: 'cortical',
    function: 'Semantic memory, word meaning',
    wumboPhases: ['empowerment', 'transmission'],
    emotionalRole: 'Conceptual emotional knowledge',
    activationTriggers: ['Reading', 'Naming', 'Concept retrieval'],
    linkedDomains: ['mind', 'creative'],
    neurochemicals: ['Glutamate', 'Acetylcholine'],
  },
  {
    id: 'itg',
    name: 'Inferior Temporal Gyrus',
    glyph: '🖼️',
    category: 'cortical',
    function: 'Visual object recognition, face processing',
    wumboPhases: ['ignition', 'resonance'],
    emotionalRole: 'Recognizing emotional expressions',
    activationTriggers: ['Face viewing', 'Object recognition', 'Reading'],
    linkedDomains: ['social', 'mind'],
    neurochemicals: ['Glutamate', 'Dopamine'],
  },
  {
    id: 'temporalpole',
    name: 'Temporal Pole',
    glyph: '🏛️',
    category: 'cortical',
    function: 'Social and emotional semantic knowledge',
    wumboPhases: ['transmission', 'reflection'],
    emotionalRole: 'Personal significance of memories',
    activationTriggers: ['Personal memories', 'Social concepts', 'Emotional meaning'],
    linkedDomains: ['social', 'spirit'],
    neurochemicals: ['Serotonin', 'Dopamine'],
  },
  {
    id: 'fusiform',
    name: 'Fusiform Gyrus',
    glyph: '👤',
    category: 'cortical',
    function: 'Face recognition, visual word form',
    wumboPhases: ['ignition', 'resonance'],
    emotionalRole: 'Face identity and emotional expression',
    activationTriggers: ['Faces', 'Word reading', 'Fine visual discrimination'],
    linkedDomains: ['social', 'mind'],
    neurochemicals: ['Glutamate', 'Norepinephrine'],
  },

  // Occipital Regions
  {
    id: 'v1',
    name: 'Primary Visual Cortex (V1)',
    glyph: '👁️',
    category: 'sensory',
    function: 'Basic visual processing',
    wumboPhases: ['ignition'],
    emotionalRole: 'Initial visual awareness',
    activationTriggers: ['Light', 'Edges', 'Motion'],
    linkedDomains: ['mind'],
    neurochemicals: ['Glutamate', 'GABA'],
  },
  {
    id: 'v4',
    name: 'Visual Area V4',
    glyph: '🎨',
    category: 'sensory',
    function: 'Color and form processing',
    wumboPhases: ['resonance', 'mania'],
    emotionalRole: 'Aesthetic appreciation',
    activationTriggers: ['Colors', 'Shapes', 'Art viewing'],
    linkedDomains: ['creative'],
    neurochemicals: ['Glutamate', 'Dopamine'],
  },
  {
    id: 'mt',
    name: 'Visual Area MT/V5',
    glyph: '🌪️',
    category: 'sensory',
    function: 'Motion perception',
    wumboPhases: ['ignition', 'mania'],
    emotionalRole: 'Detecting approaching threats',
    activationTriggers: ['Movement', 'Direction', 'Speed'],
    linkedDomains: ['body', 'mind'],
    neurochemicals: ['Glutamate', 'Norepinephrine'],
  },

  // Motor Regions
  {
    id: 'm1',
    name: 'Primary Motor Cortex',
    glyph: '💪',
    category: 'motor',
    function: 'Voluntary movement execution',
    wumboPhases: ['empowerment', 'mania'],
    emotionalRole: 'Action expression of emotion',
    activationTriggers: ['Movement initiation', 'Skilled actions', 'Exercise'],
    linkedDomains: ['body'],
    neurochemicals: ['Glutamate', 'Dopamine'],
  },
  {
    id: 'pmc',
    name: 'Premotor Cortex',
    glyph: '🎬',
    category: 'motor',
    function: 'Movement planning and preparation',
    wumboPhases: ['empowerment', 'ignition'],
    emotionalRole: 'Anticipatory action',
    activationTriggers: ['Movement planning', 'Learned sequences', 'Intention'],
    linkedDomains: ['body', 'mind'],
    neurochemicals: ['Dopamine', 'Glutamate'],
  },
  {
    id: 'sma',
    name: 'Supplementary Motor Area',
    glyph: '🔄',
    category: 'motor',
    function: 'Internally generated movements, sequences',
    wumboPhases: ['resonance', 'mania'],
    emotionalRole: 'Volitional action initiation',
    activationTriggers: ['Self-initiated movement', 'Bimanual coordination', 'Sequences'],
    linkedDomains: ['body', 'creative'],
    neurochemicals: ['Dopamine', 'GABA'],
  },
  {
    id: 'fef',
    name: 'Frontal Eye Fields',
    glyph: '🔍',
    category: 'motor',
    function: 'Voluntary eye movements, visual attention',
    wumboPhases: ['ignition', 'empowerment'],
    emotionalRole: 'Directing attention to emotional stimuli',
    activationTriggers: ['Saccades', 'Visual search', 'Attention shifts'],
    linkedDomains: ['mind'],
    neurochemicals: ['Dopamine', 'Acetylcholine'],
  },

  // Insular Cortex
  {
    id: 'anteriorinsula',
    name: 'Anterior Insula',
    glyph: '🫀',
    category: 'cortical',
    function: 'Interoception, emotional awareness',
    wumboPhases: ['resonance', 'reflection'],
    emotionalRole: 'Feeling emotions in the body',
    activationTriggers: ['Body sensations', 'Emotional experiences', 'Empathy'],
    linkedDomains: ['emotion', 'body'],
    neurochemicals: ['Norepinephrine', 'Dopamine'],
  },
  {
    id: 'posteriorinsula',
    name: 'Posterior Insula',
    glyph: '🌡️',
    category: 'cortical',
    function: 'Pain, temperature, basic interoception',
    wumboPhases: ['ignition', 'collapse'],
    emotionalRole: 'Physical comfort/discomfort',
    activationTriggers: ['Pain', 'Temperature', 'Touch'],
    linkedDomains: ['body'],
    neurochemicals: ['Glutamate', 'Substance P'],
  },

  // Somatosensory
  {
    id: 's1',
    name: 'Primary Somatosensory Cortex',
    glyph: '✋',
    category: 'sensory',
    function: 'Touch, pressure, proprioception',
    wumboPhases: ['resonance', 'reflection'],
    emotionalRole: 'Physical grounding',
    activationTriggers: ['Touch', 'Texture', 'Body position'],
    linkedDomains: ['body'],
    neurochemicals: ['Glutamate', 'GABA'],
  },
  {
    id: 's2',
    name: 'Secondary Somatosensory Cortex',
    glyph: '🤲',
    category: 'sensory',
    function: 'Complex touch, bilateral integration',
    wumboPhases: ['resonance'],
    emotionalRole: 'Affective touch processing',
    activationTriggers: ['Object manipulation', 'Texture recognition', 'Social touch'],
    linkedDomains: ['body', 'social'],
    neurochemicals: ['Oxytocin', 'Glutamate'],
  },

  // ============================================================================
  // LIMBIC REGIONS (25) - Emotion and Memory
  // ============================================================================

  {
    id: 'amygdala_bla',
    name: 'Basolateral Amygdala',
    glyph: '⚡',
    category: 'limbic',
    function: 'Fear learning, emotional memory formation',
    wumboPhases: ['ignition', 'collapse'],
    emotionalRole: 'Threat detection and fear',
    activationTriggers: ['Fear stimuli', 'Uncertainty', 'Novel threats'],
    linkedDomains: ['emotion'],
    neurochemicals: ['Norepinephrine', 'Cortisol'],
  },
  {
    id: 'amygdala_cea',
    name: 'Central Amygdala',
    glyph: '🚨',
    category: 'limbic',
    function: 'Fear expression, autonomic responses',
    wumboPhases: ['collapse'],
    emotionalRole: 'Fight/flight/freeze activation',
    activationTriggers: ['Immediate threats', 'Panic', 'Startle'],
    linkedDomains: ['emotion', 'body'],
    neurochemicals: ['CRH', 'Norepinephrine'],
  },
  {
    id: 'amygdala_mea',
    name: 'Medial Amygdala',
    glyph: '👃',
    category: 'limbic',
    function: 'Social and reproductive behaviors',
    wumboPhases: ['resonance', 'transmission'],
    emotionalRole: 'Social recognition via scent',
    activationTriggers: ['Pheromones', 'Social odors', 'Mate recognition'],
    linkedDomains: ['social'],
    neurochemicals: ['Oxytocin', 'Vasopressin'],
  },
  {
    id: 'hippocampus_ca1',
    name: 'Hippocampus CA1',
    glyph: '📝',
    category: 'limbic',
    function: 'Memory consolidation, spatial memory',
    wumboPhases: ['reflection', 'rewrite'],
    emotionalRole: 'Contextual emotional memory',
    activationTriggers: ['Learning', 'Navigation', 'Memory recall'],
    linkedDomains: ['mind'],
    neurochemicals: ['Acetylcholine', 'Glutamate'],
  },
  {
    id: 'hippocampus_ca3',
    name: 'Hippocampus CA3',
    glyph: '🔗',
    category: 'limbic',
    function: 'Pattern completion, associative memory',
    wumboPhases: ['reflection', 'nirvana'],
    emotionalRole: 'Memory association and retrieval',
    activationTriggers: ['Partial cues', 'Association formation', 'Pattern matching'],
    linkedDomains: ['mind', 'creative'],
    neurochemicals: ['Glutamate', 'GABA'],
  },
  {
    id: 'dentate',
    name: 'Dentate Gyrus',
    glyph: '🌱',
    category: 'limbic',
    function: 'Pattern separation, neurogenesis',
    wumboPhases: ['rewrite', 'ignition'],
    emotionalRole: 'Distinguishing similar experiences',
    activationTriggers: ['New learning', 'Exercise', 'Novelty'],
    linkedDomains: ['mind', 'body'],
    neurochemicals: ['BDNF', 'Serotonin'],
  },
  {
    id: 'subiculum',
    name: 'Subiculum',
    glyph: '🚪',
    category: 'limbic',
    function: 'Hippocampal output, stress response',
    wumboPhases: ['reflection', 'collapse'],
    emotionalRole: 'Stress-memory integration',
    activationTriggers: ['Memory retrieval', 'Stress', 'Context processing'],
    linkedDomains: ['mind', 'emotion'],
    neurochemicals: ['Glutamate', 'Cortisol'],
  },
  {
    id: 'entorhinal',
    name: 'Entorhinal Cortex',
    glyph: '🗺️',
    category: 'limbic',
    function: 'Gateway to hippocampus, spatial mapping',
    wumboPhases: ['ignition', 'reflection'],
    emotionalRole: 'Contextual memory encoding',
    activationTriggers: ['Navigation', 'Memory encoding', 'Time perception'],
    linkedDomains: ['mind'],
    neurochemicals: ['Acetylcholine', 'Glutamate'],
  },
  {
    id: 'perirhinal',
    name: 'Perirhinal Cortex',
    glyph: '🏷️',
    category: 'limbic',
    function: 'Object recognition memory, familiarity',
    wumboPhases: ['resonance', 'reflection'],
    emotionalRole: 'Feeling of knowing',
    activationTriggers: ['Recognition', 'Familiarity judgments', 'Object memory'],
    linkedDomains: ['mind'],
    neurochemicals: ['Acetylcholine', 'Dopamine'],
  },
  {
    id: 'parahippocampal',
    name: 'Parahippocampal Cortex',
    glyph: '🏞️',
    category: 'limbic',
    function: 'Scene processing, spatial context',
    wumboPhases: ['reflection', 'resonance'],
    emotionalRole: 'Environmental emotional associations',
    activationTriggers: ['Scenes', 'Places', 'Spatial context'],
    linkedDomains: ['mind', 'spirit'],
    neurochemicals: ['Glutamate', 'Acetylcholine'],
  },
  {
    id: 'hypothalamus_pvn',
    name: 'Paraventricular Nucleus',
    glyph: '🌊',
    category: 'limbic',
    function: 'Stress response, hormone release',
    wumboPhases: ['collapse', 'rewrite'],
    emotionalRole: 'HPA axis activation',
    activationTriggers: ['Stress', 'Homeostatic challenge', 'Social bonding'],
    linkedDomains: ['body', 'emotion'],
    neurochemicals: ['CRH', 'Oxytocin', 'Vasopressin'],
  },
  {
    id: 'hypothalamus_lh',
    name: 'Lateral Hypothalamus',
    glyph: '🍽️',
    category: 'limbic',
    function: 'Hunger, reward seeking, arousal',
    wumboPhases: ['ignition', 'mania'],
    emotionalRole: 'Wanting and craving',
    activationTriggers: ['Hunger', 'Reward anticipation', 'Wakefulness'],
    linkedDomains: ['body'],
    neurochemicals: ['Orexin', 'MCH', 'Dopamine'],
  },
  {
    id: 'hypothalamus_vmh',
    name: 'Ventromedial Hypothalamus',
    glyph: '🛑',
    category: 'limbic',
    function: 'Satiety, defensive behaviors',
    wumboPhases: ['resonance', 'collapse'],
    emotionalRole: 'Satisfaction and defense',
    activationTriggers: ['Satiation', 'Threat response', 'Female reproduction'],
    linkedDomains: ['body', 'emotion'],
    neurochemicals: ['Leptin receptors', 'GABA'],
  },
  {
    id: 'septum',
    name: 'Septal Nuclei',
    glyph: '😌',
    category: 'limbic',
    function: 'Reward, pleasure, anxiety reduction',
    wumboPhases: ['nirvana', 'resonance'],
    emotionalRole: 'Pleasure and relief',
    activationTriggers: ['Pleasure', 'Social reward', 'Anxiety relief'],
    linkedDomains: ['emotion', 'social'],
    neurochemicals: ['Endorphins', 'GABA', 'Acetylcholine'],
  },
  {
    id: 'bed_nucleus',
    name: 'Bed Nucleus of Stria Terminalis',
    glyph: '😰',
    category: 'limbic',
    function: 'Sustained anxiety, threat monitoring',
    wumboPhases: ['collapse'],
    emotionalRole: 'Chronic worry and vigilance',
    activationTriggers: ['Unpredictable threats', 'Chronic stress', 'Anticipatory anxiety'],
    linkedDomains: ['emotion'],
    neurochemicals: ['CRH', 'Norepinephrine'],
  },
  {
    id: 'mammillary',
    name: 'Mammillary Bodies',
    glyph: '🔁',
    category: 'limbic',
    function: 'Memory relay, spatial memory',
    wumboPhases: ['reflection'],
    emotionalRole: 'Memory circuit integration',
    activationTriggers: ['Memory formation', 'Spatial navigation', 'Head direction'],
    linkedDomains: ['mind'],
    neurochemicals: ['Glutamate', 'Acetylcholine'],
  },
  {
    id: 'cingulate_posterior',
    name: 'Posterior Cingulate Cortex',
    glyph: '🧘',
    category: 'limbic',
    function: 'Self-reflection, autobiographical memory',
    wumboPhases: ['reflection', 'nirvana'],
    emotionalRole: 'Self-awareness and rumination',
    activationTriggers: ['Self-focus', 'Mind-wandering', 'Memory retrieval'],
    linkedDomains: ['spirit', 'mind'],
    neurochemicals: ['Serotonin', 'Acetylcholine'],
  },
  {
    id: 'retrosplenial',
    name: 'Retrosplenial Cortex',
    glyph: '🏠',
    category: 'limbic',
    function: 'Navigation, spatial memory, context',
    wumboPhases: ['reflection', 'resonance'],
    emotionalRole: 'Sense of place and belonging',
    activationTriggers: ['Navigation', 'Familiar places', 'Context shifting'],
    linkedDomains: ['mind', 'spirit'],
    neurochemicals: ['Glutamate', 'Acetylcholine'],
  },
  {
    id: 'nucleus_accumbens_core',
    name: 'Nucleus Accumbens Core',
    glyph: '🎯',
    category: 'limbic',
    function: 'Goal-directed behavior, reward pursuit',
    wumboPhases: ['empowerment', 'mania'],
    emotionalRole: 'Motivation to act',
    activationTriggers: ['Reward cues', 'Goal pursuit', 'Effort decisions'],
    linkedDomains: ['body', 'mind'],
    neurochemicals: ['Dopamine', 'Glutamate'],
  },
  {
    id: 'nucleus_accumbens_shell',
    name: 'Nucleus Accumbens Shell',
    glyph: '🍯',
    category: 'limbic',
    function: 'Hedonic impact, liking',
    wumboPhases: ['nirvana', 'resonance'],
    emotionalRole: 'Pleasure experience',
    activationTriggers: ['Reward consumption', 'Pleasure', 'Drug effects'],
    linkedDomains: ['emotion'],
    neurochemicals: ['Dopamine', 'Opioids', 'Endocannabinoids'],
  },
  {
    id: 'olfactory_bulb',
    name: 'Olfactory Bulb',
    glyph: '🌸',
    category: 'limbic',
    function: 'Smell processing, emotional memory',
    wumboPhases: ['ignition', 'reflection'],
    emotionalRole: 'Smell-triggered emotions',
    activationTriggers: ['Odors', 'Aromatherapy', 'Memory cues'],
    linkedDomains: ['emotion', 'mind'],
    neurochemicals: ['Glutamate', 'Dopamine'],
  },
  {
    id: 'piriform',
    name: 'Piriform Cortex',
    glyph: '🍃',
    category: 'limbic',
    function: 'Olfactory processing, odor memory',
    wumboPhases: ['resonance', 'reflection'],
    emotionalRole: 'Emotional significance of smells',
    activationTriggers: ['Odor identification', 'Smell memory', 'Emotional scents'],
    linkedDomains: ['emotion'],
    neurochemicals: ['Glutamate', 'GABA'],
  },
  {
    id: 'habenula',
    name: 'Habenula',
    glyph: '🚫',
    category: 'limbic',
    function: 'Disappointment, aversion learning',
    wumboPhases: ['collapse', 'reflection'],
    emotionalRole: 'Processing failure and loss',
    activationTriggers: ['Reward omission', 'Punishment', 'Negative prediction error'],
    linkedDomains: ['emotion'],
    neurochemicals: ['Glutamate (inhibits dopamine)'],
  },
  {
    id: 'fornix',
    name: 'Fornix',
    glyph: '🌉',
    category: 'limbic',
    function: 'Memory pathway, hippocampal connections',
    wumboPhases: ['reflection'],
    emotionalRole: 'Memory circuit integrity',
    activationTriggers: ['Memory encoding', 'Retrieval', 'Spatial navigation'],
    linkedDomains: ['mind'],
    neurochemicals: ['Acetylcholine'],
  },

  // ============================================================================
  // BASAL GANGLIA (15) - Habit and Movement
  // ============================================================================

  {
    id: 'caudate_head',
    name: 'Caudate Head',
    glyph: '🎓',
    category: 'subcortical',
    function: 'Goal-directed learning, cognitive control',
    wumboPhases: ['empowerment', 'ignition'],
    emotionalRole: 'Cognitive flexibility',
    activationTriggers: ['Learning rules', 'Strategy switching', 'Cognitive tasks'],
    linkedDomains: ['mind'],
    neurochemicals: ['Dopamine', 'Acetylcholine'],
  },
  {
    id: 'caudate_body',
    name: 'Caudate Body',
    glyph: '📋',
    category: 'subcortical',
    function: 'Procedural learning, feedback processing',
    wumboPhases: ['empowerment', 'resonance'],
    emotionalRole: 'Learning from outcomes',
    activationTriggers: ['Feedback', 'Skill learning', 'Category learning'],
    linkedDomains: ['mind', 'body'],
    neurochemicals: ['Dopamine'],
  },
  {
    id: 'caudate_tail',
    name: 'Caudate Tail',
    glyph: '👀',
    category: 'subcortical',
    function: 'Visual habit learning',
    wumboPhases: ['resonance'],
    emotionalRole: 'Visual-motor habits',
    activationTriggers: ['Visual learning', 'Eye movements', 'Visual habits'],
    linkedDomains: ['mind'],
    neurochemicals: ['Dopamine'],
  },
  {
    id: 'putamen_anterior',
    name: 'Anterior Putamen',
    glyph: '🎭',
    category: 'subcortical',
    function: 'Cognitive-motor integration',
    wumboPhases: ['empowerment'],
    emotionalRole: 'Action selection',
    activationTriggers: ['Action planning', 'Motor preparation', 'Learning'],
    linkedDomains: ['body', 'mind'],
    neurochemicals: ['Dopamine', 'GABA'],
  },
  {
    id: 'putamen_posterior',
    name: 'Posterior Putamen',
    glyph: '⚙️',
    category: 'subcortical',
    function: 'Automatic movement, habits',
    wumboPhases: ['resonance', 'mania'],
    emotionalRole: 'Habitual actions',
    activationTriggers: ['Overlearned movements', 'Habits', 'Motor sequences'],
    linkedDomains: ['body'],
    neurochemicals: ['Dopamine', 'GABA'],
  },
  {
    id: 'globus_pallidus_ext',
    name: 'Globus Pallidus External',
    glyph: '🚦',
    category: 'subcortical',
    function: 'Movement regulation, indirect pathway',
    wumboPhases: ['empowerment'],
    emotionalRole: 'Stopping unwanted movements',
    activationTriggers: ['Movement suppression', 'Action selection'],
    linkedDomains: ['body'],
    neurochemicals: ['GABA'],
  },
  {
    id: 'globus_pallidus_int',
    name: 'Globus Pallidus Internal',
    glyph: '🔓',
    category: 'subcortical',
    function: 'Movement release, basal ganglia output',
    wumboPhases: ['mania', 'empowerment'],
    emotionalRole: 'Enabling desired actions',
    activationTriggers: ['Action execution', 'Movement initiation'],
    linkedDomains: ['body'],
    neurochemicals: ['GABA'],
  },
  {
    id: 'subthalamic',
    name: 'Subthalamic Nucleus',
    glyph: '⏸️',
    category: 'subcortical',
    function: 'Action suppression, impulsivity control',
    wumboPhases: ['reflection', 'empowerment'],
    emotionalRole: 'Stopping impulsive actions',
    activationTriggers: ['Stop signals', 'Conflict', 'Decision pausing'],
    linkedDomains: ['mind', 'body'],
    neurochemicals: ['Glutamate'],
  },
  {
    id: 'substantia_nigra_pc',
    name: 'Substantia Nigra Pars Compacta',
    glyph: '⚡',
    category: 'subcortical',
    function: 'Dopamine production, movement initiation',
    wumboPhases: ['ignition', 'empowerment'],
    emotionalRole: 'Motivation to move',
    activationTriggers: ['Movement planning', 'Reward', 'Novelty'],
    linkedDomains: ['body'],
    neurochemicals: ['Dopamine'],
  },
  {
    id: 'substantia_nigra_pr',
    name: 'Substantia Nigra Pars Reticulata',
    glyph: '🎛️',
    category: 'subcortical',
    function: 'Basal ganglia output, eye movements',
    wumboPhases: ['empowerment'],
    emotionalRole: 'Action gating',
    activationTriggers: ['Saccades', 'Action selection', 'Motor output'],
    linkedDomains: ['body'],
    neurochemicals: ['GABA'],
  },
  {
    id: 'vta',
    name: 'Ventral Tegmental Area',
    glyph: '🌟',
    category: 'subcortical',
    function: 'Reward, motivation, dopamine source',
    wumboPhases: ['ignition', 'mania', 'nirvana'],
    emotionalRole: 'Reward and motivation',
    activationTriggers: ['Reward', 'Novelty', 'Unexpected positive events'],
    linkedDomains: ['emotion', 'mind'],
    neurochemicals: ['Dopamine', 'GABA'],
  },
  {
    id: 'ventral_pallidum',
    name: 'Ventral Pallidum',
    glyph: '😊',
    category: 'subcortical',
    function: 'Pleasure, reward processing',
    wumboPhases: ['nirvana', 'resonance'],
    emotionalRole: 'Hedonic experience',
    activationTriggers: ['Pleasure', 'Reward consumption', 'Liking'],
    linkedDomains: ['emotion'],
    neurochemicals: ['GABA', 'Opioids'],
  },
  {
    id: 'claustrum',
    name: 'Claustrum',
    glyph: '🌐',
    category: 'subcortical',
    function: 'Consciousness integration, cross-modal binding',
    wumboPhases: ['nirvana', 'resonance'],
    emotionalRole: 'Unified conscious experience',
    activationTriggers: ['Attention', 'Consciousness', 'Salience'],
    linkedDomains: ['spirit', 'mind'],
    neurochemicals: ['Glutamate', 'GABA'],
  },
  {
    id: 'striatum_ventral',
    name: 'Ventral Striatum',
    glyph: '💎',
    category: 'subcortical',
    function: 'Reward anticipation, motivation',
    wumboPhases: ['ignition', 'empowerment'],
    emotionalRole: 'Wanting and anticipation',
    activationTriggers: ['Reward cues', 'Motivation', 'Incentives'],
    linkedDomains: ['emotion', 'body'],
    neurochemicals: ['Dopamine'],
  },
  {
    id: 'striatum_dorsal',
    name: 'Dorsal Striatum',
    glyph: '🔧',
    category: 'subcortical',
    function: 'Habit formation, skill learning',
    wumboPhases: ['resonance', 'rewrite'],
    emotionalRole: 'Automatic behavioral patterns',
    activationTriggers: ['Habits', 'Skills', 'Routines'],
    linkedDomains: ['body', 'mind'],
    neurochemicals: ['Dopamine', 'Acetylcholine'],
  },

  // ============================================================================
  // THALAMUS (10) - Relay and Integration
  // ============================================================================

  {
    id: 'thalamus_md',
    name: 'Mediodorsal Thalamus',
    glyph: '📡',
    category: 'subcortical',
    function: 'Prefrontal relay, executive function',
    wumboPhases: ['empowerment', 'ignition'],
    emotionalRole: 'Cognitive-emotional integration',
    activationTriggers: ['Decision making', 'Memory', 'Attention'],
    linkedDomains: ['mind'],
    neurochemicals: ['Glutamate'],
  },
  {
    id: 'thalamus_pulvinar',
    name: 'Pulvinar',
    glyph: '🔦',
    category: 'subcortical',
    function: 'Visual attention, salience detection',
    wumboPhases: ['ignition', 'resonance'],
    emotionalRole: 'Emotional visual attention',
    activationTriggers: ['Visual attention', 'Fear detection', 'Salience'],
    linkedDomains: ['mind', 'emotion'],
    neurochemicals: ['Glutamate', 'GABA'],
  },
  {
    id: 'thalamus_anterior',
    name: 'Anterior Thalamus',
    glyph: '🧭',
    category: 'subcortical',
    function: 'Memory, navigation, limbic relay',
    wumboPhases: ['reflection'],
    emotionalRole: 'Memory-emotion integration',
    activationTriggers: ['Memory', 'Navigation', 'Head direction'],
    linkedDomains: ['mind'],
    neurochemicals: ['Glutamate', 'Acetylcholine'],
  },
  {
    id: 'thalamus_vl',
    name: 'Ventrolateral Thalamus',
    glyph: '🎮',
    category: 'subcortical',
    function: 'Motor relay from cerebellum',
    wumboPhases: ['empowerment', 'mania'],
    emotionalRole: 'Smooth movement execution',
    activationTriggers: ['Movement', 'Motor learning', 'Coordination'],
    linkedDomains: ['body'],
    neurochemicals: ['Glutamate'],
  },
  {
    id: 'thalamus_vpm',
    name: 'Ventral Posteromedial Thalamus',
    glyph: '👅',
    category: 'subcortical',
    function: 'Face/mouth sensation relay',
    wumboPhases: ['resonance'],
    emotionalRole: 'Taste and facial touch',
    activationTriggers: ['Face touch', 'Taste', 'Oral sensations'],
    linkedDomains: ['body'],
    neurochemicals: ['Glutamate'],
  },
  {
    id: 'thalamus_vpl',
    name: 'Ventral Posterolateral Thalamus',
    glyph: '🦵',
    category: 'subcortical',
    function: 'Body sensation relay',
    wumboPhases: ['resonance'],
    emotionalRole: 'Body awareness',
    activationTriggers: ['Touch', 'Pain', 'Temperature'],
    linkedDomains: ['body'],
    neurochemicals: ['Glutamate'],
  },
  {
    id: 'thalamus_lgn',
    name: 'Lateral Geniculate Nucleus',
    glyph: '📺',
    category: 'subcortical',
    function: 'Visual relay to cortex',
    wumboPhases: ['ignition'],
    emotionalRole: 'Visual consciousness',
    activationTriggers: ['Vision', 'Light', 'Visual attention'],
    linkedDomains: ['mind'],
    neurochemicals: ['Glutamate', 'GABA'],
  },
  {
    id: 'thalamus_mgn',
    name: 'Medial Geniculate Nucleus',
    glyph: '🎵',
    category: 'subcortical',
    function: 'Auditory relay to cortex',
    wumboPhases: ['resonance'],
    emotionalRole: 'Emotional hearing',
    activationTriggers: ['Sound', 'Music', 'Speech'],
    linkedDomains: ['creative', 'social'],
    neurochemicals: ['Glutamate', 'GABA'],
  },
  {
    id: 'thalamus_reticular',
    name: 'Thalamic Reticular Nucleus',
    glyph: '🚧',
    category: 'subcortical',
    function: 'Attention gating, thalamic inhibition',
    wumboPhases: ['ignition', 'reflection'],
    emotionalRole: 'Selective attention',
    activationTriggers: ['Attention shifts', 'Sleep', 'Filtering'],
    linkedDomains: ['mind'],
    neurochemicals: ['GABA'],
  },
  {
    id: 'thalamus_intralaminar',
    name: 'Intralaminar Nuclei',
    glyph: '⏰',
    category: 'subcortical',
    function: 'Arousal, consciousness, pain',
    wumboPhases: ['ignition', 'collapse'],
    emotionalRole: 'Alertness and pain awareness',
    activationTriggers: ['Waking', 'Pain', 'Arousal'],
    linkedDomains: ['body', 'mind'],
    neurochemicals: ['Glutamate', 'Acetylcholine'],
  },

  // ============================================================================
  // BRAINSTEM (15) - Survival and Arousal
  // ============================================================================

  {
    id: 'pag',
    name: 'Periaqueductal Gray',
    glyph: '🛡️',
    category: 'brainstem',
    function: 'Pain modulation, defensive behaviors',
    wumboPhases: ['collapse', 'mania'],
    emotionalRole: 'Fight, flight, freeze, fawn',
    activationTriggers: ['Pain', 'Threat', 'Defensive behaviors'],
    linkedDomains: ['body', 'emotion'],
    neurochemicals: ['Opioids', 'GABA', 'Glutamate'],
  },
  {
    id: 'locus_coeruleus',
    name: 'Locus Coeruleus',
    glyph: '☕',
    category: 'brainstem',
    function: 'Arousal, attention, stress response',
    wumboPhases: ['ignition', 'mania'],
    emotionalRole: 'Alertness and vigilance',
    activationTriggers: ['Stress', 'Novelty', 'Attention demands'],
    linkedDomains: ['mind', 'body'],
    neurochemicals: ['Norepinephrine'],
  },
  {
    id: 'raphe_dorsal',
    name: 'Dorsal Raphe Nucleus',
    glyph: '☮️',
    category: 'brainstem',
    function: 'Mood regulation, patience',
    wumboPhases: ['resonance', 'reflection'],
    emotionalRole: 'Waiting, patience, mood stability',
    activationTriggers: ['Waiting', 'Social interaction', 'Food anticipation'],
    linkedDomains: ['emotion'],
    neurochemicals: ['Serotonin'],
  },
  {
    id: 'raphe_median',
    name: 'Median Raphe Nucleus',
    glyph: '🌅',
    category: 'brainstem',
    function: 'Anxiety reduction, hippocampal theta',
    wumboPhases: ['reflection', 'nirvana'],
    emotionalRole: 'Calm exploration',
    activationTriggers: ['Exploration', 'Anxiety reduction', 'Memory'],
    linkedDomains: ['emotion', 'mind'],
    neurochemicals: ['Serotonin'],
  },
  {
    id: 'parabrachial',
    name: 'Parabrachial Nucleus',
    glyph: '🫁',
    category: 'brainstem',
    function: 'Taste, breathing, pain, alarm',
    wumboPhases: ['ignition', 'collapse'],
    emotionalRole: 'Visceral alarm signals',
    activationTriggers: ['Taste', 'Breathing changes', 'Pain', 'Nausea'],
    linkedDomains: ['body'],
    neurochemicals: ['Glutamate', 'CGRP'],
  },
  {
    id: 'nucleus_tractus',
    name: 'Nucleus Tractus Solitarius',
    glyph: '💓',
    category: 'brainstem',
    function: 'Visceral sensation, vagal input',
    wumboPhases: ['resonance', 'collapse'],
    emotionalRole: 'Gut feelings, heart awareness',
    activationTriggers: ['Heart rate', 'Gut signals', 'Breathing'],
    linkedDomains: ['body', 'emotion'],
    neurochemicals: ['Glutamate', 'Norepinephrine'],
  },
  {
    id: 'reticular_formation',
    name: 'Reticular Formation',
    glyph: '🔋',
    category: 'brainstem',
    function: 'Arousal, consciousness, sleep-wake',
    wumboPhases: ['ignition', 'collapse'],
    emotionalRole: 'Basic alertness',
    activationTriggers: ['Waking', 'Startle', 'Sleep transitions'],
    linkedDomains: ['body'],
    neurochemicals: ['Acetylcholine', 'Norepinephrine', 'Serotonin'],
  },
  {
    id: 'superior_colliculus',
    name: 'Superior Colliculus',
    glyph: '👁️',
    category: 'brainstem',
    function: 'Visual orientation, eye movements',
    wumboPhases: ['ignition'],
    emotionalRole: 'Threat detection via vision',
    activationTriggers: ['Visual targets', 'Orienting', 'Looming objects'],
    linkedDomains: ['body', 'mind'],
    neurochemicals: ['Glutamate', 'GABA'],
  },
  {
    id: 'inferior_colliculus',
    name: 'Inferior Colliculus',
    glyph: '🔊',
    category: 'brainstem',
    function: 'Auditory processing, sound localization',
    wumboPhases: ['ignition', 'resonance'],
    emotionalRole: 'Sound-triggered startle and orientation',
    activationTriggers: ['Sounds', 'Music', 'Speech'],
    linkedDomains: ['mind'],
    neurochemicals: ['Glutamate', 'GABA'],
  },
  {
    id: 'red_nucleus',
    name: 'Red Nucleus',
    glyph: '🎯',
    category: 'brainstem',
    function: 'Motor coordination, reaching',
    wumboPhases: ['empowerment'],
    emotionalRole: 'Purposeful movement',
    activationTriggers: ['Reaching', 'Crawling', 'Motor control'],
    linkedDomains: ['body'],
    neurochemicals: ['Glutamate'],
  },
  {
    id: 'pedunculopontine',
    name: 'Pedunculopontine Nucleus',
    glyph: '🚶',
    category: 'brainstem',
    function: 'Locomotion, REM sleep, reward',
    wumboPhases: ['empowerment', 'nirvana'],
    emotionalRole: 'Movement initiation, dream states',
    activationTriggers: ['Walking', 'REM sleep', 'Reward'],
    linkedDomains: ['body'],
    neurochemicals: ['Acetylcholine', 'Glutamate'],
  },
  {
    id: 'ldtg',
    name: 'Laterodorsal Tegmental Nucleus',
    glyph: '💭',
    category: 'brainstem',
    function: 'REM sleep, arousal, reward',
    wumboPhases: ['nirvana', 'collapse'],
    emotionalRole: 'Dream generation',
    activationTriggers: ['REM sleep', 'Reward', 'Arousal'],
    linkedDomains: ['spirit'],
    neurochemicals: ['Acetylcholine'],
  },
  {
    id: 'dorsal_motor_vagus',
    name: 'Dorsal Motor Nucleus of Vagus',
    glyph: '🧘',
    category: 'brainstem',
    function: 'Parasympathetic control, rest-digest',
    wumboPhases: ['reflection', 'collapse'],
    emotionalRole: 'Calm, safety, rest',
    activationTriggers: ['Rest', 'Digestion', 'Safety'],
    linkedDomains: ['body'],
    neurochemicals: ['Acetylcholine'],
  },
  {
    id: 'nucleus_ambiguus',
    name: 'Nucleus Ambiguus',
    glyph: '🗣️',
    category: 'brainstem',
    function: 'Speech, swallowing, heart rate',
    wumboPhases: ['transmission', 'resonance'],
    emotionalRole: 'Social engagement via voice',
    activationTriggers: ['Speaking', 'Swallowing', 'Social vocalization'],
    linkedDomains: ['social', 'body'],
    neurochemicals: ['Acetylcholine'],
  },
  {
    id: 'facial_nucleus',
    name: 'Facial Motor Nucleus',
    glyph: '😀',
    category: 'brainstem',
    function: 'Facial expression',
    wumboPhases: ['transmission', 'resonance'],
    emotionalRole: 'Emotional expression via face',
    activationTriggers: ['Smiling', 'Frowning', 'Emotional expressions'],
    linkedDomains: ['social', 'emotion'],
    neurochemicals: ['Acetylcholine'],
  },

  // ============================================================================
  // CEREBELLUM (5) - Coordination and Learning
  // ============================================================================

  {
    id: 'cerebellum_anterior',
    name: 'Anterior Cerebellum',
    glyph: '🏃',
    category: 'cerebellar',
    function: 'Motor execution, posture, gait',
    wumboPhases: ['empowerment', 'mania'],
    emotionalRole: 'Physical confidence',
    activationTriggers: ['Walking', 'Balance', 'Coordinated movement'],
    linkedDomains: ['body'],
    neurochemicals: ['Glutamate', 'GABA'],
  },
  {
    id: 'cerebellum_posterior',
    name: 'Posterior Cerebellum',
    glyph: '🧩',
    category: 'cerebellar',
    function: 'Cognitive processing, language',
    wumboPhases: ['empowerment', 'resonance'],
    emotionalRole: 'Mental fluency',
    activationTriggers: ['Language', 'Cognitive tasks', 'Timing'],
    linkedDomains: ['mind'],
    neurochemicals: ['Glutamate', 'GABA'],
  },
  {
    id: 'vermis',
    name: 'Cerebellar Vermis',
    glyph: '⚖️',
    category: 'cerebellar',
    function: 'Balance, posture, emotional regulation',
    wumboPhases: ['resonance', 'reflection'],
    emotionalRole: 'Emotional balance',
    activationTriggers: ['Balance', 'Emotion regulation', 'Posture'],
    linkedDomains: ['body', 'emotion'],
    neurochemicals: ['GABA', 'Serotonin'],
  },
  {
    id: 'flocculonodular',
    name: 'Flocculonodular Lobe',
    glyph: '🌀',
    category: 'cerebellar',
    function: 'Vestibular processing, eye movements',
    wumboPhases: ['resonance'],
    emotionalRole: 'Spatial orientation',
    activationTriggers: ['Head movement', 'Balance', 'Eye tracking'],
    linkedDomains: ['body'],
    neurochemicals: ['GABA', 'Glutamate'],
  },
  {
    id: 'deep_cerebellar',
    name: 'Deep Cerebellar Nuclei',
    glyph: '🎼',
    category: 'cerebellar',
    function: 'Cerebellar output, timing',
    wumboPhases: ['empowerment', 'mania'],
    emotionalRole: 'Rhythmic precision',
    activationTriggers: ['Movement output', 'Timing', 'Learning'],
    linkedDomains: ['body', 'creative'],
    neurochemicals: ['Glutamate'],
  },
];

// ============================================================================
// Brain Region Utilities
// ============================================================================

/**
 * Get brain regions that are active during a specific Wumbo phase
 */
export function getBrainRegionsForPhase(phase: WumboPhase): BrainRegion[] {
  return BRAIN_REGIONS.filter(r => r.wumboPhases.includes(phase));
}

/**
 * Get brain regions by category
 */
export function getBrainRegionsByCategory(category: BrainRegionCategory): BrainRegion[] {
  return BRAIN_REGIONS.filter(r => r.category === category);
}

/**
 * Get brain regions linked to a life domain
 */
export function getBrainRegionsForDomain(domainId: string): BrainRegion[] {
  return BRAIN_REGIONS.filter(r => r.linkedDomains.includes(domainId));
}

/**
 * Get brain regions by neurochemical
 */
export function getBrainRegionsByNeurochemical(neurochemical: string): BrainRegion[] {
  return BRAIN_REGIONS.filter(r =>
    r.neurochemicals.some(n => n.toLowerCase().includes(neurochemical.toLowerCase()))
  );
}

/**
 * Get a brain region activation profile for a beat category
 */
export function getBrainActivationForBeat(beatCategory: string): {
  primaryRegions: BrainRegion[];
  supportingRegions: BrainRegion[];
  neurochemicalProfile: string[];
} {
  const domainMapping: Record<string, string> = {
    Workout: 'body',
    Meditation: 'mind',
    Emotion: 'emotion',
    Moderation: 'social',
    Journal: 'creative',
    Anchor: 'spirit',
    General: 'mind',
  };

  const domain = domainMapping[beatCategory] || 'mind';
  const primaryRegions = getBrainRegionsForDomain(domain).slice(0, 5);

  // Get supporting regions from related domains
  const supportingDomains = {
    body: ['mind', 'emotion'],
    mind: ['emotion', 'creative'],
    emotion: ['body', 'social'],
    social: ['emotion', 'mind'],
    creative: ['mind', 'emotion'],
    spirit: ['mind', 'emotion'],
  }[domain] || ['mind'];

  const supportingRegions = supportingDomains
    .flatMap(d => getBrainRegionsForDomain(d))
    .slice(0, 3);

  const neurochemicalProfile = [...new Set(
    [...primaryRegions, ...supportingRegions]
      .flatMap(r => r.neurochemicals)
  )];

  return { primaryRegions, supportingRegions, neurochemicalProfile };
}

/**
 * Get the neural signature for a music session based on emotional category
 */
export function getNeuralSignatureForMusic(emotionalCategory: string): {
  targetRegions: BrainRegion[];
  expectedActivation: Record<string, 'high' | 'moderate' | 'low'>;
  recommendedPhase: WumboPhase;
} {
  const emotionMapping: Record<string, { regions: string[]; phase: WumboPhase }> = {
    JOY: { regions: ['vta', 'nucleus_accumbens_shell', 'ventral_pallidum'], phase: 'nirvana' },
    CALM: { regions: ['raphe_dorsal', 'dorsal_motor_vagus', 'cingulate_posterior'], phase: 'reflection' },
    FOCUS: { regions: ['dlpfc', 'dacc', 'locus_coeruleus'], phase: 'empowerment' },
    ENERGY: { regions: ['substantia_nigra_pc', 'hypothalamus_lh', 'reticular_formation'], phase: 'ignition' },
    MELANCHOLY: { regions: ['sgacc', 'habenula', 'anteriorinsula'], phase: 'reflection' },
    LOVE: { regions: ['septum', 'hypothalamus_pvn', 'mpfc'], phase: 'resonance' },
    COURAGE: { regions: ['amygdala_bla', 'pag', 'dacc'], phase: 'empowerment' },
    WONDER: { regions: ['claustrum', 'precuneus', 'temporalpole'], phase: 'nirvana' },
    GRATITUDE: { regions: ['mpfc', 'racc', 'ventral_pallidum'], phase: 'transmission' },
    RELEASE: { regions: ['pag', 'raphe_median', 'bed_nucleus'], phase: 'collapse' },
  };

  const mapping = emotionMapping[emotionalCategory] || emotionMapping.CALM;
  const targetRegions = BRAIN_REGIONS.filter(r => mapping.regions.includes(r.id));

  const expectedActivation: Record<string, 'high' | 'moderate' | 'low'> = {};
  targetRegions.forEach(r => expectedActivation[r.name] = 'high');

  return {
    targetRegions,
    expectedActivation,
    recommendedPhase: mapping.phase,
  };
}

export const glyphSystem = {
  CORE_GLYPHS,
  PHASE_DESCRIPTIONS,
  LIFE_DOMAINS,
  BRAIN_REGIONS,
  getGlyphsForPhase,
  suggestCurrentPhase,
  getTransitionRitual,
  getGlyphReading,
  getGlyphsForBeat,
  getDomainForBeat,
  getBrainRegionsForPhase,
  getBrainRegionsByCategory,
  getBrainRegionsForDomain,
  getBrainRegionsByNeurochemical,
  getBrainActivationForBeat,
  getNeuralSignatureForMusic,
};

export default glyphSystem;
