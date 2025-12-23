/* INTEGRITY_METADATA
 * Date: 2025-12-23
 * Status: ⚠️ TRULY UNSUPPORTED - No supporting evidence found
 * Severity: HIGH RISK
 * Risk Types: unsupported_claims
 */


// Unified Challenge System - Combines Brain Region and Daily Challenges
// Links XP systems, adds interactive mini-games, and creates unique cosmetic rewards

import { notificationService } from './notificationService';
import type { BrainRegionCategory } from './glyphSystem';

// ============================================================================
// TYPES AND INTERFACES
// ============================================================================

export type ChallengeType =
  | 'brain_task'        // Brain region-specific wellness tasks
  | 'site_activity'     // Music, time spent, check-ins, tasks
  | 'metric_goal'       // DeltaHV metric targets
  | 'mini_game'         // Interactive challenges (riddles, memory, math)
  | 'secret';           // Hidden challenges with hints

export type MiniGameType =
  | 'riddle'
  | 'math_series'
  | 'color_sequence'
  | 'card_memory'
  | 'word_scramble'
  | 'reaction_time'
  | 'pattern_match'
  | 'speed_math';

export type ChallengeDifficulty = 'easy' | 'medium' | 'hard' | 'legendary';

export interface UnifiedChallenge {
  id: string;
  title: string;
  description: string;
  type: ChallengeType;
  difficulty: ChallengeDifficulty;
  xpReward: number;
  cosmeticRewardId: string;
  brainCategory?: BrainRegionCategory;
  icon: string;

  // For mini-games
  miniGameType?: MiniGameType;
  miniGameData?: MiniGameData;

  // For secret challenges
  isSecret?: boolean;
  hint?: string;
  secretCondition?: SecretCondition;

  // Progress tracking
  requirement: ChallengeRequirement;
}

export interface ChallengeRequirement {
  type: 'count' | 'threshold' | 'streak' | 'mini_game' | 'time_spent' | 'secret_trigger';
  target: number;
  metric?: string;
  category?: string;
  comparison?: 'gte' | 'lte' | 'eq';
}

export interface SecretCondition {
  type: 'streak_milestone' | 'perfect_metrics' | 'time_played' | 'all_categories' | 'night_owl' | 'early_bird';
  value?: number;
}

export interface MiniGameData {
  // Riddle
  question?: string;
  options?: string[];
  correctAnswer?: number;

  // Math series
  series?: number[];
  answer?: number;

  // Color sequence
  sequence?: string[];
  sequenceLength?: number;

  // Card memory
  pairs?: number;
  cardSymbols?: string[];

  // Word scramble
  word?: string;
  hint?: string;

  // Reaction time
  targetTime?: number;

  // Pattern match
  gridSize?: number;

  // Speed math
  problemCount?: number;
  timeLimit?: number;
}

export interface ActiveChallenge extends UnifiedChallenge {
  date: string;
  progress: number;
  completed: boolean;
  completedAt?: string;
  isPinned: boolean;
  isHighlighted: boolean;
  highlightReason?: string;
}

export interface UnifiedChallengeData {
  totalXP: number;
  level: number;
  currentStreak: number;
  longestStreak: number;
  lastActiveDate: string;
  completedChallenges: CompletedChallengeRecord[];
  unlockedCosmetics: string[];
  equippedCosmetics: Record<CosmeticCategory, string | null>;
  pinnedChallengeIds: string[];
  secretsDiscovered: string[];
  miniGameHighScores: Record<string, number>;
  // Daily mini-game XP limit tracking
  miniGameXPClaimsToday: number;
  lastMiniGameXPDate: string;
}

export interface CompletedChallengeRecord {
  challengeId: string;
  completedAt: string;
  xpEarned: number;
  cosmeticUnlocked?: string;
}

export type CosmeticCategory =
  | 'background'
  | 'gradient'
  | 'animation'
  | 'wallpaper'
  | 'tab_design'
  | 'button_style'
  | 'metrics_theme'
  | 'journal_design'
  | 'card_style'
  | 'accent_color';

// ============================================================================
// BRAIN REGION GRADIENT COSMETICS - Unique hex colors for each category
// ============================================================================

export const BRAIN_REGION_GRADIENTS: Record<BrainRegionCategory, { id: string; name: string; colors: string[]; cssVars: Record<string, string> }> = {
  cortical: {
    id: 'gradient_cortical_mind',
    name: 'Cortical Mind',
    colors: ['#6366F1', '#8B5CF6', '#A78BFA'],
    cssVars: {
      '--gradient-start': '#6366F1',
      '--gradient-mid': '#8B5CF6',
      '--gradient-end': '#A78BFA',
      '--gradient-angle': '135deg',
    }
  },
  limbic: {
    id: 'gradient_limbic_emotion',
    name: 'Limbic Emotion',
    colors: ['#EC4899', '#F472B6', '#F9A8D4'],
    cssVars: {
      '--gradient-start': '#EC4899',
      '--gradient-mid': '#F472B6',
      '--gradient-end': '#F9A8D4',
      '--gradient-angle': '135deg',
    }
  },
  subcortical: {
    id: 'gradient_subcortical_reward',
    name: 'Subcortical Reward',
    colors: ['#F59E0B', '#FBBF24', '#FCD34D'],
    cssVars: {
      '--gradient-start': '#F59E0B',
      '--gradient-mid': '#FBBF24',
      '--gradient-end': '#FCD34D',
      '--gradient-angle': '135deg',
    }
  },
  brainstem: {
    id: 'gradient_brainstem_vital',
    name: 'Brainstem Vital',
    colors: ['#06B6D4', '#22D3EE', '#67E8F9'],
    cssVars: {
      '--gradient-start': '#06B6D4',
      '--gradient-mid': '#22D3EE',
      '--gradient-end': '#67E8F9',
      '--gradient-angle': '135deg',
    }
  },
  cerebellar: {
    id: 'gradient_cerebellar_balance',
    name: 'Cerebellar Balance',
    colors: ['#10B981', '#34D399', '#6EE7B7'],
    cssVars: {
      '--gradient-start': '#10B981',
      '--gradient-mid': '#34D399',
      '--gradient-end': '#6EE7B7',
      '--gradient-angle': '135deg',
    }
  },
  sensory: {
    id: 'gradient_sensory_awareness',
    name: 'Sensory Awareness',
    colors: ['#3B82F6', '#60A5FA', '#93C5FD'],
    cssVars: {
      '--gradient-start': '#3B82F6',
      '--gradient-mid': '#60A5FA',
      '--gradient-end': '#93C5FD',
      '--gradient-angle': '135deg',
    }
  },
  motor: {
    id: 'gradient_motor_energy',
    name: 'Motor Energy',
    colors: ['#F97316', '#FB923C', '#FDBA74'],
    cssVars: {
      '--gradient-start': '#F97316',
      '--gradient-mid': '#FB923C',
      '--gradient-end': '#FDBA74',
      '--gradient-angle': '135deg',
    }
  },
  integration: {
    id: 'gradient_integration_unity',
    name: 'Integration Unity',
    colors: ['#8B5CF6', '#A78BFA', '#C4B5FD'],
    cssVars: {
      '--gradient-start': '#8B5CF6',
      '--gradient-mid': '#A78BFA',
      '--gradient-end': '#C4B5FD',
      '--gradient-angle': '135deg',
    }
  },
};

// ============================================================================
// MINI GAME GENERATORS
// ============================================================================

const RIDDLES: Array<{ question: string; options: string[]; correct: number }> = [
  { question: "I have cities, but no houses. I have mountains, but no trees. I have water, but no fish. What am I?", options: ["A globe", "A map", "A painting", "A dream"], correct: 1 },
  { question: "The more you take, the more you leave behind. What am I?", options: ["Time", "Footsteps", "Memories", "Breaths"], correct: 1 },
  { question: "What has keys but no locks, space but no room, and you can enter but can't go inside?", options: ["A piano", "A keyboard", "A house", "A car"], correct: 1 },
  { question: "I speak without a mouth and hear without ears. I have no body, but I come alive with the wind. What am I?", options: ["A ghost", "An echo", "A shadow", "A whisper"], correct: 1 },
  { question: "What can travel around the world while staying in a corner?", options: ["A spider", "A stamp", "Light", "The wind"], correct: 1 },
  { question: "I am not alive, but I grow; I don't have lungs, but I need air; I don't have a mouth, but water kills me. What am I?", options: ["A plant", "Fire", "A cloud", "A shadow"], correct: 1 },
  { question: "What has a head, a tail, is brown, and has no legs?", options: ["A snake", "A worm", "A penny", "A caterpillar"], correct: 2 },
  { question: "The person who makes it, sells it. The person who buys it never uses it. What is it?", options: ["A gift", "A coffin", "A house", "A car"], correct: 1 },
  { question: "What begins with T, ends with T, and has T in it?", options: ["A tent", "A teapot", "Toast", "Test"], correct: 1 },
  { question: "I can be cracked, made, told, and played. What am I?", options: ["A joke", "An egg", "Music", "A code"], correct: 0 },
];

const MATH_SERIES: Array<{ series: number[]; answer: number; hint: string }> = [
  { series: [2, 4, 8, 16], answer: 32, hint: "Each number is doubled" },
  { series: [1, 1, 2, 3, 5, 8], answer: 13, hint: "Sum of previous two" },
  { series: [3, 6, 9, 12], answer: 15, hint: "Add 3 each time" },
  { series: [1, 4, 9, 16, 25], answer: 36, hint: "Perfect squares" },
  { series: [2, 6, 12, 20, 30], answer: 42, hint: "Differences increase by 2" },
  { series: [1, 2, 4, 7, 11], answer: 16, hint: "Add 1, then 2, then 3..." },
  { series: [100, 50, 25], answer: 12.5, hint: "Divide by 2 each time" },
  { series: [1, 3, 6, 10, 15], answer: 21, hint: "Triangular numbers" },
  { series: [2, 3, 5, 7, 11, 13], answer: 17, hint: "Prime numbers" },
  { series: [1, 8, 27, 64], answer: 125, hint: "Perfect cubes" },
];

export function generateDailyRiddle(date: string): MiniGameData {
  const seed = hashCode(date + 'riddle');
  const index = Math.abs(seed) % RIDDLES.length;
  const riddle = RIDDLES[index];
  return {
    question: riddle.question,
    options: riddle.options,
    correctAnswer: riddle.correct,
  };
}

export function generateDailyMathSeries(date: string): MiniGameData {
  const seed = hashCode(date + 'math');
  const index = Math.abs(seed) % MATH_SERIES.length;
  const series = MATH_SERIES[index];
  return {
    series: series.series,
    answer: series.answer,
  };
}

export function generateColorSequence(length: number = 4): MiniGameData {
  const colors = ['#EF4444', '#F59E0B', '#22C55E', '#3B82F6', '#8B5CF6', '#EC4899'];
  const sequence: string[] = [];
  for (let i = 0; i < length; i++) {
    sequence.push(colors[Math.floor(Math.random() * colors.length)]);
  }
  return {
    sequence,
    sequenceLength: length,
  };
}

export function generateCardMemoryGame(pairs: number = 6): MiniGameData {
  const symbols = ['🌟', '🔥', '💎', '🌙', '⚡', '🎯', '🌈', '🎨', '🎵', '💫', '🍀', '🦋'];
  const selectedSymbols = symbols.slice(0, pairs);
  return {
    pairs,
    cardSymbols: selectedSymbols,
  };
}

const WORD_SCRAMBLES: Array<{ word: string; hint: string }> = [
  { word: 'MINDFUL', hint: 'Being present in the moment' },
  { word: 'BALANCE', hint: 'Equilibrium between work and rest' },
  { word: 'RHYTHM', hint: 'A regular pattern of movement or sound' },
  { word: 'FOCUS', hint: 'Concentrated attention on something' },
  { word: 'BREATHE', hint: 'Essential for relaxation' },
  { word: 'WELLNESS', hint: 'State of being in good health' },
  { word: 'HARMONY', hint: 'Things working together in peace' },
  { word: 'GRATITUDE', hint: 'Feeling thankful' },
  { word: 'SERENITY', hint: 'State of calm and peace' },
  { word: 'COURAGE', hint: 'Facing fears bravely' },
];

export function generateWordScramble(date: string): MiniGameData {
  const seed = hashCode(date + 'word');
  const index = Math.abs(seed) % WORD_SCRAMBLES.length;
  const wordData = WORD_SCRAMBLES[index];
  return {
    word: wordData.word,
    hint: wordData.hint,
  };
}

export function generateReactionTime(): MiniGameData {
  return {
    targetTime: 0.5, // 500ms target
  };
}

export function generatePatternMatch(): MiniGameData {
  return {
    gridSize: 3,
  };
}

export function generateSpeedMath(): MiniGameData {
  return {
    problemCount: 5,
    timeLimit: 30,
  };
}

function hashCode(str: string): number {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = ((hash << 5) - hash) + char;
    hash = hash & hash;
  }
  return hash;
}

// ============================================================================
// UNIFIED CHALLENGE POOL - Site functionality focused
// ============================================================================

export const UNIFIED_CHALLENGE_POOL: UnifiedChallenge[] = [
  // SITE ACTIVITY CHALLENGES - Music, Time, Tasks
  {
    id: 'music_listener_daily',
    title: 'Rhythm Rider',
    description: 'Listen to beats for 30 minutes today',
    type: 'site_activity',
    difficulty: 'easy',
    xpReward: 40,
    cosmeticRewardId: 'gradient_music_flow',
    icon: '🎵',
    requirement: { type: 'time_spent', target: 30, category: 'music' },
  },
  {
    id: 'music_explorer',
    title: 'Beat Explorer',
    description: 'Try 3 different beat categories in one session',
    type: 'site_activity',
    difficulty: 'medium',
    xpReward: 60,
    cosmeticRewardId: 'accent_beat_pulse',
    icon: '🎧',
    requirement: { type: 'count', target: 3, category: 'beat_categories' },
  },
  {
    id: 'time_warrior',
    title: 'Time Warrior',
    description: 'Spend 2 hours total in focused sessions today',
    type: 'site_activity',
    difficulty: 'medium',
    xpReward: 80,
    cosmeticRewardId: 'gradient_time_flow',
    icon: '⏱️',
    requirement: { type: 'time_spent', target: 120, category: 'focus' },
  },
  {
    id: 'task_master',
    title: 'Task Master',
    description: 'Complete 10 check-ins today',
    type: 'site_activity',
    difficulty: 'medium',
    xpReward: 70,
    cosmeticRewardId: 'button_task_glow',
    icon: '✅',
    requirement: { type: 'count', target: 10, category: 'check_ins' },
  },
  {
    id: 'journal_keeper',
    title: 'Journal Keeper',
    description: 'Write 3 journal entries today',
    type: 'site_activity',
    difficulty: 'easy',
    xpReward: 50,
    cosmeticRewardId: 'journal_mindful_ink',
    icon: '📝',
    requirement: { type: 'count', target: 3, category: 'journal' },
  },
  {
    id: 'streak_builder_3',
    title: 'Streak Starter',
    description: 'Build a 3-day activity streak',
    type: 'site_activity',
    difficulty: 'easy',
    xpReward: 60,
    cosmeticRewardId: 'accent_streak_flame',
    icon: '🔥',
    requirement: { type: 'streak', target: 3 },
  },
  {
    id: 'streak_builder_7',
    title: 'Week Warrior',
    description: 'Maintain a 7-day streak',
    type: 'site_activity',
    difficulty: 'hard',
    xpReward: 150,
    cosmeticRewardId: 'gradient_streak_inferno',
    icon: '👑',
    requirement: { type: 'streak', target: 7 },
  },

  // METRIC GOAL CHALLENGES - DeltaHV targets
  {
    id: 'resonance_rise',
    title: 'Resonance Rise',
    description: 'Achieve resonance score above 70',
    type: 'metric_goal',
    difficulty: 'medium',
    xpReward: 80,
    cosmeticRewardId: 'gradient_resonance_wave',
    icon: '📡',
    requirement: { type: 'threshold', target: 70, metric: 'resonance', comparison: 'gte' },
  },
  {
    id: 'stability_fortress',
    title: 'Stability Fortress',
    description: 'Keep stability above 75 for the day',
    type: 'metric_goal',
    difficulty: 'hard',
    xpReward: 100,
    cosmeticRewardId: 'gradient_stable_ground',
    icon: '🏰',
    requirement: { type: 'threshold', target: 75, metric: 'stability', comparison: 'gte' },
  },
  {
    id: 'low_friction',
    title: 'Friction Fighter',
    description: 'Keep friction below 30',
    type: 'metric_goal',
    difficulty: 'hard',
    xpReward: 90,
    cosmeticRewardId: 'gradient_smooth_flow',
    icon: '🛡️',
    requirement: { type: 'threshold', target: 30, metric: 'friction', comparison: 'lte' },
  },
  {
    id: 'symbolic_surge',
    title: 'Symbolic Surge',
    description: 'Boost symbolic density above 80',
    type: 'metric_goal',
    difficulty: 'hard',
    xpReward: 100,
    cosmeticRewardId: 'gradient_symbol_burst',
    icon: '🌀',
    requirement: { type: 'threshold', target: 80, metric: 'symbolic', comparison: 'gte' },
  },

  // MINI GAME CHALLENGES
  {
    id: 'daily_riddle',
    title: 'Daily Riddle',
    description: 'Solve today\'s brain teaser',
    type: 'mini_game',
    miniGameType: 'riddle',
    difficulty: 'easy',
    xpReward: 30,
    cosmeticRewardId: 'accent_riddle_gold',
    icon: '🧩',
    requirement: { type: 'mini_game', target: 1 },
  },
  {
    id: 'math_mind',
    title: 'Math Mind',
    description: 'Complete the number sequence',
    type: 'mini_game',
    miniGameType: 'math_series',
    difficulty: 'medium',
    xpReward: 50,
    cosmeticRewardId: 'gradient_math_matrix',
    icon: '🔢',
    requirement: { type: 'mini_game', target: 1 },
  },
  {
    id: 'color_memory',
    title: 'Color Cascade',
    description: 'Remember and repeat the color sequence',
    type: 'mini_game',
    miniGameType: 'color_sequence',
    difficulty: 'medium',
    xpReward: 60,
    cosmeticRewardId: 'gradient_color_memory',
    icon: '🌈',
    requirement: { type: 'mini_game', target: 1 },
  },
  {
    id: 'card_match',
    title: 'Memory Match',
    description: 'Find all matching pairs',
    type: 'mini_game',
    miniGameType: 'card_memory',
    difficulty: 'medium',
    xpReward: 70,
    cosmeticRewardId: 'card_memory_master',
    icon: '🃏',
    requirement: { type: 'mini_game', target: 1 },
  },
  {
    id: 'word_unscramble',
    title: 'Word Wizard',
    description: 'Unscramble the wellness word',
    type: 'mini_game',
    miniGameType: 'word_scramble',
    difficulty: 'easy',
    xpReward: 40,
    cosmeticRewardId: 'gradient_word_flow',
    icon: '📝',
    requirement: { type: 'mini_game', target: 1 },
  },
  {
    id: 'reaction_test',
    title: 'Quick Reflexes',
    description: 'Test your reaction speed',
    type: 'mini_game',
    miniGameType: 'reaction_time',
    difficulty: 'easy',
    xpReward: 35,
    cosmeticRewardId: 'accent_quick_pulse',
    icon: '⚡',
    requirement: { type: 'mini_game', target: 1 },
  },
  {
    id: 'pattern_recall',
    title: 'Pattern Master',
    description: 'Memorize and recreate visual patterns',
    type: 'mini_game',
    miniGameType: 'pattern_match',
    difficulty: 'medium',
    xpReward: 55,
    cosmeticRewardId: 'gradient_pattern_grid',
    icon: '🔲',
    requirement: { type: 'mini_game', target: 1 },
  },
  {
    id: 'speed_calc',
    title: 'Speed Calculator',
    description: 'Solve math problems against the clock',
    type: 'mini_game',
    miniGameType: 'speed_math',
    difficulty: 'hard',
    xpReward: 80,
    cosmeticRewardId: 'gradient_speed_surge',
    icon: '🧮',
    requirement: { type: 'mini_game', target: 1 },
  },

  // BRAIN TASK CHALLENGES - Category specific
  {
    id: 'brain_cortical',
    title: 'Mind Focus',
    description: 'Complete a cortical focus challenge',
    type: 'brain_task',
    brainCategory: 'cortical',
    difficulty: 'medium',
    xpReward: 50,
    cosmeticRewardId: 'gradient_cortical_mind',
    icon: '🧠',
    requirement: { type: 'count', target: 1, category: 'cortical' },
  },
  {
    id: 'brain_limbic',
    title: 'Emotional Check-In',
    description: 'Complete a limbic emotional awareness task',
    type: 'brain_task',
    brainCategory: 'limbic',
    difficulty: 'easy',
    xpReward: 40,
    cosmeticRewardId: 'gradient_limbic_emotion',
    icon: '💗',
    requirement: { type: 'count', target: 1, category: 'limbic' },
  },
  {
    id: 'brain_motor',
    title: 'Movement Break',
    description: 'Complete a physical movement challenge',
    type: 'brain_task',
    brainCategory: 'motor',
    difficulty: 'easy',
    xpReward: 35,
    cosmeticRewardId: 'gradient_motor_energy',
    icon: '🏃',
    requirement: { type: 'count', target: 1, category: 'motor' },
  },
  {
    id: 'brain_sensory',
    title: 'Sensory Awareness',
    description: 'Complete a mindful sensory exercise',
    type: 'brain_task',
    brainCategory: 'sensory',
    difficulty: 'easy',
    xpReward: 35,
    cosmeticRewardId: 'gradient_sensory_awareness',
    icon: '👁️',
    requirement: { type: 'count', target: 1, category: 'sensory' },
  },
  {
    id: 'brain_brainstem',
    title: 'Breathing Space',
    description: 'Complete a breathing or relaxation exercise',
    type: 'brain_task',
    brainCategory: 'brainstem',
    difficulty: 'easy',
    xpReward: 30,
    cosmeticRewardId: 'gradient_brainstem_vital',
    icon: '🫁',
    requirement: { type: 'count', target: 1, category: 'brainstem' },
  },

  // SECRET CHALLENGES
  {
    id: 'secret_night_owl',
    title: '???',
    description: 'Unlocks at a special time...',
    type: 'secret',
    isSecret: true,
    hint: 'The moon watches those who work late',
    difficulty: 'hard',
    xpReward: 200,
    cosmeticRewardId: 'gradient_midnight_secret',
    icon: '🦉',
    secretCondition: { type: 'night_owl' },
    requirement: { type: 'secret_trigger', target: 1 },
  },
  {
    id: 'secret_early_bird',
    title: '???',
    description: 'Unlocks at a special time...',
    type: 'secret',
    isSecret: true,
    hint: 'The early bird catches the worm',
    difficulty: 'hard',
    xpReward: 200,
    cosmeticRewardId: 'gradient_dawn_secret',
    icon: '🐦',
    secretCondition: { type: 'early_bird' },
    requirement: { type: 'secret_trigger', target: 1 },
  },
  {
    id: 'secret_perfect_day',
    title: '???',
    description: 'A legendary achievement awaits...',
    type: 'secret',
    isSecret: true,
    hint: 'When all metrics align in harmony above 80',
    difficulty: 'legendary',
    xpReward: 500,
    cosmeticRewardId: 'gradient_perfect_harmony',
    icon: '✨',
    secretCondition: { type: 'perfect_metrics', value: 80 },
    requirement: { type: 'secret_trigger', target: 1 },
  },
  {
    id: 'secret_streak_master',
    title: '???',
    description: 'Dedication reveals secrets...',
    type: 'secret',
    isSecret: true,
    hint: '30 days of unwavering commitment',
    difficulty: 'legendary',
    xpReward: 1000,
    cosmeticRewardId: 'gradient_legendary_flame',
    icon: '🏆',
    secretCondition: { type: 'streak_milestone', value: 30 },
    requirement: { type: 'secret_trigger', target: 1 },
  },
];

// ============================================================================
// MINI GAME COSMETIC REWARDS
// ============================================================================

export const MINI_GAME_COSMETICS = [
  {
    id: 'gradient_music_flow',
    name: 'Music Flow',
    colors: ['#8B5CF6', '#06B6D4', '#10B981'],
    type: 'gradient' as const,
    rarity: 'uncommon' as const,
  },
  {
    id: 'accent_beat_pulse',
    name: 'Beat Pulse',
    colors: ['#EC4899', '#F472B6'],
    type: 'accent_color' as const,
    rarity: 'rare' as const,
  },
  {
    id: 'gradient_time_flow',
    name: 'Time Flow',
    colors: ['#3B82F6', '#6366F1', '#8B5CF6'],
    type: 'gradient' as const,
    rarity: 'rare' as const,
  },
  {
    id: 'button_task_glow',
    name: 'Task Glow',
    colors: ['#22C55E', '#10B981'],
    type: 'button_style' as const,
    rarity: 'uncommon' as const,
  },
  {
    id: 'journal_mindful_ink',
    name: 'Mindful Ink',
    colors: ['#6366F1', '#818CF8'],
    type: 'journal_design' as const,
    rarity: 'uncommon' as const,
  },
  {
    id: 'accent_streak_flame',
    name: 'Streak Flame',
    colors: ['#F97316', '#EF4444'],
    type: 'accent_color' as const,
    rarity: 'uncommon' as const,
  },
  {
    id: 'gradient_streak_inferno',
    name: 'Streak Inferno',
    colors: ['#DC2626', '#F97316', '#FBBF24'],
    type: 'gradient' as const,
    rarity: 'epic' as const,
  },
  {
    id: 'gradient_resonance_wave',
    name: 'Resonance Wave',
    colors: ['#06B6D4', '#22D3EE', '#67E8F9'],
    type: 'gradient' as const,
    rarity: 'rare' as const,
  },
  {
    id: 'gradient_stable_ground',
    name: 'Stable Ground',
    colors: ['#059669', '#10B981', '#34D399'],
    type: 'gradient' as const,
    rarity: 'epic' as const,
  },
  {
    id: 'gradient_smooth_flow',
    name: 'Smooth Flow',
    colors: ['#6366F1', '#A78BFA', '#C4B5FD'],
    type: 'gradient' as const,
    rarity: 'rare' as const,
  },
  {
    id: 'gradient_symbol_burst',
    name: 'Symbol Burst',
    colors: ['#F59E0B', '#FBBF24', '#FDE047'],
    type: 'gradient' as const,
    rarity: 'epic' as const,
  },
  {
    id: 'accent_riddle_gold',
    name: 'Riddle Gold',
    colors: ['#F59E0B', '#D97706'],
    type: 'accent_color' as const,
    rarity: 'common' as const,
  },
  {
    id: 'gradient_math_matrix',
    name: 'Math Matrix',
    colors: ['#10B981', '#059669', '#047857'],
    type: 'gradient' as const,
    rarity: 'uncommon' as const,
  },
  {
    id: 'gradient_color_memory',
    name: 'Color Memory',
    colors: ['#EC4899', '#8B5CF6', '#3B82F6'],
    type: 'gradient' as const,
    rarity: 'rare' as const,
  },
  {
    id: 'card_memory_master',
    name: 'Memory Master',
    colors: ['#6366F1', '#8B5CF6'],
    type: 'card_style' as const,
    rarity: 'rare' as const,
  },
  {
    id: 'gradient_word_flow',
    name: 'Word Flow',
    colors: ['#06B6D4', '#14B8A6', '#10B981'],
    type: 'gradient' as const,
    rarity: 'uncommon' as const,
  },
  {
    id: 'accent_quick_pulse',
    name: 'Quick Pulse',
    colors: ['#FBBF24', '#F59E0B'],
    type: 'accent_color' as const,
    rarity: 'common' as const,
  },
  {
    id: 'gradient_pattern_grid',
    name: 'Pattern Grid',
    colors: ['#8B5CF6', '#6366F1', '#4F46E5'],
    type: 'gradient' as const,
    rarity: 'uncommon' as const,
  },
  {
    id: 'gradient_speed_surge',
    name: 'Speed Surge',
    colors: ['#EF4444', '#F97316', '#FBBF24'],
    type: 'gradient' as const,
    rarity: 'epic' as const,
  },
  {
    id: 'gradient_midnight_secret',
    name: 'Midnight Secret',
    colors: ['#1E1B4B', '#312E81', '#4338CA'],
    type: 'gradient' as const,
    rarity: 'epic' as const,
  },
  {
    id: 'gradient_dawn_secret',
    name: 'Dawn Secret',
    colors: ['#FEF3C7', '#FDE68A', '#FCD34D'],
    type: 'gradient' as const,
    rarity: 'epic' as const,
  },
  {
    id: 'gradient_perfect_harmony',
    name: 'Perfect Harmony',
    colors: ['#F0ABFC', '#A78BFA', '#60A5FA', '#34D399'],
    type: 'gradient' as const,
    rarity: 'legendary' as const,
  },
  {
    id: 'gradient_legendary_flame',
    name: 'Legendary Flame',
    colors: ['#FBBF24', '#F97316', '#DC2626', '#991B1B'],
    type: 'gradient' as const,
    rarity: 'legendary' as const,
  },
];

// ============================================================================
// STORAGE
// ============================================================================

const UNIFIED_STORAGE_KEY = 'pulse-unified-challenges';
const ACTIVE_CHALLENGES_KEY = 'pulse-active-challenges';
// Version number - increment when challenge pool changes to force regeneration
const CHALLENGE_POOL_VERSION = 2;

// ============================================================================
// UNIFIED CHALLENGE SERVICE
// ============================================================================

class UnifiedChallengeService {
  private data: UnifiedChallengeData;
  private activeChallenges: ActiveChallenge[] = [];
  private listeners: Set<() => void> = new Set();

  constructor() {
    this.data = this.loadData();
    this.activeChallenges = this.loadActiveChallenges();
    this.checkAndRefreshDaily();
  }

  private loadData(): UnifiedChallengeData {
    try {
      const stored = localStorage.getItem(UNIFIED_STORAGE_KEY);
      if (stored) {
        return JSON.parse(stored);
      }
    } catch (e) {
      console.error('Failed to load unified challenge data:', e);
    }

    // Also try to migrate from old systems
    return this.migrateFromOldSystems();
  }

  private migrateFromOldSystems(): UnifiedChallengeData {
    let totalXP = 0;
    let currentStreak = 0;
    let longestStreak = 0;
    const unlockedCosmetics: string[] = ['bg_default', 'gradient_default', 'tab_default', 'button_default', 'journal_default', 'metrics_default'];
    const completedChallenges: CompletedChallengeRecord[] = [];

    // Try to migrate from brain region XP
    try {
      const brainXP = localStorage.getItem('brain-region-xp');
      if (brainXP) totalXP += parseInt(brainXP) || 0;

      const brainStreak = localStorage.getItem('brain-region-streak');
      if (brainStreak) currentStreak = Math.max(currentStreak, parseInt(brainStreak) || 0);
    } catch (e) {
      console.error('Failed to migrate brain region data:', e);
    }

    // Try to migrate from challenge rewards
    try {
      const oldChallengeData = localStorage.getItem('pulse-challenge-rewards');
      if (oldChallengeData) {
        const parsed = JSON.parse(oldChallengeData);
        totalXP += parsed.totalXP || 0;
        currentStreak = Math.max(currentStreak, parsed.currentStreak || 0);
        longestStreak = Math.max(longestStreak, parsed.longestStreak || 0);
        if (parsed.unlockedCosmetics) {
          unlockedCosmetics.push(...parsed.unlockedCosmetics.filter((c: string) => !unlockedCosmetics.includes(c)));
        }
        if (parsed.completedChallenges) {
          completedChallenges.push(...parsed.completedChallenges);
        }
      }
    } catch (e) {
      console.error('Failed to migrate challenge rewards data:', e);
    }

    longestStreak = Math.max(longestStreak, currentStreak);

    return {
      totalXP,
      level: Math.floor(totalXP / 100) + 1,
      currentStreak,
      longestStreak,
      lastActiveDate: '',
      completedChallenges,
      unlockedCosmetics,
      equippedCosmetics: {
        background: 'bg_default',
        gradient: 'gradient_default',
        animation: null,
        wallpaper: null,
        tab_design: 'tab_default',
        button_style: 'button_default',
        metrics_theme: 'metrics_default',
        journal_design: 'journal_default',
        card_style: null,
        accent_color: null,
      },
      pinnedChallengeIds: [],
      secretsDiscovered: [],
      miniGameHighScores: {},
      miniGameXPClaimsToday: 0,
      lastMiniGameXPDate: '',
    };
  }

  private saveData(): void {
    try {
      localStorage.setItem(UNIFIED_STORAGE_KEY, JSON.stringify(this.data));
      this.notifyListeners();
    } catch (e) {
      console.error('Failed to save unified challenge data:', e);
    }
  }

  private loadActiveChallenges(): ActiveChallenge[] {
    try {
      const stored = localStorage.getItem(ACTIVE_CHALLENGES_KEY);
      if (stored) {
        return JSON.parse(stored);
      }
    } catch (e) {
      console.error('Failed to load active challenges:', e);
    }
    return [];
  }

  private saveActiveChallenges(): void {
    try {
      localStorage.setItem(ACTIVE_CHALLENGES_KEY, JSON.stringify(this.activeChallenges));
      this.notifyListeners();
    } catch (e) {
      console.error('Failed to save active challenges:', e);
    }
  }

  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private notifyListeners(): void {
    this.listeners.forEach(listener => listener());
  }

  // ---------------------------------------------------------------------------
  // DAILY REFRESH
  // ---------------------------------------------------------------------------

  private getTodayKey(): string {
    return new Date().toISOString().split('T')[0];
  }

  private checkAndRefreshDaily(): void {
    const today = this.getTodayKey();
    // Check stored version to force regeneration when pool changes
    const storedVersion = localStorage.getItem('pulse-challenges-version');
    const needsVersionUpdate = storedVersion !== String(CHALLENGE_POOL_VERSION);

    if (this.activeChallenges.length === 0 || this.activeChallenges[0]?.date !== today || needsVersionUpdate) {
      this.generateDailyChallenges();
      localStorage.setItem('pulse-challenges-version', String(CHALLENGE_POOL_VERSION));
    }
  }

  generateDailyChallenges(): ActiveChallenge[] {
    const today = this.getTodayKey();
    const seed = hashCode(today + 'unified');
    const completedIds = new Set(this.data.completedChallenges.map(c => c.challengeId));

    // Filter available challenges
    const available = UNIFIED_CHALLENGE_POOL.filter(c => {
      // Always include mini-games (they reset daily)
      if (c.type === 'mini_game') return true;
      // Exclude completed legendary/secret challenges
      if ((c.difficulty === 'legendary' || c.type === 'secret') && completedIds.has(c.id)) return false;
      return true;
    });

    // Shuffle with seeded random
    const seededRandom = () => {
      const x = Math.sin(seed + available.length) * 10000;
      return x - Math.floor(x);
    };
    const shuffled = [...available].sort(() => seededRandom() - 0.5);

    // Always include ALL mini-games (they're always accessible)
    const allMiniGames = available.filter(c => c.type === 'mini_game');
    const nonMiniGames = shuffled.filter(c => c.type !== 'mini_game' && c.type !== 'secret');

    // Select balanced mix of non-mini-game challenges: 2 easy, 2 medium, 1 hard
    const selected: UnifiedChallenge[] = [...allMiniGames]; // Start with all mini-games
    const typeCount = { easy: 0, medium: 0, hard: 0 };

    for (const challenge of nonMiniGames) {
      if (typeCount.easy >= 2 && typeCount.medium >= 2 && typeCount.hard >= 1) break;

      if (challenge.difficulty === 'easy' && typeCount.easy < 2) {
        selected.push(challenge);
        typeCount.easy++;
      } else if (challenge.difficulty === 'medium' && typeCount.medium < 2) {
        selected.push(challenge);
        typeCount.medium++;
      } else if (challenge.difficulty === 'hard' && typeCount.hard < 1) {
        selected.push(challenge);
        typeCount.hard++;
      }
    }

    // Add secret challenges (hidden by default)
    const secrets = UNIFIED_CHALLENGE_POOL.filter(c =>
      c.type === 'secret' && !completedIds.has(c.id) && !this.data.secretsDiscovered.includes(c.id)
    );

    // Convert to active challenges
    this.activeChallenges = selected.map(challenge => ({
      ...challenge,
      date: today,
      progress: 0,
      completed: false,
      isPinned: this.data.pinnedChallengeIds.includes(challenge.id),
      isHighlighted: false,
      // Add mini-game data if applicable
      miniGameData: challenge.miniGameType ? this.generateMiniGameData(challenge.miniGameType, today) : undefined,
    }));

    // Add hidden secrets
    for (const secret of secrets) {
      this.activeChallenges.push({
        ...secret,
        date: today,
        progress: 0,
        completed: false,
        isPinned: false,
        isHighlighted: false,
      });
    }

    this.saveActiveChallenges();
    return this.activeChallenges;
  }

  private generateMiniGameData(type: MiniGameType, date: string): MiniGameData {
    switch (type) {
      case 'riddle':
        return generateDailyRiddle(date);
      case 'math_series':
        return generateDailyMathSeries(date);
      case 'color_sequence':
        return generateColorSequence(4);
      case 'card_memory':
        return generateCardMemoryGame(6);
      case 'word_scramble':
        return generateWordScramble(date);
      case 'reaction_time':
        return generateReactionTime();
      case 'pattern_match':
        return generatePatternMatch();
      case 'speed_math':
        return generateSpeedMath();
      default:
        return {};
    }
  }

  // ---------------------------------------------------------------------------
  // CHALLENGE OPERATIONS
  // ---------------------------------------------------------------------------

  getActiveChallenges(includeSecrets: boolean = false): ActiveChallenge[] {
    this.checkAndRefreshDaily();
    if (includeSecrets) return this.activeChallenges;
    return this.activeChallenges.filter(c => !c.isSecret || this.data.secretsDiscovered.includes(c.id));
  }

  getMiniGameChallenges(): ActiveChallenge[] {
    return this.activeChallenges.filter(c => c.type === 'mini_game' && !c.completed);
  }

  getSecretHints(): Array<{ id: string; hint: string; discovered: boolean }> {
    return UNIFIED_CHALLENGE_POOL
      .filter(c => c.type === 'secret' && c.hint)
      .map(c => ({
        id: c.id,
        hint: c.hint!,
        discovered: this.data.secretsDiscovered.includes(c.id),
      }));
  }

  checkSecretConditions(metrics: { symbolic: number; resonance: number; friction: number; stability: number }): void {
    const now = new Date();
    const hour = now.getHours();

    for (const challenge of this.activeChallenges) {
      if (challenge.type !== 'secret' || !challenge.secretCondition) continue;
      if (this.data.secretsDiscovered.includes(challenge.id)) continue;

      let triggered = false;

      switch (challenge.secretCondition.type) {
        case 'night_owl':
          triggered = hour >= 23 || hour < 4;
          break;
        case 'early_bird':
          triggered = hour >= 5 && hour < 7;
          break;
        case 'perfect_metrics':
          const threshold = challenge.secretCondition.value || 80;
          triggered = metrics.symbolic >= threshold && metrics.resonance >= threshold &&
                     metrics.friction <= (100 - threshold) && metrics.stability >= threshold;
          break;
        case 'streak_milestone':
          triggered = this.data.currentStreak >= (challenge.secretCondition.value || 30);
          break;
      }

      if (triggered) {
        this.discoverSecret(challenge.id);
      }
    }
  }

  private discoverSecret(challengeId: string): void {
    if (!this.data.secretsDiscovered.includes(challengeId)) {
      this.data.secretsDiscovered.push(challengeId);

      const challenge = this.activeChallenges.find(c => c.id === challengeId);
      if (challenge) {
        try {
          notificationService.show({
            type: 'challenge_complete',
            title: '🔮 Secret Discovered!',
            body: `You've unlocked a hidden challenge: ${challenge.title}`,
          });
        } catch (e) {
          console.error('Failed to send secret notification:', e);
        }
      }

      this.saveData();
      this.notifyListeners();
    }
  }

  completeChallenge(challengeId: string): { xpEarned: number; cosmeticUnlocked: string | null; levelUp: boolean } | null {
    const challenge = this.activeChallenges.find(c => c.id === challengeId);
    if (!challenge || challenge.completed) return null;

    challenge.completed = true;
    challenge.completedAt = new Date().toISOString();
    challenge.progress = 100;

    const oldLevel = this.data.level;
    this.data.totalXP += challenge.xpReward;
    this.data.level = Math.floor(this.data.totalXP / 100) + 1;
    const levelUp = this.data.level > oldLevel;

    // Update streak
    const today = this.getTodayKey();
    if (this.data.lastActiveDate !== today) {
      const yesterday = new Date();
      yesterday.setDate(yesterday.getDate() - 1);
      const yesterdayKey = yesterday.toISOString().split('T')[0];

      if (this.data.lastActiveDate === yesterdayKey) {
        this.data.currentStreak++;
      } else if (this.data.lastActiveDate !== today) {
        this.data.currentStreak = 1;
      }

      this.data.lastActiveDate = today;
      this.data.longestStreak = Math.max(this.data.longestStreak, this.data.currentStreak);
    }

    // Record completion
    this.data.completedChallenges.push({
      challengeId,
      completedAt: challenge.completedAt,
      xpEarned: challenge.xpReward,
      cosmeticUnlocked: challenge.cosmeticRewardId,
    });

    // Unlock cosmetic
    let cosmeticUnlocked: string | null = null;
    if (!this.data.unlockedCosmetics.includes(challenge.cosmeticRewardId)) {
      this.data.unlockedCosmetics.push(challenge.cosmeticRewardId);
      cosmeticUnlocked = challenge.cosmeticRewardId;
    }

    // Auto-unpin
    if (challenge.isPinned) {
      this.unpinChallenge(challengeId);
    }

    this.saveData();
    this.saveActiveChallenges();

    return { xpEarned: challenge.xpReward, cosmeticUnlocked, levelUp };
  }

  /**
   * Complete a mini-game - always playable but XP only awarded 3 times per day
   */
  completeMiniGame(challengeId: string, score: number): { xpEarned: number; cosmeticUnlocked: string | null; highScore: boolean; xpClaimsRemaining: number } | null {
    const today = this.getTodayKey();
    const challenge = UNIFIED_CHALLENGE_POOL.find(c => c.id === challengeId);
    const gameType = challenge?.miniGameType;

    // Reset daily counter if it's a new day
    if (this.data.lastMiniGameXPDate !== today) {
      this.data.miniGameXPClaimsToday = 0;
      this.data.lastMiniGameXPDate = today;
    }

    // Check if we can still earn XP (max 3 per day)
    const canEarnXP = this.data.miniGameXPClaimsToday < 3;

    let xpEarned = 0;
    let cosmeticUnlocked: string | null = null;
    let levelUp = false;

    if (canEarnXP && challenge) {
      // Award XP and process reward
      xpEarned = challenge.xpReward;
      this.data.totalXP += xpEarned;
      this.data.miniGameXPClaimsToday++;

      const newLevel = Math.floor(this.data.totalXP / 100) + 1;
      if (newLevel > this.data.level) {
        this.data.level = newLevel;
        levelUp = true;
      }

      // Unlock cosmetic if not already owned
      if (challenge.cosmeticRewardId && !this.data.unlockedCosmetics.includes(challenge.cosmeticRewardId)) {
        this.data.unlockedCosmetics.push(challenge.cosmeticRewardId);
        cosmeticUnlocked = challenge.cosmeticRewardId;
      }

      // Record completion
      this.data.completedChallenges.push({
        challengeId,
        completedAt: new Date().toISOString(),
        xpEarned,
        cosmeticUnlocked: cosmeticUnlocked || undefined,
      });

      // Send notification (schedule immediately)
      notificationService.scheduleNotification({
        title: levelUp ? '🎮 Level Up!' : '🎮 Mini-Game Complete!',
        body: levelUp
          ? `You reached Level ${this.data.level}! +${xpEarned} XP`
          : `+${xpEarned} XP earned! ${3 - this.data.miniGameXPClaimsToday} XP rewards remaining today.`,
        tag: 'mini-game',
        type: 'challenge_complete',
      }, new Date());
    }

    // Track high score regardless of XP
    let highScore = false;
    if (gameType) {
      const currentHigh = this.data.miniGameHighScores[gameType] || 0;
      if (score > currentHigh) {
        this.data.miniGameHighScores[gameType] = score;
        highScore = true;
      }
    }

    this.saveData();
    return { xpEarned, cosmeticUnlocked, highScore, xpClaimsRemaining: Math.max(0, 3 - this.data.miniGameXPClaimsToday) };
  }

  /**
   * Get remaining XP claims for mini-games today
   */
  getMiniGameXPClaimsRemaining(): number {
    const today = this.getTodayKey();
    if (this.data.lastMiniGameXPDate !== today) {
      return 3; // New day, all 3 available
    }
    return Math.max(0, 3 - this.data.miniGameXPClaimsToday);
  }

  pinChallenge(challengeId: string): void {
    if (!this.data.pinnedChallengeIds.includes(challengeId)) {
      if (this.data.pinnedChallengeIds.length >= 5) {
        this.data.pinnedChallengeIds.shift();
      }
      this.data.pinnedChallengeIds.push(challengeId);

      const challenge = this.activeChallenges.find(c => c.id === challengeId);
      if (challenge) challenge.isPinned = true;

      this.saveData();
      this.saveActiveChallenges();
    }
  }

  unpinChallenge(challengeId: string): void {
    this.data.pinnedChallengeIds = this.data.pinnedChallengeIds.filter(id => id !== challengeId);

    const challenge = this.activeChallenges.find(c => c.id === challengeId);
    if (challenge) challenge.isPinned = false;

    this.saveData();
    this.saveActiveChallenges();
  }

  // ---------------------------------------------------------------------------
  // XP AND STATS
  // ---------------------------------------------------------------------------

  addXP(amount: number): { levelUp: boolean; newLevel: number } {
    const oldLevel = this.data.level;
    this.data.totalXP += amount;
    this.data.level = Math.floor(this.data.totalXP / 100) + 1;
    this.saveData();

    return {
      levelUp: this.data.level > oldLevel,
      newLevel: this.data.level,
    };
  }

  getStats(): {
    totalXP: number;
    level: number;
    xpToNextLevel: number;
    levelProgress: number;
    currentStreak: number;
    longestStreak: number;
    totalCompleted: number;
    unlockedCount: number;
    secretsFound: number;
    totalSecrets: number;
    miniGameXPClaimsRemaining: number;
  } {
    const xpInLevel = this.data.totalXP % 100;
    const totalSecrets = UNIFIED_CHALLENGE_POOL.filter(c => c.type === 'secret').length;

    return {
      totalXP: this.data.totalXP,
      level: this.data.level,
      xpToNextLevel: 100 - xpInLevel,
      levelProgress: xpInLevel,
      currentStreak: this.data.currentStreak,
      longestStreak: this.data.longestStreak,
      totalCompleted: this.data.completedChallenges.length,
      unlockedCount: this.data.unlockedCosmetics.length,
      secretsFound: this.data.secretsDiscovered.length,
      totalSecrets,
      miniGameXPClaimsRemaining: this.getMiniGameXPClaimsRemaining(),
    };
  }

  // ---------------------------------------------------------------------------
  // COSMETICS
  // ---------------------------------------------------------------------------

  getUnlockedCosmetics(): string[] {
    return [...this.data.unlockedCosmetics];
  }

  getEquippedCosmetics(): Record<CosmeticCategory, string | null> {
    return { ...this.data.equippedCosmetics };
  }

  equipCosmetic(cosmeticId: string, category: CosmeticCategory): boolean {
    if (!this.data.unlockedCosmetics.includes(cosmeticId)) return false;
    this.data.equippedCosmetics[category] = cosmeticId;
    this.saveData();
    return true;
  }

  unequipCosmetic(category: CosmeticCategory): void {
    const defaultId = `${category.split('_')[0]}_default`;
    this.data.equippedCosmetics[category] = this.data.unlockedCosmetics.includes(defaultId) ? defaultId : null;
    this.saveData();
  }

  getBrainRegionGradient(category: BrainRegionCategory): typeof BRAIN_REGION_GRADIENTS[BrainRegionCategory] {
    return BRAIN_REGION_GRADIENTS[category];
  }
}

// ============================================================================
// SINGLETON EXPORT
// ============================================================================

export const unifiedChallengeService = new UnifiedChallengeService();
