// Challenge Reward System - Daily challenges with cosmetic rewards
// Challenges refresh daily and unlock aesthetic improvements only

import { notificationService } from './notificationService';
import { COSMETIC_CATALOG } from './cosmeticDefinitions';

// ============================================================================
// TYPES AND INTERFACES
// ============================================================================

export type ChallengeCategory =
  | 'wellness'
  | 'productivity'
  | 'mindfulness'
  | 'social'
  | 'creativity'
  | 'physical'
  | 'learning';

export type ChallengeDifficulty = 'easy' | 'medium' | 'hard' | 'legendary';

export type CosmeticType =
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

export interface CosmeticReward {
  id: string;
  name: string;
  description: string;
  type: CosmeticType;
  rarity: 'common' | 'uncommon' | 'rare' | 'epic' | 'legendary';
  cssClass?: string;
  cssVars?: Record<string, string>;
  preview?: string; // Preview description or emoji
}

export interface Challenge {
  id: string;
  title: string;
  description: string;
  category: ChallengeCategory;
  difficulty: ChallengeDifficulty;
  requirements: ChallengeRequirement[];
  cosmetic_reward_id: string;
  xp_reward: number;
  time_estimate: string;
  icon: string;
  tips?: string[];
}

export interface ChallengeRequirement {
  type: 'check_in' | 'journal' | 'metric_threshold' | 'streak' | 'time_spent' | 'category_complete';
  target: string | number;
  current?: number;
  category?: string;
  metric?: string;
  threshold?: number;
  comparison?: 'gte' | 'lte' | 'eq';
}

export interface DailyChallenge extends Challenge {
  date: string; // YYYY-MM-DD
  progress: number; // 0-100
  completed: boolean;
  completedAt?: string;
  isPinned: boolean;
  isHighlighted: boolean;
  highlightReason?: string;
}

export interface UserChallengeData {
  completedChallenges: CompletedChallenge[];
  unlockedCosmetics: string[];
  equippedCosmetics: Record<CosmeticType, string | null>;
  pinnedChallengeIds: string[];
  totalXP: number;
  currentStreak: number;
  longestStreak: number;
  lastActiveDate: string;
  challengeHistory: ChallengeHistoryEntry[];
  dailyChallengesSeed: string; // For consistent daily generation
}

export interface CompletedChallenge {
  challengeId: string;
  completedAt: string;
  xpEarned: number;
  cosmeticUnlocked?: string;
}

export interface ChallengeHistoryEntry {
  date: string;
  challengeIds: string[];
  completedIds: string[];
}

export interface UserMetrics {
  checkInsToday: number;
  journalEntriesToday: number;
  currentStreak: number;
  deltaHV: {
    symbolic: number;
    resonance: number;
    friction: number;
    stability: number;
  };
  categoryCompletions: Record<string, number>;
  totalCheckIns: number;
  totalJournalEntries: number;
  averageDailyCheckIns: number;
  weakCategories: string[];
  strongCategories: string[];
}

// ============================================================================
// STORAGE KEYS
// ============================================================================

const STORAGE_KEYS = {
  CHALLENGE_DATA: 'pulse-challenge-rewards',
  DAILY_CHALLENGES: 'pulse-daily-challenges',
  LAST_NOTIFICATION_DATE: 'pulse-challenge-notification-date',
};

// ============================================================================
// CHALLENGE DEFINITIONS - All possible challenges
// ============================================================================

export const CHALLENGE_POOL: Challenge[] = [
  // WELLNESS CHALLENGES
  {
    id: 'wellness_morning_routine',
    title: 'Morning Momentum',
    description: 'Complete 3 check-ins before noon to start your day strong',
    category: 'wellness',
    difficulty: 'easy',
    requirements: [{ type: 'check_in', target: 3, category: 'morning' }],
    cosmetic_reward_id: 'bg_sunrise_gradient',
    xp_reward: 50,
    time_estimate: '~2 hours',
    icon: '🌅',
    tips: ['Set a morning alarm', 'Prepare the night before']
  },
  {
    id: 'wellness_hydration_hero',
    title: 'Hydration Hero',
    description: 'Log your water intake 5 times throughout the day',
    category: 'wellness',
    difficulty: 'easy',
    requirements: [{ type: 'check_in', target: 5, category: 'hydration' }],
    cosmetic_reward_id: 'bg_ocean_waves',
    xp_reward: 40,
    time_estimate: 'All day',
    icon: '💧'
  },
  {
    id: 'wellness_sleep_guardian',
    title: 'Sleep Guardian',
    description: 'Log your sleep quality for 7 consecutive days',
    category: 'wellness',
    difficulty: 'medium',
    requirements: [{ type: 'streak', target: 7, category: 'sleep' }],
    cosmetic_reward_id: 'bg_starry_night',
    xp_reward: 100,
    time_estimate: '7 days',
    icon: '🌙'
  },
  {
    id: 'wellness_meditation_master',
    title: 'Meditation Master',
    description: 'Complete 5 meditation sessions this week',
    category: 'wellness',
    difficulty: 'medium',
    requirements: [{ type: 'category_complete', target: 5, category: 'Meditation' }],
    cosmetic_reward_id: 'animation_breathing_pulse',
    xp_reward: 80,
    time_estimate: '~1 week',
    icon: '🧘'
  },
  {
    id: 'wellness_full_day',
    title: 'Wellness Warrior',
    description: 'Complete all wellness-related check-ins in a single day',
    category: 'wellness',
    difficulty: 'hard',
    requirements: [{ type: 'category_complete', target: 'all', category: 'wellness' }],
    cosmetic_reward_id: 'tab_wellness_aurora',
    xp_reward: 150,
    time_estimate: 'Full day',
    icon: '⚡'
  },

  // PRODUCTIVITY CHALLENGES
  {
    id: 'productivity_focus_flow',
    title: 'Focus Flow',
    description: 'Maintain high stability (>70) for 4 hours',
    category: 'productivity',
    difficulty: 'medium',
    requirements: [{ type: 'metric_threshold', metric: 'stability', threshold: 70, target: 4 }],
    cosmetic_reward_id: 'metrics_neon_glow',
    xp_reward: 90,
    time_estimate: '4 hours',
    icon: '🎯'
  },
  {
    id: 'productivity_task_titan',
    title: 'Task Titan',
    description: 'Complete 10 check-ins in a single day',
    category: 'productivity',
    difficulty: 'medium',
    requirements: [{ type: 'check_in', target: 10 }],
    cosmetic_reward_id: 'button_metallic_shine',
    xp_reward: 85,
    time_estimate: 'Full day',
    icon: '🏆'
  },
  {
    id: 'productivity_streak_starter',
    title: 'Streak Starter',
    description: 'Build a 3-day activity streak',
    category: 'productivity',
    difficulty: 'easy',
    requirements: [{ type: 'streak', target: 3 }],
    cosmetic_reward_id: 'accent_flame_orange',
    xp_reward: 60,
    time_estimate: '3 days',
    icon: '🔥'
  },
  {
    id: 'productivity_week_warrior',
    title: 'Week Warrior',
    description: 'Maintain a 7-day streak',
    category: 'productivity',
    difficulty: 'hard',
    requirements: [{ type: 'streak', target: 7 }],
    cosmetic_reward_id: 'wallpaper_geometric_gold',
    xp_reward: 200,
    time_estimate: '7 days',
    icon: '👑'
  },
  {
    id: 'productivity_friction_fighter',
    title: 'Friction Fighter',
    description: 'Keep friction below 30 for an entire day',
    category: 'productivity',
    difficulty: 'hard',
    requirements: [{ type: 'metric_threshold', metric: 'friction', threshold: 30, comparison: 'lte', target: 1 }],
    cosmetic_reward_id: 'gradient_smooth_silk',
    xp_reward: 120,
    time_estimate: 'Full day',
    icon: '🛡️'
  },

  // MINDFULNESS CHALLENGES
  {
    id: 'mindfulness_journal_journey',
    title: 'Journal Journey',
    description: 'Write 3 journal entries today',
    category: 'mindfulness',
    difficulty: 'easy',
    requirements: [{ type: 'journal', target: 3 }],
    cosmetic_reward_id: 'journal_parchment_classic',
    xp_reward: 45,
    time_estimate: '~30 min',
    icon: '📝'
  },
  {
    id: 'mindfulness_reflection_ritual',
    title: 'Reflection Ritual',
    description: 'Complete morning and evening journal entries for 3 days',
    category: 'mindfulness',
    difficulty: 'medium',
    requirements: [{ type: 'journal', target: 6 }],
    cosmetic_reward_id: 'journal_ink_flow',
    xp_reward: 100,
    time_estimate: '3 days',
    icon: '🔮'
  },
  {
    id: 'mindfulness_gratitude_guru',
    title: 'Gratitude Guru',
    description: 'Write entries focusing on gratitude for 5 days',
    category: 'mindfulness',
    difficulty: 'medium',
    requirements: [{ type: 'journal', target: 5, category: 'gratitude' }],
    cosmetic_reward_id: 'animation_gentle_glow',
    xp_reward: 90,
    time_estimate: '5 days',
    icon: '🙏'
  },
  {
    id: 'mindfulness_emotion_explorer',
    title: 'Emotion Explorer',
    description: 'Log emotions across all 8 emotional categories',
    category: 'mindfulness',
    difficulty: 'hard',
    requirements: [{ type: 'category_complete', target: 8, category: 'emotions' }],
    cosmetic_reward_id: 'bg_emotional_spectrum',
    xp_reward: 140,
    time_estimate: '~1 week',
    icon: '🎭'
  },
  {
    id: 'mindfulness_deep_dive',
    title: 'Deep Dive',
    description: 'Write a journal entry over 500 words',
    category: 'mindfulness',
    difficulty: 'medium',
    requirements: [{ type: 'journal', target: 500, category: 'word_count' }],
    cosmetic_reward_id: 'journal_deep_purple',
    xp_reward: 75,
    time_estimate: '~20 min',
    icon: '🌊'
  },

  // SOCIAL CHALLENGES
  {
    id: 'social_connection_catalyst',
    title: 'Connection Catalyst',
    description: 'Log 3 social interactions today',
    category: 'social',
    difficulty: 'easy',
    requirements: [{ type: 'check_in', target: 3, category: 'social' }],
    cosmetic_reward_id: 'accent_warm_coral',
    xp_reward: 50,
    time_estimate: 'All day',
    icon: '🤝'
  },
  {
    id: 'social_community_builder',
    title: 'Community Builder',
    description: 'Complete social check-ins for 5 consecutive days',
    category: 'social',
    difficulty: 'medium',
    requirements: [{ type: 'streak', target: 5, category: 'social' }],
    cosmetic_reward_id: 'tab_social_gradient',
    xp_reward: 110,
    time_estimate: '5 days',
    icon: '🏘️'
  },
  {
    id: 'social_resonance_rise',
    title: 'Resonance Rise',
    description: 'Achieve resonance score above 80',
    category: 'social',
    difficulty: 'hard',
    requirements: [{ type: 'metric_threshold', metric: 'resonance', threshold: 80, target: 1 }],
    cosmetic_reward_id: 'animation_ripple_effect',
    xp_reward: 130,
    time_estimate: 'Variable',
    icon: '📡'
  },

  // CREATIVITY CHALLENGES
  {
    id: 'creativity_artistic_awakening',
    title: 'Artistic Awakening',
    description: 'Complete 3 creative activities today',
    category: 'creativity',
    difficulty: 'easy',
    requirements: [{ type: 'check_in', target: 3, category: 'creative' }],
    cosmetic_reward_id: 'bg_paint_splash',
    xp_reward: 55,
    time_estimate: '~2 hours',
    icon: '🎨'
  },
  {
    id: 'creativity_flow_state',
    title: 'Flow State',
    description: 'Enter and maintain flow rhythm for 2 hours',
    category: 'creativity',
    difficulty: 'medium',
    requirements: [{ type: 'time_spent', target: 120, category: 'flow' }],
    cosmetic_reward_id: 'gradient_creative_burst',
    xp_reward: 95,
    time_estimate: '2 hours',
    icon: '🌀'
  },
  {
    id: 'creativity_muse_master',
    title: 'Muse Master',
    description: 'Complete creative activities for 7 days straight',
    category: 'creativity',
    difficulty: 'hard',
    requirements: [{ type: 'streak', target: 7, category: 'creative' }],
    cosmetic_reward_id: 'wallpaper_abstract_art',
    xp_reward: 180,
    time_estimate: '7 days',
    icon: '✨'
  },
  {
    id: 'creativity_symbolic_surge',
    title: 'Symbolic Surge',
    description: 'Achieve symbolic score above 85',
    category: 'creativity',
    difficulty: 'hard',
    requirements: [{ type: 'metric_threshold', metric: 'symbolic', threshold: 85, target: 1 }],
    cosmetic_reward_id: 'metrics_rainbow_shift',
    xp_reward: 140,
    time_estimate: 'Variable',
    icon: '🌈'
  },

  // PHYSICAL CHALLENGES
  {
    id: 'physical_workout_warrior',
    title: 'Workout Warrior',
    description: 'Complete 3 workout sessions this week',
    category: 'physical',
    difficulty: 'medium',
    requirements: [{ type: 'category_complete', target: 3, category: 'Workout' }],
    cosmetic_reward_id: 'button_power_pulse',
    xp_reward: 80,
    time_estimate: '~1 week',
    icon: '💪'
  },
  {
    id: 'physical_movement_momentum',
    title: 'Movement Momentum',
    description: 'Log physical activity every day for 5 days',
    category: 'physical',
    difficulty: 'medium',
    requirements: [{ type: 'streak', target: 5, category: 'physical' }],
    cosmetic_reward_id: 'animation_energy_wave',
    xp_reward: 100,
    time_estimate: '5 days',
    icon: '🏃'
  },
  {
    id: 'physical_iron_will',
    title: 'Iron Will',
    description: 'Complete 20 workout check-ins total',
    category: 'physical',
    difficulty: 'hard',
    requirements: [{ type: 'category_complete', target: 20, category: 'Workout' }],
    cosmetic_reward_id: 'tab_iron_forge',
    xp_reward: 200,
    time_estimate: '~2-3 weeks',
    icon: '🔩'
  },

  // LEARNING CHALLENGES
  {
    id: 'learning_knowledge_seeker',
    title: 'Knowledge Seeker',
    description: 'Complete 5 learning-related check-ins',
    category: 'learning',
    difficulty: 'easy',
    requirements: [{ type: 'check_in', target: 5, category: 'learning' }],
    cosmetic_reward_id: 'accent_wisdom_purple',
    xp_reward: 60,
    time_estimate: '~1 week',
    icon: '📚'
  },
  {
    id: 'learning_growth_mindset',
    title: 'Growth Mindset',
    description: 'Document learning reflections for 7 days',
    category: 'learning',
    difficulty: 'medium',
    requirements: [{ type: 'journal', target: 7, category: 'learning' }],
    cosmetic_reward_id: 'journal_scholar_theme',
    xp_reward: 110,
    time_estimate: '7 days',
    icon: '🌱'
  },
  {
    id: 'learning_master_mind',
    title: 'Master Mind',
    description: 'Achieve stability above 80 while completing learning tasks',
    category: 'learning',
    difficulty: 'hard',
    requirements: [
      { type: 'metric_threshold', metric: 'stability', threshold: 80, target: 1 },
      { type: 'check_in', target: 5, category: 'learning' }
    ],
    cosmetic_reward_id: 'wallpaper_neural_network',
    xp_reward: 160,
    time_estimate: 'Variable',
    icon: '🧠'
  },

  // LEGENDARY CHALLENGES
  {
    id: 'legendary_perfect_day',
    title: 'Perfect Day',
    description: 'Complete check-ins in all categories with all metrics above 70',
    category: 'wellness',
    difficulty: 'legendary',
    requirements: [
      { type: 'category_complete', target: 'all' },
      { type: 'metric_threshold', metric: 'all', threshold: 70, target: 1 }
    ],
    cosmetic_reward_id: 'bg_cosmic_aurora',
    xp_reward: 500,
    time_estimate: 'Full day',
    icon: '🌟'
  },
  {
    id: 'legendary_month_master',
    title: 'Month Master',
    description: 'Maintain a 30-day activity streak',
    category: 'productivity',
    difficulty: 'legendary',
    requirements: [{ type: 'streak', target: 30 }],
    cosmetic_reward_id: 'wallpaper_legendary_flames',
    xp_reward: 1000,
    time_estimate: '30 days',
    icon: '🏅'
  },
  {
    id: 'legendary_zen_master',
    title: 'Zen Master',
    description: 'Complete 100 meditation sessions total',
    category: 'mindfulness',
    difficulty: 'legendary',
    requirements: [{ type: 'category_complete', target: 100, category: 'Meditation' }],
    cosmetic_reward_id: 'animation_zen_particles',
    xp_reward: 800,
    time_estimate: '~3 months',
    icon: '☯️'
  },
  {
    id: 'legendary_journal_sage',
    title: 'Journal Sage',
    description: 'Write 50 journal entries total',
    category: 'mindfulness',
    difficulty: 'legendary',
    requirements: [{ type: 'journal', target: 50 }],
    cosmetic_reward_id: 'journal_legendary_tome',
    xp_reward: 600,
    time_estimate: '~2 months',
    icon: '📜'
  }
];

// ============================================================================
// CHALLENGE REWARD SERVICE
// ============================================================================

class ChallengeRewardService {
  private data: UserChallengeData;
  private dailyChallenges: DailyChallenge[] = [];
  private listeners: Set<() => void> = new Set();

  constructor() {
    this.data = this.loadData();
    this.dailyChallenges = this.loadDailyChallenges();
    this.checkAndRefreshDaily();
  }

  // ---------------------------------------------------------------------------
  // STORAGE OPERATIONS
  // ---------------------------------------------------------------------------

  private loadData(): UserChallengeData {
    try {
      const stored = localStorage.getItem(STORAGE_KEYS.CHALLENGE_DATA);
      if (stored) {
        return JSON.parse(stored);
      }
    } catch (e) {
      console.error('Failed to load challenge data:', e);
    }

    return {
      completedChallenges: [],
      unlockedCosmetics: ['bg_default', 'gradient_default', 'tab_default', 'button_default', 'journal_default', 'metrics_default'],
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
        accent_color: null
      },
      pinnedChallengeIds: [],
      totalXP: 0,
      currentStreak: 0,
      longestStreak: 0,
      lastActiveDate: '',
      challengeHistory: [],
      dailyChallengesSeed: ''
    };
  }

  private saveData(): void {
    try {
      localStorage.setItem(STORAGE_KEYS.CHALLENGE_DATA, JSON.stringify(this.data));
      this.notifyListeners();
    } catch (e) {
      console.error('Failed to save challenge data:', e);
    }
  }

  private loadDailyChallenges(): DailyChallenge[] {
    try {
      const stored = localStorage.getItem(STORAGE_KEYS.DAILY_CHALLENGES);
      if (stored) {
        return JSON.parse(stored);
      }
    } catch (e) {
      console.error('Failed to load daily challenges:', e);
    }
    return [];
  }

  private saveDailyChallenges(): void {
    try {
      localStorage.setItem(STORAGE_KEYS.DAILY_CHALLENGES, JSON.stringify(this.dailyChallenges));
      this.notifyListeners();
    } catch (e) {
      console.error('Failed to save daily challenges:', e);
    }
  }

  // ---------------------------------------------------------------------------
  // LISTENER MANAGEMENT
  // ---------------------------------------------------------------------------

  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private notifyListeners(): void {
    this.listeners.forEach(listener => listener());
  }

  // ---------------------------------------------------------------------------
  // DAILY CHALLENGE GENERATION
  // ---------------------------------------------------------------------------

  private getTodayKey(): string {
    return new Date().toISOString().split('T')[0];
  }

  private seededRandom(seed: string): () => number {
    let hash = 0;
    for (let i = 0; i < seed.length; i++) {
      const char = seed.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash;
    }

    return () => {
      hash = (hash * 1103515245 + 12345) & 0x7fffffff;
      return hash / 0x7fffffff;
    };
  }

  private checkAndRefreshDaily(): void {
    const today = this.getTodayKey();

    // Check if we need to refresh (new day)
    if (this.dailyChallenges.length === 0 || this.dailyChallenges[0]?.date !== today) {
      this.generateDailyChallenges();
    }
  }

  generateDailyChallenges(userMetrics?: UserMetrics): DailyChallenge[] {
    const today = this.getTodayKey();
    const seed = `${today}-pulse-challenges`;
    const random = this.seededRandom(seed);

    // Filter out already completed legendary challenges
    const completedIds = new Set(this.data.completedChallenges.map(c => c.challengeId));
    const availableChallenges = CHALLENGE_POOL.filter(c => {
      // Allow re-doing non-legendary challenges
      if (c.difficulty !== 'legendary') return true;
      // Only allow legendary challenges once
      return !completedIds.has(c.id);
    });

    // Shuffle using seeded random
    const shuffled = [...availableChallenges].sort(() => random() - 0.5);

    // Select challenges - aim for variety
    const selected: Challenge[] = [];
    const categoriesUsed = new Set<ChallengeCategory>();
    const difficultiesUsed = new Set<ChallengeDifficulty>();

    // Always include 1-2 easy challenges
    const easyChallenges = shuffled.filter(c => c.difficulty === 'easy');
    if (easyChallenges.length > 0) {
      selected.push(easyChallenges[0]);
      categoriesUsed.add(easyChallenges[0].category);
      difficultiesUsed.add('easy');
    }

    // Add medium and hard challenges
    for (const challenge of shuffled) {
      if (selected.length >= 6) break;
      if (selected.find(s => s.id === challenge.id)) continue;

      // Prefer variety in categories
      if (categoriesUsed.size < 4 && categoriesUsed.has(challenge.category)) {
        if (random() < 0.6) continue;
      }

      selected.push(challenge);
      categoriesUsed.add(challenge.category);
      difficultiesUsed.add(challenge.difficulty);
    }

    // Determine highlights based on user metrics
    const highlights = this.determineHighlights(selected, userMetrics);

    // Convert to DailyChallenges
    this.dailyChallenges = selected.map(challenge => ({
      ...challenge,
      date: today,
      progress: 0,
      completed: false,
      isPinned: this.data.pinnedChallengeIds.includes(challenge.id),
      isHighlighted: highlights.has(challenge.id),
      highlightReason: highlights.get(challenge.id)
    }));

    // Preserve pinned challenges from previous days if not completed
    const pinnedFromBefore = this.data.pinnedChallengeIds
      .filter(id => !this.dailyChallenges.find(dc => dc.id === id))
      .map(id => CHALLENGE_POOL.find(c => c.id === id))
      .filter((c): c is Challenge => c !== undefined)
      .slice(0, 2) // Max 2 carried over pinned challenges
      .map(challenge => ({
        ...challenge,
        date: today,
        progress: 0,
        completed: false,
        isPinned: true,
        isHighlighted: false
      }));

    this.dailyChallenges = [...pinnedFromBefore, ...this.dailyChallenges];

    this.saveDailyChallenges();
    return this.dailyChallenges;
  }

  private determineHighlights(
    challenges: Challenge[],
    metrics?: UserMetrics
  ): Map<string, string> {
    const highlights = new Map<string, string>();

    if (!metrics) return highlights;

    // Highlight challenges that address weak areas
    for (const challenge of challenges) {
      // Low stability? Highlight productivity challenges
      if (metrics.deltaHV.stability < 50 && challenge.category === 'productivity') {
        highlights.set(challenge.id, 'Your stability could use a boost!');
        continue;
      }

      // Low resonance? Highlight social challenges
      if (metrics.deltaHV.resonance < 50 && challenge.category === 'social') {
        highlights.set(challenge.id, 'Recommended for improving connection!');
        continue;
      }

      // High friction? Highlight wellness/mindfulness
      if (metrics.deltaHV.friction > 60 &&
          (challenge.category === 'wellness' || challenge.category === 'mindfulness')) {
        highlights.set(challenge.id, 'This could help reduce your friction!');
        continue;
      }

      // Low symbolic? Highlight creativity
      if (metrics.deltaHV.symbolic < 50 && challenge.category === 'creativity') {
        highlights.set(challenge.id, 'Great for boosting creativity!');
        continue;
      }

      // Weak categories
      if (metrics.weakCategories.some(cat =>
          challenge.category.toLowerCase().includes(cat.toLowerCase()) ||
          challenge.requirements.some(r =>
            r.category?.toLowerCase().includes(cat.toLowerCase())
          ))) {
        highlights.set(challenge.id, `You haven\'t focused on ${challenge.category} recently!`);
        continue;
      }

      // Low check-ins? Highlight easy challenges
      if (metrics.checkInsToday < 2 && challenge.difficulty === 'easy') {
        highlights.set(challenge.id, 'Quick win to get started today!');
        continue;
      }

      // Strong streak? Highlight harder challenges
      if (metrics.currentStreak >= 5 && challenge.difficulty === 'hard') {
        highlights.set(challenge.id, 'You\'re on a roll - try this challenge!');
        continue;
      }
    }

    // Ensure at least one highlight if we have metrics
    if (highlights.size === 0 && challenges.length > 0) {
      const randomChallenge = challenges[Math.floor(Math.random() * challenges.length)];
      highlights.set(randomChallenge.id, 'Recommended for you today!');
    }

    return highlights;
  }

  // ---------------------------------------------------------------------------
  // CHALLENGE OPERATIONS
  // ---------------------------------------------------------------------------

  getDailyChallenges(): DailyChallenge[] {
    this.checkAndRefreshDaily();
    return this.dailyChallenges;
  }

  getPinnedChallenges(): DailyChallenge[] {
    return this.dailyChallenges.filter(c => c.isPinned);
  }

  getHighlightedChallenges(): DailyChallenge[] {
    return this.dailyChallenges.filter(c => c.isHighlighted);
  }

  pinChallenge(challengeId: string): void {
    if (!this.data.pinnedChallengeIds.includes(challengeId)) {
      // Max 5 pinned challenges
      if (this.data.pinnedChallengeIds.length >= 5) {
        this.data.pinnedChallengeIds.shift();
      }
      this.data.pinnedChallengeIds.push(challengeId);

      // Update daily challenges
      const challenge = this.dailyChallenges.find(c => c.id === challengeId);
      if (challenge) {
        challenge.isPinned = true;
      }

      this.saveData();
      this.saveDailyChallenges();
    }
  }

  unpinChallenge(challengeId: string): void {
    this.data.pinnedChallengeIds = this.data.pinnedChallengeIds.filter(id => id !== challengeId);

    const challenge = this.dailyChallenges.find(c => c.id === challengeId);
    if (challenge) {
      challenge.isPinned = false;
    }

    this.saveData();
    this.saveDailyChallenges();
  }

  updateChallengeProgress(challengeId: string, progress: number): void {
    const challenge = this.dailyChallenges.find(c => c.id === challengeId);
    if (challenge && !challenge.completed) {
      challenge.progress = Math.min(100, Math.max(0, progress));

      if (challenge.progress >= 100) {
        this.completeChallenge(challengeId);
      } else {
        this.saveDailyChallenges();
      }
    }
  }

  completeChallenge(challengeId: string): CosmeticReward | null {
    const challenge = this.dailyChallenges.find(c => c.id === challengeId);
    if (!challenge || challenge.completed) return null;

    challenge.completed = true;
    challenge.completedAt = new Date().toISOString();
    challenge.progress = 100;

    // Award XP
    this.data.totalXP += challenge.xp_reward;

    // Update streak
    const today = this.getTodayKey();
    if (this.data.lastActiveDate !== today) {
      const yesterday = new Date();
      yesterday.setDate(yesterday.getDate() - 1);
      const yesterdayKey = yesterday.toISOString().split('T')[0];

      if (this.data.lastActiveDate === yesterdayKey) {
        this.data.currentStreak++;
      } else {
        this.data.currentStreak = 1;
      }

      this.data.lastActiveDate = today;
      this.data.longestStreak = Math.max(this.data.longestStreak, this.data.currentStreak);
    }

    // Record completion
    this.data.completedChallenges.push({
      challengeId,
      completedAt: challenge.completedAt,
      xpEarned: challenge.xp_reward,
      cosmeticUnlocked: challenge.cosmetic_reward_id
    });

    // Unlock cosmetic reward
    let unlockedCosmetic: CosmeticReward | null = null;
    if (!this.data.unlockedCosmetics.includes(challenge.cosmetic_reward_id)) {
      this.data.unlockedCosmetics.push(challenge.cosmetic_reward_id);
      unlockedCosmetic = this.getCosmeticById(challenge.cosmetic_reward_id);

      // Send notification about unlocked cosmetic
      if (unlockedCosmetic) {
        this.sendCosmeticUnlockNotification(unlockedCosmetic, challenge);
      }
    }

    // Auto-unpin completed challenges
    if (challenge.isPinned) {
      this.unpinChallenge(challengeId);
    }

    this.saveData();
    this.saveDailyChallenges();

    return unlockedCosmetic;
  }

  // ---------------------------------------------------------------------------
  // COSMETIC OPERATIONS
  // ---------------------------------------------------------------------------

  getCosmeticById(cosmeticId: string): CosmeticReward | null {
    return COSMETIC_CATALOG.find((c) => c.id === cosmeticId) || null;
  }

  getUnlockedCosmetics(): CosmeticReward[] {
    return COSMETIC_CATALOG.filter((c) =>
      this.data.unlockedCosmetics.includes(c.id)
    );
  }

  getEquippedCosmetics(): Record<CosmeticType, string | null> {
    return { ...this.data.equippedCosmetics };
  }

  equipCosmetic(cosmeticId: string): boolean {
    if (!this.data.unlockedCosmetics.includes(cosmeticId)) {
      return false;
    }

    const cosmetic = this.getCosmeticById(cosmeticId);
    if (!cosmetic) return false;

    this.data.equippedCosmetics[cosmetic.type] = cosmeticId;
    this.saveData();
    return true;
  }

  unequipCosmetic(type: CosmeticType): void {
    // Set to default if available, otherwise null
    const defaultId = `${type.split('_')[0]}_default`;
    if (this.data.unlockedCosmetics.includes(defaultId)) {
      this.data.equippedCosmetics[type] = defaultId;
    } else {
      this.data.equippedCosmetics[type] = null;
    }
    this.saveData();
  }

  getCosmeticsByType(type: CosmeticType): CosmeticReward[] {
    return COSMETIC_CATALOG.filter((c) => c.type === type);
  }

  // ---------------------------------------------------------------------------
  // NOTIFICATIONS
  // ---------------------------------------------------------------------------

  private sendCosmeticUnlockNotification(cosmetic: CosmeticReward, challenge: DailyChallenge): void {
    try {
      notificationService.show({
        type: 'cosmetic_unlocked',
        title: '🎨 New Cosmetic Unlocked!',
        body: `You earned "${cosmetic.name}" by completing "${challenge.title}"!`
      });
    } catch (e) {
      console.error('Failed to send cosmetic notification:', e);
    }
  }

  sendDailyPerformanceNotification(_metrics: UserMetrics): void {
    const today = this.getTodayKey();
    const lastNotifDate = localStorage.getItem(STORAGE_KEYS.LAST_NOTIFICATION_DATE);

    if (lastNotifDate === today) return;

    const completedToday = this.dailyChallenges.filter(c => c.completed).length;
    const highlighted = this.getHighlightedChallenges();

    let title = '';
    let body = '';

    if (completedToday >= 3) {
      title = '🌟 Outstanding Progress!';
      body = `You've completed ${completedToday} challenges today! Your ${this.data.currentStreak}-day streak is going strong.`;
    } else if (completedToday > 0) {
      title = '💪 Good Progress Today!';
      body = `You've completed ${completedToday} challenge${completedToday > 1 ? 's' : ''}. ${highlighted.length > 0 ? `Try "${highlighted[0].title}" next!` : ''}`;
    } else if (highlighted.length > 0) {
      title = '🎯 Daily Challenges Await!';
      body = `We've highlighted "${highlighted[0].title}" based on your metrics. ${highlighted[0].highlightReason || ''}`;
    } else {
      title = '📋 Your Daily Challenges Are Ready!';
      body = `${this.dailyChallenges.length} challenges available today. Complete them to unlock new cosmetics!`;
    }

    try {
      notificationService.show({
        type: 'daily_challenges_ready',
        title,
        body
      });
      localStorage.setItem(STORAGE_KEYS.LAST_NOTIFICATION_DATE, today);
    } catch (e) {
      console.error('Failed to send daily notification:', e);
    }
  }

  sendStreakNotification(): void {
    if (this.data.currentStreak > 0 && this.data.currentStreak % 5 === 0) {
      try {
        notificationService.show({
          type: 'streak_milestone',
          title: `🔥 ${this.data.currentStreak}-Day Streak!`,
          body: `Incredible dedication! Keep completing challenges to unlock more cosmetics.`
        });
      } catch (e) {
        console.error('Failed to send streak notification:', e);
      }
    }
  }

  // ---------------------------------------------------------------------------
  // STATISTICS
  // ---------------------------------------------------------------------------

  getStats(): {
    totalXP: number;
    level: number;
    xpToNextLevel: number;
    currentStreak: number;
    longestStreak: number;
    totalCompleted: number;
    unlockedCount: number;
    totalCosmetics: number;
  } {
    const level = Math.floor(this.data.totalXP / 100) + 1;
    const xpInCurrentLevel = this.data.totalXP % 100;

    return {
      totalXP: this.data.totalXP,
      level,
      xpToNextLevel: 100 - xpInCurrentLevel,
      currentStreak: this.data.currentStreak,
      longestStreak: this.data.longestStreak,
      totalCompleted: this.data.completedChallenges.length,
      unlockedCount: this.data.unlockedCosmetics.length,
      totalCosmetics: COSMETIC_CATALOG.length
    };
  }

  getCompletionHistory(): ChallengeHistoryEntry[] {
    return this.data.challengeHistory;
  }

  // ---------------------------------------------------------------------------
  // DATA ACCESS
  // ---------------------------------------------------------------------------

  getData(): UserChallengeData {
    return { ...this.data };
  }

  resetProgress(): void {
    if (confirm('Are you sure you want to reset all challenge progress? This cannot be undone.')) {
      localStorage.removeItem(STORAGE_KEYS.CHALLENGE_DATA);
      localStorage.removeItem(STORAGE_KEYS.DAILY_CHALLENGES);
      this.data = this.loadData();
      this.dailyChallenges = [];
      this.generateDailyChallenges();
      this.notifyListeners();
    }
  }
}

// ============================================================================
// SINGLETON EXPORT
// ============================================================================

export const challengeRewardService = new ChallengeRewardService();

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

export function calculateChallengeProgress(
  challenge: Challenge,
  checkIns: Array<{ category: string; done: boolean }>,
  journalCount: number,
  metrics: UserMetrics
): number {
  let totalProgress = 0;
  let requirementCount = challenge.requirements.length;

  for (const req of challenge.requirements) {
    let progress = 0;

    switch (req.type) {
      case 'check_in': {
        const targetCount = typeof req.target === 'number' ? req.target : 1;
        const relevantCheckIns = req.category
          ? checkIns.filter(c => c.done && c.category.toLowerCase().includes(req.category!.toLowerCase()))
          : checkIns.filter(c => c.done);
        progress = Math.min(100, (relevantCheckIns.length / targetCount) * 100);
        break;
      }
      case 'journal': {
        const targetCount = typeof req.target === 'number' ? req.target : 1;
        progress = Math.min(100, (journalCount / targetCount) * 100);
        break;
      }
      case 'metric_threshold': {
        if (req.metric && req.threshold) {
          const metricValue = (metrics.deltaHV as Record<string, number>)[req.metric] || 0;
          const comparison = req.comparison || 'gte';

          if (comparison === 'gte' && metricValue >= req.threshold) {
            progress = 100;
          } else if (comparison === 'lte' && metricValue <= req.threshold) {
            progress = 100;
          } else if (comparison === 'gte') {
            progress = Math.min(100, (metricValue / req.threshold) * 100);
          } else {
            progress = metricValue <= req.threshold ? 100 : Math.max(0, ((req.threshold * 2 - metricValue) / req.threshold) * 100);
          }
        }
        break;
      }
      case 'streak': {
        const targetStreak = typeof req.target === 'number' ? req.target : 1;
        progress = Math.min(100, (metrics.currentStreak / targetStreak) * 100);
        break;
      }
      case 'category_complete': {
        const targetCount = typeof req.target === 'number' ? req.target : 1;
        const categoryCount = req.category
          ? (metrics.categoryCompletions[req.category] || 0)
          : Object.values(metrics.categoryCompletions).reduce((a, b) => a + b, 0);
        progress = Math.min(100, (categoryCount / targetCount) * 100);
        break;
      }
    }

    totalProgress += progress;
  }

  return requirementCount > 0 ? totalProgress / requirementCount : 0;
}

export function getChallengesByCategory(category: ChallengeCategory): Challenge[] {
  return CHALLENGE_POOL.filter(c => c.category === category);
}

export function getChallengesByDifficulty(difficulty: ChallengeDifficulty): Challenge[] {
  return CHALLENGE_POOL.filter(c => c.difficulty === difficulty);
}
