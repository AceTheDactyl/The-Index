/**
 * User Profile Service
 * Manages user preferences, metrics storage, habit tracking, and personalized roadmap data
 */

// ============================================================================
// Core Profile Types
// ============================================================================

export interface UserNeeds {
  physical: string[];      // Body-related needs (fitness, health, energy)
  emotional: string[];     // Emotional needs (peace, joy, connection)
  mental: string[];        // Mental needs (clarity, focus, creativity)
  spiritual: string[];     // Spiritual needs (purpose, meaning, growth)
  social: string[];        // Social needs (belonging, contribution)
}

export interface UserStrengths {
  cognitive: string[];     // Thinking patterns, problem-solving
  emotional: string[];     // Emotional intelligence, resilience
  physical: string[];      // Physical capabilities
  creative: string[];      // Creative abilities
  interpersonal: string[]; // Social skills
}

export interface UserMotivations {
  intrinsic: string[];     // Internal drivers (curiosity, mastery)
  extrinsic: string[];     // External drivers (recognition, reward)
  values: string[];        // Core values that drive action
  fears: string[];         // What drives avoidance (used for friction reduction)
}

export interface UserGoals {
  id: string;
  category: 'physical' | 'emotional' | 'mental' | 'spiritual' | 'social' | 'creative';
  title: string;
  description: string;
  targetDate?: string;     // ISO date
  milestones: GoalMilestone[];
  progress: number;        // 0-100
  linkedBeats: string[];   // Beat categories linked to this goal
  createdAt: string;
  updatedAt: string;
}

export interface GoalMilestone {
  id: string;
  title: string;
  completed: boolean;
  completedAt?: string;
  order: number;
}

export interface HabitTrack {
  id: string;
  name: string;
  category: string;
  frequency: 'daily' | 'weekly' | 'custom';
  customDays?: number[];   // 0-6 for custom frequency
  targetCount: number;     // Times per frequency period
  currentStreak: number;
  longestStreak: number;
  history: HabitEntry[];
  linkedGoalId?: string;
  createdAt: string;
}

export interface HabitEntry {
  date: string;            // ISO date
  completed: boolean;
  count: number;
  notes?: string;
}

export interface BeatRoadmapConfig {
  category: string;
  questions: RoadmapQuestion[];
  currentFocus: string;    // Current improvement focus
  targetOutcome: string;   // What user wants to achieve
}

export interface RoadmapQuestion {
  id: string;
  question: string;
  answer?: string;
  category: 'improvement' | 'obstacle' | 'emotion' | 'vision';
}

export interface UserPreferences {
  theme: 'dark' | 'light' | 'auto';
  timezone: string;
  language: string;
  dashboardLayout: 'compact' | 'expanded';
  metricsDisplay: 'minimal' | 'detailed' | 'full';
  reminderStyle: 'gentle' | 'assertive' | 'silent';
  focusMode: boolean;
  showGlyphs: boolean;
  roadmapUpdateFrequency: 'daily' | 'weekly' | 'biweekly' | 'monthly';
  lastRoadmapUpdate?: string;
  nextScheduledUpdate?: string;
}

export interface MetricsSnapshot {
  date: string;
  deltaHV: number;
  rhythmScore: number;
  frictionCoefficient: number;
  symbolicDensity: number;
  resonanceCoupling: number;
  harmonicStability: number;
  completedBeats: number;
  totalBeats: number;
  journalEntries: number;
  glyphsUsed: string[];
  fieldState: 'coherent' | 'transitioning' | 'fragmented' | 'dormant';
}

export interface DomainQuestionAnswers {
  [domainId: string]: {
    improvement?: string;
    obstacle?: string;
    emotion?: string;
    vision?: string;
    currentFocus?: string;
    targetOutcome?: string;
  };
}

export interface UserProfile {
  id: string;
  displayName: string;
  avatarGlyph?: string;    // Glyph used as avatar
  needs: UserNeeds;
  strengths: UserStrengths;
  motivations: UserMotivations;
  goals: UserGoals[];
  habits: HabitTrack[];
  beatRoadmaps: BeatRoadmapConfig[];
  domainAnswers: DomainQuestionAnswers; // Life domain question answers
  preferences: UserPreferences;
  metricsHistory: MetricsSnapshot[];
  createdAt: string;
  updatedAt: string;
  onboardingComplete: boolean;
}

// ============================================================================
// Default Beat Roadmap Questions
// ============================================================================

export const DEFAULT_BEAT_QUESTIONS: Record<string, RoadmapQuestion[]> = {
  Workout: [
    { id: 'w1', question: 'What would you like to improve about your body?', category: 'improvement' },
    { id: 'w2', question: 'What physical limitations are you working to overcome?', category: 'obstacle' },
    { id: 'w3', question: 'How do you want to feel after your workouts?', category: 'emotion' },
    { id: 'w4', question: 'What does your ideal physical self look like?', category: 'vision' },
  ],
  Meditation: [
    { id: 'm1', question: 'What emotions do you want to process or overcome?', category: 'emotion' },
    { id: 'm2', question: 'What mental patterns interrupt your peace?', category: 'obstacle' },
    { id: 'm3', question: 'What state of consciousness are you cultivating?', category: 'improvement' },
    { id: 'm4', question: 'What does inner stillness look like for you?', category: 'vision' },
  ],
  Emotion: [
    { id: 'e1', question: 'Which emotional patterns do you want to transform?', category: 'improvement' },
    { id: 'e2', question: 'What triggers your emotional friction?', category: 'obstacle' },
    { id: 'e3', question: 'How do you want to respond to challenging emotions?', category: 'emotion' },
    { id: 'e4', question: 'What does emotional mastery look like for you?', category: 'vision' },
  ],
  Moderation: [
    { id: 'mod1', question: 'What areas of excess do you want to balance?', category: 'improvement' },
    { id: 'mod2', question: 'What triggers your overindulgence patterns?', category: 'obstacle' },
    { id: 'mod3', question: 'How do you want to feel about your consumption habits?', category: 'emotion' },
    { id: 'mod4', question: 'What does balanced living look like for you?', category: 'vision' },
  ],
  Journal: [
    { id: 'j1', question: 'What aspects of your inner life do you want to explore?', category: 'improvement' },
    { id: 'j2', question: 'What thoughts do you avoid writing about?', category: 'obstacle' },
    { id: 'j3', question: 'What emotions arise when you reflect deeply?', category: 'emotion' },
    { id: 'j4', question: 'What truths are you seeking through writing?', category: 'vision' },
  ],
  General: [
    { id: 'g1', question: 'What daily friction do you want to reduce?', category: 'improvement' },
    { id: 'g2', question: 'What obstacles appear repeatedly in your day?', category: 'obstacle' },
    { id: 'g3', question: 'What emotional state do you want as your baseline?', category: 'emotion' },
    { id: 'g4', question: 'What does your ideal day feel like?', category: 'vision' },
  ],
  Anchor: [
    { id: 'a1', question: 'What rituals ground you most deeply?', category: 'improvement' },
    { id: 'a2', question: 'What disrupts your anchor practices?', category: 'obstacle' },
    { id: 'a3', question: 'How do you feel when anchors are completed?', category: 'emotion' },
    { id: 'a4', question: 'What does perfect rhythm feel like?', category: 'vision' },
  ],
};

// ============================================================================
// Storage Keys
// ============================================================================

const STORAGE_KEY = 'pulse-user-profile';

// ============================================================================
// Default Profile
// ============================================================================

const generateId = () => `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

export const createDefaultProfile = (): UserProfile => ({
  id: generateId(),
  displayName: 'User',
  needs: {
    physical: [],
    emotional: [],
    mental: [],
    spiritual: [],
    social: [],
  },
  strengths: {
    cognitive: [],
    emotional: [],
    physical: [],
    creative: [],
    interpersonal: [],
  },
  motivations: {
    intrinsic: [],
    extrinsic: [],
    values: [],
    fears: [],
  },
  goals: [],
  habits: [],
  beatRoadmaps: Object.keys(DEFAULT_BEAT_QUESTIONS).map(category => ({
    category,
    questions: DEFAULT_BEAT_QUESTIONS[category].map(q => ({ ...q })),
    currentFocus: '',
    targetOutcome: '',
  })),
  domainAnswers: {}, // Life domain question answers (body, mind, emotion, spirit, social, creative)
  preferences: {
    theme: 'dark',
    timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
    language: navigator.language || 'en-US',
    dashboardLayout: 'expanded',
    metricsDisplay: 'detailed',
    reminderStyle: 'gentle',
    focusMode: false,
    showGlyphs: true,
    roadmapUpdateFrequency: 'weekly',
  },
  metricsHistory: [],
  createdAt: new Date().toISOString(),
  updatedAt: new Date().toISOString(),
  onboardingComplete: false,
});

// ============================================================================
// Profile Service Class
// ============================================================================

class UserProfileService {
  private profile: UserProfile | null = null;
  private listeners: Set<(profile: UserProfile) => void> = new Set();

  async initialize(): Promise<UserProfile> {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        this.profile = JSON.parse(stored);
        // Ensure all beat roadmaps exist
        this.ensureBeatRoadmaps();
      } else {
        this.profile = createDefaultProfile();
        await this.save();
      }
      return this.profile!;
    } catch (error) {
      console.error('Failed to initialize profile:', error);
      this.profile = createDefaultProfile();
      return this.profile;
    }
  }

  private ensureBeatRoadmaps(): void {
    if (!this.profile) return;

    const existingCategories = new Set(this.profile.beatRoadmaps.map(br => br.category));
    for (const category of Object.keys(DEFAULT_BEAT_QUESTIONS)) {
      if (!existingCategories.has(category)) {
        this.profile.beatRoadmaps.push({
          category,
          questions: DEFAULT_BEAT_QUESTIONS[category].map(q => ({ ...q })),
          currentFocus: '',
          targetOutcome: '',
        });
      }
    }
  }

  getProfile(): UserProfile | null {
    return this.profile;
  }

  async updateProfile(updates: Partial<UserProfile>): Promise<UserProfile> {
    if (!this.profile) {
      await this.initialize();
    }

    this.profile = {
      ...this.profile!,
      ...updates,
      updatedAt: new Date().toISOString(),
    };

    await this.save();
    this.notifyListeners();
    return this.profile;
  }

  async updateNeeds(needs: Partial<UserNeeds>): Promise<void> {
    await this.updateProfile({
      needs: { ...this.profile?.needs, ...needs } as UserNeeds,
    });
  }

  async updateStrengths(strengths: Partial<UserStrengths>): Promise<void> {
    await this.updateProfile({
      strengths: { ...this.profile?.strengths, ...strengths } as UserStrengths,
    });
  }

  async updateMotivations(motivations: Partial<UserMotivations>): Promise<void> {
    await this.updateProfile({
      motivations: { ...this.profile?.motivations, ...motivations } as UserMotivations,
    });
  }

  async updatePreferences(prefs: Partial<UserPreferences>): Promise<void> {
    await this.updateProfile({
      preferences: { ...this.profile?.preferences, ...prefs } as UserPreferences,
    });
  }

  // Goal Management
  async addGoal(goal: Omit<UserGoals, 'id' | 'createdAt' | 'updatedAt'>): Promise<UserGoals> {
    const newGoal: UserGoals = {
      ...goal,
      id: generateId(),
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    };

    await this.updateProfile({
      goals: [...(this.profile?.goals || []), newGoal],
    });

    return newGoal;
  }

  async updateGoal(goalId: string, updates: Partial<UserGoals>): Promise<void> {
    const goals = this.profile?.goals.map(g =>
      g.id === goalId ? { ...g, ...updates, updatedAt: new Date().toISOString() } : g
    ) || [];

    await this.updateProfile({ goals });
  }

  async removeGoal(goalId: string): Promise<void> {
    const goals = this.profile?.goals.filter(g => g.id !== goalId) || [];
    await this.updateProfile({ goals });
  }

  // Habit Management
  async addHabit(habit: Omit<HabitTrack, 'id' | 'createdAt' | 'history'>): Promise<HabitTrack> {
    const newHabit: HabitTrack = {
      ...habit,
      id: generateId(),
      history: [],
      createdAt: new Date().toISOString(),
    };

    await this.updateProfile({
      habits: [...(this.profile?.habits || []), newHabit],
    });

    return newHabit;
  }

  async logHabitEntry(habitId: string, entry: Omit<HabitEntry, 'date'>): Promise<void> {
    const today = new Date().toISOString().split('T')[0];

    const habits = this.profile?.habits.map(h => {
      if (h.id !== habitId) return h;

      // Update or add today's entry
      const existingIdx = h.history.findIndex(e => e.date === today);
      const newHistory = [...h.history];

      if (existingIdx >= 0) {
        newHistory[existingIdx] = { ...newHistory[existingIdx], ...entry, date: today };
      } else {
        newHistory.push({ ...entry, date: today });
      }

      // Calculate streak
      const streak = this.calculateStreak(newHistory, h.frequency, h.customDays);

      return {
        ...h,
        history: newHistory,
        currentStreak: streak,
        longestStreak: Math.max(h.longestStreak, streak),
      };
    }) || [];

    await this.updateProfile({ habits });
  }

  private calculateStreak(
    history: HabitEntry[],
    frequency: 'daily' | 'weekly' | 'custom',
    customDays?: number[]
  ): number {
    const sorted = history
      .filter(e => e.completed)
      .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());

    if (sorted.length === 0) return 0;

    let streak = 0;
    const today = new Date();
    today.setHours(0, 0, 0, 0);

    for (let i = 0; i < sorted.length; i++) {
      const entryDate = new Date(sorted[i].date);
      entryDate.setHours(0, 0, 0, 0);

      const expectedDate = new Date(today);
      expectedDate.setDate(today.getDate() - i);

      if (frequency === 'daily') {
        if (entryDate.getTime() === expectedDate.getTime()) {
          streak++;
        } else {
          break;
        }
      } else if (frequency === 'weekly') {
        const weeksDiff = Math.floor((today.getTime() - entryDate.getTime()) / (7 * 24 * 60 * 60 * 1000));
        if (weeksDiff === i) {
          streak++;
        } else {
          break;
        }
      } else if (frequency === 'custom' && customDays) {
        const dayOfWeek = expectedDate.getDay();
        if (customDays.includes(dayOfWeek) && entryDate.getTime() === expectedDate.getTime()) {
          streak++;
        }
      }
    }

    return streak;
  }

  // Beat Roadmap Management
  async updateBeatRoadmap(category: string, updates: Partial<BeatRoadmapConfig>): Promise<void> {
    const beatRoadmaps = this.profile?.beatRoadmaps.map(br =>
      br.category === category ? { ...br, ...updates } : br
    ) || [];

    await this.updateProfile({ beatRoadmaps });
  }

  async answerRoadmapQuestion(category: string, questionId: string, answer: string): Promise<void> {
    const beatRoadmaps = this.profile?.beatRoadmaps.map(br => {
      if (br.category !== category) return br;

      const questions = br.questions.map(q =>
        q.id === questionId ? { ...q, answer } : q
      );

      return { ...br, questions };
    }) || [];

    await this.updateProfile({ beatRoadmaps });
  }

  getBeatRoadmap(category: string): BeatRoadmapConfig | undefined {
    return this.profile?.beatRoadmaps.find(br => br.category === category);
  }

  // Domain Question Answers
  async saveDomainAnswer(
    domainId: string,
    field: 'improvement' | 'obstacle' | 'emotion' | 'vision' | 'currentFocus' | 'targetOutcome',
    value: string
  ): Promise<void> {
    const domainAnswers = { ...(this.profile?.domainAnswers || {}) };
    if (!domainAnswers[domainId]) {
      domainAnswers[domainId] = {};
    }
    domainAnswers[domainId][field] = value;
    await this.updateProfile({ domainAnswers });
  }

  async saveDomainAnswers(domainId: string, answers: {
    improvement?: string;
    obstacle?: string;
    emotion?: string;
    vision?: string;
    currentFocus?: string;
    targetOutcome?: string;
  }): Promise<void> {
    const domainAnswers = { ...(this.profile?.domainAnswers || {}) };
    domainAnswers[domainId] = { ...(domainAnswers[domainId] || {}), ...answers };
    await this.updateProfile({ domainAnswers });
  }

  getDomainAnswers(domainId: string): DomainQuestionAnswers[string] | undefined {
    return this.profile?.domainAnswers?.[domainId];
  }

  getAllDomainAnswers(): DomainQuestionAnswers {
    return this.profile?.domainAnswers || {};
  }

  // Metrics History
  async recordMetricsSnapshot(snapshot: Omit<MetricsSnapshot, 'date'>): Promise<void> {
    const today = new Date().toISOString().split('T')[0];
    const metricsHistory = [...(this.profile?.metricsHistory || [])];

    // Update or add today's snapshot
    const existingIdx = metricsHistory.findIndex(m => m.date === today);
    if (existingIdx >= 0) {
      metricsHistory[existingIdx] = { ...snapshot, date: today };
    } else {
      metricsHistory.push({ ...snapshot, date: today });
    }

    // Keep last 365 days
    const cutoff = new Date();
    cutoff.setDate(cutoff.getDate() - 365);
    const filtered = metricsHistory.filter(m => new Date(m.date) >= cutoff);

    await this.updateProfile({ metricsHistory: filtered });
  }

  getMetricsHistory(days: number = 30): MetricsSnapshot[] {
    const cutoff = new Date();
    cutoff.setDate(cutoff.getDate() - days);

    return (this.profile?.metricsHistory || [])
      .filter(m => new Date(m.date) >= cutoff)
      .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
  }

  // Roadmap Update Scheduling
  shouldUpdateRoadmap(): boolean {
    if (!this.profile?.preferences.lastRoadmapUpdate) return true;

    const lastUpdate = new Date(this.profile.preferences.lastRoadmapUpdate);
    const now = new Date();
    const daysDiff = Math.floor((now.getTime() - lastUpdate.getTime()) / (24 * 60 * 60 * 1000));

    switch (this.profile.preferences.roadmapUpdateFrequency) {
      case 'daily': return daysDiff >= 1;
      case 'weekly': return daysDiff >= 7;
      case 'biweekly': return daysDiff >= 14;
      case 'monthly': return daysDiff >= 30;
      default: return daysDiff >= 7;
    }
  }

  async markRoadmapUpdated(): Promise<void> {
    const now = new Date();
    const nextUpdate = new Date(now);

    switch (this.profile?.preferences.roadmapUpdateFrequency) {
      case 'daily': nextUpdate.setDate(nextUpdate.getDate() + 1); break;
      case 'weekly': nextUpdate.setDate(nextUpdate.getDate() + 7); break;
      case 'biweekly': nextUpdate.setDate(nextUpdate.getDate() + 14); break;
      case 'monthly': nextUpdate.setMonth(nextUpdate.getMonth() + 1); break;
    }

    await this.updatePreferences({
      lastRoadmapUpdate: now.toISOString(),
      nextScheduledUpdate: nextUpdate.toISOString(),
    });
  }

  // Subscription
  subscribe(listener: (profile: UserProfile) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private notifyListeners(): void {
    if (this.profile) {
      this.listeners.forEach(listener => listener(this.profile!));
    }
  }

  private async save(): Promise<void> {
    if (this.profile) {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(this.profile));
    }
  }

  // Export/Import
  exportProfile(): string {
    return JSON.stringify(this.profile, null, 2);
  }

  async importProfile(jsonString: string): Promise<void> {
    try {
      const imported = JSON.parse(jsonString) as UserProfile;
      imported.updatedAt = new Date().toISOString();
      this.profile = imported;
      await this.save();
      this.notifyListeners();
    } catch (error) {
      throw new Error('Invalid profile data');
    }
  }
}

// Singleton instance
export const userProfileService = new UserProfileService();

export default userProfileService;
