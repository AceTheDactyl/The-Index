/**
 * Roadmap Engine
 *
 * AI-assisted recommendation system based on the Free Energy Principle.
 * The FEP suggests organisms minimize "free energy" (surprise/uncertainty) by:
 * 1. Updating internal models to better predict the world (perception)
 * 2. Acting on the world to make it match predictions (action)
 *
 * Applied to personal development:
 * - Friction = surprise/unexpected outcomes that consume mental energy
 * - Coherence = states where reality matches internal models (low free energy)
 * - Roadmap = action sequences that systematically reduce friction
 */

import type {
  UserProfile,
  UserGoals,
  BeatRoadmapConfig,
  MetricsSnapshot,
} from './userProfile';
import type { EnhancedDeltaHVState } from './metricsHub';

/**
 * Real-time DeltaHV state for recommendations
 */
interface LiveMetricState {
  symbolic: number;
  resonance: number;
  friction: number;
  stability: number;
  deltaHV: number;
  fieldState: 'coherent' | 'transitioning' | 'fragmented' | 'dormant';
  musicInfluence?: {
    skipRatio: number;
    authorshipScore: number;
    healingProgress: number;
  };
}

// ============================================================================
// Types
// ============================================================================

export interface FrictionSource {
  id: string;
  category: 'physical' | 'emotional' | 'mental' | 'behavioral' | 'environmental' | 'social';
  description: string;
  intensity: number;         // 0-100, how much friction this causes
  frequency: 'rare' | 'occasional' | 'regular' | 'constant';
  triggers: string[];
  relatedBeats: string[];    // Beat categories that can address this
  identifiedAt: string;
}

export interface RoadmapStep {
  id: string;
  order: number;
  title: string;
  description: string;
  category: string;          // Beat category
  actionType: 'habit' | 'practice' | 'reflection' | 'challenge' | 'milestone';
  estimatedDuration: string; // e.g., "2 weeks", "1 month"
  frictionAddressed: string[];
  prerequisiteSteps: string[];
  metrics: {
    targetMetric: string;
    currentValue?: number;
    targetValue: number;
  }[];
  glyphResonance?: string;   // Glyph from Ace Codex that resonates
  status: 'pending' | 'active' | 'completed' | 'skipped';
  startedAt?: string;
  completedAt?: string;
}

export interface Roadmap {
  id: string;
  title: string;
  description: string;
  category: 'physical' | 'emotional' | 'mental' | 'spiritual' | 'social' | 'creative';
  goals: string[];           // Goal IDs linked to this roadmap
  steps: RoadmapStep[];
  frictionSources: FrictionSource[];
  predictedOutcome: string;
  coherenceTarget: number;   // Target DeltaHV score
  createdAt: string;
  updatedAt: string;
  lastEvaluatedAt: string;
}

export interface AIRecommendation {
  id: string;
  type: 'friction_reduction' | 'habit_formation' | 'goal_alignment' | 'pattern_insight' | 'energy_optimization';
  priority: 'low' | 'medium' | 'high' | 'critical';
  title: string;
  rationale: string;         // Why this is recommended (FEP logic)
  suggestedAction: string;
  expectedOutcome: string;
  relatedBeat?: string;
  relatedGoal?: string;
  confidence: number;        // 0-100, how confident the system is
  expiresAt?: string;        // Recommendations can expire
  createdAt: string;
  dismissed?: boolean;
  dismissedReason?: string;
}

export interface PatternInsight {
  id: string;
  type: 'success' | 'struggle' | 'correlation' | 'prediction';
  description: string;
  evidence: string[];        // What data supports this insight
  actionable: boolean;
  suggestedAction?: string;
  confidence: number;
  createdAt: string;
}

export interface EnergyState {
  physical: number;          // 0-100
  mental: number;
  emotional: number;
  social: number;
  creative: number;
  overall: number;
  trend: 'rising' | 'stable' | 'declining';
  predictedTomorrow: number;
}

// ============================================================================
// Free Energy Principle Calculations
// ============================================================================

/**
 * Calculate friction coefficient based on user data
 * Higher friction = more "free energy" to minimize
 */
export function calculateFriction(
  profile: UserProfile,
  recentMetrics: MetricsSnapshot[]
): { total: number; breakdown: Record<string, number> } {
  const breakdown: Record<string, number> = {
    goalMisalignment: 0,
    habitInconsistency: 0,
    emotionalTurbulence: 0,
    rhythmDisruption: 0,
    unmetNeeds: 0,
  };

  // Goal misalignment: how far are we from stated goals?
  const activeGoals = profile.goals.filter(g => g.progress < 100);
  if (activeGoals.length > 0) {
    const avgProgress = activeGoals.reduce((sum, g) => sum + g.progress, 0) / activeGoals.length;
    breakdown.goalMisalignment = Math.max(0, 50 - avgProgress / 2);
  }

  // Habit inconsistency: broken streaks create friction
  if (profile.habits.length > 0) {
    const avgStreak = profile.habits.reduce((sum, h) => sum + h.currentStreak, 0) / profile.habits.length;
    const targetStreak = 7; // One week is baseline
    breakdown.habitInconsistency = Math.max(0, (1 - avgStreak / targetStreak) * 30);
  }

  // Emotional turbulence: variance in deltaHV
  if (recentMetrics.length > 1) {
    const deltaHVValues = recentMetrics.map(m => m.deltaHV);
    const variance = calculateVariance(deltaHVValues);
    breakdown.emotionalTurbulence = Math.min(20, variance / 5);
  }

  // Rhythm disruption: low rhythm scores
  if (recentMetrics.length > 0) {
    const avgRhythm = recentMetrics.reduce((sum, m) => sum + m.rhythmScore, 0) / recentMetrics.length;
    breakdown.rhythmDisruption = Math.max(0, (100 - avgRhythm) / 4);
  }

  // Unmet needs: needs without associated goals or habits
  const needsCount = Object.values(profile.needs).flat().length;
  const addressedByGoals = profile.goals.length;
  const addressedByHabits = profile.habits.length;
  if (needsCount > 0) {
    const coverageRatio = Math.min(1, (addressedByGoals + addressedByHabits) / needsCount);
    breakdown.unmetNeeds = (1 - coverageRatio) * 20;
  }

  const total = Object.values(breakdown).reduce((sum, v) => sum + v, 0);

  return { total: Math.min(100, total), breakdown };
}

/**
 * Calculate prediction error - the difference between expected and actual outcomes
 * Lower is better (system predictions align with reality)
 */
export function calculatePredictionError(
  expectedMetrics: Partial<MetricsSnapshot>,
  actualMetrics: MetricsSnapshot
): number {
  let errorSum = 0;
  let count = 0;

  if (expectedMetrics.deltaHV !== undefined) {
    errorSum += Math.abs(expectedMetrics.deltaHV - actualMetrics.deltaHV);
    count++;
  }
  if (expectedMetrics.rhythmScore !== undefined) {
    errorSum += Math.abs(expectedMetrics.rhythmScore - actualMetrics.rhythmScore);
    count++;
  }
  if (expectedMetrics.frictionCoefficient !== undefined) {
    errorSum += Math.abs(expectedMetrics.frictionCoefficient - actualMetrics.frictionCoefficient);
    count++;
  }

  return count > 0 ? errorSum / count : 0;
}

/**
 * Active Inference: Generate action suggestions to reduce free energy
 * These are actions the user can take to align reality with their goals
 */
export function generateActiveInferences(
  profile: UserProfile,
  friction: { total: number; breakdown: Record<string, number> },
  recentMetrics: MetricsSnapshot[]
): AIRecommendation[] {
  const recommendations: AIRecommendation[] = [];
  const now = new Date().toISOString();

  // High goal misalignment → suggest goal-aligned beats
  if (friction.breakdown.goalMisalignment > 20) {
    const stalledGoals = profile.goals.filter(g => g.progress < 30);
    stalledGoals.forEach(goal => {
      recommendations.push({
        id: `rec-goal-${goal.id}`,
        type: 'goal_alignment',
        priority: friction.breakdown.goalMisalignment > 35 ? 'high' : 'medium',
        title: `Reactivate: ${goal.title}`,
        rationale: `Your progress on "${goal.title}" has stalled at ${goal.progress}%. The Free Energy Principle suggests that unmet intentions create cognitive friction. Small consistent action reduces this friction.`,
        suggestedAction: `Schedule one ${goal.linkedBeats[0] || 'General'} beat today focused on this goal's next milestone.`,
        expectedOutcome: 'Reduced goal-related friction, renewed momentum',
        relatedGoal: goal.id,
        relatedBeat: goal.linkedBeats[0],
        confidence: 75,
        createdAt: now,
      });
    });
  }

  // High habit inconsistency → suggest habit restoration
  if (friction.breakdown.habitInconsistency > 15) {
    const brokenHabits = profile.habits.filter(h => h.currentStreak === 0);
    brokenHabits.slice(0, 3).forEach(habit => {
      recommendations.push({
        id: `rec-habit-${habit.id}`,
        type: 'habit_formation',
        priority: 'medium',
        title: `Restore: ${habit.name}`,
        rationale: `Your "${habit.name}" streak broke. The FEP suggests that habit patterns reduce prediction error—when habits break, the system must expend energy predicting uncertain outcomes.`,
        suggestedAction: `Restart with a minimal version. Do the smallest possible version of "${habit.name}" today.`,
        expectedOutcome: 'New streak begins, reduced cognitive load from uncertainty',
        confidence: 80,
        createdAt: now,
      });
    });
  }

  // High emotional turbulence → suggest stabilization
  if (friction.breakdown.emotionalTurbulence > 10) {
    recommendations.push({
      id: `rec-emotional-${Date.now()}`,
      type: 'energy_optimization',
      priority: friction.breakdown.emotionalTurbulence > 15 ? 'high' : 'medium',
      title: 'Stabilize Emotional Field',
      rationale: 'Your metrics show high variance in coherence scores. The FEP suggests that emotional variability increases prediction error and drains cognitive resources.',
      suggestedAction: 'Add an Emotion check-in beat focused on "Vibe Check" or "Safety Inventory". Consider a Meditation beat using somatic or breath focus.',
      expectedOutcome: 'Smoother emotional baseline, reduced turbulence',
      relatedBeat: 'Meditation',
      confidence: 70,
      createdAt: now,
    });
  }

  // Low rhythm score → suggest anchor reinforcement
  if (friction.breakdown.rhythmDisruption > 15) {
    recommendations.push({
      id: `rec-rhythm-${Date.now()}`,
      type: 'friction_reduction',
      priority: 'high',
      title: 'Reinforce Daily Rhythm',
      rationale: 'Your rhythm score is low, indicating missed anchors. The FEP suggests that predictable routines minimize free energy by reducing uncertainty about daily structure.',
      suggestedAction: 'Focus on completing just your anchor beats today. Anchors provide the predictability your system needs.',
      expectedOutcome: 'Higher rhythm score, reduced decision fatigue',
      relatedBeat: 'Anchor',
      confidence: 85,
      createdAt: now,
    });
  }

  // Pattern insights from metrics
  if (recentMetrics.length >= 7) {
    const insight = detectPatterns(recentMetrics);
    if (insight) {
      recommendations.push({
        id: `rec-pattern-${Date.now()}`,
        type: 'pattern_insight',
        priority: insight.confidence > 70 ? 'medium' : 'low',
        title: insight.description,
        rationale: `Pattern detected: ${insight.evidence.join('; ')}. Understanding patterns allows better prediction and proactive optimization.`,
        suggestedAction: insight.suggestedAction || 'Review your recent activity and adjust based on this pattern.',
        expectedOutcome: 'Better self-understanding, optimized scheduling',
        confidence: insight.confidence,
        createdAt: now,
      });
    }
  }

  return recommendations;
}

/**
 * Detect patterns in metrics history
 */
function detectPatterns(metrics: MetricsSnapshot[]): PatternInsight | null {
  if (metrics.length < 7) return null;

  const recent = metrics.slice(-7);

  // Detect improvement trend
  const firstHalf = recent.slice(0, 3);
  const secondHalf = recent.slice(-3);
  const firstAvg = firstHalf.reduce((sum, m) => sum + m.deltaHV, 0) / firstHalf.length;
  const secondAvg = secondHalf.reduce((sum, m) => sum + m.deltaHV, 0) / secondHalf.length;

  if (secondAvg - firstAvg > 10) {
    return {
      id: `insight-${Date.now()}`,
      type: 'success',
      description: 'Your coherence is improving',
      evidence: [
        `DeltaHV increased from ${firstAvg.toFixed(0)} to ${secondAvg.toFixed(0)}`,
        'Consistent upward trend over the past week',
      ],
      actionable: true,
      suggestedAction: 'Keep doing what you\'re doing. Consider noting what\'s working in a journal beat.',
      confidence: 75,
      createdAt: new Date().toISOString(),
    };
  }

  if (firstAvg - secondAvg > 10) {
    return {
      id: `insight-${Date.now()}`,
      type: 'struggle',
      description: 'Your coherence is declining',
      evidence: [
        `DeltaHV decreased from ${firstAvg.toFixed(0)} to ${secondAvg.toFixed(0)}`,
        'Downward trend needs attention',
      ],
      actionable: true,
      suggestedAction: 'Review recent changes in routine. Consider simplifying your beat schedule and focusing on anchors.',
      confidence: 75,
      createdAt: new Date().toISOString(),
    };
  }

  // Detect weekend patterns
  const weekendMetrics = recent.filter((_, i) => {
    const date = new Date(recent[i].date);
    return date.getDay() === 0 || date.getDay() === 6;
  });

  if (weekendMetrics.length > 0) {
    const weekendAvg = weekendMetrics.reduce((sum, m) => sum + m.rhythmScore, 0) / weekendMetrics.length;
    const weekdayMetrics = recent.filter((_, i) => {
      const date = new Date(recent[i].date);
      return date.getDay() !== 0 && date.getDay() !== 6;
    });
    const weekdayAvg = weekdayMetrics.reduce((sum, m) => sum + m.rhythmScore, 0) / Math.max(1, weekdayMetrics.length);

    if (weekdayAvg - weekendAvg > 20) {
      return {
        id: `insight-weekend-${Date.now()}`,
        type: 'correlation',
        description: 'Weekend rhythm drops significantly',
        evidence: [
          `Weekday rhythm: ${weekdayAvg.toFixed(0)}%`,
          `Weekend rhythm: ${weekendAvg.toFixed(0)}%`,
        ],
        actionable: true,
        suggestedAction: 'Create a simplified weekend anchor schedule. Consider designating one day as a "deviation day".',
        confidence: 80,
        createdAt: new Date().toISOString(),
      };
    }
  }

  return null;
}

// ============================================================================
// Roadmap Generation
// ============================================================================

/**
 * Generate a personalized roadmap based on user profile and friction analysis
 */
export function generateRoadmap(
  profile: UserProfile,
  category: 'physical' | 'emotional' | 'mental' | 'spiritual' | 'social' | 'creative',
  friction: { total: number; breakdown: Record<string, number> }
): Roadmap {
  const now = new Date().toISOString();
  const relevantGoals = profile.goals.filter(g => g.category === category);
  const relevantNeeds = profile.needs[category as keyof typeof profile.needs] || [];

  // Identify friction sources for this category
  const frictionSources: FrictionSource[] = [];

  // Map needs to friction sources
  relevantNeeds.forEach((need, idx) => {
    frictionSources.push({
      id: `friction-${category}-${idx}`,
      category: category === 'spiritual' ? 'mental' : category as any,
      description: `Unmet need: ${need}`,
      intensity: 50,
      frequency: 'regular',
      triggers: [],
      relatedBeats: getCategoryBeats(category),
      identifiedAt: now,
    });
  });

  // Map low-progress goals to friction
  relevantGoals.filter(g => g.progress < 50).forEach(goal => {
    frictionSources.push({
      id: `friction-goal-${goal.id}`,
      category: category === 'spiritual' ? 'mental' : category as any,
      description: `Goal in progress: ${goal.title}`,
      intensity: Math.round((100 - goal.progress) / 2),
      frequency: 'constant',
      triggers: [],
      relatedBeats: goal.linkedBeats,
      identifiedAt: now,
    });
  });

  // Generate steps based on friction sources and goals
  const steps: RoadmapStep[] = generateStepsFromFriction(
    frictionSources,
    relevantGoals,
    profile.beatRoadmaps.find(br => getCategoryBeats(category).includes(br.category)),
    category
  );

  return {
    id: `roadmap-${category}-${Date.now()}`,
    title: `${capitalize(category)} Development Roadmap`,
    description: `A personalized path to reduce friction and achieve coherence in your ${category} domain.`,
    category,
    goals: relevantGoals.map(g => g.id),
    steps,
    frictionSources,
    predictedOutcome: `Reduced ${category} friction, improved coherence scores, and progress toward stated goals.`,
    coherenceTarget: Math.min(95, 70 + (100 - friction.total) / 3),
    createdAt: now,
    updatedAt: now,
    lastEvaluatedAt: now,
  };
}

function getCategoryBeats(category: string): string[] {
  const mapping: Record<string, string[]> = {
    physical: ['Workout', 'Anchor'],
    emotional: ['Emotion', 'Meditation', 'Journal'],
    mental: ['Meditation', 'Journal', 'General'],
    spiritual: ['Meditation', 'Journal'],
    social: ['Moderation', 'Emotion', 'General'],
    creative: ['Journal', 'General'],
  };
  return mapping[category] || ['General'];
}

function generateStepsFromFriction(
  frictionSources: FrictionSource[],
  goals: UserGoals[],
  beatRoadmap: BeatRoadmapConfig | undefined,
  category: string
): RoadmapStep[] {
  const steps: RoadmapStep[] = [];

  // Foundation step: establish baseline
  steps.push({
    id: `step-${category}-foundation`,
    order: 1,
    title: 'Establish Baseline',
    description: 'Track your current state for one week without making major changes. This provides data for the FEP prediction model.',
    category: 'General',
    actionType: 'reflection',
    estimatedDuration: '1 week',
    frictionAddressed: [],
    prerequisiteSteps: [],
    metrics: [
      { targetMetric: 'journalEntries', targetValue: 7 },
      { targetMetric: 'rhythmScore', targetValue: 50 },
    ],
    status: 'pending',
  });

  // Address highest friction sources first
  const sortedFriction = [...frictionSources].sort((a, b) => b.intensity - a.intensity);

  sortedFriction.slice(0, 3).forEach((source, idx) => {
    steps.push({
      id: `step-${category}-friction-${idx}`,
      order: idx + 2,
      title: `Reduce: ${source.description}`,
      description: `Focus on minimizing friction from "${source.description}". Use ${source.relatedBeats.join(' or ')} beats.`,
      category: source.relatedBeats[0] || 'General',
      actionType: 'practice',
      estimatedDuration: '2 weeks',
      frictionAddressed: [source.id],
      prerequisiteSteps: idx === 0 ? [`step-${category}-foundation`] : [`step-${category}-friction-${idx - 1}`],
      metrics: [
        { targetMetric: 'frictionCoefficient', targetValue: 30 },
      ],
      status: 'pending',
    });
  });

  // Goal milestone steps
  goals.forEach((goal) => {
    goal.milestones.filter(m => !m.completed).slice(0, 2).forEach((milestone, mIdx) => {
      steps.push({
        id: `step-${category}-goal-${goal.id}-${milestone.id}`,
        order: steps.length + 1,
        title: `Milestone: ${milestone.title}`,
        description: `Progress toward "${goal.title}" by completing this milestone.`,
        category: goal.linkedBeats[0] || 'General',
        actionType: 'milestone',
        estimatedDuration: '2 weeks',
        frictionAddressed: [`friction-goal-${goal.id}`],
        prerequisiteSteps: mIdx === 0 ? [] : [`step-${category}-goal-${goal.id}-${goal.milestones[mIdx - 1]?.id}`],
        metrics: [],
        status: 'pending',
      });
    });
  });

  // Add beat-specific improvement step based on roadmap answers
  if (beatRoadmap && beatRoadmap.currentFocus) {
    steps.push({
      id: `step-${category}-focus`,
      order: steps.length + 1,
      title: `Focus: ${beatRoadmap.currentFocus}`,
      description: beatRoadmap.targetOutcome || `Work toward your stated focus area.`,
      category: beatRoadmap.category,
      actionType: 'habit',
      estimatedDuration: '4 weeks',
      frictionAddressed: [],
      prerequisiteSteps: [],
      metrics: [
        { targetMetric: 'deltaHV', targetValue: 75 },
      ],
      status: 'pending',
    });
  }

  // Integration step: maintain coherence
  steps.push({
    id: `step-${category}-integration`,
    order: steps.length + 1,
    title: 'Integration & Maintenance',
    description: 'Consolidate gains. Focus on maintaining rhythm and preventing friction regression.',
    category: 'Anchor',
    actionType: 'habit',
    estimatedDuration: 'ongoing',
    frictionAddressed: frictionSources.map(f => f.id),
    prerequisiteSteps: steps.slice(0, -1).map(s => s.id),
    metrics: [
      { targetMetric: 'harmonicStability', targetValue: 80 },
      { targetMetric: 'resonanceCoupling', targetValue: 85 },
    ],
    status: 'pending',
  });

  return steps;
}

// ============================================================================
// Energy State Prediction
// ============================================================================

/**
 * Predict energy states based on patterns
 */
export function predictEnergyState(
  profile: UserProfile,
  recentMetrics: MetricsSnapshot[]
): EnergyState {
  const defaultState: EnergyState = {
    physical: 50,
    mental: 50,
    emotional: 50,
    social: 50,
    creative: 50,
    overall: 50,
    trend: 'stable',
    predictedTomorrow: 50,
  };

  if (recentMetrics.length < 3) return defaultState;

  const recent = recentMetrics.slice(-7);

  // Infer energy from metrics
  const avgDeltaHV = recent.reduce((sum, m) => sum + m.deltaHV, 0) / recent.length;
  const avgRhythm = recent.reduce((sum, m) => sum + m.rhythmScore, 0) / recent.length;
  const avgFriction = recent.reduce((sum, m) => sum + m.frictionCoefficient, 0) / recent.length;

  // Map to energy dimensions
  const overall = (avgDeltaHV + avgRhythm + (100 - avgFriction)) / 3;
  const physical = Math.min(100, overall + (profile.habits.filter(h => h.category === 'physical').length * 5));
  const mental = Math.min(100, overall + recent[recent.length - 1]?.journalEntries * 3);
  const emotional = Math.min(100, overall + (100 - avgFriction) / 2);
  const social = 50; // Would need social beat data
  const creative = Math.min(100, mental * 0.7 + emotional * 0.3);

  // Determine trend
  const firstHalf = recent.slice(0, Math.floor(recent.length / 2));
  const secondHalf = recent.slice(Math.floor(recent.length / 2));
  const firstAvg = firstHalf.reduce((sum, m) => sum + m.deltaHV, 0) / Math.max(1, firstHalf.length);
  const secondAvg = secondHalf.reduce((sum, m) => sum + m.deltaHV, 0) / Math.max(1, secondHalf.length);

  let trend: 'rising' | 'stable' | 'declining' = 'stable';
  if (secondAvg - firstAvg > 5) trend = 'rising';
  if (firstAvg - secondAvg > 5) trend = 'declining';

  // Predict tomorrow based on trend and day of week patterns
  const tomorrow = new Date();
  tomorrow.setDate(tomorrow.getDate() + 1);
  const tomorrowDay = tomorrow.getDay();

  // Check if we have data for this day of week
  const sameDayMetrics = recentMetrics.filter(m => new Date(m.date).getDay() === tomorrowDay);
  let predictedTomorrow = overall;

  if (sameDayMetrics.length > 0) {
    const sameDayAvg = sameDayMetrics.reduce((sum, m) => sum + m.deltaHV, 0) / sameDayMetrics.length;
    predictedTomorrow = (overall + sameDayAvg) / 2;
  }

  if (trend === 'rising') predictedTomorrow += 3;
  if (trend === 'declining') predictedTomorrow -= 3;

  return {
    physical: Math.round(physical),
    mental: Math.round(mental),
    emotional: Math.round(emotional),
    social: Math.round(social),
    creative: Math.round(creative),
    overall: Math.round(overall),
    trend,
    predictedTomorrow: Math.round(Math.max(0, Math.min(100, predictedTomorrow))),
  };
}

// ============================================================================
// Utility Functions
// ============================================================================

function calculateVariance(values: number[]): number {
  if (values.length === 0) return 0;
  const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
  const squaredDiffs = values.map(v => Math.pow(v - mean, 2));
  return squaredDiffs.reduce((sum, v) => sum + v, 0) / values.length;
}

function capitalize(str: string): string {
  return str.charAt(0).toUpperCase() + str.slice(1);
}

// ============================================================================
// Roadmap Service
// ============================================================================

const ROADMAPS_STORAGE_KEY = 'pulse-roadmaps';
const RECOMMENDATIONS_KEY = 'pulse-recommendations';

class RoadmapEngineService {
  private roadmaps: Roadmap[] = [];
  private recommendations: AIRecommendation[] = [];
  private listeners: Set<() => void> = new Set();
  private liveMetrics: LiveMetricState | null = null;

  async initialize(): Promise<void> {
    try {
      const stored = localStorage.getItem(ROADMAPS_STORAGE_KEY);
      if (stored) {
        this.roadmaps = JSON.parse(stored);
      }

      const recsStored = localStorage.getItem(RECOMMENDATIONS_KEY);
      if (recsStored) {
        this.recommendations = JSON.parse(recsStored);
      }
    } catch (error) {
      console.error('Failed to initialize roadmap engine:', error);
    }
  }

  getRoadmaps(): Roadmap[] {
    return this.roadmaps;
  }

  getRoadmap(category: string): Roadmap | undefined {
    return this.roadmaps.find(r => r.category === category);
  }

  getRecommendations(): AIRecommendation[] {
    // Filter out expired and dismissed recommendations
    const now = new Date();
    return this.recommendations.filter(r => {
      if (r.dismissed) return false;
      if (r.expiresAt && new Date(r.expiresAt) < now) return false;
      return true;
    });
  }

  async updateRoadmaps(
    profile: UserProfile,
    metrics: MetricsSnapshot[]
  ): Promise<{ roadmaps: Roadmap[]; recommendations: AIRecommendation[] }> {
    const friction = calculateFriction(profile, metrics);

    // Generate/update roadmaps for each category
    const categories: Array<'physical' | 'emotional' | 'mental' | 'spiritual' | 'social' | 'creative'> = [
      'physical', 'emotional', 'mental', 'spiritual', 'social', 'creative'
    ];

    this.roadmaps = categories.map(category => generateRoadmap(profile, category, friction));

    // Generate new recommendations
    const newRecs = generateActiveInferences(profile, friction, metrics);

    // Merge with existing (keep dismissed state)
    const existingIds = new Set(this.recommendations.map(r => r.id));
    newRecs.forEach(rec => {
      if (!existingIds.has(rec.id)) {
        this.recommendations.push(rec);
      }
    });

    // Keep only last 50 recommendations
    this.recommendations = this.recommendations.slice(-50);

    await this.save();
    this.notifyListeners();

    return { roadmaps: this.roadmaps, recommendations: this.getRecommendations() };
  }

  async updateStepStatus(
    roadmapId: string,
    stepId: string,
    status: RoadmapStep['status']
  ): Promise<void> {
    const roadmap = this.roadmaps.find(r => r.id === roadmapId);
    if (!roadmap) return;

    const step = roadmap.steps.find(s => s.id === stepId);
    if (!step) return;

    step.status = status;
    if (status === 'active') step.startedAt = new Date().toISOString();
    if (status === 'completed') step.completedAt = new Date().toISOString();

    roadmap.updatedAt = new Date().toISOString();

    await this.save();
    this.notifyListeners();
  }

  async dismissRecommendation(id: string, reason?: string): Promise<void> {
    const rec = this.recommendations.find(r => r.id === id);
    if (rec) {
      rec.dismissed = true;
      rec.dismissedReason = reason;
      await this.save();
      this.notifyListeners();
    }
  }

  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /**
   * Update with live metrics from metricsHub
   * Generates real-time recommendations based on field state
   */
  updateLiveMetrics(state: EnhancedDeltaHVState): void {
    const previousState = this.liveMetrics?.fieldState;

    this.liveMetrics = {
      symbolic: state.symbolicDensity,
      resonance: state.resonanceCoupling,
      friction: state.frictionCoefficient,
      stability: state.harmonicStability,
      deltaHV: state.deltaHV,
      fieldState: state.fieldState,
      musicInfluence: state.musicInfluence ? {
        skipRatio: state.musicInfluence.skipRatio,
        authorshipScore: state.musicInfluence.authorshipScore,
        healingProgress: state.musicInfluence.healingProgress,
      } : undefined,
    };

    // Generate field-state-aware recommendations
    const fieldStateRecs = this.generateFieldStateRecommendations(previousState);
    fieldStateRecs.forEach(rec => {
      const existing = this.recommendations.find(r => r.id === rec.id);
      if (!existing) {
        this.recommendations.push(rec);
      }
    });

    this.notifyListeners();
  }

  /**
   * Generate recommendations based on field state transitions
   */
  private generateFieldStateRecommendations(previousState?: string): AIRecommendation[] {
    if (!this.liveMetrics) return [];

    const recommendations: AIRecommendation[] = [];
    const now = new Date().toISOString();
    const { fieldState, symbolic, resonance, friction, stability, musicInfluence } = this.liveMetrics;

    // Field state transition recommendations
    if (previousState && previousState !== fieldState) {
      if (fieldState === 'fragmented') {
        recommendations.push({
          id: `rec-field-fragmented-${Date.now()}`,
          type: 'friction_reduction',
          priority: 'high',
          title: 'Field Fragmented - Recenter',
          rationale: `Your rhythm field transitioned to fragmented state. δφ: ${friction}%, Stability: ${stability}%. The FEP suggests immediate action to reduce surprise.`,
          suggestedAction: 'Pause and do a brief grounding exercise. Complete one anchor beat. Consider a 5-minute meditation.',
          expectedOutcome: 'Field stabilization, reduced friction',
          relatedBeat: 'Meditation',
          confidence: 85,
          createdAt: now,
          expiresAt: new Date(Date.now() + 2 * 60 * 60000).toISOString(),
        });
      } else if (fieldState === 'coherent' && previousState !== 'coherent') {
        recommendations.push({
          id: `rec-field-coherent-${Date.now()}`,
          type: 'pattern_insight',
          priority: 'low',
          title: 'Coherence Achieved!',
          rationale: `Your rhythm field reached coherent state. S: ${symbolic}%, R: ${resonance}%, H: ${stability}%. This is optimal low-free-energy operation.`,
          suggestedAction: 'Note what you did to reach this state. Consider tackling something challenging while coherent.',
          expectedOutcome: 'Sustained coherence, productive flow',
          confidence: 90,
          createdAt: now,
          expiresAt: new Date(Date.now() + 4 * 60 * 60000).toISOString(),
        });
      }
    }

    // Low symbolic density recommendations
    if (symbolic < 30) {
      recommendations.push({
        id: `rec-symbolic-low-${Date.now()}`,
        type: 'goal_alignment',
        priority: symbolic < 15 ? 'high' : 'medium',
        title: 'Symbolic Density Low',
        rationale: `Symbolic (S) is at ${symbolic}%. Low symbolic density indicates disconnection from meaning and intention. Journaling increases glyphic resonance.`,
        suggestedAction: 'Write in your journal about your current intentions. Use glyphs/emojis to express emotions symbolically.',
        expectedOutcome: 'Increased symbolic density, clearer intention',
        relatedBeat: 'Journal',
        confidence: 75,
        createdAt: now,
        expiresAt: new Date(Date.now() + 6 * 60 * 60000).toISOString(),
      });
    }

    // Low resonance recommendations
    if (resonance < 40) {
      recommendations.push({
        id: `rec-resonance-low-${Date.now()}`,
        type: 'energy_optimization',
        priority: resonance < 25 ? 'high' : 'medium',
        title: 'Resonance Coupling Low',
        rationale: `Resonance (R) is at ${resonance}%. Low resonance indicates misalignment between intentions and actions. Completing aligned tasks increases resonance.`,
        suggestedAction: 'Focus on completing one beat that aligns with your stated goals. Check your task-goal alignment.',
        expectedOutcome: 'Improved intention-action coupling',
        relatedBeat: 'General',
        confidence: 70,
        createdAt: now,
        expiresAt: new Date(Date.now() + 4 * 60 * 60000).toISOString(),
      });
    }

    // Music authorship insights
    if (musicInfluence && musicInfluence.authorshipScore > 70) {
      recommendations.push({
        id: `rec-authorship-high-${Date.now()}`,
        type: 'pattern_insight',
        priority: 'low',
        title: 'Strong Story Authorship',
        rationale: `Your authorship score is ${musicInfluence.authorshipScore}%. You're actively controlling your emotional narrative through music choices.`,
        suggestedAction: musicInfluence.healingProgress > 50
          ? 'Your healing indicators are strong. Continue processing through music.'
          : 'Keep engaging with music mindfully. Your choices are shaping your emotional trajectory.',
        expectedOutcome: 'Continued emotional sovereignty',
        confidence: 80,
        createdAt: now,
        expiresAt: new Date(Date.now() + 8 * 60 * 60000).toISOString(),
      });
    }

    return recommendations;
  }

  /**
   * Get current live metrics
   */
  getLiveMetrics(): LiveMetricState | null {
    return this.liveMetrics;
  }

  /**
   * Get recommendations filtered by current field state
   */
  getFieldStateAwareRecommendations(): AIRecommendation[] {
    const base = this.getRecommendations();
    if (!this.liveMetrics) return base;

    // Prioritize based on field state
    const { fieldState } = this.liveMetrics;

    return base.sort((a, b) => {
      // In fragmented state, prioritize friction_reduction
      if (fieldState === 'fragmented') {
        if (a.type === 'friction_reduction' && b.type !== 'friction_reduction') return -1;
        if (b.type === 'friction_reduction' && a.type !== 'friction_reduction') return 1;
      }

      // In coherent state, prioritize goal_alignment
      if (fieldState === 'coherent') {
        if (a.type === 'goal_alignment' && b.type !== 'goal_alignment') return -1;
        if (b.type === 'goal_alignment' && a.type !== 'goal_alignment') return 1;
      }

      // Default: by priority
      const priorityOrder = { critical: 4, high: 3, medium: 2, low: 1 };
      return priorityOrder[b.priority] - priorityOrder[a.priority];
    });
  }

  private notifyListeners(): void {
    this.listeners.forEach(l => l());
  }

  private async save(): Promise<void> {
    localStorage.setItem(ROADMAPS_STORAGE_KEY, JSON.stringify(this.roadmaps));
    localStorage.setItem(RECOMMENDATIONS_KEY, JSON.stringify(this.recommendations));
  }
}

export const roadmapEngine = new RoadmapEngineService();
export default roadmapEngine;
