/**
 * Metrics Hub - Central Real-Time Metrics Integration
 *
 * This is the SINGLE SOURCE OF TRUTH for all DeltaHV metrics.
 * It aggregates data from:
 * - Check-ins (task completion, timing)
 * - Journals (symbolic content)
 * - Music (skip patterns, emotional choices)
 * - Habits (consistency)
 * - Neural activation patterns
 *
 * Every space of resonance flows through this hub.
 */

import type { DeltaHVState, CheckIn, JournalEntry, RhythmProfile } from './deltaHVEngine';
import { DeltaHVEngine } from './deltaHVEngine';
import { storyShuffleEngine } from './storyShuffleEngine';
import { userProfileService } from './userProfile';

// ============================================================================
// Types
// ============================================================================

export interface MetricSource {
  id: string;
  name: string;
  weight: number;
  lastUpdate: string;
  contribution: {
    symbolic: number;
    resonance: number;
    friction: number;
    stability: number;
  };
}

export interface EnhancedDeltaHVState extends DeltaHVState {
  // Additional source contributions
  sources: MetricSource[];

  // Real-time indicators
  isLive: boolean;
  updateCount: number;
  lastAction: string;

  // Music-specific metrics
  musicInfluence: {
    skipRatio: number;
    authorshipScore: number;
    healingProgress: number;
    emotionalTrajectory: string;
  };

  // Brain region activation summary
  brainActivation: {
    symbolic: string[];    // Active regions for symbolic
    resonance: string[];   // Active regions for resonance
    friction: string[];    // Active regions for friction
    stability: string[];   // Active regions for stability
  };
}

export interface MetricsSnapshot {
  timestamp: string;
  symbolic: number;
  resonance: number;
  friction: number;
  stability: number;
  deltaHV: number;
  trigger: string;
}

// ============================================================================
// Brain Region Mapping for Metrics
// ============================================================================

const METRIC_BRAIN_REGIONS = {
  symbolic: [
    { id: 'dlpfc', name: 'Dorsolateral', glyph: '🎯' },
    { id: 'mpfc', name: 'Medial', glyph: '🪞' },
    { id: 'precuneus', name: 'Precuneus', glyph: '🌌' },
    { id: 'temporalpole', name: 'Temporal', glyph: '🏛️' },
    { id: 'hippocampus', name: 'Hippocampus', glyph: '📝' },
    { id: 'pcc', name: 'Posterior', glyph: '🧘' },
  ],
  resonance: [
    { id: 'ofc', name: 'Orbitofrontal', glyph: '⚖️' },
    { id: 'dacc', name: 'Dorsal', glyph: '⚠️' },
    { id: 'racc', name: 'Rostral', glyph: '💗' },
    { id: 'premotor', name: 'Premotor', glyph: '🎬' },
    { id: 'sma', name: 'Supplementary', glyph: '🔄' },
    { id: 'nucleus_accumbens', name: 'Nucleus', glyph: '🎯' },
  ],
  friction: [
    { id: 'sgacc', name: 'Subgenual', glyph: '🌧️' },
    { id: 'amygdala_bla', name: 'Basolateral', glyph: '⚡' },
    { id: 'amygdala_cea', name: 'Central', glyph: '🚨' },
    { id: 'pvn', name: 'Paraventricular', glyph: '🌊' },
    { id: 'bed_nucleus', name: 'Bed', glyph: '😰' },
    { id: 'habenula', name: 'Habenula', glyph: '🚫' },
  ],
  stability: [
    { id: 'septum', name: 'Septal', glyph: '😌' },
    { id: 'raphe_dorsal', name: 'Dorsal', glyph: '☮️' },
    { id: 'raphe_median', name: 'Median', glyph: '🌅' },
    { id: 'reticular', name: 'Reticular', glyph: '🔋' },
    { id: 'vermis', name: 'Dorsal', glyph: '🧘' },
    { id: 'cerebellar', name: 'Cerebellar', glyph: '⚖️' },
  ],
};

// ============================================================================
// Metrics Hub Class
// ============================================================================

const METRICS_HISTORY_KEY = 'metrics-hub-history';
const MAX_HISTORY_SIZE = 500;

class MetricsHub {
  private currentState: EnhancedDeltaHVState | null = null;
  private history: MetricsSnapshot[] = [];
  private listeners: Set<(state: EnhancedDeltaHVState) => void> = new Set();
  private updateCount = 0;
  private lastAction = 'initialized';

  // Data sources
  private checkIns: CheckIn[] = [];
  private journals: Record<string, JournalEntry[]> = {};
  private rhythmProfile: RhythmProfile | null = null;

  constructor() {
    this.loadHistory();
    this.setupMusicSubscription();
  }

  // ========================================
  // Initialization
  // ========================================

  private loadHistory(): void {
    try {
      const saved = localStorage.getItem(METRICS_HISTORY_KEY);
      if (saved) {
        this.history = JSON.parse(saved);
      }
    } catch (error) {
      console.error('Failed to load metrics history:', error);
    }
  }

  private saveHistory(): void {
    try {
      // Keep only recent history
      if (this.history.length > MAX_HISTORY_SIZE) {
        this.history = this.history.slice(-MAX_HISTORY_SIZE);
      }
      localStorage.setItem(METRICS_HISTORY_KEY, JSON.stringify(this.history));
    } catch (error) {
      console.error('Failed to save metrics history:', error);
    }
  }

  private setupMusicSubscription(): void {
    // Subscribe to music/skip events
    storyShuffleEngine.subscribe(() => {
      this.onMusicUpdate();
    });
  }

  // ========================================
  // Data Source Updates
  // ========================================

  /**
   * Update check-ins data source
   */
  updateCheckIns(checkIns: CheckIn[]): void {
    this.checkIns = checkIns;
    this.lastAction = 'check-in updated';
    this.recalculateMetrics('check-in');
  }

  /**
   * Update journals data source
   */
  updateJournals(journals: Record<string, JournalEntry[]>): void {
    this.journals = journals;
    this.lastAction = 'journal updated';
    this.recalculateMetrics('journal');
  }

  /**
   * Update rhythm profile
   */
  updateRhythmProfile(profile: RhythmProfile): void {
    this.rhythmProfile = profile;
    this.lastAction = 'profile updated';
    this.recalculateMetrics('profile');
  }

  /**
   * Called when music data changes
   */
  private onMusicUpdate(): void {
    this.lastAction = 'music/skip event';
    this.recalculateMetrics('music');
  }

  /**
   * Manual trigger for habit updates
   */
  updateFromHabit(habitName: string, completed: boolean): void {
    this.lastAction = `habit: ${habitName} ${completed ? 'completed' : 'missed'}`;
    this.recalculateMetrics('habit');
  }

  /**
   * Manual trigger for any action
   */
  recordAction(action: string): void {
    this.lastAction = action;
    this.recalculateMetrics('action');
  }

  // ========================================
  // Core Calculation
  // ========================================

  /**
   * Recalculate all metrics from all sources
   */
  private recalculateMetrics(trigger: string): void {
    this.updateCount++;

    // Base calculation from DeltaHVEngine
    const baseMetrics = this.calculateBaseMetrics();

    // Music influence
    const musicInfluence = this.calculateMusicInfluence();

    // Combine sources
    const sources = this.calculateSourceContributions(baseMetrics, musicInfluence);

    // Calculate final enhanced metrics
    const enhanced = this.combineMetrics(baseMetrics, musicInfluence, sources);

    // Calculate brain activation
    const brainActivation = this.calculateBrainActivation(enhanced);

    // Create enhanced state
    this.currentState = {
      ...enhanced,
      sources,
      isLive: true,
      updateCount: this.updateCount,
      lastAction: this.lastAction,
      musicInfluence: {
        skipRatio: musicInfluence.skipRatio,
        authorshipScore: musicInfluence.authorshipScore,
        healingProgress: musicInfluence.healingProgress,
        emotionalTrajectory: musicInfluence.emotionalTrajectory,
      },
      brainActivation,
    };

    // Record snapshot
    this.recordSnapshot(trigger);

    // Notify listeners
    this.notifyListeners();
  }

  private calculateBaseMetrics(): DeltaHVState {
    if (!this.rhythmProfile) {
      // Return default state if no rhythm profile
      return {
        symbolicDensity: 0,
        resonanceCoupling: 0,
        frictionCoefficient: 65, // Default friction from user's example
        harmonicStability: 0,
        deltaHV: 0,
        fieldState: 'dormant',
        breakdown: {
          glyphCount: 0,
          intentionCount: 0,
          alignedTasks: 0,
          totalPlannedTasks: 0,
          missedTasks: 0,
          delayedTasks: 0,
          avgDelayMinutes: 0,
          checkInIntervals: [],
          segmentCoverage: {},
        },
        calculatedAt: new Date().toISOString(),
      };
    }

    const engine = new DeltaHVEngine(
      this.checkIns,
      this.journals,
      this.rhythmProfile,
      new Date()
    );

    return engine.getDeltaHVState();
  }

  private calculateMusicInfluence(): {
    symbolic: number;
    resonance: number;
    friction: number;
    stability: number;
    skipRatio: number;
    authorshipScore: number;
    healingProgress: number;
    emotionalTrajectory: string;
  } {
    const storyMetrics = storyShuffleEngine.getStoryMetrics();
    const musicMetrics = storyShuffleEngine.getMetricInfluenceFromMusic();

    if (!storyMetrics) {
      return {
        symbolic: 0,
        resonance: 0,
        friction: 0,
        stability: 0,
        skipRatio: 0,
        authorshipScore: 0,
        healingProgress: 0,
        emotionalTrajectory: 'steady',
      };
    }

    return {
      symbolic: musicMetrics.symbolic.value,
      resonance: musicMetrics.resonance.value,
      friction: musicMetrics.friction.value,
      stability: musicMetrics.stability.value,
      skipRatio: storyMetrics.skipRatio,
      authorshipScore: storyMetrics.authorshipScore,
      healingProgress: storyMetrics.melancholyProgress.healingIndicator,
      emotionalTrajectory: storyMetrics.emotionalTrajectory,
    };
  }

  private calculateSourceContributions(
    base: DeltaHVState,
    music: ReturnType<typeof this.calculateMusicInfluence>
  ): MetricSource[] {
    const sources: MetricSource[] = [];

    // Check-ins source (40% weight)
    sources.push({
      id: 'check-ins',
      name: 'Task Completion',
      weight: 0.4,
      lastUpdate: new Date().toISOString(),
      contribution: {
        symbolic: base.breakdown.intentionCount > 0 ? 20 : 0,
        resonance: base.resonanceCoupling * 0.4,
        friction: base.frictionCoefficient * 0.4,
        stability: base.harmonicStability * 0.4,
      },
    });

    // Journals source (25% weight)
    sources.push({
      id: 'journals',
      name: 'Journaling',
      weight: 0.25,
      lastUpdate: new Date().toISOString(),
      contribution: {
        symbolic: base.symbolicDensity * 0.25,
        resonance: base.breakdown.glyphCount > 0 ? 15 : 0,
        friction: 0, // Journals reduce friction
        stability: base.breakdown.glyphCount > 2 ? 10 : 0,
      },
    });

    // Music source (25% weight)
    sources.push({
      id: 'music',
      name: 'Music Choices',
      weight: 0.25,
      lastUpdate: new Date().toISOString(),
      contribution: {
        symbolic: music.symbolic * 0.25,
        resonance: music.resonance * 0.25,
        friction: music.friction * 0.25,
        stability: music.stability * 0.25,
      },
    });

    // Authorship/Skip source (10% weight)
    sources.push({
      id: 'authorship',
      name: 'Story Authorship',
      weight: 0.1,
      lastUpdate: new Date().toISOString(),
      contribution: {
        symbolic: music.authorshipScore * 0.1,
        resonance: music.healingProgress > 50 ? 10 : 0,
        friction: music.skipRatio > 0.5 ? -5 : 5, // Skipping reduces friction
        stability: music.authorshipScore > 60 ? 10 : 0,
      },
    });

    return sources;
  }

  private combineMetrics(
    base: DeltaHVState,
    music: ReturnType<typeof this.calculateMusicInfluence>,
    _sources: MetricSource[]
  ): DeltaHVState {
    // Weighted combination of all sources
    // Base metrics: 60%, Music: 30%, Authorship: 10%

    const combinedSymbolic = Math.round(
      base.symbolicDensity * 0.6 +
      music.symbolic * 0.3 +
      music.authorshipScore * 0.1
    );

    const combinedResonance = Math.round(
      base.resonanceCoupling * 0.5 +
      music.resonance * 0.3 +
      (music.healingProgress > 50 ? 20 : 0) // Healing boosts resonance
    );

    // Friction: base is primary, music can reduce it
    // High skip ratio = taking control = less friction
    const musicFrictionReduction = music.skipRatio > 0.3 ? music.skipRatio * 20 : 0;
    const combinedFriction = Math.max(0, Math.round(
      base.frictionCoefficient * 0.7 +
      music.friction * 0.3 -
      musicFrictionReduction
    ));

    const combinedStability = Math.round(
      base.harmonicStability * 0.5 +
      music.stability * 0.3 +
      music.authorshipScore * 0.2
    );

    // Recalculate deltaHV
    const inverseFriction = 100 - combinedFriction;
    const deltaHV = Math.round(
      combinedSymbolic * 0.20 +
      combinedResonance * 0.30 +
      inverseFriction * 0.25 +
      combinedStability * 0.25
    );

    // Determine field state
    let fieldState: DeltaHVState['fieldState'] = 'dormant';
    if (deltaHV >= 75) fieldState = 'coherent';
    else if (deltaHV >= 50) fieldState = 'transitioning';
    else if (deltaHV >= 25) fieldState = 'fragmented';

    return {
      symbolicDensity: Math.min(100, Math.max(0, combinedSymbolic)),
      resonanceCoupling: Math.min(100, Math.max(0, combinedResonance)),
      frictionCoefficient: Math.min(100, Math.max(0, combinedFriction)),
      harmonicStability: Math.min(100, Math.max(0, combinedStability)),
      deltaHV: Math.min(100, Math.max(0, deltaHV)),
      fieldState,
      breakdown: base.breakdown,
      calculatedAt: new Date().toISOString(),
    };
  }

  private calculateBrainActivation(metrics: DeltaHVState): EnhancedDeltaHVState['brainActivation'] {
    // Activate regions based on metric levels
    const getActiveRegions = (
      metricValue: number,
      regions: typeof METRIC_BRAIN_REGIONS.symbolic
    ): string[] => {
      // Number of active regions based on metric value
      const activeCount = Math.ceil((metricValue / 100) * regions.length);
      return regions.slice(0, activeCount).map(r => `${r.glyph} ${r.name}`);
    };

    return {
      symbolic: getActiveRegions(metrics.symbolicDensity, METRIC_BRAIN_REGIONS.symbolic),
      resonance: getActiveRegions(metrics.resonanceCoupling, METRIC_BRAIN_REGIONS.resonance),
      friction: getActiveRegions(metrics.frictionCoefficient, METRIC_BRAIN_REGIONS.friction),
      stability: getActiveRegions(metrics.harmonicStability, METRIC_BRAIN_REGIONS.stability),
    };
  }

  private recordSnapshot(trigger: string): void {
    if (!this.currentState) return;

    this.history.push({
      timestamp: new Date().toISOString(),
      symbolic: this.currentState.symbolicDensity,
      resonance: this.currentState.resonanceCoupling,
      friction: this.currentState.frictionCoefficient,
      stability: this.currentState.harmonicStability,
      deltaHV: this.currentState.deltaHV,
      trigger,
    });

    this.saveHistory();

    // Record to user profile for long-term tracking
    // Only save significant updates (not every micro-update)
    if (this.updateCount % 5 === 0 || trigger === 'action' || trigger === 'habit') {
      this.syncToUserProfile();
    }
  }

  /**
   * Sync current metrics to user profile for historical tracking
   */
  private async syncToUserProfile(): Promise<void> {
    if (!this.currentState) return;

    try {
      // Get today's check-ins to count completed beats
      const today = new Date().toISOString().split('T')[0];
      const todayCheckIns = this.checkIns.filter(c => c.slot.startsWith(today));
      const completedBeats = todayCheckIns.filter(c => c.done).length;
      const totalBeats = todayCheckIns.length;

      // Get today's journal entries
      const todayJournals = this.journals[today] || [];

      // Extract glyphs from journals
      const glyphPattern = /[\u{1F300}-\u{1F9FF}]/gu;
      const glyphsUsed: string[] = [];
      todayJournals.forEach(entry => {
        const matches = entry.content.match(glyphPattern);
        if (matches) glyphsUsed.push(...matches);
      });

      await userProfileService.recordMetricsSnapshot({
        deltaHV: this.currentState.deltaHV,
        rhythmScore: this.currentState.resonanceCoupling,
        frictionCoefficient: this.currentState.frictionCoefficient,
        symbolicDensity: this.currentState.symbolicDensity,
        resonanceCoupling: this.currentState.resonanceCoupling,
        harmonicStability: this.currentState.harmonicStability,
        completedBeats,
        totalBeats,
        journalEntries: todayJournals.length,
        glyphsUsed: [...new Set(glyphsUsed)],
        fieldState: this.currentState.fieldState,
      });
    } catch (error) {
      console.error('Failed to sync metrics to user profile:', error);
    }
  }

  // ========================================
  // Public API
  // ========================================

  /**
   * Get current enhanced metrics state
   */
  getState(): EnhancedDeltaHVState | null {
    return this.currentState;
  }

  /**
   * Get metrics history
   */
  getHistory(): MetricsSnapshot[] {
    return this.history;
  }

  /**
   * Get brain regions for a specific metric
   */
  getBrainRegionsForMetric(
    metric: 'symbolic' | 'resonance' | 'friction' | 'stability'
  ): typeof METRIC_BRAIN_REGIONS.symbolic {
    return METRIC_BRAIN_REGIONS[metric];
  }

  /**
   * Force a full recalculation
   */
  forceRecalculate(): void {
    this.recalculateMetrics('force');
  }

  /**
   * Subscribe to metric updates
   */
  subscribe(listener: (state: EnhancedDeltaHVState) => void): () => void {
    this.listeners.add(listener);
    // Send current state immediately
    if (this.currentState) {
      listener(this.currentState);
    }
    return () => this.listeners.delete(listener);
  }

  private notifyListeners(): void {
    if (!this.currentState) return;
    this.listeners.forEach(listener => listener(this.currentState!));
  }

  /**
   * Get formatted display for a metric with brain regions
   */
  getMetricDisplay(metric: 'symbolic' | 'resonance' | 'friction' | 'stability'): {
    value: number;
    label: string;
    regions: Array<{ glyph: string; name: string; active: boolean }>;
  } {
    const state = this.currentState;
    const regions = METRIC_BRAIN_REGIONS[metric];

    const value = state ? {
      symbolic: state.symbolicDensity,
      resonance: state.resonanceCoupling,
      friction: state.frictionCoefficient,
      stability: state.harmonicStability,
    }[metric] : 0;

    const labels = {
      symbolic: 'Symbolic (S)',
      resonance: 'Resonance (R)',
      friction: 'Friction (δφ)',
      stability: 'Stability (H)',
    };

    const activeCount = Math.ceil((value / 100) * regions.length);

    return {
      value,
      label: labels[metric],
      regions: regions.map((r, i) => ({
        glyph: r.glyph,
        name: r.name,
        active: i < activeCount,
      })),
    };
  }

  /**
   * Get all metrics with their brain region displays
   */
  getAllMetricDisplays(): Array<ReturnType<typeof this.getMetricDisplay>> {
    return [
      this.getMetricDisplay('symbolic'),
      this.getMetricDisplay('resonance'),
      this.getMetricDisplay('friction'),
      this.getMetricDisplay('stability'),
    ];
  }
}

// Export singleton
export const metricsHub = new MetricsHub();

// Export class for testing
export { MetricsHub };
