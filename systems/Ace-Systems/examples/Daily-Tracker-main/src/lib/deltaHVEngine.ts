/**
 * DeltaHV Engine - Coherence Metrics for Rhythm Pulse
 *
 * This module calculates four core ΔHV (Delta Harmonic Value) metrics that measure
 * the user's daily coherence and rhythm alignment:
 *
 * - SymbolicDensity (S): Measures symbolic/intentional compression from journal entries and glyphs
 * - ResonanceCoupling (R): Alignment ratio between planned (calendar) and logged tasks
 * - FrictionCoefficient (δφ): Energy lost to missed, delayed, or incomplete logs
 * - HarmonicStability (H): Consistency of check-in intervals across day segments
 *
 * These metrics inform the DayRhythmState for UI display and AI prompt conditioning.
 */

// Glyph patterns that contribute to symbolic density
const SYMBOLIC_GLYPHS = [
  '🌀', '∞', '⚡', '🌊', '✨', '🔮', '💫', '🎯', '🧿', '☯️',
  '🌙', '☀️', '⭐', '🌸', '🍃', '🔥', '💎', '🦋', '🐉', '🌈',
  // Affirmation markers
  'affirm', 'intention', 'gratitude', 'manifest', 'align', 'flow',
  'breathe', 'ground', 'center', 'anchor', 'release', 'expand'
];

// Emotional resonance keywords that add to symbolic density
const RESONANCE_KEYWORDS = [
  'grateful', 'thankful', 'appreciate', 'love', 'peace', 'calm',
  'focused', 'clear', 'present', 'aware', 'mindful', 'connected',
  'aligned', 'balanced', 'centered', 'grounded', 'stable', 'strong'
];

export interface CheckIn {
  id: string;
  category: string;
  task: string;
  waveId?: string;
  slot: string;
  loggedAt: string;
  note?: string;
  done: boolean;
  expanded?: boolean;
  isAnchor?: boolean;
}

export interface JournalEntry {
  id: string;
  timestamp: string;
  content: string;
  waveId?: string;
}

export interface Wave {
  id: string;
  name: string;
  description: string;
  color: string;
  startHour: number;
  endHour: number;
}

export interface RhythmProfile {
  waves: Wave[];
  setupComplete: boolean;
  deviationDay?: number;
  wakeTime?: { hours: number; minutes: number };
}

/**
 * DeltaHV State - The four core coherence metrics
 */
export interface DeltaHVState {
  // Symbolic Density (0-100): Harmonic compression from symbolic/intentional content
  symbolicDensity: number;

  // Resonance Coupling (0-100): Alignment between planned and executed activities
  resonanceCoupling: number;

  // Friction Coefficient (0-100): Energy lost to delays, misses, and incomplete tasks (lower is better)
  frictionCoefficient: number;

  // Harmonic Stability (0-100): Consistency of check-in intervals
  harmonicStability: number;

  // Composite ΔHV Score (0-100): Weighted combination of all metrics
  deltaHV: number;

  // Field coherence state
  fieldState: 'coherent' | 'transitioning' | 'fragmented' | 'dormant';

  // Detailed breakdown for AI prompt conditioning
  breakdown: {
    glyphCount: number;
    intentionCount: number;
    alignedTasks: number;
    totalPlannedTasks: number;
    missedTasks: number;
    delayedTasks: number;
    avgDelayMinutes: number;
    checkInIntervals: number[];
    segmentCoverage: Record<string, boolean>;
  };

  // Timestamp of calculation
  calculatedAt: string;
}

/**
 * Extended Day Rhythm State including ΔHV metrics
 */
export interface DayRhythmState {
  date: string;
  totalFocusMinutes: number;
  totalBreakMinutes: number;
  energyScore: number;
  rhythmScore: number;
  segments: Array<{
    name: string;
    focusMinutes: number;
    breakMinutes: number;
    waveId: string;
  }>;
  deltaHV: DeltaHVState;
}

/**
 * Helper to check if two dates are the same day
 */
const sameDay = (a: Date, b: Date): boolean =>
  a.getFullYear() === b.getFullYear() &&
  a.getMonth() === b.getMonth() &&
  a.getDate() === b.getDate();

/**
 * DeltaHV Engine - Calculates coherence metrics from tracker data
 */
export class DeltaHVEngine {
  private checkIns: CheckIn[];
  private journals: Record<string, JournalEntry[]>;
  private rhythmProfile: RhythmProfile;
  private targetDate: Date;

  constructor(
    checkIns: CheckIn[],
    journals: Record<string, JournalEntry[]>,
    rhythmProfile: RhythmProfile,
    targetDate: Date = new Date()
  ) {
    this.checkIns = checkIns;
    this.journals = journals;
    this.rhythmProfile = rhythmProfile;
    this.targetDate = targetDate;
  }

  /**
   * Get the date key for journal lookup
   */
  private getDateKey(date: Date): string {
    const pad2 = (n: number) => n.toString().padStart(2, '0');
    return `${date.getFullYear()}-${pad2(date.getMonth() + 1)}-${pad2(date.getDate())}`;
  }

  /**
   * Get check-ins for the target date
   */
  private getDayCheckIns(): CheckIn[] {
    return this.checkIns.filter(c => sameDay(new Date(c.slot), this.targetDate));
  }

  /**
   * Get completed check-ins for the target date
   */
  private getCompletedCheckIns(): CheckIn[] {
    return this.checkIns.filter(c => c.done && sameDay(new Date(c.loggedAt), this.targetDate));
  }

  /**
   * Get journal entries for the target date
   */
  private getDayJournals(): JournalEntry[] {
    const dayKey = this.getDateKey(this.targetDate);
    return this.journals[dayKey] || [];
  }

  /**
   * Calculate Symbolic Density (S)
   *
   * Measures the harmonic compression of symbolic content in journal entries.
   * Higher density indicates more intentional, symbolic, and resonant journaling.
   *
   * Formula: S = (glyphScore + intentionScore + resonanceScore) / 3
   */
  calculateSymbolicDensity(): { score: number; glyphCount: number; intentionCount: number } {
    const journals = this.getDayJournals();

    if (journals.length === 0) {
      return { score: 0, glyphCount: 0, intentionCount: 0 };
    }

    let glyphCount = 0;
    let intentionCount = 0;
    let resonanceCount = 0;
    let totalWordCount = 0;

    journals.forEach(entry => {
      const content = entry.content.toLowerCase();
      const words = content.split(/\s+/).filter(w => w.length > 0);
      totalWordCount += words.length;

      // Count symbolic glyphs and patterns
      SYMBOLIC_GLYPHS.forEach(glyph => {
        const regex = new RegExp(glyph.toLowerCase(), 'gi');
        const matches = content.match(regex);
        if (matches) {
          glyphCount += matches.length;
        }
      });

      // Count intention patterns (affirmations, goals, etc.)
      const intentionPatterns = [
        /i (will|am|choose|intend|commit)/gi,
        /my (intention|goal|purpose|focus)/gi,
        /today i/gi,
        /i feel/gi,
        /gratitude/gi,
        /thankful/gi
      ];

      intentionPatterns.forEach(pattern => {
        const matches = content.match(pattern);
        if (matches) {
          intentionCount += matches.length;
        }
      });

      // Count emotional resonance keywords
      RESONANCE_KEYWORDS.forEach(keyword => {
        const regex = new RegExp(`\\b${keyword}\\b`, 'gi');
        const matches = content.match(regex);
        if (matches) {
          resonanceCount += matches.length;
        }
      });
    });

    // Calculate density scores (normalized to 0-100)
    // More entries and more symbolic content = higher density
    const entriesBonus = Math.min(journals.length * 10, 30); // Up to 30 points for multiple entries
    const glyphScore = Math.min(glyphCount * 8, 25); // Up to 25 points for glyphs
    const intentionScore = Math.min(intentionCount * 6, 25); // Up to 25 points for intentions
    const resonanceScore = Math.min(resonanceCount * 4, 20); // Up to 20 points for resonance

    const score = Math.min(100, entriesBonus + glyphScore + intentionScore + resonanceScore);

    return { score, glyphCount, intentionCount };
  }

  /**
   * Calculate Resonance Coupling (R)
   *
   * Measures alignment between planned calendar events and actual logged completions.
   * Higher coupling indicates better synchronization between intention and action.
   *
   * Formula: R = (completedPlanned / totalPlanned) * 100
   */
  calculateResonanceCoupling(): { score: number; alignedTasks: number; totalPlanned: number } {
    const dayCheckIns = this.getDayCheckIns();
    const completedCheckIns = this.getCompletedCheckIns();

    // Filter to only scheduled (planned) tasks
    const plannedTasks = dayCheckIns.filter(c => {
      // A planned task is one that was scheduled ahead of time
      const slotTime = new Date(c.slot).getTime();
      const loggedTime = new Date(c.loggedAt).getTime();
      // If logged time is more than 5 minutes before slot time, it was planned
      return loggedTime < slotTime - 5 * 60 * 1000 || c.isAnchor || c.id.startsWith('gcal-');
    });

    if (plannedTasks.length === 0) {
      // No planned tasks - check if there are any completed tasks at all
      const anyCompletedToday = completedCheckIns.length;
      if (anyCompletedToday > 0) {
        // Spontaneous activity is valued but less than aligned activity
        return { score: Math.min(50 + anyCompletedToday * 5, 70), alignedTasks: 0, totalPlanned: 0 };
      }
      return { score: 0, alignedTasks: 0, totalPlanned: 0 };
    }

    // Count how many planned tasks were completed
    const completedPlannedIds = new Set(completedCheckIns.map(c => c.id));
    const alignedTasks = plannedTasks.filter(t => completedPlannedIds.has(t.id)).length;

    // Calculate coupling score
    const baseScore = (alignedTasks / plannedTasks.length) * 100;

    // Bonus for completing anchors
    const completedAnchors = completedCheckIns.filter(c => c.isAnchor).length;
    const anchorBonus = Math.min(completedAnchors * 5, 15);

    const score = Math.min(100, baseScore + anchorBonus);

    return { score, alignedTasks, totalPlanned: plannedTasks.length };
  }

  /**
   * Calculate Friction Coefficient (δφ)
   *
   * Measures energy lost to delays, missed tasks, and incomplete activities.
   * LOWER friction is BETTER (indicates smooth flow).
   *
   * Formula: δφ = (missedWeight + delayWeight + incompleteWeight) / 3
   */
  calculateFrictionCoefficient(): {
    score: number;
    missedTasks: number;
    delayedTasks: number;
    avgDelayMinutes: number
  } {
    const dayCheckIns = this.getDayCheckIns();
    const now = new Date();
    const isToday = sameDay(this.targetDate, now);

    if (dayCheckIns.length === 0) {
      return { score: 0, missedTasks: 0, delayedTasks: 0, avgDelayMinutes: 0 };
    }

    let missedTasks = 0;
    let delayedTasks = 0;
    let totalDelayMinutes = 0;
    let delayCount = 0;

    dayCheckIns.forEach(checkIn => {
      const slotTime = new Date(checkIn.slot).getTime();
      const loggedTime = new Date(checkIn.loggedAt).getTime();
      const nowTime = now.getTime();

      if (!checkIn.done) {
        // Check if the task is overdue
        if (slotTime < nowTime && isToday) {
          missedTasks++;
        } else if (!isToday) {
          // Past day, incomplete = missed
          missedTasks++;
        }
      } else {
        // Task is done - check for delays
        // A delay is when logged completion is significantly after the slot time
        if (loggedTime > slotTime + 15 * 60 * 1000) { // 15+ minutes late
          delayedTasks++;
          const delayMinutes = (loggedTime - slotTime) / (1000 * 60);
          totalDelayMinutes += delayMinutes;
          delayCount++;
        }
      }
    });

    const avgDelayMinutes = delayCount > 0 ? totalDelayMinutes / delayCount : 0;

    // Calculate friction components
    const missedWeight = (missedTasks / dayCheckIns.length) * 50; // Up to 50 points for misses
    const delayWeight = Math.min((avgDelayMinutes / 60) * 30, 30); // Up to 30 points for delays
    const incompleteWeight = ((dayCheckIns.length - dayCheckIns.filter(c => c.done).length) / dayCheckIns.length) * 20;

    const score = Math.min(100, missedWeight + delayWeight + incompleteWeight);

    return { score, missedTasks, delayedTasks, avgDelayMinutes: Math.round(avgDelayMinutes) };
  }

  /**
   * Calculate Harmonic Stability (H)
   *
   * Measures the consistency of check-in intervals across day segments.
   * Higher stability indicates regular, rhythmic engagement throughout the day.
   *
   * Formula: H = 100 - (intervalVariance * varianceWeight)
   */
  calculateHarmonicStability(): {
    score: number;
    checkInIntervals: number[];
    segmentCoverage: Record<string, boolean>
  } {
    const completedCheckIns = this.getCompletedCheckIns()
      .sort((a, b) => new Date(a.loggedAt).getTime() - new Date(b.loggedAt).getTime());

    // Calculate segment coverage based on waves
    const segmentCoverage: Record<string, boolean> = {};
    this.rhythmProfile.waves.forEach(wave => {
      const hasActivity = completedCheckIns.some(c => c.waveId === wave.id);
      segmentCoverage[wave.id] = hasActivity;
    });

    if (completedCheckIns.length < 2) {
      // Not enough data points
      const coveredSegments = Object.values(segmentCoverage).filter(Boolean).length;
      const coverageBonus = (coveredSegments / this.rhythmProfile.waves.length) * 30;
      return {
        score: Math.min(completedCheckIns.length * 15 + coverageBonus, 40),
        checkInIntervals: [],
        segmentCoverage
      };
    }

    // Calculate intervals between check-ins (in minutes)
    const intervals: number[] = [];
    for (let i = 1; i < completedCheckIns.length; i++) {
      const prevTime = new Date(completedCheckIns[i - 1].loggedAt).getTime();
      const currTime = new Date(completedCheckIns[i].loggedAt).getTime();
      const intervalMinutes = (currTime - prevTime) / (1000 * 60);
      intervals.push(intervalMinutes);
    }

    // Calculate interval statistics
    const avgInterval = intervals.reduce((a, b) => a + b, 0) / intervals.length;
    const variance = intervals.reduce((sum, interval) => {
      return sum + Math.pow(interval - avgInterval, 2);
    }, 0) / intervals.length;
    const stdDev = Math.sqrt(variance);

    // Coefficient of variation (normalized variability)
    const cv = avgInterval > 0 ? (stdDev / avgInterval) : 1;

    // Calculate base stability score (lower CV = higher stability)
    // CV of 0 = perfect regularity (100), CV of 1+ = high variability (lower score)
    const stabilityFromVariance = Math.max(0, 100 - cv * 60);

    // Bonus for segment coverage
    const coveredSegments = Object.values(segmentCoverage).filter(Boolean).length;
    const coverageBonus = (coveredSegments / this.rhythmProfile.waves.length) * 25;

    // Bonus for number of check-ins (more engagement = more stability data)
    const engagementBonus = Math.min(completedCheckIns.length * 3, 15);

    const score = Math.min(100, stabilityFromVariance + coverageBonus + engagementBonus);

    return { score, checkInIntervals: intervals.map(i => Math.round(i)), segmentCoverage };
  }

  /**
   * Determine field coherence state based on ΔHV score
   */
  private getFieldState(deltaHV: number): 'coherent' | 'transitioning' | 'fragmented' | 'dormant' {
    if (deltaHV >= 75) return 'coherent';
    if (deltaHV >= 50) return 'transitioning';
    if (deltaHV >= 25) return 'fragmented';
    return 'dormant';
  }

  /**
   * Get the complete DeltaHV State
   *
   * This is the main method that calculates all metrics and returns the full state
   * for UI display and AI prompt conditioning.
   */
  getDeltaHVState(): DeltaHVState {
    const symbolicResult = this.calculateSymbolicDensity();
    const resonanceResult = this.calculateResonanceCoupling();
    const frictionResult = this.calculateFrictionCoefficient();
    const stabilityResult = this.calculateHarmonicStability();

    // Calculate composite ΔHV score
    // Weights: S (20%), R (30%), inverse δφ (25%), H (25%)
    // Note: Friction is inverted because lower friction is better
    const inverseFriction = 100 - frictionResult.score;
    const deltaHV =
      symbolicResult.score * 0.20 +
      resonanceResult.score * 0.30 +
      inverseFriction * 0.25 +
      stabilityResult.score * 0.25;

    return {
      symbolicDensity: Math.round(symbolicResult.score),
      resonanceCoupling: Math.round(resonanceResult.score),
      frictionCoefficient: Math.round(frictionResult.score),
      harmonicStability: Math.round(stabilityResult.score),
      deltaHV: Math.round(deltaHV),
      fieldState: this.getFieldState(deltaHV),
      breakdown: {
        glyphCount: symbolicResult.glyphCount,
        intentionCount: symbolicResult.intentionCount,
        alignedTasks: resonanceResult.alignedTasks,
        totalPlannedTasks: resonanceResult.totalPlanned,
        missedTasks: frictionResult.missedTasks,
        delayedTasks: frictionResult.delayedTasks,
        avgDelayMinutes: frictionResult.avgDelayMinutes,
        checkInIntervals: stabilityResult.checkInIntervals,
        segmentCoverage: stabilityResult.segmentCoverage
      },
      calculatedAt: new Date().toISOString()
    };
  }

  /**
   * Get the full Day Rhythm State including ΔHV metrics
   */
  getDayRhythmState(): DayRhythmState {
    const dayCheckIns = this.getDayCheckIns();
    const completedCheckIns = this.getCompletedCheckIns();
    const deltaHV = this.getDeltaHVState();

    // Calculate segment data
    const segments = this.rhythmProfile.waves.map(wave => {
      const waveCheckIns = dayCheckIns.filter(c => c.waveId === wave.id);
      const waveCompleted = completedCheckIns.filter(c => c.waveId === wave.id);

      // Estimate focus minutes (30 min per completed task as default)
      const focusMinutes = waveCompleted.length * 30;
      const breakMinutes = Math.max(0, (waveCheckIns.length - waveCompleted.length) * 15);

      return {
        name: wave.name,
        focusMinutes,
        breakMinutes,
        waveId: wave.id
      };
    });

    // Calculate totals
    const totalFocusMinutes = segments.reduce((sum, s) => sum + s.focusMinutes, 0);
    const totalBreakMinutes = segments.reduce((sum, s) => sum + s.breakMinutes, 0);

    // Calculate energy score based on ΔHV metrics
    const energyScore = Math.round(
      (100 - deltaHV.frictionCoefficient) * 0.5 +
      deltaHV.resonanceCoupling * 0.3 +
      deltaHV.harmonicStability * 0.2
    );

    // Calculate rhythm score (based on anchor completion)
    const dayAnchors = dayCheckIns.filter(c => c.isAnchor);
    const completedAnchors = completedCheckIns.filter(c => c.isAnchor);
    const rhythmScore = dayAnchors.length > 0
      ? Math.round((completedAnchors.length / dayAnchors.length) * 100)
      : 0;

    return {
      date: this.getDateKey(this.targetDate),
      totalFocusMinutes,
      totalBreakMinutes,
      energyScore,
      rhythmScore,
      segments,
      deltaHV
    };
  }

  /**
   * Generate AI prompt conditioning based on current ΔHV state
   */
  getAIPromptCondition(): string {
    const state = this.getDeltaHVState();

    let condition = `Current ΔHV State: ${state.deltaHV}/100 (${state.fieldState})\n`;
    condition += `- Symbolic Density: ${state.symbolicDensity}/100\n`;
    condition += `- Resonance Coupling: ${state.resonanceCoupling}/100\n`;
    condition += `- Friction Coefficient: ${state.frictionCoefficient}/100 (lower is better)\n`;
    condition += `- Harmonic Stability: ${state.harmonicStability}/100\n\n`;

    // Add contextual guidance
    if (state.fieldState === 'coherent') {
      condition += 'User is in a highly coherent state. Support their momentum and suggest optimization opportunities.';
    } else if (state.fieldState === 'transitioning') {
      condition += 'User is transitioning between states. Provide gentle guidance to maintain positive trajectory.';
    } else if (state.fieldState === 'fragmented') {
      condition += 'User shows fragmented coherence. Focus on grounding exercises and simplified task completion.';
    } else {
      condition += 'User field is dormant. Encourage small symbolic actions to initiate coherence.';
    }

    return condition;
  }
}

/**
 * Factory function to create DeltaHVEngine with current app state
 */
export function createDeltaHVEngine(
  checkIns: CheckIn[],
  journals: Record<string, JournalEntry[]>,
  rhythmProfile: RhythmProfile,
  targetDate?: Date
): DeltaHVEngine {
  return new DeltaHVEngine(checkIns, journals, rhythmProfile, targetDate);
}

/**
 * Quick helper to get ΔHV state without creating engine explicitly
 */
export function getDeltaHVState(
  checkIns: CheckIn[],
  journals: Record<string, JournalEntry[]>,
  rhythmProfile: RhythmProfile,
  targetDate?: Date
): DeltaHVState {
  const engine = createDeltaHVEngine(checkIns, journals, rhythmProfile, targetDate);
  return engine.getDeltaHVState();
}
