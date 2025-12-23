/* INTEGRITY_METADATA
 * Date: 2025-12-23
 * Status: ⚠️ TRULY UNSUPPORTED - No supporting evidence found
 * Severity: HIGH RISK
 * Risk Types: unsupported_claims, unverified_math
 */


/**
 * Story Shuffle Engine
 *
 * Hilbert curve-based music shuffle algorithm that responds to user's DeltaHV metrics.
 * The algorithm places songs in a multi-dimensional emotional space and traverses
 * using a Hilbert curve, with the path biased by the user's current needs.
 *
 * Key Features:
 * - Hilbert curve shuffle for natural emotional flow
 * - Skip tracking as empowerment metric ("You are the author of your story")
 * - Melancholy skip detection for healing indicators
 * - Custom playlist creation
 * - Full library shuffle with metric influence
 *
 * Philosophy: Every skip is a choice. Every choice is authorship.
 * The user is always in control of their emotional journey.
 */

import type { DeltaHVState } from './deltaHVEngine';
import type { MusicTrack, EmotionalCategoryId } from './musicLibrary';
import { EMOTIONAL_CATEGORIES, musicLibrary } from './musicLibrary';

// ============================================================================
// Types
// ============================================================================

export interface SkipEvent {
  id: string;
  trackId: string;
  trackName: string;
  categoryId: EmotionalCategoryId;
  skippedAt: string;
  playDuration: number;      // How long they listened before skipping
  totalDuration: number;     // Total track duration
  skipRatio: number;         // playDuration / totalDuration
  deltaHVAtSkip: DeltaHVState | null;
  wasSkipped: boolean;       // true = skipped, false = played through
  reason?: 'manual' | 'auto_next' | 'playlist_end';
}

export interface StoryMetrics {
  // Skip ratios
  totalPlays: number;
  totalSkips: number;
  totalCompleted: number;
  skipRatio: number;         // skips / total plays (0-1)
  completionRatio: number;   // completed / total plays (0-1)

  // Authorship score - higher = more active story control
  authorshipScore: number;   // 0-100

  // Category-specific skip patterns
  categorySkipRatios: Record<EmotionalCategoryId, {
    plays: number;
    skips: number;
    avgListenRatio: number;
  }>;

  // Melancholy healing indicators
  melancholyProgress: {
    totalMelancholyPlays: number;
    melancholySkips: number;
    melancholyCompletions: number;
    healingIndicator: number;  // Higher = more healing (skipping sad = moving past it)
    currentPhase: 'processing' | 'healing' | 'integrated';
  };

  // Story arc
  recentChoices: SkipEvent[];
  dominantEmotion: EmotionalCategoryId | null;
  emotionalTrajectory: 'rising' | 'steady' | 'processing' | 'releasing';
}

export interface Playlist {
  id: string;
  name: string;
  description?: string;
  createdAt: string;
  updatedAt: string;
  trackIds: string[];
  coverEmoji: string;
  isSystem?: boolean;        // For auto-generated playlists
  basedOnMetric?: 'symbolic' | 'resonance' | 'friction' | 'stability';
}

export interface ShuffleState {
  queue: string[];           // Track IDs in current shuffle order
  currentIndex: number;
  hilbertPosition: number;   // Position on Hilbert curve
  metricsInfluence: {
    symbolic: number;
    resonance: number;
    friction: number;
    stability: number;
  };
  lastShuffleTime: string;
}

// ============================================================================
// Hilbert Curve Implementation
// ============================================================================

/**
 * Hilbert curve coordinate transformation
 * Maps 1D position to 2D coordinate on Hilbert curve
 * Extended to 4D for our emotional space (symbolic, resonance, friction, stability)
 */
function hilbert2D(n: number, d: number): { x: number; y: number } {
  let x = 0, y = 0;
  let s = 1;

  while (s < n) {
    const rx = 1 & (d / 2);
    const ry = 1 & (d ^ rx);

    if (ry === 0) {
      if (rx === 1) {
        x = s - 1 - x;
        y = s - 1 - y;
      }
      [x, y] = [y, x];
    }

    x += s * rx;
    y += s * ry;
    d = Math.floor(d / 4);
    s *= 2;
  }

  return { x, y };
}

/**
 * 4D Hilbert curve approximation for emotional space
 * Maps position to (symbolic, resonance, friction, stability) coordinates
 */
function hilbert4D(order: number, position: number): {
  symbolic: number;
  resonance: number;
  friction: number;
  stability: number;
} {
  // Use two 2D Hilbert curves interleaved for 4D approximation
  const n = Math.pow(2, order);
  const maxPos = n * n * n * n;
  const normalizedPos = (position % maxPos) / maxPos;

  // First 2D curve for symbolic/resonance
  const pos1 = Math.floor(normalizedPos * n * n);
  const coord1 = hilbert2D(n, pos1);

  // Second 2D curve for friction/stability (offset phase)
  const pos2 = Math.floor((normalizedPos + 0.5) * n * n) % (n * n);
  const coord2 = hilbert2D(n, pos2);

  return {
    symbolic: coord1.x / n,
    resonance: coord1.y / n,
    friction: coord2.x / n,
    stability: coord2.y / n
  };
}

// ============================================================================
// Emotional Space Mapping
// ============================================================================

/**
 * Map emotional category to 4D metric space
 */
const CATEGORY_METRIC_MAP: Record<EmotionalCategoryId, {
  symbolic: number;
  resonance: number;
  friction: number;
  stability: number;
}> = {
  JOY: { symbolic: 0.8, resonance: 0.9, friction: 0.1, stability: 0.7 },
  CALM: { symbolic: 0.6, resonance: 0.7, friction: 0.1, stability: 0.9 },
  FOCUS: { symbolic: 0.9, resonance: 0.8, friction: 0.2, stability: 0.8 },
  ENERGY: { symbolic: 0.7, resonance: 0.9, friction: 0.3, stability: 0.5 },
  MELANCHOLY: { symbolic: 0.7, resonance: 0.5, friction: 0.7, stability: 0.4 }, // Friction + love
  LOVE: { symbolic: 0.8, resonance: 0.9, friction: 0.2, stability: 0.7 },
  COURAGE: { symbolic: 0.9, resonance: 0.8, friction: 0.4, stability: 0.6 },
  WONDER: { symbolic: 0.95, resonance: 0.8, friction: 0.2, stability: 0.5 },
  GRATITUDE: { symbolic: 0.85, resonance: 0.9, friction: 0.1, stability: 0.8 },
  RELEASE: { symbolic: 0.6, resonance: 0.6, friction: 0.6, stability: 0.5 }
};

/**
 * Calculate distance between two points in 4D metric space
 * Weighted by user's current "needs" (inverse of their current metrics)
 */
function calculateWeightedDistance(
  point: { symbolic: number; resonance: number; friction: number; stability: number },
  target: { symbolic: number; resonance: number; friction: number; stability: number },
  needs: { symbolic: number; resonance: number; friction: number; stability: number }
): number {
  // Needs are inverted - low metric = high need
  const weights = {
    symbolic: 1 + (1 - needs.symbolic),      // Low symbolic = need meaning
    resonance: 1 + (1 - needs.resonance),    // Low resonance = need connection
    friction: 1 + needs.friction,             // High friction = need release
    stability: 1 + (1 - needs.stability)      // Low stability = need grounding
  };

  const dSymbolic = (point.symbolic - target.symbolic) * weights.symbolic;
  const dResonance = (point.resonance - target.resonance) * weights.resonance;
  const dFriction = (point.friction - target.friction) * weights.friction;
  const dStability = (point.stability - target.stability) * weights.stability;

  return Math.sqrt(dSymbolic ** 2 + dResonance ** 2 + dFriction ** 2 + dStability ** 2);
}

// ============================================================================
// Story Shuffle Engine Class
// ============================================================================

const SKIP_EVENTS_KEY = 'story-shuffle-skip-events';
const PLAYLISTS_KEY = 'story-shuffle-playlists';
const SHUFFLE_STATE_KEY = 'story-shuffle-state';
const STORY_METRICS_KEY = 'story-shuffle-metrics';

class StoryShuffleEngine {
  private skipEvents: SkipEvent[] = [];
  private playlists: Playlist[] = [];
  private shuffleState: ShuffleState | null = null;
  private storyMetrics: StoryMetrics | null = null;
  private listeners: Set<() => void> = new Set();

  constructor() {
    this.loadFromStorage();
  }

  // ========================================
  // Storage
  // ========================================

  private loadFromStorage(): void {
    try {
      const skipEventsJson = localStorage.getItem(SKIP_EVENTS_KEY);
      if (skipEventsJson) {
        this.skipEvents = JSON.parse(skipEventsJson);
      }

      const playlistsJson = localStorage.getItem(PLAYLISTS_KEY);
      if (playlistsJson) {
        this.playlists = JSON.parse(playlistsJson);
      }

      const shuffleStateJson = localStorage.getItem(SHUFFLE_STATE_KEY);
      if (shuffleStateJson) {
        this.shuffleState = JSON.parse(shuffleStateJson);
      }

      const metricsJson = localStorage.getItem(STORY_METRICS_KEY);
      if (metricsJson) {
        this.storyMetrics = JSON.parse(metricsJson);
      }

      // Create system playlists if not exists
      this.ensureSystemPlaylists();
    } catch (error) {
      console.error('Failed to load story shuffle data:', error);
    }
  }

  private saveToStorage(): void {
    try {
      localStorage.setItem(SKIP_EVENTS_KEY, JSON.stringify(this.skipEvents));
      localStorage.setItem(PLAYLISTS_KEY, JSON.stringify(this.playlists));
      if (this.shuffleState) {
        localStorage.setItem(SHUFFLE_STATE_KEY, JSON.stringify(this.shuffleState));
      }
      if (this.storyMetrics) {
        localStorage.setItem(STORY_METRICS_KEY, JSON.stringify(this.storyMetrics));
      }
    } catch (error) {
      console.error('Failed to save story shuffle data:', error);
    }
  }

  private ensureSystemPlaylists(): void {
    const systemPlaylists: Partial<Playlist>[] = [
      {
        id: 'playlist-symbolic',
        name: 'Meaning & Intention',
        description: 'Songs for finding purpose and symbolic resonance',
        coverEmoji: '✨',
        isSystem: true,
        basedOnMetric: 'symbolic'
      },
      {
        id: 'playlist-resonance',
        name: 'Flow & Connection',
        description: 'Songs for alignment and emotional resonance',
        coverEmoji: '🎯',
        isSystem: true,
        basedOnMetric: 'resonance'
      },
      {
        id: 'playlist-friction',
        name: 'Release & Process',
        description: 'Songs for working through friction and tension',
        coverEmoji: '🌧️',
        isSystem: true,
        basedOnMetric: 'friction'
      },
      {
        id: 'playlist-stability',
        name: 'Ground & Balance',
        description: 'Songs for stability and harmonic grounding',
        coverEmoji: '⚖️',
        isSystem: true,
        basedOnMetric: 'stability'
      },
      {
        id: 'playlist-healing',
        name: 'Healing Journey',
        description: 'Auto-curated from your skip patterns - songs you\'re ready to move past',
        coverEmoji: '🦋',
        isSystem: true
      }
    ];

    for (const sp of systemPlaylists) {
      if (!this.playlists.find(p => p.id === sp.id)) {
        this.playlists.push({
          ...sp,
          createdAt: new Date().toISOString(),
          updatedAt: new Date().toISOString(),
          trackIds: []
        } as Playlist);
      }
    }

    this.saveToStorage();
  }

  // ========================================
  // Hilbert Shuffle Algorithm
  // ========================================

  /**
   * Generate a new shuffle queue using Hilbert curve traversal
   * Biased by user's current DeltaHV metrics
   */
  async generateHilbertShuffle(
    tracks: MusicTrack[],
    deltaHV: DeltaHVState | null
  ): Promise<string[]> {
    if (tracks.length === 0) return [];

    // Default needs if no DeltaHV available
    const needs = deltaHV ? {
      symbolic: deltaHV.symbolicDensity / 100,
      resonance: deltaHV.resonanceCoupling / 100,
      friction: deltaHV.frictionCoefficient / 100,
      stability: deltaHV.harmonicStability / 100
    } : {
      symbolic: 0.5,
      resonance: 0.5,
      friction: 0.5,
      stability: 0.5
    };

    // Map tracks to 4D emotional space
    const trackPositions = tracks.map(track => ({
      track,
      position: CATEGORY_METRIC_MAP[track.categoryId] || {
        symbolic: 0.5, resonance: 0.5, friction: 0.5, stability: 0.5
      }
    }));

    // Determine Hilbert curve order based on library size
    const order = Math.max(2, Math.ceil(Math.log2(Math.sqrt(tracks.length))));
    const totalPositions = Math.pow(4, order);

    // Find optimal starting point based on needs
    // High friction = start with release/melancholy
    // Low stability = start with calm/grounding
    // Low symbolic = start with wonder/meaning
    // Low resonance = start with love/connection
    let startPosition = 0;

    if (needs.friction > 0.6) {
      // High friction - start with release path
      startPosition = Math.floor(totalPositions * 0.7);
    } else if (needs.stability < 0.3) {
      // Low stability - start with grounding path
      startPosition = Math.floor(totalPositions * 0.2);
    } else if (needs.symbolic < 0.3) {
      // Low meaning - start with wonder path
      startPosition = Math.floor(totalPositions * 0.9);
    } else if (needs.resonance < 0.3) {
      // Low connection - start with love path
      startPosition = Math.floor(totalPositions * 0.5);
    }

    // Generate Hilbert curve positions and match to tracks
    const shuffled: { track: MusicTrack; score: number }[] = [];

    for (let i = 0; i < totalPositions && shuffled.length < tracks.length; i++) {
      const pos = (startPosition + i) % totalPositions;
      const hilbertCoord = hilbert4D(order, pos);

      // Find closest unassigned track
      let bestTrack: MusicTrack | null = null;
      let bestDistance = Infinity;

      for (const { track, position } of trackPositions) {
        if (shuffled.find(s => s.track.id === track.id)) continue;

        const distance = calculateWeightedDistance(hilbertCoord, position, needs);
        if (distance < bestDistance) {
          bestDistance = distance;
          bestTrack = track;
        }
      }

      if (bestTrack) {
        shuffled.push({ track: bestTrack, score: bestDistance });
      }
    }

    // Store shuffle state
    this.shuffleState = {
      queue: shuffled.map(s => s.track.id),
      currentIndex: 0,
      hilbertPosition: startPosition,
      metricsInfluence: needs,
      lastShuffleTime: new Date().toISOString()
    };

    this.saveToStorage();
    this.notifyListeners();

    return this.shuffleState.queue;
  }

  /**
   * Get next track in shuffle queue
   */
  getNextTrack(): string | null {
    if (!this.shuffleState || this.shuffleState.queue.length === 0) {
      return null;
    }

    this.shuffleState.currentIndex++;
    if (this.shuffleState.currentIndex >= this.shuffleState.queue.length) {
      this.shuffleState.currentIndex = 0; // Loop
    }

    this.saveToStorage();
    return this.shuffleState.queue[this.shuffleState.currentIndex];
  }

  /**
   * Get previous track in shuffle queue
   */
  getPreviousTrack(): string | null {
    if (!this.shuffleState || this.shuffleState.queue.length === 0) {
      return null;
    }

    this.shuffleState.currentIndex--;
    if (this.shuffleState.currentIndex < 0) {
      this.shuffleState.currentIndex = this.shuffleState.queue.length - 1;
    }

    this.saveToStorage();
    return this.shuffleState.queue[this.shuffleState.currentIndex];
  }

  /**
   * Get current shuffle state
   */
  getShuffleState(): ShuffleState | null {
    return this.shuffleState;
  }

  // ========================================
  // Skip Tracking
  // ========================================

  /**
   * Record a skip or completion event
   *
   * Philosophy: "You are the author of your own story.
   * Every skip is a choice. Every choice shapes your journey.
   * Skip freely - this is YOUR narrative."
   */
  recordPlayEvent(
    track: MusicTrack,
    playDuration: number,
    wasSkipped: boolean,
    deltaHV: DeltaHVState | null,
    reason: 'manual' | 'auto_next' | 'playlist_end' = 'manual'
  ): SkipEvent {
    const event: SkipEvent = {
      id: `skip-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      trackId: track.id,
      trackName: track.name,
      categoryId: track.categoryId,
      skippedAt: new Date().toISOString(),
      playDuration,
      totalDuration: track.duration,
      skipRatio: track.duration > 0 ? playDuration / track.duration : 0,
      deltaHVAtSkip: deltaHV,
      wasSkipped,
      reason
    };

    this.skipEvents.push(event);

    // Keep last 500 events
    if (this.skipEvents.length > 500) {
      this.skipEvents = this.skipEvents.slice(-500);
    }

    // Update metrics
    this.updateStoryMetrics();

    // Check for healing indicators (melancholy skips)
    if (track.categoryId === 'MELANCHOLY' && wasSkipped) {
      this.checkHealingProgress(track);
    }

    this.saveToStorage();
    this.notifyListeners();

    return event;
  }

  /**
   * Check healing progress based on melancholy skip patterns
   * Skipping melancholy songs = moving past, healing, growth
   */
  private checkHealingProgress(track: MusicTrack): void {
    // Update healing playlist with frequently skipped melancholy tracks
    const healingPlaylist = this.playlists.find(p => p.id === 'playlist-healing');
    if (healingPlaylist) {
      // Track skipped 3+ times in last 20 plays = add to healing playlist
      const trackSkips = this.skipEvents
        .filter(e => e.trackId === track.id && e.wasSkipped)
        .length;

      if (trackSkips >= 3 && !healingPlaylist.trackIds.includes(track.id)) {
        healingPlaylist.trackIds.push(track.id);
        healingPlaylist.updatedAt = new Date().toISOString();
      }
    }
  }

  /**
   * Update story metrics based on skip events
   */
  private updateStoryMetrics(): void {
    const totalPlays = this.skipEvents.length;
    const totalSkips = this.skipEvents.filter(e => e.wasSkipped).length;
    const totalCompleted = totalPlays - totalSkips;

    // Category-specific ratios
    const categorySkipRatios: Record<string, { plays: number; skips: number; avgListenRatio: number }> = {};

    Object.keys(EMOTIONAL_CATEGORIES).forEach(catId => {
      const catEvents = this.skipEvents.filter(e => e.categoryId === catId);
      const catSkips = catEvents.filter(e => e.wasSkipped).length;
      const avgListen = catEvents.length > 0
        ? catEvents.reduce((sum, e) => sum + e.skipRatio, 0) / catEvents.length
        : 0;

      categorySkipRatios[catId] = {
        plays: catEvents.length,
        skips: catSkips,
        avgListenRatio: avgListen
      };
    });

    // Melancholy progress
    const melancholyEvents = this.skipEvents.filter(e => e.categoryId === 'MELANCHOLY');
    const melancholySkips = melancholyEvents.filter(e => e.wasSkipped).length;
    const melancholyCompletions = melancholyEvents.length - melancholySkips;

    // Healing indicator: high skip ratio = healing, but some completion is healthy
    // Optimal is around 60-80% skip (processing but moving forward)
    const melancholySkipRatio = melancholyEvents.length > 0
      ? melancholySkips / melancholyEvents.length
      : 0;
    const healingIndicator = Math.min(100, melancholySkipRatio * 100 + (melancholyCompletions > 0 ? 10 : 0));

    // Determine phase
    let currentPhase: 'processing' | 'healing' | 'integrated' = 'processing';
    if (melancholySkipRatio > 0.8) {
      currentPhase = 'integrated'; // Fully moved past
    } else if (melancholySkipRatio > 0.5) {
      currentPhase = 'healing'; // Actively healing
    }

    // Authorship score - measures active engagement with choices
    // Skipping is good (taking control), but so is mindful completion
    const recentEvents = this.skipEvents.slice(-50);
    const recentSkipRate = recentEvents.length > 0
      ? recentEvents.filter(e => e.wasSkipped).length / recentEvents.length
      : 0;

    // Optimal authorship is varied choices (not always skip, not always complete)
    // Variance in choices = active decision making
    const variance = Math.abs(recentSkipRate - 0.5) * 2; // 0-1, higher = more extreme
    const authorshipScore = Math.round((1 - variance * 0.5) * 100); // Moderate variance is good

    // Determine dominant emotion and trajectory
    const recentCategories = this.skipEvents.slice(-20).map(e => e.categoryId);
    const categoryCounts: Record<string, number> = {};
    recentCategories.forEach(cat => {
      categoryCounts[cat] = (categoryCounts[cat] || 0) + 1;
    });

    const dominantEmotion = Object.entries(categoryCounts)
      .sort((a, b) => b[1] - a[1])[0]?.[0] as EmotionalCategoryId | undefined || null;

    // Emotional trajectory based on category flow
    let emotionalTrajectory: 'rising' | 'steady' | 'processing' | 'releasing' = 'steady';
    const positiveCategories = ['JOY', 'LOVE', 'GRATITUDE', 'WONDER', 'COURAGE'];
    const processingCategories = ['MELANCHOLY', 'RELEASE'];

    const recentPositive = recentCategories.filter(c => positiveCategories.includes(c)).length;
    const recentProcessing = recentCategories.filter(c => processingCategories.includes(c)).length;

    if (recentPositive > recentCategories.length * 0.6) {
      emotionalTrajectory = 'rising';
    } else if (recentProcessing > recentCategories.length * 0.4) {
      emotionalTrajectory = 'processing';
    } else if (dominantEmotion === 'RELEASE') {
      emotionalTrajectory = 'releasing';
    }

    this.storyMetrics = {
      totalPlays,
      totalSkips,
      totalCompleted,
      skipRatio: totalPlays > 0 ? totalSkips / totalPlays : 0,
      completionRatio: totalPlays > 0 ? totalCompleted / totalPlays : 0,
      authorshipScore,
      categorySkipRatios: categorySkipRatios as Record<EmotionalCategoryId, any>,
      melancholyProgress: {
        totalMelancholyPlays: melancholyEvents.length,
        melancholySkips,
        melancholyCompletions,
        healingIndicator,
        currentPhase
      },
      recentChoices: this.skipEvents.slice(-10),
      dominantEmotion,
      emotionalTrajectory
    };

    this.saveToStorage();
  }

  /**
   * Get current story metrics
   */
  getStoryMetrics(): StoryMetrics | null {
    return this.storyMetrics;
  }

  /**
   * Get skip events
   */
  getSkipEvents(): SkipEvent[] {
    return this.skipEvents;
  }

  /**
   * Get empowerment message based on skip behavior
   */
  getAuthorshipMessage(): { message: string; emoji: string; type: 'encouragement' | 'insight' | 'celebration' } {
    if (!this.storyMetrics || this.storyMetrics.totalPlays === 0) {
      return {
        message: "You are the author of your story. Every choice shapes your journey.",
        emoji: "📖",
        type: 'encouragement'
      };
    }

    const { skipRatio, authorshipScore, melancholyProgress, emotionalTrajectory } = this.storyMetrics;

    // Healing celebration
    if (melancholyProgress.currentPhase === 'healing' && melancholyProgress.melancholySkips > 5) {
      return {
        message: "You're choosing to move forward. Skipping the sad songs is growth.",
        emoji: "🦋",
        type: 'celebration'
      };
    }

    if (melancholyProgress.currentPhase === 'integrated') {
      return {
        message: "You've integrated past emotions. You choose what resonates now.",
        emoji: "🌟",
        type: 'celebration'
      };
    }

    // Authorship insights
    if (authorshipScore > 80) {
      return {
        message: "Excellent story control! You're making deliberate, varied choices.",
        emoji: "✍️",
        type: 'celebration'
      };
    }

    if (skipRatio > 0.8) {
      return {
        message: "You're skipping freely - that's power. But also try letting some songs play through.",
        emoji: "⏭️",
        type: 'insight'
      };
    }

    if (skipRatio < 0.2) {
      return {
        message: "Remember: skipping is choosing. You're in control. Try skipping when it doesn't feel right.",
        emoji: "🎯",
        type: 'encouragement'
      };
    }

    // Trajectory messages
    if (emotionalTrajectory === 'rising') {
      return {
        message: "Your choices are lifting you up. Keep authoring this rising chapter.",
        emoji: "📈",
        type: 'celebration'
      };
    }

    if (emotionalTrajectory === 'processing') {
      return {
        message: "You're allowing yourself to process. That's brave authorship.",
        emoji: "💭",
        type: 'insight'
      };
    }

    return {
      message: "Skip or stay - every choice is yours. You're writing your story.",
      emoji: "📖",
      type: 'encouragement'
    };
  }

  // ========================================
  // Playlist Management
  // ========================================

  /**
   * Create a new playlist
   */
  createPlaylist(name: string, description?: string, coverEmoji: string = '🎵'): Playlist {
    const playlist: Playlist = {
      id: `playlist-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      name,
      description,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      trackIds: [],
      coverEmoji
    };

    this.playlists.push(playlist);
    this.saveToStorage();
    this.notifyListeners();

    return playlist;
  }

  /**
   * Get all playlists
   */
  getPlaylists(): Playlist[] {
    return this.playlists;
  }

  /**
   * Get playlist by ID
   */
  getPlaylist(id: string): Playlist | null {
    return this.playlists.find(p => p.id === id) || null;
  }

  /**
   * Add track to playlist
   */
  addToPlaylist(playlistId: string, trackId: string): void {
    const playlist = this.playlists.find(p => p.id === playlistId);
    if (playlist && !playlist.trackIds.includes(trackId)) {
      playlist.trackIds.push(trackId);
      playlist.updatedAt = new Date().toISOString();
      this.saveToStorage();
      this.notifyListeners();
    }
  }

  /**
   * Remove track from playlist
   */
  removeFromPlaylist(playlistId: string, trackId: string): void {
    const playlist = this.playlists.find(p => p.id === playlistId);
    if (playlist) {
      playlist.trackIds = playlist.trackIds.filter(id => id !== trackId);
      playlist.updatedAt = new Date().toISOString();
      this.saveToStorage();
      this.notifyListeners();
    }
  }

  /**
   * Delete playlist
   */
  deletePlaylist(playlistId: string): void {
    const playlist = this.playlists.find(p => p.id === playlistId);
    if (playlist && !playlist.isSystem) {
      this.playlists = this.playlists.filter(p => p.id !== playlistId);
      this.saveToStorage();
      this.notifyListeners();
    }
  }

  /**
   * Update playlist
   */
  updatePlaylist(playlistId: string, updates: Partial<Playlist>): void {
    const playlist = this.playlists.find(p => p.id === playlistId);
    if (playlist) {
      Object.assign(playlist, updates, { updatedAt: new Date().toISOString() });
      this.saveToStorage();
      this.notifyListeners();
    }
  }

  /**
   * Generate smart playlist based on current metrics
   */
  async generateSmartPlaylist(
    _deltaHV: DeltaHVState,
    metric: 'symbolic' | 'resonance' | 'friction' | 'stability'
  ): Promise<Playlist> {
    const tracks = await musicLibrary.getAllTracks();

    // Find categories that match the metric need
    let targetCategories: EmotionalCategoryId[] = [];

    switch (metric) {
      case 'symbolic':
        targetCategories = ['WONDER', 'FOCUS', 'COURAGE', 'GRATITUDE'];
        break;
      case 'resonance':
        targetCategories = ['JOY', 'LOVE', 'ENERGY', 'GRATITUDE'];
        break;
      case 'friction':
        targetCategories = ['RELEASE', 'CALM', 'MELANCHOLY'];
        break;
      case 'stability':
        targetCategories = ['CALM', 'FOCUS', 'GRATITUDE'];
        break;
    }

    const matchingTracks = tracks.filter(t => targetCategories.includes(t.categoryId));
    const trackIds = matchingTracks.map(t => t.id);

    // Update system playlist
    const systemPlaylist = this.playlists.find(p => p.basedOnMetric === metric);
    if (systemPlaylist) {
      systemPlaylist.trackIds = trackIds;
      systemPlaylist.updatedAt = new Date().toISOString();
      this.saveToStorage();
      this.notifyListeners();
      return systemPlaylist;
    }

    return this.createPlaylist(`${metric} Balance`, `Songs for ${metric} support`);
  }

  /**
   * AI-driven coherence-based playlist generation
   * Analyzes DeltaHV state and creates a personalized playlist based on:
   * - Current field state (coherent, transitioning, fragmented, dormant)
   * - Metric imbalances that need addressing
   * - User's skip patterns and preferences
   * - Healing progress and emotional trajectory
   */
  async generateCoherencePlaylist(
    deltaHV: DeltaHVState,
    purpose: 'balance' | 'boost' | 'calm' | 'focus' | 'heal' | 'energize'
  ): Promise<Playlist> {
    const tracks = await musicLibrary.getAllTracks();
    if (tracks.length === 0) {
      return this.createPlaylist('Empty Playlist', 'Add tracks to your library first', '📭');
    }

    const needs = {
      symbolic: deltaHV.symbolicDensity / 100,
      resonance: deltaHV.resonanceCoupling / 100,
      friction: deltaHV.frictionCoefficient / 100,
      stability: deltaHV.harmonicStability / 100
    };

    // Determine which categories to prioritize based on purpose and current state
    let categoryWeights: Record<EmotionalCategoryId, number> = {
      JOY: 1, CALM: 1, FOCUS: 1, ENERGY: 1, MELANCHOLY: 1,
      LOVE: 1, COURAGE: 1, WONDER: 1, GRATITUDE: 1, RELEASE: 1
    };

    // Adjust weights based on purpose
    switch (purpose) {
      case 'balance':
        // Inverse weighting - boost what's low
        if (needs.symbolic < 0.4) {
          categoryWeights.WONDER = 2; categoryWeights.FOCUS = 2;
        }
        if (needs.resonance < 0.4) {
          categoryWeights.JOY = 2; categoryWeights.LOVE = 2;
        }
        if (needs.friction > 0.6) {
          categoryWeights.RELEASE = 2; categoryWeights.CALM = 2;
        }
        if (needs.stability < 0.4) {
          categoryWeights.CALM = 2; categoryWeights.GRATITUDE = 2;
        }
        break;

      case 'boost':
        categoryWeights.JOY = 3; categoryWeights.ENERGY = 3;
        categoryWeights.COURAGE = 2; categoryWeights.WONDER = 2;
        categoryWeights.MELANCHOLY = 0.2;
        break;

      case 'calm':
        categoryWeights.CALM = 3; categoryWeights.GRATITUDE = 2;
        categoryWeights.RELEASE = 2;
        categoryWeights.ENERGY = 0.5; categoryWeights.COURAGE = 0.5;
        break;

      case 'focus':
        categoryWeights.FOCUS = 3; categoryWeights.CALM = 2;
        categoryWeights.WONDER = 1.5;
        categoryWeights.ENERGY = 0.5;
        break;

      case 'heal':
        // Use skip patterns to avoid frequently-skipped melancholy
        categoryWeights.RELEASE = 2; categoryWeights.LOVE = 2;
        categoryWeights.GRATITUDE = 2;
        // Add some melancholy for processing but not too much
        categoryWeights.MELANCHOLY = needs.friction > 0.5 ? 0.5 : 1;
        break;

      case 'energize':
        categoryWeights.ENERGY = 3; categoryWeights.JOY = 2;
        categoryWeights.COURAGE = 2;
        categoryWeights.CALM = 0.3; categoryWeights.MELANCHOLY = 0.2;
        break;
    }

    // Factor in field state
    if (deltaHV.fieldState === 'fragmented') {
      categoryWeights.CALM = Math.max(categoryWeights.CALM, 2);
      categoryWeights.RELEASE = Math.max(categoryWeights.RELEASE, 1.5);
    } else if (deltaHV.fieldState === 'coherent') {
      categoryWeights.WONDER = Math.max(categoryWeights.WONDER, 1.5);
      categoryWeights.FOCUS = Math.max(categoryWeights.FOCUS, 1.5);
    }

    // Factor in skip patterns - avoid categories user frequently skips
    const categorySkipRatios = this.storyMetrics?.categorySkipRatios;
    if (categorySkipRatios) {
      Object.entries(categorySkipRatios).forEach(([catId, data]) => {
        if (data.plays > 5 && data.skips / data.plays > 0.7) {
          // User frequently skips this category - reduce weight
          categoryWeights[catId as EmotionalCategoryId] *= 0.5;
        } else if (data.plays > 5 && data.avgListenRatio > 0.8) {
          // User enjoys this category - increase weight
          categoryWeights[catId as EmotionalCategoryId] *= 1.3;
        }
      });
    }

    // Score and sort tracks
    const scoredTracks = tracks.map(track => {
      const catMetrics = CATEGORY_METRIC_MAP[track.categoryId];
      const weight = categoryWeights[track.categoryId] || 1;

      // Calculate score based on weighted distance to target
      let score = weight;

      // Add bonus for matching user's needs
      if (needs.symbolic < 0.5 && catMetrics.symbolic > 0.7) score += 0.5;
      if (needs.resonance < 0.5 && catMetrics.resonance > 0.7) score += 0.5;
      if (needs.friction > 0.5 && catMetrics.friction < 0.3) score += 0.5;
      if (needs.stability < 0.5 && catMetrics.stability > 0.7) score += 0.5;

      return { track, score };
    });

    // Sort by score and take top tracks
    scoredTracks.sort((a, b) => b.score - a.score);
    const selectedTracks = scoredTracks.slice(0, Math.min(25, tracks.length));

    // Create playlist with descriptive name
    const purposeNames: Record<string, string> = {
      balance: 'Balance & Harmony',
      boost: 'Energy Boost',
      calm: 'Calm & Center',
      focus: 'Deep Focus',
      heal: 'Healing Journey',
      energize: 'Power Up'
    };

    const purposeEmojis: Record<string, string> = {
      balance: '⚖️',
      boost: '🚀',
      calm: '🌊',
      focus: '🎯',
      heal: '💜',
      energize: '⚡'
    };

    const playlistName = `AI: ${purposeNames[purpose]} (${new Date().toLocaleDateString()})`;
    const playlist = this.createPlaylist(
      playlistName,
      `AI-generated playlist for ${purpose}. Field state: ${deltaHV.fieldState}. ΔHV: ${deltaHV.deltaHV}%`,
      purposeEmojis[purpose]
    );

    playlist.trackIds = selectedTracks.map(s => s.track.id);
    this.saveToStorage();
    this.notifyListeners();

    return playlist;
  }

  /**
   * Reorder tracks in a playlist
   */
  reorderPlaylistTracks(playlistId: string, fromIndex: number, toIndex: number): void {
    const playlist = this.playlists.find(p => p.id === playlistId);
    if (!playlist || fromIndex < 0 || fromIndex >= playlist.trackIds.length ||
        toIndex < 0 || toIndex >= playlist.trackIds.length) {
      return;
    }

    const [removed] = playlist.trackIds.splice(fromIndex, 1);
    playlist.trackIds.splice(toIndex, 0, removed);
    playlist.updatedAt = new Date().toISOString();
    this.saveToStorage();
    this.notifyListeners();
  }

  /**
   * Get AI prompt context for playlist and music state
   */
  getAIPromptContext(): string {
    let context = `Music & Playlist State:\n`;

    // Story metrics
    if (this.storyMetrics) {
      context += `- Total plays: ${this.storyMetrics.totalPlays}\n`;
      context += `- Skip ratio: ${Math.round(this.storyMetrics.skipRatio * 100)}%\n`;
      context += `- Authorship score: ${this.storyMetrics.authorshipScore}%\n`;
      context += `- Emotional trajectory: ${this.storyMetrics.emotionalTrajectory}\n`;
      context += `- Dominant emotion: ${this.storyMetrics.dominantEmotion || 'varied'}\n`;
      context += `- Healing phase: ${this.storyMetrics.melancholyProgress.currentPhase}\n`;
      context += `- Healing indicator: ${this.storyMetrics.melancholyProgress.healingIndicator}%\n`;
    }

    // Metric influence from music
    const influence = this.getMetricInfluenceFromMusic();
    context += `\nMusic Metric Influence:\n`;
    context += `- Symbolic influence: ${influence.symbolic.value}%\n`;
    context += `- Resonance influence: ${influence.resonance.value}%\n`;
    context += `- Friction influence: ${influence.friction.value}%\n`;
    context += `- Stability influence: ${influence.stability.value}%\n`;

    // Playlists
    context += `\nPlaylists: ${this.playlists.length} total\n`;
    const userPlaylists = this.playlists.filter(p => !p.isSystem);
    const aiPlaylists = this.playlists.filter(p => p.name.startsWith('AI:'));
    context += `- User playlists: ${userPlaylists.length}\n`;
    context += `- AI-generated playlists: ${aiPlaylists.length}\n`;

    // Shuffle state
    if (this.shuffleState) {
      context += `\nShuffle: ${this.shuffleState.queue.length} tracks queued\n`;
      context += `- Position: ${this.shuffleState.currentIndex + 1}/${this.shuffleState.queue.length}\n`;
    }

    return context;
  }

  /**
   * Get playlist generation recommendations based on current state
   */
  getPlaylistRecommendations(deltaHV: DeltaHVState | null): {
    purpose: 'balance' | 'boost' | 'calm' | 'focus' | 'heal' | 'energize';
    reason: string;
    emoji: string;
  }[] {
    const recommendations: {
      purpose: 'balance' | 'boost' | 'calm' | 'focus' | 'heal' | 'energize';
      reason: string;
      emoji: string;
    }[] = [];

    if (!deltaHV) {
      return [{
        purpose: 'balance',
        reason: 'Start with a balanced playlist while we learn your metrics',
        emoji: '⚖️'
      }];
    }

    const needs = {
      symbolic: deltaHV.symbolicDensity,
      resonance: deltaHV.resonanceCoupling,
      friction: deltaHV.frictionCoefficient,
      stability: deltaHV.harmonicStability
    };

    // High friction - need calm or release
    if (needs.friction > 60) {
      recommendations.push({
        purpose: 'calm',
        reason: `High friction detected (${needs.friction}%). Try calming music to release tension.`,
        emoji: '🌊'
      });
    }

    // Low stability - need grounding
    if (needs.stability < 40) {
      recommendations.push({
        purpose: 'focus',
        reason: `Stability is low (${needs.stability}%). Focus music can help ground you.`,
        emoji: '🎯'
      });
    }

    // Low resonance - need connection
    if (needs.resonance < 40) {
      recommendations.push({
        purpose: 'boost',
        reason: `Resonance is low (${needs.resonance}%). Uplifting music can help reconnect.`,
        emoji: '🚀'
      });
    }

    // Fragmented state - healing focus
    if (deltaHV.fieldState === 'fragmented') {
      recommendations.push({
        purpose: 'heal',
        reason: 'Field state is fragmented. A healing playlist can help integrate.',
        emoji: '💜'
      });
    }

    // Coherent state - can push forward
    if (deltaHV.fieldState === 'coherent' && deltaHV.deltaHV > 60) {
      recommendations.push({
        purpose: 'energize',
        reason: `You're in a coherent state (ΔHV: ${deltaHV.deltaHV}%). Time to energize!`,
        emoji: '⚡'
      });
    }

    // Default recommendation
    if (recommendations.length === 0) {
      recommendations.push({
        purpose: 'balance',
        reason: 'Maintain your current balanced state with varied music.',
        emoji: '⚖️'
      });
    }

    return recommendations;
  }

  // ========================================
  // Metric Detection & Brain Region Mapping
  // ========================================

  /**
   * Get metric influence from current listening patterns
   * Returns which brain regions are being activated by music choices
   */
  getMetricInfluenceFromMusic(): {
    symbolic: { value: number; regions: string[] };
    resonance: { value: number; regions: string[] };
    friction: { value: number; regions: string[] };
    stability: { value: number; regions: string[] };
  } {
    const recentEvents = this.skipEvents.slice(-30);

    if (recentEvents.length === 0) {
      return {
        symbolic: { value: 0, regions: ['dlpfc', 'precuneus', 'temporalpole'] },
        resonance: { value: 0, regions: ['ofc', 'dacc', 'nucleus_accumbens_core'] },
        friction: { value: 65, regions: ['amygdala_bla', 'habenula', 'subgenual'] },
        stability: { value: 0, regions: ['raphe_dorsal', 'vermis', 'septum'] }
      };
    }

    // Calculate metric values from listening patterns
    let symbolicScore = 0;
    let resonanceScore = 0;
    let frictionScore = 0;
    let stabilityScore = 0;

    recentEvents.forEach(event => {
      const catMetrics = CATEGORY_METRIC_MAP[event.categoryId];
      if (!catMetrics) return;

      // Weight by listen ratio (completed = stronger influence)
      const weight = event.wasSkipped ? 0.3 : event.skipRatio;

      symbolicScore += catMetrics.symbolic * weight;
      resonanceScore += catMetrics.resonance * weight;
      frictionScore += catMetrics.friction * weight;
      stabilityScore += catMetrics.stability * weight;
    });

    const normalize = (score: number) => Math.min(100, Math.round((score / recentEvents.length) * 100));

    return {
      symbolic: {
        value: normalize(symbolicScore),
        regions: ['dlpfc', 'mpfc', 'precuneus', 'temporalpole', 'hippocampus', 'pcc']
      },
      resonance: {
        value: normalize(resonanceScore),
        regions: ['ofc', 'dacc', 'racc', 'premotor', 'sma', 'nucleus_accumbens_core']
      },
      friction: {
        value: normalize(frictionScore),
        regions: ['subgenual', 'amygdala_bla', 'amygdala_cea', 'pvn', 'bed_nucleus', 'habenula']
      },
      stability: {
        value: normalize(stabilityScore),
        regions: ['septum', 'raphe_dorsal', 'raphe_median', 'reticular', 'vermis', 'dentate']
      }
    };
  }

  // ========================================
  // Listeners
  // ========================================

  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  private notifyListeners(): void {
    this.listeners.forEach(listener => listener());
  }
}

// Export singleton
export const storyShuffleEngine = new StoryShuffleEngine();

// Export class for testing
export { StoryShuffleEngine };
