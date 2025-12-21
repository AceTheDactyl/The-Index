/**
 * Rhythm State Engine for Rhythm Pulse
 *
 * Implements Phase 2: Rhythm Pulse State Synchronization
 *
 * This module manages the real-time rhythm state machine that aligns the user's
 * internal focus state with their calendar context. It defines discrete states
 * (FOCUS, OPEN, REFLECTIVE) that change based on time of day and calendar events.
 *
 * The state machine:
 * - FOCUS: During scheduled deep-work blocks or focus events
 * - OPEN: During unscheduled time or non-focus events (flexible state)
 * - REFLECTIVE: During designated reflection periods (evening/early morning)
 *
 * All state transitions are logged to the audit trail for transparency.
 */

import { auditLog, logRhythmStateChange } from './auditLog';

/**
 * Rhythm state constants - the three core states of daily rhythm
 */
export const RhythmState = {
  FOCUS: 'FOCUS',
  OPEN: 'OPEN',
  REFLECTIVE: 'REFLECTIVE'
} as const;

export type RhythmState = typeof RhythmState[keyof typeof RhythmState];

/**
 * State metadata providing context about the current state
 */
export interface RhythmStateInfo {
  state: RhythmState;
  since: string; // ISO timestamp of when this state began
  trigger: string; // What triggered this state (event name, time-based, etc.)
  eventId?: string; // If triggered by a calendar event
  eventTitle?: string;
  waveId?: string; // Current wave context
  suggestedActions: string[]; // Context-appropriate suggestions
  uiHint: {
    icon: string;
    color: string;
    label: string;
    description: string;
  };
}

/**
 * Configuration for the rhythm state engine
 */
export interface RhythmStateConfig {
  // Tags/keywords that indicate a focus event
  focusTags: string[];
  // Google Calendar color IDs that indicate focus
  focusColorIds: string[];
  // Hour ranges for reflective time (24h format)
  reflectiveHours: { start: number; end: number };
  // Whether to auto-transition or require manual confirmation
  autoTransition: boolean;
  // Minimum minutes before auto-transitioning back from FOCUS
  focusCooldown: number;
}

/**
 * Check-in/Beat structure (matches App.tsx)
 */
export interface CheckIn {
  id: string;
  category: string;
  task: string;
  waveId?: string;
  slot: string;
  loggedAt: string;
  note?: string;
  done: boolean;
  isAnchor?: boolean;
}

/**
 * Wave structure (matches App.tsx)
 */
export interface Wave {
  id: string;
  name: string;
  description: string;
  color: string;
  startHour: number;
  endHour: number;
}

/**
 * Rhythm profile structure
 */
export interface RhythmProfile {
  waves: Wave[];
  setupComplete: boolean;
  deviationDay?: number;
  wakeTime?: { hours: number; minutes: number };
}

// Default configuration
const DEFAULT_CONFIG: RhythmStateConfig = {
  focusTags: ['Focus', 'Deep Work', 'Flow', 'Coding', 'Writing', 'Study', '🟢', '🎯', '💻'],
  focusColorIds: ['7', '11'], // Cyan and Red in Google Calendar
  reflectiveHours: { start: 21, end: 7 }, // 9pm to 7am
  autoTransition: true,
  focusCooldown: 5 // 5 minutes before auto-exiting focus
};

/**
 * Rhythm State Engine
 *
 * Manages state transitions and provides context-aware information
 * about the user's current rhythm state.
 */
export class RhythmStateEngine {
  private currentState: RhythmState = RhythmState.OPEN;
  private stateInfo: RhythmStateInfo;
  private config: RhythmStateConfig;
  private checkIns: CheckIn[] = [];
  private rhythmProfile: RhythmProfile;
  private listeners: Set<(info: RhythmStateInfo) => void> = new Set();
  private updateInterval: ReturnType<typeof setInterval> | null = null;

  constructor(
    rhythmProfile: RhythmProfile,
    config: Partial<RhythmStateConfig> = {}
  ) {
    this.rhythmProfile = rhythmProfile;
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.stateInfo = this.createStateInfo(RhythmState.OPEN, 'Initial state');

    // Log initialization
    auditLog.addEntry(
      'RHYTHM_STATE_INIT',
      'info',
      'Rhythm state engine initialized',
      { initialState: this.currentState, config: this.config }
    );
  }

  /**
   * Create state info object with UI hints and suggestions
   */
  private createStateInfo(
    state: RhythmState,
    trigger: string,
    eventId?: string,
    eventTitle?: string,
    waveId?: string
  ): RhythmStateInfo {
    const uiHints: Record<RhythmState, RhythmStateInfo['uiHint']> = {
      [RhythmState.FOCUS]: {
        icon: '🎯',
        color: 'cyan',
        label: 'Focus Mode',
        description: 'Deep work in progress. Minimizing distractions.'
      },
      [RhythmState.OPEN]: {
        icon: '🌊',
        color: 'purple',
        label: 'Open Flow',
        description: 'Flexible time. Open to tasks and interactions.'
      },
      [RhythmState.REFLECTIVE]: {
        icon: '🌙',
        color: 'blue',
        label: 'Reflective',
        description: 'Time for rest, reflection, and integration.'
      }
    };

    const suggestions: Record<RhythmState, string[]> = {
      [RhythmState.FOCUS]: [
        'Silence notifications',
        'Set a clear intention',
        'Take short breaks every 25-50 min',
        'Defer non-urgent tasks'
      ],
      [RhythmState.OPEN]: [
        'Review your upcoming beats',
        'Tackle quick wins',
        'Connect with others',
        'Process inbox/messages'
      ],
      [RhythmState.REFLECTIVE]: [
        'Journal your thoughts',
        'Review today\'s accomplishments',
        'Set intentions for tomorrow',
        'Practice gratitude'
      ]
    };

    return {
      state,
      since: new Date().toISOString(),
      trigger,
      eventId,
      eventTitle,
      waveId,
      suggestedActions: suggestions[state],
      uiHint: uiHints[state]
    };
  }

  /**
   * Update check-ins data
   */
  updateCheckIns(checkIns: CheckIn[]): void {
    this.checkIns = checkIns;
  }

  /**
   * Update rhythm profile
   */
  updateProfile(profile: RhythmProfile): void {
    this.rhythmProfile = profile;
  }

  /**
   * Find an active beat/event at the given time
   */
  private findActiveBeat(at: Date): CheckIn | null {
    const atTime = at.getTime();

    // Find check-ins that cover this time
    // A beat "covers" a time if slot <= time < slot + 30min (default duration)
    for (const checkIn of this.checkIns) {
      const slotTime = new Date(checkIn.slot).getTime();
      const endTime = slotTime + 30 * 60 * 1000; // 30 min default duration

      if (atTime >= slotTime && atTime < endTime && !checkIn.done) {
        return checkIn;
      }
    }

    return null;
  }

  /**
   * Determine if a beat/event is a focus-type activity
   */
  private isFocusBeat(beat: CheckIn): boolean {
    // Check if task/category matches focus tags
    const textToCheck = `${beat.category} ${beat.task} ${beat.note || ''}`.toLowerCase();
    const hasFocusTag = this.config.focusTags.some(tag =>
      textToCheck.includes(tag.toLowerCase())
    );

    // Check for focus categories
    const focusCategories = ['Meditation', 'Workout', 'Focus'];
    const isFocusCategory = focusCategories.some(cat =>
      beat.category.toLowerCase().includes(cat.toLowerCase())
    );

    // Anchors are typically focus activities
    const isAnchor = beat.isAnchor === true;

    return hasFocusTag || isFocusCategory || isAnchor;
  }

  /**
   * Check if current time is in reflective hours
   */
  private isReflectiveTime(at: Date): boolean {
    const hour = at.getHours();
    const { start, end } = this.config.reflectiveHours;

    // Handle overnight range (e.g., 21-7 means 9pm to 7am)
    if (start > end) {
      return hour >= start || hour < end;
    }
    return hour >= start && hour < end;
  }

  /**
   * Get current wave based on time since wake
   */
  private getCurrentWave(at: Date): Wave | null {
    const wake = this.rhythmProfile.wakeTime ?? { hours: 8, minutes: 0 };
    const wakeToday = new Date(at);
    wakeToday.setHours(wake.hours, wake.minutes, 0, 0);

    const diffMs = at.getTime() - wakeToday.getTime();
    const hoursAwake = diffMs < 0 ? 0 : diffMs / (1000 * 60 * 60);

    return this.rhythmProfile.waves.find(w =>
      hoursAwake >= w.startHour && hoursAwake < w.endHour
    ) || null;
  }

  /**
   * Calculate the appropriate rhythm state for a given time
   */
  calculateState(at: Date = new Date()): {
    state: RhythmState;
    trigger: string;
    beat?: CheckIn;
    wave?: Wave;
  } {
    const currentWave = this.getCurrentWave(at);
    const activeBeat = this.findActiveBeat(at);

    // Priority 1: Active focus beat
    if (activeBeat && this.isFocusBeat(activeBeat)) {
      return {
        state: RhythmState.FOCUS,
        trigger: `Active focus beat: ${activeBeat.task}`,
        beat: activeBeat,
        wave: currentWave || undefined
      };
    }

    // Priority 2: Active non-focus beat (still engaged, but OPEN)
    if (activeBeat) {
      return {
        state: RhythmState.OPEN,
        trigger: `Active beat: ${activeBeat.task}`,
        beat: activeBeat,
        wave: currentWave || undefined
      };
    }

    // Priority 3: Reflective time (evening/early morning)
    if (this.isReflectiveTime(at)) {
      return {
        state: RhythmState.REFLECTIVE,
        trigger: 'Reflective hours',
        wave: currentWave || undefined
      };
    }

    // Default: OPEN state
    return {
      state: RhythmState.OPEN,
      trigger: 'No active events',
      wave: currentWave || undefined
    };
  }

  /**
   * Update the rhythm state based on current conditions
   * This is the main method to call periodically or on data changes
   */
  updateState(at: Date = new Date()): RhythmStateInfo {
    const { state: newState, trigger, beat, wave } = this.calculateState(at);
    const oldState = this.currentState;

    // Check if state changed
    if (newState !== oldState) {
      this.currentState = newState;
      this.stateInfo = this.createStateInfo(
        newState,
        trigger,
        beat?.id,
        beat?.task,
        wave?.id
      );

      // Log the state transition
      logRhythmStateChange(oldState, newState, trigger, {
        rhythmState: newState,
        waveId: wave?.id
      });

      // Notify listeners
      this.notifyListeners();
    } else {
      // Update wave context even if state didn't change
      if (wave && this.stateInfo.waveId !== wave.id) {
        this.stateInfo = {
          ...this.stateInfo,
          waveId: wave.id
        };
      }
    }

    return this.stateInfo;
  }

  /**
   * Manually set the rhythm state (user override)
   */
  setStateManually(state: RhythmState, reason: string = 'Manual override'): RhythmStateInfo {
    const oldState = this.currentState;

    if (state !== oldState) {
      this.currentState = state;
      this.stateInfo = this.createStateInfo(state, reason);

      // Log manual override
      auditLog.addEntry(
        'RHYTHM_STATE_CHANGE',
        'info',
        `Manual state change: ${oldState} → ${state}`,
        { oldState, newState: state, reason, manual: true },
        { rhythmState: state }
      );

      this.notifyListeners();
    }

    return this.stateInfo;
  }

  /**
   * Get current state info
   */
  getStateInfo(): RhythmStateInfo {
    return this.stateInfo;
  }

  /**
   * Get current state
   */
  getState(): RhythmState {
    return this.currentState;
  }

  /**
   * Subscribe to state changes
   */
  subscribe(listener: (info: RhythmStateInfo) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /**
   * Notify all listeners of state change
   */
  private notifyListeners(): void {
    this.listeners.forEach(listener => listener(this.stateInfo));
  }

  /**
   * Start automatic state updates on an interval
   */
  startAutoUpdate(intervalMs: number = 60000): void {
    this.stopAutoUpdate(); // Clear any existing interval

    this.updateInterval = setInterval(() => {
      this.updateState();
    }, intervalMs);

    auditLog.addEntry(
      'SYSTEM_INIT',
      'info',
      `Rhythm state auto-update started (${intervalMs}ms interval)`,
      { intervalMs }
    );
  }

  /**
   * Stop automatic state updates
   */
  stopAutoUpdate(): void {
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
      this.updateInterval = null;
    }
  }

  /**
   * Check if a given activity type should be deferred in current state
   */
  shouldDeferActivity(activityType: 'notification' | 'suggestion' | 'check-in'): boolean {
    if (this.currentState === RhythmState.FOCUS) {
      // In focus mode, defer non-critical activities
      return activityType === 'notification' || activityType === 'suggestion';
    }
    return false;
  }

  /**
   * Get contextual prompt for AI based on current state
   */
  getAIPromptContext(): string {
    const info = this.stateInfo;
    let prompt = `Current Rhythm State: ${info.state}\n`;
    prompt += `State Active Since: ${info.since}\n`;
    prompt += `Trigger: ${info.trigger}\n`;

    if (info.waveId) {
      const wave = this.rhythmProfile.waves.find(w => w.id === info.waveId);
      if (wave) {
        prompt += `Current Wave: ${wave.name} - ${wave.description}\n`;
      }
    }

    prompt += `\nState-specific guidance:\n`;

    switch (info.state) {
      case RhythmState.FOCUS:
        prompt += '- User is in deep focus. Minimize interruptions.\n';
        prompt += '- Keep responses brief and task-oriented.\n';
        prompt += '- Defer non-urgent suggestions.\n';
        break;
      case RhythmState.OPEN:
        prompt += '- User is in flexible time. Open to suggestions.\n';
        prompt += '- Can engage with longer discussions or planning.\n';
        prompt += '- Good time for task review and organization.\n';
        break;
      case RhythmState.REFLECTIVE:
        prompt += '- User is in reflective mode. Encourage introspection.\n';
        prompt += '- Support journaling and gratitude practices.\n';
        prompt += '- Help with daily review and tomorrow planning.\n';
        break;
    }

    return prompt;
  }

  /**
   * Clean up resources
   */
  destroy(): void {
    this.stopAutoUpdate();
    this.listeners.clear();
  }
}

/**
 * Factory function to create a rhythm state engine
 */
export function createRhythmStateEngine(
  rhythmProfile: RhythmProfile,
  config?: Partial<RhythmStateConfig>
): RhythmStateEngine {
  return new RhythmStateEngine(rhythmProfile, config);
}

export default RhythmStateEngine;
