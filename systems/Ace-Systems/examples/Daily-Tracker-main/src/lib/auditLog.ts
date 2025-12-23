/* INTEGRITY_METADATA
 * Date: 2025-12-23
 * Status: ⚠️ NEEDS REVIEW
 * Severity: MEDIUM RISK
 * Risk Types: low_integrity, unverified_math
 */


/**
 * Audit Log System for Rhythm Pulse
 *
 * Provides comprehensive logging of all state changes, AI actions, calendar operations,
 * and system events. Creates a transparent trail for debugging, analysis, and user insight.
 *
 * Key features:
 * - Timestamped entries with action types and details
 * - Persistent storage with localStorage fallback
 * - Filterable and queryable log history
 * - Supports multiple action categories
 */

// Storage abstraction (matches App.tsx pattern)
const storageGet = async (key: string): Promise<string | null> => {
  try {
    if (window.storage && typeof window.storage.get === 'function') {
      const res = await window.storage.get(key);
      return res?.value ?? null;
    }
    return localStorage.getItem(key);
  } catch {
    return null;
  }
};

const storageSet = async (key: string, value: string) => {
  try {
    if (window.storage && typeof window.storage.set === 'function') {
      await window.storage.set(key, value);
    } else {
      localStorage.setItem(key, value);
    }
  } catch {}
};

/**
 * Action types for categorizing audit entries
 */
export type AuditActionType =
  // Rhythm State Actions
  | 'RHYTHM_STATE_CHANGE'
  | 'RHYTHM_STATE_INIT'
  // Check-in Actions
  | 'CHECKIN_CREATED'
  | 'CHECKIN_COMPLETED'
  | 'CHECKIN_DELETED'
  | 'CHECKIN_SNOOZED'
  | 'ANCHOR_SET'
  | 'ANCHORS_COPIED'
  // Journal Actions
  | 'JOURNAL_CREATED'
  | 'JOURNAL_DELETED'
  // Calendar Actions
  | 'GCAL_SYNC_START'
  | 'GCAL_SYNC_SUCCESS'
  | 'GCAL_SYNC_FAILED'
  | 'GCAL_IMPORT'
  | 'GCAL_BULK_UPLOAD'
  // ΔHV Metrics Actions
  | 'DELTAHV_CALCULATED'
  | 'DELTAHV_STATE_CHANGE'
  // AI/Planner Actions
  | 'AI_SUGGESTION'
  | 'AI_INTERVENTION'
  | 'AUTO_SCHEDULE'
  | 'AUTO_BREAK_INSERTED'
  // Music Library Actions
  | 'MUSIC_TRACK_IMPORTED'
  | 'MUSIC_TRACK_DELETED'
  | 'MUSIC_SESSION_STARTED'
  | 'MUSIC_SESSION_COMPLETED'
  | 'MUSIC_SESSION_SKIPPED'
  // System Actions
  | 'SYSTEM_INIT'
  | 'PROFILE_UPDATED'
  | 'DATA_LOADED'
  | 'DATA_SAVED'
  | 'ERROR';

/**
 * Severity levels for log entries
 */
export type AuditSeverity = 'info' | 'warning' | 'error' | 'success';

/**
 * Single audit log entry
 */
export interface AuditEntry {
  id: string;
  timestamp: string;
  action: AuditActionType;
  severity: AuditSeverity;
  message: string;
  details?: Record<string, any>;
  context?: {
    rhythmState?: string;
    deltaHV?: number;
    fieldState?: string;
    waveId?: string;
  };
}

/**
 * Filter options for querying audit logs
 */
export interface AuditFilter {
  actions?: AuditActionType[];
  severity?: AuditSeverity[];
  startDate?: Date;
  endDate?: Date;
  limit?: number;
}

/**
 * Audit Log Manager
 *
 * Singleton-style class that manages the audit trail for the entire application.
 */
class AuditLogManager {
  private entries: AuditEntry[] = [];
  private maxEntries: number = 1000; // Keep last 1000 entries to prevent unbounded growth
  private storageKey: string = 'pulse-audit-log';
  private initialized: boolean = false;
  private listeners: Set<(entry: AuditEntry) => void> = new Set();

  /**
   * Initialize the audit log, loading any persisted entries
   */
  async initialize(): Promise<void> {
    if (this.initialized) return;

    try {
      const stored = await storageGet(this.storageKey);
      if (stored) {
        const parsed = JSON.parse(stored);
        if (Array.isArray(parsed)) {
          this.entries = parsed;
        }
      }
      this.initialized = true;
      this.addEntry('SYSTEM_INIT', 'info', 'Audit log initialized', {
        entriesLoaded: this.entries.length
      });
    } catch (error) {
      console.error('Failed to load audit log:', error);
      this.initialized = true;
      this.addEntry('ERROR', 'error', 'Failed to load audit log history', { error: String(error) });
    }
  }

  /**
   * Generate a unique ID for entries
   */
  private generateId(): string {
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Add a new entry to the audit log
   */
  addEntry(
    action: AuditActionType,
    severity: AuditSeverity,
    message: string,
    details?: Record<string, any>,
    context?: AuditEntry['context']
  ): AuditEntry {
    const entry: AuditEntry = {
      id: this.generateId(),
      timestamp: new Date().toISOString(),
      action,
      severity,
      message,
      details,
      context
    };

    this.entries.push(entry);

    // Trim old entries if exceeding max
    if (this.entries.length > this.maxEntries) {
      this.entries = this.entries.slice(-this.maxEntries);
    }

    // Persist asynchronously (don't block)
    this.persist();

    // Notify listeners
    this.listeners.forEach(listener => listener(entry));

    // Also log to console in development
    const consoleMethod = severity === 'error' ? 'error' : severity === 'warning' ? 'warn' : 'log';
    console[consoleMethod](`[Audit] ${action}: ${message}`, details || '');

    return entry;
  }

  /**
   * Persist entries to storage
   */
  private async persist(): Promise<void> {
    try {
      await storageSet(this.storageKey, JSON.stringify(this.entries));
    } catch (error) {
      console.error('Failed to persist audit log:', error);
    }
  }

  /**
   * Get all entries, optionally filtered
   */
  getEntries(filter?: AuditFilter): AuditEntry[] {
    let result = [...this.entries];

    if (filter) {
      if (filter.actions && filter.actions.length > 0) {
        result = result.filter(e => filter.actions!.includes(e.action));
      }

      if (filter.severity && filter.severity.length > 0) {
        result = result.filter(e => filter.severity!.includes(e.severity));
      }

      if (filter.startDate) {
        const startTime = filter.startDate.getTime();
        result = result.filter(e => new Date(e.timestamp).getTime() >= startTime);
      }

      if (filter.endDate) {
        const endTime = filter.endDate.getTime();
        result = result.filter(e => new Date(e.timestamp).getTime() <= endTime);
      }

      if (filter.limit && filter.limit > 0) {
        result = result.slice(-filter.limit);
      }
    }

    return result;
  }

  /**
   * Get entries for a specific date
   */
  getEntriesForDate(date: Date): AuditEntry[] {
    const startOfDay = new Date(date);
    startOfDay.setHours(0, 0, 0, 0);
    const endOfDay = new Date(date);
    endOfDay.setHours(23, 59, 59, 999);

    return this.getEntries({
      startDate: startOfDay,
      endDate: endOfDay
    });
  }

  /**
   * Get the most recent entries
   */
  getRecentEntries(count: number = 50): AuditEntry[] {
    return this.entries.slice(-count);
  }

  /**
   * Get entries by action type
   */
  getEntriesByAction(action: AuditActionType): AuditEntry[] {
    return this.entries.filter(e => e.action === action);
  }

  /**
   * Subscribe to new entries
   */
  subscribe(listener: (entry: AuditEntry) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /**
   * Clear all entries (with confirmation)
   */
  async clearAll(): Promise<void> {
    const count = this.entries.length;
    this.entries = [];
    await this.persist();
    this.addEntry('SYSTEM_INIT', 'warning', `Audit log cleared (${count} entries removed)`);
  }

  /**
   * Export entries as JSON
   */
  exportAsJSON(): string {
    return JSON.stringify(this.entries, null, 2);
  }

  /**
   * Get summary statistics
   */
  getStats(): {
    totalEntries: number;
    byAction: Record<string, number>;
    bySeverity: Record<string, number>;
    oldestEntry: string | null;
    newestEntry: string | null;
  } {
    const byAction: Record<string, number> = {};
    const bySeverity: Record<string, number> = {};

    this.entries.forEach(e => {
      byAction[e.action] = (byAction[e.action] || 0) + 1;
      bySeverity[e.severity] = (bySeverity[e.severity] || 0) + 1;
    });

    return {
      totalEntries: this.entries.length,
      byAction,
      bySeverity,
      oldestEntry: this.entries.length > 0 ? this.entries[0].timestamp : null,
      newestEntry: this.entries.length > 0 ? this.entries[this.entries.length - 1].timestamp : null
    };
  }
}

// Singleton instance
export const auditLog = new AuditLogManager();

// Convenience functions for common logging patterns
export const logRhythmStateChange = (
  oldState: string,
  newState: string,
  trigger: string,
  context?: AuditEntry['context']
) => {
  auditLog.addEntry(
    'RHYTHM_STATE_CHANGE',
    'info',
    `Rhythm state changed: ${oldState} → ${newState}`,
    { oldState, newState, trigger },
    context
  );
};

export const logCheckInCreated = (
  category: string,
  task: string,
  isAnchor: boolean,
  waveId?: string
) => {
  auditLog.addEntry(
    isAnchor ? 'ANCHOR_SET' : 'CHECKIN_CREATED',
    'success',
    `${isAnchor ? 'Anchor' : 'Check-in'} created: ${category} - ${task}`,
    { category, task, isAnchor, waveId },
    { waveId }
  );
};

export const logCheckInCompleted = (
  category: string,
  task: string,
  scheduledTime: string,
  completedTime: string
) => {
  const scheduled = new Date(scheduledTime).getTime();
  const completed = new Date(completedTime).getTime();
  const delayMinutes = Math.round((completed - scheduled) / 60000);

  auditLog.addEntry(
    'CHECKIN_COMPLETED',
    'success',
    `Completed: ${task}${delayMinutes > 15 ? ` (${delayMinutes}min late)` : ''}`,
    { category, task, scheduledTime, completedTime, delayMinutes }
  );
};

export const logDeltaHVCalculated = (
  deltaHV: number,
  fieldState: string,
  metrics: { S: number; R: number; F: number; H: number }
) => {
  auditLog.addEntry(
    'DELTAHV_CALCULATED',
    'info',
    `ΔHV calculated: ${deltaHV} (${fieldState})`,
    { deltaHV, fieldState, ...metrics },
    { deltaHV, fieldState }
  );
};

export const logGCalSync = (
  success: boolean,
  eventCount: number,
  error?: string
) => {
  auditLog.addEntry(
    success ? 'GCAL_SYNC_SUCCESS' : 'GCAL_SYNC_FAILED',
    success ? 'success' : 'error',
    success ? `Synced ${eventCount} event(s) to Google Calendar` : `Sync failed: ${error}`,
    { eventCount, error }
  );
};

export const logError = (
  message: string,
  error: any,
  context?: AuditEntry['context']
) => {
  auditLog.addEntry(
    'ERROR',
    'error',
    message,
    { error: error?.message || String(error), stack: error?.stack },
    context
  );
};

export default auditLog;
