/* INTEGRITY_METADATA
 * Date: 2025-12-23
 * Status: ⚠️ NEEDS REVIEW
 * Severity: MEDIUM RISK
 * Risk Types: unverified_math
 */


/**
 * Music Library Service
 *
 * Manages music files for meditation sessions with emotional categorization.
 * Uses IndexedDB for storing audio files (localStorage is too small for .wav files).
 *
 * Features:
 * - Emotional spectrum folders (categories)
 * - Import/export .wav files
 * - Session tracking and history
 * - Integration with meditation beats
 */

import { auditLog } from './auditLog';

/**
 * Emotional categories for music organization
 */
export const EMOTIONAL_CATEGORIES = {
  JOY: { id: 'joy', name: 'Joy', color: '#fbbf24', icon: '😊', description: 'Uplifting, happy, celebratory' },
  CALM: { id: 'calm', name: 'Calm', color: '#22d3ee', icon: '😌', description: 'Peaceful, serene, relaxing' },
  FOCUS: { id: 'focus', name: 'Focus', color: '#a855f7', icon: '🎯', description: 'Concentrated, clear, productive' },
  ENERGY: { id: 'energy', name: 'Energy', color: '#f97316', icon: '⚡', description: 'Energizing, motivating, powerful' },
  MELANCHOLY: { id: 'melancholy', name: 'Melancholy', color: '#3b82f6', icon: '🌧️', description: 'Reflective, wistful, contemplative' },
  LOVE: { id: 'love', name: 'Love', color: '#ec4899', icon: '💗', description: 'Warm, tender, compassionate' },
  COURAGE: { id: 'courage', name: 'Courage', color: '#ef4444', icon: '🦁', description: 'Bold, brave, empowering' },
  WONDER: { id: 'wonder', name: 'Wonder', color: '#8b5cf6', icon: '✨', description: 'Mystical, awe-inspiring, transcendent' },
  GRATITUDE: { id: 'gratitude', name: 'Gratitude', color: '#22c55e', icon: '🙏', description: 'Thankful, appreciative, blessed' },
  RELEASE: { id: 'release', name: 'Release', color: '#64748b', icon: '🍃', description: 'Letting go, cleansing, cathartic' }
} as const;

export type EmotionalCategoryId = keyof typeof EMOTIONAL_CATEGORIES;
export type EmotionalCategory = typeof EMOTIONAL_CATEGORIES[EmotionalCategoryId];

/**
 * Music track metadata
 */
export interface MusicTrack {
  id: string;
  name: string;
  fileName: string;
  categoryId: EmotionalCategoryId;
  duration: number; // in seconds
  fileSize: number; // in bytes
  mimeType: string;
  addedAt: string;
  playCount: number;
  lastPlayedAt?: string;
  tags?: string[];
  notes?: string;
}

/**
 * Music session record
 */
export interface MusicSession {
  id: string;
  trackId: string;
  trackName: string;
  categoryId: EmotionalCategoryId;
  startedAt: string;
  endedAt?: string;
  duration: number; // actual listened duration in seconds
  desiredEmotion: EmotionalCategoryId;
  linkedBeatId?: string; // if played during a beat/check-in
  linkedBeatType?: string;
  completed: boolean;
  // Post-session emotional coherence data
  postSession?: {
    experiencedEmotion: EmotionalCategoryId;
    intensityRating: number; // 1-5
    coherenceMatch: boolean; // did desired match experienced?
    notes?: string;
    moodBefore?: number; // 1-5
    moodAfter?: number; // 1-5
  };
}

/**
 * Daily music preference
 */
export interface DailyMusicPreference {
  date: string;
  selectedCategoryId: EmotionalCategoryId;
  selectedTrackId?: string;
  reason?: string;
}

/**
 * IndexedDB configuration
 */
const DB_NAME = 'pulse-music-library';
const DB_VERSION = 1;
const STORES = {
  TRACKS: 'tracks',
  AUDIO_FILES: 'audioFiles',
  SESSIONS: 'sessions',
  PREFERENCES: 'preferences'
};

/**
 * Music Library Service Class
 */
class MusicLibraryService {
  private db: IDBDatabase | null = null;
  private initialized = false;
  private currentAudio: HTMLAudioElement | null = null;
  private currentSession: MusicSession | null = null;
  private listeners: Set<() => void> = new Set();
  private playbackListeners: Set<(state: { isPlaying: boolean; trackId: string | null }) => void> = new Set();
  private currentTrackId: string | null = null;
  private playbackStartTime: number = 0;
  // Mutex-like flag to prevent race conditions when multiple play requests come in
  private isPlaybackTransitioning: boolean = false;

  /**
   * Initialize IndexedDB
   */
  async initialize(): Promise<boolean> {
    if (this.initialized) return true;

    return new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, DB_VERSION);

      request.onerror = () => {
        console.error('Failed to open music library database:', request.error);
        auditLog.addEntry('ERROR', 'error', 'Failed to initialize music library', {
          error: request.error?.message
        });
        reject(request.error);
      };

      request.onsuccess = () => {
        this.db = request.result;
        this.initialized = true;
        auditLog.addEntry('SYSTEM_INIT', 'info', 'Music library initialized');
        resolve(true);
      };

      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;

        // Tracks metadata store
        if (!db.objectStoreNames.contains(STORES.TRACKS)) {
          const tracksStore = db.createObjectStore(STORES.TRACKS, { keyPath: 'id' });
          tracksStore.createIndex('categoryId', 'categoryId', { unique: false });
          tracksStore.createIndex('name', 'name', { unique: false });
        }

        // Audio files store (binary data)
        if (!db.objectStoreNames.contains(STORES.AUDIO_FILES)) {
          db.createObjectStore(STORES.AUDIO_FILES, { keyPath: 'id' });
        }

        // Sessions store
        if (!db.objectStoreNames.contains(STORES.SESSIONS)) {
          const sessionsStore = db.createObjectStore(STORES.SESSIONS, { keyPath: 'id' });
          sessionsStore.createIndex('trackId', 'trackId', { unique: false });
          sessionsStore.createIndex('categoryId', 'categoryId', { unique: false });
          sessionsStore.createIndex('startedAt', 'startedAt', { unique: false });
        }

        // Preferences store
        if (!db.objectStoreNames.contains(STORES.PREFERENCES)) {
          const prefsStore = db.createObjectStore(STORES.PREFERENCES, { keyPath: 'date' });
          prefsStore.createIndex('categoryId', 'selectedCategoryId', { unique: false });
        }
      };
    });
  }

  /**
   * Ensure database is initialized
   */
  private async ensureDb(): Promise<IDBDatabase> {
    if (!this.db) {
      await this.initialize();
    }
    if (!this.db) {
      throw new Error('Music library database not available');
    }
    return this.db;
  }

  // ========================================
  // Track Management
  // ========================================

  /**
   * Import a music file
   */
  async importTrack(
    file: File,
    categoryId: EmotionalCategoryId,
    name?: string,
    tags?: string[]
  ): Promise<MusicTrack> {
    await this.ensureDb();

    // Validate file type
    if (!file.type.includes('audio')) {
      throw new Error('Invalid file type. Please upload an audio file.');
    }

    // Read file as ArrayBuffer
    const arrayBuffer = await file.arrayBuffer();

    // Get audio duration
    const duration = await this.getAudioDuration(file);

    const trackId = `track-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

    const track: MusicTrack = {
      id: trackId,
      name: name || file.name.replace(/\.[^/.]+$/, ''),
      fileName: file.name,
      categoryId,
      duration,
      fileSize: file.size,
      mimeType: file.type,
      addedAt: new Date().toISOString(),
      playCount: 0,
      tags
    };

    // Store track metadata
    await this.storeItem(STORES.TRACKS, track);

    // Store audio file
    await this.storeItem(STORES.AUDIO_FILES, {
      id: trackId,
      data: arrayBuffer,
      mimeType: file.type
    });

    auditLog.addEntry('CHECKIN_CREATED', 'success', `Imported music track: ${track.name}`, {
      trackId,
      categoryId,
      duration,
      fileSize: file.size
    });

    this.notifyListeners();
    return track;
  }

  /**
   * Get audio duration from file
   */
  private getAudioDuration(file: File): Promise<number> {
    return new Promise((resolve) => {
      const audio = new Audio();
      audio.addEventListener('loadedmetadata', () => {
        resolve(audio.duration);
        URL.revokeObjectURL(audio.src);
      });
      audio.addEventListener('error', () => {
        resolve(0);
        URL.revokeObjectURL(audio.src);
      });
      audio.src = URL.createObjectURL(file);
    });
  }

  /**
   * Delete a track
   */
  async deleteTrack(trackId: string): Promise<void> {
    await this.ensureDb();

    // Delete metadata
    await this.deleteItem(STORES.TRACKS, trackId);

    // Delete audio file
    await this.deleteItem(STORES.AUDIO_FILES, trackId);

    auditLog.addEntry('MUSIC_TRACK_DELETED', 'info', 'Deleted music track', { trackId });
    this.notifyListeners();
  }

  /**
   * Get all tracks
   */
  async getAllTracks(): Promise<MusicTrack[]> {
    await this.ensureDb();
    return this.getAllItems<MusicTrack>(STORES.TRACKS);
  }

  /**
   * Get tracks by category
   */
  async getTracksByCategory(categoryId: EmotionalCategoryId): Promise<MusicTrack[]> {
    await this.ensureDb();
    return this.getItemsByIndex<MusicTrack>(STORES.TRACKS, 'categoryId', categoryId);
  }

  /**
   * Get a single track
   */
  async getTrack(trackId: string): Promise<MusicTrack | null> {
    return this.getItem<MusicTrack>(STORES.TRACKS, trackId);
  }

  /**
   * Update track metadata
   */
  async updateTrack(trackId: string, updates: Partial<MusicTrack>): Promise<void> {
    const track = await this.getTrack(trackId);
    if (!track) throw new Error('Track not found');

    const updatedTrack = { ...track, ...updates, id: trackId };
    await this.storeItem(STORES.TRACKS, updatedTrack);
    this.notifyListeners();
  }

  /**
   * Get audio file as Blob URL for playback
   */
  async getAudioUrl(trackId: string): Promise<string | null> {
    const audioFile = await this.getItem<{ id: string; data: ArrayBuffer; mimeType: string }>(
      STORES.AUDIO_FILES,
      trackId
    );

    if (!audioFile) return null;

    const blob = new Blob([audioFile.data], { type: audioFile.mimeType });
    return URL.createObjectURL(blob);
  }

  /**
   * Export a track (download)
   */
  async exportTrack(trackId: string): Promise<void> {
    const track = await this.getTrack(trackId);
    const audioFile = await this.getItem<{ id: string; data: ArrayBuffer; mimeType: string }>(
      STORES.AUDIO_FILES,
      trackId
    );

    if (!track || !audioFile) {
      throw new Error('Track not found');
    }

    const blob = new Blob([audioFile.data], { type: audioFile.mimeType });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = track.fileName;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    auditLog.addEntry('DATA_SAVED', 'info', `Exported music track: ${track.name}`, { trackId });
  }

  /**
   * Export all tracks from a category
   */
  async exportCategory(categoryId: EmotionalCategoryId): Promise<void> {
    const tracks = await this.getTracksByCategory(categoryId);

    for (const track of tracks) {
      await this.exportTrack(track.id);
    }
  }

  // ========================================
  // Playback
  // ========================================

  /**
   * Play a track
   * Uses a mutex to prevent race conditions when multiple components try to play
   */
  async playTrack(
    trackId: string,
    desiredEmotion: EmotionalCategoryId,
    linkedBeatId?: string,
    linkedBeatType?: string
  ): Promise<MusicSession | null> {
    // Prevent race conditions - if already transitioning, wait briefly
    if (this.isPlaybackTransitioning) {
      console.log('Playback transition in progress, waiting...');
      await new Promise(resolve => setTimeout(resolve, 100));
      if (this.isPlaybackTransitioning) {
        console.log('Still transitioning, aborting new playback request');
        return null;
      }
    }

    this.isPlaybackTransitioning = true;

    try {
      const track = await this.getTrack(trackId);
      if (!track) {
        this.isPlaybackTransitioning = false;
        return null;
      }

      const audioUrl = await this.getAudioUrl(trackId);
      if (!audioUrl) {
        this.isPlaybackTransitioning = false;
        return null;
      }

      // Stop current playback if any - CRITICAL: prevents two songs playing
      await this.stopPlayback();

      // Create audio element
      this.currentAudio = new Audio(audioUrl);
      this.currentTrackId = trackId;
      this.playbackStartTime = Date.now();

      // Create session
      const session: MusicSession = {
        id: `session-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        trackId,
        trackName: track.name,
        categoryId: track.categoryId,
        startedAt: new Date().toISOString(),
        duration: 0,
        desiredEmotion,
        linkedBeatId,
        linkedBeatType,
        completed: false
      };

      this.currentSession = session;

      // Add event listeners for state synchronization
      this.currentAudio.addEventListener('ended', () => {
        this.handlePlaybackEnd(this.playbackStartTime, true);
        this.notifyPlaybackState(false);
      });

      this.currentAudio.addEventListener('pause', () => {
        if (this.currentSession && !this.currentSession.completed) {
          this.currentSession.duration = (Date.now() - this.playbackStartTime) / 1000;
        }
        this.notifyPlaybackState(false);
      });

      this.currentAudio.addEventListener('play', () => {
        this.notifyPlaybackState(true);
      });

      // Handle errors gracefully
      this.currentAudio.addEventListener('error', (e) => {
        console.error('Audio playback error:', e);
        this.notifyPlaybackState(false);
      });

      // Start playback
      await this.currentAudio.play();

      // Notify that playback started
      this.notifyPlaybackState(true);

      // Update play count
      await this.updateTrack(trackId, {
        playCount: track.playCount + 1,
        lastPlayedAt: new Date().toISOString()
      });

      auditLog.addEntry('MUSIC_SESSION_STARTED', 'info', `Started music session: ${track.name}`, {
        sessionId: session.id,
        trackId,
        categoryId: track.categoryId,
        desiredEmotion
      });

      this.notifyListeners();
      return session;
    } catch (error) {
      console.error('Failed to play track:', error);
      this.currentTrackId = null;
      this.notifyPlaybackState(false);
      return null;
    } finally {
      // Always release the mutex
      this.isPlaybackTransitioning = false;
    }
  }

  /**
   * Handle playback end
   */
  private async handlePlaybackEnd(startTime: number, completed: boolean): Promise<void> {
    if (!this.currentSession) return;

    this.currentSession.duration = (Date.now() - startTime) / 1000;
    this.currentSession.endedAt = new Date().toISOString();
    this.currentSession.completed = completed;

    // Store session
    await this.storeItem(STORES.SESSIONS, this.currentSession);

    auditLog.addEntry(
      completed ? 'MUSIC_SESSION_COMPLETED' : 'MUSIC_SESSION_SKIPPED',
      completed ? 'success' : 'info',
      `Music session ${completed ? 'completed' : 'stopped'}: ${this.currentSession.trackName}`,
      {
        sessionId: this.currentSession.id,
        duration: this.currentSession.duration,
        categoryId: this.currentSession.categoryId
      }
    );

    this.notifyListeners();
  }

  /**
   * Stop current playback - completely stops and cleans up
   */
  async stopPlayback(): Promise<MusicSession | null> {
    if (this.currentAudio) {
      const startTime = this.currentSession
        ? new Date(this.currentSession.startedAt).getTime()
        : Date.now();

      // Remove event listeners before stopping to prevent duplicate events
      this.currentAudio.onended = null;
      this.currentAudio.onpause = null;
      this.currentAudio.onplay = null;
      this.currentAudio.onerror = null;

      this.currentAudio.pause();
      await this.handlePlaybackEnd(startTime, false);

      URL.revokeObjectURL(this.currentAudio.src);
      this.currentAudio = null;
      this.currentTrackId = null;

      const session = this.currentSession;
      this.currentSession = null;

      this.notifyPlaybackState(false);
      return session;
    }
    return null;
  }

  /**
   * Pause playback - keeps track loaded for resuming
   */
  pausePlayback(): void {
    if (this.currentAudio && !this.currentAudio.paused) {
      this.currentAudio.pause();
      // Note: pause event listener will call notifyPlaybackState(false)
    }
  }

  /**
   * Resume playback - continues from where it left off
   */
  async resumePlayback(): Promise<boolean> {
    if (this.currentAudio && this.currentAudio.paused) {
      try {
        await this.currentAudio.play();
        // Note: play event listener will call notifyPlaybackState(true)
        return true;
      } catch (error) {
        console.error('Failed to resume playback:', error);
        return false;
      }
    }
    return false;
  }

  /**
   * Check if currently playing
   */
  isCurrentlyPlaying(): boolean {
    return this.currentAudio !== null && !this.currentAudio.paused;
  }

  /**
   * Get current track ID
   */
  getCurrentTrackId(): string | null {
    return this.currentTrackId;
  }

  /**
   * Get current playback state
   */
  getPlaybackState(): {
    isPlaying: boolean;
    currentSession: MusicSession | null;
    currentTime: number;
    duration: number;
  } {
    return {
      isPlaying: this.currentAudio ? !this.currentAudio.paused : false,
      currentSession: this.currentSession,
      currentTime: this.currentAudio?.currentTime || 0,
      duration: this.currentAudio?.duration || 0
    };
  }

  /**
   * Set playback position
   */
  seekTo(time: number): void {
    if (this.currentAudio) {
      this.currentAudio.currentTime = time;
    }
  }

  /**
   * Set volume (0-1)
   */
  setVolume(volume: number): void {
    if (this.currentAudio) {
      this.currentAudio.volume = Math.max(0, Math.min(1, volume));
    }
  }

  // ========================================
  // Session & Post-Session
  // ========================================

  /**
   * Add post-session emotional coherence data
   */
  async addPostSessionData(
    sessionId: string,
    data: MusicSession['postSession']
  ): Promise<void> {
    const session = await this.getItem<MusicSession>(STORES.SESSIONS, sessionId);
    if (!session) throw new Error('Session not found');

    // Calculate coherence match
    const coherenceMatch = data?.experiencedEmotion === session.desiredEmotion;

    const updatedSession: MusicSession = {
      ...session,
      postSession: {
        ...data!,
        coherenceMatch
      }
    };

    await this.storeItem(STORES.SESSIONS, updatedSession);

    auditLog.addEntry('PROFILE_UPDATED', 'info', 'Added post-session emotional data', {
      sessionId,
      desiredEmotion: session.desiredEmotion,
      experiencedEmotion: data?.experiencedEmotion,
      coherenceMatch,
      moodChange: data?.moodAfter && data?.moodBefore
        ? data.moodAfter - data.moodBefore
        : undefined
    });

    this.notifyListeners();
  }

  /**
   * Get all sessions
   */
  async getAllSessions(): Promise<MusicSession[]> {
    return this.getAllItems<MusicSession>(STORES.SESSIONS);
  }

  /**
   * Get sessions by date range
   */
  async getSessionsByDateRange(startDate: Date, endDate: Date): Promise<MusicSession[]> {
    const allSessions = await this.getAllSessions();
    return allSessions.filter(session => {
      const sessionDate = new Date(session.startedAt);
      return sessionDate >= startDate && sessionDate <= endDate;
    });
  }

  /**
   * Get current session (if any)
   */
  getCurrentSession(): MusicSession | null {
    return this.currentSession;
  }

  // ========================================
  // Daily Preferences
  // ========================================

  /**
   * Set today's music preference
   */
  async setDailyPreference(
    categoryId: EmotionalCategoryId,
    trackId?: string,
    reason?: string
  ): Promise<void> {
    const today = new Date().toISOString().split('T')[0];

    const preference: DailyMusicPreference = {
      date: today,
      selectedCategoryId: categoryId,
      selectedTrackId: trackId,
      reason
    };

    await this.storeItem(STORES.PREFERENCES, preference);

    auditLog.addEntry('PROFILE_UPDATED', 'info', 'Set daily music preference', {
      categoryId,
      trackId,
      date: today
    });

    this.notifyListeners();
  }

  /**
   * Get today's music preference
   */
  async getTodayPreference(): Promise<DailyMusicPreference | null> {
    const today = new Date().toISOString().split('T')[0];
    return this.getItem<DailyMusicPreference>(STORES.PREFERENCES, today);
  }

  /**
   * Get preference history
   */
  async getPreferenceHistory(days: number = 30): Promise<DailyMusicPreference[]> {
    const allPrefs = await this.getAllItems<DailyMusicPreference>(STORES.PREFERENCES);
    const cutoff = new Date();
    cutoff.setDate(cutoff.getDate() - days);

    return allPrefs
      .filter(p => new Date(p.date) >= cutoff)
      .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
  }

  // ========================================
  // Analytics
  // ========================================

  /**
   * Get emotional coherence statistics
   */
  async getCoherenceStats(days: number = 30): Promise<{
    totalSessions: number;
    completedSessions: number;
    coherentSessions: number;
    coherenceRate: number;
    avgMoodImprovement: number;
    categoryBreakdown: Record<EmotionalCategoryId, {
      sessions: number;
      coherentSessions: number;
      avgMoodChange: number;
    }>;
  }> {
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - days);
    const sessions = await this.getSessionsByDateRange(startDate, new Date());

    const completedSessions = sessions.filter(s => s.completed);
    const sessionsWithPostData = completedSessions.filter(s => s.postSession);
    const coherentSessions = sessionsWithPostData.filter(s => s.postSession?.coherenceMatch);

    // Calculate mood improvements
    const moodChanges = sessionsWithPostData
      .filter(s => s.postSession?.moodBefore && s.postSession?.moodAfter)
      .map(s => s.postSession!.moodAfter! - s.postSession!.moodBefore!);

    const avgMoodImprovement = moodChanges.length > 0
      ? moodChanges.reduce((a, b) => a + b, 0) / moodChanges.length
      : 0;

    // Category breakdown
    const categoryBreakdown: Record<string, { sessions: number; coherentSessions: number; avgMoodChange: number }> = {};

    Object.keys(EMOTIONAL_CATEGORIES).forEach(key => {
      const catSessions = completedSessions.filter(s => s.categoryId === key);
      const catCoherent = catSessions.filter(s => s.postSession?.coherenceMatch);
      const catMoodChanges = catSessions
        .filter(s => s.postSession?.moodBefore && s.postSession?.moodAfter)
        .map(s => s.postSession!.moodAfter! - s.postSession!.moodBefore!);

      categoryBreakdown[key] = {
        sessions: catSessions.length,
        coherentSessions: catCoherent.length,
        avgMoodChange: catMoodChanges.length > 0
          ? catMoodChanges.reduce((a, b) => a + b, 0) / catMoodChanges.length
          : 0
      };
    });

    return {
      totalSessions: sessions.length,
      completedSessions: completedSessions.length,
      coherentSessions: coherentSessions.length,
      coherenceRate: sessionsWithPostData.length > 0
        ? (coherentSessions.length / sessionsWithPostData.length) * 100
        : 0,
      avgMoodImprovement,
      categoryBreakdown: categoryBreakdown as Record<EmotionalCategoryId, any>
    };
  }

  /**
   * Get listening time by category
   */
  async getListeningTimeByCategory(days: number = 30): Promise<Record<EmotionalCategoryId, number>> {
    const startDate = new Date();
    startDate.setDate(startDate.getDate() - days);
    const sessions = await this.getSessionsByDateRange(startDate, new Date());

    const result: Record<string, number> = {};
    Object.keys(EMOTIONAL_CATEGORIES).forEach(key => {
      result[key] = 0;
    });

    sessions.forEach(session => {
      result[session.categoryId] = (result[session.categoryId] || 0) + session.duration;
    });

    return result as Record<EmotionalCategoryId, number>;
  }

  // ========================================
  // IndexedDB Helpers
  // ========================================

  private storeItem<T>(storeName: string, item: T): Promise<void> {
    return new Promise(async (resolve, reject) => {
      const db = await this.ensureDb();
      const transaction = db.transaction(storeName, 'readwrite');
      const store = transaction.objectStore(storeName);
      const request = store.put(item);

      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }

  private getItem<T>(storeName: string, key: string): Promise<T | null> {
    return new Promise(async (resolve, reject) => {
      const db = await this.ensureDb();
      const transaction = db.transaction(storeName, 'readonly');
      const store = transaction.objectStore(storeName);
      const request = store.get(key);

      request.onsuccess = () => resolve(request.result || null);
      request.onerror = () => reject(request.error);
    });
  }

  private getAllItems<T>(storeName: string): Promise<T[]> {
    return new Promise(async (resolve, reject) => {
      const db = await this.ensureDb();
      const transaction = db.transaction(storeName, 'readonly');
      const store = transaction.objectStore(storeName);
      const request = store.getAll();

      request.onsuccess = () => resolve(request.result || []);
      request.onerror = () => reject(request.error);
    });
  }

  private getItemsByIndex<T>(storeName: string, indexName: string, value: string): Promise<T[]> {
    return new Promise(async (resolve, reject) => {
      const db = await this.ensureDb();
      const transaction = db.transaction(storeName, 'readonly');
      const store = transaction.objectStore(storeName);
      const index = store.index(indexName);
      const request = index.getAll(value);

      request.onsuccess = () => resolve(request.result || []);
      request.onerror = () => reject(request.error);
    });
  }

  private deleteItem(storeName: string, key: string): Promise<void> {
    return new Promise(async (resolve, reject) => {
      const db = await this.ensureDb();
      const transaction = db.transaction(storeName, 'readwrite');
      const store = transaction.objectStore(storeName);
      const request = store.delete(key);

      request.onsuccess = () => resolve();
      request.onerror = () => reject(request.error);
    });
  }

  // ========================================
  // Listeners
  // ========================================

  subscribe(listener: () => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  /**
   * Subscribe to playback state changes
   * Called whenever play/pause state changes
   */
  subscribeToPlayback(listener: (state: { isPlaying: boolean; trackId: string | null }) => void): () => void {
    this.playbackListeners.add(listener);
    return () => this.playbackListeners.delete(listener);
  }

  private notifyListeners(): void {
    this.listeners.forEach(listener => listener());
  }

  private notifyPlaybackState(isPlaying: boolean): void {
    const state = { isPlaying, trackId: this.currentTrackId };
    this.playbackListeners.forEach(listener => listener(state));
  }
}

// Export singleton instance
export const musicLibrary = new MusicLibraryService();

// Export class for testing
export { MusicLibraryService };
