import React, { useState, useEffect, useRef, useMemo } from 'react';
import {
  Dumbbell, Brain, Heart, Shield, NotebookPen, Download,
  Calendar, ChevronDown, ChevronUp, Pill, Plus, X, Volume2,
  CheckCircle2, Trash2, Loader2, Clock, Sparkles, Waves, Zap, Copy, Timer,
  Activity, AlertTriangle, FileText, Settings, Bell, BarChart3, Music, User, Disc3,
  Target, Palette
} from 'lucide-react';
import { GoogleCalendarService } from './lib/googleCalendar';
import { getDeltaHVState } from './lib/deltaHVEngine';
import type { DeltaHVState } from './lib/deltaHVEngine';
import { auditLog, logCheckInCreated, logCheckInCompleted, logDeltaHVCalculated } from './lib/auditLog';
import { RhythmState, createRhythmStateEngine } from './lib/rhythmStateEngine';
import type { RhythmStateEngine, RhythmStateInfo } from './lib/rhythmStateEngine';
import { createRhythmPlanner, DEFAULT_PREFERENCES } from './lib/rhythmPlanner';
import type { RhythmPlanner, PlannerSuggestion, PlannerPreferences, CalendarServiceInterface } from './lib/rhythmPlanner';
import { notificationService } from './lib/notificationService';
import type { NotificationPreferences, PermissionStatus } from './lib/notificationService';
import { musicLibrary, EMOTIONAL_CATEGORIES, type EmotionalCategoryId } from './lib/musicLibrary';
import { metricsHub, type EnhancedDeltaHVState } from './lib/metricsHub';
import { AnalyticsPage } from './components/AnalyticsPage';
import { MusicLibrary } from './components/MusicLibrary';
import { UserProfilePage } from './components/UserProfilePage';
import { RoadmapView } from './components/RoadmapView';
import { BrainRegionChallenge } from './components/BrainRegionChallenge';
import { DJTab } from './components/DJTab';
import { FocusWellnessTools } from './components/FocusWellnessTools';
import ChallengeHub from './components/ChallengeHub';
import CosmeticsInventory from './components/CosmeticsInventory';
import { challengeRewardService } from './lib/challengeRewardSystem';
import type { UserMetrics, CosmeticType } from './lib/challengeRewardSystem';
import { generateCosmeticCSS, getEquippedCssClasses } from './lib/cosmeticDefinitions';
import { userProfileService } from './lib/userProfile';
// MetricsDisplay available for use in future enhancements
// import { MetricsDisplay, InlineMetrics } from './components/MetricsDisplay';

// Storage safety shim
declare global {
  interface Window {
    storage?: {
      get: (k: string) => Promise<{ value: string } | null>;
      set: (k: string, v: string) => Promise<void>;
    };
  }
}

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

// Wave color class mapping for Tailwind safety
const waveColorClasses: Record<string, { bg: string; text: string; border: string; bgLight: string }> = {
  cyan: { bg: 'bg-cyan-900/40', text: 'text-cyan-300', border: 'border-cyan-500/30', bgLight: 'bg-cyan-950/20' },
  purple: { bg: 'bg-purple-900/40', text: 'text-purple-300', border: 'border-purple-500/30', bgLight: 'bg-purple-950/20' },
  blue: { bg: 'bg-blue-900/40', text: 'text-blue-300', border: 'border-blue-500/30', bgLight: 'bg-blue-950/20' },
  orange: { bg: 'bg-orange-900/40', text: 'text-orange-300', border: 'border-orange-500/30', bgLight: 'bg-orange-950/20' },
  gray: { bg: 'bg-gray-900/40', text: 'text-gray-300', border: 'border-gray-500/30', bgLight: 'bg-gray-950/20' }
};

// Types
interface Wave {
  id: string;
  name: string;
  description: string;
  color: string;
  startHour: number;
  endHour: number;
}

interface CheckIn {
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
  selectedTrackId?: string;
  selectedTrackName?: string;
  selectedTrackEmotion?: string;
}

interface JournalEntry {
  id: string;
  timestamp: string;
  content: string;
  waveId?: string;
}

interface RhythmProfile {
  waves: Wave[];
  setupComplete: boolean;
  deviationDay?: number;
  wakeTime?: { hours: number; minutes: number };
}

const DEFAULT_WAVES: Wave[] = [
  { id: 'focus', name: 'Focus Wave', description: 'Deep work, analytical tasks, high-stakes decisions', color: 'cyan', startHour: 0, endHour: 8 },
  { id: 'flow', name: 'Flow Wave', description: 'Creative work, brainstorming, social connection', color: 'purple', startHour: 9, endHour: 16 },
  { id: 'recovery', name: 'Recovery Wave', description: 'Rest, reflection, wind down', color: 'blue', startHour: 17, endHour: 24 }
];

// Utility functions
const pad2 = (n: number) => n.toString().padStart(2, '0');
const formatTime = (date: Date) => {
  const hours = date.getHours();
  const minutes = date.getMinutes();
  const ampm = hours >= 12 ? 'PM' : 'AM';
  const displayHours = hours % 12 || 12;
  return `${displayHours}:${minutes.toString().padStart(2, '0')} ${ampm}`;
};
const addMinutes = (date: Date, minutes: number) => new Date(date.getTime() + minutes * 60000);
const toDateInput = (d: Date) => `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())}`;
const toTimeInput = (d: Date) => `${pad2(d.getHours())}:${pad2(d.getMinutes())}`;
const sameDay = (a: Date, b: Date) => a.getFullYear() === b.getFullYear() && a.getMonth() === b.getMonth() && a.getDate() === b.getDate();

const SUBTASKS: Record<string, string[]> = {
  Workout: ['Upper Body', 'Lower Body', 'Full Body', 'Core', 'Mobility', 'Conditioning'],
  Moderation: ['Mod-GotGames', 'Mod-GazMash', 'Meeting GotGames', 'Meeting SACS', 'General'],
  Meditation: ['Breath', 'Focus', 'Open Monitoring', 'Somatic', 'Visualization', 'Frequency', 'Silence'],
  Emotion: ['Gratitude Check', 'Vibe Check', 'Reset Walk', 'Talk to a Friend', 'Field Read', 'Shadow Pattern Spotting', 'Safety Inventory']
};

export default function App() {
  const [checkIns, setCheckIns] = useState<CheckIn[]>([]);
  const [journals, setJournals] = useState<Record<string, JournalEntry[]>>({});
  const [rhythmProfile, setRhythmProfile] = useState<RhythmProfile>({
    waves: DEFAULT_WAVES,
    setupComplete: false
  });
  const [isLoading, setIsLoading] = useState(true);
  const [saveStatus, setSaveStatus] = useState<'idle' | 'saving' | 'saved' | 'error'>('idle');

  const [selectedDate, setSelectedDate] = useState<Date>(new Date());
  const [datePickerOpen, setDatePickerOpen] = useState(false);
  const [calendarOpen, setCalendarOpen] = useState(false);
  const [visibleMonth, setVisibleMonth] = useState<Date>(new Date(new Date().getFullYear(), new Date().getMonth(), 1));

  const [setupOpen, setSetupOpen] = useState(false);
  const [waveAnchorsOpen, setWaveAnchorsOpen] = useState(false);
  const [anchorsExpanded, setAnchorsExpanded] = useState(false);

  const [miniOpen, setMiniOpen] = useState(false);
  const [miniCategory, setMiniCategory] = useState<string>('Workout');
  const [miniTask, setMiniTask] = useState('');
  const [miniTime, setMiniTime] = useState(toTimeInput(addMinutes(new Date(), 30)));
  const [miniNote, setMiniNote] = useState('');
  const [miniWaveId, setMiniWaveId] = useState<string>('');
  const [miniSelectedTrack, setMiniSelectedTrack] = useState<{ id: string; name: string; emotion: string } | null>(null);
  const [availableTracks, setAvailableTracks] = useState<Array<{ id: string; name: string; categoryId: string }>>([]);

  const [newJournalOpen, setNewJournalOpen] = useState(false);
  const [newJournalContent, setNewJournalContent] = useState('');
  const [newJournalWave, setNewJournalWave] = useState<string>('');

  const [generalNoteOpen, setGeneralNoteOpen] = useState(false);
  const [generalNoteText, setGeneralNoteText] = useState('');
  const [generalNoteTime, setGeneralNoteTime] = useState(toTimeInput(addMinutes(new Date(), 30)));

  const [mixerOpen, setMixerOpen] = useState(false);
  // DJ tab state - keeping minimal state for cleanup
  // Music state is managed by DJTab component internally

  const [glyphCanvasOpen, setGlyphCanvasOpen] = useState(false);

  // Google Calendar integration
  const [gcalService, setGcalService] = useState<GoogleCalendarService | null>(null);
  const [gcalAuthed, setGcalAuthed] = useState(false);
  const [syncEnabled, setSyncEnabled] = useState(false);

  // DeltaHV Metrics State (Enhanced with metricsHub integration)
  const [deltaHVState, setDeltaHVState] = useState<DeltaHVState | null>(null);
  const [enhancedMetrics, setEnhancedMetrics] = useState<EnhancedDeltaHVState | null>(null);
  const [deltaHVExpanded, setDeltaHVExpanded] = useState(false);

  // Rhythm State Engine (Phase 2)
  const rhythmEngineRef = useRef<RhythmStateEngine | null>(null);
  const [rhythmStateInfo, setRhythmStateInfo] = useState<RhythmStateInfo | null>(null);

  // Audit Trail UI State (Phase 4)
  const [auditTrailOpen, setAuditTrailOpen] = useState(false);

  // Rhythm Planner (Phase 3 Complete)
  const plannerRef = useRef<RhythmPlanner | null>(null);
  const [plannerSuggestions, setPlannerSuggestions] = useState<PlannerSuggestion[]>([]);
  const [suggestionsExpanded, setSuggestionsExpanded] = useState(true);
  const [plannerPreferences, setPlannerPreferences] = useState<PlannerPreferences>(DEFAULT_PREFERENCES);
  const [plannerSettingsOpen, setPlannerSettingsOpen] = useState(false);

  // Notifications & Analytics
  const [notificationPermission, setNotificationPermission] = useState<PermissionStatus>(notificationService.getPermissionStatus());
  const [notificationPrefs, setNotificationPrefs] = useState<NotificationPreferences>(notificationService.getPreferences());
  const [notificationSettingsOpen, setNotificationSettingsOpen] = useState(false);

  // Navigation & Views
  const [currentView, setCurrentView] = useState<'dashboard' | 'analytics' | 'profile' | 'roadmap' | 'challenges' | 'cosmetics'>('dashboard');
  const [currentRoadmapDomain, setCurrentRoadmapDomain] = useState<string>('');

  // Challenge & Cosmetics System
  const [equippedCosmetics, setEquippedCosmetics] = useState<Record<CosmeticType, string | null>>({} as Record<CosmeticType, string | null>);
  const [cosmeticCssVars, setCosmeticCssVars] = useState<string>('');

  // Music Library
  const [musicLibraryOpen, setMusicLibraryOpen] = useState(false);
  const [dailyMusicEmotion, setDailyMusicEmotion] = useState<EmotionalCategoryId | null>(null);

  // Load data with safe storage
  useEffect(() => {
    const load = async () => {
      try {
        const checkInsData = await storageGet('pulse-check-ins');
        const journalsData = await storageGet('pulse-journals');
        const profileData = await storageGet('pulse-rhythm-profile');

        if (checkInsData) setCheckIns(JSON.parse(checkInsData));
        if (journalsData) setJournals(JSON.parse(journalsData));
        if (profileData) {
          setRhythmProfile(JSON.parse(profileData));
        } else {
          setSetupOpen(true);
        }
      } catch (err) {
        console.log('No existing data:', err);
        setSetupOpen(true);
      } finally {
        setIsLoading(false);
      }
    };
    load();
  }, []);

  // Save data with safe storage
  useEffect(() => {
    if (!isLoading) {
      const save = async () => {
        setSaveStatus('saving');
        try {
          await storageSet('pulse-check-ins', JSON.stringify(checkIns));
          await storageSet('pulse-journals', JSON.stringify(journals));
          await storageSet('pulse-rhythm-profile', JSON.stringify(rhythmProfile));
          setSaveStatus('saved');
          setTimeout(() => setSaveStatus('idle'), 1200);
        } catch (err) {
          console.error('Save error:', err);
          setSaveStatus('error');
          setTimeout(() => setSaveStatus('idle'), 2000);
        }
      };
      save();
    }
  }, [checkIns, journals, rhythmProfile, isLoading]);

  // Initialize Audit Log, Rhythm State Engine, Planner, and Music Library (Phase 2, 3, 4 & 5)
  useEffect(() => {
    const initSystems = async () => {
      // Initialize audit log
      await auditLog.initialize();

      // Initialize music library and load daily preference
      await musicLibrary.initialize();
      const dailyPref = await musicLibrary.getTodayPreference();
      if (dailyPref) {
        setDailyMusicEmotion(dailyPref.selectedCategoryId);
      }

      // Create rhythm state engine once profile is loaded
      if (rhythmProfile.setupComplete && !rhythmEngineRef.current) {
        rhythmEngineRef.current = createRhythmStateEngine(rhythmProfile);

        // Create rhythm planner
        plannerRef.current = createRhythmPlanner();
        plannerRef.current.updateWaves(rhythmProfile.waves);

        // Subscribe to planner suggestions
        plannerRef.current.subscribe((suggestions) => {
          setPlannerSuggestions(suggestions);
        });

        // Subscribe to planner preferences changes
        plannerRef.current.subscribeToPreferences((prefs) => {
          setPlannerPreferences(prefs);
        });

        // Subscribe to rhythm state changes and notify planner
        rhythmEngineRef.current.subscribe((info) => {
          setRhythmStateInfo(info);
          // Notify planner of state change
          if (plannerRef.current) {
            plannerRef.current.onRhythmStateChange(info.state, info.trigger);
          }
        });

        // Start auto-update (every 60 seconds)
        rhythmEngineRef.current.startAutoUpdate(60000);

        // Initial state calculation
        const initialInfo = rhythmEngineRef.current.updateState();
        setRhythmStateInfo(initialInfo);
      }
    };

    if (!isLoading) {
      initSystems();
    }

    // Cleanup on unmount
    return () => {
      if (rhythmEngineRef.current) {
        rhythmEngineRef.current.destroy();
      }
    };
  }, [isLoading, rhythmProfile.setupComplete]);

  // Update rhythm engine and planner when check-ins or profile changes
  useEffect(() => {
    if (rhythmEngineRef.current) {
      rhythmEngineRef.current.updateCheckIns(checkIns);
      rhythmEngineRef.current.updateProfile(rhythmProfile);
      // Recalculate state after data update
      const newInfo = rhythmEngineRef.current.updateState();
      setRhythmStateInfo(newInfo);
    }
    if (plannerRef.current) {
      plannerRef.current.updateCheckIns(checkIns);
      plannerRef.current.updateWaves(rhythmProfile.waves);
    }
  }, [checkIns, rhythmProfile]);

  // Challenge & Cosmetics System initialization
  useEffect(() => {
    const loadCosmetics = () => {
      const data = challengeRewardService.getData();
      setEquippedCosmetics(data.equippedCosmetics);
      // Generate CSS variables from equipped cosmetics
      const cssVars = generateCosmeticCSS(data.equippedCosmetics);
      setCosmeticCssVars(cssVars);
    };

    loadCosmetics();

    // Subscribe to cosmetic changes
    const unsubscribe = challengeRewardService.subscribe(() => {
      loadCosmetics();
    });

    // Send daily performance notification
    if (!isLoading && deltaHVState) {
      const userMetrics: UserMetrics = {
        checkInsToday: checkIns.filter(c => c.done && sameDay(new Date(c.loggedAt), new Date())).length,
        journalEntriesToday: (journals[toDateInput(new Date())] || []).length,
        currentStreak: challengeRewardService.getStats().currentStreak,
        deltaHV: {
          symbolic: deltaHVState.symbolicDensity,
          resonance: deltaHVState.resonanceCoupling,
          friction: deltaHVState.frictionCoefficient,
          stability: deltaHVState.harmonicStability
        },
        categoryCompletions: checkIns.reduce((acc, c) => {
          if (c.done) {
            acc[c.category] = (acc[c.category] || 0) + 1;
          }
          return acc;
        }, {} as Record<string, number>),
        totalCheckIns: checkIns.filter(c => c.done).length,
        totalJournalEntries: Object.values(journals).flat().length,
        averageDailyCheckIns: 5, // Placeholder
        weakCategories: [],
        strongCategories: []
      };

      challengeRewardService.sendDailyPerformanceNotification(userMetrics);
    }

    return unsubscribe;
  }, [isLoading, deltaHVState]);

  // Compute user metrics for challenge system
  const userMetrics = useMemo<UserMetrics>(() => {
    const todayKey = toDateInput(new Date());
    const todayCheckIns = checkIns.filter(c => c.done && sameDay(new Date(c.loggedAt), new Date()));

    const categoryCompletions = checkIns.reduce((acc, c) => {
      if (c.done) {
        acc[c.category] = (acc[c.category] || 0) + 1;
      }
      return acc;
    }, {} as Record<string, number>);

    // Find weak and strong categories
    const categoryEntries = Object.entries(categoryCompletions);
    const avgCount = categoryEntries.length > 0
      ? categoryEntries.reduce((sum, [, count]) => sum + count, 0) / categoryEntries.length
      : 0;

    const weakCategories = categoryEntries
      .filter(([, count]) => count < avgCount * 0.5)
      .map(([cat]) => cat);

    const strongCategories = categoryEntries
      .filter(([, count]) => count > avgCount * 1.5)
      .map(([cat]) => cat);

    return {
      checkInsToday: todayCheckIns.length,
      journalEntriesToday: (journals[todayKey] || []).length,
      currentStreak: challengeRewardService.getStats().currentStreak,
      deltaHV: {
        symbolic: deltaHVState?.symbolicDensity || 0,
        resonance: deltaHVState?.resonanceCoupling || 0,
        friction: deltaHVState?.frictionCoefficient || 0,
        stability: deltaHVState?.harmonicStability || 0
      },
      categoryCompletions,
      totalCheckIns: checkIns.filter(c => c.done).length,
      totalJournalEntries: Object.values(journals).flat().length,
      averageDailyCheckIns: todayCheckIns.length,
      weakCategories,
      strongCategories
    };
  }, [checkIns, journals, deltaHVState]);

  // Apply cosmetic CSS variables to document root
  useEffect(() => {
    if (cosmeticCssVars) {
      const root = document.documentElement;
      // Parse and apply CSS variables
      cosmeticCssVars.split('\n').forEach(line => {
        const match = line.match(/^(--[\w-]+):\s*(.+);?$/);
        if (match) {
          root.style.setProperty(match[1], match[2].replace(';', ''));
        }
      });
    }
  }, [cosmeticCssVars]);

  // Notification integration
  useEffect(() => {
    // Subscribe to permission changes
    const unsubPermission = notificationService.subscribeToPermission((status) => {
      setNotificationPermission(status);
    });

    // Subscribe to planner suggestions for notifications
    const handleSuggestion = (suggestion: PlannerSuggestion) => {
      if (!notificationPrefs.enabled) return;

      if (suggestion.type === 'BREAK_NEEDED' && notificationPrefs.breakReminders) {
        const focusDuration = suggestion.action?.payload?.focusDuration || 90;
        const breakDuration = suggestion.action?.payload?.durationMinutes || 15;
        notificationService.sendBreakReminder(focusDuration, breakDuration);
      }

      if (suggestion.type === 'ANCHOR_REMINDER' && notificationPrefs.anchorReminders) {
        const taskName = suggestion.description.match(/"([^"]+)"/)?.[1] || 'Anchor';
        const minutes = parseInt(suggestion.description.match(/(\d+) minutes/)?.[1] || '15');
        notificationService.sendAnchorReminder(taskName, minutes, suggestion.action?.payload?.anchorId);
      }

      if (suggestion.type === 'FRICTION_WARNING' && notificationPrefs.frictionWarnings) {
        const count = parseInt(suggestion.description.match(/(\d+) tasks/)?.[1] || '3');
        notificationService.sendFrictionWarning(count);
      }

      if (suggestion.type === 'AUTO_SCHEDULED' && notificationPrefs.autoScheduledAlerts) {
        const time = suggestion.action?.payload?.start
          ? new Date(suggestion.action.payload.start).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })
          : 'soon';
        notificationService.sendAutoScheduled(suggestion.title, time);
      }
    };

    // Connect to planner suggestions
    if (plannerRef.current) {
      const unsubPlanner = plannerRef.current.subscribe((suggestions) => {
        setPlannerSuggestions(suggestions);
        // Send notifications for new high-priority suggestions
        suggestions.forEach(s => {
          if (s.priority === 'high' && !s.dismissed) {
            handleSuggestion(s);
          }
        });
      });

      return () => {
        unsubPermission();
        unsubPlanner();
      };
    }

    return unsubPermission;
  }, [notificationPrefs]);

  // Calculate DeltaHV metrics as side-effect after every log/state update
  useEffect(() => {
    if (!isLoading && rhythmProfile.setupComplete) {
      const newDeltaHVState = getDeltaHVState(checkIns, journals, rhythmProfile, selectedDate);
      setDeltaHVState(newDeltaHVState);

      // Update metricsHub with all data sources for comprehensive metric calculation
      metricsHub.updateCheckIns(checkIns);
      metricsHub.updateJournals(journals);
      metricsHub.updateRhythmProfile(rhythmProfile);

      // Update enhanced metrics state from metricsHub
      setEnhancedMetrics(metricsHub.getState());

      // Log ΔHV calculation to audit trail
      if (newDeltaHVState) {
        logDeltaHVCalculated(
          newDeltaHVState.deltaHV,
          newDeltaHVState.fieldState,
          {
            S: newDeltaHVState.symbolicDensity,
            R: newDeltaHVState.resonanceCoupling,
            F: newDeltaHVState.frictionCoefficient,
            H: newDeltaHVState.harmonicStability
          }
        );

        // Record metrics snapshot for energy state and friction calculations
        const todayCheckIns = checkIns.filter(c => sameDay(new Date(c.loggedAt), new Date()));
        const completedToday = todayCheckIns.filter(c => c.done).length;
        const todayJournals = Object.values(journals).flat().filter(j => sameDay(new Date(j.timestamp), new Date()));
        userProfileService.recordMetricsSnapshot({
          deltaHV: newDeltaHVState.deltaHV,
          symbolicDensity: newDeltaHVState.symbolicDensity,
          resonanceCoupling: newDeltaHVState.resonanceCoupling,
          frictionCoefficient: newDeltaHVState.frictionCoefficient,
          harmonicStability: newDeltaHVState.harmonicStability,
          rhythmScore: newDeltaHVState.resonanceCoupling, // Use resonance as rhythm score
          completedBeats: completedToday,
          totalBeats: todayCheckIns.length,
          journalEntries: todayJournals.length,
          glyphsUsed: [],
          fieldState: newDeltaHVState.fieldState,
        });
      }
    }
  }, [checkIns, journals, rhythmProfile, selectedDate, isLoading, rhythmStateInfo]);

  useEffect(() => {
    const initGCal = async () => {
      const clientId = import.meta.env.VITE_GOOGLE_CLIENT_ID;
      const apiKey = import.meta.env.VITE_GOOGLE_API_KEY;

      if (!clientId || !apiKey || clientId === 'YOUR_CLIENT_ID_HERE' || apiKey === 'YOUR_API_KEY_HERE') {
        console.log('Google Calendar API credentials not configured');
        return;
      }

      const service = new GoogleCalendarService({
        apiKey: apiKey,
        clientId: clientId,
        calendarId: import.meta.env.VITE_GOOGLE_CALENDAR_ID || 'primary',
      });

      // Set up auth state change callback
      service.setAuthChangeCallback((isAuthenticated) => {
        setGcalAuthed(isAuthenticated);
        if (isAuthenticated) {
          setSyncEnabled(true);
        }
      });

      try {
        await service.initialize();
        setGcalService(service);

        // Check if token was restored during initialize
        if (service.isAuthenticated()) {
          setGcalAuthed(true);
          setSyncEnabled(true);
          console.log('Google Calendar: Auto-authenticated from stored token');
        }
      } catch (error) {
        console.error('Failed to initialize Google Calendar:', error);
      }
    };

    initGCal();
  }, []);

  // Connect planner to Google Calendar when authenticated (Phase 3 Complete)
  useEffect(() => {
    if (plannerRef.current && gcalService && gcalAuthed) {
      // Create a wrapper that adapts GoogleCalendarService to CalendarServiceInterface
      const calendarServiceAdapter: CalendarServiceInterface = {
        createEvent: async (event) => {
          const start = new Date(event.start);
          const end = new Date(event.end);
          return await gcalService.createEvent({
            summary: event.summary,
            description: event.description || '',
            start: start.toISOString(),
            end: end.toISOString(),
            colorId: event.colorId
          });
        },
        listEvents: async (timeMin, timeMax) => {
          return await gcalService.listEvents(timeMin, timeMax);
        },
        deleteEvent: async (eventId) => {
          await gcalService.deleteEvent(eventId);
        }
      };

      plannerRef.current.setCalendarService(calendarServiceAdapter);

      // Refresh calendar events for conflict detection
      plannerRef.current.refreshCalendarEvents();
    } else if (plannerRef.current && !gcalAuthed) {
      // Disconnect calendar service if not authenticated
      plannerRef.current.setCalendarService(null);
    }
  }, [gcalService, gcalAuthed]);

  // Load available tracks when mini scheduler opens
  useEffect(() => {
    const loadTracks = async () => {
      if (miniOpen) {
        const tracks = await musicLibrary.getAllTracks();
        setAvailableTracks(tracks.map(t => ({ id: t.id, name: t.name, categoryId: t.categoryId })));
      }
    };
    loadTracks();
  }, [miniOpen]);

  // Sync to Google Calendar
  const syncToGoogleCalendar = async (checkIn: CheckIn, forceSend: boolean = false) => {
    if (!gcalService || !gcalAuthed) {
      console.log('Cannot sync: Calendar service not available or not authenticated');
      return { success: false, error: 'Not authenticated' };
    }

    if (!forceSend && !syncEnabled) {
      console.log('Auto-sync disabled');
      return { success: false, error: 'Auto-sync disabled' };
    }

    try {
      const start = new Date(checkIn.slot);
      const end = new Date(start.getTime() + 30 * 60000); // 30 min default

      // Build comprehensive description
      const wave = rhythmProfile.waves.find(w => w.id === checkIn.waveId);
      const descriptionParts = [
        `📋 Category: ${checkIn.category}`,
        `🎯 Task: ${checkIn.task}`,
        wave ? `🌊 Wave: ${wave.name} (${wave.description})` : '',
        checkIn.note ? `📝 Note: ${checkIn.note}` : '',
        checkIn.isAnchor ? '⚡ Daily Anchor Beat' : '',
        checkIn.selectedTrackName ? `🎵 Music: "${checkIn.selectedTrackName}" (${checkIn.selectedTrackEmotion || 'Unspecified'})` : '',
        `🕐 Scheduled: ${formatTime(start)}`,
        `\n🔗 Synced from Pulse Check Rhythm`
      ].filter(Boolean).join('\n');

      console.log(`Syncing to Google Calendar: ${checkIn.category} - ${checkIn.task}`);

      const result = await gcalService.createEvent({
        summary: `${getCategoryIcon(checkIn.category)} ${checkIn.task}`,
        description: descriptionParts,
        start: start.toISOString(),
        end: end.toISOString(),
        colorId: getGoogleCalendarColor(checkIn.waveId),
      });

      console.log('Successfully synced to Google Calendar:', result);
      return { success: true, result };
    } catch (error: any) {
      console.error('Failed to sync to Google Calendar:', error);
      return { success: false, error: error.message || 'Unknown error' };
    }
  };

  const getCategoryIcon = (category: string): string => {
    const icons: Record<string, string> = {
      'Workout': '💪',
      'Moderation': '🛡️',
      'Meditation': '🧘',
      'Emotion': '💜',
      'Journal': '📓',
      'Med': '💊',
      'Anchor': '⚡',
      'General': '📌'
    };
    return icons[category] || '📌';
  };

  const getGoogleCalendarColor = (waveId?: string): string => {
    // Google Calendar color IDs
    const colorMap: Record<string, string> = {
      'cyan': '7',    // Cyan
      'purple': '3',  // Purple
      'blue': '9',    // Blue
      'orange': '6',  // Orange
    };

    const color = getWaveColor(waveId);
    return colorMap[color] || '1';
  };

  // Import events from Google Calendar
  const importFromGoogleCalendar = async (date: Date) => {
    if (!gcalService || !gcalAuthed) {
      console.log('Google Calendar not authenticated');
      return;
    }

    try {
      setSaveStatus('saving');

      // Get events for the selected day
      const dayStart = new Date(date);
      dayStart.setHours(0, 0, 0, 0);
      const dayEnd = new Date(date);
      dayEnd.setHours(23, 59, 59, 999);

      const events = await gcalService.listEvents(
        dayStart.toISOString(),
        dayEnd.toISOString()
      );

      // Convert Google Calendar events to CheckIns
      const newBeats: CheckIn[] = events.map((event: any) => {
        const startTime = event.start.dateTime ? new Date(event.start.dateTime) : new Date(event.start.date);

        // Try to match wave based on time
        const wake = rhythmProfile.wakeTime ?? { hours: 8, minutes: 0 };
        const wakeToday = new Date(startTime);
        wakeToday.setHours(wake.hours, wake.minutes, 0, 0);
        const diffMs = startTime.getTime() - wakeToday.getTime();
        const hoursAwake = diffMs < 0 ? 0 : diffMs / (1000 * 60 * 60);
        const matchedWave = rhythmProfile.waves.find(w => hoursAwake >= w.startHour && hoursAwake < w.endHour);

        return {
          id: `gcal-${event.id}`,
          category: 'General',
          task: event.summary || 'Untitled Event',
          waveId: matchedWave?.id,
          slot: startTime.toISOString(),
          loggedAt: new Date().toISOString(),
          note: event.description || `Imported from Google Calendar`,
          done: false,
          isAnchor: false
        };
      });

      // Filter out events that already exist (by checking if ID starts with 'gcal-')
      setCheckIns(prev => {
        const existingGcalIds = new Set(
          prev.filter(c => c.id.startsWith('gcal-')).map(c => c.id)
        );
        const uniqueNewBeats = newBeats.filter(b => !existingGcalIds.has(b.id));
        return [...uniqueNewBeats, ...prev];
      });

      setSaveStatus('saved');
      setTimeout(() => setSaveStatus('idle'), 1200);

      return newBeats.length;
    } catch (error) {
      console.error('Failed to import from Google Calendar:', error);
      setSaveStatus('error');
      setTimeout(() => setSaveStatus('idle'), 2000);
      return 0;
    }
  };

  const scheduleBeat = async (
    category: string,
    task: string,
    when: Date,
    note?: string,
    waveId?: string,
    isAnchor?: boolean,
    trackInfo?: { id: string; name: string; emotion: string }
  ) => {
    const entry: CheckIn = {
      id: Date.now().toString() + Math.random(),
      category,
      task,
      waveId,
      slot: when.toISOString(),
      loggedAt: new Date().toISOString(),
      note,
      done: false,
      isAnchor,
      selectedTrackId: trackInfo?.id,
      selectedTrackName: trackInfo?.name,
      selectedTrackEmotion: trackInfo?.emotion
    };
    setCheckIns(prev => [entry, ...prev]);

    // Log to audit trail
    logCheckInCreated(category, task, isAnchor || false, waveId);

    // Auto-sync to Google Calendar
    if (syncEnabled && gcalAuthed) {
      await syncToGoogleCalendar(entry);
    }
  };

  const markDone = (id: string) => {
    const checkIn = checkIns.find(c => c.id === id);
    if (checkIn) {
      // Log completion to audit trail
      logCheckInCompleted(checkIn.category, checkIn.task, checkIn.slot, new Date().toISOString());

      // Record action to metricsHub for real-time metric updates
      metricsHub.recordAction(`completed: ${checkIn.category} - ${checkIn.task}`);
    }
    setCheckIns(prev => prev.map(c => c.id === id ? { ...c, done: true, loggedAt: new Date().toISOString() } : c));
  };

  const removeCheckIn = (id: string) => {
    const checkIn = checkIns.find(c => c.id === id);
    if (checkIn) {
      auditLog.addEntry('CHECKIN_DELETED', 'info', `Deleted: ${checkIn.task}`, {
        category: checkIn.category,
        task: checkIn.task,
        wasCompleted: checkIn.done
      });
    }
    setCheckIns(prev => prev.filter(c => c.id !== id));
  };

  const toggleExpanded = (id: string) => {
    setCheckIns(prev => prev.map(c => c.id === id ? { ...c, expanded: !c.expanded } : c));
  };

  const snoozeAnchor = (id: string, minutes: number) => {
    const anchor = checkIns.find(c => c.id === id);
    if (!anchor) return;

    // Log snooze to audit trail
    auditLog.addEntry('CHECKIN_SNOOZED', 'info', `Snoozed: ${anchor.task} by ${minutes} min`, {
      task: anchor.task,
      snoozeMinutes: minutes,
      originalSlot: anchor.slot
    });

    const newTime = addMinutes(new Date(anchor.slot), minutes);
    scheduleBeat(anchor.category, anchor.task, newTime, anchor.note, anchor.waveId, anchor.isAnchor);
    removeCheckIn(id);
  };

  const copyYesterdayAnchors = () => {
    const yesterday = new Date(selectedDate);
    yesterday.setDate(yesterday.getDate() - 1);

    let copiedCount = 0;
    rhythmProfile.waves.forEach(wave => {
      const yAnchor = checkIns.find(
        c => c.isAnchor && c.waveId === wave.id && sameDay(new Date(c.slot), yesterday)
      );
      if (!yAnchor) return;

      const when = new Date(selectedDate);
      const anchorTime = new Date(yAnchor.slot);
      when.setHours(anchorTime.getHours(), anchorTime.getMinutes(), 0, 0);

      scheduleBeat('Anchor', yAnchor.task, when, yAnchor.note, wave.id, true);
      copiedCount++;
    });

    // Log anchors copied
    if (copiedCount > 0) {
      auditLog.addEntry('ANCHORS_COPIED', 'success', `Copied ${copiedCount} anchor(s) from yesterday`, {
        count: copiedCount,
        fromDate: yesterday.toISOString(),
        toDate: selectedDate.toISOString()
      });
    }
  };

  const addJournalEntry = () => {
    if (!newJournalContent.trim()) return;
    const dayKey = toDateInput(selectedDate);
    const entry: JournalEntry = {
      id: Date.now().toString() + Math.random(),
      timestamp: new Date().toISOString(),
      content: newJournalContent.trim(),
      waveId: newJournalWave || undefined
    };
    setJournals(prev => ({
      ...prev,
      [dayKey]: [...(prev[dayKey] || []), entry]
    }));

    // Log journal creation to audit trail
    auditLog.addEntry('JOURNAL_CREATED', 'success', 'Journal entry created', {
      wordCount: newJournalContent.trim().split(/\s+/).length,
      waveId: newJournalWave || 'none',
      date: dayKey
    }, { waveId: newJournalWave || undefined });

    setNewJournalContent('');
    setNewJournalWave('');
    setNewJournalOpen(false);
  };

  const deleteJournalEntry = (entryId: string) => {
    const dayKey = toDateInput(selectedDate);
    const entry = journals[dayKey]?.find(e => e.id === entryId);

    // Log journal deletion to audit trail
    if (entry) {
      auditLog.addEntry('JOURNAL_DELETED', 'info', 'Journal entry deleted', {
        wordCount: entry.content.split(/\s+/).length,
        date: dayKey
      });
    }

    setJournals(prev => ({
      ...prev,
      [dayKey]: (prev[dayKey] || []).filter(e => e.id !== entryId)
    }));
  };

  const downloadJournalEntry = (entry: JournalEntry) => {
    const blob = new Blob([entry.content], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `journal_${new Date(entry.timestamp).toISOString()}.txt`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  const downloadAllJournals = () => {
    const dayKey = toDateInput(selectedDate);
    const entries = journals[dayKey] || [];
    const combined = entries.map(e =>
      `[${new Date(e.timestamp).toLocaleString()}]\n${e.content}\n\n`
    ).join('---\n\n');
    const blob = new Blob([combined], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `journal_all_${dayKey}.txt`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  // Get current wave based on actual wake time
  const getCurrentWave = (): Wave | null => {
    const now = new Date();
    const wake = rhythmProfile.wakeTime ?? { hours: 8, minutes: 0 };
    const wakeToday = new Date(now);
    wakeToday.setHours(wake.hours, wake.minutes, 0, 0);

    const diffMs = now.getTime() - wakeToday.getTime();
    const hoursAwake = diffMs < 0 ? 0 : diffMs / (1000 * 60 * 60);

    return rhythmProfile.waves.find(w => hoursAwake >= w.startHour && hoursAwake < w.endHour) || null;
  };

  const isDeviationDay = (): boolean => {
    return selectedDate.getDay() === rhythmProfile.deviationDay;
  };

  const getWaveColor = (waveId?: string) => {
    if (isDeviationDay()) return 'orange';
    const wave = rhythmProfile.waves.find(w => w.id === waveId);
    return wave?.color || 'gray';
  };

  const calculateRhythmScore = (date: Date): number => {
    const dayAnchors = checkIns.filter(c => c.isAnchor && sameDay(new Date(c.loggedAt), date));
    const completedAnchors = dayAnchors.filter(c => c.done);
    return dayAnchors.length > 0 ? Math.round((completedAnchors.length / dayAnchors.length) * 100) : 0;
  };

  const upcoming = checkIns
    .filter(c => !c.done && sameDay(new Date(c.slot), selectedDate))
    .sort((a, b) => new Date(a.slot).getTime() - new Date(b.slot).getTime());

  const dayCompleted = checkIns.filter(c => c.done && sameDay(new Date(c.loggedAt), selectedDate));
  const dayKey = toDateInput(selectedDate);
  const dayJournals = journals[dayKey] || [];
  const currentWave = getCurrentWave();
  const rhythmScore = calculateRhythmScore(selectedDate);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-black via-gray-900 to-blue-950 text-gray-100 flex items-center justify-center">
        <div className="text-center space-y-4">
          <Loader2 className="w-12 h-12 animate-spin mx-auto text-purple-400" />
          <p className="text-gray-400">Tuning to your rhythm...</p>
        </div>
      </div>
    );
  }

  // Render Analytics Page when navigated
  if (currentView === 'analytics') {
    return (
      <AnalyticsPage
        checkIns={checkIns}
        waves={rhythmProfile.waves}
        onBack={() => setCurrentView('dashboard')}
      />
    );
  }

  // Render User Profile Page when navigated
  if (currentView === 'profile') {
    return (
      <UserProfilePage
        onBack={() => setCurrentView('dashboard')}
        onNavigateToRoadmap={(domain) => {
          setCurrentRoadmapDomain(domain);
          setCurrentView('roadmap');
        }}
        deltaHV={deltaHVState || undefined}
        onSelectBeat={(category) => {
          setMiniCategory(category);
          setMiniTime(toTimeInput(addMinutes(new Date(), 30)));
          setMiniOpen(true);
          setCurrentView('dashboard');
        }}
      />
    );
  }

  // Render Roadmap View when navigated
  if (currentView === 'roadmap' && currentRoadmapDomain) {
    return (
      <RoadmapView
        domainId={currentRoadmapDomain}
        onBack={() => setCurrentView('profile')}
        onScheduleBeat={(category, context) => {
          // Pre-fill the mini scheduler with roadmap context
          setMiniCategory(category);
          setMiniNote(`Roadmap: ${context}`);
          setMiniTime(toTimeInput(addMinutes(new Date(), 30)));
          setMiniOpen(true);
          setCurrentView('dashboard');
        }}
      />
    );
  }

  // Render Challenges View when navigated
  if (currentView === 'challenges') {
    return (
      <div className="min-h-screen text-gray-100 p-4 md:p-8" style={{
        background: `linear-gradient(var(--gradient-angle, 180deg), var(--bg-primary, #000) 0%, var(--bg-secondary, #111827) 50%, var(--bg-tertiary, #1e3a5f) 100%)`
      }}>
        <div className="max-w-4xl mx-auto">
          <div className="flex items-center justify-between mb-6">
            <button
              onClick={() => setCurrentView('dashboard')}
              className="flex items-center gap-2 px-4 py-2 bg-gray-800/50 rounded-lg hover:bg-gray-700/50 transition-colors"
            >
              <ChevronDown className="w-4 h-4 rotate-90" />
              Back to Dashboard
            </button>
            <button
              onClick={() => setCurrentView('cosmetics')}
              className="flex items-center gap-2 px-4 py-2 bg-purple-600/30 border border-purple-500/50 rounded-lg hover:bg-purple-600/40 transition-colors text-purple-300"
            >
              <Palette className="w-4 h-4" />
              View Cosmetics
            </button>
          </div>
          <ChallengeHub
            userMetrics={userMetrics}
            checkIns={checkIns.map(c => ({ category: c.category, done: c.done }))}
            journalCount={(journals[toDateInput(selectedDate)] || []).length}
            onCosmeticUnlocked={() => {
              // Refresh cosmetics
              const data = challengeRewardService.getData();
              setEquippedCosmetics(data.equippedCosmetics);
              setCosmeticCssVars(generateCosmeticCSS(data.equippedCosmetics));
            }}
          />
        </div>
      </div>
    );
  }

  // Render Cosmetics Inventory when navigated
  if (currentView === 'cosmetics') {
    return (
      <div className="min-h-screen text-gray-100 p-4 md:p-8" style={{
        background: `linear-gradient(var(--gradient-angle, 180deg), var(--bg-primary, #000) 0%, var(--bg-secondary, #111827) 50%, var(--bg-tertiary, #1e3a5f) 100%)`
      }}>
        <div className="max-w-6xl mx-auto">
          <div className="flex items-center justify-between mb-6">
            <button
              onClick={() => setCurrentView('dashboard')}
              className="flex items-center gap-2 px-4 py-2 bg-gray-800/50 rounded-lg hover:bg-gray-700/50 transition-colors"
            >
              <ChevronDown className="w-4 h-4 rotate-90" />
              Back to Dashboard
            </button>
            <button
              onClick={() => setCurrentView('challenges')}
              className="flex items-center gap-2 px-4 py-2 bg-orange-600/30 border border-orange-500/50 rounded-lg hover:bg-orange-600/40 transition-colors text-orange-300"
            >
              <Target className="w-4 h-4" />
              View Challenges
            </button>
          </div>
          <CosmeticsInventory
            onEquipmentChange={(equipped) => {
              setEquippedCosmetics(equipped);
              setCosmeticCssVars(generateCosmeticCSS(equipped));
            }}
          />
        </div>
      </div>
    );
  }

  // Get equipped cosmetic CSS classes
  const cosmeticClasses = getEquippedCssClasses(equippedCosmetics);

  return (
    <div
      className={`min-h-screen text-gray-100 p-4 md:p-8 ${cosmeticClasses.join(' ')}`}
      style={{
        background: `linear-gradient(var(--gradient-angle, 180deg), var(--bg-primary, #000) 0%, var(--bg-secondary, #111827) 50%, var(--bg-tertiary, #1e3a5f) 100%)`
      }}
    >
      <div className="max-w-4xl mx-auto space-y-6">

        {/* Header */}
        <div className="text-center space-y-3">
          <div className="flex items-start justify-between">
            {/* Collapsed Anchors Icon - Upper Left */}
            <button
              onClick={() => setAnchorsExpanded(!anchorsExpanded)}
              className="relative group"
            >
              <div className="w-12 h-12 rounded-full bg-gradient-to-br from-amber-900 to-amber-950 border-2 border-amber-700 hover:border-amber-500 flex items-center justify-center transition-all group-hover:scale-105 overflow-hidden">
                <div className="absolute inset-0 bg-gradient-to-br from-white/20 via-transparent to-transparent opacity-40" />
                <div className="relative z-10">
                  <Zap className="w-6 h-6 text-amber-400" />
                </div>
              </div>
              <div className="absolute -top-1 -right-1 w-5 h-5 rounded-full bg-purple-600 border-2 border-black flex items-center justify-center text-xs font-bold">
                {checkIns.filter(c => c.isAnchor && c.done && sameDay(new Date(c.loggedAt), selectedDate)).length}/{rhythmProfile.waves.length}
              </div>
            </button>

            <div className="flex-1">
              <h1 className="text-4xl md:text-6xl font-thin tracking-wider bg-gradient-to-r from-cyan-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
                Pulse Check Rhythm
              </h1>
              <p className="text-sm text-gray-500">by <span className="italic font-serif">sKiDa</span></p>
            </div>

            <div className="w-12"></div>
          </div>

          <div className="flex items-center justify-center gap-4 flex-wrap">
            <button onClick={() => setDatePickerOpen(!datePickerOpen)} className="px-4 py-2 rounded-lg bg-gray-900/70 border border-gray-800 hover:bg-gray-800 flex items-center gap-2">
              <Calendar className="w-4 h-4" />
              {selectedDate.toDateString()}
              {isDeviationDay() && <span className="text-orange-400 text-xs ml-2">⚡ Deviation Day</span>}
            </button>
            <button onClick={() => setSetupOpen(true)} className="px-4 py-2 rounded-lg bg-purple-900/70 border border-purple-700 hover:bg-purple-800 flex items-center gap-2">
              <Waves className="w-4 h-4" />
              Wave Setup
            </button>

            {/* Google Calendar Integration */}
            {gcalService && (
              <>
                {!gcalAuthed ? (
                  <button
                    onClick={async () => {
                      const authed = await gcalService.authenticate();
                      setGcalAuthed(authed);
                    }}
                    className="px-3 py-1.5 rounded-lg bg-green-600 hover:bg-green-500 text-sm flex items-center gap-2"
                  >
                    <Calendar className="w-4 h-4" />
                    Connect Google Calendar
                  </button>
                ) : (
                  <>
                    <label className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-gray-900/70 border border-gray-800 text-sm cursor-pointer hover:bg-gray-800">
                      <input
                        type="checkbox"
                        checked={syncEnabled}
                        onChange={(e) => setSyncEnabled(e.target.checked)}
                        className="rounded"
                      />
                      Sync to Calendar
                    </label>
                    <button
                      onClick={async () => {
                        const count = await importFromGoogleCalendar(selectedDate);
                        if (count !== undefined && count > 0) {
                          alert(`Imported ${count} event(s) from Google Calendar`);
                        } else if (count === 0) {
                          alert('No new events to import');
                        }
                      }}
                      className="px-3 py-1.5 rounded-lg bg-blue-600 hover:bg-blue-500 text-sm flex items-center gap-2"
                      title="Import today's events from Google Calendar"
                    >
                      <Download className="w-4 h-4" />
                      Import Events
                    </button>
                  </>
                )}
              </>
            )}

            {/* Notifications Button */}
            <button
              onClick={() => {
                if (notificationPermission === 'default') {
                  notificationService.requestPermission();
                } else {
                  setNotificationSettingsOpen(true);
                }
              }}
              className={`px-3 py-1.5 rounded-lg text-sm flex items-center gap-2 ${
                notificationPermission === 'granted'
                  ? 'bg-emerald-600 hover:bg-emerald-500'
                  : notificationPermission === 'denied'
                  ? 'bg-rose-900/50 border border-rose-700/50 text-rose-300'
                  : 'bg-amber-600 hover:bg-amber-500'
              }`}
              title={notificationPermission === 'granted' ? 'Notification Settings' : 'Enable Notifications'}
            >
              <Bell className="w-4 h-4" />
              {notificationPermission === 'granted' ? 'Notifications' :
               notificationPermission === 'denied' ? 'Blocked' : 'Enable'}
            </button>

            {/* Analytics Button */}
            <button
              onClick={() => setCurrentView('analytics')}
              className="px-3 py-1.5 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-sm flex items-center gap-2"
              title="View Analytics Dashboard"
            >
              <BarChart3 className="w-4 h-4" />
              Analytics
            </button>

            {/* Music Library Button */}
            <button
              onClick={() => setMusicLibraryOpen(true)}
              className="px-3 py-1.5 rounded-lg bg-purple-600 hover:bg-purple-500 text-sm flex items-center gap-2"
              title={dailyMusicEmotion ? `Today: ${EMOTIONAL_CATEGORIES[dailyMusicEmotion].name}` : 'Music Library'}
            >
              <Music className="w-4 h-4" />
              {dailyMusicEmotion ? (
                <span className="flex items-center gap-1">
                  <span>{EMOTIONAL_CATEGORIES[dailyMusicEmotion].icon}</span>
                  <span>{EMOTIONAL_CATEGORIES[dailyMusicEmotion].name}</span>
                </span>
              ) : 'Music'}
            </button>

            {/* User Profile Button */}
            <button
              onClick={() => setCurrentView('profile')}
              className="px-3 py-1.5 rounded-lg bg-pink-600 hover:bg-pink-500 text-sm flex items-center gap-2"
              title="User Profile & Roadmap"
            >
              <User className="w-4 h-4" />
              Profile
            </button>

          </div>
          <div className="text-sm text-gray-500 h-5">
            {saveStatus === 'saving' && 'Syncing...'}
            {saveStatus === 'saved' && 'Synced ✓'}
            {saveStatus === 'error' && 'Sync failed'}
          </div>
        </div>

        {/* Current Wave, Rhythm State & Score */}
        <div className="bg-gray-950/60 backdrop-blur border border-gray-800 rounded-2xl p-6">
          <div className="flex items-center justify-between flex-wrap gap-4">
            {/* Wave Info */}
            <div className="flex items-center gap-3">
              <Waves className={`w-8 h-8 ${waveColorClasses[getWaveColor(currentWave?.id)].text}`} />
              <div>
                <p className="text-lg font-medium">{currentWave?.name || 'No Active Wave'}</p>
                <p className="text-sm text-gray-400">{currentWave?.description || 'Define your waves'}</p>
              </div>
            </div>

            {/* Rhythm State Indicator (Phase 2) */}
            {rhythmStateInfo && (
              <div className={`flex items-center gap-2 px-4 py-2 rounded-xl border ${
                rhythmStateInfo.state === RhythmState.FOCUS ? 'bg-cyan-950/40 border-cyan-700/50 text-cyan-300' :
                rhythmStateInfo.state === RhythmState.REFLECTIVE ? 'bg-blue-950/40 border-blue-700/50 text-blue-300' :
                'bg-purple-950/40 border-purple-700/50 text-purple-300'
              }`}>
                <span className="text-xl">{rhythmStateInfo.uiHint.icon}</span>
                <div>
                  <p className="text-sm font-medium">{rhythmStateInfo.uiHint.label}</p>
                  <p className="text-xs opacity-70">{rhythmStateInfo.trigger}</p>
                </div>
              </div>
            )}

            {/* Rhythm Score */}
            <div className="text-center">
              <p className="text-3xl font-bold text-purple-400">{rhythmScore}%</p>
              <p className="text-xs text-gray-500">Rhythm Score</p>
            </div>
          </div>

          {/* Rhythm State Suggestions (collapsible) */}
          {rhythmStateInfo && (
            <div className="mt-4 pt-4 border-t border-gray-800/50">
              <div className="flex items-center justify-between mb-2">
                <p className="text-xs text-gray-400">Suggested Actions for {rhythmStateInfo.uiHint.label}</p>
                <button
                  onClick={() => setAuditTrailOpen(true)}
                  className="text-xs text-gray-500 hover:text-gray-300 flex items-center gap-1"
                >
                  <FileText className="w-3 h-3" />
                  View Audit Trail
                </button>
              </div>
              <div className="flex flex-wrap gap-2">
                {rhythmStateInfo.suggestedActions.slice(0, 3).map((action, i) => (
                  <span key={i} className="text-xs px-2 py-1 bg-gray-900/60 rounded-lg text-gray-400">
                    {action}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* DeltaHV Metrics Panel */}
        {deltaHVState && (
          <div className="bg-gray-950/60 backdrop-blur border border-gray-800 rounded-2xl overflow-hidden">
            <button
              onClick={() => setDeltaHVExpanded(!deltaHVExpanded)}
              className="w-full p-4 flex items-center justify-between hover:bg-gray-900/40 transition-colors"
            >
              <div className="flex items-center gap-3">
                <div className={`w-10 h-10 rounded-full flex items-center justify-center ${
                  deltaHVState.fieldState === 'coherent' ? 'bg-emerald-900/60 text-emerald-400' :
                  deltaHVState.fieldState === 'transitioning' ? 'bg-amber-900/60 text-amber-400' :
                  deltaHVState.fieldState === 'fragmented' ? 'bg-orange-900/60 text-orange-400' :
                  'bg-gray-800/60 text-gray-400'
                }`}>
                  <Activity className="w-5 h-5" />
                </div>
                <div className="text-left">
                  <p className="font-medium flex items-center gap-2">
                    ΔHV Field State
                    <span className={`text-xs px-2 py-0.5 rounded-full ${
                      deltaHVState.fieldState === 'coherent' ? 'bg-emerald-900/60 text-emerald-400' :
                      deltaHVState.fieldState === 'transitioning' ? 'bg-amber-900/60 text-amber-400' :
                      deltaHVState.fieldState === 'fragmented' ? 'bg-orange-900/60 text-orange-400' :
                      'bg-gray-800/60 text-gray-400'
                    }`}>
                      {deltaHVState.fieldState}
                    </span>
                  </p>
                  <p className="text-sm text-gray-400">Coherence metrics for {selectedDate.toDateString()}</p>
                </div>
              </div>
              <div className="flex items-center gap-4">
                <div className="text-right">
                  <p className={`text-2xl font-bold ${
                    deltaHVState.deltaHV >= 75 ? 'text-emerald-400' :
                    deltaHVState.deltaHV >= 50 ? 'text-amber-400' :
                    deltaHVState.deltaHV >= 25 ? 'text-orange-400' :
                    'text-gray-400'
                  }`}>{deltaHVState.deltaHV}</p>
                  <p className="text-xs text-gray-500">ΔHV Score</p>
                </div>
                {deltaHVExpanded ? <ChevronUp className="w-5 h-5 text-gray-400" /> : <ChevronDown className="w-5 h-5 text-gray-400" />}
              </div>
            </button>

            {deltaHVExpanded && (
              <div className="px-4 pb-4 space-y-4 border-t border-gray-800 pt-4">
                {/* Four Core Metrics with Brain Regions */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  {/* Symbolic Density */}
                  <div className="bg-gray-900/60 rounded-xl p-3 border border-violet-800/30">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <span className="text-lg">✨</span>
                        <span className="text-xs text-violet-300">Symbolic (S)</span>
                      </div>
                      <p className="text-xl font-bold text-violet-400">{deltaHVState.symbolicDensity}%</p>
                    </div>
                    <div className="flex flex-wrap gap-1 text-xs text-violet-400/70">
                      <span title="Dorsolateral">🎯</span>
                      <span title="Medial">🪞</span>
                      <span title="Precuneus">🌌</span>
                      <span title="Temporal">🏛️</span>
                      <span title="Hippocampus">📝</span>
                      <span title="Posterior">🧘</span>
                    </div>
                    <p className="text-xs text-gray-500 mt-1">{deltaHVState.breakdown.glyphCount} glyphs, {deltaHVState.breakdown.intentionCount} intentions</p>
                  </div>

                  {/* Resonance Coupling */}
                  <div className="bg-gray-900/60 rounded-xl p-3 border border-cyan-800/30">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <span className="text-lg">🎯</span>
                        <span className="text-xs text-cyan-300">Resonance (R)</span>
                      </div>
                      <p className="text-xl font-bold text-cyan-400">{deltaHVState.resonanceCoupling}%</p>
                    </div>
                    <div className="flex flex-wrap gap-1 text-xs text-cyan-400/70">
                      <span title="Orbitofrontal">⚖️</span>
                      <span title="Dorsal ACC">⚠️</span>
                      <span title="Rostral ACC">💗</span>
                      <span title="Premotor">🎬</span>
                      <span title="Supplementary">🔄</span>
                      <span title="Nucleus Accumbens">🎯</span>
                    </div>
                    <p className="text-xs text-gray-500 mt-1">{deltaHVState.breakdown.alignedTasks}/{deltaHVState.breakdown.totalPlannedTasks} aligned</p>
                  </div>

                  {/* Friction Coefficient */}
                  <div className="bg-gray-900/60 rounded-xl p-3 border border-orange-800/30">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <span className="text-lg">🌧️</span>
                        <span className="text-xs text-orange-300">Friction (δφ)</span>
                      </div>
                      <p className="text-xl font-bold text-orange-400">{deltaHVState.frictionCoefficient}%</p>
                    </div>
                    <div className="flex flex-wrap gap-1 text-xs text-orange-400/70">
                      <span title="Subgenual">🌧️</span>
                      <span title="Basolateral">⚡</span>
                      <span title="Central">🚨</span>
                      <span title="Paraventricular">🌊</span>
                      <span title="Bed Nucleus">😰</span>
                      <span title="Habenula">🚫</span>
                    </div>
                    <p className="text-xs text-gray-500 mt-1">{deltaHVState.breakdown.missedTasks} missed, {deltaHVState.breakdown.delayedTasks} delayed</p>
                  </div>

                  {/* Harmonic Stability */}
                  <div className="bg-gray-900/60 rounded-xl p-3 border border-emerald-800/30">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <span className="text-lg">⚖️</span>
                        <span className="text-xs text-emerald-300">Stability (H)</span>
                      </div>
                      <p className="text-xl font-bold text-emerald-400">{deltaHVState.harmonicStability}%</p>
                    </div>
                    <div className="flex flex-wrap gap-1 text-xs text-emerald-400/70">
                      <span title="Septal">😌</span>
                      <span title="Dorsal Raphe">☮️</span>
                      <span title="Median Raphe">🌅</span>
                      <span title="Reticular">🔋</span>
                      <span title="Vermis">🧘</span>
                      <span title="Cerebellar">⚖️</span>
                    </div>
                    <p className="text-xs text-gray-500 mt-1">{Object.values(deltaHVState.breakdown.segmentCoverage).filter(Boolean).length}/{rhythmProfile.waves.length} waves active</p>
                  </div>
                </div>

                {/* Music Influence (if available) */}
                {enhancedMetrics?.musicInfluence && enhancedMetrics.musicInfluence.authorshipScore > 0 && (
                  <div className="bg-gray-900/40 rounded-xl p-3 border border-purple-800/30">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Music className="w-4 h-4 text-purple-400" />
                        <span className="text-sm text-purple-300">Music Story Influence</span>
                      </div>
                      <div className="flex items-center gap-4 text-xs">
                        <span className="text-gray-400">
                          Authorship: <span className="text-purple-400">{enhancedMetrics.musicInfluence.authorshipScore}%</span>
                        </span>
                        <span className="text-gray-400">
                          Skip Ratio: <span className="text-cyan-400">{Math.round(enhancedMetrics.musicInfluence.skipRatio * 100)}%</span>
                        </span>
                        <span className={`capitalize ${
                          enhancedMetrics.musicInfluence.emotionalTrajectory === 'rising' ? 'text-green-400' :
                          enhancedMetrics.musicInfluence.emotionalTrajectory === 'processing' ? 'text-yellow-400' :
                          'text-gray-400'
                        }`}>
                          {enhancedMetrics.musicInfluence.emotionalTrajectory}
                        </span>
                      </div>
                    </div>
                  </div>
                )}

                {/* Segment Coverage Visualization */}
                <div className="bg-gray-900/40 rounded-xl p-3 border border-gray-800">
                  <p className="text-xs text-gray-400 mb-2">Wave Segment Coverage</p>
                  <div className="flex gap-2">
                    {rhythmProfile.waves.map(wave => {
                      const isActive = deltaHVState.breakdown.segmentCoverage[wave.id];
                      const colors = waveColorClasses[wave.color];
                      return (
                        <div
                          key={wave.id}
                          className={`flex-1 h-8 rounded-lg flex items-center justify-center text-xs font-medium transition-all ${
                            isActive ? `${colors.bg} ${colors.text} border ${colors.border}` : 'bg-gray-800/40 text-gray-600 border border-gray-800'
                          }`}
                          title={`${wave.name}: ${isActive ? 'Active' : 'Inactive'}`}
                        >
                          {wave.name.split(' ')[0]}
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* ΔHV Score Interpretation */}
                <div className={`rounded-xl p-3 border ${
                  deltaHVState.fieldState === 'coherent' ? 'bg-emerald-950/30 border-emerald-800/30' :
                  deltaHVState.fieldState === 'transitioning' ? 'bg-amber-950/30 border-amber-800/30' :
                  deltaHVState.fieldState === 'fragmented' ? 'bg-orange-950/30 border-orange-800/30' :
                  'bg-gray-900/40 border-gray-800'
                }`}>
                  <p className="text-sm">
                    {deltaHVState.fieldState === 'coherent' && (
                      <span className="text-emerald-300">🌟 Your field is highly coherent. Maintain momentum and consider optimizing for peak performance.</span>
                    )}
                    {deltaHVState.fieldState === 'transitioning' && (
                      <span className="text-amber-300">🔄 You're transitioning between states. Stay consistent with your rhythm to build coherence.</span>
                    )}
                    {deltaHVState.fieldState === 'fragmented' && (
                      <span className="text-orange-300">⚡ Some fragmentation detected. Focus on completing planned tasks and grounding activities.</span>
                    )}
                    {deltaHVState.fieldState === 'dormant' && (
                      <span className="text-gray-300">💤 Field is dormant. Start with small symbolic actions like journaling or completing one anchor task.</span>
                    )}
                  </p>
                </div>

                {/* Formula Reference */}
                <div className="text-xs text-gray-500 flex flex-wrap gap-3 justify-center">
                  <span>ΔHV = S(20%) + R(30%) + (100-δφ)(25%) + H(25%)</span>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Focus & Wellness Tools - Integrated Tabs Section */}
        <FocusWellnessTools
          deltaHV={deltaHVState}
          enhancedMetrics={enhancedMetrics}
          onOpenFullDJ={() => setMixerOpen(true)}
          onOpenFullChallenges={() => setGlyphCanvasOpen(true)}
          onCompleteChallenge={(regionId, xp) => {
            console.log(`Challenge completed: ${regionId} +${xp}XP`);
            metricsHub.recordAction(`challenge_complete_${regionId}`);
          }}
        />

        {/* AI Planner Suggestions (Phase 3 Complete) */}
        {(plannerSuggestions.length > 0 || gcalAuthed) && (
          <div className="bg-gradient-to-r from-purple-950/40 to-cyan-950/40 backdrop-blur border border-purple-700/30 rounded-2xl overflow-hidden">
            <div className="p-4 flex items-center justify-between">
              <button
                onClick={() => setSuggestionsExpanded(!suggestionsExpanded)}
                className="flex items-center gap-3 flex-1 hover:bg-white/5 transition-colors rounded-lg -m-2 p-2"
              >
                <div className="w-10 h-10 rounded-full bg-purple-900/60 flex items-center justify-center">
                  <Sparkles className="w-5 h-5 text-purple-400" />
                </div>
                <div className="text-left">
                  <p className="font-medium flex items-center gap-2">
                    AI Planner
                    {plannerSuggestions.length > 0 && (
                      <span className="text-xs px-2 py-0.5 rounded-full bg-purple-900/60 text-purple-300">
                        {plannerSuggestions.length} active
                      </span>
                    )}
                    {plannerPreferences.autoScheduleEnabled && (
                      <span className="text-xs px-2 py-0.5 rounded-full bg-cyan-900/60 text-cyan-300">
                        Auto
                      </span>
                    )}
                  </p>
                  <p className="text-sm text-gray-400">
                    {plannerPreferences.autoScheduleEnabled
                      ? 'Auto-scheduling enabled'
                      : 'Rhythm optimization recommendations'}
                  </p>
                </div>
              </button>
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setPlannerSettingsOpen(true)}
                  className="p-2 rounded-lg hover:bg-white/10 transition-colors text-gray-400 hover:text-white"
                  title="Planner Settings"
                >
                  <Settings className="w-5 h-5" />
                </button>
                {suggestionsExpanded ? <ChevronUp className="w-5 h-5 text-gray-400" /> : <ChevronDown className="w-5 h-5 text-gray-400" />}
              </div>
            </div>

            {suggestionsExpanded && (
              <div className="px-4 pb-4 space-y-3">
                {plannerSuggestions.length === 0 ? (
                  <div className="text-center py-6 text-gray-500">
                    <Sparkles className="w-8 h-8 mx-auto mb-2 opacity-50" />
                    <p className="text-sm">No suggestions yet</p>
                    <p className="text-xs mt-1">Suggestions appear based on your rhythm patterns</p>
                  </div>
                ) : (
                  plannerSuggestions.map(suggestion => (
                    <div
                      key={suggestion.id}
                      className={`rounded-xl p-4 border ${
                        suggestion.autoScheduled ? 'bg-cyan-950/30 border-cyan-700/40' :
                        suggestion.priority === 'high' ? 'bg-rose-950/30 border-rose-700/40' :
                        suggestion.priority === 'medium' ? 'bg-amber-950/30 border-amber-700/40' :
                        'bg-gray-900/40 border-gray-700/40'
                      }`}
                    >
                      <div className="flex items-start justify-between gap-3">
                        <div className="flex-1">
                          <p className={`font-medium ${
                            suggestion.autoScheduled ? 'text-cyan-300' :
                            suggestion.priority === 'high' ? 'text-rose-300' :
                            suggestion.priority === 'medium' ? 'text-amber-300' :
                            'text-gray-300'
                          }`}>
                            {suggestion.title}
                          </p>
                          <p className="text-sm text-gray-400 mt-1">{suggestion.description}</p>
                        </div>
                        <div className="flex gap-2">
                          {suggestion.action?.type === 'create_event' && (
                            <button
                              onClick={() => {
                                const accepted = plannerRef.current?.acceptSuggestion(suggestion.id);
                                if (accepted?.action?.payload) {
                                  const payload = accepted.action.payload;
                                  const start = new Date();
                                  start.setMinutes(start.getMinutes() + 5);
                                  scheduleBeat(
                                    'General',
                                    payload.summary,
                                    start,
                                    payload.description,
                                    undefined,
                                    false
                                  );
                                }
                              }}
                              className="px-3 py-1.5 rounded-lg bg-purple-600 hover:bg-purple-500 text-sm"
                            >
                              Schedule
                            </button>
                          )}
                          {suggestion.autoScheduled && (
                            <button
                              onClick={async () => {
                                await plannerRef.current?.cancelAutoScheduledEvent(suggestion.id);
                              }}
                              className="px-3 py-1.5 rounded-lg bg-rose-600 hover:bg-rose-500 text-sm"
                            >
                              Cancel Event
                            </button>
                          )}
                          <button
                            onClick={() => plannerRef.current?.dismissSuggestion(suggestion.id)}
                            className="px-3 py-1.5 rounded-lg bg-gray-800 hover:bg-gray-700 text-sm text-gray-400"
                          >
                            Dismiss
                          </button>
                        </div>
                      </div>
                      <div className="flex items-center gap-2 mt-2">
                        <span className={`text-xs px-2 py-0.5 rounded ${
                          suggestion.autoScheduled ? 'bg-cyan-900/50 text-cyan-300' :
                          suggestion.priority === 'high' ? 'bg-rose-900/50 text-rose-300' :
                          suggestion.priority === 'medium' ? 'bg-amber-900/50 text-amber-300' :
                          'bg-gray-800 text-gray-400'
                        }`}>
                          {suggestion.autoScheduled ? 'auto-scheduled' : suggestion.priority}
                        </span>
                        <span className="text-xs text-gray-500">{suggestion.type.replace(/_/g, ' ')}</span>
                      </div>
                    </div>
                  ))
                )}
              </div>
            )}
          </div>
        )}

        {/* Wave Anchors - Expandable */}
        {anchorsExpanded && (
          <div className="bg-gray-950/60 backdrop-blur border border-gray-800 rounded-2xl p-6 animate-in fade-in slide-in-from-top duration-300">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-light flex items-center gap-3">
                <Zap className="w-6 h-6 text-amber-400" />
                Daily Anchors
              </h2>
              <div className="flex gap-2">
                <button onClick={copyYesterdayAnchors} className="px-3 py-1.5 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-sm flex items-center gap-2">
                  <Copy className="w-4 h-4" />
                  Yesterday
                </button>
                <button onClick={() => setWaveAnchorsOpen(true)} className="px-3 py-1.5 rounded-lg bg-amber-600 hover:bg-amber-500 text-sm">
                  Set Anchors
                </button>
                <button onClick={() => setAnchorsExpanded(false)} className="px-3 py-1.5 rounded-lg bg-gray-800 hover:bg-gray-700 text-sm">
                  Hide
                </button>
              </div>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {rhythmProfile.waves.map(wave => {
                const anchor = checkIns.find(c => c.isAnchor && c.waveId === wave.id && sameDay(new Date(c.slot), selectedDate));
                const colors = waveColorClasses[wave.color];
                return (
                  <div key={wave.id} className={`border-2 ${colors.border} ${colors.bgLight} rounded-xl p-4`}>
                    <p className={`${colors.text} font-medium mb-2`}>{wave.name}</p>
                    {anchor ? (
                      <div className="flex items-center justify-between">
                        <p className="text-sm">{anchor.task}</p>
                        {anchor.done ? (
                          <CheckCircle2 className="w-5 h-5 text-green-400" />
                        ) : (
                          <button onClick={() => markDone(anchor.id)} className="p-1 hover:bg-gray-800 rounded">
                            <Clock className="w-5 h-5 text-gray-400" />
                          </button>
                        )}
                      </div>
                    ) : (
                      <p className="text-xs text-gray-500 italic">No anchor set</p>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Round Button Grid 3x3 */}
        <div className="grid grid-cols-3 gap-3 max-w-sm mx-auto">
          <RoundButton icon={<Dumbbell className="w-8 h-8 text-pink-400"/>} label="Workout" onClick={() => {
            setMiniCategory('Workout');
            setMiniTask(''); setMiniNote('');
            setMiniWaveId(currentWave?.id || '');
            setMiniTime(toTimeInput(addMinutes(new Date(), 30)));
            setMiniOpen(true);
          }} />
          <RoundButton icon={<Shield className="w-8 h-8 text-cyan-400"/>} label="Moderation" onClick={() => {
            setMiniCategory('Moderation');
            setMiniTask(''); setMiniNote('');
            setMiniWaveId(currentWave?.id || '');
            setMiniTime(toTimeInput(addMinutes(new Date(), 30)));
            setMiniOpen(true);
          }} />
          <RoundButton icon={<Brain className="w-8 h-8 text-green-400"/>} label="Meditation" onClick={() => {
            setMiniCategory('Meditation');
            setMiniTask(''); setMiniNote('');
            setMiniWaveId(currentWave?.id || '');
            setMiniTime(toTimeInput(addMinutes(new Date(), 30)));
            setMiniOpen(true);
          }} />
          <RoundButton icon={<Heart className="w-8 h-8 text-purple-400"/>} label="Emotion" onClick={() => {
            setMiniCategory('Emotion');
            setMiniTask(''); setMiniNote('');
            setMiniWaveId(currentWave?.id || '');
            setMiniTime(toTimeInput(addMinutes(new Date(), 30)));
            setMiniOpen(true);
          }} />
          <RoundButton icon={<NotebookPen className="w-8 h-8 text-emerald-400"/>} label="Journal" onClick={() => {
            setNewJournalContent('');
            setNewJournalWave(currentWave?.id || '');
            setNewJournalOpen(true);
          }} />
          <RoundButton icon={<Pill className="w-8 h-8 text-indigo-400"/>} label="Med" onClick={() => {
            const when = addMinutes(selectedDate, 15);
            scheduleBeat('Moderation', 'Med Reminder', when, '', currentWave?.id);
          }} />
          <RoundButton icon={<Brain className="w-8 h-8 text-purple-400" />} label="Challenges" onClick={() => setGlyphCanvasOpen(true)} />
          <RoundButton icon={<Disc3 className="w-8 h-8 text-violet-400"/>} label="DJ" onClick={() => setMixerOpen(true)} />
          <RoundButton icon={<span className="text-4xl">∞</span>} label="General" onClick={() => {
            setGeneralNoteText('');
            setGeneralNoteTime(toTimeInput(addMinutes(new Date(), 30)));
            setGeneralNoteOpen(true);
          }} />
        </div>

        {/* Upcoming Beats */}
        <div className="bg-gray-950/60 backdrop-blur border border-gray-800 rounded-2xl p-6">
          <h2 className="text-xl font-light mb-4 flex items-center gap-3">
            <Clock className="w-6 h-6 text-cyan-400" />
            Upcoming Beats - {selectedDate.toDateString()}
          </h2>
          {upcoming.length === 0 ? (
            <p className="text-gray-500 italic">No scheduled beats - flow with your wave</p>
          ) : (
            <div className="space-y-3">
              {upcoming.map(b => {
                const colors = waveColorClasses[getWaveColor(b.waveId)];
                return (
                  <div key={b.id} className="bg-gray-900/60 rounded-xl overflow-hidden">
                    <div className="flex items-center justify-between p-4 cursor-pointer" onClick={() => toggleExpanded(b.id)}>
                      <div className="flex items-center gap-4">
                        <div className="text-2xl">{getIcon(b.category)}</div>
                        <div>
                          <div className="flex items-center gap-2">
                            <p className="font-medium">{b.task}</p>
                            {b.isAnchor && <Zap className="w-4 h-4 text-amber-400" />}
                            {b.waveId && <span className={`text-xs px-2 py-0.5 rounded ${colors.bg} ${colors.text}`}>
                              {rhythmProfile.waves.find(w => w.id === b.waveId)?.name}
                            </span>}
                          </div>
                          <p className="text-sm text-gray-400">{formatTime(new Date(b.slot))}</p>
                        </div>
                      </div>
                      <div className="flex gap-2 items-center">
                        {b.isAnchor && (
                          <>
                            <button onClick={(e) => { e.stopPropagation(); snoozeAnchor(b.id, 15); }} className="p-2 bg-blue-900/60 hover:bg-blue-800 rounded-full transition-colors" title="Snooze 15m">
                              <Timer className="w-4 h-4" />
                            </button>
                            <button onClick={(e) => { e.stopPropagation(); snoozeAnchor(b.id, 60); }} className="p-2 bg-blue-900/60 hover:bg-blue-800 rounded-full transition-colors" title="Snooze 1h">
                              <Clock className="w-4 h-4" />
                            </button>
                          </>
                        )}
                        <button onClick={(e) => { e.stopPropagation(); markDone(b.id); }} className="p-3 bg-green-900/60 hover:bg-green-800 rounded-full transition-colors">
                          <CheckCircle2 className="w-6 h-6" />
                        </button>
                        <button onClick={(e) => { e.stopPropagation(); removeCheckIn(b.id); }} className="p-3 bg-red-900/60 hover:bg-red-800 rounded-full transition-colors">
                          <Trash2 className="w-5 h-5" />
                        </button>
                        {b.expanded ? <ChevronUp className="w-5 h-5 text-gray-400" /> : <ChevronDown className="w-5 h-5 text-gray-400" />}
                      </div>
                    </div>
                    {b.expanded && b.note && (
                      <div className="px-4 pb-4 text-sm text-gray-300 bg-gray-900/80 border-t border-gray-800 pt-3">
                        <p className="font-medium text-gray-400 mb-1">Note:</p>
                        <p>{b.note}</p>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Journal Entries */}
        <div className="bg-gray-950/60 backdrop-blur border border-gray-800 rounded-2xl p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2">
              <NotebookPen className="w-5 h-5 text-emerald-400" />
              <span className="text-lg">Journal Entries - {selectedDate.toDateString()}</span>
            </div>
            <div className="flex gap-2">
              {dayJournals.length > 0 && (
                <button onClick={downloadAllJournals} className="px-3 py-1.5 rounded-lg bg-gray-900/70 border border-gray-800 hover:bg-gray-800 flex items-center gap-2 text-sm">
                  <Download className="w-4 h-4"/>Export All
                </button>
              )}
              <button onClick={() => { setNewJournalWave(currentWave?.id || ''); setNewJournalOpen(true); }} className="px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-500 flex items-center gap-2 text-sm">
                <Plus className="w-4 h-4"/>New Entry
              </button>
            </div>
          </div>

          {dayJournals.length === 0 ? (
            <p className="text-gray-500 italic text-sm">No journal entries for this day</p>
          ) : (
            <div className="flex gap-3 overflow-x-auto pb-2">
              {dayJournals.map(entry => {
                const colors = waveColorClasses[getWaveColor(entry.waveId)];
                return (
                  <div key={entry.id} className="min-w-[280px] bg-gray-900/60 rounded-xl p-4 border border-gray-800">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-gray-400">{new Date(entry.timestamp).toLocaleTimeString()}</span>
                        {entry.waveId && <span className={`text-xs px-2 py-0.5 rounded ${colors.bg} ${colors.text}`}>
                          {rhythmProfile.waves.find(w => w.id === entry.waveId)?.name}
                        </span>}
                      </div>
                      <div className="flex gap-1">
                        <button onClick={() => downloadJournalEntry(entry)} className="p-1 hover:bg-gray-800 rounded">
                          <Download className="w-4 h-4 text-gray-400" />
                        </button>
                        <button onClick={() => deleteJournalEntry(entry.id)} className="p-1 hover:bg-gray-800 rounded">
                          <X className="w-4 h-4 text-gray-400" />
                        </button>
                      </div>
                    </div>
                    <p className="text-sm text-gray-300 line-clamp-4">{entry.content}</p>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* Completed Today */}
        <div className="bg-gray-950/60 backdrop-blur border border-gray-800 rounded-2xl p-6">
          <h2 className="text-xl font-light mb-4">Completed Today</h2>
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {dayCompleted.length === 0 ? (
              <p className="text-gray-500 italic">No completed beats yet</p>
            ) : (
              dayCompleted.map(c => (
                <div key={c.id} className="flex items-center justify-between text-sm bg-gray-900/40 rounded-lg p-3">
                  <span className="text-gray-400">{formatTime(new Date(c.loggedAt))}</span>
                  <div className="flex items-center gap-2">
                    <span className="font-medium">{c.task}</span>
                    {c.isAnchor && <Zap className="w-4 h-4 text-amber-400" />}
                  </div>
                  <span className="text-gray-500">{c.category}</span>
                </div>
              ))
            )}
          </div>
        </div>

        {/* Calendar */}
        <div className="bg-gray-950/60 backdrop-blur border border-gray-800 rounded-2xl">
          <div className="flex items-center justify-between px-4 py-3 cursor-pointer" onClick={() => setCalendarOpen(!calendarOpen)}>
            <div className="flex items-center gap-2 text-sm text-gray-300">
              <Calendar className="w-5 h-5"/>
              Calendar
            </div>
            <button className="px-3 py-1.5 rounded-lg bg-gray-900/70 border border-gray-800 hover:bg-gray-800 flex items-center gap-2 text-sm">
              {calendarOpen ? <><ChevronUp className="w-4 h-4"/>Hide</> : <><ChevronDown className="w-4 h-4"/>Show</>}
            </button>
          </div>
          {calendarOpen && (
            <div className="p-4 pt-0">
              <CalendarMonth
                month={visibleMonth}
                selectedDate={selectedDate}
                deviationDay={rhythmProfile.deviationDay}
                checkIns={checkIns}
                onPrev={() => setVisibleMonth(new Date(visibleMonth.getFullYear(), visibleMonth.getMonth() - 1, 1))}
                onNext={() => setVisibleMonth(new Date(visibleMonth.getFullYear(), visibleMonth.getMonth() + 1, 1))}
                onSelect={(d) => setSelectedDate(d)}
              />

              {/* Bulk Upload to Google Calendar */}
              {gcalAuthed && checkIns.filter(c => !c.done && new Date(c.slot) >= new Date()).length > 0 && (
                <div className="mt-4 pt-4 border-t border-gray-800">
                  <button
                    onClick={async () => {
                      const upcomingBeats = checkIns.filter(c => !c.done && new Date(c.slot) >= new Date());
                      if (!confirm(`Upload ${upcomingBeats.length} upcoming beats to Google Calendar?\n\nThis will create calendar events with full details for each beat.`)) return;

                      setSaveStatus('saving');
                      let successCount = 0;
                      let failCount = 0;
                      const errors: string[] = [];

                      for (const beat of upcomingBeats) {
                        const result = await syncToGoogleCalendar(beat, true); // Force send
                        if (result.success) {
                          successCount++;
                        } else {
                          failCount++;
                          errors.push(`${beat.category} - ${beat.task}: ${result.error}`);
                        }
                      }

                      if (failCount === 0) {
                        setSaveStatus('saved');
                        alert(`✅ Successfully uploaded ${successCount} beats to Google Calendar!\n\nCheck your Google Calendar to see all your beats with full details.`);
                      } else {
                        setSaveStatus('error');
                        const errorMsg = errors.slice(0, 5).join('\n');
                        alert(`⚠️ Upload completed with issues:\n\n✅ Success: ${successCount}\n❌ Failed: ${failCount}\n\nErrors:\n${errorMsg}${errors.length > 5 ? '\n...(and more)' : ''}\n\nCheck browser console for details.`);
                      }
                      setTimeout(() => setSaveStatus('idle'), 2000);
                    }}
                    className="w-full px-4 py-2 rounded-lg bg-green-600 hover:bg-green-500 flex items-center justify-center gap-2 text-sm"
                  >
                    <Calendar className="w-4 h-4" />
                    Upload All Upcoming Beats to Google Calendar ({checkIns.filter(c => !c.done && new Date(c.slot) >= new Date()).length})
                  </button>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Modals */}
        {setupOpen && (
          <WaveSetupWizard
            profile={rhythmProfile}
            onClose={() => setSetupOpen(false)}
            onSave={(profile: RhythmProfile) => {
              setRhythmProfile(profile);
              setSetupOpen(false);
            }}
          />
        )}

        {waveAnchorsOpen && (
          <WaveAnchorsModal
            waves={rhythmProfile.waves}
            selectedDate={selectedDate}
            checkIns={checkIns}
            onClose={() => setWaveAnchorsOpen(false)}
            onSetAnchor={(waveId: string, task: string, note: string) => {
              const when = new Date(selectedDate);
              const wave = rhythmProfile.waves.find(w => w.id === waveId);
              if (wave) {
                when.setHours(wave.startHour, 0, 0, 0);
              }
              scheduleBeat('Anchor', task, when, note, waveId, true);
              setWaveAnchorsOpen(false);
            }}
          />
        )}

        {miniOpen && (
          <MiniScheduler
            date={selectedDate}
            category={miniCategory}
            presetTasks={SUBTASKS[miniCategory] || []}
            time={miniTime}
            note={miniNote}
            waveId={miniWaveId}
            waves={rhythmProfile.waves}
            availableTracks={availableTracks}
            selectedTrack={miniSelectedTrack}
            emotionalCategories={EMOTIONAL_CATEGORIES}
            onChange={(p: any) => {
              if (p.time !== undefined) setMiniTime(p.time);
              if (p.task !== undefined) setMiniTask(p.task);
              if (p.note !== undefined) setMiniNote(p.note);
              if (p.waveId !== undefined) setMiniWaveId(p.waveId);
              if (p.track !== undefined) setMiniSelectedTrack(p.track);
            }}
            onClose={() => {
              setMiniOpen(false);
              setMiniSelectedTrack(null);
            }}
            onSubmit={() => {
              const [hh, mm] = miniTime.split(':').map(n => parseInt(n));
              const when = new Date(selectedDate);
              when.setHours(hh || 0, mm || 0, 0, 0);
              const task = miniTask || `${miniCategory} Beat`;
              scheduleBeat(miniCategory, task, when, miniNote, miniWaveId, false, miniSelectedTrack || undefined);
              setMiniOpen(false);
              setMiniSelectedTrack(null);
            }}
          />
        )}

        {newJournalOpen && (
          <JournalModal
            date={selectedDate}
            value={newJournalContent}
            waveId={newJournalWave}
            waves={rhythmProfile.waves}
            onChange={setNewJournalContent}
            onWaveChange={setNewJournalWave}
            onClose={() => setNewJournalOpen(false)}
            onSave={addJournalEntry}
          />
        )}

        {datePickerOpen && (
          <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4" onClick={() => setDatePickerOpen(false)}>
            <div className="bg-gray-950 border border-gray-800 rounded-2xl p-5" onClick={(e) => e.stopPropagation()}>
              <input
                type="date"
                value={toDateInput(selectedDate)}
                onChange={(e) => {
                  setSelectedDate(new Date(e.target.value + 'T12:00:00'));
                  setDatePickerOpen(false);
                }}
                className="px-4 py-2 rounded-lg bg-gray-900/70 border border-gray-800 text-gray-100 outline-none focus:border-cyan-500"
              />
            </div>
          </div>
        )}

        {generalNoteOpen && (
          <GeneralNoteModal
            date={selectedDate}
            text={generalNoteText}
            time={generalNoteTime}
            onChangeText={setGeneralNoteText}
            onChangeTime={setGeneralNoteTime}
            onClose={() => setGeneralNoteOpen(false)}
            onSave={() => {
              const [hh, mm] = generalNoteTime.split(':').map(n => parseInt(n));
              const when = new Date(selectedDate);
              when.setHours(hh || 0, mm || 0, 0, 0);
              scheduleBeat('General', generalNoteText || 'General Note', when, '', currentWave?.id);
              setGeneralNoteOpen(false);
            }}
          />
        )}

        {mixerOpen && (
          <DJTab
            deltaHV={deltaHVState}
            onClose={() => {
              // Don't stop music when closing DJ tab - let it continue playing
              setMixerOpen(false);
            }}
          />
        )}

        {glyphCanvasOpen && (
          <BrainRegionChallenge
            deltaHV={deltaHVState}
            onClose={() => setGlyphCanvasOpen(false)}
            onCompleteChallenge={(regionId, xp) => {
              console.log(`Challenge completed: ${regionId} +${xp}XP`);
              // Record action to metrics hub
              metricsHub.recordAction(`challenge_complete_${regionId}`);
            }}
            onCreateBeat={(beat) => {
              // Create a new check-in for the challenge/task
              const entry: CheckIn = {
                id: Date.now().toString() + Math.random(),
                category: beat.category,
                task: beat.task,
                waveId: beat.waveId,
                slot: beat.slot,
                loggedAt: new Date().toISOString(),
                note: beat.note,
                done: false,
                isAnchor: beat.isAnchor,
              };
              setCheckIns(prev => [entry, ...prev]);
              console.log('Beat created for challenge:', beat.task);
            }}
            onNavigateToChallenges={() => setCurrentView('challenges')}
            onNavigateToCosmetics={() => setCurrentView('cosmetics')}
          />
        )}

        {/* Audit Trail Modal (Phase 4) */}
        {auditTrailOpen && (
          <AuditTrailModal onClose={() => setAuditTrailOpen(false)} />
        )}

        {/* Planner Settings Modal (Phase 3 Complete) */}
        {plannerSettingsOpen && (
          <PlannerSettingsModal
            preferences={plannerPreferences}
            gcalConnected={gcalAuthed}
            onClose={() => setPlannerSettingsOpen(false)}
            onSave={(prefs) => {
              plannerRef.current?.updatePreferences(prefs);
              setPlannerSettingsOpen(false);
            }}
          />
        )}

        {/* Notification Settings Modal */}
        {notificationSettingsOpen && (
          <NotificationSettingsModal
            preferences={notificationPrefs}
            permissionStatus={notificationPermission}
            platform={notificationService.getPlatform()}
            onClose={() => setNotificationSettingsOpen(false)}
            onSave={(prefs) => {
              notificationService.updatePreferences(prefs);
              setNotificationPrefs(prefs);
              setNotificationSettingsOpen(false);
            }}
            onRequestPermission={async () => {
              const status = await notificationService.requestPermission();
              setNotificationPermission(status);
            }}
          />
        )}

        {/* Music Library Modal */}
        {musicLibraryOpen && (
          <MusicLibrary
            onClose={() => setMusicLibraryOpen(false)}
            onSelectTrack={(_track, emotion) => {
              setDailyMusicEmotion(emotion);
              setMusicLibraryOpen(false);
            }}
          />
        )}
      </div>
    </div>
  );
}

function getIcon(cat: string) {
  const iconClass = 'w-8 h-8';
  switch (cat) {
    case 'Workout': return <Dumbbell className={`${iconClass} text-pink-400`} />;
    case 'Moderation': return <Shield className={`${iconClass} text-cyan-400`} />;
    case 'Meditation': return <Brain className={`${iconClass} text-green-400`} />;
    case 'Emotion': return <Heart className={`${iconClass} text-purple-400`} />;
    case 'Anchor': return <Zap className={`${iconClass} text-amber-400`} />;
    default: return <Sparkles className={iconClass} />;
  }
}

function RoundButton({ icon, label, onClick }: { icon: React.ReactElement; label: string; onClick: () => void }) {
  return (
    <button onClick={onClick} className="flex flex-col items-center gap-2 group">
      <div className="relative w-20 h-20 rounded-full bg-gradient-to-br from-gray-900 to-gray-950 border-2 border-gray-800 hover:border-cyan-500 flex items-center justify-center transition-all group-hover:scale-105 overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-white/20 via-transparent to-transparent opacity-40" />
        <div className="relative z-10">{icon}</div>
      </div>
      <span className="text-xs text-gray-400 group-hover:text-gray-200">{label}</span>
    </button>
  );
}

function CalendarMonth({ month, selectedDate, deviationDay, checkIns, onPrev, onNext, onSelect }: {
  month: Date; selectedDate: Date; deviationDay?: number; checkIns: CheckIn[]; onPrev:()=>void; onNext:()=>void; onSelect:(d:Date)=>void;
}) {
  const startOfMonth = (d: Date) => new Date(d.getFullYear(), d.getMonth(), 1);
  const endOfMonth = (d: Date) => new Date(d.getFullYear(), d.getMonth()+1, 0);

  const first = startOfMonth(month);
  const last = endOfMonth(month);
  const startWeekday = first.getDay();

  const days: Date[] = [];
  for(let i=0; i<startWeekday; i++) days.push(new Date(first.getFullYear(), first.getMonth(), 1 - (startWeekday - i)));
  for(let d=1; d<=last.getDate(); d++) days.push(new Date(month.getFullYear(), month.getMonth(), d));
  while(days.length % 7 !== 0) days.push(new Date(last.getFullYear(), last.getMonth()+1, days.length % 7));

  const isCurrentMonth = (d:Date) => d.getMonth() === month.getMonth();

  const getBeatsForDay = (d: Date) => {
    return checkIns.filter(c => sameDay(new Date(c.slot), d));
  };

  const getCategoryColor = (category: string): string => {
    const colors: Record<string, string> = {
      'Workout': 'bg-pink-500',
      'Moderation': 'bg-cyan-500',
      'Meditation': 'bg-green-500',
      'Emotion': 'bg-purple-500',
      'Journal': 'bg-emerald-500',
      'Med': 'bg-indigo-500',
      'Anchor': 'bg-amber-500',
      'General': 'bg-gray-500'
    };
    return colors[category] || 'bg-blue-500';
  };

  const getBeatsBreakdown = (beats: CheckIn[]) => {
    const breakdown: Record<string, { upcoming: number; completed: number; color: string }> = {};
    beats.forEach(beat => {
      if (!breakdown[beat.category]) {
        breakdown[beat.category] = { upcoming: 0, completed: 0, color: getCategoryColor(beat.category) };
      }
      if (beat.done) {
        breakdown[beat.category].completed++;
      } else {
        breakdown[beat.category].upcoming++;
      }
    });
    return breakdown;
  };

  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <button onClick={onPrev} className="px-3 py-1.5 rounded-lg bg-gray-900/70 border border-gray-800 hover:bg-gray-800">{'<'}</button>
        <div className="text-sm text-gray-300">{month.toLocaleString(undefined,{ month:'long', year:'numeric' })}</div>
        <button onClick={onNext} className="px-3 py-1.5 rounded-lg bg-gray-900/70 border border-gray-800 hover:bg-gray-800">{'>'}</button>
      </div>

      {/* Legend */}
      <div className="mb-3 p-2 bg-gray-900/40 rounded-lg border border-gray-800">
        <div className="text-xs text-gray-400 mb-1">Beat Categories:</div>
        <div className="flex flex-wrap gap-2 text-xs">
          <div className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-pink-500"></div><span>Workout</span></div>
          <div className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-cyan-500"></div><span>Moderation</span></div>
          <div className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-green-500"></div><span>Meditation</span></div>
          <div className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-purple-500"></div><span>Emotion</span></div>
          <div className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-amber-500"></div><span>Anchor</span></div>
        </div>
      </div>

      <div className="grid grid-cols-7 gap-1 text-center text-xs text-gray-400 mb-1">
        {['Sun','Mon','Tue','Wed','Thu','Fri','Sat'].map(w => <div key={w}>{w}</div>)}
      </div>
      <div className="grid grid-cols-7 gap-1">
        {days.map((d,i)=>{
          const disabled = !isCurrentMonth(d);
          const active = sameDay(d, selectedDate);
          const isDeviation = d.getDay() === deviationDay;
          const dayBeats = getBeatsForDay(d);
          const breakdown = getBeatsBreakdown(dayBeats);

          return (
            <button key={i} onClick={()=>onSelect(d)}
              className={`relative aspect-square rounded-lg border text-sm ${disabled?'text-gray-600 border-gray-800 bg-gray-950/40':'text-gray-200 border-gray-800 bg-gray-900/60 hover:bg-gray-800/60'} ${active?'!border-cyan-500 !bg-cyan-600 !text-black':''} ${isDeviation && !disabled?'!border-orange-500/50':''}`}
            >
              <div className="flex flex-col items-center justify-center h-full p-0.5">
                <span className="font-medium">{d.getDate()}</span>
                {!disabled && dayBeats.length > 0 && (
                  <div className="flex flex-wrap gap-0.5 justify-center mt-0.5 w-full">
                    {Object.entries(breakdown).map(([category, data]) => (
                      <div key={category} className="flex items-center gap-px">
                        {data.upcoming > 0 && (
                          <div
                            className={`w-1.5 h-1.5 rounded-full ${data.color} opacity-60`}
                            title={`${category}: ${data.upcoming} upcoming`}
                          ></div>
                        )}
                        {data.completed > 0 && (
                          <div
                            className={`w-1.5 h-1.5 rounded-full ${data.color}`}
                            title={`${category}: ${data.completed} completed`}
                          ></div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
              {isDeviation && !disabled && <span className="absolute top-0.5 right-0.5 text-orange-400 text-xs">⚡</span>}
            </button>
          );
        })}
      </div>
    </div>
  );
}

function WaveSetupWizard({ profile, onClose, onSave }: any) {
  const [step, setStep] = useState(0);
  const [waves, setWaves] = useState(profile.waves);
  const [deviationDay, setDeviationDay] = useState(profile.deviationDay ?? 6);
  const [wakeTime, setWakeTime] = useState(profile.wakeTime ?? { hours: 8, minutes: 0 });

  // Enhanced wave settings for each wave
  const [waveSettings, setWaveSettings] = useState<Record<string, {
    activities: string[];
    moodPreset: string;
    metricFocus: 'symbolic' | 'resonance' | 'friction' | 'stability';
  }>>(() => {
    const settings: Record<string, any> = {};
    profile.waves.forEach((w: Wave) => {
      settings[w.id] = {
        activities: [],
        moodPreset: w.id === 'focus' ? 'focus' : w.id === 'flow' ? 'energize' : 'calm',
        metricFocus: w.id === 'focus' ? 'symbolic' : w.id === 'flow' ? 'resonance' : 'stability',
      };
    });
    return settings;
  });

  // Get current metrics for recommendations
  const currentMetrics = metricsHub.getState();

  // Activity suggestions per wave type
  const waveActivitySuggestions: Record<string, string[]> = {
    focus: ['Deep work sessions', 'Analysis tasks', 'Strategic planning', 'Learning new skills', 'Important meetings', 'Problem solving'],
    flow: ['Creative projects', 'Brainstorming', 'Social calls', 'Collaboration', 'Exercise', 'Music practice'],
    recovery: ['Journaling', 'Meditation', 'Light reading', 'Meal prep', 'Family time', 'Gentle stretching'],
  };

  // Metric focus recommendations
  const metricDescriptions = {
    symbolic: { name: 'Symbolic (S)', desc: 'Meaning & intention - best for journaling, goal-setting, creative visualization', color: 'purple' },
    resonance: { name: 'Resonance (R)', desc: 'Rhythm alignment - best for social connection, team work, collaborative tasks', color: 'cyan' },
    friction: { name: 'Friction (δφ)', desc: 'Reduce resistance - best for tackling obstacles, clearing backlogs, decisive action', color: 'amber' },
    stability: { name: 'Stability (H)', desc: 'Harmonic balance - best for routines, anchor activities, recovery work', color: 'emerald' },
  };

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="w-full max-w-2xl bg-gray-950 border-2 border-purple-700 rounded-2xl p-6 space-y-6 max-h-[90vh] overflow-y-auto">
        <h2 className="text-2xl font-light text-purple-400 flex items-center gap-3">
          <Waves className="w-8 h-8" /> Wave Setup Wizard
        </h2>

        {/* Progress indicator */}
        <div className="flex gap-2">
          {[0, 1, 2, 3, 4].map(s => (
            <div key={s} className={`flex-1 h-1 rounded-full ${step >= s ? 'bg-purple-500' : 'bg-gray-800'}`} />
          ))}
        </div>

        {step === 0 && (
          <div className="space-y-4">
            <p className="text-gray-300">Your brain flows through three biological waves each day. Let's map YOUR personal rhythm.</p>

            {/* Show current metrics if available */}
            {currentMetrics && (
              <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-4">
                <p className="text-sm text-gray-400 mb-2">Your current DeltaHV state:</p>
                <div className="grid grid-cols-4 gap-2 text-center">
                  <div>
                    <p className="text-purple-400 text-lg font-bold">{currentMetrics.symbolicDensity}</p>
                    <p className="text-[10px] text-gray-500">Symbolic</p>
                  </div>
                  <div>
                    <p className="text-cyan-400 text-lg font-bold">{currentMetrics.resonanceCoupling}</p>
                    <p className="text-[10px] text-gray-500">Resonance</p>
                  </div>
                  <div>
                    <p className="text-amber-400 text-lg font-bold">{currentMetrics.frictionCoefficient}</p>
                    <p className="text-[10px] text-gray-500">Friction</p>
                  </div>
                  <div>
                    <p className="text-emerald-400 text-lg font-bold">{currentMetrics.harmonicStability}</p>
                    <p className="text-[10px] text-gray-500">Stability</p>
                  </div>
                </div>
              </div>
            )}

            <div className="bg-purple-900/20 border border-purple-700/50 rounded-xl p-4 space-y-3">
              <p className="text-sm text-purple-300">Answer these questions to find your rhythm:</p>
              <ul className="text-sm text-gray-300 space-y-2 list-disc list-inside">
                <li>When does your mind feel sharpest?</li>
                <li>When do you get restless?</li>
                <li>When do ideas flow naturally?</li>
                <li>When does your energy fade?</li>
              </ul>
            </div>
            <button onClick={() => setStep(1)} className="w-full px-6 py-3 rounded-xl bg-purple-600 hover:bg-purple-500 text-white font-medium">
              Start Mapping
            </button>
          </div>
        )}

        {step === 1 && (
          <div className="space-y-4">
            <p className="text-gray-300">What time do you usually wake up?</p>
            <div className="bg-cyan-900/20 border border-cyan-700/50 rounded-xl p-4 space-y-3">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-xs text-gray-400">Hour (24h)</label>
                  <input
                    type="number"
                    min={0}
                    max={23}
                    value={wakeTime.hours}
                    onChange={(e) => setWakeTime({ ...wakeTime, hours: parseInt(e.target.value) || 0 })}
                    className="w-full px-3 py-2 rounded-lg bg-gray-900/70 border border-gray-800 text-gray-100 outline-none focus:border-cyan-500"
                  />
                </div>
                <div>
                  <label className="text-xs text-gray-400">Minutes</label>
                  <input
                    type="number"
                    min={0}
                    max={59}
                    value={wakeTime.minutes}
                    onChange={(e) => setWakeTime({ ...wakeTime, minutes: parseInt(e.target.value) || 0 })}
                    className="w-full px-3 py-2 rounded-lg bg-gray-900/70 border border-gray-800 text-gray-100 outline-none focus:border-cyan-500"
                  />
                </div>
              </div>
              <p className="text-xs text-gray-400">Your waves will be calculated from this time</p>
            </div>
            <button onClick={() => setStep(2)} className="w-full px-6 py-3 rounded-xl bg-purple-600 hover:bg-purple-500 text-white font-medium">
              Next: Customize Waves
            </button>
          </div>
        )}

        {step === 2 && (
          <div className="space-y-4">
            <p className="text-gray-300">Customize your three waves (hours since waking):</p>
            {waves.map((wave: Wave, idx: number) => (
              <div key={wave.id} className="bg-gray-900/60 border border-gray-800 rounded-xl p-4 space-y-3">
                <input
                  type="text"
                  value={wave.name}
                  onChange={(e) => {
                    const next = [...waves];
                    next[idx].name = e.target.value;
                    setWaves(next);
                  }}
                  className="w-full px-3 py-2 rounded-lg bg-gray-900/70 border border-gray-800 text-gray-100 outline-none focus:border-purple-500"
                  placeholder="Wave name"
                />
                <textarea
                  value={wave.description}
                  onChange={(e) => {
                    const next = [...waves];
                    next[idx].description = e.target.value;
                    setWaves(next);
                  }}
                  className="w-full px-3 py-2 rounded-lg bg-gray-900/70 border border-gray-800 text-gray-100 outline-none focus:border-purple-500"
                  placeholder="What happens in this wave?"
                  rows={2}
                />
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="text-xs text-gray-400">Start (hours awake)</label>
                    <input
                      type="number"
                      value={wave.startHour}
                      onChange={(e) => {
                        const next = [...waves];
                        next[idx].startHour = parseInt(e.target.value);
                        setWaves(next);
                      }}
                      className="w-full px-3 py-2 rounded-lg bg-gray-900/70 border border-gray-800 text-gray-100 outline-none focus:border-purple-500"
                    />
                  </div>
                  <div>
                    <label className="text-xs text-gray-400">End (hours awake)</label>
                    <input
                      type="number"
                      value={wave.endHour}
                      onChange={(e) => {
                        const next = [...waves];
                        next[idx].endHour = parseInt(e.target.value);
                        setWaves(next);
                      }}
                      className="w-full px-3 py-2 rounded-lg bg-gray-900/70 border border-gray-800 text-gray-100 outline-none focus:border-purple-500"
                    />
                  </div>
                </div>
              </div>
            ))}
            <button onClick={() => setStep(3)} className="w-full px-6 py-3 rounded-xl bg-purple-600 hover:bg-purple-500 text-white font-medium">
              Next: Wave Activities & Metrics
            </button>
          </div>
        )}

        {step === 3 && (
          <div className="space-y-4">
            <p className="text-gray-300">Configure activities and metric focus for each wave:</p>

            {waves.map((wave: Wave) => {
              const colors = waveColorClasses[wave.color];
              const settings = waveSettings[wave.id] || { activities: [], moodPreset: 'balance', metricFocus: 'symbolic' };
              const suggestions = waveActivitySuggestions[wave.id] || waveActivitySuggestions.focus;

              return (
                <div key={wave.id} className={`border-2 ${colors.border} ${colors.bgLight} rounded-xl p-4 space-y-3`}>
                  <p className={`${colors.text} font-medium flex items-center gap-2`}>
                    <Waves className="w-4 h-4" />
                    {wave.name}
                  </p>

                  {/* Metric Focus */}
                  <div>
                    <label className="text-xs text-gray-400 block mb-1">Metric Focus for this wave:</label>
                    <div className="grid grid-cols-2 gap-2">
                      {(Object.entries(metricDescriptions) as [string, {name: string; desc: string; color: string}][]).map(([key, info]) => (
                        <button
                          key={key}
                          onClick={() => setWaveSettings({
                            ...waveSettings,
                            [wave.id]: { ...settings, metricFocus: key as any }
                          })}
                          className={`p-2 rounded-lg border text-left text-xs ${
                            settings.metricFocus === key
                              ? `bg-${info.color}-600/30 border-${info.color}-500 text-${info.color}-300`
                              : 'bg-gray-900/50 border-gray-700 text-gray-400'
                          }`}
                        >
                          <p className="font-medium">{info.name}</p>
                          <p className="text-[10px] opacity-70">{info.desc.split(' - ')[0]}</p>
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Music Mood */}
                  <div>
                    <label className="text-xs text-gray-400 block mb-1">Preferred music mood:</label>
                    <div className="flex flex-wrap gap-2">
                      {['focus', 'energize', 'calm', 'boost', 'heal'].map(mood => (
                        <button
                          key={mood}
                          onClick={() => setWaveSettings({
                            ...waveSettings,
                            [wave.id]: { ...settings, moodPreset: mood }
                          })}
                          className={`px-2 py-1 rounded-lg text-xs ${
                            settings.moodPreset === mood
                              ? 'bg-purple-600 text-white'
                              : 'bg-gray-800 text-gray-400'
                          }`}
                        >
                          {mood === 'focus' ? '🎯' : mood === 'energize' ? '⚡' : mood === 'calm' ? '🌊' : mood === 'boost' ? '🚀' : '💜'} {mood}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Activity Suggestions */}
                  <div>
                    <label className="text-xs text-gray-400 block mb-1">Suggested activities (tap to add):</label>
                    <div className="flex flex-wrap gap-1">
                      {suggestions.map(activity => (
                        <button
                          key={activity}
                          onClick={() => {
                            const current = settings.activities || [];
                            if (current.includes(activity)) {
                              setWaveSettings({
                                ...waveSettings,
                                [wave.id]: { ...settings, activities: current.filter(a => a !== activity) }
                              });
                            } else {
                              setWaveSettings({
                                ...waveSettings,
                                [wave.id]: { ...settings, activities: [...current, activity] }
                              });
                            }
                          }}
                          className={`px-2 py-0.5 rounded text-[10px] ${
                            settings.activities?.includes(activity)
                              ? 'bg-emerald-600/50 text-emerald-300'
                              : 'bg-gray-800/50 text-gray-500'
                          }`}
                        >
                          {activity}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              );
            })}

            <div className="flex gap-3">
              <button onClick={() => setStep(2)} className="flex-1 px-6 py-3 rounded-xl bg-gray-800 hover:bg-gray-700 text-white">
                Back
              </button>
              <button onClick={() => setStep(4)} className="flex-1 px-6 py-3 rounded-xl bg-purple-600 hover:bg-purple-500 text-white font-medium">
                Next: Deviation Day
              </button>
            </div>
          </div>
        )}

        {step === 4 && (
          <div className="space-y-4">
            <p className="text-gray-300">Pick your weekly Deviation Day (planned chaos = better discipline):</p>
            <div className="grid grid-cols-7 gap-2">
              {['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'].map((day, idx) => (
                <button
                  key={idx}
                  onClick={() => setDeviationDay(idx)}
                  className={`px-3 py-2 rounded-lg border text-sm ${deviationDay === idx ? 'bg-orange-600 border-orange-500 text-white' : 'bg-gray-900/60 border-gray-800 text-gray-300'}`}
                >
                  {day}
                </button>
              ))}
            </div>
            <p className="text-xs text-gray-400 italic">On deviation days, the UI changes to orange and removes guilt from missing anchors</p>

            {/* Summary */}
            <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-4 space-y-2">
              <p className="text-sm font-medium text-gray-300">Setup Summary:</p>
              <div className="text-xs text-gray-400 space-y-1">
                <p>⏰ Wake time: {String(wakeTime.hours).padStart(2, '0')}:{String(wakeTime.minutes).padStart(2, '0')}</p>
                <p>🌊 {waves.length} waves configured</p>
                <p>🎲 Deviation day: {['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'][deviationDay]}</p>
              </div>
              {waves.map((w: Wave) => (
                <div key={w.id} className="text-xs text-gray-500">
                  • {w.name}: {waveSettings[w.id]?.metricFocus || 'symbolic'} focus, {waveSettings[w.id]?.moodPreset || 'balance'} music
                </div>
              ))}
            </div>

            <div className="flex gap-3">
              <button onClick={() => setStep(3)} className="flex-1 px-6 py-3 rounded-xl bg-gray-800 hover:bg-gray-700 text-white">
                Back
              </button>
              <button
                onClick={() => {
                  // Enhance waves with settings
                  const enhancedWaves = waves.map((w: Wave) => ({
                    ...w,
                    metricFocus: waveSettings[w.id]?.metricFocus,
                    moodPreset: waveSettings[w.id]?.moodPreset,
                    activities: waveSettings[w.id]?.activities,
                  }));
                  onSave({
                    waves: enhancedWaves,
                    deviationDay,
                    wakeTime,
                    waveSettings,
                    setupComplete: true
                  });
                }}
                className="flex-1 px-6 py-3 rounded-xl bg-purple-600 hover:bg-purple-500 text-white font-medium"
              >
                Complete Setup
              </button>
            </div>
          </div>
        )}

        <button onClick={onClose} className="absolute top-4 right-4 p-2 rounded-lg bg-gray-900/70 border border-gray-800 hover:bg-gray-800">
          <X className="w-5 h-5" />
        </button>
      </div>
    </div>
  );
}

function WaveAnchorsModal({ waves, selectedDate, checkIns, onClose, onSetAnchor }: any) {
  const [anchors, setAnchors] = useState<Record<string, {task: string; note: string}>>({});

  useEffect(() => {
    const existing: Record<string, {task: string; note: string}> = {};
    waves.forEach((w: Wave) => {
      const anchor = checkIns.find((c: CheckIn) => c.isAnchor && c.waveId === w.id && sameDay(new Date(c.slot), selectedDate));
      if (anchor) {
        existing[w.id] = { task: anchor.task, note: anchor.note || '' };
      } else {
        existing[w.id] = { task: '', note: '' };
      }
    });
    setAnchors(existing);
  }, [waves, checkIns, selectedDate]);

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="w-full max-w-xl bg-gray-950 border border-gray-800 rounded-2xl p-6 space-y-6">
        <h2 className="text-xl font-light flex items-center gap-3">
          <Zap className="w-6 h-6 text-amber-400" /> Set Daily Anchors
        </h2>
        <p className="text-sm text-gray-400">One non-negotiable activity per wave to keep you stable</p>

        {waves.map((wave: Wave) => {
          const colors = waveColorClasses[wave.color];
          return (
            <div key={wave.id} className={`border-2 ${colors.border} ${colors.bgLight} rounded-xl p-4 space-y-3`}>
              <p className={`${colors.text} font-medium`}>{wave.name}</p>
              <input
                type="text"
                value={anchors[wave.id]?.task || ''}
                onChange={(e) => setAnchors({...anchors, [wave.id]: {...anchors[wave.id], task: e.target.value}})}
                placeholder="e.g., Morning workout, Creative work, Evening walk"
                className="w-full px-3 py-2 rounded-lg bg-gray-900/70 border border-gray-800 text-gray-100 outline-none focus:border-cyan-500"
              />
              <input
                type="text"
                value={anchors[wave.id]?.note || ''}
                onChange={(e) => setAnchors({...anchors, [wave.id]: {...anchors[wave.id], note: e.target.value}})}
                placeholder="Optional note"
                className="w-full px-3 py-2 rounded-lg bg-gray-900/70 border border-gray-800 text-gray-100 outline-none focus:border-cyan-500"
              />
            </div>
          );
        })}

        <div className="flex gap-3">
          <button onClick={onClose} className="flex-1 px-6 py-3 rounded-xl bg-gray-800 hover:bg-gray-700 text-white">
            Cancel
          </button>
          <button
            onClick={() => {
              Object.entries(anchors).forEach(([waveId, data]) => {
                if (data.task) {
                  onSetAnchor(waveId, data.task, data.note);
                }
              });
            }}
            className="flex-1 px-6 py-3 rounded-xl bg-amber-600 hover:bg-amber-500 text-black font-medium"
          >
            Save Anchors
          </button>
        </div>
      </div>
    </div>
  );
}

function MiniScheduler({ date, category, presetTasks, time, note, waveId, waves, availableTracks, selectedTrack, emotionalCategories, onChange, onClose, onSubmit }: any) {
  const [selectedTask, setSelectedTask] = useState<string>('');
  const [showTrackPicker, setShowTrackPicker] = useState(false);

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-end md:items-center justify-center z-50 p-4">
      <div className="w-full max-w-md bg-gray-950 border border-gray-800 rounded-2xl p-5 space-y-3 max-h-[90vh] overflow-y-auto">
        <div className="flex items-center justify-between">
          <div className="text-sm text-gray-400">{date.toDateString()}</div>
          <button onClick={onClose} className="px-2 py-1 rounded-lg bg-gray-900/70 border border-gray-800 hover:bg-gray-800">Close</button>
        </div>

        <div className="text-lg font-medium">{category}</div>

        <div>
          <label className="text-xs text-gray-400 mb-1 block">Assign to Wave</label>
          <select
            value={waveId}
            onChange={(e) => onChange({ waveId: e.target.value })}
            className="w-full px-3 py-2 rounded-xl bg-gray-900/70 border border-gray-800 text-gray-100 outline-none focus:border-cyan-500"
          >
            <option value="">No wave</option>
            {waves.map((w: Wave) => (
              <option key={w.id} value={w.id}>{w.name}</option>
            ))}
          </select>
        </div>

        <div className="grid grid-cols-2 gap-2 max-h-64 overflow-y-auto">
          {presetTasks.map((t: string) => (
            <button key={t} onClick={() => { setSelectedTask(t); onChange({ task: t }); }}
              className={`px-3 py-2 rounded-xl border text-left text-sm ${selectedTask === t ? 'bg-cyan-900/50 border-cyan-500' : 'bg-gray-900/60 border-gray-800 hover:bg-gray-800'}`}
            >{t}</button>
          ))}
        </div>

        <input value={time} onChange={(e) => onChange({ time: e.target.value })} type="time" className="w-full px-3 py-2 rounded-xl bg-gray-900/70 border border-gray-800 outline-none focus:border-cyan-500" />
        <input onChange={(e) => onChange({ task: e.target.value })} placeholder="or type custom task..." className="w-full px-3 py-2 rounded-xl bg-gray-900/70 border border-gray-800 outline-none focus:border-cyan-500" />
        <textarea value={note} onChange={(e) => onChange({ note: e.target.value })} placeholder="note (optional)" className="w-full min-h-[64px] px-3 py-2 rounded-xl bg-gray-900/70 border border-gray-800 outline-none focus:border-cyan-500" />

        {/* Music Selection */}
        <div>
          <label className="text-xs text-gray-400 mb-1 block">🎵 Add Music (optional)</label>
          {selectedTrack ? (
            <div className="flex items-center justify-between px-3 py-2 rounded-xl bg-purple-900/30 border border-purple-500/50">
              <div className="flex items-center gap-2">
                <span>{emotionalCategories?.[selectedTrack.emotion]?.icon || '🎵'}</span>
                <span className="text-sm text-purple-200">{selectedTrack.name}</span>
                <span className="text-xs text-purple-400">({emotionalCategories?.[selectedTrack.emotion]?.name || selectedTrack.emotion})</span>
              </div>
              <button
                onClick={() => onChange({ track: null })}
                className="text-xs text-red-400 hover:text-red-300"
              >
                Remove
              </button>
            </div>
          ) : (
            <button
              onClick={() => setShowTrackPicker(!showTrackPicker)}
              className="w-full px-3 py-2 rounded-xl bg-gray-900/70 border border-gray-800 hover:border-purple-500/50 text-left text-sm text-gray-400 hover:text-purple-300 transition-colors"
            >
              {showTrackPicker ? 'Hide tracks...' : 'Select a track for this beat...'}
            </button>
          )}

          {showTrackPicker && !selectedTrack && availableTracks && availableTracks.length > 0 && (
            <div className="mt-2 max-h-40 overflow-y-auto space-y-1">
              {availableTracks.map((track: any) => (
                <button
                  key={track.id}
                  onClick={() => {
                    onChange({ track: { id: track.id, name: track.name, emotion: track.categoryId } });
                    setShowTrackPicker(false);
                  }}
                  className="w-full px-3 py-2 rounded-lg bg-gray-900/60 border border-gray-800 hover:border-purple-500/50 text-left flex items-center gap-2"
                >
                  <span>{emotionalCategories?.[track.categoryId]?.icon || '🎵'}</span>
                  <span className="text-sm text-gray-200 flex-1">{track.name}</span>
                  <span className="text-xs text-gray-500">{emotionalCategories?.[track.categoryId]?.name || track.categoryId}</span>
                </button>
              ))}
            </div>
          )}

          {showTrackPicker && (!availableTracks || availableTracks.length === 0) && (
            <div className="mt-2 text-xs text-gray-500 text-center py-2">
              No tracks in library. Add music via the Music button above.
            </div>
          )}
        </div>

        <div className="flex justify-end">
          <button onClick={onSubmit} className="px-4 py-2 rounded-xl bg-amber-500 text-black hover:bg-amber-400">Schedule</button>
        </div>
      </div>
    </div>
  );
}

function JournalModal({ date, value, waveId, waves, onChange, onWaveChange, onClose, onSave }: any) {
  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-end md:items-center justify-center z-50 p-4">
      <div className="w-full max-w-xl bg-gray-950 border border-gray-800 rounded-2xl p-5 space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <NotebookPen className="w-5 h-5 text-emerald-400" />
            <span className="text-sm text-gray-400">{date.toDateString()}</span>
          </div>
          <button onClick={onClose} className="px-2 py-1 rounded-lg bg-gray-900/70 border border-gray-800 hover:bg-gray-800">Close</button>
        </div>

        <div>
          <label className="text-xs text-gray-400 mb-1 block">Tag with Wave</label>
          <select
            value={waveId}
            onChange={(e) => onWaveChange(e.target.value)}
            className="w-full px-3 py-2 rounded-xl bg-gray-900/70 border border-gray-800 text-gray-100 outline-none focus:border-cyan-500"
          >
            <option value="">No wave</option>
            {waves.map((w: Wave) => (
              <option key={w.id} value={w.id}>{w.name}</option>
            ))}
          </select>
        </div>

        <textarea value={value} onChange={(e) => onChange(e.target.value)} placeholder="Write your entry..." className="w-full min-h-[220px] px-3 py-2 rounded-xl bg-gray-900/70 border border-gray-800 outline-none focus:border-cyan-500" />

        <div className="flex justify-end">
          <button onClick={onSave} className="px-4 py-2 rounded-xl bg-emerald-500 text-black hover:bg-emerald-400">Save Entry</button>
        </div>
      </div>
    </div>
  );
}

function GeneralNoteModal({ date, text, time, onChangeText, onChangeTime, onClose, onSave }: any) {
  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-end md:items-center justify-center z-50 p-4">
      <div className="w-full max-w-md bg-gray-950 border border-gray-800 rounded-2xl p-5 space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-2xl">∞</span>
            <span className="text-sm text-gray-400">General Note - {date.toDateString()}</span>
          </div>
          <button onClick={onClose} className="px-2 py-1 rounded-lg bg-gray-900/70 border border-gray-800 hover:bg-gray-800">Close</button>
        </div>

        <input type="text" value={text} onChange={(e) => onChangeText(e.target.value)} placeholder="e.g., Dinner at 7pm, Call dentist, etc." className="w-full px-3 py-2 rounded-xl bg-gray-900/70 border border-gray-800 outline-none focus:border-cyan-500" />
        <input type="time" value={time} onChange={(e) => onChangeTime(e.target.value)} className="w-full px-3 py-2 rounded-xl bg-gray-900/70 border border-gray-800 outline-none focus:border-cyan-500" />

        <div className="flex justify-end">
          <button onClick={onSave} className="px-4 py-2 rounded-xl bg-amber-500 text-black hover:bg-amber-400">Schedule</button>
        </div>
      </div>
    </div>
  );
}

// ToneMixer and GlyphCanvas have been replaced by DJTab and BrainRegionChallenge components

/**
 * Audit Trail Modal (Phase 4)
 * Displays the audit log with filtering and export capabilities
 */
function AuditTrailModal({ onClose }: { onClose: () => void }) {
  const [entries, setEntries] = useState<any[]>([]);
  const [filter, setFilter] = useState<string>('all');
  const [stats, setStats] = useState<any>(null);

  useEffect(() => {
    // Load entries on mount
    const loadEntries = () => {
      const allEntries = auditLog.getRecentEntries(100);
      setEntries(allEntries);
      setStats(auditLog.getStats());
    };
    loadEntries();

    // Subscribe to new entries
    const unsubscribe = auditLog.subscribe((entry) => {
      setEntries(prev => [...prev.slice(-99), entry]);
    });

    return unsubscribe;
  }, []);

  const filteredEntries = filter === 'all'
    ? entries
    : entries.filter(e => e.severity === filter || e.action.includes(filter.toUpperCase()));

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'success': return 'text-emerald-400 bg-emerald-950/30';
      case 'warning': return 'text-amber-400 bg-amber-950/30';
      case 'error': return 'text-rose-400 bg-rose-950/30';
      default: return 'text-gray-400 bg-gray-900/30';
    }
  };

  const getActionIcon = (action: string) => {
    if (action.includes('RHYTHM')) return '🌊';
    if (action.includes('CHECKIN')) return '✅';
    if (action.includes('JOURNAL')) return '📓';
    if (action.includes('DELTAHV')) return '⚡';
    if (action.includes('GCAL')) return '📅';
    if (action.includes('ANCHOR')) return '⚓';
    if (action.includes('ERROR')) return '❌';
    if (action.includes('SYSTEM')) return '⚙️';
    return '📝';
  };

  const exportLog = () => {
    const json = auditLog.exportAsJSON();
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `audit-log-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="w-full max-w-4xl h-[80vh] bg-gray-950 border border-gray-800 rounded-2xl flex flex-col overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-800">
          <div className="flex items-center gap-3">
            <FileText className="w-6 h-6 text-purple-400" />
            <div>
              <h2 className="text-xl font-light">Audit Trail</h2>
              <p className="text-xs text-gray-500">
                {stats ? `${stats.totalEntries} total entries` : 'Loading...'}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={exportLog}
              className="px-3 py-1.5 rounded-lg bg-purple-600 hover:bg-purple-500 text-sm flex items-center gap-2"
            >
              <Download className="w-4 h-4" />
              Export
            </button>
            <button
              onClick={onClose}
              className="p-2 rounded-lg bg-gray-900/70 border border-gray-800 hover:bg-gray-800"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Filters */}
        <div className="flex items-center gap-2 p-3 border-b border-gray-800 bg-gray-900/30">
          <span className="text-xs text-gray-400">Filter:</span>
          {['all', 'success', 'warning', 'error', 'RHYTHM', 'CHECKIN', 'DELTAHV'].map(f => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-2 py-1 rounded text-xs ${
                filter === f
                  ? 'bg-purple-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
              }`}
            >
              {f}
            </button>
          ))}
        </div>

        {/* Stats Summary */}
        {stats && (
          <div className="flex items-center gap-4 p-3 border-b border-gray-800 bg-gray-900/20 text-xs">
            <div className="flex items-center gap-1">
              <span className="text-emerald-400">●</span>
              <span className="text-gray-400">{stats.bySeverity.success || 0} success</span>
            </div>
            <div className="flex items-center gap-1">
              <span className="text-amber-400">●</span>
              <span className="text-gray-400">{stats.bySeverity.warning || 0} warning</span>
            </div>
            <div className="flex items-center gap-1">
              <span className="text-rose-400">●</span>
              <span className="text-gray-400">{stats.bySeverity.error || 0} error</span>
            </div>
            <div className="flex items-center gap-1">
              <span className="text-gray-400">●</span>
              <span className="text-gray-400">{stats.bySeverity.info || 0} info</span>
            </div>
          </div>
        )}

        {/* Log Entries */}
        <div className="flex-1 overflow-y-auto p-3 space-y-2">
          {filteredEntries.length === 0 ? (
            <p className="text-gray-500 text-center py-8">No entries match the current filter</p>
          ) : (
            [...filteredEntries].reverse().map((entry) => (
              <div
                key={entry.id}
                className={`rounded-lg p-3 border border-gray-800 ${getSeverityColor(entry.severity)}`}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="flex items-start gap-2">
                    <span className="text-lg">{getActionIcon(entry.action)}</span>
                    <div>
                      <p className="text-sm font-medium">{entry.message}</p>
                      <p className="text-xs text-gray-500 mt-0.5">
                        {entry.action} • {new Date(entry.timestamp).toLocaleTimeString()}
                      </p>
                    </div>
                  </div>
                  <span className={`text-xs px-2 py-0.5 rounded ${
                    entry.severity === 'success' ? 'bg-emerald-900/50 text-emerald-300' :
                    entry.severity === 'warning' ? 'bg-amber-900/50 text-amber-300' :
                    entry.severity === 'error' ? 'bg-rose-900/50 text-rose-300' :
                    'bg-gray-800 text-gray-400'
                  }`}>
                    {entry.severity}
                  </span>
                </div>
                {entry.details && Object.keys(entry.details).length > 0 && (
                  <div className="mt-2 text-xs text-gray-500 bg-black/20 rounded p-2 font-mono">
                    {JSON.stringify(entry.details, null, 2).substring(0, 200)}
                    {JSON.stringify(entry.details).length > 200 && '...'}
                  </div>
                )}
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}

/**
 * Planner Settings Modal (Phase 3 Complete)
 * Configure AI planner behavior and auto-scheduling preferences
 */
function PlannerSettingsModal({
  preferences,
  gcalConnected,
  onClose,
  onSave
}: {
  preferences: PlannerPreferences;
  gcalConnected: boolean;
  onClose: () => void;
  onSave: (prefs: PlannerPreferences) => void;
}) {
  const [localPrefs, setLocalPrefs] = useState<PlannerPreferences>(preferences);

  const updatePref = <K extends keyof PlannerPreferences>(key: K, value: PlannerPreferences[K]) => {
    setLocalPrefs(prev => ({ ...prev, [key]: value }));
  };

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="w-full max-w-lg bg-gray-950 border border-gray-800 rounded-2xl overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-800">
          <div className="flex items-center gap-3">
            <Settings className="w-6 h-6 text-purple-400" />
            <div>
              <h2 className="text-xl font-light">AI Planner Settings</h2>
              <p className="text-xs text-gray-500">Configure rhythm optimization behavior</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg bg-gray-900/70 border border-gray-800 hover:bg-gray-800"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-4 space-y-6 max-h-[70vh] overflow-y-auto">
          {/* Auto-Scheduling Section */}
          <div className="space-y-4">
            <h3 className="text-sm font-medium text-gray-300 flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-purple-400" />
              Auto-Scheduling
            </h3>

            {!gcalConnected && (
              <div className="rounded-lg bg-amber-950/30 border border-amber-700/40 p-3 text-sm text-amber-300">
                <AlertTriangle className="w-4 h-4 inline mr-2" />
                Connect Google Calendar to enable auto-scheduling features
              </div>
            )}

            <label className="flex items-center justify-between p-3 rounded-lg bg-gray-900/50 border border-gray-800 cursor-pointer hover:bg-gray-900/70">
              <div>
                <p className="text-sm font-medium">Enable Auto-Scheduling</p>
                <p className="text-xs text-gray-500">Automatically create calendar events for suggestions</p>
              </div>
              <input
                type="checkbox"
                checked={localPrefs.autoScheduleEnabled}
                onChange={(e) => updatePref('autoScheduleEnabled', e.target.checked)}
                disabled={!gcalConnected}
                className="rounded w-5 h-5"
              />
            </label>

            <label className="flex items-center justify-between p-3 rounded-lg bg-gray-900/50 border border-gray-800 cursor-pointer hover:bg-gray-900/70">
              <div>
                <p className="text-sm font-medium">Auto-Schedule Breaks</p>
                <p className="text-xs text-gray-500">Automatically add breaks after extended focus</p>
              </div>
              <input
                type="checkbox"
                checked={localPrefs.autoScheduleBreaks}
                onChange={(e) => updatePref('autoScheduleBreaks', e.target.checked)}
                disabled={!gcalConnected || !localPrefs.autoScheduleEnabled}
                className="rounded w-5 h-5"
              />
            </label>

            <label className="flex items-center justify-between p-3 rounded-lg bg-gray-900/50 border border-gray-800 cursor-pointer hover:bg-gray-900/70">
              <div>
                <p className="text-sm font-medium">Auto-Schedule Focus Blocks</p>
                <p className="text-xs text-gray-500">Automatically create focus blocks when suggested</p>
              </div>
              <input
                type="checkbox"
                checked={localPrefs.autoScheduleFocusBlocks}
                onChange={(e) => updatePref('autoScheduleFocusBlocks', e.target.checked)}
                disabled={!gcalConnected || !localPrefs.autoScheduleEnabled}
                className="rounded w-5 h-5"
              />
            </label>

            <label className="flex items-center justify-between p-3 rounded-lg bg-gray-900/50 border border-gray-800 cursor-pointer hover:bg-gray-900/70">
              <div>
                <p className="text-sm font-medium">Require Confirmation</p>
                <p className="text-xs text-gray-500">Ask before auto-scheduling events</p>
              </div>
              <input
                type="checkbox"
                checked={localPrefs.requireConfirmation}
                onChange={(e) => updatePref('requireConfirmation', e.target.checked)}
                disabled={!gcalConnected || !localPrefs.autoScheduleEnabled}
                className="rounded w-5 h-5"
              />
            </label>
          </div>

          {/* Timing Settings */}
          <div className="space-y-4">
            <h3 className="text-sm font-medium text-gray-300 flex items-center gap-2">
              <Clock className="w-4 h-4 text-cyan-400" />
              Timing Settings
            </h3>

            <div className="grid grid-cols-2 gap-3">
              <div className="p-3 rounded-lg bg-gray-900/50 border border-gray-800">
                <label className="text-xs text-gray-400">Focus Break Threshold (min)</label>
                <input
                  type="number"
                  min="30"
                  max="180"
                  value={localPrefs.focusBreakThreshold}
                  onChange={(e) => updatePref('focusBreakThreshold', parseInt(e.target.value) || 90)}
                  className="w-full mt-1 px-2 py-1 rounded bg-gray-800 border border-gray-700 text-sm"
                />
              </div>

              <div className="p-3 rounded-lg bg-gray-900/50 border border-gray-800">
                <label className="text-xs text-gray-400">Default Break Duration (min)</label>
                <input
                  type="number"
                  min="5"
                  max="60"
                  value={localPrefs.defaultBreakDuration}
                  onChange={(e) => updatePref('defaultBreakDuration', parseInt(e.target.value) || 15)}
                  className="w-full mt-1 px-2 py-1 rounded bg-gray-800 border border-gray-700 text-sm"
                />
              </div>

              <div className="p-3 rounded-lg bg-gray-900/50 border border-gray-800">
                <label className="text-xs text-gray-400">Default Focus Block (min)</label>
                <input
                  type="number"
                  min="25"
                  max="120"
                  value={localPrefs.defaultFocusBlockDuration}
                  onChange={(e) => updatePref('defaultFocusBlockDuration', parseInt(e.target.value) || 50)}
                  className="w-full mt-1 px-2 py-1 rounded bg-gray-800 border border-gray-700 text-sm"
                />
              </div>

              <div className="p-3 rounded-lg bg-gray-900/50 border border-gray-800">
                <label className="text-xs text-gray-400">Working Hours</label>
                <div className="flex items-center gap-1 mt-1">
                  <input
                    type="number"
                    min="0"
                    max="23"
                    value={localPrefs.workingHoursStart}
                    onChange={(e) => updatePref('workingHoursStart', parseInt(e.target.value) || 9)}
                    className="w-14 px-2 py-1 rounded bg-gray-800 border border-gray-700 text-sm"
                  />
                  <span className="text-gray-500">-</span>
                  <input
                    type="number"
                    min="0"
                    max="23"
                    value={localPrefs.workingHoursEnd}
                    onChange={(e) => updatePref('workingHoursEnd', parseInt(e.target.value) || 18)}
                    className="w-14 px-2 py-1 rounded bg-gray-800 border border-gray-700 text-sm"
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Detection Settings */}
          <div className="space-y-4">
            <h3 className="text-sm font-medium text-gray-300 flex items-center gap-2">
              <Activity className="w-4 h-4 text-amber-400" />
              Detection & Notifications
            </h3>

            <label className="flex items-center justify-between p-3 rounded-lg bg-gray-900/50 border border-gray-800 cursor-pointer hover:bg-gray-900/70">
              <div>
                <p className="text-sm font-medium">Friction Detection</p>
                <p className="text-xs text-gray-500">Warn when multiple tasks are overdue</p>
              </div>
              <input
                type="checkbox"
                checked={localPrefs.enableFrictionDetection}
                onChange={(e) => updatePref('enableFrictionDetection', e.target.checked)}
                className="rounded w-5 h-5"
              />
            </label>

            <label className="flex items-center justify-between p-3 rounded-lg bg-gray-900/50 border border-gray-800 cursor-pointer hover:bg-gray-900/70">
              <div>
                <p className="text-sm font-medium">Planner Notifications</p>
                <p className="text-xs text-gray-500">Show suggestions and reminders</p>
              </div>
              <input
                type="checkbox"
                checked={localPrefs.notificationsEnabled}
                onChange={(e) => updatePref('notificationsEnabled', e.target.checked)}
                className="rounded w-5 h-5"
              />
            </label>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between p-4 border-t border-gray-800 bg-gray-900/30">
          <button
            onClick={() => setLocalPrefs(DEFAULT_PREFERENCES)}
            className="px-3 py-1.5 rounded-lg bg-gray-800 hover:bg-gray-700 text-sm text-gray-400"
          >
            Reset to Defaults
          </button>
          <div className="flex gap-2">
            <button
              onClick={onClose}
              className="px-4 py-2 rounded-lg bg-gray-800 hover:bg-gray-700 text-sm"
            >
              Cancel
            </button>
            <button
              onClick={() => onSave(localPrefs)}
              className="px-4 py-2 rounded-lg bg-purple-600 hover:bg-purple-500 text-sm font-medium"
            >
              Save Settings
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * Notification Settings Modal
 * Configure notification preferences and request permissions
 */
function NotificationSettingsModal({
  preferences,
  permissionStatus,
  platform,
  onClose,
  onSave,
  onRequestPermission
}: {
  preferences: NotificationPreferences;
  permissionStatus: PermissionStatus;
  platform: { isIOS: boolean; isAndroid: boolean; isPWA: boolean; supportsNotifications: boolean };
  onClose: () => void;
  onSave: (prefs: NotificationPreferences) => void;
  onRequestPermission: () => Promise<void>;
}) {
  const [localPrefs, setLocalPrefs] = useState<NotificationPreferences>(preferences);
  const [requesting, setRequesting] = useState(false);

  const updatePref = <K extends keyof NotificationPreferences>(key: K, value: NotificationPreferences[K]) => {
    setLocalPrefs(prev => ({ ...prev, [key]: value }));
  };

  const handleRequestPermission = async () => {
    setRequesting(true);
    await onRequestPermission();
    setRequesting(false);
  };

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="w-full max-w-lg bg-gray-950 border border-gray-800 rounded-2xl overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-800">
          <div className="flex items-center gap-3">
            <Bell className="w-6 h-6 text-amber-400" />
            <div>
              <h2 className="text-xl font-light">Notification Settings</h2>
              <p className="text-xs text-gray-500">Configure alerts and reminders</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg bg-gray-900/70 border border-gray-800 hover:bg-gray-800"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-4 space-y-6 max-h-[70vh] overflow-y-auto">
          {/* Permission Status */}
          <div className="space-y-3">
            <h3 className="text-sm font-medium text-gray-300">Permission Status</h3>

            {permissionStatus === 'granted' && (
              <div className="rounded-lg bg-emerald-950/30 border border-emerald-700/40 p-3 text-sm text-emerald-300 flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4" />
                Notifications enabled
              </div>
            )}

            {permissionStatus === 'denied' && (
              <div className="rounded-lg bg-rose-950/30 border border-rose-700/40 p-3 text-sm text-rose-300">
                <AlertTriangle className="w-4 h-4 inline mr-2" />
                Notifications blocked
                <p className="text-xs mt-2 text-gray-400">
                  {platform.isIOS
                    ? notificationService.getIOSInstructions()
                    : platform.isAndroid
                    ? notificationService.getAndroidInstructions()
                    : 'To enable, click the lock icon in your browser\'s address bar and allow notifications.'}
                </p>
              </div>
            )}

            {permissionStatus === 'default' && (
              <div className="space-y-3">
                <div className="rounded-lg bg-amber-950/30 border border-amber-700/40 p-3 text-sm text-amber-300">
                  <Bell className="w-4 h-4 inline mr-2" />
                  Notifications not yet enabled
                  {platform.isIOS && !platform.isPWA && (
                    <p className="text-xs mt-2 text-gray-400">
                      {notificationService.getIOSInstructions()}
                    </p>
                  )}
                </div>
                <button
                  onClick={handleRequestPermission}
                  disabled={requesting}
                  className="w-full px-4 py-3 rounded-lg bg-amber-600 hover:bg-amber-500 disabled:opacity-50 text-sm font-medium flex items-center justify-center gap-2"
                >
                  {requesting ? (
                    <>
                      <Loader2 className="w-4 h-4 animate-spin" />
                      Requesting...
                    </>
                  ) : (
                    <>
                      <Bell className="w-4 h-4" />
                      Enable Notifications
                    </>
                  )}
                </button>
              </div>
            )}

            {permissionStatus === 'unsupported' && (
              <div className="rounded-lg bg-gray-900/50 border border-gray-700 p-3 text-sm text-gray-400">
                <AlertTriangle className="w-4 h-4 inline mr-2" />
                Notifications not supported on this browser/device
              </div>
            )}
          </div>

          {/* Notification Types */}
          <div className="space-y-4">
            <h3 className="text-sm font-medium text-gray-300 flex items-center gap-2">
              <Settings className="w-4 h-4 text-gray-400" />
              Notification Types
            </h3>

            <label className="flex items-center justify-between p-3 rounded-lg bg-gray-900/50 border border-gray-800 cursor-pointer hover:bg-gray-900/70">
              <div>
                <p className="text-sm font-medium">Master Toggle</p>
                <p className="text-xs text-gray-500">Enable/disable all notifications</p>
              </div>
              <input
                type="checkbox"
                checked={localPrefs.enabled}
                onChange={(e) => updatePref('enabled', e.target.checked)}
                className="rounded w-5 h-5"
              />
            </label>

            <label className="flex items-center justify-between p-3 rounded-lg bg-gray-900/50 border border-gray-800 cursor-pointer hover:bg-gray-900/70">
              <div>
                <p className="text-sm font-medium">⚓ Anchor Reminders</p>
                <p className="text-xs text-gray-500">Alerts before scheduled anchors</p>
              </div>
              <input
                type="checkbox"
                checked={localPrefs.anchorReminders}
                onChange={(e) => updatePref('anchorReminders', e.target.checked)}
                disabled={!localPrefs.enabled}
                className="rounded w-5 h-5"
              />
            </label>

            <label className="flex items-center justify-between p-3 rounded-lg bg-gray-900/50 border border-gray-800 cursor-pointer hover:bg-gray-900/70">
              <div>
                <p className="text-sm font-medium">💤 Break Reminders</p>
                <p className="text-xs text-gray-500">Alerts to take breaks after focus</p>
              </div>
              <input
                type="checkbox"
                checked={localPrefs.breakReminders}
                onChange={(e) => updatePref('breakReminders', e.target.checked)}
                disabled={!localPrefs.enabled}
                className="rounded w-5 h-5"
              />
            </label>

            <label className="flex items-center justify-between p-3 rounded-lg bg-gray-900/50 border border-gray-800 cursor-pointer hover:bg-gray-900/70">
              <div>
                <p className="text-sm font-medium">🎯 Focus Alerts</p>
                <p className="text-xs text-gray-500">Notifications for focus sessions</p>
              </div>
              <input
                type="checkbox"
                checked={localPrefs.focusAlerts}
                onChange={(e) => updatePref('focusAlerts', e.target.checked)}
                disabled={!localPrefs.enabled}
                className="rounded w-5 h-5"
              />
            </label>

            <label className="flex items-center justify-between p-3 rounded-lg bg-gray-900/50 border border-gray-800 cursor-pointer hover:bg-gray-900/70">
              <div>
                <p className="text-sm font-medium">⚡ Friction Warnings</p>
                <p className="text-xs text-gray-500">Alerts when tasks are overdue</p>
              </div>
              <input
                type="checkbox"
                checked={localPrefs.frictionWarnings}
                onChange={(e) => updatePref('frictionWarnings', e.target.checked)}
                disabled={!localPrefs.enabled}
                className="rounded w-5 h-5"
              />
            </label>

            <label className="flex items-center justify-between p-3 rounded-lg bg-gray-900/50 border border-gray-800 cursor-pointer hover:bg-gray-900/70">
              <div>
                <p className="text-sm font-medium">📅 Auto-Scheduled Events</p>
                <p className="text-xs text-gray-500">Alerts for auto-scheduled calendar events</p>
              </div>
              <input
                type="checkbox"
                checked={localPrefs.autoScheduledAlerts}
                onChange={(e) => updatePref('autoScheduledAlerts', e.target.checked)}
                disabled={!localPrefs.enabled}
                className="rounded w-5 h-5"
              />
            </label>
          </div>

          {/* Quiet Hours */}
          <div className="space-y-4">
            <h3 className="text-sm font-medium text-gray-300 flex items-center gap-2">
              <Clock className="w-4 h-4 text-blue-400" />
              Quiet Hours
            </h3>

            <label className="flex items-center justify-between p-3 rounded-lg bg-gray-900/50 border border-gray-800 cursor-pointer hover:bg-gray-900/70">
              <div>
                <p className="text-sm font-medium">Enable Quiet Hours</p>
                <p className="text-xs text-gray-500">Silence notifications during set times</p>
              </div>
              <input
                type="checkbox"
                checked={localPrefs.quietHoursEnabled}
                onChange={(e) => updatePref('quietHoursEnabled', e.target.checked)}
                disabled={!localPrefs.enabled}
                className="rounded w-5 h-5"
              />
            </label>

            {localPrefs.quietHoursEnabled && (
              <div className="flex items-center gap-3 p-3 rounded-lg bg-gray-900/50 border border-gray-800">
                <span className="text-sm text-gray-400">From</span>
                <input
                  type="number"
                  min="0"
                  max="23"
                  value={localPrefs.quietHoursStart}
                  onChange={(e) => updatePref('quietHoursStart', parseInt(e.target.value) || 22)}
                  className="w-16 px-2 py-1 rounded bg-gray-800 border border-gray-700 text-sm text-center"
                />
                <span className="text-sm text-gray-400">to</span>
                <input
                  type="number"
                  min="0"
                  max="23"
                  value={localPrefs.quietHoursEnd}
                  onChange={(e) => updatePref('quietHoursEnd', parseInt(e.target.value) || 7)}
                  className="w-16 px-2 py-1 rounded bg-gray-800 border border-gray-700 text-sm text-center"
                />
                <span className="text-sm text-gray-500">(24h)</span>
              </div>
            )}
          </div>

          {/* Sound & Vibration */}
          <div className="space-y-4">
            <h3 className="text-sm font-medium text-gray-300 flex items-center gap-2">
              <Volume2 className="w-4 h-4 text-purple-400" />
              Sound & Vibration
            </h3>

            <label className="flex items-center justify-between p-3 rounded-lg bg-gray-900/50 border border-gray-800 cursor-pointer hover:bg-gray-900/70">
              <div>
                <p className="text-sm font-medium">Sound</p>
                <p className="text-xs text-gray-500">Play sound with notifications</p>
              </div>
              <input
                type="checkbox"
                checked={localPrefs.soundEnabled}
                onChange={(e) => updatePref('soundEnabled', e.target.checked)}
                disabled={!localPrefs.enabled}
                className="rounded w-5 h-5"
              />
            </label>

            <label className="flex items-center justify-between p-3 rounded-lg bg-gray-900/50 border border-gray-800 cursor-pointer hover:bg-gray-900/70">
              <div>
                <p className="text-sm font-medium">Vibration</p>
                <p className="text-xs text-gray-500">Vibrate on mobile devices</p>
              </div>
              <input
                type="checkbox"
                checked={localPrefs.vibrationEnabled}
                onChange={(e) => updatePref('vibrationEnabled', e.target.checked)}
                disabled={!localPrefs.enabled}
                className="rounded w-5 h-5"
              />
            </label>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between p-4 border-t border-gray-800 bg-gray-900/30">
          <button
            onClick={() => {
              notificationService.sendBreakReminder(60, 10);
            }}
            className="px-3 py-1.5 rounded-lg bg-gray-800 hover:bg-gray-700 text-sm text-gray-400"
            disabled={permissionStatus !== 'granted'}
          >
            Test Notification
          </button>
          <div className="flex gap-2">
            <button
              onClick={onClose}
              className="px-4 py-2 rounded-lg bg-gray-800 hover:bg-gray-700 text-sm"
            >
              Cancel
            </button>
            <button
              onClick={() => onSave(localPrefs)}
              className="px-4 py-2 rounded-lg bg-amber-600 hover:bg-amber-500 text-sm font-medium"
            >
              Save Settings
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
