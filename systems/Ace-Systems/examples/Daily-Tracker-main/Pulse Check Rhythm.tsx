import React, { useState, useEffect, useRef, useMemo } from 'react';
import {
  Dumbbell, Brain, Heart, Shield, NotebookPen, Download, 
  Calendar, ChevronDown, ChevronUp, Pill, Plus, X, Volume2, 
  CheckCircle2, Trash2, Loader2, Clock, Sparkles, Waves, Zap, Moon, Copy, Timer
} from 'lucide-react';

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
  
  const [newJournalOpen, setNewJournalOpen] = useState(false);
  const [newJournalContent, setNewJournalContent] = useState('');
  const [newJournalWave, setNewJournalWave] = useState<string>('');
  
  const [generalNoteOpen, setGeneralNoteOpen] = useState(false);
  const [generalNoteText, setGeneralNoteText] = useState('');
  const [generalNoteTime, setGeneralNoteTime] = useState(toTimeInput(addMinutes(new Date(), 30)));
  
  const [mixerOpen, setMixerOpen] = useState(false);
  const [mixFreqs, setMixFreqs] = useState<number[]>([432, 0, 0]);
  const mixCtx = useRef<AudioContext | null>(null);
  const mixOsc = useRef<(OscillatorNode | null)[]>([null, null, null]);
  const mixGain = useRef<(GainNode | null)[]>([null, null, null]);
  const [mixPlaying, setMixPlaying] = useState(false);
  
  const [glyphCanvasOpen, setGlyphCanvasOpen] = useState(false);

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

  const scheduleBeat = (category: string, task: string, when: Date, note?: string, waveId?: string, isAnchor?: boolean) => {
    const entry: CheckIn = {
      id: Date.now().toString() + Math.random(),
      category,
      task,
      waveId,
      slot: when.toISOString(),
      loggedAt: new Date().toISOString(),
      note,
      done: false,
      isAnchor
    };
    setCheckIns(prev => [entry, ...prev]);
  };

  const markDone = (id: string) => {
    setCheckIns(prev => prev.map(c => c.id === id ? { ...c, done: true, loggedAt: new Date().toISOString() } : c));
  };

  const removeCheckIn = (id: string) => setCheckIns(prev => prev.filter(c => c.id !== id));
  
  const toggleExpanded = (id: string) => {
    setCheckIns(prev => prev.map(c => c.id === id ? { ...c, expanded: !c.expanded } : c));
  };

  const snoozeAnchor = (id: string, minutes: number) => {
    const anchor = checkIns.find(c => c.id === id);
    if (!anchor) return;
    
    const newTime = addMinutes(new Date(anchor.slot), minutes);
    scheduleBeat(anchor.category, anchor.task, newTime, anchor.note, anchor.waveId, anchor.isAnchor);
    removeCheckIn(id);
  };

  const copyYesterdayAnchors = () => {
    const yesterday = new Date(selectedDate);
    yesterday.setDate(yesterday.getDate() - 1);

    rhythmProfile.waves.forEach(wave => {
      const yAnchor = checkIns.find(
        c => c.isAnchor && c.waveId === wave.id && sameDay(new Date(c.slot), yesterday)
      );
      if (!yAnchor) return;

      const when = new Date(selectedDate);
      const anchorTime = new Date(yAnchor.slot);
      when.setHours(anchorTime.getHours(), anchorTime.getMinutes(), 0, 0);

      scheduleBeat('Anchor', yAnchor.task, when, yAnchor.note, wave.id, true);
    });
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
    setNewJournalContent('');
    setNewJournalWave('');
    setNewJournalOpen(false);
  };

  const deleteJournalEntry = (entryId: string) => {
    const dayKey = toDateInput(selectedDate);
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

  return (
    <div className="min-h-screen bg-gradient-to-b from-black via-gray-900 to-blue-950 text-gray-100 p-4 md:p-8">
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
          </div>
          <div className="text-sm text-gray-500 h-5">
            {saveStatus === 'saving' && 'Syncing...'}
            {saveStatus === 'saved' && 'Synced ✓'}
            {saveStatus === 'error' && 'Sync failed'}
          </div>
        </div>

        {/* Current Wave & Rhythm Score */}
        <div className="bg-gray-950/60 backdrop-blur border border-gray-800 rounded-2xl p-6">
          <div className="flex items-center justify-between flex-wrap gap-4">
            <div className="flex items-center gap-3">
              <Waves className={`w-8 h-8 ${waveColorClasses[getWaveColor(currentWave?.id)].text}`} />
              <div>
                <p className="text-lg font-medium">{currentWave?.name || 'No Active Wave'}</p>
                <p className="text-sm text-gray-400">{currentWave?.description || 'Define your waves'}</p>
              </div>
            </div>
            <div className="text-center">
              <p className="text-3xl font-bold text-purple-400">{rhythmScore}%</p>
              <p className="text-xs text-gray-500">Rhythm Score</p>
            </div>
          </div>
        </div>

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
          <RoundButton icon={<SpiralGlyph size={40} />} label="Glyph" onClick={() => setGlyphCanvasOpen(true)} />
          <RoundButton icon={<Volume2 className="w-8 h-8 text-violet-400"/>} label="Frequency" onClick={() => setMixerOpen(true)} />
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
                onPrev={() => setVisibleMonth(new Date(visibleMonth.getFullYear(), visibleMonth.getMonth() - 1, 1))}
                onNext={() => setVisibleMonth(new Date(visibleMonth.getFullYear(), visibleMonth.getMonth() + 1, 1))}
                onSelect={(d) => setSelectedDate(d)}
              />
            </div>
          )}
        </div>

        {/* Modals */}
        {setupOpen && (
          <WaveSetupWizard
            profile={rhythmProfile}
            onClose={() => setSetupOpen(false)}
            onSave={(profile) => {
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
            onSetAnchor={(waveId, task, note) => {
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
            onChange={(p) => { 
              if (p.time !== undefined) setMiniTime(p.time); 
              if (p.task !== undefined) setMiniTask(p.task); 
              if (p.note !== undefined) setMiniNote(p.note);
              if (p.waveId !== undefined) setMiniWaveId(p.waveId);
            }}
            onClose={() => setMiniOpen(false)}
            onSubmit={() => {
              const [hh, mm] = miniTime.split(':').map(n => parseInt(n));
              const when = new Date(selectedDate); 
              when.setHours(hh || 0, mm || 0, 0, 0);
              const task = miniTask || `${miniCategory} Beat`;
              scheduleBeat(miniCategory, task, when, miniNote, miniWaveId);
              setMiniOpen(false);
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
          <ToneMixer
            freqs={mixFreqs}
            playing={mixPlaying}
            onChangeFreq={(i, v) => {
              const next = [...mixFreqs];
              next[i] = v;
              setMixFreqs(next);
              if (mixPlaying && mixOsc.current[i]) {
                mixOsc.current[i]!.frequency.setValueAtTime(v, (mixCtx.current as AudioContext).currentTime);
              }
            }}
            onToggle={() => {
              if (!mixPlaying) {
                if (!mixCtx.current) mixCtx.current = new (window.AudioContext || (window as any).webkitAudioContext)();
                for (let i = 0; i < 3; i++) {
                  if (mixFreqs[i] && mixFreqs[i] > 0) {
                    const osc = mixCtx.current.createOscillator();
                    const g = mixCtx.current.createGain();
                    osc.type = 'sine';
                    osc.frequency.setValueAtTime(mixFreqs[i], mixCtx.current.currentTime);
                    g.gain.setValueAtTime(0.0, mixCtx.current.currentTime);
                    g.gain.linearRampToValueAtTime(0.12, mixCtx.current.currentTime + 0.4);
                    osc.connect(g);
                    g.connect(mixCtx.current.destination);
                    osc.start();
                    mixOsc.current[i] = osc;
                    mixGain.current[i] = g;
                  }
                }
                setMixPlaying(true);
              } else {
                for (let i = 0; i < 3; i++) {
                  try {
                    mixOsc.current[i]?.stop();
                  } catch {}
                  mixOsc.current[i] = null;
                  mixGain.current[i] = null;
                }
                setMixPlaying(false);
              }
            }}
            onClose={() => {
              if (mixPlaying) {
                for (let i = 0; i < 3; i++) {
                  try {
                    mixOsc.current[i]?.stop();
                  } catch {}
                }
                setMixPlaying(false);
              }
              setMixerOpen(false);
            }}
          />
        )}

        {glyphCanvasOpen && (
          <GlyphCanvas onClose={() => setGlyphCanvasOpen(false)} />
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

function SpiralGlyph({ size = 32 }: { size?: number }) {
  return <span style={{ fontSize: size }} className="text-amber-400">🌀</span>;
}

function RoundButton({ icon, label, onClick }: { icon: JSX.Element; label: string; onClick: () => void }) {
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

function CalendarMonth({ month, selectedDate, deviationDay, onPrev, onNext, onSelect }: {
  month: Date; selectedDate: Date; deviationDay?: number; onPrev:()=>void; onNext:()=>void; onSelect:(d:Date)=>void;
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

  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <button onClick={onPrev} className="px-3 py-1.5 rounded-lg bg-gray-900/70 border border-gray-800 hover:bg-gray-800">{'<'}</button>
        <div className="text-sm text-gray-300">{month.toLocaleString(undefined,{ month:'long', year:'numeric' })}</div>
        <button onClick={onNext} className="px-3 py-1.5 rounded-lg bg-gray-900/70 border border-gray-800 hover:bg-gray-800">{'>'}</button>
      </div>
      <div className="grid grid-cols-7 gap-1 text-center text-xs text-gray-400 mb-1">
        {['Sun','Mon','Tue','Wed','Thu','Fri','Sat'].map(w => <div key={w}>{w}</div>)}
      </div>
      <div className="grid grid-cols-7 gap-1">
        {days.map((d,i)=>{
          const disabled = !isCurrentMonth(d);
          const active = sameDay(d, selectedDate);
          const isDeviation = d.getDay() === deviationDay;
          return (
            <button key={i} onClick={()=>onSelect(d)}
              className={`relative aspect-square rounded-lg border text-sm ${disabled?'text-gray-600 border-gray-800 bg-gray-950/40':'text-gray-200 border-gray-800 bg-gray-900/60 hover:bg-gray-800/60'} ${active?'!border-cyan-500 !bg-cyan-600 !text-black':''} ${isDeviation && !disabled?'!border-orange-500/50':''}`}
            >
              {d.getDate()}
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

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="w-full max-w-2xl bg-gray-950 border-2 border-purple-700 rounded-2xl p-6 space-y-6 max-h-[90vh] overflow-y-auto">
        <h2 className="text-2xl font-light text-purple-400 flex items-center gap-3">
          <Waves className="w-8 h-8" /> Wave Setup Wizard
        </h2>
        
        {step === 0 && (
          <div className="space-y-4">
            <p className="text-gray-300">Your brain flows through three biological waves each day. Let's map YOUR personal rhythm.</p>
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
              Next: Deviation Day
            </button>
          </div>
        )}

        {step === 3 && (
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
            <div className="flex gap-3">
              <button onClick={() => setStep(2)} className="flex-1 px-6 py-3 rounded-xl bg-gray-800 hover:bg-gray-700 text-white">
                Back
              </button>
              <button
                onClick={() => {
                  onSave({
                    waves,
                    deviationDay,
                    wakeTime,
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

function MiniScheduler({ date, category, presetTasks, time, note, waveId, waves, onChange, onClose, onSubmit }: any) {
  const [selectedTask, setSelectedTask] = useState<string>('');

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-end md:items-center justify-center z-50 p-4">
      <div className="w-full max-w-md bg-gray-950 border border-gray-800 rounded-2xl p-5 space-y-3">
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

function ToneMixer({ freqs, playing, onChangeFreq, onToggle, onClose }: any) {
  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-end md:items-center justify-center z-50 p-4">
      <div className="w-full max-w-md bg-gray-950 border border-gray-800 rounded-2xl p-5 space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Volume2 className="w-5 h-5 text-violet-400" />
            <span className="text-sm text-gray-400">3-Tier Frequency Modulator</span>
          </div>
          <button onClick={onClose} className="px-2 py-1 rounded-lg bg-gray-900/70 border border-gray-800 hover:bg-gray-800">Close</button>
        </div>
        {[0, 1, 2].map(i => (
          <div key={i} className="space-y-2">
            <div className="flex items-center justify-between text-xs text-gray-400">
              <span>Layer {i + 1}</span>
              <span>{freqs[i] ? `${freqs[i]} Hz` : 'off'}</span>
            </div>
            <input type="range" min={0} max={1000} step={1} value={freqs[i]} onChange={(e) => onChangeFreq(i, parseInt(e.target.value || '0'))} className="w-full" />
          </div>
        ))}
        <div className="flex items-center justify-end gap-2">
          <button onClick={onToggle} className={`px-4 py-2 rounded-xl ${playing ? 'bg-rose-500 text-black hover:bg-rose-400' : 'bg-violet-500 text-black hover:bg-violet-400'}`}>
            {playing ? 'Stop' : 'Play'}
          </button>
        </div>
      </div>
    </div>
  );
}

function GlyphCanvas({ onClose }: { onClose: () => void }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
    ctx.strokeStyle = '#a78bfa';
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
  }, []);

  const startDrawing = (e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    setIsDrawing(true);
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const rect = canvas.getBoundingClientRect();
    const x = 'touches' in e ? e.touches[0].clientX - rect.left : e.clientX - rect.left;
    const y = 'touches' in e ? e.touches[0].clientY - rect.top : e.clientY - rect.top;
    ctx.beginPath();
    ctx.moveTo(x, y);
  };

  const draw = (e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    if (!isDrawing) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const rect = canvas.getBoundingClientRect();
    const x = 'touches' in e ? e.touches[0].clientX - rect.left : e.clientX - rect.left;
    const y = 'touches' in e ? e.touches[0].clientY - rect.top : e.clientY - rect.top;
    ctx.lineTo(x, y);
    ctx.stroke();
  };

  const stopDrawing = () => setIsDrawing(false);

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  };

  const saveGlyph = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dataUrl = canvas.toDataURL('image/png');
    const link = document.createElement('a');
    link.download = `glyph_${Date.now()}.png`;
    link.href = dataUrl;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="fixed inset-0 bg-black z-50 flex flex-col">
      <div className="flex items-center justify-between p-4 border-b border-gray-800">
        <button onClick={onClose} className="p-2 rounded-lg bg-gray-900/70 border border-gray-800 hover:bg-gray-800">
          <X className="w-6 h-6" />
        </button>
        <h2 className="text-xl font-light text-purple-400 flex items-center gap-2">
          🌀 Glyph Canvas
        </h2>
        <div className="flex gap-2">
          <button onClick={saveGlyph} className="px-4 py-2 rounded-lg bg-purple-600 hover:bg-purple-500 text-white flex items-center gap-2">
            <Download className="w-4 h-4" />Save
          </button>
          <button onClick={clearCanvas} className="px-4 py-2 rounded-lg bg-red-600 hover:bg-red-500 text-white">
            Clear
          </button>
        </div>
      </div>
      <canvas ref={canvasRef} className="flex-1 bg-gray-950 touch-none" onMouseDown={startDrawing} onMouseMove={draw} onMouseUp={stopDrawing} onMouseLeave={stopDrawing} onTouchStart={startDrawing} onTouchMove={draw} onTouchEnd={stopDrawing} />
    </div>
  );
}