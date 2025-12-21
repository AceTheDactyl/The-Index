/**
 * Focus & Wellness Tools Component
 *
 * Integrated tabbed section with:
 * - Frequency: 3-tier tone generator affecting drawing gradients
 * - Glyph: Drawing canvas with daily challenges and gradient rewards
 * - Metrics: DeltaHV metrics display
 */

import React, { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import {
  Activity,
  Volume2,
  VolumeX,
  Music,
  Sparkles,
  Heart,
  Zap,
  Moon,
  Target,
  Trophy,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Download,
  Trash2,
  Pencil,
  Gift,
  Calendar,
  Lock,
  Unlock,
} from 'lucide-react';
import type { DeltaHVState } from '../lib/deltaHVEngine';
import type { EnhancedDeltaHVState } from '../lib/metricsHub';
import { metricsHub } from '../lib/metricsHub';

interface FocusWellnessToolsProps {
  deltaHV: DeltaHVState | null;
  enhancedMetrics: EnhancedDeltaHVState | null;
  onOpenFullDJ?: () => void;
  onOpenFullChallenges?: () => void;
  onCompleteChallenge?: (regionId: string, xp: number) => void;
}

type ActiveTab = 'metrics' | 'frequency' | 'glyph';

// Gradient definitions for unlockable palettes
interface GradientDef {
  id: string;
  name: string;
  colors: string[]; // Base colors that get modulated by frequencies
  unlockDay: number;
}

const GRADIENTS: GradientDef[] = [
  { id: 'viridis', name: 'Viridis', colors: ['#440154', '#31688e', '#35b779', '#fde725'], unlockDay: 0 },
  { id: 'plasma', name: 'Plasma', colors: ['#0d0887', '#7e03a8', '#cc4778', '#f89540'], unlockDay: 1 },
  { id: 'inferno', name: 'Inferno', colors: ['#000004', '#932667', '#dd513a', '#fcffa4'], unlockDay: 2 },
  { id: 'magma', name: 'Magma', colors: ['#000004', '#8c2981', '#de4968', '#fcfdbf'], unlockDay: 3 },
  { id: 'ocean', name: 'Ocean', colors: ['#023e8a', '#0096c7', '#48cae4', '#90e0ef'], unlockDay: 5 },
  { id: 'sunset', name: 'Sunset', colors: ['#ff6b6b', '#f77f00', '#fcbf49', '#eae2b7'], unlockDay: 7 },
  { id: 'aurora', name: 'Aurora', colors: ['#00ff87', '#60efff', '#ff00c8', '#ff006e'], unlockDay: 10 },
  { id: 'cosmic', name: 'Cosmic', colors: ['#667eea', '#764ba2', '#f5576c', '#4facfe'], unlockDay: 14 },
];

// Frequency presets for the 3-tier system
interface FrequencyPreset {
  id: string;
  name: string;
  icon: React.ReactNode;
  freqs: [number, number, number];
  description: string;
}

const FREQUENCY_PRESETS: FrequencyPreset[] = [
  { id: 'focus', name: 'Focus', icon: <Target className="w-4 h-4" />, freqs: [40, 14, 7], description: 'Gamma + Beta + Theta' },
  { id: 'calm', name: 'Calm', icon: <Moon className="w-4 h-4" />, freqs: [432, 528, 396], description: 'Solfeggio healing' },
  { id: 'energy', name: 'Energy', icon: <Zap className="w-4 h-4" />, freqs: [285, 417, 639], description: 'Activation tones' },
  { id: 'love', name: 'Love', icon: <Heart className="w-4 h-4" />, freqs: [528, 639, 741], description: 'Heart frequencies' },
];

// Storage keys
const STORAGE_KEYS = {
  GLYPHS: 'glyph-gallery',
  XP: 'glyph-xp',
  UNLOCKED_GRADIENTS: 'unlocked-gradients',
  DAILY_STREAK: 'daily-glyph-streak',
  LAST_DAILY: 'last-daily-glyph',
};

// Generate multi-color based on 3 frequencies and gradient
// Freq1 controls position in gradient, Freq2 controls saturation, Freq3 controls brightness
function getMultiColor(gradient: GradientDef, freqs: [number, number, number]): string {
  const [f1, f2, f3] = freqs;

  // All zeros = white
  if (f1 === 0 && f2 === 0 && f3 === 0) return '#ffffff';

  const colors = gradient.colors;

  // F1: Position in gradient (0-600Hz maps to gradient)
  const t = Math.min(f1 / 600, 1);
  const idx = t * (colors.length - 1);
  const lower = Math.floor(idx);
  const upper = Math.min(lower + 1, colors.length - 1);
  const mix = idx - lower;

  const c1 = hexToRgb(colors[lower]);
  const c2 = hexToRgb(colors[upper]);
  if (!c1 || !c2) return colors[lower];

  let r = c1.r + (c2.r - c1.r) * mix;
  let g = c1.g + (c2.g - c1.g) * mix;
  let b = c1.b + (c2.b - c1.b) * mix;

  // F2: Saturation boost (0-1000Hz, higher = more saturated)
  const satBoost = Math.min(f2 / 500, 1);
  const gray = (r + g + b) / 3;
  r = gray + (r - gray) * (0.5 + satBoost * 0.5);
  g = gray + (g - gray) * (0.5 + satBoost * 0.5);
  b = gray + (b - gray) * (0.5 + satBoost * 0.5);

  // F3: Brightness/luminance shift (0-1000Hz, higher = brighter)
  const brightBoost = Math.min(f3 / 800, 1);
  r = r + (255 - r) * brightBoost * 0.3;
  g = g + (255 - g) * brightBoost * 0.3;
  b = b + (255 - b) * brightBoost * 0.3;

  return `rgb(${Math.round(Math.max(0, Math.min(255, r)))},${Math.round(Math.max(0, Math.min(255, g)))},${Math.round(Math.max(0, Math.min(255, b)))})`;
}

// Create a canvas gradient from the 3 frequencies
function createStrokeGradient(
  ctx: CanvasRenderingContext2D,
  gradient: GradientDef,
  freqs: [number, number, number],
  x1: number, y1: number, x2: number, y2: number
): CanvasGradient {
  const canvasGrad = ctx.createLinearGradient(x1, y1, x2, y2);

  // Generate multiple colors based on frequency variations
  const [f1, f2, f3] = freqs;

  // Color 1: Base from f1
  canvasGrad.addColorStop(0, getMultiColor(gradient, [f1, f2, f3]));
  // Color 2: Shifted by f2
  canvasGrad.addColorStop(0.5, getMultiColor(gradient, [f1 + f2 * 0.3, f2 * 0.8, f3 * 1.2]));
  // Color 3: Shifted by f3
  canvasGrad.addColorStop(1, getMultiColor(gradient, [f1 * 0.7 + f3 * 0.3, f2 * 1.1, f3]));

  return canvasGrad;
}

function hexToRgb(hex: string): { r: number; g: number; b: number } | null {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result ? {
    r: parseInt(result[1], 16),
    g: parseInt(result[2], 16),
    b: parseInt(result[3], 16)
  } : null;
}

// Daily challenge prompts
const DAILY_PROMPTS = [
  'Draw your current mood',
  'Sketch a symbol of peace',
  'Create a pattern that flows',
  'Draw what energy looks like',
  'Illustrate your intention',
  'Draw a protective symbol',
  'Sketch your inner calm',
];

export const FocusWellnessTools: React.FC<FocusWellnessToolsProps> = ({
  deltaHV,
  enhancedMetrics,
  onOpenFullDJ,
}) => {
  const [activeTab, setActiveTab] = useState<ActiveTab>('metrics');
  const [isExpanded, setIsExpanded] = useState(true);

  // 3-Tier Frequency State
  const [freqs, setFreqs] = useState<[number, number, number]>([432, 528, 396]);
  const [isPlaying, setIsPlaying] = useState(false);
  const audioCtx = useRef<AudioContext | null>(null);
  const oscillators = useRef<(OscillatorNode | null)[]>([null, null, null]);
  const gainNodes = useRef<(GainNode | null)[]>([null, null, null]);

  // Glyph/Drawing State
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [showCanvas, setShowCanvas] = useState(false);
  const [savedGlyphs, setSavedGlyphs] = useState<string[]>([]);
  const [totalXP, setTotalXP] = useState(0);
  const [selectedGradient, setSelectedGradient] = useState<GradientDef>(GRADIENTS[0]);
  const [unlockedGradients, setUnlockedGradients] = useState<string[]>(['viridis']);
  const [dailyStreak, setDailyStreak] = useState(0);
  const [dailyCompleted, setDailyCompleted] = useState(false);
  const [showGradientPicker, setShowGradientPicker] = useState(false);
  const [lastPoint, setLastPoint] = useState<{x: number, y: number} | null>(null);

  // Get today's daily prompt
  const dailyPrompt = useMemo(() => {
    const dayOfYear = Math.floor((Date.now() - new Date(new Date().getFullYear(), 0, 0).getTime()) / 86400000);
    return DAILY_PROMPTS[dayOfYear % DAILY_PROMPTS.length];
  }, []);

  // Load saved data
  useEffect(() => {
    try {
      const glyphs = localStorage.getItem(STORAGE_KEYS.GLYPHS);
      if (glyphs) setSavedGlyphs(JSON.parse(glyphs));

      const xp = localStorage.getItem(STORAGE_KEYS.XP);
      if (xp) setTotalXP(parseInt(xp));

      const unlocked = localStorage.getItem(STORAGE_KEYS.UNLOCKED_GRADIENTS);
      if (unlocked) setUnlockedGradients(JSON.parse(unlocked));

      const streak = localStorage.getItem(STORAGE_KEYS.DAILY_STREAK);
      if (streak) setDailyStreak(parseInt(streak));

      const lastDaily = localStorage.getItem(STORAGE_KEYS.LAST_DAILY);
      if (lastDaily) {
        const today = new Date().toDateString();
        const lastDate = new Date(lastDaily).toDateString();
        setDailyCompleted(today === lastDate);

        const yesterday = new Date();
        yesterday.setDate(yesterday.getDate() - 1);
        if (lastDate !== today && lastDate !== yesterday.toDateString()) {
          setDailyStreak(0);
          localStorage.setItem(STORAGE_KEYS.DAILY_STREAK, '0');
        }
      }
    } catch (e) {
      console.error('Failed to load data:', e);
    }
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopAudio();
    };
  }, []);

  // Initialize canvas
  useEffect(() => {
    if (showCanvas && canvasRef.current) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * window.devicePixelRatio;
      canvas.height = rect.height * window.devicePixelRatio;
      ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
      ctx.lineWidth = 5;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
    }
  }, [showCanvas]);

  // Audio controls - 3 oscillators
  const startAudio = useCallback(() => {
    if (!audioCtx.current) {
      audioCtx.current = new (window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext)();
    }

    if (audioCtx.current.state === 'suspended') {
      audioCtx.current.resume();
    }

    for (let i = 0; i < 3; i++) {
      if (freqs[i] > 0) {
        const osc = audioCtx.current.createOscillator();
        const gain = audioCtx.current.createGain();

        osc.type = 'sine';
        osc.frequency.setValueAtTime(freqs[i], audioCtx.current.currentTime);

        gain.gain.setValueAtTime(0, audioCtx.current.currentTime);
        gain.gain.linearRampToValueAtTime(0.1, audioCtx.current.currentTime + 0.3);

        osc.connect(gain);
        gain.connect(audioCtx.current.destination);
        osc.start();

        oscillators.current[i] = osc;
        gainNodes.current[i] = gain;
      }
    }

    setIsPlaying(true);
  }, [freqs]);

  const stopAudio = useCallback(() => {
    for (let i = 0; i < 3; i++) {
      if (gainNodes.current[i] && audioCtx.current) {
        gainNodes.current[i]!.gain.linearRampToValueAtTime(0, audioCtx.current.currentTime + 0.2);
      }
    }

    setTimeout(() => {
      for (let i = 0; i < 3; i++) {
        try {
          oscillators.current[i]?.stop();
        } catch {}
        oscillators.current[i] = null;
        gainNodes.current[i] = null;
      }
    }, 250);

    setIsPlaying(false);
  }, []);

  const toggleAudio = useCallback(() => {
    if (isPlaying) {
      stopAudio();
    } else {
      startAudio();
    }
  }, [isPlaying, startAudio, stopAudio]);

  // Update individual frequency
  const handleFreqChange = useCallback((index: number, value: number) => {
    setFreqs(prev => {
      const next = [...prev] as [number, number, number];
      next[index] = value;
      return next;
    });

    if (isPlaying && oscillators.current[index] && audioCtx.current) {
      if (value > 0) {
        oscillators.current[index]!.frequency.setValueAtTime(value, audioCtx.current.currentTime);
      }
    }
  }, [isPlaying]);

  const applyPreset = useCallback((preset: FrequencyPreset) => {
    setFreqs(preset.freqs);
    if (isPlaying) {
      preset.freqs.forEach((freq, i) => {
        if (oscillators.current[i] && audioCtx.current && freq > 0) {
          oscillators.current[i]!.frequency.setValueAtTime(freq, audioCtx.current.currentTime);
        }
      });
    }
  }, [isPlaying]);

  // Drawing functions with multi-color gradient strokes
  const getCanvasCoords = useCallback((e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    const x = 'touches' in e ? e.touches[0].clientX - rect.left : e.clientX - rect.left;
    const y = 'touches' in e ? e.touches[0].clientY - rect.top : e.clientY - rect.top;
    return { x, y };
  }, []);

  const startDrawing = useCallback((e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    setIsDrawing(true);
    const { x, y } = getCanvasCoords(e);
    setLastPoint({ x, y });
  }, [getCanvasCoords]);

  const draw = useCallback((e: React.MouseEvent<HTMLCanvasElement> | React.TouchEvent<HTMLCanvasElement>) => {
    if (!isDrawing || !lastPoint) return;
    e.preventDefault();

    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const { x, y } = getCanvasCoords(e);

    // Create gradient stroke based on 3 frequencies
    const gradient = createStrokeGradient(ctx, selectedGradient, freqs, lastPoint.x, lastPoint.y, x, y);
    ctx.strokeStyle = gradient;

    ctx.beginPath();
    ctx.moveTo(lastPoint.x, lastPoint.y);
    ctx.lineTo(x, y);
    ctx.stroke();

    setLastPoint({ x, y });
  }, [isDrawing, lastPoint, getCanvasCoords, selectedGradient, freqs]);

  const stopDrawing = useCallback(() => {
    setIsDrawing(false);
    setLastPoint(null);
  }, []);

  const clearCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }, []);

  const saveGlyph = useCallback((isDaily = false) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const dataUrl = canvas.toDataURL('image/png');

    setSavedGlyphs(prev => {
      const updated = [dataUrl, ...prev].slice(0, 20);
      localStorage.setItem(STORAGE_KEYS.GLYPHS, JSON.stringify(updated));
      return updated;
    });

    const xpEarned = isDaily ? 50 : 25;
    setTotalXP(prev => {
      const updated = prev + xpEarned;
      localStorage.setItem(STORAGE_KEYS.XP, updated.toString());
      return updated;
    });

    if (isDaily && !dailyCompleted) {
      const newStreak = dailyStreak + 1;
      setDailyStreak(newStreak);
      setDailyCompleted(true);
      localStorage.setItem(STORAGE_KEYS.DAILY_STREAK, newStreak.toString());
      localStorage.setItem(STORAGE_KEYS.LAST_DAILY, new Date().toISOString());

      const newUnlocks = GRADIENTS.filter(g =>
        g.unlockDay <= newStreak && !unlockedGradients.includes(g.id)
      ).map(g => g.id);

      if (newUnlocks.length > 0) {
        const allUnlocked = [...unlockedGradients, ...newUnlocks];
        setUnlockedGradients(allUnlocked);
        localStorage.setItem(STORAGE_KEYS.UNLOCKED_GRADIENTS, JSON.stringify(allUnlocked));
      }
    }

    metricsHub.recordAction('glyph_saved');
    clearCanvas();
    setShowCanvas(false);
  }, [dailyCompleted, dailyStreak, unlockedGradients, clearCanvas]);

  const downloadGlyph = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dataUrl = canvas.toDataURL('image/png');
    const link = document.createElement('a');
    link.download = `glyph_${Date.now()}.png`;
    link.href = dataUrl;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }, []);

  // Gradient preview
  const renderGradientPreview = (gradient: GradientDef, size = 24) => (
    <div
      className="rounded"
      style={{
        width: size,
        height: size,
        background: `linear-gradient(135deg, ${gradient.colors.join(', ')})`,
      }}
    />
  );

  // Current stroke color preview
  const currentStrokeColor = getMultiColor(selectedGradient, freqs);

  return (
    <div className="bg-gray-950/60 backdrop-blur border border-gray-800 rounded-2xl overflow-hidden">
      {/* Header with Tabs */}
      <div className="p-4 border-b border-gray-800">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-lg font-medium text-white flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-purple-400" />
            Focus & Wellness
          </h2>
          <button
            onClick={() => setIsExpanded(!isExpanded)}
            className="p-1.5 rounded-lg hover:bg-gray-800 transition-colors"
          >
            {isExpanded ? <ChevronUp className="w-5 h-5 text-gray-400" /> : <ChevronDown className="w-5 h-5 text-gray-400" />}
          </button>
        </div>

        {/* Tab Navigation */}
        <div className="flex gap-2">
          <button
            onClick={() => setActiveTab('metrics')}
            className={`flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded-xl text-sm font-medium transition-all ${
              activeTab === 'metrics'
                ? 'bg-gradient-to-r from-cyan-600/30 to-purple-600/30 border border-cyan-500/30 text-white'
                : 'bg-gray-900/50 border border-gray-800 text-gray-400 hover:text-white hover:border-gray-700'
            }`}
          >
            <Activity className="w-4 h-4" />
            <span className="hidden sm:inline">Metrics</span>
          </button>
          <button
            onClick={() => setActiveTab('frequency')}
            className={`flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded-xl text-sm font-medium transition-all ${
              activeTab === 'frequency'
                ? 'bg-gradient-to-r from-violet-600/30 to-pink-600/30 border border-violet-500/30 text-white'
                : 'bg-gray-900/50 border border-gray-800 text-gray-400 hover:text-white hover:border-gray-700'
            }`}
          >
            {isPlaying ? <Volume2 className="w-4 h-4 text-violet-400" /> : <VolumeX className="w-4 h-4" />}
            <span className="hidden sm:inline">Tones</span>
            {isPlaying && <span className="w-2 h-2 bg-violet-400 rounded-full animate-pulse" />}
          </button>
          <button
            onClick={() => setActiveTab('glyph')}
            className={`flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded-xl text-sm font-medium transition-all ${
              activeTab === 'glyph'
                ? 'bg-gradient-to-r from-amber-600/30 to-orange-600/30 border border-amber-500/30 text-white'
                : 'bg-gray-900/50 border border-gray-800 text-gray-400 hover:text-white hover:border-gray-700'
            }`}
          >
            <Pencil className="w-4 h-4" />
            <span className="hidden sm:inline">Draw</span>
            {!dailyCompleted && <span className="w-2 h-2 bg-amber-400 rounded-full" />}
          </button>
        </div>
      </div>

      {/* Content */}
      {isExpanded && (
        <div className="p-4">
          {/* Metrics Tab */}
          {activeTab === 'metrics' && (
            <div className="space-y-4">
              {deltaHV && (
                <>
                  <div className="flex items-center justify-between p-4 rounded-xl bg-gradient-to-r from-purple-900/30 to-cyan-900/30 border border-purple-500/20">
                    <div className="flex items-center gap-3">
                      <div className={`w-12 h-12 rounded-full flex items-center justify-center ${
                        deltaHV.fieldState === 'coherent' ? 'bg-emerald-900/60 text-emerald-400' :
                        deltaHV.fieldState === 'transitioning' ? 'bg-amber-900/60 text-amber-400' :
                        deltaHV.fieldState === 'fragmented' ? 'bg-orange-900/60 text-orange-400' :
                        'bg-gray-800/60 text-gray-400'
                      }`}>
                        <Activity className="w-6 h-6" />
                      </div>
                      <div>
                        <p className="text-lg font-medium text-white">ΔHV Field</p>
                        <p className="text-sm text-gray-400 capitalize">{deltaHV.fieldState}</p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className={`text-3xl font-bold ${
                        deltaHV.deltaHV >= 75 ? 'text-emerald-400' :
                        deltaHV.deltaHV >= 50 ? 'text-amber-400' :
                        deltaHV.deltaHV >= 25 ? 'text-orange-400' :
                        'text-gray-400'
                      }`}>{deltaHV.deltaHV}</p>
                      <p className="text-xs text-gray-500">Score</p>
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                    {[
                      { label: 'Symbolic', value: deltaHV.symbolicDensity, color: 'violet' },
                      { label: 'Resonance', value: deltaHV.resonanceCoupling, color: 'cyan' },
                      { label: 'Friction', value: deltaHV.frictionCoefficient, color: 'orange' },
                      { label: 'Stability', value: deltaHV.harmonicStability, color: 'emerald' },
                    ].map(({ label, value, color }) => (
                      <div key={label} className={`bg-gray-900/60 rounded-xl p-3 border border-${color}-800/30`}>
                        <div className="flex items-center justify-between mb-1">
                          <span className={`text-sm text-${color}-300`}>{label}</span>
                          <span className={`text-lg font-bold text-${color}-400`}>{value}%</span>
                        </div>
                        <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
                          <div className={`h-full bg-${color}-500 rounded-full transition-all`} style={{ width: `${value}%` }} />
                        </div>
                      </div>
                    ))}
                  </div>

                  {enhancedMetrics?.musicInfluence && enhancedMetrics.musicInfluence.authorshipScore > 0 && (
                    <div className="p-3 bg-gray-900/40 rounded-xl border border-purple-800/30">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <Music className="w-4 h-4 text-purple-400" />
                          <span className="text-sm text-purple-300">Music</span>
                        </div>
                        <span className="text-sm text-purple-400">{enhancedMetrics.musicInfluence.authorshipScore}%</span>
                      </div>
                    </div>
                  )}
                </>
              )}

              {!deltaHV && (
                <div className="text-center py-8 text-gray-500">
                  <Activity className="w-10 h-10 mx-auto mb-3 opacity-50" />
                  <p>Complete rhythm setup to see metrics</p>
                </div>
              )}
            </div>
          )}

          {/* Frequency Tab - 3 Tiers */}
          {activeTab === 'frequency' && (
            <div className="space-y-4">
              {/* Frequency Display */}
              <div className="grid grid-cols-3 gap-2 text-center">
                {['Hue', 'Saturation', 'Brightness'].map((label, i) => (
                  <div key={label} className="p-2 bg-gray-900/50 rounded-lg">
                    <p className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-violet-400 to-pink-400">
                      {freqs[i]}
                    </p>
                    <p className="text-xs text-gray-500">{label} Hz</p>
                  </div>
                ))}
              </div>

              {/* Color Preview */}
              <div className="flex items-center gap-3 p-3 bg-gray-900/50 rounded-xl">
                <div
                  className="w-10 h-10 rounded-lg border border-gray-600"
                  style={{ backgroundColor: currentStrokeColor }}
                />
                <div className="flex-1">
                  <p className="text-sm text-gray-400">Current Stroke Color</p>
                  <p className="text-xs text-gray-600">Based on frequency mix</p>
                </div>
              </div>

              {/* Presets */}
              <div className="grid grid-cols-4 gap-2">
                {FREQUENCY_PRESETS.map(preset => (
                  <button
                    key={preset.id}
                    onClick={() => applyPreset(preset)}
                    className={`p-2 rounded-xl flex flex-col items-center gap-1 transition-all ${
                      freqs[0] === preset.freqs[0] && freqs[1] === preset.freqs[1] && freqs[2] === preset.freqs[2]
                        ? 'bg-violet-600/30 border-2 border-violet-400'
                        : 'bg-gray-900/50 border border-gray-800 hover:border-violet-500/50'
                    }`}
                  >
                    <span className="text-gray-400">{preset.icon}</span>
                    <span className="text-xs text-gray-400">{preset.name}</span>
                  </button>
                ))}
              </div>

              {/* 3 Frequency Sliders */}
              <div className="space-y-3 p-3 bg-gray-900/40 rounded-xl">
                {[
                  { label: 'Layer 1 (Hue)', color: 'violet' },
                  { label: 'Layer 2 (Saturation)', color: 'pink' },
                  { label: 'Layer 3 (Brightness)', color: 'amber' },
                ].map(({ label, color }, i) => (
                  <div key={i}>
                    <div className="flex justify-between text-xs mb-1">
                      <span className="text-gray-500">{label}</span>
                      <span className={`text-${color}-400`}>{freqs[i]} Hz</span>
                    </div>
                    <input
                      type="range"
                      min={0}
                      max={1000}
                      value={freqs[i]}
                      onChange={(e) => handleFreqChange(i, parseInt(e.target.value))}
                      className={`w-full h-2 bg-gray-800 rounded-full appearance-none cursor-pointer accent-${color}-500`}
                    />
                  </div>
                ))}
              </div>

              {/* Play Button */}
              <button
                onClick={toggleAudio}
                className={`w-full py-3 rounded-xl font-medium flex items-center justify-center gap-2 transition-all ${
                  isPlaying
                    ? 'bg-rose-600 hover:bg-rose-500 text-white'
                    : 'bg-violet-600 hover:bg-violet-500 text-white'
                }`}
              >
                {isPlaying ? <VolumeX className="w-5 h-5" /> : <Volume2 className="w-5 h-5" />}
                {isPlaying ? 'Stop Tones' : 'Play Tones'}
              </button>

              {onOpenFullDJ && (
                <button
                  onClick={onOpenFullDJ}
                  className="w-full py-2 rounded-xl bg-gray-900/50 hover:bg-gray-800/50 text-gray-400 text-sm transition-colors border border-gray-800"
                >
                  Open Full DJ Mode
                </button>
              )}
            </div>
          )}

          {/* Glyph/Drawing Tab */}
          {activeTab === 'glyph' && (
            <div className="space-y-4">
              {showCanvas ? (
                <div className="space-y-3">
                  {/* Canvas Header */}
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <button
                        onClick={() => setShowGradientPicker(!showGradientPicker)}
                        className="p-2 rounded-lg bg-gray-800 hover:bg-gray-700 transition-colors"
                      >
                        {renderGradientPreview(selectedGradient, 20)}
                      </button>
                      <div
                        className="w-5 h-5 rounded-full border border-gray-600"
                        style={{ backgroundColor: currentStrokeColor }}
                        title="Current color from frequencies"
                      />
                    </div>
                    <div className="flex gap-2">
                      <button onClick={clearCanvas} className="p-2 rounded-lg bg-gray-800 hover:bg-gray-700">
                        <Trash2 className="w-4 h-4 text-gray-400" />
                      </button>
                      <button onClick={downloadGlyph} className="p-2 rounded-lg bg-gray-800 hover:bg-gray-700">
                        <Download className="w-4 h-4 text-gray-400" />
                      </button>
                    </div>
                  </div>

                  {/* Gradient Picker */}
                  {showGradientPicker && (
                    <div className="p-3 bg-gray-900/80 rounded-xl border border-gray-700">
                      <p className="text-xs text-gray-500 uppercase tracking-wider mb-2">Select Gradient</p>
                      <div className="grid grid-cols-4 gap-2">
                        {GRADIENTS.map(gradient => {
                          const isUnlocked = unlockedGradients.includes(gradient.id);
                          return (
                            <button
                              key={gradient.id}
                              onClick={() => isUnlocked && setSelectedGradient(gradient)}
                              disabled={!isUnlocked}
                              className={`p-2 rounded-lg flex flex-col items-center gap-1 ${
                                selectedGradient.id === gradient.id
                                  ? 'bg-violet-600/30 border-2 border-violet-400'
                                  : isUnlocked
                                    ? 'bg-gray-800 hover:bg-gray-700 border border-gray-700'
                                    : 'bg-gray-900/50 border border-gray-800 opacity-50'
                              }`}
                            >
                              {renderGradientPreview(gradient, 24)}
                              <span className="text-xs text-gray-400 flex items-center gap-1">
                                {!isUnlocked && <Lock className="w-3 h-3" />}
                                {gradient.name}
                              </span>
                            </button>
                          );
                        })}
                      </div>
                    </div>
                  )}

                  {/* Hint */}
                  <p className="text-xs text-gray-600 text-center">
                    Adjust Tones tab frequencies to change stroke colors
                  </p>

                  {/* Large Canvas */}
                  <canvas
                    ref={canvasRef}
                    className="w-full rounded-xl border-2 touch-none cursor-crosshair"
                    style={{
                      height: '360px',
                      backgroundColor: '#050505',
                      borderColor: currentStrokeColor,
                    }}
                    onMouseDown={startDrawing}
                    onMouseMove={draw}
                    onMouseUp={stopDrawing}
                    onMouseLeave={stopDrawing}
                    onTouchStart={startDrawing}
                    onTouchMove={draw}
                    onTouchEnd={stopDrawing}
                  />

                  {/* Actions */}
                  <div className="flex gap-2">
                    <button
                      onClick={() => { clearCanvas(); setShowCanvas(false); }}
                      className="flex-1 py-2.5 rounded-xl bg-gray-800 hover:bg-gray-700 text-gray-300 text-sm font-medium"
                    >
                      Cancel
                    </button>
                    <button
                      onClick={() => saveGlyph(false)}
                      className="flex-1 py-2.5 rounded-xl bg-violet-600 hover:bg-violet-500 text-white text-sm font-medium"
                    >
                      Save (+25 XP)
                    </button>
                  </div>
                </div>
              ) : (
                <>
                  {/* Stats */}
                  <div className="flex items-center gap-3 p-3 rounded-xl bg-gray-900/50 border border-gray-800">
                    <div className="flex items-center gap-2">
                      <Trophy className="w-5 h-5 text-amber-400" />
                      <span className="text-lg font-bold text-amber-400">{totalXP} XP</span>
                    </div>
                    <div className="h-6 w-px bg-gray-700" />
                    <div className="flex items-center gap-2">
                      <Calendar className="w-4 h-4 text-violet-400" />
                      <span className="text-sm text-violet-300">{dailyStreak} days</span>
                    </div>
                    <div className="h-6 w-px bg-gray-700" />
                    <div className="flex items-center gap-2">
                      <Unlock className="w-4 h-4 text-emerald-400" />
                      <span className="text-sm text-emerald-300">{unlockedGradients.length}/{GRADIENTS.length}</span>
                    </div>
                  </div>

                  {/* Daily Challenge */}
                  <div className={`p-4 rounded-xl border-2 ${
                    dailyCompleted
                      ? 'bg-emerald-950/30 border-emerald-600/40'
                      : 'bg-gradient-to-r from-amber-900/30 to-orange-900/30 border-amber-500/40'
                  }`}>
                    <div className="flex items-start gap-3">
                      <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${
                        dailyCompleted ? 'bg-emerald-600/30' : 'bg-amber-600/30'
                      }`}>
                        {dailyCompleted ? <CheckCircle2 className="w-6 h-6 text-emerald-400" /> : <Gift className="w-6 h-6 text-amber-400" />}
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center gap-2">
                          <p className={`font-medium ${dailyCompleted ? 'text-emerald-300' : 'text-white'}`}>
                            Daily Glyph
                          </p>
                          <span className="text-xs px-2 py-0.5 rounded bg-amber-500/20 text-amber-300">+50 XP</span>
                        </div>
                        <p className="text-sm text-gray-400 mt-1">{dailyPrompt}</p>
                        {!dailyCompleted && (
                          <button
                            onClick={() => setShowCanvas(true)}
                            className="mt-3 px-4 py-2 rounded-lg bg-amber-600 hover:bg-amber-500 text-black text-sm font-medium"
                          >
                            Start Daily
                          </button>
                        )}
                        {dailyCompleted && <p className="text-xs text-emerald-400 mt-2">Done! Come back tomorrow.</p>}
                      </div>
                    </div>
                  </div>

                  {/* Free Draw */}
                  <button
                    onClick={() => setShowCanvas(true)}
                    className="w-full py-3 rounded-xl bg-gray-900/50 border border-gray-800 hover:border-violet-500/50 text-white font-medium flex items-center justify-center gap-2"
                  >
                    <Pencil className="w-5 h-5 text-violet-400" />
                    Free Draw (+25 XP)
                  </button>

                  {/* Next Unlock */}
                  {unlockedGradients.length < GRADIENTS.length && (
                    <div className="p-3 rounded-xl bg-gray-900/50 border border-gray-800">
                      <div className="flex items-center gap-3">
                        <Lock className="w-4 h-4 text-gray-500" />
                        <div className="flex-1">
                          <p className="text-sm text-gray-400">Next unlock:</p>
                          <div className="flex items-center gap-2 mt-1">
                            {renderGradientPreview(GRADIENTS.find(g => !unlockedGradients.includes(g.id))!, 20)}
                            <span className="text-sm text-gray-300">
                              {GRADIENTS.find(g => !unlockedGradients.includes(g.id))?.name}
                            </span>
                            <span className="text-xs text-gray-500">
                              ({Math.max(0, (GRADIENTS.find(g => !unlockedGradients.includes(g.id))?.unlockDay || 0) - dailyStreak)} days)
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Gallery */}
                  {savedGlyphs.length > 0 && (
                    <div className="space-y-2">
                      <p className="text-xs text-gray-500 uppercase tracking-wider">Gallery ({savedGlyphs.length})</p>
                      <div className="grid grid-cols-4 gap-2">
                        {savedGlyphs.slice(0, 8).map((glyph, idx) => (
                          <img
                            key={idx}
                            src={glyph}
                            alt={`Glyph ${idx + 1}`}
                            className="w-full aspect-square rounded-lg border border-gray-700 bg-gray-900 object-cover"
                          />
                        ))}
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default FocusWellnessTools;
