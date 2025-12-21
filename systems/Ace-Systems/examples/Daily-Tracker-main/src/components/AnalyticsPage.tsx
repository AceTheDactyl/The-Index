/**
 * Analytics Page
 *
 * Comprehensive analytics dashboard accessible via navigation.
 * Includes rhythm patterns, music meditation metrics, emotional coherence tracking,
 * and full DeltaHV metrics integration with neural map visualization.
 */

import React, { useState, useEffect, useMemo } from 'react';
import {
  TrendingUp, BarChart3, Activity, Music,
  Heart, Brain, ArrowLeft, Target,
  Sparkles, Zap, Shield, Waves, AlertTriangle, CheckCircle2,
  Plus, Play, Trash2, ChevronDown, ListMusic, X
} from 'lucide-react';
import {
  musicLibrary,
  EMOTIONAL_CATEGORIES,
  type EmotionalCategoryId,
  type MusicSession
} from '../lib/musicLibrary';
import { MusicLibrary } from './MusicLibrary';
import { metricsHub, type EnhancedDeltaHVState } from '../lib/metricsHub';
import { userProfileService, type MetricsSnapshot as ProfileMetricsSnapshot } from '../lib/userProfile';
import type { DeltaHVState } from '../lib/deltaHVEngine';
import { storyShuffleEngine, type Playlist } from '../lib/storyShuffleEngine';

// Types
interface CheckIn {
  id: string;
  category: string;
  task: string;
  waveId?: string;
  slot: string;
  loggedAt: string;
  done: boolean;
  isAnchor?: boolean;
}

interface Wave {
  id: string;
  name: string;
  color: string;
}

interface AnalyticsPageProps {
  checkIns: CheckIn[];
  waves: Wave[];
  onBack: () => void;
  deltaHV?: DeltaHVState | null;
}

// Color mapping
const COLORS = {
  cyan: '#22d3ee',
  purple: '#a855f7',
  blue: '#3b82f6',
  orange: '#f97316',
  pink: '#ec4899',
  green: '#22c55e',
  amber: '#f59e0b',
  rose: '#f43f5e',
  gray: '#6b7280'
};

// Utility functions
const sameDay = (a: Date, b: Date): boolean =>
  a.getFullYear() === b.getFullYear() &&
  a.getMonth() === b.getMonth() &&
  a.getDate() === b.getDate();

const formatDate = (date: Date): string =>
  date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });

const getDateRange = (days: number): Date[] => {
  const dates: Date[] = [];
  const today = new Date();
  today.setHours(0, 0, 0, 0);

  for (let i = days - 1; i >= 0; i--) {
    const date = new Date(today);
    date.setDate(date.getDate() - i);
    dates.push(date);
  }
  return dates;
};

/**
 * Simple Line Chart
 */
function LineChart({
  data,
  width = 300,
  height = 150,
  color = COLORS.cyan,
  showArea = false
}: {
  data: { label: string; value: number }[];
  width?: number;
  height?: number;
  color?: string;
  showArea?: boolean;
}) {
  if (data.length === 0) return null;

  const padding = { top: 10, right: 10, bottom: 25, left: 35 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  const maxValue = Math.max(...data.map(d => d.value), 1);
  const minValue = Math.min(...data.map(d => d.value), 0);
  const range = maxValue - minValue || 1;

  const points = data.map((d, i) => ({
    x: padding.left + (i / (data.length - 1 || 1)) * chartWidth,
    y: padding.top + chartHeight - ((d.value - minValue) / range) * chartHeight
  }));

  const linePath = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ');
  const areaPath = linePath +
    ` L ${points[points.length - 1].x} ${padding.top + chartHeight}` +
    ` L ${points[0].x} ${padding.top + chartHeight} Z`;

  return (
    <svg width={width} height={height} className="overflow-visible">
      {[0, 0.25, 0.5, 0.75, 1].map(ratio => (
        <line
          key={ratio}
          x1={padding.left}
          y1={padding.top + chartHeight * (1 - ratio)}
          x2={padding.left + chartWidth}
          y2={padding.top + chartHeight * (1 - ratio)}
          stroke="#374151"
          strokeWidth="1"
          strokeDasharray="4 4"
        />
      ))}
      {[0, 0.5, 1].map(ratio => (
        <text
          key={ratio}
          x={padding.left - 5}
          y={padding.top + chartHeight * (1 - ratio) + 4}
          textAnchor="end"
          fontSize="10"
          fill="#6b7280"
        >
          {Math.round(minValue + range * ratio)}
        </text>
      ))}
      {showArea && <path d={areaPath} fill={color} fillOpacity="0.1" />}
      <path d={linePath} fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      {points.map((p, i) => <circle key={i} cx={p.x} cy={p.y} r="3" fill={color} />)}
      {data.map((d, i) => (
        i % Math.ceil(data.length / 5) === 0 && (
          <text key={i} x={points[i].x} y={height - 5} textAnchor="middle" fontSize="9" fill="#6b7280">
            {d.label}
          </text>
        )
      ))}
    </svg>
  );
}

/**
 * Stat Card
 */
function StatCard({
  label,
  value,
  subValue,
  icon: Icon,
  color = 'cyan',
  trend
}: {
  label: string;
  value: string | number;
  subValue?: string;
  icon: React.ElementType;
  color?: string;
  trend?: 'up' | 'down' | 'neutral';
}) {
  const colorClasses: Record<string, string> = {
    cyan: 'from-cyan-950/50 to-cyan-900/30 border-cyan-700/30 text-cyan-400',
    purple: 'from-purple-950/50 to-purple-900/30 border-purple-700/30 text-purple-400',
    amber: 'from-amber-950/50 to-amber-900/30 border-amber-700/30 text-amber-400',
    rose: 'from-rose-950/50 to-rose-900/30 border-rose-700/30 text-rose-400',
    green: 'from-emerald-950/50 to-emerald-900/30 border-emerald-700/30 text-emerald-400',
    pink: 'from-pink-950/50 to-pink-900/30 border-pink-700/30 text-pink-400'
  };

  return (
    <div className={`rounded-xl p-4 bg-gradient-to-br ${colorClasses[color] || colorClasses.cyan} border`}>
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs text-gray-400">{label}</p>
          <p className="text-2xl font-bold mt-1">{value}</p>
          {subValue && <p className="text-xs text-gray-500 mt-1">{subValue}</p>}
        </div>
        <div className="p-2 rounded-lg bg-black/20">
          <Icon className="w-5 h-5" />
        </div>
      </div>
      {trend && (
        <div className={`text-xs mt-2 flex items-center gap-1 ${
          trend === 'up' ? 'text-emerald-400' : trend === 'down' ? 'text-rose-400' : 'text-gray-400'
        }`}>
          {trend === 'up' ? '↑' : trend === 'down' ? '↓' : '→'}
          {trend === 'up' ? 'Improving' : trend === 'down' ? 'Declining' : 'Stable'}
        </div>
      )}
    </div>
  );
}

/**
 * DeltaHV Metric Display Card
 */
function DeltaHVCard({
  metric,
  value,
  regions,
  color
}: {
  metric: 'symbolic' | 'resonance' | 'friction' | 'stability';
  value: number;
  regions: Array<{ glyph: string; name: string; active: boolean }>;
  color: string;
}) {
  const labels = {
    symbolic: { name: 'Symbolic (S)', desc: 'Meaning & intention density', icon: Sparkles },
    resonance: { name: 'Resonance (R)', desc: 'Alignment with rhythm', icon: Waves },
    friction: { name: 'Friction (δφ)', desc: 'Resistance in flow', icon: Zap },
    stability: { name: 'Stability (H)', desc: 'Harmonic coherence', icon: Shield },
  };

  const { name, desc, icon: Icon } = labels[metric];

  return (
    <div className="rounded-xl bg-gray-900/60 border border-gray-800 p-4 space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className={`p-2 rounded-lg bg-${color}-500/20`}>
            <Icon className={`w-5 h-5 text-${color}-400`} />
          </div>
          <div>
            <p className="font-medium text-sm">{name}</p>
            <p className="text-xs text-gray-500">{desc}</p>
          </div>
        </div>
        <div className="text-right">
          <p className={`text-2xl font-bold text-${color}-400`}>{value}</p>
          <p className="text-xs text-gray-500">/100</p>
        </div>
      </div>

      {/* Progress bar */}
      <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500`}
          style={{ width: `${value}%`, backgroundColor: `var(--color-${color}-500, ${COLORS[color as keyof typeof COLORS] || COLORS.cyan})` }}
        />
      </div>

      {/* Brain regions */}
      <div className="flex flex-wrap gap-1">
        {regions.map((region, i) => (
          <span
            key={i}
            className={`text-xs px-2 py-0.5 rounded-full ${
              region.active
                ? `bg-${color}-500/30 text-${color}-300`
                : 'bg-gray-800/50 text-gray-600'
            }`}
            title={region.name}
          >
            {region.glyph}
          </span>
        ))}
      </div>
    </div>
  );
}

/**
 * Health Insights Panel - Detection for good AND bad health signs
 */
function HealthInsights({
  metricsState,
  profileHistory
}: {
  metricsState: EnhancedDeltaHVState | null;
  profileHistory: ProfileMetricsSnapshot[];
}) {
  if (!metricsState) return null;

  const insights: Array<{ type: 'positive' | 'warning' | 'neutral'; message: string; icon: React.ElementType }> = [];

  // Check symbolic density
  if (metricsState.symbolicDensity >= 70) {
    insights.push({
      type: 'positive',
      message: 'High symbolic engagement - your intentions are clear',
      icon: CheckCircle2
    });
  } else if (metricsState.symbolicDensity < 30) {
    insights.push({
      type: 'warning',
      message: 'Low symbolic density - try journaling or setting intentions',
      icon: AlertTriangle
    });
  }

  // Check resonance
  if (metricsState.resonanceCoupling >= 70) {
    insights.push({
      type: 'positive',
      message: 'Excellent rhythm alignment - you\'re in sync with your schedule',
      icon: CheckCircle2
    });
  } else if (metricsState.resonanceCoupling < 30) {
    insights.push({
      type: 'warning',
      message: 'Low resonance - tasks may not align with your natural rhythm',
      icon: AlertTriangle
    });
  }

  // Check friction
  if (metricsState.frictionCoefficient <= 30) {
    insights.push({
      type: 'positive',
      message: 'Low friction - smooth flow through your day',
      icon: CheckCircle2
    });
  } else if (metricsState.frictionCoefficient >= 70) {
    insights.push({
      type: 'warning',
      message: 'High friction detected - consider simplifying or delegating',
      icon: AlertTriangle
    });
  }

  // Check stability
  if (metricsState.harmonicStability >= 70) {
    insights.push({
      type: 'positive',
      message: 'Strong harmonic stability - consistent patterns established',
      icon: CheckCircle2
    });
  } else if (metricsState.harmonicStability < 30) {
    insights.push({
      type: 'warning',
      message: 'Low stability - try establishing anchor routines',
      icon: AlertTriangle
    });
  }

  // Check overall coherence
  if (metricsState.fieldState === 'coherent') {
    insights.unshift({
      type: 'positive',
      message: 'You\'re in a coherent state! Excellent work maintaining balance.',
      icon: Sparkles
    });
  }

  // Check for positive trends in profile history
  if (profileHistory.length >= 3) {
    const recent = profileHistory.slice(-3);
    const avgDeltaHV = recent.reduce((sum, s) => sum + s.deltaHV, 0) / recent.length;
    const firstDeltaHV = recent[0].deltaHV;

    if (avgDeltaHV > firstDeltaHV + 10) {
      insights.push({
        type: 'positive',
        message: 'Your DeltaHV is trending upward - keep up the momentum!',
        icon: TrendingUp
      });
    }
  }

  if (insights.length === 0) return null;

  return (
    <div className="rounded-xl bg-gray-900/50 border border-gray-800 p-4">
      <h4 className="text-sm font-medium text-gray-300 mb-3 flex items-center gap-2">
        <Heart className="w-4 h-4 text-pink-400" />
        Health Insights
      </h4>
      <div className="space-y-2">
        {insights.map((insight, i) => (
          <div
            key={i}
            className={`flex items-start gap-2 p-2 rounded-lg ${
              insight.type === 'positive' ? 'bg-emerald-500/10 text-emerald-300' :
              insight.type === 'warning' ? 'bg-amber-500/10 text-amber-300' :
              'bg-gray-500/10 text-gray-300'
            }`}
          >
            <insight.icon className="w-4 h-4 mt-0.5 flex-shrink-0" />
            <p className="text-sm">{insight.message}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

/**
 * Library with Playlist Manager Component
 */
function LibraryWithPlaylists() {
  const [playlists, setPlaylists] = useState<Playlist[]>([]);
  const [tracks, setTracks] = useState<any[]>([]);
  const [expandedPlaylist, setExpandedPlaylist] = useState<string | null>(null);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newPlaylistName, setNewPlaylistName] = useState('');
  const [newPlaylistEmoji, setNewPlaylistEmoji] = useState('🎵');
  // Collapsible sections
  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({
    userPlaylists: false,
    systemPlaylists: false,
    musicLibrary: false
  });

  const toggleSection = (section: string) => {
    setExpandedSections(prev => ({ ...prev, [section]: !prev[section] }));
  };

  useEffect(() => {
    const loadData = async () => {
      await musicLibrary.initialize();
      const allTracks = await musicLibrary.getAllTracks();
      setTracks(allTracks);
      setPlaylists(storyShuffleEngine.getPlaylists());
    };
    loadData();

    const unsub = storyShuffleEngine.subscribe(() => {
      setPlaylists(storyShuffleEngine.getPlaylists());
    });
    return () => unsub();
  }, []);

  const createPlaylist = () => {
    if (!newPlaylistName.trim()) return;
    storyShuffleEngine.createPlaylist(newPlaylistName.trim(), undefined, newPlaylistEmoji);
    setNewPlaylistName('');
    setNewPlaylistEmoji('🎵');
    setShowCreateModal(false);
  };

  const deletePlaylist = (playlistId: string) => {
    storyShuffleEngine.deletePlaylist(playlistId);
  };

  const removeTrackFromPlaylist = (playlistId: string, trackId: string) => {
    storyShuffleEngine.removeFromPlaylist(playlistId, trackId);
  };

  const playPlaylist = async (playlist: Playlist) => {
    if (playlist.trackIds.length > 0) {
      const firstTrack = tracks.find(t => t.id === playlist.trackIds[0]);
      if (firstTrack) {
        await musicLibrary.playTrack(firstTrack.id, firstTrack.categoryId);
      }
    }
  };

  const userPlaylists = playlists.filter(p => !p.isSystem);
  const systemPlaylists = playlists.filter(p => p.isSystem);

  return (
    <div className="space-y-4">
      <h2 className="text-2xl font-light flex items-center gap-3">
        <Music className="w-7 h-7 text-purple-400" />
        Music Library
      </h2>

      {/* User Playlists Section - Collapsible */}
      <div className="bg-gray-950/60 backdrop-blur border border-gray-800 rounded-xl overflow-hidden">
        <button
          onClick={() => toggleSection('userPlaylists')}
          className="w-full flex items-center justify-between p-4 hover:bg-gray-900/50 transition-colors"
        >
          <div className="flex items-center gap-3">
            <ListMusic className="w-5 h-5 text-cyan-400" />
            <h3 className="text-lg font-medium text-white">Your Playlists</h3>
            <span className="text-sm text-gray-500">({userPlaylists.length})</span>
          </div>
          <ChevronDown className={`w-5 h-5 text-gray-400 transition-transform ${expandedSections.userPlaylists ? 'rotate-180' : ''}`} />
        </button>

        {expandedSections.userPlaylists && (
          <div className="px-4 pb-4 border-t border-gray-800">
            <div className="flex justify-end mt-3 mb-3">
              <button
                onClick={() => setShowCreateModal(true)}
                className="flex items-center gap-2 px-3 py-1.5 bg-purple-600/20 hover:bg-purple-600/30 text-purple-300 rounded-lg text-sm transition-colors"
              >
                <Plus className="w-4 h-4" />
                New Playlist
              </button>
            </div>

            {userPlaylists.length > 0 ? (
              <div className="space-y-3">
                {userPlaylists.map(playlist => (
                  <div
                    key={playlist.id}
                    className="bg-gray-900/50 rounded-xl border border-gray-800 overflow-hidden"
                  >
                    <div className="flex items-center gap-4 p-3">
                      <span className="text-2xl">{playlist.coverEmoji}</span>
                      <div className="flex-1">
                        <h4 className="text-white font-medium text-sm">{playlist.name}</h4>
                        <p className="text-xs text-gray-400">{playlist.trackIds.length} tracks</p>
                      </div>
                      <button
                        onClick={() => playPlaylist(playlist)}
                        className="p-1.5 bg-purple-600/20 hover:bg-purple-600/30 rounded-lg transition-colors"
                        disabled={playlist.trackIds.length === 0}
                      >
                        <Play className="w-4 h-4 text-purple-300" />
                      </button>
                      <button
                        onClick={() => setExpandedPlaylist(expandedPlaylist === playlist.id ? null : playlist.id)}
                        className="p-1.5 bg-gray-800 hover:bg-gray-700 rounded-lg transition-colors"
                      >
                        <ChevronDown className={`w-4 h-4 text-gray-400 transition-transform ${
                          expandedPlaylist === playlist.id ? 'rotate-180' : ''
                        }`} />
                      </button>
                      <button
                        onClick={() => deletePlaylist(playlist.id)}
                        className="p-1.5 bg-red-500/10 hover:bg-red-500/20 rounded-lg transition-colors"
                      >
                        <Trash2 className="w-4 h-4 text-red-400" />
                      </button>
                    </div>

                    {expandedPlaylist === playlist.id && (
                      <div className="px-3 pb-3 border-t border-gray-800">
                        {playlist.trackIds.length > 0 ? (
                          <div className="space-y-1.5 mt-2">
                            {playlist.trackIds.map((trackId, idx) => {
                              const track = tracks.find(t => t.id === trackId);
                              if (!track) return null;
                              return (
                                <div
                                  key={trackId}
                                  className="flex items-center gap-2 p-2 bg-gray-800/50 rounded-lg text-sm"
                                >
                                  <span className="text-xs text-gray-500 w-5">{idx + 1}</span>
                                  <span>{EMOTIONAL_CATEGORIES[track.categoryId as EmotionalCategoryId]?.icon}</span>
                                  <div className="flex-1 truncate">
                                    <span className="text-white">{track.name}</span>
                                  </div>
                                  <button
                                    onClick={() => removeTrackFromPlaylist(playlist.id, trackId)}
                                    className="p-1 hover:bg-red-500/20 rounded transition-colors"
                                  >
                                    <X className="w-3 h-3 text-red-400" />
                                  </button>
                                </div>
                              );
                            })}
                          </div>
                        ) : (
                          <p className="text-xs text-gray-500 text-center py-3">
                            No tracks yet. Add songs from the DJ tab.
                          </p>
                        )}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-6 bg-gray-900/30 rounded-xl border border-gray-800">
                <ListMusic className="w-8 h-8 mx-auto mb-2 text-gray-600" />
                <p className="text-sm text-gray-400">No custom playlists yet</p>
              </div>
            )}
          </div>
        )}
      </div>

      {/* System Playlists Section - Collapsible */}
      <div className="bg-gray-950/60 backdrop-blur border border-gray-800 rounded-xl overflow-hidden">
        <button
          onClick={() => toggleSection('systemPlaylists')}
          className="w-full flex items-center justify-between p-4 hover:bg-gray-900/50 transition-colors"
        >
          <div className="flex items-center gap-3">
            <Sparkles className="w-5 h-5 text-purple-400" />
            <h3 className="text-lg font-medium text-white">AI-Generated Playlists</h3>
            <span className="text-sm text-gray-500">({systemPlaylists.length})</span>
          </div>
          <ChevronDown className={`w-5 h-5 text-gray-400 transition-transform ${expandedSections.systemPlaylists ? 'rotate-180' : ''}`} />
        </button>

        {expandedSections.systemPlaylists && (
          <div className="px-4 pb-4 border-t border-gray-800">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mt-3">
              {systemPlaylists.map(playlist => (
                <button
                  key={playlist.id}
                  onClick={() => playPlaylist(playlist)}
                  className="p-3 bg-gray-900/50 border border-gray-800 rounded-lg hover:border-gray-700 transition-all text-left"
                  disabled={playlist.trackIds.length === 0}
                >
                  <span className="text-xl block mb-1">{playlist.coverEmoji}</span>
                  <h5 className="text-xs text-white font-medium truncate">{playlist.name}</h5>
                  <p className="text-xs text-gray-500">{playlist.trackIds.length} tracks</p>
                  {playlist.basedOnMetric && (
                    <span className={`mt-1 inline-block px-1.5 py-0.5 rounded text-xs ${
                      playlist.basedOnMetric === 'symbolic' ? 'bg-purple-500/20 text-purple-300' :
                      playlist.basedOnMetric === 'resonance' ? 'bg-cyan-500/20 text-cyan-300' :
                      playlist.basedOnMetric === 'friction' ? 'bg-amber-500/20 text-amber-300' :
                      'bg-green-500/20 text-green-300'
                    }`}>
                      {playlist.basedOnMetric}
                    </span>
                  )}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Music Library Section - Collapsible */}
      <div className="bg-gray-950/60 backdrop-blur border border-gray-800 rounded-xl overflow-hidden">
        <button
          onClick={() => toggleSection('musicLibrary')}
          className="w-full flex items-center justify-between p-4 hover:bg-gray-900/50 transition-colors"
        >
          <div className="flex items-center gap-3">
            <Music className="w-5 h-5 text-pink-400" />
            <h3 className="text-lg font-medium text-white">Full Music Library</h3>
            <span className="text-sm text-gray-500">({tracks.length} tracks)</span>
          </div>
          <ChevronDown className={`w-5 h-5 text-gray-400 transition-transform ${expandedSections.musicLibrary ? 'rotate-180' : ''}`} />
        </button>

        {expandedSections.musicLibrary && (
          <div className="border-t border-gray-800">
            <MusicLibrary compact />
          </div>
        )}
      </div>

      {/* Create Playlist Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-black/80 z-50 flex items-center justify-center p-4">
          <div className="bg-gray-900 rounded-2xl border border-gray-800 w-full max-w-md p-6">
            <h3 className="text-lg font-medium text-white mb-4">Create New Playlist</h3>
            <div className="space-y-4">
              <div>
                <label className="text-sm text-gray-400 block mb-2">Playlist Name</label>
                <input
                  type="text"
                  value={newPlaylistName}
                  onChange={(e) => setNewPlaylistName(e.target.value)}
                  placeholder="My awesome playlist..."
                  className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2.5 text-white focus:border-purple-500 focus:outline-none"
                  autoFocus
                />
              </div>
              <div>
                <label className="text-sm text-gray-400 block mb-2">Choose an Emoji</label>
                <div className="flex gap-2 flex-wrap">
                  {['🎵', '🎶', '🎸', '🎹', '🎤', '🎧', '💜', '🌙', '☀️', '⚡', '🔥', '🌊', '🌈', '✨', '💫', '🎯'].map(emoji => (
                    <button
                      key={emoji}
                      onClick={() => setNewPlaylistEmoji(emoji)}
                      className={`w-10 h-10 rounded-lg flex items-center justify-center text-xl transition-all ${
                        newPlaylistEmoji === emoji
                          ? 'bg-purple-600/30 border-2 border-purple-400'
                          : 'bg-gray-800 border border-gray-700 hover:border-gray-600'
                      }`}
                    >
                      {emoji}
                    </button>
                  ))}
                </div>
              </div>
            </div>
            <div className="flex gap-3 mt-6">
              <button
                onClick={() => {
                  setShowCreateModal(false);
                  setNewPlaylistName('');
                }}
                className="flex-1 px-4 py-2.5 bg-gray-800 hover:bg-gray-700 text-gray-300 rounded-lg transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={createPlaylist}
                disabled={!newPlaylistName.trim()}
                className="flex-1 px-4 py-2.5 bg-purple-600 hover:bg-purple-500 text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Create
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * Main Analytics Page Component
 */
export function AnalyticsPage({ checkIns, waves: _waves, onBack }: AnalyticsPageProps) {
  const [activeTab, setActiveTab] = useState<'overview' | 'coherence' | 'library'>('overview');
  const [timeRange, setTimeRange] = useState<7 | 14 | 30>(7);
  const [musicSessions, setMusicSessions] = useState<MusicSession[]>([]);
  // Backend data kept for metrics but UI removed
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [coherenceStats, setCoherenceStats] = useState<any>(null);

  // Real-time metrics from metricsHub
  const [metricsState, setMetricsState] = useState<EnhancedDeltaHVState | null>(null);
  const [profileHistory, setProfileHistory] = useState<ProfileMetricsSnapshot[]>([]);

  // Load music data and subscribe to changes (backend tracking)
  useEffect(() => {
    const loadMusicData = async () => {
      await musicLibrary.initialize();
      const sessions = await musicLibrary.getAllSessions();
      setMusicSessions(sessions);

      const stats = await musicLibrary.getCoherenceStats(timeRange);
      setCoherenceStats(stats);
    };

    loadMusicData();

    // Subscribe to music library changes for real-time updates
    const unsubscribe = musicLibrary.subscribe(() => {
      loadMusicData();
    });

    return () => unsubscribe();
  }, [timeRange]);

  // Subscribe to real-time metrics from metricsHub
  useEffect(() => {
    const unsubscribe = metricsHub.subscribe((state) => {
      setMetricsState(state);
    });

    // Get initial state
    const initial = metricsHub.getState();
    if (initial) setMetricsState(initial);

    return unsubscribe;
  }, []);

  // Load profile history for cross-referencing
  useEffect(() => {
    const loadProfileData = async () => {
      await userProfileService.initialize();
      const history = userProfileService.getMetricsHistory(timeRange);
      setProfileHistory(history);
    };

    loadProfileData();
  }, [timeRange]);

  // Calculate rhythm analytics
  const rhythmAnalytics = useMemo(() => {
    const dates = getDateRange(timeRange);
    const today = new Date();
    today.setHours(0, 0, 0, 0);

    const dailyStats = dates.map(date => {
      const dayCheckIns = checkIns.filter(c => sameDay(new Date(c.slot), date));
      const completed = dayCheckIns.filter(c => c.done).length;
      const total = dayCheckIns.length;
      return {
        date,
        label: formatDate(date),
        completed,
        total,
        completionRate: total > 0 ? Math.round((completed / total) * 100) : 0
      };
    });

    const categoryStats: Record<string, { total: number; completed: number }> = {};
    checkIns.forEach(c => {
      const date = new Date(c.slot);
      if (date >= dates[0] && date <= dates[dates.length - 1]) {
        if (!categoryStats[c.category]) categoryStats[c.category] = { total: 0, completed: 0 };
        categoryStats[c.category].total++;
        if (c.done) categoryStats[c.category].completed++;
      }
    });

    const waveStats: Record<string, number> = {};
    checkIns.forEach(c => {
      const date = new Date(c.slot);
      if (date >= dates[0] && date <= dates[dates.length - 1] && c.waveId) {
        waveStats[c.waveId] = (waveStats[c.waveId] || 0) + 1;
      }
    });

    const totalTasks = dailyStats.reduce((sum, d) => sum + d.total, 0);
    const completedTasks = dailyStats.reduce((sum, d) => sum + d.completed, 0);

    const midpoint = Math.floor(dailyStats.length / 2);
    const firstHalf = dailyStats.slice(0, midpoint);
    const secondHalf = dailyStats.slice(midpoint);
    const firstHalfAvg = firstHalf.length > 0 ? firstHalf.reduce((sum, d) => sum + d.completionRate, 0) / firstHalf.length : 0;
    const secondHalfAvg = secondHalf.length > 0 ? secondHalf.reduce((sum, d) => sum + d.completionRate, 0) / secondHalf.length : 0;
    const trend: 'up' | 'down' | 'neutral' = secondHalfAvg > firstHalfAvg + 5 ? 'up' : secondHalfAvg < firstHalfAvg - 5 ? 'down' : 'neutral';

    return { dailyStats, categoryStats, waveStats, totalTasks, completedTasks, trend };
  }, [checkIns, timeRange]);

  // Music session analytics - now also includes listening time breakdown for donut chart
  const musicAnalytics = useMemo(() => {
    const dates = getDateRange(timeRange);
    const filteredSessions = musicSessions.filter(s => {
      const date = new Date(s.startedAt);
      return date >= dates[0] && date <= dates[dates.length - 1];
    });

    const dailyListening = dates.map(date => {
      const daySessions = filteredSessions.filter(s => sameDay(new Date(s.startedAt), date));
      const totalDuration = daySessions.reduce((sum, s) => sum + s.duration, 0);
      return { label: formatDate(date), value: Math.round(totalDuration / 60) };
    });

    // Calculate category usage by sessions AND listening time for better accuracy
    const categoryUsage = Object.entries(EMOTIONAL_CATEGORIES).map(([key, cat]) => {
      const catSessions = filteredSessions.filter(s => s.categoryId === key);
      const totalListeningTime = catSessions.reduce((sum, s) => sum + s.duration, 0);
      return {
        label: cat.name,
        value: catSessions.length,
        listeningTime: totalListeningTime,
        color: cat.color,
        icon: cat.icon
      };
    }).filter(c => c.value > 0);

    // Sessions with post-session data for mood tracking
    const sessionsWithMoodData = filteredSessions.filter(
      s => s.postSession?.moodBefore !== undefined && s.postSession?.moodAfter !== undefined
    );

    // Calculate mood improvement stats
    const moodChanges = sessionsWithMoodData.map(s =>
      (s.postSession?.moodAfter || 0) - (s.postSession?.moodBefore || 0)
    );
    const avgMoodChange = moodChanges.length > 0
      ? moodChanges.reduce((a, b) => a + b, 0) / moodChanges.length
      : 0;

    return {
      dailyListening,
      categoryUsage,
      totalSessions: filteredSessions.length,
      sessionsWithData: sessionsWithMoodData.length,
      avgMoodChange
    };
  }, [musicSessions, timeRange]);

  const tabs = [
    { id: 'overview', label: 'Overview', icon: BarChart3 },
    { id: 'coherence', label: 'Coherence', icon: Heart },
    { id: 'library', label: 'Library', icon: Music }
  ] as const;

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      {/* Header */}
      <div className="sticky top-0 z-10 bg-gray-950/95 backdrop-blur border-b border-gray-800">
        <div className="max-w-6xl mx-auto px-4 py-3">
          <div className="flex items-center justify-between">
            <button
              onClick={onBack}
              className="flex items-center gap-2 text-gray-400 hover:text-white transition-colors"
            >
              <ArrowLeft className="w-5 h-5" />
              <span>Back to Dashboard</span>
            </button>

            <div className="flex items-center gap-2">
              {([7, 14, 30] as const).map(days => (
                <button
                  key={days}
                  onClick={() => setTimeRange(days)}
                  className={`px-3 py-1.5 rounded-lg text-sm ${
                    timeRange === days
                      ? 'bg-purple-600 text-white'
                      : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
                  }`}
                >
                  {days}D
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-800">
        <div className="max-w-6xl mx-auto px-4">
          <div className="flex gap-1 overflow-x-auto py-2">
            {tabs.map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-4 py-2 rounded-lg text-sm flex items-center gap-2 whitespace-nowrap transition-colors ${
                  activeTab === tab.id
                    ? 'bg-purple-600 text-white'
                    : 'text-gray-400 hover:text-white hover:bg-gray-800'
                }`}
              >
                <tab.icon className="w-4 h-4" />
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-6xl mx-auto px-4 py-6">
        {/* Overview Tab */}
        {activeTab === 'overview' && (
          <div className="space-y-6">
            <h2 className="text-2xl font-light flex items-center gap-3">
              <BarChart3 className="w-7 h-7 text-purple-400" />
              Analytics Overview
            </h2>

            {/* Stat Cards */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <StatCard
                label="Task Completion"
                value={`${rhythmAnalytics.totalTasks > 0 ? Math.round((rhythmAnalytics.completedTasks / rhythmAnalytics.totalTasks) * 100) : 0}%`}
                subValue={`${rhythmAnalytics.completedTasks}/${rhythmAnalytics.totalTasks} tasks`}
                icon={Target}
                color="cyan"
                trend={rhythmAnalytics.trend}
              />
              <StatCard
                label="Music Sessions"
                value={musicAnalytics.totalSessions}
                subValue={`Over ${timeRange} days`}
                icon={Music}
                color="purple"
              />
              <StatCard
                label="Coherence Rate"
                value={coherenceStats ? `${Math.round(coherenceStats.coherenceRate)}%` : '—'}
                subValue="Desired vs Experienced"
                icon={Heart}
                color="pink"
              />
              <StatCard
                label="Mood Improvement"
                value={coherenceStats ? `+${coherenceStats.avgMoodImprovement.toFixed(1)}` : '—'}
                subValue="Avg session change"
                icon={TrendingUp}
                color="green"
              />
            </div>

            {/* Charts Row */}
            <div className="grid md:grid-cols-2 gap-6">
              <div className="rounded-xl bg-gray-900/50 border border-gray-800 p-4">
                <h3 className="text-sm font-medium text-gray-300 mb-4 flex items-center gap-2">
                  <Activity className="w-4 h-4 text-cyan-400" />
                  Task Completion Trend
                </h3>
                <LineChart
                  data={rhythmAnalytics.dailyStats.map(d => ({ label: d.label, value: d.completionRate }))}
                  width={350}
                  height={180}
                  color={COLORS.cyan}
                  showArea
                />
              </div>

              <div className="rounded-xl bg-gray-900/50 border border-gray-800 p-4">
                <h3 className="text-sm font-medium text-gray-300 mb-4 flex items-center gap-2">
                  <Music className="w-4 h-4 text-purple-400" />
                  Daily Listening (minutes)
                </h3>
                <LineChart
                  data={musicAnalytics.dailyListening}
                  width={350}
                  height={180}
                  color={COLORS.purple}
                  showArea
                />
              </div>
            </div>

            {/* DeltaHV Metrics Section */}
            <div className="space-y-4">
              <h3 className="text-lg font-medium flex items-center gap-2">
                <Brain className="w-5 h-5 text-purple-400" />
                DeltaHV Metrics (Real-Time)
                {metricsState?.isLive && (
                  <span className="text-xs px-2 py-0.5 bg-emerald-500/20 text-emerald-400 rounded-full">LIVE</span>
                )}
              </h3>

              {metricsState ? (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <DeltaHVCard
                    metric="symbolic"
                    value={metricsState.symbolicDensity}
                    regions={metricsHub.getMetricDisplay('symbolic').regions}
                    color="purple"
                  />
                  <DeltaHVCard
                    metric="resonance"
                    value={metricsState.resonanceCoupling}
                    regions={metricsHub.getMetricDisplay('resonance').regions}
                    color="cyan"
                  />
                  <DeltaHVCard
                    metric="friction"
                    value={metricsState.frictionCoefficient}
                    regions={metricsHub.getMetricDisplay('friction').regions}
                    color="amber"
                  />
                  <DeltaHVCard
                    metric="stability"
                    value={metricsState.harmonicStability}
                    regions={metricsHub.getMetricDisplay('stability').regions}
                    color="green"
                  />
                </div>
              ) : (
                <div className="text-center py-8 text-gray-500">
                  <Brain className="w-8 h-8 mx-auto mb-2 opacity-50" />
                  <p className="text-sm">No real-time metrics available yet</p>
                  <p className="text-xs mt-1">Complete tasks and activities to see your DeltaHV state</p>
                </div>
              )}
            </div>

            {/* Health Insights */}
            <HealthInsights metricsState={metricsState} profileHistory={profileHistory} />

            {/* Profile History Trend */}
            {profileHistory.length > 0 && (
              <div className="rounded-xl bg-gray-900/50 border border-gray-800 p-4">
                <h3 className="text-sm font-medium text-gray-300 mb-4 flex items-center gap-2">
                  <TrendingUp className="w-4 h-4 text-emerald-400" />
                  DeltaHV History ({timeRange} days)
                </h3>
                <LineChart
                  data={profileHistory.map(p => ({ label: new Date(p.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }), value: p.deltaHV }))}
                  width={700}
                  height={180}
                  color={COLORS.purple}
                  showArea
                />
              </div>
            )}
          </div>
        )}

        {/* Coherence Tab */}
        {activeTab === 'coherence' && (
          <div className="space-y-6">
            <h2 className="text-2xl font-light flex items-center gap-3">
              <Heart className="w-7 h-7 text-pink-400" />
              Emotional Coherence
              {metricsState?.isLive && (
                <span className="text-xs px-2 py-0.5 bg-emerald-500/20 text-emerald-400 rounded-full">LIVE</span>
              )}
            </h2>

            {/* Current Field State */}
            {metricsState && (
              <div className={`rounded-xl p-6 border text-center ${
                metricsState.fieldState === 'coherent' ? 'bg-emerald-950/30 border-emerald-500/30' :
                metricsState.fieldState === 'transitioning' ? 'bg-amber-950/30 border-amber-500/30' :
                metricsState.fieldState === 'fragmented' ? 'bg-rose-950/30 border-rose-500/30' :
                'bg-gray-900/50 border-gray-700'
              }`}>
                <div className="flex items-center justify-center gap-3 mb-3">
                  <Brain className={`w-10 h-10 ${
                    metricsState.fieldState === 'coherent' ? 'text-emerald-400' :
                    metricsState.fieldState === 'transitioning' ? 'text-amber-400' :
                    metricsState.fieldState === 'fragmented' ? 'text-rose-400' :
                    'text-gray-400'
                  }`} />
                  <div>
                    <p className="text-3xl font-bold">{metricsState.deltaHV}</p>
                    <p className="text-sm text-gray-400">Overall Coherence Score</p>
                  </div>
                </div>
                <p className={`text-lg font-medium capitalize ${
                  metricsState.fieldState === 'coherent' ? 'text-emerald-300' :
                  metricsState.fieldState === 'transitioning' ? 'text-amber-300' :
                  metricsState.fieldState === 'fragmented' ? 'text-rose-300' :
                  'text-gray-300'
                }`}>
                  {metricsState.fieldState} Field State
                </p>
                <p className="text-sm text-gray-500 mt-1">
                  {metricsState.fieldState === 'coherent' && 'Excellent! High alignment between intention and action'}
                  {metricsState.fieldState === 'transitioning' && 'Building momentum toward coherence'}
                  {metricsState.fieldState === 'fragmented' && 'Multiple focus points - consider simplifying'}
                  {metricsState.fieldState === 'dormant' && 'Low activity - engage to build coherence'}
                </p>
              </div>
            )}

            {/* Core Coherence Metrics */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <StatCard
                label="Coherence Rate"
                value={metricsState ? `${metricsState.harmonicStability}%` : '—'}
                subValue="Harmonic stability"
                icon={Target}
                color="pink"
              />
              <StatCard
                label="Resonance"
                value={metricsState ? `${metricsState.resonanceCoupling}%` : '—'}
                subValue="Rhythm alignment"
                icon={Waves}
                color="cyan"
              />
              <StatCard
                label="Friction Level"
                value={metricsState ? `${metricsState.frictionCoefficient}%` : '—'}
                subValue={metricsState && metricsState.frictionCoefficient <= 30 ? 'Low friction ✓' : 'Room to improve'}
                icon={Zap}
                color={metricsState && metricsState.frictionCoefficient <= 30 ? 'green' : 'amber'}
              />
              <StatCard
                label="Symbolic Density"
                value={metricsState ? `${metricsState.symbolicDensity}%` : '—'}
                subValue="Intention clarity"
                icon={Sparkles}
                color="purple"
              />
            </div>

            {/* Mood Change Over Time */}
            {profileHistory.length > 0 && (
              <div className="rounded-xl bg-gray-900/50 border border-gray-800 p-5">
                <h3 className="text-sm font-medium text-gray-300 mb-4 flex items-center gap-2">
                  <TrendingUp className="w-4 h-4 text-emerald-400" />
                  Coherence Trend ({timeRange} days)
                </h3>
                <div className="grid md:grid-cols-2 gap-6">
                  <div>
                    <p className="text-xs text-gray-500 mb-2">ΔHV Score Over Time</p>
                    <LineChart
                      data={profileHistory.map(p => ({
                        label: new Date(p.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
                        value: p.deltaHV
                      }))}
                      width={300}
                      height={150}
                      color={COLORS.pink}
                      showArea
                    />
                    {profileHistory.length >= 2 && (
                      <div className="mt-2 text-sm">
                        {(() => {
                          const first = profileHistory[0].deltaHV;
                          const last = profileHistory[profileHistory.length - 1].deltaHV;
                          const change = last - first;
                          return (
                            <span className={change > 0 ? 'text-emerald-400' : change < 0 ? 'text-rose-400' : 'text-gray-400'}>
                              {change > 0 ? '↑' : change < 0 ? '↓' : '→'} {Math.abs(change).toFixed(0)} points {change > 0 ? 'improvement' : change < 0 ? 'decline' : 'stable'}
                            </span>
                          );
                        })()}
                      </div>
                    )}
                  </div>
                  <div>
                    <p className="text-xs text-gray-500 mb-2">Stability vs Friction</p>
                    <LineChart
                      data={profileHistory.map(p => ({
                        label: new Date(p.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
                        value: Math.round(p.harmonicStability - p.frictionCoefficient + 50) // Normalized to 0-100 scale
                      }))}
                      width={300}
                      height={150}
                      color={COLORS.cyan}
                      showArea
                    />
                    <p className="mt-2 text-xs text-gray-500">Higher = more stable, lower friction</p>
                  </div>
                </div>
              </div>
            )}

            {/* Daily Coherence Metrics */}
            {profileHistory.length > 0 && (
              <div className="rounded-xl bg-gray-900/50 border border-gray-800 p-5">
                <h3 className="text-sm font-medium text-gray-300 mb-4">Daily Coherence Breakdown</h3>
                <div className="space-y-3 max-h-64 overflow-y-auto">
                  {profileHistory.slice().reverse().slice(0, 7).map((day, i) => {
                    const fieldState = day.deltaHV >= 70 ? 'coherent' :
                                       day.deltaHV >= 40 ? 'transitioning' :
                                       day.deltaHV > 0 ? 'fragmented' : 'dormant';
                    return (
                      <div key={i} className="flex items-center gap-4 p-3 rounded-lg bg-gray-800/50">
                        <div className="w-20 text-sm text-gray-400">
                          {new Date(day.date).toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' })}
                        </div>
                        <div className="flex-1">
                          <div className="h-3 bg-gray-700 rounded-full overflow-hidden">
                            <div
                              className={`h-full rounded-full transition-all ${
                                fieldState === 'coherent' ? 'bg-emerald-500' :
                                fieldState === 'transitioning' ? 'bg-amber-500' :
                                fieldState === 'fragmented' ? 'bg-rose-500' :
                                'bg-gray-600'
                              }`}
                              style={{ width: `${day.deltaHV}%` }}
                            />
                          </div>
                        </div>
                        <div className="w-16 text-right">
                          <span className={`text-sm font-medium ${
                            fieldState === 'coherent' ? 'text-emerald-400' :
                            fieldState === 'transitioning' ? 'text-amber-400' :
                            fieldState === 'fragmented' ? 'text-rose-400' :
                            'text-gray-400'
                          }`}>
                            {day.deltaHV}
                          </span>
                        </div>
                        <div className="w-24 text-right text-xs text-gray-500 capitalize">
                          {fieldState}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {/* Health Insights */}
            <HealthInsights metricsState={metricsState} profileHistory={profileHistory} />

            {/* Empty State */}
            {!metricsState && profileHistory.length === 0 && (
              <div className="text-center py-12 text-gray-500">
                <Heart className="w-12 h-12 mx-auto mb-3 opacity-30" />
                <p className="text-lg">No coherence data yet</p>
                <p className="text-sm mt-2">Complete tasks and activities to track your emotional coherence</p>
              </div>
            )}
          </div>
        )}

        {/* Library Tab */}
        {activeTab === 'library' && (
          <LibraryWithPlaylists />
        )}
      </div>
    </div>
  );
}

export default AnalyticsPage;
