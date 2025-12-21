import React, { useState, useMemo, useEffect } from 'react';
import {
  Brain,
  Activity,
  Target,
  AlertTriangle,
  ChevronRight,
  ChevronDown,
  X,
  BookOpen,
  Music,
  CheckSquare,
  Sparkles,
  TrendingUp,
  Calendar,
  ListTodo,
  Zap,
} from 'lucide-react';
import type { DeltaHVState, JournalEntry } from '../lib/deltaHVEngine';
import {
  generateNeuralMap,
  getRegionsNeedingAttention,
  type NeuralActivation,
  type GlyphicResonanceField,
} from '../lib/neuralMapEngine';
import { storyShuffleEngine, type Playlist } from '../lib/storyShuffleEngine';
import { userProfileService, type HabitTrack } from '../lib/userProfile';

interface NeuralMapViewProps {
  deltaHV: DeltaHVState;
  journals?: Record<string, JournalEntry[]>;
  onSelectBeat?: (category: string) => void;
}

type ViewMode = 'dashboard' | 'map' | 'journals' | 'metrics';

export const NeuralMapView: React.FC<NeuralMapViewProps> = ({
  deltaHV,
  journals = {},
  onSelectBeat,
}) => {
  const [viewMode, setViewMode] = useState<ViewMode>('dashboard');
  const [selectedField, setSelectedField] = useState<GlyphicResonanceField | null>(null);
  const [selectedRegion, setSelectedRegion] = useState<NeuralActivation | null>(null);
  const [expandedJournal, setExpandedJournal] = useState<string | null>(null);
  const [playlists, setPlaylists] = useState<Playlist[]>([]);
  const [habits, setHabits] = useState<HabitTrack[]>([]);

  // Load playlists and habits
  useEffect(() => {
    setPlaylists(storyShuffleEngine.getPlaylists());
    const loadProfile = async () => {
      const profile = await userProfileService.initialize();
      setHabits(profile.habits || []);
    };
    loadProfile();

    const unsub = storyShuffleEngine.subscribe(() => {
      setPlaylists(storyShuffleEngine.getPlaylists());
    });
    return () => unsub();
  }, []);

  // Generate neural map from deltaHV metrics
  const neuralMap = useMemo(() => generateNeuralMap(deltaHV), [deltaHV]);

  // Get regions needing attention
  const attentionRegions = useMemo(() => getRegionsNeedingAttention(neuralMap, 50), [neuralMap]);

  // Get recent journals
  const recentJournals = useMemo(() => {
    const entries: Array<{ date: string; entry: JournalEntry }> = [];
    Object.entries(journals).forEach(([date, dayEntries]) => {
      dayEntries.forEach(entry => {
        entries.push({ date, entry });
      });
    });
    return entries.sort((a, b) =>
      new Date(b.entry.timestamp).getTime() - new Date(a.entry.timestamp).getTime()
    ).slice(0, 10);
  }, [journals]);

  // Get story metrics
  const storyMetrics = useMemo(() => storyShuffleEngine.getStoryMetrics(), []);

  // Get phase color
  const getPhaseColor = (phase: string): string => {
    const colors: Record<string, string> = {
      ignition: '#f97316',
      empowerment: '#eab308',
      resonance: '#22c55e',
      mania: '#ef4444',
      nirvana: '#a855f7',
      transmission: '#3b82f6',
      reflection: '#6366f1',
      collapse: '#64748b',
      rewrite: '#14b8a6',
    };
    return colors[phase] || '#6b7280';
  };

  // Get activation color gradient
  const getActivationColor = (activation: number, coherence: number): string => {
    if (coherence > 70) {
      if (activation > 70) return 'from-green-500 to-emerald-400';
      if (activation > 40) return 'from-green-600 to-green-500';
      return 'from-green-700 to-green-600';
    } else if (coherence > 40) {
      if (activation > 70) return 'from-yellow-500 to-amber-400';
      if (activation > 40) return 'from-yellow-600 to-yellow-500';
      return 'from-yellow-700 to-yellow-600';
    } else {
      if (activation > 70) return 'from-red-500 to-orange-400';
      if (activation > 40) return 'from-red-600 to-red-500';
      return 'from-red-700 to-red-600';
    }
  };

  // Format date for display
  const formatDate = (dateStr: string): string => {
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  };

  return (
    <div className="space-y-6">
      {/* Simplified Header with Key Metrics */}
      <div className="bg-gradient-to-br from-purple-900/30 to-cyan-900/30 backdrop-blur border border-purple-500/20 rounded-2xl p-6">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-purple-600 to-cyan-600 flex items-center justify-center">
              <Brain className="w-8 h-8 text-white" />
            </div>
            <div>
              <h3 className="text-xl font-medium text-white">Neural Dashboard</h3>
              <div className="flex items-center gap-2 mt-1">
                <span
                  className="px-2 py-0.5 rounded-full text-xs"
                  style={{ backgroundColor: `${getPhaseColor(neuralMap.dominantPhase)}20`, color: getPhaseColor(neuralMap.dominantPhase) }}
                >
                  {neuralMap.dominantPhase}
                </span>
                <span className={`px-2 py-0.5 rounded-full text-xs ${
                  neuralMap.fieldState === 'coherent' ? 'bg-green-500/20 text-green-300' :
                  neuralMap.fieldState === 'transitioning' ? 'bg-yellow-500/20 text-yellow-300' :
                  'bg-orange-500/20 text-orange-300'
                }`}>
                  {neuralMap.fieldState}
                </span>
              </div>
            </div>
          </div>
          <div className="text-right">
            <div className="text-4xl font-light text-white">{deltaHV.deltaHV}%</div>
            <div className="text-sm text-gray-400">Overall Coherence</div>
          </div>
        </div>

        {/* Quick Metrics Bar */}
        <div className="grid grid-cols-4 gap-3">
          <div className="bg-black/20 rounded-xl p-3 text-center">
            <Sparkles className="w-5 h-5 mx-auto mb-1 text-purple-400" />
            <div className="text-xl font-light text-white">{deltaHV.symbolicDensity}</div>
            <div className="text-xs text-gray-400">Symbolic</div>
          </div>
          <div className="bg-black/20 rounded-xl p-3 text-center">
            <Target className="w-5 h-5 mx-auto mb-1 text-cyan-400" />
            <div className="text-xl font-light text-white">{deltaHV.resonanceCoupling}</div>
            <div className="text-xs text-gray-400">Resonance</div>
          </div>
          <div className="bg-black/20 rounded-xl p-3 text-center">
            <AlertTriangle className="w-5 h-5 mx-auto mb-1 text-amber-400" />
            <div className="text-xl font-light text-white">{deltaHV.frictionCoefficient}</div>
            <div className="text-xs text-gray-400">Friction</div>
          </div>
          <div className="bg-black/20 rounded-xl p-3 text-center">
            <Activity className="w-5 h-5 mx-auto mb-1 text-green-400" />
            <div className="text-xl font-light text-white">{deltaHV.harmonicStability}</div>
            <div className="text-xs text-gray-400">Stability</div>
          </div>
        </div>
      </div>

      {/* Simple Tab Navigation */}
      <div className="flex gap-2 overflow-x-auto pb-2">
        {([
          { id: 'dashboard', label: 'Overview', icon: Zap },
          { id: 'map', label: 'Neural Map', icon: Brain },
          { id: 'journals', label: 'Journals & Music', icon: BookOpen },
          { id: 'metrics', label: 'Deep Metrics', icon: TrendingUp },
        ] as const).map(tab => (
          <button
            key={tab.id}
            onClick={() => setViewMode(tab.id)}
            className={`px-4 py-2.5 rounded-xl text-sm whitespace-nowrap transition-all flex items-center gap-2 ${
              viewMode === tab.id
                ? 'bg-purple-500/20 text-purple-300 border border-purple-500/30'
                : 'bg-gray-800/50 text-gray-400 border border-gray-700 hover:border-gray-600'
            }`}
          >
            <tab.icon className="w-4 h-4" />
            {tab.label}
          </button>
        ))}
      </div>

      {/* Dashboard View - Clean Overview */}
      {viewMode === 'dashboard' && (
        <div className="grid md:grid-cols-2 gap-6">
          {/* Today's Progress */}
          <div className="bg-gray-950/60 backdrop-blur border border-gray-800 rounded-xl p-5">
            <h4 className="text-white font-medium mb-4 flex items-center gap-2">
              <CheckSquare className="w-5 h-5 text-green-400" />
              Today's Progress
            </h4>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span className="text-gray-400">Tasks Aligned</span>
                  <span className="text-white">{deltaHV.breakdown.alignedTasks}/{deltaHV.breakdown.totalPlannedTasks || 0}</span>
                </div>
                <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-green-600 to-green-400 rounded-full"
                    style={{ width: `${deltaHV.breakdown.totalPlannedTasks ? (deltaHV.breakdown.alignedTasks / deltaHV.breakdown.totalPlannedTasks) * 100 : 0}%` }}
                  />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div className="bg-gray-900/50 rounded-lg p-3">
                  <div className="text-gray-400">Glyphs Used</div>
                  <div className="text-2xl font-light text-purple-300">{deltaHV.breakdown.glyphCount}</div>
                </div>
                <div className="bg-gray-900/50 rounded-lg p-3">
                  <div className="text-gray-400">Intentions Set</div>
                  <div className="text-2xl font-light text-cyan-300">{deltaHV.breakdown.intentionCount}</div>
                </div>
              </div>
              {deltaHV.breakdown.missedTasks > 0 && (
                <div className="flex items-center gap-2 text-sm text-amber-400 bg-amber-500/10 rounded-lg p-2">
                  <AlertTriangle className="w-4 h-4" />
                  <span>{deltaHV.breakdown.missedTasks} missed, {deltaHV.breakdown.delayedTasks} delayed</span>
                </div>
              )}
            </div>
          </div>

          {/* Active Habits */}
          <div className="bg-gray-950/60 backdrop-blur border border-gray-800 rounded-xl p-5">
            <h4 className="text-white font-medium mb-4 flex items-center gap-2">
              <ListTodo className="w-5 h-5 text-cyan-400" />
              Active Habits
            </h4>
            {habits.length > 0 ? (
              <div className="space-y-3">
                {habits.slice(0, 5).map(habit => (
                  <div key={habit.name} className="flex items-center gap-3 p-2 bg-gray-900/50 rounded-lg">
                    <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${
                      habit.currentStreak > 0 ? 'bg-green-500/20 text-green-400' : 'bg-gray-800 text-gray-500'
                    }`}>
                      {habit.currentStreak > 0 ? '🔥' : '○'}
                    </div>
                    <div className="flex-1">
                      <div className="text-sm text-white">{habit.name}</div>
                      <div className="text-xs text-gray-500">{habit.currentStreak} day streak</div>
                    </div>
                    <div className="text-xs text-gray-400">{habit.frequency}</div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-6 text-gray-500">
                <ListTodo className="w-8 h-8 mx-auto mb-2 opacity-50" />
                <p className="text-sm">No habits tracked yet</p>
              </div>
            )}
          </div>

          {/* Recent Playlists with Metrics */}
          <div className="bg-gray-950/60 backdrop-blur border border-gray-800 rounded-xl p-5">
            <h4 className="text-white font-medium mb-4 flex items-center gap-2">
              <Music className="w-5 h-5 text-pink-400" />
              Music & Emotional Journey
            </h4>
            {storyMetrics && storyMetrics.totalPlays > 0 ? (
              <div className="space-y-4">
                <div className="grid grid-cols-3 gap-3 text-center">
                  <div className="bg-gray-900/50 rounded-lg p-3">
                    <div className="text-xl font-light text-white">{storyMetrics.totalPlays}</div>
                    <div className="text-xs text-gray-400">Plays</div>
                  </div>
                  <div className="bg-gray-900/50 rounded-lg p-3">
                    <div className="text-xl font-light text-purple-300">{storyMetrics.authorshipScore}%</div>
                    <div className="text-xs text-gray-400">Authorship</div>
                  </div>
                  <div className="bg-gray-900/50 rounded-lg p-3">
                    <div className="text-xl font-light text-pink-300">{storyMetrics.melancholyProgress.healingIndicator}%</div>
                    <div className="text-xs text-gray-400">Healing</div>
                  </div>
                </div>
                <div className="text-sm text-gray-400">
                  <span className="text-white">Emotional trajectory:</span> {storyMetrics.emotionalTrajectory}
                </div>
              </div>
            ) : (
              <div className="text-center py-4 text-gray-500">
                <Music className="w-6 h-6 mx-auto mb-2 opacity-50" />
                <p className="text-sm">Play music to track your emotional journey</p>
              </div>
            )}

            {/* Quick Playlist Access */}
            <div className="mt-4 pt-4 border-t border-gray-800">
              <div className="text-xs text-gray-500 mb-2">Quick Play:</div>
              <div className="flex gap-2 overflow-x-auto pb-1">
                {playlists.slice(0, 5).map(playlist => (
                  <button
                    key={playlist.id}
                    className="px-3 py-1.5 bg-gray-800 hover:bg-gray-700 rounded-lg text-xs text-gray-300 whitespace-nowrap flex items-center gap-1 transition-colors"
                  >
                    <span>{playlist.coverEmoji}</span>
                    <span>{playlist.name}</span>
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Attention Areas */}
          <div className="bg-gray-950/60 backdrop-blur border border-gray-800 rounded-xl p-5">
            <h4 className="text-white font-medium mb-4 flex items-center gap-2">
              <AlertTriangle className="w-5 h-5 text-amber-400" />
              Areas Needing Attention
            </h4>
            {attentionRegions.length > 0 ? (
              <div className="space-y-2">
                {attentionRegions.slice(0, 4).map(region => (
                  <button
                    key={region.regionId}
                    onClick={() => {
                      setViewMode('map');
                      const activation = neuralMap.activations.find(a => a.regionId === region.regionId);
                      if (activation) setSelectedRegion(activation);
                    }}
                    className="w-full flex items-center gap-3 p-3 bg-amber-500/5 hover:bg-amber-500/10 border border-amber-500/20 rounded-lg transition-colors text-left"
                  >
                    <span className="text-2xl">{region.glyph}</span>
                    <div className="flex-1">
                      <div className="text-sm text-white">{region.regionName}</div>
                      <div className="text-xs text-amber-400">Free Energy: {region.freeEnergy}%</div>
                    </div>
                    <ChevronRight className="w-4 h-4 text-gray-500" />
                  </button>
                ))}
              </div>
            ) : (
              <div className="text-center py-6">
                <span className="text-4xl mb-2 block">✨</span>
                <p className="text-green-400 text-sm">All systems balanced!</p>
                <p className="text-gray-500 text-xs mt-1">Your neural field is coherent</p>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Neural Map View - Interactive Brain Regions */}
      {viewMode === 'map' && (
        <div className="space-y-6">
          {/* Resonance Fields - Simplified Cards */}
          <div>
            <h4 className="text-sm font-medium text-gray-400 uppercase tracking-wider mb-3">Resonance Fields</h4>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              {neuralMap.resonanceFields.map(field => {
                const isSelected = selectedField?.id === field.id;
                return (
                  <button
                    key={field.id}
                    onClick={() => setSelectedField(isSelected ? null : field)}
                    className={`p-4 rounded-xl border text-left transition-all ${
                      isSelected
                        ? 'border-purple-500 bg-purple-500/10 scale-105'
                        : 'border-gray-800 bg-gray-950/60 hover:border-gray-700 hover:scale-102'
                    }`}
                  >
                    <div className="flex items-center gap-3 mb-3">
                      <span className="text-3xl">{field.glyph}</span>
                      <div>
                        <h4 className="text-white font-medium">{field.name}</h4>
                        <div className="text-xs text-gray-400">{field.activeRegions.length} regions</div>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="flex-1 h-2 bg-gray-800 rounded-full overflow-hidden">
                        <div
                          className="h-full rounded-full"
                          style={{
                            width: `${field.strength}%`,
                            background: `linear-gradient(to right, ${field.color}80, ${field.color})`,
                          }}
                        />
                      </div>
                      <span className="text-sm text-gray-400">{field.strength}%</span>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Selected Field Detail */}
          {selectedField && (
            <div className="bg-gray-950/60 backdrop-blur border border-purple-500/30 rounded-xl p-5 animate-in slide-in-from-top-2">
              <div className="flex items-center justify-between mb-4">
                <h4 className="text-white font-medium flex items-center gap-2">
                  <span className="text-2xl">{selectedField.glyph}</span>
                  {selectedField.name} - Active Regions
                </h4>
                <button onClick={() => setSelectedField(null)} className="text-gray-400 hover:text-white">
                  <X className="w-5 h-5" />
                </button>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {selectedField.activeRegions.slice(0, 8).map(regionId => {
                  const activation = neuralMap.activations.find(a => a.regionId === regionId);
                  if (!activation) return null;
                  return (
                    <button
                      key={regionId}
                      onClick={() => setSelectedRegion(activation)}
                      className="p-3 bg-gray-900/50 rounded-lg border border-gray-800 hover:border-purple-500/30 transition-colors text-left"
                    >
                      <div className="flex items-center gap-2 mb-2">
                        <span className="text-xl">{activation.glyph}</span>
                        <span className="text-sm text-gray-300 truncate">{activation.regionName.split(' ').slice(0, 2).join(' ')}</span>
                      </div>
                      <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full bg-gradient-to-r ${getActivationColor(activation.activation, activation.coherence)}`}
                          style={{ width: `${activation.activation}%` }}
                        />
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          {/* Interactive Region Grid */}
          <div>
            <h4 className="text-sm font-medium text-gray-400 uppercase tracking-wider mb-3">All Brain Regions</h4>
            <div className="grid grid-cols-5 md:grid-cols-8 lg:grid-cols-10 gap-2">
              {neuralMap.activations.slice(0, 40).map(region => (
                <button
                  key={region.regionId}
                  onClick={() => setSelectedRegion(selectedRegion?.regionId === region.regionId ? null : region)}
                  className={`aspect-square p-2 rounded-xl transition-all hover:scale-110 ${
                    selectedRegion?.regionId === region.regionId
                      ? 'ring-2 ring-purple-500 bg-purple-500/20 scale-110'
                      : 'bg-gray-900/50 hover:bg-gray-800/50'
                  }`}
                  title={region.regionName}
                >
                  <div className="text-2xl text-center">{region.glyph}</div>
                  <div className="h-1 mt-1 bg-gray-800 rounded-full overflow-hidden">
                    <div
                      className={`h-full rounded-full bg-gradient-to-r ${getActivationColor(region.activation, region.coherence)}`}
                      style={{ width: `${region.activation}%` }}
                    />
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Selected Region Detail */}
          {selectedRegion && (
            <div className="bg-gray-950/60 backdrop-blur border border-purple-500/30 rounded-xl p-5 animate-in slide-in-from-bottom-2">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  <span className="text-4xl">{selectedRegion.glyph}</span>
                  <div>
                    <h4 className="text-white font-medium">{selectedRegion.regionName}</h4>
                    <div className="flex items-center gap-2 mt-1">
                      <span className="text-xs px-2 py-0.5 rounded-full bg-gray-800 text-gray-400 capitalize">
                        {selectedRegion.category}
                      </span>
                      <span
                        className="text-xs px-2 py-0.5 rounded-full"
                        style={{ backgroundColor: `${getPhaseColor(selectedRegion.phase)}20`, color: getPhaseColor(selectedRegion.phase) }}
                      >
                        {selectedRegion.phase}
                      </span>
                    </div>
                  </div>
                </div>
                <button onClick={() => setSelectedRegion(null)} className="text-gray-400 hover:text-white">
                  <X className="w-5 h-5" />
                </button>
              </div>
              <div className="grid grid-cols-3 gap-4 mb-4">
                <div className="text-center p-3 bg-gray-900/50 rounded-lg">
                  <div className={`text-2xl font-light ${selectedRegion.activation > 60 ? 'text-green-400' : 'text-yellow-400'}`}>
                    {selectedRegion.activation}%
                  </div>
                  <div className="text-xs text-gray-400">Activation</div>
                </div>
                <div className="text-center p-3 bg-gray-900/50 rounded-lg">
                  <div className={`text-2xl font-light ${selectedRegion.coherence > 60 ? 'text-green-400' : 'text-yellow-400'}`}>
                    {selectedRegion.coherence}%
                  </div>
                  <div className="text-xs text-gray-400">Coherence</div>
                </div>
                <div className="text-center p-3 bg-gray-900/50 rounded-lg">
                  <div className={`text-2xl font-light ${selectedRegion.freeEnergy < 40 ? 'text-green-400' : 'text-red-400'}`}>
                    {selectedRegion.freeEnergy}%
                  </div>
                  <div className="text-xs text-gray-400">Free Energy</div>
                </div>
              </div>
              <div className="text-sm text-gray-400 space-y-1">
                <div>
                  <span className="text-gray-500">Primary Metric: </span>
                  <span className="text-white capitalize">{selectedRegion.dominantMetric}</span>
                </div>
                <div>
                  <span className="text-gray-500">Neurochemicals: </span>
                  <span className="text-purple-300">{selectedRegion.neurochemicals.join(', ')}</span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Journals & Music View - Linked Content */}
      {viewMode === 'journals' && (
        <div className="grid md:grid-cols-2 gap-6">
          {/* Recent Journals */}
          <div className="bg-gray-950/60 backdrop-blur border border-gray-800 rounded-xl p-5">
            <h4 className="text-white font-medium mb-4 flex items-center gap-2">
              <BookOpen className="w-5 h-5 text-purple-400" />
              Recent Journal Entries
            </h4>
            {recentJournals.length > 0 ? (
              <div className="space-y-3">
                {recentJournals.map(({ date, entry }) => (
                  <div
                    key={entry.id}
                    className="bg-gray-900/50 rounded-lg overflow-hidden"
                  >
                    <button
                      onClick={() => setExpandedJournal(expandedJournal === entry.id ? null : entry.id)}
                      className="w-full p-3 flex items-center gap-3 text-left hover:bg-gray-800/50 transition-colors"
                    >
                      <Calendar className="w-4 h-4 text-gray-500" />
                      <div className="flex-1">
                        <div className="text-sm text-white">{formatDate(date)}</div>
                        <div className="text-xs text-gray-500 truncate">{entry.content.slice(0, 50)}...</div>
                      </div>
                      {entry.waveId && (
                        <span className="text-xs text-purple-400/60">Wave linked</span>
                      )}
                      <ChevronDown className={`w-4 h-4 text-gray-500 transition-transform ${
                        expandedJournal === entry.id ? 'rotate-180' : ''
                      }`} />
                    </button>
                    {expandedJournal === entry.id && (
                      <div className="px-3 pb-3 pt-0 border-t border-gray-800">
                        <p className="text-sm text-gray-300 whitespace-pre-wrap mt-2">{entry.content}</p>
                        {entry.waveId && (
                          <div className="mt-2 flex items-center gap-2 text-xs text-gray-500">
                            <Calendar className="w-3 h-3" />
                            <span>Linked to wave session</span>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500">
                <BookOpen className="w-8 h-8 mx-auto mb-2 opacity-50" />
                <p className="text-sm">No journal entries yet</p>
                <p className="text-xs mt-1">Start journaling to track your thoughts</p>
              </div>
            )}
          </div>

          {/* Playlists with Neural Connection */}
          <div className="bg-gray-950/60 backdrop-blur border border-gray-800 rounded-xl p-5">
            <h4 className="text-white font-medium mb-4 flex items-center gap-2">
              <Music className="w-5 h-5 text-pink-400" />
              Your Playlists
            </h4>
            {playlists.length > 0 ? (
              <div className="space-y-2">
                {playlists.map(playlist => (
                  <div
                    key={playlist.id}
                    className="flex items-center gap-3 p-3 bg-gray-900/50 rounded-lg hover:bg-gray-800/50 transition-colors"
                  >
                    <span className="text-2xl">{playlist.coverEmoji}</span>
                    <div className="flex-1">
                      <div className="text-sm text-white">{playlist.name}</div>
                      <div className="text-xs text-gray-500">{playlist.trackIds.length} tracks</div>
                    </div>
                    {playlist.basedOnMetric && (
                      <span className={`px-2 py-0.5 rounded-full text-xs ${
                        playlist.basedOnMetric === 'symbolic' ? 'bg-purple-500/20 text-purple-300' :
                        playlist.basedOnMetric === 'resonance' ? 'bg-cyan-500/20 text-cyan-300' :
                        playlist.basedOnMetric === 'friction' ? 'bg-amber-500/20 text-amber-300' :
                        'bg-green-500/20 text-green-300'
                      }`}>
                        {playlist.basedOnMetric}
                      </span>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-8 text-gray-500">
                <Music className="w-8 h-8 mx-auto mb-2 opacity-50" />
                <p className="text-sm">No playlists yet</p>
                <p className="text-xs mt-1">Create playlists in the DJ tab</p>
              </div>
            )}

            {/* Emotional Story */}
            {storyMetrics && storyMetrics.totalPlays > 0 && (
              <div className="mt-4 pt-4 border-t border-gray-800">
                <h5 className="text-sm text-gray-400 mb-3">Your Emotional Story</h5>
                <div className="bg-gradient-to-br from-purple-900/20 to-pink-900/20 rounded-lg p-4">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="text-2xl">
                      {storyMetrics.melancholyProgress.currentPhase === 'integrated' ? '🦋' :
                       storyMetrics.melancholyProgress.currentPhase === 'healing' ? '🌱' : '💭'}
                    </span>
                    <span className="text-white capitalize">{storyMetrics.melancholyProgress.currentPhase}</span>
                  </div>
                  <p className="text-sm text-gray-400">
                    {storyMetrics.emotionalTrajectory === 'rising' && "You're lifting up. Your music choices reflect growth."}
                    {storyMetrics.emotionalTrajectory === 'processing' && "You're processing emotions. This is healthy."}
                    {storyMetrics.emotionalTrajectory === 'releasing' && "You're letting go. Trust the process."}
                    {storyMetrics.emotionalTrajectory === 'steady' && "You're in a stable place. Keep exploring."}
                  </p>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Deep Metrics View */}
      {viewMode === 'metrics' && (
        <div className="space-y-6">
          {/* Metric Influence Cards */}
          {Object.entries(neuralMap.metricInfluence).map(([metric, data]) => {
            const metricConfig: Record<string, { icon: React.ReactNode; color: string; gradient: string }> = {
              symbolic: { icon: <Sparkles className="w-5 h-5" />, color: 'purple', gradient: 'from-purple-600 to-purple-400' },
              resonance: { icon: <Target className="w-5 h-5" />, color: 'cyan', gradient: 'from-cyan-600 to-cyan-400' },
              friction: { icon: <AlertTriangle className="w-5 h-5" />, color: 'amber', gradient: 'from-amber-600 to-amber-400' },
              stability: { icon: <Activity className="w-5 h-5" />, color: 'green', gradient: 'from-green-600 to-green-400' },
            };
            const config = metricConfig[metric];
            const regions = neuralMap.activations.filter(a => data.regions.includes(a.regionId));

            return (
              <div key={metric} className="bg-gray-950/60 backdrop-blur border border-gray-800 rounded-xl p-5">
                <div className="flex items-center justify-between mb-4">
                  <h4 className={`text-white font-medium capitalize flex items-center gap-2 text-${config.color}-400`}>
                    {config.icon}
                    {metric}
                  </h4>
                  <div className={`text-2xl font-light text-${config.color}-400`}>{data.strength}%</div>
                </div>

                <div className="h-3 bg-gray-800 rounded-full overflow-hidden mb-4">
                  <div
                    className={`h-full rounded-full bg-gradient-to-r ${config.gradient}`}
                    style={{ width: `${data.strength}%` }}
                  />
                </div>

                <div className="flex flex-wrap gap-2">
                  {regions.slice(0, 6).map(region => (
                    <button
                      key={region.regionId}
                      onClick={() => {
                        setViewMode('map');
                        setSelectedRegion(region);
                      }}
                      className={`px-3 py-1.5 bg-${config.color}-500/10 text-${config.color}-300 rounded-lg text-sm flex items-center gap-1.5 hover:bg-${config.color}-500/20 transition-colors`}
                    >
                      <span>{region.glyph}</span>
                      <span className="text-xs">{region.regionName.split(' ')[0]}</span>
                    </button>
                  ))}
                </div>
              </div>
            );
          })}

          {/* Recommendations */}
          <div className="bg-gray-950/60 backdrop-blur border border-gray-800 rounded-xl p-5">
            <h4 className="text-white font-medium mb-4 flex items-center gap-2">
              💡 AI Recommendations
            </h4>
            {neuralMap.recommendations.length > 0 ? (
              <div className="space-y-3">
                {neuralMap.recommendations.slice(0, 3).map(rec => (
                  <div
                    key={rec.id}
                    className={`p-4 rounded-lg border ${
                      rec.priority === 'high' ? 'bg-red-500/5 border-red-500/30' :
                      rec.priority === 'medium' ? 'bg-yellow-500/5 border-yellow-500/30' :
                      'bg-gray-900/50 border-gray-800'
                    }`}
                  >
                    <div className="flex items-start gap-3">
                      <span className="text-2xl">{rec.glyph}</span>
                      <div className="flex-1">
                        <h5 className="text-white font-medium">{rec.title}</h5>
                        <p className="text-sm text-gray-400 mt-1">{rec.description}</p>
                        <button
                          onClick={() => onSelectBeat?.(rec.suggestedBeat)}
                          className="mt-3 px-4 py-2 bg-purple-500/20 text-purple-300 rounded-lg hover:bg-purple-500/30 transition-colors text-sm flex items-center gap-2"
                        >
                          Start {rec.suggestedBeat} Beat
                          <ChevronRight className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-6">
                <span className="text-4xl mb-2 block">🌟</span>
                <p className="text-green-400">All systems coherent!</p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default NeuralMapView;
