/**
 * DJ Tab Component
 *
 * Streamlined music mixing interface with:
 * - Current track display with transport controls
 * - Mood presets for quick filtering
 * - Compact playlist access
 * - Clean track list
 */

import React, { useState, useEffect, useMemo, useCallback } from 'react';
import {
  X,
  Play,
  Pause,
  SkipForward,
  SkipBack,
  Music,
  Disc3,
  Shuffle,
  Wand2,
  Heart,
  Zap,
  Sun,
  Moon,
  Target,
  Plus,
  ChevronRight,
} from 'lucide-react';
import type { DeltaHVState } from '../lib/deltaHVEngine';
import type { MusicTrack, EmotionalCategoryId } from '../lib/musicLibrary';
import { musicLibrary, EMOTIONAL_CATEGORIES } from '../lib/musicLibrary';
import { storyShuffleEngine, type Playlist } from '../lib/storyShuffleEngine';

interface DJTabProps {
  deltaHV: DeltaHVState | null;
  onClose: () => void;
}

interface MoodPreset {
  id: string;
  name: string;
  icon: React.ReactNode;
  categories: EmotionalCategoryId[];
}

const MOOD_PRESETS: MoodPreset[] = [
  { id: 'energize', name: 'Energy', icon: <Zap className="w-4 h-4" />, categories: ['ENERGY', 'JOY', 'COURAGE'] },
  { id: 'focus', name: 'Focus', icon: <Target className="w-4 h-4" />, categories: ['FOCUS', 'CALM'] },
  { id: 'chill', name: 'Chill', icon: <Moon className="w-4 h-4" />, categories: ['CALM', 'RELEASE', 'GRATITUDE'] },
  { id: 'uplift', name: 'Uplift', icon: <Sun className="w-4 h-4" />, categories: ['JOY', 'LOVE', 'GRATITUDE', 'WONDER'] },
  { id: 'process', name: 'Feel', icon: <Heart className="w-4 h-4" />, categories: ['MELANCHOLY', 'RELEASE'] },
];

export const DJTab: React.FC<DJTabProps> = ({ deltaHV, onClose }) => {
  const [tracks, setTracks] = useState<MusicTrack[]>([]);
  const [playlists, setPlaylists] = useState<Playlist[]>([]);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTrack, setCurrentTrack] = useState<MusicTrack | null>(null);
  const [selectedMood, setSelectedMood] = useState<string | null>(null);
  const [queue, setQueue] = useState<string[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [expandedPlaylist, setExpandedPlaylist] = useState<string | null>(null);

  // Load tracks and playlists
  useEffect(() => {
    const loadData = async () => {
      await musicLibrary.initialize();
      const allTracks = await musicLibrary.getAllTracks();
      setTracks(allTracks);
      setPlaylists(storyShuffleEngine.getPlaylists());
    };
    loadData();

    const unsub1 = musicLibrary.subscribe(async () => {
      const allTracks = await musicLibrary.getAllTracks();
      setTracks(allTracks);
    });

    const unsub2 = storyShuffleEngine.subscribe(() => {
      setPlaylists(storyShuffleEngine.getPlaylists());
    });

    const unsubPlayback = musicLibrary.subscribeToPlayback((state) => {
      setIsPlaying(state.isPlaying);
      if (state.trackId) {
        musicLibrary.getTrack(state.trackId).then(track => {
          if (track) setCurrentTrack(track);
        });
      }
    });

    return () => {
      unsub1();
      unsub2();
      unsubPlayback();
    };
  }, []);

  // Stop playback when closing
  const handleClose = useCallback(() => {
    if (musicLibrary.isCurrentlyPlaying()) {
      musicLibrary.pausePlayback();
    }
    onClose();
  }, [onClose]);

  // Filter tracks by mood
  const moodTracks = useMemo(() => {
    if (!selectedMood) return tracks;
    const preset = MOOD_PRESETS.find(p => p.id === selectedMood);
    if (!preset) return tracks;
    return tracks.filter(t => preset.categories.includes(t.categoryId));
  }, [tracks, selectedMood]);

  // Play a track
  const playTrack = useCallback(async (track: MusicTrack) => {
    await musicLibrary.playTrack(track.id, track.categoryId);
    setCurrentTrack(track);
  }, []);

  // Play/pause toggle
  const togglePlay = useCallback(async () => {
    if (musicLibrary.isCurrentlyPlaying()) {
      musicLibrary.pausePlayback();
    } else {
      await musicLibrary.resumePlayback();
    }
  }, []);

  // Skip to next track in queue
  const skipNext = useCallback(async () => {
    if (queue.length > 0 && currentIndex < queue.length - 1) {
      const nextIndex = currentIndex + 1;
      const nextTrack = tracks.find(t => t.id === queue[nextIndex]);
      if (nextTrack) {
        await playTrack(nextTrack);
        setCurrentIndex(nextIndex);
      }
    }
  }, [queue, currentIndex, tracks, playTrack]);

  // Skip to previous track in queue
  const skipPrev = useCallback(async () => {
    if (queue.length > 0 && currentIndex > 0) {
      const prevIndex = currentIndex - 1;
      const prevTrack = tracks.find(t => t.id === queue[prevIndex]);
      if (prevTrack) {
        await playTrack(prevTrack);
        setCurrentIndex(prevIndex);
      }
    }
  }, [queue, currentIndex, tracks, playTrack]);

  // Shuffle and play mood tracks
  const shuffleMood = useCallback(async () => {
    if (!deltaHV || moodTracks.length === 0) return;
    const shuffled = await storyShuffleEngine.generateHilbertShuffle(moodTracks, deltaHV);
    setQueue(shuffled);
    setCurrentIndex(0);
    if (shuffled.length > 0) {
      const firstTrack = tracks.find(t => t.id === shuffled[0]);
      if (firstTrack) await playTrack(firstTrack);
    }
  }, [deltaHV, moodTracks, tracks, playTrack]);

  // Generate AI playlist
  const generateAIPlaylist = useCallback(async () => {
    if (!deltaHV) return;
    const purposeMap: Record<string, 'balance' | 'boost' | 'calm' | 'focus' | 'heal' | 'energize'> = {
      energize: 'energize',
      focus: 'focus',
      chill: 'calm',
      uplift: 'boost',
      process: 'heal',
    };
    const purpose = selectedMood ? purposeMap[selectedMood] || 'balance' : 'balance';
    const playlist = await storyShuffleEngine.generateCoherencePlaylist(deltaHV, purpose);
    setQueue(playlist.trackIds);
    setCurrentIndex(0);
    if (playlist.trackIds.length > 0) {
      const firstTrack = tracks.find(t => t.id === playlist.trackIds[0]);
      if (firstTrack) await playTrack(firstTrack);
    }
  }, [deltaHV, selectedMood, tracks, playTrack]);

  // Play a playlist
  const playPlaylist = useCallback(async (playlist: Playlist) => {
    if (playlist.trackIds.length > 0) {
      setQueue(playlist.trackIds);
      setCurrentIndex(0);
      const firstTrack = tracks.find(t => t.id === playlist.trackIds[0]);
      if (firstTrack) await playTrack(firstTrack);
    }
  }, [tracks, playTrack]);

  // Add current track to playlist
  const addToPlaylist = useCallback((playlistId: string) => {
    if (currentTrack) {
      storyShuffleEngine.addToPlaylist(playlistId, currentTrack.id);
    }
  }, [currentTrack]);

  const formatDuration = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Get user playlists (non-system)
  const userPlaylists = useMemo(() => playlists.filter(p => !p.isSystem), [playlists]);

  return (
    <div className="fixed inset-0 bg-black/95 backdrop-blur-sm z-50 flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-800">
        <button onClick={handleClose} className="p-2 rounded-lg bg-gray-900/70 border border-gray-800 hover:bg-gray-800">
          <X className="w-6 h-6" />
        </button>
        <div className="flex items-center gap-3">
          <Disc3 className={`w-7 h-7 text-purple-400 ${isPlaying ? 'animate-spin' : ''}`} />
          <span className="text-lg font-medium text-white">DJ Mode</span>
        </div>
        <div className="w-10" />
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-y-auto">
        {/* Now Playing */}
        <div className="p-4 bg-gradient-to-b from-purple-900/20 to-transparent">
          <div className="flex items-center gap-4">
            <div className="w-16 h-16 bg-gradient-to-br from-purple-600 to-cyan-600 rounded-xl flex items-center justify-center flex-shrink-0">
              {currentTrack ? (
                <span className="text-3xl">{EMOTIONAL_CATEGORIES[currentTrack.categoryId]?.icon}</span>
              ) : (
                <Music className="w-8 h-8 text-white/50" />
              )}
            </div>
            <div className="flex-1 min-w-0">
              <h3 className="text-lg font-medium text-white truncate">
                {currentTrack?.name || 'No track playing'}
              </h3>
              <p className="text-sm text-gray-400 truncate">
                {currentTrack ? EMOTIONAL_CATEGORIES[currentTrack.categoryId]?.name : 'Select a track to start'}
              </p>
            </div>
          </div>

          {/* Transport */}
          <div className="flex items-center justify-center gap-3 mt-4">
            <button
              onClick={skipPrev}
              disabled={currentIndex === 0}
              className="p-2 rounded-full bg-gray-800/50 hover:bg-gray-700/50 disabled:opacity-30 transition-colors"
            >
              <SkipBack className="w-5 h-5" />
            </button>
            <button
              onClick={togglePlay}
              className="p-4 rounded-full bg-gradient-to-r from-purple-600 to-cyan-600 hover:from-purple-500 hover:to-cyan-500 transition-colors"
            >
              {isPlaying ? <Pause className="w-6 h-6" /> : <Play className="w-6 h-6" />}
            </button>
            <button
              onClick={skipNext}
              disabled={currentIndex >= queue.length - 1}
              className="p-2 rounded-full bg-gray-800/50 hover:bg-gray-700/50 disabled:opacity-30 transition-colors"
            >
              <SkipForward className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Mood Presets */}
        <div className="px-4 py-3">
          <div className="flex gap-2 overflow-x-auto pb-1">
            {MOOD_PRESETS.map(preset => (
              <button
                key={preset.id}
                onClick={() => setSelectedMood(selectedMood === preset.id ? null : preset.id)}
                className={`flex items-center gap-2 px-3 py-2 rounded-lg whitespace-nowrap transition-all ${
                  selectedMood === preset.id
                    ? 'bg-purple-600/30 border border-purple-400 text-purple-300'
                    : 'bg-gray-900/50 border border-gray-800 text-gray-400 hover:border-gray-700'
                }`}
              >
                {preset.icon}
                <span className="text-sm">{preset.name}</span>
              </button>
            ))}
          </div>
        </div>

        {/* Quick Actions */}
        {selectedMood && (
          <div className="px-4 pb-3 flex gap-2">
            <button
              onClick={shuffleMood}
              className="flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded-lg bg-purple-600/20 hover:bg-purple-600/30 text-purple-300 text-sm transition-colors"
            >
              <Shuffle className="w-4 h-4" />
              Shuffle
            </button>
            <button
              onClick={generateAIPlaylist}
              className="flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded-lg bg-cyan-600/20 hover:bg-cyan-600/30 text-cyan-300 text-sm transition-colors"
            >
              <Wand2 className="w-4 h-4" />
              AI Mix
            </button>
          </div>
        )}

        {/* Playlists - Compact */}
        {userPlaylists.length > 0 && (
          <div className="px-4 py-3 border-t border-gray-800/50">
            <p className="text-xs text-gray-500 uppercase tracking-wider mb-2">Your Playlists</p>
            <div className="space-y-1">
              {userPlaylists.slice(0, 4).map(playlist => (
                <div key={playlist.id} className="rounded-lg bg-gray-900/30 overflow-hidden">
                  <button
                    onClick={() => setExpandedPlaylist(expandedPlaylist === playlist.id ? null : playlist.id)}
                    className="w-full flex items-center gap-3 p-2 hover:bg-gray-800/30 transition-colors"
                  >
                    <span className="text-xl">{playlist.coverEmoji}</span>
                    <div className="flex-1 text-left min-w-0">
                      <p className="text-sm text-white truncate">{playlist.name}</p>
                      <p className="text-xs text-gray-500">{playlist.trackIds.length} tracks</p>
                    </div>
                    <ChevronRight className={`w-4 h-4 text-gray-500 transition-transform ${expandedPlaylist === playlist.id ? 'rotate-90' : ''}`} />
                  </button>

                  {expandedPlaylist === playlist.id && (
                    <div className="px-2 pb-2 space-y-1">
                      <button
                        onClick={() => playPlaylist(playlist)}
                        className="w-full flex items-center gap-2 p-2 rounded bg-purple-600/20 hover:bg-purple-600/30 text-purple-300 text-sm"
                      >
                        <Play className="w-4 h-4" />
                        Play All
                      </button>
                      {playlist.trackIds.slice(0, 3).map(trackId => {
                        const track = tracks.find(t => t.id === trackId);
                        if (!track) return null;
                        return (
                          <button
                            key={trackId}
                            onClick={() => playTrack(track)}
                            className="w-full flex items-center gap-2 p-1.5 rounded hover:bg-gray-800/50 text-left"
                          >
                            <span className="text-sm">{EMOTIONAL_CATEGORIES[track.categoryId]?.icon}</span>
                            <span className="text-xs text-gray-400 truncate flex-1">{track.name}</span>
                          </button>
                        );
                      })}
                      {playlist.trackIds.length > 3 && (
                        <p className="text-xs text-gray-600 text-center">+{playlist.trackIds.length - 3} more</p>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Track List */}
        <div className="px-4 py-3 border-t border-gray-800/50">
          <p className="text-xs text-gray-500 uppercase tracking-wider mb-2">
            {selectedMood ? `${MOOD_PRESETS.find(p => p.id === selectedMood)?.name} Tracks` : 'All Tracks'} ({moodTracks.length})
          </p>
          <div className="space-y-1 max-h-72 overflow-y-auto">
            {moodTracks.slice(0, 15).map(track => (
              <button
                key={track.id}
                onClick={() => playTrack(track)}
                className={`w-full flex items-center gap-3 p-2 rounded-lg transition-colors text-left ${
                  currentTrack?.id === track.id
                    ? 'bg-purple-900/30 border border-purple-500/30'
                    : 'hover:bg-gray-800/50'
                }`}
              >
                <span className="text-lg">{EMOTIONAL_CATEGORIES[track.categoryId]?.icon}</span>
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-white truncate">{track.name}</p>
                  <p className="text-xs text-gray-500">{formatDuration(track.duration)}</p>
                </div>
                {currentTrack?.id === track.id && isPlaying && (
                  <div className="flex gap-0.5">
                    <div className="w-1 h-3 bg-purple-400 rounded-full animate-pulse" />
                    <div className="w-1 h-3 bg-purple-400 rounded-full animate-pulse delay-75" />
                    <div className="w-1 h-3 bg-purple-400 rounded-full animate-pulse delay-150" />
                  </div>
                )}
              </button>
            ))}
          </div>
        </div>

        {/* Add to Playlist (when track playing) */}
        {currentTrack && userPlaylists.length > 0 && (
          <div className="px-4 py-3 border-t border-gray-800/50">
            <p className="text-xs text-gray-500 uppercase tracking-wider mb-2">Add to Playlist</p>
            <div className="flex gap-2 overflow-x-auto pb-1">
              {userPlaylists.map(playlist => (
                <button
                  key={playlist.id}
                  onClick={() => addToPlaylist(playlist.id)}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-gray-900/50 border border-gray-800 hover:border-purple-500/50 text-sm whitespace-nowrap"
                >
                  <Plus className="w-3 h-3 text-gray-500" />
                  <span className="text-gray-400">{playlist.coverEmoji} {playlist.name}</span>
                </button>
              ))}
            </div>
          </div>
        )}

        {tracks.length === 0 && (
          <div className="text-center py-12 text-gray-500">
            <Music className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p>No tracks in library</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default DJTab;
