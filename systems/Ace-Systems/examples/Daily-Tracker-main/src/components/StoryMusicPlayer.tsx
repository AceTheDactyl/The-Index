/**
 * Story Music Player Component
 *
 * An empowering music player that encourages users to skip freely.
 * "You are the author of your story. Every skip is a choice."
 *
 * Features:
 * - Hilbert curve shuffle visualization
 * - Skip tracking with authorship metrics
 * - Melancholy healing indicators
 * - Custom playlist management
 * - Metric-responsive song selection
 */

import React, { useState, useEffect, useMemo, useCallback } from 'react';
import {
  Play,
  Pause,
  SkipForward,
  SkipBack,
  Shuffle,
  Repeat,
  Volume2,
  VolumeX,
  ListMusic,
  Plus,
  X,
  Sparkles,
  TrendingUp,
  ChevronRight,
  Check,
  Music,
  BookOpen,
  Wand2,
  Edit3,
  GripVertical,
  Trash2,
  Loader2,
} from 'lucide-react';
import type { DeltaHVState } from '../lib/deltaHVEngine';
import type { MusicTrack, EmotionalCategoryId } from '../lib/musicLibrary';
import { musicLibrary, EMOTIONAL_CATEGORIES } from '../lib/musicLibrary';
import {
  storyShuffleEngine,
  type StoryMetrics,
  type Playlist,
} from '../lib/storyShuffleEngine';

interface StoryMusicPlayerProps {
  deltaHV: DeltaHVState | null;
  onMetricsUpdate?: (metrics: StoryMetrics) => void;
}

export const StoryMusicPlayer: React.FC<StoryMusicPlayerProps> = ({
  deltaHV,
  onMetricsUpdate,
}) => {
  // Player state
  const [tracks, setTracks] = useState<MusicTrack[]>([]);
  const [currentTrack, setCurrentTrack] = useState<MusicTrack | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [progress, setProgress] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(0.7);
  const [isMuted, setIsMuted] = useState(false);
  const [isShuffled, setIsShuffled] = useState(true);
  const [repeatMode, setRepeatMode] = useState<'off' | 'all' | 'one'>('off');

  // UI state
  const [showPlaylists, setShowPlaylists] = useState(false);
  const [showStoryMetrics, setShowStoryMetrics] = useState(false);
  const [showCreatePlaylist, setShowCreatePlaylist] = useState(false);
  const [newPlaylistName, setNewPlaylistName] = useState('');
  const [showAIRecommendations, setShowAIRecommendations] = useState(false);
  const [generatingPlaylist, setGeneratingPlaylist] = useState(false);
  const [editingPlaylistId, setEditingPlaylistId] = useState<string | null>(null);

  // Story metrics
  const [storyMetrics, setStoryMetrics] = useState<StoryMetrics | null>(null);
  const [playlists, setPlaylists] = useState<Playlist[]>([]);
  const [shuffleQueue, setShuffleQueue] = useState<string[]>([]);

  // Authorship message
  const [authorshipMessage, setAuthorshipMessage] = useState<{
    message: string;
    emoji: string;
    type: 'encouragement' | 'insight' | 'celebration';
  }>({ message: '', emoji: '', type: 'encouragement' });

  // Play start time for skip tracking
  const [playStartTime, setPlayStartTime] = useState<number>(0);

  // Track if we should auto-play next (to handle track ending)
  const [wasPlayingBeforeEnd, setWasPlayingBeforeEnd] = useState(false);

  // Initialize
  useEffect(() => {
    const init = async () => {
      await musicLibrary.initialize();
      const allTracks = await musicLibrary.getAllTracks();
      setTracks(allTracks);
      setPlaylists(storyShuffleEngine.getPlaylists());
      setStoryMetrics(storyShuffleEngine.getStoryMetrics());
      setAuthorshipMessage(storyShuffleEngine.getAuthorshipMessage());

      // Generate initial shuffle
      if (allTracks.length > 0) {
        const queue = await storyShuffleEngine.generateHilbertShuffle(allTracks, deltaHV);
        setShuffleQueue(queue);
      }

      // Sync initial playback state from library
      const playbackState = musicLibrary.getPlaybackState();
      setIsPlaying(playbackState.isPlaying);
      if (playbackState.currentSession) {
        const currentTrackId = musicLibrary.getCurrentTrackId();
        if (currentTrackId) {
          const track = allTracks.find(t => t.id === currentTrackId);
          if (track) {
            setCurrentTrack(track);
            setPlayStartTime(new Date(playbackState.currentSession.startedAt).getTime());
          }
        }
      }
    };

    init();

    // Subscribe to track library updates
    const unsubLibrary = musicLibrary.subscribe(() => {
      musicLibrary.getAllTracks().then(setTracks);
    });

    // Subscribe to playback state changes - CRITICAL for play/pause sync
    const unsubPlayback = musicLibrary.subscribeToPlayback((state) => {
      const wasPlaying = isPlaying;
      setIsPlaying(state.isPlaying);

      // Detect track ended (was playing, now not playing)
      // This will trigger auto-play of next track if repeat mode allows
      if (wasPlaying && !state.isPlaying) {
        setWasPlayingBeforeEnd(true);
      }
    });

    const unsubShuffle = storyShuffleEngine.subscribe(() => {
      setStoryMetrics(storyShuffleEngine.getStoryMetrics());
      setPlaylists(storyShuffleEngine.getPlaylists());
      setAuthorshipMessage(storyShuffleEngine.getAuthorshipMessage());
    });

    return () => {
      unsubLibrary();
      unsubPlayback();
      unsubShuffle();
    };
  }, [deltaHV]);

  // Update parent with metrics
  useEffect(() => {
    if (storyMetrics && onMetricsUpdate) {
      onMetricsUpdate(storyMetrics);
    }
  }, [storyMetrics, onMetricsUpdate]);

  // Track playback progress
  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    if (isPlaying) {
      interval = setInterval(() => {
        const state = musicLibrary.getPlaybackState();
        setProgress(state.currentTime);
        setDuration(state.duration);
      }, 100);
    }
    return () => clearInterval(interval);
  }, [isPlaying]);

  // Re-shuffle when metrics change significantly
  useEffect(() => {
    if (deltaHV && tracks.length > 0 && isShuffled) {
      storyShuffleEngine.generateHilbertShuffle(tracks, deltaHV).then(setShuffleQueue);
    }
  }, [deltaHV?.fieldState, tracks.length, isShuffled]);

  // Play track - musicLibrary handles stopping any current playback
  // MUST be defined before effects that use it
  const playTrack = useCallback(async (track: MusicTrack) => {
    // Record skip for previous track if it was playing
    const wasPlaying = musicLibrary.isCurrentlyPlaying();
    if (currentTrack && wasPlaying) {
      const listenedTime = (Date.now() - playStartTime) / 1000;
      storyShuffleEngine.recordPlayEvent(
        currentTrack,
        listenedTime,
        true, // was skipped
        deltaHV,
        'manual'
      );
    }

    // musicLibrary.playTrack will stop any current playback first
    const session = await musicLibrary.playTrack(
      track.id,
      track.categoryId,
      undefined,
      undefined
    );

    if (session) {
      setCurrentTrack(track);
      setPlayStartTime(Date.now());
      // Note: isPlaying will be set by the playback subscription
    }
  }, [currentTrack, playStartTime, deltaHV]);

  // Auto-play next track when current track ends
  useEffect(() => {
    if (!wasPlayingBeforeEnd || !currentTrack) return;

    // Reset the flag
    setWasPlayingBeforeEnd(false);

    // Record that track completed (not skipped)
    const listenedTime = (Date.now() - playStartTime) / 1000;
    storyShuffleEngine.recordPlayEvent(
      currentTrack,
      listenedTime,
      false, // was NOT skipped - played through
      deltaHV,
      'auto_next'
    );

    // Handle repeat modes
    if (repeatMode === 'one') {
      // Repeat the same track
      playTrack(currentTrack);
      return;
    }

    // Get next track based on shuffle state
    const nextTrackId = storyShuffleEngine.getNextTrack();

    if (nextTrackId) {
      const nextTrack = tracks.find(t => t.id === nextTrackId);
      if (nextTrack) {
        playTrack(nextTrack);
      }
    } else if (repeatMode === 'all' && shuffleQueue.length > 0) {
      // Loop back to start of queue
      const firstTrack = tracks.find(t => t.id === shuffleQueue[0]);
      if (firstTrack) {
        playTrack(firstTrack);
      }
    }
    // If repeatMode === 'off' and no more tracks, just stop (do nothing)
  }, [wasPlayingBeforeEnd, currentTrack, repeatMode, tracks, shuffleQueue, deltaHV, playStartTime, playTrack]);

  // Skip forward (empowerment action)
  const handleSkip = useCallback(async () => {
    if (currentTrack) {
      const listenedTime = (Date.now() - playStartTime) / 1000;
      storyShuffleEngine.recordPlayEvent(
        currentTrack,
        listenedTime,
        true,
        deltaHV,
        'manual'
      );
    }

    // Get next track from shuffle queue
    let nextTrackId = storyShuffleEngine.getNextTrack();

    // If no next track from queue, try to generate a new shuffle or pick randomly
    if (!nextTrackId && tracks.length > 0) {
      // Generate a new shuffle queue if empty
      const newQueue = await storyShuffleEngine.generateHilbertShuffle(tracks, deltaHV);
      setShuffleQueue(newQueue);

      // Get the first track from the new queue (or second if current track is first)
      if (newQueue.length > 0) {
        const currentIndex = currentTrack ? newQueue.indexOf(currentTrack.id) : -1;
        if (currentIndex >= 0 && currentIndex < newQueue.length - 1) {
          nextTrackId = newQueue[currentIndex + 1];
        } else {
          nextTrackId = newQueue[0];
        }
      }
    }

    if (nextTrackId) {
      const nextTrack = tracks.find(t => t.id === nextTrackId);
      if (nextTrack) {
        await playTrack(nextTrack);
      }
    } else if (tracks.length > 0) {
      // Fallback: just pick a random track that's not the current one
      const availableTracks = currentTrack
        ? tracks.filter(t => t.id !== currentTrack.id)
        : tracks;
      if (availableTracks.length > 0) {
        const randomTrack = availableTracks[Math.floor(Math.random() * availableTracks.length)];
        await playTrack(randomTrack);
      }
    }
  }, [currentTrack, playStartTime, deltaHV, tracks, playTrack]);

  // Skip back
  const handlePrevious = useCallback(async () => {
    const prevTrackId = storyShuffleEngine.getPreviousTrack();
    if (prevTrackId) {
      const prevTrack = tracks.find(t => t.id === prevTrackId);
      if (prevTrack) {
        await playTrack(prevTrack);
      }
    }
  }, [tracks, playTrack]);

  // Toggle play/pause - async to properly handle resumePlayback
  const togglePlay = useCallback(async () => {
    // Check actual playback state from library to avoid sync issues
    const actuallyPlaying = musicLibrary.isCurrentlyPlaying();

    if (actuallyPlaying) {
      // Currently playing - pause it
      musicLibrary.pausePlayback();
      // Note: playback listener will update isPlaying state
    } else if (currentTrack && musicLibrary.getCurrentTrackId() === currentTrack.id) {
      // Have a track loaded that matches - resume it
      await musicLibrary.resumePlayback();
      // Note: playback listener will update isPlaying state
    } else if (currentTrack) {
      // Have a track but it's not the one loaded in library - play it fresh
      await playTrack(currentTrack);
    } else if (shuffleQueue.length > 0) {
      // No track selected - play first from shuffle queue
      const firstTrack = tracks.find(t => t.id === shuffleQueue[0]);
      if (firstTrack) {
        await playTrack(firstTrack);
      }
    } else if (tracks.length > 0) {
      // No shuffle queue - generate one and start playing
      const newQueue = await storyShuffleEngine.generateHilbertShuffle(tracks, deltaHV);
      setShuffleQueue(newQueue);
      if (newQueue.length > 0) {
        const firstTrack = tracks.find(t => t.id === newQueue[0]);
        if (firstTrack) {
          await playTrack(firstTrack);
        }
      }
    }
  }, [currentTrack, shuffleQueue, tracks, playTrack, deltaHV]);

  // Reshuffle with new Hilbert curve
  const handleReshuffle = useCallback(async () => {
    const queue = await storyShuffleEngine.generateHilbertShuffle(tracks, deltaHV);
    setShuffleQueue(queue);
    setIsShuffled(true);
  }, [tracks, deltaHV]);

  // Create playlist
  const handleCreatePlaylist = useCallback(() => {
    if (newPlaylistName.trim()) {
      storyShuffleEngine.createPlaylist(newPlaylistName.trim());
      setNewPlaylistName('');
      setShowCreatePlaylist(false);
    }
  }, [newPlaylistName]);

  // Generate AI playlist
  const handleGenerateAIPlaylist = useCallback(async (
    purpose: 'balance' | 'boost' | 'calm' | 'focus' | 'heal' | 'energize'
  ) => {
    if (!deltaHV) return;
    setGeneratingPlaylist(true);
    try {
      await storyShuffleEngine.generateCoherencePlaylist(deltaHV, purpose);
      setShowAIRecommendations(false);
    } finally {
      setGeneratingPlaylist(false);
    }
  }, [deltaHV]);

  // Play entire playlist
  const handlePlayPlaylist = useCallback(async (playlist: Playlist) => {
    if (playlist.trackIds.length === 0) return;
    const firstTrackId = playlist.trackIds[0];
    const firstTrack = tracks.find(t => t.id === firstTrackId);
    if (firstTrack) {
      // Set the shuffle queue to the playlist tracks
      setShuffleQueue(playlist.trackIds);
      await playTrack(firstTrack);
    }
  }, [tracks, playTrack]);

  // Delete playlist (only non-system)
  const handleDeletePlaylist = useCallback((playlistId: string) => {
    if (confirm('Delete this playlist?')) {
      storyShuffleEngine.deletePlaylist(playlistId);
    }
  }, []);

  // Remove track from playlist
  const handleRemoveFromPlaylist = useCallback((playlistId: string, trackId: string) => {
    storyShuffleEngine.removeFromPlaylist(playlistId, trackId);
  }, []);

  // Get AI playlist recommendations
  const aiRecommendations = useMemo(() => {
    return storyShuffleEngine.getPlaylistRecommendations(deltaHV);
  }, [deltaHV]);

  // Get category color
  const getCategoryColor = (categoryId: EmotionalCategoryId): string => {
    return EMOTIONAL_CATEGORIES[categoryId]?.color || '#6b7280';
  };

  // Get metric influence from music
  const metricInfluence = useMemo(() => {
    return storyShuffleEngine.getMetricInfluenceFromMusic();
  }, [storyMetrics]);

  // Format time
  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="bg-gray-950/80 backdrop-blur-xl border border-gray-800 rounded-2xl overflow-hidden">
      {/* Authorship Message Banner */}
      <div className={`px-4 py-3 border-b border-gray-800 ${
        authorshipMessage.type === 'celebration' ? 'bg-gradient-to-r from-purple-900/40 to-pink-900/40' :
        authorshipMessage.type === 'insight' ? 'bg-gradient-to-r from-cyan-900/40 to-blue-900/40' :
        'bg-gradient-to-r from-gray-900/40 to-gray-800/40'
      }`}>
        <div className="flex items-center gap-3">
          <span className="text-2xl">{authorshipMessage.emoji}</span>
          <div className="flex-1">
            <p className="text-sm text-gray-200">{authorshipMessage.message}</p>
            <p className="text-xs text-gray-500 mt-0.5">
              Skip freely - you are the author of your story
            </p>
          </div>
          <BookOpen className="w-5 h-5 text-gray-500" />
        </div>
      </div>

      {/* Now Playing */}
      {currentTrack ? (
        <div className="p-6">
          <div className="flex items-center gap-4 mb-4">
            <div
              className="w-16 h-16 rounded-xl flex items-center justify-center text-3xl"
              style={{ backgroundColor: getCategoryColor(currentTrack.categoryId) + '30' }}
            >
              {EMOTIONAL_CATEGORIES[currentTrack.categoryId]?.icon || '🎵'}
            </div>
            <div className="flex-1 min-w-0">
              <h3 className="text-lg font-medium text-white truncate">{currentTrack.name}</h3>
              <p className="text-sm text-gray-400">
                {EMOTIONAL_CATEGORIES[currentTrack.categoryId]?.name}
              </p>
            </div>
            <button
              onClick={() => setShowStoryMetrics(!showStoryMetrics)}
              className="p-2 rounded-lg hover:bg-gray-800 text-gray-400 hover:text-white transition-colors"
              title="View Story Metrics"
            >
              <TrendingUp className="w-5 h-5" />
            </button>
          </div>

          {/* Progress bar */}
          <div className="mb-4">
            <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-all"
                style={{
                  width: duration > 0 ? `${(progress / duration) * 100}%` : '0%',
                  backgroundColor: getCategoryColor(currentTrack.categoryId)
                }}
              />
            </div>
            <div className="flex justify-between mt-1 text-xs text-gray-500">
              <span>{formatTime(progress)}</span>
              <span>{formatTime(duration)}</span>
            </div>
          </div>

          {/* Controls */}
          <div className="flex items-center justify-center gap-4">
            <button
              onClick={() => setIsShuffled(!isShuffled)}
              className={`p-2 rounded-lg transition-colors ${
                isShuffled ? 'text-cyan-400 bg-cyan-500/20' : 'text-gray-400 hover:text-white'
              }`}
              title="Hilbert Shuffle"
            >
              <Shuffle className="w-5 h-5" />
            </button>

            <button
              onClick={handlePrevious}
              className="p-2 rounded-lg text-gray-400 hover:text-white transition-colors"
            >
              <SkipBack className="w-6 h-6" />
            </button>

            <button
              onClick={togglePlay}
              className="p-4 rounded-full bg-white text-black hover:scale-105 transition-transform"
            >
              {isPlaying ? <Pause className="w-6 h-6" /> : <Play className="w-6 h-6 ml-0.5" />}
            </button>

            {/* SKIP BUTTON - emphasized for empowerment */}
            <button
              onClick={handleSkip}
              className="p-2 rounded-lg text-gray-400 hover:text-white transition-colors group relative"
              title="Skip - Take control of your story"
            >
              <SkipForward className="w-6 h-6" />
              <span className="absolute -top-8 left-1/2 -translate-x-1/2 text-xs text-cyan-400 opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap">
                Your choice!
              </span>
            </button>

            <button
              onClick={() => setRepeatMode(
                repeatMode === 'off' ? 'all' : repeatMode === 'all' ? 'one' : 'off'
              )}
              className={`p-2 rounded-lg transition-colors ${
                repeatMode !== 'off' ? 'text-purple-400 bg-purple-500/20' : 'text-gray-400 hover:text-white'
              }`}
            >
              <Repeat className="w-5 h-5" />
              {repeatMode === 'one' && <span className="text-xs ml-0.5">1</span>}
            </button>
          </div>

          {/* Volume */}
          <div className="flex items-center gap-3 mt-4 px-4">
            <button
              onClick={() => setIsMuted(!isMuted)}
              className="text-gray-400 hover:text-white transition-colors"
            >
              {isMuted ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
            </button>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={isMuted ? 0 : volume}
              onChange={(e) => {
                setVolume(parseFloat(e.target.value));
                setIsMuted(false);
                musicLibrary.setVolume(parseFloat(e.target.value));
              }}
              className="flex-1 h-1 bg-gray-700 rounded-full appearance-none cursor-pointer"
            />
          </div>
        </div>
      ) : (
        <div className="p-8 text-center">
          <Music className="w-12 h-12 text-gray-600 mx-auto mb-4" />
          <p className="text-gray-400 mb-4">No track selected</p>
          <button
            onClick={handleReshuffle}
            className="px-4 py-2 bg-cyan-500/20 text-cyan-300 rounded-lg hover:bg-cyan-500/30 transition-colors"
          >
            <Shuffle className="w-4 h-4 inline mr-2" />
            Start Hilbert Shuffle
          </button>
        </div>
      )}

      {/* Story Metrics Panel */}
      {showStoryMetrics && storyMetrics && (
        <div className="border-t border-gray-800 p-4 bg-gray-900/50">
          <h4 className="text-sm font-medium text-white mb-3 flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-purple-400" />
            Your Story Metrics
          </h4>

          <div className="grid grid-cols-2 gap-3 mb-4">
            {/* Authorship Score */}
            <div className="p-3 bg-gray-800/50 rounded-lg">
              <div className="text-2xl font-light text-cyan-400">{storyMetrics.authorshipScore}%</div>
              <div className="text-xs text-gray-500">Authorship Score</div>
            </div>

            {/* Skip Ratio */}
            <div className="p-3 bg-gray-800/50 rounded-lg">
              <div className="text-2xl font-light text-purple-400">
                {Math.round(storyMetrics.skipRatio * 100)}%
              </div>
              <div className="text-xs text-gray-500">Skip Ratio</div>
            </div>

            {/* Total Plays */}
            <div className="p-3 bg-gray-800/50 rounded-lg">
              <div className="text-2xl font-light text-white">{storyMetrics.totalPlays}</div>
              <div className="text-xs text-gray-500">Total Plays</div>
            </div>

            {/* Emotional Trajectory */}
            <div className="p-3 bg-gray-800/50 rounded-lg">
              <div className="text-lg font-light text-pink-400 capitalize">
                {storyMetrics.emotionalTrajectory}
              </div>
              <div className="text-xs text-gray-500">Trajectory</div>
            </div>
          </div>

          {/* Melancholy Healing Progress */}
          <div className="p-3 bg-gradient-to-r from-blue-900/30 to-purple-900/30 rounded-lg mb-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-gray-300">Healing Progress</span>
              <span className="text-xs text-gray-500 capitalize">
                {storyMetrics.melancholyProgress.currentPhase}
              </span>
            </div>
            <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-blue-500 to-purple-500 rounded-full transition-all"
                style={{ width: `${storyMetrics.melancholyProgress.healingIndicator}%` }}
              />
            </div>
            <p className="text-xs text-gray-500 mt-2">
              {storyMetrics.melancholyProgress.melancholySkips} melancholy songs skipped
              {storyMetrics.melancholyProgress.currentPhase === 'healing' && ' - moving forward!'}
            </p>
          </div>

          {/* Metric Influence from Music */}
          <div className="space-y-2">
            <h5 className="text-xs font-medium text-gray-400 uppercase tracking-wider">
              Music Metric Influence
            </h5>
            {(['symbolic', 'resonance', 'friction', 'stability'] as const).map(metric => (
              <div key={metric} className="flex items-center gap-3">
                <span className="text-xs text-gray-500 w-20 capitalize">{metric}</span>
                <div className="flex-1 h-1.5 bg-gray-800 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full ${
                      metric === 'symbolic' ? 'bg-purple-500' :
                      metric === 'resonance' ? 'bg-cyan-500' :
                      metric === 'friction' ? 'bg-orange-500' :
                      'bg-green-500'
                    }`}
                    style={{ width: `${metricInfluence[metric].value}%` }}
                  />
                </div>
                <span className="text-xs text-gray-400 w-8">{metricInfluence[metric].value}%</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Playlists Panel */}
      <div className="border-t border-gray-800">
        <button
          onClick={() => setShowPlaylists(!showPlaylists)}
          className="w-full px-4 py-3 flex items-center justify-between text-gray-300 hover:bg-gray-800/50 transition-colors"
        >
          <div className="flex items-center gap-2">
            <ListMusic className="w-4 h-4" />
            <span className="text-sm">Playlists</span>
            <span className="text-xs text-gray-500">({playlists.length})</span>
          </div>
          <ChevronRight className={`w-4 h-4 transition-transform ${showPlaylists ? 'rotate-90' : ''}`} />
        </button>

        {showPlaylists && (
          <div className="px-4 pb-4 space-y-3">
            {/* AI Playlist Generator */}
            <div className="p-3 bg-gradient-to-r from-purple-900/30 to-cyan-900/30 rounded-xl border border-purple-500/20">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <Wand2 className="w-4 h-4 text-purple-400" />
                  <span className="text-sm font-medium text-white">AI Playlist Generator</span>
                </div>
                <button
                  onClick={() => setShowAIRecommendations(!showAIRecommendations)}
                  className="text-xs text-purple-300 hover:text-purple-200"
                >
                  {showAIRecommendations ? 'Hide' : 'Show'} Recommendations
                </button>
              </div>

              {showAIRecommendations && (
                <div className="space-y-2 mt-3">
                  {generatingPlaylist ? (
                    <div className="flex items-center justify-center gap-2 py-4 text-purple-300">
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span className="text-sm">Generating your playlist...</span>
                    </div>
                  ) : (
                    <>
                      <p className="text-xs text-gray-400 mb-2">
                        Based on your current metrics, here are playlist suggestions:
                      </p>
                      {aiRecommendations.map((rec, idx) => (
                        <button
                          key={idx}
                          onClick={() => handleGenerateAIPlaylist(rec.purpose)}
                          className="w-full p-2 rounded-lg bg-gray-800/50 hover:bg-gray-800 text-left transition-colors group"
                        >
                          <div className="flex items-center gap-2">
                            <span className="text-lg">{rec.emoji}</span>
                            <div className="flex-1">
                              <div className="text-sm text-white capitalize">{rec.purpose}</div>
                              <div className="text-xs text-gray-500">{rec.reason}</div>
                            </div>
                            <Wand2 className="w-4 h-4 text-purple-400 opacity-0 group-hover:opacity-100 transition-opacity" />
                          </div>
                        </button>
                      ))}
                    </>
                  )}
                </div>
              )}

              {!showAIRecommendations && (
                <p className="text-xs text-gray-400">
                  Let AI create a playlist based on your coherence needs
                </p>
              )}
            </div>

            {/* Playlists List */}
            <div className="space-y-2">
              {playlists.map(playlist => (
                <div key={playlist.id}>
                  <div
                    className="flex items-center gap-3 p-2 rounded-lg hover:bg-gray-800/50 transition-colors cursor-pointer group"
                    onClick={() => editingPlaylistId === playlist.id ? setEditingPlaylistId(null) : null}
                  >
                    <span className="text-xl">{playlist.coverEmoji}</span>
                    <div className="flex-1 min-w-0">
                      <div className="text-sm text-white truncate flex items-center gap-2">
                        {playlist.name}
                        {playlist.name.startsWith('AI:') && (
                          <span className="text-[10px] px-1.5 py-0.5 bg-purple-500/20 text-purple-300 rounded">AI</span>
                        )}
                      </div>
                      <div className="text-xs text-gray-500">
                        {playlist.trackIds.length} tracks
                        {playlist.isSystem && ' • System'}
                      </div>
                    </div>
                    <div className="flex items-center gap-1">
                      <button
                        onClick={(e) => { e.stopPropagation(); handlePlayPlaylist(playlist); }}
                        className="opacity-0 group-hover:opacity-100 p-1.5 text-gray-400 hover:text-white transition-all"
                        title="Play playlist"
                      >
                        <Play className="w-4 h-4" />
                      </button>
                      <button
                        onClick={(e) => { e.stopPropagation(); setEditingPlaylistId(editingPlaylistId === playlist.id ? null : playlist.id); }}
                        className="opacity-0 group-hover:opacity-100 p-1.5 text-gray-400 hover:text-cyan-400 transition-all"
                        title="Edit playlist"
                      >
                        <Edit3 className="w-4 h-4" />
                      </button>
                      {!playlist.isSystem && (
                        <button
                          onClick={(e) => { e.stopPropagation(); handleDeletePlaylist(playlist.id); }}
                          className="opacity-0 group-hover:opacity-100 p-1.5 text-gray-400 hover:text-red-400 transition-all"
                          title="Delete playlist"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      )}
                    </div>
                  </div>

                  {/* Expanded Playlist Editor */}
                  {editingPlaylistId === playlist.id && (
                    <div className="ml-8 mt-2 mb-2 p-2 bg-gray-900/50 rounded-lg border border-gray-800 max-h-48 overflow-y-auto">
                      {playlist.trackIds.length === 0 ? (
                        <p className="text-xs text-gray-500 text-center py-2">No tracks in this playlist</p>
                      ) : (
                        <div className="space-y-1">
                          {playlist.trackIds.map((trackId, idx) => {
                            const track = tracks.find(t => t.id === trackId);
                            if (!track) return null;
                            return (
                              <div key={trackId} className="flex items-center gap-2 p-1.5 rounded hover:bg-gray-800/50 group/track">
                                <GripVertical className="w-3 h-3 text-gray-600" />
                                <span className="text-xs text-gray-400 w-4">{idx + 1}</span>
                                <span className="text-sm">{EMOTIONAL_CATEGORIES[track.categoryId]?.icon}</span>
                                <span className="text-xs text-gray-300 flex-1 truncate">{track.name}</span>
                                <button
                                  onClick={() => handleRemoveFromPlaylist(playlist.id, trackId)}
                                  className="opacity-0 group-hover/track:opacity-100 p-1 text-gray-500 hover:text-red-400"
                                >
                                  <X className="w-3 h-3" />
                                </button>
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>

            {/* Create Playlist Button */}
            {showCreatePlaylist ? (
              <div className="flex items-center gap-2 p-2 bg-gray-800/50 rounded-lg">
                <input
                  type="text"
                  value={newPlaylistName}
                  onChange={(e) => setNewPlaylistName(e.target.value)}
                  placeholder="Playlist name..."
                  className="flex-1 bg-transparent text-sm text-white focus:outline-none"
                  autoFocus
                  onKeyDown={(e) => e.key === 'Enter' && handleCreatePlaylist()}
                />
                <button
                  onClick={handleCreatePlaylist}
                  className="p-1 text-green-400 hover:text-green-300"
                >
                  <Check className="w-4 h-4" />
                </button>
                <button
                  onClick={() => {
                    setShowCreatePlaylist(false);
                    setNewPlaylistName('');
                  }}
                  className="p-1 text-gray-400 hover:text-white"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            ) : (
              <button
                onClick={() => setShowCreatePlaylist(true)}
                className="flex items-center gap-2 p-2 w-full rounded-lg border border-dashed border-gray-700 text-gray-400 hover:text-white hover:border-gray-600 transition-colors"
              >
                <Plus className="w-4 h-4" />
                <span className="text-sm">Create Playlist</span>
              </button>
            )}
          </div>
        )}
      </div>

      {/* Skip Encouragement Footer */}
      <div className="px-4 py-2 bg-gradient-to-r from-cyan-900/20 to-purple-900/20 border-t border-gray-800">
        <p className="text-xs text-center text-gray-400">
          💫 Remember: Every skip is authorship. You're in control of your story.
        </p>
      </div>
    </div>
  );
};

export default StoryMusicPlayer;
