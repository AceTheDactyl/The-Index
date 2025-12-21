/**
 * Music Library Component
 *
 * UI for managing music files, emotional folders, and playback.
 * Integrates with meditation beats and emotional coherence tracking.
 */

import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  X, Download, Play, Pause, Volume2, VolumeX,
  Music, Folder, FolderOpen, Trash2, Plus,
  ChevronDown, ChevronUp, Loader2
} from 'lucide-react';
import {
  musicLibrary,
  EMOTIONAL_CATEGORIES,
  type EmotionalCategoryId,
  type MusicTrack,
  type MusicSession
} from '../lib/musicLibrary';

interface MusicLibraryProps {
  onClose?: () => void;
  onSelectTrack?: (track: MusicTrack, desiredEmotion: EmotionalCategoryId) => void;
  linkedBeatId?: string;
  linkedBeatType?: string;
  compact?: boolean;
}

/**
 * Format duration in mm:ss
 */
function formatDuration(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}:${secs.toString().padStart(2, '0')}`;
}

/**
 * Format file size
 */
function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/**
 * Post-Session Modal Component
 */
function PostSessionModal({
  session,
  onSubmit,
  onClose
}: {
  session: MusicSession;
  onSubmit: (data: MusicSession['postSession']) => void;
  onClose: () => void;
}) {
  const [experiencedEmotion, setExperiencedEmotion] = useState<EmotionalCategoryId>(session.desiredEmotion);
  const [intensityRating, setIntensityRating] = useState(3);
  const [moodBefore, setMoodBefore] = useState(3);
  const [moodAfter, setMoodAfter] = useState(3);
  const [notes, setNotes] = useState('');

  const handleSubmit = () => {
    onSubmit({
      experiencedEmotion,
      intensityRating,
      moodBefore,
      moodAfter,
      notes: notes || undefined,
      coherenceMatch: experiencedEmotion === session.desiredEmotion
    });
  };

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-[60] p-4">
      <div className="w-full max-w-md bg-gray-950 border border-gray-800 rounded-2xl overflow-hidden">
        <div className="p-4 border-b border-gray-800">
          <h3 className="text-lg font-medium">How was your session?</h3>
          <p className="text-sm text-gray-400">
            You listened to "{session.trackName}" for {formatDuration(session.duration)}
          </p>
        </div>

        <div className="p-4 space-y-5">
          {/* Desired vs Experienced */}
          <div className="space-y-3">
            <p className="text-sm text-gray-400">
              You wanted to feel: <span className="text-white font-medium">
                {EMOTIONAL_CATEGORIES[session.desiredEmotion].icon} {EMOTIONAL_CATEGORIES[session.desiredEmotion].name}
              </span>
            </p>

            <div>
              <label className="text-sm text-gray-400 block mb-2">What did you actually experience?</label>
              <div className="grid grid-cols-5 gap-2">
                {Object.entries(EMOTIONAL_CATEGORIES).map(([key, cat]) => (
                  <button
                    key={key}
                    onClick={() => setExperiencedEmotion(key as EmotionalCategoryId)}
                    className={`p-2 rounded-lg border text-center transition-all ${
                      experiencedEmotion === key
                        ? 'border-white bg-white/10'
                        : 'border-gray-700 hover:border-gray-600'
                    }`}
                    title={cat.name}
                  >
                    <span className="text-xl">{cat.icon}</span>
                    <p className="text-[10px] text-gray-400 mt-1 truncate">{cat.name}</p>
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Intensity Rating */}
          <div>
            <label className="text-sm text-gray-400 block mb-2">
              Emotional intensity (1-5)
            </label>
            <div className="flex gap-2">
              {[1, 2, 3, 4, 5].map(n => (
                <button
                  key={n}
                  onClick={() => setIntensityRating(n)}
                  className={`flex-1 py-2 rounded-lg border ${
                    intensityRating === n
                      ? 'bg-purple-600 border-purple-500'
                      : 'bg-gray-900 border-gray-700 hover:border-gray-600'
                  }`}
                >
                  {n}
                </button>
              ))}
            </div>
          </div>

          {/* Mood Before/After */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-sm text-gray-400 block mb-2">Mood before (1-5)</label>
              <div className="flex gap-1">
                {[1, 2, 3, 4, 5].map(n => (
                  <button
                    key={n}
                    onClick={() => setMoodBefore(n)}
                    className={`flex-1 py-1.5 rounded text-sm ${
                      moodBefore === n
                        ? 'bg-cyan-600'
                        : 'bg-gray-800 hover:bg-gray-700'
                    }`}
                  >
                    {n}
                  </button>
                ))}
              </div>
            </div>
            <div>
              <label className="text-sm text-gray-400 block mb-2">Mood after (1-5)</label>
              <div className="flex gap-1">
                {[1, 2, 3, 4, 5].map(n => (
                  <button
                    key={n}
                    onClick={() => setMoodAfter(n)}
                    className={`flex-1 py-1.5 rounded text-sm ${
                      moodAfter === n
                        ? 'bg-emerald-600'
                        : 'bg-gray-800 hover:bg-gray-700'
                    }`}
                  >
                    {n}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Notes */}
          <div>
            <label className="text-sm text-gray-400 block mb-2">Notes (optional)</label>
            <textarea
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="How did the music make you feel?"
              className="w-full px-3 py-2 rounded-lg bg-gray-900 border border-gray-700 text-sm resize-none"
              rows={2}
            />
          </div>
        </div>

        <div className="flex items-center justify-between p-4 border-t border-gray-800">
          <button
            onClick={onClose}
            className="px-4 py-2 rounded-lg bg-gray-800 hover:bg-gray-700 text-sm"
          >
            Skip
          </button>
          <button
            onClick={handleSubmit}
            className="px-4 py-2 rounded-lg bg-purple-600 hover:bg-purple-500 text-sm font-medium"
          >
            Save Feedback
          </button>
        </div>
      </div>
    </div>
  );
}

/**
 * Main Music Library Component
 */
export function MusicLibrary({
  onClose,
  onSelectTrack,
  linkedBeatId,
  linkedBeatType,
  compact = false
}: MusicLibraryProps) {
  const [isLoading, setIsLoading] = useState(true);
  const [tracks, setTracks] = useState<MusicTrack[]>([]);
  const [selectedCategory, setSelectedCategory] = useState<EmotionalCategoryId | null>(null);
  const [expandedCategories, setExpandedCategories] = useState<Set<EmotionalCategoryId>>(new Set());
  const [playbackState, setPlaybackState] = useState(musicLibrary.getPlaybackState());
  const [volume, setVolume] = useState(1);
  const [showPostSession, setShowPostSession] = useState<MusicSession | null>(null);
  const [importing, setImporting] = useState(false);
  const [desiredEmotion, setDesiredEmotion] = useState<EmotionalCategoryId>('CALM');

  const fileInputRef = useRef<HTMLInputElement>(null);
  const playbackIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Load tracks and initialize
  useEffect(() => {
    const init = async () => {
      await musicLibrary.initialize();
      const allTracks = await musicLibrary.getAllTracks();
      setTracks(allTracks);
      setIsLoading(false);

      // Sync initial playback state
      setPlaybackState(musicLibrary.getPlaybackState());
    };

    init();

    // Subscribe to track library changes
    const unsubscribe = musicLibrary.subscribe(async () => {
      const allTracks = await musicLibrary.getAllTracks();
      setTracks(allTracks);
      setPlaybackState(musicLibrary.getPlaybackState());
    });

    // CRITICAL: Subscribe to playback state changes to sync with other components
    // This ensures when StoryMusicPlayer plays/pauses, our UI updates immediately
    const unsubscribePlayback = musicLibrary.subscribeToPlayback(() => {
      // Update playback state when any component changes playback
      setPlaybackState(musicLibrary.getPlaybackState());
    });

    return () => {
      unsubscribe();
      unsubscribePlayback();
      if (playbackIntervalRef.current) {
        clearInterval(playbackIntervalRef.current);
      }
    };
  }, []);

  // Update playback state periodically
  useEffect(() => {
    if (playbackState.isPlaying) {
      playbackIntervalRef.current = setInterval(() => {
        setPlaybackState(musicLibrary.getPlaybackState());
      }, 500);
    } else {
      if (playbackIntervalRef.current) {
        clearInterval(playbackIntervalRef.current);
        playbackIntervalRef.current = null;
      }
    }

    return () => {
      if (playbackIntervalRef.current) {
        clearInterval(playbackIntervalRef.current);
      }
    };
  }, [playbackState.isPlaying]);

  // Handle file import
  const handleFileImport = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = event.target.files;
    if (!files || files.length === 0 || !selectedCategory) return;

    setImporting(true);

    try {
      for (const file of Array.from(files)) {
        await musicLibrary.importTrack(file, selectedCategory);
      }
    } catch (error) {
      console.error('Failed to import:', error);
      alert('Failed to import file. Please make sure it\'s a valid audio file.');
    } finally {
      setImporting(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  }, [selectedCategory]);

  // Play track
  const handlePlay = async (track: MusicTrack) => {
    if (onSelectTrack) {
      onSelectTrack(track, desiredEmotion);
    }

    await musicLibrary.playTrack(track.id, desiredEmotion, linkedBeatId, linkedBeatType);
    setPlaybackState(musicLibrary.getPlaybackState());
  };

  // Stop playback with post-session prompt
  const handleStop = async () => {
    const session = await musicLibrary.stopPlayback();
    setPlaybackState(musicLibrary.getPlaybackState());

    if (session && session.completed && session.duration > 30) {
      setShowPostSession(session);
    }
  };

  // Handle post-session data
  const handlePostSessionSubmit = async (data: MusicSession['postSession']) => {
    if (showPostSession) {
      await musicLibrary.addPostSessionData(showPostSession.id, data);
      setShowPostSession(null);
    }
  };

  // Delete track
  const handleDelete = async (trackId: string) => {
    if (confirm('Delete this track? This cannot be undone.')) {
      await musicLibrary.deleteTrack(trackId);
    }
  };

  // Export track
  const handleExport = async (trackId: string) => {
    await musicLibrary.exportTrack(trackId);
  };

  // Toggle category expansion
  const toggleCategory = (categoryId: EmotionalCategoryId) => {
    setExpandedCategories(prev => {
      const next = new Set(prev);
      if (next.has(categoryId)) {
        next.delete(categoryId);
      } else {
        next.add(categoryId);
      }
      return next;
    });
  };

  // Get tracks by category
  const getTracksByCategory = (categoryId: EmotionalCategoryId) => {
    return tracks.filter(t => t.categoryId === categoryId);
  };

  // Current track info
  const currentTrack = playbackState.currentSession
    ? tracks.find(t => t.id === playbackState.currentSession?.trackId)
    : null;

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-8 h-8 animate-spin text-purple-400" />
      </div>
    );
  }

  return (
    <div className={`flex flex-col ${compact ? 'h-full' : 'min-h-[500px]'}`}>
      {/* Header */}
      {!compact && (
        <div className="flex items-center justify-between p-4 border-b border-gray-800">
          <div className="flex items-center gap-3">
            <Music className="w-6 h-6 text-purple-400" />
            <div>
              <h2 className="text-xl font-light">Music Library</h2>
              <p className="text-xs text-gray-500">
                {tracks.length} tracks across {Object.keys(EMOTIONAL_CATEGORIES).length} emotional categories
              </p>
            </div>
          </div>
          {onClose && (
            <button
              onClick={onClose}
              className="p-2 rounded-lg bg-gray-900/70 border border-gray-800 hover:bg-gray-800"
            >
              <X className="w-5 h-5" />
            </button>
          )}
        </div>
      )}

      {/* Playback Controls (if playing) */}
      {(playbackState.isPlaying || currentTrack) && (
        <div className="p-4 border-b border-gray-800 bg-gradient-to-r from-purple-950/30 to-cyan-950/30">
          <div className="flex items-center gap-4">
            <div
              className="w-12 h-12 rounded-lg flex items-center justify-center"
              style={{ backgroundColor: EMOTIONAL_CATEGORIES[currentTrack?.categoryId || 'CALM'].color + '40' }}
            >
              <span className="text-2xl">
                {EMOTIONAL_CATEGORIES[currentTrack?.categoryId || 'CALM'].icon}
              </span>
            </div>

            <div className="flex-1 min-w-0">
              <p className="font-medium truncate">{currentTrack?.name || 'Unknown'}</p>
              <p className="text-sm text-gray-400">
                {formatDuration(playbackState.currentTime)} / {formatDuration(playbackState.duration)}
              </p>
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={async () => {
                  // Check actual state to avoid race conditions with other components
                  const actuallyPlaying = musicLibrary.isCurrentlyPlaying();
                  if (actuallyPlaying) {
                    musicLibrary.pausePlayback();
                  } else {
                    await musicLibrary.resumePlayback();
                  }
                }}
                className="p-2 rounded-full bg-purple-600 hover:bg-purple-500"
              >
                {playbackState.isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
              </button>
              <button
                onClick={handleStop}
                className="p-2 rounded-lg bg-gray-800 hover:bg-gray-700"
              >
                <X className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* Progress bar */}
          <div className="mt-3">
            <input
              type="range"
              min="0"
              max={playbackState.duration || 100}
              value={playbackState.currentTime}
              onChange={(e) => musicLibrary.seekTo(parseFloat(e.target.value))}
              className="w-full h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer"
            />
          </div>

          {/* Volume */}
          <div className="flex items-center gap-2 mt-2">
            <button onClick={() => setVolume(v => v > 0 ? 0 : 1)}>
              {volume > 0 ? <Volume2 className="w-4 h-4 text-gray-400" /> : <VolumeX className="w-4 h-4 text-gray-400" />}
            </button>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={volume}
              onChange={(e) => {
                const v = parseFloat(e.target.value);
                setVolume(v);
                musicLibrary.setVolume(v);
              }}
              className="w-20 h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer"
            />
          </div>
        </div>
      )}

      {/* Desired Emotion Selector */}
      <div className="p-4 border-b border-gray-800 bg-gray-900/30">
        <label className="text-sm text-gray-400 block mb-2">What emotion do you want to feel?</label>
        <div className="flex flex-wrap gap-2">
          {Object.entries(EMOTIONAL_CATEGORIES).map(([key, cat]) => (
            <button
              key={key}
              onClick={() => setDesiredEmotion(key as EmotionalCategoryId)}
              className={`px-3 py-1.5 rounded-lg text-sm flex items-center gap-1.5 transition-all ${
                desiredEmotion === key
                  ? 'bg-white/10 border-2'
                  : 'bg-gray-800 border border-gray-700 hover:border-gray-600'
              }`}
              style={{
                borderColor: desiredEmotion === key ? cat.color : undefined
              }}
            >
              <span>{cat.icon}</span>
              <span>{cat.name}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Categories & Tracks */}
      <div className="flex-1 overflow-y-auto p-4 space-y-2">
        {Object.entries(EMOTIONAL_CATEGORIES).map(([key, category]) => {
          const categoryTracks = getTracksByCategory(key as EmotionalCategoryId);
          const isExpanded = expandedCategories.has(key as EmotionalCategoryId);

          return (
            <div
              key={key}
              className="rounded-xl border border-gray-800 overflow-hidden"
              style={{ borderColor: isExpanded ? category.color + '50' : undefined }}
            >
              {/* Category Header */}
              <button
                onClick={() => toggleCategory(key as EmotionalCategoryId)}
                className="w-full p-3 flex items-center justify-between hover:bg-gray-900/50 transition-colors"
              >
                <div className="flex items-center gap-3">
                  {isExpanded ? (
                    <FolderOpen className="w-5 h-5" style={{ color: category.color }} />
                  ) : (
                    <Folder className="w-5 h-5" style={{ color: category.color }} />
                  )}
                  <div className="text-left">
                    <p className="font-medium flex items-center gap-2">
                      <span>{category.icon}</span>
                      {category.name}
                      <span className="text-xs px-2 py-0.5 rounded-full bg-gray-800 text-gray-400">
                        {categoryTracks.length}
                      </span>
                    </p>
                    <p className="text-xs text-gray-500">{category.description}</p>
                  </div>
                </div>
                {isExpanded ? <ChevronUp className="w-4 h-4 text-gray-400" /> : <ChevronDown className="w-4 h-4 text-gray-400" />}
              </button>

              {/* Tracks List */}
              {isExpanded && (
                <div className="border-t border-gray-800">
                  {categoryTracks.length === 0 ? (
                    <div className="p-4 text-center text-gray-500 text-sm">
                      <Music className="w-8 h-8 mx-auto mb-2 opacity-50" />
                      <p>No tracks in this category</p>
                      <button
                        onClick={() => {
                          setSelectedCategory(key as EmotionalCategoryId);
                          fileInputRef.current?.click();
                        }}
                        className="mt-2 text-purple-400 hover:text-purple-300 text-xs"
                      >
                        + Import audio file
                      </button>
                    </div>
                  ) : (
                    <div className="divide-y divide-gray-800">
                      {categoryTracks.map(track => (
                        <div
                          key={track.id}
                          className="p-3 flex items-center justify-between hover:bg-gray-900/30 group"
                        >
                          <div className="flex items-center gap-3 flex-1 min-w-0">
                            <button
                              onClick={() => handlePlay(track)}
                              className="p-2 rounded-full bg-gray-800 hover:bg-purple-600 transition-colors"
                            >
                              <Play className="w-4 h-4" />
                            </button>
                            <div className="min-w-0">
                              <p className="font-medium truncate">{track.name}</p>
                              <p className="text-xs text-gray-500">
                                {formatDuration(track.duration)} • {formatFileSize(track.fileSize)}
                                {track.playCount > 0 && ` • ${track.playCount} plays`}
                              </p>
                            </div>
                          </div>

                          <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                            <button
                              onClick={() => handleExport(track.id)}
                              className="p-1.5 rounded hover:bg-gray-800"
                              title="Export"
                            >
                              <Download className="w-4 h-4 text-gray-400" />
                            </button>
                            <button
                              onClick={() => handleDelete(track.id)}
                              className="p-1.5 rounded hover:bg-gray-800"
                              title="Delete"
                            >
                              <Trash2 className="w-4 h-4 text-rose-400" />
                            </button>
                          </div>
                        </div>
                      ))}

                      {/* Import button */}
                      <button
                        onClick={() => {
                          setSelectedCategory(key as EmotionalCategoryId);
                          fileInputRef.current?.click();
                        }}
                        className="w-full p-3 text-sm text-gray-400 hover:text-gray-300 hover:bg-gray-900/30 flex items-center justify-center gap-2"
                      >
                        <Plus className="w-4 h-4" />
                        Import audio file
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept="audio/*,.wav,.mp3,.ogg,.m4a,.flac"
        multiple
        onChange={handleFileImport}
        className="hidden"
      />

      {/* Import loading overlay */}
      {importing && (
        <div className="absolute inset-0 bg-black/60 flex items-center justify-center">
          <div className="flex items-center gap-3 text-white">
            <Loader2 className="w-6 h-6 animate-spin" />
            <span>Importing...</span>
          </div>
        </div>
      )}

      {/* Post-session modal */}
      {showPostSession && (
        <PostSessionModal
          session={showPostSession}
          onSubmit={handlePostSessionSubmit}
          onClose={() => setShowPostSession(null)}
        />
      )}
    </div>
  );
}

export default MusicLibrary;
