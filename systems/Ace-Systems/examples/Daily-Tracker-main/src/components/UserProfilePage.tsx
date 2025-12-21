import React, { useState, useEffect } from 'react';
import {
  Target,
  Zap,
  Heart,
  Brain,
  Sparkles,
  Users,
  Palette,
  Settings,
  ArrowLeft,
  Plus,
  X,
  Check,
  ChevronRight,
  TrendingUp,
  Flame,
  BarChart2,
} from 'lucide-react';
import { userProfileService } from '../lib/userProfile';
import type { UserProfile, UserGoals, HabitTrack, UserPreferences } from '../lib/userProfile';
import { roadmapEngine, calculateFriction, predictEnergyState } from '../lib/roadmapEngine';
import type { Roadmap, AIRecommendation, EnergyState } from '../lib/roadmapEngine';
import { LIFE_DOMAINS } from '../lib/glyphSystem';
import { NeuralMapView } from './NeuralMapView';
import type { DeltaHVState } from '../lib/deltaHVEngine';

interface UserProfilePageProps {
  onBack: () => void;
  onNavigateToRoadmap?: (category: string) => void;
  deltaHV?: DeltaHVState;
  onSelectBeat?: (category: string) => void;
}

type ProfileTab = 'overview' | 'neural' | 'goals' | 'habits' | 'preferences' | 'roadmap';

const categoryIcons: Record<string, React.ReactNode> = {
  physical: <Zap className="w-4 h-4" />,
  emotional: <Heart className="w-4 h-4" />,
  mental: <Brain className="w-4 h-4" />,
  spiritual: <Sparkles className="w-4 h-4" />,
  social: <Users className="w-4 h-4" />,
  creative: <Palette className="w-4 h-4" />,
};

export const UserProfilePage: React.FC<UserProfilePageProps> = ({
  onBack,
  onNavigateToRoadmap,
  deltaHV,
  onSelectBeat,
}) => {
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [activeTab, setActiveTab] = useState<ProfileTab>('overview');
  const [loading, setLoading] = useState(true);
  const [recommendations, setRecommendations] = useState<AIRecommendation[]>([]);
  const [energyState, setEnergyState] = useState<EnergyState | null>(null);
  const [friction, setFriction] = useState<{ total: number; breakdown: Record<string, number> } | null>(null);
  // Modal states
  const [showGoalModal, setShowGoalModal] = useState(false);
  const [showHabitModal, setShowHabitModal] = useState(false);
  const [editingGoal, setEditingGoal] = useState<UserGoals | null>(null);
  const [showSettingsPanel, setShowSettingsPanel] = useState(false);

  useEffect(() => {
    initializeProfile();
  }, []);

  const initializeProfile = async () => {
    setLoading(true);
    try {
      const loadedProfile = await userProfileService.initialize();
      setProfile(loadedProfile);

      await roadmapEngine.initialize();
      setRecommendations(roadmapEngine.getRecommendations());

      // Calculate friction and energy
      const metrics = loadedProfile.metricsHistory;
      if (metrics.length > 0) {
        const frictionCalc = calculateFriction(loadedProfile, metrics);
        setFriction(frictionCalc);
        setEnergyState(predictEnergyState(loadedProfile, metrics));
      }
    } catch (error) {
      console.error('Failed to initialize profile:', error);
    }
    setLoading(false);
  };

  const handleDismissRecommendation = async (id: string) => {
    await roadmapEngine.dismissRecommendation(id);
    setRecommendations(roadmapEngine.getRecommendations());
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-black via-gray-900 to-blue-950 flex items-center justify-center">
        <div className="text-gray-400 animate-pulse">Loading profile...</div>
      </div>
    );
  }

  if (!profile) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-black via-gray-900 to-blue-950 flex items-center justify-center">
        <div className="text-red-400">Failed to load profile</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-black via-gray-900 to-blue-950 pb-20">
      {/* Header */}
      <header className="sticky top-0 z-40 backdrop-blur-xl bg-black/60 border-b border-gray-800">
        <div className="max-w-6xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <button
              onClick={onBack}
              className="flex items-center gap-2 text-gray-400 hover:text-white transition-colors"
            >
              <ArrowLeft className="w-5 h-5" />
              <span>Back</span>
            </button>
            <h1 className="text-xl font-light text-white">Profile</h1>
            <button
              onClick={() => setShowSettingsPanel(true)}
              className="p-2 rounded-lg hover:bg-gray-800 text-gray-400 hover:text-white transition-colors"
            >
              <Settings className="w-5 h-5" />
            </button>
          </div>
        </div>
      </header>

      {/* Profile Header Card */}
      <div className="max-w-6xl mx-auto px-4 py-6">
        <div className="bg-gray-950/60 backdrop-blur border border-gray-800 rounded-2xl p-6 mb-6">
          <div className="flex items-start gap-6">
            {/* Avatar */}
            <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-cyan-500/20 to-purple-500/20 border border-gray-700 flex items-center justify-center text-4xl">
              {profile.avatarGlyph || '🌀'}
            </div>

            {/* Info */}
            <div className="flex-1">
              <h2 className="text-2xl font-light text-white mb-1">{profile.displayName}</h2>
              <div className="flex items-center gap-4 text-sm text-gray-400">
                <span>Member since {new Date(profile.createdAt).toLocaleDateString()}</span>
                {profile.onboardingComplete && (
                  <span className="flex items-center gap-1 text-green-400">
                    <Check className="w-4 h-4" /> Profile complete
                  </span>
                )}
              </div>

              {/* Quick Stats */}
              <div className="flex gap-6 mt-4">
                <div className="text-center">
                  <div className="text-2xl font-light text-cyan-400">{profile.goals.length}</div>
                  <div className="text-xs text-gray-500">Goals</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-light text-purple-400">{profile.habits.length}</div>
                  <div className="text-xs text-gray-500">Habits</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-light text-pink-400">
                    {profile.metricsHistory.length}
                  </div>
                  <div className="text-xs text-gray-500">Days Tracked</div>
                </div>
                {friction && (
                  <div className="text-center">
                    <div className={`text-2xl font-light ${friction.total < 30 ? 'text-green-400' : friction.total < 60 ? 'text-yellow-400' : 'text-red-400'}`}>
                      {(100 - friction.total).toFixed(0)}%
                    </div>
                    <div className="text-xs text-gray-500">Coherence</div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Tab Navigation */}
        <div className="flex gap-2 mb-6 overflow-x-auto pb-2">
          {(['overview', 'neural', 'goals', 'habits', 'preferences', 'roadmap'] as ProfileTab[]).map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all whitespace-nowrap ${
                activeTab === tab
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                  : 'bg-gray-900/50 text-gray-400 border border-gray-800 hover:text-white hover:border-gray-700'
              }`}
            >
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
            </button>
          ))}
        </div>

        {/* Tab Content */}
        {activeTab === 'overview' && (
          <OverviewTab
            recommendations={recommendations}
            energyState={energyState}
            friction={friction}
            onDismissRecommendation={handleDismissRecommendation}
            onNavigateToRoadmap={onNavigateToRoadmap}
          />
        )}

        {activeTab === 'neural' && deltaHV && (
          <NeuralMapView
            deltaHV={deltaHV}
            onSelectBeat={onSelectBeat}
          />
        )}

        {activeTab === 'neural' && !deltaHV && (
          <div className="bg-gray-950/60 backdrop-blur border border-gray-800 rounded-xl p-8 text-center">
            <Brain className="w-12 h-12 text-gray-600 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-white mb-2">Neural Map Loading</h3>
            <p className="text-gray-400">
              Start tracking your daily beats to generate your personalized neural map based on
              DeltaHV metrics and Free Energy Principle analysis.
            </p>
          </div>
        )}

        {activeTab === 'goals' && (
          <GoalsTab
            profile={profile}
            onAddGoal={() => setShowGoalModal(true)}
            onEditGoal={setEditingGoal}
          />
        )}

        {activeTab === 'habits' && (
          <HabitsTab
            profile={profile}
            onAddHabit={() => setShowHabitModal(true)}
            onLogHabit={async (habitId, completed) => {
              await userProfileService.logHabitEntry(habitId, { completed, count: 1 });
              const updated = userProfileService.getProfile();
              if (updated) setProfile(updated);
            }}
          />
        )}

        {activeTab === 'preferences' && (
          <PreferencesTab
            profile={profile}
            onUpdatePreferences={async (prefs) => {
              await userProfileService.updatePreferences(prefs);
              const updated = userProfileService.getProfile();
              if (updated) setProfile(updated);
            }}
          />
        )}

        {activeTab === 'roadmap' && (
          <RoadmapTab
            profile={profile}
            onNavigateToRoadmap={onNavigateToRoadmap}
          />
        )}
      </div>

      {/* Goal Modal */}
      {(showGoalModal || editingGoal) && (
        <GoalModal
          goal={editingGoal}
          onClose={() => {
            setShowGoalModal(false);
            setEditingGoal(null);
          }}
          onSave={async (goal) => {
            if (editingGoal) {
              await userProfileService.updateGoal(editingGoal.id, goal);
            } else {
              await userProfileService.addGoal(goal as any);
            }
            const updated = userProfileService.getProfile();
            if (updated) setProfile(updated);
            setShowGoalModal(false);
            setEditingGoal(null);
          }}
        />
      )}

      {/* Habit Modal */}
      {showHabitModal && (
        <HabitModal
          onClose={() => setShowHabitModal(false)}
          onSave={async (habit) => {
            await userProfileService.addHabit(habit);
            const updated = userProfileService.getProfile();
            if (updated) setProfile(updated);
            setShowHabitModal(false);
          }}
        />
      )}

      {/* Settings Panel */}
      {showSettingsPanel && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm">
          <div className="h-full w-full max-w-lg ml-auto bg-gray-950 border-l border-gray-800 overflow-y-auto">
            {/* Settings Header */}
            <div className="sticky top-0 bg-gray-950/95 backdrop-blur border-b border-gray-800 p-4">
              <div className="flex items-center justify-between">
                <h2 className="text-xl font-light text-white flex items-center gap-2">
                  <Settings className="w-5 h-5 text-cyan-400" />
                  Settings
                </h2>
                <button
                  onClick={() => setShowSettingsPanel(false)}
                  className="p-2 rounded-lg hover:bg-gray-800 text-gray-400 hover:text-white transition-colors"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
            </div>

            <div className="p-4 space-y-6">
              {/* Profile Settings */}
              <div className="bg-gray-900/50 rounded-xl p-4 border border-gray-800">
                <h3 className="text-lg font-medium text-white mb-4">Profile</h3>
                <div className="space-y-4">
                  <div>
                    <label className="text-sm text-gray-400 block mb-2">Display Name</label>
                    <input
                      type="text"
                      value={profile.displayName}
                      onChange={async (e) => {
                        await userProfileService.updateProfile({ displayName: e.target.value });
                        setProfile(userProfileService.getProfile()!);
                      }}
                      className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2.5 text-white focus:border-purple-500 focus:outline-none"
                    />
                  </div>
                  <div>
                    <label className="text-sm text-gray-400 block mb-2">Avatar Glyph</label>
                    <div className="flex gap-2 flex-wrap">
                      {['🌀', '✨', '🔮', '🌟', '💫', '🦋', '🌊', '⚡', '🔥', '🌈', '🌙', '☀️'].map(glyph => (
                        <button
                          key={glyph}
                          onClick={async () => {
                            await userProfileService.updateProfile({ avatarGlyph: glyph });
                            setProfile(userProfileService.getProfile()!);
                          }}
                          className={`w-10 h-10 rounded-lg flex items-center justify-center text-xl transition-all ${
                            profile.avatarGlyph === glyph
                              ? 'bg-purple-600/30 border-2 border-purple-400'
                              : 'bg-gray-800 border border-gray-700 hover:border-gray-600'
                          }`}
                        >
                          {glyph}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              {/* Display Preferences */}
              <div className="bg-gray-900/50 rounded-xl p-4 border border-gray-800">
                <h3 className="text-lg font-medium text-white mb-4">Display</h3>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <span className="text-white">Dark Theme</span>
                      <p className="text-xs text-gray-500">Use dark mode interface</p>
                    </div>
                    <button
                      onClick={async () => {
                        const newTheme = profile.preferences.theme === 'dark' ? 'light' : 'dark';
                        await userProfileService.updatePreferences({ theme: newTheme });
                        setProfile(userProfileService.getProfile()!);
                      }}
                      className={`w-12 h-6 rounded-full transition-colors ${
                        profile.preferences.theme === 'dark' ? 'bg-purple-600' : 'bg-gray-600'
                      }`}
                    >
                      <div className={`w-5 h-5 rounded-full bg-white transform transition-transform ${
                        profile.preferences.theme === 'dark' ? 'translate-x-6' : 'translate-x-0.5'
                      }`} />
                    </button>
                  </div>
                  <div className="flex items-center justify-between">
                    <div>
                      <span className="text-white">Show Glyphs</span>
                      <p className="text-xs text-gray-500">Display symbolic glyphs in UI</p>
                    </div>
                    <button
                      onClick={async () => {
                        await userProfileService.updatePreferences({ showGlyphs: !profile.preferences.showGlyphs });
                        setProfile(userProfileService.getProfile()!);
                      }}
                      className={`w-12 h-6 rounded-full transition-colors ${
                        profile.preferences.showGlyphs ? 'bg-purple-600' : 'bg-gray-600'
                      }`}
                    >
                      <div className={`w-5 h-5 rounded-full bg-white transform transition-transform ${
                        profile.preferences.showGlyphs ? 'translate-x-6' : 'translate-x-0.5'
                      }`} />
                    </button>
                  </div>
                  <div className="flex items-center justify-between">
                    <div>
                      <span className="text-white">Focus Mode</span>
                      <p className="text-xs text-gray-500">Minimize distractions</p>
                    </div>
                    <button
                      onClick={async () => {
                        await userProfileService.updatePreferences({ focusMode: !profile.preferences.focusMode });
                        setProfile(userProfileService.getProfile()!);
                      }}
                      className={`w-12 h-6 rounded-full transition-colors ${
                        profile.preferences.focusMode ? 'bg-purple-600' : 'bg-gray-600'
                      }`}
                    >
                      <div className={`w-5 h-5 rounded-full bg-white transform transition-transform ${
                        profile.preferences.focusMode ? 'translate-x-6' : 'translate-x-0.5'
                      }`} />
                    </button>
                  </div>
                </div>
              </div>

              {/* Metrics Display */}
              <div className="bg-gray-900/50 rounded-xl p-4 border border-gray-800">
                <h3 className="text-lg font-medium text-white mb-4">Metrics Display</h3>
                <div className="flex gap-2 flex-wrap">
                  {(['minimal', 'detailed', 'full'] as const).map(mode => (
                    <button
                      key={mode}
                      onClick={async () => {
                        await userProfileService.updatePreferences({ metricsDisplay: mode });
                        setProfile(userProfileService.getProfile()!);
                      }}
                      className={`px-4 py-2 rounded-lg text-sm capitalize ${
                        profile.preferences.metricsDisplay === mode
                          ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                          : 'bg-gray-800 text-gray-400 border border-gray-700'
                      }`}
                    >
                      {mode}
                    </button>
                  ))}
                </div>
              </div>

              {/* Dashboard Layout */}
              <div className="bg-gray-900/50 rounded-xl p-4 border border-gray-800">
                <h3 className="text-lg font-medium text-white mb-4">Dashboard Layout</h3>
                <div className="flex gap-2">
                  {(['compact', 'expanded'] as const).map(layout => (
                    <button
                      key={layout}
                      onClick={async () => {
                        await userProfileService.updatePreferences({ dashboardLayout: layout });
                        setProfile(userProfileService.getProfile()!);
                      }}
                      className={`px-4 py-2 rounded-lg text-sm capitalize ${
                        profile.preferences.dashboardLayout === layout
                          ? 'bg-purple-500/20 text-purple-300 border border-purple-500/30'
                          : 'bg-gray-800 text-gray-400 border border-gray-700'
                      }`}
                    >
                      {layout}
                    </button>
                  ))}
                </div>
              </div>

              {/* Reminders */}
              <div className="bg-gray-900/50 rounded-xl p-4 border border-gray-800">
                <h3 className="text-lg font-medium text-white mb-4">Reminders</h3>
                <div className="flex gap-2 flex-wrap">
                  {(['gentle', 'assertive', 'silent'] as const).map(style => (
                    <button
                      key={style}
                      onClick={async () => {
                        await userProfileService.updatePreferences({ reminderStyle: style });
                        setProfile(userProfileService.getProfile()!);
                      }}
                      className={`px-4 py-2 rounded-lg text-sm capitalize ${
                        profile.preferences.reminderStyle === style
                          ? 'bg-pink-500/20 text-pink-300 border border-pink-500/30'
                          : 'bg-gray-800 text-gray-400 border border-gray-700'
                      }`}
                    >
                      {style}
                    </button>
                  ))}
                </div>
              </div>

              {/* Roadmap Updates */}
              <div className="bg-gray-900/50 rounded-xl p-4 border border-gray-800">
                <h3 className="text-lg font-medium text-white mb-4">Roadmap Updates</h3>
                <div className="flex gap-2 flex-wrap">
                  {(['daily', 'weekly', 'biweekly', 'monthly'] as const).map(freq => (
                    <button
                      key={freq}
                      onClick={async () => {
                        await userProfileService.updatePreferences({ roadmapUpdateFrequency: freq });
                        setProfile(userProfileService.getProfile()!);
                      }}
                      className={`px-4 py-2 rounded-lg text-sm capitalize ${
                        profile.preferences.roadmapUpdateFrequency === freq
                          ? 'bg-green-500/20 text-green-300 border border-green-500/30'
                          : 'bg-gray-800 text-gray-400 border border-gray-700'
                      }`}
                    >
                      {freq}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// ============================================================================
// Overview Tab
// ============================================================================

interface OverviewTabProps {
  recommendations: AIRecommendation[];
  energyState: EnergyState | null;
  friction: { total: number; breakdown: Record<string, number> } | null;
  onDismissRecommendation: (id: string) => void;
  onNavigateToRoadmap?: (category: string) => void;
}

const OverviewTab: React.FC<OverviewTabProps> = ({
  recommendations,
  energyState,
  friction,
  onDismissRecommendation,
  onNavigateToRoadmap,
}) => {
  return (
    <div className="space-y-6">
      {/* Energy State - Redesigned */}
      {energyState && (
        <div className="bg-gradient-to-br from-gray-950/80 via-gray-900/60 to-gray-950/80 backdrop-blur-xl border border-gray-700/50 rounded-2xl p-6 shadow-2xl">
          {/* Header with gradient text */}
          <div className="flex items-center justify-between mb-6">
            <h3 className="text-xl font-semibold flex items-center gap-3">
              <div className="p-2 rounded-xl bg-gradient-to-br from-yellow-500/20 to-orange-500/20 border border-yellow-500/30">
                <Zap className="w-5 h-5 text-yellow-400" />
              </div>
              <span className="bg-gradient-to-r from-white to-gray-400 bg-clip-text text-transparent">
                Energy State
              </span>
            </h3>
            <div className={`px-3 py-1 rounded-full text-xs font-medium flex items-center gap-1.5 ${
              energyState.trend === 'rising' ? 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30' :
              energyState.trend === 'declining' ? 'bg-red-500/20 text-red-300 border border-red-500/30' :
              'bg-amber-500/20 text-amber-300 border border-amber-500/30'
            }`}>
              <span className={`w-1.5 h-1.5 rounded-full ${
                energyState.trend === 'rising' ? 'bg-emerald-400' :
                energyState.trend === 'declining' ? 'bg-red-400' :
                'bg-amber-400'
              } animate-pulse`}></span>
              {energyState.trend === 'rising' ? 'Rising' : energyState.trend === 'declining' ? 'Declining' : 'Stable'}
            </div>
          </div>

          {/* Energy Dimensions - Modern Cards */}
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
            {([
              { key: 'physical', icon: '💪', gradient: 'from-cyan-500 to-blue-600', bg: 'from-cyan-500/10 to-blue-600/10', border: 'border-cyan-500/30' },
              { key: 'mental', icon: '🧠', gradient: 'from-purple-500 to-violet-600', bg: 'from-purple-500/10 to-violet-600/10', border: 'border-purple-500/30' },
              { key: 'emotional', icon: '💗', gradient: 'from-pink-500 to-rose-600', bg: 'from-pink-500/10 to-rose-600/10', border: 'border-pink-500/30' },
              { key: 'social', icon: '🤝', gradient: 'from-emerald-500 to-green-600', bg: 'from-emerald-500/10 to-green-600/10', border: 'border-emerald-500/30' },
              { key: 'creative', icon: '✨', gradient: 'from-orange-500 to-amber-600', bg: 'from-orange-500/10 to-amber-600/10', border: 'border-orange-500/30' }
            ] as const).map(({ key, icon, gradient, bg, border }) => {
              const value = energyState[key as keyof typeof energyState] as number;
              const isLow = value < 40;
              const isHigh = value >= 70;

              return (
                <div
                  key={key}
                  className={`relative p-4 rounded-xl bg-gradient-to-br ${bg} border ${border} backdrop-blur-sm transition-all duration-300 hover:scale-[1.02] hover:shadow-lg group overflow-hidden`}
                >
                  {/* Glow effect for high energy */}
                  {isHigh && (
                    <div className={`absolute inset-0 bg-gradient-to-br ${gradient} opacity-10 animate-pulse`}></div>
                  )}

                  {/* Low energy warning pulse */}
                  {isLow && (
                    <div className="absolute top-2 right-2">
                      <span className="relative flex h-2 w-2">
                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                        <span className="relative inline-flex rounded-full h-2 w-2 bg-red-500"></span>
                      </span>
                    </div>
                  )}

                  {/* Icon and label */}
                  <div className="flex items-center gap-2 mb-3">
                    <span className="text-lg">{icon}</span>
                    <span className="text-xs font-medium text-gray-400 capitalize">{key}</span>
                  </div>

                  {/* Progress bar */}
                  <div className="relative h-2 bg-gray-800/50 rounded-full overflow-hidden mb-2">
                    <div
                      className={`absolute inset-y-0 left-0 bg-gradient-to-r ${gradient} rounded-full transition-all duration-500 shadow-lg`}
                      style={{ width: `${value}%`, boxShadow: isHigh ? `0 0 10px ${key === 'physical' ? '#22d3ee' : key === 'mental' ? '#a855f7' : key === 'emotional' ? '#ec4899' : key === 'social' ? '#22c55e' : '#fb923c'}40` : 'none' }}
                    />
                  </div>

                  {/* Value display */}
                  <div className="flex items-baseline justify-between">
                    <span className={`text-2xl font-bold bg-gradient-to-r ${gradient} bg-clip-text text-transparent`}>
                      {value}
                    </span>
                    <span className="text-xs text-gray-500">/ 100</span>
                  </div>

                  {/* Status text */}
                  <div className={`mt-1 text-[10px] ${
                    isHigh ? 'text-emerald-400' : isLow ? 'text-red-400' : 'text-gray-500'
                  }`}>
                    {isHigh ? 'Optimal' : isLow ? 'Needs attention' : 'Moderate'}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Tomorrow Forecast - Enhanced */}
          <div className="mt-5 p-4 rounded-xl bg-gradient-to-r from-indigo-500/10 via-purple-500/10 to-pink-500/10 border border-purple-500/20">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-purple-500/20">
                  <TrendingUp className="w-4 h-4 text-purple-400" />
                </div>
                <div>
                  <div className="text-xs text-gray-400">Tomorrow's Forecast</div>
                  <div className="text-sm font-medium text-white">Predicted Energy Level</div>
                </div>
              </div>
              <div className="text-right">
                <div className={`text-3xl font-bold ${
                  energyState.predictedTomorrow >= 70 ? 'text-emerald-400' :
                  energyState.predictedTomorrow >= 40 ? 'text-amber-400' :
                  'text-red-400'
                }`}>
                  {energyState.predictedTomorrow}%
                </div>
                <div className="text-[10px] text-gray-500">
                  {energyState.predictedTomorrow >= 70 ? 'Looking great!' :
                   energyState.predictedTomorrow >= 40 ? 'Room for improvement' :
                   'Consider extra rest'}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* AI Recommendations */}
      {recommendations.length > 0 && (
        <div className="bg-gray-950/60 backdrop-blur border border-gray-800 rounded-xl p-5">
          <h3 className="text-lg font-medium text-white mb-4 flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-purple-400" />
            AI Recommendations
          </h3>
          <div className="space-y-3">
            {recommendations.slice(0, 5).map(rec => (
              <div
                key={rec.id}
                className={`p-4 rounded-lg border ${
                  rec.priority === 'critical' ? 'bg-red-500/10 border-red-500/30' :
                  rec.priority === 'high' ? 'bg-orange-500/10 border-orange-500/30' :
                  rec.priority === 'medium' ? 'bg-yellow-500/10 border-yellow-500/30' :
                  'bg-gray-800/50 border-gray-700'
                }`}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <h4 className="text-white font-medium">{rec.title}</h4>
                    <p className="text-sm text-gray-400 mt-1">{rec.rationale}</p>
                    <p className="text-sm text-cyan-400 mt-2">{rec.suggestedAction}</p>
                  </div>
                  <button
                    onClick={() => onDismissRecommendation(rec.id)}
                    className="p-1 text-gray-500 hover:text-gray-300"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Friction Breakdown */}
      {friction && (
        <div className="bg-gray-950/60 backdrop-blur border border-gray-800 rounded-xl p-5">
          <h3 className="text-lg font-medium text-white mb-4 flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-cyan-400" />
            Friction Analysis
          </h3>
          <div className="space-y-3">
            {Object.entries(friction.breakdown).map(([key, value]) => (
              <div key={key}>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-gray-400 capitalize">{key.replace(/([A-Z])/g, ' $1')}</span>
                  <span className={value < 10 ? 'text-green-400' : value < 20 ? 'text-yellow-400' : 'text-red-400'}>
                    {value.toFixed(1)}
                  </span>
                </div>
                <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full ${
                      value < 10 ? 'bg-green-500' : value < 20 ? 'bg-yellow-500' : 'bg-red-500'
                    }`}
                    style={{ width: `${Math.min(100, value * 2)}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
          <div className="mt-4 pt-4 border-t border-gray-800">
            <div className="text-center">
              <div className={`text-3xl font-light ${
                friction.total < 30 ? 'text-green-400' :
                friction.total < 60 ? 'text-yellow-400' :
                'text-red-400'
              }`}>
                {friction.total.toFixed(0)}
              </div>
              <div className="text-xs text-gray-500">Total Friction Score</div>
            </div>
          </div>
        </div>
      )}

      {/* Life Domains Quick Access */}
      <div className="bg-gray-950/60 backdrop-blur border border-gray-800 rounded-xl p-5">
        <h3 className="text-lg font-medium text-white mb-4 flex items-center gap-2">
          <Target className="w-5 h-5 text-pink-400" />
          Life Domains
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
          {LIFE_DOMAINS.map(domain => (
            <button
              key={domain.id}
              onClick={() => onNavigateToRoadmap?.(domain.id)}
              className="p-4 rounded-lg bg-gray-900/50 border border-gray-800 hover:border-gray-600 transition-colors text-left group"
            >
              <div className="flex items-center gap-3 mb-2">
                <span className="text-2xl">{domain.glyphs[0]}</span>
                <span className="text-white group-hover:text-cyan-300 transition-colors">
                  {domain.name}
                </span>
              </div>
              <p className="text-xs text-gray-500 line-clamp-2">{domain.description}</p>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

// ============================================================================
// Goals Tab
// ============================================================================

interface GoalsTabProps {
  profile: UserProfile;
  onAddGoal: () => void;
  onEditGoal: (goal: UserGoals) => void;
}

const GoalsTab: React.FC<GoalsTabProps> = ({ profile, onAddGoal, onEditGoal }) => {
  const groupedGoals = profile.goals.reduce((acc, goal) => {
    if (!acc[goal.category]) acc[goal.category] = [];
    acc[goal.category].push(goal);
    return acc;
  }, {} as Record<string, UserGoals[]>);

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-medium text-white">Your Goals</h3>
        <button
          onClick={onAddGoal}
          className="flex items-center gap-2 px-4 py-2 bg-cyan-500/20 text-cyan-300 rounded-lg hover:bg-cyan-500/30 transition-colors"
        >
          <Plus className="w-4 h-4" />
          Add Goal
        </button>
      </div>

      {profile.goals.length === 0 ? (
        <div className="bg-gray-950/60 backdrop-blur border border-gray-800 rounded-xl p-8 text-center">
          <Target className="w-12 h-12 text-gray-600 mx-auto mb-4" />
          <p className="text-gray-400 mb-4">No goals yet. Start by defining what you want to achieve.</p>
          <button
            onClick={onAddGoal}
            className="px-6 py-3 bg-cyan-500/20 text-cyan-300 rounded-lg hover:bg-cyan-500/30 transition-colors"
          >
            Create Your First Goal
          </button>
        </div>
      ) : (
        Object.entries(groupedGoals).map(([category, goals]) => (
          <div key={category} className="bg-gray-950/60 backdrop-blur border border-gray-800 rounded-xl p-5">
            <h4 className="text-white font-medium mb-4 flex items-center gap-2 capitalize">
              {categoryIcons[category]}
              {category}
            </h4>
            <div className="space-y-3">
              {goals.map(goal => (
                <div
                  key={goal.id}
                  className="p-4 bg-gray-900/50 rounded-lg border border-gray-800 hover:border-gray-700 transition-colors cursor-pointer"
                  onClick={() => onEditGoal(goal)}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <h5 className="text-white font-medium">{goal.title}</h5>
                      <p className="text-sm text-gray-400 mt-1 line-clamp-2">{goal.description}</p>
                    </div>
                    <div className="text-right">
                      <div className="text-2xl font-light text-cyan-400">{goal.progress}%</div>
                      <div className="text-xs text-gray-500">progress</div>
                    </div>
                  </div>
                  <div className="mt-3">
                    <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-cyan-500 to-purple-500 rounded-full transition-all"
                        style={{ width: `${goal.progress}%` }}
                      />
                    </div>
                  </div>
                  {goal.milestones.length > 0 && (
                    <div className="mt-3 flex flex-wrap gap-2">
                      {goal.milestones.slice(0, 3).map(m => (
                        <span
                          key={m.id}
                          className={`text-xs px-2 py-1 rounded-full ${
                            m.completed
                              ? 'bg-green-500/20 text-green-300'
                              : 'bg-gray-800 text-gray-400'
                          }`}
                        >
                          {m.completed && <Check className="w-3 h-3 inline mr-1" />}
                          {m.title}
                        </span>
                      ))}
                      {goal.milestones.length > 3 && (
                        <span className="text-xs text-gray-500">
                          +{goal.milestones.length - 3} more
                        </span>
                      )}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        ))
      )}
    </div>
  );
};

// ============================================================================
// Habits Tab
// ============================================================================

interface HabitsTabProps {
  profile: UserProfile;
  onAddHabit: () => void;
  onLogHabit: (habitId: string, completed: boolean) => void;
}

const HabitsTab: React.FC<HabitsTabProps> = ({ profile, onAddHabit, onLogHabit }) => {
  const today = new Date().toISOString().split('T')[0];

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-medium text-white">Your Habits</h3>
        <button
          onClick={onAddHabit}
          className="flex items-center gap-2 px-4 py-2 bg-purple-500/20 text-purple-300 rounded-lg hover:bg-purple-500/30 transition-colors"
        >
          <Plus className="w-4 h-4" />
          Add Habit
        </button>
      </div>

      {profile.habits.length === 0 ? (
        <div className="bg-gray-950/60 backdrop-blur border border-gray-800 rounded-xl p-8 text-center">
          <Flame className="w-12 h-12 text-gray-600 mx-auto mb-4" />
          <p className="text-gray-400 mb-4">No habits tracked yet. Build consistency with habit tracking.</p>
          <button
            onClick={onAddHabit}
            className="px-6 py-3 bg-purple-500/20 text-purple-300 rounded-lg hover:bg-purple-500/30 transition-colors"
          >
            Create Your First Habit
          </button>
        </div>
      ) : (
        <div className="grid gap-4">
          {profile.habits.map(habit => {
            const todayEntry = habit.history.find(h => h.date === today);
            const isCompletedToday = todayEntry?.completed || false;

            return (
              <div
                key={habit.id}
                className="bg-gray-950/60 backdrop-blur border border-gray-800 rounded-xl p-5"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <button
                      onClick={() => onLogHabit(habit.id, !isCompletedToday)}
                      className={`w-10 h-10 rounded-full border-2 flex items-center justify-center transition-all ${
                        isCompletedToday
                          ? 'bg-green-500/20 border-green-500 text-green-400'
                          : 'border-gray-600 text-gray-600 hover:border-gray-400'
                      }`}
                    >
                      {isCompletedToday && <Check className="w-5 h-5" />}
                    </button>
                    <div>
                      <h4 className="text-white font-medium">{habit.name}</h4>
                      <p className="text-sm text-gray-400">
                        {habit.frequency === 'daily' ? 'Daily' :
                         habit.frequency === 'weekly' ? 'Weekly' : 'Custom'} habit
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="flex items-center gap-2">
                      <Flame className={`w-4 h-4 ${habit.currentStreak > 0 ? 'text-orange-400' : 'text-gray-600'}`} />
                      <span className="text-xl font-light text-white">{habit.currentStreak}</span>
                    </div>
                    <p className="text-xs text-gray-500">day streak</p>
                  </div>
                </div>

                {/* Weekly visualization */}
                <div className="mt-4 flex gap-1">
                  {Array.from({ length: 7 }).map((_, i) => {
                    const date = new Date();
                    date.setDate(date.getDate() - (6 - i));
                    const dateStr = date.toISOString().split('T')[0];
                    const entry = habit.history.find(h => h.date === dateStr);

                    return (
                      <div
                        key={i}
                        className={`flex-1 h-8 rounded ${
                          entry?.completed
                            ? 'bg-green-500/30'
                            : 'bg-gray-800'
                        }`}
                        title={dateStr}
                      />
                    );
                  })}
                </div>
                <div className="flex justify-between mt-1 text-xs text-gray-500">
                  <span>7 days ago</span>
                  <span>Today</span>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

// ============================================================================
// Preferences Tab
// ============================================================================

interface PreferencesTabProps {
  profile: UserProfile;
  onUpdatePreferences: (prefs: Partial<UserPreferences>) => void;
}

const PreferencesTab: React.FC<PreferencesTabProps> = ({ profile, onUpdatePreferences }) => {
  return (
    <div className="space-y-6">
      <div className="bg-gray-950/60 backdrop-blur border border-gray-800 rounded-xl p-5">
        <h3 className="text-lg font-medium text-white mb-4">Display</h3>
        <div className="space-y-4">
          <div>
            <label className="text-sm text-gray-400 block mb-2">Dashboard Layout</label>
            <div className="flex gap-2">
              {(['compact', 'expanded'] as const).map(layout => (
                <button
                  key={layout}
                  onClick={() => onUpdatePreferences({ dashboardLayout: layout })}
                  className={`px-4 py-2 rounded-lg text-sm ${
                    profile.preferences.dashboardLayout === layout
                      ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                      : 'bg-gray-900 text-gray-400 border border-gray-800'
                  }`}
                >
                  {layout.charAt(0).toUpperCase() + layout.slice(1)}
                </button>
              ))}
            </div>
          </div>

          <div>
            <label className="text-sm text-gray-400 block mb-2">Metrics Display</label>
            <div className="flex gap-2">
              {(['minimal', 'detailed', 'full'] as const).map(display => (
                <button
                  key={display}
                  onClick={() => onUpdatePreferences({ metricsDisplay: display })}
                  className={`px-4 py-2 rounded-lg text-sm ${
                    profile.preferences.metricsDisplay === display
                      ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                      : 'bg-gray-900 text-gray-400 border border-gray-800'
                  }`}
                >
                  {display.charAt(0).toUpperCase() + display.slice(1)}
                </button>
              ))}
            </div>
          </div>

          <div className="flex items-center justify-between">
            <div>
              <label className="text-white block">Show Glyphs</label>
              <p className="text-xs text-gray-500">Display symbolic glyphs in the interface</p>
            </div>
            <button
              onClick={() => onUpdatePreferences({ showGlyphs: !profile.preferences.showGlyphs })}
              className={`w-12 h-6 rounded-full transition-colors ${
                profile.preferences.showGlyphs ? 'bg-cyan-500' : 'bg-gray-700'
              }`}
            >
              <div className={`w-5 h-5 rounded-full bg-white transform transition-transform ${
                profile.preferences.showGlyphs ? 'translate-x-6' : 'translate-x-0.5'
              }`} />
            </button>
          </div>
        </div>
      </div>

      <div className="bg-gray-950/60 backdrop-blur border border-gray-800 rounded-xl p-5">
        <h3 className="text-lg font-medium text-white mb-4">Roadmap Updates</h3>
        <div className="space-y-4">
          <div>
            <label className="text-sm text-gray-400 block mb-2">Update Frequency</label>
            <div className="flex flex-wrap gap-2">
              {(['daily', 'weekly', 'biweekly', 'monthly'] as const).map(freq => (
                <button
                  key={freq}
                  onClick={() => onUpdatePreferences({ roadmapUpdateFrequency: freq })}
                  className={`px-4 py-2 rounded-lg text-sm ${
                    profile.preferences.roadmapUpdateFrequency === freq
                      ? 'bg-purple-500/20 text-purple-300 border border-purple-500/30'
                      : 'bg-gray-900 text-gray-400 border border-gray-800'
                  }`}
                >
                  {freq.charAt(0).toUpperCase() + freq.slice(1)}
                </button>
              ))}
            </div>
          </div>

          {profile.preferences.nextScheduledUpdate && (
            <div className="text-sm text-gray-400">
              Next update: {new Date(profile.preferences.nextScheduledUpdate).toLocaleDateString()}
            </div>
          )}
        </div>
      </div>

      <div className="bg-gray-950/60 backdrop-blur border border-gray-800 rounded-xl p-5">
        <h3 className="text-lg font-medium text-white mb-4">Notifications</h3>
        <div className="space-y-4">
          <div>
            <label className="text-sm text-gray-400 block mb-2">Reminder Style</label>
            <div className="flex gap-2">
              {(['gentle', 'assertive', 'silent'] as const).map(style => (
                <button
                  key={style}
                  onClick={() => onUpdatePreferences({ reminderStyle: style })}
                  className={`px-4 py-2 rounded-lg text-sm ${
                    profile.preferences.reminderStyle === style
                      ? 'bg-pink-500/20 text-pink-300 border border-pink-500/30'
                      : 'bg-gray-900 text-gray-400 border border-gray-800'
                  }`}
                >
                  {style.charAt(0).toUpperCase() + style.slice(1)}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Usage Statistics Section */}
      <div className="bg-gradient-to-br from-gray-950/80 via-indigo-950/30 to-gray-950/80 backdrop-blur border border-indigo-500/20 rounded-xl p-5">
        <h3 className="text-lg font-medium text-white mb-4 flex items-center gap-2">
          <BarChart2 className="w-5 h-5 text-indigo-400" />
          Your Usage Patterns
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {/* Total Days Active */}
          <div className="bg-gray-900/50 rounded-lg p-3 text-center">
            <p className="text-2xl font-bold text-indigo-300">
              {profile.metricsHistory.length}
            </p>
            <p className="text-xs text-gray-500">Days Tracked</p>
          </div>

          {/* Average Delta HV */}
          <div className="bg-gray-900/50 rounded-lg p-3 text-center">
            <p className="text-2xl font-bold text-cyan-300">
              {profile.metricsHistory.length > 0
                ? Math.round(profile.metricsHistory.reduce((sum, m) => sum + m.deltaHV, 0) / profile.metricsHistory.length)
                : '—'}
            </p>
            <p className="text-xs text-gray-500">Avg ΔHV Score</p>
          </div>

          {/* Total Beats Completed */}
          <div className="bg-gray-900/50 rounded-lg p-3 text-center">
            <p className="text-2xl font-bold text-emerald-300">
              {profile.metricsHistory.reduce((sum, m) => sum + m.completedBeats, 0)}
            </p>
            <p className="text-xs text-gray-500">Total Beats</p>
          </div>

          {/* Total Journal Entries */}
          <div className="bg-gray-900/50 rounded-lg p-3 text-center">
            <p className="text-2xl font-bold text-amber-300">
              {profile.metricsHistory.reduce((sum, m) => sum + m.journalEntries, 0)}
            </p>
            <p className="text-xs text-gray-500">Journal Entries</p>
          </div>
        </div>

        {/* Field State Distribution */}
        {profile.metricsHistory.length > 0 && (
          <div className="mt-4">
            <p className="text-sm text-gray-400 mb-2">Field State Distribution</p>
            <div className="flex gap-2 flex-wrap">
              {(['coherent', 'transitioning', 'fragmented', 'dormant'] as const).map(state => {
                const count = profile.metricsHistory.filter(m => m.fieldState === state).length;
                const percent = Math.round((count / profile.metricsHistory.length) * 100);
                const colors = {
                  coherent: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30',
                  transitioning: 'bg-amber-500/20 text-amber-300 border-amber-500/30',
                  fragmented: 'bg-rose-500/20 text-rose-300 border-rose-500/30',
                  dormant: 'bg-gray-500/20 text-gray-300 border-gray-500/30',
                };
                return (
                  <div key={state} className={`px-3 py-1 rounded-lg text-xs border ${colors[state]}`}>
                    {state}: {percent}%
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Performance Trend */}
        {profile.metricsHistory.length >= 7 && (
          <div className="mt-4">
            <p className="text-sm text-gray-400 mb-2">Recent Trend (7 days)</p>
            <div className="flex items-end gap-1 h-16">
              {profile.metricsHistory.slice(-7).map((m, i) => (
                <div
                  key={i}
                  className="flex-1 bg-gradient-to-t from-indigo-500/50 to-cyan-500/50 rounded-t"
                  style={{ height: `${Math.max(10, m.deltaHV)}%` }}
                  title={`${new Date(m.date).toLocaleDateString()}: ${m.deltaHV}`}
                />
              ))}
            </div>
            <div className="flex justify-between text-xs text-gray-600 mt-1">
              <span>7d ago</span>
              <span>Today</span>
            </div>
          </div>
        )}

        {profile.metricsHistory.length === 0 && (
          <p className="text-sm text-gray-500 mt-4 text-center">
            Start tracking your daily beats to see your usage patterns here.
          </p>
        )}
      </div>
    </div>
  );
};

// ============================================================================
// Roadmap Tab
// ============================================================================

interface RoadmapTabProps {
  profile: UserProfile;
  onNavigateToRoadmap?: (category: string) => void;
}

const RoadmapTab: React.FC<RoadmapTabProps> = ({ profile, onNavigateToRoadmap }) => {
  const [roadmaps, setRoadmaps] = useState<Roadmap[]>([]);

  useEffect(() => {
    setRoadmaps(roadmapEngine.getRoadmaps());
  }, []);

  return (
    <div className="space-y-6">
      <div className="bg-gray-950/60 backdrop-blur border border-gray-800 rounded-xl p-5">
        <h3 className="text-lg font-medium text-white mb-4">Your Roadmaps</h3>
        <p className="text-sm text-gray-400 mb-4">
          Personalized paths for each life domain, generated using the Free Energy Principle to minimize friction and maximize coherence.
        </p>

        <div className="grid gap-4">
          {LIFE_DOMAINS.map(domain => {
            const roadmap = roadmaps.find(r => r.category === domain.id);
            const completedSteps = roadmap?.steps.filter(s => s.status === 'completed').length || 0;
            const totalSteps = roadmap?.steps.length || 0;
            const progress = totalSteps > 0 ? (completedSteps / totalSteps) * 100 : 0;

            return (
              <button
                key={domain.id}
                onClick={() => onNavigateToRoadmap?.(domain.id)}
                className="p-4 bg-gray-900/50 rounded-lg border border-gray-800 hover:border-gray-600 transition-colors text-left group"
              >
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-3">
                    <span className="text-2xl">{domain.glyphs[0]}</span>
                    <div>
                      <h4 className="text-white group-hover:text-cyan-300 transition-colors">
                        {domain.name}
                      </h4>
                      <p className="text-xs text-gray-500">{domain.description}</p>
                    </div>
                  </div>
                  <ChevronRight className="w-5 h-5 text-gray-600 group-hover:text-gray-400" />
                </div>
                <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-cyan-500 to-purple-500 rounded-full"
                    style={{ width: `${progress}%` }}
                  />
                </div>
                <div className="flex justify-between mt-2 text-xs text-gray-500">
                  <span>{completedSteps}/{totalSteps} steps</span>
                  <span>{progress.toFixed(0)}% complete</span>
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* Beat Roadmap Questions */}
      <div className="bg-gray-950/60 backdrop-blur border border-gray-800 rounded-xl p-5">
        <h3 className="text-lg font-medium text-white mb-4">Beat Focus Areas</h3>
        <p className="text-sm text-gray-400 mb-4">
          Answer questions for each beat type to personalize your roadmap.
        </p>

        <div className="space-y-3">
          {profile.beatRoadmaps.map(br => {
            const answeredCount = br.questions.filter(q => q.answer).length;
            const totalQuestions = br.questions.length;

            return (
              <div
                key={br.category}
                className="p-3 bg-gray-900/50 rounded-lg border border-gray-800"
              >
                <div className="flex items-center justify-between">
                  <span className="text-white">{br.category}</span>
                  <span className="text-xs text-gray-400">
                    {answeredCount}/{totalQuestions} answered
                  </span>
                </div>
                {br.currentFocus && (
                  <p className="text-sm text-cyan-400 mt-1">Focus: {br.currentFocus}</p>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

// ============================================================================
// Goal Modal
// ============================================================================

interface GoalModalProps {
  goal: UserGoals | null;
  onClose: () => void;
  onSave: (goal: Partial<UserGoals>) => void;
}

const GoalModal: React.FC<GoalModalProps> = ({ goal, onClose, onSave }) => {
  const [title, setTitle] = useState(goal?.title || '');
  const [description, setDescription] = useState(goal?.description || '');
  const [category, setCategory] = useState<UserGoals['category']>(goal?.category || 'physical');
  const [progress, setProgress] = useState(goal?.progress || 0);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
      <div className="bg-gray-950 border border-gray-800 rounded-2xl p-6 w-full max-w-md mx-4">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-light text-white">
            {goal ? 'Edit Goal' : 'New Goal'}
          </h2>
          <button onClick={onClose} className="text-gray-400 hover:text-white">
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="space-y-4">
          <div>
            <label className="text-sm text-gray-400 block mb-2">Title</label>
            <input
              type="text"
              value={title}
              onChange={e => setTitle(e.target.value)}
              className="w-full bg-gray-900 border border-gray-800 rounded-lg px-4 py-2 text-white focus:border-cyan-500 focus:outline-none"
              placeholder="What do you want to achieve?"
            />
          </div>

          <div>
            <label className="text-sm text-gray-400 block mb-2">Category</label>
            <div className="flex flex-wrap gap-2">
              {(['physical', 'emotional', 'mental', 'spiritual', 'social', 'creative'] as const).map(cat => (
                <button
                  key={cat}
                  onClick={() => setCategory(cat)}
                  className={`px-3 py-1 rounded-full text-sm flex items-center gap-1 ${
                    category === cat
                      ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                      : 'bg-gray-900 text-gray-400 border border-gray-800'
                  }`}
                >
                  {categoryIcons[cat]}
                  {cat}
                </button>
              ))}
            </div>
          </div>

          <div>
            <label className="text-sm text-gray-400 block mb-2">Description</label>
            <textarea
              value={description}
              onChange={e => setDescription(e.target.value)}
              className="w-full bg-gray-900 border border-gray-800 rounded-lg px-4 py-2 text-white focus:border-cyan-500 focus:outline-none resize-none h-24"
              placeholder="Describe your goal..."
            />
          </div>

          {goal && (
            <div>
              <label className="text-sm text-gray-400 block mb-2">Progress: {progress}%</label>
              <input
                type="range"
                min="0"
                max="100"
                value={progress}
                onChange={e => setProgress(parseInt(e.target.value))}
                className="w-full"
              />
            </div>
          )}
        </div>

        <div className="flex gap-3 mt-6">
          <button
            onClick={onClose}
            className="flex-1 px-4 py-2 bg-gray-800 text-gray-300 rounded-lg hover:bg-gray-700 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={() => onSave({ title, description, category, progress, milestones: goal?.milestones || [], linkedBeats: goal?.linkedBeats || [] })}
            disabled={!title.trim()}
            className="flex-1 px-4 py-2 bg-cyan-500/20 text-cyan-300 rounded-lg hover:bg-cyan-500/30 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {goal ? 'Save Changes' : 'Create Goal'}
          </button>
        </div>
      </div>
    </div>
  );
};

// ============================================================================
// Habit Modal
// ============================================================================

interface HabitModalProps {
  onClose: () => void;
  onSave: (habit: Omit<HabitTrack, 'id' | 'createdAt' | 'history'>) => void;
}

const HabitModal: React.FC<HabitModalProps> = ({ onClose, onSave }) => {
  const [name, setName] = useState('');
  const [category, setCategory] = useState('General');
  const [frequency, setFrequency] = useState<'daily' | 'weekly' | 'custom'>('daily');
  const [targetCount] = useState(1);

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm">
      <div className="bg-gray-950 border border-gray-800 rounded-2xl p-6 w-full max-w-md mx-4">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-light text-white">New Habit</h2>
          <button onClick={onClose} className="text-gray-400 hover:text-white">
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="space-y-4">
          <div>
            <label className="text-sm text-gray-400 block mb-2">Habit Name</label>
            <input
              type="text"
              value={name}
              onChange={e => setName(e.target.value)}
              className="w-full bg-gray-900 border border-gray-800 rounded-lg px-4 py-2 text-white focus:border-purple-500 focus:outline-none"
              placeholder="What habit do you want to build?"
            />
          </div>

          <div>
            <label className="text-sm text-gray-400 block mb-2">Category</label>
            <select
              value={category}
              onChange={e => setCategory(e.target.value)}
              className="w-full bg-gray-900 border border-gray-800 rounded-lg px-4 py-2 text-white focus:border-purple-500 focus:outline-none"
            >
              {['Workout', 'Meditation', 'Emotion', 'Moderation', 'Journal', 'General'].map(cat => (
                <option key={cat} value={cat}>{cat}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="text-sm text-gray-400 block mb-2">Frequency</label>
            <div className="flex gap-2">
              {(['daily', 'weekly'] as const).map(freq => (
                <button
                  key={freq}
                  onClick={() => setFrequency(freq)}
                  className={`px-4 py-2 rounded-lg text-sm ${
                    frequency === freq
                      ? 'bg-purple-500/20 text-purple-300 border border-purple-500/30'
                      : 'bg-gray-900 text-gray-400 border border-gray-800'
                  }`}
                >
                  {freq.charAt(0).toUpperCase() + freq.slice(1)}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="flex gap-3 mt-6">
          <button
            onClick={onClose}
            className="flex-1 px-4 py-2 bg-gray-800 text-gray-300 rounded-lg hover:bg-gray-700 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={() => onSave({
              name,
              category,
              frequency,
              targetCount,
              currentStreak: 0,
              longestStreak: 0,
            })}
            disabled={!name.trim()}
            className="flex-1 px-4 py-2 bg-purple-500/20 text-purple-300 rounded-lg hover:bg-purple-500/30 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Create Habit
          </button>
        </div>
      </div>
    </div>
  );
};

export default UserProfilePage;
