/**
 * Unified Brain Region Challenge Component
 *
 * Combines brain region wellness tasks with daily challenges, mini-games,
 * and cosmetic rewards. Uses unified XP system across all challenge types.
 *
 * Features:
 * - Brain region-specific wellness tasks
 * - Daily challenges tied to site activity (music, time, tasks)
 * - Interactive mini-games (riddles, math, color sequence, memory)
 * - Secret challenges with hints
 * - Unified XP and cosmetic reward system
 * - Unique hex gradient cosmetics per brain region
 */

import React, { useState, useEffect, useCallback, useMemo } from 'react';
import {
  X,
  Trophy,
  Zap,
  Target,
  Brain,
  Sparkles,
  CheckCircle2,
  Clock,
  Flame,
  Star,
  Crown,
  Eye,
  Activity,
  Layers,
  Pin,
  PinOff,
  Lock,
  HelpCircle,
  Gamepad2,
  Music,
  ListChecks,
  TrendingUp,
  Gift,
  ChevronRight,
  Calendar,
  Plus,
  Palette,
} from 'lucide-react';
import type { DeltaHVState } from '../lib/deltaHVEngine';
import { BRAIN_REGIONS, type BrainRegion, type BrainRegionCategory } from '../lib/glyphSystem';
import {
  unifiedChallengeService,
  BRAIN_REGION_GRADIENTS,
  type ActiveChallenge,
  type ChallengeType,
} from '../lib/unifiedChallengeSystem';
import MiniGames from './MiniGames';

// Check-in type matching App.tsx
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

interface BrainRegionChallengeProps {
  deltaHV: DeltaHVState | null;
  onClose: () => void;
  onCompleteChallenge?: (challengeId: string, xp: number) => void;
  // Calendar beat creation
  onCreateBeat?: (beat: Omit<CheckIn, 'id' | 'loggedAt' | 'done'>) => void;
  // Navigation to other views
  onNavigateToChallenges?: () => void;
  onNavigateToCosmetics?: () => void;
}

type ViewTab = 'challenges' | 'brain' | 'mini_games' | 'secrets';

// Brain task templates for each category
const BRAIN_TASKS: Record<BrainRegionCategory, Array<{ title: string; description: string; timeEstimate: string }>> = {
  cortical: [
    { title: 'Deep Focus Session', description: 'Complete 25 minutes of uninterrupted focus work', timeEstimate: '25 min' },
    { title: 'Problem Solving', description: 'Work through a challenging problem step by step', timeEstimate: '15 min' },
    { title: 'Learning Sprint', description: 'Learn something new for 15 minutes', timeEstimate: '15 min' },
  ],
  limbic: [
    { title: 'Gratitude Practice', description: 'Write down 3 things you\'re grateful for', timeEstimate: '5 min' },
    { title: 'Emotional Check-In', description: 'Sit quietly and identify your current emotions', timeEstimate: '5 min' },
    { title: 'Self-Compassion Break', description: 'Speak kindly to yourself for 5 minutes', timeEstimate: '5 min' },
  ],
  subcortical: [
    { title: 'Reward Reflection', description: 'Celebrate a recent accomplishment, no matter how small', timeEstimate: '5 min' },
    { title: 'Habit Stack', description: 'Complete a small habit immediately after an existing one', timeEstimate: '5 min' },
    { title: 'Dopamine Detox', description: 'Avoid screens for 30 minutes', timeEstimate: '30 min' },
  ],
  brainstem: [
    { title: 'Breathing Exercise', description: 'Complete 4-7-8 breathing pattern for 5 minutes', timeEstimate: '5 min' },
    { title: 'Body Scan', description: 'Do a full body relaxation scan from head to toe', timeEstimate: '10 min' },
    { title: 'Grounding Exercise', description: '5-4-3-2-1 sensory grounding technique', timeEstimate: '5 min' },
  ],
  cerebellar: [
    { title: 'Balance Practice', description: 'Stand on one foot for 1 minute each side', timeEstimate: '3 min' },
    { title: 'Coordination Drill', description: 'Practice a skill that requires coordination', timeEstimate: '10 min' },
    { title: 'Rhythm Exercise', description: 'Tap out a rhythm pattern or dance to music', timeEstimate: '5 min' },
  ],
  sensory: [
    { title: 'Mindful Listening', description: 'Close your eyes and focus on sounds for 5 minutes', timeEstimate: '5 min' },
    { title: 'Texture Exploration', description: 'Touch 5 different textures and notice sensations', timeEstimate: '5 min' },
    { title: 'Visual Focus', description: 'Practice focused gazing on a single point for 2 minutes', timeEstimate: '2 min' },
  ],
  motor: [
    { title: 'Stretching Routine', description: 'Complete a 10-minute stretch sequence', timeEstimate: '10 min' },
    { title: 'Walking Meditation', description: 'Take a mindful walk, noticing each step', timeEstimate: '10 min' },
    { title: 'Dance Break', description: 'Put on music and move freely for one song', timeEstimate: '5 min' },
  ],
  integration: [
    { title: 'Mind-Body Sync', description: 'Do yoga or tai chi for 15 minutes', timeEstimate: '15 min' },
    { title: 'Cross-Brain Exercise', description: 'Do an activity that uses both hands equally', timeEstimate: '10 min' },
    { title: 'Integration Meditation', description: 'Visualize energy flowing through your entire body', timeEstimate: '10 min' },
  ],
};

export const BrainRegionChallenge: React.FC<BrainRegionChallengeProps> = ({
  deltaHV,
  onClose,
  onCompleteChallenge,
  onCreateBeat,
  onNavigateToChallenges,
  onNavigateToCosmetics,
}) => {
  const [activeTab, setActiveTab] = useState<ViewTab>('challenges');
  const [selectedCategory, setSelectedCategory] = useState<BrainRegionCategory>('cortical');
  const [selectedChallenge, setSelectedChallenge] = useState<ActiveChallenge | null>(null);
  const [selectedBrainTask, setSelectedBrainTask] = useState<{ region: BrainRegion; task: typeof BRAIN_TASKS.cortical[0] } | null>(null);
  const [activeMiniGame, setActiveMiniGame] = useState<ActiveChallenge | null>(null);
  const [stats, setStats] = useState(unifiedChallengeService.getStats());
  const [activeChallenges, setActiveChallenges] = useState<ActiveChallenge[]>([]);
  const [showRewardAnimation, setShowRewardAnimation] = useState<{ xp: number; cosmetic?: string } | null>(null);

  // Load challenges and stats
  useEffect(() => {
    const loadData = () => {
      setStats(unifiedChallengeService.getStats());
      setActiveChallenges(unifiedChallengeService.getActiveChallenges(true));
    };

    loadData();
    const unsubscribe = unifiedChallengeService.subscribe(loadData);
    return unsubscribe;
  }, []);

  // Check secret conditions when metrics change
  useEffect(() => {
    if (deltaHV) {
      unifiedChallengeService.checkSecretConditions({
        symbolic: deltaHV.symbolicDensity,
        resonance: deltaHV.resonanceCoupling,
        friction: deltaHV.frictionCoefficient,
        stability: deltaHV.harmonicStability,
      });
    }
  }, [deltaHV]);

  // Filter challenges by type
  const challengesByType = useMemo(() => {
    const daily = activeChallenges.filter(c => c.type === 'site_activity' || c.type === 'metric_goal' || c.type === 'brain_task');
    // Show ALL mini-games (they're always playable, just limited XP rewards)
    const miniGames = activeChallenges.filter(c => c.type === 'mini_game');
    const secrets = activeChallenges.filter(c => c.type === 'secret');

    return { daily, miniGames, secrets };
  }, [activeChallenges]);

  // Get brain regions filtered by category
  const filteredRegions = useMemo(() => {
    return BRAIN_REGIONS.filter(r => r.category === selectedCategory).slice(0, 8);
  }, [selectedCategory]);

  // Complete a challenge
  const handleCompleteChallenge = useCallback((challengeId: string) => {
    const result = unifiedChallengeService.completeChallenge(challengeId);
    if (result) {
      setShowRewardAnimation({ xp: result.xpEarned, cosmetic: result.cosmeticUnlocked || undefined });
      setTimeout(() => setShowRewardAnimation(null), 3000);
      onCompleteChallenge?.(challengeId, result.xpEarned);
    }
    setSelectedChallenge(null);
  }, [onCompleteChallenge]);

  // Complete a brain task (awards XP)
  const handleCompleteBrainTask = useCallback((region: BrainRegion) => {
    const xp = region.category === 'cortical' ? 30 :
               region.category === 'limbic' ? 25 :
               region.category === 'subcortical' ? 20 : 20;

    const result = unifiedChallengeService.addXP(xp);
    setShowRewardAnimation({ xp });
    setTimeout(() => setShowRewardAnimation(null), 3000);
    onCompleteChallenge?.(region.id, xp);
    setSelectedBrainTask(null);

    if (result.levelUp) {
      // Could show level up animation here
    }
  }, [onCompleteChallenge]);

  // Complete mini-game (always playable, XP limited to 3/day)
  const handleMiniGameComplete = useCallback((score: number) => {
    if (!activeMiniGame) return;

    const result = unifiedChallengeService.completeMiniGame(activeMiniGame.id, score);
    if (result) {
      // Show appropriate feedback based on whether XP was earned
      if (result.xpEarned > 0) {
        setShowRewardAnimation({ xp: result.xpEarned, cosmetic: result.cosmeticUnlocked || undefined });
        onCompleteChallenge?.(activeMiniGame.id, result.xpEarned);
      } else if (result.highScore) {
        // No XP but got a high score
        setShowRewardAnimation({ xp: 0, cosmetic: '🏆 New High Score!' });
      }
      // Always refresh stats to update XP claims remaining
      setStats(unifiedChallengeService.getStats());
      if (result.xpEarned > 0 || result.highScore) {
        setTimeout(() => setShowRewardAnimation(null), 3000);
      }
    }
    setActiveMiniGame(null);
  }, [activeMiniGame, onCompleteChallenge]);

  // Pin/unpin challenge
  const handleTogglePin = useCallback((challengeId: string, isPinned: boolean) => {
    if (isPinned) {
      unifiedChallengeService.unpinChallenge(challengeId);
    } else {
      unifiedChallengeService.pinChallenge(challengeId);
    }
  }, []);

  // Create a calendar beat for task verification
  const handleCreateBeat = useCallback((challenge: ActiveChallenge) => {
    if (!onCreateBeat) {
      // If no beat creation callback, allow direct completion
      handleCompleteChallenge(challenge.id);
      return;
    }

    // Calculate time slot (30 min from now)
    const now = new Date();
    const futureTime = new Date(now.getTime() + 30 * 60000);
    const hours = futureTime.getHours();
    const minutes = futureTime.getMinutes();
    const slot = `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}`;

    // Create a beat for verification
    onCreateBeat({
      category: 'Challenge',
      task: `${challenge.icon} ${challenge.title}`,
      slot,
      note: `Complete: ${challenge.description}. Mark done when finished to earn +${challenge.xpReward} XP!`,
      isAnchor: false,
    });

    setSelectedChallenge(null);
  }, [onCreateBeat, handleCompleteChallenge]);

  // Create a beat for brain task verification
  const handleCreateBrainBeat = useCallback((region: BrainRegion, task: typeof BRAIN_TASKS.cortical[0]) => {
    if (!onCreateBeat) {
      // If no beat creation callback, allow direct completion
      handleCompleteBrainTask(region);
      return;
    }

    const now = new Date();
    const futureTime = new Date(now.getTime() + 30 * 60000);
    const hours = futureTime.getHours();
    const minutes = futureTime.getMinutes();
    const slot = `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}`;

    const xp = region.category === 'cortical' ? 30 :
               region.category === 'limbic' ? 25 :
               region.category === 'subcortical' ? 20 : 20;

    onCreateBeat({
      category: 'Brain Task',
      task: `${region.glyph} ${task.title}`,
      slot,
      note: `${task.description} (~${task.timeEstimate}). Mark done when finished to earn +${xp} XP!`,
      isAnchor: false,
    });

    setSelectedBrainTask(null);
  }, [onCreateBeat, handleCompleteBrainTask]);

  // Get icon for challenge type
  const getChallengeTypeIcon = (type: ChallengeType) => {
    switch (type) {
      case 'site_activity': return <Music className="w-4 h-4" />;
      case 'metric_goal': return <TrendingUp className="w-4 h-4" />;
      case 'brain_task': return <Brain className="w-4 h-4" />;
      case 'mini_game': return <Gamepad2 className="w-4 h-4" />;
      case 'secret': return <Lock className="w-4 h-4" />;
      default: return <Star className="w-4 h-4" />;
    }
  };

  const difficultyColors = {
    easy: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30',
    medium: 'bg-amber-500/20 text-amber-300 border-amber-500/30',
    hard: 'bg-rose-500/20 text-rose-300 border-rose-500/30',
    legendary: 'bg-purple-500/20 text-purple-300 border-purple-500/30',
  };

  const categories: Array<{ id: BrainRegionCategory; label: string; icon: React.ReactNode; color: string }> = [
    { id: 'cortical', label: 'Mind', icon: <Sparkles className="w-4 h-4" />, color: 'purple' },
    { id: 'limbic', label: 'Emotion', icon: <Heart className="w-4 h-4" />, color: 'pink' },
    { id: 'subcortical', label: 'Reward', icon: <Star className="w-4 h-4" />, color: 'amber' },
    { id: 'brainstem', label: 'Body', icon: <Zap className="w-4 h-4" />, color: 'cyan' },
    { id: 'cerebellar', label: 'Balance', icon: <Target className="w-4 h-4" />, color: 'green' },
    { id: 'sensory', label: 'Senses', icon: <Eye className="w-4 h-4" />, color: 'blue' },
    { id: 'motor', label: 'Movement', icon: <Activity className="w-4 h-4" />, color: 'orange' },
    { id: 'integration', label: 'Integrate', icon: <Layers className="w-4 h-4" />, color: 'indigo' },
  ];

  const tabs: Array<{ id: ViewTab; label: string; icon: React.ReactNode; count?: number; badge?: string }> = [
    { id: 'challenges', label: 'Daily', icon: <ListChecks className="w-4 h-4" />, count: challengesByType.daily.filter(c => !c.completed).length },
    { id: 'brain', label: 'Brain Tasks', icon: <Brain className="w-4 h-4" /> },
    { id: 'mini_games', label: 'Mini Games', icon: <Gamepad2 className="w-4 h-4" />, count: challengesByType.miniGames.length, badge: stats.miniGameXPClaimsRemaining > 0 ? `${stats.miniGameXPClaimsRemaining} XP` : undefined },
    { id: 'secrets', label: 'Secrets', icon: <HelpCircle className="w-4 h-4" />, count: stats.secretsFound },
  ];

  const levelProgress = (stats.totalXP % 100);

  return (
    <div className="fixed inset-0 bg-black/90 backdrop-blur-sm z-50 flex flex-col overflow-hidden">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-800">
        <button onClick={onClose} className="p-2 rounded-lg bg-gray-900/70 border border-gray-800 hover:bg-gray-800">
          <X className="w-6 h-6" />
        </button>

        {/* Level & XP */}
        <div className="flex items-center gap-3">
          <div className="text-center">
            <div className="flex items-center gap-2">
              <Crown className="w-5 h-5 text-amber-400" />
              <span className="text-lg font-medium text-white">Level {stats.level}</span>
            </div>
            <div className="w-32 h-1.5 bg-gray-800 rounded-full overflow-hidden mt-1">
              <div
                className="h-full bg-gradient-to-r from-amber-500 to-amber-400 rounded-full transition-all"
                style={{ width: `${levelProgress}%` }}
              />
            </div>
            <span className="text-xs text-gray-500">{stats.xpToNextLevel} XP to next</span>
          </div>
        </div>

        {/* Stats & Navigation */}
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-2 px-3 py-1.5 bg-orange-500/20 rounded-lg border border-orange-500/30">
            <Flame className="w-4 h-4 text-orange-400" />
            <span className="text-sm text-orange-300">{stats.currentStreak}</span>
          </div>
          <div className="flex items-center gap-2 px-3 py-1.5 bg-purple-500/20 rounded-lg border border-purple-500/30">
            <Sparkles className="w-4 h-4 text-purple-400" />
            <span className="text-sm text-purple-300">{stats.totalXP} XP</span>
          </div>
          {/* Hub & Cosmetics Navigation */}
          {onNavigateToChallenges && (
            <button
              onClick={onNavigateToChallenges}
              className="px-3 py-1.5 rounded-lg bg-amber-500/20 border border-amber-500/30 text-amber-300 text-sm flex items-center gap-2 hover:bg-amber-500/30 transition-colors"
              title="Challenge Hub"
            >
              <Target className="w-4 h-4" />
              Hub
            </button>
          )}
          {onNavigateToCosmetics && (
            <button
              onClick={onNavigateToCosmetics}
              className="px-3 py-1.5 rounded-lg bg-pink-500/20 border border-pink-500/30 text-pink-300 text-sm flex items-center gap-2 hover:bg-pink-500/30 transition-colors"
              title="Cosmetics Inventory"
            >
              <Palette className="w-4 h-4" />
              Cosmetics
            </button>
          )}
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="flex items-center gap-2 p-3 border-b border-gray-800 overflow-x-auto">
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm whitespace-nowrap transition-colors ${
              activeTab === tab.id
                ? 'bg-purple-500/30 text-purple-300 border border-purple-500/50'
                : 'bg-gray-800/50 text-gray-400 border border-gray-700 hover:border-gray-600'
            }`}
          >
            {tab.icon}
            {tab.label}
            {tab.count !== undefined && tab.count > 0 && (
              <span className={`px-1.5 py-0.5 rounded text-xs ${
                activeTab === tab.id ? 'bg-purple-500/50' : 'bg-gray-700'
              }`}>
                {tab.count}
              </span>
            )}
            {tab.badge && (
              <span className="px-1.5 py-0.5 rounded text-xs bg-emerald-500/30 text-emerald-300 border border-emerald-500/40">
                {tab.badge}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-y-auto p-4">
        {/* DAILY CHALLENGES TAB */}
        {activeTab === 'challenges' && (
          <div className="space-y-6">
            <div className="text-center mb-4">
              <h2 className="text-xl font-light text-white flex items-center justify-center gap-3">
                <Target className="w-6 h-6 text-cyan-400" />
                Daily Challenges
              </h2>
              <p className="text-sm text-gray-400 mt-1">
                Complete challenges to earn XP and unlock cosmetics
              </p>
            </div>

            {/* Pinned Challenges */}
            {challengesByType.daily.filter(c => c.isPinned && !c.completed).length > 0 && (
              <div className="mb-6">
                <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wider mb-3 flex items-center gap-2">
                  <Pin className="w-4 h-4 text-amber-400" />
                  Pinned
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {challengesByType.daily.filter(c => c.isPinned && !c.completed).map(challenge => (
                    <ChallengeCard
                      key={challenge.id}
                      challenge={challenge}
                      onSelect={() => setSelectedChallenge(challenge)}
                      onTogglePin={() => handleTogglePin(challenge.id, true)}
                      difficultyColors={difficultyColors}
                      getTypeIcon={getChallengeTypeIcon}
                    />
                  ))}
                </div>
              </div>
            )}

            {/* Active Challenges */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {challengesByType.daily.filter(c => !c.isPinned && !c.completed).map(challenge => (
                <ChallengeCard
                  key={challenge.id}
                  challenge={challenge}
                  onSelect={() => setSelectedChallenge(challenge)}
                  onTogglePin={() => handleTogglePin(challenge.id, false)}
                  difficultyColors={difficultyColors}
                  getTypeIcon={getChallengeTypeIcon}
                />
              ))}
            </div>

            {/* Completed Today */}
            {challengesByType.daily.filter(c => c.completed).length > 0 && (
              <div className="mt-6">
                <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wider mb-3 flex items-center gap-2">
                  <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                  Completed Today ({challengesByType.daily.filter(c => c.completed).length})
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
                  {challengesByType.daily.filter(c => c.completed).map(challenge => (
                    <div
                      key={challenge.id}
                      className="p-3 bg-emerald-900/20 rounded-xl border border-emerald-500/20 opacity-60"
                    >
                      <div className="flex items-center gap-2">
                        <span className="text-lg">{challenge.icon}</span>
                        <span className="text-sm text-emerald-300">{challenge.title}</span>
                        <CheckCircle2 className="w-4 h-4 text-emerald-400 ml-auto" />
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* BRAIN TASKS TAB */}
        {activeTab === 'brain' && (
          <div className="space-y-6">
            <div className="text-center mb-4">
              <h2 className="text-xl font-light text-white flex items-center justify-center gap-3">
                <Brain className="w-6 h-6 text-purple-400" />
                Brain Region Tasks
              </h2>
              <p className="text-sm text-gray-400 mt-1">
                Target specific brain regions with wellness activities
              </p>
            </div>

            {/* Category Filter */}
            <div className="flex items-center gap-2 overflow-x-auto pb-2">
              {categories.map(cat => (
                <button
                  key={cat.id}
                  onClick={() => setSelectedCategory(cat.id)}
                  className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm whitespace-nowrap transition-colors ${
                    selectedCategory === cat.id
                      ? 'bg-purple-500/30 text-purple-300 border border-purple-500/50'
                      : 'bg-gray-800/50 text-gray-400 border border-gray-700 hover:border-gray-600'
                  }`}
                >
                  {cat.icon}
                  {cat.label}
                </button>
              ))}
            </div>

            {/* Brain Region Grid */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              {filteredRegions.map(region => {
                const tasks = BRAIN_TASKS[region.category];
                const task = tasks[Math.floor(Math.random() * tasks.length)];
                const gradient = BRAIN_REGION_GRADIENTS[region.category];

                return (
                  <button
                    key={region.id}
                    onClick={() => setSelectedBrainTask({ region, task })}
                    className="p-4 rounded-xl border border-gray-800 bg-gray-900/50 hover:border-gray-700 transition-all text-left group"
                    style={{
                      background: `linear-gradient(135deg, ${gradient.colors[0]}10, ${gradient.colors[1]}10, ${gradient.colors[2]}10)`,
                    }}
                  >
                    <div className="flex items-start justify-between mb-2">
                      <span className="text-3xl">{region.glyph}</span>
                      <ChevronRight className="w-4 h-4 text-gray-600 group-hover:text-gray-400 transition-colors" />
                    </div>
                    <h4 className="text-sm font-medium text-white truncate">{region.name}</h4>
                    <p className="text-xs text-gray-500 truncate">{region.category}</p>

                    {/* Gradient preview */}
                    <div
                      className="mt-2 h-1 rounded-full"
                      style={{
                        background: `linear-gradient(90deg, ${gradient.colors.join(', ')})`,
                      }}
                    />
                  </button>
                );
              })}
            </div>
          </div>
        )}

        {/* MINI GAMES TAB */}
        {activeTab === 'mini_games' && (
          <div className="space-y-6">
            <div className="text-center mb-4">
              <h2 className="text-xl font-light text-white flex items-center justify-center gap-3">
                <Gamepad2 className="w-6 h-6 text-cyan-400" />
                Mini Games
              </h2>
              <p className="text-sm text-gray-400 mt-1">
                Always playable • XP rewards {stats.miniGameXPClaimsRemaining}/3 remaining today
              </p>
              {stats.miniGameXPClaimsRemaining > 0 && (
                <div className="mt-2 inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/20 border border-emerald-500/30">
                  <Sparkles className="w-3 h-3 text-emerald-400" />
                  <span className="text-xs text-emerald-300">{stats.miniGameXPClaimsRemaining} XP bonuses available!</span>
                </div>
              )}
              {stats.miniGameXPClaimsRemaining === 0 && (
                <div className="mt-2 inline-flex items-center gap-2 px-3 py-1 rounded-full bg-gray-500/20 border border-gray-500/30">
                  <Clock className="w-3 h-3 text-gray-400" />
                  <span className="text-xs text-gray-400">Play for fun! XP resets at midnight</span>
                </div>
              )}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {challengesByType.miniGames.map(game => (
                <button
                  key={game.id}
                  onClick={() => setActiveMiniGame(game)}
                  className="p-5 rounded-2xl border border-gray-800 bg-gradient-to-br from-purple-900/20 to-cyan-900/20 hover:border-purple-500/50 transition-all text-left group"
                >
                  <div className="flex items-start gap-4">
                    <span className="text-4xl">{game.icon}</span>
                    <div className="flex-1">
                      <h3 className="text-lg font-medium text-white group-hover:text-purple-300 transition-colors">
                        {game.title}
                      </h3>
                      <p className="text-sm text-gray-400 mt-1">{game.description}</p>
                      <div className="flex items-center gap-3 mt-3">
                        <span className={`text-xs px-2 py-0.5 rounded border ${difficultyColors[game.difficulty]}`}>
                          {game.difficulty}
                        </span>
                        <span className="text-sm text-purple-300">+{game.xpReward} XP</span>
                      </div>
                    </div>
                    <Zap className="w-5 h-5 text-amber-400 opacity-0 group-hover:opacity-100 transition-opacity" />
                  </div>
                </button>
              ))}

              {challengesByType.miniGames.length === 0 && (
                <div className="col-span-2 text-center py-12">
                  <Trophy className="w-12 h-12 text-amber-400 mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-white">All Games Complete!</h3>
                  <p className="text-sm text-gray-400 mt-2">Come back tomorrow for new challenges</p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* SECRETS TAB */}
        {activeTab === 'secrets' && (
          <div className="space-y-6">
            <div className="text-center mb-4">
              <h2 className="text-xl font-light text-white flex items-center justify-center gap-3">
                <HelpCircle className="w-6 h-6 text-amber-400" />
                Secret Challenges
              </h2>
              <p className="text-sm text-gray-400 mt-1">
                Hidden challenges with special rewards
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {unifiedChallengeService.getSecretHints().map(secret => {
                const challenge = challengesByType.secrets.find(c => c.id === secret.id);
                const isDiscovered = secret.discovered;
                const isCompleted = challenge?.completed;

                return (
                  <div
                    key={secret.id}
                    className={`p-5 rounded-2xl border ${
                      isCompleted
                        ? 'bg-emerald-900/20 border-emerald-500/30'
                        : isDiscovered
                        ? 'bg-purple-900/20 border-purple-500/30 cursor-pointer hover:border-purple-400/50'
                        : 'bg-gray-900/50 border-gray-800'
                    }`}
                    onClick={() => isDiscovered && !isCompleted && challenge && setSelectedChallenge(challenge)}
                  >
                    <div className="flex items-start gap-4">
                      <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${
                        isCompleted ? 'bg-emerald-500/20' : isDiscovered ? 'bg-purple-500/20' : 'bg-gray-800'
                      }`}>
                        {isCompleted ? (
                          <CheckCircle2 className="w-6 h-6 text-emerald-400" />
                        ) : isDiscovered ? (
                          <span className="text-2xl">{challenge?.icon}</span>
                        ) : (
                          <Lock className="w-6 h-6 text-gray-600" />
                        )}
                      </div>
                      <div className="flex-1">
                        <h3 className={`text-lg font-medium ${
                          isCompleted ? 'text-emerald-300' : isDiscovered ? 'text-white' : 'text-gray-600'
                        }`}>
                          {isDiscovered ? challenge?.title || 'Secret' : '???'}
                        </h3>
                        <p className={`text-sm mt-1 ${isDiscovered ? 'text-gray-400' : 'text-gray-600 italic'}`}>
                          {isDiscovered ? challenge?.description : `Hint: ${secret.hint}`}
                        </p>
                        {isDiscovered && challenge && (
                          <div className="flex items-center gap-2 mt-2">
                            <span className={`text-xs px-2 py-0.5 rounded border ${difficultyColors[challenge.difficulty]}`}>
                              {challenge.difficulty}
                            </span>
                            <span className="text-sm text-purple-300">+{challenge.xpReward} XP</span>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>

            <div className="text-center text-sm text-gray-500 mt-6">
              {stats.secretsFound} of {stats.totalSecrets} secrets discovered
            </div>
          </div>
        )}
      </div>

      {/* Challenge Detail Modal */}
      {selectedChallenge && (
        <ChallengeModal
          challenge={selectedChallenge}
          onClose={() => setSelectedChallenge(null)}
          onComplete={() => handleCompleteChallenge(selectedChallenge.id)}
          onScheduleBeat={() => handleCreateBeat(selectedChallenge)}
          showBeatOption={!!onCreateBeat && selectedChallenge.type !== 'mini_game'}
          difficultyColors={difficultyColors}
        />
      )}

      {/* Brain Task Modal */}
      {selectedBrainTask && (
        <BrainTaskModal
          region={selectedBrainTask.region}
          task={selectedBrainTask.task}
          onClose={() => setSelectedBrainTask(null)}
          onComplete={() => handleCompleteBrainTask(selectedBrainTask.region)}
          onScheduleBeat={() => handleCreateBrainBeat(selectedBrainTask.region, selectedBrainTask.task)}
          showBeatOption={!!onCreateBeat}
        />
      )}

      {/* Mini Game Modal */}
      {activeMiniGame && (
        <MiniGames
          challenge={activeMiniGame}
          onComplete={handleMiniGameComplete}
          onClose={() => setActiveMiniGame(null)}
        />
      )}

      {/* Reward Animation */}
      {showRewardAnimation && (
        <div className="fixed inset-0 pointer-events-none flex items-center justify-center z-60">
          <div className="animate-bounce bg-gradient-to-r from-amber-500 to-purple-500 rounded-2xl px-8 py-4 shadow-2xl">
            <div className="flex items-center gap-3">
              <Sparkles className="w-8 h-8 text-white" />
              <div>
                <p className="text-2xl font-bold text-white">+{showRewardAnimation.xp} XP</p>
                {showRewardAnimation.cosmetic && (
                  <p className="text-sm text-white/80">New cosmetic unlocked!</p>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// ============================================================================
// SUB-COMPONENTS
// ============================================================================

interface ChallengeCardProps {
  challenge: ActiveChallenge;
  onSelect: () => void;
  onTogglePin: () => void;
  difficultyColors: Record<string, string>;
  getTypeIcon: (type: ChallengeType) => React.ReactNode;
}

const ChallengeCard: React.FC<ChallengeCardProps> = ({
  challenge,
  onSelect,
  onTogglePin,
  difficultyColors,
  getTypeIcon,
}) => (
  <div className="p-4 rounded-xl border border-gray-800 bg-gray-900/50 hover:border-gray-700 transition-all">
    <div className="flex items-start gap-3">
      <span className="text-2xl">{challenge.icon}</span>
      <div className="flex-1 min-w-0">
        <div className="flex items-start justify-between gap-2">
          <h4 className="text-sm font-medium text-white truncate">{challenge.title}</h4>
          <button
            onClick={(e) => { e.stopPropagation(); onTogglePin(); }}
            className="p-1 rounded hover:bg-gray-800 transition-colors"
          >
            {challenge.isPinned ? (
              <PinOff className="w-4 h-4 text-amber-400" />
            ) : (
              <Pin className="w-4 h-4 text-gray-600" />
            )}
          </button>
        </div>
        <p className="text-xs text-gray-500 mt-1 line-clamp-2">{challenge.description}</p>
        <div className="flex items-center gap-2 mt-2">
          <span className="text-gray-600">{getTypeIcon(challenge.type)}</span>
          <span className={`text-xs px-1.5 py-0.5 rounded border ${difficultyColors[challenge.difficulty]}`}>
            {challenge.difficulty}
          </span>
          <span className="text-xs text-purple-300 ml-auto">+{challenge.xpReward} XP</span>
        </div>
      </div>
    </div>
    <button
      onClick={onSelect}
      className="w-full mt-3 py-2 rounded-lg bg-purple-600/20 border border-purple-500/30 text-purple-300 text-sm hover:bg-purple-600/30 transition-colors flex items-center justify-center gap-2"
    >
      View Challenge
      <ChevronRight className="w-4 h-4" />
    </button>
  </div>
);

interface ChallengeModalProps {
  challenge: ActiveChallenge;
  onClose: () => void;
  onComplete: () => void;
  onScheduleBeat?: () => void;
  showBeatOption?: boolean;
  difficultyColors: Record<string, string>;
}

const ChallengeModal: React.FC<ChallengeModalProps> = ({
  challenge,
  onClose,
  onComplete,
  onScheduleBeat,
  showBeatOption,
  difficultyColors,
}) => (
  <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-60 p-4">
    <div className="w-full max-w-md bg-gray-950 border border-gray-800 rounded-2xl overflow-hidden">
      <div className="p-4 bg-gradient-to-r from-purple-900/30 to-cyan-900/30 border-b border-gray-800">
        <div className="flex items-center gap-3">
          <span className="text-4xl">{challenge.icon}</span>
          <div>
            <h3 className="text-lg font-medium text-white">{challenge.title}</h3>
            <p className="text-sm text-gray-400">{challenge.type.replace('_', ' ')}</p>
          </div>
        </div>
      </div>

      <div className="p-4 space-y-4">
        <p className="text-gray-300">{challenge.description}</p>

        <div className="grid grid-cols-2 gap-3 text-center">
          <div className="p-2 bg-gray-900/50 rounded-lg">
            <span className={`text-xs px-2 py-0.5 rounded border ${difficultyColors[challenge.difficulty]}`}>
              {challenge.difficulty}
            </span>
          </div>
          <div className="p-2 bg-gray-900/50 rounded-lg">
            <div className="flex items-center justify-center gap-1">
              <Sparkles className="w-3 h-3 text-purple-400" />
              <span className="text-sm text-purple-300">+{challenge.xpReward} XP</span>
            </div>
          </div>
        </div>

        <div className="p-3 bg-amber-900/20 rounded-lg border border-amber-500/20">
          <div className="text-xs text-amber-400 uppercase tracking-wider mb-1">Reward</div>
          <div className="text-sm text-gray-300 flex items-center gap-2">
            <Gift className="w-4 h-4 text-amber-400" />
            New cosmetic unlock on completion
          </div>
        </div>

        {/* Beat scheduling option */}
        {showBeatOption && onScheduleBeat && (
          <button
            onClick={onScheduleBeat}
            className="w-full px-4 py-3 rounded-lg bg-cyan-900/30 border border-cyan-500/30 text-cyan-300 text-sm hover:bg-cyan-900/50 transition-colors flex items-center justify-center gap-2"
          >
            <Calendar className="w-4 h-4" />
            <span>Add to Calendar & Complete Later</span>
            <Plus className="w-3 h-3" />
          </button>
        )}

        <div className="flex gap-3 pt-2">
          <button
            onClick={onClose}
            className="flex-1 px-4 py-2 rounded-lg bg-gray-800 hover:bg-gray-700 text-sm"
          >
            Cancel
          </button>
          <button
            onClick={onComplete}
            className="flex-1 px-4 py-2 rounded-lg bg-gradient-to-r from-emerald-600 to-cyan-600 hover:from-emerald-500 hover:to-cyan-500 text-white text-sm font-medium flex items-center justify-center gap-2"
          >
            <CheckCircle2 className="w-4 h-4" />
            {showBeatOption ? 'Done Now' : 'Complete'}
          </button>
        </div>
      </div>
    </div>
  </div>
);

interface BrainTaskModalProps {
  region: BrainRegion;
  task: typeof BRAIN_TASKS.cortical[0];
  onClose: () => void;
  onComplete: () => void;
  onScheduleBeat?: () => void;
  showBeatOption?: boolean;
}

const BrainTaskModal: React.FC<BrainTaskModalProps> = ({
  region,
  task,
  onClose,
  onComplete,
  onScheduleBeat,
  showBeatOption,
}) => {
  const gradient = BRAIN_REGION_GRADIENTS[region.category];
  const xp = region.category === 'cortical' ? 30 :
             region.category === 'limbic' ? 25 :
             region.category === 'subcortical' ? 20 : 20;

  return (
    <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-60 p-4">
      <div className="w-full max-w-md bg-gray-950 border border-gray-800 rounded-2xl overflow-hidden">
        <div
          className="p-4 border-b border-gray-800"
          style={{
            background: `linear-gradient(135deg, ${gradient.colors[0]}30, ${gradient.colors[1]}30, ${gradient.colors[2]}30)`,
          }}
        >
          <div className="flex items-center gap-3">
            <span className="text-4xl">{region.glyph}</span>
            <div>
              <h3 className="text-lg font-medium text-white">{region.name}</h3>
              <p className="text-sm text-gray-400 capitalize">{region.category} Region</p>
            </div>
          </div>
        </div>

        <div className="p-4 space-y-4">
          <div className="p-3 bg-gray-900/50 rounded-lg">
            <h4 className="font-medium text-white mb-1">{task.title}</h4>
            <p className="text-sm text-gray-400">{task.description}</p>
          </div>

          <div className="grid grid-cols-2 gap-3 text-center">
            <div className="p-2 bg-gray-900/50 rounded-lg">
              <div className="flex items-center justify-center gap-1">
                <Clock className="w-3 h-3 text-gray-400" />
                <span className="text-sm text-gray-300">{task.timeEstimate}</span>
              </div>
            </div>
            <div className="p-2 bg-gray-900/50 rounded-lg">
              <div className="flex items-center justify-center gap-1">
                <Sparkles className="w-3 h-3 text-purple-400" />
                <span className="text-sm text-purple-300">+{xp} XP</span>
              </div>
            </div>
          </div>

          {/* Gradient Preview */}
          <div className="p-3 bg-gray-900/50 rounded-lg">
            <div className="text-xs text-gray-400 uppercase tracking-wider mb-2">Region Gradient</div>
            <div
              className="h-8 rounded-lg"
              style={{
                background: `linear-gradient(90deg, ${gradient.colors.join(', ')})`,
              }}
            />
            <div className="flex justify-between mt-2">
              {gradient.colors.map((color, i) => (
                <span key={i} className="text-xs font-mono text-gray-500">{color}</span>
              ))}
            </div>
          </div>

          {/* Beat scheduling option */}
          {showBeatOption && onScheduleBeat && (
            <button
              onClick={onScheduleBeat}
              className="w-full px-4 py-3 rounded-lg bg-cyan-900/30 border border-cyan-500/30 text-cyan-300 text-sm hover:bg-cyan-900/50 transition-colors flex items-center justify-center gap-2"
            >
              <Calendar className="w-4 h-4" />
              <span>Add to Calendar & Complete Later</span>
              <Plus className="w-3 h-3" />
            </button>
          )}

          <div className="flex gap-3 pt-2">
            <button
              onClick={onClose}
              className="flex-1 px-4 py-2 rounded-lg bg-gray-800 hover:bg-gray-700 text-sm"
            >
              Cancel
            </button>
            <button
              onClick={onComplete}
              className="flex-1 px-4 py-2 rounded-lg text-white text-sm font-medium flex items-center justify-center gap-2"
              style={{
                background: `linear-gradient(90deg, ${gradient.colors[0]}, ${gradient.colors[1]})`,
              }}
            >
              <CheckCircle2 className="w-4 h-4" />
              {showBeatOption ? 'Done Now' : 'Complete Task'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

// Heart icon not in lucide-react import, use inline
function Heart({ className }: { className?: string }) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z" />
    </svg>
  );
}

export default BrainRegionChallenge;
