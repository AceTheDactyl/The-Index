// Challenge Hub - Daily challenges with pinning and smart highlighting
import { useState, useEffect, useMemo } from 'react';
import {
  Target,
  Pin,
  PinOff,
  Sparkles,
  Trophy,
  Clock,
  CheckCircle2,
  ChevronDown,
  ChevronUp,
  Filter,
  RefreshCw,
  Star,
  Zap,
  Gift,
  TrendingUp,
  AlertCircle,
  Award
} from 'lucide-react';
import {
  challengeRewardService,
  calculateChallengeProgress
} from '../lib/challengeRewardSystem';
import type {
  DailyChallenge,
  ChallengeCategory,
  ChallengeDifficulty,
  UserMetrics,
  CosmeticReward
} from '../lib/challengeRewardSystem';
import {
  getRarityColor,
  getRarityLabel,
  getCosmeticById,
  getTypeIcon
} from '../lib/cosmeticDefinitions';

interface ChallengeHubProps {
  userMetrics: UserMetrics;
  onCosmeticUnlocked?: (cosmetic: CosmeticReward) => void;
  checkIns: Array<{ category: string; done: boolean }>;
  journalCount: number;
}

const CATEGORY_COLORS: Record<ChallengeCategory, { bg: string; text: string; border: string }> = {
  wellness: { bg: 'bg-green-900/30', text: 'text-green-400', border: 'border-green-500/30' },
  productivity: { bg: 'bg-blue-900/30', text: 'text-blue-400', border: 'border-blue-500/30' },
  mindfulness: { bg: 'bg-purple-900/30', text: 'text-purple-400', border: 'border-purple-500/30' },
  social: { bg: 'bg-orange-900/30', text: 'text-orange-400', border: 'border-orange-500/30' },
  creativity: { bg: 'bg-pink-900/30', text: 'text-pink-400', border: 'border-pink-500/30' },
  physical: { bg: 'bg-red-900/30', text: 'text-red-400', border: 'border-red-500/30' },
  learning: { bg: 'bg-cyan-900/30', text: 'text-cyan-400', border: 'border-cyan-500/30' },
};

const DIFFICULTY_COLORS: Record<ChallengeDifficulty, { bg: string; text: string }> = {
  easy: { bg: 'bg-green-500/20', text: 'text-green-400' },
  medium: { bg: 'bg-yellow-500/20', text: 'text-yellow-400' },
  hard: { bg: 'bg-orange-500/20', text: 'text-orange-400' },
  legendary: { bg: 'bg-purple-500/20', text: 'text-purple-400' },
};

export default function ChallengeHub({
  userMetrics,
  onCosmeticUnlocked,
  checkIns,
  journalCount
}: ChallengeHubProps) {
  const [challenges, setChallenges] = useState<DailyChallenge[]>([]);
  const [expandedChallenge, setExpandedChallenge] = useState<string | null>(null);
  const [filterCategory, setFilterCategory] = useState<ChallengeCategory | 'all'>('all');
  const [filterDifficulty, setFilterDifficulty] = useState<ChallengeDifficulty | 'all'>('all');
  const [showFilters, setShowFilters] = useState(false);
  const [recentUnlock, setRecentUnlock] = useState<CosmeticReward | null>(null);

  const stats = challengeRewardService.getStats();

  // Load challenges on mount and subscribe to updates
  useEffect(() => {
    const loadChallenges = () => {
      const dailyChallenges = challengeRewardService.getDailyChallenges();
      if (dailyChallenges.length === 0) {
        challengeRewardService.generateDailyChallenges(userMetrics);
      }
      setChallenges(challengeRewardService.getDailyChallenges());
    };

    loadChallenges();

    const unsubscribe = challengeRewardService.subscribe(() => {
      setChallenges([...challengeRewardService.getDailyChallenges()]);
    });

    return unsubscribe;
  }, []);

  // Update challenge progress when metrics/checkins change
  useEffect(() => {
    challenges.forEach(challenge => {
      if (!challenge.completed) {
        const progress = calculateChallengeProgress(
          challenge,
          checkIns,
          journalCount,
          userMetrics
        );
        if (progress !== challenge.progress) {
          challengeRewardService.updateChallengeProgress(challenge.id, progress);
        }
      }
    });
  }, [checkIns, journalCount, userMetrics, challenges]);

  // Filter challenges
  const filteredChallenges = useMemo(() => {
    return challenges.filter(c => {
      if (filterCategory !== 'all' && c.category !== filterCategory) return false;
      if (filterDifficulty !== 'all' && c.difficulty !== filterDifficulty) return false;
      return true;
    });
  }, [challenges, filterCategory, filterDifficulty]);

  // Separate pinned and highlighted
  const pinnedChallenges = filteredChallenges.filter(c => c.isPinned && !c.completed);
  const highlightedChallenges = filteredChallenges.filter(c => c.isHighlighted && !c.isPinned && !c.completed);
  const otherChallenges = filteredChallenges.filter(c => !c.isPinned && !c.isHighlighted && !c.completed);
  const completedChallenges = filteredChallenges.filter(c => c.completed);

  const handlePinToggle = (challengeId: string) => {
    const challenge = challenges.find(c => c.id === challengeId);
    if (challenge?.isPinned) {
      challengeRewardService.unpinChallenge(challengeId);
    } else {
      challengeRewardService.pinChallenge(challengeId);
    }
  };

  const handleCompleteChallenge = (challengeId: string) => {
    const cosmetic = challengeRewardService.completeChallenge(challengeId);
    if (cosmetic) {
      setRecentUnlock(cosmetic);
      onCosmeticUnlocked?.(cosmetic);
      setTimeout(() => setRecentUnlock(null), 5000);
    }
  };

  const refreshChallenges = () => {
    challengeRewardService.generateDailyChallenges(userMetrics);
    setChallenges(challengeRewardService.getDailyChallenges());
  };

  const renderChallengeCard = (challenge: DailyChallenge, showHighlight = false) => {
    const isExpanded = expandedChallenge === challenge.id;
    const categoryStyle = CATEGORY_COLORS[challenge.category];
    const difficultyStyle = DIFFICULTY_COLORS[challenge.difficulty];
    const cosmetic = getCosmeticById(challenge.cosmetic_reward_id);

    return (
      <div
        key={challenge.id}
        className={`relative rounded-xl border transition-all duration-300 ${
          challenge.completed
            ? 'bg-gray-800/30 border-gray-700/30 opacity-75'
            : challenge.isPinned
            ? 'bg-yellow-900/20 border-yellow-500/40 ring-1 ring-yellow-500/20'
            : showHighlight && challenge.isHighlighted
            ? 'bg-blue-900/20 border-blue-500/40 ring-1 ring-blue-500/20'
            : `${categoryStyle.bg} ${categoryStyle.border}`
        }`}
      >
        {/* Highlight indicator */}
        {showHighlight && challenge.isHighlighted && !challenge.completed && (
          <div className="absolute -top-2 -right-2 bg-blue-500 rounded-full p-1 animate-pulse">
            <Sparkles className="w-3 h-3 text-white" />
          </div>
        )}

        {/* Pin indicator */}
        {challenge.isPinned && !challenge.completed && (
          <div className="absolute -top-2 -left-2 bg-yellow-500 rounded-full p-1">
            <Pin className="w-3 h-3 text-black" />
          </div>
        )}

        {/* Main content */}
        <div
          className="p-4 cursor-pointer"
          onClick={() => setExpandedChallenge(isExpanded ? null : challenge.id)}
        >
          {/* Header */}
          <div className="flex items-start justify-between mb-2">
            <div className="flex items-center gap-2">
              <span className="text-2xl">{challenge.icon}</span>
              <div>
                <h3 className={`font-semibold ${challenge.completed ? 'text-gray-400 line-through' : 'text-white'}`}>
                  {challenge.title}
                </h3>
                <div className="flex items-center gap-2 mt-1">
                  <span className={`text-xs px-2 py-0.5 rounded-full ${categoryStyle.bg} ${categoryStyle.text}`}>
                    {challenge.category}
                  </span>
                  <span className={`text-xs px-2 py-0.5 rounded-full ${difficultyStyle.bg} ${difficultyStyle.text}`}>
                    {challenge.difficulty}
                  </span>
                </div>
              </div>
            </div>
            <div className="flex items-center gap-2">
              {!challenge.completed && (
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    handlePinToggle(challenge.id);
                  }}
                  className={`p-1.5 rounded-lg transition-colors ${
                    challenge.isPinned
                      ? 'bg-yellow-500/20 text-yellow-400 hover:bg-yellow-500/30'
                      : 'bg-gray-700/50 text-gray-400 hover:bg-gray-600/50'
                  }`}
                  title={challenge.isPinned ? 'Unpin challenge' : 'Pin challenge'}
                >
                  {challenge.isPinned ? <PinOff className="w-4 h-4" /> : <Pin className="w-4 h-4" />}
                </button>
              )}
              {isExpanded ? (
                <ChevronUp className="w-5 h-5 text-gray-400" />
              ) : (
                <ChevronDown className="w-5 h-5 text-gray-400" />
              )}
            </div>
          </div>

          {/* Progress bar */}
          <div className="mt-3">
            <div className="flex items-center justify-between text-xs mb-1">
              <span className="text-gray-400">Progress</span>
              <span className={challenge.completed ? 'text-green-400' : 'text-gray-300'}>
                {Math.round(challenge.progress)}%
              </span>
            </div>
            <div className="h-2 bg-gray-700/50 rounded-full overflow-hidden">
              <div
                className={`h-full transition-all duration-500 ${
                  challenge.completed
                    ? 'bg-green-500'
                    : challenge.progress >= 75
                    ? 'bg-yellow-500'
                    : 'bg-blue-500'
                }`}
                style={{ width: `${challenge.progress}%` }}
              />
            </div>
          </div>

          {/* Highlight reason */}
          {challenge.isHighlighted && challenge.highlightReason && !challenge.completed && (
            <div className="mt-2 flex items-center gap-1.5 text-xs text-blue-400">
              <AlertCircle className="w-3 h-3" />
              {challenge.highlightReason}
            </div>
          )}
        </div>

        {/* Expanded content */}
        {isExpanded && (
          <div className="px-4 pb-4 border-t border-gray-700/30 pt-3 space-y-3">
            <p className="text-sm text-gray-300">{challenge.description}</p>

            {/* Details */}
            <div className="grid grid-cols-2 gap-3 text-sm">
              <div className="flex items-center gap-2 text-gray-400">
                <Clock className="w-4 h-4" />
                <span>{challenge.time_estimate}</span>
              </div>
              <div className="flex items-center gap-2 text-yellow-400">
                <Zap className="w-4 h-4" />
                <span>{challenge.xp_reward} XP</span>
              </div>
            </div>

            {/* Reward preview */}
            {cosmetic && (
              <div className="bg-gray-800/50 rounded-lg p-3">
                <div className="flex items-center gap-2 mb-2">
                  <Gift className="w-4 h-4 text-purple-400" />
                  <span className="text-sm font-medium text-purple-300">Reward Preview</span>
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-2xl">{cosmetic.preview}</span>
                  <div>
                    <p className="text-sm text-white">{cosmetic.name}</p>
                    <div className="flex items-center gap-2 text-xs">
                      <span style={{ color: getRarityColor(cosmetic.rarity) }}>
                        {getRarityLabel(cosmetic.rarity)}
                      </span>
                      <span className="text-gray-500">•</span>
                      <span className="text-gray-400">
                        {getTypeIcon(cosmetic.type)} {cosmetic.type.split('_').join(' ')}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Tips */}
            {challenge.tips && challenge.tips.length > 0 && (
              <div className="text-xs text-gray-400">
                <p className="font-medium mb-1">Tips:</p>
                <ul className="list-disc list-inside space-y-0.5">
                  {challenge.tips.map((tip, i) => (
                    <li key={i}>{tip}</li>
                  ))}
                </ul>
              </div>
            )}

            {/* Complete button (for testing/manual completion) */}
            {!challenge.completed && challenge.progress >= 100 && (
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  handleCompleteChallenge(challenge.id);
                }}
                className="w-full py-2 bg-green-600/30 border border-green-500/50 text-green-400 rounded-lg hover:bg-green-600/40 transition-colors flex items-center justify-center gap-2"
              >
                <CheckCircle2 className="w-4 h-4" />
                Claim Reward
              </button>
            )}

            {challenge.completed && (
              <div className="flex items-center justify-center gap-2 text-green-400 py-2">
                <CheckCircle2 className="w-5 h-5" />
                <span className="font-medium">Completed!</span>
              </div>
            )}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="space-y-6">
      {/* Recent unlock notification */}
      {recentUnlock && (
        <div className="fixed top-4 right-4 z-50 bg-gradient-to-r from-purple-900/90 to-pink-900/90 border border-purple-500/50 rounded-xl p-4 shadow-2xl animate-slide-in-right">
          <div className="flex items-center gap-3">
            <div className="text-3xl">{recentUnlock.preview}</div>
            <div>
              <p className="text-xs text-purple-300 mb-1">New Cosmetic Unlocked!</p>
              <p className="font-semibold text-white">{recentUnlock.name}</p>
              <p className="text-xs" style={{ color: getRarityColor(recentUnlock.rarity) }}>
                {getRarityLabel(recentUnlock.rarity)} {recentUnlock.type.split('_').join(' ')}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Header with stats */}
      <div className="bg-gradient-to-r from-gray-900/80 to-gray-800/80 rounded-xl border border-gray-700/50 p-4">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-purple-500/20 rounded-lg">
              <Target className="w-6 h-6 text-purple-400" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-white">Daily Challenges</h2>
              <p className="text-sm text-gray-400">Complete challenges to unlock cosmetics</p>
            </div>
          </div>
          <button
            onClick={refreshChallenges}
            className="p-2 bg-gray-700/50 rounded-lg hover:bg-gray-600/50 transition-colors"
            title="Refresh challenges"
          >
            <RefreshCw className="w-5 h-5 text-gray-400" />
          </button>
        </div>

        {/* Stats row */}
        <div className="grid grid-cols-4 gap-3">
          <div className="bg-gray-800/50 rounded-lg p-3 text-center">
            <div className="flex items-center justify-center gap-1 text-yellow-400 mb-1">
              <Zap className="w-4 h-4" />
              <span className="font-bold">{stats.totalXP}</span>
            </div>
            <p className="text-xs text-gray-400">Total XP</p>
          </div>
          <div className="bg-gray-800/50 rounded-lg p-3 text-center">
            <div className="flex items-center justify-center gap-1 text-blue-400 mb-1">
              <Star className="w-4 h-4" />
              <span className="font-bold">Lvl {stats.level}</span>
            </div>
            <p className="text-xs text-gray-400">{stats.xpToNextLevel} to next</p>
          </div>
          <div className="bg-gray-800/50 rounded-lg p-3 text-center">
            <div className="flex items-center justify-center gap-1 text-orange-400 mb-1">
              <TrendingUp className="w-4 h-4" />
              <span className="font-bold">{stats.currentStreak}</span>
            </div>
            <p className="text-xs text-gray-400">Day Streak</p>
          </div>
          <div className="bg-gray-800/50 rounded-lg p-3 text-center">
            <div className="flex items-center justify-center gap-1 text-purple-400 mb-1">
              <Award className="w-4 h-4" />
              <span className="font-bold">{stats.unlockedCount}/{stats.totalCosmetics}</span>
            </div>
            <p className="text-xs text-gray-400">Cosmetics</p>
          </div>
        </div>

        {/* Level progress */}
        <div className="mt-3">
          <div className="h-1.5 bg-gray-700/50 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-blue-500 to-purple-500 transition-all duration-500"
              style={{ width: `${((100 - stats.xpToNextLevel) / 100) * 100}%` }}
            />
          </div>
        </div>
      </div>

      {/* Filters */}
      <div className="flex items-center justify-between">
        <button
          onClick={() => setShowFilters(!showFilters)}
          className={`flex items-center gap-2 px-3 py-1.5 rounded-lg transition-colors ${
            showFilters ? 'bg-blue-500/20 text-blue-400' : 'bg-gray-800/50 text-gray-400 hover:bg-gray-700/50'
          }`}
        >
          <Filter className="w-4 h-4" />
          Filters
        </button>
        <div className="text-sm text-gray-400">
          {filteredChallenges.filter(c => !c.completed).length} active •{' '}
          {completedChallenges.length} completed today
        </div>
      </div>

      {showFilters && (
        <div className="bg-gray-800/50 rounded-lg p-4 flex flex-wrap gap-4">
          <div>
            <label className="text-xs text-gray-400 mb-2 block">Category</label>
            <select
              value={filterCategory}
              onChange={(e) => setFilterCategory(e.target.value as ChallengeCategory | 'all')}
              className="bg-gray-700 text-white text-sm rounded-lg px-3 py-1.5 border border-gray-600"
            >
              <option value="all">All Categories</option>
              {Object.keys(CATEGORY_COLORS).map(cat => (
                <option key={cat} value={cat}>{cat.charAt(0).toUpperCase() + cat.slice(1)}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="text-xs text-gray-400 mb-2 block">Difficulty</label>
            <select
              value={filterDifficulty}
              onChange={(e) => setFilterDifficulty(e.target.value as ChallengeDifficulty | 'all')}
              className="bg-gray-700 text-white text-sm rounded-lg px-3 py-1.5 border border-gray-600"
            >
              <option value="all">All Difficulties</option>
              <option value="easy">Easy</option>
              <option value="medium">Medium</option>
              <option value="hard">Hard</option>
              <option value="legendary">Legendary</option>
            </select>
          </div>
        </div>
      )}

      {/* Pinned challenges */}
      {pinnedChallenges.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center gap-2 text-yellow-400">
            <Pin className="w-4 h-4" />
            <h3 className="font-medium">Pinned Challenges</h3>
          </div>
          <div className="grid gap-3">
            {pinnedChallenges.map(c => renderChallengeCard(c))}
          </div>
        </div>
      )}

      {/* Highlighted challenges */}
      {highlightedChallenges.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center gap-2 text-blue-400">
            <Sparkles className="w-4 h-4" />
            <h3 className="font-medium">Recommended for You</h3>
          </div>
          <div className="grid gap-3">
            {highlightedChallenges.map(c => renderChallengeCard(c, true))}
          </div>
        </div>
      )}

      {/* Other active challenges */}
      {otherChallenges.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center gap-2 text-gray-300">
            <Target className="w-4 h-4" />
            <h3 className="font-medium">Available Challenges</h3>
          </div>
          <div className="grid gap-3">
            {otherChallenges.map(c => renderChallengeCard(c, true))}
          </div>
        </div>
      )}

      {/* Completed challenges */}
      {completedChallenges.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center gap-2 text-green-400">
            <Trophy className="w-4 h-4" />
            <h3 className="font-medium">Completed Today</h3>
          </div>
          <div className="grid gap-3">
            {completedChallenges.map(c => renderChallengeCard(c))}
          </div>
        </div>
      )}

      {/* Empty state */}
      {filteredChallenges.length === 0 && (
        <div className="text-center py-12">
          <Target className="w-12 h-12 text-gray-600 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-400 mb-2">No Challenges Found</h3>
          <p className="text-sm text-gray-500">
            Try adjusting your filters or refresh to get new challenges.
          </p>
        </div>
      )}
    </div>
  );
}
