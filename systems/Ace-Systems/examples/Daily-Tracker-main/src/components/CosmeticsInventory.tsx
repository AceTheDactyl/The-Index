// Cosmetics Inventory - View, preview, and equip unlocked cosmetics
import { useState, useEffect, useMemo } from 'react';
import {
  Palette,
  Sparkles,
  Lock,
  Check,
  Eye,
  Grid3X3,
  List,
  Search,
  X
} from 'lucide-react';
import { challengeRewardService } from '../lib/challengeRewardSystem';
import type { CosmeticType, CosmeticReward } from '../lib/challengeRewardSystem';
import {
  COSMETIC_CATALOG,
  getRarityColor,
  getRarityLabel,
  getTypeIcon,
  getTypeLabel,
  getCosmeticById
} from '../lib/cosmeticDefinitions';

interface CosmeticsInventoryProps {
  onEquipmentChange?: (equipped: Record<CosmeticType, string | null>) => void;
}

const COSMETIC_TYPES: CosmeticType[] = [
  'background',
  'gradient',
  'animation',
  'wallpaper',
  'tab_design',
  'button_style',
  'metrics_theme',
  'journal_design',
  'card_style',
  'accent_color'
];

const TYPE_DESCRIPTIONS: Record<CosmeticType, string> = {
  background: 'Change the overall background color scheme',
  gradient: 'Apply gradient effects to the app background',
  animation: 'Add subtle animations to the interface',
  wallpaper: 'Apply decorative patterns to the background',
  tab_design: 'Customize the appearance of navigation tabs',
  button_style: 'Change how buttons look throughout the app',
  metrics_theme: 'Alter the color scheme for your metrics display',
  journal_design: 'Personalize your journal\'s appearance',
  card_style: 'Modify the style of content cards',
  accent_color: 'Set your primary accent color'
};

export default function CosmeticsInventory({ onEquipmentChange }: CosmeticsInventoryProps) {
  const [selectedType, setSelectedType] = useState<CosmeticType | 'all'>('all');
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [searchQuery, setSearchQuery] = useState('');
  const [showLocked, setShowLocked] = useState(true);
  const [previewCosmetic, setPreviewCosmetic] = useState<CosmeticReward | null>(null);
  const [unlockedIds, setUnlockedIds] = useState<string[]>([]);
  const [equippedCosmetics, setEquippedCosmetics] = useState<Record<CosmeticType, string | null>>({} as Record<CosmeticType, string | null>);

  const stats = challengeRewardService.getStats();

  // Load data on mount
  useEffect(() => {
    const loadData = () => {
      const data = challengeRewardService.getData();
      setUnlockedIds(data.unlockedCosmetics);
      setEquippedCosmetics(data.equippedCosmetics);
    };

    loadData();
    const unsubscribe = challengeRewardService.subscribe(loadData);
    return unsubscribe;
  }, []);

  // Filter cosmetics
  const filteredCosmetics = useMemo(() => {
    return COSMETIC_CATALOG.filter(c => {
      // Type filter
      if (selectedType !== 'all' && c.type !== selectedType) return false;

      // Search filter
      if (searchQuery) {
        const query = searchQuery.toLowerCase();
        if (!c.name.toLowerCase().includes(query) &&
            !c.description.toLowerCase().includes(query) &&
            !c.type.toLowerCase().includes(query)) {
          return false;
        }
      }

      // Locked filter
      if (!showLocked && !unlockedIds.includes(c.id)) return false;

      return true;
    });
  }, [selectedType, searchQuery, showLocked, unlockedIds]);

  // Group by type for sidebar counts
  const typeCounts = useMemo(() => {
    const counts: Record<CosmeticType | 'all', { total: number; unlocked: number }> = {
      all: { total: 0, unlocked: 0 }
    } as Record<CosmeticType | 'all', { total: number; unlocked: number }>;

    COSMETIC_TYPES.forEach(type => {
      counts[type] = { total: 0, unlocked: 0 };
    });

    COSMETIC_CATALOG.forEach(c => {
      counts[c.type].total++;
      counts.all.total++;
      if (unlockedIds.includes(c.id)) {
        counts[c.type].unlocked++;
        counts.all.unlocked++;
      }
    });

    return counts;
  }, [unlockedIds]);

  const handleEquip = (cosmetic: CosmeticReward) => {
    if (!unlockedIds.includes(cosmetic.id)) return;

    const success = challengeRewardService.equipCosmetic(cosmetic.id);
    if (success) {
      const newEquipped = { ...equippedCosmetics, [cosmetic.type]: cosmetic.id };
      setEquippedCosmetics(newEquipped);
      onEquipmentChange?.(newEquipped);
    }
  };

  const handleUnequip = (type: CosmeticType) => {
    challengeRewardService.unequipCosmetic(type);
    const data = challengeRewardService.getData();
    setEquippedCosmetics(data.equippedCosmetics);
    onEquipmentChange?.(data.equippedCosmetics);
  };

  const isEquipped = (cosmetic: CosmeticReward): boolean => {
    return equippedCosmetics[cosmetic.type] === cosmetic.id;
  };

  const renderCosmeticCard = (cosmetic: CosmeticReward) => {
    const isUnlocked = unlockedIds.includes(cosmetic.id);
    const equipped = isEquipped(cosmetic);

    return (
      <div
        key={cosmetic.id}
        className={`relative rounded-xl border transition-all duration-200 overflow-hidden ${
          equipped
            ? 'border-green-500/50 bg-green-900/20 ring-1 ring-green-500/30'
            : isUnlocked
            ? 'border-gray-600/50 bg-gray-800/50 hover:border-gray-500/50 hover:bg-gray-700/50'
            : 'border-gray-700/30 bg-gray-900/30 opacity-60'
        }`}
      >
        {/* Lock overlay for locked cosmetics */}
        {!isUnlocked && (
          <div className="absolute inset-0 bg-black/50 flex items-center justify-center z-10">
            <div className="text-center">
              <Lock className="w-6 h-6 text-gray-500 mx-auto mb-1" />
              <span className="text-xs text-gray-500">Complete challenges to unlock</span>
            </div>
          </div>
        )}

        {/* Equipped badge */}
        {equipped && (
          <div className="absolute top-2 right-2 bg-green-500 rounded-full p-1 z-20">
            <Check className="w-3 h-3 text-white" />
          </div>
        )}

        {/* Content */}
        <div className={`p-4 ${viewMode === 'list' ? 'flex items-center gap-4' : ''}`}>
          {/* Preview */}
          <div className={`${viewMode === 'list' ? 'flex-shrink-0' : 'mb-3 text-center'}`}>
            <span className={`${viewMode === 'grid' ? 'text-4xl' : 'text-3xl'}`}>
              {cosmetic.preview}
            </span>
          </div>

          {/* Info */}
          <div className={`${viewMode === 'list' ? 'flex-grow min-w-0' : ''}`}>
            <h3 className={`font-medium text-white ${viewMode === 'list' ? 'truncate' : 'text-center mb-1'}`}>
              {cosmetic.name}
            </h3>

            {viewMode === 'grid' && (
              <p className="text-xs text-gray-400 text-center line-clamp-2 mb-2">
                {cosmetic.description}
              </p>
            )}

            <div className={`flex items-center gap-2 ${viewMode === 'grid' ? 'justify-center' : ''}`}>
              <span
                className="text-xs px-2 py-0.5 rounded-full"
                style={{
                  backgroundColor: `${getRarityColor(cosmetic.rarity)}20`,
                  color: getRarityColor(cosmetic.rarity)
                }}
              >
                {getRarityLabel(cosmetic.rarity)}
              </span>
              {viewMode === 'list' && (
                <span className="text-xs text-gray-500">
                  {getTypeIcon(cosmetic.type)} {getTypeLabel(cosmetic.type)}
                </span>
              )}
            </div>

            {viewMode === 'list' && (
              <p className="text-xs text-gray-400 mt-1 truncate">
                {cosmetic.description}
              </p>
            )}
          </div>

          {/* Actions */}
          {isUnlocked && (
            <div className={`${viewMode === 'list' ? 'flex-shrink-0 flex gap-2' : 'mt-3 flex gap-2 justify-center'}`}>
              <button
                onClick={() => setPreviewCosmetic(cosmetic)}
                className="p-2 bg-gray-700/50 rounded-lg hover:bg-gray-600/50 transition-colors"
                title="Preview"
              >
                <Eye className="w-4 h-4 text-gray-400" />
              </button>
              {equipped ? (
                <button
                  onClick={() => handleUnequip(cosmetic.type)}
                  className="px-3 py-1.5 bg-gray-700/50 text-gray-300 rounded-lg hover:bg-gray-600/50 transition-colors text-sm"
                >
                  Unequip
                </button>
              ) : (
                <button
                  onClick={() => handleEquip(cosmetic)}
                  className="px-3 py-1.5 bg-blue-600/30 text-blue-400 rounded-lg hover:bg-blue-600/40 transition-colors text-sm"
                >
                  Equip
                </button>
              )}
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="flex gap-6 h-full">
      {/* Sidebar - Type Categories */}
      <div className="w-64 flex-shrink-0 space-y-2">
        <div className="bg-gray-800/50 rounded-xl p-4 mb-4">
          <div className="flex items-center gap-2 mb-3">
            <Palette className="w-5 h-5 text-purple-400" />
            <h2 className="font-semibold text-white">Cosmetics</h2>
          </div>
          <div className="grid grid-cols-2 gap-2 text-center">
            <div className="bg-gray-700/50 rounded-lg p-2">
              <div className="text-lg font-bold text-purple-400">{stats.unlockedCount}</div>
              <div className="text-xs text-gray-400">Unlocked</div>
            </div>
            <div className="bg-gray-700/50 rounded-lg p-2">
              <div className="text-lg font-bold text-gray-400">{stats.totalCosmetics}</div>
              <div className="text-xs text-gray-400">Total</div>
            </div>
          </div>
        </div>

        {/* Category list */}
        <button
          onClick={() => setSelectedType('all')}
          className={`w-full flex items-center justify-between px-3 py-2 rounded-lg transition-colors ${
            selectedType === 'all'
              ? 'bg-purple-600/30 text-purple-300 border border-purple-500/30'
              : 'bg-gray-800/50 text-gray-400 hover:bg-gray-700/50'
          }`}
        >
          <div className="flex items-center gap-2">
            <Grid3X3 className="w-4 h-4" />
            <span>All Cosmetics</span>
          </div>
          <span className="text-xs">{typeCounts.all.unlocked}/{typeCounts.all.total}</span>
        </button>

        {COSMETIC_TYPES.map(type => (
          <button
            key={type}
            onClick={() => setSelectedType(type)}
            className={`w-full flex items-center justify-between px-3 py-2 rounded-lg transition-colors ${
              selectedType === type
                ? 'bg-purple-600/30 text-purple-300 border border-purple-500/30'
                : 'bg-gray-800/50 text-gray-400 hover:bg-gray-700/50'
            }`}
          >
            <div className="flex items-center gap-2">
              <span>{getTypeIcon(type)}</span>
              <span className="text-sm">{getTypeLabel(type)}</span>
            </div>
            <span className="text-xs">{typeCounts[type].unlocked}/{typeCounts[type].total}</span>
          </button>
        ))}
      </div>

      {/* Main content */}
      <div className="flex-grow min-w-0">
        {/* Header & Controls */}
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-xl font-bold text-white">
              {selectedType === 'all' ? 'All Cosmetics' : getTypeLabel(selectedType)}
            </h2>
            {selectedType !== 'all' && (
              <p className="text-sm text-gray-400">{TYPE_DESCRIPTIONS[selectedType]}</p>
            )}
          </div>

          <div className="flex items-center gap-3">
            {/* Search */}
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500" />
              <input
                type="text"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search cosmetics..."
                className="pl-9 pr-4 py-2 bg-gray-800/50 border border-gray-700/50 rounded-lg text-sm text-white placeholder-gray-500 focus:outline-none focus:border-purple-500/50 w-48"
              />
              {searchQuery && (
                <button
                  onClick={() => setSearchQuery('')}
                  className="absolute right-2 top-1/2 -translate-y-1/2"
                >
                  <X className="w-4 h-4 text-gray-500 hover:text-gray-300" />
                </button>
              )}
            </div>

            {/* Show locked toggle */}
            <button
              onClick={() => setShowLocked(!showLocked)}
              className={`flex items-center gap-2 px-3 py-2 rounded-lg transition-colors ${
                showLocked
                  ? 'bg-gray-700/50 text-gray-300'
                  : 'bg-purple-600/30 text-purple-300'
              }`}
            >
              <Lock className="w-4 h-4" />
              <span className="text-sm">{showLocked ? 'Hide Locked' : 'Show Locked'}</span>
            </button>

            {/* View mode toggle */}
            <div className="flex items-center bg-gray-800/50 rounded-lg p-1">
              <button
                onClick={() => setViewMode('grid')}
                className={`p-1.5 rounded ${viewMode === 'grid' ? 'bg-gray-700 text-white' : 'text-gray-500'}`}
              >
                <Grid3X3 className="w-4 h-4" />
              </button>
              <button
                onClick={() => setViewMode('list')}
                className={`p-1.5 rounded ${viewMode === 'list' ? 'bg-gray-700 text-white' : 'text-gray-500'}`}
              >
                <List className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>

        {/* Currently equipped section */}
        {selectedType !== 'all' && equippedCosmetics[selectedType] && (
          <div className="bg-gradient-to-r from-green-900/30 to-emerald-900/30 border border-green-500/30 rounded-xl p-4 mb-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <Check className="w-5 h-5 text-green-400" />
                <span className="text-green-300 font-medium">Currently Equipped:</span>
                {(() => {
                  const equipped = getCosmeticById(equippedCosmetics[selectedType]!);
                  return equipped ? (
                    <div className="flex items-center gap-2">
                      <span className="text-xl">{equipped.preview}</span>
                      <span className="text-white">{equipped.name}</span>
                    </div>
                  ) : null;
                })()}
              </div>
              <button
                onClick={() => handleUnequip(selectedType)}
                className="text-sm text-gray-400 hover:text-gray-300"
              >
                Reset to default
              </button>
            </div>
          </div>
        )}

        {/* Cosmetics grid/list */}
        <div className={`${viewMode === 'grid' ? 'grid grid-cols-3 gap-4' : 'space-y-3'}`}>
          {filteredCosmetics.map(renderCosmeticCard)}
        </div>

        {/* Empty state */}
        {filteredCosmetics.length === 0 && (
          <div className="text-center py-12">
            <Sparkles className="w-12 h-12 text-gray-600 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-400 mb-2">No Cosmetics Found</h3>
            <p className="text-sm text-gray-500">
              {searchQuery
                ? 'Try a different search term'
                : showLocked
                ? 'Complete challenges to unlock more cosmetics!'
                : 'Toggle "Show Locked" to see all available cosmetics'}
            </p>
          </div>
        )}
      </div>

      {/* Preview modal */}
      {previewCosmetic && (
        <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-50" onClick={() => setPreviewCosmetic(null)}>
          <div
            className="bg-gray-900 border border-gray-700 rounded-2xl p-6 max-w-md w-full mx-4"
            onClick={e => e.stopPropagation()}
          >
            <div className="text-center mb-6">
              <span className="text-6xl mb-4 block">{previewCosmetic.preview}</span>
              <h3 className="text-xl font-bold text-white mb-2">{previewCosmetic.name}</h3>
              <span
                className="inline-block px-3 py-1 rounded-full text-sm"
                style={{
                  backgroundColor: `${getRarityColor(previewCosmetic.rarity)}20`,
                  color: getRarityColor(previewCosmetic.rarity)
                }}
              >
                {getRarityLabel(previewCosmetic.rarity)}
              </span>
            </div>

            <p className="text-gray-300 text-center mb-6">{previewCosmetic.description}</p>

            <div className="bg-gray-800/50 rounded-lg p-4 mb-6">
              <h4 className="text-sm font-medium text-gray-400 mb-2">Type</h4>
              <div className="flex items-center gap-2">
                <span>{getTypeIcon(previewCosmetic.type)}</span>
                <span className="text-white">{getTypeLabel(previewCosmetic.type)}</span>
              </div>
              <p className="text-xs text-gray-500 mt-1">{TYPE_DESCRIPTIONS[previewCosmetic.type]}</p>
            </div>

            {/* CSS Variables preview */}
            {previewCosmetic.cssVars && Object.keys(previewCosmetic.cssVars).length > 0 && (
              <div className="bg-gray-800/50 rounded-lg p-4 mb-6">
                <h4 className="text-sm font-medium text-gray-400 mb-2">Style Preview</h4>
                <div className="flex flex-wrap gap-2">
                  {Object.entries(previewCosmetic.cssVars).slice(0, 3).map(([key, value]) => (
                    <div
                      key={key}
                      className="w-8 h-8 rounded-lg border border-gray-600"
                      style={{ background: value }}
                      title={`${key}: ${value}`}
                    />
                  ))}
                </div>
              </div>
            )}

            <div className="flex gap-3">
              <button
                onClick={() => setPreviewCosmetic(null)}
                className="flex-1 py-2 bg-gray-700 text-gray-300 rounded-lg hover:bg-gray-600 transition-colors"
              >
                Close
              </button>
              {unlockedIds.includes(previewCosmetic.id) && !isEquipped(previewCosmetic) && (
                <button
                  onClick={() => {
                    handleEquip(previewCosmetic);
                    setPreviewCosmetic(null);
                  }}
                  className="flex-1 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-500 transition-colors"
                >
                  Equip Now
                </button>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
