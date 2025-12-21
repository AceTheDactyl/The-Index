// Cosmetic Definitions - All unlockable aesthetic rewards
// These are purely visual enhancements that don't affect functionality

import type { CosmeticReward, CosmeticType } from './challengeRewardSystem';

// ============================================================================
// CSS VARIABLE MAPPINGS FOR THEMING
// ============================================================================

export const CSS_VAR_DEFAULTS = {
  // Background
  '--bg-primary': '#000000',
  '--bg-secondary': '#111827',
  '--bg-tertiary': '#1e3a5f',

  // Gradients
  '--gradient-start': '#000000',
  '--gradient-mid': '#111827',
  '--gradient-end': '#1e3a5f',
  '--gradient-angle': '180deg',

  // Cards
  '--card-bg': 'rgba(3, 7, 18, 0.6)',
  '--card-border': 'rgba(255, 255, 255, 0.1)',
  '--card-shadow': '0 4px 6px rgba(0, 0, 0, 0.3)',

  // Buttons
  '--button-bg': 'rgba(59, 130, 246, 0.2)',
  '--button-border': 'rgba(59, 130, 246, 0.5)',
  '--button-text': '#60a5fa',
  '--button-hover-bg': 'rgba(59, 130, 246, 0.3)',

  // Tabs
  '--tab-bg': 'transparent',
  '--tab-active-bg': 'rgba(59, 130, 246, 0.2)',
  '--tab-border': 'rgba(59, 130, 246, 0.5)',
  '--tab-text': '#9ca3af',
  '--tab-active-text': '#60a5fa',

  // Metrics
  '--metric-high': '#22c55e',
  '--metric-medium': '#eab308',
  '--metric-low': '#ef4444',
  '--metric-bg': 'rgba(0, 0, 0, 0.3)',

  // Journal
  '--journal-bg': 'rgba(3, 7, 18, 0.8)',
  '--journal-border': 'rgba(147, 51, 234, 0.3)',
  '--journal-text': '#e5e7eb',
  '--journal-accent': '#a855f7',

  // Accents
  '--accent-primary': '#3b82f6',
  '--accent-secondary': '#8b5cf6',
  '--accent-glow': 'rgba(59, 130, 246, 0.5)',
};

// ============================================================================
// COSMETIC CATALOG - All available cosmetics
// ============================================================================

export const COSMETIC_CATALOG: CosmeticReward[] = [
  // ---------------------------------------------------------------------------
  // DEFAULT COSMETICS (Unlocked by default)
  // ---------------------------------------------------------------------------
  {
    id: 'bg_default',
    name: 'Classic Dark',
    description: 'The default dark theme background',
    type: 'background',
    rarity: 'common',
    cssVars: {
      '--bg-primary': '#000000',
      '--bg-secondary': '#111827',
      '--bg-tertiary': '#1e3a5f',
    },
    preview: '🌑'
  },
  {
    id: 'gradient_default',
    name: 'Midnight Blue',
    description: 'Classic dark blue gradient',
    type: 'gradient',
    rarity: 'common',
    cssClass: 'gradient-default',
    cssVars: {
      '--gradient-start': '#000000',
      '--gradient-mid': '#111827',
      '--gradient-end': '#1e3a5f',
      '--gradient-angle': '180deg',
    },
    preview: '🌌'
  },
  {
    id: 'tab_default',
    name: 'Clean Tabs',
    description: 'Simple, clean tab design',
    type: 'tab_design',
    rarity: 'common',
    cssVars: {
      '--tab-bg': 'transparent',
      '--tab-active-bg': 'rgba(59, 130, 246, 0.2)',
      '--tab-border': 'rgba(59, 130, 246, 0.5)',
    },
    preview: '📑'
  },
  {
    id: 'button_default',
    name: 'Standard Buttons',
    description: 'Default button styling',
    type: 'button_style',
    rarity: 'common',
    cssVars: {
      '--button-bg': 'rgba(59, 130, 246, 0.2)',
      '--button-border': 'rgba(59, 130, 246, 0.5)',
      '--button-text': '#60a5fa',
    },
    preview: '🔘'
  },
  {
    id: 'journal_default',
    name: 'Classic Journal',
    description: 'Standard journal appearance',
    type: 'journal_design',
    rarity: 'common',
    cssVars: {
      '--journal-bg': 'rgba(3, 7, 18, 0.8)',
      '--journal-border': 'rgba(147, 51, 234, 0.3)',
      '--journal-text': '#e5e7eb',
    },
    preview: '📓'
  },
  {
    id: 'metrics_default',
    name: 'Standard Metrics',
    description: 'Default metric color scheme',
    type: 'metrics_theme',
    rarity: 'common',
    cssVars: {
      '--metric-high': '#22c55e',
      '--metric-medium': '#eab308',
      '--metric-low': '#ef4444',
    },
    preview: '📊'
  },

  // ---------------------------------------------------------------------------
  // BACKGROUND COSMETICS
  // ---------------------------------------------------------------------------
  {
    id: 'bg_sunrise_gradient',
    name: 'Sunrise Warmth',
    description: 'Warm amber and orange tones for a fresh morning feel',
    type: 'background',
    rarity: 'common',
    cssVars: {
      '--bg-primary': '#1a0a00',
      '--bg-secondary': '#2d1810',
      '--bg-tertiary': '#4a2c20',
    },
    preview: '🌅'
  },
  {
    id: 'bg_ocean_waves',
    name: 'Ocean Depths',
    description: 'Deep sea blues and teals',
    type: 'background',
    rarity: 'common',
    cssVars: {
      '--bg-primary': '#001219',
      '--bg-secondary': '#005f73',
      '--bg-tertiary': '#0a9396',
    },
    preview: '🌊'
  },
  {
    id: 'bg_starry_night',
    name: 'Starry Night',
    description: 'Deep purples with starlight accents',
    type: 'background',
    rarity: 'uncommon',
    cssVars: {
      '--bg-primary': '#0d0221',
      '--bg-secondary': '#1a0533',
      '--bg-tertiary': '#2d1b4e',
    },
    preview: '✨'
  },
  {
    id: 'bg_paint_splash',
    name: 'Artist\'s Canvas',
    description: 'Creative splashes of color on dark canvas',
    type: 'background',
    rarity: 'uncommon',
    cssVars: {
      '--bg-primary': '#0f0f0f',
      '--bg-secondary': '#1a1a2e',
      '--bg-tertiary': '#16213e',
    },
    preview: '🎨'
  },
  {
    id: 'bg_emotional_spectrum',
    name: 'Emotional Spectrum',
    description: 'Colors that shift through the emotional rainbow',
    type: 'background',
    rarity: 'rare',
    cssVars: {
      '--bg-primary': '#0f0716',
      '--bg-secondary': '#1a0d2e',
      '--bg-tertiary': '#2d1945',
    },
    preview: '🎭'
  },
  {
    id: 'bg_cosmic_aurora',
    name: 'Cosmic Aurora',
    description: 'Northern lights dancing across the cosmos',
    type: 'background',
    rarity: 'legendary',
    cssVars: {
      '--bg-primary': '#020617',
      '--bg-secondary': '#0c1a3d',
      '--bg-tertiary': '#1e3a5f',
    },
    preview: '🌌'
  },

  // ---------------------------------------------------------------------------
  // GRADIENT COSMETICS
  // ---------------------------------------------------------------------------
  {
    id: 'gradient_smooth_silk',
    name: 'Smooth Silk',
    description: 'Silky smooth gradient transitions',
    type: 'gradient',
    rarity: 'uncommon',
    cssClass: 'gradient-smooth-silk',
    cssVars: {
      '--gradient-start': '#0f0f0f',
      '--gradient-mid': '#1a1a2e',
      '--gradient-end': '#2d2d44',
      '--gradient-angle': '135deg',
    },
    preview: '🪡'
  },
  {
    id: 'gradient_creative_burst',
    name: 'Creative Burst',
    description: 'Explosive creativity in gradient form',
    type: 'gradient',
    rarity: 'rare',
    cssClass: 'gradient-creative-burst',
    cssVars: {
      '--gradient-start': '#1a0533',
      '--gradient-mid': '#2d1b69',
      '--gradient-end': '#4a1d96',
      '--gradient-angle': '45deg',
    },
    preview: '💥'
  },
  {
    id: 'gradient_forest_mist',
    name: 'Forest Mist',
    description: 'Misty greens of an ancient forest',
    type: 'gradient',
    rarity: 'uncommon',
    cssClass: 'gradient-forest-mist',
    cssVars: {
      '--gradient-start': '#0d1f0d',
      '--gradient-mid': '#1a3a1a',
      '--gradient-end': '#2d5a2d',
      '--gradient-angle': '180deg',
    },
    preview: '🌲'
  },
  {
    id: 'gradient_sunset_dream',
    name: 'Sunset Dream',
    description: 'Warm sunset hues fading into night',
    type: 'gradient',
    rarity: 'rare',
    cssClass: 'gradient-sunset-dream',
    cssVars: {
      '--gradient-start': '#1a0a00',
      '--gradient-mid': '#4a1a1a',
      '--gradient-end': '#2d1b4e',
      '--gradient-angle': '135deg',
    },
    preview: '🌇'
  },
  {
    id: 'gradient_cyber_neon',
    name: 'Cyber Neon',
    description: 'Futuristic neon cyberpunk vibes',
    type: 'gradient',
    rarity: 'epic',
    cssClass: 'gradient-cyber-neon',
    cssVars: {
      '--gradient-start': '#0a0a1a',
      '--gradient-mid': '#1a0a2e',
      '--gradient-end': '#0a1a2e',
      '--gradient-angle': '120deg',
    },
    preview: '🤖'
  },

  // ---------------------------------------------------------------------------
  // ANIMATION COSMETICS
  // ---------------------------------------------------------------------------
  {
    id: 'animation_breathing_pulse',
    name: 'Breathing Pulse',
    description: 'Gentle pulsing animation that follows your breath',
    type: 'animation',
    rarity: 'uncommon',
    cssClass: 'animation-breathing-pulse',
    preview: '🫁'
  },
  {
    id: 'animation_gentle_glow',
    name: 'Gentle Glow',
    description: 'Soft ambient glow effect',
    type: 'animation',
    rarity: 'uncommon',
    cssClass: 'animation-gentle-glow',
    preview: '✨'
  },
  {
    id: 'animation_ripple_effect',
    name: 'Ripple Effect',
    description: 'Ripples emanate from interactions',
    type: 'animation',
    rarity: 'rare',
    cssClass: 'animation-ripple-effect',
    preview: '💫'
  },
  {
    id: 'animation_energy_wave',
    name: 'Energy Wave',
    description: 'Dynamic energy waves flowing across the screen',
    type: 'animation',
    rarity: 'rare',
    cssClass: 'animation-energy-wave',
    preview: '⚡'
  },
  {
    id: 'animation_particle_float',
    name: 'Floating Particles',
    description: 'Gentle particles floating in the background',
    type: 'animation',
    rarity: 'epic',
    cssClass: 'animation-particle-float',
    preview: '🔮'
  },
  {
    id: 'animation_zen_particles',
    name: 'Zen Particles',
    description: 'Calm, meditative particle movement',
    type: 'animation',
    rarity: 'legendary',
    cssClass: 'animation-zen-particles',
    preview: '☯️'
  },

  // ---------------------------------------------------------------------------
  // WALLPAPER COSMETICS
  // ---------------------------------------------------------------------------
  {
    id: 'wallpaper_geometric_gold',
    name: 'Geometric Gold',
    description: 'Elegant gold geometric patterns',
    type: 'wallpaper',
    rarity: 'epic',
    cssClass: 'wallpaper-geometric-gold',
    preview: '🔷'
  },
  {
    id: 'wallpaper_abstract_art',
    name: 'Abstract Art',
    description: 'Creative abstract art background',
    type: 'wallpaper',
    rarity: 'epic',
    cssClass: 'wallpaper-abstract-art',
    preview: '🎭'
  },
  {
    id: 'wallpaper_neural_network',
    name: 'Neural Network',
    description: 'Connected nodes representing neural pathways',
    type: 'wallpaper',
    rarity: 'epic',
    cssClass: 'wallpaper-neural-network',
    preview: '🧠'
  },
  {
    id: 'wallpaper_legendary_flames',
    name: 'Legendary Flames',
    description: 'Majestic flames of achievement',
    type: 'wallpaper',
    rarity: 'legendary',
    cssClass: 'wallpaper-legendary-flames',
    preview: '🔥'
  },
  {
    id: 'wallpaper_constellation',
    name: 'Constellation Map',
    description: 'Star patterns and constellation lines',
    type: 'wallpaper',
    rarity: 'rare',
    cssClass: 'wallpaper-constellation',
    preview: '⭐'
  },

  // ---------------------------------------------------------------------------
  // TAB DESIGN COSMETICS
  // ---------------------------------------------------------------------------
  {
    id: 'tab_wellness_aurora',
    name: 'Wellness Aurora',
    description: 'Calming aurora-inspired tab design',
    type: 'tab_design',
    rarity: 'rare',
    cssClass: 'tabs-wellness-aurora',
    cssVars: {
      '--tab-bg': 'rgba(34, 197, 94, 0.1)',
      '--tab-active-bg': 'rgba(34, 197, 94, 0.3)',
      '--tab-border': 'rgba(34, 197, 94, 0.5)',
      '--tab-active-text': '#22c55e',
    },
    preview: '🌿'
  },
  {
    id: 'tab_social_gradient',
    name: 'Social Gradient',
    description: 'Warm social connection-themed tabs',
    type: 'tab_design',
    rarity: 'uncommon',
    cssClass: 'tabs-social-gradient',
    cssVars: {
      '--tab-bg': 'rgba(251, 146, 60, 0.1)',
      '--tab-active-bg': 'rgba(251, 146, 60, 0.3)',
      '--tab-border': 'rgba(251, 146, 60, 0.5)',
      '--tab-active-text': '#fb923c',
    },
    preview: '🤝'
  },
  {
    id: 'tab_iron_forge',
    name: 'Iron Forge',
    description: 'Strong metallic tab styling',
    type: 'tab_design',
    rarity: 'epic',
    cssClass: 'tabs-iron-forge',
    cssVars: {
      '--tab-bg': 'rgba(148, 163, 184, 0.1)',
      '--tab-active-bg': 'rgba(148, 163, 184, 0.3)',
      '--tab-border': 'rgba(148, 163, 184, 0.6)',
      '--tab-active-text': '#cbd5e1',
    },
    preview: '⚔️'
  },
  {
    id: 'tab_neon_glow',
    name: 'Neon Glow Tabs',
    description: 'Vibrant neon-styled tabs',
    type: 'tab_design',
    rarity: 'rare',
    cssClass: 'tabs-neon-glow',
    cssVars: {
      '--tab-bg': 'rgba(236, 72, 153, 0.1)',
      '--tab-active-bg': 'rgba(236, 72, 153, 0.2)',
      '--tab-border': 'rgba(236, 72, 153, 0.6)',
      '--tab-active-text': '#ec4899',
    },
    preview: '💖'
  },
  {
    id: 'tab_crystal_ice',
    name: 'Crystal Ice',
    description: 'Cool, crystalline tab appearance',
    type: 'tab_design',
    rarity: 'uncommon',
    cssClass: 'tabs-crystal-ice',
    cssVars: {
      '--tab-bg': 'rgba(125, 211, 252, 0.1)',
      '--tab-active-bg': 'rgba(125, 211, 252, 0.2)',
      '--tab-border': 'rgba(125, 211, 252, 0.5)',
      '--tab-active-text': '#7dd3fc',
    },
    preview: '❄️'
  },

  // ---------------------------------------------------------------------------
  // BUTTON STYLE COSMETICS
  // ---------------------------------------------------------------------------
  {
    id: 'button_metallic_shine',
    name: 'Metallic Shine',
    description: 'Sleek metallic button appearance',
    type: 'button_style',
    rarity: 'uncommon',
    cssClass: 'buttons-metallic-shine',
    cssVars: {
      '--button-bg': 'linear-gradient(180deg, rgba(148, 163, 184, 0.3), rgba(100, 116, 139, 0.2))',
      '--button-border': 'rgba(148, 163, 184, 0.5)',
      '--button-text': '#e2e8f0',
    },
    preview: '🔩'
  },
  {
    id: 'button_power_pulse',
    name: 'Power Pulse',
    description: 'Energetic button styling with pulse effect',
    type: 'button_style',
    rarity: 'rare',
    cssClass: 'buttons-power-pulse',
    cssVars: {
      '--button-bg': 'rgba(239, 68, 68, 0.2)',
      '--button-border': 'rgba(239, 68, 68, 0.5)',
      '--button-text': '#fca5a5',
      '--button-hover-bg': 'rgba(239, 68, 68, 0.3)',
    },
    preview: '💪'
  },
  {
    id: 'button_emerald_glow',
    name: 'Emerald Glow',
    description: 'Rich emerald green buttons',
    type: 'button_style',
    rarity: 'uncommon',
    cssClass: 'buttons-emerald-glow',
    cssVars: {
      '--button-bg': 'rgba(16, 185, 129, 0.2)',
      '--button-border': 'rgba(16, 185, 129, 0.5)',
      '--button-text': '#6ee7b7',
      '--button-hover-bg': 'rgba(16, 185, 129, 0.3)',
    },
    preview: '💚'
  },
  {
    id: 'button_royal_purple',
    name: 'Royal Purple',
    description: 'Regal purple button styling',
    type: 'button_style',
    rarity: 'rare',
    cssClass: 'buttons-royal-purple',
    cssVars: {
      '--button-bg': 'rgba(139, 92, 246, 0.2)',
      '--button-border': 'rgba(139, 92, 246, 0.5)',
      '--button-text': '#c4b5fd',
      '--button-hover-bg': 'rgba(139, 92, 246, 0.3)',
    },
    preview: '👑'
  },
  {
    id: 'button_golden_touch',
    name: 'Golden Touch',
    description: 'Luxurious gold-accented buttons',
    type: 'button_style',
    rarity: 'epic',
    cssClass: 'buttons-golden-touch',
    cssVars: {
      '--button-bg': 'rgba(234, 179, 8, 0.2)',
      '--button-border': 'rgba(234, 179, 8, 0.5)',
      '--button-text': '#fde047',
      '--button-hover-bg': 'rgba(234, 179, 8, 0.3)',
    },
    preview: '✨'
  },

  // ---------------------------------------------------------------------------
  // METRICS THEME COSMETICS
  // ---------------------------------------------------------------------------
  {
    id: 'metrics_neon_glow',
    name: 'Neon Metrics',
    description: 'Vibrant neon colors for your metrics',
    type: 'metrics_theme',
    rarity: 'uncommon',
    cssClass: 'metrics-neon-glow',
    cssVars: {
      '--metric-high': '#00ff88',
      '--metric-medium': '#ffff00',
      '--metric-low': '#ff0066',
      '--metric-bg': 'rgba(0, 255, 136, 0.1)',
    },
    preview: '📈'
  },
  {
    id: 'metrics_rainbow_shift',
    name: 'Rainbow Shift',
    description: 'Full spectrum rainbow metric colors',
    type: 'metrics_theme',
    rarity: 'epic',
    cssClass: 'metrics-rainbow-shift',
    cssVars: {
      '--metric-high': '#a855f7',
      '--metric-medium': '#3b82f6',
      '--metric-low': '#ec4899',
      '--metric-bg': 'rgba(168, 85, 247, 0.1)',
    },
    preview: '🌈'
  },
  {
    id: 'metrics_ocean_depths',
    name: 'Ocean Depths',
    description: 'Deep sea-inspired metric colors',
    type: 'metrics_theme',
    rarity: 'uncommon',
    cssClass: 'metrics-ocean-depths',
    cssVars: {
      '--metric-high': '#06b6d4',
      '--metric-medium': '#0891b2',
      '--metric-low': '#0e7490',
      '--metric-bg': 'rgba(6, 182, 212, 0.1)',
    },
    preview: '🌊'
  },
  {
    id: 'metrics_fire_forge',
    name: 'Fire Forge',
    description: 'Intense fire-themed metric display',
    type: 'metrics_theme',
    rarity: 'rare',
    cssClass: 'metrics-fire-forge',
    cssVars: {
      '--metric-high': '#f97316',
      '--metric-medium': '#dc2626',
      '--metric-low': '#7f1d1d',
      '--metric-bg': 'rgba(249, 115, 22, 0.1)',
    },
    preview: '🔥'
  },
  {
    id: 'metrics_aurora_borealis',
    name: 'Aurora Borealis',
    description: 'Northern lights metric colors',
    type: 'metrics_theme',
    rarity: 'legendary',
    cssClass: 'metrics-aurora-borealis',
    cssVars: {
      '--metric-high': '#22d3ee',
      '--metric-medium': '#a78bfa',
      '--metric-low': '#f472b6',
      '--metric-bg': 'rgba(34, 211, 238, 0.1)',
    },
    preview: '🌌'
  },

  // ---------------------------------------------------------------------------
  // JOURNAL DESIGN COSMETICS
  // ---------------------------------------------------------------------------
  {
    id: 'journal_parchment_classic',
    name: 'Parchment Classic',
    description: 'Aged parchment style for your journal',
    type: 'journal_design',
    rarity: 'common',
    cssClass: 'journal-parchment',
    cssVars: {
      '--journal-bg': 'rgba(45, 38, 28, 0.9)',
      '--journal-border': 'rgba(194, 154, 108, 0.4)',
      '--journal-text': '#d4c5a9',
      '--journal-accent': '#c29a6c',
    },
    preview: '📜'
  },
  {
    id: 'journal_ink_flow',
    name: 'Ink Flow',
    description: 'Flowing ink aesthetic for your entries',
    type: 'journal_design',
    rarity: 'uncommon',
    cssClass: 'journal-ink-flow',
    cssVars: {
      '--journal-bg': 'rgba(15, 15, 25, 0.9)',
      '--journal-border': 'rgba(99, 102, 241, 0.4)',
      '--journal-text': '#c7d2fe',
      '--journal-accent': '#818cf8',
    },
    preview: '🖋️'
  },
  {
    id: 'journal_deep_purple',
    name: 'Deep Purple',
    description: 'Rich purple tones for thoughtful writing',
    type: 'journal_design',
    rarity: 'uncommon',
    cssClass: 'journal-deep-purple',
    cssVars: {
      '--journal-bg': 'rgba(30, 10, 45, 0.9)',
      '--journal-border': 'rgba(168, 85, 247, 0.4)',
      '--journal-text': '#e9d5ff',
      '--journal-accent': '#a855f7',
    },
    preview: '💜'
  },
  {
    id: 'journal_scholar_theme',
    name: 'Scholar\'s Study',
    description: 'Academic and scholarly journal appearance',
    type: 'journal_design',
    rarity: 'rare',
    cssClass: 'journal-scholar',
    cssVars: {
      '--journal-bg': 'rgba(20, 25, 20, 0.9)',
      '--journal-border': 'rgba(134, 239, 172, 0.3)',
      '--journal-text': '#bbf7d0',
      '--journal-accent': '#4ade80',
    },
    preview: '📚'
  },
  {
    id: 'journal_legendary_tome',
    name: 'Legendary Tome',
    description: 'Ancient tome styling with golden accents',
    type: 'journal_design',
    rarity: 'legendary',
    cssClass: 'journal-legendary-tome',
    cssVars: {
      '--journal-bg': 'rgba(35, 25, 15, 0.95)',
      '--journal-border': 'rgba(234, 179, 8, 0.5)',
      '--journal-text': '#fef3c7',
      '--journal-accent': '#fbbf24',
    },
    preview: '📖'
  },
  {
    id: 'journal_midnight_moon',
    name: 'Midnight Moon',
    description: 'Mystical moonlit journal design',
    type: 'journal_design',
    rarity: 'rare',
    cssClass: 'journal-midnight-moon',
    cssVars: {
      '--journal-bg': 'rgba(15, 10, 30, 0.9)',
      '--journal-border': 'rgba(192, 132, 252, 0.4)',
      '--journal-text': '#ddd6fe',
      '--journal-accent': '#a78bfa',
    },
    preview: '🌙'
  },

  // ---------------------------------------------------------------------------
  // CARD STYLE COSMETICS
  // ---------------------------------------------------------------------------
  {
    id: 'card_glass_morphism',
    name: 'Glass Morphism',
    description: 'Frosted glass effect for cards',
    type: 'card_style',
    rarity: 'uncommon',
    cssClass: 'cards-glass-morphism',
    cssVars: {
      '--card-bg': 'rgba(255, 255, 255, 0.05)',
      '--card-border': 'rgba(255, 255, 255, 0.1)',
      '--card-shadow': '0 8px 32px rgba(0, 0, 0, 0.3)',
    },
    preview: '🪟'
  },
  {
    id: 'card_neon_border',
    name: 'Neon Border',
    description: 'Glowing neon borders on cards',
    type: 'card_style',
    rarity: 'rare',
    cssClass: 'cards-neon-border',
    cssVars: {
      '--card-bg': 'rgba(0, 0, 0, 0.7)',
      '--card-border': 'rgba(236, 72, 153, 0.6)',
      '--card-shadow': '0 0 20px rgba(236, 72, 153, 0.3)',
    },
    preview: '💗'
  },
  {
    id: 'card_golden_frame',
    name: 'Golden Frame',
    description: 'Elegant gold-framed cards',
    type: 'card_style',
    rarity: 'epic',
    cssClass: 'cards-golden-frame',
    cssVars: {
      '--card-bg': 'rgba(15, 10, 5, 0.8)',
      '--card-border': 'rgba(234, 179, 8, 0.5)',
      '--card-shadow': '0 4px 20px rgba(234, 179, 8, 0.2)',
    },
    preview: '🏆'
  },
  {
    id: 'card_holographic',
    name: 'Holographic',
    description: 'Shimmering holographic card effect',
    type: 'card_style',
    rarity: 'legendary',
    cssClass: 'cards-holographic',
    cssVars: {
      '--card-bg': 'rgba(10, 10, 20, 0.8)',
      '--card-border': 'rgba(147, 51, 234, 0.4)',
      '--card-shadow': '0 0 30px rgba(147, 51, 234, 0.2)',
    },
    preview: '🌟'
  },

  // ---------------------------------------------------------------------------
  // ACCENT COLOR COSMETICS
  // ---------------------------------------------------------------------------
  {
    id: 'accent_flame_orange',
    name: 'Flame Orange',
    description: 'Fiery orange accent color',
    type: 'accent_color',
    rarity: 'common',
    cssVars: {
      '--accent-primary': '#f97316',
      '--accent-secondary': '#ea580c',
      '--accent-glow': 'rgba(249, 115, 22, 0.5)',
    },
    preview: '🔥'
  },
  {
    id: 'accent_warm_coral',
    name: 'Warm Coral',
    description: 'Soft coral accent for a welcoming feel',
    type: 'accent_color',
    rarity: 'common',
    cssVars: {
      '--accent-primary': '#fb7185',
      '--accent-secondary': '#f43f5e',
      '--accent-glow': 'rgba(251, 113, 133, 0.5)',
    },
    preview: '🪸'
  },
  {
    id: 'accent_wisdom_purple',
    name: 'Wisdom Purple',
    description: 'Deep purple symbolizing knowledge',
    type: 'accent_color',
    rarity: 'uncommon',
    cssVars: {
      '--accent-primary': '#a855f7',
      '--accent-secondary': '#9333ea',
      '--accent-glow': 'rgba(168, 85, 247, 0.5)',
    },
    preview: '💜'
  },
  {
    id: 'accent_emerald_life',
    name: 'Emerald Life',
    description: 'Vibrant emerald green accent',
    type: 'accent_color',
    rarity: 'uncommon',
    cssVars: {
      '--accent-primary': '#10b981',
      '--accent-secondary': '#059669',
      '--accent-glow': 'rgba(16, 185, 129, 0.5)',
    },
    preview: '💚'
  },
  {
    id: 'accent_royal_gold',
    name: 'Royal Gold',
    description: 'Luxurious golden accent',
    type: 'accent_color',
    rarity: 'epic',
    cssVars: {
      '--accent-primary': '#fbbf24',
      '--accent-secondary': '#f59e0b',
      '--accent-glow': 'rgba(251, 191, 36, 0.5)',
    },
    preview: '👑'
  },
  {
    id: 'accent_celestial_cyan',
    name: 'Celestial Cyan',
    description: 'Ethereal cyan accent color',
    type: 'accent_color',
    rarity: 'rare',
    cssVars: {
      '--accent-primary': '#22d3ee',
      '--accent-secondary': '#06b6d4',
      '--accent-glow': 'rgba(34, 211, 238, 0.5)',
    },
    preview: '💎'
  },
];

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

export function getCosmeticsByRarity(rarity: CosmeticReward['rarity']): CosmeticReward[] {
  return COSMETIC_CATALOG.filter(c => c.rarity === rarity);
}

export function getCosmeticsByType(type: CosmeticType): CosmeticReward[] {
  return COSMETIC_CATALOG.filter(c => c.type === type);
}

export function getCosmeticById(id: string): CosmeticReward | undefined {
  return COSMETIC_CATALOG.find(c => c.id === id);
}

export function getRarityColor(rarity: CosmeticReward['rarity']): string {
  switch (rarity) {
    case 'common': return '#9ca3af';
    case 'uncommon': return '#22c55e';
    case 'rare': return '#3b82f6';
    case 'epic': return '#a855f7';
    case 'legendary': return '#f59e0b';
    default: return '#9ca3af';
  }
}

export function getRarityLabel(rarity: CosmeticReward['rarity']): string {
  return rarity.charAt(0).toUpperCase() + rarity.slice(1);
}

export function getTypeIcon(type: CosmeticType): string {
  switch (type) {
    case 'background': return '🖼️';
    case 'gradient': return '🌈';
    case 'animation': return '✨';
    case 'wallpaper': return '🎨';
    case 'tab_design': return '📑';
    case 'button_style': return '🔘';
    case 'metrics_theme': return '📊';
    case 'journal_design': return '📓';
    case 'card_style': return '🃏';
    case 'accent_color': return '🎯';
    default: return '🎁';
  }
}

export function getTypeLabel(type: CosmeticType): string {
  return type.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
}

// ============================================================================
// CSS GENERATION
// ============================================================================

export function generateCosmeticCSS(equippedCosmetics: Record<CosmeticType, string | null>): string {
  let cssVars: Record<string, string> = { ...CSS_VAR_DEFAULTS };
  let cssClasses: string[] = [];

  // Apply each equipped cosmetic
  for (const [_type, cosmeticId] of Object.entries(equippedCosmetics)) {
    if (!cosmeticId) continue;

    const cosmetic = getCosmeticById(cosmeticId);
    if (!cosmetic) continue;

    // Merge CSS variables
    if (cosmetic.cssVars) {
      cssVars = { ...cssVars, ...cosmetic.cssVars };
    }

    // Add CSS classes
    if (cosmetic.cssClass) {
      cssClasses.push(cosmetic.cssClass);
    }
  }

  return Object.entries(cssVars)
    .map(([key, value]) => `${key}: ${value};`)
    .join('\n');
}

export function getEquippedCssClasses(equippedCosmetics: Record<CosmeticType, string | null>): string[] {
  const classes: string[] = [];

  for (const cosmeticId of Object.values(equippedCosmetics)) {
    if (!cosmeticId) continue;
    const cosmetic = getCosmeticById(cosmeticId);
    if (cosmetic?.cssClass) {
      classes.push(cosmetic.cssClass);
    }
  }

  return classes;
}
