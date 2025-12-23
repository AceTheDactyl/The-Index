/* INTEGRITY_METADATA
 * Date: 2025-12-23
 * Status: ⚠️ NEEDS REVIEW
 * Severity: MEDIUM RISK
 * Risk Types: unverified_math
 */


/**
 * Neural Map Engine
 *
 * Integrates DeltaHV metrics (Symbolic, Resonance, Friction, Stability) with
 * brain regions using the Free Energy Principle to generate dynamic neural maps.
 *
 * The FEP framework views the brain as constantly trying to minimize "free energy"
 * (surprise/prediction error). This engine maps user metrics to brain activation
 * patterns, showing which neural regions are engaged or need support.
 */

import type { DeltaHVState } from './deltaHVEngine';
import type { UserProfile } from './userProfile';
import {
  BRAIN_REGIONS,
  type BrainRegion,
  type WumboPhase,
  type BrainRegionCategory
} from './glyphSystem';

// ============================================================================
// Types
// ============================================================================

export interface NeuralActivation {
  regionId: string;
  regionName: string;
  glyph: string;
  category: BrainRegionCategory;
  activation: number;        // 0-100 activation level
  coherence: number;         // 0-100 how aligned with user's state
  freeEnergy: number;        // 0-100 prediction error / friction
  dominantMetric: 'symbolic' | 'resonance' | 'friction' | 'stability';
  phase: WumboPhase;
  neurochemicals: string[];
  tooltip: string;
}

export interface GlyphicResonanceField {
  id: string;
  name: string;
  glyph: string;
  domain: string;
  strength: number;          // 0-100 field strength
  coherence: number;         // 0-100 alignment with user intent
  activeRegions: string[];   // Brain region IDs in this field
  phase: WumboPhase;
  color: string;
}

export interface NeuralMap {
  // Overall state
  overallCoherence: number;
  overallFreeEnergy: number;
  dominantPhase: WumboPhase;
  fieldState: 'coherent' | 'transitioning' | 'fragmented' | 'dormant';

  // Regional activations
  activations: NeuralActivation[];

  // Glyphic resonance fields (clusters of related regions)
  resonanceFields: GlyphicResonanceField[];

  // Metric-to-region mappings
  metricInfluence: {
    symbolic: { regions: string[]; strength: number };
    resonance: { regions: string[]; strength: number };
    friction: { regions: string[]; strength: number };
    stability: { regions: string[]; strength: number };
  };

  // Recommendations based on neural state
  recommendations: NeuralRecommendation[];

  // Timestamp
  generatedAt: string;
}

export interface NeuralRecommendation {
  id: string;
  type: 'activation' | 'regulation' | 'integration' | 'rest';
  title: string;
  description: string;
  targetRegions: string[];
  suggestedBeat: string;
  glyph: string;
  priority: 'high' | 'medium' | 'low';
}

// ============================================================================
// Metric-to-Region Mappings
// ============================================================================

/**
 * Maps DeltaHV metrics to brain regions that are primarily involved
 */
const METRIC_REGION_MAP: Record<string, string[]> = {
  // Symbolic Density - regions involved in meaning, symbolism, self-reflection
  symbolic: [
    'mpfc',              // Self-referential thinking
    'precuneus',         // Self-consciousness, imagery
    'temporalpole',      // Emotional semantic knowledge
    'cingulate_posterior', // Self-reflection
    'hippocampus_ca1',   // Memory consolidation
    'angular_gyrus',     // Semantic processing (if present)
    'dlpfc',             // Working memory for symbols
  ],

  // Resonance Coupling - regions for alignment, intention, action matching
  resonance: [
    'dacc',              // Conflict monitoring (alignment)
    'racc',              // Emotional alignment
    'ofc',               // Value-based decisions
    'nucleus_accumbens_core', // Goal-directed behavior
    'striatum_dorsal',   // Habit formation
    'pmc',               // Movement planning
    'sma',               // Internally generated actions
  ],

  // Friction Coefficient - regions activated by stress, missed goals, delays
  friction: [
    'amygdala_bla',      // Fear/stress
    'amygdala_cea',      // Autonomic stress response
    'bed_nucleus',       // Sustained anxiety
    'habenula',          // Disappointment
    'sgacc',             // Mood regulation under stress
    'locus_coeruleus',   // Arousal/stress
    'hypothalamus_pvn',  // HPA axis
  ],

  // Harmonic Stability - regions for rhythm, consistency, regulation
  stability: [
    'raphe_dorsal',      // Mood stability (serotonin)
    'raphe_median',      // Anxiety regulation
    'vermis',            // Emotional balance
    'septum',            // Reward/pleasure baseline
    'reticular_formation', // Arousal regulation
    'dorsal_motor_vagus', // Rest/digest
    'deep_cerebellar',   // Rhythmic timing
  ],
};

/**
 * Domain-specific color schemes for visualization
 */
const DOMAIN_COLORS: Record<string, string> = {
  body: '#22d3ee',      // cyan
  mind: '#a855f7',      // purple
  emotion: '#ec4899',   // pink
  spirit: '#fbbf24',    // yellow
  social: '#22c55e',    // green
  creative: '#f97316',  // orange
};

// ============================================================================
// Neural Map Generation
// ============================================================================

/**
 * Calculate activation level for a brain region based on metrics
 */
function calculateRegionActivation(
  region: BrainRegion,
  deltaHV: DeltaHVState,
  profile: UserProfile | null
): NeuralActivation {
  const regionId = region.id;

  // Determine which metric most influences this region
  let dominantMetric: 'symbolic' | 'resonance' | 'friction' | 'stability' = 'symbolic';
  let metricInfluence = 0;

  if (METRIC_REGION_MAP.symbolic.includes(regionId)) {
    dominantMetric = 'symbolic';
    metricInfluence = deltaHV.symbolicDensity;
  } else if (METRIC_REGION_MAP.resonance.includes(regionId)) {
    dominantMetric = 'resonance';
    metricInfluence = deltaHV.resonanceCoupling;
  } else if (METRIC_REGION_MAP.friction.includes(regionId)) {
    dominantMetric = 'friction';
    // Friction regions activate MORE when friction is HIGH
    metricInfluence = deltaHV.frictionCoefficient;
  } else if (METRIC_REGION_MAP.stability.includes(regionId)) {
    dominantMetric = 'stability';
    metricInfluence = deltaHV.harmonicStability;
  } else {
    // Default based on domain
    const domain = region.linkedDomains[0] || 'mind';
    switch (domain) {
      case 'body':
        dominantMetric = 'resonance';
        metricInfluence = deltaHV.resonanceCoupling;
        break;
      case 'emotion':
        dominantMetric = 'friction';
        metricInfluence = 100 - deltaHV.frictionCoefficient; // Inverted for emotion
        break;
      case 'spirit':
        dominantMetric = 'symbolic';
        metricInfluence = deltaHV.symbolicDensity;
        break;
      default:
        metricInfluence = deltaHV.deltaHV;
    }
  }

  // Calculate activation based on metric and region function
  let activation = metricInfluence;

  // Adjust for phase alignment
  const currentPhase = getPhaseFromMetrics(deltaHV);
  if (region.wumboPhases.includes(currentPhase)) {
    activation = Math.min(100, activation * 1.2);
  }

  // Calculate coherence (how well the region's function matches user state)
  let coherence = deltaHV.deltaHV;
  if (dominantMetric === 'friction') {
    // Friction regions are "coherent" when they're appropriately regulated
    coherence = 100 - deltaHV.frictionCoefficient;
  }

  // Calculate free energy (prediction error / mismatch)
  // Lower free energy = better alignment
  const freeEnergy = calculateFreeEnergy(region, deltaHV, profile);

  // Determine phase for this region
  const phase = region.wumboPhases[0] || currentPhase;

  // Generate tooltip
  const tooltip = generateRegionTooltip(region, activation, coherence, freeEnergy, dominantMetric);

  return {
    regionId: region.id,
    regionName: region.name,
    glyph: region.glyph,
    category: region.category,
    activation: Math.round(activation),
    coherence: Math.round(coherence),
    freeEnergy: Math.round(freeEnergy),
    dominantMetric,
    phase,
    neurochemicals: region.neurochemicals,
    tooltip,
  };
}

/**
 * Calculate Free Energy (prediction error) for a region
 * Based on FEP: organisms try to minimize surprise/uncertainty
 */
function calculateFreeEnergy(
  region: BrainRegion,
  deltaHV: DeltaHVState,
  _profile: UserProfile | null
): number {
  // Base free energy from friction (misalignment = surprise)
  let freeEnergy = deltaHV.frictionCoefficient * 0.4;

  // Add energy from low resonance (unmet expectations)
  freeEnergy += (100 - deltaHV.resonanceCoupling) * 0.3;

  // Add energy from instability (unpredictable patterns)
  freeEnergy += (100 - deltaHV.harmonicStability) * 0.2;

  // Reduce energy with high symbolic density (meaning reduces uncertainty)
  freeEnergy -= deltaHV.symbolicDensity * 0.1;

  // Clamp to 0-100
  freeEnergy = Math.max(0, Math.min(100, freeEnergy));

  // Adjust based on region's stress sensitivity
  if (region.category === 'limbic') {
    freeEnergy *= 1.2; // Limbic regions more sensitive to free energy
  } else if (region.category === 'cortical') {
    freeEnergy *= 0.9; // Cortical regions can regulate better
  }

  return Math.min(100, freeEnergy);
}

/**
 * Determine current Wumbo phase from metrics
 */
function getPhaseFromMetrics(deltaHV: DeltaHVState): WumboPhase {
  const { symbolicDensity, resonanceCoupling, frictionCoefficient, harmonicStability, fieldState } = deltaHV;

  if (fieldState === 'dormant') return 'collapse';
  if (fieldState === 'fragmented') return 'reflection';

  // High activation metrics
  if (resonanceCoupling > 80 && harmonicStability > 70) {
    if (symbolicDensity > 70) return 'nirvana';
    return 'mania';
  }

  if (resonanceCoupling > 60) {
    if (symbolicDensity > 50) return 'resonance';
    return 'empowerment';
  }

  if (symbolicDensity > 60) return 'transmission';

  if (frictionCoefficient > 50) return 'collapse';

  if (harmonicStability > 50) return 'reflection';

  return 'ignition';
}

/**
 * Generate tooltip for region display
 */
function generateRegionTooltip(
  region: BrainRegion,
  activation: number,
  coherence: number,
  freeEnergy: number,
  metric: string
): string {
  const activationLevel = activation > 70 ? 'High' : activation > 40 ? 'Moderate' : 'Low';
  const coherenceLevel = coherence > 70 ? 'aligned' : coherence > 40 ? 'adjusting' : 'misaligned';
  const energyLevel = freeEnergy > 50 ? 'elevated' : freeEnergy > 25 ? 'moderate' : 'low';

  return `${region.name}\n` +
    `${region.function}\n\n` +
    `Activation: ${activationLevel} (${activation}%)\n` +
    `Coherence: ${coherenceLevel} (${coherence}%)\n` +
    `Free Energy: ${energyLevel} (${freeEnergy}%)\n` +
    `Primary Metric: ${metric}\n` +
    `Neurochemicals: ${region.neurochemicals.join(', ')}`;
}

/**
 * Generate glyphic resonance fields from activations
 */
function generateResonanceFields(
  activations: NeuralActivation[],
  deltaHV: DeltaHVState
): GlyphicResonanceField[] {
  const fields: GlyphicResonanceField[] = [];
  const currentPhase = getPhaseFromMetrics(deltaHV);

  // Create fields for each life domain
  const domains = ['body', 'mind', 'emotion', 'spirit', 'social', 'creative'];

  domains.forEach(domain => {
    const domainRegions = BRAIN_REGIONS.filter(r => r.linkedDomains.includes(domain));
    const domainActivations = activations.filter(a =>
      domainRegions.some(r => r.id === a.regionId)
    );

    if (domainActivations.length === 0) return;

    // Calculate field strength from average activation
    const avgActivation = domainActivations.reduce((sum, a) => sum + a.activation, 0) / domainActivations.length;
    const avgCoherence = domainActivations.reduce((sum, a) => sum + a.coherence, 0) / domainActivations.length;

    // Get dominant glyph for this domain
    const glyphCounts: Record<string, number> = {};
    domainActivations.forEach(a => {
      glyphCounts[a.glyph] = (glyphCounts[a.glyph] || 0) + a.activation;
    });
    const dominantGlyph = Object.entries(glyphCounts)
      .sort((a, b) => b[1] - a[1])[0]?.[0] || '🌀';

    fields.push({
      id: `field-${domain}`,
      name: domain.charAt(0).toUpperCase() + domain.slice(1),
      glyph: dominantGlyph,
      domain,
      strength: Math.round(avgActivation),
      coherence: Math.round(avgCoherence),
      activeRegions: domainActivations.map(a => a.regionId),
      phase: domainActivations[0]?.phase || currentPhase,
      color: DOMAIN_COLORS[domain] || '#ffffff',
    });
  });

  return fields.sort((a, b) => b.strength - a.strength);
}

/**
 * Generate recommendations based on neural state
 */
function generateNeuralRecommendations(
  activations: NeuralActivation[],
  deltaHV: DeltaHVState
): NeuralRecommendation[] {
  const recommendations: NeuralRecommendation[] = [];

  // Check for high free energy regions (need regulation)
  const highEnergyRegions = activations.filter(a => a.freeEnergy > 60);
  if (highEnergyRegions.length > 0) {
    const targetRegions = highEnergyRegions.slice(0, 3).map(r => r.regionId);
    recommendations.push({
      id: 'rec-regulate-energy',
      type: 'regulation',
      title: 'Reduce Neural Friction',
      description: `${highEnergyRegions.length} brain regions show elevated free energy. Practice grounding to reduce prediction error.`,
      targetRegions,
      suggestedBeat: 'Meditation',
      glyph: '🌊',
      priority: highEnergyRegions.length > 5 ? 'high' : 'medium',
    });
  }

  // Check for low symbolic density
  if (deltaHV.symbolicDensity < 30) {
    recommendations.push({
      id: 'rec-boost-symbolic',
      type: 'activation',
      title: 'Increase Symbolic Density',
      description: 'Low symbolic content detected. Journal with intention glyphs to activate meaning-processing regions.',
      targetRegions: METRIC_REGION_MAP.symbolic,
      suggestedBeat: 'Journal',
      glyph: '✨',
      priority: 'medium',
    });
  }

  // Check for low resonance
  if (deltaHV.resonanceCoupling < 40) {
    recommendations.push({
      id: 'rec-boost-resonance',
      type: 'activation',
      title: 'Align Intention with Action',
      description: 'Low resonance coupling. Complete a planned anchor beat to synchronize motor-intention circuits.',
      targetRegions: METRIC_REGION_MAP.resonance,
      suggestedBeat: 'Anchor',
      glyph: '🎯',
      priority: 'high',
    });
  }

  // Check for high friction
  if (deltaHV.frictionCoefficient > 50) {
    recommendations.push({
      id: 'rec-reduce-friction',
      type: 'regulation',
      title: 'Release Accumulated Friction',
      description: 'High friction detected from missed or delayed tasks. Practice emotional release to calm stress circuits.',
      targetRegions: METRIC_REGION_MAP.friction,
      suggestedBeat: 'Emotion',
      glyph: '🛡️',
      priority: 'high',
    });
  }

  // Check for low stability
  if (deltaHV.harmonicStability < 40) {
    recommendations.push({
      id: 'rec-boost-stability',
      type: 'integration',
      title: 'Establish Rhythmic Patterns',
      description: 'Irregular check-in patterns. Complete beats at consistent intervals to stabilize serotonin-related circuits.',
      targetRegions: METRIC_REGION_MAP.stability,
      suggestedBeat: 'General',
      glyph: '⟲',
      priority: 'medium',
    });
  }

  // If coherent, suggest maintenance
  if (deltaHV.fieldState === 'coherent') {
    recommendations.push({
      id: 'rec-maintain-coherence',
      type: 'integration',
      title: 'Maintain Coherent Flow',
      description: 'Neural field is coherent. Continue current practices and consider transmission activities.',
      targetRegions: ['vta', 'nucleus_accumbens_shell', 'precuneus'],
      suggestedBeat: 'Moderation',
      glyph: '🌟',
      priority: 'low',
    });
  }

  // Rest recommendation if energy is depleted
  const avgFreeEnergy = activations.reduce((sum, a) => sum + a.freeEnergy, 0) / activations.length;
  if (avgFreeEnergy > 70 || deltaHV.fieldState === 'dormant') {
    recommendations.push({
      id: 'rec-rest',
      type: 'rest',
      title: 'Neural Recovery Needed',
      description: 'High overall free energy indicates system strain. Rest and parasympathetic activation recommended.',
      targetRegions: ['dorsal_motor_vagus', 'raphe_dorsal', 'septum'],
      suggestedBeat: 'Anchor',
      glyph: '🌙',
      priority: 'high',
    });
  }

  return recommendations.sort((a, b) => {
    const priorityOrder = { high: 0, medium: 1, low: 2 };
    return priorityOrder[a.priority] - priorityOrder[b.priority];
  });
}

// ============================================================================
// Main Export Functions
// ============================================================================

/**
 * Generate a complete neural map from DeltaHV metrics
 */
export function generateNeuralMap(
  deltaHV: DeltaHVState,
  profile: UserProfile | null = null
): NeuralMap {
  // Calculate activations for all brain regions
  const activations = BRAIN_REGIONS.map(region =>
    calculateRegionActivation(region, deltaHV, profile)
  );

  // Generate resonance fields
  const resonanceFields = generateResonanceFields(activations, deltaHV);

  // Calculate metric influence
  const metricInfluence = {
    symbolic: {
      regions: METRIC_REGION_MAP.symbolic,
      strength: deltaHV.symbolicDensity,
    },
    resonance: {
      regions: METRIC_REGION_MAP.resonance,
      strength: deltaHV.resonanceCoupling,
    },
    friction: {
      regions: METRIC_REGION_MAP.friction,
      strength: deltaHV.frictionCoefficient,
    },
    stability: {
      regions: METRIC_REGION_MAP.stability,
      strength: deltaHV.harmonicStability,
    },
  };

  // Generate recommendations
  const recommendations = generateNeuralRecommendations(activations, deltaHV);

  // Calculate overall metrics
  const overallCoherence = activations.reduce((sum, a) => sum + a.coherence, 0) / activations.length;
  const overallFreeEnergy = activations.reduce((sum, a) => sum + a.freeEnergy, 0) / activations.length;
  const dominantPhase = getPhaseFromMetrics(deltaHV);

  return {
    overallCoherence: Math.round(overallCoherence),
    overallFreeEnergy: Math.round(overallFreeEnergy),
    dominantPhase,
    fieldState: deltaHV.fieldState,
    activations,
    resonanceFields,
    metricInfluence,
    recommendations,
    generatedAt: new Date().toISOString(),
  };
}

/**
 * Get top activated regions for a specific metric
 */
export function getTopRegionsForMetric(
  neuralMap: NeuralMap,
  metric: 'symbolic' | 'resonance' | 'friction' | 'stability',
  limit: number = 5
): NeuralActivation[] {
  return neuralMap.activations
    .filter(a => a.dominantMetric === metric)
    .sort((a, b) => b.activation - a.activation)
    .slice(0, limit);
}

/**
 * Get regions that need attention (high free energy)
 */
export function getRegionsNeedingAttention(
  neuralMap: NeuralMap,
  threshold: number = 50
): NeuralActivation[] {
  return neuralMap.activations
    .filter(a => a.freeEnergy > threshold)
    .sort((a, b) => b.freeEnergy - a.freeEnergy);
}

/**
 * Get the most coherent regions
 */
export function getMostCoherentRegions(
  neuralMap: NeuralMap,
  limit: number = 10
): NeuralActivation[] {
  return neuralMap.activations
    .sort((a, b) => b.coherence - a.coherence)
    .slice(0, limit);
}

/**
 * Calculate domain-specific coherence
 */
export function getDomainCoherence(
  neuralMap: NeuralMap,
  domain: string
): { coherence: number; activation: number; freeEnergy: number } {
  const domainField = neuralMap.resonanceFields.find(f => f.domain === domain);
  if (!domainField) {
    return { coherence: 0, activation: 0, freeEnergy: 100 };
  }

  const domainActivations = neuralMap.activations.filter(a =>
    domainField.activeRegions.includes(a.regionId)
  );

  if (domainActivations.length === 0) {
    return { coherence: 0, activation: 0, freeEnergy: 100 };
  }

  return {
    coherence: Math.round(domainActivations.reduce((sum, a) => sum + a.coherence, 0) / domainActivations.length),
    activation: Math.round(domainActivations.reduce((sum, a) => sum + a.activation, 0) / domainActivations.length),
    freeEnergy: Math.round(domainActivations.reduce((sum, a) => sum + a.freeEnergy, 0) / domainActivations.length),
  };
}

export default {
  generateNeuralMap,
  getTopRegionsForMetric,
  getRegionsNeedingAttention,
  getMostCoherentRegions,
  getDomainCoherence,
};
