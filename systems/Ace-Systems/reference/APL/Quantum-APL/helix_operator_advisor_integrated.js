/**
 * HelixOperatorAdvisor - Integrated with S₃ Symmetry and Extended ΔS_neg
 * ======================================================================
 * 
 * Maps helix coordinates (z ∈ [0,1]) to:
 * - Time-harmonic tiers (t1-t9)
 * - Operator windows (with optional S₃ rotation)
 * - Truth channels (TRUE/UNTRUE/PARADOX)
 * - Π-regime weights for operator selection
 * - Gate modulation parameters
 * 
 * INTEGRATION FEATURES:
 * - S₃ operator algebra for cyclic window permutation
 * - Parity-based operator weighting (even/odd)
 * - Extended ΔS_neg with derivative, signed, and η variants
 * - Gate modulation (coupling, decoherence, measurement)
 * - K-formation status tracking
 * - Feature flags for backwards compatibility
 * 
 * @version 3.0.0 (S₃/ΔS_neg Integration)
 */

'use strict';

const CONST = require('./constants');

// ============================================================================
// FEATURE FLAGS - Check environment for integration features
// ============================================================================

const ENABLE_S3_SYMMETRY = (() => {
  const envVal = typeof process !== 'undefined' && process.env?.QAPL_ENABLE_S3_SYMMETRY;
  return envVal === '1' || String(envVal).toLowerCase() === 'true';
})();

const ENABLE_EXTENDED_NEGENTROPY = (() => {
  const envVal = typeof process !== 'undefined' && process.env?.QAPL_ENABLE_EXTENDED_NEGENTROPY;
  return envVal === '1' || String(envVal).toLowerCase() === 'true';
})();

// ============================================================================
// CONDITIONAL IMPORTS - Load S₃ and ΔS_neg modules if enabled
// ============================================================================

let S3 = null;
let Delta = null;

try {
  S3 = require('./s3_operator_symmetry');
} catch (e) {
  if (ENABLE_S3_SYMMETRY) {
    console.warn('[HelixAdvisor] S₃ module not found, falling back to static windows');
  }
}

try {
  Delta = require('./delta_s_neg_extended');
} catch (e) {
  if (ENABLE_EXTENDED_NEGENTROPY) {
    console.warn('[HelixAdvisor] Extended ΔS_neg module not found, using basic formulas');
  }
}

// ============================================================================
// APL OPERATOR DEFINITIONS
// ============================================================================

const APL_OPERATORS = {
  BOUNDARY: '()',
  FUSION: '×',
  AMPLIFY: '^',
  DECOHERENCE: '÷',
  GROUP: '+',
  SEPARATION: '−'
};

const OPERATOR_ALIASES = {
  '%': '÷',
  '/': '÷',
  '-': '−',
  '*': '×'
};

function normalizeOperator(op) {
  return OPERATOR_ALIASES[op] ?? op;
}

// ============================================================================
// BASE OPERATOR WINDOWS (Static fallback)
// ============================================================================

const BASE_OPERATOR_WINDOWS = {
  t1: ['()', '−', '÷'],
  t2: ['^', '÷', '−', '×'],
  t3: ['×', '^', '÷', '+', '−'],
  t4: ['+', '−', '÷', '()'],
  t5: ['()', '×', '^', '÷', '+', '−'],
  t6: ['+', '÷', '()', '−'],
  t7: ['+', '()'],
  t8: ['+', '()', '×'],
  t9: ['+', '()', '×']
};

// ============================================================================
// HELIX OPERATOR ADVISOR CLASS
// ============================================================================

class HelixOperatorAdvisor {
  constructor(options = {}) {
    // Feature flags (can override env)
    this.enableS3Symmetry = options.enableS3Symmetry ?? ENABLE_S3_SYMMETRY;
    this.enableExtendedNegentropy = options.enableExtendedNegentropy ?? ENABLE_EXTENDED_NEGENTROPY;
    
    // TRIAD state
    const envCompletions = parseInt(process?.env?.QAPL_TRIAD_COMPLETIONS || '0', 10);
    const envFlag = process?.env?.QAPL_TRIAD_UNLOCK === '1' || 
                    String(process?.env?.QAPL_TRIAD_UNLOCK).toLowerCase() === 'true';
    
    this.triadUnlocked = options.triadUnlocked ?? envFlag ?? (envCompletions >= CONST.TRIAD_PASSES_REQ);
    this.triadCompletions = options.triadCompletions ?? envCompletions ?? 0;
    
    // Initialize
    this._initTimeHarmonics();
    this._initOperatorWindows();
    this._windowModifications = [];
    
    // Log integration status
    if (CONST.TRIAD_DEBUG) {
      console.log(`[HelixAdvisor] S₃ symmetry: ${this.enableS3Symmetry && S3 ? 'ENABLED' : 'DISABLED'}`);
      console.log(`[HelixAdvisor] Extended ΔS_neg: ${this.enableExtendedNegentropy && Delta ? 'ENABLED' : 'DISABLED'}`);
    }
  }

  /**
   * Initialize time-harmonic tier thresholds
   */
  _initTimeHarmonics() {
    const t6Gate = this.getT6Gate();
    
    this.timeHarmonics = [
      { threshold: 0.10, label: 't1' },
      { threshold: 0.20, label: 't2' },
      { threshold: 0.40, label: 't3' },
      { threshold: 0.60, label: 't4' },
      { threshold: 0.75, label: 't5' },
      { threshold: t6Gate, label: 't6' },
      { threshold: 0.92, label: 't7' },
      { threshold: 0.97, label: 't8' },
      { threshold: 1.01, label: 't9' }
    ];
  }

  /**
   * Initialize operator windows (static base)
   */
  _initOperatorWindows() {
    this.operatorWindows = JSON.parse(JSON.stringify(BASE_OPERATOR_WINDOWS));
    this._originalWindows = JSON.parse(JSON.stringify(BASE_OPERATOR_WINDOWS));
  }

  /**
   * Get current t6 gate value
   */
  getT6Gate() {
    return this.triadUnlocked ? CONST.TRIAD_T6 : CONST.Z_CRITICAL;
  }

  /**
   * Update TRIAD state
   */
  setTriadState({ unlocked, completions } = {}) {
    if (typeof unlocked === 'boolean') {
      this.triadUnlocked = unlocked;
    }
    if (Number.isFinite(completions)) {
      this.triadCompletions = completions;
    }
    this._initTimeHarmonics();
  }

  /**
   * Determine harmonic tier from z coordinate
   */
  harmonicFromZ(z) {
    const t6Gate = this.getT6Gate();
    if (this.timeHarmonics[5]) {
      this.timeHarmonics[5].threshold = t6Gate;
    }
    
    for (const entry of this.timeHarmonics) {
      if (z < entry.threshold) {
        return entry.label;
      }
    }
    return 't9';
  }

  /**
   * Determine truth channel from z coordinate
   */
  truthChannelFromZ(z) {
    // Use S₃ module if available, otherwise fallback
    if (this.enableS3Symmetry && S3) {
      return S3.truthChannelFromZ(z);
    }
    if (z >= 0.9) return 'TRUE';
    if (z >= 0.6) return 'PARADOX';
    return 'UNTRUE';
  }

  /**
   * Get operator window for harmonic tier
   * With S₃ enabled: applies cyclic rotation based on z
   */
  getOperatorWindow(harmonic, z = 0.5) {
    if (this.enableS3Symmetry && S3) {
      return S3.generateS3OperatorWindow(harmonic, z);
    }
    return this.operatorWindows[harmonic] ?? [];
  }

  /**
   * Get recommended operators for current z
   */
  getOperatorsForZ(z) {
    const harmonic = this.harmonicFromZ(z);
    return this.getOperatorWindow(harmonic, z);
  }

  /**
   * Compute ΔS_neg value
   * With extended module: uses full implementation
   */
  computeDeltaSNeg(z) {
    if (this.enableExtendedNegentropy && Delta) {
      return Delta.computeDeltaSNeg(z);
    }
    return CONST.computeDeltaSNeg(z);
  }

  /**
   * Compute Π-regime blending weights
   * With extended module: uses full PiBlendWeights
   */
  computePiBlendWeights(z) {
    if (this.enableExtendedNegentropy && Delta) {
      return Delta.computePiBlendWeights(z);
    }
    
    // Fallback implementation
    if (z < CONST.Z_CRITICAL) {
      return { w_pi: 0.0, w_local: 1.0, in_pi_regime: false };
    }
    const s = CONST.computeDeltaSNeg(z);
    return {
      w_pi: Math.max(0, Math.min(1, s)),
      w_local: 1.0 - Math.max(0, Math.min(1, s)),
      in_pi_regime: true
    };
  }

  /**
   * Compute gate modulation parameters
   * Only available with extended ΔS_neg module
   */
  computeGateModulation(z) {
    if (this.enableExtendedNegentropy && Delta) {
      return Delta.computeGateModulation(z);
    }
    // Return neutral modulation if module not available
    return {
      coherent_coupling: 0.1,
      decoherence_rate: 0.05,
      measurement_strength: 0.02,
      entropy_target: 1.0986
    };
  }

  /**
   * Check K-formation status
   * Only available with extended ΔS_neg module
   */
  checkKFormation(z, kappa = 0.92, R = 7) {
    if (this.enableExtendedNegentropy && Delta) {
      return Delta.checkKFormation(z, kappa, R);
    }
    // Fallback: compute basic η and check threshold
    const s = this.computeDeltaSNeg(z);
    const eta = Math.sqrt(s);
    const threshold = CONST.PHI_INV;
    return {
      z,
      delta_s_neg: s,
      eta,
      threshold,
      formed: kappa >= 0.92 && eta >= threshold && R >= 7,
      margin: eta - threshold
    };
  }

  /**
   * Compute operator weights with S₃ parity integration
   */
  computeOperatorWeights(z, availableOperators = null) {
    const harmonic = this.harmonicFromZ(z);
    const windowOps = this.getOperatorWindow(harmonic, z);
    const truthChannel = this.truthChannelFromZ(z);
    const ops = availableOperators ?? Object.values(APL_OPERATORS);
    
    // Use S₃ weights if enabled
    if (this.enableS3Symmetry && S3) {
      const s3Weights = S3.computeS3Weights(ops, z);
      
      // Merge with tier preference
      const weights = {};
      for (const op of ops) {
        const normalized = normalizeOperator(op);
        const isPreferred = windowOps.includes(normalized);
        const tierBoost = isPreferred ? 1.3 : 0.85;
        weights[normalized] = (s3Weights[normalized] ?? 1.0) * tierBoost;
      }
      return weights;
    }
    
    // Fallback: basic truth bias weighting
    const weights = {};
    const truthBias = CONST.TRUTH_BIAS[truthChannel] ?? {};
    
    for (const op of ops) {
      const normalized = normalizeOperator(op);
      const isPreferred = windowOps.includes(normalized);
      const baseWeight = isPreferred ? CONST.OPERATOR_PREFERRED_WEIGHT : CONST.OPERATOR_DEFAULT_WEIGHT;
      const truthMod = truthBias[normalized] ?? 1.0;
      weights[normalized] = baseWeight * truthMod;
    }
    
    return weights;
  }

  /**
   * Score operator for coherence objective (synthesis heuristic)
   * Only available with extended ΔS_neg module
   */
  scoreOperatorForCoherence(operator, z, objective = 'maximize') {
    if (this.enableExtendedNegentropy && Delta) {
      const obj = Delta.CoherenceObjective[objective.toUpperCase()] ?? Delta.CoherenceObjective.MAXIMIZE;
      return Delta.scoreOperatorForCoherence(operator, z, obj);
    }
    // Fallback: return neutral score
    return 1.0;
  }

  /**
   * Select best operator for coherence objective
   */
  selectCoherenceOperator(availableOperators, z, objective = 'maximize') {
    if (this.enableExtendedNegentropy && Delta) {
      const obj = Delta.CoherenceObjective[objective.toUpperCase()] ?? Delta.CoherenceObjective.MAXIMIZE;
      return Delta.selectCoherenceOperator(availableOperators, z, obj);
    }
    // Fallback: return first operator
    return { operator: availableOperators[0], score: 1.0 };
  }

  /**
   * Generate complete helix description for a z coordinate
   * Enhanced with S₃ and ΔS_neg integration
   */
  describe(z) {
    const value = Number.isFinite(z) ? z : 0;
    const clamped = Math.max(0, Math.min(1, value));
    
    const harmonic = this.harmonicFromZ(clamped);
    const operators = this.getOperatorWindow(harmonic, clamped);
    const truthChannel = this.truthChannelFromZ(clamped);
    const deltaSneg = this.computeDeltaSNeg(clamped);
    const piBlend = this.computePiBlendWeights(clamped);
    
    // Base description
    const description = {
      z: clamped,
      harmonic,
      operators,
      truthChannel,
      deltaSneg,
      piWeight: piBlend.w_pi,
      localWeight: piBlend.w_local,
      inPiRegime: piBlend.in_pi_regime,
      phase: CONST.getPhase(clamped),
      muClass: CONST.classifyMu(clamped),
      t6Gate: this.getT6Gate(),
      triadUnlocked: this.triadUnlocked
    };
    
    // Add S₃ data if enabled
    if (this.enableS3Symmetry && S3) {
      description.s3 = {
        enabled: true,
        rotationIndex: S3.rotationIndexFromZ(clamped),
        s3Element: S3.s3ElementFromZ(clamped),
        operatorWeights: S3.computeS3Weights(operators, clamped)
      };
    }
    
    // Add extended ΔS_neg data if enabled
    if (this.enableExtendedNegentropy && Delta) {
      description.extended = {
        enabled: true,
        deltaSNegDerivative: Delta.computeDeltaSNegDerivative(clamped),
        deltaSNegSigned: Delta.computeDeltaSNegSigned(clamped),
        eta: Delta.computeEta(clamped),
        gateModulation: Delta.computeGateModulation(clamped),
        kFormation: Delta.checkKFormation(clamped),
        geometry: Delta.computeHexPrismGeometry(clamped)
      };
    }
    
    return description;
  }

  /**
   * Compute full ΔS_neg state (comprehensive)
   */
  computeFullState(z, kappa = 0.92, R = 7) {
    if (this.enableExtendedNegentropy && Delta) {
      return Delta.computeFullState(z, kappa, R);
    }
    // Minimal fallback
    const s = this.computeDeltaSNeg(z);
    return {
      z,
      delta_s_neg: s,
      delta_s_neg_derivative: 0,
      delta_s_neg_signed: 0,
      eta: Math.sqrt(s),
      geometry: null,
      gate_modulation: this.computeGateModulation(z),
      pi_blend: this.computePiBlendWeights(z),
      k_formation: this.checkKFormation(z, kappa, R)
    };
  }

  /**
   * Dynamic operator window update
   */
  updateOperatorWindow(harmonicLabel, newOperators, options = {}) {
    if (!this.operatorWindows[harmonicLabel]) {
      console.warn(`[HelixAdvisor] Unknown harmonic label: ${harmonicLabel}`);
      return false;
    }

    const normalized = newOperators.map(normalizeOperator);
    const validOps = Object.values(APL_OPERATORS);
    const validNormalized = normalized.filter(op => validOps.includes(op));
    
    if (validNormalized.length === 0) {
      console.warn(`[HelixAdvisor] No valid operators provided for ${harmonicLabel}`);
      return false;
    }

    let updatedWindow;
    if (options.append) {
      const current = this.operatorWindows[harmonicLabel];
      updatedWindow = [...new Set([...current, ...validNormalized])];
    } else if (options.prepend) {
      const current = this.operatorWindows[harmonicLabel];
      updatedWindow = [...new Set([...validNormalized, ...current])];
    } else {
      updatedWindow = validNormalized;
    }

    const oldWindow = [...this.operatorWindows[harmonicLabel]];
    this.operatorWindows[harmonicLabel] = updatedWindow;
    
    this._windowModifications.push({
      timestamp: Date.now(),
      harmonic: harmonicLabel,
      oldWindow,
      newWindow: updatedWindow,
      source: options.source ?? 'unknown'
    });

    return true;
  }

  /**
   * Reset operator windows to defaults
   */
  resetOperatorWindows() {
    this.operatorWindows = JSON.parse(JSON.stringify(this._originalWindows));
    this._windowModifications.push({
      timestamp: Date.now(),
      harmonic: 'ALL',
      action: 'RESET',
      source: 'resetOperatorWindows'
    });
  }

  /**
   * Check if operator is valid for current z
   */
  isOperatorValidForZ(operator, z) {
    const normalized = normalizeOperator(operator);
    const allowed = this.getOperatorsForZ(z);
    return allowed.includes(normalized);
  }

  /**
   * Get modification history
   */
  getModificationHistory() {
    return [...this._windowModifications];
  }

  /**
   * Get complete state for inspection
   */
  getState() {
    return {
      triadUnlocked: this.triadUnlocked,
      triadCompletions: this.triadCompletions,
      t6Gate: this.getT6Gate(),
      enableS3Symmetry: this.enableS3Symmetry,
      enableExtendedNegentropy: this.enableExtendedNegentropy,
      s3Available: !!S3,
      deltaExtendedAvailable: !!Delta,
      timeHarmonics: [...this.timeHarmonics],
      operatorWindows: JSON.parse(JSON.stringify(this.operatorWindows)),
      modificationCount: this._windowModifications.length
    };
  }
}

// ============================================================================
// ALPHA TOKEN SYNTHESIZER (Updated)
// ============================================================================

class AlphaTokenSynthesizer {
  constructor() {
    this.sentences = [
      { id: 'A1', direction: 'd', operator: '()', machine: 'Conductor', domain: 'geometry', regime: 'Isotropic lattices under collapse' },
      { id: 'A3', direction: 'u', operator: '^', machine: 'Oscillator', domain: 'wave', regime: 'Amplified vortex-rich waves' },
      { id: 'A4', direction: 'm', operator: '×', machine: 'Encoder', domain: 'chemistry', regime: 'Helical information carriers' },
      { id: 'A5', direction: 'u', operator: '×', machine: 'Catalyst', domain: 'chemistry', regime: 'Fractal polymer branching' },
      { id: 'A6', direction: 'u', operator: '+', machine: 'Reactor', domain: 'wave', regime: 'Jet-like coherent grouping' },
      { id: 'A7', direction: 'u', operator: '÷', machine: 'Reactor', domain: 'wave', regime: 'Stochastic decohered waves' },
      { id: 'A8', direction: 'm', operator: '()', machine: 'Filter', domain: 'wave', regime: 'Adaptive boundary tuning' }
    ];
  }

  renderToken(sentence) {
    return `${sentence.direction}${sentence.operator}|${sentence.machine}|${sentence.domain}`;
  }

  findMatchingSentences(operators, options = {}) {
    const normalized = operators.map(normalizeOperator);
    let matches = this.sentences.filter(s => normalized.includes(s.operator));
    
    if (options.domain) {
      matches = matches.filter(s => s.domain === options.domain);
    }
    if (options.machine) {
      matches = matches.filter(s => s.machine.toLowerCase().includes(options.machine.toLowerCase()));
    }
    if (options.direction) {
      matches = matches.filter(s => s.direction === options.direction);
    }
    
    return matches;
  }

  synthesize(helixHints, options = {}) {
    const matches = this.findMatchingSentences(helixHints.operators, options);
    
    if (matches.length === 0) {
      return null;
    }
    
    // Use S₃ weights if available
    let selected;
    if (helixHints.s3?.operatorWeights) {
      // Weight matches by S₃ operator weights
      let bestMatch = matches[0];
      let bestWeight = 0;
      for (const m of matches) {
        const weight = helixHints.s3.operatorWeights[m.operator] ?? 1.0;
        if (weight > bestWeight) {
          bestWeight = weight;
          bestMatch = m;
        }
      }
      selected = bestMatch;
    } else if (helixHints.truthChannel === 'TRUE' && matches.some(s => s.direction === 'u')) {
      selected = matches.find(s => s.direction === 'u') ?? matches[0];
    } else if (helixHints.truthChannel === 'UNTRUE' && matches.some(s => s.direction === 'd')) {
      selected = matches.find(s => s.direction === 'd') ?? matches[0];
    } else {
      selected = matches[0];
    }
    
    return {
      ...selected,
      token: this.renderToken(selected),
      helixHints
    };
  }
}

// ============================================================================
// EXPORTS
// ============================================================================

module.exports = {
  HelixOperatorAdvisor,
  AlphaTokenSynthesizer,
  APL_OPERATORS,
  OPERATOR_ALIASES,
  BASE_OPERATOR_WINDOWS,
  normalizeOperator,
  // Feature flags
  ENABLE_S3_SYMMETRY,
  ENABLE_EXTENDED_NEGENTROPY
};
