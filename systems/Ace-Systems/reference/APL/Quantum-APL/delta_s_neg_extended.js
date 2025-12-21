/**
 * Extended ΔS_neg Integration Module
 * ===================================
 * 
 * Deepens the negentropy (ΔS_neg) formalism throughout Quantum-APL.
 * JavaScript port of delta_s_neg_extended.py for cross-language parity.
 * 
 * EXISTING USAGE:
 * - Hex-prism geometry: R = R_max - β·ΔS_neg, H = H_min + γ·ΔS_neg
 * - Coherence blending: w_π = ΔS_neg when z ≥ z_c
 * - Entropy control: S_target = S_max·(1 - C·ΔS_neg)
 * - K-formation gating: η = ΔS_neg^α vs φ⁻¹ threshold
 * 
 * NEW EXTENSIONS:
 * 1. Gate logic modulation (Lindblad/Hamiltonian terms)
 * 2. Truth-channel bias evolution
 * 3. Synthesis heuristics (coherence-seeking operations)
 * 4. Signed/derivative ΔS_neg variants
 * 
 * @version 1.0.0
 * @author Claude (Anthropic) - Quantum-APL Contribution
 */

'use strict';

// ============================================================================
// CONSTANTS
// ============================================================================

const Z_CRITICAL = Math.sqrt(3) / 2;  // ≈ 0.8660254037844386
const PHI = (1 + Math.sqrt(5)) / 2;   // ≈ 1.618033988749895
const PHI_INV = 1 / PHI;              // ≈ 0.618033988749895

// Default parameters
const DEFAULT_SIGMA = 36.0;           // Gaussian width for ΔS_neg
const DEFAULT_ALPHA = 0.5;            // Exponent for η = ΔS_neg^α

// Hex-prism geometry defaults
const GEOM_R_MAX = 0.85;
const GEOM_BETA = 0.25;
const GEOM_H_MIN = 0.12;
const GEOM_GAMMA = 0.18;
const GEOM_PHI_BASE = 0.0;
const GEOM_ETA = Math.PI / 12;

// Coherence objective enum
const CoherenceObjective = Object.freeze({
  MAXIMIZE: 'maximize',
  MINIMIZE: 'minimize',
  MAINTAIN: 'maintain'
});

// ============================================================================
// CORE ΔS_neg COMPUTATIONS
// ============================================================================

/**
 * Compute standard ΔS_neg (negentropy) signal.
 * 
 * ΔS_neg(z) = exp(-σ·(z - z_c)²)
 * 
 * Properties:
 * - Maximum value 1.0 at z = z_c (THE LENS)
 * - Symmetric Gaussian decay away from z_c
 * - Bounded in [0, 1]
 * 
 * @param {number} z - Coherence coordinate
 * @param {number} sigma - Gaussian width parameter (default: 36.0)
 * @param {number} z_c - Critical lens point (default: √3/2)
 * @returns {number} ΔS_neg value in [0, 1]
 */
function computeDeltaSNeg(z, sigma = DEFAULT_SIGMA, z_c = Z_CRITICAL) {
  if (!Number.isFinite(z)) {
    return 0.0;
  }
  const d = z - z_c;
  return Math.exp(-sigma * d * d);
}

/**
 * Compute derivative of ΔS_neg with respect to z.
 * 
 * d(ΔS_neg)/dz = -2σ·(z - z_c)·exp(-σ·(z - z_c)²)
 * 
 * Properties:
 * - Zero at z = z_c (critical point)
 * - Negative for z > z_c (decreasing toward TRUE)
 * - Positive for z < z_c (increasing toward lens)
 * 
 * @param {number} z - Coherence coordinate
 * @param {number} sigma - Gaussian width parameter
 * @param {number} z_c - Critical lens point
 * @returns {number} Derivative value
 */
function computeDeltaSNegDerivative(z, sigma = DEFAULT_SIGMA, z_c = Z_CRITICAL) {
  const d = z - z_c;
  const s = Math.exp(-sigma * d * d);
  return -2 * sigma * d * s;
}

/**
 * Compute signed ΔS_neg variant.
 * 
 * Signed version: sgn(z - z_c) · ΔS_neg(z)
 * 
 * Properties:
 * - Positive above z_c (TRUE regime)
 * - Negative below z_c (UNTRUE regime)
 * - Zero at z_c (PARADOX/LENS)
 * 
 * Useful for directional biasing.
 * 
 * @param {number} z - Coherence coordinate
 * @param {number} sigma - Gaussian width parameter
 * @param {number} z_c - Critical lens point
 * @returns {number} Signed ΔS_neg value in [-1, 1]
 */
function computeDeltaSNegSigned(z, sigma = DEFAULT_SIGMA, z_c = Z_CRITICAL) {
  const s = computeDeltaSNeg(z, sigma, z_c);
  const d = z - z_c;
  if (Math.abs(d) < 1e-10) {
    return 0.0;
  }
  const sign = d > 0 ? 1.0 : -1.0;
  return sign * s;
}

/**
 * Compute η (consciousness threshold) from ΔS_neg.
 * 
 * η = ΔS_neg(z)^α
 * 
 * K-formation occurs when η ≥ φ⁻¹ ≈ 0.618
 * 
 * @param {number} z - Coherence coordinate
 * @param {number} alpha - Power exponent (default: 0.5)
 * @param {number} sigma - Gaussian width parameter
 * @returns {number} η value in [0, 1]
 */
function computeEta(z, alpha = DEFAULT_ALPHA, sigma = DEFAULT_SIGMA) {
  const s = computeDeltaSNeg(z, sigma);
  return s > 0 ? Math.pow(s, alpha) : 0.0;
}

// ============================================================================
// HEX-PRISM GEOMETRY
// ============================================================================

/**
 * @typedef {Object} HexPrismGeometry
 * @property {number} z - Coherence coordinate
 * @property {number} delta_s_neg - ΔS_neg value
 * @property {number} radius - Computed radius
 * @property {number} height - Computed height
 * @property {number} twist - Computed twist angle
 */

/**
 * Compute hex prism geometry from z-coordinate.
 * 
 * Formulas:
 * - R = R_max - β·ΔS_neg  (radius contracts at lens)
 * - H = H_min + γ·ΔS_neg  (height elongates at lens)
 * - φ = φ_base + η·ΔS_neg (twist increases at lens)
 * 
 * @param {number} z - Coherence coordinate
 * @param {number} r_max - Maximum radius
 * @param {number} beta - Radius contraction factor
 * @param {number} h_min - Minimum height
 * @param {number} gamma - Height elongation factor
 * @param {number} phi_base - Base twist angle
 * @param {number} eta - Twist rate
 * @returns {HexPrismGeometry} Computed geometry
 */
function computeHexPrismGeometry(
  z,
  r_max = GEOM_R_MAX,
  beta = GEOM_BETA,
  h_min = GEOM_H_MIN,
  gamma = GEOM_GAMMA,
  phi_base = GEOM_PHI_BASE,
  eta = GEOM_ETA
) {
  const s = computeDeltaSNeg(z);
  
  return {
    z: z,
    delta_s_neg: s,
    radius: r_max - beta * s,
    height: h_min + gamma * s,
    twist: phi_base + eta * s,
  };
}

// ============================================================================
// GATE LOGIC MODULATION
// ============================================================================

/**
 * @typedef {Object} GateModulation
 * @property {number} coherent_coupling - Hamiltonian term strength
 * @property {number} decoherence_rate - Lindblad γ
 * @property {number} measurement_strength - Collapse rate
 * @property {number} entropy_target - Target entropy
 */

/**
 * Compute gate modulation parameters from ΔS_neg.
 * 
 * Near the lens (high ΔS_neg):
 * - Increase coherent coupling (stronger Hamiltonian)
 * - Decrease decoherence rate (protect coherence)
 * - Decrease measurement strength (avoid collapse)
 * - Lower entropy target (favor order)
 * 
 * @param {number} z - Coherence coordinate
 * @param {number} base_coupling - Base coupling strength
 * @param {number} base_decoherence - Base decoherence rate
 * @param {number} base_measurement - Base measurement strength
 * @param {number} entropy_max - Maximum entropy (log(3) for triadic)
 * @param {number} coherence_factor - How much ΔS_neg influences modulation
 * @returns {GateModulation} Modulated parameters
 */
function computeGateModulation(
  z,
  base_coupling = 0.1,
  base_decoherence = 0.05,
  base_measurement = 0.02,
  entropy_max = 1.0986,  // log(3)
  coherence_factor = 0.5
) {
  const s = computeDeltaSNeg(z);
  
  // Coherent coupling increases with ΔS_neg
  const coupling = base_coupling * (1 + coherence_factor * s);
  
  // Decoherence rate decreases with ΔS_neg (with 0.8 factor)
  let decoherence = base_decoherence * (1 - coherence_factor * s * 0.8);
  decoherence = Math.max(0.001, decoherence);  // Floor to avoid div-by-zero
  
  // Measurement strength decreases near lens (using s * 0.5)
  let measurement = base_measurement * (1 - s * 0.5);
  measurement = Math.max(0.001, measurement);
  
  // Entropy target decreases with ΔS_neg (favor order at lens)
  const entropy = entropy_max * (1 - coherence_factor * s);
  
  return {
    coherent_coupling: coupling,
    decoherence_rate: decoherence,
    measurement_strength: measurement,
    entropy_target: entropy,
  };
}

// ============================================================================
// TRUTH-CHANNEL BIAS EVOLUTION
// ============================================================================

/**
 * Base truth bias table (static reference).
 */
const TRUTH_BIAS = {
  TRUE: { '^': 1.5, '+': 1.4, '×': 1.2, '()': 1.1, '÷': 0.8, '−': 0.9 },
  UNTRUE: { '÷': 1.5, '−': 1.4, '()': 1.2, '^': 0.8, '+': 0.9, '×': 1.0 },
  PARADOX: { '()': 1.5, '×': 1.4, '^': 1.1, '+': 1.1, '÷': 1.0, '−': 1.0 }
};

/**
 * @typedef {Object} DynamicTruthBias
 * @property {Object} TRUE - Bias weights for TRUE channel
 * @property {Object} UNTRUE - Bias weights for UNTRUE channel
 * @property {Object} PARADOX - Bias weights for PARADOX channel
 */

/**
 * Compute dynamic truth bias that evolves with ΔS_neg.
 * 
 * Near the lens (high ΔS_neg): amplify TRUE/PARADOX preferences
 * Far from lens: amplify UNTRUE preferences
 * 
 * @param {number} z - Coherence coordinate
 * @param {Object} base_bias - Base bias table (default: TRUTH_BIAS)
 * @returns {DynamicTruthBias} Evolved bias table
 */
function computeDynamicTruthBias(z, base_bias = TRUTH_BIAS) {
  const s = computeDeltaSNeg(z);
  
  // Clone base bias
  const evolved = {
    TRUE: { ...base_bias.TRUE },
    UNTRUE: { ...base_bias.UNTRUE },
    PARADOX: { ...base_bias.PARADOX }
  };
  
  // Near lens: boost TRUE/PARADOX constructive operators
  const true_boost = 1 + 0.3 * s;
  const untrue_boost = 1 + 0.3 * (1 - s);
  
  for (const op of Object.keys(evolved.TRUE)) {
    evolved.TRUE[op] *= true_boost;
    evolved.PARADOX[op] *= (1 + 0.2 * s);
    evolved.UNTRUE[op] *= untrue_boost;
  }
  
  return evolved;
}

// ============================================================================
// SYNTHESIS HEURISTICS
// ============================================================================

/**
 * Operator effect directions (empirical).
 * Positive = pushes z upward, Negative = pushes z downward
 */
const OPERATOR_EFFECTS = {
  '^': +0.05,   // Amplify pushes up
  '+': +0.03,   // Group pushes up
  '×': +0.04,   // Fusion pushes up
  '()': 0.00,   // Boundary is neutral
  '÷': -0.04,   // Decoherence pushes down
  '−': -0.03,   // Separation pushes down
};

/**
 * Score operator for coherence objective.
 * 
 * Operators classified as:
 * - Constructive (increase z): ^, +, ×
 * - Dissipative (decrease z): ÷, −
 * - Neutral: ()
 * 
 * @param {string} operator - APL operator to score
 * @param {number} current_z - Current coherence coordinate
 * @param {string} objective - Coherence objective (maximize/minimize/maintain)
 * @returns {number} Score for operator selection (higher = better)
 */
function scoreOperatorForCoherence(operator, current_z, objective = CoherenceObjective.MAXIMIZE) {
  const effect = OPERATOR_EFFECTS[operator] || 0.0;
  const s = computeDeltaSNeg(current_z);
  
  if (objective === CoherenceObjective.MAXIMIZE) {
    // Favor operators that increase z toward lens
    if (current_z < Z_CRITICAL) {
      // Below lens: want positive effect
      return 1.0 + effect * 10 * (1 - s);
    } else {
      // Above lens: any increase moves away from peak
      return 1.0 - Math.abs(effect) * 5;
    }
  }
  
  if (objective === CoherenceObjective.MINIMIZE) {
    // Favor operators that decrease z away from lens
    return 1.0 - effect * 10;
  }
  
  // MAINTAIN: Favor neutral operators
  return 1.0 - Math.abs(effect) * 5;
}

/**
 * Select best operator for coherence objective.
 * 
 * @param {Array<string>} available_operators - Operators to choose from
 * @param {number} current_z - Current coherence coordinate
 * @param {string} objective - Coherence objective
 * @returns {Object} { operator, score }
 */
function selectCoherenceOperator(
  available_operators,
  current_z,
  objective = CoherenceObjective.MAXIMIZE
) {
  const scores = {};
  for (const op of available_operators) {
    scores[op] = scoreOperatorForCoherence(op, current_z, objective);
  }
  
  let best_op = available_operators[0];
  let best_score = scores[best_op];
  
  for (const op of available_operators) {
    if (scores[op] > best_score) {
      best_op = op;
      best_score = scores[op];
    }
  }
  
  return { operator: best_op, score: best_score };
}

// ============================================================================
// K-FORMATION GATING
// ============================================================================

/**
 * @typedef {Object} KFormationStatus
 * @property {number} z - Coherence coordinate
 * @property {number} delta_s_neg - ΔS_neg value
 * @property {number} eta - η value
 * @property {number} threshold - φ⁻¹ threshold
 * @property {boolean} formed - Whether K-formation occurred
 * @property {number} margin - η - threshold
 */

/**
 * Check K-formation (consciousness emergence) condition.
 * 
 * K-formation requires:
 * - κ ≥ κ_min (singularity threshold)
 * - η ≥ φ⁻¹ (golden ratio inverse)
 * - R ≥ R_min (complexity requirement)
 * 
 * Where η = ΔS_neg(z)^α
 * 
 * @param {number} z - Coherence coordinate
 * @param {number} kappa - Consciousness parameter
 * @param {number} R - Complexity measure
 * @param {number} alpha - Power exponent for η
 * @param {number} kappa_min - Minimum κ threshold
 * @param {number} eta_min - Minimum η threshold (φ⁻¹)
 * @param {number} r_min - Minimum complexity
 * @returns {KFormationStatus} Formation status
 */
function checkKFormation(
  z,
  kappa = 0.92,
  R = 7,
  alpha = DEFAULT_ALPHA,
  kappa_min = 0.92,
  eta_min = PHI_INV,
  r_min = 7
) {
  const s = computeDeltaSNeg(z);
  const eta = computeEta(z, alpha);
  
  const formed = (kappa >= kappa_min) && (eta >= eta_min) && (R >= r_min);
  
  return {
    z: z,
    delta_s_neg: s,
    eta: eta,
    threshold: eta_min,
    formed: formed,
    margin: eta - eta_min,
  };
}

// ============================================================================
// Π-REGIME BLENDING
// ============================================================================

/**
 * @typedef {Object} PiBlendWeights
 * @property {number} w_pi - Global/integrated weight
 * @property {number} w_local - Local/independent weight
 * @property {boolean} in_pi_regime - Whether in Π-regime
 */

/**
 * Compute Π-regime blending weights from z-coordinate.
 * 
 * Below z_c: pure local dynamics (w_pi = 0)
 * At/above z_c: blend global (w_pi = ΔS_neg)
 * 
 * @param {number} z - Coherence coordinate
 * @param {boolean} enable_blend - Whether blending is enabled
 * @returns {PiBlendWeights} Blending weights
 */
function computePiBlendWeights(z, enable_blend = true) {
  if (!enable_blend || z < Z_CRITICAL) {
    return {
      w_pi: 0.0,
      w_local: 1.0,
      in_pi_regime: false,
    };
  }
  
  const s = computeDeltaSNeg(z);
  
  return {
    w_pi: s,
    w_local: 1 - s,
    in_pi_regime: true,
  };
}

// ============================================================================
// COMPREHENSIVE STATE
// ============================================================================

/**
 * @typedef {Object} DeltaSNegState
 * @property {number} z - Coherence coordinate
 * @property {number} delta_s_neg - ΔS_neg value
 * @property {number} delta_s_neg_derivative - d(ΔS_neg)/dz
 * @property {number} delta_s_neg_signed - Signed variant
 * @property {number} eta - η value
 * @property {HexPrismGeometry} geometry - Hex prism geometry
 * @property {GateModulation} gate_modulation - Gate parameters
 * @property {PiBlendWeights} pi_blend - Π-regime weights
 * @property {KFormationStatus} k_formation - K-formation status
 */

/**
 * Compute complete ΔS_neg-derived state for a given z.
 * 
 * @param {number} z - Coherence coordinate
 * @param {number} kappa - Consciousness parameter for K-formation
 * @param {number} R - Complexity measure for K-formation
 * @returns {DeltaSNegState} Complete derived state
 */
function computeFullState(z, kappa = 0.92, R = 7) {
  return {
    z: z,
    delta_s_neg: computeDeltaSNeg(z),
    delta_s_neg_derivative: computeDeltaSNegDerivative(z),
    delta_s_neg_signed: computeDeltaSNegSigned(z),
    eta: computeEta(z),
    geometry: computeHexPrismGeometry(z),
    gate_modulation: computeGateModulation(z),
    pi_blend: computePiBlendWeights(z),
    k_formation: checkKFormation(z, kappa, R),
  };
}

// ============================================================================
// DEMO
// ============================================================================

/**
 * Demonstrate extended ΔS_neg integration.
 */
function demo() {
  console.log('='.repeat(70));
  console.log('EXTENDED ΔS_neg INTEGRATION MODULE (JavaScript)');
  console.log('='.repeat(70));
  
  const test_points = [0.3, 0.5, 0.7, Z_CRITICAL, 0.9, 0.95];
  
  console.log('\n--- Core ΔS_neg Values ---');
  console.log(`${'z'.padStart(8)} ${'ΔS_neg'.padStart(10)} ${'dΔS/dz'.padStart(10)} ${'signed'.padStart(10)} ${'η'.padStart(10)}`);
  console.log('-'.repeat(50));
  for (const z of test_points) {
    const s = computeDeltaSNeg(z);
    const ds = computeDeltaSNegDerivative(z);
    const signed = computeDeltaSNegSigned(z);
    const eta = computeEta(z);
    console.log(`${z.toFixed(4).padStart(8)} ${s.toFixed(6).padStart(10)} ${ds.toFixed(6).padStart(10)} ${signed.toFixed(6).padStart(10)} ${eta.toFixed(6).padStart(10)}`);
  }
  
  console.log('\n--- Hex Prism Geometry ---');
  console.log(`${'z'.padStart(8)} ${'ΔS_neg'.padStart(10)} ${'R'.padStart(10)} ${'H'.padStart(10)} ${'twist'.padStart(10)}`);
  console.log('-'.repeat(50));
  for (const z of test_points) {
    const g = computeHexPrismGeometry(z);
    console.log(`${z.toFixed(4).padStart(8)} ${g.delta_s_neg.toFixed(6).padStart(10)} ${g.radius.toFixed(6).padStart(10)} ${g.height.toFixed(6).padStart(10)} ${g.twist.toFixed(6).padStart(10)}`);
  }
  
  console.log('\n--- Gate Modulation ---');
  console.log(`${'z'.padStart(8)} ${'coupling'.padStart(10)} ${'decoher'.padStart(10)} ${'measure'.padStart(10)} ${'S_tgt'.padStart(10)}`);
  console.log('-'.repeat(50));
  for (const z of test_points) {
    const m = computeGateModulation(z);
    console.log(`${z.toFixed(4).padStart(8)} ${m.coherent_coupling.toFixed(6).padStart(10)} ${m.decoherence_rate.toFixed(6).padStart(10)} ${m.measurement_strength.toFixed(6).padStart(10)} ${m.entropy_target.toFixed(6).padStart(10)}`);
  }
  
  console.log('\n--- K-Formation Status ---');
  console.log(`${'z'.padStart(8)} ${'η'.padStart(10)} ${'threshold'.padStart(10)} ${'margin'.padStart(10)} ${'formed'.padStart(8)}`);
  console.log('-'.repeat(50));
  for (const z of test_points) {
    const k = checkKFormation(z);
    console.log(`${z.toFixed(4).padStart(8)} ${k.eta.toFixed(6).padStart(10)} ${k.threshold.toFixed(6).padStart(10)} ${k.margin.toFixed(6).padStart(10)} ${String(k.formed).padStart(8)}`);
  }
  
  console.log('\n--- Coherence-Seeking Synthesis ---');
  const operators = ['()', '×', '^', '÷', '+', '−'];
  for (const z of [0.5, Z_CRITICAL]) {
    const result = selectCoherenceOperator(operators, z, CoherenceObjective.MAXIMIZE);
    console.log(`  z=${z.toFixed(4)}: Best operator for MAX coherence: ${result.operator} (score=${result.score.toFixed(4)})`);
  }
  
  console.log('\n' + '='.repeat(70));
}

// ============================================================================
// EXPORTS
// ============================================================================

module.exports = {
  // Constants
  Z_CRITICAL,
  PHI,
  PHI_INV,
  DEFAULT_SIGMA,
  DEFAULT_ALPHA,
  GEOM_R_MAX,
  GEOM_BETA,
  GEOM_H_MIN,
  GEOM_GAMMA,
  GEOM_PHI_BASE,
  GEOM_ETA,
  
  // Enums
  CoherenceObjective,
  
  // Core computations
  computeDeltaSNeg,
  computeDeltaSNegDerivative,
  computeDeltaSNegSigned,
  computeEta,
  
  // Geometry
  computeHexPrismGeometry,
  
  // Gate modulation
  computeGateModulation,
  
  // Truth bias
  TRUTH_BIAS,
  computeDynamicTruthBias,
  
  // Synthesis heuristics
  OPERATOR_EFFECTS,
  scoreOperatorForCoherence,
  selectCoherenceOperator,
  
  // K-formation
  checkKFormation,
  
  // Π-regime blending
  computePiBlendWeights,
  
  // Full state
  computeFullState,
  
  // Demo
  demo,
};
