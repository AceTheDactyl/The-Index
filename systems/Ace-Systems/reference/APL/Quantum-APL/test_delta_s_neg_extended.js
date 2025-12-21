/**
 * Test Module: Extended ΔS_neg JavaScript Implementation
 * ======================================================
 * 
 * Tests for cross-language parity with Python implementation.
 * 
 * @version 1.0.0
 */

'use strict';

const Delta = require('./delta_s_neg_extended');

// ============================================================================
// TEST UTILITIES
// ============================================================================

class TestResults {
  constructor() {
    this.passed = 0;
    this.failed = 0;
    this.errors = [];
  }

  ok(message) {
    this.passed++;
    console.log(`  ✓ ${message}`);
  }

  fail(message, detail = '') {
    this.failed++;
    this.errors.push(`${message}: ${detail}`);
    console.log(`  ✗ ${message}`);
    if (detail) console.log(`    → ${detail}`);
  }

  summary() {
    console.log('\n' + '='.repeat(60));
    console.log(`RESULTS: ${this.passed} passed, ${this.failed} failed`);
    if (this.errors.length > 0) {
      console.log('Failures:');
      for (const e of this.errors) {
        console.log(`  - ${e}`);
      }
    }
    console.log('='.repeat(60));
    return this.failed === 0;
  }
}

function assertClose(a, b, tol = 1e-10) {
  return Math.abs(a - b) < tol;
}

function section(title) {
  console.log(`\n${'─'.repeat(60)}`);
  console.log(`  ${title}`);
  console.log(`${'─'.repeat(60)}`);
}

// ============================================================================
// ΔS_neg TESTS
// ============================================================================

function testDeltaSNegBasic(results) {
  section('ΔS_neg Basic Properties');

  // Peak at z_c
  const s_at_lens = Delta.computeDeltaSNeg(Delta.Z_CRITICAL);
  if (assertClose(s_at_lens, 1.0, 1e-6)) {
    results.ok('ΔS_neg(z_c) = 1.0 (peak at lens)');
  } else {
    results.fail('Peak at lens', `expected 1.0, got ${s_at_lens}`);
  }

  // Bounded in [0, 1]
  const test_points = [0.0, 0.3, 0.5, 0.7, Delta.Z_CRITICAL, 0.9, 1.0];
  const all_bounded = test_points.every(z => {
    const s = Delta.computeDeltaSNeg(z);
    return s >= 0 && s <= 1;
  });
  if (all_bounded) {
    results.ok('ΔS_neg bounded in [0, 1]');
  } else {
    results.fail('Boundedness', 'value outside [0, 1]');
  }

  // Symmetric around z_c
  const offset = 0.1;
  const s_above = Delta.computeDeltaSNeg(Delta.Z_CRITICAL + offset);
  const s_below = Delta.computeDeltaSNeg(Delta.Z_CRITICAL - offset);
  if (assertClose(s_above, s_below, 1e-10)) {
    results.ok('Symmetric: ΔS_neg(z_c+0.1) ≈ ΔS_neg(z_c-0.1)');
  } else {
    results.fail('Symmetry', `above=${s_above}, below=${s_below}`);
  }
}

function testDeltaSNegDerivative(results) {
  section('ΔS_neg Derivative');

  // Zero at z_c
  const d_at_lens = Delta.computeDeltaSNegDerivative(Delta.Z_CRITICAL);
  if (assertClose(d_at_lens, 0.0, 1e-6)) {
    results.ok('d(ΔS_neg)/dz = 0 at z_c (critical point)');
  } else {
    results.fail('Derivative at z_c', `expected 0, got ${d_at_lens}`);
  }

  // Positive below z_c (increasing toward lens)
  const d_below = Delta.computeDeltaSNegDerivative(Delta.Z_CRITICAL - 0.1);
  if (d_below > 0) {
    results.ok('d(ΔS_neg)/dz > 0 below z_c');
  } else {
    results.fail('Derivative below z_c', `expected positive, got ${d_below}`);
  }

  // Negative above z_c (decreasing away from lens)
  const d_above = Delta.computeDeltaSNegDerivative(Delta.Z_CRITICAL + 0.1);
  if (d_above < 0) {
    results.ok('d(ΔS_neg)/dz < 0 above z_c');
  } else {
    results.fail('Derivative above z_c', `expected negative, got ${d_above}`);
  }
}

function testDeltaSNegSigned(results) {
  section('Signed ΔS_neg');

  // Zero at z_c
  const signed_at_lens = Delta.computeDeltaSNegSigned(Delta.Z_CRITICAL);
  if (assertClose(signed_at_lens, 0.0, 1e-6)) {
    results.ok('Signed ΔS_neg = 0 at z_c');
  } else {
    results.fail('Signed at z_c', `expected 0, got ${signed_at_lens}`);
  }

  // Positive above z_c
  const signed_above = Delta.computeDeltaSNegSigned(Delta.Z_CRITICAL + 0.05);
  if (signed_above > 0) {
    results.ok('Signed ΔS_neg > 0 above z_c');
  } else {
    results.fail('Signed above z_c', `expected positive, got ${signed_above}`);
  }

  // Negative below z_c
  const signed_below = Delta.computeDeltaSNegSigned(Delta.Z_CRITICAL - 0.05);
  if (signed_below < 0) {
    results.ok('Signed ΔS_neg < 0 below z_c');
  } else {
    results.fail('Signed below z_c', `expected negative, got ${signed_below}`);
  }
}

// ============================================================================
// GEOMETRY TESTS
// ============================================================================

function testHexPrismGeometry(results) {
  section('Hex-Prism Geometry');

  const g_lens = Delta.computeHexPrismGeometry(Delta.Z_CRITICAL);
  const g_far = Delta.computeHexPrismGeometry(0.5);

  // At lens: R = R_max - β·1 = 0.85 - 0.25 = 0.6
  if (assertClose(g_lens.radius, 0.6, 1e-6)) {
    results.ok('R(z_c) = 0.6 (radius contracts at lens)');
  } else {
    results.fail('Radius at lens', `expected 0.6, got ${g_lens.radius}`);
  }

  // At lens: H = H_min + γ·1 = 0.12 + 0.18 = 0.3
  if (assertClose(g_lens.height, 0.3, 1e-6)) {
    results.ok('H(z_c) = 0.3 (height elongates at lens)');
  } else {
    results.fail('Height at lens', `expected 0.3, got ${g_lens.height}`);
  }

  // Radius larger far from lens
  if (g_far.radius > g_lens.radius) {
    results.ok('Radius larger far from lens');
  } else {
    results.fail('Radius comparison', `far=${g_far.radius}, lens=${g_lens.radius}`);
  }

  // Height smaller far from lens
  if (g_far.height < g_lens.height) {
    results.ok('Height smaller far from lens');
  } else {
    results.fail('Height comparison', `far=${g_far.height}, lens=${g_lens.height}`);
  }
}

// ============================================================================
// K-FORMATION TESTS
// ============================================================================

function testKFormation(results) {
  section('K-Formation Gating');

  const k_lens = Delta.checkKFormation(Delta.Z_CRITICAL);
  const k_far = Delta.checkKFormation(0.5);

  if (k_lens.formed) {
    results.ok('K-formation at z_c (η=1 > φ⁻¹)');
  } else {
    results.fail('K-formation at z_c', `η=${k_lens.eta}, should be formed`);
  }

  if (!k_far.formed) {
    results.ok('No K-formation far from lens');
  } else {
    results.fail('K-formation at z=0.5', `unexpected formation, η=${k_far.eta}`);
  }

  if (assertClose(k_lens.threshold, Delta.PHI_INV, 1e-6)) {
    results.ok(`Threshold = φ⁻¹ ≈ ${Delta.PHI_INV.toFixed(6)}`);
  } else {
    results.fail('Threshold', `expected ${Delta.PHI_INV}, got ${k_lens.threshold}`);
  }
}

// ============================================================================
// Π-REGIME BLENDING TESTS
// ============================================================================

function testPiBlending(results) {
  section('Π-Regime Blending');

  const blend_below = Delta.computePiBlendWeights(Delta.Z_CRITICAL - 0.1);
  
  if (blend_below.w_pi === 0 && blend_below.w_local === 1) {
    results.ok('Below z_c: w_π=0, w_local=1 (pure local)');
  } else {
    results.fail('Below z_c', `w_π=${blend_below.w_pi}`);
  }

  if (!blend_below.in_pi_regime) {
    results.ok('Below z_c: not in Π-regime');
  } else {
    results.fail('Π-regime flag below z_c', 'should be false');
  }

  const blend_lens = Delta.computePiBlendWeights(Delta.Z_CRITICAL);
  
  if (assertClose(blend_lens.w_pi, 1.0, 1e-6)) {
    results.ok('At z_c: w_π = 1.0');
  } else {
    results.fail('At z_c', `w_π=${blend_lens.w_pi}`);
  }

  if (blend_lens.in_pi_regime) {
    results.ok('At z_c: in Π-regime');
  } else {
    results.fail('Π-regime flag at z_c', 'should be true');
  }

  const blend_above = Delta.computePiBlendWeights(Delta.Z_CRITICAL + 0.05);
  
  if (blend_above.w_pi > 0 && blend_above.w_pi < 1) {
    results.ok(`Above z_c: 0 < w_π < 1 (w_π=${blend_above.w_pi.toFixed(4)})`);
  } else {
    results.fail('Above z_c', `w_π=${blend_above.w_pi} not in (0,1)`);
  }
}

// ============================================================================
// GATE MODULATION TESTS
// ============================================================================

function testGateModulation(results) {
  section('Gate Modulation');

  const mod_far = Delta.computeGateModulation(0.5);
  const mod_lens = Delta.computeGateModulation(Delta.Z_CRITICAL);

  if (mod_lens.coherent_coupling > mod_far.coherent_coupling) {
    results.ok('Coherent coupling increases at lens');
  } else {
    results.fail('Coherent coupling', `lens=${mod_lens.coherent_coupling}, far=${mod_far.coherent_coupling}`);
  }

  if (mod_lens.decoherence_rate < mod_far.decoherence_rate) {
    results.ok('Decoherence rate decreases at lens');
  } else {
    results.fail('Decoherence rate', `lens=${mod_lens.decoherence_rate}, far=${mod_far.decoherence_rate}`);
  }

  if (mod_lens.entropy_target < mod_far.entropy_target) {
    results.ok('Entropy target decreases at lens');
  } else {
    results.fail('Entropy target', `lens=${mod_lens.entropy_target}, far=${mod_far.entropy_target}`);
  }
}

// ============================================================================
// COHERENCE SYNTHESIS TESTS
// ============================================================================

function testCoherenceSynthesis(results) {
  section('Coherence Synthesis Heuristics');

  const operators = ['()', '×', '^', '÷', '+', '−'];
  const z_below = 0.5;

  // MAXIMIZE below lens
  const scores_max = {};
  for (const op of operators) {
    scores_max[op] = Delta.scoreOperatorForCoherence(op, z_below, Delta.CoherenceObjective.MAXIMIZE);
  }

  const constructive = ['^', '+', '×'];
  const dissipative = ['÷', '−'];

  const avg_constructive = constructive.reduce((s, op) => s + scores_max[op], 0) / 3;
  const avg_dissipative = dissipative.reduce((s, op) => s + scores_max[op], 0) / 2;

  if (avg_constructive > avg_dissipative) {
    results.ok('MAXIMIZE below lens favors constructive operators');
  } else {
    results.fail('MAXIMIZE heuristic', `constructive=${avg_constructive}, dissipative=${avg_dissipative}`);
  }

  // MINIMIZE
  const scores_min = {};
  for (const op of operators) {
    scores_min[op] = Delta.scoreOperatorForCoherence(op, z_below, Delta.CoherenceObjective.MINIMIZE);
  }

  const avg_constructive_min = constructive.reduce((s, op) => s + scores_min[op], 0) / 3;
  const avg_dissipative_min = dissipative.reduce((s, op) => s + scores_min[op], 0) / 2;

  if (avg_dissipative_min > avg_constructive_min) {
    results.ok('MINIMIZE favors dissipative operators');
  } else {
    results.fail('MINIMIZE heuristic', `constructive=${avg_constructive_min}, dissipative=${avg_dissipative_min}`);
  }
}

// ============================================================================
// FULL STATE INTEGRATION TESTS
// ============================================================================

function testFullStateIntegration(results) {
  section('Full State Integration');

  const state = Delta.computeFullState(Delta.Z_CRITICAL);

  if (state.z === Delta.Z_CRITICAL) {
    results.ok('z-coordinate preserved');
  } else {
    results.fail('z-coordinate', `expected ${Delta.Z_CRITICAL}, got ${state.z}`);
  }

  if (assertClose(state.delta_s_neg, 1.0, 1e-6)) {
    results.ok('ΔS_neg computed correctly');
  } else {
    results.fail('ΔS_neg', `expected 1.0, got ${state.delta_s_neg}`);
  }

  if (state.geometry !== null && state.geometry !== undefined) {
    results.ok('Geometry computed');
  } else {
    results.fail('Geometry', 'missing');
  }

  if (state.gate_modulation !== null && state.gate_modulation !== undefined) {
    results.ok('Gate modulation computed');
  } else {
    results.fail('Gate modulation', 'missing');
  }

  if (state.pi_blend.in_pi_regime) {
    results.ok('Π-blend computed and correct');
  } else {
    results.fail('Π-blend', 'should be in Π-regime at z_c');
  }

  if (state.k_formation.formed) {
    results.ok('K-formation computed and correct');
  } else {
    results.fail('K-formation', 'should be formed at z_c');
  }
}

// ============================================================================
// CROSS-LANGUAGE PARITY TESTS
// ============================================================================

function testCrossLanguageParity(results) {
  section('Cross-Language Parity (JS vs Python expected values)');

  // These values are computed from the Python implementation
  const z_test = 0.85;
  
  // ΔS_neg at z=0.85 (Python: 0.990797317882842)
  const s = Delta.computeDeltaSNeg(z_test);
  if (assertClose(s, 0.990797317882842, 1e-12)) {
    results.ok(`ΔS_neg(0.85) = ${s.toFixed(15)} matches Python`);
  } else {
    results.fail('ΔS_neg parity', `expected 0.990797317882842, got ${s}`);
  }

  // η at z=0.85 with α=0.5 (Python: 0.995388023779090)
  const eta = Delta.computeEta(z_test);
  if (assertClose(eta, 0.995388023779090, 1e-12)) {
    results.ok(`η(0.85) = ${eta.toFixed(15)} matches Python`);
  } else {
    results.fail('η parity', `expected 0.995388023779090, got ${eta}`);
  }

  // Z_CRITICAL exact value
  if (assertClose(Delta.Z_CRITICAL, 0.8660254037844386, 1e-15)) {
    results.ok(`Z_CRITICAL = ${Delta.Z_CRITICAL} exact`);
  } else {
    results.fail('Z_CRITICAL', `expected √3/2, got ${Delta.Z_CRITICAL}`);
  }

  // PHI_INV exact value
  if (assertClose(Delta.PHI_INV, 0.6180339887498949, 1e-14)) {
    results.ok(`PHI_INV = ${Delta.PHI_INV} exact`);
  } else {
    results.fail('PHI_INV', `expected 1/φ, got ${Delta.PHI_INV}`);
  }
}

// ============================================================================
// MAIN
// ============================================================================

function main() {
  console.log('='.repeat(60));
  console.log('EXTENDED ΔS_neg JAVASCRIPT TESTS');
  console.log('='.repeat(60));

  const results = new TestResults();

  // Core ΔS_neg tests
  testDeltaSNegBasic(results);
  testDeltaSNegDerivative(results);
  testDeltaSNegSigned(results);

  // Geometry tests
  testHexPrismGeometry(results);

  // Gating tests
  testKFormation(results);
  testPiBlending(results);
  testGateModulation(results);

  // Synthesis tests
  testCoherenceSynthesis(results);

  // Integration tests
  testFullStateIntegration(results);

  // Cross-language parity
  testCrossLanguageParity(results);

  // Summary
  const success = results.summary();

  process.exit(success ? 0 : 1);
}

main();
