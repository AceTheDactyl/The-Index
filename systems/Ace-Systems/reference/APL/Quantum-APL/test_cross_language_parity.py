#!/usr/bin/env python3
"""
Cross-Language Parity Test Suite
================================

Runs JS and Python implementations side-by-side and verifies numerical parity.
Executes JS via subprocess and compares results with Python native calls.

Tests:
1. S₃ group operations
2. Operator window generation
3. ΔS_neg computations (core, derivative, signed, η)
4. Hex-prism geometry
5. Gate modulation
6. K-formation
7. Π-regime blending
8. Operator weights

@version 1.0.0
"""

import subprocess
import json
import math
import sys
from typing import Dict, List, Any, Tuple

# Import Python modules
from s3_operator_symmetry import (
    S3_ELEMENTS, OPERATOR_S3_MAP, BASE_OPERATORS,
    apply_s3, compose_s3, inverse_s3, parity_s3, sign_s3,
    rotation_index_from_z, truth_channel_from_z,
    generate_s3_operator_window, compute_s3_weights,
)

from delta_s_neg_extended import (
    Z_CRITICAL, PHI_INV,
    compute_delta_s_neg, compute_delta_s_neg_derivative, compute_delta_s_neg_signed,
    compute_eta, compute_hex_prism_geometry, compute_gate_modulation,
    compute_pi_blend_weights, check_k_formation, compute_full_state,
)

# ============================================================================
# UTILITIES
# ============================================================================

class ParityResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors: List[str] = []
    
    def ok(self, message: str):
        self.passed += 1
        print(f"  ✓ {message}")
    
    def fail(self, message: str, py_val: Any, js_val: Any):
        self.failed += 1
        self.errors.append(f"{message}: Python={py_val}, JS={js_val}")
        print(f"  ✗ {message}")
        print(f"    Python: {py_val}")
        print(f"    JS:     {js_val}")
    
    def summary(self) -> bool:
        print(f"\n{'='*70}")
        print(f"PARITY RESULTS: {self.passed} passed, {self.failed} failed")
        if self.errors:
            print("\nDiscrepancies:")
            for e in self.errors:
                print(f"  - {e}")
        print('='*70)
        return self.failed == 0


def run_js(code: str) -> Any:
    """Execute JavaScript code and return JSON result."""
    # Wrap code to output JSON
    wrapper = f"""
const S3 = require('./s3_operator_symmetry');
const Delta = require('./delta_s_neg_extended');
const result = (function() {{
  {code}
}})();
console.log(JSON.stringify(result));
"""
    try:
        proc = subprocess.run(
            ['node', '-e', wrapper],
            capture_output=True,
            text=True,
            timeout=10
        )
        if proc.returncode != 0:
            print(f"JS Error: {proc.stderr}")
            return None
        return json.loads(proc.stdout.strip())
    except Exception as e:
        print(f"JS Execution Error: {e}")
        return None


def close_enough(a: float, b: float, tol: float = 1e-10) -> bool:
    """Check if two floats are close enough."""
    if a is None or b is None:
        return a == b
    return abs(a - b) < tol


def section(title: str):
    print(f"\n{'─'*70}")
    print(f"  {title}")
    print(f"{'─'*70}")


# ============================================================================
# PARITY TESTS
# ============================================================================

def test_constants_parity(results: ParityResults):
    """Test that fundamental constants match."""
    section("Constants Parity")
    
    js_z_critical = run_js("return Delta.Z_CRITICAL;")
    if close_enough(Z_CRITICAL, js_z_critical, 1e-15):
        results.ok(f"Z_CRITICAL = {Z_CRITICAL}")
    else:
        results.fail("Z_CRITICAL", Z_CRITICAL, js_z_critical)
    
    js_phi_inv = run_js("return Delta.PHI_INV;")
    if close_enough(PHI_INV, js_phi_inv, 1e-14):
        results.ok(f"PHI_INV = {PHI_INV}")
    else:
        results.fail("PHI_INV", PHI_INV, js_phi_inv)


def test_s3_group_parity(results: ParityResults):
    """Test S₃ group operations match."""
    section("S₃ Group Operations Parity")
    
    # Test compose
    for a in ["e", "σ", "σ2", "τ1", "τ2", "τ3"]:
        for b in ["e", "σ"]:  # Sample pairs
            py_result = compose_s3(a, b)
            js_result = run_js(f"return S3.composeS3('{a}', '{b}');")
            if py_result == js_result:
                results.ok(f"compose({a}, {b}) = {py_result}")
            else:
                results.fail(f"compose({a}, {b})", py_result, js_result)
    
    # Test inverse
    for elem in ["e", "σ", "σ2", "τ1"]:
        py_result = inverse_s3(elem)
        js_result = run_js(f"return S3.inverseS3('{elem}');")
        if py_result == js_result:
            results.ok(f"inverse({elem}) = {py_result}")
        else:
            results.fail(f"inverse({elem})", py_result, js_result)
    
    # Test parity
    for elem in ["e", "σ", "τ1"]:
        py_parity = parity_s3(elem).value
        js_parity = run_js(f"return S3.parityS3('{elem}');")
        if py_parity == js_parity:
            results.ok(f"parity({elem}) = {py_parity}")
        else:
            results.fail(f"parity({elem})", py_parity, js_parity)


def test_s3_permutation_parity(results: ParityResults):
    """Test S₃ permutation action matches."""
    section("S₃ Permutation Parity")
    
    test_array = ["A", "B", "C"]
    
    for elem in ["e", "σ", "σ2", "τ1", "τ2", "τ3"]:
        py_result = apply_s3(test_array, elem)
        js_result = run_js(f"return S3.applyS3({json.dumps(test_array)}, '{elem}');")
        if py_result == js_result:
            results.ok(f"apply({elem}) on [A,B,C] = {py_result}")
        else:
            results.fail(f"apply({elem})", py_result, js_result)


def test_rotation_index_parity(results: ParityResults):
    """Test z → rotation index mapping."""
    section("Rotation Index Parity")
    
    test_z = [0.1, 0.2, 0.333, 0.4, 0.5, 0.666, 0.7, 0.8, 0.9]
    
    for z in test_z:
        py_idx = rotation_index_from_z(z)
        js_idx = run_js(f"return S3.rotationIndexFromZ({z});")
        if py_idx == js_idx:
            results.ok(f"rotationIndex({z}) = {py_idx}")
        else:
            results.fail(f"rotationIndex({z})", py_idx, js_idx)


def test_truth_channel_parity(results: ParityResults):
    """Test z → truth channel mapping."""
    section("Truth Channel Parity")
    
    test_z = [0.3, 0.5, 0.7, 0.85, 0.9, 0.95]
    
    for z in test_z:
        py_ch = truth_channel_from_z(z)
        js_ch = run_js(f"return S3.truthChannelFromZ({z});")
        if py_ch == js_ch:
            results.ok(f"truthChannel({z}) = {py_ch}")
        else:
            results.fail(f"truthChannel({z})", py_ch, js_ch)


def test_operator_window_parity(results: ParityResults):
    """Test S₃ operator window generation."""
    section("Operator Window Parity")
    
    harmonics = ["t1", "t3", "t5", "t6", "t9"]
    test_z = [0.2, 0.5, 0.8]
    
    for h in harmonics:
        for z in test_z:
            py_window = generate_s3_operator_window(h, z)
            js_window = run_js(f"return S3.generateS3OperatorWindow('{h}', {z});")
            
            # Windows should have same operators (possibly different order due to rotation)
            py_set = set(py_window)
            js_set = set(js_window) if js_window else set()
            
            if py_set == js_set:
                results.ok(f"window({h}, z={z}): {len(py_set)} operators match")
            else:
                results.fail(f"window({h}, z={z})", py_window, js_window)


def test_delta_s_neg_parity(results: ParityResults):
    """Test ΔS_neg computations."""
    section("ΔS_neg Computations Parity")
    
    test_z = [0.3, 0.5, 0.7, Z_CRITICAL, 0.9, 0.95]
    
    for z in test_z:
        # Core ΔS_neg
        py_s = compute_delta_s_neg(z)
        js_s = run_js(f"return Delta.computeDeltaSNeg({z});")
        if close_enough(py_s, js_s, 1e-12):
            results.ok(f"ΔS_neg({z:.4f}) = {py_s:.10f}")
        else:
            results.fail(f"ΔS_neg({z:.4f})", py_s, js_s)
        
        # Derivative
        py_d = compute_delta_s_neg_derivative(z)
        js_d = run_js(f"return Delta.computeDeltaSNegDerivative({z});")
        if close_enough(py_d, js_d, 1e-10):
            results.ok(f"d(ΔS_neg)/dz({z:.4f}) = {py_d:.10f}")
        else:
            results.fail(f"d(ΔS_neg)/dz({z:.4f})", py_d, js_d)
        
        # Signed
        py_signed = compute_delta_s_neg_signed(z)
        js_signed = run_js(f"return Delta.computeDeltaSNegSigned({z});")
        if close_enough(py_signed, js_signed, 1e-10):
            results.ok(f"signed ΔS_neg({z:.4f}) = {py_signed:.10f}")
        else:
            results.fail(f"signed ΔS_neg({z:.4f})", py_signed, js_signed)
        
        # η
        py_eta = compute_eta(z)
        js_eta = run_js(f"return Delta.computeEta({z});")
        if close_enough(py_eta, js_eta, 1e-12):
            results.ok(f"η({z:.4f}) = {py_eta:.10f}")
        else:
            results.fail(f"η({z:.4f})", py_eta, js_eta)


def test_geometry_parity(results: ParityResults):
    """Test hex-prism geometry."""
    section("Hex-Prism Geometry Parity")
    
    test_z = [0.3, 0.5, Z_CRITICAL, 0.9]
    
    for z in test_z:
        py_geom = compute_hex_prism_geometry(z)
        js_geom = run_js(f"return Delta.computeHexPrismGeometry({z});")
        
        if js_geom is None:
            results.fail(f"geometry({z:.4f})", py_geom, "JS returned None")
            continue
        
        # Check each field
        fields_ok = True
        for field in ['radius', 'height', 'twist']:
            py_val = getattr(py_geom, field)
            js_val = js_geom.get(field)
            if not close_enough(py_val, js_val, 1e-10):
                results.fail(f"geometry.{field}({z:.4f})", py_val, js_val)
                fields_ok = False
        
        if fields_ok:
            results.ok(f"geometry({z:.4f}): R={py_geom.radius:.6f}, H={py_geom.height:.6f}")


def test_gate_modulation_parity(results: ParityResults):
    """Test gate modulation parameters."""
    section("Gate Modulation Parity")
    
    test_z = [0.3, 0.5, Z_CRITICAL, 0.9]
    
    for z in test_z:
        py_mod = compute_gate_modulation(z)
        js_mod = run_js(f"return Delta.computeGateModulation({z});")
        
        if js_mod is None:
            results.fail(f"gateModulation({z:.4f})", py_mod, "JS returned None")
            continue
        
        fields_ok = True
        for py_field, js_field in [
            ('coherent_coupling', 'coherent_coupling'),
            ('decoherence_rate', 'decoherence_rate'),
            ('measurement_strength', 'measurement_strength'),
            ('entropy_target', 'entropy_target')
        ]:
            py_val = getattr(py_mod, py_field)
            js_val = js_mod.get(js_field)
            if not close_enough(py_val, js_val, 1e-10):
                results.fail(f"gate.{py_field}({z:.4f})", py_val, js_val)
                fields_ok = False
        
        if fields_ok:
            results.ok(f"gateModulation({z:.4f}): coupling={py_mod.coherent_coupling:.6f}")


def test_k_formation_parity(results: ParityResults):
    """Test K-formation status."""
    section("K-Formation Parity")
    
    test_z = [0.3, 0.5, 0.7, Z_CRITICAL, 0.9]
    
    for z in test_z:
        py_k = check_k_formation(z)
        js_k = run_js(f"return Delta.checkKFormation({z});")
        
        if js_k is None:
            results.fail(f"kFormation({z:.4f})", py_k, "JS returned None")
            continue
        
        # Check eta and formed
        if close_enough(py_k.eta, js_k.get('eta'), 1e-10) and py_k.formed == js_k.get('formed'):
            results.ok(f"kFormation({z:.4f}): η={py_k.eta:.6f}, formed={py_k.formed}")
        else:
            results.fail(f"kFormation({z:.4f})", 
                        f"η={py_k.eta}, formed={py_k.formed}",
                        f"η={js_k.get('eta')}, formed={js_k.get('formed')}")


def test_pi_blend_parity(results: ParityResults):
    """Test Π-regime blending weights."""
    section("Π-Regime Blending Parity")
    
    test_z = [0.5, 0.7, Z_CRITICAL - 0.01, Z_CRITICAL, Z_CRITICAL + 0.05, 0.95]
    
    for z in test_z:
        py_blend = compute_pi_blend_weights(z)
        js_blend = run_js(f"return Delta.computePiBlendWeights({z});")
        
        if js_blend is None:
            results.fail(f"piBlend({z:.4f})", py_blend, "JS returned None")
            continue
        
        w_pi_match = close_enough(py_blend.w_pi, js_blend.get('w_pi'), 1e-10)
        regime_match = py_blend.in_pi_regime == js_blend.get('in_pi_regime')
        
        if w_pi_match and regime_match:
            results.ok(f"piBlend({z:.4f}): w_π={py_blend.w_pi:.6f}, inΠ={py_blend.in_pi_regime}")
        else:
            results.fail(f"piBlend({z:.4f})",
                        f"w_π={py_blend.w_pi}, inΠ={py_blend.in_pi_regime}",
                        f"w_π={js_blend.get('w_pi')}, inΠ={js_blend.get('in_pi_regime')}")


def test_s3_weights_parity(results: ParityResults):
    """Test S₃ operator weight computations."""
    section("S₃ Operator Weights Parity")
    
    test_z = [0.3, 0.5, 0.85, Z_CRITICAL]
    
    for z in test_z:
        py_weights = compute_s3_weights(BASE_OPERATORS, z)
        js_weights = run_js(f"return S3.computeS3Weights({json.dumps(BASE_OPERATORS)}, {z});")
        
        if js_weights is None:
            results.fail(f"s3Weights({z:.4f})", py_weights, "JS returned None")
            continue
        
        all_match = True
        for op in BASE_OPERATORS:
            py_w = py_weights.get(op, 0)
            js_w = js_weights.get(op, 0)
            if not close_enough(py_w, js_w, 1e-8):
                results.fail(f"s3Weight[{op}]({z:.4f})", py_w, js_w)
                all_match = False
        
        if all_match:
            results.ok(f"s3Weights({z:.4f}): all 6 operators match")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*70)
    print("CROSS-LANGUAGE PARITY TEST SUITE")
    print("Python ↔ JavaScript")
    print("="*70)
    
    results = ParityResults()
    
    # Run all parity tests
    test_constants_parity(results)
    test_s3_group_parity(results)
    test_s3_permutation_parity(results)
    test_rotation_index_parity(results)
    test_truth_channel_parity(results)
    test_operator_window_parity(results)
    test_delta_s_neg_parity(results)
    test_geometry_parity(results)
    test_gate_modulation_parity(results)
    test_k_formation_parity(results)
    test_pi_blend_parity(results)
    test_s3_weights_parity(results)
    
    success = results.summary()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
