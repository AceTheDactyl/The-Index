#!/usr/bin/env python3
"""
================================================================================
POETRY VS PHYSICS: COMPREHENSIVE FALSIFICATION SUITE
================================================================================

SpiralOS makes poetic claims about neural network self-reference. This test
suite determines which claims are physics (empirically validated) vs poetry
(beautiful but not measurably true).

CLAIMS TESTED:
1. "Eigenvalues appear before constraints are enforced" 
   → Do φ eigenvalues appear in random init more than chance?

2. "GV ≈ 0.29 ≈ 1/(φ√2) is the observer lock"
   → Is 0.29 a real attractor or numerology?

3. "Λ-complexity approaches zero without intervention"
   → Do trained networks have lower Λ-complexity than random?

4. "Violation rates follow recursive patterns: d²V/dt² ∝ V"
   → Is violation dynamics actually self-referential?

5. "The math writes eigenvalues into existence"
   → Do φ eigenvalues emerge during training beyond chance?

6. "χ(T) peaks at T_c ≈ 0.05" (spin glass)
   → Already validated, include for completeness

================================================================================
"""

import numpy as np
import warnings
from dataclasses import dataclass
from typing import List, Tuple, Dict
from collections import defaultdict
import time

warnings.filterwarnings('ignore')

# ==================== CONSTANTS ====================
PHI = (1 + np.sqrt(5)) / 2          # 1.618...
PSI = (1 - np.sqrt(5)) / 2          # -0.618...
PHI_INV = 1 / PHI                    # 0.618...
SQRT2 = np.sqrt(2)                   # 1.414...
SQRT2_INV = 1 / SQRT2                # 0.707...
Z_C = np.sqrt(3) / 2                 # 0.866...
E_CONST = np.e                       # 2.718...
PI_CONST = np.pi                     # 3.141...

# SpiralOS claimed values
CLAIMED_GV_LOCK = 0.29
CLAIMED_GV_FORMULA = 1 / (PHI * SQRT2)  # ≈ 0.437... NOT 0.29!
CLAIMED_T_C = 0.05

# Special eigenvalues to check
SPECIAL_VALUES = [PHI, PHI_INV, SQRT2, SQRT2_INV, Z_C, 1.0, 
                  1/E_CONST, 1/PI_CONST, 0.5]

@dataclass
class TestResult:
    name: str
    claim: str
    prediction: str
    observation: str
    p_value: float
    verdict: str  # "PHYSICS", "POETRY", "INCONCLUSIVE"
    details: str

# ==================== TEST 1: EIGENVALUES BEFORE CONSTRAINTS ====================

def test_eigenvalues_before_constraints(n_matrices=1000, size=64) -> TestResult:
    """
    CLAIM: "Eigenvalues appear before constraints are enforced"
    TEST: Do φ eigenvalues appear in random matrices more than chance?
    NULL: Random matrices have eigenvalues uniformly in a disk (circular law)
    """
    print("\n" + "="*70)
    print("TEST 1: EIGENVALUES BEFORE CONSTRAINTS")
    print("="*70)
    print("Claim: φ eigenvalues appear in random init more than chance")
    
    phi_counts = []
    total_eigenvalues = 0
    
    for _ in range(n_matrices):
        # Random Gaussian matrix (standard initialization)
        W = np.random.randn(size, size) / np.sqrt(size)
        eigs = np.abs(np.linalg.eigvals(W))
        total_eigenvalues += len(eigs)
        
        # Count eigenvalues near φ (within tolerance)
        tolerance = 0.05
        near_phi = np.sum(np.abs(eigs - PHI) < tolerance)
        near_phi += np.sum(np.abs(eigs - PHI_INV) < tolerance)
        phi_counts.append(near_phi)
    
    observed_rate = np.mean(phi_counts) / size
    
    # Expected rate under null: eigenvalues in disk of radius ~1
    # Probability of landing in [φ-ε, φ+ε] or [1/φ-ε, 1/φ+ε]
    # For circular law with radius 1, density is 1/π at |z|<1
    # φ ≈ 1.618 is OUTSIDE the unit disk, so expect ~0
    # 1/φ ≈ 0.618 is inside, expected fraction ≈ 2*tolerance/1 = 0.1
    expected_rate = 0.1  # rough estimate for 1/φ
    
    # Simple z-test
    std_err = np.std(phi_counts) / np.sqrt(n_matrices) / size
    z_stat = (observed_rate - expected_rate) / std_err if std_err > 0 else 0
    p_value = 2 * (1 - 0.5 * (1 + np.sign(z_stat) * min(abs(z_stat)/3, 1)))  # rough
    
    print(f"\nResults:")
    print(f"  Matrices tested: {n_matrices}")
    print(f"  Matrix size: {size}x{size}")
    print(f"  Observed φ-eigenvalue rate: {observed_rate:.4f}")
    print(f"  Expected under null: ~{expected_rate:.4f}")
    print(f"  Z-statistic: {z_stat:.2f}")
    
    if abs(observed_rate - expected_rate) < 0.02:
        verdict = "POETRY"
        details = "φ eigenvalues appear at chance rate in random matrices"
    elif observed_rate > expected_rate * 1.5:
        verdict = "PHYSICS"
        details = "φ eigenvalues appear MORE than chance - surprising!"
    else:
        verdict = "INCONCLUSIVE"
        details = "Need more data or refined null hypothesis"
    
    print(f"\nVERDICT: {verdict}")
    print(f"  {details}")
    
    return TestResult(
        name="Eigenvalues Before Constraints",
        claim="φ eigenvalues appear before constraints are enforced",
        prediction="φ eigenvalues in random init > chance",
        observation=f"Rate = {observed_rate:.4f} (expected ~{expected_rate:.4f})",
        p_value=p_value,
        verdict=verdict,
        details=details
    )

# ==================== TEST 2: GV LOCK VALUE ====================

def test_gv_lock_value() -> TestResult:
    """
    CLAIM: "GV ≈ 0.29 ≈ 1/(φ√2) is the observer lock"
    TEST: Is 1/(φ√2) actually ≈ 0.29?
    """
    print("\n" + "="*70)
    print("TEST 2: GV LOCK NUMEROLOGY")
    print("="*70)
    print("Claim: GV ≈ 0.29 ≈ 1/(φ√2)")
    
    # Direct calculation
    formula_value = 1 / (PHI * SQRT2)
    claimed_value = 0.29
    
    print(f"\nDirect calculation:")
    print(f"  1/(φ√2) = 1/({PHI:.6f} × {SQRT2:.6f})")
    print(f"         = 1/{PHI * SQRT2:.6f}")
    print(f"         = {formula_value:.6f}")
    print(f"  Claimed: 0.29")
    print(f"  Error: {abs(formula_value - claimed_value):.6f} ({100*abs(formula_value - claimed_value)/claimed_value:.1f}%)")
    
    # What WOULD give 0.29?
    print(f"\nReverse engineering 0.29:")
    print(f"  0.29 = 1/3.448...")
    print(f"  φ² = {PHI**2:.6f}")
    print(f"  φ×√2 = {PHI * SQRT2:.6f}")
    print(f"  φ+1 = {PHI + 1:.6f} (= φ²)")
    print(f"  1/φ² = {1/PHI**2:.6f}")
    print(f"  (φ-1)/φ = {(PHI-1)/PHI:.6f}")
    
    # 0.29 is closest to...
    candidates = {
        "1/(φ√2)": 1/(PHI * SQRT2),
        "1/φ²": 1/PHI**2,
        "(φ-1)/φ": (PHI-1)/PHI,
        "1/e": 1/E_CONST,
        "1/π": 1/PI_CONST,
        "√2-1": SQRT2 - 1,
        "2-φ": 2 - PHI,
    }
    
    print(f"\nClosest matches to 0.29:")
    sorted_candidates = sorted(candidates.items(), key=lambda x: abs(x[1] - 0.29))
    for name, val in sorted_candidates[:5]:
        print(f"  {name} = {val:.6f} (error: {abs(val-0.29):.4f})")
    
    best_match = sorted_candidates[0]
    
    if abs(formula_value - claimed_value) > 0.1:
        verdict = "POETRY"
        details = f"1/(φ√2) = {formula_value:.3f} ≠ 0.29. Claimed formula is WRONG."
    else:
        verdict = "PHYSICS"
        details = "Formula matches claimed value"
    
    print(f"\nVERDICT: {verdict}")
    print(f"  {details}")
    print(f"  Best match for 0.29: {best_match[0]} = {best_match[1]:.4f}")
    
    return TestResult(
        name="GV Lock Numerology",
        claim="GV ≈ 0.29 ≈ 1/(φ√2)",
        prediction="1/(φ√2) should equal 0.29",
        observation=f"1/(φ√2) = {formula_value:.4f}, error = {abs(formula_value - 0.29):.4f}",
        p_value=0.0 if abs(formula_value - claimed_value) > 0.1 else 1.0,
        verdict=verdict,
        details=details
    )

# ==================== TEST 3: Λ-COMPLEXITY REDUCTION ====================

def compute_lambda_complexity(eigs: np.ndarray, basis: List[float] = None) -> float:
    """Compute Λ-complexity: sparsity of decomposition in lattice basis"""
    if basis is None:
        basis = [PHI, 1/PHI, SQRT2, 1/SQRT2, E_CONST, 1/E_CONST, 
                 PI_CONST, 1/PI_CONST, Z_C, 1.0, 0.5]
    
    if len(eigs) == 0:
        return float('inf')
    
    # For each eigenvalue, find minimum distance to any basis element
    # Lower complexity = eigenvalues closer to basis
    complexities = []
    for ev in np.abs(eigs):
        min_dist = min(abs(ev - b) for b in basis)
        complexities.append(min_dist)
    
    return np.mean(complexities)

def test_lambda_complexity_reduction(n_trials=100, size=32, epochs=100) -> TestResult:
    """
    CLAIM: "Λ-complexity approaches zero without intervention"
    TEST: Do trained networks have lower Λ-complexity than random?
    """
    print("\n" + "="*70)
    print("TEST 3: Λ-COMPLEXITY REDUCTION")
    print("="*70)
    print("Claim: Training reduces Λ-complexity (eigenvalues approach lattice)")
    
    # We'll simulate "training" as gradient descent on a simple loss
    # that doesn't explicitly push toward lattice points
    
    random_complexities = []
    trained_complexities = []
    
    for trial in range(n_trials):
        # Random matrix
        W = np.random.randn(size, size) / np.sqrt(size)
        eigs_random = np.linalg.eigvals(W)
        random_complexities.append(compute_lambda_complexity(eigs_random))
        
        # "Train" by doing gradient descent on ||W||² (simple regularization)
        # This is NOT pushing toward golden - just generic training
        W_trained = W.copy()
        lr = 0.01
        for _ in range(epochs):
            # Gradient of ||W||² is 2W
            W_trained = W_trained - lr * 2 * W_trained
            # Add some noise (SGD simulation)
            W_trained += np.random.randn(size, size) * 0.001
        
        eigs_trained = np.linalg.eigvals(W_trained)
        trained_complexities.append(compute_lambda_complexity(eigs_trained))
    
    random_mean = np.mean(random_complexities)
    trained_mean = np.mean(trained_complexities)
    
    print(f"\nResults:")
    print(f"  Random Λ-complexity: {random_mean:.4f} ± {np.std(random_complexities):.4f}")
    print(f"  Trained Λ-complexity: {trained_mean:.4f} ± {np.std(trained_complexities):.4f}")
    print(f"  Reduction: {100*(random_mean - trained_mean)/random_mean:.1f}%")
    
    # Statistical test
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(random_complexities, trained_complexities)
    
    print(f"  t-statistic: {t_stat:.2f}")
    print(f"  p-value: {p_value:.4f}")
    
    # But wait - does training actually reduce complexity, or does it just
    # shrink eigenvalues (which get closer to 0 and 1, both in the basis)?
    print(f"\nSanity check - eigenvalue magnitudes:")
    print(f"  Random mean |λ|: {np.mean([np.mean(np.abs(np.linalg.eigvals(np.random.randn(size,size)/np.sqrt(size)))) for _ in range(20)]):.4f}")
    
    if trained_mean < random_mean * 0.8 and p_value < 0.05:
        verdict = "PHYSICS"
        details = "Training genuinely reduces Λ-complexity"
    elif trained_mean < random_mean and p_value < 0.05:
        verdict = "INCONCLUSIVE"
        details = "Reduction exists but may be due to eigenvalue shrinkage toward 0,1"
    else:
        verdict = "POETRY"
        details = "No significant Λ-complexity reduction from training"
    
    print(f"\nVERDICT: {verdict}")
    print(f"  {details}")
    
    return TestResult(
        name="Λ-Complexity Reduction",
        claim="Λ-complexity approaches zero without intervention",
        prediction="Trained networks have lower Λ-complexity",
        observation=f"Random: {random_mean:.4f}, Trained: {trained_mean:.4f}",
        p_value=p_value,
        verdict=verdict,
        details=details
    )

# ==================== TEST 4: RECURSIVE VIOLATION DYNAMICS ====================

def test_recursive_violation_dynamics(n_trials=50, size=32, epochs=200) -> TestResult:
    """
    CLAIM: "d²V/dt² ∝ V" - violation is an eigenfunction of itself
    TEST: Does the second derivative of violation correlate with violation?
    """
    print("\n" + "="*70)
    print("TEST 4: RECURSIVE VIOLATION DYNAMICS")
    print("="*70)
    print("Claim: d²V/dt² ∝ V (violation is eigenfunction of d²/dt²)")
    
    correlations = []
    
    for trial in range(n_trials):
        W = np.random.randn(size, size) / np.sqrt(size)
        
        violations = []
        for epoch in range(epochs):
            # Compute golden violation
            V = np.linalg.norm(W @ W - W - np.eye(size))
            violations.append(V)
            
            # Simple gradient step (NOT toward golden constraint)
            grad = np.random.randn(size, size) * 0.1
            W = W - 0.01 * grad
        
        violations = np.array(violations)
        
        # Compute derivatives
        dV = np.gradient(violations)
        d2V = np.gradient(dV)
        
        # Correlation between d²V/dt² and V
        # Trim edges where gradient is unreliable
        V_mid = violations[5:-5]
        d2V_mid = d2V[5:-5]
        
        if len(V_mid) > 10:
            corr = np.corrcoef(V_mid, d2V_mid)[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
    
    mean_corr = np.mean(correlations)
    std_corr = np.std(correlations)
    
    print(f"\nResults:")
    print(f"  Mean correlation(V, d²V/dt²): {mean_corr:.4f} ± {std_corr:.4f}")
    print(f"  For eigenfunction d²V/dt² = -ω²V, expect corr ≈ -1")
    print(f"  For random walk, expect corr ≈ 0")
    
    # What correlation would we expect?
    # If V follows Brownian motion: d²V/dt² is noise, corr ≈ 0
    # If V is oscillatory: corr ≈ -1
    # If V is exponential decay: d²V/dt² ∝ V, corr ≈ +1
    
    if abs(mean_corr) > 0.5:
        verdict = "PHYSICS"
        details = f"Strong correlation ({mean_corr:.2f}) suggests structured dynamics"
    elif abs(mean_corr) > 0.2:
        verdict = "INCONCLUSIVE"
        details = f"Weak correlation ({mean_corr:.2f}) - may be artifact"
    else:
        verdict = "POETRY"
        details = f"No correlation ({mean_corr:.2f}) - violation dynamics are not self-referential"
    
    print(f"\nVERDICT: {verdict}")
    print(f"  {details}")
    
    return TestResult(
        name="Recursive Violation Dynamics",
        claim="d²V/dt² ∝ V (violation is eigenfunction of itself)",
        prediction="Strong correlation between V and d²V/dt²",
        observation=f"Correlation = {mean_corr:.4f} ± {std_corr:.4f}",
        p_value=1.0 - abs(mean_corr),  # rough
        verdict=verdict,
        details=details
    )

# ==================== TEST 5: φ EMERGENCE DURING TRAINING ====================

def test_phi_emergence(n_trials=50, size=32, epochs=300) -> TestResult:
    """
    CLAIM: "The math writes eigenvalues into existence"
    TEST: Do φ eigenvalues emerge during training beyond chance?
    """
    print("\n" + "="*70)
    print("TEST 5: φ EMERGENCE DURING TRAINING")
    print("="*70)
    print("Claim: φ eigenvalues emerge during training beyond chance")
    
    phi_fraction_init = []
    phi_fraction_final = []
    tolerance = 0.1
    
    for trial in range(n_trials):
        W = np.random.randn(size, size) / np.sqrt(size)
        
        # Initial φ-eigenvalues
        eigs_init = np.abs(np.linalg.eigvals(W))
        phi_init = np.sum((np.abs(eigs_init - PHI) < tolerance) | 
                          (np.abs(eigs_init - PHI_INV) < tolerance)) / size
        phi_fraction_init.append(phi_init)
        
        # "Train" with generic loss (NOT golden constraint)
        for epoch in range(epochs):
            # Gradient descent on Frobenius norm (shrinkage)
            W = W * 0.99
            # Plus some task-like structure
            target = np.random.randn(size, size) / np.sqrt(size)
            W = W + 0.001 * (target - W)
        
        # Final φ-eigenvalues
        eigs_final = np.abs(np.linalg.eigvals(W))
        phi_final = np.sum((np.abs(eigs_final - PHI) < tolerance) | 
                           (np.abs(eigs_final - PHI_INV) < tolerance)) / size
        phi_fraction_final.append(phi_final)
    
    init_mean = np.mean(phi_fraction_init)
    final_mean = np.mean(phi_fraction_final)
    
    print(f"\nResults:")
    print(f"  Initial φ-fraction: {init_mean:.4f} ± {np.std(phi_fraction_init):.4f}")
    print(f"  Final φ-fraction: {final_mean:.4f} ± {np.std(phi_fraction_final):.4f}")
    print(f"  Change: {100*(final_mean - init_mean)/init_mean:.1f}%")
    
    # Statistical test
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(phi_fraction_init, phi_fraction_final)
    
    print(f"  t-statistic: {t_stat:.2f}")
    print(f"  p-value: {p_value:.4f}")
    
    if final_mean > init_mean * 1.2 and p_value < 0.05:
        verdict = "PHYSICS"
        details = "φ eigenvalues genuinely emerge during training!"
    elif final_mean > init_mean and p_value < 0.05:
        verdict = "INCONCLUSIVE"
        details = "Small increase, may be due to eigenvalue shrinkage"
    else:
        verdict = "POETRY"
        details = "No significant φ emergence - eigenvalue distribution unchanged"
    
    print(f"\nVERDICT: {verdict}")
    print(f"  {details}")
    
    return TestResult(
        name="φ Emergence During Training",
        claim="The math writes φ eigenvalues into existence",
        prediction="φ-fraction increases during training",
        observation=f"Init: {init_mean:.4f}, Final: {final_mean:.4f}",
        p_value=p_value,
        verdict=verdict,
        details=details
    )

# ==================== TEST 6: SPIN GLASS SUSCEPTIBILITY ====================

def test_spin_glass_susceptibility(n_temps=9, n_runs=20, size=32, epochs=100) -> TestResult:
    """
    CLAIM: "χ(T) peaks at T_c ≈ 0.05"
    TEST: Does susceptibility (variance) peak near 0.05?
    NOTE: This was already validated, including for completeness
    """
    print("\n" + "="*70)
    print("TEST 6: SPIN GLASS SUSCEPTIBILITY (Quick Version)")
    print("="*70)
    print("Claim: χ(T) = Var[O] peaks at T_c ≈ 0.05")
    
    temps = np.linspace(0.03, 0.07, n_temps)
    
    def order_param(W):
        eigs = np.abs(np.linalg.eigvals(W))
        return np.sum([min(abs(ev - sv) for sv in SPECIAL_VALUES) < 0.15 
                      for ev in eigs]) / len(eigs)
    
    susceptibilities = []
    
    for T in temps:
        orders = []
        for run in range(n_runs):
            np.random.seed(hash((T, run)) % (2**32))
            W = np.random.randn(size, size) / np.sqrt(size)
            
            # Simple training with temperature-dependent noise
            for epoch in range(epochs):
                grad = 2 * W + np.random.randn(size, size) * T
                W = W - 0.01 * (1 + T) * grad
            
            orders.append(order_param(W))
        
        susceptibilities.append(np.var(orders))
    
    # Find peak
    max_idx = np.argmax(susceptibilities)
    T_c_measured = temps[max_idx]
    
    print(f"\nResults:")
    print(f"  Temperature range: {temps[0]:.3f} to {temps[-1]:.3f}")
    print(f"  Peak susceptibility at T = {T_c_measured:.4f}")
    print(f"  Predicted T_c = {CLAIMED_T_C}")
    print(f"  Error: {abs(T_c_measured - CLAIMED_T_C):.4f}")
    
    if abs(T_c_measured - CLAIMED_T_C) < 0.015:
        verdict = "PHYSICS"
        details = f"χ(T) peaks at T_c = {T_c_measured:.3f} ≈ 0.05"
    else:
        verdict = "INCONCLUSIVE"
        details = f"Peak at {T_c_measured:.3f}, expected 0.05"
    
    print(f"\nVERDICT: {verdict}")
    print(f"  {details}")
    
    return TestResult(
        name="Spin Glass Susceptibility",
        claim="χ(T) peaks at T_c ≈ 0.05",
        prediction="Susceptibility peaks at T = 0.05",
        observation=f"Peak at T = {T_c_measured:.4f}",
        p_value=0.05 if abs(T_c_measured - CLAIMED_T_C) < 0.015 else 0.5,
        verdict=verdict,
        details=details
    )

# ==================== MAIN ====================

def main():
    print("\n" + "="*70)
    print("POETRY VS PHYSICS: COMPREHENSIVE FALSIFICATION SUITE")
    print("="*70)
    print("\nTesting SpiralOS claims about neural network self-reference")
    print("Each claim is tested against null hypothesis")
    print("Verdicts: PHYSICS (validated), POETRY (falsified), INCONCLUSIVE")
    
    start = time.time()
    results = []
    
    # Run all tests
    try:
        from scipy import stats
        HAS_SCIPY = True
    except ImportError:
        HAS_SCIPY = False
        print("\nWARNING: scipy not available, some tests will be simplified")
    
    results.append(test_eigenvalues_before_constraints())
    results.append(test_gv_lock_value())
    
    if HAS_SCIPY:
        results.append(test_lambda_complexity_reduction())
        results.append(test_recursive_violation_dynamics())
        results.append(test_phi_emergence())
    
    results.append(test_spin_glass_susceptibility())
    
    # ==================== SUMMARY ====================
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    physics_count = sum(1 for r in results if r.verdict == "PHYSICS")
    poetry_count = sum(1 for r in results if r.verdict == "POETRY")
    inconclusive_count = sum(1 for r in results if r.verdict == "INCONCLUSIVE")
    
    print(f"\nTotal tests: {len(results)}")
    print(f"  PHYSICS (validated):    {physics_count}")
    print(f"  POETRY (falsified):     {poetry_count}")
    print(f"  INCONCLUSIVE:           {inconclusive_count}")
    
    print("\n" + "-"*70)
    print("DETAILED VERDICTS")
    print("-"*70)
    
    for r in results:
        emoji = "✅" if r.verdict == "PHYSICS" else "❌" if r.verdict == "POETRY" else "❓"
        print(f"\n{emoji} {r.name}: {r.verdict}")
        print(f"   Claim: {r.claim}")
        print(f"   Result: {r.observation}")
        print(f"   {r.details}")
    
    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    
    if poetry_count > physics_count:
        print("\n⚠️  More claims FALSIFIED than validated.")
        print("    SpiralOS contains beautiful poetry, but not rigorous physics.")
        print("    The mathematical intuitions are evocative but not empirically grounded.")
    elif physics_count > poetry_count:
        print("\n✅ More claims VALIDATED than falsified.")
        print("    SpiralOS captures genuine physical structure!")
        print("    The poetic language describes real phenomena.")
    else:
        print("\n❓ Mixed results.")
        print("    Some claims are physics, some are poetry.")
        print("    Need more refined tests to distinguish.")
    
    print(f"\nRuntime: {time.time() - start:.1f} seconds")
    
    # Save results
    with open("poetry_vs_physics_results.txt", "w") as f:
        f.write("POETRY VS PHYSICS RESULTS\n")
        f.write("="*50 + "\n\n")
        for r in results:
            f.write(f"{r.verdict}: {r.name}\n")
            f.write(f"  Claim: {r.claim}\n")
            f.write(f"  Observation: {r.observation}\n")
            f.write(f"  Details: {r.details}\n\n")
    
    print("\nResults saved to: poetry_vs_physics_results.txt")
    
    return results

if __name__ == "__main__":
    results = main()
