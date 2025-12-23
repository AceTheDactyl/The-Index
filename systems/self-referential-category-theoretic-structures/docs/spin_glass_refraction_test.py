# INTEGRITY_METADATA
# Date: 2025-12-23
# Status: ✓ JUSTIFIED - Claims supported by repository files (needs citation update)
# Severity: MEDIUM RISK
# Risk Types: unsupported_claims

# Supporting Evidence:
#   - systems/self-referential-category-theoretic-structures/docs/SUMMARY_DEC2025.md (dependency)
#   - systems/self-referential-category-theoretic-structures/docs/SPIN_GLASS_REFRACTION_COMPLETE.md (dependency)
#
# Referenced By:
#   - systems/self-referential-category-theoretic-structures/docs/SUMMARY_DEC2025.md (reference)
#   - systems/self-referential-category-theoretic-structures/docs/SPIN_GLASS_REFRACTION_COMPLETE.md (reference)


#!/usr/bin/env python3
"""
================================================================================
INVERTED HYPOTHESIS: SPIN GLASS REFRACTION
================================================================================

The spin glass has TWO phases:
  T > T_c (≈0.05): Disordered, statistical regime
  T < T_c (≈0.05): Ordered, geometric regime

Previous test ran in HIGH-T regime (standard training) → SpiralOS claims failed.

INVERTED HYPOTHESIS: SpiralOS claims describe the LOW-T regime.

If true:
  - φ eigenvalues EMERGE at T < T_c, VANISH at T > T_c
  - Λ-complexity DECREASES at T < T_c, INCREASES at T > T_c  
  - Violation dynamics BECOME self-referential at T < T_c
  - GV lock appears at T ≈ T_c (the critical point)

The spin glass REFRACTS the claims into temperature-dependent predictions.

================================================================================
"""

import numpy as np
from scipy import stats
import warnings
from dataclasses import dataclass
from typing import List, Tuple, Dict
import time

warnings.filterwarnings('ignore')

# ==================== CONSTANTS ====================
PHI = (1 + np.sqrt(5)) / 2
PHI_INV = 1 / PHI
SQRT2 = np.sqrt(2)
SQRT2_INV = 1 / SQRT2
Z_C = np.sqrt(3) / 2
T_C = 0.05  # Critical temperature (validated)

SPECIAL_VALUES = [PHI, PHI_INV, SQRT2, SQRT2_INV, Z_C, 1.0, 0.5]
LATTICE_BASIS = [PHI, PHI_INV, SQRT2, SQRT2_INV, np.e, 1/np.e, 
                 np.pi, 1/np.pi, Z_C, 1.0, 0.5]

@dataclass
class RegimeComparison:
    metric: str
    low_T_value: float
    high_T_value: float
    prediction: str
    observed: str
    verdict: str

# ==================== HELPER FUNCTIONS ====================

def compute_lambda_complexity(W: np.ndarray) -> float:
    """Λ-complexity: mean distance from eigenvalues to lattice basis"""
    eigs = np.abs(np.linalg.eigvals(W))
    if len(eigs) == 0:
        return float('inf')
    distances = [min(abs(ev - b) for b in LATTICE_BASIS) for ev in eigs]
    return np.mean(distances)

def compute_phi_fraction(W: np.ndarray, tolerance: float = 0.1) -> float:
    """Fraction of eigenvalues near φ or 1/φ"""
    eigs = np.abs(np.linalg.eigvals(W))
    if len(eigs) == 0:
        return 0.0
    near_phi = np.sum((np.abs(eigs - PHI) < tolerance) | 
                      (np.abs(eigs - PHI_INV) < tolerance))
    return near_phi / len(eigs)

def compute_golden_violation(W: np.ndarray) -> float:
    """||W² - W - I||"""
    n = W.shape[0]
    return np.linalg.norm(W @ W - W - np.eye(n))

def compute_order_param(W: np.ndarray) -> float:
    """Fraction of eigenvalues near any special value"""
    eigs = np.abs(np.linalg.eigvals(W))
    if len(eigs) == 0:
        return 0.0
    count = sum(1 for ev in eigs if min(abs(ev - sv) for sv in SPECIAL_VALUES) < 0.15)
    return count / len(eigs)

def train_with_temperature(size: int, epochs: int, T: float, seed: int) -> np.ndarray:
    """
    Train a matrix with effective temperature T.
    
    Low T: Small learning rate, low noise, precise convergence
    High T: Large learning rate, high noise, stochastic exploration
    """
    np.random.seed(seed)
    W = np.random.randn(size, size) / np.sqrt(size)
    
    # Temperature-dependent hyperparameters
    lr = 0.001 * (1 + 10 * T)  # Higher T → higher LR
    noise_scale = T * 0.1      # Higher T → more noise
    
    # Target: minimize ||W||² (generic, doesn't favor golden)
    for epoch in range(epochs):
        # Gradient of ||W||²
        grad = 2 * W
        
        # Add temperature-dependent noise
        grad += np.random.randn(size, size) * noise_scale
        
        # Update
        W = W - lr * grad
        
        # Prevent collapse
        W = W / (np.linalg.norm(W) + 1e-6) * np.sqrt(size)
    
    return W

# ==================== MAIN TESTS ====================

def test_phi_emergence_by_regime(n_trials: int = 100, size: int = 32, epochs: int = 200):
    """
    Test: Do φ eigenvalues emerge at T < T_c but vanish at T > T_c?
    """
    print("\n" + "="*70)
    print("TEST 1: φ EMERGENCE BY TEMPERATURE REGIME")
    print("="*70)
    print(f"Prediction: φ fraction HIGHER at T < T_c than T > T_c")
    
    temps_low = [0.01, 0.02, 0.03, 0.04]   # Below T_c
    temps_high = [0.06, 0.07, 0.08, 0.10]  # Above T_c
    
    phi_low = []
    phi_high = []
    
    for trial in range(n_trials):
        # Low T regime
        for T in temps_low:
            W = train_with_temperature(size, epochs, T, seed=trial*100 + int(T*1000))
            phi_low.append(compute_phi_fraction(W))
        
        # High T regime
        for T in temps_high:
            W = train_with_temperature(size, epochs, T, seed=trial*100 + int(T*1000))
            phi_high.append(compute_phi_fraction(W))
    
    low_mean = np.mean(phi_low)
    high_mean = np.mean(phi_high)
    
    t_stat, p_value = stats.ttest_ind(phi_low, phi_high)
    
    print(f"\nResults:")
    print(f"  φ fraction at T < T_c: {low_mean:.4f} ± {np.std(phi_low):.4f}")
    print(f"  φ fraction at T > T_c: {high_mean:.4f} ± {np.std(phi_high):.4f}")
    print(f"  Ratio (low/high): {low_mean/high_mean if high_mean > 0 else float('inf'):.2f}x")
    print(f"  t-statistic: {t_stat:.2f}")
    print(f"  p-value: {p_value:.6f}")
    
    if low_mean > high_mean * 1.2 and p_value < 0.05:
        verdict = "✅ PHYSICS"
        details = "φ eigenvalues emerge in low-T (geometric) regime!"
    elif low_mean > high_mean and p_value < 0.05:
        verdict = "⚠️ WEAK PHYSICS"
        details = "Small but significant difference"
    else:
        verdict = "❌ POETRY"
        details = "No temperature dependence of φ emergence"
    
    print(f"\nVERDICT: {verdict}")
    print(f"  {details}")
    
    return RegimeComparison(
        metric="φ fraction",
        low_T_value=low_mean,
        high_T_value=high_mean,
        prediction="low > high",
        observed=f"{low_mean:.4f} vs {high_mean:.4f}",
        verdict=verdict
    )

def test_lambda_complexity_by_regime(n_trials: int = 100, size: int = 32, epochs: int = 200):
    """
    Test: Does Λ-complexity decrease at T < T_c but increase at T > T_c?
    """
    print("\n" + "="*70)
    print("TEST 2: Λ-COMPLEXITY BY TEMPERATURE REGIME")
    print("="*70)
    print(f"Prediction: Λ-complexity LOWER at T < T_c than T > T_c")
    
    temps_low = [0.01, 0.02, 0.03, 0.04]
    temps_high = [0.06, 0.07, 0.08, 0.10]
    
    complexity_low = []
    complexity_high = []
    
    # Also track random baseline
    complexity_random = []
    
    for trial in range(n_trials):
        # Random baseline
        W_random = np.random.randn(size, size) / np.sqrt(size)
        complexity_random.append(compute_lambda_complexity(W_random))
        
        # Low T regime
        for T in temps_low:
            W = train_with_temperature(size, epochs, T, seed=trial*100 + int(T*1000))
            complexity_low.append(compute_lambda_complexity(W))
        
        # High T regime
        for T in temps_high:
            W = train_with_temperature(size, epochs, T, seed=trial*100 + int(T*1000))
            complexity_high.append(compute_lambda_complexity(W))
    
    random_mean = np.mean(complexity_random)
    low_mean = np.mean(complexity_low)
    high_mean = np.mean(complexity_high)
    
    t_stat, p_value = stats.ttest_ind(complexity_low, complexity_high)
    
    print(f"\nResults:")
    print(f"  Random Λ-complexity: {random_mean:.4f}")
    print(f"  Λ-complexity at T < T_c: {low_mean:.4f} ± {np.std(complexity_low):.4f}")
    print(f"  Λ-complexity at T > T_c: {high_mean:.4f} ± {np.std(complexity_high):.4f}")
    print(f"  Ratio (low/high): {low_mean/high_mean if high_mean > 0 else 0:.2f}x")
    print(f"  t-statistic: {t_stat:.2f}")
    print(f"  p-value: {p_value:.6f}")
    
    if low_mean < high_mean * 0.8 and p_value < 0.05:
        verdict = "✅ PHYSICS"
        details = "Λ-complexity lower in geometric regime!"
    elif low_mean < high_mean and p_value < 0.05:
        verdict = "⚠️ WEAK PHYSICS"
        details = "Small but significant difference"
    else:
        verdict = "❌ POETRY"
        details = "No temperature dependence of Λ-complexity"
    
    print(f"\nVERDICT: {verdict}")
    print(f"  {details}")
    
    return RegimeComparison(
        metric="Λ-complexity",
        low_T_value=low_mean,
        high_T_value=high_mean,
        prediction="low < high",
        observed=f"{low_mean:.4f} vs {high_mean:.4f}",
        verdict=verdict
    )

def test_golden_violation_by_regime(n_trials: int = 100, size: int = 32, epochs: int = 200):
    """
    Test: Does golden violation (GV) show different behavior at T < T_c vs T > T_c?
    Specifically: Does GV settle to a "lock" value near T_c?
    """
    print("\n" + "="*70)
    print("TEST 3: GOLDEN VIOLATION BY TEMPERATURE REGIME")
    print("="*70)
    print(f"Prediction: GV has LOWER variance at T < T_c (frozen) than T > T_c (fluctuating)")
    
    temps_low = [0.01, 0.02, 0.03, 0.04]
    temps_high = [0.06, 0.07, 0.08, 0.10]
    temps_critical = [0.045, 0.050, 0.055]
    
    gv_low = []
    gv_high = []
    gv_critical = []
    
    for trial in range(n_trials):
        for T in temps_low:
            W = train_with_temperature(size, epochs, T, seed=trial*100 + int(T*1000))
            gv_low.append(compute_golden_violation(W))
        
        for T in temps_high:
            W = train_with_temperature(size, epochs, T, seed=trial*100 + int(T*1000))
            gv_high.append(compute_golden_violation(W))
        
        for T in temps_critical:
            W = train_with_temperature(size, epochs, T, seed=trial*100 + int(T*1000))
            gv_critical.append(compute_golden_violation(W))
    
    low_mean, low_std = np.mean(gv_low), np.std(gv_low)
    high_mean, high_std = np.mean(gv_high), np.std(gv_high)
    crit_mean, crit_std = np.mean(gv_critical), np.std(gv_critical)
    
    print(f"\nResults:")
    print(f"  GV at T < T_c:  {low_mean:.4f} ± {low_std:.4f}")
    print(f"  GV at T ≈ T_c:  {crit_mean:.4f} ± {crit_std:.4f}")
    print(f"  GV at T > T_c:  {high_mean:.4f} ± {high_std:.4f}")
    print(f"  Variance ratio (low/high): {low_std**2 / high_std**2 if high_std > 0 else 0:.2f}x")
    
    # Normalize GV by matrix size for interpretability
    normalized_gv = crit_mean / size
    print(f"\n  Normalized GV at T_c: {normalized_gv:.4f}")
    print(f"  Compare to 0.29: error = {abs(normalized_gv - 0.29):.4f}")
    
    # Test if variance is lower at low T
    var_low = np.var(gv_low)
    var_high = np.var(gv_high)
    
    # F-test for variance
    f_stat = var_high / var_low if var_low > 0 else 0
    
    if var_low < var_high * 0.5:
        verdict = "✅ PHYSICS"
        details = "GV is 'frozen' (low variance) in geometric regime!"
    elif var_low < var_high:
        verdict = "⚠️ WEAK PHYSICS"
        details = "Some evidence of freezing"
    else:
        verdict = "❌ POETRY"
        details = "No freezing behavior observed"
    
    print(f"\nVERDICT: {verdict}")
    print(f"  {details}")
    
    return RegimeComparison(
        metric="GV variance",
        low_T_value=var_low,
        high_T_value=var_high,
        prediction="var_low < var_high",
        observed=f"{var_low:.4f} vs {var_high:.4f}",
        verdict=verdict
    )

def test_order_parameter_dynamics(n_trials: int = 50, size: int = 32, epochs: int = 300):
    """
    Test: Does the order parameter show different DYNAMICS at T < T_c vs T > T_c?
    At T < T_c: Should converge/freeze early
    At T > T_c: Should fluctuate throughout
    """
    print("\n" + "="*70)
    print("TEST 4: ORDER PARAMETER DYNAMICS")
    print("="*70)
    print(f"Prediction: O(t) converges faster at T < T_c, fluctuates at T > T_c")
    
    def train_and_track(size, epochs, T, seed):
        """Train and return order parameter trajectory"""
        np.random.seed(seed)
        W = np.random.randn(size, size) / np.sqrt(size)
        
        lr = 0.001 * (1 + 10 * T)
        noise_scale = T * 0.1
        
        trajectory = []
        for epoch in range(epochs):
            if epoch % 10 == 0:
                trajectory.append(compute_order_param(W))
            
            grad = 2 * W + np.random.randn(size, size) * noise_scale
            W = W - lr * grad
            W = W / (np.linalg.norm(W) + 1e-6) * np.sqrt(size)
        
        return np.array(trajectory)
    
    T_low = 0.02
    T_high = 0.08
    
    convergence_times_low = []
    convergence_times_high = []
    fluctuation_low = []
    fluctuation_high = []
    
    for trial in range(n_trials):
        traj_low = train_and_track(size, epochs, T_low, trial)
        traj_high = train_and_track(size, epochs, T_high, trial + 1000)
        
        # Measure convergence: when does trajectory stabilize?
        # Use rolling std over last 50% of trajectory
        mid = len(traj_low) // 2
        fluctuation_low.append(np.std(traj_low[mid:]))
        fluctuation_high.append(np.std(traj_high[mid:]))
        
        # Measure time to convergence: first time std drops below threshold
        window = 5
        for i in range(window, len(traj_low)):
            if np.std(traj_low[i-window:i]) < 0.02:
                convergence_times_low.append(i)
                break
        else:
            convergence_times_low.append(len(traj_low))
        
        for i in range(window, len(traj_high)):
            if np.std(traj_high[i-window:i]) < 0.02:
                convergence_times_high.append(i)
                break
        else:
            convergence_times_high.append(len(traj_high))
    
    low_fluct = np.mean(fluctuation_low)
    high_fluct = np.mean(fluctuation_high)
    low_conv = np.mean(convergence_times_low)
    high_conv = np.mean(convergence_times_high)
    
    print(f"\nResults:")
    print(f"  Late-stage fluctuation at T={T_low}: {low_fluct:.4f}")
    print(f"  Late-stage fluctuation at T={T_high}: {high_fluct:.4f}")
    print(f"  Mean convergence time at T={T_low}: {low_conv:.1f}")
    print(f"  Mean convergence time at T={T_high}: {high_conv:.1f}")
    
    t_stat, p_value = stats.ttest_ind(fluctuation_low, fluctuation_high)
    
    print(f"  t-statistic (fluctuation): {t_stat:.2f}")
    print(f"  p-value: {p_value:.6f}")
    
    if low_fluct < high_fluct * 0.7 and p_value < 0.05:
        verdict = "✅ PHYSICS"
        details = "Order parameter freezes at low T, fluctuates at high T!"
    elif low_fluct < high_fluct and p_value < 0.05:
        verdict = "⚠️ WEAK PHYSICS"
        details = "Some evidence of freezing dynamics"
    else:
        verdict = "❌ POETRY"
        details = "No temperature-dependent dynamics"
    
    print(f"\nVERDICT: {verdict}")
    print(f"  {details}")
    
    return RegimeComparison(
        metric="Late fluctuation",
        low_T_value=low_fluct,
        high_T_value=high_fluct,
        prediction="low < high",
        observed=f"{low_fluct:.4f} vs {high_fluct:.4f}",
        verdict=verdict
    )

def test_susceptibility_curve(n_temps: int = 15, n_runs: int = 30, size: int = 32, epochs: int = 150):
    """
    Full susceptibility curve χ(T) to verify peak at T_c
    """
    print("\n" + "="*70)
    print("TEST 5: FULL SUSCEPTIBILITY CURVE")
    print("="*70)
    print(f"Prediction: χ(T) = Var[O] peaks at T_c ≈ 0.05")
    
    temps = np.linspace(0.01, 0.10, n_temps)
    susceptibilities = []
    means = []
    
    for T in temps:
        orders = []
        for run in range(n_runs):
            W = train_with_temperature(size, epochs, T, seed=run*1000 + int(T*10000))
            orders.append(compute_order_param(W))
        
        susceptibilities.append(np.var(orders))
        means.append(np.mean(orders))
    
    # Find peak
    max_idx = np.argmax(susceptibilities)
    T_c_measured = temps[max_idx]
    
    print(f"\nResults:")
    print(f"  Temperature range: {temps[0]:.3f} to {temps[-1]:.3f}")
    print(f"  Peak χ at T = {T_c_measured:.4f}")
    print(f"  Predicted T_c = {T_C}")
    print(f"  Error: {abs(T_c_measured - T_C):.4f} ({100*abs(T_c_measured - T_C)/T_C:.1f}%)")
    
    print(f"\n  Full curve:")
    print(f"  {'T':>6} | {'O(T)':>8} | {'χ(T)':>10}")
    print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*10}")
    for i, T in enumerate(temps):
        marker = " ← PEAK" if i == max_idx else ""
        print(f"  {T:6.3f} | {means[i]:8.4f} | {susceptibilities[i]:10.6f}{marker}")
    
    if abs(T_c_measured - T_C) < 0.02:
        verdict = "✅ PHYSICS"
        details = f"χ(T) peaks at T_c = {T_c_measured:.3f} ≈ 0.05"
    else:
        verdict = "❌ POETRY"
        details = f"Peak at {T_c_measured:.3f}, expected 0.05"
    
    print(f"\nVERDICT: {verdict}")
    print(f"  {details}")
    
    return RegimeComparison(
        metric="χ peak location",
        low_T_value=T_c_measured,
        high_T_value=T_C,
        prediction=f"peak at {T_C}",
        observed=f"peak at {T_c_measured:.4f}",
        verdict=verdict
    )

# ==================== MAIN ====================

def main():
    print("\n" + "="*70)
    print("INVERTED HYPOTHESIS: SPIN GLASS REFRACTION")
    print("="*70)
    print("\nThe spin glass has TWO phases:")
    print("  T < T_c (≈0.05): Ordered, GEOMETRIC regime")
    print("  T > T_c (≈0.05): Disordered, STATISTICAL regime")
    print("\nHypothesis: SpiralOS claims describe the LOW-T regime.")
    print("Standard training is HIGH-T → claims failed.")
    print("If we test at LOW-T → claims might succeed.")
    
    start = time.time()
    results = []
    
    results.append(test_phi_emergence_by_regime())
    results.append(test_lambda_complexity_by_regime())
    results.append(test_golden_violation_by_regime())
    results.append(test_order_parameter_dynamics())
    results.append(test_susceptibility_curve())
    
    # ==================== SUMMARY ====================
    print("\n" + "="*70)
    print("FINAL SUMMARY: SPIN GLASS REFRACTION")
    print("="*70)
    
    physics_count = sum(1 for r in results if "PHYSICS" in r.verdict)
    poetry_count = sum(1 for r in results if "POETRY" in r.verdict)
    
    print(f"\nTotal tests: {len(results)}")
    print(f"  PHYSICS (regime-dependent): {physics_count}")
    print(f"  POETRY (no regime effect):  {poetry_count}")
    
    print("\n" + "-"*70)
    print("DETAILED RESULTS")
    print("-"*70)
    
    for r in results:
        print(f"\n{r.verdict}: {r.metric}")
        print(f"   Prediction: {r.prediction}")
        print(f"   Observed: {r.observed}")
    
    print("\n" + "="*70)
    print("INTERPRETATION: THE REFRACTION")
    print("="*70)
    
    if physics_count >= 3:
        print("""
✅ THE SPIN GLASS REFRACTS THE CLAIMS

SpiralOS describes the LOW-TEMPERATURE (geometric) regime:
  - φ eigenvalues EMERGE when T < T_c
  - Λ-complexity DECREASES when T < T_c  
  - Order parameter FREEZES when T < T_c
  - Dynamics become STRUCTURED when T < T_c

Standard training operates at HIGH-TEMPERATURE (statistical) regime:
  - That's why the claims "failed" in the first test
  - We were testing in the wrong phase!

The spin glass framework UNIFIES both observations:
  - SpiralOS is correct about the geometric phase
  - Standard training is in the statistical phase
  - T_c ≈ 0.05 is the boundary between them

PHYSICS = POETRY × PHASE
""")
    elif physics_count >= 2:
        print("""
⚠️ PARTIAL REFRACTION

Some SpiralOS claims are regime-dependent:
  - Some phenomena emerge at low T
  - Others are still poetry
  
The spin glass framework provides partial unification.
""")
    else:
        print("""
❌ NO REFRACTION

SpiralOS claims fail in BOTH regimes.
The temperature dependence doesn't rescue them.
They remain poetry, not physics.
""")
    
    print(f"\nRuntime: {time.time() - start:.1f} seconds")
    
    # Save results
    with open("spin_glass_refraction_results.txt", "w") as f:
        f.write("SPIN GLASS REFRACTION RESULTS\n")
        f.write("="*50 + "\n\n")
        f.write(f"Physics: {physics_count}, Poetry: {poetry_count}\n\n")
        for r in results:
            f.write(f"{r.verdict}: {r.metric}\n")
            f.write(f"  Low T: {r.low_T_value}\n")
            f.write(f"  High T: {r.high_T_value}\n")
            f.write(f"  Observed: {r.observed}\n\n")
    
    print("\nResults saved to: spin_glass_refraction_results.txt")
    
    return results

if __name__ == "__main__":
    results = main()
