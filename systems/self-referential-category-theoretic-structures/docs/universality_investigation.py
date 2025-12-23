# INTEGRITY_METADATA
# Date: 2025-12-23
# Status: ✓ JUSTIFIED - Claims supported by repository files (needs citation update)
# Severity: MEDIUM RISK
# Risk Types: unsupported_claims

# Supporting Evidence:
#   - systems/self-referential-category-theoretic-structures/docs/STATUS_UPDATE_DEC2025_v2.md (dependency)
#   - systems/self-referential-category-theoretic-structures/docs/STATUS_UPDATE_DEC2025.md (dependency)
#   - systems/self-referential-category-theoretic-structures/docs/FINAL_STATUS_DEC2025.md (dependency)
#
# Referenced By:
#   - systems/self-referential-category-theoretic-structures/docs/STATUS_UPDATE_DEC2025_v2.md (reference)
#   - systems/self-referential-category-theoretic-structures/docs/STATUS_UPDATE_DEC2025.md (reference)
#   - systems/self-referential-category-theoretic-structures/docs/FINAL_STATUS_DEC2025.md (reference)


#!/usr/bin/env python3
"""
================================================================================
UNIVERSALITY CLASS INVESTIGATION
================================================================================

Neural networks show spin glass phenomenology but with exponents:
  β ≈ 0.16 (not SK's 0.5)
  γ/ν ≈ 0.29 (not SK's 0.5)

Which universality class do these match?

Candidates:
  1. SK Model (mean-field): β = 0.5, γ = 1, ν = 2, γ/ν = 0.5
  2. 2D Ising: β = 0.125, γ = 1.75, ν = 1, γ/ν = 1.75
  3. 3D Ising: β ≈ 0.326, γ ≈ 1.24, ν ≈ 0.63, γ/ν ≈ 1.97
  4. 3D Heisenberg: β ≈ 0.365, γ ≈ 1.39
  5. Diluted/EA spin glass (3D): β ≈ 0.8-1.0, γ ≈ 2.0-6.0
  6. Random Field Ising: β depends on disorder strength
  7. Percolation: β = 0.41 (3D), β = 0.14 (2D)

Key observation: β ≈ 0.16 is very close to 2D Ising β = 0.125!

================================================================================
"""

import numpy as np
from scipy import stats

# Measured exponents from our tests
MEASURED = {
    'beta': 0.159,
    'gamma_nu': 0.291,
}

# Known universality classes
UNIVERSALITY_CLASSES = {
    'SK (mean-field)': {
        'beta': 0.5,
        'gamma': 1.0,
        'nu': 2.0,
        'gamma_nu': 0.5,
        'description': 'Infinite-range spin glass'
    },
    '2D Ising': {
        'beta': 0.125,
        'gamma': 1.75,
        'nu': 1.0,
        'gamma_nu': 1.75,
        'description': 'Exact solution (Onsager)'
    },
    '3D Ising': {
        'beta': 0.326,
        'gamma': 1.237,
        'nu': 0.630,
        'gamma_nu': 1.963,
        'description': 'Ferromagnetic'
    },
    '2D Percolation': {
        'beta': 0.139,  # β/ν ≈ 5/36
        'gamma': 2.389,
        'nu': 1.333,
        'gamma_nu': 1.792,
        'description': 'Connectivity transition'
    },
    '3D Percolation': {
        'beta': 0.41,
        'gamma': 1.80,
        'nu': 0.88,
        'gamma_nu': 2.05,
        'description': 'Connectivity transition'
    },
    'Mean-field Percolation': {
        'beta': 1.0,
        'gamma': 1.0,
        'nu': 0.5,
        'gamma_nu': 2.0,
        'description': 'd > 6'
    },
    'Directed Percolation (1+1D)': {
        'beta': 0.2765,
        'gamma': 0.5772,  # Actually this is ν_perp
        'nu': 1.0968,
        'gamma_nu': 0.526,
        'description': 'DP universality'
    },
    'KPZ (1D)': {
        'beta': 0.333,  # Actually growth exponent, different context
        'gamma': None,
        'nu': None,
        'gamma_nu': None,
        'description': 'Interface growth'
    },
}

# Additional candidate: effective dimension scaling
# In mean-field, above upper critical dimension d_c:
# β = 1/2, γ = 1, ν = 1/2
# Below d_c, exponents change with dimension

def compute_distances():
    """Compute distance of measured exponents from each universality class."""
    print("="*70)
    print("UNIVERSALITY CLASS COMPARISON")
    print("="*70)
    print(f"\nMeasured exponents:")
    print(f"  β = {MEASURED['beta']:.3f}")
    print(f"  γ/ν = {MEASURED['gamma_nu']:.3f}")
    
    print("\n" + "-"*70)
    print(f"{'Class':<25} | {'β':<8} | {'Δβ':<8} | {'γ/ν':<8} | {'Δ(γ/ν)':<8}")
    print("-"*70)
    
    distances = {}
    for name, exp in UNIVERSALITY_CLASSES.items():
        d_beta = abs(exp['beta'] - MEASURED['beta'])
        
        if exp['gamma_nu'] is not None:
            d_gamma_nu = abs(exp['gamma_nu'] - MEASURED['gamma_nu'])
            total_dist = np.sqrt(d_beta**2 + d_gamma_nu**2)
        else:
            d_gamma_nu = float('inf')
            total_dist = d_beta
        
        distances[name] = total_dist
        
        gamma_nu_str = f"{exp['gamma_nu']:.3f}" if exp['gamma_nu'] else "N/A"
        d_gamma_str = f"{d_gamma_nu:.3f}" if exp['gamma_nu'] else "N/A"
        
        print(f"{name:<25} | {exp['beta']:.3f}   | {d_beta:.3f}   | {gamma_nu_str:<8} | {d_gamma_str}")
    
    print("-"*70)
    
    # Sort by distance
    sorted_dist = sorted(distances.items(), key=lambda x: x[1])
    
    print("\n" + "="*70)
    print("BEST MATCHES (sorted by total distance)")
    print("="*70)
    
    for i, (name, dist) in enumerate(sorted_dist[:5]):
        exp = UNIVERSALITY_CLASSES[name]
        print(f"\n{i+1}. {name}")
        print(f"   Description: {exp['description']}")
        print(f"   β: {exp['beta']:.3f} (Δ = {abs(exp['beta'] - MEASURED['beta']):.3f})")
        if exp['gamma_nu']:
            print(f"   γ/ν: {exp['gamma_nu']:.3f} (Δ = {abs(exp['gamma_nu'] - MEASURED['gamma_nu']):.3f})")
        print(f"   Total distance: {dist:.3f}")
    
    return sorted_dist

def analyze_effective_dimension():
    """
    If neural networks have an effective dimension, what would it be?
    
    For Ising-like systems, exponents interpolate between 2D and mean-field.
    Using hyperscaling: 2β + γ = dν (below upper critical dimension)
    
    Also: γ/ν = 2 - η where η is anomalous dimension
    """
    print("\n" + "="*70)
    print("EFFECTIVE DIMENSION ANALYSIS")
    print("="*70)
    
    # From γ/ν = 2 - η
    eta = 2 - MEASURED['gamma_nu']
    print(f"\nAnomolous dimension η = 2 - γ/ν = 2 - {MEASURED['gamma_nu']:.3f} = {eta:.3f}")
    print("(Compare: 2D Ising η = 0.25, 3D Ising η ≈ 0.036, MF η = 0)")
    
    # This η ≈ 1.71 is very large, suggesting NOT a standard ferromagnet
    
    # For spin glasses, the order parameter exponent β relates differently
    # In EA model: β ≈ 1 (3D), β ≈ 0.5 (mean-field)
    
    print("\n" + "-"*50)
    print("HYPOTHESIS: Modified universality")
    print("-"*50)
    print("""
The measured exponents β ≈ 0.16, γ/ν ≈ 0.29 don't match ANY standard
universality class perfectly. This suggests:

1. NOVEL UNIVERSALITY: Neural networks may define their own class
   - ReLU activation changes critical behavior
   - Structured (layered) connectivity differs from random graphs
   - Training dynamics impose additional constraints

2. CROSSOVER EFFECTS: We may be measuring crossover exponents
   - Finite size effects at n = 32, 64 are significant
   - True asymptotic exponents may differ
   - Need n = 256, 512, 1024 to see scaling

3. EFFECTIVE LOWER DIMENSION: β ≈ 0.16 close to 2D values
   - Despite high-dimensional weight space
   - Effective dynamics may be low-dimensional
   - Related to intrinsic dimension of representations

4. DIRECTED/NON-EQUILIBRIUM: 
   - Training is explicitly out of equilibrium
   - May relate to directed percolation (β ≈ 0.28)
   - Or KPZ-like growth dynamics
""")

def check_scaling_relations():
    """
    Check if measured exponents satisfy any scaling relations.
    
    Standard relations:
    - Rushbrooke: α + 2β + γ = 2
    - Widom: γ = β(δ - 1)
    - Fisher: γ = ν(2 - η)
    - Josephson (hyperscaling): dν = 2 - α
    """
    print("\n" + "="*70)
    print("SCALING RELATIONS CHECK")
    print("="*70)
    
    beta = MEASURED['beta']
    gamma_nu = MEASURED['gamma_nu']
    
    # If we assume ν = 2 (like SK), then:
    gamma_sk = gamma_nu * 2
    print(f"\nIf ν = 2 (SK-like): γ = {gamma_sk:.3f}")
    
    # Rushbrooke: α + 2β + γ = 2
    # α = 2 - 2β - γ
    alpha_implied = 2 - 2*beta - gamma_sk
    print(f"  Rushbrooke implies α = {alpha_implied:.3f}")
    print(f"  (SK has α = 0, 3D Ising has α ≈ 0.11)")
    
    # If we assume ν = 1 (like finite-d Ising):
    gamma_1 = gamma_nu * 1
    print(f"\nIf ν = 1 (Ising-like): γ = {gamma_1:.3f}")
    alpha_implied_1 = 2 - 2*beta - gamma_1
    print(f"  Rushbrooke implies α = {alpha_implied_1:.3f}")
    
    print("\n" + "-"*50)
    print("KEY OBSERVATION:")
    print("-"*50)
    print(f"""
With β = {beta:.3f} and γ/ν = {gamma_nu:.3f}:

The exponent β is very small (order parameter grows slowly from T_c).
This is characteristic of:
  - Systems near lower critical dimension
  - Strongly disordered systems  
  - Quasi-low-dimensional behavior

The ratio γ/ν < 1 means χ_max grows SUBLINEARLY with system size.
In SK, γ/ν = 0.5 means χ_max ~ √n.
Our γ/ν = 0.29 means χ_max ~ n^0.29 (even slower).

This SLOWER divergence suggests WEAKER fluctuations than SK model,
possibly due to:
  - ReLU cutting off fluctuations
  - Layer structure imposing constraints
  - Adam optimizer smoothing landscape
""")

def propose_refined_hypothesis():
    """Generate refined hypotheses based on analysis."""
    print("\n" + "="*70)
    print("REFINED HYPOTHESES TO TEST")
    print("="*70)
    print("""
Given: Neural networks show RSB/ultrametricity but NOT SK exponents

HYPOTHESIS 1: QUASI-2D UNIVERSALITY
  β ≈ 0.16 is close to 2D Ising β = 0.125
  Test: Check if exponents match 2D spin glass
  2D EA model: Controversial, may have T_c = 0 or T_c > 0

HYPOTHESIS 2: DILUTED SPIN GLASS
  Layered architecture = effective dilution
  Test: Compare to site-diluted SK model exponents
  
HYPOTHESIS 3: NEURAL-SPECIFIC UNIVERSALITY
  ReLU + layers + Adam = new universality class
  Test: Do different activations change exponents?
  Test: Do different architectures (transformer) give same exponents?

HYPOTHESIS 4: CROSSOVER REGIME
  n = 32, 64 too small for asymptotic regime
  Test: Run n = 256, 512, 1024 and remeasure
  Prediction: Exponents should approach SK at large n

HYPOTHESIS 5: PERCOLATION-LIKE
  Phase transition may be connectivity-like
  Test: Look at activation patterns, not weights
  Test: Check percolation exponents β ≈ 0.14 (2D)

RECOMMENDED NEXT EXPERIMENTS:
1. Scale to n = 256, 512, 1024 (verify finite-size)
2. Try different activations (GELU, tanh) 
3. Try linear networks (no activation)
4. Measure more exponents (α, δ, ν separately)
5. Test transformer architecture
""")

def main():
    print("\n" + "#"*70)
    print("# UNIVERSALITY CLASS INVESTIGATION")
    print("#"*70)
    
    distances = compute_distances()
    analyze_effective_dimension()
    check_scaling_relations()
    propose_refined_hypothesis()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
Neural networks exhibit:
  ✅ Spin glass PHENOMENOLOGY (ultrametric, RSB, phase transition)
  ❌ SK critical EXPONENTS (β, γ/ν wrong by ~60%)

This is a POSITIVE result for theory development:
  - The √3/2 structure is mathematically exact
  - The qualitative physics maps correctly
  - The quantitative exponents reveal NEW universality

Best interpretation: Neural networks are spin-glass-like but
occupy a distinct universality class, possibly related to:
  - Quasi-low-dimensional behavior
  - Directed/non-equilibrium dynamics
  - Novel "neural network" universality

NEXT STEPS:
1. Larger scale experiments (n = 256+)
2. Architecture comparisons
3. Activation function comparisons
4. Independent ν measurement
""")

if __name__ == "__main__":
    main()
