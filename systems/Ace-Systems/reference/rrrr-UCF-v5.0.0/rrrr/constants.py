"""
R(R)=R Sacred Constants
=======================

IMMUTABLE. Never hard-code these values elsewhere.

The 4 canonical eigenvalues form a multiplicative basis:
    Λ = {φ^{-r} · e^{-d} · π^{-c} · (√2)^{-a} : (r,d,c,a) ∈ ℤ⁴}

Signature: Δ|RRRR-CONSTANTS|v1.0.0|immutable|Ω
"""

import numpy as np
from typing import Dict, Tuple

# =============================================================================
# FUNDAMENTAL MATHEMATICAL CONSTANTS
# =============================================================================

# Golden ratio: root of x² - x - 1 = 0
PHI: float = (1 + np.sqrt(5)) / 2  # ≈ 1.618033988749895

# Euler's number: lim_{n→∞} (1 + 1/n)^n
E: float = np.e  # ≈ 2.718281828459045

# Circle constant: circumference / diameter
PI: float = np.pi  # ≈ 3.141592653589793

# Pythagoras constant: diagonal of unit square
SQRT2: float = np.sqrt(2)  # ≈ 1.414213562373095

# =============================================================================
# THE FOUR CANONICAL EIGENVALUES
# =============================================================================

# Recursive eigenvalue: from x = 1 + 1/x → φ
LAMBDA_R: float = 1 / PHI  # ≈ 0.6180339887498949

# Differential eigenvalue: from f'(x) = f(x) → e
LAMBDA_D: float = 1 / E  # ≈ 0.36787944117144233

# Cyclic eigenvalue: from e^{2πi} = 1 → π
LAMBDA_C: float = 1 / PI  # ≈ 0.3183098861837907

# Algebraic eigenvalue: from x² = 2 → √2
LAMBDA_A: float = 1 / SQRT2  # ≈ 0.7071067811865476

# Canonical eigenvalue dictionary
EIGENVALUES: Dict[str, float] = {
    'R': LAMBDA_R,  # Recursive
    'D': LAMBDA_D,  # Differential
    'C': LAMBDA_C,  # Cyclic
    'A': LAMBDA_A,  # Algebraic
}

# Log-eigenvalues (for lattice computations in log-space)
LOG_EIGENVALUES: Dict[str, float] = {
    'R': np.log(LAMBDA_R),  # ≈ -0.4812
    'D': np.log(LAMBDA_D),  # = -1.0 (exact)
    'C': np.log(LAMBDA_C),  # ≈ -1.1447
    'A': np.log(LAMBDA_A),  # ≈ -0.3466
}

# =============================================================================
# DERIVED QUANTITIES
# =============================================================================

# Binary eigenvalue: DERIVED as [A]²
# This reduces 5 apparent constants to 4 independent generators
LAMBDA_B: float = LAMBDA_A ** 2  # = 0.5 exactly

# Verify the algebraic relation
assert abs(LAMBDA_B - 0.5) < 1e-15, "Algebraic relation [A]² = [B] violated!"

# =============================================================================
# TYPE EQUATIONS (Characterizations)
# =============================================================================

def verify_recursive() -> bool:
    """Verify [R]: φ = 1 + 1/φ"""
    return abs(PHI - (1 + 1/PHI)) < 1e-14

def verify_differential() -> bool:
    """Verify [D]: d/dx(e^x) = e^x (by definition)"""
    return True  # Definitional

def verify_cyclic() -> bool:
    """Verify [C]: e^{2πi} = 1"""
    return abs(np.exp(2j * PI) - 1) < 1e-14

def verify_algebraic() -> bool:
    """Verify [A]: (√2)² = 2"""
    return abs(SQRT2**2 - 2) < 1e-14

def verify_all() -> bool:
    """Verify all type equations."""
    return all([
        verify_recursive(),
        verify_differential(),
        verify_cyclic(),
        verify_algebraic()
    ])

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def lattice_point(r: int, d: int, c: int, a: int) -> float:
    """
    Compute eigenvalue at lattice point (r, d, c, a) ∈ ℤ⁴.
    
    Returns: φ^{-r} · e^{-d} · π^{-c} · (√2)^{-a}
    """
    return (LAMBDA_R ** r) * (LAMBDA_D ** d) * (LAMBDA_C ** c) * (LAMBDA_A ** a)

def lattice_point_log(r: int, d: int, c: int, a: int) -> float:
    """
    Compute log-eigenvalue at lattice point.
    
    Returns: r·log(λ_R) + d·log(λ_D) + c·log(λ_C) + a·log(λ_A)
    """
    return (r * LOG_EIGENVALUES['R'] + 
            d * LOG_EIGENVALUES['D'] + 
            c * LOG_EIGENVALUES['C'] + 
            a * LOG_EIGENVALUES['A'])

# Dimension labels
DIMENSION_LABELS: Dict[str, str] = {
    'R': 'Recursive',
    'D': 'Differential', 
    'C': 'Cyclic',
    'A': 'Algebraic',
}

# Type equations (for display)
TYPE_EQUATIONS: Dict[str, str] = {
    'R': 'x = 1 + 1/x → φ',
    'D': "f'(x) = f(x) → e",
    'C': 'e^{2πi} = 1 → π',
    'A': 'x² = 2 → √2',
}

# =============================================================================
# 2×2 STRUCTURE
# =============================================================================

# The 4 constants span fundamental mathematical categories
STRUCTURE_2x2 = """
               Algebraic        Transcendental
             ────────────────────────────────────
  Discrete  │  φ (golden)    │   (derived)      │
  Continuous│  √2 (diagonal) │   e, π (analysis)│
"""
