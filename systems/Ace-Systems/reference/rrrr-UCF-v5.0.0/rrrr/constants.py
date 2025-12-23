#!/usr/bin/env python3

# INTEGRITY_METADATA
# Date: 2025-12-23
# Status: NEEDS_REVIEW - Reference implementation
# Severity: MEDIUM RISK
# Risk Types: ['reference_material']
# File: systems/Ace-Systems/reference/rrrr-UCF-v5.0.0/rrrr/constants.py

"""
Rosetta Helix Core Constants
============================
Single source of truth for all physics constants.

The golden ratio identity φ⁻¹ + φ⁻² = 1 is the defining property
that ensures coupling conservation throughout the architecture.
"""

import math

# Golden Ratio and its powers
PHI = (1 + math.sqrt(5)) / 2                    # ≈ 1.618034
PHI_INV = 1 / PHI                               # ≈ 0.618034
PHI_SQ = PHI ** 2                               # ≈ 2.618034
PHI_INV_SQ = PHI_INV ** 2                       # ≈ 0.381966

# Critical thresholds derived from quasicrystal geometry
Z_CRITICAL = math.sqrt(3) / 2                   # ≈ 0.866025 (THE LENS)
Z_ORIGIN = Z_CRITICAL * PHI_INV                 # ≈ 0.535234 (collapse reset)

# μ Threshold Hierarchy (consciousness gates)
MU_P = 2.0 / (PHI ** 2.5)                       # Paradox threshold
MU_1 = MU_P / math.sqrt(PHI)                    # Sub-paradox
MU_2 = MU_P * math.sqrt(PHI)                    # Amplified paradox
MU_S = 0.920                                    # K-formation threshold (≈ KAPPA_S)
MU_3 = 0.992                                    # Ultra-integration / teachability
KAPPA_S = MU_S                                  # Consciousness gate (alias)
UNITY = 0.9999                                  # Collapse trigger

# TRIAD thresholds (hysteresis band for stable high-z)
TRIAD_HIGH = 0.85                               # Enter band from above
TRIAD_LOW = 0.82                                # Exit band from below
TRIAD_T6 = 0.83                                 # t6 gate on TRIAD unlock
TRIAD_PASSES_REQUIRED = 3                       # Passes needed for unlock

# S₃ Group parameters
LENS_SIGMA = 36.0                               # ΔS_neg Gaussian width

# Tier boundaries (from sub-quantum to unity)
TIER_BOUNDS = [
    0.0,            # t1 min
    0.10,           # t2 min
    0.20,           # t3 min
    0.40,           # t4 min
    0.60,           # t5 min
    0.75,           # t6 min (approach to lens)
    Z_CRITICAL,     # t7 min (THE LENS)
    0.92,           # t8 min (≈ KAPPA_S)
    0.97,           # t9 min (approach to unity)
    1.0             # upper bound
]

# APL Operators (S₃ permutation group)
APL_OPERATORS = ['()', '^', '+', '×', '÷', '−']

# S₃ Group structure
S3_EVEN = ['()', '×', '^']      # Even parity: e, σ, σ²
S3_ODD = ['+', '÷', '−']        # Odd parity: τ₂, τ₁, τ₃

# Tier-gated operator availability
TIER_OPERATORS = {
    1: [0, 4, 5],           # (), ÷, −
    2: [1, 4, 5, 3],        # ^, ÷, −, ×
    3: [2, 1, 5, 4, 0],     # +, ^, −, ÷, ()
    4: [0, 4, 5, 2],        # (), ÷, −, +
    5: [0, 1, 2, 3, 4, 5],  # All operators
    6: [0, 5, 2, 4],        # (), −, +, ÷
    7: [0, 2],              # (), +
    8: [0, 2, 1],           # (), +, ^
    9: [0, 2, 1],           # (), +, ^
}


def verify_phi_identity():
    """
    Verify the golden ratio identity: φ⁻¹ + φ⁻² = 1
    
    This is THE defining property of φ - the unique positive solution to c + c² = 1.
    """
    identity_sum = PHI_INV + PHI_INV_SQ
    error = abs(identity_sum - 1.0)
    assert error < 1e-14, f"Golden ratio identity violated: {identity_sum} ≠ 1"
    return True


def get_tier(z: float) -> int:
    """Get the tier number (1-9) for a given z-coordinate."""
    for i in range(len(TIER_BOUNDS) - 1):
        if TIER_BOUNDS[i] <= z < TIER_BOUNDS[i + 1]:
            return i + 1
    return 9


def get_delta_s_neg(z: float) -> float:
    """
    Compute ΔS_neg = exp(-σ(z - z_c)²)
    
    The negentropy measure peaks at Z_CRITICAL (THE LENS).
    """
    return math.exp(-LENS_SIGMA * (z - Z_CRITICAL) ** 2)


def get_legal_operators(z: float) -> list:
    """Get list of legal operators for current tier."""
    tier = get_tier(z)
    indices = TIER_OPERATORS.get(tier, [0])
    return [APL_OPERATORS[i] for i in indices]


# Verify on import
verify_phi_identity()
