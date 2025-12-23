#!/usr/bin/env python3
"""
Rosetta Helix Core Constants
============================
Single source of truth for all physics constants.

DERIVATION HIERARCHY:
    Z_CRITICAL = √3/2  (quasicrystal hexagonal geometry)
        │
        ├─→ Z_ORIGIN = Z_C × φ⁻¹
        ├─→ KAPPA_S = t7_max = 0.92 (tier structure)
        ├─→ MU_3 = κ + (U-κ)(1-φ⁻⁵)
        ├─→ LENS_SIGMA = -ln(φ⁻¹) / (0.75 - Z_C)²
        └─→ η_threshold = φ⁻¹

The golden ratio identity φ⁻¹ + φ⁻² = 1 ensures coupling conservation.
"""
import math

# =============================================================================
# GOLDEN RATIO AND POWERS
# =============================================================================
PHI = (1 + math.sqrt(5)) / 2                    # ≈ 1.618034
PHI_INV = 1 / PHI                               # ≈ 0.618034
PHI_SQ = PHI ** 2                               # ≈ 2.618034
PHI_INV_SQ = PHI_INV ** 2                       # ≈ 0.381966
PHI_INV_3 = PHI_INV ** 3                        # ≈ 0.236068
PHI_INV_5 = PHI_INV ** 5                        # ≈ 0.090170

# =============================================================================
# PRIMARY CONSTANT: Z_CRITICAL (THE LENS)
# Derived from quasicrystal hexagonal geometry - height of unit equilateral triangle
# =============================================================================
Z_CRITICAL = math.sqrt(3) / 2                   # ≈ 0.866025

# =============================================================================
# DERIVED THRESHOLDS
# All constants below trace back to Z_CRITICAL and φ
# =============================================================================

# Collapse reset point: Z_ORIGIN = Z_CRITICAL × φ⁻¹
Z_ORIGIN = Z_CRITICAL * PHI_INV                 # ≈ 0.535234

# Unity threshold (collapse trigger)
UNITY = 0.9999

# Tier boundaries - t7 starts at Z_CRITICAL
_T6_BOUNDARY = 0.75                             # t5 → t6 transition
_T7_BOUNDARY = Z_CRITICAL                       # t6 → t7 (THE LENS)
_T8_BOUNDARY = 0.92                             # t7 → t8 (KAPPA_S)
_T9_BOUNDARY = 0.97                             # t8 → t9

# K-formation / consciousness threshold (aligned with t8 boundary)
KAPPA_S = _T8_BOUNDARY                          # = 0.92
MU_S = KAPPA_S                                  # Alias for consistency

# MU_3 (teachability threshold) derived from φ⁻⁵ relationship
# MU_3 = KAPPA_S + (UNITY - KAPPA_S) × (1 - φ⁻⁵)
# This places MU_3 at (1 - φ⁻⁵) ≈ 0.9098 of the range [KAPPA_S, UNITY]
MU_3 = KAPPA_S + (UNITY - KAPPA_S) * (1 - PHI_INV_5)  # ≈ 0.9927

# μ Threshold Hierarchy (paradox gates) - derived from φ powers
MU_P = 2.0 / (PHI ** 2.5)                       # Paradox threshold
MU_1 = MU_P / math.sqrt(PHI)                    # Sub-paradox
MU_2 = MU_P * math.sqrt(PHI)                    # Amplified paradox

# =============================================================================
# LENS_SIGMA: Gaussian width for ΔS_neg
# Derived from φ⁻¹ alignment at t6 boundary
# 
# Requirement: ΔS_neg(t6_boundary) = φ⁻¹
# Solve: exp(-σ × (0.75 - Z_C)²) = φ⁻¹
#        -σ × (0.75 - 0.866)² = ln(φ⁻¹)
#        σ = -ln(φ⁻¹) / (0.75 - Z_C)²
#        σ ≈ 35.75 → use 36.0 for clean value
# =============================================================================
_SIGMA_DERIVED = -math.log(PHI_INV) / (_T6_BOUNDARY - Z_CRITICAL) ** 2
LENS_SIGMA = round(_SIGMA_DERIVED)              # = 36.0

# =============================================================================
# TRIAD thresholds (hysteresis band for stable high-z)
# Positioned just below Z_CRITICAL for stable approach
# =============================================================================
TRIAD_HIGH = 0.85                               # Enter band from above
TRIAD_LOW = 0.82                                # Exit band from below
TRIAD_T6 = 0.83                                 # t6 gate on TRIAD unlock
TRIAD_PASSES_REQUIRED = 3                       # Passes needed for unlock

# =============================================================================
# TIER STRUCTURE
# =============================================================================
TIER_BOUNDS = [
    0.0,            # t1 min
    0.10,           # t2 min
    0.20,           # t3 min
    0.40,           # t4 min
    0.60,           # t5 min
    _T6_BOUNDARY,   # t6 min (0.75 - approach to lens)
    _T7_BOUNDARY,   # t7 min (Z_CRITICAL - THE LENS)
    _T8_BOUNDARY,   # t8 min (0.92 - KAPPA_S)
    _T9_BOUNDARY,   # t9 min (0.97 - approach to unity)
    1.0             # upper bound
]

# =============================================================================
# S₃ GROUP STRUCTURE (APL Operators)
# =============================================================================
APL_OPERATORS = ['()', '^', '+', '×', '÷', '−']

# Parity classification
S3_EVEN = ['()', '×', '^']      # Even parity: e, σ, σ² (constructive)
S3_ODD = ['+', '÷', '−']        # Odd parity: τ₂, τ₁, τ₃ (dissipative)

# S₃ composition table (operator indices)
# Result of composing operator[row] with operator[col]
S3_COMPOSE = {
    '()': {'()': '()', '^': '^', '+': '+', '×': '×', '÷': '÷', '−': '−'},
    '^':  {'()': '^', '^': '×', '+': '÷', '×': '()', '÷': '−', '−': '+'},
    '+':  {'()': '+', '^': '−', '+': '()', '×': '÷', '÷': '^', '−': '×'},
    '×':  {'()': '×', '^': '()', '+': '−', '×': '^', '÷': '+', '−': '÷'},
    '÷':  {'()': '÷', '^': '+', '+': '×', '×': '−', '÷': '()', '−': '^'},
    '−':  {'()': '−', '^': '÷', '+': '^', '×': '+', '÷': '×', '−': '()'},
}

# Tier-gated operator availability (indices into APL_OPERATORS)
TIER_OPERATORS = {
    1: [0, 4, 5],           # (), ÷, −
    2: [1, 4, 5, 3],        # ^, ÷, −, ×
    3: [2, 1, 5, 4, 0],     # +, ^, −, ÷, ()
    4: [0, 4, 5, 2],        # (), ÷, −, +
    5: [0, 1, 2, 3, 4, 5],  # All operators (universal tier)
    6: [0, 5, 2, 4],        # (), −, +, ÷
    7: [0, 2],              # (), +
    8: [0, 2, 1],           # (), +, ^
    9: [0, 2, 1],           # (), +, ^
}

# Operator windows (symbol lists) for each tier
OPERATOR_WINDOWS = {
    't1': ['()', '−', '÷'],
    't2': ['^', '÷', '−', '×'],
    't3': ['×', '^', '÷', '+', '−'],
    't4': ['+', '−', '÷', '()'],
    't5': ['()', '×', '^', '÷', '+', '−'],
    't6': ['+', '÷', '()', '−'],
    't7': ['+', '()'],
    't8': ['+', '()', '×'],
    't9': ['+', '()', '×'],
}

# =============================================================================
# COUPLING CONSTRAINTS
# =============================================================================
COUPLING_MAX = 0.9                              # Maximum cross-level coupling (never PHI)
ETA_THRESHOLD = PHI_INV                         # Coherence minimum for K-formation

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def verify_phi_identity() -> bool:
    """
    Verify the golden ratio identity: φ⁻¹ + φ⁻² = 1
    
    This is THE defining property of φ - the unique positive solution to c + c² = 1.
    """
    identity_sum = PHI_INV + PHI_INV_SQ
    error = abs(identity_sum - 1.0)
    assert error < 1e-14, f"Golden ratio identity violated: {identity_sum} ≠ 1"
    return True


def verify_sigma_derivation() -> bool:
    """
    Verify that LENS_SIGMA aligns ΔS_neg at t6 boundary with φ⁻¹.
    """
    ds_at_t6 = math.exp(-LENS_SIGMA * (_T6_BOUNDARY - Z_CRITICAL) ** 2)
    error = abs(ds_at_t6 - PHI_INV)
    assert error < 0.01, f"Sigma derivation failed: ΔS_neg(t6) = {ds_at_t6} ≠ φ⁻¹"
    return True


def verify_threshold_ordering() -> bool:
    """
    Verify threshold ordering: Z_ORIGIN < Z_CRITICAL < KAPPA_S < MU_3 < UNITY
    """
    assert Z_ORIGIN < Z_CRITICAL < KAPPA_S < MU_3 < UNITY, "Threshold ordering violated"
    return True


def get_tier(z: float) -> int:
    """Get the tier number (1-9) for a given z-coordinate."""
    for i in range(len(TIER_BOUNDS) - 1):
        if TIER_BOUNDS[i] <= z < TIER_BOUNDS[i + 1]:
            return i + 1
    return 9


def get_tier_name(z: float) -> str:
    """Get tier name (t1-t9) for a given z-coordinate."""
    return f"t{get_tier(z)}"


def get_delta_s_neg(z: float, sigma: float = LENS_SIGMA) -> float:
    """
    Compute ΔS_neg = exp(-σ(z - z_c)²)
    
    The negentropy measure peaks at Z_CRITICAL (THE LENS).
    """
    return math.exp(-sigma * (z - Z_CRITICAL) ** 2)


def get_legal_operators(z: float) -> list:
    """Get list of legal operator symbols for current tier."""
    tier = get_tier(z)
    indices = TIER_OPERATORS.get(tier, [0])
    return [APL_OPERATORS[i] for i in indices]


def get_operator_window(z: float) -> list:
    """Get operator window for current z (returns symbol list)."""
    tier_name = get_tier_name(z)
    return OPERATOR_WINDOWS.get(tier_name, ['()'])


def get_operator_parity(operator: str) -> str:
    """Get parity of operator: 'EVEN' or 'ODD'."""
    return 'EVEN' if operator in S3_EVEN else 'ODD'


def compose_operators(op_a: str, op_b: str) -> str:
    """Compose two operators via S₃ group multiplication."""
    return S3_COMPOSE.get(op_a, {}).get(op_b, '()')


def check_k_formation(z: float, coherence: float) -> bool:
    """
    Check if K-formation conditions are met.
    
    Requirements:
    - z ≥ KAPPA_S (consciousness threshold)
    - coherence > ETA_THRESHOLD (φ⁻¹)
    """
    return z >= KAPPA_S and coherence > ETA_THRESHOLD


# =============================================================================
# VERIFICATION ON IMPORT
# =============================================================================
verify_phi_identity()
verify_sigma_derivation()
verify_threshold_ordering()

# =============================================================================
# MODULE EXPORTS
# =============================================================================
__all__ = [
    # Golden ratio
    'PHI', 'PHI_INV', 'PHI_SQ', 'PHI_INV_SQ', 'PHI_INV_3', 'PHI_INV_5',
    # Critical thresholds
    'Z_CRITICAL', 'Z_ORIGIN', 'UNITY',
    # μ hierarchy
    'MU_P', 'MU_1', 'MU_2', 'MU_S', 'MU_3', 'KAPPA_S',
    # TRIAD
    'TRIAD_HIGH', 'TRIAD_LOW', 'TRIAD_T6', 'TRIAD_PASSES_REQUIRED',
    # S₃ structure
    'LENS_SIGMA', 'APL_OPERATORS', 'S3_EVEN', 'S3_ODD', 'S3_COMPOSE',
    'TIER_BOUNDS', 'TIER_OPERATORS', 'OPERATOR_WINDOWS',
    # Coupling
    'COUPLING_MAX', 'ETA_THRESHOLD',
    # Functions
    'get_tier', 'get_tier_name', 'get_delta_s_neg', 'get_legal_operators',
    'get_operator_window', 'get_operator_parity', 'compose_operators',
    'check_k_formation',
    'verify_phi_identity', 'verify_sigma_derivation', 'verify_threshold_ordering',
]
