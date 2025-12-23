# INTEGRITY_METADATA
# Date: 2025-12-23
# Status: NEEDS_REVIEW - Reference implementation
# Severity: MEDIUM RISK
# Risk Types: ['reference_material']
# File: systems/Ace-Systems/reference/rrrr-UCF-v5.0.0/rrrr/__init__.py

"""
R(R)=R Framework
================

A Four-Dimensional Basis for Self-Referential Computation

The four canonical eigenvalues:
    λ_R = φ⁻¹ ≈ 0.618  (Recursive)
    λ_D = e⁻¹ ≈ 0.368  (Differential)
    λ_C = π⁻¹ ≈ 0.318  (Cyclic)
    λ_A = √2⁻¹ ≈ 0.707 (Algebraic)

Usage:
    from rrrr import EIGENVALUES, decompose, analyze_ntk
    
    # Get canonical eigenvalues
    print(EIGENVALUES['R'])  # 0.618...
    
    # Decompose arbitrary eigenvalue
    result = decompose(0.5)
    print(result)  # [A]² = 0.5
    
    # Analyze NTK eigenvalues
    analysis = analyze_ntk('resnet')
    print(analysis.mean_error)  # < 1%

Signature: Δ|RRRR-FRAMEWORK|v1.0.0|complete|Ω
"""

__version__ = "1.0.0"
__author__ = "Anonymous"

# Constants
from .constants import (
    # Fundamental constants
    PHI, E, PI, SQRT2,
    # Canonical eigenvalues
    LAMBDA_R, LAMBDA_D, LAMBDA_C, LAMBDA_A, LAMBDA_B,
    EIGENVALUES, LOG_EIGENVALUES,
    # Dimension info
    DIMENSION_LABELS, TYPE_EQUATIONS,
    # Utility functions
    lattice_point, lattice_point_log,
    verify_all as verify_constants
)

# Lattice operations
from .lattice import (
    LatticePoint, Decomposition,
    decompose, verify_density,
    CANONICAL_POINTS
)

# Composition algebra
from .composition import (
    ArchitectureSignature,
    COMPONENTS, ARCHITECTURES,
    compose, analyze_architecture,
    predict_scaling_exponent,
    print_architecture_analysis
)

# NTK analysis
from .ntk import (
    NTKAnalysis,
    compute_proxy_ntk, get_ntk_eigenvalues,
    analyze_ntk, analyze_all_architectures,
    print_ntk_analysis, print_summary
)

# Verification
from .verify import (
    run_all_verifications,
    print_verification_report,
    main as verify
)

# Convenience exports
__all__ = [
    # Version
    '__version__',
    
    # Constants
    'PHI', 'E', 'PI', 'SQRT2',
    'LAMBDA_R', 'LAMBDA_D', 'LAMBDA_C', 'LAMBDA_A', 'LAMBDA_B',
    'EIGENVALUES', 'LOG_EIGENVALUES',
    
    # Classes
    'LatticePoint', 'Decomposition', 'ArchitectureSignature', 'NTKAnalysis',
    
    # Core functions
    'decompose', 'compose', 'analyze_ntk', 'analyze_architecture',
    
    # Analysis
    'verify', 'verify_density', 'analyze_all_architectures',
    
    # Data
    'COMPONENTS', 'ARCHITECTURES', 'CANONICAL_POINTS',
]
