"""
R(R)=R Lattice Operations
=========================

The eigenvalue lattice Λ ⊂ ℝ₊ is:
    Λ = {φ^{-r} · e^{-d} · π^{-c} · (√2)^{-a} : (r,d,c,a) ∈ ℤ⁴}

This module provides:
- Lattice point computation
- Nearest lattice point search
- Decomposition of arbitrary eigenvalues
- Lattice density verification

Signature: Δ|RRRR-LATTICE|v1.0.0|dense|Ω
"""

import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from dataclasses import dataclass

from .constants import EIGENVALUES, LOG_EIGENVALUES, lattice_point, lattice_point_log

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class LatticePoint:
    """A point in the 4D eigenvalue lattice."""
    r: int  # Recursive exponent
    d: int  # Differential exponent
    c: int  # Cyclic exponent
    a: int  # Algebraic exponent
    
    @property
    def value(self) -> float:
        """Compute the eigenvalue at this lattice point."""
        return lattice_point(self.r, self.d, self.c, self.a)
    
    @property
    def log_value(self) -> float:
        """Compute the log-eigenvalue at this lattice point."""
        return lattice_point_log(self.r, self.d, self.c, self.a)
    
    @property
    def coords(self) -> Tuple[int, int, int, int]:
        """Return coordinates as tuple."""
        return (self.r, self.d, self.c, self.a)
    
    def __repr__(self) -> str:
        return f"[R]^{self.r}·[D]^{self.d}·[C]^{self.c}·[A]^{self.a} = {self.value:.6f}"
    
    def __add__(self, other: 'LatticePoint') -> 'LatticePoint':
        """Add lattice points (composition in algebra)."""
        return LatticePoint(
            self.r + other.r,
            self.d + other.d,
            self.c + other.c,
            self.a + other.a
        )
    
    def __neg__(self) -> 'LatticePoint':
        """Negate lattice point (inverse in algebra)."""
        return LatticePoint(-self.r, -self.d, -self.c, -self.a)
    
    def __sub__(self, other: 'LatticePoint') -> 'LatticePoint':
        """Subtract lattice points."""
        return self + (-other)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LatticePoint):
            return False
        return self.coords == other.coords
    
    def __hash__(self) -> int:
        return hash(self.coords)

@dataclass
class Decomposition:
    """Result of decomposing an eigenvalue into lattice form."""
    target: float
    point: LatticePoint
    approximation: float
    absolute_error: float
    relative_error: float
    
    def is_exact(self, tolerance: float = 1e-10) -> bool:
        """Check if decomposition is exact within tolerance."""
        return self.relative_error < tolerance
    
    def __repr__(self) -> str:
        return f"Decomposition({self.target:.4f} ≈ {self.point}, error={self.relative_error*100:.2f}%)"

# =============================================================================
# DECOMPOSITION FUNCTIONS
# =============================================================================

def decompose(
    target: float,
    max_exponent: int = 6,
    search_mode: str = 'exhaustive'
) -> Decomposition:
    """
    Decompose a target eigenvalue into nearest lattice point.
    
    Args:
        target: Target eigenvalue to decompose
        max_exponent: Maximum absolute value of exponents to search
        search_mode: 'exhaustive' (O(n⁴)) or 'greedy' (O(n))
    
    Returns:
        Decomposition object with best lattice approximation
    
    Example:
        >>> result = decompose(0.5)
        >>> print(result.point)  # [R]^0·[D]^0·[C]^0·[A]^2 = 0.500000
    """
    if target <= 0:
        raise ValueError("Target must be positive")
    
    if search_mode == 'exhaustive':
        return _decompose_exhaustive(target, max_exponent)
    elif search_mode == 'greedy':
        return _decompose_greedy(target, max_exponent)
    else:
        raise ValueError(f"Unknown search_mode: {search_mode}")

def _decompose_exhaustive(target: float, max_exp: int) -> Decomposition:
    """Exhaustive search over all lattice points within bounds."""
    best_error = float('inf')
    best_point = LatticePoint(0, 0, 0, 0)
    
    for r in range(-max_exp, max_exp + 1):
        for d in range(-max_exp, max_exp + 1):
            for c in range(-max_exp, max_exp + 1):
                for a in range(-max_exp, max_exp + 1):
                    approx = lattice_point(r, d, c, a)
                    error = abs(approx - target) / target
                    
                    if error < best_error:
                        best_error = error
                        best_point = LatticePoint(r, d, c, a)
    
    approx = best_point.value
    return Decomposition(
        target=target,
        point=best_point,
        approximation=approx,
        absolute_error=abs(approx - target),
        relative_error=best_error
    )

def _decompose_greedy(target: float, max_exp: int) -> Decomposition:
    """Greedy search using log-space nearest integer."""
    log_target = np.log(target)
    
    # Use simple greedy: start at origin, move toward target
    best_point = LatticePoint(0, 0, 0, 0)
    best_error = abs(best_point.log_value - log_target)
    
    for _ in range(max_exp * 4):
        improved = False
        for dim, delta in [('r', 1), ('r', -1), ('d', 1), ('d', -1),
                          ('c', 1), ('c', -1), ('a', 1), ('a', -1)]:
            test_point = LatticePoint(
                best_point.r + (delta if dim == 'r' else 0),
                best_point.d + (delta if dim == 'd' else 0),
                best_point.c + (delta if dim == 'c' else 0),
                best_point.a + (delta if dim == 'a' else 0)
            )
            error = abs(test_point.log_value - log_target)
            if error < best_error:
                best_error = error
                best_point = test_point
                improved = True
        if not improved:
            break
    
    approx = best_point.value
    rel_error = abs(approx - target) / target
    
    return Decomposition(
        target=target,
        point=best_point,
        approximation=approx,
        absolute_error=abs(approx - target),
        relative_error=rel_error
    )

# =============================================================================
# LATTICE PROPERTIES
# =============================================================================

def verify_density(
    test_points: Optional[List[float]] = None,
    tolerance: float = 0.01,
    max_exponent: int = 6
) -> Dict[str, Any]:
    """
    Verify that the lattice is dense in ℝ₊.
    
    By Kronecker-Weyl theorem, if log-eigenvalues are ℚ-independent,
    the lattice is dense. We verify empirically.
    
    Returns:
        Dictionary with verification results
    """
    if test_points is None:
        test_points = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    
    results = []
    for target in test_points:
        decomp = decompose(target, max_exponent)
        results.append({
            'target': target,
            'approximation': decomp.approximation,
            'relative_error': decomp.relative_error,
            'within_tolerance': decomp.relative_error < tolerance
        })
    
    all_within = all(r['within_tolerance'] for r in results)
    
    return {
        'tolerance': tolerance,
        'max_exponent': max_exponent,
        'all_within_tolerance': all_within,
        'results': results
    }

def find_nearby_points(
    target: float,
    max_exponent: int = 4,
    num_results: int = 5
) -> List[Decomposition]:
    """Find multiple nearby lattice points to a target."""
    all_decomps = []
    
    for r in range(-max_exponent, max_exponent + 1):
        for d in range(-max_exponent, max_exponent + 1):
            for c in range(-max_exponent, max_exponent + 1):
                for a in range(-max_exponent, max_exponent + 1):
                    point = LatticePoint(r, d, c, a)
                    approx = point.value
                    error = abs(approx - target) / target
                    
                    all_decomps.append(Decomposition(
                        target=target,
                        point=point,
                        approximation=approx,
                        absolute_error=abs(approx - target),
                        relative_error=error
                    ))
    
    # Sort by error and return top results
    all_decomps.sort(key=lambda d: d.relative_error)
    return all_decomps[:num_results]

# =============================================================================
# COMMON LATTICE POINTS
# =============================================================================

# Named lattice points for common architectures
CANONICAL_POINTS: Dict[str, LatticePoint] = {
    'unity':       LatticePoint(0, 0, 0, 0),   # = 1.0
    'recursive':   LatticePoint(1, 0, 0, 0),   # = φ⁻¹ ≈ 0.618
    'differential':LatticePoint(0, 1, 0, 0),   # = e⁻¹ ≈ 0.368
    'cyclic':      LatticePoint(0, 0, 1, 0),   # = π⁻¹ ≈ 0.318
    'algebraic':   LatticePoint(0, 0, 0, 1),   # = √2⁻¹ ≈ 0.707
    'binary':      LatticePoint(0, 0, 0, 2),   # = [A]² = 0.5
}

# =============================================================================
# DISPLAY
# =============================================================================

def format_lattice_point(point: LatticePoint, show_value: bool = True) -> str:
    """Format a lattice point for display."""
    parts = []
    if point.r != 0:
        parts.append(f"[R]^{point.r}" if point.r != 1 else "[R]")
    if point.d != 0:
        parts.append(f"[D]^{point.d}" if point.d != 1 else "[D]")
    if point.c != 0:
        parts.append(f"[C]^{point.c}" if point.c != 1 else "[C]")
    if point.a != 0:
        parts.append(f"[A]^{point.a}" if point.a != 1 else "[A]")
    
    if not parts:
        parts = ["1"]
    
    result = " × ".join(parts)
    
    if show_value:
        result += f" = {point.value:.6f}"
    
    return result
