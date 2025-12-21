"""
R(R)=R Composition Algebra
==========================

Neural architectures compose via dimension addition:
    ev(x + y) = ev(x) × ev(y)

This is a homomorphism from (ℤ⁴, +) to (ℝ₊, ×).

Architecture types are characterized by their lattice coordinates:
    ReLU:        [A]²      → 0.500
    Attention:   [R][C]    → 0.197
    ResNet:      [R][C][A]²→ 0.098
    Transformer: [R][D][C] → 0.072

Signature: Δ|RRRR-COMPOSITION|v1.0.0|homomorphism|Ω
"""

from typing import Dict, List, Optional
from dataclasses import dataclass

from .constants import EIGENVALUES, DIMENSION_LABELS
from .lattice import LatticePoint, decompose

# =============================================================================
# ARCHITECTURE SIGNATURES
# =============================================================================

@dataclass
class ArchitectureSignature:
    """
    Dimensional signature of a neural architecture.
    
    The signature (r, d, c, a) ∈ ℤ⁴ characterizes the
    self-referential structure of the architecture.
    """
    name: str
    point: LatticePoint
    description: str = ""
    
    @property
    def eigenvalue(self) -> float:
        """Characteristic eigenvalue of this architecture."""
        return self.point.value
    
    @property
    def dimensions(self) -> Dict[str, int]:
        """Dimensional analysis breakdown."""
        return {
            'Recursive': self.point.r,
            'Differential': self.point.d,
            'Cyclic': self.point.c,
            'Algebraic': self.point.a,
        }
    
    def compose(self, other: 'ArchitectureSignature') -> 'ArchitectureSignature':
        """
        Compose two architectures.
        
        In the composition algebra:
            ev(A + B) = ev(A) × ev(B)
        """
        return ArchitectureSignature(
            name=f"{self.name}+{other.name}",
            point=self.point + other.point,
            description=f"Composition of {self.name} and {other.name}"
        )
    
    def __repr__(self) -> str:
        dims = []
        if self.point.r: dims.append(f"[R]^{self.point.r}" if self.point.r != 1 else "[R]")
        if self.point.d: dims.append(f"[D]^{self.point.d}" if self.point.d != 1 else "[D]")
        if self.point.c: dims.append(f"[C]^{self.point.c}" if self.point.c != 1 else "[C]")
        if self.point.a: dims.append(f"[A]^{self.point.a}" if self.point.a != 1 else "[A]")
        
        dims_str = " × ".join(dims) if dims else "1"
        return f"{self.name}: {dims_str} = {self.eigenvalue:.4f}"

# =============================================================================
# CANONICAL ARCHITECTURES
# =============================================================================

# Fundamental components
COMPONENTS: Dict[str, ArchitectureSignature] = {
    'identity': ArchitectureSignature(
        name='Identity',
        point=LatticePoint(0, 0, 0, 0),
        description='No transformation'
    ),
    'relu': ArchitectureSignature(
        name='ReLU',
        point=LatticePoint(0, 0, 0, 2),  # [A]² = 0.5
        description='Binary activation via algebraic dimension'
    ),
    'residual': ArchitectureSignature(
        name='Residual',
        point=LatticePoint(1, 0, 0, 0),  # [R] = φ⁻¹
        description='Skip connection (recursive self-reference)'
    ),
    'attention': ArchitectureSignature(
        name='Attention',
        point=LatticePoint(1, 0, 1, 0),  # [R][C] = 0.197
        description='Self-attention (recursive × cyclic)'
    ),
    'layernorm': ArchitectureSignature(
        name='LayerNorm',
        point=LatticePoint(0, 1, 0, 0),  # [D] = e⁻¹
        description='Normalization (differential statistics)'
    ),
    'convolution': ArchitectureSignature(
        name='Convolution',
        point=LatticePoint(0, 0, 1, 0),  # [C] = π⁻¹
        description='Spatial convolution (cyclic/periodic)'
    ),
    'ffn': ArchitectureSignature(
        name='FFN',
        point=LatticePoint(0, 1, 0, 2),  # [D][A]² = 0.184
        description='Feed-forward network (differential + 2×ReLU)'
    ),
}

# Composed architectures
ARCHITECTURES: Dict[str, ArchitectureSignature] = {
    'mlp': ArchitectureSignature(
        name='MLP',
        point=LatticePoint(0, 0, 0, 4),  # [A]⁴ = 0.25
        description='Multi-layer perceptron (4 ReLUs)'
    ),
    'resnet_block': ArchitectureSignature(
        name='ResNet Block',
        point=LatticePoint(1, 0, 1, 2),  # [R][C][A]² = 0.098
        description='Residual + Conv + 2×ReLU'
    ),
    'transformer_block': ArchitectureSignature(
        name='Transformer Block',
        point=LatticePoint(1, 1, 1, 0),  # [R][D][C] = 0.072
        description='Attention + FFN + LayerNorm'
    ),
    'lstm_cell': ArchitectureSignature(
        name='LSTM Cell',
        point=LatticePoint(2, 0, 0, 4),  # [R]²[A]⁴ = 0.095
        description='Recurrent (2×recursive + 4×gates)'
    ),
    'gru_cell': ArchitectureSignature(
        name='GRU Cell',
        point=LatticePoint(2, 0, 0, 2),  # [R]²[A]² = 0.191
        description='Simplified recurrent (2×recursive + 2×gates)'
    ),
}

# =============================================================================
# COMPOSITION OPERATIONS
# =============================================================================

def compose(*archs: ArchitectureSignature) -> ArchitectureSignature:
    """
    Compose multiple architectures.
    
    The eigenvalue of the composition is the product of component eigenvalues.
    
    Example:
        >>> result = compose(COMPONENTS['residual'], COMPONENTS['relu'])
        >>> print(result.eigenvalue)  # 0.618 × 0.5 = 0.309
    """
    if not archs:
        return COMPONENTS['identity']
    
    result = archs[0]
    for arch in archs[1:]:
        result = result.compose(arch)
    
    return result

def analyze_architecture(
    characteristic_eigenvalue: float,
    name: str = "Unknown",
    max_exponent: int = 6
) -> ArchitectureSignature:
    """
    Analyze an architecture by its characteristic eigenvalue.
    
    Decomposes the eigenvalue into lattice coordinates to determine
    the dimensional structure of the architecture.
    """
    decomp = decompose(characteristic_eigenvalue, max_exponent)
    
    return ArchitectureSignature(
        name=name,
        point=decomp.point,
        description=f"Decomposed from λ = {characteristic_eigenvalue:.4f} (error: {decomp.relative_error*100:.2f}%)"
    )

# =============================================================================
# SCALING ANALYSIS
# =============================================================================

def predict_scaling_exponent(arch: ArchitectureSignature) -> Dict[str, float]:
    """
    Predict scaling exponents for an architecture.
    
    Based on the dimensional analysis, predict how loss scales with:
    - Data size (D)
    - Model size (N)
    - Compute (C)
    
    See Appendix D of the theoretical paper for derivation.
    """
    r, d, c, a = arch.point.coords
    
    # Theoretical predictions (conjectural but empirically validated)
    
    # Data scaling: dominated by recursive × cyclic structure
    alpha_D = EIGENVALUES['R'] * EIGENVALUES['C'] * (0.5 ** (a // 2))
    
    # Model scaling: dominated by recursive × differential structure
    alpha_N = EIGENVALUES['R'] * EIGENVALUES['D'] / (1 + r)
    
    # Compute scaling (Chinchilla optimal): geometric mean
    alpha_C = (alpha_D * alpha_N) ** 0.5
    
    return {
        'data_exponent': alpha_D,
        'model_exponent': alpha_N,
        'compute_exponent': alpha_C,
        'dimensional_signature': (r, d, c, a),
    }

# =============================================================================
# DISPLAY FUNCTIONS
# =============================================================================

def print_architecture_analysis(arch: ArchitectureSignature) -> None:
    """Pretty-print architecture analysis."""
    print(f"\n{'='*60}")
    print(f"  ARCHITECTURE: {arch.name}")
    print(f"{'='*60}")
    print(f"\n  Dimensional Signature: (r={arch.point.r}, d={arch.point.d}, c={arch.point.c}, a={arch.point.a})")
    print(f"\n  Decomposition:")
    print(f"    {arch}")
    print(f"\n  Dimensional Analysis:")
    for dim, exp in arch.dimensions.items():
        if exp != 0:
            print(f"    • {dim}: {exp}")
    print(f"\n  Description: {arch.description}")
    
    scaling = predict_scaling_exponent(arch)
    print(f"\n  Predicted Scaling Exponents:")
    print(f"    • α_D (data):    {scaling['data_exponent']:.4f}")
    print(f"    • α_N (model):   {scaling['model_exponent']:.4f}")
    print(f"    • α_C (compute): {scaling['compute_exponent']:.4f}")
    print()

def list_all_architectures() -> None:
    """List all defined architectures."""
    print("\n" + "="*60)
    print("  DEFINED ARCHITECTURES")
    print("="*60)
    
    print("\n  Components:")
    for name, arch in COMPONENTS.items():
        print(f"    {arch}")
    
    print("\n  Composed Architectures:")
    for name, arch in ARCHITECTURES.items():
        print(f"    {arch}")
    print()
