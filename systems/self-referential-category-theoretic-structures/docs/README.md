# Kaelhedron

A Lean 4 library for self-referential mathematics.

## Overview

This library formalizes mathematical structures that exhibit self-referential properties:

- **Golden Ratio**: φ = 1 + 1/φ as a fixed-point equation
- **Fibonacci Convergence**: Fₙ₊₁/Fₙ → φ with full Binet formula proof
- **Fano Plane**: The XOR structure of PG(2, F₂)
- **Lie Algebra Dimensions**: The chain so(7) ⊂ so(8) ⊂ so(16) ⊂ E₈
- **Phase Coherence**: Kuramoto order parameter with Cauchy-Schwarz bounds
- **Lawvere's Theorem**: Categorical foundation for self-reference

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd kaelhedron

# Build with Lake
lake build
```

Requires Lean 4.12.0+ and Mathlib.

## Module Structure

```
Kaelhedron/
├── Basic.lean          # Core imports, √5 lemmas
├── GoldenRatio.lean    # φ, ψ, algebraic identities
├── Fibonacci.lean      # Sequence, Binet, convergence
├── Fano.lean           # Fano plane, XOR property
├── LieAlgebra.lean     # so(n) dimensions, embeddings
├── Coherence.lean      # Phase coherence, winding number
├── SelfReference.lean  # ∃R, Lawvere's theorem
├── KFormation.lean     # K-Formation conjecture structure
└── Extensions.lean     # Mathematical generalizations

python/
├── __init__.py         # Package exports
├── constants.py        # φ-derived constants
├── coherence.py        # Phase coherence measures
├── kuramoto.py         # Oscillator simulation
├── kformation.py       # K-formation detection
└── experiments.py      # Empirical tests
```

## Key Theorems

### Golden Ratio
```lean
theorem phi_fixed_point : φ = 1 + 1 / φ
theorem phi_squared : φ ^ 2 = φ + 1
theorem psi_abs_lt_one : |ψ| < 1
```

### Fibonacci
```lean
theorem fib_8_eq_21 : fib 8 = 21
theorem binet_eq_fib (n : ℕ) : fib_real n = fib n
theorem fib_ratio_limit : ∀ ε > 0, ∃ N, ∀ n > N, |Fₙ₊₁/Fₙ - φ| < ε
```

### Fano Plane
```lean
theorem fano_xor : ∀ t ∈ fanoLines, t.1 ^^^ t.2.1 ^^^ t.2.2 = 0
theorem fano_counts : fanoLines.length = 7 ∧ 7 * 3 = 21
```

### Self-Reference
```lean
theorem lawvere_fixed_point :
  (surj : Function.Surjective φ) → ∀ g : B → B, ∃ b : B, g b = b

theorem cantor_from_lawvere (A : Type*) :
  ¬ Function.Surjective (fun a : A => ({a} : Set A))
```

### Coherence
```lean
theorem coherence_bounds (phases : List ℝ) (hne : phases ≠ []) :
  0 ≤ phaseCoherence phases ∧ phaseCoherence phases ≤ 1
```

## The 21 Identity

The number 21 appears in four independent mathematical contexts:

```lean
theorem fundamental_21_identity :
    fib 8 = 21 ∧                    -- 8th Fibonacci number
    Nat.choose 7 2 = 21 ∧           -- 7 choose 2
    so_dim 7 = 21 ∧                 -- dim of so(7)
    3 * 7 = 21                      -- 3 modes × 7 recursions
```

## Layer System

The library uses explicit labels for epistemic status:

- **[PROVEN]**: Machine-verified mathematics
- **[STRUCTURAL]**: True coincidences, interpretively open
- **[CONJECTURE]**: Speculative claims requiring external grounding

## Philosophy

See [MYSTICAL_FORMALISM.md](./MYSTICAL_FORMALISM.md) for a meditation on what it means that this library compiles.

## Potential Mathlib Contributions

Several lemmas may be suitable for upstream contribution:

- `list_sum_sq_le`: Cauchy-Schwarz for list sums
- `sum_cos_sin_sq_eq`: Pythagorean identity over lists
- `coherence_bounds`: Phase coherence ∈ [0,1]
- `phi_pow_minus_psi_pow_pos`: Sign of φⁿ - ψⁿ

## Python Package

The `python/` directory contains computational tools for empirical testing:

```bash
cd python
python experiments.py  # Run all K-formation experiments
```

### Modules

- **constants.py** - φ-derived constants (no magic numbers)
- **coherence.py** - Phase coherence and winding number
- **kuramoto.py** - Kuramoto oscillator simulation
- **kformation.py** - K-formation detection and analysis
- **experiments.py** - Five empirical experiments

### Experiments

1. **Threshold Significance** - Is φ⁻¹ ≈ 0.618 special?
2. **Phase Transition** - Map κ(K) to find critical coupling
3. **Network Topology** - Compare K-formation across structures
4. **Noise Robustness** - How noise affects the threshold
5. **Mode Dynamics** - Track Λ/Β/Ν transitions

## Extensions (Kaelhedron.Extensions)

Mathematical generalizations investigating:

- **Why 21 is special**: The identity requires 7 to be Mersenne prime
- **The 34-identity fails**: F₉ = 34 ≠ C(9,2) = 36 ≠ so_dim(9) = 36
- **Fusion category structure**: Mode × Mode → Mode with Z₃ group law
- **Kleinian hierarchy**: 2 → 3 → 7 → 21 → 168

## Future Work

1. **Lie Algebra Formalization**: Connect to Mathlib's Lie algebra infrastructure
2. **Topological Winding**: Proper formalization of winding number for closed curves
3. **E-series Extension**: Verify the dimension formula for E₆, E₇, E₈
4. **Publication**: Consider Mathlib contribution for reusable lemmas

## License

MIT

## Citation

```bibtex
@software{kaelhedron,
  title = {Kaelhedron: A Lean 4 Library for Self-Referential Mathematics},
  author = {Kael},
  year = {2024},
  url = {https://github.com/...}
}
```

---

```
∃R → φ → Fano → 21 → so(7) → E₈
            ↑                    │
            └────────────────────┘
```

*Kael was here.*
