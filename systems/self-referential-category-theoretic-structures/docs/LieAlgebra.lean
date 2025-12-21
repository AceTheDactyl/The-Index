/-
  Kaelhedron.LieAlgebra
  =====================

  Dimension formulas for classical Lie algebras and embedding chains.

  Key results:
  - dim(so(n)) = n(n-1)/2
  - dim(so(7)) = 21 (the fundamental identity)
  - Embedding chain: so(7) ⊂ so(8) ⊂ so(16) ⊂ E₈
  - PSL(3,2) has order 168 = 8 × 21
-/

import Kaelhedron.Fibonacci

namespace Kaelhedron

/-! ## Dimension Formulas -/

/-- Dimension of so(n) = n(n-1)/2 -/
def so_dim (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Key dimensions in the embedding chain -/
theorem lie_dimensions :
    so_dim 7 = 21 ∧ so_dim 8 = 28 ∧ so_dim 16 = 120 := ⟨rfl, rfl, rfl⟩

/-- Dimension of E₈ -/
def e8_dim : ℕ := 248

/-- Dimension of G₂ -/
def g2_dim : ℕ := 14

/-! ## The Fundamental 21 Identity -/

/-- 21 appears in four different mathematical contexts -/
theorem fundamental_21_identity :
    fib 8 = 21 ∧                    -- 8th Fibonacci number
    Nat.choose 7 2 = 21 ∧           -- 7 choose 2
    so_dim 7 = 21 ∧                 -- dim of so(7) Lie algebra
    3 * 7 = 21 ∧                    -- 3 modes × 7 seals
    7 * 3 = 21 :=                   -- 7 Fano lines × 3 points each
  ⟨rfl, rfl, rfl, rfl, rfl⟩

/-- The coincidence chain -/
theorem twentyone_chain :
    fib 8 = Nat.choose 7 2 ∧
    Nat.choose 7 2 = so_dim 7 ∧
    so_dim 7 = 3 * 7 :=
  ⟨rfl, rfl, rfl⟩

/-! ## Embedding Chain -/

/-- The embedding chain dimensions are strictly increasing -/
theorem embedding_dimensions :
    so_dim 7 < so_dim 8 ∧
    so_dim 8 < so_dim 16 ∧
    so_dim 16 < e8_dim := by
  unfold so_dim e8_dim; omega

/-- Dimension values -/
theorem dimension_values :
    [so_dim 7, so_dim 8, so_dim 16, e8_dim] = [21, 28, 120, 248] := rfl

/-- E₈ = so(16) ⊕ spinor (248 = 120 + 128) -/
theorem e8_decomp : e8_dim = so_dim 16 + 128 := rfl

/-- so(7) = G₂ ⊕ R⁷ (21 = 14 + 7) -/
theorem so7_decomp : so_dim 7 = g2_dim + 7 := rfl

/-! ## Symmetry Groups -/

/-- Order of PSL(3,2) = GL(3, F₂) -/
theorem psl32_order : 168 = 8 * 21 ∧ 168 = 2^3 * 3 * 7 := ⟨rfl, rfl⟩

/-- 168 is the automorphism group order of the Fano plane -/
-- The Klein quartic also has 168 automorphisms

/-- 7 is a Mersenne prime -/
theorem seven_mersenne : 7 = 2^3 - 1 := rfl

/-- 7 is prime -/
theorem seven_prime : Nat.Prime 7 := by decide

/-- 3 is prime -/
theorem three_prime : Nat.Prime 3 := by decide

/-- First few Mersenne numbers -/
theorem mersenne_small :
    Nat.Prime (2^2 - 1) ∧     -- M₂ = 3 is prime
    Nat.Prime (2^3 - 1) ∧     -- M₃ = 7 is prime
    ¬Nat.Prime (2^4 - 1) ∧    -- M₄ = 15 is not prime
    Nat.Prime (2^5 - 1) := by -- M₅ = 31 is prime
  refine ⟨by decide, by decide, by decide, by decide⟩

end Kaelhedron
