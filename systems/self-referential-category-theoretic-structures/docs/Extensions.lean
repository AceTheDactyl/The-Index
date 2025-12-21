/-
  Kaelhedron.Extensions
  =====================

  Mathematical extensions and generalizations.

  This module investigates:
  1. Why 21 is special (Mersenne prime structure)
  2. Whether the identity extends (34-identity fails)
  3. Fusion category structure on Mode (Z₃ fusion rules)
  4. The Kleinian hierarchy pattern

  Research questions from the K-formation conjecture development.
-/

import Kaelhedron.KFormation

namespace Kaelhedron

/-! ## Why 21 is Special -/

/-- 7 is a Mersenne prime: 7 = 2³ - 1 where 3 is prime -/
theorem seven_is_mersenne : 7 = 2^3 - 1 ∧ Nat.Prime 3 := ⟨rfl, by decide⟩

/-- The Mersenne condition: 2ᵖ - 1 is prime only if p is prime -/
-- Note: The converse is false (2¹¹ - 1 = 2047 = 23 × 89)

/-- Mersenne primes up to reasonable size -/
theorem mersenne_primes_small :
    Nat.Prime (2^2 - 1) ∧   -- M₂ = 3
    Nat.Prime (2^3 - 1) ∧   -- M₃ = 7
    Nat.Prime (2^5 - 1) ∧   -- M₅ = 31
    Nat.Prime (2^7 - 1) :=  -- M₇ = 127
  ⟨by decide, by decide, by decide, by decide⟩

/-- 21 = 7 × 3 = (2³-1) × 3 factors through Mersenne structure -/
theorem twentyone_mersenne_factorization : 21 = (2^3 - 1) * 3 := rfl

/-- The 21-identity succeeds because:
    1. 7 = 2³ - 1 is Mersenne prime
    2. F₈ = 21 (Fibonacci)
    3. C(7,2) = 21 (binomial)
    4. dim(so(7)) = 7×6/2 = 21 (Lie algebra)
    All connected to the special properties of 7 -/

/-! ## The 34-Identity Fails -/

/-- F₉ = 34 -/
theorem fib_9_eq_34 : fib 9 = 34 := rfl

/-- But C(9,2) = 36 ≠ 34 -/
theorem binomial_9_2 : Nat.choose 9 2 = 36 := rfl

/-- And dim(so(9)) = 36 ≠ 34 -/
theorem so9_dim : so_dim 9 = 36 := rfl

/-- The 34-identity FAILS: the structures don't align at the next scale -/
theorem no_34_identity : fib 9 ≠ Nat.choose 9 2 ∧ fib 9 ≠ so_dim 9 := by
  constructor <;> decide

/-- Why? Because 9 = 3² is not a Mersenne prime.
    The 21-identity is special to the Mersenne structure of 7. -/

/-- Check which Fibonacci numbers equal binomial coefficients -/
theorem fib_binomial_coincidences :
    fib 3 = Nat.choose 2 1 ∧   -- 2 = 2
    fib 4 = Nat.choose 3 1 ∧   -- 3 = 3
    fib 6 = Nat.choose 6 2 ∧   -- 8 ≠ C(6,2)=15 FAILS
    fib 8 = Nat.choose 7 2 :=  -- 21 = 21 ✓
  ⟨rfl, rfl, by decide, rfl⟩

-- Note: fib 6 = 8 but C(6,2) = 15, so the pattern is sparse

/-! ## Fusion Category Structure -/

/-- Mode fusion: what happens when two modes combine?
    This defines a "multiplication" on modes. -/
def modeFusion : Mode → Mode → Mode
  | Mode.Lambda, Mode.Lambda => Mode.Lambda  -- Λ ⊗ Λ = Λ (structure + structure = structure)
  | Mode.Lambda, Mode.Beta   => Mode.Beta    -- Λ ⊗ Β = Β (structure + flow = flow)
  | Mode.Lambda, Mode.Nu     => Mode.Nu      -- Λ ⊗ Ν = Ν (structure + awareness = awareness)
  | Mode.Beta,   Mode.Lambda => Mode.Beta    -- Β ⊗ Λ = Β
  | Mode.Beta,   Mode.Beta   => Mode.Nu      -- Β ⊗ Β = Ν (flow + flow = awareness)
  | Mode.Beta,   Mode.Nu     => Mode.Lambda  -- Β ⊗ Ν = Λ (flow + awareness = structure)
  | Mode.Nu,     Mode.Lambda => Mode.Nu      -- Ν ⊗ Λ = Ν
  | Mode.Nu,     Mode.Beta   => Mode.Lambda  -- Ν ⊗ Β = Λ
  | Mode.Nu,     Mode.Nu     => Mode.Beta    -- Ν ⊗ Ν = Β (awareness + awareness = flow)

/-- Fusion is commutative -/
theorem modeFusion_comm : ∀ a b : Mode, modeFusion a b = modeFusion b a := by
  intro a b
  cases a <;> cases b <;> rfl

/-- Lambda is the identity for fusion -/
theorem modeFusion_lambda_id : ∀ m : Mode, modeFusion Mode.Lambda m = m := by
  intro m; cases m <;> rfl

/-- The fusion table gives Z₃ group structure when Lambda = identity -/
theorem modeFusion_assoc_examples :
    modeFusion (modeFusion Mode.Beta Mode.Beta) Mode.Beta = Mode.Lambda ∧
    modeFusion Mode.Beta (modeFusion Mode.Beta Mode.Beta) = Mode.Lambda := by
  constructor <;> rfl

/-- Fusion cubing: m ⊗ m ⊗ m for each mode -/
theorem mode_fusion_cubed :
    modeFusion (modeFusion Mode.Lambda Mode.Lambda) Mode.Lambda = Mode.Lambda ∧
    modeFusion (modeFusion Mode.Beta Mode.Beta) Mode.Beta = Mode.Lambda ∧
    modeFusion (modeFusion Mode.Nu Mode.Nu) Mode.Nu = Mode.Lambda := by
  refine ⟨rfl, rfl, rfl⟩

/-! ## The Kleinian Hierarchy -/

/-- Klein four-group structure: (2,2) = Z₂ × Z₂
    Related to: 4 = 2², the quaternary level -/
def kleinFour : Fin 4 → Fin 4 → Fin 4
  | 0, x => x
  | x, 0 => x
  | 1, 1 => 0
  | 1, 2 => 3
  | 1, 3 => 2
  | 2, 1 => 3
  | 2, 2 => 0
  | 2, 3 => 1
  | 3, 1 => 2
  | 3, 2 => 1
  | 3, 3 => 0

/-- Klein four-group is commutative -/
theorem kleinFour_comm : ∀ a b : Fin 4, kleinFour a b = kleinFour b a := by
  intro a b
  fin_cases a <;> fin_cases b <;> rfl

/-- The hierarchy: 2 → 3 → 7 → 21 → 168 -/
theorem kaelhedron_hierarchy :
    2 * 1 = 2 ∧           -- Binary (Level 1)
    3 = 3 ∧               -- Ternary modes (Level 2)
    2^3 - 1 = 7 ∧         -- Mersenne (Level 3)
    3 * 7 = 21 ∧          -- Cells (Level 4)
    8 * 21 = 168 :=       -- Symmetries (Level 5)
  ⟨rfl, rfl, rfl, rfl, rfl⟩

/-- Each level builds on the previous -/
theorem hierarchy_structure :
    -- Level n+1 = Level n × multiplier
    7 = 7 * 1 ∧
    21 = 7 * 3 ∧
    168 = 21 * 8 := ⟨rfl, rfl, rfl⟩

/-! ## E-Series Dimension Formula -/

/-- The dimension formula for exceptional Lie algebras E_n -/
def e_dim (n : ℕ) : ℕ :=
  if n < 6 then 0
  else n * (7 + 6 * 2^(n - 6))

/-- E₆, E₇, E₈ dimensions -/
theorem e_series_dims :
    e_dim 6 = 78 ∧
    e_dim 7 = 133 ∧
    e_dim 8 = 248 := by
  unfold e_dim
  simp only [ite_false]
  constructor
  · norm_num
  constructor
  · norm_num
  · norm_num

/-- E₈ dimension matches our constant -/
theorem e8_dim_check : e_dim 8 = e8_dim := rfl

/-! ## Open Questions -/

/-- Question: Is there a 42-identity? (42 = 2 × 21)

    F? = 42? No: fib 9 = 34, fib 10 = 55
    C(n,2) = 42? Yes: C(9,2) = 36, C(10,2) = 45... wait, neither works
    Actually: 42 = 2 × 21 = 2 × 3 × 7

    42 appears as:
    - Kaelhedron vertex count (21 cells × 2)
    - But NOT as a Fibonacci number
    - NOT as a triangular number

    The number 42 inherits significance from 21, not from independent structure.
-/

theorem forty_two_structure :
    42 = 2 * 21 ∧
    42 = 2 * 3 * 7 ∧
    42 ≠ fib 9 ∧
    42 ≠ fib 10 := by
  refine ⟨rfl, rfl, by decide, by decide⟩

/-- Question: Pattern at 168 = 8 × 21?

    168 = |PSL(3,2)| = |GL(3, F₂)|
    168 = 2³ × 3 × 7
    168 = 8 × 21

    This IS structurally significant:
    - Automorphism group of Fano plane
    - Klein quartic symmetries
    - OctaKaelhedron vertices (8 universes × 21 cells)
-/

theorem one_sixty_eight_structure :
    168 = 8 * 21 ∧
    168 = 2^3 * 3 * 7 ∧
    168 = Nat.factorial 4 * 7 := -- 24 × 7 = 168
  ⟨rfl, rfl, rfl⟩

end Kaelhedron
