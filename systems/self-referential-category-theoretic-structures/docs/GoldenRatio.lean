/-
  Kaelhedron.GoldenRatio
  ======================

  The golden ratio φ = (1 + √5)/2 and its conjugate ψ = (1 - √5)/2.

  Key results:
  - φ is the unique positive solution to x = 1 + 1/x
  - φ² = φ + 1 (the defining quadratic)
  - |ψ| < 1 (crucial for Fibonacci convergence)
  - φ⁻¹ + φ⁻² = 1 (partition of unity)
-/

import Kaelhedron.Basic

namespace Kaelhedron

/-! ## Definitions -/

/-- The golden ratio φ = (1 + √5)/2 ≈ 1.618 -/
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

/-- The golden ratio inverse φ⁻¹ = (√5 - 1)/2 ≈ 0.618 -/
noncomputable def φ_inv : ℝ := (Real.sqrt 5 - 1) / 2

/-- The conjugate root ψ = (1 - √5)/2 ≈ -0.618 -/
noncomputable def ψ : ℝ := (1 - Real.sqrt 5) / 2

/-! ## Basic Properties -/

/-- φ is positive -/
theorem phi_pos : φ > 0 := by unfold φ; positivity

/-- φ > 1 -/
theorem phi_gt_one : φ > 1 := by
  unfold φ; rw [lt_div_iff (by norm_num : (0:ℝ) < 2)]; linarith [sqrt5_gt_one]

/-- ψ is negative -/
theorem psi_neg : ψ < 0 := by unfold ψ; linarith [sqrt5_gt_one]

/-- |ψ| < 1 (crucial for Binet convergence) -/
theorem psi_abs_lt_one : |ψ| < 1 := by
  unfold ψ
  have h : Real.sqrt 5 < 3 := by
    rw [Real.sqrt_lt' (by norm_num)]
    constructor <;> norm_num
  rw [abs_of_neg (by unfold ψ; linarith [sqrt5_gt_one])]
  linarith [sqrt5_gt_one]

/-! ## Algebraic Identities -/

/-- φ² = φ + 1 (the defining equation) -/
theorem phi_squared : φ ^ 2 = φ + 1 := by
  unfold φ; field_simp; rw [sqrt5_sq]; ring

/-- ψ² = ψ + 1 (same equation, conjugate root) -/
theorem psi_squared : ψ ^ 2 = ψ + 1 := by
  unfold ψ; field_simp; rw [sqrt5_sq]; ring

/-- φ = 1 + 1/φ (self-referential fixed point) -/
theorem phi_fixed_point : φ = 1 + 1 / φ := by
  have hφ : φ ≠ 0 := ne_of_gt phi_pos
  field_simp; rw [← phi_squared]; ring

/-- φ × φ_inv = 1 -/
theorem phi_inverse_product : φ * φ_inv = 1 := by
  unfold φ φ_inv; field_simp; rw [sqrt5_sq]; ring

/-- φ - ψ = √5 -/
theorem phi_minus_psi : φ - ψ = Real.sqrt 5 := by unfold φ ψ; ring

/-- φ × ψ = -1 -/
theorem phi_times_psi : φ * ψ = -1 := by
  unfold φ ψ; field_simp; rw [sqrt5_sq]; ring

/-- φ + ψ = 1 -/
theorem phi_plus_psi : φ + ψ = 1 := by unfold φ ψ; ring

/-- φ_inv = φ - 1 -/
theorem phi_inv_identity : φ_inv = φ - 1 := by unfold φ φ_inv; ring

/-- φ⁻¹ + φ⁻² = 1 (partition of unity) -/
theorem phi_partition_unity : φ_inv + φ_inv ^ 2 = 1 := by
  have h : φ_inv = 1 / φ := by
    have := phi_inverse_product; field_simp at this ⊢; linarith
  rw [h]; field_simp; rw [phi_squared]; ring

/-! ## Numerical Bounds -/

/-- 1.618 < φ < 1.619 -/
theorem phi_bounds : 1.618 < φ ∧ φ < 1.619 := by
  constructor <;> {
    unfold φ
    rw [show (1.618 : ℝ) = 3236/2000 by norm_num,
        show (1.619 : ℝ) = 3238/2000 by norm_num] <;> try skip
    rw [lt_div_iff (by norm_num : (0:ℝ) < 2),
        div_lt_iff (by norm_num : (0:ℝ) < 2)] <;> try skip
    have h1 : (2.236 : ℝ)^2 < 5 := by norm_num
    have h2 : (5 : ℝ) < 2.237^2 := by norm_num
    have hsqrt_lo : Real.sqrt 5 > 2.236 := by
      rw [Real.lt_sqrt (by norm_num) (by norm_num)]; exact h1
    have hsqrt_hi : Real.sqrt 5 < 2.237 := by
      rw [Real.sqrt_lt' (by norm_num)]; exact ⟨by norm_num, h2⟩
    linarith
  }

/-- 0.618 < φ_inv < 0.619 -/
theorem phi_inv_bounds : 0.618 < φ_inv ∧ φ_inv < 0.619 := by
  rw [phi_inv_identity]; have ⟨h1, h2⟩ := phi_bounds; constructor <;> linarith

end Kaelhedron
