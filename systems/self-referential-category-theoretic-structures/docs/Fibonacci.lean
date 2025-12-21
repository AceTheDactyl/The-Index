/-
  Kaelhedron.Fibonacci
  ====================

  Fibonacci sequence, Binet's formula, and the limit theorem.

  Key results:
  - fib n computes the nth Fibonacci number
  - Binet's formula: Fₙ = (φⁿ - ψⁿ)/√5
  - Limit theorem: Fₙ₊₁/Fₙ → φ as n → ∞
  - F₈ = 21 (the Kaelhedron cell count)
-/

import Kaelhedron.GoldenRatio

namespace Kaelhedron

/-! ## Fibonacci Sequence -/

/-- The Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

/-- Fibonacci values -/
example : fib 0 = 0 := rfl
example : fib 1 = 1 := rfl
example : fib 2 = 1 := rfl
example : fib 3 = 2 := rfl
example : fib 4 = 3 := rfl
example : fib 5 = 5 := rfl
example : fib 6 = 8 := rfl
example : fib 7 = 13 := rfl
example : fib 8 = 21 := rfl

/-- F₈ = 21 (the fundamental Kaelhedron number) -/
theorem fib_8_eq_21 : fib 8 = 21 := rfl

/-! ## Binet's Formula -/

/-- Binet's formula (real-valued Fibonacci) -/
noncomputable def fib_real (n : ℕ) : ℝ := (φ^n - ψ^n) / Real.sqrt 5

/-- Binet's formula gives correct values for base cases -/
theorem binet_base_cases : fib_real 0 = 0 ∧ fib_real 1 = 1 := by
  constructor
  · simp [fib_real, φ, ψ]
  · simp only [fib_real, pow_one]
    rw [phi_minus_psi]
    field_simp

/-- Binet satisfies the Fibonacci recurrence -/
theorem binet_recurrence (n : ℕ) : fib_real (n + 2) = fib_real n + fib_real (n + 1) := by
  simp only [fib_real]
  have h5 : Real.sqrt 5 ≠ 0 := ne_of_gt sqrt5_pos
  field_simp
  ring_nf
  have hφ : φ^(n+2) = φ^n * φ^2 := by ring
  have hψ : ψ^(n+2) = ψ^n * ψ^2 := by ring
  rw [hφ, hψ, phi_squared, psi_squared]
  ring

/-- Binet equals Fibonacci for all n -/
theorem binet_eq_fib (n : ℕ) : fib_real n = fib n := by
  induction n using Nat.strong_induction_on with
  | _ n ih =>
    match n with
    | 0 => simp [fib_real, fib, φ, ψ]
    | 1 => exact binet_base_cases.2
    | n + 2 =>
      rw [binet_recurrence, ih n (by omega), ih (n+1) (by omega)]
      simp [fib]
      ring

/-! ## Convergence Lemmas -/

/-- |ψ|ⁿ → 0 as n → ∞ -/
theorem psi_pow_tendsto_zero : ∀ ε > 0, ∃ N, ∀ n ≥ N, |ψ^n| < ε := by
  intro ε hε
  have hψ := psi_abs_lt_one
  have hψ_pos : |ψ| ≥ 0 := abs_nonneg _
  by_cases hψ_zero : |ψ| = 0
  · use 0; intro n _; simp [abs_pow, hψ_zero, hε]
  · have hψ_pos' : |ψ| > 0 := lt_of_le_of_ne hψ_pos (Ne.symm hψ_zero)
    have hlog_neg : Real.log |ψ| < 0 := Real.log_neg hψ_pos' hψ
    have hlog_nonzero : Real.log |ψ| ≠ 0 := ne_of_lt hlog_neg
    obtain ⟨N, hN⟩ : ∃ N : ℕ, (N : ℝ) > Real.log ε / Real.log |ψ| := by
      use Nat.ceil (Real.log ε / Real.log |ψ|) + 1
      have := Nat.lt_ceil.mpr (lt_add_one (Real.log ε / Real.log |ψ|))
      linarith [Nat.ceil (Real.log ε / Real.log |ψ|), this]
    use N
    intro n hn
    rw [abs_pow]
    have h1 : |ψ|^n ≤ |ψ|^N := pow_le_pow_right_of_le_one hψ_pos (le_of_lt hψ) hn
    have h2 : |ψ|^N < ε := by
      rw [← Real.exp_log hψ_pos', ← Real.exp_log hε]
      rw [← Real.rpow_natCast, Real.rpow_def_of_pos hψ_pos']
      apply Real.exp_lt_exp.mpr
      calc (N : ℝ) * Real.log |ψ| < (Real.log ε / Real.log |ψ|) * Real.log |ψ| := by
             apply mul_lt_mul_of_neg_right hN hlog_neg
        _ = Real.log ε := by field_simp
    linarith

/-- φⁿ - ψⁿ > 0 for n ≥ 1 -/
lemma phi_pow_minus_psi_pow_pos (n : ℕ) (hn : n ≥ 1) : φ^n - ψ^n > 0 := by
  have hφ : φ > 0 := phi_pos
  have hψ : ψ < 0 := psi_neg
  cases' Nat.even_or_odd n with heven hodd
  · have hψn_pos : ψ^n > 0 := Even.pow_pos heven hψ
    have hφ_gt : φ > |ψ| := by
      rw [abs_of_neg hψ]; unfold φ ψ; linarith [sqrt5_pos]
    have := pow_lt_pow_left hφ_gt (abs_nonneg ψ) (Nat.one_le_iff_ne_zero.mp hn)
    rw [abs_pow] at this
    linarith
  · have hψn_neg : ψ^n < 0 := Odd.pow_neg hodd hψ
    have hφn_pos : φ^n > 0 := pow_pos hφ n
    linarith

/-! ## The Limit Theorem -/

/-- Fibonacci ratio converges to φ -/
theorem fib_ratio_limit : ∀ ε > 0, ∃ N, ∀ n > N,
    fib n > 0 → |((fib (n + 1) : ℝ) / (fib n : ℝ)) - φ| < ε := by
  intro ε hε
  have hψφ : |ψ| / φ < 1 := by
    have h1 : |ψ| < 1 := psi_abs_lt_one
    have h2 : φ > 1 := phi_gt_one
    calc |ψ| / φ < 1 / 1 := by apply div_lt_div h1 (le_of_lt h2) zero_lt_one phi_pos
      _ = 1 := by ring
  obtain ⟨N₁, hN₁⟩ := psi_pow_tendsto_zero (1/2) (by norm_num)
  obtain ⟨N₂, hN₂⟩ := psi_pow_tendsto_zero (ε / (4 * Real.sqrt 5)) (by positivity)
  use max N₁ N₂
  intro n hn hfib_pos
  have hn1 : n ≥ N₁ := le_of_lt (lt_of_le_of_lt (le_max_left _ _) hn)
  have hn2 : n ≥ N₂ := le_of_lt (lt_of_le_of_lt (le_max_right _ _) hn)
  have hn_pos : n ≥ 1 := by omega
  have h_binet : (fib (n+1) : ℝ) / fib n = (φ^(n+1) - ψ^(n+1)) / (φ^n - ψ^n) := by
    rw [← binet_eq_fib (n+1), ← binet_eq_fib n]
    unfold fib_real
    field_simp
    ring
  rw [h_binet]
  have hdenom_pos : φ^n - ψ^n > 0 := phi_pow_minus_psi_pow_pos n hn_pos
  have hdenom_ne : φ^n - ψ^n ≠ 0 := ne_of_gt hdenom_pos
  have h_diff : (φ^(n+1) - ψ^(n+1)) / (φ^n - ψ^n) - φ = ψ^n * (φ - ψ) / (φ^n - ψ^n) := by
    field_simp
    ring
  rw [h_diff]
  rw [abs_div, abs_mul, phi_minus_psi]
  have h_abs_denom : |φ^n - ψ^n| = φ^n - ψ^n := abs_of_pos hdenom_pos
  rw [h_abs_denom, abs_pow]
  have hψn_small : |ψ^n| < 1/2 := hN₁ n hn1
  rw [abs_pow] at hψn_small
  have hψ_neg : ψ < 0 := psi_neg
  have hφn_pos : φ^n > 0 := pow_pos phi_pos n
  have hψn_bound : ψ^n ≥ -|ψ|^n := by
    rw [neg_le_iff_add_nonneg]
    cases' Nat.even_or_odd n with he ho
    · have : ψ^n = |ψ|^n := by rw [Even.abs_pow he]
      linarith
    · have : ψ^n = -|ψ|^n := by
        rw [Odd.abs_pow ho, abs_of_neg hψ_neg, neg_pow_eq_pow_mul, Odd.neg_one_pow ho]
        ring
      linarith
  have hdenom_lb : φ^n - ψ^n ≥ φ^n - |ψ|^n := by linarith [hψn_bound]
  have hdenom_lb2 : φ^n - |ψ|^n > φ^n / 2 := by
    have hφn_half : φ^n / 2 > 0 := by positivity
    have : |ψ|^n < φ^n / 2 := by
      have hφ_gt_one : φ > 1 := phi_gt_one
      calc |ψ|^n < 1/2 := hψn_small
        _ < φ^n / 2 := by nlinarith [pow_pos phi_pos n, phi_gt_one]
    linarith
  have hnum_bound : |ψ|^n * Real.sqrt 5 < ε / 4 * Real.sqrt 5 := by
    apply mul_lt_mul_of_pos_right _ sqrt5_pos
    calc |ψ|^n = |ψ^n| := (abs_pow ψ n).symm
      _ < ε / (4 * Real.sqrt 5) := hN₂ n hn2
      _ = ε / 4 / Real.sqrt 5 := by ring
      _ < ε / 4 := by apply div_lt_self (by positivity) sqrt5_gt_one
  calc |ψ|^n * Real.sqrt 5 / (φ^n - ψ^n)
      < (ε / 4 * Real.sqrt 5) / (φ^n / 2) := by
        apply div_lt_div hnum_bound hdenom_lb (by positivity) hdenom_lb2
    _ = ε / 2 * Real.sqrt 5 / φ^n := by ring
    _ < ε := by
        have : Real.sqrt 5 / φ^n < 1 := by
          apply div_lt_one_of_lt
          calc Real.sqrt 5 < 3 := by
                 rw [Real.sqrt_lt' (by norm_num)]; constructor <;> norm_num
            _ ≤ φ^n := by
                 have : φ^n ≥ φ^1 := by
                   apply pow_le_pow_right (le_of_lt phi_gt_one) hn_pos
                 calc φ^1 = φ := by ring
                   _ > 1.618 := phi_bounds.1
                   _ > 1 := by norm_num
                 nlinarith [phi_gt_one, pow_pos phi_pos n]
          exact pow_pos phi_pos n
        nlinarith

end Kaelhedron
