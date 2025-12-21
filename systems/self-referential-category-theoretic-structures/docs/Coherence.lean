/-
  Kaelhedron.Coherence
  ====================

  Phase coherence (Kuramoto order parameter) and winding number.

  These are standard measures in dynamical systems theory:
  - Phase coherence κ ∈ [0,1] measures synchronization of oscillators
  - Winding number Q ∈ ℤ measures topological charge of phase curves

  Key results:
  - κ = |1/N Σᵢ exp(iθᵢ)| is well-defined and bounded
  - Cauchy-Schwarz gives κ ≤ 1
  - Pythagorean identity: cos²θ + sin²θ = 1
-/

import Kaelhedron.GoldenRatio

namespace Kaelhedron

/-! ## Auxiliary Lemmas -/

/-- Sum of cos² + sin² over a list equals the list length -/
lemma sum_cos_sin_sq_eq (phases : List ℝ) :
    (phases.map (fun θ => Real.cos θ ^ 2 + Real.sin θ ^ 2)).sum = phases.length := by
  induction phases with
  | nil => simp
  | cons h t ih =>
    simp [List.map_cons, List.sum_cons]
    rw [Real.cos_sq_add_sin_sq, ih]
    ring

/-- Cauchy-Schwarz for list sums: (Σ aᵢ)² ≤ n · Σ aᵢ² -/
lemma list_sum_sq_le (l : List ℝ) :
    (l.sum)^2 ≤ l.length * (l.map (·^2)).sum := by
  induction l with
  | nil => simp
  | cons h t ih =>
    simp only [List.sum_cons, List.length_cons, List.map_cons, Nat.cast_add, Nat.cast_one]
    have ht_len : (0 : ℝ) ≤ t.length := Nat.cast_nonneg _
    have h_sum_sq : (l.map (·^2)).sum ≥ 0 := by
      apply List.sum_nonneg; intro x hx
      simp only [List.mem_map] at hx
      obtain ⟨a, _, rfl⟩ := hx; positivity
    nlinarith [sq_nonneg h, sq_nonneg t.sum, sq_nonneg (h - t.sum), ih]

/-! ## Phase Coherence -/

/-- Phase coherence: order parameter for N oscillators with phases θᵢ
    κ = |1/N Σᵢ exp(i·θᵢ)| = √(sum_cos² + sum_sin²) / N
    κ ∈ [0,1], with κ=1 for perfect sync, κ≈0 for incoherence -/
noncomputable def phaseCoherence (phases : List ℝ) : ℝ :=
  if phases.isEmpty then 0 else
  let n := phases.length
  let sum_cos := (phases.map Real.cos).sum
  let sum_sin := (phases.map Real.sin).sum
  Real.sqrt (sum_cos^2 + sum_sin^2) / n

/-- Coherence bounds: 0 ≤ κ ≤ 1 -/
theorem coherence_bounds (phases : List ℝ) (hne : phases ≠ []) :
    0 ≤ phaseCoherence phases ∧ phaseCoherence phases ≤ 1 := by
  constructor
  · unfold phaseCoherence; split_ifs with h; · norm_num; · positivity
  · unfold phaseCoherence
    simp only [List.isEmpty_iff] at hne
    simp only [hne, ↓reduceIte]
    have hn_pos : (phases.length : ℝ) > 0 := by
      simp only [Nat.cast_pos]; exact List.length_pos.mpr hne
    rw [div_le_one hn_pos]
    let cos_list := phases.map Real.cos
    let sin_list := phases.map Real.sin
    have h_cos_cs := list_sum_sq_le cos_list
    have h_sin_cs := list_sum_sq_le sin_list
    have h_cos_len : cos_list.length = phases.length := List.length_map phases Real.cos
    have h_sin_len : sin_list.length = phases.length := List.length_map phases Real.sin
    have h_sum_identity : (cos_list.map (·^2)).sum + (sin_list.map (·^2)).sum = phases.length := by
      simp only [cos_list, sin_list]
      have h1 : (phases.map Real.cos).map (·^2) = phases.map (fun θ => (Real.cos θ)^2) := by
        simp [List.map_map]
      have h2 : (phases.map Real.sin).map (·^2) = phases.map (fun θ => (Real.sin θ)^2) := by
        simp [List.map_map]
      rw [h1, h2]
      rw [← List.sum_add_sum_eq_sum_add phases (fun θ => (Real.cos θ)^2) (fun θ => (Real.sin θ)^2)]
      convert sum_cos_sin_sq_eq phases
      ext θ; ring
    have h_combined : cos_list.sum^2 + sin_list.sum^2 ≤ phases.length ^ 2 := by
      calc cos_list.sum^2 + sin_list.sum^2
          ≤ cos_list.length * (cos_list.map (·^2)).sum +
            sin_list.length * (sin_list.map (·^2)).sum := by linarith [h_cos_cs, h_sin_cs]
        _ = phases.length * ((cos_list.map (·^2)).sum + (sin_list.map (·^2)).sum) := by
            rw [h_cos_len, h_sin_len]; ring
        _ = phases.length * phases.length := by rw [h_sum_identity]
        _ = phases.length ^ 2 := by ring
    calc Real.sqrt (cos_list.sum^2 + sin_list.sum^2)
        ≤ Real.sqrt (phases.length^2) := by
          apply Real.sqrt_le_sqrt; exact h_combined
      _ = |phases.length| := Real.sqrt_sq_eq_abs _
      _ = phases.length := abs_of_nonneg (Nat.cast_nonneg _)

/-- Coherence of synchronized phases is 1 -/
theorem coherence_synchronized (θ : ℝ) (n : ℕ) (hn : n > 0) :
    phaseCoherence (List.replicate n θ) = 1 := by
  unfold phaseCoherence
  simp only [List.isEmpty_replicate, hn, ↓reduceIte, List.length_replicate]
  simp only [List.map_replicate, List.sum_replicate, nsmul_eq_mul]
  have hn' : (n : ℝ) ≠ 0 := Nat.cast_ne_zero.mpr (Nat.not_eq_zero_of_lt hn)
  rw [show (n : ℝ) * Real.cos θ = n * Real.cos θ by ring]
  rw [show (n : ℝ) * Real.sin θ = n * Real.sin θ by ring]
  have h : (n * Real.cos θ)^2 + (n * Real.sin θ)^2 = n^2 := by
    ring_nf; rw [Real.cos_sq_add_sin_sq]; ring
  rw [h]
  rw [Real.sqrt_sq (Nat.cast_nonneg n)]
  field_simp

/-! ## Winding Number -/

/-- Winding number (topological charge) for a sequence of phases
    Q = (1/2π) · (total phase accumulation)
    For a closed curve, this is an integer -/
noncomputable def windingNumber (phases : List ℝ) : ℝ :=
  if h : phases.length < 2 then 0 else
  let deltas := List.zipWith (· - ·) phases.tail phases.dropLast
  deltas.sum / (2 * Real.pi)

/-- Winding number of empty list is 0 -/
theorem windingNumber_empty : windingNumber [] = 0 := by
  simp [windingNumber]

/-- Winding number of singleton is 0 -/
theorem windingNumber_singleton (θ : ℝ) : windingNumber [θ] = 0 := by
  simp [windingNumber]

end Kaelhedron
