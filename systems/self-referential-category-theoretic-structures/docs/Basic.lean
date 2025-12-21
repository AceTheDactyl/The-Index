/-
  Kaelhedron.Basic
  ================

  Core imports and basic utilities for the Kaelhedron library.

  This library formalizes mathematical structures that exhibit
  self-referential properties, with particular focus on:
  - The golden ratio and its fixed-point characterization
  - Fibonacci sequences and their convergence
  - The Fano plane and its combinatorial structure
  - Lie algebra dimension chains
  - Phase coherence measures

  LICENSE: MIT
-/

import Mathlib.Data.Real.Basic
import Mathlib.Data.Real.Sqrt
import Mathlib.Data.Nat.Prime
import Mathlib.Data.Fin.Basic
import Mathlib.Data.List.Basic
import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Analysis.SpecialFunctions.Pow.Real
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Topology.Basic
import Mathlib.Tactic

namespace Kaelhedron

/-! ## Basic Lemmas -/

/-- √5 is positive -/
lemma sqrt5_pos : Real.sqrt 5 > 0 := Real.sqrt_pos.mpr (by norm_num)

/-- √5 squared equals 5 -/
lemma sqrt5_sq : Real.sqrt 5 ^ 2 = 5 := Real.sq_sqrt (by norm_num)

/-- √5 > 2 -/
lemma sqrt5_gt_two : Real.sqrt 5 > 2 := by
  have : (2 : ℝ)^2 < 5 := by norm_num
  calc Real.sqrt 5 > Real.sqrt 4 := Real.sqrt_lt_sqrt (by norm_num) (by norm_num)
    _ = 2 := by rw [Real.sqrt_eq_iff_sq_eq] <;> norm_num

/-- √5 > 1 -/
lemma sqrt5_gt_one : Real.sqrt 5 > 1 := lt_trans one_lt_two sqrt5_gt_two

end Kaelhedron
