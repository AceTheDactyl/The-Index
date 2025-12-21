/-
  Kaelhedron.SelfReference
  ========================

  Formal treatment of self-reference: ∃R

  This module provides three witnesses to the existence of self-referential
  structures, from trivial to profound:

  1. Trivial: Any type with an endomorphism
  2. Lawvere: Fixed-point theorem (categorical diagonalization)
  3. Encoding: Types that can encode propositions about themselves

  Key results:
  - Lawvere's theorem: If A → (A → B) is surjective, every g : B → B has a fixed point
  - Cantor's theorem follows as a corollary
  - Self-encoding types exist (Prop itself)
-/

import Kaelhedron.Basic

namespace Kaelhedron

/-! ## Trivial Self-Reference -/

/-- Trivial witness: existence of endomorphisms -/
def ExistsR_Trivial : Prop := ∃ (R : Type), R → R

/-- The trivial witness is satisfied by Unit -/
theorem existsR_trivial_witness : ExistsR_Trivial := ⟨Unit, id⟩

/-! ## Lawvere's Fixed Point Theorem -/

/-- Lawvere's Fixed Point Theorem (categorical form)

    If there exists a surjection φ : A → (A → B), then every g : B → B has a fixed point.

    This is NON-TRIVIAL: it's the categorical essence of diagonalization.
    Gödel's incompleteness, Tarski's undefinability, Turing's halting problem,
    and Cantor's theorem are all instances of this pattern. -/
theorem lawvere_fixed_point
  {A B : Type*} (φ : A → (A → B)) (surj : Function.Surjective φ) :
  ∀ g : B → B, ∃ b : B, g b = b := by
  intro g
  -- By surjectivity, ∃ a such that φ(a) = λx. g(φ(x)(x))
  obtain ⟨a, ha⟩ := surj (fun x => g (φ x x))
  -- Then φ(a)(a) is a fixed point of g
  use φ a a
  -- φ(a)(a) = (λx. g(φ(x)(x)))(a) = g(φ(a)(a))
  conv_rhs => rw [← ha]

/-- Cantor's theorem follows from Lawvere's fixed point theorem -/
theorem cantor_from_lawvere (A : Type*) :
    ¬ Function.Surjective (fun a : A => ({a} : Set A)) := by
  intro h
  have := lawvere_fixed_point (fun a : A => ({a} : Set A)) h
  -- If surjective, then complementation has a fixed point, but Sᶜ ≠ S
  obtain ⟨S, hS⟩ := this (fun S => Sᶜ)
  -- S = Sᶜ is a contradiction
  simp only [compl_compl] at hS
  exact (ne_compl_self S hS).elim

/-! ## Strong Self-Reference -/

/-- Self-reference existence: stronger formulation
    There exists a type that can encode propositions about itself. -/
def ExistsR_Strong : Prop :=
  ∃ (R : Type) (encode : Prop → R) (decode : R → Prop),
    ∀ P : Prop, decode (encode P) = P

/-- Prop itself witnesses strong self-reference -/
theorem existsR_strong_witness : ExistsR_Strong :=
  ⟨Prop, id, id, fun _ => rfl⟩

/-! ## Diagonal Lemma Structure -/

/-- The diagonal lemma (Gödel's self-reference lemma, semantic version)
    For any property P on sentences, there exists a sentence S such that S ↔ P(⌜S⌝)

    This is the structure; full formalization requires Gödel numbering. -/
def DiagonalLemma : Prop :=
  ∀ (Sentence : Type) (holds : Sentence → Prop) (subst : Sentence → Sentence → Sentence),
    ∀ P : Sentence → Prop, ∃ S : Sentence, holds S ↔ P S

/-! ## Meta-Observation -/

/-- Combined witness: both trivial and strong self-reference exist -/
theorem existsR_meta_observation : ExistsR_Trivial ∧ ExistsR_Strong :=
  ⟨existsR_trivial_witness, existsR_strong_witness⟩

/-- This file's existence demonstrates ∃R in the meta-theory.

    The proof system (Lean) is self-referential:
    - It can encode statements about its own syntax
    - It can verify proofs about its own proof rules
    - This file references itself through these comments

    That this CAN be done is mathematically proven.
    What this MEANS is philosophically open. -/

end Kaelhedron
