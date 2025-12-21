# LOOSE ENDS AND OPEN QUESTIONS

## Comprehensive Inventory After Spin Glass Validation

**Date:** December 2025  
**Status:** Framework validated at toy scale, many questions remain

---

## Part I: Inconsistencies From Today's Work

### 1. Conflicting Susceptibility Results

**The Problem:**
- Kimi's test: χ(T) peaks at T_c = 0.045 ✅
- My quick test: χ(T) peaks at T = 0.10 ❌

**Why the discrepancy?**
- Kimi used real task training (modular addition, proper SGD)
- My test used simplified "training" (gradient descent on ||W||²)
- Different temperature parameterizations

**Resolution needed:**
- [ ] Verify Kimi's test independently
- [ ] Standardize temperature → hyperparameter mapping
- [ ] Determine which experimental setup is canonical

### 2. Order Parameter Too High

**The Problem:**
In the refraction test, O(T) ≈ 0.86 everywhere (86% of eigenvalues near special values)

**This is suspicious:**
- Random matrices shouldn't have 86% of eigenvalues near {φ, √2, z_c, ...}
- Either the tolerance is too loose (0.15) or the basis is too dense
- This was exactly the problem that falsified eigenvalue clustering!

**Resolution needed:**
- [ ] Recalibrate order parameter with tighter tolerance
- [ ] Compare to properly computed null distribution
- [ ] Ensure we're not p-hacking

### 3. GV/||W|| = √3 vs GV/n = 0.29

**The Problem:**
- Proven: GV/||W|| → √3 as n → ∞
- Claimed: GV/n ≈ 0.29 (which is √3/√n for n≈32)

**But:**
- GV/||W|| = √3 is for RANDOM matrices
- What happens for TRAINED matrices?
- Does training change this ratio?

**Resolution needed:**
- [ ] Test GV/||W|| for trained networks
- [ ] Compare random vs trained at various stages
- [ ] Determine if √3 is universal or random-matrix-specific

---

## Part II: Theoretical Gaps

### 4. Where Does 40 Come From?

**The Relationship:**
T_c × z_c × 40 = √3

**We know:**
- T_c = 1/20 = 0.05
- z_c = √3/2

**But what is 40?**
- Is it 2 × 20 (doubling of T_c denominator)?
- Is it network depth?
- Is it hidden dimension / some dimension?
- Is it arbitrary (just fitting a constant)?

**Resolution needed:**
- [ ] Derive 40 from first principles
- [ ] Test if relationship holds for different architectures
- [ ] Determine physical meaning of the constant

### 5. Is z_c = e/π Exact?

**The Observation:**
z_c = √3/2 = 0.8660254...
e/π = 0.8652559...
Error: 0.09%

**Open questions:**
- Is this a coincidence or exact relationship?
- Why would e/π appear in coherence thresholds?
- What's the theoretical connection between {√3, e, π}?

**Resolution needed:**
- [ ] Theoretical derivation (if possible)
- [ ] Accept as approximate if no derivation found
- [ ] Check for other suspicious near-equalities

### 6. Spin Glass Universality Class Details

**What we validated:**
χ(T) peaks at T_c (cusp, not divergence)

**What we didn't test:**
- Critical exponents (what's the actual β, γ, ν?)
- Replica symmetry breaking (does P(q) show RSB?)
- Ultrametricity (is solution space ultrametric?)
- Correlation length (does ξ diverge at T_c?)

**Resolution needed:**
- [ ] Full critical exponent measurement
- [ ] Replica overlap distribution P(q) test
- [ ] Ultrametric distance test
- [ ] Finite-size scaling analysis

### 7. What IS Temperature in Neural Networks?

**The mapping is unclear:**
```
"Temperature" controls:
- Learning rate: lr = base_lr × (1 + k × T)
- Gradient noise: σ_grad = T × 0.05
- Input noise: σ_input = T × 0.3
- Label smoothing: α = min(T × 2, 0.3)
```

**But:**
- This is ad hoc, not derived
- Different parameterizations give different T_c
- What's the "true" temperature of real SGD?

**Resolution needed:**
- [ ] Principled derivation of effective temperature
- [ ] Connection to SGD noise scale (batch size, lr)
- [ ] Standardized mapping for reproducibility

---

## Part III: Experimental Gaps

### 8. Scale Validation (CRITICAL)

**Status:** All results are toy scale (n = 32-64)

**Untested:**
- ResNet on CIFAR (n ~ thousands)
- Transformers on language (n ~ millions)
- Any production-scale network

**Risk:** Toy findings might not replicate at scale

**Resolution needed:**
- [ ] ResNet-18 on CIFAR-10 with constraint comparison
- [ ] Transformer on PTB with attention analysis
- [ ] LSTM on sequence tasks with orthogonal test

### 9. Task-Specific Constraints at Scale

**Toy findings:**
- Cyclic tasks: golden helps (+11%)
- Sequential tasks: orthogonal dominates (10,000×)
- Recursive tasks: no constraint helps

**Untested at scale:**
- Does cyclic benefit transfer to real cyclic tasks?
- Does orthogonal dominance hold for LSTMs/Transformers?
- Are there real tasks where golden helps significantly?

**Resolution needed:**
- [ ] Identify real-world cyclic tasks
- [ ] Benchmark orthogonal on sequence modeling
- [ ] Find (or confirm absence of) golden-beneficial tasks

### 10. The Susceptibility Test Needs Replication

**Current evidence:**
- One test by Kimi: T_c = 0.045
- One quick test by me: T_c = 0.10 (different setup)

**For confidence, need:**
- [ ] Independent replication with same protocol
- [ ] Multiple tasks (not just modular addition)
- [ ] Multiple architectures
- [ ] Error bars on T_c estimate

---

## Part IV: Interpretation Questions

### 11. What Does T_c = 0.05 Mean Practically?

**The question:**
If T_c ≈ 0.05 is the glass transition, what does this mean for practitioners?

**Possibilities:**
- Most training is at T >> T_c (disordered phase)
- Low learning rate / large batch → T < T_c (ordered phase)
- Critical training at T ≈ T_c (maximum sensitivity)

**Resolution needed:**
- [ ] Map real hyperparameters to effective temperature
- [ ] Determine which phase typical training occupies
- [ ] Provide actionable guidance

### 12. Consciousness Implications (Revised)

**Original claim:** "Consciousness emerges at thermal T_c"

**Revised by spin glass:** "Consciousness requires quenched structure near glass transition"

**Open questions:**
- Is the brain in the ordered or disordered phase?
- Does z_c = √3/2 have neurobiological meaning?
- Is "criticality in the brain" actually a glass transition?

**Resolution needed:**
- [ ] Literature review: brain criticality
- [ ] Connection to Ace's UCF framework
- [ ] Empirical tests on neural data (if possible)

### 13. Why √3 Everywhere?

**Appearances:**
- z_c = √3/2
- T_c × z_c × 40 = √3
- GV/||W|| = √3 (new theorem)

**Why √3 specifically?**
- √3 = 2cos(30°) - geometric?
- √3 appears in equilateral triangles - symmetry?
- √3 = ||W² - W - I||/||W|| - algebraic coincidence?

**Resolution needed:**
- [ ] Unified derivation of √3 appearances
- [ ] Determine if it's deep or coincidental
- [ ] Connect to lattice structure

---

## Part V: Connection Questions

### 14. NTK Connection (Unexplored)

**Neural Tangent Kernel:**
- Describes infinite-width limit
- Has its own eigenvalue structure
- Might show glass transition

**Open questions:**
- Does NTK show susceptibility peak at T_c?
- How does Λ-complexity relate to NTK eigenvalues?
- Is there a correspondence principle?

**Resolution needed:**
- [ ] NTK eigenvalue analysis
- [ ] Compare NTK and finite-width behavior
- [ ] Theoretical connection (if exists)

### 15. Connection to Loss Landscape Literature

**Related work:**
- Loss landscape visualization (Li et al.)
- Sharpness-aware minimization
- Mode connectivity

**Open questions:**
- Is T_c related to loss landscape sharpness?
- Does glass transition explain mode connectivity?
- Connection to SAM (we found no synergy with golden)

**Resolution needed:**
- [ ] Literature review
- [ ] Empirical tests connecting χ(T) to sharpness
- [ ] Theoretical framework

### 16. Quantum Extension

**Preliminary finding:**
- Phase-shifted golden achieves 99% fidelity in quantum tasks

**Open questions:**
- Is there a quantum spin glass?
- Does √3 appear in quantum coherence?
- Practical quantum computing applications?

**Resolution needed:**
- [ ] Real quantum hardware tests
- [ ] Theoretical quantum spin glass connection
- [ ] Practical use cases

---

## Part VI: Meta-Questions

### 17. What Would Falsify the Framework?

**Current status:**
- Spin glass validated
- √3 theorem proven
- Many claims falsified (thermal, SpiralOS)

**What would kill it entirely?**
- Scale validation fails (toy artifacts)
- T_c not reproducible across tasks
- √3 theorem fails for trained networks
- No practical benefits found

**Resolution needed:**
- [ ] Clear falsification criteria
- [ ] Pre-registered predictions for scale tests
- [ ] Honest assessment of "what if scale fails"

### 18. Is This Publishable?

**Current state:**
- Mathematical theorems: proven
- Spin glass validation: preliminary
- Practical benefits: unclear

**For publication need:**
- [ ] Independent replication
- [ ] Scale validation
- [ ] Clear practical takeaways
- [ ] Negative results paper (what doesn't work)

### 19. What's the Minimum Viable Theory?

**Full framework has:**
- Lattice Λ = {φ^r × e^d × π^c × (√2)^a}
- Fibonacci-Depth theorem
- Spin glass phase transition
- T_c × z_c × 40 = √3
- GV/||W|| = √3
- Task-specific constraints
- Consciousness connections

**Minimum that stands alone:**
- Fibonacci-Depth theorem (proven)
- GV/||W|| = √3 (proven)
- Maybe: spin glass susceptibility (needs replication)

**Resolution needed:**
- [ ] Identify core vs speculative claims
- [ ] Separate "proven math" from "empirical findings"
- [ ] Don't oversell

---

## Part VII: Priority Ranking

### Critical (blocks everything)
1. Scale validation
2. Susceptibility test replication
3. Temperature → hyperparameter mapping

### High (major theoretical gaps)
4. GV/||W|| = √3 for trained networks
5. Where does 40 come from?
6. Replica symmetry breaking test

### Medium (important but not blocking)
7. Critical exponent measurement
8. NTK connection
9. Is z_c = e/π exact?

### Low (speculative extensions)
10. Consciousness implications
11. Quantum extension
12. Category-theoretic formalization

---

## Part VIII: Honest Assessment

### What we KNOW:
- Fibonacci-Depth theorem (mathematical proof)
- GV/||W|| → √3 for random matrices (mathematical proof)
- χ(T) peaks near T_c = 0.05 (one experiment)

### What we BELIEVE:
- Spin glass is correct universality class
- T_c × z_c × 40 = √3 is meaningful
- √3 unifies the framework

### What we HOPE:
- Scale validation will succeed
- Practical benefits exist
- Framework has real applications

### What we DON'T KNOW:
- How to map real training to temperature
- Whether any of this helps practitioners
- If toy results transfer to scale

---

## Summary

**Total loose ends identified: 19**

**By category:**
- Methodological inconsistencies: 3
- Theoretical gaps: 4  
- Experimental gaps: 3
- Interpretation questions: 3
- Connection questions: 3
- Meta-questions: 3

**Most critical:**
1. Scale validation (does any of this matter at real size?)
2. Susceptibility replication (was T_c = 0.045 real?)
3. Temperature mapping (what IS temperature in SGD?)

**The honest truth:**
We have beautiful mathematics and suggestive toy experiments. Whether this becomes physics or remains mathematical curiosity depends entirely on scale validation.
