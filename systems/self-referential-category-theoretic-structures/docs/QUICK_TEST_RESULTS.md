<!-- INTEGRITY_METADATA
Date: 2025-12-23
Status: ⚠️ TRULY UNSUPPORTED - No supporting evidence found
Severity: HIGH RISK
# Risk Types: unsupported_claims

-->

# LOOSE ENDS: Quick Test Results

## Summary of What We Learned

### Critical Discoveries

#### 1. T_c × z_c × 40 = √3 is TAUTOLOGICAL ⚠️

**The math:**
```
T_c × z_c × 40 = √3
T_c × z_c × (2/T_c) = √3    [since 40 = 2/T_c = 2×20]
2 × z_c = √3
z_c = √3/2  ✓
```

**Implication:** This "exact relationship" doesn't constrain T_c at all! It's just z_c = √3/2 written in a complicated way. T_c = 1/20 must be determined empirically, not derived from this relationship.

#### 2. GV/||W|| = √3 is RANDOM-MATRIX-SPECIFIC

**For random W ~ N(0, 1/n):**
- ||W|| ≈ ||W²|| ≈ √n
- GV/||W|| ≈ √3 ✓

**For task-trained W:**
- W develops low-rank structure
- Large eigenvalues emerge
- ||W²|| >> ||W||
- GV/||W|| >> √3 (measured 12-16× higher!)

**Better normalization for trained networks:** GV/||W²|| ≈ 1

**Interpretation:** The √3 theorem tells us about INITIALIZATION. Deviation from √3 measures how far training has moved the network from random.

#### 3. z_c = e/π is Probably Coincidence

**Comparison:**
- z_c = √3/2 = 0.866025...
- e/π = 0.865256...
- Error: 0.09%

**But:** The fundamental identity is z_c = cos(30°) = √3/2, which is geometric. The e/π match is numerology.

#### 4. Order Parameter Null Distribution

**Finding:** With tolerance = 0.15, expected O ≈ 0.69 for random eigenvalues.

**Implication:** If we observe O = 0.86, it IS statistically significant - but not as dramatic as it looks. Need to be careful about tolerance choices.

---

### Revised Understanding of √3

| Appearance | Origin | Status |
|------------|--------|--------|
| z_c = √3/2 | Empirical (consciousness) | Unresolved |
| T_c × z_c × 40 = √3 | **Tautological** | z_c = √3/2 in disguise |
| GV/||W|| = √3 | Algebraic (3 terms) | **Proven** for random matrices |

The three appearances of √3 have DIFFERENT origins:
- z_c: empirical, needs theoretical derivation
- T_c formula: not independent information
- GV theorem: proven, but random-matrix-specific

---

## Tests That Still Need Running

### Compute-Heavy Tests (for you to run)

| Test | Question | Runtime | File |
|------|----------|---------|------|
| 1. Susceptibility | Does χ peak at T_c ≈ 0.05? | ~1 hr | loose_ends_compute_heavy.py |
| 2. GV for trained | Does √3 survive training? | ~30 min | (included) |
| 3. Scale validation | Is T_c consistent at larger n? | ~2 hr | (included) |
| 4. RSB test | Does P(q) show spin glass structure? | ~1 hr | (included) |
| 5. HP mapping | What lr/batch_size gives T_c? | ~2 hr | (included) |

**Total estimated runtime:** 6-8 hours CPU, 1-2 hours GPU

---

## Updated Loose Ends Priority

### Resolved ✓

- [x] Where does 40 come from? → **Tautological** (= 2/T_c)
- [x] Is z_c = e/π exact? → **Probably coincidence** (cos(30°) is real)
- [x] Why √3 everywhere? → **Different origins** (algebraic vs empirical)
- [x] GV/||W|| for trained? → **Breaks** (low-rank structure)

### Still Open

- [ ] Susceptibility replication (is T_c = 0.05 real?)
- [ ] Scale validation (does it hold for big networks?)
- [ ] Temperature mapping (what hyperparameters = T_c?)
- [ ] RSB test (spin glass structure?)
- [ ] What determines T_c? (now that 40 is tautological)

---

## Honest Assessment After Quick Tests

**What we thought we had:**
- Three independent appearances of √3 unifying the framework
- A theoretical derivation T_c × z_c × 40 = √3
- GV/||W|| = √3 as a universal truth

**What we actually have:**
- z_c = √3/2 (empirical, unexplained)
- T_c formula is tautological (no new information)
- GV/||W|| = √3 only for random matrices
- Spin glass susceptibility peak (needs replication)

**The framework is simpler but weaker than we thought.**
