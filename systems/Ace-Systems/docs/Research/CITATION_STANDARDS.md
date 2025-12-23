# Citation Standards for The-Index

**Status:** ✓ JUSTIFIED - Documentation standard
**Severity:** LOW RISK
**Purpose:** Establish citation requirements for all research claims
**Version:** 1.0
**Date:** 2025-12-23

---

## Core Principle

**All verifiable claims must be either cited or marked as original synthesis.**

This standard ensures:
- Reproducibility of research
- Proper attribution to original sources
- Clear distinction between established knowledge and novel ideas
- Ability to verify claims against external sources

---

## Citation Requirements by Claim Type

### 1. Mathematical Claims

**MUST cite** when claiming:
- Specific theorems (e.g., "By the honeycomb conjecture...")
- Mathematical constants (e.g., "The golden ratio φ = 1.618...")
- Proofs or derivations from literature
- Specific mathematical formulations

**Example:**
```markdown
✓ CORRECT:
The honeycomb structure minimizes perimeter for a given area (Hales, 2001).

✗ INCORRECT:
The honeycomb structure minimizes perimeter for a given area.
```

**For original derivations:**
```markdown
We derive a new relationship between z_c and φ:

[derivation steps]

This extends the work of Levine & Steinhardt (1984) on quasi-crystal geometry.
```

---

### 2. Physical Constants & Experimental Data

**MUST cite** when referencing:
- Measured physical constants
- Experimental results
- Material properties
- Observable phenomena

**Example:**
```markdown
✓ CORRECT:
Graphene has a lattice constant of 2.46 Å (measured via X-ray diffraction).
Citation: Katsnelson (2012), "Graphene: Carbon in Two Dimensions"

✗ INCORRECT:
Graphene has a lattice constant of 2.46 Å.
```

**For well-known constants:**
```markdown
ACCEPTABLE (no citation needed for basic constants):
The speed of light c = 3×10⁸ m/s
π = 3.14159...
The golden ratio φ = (1+√5)/2
```

---

### 3. Published Theories & Models

**MUST cite** when using:
- Named models (Kuramoto model, Hopfield networks, etc.)
- Published theoretical frameworks
- Established research paradigms

**Example:**
```markdown
✓ CORRECT:
We implement the Kuramoto model (Kuramoto, 1984) to simulate phase oscillator synchronization.

✗ INCORRECT:
We use phase oscillators that synchronize via coupling.
(This describes the Kuramoto model but doesn't cite it)
```

---

### 4. Historical Discoveries

**MUST cite:**
- Nobel Prize-winning discoveries
- Landmark experiments
- First observations of phenomena

**Example:**
```markdown
✓ CORRECT:
Quasi-crystals were discovered by Shechtman et al. (1984), earning the 2011 Nobel Prize in Chemistry.

✗ INCORRECT:
Quasi-crystals were discovered in the 1980s.
```

---

### 5. Algorithms & Computational Methods

**MUST cite or document:**
- Published algorithms
- Optimization techniques from literature
- Standard computational methods

**Example:**
```python
# ✓ CORRECT:
# Kuramoto model implementation
# Based on: Kuramoto (1984), "Chemical Oscillations, Waves, and Turbulence"
# See: BIBLIOGRAPHY.md
def kuramoto_step(phases, omega, K, dt):
    ...

# ✗ INCORRECT:
def kuramoto_step(phases, omega, K, dt):
    # Phase oscillator update
    ...
```

---

### 6. Implementation Details

**Documentation required, citation optional:**
- Code architecture decisions
- Performance optimizations
- API design choices

**Example:**
```python
# Documentation: We use NumPy vectorization for performance
# (Standard practice, no citation needed)
coupling = K * np.sin(phases[:, None] - phases)
```

---

## What DOESN'T Need Citation

### Original Work

**No citation needed for:**
- Novel synthesis of existing ideas (but note which ideas)
- Original implementations of known concepts
- Application of known methods to new problems
- Creative interpretations (marked as such)

**Example:**
```markdown
✓ CORRECT:
We synthesize the Kuramoto model (Kuramoto, 1984) with quasi-crystal geometry
(Shechtman et al., 1984) to create a novel phase-coupled oscillator network
on a Penrose tiling. This synthesis is original to this work.

✗ INCORRECT:
We create a novel oscillator network on a Penrose tiling.
(Unclear what's novel vs what's from literature)
```

### Common Knowledge in Domain

**No citation needed for:**
- Basic definitions in the field
- Standard notation
- Widely-known relationships

**Example:**
```markdown
ACCEPTABLE (no citation):
A triangle has three sides.
The derivative measures rate of change.
Neural networks consist of interconnected nodes.

CITATION REQUIRED:
The universal approximation theorem states that neural networks can approximate any continuous function (Cybenko, 1989).
```

---

## Citation Format Standards

### In Markdown Documents

**Inline citations:**
```markdown
According to Kuramoto (1984), synchronization emerges when...

The critical coupling is K_c = 2/π (Strogatz, 2000).

Multiple studies [Kuramoto, 1984; Strogatz, 2000; Acebrón et al., 2005] show...
```

**References section:**
```markdown
## References

1. Kuramoto, Y. (1984). *Chemical Oscillations, Waves, and Turbulence*. Springer.
2. Strogatz, S. H. (2000). From Kuramoto to Crawford. *Physica D* 143(1-4), 1–20.

For complete citations, see: [BIBLIOGRAPHY.md](BIBLIOGRAPHY.md)
```

### In Python Code

**Module docstring:**
```python
"""
Kuramoto model implementation for phase oscillator synchronization.

Based on:
    Kuramoto, Y. (1984). Chemical Oscillations, Waves, and Turbulence. Springer.

See also:
    BIBLIOGRAPHY.md for complete references
    PHYSICS_GROUNDING.md for physical justification
"""
```

**Function/class comments:**
```python
def calculate_order_parameter(phases):
    """
    Calculate Kuramoto order parameter r.

    The order parameter measures synchronization (Kuramoto, 1984).
    r = 0: no synchronization
    r = 1: perfect synchronization

    References:
        See BIBLIOGRAPHY.md - Kuramoto (1984)
    """
    ...
```

### In INTEGRITY_METADATA

```markdown
# INTEGRITY_METADATA
# Status: ✓ JUSTIFIED - Claims supported by citations
# Severity: LOW RISK
#
# Supporting Evidence:
#   - Shechtman et al. (1984) - Quasi-crystal discovery [Nobel Prize]
#   - Levine & Steinhardt (1984) - Theoretical foundation
#   - Full references in BIBLIOGRAPHY.md
#
# Original Contributions:
#   - Novel synthesis of quasi-crystal geometry with Kuramoto dynamics
#   - Implementation details are original to this repository
```

---

## Citation Templates

### Template 1: Single Paper

```markdown
According to [Author(s)] ([Year]), [claim or finding].

Example:
According to Shechtman et al. (1984), quasi-crystals exhibit long-range
orientational order without translational symmetry.
```

### Template 2: Multiple Papers

```markdown
Multiple studies [Author1, Year1; Author2, Year2] demonstrate [finding].

Example:
Multiple studies [Kuramoto, 1984; Strogatz, 2000; Acebrón et al., 2005]
demonstrate that coupled oscillators can synchronize.
```

### Template 3: With Quote

```markdown
As [Author(s)] ([Year]) state: "[direct quote]"

Example:
As Hales (2001) proves: "The honeycomb conjecture holds - the regular
hexagonal grid has the smallest perimeter among all partitions of the plane."
```

### Template 4: Mathematical Result

```markdown
By [theorem/lemma] ([Author(s)], [Year]), we have [result].

Example:
By the honeycomb conjecture (Hales, 2001), hexagonal packing minimizes perimeter.
```

### Template 5: Code Citation

```python
# Based on: [Model/Algorithm] ([Author(s)], [Year])
# Reference: BIBLIOGRAPHY.md - [Key]
# Implementation: [Original/Modified/Adapted]

Example:
# Based on: Kuramoto model (Kuramoto, 1984)
# Reference: BIBLIOGRAPHY.md - kuramoto1984
# Implementation: Original Python implementation with NumPy optimization
```

---

## Adding New Citations

### Step-by-Step Process

1. **Find the source:**
   - Locate original paper, book, or authoritative source
   - Verify it's peer-reviewed or authoritative
   - Get DOI, ISBN, or stable URL

2. **Add to BIBLIOGRAPHY.bib:**
   ```bibtex
   @article{key2025,
     title={Paper Title},
     author={Last, First and Last, First},
     journal={Journal Name},
     volume={10},
     pages={1--20},
     year={2025},
     doi={10.xxxx/xxxxx}
   }
   ```

3. **Add to BIBLIOGRAPHY.md:**
   - Choose appropriate section (Physics, Sync, Neuro, etc.)
   - Follow format: **Author(s) (Year)** - Description
   - Include title, journal/publisher, DOI/ISBN

4. **Use in documents:**
   - Cite using (Author, Year) format
   - Add to References section
   - Point readers to BIBLIOGRAPHY.md

5. **Update INTEGRITY_METADATA:**
   - List new citation in Supporting Evidence
   - Specify what it supports

---

## Enforcement

### Required for Pull Requests

All new research claims in PRs must:
- ✓ Be cited if from literature
- ✓ Be marked as "original synthesis" if novel
- ✓ Include references in appropriate format
- ✓ Update BIBLIOGRAPHY.md if adding new sources

### Metadata Requirements

Files with research claims must have:
```markdown
# INTEGRITY_METADATA
# Status: ✓ JUSTIFIED - Claims cited
# Severity: LOW RISK
#
# Supporting Evidence:
#   - [Citation 1]
#   - [Citation 2]
#   - See BIBLIOGRAPHY.md
```

---

## Examples: Before & After

### Example 1: Physics Claim

**BEFORE (missing citation):**
```markdown
The critical value z_c = √3/2 appears in hexagonal lattices.
```

**AFTER (properly cited):**
```markdown
The critical value z_c = √3/2 emerges from hexagonal close-packed structures,
observed in graphene (Novoselov et al., 2004) and HCP metals (Ashcroft & Mermin, 1976).
```

### Example 2: Algorithm Implementation

**BEFORE (unclear origin):**
```python
def synchronize_oscillators(phases):
    # Update phases
    return new_phases
```

**AFTER (properly documented):**
```python
def kuramoto_synchronization(phases, coupling, frequencies, dt):
    """
    Kuramoto model synchronization step.

    Based on: Kuramoto (1984), Chemical Oscillations, Waves, and Turbulence
    Reference: BIBLIOGRAPHY.md - kuramoto1984

    Implementation: Original Python code following Eq. 4.1 from Kuramoto (1984)
    """
    # Kuramoto coupling: K/N * sum(sin(theta_j - theta_i))
    coupling_term = coupling * np.mean(np.sin(phases[:, None] - phases), axis=1)
    return phases + (frequencies + coupling_term) * dt
```

### Example 3: Novel Synthesis

**BEFORE (unclear what's novel):**
```markdown
We combine oscillators with quasi-crystal geometry for improved synchronization.
```

**AFTER (clearly attributed):**
```markdown
We synthesize the Kuramoto model (Kuramoto, 1984) with Penrose tiling geometry
(Penrose, 1974) to create a quasi-crystalline oscillator network. While both
components are established, their combination for synchronization analysis is
novel to this work. The coupling topology follows quasi-crystal symmetry
(Levine & Steinhardt, 1984) rather than traditional lattices.
```

---

## Verification Checklist

Before merging research content, verify:

- [ ] All mathematical theorems cited
- [ ] All physical constants sourced or marked as standard
- [ ] All named models/theories cited
- [ ] All historical discoveries attributed
- [ ] Novel work clearly marked as "original synthesis"
- [ ] BIBLIOGRAPHY.md updated if new sources added
- [ ] BIBLIOGRAPHY.bib updated with BibTeX entries
- [ ] INTEGRITY_METADATA includes Supporting Evidence
- [ ] References section present in document

---

## Related Documentation

- `BIBLIOGRAPHY.md` - Central reference list
- `BIBLIOGRAPHY.bib` - BibTeX format for citations
- `INTEGRITY_METADATA_GUIDE.md` - How to document file integrity
- Individual research files - Examples of proper citation

---

**Enforcement Level:** REQUIRED for all research claims
**Review Process:** Checked in PR reviews
**Exceptions:** None (but "common knowledge" exemptions apply)

**Last Updated:** 2025-12-23
**Next Review:** When citation practices need refinement
**Maintainer:** Repository integrity system
