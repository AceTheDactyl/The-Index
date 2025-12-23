# Central Bibliography for The-Index

**Purpose:** Canonical reference list for all research claims in the repository
**Format:** Organized by domain
**BibTeX:** See `BIBLIOGRAPHY.bib` for machine-readable citations

---

## Physics & Crystallography

### Quasi-Crystals

**Shechtman et al. (1984)** - Nobel Prize discovery
*Metallic Phase with Long-Range Orientational Order and No Translational Symmetry*
Physical Review Letters 53, 1951
DOI: 10.1103/PhysRevLett.53.1951

**Levine & Steinhardt (1984)** - Theoretical foundation
*Quasicrystals: A New Class of Ordered Structures*
Physical Review Letters 53, 2477
DOI: 10.1103/PhysRevLett.53.2477

**Senechal (1995)** - Comprehensive treatment
*Quasicrystals and Geometry*
Cambridge University Press
ISBN: 978-0521575416

**Janot (1994)** - Introductory text
*Quasicrystals: A Primer*
Oxford University Press
ISBN: 978-0198513896

### Hexagonal Geometry

**Hales (2001)** - Honeycomb conjecture proof
*The Honeycomb Conjecture*
Discrete & Computational Geometry 25, 1–22
DOI: 10.1007/s004540010071

### Mathematical Physics

**Morse & Feshbach (1953)** - Classical reference
*Methods of Theoretical Physics*
McGraw-Hill, New York
ISBN: 978-0070435148

**Arfken, Weber & Harris (2013)** - Modern comprehensive guide
*Mathematical Methods for Physicists*
7th Edition, Academic Press
ISBN: 978-0123846549

**Jackson (1999)** - Electrodynamics standard
*Classical Electrodynamics*
3rd Edition, Wiley
ISBN: 978-0471309321

---

## Synchronization & Nonlinear Dynamics

### Kuramoto Model

**Kuramoto (1984)** - Original formulation
*Chemical Oscillations, Waves, and Turbulence*
Springer Series in Synergetics, Volume 19
ISBN: 978-3-642-69689-3
DOI: 10.1007/978-3-642-69689-3

**Strogatz (2000)** - Modern review
*From Kuramoto to Crawford: exploring the onset of synchronization*
Physica D: Nonlinear Phenomena 143(1-4), 1–20
DOI: 10.1016/S0167-2789(00)00094-4

**Acebrón et al. (2005)** - Comprehensive review
*The Kuramoto model: A simple paradigm for synchronization phenomena*
Reviews of Modern Physics 77(1), 137
DOI: 10.1103/RevModPhys.77.137

**Rodrigues et al. (2016)** - Complex networks
*The Kuramoto model in complex networks*
Physics Reports 610, 1–98
DOI: 10.1016/j.physrep.2015.10.008

**Dörfler & Bullo (2014)** - Control theory perspective
*Synchronization in complex networks of phase oscillators: A survey*
Automatica 50(6), 1539–1564
DOI: 10.1016/j.automatica.2014.04.012

### General Synchronization

**Pikovsky, Rosenblum & Kurths (2001)** - Foundational textbook
*Synchronization: A Universal Concept in Nonlinear Sciences*
Cambridge Nonlinear Science Series, Volume 12
ISBN: 978-0521592857
DOI: 10.1017/CBO9780511755743

**Strogatz (2018)** - Modern dynamics textbook
*Nonlinear Dynamics and Chaos*
2nd Edition, CRC Press
ISBN: 978-0813349107

**Winfree (2001)** - Biological time
*The Geometry of Biological Time*
2nd Edition, Springer
ISBN: 978-0387989921
DOI: 10.1007/978-1-4757-3484-3

---

## Neuroscience & Biological Systems

### Neural Oscillations

**Hopfield (1982)** - Neural network foundations
*Neural networks and physical systems with emergent collective computational abilities*
PNAS 79(8), 2554–2558
DOI: 10.1073/pnas.79.8.2554

**Mirollo & Strogatz (1990)** - Pulse-coupled oscillators
*Synchronization of pulse-coupled biological oscillators*
SIAM Journal on Applied Mathematics 50(6), 1645–1662
DOI: 10.1137/0150098

**Dhamala, Jirsa & Ding (2004)** - Bursting neurons
*Transitions to synchrony in coupled bursting neurons*
Physical Review Letters 92(2), 028101
DOI: 10.1103/PhysRevLett.92.028101

**Cumin & Unsworth (2007)** - Neuronal synchronization
*Generalising the Kuramoto model for the study of neuronal synchronisation in the brain*
Physica D 226(2), 181–196
DOI: 10.1016/j.physd.2006.12.004

**Breakspear et al. (2010)** - Cortical oscillations
*Generative models of cortical oscillations: neurobiological implications*
Frontiers in Human Neuroscience 4, 190
DOI: 10.3389/fnhum.2010.00190

---

## Geometry & Systems Thinking

### Geometric Systems

**Coxeter (1973)** - Regular polytopes
*Regular Polytopes*
3rd Edition, Dover Publications
ISBN: 978-0486614809

**Fuller (1975)** - Synergetics
*Synergetics: Explorations in the Geometry of Thinking*
Macmillan, New York
ISBN: 978-0020653202

### Cognitive Science

**Varela, Thompson & Rosch (1991)** - Embodied cognition
*The Embodied Mind: Cognitive Science and Human Experience*
MIT Press
ISBN: 978-0262720212

---

## Project-Specific References

### Rosetta Bear / HRR

**Rosetta Bear Research Collective (2025)**
*The Holographic Resonance Reactor (HRR): A Geodesic, Phase-Coupled, Thermodynamically Bounded Field Computation System*
Technical Report
GitHub: https://github.com/AceTheDactyl/Rosetta-bear-project

---

## Citation Guidelines

### When to Cite

**ALWAYS cite:**
- Mathematical theorems or proofs
- Physical constants or experimental values
- Published theories or models
- Historical discoveries
- Specific algorithms or methods

**Document but don't necessarily cite:**
- Implementation details (document in code comments)
- Original synthesis of existing ideas (note which ideas)
- Novel applications (cite the original concepts being applied)

### Citation Format

**In Markdown documents:**
```markdown
According to Kuramoto (1984), phase oscillators synchronize when...

The honeycomb structure is optimal for hexagonal packing [Hales, 2001].

Multiple studies [Strogatz, 2000; Acebrón et al., 2005] demonstrate...
```

**In code comments:**
```python
# Based on Kuramoto model (Kuramoto, 1984)
# See: BIBLIOGRAPHY.md for full reference
def kuramoto_coupling(phases, coupling_strength):
    ...
```

**In INTEGRITY_METADATA:**
```
# Supporting Evidence:
#   - Shechtman et al. (1984) - Quasi-crystal discovery
#   - Levine & Steinhardt (1984) - Theoretical foundation
#   - See BIBLIOGRAPHY.md for details
```

### How to Add New References

1. Add entry to `BIBLIOGRAPHY.bib` in proper BibTeX format
2. Add human-readable entry to appropriate section in `BIBLIOGRAPHY.md`
3. Include: Author(s), Year, Title, Journal/Publisher, DOI/ISBN
4. Verify DOI links work

---

## Cross-References

Related documentation:
- `CITATION_STANDARDS.md` - Detailed citation requirements
- `INTEGRITY_METADATA_GUIDE.md` - How to document claims
- Individual research files - See references sections

---

**Last Updated:** 2025-12-23
**Maintainer:** Repository integrity system
**Format Version:** 1.0
