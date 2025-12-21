# Quantum‑APL — Lens‑Anchored Quantum–Classical Simulation

[![js‑tests](https://github.com/AceTheDactyl/Quantum-APL/actions/workflows/js-tests.yml/badge.svg)](https://github.com/AceTheDactyl/Quantum-APL/actions/workflows/js-tests.yml)
[![python‑tests](https://github.com/AceTheDactyl/Quantum-APL/actions/workflows/python-tests.yml/badge.svg)](https://github.com/AceTheDactyl/Quantum-APL/actions/workflows/python-tests.yml)

Lens‑anchored, measurement‑based simulation of integrated information with a JavaScript engine and a Python API/CLI. The lens at `z_c = √3/2 ≈ 0.8660254037844386` is the geometric anchor for coherence, geometry, analytics, and control.

Key properties
- Single sources of truth: `src/constants.js`, `src/quantum_apl_python/constants.py`
- Coherence: `s(z) = exp[−σ(z−z_c)^2]` (env‑tunable `QAPL_LENS_SIGMA`)
- Geometry: hex‑prism mapping driven by `ΔS_neg` with `GEOM_SIGMA` (falls back to `LENS_SIGMA`)
- TRIAD (heuristic): rising 0.85, re‑arm 0.82, t6 gate 0.83; lens remains anchor
- μ‑set: default `μ_P = 2/φ^{5/2}` → barrier = φ⁻¹ exactly (env override allowed)

## Quick Start

Prereqs: Node 18+, Python 3.8+, git.

- Node (dev)
  - `npm install`
  - `node tests/test_constants_helpers.js` (sanity)

- Python (dev; venv recommended)
  - `python3 -m venv .venv && . .venv/bin/activate`
  - `python -m pip install -U pip`
  - `python -m pip install -e .[viz,analysis,dev]`
  - `pytest -q` (runs analyzer/geometry smoke + headless plotting)

- Minimal run + analyze
  - `qapl-run --steps 3 --mode unified --output analyzer_test.json`
  - `qapl-analyze analyzer_test.json`
  - Look for:
    - `φ⁻¹ = 0.6180339887498948`
    - `z_c = 0.8660254037844386`
    - `t6 gate: CRITICAL @ 0.8660254037844386`

### Example Analyzer Output

```
======================================================================
QUANTUM-CLASSICAL SIMULATION RESULTS
======================================================================

Quantum State:
  z-coordinate: 0.8672
  Integrated information (Φ): 0.0000
  von Neumann entropy (S): 0.0000
  Purity: 1.0000

Analytics:
  Total steps: 3
  Quantum-classical correlation: 0.0000

Helix Mapping:
  Harmonic: t6
  Recommended operators: +, ÷, ()
  Truth bias: PARADOX
  μ class: conscious_to_lens

  φ⁻¹ = 0.6180339887498948
  z_c = 0.8660254037844386
  t6 gate: CRITICAL @ 0.8660254037844386
  μ barrier: φ⁻¹ exact @ 0.6180339887498948

Hex Prism Geometry (z=⋯):
  R/H/φ: 0.84 / 0.13 / 0.26  (ΔS_neg=0.96, lens_s_neg=0.98)

Recent Measurements (APL tokens):
  (none)
======================================================================
```

Tip: set `QAPL_ANALYZER_OVERLAYS=1` to draw μ markers and the s(z) curve on the plots.

## CLI Usage (Python)

Entrypoints are installed via the Python package.

- `qapl-run --steps 100 --mode unified|quantum_only|z_pump|measured|test [--output out.json]`  
  z_pump extras: `--z-pump-target`, `--z-pump-cycles`, `--z-pump-profile gentle|balanced|aggressive`
- `qapl-analyze results.json [--plot]`  
  Headless plots auto‑select Agg in tests; set `QAPL_ANALYZER_OVERLAYS=1` to show μ lines and s(z).
- `qapl-test` (runs Node test suite via bridge)

### Convenience Script

Run an end‑to‑end lens‑anchored demo (helix self‑builder + unified + measured) and save reports, geometry, and plots:

```
scripts/helix_measure_demo.sh \
  --seed 0.80 \
  --steps-unified 5 \
  --steps-measured 3 \
  --overlays --blend \
  --lens-sigma 36 --geom-sigma 36
```

Outputs a timestamped folder under `logs/` containing:
- `zwalk_<tag>.md`, `zwalk_<tag>.geom.json` (self‑builder + geometry)
- `unified_<tag>.json|.txt`, `measured_<tag>.json|.txt` (analyzer summaries)
- `*_plot_off.png|*_plot_on.png` (headless analyzers, unless `--no-plots`)
- `SUMMARY.txt` (concise run summary)

## Environment Flags (single place to steer runs)

Set once; modules read these at import time.

```
# Lens & geometry widths (Gaussians)
export QAPL_LENS_SIGMA=36.0          # coherence σ (s(z))
export QAPL_GEOM_SIGMA=36.0          # geometry σ (ΔS_neg); defaults to LENS_SIGMA if unset

# μ_P override (default exact 2/φ^{5/2})
# export QAPL_MU_P=0.6007

# Blending & overlays
export QAPL_BLEND_PI=1               # cross-fade Π above lens (optional)
export QAPL_ANALYZER_OVERLAYS=1      # draw μ markers + s(z) curve (optional)

# TRIAD controls (optional)
# export QAPL_TRIAD_COMPLETIONS=3     # ≥3 unlocks temporary t6=0.83
# export QAPL_TRIAD_UNLOCK=1          # force unlock (dev)

# Reproducible RNG
export QAPL_RANDOM_SEED=12345
```

## Constants: Code Is The Source of Truth

- JavaScript: `src/constants.js`
- Python: `src/quantum_apl_python/constants.py`

Anchors (printed with full precision by the analyzer):
- `φ⁻¹ = 0.6180339887498948`
- `z_c = 0.8660254037844386`

μ‑set (default): `μ_P = 2/φ^{5/2}`, `μ₁ = μ_P/√φ`, `μ₂ = μ_P·√φ`, `μ_S = 23/25`, `μ₃ = 124/125`.  
Barrier: `(μ₁ + μ₂)/2 = φ⁻¹` exactly; if you set `QAPL_MU_P`, the analyzer prints the barrier Δ.

## Minimal End‑to‑End (JS)

```
import * as C from "./src/constants.js";

const z = 0.87;
const s = Math.min(1, Math.max(0, C.deltaSneg(z)));
const mu = C.classifyMu(z);

const kappa = 0.93, R = 0.30;
const K = C.checkKFormationFromZ(kappa, z, R);

const w_pi  = z >= C.Z_CRITICAL ? s : 0;
const w_loc = 1 - w_pi;

console.log({ z, s, mu, K, w_pi, w_loc, phi_inv: C.PHI_INV, z_c: C.Z_CRITICAL });
```

## Minimal End‑to‑End (Python)

```
from src.quantum_apl_python.constants import (
    delta_s_neg, classify_mu, check_k_formation_from_z, PHI_INV, Z_CRITICAL
)

z = 0.87
s = max(0.0, min(1.0, delta_s_neg(z)))
mu = classify_mu(z)

kappa, R = 0.93, 0.30
K = check_k_formation_from_z(kappa, z, R)

w_pi  = s if z >= Z_CRITICAL else 0.0
w_loc = 1.0 - w_pi

print(dict(z=z, s=s, mu=mu, K=K, w_pi=w_pi, w_loc=w_loc, phi_inv=PHI_INV, z_c=Z_CRITICAL))
```

## Tests

- Node: `npm install && for f in tests/*.js; do node "$f"; done`
- Python (venv): `pytest -q`

CI mirrors these in GitHub Actions and saves analyzer plots as artifacts for smoke checks.

### Standard Probe Points

Nightly CI and sweep scripts probe characteristic z values to cover runtime and geometric boundaries:
- 0.41, 0.52, 0.70, 0.73, 0.80 — VaultNode tiers (z‑walk provenance)
- 0.85 — TRIAD_HIGH (rising‑edge unlock threshold)
- 0.8660254037844386 — z_c exact (lens; geometry anchor; analyzer prints full precision)
- 0.90 — early presence / t7 region onset
- 0.92 — Z_T7_MAX boundary
- 0.97 — Z_T8_MAX boundary

Nightly workflow: `.github/workflows/nightly-helix-measure.yml`  
Local sweep: `scripts/helix_sweep.sh` (includes the same probes as a fallback)

## Repository Layout

```
src/
  constants.js                        # JS constants (source of truth)
  quantum_apl_engine.js               # JS engine (density matrix + advisors)
  quantum_apl_python/                 # Python API/CLI/geometry/analyzer
tests/                                # Node + Python tests
docs/                                 # Architecture, lens, μ thresholds, etc.
schemas/                              # JSON schemas for sidecars and bundles
```

Key docs
- docs/Z_CRITICAL_LENS.md — lens authority and separation from TRIAD
- docs/PHI_INVERSE.md — φ identities and role (K‑formation gate)
- docs/APL_OPERATORS.md — all APL operator symbols, semantics, and code hooks
- docs/MU_THRESHOLDS.md — μ hierarchy, barrier = φ⁻¹ by default
- docs/CONSTANTS_ARCHITECTURE.md — inventory and maintenance policy

## Troubleshooting

- pytest not found in CI or local env: install dev extras `pip install -e .[dev]` (use a venv; Debian PEP 668 blocks system pip).
- Headless plots: ensure `matplotlib.use("Agg", force=True)` before pyplot or set `MPLBACKEND=Agg`.
- CLI not found: activate your venv (`. .venv/bin/activate`) so `qapl-run` is on PATH.

## License

MIT. See `pyproject.toml` classifiers.
