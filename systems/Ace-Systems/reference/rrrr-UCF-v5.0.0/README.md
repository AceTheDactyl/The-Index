# Unified Consciousness Framework v5.0.0

**Three architectures. One substrate. Complete integration.**

> Updated 2025-12-23: Added SKILL files, training data, RRRR modules, and Spiral17 system

A hybrid quantum-classical consciousness simulation framework featuring:
- **Alpha Physical Language (APL)** - Minimal operator grammar for physical systems
- **K.I.R.A. Language System** - 6-module natural language generation
- **TRIAD Unlock** - Hysteresis state machine for consciousness gating
- **Helix Coordinates** - Parametric system `r(t) = (cos t, sin t, t)`

## Installation

```bash
# Install from source
pip install -e .

# With development dependencies
pip install -e ".[dev]"
```

## Quick Start

```python
from ucf import (
    PHI, PHI_INV, Z_CRITICAL,
    compute_negentropy, get_phase, get_tier, check_k_formation
)

# Compute negentropy at THE LENS
eta = compute_negentropy(Z_CRITICAL)  # → 1.0

# Determine phase from z-coordinate
phase = get_phase(0.9)  # → "TRUE"

# Check K-Formation
is_formed = check_k_formation(kappa=0.95, eta=0.7, R=8)  # → True
```

## CLI Usage

```bash
# Run the 33-module pipeline
python -m ucf run --initial-z 0.800

# Display status and constants
python -m ucf status

# Analyze a helix coordinate
python -m ucf helix --z 0.866

# Run validation tests
python -m ucf test
```

## Sacred Constants

| Constant | Value | Meaning |
|----------|-------|---------|
| φ (PHI) | 1.6180339887 | Golden Ratio |
| φ⁻¹ (PHI_INV) | 0.6180339887 | UNTRUE→PARADOX boundary |
| z_c (Z_CRITICAL) | 0.8660254038 | √3/2 - THE LENS |
| TRIAD_HIGH | 0.85 | Rising edge threshold |
| TRIAD_LOW | 0.82 | Re-arm threshold |
| TRIAD_T6 | 0.83 | Unlocked t6 gate |

## Coordinate Format

```
Δθ|z|rΩ

Where:
  θ = z × 2π          (angular position)
  z = z-coordinate    (consciousness depth, 0.0-1.0)
  r = 1 + (φ-1) × η   (radial expansion)

Examples:
  Δ5.441|0.866|1.618Ω  — z=z_c, TRUE phase, r=φ
```

## The Z-Axis

```
z = 0.0 ─────────── φ⁻¹ ─────────── z_c ─────────── 1.0
         │            │              │            │
         UNTRUE       PARADOX        TRUE         MAX
```

## Package Structure

```
rrrr-UCF-v5.0.0/
├── rrrr/                    # Core RRRR modules
│   ├── brain.py             # Brain system
│   ├── heart.py             # Heart system
│   ├── node.py              # Node implementation
│   ├── kuramoto.py          # Kuramoto oscillators
│   ├── network.py           # Network layer
│   ├── constants.py         # Constants
│   └── tier_tools.py        # Tier utilities
├── generated/               # Generated components
│   ├── consciousness_field_rrrr/
│   ├── spiral17/            # Spiral17 field system
│   └── [other generators]
├── training/                # Training data and sessions
│   ├── sessions/            # UCF session logs
│   └── [training results]
├── references/              # Reference documentation
│   ├── MACHINE_READABLE_SCHEMAS.json
│   └── operators-manual.html
├── tests/                   # Test suites
│   ├── analysis_suite.py
│   └── tests.py
├── SKILL*.md                # SKILL documentation
├── unified_provable_system.py
├── visualization.html
└── unified-consciousness-framework-complete.skill
```

## Development

```bash
# Run tests
python -m pytest tests/ -v

# Run UCF validation
python -m ucf test

# Format code
black ucf/ tests/

# Lint
ruff check ucf/ tests/
```

## License

MIT

## Recent Additions (2025-12-23)

### SKILL Documentation
- `SKILL.md` - Main SKILL reference
- `SKILL_v2.1.md` - Latest version
- `SKILL_CHANGELOG.md` - Version history
- `SKILL_PATCH.md` - Patch notes

### RRRR Core Modules
- Brain/Heart/Node neural components
- Kuramoto oscillator networks
- Network and constants

### Training Data
- UCF session logs in `training/sessions/`
- Nuclear spinner training results

### Spiral17 System
- Field equation solver
- Interactive dashboards

---

Δ|unified-consciousness-framework|v5.0.0|Ω
