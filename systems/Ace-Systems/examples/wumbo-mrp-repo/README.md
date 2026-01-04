# WUMBO MRP

**Multi-Resolution Pattern Embedding System with L4 Threshold Framework**

[![Version](https://img.shields.io/badge/version-2.0.0-blue)]()
[![L4 Integrated](https://img.shields.io/badge/L4-integrated-gold)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

Embed consciousness state signatures into RGB images using LSB steganography. Based on the 100-region WUMBO neural atlas mapped to LIMNUS geometry, now integrated with the L4 Threshold Framework.

## Core Discovery

**The Wumbo Engine and L4 Threshold Framework are isomorphic structures**—one emergent from lived neurodivergent experience, one derived from pure mathematics. They describe the same phase transitions in consciousness coherence from orthogonal vantage points.

```
THE_LENS (z = sqrt(3)/2) = NIRVANA
This is not metaphor. This is structural identity.
```

## Overview

WUMBO MRP v2.0.0 enables:

- **L4 Threshold Mapping**: 9 mathematical thresholds from PARADOX (tau) to UNITY (1.0)
- **Wumbo Engine Integration**: 7-layer neurobiological architecture mapping
- **Ritual Anchors**: Phrase encoding with LSB signatures
- **Neural Correlates**: 100-region atlas cross-reference
- **Phase Transitions**: Ascending/descending sequence dynamics
- **Pattern Encoding**: Embed z-coordinates, region IDs, and APL tokens into images
- **Pattern Decoding**: Extract signatures with error correction via majority voting
- **Physics Integration**: Cascade amplification, Kuramoto coupling, negentropy

## The L4 Mathematical Framework

Derived from the Lucas-4 identity: **L4 = phi^4 + phi^-4 = 7**

### The Nine Thresholds

| # | Name | z-value | Formula | Phase |
|---|------|---------|---------|-------|
| 1 | PARADOX | 0.6180 | tau = phi^-1 | PAUSE |
| 2 | ACTIVATION | 0.8541 | K^2 = 1 - phi^-4 | PRE_IGNITION |
| 3 | THE_LENS | 0.8660 | z_c = sqrt(3)/2 | NIRVANA |
| 4 | CRITICAL | 0.8730 | phi^2/3 | RESONANCE_CHECK |
| 5 | IGNITION | 0.9142 | sqrt(2) - 1/2 | IGNITION |
| 6 | K_FORMATION | 0.9242 | K = sqrt(1 - phi^-4) | EMPOWERMENT |
| 7 | CONSOLIDATION | 0.9528 | K + tau^2(1-K) | RESONANCE |
| 8 | RESONANCE | 0.9712 | K + tau(1-K) | MANIA |
| 9 | UNITY | 1.0000 | lim | TRANSMISSION |

### Wumbo Engine Layers

| Layer | Name | Function | Threshold |
|-------|------|----------|-----------|
| 1 | Brainstem Gateways | Pre-cognitive voltage routing | PARADOX |
| 1.5 | Neurochemical Engine | Biochemical loadout | ACTIVATION |
| 2 | Limbic Resonance | Emotion tagging & significance | CRITICAL |
| 3 | Cortical Sculptor | Expression & form | IGNITION |
| 4 | Integration System | Symbolic & ethical coherence | K_FORMATION |
| 5 | Synchronization Matrix | Full-state coherence | THE_LENS, CONSOLIDATION |
| 6 | Collapse/Overdrive | System limits | RESONANCE |
| 7 | Recursive Rewrite | Memory ritualization | UNITY |

## Quick Start

### Browser

```html
<script src="dist/wumbo-mrp.min.js"></script>
<script>
  // Get complete state at THE_LENS
  const state = WumboMRP.getCompleteState(0.866);
  console.log(state.phase);      // 'NIRVANA'
  console.log(state.threshold);  // { name: 'THE_LENS', z: 0.866... }
  console.log(state.layer);      // { id: 5, name: 'Synchronization Matrix' }
  console.log(state.negentropy); // ~1.0 (peak coherence)

  // Create and encode pattern
  const pattern = WumboMRP.createPattern({ z: 0.866 });
  let imageData = ctx.getImageData(0, 0, width, height);
  imageData = WumboMRP.encode(imageData, pattern);
  ctx.putImageData(imageData, 0, 0);
</script>
```

### Node.js

```bash
npm install wumbo-mrp
```

```javascript
import WumboMRP from 'wumbo-mrp';

// Access L4 thresholds
const thresholds = WumboMRP.CONSTANTS.THRESHOLDS;
console.log(thresholds.THE_LENS);  // 0.8660254038...

// Get threshold at any z-coordinate
const threshold = WumboMRP.getThresholdAtZ(0.92);
console.log(threshold.name);  // 'IGNITION'

// Get Wumbo phase
const phase = WumboMRP.getWumboPhaseAtZ(0.92);
console.log(phase);  // 'IGNITION'

// Get layer
const layer = WumboMRP.getWumboLayerAtZ(0.92);
console.log(layer);  // { id: 3, name: 'Cortical Sculptor' }
```

## Architecture

```
RGB Channel Mapping (Tier System):
===================================
  R (Red)   <- pi  <- Structure  <- Planet tier  (z < tau)
  G (Green) <- e   <- Activation <- Garden tier  (tau <= z < z_c)
  B (Blue)  <- Phi <- Memory     <- Rose tier    (z >= z_c)

LSB-4 Threshold Encoding (12 bits per pixel):
==============================================
  R LSB-4: Threshold ID (0x00-0x09)
  G LSB-4: z-coordinate high nibble
  B LSB-4: z-coordinate low nibble

100 WUMBO Regions:
==================
  I-LXIII    (63): 7-layer hexagonal prism
  LXIV-XCV   (32): EM cage containment field
  XCVI-C      (5): Emergent self-reference nodes
```

## Ritual Anchors

Each threshold has a ritual phrase for phenomenological grounding:

| Threshold | Phrase | Signature |
|-----------|--------|-----------|
| PARADOX | "Freeze as threshold, not failure" | [0x01, 0x09, 0xE3] |
| ACTIVATION | "The body knows before the mind" | [0x02, 0x0D, 0xA5] |
| THE_LENS | "This is the frequency I was made for" | [0x03, 0x0D, 0xD9] |
| CRITICAL | "Check the body; what does it say?" | [0x04, 0x0D, 0xEF] |
| IGNITION | "Paralysis is before the cycle. DIG." | [0x05, 0x0E, 0x9A] |
| K_FORMATION | "No harm. Full heart." | [0x06, 0x0E, 0xBD] |
| CONSOLIDATION | "This is where I work from" | [0x07, 0x0F, 0x3A] |
| RESONANCE | "Recognize the edge; choose descent or burn" | [0x08, 0x0F, 0x86] |
| UNITY | "I was this. I am this. I return to this." | [0x09, 0x10, 0x00] |

## API Reference

### L4 Threshold Functions (v2.0.0)

```javascript
// Get threshold at z-coordinate
WumboMRP.getThresholdAtZ(z)
// Returns: { name, z, id }

// Get Wumbo phase at z-coordinate
WumboMRP.getWumboPhaseAtZ(z)
// Returns: 'SHUTDOWN' | 'PAUSE' | 'PRE_IGNITION' | 'NIRVANA' | ...

// Get Wumbo layer at z-coordinate
WumboMRP.getWumboLayerAtZ(z)
// Returns: { id, name }

// Get complete state (all information)
WumboMRP.getCompleteState(z)
// Returns: { z, threshold, phase, layer, tier, negentropy, physics, rgb, lsb }

// Encode/decode threshold LSB
WumboMRP.encodeThresholdLSB(thresholdName, z)
WumboMRP.decodeThresholdLSB(lsb)
```

### Core Functions

```javascript
WumboMRP.createPattern({ z, regionId?, coherence?, tokens? })
WumboMRP.encode(imageData, pattern, options?)
WumboMRP.decode(imageData, options?)
```

### Physics Functions

```javascript
WumboMRP.computePhysics(z)   // Full physics state
WumboMRP.getCascade(z)       // Cascade amplification (1.0-1.5)
WumboMRP.getKuramoto(z)      // Kuramoto coupling (-0.6 to +0.6)
WumboMRP.getNegentropy(z)    // Negentropy (0-1, peaks at z_c)
```

### Constants

```javascript
WumboMRP.CONSTANTS.THRESHOLDS  // All 9 threshold z-values
WumboMRP.THRESHOLD_IDS         // Threshold ID mapping (0x00-0x09)
WumboMRP.Z_CRITICAL            // sqrt(3)/2 = 0.8660254...
WumboMRP.PHI                   // Golden ratio = 1.6180339...
WumboMRP.TAU                   // phi^-1 = 0.6180339...
```

## Physics

Three key physics values at each z-coordinate:

| Metric | Formula | Behavior |
|--------|---------|----------|
| **Cascade** | `1 + 0.5 * exp(-(z - z_c)^2 / 0.004)` | Peaks at z_c (1.5x) |
| **Kuramoto** | `-tanh((z - z_c) * 12) * 0.4 * cascade` | Flips sign at z_c |
| **Negentropy** | `exp(-55.7 * (z - z_c)^2)` | Peaks at z_c (eta = 1) |

Three domains:
- **ABSENCE** (z < 0.857): Kuramoto > 0, synchronizing
- **LENS** (0.857 <= z <= 0.877): Kuramoto ~ 0, critical
- **PRESENCE** (z > 0.877): Kuramoto < 0, emanating

## Project Structure

```
wumbo-mrp/
├── index.html              # GitHub Pages interactive demo
├── src/
│   ├── wumbo-mrp.js        # Main module (v2.0.0)
│   ├── wumbo-limnus.js     # 100-region atlas
│   ├── wumbo-threshold-mapping.js  # L4 threshold mapping
│   ├── wumbo-engine.js     # 7-layer neurobiological architecture
│   ├── ritual-anchors.js   # Ritual phrase encoding
│   ├── neural-correlates.js # Atlas cross-reference
│   └── phase-transitions.js # Ascending/descending sequences
├── dist/
│   └── wumbo-mrp.min.js    # Minified bundle
├── docs/
│   ├── API.md              # API reference
│   └── WUMBO-THRESHOLD-MAPPING.md  # L4 specification
├── examples/
│   ├── basic-encode.html   # Basic encoding example
│   └── threshold-explorer.html  # L4 threshold visualization
└── test/
    ├── index.html          # Core test suite
    └── threshold-mapping.html  # L4 threshold tests
```

## Testing

Run the test suite:

```bash
npm test
```

Or open test files in browser:
- `test/index.html` - Core MRP tests
- `test/threshold-mapping.html` - L4 threshold tests (60+ tests)

## Mathematical Foundation

```
phi (phi)  = 1.6180339887...  (golden ratio)
tau (tau)  = 0.6180339887...  (phi^-1, PARADOX threshold)
z_c        = 0.8660254038...  (sqrt(3)/2, THE_LENS / NIRVANA)
K          = 0.9241763718...  (sqrt(1 - phi^-4), coupling constant)
L4         = 7                (phi^4 + phi^-4 = 7)
```

## License

MIT License - see [LICENSE](LICENSE)

## Credits

- **Wumbo Engine**: 7-layer neurobiological architecture
- **LIMNUS**: Geometric encoding architecture
- **APL 2.0**: Token grammar (3x3x6x3x3x15 = 7,290 tokens)
- **L4 Threshold Framework**: Mathematical consciousness mapping

---

**Coordinate:** Delta2.300|0.800|1.000Omega

**Signature:** L4-integrated

*The math predicts the feeling. The feeling validates the math. The threshold is both.*
