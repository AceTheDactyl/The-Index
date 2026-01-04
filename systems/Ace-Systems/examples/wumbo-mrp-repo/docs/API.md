# WUMBO MRP API Reference

## Table of Contents

- [Core Functions](#core-functions)
- [Physics Functions](#physics-functions)
- [Token Functions](#token-functions)
- [Encoding Functions](#encoding-functions)
- [Utility Functions](#utility-functions)
- [Constants](#constants)
- [Data Structures](#data-structures)

---

## Core Functions

### `WumboMRP.createPattern(params)`

Create an encoded pattern from state parameters.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `params.z` | `number` | Yes | Z-coordinate (0.0-1.0) |
| `params.regionId` | `number\|string` | No | Region ID (1-100) or Roman numeral |
| `params.coherence` | `number` | No | Coherence level (0.0-1.0), default 0.85 |
| `params.tokens` | `[string, string]` | No | Override APL tokens |

**Returns:** `EncodedPattern`

**Example:**
```javascript
const pattern = WumboMRP.createPattern({
  z: 0.866,
  regionId: 47,
  coherence: 0.9
});

console.log(pattern.regionRoman);  // "XLVII"
console.log(pattern.fieldBalance); // { r: 0.33, g: 0.34, b: 0.33 }
```

---

### `WumboMRP.encode(imageData, pattern, options?)`

Embed pattern into ImageData using LSB-4 steganography.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `imageData` | `ImageData` | Yes | Canvas ImageData object |
| `pattern` | `EncodedPattern` | Yes | Pattern from createPattern() |
| `options.startOffset` | `number` | No | Starting pixel index (default: 0) |
| `options.stride` | `number` | No | Embed every Nth pixel (default: 4) |
| `options.redundancy` | `number` | No | Repeat pattern N times (default: 8) |

**Returns:** `ImageData` (modified)

**Example:**
```javascript
const ctx = canvas.getContext('2d');
let imageData = ctx.getImageData(0, 0, 128, 128);

imageData = WumboMRP.encode(imageData, pattern, {
  stride: 4,
  redundancy: 32
});

ctx.putImageData(imageData, 0, 0);
```

---

### `WumboMRP.decode(imageData, options?)`

Extract pattern from ImageData with error correction.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `imageData` | `ImageData` | Yes | Canvas ImageData to decode |
| `options.startOffset` | `number` | No | Starting pixel index (default: 0) |
| `options.stride` | `number` | No | Stride used during encoding (default: 4) |
| `options.confidenceThreshold` | `number` | No | Minimum confidence (default: 0.7) |

**Returns:** `DecodedSignature`

**Example:**
```javascript
const decoded = WumboMRP.decode(imageData);

if (decoded.tokensValid) {
  console.log(`Region: ${decoded.regionRoman}`);
  console.log(`Z: ${decoded.z.toFixed(4)}`);
  console.log(`Confidence: ${(decoded.confidence * 100).toFixed(1)}%`);
}
```

---

## Physics Functions

### `WumboMRP.computePhysics(z)`

Compute full physics state at a z-coordinate.

**Returns:** `PhysicsState`
```javascript
{
  z: number,
  domain: 'ABSENCE' | 'LENS' | 'PRESENCE',
  cascade: number,      // 1.0 - 1.5
  kuramoto: number,     // -0.6 to +0.6
  negentropy: number,   // 0.0 - 1.0
  truthBias: 'TRUE' | 'UNTRUE' | 'PARADOX',
  machineAffinity: 'D' | 'M' | 'U'
}
```

---

### `WumboMRP.getCascade(z)`

Get cascade amplification factor.

- **Formula:** `1 + 0.5 × exp(-(z - z_c)² / 0.004)`
- **Range:** 1.0 to 1.5
- **Peaks at:** z_c = √3/2

---

### `WumboMRP.getKuramoto(z)`

Get Kuramoto coupling coefficient.

- **Formula:** `-tanh((z - z_c) × 12) × 0.4 × cascade`
- **Range:** -0.6 to +0.6
- **Sign flip at:** z_c = √3/2

---

### `WumboMRP.getNegentropy(z)`

Get negentropy (order measure).

- **Formula:** `exp(-55.7 × (z - z_c)²)`
- **Range:** 0.0 to 1.0
- **Peaks at:** z_c = √3/2

---

## Token Functions

### `WumboMRP.encodeAPLToken(token)`

Encode APL token string to 15-bit integer.

**Format:** `SPIRAL:OPERATOR:INTERACTION:DOMAIN:TRUTH:ALPHA`

**Example:**
```javascript
const encoded = WumboMRP.encodeAPLToken('e:U:A:CHEM:T:α10');
// Returns: 585 (0x0249)
```

**Bit Layout:**
| Bits | Component | Values |
|------|-----------|--------|
| 14-13 | Spiral | e=00, Φ=01, π=10 |
| 12-11 | Operator | U=00, M=01, C=10 |
| 10-8 | Interaction | B=000, F=001, A=010, D=011, G=100, S=101 |
| 7-6 | Domain | GEO=00, CHEM=01, BIO=10 |
| 5-4 | Truth | T=00, U=01, P=10 |
| 3-0 | Alpha | 0-14 (α1-α15) |

---

### `WumboMRP.decodeAPLToken(encoded)`

Decode 15-bit integer to APL token object.

**Returns:**
```javascript
{
  raw: string,           // Full token string
  spiral: 'e' | 'Φ' | 'π',
  operator: 'U' | 'M' | 'C',
  interaction: 'B' | 'F' | 'A' | 'D' | 'G' | 'S',
  domain: 'GEO' | 'CHEM' | 'BIO',
  truth: 'T' | 'U' | 'P',
  alpha: number,
  encoded: number
}
```

---

### `WumboMRP.validateToken(token)`

Validate APL token string format.

**Returns:** `{ valid: boolean, errors: string[] }`

---

## Encoding Functions

### `WumboMRP.encodeRegionId(regionId)`

Encode region ID (1-100) to LSB nibbles with parity.

**Returns:** `{ primary: number, secondary: number }`

---

### `WumboMRP.decodeRegionId(primary, secondary)`

Decode region ID from LSB nibbles.

**Returns:** `{ regionId: number, parityValid: boolean }`

---

### `WumboMRP.encodeZ(z)`

Encode z-coordinate to LSB nibbles.

**Returns:** `{ primary: number, secondary: number }`

---

### `WumboMRP.decodeZ(primary, secondary)`

Decode z-coordinate from LSB nibbles.

**Returns:** `number` (0.0-1.0)

---

### `WumboMRP.generateChecksum(token1, token2)`

Generate 4-bit checksum from two encoded tokens.

**Returns:** `number` (0-15)

---

## Utility Functions

### `WumboMRP.computeRGBWeightsFromZ(z)`

Compute RGB channel weights from z-coordinate.

**Returns:** `{ r: number, g: number, b: number }` (sums to 1.0)

```javascript
// At z_c (critical point):
// { r: 0.33, g: 0.34, b: 0.33 }  // Balanced

// Below z_c (ABSENCE):
// { r: 0.6, g: 0.25, b: 0.15 }   // π dominant

// Above z_c (PRESENCE):
// { r: 0.15, g: 0.45, b: 0.40 }  // e/Φ dominant
```

---

### `WumboMRP.generateCarrierImage(width, height, fieldBalance)`

Generate carrier image canvas with field-balanced gradient.

**Returns:** `HTMLCanvasElement`

---

### `WumboMRP.generateEncodedCarrier(params, options?)`

Generate complete encoded carrier image.

**Returns:**
```javascript
{
  canvas: HTMLCanvasElement,
  dataUrl: string,
  pattern: EncodedPattern
}
```

---

### `WumboMRP.verifyRoundtrip(params, options?)`

Verify encode/decode roundtrip integrity.

**Returns:**
```javascript
{
  success: boolean,
  original: EncodedPattern,
  decoded: DecodedSignature,
  errors: {
    zError: number,
    regionMatch: boolean,
    tokensMatch: boolean,
    confidence: number
  }
}
```

---

### `WumboMRP.numericToRoman(num)`

Convert number to Roman numeral.

```javascript
WumboMRP.numericToRoman(47)  // "XLVII"
WumboMRP.numericToRoman(100) // "C"
```

---

### `WumboMRP.romanToNumeric(roman)`

Convert Roman numeral to number.

```javascript
WumboMRP.romanToNumeric('XLVII') // 47
WumboMRP.romanToNumeric('C')     // 100
```

---

## Constants

### `WumboMRP.CONSTANTS`

```javascript
{
  Z_CRITICAL: 0.8660254037844387,    // √3/2
  PHI: 1.6180339887498949,           // (1 + √5) / 2
  TAU: 0.6180339887498949,           // φ⁻¹
  PHI_INV_4: 0.1458980337503154,     // φ⁻⁴
  L4: 7,                              // φ⁴ + φ⁻⁴
  K_THRESHOLD: 0.9241763718,          // √(1 - φ⁻⁴)
  IGNITION: 0.9142135623730951,       // √2 - ½
  CASCADE_SIGMA: 0.004,
  KURAMOTO_SCALE: 12,
  NEGENTROPY_SIGMA: 55.71281292
}
```

---

### `WumboMRP.APL_MAPS`

APL grammar encoding/decoding maps.

---

### `WumboMRP.LAYER_Z_MAP`

Z-coordinate to layer mapping.

---

## Data Structures

### EncodedPattern

```typescript
interface EncodedPattern {
  regionId: number;
  regionRoman: string;
  z: number;
  coherence: number;
  fieldBalance: { r: number; g: number; b: number };
  tokens: [number, number];
  rawTokens: [string, string];
  timestamp: number;
  checksum: number;
  version: number;
}
```

### DecodedSignature

```typescript
interface DecodedSignature {
  regionId: number;
  regionRoman: string;
  z: number;
  tokensValid: boolean;
  tokens: [APLToken, APLToken];
  confidence: number;
  sampleCount: number;
  physics: PhysicsState;
  errors: string[];
}
```

### PhysicsState

```typescript
interface PhysicsState {
  z: number;
  domain: 'ABSENCE' | 'LENS' | 'PRESENCE';
  cascade: number;
  kuramoto: number;
  negentropy: number;
  truthBias: 'TRUE' | 'UNTRUE' | 'PARADOX';
  machineAffinity: 'D' | 'M' | 'U';
}
```

---

**Coordinate:** Δ2.300|0.800|1.000Ω
