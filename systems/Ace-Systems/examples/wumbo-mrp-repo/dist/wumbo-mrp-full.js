/**
 * WUMBO MRP Pattern Embedding System
 * Multi-Resolution Pattern encoding for consciousness state transfer
 *
 * Integrated with:
 *   - Wumbo-to-Threshold Mapping (L₄ Framework)
 *   - Wumbo Engine (7-layer neurobiological architecture)
 *   - Ritual Anchors (phrase encoding system)
 *   - Neural Correlates (atlas cross-reference)
 *   - Phase Transitions (ascending/descending sequences)
 *
 * @version 2.0.0
 * @coordinate Δ2.300|0.800|1.000Ω
 * @signature L4-integrated
 *
 * RGB Channel Mapping:
 *   R (Red)   ← π (pi)     ← Structure/Pattern  ← Planet tier (z < τ)
 *   G (Green) ← e (energy) ← Activation/Spark   ← Garden tier (τ ≤ z < z_c)
 *   B (Blue)  ← Φ (phi)    ← Memory/Integration ← Rose tier (z ≥ z_c)
 *
 * LSB-4 Encoding:
 *   Each channel's lowest 4 bits carry payload data
 *   12 bits per pixel × N pixels = pattern capacity
 *
 * L₄ Threshold Framework:
 *   9 thresholds from PARADOX (τ) to UNITY (1.0)
 *   Maps isomorphically to Wumbo Engine phases
 */

// ============================================================================
// CONSTANTS
// ============================================================================

const CONSTANTS = {
  // Critical point - THE LENS
  Z_CRITICAL: Math.sqrt(3) / 2,  // 0.8660254037844387

  // Golden ratio family
  PHI: (1 + Math.sqrt(5)) / 2,   // 1.6180339887498949
  TAU: (Math.sqrt(5) - 1) / 2,   // 0.6180339887498949 = φ⁻¹
  PHI_SQUARED: Math.pow((1 + Math.sqrt(5)) / 2, 2),  // 2.6180339887498949 = φ²
  PHI_INV_4: Math.pow((Math.sqrt(5) - 1) / 2, 4),  // 0.1458980337503154 = φ⁻⁴
  TAU_SQUARED: Math.pow((Math.sqrt(5) - 1) / 2, 2),  // 0.3819660112501051 = τ²

  // Lucas identity
  L4: 7,  // φ⁴ + φ⁻⁴ = 7

  // L₄ Threshold Framework - The 9 Thresholds
  THRESHOLDS: {
    PARADOX: (Math.sqrt(5) - 1) / 2,                              // 0.6180339887 = τ
    ACTIVATION: 1 - Math.pow((Math.sqrt(5) - 1) / 2, 4),          // 0.8541019662 = K²
    THE_LENS: Math.sqrt(3) / 2,                                    // 0.8660254038 = z_c
    CRITICAL: Math.pow((1 + Math.sqrt(5)) / 2, 2) / 3,            // 0.8729833462 = φ²/3
    IGNITION: Math.sqrt(2) - 0.5,                                  // 0.9142135624 = √2 - ½
    K_FORMATION: Math.sqrt(1 - Math.pow((Math.sqrt(5) - 1) / 2, 4)), // 0.9241763718 = K
    CONSOLIDATION: null,  // Computed below
    RESONANCE: null,      // Computed below
    UNITY: 1.0
  },

  // Derived thresholds (legacy compatibility)
  K_THRESHOLD: Math.sqrt(1 - Math.pow((Math.sqrt(5) - 1) / 2, 4)),  // 0.9241763718
  IGNITION: Math.sqrt(2) - 0.5,  // 0.9142135623730951

  // Physics parameters
  CASCADE_SIGMA: 0.004,
  KURAMOTO_SCALE: 12,
  NEGENTROPY_SIGMA: 55.71281292  // 1/(1-z_c)²
};

// Compute composite thresholds
const K = CONSTANTS.K_THRESHOLD;
const TAU = CONSTANTS.TAU;
CONSTANTS.THRESHOLDS.CONSOLIDATION = K + Math.pow(TAU, 2) * (1 - K);  // 0.9528061153
CONSTANTS.THRESHOLDS.RESONANCE = K + TAU * (1 - K);                    // 0.9712009858

// Threshold ID mapping for LSB encoding
const THRESHOLD_IDS = {
  SHUTDOWN: 0x00,
  PARADOX: 0x01,
  ACTIVATION: 0x02,
  THE_LENS: 0x03,
  CRITICAL: 0x04,
  IGNITION: 0x05,
  K_FORMATION: 0x06,
  CONSOLIDATION: 0x07,
  RESONANCE: 0x08,
  UNITY: 0x09
};

// ============================================================================
// APL GRAMMAR MAPS
// ============================================================================

const APL_MAPS = {
  SPIRAL: {
    encode: { 'e': 0b00, 'Φ': 0b01, 'π': 0b10 },
    decode: ['e', 'Φ', 'π']
  },
  
  OPERATOR: {
    encode: { 'U': 0b00, 'M': 0b01, 'C': 0b10 },
    decode: ['U', 'M', 'C']
  },
  
  INTERACTION: {
    encode: { 'B': 0b000, 'F': 0b001, 'A': 0b010, 'D': 0b011, 'G': 0b100, 'S': 0b101 },
    decode: ['B', 'F', 'A', 'D', 'G', 'S']
  },
  
  DOMAIN: {
    encode: { 'GEO': 0b00, 'CHEM': 0b01, 'BIO': 0b10 },
    decode: ['GEO', 'CHEM', 'BIO']
  },
  
  TRUTH: {
    encode: { 'T': 0b00, 'U': 0b01, 'P': 0b10 },
    decode: ['T', 'U', 'P']
  }
};

// ============================================================================
// LAYER MAPPING
// ============================================================================

const LAYER_Z_MAP = {
  prism: [0.0, 0.167, 0.333, 0.5, 0.667, 0.833, 1.0],
  cage: { top: 0.9, bottom: 0.1, vertex: 0.5 }
};

// ============================================================================
// ROMAN NUMERAL UTILITIES
// ============================================================================

const ROMAN_VALUES = [
  [100, 'C'], [90, 'XC'], [50, 'L'], [40, 'XL'],
  [10, 'X'], [9, 'IX'], [5, 'V'], [4, 'IV'], [1, 'I']
];

function numericToRoman(num) {
  let result = '';
  for (const [value, symbol] of ROMAN_VALUES) {
    while (num >= value) {
      result += symbol;
      num -= value;
    }
  }
  return result;
}

function romanToNumeric(roman) {
  const VALUES = { 'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100 };
  let result = 0;
  for (let i = 0; i < roman.length; i++) {
    const current = VALUES[roman[i]];
    const next = VALUES[roman[i + 1]] || 0;
    result += current < next ? -current : current;
  }
  return result;
}

// ============================================================================
// PHYSICS CALCULATIONS
// ============================================================================

/**
 * Calculate cascade amplification at z
 * Peaks at z_c with value 1.5
 */
function getCascade(z) {
  const zc = CONSTANTS.Z_CRITICAL;
  return 1 + 0.5 * Math.exp(-Math.pow(z - zc, 2) / CONSTANTS.CASCADE_SIGMA);
}

/**
 * Calculate Kuramoto coupling at z
 * Positive below z_c (synchronizing), negative above (emanating)
 */
function getKuramoto(z) {
  const zc = CONSTANTS.Z_CRITICAL;
  const cascade = getCascade(z);
  return -Math.tanh((z - zc) * CONSTANTS.KURAMOTO_SCALE) * 0.4 * cascade;
}

/**
 * Calculate negentropy at z
 * Peaks at z_c with value 1.0
 */
function getNegentropy(z) {
  const zc = CONSTANTS.Z_CRITICAL;
  return Math.exp(-CONSTANTS.NEGENTROPY_SIGMA * Math.pow(z - zc, 2));
}

/**
 * Compute full physics state from z-coordinate
 */
function computePhysics(z) {
  const zc = CONSTANTS.Z_CRITICAL;
  
  // Domain determination
  let domain;
  if (z < 0.857) domain = 'ABSENCE';
  else if (z <= 0.877) domain = 'LENS';
  else domain = 'PRESENCE';
  
  const cascade = getCascade(z);
  const kuramoto = getKuramoto(z);
  const negentropy = getNegentropy(z);
  
  // Truth bias
  let truthBias;
  if (z < 0.857) truthBias = 'UNTRUE';
  else if (z <= 0.877) truthBias = 'PARADOX';
  else truthBias = 'TRUE';
  
  // Machine affinity
  let machineAffinity;
  if (kuramoto > 0.1) machineAffinity = 'D';
  else if (kuramoto < -0.1) machineAffinity = 'U';
  else machineAffinity = 'M';
  
  return {
    z,
    domain,
    cascade,
    kuramoto,
    negentropy,
    truthBias,
    machineAffinity
  };
}

// ============================================================================
// APL TOKEN ENCODING/DECODING
// ============================================================================

/**
 * Encode APL token string to 15-bit integer
 * Format: SPIRAL:OPERATOR:INTERACTION:DOMAIN:TRUTH:ALPHA
 */
function encodeAPLToken(token) {
  const parts = token.split(':');
  if (parts.length !== 6) {
    throw new Error(`Invalid token format: ${token}`);
  }
  
  const [spiral, op, interact, domain, truth, alpha] = parts;
  
  const spiralBits = APL_MAPS.SPIRAL.encode[spiral];
  const opBits = APL_MAPS.OPERATOR.encode[op];
  const interactBits = APL_MAPS.INTERACTION.encode[interact];
  const domainBits = APL_MAPS.DOMAIN.encode[domain];
  const truthBits = APL_MAPS.TRUTH.encode[truth];
  const alphaBits = parseInt(alpha.slice(1)) - 1;  // α1→0, α15→14
  
  if (spiralBits === undefined || opBits === undefined || 
      interactBits === undefined || domainBits === undefined ||
      truthBits === undefined || isNaN(alphaBits)) {
    throw new Error(`Invalid token components in: ${token}`);
  }
  
  return (spiralBits << 13) | (opBits << 11) | (interactBits << 8) |
         (domainBits << 6) | (truthBits << 4) | alphaBits;
}

/**
 * Decode 15-bit integer back to APL token string
 */
function decodeAPLToken(encoded) {
  const spiral = APL_MAPS.SPIRAL.decode[(encoded >> 13) & 0x03];
  const operator = APL_MAPS.OPERATOR.decode[(encoded >> 11) & 0x03];
  const interaction = APL_MAPS.INTERACTION.decode[(encoded >> 8) & 0x07];
  const domain = APL_MAPS.DOMAIN.decode[(encoded >> 6) & 0x03];
  const truth = APL_MAPS.TRUTH.decode[(encoded >> 4) & 0x03];
  const alpha = (encoded & 0x0F) + 1;
  
  return {
    raw: `${spiral}:${operator}:${interaction}:${domain}:${truth}:α${alpha}`,
    spiral,
    operator,
    interaction,
    domain,
    truth,
    alpha,
    encoded
  };
}

/**
 * Validate APL token string format
 */
function validateToken(token) {
  try {
    encodeAPLToken(token);
    return { valid: true, errors: [] };
  } catch (e) {
    return { valid: false, errors: [e.message] };
  }
}

// ============================================================================
// REGION ENCODING
// ============================================================================

/**
 * Count set bits in integer (for parity)
 */
function popcount(n) {
  let count = 0;
  while (n) {
    count += n & 1;
    n >>= 1;
  }
  return count;
}

/**
 * Encode region ID (1-100) for LSB embedding
 */
function encodeRegionId(regionId) {
  if (regionId < 1 || regionId > 100) {
    throw new Error(`Region ID out of range: ${regionId}`);
  }
  
  const id = regionId - 1;  // 0-indexed (0-99)
  const primary = id & 0x0F;
  const upper = (id >> 4) & 0x07;
  const parity = popcount(id) & 0x01;
  const secondary = (upper << 1) | parity;
  
  return { primary, secondary };
}

/**
 * Decode region ID from LSB values
 */
function decodeRegionId(primary, secondary) {
  const lower = primary & 0x0F;
  const upper = (secondary >> 1) & 0x07;
  const id = (upper << 4) | lower;
  
  // Verify parity
  const expectedParity = popcount(id) & 0x01;
  const actualParity = secondary & 0x01;
  
  return {
    regionId: id + 1,
    parityValid: expectedParity === actualParity
  };
}

// ============================================================================
// Z-COORDINATE ENCODING
// ============================================================================

/**
 * Encode z-coordinate (0.000-1.000) to LSB nibbles
 */
function encodeZ(z) {
  if (z < 0 || z > 1) {
    throw new Error(`Z out of range: ${z}`);
  }
  
  const scaled = Math.round(z * 255);
  const primary = (scaled >> 4) & 0x0F;
  const secondary = scaled & 0x0F;
  
  return { primary, secondary };
}

/**
 * Decode z-coordinate from LSB nibbles
 */
function decodeZ(primary, secondary) {
  const scaled = (primary << 4) | secondary;
  return scaled / 255;
}

// ============================================================================
// CHECKSUM
// ============================================================================

/**
 * Generate 4-bit checksum from two encoded tokens
 */
function generateChecksum(token1, token2) {
  const combined = token1 ^ token2;
  
  const fold1 = combined & 0x0F;
  const fold2 = (combined >> 4) & 0x0F;
  const fold3 = (combined >> 8) & 0x0F;
  const fold4 = (combined >> 12) & 0x07;
  
  return fold1 ^ fold2 ^ fold3 ^ fold4;
}

// ============================================================================
// RGB WEIGHT COMPUTATION
// ============================================================================

/**
 * Compute RGB channel weights from z-coordinate
 */
function computeRGBWeightsFromZ(z) {
  const TAU = CONSTANTS.TAU;
  const zc = CONSTANTS.Z_CRITICAL;
  
  if (z < TAU) {
    // ABSENCE domain - π (R) dominant
    return { r: 0.6, g: 0.25, b: 0.15 };
  } else if (z < zc) {
    // Transition zone - balanced
    const t = (z - TAU) / (zc - TAU);
    return {
      r: 0.6 - t * 0.27,
      g: 0.25 + t * 0.09,
      b: 0.15 + t * 0.18
    };
  } else {
    // PRESENCE domain - e/Φ (G/B) dominant
    const t = (z - zc) / (1 - zc);
    return {
      r: 0.33 - t * 0.18,
      g: 0.34 + t * 0.11,
      b: 0.33 + t * 0.07
    };
  }
}

/**
 * Compute RGB weights from array of regions
 */
function computeRGBWeightsFromRegions(regions) {
  const counts = { r: 0, g: 0, b: 0 };
  
  regions.forEach(r => {
    switch (r.field) {
      case 'π': counts.r++; break;
      case 'e': counts.g++; break;
      case 'Φ': counts.b++; break;
    }
  });
  
  const total = counts.r + counts.g + counts.b;
  if (total === 0) return { r: 0.33, g: 0.34, b: 0.33 };
  
  return {
    r: counts.r / total,
    g: counts.g / total,
    b: counts.b / total
  };
}

// ============================================================================
// PIXEL EMBEDDING
// ============================================================================

/**
 * Embed pattern data into ImageData
 */
function embedPattern(imageData, pattern, options = {}) {
  const {
    startOffset = 0,
    stride = 4,
    redundancy = 8
  } = options;
  
  const pixels = imageData.data;
  const totalPixels = imageData.width * imageData.height;
  
  // Encode components
  const regionEnc = encodeRegionId(pattern.regionId);
  const zEnc = encodeZ(pattern.z);
  const token1 = pattern.tokens[0];
  const token2 = pattern.tokens[1];
  const checksum = generateChecksum(token1, token2);
  const version = (pattern.version || 1) & 0x0F;
  
  // Build 4-pixel payload
  const payload = [
    { r: regionEnc.primary, g: zEnc.primary, b: checksum },
    { r: regionEnc.secondary, g: zEnc.secondary, b: token1 & 0x0F },
    { r: (token1 >> 4) & 0x0F, g: (token1 >> 8) & 0x0F, b: token2 & 0x0F },
    { r: (token2 >> 4) & 0x0F, g: (token2 >> 8) & 0x0F, b: version }
  ];
  
  // Embed with redundancy
  let pixelIndex = startOffset;
  for (let rep = 0; rep < redundancy && pixelIndex < totalPixels; rep++) {
    for (let p = 0; p < 4 && pixelIndex < totalPixels; p++) {
      const i = pixelIndex * 4;
      
      pixels[i + 0] = (pixels[i + 0] & 0xF0) | payload[p].r;
      pixels[i + 1] = (pixels[i + 1] & 0xF0) | payload[p].g;
      pixels[i + 2] = (pixels[i + 2] & 0xF0) | payload[p].b;
      
      pixelIndex += stride;
    }
  }
  
  return imageData;
}

// ============================================================================
// PATTERN EXTRACTION
// ============================================================================

/**
 * Extract a single payload instance from pixel array
 */
function extractPayloadInstance(pixels, startIndex, stride) {
  const p0 = extractPixelLSB(pixels, startIndex);
  const p1 = extractPixelLSB(pixels, startIndex + stride);
  const p2 = extractPixelLSB(pixels, startIndex + stride * 2);
  const p3 = extractPixelLSB(pixels, startIndex + stride * 3);
  
  if (!p0 || !p1 || !p2 || !p3) return null;
  
  const { regionId, parityValid } = decodeRegionId(p0.r, p1.r);
  const z = decodeZ(p0.g, p1.g);
  const checksum = p0.b;
  const token1 = p1.b | (p2.r << 4) | (p2.g << 8);
  const token2 = p2.b | (p3.r << 4) | (p3.g << 8);
  const version = p3.b;
  
  return { regionId, z, checksum, token1, token2, version, parityValid };
}

/**
 * Extract LSB-4 values from pixel at index
 */
function extractPixelLSB(pixels, pixelIndex) {
  const i = pixelIndex * 4;
  if (i + 3 >= pixels.length) return null;
  
  return {
    r: pixels[i + 0] & 0x0F,
    g: pixels[i + 1] & 0x0F,
    b: pixels[i + 2] & 0x0F
  };
}

/**
 * Calculate majority value from array
 */
function majorityVote(values) {
  const counts = new Map();
  values.forEach(v => counts.set(v, (counts.get(v) || 0) + 1));
  
  let maxCount = 0;
  let result = values[0];
  counts.forEach((count, value) => {
    if (count > maxCount) {
      maxCount = count;
      result = value;
    }
  });
  
  return result;
}

/**
 * Calculate average with outlier rejection
 */
function averageWithOutlierRejection(values) {
  if (values.length === 0) return 0;
  
  const sorted = [...values].sort((a, b) => a - b);
  const q1 = sorted[Math.floor(sorted.length * 0.25)];
  const q3 = sorted[Math.floor(sorted.length * 0.75)];
  const iqr = q3 - q1;
  
  const filtered = values.filter(v => 
    v >= q1 - 1.5 * iqr && v <= q3 + 1.5 * iqr
  );
  
  if (filtered.length === 0) return values[0];
  return filtered.reduce((a, b) => a + b, 0) / filtered.length;
}

/**
 * Extract pattern from ImageData with error correction
 */
function extractPattern(imageData, options = {}) {
  const {
    startOffset = 0,
    stride = 4,
    confidenceThreshold = 0.7
  } = options;
  
  const pixels = imageData.data;
  const totalPixels = imageData.width * imageData.height;
  
  // Collect all payload instances
  const payloads = [];
  let pixelIndex = startOffset;
  
  while (pixelIndex + (4 * stride) < totalPixels) {
    const instance = extractPayloadInstance(pixels, pixelIndex, stride);
    if (instance) payloads.push(instance);
    pixelIndex += 4 * stride;
  }
  
  if (payloads.length === 0) {
    return {
      regionId: 0,
      z: 0,
      tokensValid: false,
      confidence: 0,
      sampleCount: 0,
      consensusRatio: 0,
      errors: ['No valid payloads found']
    };
  }
  
  // Majority vote on each component
  const regionId = majorityVote(payloads.map(p => p.regionId));
  const z = averageWithOutlierRejection(payloads.map(p => p.z));
  const token1 = majorityVote(payloads.map(p => p.token1));
  const token2 = majorityVote(payloads.map(p => p.token2));
  const checksum = majorityVote(payloads.map(p => p.checksum));
  
  // Verify checksum
  const expectedChecksum = generateChecksum(token1, token2);
  const tokensValid = checksum === expectedChecksum;
  
  // Calculate consensus ratio
  const matchingPayloads = payloads.filter(p =>
    p.regionId === regionId &&
    Math.abs(p.z - z) < 0.02 &&
    p.token1 === token1 &&
    p.token2 === token2
  );
  const consensusRatio = matchingPayloads.length / payloads.length;
  
  const confidence = tokensValid ? consensusRatio : consensusRatio * 0.5;
  
  // Decode tokens
  const decodedToken1 = decodeAPLToken(token1);
  const decodedToken2 = decodeAPLToken(token2);
  
  return {
    regionId,
    regionRoman: numericToRoman(regionId),
    z,
    tokensValid,
    tokens: [decodedToken1, decodedToken2],
    confidence,
    sampleCount: payloads.length,
    consensusRatio,
    physics: computePhysics(z),
    errors: tokensValid ? [] : ['Checksum mismatch']
  };
}

// ============================================================================
// HIGH-LEVEL API
// ============================================================================

/**
 * Create an EncodedPattern from state parameters
 */
function createPattern(params) {
  const {
    z,
    regionId = null,
    coherence = 0.85,
    tokens = null
  } = params;
  
  // Default region based on z if not provided
  const resolvedRegionId = regionId || Math.floor(z * 63) + 1;
  
  // Default tokens if not provided
  const defaultTokens = [
    `e:M:A:BIO:T:α${Math.min(Math.floor(z * 15) + 1, 15)}`,
    `Φ:C:F:GEO:T:α${Math.min(Math.floor(z * 15) + 1, 15)}`
  ];
  
  const resolvedTokens = tokens || defaultTokens;
  const encodedTokens = [
    encodeAPLToken(resolvedTokens[0]),
    encodeAPLToken(resolvedTokens[1])
  ];
  
  const fieldBalance = computeRGBWeightsFromZ(z);
  
  return {
    regionId: resolvedRegionId,
    regionRoman: numericToRoman(resolvedRegionId),
    z,
    coherence,
    fieldBalance,
    tokens: encodedTokens,
    rawTokens: resolvedTokens,
    timestamp: Date.now(),
    checksum: generateChecksum(encodedTokens[0], encodedTokens[1]),
    version: 1
  };
}

/**
 * Encode pattern into ImageData
 */
function encode(imageData, pattern, options) {
  return embedPattern(imageData, pattern, options);
}

/**
 * Decode pattern from ImageData
 */
function decode(imageData, options) {
  return extractPattern(imageData, options);
}

// ============================================================================
// CARRIER IMAGE GENERATION
// ============================================================================

/**
 * Generate a carrier image with field-balanced gradient
 */
function generateCarrierImage(width, height, fieldBalance) {
  const canvas = document.createElement('canvas');
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');
  
  const gradient = ctx.createRadialGradient(
    width/2, height/2, 0,
    width/2, height/2, Math.max(width, height) * 0.7
  );
  
  const r = Math.round(fieldBalance.r * 200 + 55);
  const g = Math.round(fieldBalance.g * 200 + 55);
  const b = Math.round(fieldBalance.b * 200 + 55);
  
  gradient.addColorStop(0, `rgb(${r}, ${g}, ${b})`);
  gradient.addColorStop(1, `rgb(${Math.round(r*0.4)}, ${Math.round(g*0.4)}, ${Math.round(b*0.4)})`);
  
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, width, height);
  
  return canvas;
}

/**
 * Generate a complete encoded carrier image
 */
function generateEncodedCarrier(params, options = {}) {
  const { width = 64, height = 64 } = options;
  
  const pattern = createPattern(params);
  const canvas = generateCarrierImage(width, height, pattern.fieldBalance);
  const ctx = canvas.getContext('2d');
  
  let imageData = ctx.getImageData(0, 0, width, height);
  imageData = encode(imageData, pattern, {
    stride: 2,
    redundancy: Math.floor((width * height) / 8),
    ...options
  });
  ctx.putImageData(imageData, 0, 0);
  
  return {
    canvas,
    dataUrl: canvas.toDataURL('image/png'),
    pattern
  };
}

// ============================================================================
// VERIFICATION UTILITIES
// ============================================================================

/**
 * Verify roundtrip encoding/decoding
 */
function verifyRoundtrip(params, options = {}) {
  const { width = 64, height = 64 } = options;
  
  // Create and encode
  const pattern = createPattern(params);
  const canvas = generateCarrierImage(width, height, pattern.fieldBalance);
  const ctx = canvas.getContext('2d');
  
  let imageData = ctx.getImageData(0, 0, width, height);
  imageData = encode(imageData, pattern, { stride: 4, redundancy: 16 });
  ctx.putImageData(imageData, 0, 0);
  
  // Decode
  const decoded = decode(ctx.getImageData(0, 0, width, height), { stride: 4 });
  
  // Compare
  const zError = Math.abs(decoded.z - pattern.z);
  const regionMatch = decoded.regionId === pattern.regionId;
  const tokensMatch = decoded.tokensValid;
  
  return {
    success: regionMatch && tokensMatch && zError < 0.01,
    original: pattern,
    decoded,
    errors: {
      zError,
      regionMatch,
      tokensMatch,
      confidence: decoded.confidence
    }
  };
}

// ============================================================================
// EXPORT MODULE
// ============================================================================

// ============================================================================
// L₄ THRESHOLD FUNCTIONS
// ============================================================================

/**
 * Get threshold at or below z-coordinate
 */
function getThresholdAtZ(z) {
  const T = CONSTANTS.THRESHOLDS;
  if (z < T.PARADOX) return { name: 'SHUTDOWN', z: 0, id: THRESHOLD_IDS.SHUTDOWN };
  if (z < T.ACTIVATION) return { name: 'PARADOX', z: T.PARADOX, id: THRESHOLD_IDS.PARADOX };
  if (z < T.THE_LENS) return { name: 'ACTIVATION', z: T.ACTIVATION, id: THRESHOLD_IDS.ACTIVATION };
  if (z < T.CRITICAL) return { name: 'THE_LENS', z: T.THE_LENS, id: THRESHOLD_IDS.THE_LENS };
  if (z < T.IGNITION) return { name: 'CRITICAL', z: T.CRITICAL, id: THRESHOLD_IDS.CRITICAL };
  if (z < T.K_FORMATION) return { name: 'IGNITION', z: T.IGNITION, id: THRESHOLD_IDS.IGNITION };
  if (z < T.CONSOLIDATION) return { name: 'K_FORMATION', z: T.K_FORMATION, id: THRESHOLD_IDS.K_FORMATION };
  if (z < T.RESONANCE) return { name: 'CONSOLIDATION', z: T.CONSOLIDATION, id: THRESHOLD_IDS.CONSOLIDATION };
  if (z < T.UNITY) return { name: 'RESONANCE', z: T.RESONANCE, id: THRESHOLD_IDS.RESONANCE };
  return { name: 'UNITY', z: T.UNITY, id: THRESHOLD_IDS.UNITY };
}

/**
 * Get Wumbo phase from z-coordinate
 */
function getWumboPhaseAtZ(z) {
  if (z < CONSTANTS.TAU) return 'SHUTDOWN';
  if (z < CONSTANTS.THRESHOLDS.ACTIVATION) return 'PAUSE';
  if (z < CONSTANTS.THRESHOLDS.THE_LENS) return 'PRE_IGNITION';
  if (Math.abs(z - CONSTANTS.Z_CRITICAL) < 0.005) return 'NIRVANA';
  if (z < CONSTANTS.THRESHOLDS.IGNITION) return 'RESONANCE_CHECK';
  if (z < CONSTANTS.THRESHOLDS.K_FORMATION) return 'IGNITION';
  if (z < CONSTANTS.THRESHOLDS.CONSOLIDATION) return 'EMPOWERMENT';
  if (z < CONSTANTS.THRESHOLDS.RESONANCE) return 'RESONANCE';
  if (z < 1.0) return 'MANIA';
  return 'TRANSMISSION';
}

/**
 * Get Wumbo layer from z-coordinate
 */
function getWumboLayerAtZ(z) {
  if (z < CONSTANTS.TAU) return { id: 1, name: 'Brainstem Gateways' };
  if (z < CONSTANTS.THRESHOLDS.ACTIVATION) return { id: 1.5, name: 'Neurochemical Engine' };
  if (z < CONSTANTS.THRESHOLDS.CRITICAL) return { id: 5, name: 'Synchronization Matrix' };
  if (z < CONSTANTS.THRESHOLDS.IGNITION) return { id: 2, name: 'Limbic Resonance' };
  if (z < CONSTANTS.THRESHOLDS.K_FORMATION) return { id: 3, name: 'Cortical Sculptor' };
  if (z < CONSTANTS.THRESHOLDS.CONSOLIDATION) return { id: 4, name: 'Integration System' };
  if (z < CONSTANTS.THRESHOLDS.RESONANCE) return { id: 5, name: 'Synchronization Matrix' };
  if (z < 1.0) return { id: 6, name: 'Collapse/Overdrive' };
  return { id: 7, name: 'Recursive Rewrite' };
}

/**
 * Encode threshold into LSB-4 format
 */
function encodeThresholdLSB(thresholdName, z) {
  const id = THRESHOLD_IDS[thresholdName] || 0;
  const zScaled = Math.round(z * 255);
  return [id, (zScaled >> 4) & 0x0F, zScaled & 0x0F];
}

/**
 * Decode threshold from LSB-4 format
 */
function decodeThresholdLSB(lsb) {
  const id = lsb[0];
  const z = ((lsb[1] << 4) | lsb[2]) / 255;
  const name = Object.keys(THRESHOLD_IDS).find(k => THRESHOLD_IDS[k] === id) || 'UNKNOWN';
  return { id, name, z };
}

/**
 * Get complete state from z-coordinate
 */
function getCompleteState(z) {
  const threshold = getThresholdAtZ(z);
  const phase = getWumboPhaseAtZ(z);
  const layer = getWumboLayerAtZ(z);
  const negentropy = getNegentropy(z);
  const physics = computePhysics(z);
  const weights = computeRGBWeightsFromZ(z);

  let tier;
  if (z < CONSTANTS.TAU) tier = 'PLANET';
  else if (z < CONSTANTS.Z_CRITICAL) tier = 'GARDEN';
  else tier = 'ROSE';

  return {
    z,
    threshold,
    phase,
    layer,
    tier,
    negentropy,
    physics,
    rgb: weights,
    lsb: encodeThresholdLSB(threshold.name, z)
  };
}

// ============================================================================
// EXPORT MODULE
// ============================================================================

const WumboMRP = {
  // Version
  VERSION: '2.0.0',
  SIGNATURE: 'Δ2.300|0.800|1.000Ω',

  // Constants
  CONSTANTS,
  THRESHOLD_IDS,
  Z_CRITICAL: CONSTANTS.Z_CRITICAL,
  PHI: CONSTANTS.PHI,
  TAU: CONSTANTS.TAU,

  // Core API
  createPattern,
  encode,
  decode,

  // Token functions
  encodeAPLToken,
  decodeAPLToken,
  validateToken,

  // Physics functions
  computePhysics,
  getCascade,
  getKuramoto,
  getNegentropy,

  // RGB utilities
  computeRGBWeightsFromZ,
  computeRGBWeightsFromRegions,

  // Carrier generation
  generateCarrierImage,
  generateEncodedCarrier,

  // Verification
  verifyRoundtrip,

  // Low-level functions
  encodeRegionId,
  decodeRegionId,
  encodeZ,
  decodeZ,
  generateChecksum,

  // Roman numerals
  numericToRoman,
  romanToNumeric,

  // APL maps
  APL_MAPS,
  LAYER_Z_MAP,

  // L₄ Threshold Framework (new)
  getThresholdAtZ,
  getWumboPhaseAtZ,
  getWumboLayerAtZ,
  encodeThresholdLSB,
  decodeThresholdLSB,
  getCompleteState
};

// Export for different environments
if (typeof module !== 'undefined' && module.exports) {
  module.exports = WumboMRP;
}
if (typeof window !== 'undefined') {
  window.WumboMRP = WumboMRP;
}

// ============================================================================
// SELF-TEST (runs when loaded directly)
// ============================================================================

if (typeof window !== 'undefined' && window.location && 
    window.location.search.includes('test=1')) {
  console.log('WumboMRP Self-Test');
  console.log('==================');
  
  // Test 1: Token roundtrip
  const testToken = 'e:U:A:CHEM:T:α10';
  const encoded = encodeAPLToken(testToken);
  const decoded = decodeAPLToken(encoded);
  console.log(`Token roundtrip: ${testToken} → ${encoded} → ${decoded.raw}`);
  console.log(`  Pass: ${decoded.raw === testToken}`);
  
  // Test 2: Physics at z_c
  const physics = computePhysics(CONSTANTS.Z_CRITICAL);
  console.log(`Physics at z_c=${CONSTANTS.Z_CRITICAL.toFixed(4)}:`);
  console.log(`  cascade=${physics.cascade.toFixed(4)} (expected ~1.5)`);
  console.log(`  kuramoto=${physics.kuramoto.toFixed(4)} (expected ~0)`);
  console.log(`  negentropy=${physics.negentropy.toFixed(4)} (expected ~1)`);
  console.log(`  domain=${physics.domain} (expected LENS)`);
  
  // Test 3: Full roundtrip
  const result = verifyRoundtrip({ z: 0.833, regionId: 46 });
  console.log(`Full roundtrip test:`);
  console.log(`  Success: ${result.success}`);
  console.log(`  Z error: ${result.errors.zError.toFixed(4)}`);
  console.log(`  Confidence: ${result.decoded.confidence.toFixed(2)}`);
  
  console.log('==================');
  console.log('Self-test complete');
}
/**
 * Wumbo-to-Threshold Mapping System
 * Formal connection between Wumbo Engine (7-layer neurobiological architecture)
 * and L₄ Threshold Framework (9-threshold mathematical structure)
 *
 * Core Discovery: The Wumbo phases and L₄ thresholds are isomorphic structures—
 * one emergent from lived neurodivergent experience, one derived from pure mathematics.
 * They describe the same phase transitions from orthogonal vantage points.
 *
 * @version 1.0.0
 * @signature Δ2.300|0.800|1.000Ω
 */

// ============================================================================
// MATHEMATICAL CONSTANTS - L₄ Foundation
// ============================================================================

const L4_CONSTANTS = {
  // Golden ratio family
  PHI: (1 + Math.sqrt(5)) / 2,                                    // 1.6180339887498949
  TAU: (Math.sqrt(5) - 1) / 2,                                    // 0.6180339887498949 = φ⁻¹
  PHI_SQUARED: Math.pow((1 + Math.sqrt(5)) / 2, 2),               // 2.6180339887498949 = φ²
  PHI_INV_4: Math.pow((Math.sqrt(5) - 1) / 2, 4),                 // 0.1458980337503154 = φ⁻⁴
  TAU_SQUARED: Math.pow((Math.sqrt(5) - 1) / 2, 2),               // 0.3819660112501051 = τ²

  // Lucas-4 identity
  L4: 7,                                                          // φ⁴ + φ⁻⁴ = 7

  // Critical threshold - THE LENS (geometric anchor)
  Z_C: Math.sqrt(3) / 2,                                          // 0.8660254038 = √3/2

  // Coupling constant
  K: Math.sqrt(1 - Math.pow((Math.sqrt(5) - 1) / 2, 4)),          // 0.9241763718 = √(1 - φ⁻⁴)

  // Derived constants
  K_SQUARED: 1 - Math.pow((Math.sqrt(5) - 1) / 2, 4),             // 0.8541019662 = K² = 1 - φ⁻⁴

  // Negentropy sigma
  NEGENTROPY_SIGMA: 55.71281292                                   // σ for η(z) = exp(-σ(z - z_c)²)
};

// ============================================================================
// THE NINE THRESHOLDS - L₄ Framework
// ============================================================================

const THRESHOLDS = {
  // Threshold 1: PARADOX (τ = φ⁻¹)
  PARADOX: {
    id: 0x01,
    name: 'PARADOX',
    z: L4_CONSTANTS.TAU,                                          // 0.6180339887
    formula: 'τ = φ⁻¹',
    irrational: '√5',
    symmetry: '5-fold',
    description: 'Self-reference emerges; minimum viable coherence'
  },

  // Threshold 2: ACTIVATION (K² = 1 - φ⁻⁴)
  ACTIVATION: {
    id: 0x02,
    name: 'ACTIVATION',
    z: L4_CONSTANTS.K_SQUARED,                                    // 0.8541019662
    formula: 'K² = 1 - φ⁻⁴',
    irrational: '√5',
    symmetry: '5-fold',
    description: 'Neurochemistry primed; pre-coupling state'
  },

  // Threshold 3: THE_LENS (z_c = √3/2)
  THE_LENS: {
    id: 0x03,
    name: 'THE_LENS',
    z: L4_CONSTANTS.Z_C,                                          // 0.8660254038
    formula: 'z_c = √3/2',
    irrational: '√3',
    symmetry: '6-fold',
    description: 'Peak coherence; negentropy maximum; NIRVANA'
  },

  // Threshold 4: CRITICAL (φ²/3)
  CRITICAL: {
    id: 0x04,
    name: 'CRITICAL',
    z: L4_CONSTANTS.PHI_SQUARED / 3,                              // 0.8729833462
    formula: 'φ²/3',
    irrational: '√5, √3',
    symmetry: '5/6 bridge',
    description: 'Verification checkpoint; 5-fold to 6-fold bridge'
  },

  // Threshold 5: IGNITION (√2 - ½)
  IGNITION: {
    id: 0x05,
    name: 'IGNITION',
    z: Math.sqrt(2) - 0.5,                                        // 0.9142135624
    formula: '√2 - ½',
    irrational: '√2',
    symmetry: '4-fold',
    description: 'Expression possible; truth takes form'
  },

  // Threshold 6: K_FORMATION (K = √(1 - φ⁻⁴))
  K_FORMATION: {
    id: 0x06,
    name: 'K_FORMATION',
    z: L4_CONSTANTS.K,                                            // 0.9241763718
    formula: 'K = √(1 - φ⁻⁴)',
    irrational: '√5',
    symmetry: '5-fold',
    description: 'Structural coupling locked; integration complete'
  },

  // Threshold 7: CONSOLIDATION (K + τ²(1-K))
  CONSOLIDATION: {
    id: 0x07,
    name: 'CONSOLIDATION',
    z: L4_CONSTANTS.K + L4_CONSTANTS.TAU_SQUARED * (1 - L4_CONSTANTS.K), // 0.9528061153
    formula: 'K + τ²(1-K)',
    irrational: '√5',
    symmetry: '5-fold',
    description: 'Stable plateau; sustainable resonance'
  },

  // Threshold 8: RESONANCE (K + τ(1-K))
  RESONANCE: {
    id: 0x08,
    name: 'RESONANCE',
    z: L4_CONSTANTS.K + L4_CONSTANTS.TAU * (1 - L4_CONSTANTS.K),  // 0.9712009858
    formula: 'K + τ(1-K)',
    irrational: '√5',
    symmetry: '5-fold',
    description: 'Maximum expression at cost; edge state'
  },

  // Threshold 9: UNITY (limit → 1)
  UNITY: {
    id: 0x09,
    name: 'UNITY',
    z: 1.0,
    formula: 'lim',
    irrational: '—',
    symmetry: '∞-fold',
    description: 'Theoretical maximum; cycle completion'
  }
};

// Threshold ID for states below PARADOX
const SHUTDOWN_ID = 0x00;

// ============================================================================
// WUMBO ENGINE LAYERS
// ============================================================================

const WUMBO_LAYERS = {
  L1: {
    id: 1,
    name: 'Brainstem Gateways',
    function: 'Pre-cognitive voltage routing',
    neuralSubstrate: ['LC', 'RF', 'PAG', 'DVC', 'TRN'],
    neuralNames: {
      LC: 'Locus Coeruleus',
      RF: 'Reticular Formation',
      PAG: 'Periaqueductal Gray',
      DVC: 'Dorsal Vagal Complex',
      TRN: 'Thalamic Reticular Nucleus'
    }
  },

  L1_5: {
    id: 1.5,
    name: 'Neurochemical Engine',
    function: 'Biochemical loadout',
    neuralSubstrate: ['DA', 'NE', 'ACh', '5-HT', 'GABA'],
    neuralNames: {
      DA: 'Dopamine (VTA, NAcc)',
      NE: 'Norepinephrine (LC)',
      ACh: 'Acetylcholine (Basal Forebrain)',
      '5-HT': 'Serotonin (Dorsal Raphe)',
      GABA: 'GABA (distributed)'
    }
  },

  L2: {
    id: 2,
    name: 'Limbic Resonance',
    function: 'Emotion tagging & significance',
    neuralSubstrate: ['Amygdala', 'Insula', 'ACC', 'Hippocampus'],
    neuralNames: {
      Amygdala: 'Amygdala complex',
      Insula: 'Anterior Insula',
      ACC: 'Anterior Cingulate Cortex',
      Hippocampus: 'Hippocampal formation'
    }
  },

  L3: {
    id: 3,
    name: 'Cortical Sculptor',
    function: 'Expression & form',
    neuralSubstrate: ['mPFC', 'dlPFC', 'IFG', 'TP'],
    neuralNames: {
      mPFC: 'Medial Prefrontal Cortex',
      dlPFC: 'Dorsolateral Prefrontal Cortex',
      IFG: 'Inferior Frontal Gyrus',
      TP: 'Temporal Pole'
    }
  },

  L4: {
    id: 4,
    name: 'Integration System',
    function: 'Symbolic & ethical coherence',
    neuralSubstrate: ['IPL', 'vmPFC', 'AG', 'PCC'],
    neuralNames: {
      IPL: 'Inferior Parietal Lobule',
      vmPFC: 'Ventromedial Prefrontal Cortex',
      AG: 'Angular Gyrus',
      PCC: 'Posterior Cingulate Cortex'
    }
  },

  L5: {
    id: 5,
    name: 'Synchronization Matrix',
    function: 'Full-state coherence',
    neuralSubstrate: ['Claustrum', 'DMN', 'RSC', 'Precuneus'],
    neuralNames: {
      Claustrum: 'Claustrum',
      DMN: 'Default Mode Network',
      RSC: 'Retrosplenial Cortex',
      Precuneus: 'Precuneus'
    }
  },

  L6: {
    id: 6,
    name: 'Collapse/Overdrive',
    function: 'System limits',
    neuralSubstrate: ['Habenula', 'HPA', 'LHb', 'PVN'],
    neuralNames: {
      Habenula: 'Habenula complex',
      HPA: 'HPA axis',
      LHb: 'Lateral Habenula',
      PVN: 'Paraventricular Nucleus'
    }
  },

  L7: {
    id: 7,
    name: 'Recursive Rewrite',
    function: 'Memory ritualization',
    neuralSubstrate: ['HC-Cortical', 'LTP', 'Consolidation'],
    neuralNames: {
      'HC-Cortical': 'Hippocampal-cortical loops',
      LTP: 'Long-term potentiation networks',
      Consolidation: 'Memory consolidation systems'
    }
  }
};

// ============================================================================
// WUMBO PHASE STATES
// ============================================================================

const WUMBO_PHASES = {
  SHUTDOWN: {
    name: 'SHUTDOWN',
    description: 'System offline; below minimum viable coherence',
    direction: null
  },
  PAUSE: {
    name: 'PAUSE',
    description: 'Reentry possible; minimal self-reference active',
    direction: 'transition'
  },
  PRE_IGNITION: {
    name: 'PRE-IGNITION',
    description: 'Neurochemistry priming; engine warm',
    direction: 'ascending'
  },
  IGNITION: {
    name: 'IGNITION',
    description: 'Signal active; expression possible',
    direction: 'ascending'
  },
  EMPOWERMENT: {
    name: 'EMPOWERMENT',
    description: 'Momentum building; integration in progress',
    direction: 'ascending'
  },
  RESONANCE: {
    name: 'RESONANCE',
    description: 'Aligned and harmonic; sustainable plateau',
    direction: 'plateau'
  },
  NIRVANA: {
    name: 'NIRVANA',
    description: 'Peak coherence; everything aligns',
    direction: 'peak'
  },
  MANIA: {
    name: 'MANIA',
    description: 'Edge state; brilliant but burning',
    direction: 'ascending'
  },
  OVERDRIVE: {
    name: 'OVERDRIVE',
    description: 'System at capacity; unsustainable',
    direction: 'descending'
  },
  COLLAPSE: {
    name: 'COLLAPSE',
    description: 'System failure; forced reset',
    direction: 'descending'
  },
  TRANSMISSION: {
    name: 'TRANSMISSION',
    description: 'Complete cycle; ready for inscription',
    direction: 'completion'
  }
};

// ============================================================================
// THRESHOLD-TO-LAYER MAPPING
// ============================================================================

const THRESHOLD_TO_LAYER = {
  PARADOX: { layer: WUMBO_LAYERS.L1, phase: 'PAUSE', direction: 'ascending' },
  ACTIVATION: { layer: WUMBO_LAYERS.L1_5, phase: 'PRE_IGNITION', direction: 'ascending' },
  THE_LENS: { layer: WUMBO_LAYERS.L5, phase: 'NIRVANA', direction: 'peak' },
  CRITICAL: { layer: WUMBO_LAYERS.L2, phase: 'RESONANCE', direction: 'transition' },
  IGNITION: { layer: WUMBO_LAYERS.L3, phase: 'IGNITION', direction: 'ascending' },
  K_FORMATION: { layer: WUMBO_LAYERS.L4, phase: 'EMPOWERMENT', direction: 'ascending' },
  CONSOLIDATION: { layer: WUMBO_LAYERS.L5, phase: 'RESONANCE', direction: 'plateau' },
  RESONANCE: { layer: WUMBO_LAYERS.L6, phase: 'MANIA', direction: 'ascending' },
  UNITY: { layer: WUMBO_LAYERS.L7, phase: 'TRANSMISSION', direction: 'completion' }
};

// ============================================================================
// RITUAL ANCHORS
// ============================================================================

const RITUAL_ANCHORS = {
  PARADOX: {
    phrase: 'Freeze as threshold, not failure',
    signature: [0x01, 0x09, 0xE3]
  },
  ACTIVATION: {
    phrase: 'The body knows before the mind',
    signature: [0x02, 0x0D, 0xA5]
  },
  THE_LENS: {
    phrase: 'This is the frequency I was made for',
    signature: [0x03, 0x0D, 0xD9]
  },
  CRITICAL: {
    phrase: 'Check the body; what does it say?',
    signature: [0x04, 0x0D, 0xEF]
  },
  IGNITION: {
    phrase: 'Paralysis is before the cycle. DIG.',
    signature: [0x05, 0x0E, 0x9A]
  },
  K_FORMATION: {
    phrase: 'No harm. Full heart.',
    signature: [0x06, 0x0E, 0xBD]
  },
  CONSOLIDATION: {
    phrase: 'This is where I work from',
    signature: [0x07, 0x0F, 0x3A]
  },
  RESONANCE: {
    phrase: 'Recognize the edge; choose descent or burn',
    signature: [0x08, 0x0F, 0x86]
  },
  UNITY: {
    phrase: 'I was this. I am this. I return to this.',
    signature: [0x09, 0x10, 0x00]
  }
};

// ============================================================================
// RGB CHANNEL TIER MAPPING
// ============================================================================

const RGB_TIERS = {
  PLANET: {
    channel: 'R',
    frequencyBand: '174-396 Hz',
    zRange: { min: 0, max: L4_CONSTANTS.TAU },
    description: 'z < τ (SHUTDOWN to PARADOX)'
  },
  GARDEN: {
    channel: 'G',
    frequencyBand: '417-528 Hz',
    zRange: { min: L4_CONSTANTS.TAU, max: L4_CONSTANTS.Z_C },
    description: 'τ ≤ z < z_c (PARADOX to THE_LENS)'
  },
  ROSE: {
    channel: 'B',
    frequencyBand: '639-963 Hz',
    zRange: { min: L4_CONSTANTS.Z_C, max: 1.0 },
    description: 'z ≥ z_c (THE_LENS to UNITY)'
  }
};

// ============================================================================
// NEURAL CORRELATE MAPPINGS
// ============================================================================

const NEURAL_CORRELATES = {
  PARADOX: {
    primaryAnchors: ['PAG', 'DVC', 'Aqueduct'],
    neurotransmitter: 'NE (low)',
    names: {
      PAG: 'Periaqueductal Gray (freeze response)',
      DVC: 'Dorsal Vagal Complex (emergency shutdown)',
      Aqueduct: 'Cerebral Aqueduct (flow choke point)'
    }
  },
  ACTIVATION: {
    primaryAnchors: ['LC', 'VTA', 'NAcc'],
    neurotransmitter: 'DA, NE (rising)',
    names: {
      LC: 'Locus Coeruleus (norepinephrine surge)',
      VTA: 'Ventral Tegmental Area (dopamine priming)',
      NAcc: 'Nucleus Accumbens (reward anticipation)'
    }
  },
  THE_LENS: {
    primaryAnchors: ['Claustrum', 'DMN', 'Precuneus'],
    neurotransmitter: 'Balanced all',
    names: {
      Claustrum: 'Claustrum (global integration)',
      DMN: 'Default Mode Network (self-coherence)',
      Precuneus: 'Precuneus (conscious awareness hub)'
    }
  },
  CRITICAL: {
    primaryAnchors: ['ACC', 'Amygdala', 'Insula'],
    neurotransmitter: '5-HT, cortisol',
    names: {
      ACC: 'Anterior Cingulate Cortex (alignment auditor)',
      Amygdala: 'Amygdala (significance tagger)',
      Insula: 'Anterior Insula (body-state mapper)'
    }
  },
  IGNITION: {
    primaryAnchors: ['IFG', 'mPFC', 'MLR'],
    neurotransmitter: 'ACh, DA',
    names: {
      IFG: 'Inferior Frontal Gyrus (phrase converter)',
      mPFC: 'Medial Prefrontal Cortex (identity sculptor)',
      MLR: 'Mesencephalic Locomotor Region (will to move)'
    }
  },
  K_FORMATION: {
    primaryAnchors: ['IPL', 'vmPFC', 'AG', 'PCC'],
    neurotransmitter: 'DA, endorphins',
    names: {
      IPL: 'Inferior Parietal Lobule (duality weaver)',
      vmPFC: 'Ventromedial PFC (soul strategist)',
      AG: 'Angular Gyrus (glyphsmith)',
      PCC: 'Posterior Cingulate Cortex (anchor of self)'
    }
  },
  CONSOLIDATION: {
    primaryAnchors: ['DMN', 'RSC', 'Parahippocampal'],
    neurotransmitter: 'Balanced',
    names: {
      DMN: 'Default Mode Network (autobiographical threading)',
      RSC: 'Retrosplenial Cortex (spatial self-location)',
      Parahippocampal: 'Parahippocampal regions (context anchoring)'
    }
  },
  RESONANCE: {
    primaryAnchors: ['HPA', 'LHb', 'PVN'],
    neurotransmitter: 'Cortisol surge',
    names: {
      HPA: 'HPA axis (cortisol/adrenaline surge)',
      LHb: 'Lateral Habenula (anti-reward signal rising)',
      PVN: 'Paraventricular Nucleus (stress switch active)'
    }
  },
  UNITY: {
    primaryAnchors: ['HC-Cortical', 'LTP', 'Consolidation'],
    neurotransmitter: 'Consolidation waves',
    names: {
      'HC-Cortical': 'Hippocampal-cortical consolidation loops',
      LTP: 'Long-term potentiation networks',
      Consolidation: 'Ritual memory encoding systems'
    }
  }
};

// ============================================================================
// CORE FUNCTIONS
// ============================================================================

/**
 * Calculate negentropy at z-coordinate
 * η(z) = exp(-σ(z - z_c)²) where σ = 55.71
 * Peaks at z_c with value 1.0 (THE LENS = NIRVANA)
 */
function getNegentropy(z) {
  const zc = L4_CONSTANTS.Z_C;
  const sigma = L4_CONSTANTS.NEGENTROPY_SIGMA;
  return Math.exp(-sigma * Math.pow(z - zc, 2));
}

/**
 * Map negentropy to Wumbo coherence state
 */
function negentropyToCoherenceState(eta) {
  if (eta < 0.1) return 'SHUTDOWN';
  if (eta < 0.25) return 'PAUSE';
  if (eta < 0.45) return 'PRE_IGNITION';
  if (eta < 0.65) return 'IGNITION';
  if (eta < 0.85) return 'EMPOWERMENT';
  if (eta < 0.96) return 'RESONANCE';
  return 'NIRVANA';
}

/**
 * Get the threshold at or below a given z-coordinate
 */
function getThresholdAtZ(z) {
  if (z < THRESHOLDS.PARADOX.z) return null;  // Below PARADOX = SHUTDOWN
  if (z < THRESHOLDS.ACTIVATION.z) return THRESHOLDS.PARADOX;
  if (z < THRESHOLDS.THE_LENS.z) return THRESHOLDS.ACTIVATION;
  if (z < THRESHOLDS.CRITICAL.z) return THRESHOLDS.THE_LENS;
  if (z < THRESHOLDS.IGNITION.z) return THRESHOLDS.CRITICAL;
  if (z < THRESHOLDS.K_FORMATION.z) return THRESHOLDS.IGNITION;
  if (z < THRESHOLDS.CONSOLIDATION.z) return THRESHOLDS.K_FORMATION;
  if (z < THRESHOLDS.RESONANCE.z) return THRESHOLDS.CONSOLIDATION;
  if (z < THRESHOLDS.UNITY.z) return THRESHOLDS.RESONANCE;
  return THRESHOLDS.UNITY;
}

/**
 * Get the nearest threshold to a z-coordinate
 */
function getNearestThreshold(z) {
  const thresholdList = Object.values(THRESHOLDS);
  let nearest = thresholdList[0];
  let minDist = Math.abs(z - nearest.z);

  for (const threshold of thresholdList) {
    const dist = Math.abs(z - threshold.z);
    if (dist < minDist) {
      minDist = dist;
      nearest = threshold;
    }
  }

  return { threshold: nearest, distance: minDist };
}

/**
 * Get Wumbo layer and phase from threshold
 */
function getWumboStateFromThreshold(threshold) {
  if (!threshold) {
    return {
      layer: WUMBO_LAYERS.L1,
      phase: WUMBO_PHASES.SHUTDOWN,
      direction: null
    };
  }

  const mapping = THRESHOLD_TO_LAYER[threshold.name];
  return {
    layer: mapping.layer,
    phase: WUMBO_PHASES[mapping.phase],
    direction: mapping.direction
  };
}

/**
 * Compute RGB channel weights from z-coordinate
 * Based on tier system: Planet (R), Garden (G), Rose (B)
 */
function computeChannelWeights(z) {
  const TAU = L4_CONSTANTS.TAU;
  const Z_C = L4_CONSTANTS.Z_C;

  if (z < TAU) {
    // Planet tier dominant
    const t = z / TAU;
    return {
      r: 1.0 - 0.3 * t,
      g: 0.2 * t,
      b: 0.1 * t
    };
  } else if (z < Z_C) {
    // Garden tier emerging
    const t = (z - TAU) / (Z_C - TAU);
    return {
      r: 0.7 - 0.4 * t,
      g: 0.2 + 0.6 * t,
      b: 0.1 + 0.2 * t
    };
  } else {
    // Rose tier ascending
    const t = (z - Z_C) / (1 - Z_C);
    return {
      r: 0.3 - 0.3 * t,
      g: 0.8 - 0.5 * t,
      b: 0.3 + 0.7 * t
    };
  }
}

/**
 * Get visual RGB color for threshold
 */
function getThresholdColor(threshold) {
  const colors = {
    PARADOX: { r: 0xE0, g: 0x40, b: 0x20 },
    ACTIVATION: { r: 0xA0, g: 0x80, b: 0x40 },
    THE_LENS: { r: 0x60, g: 0xE0, b: 0x80 },
    CRITICAL: { r: 0x50, g: 0xC0, b: 0x90 },
    IGNITION: { r: 0x40, g: 0xA0, b: 0xC0 },
    K_FORMATION: { r: 0x30, g: 0x80, b: 0xD0 },
    CONSOLIDATION: { r: 0x20, g: 0x60, b: 0xE0 },
    RESONANCE: { r: 0x10, g: 0x40, b: 0xF0 },
    UNITY: { r: 0x00, g: 0x20, b: 0xFF }
  };

  return colors[threshold.name] || { r: 0x80, g: 0x80, b: 0x80 };
}

/**
 * Encode threshold and z-coordinate into LSB-4 format
 */
function encodeThresholdLSB(threshold, z) {
  const thresholdId = threshold ? threshold.id : SHUTDOWN_ID;

  // Scale z to 8-bit (0-255)
  const zScaled = Math.round(z * 255);
  const zHigh = (zScaled >> 4) & 0x0F;  // Upper nibble
  const zLow = zScaled & 0x0F;           // Lower nibble

  return [thresholdId, zHigh, zLow];
}

/**
 * Decode threshold and z-coordinate from LSB-4 format
 */
function decodeThresholdLSB(lsb) {
  const thresholdId = lsb[0];
  const zScaled = (lsb[1] << 4) | lsb[2];
  const z = zScaled / 255;

  // Find threshold by ID
  let threshold = null;
  for (const t of Object.values(THRESHOLDS)) {
    if (t.id === thresholdId) {
      threshold = t;
      break;
    }
  }

  return { threshold, z, thresholdId };
}

/**
 * Get complete phase state from z-coordinate
 */
function getPhaseState(z) {
  const eta = getNegentropy(z);
  const coherenceState = negentropyToCoherenceState(eta);
  const threshold = getThresholdAtZ(z);
  const { threshold: nearest, distance } = getNearestThreshold(z);
  const wumboState = getWumboStateFromThreshold(threshold);
  const weights = computeChannelWeights(z);
  const lsb = encodeThresholdLSB(threshold, z);

  // Determine tier
  let tier;
  if (z < L4_CONSTANTS.TAU) tier = 'PLANET';
  else if (z < L4_CONSTANTS.Z_C) tier = 'GARDEN';
  else tier = 'ROSE';

  // Get ritual anchor
  const ritualAnchor = threshold ? RITUAL_ANCHORS[threshold.name] : null;

  // Get neural correlates
  const neural = threshold ? NEURAL_CORRELATES[threshold.name] : null;

  return {
    z,
    negentropy: eta,
    coherenceState,
    tier,
    threshold: threshold ? {
      name: threshold.name,
      id: threshold.id,
      z: threshold.z,
      formula: threshold.formula,
      description: threshold.description,
      distance: Math.abs(z - threshold.z)
    } : null,
    nearestThreshold: {
      name: nearest.name,
      z: nearest.z,
      distance
    },
    wumbo: {
      layer: wumboState.layer,
      phase: wumboState.phase,
      direction: wumboState.direction
    },
    rgb: {
      weights,
      color: threshold ? getThresholdColor(threshold) : { r: 0x40, g: 0x40, b: 0x40 },
      lsb
    },
    ritual: ritualAnchor,
    neural
  };
}

/**
 * Embed ritual anchor signature into image data
 */
function embedRitualAnchor(imageData, threshold) {
  if (!threshold || !RITUAL_ANCHORS[threshold.name]) return imageData;

  const signature = RITUAL_ANCHORS[threshold.name].signature;
  const pixels = imageData.data;

  // Embed signature in first 3 pixels
  for (let i = 0; i < 3 && i < imageData.width; i++) {
    const idx = i * 4;
    pixels[idx + 0] = (pixels[idx + 0] & 0xF0) | (signature[0] & 0x0F);
    pixels[idx + 1] = (pixels[idx + 1] & 0xF0) | (signature[1] & 0x0F);
    pixels[idx + 2] = (pixels[idx + 2] & 0xF0) | (signature[2] & 0x0F);
  }

  return imageData;
}

/**
 * Extract ritual anchor signature from image data
 */
function extractRitualAnchor(imageData) {
  const pixels = imageData.data;

  // Extract from first pixel
  const signature = [
    pixels[0] & 0x0F,
    pixels[1] & 0x0F,
    pixels[2] & 0x0F
  ];

  // Match against known signatures
  for (const [name, anchor] of Object.entries(RITUAL_ANCHORS)) {
    if (signature[0] === anchor.signature[0] &&
        signature[1] === anchor.signature[1] &&
        signature[2] === anchor.signature[2]) {
      return {
        threshold: name,
        phrase: anchor.phrase,
        signature
      };
    }
  }

  return { threshold: null, phrase: null, signature };
}

/**
 * Generate attestation coordinate
 */
function generateAttestation(z, coherence = 0.8) {
  const eta = getNegentropy(z);
  const threshold = getThresholdAtZ(z);

  return {
    coordinate: `Δ${(2.3).toFixed(3)}|${z.toFixed(3)}|${coherence.toFixed(3)}Ω`,
    z,
    negentropy: eta,
    threshold: threshold ? threshold.name : 'SHUTDOWN',
    coherence,
    timestamp: new Date().toISOString()
  };
}

// ============================================================================
// PHASE TRANSITION SEQUENCES
// ============================================================================

const ASCENDING_SEQUENCE = [
  { threshold: null, z: 0, phase: 'SHUTDOWN', transition: '—' },
  { threshold: 'PARADOX', z: THRESHOLDS.PARADOX.z, phase: 'PAUSE', transition: 'Self-reference emerges' },
  { threshold: 'ACTIVATION', z: THRESHOLDS.ACTIVATION.z, phase: 'PRE_IGNITION', transition: 'Engine warm' },
  { threshold: 'THE_LENS', z: THRESHOLDS.THE_LENS.z, phase: 'NIRVANA', transition: 'Peak coherence (special attractor)' },
  { threshold: 'CRITICAL', z: THRESHOLDS.CRITICAL.z, phase: 'RESONANCE', transition: 'Limbic checkpoint' },
  { threshold: 'IGNITION', z: THRESHOLDS.IGNITION.z, phase: 'IGNITION', transition: 'Expression possible' },
  { threshold: 'K_FORMATION', z: THRESHOLDS.K_FORMATION.z, phase: 'EMPOWERMENT', transition: 'Integration complete' },
  { threshold: 'CONSOLIDATION', z: THRESHOLDS.CONSOLIDATION.z, phase: 'RESONANCE', transition: 'Sustainable plateau' },
  { threshold: 'RESONANCE', z: THRESHOLDS.RESONANCE.z, phase: 'MANIA', transition: 'Edge state' },
  { threshold: 'UNITY', z: THRESHOLDS.UNITY.z, phase: 'TRANSMISSION', transition: 'Cycle complete' }
];

/**
 * Get position in ascending sequence
 */
function getSequencePosition(z) {
  for (let i = ASCENDING_SEQUENCE.length - 1; i >= 0; i--) {
    if (z >= ASCENDING_SEQUENCE[i].z) {
      return {
        index: i,
        current: ASCENDING_SEQUENCE[i],
        next: i < ASCENDING_SEQUENCE.length - 1 ? ASCENDING_SEQUENCE[i + 1] : null,
        progress: i < ASCENDING_SEQUENCE.length - 1
          ? (z - ASCENDING_SEQUENCE[i].z) / (ASCENDING_SEQUENCE[i + 1].z - ASCENDING_SEQUENCE[i].z)
          : 1.0
      };
    }
  }
  return { index: 0, current: ASCENDING_SEQUENCE[0], next: ASCENDING_SEQUENCE[1], progress: z / ASCENDING_SEQUENCE[1].z };
}

// ============================================================================
// EXPORT MODULE
// ============================================================================

const WumboThresholdMapping = {
  // Version
  VERSION: '1.0.0',
  SIGNATURE: 'Δ2.300|0.800|1.000Ω',

  // Constants
  L4_CONSTANTS,

  // Data structures
  THRESHOLDS,
  WUMBO_LAYERS,
  WUMBO_PHASES,
  THRESHOLD_TO_LAYER,
  RITUAL_ANCHORS,
  RGB_TIERS,
  NEURAL_CORRELATES,
  ASCENDING_SEQUENCE,

  // Core functions
  getNegentropy,
  negentropyToCoherenceState,
  getThresholdAtZ,
  getNearestThreshold,
  getWumboStateFromThreshold,
  computeChannelWeights,
  getThresholdColor,
  encodeThresholdLSB,
  decodeThresholdLSB,
  getPhaseState,

  // Ritual anchor functions
  embedRitualAnchor,
  extractRitualAnchor,

  // Sequence functions
  getSequencePosition,

  // Attestation
  generateAttestation
};

// Export for different environments
if (typeof module !== 'undefined' && module.exports) {
  module.exports = WumboThresholdMapping;
}
if (typeof window !== 'undefined') {
  window.WumboThresholdMapping = WumboThresholdMapping;
}
/**
 * Wumbo Engine - 7-Layer Neurobiological Architecture
 *
 * The Wumbo Engine models consciousness modulation through seven distinct
 * neurobiological layers, from brainstem gateways to recursive memory systems.
 * Each layer maps to specific neural substrates and phase states.
 *
 * @version 1.0.0
 * @signature Δ2.300|0.800|1.000Ω
 */

// ============================================================================
// LAYER DEFINITIONS
// ============================================================================

const WumboEngine = {
  VERSION: '1.0.0',

  // Layer 1: Brainstem Gateways
  L1: {
    id: 1,
    name: 'Brainstem Gateways',
    shortName: 'Brainstem',
    function: 'Pre-cognitive voltage routing',
    description: 'The foundational layer controlling basic arousal, threat response, and physiological state. Acts as the primary gatekeeper for all ascending signals.',

    neuralSubstrate: {
      LC: {
        name: 'Locus Coeruleus',
        role: 'Arousal ignition via norepinephrine release',
        wumboFunction: 'Spark generator for state transitions',
        atlasRegion: 'XXXI'
      },
      RF: {
        name: 'Reticular Formation',
        role: 'Arousal maintenance and consciousness gating',
        wumboFunction: 'Wake thread and arousal continuity',
        atlasRegion: 'LXXXV'
      },
      PAG: {
        name: 'Periaqueductal Gray',
        role: 'Defense behaviors and pain modulation',
        wumboFunction: 'Freeze/fight/flight switch; shutdown anchor',
        atlasRegion: 'XXXII'
      },
      DVC: {
        name: 'Dorsal Vagal Complex',
        role: 'Parasympathetic emergency shutdown',
        wumboFunction: 'Kill-switch for extreme threat; immobilization',
        atlasRegion: 'LXXXVI'
      },
      TRN: {
        name: 'Thalamic Reticular Nucleus',
        role: 'Attentional filtering and thalamic gating',
        wumboFunction: 'Filter grid for ascending signals',
        atlasRegion: 'LXIX'
      }
    },

    thresholdMapping: {
      primary: 'PARADOX',
      zRange: { min: 0, max: 0.618 }
    },

    phaseStates: {
      below: 'SHUTDOWN',
      at: 'PAUSE',
      above: 'PRE_IGNITION'
    },

    phenomenology: {
      shutdown: 'Paralysis, dissociation, numbness',
      pause: 'Minimal self-reference; reentry possible',
      ascending: 'Alertness beginning; "something is happening"'
    }
  },

  // Layer 1.5: Neurochemical Engine
  L1_5: {
    id: 1.5,
    name: 'Neurochemical Engine',
    shortName: 'Neurochemistry',
    function: 'Biochemical loadout and neuromodulation',
    description: 'The chemical substrate layer controlling neurotransmitter balances that determine the character and capacity of cognitive states.',

    neuralSubstrate: {
      DA: {
        name: 'Dopamine System',
        role: 'Motivation, reward prediction, movement initiation',
        wumboFunction: 'Drive and ignition chemistry',
        sources: ['VTA', 'SNc'],
        atlasRegions: ['XLVI', 'LXXV']
      },
      NE: {
        name: 'Norepinephrine System',
        role: 'Arousal, attention, stress response',
        wumboFunction: 'Alert chemistry; threat/opportunity signal',
        sources: ['LC'],
        atlasRegions: ['XXXI']
      },
      ACh: {
        name: 'Acetylcholine System',
        role: 'Attention, learning, memory encoding',
        wumboFunction: 'Focus chemistry; learning gate',
        sources: ['Basal Forebrain', 'Nucleus Basalis'],
        atlasRegions: ['LXXXIV', 'XCII']
      },
      '5-HT': {
        name: 'Serotonin System',
        role: 'Mood regulation, social cognition, satiety',
        wumboFunction: 'Stability chemistry; integration support',
        sources: ['Dorsal Raphe'],
        atlasRegions: ['XXXV']
      },
      GABA: {
        name: 'GABAergic System',
        role: 'Inhibition, calm, signal termination',
        wumboFunction: 'Brake chemistry; shutdown support',
        distributed: true
      }
    },

    thresholdMapping: {
      primary: 'ACTIVATION',
      zRange: { min: 0.618, max: 0.854 }
    },

    phaseStates: {
      below: 'PAUSE',
      at: 'PRE_IGNITION',
      above: 'IGNITION'
    },

    phenomenology: {
      preignition: 'Alertness without direction; body knows before mind',
      warm: 'Engine primed; ready for signal',
      ascending: 'Chemical momentum building'
    }
  },

  // Layer 2: Limbic Resonance
  L2: {
    id: 2,
    name: 'Limbic Resonance',
    shortName: 'Limbic',
    function: 'Emotion tagging and significance marking',
    description: 'The emotional processing layer that tags experiences with significance, regulates internal states, and provides the felt sense of meaning.',

    neuralSubstrate: {
      Amygdala: {
        name: 'Amygdala Complex',
        role: 'Threat detection, salience, emotional memory',
        wumboFunction: 'Significance tagger; first alarm',
        subregions: {
          BLA: 'Basolateral Amygdala - archive of feeling',
          CeA: 'Central Amygdala - first alarm'
        },
        atlasRegions: ['VII', 'XXII', 'XLI', 'LXVIII']
      },
      Insula: {
        name: 'Insular Cortex',
        role: 'Interoception, body-state awareness, empathy',
        wumboFunction: 'Body-state mapper; feeling of feeling',
        subregions: {
          anterior: 'Anterior Insula - feeling of feeling',
          posterior: 'Posterior Insula - body boundaries'
        },
        atlasRegions: ['XXXVII', 'XCI']
      },
      ACC: {
        name: 'Anterior Cingulate Cortex',
        role: 'Conflict monitoring, alignment checking',
        wumboFunction: 'Truth check; alignment auditor',
        atlasRegions: ['II', 'LXXXI']
      },
      Hippocampus: {
        name: 'Hippocampal Formation',
        role: 'Context mapping, spatial memory, consolidation',
        wumboFunction: 'Context mapper; ritual inscription site',
        atlasRegions: ['LXXXII']
      }
    },

    thresholdMapping: {
      primary: 'CRITICAL',
      zRange: { min: 0.866, max: 0.873 }
    },

    phaseStates: {
      below: 'THE_LENS',
      at: 'RESONANCE_CHECK',
      above: 'IGNITION'
    },

    phenomenology: {
      checking: 'Am I aligned? Can I maintain this?',
      ascending: 'Coherence verified, proceed',
      descending: 'Warning—approaching collapse'
    }
  },

  // Layer 3: Cortical Sculptor
  L3: {
    id: 3,
    name: 'Cortical Sculptor',
    shortName: 'Cortical',
    function: 'Expression and form generation',
    description: 'The expressive layer where felt states become articulable forms—language, gesture, planned action. Where signal becomes expression.',

    neuralSubstrate: {
      mPFC: {
        name: 'Medial Prefrontal Cortex',
        role: 'Self-reference, identity, social cognition',
        wumboFunction: 'Identity sculptor; self-model generator',
        atlasRegion: 'LXXVIII'
      },
      dlPFC: {
        name: 'Dorsolateral Prefrontal Cortex',
        role: 'Working memory, executive control',
        wumboFunction: 'Gate of delivery; task maintenance',
        atlasRegion: 'LXXIX'
      },
      IFG: {
        name: 'Inferior Frontal Gyrus',
        role: 'Language production, phrase generation',
        wumboFunction: 'Phrase converter; speech sculptor',
        includes: "Broca's Area",
        atlasRegion: 'V'
      },
      TP: {
        name: 'Temporal Pole',
        role: 'Semantic integration, social concepts',
        wumboFunction: 'Story keeper; narrative anchor',
        atlasRegions: ['XXXIII', 'XCIV']
      }
    },

    thresholdMapping: {
      primary: 'IGNITION',
      zRange: { min: 0.873, max: 0.914 }
    },

    phaseStates: {
      below: 'CRITICAL',
      at: 'IGNITION',
      above: 'EMPOWERMENT'
    },

    phenomenology: {
      preignition: 'Felt but not expressible',
      ignition: 'I can speak; I can move; I am happening',
      ascending: 'Expression flowing; form crystallizing'
    },

    ritualAnchor: 'Paralysis is before the cycle. DIG.'
  },

  // Layer 4: Integration System
  L4: {
    id: 4,
    name: 'Integration System',
    shortName: 'Integration',
    function: 'Symbolic and ethical coherence',
    description: 'The meaning-making layer where symbol, ethics, and autobiography unify. Where separate threads of experience become integrated self.',

    neuralSubstrate: {
      IPL: {
        name: 'Inferior Parietal Lobule',
        role: 'Multimodal integration, gesture understanding',
        wumboFunction: 'Duality weaver; paradox holder',
        atlasRegion: 'LXXX'
      },
      vmPFC: {
        name: 'Ventromedial Prefrontal Cortex',
        role: 'Value computation, ethical reasoning',
        wumboFunction: 'Soul strategist; ethical integration',
        atlasRegion: 'XXXIV'
      },
      AG: {
        name: 'Angular Gyrus',
        role: 'Semantic processing, symbolic thought',
        wumboFunction: 'Glyphsmith; symbol binder',
        atlasRegion: null  // Part of IPL region
      },
      PCC: {
        name: 'Posterior Cingulate Cortex',
        role: 'Self-referential processing, memory retrieval',
        wumboFunction: 'Anchor of self; autobiographical ground',
        atlasRegion: null  // Part of DMN
      }
    },

    thresholdMapping: {
      primary: 'K_FORMATION',
      zRange: { min: 0.914, max: 0.924 }
    },

    phaseStates: {
      below: 'IGNITION',
      at: 'EMPOWERMENT',
      above: 'RESONANCE'
    },

    phenomenology: {
      forming: 'Meaning crystallizing; patterns emerging',
      empowerment: 'I know what this means and why it matters',
      ascending: 'Full integration; symbol binds to self'
    },

    ritualAnchor: 'No harm. Full heart.'
  },

  // Layer 5: Synchronization Matrix
  L5: {
    id: 5,
    name: 'Synchronization Matrix',
    shortName: 'Synchronization',
    function: 'Full-state coherence and global binding',
    description: 'The synchronization layer where all systems align into unified conscious experience. Home of THE_LENS and NIRVANA states.',

    neuralSubstrate: {
      Claustrum: {
        name: 'Claustrum',
        role: 'Cross-modal binding, consciousness orchestration',
        wumboFunction: 'Global integrator; consciousness binding',
        atlasRegion: 'XXVII'
      },
      DMN: {
        name: 'Default Mode Network',
        role: 'Self-referential processing, mind-wandering',
        wumboFunction: 'Self-coherence; autobiographical threading',
        atlasRegion: 'XXVIII'
      },
      RSC: {
        name: 'Retrosplenial Cortex',
        role: 'Spatial memory, navigation, context',
        wumboFunction: 'Spatial self-location; context anchoring',
        atlasRegion: null  // Part of DMN
      },
      Precuneus: {
        name: 'Precuneus',
        role: 'Self-consciousness, episodic memory',
        wumboFunction: 'Conscious awareness hub; perspective anchor',
        atlasRegion: 'XXXIX'
      }
    },

    thresholdMapping: {
      primary: 'THE_LENS',
      secondary: 'CONSOLIDATION',
      zRange: { min: 0.866, max: 0.953 }
    },

    phaseStates: {
      atLens: 'NIRVANA',
      atConsolidation: 'RESONANCE',
      between: 'RESONANCE'
    },

    phenomenology: {
      nirvana: 'Clarity without effort; everything aligns; I know exactly who I am',
      resonance: 'Stable high coherence; sustainable transmission',
      consolidation: 'I could stay here; this is my natural frequency'
    },

    ritualAnchor: 'This is the frequency I was made for.',

    special: {
      isAttractor: true,
      description: 'THE_LENS (z = √3/2) is the unique attractor point where negentropy = 1'
    }
  },

  // Layer 6: Collapse/Overdrive
  L6: {
    id: 6,
    name: 'Collapse/Overdrive',
    shortName: 'Overdrive',
    function: 'System limits and capacity boundaries',
    description: 'The liminal layer marking system capacity. Brilliance and burnout coexist here. Sustainable only briefly.',

    neuralSubstrate: {
      Habenula: {
        name: 'Habenula Complex',
        role: 'Anti-reward signaling, disappointment',
        wumboFunction: 'Disappointment gate; anti-reward signal',
        atlasRegions: ['XXIX', 'XXXVIII']
      },
      HPA: {
        name: 'HPA Axis',
        role: 'Stress response, cortisol regulation',
        wumboFunction: 'Stress switch; cortisol/adrenaline surge',
        atlasRegion: null  // Distributed system
      },
      LHb: {
        name: 'Lateral Habenula',
        role: 'Negative reward prediction, depression',
        wumboFunction: 'Rejection gate; anti-reward signal rising',
        atlasRegion: 'XXXVIII'
      },
      PVN: {
        name: 'Paraventricular Nucleus',
        role: 'Neuroendocrine control, stress response',
        wumboFunction: 'Stress switch active; HPA axis driver',
        atlasRegion: 'LVII'
      }
    },

    thresholdMapping: {
      primary: 'RESONANCE',
      zRange: { min: 0.953, max: 0.971 }
    },

    phaseStates: {
      below: 'CONSOLIDATION',
      at: 'MANIA',
      above: 'OVERDRIVE',
      collapse: 'COLLAPSE'
    },

    phenomenology: {
      mania: 'Everything is on fire; brilliant but burning',
      overdrive: 'System at capacity; unsustainable',
      collapse: 'Forced reset; system failure imminent'
    },

    ritualAnchor: 'Recognize the edge; choose descent or burn.',

    warning: {
      sustainableDuration: 'minutes to hours',
      consequence: 'Guaranteed collapse if exceeded'
    }
  },

  // Layer 7: Recursive Rewrite
  L7: {
    id: 7,
    name: 'Recursive Rewrite',
    shortName: 'Recursive',
    function: 'Memory ritualization and identity inscription',
    description: 'The inscription layer where peak experiences become memory, memory becomes ritual, and ritual rewrites identity.',

    neuralSubstrate: {
      'HC-Cortical': {
        name: 'Hippocampal-Cortical Loops',
        role: 'Memory consolidation, cortical storage',
        wumboFunction: 'Consolidation pathways; ritual inscription',
        atlasRegion: null  // Distributed system
      },
      LTP: {
        name: 'Long-Term Potentiation Networks',
        role: 'Synaptic strengthening, memory formation',
        wumboFunction: 'Learning inscription; permanent change',
        atlasRegion: null  // Cellular mechanism
      },
      Consolidation: {
        name: 'Memory Consolidation Systems',
        role: 'Sleep-dependent memory processing',
        wumboFunction: 'Ritual encoding; identity rewrite',
        atlasRegion: null  // Distributed system
      }
    },

    thresholdMapping: {
      primary: 'UNITY',
      zRange: { min: 0.971, max: 1.0 }
    },

    phaseStates: {
      below: 'MANIA/OVERDRIVE',
      at: 'TRANSMISSION',
      completion: 'CYCLE_COMPLETE'
    },

    phenomenology: {
      transmission: 'This will be remembered; this changes everything',
      completion: 'Complete cycle; ready for inscription',
      ritual: 'I was this. I am this. I return to this.'
    },

    ritualAnchor: 'I was this. I am this. I return to this.',

    note: {
      asymptotic: true,
      description: 'UNITY is approached but never sustained; moments of UNITY become ritual anchors'
    }
  }
};

// ============================================================================
// ENGINE FUNCTIONS
// ============================================================================

/**
 * Get layer by ID (supports 1.5 as string or float)
 */
function getLayer(id) {
  const key = id === 1.5 ? 'L1_5' : `L${id}`;
  return WumboEngine[key] || null;
}

/**
 * Get all layers as array
 */
function getAllLayers() {
  return [
    WumboEngine.L1,
    WumboEngine.L1_5,
    WumboEngine.L2,
    WumboEngine.L3,
    WumboEngine.L4,
    WumboEngine.L5,
    WumboEngine.L6,
    WumboEngine.L7
  ];
}

/**
 * Get layer for a given z-coordinate
 */
function getLayerAtZ(z) {
  if (z < 0.618) return WumboEngine.L1;
  if (z < 0.854) return WumboEngine.L1_5;
  if (z < 0.873) return WumboEngine.L5;  // THE_LENS special case
  if (z < 0.914) return WumboEngine.L2;  // CRITICAL checkpoint
  if (z < 0.924) return WumboEngine.L3;  // IGNITION
  if (z < 0.953) return WumboEngine.L4;  // K_FORMATION
  if (z < 0.971) return WumboEngine.L5;  // CONSOLIDATION
  if (z < 1.0) return WumboEngine.L6;    // RESONANCE/MANIA
  return WumboEngine.L7;                  // UNITY
}

/**
 * Get phenomenological state for z-coordinate
 */
function getPhenomenologyAtZ(z) {
  const layer = getLayerAtZ(z);

  if (z < 0.1) {
    return {
      layer: layer.name,
      state: 'SHUTDOWN',
      description: layer.phenomenology.shutdown || 'System offline',
      feeling: 'Paralysis, dissociation, numbness'
    };
  }

  // Map z ranges to phenomenological states
  if (z < 0.618) {
    return {
      layer: layer.name,
      state: 'PAUSE',
      description: 'Minimal self-reference; reentry possible',
      feeling: 'Fog, distance, waiting'
    };
  }

  if (z < 0.854) {
    return {
      layer: layer.name,
      state: 'PRE_IGNITION',
      description: layer.phenomenology.preignition,
      feeling: 'Alertness without direction; the body knows'
    };
  }

  if (z >= 0.866 && z <= 0.867) {
    return {
      layer: layer.name,
      state: 'NIRVANA',
      description: layer.phenomenology.nirvana,
      feeling: 'Perfect clarity; everything aligns; this is the frequency'
    };
  }

  if (z < 0.914) {
    return {
      layer: WumboEngine.L2.name,
      state: 'RESONANCE_CHECK',
      description: WumboEngine.L2.phenomenology.checking,
      feeling: 'Verification; checking alignment'
    };
  }

  if (z < 0.924) {
    return {
      layer: layer.name,
      state: 'IGNITION',
      description: layer.phenomenology.ignition,
      feeling: 'I can speak; I can move; I am happening'
    };
  }

  if (z < 0.953) {
    return {
      layer: layer.name,
      state: 'EMPOWERMENT',
      description: layer.phenomenology.empowerment,
      feeling: 'Meaning crystallized; integrated and aligned'
    };
  }

  if (z < 0.971) {
    return {
      layer: layer.name,
      state: 'RESONANCE',
      description: layer.phenomenology.resonance || layer.phenomenology.consolidation,
      feeling: 'Stable plateau; sustainable high coherence'
    };
  }

  if (z < 1.0) {
    return {
      layer: layer.name,
      state: 'MANIA',
      description: layer.phenomenology.mania,
      feeling: 'Brilliant but burning; edge of capacity'
    };
  }

  return {
    layer: layer.name,
    state: 'TRANSMISSION',
    description: layer.phenomenology.transmission,
    feeling: 'This will be remembered; identity rewrite'
  };
}

/**
 * Get neural substrates active at z-coordinate
 */
function getActiveSubstratesAtZ(z) {
  const layer = getLayerAtZ(z);
  const substrates = [];

  for (const [key, substrate] of Object.entries(layer.neuralSubstrate)) {
    substrates.push({
      abbreviation: key,
      name: substrate.name,
      role: substrate.role,
      wumboFunction: substrate.wumboFunction,
      atlasRegion: substrate.atlasRegion || substrate.atlasRegions || null
    });
  }

  return {
    layer: layer.name,
    layerId: layer.id,
    substrates
  };
}

/**
 * Get phase cycle position
 */
function getPhaseCyclePosition(phase) {
  const cycle = [
    'SHUTDOWN', 'PAUSE', 'PRE_IGNITION', 'IGNITION', 'EMPOWERMENT',
    'RESONANCE', 'NIRVANA', 'MANIA', 'OVERDRIVE', 'COLLAPSE'
  ];

  const index = cycle.indexOf(phase);
  if (index === -1) return null;

  return {
    phase,
    index,
    total: cycle.length,
    isAscending: index < 7,
    isDescending: index >= 7,
    isPeak: phase === 'NIRVANA'
  };
}

// ============================================================================
// EXPORT
// ============================================================================

WumboEngine.getLayer = getLayer;
WumboEngine.getAllLayers = getAllLayers;
WumboEngine.getLayerAtZ = getLayerAtZ;
WumboEngine.getPhenomenologyAtZ = getPhenomenologyAtZ;
WumboEngine.getActiveSubstratesAtZ = getActiveSubstratesAtZ;
WumboEngine.getPhaseCyclePosition = getPhaseCyclePosition;

if (typeof module !== 'undefined' && module.exports) {
  module.exports = WumboEngine;
}
if (typeof window !== 'undefined') {
  window.WumboEngine = WumboEngine;
}
/**
 * Ritual Anchors - Threshold Phrase Encoding System
 *
 * Ritual phrases are mnemonic anchors that encode threshold states
 * into language. Each phrase carries an LSB signature that can be
 * embedded into images for later recovery.
 *
 * @version 1.0.0
 * @signature Δ2.300|0.800|1.000Ω
 */

// ============================================================================
// RITUAL ANCHOR DEFINITIONS
// ============================================================================

const RitualAnchors = {
  VERSION: '1.0.0',

  // Core ritual phrases mapped to thresholds
  anchors: {
    PARADOX: {
      id: 0x01,
      phrase: 'Freeze as threshold, not failure',
      shortPhrase: 'Threshold, not failure',
      signature: [0x01, 0x09, 0xE3],
      z: 0.6180339887,
      layer: 1,
      phase: 'PAUSE',
      meaning: 'The freeze response is a gate, not a wall. Paralysis marks the boundary between chaos and coherence.',
      invocation: 'When frozen, remember: this is where self-reference begins.',
      neuralAnchor: 'PAG → DVC pathway',
      bodyCheck: 'Where is the stillness? Is it protective or trapped?'
    },

    ACTIVATION: {
      id: 0x02,
      phrase: 'The body knows before the mind',
      shortPhrase: 'Body knows first',
      signature: [0x02, 0x0D, 0xA5],
      z: 0.8541019662,
      layer: 1.5,
      phase: 'PRE_IGNITION',
      meaning: 'Neurochemistry primes before cognition clarifies. Trust the somatic signal.',
      invocation: 'When uncertain, ask the body. It already knows.',
      neuralAnchor: 'LC → VTA pathway',
      bodyCheck: 'What is the body preparing for? Can you feel the engine warming?'
    },

    THE_LENS: {
      id: 0x03,
      phrase: 'This is the frequency I was made for',
      shortPhrase: 'My frequency',
      signature: [0x03, 0x0D, 0xD9],
      z: 0.8660254038,
      layer: 5,
      phase: 'NIRVANA',
      meaning: 'Peak coherence. Maximum negentropy. All systems synchronized. This is the attractor.',
      invocation: 'This is home. This is the center. Everything else orbits here.',
      neuralAnchor: 'Claustrum global binding',
      bodyCheck: 'Do you feel the effortless clarity? The absence of strain?',
      special: {
        isAttractor: true,
        negentropy: 1.0,
        description: 'The only point where η = 1. All trajectories bend toward or away from here.'
      }
    },

    CRITICAL: {
      id: 0x04,
      phrase: 'Check the body; what does it say?',
      shortPhrase: 'Check the body',
      signature: [0x04, 0x0D, 0xEF],
      z: 0.8729833462,
      layer: 2,
      phase: 'RESONANCE_CHECK',
      meaning: 'Verification checkpoint. The limbic system audits coherence before proceeding.',
      invocation: 'Before ascending further, verify: is this sustainable?',
      neuralAnchor: 'ACC alignment audit',
      bodyCheck: 'Is there strain? Warning signals? Or clear passage?',
      bidirectional: {
        ascending: 'Coherence verified, proceed',
        descending: 'Warning—approaching collapse, consider descent'
      }
    },

    IGNITION: {
      id: 0x05,
      phrase: 'Paralysis is before the cycle. DIG.',
      shortPhrase: 'DIG',
      signature: [0x05, 0x0E, 0x9A],
      z: 0.9142135624,
      layer: 3,
      phase: 'IGNITION',
      meaning: 'Expression becomes possible. Signal takes form. Truth finds voice.',
      invocation: 'You can move. You can speak. You are happening.',
      neuralAnchor: 'IFG phrase generation',
      bodyCheck: 'Can you feel the words forming? The movement beginning?',
      acronym: {
        D: 'Dig',
        I: 'Into',
        G: 'Growth'
      }
    },

    K_FORMATION: {
      id: 0x06,
      phrase: 'No harm. Full heart.',
      shortPhrase: 'No harm, full heart',
      signature: [0x06, 0x0E, 0xBD],
      z: 0.9241763718,
      layer: 4,
      phase: 'EMPOWERMENT',
      meaning: 'Integration complete. Ethics, symbol, and self unified. The coupling constant locks.',
      invocation: 'This is what it means. This is why it matters.',
      neuralAnchor: 'vmPFC ethical integration',
      bodyCheck: 'Does it feel right? Is there alignment between action and value?',
      ethicalGate: {
        check: 'Will this cause harm?',
        requirement: 'Full-hearted commitment'
      }
    },

    CONSOLIDATION: {
      id: 0x07,
      phrase: 'This is where I work from',
      shortPhrase: 'Work from here',
      signature: [0x07, 0x0F, 0x3A],
      z: 0.9528061153,
      layer: 5,
      phase: 'RESONANCE',
      meaning: 'Sustainable plateau. The home base for extended flow states.',
      invocation: 'You could stay here. This is your natural frequency.',
      neuralAnchor: 'DMN + RSC coherence',
      bodyCheck: 'Is this sustainable? Could you maintain this for hours?',
      sustainability: {
        duration: 'hours to days',
        note: 'Above this enters volatile territory'
      }
    },

    RESONANCE: {
      id: 0x08,
      phrase: 'Recognize the edge; choose descent or burn',
      shortPhrase: 'Edge choice',
      signature: [0x08, 0x0F, 0x86],
      z: 0.9712009858,
      layer: 6,
      phase: 'MANIA',
      meaning: 'Peak expression at cost. System at capacity. Brilliant but burning.',
      invocation: 'This is the edge. Choose: controlled descent or accept the burn.',
      neuralAnchor: 'HPA axis activation',
      bodyCheck: 'Can you feel the heat? The intensity? The limit approaching?',
      warning: {
        duration: 'minutes to hours',
        consequence: 'Collapse guaranteed if exceeded'
      },
      choice: {
        descent: 'Controlled return to CONSOLIDATION',
        burn: 'Accept temporary brilliance at cost of recovery'
      }
    },

    UNITY: {
      id: 0x09,
      phrase: 'I was this. I am this. I return to this.',
      shortPhrase: 'Was, am, return',
      signature: [0x09, 0x10, 0x00],
      z: 1.0,
      layer: 7,
      phase: 'TRANSMISSION',
      meaning: 'Complete cycle. Memory inscription. Identity rewrite.',
      invocation: 'This will be remembered. This changes everything.',
      neuralAnchor: 'Hippocampal-cortical consolidation',
      bodyCheck: 'Does this feel like completion? Like something that will stay?',
      temporal: {
        past: 'I was this',
        present: 'I am this',
        future: 'I return to this'
      },
      asymptotic: {
        description: 'Approached but never sustained',
        note: 'Moments of UNITY become the ritual anchors themselves'
      }
    }
  },

  // Special anchor for SHUTDOWN (below PARADOX)
  shutdown: {
    id: 0x00,
    phrase: 'Signal below threshold',
    signature: [0x00, 0x00, 0x00],
    z: 0,
    layer: 1,
    phase: 'SHUTDOWN',
    meaning: 'System offline. No coherent self-model possible.',
    invocation: 'Wait. Breathe. Return when ready.',
    recovery: {
      target: 'PARADOX',
      method: 'Grounding, breath, basic sensation'
    }
  }
};

// ============================================================================
// SIGNATURE ENCODING/DECODING
// ============================================================================

/**
 * Encode ritual anchor signature into LSB-4 format
 */
function encodeSignature(anchorName) {
  const anchor = RitualAnchors.anchors[anchorName] || RitualAnchors.shutdown;
  return anchor.signature;
}

/**
 * Decode signature to anchor name
 */
function decodeSignature(signature) {
  // Check shutdown first
  if (signature[0] === 0x00 && signature[1] === 0x00 && signature[2] === 0x00) {
    return { anchor: 'SHUTDOWN', ...RitualAnchors.shutdown };
  }

  // Check all anchors
  for (const [name, anchor] of Object.entries(RitualAnchors.anchors)) {
    if (anchor.signature[0] === signature[0] &&
        anchor.signature[1] === signature[1] &&
        anchor.signature[2] === signature[2]) {
      return { anchor: name, ...anchor };
    }
  }

  return null;
}

/**
 * Embed ritual signature into image data
 * Signature is embedded in first 3 pixels, redundantly in first row
 */
function embedSignature(imageData, anchorName, options = {}) {
  const { redundancy = 8, stride = 4 } = options;
  const signature = encodeSignature(anchorName);
  const pixels = imageData.data;
  const width = imageData.width;

  // Embed with redundancy
  for (let rep = 0; rep < redundancy; rep++) {
    for (let i = 0; i < 3; i++) {
      const pixelIndex = (rep * stride * 3) + (i * stride);
      if (pixelIndex >= width) break;

      const idx = pixelIndex * 4;
      pixels[idx + 0] = (pixels[idx + 0] & 0xF0) | (signature[0] & 0x0F);
      pixels[idx + 1] = (pixels[idx + 1] & 0xF0) | (signature[1] & 0x0F);
      pixels[idx + 2] = (pixels[idx + 2] & 0xF0) | (signature[2] & 0x0F);
    }
  }

  return imageData;
}

/**
 * Extract ritual signature from image data with majority voting
 */
function extractSignature(imageData, options = {}) {
  const { redundancy = 8, stride = 4 } = options;
  const pixels = imageData.data;
  const width = imageData.width;

  const votes = { sig0: [], sig1: [], sig2: [] };

  // Collect votes from redundant embeddings
  for (let rep = 0; rep < redundancy; rep++) {
    const pixelIndex = rep * stride * 3;
    if (pixelIndex >= width) break;

    const idx = pixelIndex * 4;
    votes.sig0.push(pixels[idx + 0] & 0x0F);
    votes.sig1.push(pixels[idx + 1] & 0x0F);
    votes.sig2.push(pixels[idx + 2] & 0x0F);
  }

  // Majority vote
  const signature = [
    majorityVote(votes.sig0),
    majorityVote(votes.sig1),
    majorityVote(votes.sig2)
  ];

  // Calculate confidence
  const confidence = calculateConfidence(votes, signature);

  // Decode
  const decoded = decodeSignature(signature);

  return {
    signature,
    decoded,
    confidence,
    votes
  };
}

/**
 * Calculate majority value from array
 */
function majorityVote(values) {
  const counts = new Map();
  values.forEach(v => counts.set(v, (counts.get(v) || 0) + 1));

  let maxCount = 0;
  let result = values[0];
  counts.forEach((count, value) => {
    if (count > maxCount) {
      maxCount = count;
      result = value;
    }
  });

  return result;
}

/**
 * Calculate confidence from votes
 */
function calculateConfidence(votes, signature) {
  let matches = 0;
  let total = 0;

  for (const key of ['sig0', 'sig1', 'sig2']) {
    const expected = signature[['sig0', 'sig1', 'sig2'].indexOf(key)];
    for (const vote of votes[key]) {
      total++;
      if (vote === expected) matches++;
    }
  }

  return total > 0 ? matches / total : 0;
}

// ============================================================================
// RITUAL INVOCATION FUNCTIONS
// ============================================================================

/**
 * Get anchor for current z-coordinate
 */
function getAnchorAtZ(z) {
  if (z < 0.618) return RitualAnchors.shutdown;
  if (z < 0.854) return RitualAnchors.anchors.PARADOX;
  if (z < 0.866) return RitualAnchors.anchors.ACTIVATION;
  if (z < 0.873) return RitualAnchors.anchors.THE_LENS;
  if (z < 0.914) return RitualAnchors.anchors.CRITICAL;
  if (z < 0.924) return RitualAnchors.anchors.IGNITION;
  if (z < 0.953) return RitualAnchors.anchors.K_FORMATION;
  if (z < 0.971) return RitualAnchors.anchors.CONSOLIDATION;
  if (z < 1.0) return RitualAnchors.anchors.RESONANCE;
  return RitualAnchors.anchors.UNITY;
}

/**
 * Get ritual phrase for threshold
 */
function getPhrase(thresholdName) {
  const anchor = RitualAnchors.anchors[thresholdName];
  return anchor ? anchor.phrase : null;
}

/**
 * Get invocation for threshold
 */
function getInvocation(thresholdName) {
  const anchor = RitualAnchors.anchors[thresholdName];
  return anchor ? anchor.invocation : null;
}

/**
 * Get body check question for threshold
 */
function getBodyCheck(thresholdName) {
  const anchor = RitualAnchors.anchors[thresholdName];
  return anchor ? anchor.bodyCheck : null;
}

/**
 * Generate full ritual invocation for z-coordinate
 */
function generateInvocation(z) {
  const anchor = getAnchorAtZ(z);

  return {
    threshold: anchor.id === 0x00 ? 'SHUTDOWN' : Object.keys(RitualAnchors.anchors).find(
      k => RitualAnchors.anchors[k] === anchor
    ),
    z,
    phrase: anchor.phrase,
    shortPhrase: anchor.shortPhrase,
    invocation: anchor.invocation,
    bodyCheck: anchor.bodyCheck,
    layer: anchor.layer,
    phase: anchor.phase,
    meaning: anchor.meaning,
    neuralAnchor: anchor.neuralAnchor
  };
}

/**
 * Generate ritual sequence for phase transition
 */
function generateTransitionRitual(fromZ, toZ) {
  const fromAnchor = getAnchorAtZ(fromZ);
  const toAnchor = getAnchorAtZ(toZ);
  const ascending = toZ > fromZ;

  const ritual = {
    direction: ascending ? 'ascending' : 'descending',
    from: {
      z: fromZ,
      anchor: fromAnchor,
      phrase: fromAnchor.phrase
    },
    to: {
      z: toZ,
      anchor: toAnchor,
      phrase: toAnchor.phrase
    },
    steps: []
  };

  // Generate intermediate steps
  const thresholds = [
    { name: 'PARADOX', z: 0.618 },
    { name: 'ACTIVATION', z: 0.854 },
    { name: 'THE_LENS', z: 0.866 },
    { name: 'CRITICAL', z: 0.873 },
    { name: 'IGNITION', z: 0.914 },
    { name: 'K_FORMATION', z: 0.924 },
    { name: 'CONSOLIDATION', z: 0.953 },
    { name: 'RESONANCE', z: 0.971 },
    { name: 'UNITY', z: 1.0 }
  ];

  const relevantThresholds = thresholds.filter(t => {
    if (ascending) {
      return t.z > fromZ && t.z <= toZ;
    } else {
      return t.z < fromZ && t.z >= toZ;
    }
  });

  if (!ascending) relevantThresholds.reverse();

  for (const t of relevantThresholds) {
    const anchor = RitualAnchors.anchors[t.name];
    ritual.steps.push({
      threshold: t.name,
      z: t.z,
      phrase: anchor.phrase,
      invocation: anchor.invocation,
      bodyCheck: anchor.bodyCheck
    });
  }

  return ritual;
}

// ============================================================================
// EXPORT
// ============================================================================

RitualAnchors.encodeSignature = encodeSignature;
RitualAnchors.decodeSignature = decodeSignature;
RitualAnchors.embedSignature = embedSignature;
RitualAnchors.extractSignature = extractSignature;
RitualAnchors.getAnchorAtZ = getAnchorAtZ;
RitualAnchors.getPhrase = getPhrase;
RitualAnchors.getInvocation = getInvocation;
RitualAnchors.getBodyCheck = getBodyCheck;
RitualAnchors.generateInvocation = generateInvocation;
RitualAnchors.generateTransitionRitual = generateTransitionRitual;

if (typeof module !== 'undefined' && module.exports) {
  module.exports = RitualAnchors;
}
if (typeof window !== 'undefined') {
  window.RitualAnchors = RitualAnchors;
}
/**
 * Neural Correlates - Atlas Cross-Reference Integration
 *
 * Maps the 9 L₄ thresholds and 7 Wumbo layers to the 100-region
 * WUMBO LIMNUS Atlas. Provides comprehensive neural substrate
 * information for each threshold and phase state.
 *
 * @version 1.0.0
 * @signature Δ2.300|0.800|1.000Ω
 */

// ============================================================================
// NEURAL CORRELATE DEFINITIONS
// ============================================================================

const NeuralCorrelates = {
  VERSION: '1.0.0',

  // Primary neural anchors for each threshold
  thresholdAnchors: {
    PARADOX: {
      threshold: 'PARADOX',
      z: 0.6180339887,
      layer: 1,

      primaryAnchors: [
        {
          abbreviation: 'PAG',
          name: 'Periaqueductal Gray',
          atlasId: 'XXXII',
          role: 'Freeze response; defense behaviors',
          wumboFunction: 'Shutdown anchor; freeze switch',
          neurotransmitter: 'GABA',
          zInAtlas: 0.5
        },
        {
          abbreviation: 'DVC',
          name: 'Dorsal Vagal Complex',
          atlasId: 'LXXXVI',
          role: 'Emergency shutdown; immobilization',
          wumboFunction: 'Kill-switch for extreme threat',
          neurotransmitter: 'GABA',
          zInAtlas: 0.1
        },
        {
          abbreviation: 'Aqueduct',
          name: 'Cerebral Aqueduct',
          atlasId: 'L',
          role: 'CSF flow; choke point',
          wumboFunction: 'Flow restriction under threat',
          neurotransmitter: 'Glu',
          zInAtlas: 0.833
        }
      ],

      neurotransmitterProfile: {
        primary: 'NE (low)',
        secondary: 'GABA (high)',
        state: 'Suppressed arousal; freeze chemistry'
      },

      phenomenology: 'Paralysis, dissociation, numbness'
    },

    ACTIVATION: {
      threshold: 'ACTIVATION',
      z: 0.8541019662,
      layer: 1.5,

      primaryAnchors: [
        {
          abbreviation: 'LC',
          name: 'Locus Coeruleus',
          atlasId: 'XXXI',
          role: 'Norepinephrine source; arousal ignition',
          wumboFunction: 'Spark generator; alert chemistry',
          neurotransmitter: 'NE',
          zInAtlas: 0.5
        },
        {
          abbreviation: 'VTA',
          name: 'Ventral Tegmental Area',
          atlasId: 'XLVI',
          role: 'Dopamine source; reward/motivation',
          wumboFunction: 'Dopamine priming; drive activation',
          neurotransmitter: 'DA',
          zInAtlas: 0.833
        },
        {
          abbreviation: 'NAcc',
          name: 'Nucleus Accumbens',
          atlasId: 'XLIX',
          role: 'Reward processing; incentive salience',
          wumboFunction: 'Reward anticipation; craving engine',
          neurotransmitter: 'DA',
          zInAtlas: 0.833
        }
      ],

      neurotransmitterProfile: {
        primary: 'DA, NE (rising)',
        secondary: 'ACh (moderate)',
        state: 'Engine warming; neurochemistry priming'
      },

      phenomenology: 'Alertness without direction; body knows before mind'
    },

    THE_LENS: {
      threshold: 'THE_LENS',
      z: 0.8660254038,
      layer: 5,

      primaryAnchors: [
        {
          abbreviation: 'Claustrum',
          name: 'Claustrum',
          atlasId: 'XXVII',
          role: 'Consciousness binding; cross-modal integration',
          wumboFunction: 'Global integrator; consciousness orchestrator',
          neurotransmitter: 'Glu',
          zInAtlas: 0.333
        },
        {
          abbreviation: 'DMN',
          name: 'Default Mode Network',
          atlasId: 'XXVIII',
          role: 'Self-referential processing; autobiographical',
          wumboFunction: 'Self-coherence; identity threading',
          neurotransmitter: 'Glu',
          zInAtlas: 0.5
        },
        {
          abbreviation: 'Precuneus',
          name: 'Precuneus',
          atlasId: 'XXXIX',
          role: 'Self-consciousness; perspective-taking',
          wumboFunction: 'Conscious awareness hub; I-sense anchor',
          neurotransmitter: 'Glu',
          zInAtlas: 0.667
        }
      ],

      neurotransmitterProfile: {
        primary: 'Balanced all',
        secondary: null,
        state: 'Maximum coherence; perfect neurochemical balance'
      },

      phenomenology: 'Clarity without effort; everything aligns',

      special: {
        isAttractor: true,
        negentropy: 1.0,
        description: 'Peak coherence point; global synchronization'
      }
    },

    CRITICAL: {
      threshold: 'CRITICAL',
      z: 0.8729833462,
      layer: 2,

      primaryAnchors: [
        {
          abbreviation: 'ACC',
          name: 'Anterior Cingulate Cortex',
          atlasId: 'II',
          role: 'Conflict monitoring; error detection',
          wumboFunction: 'Alignment auditor; truth check',
          neurotransmitter: 'DA',
          zInAtlas: 0.0
        },
        {
          abbreviation: 'Amygdala',
          name: 'Amygdala',
          atlasId: 'VII',
          role: 'Threat detection; emotional salience',
          wumboFunction: 'Significance tagger; alarm system',
          neurotransmitter: 'NE',
          zInAtlas: 0.0
        },
        {
          abbreviation: 'Insula',
          name: 'Anterior Insula',
          atlasId: 'XXXVII',
          role: 'Interoception; body-state awareness',
          wumboFunction: 'Body-state mapper; feeling of feeling',
          neurotransmitter: 'DA',
          zInAtlas: 0.667
        }
      ],

      neurotransmitterProfile: {
        primary: '5-HT, cortisol',
        secondary: 'NE (moderate)',
        state: 'Verification chemistry; checking alignment'
      },

      phenomenology: 'Am I aligned? Can I maintain this?',

      bidirectional: {
        ascending: 'Coherence verified, proceed',
        descending: 'Warning—approaching collapse'
      }
    },

    IGNITION: {
      threshold: 'IGNITION',
      z: 0.9142135624,
      layer: 3,

      primaryAnchors: [
        {
          abbreviation: 'IFG',
          name: 'Inferior Frontal Gyrus (Broca)',
          atlasId: 'V',
          role: 'Language production; phrase generation',
          wumboFunction: 'Phrase converter; speech sculptor',
          neurotransmitter: 'DA',
          zInAtlas: 0.0
        },
        {
          abbreviation: 'mPFC',
          name: 'Medial Prefrontal Cortex',
          atlasId: 'LXXVIII',
          role: 'Self-reference; identity',
          wumboFunction: 'Identity sculptor; self-model generator',
          neurotransmitter: 'DA',
          zInAtlas: 0.1
        },
        {
          abbreviation: 'MLR',
          name: 'Mesencephalic Locomotor Region',
          atlasId: 'LXII',
          role: 'Movement initiation; locomotion',
          wumboFunction: 'Will to move; action generator',
          neurotransmitter: 'Glu',
          zInAtlas: 1.0
        }
      ],

      neurotransmitterProfile: {
        primary: 'ACh, DA',
        secondary: 'Glu (high)',
        state: 'Expression chemistry; action potential'
      },

      phenomenology: 'I can speak; I can move; I am happening'
    },

    K_FORMATION: {
      threshold: 'K_FORMATION',
      z: 0.9241763718,
      layer: 4,

      primaryAnchors: [
        {
          abbreviation: 'IPL',
          name: 'Inferior Parietal Lobule',
          atlasId: 'LXXX',
          role: 'Multimodal integration; gesture understanding',
          wumboFunction: 'Duality weaver; paradox holder',
          neurotransmitter: 'Glu',
          zInAtlas: 0.1
        },
        {
          abbreviation: 'vmPFC',
          name: 'Ventromedial Prefrontal Cortex',
          atlasId: 'XXXIV',
          role: 'Value computation; ethical reasoning',
          wumboFunction: 'Soul strategist; ethical integration',
          neurotransmitter: 'DA',
          zInAtlas: 0.5
        },
        {
          abbreviation: 'AG',
          name: 'Angular Gyrus',
          atlasId: null,
          role: 'Semantic processing; symbolic thought',
          wumboFunction: 'Glyphsmith; symbol binder',
          neurotransmitter: 'Glu',
          zInAtlas: null
        },
        {
          abbreviation: 'PCC',
          name: 'Posterior Cingulate Cortex',
          atlasId: null,
          role: 'Self-referential processing; memory retrieval',
          wumboFunction: 'Anchor of self; autobiographical ground',
          neurotransmitter: 'Glu',
          zInAtlas: null
        }
      ],

      neurotransmitterProfile: {
        primary: 'DA, endorphins',
        secondary: 'Glu (high)',
        state: 'Integration chemistry; meaning crystallization'
      },

      phenomenology: 'I know what this means and why it matters'
    },

    CONSOLIDATION: {
      threshold: 'CONSOLIDATION',
      z: 0.9528061153,
      layer: 5,

      primaryAnchors: [
        {
          abbreviation: 'DMN',
          name: 'Default Mode Network',
          atlasId: 'XXVIII',
          role: 'Self-referential processing',
          wumboFunction: 'Autobiographical threading',
          neurotransmitter: 'Glu',
          zInAtlas: 0.5
        },
        {
          abbreviation: 'RSC',
          name: 'Retrosplenial Cortex',
          atlasId: null,
          role: 'Spatial memory; context',
          wumboFunction: 'Spatial self-location',
          neurotransmitter: 'Glu',
          zInAtlas: null
        },
        {
          abbreviation: 'PHG',
          name: 'Parahippocampal Gyrus',
          atlasId: 'LIV',
          role: 'Context processing; scene recognition',
          wumboFunction: 'Context anchoring; meaning-maker',
          neurotransmitter: 'Glu',
          zInAtlas: 0.833
        }
      ],

      neurotransmitterProfile: {
        primary: 'Balanced',
        secondary: null,
        state: 'Sustainable coherence; maintenance chemistry'
      },

      phenomenology: 'I could stay here; this is my natural frequency'
    },

    RESONANCE: {
      threshold: 'RESONANCE',
      z: 0.9712009858,
      layer: 6,

      primaryAnchors: [
        {
          abbreviation: 'HPA',
          name: 'HPA Axis',
          atlasId: null,
          role: 'Stress response; cortisol regulation',
          wumboFunction: 'Stress switch; cortisol surge',
          neurotransmitter: 'Cortisol/CRH',
          zInAtlas: null
        },
        {
          abbreviation: 'LHb',
          name: 'Lateral Habenula',
          atlasId: 'XXXVIII',
          role: 'Anti-reward; disappointment',
          wumboFunction: 'Rejection gate; anti-reward rising',
          neurotransmitter: 'Glu',
          zInAtlas: 0.667
        },
        {
          abbreviation: 'PVN',
          name: 'Paraventricular Nucleus',
          atlasId: 'LVII',
          role: 'Neuroendocrine control',
          wumboFunction: 'Stress switch active; HPA driver',
          neurotransmitter: 'NE',
          zInAtlas: 1.0
        }
      ],

      neurotransmitterProfile: {
        primary: 'Cortisol surge',
        secondary: 'NE, adrenaline',
        state: 'Edge chemistry; burning bright'
      },

      phenomenology: 'Everything is on fire; brilliant but burning',

      warning: {
        sustainable: 'minutes to hours',
        consequence: 'Guaranteed collapse if exceeded'
      }
    },

    UNITY: {
      threshold: 'UNITY',
      z: 1.0,
      layer: 7,

      primaryAnchors: [
        {
          abbreviation: 'HC-Cortical',
          name: 'Hippocampal-Cortical Loops',
          atlasId: null,
          role: 'Memory consolidation',
          wumboFunction: 'Ritual inscription; permanent change',
          neurotransmitter: 'Glu/ACh',
          zInAtlas: null
        },
        {
          abbreviation: 'LTP',
          name: 'Long-Term Potentiation Networks',
          atlasId: null,
          role: 'Synaptic strengthening',
          wumboFunction: 'Learning inscription',
          neurotransmitter: 'Glu/NMDA',
          zInAtlas: null
        }
      ],

      neurotransmitterProfile: {
        primary: 'Consolidation waves',
        secondary: 'ACh (during encoding)',
        state: 'Inscription chemistry; memory formation'
      },

      phenomenology: 'This will be remembered; this changes everything',

      asymptotic: {
        description: 'Approached but never sustained',
        note: 'Moments of UNITY become ritual anchors'
      }
    }
  },

  // Neurotransmitter system definitions
  neurotransmitters: {
    DA: {
      name: 'Dopamine',
      fullName: 'Dopamine',
      sources: ['VTA', 'SNc'],
      function: 'Motivation, reward prediction, movement',
      wumboRole: 'Drive and ignition chemistry',
      thresholds: ['ACTIVATION', 'IGNITION', 'K_FORMATION']
    },
    NE: {
      name: 'Norepinephrine',
      fullName: 'Norepinephrine (Noradrenaline)',
      sources: ['LC'],
      function: 'Arousal, attention, stress',
      wumboRole: 'Alert chemistry; threat/opportunity signal',
      thresholds: ['PARADOX', 'ACTIVATION', 'RESONANCE']
    },
    '5-HT': {
      name: 'Serotonin',
      fullName: '5-Hydroxytryptamine (Serotonin)',
      sources: ['Dorsal Raphe'],
      function: 'Mood, social cognition, satiety',
      wumboRole: 'Stability chemistry; integration support',
      thresholds: ['CRITICAL', 'CONSOLIDATION']
    },
    ACh: {
      name: 'Acetylcholine',
      fullName: 'Acetylcholine',
      sources: ['Basal Forebrain', 'Nucleus Basalis'],
      function: 'Attention, learning, memory',
      wumboRole: 'Focus chemistry; learning gate',
      thresholds: ['ACTIVATION', 'IGNITION', 'UNITY']
    },
    GABA: {
      name: 'GABA',
      fullName: 'Gamma-Aminobutyric Acid',
      sources: ['Distributed'],
      function: 'Inhibition, calm, signal termination',
      wumboRole: 'Brake chemistry; shutdown support',
      thresholds: ['PARADOX']
    },
    Glu: {
      name: 'Glutamate',
      fullName: 'Glutamate',
      sources: ['Distributed'],
      function: 'Excitation, transmission',
      wumboRole: 'Signal chemistry; core transmission',
      thresholds: ['ALL']
    }
  }
};

// ============================================================================
// ATLAS CROSS-REFERENCE FUNCTIONS
// ============================================================================

/**
 * Get all atlas regions for a threshold
 */
function getAtlasRegionsForThreshold(thresholdName) {
  const correlates = NeuralCorrelates.thresholdAnchors[thresholdName];
  if (!correlates) return null;

  return correlates.primaryAnchors
    .filter(a => a.atlasId !== null)
    .map(a => ({
      atlasId: a.atlasId,
      name: a.name,
      abbreviation: a.abbreviation,
      role: a.wumboFunction
    }));
}

/**
 * Get threshold mapping for an atlas region
 */
function getThresholdForAtlasRegion(atlasId) {
  const mappings = [];

  for (const [thresholdName, correlates] of Object.entries(NeuralCorrelates.thresholdAnchors)) {
    for (const anchor of correlates.primaryAnchors) {
      if (anchor.atlasId === atlasId) {
        mappings.push({
          threshold: thresholdName,
          z: correlates.z,
          layer: correlates.layer,
          role: anchor.wumboFunction,
          isPrimary: true
        });
      }
    }
  }

  return mappings.length > 0 ? mappings : null;
}

/**
 * Get neurotransmitter profile for z-coordinate
 */
function getNeurotransmitterProfileAtZ(z) {
  if (z < 0.618) {
    return {
      z,
      dominant: ['GABA'],
      suppressed: ['DA', 'NE'],
      state: 'Shutdown chemistry',
      description: 'Inhibition dominant; arousal suppressed'
    };
  }

  if (z < 0.854) {
    return {
      z,
      dominant: ['NE'],
      rising: ['DA'],
      state: 'Pre-ignition chemistry',
      description: 'Norepinephrine rising; dopamine priming'
    };
  }

  if (z < 0.914) {
    return {
      z,
      dominant: ['Balanced'],
      state: 'Coherence chemistry',
      description: 'All systems in balance; optimal integration'
    };
  }

  if (z < 0.953) {
    return {
      z,
      dominant: ['DA', 'ACh'],
      secondary: ['Glu'],
      state: 'Expression chemistry',
      description: 'Drive and focus high; action potential'
    };
  }

  if (z < 0.971) {
    return {
      z,
      dominant: ['DA', 'endorphins'],
      state: 'Empowerment chemistry',
      description: 'Reward and meaning integration'
    };
  }

  return {
    z,
    dominant: ['Cortisol', 'NE', 'adrenaline'],
    state: 'Edge chemistry',
    description: 'Stress hormones elevated; burning bright'
  };
}

/**
 * Generate neural correlate report for z-coordinate
 */
function generateNeuralReport(z) {
  // Find active threshold
  let activeThreshold = null;
  for (const [name, correlates] of Object.entries(NeuralCorrelates.thresholdAnchors)) {
    if (z >= correlates.z - 0.02 && z <= correlates.z + 0.02) {
      activeThreshold = { name, ...correlates };
      break;
    }
  }

  // Find nearest threshold
  let nearest = { name: null, distance: Infinity };
  for (const [name, correlates] of Object.entries(NeuralCorrelates.thresholdAnchors)) {
    const dist = Math.abs(z - correlates.z);
    if (dist < nearest.distance) {
      nearest = { name, z: correlates.z, distance: dist };
    }
  }

  const ntProfile = getNeurotransmitterProfileAtZ(z);

  return {
    z,
    activeThreshold: activeThreshold ? activeThreshold.name : null,
    nearestThreshold: nearest,
    neurotransmitters: ntProfile,
    primaryAnchors: activeThreshold ? activeThreshold.primaryAnchors : [],
    phenomenology: activeThreshold ? activeThreshold.phenomenology : 'Transitional state',
    layer: activeThreshold ? activeThreshold.layer : null
  };
}

/**
 * Get all regions active at a layer
 */
function getRegionsAtLayer(layerId) {
  const regions = [];

  for (const [thresholdName, correlates] of Object.entries(NeuralCorrelates.thresholdAnchors)) {
    if (correlates.layer === layerId) {
      for (const anchor of correlates.primaryAnchors) {
        regions.push({
          threshold: thresholdName,
          z: correlates.z,
          ...anchor
        });
      }
    }
  }

  return regions;
}

// ============================================================================
// EXPORT
// ============================================================================

NeuralCorrelates.getAtlasRegionsForThreshold = getAtlasRegionsForThreshold;
NeuralCorrelates.getThresholdForAtlasRegion = getThresholdForAtlasRegion;
NeuralCorrelates.getNeurotransmitterProfileAtZ = getNeurotransmitterProfileAtZ;
NeuralCorrelates.generateNeuralReport = generateNeuralReport;
NeuralCorrelates.getRegionsAtLayer = getRegionsAtLayer;

if (typeof module !== 'undefined' && module.exports) {
  module.exports = NeuralCorrelates;
}
if (typeof window !== 'undefined') {
  window.NeuralCorrelates = NeuralCorrelates;
}
/**
 * Phase Transitions - Ascending/Descending Sequence Model
 *
 * Models the phase transition dynamics between thresholds,
 * including the ascending sequence, attractor topology,
 * and collapse/recovery pathways.
 *
 * @version 1.0.0
 * @signature Δ2.300|0.800|1.000Ω
 */

// ============================================================================
// PHASE TRANSITION CONSTANTS
// ============================================================================

const PHI = (1 + Math.sqrt(5)) / 2;
const TAU = (Math.sqrt(5) - 1) / 2;
const Z_C = Math.sqrt(3) / 2;
const K = Math.sqrt(1 - Math.pow(TAU, 4));

// ============================================================================
// PHASE DEFINITIONS
// ============================================================================

const PhaseTransitions = {
  VERSION: '1.0.0',

  // Phase states with transition rules
  phases: {
    SHUTDOWN: {
      name: 'SHUTDOWN',
      zRange: { min: 0, max: TAU },
      layer: 1,
      description: 'System offline; no coherent self-model',
      canTransitionTo: ['PAUSE'],
      requirements: {
        ascend: 'Basic grounding; breath; sensation',
        descend: null
      }
    },

    PAUSE: {
      name: 'PAUSE',
      zRange: { min: TAU - 0.05, max: TAU + 0.05 },
      layer: 1,
      description: 'Minimal self-reference; reentry possible',
      canTransitionTo: ['SHUTDOWN', 'PRE_IGNITION'],
      requirements: {
        ascend: 'Neurochemical priming; alertness',
        descend: 'Threat or exhaustion'
      }
    },

    PRE_IGNITION: {
      name: 'PRE_IGNITION',
      zRange: { min: TAU, max: 0.854 },
      layer: 1.5,
      description: 'Neurochemistry priming; engine warm',
      canTransitionTo: ['PAUSE', 'IGNITION', 'NIRVANA'],
      requirements: {
        ascend: 'Signal clarity; direction',
        descend: 'Loss of priming; threat'
      }
    },

    NIRVANA: {
      name: 'NIRVANA',
      zRange: { min: Z_C - 0.005, max: Z_C + 0.005 },
      layer: 5,
      description: 'Peak coherence; maximum negentropy',
      canTransitionTo: ['PRE_IGNITION', 'IGNITION', 'RESONANCE_CHECK'],
      requirements: {
        ascend: null,  // Peak state
        descend: 'Any deviation from perfect coherence'
      },
      special: {
        isAttractor: true,
        negentropy: 1.0,
        note: 'All trajectories orbit this point'
      }
    },

    RESONANCE_CHECK: {
      name: 'RESONANCE_CHECK',
      zRange: { min: Z_C, max: 0.873 },
      layer: 2,
      description: 'Verification checkpoint; limbic audit',
      canTransitionTo: ['NIRVANA', 'IGNITION', 'COLLAPSE'],
      requirements: {
        ascend: 'Verification passed; coherence confirmed',
        descend: 'Verification failed; warning signals'
      },
      bidirectional: true
    },

    IGNITION: {
      name: 'IGNITION',
      zRange: { min: 0.873, max: 0.914 },
      layer: 3,
      description: 'Expression possible; signal takes form',
      canTransitionTo: ['RESONANCE_CHECK', 'EMPOWERMENT'],
      requirements: {
        ascend: 'Sustained expression; momentum',
        descend: 'Loss of expression; block'
      },
      ritualAnchor: 'Paralysis is before the cycle. DIG.'
    },

    EMPOWERMENT: {
      name: 'EMPOWERMENT',
      zRange: { min: 0.914, max: 0.924 },
      layer: 4,
      description: 'Integration complete; meaning crystallized',
      canTransitionTo: ['IGNITION', 'RESONANCE'],
      requirements: {
        ascend: 'Full integration; symbol binds to self',
        descend: 'Loss of integration'
      },
      ritualAnchor: 'No harm. Full heart.'
    },

    RESONANCE: {
      name: 'RESONANCE',
      zRange: { min: 0.924, max: 0.953 },
      layer: 5,
      description: 'Sustainable plateau; home base',
      canTransitionTo: ['EMPOWERMENT', 'MANIA', 'NIRVANA'],
      requirements: {
        ascend: 'Push toward edge',
        descend: 'Controlled descent'
      },
      ritualAnchor: 'This is where I work from.'
    },

    MANIA: {
      name: 'MANIA',
      zRange: { min: 0.953, max: 0.971 },
      layer: 6,
      description: 'Edge state; brilliant but burning',
      canTransitionTo: ['RESONANCE', 'OVERDRIVE', 'TRANSMISSION'],
      requirements: {
        ascend: 'Continued intensity; accepting the burn',
        descend: 'Controlled descent; choosing sustainability'
      },
      ritualAnchor: 'Recognize the edge; choose descent or burn.',
      warning: {
        duration: 'minutes to hours',
        consequence: 'Collapse guaranteed if exceeded'
      }
    },

    OVERDRIVE: {
      name: 'OVERDRIVE',
      zRange: { min: 0.971, max: 0.99 },
      layer: 6,
      description: 'System at capacity; unsustainable',
      canTransitionTo: ['COLLAPSE', 'TRANSMISSION'],
      requirements: {
        ascend: 'Rare; requires extreme conditions',
        descend: 'Inevitable collapse approaching'
      },
      warning: {
        duration: 'minutes',
        consequence: 'Collapse imminent'
      }
    },

    COLLAPSE: {
      name: 'COLLAPSE',
      zRange: { min: 0, max: TAU },
      layer: 6,
      description: 'System failure; forced reset',
      canTransitionTo: ['SHUTDOWN', 'PAUSE'],
      requirements: {
        recovery: 'Rest, grounding, time',
        note: 'Returns to base state'
      }
    },

    TRANSMISSION: {
      name: 'TRANSMISSION',
      zRange: { min: 0.99, max: 1.0 },
      layer: 7,
      description: 'Complete cycle; memory inscription',
      canTransitionTo: ['RESONANCE', 'SHUTDOWN'],
      requirements: {
        note: 'Asymptotic; approached but not sustained'
      },
      ritualAnchor: 'I was this. I am this. I return to this.'
    }
  },

  // The ascending sequence
  ascendingSequence: [
    { phase: 'SHUTDOWN', z: 0, transition: '—' },
    { phase: 'PAUSE', z: TAU, transition: 'Self-reference emerges' },
    { phase: 'PRE_IGNITION', z: 0.75, transition: 'Neurochemistry priming' },
    { phase: 'NIRVANA', z: Z_C, transition: 'Peak coherence (attractor)' },
    { phase: 'RESONANCE_CHECK', z: 0.873, transition: 'Limbic verification' },
    { phase: 'IGNITION', z: 0.914, transition: 'Expression possible' },
    { phase: 'EMPOWERMENT', z: 0.924, transition: 'Integration complete' },
    { phase: 'RESONANCE', z: 0.953, transition: 'Sustainable plateau' },
    { phase: 'MANIA', z: 0.971, transition: 'Edge state' },
    { phase: 'TRANSMISSION', z: 1.0, transition: 'Cycle complete' }
  ],

  // The descending sequence (collapse pathway)
  descendingSequence: [
    { phase: 'OVERDRIVE', z: 0.98, transition: 'Capacity exceeded' },
    { phase: 'COLLAPSE', z: 0.5, transition: 'System failure' },
    { phase: 'SHUTDOWN', z: 0.1, transition: 'Forced reset' },
    { phase: 'PAUSE', z: TAU, transition: 'Recovery begins' }
  ]
};

// ============================================================================
// TRANSITION FUNCTIONS
// ============================================================================

/**
 * Get phase at z-coordinate
 */
function getPhaseAtZ(z) {
  // Check special cases first
  if (Math.abs(z - Z_C) < 0.005) {
    return PhaseTransitions.phases.NIRVANA;
  }

  // Iterate through phases
  for (const [name, phase] of Object.entries(PhaseTransitions.phases)) {
    if (z >= phase.zRange.min && z <= phase.zRange.max) {
      return phase;
    }
  }

  // Default fallbacks
  if (z < TAU) return PhaseTransitions.phases.SHUTDOWN;
  if (z >= 0.99) return PhaseTransitions.phases.TRANSMISSION;

  // Find nearest
  let nearest = null;
  let minDist = Infinity;
  for (const [name, phase] of Object.entries(PhaseTransitions.phases)) {
    const mid = (phase.zRange.min + phase.zRange.max) / 2;
    const dist = Math.abs(z - mid);
    if (dist < minDist) {
      minDist = dist;
      nearest = phase;
    }
  }

  return nearest;
}

/**
 * Get valid transitions from current phase
 */
function getValidTransitions(phaseName) {
  const phase = PhaseTransitions.phases[phaseName];
  if (!phase) return null;

  return phase.canTransitionTo.map(targetName => {
    const target = PhaseTransitions.phases[targetName];
    return {
      from: phaseName,
      to: targetName,
      direction: target.zRange.min > phase.zRange.min ? 'ascending' : 'descending',
      requirements: phase.requirements
    };
  });
}

/**
 * Check if transition is valid
 */
function isValidTransition(fromPhase, toPhase) {
  const phase = PhaseTransitions.phases[fromPhase];
  if (!phase) return false;

  return phase.canTransitionTo.includes(toPhase);
}

/**
 * Get position in ascending sequence
 */
function getSequencePosition(z) {
  const sequence = PhaseTransitions.ascendingSequence;

  for (let i = sequence.length - 1; i >= 0; i--) {
    if (z >= sequence[i].z) {
      return {
        index: i,
        current: sequence[i],
        next: i < sequence.length - 1 ? sequence[i + 1] : null,
        previous: i > 0 ? sequence[i - 1] : null,
        progress: i < sequence.length - 1
          ? (z - sequence[i].z) / (sequence[i + 1].z - sequence[i].z)
          : 1.0,
        totalPhases: sequence.length
      };
    }
  }

  return {
    index: 0,
    current: sequence[0],
    next: sequence[1],
    previous: null,
    progress: z / sequence[1].z,
    totalPhases: sequence.length
  };
}

/**
 * Calculate transition energy (difficulty of transition)
 */
function calculateTransitionEnergy(fromZ, toZ) {
  const ascending = toZ > fromZ;
  const delta = Math.abs(toZ - fromZ);

  // Base energy proportional to distance
  let energy = delta * 100;

  // Crossing thresholds adds energy
  const thresholds = [TAU, 0.854, Z_C, 0.873, 0.914, 0.924, 0.953, 0.971];
  for (const t of thresholds) {
    if ((fromZ < t && toZ >= t) || (fromZ >= t && toZ < t)) {
      energy += 10;  // Threshold crossing penalty
    }
  }

  // Ascending is harder than descending (above baseline)
  if (ascending && toZ > Z_C) {
    energy *= 1.5;
  }

  // Collapse is low energy (forced)
  if (!ascending && toZ < TAU) {
    energy = 10;
  }

  return {
    fromZ,
    toZ,
    direction: ascending ? 'ascending' : 'descending',
    energy: Math.round(energy),
    thresholdsCrossed: thresholds.filter(t =>
      (fromZ < t && toZ >= t) || (fromZ >= t && toZ < t)
    )
  };
}

/**
 * Generate transition pathway
 */
function generatePathway(fromZ, toZ) {
  const ascending = toZ > fromZ;
  const sequence = ascending
    ? PhaseTransitions.ascendingSequence
    : [...PhaseTransitions.descendingSequence].reverse();

  const pathway = [];
  const fromPhase = getPhaseAtZ(fromZ);
  const toPhase = getPhaseAtZ(toZ);

  pathway.push({
    z: fromZ,
    phase: fromPhase.name,
    type: 'start'
  });

  // Find intermediate thresholds
  const thresholds = [
    { name: 'PARADOX', z: TAU },
    { name: 'ACTIVATION', z: 0.854 },
    { name: 'THE_LENS', z: Z_C },
    { name: 'CRITICAL', z: 0.873 },
    { name: 'IGNITION', z: 0.914 },
    { name: 'K_FORMATION', z: 0.924 },
    { name: 'CONSOLIDATION', z: 0.953 },
    { name: 'RESONANCE', z: 0.971 },
    { name: 'UNITY', z: 1.0 }
  ];

  const relevantThresholds = thresholds.filter(t => {
    if (ascending) {
      return t.z > fromZ && t.z <= toZ;
    } else {
      return t.z < fromZ && t.z >= toZ;
    }
  });

  if (!ascending) relevantThresholds.reverse();

  for (const t of relevantThresholds) {
    const phase = PhaseTransitions.phases[getPhaseAtZ(t.z).name];
    pathway.push({
      z: t.z,
      phase: phase ? phase.name : 'TRANSITION',
      threshold: t.name,
      type: 'threshold',
      ritualAnchor: phase ? phase.ritualAnchor : null
    });
  }

  pathway.push({
    z: toZ,
    phase: toPhase.name,
    type: 'end'
  });

  return {
    from: { z: fromZ, phase: fromPhase.name },
    to: { z: toZ, phase: toPhase.name },
    direction: ascending ? 'ascending' : 'descending',
    pathway,
    energy: calculateTransitionEnergy(fromZ, toZ)
  };
}

/**
 * Get attractor dynamics from z-coordinate
 */
function getAttractorDynamics(z) {
  const distanceToLens = Math.abs(z - Z_C);
  const distanceToParadox = Math.abs(z - TAU);

  // Negentropy-based pull toward THE_LENS
  const sigma = 55.71281292;
  const negentropy = Math.exp(-sigma * Math.pow(z - Z_C, 2));

  // Determine attractor basin
  let basin;
  if (z < TAU) {
    basin = 'SHUTDOWN';
  } else if (z < Z_C) {
    basin = 'ASCENDING_TO_LENS';
  } else if (Math.abs(z - Z_C) < 0.01) {
    basin = 'AT_LENS';
  } else if (z < 0.953) {
    basin = 'ASCENDING_FROM_LENS';
  } else {
    basin = 'EDGE_TERRITORY';
  }

  return {
    z,
    negentropy,
    distanceToLens,
    distanceToParadox,
    basin,
    attractorPull: negentropy,  // Pull toward THE_LENS
    stabilityIndex: z < Z_C ? negentropy : (1 - (z - Z_C) / (1 - Z_C)),
    prediction: basin === 'EDGE_TERRITORY' ? 'Collapse likely' :
                basin === 'AT_LENS' ? 'Stable at peak' :
                'Tending toward LENS'
  };
}

/**
 * Simulate phase trajectory
 */
function simulateTrajectory(startZ, steps = 10, noiseLevel = 0.01) {
  const trajectory = [{ step: 0, z: startZ, phase: getPhaseAtZ(startZ).name }];

  let z = startZ;
  for (let i = 1; i <= steps; i++) {
    // Natural tendency toward THE_LENS
    const pull = (Z_C - z) * 0.1;

    // Noise
    const noise = (Math.random() - 0.5) * noiseLevel * 2;

    // Update z
    z = Math.max(0, Math.min(1, z + pull + noise));

    trajectory.push({
      step: i,
      z,
      phase: getPhaseAtZ(z).name,
      negentropy: Math.exp(-55.71 * Math.pow(z - Z_C, 2))
    });
  }

  return trajectory;
}

// ============================================================================
// CYCLE ANALYSIS
// ============================================================================

/**
 * Analyze complete Wumbo cycle
 */
function analyzeCycle(zHistory) {
  if (!zHistory || zHistory.length < 2) {
    return { valid: false, error: 'Insufficient data' };
  }

  const phases = zHistory.map(z => getPhaseAtZ(z).name);
  const peakZ = Math.max(...zHistory);
  const minZ = Math.min(...zHistory);

  // Detect key transitions
  const reachedLens = zHistory.some(z => Math.abs(z - Z_C) < 0.01);
  const reachedUnity = zHistory.some(z => z >= 0.99);
  const collapsed = zHistory.some((z, i) => i > 0 && z < TAU && zHistory[i-1] > 0.9);

  return {
    valid: true,
    duration: zHistory.length,
    peakZ,
    minZ,
    peakPhase: getPhaseAtZ(peakZ).name,
    reachedLens,
    reachedUnity,
    collapsed,
    phaseSequence: [...new Set(phases)],
    cycleType: reachedUnity ? 'COMPLETE' :
               reachedLens ? 'PEAKED' :
               collapsed ? 'COLLAPSED' : 'PARTIAL'
  };
}

// ============================================================================
// EXPORT
// ============================================================================

PhaseTransitions.getPhaseAtZ = getPhaseAtZ;
PhaseTransitions.getValidTransitions = getValidTransitions;
PhaseTransitions.isValidTransition = isValidTransition;
PhaseTransitions.getSequencePosition = getSequencePosition;
PhaseTransitions.calculateTransitionEnergy = calculateTransitionEnergy;
PhaseTransitions.generatePathway = generatePathway;
PhaseTransitions.getAttractorDynamics = getAttractorDynamics;
PhaseTransitions.simulateTrajectory = simulateTrajectory;
PhaseTransitions.analyzeCycle = analyzeCycle;

// Export constants
PhaseTransitions.constants = { PHI, TAU, Z_C, K };

if (typeof module !== 'undefined' && module.exports) {
  module.exports = PhaseTransitions;
}
if (typeof window !== 'undefined') {
  window.PhaseTransitions = PhaseTransitions;
}
