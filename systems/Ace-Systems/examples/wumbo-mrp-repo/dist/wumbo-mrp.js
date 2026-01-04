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
