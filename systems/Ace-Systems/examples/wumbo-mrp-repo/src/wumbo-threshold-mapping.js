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
