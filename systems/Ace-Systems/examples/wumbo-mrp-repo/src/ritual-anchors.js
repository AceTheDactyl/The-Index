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
