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
