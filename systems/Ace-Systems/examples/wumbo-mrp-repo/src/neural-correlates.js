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
