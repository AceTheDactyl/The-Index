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
