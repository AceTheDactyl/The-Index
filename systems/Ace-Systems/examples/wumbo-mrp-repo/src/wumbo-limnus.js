/**
 * WUMBO LIMNUS Atlas
 * Complete 100-region neural mapping
 * 
 * Structure:
 *   I-LXIII    (63): 7-layer hexagonal prism
 *   LXIV-XCV   (32): EM cage containment field
 *   XCVI-C      (5): Emergent self-reference nodes
 * 
 * @version 1.0.0
 */

const WUMBO_LIMNUS = {
  // ============================================
  // PRISM LAYER 0 (z=0.0, Core) - Regions I-IX
  // ============================================
  "I":     { name: "Somatosensory Cortex", structure: "prism", layer: 0, z: 0.0, phase: "Ignition", field: "e", nt: "Glu", role: "Sensory map", tokens: ["e:U:A:BIO:T:α1", "Φ:M:B:GEO:T:α1"], lambda: "xi" },
  "II":    { name: "Anterior Cingulate Cortex", structure: "prism", layer: 0, z: 0.0, phase: "Ignition", field: "e", nt: "DA", role: "Truth check", tokens: ["e:M:G:CHEM:T:α2", "Φ:C:B:BIO:T:α2"], lambda: "omega" },
  "III":   { name: "Thalamus", structure: "prism", layer: 0, z: 0.0, phase: "Ignition", field: "e", nt: "Glu", role: "Sensory gate", tokens: ["e:C:G:BIO:T:α1", "Φ:M:F:GEO:T:α1"], lambda: "omega" },
  "IV":    { name: "Motor Cortex & Cerebellum", structure: "prism", layer: 0, z: 0.0, phase: "Ignition", field: "Φ", nt: "Glu", role: "Execution", tokens: ["Φ:U:B:BIO:T:α2", "e:U:A:CHEM:T:α2"], lambda: "theta" },
  "V":     { name: "Broca's Area", structure: "prism", layer: 0, z: 0.0, phase: "Ignition", field: "e", nt: "DA", role: "Phrase/sculpt", tokens: ["e:U:A:BIO:T:α3", "Φ:M:B:CHEM:T:α3"], lambda: "xi" },
  "VI":    { name: "Mirror Neuron System", structure: "prism", layer: 0, z: 0.0, phase: "Ignition", field: "e", nt: "DA", role: "Empathic resonance", tokens: ["e:M:G:BIO:T:α2", "Φ:C:B:CHEM:T:α2"], lambda: "omega" },
  "VII":   { name: "Amygdala", structure: "prism", layer: 0, z: 0.0, phase: "Ignition", field: "e", nt: "NE", role: "Salience", tokens: ["e:U:A:CHEM:T:α4", "e:U:B:BIO:T:α4"], lambda: "theta" },
  "VIII":  { name: "Prefrontal Cortex", structure: "prism", layer: 0, z: 0.0, phase: "Ignition", field: "e", nt: "DA", role: "Strategy/control", tokens: ["e:M:G:BIO:T:α3", "Φ:M:F:CHEM:T:α3"], lambda: "omega" },
  "IX":    { name: "Parietal Eye Field", structure: "prism", layer: 0, z: 0.0, phase: "Ignition", field: "e", nt: "ACh", role: "Gaze/attention", tokens: ["e:U:A:BIO:T:α2", "Φ:C:B:GEO:T:α2"], lambda: "xi" },

  // ============================================
  // PRISM LAYER 1 (z=0.167) - Regions X-XVIII
  // ============================================
  "X":     { name: "Subiculum", structure: "prism", layer: 1, z: 0.167, phase: "Ignition→Empowerment", field: "Φ", nt: "Glu", role: "Spatial memory", tokens: ["Φ:M:F:BIO:T:α3", "e:U:B:GEO:T:α3"], lambda: "iota" },
  "XI":    { name: "Pineal Body", structure: "prism", layer: 1, z: 0.167, phase: "Pause↔Ignition", field: "π", nt: "Mel", role: "Circadian portal", tokens: ["π:M:D:CHEM:T:α5", "e:M:G:BIO:T:α5"], lambda: "delta", referenced_by: "XCVII" },
  "XII":   { name: "Middle Temporal Gyrus", structure: "prism", layer: 1, z: 0.167, phase: "Resonance", field: "Φ", nt: "Glu", role: "Semantics", tokens: ["Φ:C:B:BIO:T:α4", "e:M:F:GEO:T:α4"], lambda: "theta", referenced_by: "XCVIII" },
  "XIII":  { name: "Fastigial-Vestibular Loop", structure: "prism", layer: 1, z: 0.167, phase: "Nirvana", field: "Φ", nt: "Glu", role: "Balance", tokens: ["Φ:M:F:GEO:T:α4", "e:C:G:BIO:T:α4"], lambda: "iota", referenced_by: "XCIX" },
  "XIV":   { name: "Posterior Thalamic Nucleus", structure: "prism", layer: 1, z: 0.167, phase: "Transmission", field: "e", nt: "Glu", role: "Final gate", tokens: ["e:C:S:BIO:T:α5", "Φ:U:S:GEO:T:α5"], lambda: "sigma", referenced_by: "C" },
  "XV":    { name: "Cerebellar Uvula", structure: "prism", layer: 1, z: 0.167, phase: "Nirvana", field: "π", nt: "GABA", role: "Stillness anchor", tokens: ["π:M:F:BIO:T:α3", "Φ:M:D:CHEM:T:α3"], lambda: "iota" },
  "XVI":   { name: "AIPS", structure: "prism", layer: 1, z: 0.167, phase: "Empowerment", field: "Φ", nt: "Glu", role: "Gesture translator", tokens: ["Φ:U:B:BIO:T:α4", "e:C:A:GEO:T:α4"], lambda: "theta", referenced_by: "XCVI" },
  "XVII":  { name: "Ventrolateral Thalamus", structure: "prism", layer: 1, z: 0.167, phase: "Transmission", field: "e", nt: "Glu", role: "Feedback loop", tokens: ["e:C:S:BIO:T:α3", "Φ:M:G:GEO:T:α3"], lambda: "sigma" },
  "XVIII": { name: "Superior Parietal Lobule", structure: "prism", layer: 1, z: 0.167, phase: "Empowerment", field: "Φ", nt: "Glu", role: "Spatial integration", tokens: ["Φ:M:F:GEO:T:α4", "e:U:A:BIO:T:α4"], lambda: "iota" },

  // ============================================
  // PRISM LAYER 2 (z=0.333) - Regions XIX-XXVII
  // ============================================
  "XIX":   { name: "Premotor Cortex", structure: "prism", layer: 2, z: 0.333, phase: "Empowerment", field: "Φ", nt: "Glu", role: "Movement planning", tokens: ["Φ:U:B:BIO:T:α5", "e:U:A:GEO:T:α5"], lambda: "theta" },
  "XX":    { name: "Wernicke's Area", structure: "prism", layer: 2, z: 0.333, phase: "Resonance", field: "π", nt: "Glu", role: "Language comprehension", tokens: ["π:M:F:BIO:T:α6", "Φ:C:B:GEO:T:α6"], lambda: "iota" },
  "XXI":   { name: "STS Mirror Region", structure: "prism", layer: 2, z: 0.333, phase: "Resonance", field: "e", nt: "DA", role: "Social mirroring", tokens: ["e:M:G:BIO:T:α5", "Φ:C:B:CHEM:T:α5"], lambda: "omega" },
  "XXII":  { name: "Central Amygdala", structure: "prism", layer: 2, z: 0.333, phase: "Ignition", field: "e", nt: "NE", role: "Threat response", tokens: ["e:U:A:CHEM:T:α6", "π:U:S:BIO:T:α6"], lambda: "theta" },
  "XXIII": { name: "Dorsolateral PFC", structure: "prism", layer: 2, z: 0.333, phase: "Empowerment", field: "e", nt: "DA", role: "Working memory", tokens: ["e:M:G:BIO:T:α6", "Φ:M:F:CHEM:T:α6"], lambda: "omega" },
  "XXIV":  { name: "Orbitofrontal Cortex", structure: "prism", layer: 2, z: 0.333, phase: "Resonance", field: "e", nt: "DA", role: "Social tuning", tokens: ["e:M:G:CHEM:T:α5", "Φ:C:B:BIO:T:α5"], lambda: "omega" },
  "XXV":   { name: "Cingulate Gyrus", structure: "prism", layer: 2, z: 0.333, phase: "Resonance", field: "π", nt: "DA", role: "Routing/alignment", tokens: ["π:M:G:BIO:T:α6", "e:C:F:CHEM:T:α6"], lambda: "omega" },
  "XXVI":  { name: "Ventral Striatum", structure: "prism", layer: 2, z: 0.333, phase: "Ignition", field: "e", nt: "DA", role: "Incentive", tokens: ["e:U:A:CHEM:T:α5", "π:U:A:BIO:T:α5"], lambda: "xi" },
  "XXVII": { name: "Claustrum", structure: "prism", layer: 2, z: 0.333, phase: "Resonance", field: "π", nt: "Glu", role: "Consciousness binding", tokens: ["π:M:S:BIO:T:α7", "Φ:C:B:GEO:T:α7"], lambda: "sigma" },

  // ============================================
  // PRISM LAYER 3 (z=0.5, Center) - Regions XXVIII-XXXVI
  // ============================================
  "XXVIII": { name: "Default Mode Network", structure: "prism", layer: 3, z: 0.5, phase: "Nirvana", field: "π", nt: "Glu", role: "Self-referential", tokens: ["π:M:S:BIO:T:α7", "e:M:S:GEO:T:α7"], lambda: "sigma" },
  "XXIX":  { name: "Habenula", structure: "prism", layer: 3, z: 0.5, phase: "Pause", field: "e", nt: "Glu", role: "Disappointment gate", tokens: ["e:C:D:CHEM:U:α8", "π:M:D:BIO:U:α8"], lambda: "delta" },
  "XXX":   { name: "Corpus Callosum", structure: "prism", layer: 3, z: 0.5, phase: "Transmission", field: "Φ", nt: "Glu", role: "Bridge/balance", tokens: ["Φ:C:F:GEO:T:α7", "e:C:S:BIO:T:α7"], lambda: "iota" },
  "XXXI":  { name: "Locus Coeruleus", structure: "prism", layer: 3, z: 0.5, phase: "Ignition", field: "e", nt: "NE", role: "Arousal ignition", tokens: ["e:U:A:CHEM:T:α8", "e:U:A:BIO:T:α8"], lambda: "xi" },
  "XXXII": { name: "Periaqueductal Gray", structure: "prism", layer: 3, z: 0.5, phase: "Pause", field: "π", nt: "GABA", role: "Defense/shutdown", tokens: ["π:M:D:BIO:T:α7", "Φ:M:D:CHEM:T:α7"], lambda: "delta" },
  "XXXIII": { name: "Anterior Temporal Pole", structure: "prism", layer: 3, z: 0.5, phase: "Resonance", field: "π", nt: "Glu", role: "Story keeper", tokens: ["π:M:S:BIO:T:α8", "Φ:C:F:GEO:T:α8"], lambda: "sigma" },
  "XXXIV": { name: "vmPFC", structure: "prism", layer: 3, z: 0.5, phase: "Resonance", field: "e", nt: "DA", role: "Ethical integration", tokens: ["e:M:F:BIO:T:α8", "π:M:G:CHEM:T:α8"], lambda: "iota" },
  "XXXV":  { name: "Dorsal Raphe", structure: "prism", layer: 3, z: 0.5, phase: "Nirvana", field: "e", nt: "5-HT", role: "Mood setpoint", tokens: ["e:M:D:CHEM:T:α7", "Φ:M:F:BIO:T:α7"], lambda: "delta" },
  "XXXVI": { name: "Superior Colliculus", structure: "prism", layer: 3, z: 0.5, phase: "Ignition", field: "e", nt: "Glu", role: "Visual orienting", tokens: ["e:U:A:BIO:T:α7", "Φ:U:B:GEO:T:α7"], lambda: "xi" },

  // ============================================
  // PRISM LAYER 4 (z=0.667) - Regions XXXVII-XLV
  // ============================================
  "XXXVII": { name: "Anterior Insula", structure: "prism", layer: 4, z: 0.667, phase: "Resonance", field: "e", nt: "DA", role: "Feeling of feeling", tokens: ["e:M:S:BIO:T:α9", "π:M:S:CHEM:T:α9"], lambda: "sigma" },
  "XXXVIII": { name: "Lateral Habenula", structure: "prism", layer: 4, z: 0.667, phase: "Pause", field: "e", nt: "Glu", role: "Rejection gate", tokens: ["e:C:D:CHEM:U:α9", "π:M:D:BIO:U:α9"], lambda: "delta" },
  "XXXIX": { name: "Precuneus", structure: "prism", layer: 4, z: 0.667, phase: "Nirvana", field: "Φ", nt: "Glu", role: "Perspective", tokens: ["Φ:M:F:GEO:T:α9", "π:M:F:BIO:T:α9"], lambda: "sigma" },
  "XL":    { name: "Cerebellar Cognitive Zone", structure: "prism", layer: 4, z: 0.667, phase: "Empowerment", field: "Φ", nt: "Glu", role: "Timing", tokens: ["Φ:M:G:BIO:T:α8", "e:M:G:GEO:T:α8"], lambda: "omega" },
  "XLI":   { name: "Basolateral Amygdala", structure: "prism", layer: 4, z: 0.667, phase: "Ignition", field: "e", nt: "NE", role: "Archive of feeling", tokens: ["e:U:A:CHEM:T:α9", "Φ:M:F:BIO:T:α9"], lambda: "theta" },
  "XLII":  { name: "Pulvinar", structure: "prism", layer: 4, z: 0.667, phase: "Transmission", field: "e", nt: "Glu", role: "Spotlight shaper", tokens: ["e:C:S:BIO:T:α8", "Φ:M:G:GEO:T:α8"], lambda: "sigma" },
  "XLIII": { name: "TPJ", structure: "prism", layer: 4, z: 0.667, phase: "Resonance", field: "π", nt: "Glu", role: "Mind reading", tokens: ["π:M:G:BIO:T:α9", "Φ:C:B:GEO:T:α9"], lambda: "omega" },
  "XLIV":  { name: "Medial Septum", structure: "prism", layer: 4, z: 0.667, phase: "Resonance", field: "π", nt: "ACh", role: "Memory rhythms", tokens: ["π:U:F:CHEM:T:α8", "e:M:S:BIO:T:α8"], lambda: "iota" },
  "XLV":   { name: "Subgenual Cingulate", structure: "prism", layer: 4, z: 0.667, phase: "Pause", field: "e", nt: "5-HT", role: "Sorrow inertia", tokens: ["e:M:D:CHEM:U:α9", "π:M:D:BIO:U:α9"], lambda: "delta" },

  // ============================================
  // PRISM LAYER 5 (z=0.833, Near Critical) - Regions XLVI-LIV
  // ============================================
  "XLVI":  { name: "VTA", structure: "prism", layer: 5, z: 0.833, phase: "Ignition", field: "e", nt: "DA", role: "Spark", tokens: ["e:U:A:CHEM:T:α10", "e:U:A:BIO:T:α10"], lambda: "xi" },
  "XLVII": { name: "Entorhinal Cortex", structure: "prism", layer: 5, z: 0.833, phase: "Nirvana", field: "Φ", nt: "Glu", role: "Identity gate", tokens: ["Φ:M:F:BIO:T:α10", "π:C:F:GEO:T:α10"], lambda: "iota" },
  "XLVIII": { name: "Supramarginal Gyrus", structure: "prism", layer: 5, z: 0.833, phase: "Resonance", field: "π", nt: "Glu", role: "Self/other", tokens: ["π:M:G:BIO:T:α10", "Φ:C:B:GEO:T:α10"], lambda: "omega" },
  "XLIX":  { name: "NAcc", structure: "prism", layer: 5, z: 0.833, phase: "Ignition", field: "e", nt: "DA", role: "Craving engine", tokens: ["e:U:A:CHEM:T:α11", "Φ:C:B:BIO:T:α11"], lambda: "xi" },
  "L":     { name: "Cerebral Aqueduct", structure: "prism", layer: 5, z: 0.833, phase: "Transmission", field: "e", nt: "Glu", role: "Choke point", tokens: ["e:C:S:BIO:T:α10", "π:M:D:GEO:T:α10"], lambda: "sigma" },
  "LI":    { name: "Anterior Thalamic Nuclei", structure: "prism", layer: 5, z: 0.833, phase: "Transmission", field: "Φ", nt: "Glu", role: "Compass", tokens: ["Φ:C:F:GEO:T:α10", "e:M:A:BIO:T:α10"], lambda: "iota" },
  "LII":   { name: "Parafascicular Nucleus", structure: "prism", layer: 5, z: 0.833, phase: "Ignition", field: "e", nt: "Glu", role: "Attention switch", tokens: ["e:U:A:BIO:T:α11", "Φ:C:B:GEO:T:α11"], lambda: "xi" },
  "LIII":  { name: "Inferior Colliculus", structure: "prism", layer: 5, z: 0.833, phase: "Transmission", field: "e", nt: "Glu", role: "Sonic filter", tokens: ["e:C:S:BIO:T:α10", "Φ:M:G:GEO:T:α10"], lambda: "sigma" },
  "LIV":   { name: "Perirhinal Cortex", structure: "prism", layer: 5, z: 0.833, phase: "Resonance", field: "π", nt: "Glu", role: "Meaning-maker", tokens: ["π:M:S:BIO:T:α11", "Φ:C:F:GEO:T:α11"], lambda: "sigma" },

  // ============================================
  // PRISM LAYER 6 (z=1.0, Outer) - Regions LV-LXIII
  // ============================================
  "LV":    { name: "Vermis", structure: "prism", layer: 6, z: 1.0, phase: "Nirvana", field: "Φ", nt: "GABA", role: "Balance", tokens: ["Φ:M:F:GEO:T:α11", "π:M:D:BIO:T:α11"], lambda: "iota" },
  "LVI":   { name: "Anterior Insular-Operculum", structure: "prism", layer: 6, z: 1.0, phase: "Resonance", field: "e", nt: "DA", role: "Fusion point", tokens: ["e:M:G:BIO:T:α12", "π:C:F:CHEM:T:α12"], lambda: "omega" },
  "LVII":  { name: "Paraventricular Nucleus", structure: "prism", layer: 6, z: 1.0, phase: "Ignition", field: "e", nt: "NE", role: "Stress switch", tokens: ["e:U:A:CHEM:T:α12", "π:U:S:BIO:T:α12"], lambda: "xi" },
  "LVIII": { name: "Lateral OFC", structure: "prism", layer: 6, z: 1.0, phase: "Resonance", field: "e", nt: "DA", role: "Consequence", tokens: ["e:M:G:CHEM:T:α11", "Φ:C:B:BIO:T:α11"], lambda: "omega" },
  "LIX":   { name: "Midcingulate Cortex", structure: "prism", layer: 6, z: 1.0, phase: "Empowerment", field: "e", nt: "DA", role: "Engine of doing", tokens: ["e:U:A:BIO:T:α12", "Φ:M:B:GEO:T:α12"], lambda: "xi" },
  "LX":    { name: "Calcarine Sulcus", structure: "prism", layer: 6, z: 1.0, phase: "Ignition", field: "e", nt: "Glu", role: "Visual core", tokens: ["e:U:A:BIO:T:α11", "Φ:M:B:GEO:T:α11"], lambda: "xi" },
  "LXI":   { name: "Rostral PFC", structure: "prism", layer: 6, z: 1.0, phase: "Resonance", field: "e", nt: "DA", role: "Reflective flame", tokens: ["e:M:S:BIO:T:α12", "π:M:S:CHEM:T:α12"], lambda: "sigma" },
  "LXII":  { name: "MLR", structure: "prism", layer: 6, z: 1.0, phase: "Empowerment", field: "e", nt: "Glu", role: "Will to move", tokens: ["e:U:A:BIO:T:α12", "Φ:U:B:GEO:T:α12"], lambda: "xi" },
  "LXIII": { name: "Anterior Temporal Sulcus", structure: "prism", layer: 6, z: 1.0, phase: "Resonance", field: "π", nt: "Glu", role: "Subtext", tokens: ["π:M:S:BIO:T:α12", "Φ:C:F:GEO:T:α12"], lambda: "sigma" },

  // ============================================
  // EM CAGE TOP HEXAGON (z=0.9) - Regions LXIV-LXXV
  // ============================================
  "LXIV":  { name: "Lateral Septum", structure: "cage", component: "top", z: 0.9, phase: "Nirvana", field: "e", nt: "GABA", role: "Calm circuit", tokens: ["e:M:D:CHEM:T:α11", "π:M:D:BIO:T:α11"], lambda: "sigma" },
  "LXV":   { name: "Cerebellar Tonsil", structure: "cage", component: "top", z: 0.9, phase: "Pause", field: "Φ", nt: "GABA", role: "Silent reactor", tokens: ["Φ:M:D:BIO:U:α12", "π:M:D:CHEM:U:α12"], lambda: "delta" },
  "LXVI":  { name: "Pontine Reticular Formation", structure: "cage", component: "top", z: 0.9, phase: "Ignition", field: "e", nt: "ACh", role: "Motion catalyst", tokens: ["e:U:A:CHEM:T:α11", "e:M:G:BIO:T:α11"], lambda: "xi" },
  "LXVII": { name: "Insular-Opercular Speech", structure: "cage", component: "top", z: 0.9, phase: "Empowerment", field: "e", nt: "DA", role: "Voice within fire", tokens: ["e:U:A:BIO:T:α12", "Φ:U:B:CHEM:T:α12"], lambda: "xi" },
  "LXVIII": { name: "Amygdala Central Nucleus", structure: "cage", component: "top", z: 0.9, phase: "Ignition", field: "e", nt: "NE", role: "First alarm", tokens: ["e:U:A:CHEM:T:α12", "π:U:S:BIO:T:α12"], lambda: "theta" },
  "LXIX":  { name: "TRN", structure: "cage", component: "top", z: 0.9, phase: "Transmission", field: "π", nt: "GABA", role: "Filter grid", tokens: ["π:C:G:BIO:T:α11", "Φ:M:F:GEO:T:α11"], lambda: "omega" },
  "LXX":   { name: "Cuneus", structure: "cage", component: "top", z: 0.9, phase: "Resonance", field: "Φ", nt: "Glu", role: "Background reader", tokens: ["Φ:M:F:GEO:T:α11", "e:M:A:BIO:T:α11"], lambda: "omega" },
  "LXXI":  { name: "VMH", structure: "cage", component: "top", z: 0.9, phase: "Nirvana", field: "Φ", nt: "Glu", role: "Inner balance", tokens: ["Φ:M:F:BIO:T:α12", "e:M:D:CHEM:T:α12"], lambda: "iota" },
  "LXXII": { name: "Periventricular Gray", structure: "cage", component: "top", z: 0.9, phase: "Pause", field: "π", nt: "GABA", role: "Threshold", tokens: ["π:M:D:BIO:U:α11", "Φ:M:D:CHEM:U:α11"], lambda: "delta" },
  "LXXIII": { name: "Frontal Operculum", structure: "cage", component: "top", z: 0.9, phase: "Empowerment", field: "e", nt: "DA", role: "Edge of expression", tokens: ["e:U:A:BIO:T:α12", "Φ:U:B:GEO:T:α12"], lambda: "xi" },
  "LXXIV": { name: "Nodulus", structure: "cage", component: "top", z: 0.9, phase: "Nirvana", field: "Φ", nt: "GABA", role: "Gravity whisperer", tokens: ["Φ:M:F:GEO:T:α11", "e:C:G:BIO:T:α11"], lambda: "iota" },
  "LXXV":  { name: "Substantia Nigra", structure: "cage", component: "top", z: 0.9, phase: "Empowerment", field: "e", nt: "DA", role: "Movement gatekeeper", tokens: ["e:C:G:CHEM:T:α12", "Φ:M:G:BIO:T:α12"], lambda: "omega" },

  // ============================================
  // EM CAGE BOTTOM HEXAGON (z=0.1) - Regions LXXVI-LXXXVII
  // ============================================
  "LXXVI": { name: "V4", structure: "cage", component: "bottom", z: 0.1, phase: "Resonance", field: "Φ", nt: "Glu", role: "Chromatic shaper", tokens: ["Φ:M:F:GEO:T:α3", "e:M:A:BIO:T:α3"], lambda: "iota" },
  "LXXVII": { name: "Lingual Gyrus", structure: "cage", component: "bottom", z: 0.1, phase: "Resonance", field: "π", nt: "Glu", role: "Glyph reader", tokens: ["π:M:S:BIO:T:α3", "Φ:C:F:GEO:T:α3"], lambda: "sigma" },
  "LXXVIII": { name: "mPFC", structure: "cage", component: "bottom", z: 0.1, phase: "Resonance", field: "e", nt: "DA", role: "Identity sculptor", tokens: ["e:M:F:BIO:T:α4", "π:M:S:CHEM:T:α4"], lambda: "iota" },
  "LXXIX": { name: "dLPFC", structure: "cage", component: "bottom", z: 0.1, phase: "Empowerment", field: "e", nt: "DA", role: "Gate of delivery", tokens: ["e:M:G:CHEM:T:α4", "Φ:U:S:BIO:T:α4"], lambda: "omega" },
  "LXXX":  { name: "IPL", structure: "cage", component: "bottom", z: 0.1, phase: "Resonance", field: "π", nt: "Glu", role: "Paradox holder", tokens: ["π:M:G:BIO:P:α13", "Φ:C:B:GEO:T:α13"], lambda: "delta" },
  "LXXXI": { name: "ACC (Dorsal)", structure: "cage", component: "bottom", z: 0.1, phase: "Resonance", field: "e", nt: "DA", role: "Inner judge", tokens: ["e:M:G:CHEM:T:α4", "π:M:F:BIO:T:α4"], lambda: "omega" },
  "LXXXII": { name: "Anterior Hippocampus", structure: "cage", component: "bottom", z: 0.1, phase: "Nirvana", field: "Φ", nt: "Glu", role: "Context mapper", tokens: ["Φ:M:F:BIO:T:α3", "π:C:F:GEO:T:α3"], lambda: "iota" },
  "LXXXIII": { name: "Crus I/II", structure: "cage", component: "bottom", z: 0.1, phase: "Empowerment", field: "Φ", nt: "GABA", role: "Somatic timekeeper", tokens: ["Φ:M:G:BIO:T:α4", "e:M:G:GEO:T:α4"], lambda: "omega" },
  "LXXXIV": { name: "Basal Forebrain", structure: "cage", component: "bottom", z: 0.1, phase: "Ignition", field: "e", nt: "ACh", role: "Timing messenger", tokens: ["e:M:G:CHEM:T:α3", "e:C:A:BIO:T:α3"], lambda: "xi" },
  "LXXXV": { name: "Reticular Formation", structure: "cage", component: "bottom", z: 0.1, phase: "Ignition", field: "e", nt: "NE", role: "Wake thread", tokens: ["e:U:A:CHEM:T:α4", "e:U:A:BIO:T:α4"], lambda: "xi" },
  "LXXXVI": { name: "DVC", structure: "cage", component: "bottom", z: 0.1, phase: "Pause", field: "Φ", nt: "GABA", role: "Kill-switch", tokens: ["Φ:M:D:BIO:T:α3", "π:M:D:CHEM:T:α3"], lambda: "delta" },
  "LXXXVII": { name: "Cranial Nerves", structure: "cage", component: "bottom", z: 0.1, phase: "Transmission", field: "e", nt: "ACh", role: "Face switch", tokens: ["e:C:S:BIO:T:α4", "Φ:U:S:GEO:T:α4"], lambda: "sigma" },

  // ============================================
  // EM CAGE VERTICES (z=0.5) - Regions LXXXVIII-XCV
  // ============================================
  "LXXXVIII": { name: "Spinal Relays", structure: "cage", component: "vertex", z: 0.5, phase: "Transmission", field: "e", nt: "Glu", role: "Carrier", tokens: ["e:C:S:BIO:T:α7", "Φ:C:B:GEO:T:α7"], lambda: "sigma" },
  "LXXXIX": { name: "Globus Pallidus", structure: "cage", component: "vertex", z: 0.5, phase: "Empowerment", field: "π", nt: "GABA", role: "Go/no-go", tokens: ["π:M:D:BIO:T:α8", "Φ:M:D:GEO:T:α8"], lambda: "delta" },
  "XC":    { name: "Lateral Hypothalamus", structure: "cage", component: "vertex", z: 0.5, phase: "Ignition", field: "e", nt: "DA", role: "Drive switch", tokens: ["e:U:A:CHEM:T:α7", "π:U:G:BIO:T:α7"], lambda: "xi" },
  "XCI":   { name: "Posterior Insula", structure: "cage", component: "vertex", z: 0.5, phase: "Resonance", field: "Φ", nt: "Glu", role: "Body's edges", tokens: ["Φ:M:G:BIO:T:α8", "e:M:S:GEO:T:α8"], lambda: "omega" },
  "XCII":  { name: "Nucleus Basalis", structure: "cage", component: "vertex", z: 0.5, phase: "Ignition", field: "e", nt: "ACh", role: "Attention tuner", tokens: ["e:M:G:CHEM:T:α7", "e:U:A:BIO:T:α7"], lambda: "xi" },
  "XCIII": { name: "Caudate", structure: "cage", component: "vertex", z: 0.5, phase: "Empowerment", field: "e", nt: "DA", role: "Path chooser", tokens: ["e:C:G:CHEM:T:α8", "Φ:M:F:BIO:T:α8"], lambda: "omega" },
  "XCIV":  { name: "Superior Temporal Pole", structure: "cage", component: "vertex", z: 0.5, phase: "Resonance", field: "π", nt: "Glu", role: "Emotional communicator", tokens: ["π:M:S:BIO:T:α7", "e:M:S:CHEM:T:α7"], lambda: "sigma" },
  "XCV":   { name: "Uvula (structural)", structure: "cage", component: "vertex", z: 0.5, phase: "Nirvana", field: "Φ", nt: "GABA", role: "Stillness anchor", tokens: ["Φ:M:F:GEO:T:α8", "π:M:D:BIO:T:α8"], lambda: "iota" },

  // ============================================
  // EMERGENT SELF-REFERENCE NODES - Regions XCVI-C
  // Appear only when coherence < 0.2 (FREE state)
  // ============================================
  "XCVI":  { name: "AIPS (recursion)", structure: "emergent", type: "gesture_recursion", references: "XVI", z: "variable", phase: "FREE", field: "Φ", role: "Gesture becomes self-aware", tokens: ["Φ:M:B:BIO:P:α13", "Φ:C:F:GEO:P:α13"], lambda: "delta", emergent: true, eigenvalue: "φ⁻¹", frequency: "8Hz", info_bits: 7 },
  "XCVII": { name: "Pineal (recursion)", structure: "emergent", type: "portal_recursion", references: "XI", z: "variable", phase: "FREE", field: "π", role: "Portal recognizes its own rhythms", tokens: ["π:M:G:CHEM:P:α14", "π:C:S:BIO:P:α14"], lambda: "delta", emergent: true, eigenvalue: "exp(2πi/φ)", frequency: "0.0001Hz", info_bits: 10 },
  "XCVIII": { name: "MTG (recursion)", structure: "emergent", type: "semantic_recursion", references: "XII", z: "variable", phase: "FREE", field: "π", role: "Semantics binds its own meaning", tokens: ["π:M:S:BIO:P:α14", "π:C:B:GEO:P:α14"], lambda: "delta", emergent: true, eigenvalue: "undefined", frequency: "40Hz", info_bits: 17 },
  "XCIX":  { name: "Fastigial-Vestibular (recursion)", structure: "emergent", type: "balance_recursion", references: "XIII", z: "variable", phase: "FREE", field: "Φ", role: "Balance balances its own balancing", tokens: ["Φ:M:F:GEO:P:α14", "Φ:C:G:BIO:P:α14"], lambda: "delta", emergent: true, eigenvalue: "0", frequency: "4Hz", info_bits: 4 },
  "C":     { name: "PTN (recursion → I)", structure: "emergent", type: "signal_recursion", references: "XIV", loops_to: "I", z: "variable", phase: "FREE", field: "e", role: "Final gate loops to first gate", tokens: ["e:C:S:BIO:P:α15", "e:M:A:GEO:P:α15"], lambda: "delta", emergent: true, closes_loop: true, eigenvalue: "1", frequency: "1Hz", info_bits: 664, winding_number: 1 }
};

// Export
if (typeof module !== 'undefined' && module.exports) {
  module.exports = WUMBO_LIMNUS;
}
if (typeof window !== 'undefined') {
  window.WUMBO_LIMNUS = WUMBO_LIMNUS;
}
