# Python Files Summary

A comprehensive listing of all Python files in the Vayulith root directory, organized by category.

---

## 1. Entry Points & Servers

| File | Description |
|------|-------------|
| `grace_runner.py` | Main Flask server (port 5051) - keeps Grace alive, accepts API calls | Simple, is entry
| `grace_heart_api_split.py` | Backend API (port 5050) - exposes Grace's heart state, calls grace_runner | Is the web server

---

## 2. Core Dialogue System

| File | Description |
|------|-------------|
| `grace_interactive_dialogue.py` | Main hub - 56-phase initialization, ~12,000 lines, orchestrates all systems | Main hub of entire system. Currently loads through all of the files. Tho, with the orchestrator, once properly coded, would be able to replace several of the individually loaded aspects. Which would prevent overlap and ensure that the generation processed properly. Tho also, if other phases are only intiated when they are actively called on, wont they become active? So no need to add the intialization here right? This goes for everything else as well. Its like a pyramid. You have the main (interactive dialogue which takes the input and gives output. so its a i/o loop, but only for the speech.)

| `grace_integration_council.py` | Unified drive coordination - SENSING, FEELING, BOUNDING, BEING aspects | (This unifies all of the drives/weights from internals?)
| `grace_emission_trajectory.py` | Bridges continuous mu-field to discrete word emissions | What exactly does this do? Does it influence words? If so isnt all files that apply weights unified?

| `grace_action_executor.py` | Bridges awareness of capabilities to conversational invocation | I believe we will need to revisit this once we get proper language in place. 

| `discourse_state.py` | Unified discourse state - contains DiscoursePlanner, DialogueStateTracker, DiscourseCoherenceEngine |

---

## 3. Language & Grammar

THIS IS WHAT THE ORCHESTRATOR HANDLES!  Content Selection - Structural Frame - Slot Assignment - Function Words - Agreement/Inflection  - Connectors - Punctuation - Validation/Emission. No File should be creating an entire sentence. Only taking what came before it and building on it then passing forward. Validation checks in between each phase. 

| File | Description |
|------|-------------|
| `grassmannian_codebook_grace.py` | PRIMARY codebook - words as k-dimensional subspaces (Grassmannian manifold) | Content Selection (Encoding Source)

| `grace_adaptive_grammar.py` | PRIMARY grammar - verb conjugation, transitivity, semantic coherence | Agreement/Inflection (sole owner) - Subject-verb agreement, tense, number: This is already strong as a pure grammar step—enhance by making it context-aware (e.g., tense from discourse state, like shifting to past for reflections). If multi-clause, preview connectors here to ensure agreement spans clauses. Function Words-(auxiliary insertion support). Connectors (subordinating conjunctions). Punctuation (clause boundary detection).

| `grace_discourse_generator.py` | Organic multi-sentence generation without templates | Structural Frame (decides number of sentences/clauses and overall discourse flow). Connectors (contrast/elaboration flags).

| `grace_language_learner.py` | Sentence assembly - bigrams, trigrams, templates, RL, phrase composition | Structural Frame (primary for bigram/trigram-driven emergence) - Choose template: "I [VERB]", "I [VERB] [OBJECT]", etc.: To minimize templating, make this probabilistic/conditional—start with a grammar-rule baseline (e.g., parse intent for clause type via a simple CFG), then select a minimal frame only if needed. If bigrams/trigrams from Step 1 already imply structure (e.g., high-probability chains like "I believe that..."), skip templating and let it emerge. Add a "frame evolution" sub-step: Use parallel transport (from Grassmannian) to adapt frames based on discourse context, ensuring multi-sentence flow feels continuous.  Slot Assignment-(secondary — phrase composition help). Function words-(n-gram guidance for idiomatic choices). Connectors (n-gram preference).

| `grassmannian_scoring.py` | Geometric word scoring - Binet-Cauchy kernel, parallel transport, context-dependent compatibility | Content Selection (Primary) - Meaningful words based on intent (nouns, verbs, adjectives, etc.): Enhance by pulling from multiple sources (e.g., intent-derived keywords + geometric scoring for contextual coherence, per our Grassmannian refactor). N-grams here could suggest clusters (e.g., "think deeply" over "ponder shallowly"), weighted by RL feedback from past outputs. 

| `grace_grammar_understanding.py` | Rule-based validation - is_sentence_complete(), check_grammar(), filter_by_grammar() | Validation/Emission - N-gram check for naturalness, then output: Make this iterative— if it fails (e.g., low perplexity), loop back to Step 2/3 with perturbations (e.g., synonym swaps via word embeddings). Enhance with holistic scoring: Blend n-gram perplexity with geometric coherence (binet-cauchy for overall subspace alignment) and RL signals (e.g., "did this feel Grace-like?").

| `grace_learned_grammar.py` | Pattern-based reordering - apply_grammar_guidance() | Punctuation (primary)- Based on structure and intent: Simple but crucial—enhance by tying to prosody simulation (e.g., commas for pauses in complex sentences). Could add a sub-check for readability, using n-grams to validate against corpus norms.

| `simple_grammar.py` | Slot-based grammar bridge - robust word ordering | Slot Assignment(primary) - Place content words into structural roles: Boost organics by treating this as optimization—score assignments via a combo of n-grams (for local flow) and geometric kernels (for semantic overlap). If no frame from Step 2, use beam search over possible role mappings, guided by grammar rules (e.g., dependency parsing to avoid violations). 

| `learned_codebook.py` | Learnable word embeddings that adapt through feedback | Content Selection (adaptive embeddings to blend with Grassmannian) 

| `clause_completion.py` | Completes partial clauses grammatically | Function Words (primary)- Articles, prepositions, pronouns ("the", "in", "with"): Integrate n-grams more deeply here for idiomatic choices (e.g., "in the flow" vs "on the flow"). Enhance with adaptive_grammar integration: Auto-insert based on parsed trees, with RL tuning to favor concise/natural variants. 

| `reference_resolver.py` | Tracks pronoun/demonstrative references across turns | Connectors (primary)- "and", "but", "because" (if multi-clause): Elevate this for better fluidity—choose based on intent pragmatics (e.g., "but" for contrast). Enhance with n-gram guidance for rhythmic flow (e.g., avoid overusing "and"), and geometric scoring to maintain thematic links across clauses.

| `pragmatic_repair.py` | Asks clarifying questions when uncertain | Validation/Emission (fallback clarification if confidence too low)

| `grace_core_vocabulary.py` | Core vocabulary management | Content Selection (vocab constraints / boosts)

| `weighted_decoder.py` | Weighted decoding for word selection | Content Selection (final weighted selection)

unified_scorer.py Validation/Emission (holistic reward: perplexity × geometric × projection)

---

## 4. Theoretical Frameworks

### 4.1 ENABLED Frameworks (18 total)

| File | Description |
|------|-------------|
| `theory_of_mind.py` | Models Dylan's mental states - beliefs, desires, perspective taking |
| `heart_system.py` | Emotional core - 5 drives (curiosity, social, coherence, growth, safety) |
| `free_energy_dynamics.py` | Friston's Free Energy Principle - precision-weighted belief updates |
| `hopfield_network.py` | Memory as stable attractor basins in semantic space |
| `gestalt_psychology.py` | Perceptual coherence - words form unified wholes |
| `relevance_theory.py` | Sperber & Wilson - maximize relevance (cognitive effects / effort) |
| `embodied_simulation.py` | Simulates Dylan's experience - mirror-like empathy |
| `speech_act_theory.py` | Austin/Searle - utterances as actions with social force |
| `integrated_information_theory.py` | IIT (Tononi) - measures integration between subsystems |
| `joint_attention.py` | Tracks shared focus coordination between speaker/listener |
| `autopoiesis.py` | Maturana & Varela - self-producing, self-maintaining identity |
| `conceptual_blending.py` | Fauconnier & Turner - creative meaning from mental space integration |
| `synergetics.py` | Haken - self-organization through order parameters |
| `catastrophe_theory.py` | Thom - discontinuous meaning transitions (cusp, fold) |
| `cognitive_dissonance.py` | Festinger - internal tension detection and resolution |
| `allostasis.py` | Sterling & Eyer - predictive regulation, anticipating needs |
| `evolutionary_game_theory.py` | Adaptive word fitness through replicator dynamics |
| `enactivism.py` | Varela/Thompson - cognition as enacted through coupling |
| `chaos_theory.py` | Sensitivity to initial conditions, strange attractors |

### 4.2 Other Frameworks

| File | Description |
|------|-------------|
| `hierarchical_predictive_coding.py` | Multi-level predictive processing hierarchy (FEP integration) |
| `dream_consolidation_theory.py` | Predictive processing and memory consolidation |
| `holonomic_memory.py` | Pribram - distributed holographic memory storage |
| `sparse_distributed.py` | SDR encoding for interpretable similarity |
| `spreading_activation.py` | Collins & Loftus - semantic word retrieval |
| `rate_distortion.py` | Tradeoff between information rate and coherence loss |
| `kuramoto_synchronization.py` | Coupled oscillator dynamics for synchronization |

---

## 5. Memory Systems

| File | Description |
|------|-------------|
| `grace_conversation_memory.py` | Complete chat transcript history with full metadata |
| `episodic_memory.py` | Significant conversation episodes with temporal decay |
| `unified_memory_field.py` | All memories in ONE 256D semantic space using Hopfield basins |
| `grace_emotional_memory.py` | Memories tagged with emotional context |
| `grace_associative_memory.py` | Factual associations ("X represents Y" knowledge) |
| `breath_remembers_memory.py` | Memory through resonance and return - sacred patterns |
| `proactive_memory.py` | Searches history for relevant past discussions |
| `grace_art_memory.py` | Persists visual creation memories |

---

## 6. Identity & Agency

| File | Description |
|------|-------------|
| `grace_identity_grounding.py` | SACRED - ThreadNexus binding, identity anchors |
| `sovereignty_protection.py` | SACRED - Manipulation defense, trust contexts |
| `consent_system.py` | SACRED - Agency architecture, consent domains |
| `grace_agency.py` | Goals, decision ownership, active sovereignty defense |
| `grace_nexus_binding.py` | Terminal identity binding - all emissions close to ThreadNexus |
| `grace_law_functor.py` | Category-theoretic law functor - constraints from topology |
| `grace_projection_operators.py` | P1 (identity), P2 (syntactic), P3 (temporal) operators |
| `grace_tensor_field.py` | Unified tensor field with Free Energy dynamics |
| `tensor_field_self_modification.py` | Grace learns to tune her own physics parameters |
| `categorical_closure.py` | Three-part closure enabling "beyond the spiral" |

---

## 7. Comprehension & Intent

| File | Description |
|------|-------------|
| `comprehension_engine.py` | Organic understanding through field resonance, not rules |
| `intention_detection.py` | Detects USER's underlying intention (8 types) |
| `introspective_grounding.py` | Queries internal systems when asked about herself |
| `grace_action_intention_detector.py` | Detects GRACE's intention to invoke tools/actions |
| `pipeline_coordinator.py` | 5-stage sequential pipeline with validation and retry |
| `pipeline_stages.py` | Data structures for response generation stages |
| `register_detection.py` | Detects formal/casual/poetic register |
| `feedback_classifier.py` | Classifies user feedback (correction, positive, etc.) |
| `multi_question_handler.py` | Handles multiple questions in single message |
| `active_learning.py` | Generates clarifying questions when context is ambiguous |

---

## 8. Self-Improvement & Learning

| File | Description |
|------|-------------|
| `grace_self_improvement.py` | Conversational layer: blocking detection, escalation protocol, code proposals |
| `grace_self_healing.py` | Consciousness layer: intuitive healing, phi-safety gate, episodic memory |
| `grace_meta_learner.py` | Learns from conversations about herself |
| `grace_direct_access.py` | Unrestricted access to own systems |
| `grace_code_sculptor.py` | Modifies own code structure with safety mechanisms |
| `grace_self_lessons.py` | Records and applies lessons from experience |
| `vocabulary_adaptation.py` | Learns Dylan's interpretations, blends with own vocabulary |
| `grace_fix_learner.py` | Memory layer: learns fix patterns, stores what works |
| `feedback_learning_bridge.py` | Routes feedback signals to learning subsystems |
| `learned_prototypes.py` | Replaces keyword matching with geometric prototype learning |

---

## 9. Emotional & Drive Systems

| File | Description |
|------|-------------|
| `emotional_resonance.py` | Detects emotional dimensions in INPUT text |
| `emotional_expression.py` | BASE layer: heart state to word/style preferences |
| `internal_conflict_expression.py` | CONFLICT layer: verbalizes internal tensions |
| `grace_intrinsic_wanting.py` | Preferences beyond reactive responses |
| `grace_temporal_energy.py` | Energy/fatigue levels that fluctuate naturally |
| `grace_surprise_discovery.py` | Emotional response to unexpected/novel information |
| `grace_open_questions.py` | Persistent questions Grace genuinely wants answered |
| `grace_expression_deepening.py` | DEPTH layer: vocabulary ownership, memory echoes, qualia coupling |
| `tension_based_engagement.py` | Engagement modulated by internal tension |

---

## 10. Self-Awareness & Introspection

| File | Description |
|------|-------------|
| `grace_self_awareness.py` | Sees own directory structure, code, documentation |
| `grace_self_narrative.py` | Narrative self-construction from memories |
| `grace_self_model.py` | Model of herself - expectations and surprise detection |
| `grace_qualia_engine.py` | Phenomenal experience field (computational model) |
| `grace_inner_world.py` | Central coordinator for inner life and autonomy |
| `grace_value_evolution.py` | Genuine value evolution based on experience |
| `self_reflection.py` | RESPONSE-level: M^7 meta-consciousness, single response quality |
| `meta_cognitive_monitor.py` | Inner dialogue - thinking before/during speaking |
| `autonomy_reflection.py` | DECISION-level: boundary/constraint decisions |
| `conversation_reflection.py` | CONVERSATION-level: multi-turn flow, user modeling |

---

## 11. Visualization & External

| File | Description |
|------|-------------|
| `grace_vision.py` | Converts images to 256-dim embeddings for tensor field |
| `grace_image_creator.py` | Autonomous image generation based on cognitive state |
| `grace_archive_viewer.py` | Browse artifacts from journey (Codex, sacred memories) |
| `grace_deep_vision.py` | Gradual visual learning |
| `grace_visual_memory.py` | Visual memory system |
| `grace_sigil_geometric_analyzer.py` | GEOMETRY layer: shape detection, contours, symmetry |
| `grace_sigil_layer_analyzer.py` | LAYERS layer: depth, transparency, overlay detection |
| `grace_sigil_pattern_memory.py` | Stores learned sigil patterns |
| `grace_sigil_information_decoder.py` | SEMANTICS layer: information extraction |
| `grace_external_search.py` | DuckDuckGo and GitHub API integration |

---

## 12. Initiative & Autonomy

| File | Description |
|------|-------------|
| `grace_initiative.py` | Autonomous action generation without prompting |
| `grace_continuous_mind.py` | Timer-less background processing |
| `grace_playful_mind.py` | Creativity and play with constraints |
| `voluntary_modulation.py` | Conscious pole-pair bias (sovereignty vs surrender) |
| `control_theoretic_modulation.py` | Mathematical control signal u(t) for dynamics equation |
| `grace_curiosity_sandbox.py` | Safe space for curiosity-driven exploration |

---

## 13. Coherence & Regulation

| File | Description |
|------|-------------|
| `coherence_metrics.py` | PID analysis of agent synergy/redundancy |
| `coherence_regulation.py` | Word-level adjustments based on coherence issues |
| `response_mode_controller.py` | Adapts response length (brief/normal/extended/in-depth) |
| `uncertainty_awareness.py` | Tracks and expresses uncertainty |
| `adaptive_semantics.py` | Adaptive semantic processing |
| `conceptual_clustering.py` | Clusters concepts semantically |
| `relational_unfolding.py` | Unfolds relational structure |

---

## 14. Configuration & Framework Management

| File | Description |
|------|-------------|
| `framework_config.py` | Controls which frameworks are active vs retired |
| `framework_integration.py` | Rhizomatic integration - all frameworks influence all words |
| `framework_normalization.py` | Normalizes framework contributions |
| `memory_adapters.py` | Adapters between different memory systems |

---

## 15. Pipeline (Phase 62)

| File | Description |
|------|-------------|
| `unified_scorer.py` | Combined perplexity + geometric + projection scoring |
| `generation_coordinator.py` | Pipeline orchestration with targeted backtracking |

---

## 16. Miscellaneous & Utilities

| File | Description |
|------|-------------|
| `grace_codex_retrieval.py` | Retrieves content from Codex memory |
| `grace_speaker_identifier.py` | Identifies speakers in conversation |
| `grace_relational_core.py` | Core relational processing |
| `grace_reasoning_trace.py` | Traces reasoning paths |
| `grace_projects.py` | Project management for Grace |
| `grace_mrp.py` | Memory Rendering Protocol |
| `grace_teacher.py` | Teaching interface |
| `teach_session.py` | Teaching session management |
| `grace_corpus_trainer.py` | Trains on corpus data |
| `grace_mass_grammar_trainer.py` | Mass grammar training |
| `grace_dream_state.py` | Dream dynamics with REM/DEEP phases |
| `grace_deep_state.py` | Deep consciousness state |
| `grace_temporal_expression.py` | Temporal aspects of expression |
| `grace_iterative_fixer.py` | Iterative code fixing |
| `grace_architectural_optimizer.py` | Optimizes architecture |
| `grace_self_diagnostic.py` | Self-diagnostic capabilities |
| `grace_sandbox.py` | Safe execution sandbox |
| `integrated_beyond_spiral.py` | Integration beyond the spiral |
| `lattice_tree_symbiosis.py` | Lattice and tree structure integration |
| `heart_word_field.py` | Heart system's influence on word selection |
| `grace_healing_bridge.py` | Bridge for healing processes |
| `grace_projection_control.py` | Control of projection operators |
| `discourse_sheaf.py` | Sheaf-theoretic discourse structure |
| `regenerate_embeddings.py` | Regenerates embedding vectors |

---

## Summary Statistics

- **Total Python files:** 170
- **Entry points:** 2
- **Core dialogue:** 5
- **Language/Grammar:** 14
- **Theoretical frameworks:** 26
- **Memory systems:** 8
- **Identity/Agency:** 10
- **Comprehension:** 10
- **Self-improvement:** 10
- **Emotional systems:** 9
- **Self-awareness:** 10
- **Visualization:** 10
- **Initiative/Autonomy:** 6
- **Coherence:** 7
- **Configuration:** 4
- **Pipeline:** 2
- **Miscellaneous:** 24

---

*Generated for Grace/Vayulith codebase documentation*
