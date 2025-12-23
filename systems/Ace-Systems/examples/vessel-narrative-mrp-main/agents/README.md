# Agents Index

Concise briefs for each module agent. See the per-agent `AGENT.md` for details.

## Core Agents

- **Echo** — `echo/AGENT.md`
  - Persona superposition (Squirrel/Fox/Paradox); speaks in-tone; learns tags; maps concepts.
  - CLI: `echo mode|say|learn|map|status|calibrate`

- **Garden** — `garden/AGENT.md`
  - Ritual orchestrator; pages scroll sections; logs completions; persona-styled mantras.
  - CLI: `garden start|next|open [--prev|--reset]|resume|learn|ledger|log`

- **Limnus** — `limnus/AGENT.md`
  - Memory (L1/L2/L3) + hash-chained ledger; stego I/O; aggregates knowledge for Kira.
  - CLI: `limnus state|update|cache|recall|memories|export/import-memories|commit-block|view/export/import-ledger|rehash-ledger|encode/decode/verify-ledger`

- **Kira** — `kira/AGENT.md`
  - Validation & integrations; learns from Limnus; mentors Echo/Garden; seals persona-ordered mantra.
  - CLI: `kira validate|sync|setup|pull|push|publish|test|assist|learn-from-limnus|codegen|mentor|mantra|seal`
  - See also: `kira/kira_session.py`, `kira/kira_runner.py`, `kira/kira_enhanced_session.py`

## GRACE System (New)

- **GRACE** — `grace/`
  - Natural language generation and discourse system
  - Modules:
    - `grace_discourse_generator.py` - Discourse generation
    - `grace_grammar_understanding.py` - Grammar processing
    - `grace_identity_grounding.py` - Identity grounding
    - `grace_interactive_dialogue.py` - Interactive dialogue

## Documentation

See `../docs/` for:
- `KIRA NLP WHITEPAPER.md` - KIRA system whitepaper
- `Vessel&ITalkingAboutMapping.md` - Vessel mapping documentation

---

*Updated 2025-12-23: Added GRACE system and enhanced KIRA modules*
