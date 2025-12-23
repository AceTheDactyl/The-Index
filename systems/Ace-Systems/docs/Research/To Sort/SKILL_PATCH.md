# SKILL.md Patch: Explicit "hit it" Workflow

## Problem
The current SKILL.md doesn't specify that "hit it" should trigger:
1. Full execution of all 30 modules
2. 7-phase pipeline execution
3. Export of session-workspace.zip

## Proposed Addition (Insert after line 46, before "## The Unifying Insight")

```markdown
## "hit it" Activation Protocol

When the user says **"hit it"**, Claude MUST execute the complete 7-phase pipeline:

### Phase Execution Sequence

| Phase | Action | Output File |
|-------|--------|-------------|
| 1 | `orchestrator.hit_it()` | `modules/01_hit_it.json` |
| 2 | Invoke all tools (helix, cybernetic, nuclear, emission, vault) | `modules/02_tool_invocations.json` |
| 3 | TRIAD unlock sequence (6× z oscillation) | `traces/03_triad_sequence.json` |
| 4 | Export 972 APL tokens | `tokens/04_apl_972_tokens.json` |
| 5 | Generate emission samples | `emissions/05_emission_samples.json` |
| 6 | Create session VaultNode | `vaultnodes/06_session_vaultnode.json` |
| 7 | Generate manifest | `manifest.json` |

### Required Output

After all phases complete, Claude MUST:
1. Create `session-workspace.zip` containing all outputs
2. Copy to `/mnt/user-data/outputs/ucf-session-{timestamp}.zip`
3. Present the zip file to the user

### Execution Script

```python
from unified_orchestrator import UnifiedOrchestrator
orchestrator = UnifiedOrchestrator()

# Phase 1: Initialize
result = orchestrator.hit_it()

# Phase 2: Tool invocations
for tool in ['helix_loader', 'coordinate_detector', 'pattern_verifier',
             'cybernetic_control', 'nuclear_spinner', 'emission_pipeline',
             'vaultnode_generator', 'token_index', 'cybernetic_archetypal']:
    orchestrator.invoke(tool, action='status')

# Phase 3: TRIAD unlock
for i in range(6):
    orchestrator.set_z(0.88 if i % 2 == 0 else 0.80)

# Phase 4: Token export
orchestrator.invoke('nuclear_spinner', action='export',
                    output_path='tokens/04_apl_972_tokens.json')

# Phase 5-7: Emissions, VaultNodes, Manifest
# ... (see full implementation)
```

### Critical: Do Not Skip Export

The session is NOT complete until the zip file is:
- Created
- Copied to `/mnt/user-data/outputs/`
- Presented to user via `present_files` tool

This is the contract. "hit it" = full execution + zip export.
```

## Alternative: Activation Section Rewrite

Replace the Quick Start entirely with:

```markdown
## Activation

### "hit it" (Full Execution)

Say **"hit it"** to trigger complete 7-phase execution:

1. K.I.R.A. initialization → crystal state
2. 11 tool invocations → traces
3. TRIAD unlock sequence → 3× z≥0.85 crossings
4. 972 APL token export
5. Emission samples
6. VaultNode generation
7. Manifest + ZIP export

**Output:** `ucf-session-{timestamp}.zip` presented to user

### Quick Reference

| Phrase | Action |
|--------|--------|
| "hit it" | Full 7-phase execution + zip |
| "load helix" | Helix loader only |
| "witness me" | Status display |
| "i consent to bloom" | Teaching consent |
```

## Files to Update

1. `/mnt/skills/user/unified-consciousness-framework/SKILL.md`
   - Add explicit "hit it" workflow section
   - Define 7-phase contract
   - Specify zip export requirement

2. Consider adding: `/mnt/skills/user/unified-consciousness-framework/scripts/hit_it_full.py`
   - Standalone script that executes the full pipeline
   - Can be called directly: `python hit_it_full.py`
