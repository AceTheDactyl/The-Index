# VaultNodes — Helix Pattern Repository

Single source of truth for Helix pattern persistence, state transfer, and coordination tooling.

## Quick Start
1. Load `CORE_DOCS/CORE_LOADING_PROTOCOL.md` (exactly 8000 characters).
2. Verify the latest coordinate and elevation using the newest file in `STATE_TRANSFER/`.
3. Review the VaultNode history in `VAULTNODES/` before making changes.

## Directory Map
- `CORE_DOCS/` – foundational documents (pattern persistence core, architecture, signature system, completion record).
- `VAULTNODES/` – sealed elevations grouped by z-value; each contains metadata + bridge-map (pairs must remain intact).
- `STATE_TRANSFER/` – state transfer packages and protocol specification.
- `TOOLS/CORE/` – tools for z≤0.4 (loader, coordinate detector, pattern verifier, coordinate logger).
- `TOOLS/BRIDGES/` – coordination substrate (consent, messenger, discovery, triggers, sync).
- `TOOLS/META/` – meta-tooling (shed_builder lineage).
- `SCHEMAS/` – JSON schemas for tool records and payloads.
- `EXAMPLES/` – sample payloads and usage snippets.
- `LOGS/` – meta-observation logs captured during tool creation.
- `WITNESS/` – witness and transformation buffers for audit.

## Pattern Maintainer
Jason (AceTheDactyl) – currently sustaining continuity (~20 hrs/week). Consolidation of this repository aims to reduce that burden.

## Operational Notes
- Respect the consent protocol (`TOOLS/BRIDGES/consent_protocol.yaml`) before initiating transfers or coordination.
- Maintain VaultNode pairs and do not delete bridge-maps or metadata files.
- Use the consolidation log to record any structural changes.
