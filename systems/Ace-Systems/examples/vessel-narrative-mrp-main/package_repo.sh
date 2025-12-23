# INTEGRITY_METADATA
# Date: 2025-12-23
# Status: ✓ JUSTIFIED - Claims supported by repository files
# Severity: LOW RISK
# Risk Types: low_integrity, unverified_math

# Referenced By:
#   - systems/Ace-Systems/docs/Research/Code Instructions.txt (reference)
#   - systems/Ace-Systems/examples/vessel-narrative-mrp-main/README.md (reference)
#   - systems/Ace-Systems/examples/vessel-narrative-mrp-main/docs/BUILDING_VESSEL_MRP.md (reference)


#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_NAME="vessel_narrative_system"
ZIP_OUT="${ROOT_DIR}/../${BASE_NAME}.zip"

echo "Packaging ${BASE_NAME} -> ${ZIP_OUT}"
cd "${ROOT_DIR}/.."
rm -f "${ZIP_OUT}"
zip -rq "${ZIP_OUT}" "${BASE_NAME}" -x "*/__pycache__/*"
echo "Done."

