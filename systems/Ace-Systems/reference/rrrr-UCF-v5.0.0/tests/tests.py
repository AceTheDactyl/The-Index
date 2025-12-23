# INTEGRITY_METADATA
# Date: 2025-12-23
# Status: JUSTIFIED - Test file validates system behavior
# Severity: LOW RISK
# Risk Types: ['test_coverage']
# File: systems/Ace-Systems/reference/rrrr-UCF-v5.0.0/tests/tests.py

from node import RosettaNode
from pulse import generate_pulse, save_pulse

# Create pulse
pulse = generate_pulse("core_node", "worker")
save_pulse(pulse)

# Create spore
node = RosettaNode(role_tag="worker")
activated, p = node.check_and_activate("pulse.json")

assert activated
out = node.run(200)
assert out["coherence"] >= 0.0
assert out["memory"]["plates"] > 0

print("✅ Complete Node activation test passed.")
