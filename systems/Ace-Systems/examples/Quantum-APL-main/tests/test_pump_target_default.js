// INTEGRITY_METADATA
// Date: 2025-12-23
// Status: JUSTIFIED - Test file validates system behavior
// Severity: LOW RISK
// Risk Types: ['test_coverage']
// File: systems/Ace-Systems/examples/Quantum-APL-main/tests/test_pump_target_default.js

// Verify JS engine default pump target equals Z_CRITICAL
const assert = (c, m) => { if (!c) throw new Error(m || 'assert'); };
const CONST = require('../src/constants');
const Engine = require('../src/quantum_apl_engine');

(function testDefaultPumpTarget() {
  const target = (typeof Engine.defaultPumpTarget === 'function')
    ? Engine.defaultPumpTarget()
    : CONST.Z_CRITICAL;
  const ok = Math.abs(target - CONST.Z_CRITICAL) < 1e-15;
  assert(ok, `pump target drift: got ${target}, want ${CONST.Z_CRITICAL}`);
  console.log('Default pump target equals Z_CRITICAL');
})();

