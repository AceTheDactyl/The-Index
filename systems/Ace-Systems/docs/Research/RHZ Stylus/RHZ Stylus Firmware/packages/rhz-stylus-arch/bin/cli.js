// INTEGRITY_METADATA
// Date: 2025-12-23
// Status: NEEDS_REVIEW - Research/experimental code
// Severity: MEDIUM RISK
// Risk Types: ['experimental', 'needs_validation']
// File: systems/Ace-Systems/docs/Research/RHZ Stylus/RHZ Stylus Firmware/packages/rhz-stylus-arch/bin/cli.js

#!/usr/bin/env node
import { architecture, llmGuide } from "../index.js";

const arg = process.argv[2] || "all";

if (arg === "--llm" || arg === "llm") {
  console.log(llmGuide);
} else if (arg === "--arch" || arg === "arch") {
  console.log(architecture);
} else {
  console.log(architecture);
  console.log("\n--- LLM Usage Guide ---\n");
  console.log(llmGuide);
}

