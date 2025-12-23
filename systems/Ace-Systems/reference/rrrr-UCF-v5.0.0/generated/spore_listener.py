# INTEGRITY_METADATA
# Date: 2025-12-23
# Status: JUSTIFIED - Auto-generated code
# Severity: LOW RISK
# Risk Types: ['generated']
# File: systems/Ace-Systems/reference/rrrr-UCF-v5.0.0/generated/spore_listener.py

import json, time

class SporeListener:
    def __init__(self, role_tag):
        self.role_tag = role_tag
        self.dormant = True

    def check_pulse(self, pulse):
        if pulse["intent"] == self.role_tag:
            return True
        return False

    def listen(self, pulse_path):
        try:
            with open(pulse_path) as f:
                pulse = json.load(f)
            if self.check_pulse(pulse):
                self.dormant = False
                return True, pulse
        except FileNotFoundError:
            pass
        return False, None
