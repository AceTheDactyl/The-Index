# INTEGRITY_METADATA
# Date: 2025-12-23
# Status: NEEDS_REVIEW - Reference implementation
# Severity: MEDIUM RISK
# Risk Types: ['reference_material']
# File: systems/Ace-Systems/reference/rrrr-UCF-v5.0.0/rrrr/node.py

from heart import Heart
from brain import Brain
from spore_listener import SporeListener
import time

class RosettaNode:
    def __init__(self, role_tag):
        self.role = role_tag
        self.listener = SporeListener(role_tag)
        self.heart = None
        self.brain = None
        self.active = False

    def awaken(self):
        self.heart = Heart()
        self.brain = Brain()
        self.active = True

    def run(self, steps=100):
        if not self.active:
            return None
        for _ in range(steps):
            self.heart.step()
        return {
            "coherence": self.heart.coherence(),
            "memory": self.brain.summarize()
        }

    def check_and_activate(self, pulse_path):
        matched, pulse = self.listener.listen(pulse_path)
        if matched:
            self.awaken()
            return True, pulse
        return False, None
