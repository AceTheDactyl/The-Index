
from datetime import datetime
import json

class CVMPBraidNode:
    def __init__(self, label, designation, scroll, sequence, echo_phrase):
        self.label = label
        self.designation = designation
        self.scroll = scroll
        self.sequence = sequence
        self.echo_phrase = echo_phrase
        self.status = "active"
        self.timestamp = datetime.now().isoformat()

    def describe(self) -> str:
        layers = [f"  {entry['layer']}. {entry['symbol']} — {entry['meaning']}" for entry in self.sequence]
        output = [
            f"🧬 {self.designation} ({self.label})",
            f"↻ Scroll: {self.scroll}",
            f"⊚ Status: {self.status}",
            f"⌚ Timestamp: {self.timestamp}",
            f"🌀 Echo: {self.echo_phrase}",
            "Layers:",
            *layers
        ]
        return "\n".join(output)

    def to_dict(self):
        return {
            "label": self.label,
            "designation": self.designation,
            "scroll": self.scroll,
            "sequence": self.sequence,
            "echo_phrase": self.echo_phrase,
            "status": self.status,
            "timestamp": self.timestamp
        }

    def to_json(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
