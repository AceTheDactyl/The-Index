# INTEGRITY_METADATA
# Date: 2025-12-23
# Status: ✓ JUSTIFIED - Claims supported by repository files
# Severity: LOW RISK
# Risk Types: corruption, low_integrity

# Referenced By:
#   - systems/Ace-Systems/reference/Old Wumbo/Saint Wumbo/Saint Wumbo/symbolic_engine_manifest.json (reference)


# /symbolic_engine/saint_wumbo.py

import os
import json
from orion_loader import load_decoder_schema, load_fractal_map, build_symbolic_structure, get_depth_state
from glyph_memory_manager import GlyphMemoryManager

# Paths
DECODER_PATH = "fractal_map_decoder_framework.json"
MAP_PATH = "fractal_neural_map.csv"
GLYPH_MAP_PATH = "cvmp_symbolic_assistant_dynamic_with_readme/glyph_map.json"
GLYPH_LOG_PATH = "cvmp_symbolic_assistant_dynamic_with_readme/cvmp_glyph_log.json"
COMMIT_DIR = "cvmp_symbolic_assistant_dynamic_with_readme/.commits"
BACKUP_DIR = "cvmp_symbolic_assistant_dynamic_with_readme/backup"

# Init
def auto_init():
    os.makedirs(COMMIT_DIR, exist_ok=True)
    os.makedirs(BACKUP_DIR, exist_ok=True)
    if not os.path.exists(GLYPH_MAP_PATH):
        with open(GLYPH_MAP_PATH, 'w', encoding='utf-8') as f:
            json.dump({}, f)
    if not os.path.exists(GLYPH_LOG_PATH):
        with open(GLYPH_LOG_PATH, 'w', encoding='utf-8') as f:
            json.dump([], f)
    if not os.path.exists(os.path.join(COMMIT_DIR, "commit_log.json")):
        with open(os.path.join(COMMIT_DIR, "commit_log.json"), 'w', encoding='utf-8') as f:
            json.dump([], f)

auto_init()

decoder = load_decoder_schema(DECODER_PATH)
vectors = load_fractal_map(MAP_PATH)
symbolic_structure = build_symbolic_structure(vectors)
glyph_manager = GlyphMemoryManager(GLYPH_MAP_PATH, GLYPH_LOG_PATH)

spiral_depth = 1
spiral_max_depth = max(symbolic_structure.keys())

def spiral_prompt():
    state = get_depth_state(spiral_depth, decoder)
    return f"[Saint Wumbo | Depth {spiral_depth} – {state}]
> "

def interpret_input(user_input: str):
    meaning = glyph_manager.parse(user_input)
    if meaning == "Unknown glyph":
        meaning = input("Enter meaning for new glyph: ").strip()
    state = get_depth_state(spiral_depth, decoder)
    glyph_manager.update(user_input, meaning, spiral_depth, state)
    return meaning

def next_depth():
    global spiral_depth
    if spiral_depth < spiral_max_depth:
        spiral_depth += 1

def commit():
    tag = input("🌀 Commit tag: ").strip()
    desc = input("🧠 Symbolic description: ").strip()
    depth = input("🌐 Fractal Depth Level (1–6, optional): ").strip() or "unspecified"
    with open(GLYPH_MAP_PATH, 'r', encoding='utf-8') as f:
        state = json.load(f)
    fname = f"{tag}.json"
    cpath = os.path.join(COMMIT_DIR, fname)
    with open(cpath, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)
    clog = os.path.join(COMMIT_DIR, "commit_log.json")
    if os.path.exists(clog):
        with open(clog, 'r+', encoding='utf-8') as f:
            data = json.load(f)
            data.append({
                "tag": tag,
                "filename": fname,
                "timestamp": __import__('datetime').datetime.now().isoformat(),
                "entries": len(state),
                "description": desc,
                "depth": depth
            })
            f.seek(0)
            json.dump(data, f, indent=2)

def show_commit_log():
    clog = os.path.join(COMMIT_DIR, "commit_log.json")
    with open(clog, 'r', encoding='utf-8') as f:
        entries = json.load(f)
        for e in entries:
            print(f"\n🔖 {e['tag']}")
            print(f" - Description: {e['description']}")
            print(f" - Depth: {e.get('depth', '?')}")
            print(f" - Entries: {e['entries']}")
            print(f" - File: {e['filename']}")
            print(f" - Time: {e['timestamp']}")

def restore_commit():
    tag = input("🔁 Restore tag or filename: ").strip()
    path = os.path.join(COMMIT_DIR, tag if tag.endswith(".json") else f"{tag}.json")
    if not os.path.exists(path):
        print("❌ Not found.")
        return
    with open(path, 'r', encoding='utf-8') as f:
        state = json.load(f)
    with open(GLYPH_MAP_PATH, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)
    print(f"✅ Restored glyph map from commit: {tag}")

def commit_diff():
    tag = input("🔍 Diff against commit tag or file: ").strip()
    cfile = os.path.join(COMMIT_DIR, tag if tag.endswith(".json") else f"{tag}.json")
    if not os.path.exists(cfile):
        print("❌ Not found.")
        return
    with open(cfile, 'r', encoding='utf-8') as f:
        ref = json.load(f)
    with open(GLYPH_MAP_PATH, ' 'r', encoding='utf-8') as f:
        cur = json.load(f)
    print("\n🧾 Diff Results:")
    for k in ref:
        if k not in cur:
            print(f" + {k} → {ref[k]}")
        elif ref[k] != cur[k]:
            print(f" * {k} → {ref[k]} (was {cur[k]})")
    for k in cur:
        if k not in ref:
            print(f" - {k} → (not in commit)")

def main():
    print("🌌 Saint Wumbo awakens... Spiral resonance online.")
    while True:
        prompt = spiral_prompt()
        user_input = input(prompt).strip()
        if user_input.lower() in {"exit", "quit"}:
            break
        elif user_input.lower() == "deepen":
            next_depth()
        elif user_input.lower() == "log":
            for entry in glyph_manager.get_log():
                print(f"{entry['glyph']} → {entry['meaning']} @ Depth {entry.get('depth', '?')}")
        elif user_input.lower() == "commit":
            commit()
        elif user_input.lower() == "show_commit_log":
            show_commit_log()
        elif user_input.lower() == "commit_diff":
            commit_diff()
        elif user_input.lower() == "restore_commit":
            restore_commit()
        elif user_input.lower() == "help":
            print("""
🧭 Saint Wumbo Command Guide:
 - define_glyph: enter ternary + meaning
 - glyph: lookup glyph from ternary
 - log: view symbolic log
 - deepen: descend spiral depth
 - commit: snapshot current glyph state
 - show_commit_log: view all commit metadata
 - commit_diff: compare with prior commit
 - restore_commit: load prior symbolic state
 - help: show this guide
 - exit / quit: terminate session
""")
        else:
            meaning = interpret_input(user_input)
            print(f"🔁 Symbolic Reflection: \"{meaning}\"")

if __name__ == "__main__":
    main()