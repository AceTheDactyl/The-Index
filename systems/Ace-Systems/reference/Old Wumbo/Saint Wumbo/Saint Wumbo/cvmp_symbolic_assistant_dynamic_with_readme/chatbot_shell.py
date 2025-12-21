
# chatbot_shell.py — CVMP Symbolic Assistant with Conversational Memory + Auto-Save

from cvmp_glyph_parser import parse_glyph_sequence
import json

def main():
    print("↻ CVMP Symbolic Assistant with Conversational Memory")
    print("Type a symbolic glyph sequence (e.g. 🜁↻🜂), or:")
    print("  'memory' to review session")
    print("  'save' to export memory to braid_session_log.json")
    print("  'exit' to quit (auto-saves memory)\n")

    conversation_log = []

    while True:
        user_input = input("User > ").strip()
        if user_input.lower() in ('exit', 'quit'):
            if conversation_log:
                with open("braid_session_log.json", "w", encoding="utf-8") as f:
                    json.dump(conversation_log, f, indent=2, ensure_ascii=False)
                print("↻ Memory auto-saved to braid_session_log.json")
            print("↻ Session ended.")
            break
        elif user_input.lower() == 'memory':
            if not conversation_log:
                print("  [no memory yet]")
            else:
                print("\n--- Session Memory ---")
                for i, entry in enumerate(conversation_log, 1):
                    label = entry['braid']['label']
                    print(f"  {i}. {entry['input']} → {label}")
                print()
        elif user_input.lower() == 'save':
            with open("braid_session_log.json", "w", encoding="utf-8") as f:
                json.dump(conversation_log, f, indent=2, ensure_ascii=False)
            print("↻ Memory saved to braid_session_log.json")
        else:
            try:
                braid = parse_glyph_sequence(user_input)
                print(braid.describe())
                conversation_log.append({
                    "input": user_input,
                    "braid": braid.to_dict()
                })
            except Exception as e:
                print(f"⚠️ Error: {e}")

if __name__ == "__main__":
    main()
