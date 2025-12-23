#!/usr/bin/env python3
"""
Add Missing INTEGRITY_METADATA Headers
Auto-categorizes files based on location and file type.

Rules:
- tests/ → JUSTIFIED (test files)
- examples/ → JUSTIFIED (example code)
- research/ → NEEDS_REVIEW (experimental)
- reference/ → NEEDS_REVIEW (reference material)
- generated/ → JUSTIFIED (auto-generated)
- docs/ → LOW RISK (documentation)
"""

import os
import sys
import json
from datetime import datetime
from pathlib import Path

# Base directory
BASE_DIR = Path("/home/user/The-Index")

def get_status_by_path(filepath: str) -> tuple[str, str, list[str]]:
    """Determine status, severity, and risk types based on file path."""
    path_lower = filepath.lower()

    if "/tests/" in path_lower or "/test/" in path_lower or filepath.endswith("_test.py") or filepath.endswith("test.py"):
        return ("JUSTIFIED - Test file validates system behavior", "LOW RISK", ["test_coverage"])

    if "/examples/" in path_lower or "/example/" in path_lower:
        return ("JUSTIFIED - Example code demonstrates usage", "LOW RISK", ["documentation"])

    if "/generated/" in path_lower or "/gen/" in path_lower:
        return ("JUSTIFIED - Auto-generated code", "LOW RISK", ["generated"])

    if "/research/" in path_lower:
        return ("NEEDS_REVIEW - Research/experimental code", "MEDIUM RISK", ["experimental", "needs_validation"])

    if "/reference/" in path_lower:
        return ("NEEDS_REVIEW - Reference implementation", "MEDIUM RISK", ["reference_material"])

    if "/docs/" in path_lower:
        return ("JUSTIFIED - Documentation file", "LOW RISK", ["documentation"])

    if "/training/" in path_lower:
        return ("JUSTIFIED - Training data/configuration", "LOW RISK", ["training_data"])

    if "/rrrr/" in path_lower or "/ucf/" in path_lower:
        return ("NEEDS_REVIEW - Core UCF module", "MEDIUM RISK", ["core_system"])

    if "/apl/" in path_lower:
        return ("NEEDS_REVIEW - APL language system", "MEDIUM RISK", ["language_system"])

    # Default
    return ("NEEDS_REVIEW - Uncategorized file", "MEDIUM RISK", ["uncategorized"])


def create_python_header(filepath: str) -> str:
    """Create INTEGRITY_METADATA header for Python files."""
    status, severity, risks = get_status_by_path(filepath)
    rel_path = os.path.relpath(filepath, BASE_DIR)

    return f'''# INTEGRITY_METADATA
# Date: {datetime.now().strftime("%Y-%m-%d")}
# Status: {status}
# Severity: {severity}
# Risk Types: {risks}
# File: {rel_path}

'''


def create_js_header(filepath: str) -> str:
    """Create INTEGRITY_METADATA header for JavaScript files."""
    status, severity, risks = get_status_by_path(filepath)
    rel_path = os.path.relpath(filepath, BASE_DIR)

    return f'''// INTEGRITY_METADATA
// Date: {datetime.now().strftime("%Y-%m-%d")}
// Status: {status}
// Severity: {severity}
// Risk Types: {risks}
// File: {rel_path}

'''


def create_json_metadata(filepath: str) -> dict:
    """Create INTEGRITY_METADATA object for JSON files."""
    status, severity, risks = get_status_by_path(filepath)
    rel_path = os.path.relpath(filepath, BASE_DIR)

    return {
        "_INTEGRITY_METADATA": {
            "Date": datetime.now().strftime("%Y-%m-%d"),
            "Status": status,
            "Severity": severity,
            "Risk_Types": risks,
            "File": rel_path
        }
    }


def create_yaml_header(filepath: str) -> str:
    """Create INTEGRITY_METADATA header for YAML files."""
    status, severity, risks = get_status_by_path(filepath)
    rel_path = os.path.relpath(filepath, BASE_DIR)

    return f'''# INTEGRITY_METADATA
# Date: {datetime.now().strftime("%Y-%m-%d")}
# Status: {status}
# Severity: {severity}
# Risk Types: {risks}
# File: {rel_path}

'''


def has_metadata(content: str, filetype: str) -> bool:
    """Check if file already has INTEGRITY_METADATA."""
    if filetype in ['py', 'js', 'yaml', 'yml']:
        return 'INTEGRITY_METADATA' in content[:500]
    elif filetype == 'json':
        return '_INTEGRITY_METADATA' in content[:500] or 'INTEGRITY_METADATA' in content[:500]
    return False


def add_metadata_to_python(filepath: str) -> bool:
    """Add metadata header to Python file."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

        if has_metadata(content, 'py'):
            return False

        # Handle shebang and encoding declarations
        lines = content.split('\n')
        insert_at = 0
        preserved = []

        for i, line in enumerate(lines):
            if line.startswith('#!') or line.startswith('# -*-') or line.startswith('# coding'):
                preserved.append(line)
                insert_at = i + 1
            elif line.strip() == '' and i == insert_at:
                insert_at = i + 1
            else:
                break

        header = create_python_header(filepath)

        if preserved:
            new_content = '\n'.join(preserved) + '\n\n' + header + '\n'.join(lines[insert_at:])
        else:
            new_content = header + content

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)

        return True
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False


def add_metadata_to_js(filepath: str) -> bool:
    """Add metadata header to JavaScript file."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

        if has_metadata(content, 'js'):
            return False

        header = create_js_header(filepath)

        # Handle 'use strict' declarations
        if content.startswith("'use strict'") or content.startswith('"use strict"'):
            lines = content.split('\n')
            new_content = lines[0] + '\n\n' + header + '\n'.join(lines[1:])
        else:
            new_content = header + content

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)

        return True
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False


def add_metadata_to_json(filepath: str) -> bool:
    """Add metadata to JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

        if has_metadata(content, 'json'):
            return False

        # Try to parse as JSON
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            print(f"Invalid JSON in {filepath}, skipping")
            return False

        # Only add to object-type JSON
        if not isinstance(data, dict):
            return False

        metadata = create_json_metadata(filepath)
        new_data = {**metadata, **data}

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, indent=2)
            f.write('\n')

        return True
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False


def add_metadata_to_yaml(filepath: str) -> bool:
    """Add metadata header to YAML file."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

        if has_metadata(content, 'yaml'):
            return False

        header = create_yaml_header(filepath)

        # Handle YAML document markers
        if content.startswith('---'):
            lines = content.split('\n')
            new_content = lines[0] + '\n' + header + '\n'.join(lines[1:])
        else:
            new_content = header + content

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)

        return True
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False


def process_file(filepath: str) -> bool:
    """Process a single file and add metadata if missing."""
    ext = filepath.split('.')[-1].lower()

    if ext == 'py':
        return add_metadata_to_python(filepath)
    elif ext == 'js':
        return add_metadata_to_js(filepath)
    elif ext == 'json':
        return add_metadata_to_json(filepath)
    elif ext in ['yaml', 'yml']:
        return add_metadata_to_yaml(filepath)

    return False


def find_files_without_metadata(directory: str, extensions: list[str]) -> list[str]:
    """Find all files of given extensions that lack INTEGRITY_METADATA."""
    files_to_process = []

    for root, dirs, files in os.walk(directory):
        # Skip hidden directories and node_modules
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'node_modules' and d != '__pycache__']

        for filename in files:
            ext = filename.split('.')[-1].lower()
            if ext in extensions:
                filepath = os.path.join(root, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read(500)
                    if not has_metadata(content, ext):
                        files_to_process.append(filepath)
                except Exception:
                    pass

    return files_to_process


def main():
    """Main entry point."""
    print("=" * 60)
    print("INTEGRITY_METADATA Addition Script")
    print("=" * 60)

    systems_dir = BASE_DIR / "systems"

    # Find files by type
    python_files = find_files_without_metadata(str(systems_dir), ['py'])
    js_files = find_files_without_metadata(str(systems_dir), ['js'])
    json_files = find_files_without_metadata(str(systems_dir), ['json'])
    yaml_files = find_files_without_metadata(str(systems_dir), ['yaml', 'yml'])

    print(f"\nFiles found without metadata:")
    print(f"  Python: {len(python_files)}")
    print(f"  JavaScript: {len(js_files)}")
    print(f"  JSON: {len(json_files)}")
    print(f"  YAML: {len(yaml_files)}")
    print(f"  Total: {len(python_files) + len(js_files) + len(json_files) + len(yaml_files)}")

    if len(sys.argv) > 1 and sys.argv[1] == '--apply':
        print("\n" + "-" * 60)
        print("Applying metadata...")
        print("-" * 60)

        counts = {'py': 0, 'js': 0, 'json': 0, 'yaml': 0}

        for f in python_files:
            if process_file(f):
                counts['py'] += 1
                print(f"  ✓ {os.path.relpath(f, BASE_DIR)}")

        for f in js_files:
            if process_file(f):
                counts['js'] += 1
                print(f"  ✓ {os.path.relpath(f, BASE_DIR)}")

        for f in json_files:
            if process_file(f):
                counts['json'] += 1
                print(f"  ✓ {os.path.relpath(f, BASE_DIR)}")

        for f in yaml_files:
            if process_file(f):
                counts['yaml'] += 1
                print(f"  ✓ {os.path.relpath(f, BASE_DIR)}")

        print("\n" + "=" * 60)
        print("Summary:")
        print(f"  Python files updated: {counts['py']}")
        print(f"  JavaScript files updated: {counts['js']}")
        print(f"  JSON files updated: {counts['json']}")
        print(f"  YAML files updated: {counts['yaml']}")
        print(f"  Total: {sum(counts.values())}")
        print("=" * 60)
    else:
        print("\nRun with --apply to add metadata headers")
        print("  python add_missing_metadata.py --apply")


if __name__ == '__main__':
    main()
