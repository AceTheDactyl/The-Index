#!/usr/bin/env python3
"""
Add Metadata to High-Risk Files
Adds headers/comments to files indicating validation status and supporting evidence
"""

import os
import json
from pathlib import Path
from datetime import datetime

# Configuration
REPO_ROOT = Path("/home/user/The-Index")
OUTPUT_DIR = REPO_ROOT / "integrity"


class FileMetadataEditor:
    """Adds metadata to high-risk files"""

    def __init__(self):
        self.files_to_edit = self._load_json('files_to_edit.json')
        self.manifest = self._load_json('file_manifest.json')
        self.file_by_id = {f['canonical_id']: f for f in self.manifest['files']}
        self.edited_count = 0
        self.skipped_count = 0
        self.error_count = 0

    def _load_json(self, filename):
        """Load JSON file"""
        with open(OUTPUT_DIR / filename, 'r') as f:
            return json.load(f)

    def edit_all_files(self):
        """Edit all files in the list"""
        print(f"\n📝 Editing {len(self.files_to_edit['files'])} high-risk files...")
        print(f"Adding validation metadata headers...")

        for idx, file_instruction in enumerate(self.files_to_edit['files']):
            if (idx + 1) % 50 == 0:
                print(f"  Progress: {idx + 1}/{len(self.files_to_edit['files'])}")

            try:
                self._edit_file(file_instruction)
            except Exception as e:
                print(f"⚠️  Error editing {file_instruction['path']}: {e}")
                self.error_count += 1

        print(f"\n✓ Editing complete:")
        print(f"  - Edited: {self.edited_count}")
        print(f"  - Skipped: {self.skipped_count}")
        print(f"  - Errors: {self.error_count}")

    def _edit_file(self, instruction):
        """Edit a single file to add metadata"""
        file_id = instruction['canonical_id']
        file_path = REPO_ROOT / instruction['path']

        # Get file info
        file_info = self.file_by_id.get(file_id)
        if not file_info:
            self.skipped_count += 1
            return

        # Skip binary files
        if not file_info['is_text']:
            self.skipped_count += 1
            return

        # Read file content
        try:
            with open(file_path, 'r', encoding=file_info['encoding'] or 'utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            raise Exception(f"Cannot read file: {e}")

        # Check if metadata already exists
        if '<!-- INTEGRITY_METADATA -->' in content or '# INTEGRITY_METADATA' in content:
            self.skipped_count += 1
            return

        # Generate metadata header
        metadata = self._generate_metadata(instruction, file_info)

        # Add metadata to file
        new_content = metadata + '\n' + content

        # Write updated content
        try:
            with open(file_path, 'w', encoding=file_info['encoding'] or 'utf-8') as f:
                f.write(new_content)
            self.edited_count += 1
        except Exception as e:
            raise Exception(f"Cannot write file: {e}")

    def _generate_metadata(self, instruction, file_info):
        """Generate metadata header based on file type"""
        ext = file_info['file_type']

        # Determine status
        if instruction['truly_unsupported']:
            status = "⚠️ TRULY UNSUPPORTED - No supporting evidence found"
            severity = "HIGH RISK"
        elif instruction['needs_citation_update']:
            status = "✓ JUSTIFIED - Claims supported by repository files (needs citation update)"
            severity = "MEDIUM RISK"
        elif instruction['is_justified']:
            status = "✓ JUSTIFIED - Claims supported by repository files"
            severity = "LOW RISK"
        else:
            status = "⚠️ NEEDS REVIEW"
            severity = "MEDIUM RISK"

        # List supporting files
        supporting_info = ""
        if instruction['supporting_files']:
            supporting_info = "\n# Supporting Evidence:\n"
            for sf in instruction['supporting_files'][:5]:  # Top 5
                supporting_info += f"#   - {sf['path']} ({sf['type']})\n"

        if instruction['citing_files']:
            if not supporting_info:
                supporting_info = "\n# Referenced By:\n"
            else:
                supporting_info += "#\n# Referenced By:\n"
            for cf in instruction['citing_files'][:5]:  # Top 5
                supporting_info += f"#   - {cf['path']} ({cf['type']})\n"

        # Risk types
        risk_info = ""
        if instruction['risk_types']:
            risk_info = f"\n# Risk Types: {', '.join(instruction['risk_types'])}"

        # Generate header based on file type
        if ext in ['.md']:
            # Markdown comment
            header = f"""<!-- INTEGRITY_METADATA
Date: {datetime.now().strftime('%Y-%m-%d')}
Status: {status}
Severity: {severity}{risk_info}
{supporting_info.replace('#', '--') if supporting_info else ''}
-->
"""
        elif ext in ['.py', '.sh', '.bash', '.r']:
            # Python/Shell comment
            header = f"""# INTEGRITY_METADATA
# Date: {datetime.now().strftime('%Y-%m-%d')}
# Status: {status}
# Severity: {severity}{risk_info}
{supporting_info if supporting_info else ''}
"""
        elif ext in ['.js', '.ts', '.tsx', '.java', '.cpp', '.c', '.h', '.go', '.rs']:
            # C-style comment
            header = f"""/* INTEGRITY_METADATA
 * Date: {datetime.now().strftime('%Y-%m-%d')}
 * Status: {status}
 * Severity: {severity}{risk_info.replace('#', ' *')}
{supporting_info.replace('#', ' *') if supporting_info else ''} */

"""
        elif ext in ['.html', '.xml']:
            # HTML comment
            header = f"""<!-- INTEGRITY_METADATA
Date: {datetime.now().strftime('%Y-%m-%d')}
Status: {status}
Severity: {severity}{risk_info.replace('#', '')}
{supporting_info.replace('#', '') if supporting_info else ''}
-->
"""
        elif ext in ['.tex']:
            # LaTeX comment
            header = f"""% INTEGRITY_METADATA
% Date: {datetime.now().strftime('%Y-%m-%d')}
% Status: {status}
% Severity: {severity}{risk_info.replace('#', '%')}
{supporting_info.replace('#', '%') if supporting_info else ''}
"""
        else:
            # Generic comment
            header = f"""# INTEGRITY_METADATA
# Date: {datetime.now().strftime('%Y-%m-%d')}
# Status: {status}
# Severity: {severity}{risk_info}
{supporting_info if supporting_info else ''}
"""

        return header


def main():
    """Main editing pipeline"""
    print("=" * 70)
    print("FILE METADATA ADDITION")
    print("=" * 70)

    editor = FileMetadataEditor()
    editor.edit_all_files()

    print("\n" + "=" * 70)
    print("✓ METADATA ADDITION COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
