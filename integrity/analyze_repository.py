#!/usr/bin/env python3
"""
Repository Integrity & Research Validation Analysis
Full-repo verification, deduplication, provenance scoring, and integrity reporting
"""

import os
import hashlib
import json
import re
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import mimetypes

# Configuration
REPO_ROOT = Path("/home/user/The-Index")
OUTPUT_DIR = REPO_ROOT / "integrity"
EXCLUDE_DIRS = {'.git', '__pycache__', 'node_modules', '.venv', 'venv'}
EXCLUDE_FILES = {'.gitignore', '.DS_Store'}

class FileInventory:
    """Manages file enumeration and metadata collection"""

    def __init__(self):
        self.files = []
        self.canonical_map = {}

    def enumerate_files(self):
        """Recursively walk repository and collect file metadata"""
        print("📁 Enumerating repository files...")

        file_id = 1
        for root, dirs, files in os.walk(REPO_ROOT):
            # Exclude certain directories
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

            for filename in files:
                if filename in EXCLUDE_FILES:
                    continue

                filepath = Path(root) / filename
                rel_path = filepath.relative_to(REPO_ROOT)

                # Skip the integrity directory itself
                if str(rel_path).startswith('integrity/'):
                    continue

                try:
                    file_info = self._gather_metadata(filepath, rel_path, file_id)
                    self.files.append(file_info)
                    self.canonical_map[file_info['canonical_id']] = {
                        'path': str(rel_path),
                        'sha256': file_info['sha256'],
                        'md5': file_info['md5']
                    }
                    file_id += 1
                except Exception as e:
                    print(f"⚠️  Error processing {rel_path}: {e}")

        print(f"✓ Enumerated {len(self.files)} files")
        return self.files

    def _gather_metadata(self, filepath, rel_path, file_id):
        """Collect comprehensive metadata for a single file"""
        stat = filepath.stat()

        # Determine file type
        extension = filepath.suffix.lower()
        mime_type, _ = mimetypes.guess_type(str(filepath))

        # Read file for hashing
        try:
            with open(filepath, 'rb') as f:
                content = f.read()
        except Exception as e:
            raise Exception(f"Cannot read file: {e}")

        # Compute hashes
        sha256_hash = hashlib.sha256(content).hexdigest()
        md5_hash = hashlib.md5(content).hexdigest()

        # Try to detect encoding and count lines for text files
        encoding = None
        line_count = None
        is_text = False

        if self._is_text_file(extension, mime_type):
            encoding, line_count = self._analyze_text_file(content)
            is_text = encoding is not None

        return {
            'canonical_id': f'FILE-{file_id:04d}',
            'relative_path': str(rel_path),
            'file_type': extension,
            'mime_type': mime_type,
            'byte_size': stat.st_size,
            'last_modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'sha256': sha256_hash,
            'md5': md5_hash,
            'encoding': encoding,
            'line_count': line_count,
            'is_text': is_text
        }

    def _is_text_file(self, extension, mime_type):
        """Determine if file is likely text-based"""
        text_extensions = {
            '.md', '.txt', '.py', '.js', '.json', '.csv', '.html', '.css',
            '.xml', '.yaml', '.yml', '.toml', '.ini', '.sh', '.bash',
            '.cpp', '.c', '.h', '.hpp', '.java', '.rs', '.go', '.rb',
            '.tex', '.bib', '.sql', '.r', '.m', '.swift', '.kt'
        }

        if extension in text_extensions:
            return True

        if mime_type and mime_type.startswith('text/'):
            return True

        return False

    def _analyze_text_file(self, content):
        """Detect encoding and count lines for text files"""
        # Try multiple encodings
        encodings = ['utf-8', 'ascii', 'latin-1', 'cp1252']

        for enc in encodings:
            try:
                text = content.decode(enc)
                line_count = text.count('\n') + (1 if text and not text.endswith('\n') else 0)
                return enc, line_count
            except (UnicodeDecodeError, AttributeError):
                continue

        return None, None

    def save_manifest(self):
        """Save file manifest to JSON"""
        output_file = OUTPUT_DIR / 'file_manifest.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'generated': datetime.now().isoformat(),
                'total_files': len(self.files),
                'files': self.files
            }, f, indent=2)
        print(f"✓ Saved file manifest: {output_file}")

        # Save canonical mapping
        canonical_file = OUTPUT_DIR / 'canonical_mapping.json'
        with open(canonical_file, 'w', encoding='utf-8') as f:
            json.dump(self.canonical_map, f, indent=2)
        print(f"✓ Saved canonical mapping: {canonical_file}")


class FormatValidator:
    """Validates file formats and structure"""

    def __init__(self, files):
        self.files = files
        self.validation_results = []

    def validate_all(self):
        """Validate all files based on their type"""
        print("\n🔍 Validating file formats...")

        for file_info in self.files:
            result = self._validate_file(file_info)
            self.validation_results.append(result)

        valid_count = sum(1 for r in self.validation_results if r['status'] == 'valid')
        print(f"✓ Validated {len(self.files)} files ({valid_count} valid)")

        return self.validation_results

    def _validate_file(self, file_info):
        """Validate a single file based on its type"""
        filepath = REPO_ROOT / file_info['relative_path']
        extension = file_info['file_type']

        result = {
            'canonical_id': file_info['canonical_id'],
            'path': file_info['relative_path'],
            'file_type': extension,
            'status': 'unknown',
            'issues': []
        }

        # Skip binary files that we can't validate
        if not file_info['is_text'] and extension not in ['.pdf', '.png', '.jpg', '.jpeg', '.gif']:
            result['status'] = 'skipped'
            result['issues'].append('binary file - validation skipped')
            return result

        try:
            # Markdown validation
            if extension == '.md':
                return self._validate_markdown(filepath, result, file_info)

            # JSON validation
            elif extension == '.json':
                return self._validate_json(filepath, result)

            # CSV validation
            elif extension == '.csv':
                return self._validate_csv(filepath, result, file_info)

            # Python validation
            elif extension == '.py':
                return self._validate_python(filepath, result, file_info)

            # JavaScript validation
            elif extension == '.js':
                return self._validate_javascript(filepath, result, file_info)

            # PDF validation
            elif extension == '.pdf':
                return self._validate_pdf(filepath, result)

            # Plain text validation
            elif extension == '.txt':
                return self._validate_text(filepath, result, file_info)

            # Generic text file
            elif file_info['is_text']:
                result['status'] = 'valid'
                return result

            else:
                result['status'] = 'skipped'
                result['issues'].append('unknown file type')
                return result

        except Exception as e:
            result['status'] = 'error'
            result['issues'].append(f'validation error: {str(e)}')
            return result

    def _validate_markdown(self, filepath, result, file_info):
        """Validate Markdown file"""
        if not file_info['encoding']:
            result['status'] = 'corrupt'
            result['issues'].append('unreadable - encoding detection failed')
            return result

        try:
            with open(filepath, 'r', encoding=file_info['encoding']) as f:
                content = f.read()

            # Basic Markdown checks
            if not content.strip():
                result['issues'].append('empty file')
                result['status'] = 'warning'
            else:
                result['status'] = 'valid'

            return result
        except Exception as e:
            result['status'] = 'corrupt'
            result['issues'].append(f'read error: {str(e)}')
            return result

    def _validate_json(self, filepath, result):
        """Validate JSON file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                json.load(f)
            result['status'] = 'valid'
        except json.JSONDecodeError as e:
            result['status'] = 'corrupt'
            result['issues'].append(f'invalid JSON: {str(e)}')
        except Exception as e:
            result['status'] = 'error'
            result['issues'].append(f'read error: {str(e)}')

        return result

    def _validate_csv(self, filepath, result, file_info):
        """Validate CSV file"""
        try:
            with open(filepath, 'r', encoding=file_info['encoding'] or 'utf-8') as f:
                lines = f.readlines()

            if not lines:
                result['status'] = 'warning'
                result['issues'].append('empty file')
                return result

            # Check for consistent column count
            col_counts = [len(line.split(',')) for line in lines if line.strip()]
            if len(set(col_counts)) > 1:
                result['status'] = 'warning'
                result['issues'].append('inconsistent column count')
            else:
                result['status'] = 'valid'

        except Exception as e:
            result['status'] = 'error'
            result['issues'].append(f'validation error: {str(e)}')

        return result

    def _validate_python(self, filepath, result, file_info):
        """Validate Python file"""
        try:
            with open(filepath, 'r', encoding=file_info['encoding'] or 'utf-8') as f:
                code = f.read()

            # Try to compile
            compile(code, str(filepath), 'exec')
            result['status'] = 'valid'

        except SyntaxError as e:
            result['status'] = 'corrupt'
            result['issues'].append(f'syntax error: {str(e)}')
        except Exception as e:
            result['status'] = 'error'
            result['issues'].append(f'validation error: {str(e)}')

        return result

    def _validate_javascript(self, filepath, result, file_info):
        """Validate JavaScript file (basic check)"""
        try:
            with open(filepath, 'r', encoding=file_info['encoding'] or 'utf-8') as f:
                content = f.read()

            # Basic syntax checks
            if content.strip():
                result['status'] = 'valid'
            else:
                result['status'] = 'warning'
                result['issues'].append('empty file')

        except Exception as e:
            result['status'] = 'error'
            result['issues'].append(f'validation error: {str(e)}')

        return result

    def _validate_pdf(self, filepath, result):
        """Validate PDF file"""
        try:
            # Check if file starts with PDF magic bytes
            with open(filepath, 'rb') as f:
                magic = f.read(4)

            if magic == b'%PDF':
                result['status'] = 'valid'
            else:
                result['status'] = 'corrupt'
                result['issues'].append('not a valid PDF file')

        except Exception as e:
            result['status'] = 'error'
            result['issues'].append(f'validation error: {str(e)}')

        return result

    def _validate_text(self, filepath, result, file_info):
        """Validate plain text file"""
        if not file_info['encoding']:
            result['status'] = 'corrupt'
            result['issues'].append('encoding detection failed')
            return result

        try:
            with open(filepath, 'r', encoding=file_info['encoding']) as f:
                f.read()
            result['status'] = 'valid'
        except Exception as e:
            result['status'] = 'error'
            result['issues'].append(f'read error: {str(e)}')

        return result

    def save_validation_results(self):
        """Save validation results to JSON"""
        output_file = OUTPUT_DIR / 'format_validation.json'

        # Compute statistics
        stats = defaultdict(int)
        for result in self.validation_results:
            stats[result['status']] += 1

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'generated': datetime.now().isoformat(),
                'statistics': dict(stats),
                'results': self.validation_results
            }, f, indent=2)

        print(f"✓ Saved validation results: {output_file}")


class DuplicateDetector:
    """Detects exact and near-duplicate files"""

    def __init__(self, files):
        self.files = files
        self.exact_duplicates = []
        self.near_duplicates = []

    def find_exact_duplicates(self):
        """Find files with identical SHA-256 hashes"""
        print("\n🔍 Detecting exact duplicates...")

        hash_map = defaultdict(list)

        for file_info in self.files:
            hash_map[file_info['sha256']].append(file_info)

        # Find duplicates
        for sha256, file_list in hash_map.items():
            if len(file_list) > 1:
                # Sort by modification time to find canonical (earliest)
                sorted_files = sorted(file_list, key=lambda x: x['last_modified'])

                canonical = sorted_files[0]
                duplicates = sorted_files[1:]

                total_duplicate_size = sum(f['byte_size'] for f in duplicates)

                self.exact_duplicates.append({
                    'sha256': sha256,
                    'canonical': {
                        'id': canonical['canonical_id'],
                        'path': canonical['relative_path'],
                        'size': canonical['byte_size']
                    },
                    'duplicates': [
                        {
                            'id': f['canonical_id'],
                            'path': f['relative_path'],
                            'size': f['byte_size']
                        }
                        for f in duplicates
                    ],
                    'storage_savings_bytes': total_duplicate_size
                })

        total_savings = sum(d['storage_savings_bytes'] for d in self.exact_duplicates)
        print(f"✓ Found {len(self.exact_duplicates)} duplicate groups (potential savings: {total_savings:,} bytes)")

        return self.exact_duplicates

    def find_near_duplicates(self):
        """Find semantically similar text files"""
        print("\n🔍 Detecting near-duplicates (text files)...")

        # Filter text files only
        text_files = [f for f in self.files if f['is_text']]

        # Group by file type for comparison
        by_extension = defaultdict(list)
        for file_info in text_files:
            by_extension[file_info['file_type']].append(file_info)

        # Compare files of the same type
        for ext, file_list in by_extension.items():
            if len(file_list) < 2:
                continue

            # Compare pairwise (for small repos this is acceptable)
            for i in range(len(file_list)):
                for j in range(i + 1, len(file_list)):
                    similarity = self._compute_similarity(file_list[i], file_list[j])

                    if similarity >= 0.80:
                        self.near_duplicates.append({
                            'file_1': {
                                'id': file_list[i]['canonical_id'],
                                'path': file_list[i]['relative_path']
                            },
                            'file_2': {
                                'id': file_list[j]['canonical_id'],
                                'path': file_list[j]['relative_path']
                            },
                            'similarity': round(similarity, 3),
                            'category': 'near-duplicate' if similarity >= 0.95 else 'derivative'
                        })

        print(f"✓ Found {len(self.near_duplicates)} near-duplicate pairs")
        return self.near_duplicates

    def _compute_similarity(self, file1, file2):
        """Compute text similarity between two files"""
        try:
            filepath1 = REPO_ROOT / file1['relative_path']
            filepath2 = REPO_ROOT / file2['relative_path']

            # Read contents
            with open(filepath1, 'r', encoding=file1['encoding'] or 'utf-8') as f:
                content1 = f.read()
            with open(filepath2, 'r', encoding=file2['encoding'] or 'utf-8') as f:
                content2 = f.read()

            # Simple similarity: line-based Jaccard similarity
            lines1 = set(content1.split('\n'))
            lines2 = set(content2.split('\n'))

            if not lines1 or not lines2:
                return 0.0

            intersection = len(lines1 & lines2)
            union = len(lines1 | lines2)

            return intersection / union if union > 0 else 0.0

        except Exception as e:
            return 0.0

    def save_redundancy_report(self):
        """Save redundancy analysis to JSON"""
        output_file = OUTPUT_DIR / 'redundancy_clusters.json'

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'generated': datetime.now().isoformat(),
                'exact_duplicates': {
                    'count': len(self.exact_duplicates),
                    'groups': self.exact_duplicates
                },
                'near_duplicates': {
                    'count': len(self.near_duplicates),
                    'pairs': self.near_duplicates
                }
            }, f, indent=2)

        print(f"✓ Saved redundancy report: {output_file}")


def main():
    """Main analysis pipeline"""
    print("=" * 70)
    print("REPOSITORY INTEGRITY & RESEARCH VALIDATION ANALYSIS")
    print("=" * 70)

    # Phase 1: File Inventory
    print("\n📋 PHASE 1: FILE INVENTORY & FINGERPRINTING")
    inventory = FileInventory()
    files = inventory.enumerate_files()
    inventory.save_manifest()

    # Phase 2: Format Validation
    print("\n📋 PHASE 2: FORMAT VALIDATION")
    validator = FormatValidator(files)
    validation_results = validator.validate_all()
    validator.save_validation_results()

    # Phase 3: Duplicate Detection
    print("\n📋 PHASE 3: DUPLICATE DETECTION")
    detector = DuplicateDetector(files)
    detector.find_exact_duplicates()
    detector.find_near_duplicates()
    detector.save_redundancy_report()

    print("\n" + "=" * 70)
    print("✓ Phase 1-3 Complete: Inventory, Validation, Deduplication")
    print("=" * 70)
    print(f"\nProcessed {len(files)} files")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
