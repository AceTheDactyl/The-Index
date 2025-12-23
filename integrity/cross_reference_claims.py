#!/usr/bin/env python3
"""
Cross-Reference Claims Analysis
Identifies which claims are supported by other files, finds duplicates,
and marks files with unsupported/outdated claims
"""

import os
import json
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from difflib import SequenceMatcher

# Configuration
REPO_ROOT = Path("/home/user/The-Index")
OUTPUT_DIR = REPO_ROOT / "integrity"

# Similarity threshold for duplicate claims
CLAIM_SIMILARITY_THRESHOLD = 0.85


class ClaimCrossReferencer:
    """Cross-references claims across files to identify support and duplicates"""

    def __init__(self):
        self.risk_flags = self._load_json('risk_flags.json')
        self.claim_data = self._load_json('claim_citation_analysis.json')
        self.manifest = self._load_json('file_manifest.json')
        self.dependencies = self._load_json('consistency_analysis.json')

        # Build lookup maps
        self.file_by_id = {f['canonical_id']: f for f in self.manifest['files']}
        self.claims_by_id = {f['canonical_id']: f for f in self.claim_data['files']}

        # Analysis results
        self.claim_support_map = {}
        self.duplicate_claims = []
        self.validated_files = []
        self.file_content_cache = {}

    def _load_json(self, filename):
        """Load JSON file"""
        with open(OUTPUT_DIR / filename, 'r') as f:
            return json.load(f)

    def analyze_claim_support(self):
        """Analyze which high-risk files have claims supported by other files"""
        print("\n🔍 Cross-referencing claims across repository...")

        high_risk_files = [f for f in self.risk_flags['flagged_files'] if f['max_severity'] == 'HIGH']

        print(f"Analyzing {len(high_risk_files)} high-risk files...")

        for idx, risk_file in enumerate(high_risk_files):
            if (idx + 1) % 50 == 0:
                print(f"  Progress: {idx + 1}/{len(high_risk_files)}")

            file_id = risk_file['canonical_id']
            file_path = risk_file['path']

            # Get file content
            content = self._get_file_content(file_id)
            if not content:
                continue

            # Extract key claims from this file
            file_claims = self._extract_key_claims(content)

            # Find supporting files
            supporting_files = self._find_supporting_files(file_id, file_claims)

            # Find citing files (files that reference this one)
            citing_files = self._find_citing_files(file_id)

            # Analyze risk types
            risk_types = [r['type'] for r in risk_file['risks']]
            has_unsupported_claims = 'unsupported_claims' in risk_types
            has_unverified_math = 'unverified_math' in risk_types

            # Determine validation status
            is_justified = len(supporting_files) > 0 or len(citing_files) > 0

            validation = {
                'canonical_id': file_id,
                'path': file_path,
                'risk_severity': risk_file['max_severity'],
                'risk_types': risk_types,
                'has_unsupported_claims': has_unsupported_claims,
                'has_unverified_math': has_unverified_math,
                'claim_count': len(file_claims),
                'supporting_files': supporting_files,
                'citing_files': citing_files,
                'is_justified': is_justified,
                'justification_strength': len(supporting_files) + len(citing_files),
                'needs_citation_update': has_unsupported_claims and is_justified,
                'truly_unsupported': has_unsupported_claims and not is_justified
            }

            self.validated_files.append(validation)

        justified_count = sum(1 for v in self.validated_files if v['is_justified'])
        truly_unsupported = sum(1 for v in self.validated_files if v['truly_unsupported'])

        print(f"✓ Analysis complete:")
        print(f"  - {justified_count} files have supporting evidence in repository")
        print(f"  - {truly_unsupported} files are truly unsupported")
        print(f"  - {len(self.validated_files) - justified_count - truly_unsupported} files need review")

        return self.validated_files

    def _get_file_content(self, file_id):
        """Get file content with caching"""
        if file_id in self.file_content_cache:
            return self.file_content_cache[file_id]

        file_info = self.file_by_id.get(file_id)
        if not file_info or not file_info['is_text']:
            return None

        filepath = REPO_ROOT / file_info['relative_path']
        try:
            with open(filepath, 'r', encoding=file_info['encoding'] or 'utf-8', errors='ignore') as f:
                content = f.read()
            self.file_content_cache[file_id] = content
            return content
        except Exception as e:
            return None

    def _extract_key_claims(self, content):
        """Extract key declarative claims from content"""
        # Split into sentences
        sentences = re.split(r'[.!?]+\n', content)

        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            # Look for substantial declarative statements
            if len(sentence) > 40 and len(sentence) < 500:
                # Check for claim indicators
                if re.search(r'\b(is|are|demonstrates|shows|proves|indicates|theorem|lemma|proposition)\b',
                            sentence, re.IGNORECASE):
                    claims.append(sentence)

        return claims[:50]  # Limit to top 50 claims per file

    def _find_supporting_files(self, file_id, claims):
        """Find files that contain similar claims or evidence"""
        supporting = []

        # Check dependency graph
        file_deps = [d for d in self.dependencies['dependencies']['items']
                    if d['target'] == file_id]

        for dep in file_deps:
            source_id = dep['source']
            source_content = self._get_file_content(source_id)

            if source_content:
                # Check if source file contains similar claims
                for claim in claims[:10]:  # Check first 10 claims
                    if self._claim_similarity(claim, source_content) > 0.3:
                        supporting.append({
                            'id': source_id,
                            'path': self.file_by_id[source_id]['relative_path'],
                            'type': 'dependency'
                        })
                        break

        return supporting

    def _find_citing_files(self, file_id):
        """Find files that reference this file"""
        citing = []

        file_path = self.file_by_id[file_id]['relative_path']
        filename = Path(file_path).name

        # Check dependency graph
        file_refs = [d for d in self.dependencies['dependencies']['items']
                    if d['target'] == file_id]

        for ref in file_refs[:10]:  # Limit to 10
            citing.append({
                'id': ref['source'],
                'path': ref['source_path'],
                'type': 'reference'
            })

        return citing

    def _claim_similarity(self, claim, content):
        """Calculate similarity between claim and content"""
        # Simple word overlap
        claim_words = set(re.findall(r'\b\w+\b', claim.lower()))
        content_words = set(re.findall(r'\b\w+\b', content.lower()))

        if not claim_words:
            return 0.0

        overlap = len(claim_words & content_words)
        return overlap / len(claim_words)

    def find_duplicate_claims(self):
        """Find duplicate claims across files using efficient hash-based approach"""
        print("\n🔍 Identifying duplicate claims (hash-based)...")

        # Use hash-based deduplication for efficiency
        claim_hash_map = defaultdict(list)

        # Process high-risk files only for efficiency
        high_risk_ids = [v['canonical_id'] for v in self.validated_files]

        print(f"Analyzing claims from {len(high_risk_ids)} high-risk files...")

        for file_id in high_risk_ids:
            file_info = self.file_by_id.get(file_id)
            if not file_info or not file_info['is_text']:
                continue

            content = self._get_file_content(file_id)
            if content:
                claims = self._extract_key_claims(content)
                for claim in claims:
                    # Normalize and hash the claim
                    normalized = ' '.join(claim.lower().split())
                    claim_hash = hash(normalized)

                    claim_hash_map[claim_hash].append({
                        'claim': claim[:200],  # First 200 chars
                        'file_id': file_id,
                        'file_path': file_info['relative_path']
                    })

        # Find duplicates (hash collisions indicate same/very similar claims)
        duplicate_groups = []
        for claim_hash, occurrences in claim_hash_map.items():
            if len(occurrences) > 1:
                # Check if they're from different files
                unique_files = set(occ['file_id'] for occ in occurrences)
                if len(unique_files) > 1:
                    duplicate_groups.append({
                        'claim_text': occurrences[0]['claim'],
                        'occurrence_count': len(occurrences),
                        'unique_files': len(unique_files),
                        'files': [{'id': o['file_id'], 'path': o['file_path']} for o in occurrences]
                    })

        # Sort by occurrence count
        duplicate_groups.sort(key=lambda x: x['occurrence_count'], reverse=True)

        print(f"✓ Found {len(duplicate_groups)} duplicate claim groups")
        if duplicate_groups:
            print(f"  Top duplicate appears in {duplicate_groups[0]['occurrence_count']} instances across {duplicate_groups[0]['unique_files']} files")

        self.duplicate_claims = duplicate_groups
        return duplicate_groups

    def _text_similarity(self, text1, text2):
        """Calculate text similarity using SequenceMatcher"""
        return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

    def save_reports(self):
        """Save analysis reports"""
        print("\n💾 Saving reports...")

        # Claim validation report
        validation_file = OUTPUT_DIR / 'claim_validation_report.json'
        with open(validation_file, 'w', encoding='utf-8') as f:
            json.dump({
                'generated': datetime.now().isoformat(),
                'total_high_risk_files': len(self.validated_files),
                'justified_files': sum(1 for v in self.validated_files if v['is_justified']),
                'truly_unsupported': sum(1 for v in self.validated_files if v['truly_unsupported']),
                'needs_citation_update': sum(1 for v in self.validated_files if v['needs_citation_update']),
                'files': self.validated_files
            }, f, indent=2)

        print(f"✓ Saved claim validation: {validation_file}")

        # Duplicate claims report
        duplicates_file = OUTPUT_DIR / 'duplicate_claims_report.json'
        with open(duplicates_file, 'w', encoding='utf-8') as f:
            json.dump({
                'generated': datetime.now().isoformat(),
                'total_duplicate_groups': len(self.duplicate_claims),
                'duplicate_groups': self.duplicate_claims[:500]  # Top 500 groups
            }, f, indent=2)

        print(f"✓ Saved duplicate claims: {duplicates_file}")

        # Create editing instructions for the agent
        files_to_edit = []
        for validation in self.validated_files:
            edit_instruction = {
                'canonical_id': validation['canonical_id'],
                'path': validation['path'],
                'is_justified': validation['is_justified'],
                'truly_unsupported': validation['truly_unsupported'],
                'needs_citation_update': validation['needs_citation_update'],
                'supporting_files': validation['supporting_files'],
                'citing_files': validation['citing_files'],
                'risk_types': validation['risk_types']
            }
            files_to_edit.append(edit_instruction)

        edit_file = OUTPUT_DIR / 'files_to_edit.json'
        with open(edit_file, 'w', encoding='utf-8') as f:
            json.dump({
                'generated': datetime.now().isoformat(),
                'total_files': len(files_to_edit),
                'files': files_to_edit
            }, f, indent=2)

        print(f"✓ Saved editing instructions: {edit_file}")


def main():
    """Main analysis pipeline"""
    print("=" * 70)
    print("CLAIM CROSS-REFERENCE ANALYSIS")
    print("=" * 70)

    analyzer = ClaimCrossReferencer()

    # Analyze claim support
    analyzer.analyze_claim_support()

    # Find duplicate claims
    analyzer.find_duplicate_claims()

    # Save reports
    analyzer.save_reports()

    print("\n" + "=" * 70)
    print("✓ ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
