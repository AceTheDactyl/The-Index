#!/usr/bin/env python3
"""
Analyze redundancy metadata from integrity system to identify files for deletion.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict

def load_json(filepath):
    """Load JSON file with error handling."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def get_file_info(filepath):
    """Get file metadata if file exists."""
    try:
        stat = os.stat(filepath)
        return {
            'size': stat.st_size,
            'mtime': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'exists': True
        }
    except:
        return {
            'size': 0,
            'mtime': None,
            'exists': False
        }

def analyze_context(filepath):
    """Analyze file context to determine if it should be kept."""
    path_parts = Path(filepath).parts

    # Categorize by directory context
    contexts = {
        'active_project': False,
        'documentation': False,
        'examples': False,
        'tests': False,
        'archived': False,
        'temp': False
    }

    path_lower = filepath.lower()

    # Check for active project indicators
    if any(x in path_parts for x in ['src', 'lib', 'app', 'core']):
        contexts['active_project'] = True

    # Check for documentation
    if any(x in path_lower for x in ['doc', 'readme', 'wiki', 'guide']):
        contexts['documentation'] = True

    # Check for examples
    if any(x in path_lower for x in ['example', 'sample', 'demo', 'template']):
        contexts['examples'] = True

    # Check for tests
    if any(x in path_lower for x in ['test', 'spec', '__tests__']):
        contexts['tests'] = True

    # Check for archived/old content
    if any(x in path_lower for x in ['old', 'archive', 'backup', 'deprecated']):
        contexts['archived'] = True

    # Check for temp files
    if any(x in path_lower for x in ['temp', 'tmp', 'cache']):
        contexts['temp'] = True

    return contexts

def main():
    base_dir = Path('/home/user/The-Index')
    integrity_dir = base_dir / 'integrity'

    print("=" * 80)
    print("REDUNDANCY ANALYSIS REPORT")
    print("=" * 80)
    print()

    # Load data files
    print("Loading metadata files...")
    redundancy_clusters = load_json(integrity_dir / 'redundancy_clusters.json')
    duplicate_claims = load_json(integrity_dir / 'duplicate_claims_report.json')
    canonical_mapping = load_json(integrity_dir / 'canonical_mapping.json')

    if not redundancy_clusters or not duplicate_claims or not canonical_mapping:
        print("Error: Failed to load one or more metadata files")
        return

    exact_dup_groups = redundancy_clusters.get('exact_duplicates', {}).get('groups', [])
    duplicate_groups = duplicate_claims.get('duplicate_groups', [])

    print(f"Loaded {len(exact_dup_groups)} exact duplicate groups")
    print(f"Loaded {len(duplicate_groups)} duplicate claim groups")
    print(f"Loaded {len(canonical_mapping)} canonical mappings")
    print()

    # Analyze exact duplicates (redundancy clusters)
    print("=" * 80)
    print("EXACT DUPLICATE FILES ANALYSIS")
    print("=" * 80)
    print()

    deletion_candidates = []
    keep_despite_duplicate = []
    canonical_groups = defaultdict(list)

    for cluster in exact_dup_groups:
        canonical_info = cluster.get('canonical', {})
        canonical = canonical_info.get('path', '')
        canonical_size = canonical_info.get('size', 0)
        canonical_id = canonical_info.get('id', '')

        duplicates_list = cluster.get('duplicates', [])
        if not duplicates_list:
            continue

        # Get file info for canonical file
        canonical_file_info = get_file_info(canonical)

        for dup_entry in duplicates_list:
            dup_path = dup_entry.get('path', '')
            dup_size = dup_entry.get('size', 0)
            dup_id = dup_entry.get('id', '')

            dup_info = get_file_info(dup_path)
            dup_context = analyze_context(dup_path)

            entry = {
                'duplicate_file': dup_path,
                'duplicate_id': dup_id,
                'canonical_file': canonical,
                'canonical_id': canonical_id,
                'size': dup_info['size'],
                'reported_size': dup_size,
                'mtime': dup_info['mtime'],
                'exists': dup_info['exists'],
                'contexts': dup_context,
                'cluster_size': len(duplicates_list) + 1,  # +1 for canonical
                'sha256': cluster.get('sha256', '')
            }

            # Determine if this should be kept despite being duplicate
            should_keep = False
            reason = []

            # Keep if in different active contexts
            canonical_context = analyze_context(canonical)
            if (dup_context['active_project'] and canonical_context['active_project'] and
                Path(dup_path).parts[0:2] != Path(canonical).parts[0:2]):
                should_keep = True
                reason.append("Different active project contexts")

            # Keep documentation if in different doc contexts
            if (dup_context['documentation'] and canonical_context['documentation'] and
                Path(dup_path).parent != Path(canonical).parent):
                should_keep = True
                reason.append("Different documentation locations")

            if should_keep:
                entry['keep_reason'] = '; '.join(reason)
                keep_despite_duplicate.append(entry)
            else:
                deletion_candidates.append(entry)
                canonical_groups[canonical].append(entry)

    # Analyze duplicate claims
    print("=" * 80)
    print("DUPLICATE CLAIMS ANALYSIS")
    print("=" * 80)
    print()

    claim_duplicates = []
    for group in duplicate_claims.get('duplicate_groups', []):
        files = group.get('files', [])
        claim_text = group.get('claim_text', '')[:100]  # First 100 chars

        if len(files) < 2:
            continue

        canonical = files[0]
        duplicates = files[1:]

        for dup_path in duplicates:
            dup_info = get_file_info(dup_path)
            claim_duplicates.append({
                'duplicate_file': dup_path,
                'canonical_file': canonical,
                'claim_preview': claim_text,
                'size': dup_info['size'],
                'exists': dup_info['exists']
            })

    # Generate comprehensive report
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print()

    total_deletable = len(deletion_candidates)
    total_space_savings = sum(d['size'] for d in deletion_candidates if d['exists'])

    print(f"Total exact duplicate files (deletable): {total_deletable}")
    print(f"Total files to keep despite duplication: {len(keep_despite_duplicate)}")
    print(f"Total duplicate claim instances: {len(claim_duplicates)}")
    print(f"Potential space savings: {total_space_savings:,} bytes ({total_space_savings / (1024*1024):.2f} MB)")
    print()

    # Sort by size for deletion order
    deletion_candidates.sort(key=lambda x: x['size'], reverse=True)

    print("=" * 80)
    print("TOP 50 LARGEST DUPLICATES (RECOMMENDED DELETION ORDER)")
    print("=" * 80)
    print()

    for i, entry in enumerate(deletion_candidates[:50], 1):
        print(f"{i}. {entry['duplicate_file']}")
        print(f"   Canonical: {entry['canonical_file']}")
        print(f"   Size: {entry['size']:,} bytes ({entry['size'] / 1024:.2f} KB)")
        print(f"   Modified: {entry['mtime']}")
        print(f"   Cluster size: {entry['cluster_size']} files")
        print()

    # Save detailed reports
    print("=" * 80)
    print("SAVING DETAILED REPORTS")
    print("=" * 80)
    print()

    output_dir = integrity_dir / 'deletion_analysis'
    output_dir.mkdir(exist_ok=True)

    # Save deletion candidates
    with open(output_dir / 'deletion_candidates.json', 'w', encoding='utf-8') as f:
        json.dump({
            'total_candidates': len(deletion_candidates),
            'total_space_savings_bytes': total_space_savings,
            'total_space_savings_mb': total_space_savings / (1024*1024),
            'candidates': deletion_candidates
        }, f, indent=2)
    print(f"Saved: {output_dir / 'deletion_candidates.json'}")

    # Save keep-despite-duplicate list
    with open(output_dir / 'keep_despite_duplicate.json', 'w', encoding='utf-8') as f:
        json.dump({
            'total_files': len(keep_despite_duplicate),
            'files': keep_despite_duplicate
        }, f, indent=2)
    print(f"Saved: {output_dir / 'keep_despite_duplicate.json'}")

    # Save canonical groups
    with open(output_dir / 'canonical_groups.json', 'w', encoding='utf-8') as f:
        canonical_groups_list = [
            {
                'canonical_file': canonical,
                'duplicate_count': len(duplicates),
                'total_duplicate_size': sum(d['size'] for d in duplicates),
                'duplicates': duplicates
            }
            for canonical, duplicates in canonical_groups.items()
        ]
        canonical_groups_list.sort(key=lambda x: x['total_duplicate_size'], reverse=True)
        json.dump({
            'total_canonical_files': len(canonical_groups_list),
            'groups': canonical_groups_list
        }, f, indent=2)
    print(f"Saved: {output_dir / 'canonical_groups.json'}")

    # Save duplicate claims
    with open(output_dir / 'duplicate_claims.json', 'w', encoding='utf-8') as f:
        json.dump({
            'total_duplicate_claims': len(claim_duplicates),
            'duplicates': claim_duplicates
        }, f, indent=2)
    print(f"Saved: {output_dir / 'duplicate_claims.json'}")

    # Generate deletion script
    print()
    print("=" * 80)
    print("GENERATING DELETION SCRIPT")
    print("=" * 80)
    print()

    script_path = output_dir / 'delete_duplicates.sh'
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write("#!/bin/bash\n")
        f.write("# Auto-generated script to delete duplicate files\n")
        f.write("# Review carefully before running!\n")
        f.write("#\n")
        f.write(f"# Total files to delete: {len(deletion_candidates)}\n")
        f.write(f"# Total space to recover: {total_space_savings:,} bytes ({total_space_savings / (1024*1024):.2f} MB)\n")
        f.write("#\n\n")
        f.write("set -e\n\n")
        f.write("echo 'Starting deletion of duplicate files...'\n")
        f.write(f"echo 'Total files to delete: {len(deletion_candidates)}'\n\n")

        for i, entry in enumerate(deletion_candidates, 1):
            if entry['exists']:
                f.write(f"# {i}/{len(deletion_candidates)}: {entry['size']:,} bytes\n")
                f.write(f"# Canonical: {entry['canonical_file']}\n")
                f.write(f"rm -f '{entry['duplicate_file']}'\n\n")

        f.write("echo 'Deletion complete!'\n")

    os.chmod(script_path, 0o755)
    print(f"Saved: {script_path}")
    print()

    # Print analysis by directory
    print("=" * 80)
    print("ANALYSIS BY DIRECTORY")
    print("=" * 80)
    print()

    dir_stats = defaultdict(lambda: {'count': 0, 'size': 0})
    for entry in deletion_candidates:
        parent_dir = str(Path(entry['duplicate_file']).parent)
        dir_stats[parent_dir]['count'] += 1
        dir_stats[parent_dir]['size'] += entry['size']

    dir_stats_list = [
        {
            'directory': dir_path,
            'duplicate_count': stats['count'],
            'total_size': stats['size']
        }
        for dir_path, stats in dir_stats.items()
    ]
    dir_stats_list.sort(key=lambda x: x['total_size'], reverse=True)

    print("Top 20 directories with most duplicate data:")
    print()
    for i, stat in enumerate(dir_stats_list[:20], 1):
        print(f"{i}. {stat['directory']}")
        print(f"   Files: {stat['duplicate_count']}, Size: {stat['total_size']:,} bytes ({stat['total_size'] / 1024:.2f} KB)")
        print()

    with open(output_dir / 'directory_stats.json', 'w', encoding='utf-8') as f:
        json.dump(dir_stats_list, f, indent=2)
    print(f"Saved: {output_dir / 'directory_stats.json'}")

    print()
    print("=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print()
    print(f"All reports saved to: {output_dir}")
    print()

if __name__ == '__main__':
    main()
