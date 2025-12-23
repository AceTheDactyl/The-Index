#!/usr/bin/env python3
"""
Analyze redundancy clusters in detail
"""
import json
from pathlib import Path
from collections import defaultdict

def analyze_clusters():
    with open('/home/user/The-Index/integrity/COMPLETE_METADATA_CATALOG.json', 'r') as f:
        data = json.load(f)

    print("=" * 80)
    print("INTEGRITY METADATA CATALOG ANALYSIS")
    print("=" * 80)
    print(f"\nGenerated: {data['generated']}")
    print(f"Total files with metadata: {data['total_files']}")

    print("\n" + "=" * 80)
    print("STATUS BREAKDOWN")
    print("=" * 80)
    for status, count in sorted(data['files_by_status'].items(), key=lambda x: -x[1]):
        status_label = status if status else "NO STATUS (missing metadata)"
        print(f"  {status_label:40} {count:4} files")

    print("\n" + "=" * 80)
    print("SEVERITY BREAKDOWN")
    print("=" * 80)
    for severity, count in sorted(data['files_by_severity'].items(), key=lambda x: -x[1]):
        severity_label = severity if severity else "NO SEVERITY (missing metadata)"
        print(f"  {severity_label:40} {count:4} files")

    print("\n" + "=" * 80)
    print("FILE TYPE BREAKDOWN")
    print("=" * 80)
    for ftype, count in sorted(data['files_by_type'].items(), key=lambda x: -x[1]):
        print(f"  {ftype:40} {count:4} files")

    print("\n" + "=" * 80)
    print("REDUNDANCY CLUSTERS ANALYSIS")
    print("=" * 80)
    print(f"Total clusters identified: {len(data['redundancy_clusters'])}")

    # Analyze by cluster type
    cluster_types = defaultdict(list)
    for cluster in data['redundancy_clusters']:
        cluster_types[cluster['type']].append(cluster)

    print("\nBy cluster type:")
    for ctype, clusters in sorted(cluster_types.items()):
        print(f"  {ctype:30} {len(clusters):3} clusters")

    print("\n" + "=" * 80)
    print("TOP 20 LARGEST REDUNDANCY CLUSTERS")
    print("=" * 80)

    for i, cluster in enumerate(data['redundancy_clusters'][:20], 1):
        print(f"\n{i}. {cluster['type'].upper()}: {cluster['count']} files")

        if cluster['type'] == 'directory_cluster':
            print(f"   Directory: {cluster['directory']}")
            # Show status breakdown for this directory
            status_counts = defaultdict(int)
            severity_counts = defaultdict(int)
            type_counts = defaultdict(int)
            total_size = 0

            for f in cluster['files']:
                status_counts[f['status'] or 'NO_STATUS'] += 1
                severity_counts[f['severity'] or 'NO_SEVERITY'] += 1
                type_counts[f['file_type']] += 1
                total_size += f['size']

            print(f"   Total size: {total_size:,} bytes ({total_size/1024/1024:.2f} MB)")
            print(f"   Status: {dict(status_counts)}")
            print(f"   Severity: {dict(severity_counts)}")
            print(f"   Types: {dict(type_counts)}")

        elif cluster['type'] == 'name_similarity':
            print(f"   Pattern: {cluster['pattern']}")
            print(f"   Files:")
            for f in cluster['files'][:10]:  # Show first 10
                fname = Path(f['path']).name
                print(f"     - {fname:50} ({f['size']:,} bytes, {f['status'] or 'NO_STATUS'})")
            if len(cluster['files']) > 10:
                print(f"     ... and {len(cluster['files']) - 10} more")

        elif cluster['type'] == 'exact_duplicate':
            print(f"   Hash: {cluster['hash'][:16]}...")
            print(f"   Duplicate files:")
            for f in cluster['files']:
                print(f"     - {f['path']}")

    # Identify specific patterns
    print("\n" + "=" * 80)
    print("SPECIFIC REDUNDANCY PATTERNS")
    print("=" * 80)

    # Find DOMAIN_ files
    domain_files = [f for f in data['all_files'] if 'DOMAIN_' in Path(f['path']).name]
    print(f"\nDOMAIN_ files: {len(domain_files)}")
    domain_groups = defaultdict(list)
    for f in domain_files:
        name = Path(f['path']).name
        domain_groups[name].append(f['path'])

    for name, paths in sorted(domain_groups.items()):
        if len(paths) > 1:
            print(f"  {name}: {len(paths)} copies")
            for p in paths:
                print(f"    - {p}")

    # Find GRAND_SYNTHESIS files
    synthesis_files = [f for f in data['all_files'] if 'GRAND_SYNTHESIS' in Path(f['path']).name or 'SYNTHESIS' in Path(f['path']).name]
    print(f"\nGRAND_SYNTHESIS/SYNTHESIS files: {len(synthesis_files)}")
    synth_groups = defaultdict(list)
    for f in synthesis_files:
        name = Path(f['path']).name
        synth_groups[name].append(f['path'])

    for name, paths in sorted(synth_groups.items()):
        if len(paths) > 1:
            print(f"  {name}: {len(paths)} copies")
            for p in paths:
                print(f"    - {p}")

    # Find version indicators
    versioned_files = []
    for f in data['all_files']:
        name = Path(f['path']).name
        if any(pattern in name for pattern in ['_v2', '_v3', ' v2', ' v3', '_old', '_new', 'backup']):
            versioned_files.append(f)

    print(f"\nFiles with version indicators: {len(versioned_files)}")
    for f in versioned_files[:20]:  # Show first 20
        print(f"  - {Path(f['path']).name}")
        print(f"    Path: {f['path']}")
        print(f"    Status: {f['status'] or 'NO_STATUS'}, Size: {f['size']:,} bytes")

    # Find Research directory files
    research_files = [f for f in data['all_files'] if '/Research/' in f['path']]
    print(f"\nFiles in Research directories: {len(research_files)}")
    research_dirs = defaultdict(list)
    for f in research_files:
        dir_path = str(Path(f['path']).parent)
        research_dirs[dir_path].append(f)

    print(f"Distinct Research directories: {len(research_dirs)}")
    for dir_path, files in sorted(research_dirs.items(), key=lambda x: -len(x[1]))[:5]:
        print(f"  {dir_path}: {len(files)} files")

    print("\n" + "=" * 80)
    print("PRIORITY RECOMMENDATIONS")
    print("=" * 80)
    print("\nBased on this analysis, the following areas have highest redundancy risk:")
    print("\n1. DIRECTORY CLUSTERS - Large directories with many files")
    for cluster in data['redundancy_clusters'][:5]:
        if cluster['type'] == 'directory_cluster':
            print(f"   - {cluster['directory']}: {cluster['count']} files")

    print("\n2. NAME SIMILARITY - Files with similar names (likely duplicates)")
    for cluster in data['redundancy_clusters']:
        if cluster['type'] == 'name_similarity' and cluster['count'] >= 5:
            print(f"   - {cluster['pattern']}: {cluster['count']} files")

    print("\n3. VERSION INDICATORS - Files with version numbers")
    print(f"   - {len(versioned_files)} files have version indicators")

    print("\n4. EXACT DUPLICATES - Files with identical content")
    exact_dup_count = sum(1 for c in data['redundancy_clusters'] if c['type'] == 'exact_duplicate')
    if exact_dup_count > 0:
        print(f"   - {exact_dup_count} sets of exact duplicates")

if __name__ == '__main__':
    analyze_clusters()
