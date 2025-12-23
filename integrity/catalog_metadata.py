#!/usr/bin/env python3
"""
Catalog all files with INTEGRITY_METADATA headers
"""
import os
import re
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import hashlib

def get_file_stats(filepath):
    """Get file statistics"""
    stat = os.stat(filepath)
    return {
        'size': stat.st_size,
        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
        'type': Path(filepath).suffix
    }

def extract_metadata(filepath):
    """Extract INTEGRITY_METADATA from a file"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Count total lines
        lines = content.split('\n')
        total_lines = len(lines)

        # Get first 500 and last 500 chars for fingerprint
        first_500 = content[:500]
        last_500 = content[-500:] if len(content) > 500 else content

        # Extract metadata block
        metadata_match = re.search(r'<!-- INTEGRITY_METADATA\s*(.*?)\s*-->', content, re.DOTALL)

        if not metadata_match:
            # Try other formats (Python, etc.)
            metadata_match = re.search(r'# INTEGRITY_METADATA\s*(.*?)(?=\n#[^#]|\Z)', content, re.DOTALL)

        metadata = {}
        if metadata_match:
            meta_text = metadata_match.group(1)

            # Extract Date
            date_match = re.search(r'Date:\s*(.+)', meta_text)
            if date_match:
                metadata['date'] = date_match.group(1).strip()

            # Extract Status
            status_match = re.search(r'Status:\s*(.+)', meta_text)
            if status_match:
                metadata['status'] = status_match.group(1).strip()

            # Extract Severity
            severity_match = re.search(r'Severity:\s*(.+)', meta_text)
            if severity_match:
                metadata['severity'] = severity_match.group(1).strip()

            # Extract Risk Types
            risk_match = re.search(r'Risk Types:\s*(.+)', meta_text)
            if risk_match:
                metadata['risk_types'] = risk_match.group(1).strip()

            # Extract Supporting Evidence
            supporting = []
            for line in meta_text.split('\n'):
                if re.match(r'--\s+-\s+', line):
                    file_ref = re.search(r'--\s+-\s+(.+?)\s*\(', line)
                    if file_ref:
                        supporting.append(file_ref.group(1).strip())
            metadata['supporting_files'] = supporting

            # Extract Referenced By
            referenced_by = []
            in_referenced = False
            for line in meta_text.split('\n'):
                if 'Referenced By:' in line or 'Citing Files:' in line:
                    in_referenced = True
                    continue
                if in_referenced and re.match(r'--\s+-\s+', line):
                    file_ref = re.search(r'--\s+-\s+(.+?)\s*\(', line)
                    if file_ref:
                        referenced_by.append(file_ref.group(1).strip())
            metadata['citing_files'] = referenced_by

        return {
            'metadata': metadata,
            'total_lines': total_lines,
            'first_500': first_500,
            'last_500': last_500,
            'content_hash': hashlib.md5(content.encode()).hexdigest()
        }
    except Exception as e:
        return {
            'error': str(e),
            'metadata': {},
            'total_lines': 0,
            'first_500': '',
            'last_500': '',
            'content_hash': ''
        }

def normalize_status(status):
    """Normalize status to standard values"""
    if 'JUSTIFIED' in status:
        return 'JUSTIFIED'
    elif 'TRULY UNSUPPORTED' in status:
        return 'TRULY_UNSUPPORTED'
    elif 'NEEDS REVIEW' in status:
        return 'NEEDS_REVIEW'
    return status

def find_redundancy_clusters(catalog):
    """Identify potential redundancy clusters"""
    clusters = []

    # Group by similar names
    name_groups = defaultdict(list)
    for entry in catalog:
        basename = Path(entry['path']).stem
        # Extract base pattern (remove version numbers, etc.)
        base_pattern = re.sub(r'(_v\d+|_old|_new|\d+|\.backup)', '', basename)
        base_pattern = re.sub(r'\s+v\d+', '', base_pattern)
        name_groups[base_pattern].append(entry)

    # Find clusters with multiple files
    for pattern, entries in name_groups.items():
        if len(entries) > 1:
            clusters.append({
                'type': 'name_similarity',
                'pattern': pattern,
                'files': entries,
                'count': len(entries)
            })

    # Group by directory
    dir_groups = defaultdict(list)
    for entry in catalog:
        directory = str(Path(entry['path']).parent)
        dir_groups[directory].append(entry)

    # Find directories with many files
    for directory, entries in dir_groups.items():
        if len(entries) >= 5:  # At least 5 files in same directory
            clusters.append({
                'type': 'directory_cluster',
                'directory': directory,
                'files': entries,
                'count': len(entries)
            })

    # Group by content hash (exact duplicates)
    hash_groups = defaultdict(list)
    for entry in catalog:
        if entry.get('content_hash'):
            hash_groups[entry['content_hash']].append(entry)

    for hash_val, entries in hash_groups.items():
        if len(entries) > 1:
            clusters.append({
                'type': 'exact_duplicate',
                'hash': hash_val,
                'files': entries,
                'count': len(entries)
            })

    # Sort clusters by count (largest first)
    clusters.sort(key=lambda x: x['count'], reverse=True)

    return clusters

def main():
    repo_root = Path('/home/user/The-Index')

    # Find all files with INTEGRITY_METADATA
    print("Searching for files with INTEGRITY_METADATA...")
    result = os.popen('cd /home/user/The-Index && grep -r -l "INTEGRITY_METADATA" --include="*.md" --include="*.html" --include="*.py" --include="*.js" --include="*.txt" --include="*.yaml" --include="*.json" --include="*.tex" --include="*.ts" --include="*.sh" 2>/dev/null').read()

    files = [line.strip() for line in result.split('\n') if line.strip()]

    print(f"Found {len(files)} files with INTEGRITY_METADATA")

    catalog = []

    for i, filepath in enumerate(files):
        if not filepath.startswith('/'):
            filepath = str(repo_root / filepath)

        if i % 50 == 0:
            print(f"Processing {i+1}/{len(files)}...")

        # Get file stats
        try:
            stats = get_file_stats(filepath)
        except Exception as e:
            print(f"Error getting stats for {filepath}: {e}")
            continue

        # Extract metadata
        extracted = extract_metadata(filepath)

        entry = {
            'path': filepath,
            'size': stats['size'],
            'modified': stats['modified'],
            'file_type': stats['type'],
            'total_lines': extracted['total_lines'],
            'content_hash': extracted['content_hash'],
            'first_500_chars': extracted['first_500'],
            'last_500_chars': extracted['last_500'],
        }

        # Add metadata fields
        meta = extracted['metadata']
        entry['date'] = meta.get('date', '')
        entry['status'] = normalize_status(meta.get('status', ''))
        entry['severity'] = meta.get('severity', '')
        entry['risk_types'] = meta.get('risk_types', '')
        entry['supporting_files'] = meta.get('supporting_files', [])
        entry['citing_files'] = meta.get('citing_files', [])

        catalog.append(entry)

    print(f"\nProcessed {len(catalog)} files")

    # Identify redundancy clusters
    print("\nIdentifying redundancy clusters...")
    clusters = find_redundancy_clusters(catalog)

    # Build final report
    report = {
        'generated': datetime.now().isoformat(),
        'total_files': len(catalog),
        'files_by_status': {},
        'files_by_severity': {},
        'files_by_type': {},
        'redundancy_clusters': clusters,
        'all_files': catalog
    }

    # Count by status
    for entry in catalog:
        status = entry['status']
        report['files_by_status'][status] = report['files_by_status'].get(status, 0) + 1

    # Count by severity
    for entry in catalog:
        severity = entry['severity']
        report['files_by_severity'][severity] = report['files_by_severity'].get(severity, 0) + 1

    # Count by file type
    for entry in catalog:
        ftype = entry['file_type']
        report['files_by_type'][ftype] = report['files_by_type'].get(ftype, 0) + 1

    # Save report
    output_file = repo_root / 'integrity' / 'COMPLETE_METADATA_CATALOG.json'
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nReport saved to: {output_file}")
    print(f"\nSummary:")
    print(f"  Total files: {report['total_files']}")
    print(f"  By status: {report['files_by_status']}")
    print(f"  By severity: {report['files_by_severity']}")
    print(f"  By type: {report['files_by_type']}")
    print(f"  Redundancy clusters found: {len(clusters)}")
    print(f"\nTop 10 largest clusters:")
    for i, cluster in enumerate(clusters[:10], 1):
        print(f"    {i}. {cluster['type']}: {cluster['count']} files")

if __name__ == '__main__':
    main()
