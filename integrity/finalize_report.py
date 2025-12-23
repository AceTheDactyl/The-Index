#!/usr/bin/env python3
"""
Final Analysis Phase: Provenance, Risk Flagging, and Reporting
Completes the integrity analysis with risk assessment and comprehensive reports
"""

import os
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

# Configuration
REPO_ROOT = Path("/home/user/The-Index")
OUTPUT_DIR = REPO_ROOT / "integrity"


class ProvenanceAnalyzer:
    """Analyzes file provenance and lineage"""

    def __init__(self):
        self.manifest = self._load_manifest()
        self.redundancy = self._load_redundancy()
        self.provenance_data = []

    def _load_manifest(self):
        """Load file manifest"""
        with open(OUTPUT_DIR / 'file_manifest.json', 'r') as f:
            return json.load(f)

    def _load_redundancy(self):
        """Load redundancy analysis"""
        with open(OUTPUT_DIR / 'redundancy_clusters.json', 'r') as f:
            return json.load(f)

    def analyze_provenance(self):
        """Analyze provenance and lineage for file clusters"""
        print("\n📋 PHASE 7: PROVENANCE & LINEAGE ANALYSIS")
        print("🔍 Analyzing file lineage...")

        # Analyze duplicate groups for provenance
        for group in self.redundancy['exact_duplicates']['groups']:
            canonical = group['canonical']
            duplicates = group['duplicates']

            # Identify lineage
            all_files = [canonical] + duplicates
            paths = [canonical['path']] + [d['path'] for d in duplicates]

            # Determine if this is a versioned series
            is_versioned = self._detect_versioning(paths)

            # Get file metadata
            file_metadata = []
            for file_info in self.manifest['files']:
                if file_info['relative_path'] in paths:
                    file_metadata.append({
                        'id': file_info['canonical_id'],
                        'path': file_info['relative_path'],
                        'modified': file_info['last_modified']
                    })

            # Sort by modification time
            file_metadata.sort(key=lambda x: x['modified'])

            lineage = {
                'cluster_hash': group['sha256'][:16],
                'is_versioned': is_versioned,
                'earliest_version': file_metadata[0] if file_metadata else None,
                'latest_version': file_metadata[-1] if file_metadata else None,
                'all_versions': file_metadata,
                'version_count': len(file_metadata),
                'direction': 'temporal' if file_metadata else 'unknown'
            }

            self.provenance_data.append(lineage)

        # Analyze near-duplicates
        for pair in self.redundancy['near_duplicates']['pairs']:
            if pair['category'] == 'derivative':
                # This is a potential fork/derivation
                file1_meta = next((f for f in self.manifest['files'] if f['canonical_id'] == pair['file_1']['id']), None)
                file2_meta = next((f for f in self.manifest['files'] if f['canonical_id'] == pair['file_2']['id']), None)

                if file1_meta and file2_meta:
                    # Determine which is earlier
                    if file1_meta['last_modified'] < file2_meta['last_modified']:
                        source = file1_meta
                        derived = file2_meta
                    else:
                        source = file2_meta
                        derived = file1_meta

                    self.provenance_data.append({
                        'type': 'derivation',
                        'similarity': pair['similarity'],
                        'source': {
                            'id': source['canonical_id'],
                            'path': source['relative_path'],
                            'modified': source['last_modified']
                        },
                        'derived': {
                            'id': derived['canonical_id'],
                            'path': derived['relative_path'],
                            'modified': derived['last_modified']
                        }
                    })

        print(f"✓ Analyzed {len(self.provenance_data)} lineage chains")
        return self.provenance_data

    def _detect_versioning(self, paths):
        """Detect if paths follow a versioning pattern"""
        # Look for common versioning patterns: v1, v2, _v1, -v2, etc.
        version_pattern = r'[_-]?v\d+'
        import re

        versioned_count = sum(1 for p in paths if re.search(version_pattern, p, re.IGNORECASE))
        return versioned_count >= len(paths) * 0.5

    def save_provenance_analysis(self):
        """Save provenance analysis"""
        output_file = OUTPUT_DIR / 'provenance_analysis.json'

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'generated': datetime.now().isoformat(),
                'total_lineages': len(self.provenance_data),
                'lineages': self.provenance_data
            }, f, indent=2)

        print(f"✓ Saved provenance analysis: {output_file}")


class RiskFlagger:
    """Flags files with potential integrity risks"""

    def __init__(self):
        self.manifest = self._load_manifest()
        self.scores = self._load_scores()
        self.claim_data = self._load_claim_data()
        self.validation = self._load_validation()
        self.risk_flags = []

    def _load_manifest(self):
        """Load file manifest"""
        with open(OUTPUT_DIR / 'file_manifest.json', 'r') as f:
            return json.load(f)

    def _load_scores(self):
        """Load epistemic scores"""
        with open(OUTPUT_DIR / 'epistemic_scores.json', 'r') as f:
            data = json.load(f)
        return {item['canonical_id']: item for item in data['scores']}

    def _load_claim_data(self):
        """Load claim analysis"""
        with open(OUTPUT_DIR / 'claim_citation_analysis.json', 'r') as f:
            data = json.load(f)
        return {item['canonical_id']: item for item in data['files']}

    def _load_validation(self):
        """Load validation results"""
        with open(OUTPUT_DIR / 'format_validation.json', 'r') as f:
            data = json.load(f)
        return {item['canonical_id']: item for item in data['results']}

    def flag_risks(self):
        """Flag files with integrity risks"""
        print("\n📋 PHASE 8: RISK & CONTAMINATION FLAGGING")
        print("🔍 Identifying integrity risks...")

        for file_info in self.manifest['files']:
            file_id = file_info['canonical_id']
            risks = []

            # Get associated data
            score_data = self.scores.get(file_id)
            claim_data = self.claim_data.get(file_id)
            validation = self.validation.get(file_id)

            # Flag 1: Corrupted files
            if validation and validation['status'] == 'corrupt':
                risks.append({
                    'type': 'corruption',
                    'severity': 'HIGH',
                    'description': 'File is corrupted or unparseable'
                })

            # Flag 2: Low epistemic score
            if score_data and score_data['final_score'] <= 1:
                risks.append({
                    'type': 'low_integrity',
                    'severity': 'MEDIUM',
                    'description': f'Low epistemic score: {score_data["final_score"]}'
                })

            # Flag 3: Unsupported claims (high claims, low citations)
            if claim_data:
                if claim_data['claim_count'] > 10 and claim_data['citations']['total'] == 0:
                    risks.append({
                        'type': 'unsupported_claims',
                        'severity': 'HIGH',
                        'description': f'{claim_data["claim_count"]} claims with no citations'
                    })

                # Flag 4: High speculation
                if claim_data['metrics']['speculation_density'] > 0.2:
                    risks.append({
                        'type': 'high_speculation',
                        'severity': 'MEDIUM',
                        'description': f'High speculation density: {claim_data["metrics"]["speculation_density"]:.2f}'
                    })

                # Flag 5: Mathematical inconsistency (equations without citations)
                if claim_data['equation_count'] > 3 and claim_data['citations']['total'] == 0:
                    risks.append({
                        'type': 'unverified_math',
                        'severity': 'HIGH',
                        'description': f'{claim_data["equation_count"]} equations with no citations'
                    })

            # Flag 6: Check for hallucination indicators
            filepath = REPO_ROOT / file_info['relative_path']
            if file_info['is_text']:
                try:
                    with open(filepath, 'r', encoding=file_info['encoding'] or 'utf-8', errors='ignore') as f:
                        content = f.read().lower()

                    # Look for suspicious patterns
                    hallucination_indicators = [
                        'doi:10.0000',  # Fake DOI
                        'arxiv:0000',    # Fake arXiv
                        'unpublished',
                        'personal communication',
                        'need citation',
                        '[citation needed]'
                    ]

                    for indicator in hallucination_indicators:
                        if indicator in content:
                            risks.append({
                                'type': 'hallucinated_citation',
                                'severity': 'HIGH',
                                'description': f'Potential hallucinated citation: "{indicator}"'
                            })
                            break
                except:
                    pass

            # Add to risk list if any risks found
            if risks:
                self.risk_flags.append({
                    'canonical_id': file_id,
                    'path': file_info['relative_path'],
                    'risk_count': len(risks),
                    'max_severity': max((r['severity'] for r in risks), key=lambda x: {'HIGH': 3, 'MEDIUM': 2, 'LOW': 1}[x]),
                    'risks': risks
                })

        # Sort by severity
        severity_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        self.risk_flags.sort(key=lambda x: (severity_order[x['max_severity']], -x['risk_count']))

        high_risk = sum(1 for r in self.risk_flags if r['max_severity'] == 'HIGH')
        medium_risk = sum(1 for r in self.risk_flags if r['max_severity'] == 'MEDIUM')

        print(f"✓ Flagged {len(self.risk_flags)} files with risks")
        print(f"  HIGH: {high_risk}, MEDIUM: {medium_risk}")

        return self.risk_flags

    def save_risk_flags(self):
        """Save risk flags"""
        output_file = OUTPUT_DIR / 'risk_flags.json'

        # Count by type
        risk_types = Counter()
        for flag in self.risk_flags:
            for risk in flag['risks']:
                risk_types[risk['type']] += 1

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'generated': datetime.now().isoformat(),
                'total_flagged_files': len(self.risk_flags),
                'risk_distribution': {
                    'HIGH': sum(1 for r in self.risk_flags if r['max_severity'] == 'HIGH'),
                    'MEDIUM': sum(1 for r in self.risk_flags if r['max_severity'] == 'MEDIUM'),
                    'LOW': sum(1 for r in self.risk_flags if r['max_severity'] == 'LOW')
                },
                'risk_types': dict(risk_types),
                'flagged_files': self.risk_flags
            }, f, indent=2)

        print(f"✓ Saved risk flags: {output_file}")


class ReportGenerator:
    """Generates comprehensive integrity reports"""

    def __init__(self):
        self.manifest = self._load_json('file_manifest.json')
        self.validation = self._load_json('format_validation.json')
        self.redundancy = self._load_json('redundancy_clusters.json')
        self.claims = self._load_json('claim_citation_analysis.json')
        self.scores = self._load_json('epistemic_scores.json')
        self.consistency = self._load_json('consistency_analysis.json')
        self.provenance = self._load_json('provenance_analysis.json')
        self.risks = self._load_json('risk_flags.json')

    def _load_json(self, filename):
        """Load JSON file"""
        with open(OUTPUT_DIR / filename, 'r') as f:
            return json.load(f)

    def generate_report(self):
        """Generate human-readable markdown report"""
        print("\n📋 PHASE 9: GENERATING FINAL REPORT")
        print("🔍 Compiling comprehensive report...")

        report_lines = []

        # Header
        report_lines.extend([
            "# Repository Integrity Report",
            "",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Repository:** The-Index",
            f"**Analysis Date:** {datetime.now().strftime('%Y-%m-%d')}",
            "",
            "---",
            ""
        ])

        # Executive Summary
        report_lines.extend(self._generate_executive_summary())

        # Detailed Findings
        report_lines.extend(self._generate_inventory_section())
        report_lines.extend(self._generate_validation_section())
        report_lines.extend(self._generate_redundancy_section())
        report_lines.extend(self._generate_epistemic_section())
        report_lines.extend(self._generate_consistency_section())
        report_lines.extend(self._generate_risk_section())
        report_lines.extend(self._generate_recommendations())

        # Save report
        output_file = OUTPUT_DIR / 'INTEGRITY_REPORT.md'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        print(f"✓ Saved integrity report: {output_file}")

        # Also generate a summary report in the root
        self._generate_summary_report(report_lines)

    def _generate_executive_summary(self):
        """Generate executive summary"""
        total_files = self.manifest['total_files']
        avg_score = sum(s['final_score'] for s in self.scores['scores']) / len(self.scores['scores'])
        high_quality = sum(1 for s in self.scores['scores'] if s['final_score'] >= 4)
        high_risk = self.risks['risk_distribution']['HIGH']

        # Calculate repo health score (0-100)
        health_components = {
            'avg_epistemic_score': (avg_score / 5) * 30,  # 30 points
            'high_quality_ratio': (high_quality / total_files) * 30,  # 30 points
            'low_risk_ratio': (1 - (high_risk / total_files)) * 25,  # 25 points
            'validation_success': (self.validation['statistics'].get('valid', 0) / total_files) * 15  # 15 points
        }
        health_score = int(sum(health_components.values()))

        # Determine status
        if health_score >= 80:
            status = "🟢 EXCELLENT"
        elif health_score >= 60:
            status = "🟡 GOOD"
        elif health_score >= 40:
            status = "🟠 FAIR"
        else:
            status = "🔴 NEEDS ATTENTION"

        return [
            "## Executive Summary",
            "",
            f"### Repository Health Score: {health_score}/100 {status}",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Files | {total_files:,} |",
            f"| Average Epistemic Score | {avg_score:.2f}/5 |",
            f"| High-Quality Files (≥4) | {high_quality} ({high_quality/total_files*100:.1f}%) |",
            f"| High-Risk Files | {high_risk} ({high_risk/total_files*100:.1f}%) |",
            f"| Valid Files | {self.validation['statistics'].get('valid', 0):,} ({self.validation['statistics'].get('valid', 0)/total_files*100:.1f}%) |",
            f"| Duplicate Groups | {self.redundancy['exact_duplicates']['count']} |",
            f"| Potential Storage Savings | {sum(g['storage_savings_bytes'] for g in self.redundancy['exact_duplicates']['groups']):,} bytes |",
            "",
            "---",
            ""
        ]

    def _generate_inventory_section(self):
        """Generate inventory section"""
        by_type = Counter()
        for file_info in self.manifest['files']:
            by_type[file_info['file_type']] += 1

        lines = [
            "## 1. File Inventory",
            "",
            f"**Total Files Cataloged:** {self.manifest['total_files']:,}",
            "",
            "### File Type Distribution",
            ""
        ]

        # Top 15 file types
        for ext, count in by_type.most_common(15):
            pct = count / self.manifest['total_files'] * 100
            lines.append(f"- `{ext or 'no extension'}`: {count:,} files ({pct:.1f}%)")

        lines.extend(["", "---", ""])
        return lines

    def _generate_validation_section(self):
        """Generate validation section"""
        stats = self.validation['statistics']

        lines = [
            "## 2. Format Validation",
            "",
            "| Status | Count | Percentage |",
            "|--------|-------|------------|"
        ]

        total = sum(stats.values())
        for status, count in sorted(stats.items(), key=lambda x: -x[1]):
            pct = count / total * 100
            lines.append(f"| {status} | {count:,} | {pct:.1f}% |")

        # Highlight issues
        issues = [r for r in self.validation['results'] if r['status'] in ['corrupt', 'error']]
        if issues:
            lines.extend([
                "",
                "### ⚠️ Files Requiring Attention",
                ""
            ])
            for issue in issues[:10]:  # Show first 10
                lines.append(f"- `{issue['path']}`: {', '.join(issue['issues'])}")
            if len(issues) > 10:
                lines.append(f"- *...and {len(issues) - 10} more*")

        lines.extend(["", "---", ""])
        return lines

    def _generate_redundancy_section(self):
        """Generate redundancy section"""
        exact_count = self.redundancy['exact_duplicates']['count']
        near_count = self.redundancy['near_duplicates']['count']
        total_savings = sum(g['storage_savings_bytes'] for g in self.redundancy['exact_duplicates']['groups'])

        lines = [
            "## 3. Redundancy Analysis",
            "",
            f"**Exact Duplicate Groups:** {exact_count}",
            f"**Near-Duplicate Pairs:** {near_count}",
            f"**Potential Storage Savings:** {total_savings:,} bytes ({total_savings/1024/1024:.2f} MB)",
            "",
            "### Largest Duplicate Groups",
            ""
        ]

        # Show top duplicate groups
        sorted_groups = sorted(
            self.redundancy['exact_duplicates']['groups'],
            key=lambda x: x['storage_savings_bytes'],
            reverse=True
        )

        for group in sorted_groups[:5]:
            lines.append(f"**Canonical:** `{group['canonical']['path']}`")
            lines.append(f"- Size: {group['canonical']['size']:,} bytes")
            lines.append(f"- Duplicates: {len(group['duplicates'])}")
            lines.append(f"- Savings: {group['storage_savings_bytes']:,} bytes")
            for dup in group['duplicates'][:3]:
                lines.append(f"  - `{dup['path']}`")
            if len(group['duplicates']) > 3:
                lines.append(f"  - *...and {len(group['duplicates']) - 3} more*")
            lines.append("")

        lines.extend(["---", ""])
        return lines

    def _generate_epistemic_section(self):
        """Generate epistemic integrity section"""
        dist = self.scores['score_distribution']

        lines = [
            "## 4. Epistemic Integrity",
            "",
            "### Score Distribution",
            "",
            "| Score | Meaning | Count | Percentage |",
            "|-------|---------|-------|------------|"
        ]

        meanings = {
            0: "Unparseable/Corrupted",
            1: "Notes/Speculation",
            2: "Unsourced Claims",
            3: "Internally Consistent",
            4: "Externally Corroborated",
            5: "Reproducible + Cited"
        }

        total = sum(dist.values())
        for score in range(6):
            count = dist.get(score, 0)
            pct = count / total * 100
            lines.append(f"| {score} | {meanings[score]} | {count:,} | {pct:.1f}% |")

        # High-confidence corpus
        high_quality = [s for s in self.scores['scores'] if s['final_score'] >= 4]
        lines.extend([
            "",
            f"### 📚 High-Confidence Corpus ({len(high_quality)} files)",
            "",
            "Files with epistemic score ≥ 4:",
            ""
        ])

        for item in high_quality[:15]:
            lines.append(f"- `{item['path']}` (Score: {item['final_score']})")
        if len(high_quality) > 15:
            lines.append(f"- *...and {len(high_quality) - 15} more*")

        lines.extend(["", "---", ""])
        return lines

    def _generate_consistency_section(self):
        """Generate consistency section"""
        dep_count = self.consistency['dependencies']['count']

        lines = [
            "## 5. Cross-File Consistency",
            "",
            f"**File Dependencies Detected:** {dep_count:,}",
            f"**Contradictions Found:** {self.consistency['contradictions']['count']}",
            "",
            "### Dependency Analysis",
            "",
            f"The repository has {dep_count:,} file references, indicating a highly interconnected knowledge base.",
            ""
        ]

        if dep_count > 0:
            lines.append("📊 **Dependency graph:** See `integrity/dependency_graph.dot` (can be visualized with Graphviz)")

        lines.extend(["", "---", ""])
        return lines

    def _generate_risk_section(self):
        """Generate risk section"""
        dist = self.risks['risk_distribution']

        lines = [
            "## 6. Risk Assessment",
            "",
            "### Risk Distribution",
            "",
            "| Severity | Count | Percentage |",
            "|----------|-------|------------|"
        ]

        total_flagged = self.risks['total_flagged_files']
        for severity in ['HIGH', 'MEDIUM', 'LOW']:
            count = dist.get(severity, 0)
            pct = count / total_flagged * 100 if total_flagged > 0 else 0
            lines.append(f"| {severity} | {count} | {pct:.1f}% |")

        lines.extend([
            "",
            "### Risk Types",
            ""
        ])

        for risk_type, count in sorted(self.risks['risk_types'].items(), key=lambda x: -x[1]):
            lines.append(f"- **{risk_type.replace('_', ' ').title()}**: {count} occurrences")

        # Show high-risk files
        high_risk_files = [f for f in self.risks['flagged_files'] if f['max_severity'] == 'HIGH']
        if high_risk_files:
            lines.extend([
                "",
                "### ⚠️ High-Risk Files Requiring Review",
                ""
            ])

            for item in high_risk_files[:10]:
                lines.append(f"**`{item['path']}`**")
                for risk in item['risks']:
                    lines.append(f"- [{risk['severity']}] {risk['description']}")
                lines.append("")

            if len(high_risk_files) > 10:
                lines.append(f"*...and {len(high_risk_files) - 10} more high-risk files*")

        lines.extend(["---", ""])
        return lines

    def _generate_recommendations(self):
        """Generate recommendations"""
        return [
            "## 7. Recommendations",
            "",
            "### Immediate Actions",
            "",
            f"1. **Review High-Risk Files**: {self.risks['risk_distribution'].get('HIGH', 0)} files flagged with HIGH severity",
            f"2. **Address Corruption**: {self.validation['statistics'].get('corrupt', 0)} files are corrupted or unreadable",
            f"3. **Deduplicate**: Remove {self.redundancy['exact_duplicates']['count']} duplicate groups to save {sum(g['storage_savings_bytes'] for g in self.redundancy['exact_duplicates']['groups'])/1024/1024:.2f} MB",
            "",
            "### Improvement Opportunities",
            "",
            "1. **Citation Enhancement**: Add citations to files with unsupported claims",
            "2. **Consistency Review**: Resolve any identified contradictions",
            "3. **Documentation**: Improve low-scoring files (score ≤ 2) with better sourcing",
            "",
            "### Canonical File Set",
            "",
            "Consider promoting high-quality files (score ≥ 4) as canonical references:",
            f"- {len([s for s in self.scores['scores'] if s['final_score'] >= 4])} files qualify",
            f"- See `epistemic_scores.json` for complete list",
            "",
            "---",
            "",
            "## Data Files",
            "",
            "All detailed data available in JSON format:",
            "",
            "- `file_manifest.json` - Complete file inventory",
            "- `canonical_mapping.json` - Canonical ID mappings",
            "- `format_validation.json` - Validation results",
            "- `redundancy_clusters.json` - Duplicate analysis",
            "- `claim_citation_analysis.json` - Claims and citations",
            "- `epistemic_scores.json` - Integrity scores",
            "- `consistency_analysis.json` - Cross-file consistency",
            "- `provenance_analysis.json` - File lineage",
            "- `risk_flags.json` - Risk assessment",
            "- `dependency_graph.dot` - Dependency visualization",
            "",
            "---",
            "",
            f"*Report generated by Repository Integrity Analysis Tool v1.0*"
        ]

    def _generate_summary_report(self, full_report_lines):
        """Generate a shorter summary report for the repository root"""
        # Extract key sections
        summary_lines = []
        in_summary = False
        in_recommendations = False

        for line in full_report_lines:
            if line.startswith("# Repository Integrity Report"):
                summary_lines.append(line)
                in_summary = True
            elif line.startswith("## Executive Summary"):
                summary_lines.append(line)
            elif line.startswith("## 1."):
                in_summary = False
            elif line.startswith("## 7. Recommendations"):
                in_recommendations = True
                summary_lines.append(line)
            elif in_recommendations and line.startswith("##"):
                break
            elif in_summary or in_recommendations:
                summary_lines.append(line)

        # Add pointer to full report
        summary_lines.extend([
            "",
            "---",
            "",
            "📁 **Full detailed report:** `integrity/INTEGRITY_REPORT.md`",
            ""
        ])

        # Save summary
        output_file = REPO_ROOT / 'INTEGRITY_SUMMARY.md'
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(summary_lines))

        print(f"✓ Saved summary report: {output_file}")


def main():
    """Main finalization pipeline"""
    print("=" * 70)
    print("FINAL ANALYSIS: Provenance, Risks, and Reporting")
    print("=" * 70)

    # Phase 7: Provenance analysis
    provenance = ProvenanceAnalyzer()
    provenance.analyze_provenance()
    provenance.save_provenance_analysis()

    # Phase 8: Risk flagging
    risk_flagger = RiskFlagger()
    risk_flagger.flag_risks()
    risk_flagger.save_risk_flags()

    # Phase 9: Report generation
    reporter = ReportGenerator()
    reporter.generate_report()

    print("\n" + "=" * 70)
    print("✓ ANALYSIS COMPLETE")
    print("=" * 70)
    print("\n📊 All reports generated:")
    print(f"  - Main Report: integrity/INTEGRITY_REPORT.md")
    print(f"  - Summary: INTEGRITY_SUMMARY.md")
    print(f"  - Data Files: integrity/*.json")
    print()


if __name__ == '__main__':
    main()
