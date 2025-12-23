#!/usr/bin/env python3
"""
Content Analysis Phase: Claims, Citations, Epistemic Scoring
Analyzes research content, extracts claims/citations, scores integrity
"""

import os
import json
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

# Configuration
REPO_ROOT = Path("/home/user/The-Index")
OUTPUT_DIR = REPO_ROOT / "integrity"

# Citation patterns
DOI_PATTERN = r'\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b'
ARXIV_PATTERN = r'\barXiv:\d{4}\.\d{4,5}(?:v\d+)?\b'
ISBN_PATTERN = r'\bISBN[-\s]?(?:\d{1,5}[-\s]?){3}\d{1,5}[X]?\b'
URL_PATTERN = r'https?://[^\s\)>\]"]+'

# Claim indicators (declarative statements)
CLAIM_INDICATORS = [
    r'\b(?:is|are|was|were|has|have|had|will|would|can|could|may|might|must|shall|should)\b',
    r'\b(?:demonstrates|shows|proves|indicates|suggests|reveals|confirms|establishes)\b',
    r'\b(?:therefore|thus|hence|consequently|accordingly)\b',
]

# Mathematical patterns
EQUATION_PATTERN = r'(?:\$.*?\$|\\begin\{equation\}.*?\\end\{equation\}|\\begin\{align\}.*?\\end\{align\})'

# Speculative language
SPECULATIVE_INDICATORS = [
    r'\b(?:might|perhaps|possibly|potentially|could be|may be|seems|appears)\b',
    r'\b(?:hypothesis|speculation|conjecture|theoretical)\b',
    r'\b(?:unclear|uncertain|unknown|unexplained)\b',
]


class ClaimExtractor:
    """Extracts claims and citations from research files"""

    def __init__(self):
        self.manifest = self._load_manifest()
        self.claim_data = []

    def _load_manifest(self):
        """Load file manifest"""
        with open(OUTPUT_DIR / 'file_manifest.json', 'r') as f:
            return json.load(f)

    def analyze_all(self):
        """Analyze all text files for claims and citations"""
        print("\n📋 PHASE 4: CLAIM & CITATION EXTRACTION")
        print("🔍 Analyzing research content...")

        files = self.manifest['files']
        text_files = [f for f in files if f['is_text']]

        for file_info in text_files:
            analysis = self._analyze_file(file_info)
            if analysis:
                self.claim_data.append(analysis)

        print(f"✓ Analyzed {len(text_files)} text files")
        return self.claim_data

    def _analyze_file(self, file_info):
        """Analyze a single file for claims and citations"""
        filepath = REPO_ROOT / file_info['relative_path']

        try:
            with open(filepath, 'r', encoding=file_info['encoding'] or 'utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            return None

        # Extract citations
        dois = re.findall(DOI_PATTERN, content, re.IGNORECASE)
        arxiv_ids = re.findall(ARXIV_PATTERN, content, re.IGNORECASE)
        isbns = re.findall(ISBN_PATTERN, content, re.IGNORECASE)
        urls = re.findall(URL_PATTERN, content)

        # Count claims (sentences with declarative patterns)
        sentences = re.split(r'[.!?]+', content)
        claim_count = 0
        for sentence in sentences:
            if len(sentence.strip()) > 20:  # Reasonable sentence length
                for pattern in CLAIM_INDICATORS:
                    if re.search(pattern, sentence, re.IGNORECASE):
                        claim_count += 1
                        break

        # Count equations
        equations = re.findall(EQUATION_PATTERN, content, re.DOTALL)

        # Count speculative language
        speculation_count = 0
        for pattern in SPECULATIVE_INDICATORS:
            speculation_count += len(re.findall(pattern, content, re.IGNORECASE))

        # Extract named entities (simple heuristic: capitalized words)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
        entity_freq = Counter(capitalized)
        # Filter common words
        common_words = {'The', 'This', 'That', 'These', 'Those', 'There', 'Here', 'When', 'Where'}
        named_entities = {k: v for k, v in entity_freq.items() if k not in common_words and v >= 2}

        total_citations = len(dois) + len(arxiv_ids) + len(isbns) + len(urls)

        # Calculate citation ratio
        citation_ratio = total_citations / claim_count if claim_count > 0 else 0

        return {
            'canonical_id': file_info['canonical_id'],
            'path': file_info['relative_path'],
            'file_type': file_info['file_type'],
            'claim_count': claim_count,
            'citations': {
                'dois': len(dois),
                'arxiv': len(arxiv_ids),
                'isbns': len(isbns),
                'urls': len(urls),
                'total': total_citations
            },
            'citation_ratio': round(citation_ratio, 3),
            'equation_count': len(equations),
            'speculation_count': speculation_count,
            'named_entities': dict(list(named_entities.items())[:10]),  # Top 10
            'metrics': {
                'sentences': len(sentences),
                'claim_density': round(claim_count / len(sentences), 3) if sentences else 0,
                'speculation_density': round(speculation_count / len(sentences), 3) if sentences else 0
            }
        }

    def save_claim_analysis(self):
        """Save claim analysis results"""
        output_file = OUTPUT_DIR / 'claim_citation_analysis.json'

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'generated': datetime.now().isoformat(),
                'total_files_analyzed': len(self.claim_data),
                'files': self.claim_data
            }, f, indent=2)

        print(f"✓ Saved claim analysis: {output_file}")


class EpistemicScorer:
    """Scores files for epistemic integrity (0-5 scale)"""

    def __init__(self):
        self.claim_data = self._load_claim_data()
        self.validation_data = self._load_validation_data()
        self.scores = []

    def _load_claim_data(self):
        """Load claim analysis data"""
        with open(OUTPUT_DIR / 'claim_citation_analysis.json', 'r') as f:
            data = json.load(f)
        return {item['canonical_id']: item for item in data['files']}

    def _load_validation_data(self):
        """Load validation results"""
        with open(OUTPUT_DIR / 'format_validation.json', 'r') as f:
            data = json.load(f)
        return {item['canonical_id']: item for item in data['results']}

    def score_all(self):
        """Score all files for epistemic integrity"""
        print("\n📋 PHASE 5: EPISTEMIC INTEGRITY SCORING")
        print("🔍 Computing integrity scores...")

        manifest_path = OUTPUT_DIR / 'file_manifest.json'
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)

        for file_info in manifest['files']:
            score_data = self._score_file(file_info)
            self.scores.append(score_data)

        # Compute statistics
        avg_score = sum(s['final_score'] for s in self.scores) / len(self.scores)
        score_dist = Counter(s['final_score'] for s in self.scores)

        print(f"✓ Scored {len(self.scores)} files")
        print(f"  Average score: {avg_score:.2f}")
        print(f"  Distribution: {dict(score_dist)}")

        return self.scores

    def _score_file(self, file_info):
        """Score a single file for epistemic integrity"""
        canonical_id = file_info['canonical_id']

        # Get validation status
        validation = self.validation_data.get(canonical_id, {})
        val_status = validation.get('status', 'unknown')

        # Get claim data if available
        claims = self.claim_data.get(canonical_id)

        # Initialize score components
        scores = {
            'parseable': 0,
            'internal_consistency': 0,
            'source_traceability': 0,
            'method_clarity': 0,
            'mathematical_coherence': 0
        }

        # Score 0: Unparseable/corrupted
        if val_status == 'corrupt' or val_status == 'error':
            final_score = 0
            reason = 'File is corrupted or unparseable'

        # Score for parseable files
        elif val_status == 'valid':
            scores['parseable'] = 1

            if claims:
                # Has content analysis

                # Internal consistency: low speculation, reasonable claim density
                if claims['speculation_count'] < claims['claim_count'] * 0.3:
                    scores['internal_consistency'] = 1

                # Source traceability: has citations
                if claims['citations']['total'] > 0:
                    scores['source_traceability'] = 1

                    # High citation ratio
                    if claims['citation_ratio'] >= 0.1:
                        scores['source_traceability'] = 2

                # Method clarity: has structured content
                if file_info['file_type'] in ['.md', '.tex', '.py']:
                    scores['method_clarity'] = 1

                # Mathematical coherence: has equations with citations
                if claims['equation_count'] > 0 and claims['citations']['total'] > 0:
                    scores['mathematical_coherence'] = 1

                # Determine final score
                total_indicators = sum(scores.values())

                if total_indicators >= 5:
                    final_score = 5
                    reason = 'Reproducible with strong citations'
                elif total_indicators >= 4:
                    final_score = 4
                    reason = 'Externally corroborated'
                elif total_indicators >= 3:
                    final_score = 3
                    reason = 'Internally consistent'
                elif total_indicators >= 2:
                    final_score = 2
                    reason = 'Partially sourced'
                else:
                    final_score = 1
                    reason = 'Notes or speculation'

            else:
                # No claims analysis (binary, non-text, etc.)
                if file_info['file_type'] in ['.json', '.csv', '.py', '.js']:
                    final_score = 3
                    reason = 'Structured data or code'
                else:
                    final_score = 2
                    reason = 'Valid but unanalyzed'

        else:
            # Skipped or unknown
            final_score = 2
            reason = 'Not analyzed'

        return {
            'canonical_id': canonical_id,
            'path': file_info['relative_path'],
            'final_score': final_score,
            'reason': reason,
            'sub_scores': scores,
            'validation_status': val_status
        }

    def save_scores(self):
        """Save epistemic scores"""
        output_file = OUTPUT_DIR / 'epistemic_scores.json'

        # Group by score
        by_score = defaultdict(list)
        for score_data in self.scores:
            by_score[score_data['final_score']].append(score_data)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'generated': datetime.now().isoformat(),
                'total_files': len(self.scores),
                'score_distribution': {
                    score: len(files) for score, files in sorted(by_score.items())
                },
                'scores': self.scores,
                'by_score': {
                    str(score): files for score, files in sorted(by_score.items())
                }
            }, f, indent=2)

        print(f"✓ Saved epistemic scores: {output_file}")


class ConsistencyChecker:
    """Checks cross-file consistency and builds dependency graph"""

    def __init__(self):
        self.manifest = self._load_manifest()
        self.claim_data = self._load_claim_data()
        self.contradictions = []
        self.dependencies = []

    def _load_manifest(self):
        """Load file manifest"""
        with open(OUTPUT_DIR / 'file_manifest.json', 'r') as f:
            return json.load(f)

    def _load_claim_data(self):
        """Load claim analysis"""
        with open(OUTPUT_DIR / 'claim_citation_analysis.json', 'r') as f:
            data = json.load(f)
        return {item['canonical_id']: item for item in data['files']}

    def analyze_consistency(self):
        """Perform cross-file consistency checks"""
        print("\n📋 PHASE 6: CROSS-FILE CONSISTENCY")
        print("🔍 Checking for contradictions and dependencies...")

        # Build file content index
        content_index = {}
        for file_info in self.manifest['files']:
            if file_info['is_text']:
                filepath = REPO_ROOT / file_info['relative_path']
                try:
                    with open(filepath, 'r', encoding=file_info['encoding'] or 'utf-8', errors='ignore') as f:
                        content = f.read()
                    content_index[file_info['canonical_id']] = {
                        'path': file_info['relative_path'],
                        'content': content.lower()
                    }
                except:
                    pass

        # Check for file references (dependencies)
        for file_id, data in content_index.items():
            for other_id, other_data in content_index.items():
                if file_id == other_id:
                    continue

                # Check if one file references another
                other_filename = Path(other_data['path']).name
                if other_filename.lower() in data['content']:
                    self.dependencies.append({
                        'source': file_id,
                        'source_path': data['path'],
                        'target': other_id,
                        'target_path': other_data['path'],
                        'type': 'file_reference'
                    })

        print(f"✓ Found {len(self.dependencies)} file dependencies")

        return {
            'contradictions': self.contradictions,
            'dependencies': self.dependencies
        }

    def build_dependency_graph(self):
        """Build dependency graph in DOT format"""
        print("🔍 Building dependency graph...")

        dot_content = ['digraph DependencyGraph {']
        dot_content.append('  rankdir=LR;')
        dot_content.append('  node [shape=box, style=rounded];')
        dot_content.append('')

        # Add nodes
        nodes = set()
        for dep in self.dependencies:
            nodes.add((dep['source'], dep['source_path']))
            nodes.add((dep['target'], dep['target_path']))

        for node_id, path in nodes:
            label = Path(path).name
            dot_content.append(f'  "{node_id}" [label="{label}"];')

        dot_content.append('')

        # Add edges
        for dep in self.dependencies:
            dot_content.append(f'  "{dep["source"]}" -> "{dep["target"]}";')

        dot_content.append('}')

        # Save DOT file
        output_file = OUTPUT_DIR / 'dependency_graph.dot'
        with open(output_file, 'w') as f:
            f.write('\n'.join(dot_content))

        print(f"✓ Saved dependency graph: {output_file}")

    def save_consistency_report(self):
        """Save consistency analysis"""
        output_file = OUTPUT_DIR / 'consistency_analysis.json'

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'generated': datetime.now().isoformat(),
                'contradictions': {
                    'count': len(self.contradictions),
                    'items': self.contradictions
                },
                'dependencies': {
                    'count': len(self.dependencies),
                    'items': self.dependencies
                }
            }, f, indent=2)

        print(f"✓ Saved consistency report: {output_file}")


def main():
    """Main content analysis pipeline"""
    print("=" * 70)
    print("CONTENT ANALYSIS: Claims, Citations, Scoring")
    print("=" * 70)

    # Phase 4: Claim extraction
    extractor = ClaimExtractor()
    extractor.analyze_all()
    extractor.save_claim_analysis()

    # Phase 5: Epistemic scoring
    scorer = EpistemicScorer()
    scorer.score_all()
    scorer.save_scores()

    # Phase 6: Consistency checking
    checker = ConsistencyChecker()
    checker.analyze_consistency()
    checker.build_dependency_graph()
    checker.save_consistency_report()

    print("\n" + "=" * 70)
    print("✓ Phase 4-6 Complete: Claims, Scoring, Consistency")
    print("=" * 70)


if __name__ == '__main__':
    main()
