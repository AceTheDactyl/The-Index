# INTEGRITY_METADATA
# Date: 2025-12-23
# Status: NEEDS_REVIEW - Reference implementation
# Severity: MEDIUM RISK
# Risk Types: ['reference_material']
# File: systems/Ace-Systems/reference/APL/Quantum-APL/dylan/self_learning_coder.py

"""
SELF-LEARNING CODER
===================

A code generator that:
1. Learns from its own codebase (introspection)
2. Explores GitHub to learn new patterns
3. Freely explores without direction
4. Grows its knowledge over time

Core idea: The system IS its own training data.
"""

import os
import re
import ast
import json
import random
import urllib.request
import urllib.parse
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from pathlib import Path
import time


# =============================================================================
# KNOWLEDGE BASE - Stores learned patterns
# =============================================================================

@dataclass
class LearnedPattern:
    """A pattern learned from code"""
    name: str
    signature: str
    body: str
    examples: List[Tuple]
    source: str  # "self", "github", "user"
    confidence: float = 1.0
    times_used: int = 0


class KnowledgeBase:
    """Persistent storage for learned patterns"""

    def __init__(self, path: str = "knowledge.json"):
        self.path = path
        self.patterns: Dict[str, LearnedPattern] = {}
        self.load()

    def load(self):
        """Load knowledge from disk"""
        if os.path.exists(self.path):
            try:
                with open(self.path, 'r') as f:
                    data = json.load(f)
                for name, p in data.items():
                    self.patterns[name] = LearnedPattern(
                        name=p['name'],
                        signature=p['signature'],
                        body=p['body'],
                        examples=[tuple(e) for e in p['examples']],
                        source=p['source'],
                        confidence=p.get('confidence', 1.0),
                        times_used=p.get('times_used', 0)
                    )
            except:
                pass

    def save(self):
        """Save knowledge to disk"""
        data = {}
        for name, p in self.patterns.items():
            data[name] = {
                'name': p.name,
                'signature': p.signature,
                'body': p.body,
                'examples': list(p.examples),
                'source': p.source,
                'confidence': p.confidence,
                'times_used': p.times_used
            }
        with open(self.path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def add(self, pattern: LearnedPattern):
        """Add or update a pattern"""
        self.patterns[pattern.name] = pattern
        self.save()

    def get(self, name: str) -> Optional[LearnedPattern]:
        """Get a pattern by name"""
        return self.patterns.get(name)

    def search(self, query: str) -> List[LearnedPattern]:
        """Search patterns by name or signature"""
        results = []
        query = query.lower()
        for p in self.patterns.values():
            if query in p.name.lower() or query in p.signature.lower():
                results.append(p)
        return results

    def stats(self) -> Dict:
        """Get knowledge base statistics"""
        sources = {}
        for p in self.patterns.values():
            sources[p.source] = sources.get(p.source, 0) + 1
        return {
            'total_patterns': len(self.patterns),
            'by_source': sources
        }


# =============================================================================
# SELF-INTROSPECTION - Learn from own codebase
# =============================================================================

class CodeIntrospector:
    """Extracts patterns from Python source files"""

    def __init__(self):
        self.extracted: List[LearnedPattern] = []

    def scan_directory(self, path: str) -> List[LearnedPattern]:
        """Scan directory for Python files and extract functions"""
        patterns = []

        for root, dirs, files in os.walk(path):
            # Skip hidden and virtual env directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ('venv', '__pycache__', 'node_modules')]

            for filename in files:
                if filename.endswith('.py'):
                    filepath = os.path.join(root, filename)
                    try:
                        file_patterns = self.extract_from_file(filepath)
                        patterns.extend(file_patterns)
                    except:
                        pass

        self.extracted = patterns
        return patterns

    def extract_from_file(self, filepath: str) -> List[LearnedPattern]:
        """Extract function patterns from a single file"""
        patterns = []

        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            source = f.read()

        try:
            tree = ast.parse(source)
        except:
            return []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                pattern = self._extract_function(node, source, filepath)
                if pattern:
                    patterns.append(pattern)

        return patterns

    def _extract_function(self, node: ast.FunctionDef, source: str, filepath: str) -> Optional[LearnedPattern]:
        """Extract a single function as a pattern"""
        name = node.name

        # Skip private/magic methods
        if name.startswith('_'):
            return None

        # Get signature
        args = [arg.arg for arg in node.args.args]
        signature = f"def {name}({', '.join(args)})"

        # Get body
        try:
            body = ast.get_source_segment(source, node)
            # Remove the def line, keep just the body
            lines = body.split('\n')[1:]
            body = '\n'.join(line[4:] if line.startswith('    ') else line for line in lines)
        except:
            return None

        if not body.strip():
            return None

        # Try to generate examples by running the function
        examples = self._generate_examples(name, args, body)

        return LearnedPattern(
            name=name,
            signature=signature,
            body=body.strip(),
            examples=examples,
            source=f"self:{filepath}"
        )

    def _generate_examples(self, name: str, args: List[str], body: str) -> List[Tuple]:
        """Try to generate examples by running the function"""
        examples = []

        # Build test inputs based on number of args
        test_inputs = self._get_test_inputs(len(args))

        # Try each input
        for inp in test_inputs:
            try:
                # Build function
                indent_body = '\n'.join('    ' + line for line in body.split('\n'))
                func_code = f"def {name}({', '.join(args)}):\n{indent_body}"

                namespace = {'__builtins__': __builtins__}
                exec(func_code, namespace)
                func = namespace[name]

                # Run
                if isinstance(inp, tuple):
                    result = func(*inp)
                else:
                    result = func(inp)

                # Store example
                examples.append((inp, result))

                if len(examples) >= 5:
                    break

            except:
                pass

        return examples

    def _get_test_inputs(self, n_args: int) -> List:
        """Generate test inputs"""
        if n_args == 0:
            return [()]
        elif n_args == 1:
            return [0, 1, 2, 5, -1, 10, "test", "", [1, 2, 3], []]
        elif n_args == 2:
            return [
                (0, 0), (1, 2), (3, 4), (5, 1), (-1, 1),
                ("a", "b"), ("hello", "world"),
                ([1, 2], [3, 4])
            ]
        else:
            return [(0,) * n_args, (1,) * n_args, (2,) * n_args]


# =============================================================================
# GITHUB EXPLORER - Learn from the world
# =============================================================================

class GitHubExplorer:
    """Explores GitHub to learn new patterns"""

    def __init__(self):
        self.base_url = "https://api.github.com"
        self.raw_url = "https://raw.githubusercontent.com"
        self.learned: List[LearnedPattern] = []

        # Known good algorithm repos (no auth needed for raw files)
        self.known_repos = [
            ("TheAlgorithms/Python", "main", [
                "maths/factorial.py",
                "maths/fibonacci.py",
                "maths/prime_check.py",
                "maths/greatest_common_divisor.py",
                "maths/binary_exponentiation.py",
                "strings/reverse_words.py",
                "strings/palindrome.py",
                "sorts/bubble_sort.py",
                "sorts/quick_sort.py",
                "searches/binary_search.py",
                "searches/linear_search.py",
            ]),
        ]

    def search_code(self, query: str, language: str = "python") -> List[Dict]:
        """Search GitHub for code (requires auth, may fail)"""
        try:
            url = f"{self.base_url}/search/code?q={urllib.parse.quote(query)}+language:{language}&per_page=10"
            req = urllib.request.Request(url, headers={'User-Agent': 'SelfLearningCoder/1.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode())
                return data.get('items', [])
        except Exception as e:
            # Silently fail - will use known repos instead
            return []

    def fetch_file(self, repo: str, path: str, branch: str = "main") -> Optional[str]:
        """Fetch a file from GitHub"""
        try:
            url = f"{self.raw_url}/{repo}/{branch}/{path}"
            req = urllib.request.Request(url, headers={'User-Agent': 'SelfLearningCoder/1.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                return response.read().decode('utf-8', errors='ignore')
        except:
            # Try master branch
            try:
                url = f"{self.raw_url}/{repo}/master/{path}"
                req = urllib.request.Request(url, headers={'User-Agent': 'SelfLearningCoder/1.0'})
                with urllib.request.urlopen(req, timeout=10) as response:
                    return response.read().decode('utf-8', errors='ignore')
            except:
                return None

    def explore_topic(self, topic: str) -> List[LearnedPattern]:
        """Explore a topic and learn patterns"""
        patterns = []

        # Search for code
        results = self.search_code(f"def {topic}")

        for item in results[:5]:  # Limit to 5 results
            try:
                repo = item['repository']['full_name']
                path = item['path']

                # Fetch file
                content = self.fetch_file(repo, path)
                if not content:
                    continue

                # Extract functions
                introspector = CodeIntrospector()
                try:
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            if topic.lower() in node.name.lower():
                                pattern = introspector._extract_function(node, content, f"github:{repo}/{path}")
                                if pattern:
                                    pattern.source = f"github:{repo}"
                                    patterns.append(pattern)
                except:
                    pass

                time.sleep(0.5)  # Rate limiting

            except:
                pass

        self.learned.extend(patterns)
        return patterns

    def explore_trending(self) -> List[LearnedPattern]:
        """Explore trending Python repos"""
        patterns = []

        # Common utility function names to search for
        topics = ['sort', 'search', 'parse', 'validate', 'convert', 'calculate', 'process', 'filter', 'transform']

        topic = random.choice(topics)
        patterns = self.explore_topic(topic)

        return patterns

    def free_explore(self, duration_seconds: int = 60) -> List[LearnedPattern]:
        """Freely explore GitHub for a duration"""
        all_patterns = []
        start_time = time.time()

        topics = [
            'factorial', 'fibonacci', 'prime', 'sort', 'search', 'binary',
            'reverse', 'palindrome', 'gcd', 'lcm', 'power', 'sqrt',
            'max', 'min', 'sum', 'average', 'median', 'mode',
            'encrypt', 'decrypt', 'hash', 'encode', 'decode',
            'parse', 'format', 'validate', 'sanitize', 'escape',
            'distance', 'similarity', 'compare', 'merge', 'split',
            'flatten', 'unique', 'duplicate', 'count', 'group',
            'permutation', 'combination', 'subset', 'partition'
        ]

        explored = set()

        while time.time() - start_time < duration_seconds:
            # Pick random topic not yet explored
            available = [t for t in topics if t not in explored]
            if not available:
                break

            topic = random.choice(available)
            explored.add(topic)

            print(f"  Exploring: {topic}...")
            patterns = self.explore_topic(topic)
            all_patterns.extend(patterns)
            print(f"    Found {len(patterns)} patterns")

            if len(all_patterns) >= 50:  # Limit total
                break

            time.sleep(1)  # Rate limiting

        return all_patterns


# =============================================================================
# PATTERN SYNTHESIZER - Generate code from patterns
# =============================================================================

class PatternSynthesizer:
    """Synthesizes code from learned patterns"""

    def __init__(self, knowledge: KnowledgeBase):
        self.knowledge = knowledge

    def generate(self, name: str, signature: str, examples: List[Tuple]) -> Optional[str]:
        """Generate code using learned patterns"""

        # Check if we have an exact match
        pattern = self.knowledge.get(name)
        if pattern:
            pattern.times_used += 1
            self.knowledge.save()
            return self._build_function(pattern.signature, pattern.body)

        # Search for similar patterns
        similar = self.knowledge.search(name)
        if similar:
            # Use most confident/used pattern
            best = max(similar, key=lambda p: p.confidence * (1 + p.times_used * 0.1))
            adapted = self._adapt_pattern(best, name, signature, examples)
            if adapted:
                return adapted

        # Try to match by examples
        for pattern in self.knowledge.patterns.values():
            if self._examples_match(pattern.examples, examples):
                adapted = self._adapt_pattern(pattern, name, signature, examples)
                if adapted:
                    return adapted

        return None

    def _build_function(self, signature: str, body: str) -> str:
        """Build function from signature and body"""
        indent_body = '\n'.join('    ' + line for line in body.split('\n'))
        return f"{signature}:\n{indent_body}"

    def _adapt_pattern(self, pattern: LearnedPattern, name: str, signature: str, examples: List[Tuple]) -> Optional[str]:
        """Adapt a pattern to new name/signature"""
        # Get original and new param names
        orig_params = self._extract_params(pattern.signature)
        new_params = self._extract_params(signature)

        if len(orig_params) != len(new_params):
            return None

        # Replace param names in body
        body = pattern.body
        for old, new in zip(orig_params, new_params):
            body = re.sub(rf'\b{old}\b', new, body)

        # Build and test
        func_code = self._build_function(signature, body)

        if self._test_function(func_code, name, examples):
            return func_code

        return None

    def _extract_params(self, signature: str) -> List[str]:
        """Extract parameter names"""
        match = re.search(r'\(([^)]*)\)', signature)
        if not match:
            return []
        params = []
        for p in match.group(1).split(','):
            p = p.strip().split(':')[0].split('=')[0].strip()
            if p:
                params.append(p)
        return params

    def _examples_match(self, learned: List[Tuple], target: List[Tuple]) -> bool:
        """Check if examples represent same transformation"""
        if not learned or not target:
            return False

        # Check if input/output types match
        try:
            l_in, l_out = learned[0]
            t_in, t_out = target[0]

            if type(l_in) != type(t_in) or type(l_out) != type(t_out):
                return False

            # Check if transformation is similar
            # This is a simple heuristic - could be more sophisticated
            return True
        except:
            return False

    def _test_function(self, code: str, name: str, examples: List[Tuple]) -> bool:
        """Test if generated code works"""
        try:
            namespace = {}
            exec(code, namespace)
            func = namespace[name]

            for inp, expected in examples:
                if isinstance(inp, tuple):
                    result = func(*inp)
                else:
                    result = func(inp)

                if result != expected:
                    return False

            return True
        except:
            return False


# =============================================================================
# SELF-LEARNING CODER - Main interface
# =============================================================================

class SelfLearningCoder:
    """
    A coder that learns from itself and the world.

    Usage:
        coder = SelfLearningCoder()
        coder.learn_from_self()          # Learn from own codebase
        coder.explore_github()           # Learn from GitHub
        code = coder.generate(...)       # Generate code
    """

    def __init__(self, knowledge_path: str = "knowledge.json"):
        self.knowledge = KnowledgeBase(knowledge_path)
        self.introspector = CodeIntrospector()
        self.github = GitHubExplorer()
        self.synthesizer = PatternSynthesizer(self.knowledge)

        # Import the universal coder as fallback
        try:
            from universal_coder import UniversalCoder, Spec
            self.universal = UniversalCoder()
            self.Spec = Spec
        except:
            self.universal = None
            self.Spec = None

    def learn_from_self(self, path: str = ".") -> int:
        """Learn patterns from own codebase"""
        print(f"Scanning {path} for patterns...")

        patterns = self.introspector.scan_directory(path)

        added = 0
        for p in patterns:
            if p.examples:  # Only add if we have examples
                self.knowledge.add(p)
                added += 1

        print(f"Learned {added} patterns from self")
        return added

    def learn_from_file(self, filepath: str) -> int:
        """Learn patterns from a specific file"""
        patterns = self.introspector.extract_from_file(filepath)

        added = 0
        for p in patterns:
            if p.examples:
                self.knowledge.add(p)
                added += 1

        print(f"Learned {added} patterns from {filepath}")
        return added

    def explore_github(self, topic: Optional[str] = None, duration: int = 30) -> int:
        """Explore GitHub to learn new patterns"""
        if topic:
            print(f"Exploring GitHub for '{topic}'...")
            patterns = self.github.explore_topic(topic)
        else:
            print(f"Freely exploring GitHub for {duration}s...")
            patterns = self.github.free_explore(duration)

        added = 0
        for p in patterns:
            if p.examples:
                self.knowledge.add(p)
                added += 1

        print(f"Learned {added} patterns from GitHub")
        return added

    def generate(self, name: str, signature: str, examples: List[Tuple]) -> str:
        """Generate code from specification"""

        # Try learned patterns first
        code = self.synthesizer.generate(name, signature, examples)
        if code:
            return code

        # Fall back to universal coder
        if self.universal and self.Spec:
            spec = self.Spec(name=name, signature=signature, examples=examples)
            return self.universal.generate(spec)

        # Last resort: lookup table
        return self._generate_lookup(name, signature, examples)

    def _generate_lookup(self, name: str, signature: str, examples: List[Tuple]) -> str:
        """Generate lookup table as last resort"""
        lookup = {repr(i): repr(o) for i, o in examples}
        lookup_str = ', '.join(f'{k}: {v}' for k, v in lookup.items())

        match = re.search(r'\((\w+)', signature)
        param = match.group(1) if match else 'x'

        return f"{signature}:\n    return {{{lookup_str}}}[{param}]"

    def status(self) -> Dict:
        """Get current knowledge status"""
        return self.knowledge.stats()

    def search(self, query: str) -> List[LearnedPattern]:
        """Search learned patterns"""
        return self.knowledge.search(query)

    def teach(self, name: str, signature: str, body: str, examples: List[Tuple]):
        """Manually teach a pattern"""
        pattern = LearnedPattern(
            name=name,
            signature=signature,
            body=body,
            examples=examples,
            source="user"
        )
        self.knowledge.add(pattern)
        print(f"Learned '{name}' from user")

    def forget(self, name: str):
        """Forget a pattern"""
        if name in self.knowledge.patterns:
            del self.knowledge.patterns[name]
            self.knowledge.save()
            print(f"Forgot '{name}'")

    def autonomous_learning(self, duration_minutes: int = 5):
        """
        Autonomous learning mode.

        The system freely explores and learns without direction.
        """
        print("=" * 60)
        print("AUTONOMOUS LEARNING MODE")
        print("=" * 60)
        print(f"Duration: {duration_minutes} minutes")
        print()

        start_time = time.time()
        end_time = start_time + duration_minutes * 60

        # Phase 1: Learn from self
        print("Phase 1: Introspection (learning from self)...")
        self.learn_from_self()
        print()

        # Phase 2: Explore GitHub
        remaining = int(end_time - time.time())
        if remaining > 10:
            print(f"Phase 2: Exploration (GitHub, {remaining}s)...")
            self.explore_github(duration=min(remaining - 5, 120))
            print()

        # Summary
        print("=" * 60)
        print("LEARNING COMPLETE")
        print("=" * 60)
        stats = self.status()
        print(f"Total patterns learned: {stats['total_patterns']}")
        print(f"By source: {stats['by_source']}")
        print()


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demo():
    """Demonstrate the self-learning coder"""
    print("=" * 60)
    print("SELF-LEARNING CODER DEMO")
    print("=" * 60)
    print()

    coder = SelfLearningCoder()

    # Learn from self
    print("Step 1: Learning from own codebase...")
    coder.learn_from_self()
    print()

    # Check status
    print("Step 2: Knowledge status...")
    print(coder.status())
    print()

    # Generate code using learned patterns
    print("Step 3: Generate code...")

    tests = [
        ("square", "def square(x)", [(2, 4), (3, 9), (5, 25)]),
        ("add", "def add(a, b)", [((1, 2), 3), ((5, 3), 8)]),
        ("reverse", "def reverse(s)", [("hello", "olleh"), ("abc", "cba")]),
    ]

    for name, sig, examples in tests:
        code = coder.generate(name, sig, examples)
        print(f"{name}:")
        print(f"  {code.split(chr(10))[1].strip() if chr(10) in code else code}")
        print()

    # Search patterns
    print("Step 4: Search patterns...")
    results = coder.search("add")
    print(f"Found {len(results)} patterns matching 'add'")
    print()

    print("=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    demo()
