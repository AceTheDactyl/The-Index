"""
PATTERN LEARNER
===============

Learns RULES, not examples.

Instead of storing:
    examples = [(1,2), (2,4), (3,6), (4,8), (5,10)]

It learns:
    rule = "multiply by 2"
    code = "return x * 2"

This is:
- More memory efficient
- Generalizes to unseen inputs
- Closer to how humans learn
"""

import re
import ast
import json
import os
import subprocess
import sys
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field


# =============================================================================
# DEPENDENCY INSTALLER - Auto-installs missing packages
# =============================================================================

class DependencyInstaller:
    """Auto-installs missing Python packages"""

    # Map import names to pip package names (when different)
    IMPORT_TO_PIP = {
        'cv2': 'opencv-python',
        'PIL': 'Pillow',
        'sklearn': 'scikit-learn',
        'yaml': 'pyyaml',
        'bs4': 'beautifulsoup4',
        'dotenv': 'python-dotenv',
    }

    # Common packages that are safe to auto-install
    SAFE_PACKAGES = {
        'numpy', 'pandas', 'requests', 'flask', 'django',
        'matplotlib', 'seaborn', 'scipy', 'networkx',
        'opencv-python', 'Pillow', 'scikit-learn',
        'beautifulsoup4', 'lxml', 'pyyaml', 'python-dotenv',
        'pytest', 'black', 'mypy', 'tqdm', 'rich',
        'httpx', 'aiohttp', 'fastapi', 'uvicorn',
        'sqlalchemy', 'redis', 'pymongo', 'psycopg2-binary',
        'cryptography', 'pyjwt', 'boto3', 'google-cloud-storage',
    }

    @classmethod
    def get_pip_name(cls, import_name: str) -> str:
        """Convert import name to pip package name"""
        return cls.IMPORT_TO_PIP.get(import_name, import_name)

    @classmethod
    def is_installed(cls, package: str) -> bool:
        """Check if a package is installed"""
        try:
            __import__(package)
            return True
        except ImportError:
            return False

    @classmethod
    def install(cls, package: str, quiet: bool = True) -> bool:
        """Install a package using pip"""
        pip_name = cls.get_pip_name(package)

        # Safety check
        if pip_name.lower() not in {p.lower() for p in cls.SAFE_PACKAGES}:
            print(f"  Warning: {pip_name} not in safe list, skipping auto-install")
            return False

        try:
            args = [sys.executable, '-m', 'pip', 'install', pip_name]
            if quiet:
                args.append('--quiet')

            print(f"  Installing {pip_name}...")
            result = subprocess.run(args, capture_output=True, text=True, timeout=120)

            if result.returncode == 0:
                print(f"  Installed {pip_name} successfully")
                return True
            else:
                print(f"  Failed to install {pip_name}: {result.stderr[:100]}")
                return False
        except Exception as e:
            print(f"  Error installing {pip_name}: {e}")
            return False

    @classmethod
    def ensure_installed(cls, packages: List[str]) -> List[str]:
        """Ensure packages are installed, return list of failed installs"""
        failed = []
        for pkg in packages:
            if not cls.is_installed(pkg):
                if not cls.install(pkg):
                    failed.append(pkg)
        return failed

    @classmethod
    def detect_from_code(cls, code: str) -> List[str]:
        """Detect external packages needed from code"""
        packages = set()

        # Find import statements
        import_patterns = [
            r'^import\s+(\w+)',
            r'^from\s+(\w+)\s+import',
        ]

        for line in code.split('\n'):
            line = line.strip()
            for pattern in import_patterns:
                match = re.match(pattern, line)
                if match:
                    pkg = match.group(1)
                    # Skip standard library
                    if pkg not in {'os', 'sys', 're', 'json', 'math', 'random',
                                   'datetime', 'time', 'collections', 'itertools',
                                   'functools', 'typing', 'pathlib', 'subprocess',
                                   'threading', 'multiprocessing', 'asyncio',
                                   'unittest', 'logging', 'copy', 'heapq', 'bisect',
                                   'string', 'operator', 'io', 'pickle', 'csv',
                                   'hashlib', 'base64', 'urllib', 'http', 'socket',
                                   'ast', 'inspect', 'abc', 'contextlib', 'dataclasses'}:
                        packages.add(pkg)

        return list(packages)


@dataclass
class Rule:
    """A learned rule (not examples)"""
    name: str
    pattern_type: str  # "arithmetic", "string", "list", "conditional", "loop"
    code_template: str  # The actual code
    description: str   # Human readable
    confidence: float = 1.0
    times_verified: int = 0
    required_imports: List[str] = field(default_factory=list)
    helper_functions: List[str] = field(default_factory=list)

    def generate(self, params: List[str]) -> str:
        """Generate code from this rule"""
        code = self.code_template
        # Replace placeholder params
        for i, p in enumerate(params):
            code = code.replace(f"$P{i}", p)
            code = code.replace(f"$p{i}", p)
        # Default: replace $P with first param
        if params:
            code = code.replace("$P", params[0])
            code = code.replace("$p", params[0])
        return code


# =============================================================================
# IMPORT DETECTOR - Automatically detects required imports
# =============================================================================

class ImportDetector:
    """Detects required imports from code"""

    # Common module -> functions mapping
    KNOWN_IMPORTS = {
        'math': ['sqrt', 'ceil', 'floor', 'log', 'log2', 'log10', 'exp', 'sin', 'cos',
                 'tan', 'pi', 'e', 'gcd', 'factorial', 'pow', 'fabs', 'inf'],
        'functools': ['reduce', 'lru_cache', 'partial', 'wraps'],
        'itertools': ['permutations', 'combinations', 'product', 'chain', 'groupby',
                      'accumulate', 'count', 'cycle', 'repeat'],
        'collections': ['Counter', 'defaultdict', 'deque', 'OrderedDict', 'namedtuple'],
        'heapq': ['heappush', 'heappop', 'heapify', 'heapreplace', 'nlargest', 'nsmallest'],
        'random': ['random', 'randint', 'choice', 'shuffle', 'sample', 'uniform'],
        'typing': ['List', 'Dict', 'Set', 'Tuple', 'Optional', 'Union', 'Any', 'Callable'],
        're': ['match', 'search', 'findall', 'sub', 'split', 'compile'],
        'json': ['dumps', 'loads', 'dump', 'load'],
        'os': ['path', 'listdir', 'makedirs', 'remove', 'rename', 'getcwd'],
        'sys': ['argv', 'exit', 'stdin', 'stdout', 'stderr'],
        'copy': ['copy', 'deepcopy'],
        'bisect': ['bisect_left', 'bisect_right', 'insort'],
        'operator': ['itemgetter', 'attrgetter', 'add', 'mul'],
        'string': ['ascii_lowercase', 'ascii_uppercase', 'digits'],
    }

    # Reverse mapping: function -> module
    FUNC_TO_MODULE = {}
    for module, funcs in KNOWN_IMPORTS.items():
        for func in funcs:
            FUNC_TO_MODULE[func] = module

    # Common constants that need imports
    CONSTANTS = {
        'MOD': '10**9 + 7',
        'INF': 'float("inf")',
        'NINF': 'float("-inf")',
    }

    @classmethod
    def detect(cls, code: str) -> List[str]:
        """Detect required imports from code"""
        imports = set()

        # Check for known functions
        for func, module in cls.FUNC_TO_MODULE.items():
            # Match function calls like func( or references like math.func
            if re.search(rf'\b{func}\s*\(', code):
                if module == 'math':
                    imports.add(f'import {module}')
                else:
                    imports.add(f'from {module} import {func}')

        # Check for module.function patterns
        for module in cls.KNOWN_IMPORTS:
            if re.search(rf'\b{module}\.\w+', code):
                imports.add(f'import {module}')

        return sorted(list(imports))

    @classmethod
    def detect_constants(cls, code: str) -> Dict[str, str]:
        """Detect undefined constants that need definitions"""
        needed = {}
        for const, value in cls.CONSTANTS.items():
            if re.search(rf'\b{const}\b', code):
                needed[const] = value
        return needed


# =============================================================================
# HELPER BUNDLER - Bundles related helper functions
# =============================================================================

class HelperBundler:
    """Bundles helper functions with main functions"""

    # Known function dependencies
    DEPENDENCIES = {
        'merge_sort': ['merge'],
        'heap_sort': ['heapify'],
        'quick_sort': ['partition'],
        'dfs': ['visited'],
        'bfs': ['visited'],
        'dijkstra': ['heapq'],
        'kruskal': ['find', 'union'],
    }

    def __init__(self, learner: 'PatternLearner'):
        self.learner = learner

    def get_helpers_for_code(self, code: str) -> List[str]:
        """
        Get helper functions that are actually REFERENCED in the code.
        Only adds helpers if the code calls them.
        """
        helpers = []

        # Check what functions are called in the code
        # Look for function calls like "helper_name("
        called_funcs = set(re.findall(r'\b([a-z_][a-z0-9_]*)\s*\(', code, re.IGNORECASE))

        for func_name in called_funcs:
            # Skip common builtins and the main function itself
            if func_name in {'len', 'range', 'print', 'sorted', 'list', 'set', 'dict',
                             'min', 'max', 'sum', 'abs', 'int', 'str', 'float', 'bool',
                             'enumerate', 'zip', 'map', 'filter', 'any', 'all', 'isinstance',
                             'if', 'for', 'while', 'return', 'def', 'class'}:
                continue

            # Check if this function is defined elsewhere in the code
            if re.search(rf'\bdef\s+{func_name}\s*\(', code):
                continue

            # Look for a helper in learned rules
            helper_code = self._find_and_generate_helper(func_name)
            if helper_code:
                helpers.append(helper_code)

        return helpers

    def get_helpers(self, func_name: str) -> List[str]:
        """Get helper function code for a given function (deprecated, use get_helpers_for_code)"""
        # Keep for backwards compatibility but return empty - let get_helpers_for_code do the work
        return []

    def _find_and_generate_helper(self, helper_name: str) -> Optional[str]:
        """Find helper function in learned rules and generate valid code"""
        best_rule = None
        best_score = -1

        for rule_name, rule in self.learner.rules.items():
            # Check if rule name matches helper
            rule_name_lower = rule_name.lower()
            core_name = re.sub(r'^stack_', '', rule_name_lower)
            core_name = re.sub(r'_\d+$', '', core_name)

            if core_name == helper_name.lower():
                # Prefer rules with fewer params (simpler implementations)
                param_match = re.search(r'\((\d+)\s*params?\)', rule.description)
                params = int(param_match.group(1)) if param_match else 0
                score = 100 - params - len(rule.code_template) / 100

                if score > best_score:
                    best_score = score
                    best_rule = rule

        if best_rule:
            # Generate helper with generic parameter names
            body = best_rule.code_template

            # Replace placeholders with generic arg names
            body = re.sub(r'\$P0', 'args[0]', body)
            body = re.sub(r'\$P1', 'args[1]', body)
            body = re.sub(r'\$P2', 'args[2]', body)
            body = re.sub(r'\$P', 'args[0]', body)

            # Check if the body still has unresolved placeholders
            if '$P' in body or '$p' in body:
                return None  # Can't generate valid helper

            return f"def {helper_name}(*args):\n    " + body.replace('\n', '\n    ')

        return None


# =============================================================================
# FAILURE ANALYZER - Learns from execution failures
# =============================================================================

@dataclass
class FailureRecord:
    """Record of a code generation failure"""
    func_name: str
    error_type: str
    error_message: str
    missing_import: Optional[str] = None
    missing_helper: Optional[str] = None
    missing_constant: Optional[str] = None


class FailureAnalyzer:
    """Analyzes failures and learns fixes"""

    def __init__(self):
        self.failures: List[FailureRecord] = []
        self.fixes: Dict[str, str] = {}  # error pattern -> fix

    def analyze(self, func_name: str, code: str, error: Exception) -> FailureRecord:
        """Analyze a failure and suggest fixes"""
        error_str = str(error)
        error_type = type(error).__name__

        record = FailureRecord(
            func_name=func_name,
            error_type=error_type,
            error_message=error_str
        )

        # Analyze NameError - missing name
        if isinstance(error, NameError):
            match = re.search(r"name '(\w+)' is not defined", error_str)
            if match:
                missing_name = match.group(1)

                # Check if it's a known import
                if missing_name in ImportDetector.FUNC_TO_MODULE:
                    record.missing_import = ImportDetector.FUNC_TO_MODULE[missing_name]

                # Check if it's a constant
                elif missing_name in ImportDetector.CONSTANTS:
                    record.missing_constant = missing_name

                # Otherwise it's likely a helper function
                else:
                    record.missing_helper = missing_name

        self.failures.append(record)
        return record

    def suggest_fix(self, record: FailureRecord) -> str:
        """Suggest a fix for a failure"""
        fixes = []

        if record.missing_import:
            module = record.missing_import
            if module in ImportDetector.KNOWN_IMPORTS:
                fixes.append(f"import {module}")
            else:
                fixes.append(f"from {module} import *")

        if record.missing_constant:
            const = record.missing_constant
            if const in ImportDetector.CONSTANTS:
                fixes.append(f"{const} = {ImportDetector.CONSTANTS[const]}")

        if record.missing_helper:
            fixes.append(f"# Need to define helper: {record.missing_helper}()")

        return '\n'.join(fixes)

    def get_failure_stats(self) -> Dict:
        """Get statistics on failures"""
        stats = {
            'total': len(self.failures),
            'by_type': {},
            'missing_imports': [],
            'missing_helpers': [],
        }

        for f in self.failures:
            stats['by_type'][f.error_type] = stats['by_type'].get(f.error_type, 0) + 1
            if f.missing_import:
                stats['missing_imports'].append(f.missing_import)
            if f.missing_helper:
                stats['missing_helpers'].append(f.missing_helper)

        return stats


class RuleInducer:
    """
    Induces RULES from examples.

    Given examples, figures out the underlying pattern
    and stores only the rule, not the examples.
    """

    def induce(self, examples: List[Tuple]) -> Optional[Rule]:
        """Induce a rule from examples"""
        if not examples:
            return None

        inp0, out0 = examples[0]

        # Try different inducers based on types
        if isinstance(inp0, (int, float)) and isinstance(out0, (int, float)):
            return self._induce_numeric(examples)

        if isinstance(inp0, tuple) and isinstance(out0, (int, float)):
            return self._induce_multi_numeric(examples)

        if isinstance(inp0, str):
            return self._induce_string(examples)

        if isinstance(inp0, (list, tuple)):
            return self._induce_list(examples)

        return None

    def _induce_numeric(self, examples: List[Tuple]) -> Optional[Rule]:
        """Induce numeric transformation rule"""

        # Check for multiplication
        non_zero = [(i, o) for i, o in examples if i != 0]
        if non_zero:
            ratios = [o / i for i, o in non_zero]
            if all(abs(r - ratios[0]) < 1e-9 for r in ratios):
                factor = ratios[0]
                if factor == int(factor):
                    factor = int(factor)
                return Rule(
                    name="multiply",
                    pattern_type="arithmetic",
                    code_template=f"return $P * {factor}",
                    description=f"multiply by {factor}"
                )

        # Check for addition
        diffs = [o - i for i, o in examples]
        if all(abs(d - diffs[0]) < 1e-9 for d in diffs):
            offset = diffs[0]
            if offset == int(offset):
                offset = int(offset)
            if offset > 0:
                return Rule(
                    name="add",
                    pattern_type="arithmetic",
                    code_template=f"return $P + {offset}",
                    description=f"add {offset}"
                )
            elif offset < 0:
                return Rule(
                    name="subtract",
                    pattern_type="arithmetic",
                    code_template=f"return $P - {-offset}",
                    description=f"subtract {-offset}"
                )

        # Check for power
        for power in [2, 3]:
            if all(abs(o - i**power) < 1e-9 for i, o in examples):
                return Rule(
                    name=f"power_{power}",
                    pattern_type="arithmetic",
                    code_template=f"return $P ** {power}",
                    description=f"raise to power {power}"
                )

        # Check for absolute value
        if all(abs(o - abs(i)) < 1e-9 for i, o in examples):
            return Rule(
                name="absolute",
                pattern_type="arithmetic",
                code_template="return abs($P)",
                description="absolute value"
            )

        # Check for factorial
        def fact(n):
            if n <= 1: return 1
            r = 1
            for i in range(2, n+1): r *= i
            return r

        if all(isinstance(i, int) and i >= 0 and i <= 12 and abs(o - fact(i)) < 1e-9 for i, o in examples):
            return Rule(
                name="factorial",
                pattern_type="loop",
                code_template="""if $P <= 1:
    return 1
result = 1
for i in range(2, $P + 1):
    result *= i
return result""",
                description="factorial"
            )

        # Check for fibonacci
        def fib(n):
            if n <= 1: return n
            a, b = 0, 1
            for _ in range(2, n+1): a, b = b, a+b
            return b

        if all(isinstance(i, int) and i >= 0 and abs(o - fib(i)) < 1e-9 for i, o in examples):
            return Rule(
                name="fibonacci",
                pattern_type="loop",
                code_template="""if $P <= 1:
    return $P
a, b = 0, 1
for _ in range(2, $P + 1):
    a, b = b, a + b
return b""",
                description="fibonacci"
            )

        # Check for is_even
        if all(isinstance(o, bool) for _, o in examples):
            if all((i % 2 == 0) == o for i, o in examples):
                return Rule(
                    name="is_even",
                    pattern_type="arithmetic",
                    code_template="return $P % 2 == 0",
                    description="check if even"
                )
            if all((i % 2 == 1) == o for i, o in examples):
                return Rule(
                    name="is_odd",
                    pattern_type="arithmetic",
                    code_template="return $P % 2 == 1",
                    description="check if odd"
                )

        # Check for is_prime
        def is_prime(n):
            if n < 2: return False
            for i in range(2, int(n**0.5)+1):
                if n % i == 0: return False
            return True

        if all(isinstance(o, bool) and is_prime(i) == o for i, o in examples):
            return Rule(
                name="is_prime",
                pattern_type="loop",
                code_template="""if $P < 2:
    return False
for i in range(2, int($P ** 0.5) + 1):
    if $P % i == 0:
        return False
return True""",
                description="check if prime"
            )

        return None

    def _induce_multi_numeric(self, examples: List[Tuple]) -> Optional[Rule]:
        """Induce multi-parameter numeric rule"""

        # Try common operations
        ops = [
            ("add", "$P0 + $P1", lambda a, b: a + b),
            ("subtract", "$P0 - $P1", lambda a, b: a - b),
            ("multiply", "$P0 * $P1", lambda a, b: a * b),
            ("max", "max($P0, $P1)", lambda a, b: max(a, b)),
            ("min", "min($P0, $P1)", lambda a, b: min(a, b)),
            ("power", "$P0 ** $P1", lambda a, b: a ** b),
        ]

        for name, template, func in ops:
            try:
                if all(abs(func(i[0], i[1]) - o) < 1e-9 for i, o in examples):
                    return Rule(
                        name=name,
                        pattern_type="arithmetic",
                        code_template=f"return {template}",
                        description=f"{name} two numbers"
                    )
            except:
                pass

        # Modulo (need to avoid zero)
        try:
            if all(i[1] != 0 and i[0] % i[1] == o for i, o in examples):
                return Rule(
                    name="modulo",
                    pattern_type="arithmetic",
                    code_template="return $P0 % $P1",
                    description="modulo"
                )
        except:
            pass

        return None

    def _induce_string(self, examples: List[Tuple]) -> Optional[Rule]:
        """Induce string transformation rule"""

        inp0, out0 = examples[0]

        # String -> String
        if isinstance(out0, str):
            ops = [
                ("reverse", "$P[::-1]", lambda s: s[::-1]),
                ("upper", "$P.upper()", lambda s: s.upper()),
                ("lower", "$P.lower()", lambda s: s.lower()),
                ("strip", "$P.strip()", lambda s: s.strip()),
                ("title", "$P.title()", lambda s: s.title()),
            ]

            for name, template, func in ops:
                if all(func(i) == o for i, o in examples):
                    return Rule(
                        name=name,
                        pattern_type="string",
                        code_template=f"return {template}",
                        description=f"string {name}"
                    )

            # Palindrome check
            if all(isinstance(o, bool) and (i == i[::-1]) == o for i, o in examples):
                return Rule(
                    name="is_palindrome",
                    pattern_type="string",
                    code_template="return $P == $P[::-1]",
                    description="check palindrome"
                )

        # String -> Int
        if isinstance(out0, int):
            if all(len(i) == o for i, o in examples):
                return Rule(
                    name="length",
                    pattern_type="string",
                    code_template="return len($P)",
                    description="string length"
                )

        return None

    def _induce_list(self, examples: List[Tuple]) -> Optional[Rule]:
        """Induce list transformation rule"""

        inp0, out0 = examples[0]

        # List -> Number
        if isinstance(out0, (int, float)):
            ops = [
                ("sum", "sum($P)", lambda x: sum(x)),
                ("len", "len($P)", lambda x: len(x)),
                ("max", "max($P) if $P else 0", lambda x: max(x) if x else 0),
                ("min", "min($P) if $P else 0", lambda x: min(x) if x else 0),
            ]

            for name, template, func in ops:
                try:
                    if all(func(i) == o for i, o in examples):
                        return Rule(
                            name=f"list_{name}",
                            pattern_type="list",
                            code_template=f"return {template}",
                            description=f"list {name}"
                        )
                except:
                    pass

        # List -> List
        if isinstance(out0, (list, tuple)):
            ops = [
                ("reverse", "$P[::-1]", lambda x: x[::-1]),
                ("sort", "sorted($P)", lambda x: sorted(x)),
                ("unique", "list(set($P))", lambda x: sorted(set(x))),
            ]

            for name, template, func in ops:
                try:
                    if all(list(func(i)) == list(o) for i, o in examples):
                        return Rule(
                            name=f"list_{name}",
                            pattern_type="list",
                            code_template=f"return {template}",
                            description=f"list {name}"
                        )
                except:
                    pass

        return None


class PatternLearner:
    """
    Main learner - stores RULES not examples.

    Memory efficient: Only stores the pattern, not training data.
    """

    def __init__(self, rules_path: str = "rules.json"):
        self.rules_path = rules_path
        self.rules: Dict[str, Rule] = {}
        self.inducer = RuleInducer()
        self.load()

    def load(self):
        """Load rules from disk"""
        if os.path.exists(self.rules_path):
            try:
                with open(self.rules_path, 'r') as f:
                    data = json.load(f)
                for name, r in data.items():
                    self.rules[name] = Rule(
                        name=r['name'],
                        pattern_type=r['pattern_type'],
                        code_template=r['code_template'],
                        description=r['description'],
                        confidence=r.get('confidence', 1.0),
                        times_verified=r.get('times_verified', 0)
                    )
            except:
                pass

    def save(self):
        """Save rules to disk"""
        data = {}
        for name, r in self.rules.items():
            data[name] = {
                'name': r.name,
                'pattern_type': r.pattern_type,
                'code_template': r.code_template,
                'description': r.description,
                'confidence': r.confidence,
                'times_verified': r.times_verified
            }
        with open(self.rules_path, 'w') as f:
            json.dump(data, f, indent=2)

    def learn(self, name: str, examples: List[Tuple]) -> Optional[Rule]:
        """
        Learn a rule from examples.

        The examples are used to INDUCE the rule,
        then discarded. Only the rule is stored.
        """
        rule = self.inducer.induce(examples)

        if rule:
            rule.name = name  # Use provided name
            self.rules[name] = rule
            self.save()
            return rule

        return None

    def _score_rule_match(self, query_name: str, query_params: int, rule_name: str, rule: Rule) -> float:
        """
        Score how well a rule matches the query.
        Higher score = better match.

        Scoring factors:
        1. Exact name match (highest priority)
        2. Parameter count match
        3. Name similarity (fuzzy)
        """
        score = 0.0
        query_lower = query_name.lower()
        rule_name_lower = rule_name.lower()

        # Extract the core function name from rule (strip prefix/suffix numbers)
        # e.g., "stack_quick_sort_109" -> "quick_sort"
        core_name = re.sub(r'^stack_', '', rule_name_lower)
        core_name = re.sub(r'_\d+$', '', core_name)

        # Exact name match (highest priority)
        if query_lower == core_name:
            score += 1000
        elif query_lower == rule_name_lower:
            score += 1000

        # Check if query is in description
        desc_lower = rule.description.lower()
        if query_lower in desc_lower:
            score += 100

        # Partial name match
        if query_lower in core_name or core_name in query_lower:
            score += 50

        # Extract param count from rule description (e.g., "quick_sort (1 params)")
        param_match = re.search(r'\((\d+)\s*params?\)', rule.description)
        rule_params = int(param_match.group(1)) if param_match else -1

        # Strong bonus for matching parameter count
        if rule_params == query_params:
            score += 200
        elif rule_params == -1:
            pass  # Unknown, no penalty
        else:
            # Penalty for mismatched params (bigger mismatch = bigger penalty)
            score -= abs(rule_params - query_params) * 50

        # Prefer simpler implementations (shorter code often more canonical)
        code_len = len(rule.code_template)
        if code_len < 500:
            score += 10
        elif code_len > 2000:
            score -= 10

        return score

    def generate(self, name: str, signature: str, examples: List[Tuple] = None) -> str:
        """Generate code from learned rules or new examples"""

        # Extract params from signature
        match = re.search(r'\(([^)]*)\)', signature)
        params = []
        if match:
            for p in match.group(1).split(','):
                p = p.strip().split(':')[0].split('=')[0].strip()
                if p:
                    params.append(p)
        if not params:
            params = ['x']

        num_params = len(params)

        # Check if we have a rule for this (exact match first)
        if name in self.rules:
            rule = self.rules[name]
            body = rule.generate(params)
            return self._build_func(signature, body)

        # Smart fuzzy match - score all matching rules
        name_lower = name.lower()
        candidates = []

        for rule_name, rule in self.rules.items():
            rule_name_lower = rule_name.lower()
            desc_lower = rule.description.lower()

            # Check if this rule might be relevant
            if (name_lower in rule_name_lower or
                name_lower in desc_lower or
                rule_name_lower.replace('stack_', '').split('_')[0] == name_lower.split('_')[0]):

                score = self._score_rule_match(name, num_params, rule_name, rule)
                if score > 0:
                    candidates.append((score, rule_name, rule))

        if candidates:
            # Sort by score (highest first)
            candidates.sort(key=lambda x: -x[0])
            best_score, best_name, best_rule = candidates[0]
            body = best_rule.generate(params)
            return self._build_func(signature, body)

        # Try to learn from examples
        if examples:
            rule = self.learn(name, examples)
            if rule:
                body = rule.generate(params)
                return self._build_func(signature, body)

        # Fallback
        return self._build_func(signature, f"pass  # Could not learn rule")

    def generate_complete(self, name: str, signature: str, examples: List[Tuple] = None) -> str:
        """
        Generate COMPLETE code with imports, helpers, and constants.

        This is the improved version that:
        1. Detects and adds required imports
        2. Bundles helper functions
        3. Defines needed constants
        """
        # Generate base code
        base_code = self.generate(name, signature, examples)

        if 'pass  # Could not' in base_code:
            return base_code

        # Extract just the body for analysis
        body = '\n'.join(base_code.split('\n')[1:])

        # Detect required imports
        imports = ImportDetector.detect(body)

        # Detect required constants
        constants = ImportDetector.detect_constants(body)

        # Find helper functions that are actually used in the code
        bundler = HelperBundler(self)
        helpers = bundler.get_helpers_for_code(base_code)

        # Build complete code
        parts = []

        # Imports first
        if imports:
            parts.extend(imports)
            parts.append('')

        # Constants
        for const_name, const_value in constants.items():
            parts.append(f'{const_name} = {const_value}')
        if constants:
            parts.append('')

        # Helper functions
        for helper in helpers:
            parts.append(helper)
            parts.append('')

        # Main function
        parts.append(base_code)

        return '\n'.join(parts)

    def generate_and_test(self, name: str, signature: str, test_func: Callable = None,
                          examples: List[Tuple] = None, max_retries: int = 3,
                          auto_install: bool = True) -> Tuple[str, bool]:
        """
        Generate code and test it, with automatic failure analysis and retry.

        Args:
            auto_install: If True, auto-install missing packages

        Returns: (code, success)
        """
        analyzer = FailureAnalyzer()

        for attempt in range(max_retries):
            # Generate complete code
            code = self.generate_complete(name, signature, examples)

            if 'pass  # Could not' in code:
                return code, False

            # Check for missing packages and auto-install
            if auto_install:
                packages = DependencyInstaller.detect_from_code(code)
                if packages:
                    failed = DependencyInstaller.ensure_installed(packages)
                    if failed:
                        print(f"  Could not install: {failed}")

            # Try to execute
            try:
                local_ns = {}
                exec(code, local_ns)
                func = local_ns.get(name)

                # If test function provided, run it
                if test_func and func:
                    if test_func(func):
                        return code, True
                    else:
                        return code, False
                else:
                    # No test, just check it compiles
                    return code, True

            except Exception as e:
                # Analyze the failure
                record = analyzer.analyze(name, code, e)

                # Try to fix
                if record.missing_import:
                    # Add the missing import
                    module = record.missing_import
                    if module in ImportDetector.KNOWN_IMPORTS:
                        fix = f"import {module}\n"
                    else:
                        fix = f"from {module} import *\n"
                    code = fix + code

                elif record.missing_constant:
                    const = record.missing_constant
                    fix = f"{const} = {ImportDetector.CONSTANTS.get(const, '0')}\n"
                    code = fix + code

                elif record.missing_helper:
                    # Try to find and add helper
                    helper_name = record.missing_helper
                    for rule_name, rule in self.rules.items():
                        if helper_name.lower() in rule_name.lower():
                            helper_code = f"def {helper_name}(left, right):\n"
                            helper_code += "    " + rule.code_template.replace('\n', '\n    ')
                            code = helper_code + '\n\n' + code
                            break

                # Retry with fixed code
                try:
                    local_ns = {}
                    exec(code, local_ns)
                    func = local_ns.get(name)
                    if test_func and func:
                        if test_func(func):
                            return code, True
                    else:
                        return code, True
                except:
                    pass  # Continue to next attempt

        return code, False

    def _build_func(self, signature: str, body: str) -> str:
        """Build function from signature and body"""
        sig = signature.strip()
        if not sig.startswith('def '):
            sig = f'def {sig}'
        if not sig.endswith(':'):
            sig = f'{sig}:'

        # Clean type hints
        sig = re.sub(r':\s*\w+(\[[\w\[\], ]+\])?', '', sig)
        sig = re.sub(r'\s*->\s*\w+(\[[\w\[\], ]+\])?', '', sig)

        indent_body = '\n'.join('    ' + line for line in body.split('\n'))
        return f"{sig}\n{indent_body}"

    def stats(self) -> Dict:
        """Get statistics"""
        by_type = {}
        for r in self.rules.values():
            by_type[r.pattern_type] = by_type.get(r.pattern_type, 0) + 1

        return {
            'total_rules': len(self.rules),
            'by_type': by_type,
            'memory': f"{len(json.dumps({n: r.__dict__ for n, r in self.rules.items()}))} bytes"
        }

    def list_rules(self) -> List[str]:
        """List all learned rules"""
        return [f"{r.name}: {r.description}" for r in self.rules.values()]

    def forget(self, name: str):
        """Forget a rule"""
        if name in self.rules:
            del self.rules[name]
            self.save()


# =============================================================================
# PATTERN EVOLVER - Self-improvement through experimentation
# =============================================================================

@dataclass
class TestCase:
    """A test case for validating patterns"""
    inputs: Tuple
    expected_output: Any
    description: str = ""


class PatternEvolver:
    """
    Evolves patterns through experimentation.

    Takes existing patterns, creates variations, tests them,
    and keeps the ones that work better.

    Strategies:
    1. COMBINE: Merge parts of two similar patterns
    2. SIMPLIFY: Remove unnecessary code
    3. OPTIMIZE: Try common optimizations
    4. GENERALIZE: Make patterns more flexible
    """

    def __init__(self, learner: PatternLearner):
        self.learner = learner
        self.test_cases: Dict[str, List[TestCase]] = {}
        self.evolved_count = 0

        # Built-in test cases for common algorithms
        self._init_test_cases()

    def _init_test_cases(self):
        """Initialize test cases for common algorithms"""
        self.test_cases = {
            'quick_sort': [
                TestCase(([5,2,8,1,9],), [1,2,5,8,9], "basic"),
                TestCase(([],), [], "empty"),
                TestCase(([1],), [1], "single"),
                TestCase(([3,3,3],), [3,3,3], "duplicates"),
                TestCase(([-1,5,-3,2],), [-3,-1,2,5], "negatives"),
            ],
            'merge_sort': [
                TestCase(([5,2,8,1,9],), [1,2,5,8,9], "basic"),
                TestCase(([],), [], "empty"),
                TestCase(([1],), [1], "single"),
            ],
            'bubble_sort': [
                TestCase(([5,2,8,1,9],), [1,2,5,8,9], "basic"),
                TestCase(([],), [], "empty"),
            ],
            'fibonacci': [
                TestCase((0,), 0, "zero"),
                TestCase((1,), 1, "one"),
                TestCase((10,), 55, "ten"),
                TestCase((20,), 6765, "twenty"),
            ],
            'factorial': [
                TestCase((0,), 1, "zero"),
                TestCase((1,), 1, "one"),
                TestCase((5,), 120, "five"),
                TestCase((10,), 3628800, "ten"),
            ],
            'gcd': [
                TestCase((48, 18), 6, "basic"),
                TestCase((17, 13), 1, "coprime"),
                TestCase((100, 25), 25, "divisible"),
            ],
            'is_prime': [
                TestCase((2,), True, "two"),
                TestCase((17,), True, "prime"),
                TestCase((18,), False, "composite"),
                TestCase((1,), False, "one"),
            ],
            'binary_search': [
                TestCase(([1,2,3,4,5], 3), 2, "found"),
                TestCase(([1,2,3,4,5], 6), -1, "not found"),
            ],
            'is_palindrome': [
                TestCase(("racecar",), True, "palindrome"),
                TestCase(("hello",), False, "not palindrome"),
                TestCase(("",), True, "empty"),
            ],
        }

    def add_test_case(self, func_name: str, inputs: Tuple, expected: Any, desc: str = ""):
        """Add a test case for a function"""
        if func_name not in self.test_cases:
            self.test_cases[func_name] = []
        self.test_cases[func_name].append(TestCase(inputs, expected, desc))

    def _test_pattern(self, func_name: str, code: str) -> Tuple[int, int, float]:
        """
        Test a pattern against test cases.
        Returns: (passed, total, execution_time)
        """
        if func_name not in self.test_cases:
            return 0, 0, 0.0

        import time
        passed = 0
        total = len(self.test_cases[func_name])
        total_time = 0.0

        try:
            exec_globals = {}
            exec(code, exec_globals)
            func = exec_globals.get(func_name)

            if not func:
                return 0, total, 0.0

            for tc in self.test_cases[func_name]:
                try:
                    start = time.perf_counter()
                    result = func(*tc.inputs)
                    elapsed = time.perf_counter() - start
                    total_time += elapsed

                    if result == tc.expected_output:
                        passed += 1
                except:
                    pass

        except:
            return 0, total, 0.0

        return passed, total, total_time

    def _find_similar_patterns(self, func_name: str) -> List[Tuple[str, Rule]]:
        """Find all patterns similar to a function name"""
        similar = []
        func_lower = func_name.lower()

        for rule_name, rule in self.learner.rules.items():
            core_name = re.sub(r'^stack_', '', rule_name.lower())
            core_name = re.sub(r'_\d+$', '', core_name)

            if func_lower == core_name or func_lower in core_name:
                similar.append((rule_name, rule))

        return similar

    def _mutate_simplify(self, code: str) -> List[str]:
        """Generate simplified variations"""
        mutations = []

        # Try removing comments
        no_comments = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        if no_comments.strip() != code.strip():
            mutations.append(no_comments)

        # Try removing type hints
        no_hints = re.sub(r':\s*\w+(\[[\w\[\], ]+\])?\s*(?=[,\)])', '', code)
        if no_hints != code:
            mutations.append(no_hints)

        # Try simplifying conditionals
        simplified = code.replace('if x == True:', 'if x:')
        simplified = simplified.replace('if x == False:', 'if not x:')
        simplified = simplified.replace('== None', 'is None')
        simplified = simplified.replace('!= None', 'is not None')
        if simplified != code:
            mutations.append(simplified)

        # Remove empty lines
        no_empty = re.sub(r'\n\s*\n', '\n', code)
        if no_empty != code:
            mutations.append(no_empty)

        return mutations

    def _mutate_optimize(self, code: str) -> List[str]:
        """Generate optimized variations"""
        mutations = []

        # Early return optimization
        if 'else:\n        return' in code:
            optimized = re.sub(
                r'if\s+(.+?):\s*\n\s+return\s+(.+?)\s*\n\s*else:\s*\n\s+return\s+(.+)',
                r'if \1:\n        return \2\n    return \3',
                code
            )
            if optimized != code:
                mutations.append(optimized)

        # Try converting recursion to iteration for simple cases
        # (This is complex, but we can try for base patterns)

        return mutations

    def _mutate_ast(self, code: str) -> List[str]:
        """AST-based mutations - smarter code transformations"""
        mutations = []

        try:
            tree = ast.parse(code)
        except:
            return mutations

        # Find function body
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Try different loop structures
                for i, stmt in enumerate(node.body):
                    # Convert while to for where possible
                    if isinstance(stmt, ast.While):
                        # Check if it's a counting loop
                        pass  # Complex - skip for now

                    # Convert for to list comprehension where possible
                    if isinstance(stmt, ast.For):
                        # Simple append pattern: for x in y: result.append(f(x))
                        if (len(node.body) >= 2 and
                            isinstance(stmt.body[0], ast.Expr) and
                            isinstance(stmt.body[0].value, ast.Call)):
                            # Could be converted to list comp
                            pass

        return mutations

    def _mutate_algorithmic(self, code: str, func_name: str) -> List[str]:
        """Algorithm-specific mutations"""
        mutations = []

        # Sorting algorithm optimizations
        if 'sort' in func_name.lower():
            # Try adding early termination
            if 'bubble' in func_name.lower() and 'swapped' not in code:
                optimized = code.replace(
                    'for i in range',
                    'swapped = True\n    while swapped:\n        swapped = False\n        for i in range'
                )
                if optimized != code:
                    mutations.append(optimized)

        # Recursive function optimizations
        if func_name in code:  # Self-referential (recursive)
            # Try adding memoization
            if '@lru_cache' not in code and 'memo' not in code:
                lines = code.split('\n')
                if lines[0].startswith('def '):
                    memoized = 'from functools import lru_cache\n\n@lru_cache(maxsize=None)\n' + code
                    mutations.append(memoized)

        # Fix "return list" patterns - return last element instead
        # e.g., return fact -> return fact[n] or return fact[-1]
        if 'return ' in code:
            # Try returning indexed version
            match = re.search(r'return\s+(\w+)\s*$', code, re.MULTILINE)
            if match:
                var_name = match.group(1)
                # Check if there's an array being built
                if f'{var_name} = [' in code or f'{var_name}=[' in code:
                    # Try returning the last element
                    fixed1 = code.replace(f'return {var_name}', f'return {var_name}[-1]')
                    mutations.append(fixed1)

                    # Try returning indexed by input param (common for factorial/fib)
                    # Find first param name
                    param_match = re.search(r'def\s+\w+\((\w+)', code)
                    if param_match:
                        param = param_match.group(1)
                        fixed2 = code.replace(f'return {var_name}', f'return {var_name}[{param}]')
                        mutations.append(fixed2)

        return mutations

    def _synthesize(self, patterns: List[Tuple[str, Rule]], func_name: str, params: List[str]) -> List[str]:
        """
        Synthesize new code by combining fragments from multiple patterns.
        This is more sophisticated than simple crossover.
        """
        mutations = []

        if len(patterns) < 2:
            return mutations

        # Extract code blocks from each pattern
        blocks = []
        for rule_name, rule in patterns[:5]:
            code = rule.code_template
            # Split into logical blocks (by empty lines or control structures)
            parts = re.split(r'\n(?=\s*(?:if|for|while|return|def))', code)
            blocks.append(parts)

        # Try combining different parts
        import random
        for _ in range(3):  # Generate 3 synthetic variations
            if len(blocks) >= 2:
                # Pick random blocks from different patterns
                b1 = random.choice(blocks)
                b2 = random.choice(blocks)

                if b1 != b2 and len(b1) > 1 and len(b2) > 1:
                    # Take beginning from b1, end from b2
                    mid = len(b1) // 2
                    synthetic = '\n'.join(b1[:mid] + b2[mid:])

                    # Substitute params
                    for i, p in enumerate(params):
                        synthetic = synthetic.replace(f'$P{i}', p)

                    # Build function
                    sig = f"def {func_name}({', '.join(params)}):"
                    full_code = sig + '\n    ' + synthetic.replace('\n', '\n    ')
                    mutations.append(full_code)

        return mutations

    def _crossover(self, code1: str, code2: str) -> List[str]:
        """Combine two patterns - improved version"""
        mutations = []

        lines1 = code1.strip().split('\n')
        lines2 = code2.strip().split('\n')

        if len(lines1) < 2 or len(lines2) < 2:
            return mutations

        # Strategy 1: Keep header from code1, body from code2
        hybrid1 = lines1[0] + '\n' + '\n'.join(lines2[1:])
        mutations.append(hybrid1)

        # Strategy 2: Take first half of code1, second half of code2
        mid1 = len(lines1) // 2
        mid2 = len(lines2) // 2
        if mid1 > 0 and mid2 < len(lines2):
            hybrid2 = '\n'.join(lines1[:mid1] + lines2[mid2:])
            mutations.append(hybrid2)

        # Strategy 3: Interleave (take odd lines from code1, even from code2)
        max_len = max(len(lines1), len(lines2))
        interleaved = []
        for i in range(max_len):
            if i % 2 == 0 and i < len(lines1):
                interleaved.append(lines1[i])
            elif i < len(lines2):
                interleaved.append(lines2[i])
        if len(interleaved) > 1:
            mutations.append('\n'.join(interleaved))

        return mutations

    def _mine_idioms(self, patterns: List[Tuple[str, Rule]]) -> Dict[str, int]:
        """
        Mine common idioms/patterns across multiple implementations.
        Returns idiom -> count mapping.
        """
        idioms = {}

        # Common code patterns to look for
        idiom_patterns = [
            (r'if\s+len\(\w+\)\s*[<=>]=?\s*\d+:', 'length_check'),
            (r'for\s+\w+\s+in\s+range\(len\(\w+\)\)', 'index_loop'),
            (r'for\s+\w+\s+in\s+\w+:', 'foreach_loop'),
            (r'while\s+\w+\s*[<>]=?\s*\w+:', 'while_compare'),
            (r'\w+\s*=\s*\[\]', 'empty_list_init'),
            (r'\w+\.append\(', 'list_append'),
            (r'return\s+\w+\[::-1\]', 'reverse_slice'),
            (r'left,\s*right\s*=', 'two_pointer'),
            (r'pivot\s*=', 'pivot_select'),
            (r'if\s+\w+\s*==\s*\w+:', 'equality_check'),
            (r'sorted\(', 'builtin_sort'),
            (r'\.sort\(', 'inplace_sort'),
        ]

        for rule_name, rule in patterns:
            code = rule.code_template
            for pattern, idiom_name in idiom_patterns:
                if re.search(pattern, code):
                    idioms[idiom_name] = idioms.get(idiom_name, 0) + 1

        return idioms

    def evolve_pattern(self, func_name: str, verbose: bool = True) -> Optional[str]:
        """
        Evolve a pattern for a specific function.

        1. Find all existing patterns for this function
        2. Test them to find the best baseline
        3. Generate mutations and crossovers
        4. Test variations and keep improvements
        """
        if verbose:
            print(f"\nEvolving pattern for: {func_name}")
            print("=" * 50)

        # Find similar patterns
        similar = self._find_similar_patterns(func_name)
        if not similar:
            if verbose:
                print(f"  No existing patterns found for {func_name}")
            return None

        if verbose:
            print(f"  Found {len(similar)} candidate patterns")

        # Test all existing patterns
        best_score = -1
        best_code = None
        best_name = None
        best_time = float('inf')

        # Get signature for generating code
        # Guess param count from most common in similar patterns
        param_counts = {}
        for name, rule in similar:
            match = re.search(r'\((\d+)\s*params?\)', rule.description)
            if match:
                pc = int(match.group(1))
                param_counts[pc] = param_counts.get(pc, 0) + 1

        most_common_params = max(param_counts.keys(), key=lambda k: param_counts[k]) if param_counts else 1

        # Build signature
        param_names = ['arr', 'target', 'n', 'a', 'b', 's', 'x', 'y'][:most_common_params]
        sig = f"def {func_name}({', '.join(param_names)})"

        for rule_name, rule in similar:
            code = self.learner.generate_complete(func_name, sig)
            passed, total, exec_time = self._test_pattern(func_name, code)

            if total > 0:
                score = passed / total
                if score > best_score or (score == best_score and exec_time < best_time):
                    best_score = score
                    best_code = code
                    best_name = rule_name
                    best_time = exec_time

        if verbose:
            if best_code:
                print(f"  Best baseline: {best_name}")
                print(f"  Score: {best_score*100:.0f}% ({int(best_score * len(self.test_cases.get(func_name, [])))} tests passed)")
            else:
                print(f"  No working pattern found")
                return None

        if best_score == 1.0:
            if verbose:
                print(f"  Already perfect! No evolution needed.")
            return best_code

        # Mine idioms from all patterns
        idioms = self._mine_idioms(similar)
        if verbose and idioms:
            top_idioms = sorted(idioms.items(), key=lambda x: -x[1])[:3]
            print(f"  Common idioms: {', '.join(f'{k}({v})' for k,v in top_idioms)}")

        # Generate mutations using ALL strategies
        mutations = []

        # Basic mutations
        mutations.extend(self._mutate_simplify(best_code))
        mutations.extend(self._mutate_optimize(best_code))

        # AST-based mutations
        mutations.extend(self._mutate_ast(best_code))

        # Algorithm-specific mutations
        mutations.extend(self._mutate_algorithmic(best_code, func_name))

        # Synthesis from multiple patterns
        mutations.extend(self._synthesize(similar, func_name, param_names))

        # Crossovers with other patterns
        for rule_name, rule in similar[:5]:  # Limit to top 5
            other_code = self.learner._build_func(sig, rule.generate(param_names))
            mutations.extend(self._crossover(best_code, other_code))

        # Remove duplicates
        mutations = list(set(mutations))

        if verbose:
            print(f"  Generated {len(mutations)} mutations")

        # Test mutations
        improved = False
        for mutation in mutations:
            passed, total, exec_time = self._test_pattern(func_name, mutation)
            if total > 0:
                score = passed / total
                if score > best_score or (score == best_score and exec_time < best_time * 0.9):
                    best_score = score
                    best_code = mutation
                    best_time = exec_time
                    improved = True
                    if verbose:
                        print(f"  Improvement found! Score: {score*100:.0f}%")

        if improved:
            # Save the improved pattern
            self._save_evolved_pattern(func_name, best_code, param_names)
            self.evolved_count += 1

        if verbose:
            print(f"  Final score: {best_score*100:.0f}%")

        return best_code

    def _save_evolved_pattern(self, func_name: str, code: str, params: List[str]):
        """Save an evolved pattern back to the learner"""
        # Extract the body
        lines = code.strip().split('\n')
        if lines[0].startswith('def '):
            body = '\n'.join(lines[1:])
            # Remove leading indentation
            body = re.sub(r'^    ', '', body, flags=re.MULTILINE)
        else:
            body = code

        # Create template with placeholders
        template = body
        for i, param in enumerate(params):
            template = template.replace(param, f'$P{i}')

        # Create rule
        rule = Rule(
            name=f"evolved_{func_name}",
            pattern_type="evolved",
            code_template=template,
            description=f"Evolved {func_name} ({len(params)} params)",
            confidence=1.0
        )

        self.learner.rules[f"evolved_{func_name}"] = rule
        self.learner.save()

    def evolve_all(self, verbose: bool = True) -> Dict[str, float]:
        """Evolve all patterns that have test cases"""
        results = {}

        if verbose:
            print("\n" + "=" * 60)
            print("PATTERN EVOLUTION")
            print("=" * 60)

        for func_name in self.test_cases.keys():
            code = self.evolve_pattern(func_name, verbose=verbose)
            if code:
                passed, total, _ = self._test_pattern(func_name, code)
                results[func_name] = passed / total if total > 0 else 0.0

        if verbose:
            print("\n" + "=" * 60)
            print("EVOLUTION SUMMARY")
            print("=" * 60)
            print(f"Patterns evolved: {self.evolved_count}")
            print(f"Functions tested: {len(results)}")
            avg_score = sum(results.values()) / len(results) if results else 0
            print(f"Average score: {avg_score*100:.0f}%")

        return results

    def tournament(self, func_name: str, generations: int = 5, verbose: bool = True) -> str:
        """
        Run multiple generations of evolution (genetic algorithm style).

        Each generation:
        1. Test all variants
        2. Keep top performers
        3. Generate new mutations from winners
        """
        if verbose:
            print(f"\nTournament for: {func_name}")
            print(f"Generations: {generations}")
            print("=" * 50)

        similar = self._find_similar_patterns(func_name)
        if not similar:
            return None

        # Determine params
        param_counts = {}
        for name, rule in similar:
            match = re.search(r'\((\d+)\s*params?\)', rule.description)
            if match:
                pc = int(match.group(1))
                param_counts[pc] = param_counts.get(pc, 0) + 1

        most_common_params = max(param_counts.keys(), key=lambda k: param_counts[k]) if param_counts else 1
        param_names = ['arr', 'target', 'n', 'a', 'b', 's', 'x', 'y'][:most_common_params]
        sig = f"def {func_name}({', '.join(param_names)})"

        # Initial population
        population = []
        for rule_name, rule in similar[:10]:  # Top 10
            code = self.learner._build_func(sig, rule.generate(param_names))
            population.append(code)

        best_overall = None
        best_score = -1

        for gen in range(generations):
            if verbose:
                print(f"\n  Generation {gen + 1}/{generations}")

            # Score population
            scored = []
            for code in population:
                passed, total, exec_time = self._test_pattern(func_name, code)
                score = passed / total if total > 0 else 0
                scored.append((score, exec_time, code))

            # Sort by score (desc), then by time (asc)
            scored.sort(key=lambda x: (-x[0], x[1]))

            # Track best
            if scored and scored[0][0] > best_score:
                best_score = scored[0][0]
                best_overall = scored[0][2]
                if verbose:
                    print(f"    New best: {best_score*100:.0f}%")

            if best_score == 1.0:
                if verbose:
                    print(f"    Perfect score achieved!")
                break

            # Keep top 50%
            survivors = [code for _, _, code in scored[:len(scored)//2 + 1]]

            # Generate new mutations from survivors
            new_population = survivors.copy()
            for code in survivors:
                new_population.extend(self._mutate_simplify(code))
                new_population.extend(self._mutate_optimize(code))

            # Crossovers
            import random
            for _ in range(min(5, len(survivors))):
                if len(survivors) >= 2:
                    p1, p2 = random.sample(survivors, 2)
                    new_population.extend(self._crossover(p1, p2))

            # Deduplicate and limit population size
            population = list(set(new_population))[:20]

        if best_overall and best_score > 0.5:  # Only save if reasonably good
            self._save_evolved_pattern(func_name, best_overall, param_names)
            self.evolved_count += 1

        if verbose:
            print(f"\n  Final best score: {best_score*100:.0f}%")

        return best_overall


# =============================================================================
# THE STACK EXPLORER (BigCode/Hugging Face)
# =============================================================================

class StackExplorer:
    """
    Learn patterns from The Stack dataset (BigCode).

    The Stack is 6TB+ of permissively licensed code.
    We stream samples on-demand - no large download needed.
    """

    def __init__(self, learner: PatternLearner):
        self.learner = learner
        self.dataset = None

    def load(self, language: str = 'python', streaming: bool = True, dataset_name: str = None):
        """
        Load a code dataset with streaming.

        Args:
            language: Programming language to learn from
            streaming: If True, stream samples (recommended)
            dataset_name: Override dataset name (default tries multiple)
        """
        from datasets import load_dataset

        # Try multiple datasets in order of preference
        datasets_to_try = [
            # Ling-Coder-SFT - ungated, high quality instruction-following code!
            ("inclusionAI/Ling-Coder-SFT", {"streaming": streaming}),
            # python-stack-v1-functions-filtered - ungated, pre-filtered Python functions
            ("bigcode/python-stack-v1-functions-filtered", {"streaming": streaming}),
            # The Stack (requires HF token for gated access)
            ("bigcode/the-stack", {"data_dir": f"data/{language}", "streaming": streaming}),
        ]

        if dataset_name:
            datasets_to_try = [(dataset_name, {"streaming": streaming})]

        for ds_name, kwargs in datasets_to_try:
            try:
                print(f"Trying {ds_name}...")
                self.dataset = load_dataset(
                    ds_name,
                    split="train",
                    **kwargs
                )
                print(f"Loaded {ds_name} (streaming mode)")
                return True
            except Exception as e:
                print(f"  Failed: {str(e)[:60]}...")
                continue

        print("ERROR: Could not load any code dataset")
        print("Options:")
        print("  1. Login to HuggingFace: huggingface-cli login")
        print("  2. Request access to gated datasets")
        return False

    def extract_functions(self, code: str) -> List[Dict]:
        """Extract functions from Python code using AST"""
        functions = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Get function info
                    name = node.name
                    args = [arg.arg for arg in node.args.args]

                    # Get body as string
                    try:
                        body_lines = []
                        for stmt in node.body:
                            body_lines.append(ast.unparse(stmt))
                        body = '\n'.join(body_lines)
                    except:
                        body = None

                    if name and body:  # args can be empty
                        functions.append({
                            'name': name,
                            'args': args,
                            'body': body,
                            'num_lines': len(node.body)
                        })
        except:
            pass
        return functions

    def _extract_code_from_sample(self, sample: Dict) -> str:
        """Extract code from various dataset formats"""
        # Direct content field
        if 'content' in sample:
            return sample['content']

        # Ling-Coder-SFT format: messages with ASSISTANT response containing code
        if 'messages' in sample:
            for msg in sample['messages']:
                if msg.get('role') == 'ASSISTANT':
                    content = msg.get('content', '')
                    # Extract code blocks from markdown
                    code_blocks = re.findall(r'```(?:python)?\n(.*?)```', content, re.DOTALL)
                    if code_blocks:
                        return '\n\n'.join(code_blocks)
                    # If no code blocks, maybe it's raw code
                    if 'def ' in content or 'class ' in content:
                        return content

        # code field (some datasets)
        if 'code' in sample:
            return sample['code']

        return ''

    def learn_from_stream(self, n_samples: int = 100, max_funcs_per_sample: int = 5):
        """
        Learn patterns from streaming samples.

        Args:
            n_samples: Number of code files to sample
            max_funcs_per_sample: Max functions to extract per file
        """
        if not self.dataset:
            print("Dataset not loaded. Call load() first.")
            return

        print(f"Learning from {n_samples} samples...")
        learned = 0
        seen_patterns = set()

        for i, sample in enumerate(self.dataset):
            if i >= n_samples:
                break

            code = self._extract_code_from_sample(sample)
            if not code:
                continue

            functions = self.extract_functions(code)[:max_funcs_per_sample]

            for func in functions:
                # Skip very long or very short functions
                if func['num_lines'] < 1 or func['num_lines'] > 10:
                    continue

                # Skip private/dunder methods
                if func['name'].startswith('_'):
                    continue

                # Create a pattern signature
                pattern_sig = f"{len(func['args'])}_{func['body'][:50]}"
                if pattern_sig in seen_patterns:
                    continue
                seen_patterns.add(pattern_sig)

                # Try to induce a rule from this function
                rule = self._try_induce_from_function(func)
                if rule:
                    # Use descriptive name
                    rule_name = f"stack_{func['name']}_{learned}"
                    rule.name = rule_name
                    self.learner.rules[rule_name] = rule
                    learned += 1
                    print(f"  Learned: {func['name']} -> {rule.description}")

            if (i + 1) % 20 == 0:
                print(f"  Processed {i+1}/{n_samples} samples, learned {learned} rules")

        self.learner.save()
        print(f"Done! Learned {learned} new rules.")
        return learned

    def _try_induce_from_function(self, func: Dict) -> Optional[Rule]:
        """Try to create a rule from a function"""
        body = func['body']
        args = func['args']
        name = func['name']

        # Skip docstring-only functions
        if body.strip().startswith("'") or body.strip().startswith('"'):
            first_line = body.split('\n')[0].strip()
            if first_line.endswith("'") or first_line.endswith('"'):
                return None

        # STRATEGY 1: Store entire function as a template
        # Replace args with placeholders to make it reusable
        template = body
        for i, arg in enumerate(args):
            # Replace argument references with placeholders
            template = re.sub(rf'\b{arg}\b', f'$P{i}', template)

        # Determine pattern type from function name or content
        pattern_type = "general"
        if any(kw in name.lower() for kw in ['sort', 'search', 'find']):
            pattern_type = "algorithm"
        elif any(kw in name.lower() for kw in ['prime', 'fibonacci', 'factorial', 'gcd']):
            pattern_type = "math"
        elif any(kw in name.lower() for kw in ['list', 'array', 'filter', 'map']):
            pattern_type = "list"
        elif any(kw in name.lower() for kw in ['string', 'str', 'text', 'parse']):
            pattern_type = "string"

        # Create rule from the function template
        return Rule(
            name=name,
            pattern_type=pattern_type,
            code_template=template,
            description=f"{name} ({len(args)} params)"
        )

        # STRATEGY 2: Simple return pattern detection (original approach below)
        # Handle zero-arg functions
        if not args:
            if body.startswith('return '):
                expr = body[7:].strip()
                if expr in ['True', 'False', 'None']:
                    return Rule(
                        name="constant",
                        pattern_type="arithmetic",
                        code_template=f"return {expr}",
                        description=f"return {expr}"
                    )
            return None

        # Simple return statement patterns
        if body.startswith('return '):
            expr = body[7:].strip()

            # Detect common patterns
            if len(args) == 1:
                arg = args[0]

                # Multiplication: return x * N
                match = re.match(rf'{arg}\s*\*\s*(\d+)', expr)
                if match:
                    return Rule(
                        name="multiply",
                        pattern_type="arithmetic",
                        code_template=f"return $P * {match.group(1)}",
                        description=f"multiply by {match.group(1)}"
                    )

                # Addition: return x + N
                match = re.match(rf'{arg}\s*\+\s*(\d+)', expr)
                if match:
                    return Rule(
                        name="add",
                        pattern_type="arithmetic",
                        code_template=f"return $P + {match.group(1)}",
                        description=f"add {match.group(1)}"
                    )

                # len(): return len(x)
                if expr == f'len({arg})':
                    return Rule(
                        name="length",
                        pattern_type="string",
                        code_template="return len($P)",
                        description="length"
                    )

                # sum(): return sum(x)
                if expr == f'sum({arg})':
                    return Rule(
                        name="sum",
                        pattern_type="list",
                        code_template="return sum($P)",
                        description="sum"
                    )

                # sorted(): return sorted(x)
                if expr == f'sorted({arg})':
                    return Rule(
                        name="sort",
                        pattern_type="list",
                        code_template="return sorted($P)",
                        description="sort"
                    )

                # Reverse: return x[::-1]
                if expr == f'{arg}[::-1]':
                    return Rule(
                        name="reverse",
                        pattern_type="string",
                        code_template="return $P[::-1]",
                        description="reverse"
                    )

            elif len(args) == 2:
                a, b = args

                # Addition: return a + b
                if expr == f'{a} + {b}' or expr == f'{b} + {a}':
                    return Rule(
                        name="add_two",
                        pattern_type="arithmetic",
                        code_template="return $P0 + $P1",
                        description="add two values"
                    )

                # Multiplication: return a * b
                if expr == f'{a} * {b}' or expr == f'{b} * {a}':
                    return Rule(
                        name="multiply_two",
                        pattern_type="arithmetic",
                        code_template="return $P0 * $P1",
                        description="multiply two values"
                    )

                # Max: return max(a, b)
                if expr == f'max({a}, {b})' or expr == f'max({b}, {a})':
                    return Rule(
                        name="max_two",
                        pattern_type="arithmetic",
                        code_template="return max($P0, $P1)",
                        description="max of two values"
                    )

                # Min: return min(a, b)
                if expr == f'min({a}, {b})' or expr == f'min({b}, {a})':
                    return Rule(
                        name="min_two",
                        pattern_type="arithmetic",
                        code_template="return min($P0, $P1)",
                        description="min of two values"
                    )

        return None


# =============================================================================
# DEMO
# =============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("PATTERN LEARNER - Rules, not examples")
    print("=" * 60)
    print()

    learner = PatternLearner()

    # Learn from examples (examples are discarded after learning)
    print("Learning rules from examples...")
    print()

    tests = [
        ("double", [(1, 2), (2, 4), (3, 6), (5, 10)]),
        ("square", [(1, 1), (2, 4), (3, 9), (5, 25)]),
        ("increment", [(0, 1), (5, 6), (10, 11)]),
        ("factorial", [(0, 1), (1, 1), (3, 6), (5, 120)]),
        ("fibonacci", [(0, 0), (1, 1), (5, 5), (10, 55)]),
        ("is_even", [(0, True), (1, False), (2, True), (3, False)]),
        ("is_prime", [(2, True), (4, False), (7, True), (9, False)]),
        ("add", [((1, 2), 3), ((5, 3), 8)]),
        ("multiply", [((2, 3), 6), ((4, 5), 20)]),
        ("reverse", [("hello", "olleh"), ("abc", "cba")]),
        ("length", [("hello", 5), ("", 0), ("a", 1)]),
        ("sum_list", [([1, 2, 3], 6), ([], 0)]),
        ("sort_list", [([3, 1, 2], [1, 2, 3])]),
    ]

    for name, examples in tests:
        rule = learner.learn(name, examples)
        if rule:
            print(f"  {name}: learned '{rule.description}'")
        else:
            print(f"  {name}: could not induce rule")

    print()
    print("-" * 60)
    print("Statistics:")
    print("-" * 60)
    stats = learner.stats()
    print(f"  Total rules: {stats['total_rules']}")
    print(f"  By type: {stats['by_type']}")
    print(f"  Memory: {stats['memory']}")
    print()

    print("-" * 60)
    print("Generating code from rules (no examples needed):")
    print("-" * 60)
    print()

    for name in ['double', 'factorial', 'is_prime', 'add', 'reverse']:
        code = learner.generate(name, f"def {name}(x)")
        body = code.split('\n')[1].strip()
        print(f"  {name}: {body}")

    print()
    print("=" * 60)
    print("KEY INSIGHT:")
    print("  Examples are used to LEARN the rule, then discarded.")
    print("  Only the rule is stored - much more efficient!")
    print("=" * 60)
    print()

    # The Stack integration demo
    print()
    print("=" * 60)
    print("THE STACK INTEGRATION (BigCode)")
    print("=" * 60)
    print()
    print("To learn from The Stack dataset:")
    print()
    print("  1. Install: pip install datasets")
    print()
    print("  2. Use:")
    print("     explorer = StackExplorer(learner)")
    print("     explorer.load('python')  # Streaming mode")
    print("     explorer.learn_from_stream(n_samples=100)")
    print()
    print("  This streams code samples, extracts patterns,")
    print("  learns rules, then discards the code.")
    print()

    # Try to load The Stack if datasets is available
    try:
        print("Checking if 'datasets' library is installed...")
        import datasets
        print("  datasets library found!")
        print()
        print("  Would you like to try streaming from The Stack?")
        print("  (This requires internet and Hugging Face access)")
        print()
    except ImportError:
        print("  datasets library not installed.")
        print("  Install with: pip install datasets")
        print()
