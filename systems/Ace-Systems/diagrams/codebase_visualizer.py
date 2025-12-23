"""
Codebase Visualizer - Single file dependency graph generator

Drop this file into any Python codebase, run it, and open the generated HTML.
Shows file connections, orphans, hubs, and gaps in your codebase.

Usage:
    python codebase_visualizer.py [directory] [--watch]

    directory: Path to scan (default: current directory)
    --watch: Auto-regenerate when files change
"""

import ast
import os
import sys
import json
import hashlib
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================

IGNORE_DIRS = {
    '__pycache__', '.git', '.venv', 'venv', 'env', 'node_modules',
    '.pytest_cache', '.mypy_cache', 'dist', 'build', 'egg-info',
    '.grace_backups', 'archived_scripts', '.claude'
}

IGNORE_FILES = {
    '__init__.py',  # Usually just re-exports
}

# File extensions to analyze
ANALYZABLE_EXTENSIONS = {'.py'}

# Additional extensions to show (but not analyze imports)
ADDITIONAL_EXTENSIONS = {'.json', '.md', '.txt', '.yaml', '.yml', '.toml'}

# ============================================================================
# DEPENDENCY ANALYZER
# ============================================================================

class DependencyAnalyzer:
    """Analyzes Python files for import dependencies and cross-references."""

    def __init__(self, root_path: str):
        self.root = Path(root_path).resolve()
        self.files = {}  # path -> FileInfo
        self.edges = []  # list of (source, target, edge_type)
        self.modules = {}  # module_name -> file_path

    def scan(self):
        """Scan the directory and build the dependency graph."""
        print(f"Scanning: {self.root}")

        # First pass: collect all files
        for filepath in self._iter_files():
            rel_path = filepath.relative_to(self.root)
            self.files[str(rel_path)] = self._analyze_file(filepath)

            # Map module names to files
            if filepath.suffix == '.py':
                module_name = self._path_to_module(rel_path)
                self.modules[module_name] = str(rel_path)

        print(f"  Found {len(self.files)} files")

        # Second pass: resolve imports to edges
        for rel_path, info in self.files.items():
            if info['type'] == 'python':
                self._resolve_imports(rel_path, info)

        # Third pass: find string references (file paths, etc.)
        for rel_path, info in self.files.items():
            if info['type'] == 'python':
                self._find_string_references(rel_path, info)

        print(f"  Found {len(self.edges)} connections")

        return self._build_graph_data()

    def _iter_files(self):
        """Iterate over relevant files in the directory."""
        all_extensions = ANALYZABLE_EXTENSIONS | ADDITIONAL_EXTENSIONS

        for root, dirs, files in os.walk(self.root):
            # Filter ignored directories
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

            for filename in files:
                if filename in IGNORE_FILES:
                    continue

                filepath = Path(root) / filename
                if filepath.suffix in all_extensions:
                    yield filepath

    def _path_to_module(self, rel_path: Path) -> str:
        """Convert file path to Python module name."""
        parts = list(rel_path.parts)
        if parts[-1].endswith('.py'):
            parts[-1] = parts[-1][:-3]
        return '.'.join(parts)

    def _analyze_file(self, filepath: Path) -> dict:
        """Extract information from a single file."""
        info = {
            'path': str(filepath.relative_to(self.root)),
            'name': filepath.name,
            'type': 'python' if filepath.suffix == '.py' else 'data',
            'size': filepath.stat().st_size,
            'lines': 0,
            'imports': [],
            'from_imports': [],
            'classes': [],
            'functions': [],
            'references': [],
            'directory': str(filepath.parent.relative_to(self.root)) if filepath.parent != self.root else '.'
        }

        if filepath.suffix == '.py':
            try:
                content = filepath.read_text(encoding='utf-8', errors='replace')
                info['lines'] = len(content.splitlines())

                # Parse AST for imports and definitions
                try:
                    tree = ast.parse(content)
                    info['imports'], info['from_imports'] = self._extract_imports(tree)
                    info['classes'] = self._extract_classes(tree)
                    info['functions'] = self._extract_functions(tree)
                except SyntaxError as e:
                    info['parse_error'] = str(e)

            except Exception as e:
                info['read_error'] = str(e)
        else:
            try:
                content = filepath.read_text(encoding='utf-8', errors='replace')
                info['lines'] = len(content.splitlines())
            except:
                pass

        return info

    def _extract_imports(self, tree: ast.AST) -> tuple:
        """Extract import statements from AST."""
        imports = []
        from_imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    from_imports.append({
                        'module': node.module,
                        'names': [a.name for a in node.names]
                    })

        return imports, from_imports

    def _extract_classes(self, tree: ast.AST) -> list:
        """Extract class definitions."""
        classes = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                classes.append(node.name)
        return classes

    def _extract_functions(self, tree: ast.AST) -> list:
        """Extract top-level function definitions."""
        functions = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                functions.append(node.name)
        return functions

    def _resolve_imports(self, source_path: str, info: dict):
        """Resolve import statements to actual file edges."""
        # Direct imports
        for module in info['imports']:
            target = self._find_module(module)
            if target and target != source_path:
                self.edges.append((source_path, target, 'import'))

        # From imports
        for imp in info['from_imports']:
            target = self._find_module(imp['module'])
            if target and target != source_path:
                self.edges.append((source_path, target, 'from_import'))

    def _find_module(self, module_name: str) -> str:
        """Find the file path for a module name."""
        # Try exact match
        if module_name in self.modules:
            return self.modules[module_name]

        # Try as relative import (just the last part)
        parts = module_name.split('.')
        for i in range(len(parts)):
            partial = '.'.join(parts[i:])
            if partial in self.modules:
                return self.modules[partial]

        # Try just the module name as a file
        simple_name = parts[-1]
        possible_file = f"{simple_name}.py"
        for path in self.files:
            if path.endswith(possible_file):
                return path

        return None

    def _find_string_references(self, source_path: str, info: dict):
        """Find references to other files via string literals."""
        try:
            filepath = self.root / source_path
            content = filepath.read_text(encoding='utf-8', errors='replace')

            # Look for references to other files in the codebase
            for other_path in self.files:
                if other_path == source_path:
                    continue

                # Check for filename references
                other_name = Path(other_path).stem
                patterns = [
                    f"'{other_name}'",
                    f'"{other_name}"',
                    f"'{other_path}'",
                    f'"{other_path}"',
                ]

                for pattern in patterns:
                    if pattern in content:
                        # Avoid duplicates
                        edge = (source_path, other_path, 'reference')
                        if edge not in self.edges:
                            self.edges.append(edge)
                        break

        except Exception:
            pass

    def _build_graph_data(self) -> dict:
        """Build the final graph data structure for visualization."""
        # Calculate metrics
        incoming = defaultdict(int)
        outgoing = defaultdict(int)

        for source, target, _ in self.edges:
            outgoing[source] += 1
            incoming[target] += 1

        # Build nodes
        nodes = []
        for path, info in self.files.items():
            node = {
                'id': path,
                'name': info['name'],
                'directory': info['directory'],
                'type': info['type'],
                'lines': info['lines'],
                'size': info['size'],
                'classes': info.get('classes', []),
                'functions': info.get('functions', []),
                'incoming': incoming[path],
                'outgoing': outgoing[path],
                'total_connections': incoming[path] + outgoing[path],
                'is_orphan': incoming[path] == 0 and info['type'] == 'python',
                'is_hub': incoming[path] >= 5,
                'is_entry': outgoing[path] > 0 and incoming[path] == 0,
            }
            nodes.append(node)

        # Build edges
        edges = [
            {'source': s, 'target': t, 'type': typ}
            for s, t, typ in self.edges
        ]

        # Calculate directory groups
        directories = list(set(n['directory'] for n in nodes))

        return {
            'nodes': nodes,
            'edges': edges,
            'directories': directories,
            'stats': {
                'total_files': len(nodes),
                'total_connections': len(edges),
                'orphan_files': sum(1 for n in nodes if n['is_orphan']),
                'hub_files': sum(1 for n in nodes if n['is_hub']),
                'total_lines': sum(n['lines'] for n in nodes),
            },
            'generated': datetime.now().isoformat(),
            'root': str(self.root),
        }


# ============================================================================
# HTML GENERATOR
# ============================================================================

def generate_html(graph_data: dict, output_path: str):
    """Generate a self-contained HTML visualization."""

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Codebase Map - {Path(graph_data['root']).name}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #0d1117;
            color: #c9d1d9;
            overflow: hidden;
        }}

        #container {{
            display: flex;
            height: 100vh;
        }}

        #sidebar {{
            width: 320px;
            background: #161b22;
            border-right: 1px solid #30363d;
            padding: 20px;
            overflow-y: auto;
            flex-shrink: 0;
        }}

        #graph {{
            flex: 1;
            position: relative;
        }}

        h1 {{
            font-size: 18px;
            margin-bottom: 5px;
            color: #58a6ff;
        }}

        .subtitle {{
            font-size: 12px;
            color: #8b949e;
            margin-bottom: 20px;
        }}

        .stats {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin-bottom: 20px;
        }}

        .stat {{
            background: #21262d;
            padding: 12px;
            border-radius: 6px;
            text-align: center;
        }}

        .stat-value {{
            font-size: 24px;
            font-weight: bold;
            color: #58a6ff;
        }}

        .stat-label {{
            font-size: 11px;
            color: #8b949e;
            margin-top: 4px;
        }}

        .legend {{
            margin-bottom: 20px;
        }}

        .legend-title {{
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 10px;
            color: #c9d1d9;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 6px 0;
            font-size: 12px;
        }}

        .legend-color {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }}

        .controls {{
            margin-bottom: 20px;
        }}

        .control-group {{
            margin-bottom: 15px;
        }}

        .control-label {{
            font-size: 12px;
            color: #8b949e;
            margin-bottom: 6px;
        }}

        input[type="range"] {{
            width: 100%;
            accent-color: #58a6ff;
        }}

        .checkbox-group {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}

        .checkbox-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
            cursor: pointer;
        }}

        .checkbox-item input {{
            accent-color: #58a6ff;
        }}

        .search-box {{
            width: 100%;
            padding: 10px 12px;
            background: #21262d;
            border: 1px solid #30363d;
            border-radius: 6px;
            color: #c9d1d9;
            font-size: 13px;
            margin-bottom: 15px;
        }}

        .search-box:focus {{
            outline: none;
            border-color: #58a6ff;
        }}

        .file-list {{
            max-height: 300px;
            overflow-y: auto;
        }}

        .file-item {{
            padding: 8px 10px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .file-item:hover {{
            background: #21262d;
        }}

        .file-item.orphan {{
            border-left: 3px solid #f85149;
        }}

        .file-item.hub {{
            border-left: 3px solid #a371f7;
        }}

        .file-badge {{
            font-size: 10px;
            padding: 2px 6px;
            border-radius: 10px;
            background: #30363d;
        }}

        #tooltip {{
            position: absolute;
            background: #21262d;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 12px 16px;
            font-size: 12px;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.15s;
            max-width: 350px;
            z-index: 1000;
            box-shadow: 0 8px 24px rgba(0,0,0,0.4);
        }}

        #tooltip.visible {{
            opacity: 1;
        }}

        .tooltip-title {{
            font-weight: 600;
            color: #58a6ff;
            margin-bottom: 8px;
            word-break: break-all;
        }}

        .tooltip-row {{
            display: flex;
            justify-content: space-between;
            padding: 3px 0;
            color: #8b949e;
        }}

        .tooltip-row span:last-child {{
            color: #c9d1d9;
        }}

        .tooltip-section {{
            margin-top: 8px;
            padding-top: 8px;
            border-top: 1px solid #30363d;
        }}

        .tooltip-list {{
            color: #7ee787;
            font-size: 11px;
        }}

        svg {{
            width: 100%;
            height: 100%;
        }}

        .node {{
            cursor: pointer;
            transition: opacity 0.2s;
        }}

        .node:hover {{
            filter: brightness(1.2);
        }}

        .node.dimmed {{
            opacity: 0.15;
        }}

        .node.highlighted {{
            filter: brightness(1.3);
        }}

        .link {{
            fill: none;
            stroke-opacity: 0.4;
            transition: stroke-opacity 0.2s, stroke-width 0.2s;
        }}

        .link.dimmed {{
            stroke-opacity: 0.05;
        }}

        .link.highlighted {{
            stroke-opacity: 0.9;
            stroke-width: 2px;
        }}

        .node-label {{
            font-size: 10px;
            fill: #8b949e;
            pointer-events: none;
            text-anchor: middle;
        }}

        .node-label.visible {{
            fill: #c9d1d9;
        }}

        #info-panel {{
            position: absolute;
            bottom: 20px;
            left: 20px;
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            padding: 15px;
            font-size: 12px;
            max-width: 280px;
        }}

        .info-title {{
            font-weight: 600;
            color: #58a6ff;
            margin-bottom: 8px;
        }}

        .btn {{
            padding: 8px 16px;
            background: #21262d;
            border: 1px solid #30363d;
            border-radius: 6px;
            color: #c9d1d9;
            cursor: pointer;
            font-size: 12px;
            margin-top: 10px;
        }}

        .btn:hover {{
            background: #30363d;
        }}

        .btn-primary {{
            background: #238636;
            border-color: #238636;
        }}

        .btn-primary:hover {{
            background: #2ea043;
        }}
    </style>
</head>
<body>
    <div id="container">
        <div id="sidebar">
            <h1>Codebase Map</h1>
            <div class="subtitle">{Path(graph_data['root']).name}</div>

            <div class="stats">
                <div class="stat">
                    <div class="stat-value">{graph_data['stats']['total_files']}</div>
                    <div class="stat-label">Files</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{graph_data['stats']['total_connections']}</div>
                    <div class="stat-label">Connections</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{graph_data['stats']['orphan_files']}</div>
                    <div class="stat-label">Orphans</div>
                </div>
                <div class="stat">
                    <div class="stat-value">{graph_data['stats']['hub_files']}</div>
                    <div class="stat-label">Hubs</div>
                </div>
            </div>

            <div class="legend">
                <div class="legend-title">Node Types</div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #58a6ff;"></div>
                    <span>Python file</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #f85149;"></div>
                    <span>Orphan (no incoming)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #a371f7;"></div>
                    <span>Hub (5+ incoming)</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #7ee787;"></div>
                    <span>Entry point</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #848d97;"></div>
                    <span>Data file</span>
                </div>
            </div>

            <div class="legend">
                <div class="legend-title">Edge Types</div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #58a6ff;"></div>
                    <span>Import</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #a371f7;"></div>
                    <span>From import</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: #848d97;"></div>
                    <span>String reference</span>
                </div>
            </div>

            <div class="controls">
                <div class="control-group">
                    <div class="control-label">Node Size</div>
                    <input type="range" id="node-size" min="1" max="20" value="8">
                </div>
                <div class="control-group">
                    <div class="control-label">Link Distance</div>
                    <input type="range" id="link-distance" min="30" max="200" value="80">
                </div>
                <div class="control-group">
                    <div class="checkbox-group">
                        <label class="checkbox-item">
                            <input type="checkbox" id="show-labels" checked>
                            Show labels
                        </label>
                        <label class="checkbox-item">
                            <input type="checkbox" id="show-orphans" checked>
                            Highlight orphans
                        </label>
                        <label class="checkbox-item">
                            <input type="checkbox" id="show-data-files">
                            Show data files
                        </label>
                    </div>
                </div>
            </div>

            <input type="text" class="search-box" id="search" placeholder="Search files...">

            <div class="legend-title">Files</div>
            <div class="file-list" id="file-list"></div>
        </div>

        <div id="graph">
            <svg id="svg"></svg>
            <div id="tooltip"></div>
        </div>
    </div>

    <script>
        // Graph data embedded
        const graphData = {json.dumps(graph_data, indent=2)};

        // D3.js v7 (minified, embedded)
        // Using a simplified force simulation

        class ForceGraph {{
            constructor(container, data) {{
                this.container = container;
                this.data = data;
                this.svg = document.getElementById('svg');
                this.tooltip = document.getElementById('tooltip');
                this.width = container.clientWidth;
                this.height = container.clientHeight;

                this.nodes = data.nodes.filter(n => n.type === 'python' || document.getElementById('show-data-files')?.checked);
                this.edges = data.edges.filter(e =>
                    this.nodes.find(n => n.id === e.source) &&
                    this.nodes.find(n => n.id === e.target)
                );

                this.simulation = null;
                this.nodeElements = [];
                this.linkElements = [];
                this.labelElements = [];

                this.selectedNode = null;
                this.baseNodeSize = 8;
                this.linkDistance = 80;

                this.init();
            }}

            init() {{
                // Create SVG groups
                this.svg.innerHTML = `
                    <defs>
                        <marker id="arrow" viewBox="0 0 10 10" refX="20" refY="5"
                            markerWidth="6" markerHeight="6" orient="auto">
                            <path d="M 0 0 L 10 5 L 0 10 z" fill="#30363d"/>
                        </marker>
                    </defs>
                    <g class="links"></g>
                    <g class="nodes"></g>
                    <g class="labels"></g>
                `;

                this.linksGroup = this.svg.querySelector('.links');
                this.nodesGroup = this.svg.querySelector('.nodes');
                this.labelsGroup = this.svg.querySelector('.labels');

                // Initialize node positions
                this.nodes.forEach((node, i) => {{
                    node.x = this.width / 2 + (Math.random() - 0.5) * 200;
                    node.y = this.height / 2 + (Math.random() - 0.5) * 200;
                    node.vx = 0;
                    node.vy = 0;
                }});

                this.createElements();
                this.startSimulation();
                this.setupInteractions();
                this.populateFileList();
            }}

            getNodeColor(node) {{
                if (node.is_orphan) return '#f85149';
                if (node.is_hub) return '#a371f7';
                if (node.is_entry) return '#7ee787';
                if (node.type === 'data') return '#848d97';
                return '#58a6ff';
            }}

            getEdgeColor(edge) {{
                if (edge.type === 'import') return '#58a6ff';
                if (edge.type === 'from_import') return '#a371f7';
                return '#848d97';
            }}

            getNodeSize(node) {{
                const base = this.baseNodeSize;
                const linesFactor = Math.sqrt(node.lines) / 10;
                const connectionsFactor = Math.sqrt(node.total_connections + 1);
                return base + linesFactor + connectionsFactor;
            }}

            createElements() {{
                // Create links
                this.edges.forEach(edge => {{
                    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                    line.classList.add('link');
                    line.setAttribute('stroke', this.getEdgeColor(edge));
                    line.setAttribute('stroke-width', '1');
                    line.dataset.source = edge.source;
                    line.dataset.target = edge.target;
                    this.linksGroup.appendChild(line);
                    this.linkElements.push(line);
                }});

                // Create nodes
                this.nodes.forEach(node => {{
                    const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
                    circle.classList.add('node');
                    circle.setAttribute('r', this.getNodeSize(node));
                    circle.setAttribute('fill', this.getNodeColor(node));
                    circle.dataset.id = node.id;
                    this.nodesGroup.appendChild(circle);
                    this.nodeElements.push(circle);

                    // Create label
                    const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                    label.classList.add('node-label');
                    label.textContent = node.name.replace('.py', '');
                    label.dataset.id = node.id;
                    this.labelsGroup.appendChild(label);
                    this.labelElements.push(label);
                }});
            }}

            startSimulation() {{
                this.alpha = 1.0;  // Simulation energy (1 = hot, 0 = frozen)
                this.alphaDecay = 0.99;  // Cools down 1% per frame
                this.alphaMin = 0.001;  // Stop when this cold

                const tick = () => {{
                    if (this.alpha > this.alphaMin) {{
                        // Apply forces with decreasing strength
                        this.applyForces();
                        this.alpha *= this.alphaDecay;

                        // Update positions
                        this.updatePositions();
                    }}

                    requestAnimationFrame(tick);
                }};
                tick();
            }}

            reheat() {{
                // Restart simulation (called after dragging)
                this.alpha = 0.5;
            }}

            applyForces() {{
                const alpha = this.alpha * 0.1;  // Scale by simulation energy
                const repulsion = 500;
                const linkStrength = 0.1;
                const centerX = this.width / 2;
                const centerY = this.height / 2;

                // Repulsion between nodes
                for (let i = 0; i < this.nodes.length; i++) {{
                    for (let j = i + 1; j < this.nodes.length; j++) {{
                        const a = this.nodes[i];
                        const b = this.nodes[j];
                        const dx = b.x - a.x;
                        const dy = b.y - a.y;
                        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
                        const force = repulsion / (dist * dist);

                        a.vx -= (dx / dist) * force * alpha;
                        a.vy -= (dy / dist) * force * alpha;
                        b.vx += (dx / dist) * force * alpha;
                        b.vy += (dy / dist) * force * alpha;
                    }}
                }}

                // Link attraction
                this.edges.forEach(edge => {{
                    const source = this.nodes.find(n => n.id === edge.source);
                    const target = this.nodes.find(n => n.id === edge.target);
                    if (!source || !target) return;

                    const dx = target.x - source.x;
                    const dy = target.y - source.y;
                    const dist = Math.sqrt(dx * dx + dy * dy) || 1;
                    const force = (dist - this.linkDistance) * linkStrength * alpha;

                    source.vx += (dx / dist) * force;
                    source.vy += (dy / dist) * force;
                    target.vx -= (dx / dist) * force;
                    target.vy -= (dy / dist) * force;
                }});

                // Center gravity
                this.nodes.forEach(node => {{
                    node.vx += (centerX - node.x) * 0.01 * alpha;
                    node.vy += (centerY - node.y) * 0.01 * alpha;
                }});

                // Apply velocity with damping
                this.nodes.forEach(node => {{
                    node.vx *= 0.9;
                    node.vy *= 0.9;
                    node.x += node.vx;
                    node.y += node.vy;

                    // Boundary constraints
                    const padding = 50;
                    node.x = Math.max(padding, Math.min(this.width - padding, node.x));
                    node.y = Math.max(padding, Math.min(this.height - padding, node.y));
                }});
            }}

            updatePositions() {{
                this.nodes.forEach((node, i) => {{
                    const circle = this.nodeElements[i];
                    const label = this.labelElements[i];

                    circle.setAttribute('cx', node.x);
                    circle.setAttribute('cy', node.y);

                    label.setAttribute('x', node.x);
                    label.setAttribute('y', node.y - this.getNodeSize(node) - 5);
                }});

                this.edges.forEach((edge, i) => {{
                    const source = this.nodes.find(n => n.id === edge.source);
                    const target = this.nodes.find(n => n.id === edge.target);
                    if (!source || !target) return;

                    const line = this.linkElements[i];
                    line.setAttribute('x1', source.x);
                    line.setAttribute('y1', source.y);
                    line.setAttribute('x2', target.x);
                    line.setAttribute('y2', target.y);
                }});
            }}

            setupInteractions() {{
                // Node hover and click
                this.nodeElements.forEach((el, i) => {{
                    const node = this.nodes[i];

                    el.addEventListener('mouseenter', (e) => {{
                        this.showTooltip(node, e);
                        if (!this.selectedNode) {{
                            this.highlightConnections(node);
                        }}
                    }});

                    el.addEventListener('mouseleave', () => {{
                        this.hideTooltip();
                        if (!this.selectedNode) {{
                            this.clearHighlight();
                        }}
                    }});

                    el.addEventListener('click', () => {{
                        if (this.selectedNode === node) {{
                            this.selectedNode = null;
                            this.clearHighlight();
                        }} else {{
                            this.selectedNode = node;
                            this.highlightConnections(node);
                        }}
                    }});
                }});

                // Drag
                let dragNode = null;
                let dragOffset = {{ x: 0, y: 0 }};

                this.svg.addEventListener('mousedown', (e) => {{
                    const target = e.target;
                    if (target.classList.contains('node')) {{
                        const id = target.dataset.id;
                        dragNode = this.nodes.find(n => n.id === id);
                        if (dragNode) {{
                            dragOffset.x = e.clientX - dragNode.x;
                            dragOffset.y = e.clientY - dragNode.y;
                        }}
                    }}
                }});

                this.svg.addEventListener('mousemove', (e) => {{
                    if (dragNode) {{
                        dragNode.x = e.clientX - dragOffset.x;
                        dragNode.y = e.clientY - dragOffset.y;
                        dragNode.vx = 0;
                        dragNode.vy = 0;
                    }}
                }});

                this.svg.addEventListener('mouseup', () => {{
                    if (dragNode) {{
                        this.reheat();  // Briefly restart simulation after drag
                    }}
                    dragNode = null;
                }});

                // Controls
                document.getElementById('node-size').addEventListener('input', (e) => {{
                    this.baseNodeSize = parseInt(e.target.value);
                    this.nodeElements.forEach((el, i) => {{
                        el.setAttribute('r', this.getNodeSize(this.nodes[i]));
                    }});
                }});

                document.getElementById('link-distance').addEventListener('input', (e) => {{
                    this.linkDistance = parseInt(e.target.value);
                }});

                document.getElementById('show-labels').addEventListener('change', (e) => {{
                    this.labelElements.forEach(el => {{
                        el.style.display = e.target.checked ? '' : 'none';
                    }});
                }});

                document.getElementById('search').addEventListener('input', (e) => {{
                    const query = e.target.value.toLowerCase();
                    if (query) {{
                        const matches = this.nodes.filter(n =>
                            n.id.toLowerCase().includes(query) ||
                            n.name.toLowerCase().includes(query)
                        );
                        this.highlightNodes(matches);
                    }} else {{
                        this.clearHighlight();
                    }}
                    this.filterFileList(query);
                }});

                // Resize
                window.addEventListener('resize', () => {{
                    this.width = this.container.clientWidth;
                    this.height = this.container.clientHeight;
                }});
            }}

            showTooltip(node, event) {{
                const tooltip = this.tooltip;

                let content = `
                    <div class="tooltip-title">${{node.id}}</div>
                    <div class="tooltip-row"><span>Lines:</span><span>${{node.lines}}</span></div>
                    <div class="tooltip-row"><span>Directory:</span><span>${{node.directory}}</span></div>
                    <div class="tooltip-row"><span>Incoming:</span><span>${{node.incoming}}</span></div>
                    <div class="tooltip-row"><span>Outgoing:</span><span>${{node.outgoing}}</span></div>
                `;

                if (node.classes.length > 0) {{
                    content += `
                        <div class="tooltip-section">
                            <div class="tooltip-row"><span>Classes:</span></div>
                            <div class="tooltip-list">${{node.classes.slice(0, 5).join(', ')}}${{node.classes.length > 5 ? '...' : ''}}</div>
                        </div>
                    `;
                }}

                if (node.functions.length > 0) {{
                    content += `
                        <div class="tooltip-section">
                            <div class="tooltip-row"><span>Functions:</span></div>
                            <div class="tooltip-list">${{node.functions.slice(0, 5).join(', ')}}${{node.functions.length > 5 ? '...' : ''}}</div>
                        </div>
                    `;
                }}

                tooltip.innerHTML = content;
                tooltip.style.left = (event.clientX + 15) + 'px';
                tooltip.style.top = (event.clientY + 15) + 'px';
                tooltip.classList.add('visible');
            }}

            hideTooltip() {{
                this.tooltip.classList.remove('visible');
            }}

            highlightConnections(node) {{
                const connectedIds = new Set([node.id]);

                this.edges.forEach(edge => {{
                    if (edge.source === node.id) connectedIds.add(edge.target);
                    if (edge.target === node.id) connectedIds.add(edge.source);
                }});

                this.nodeElements.forEach((el, i) => {{
                    if (connectedIds.has(this.nodes[i].id)) {{
                        el.classList.remove('dimmed');
                        el.classList.add('highlighted');
                    }} else {{
                        el.classList.add('dimmed');
                        el.classList.remove('highlighted');
                    }}
                }});

                this.linkElements.forEach((el, i) => {{
                    const edge = this.edges[i];
                    if (edge.source === node.id || edge.target === node.id) {{
                        el.classList.remove('dimmed');
                        el.classList.add('highlighted');
                    }} else {{
                        el.classList.add('dimmed');
                        el.classList.remove('highlighted');
                    }}
                }});

                this.labelElements.forEach((el, i) => {{
                    if (connectedIds.has(this.nodes[i].id)) {{
                        el.classList.add('visible');
                    }}
                }});
            }}

            highlightNodes(nodes) {{
                const ids = new Set(nodes.map(n => n.id));

                this.nodeElements.forEach((el, i) => {{
                    if (ids.has(this.nodes[i].id)) {{
                        el.classList.remove('dimmed');
                        el.classList.add('highlighted');
                    }} else {{
                        el.classList.add('dimmed');
                        el.classList.remove('highlighted');
                    }}
                }});
            }}

            clearHighlight() {{
                this.nodeElements.forEach(el => {{
                    el.classList.remove('dimmed', 'highlighted');
                }});
                this.linkElements.forEach(el => {{
                    el.classList.remove('dimmed', 'highlighted');
                }});
                this.labelElements.forEach(el => {{
                    el.classList.remove('visible');
                }});
            }}

            populateFileList() {{
                const list = document.getElementById('file-list');
                const sortedNodes = [...this.nodes].sort((a, b) => b.total_connections - a.total_connections);

                sortedNodes.forEach(node => {{
                    const item = document.createElement('div');
                    item.className = 'file-item';
                    if (node.is_orphan) item.classList.add('orphan');
                    if (node.is_hub) item.classList.add('hub');

                    item.innerHTML = `
                        <span>${{node.name}}</span>
                        <span class="file-badge">${{node.total_connections}}</span>
                    `;

                    item.addEventListener('click', () => {{
                        this.selectedNode = node;
                        this.highlightConnections(node);
                        // Center on node
                        node.x = this.width / 2;
                        node.y = this.height / 2;
                    }});

                    item.dataset.id = node.id;
                    list.appendChild(item);
                }});
            }}

            filterFileList(query) {{
                const items = document.querySelectorAll('.file-item');
                items.forEach(item => {{
                    const matches = item.dataset.id.toLowerCase().includes(query);
                    item.style.display = matches ? '' : 'none';
                }});
            }}
        }}

        // Initialize
        const container = document.getElementById('graph');
        const graph = new ForceGraph(container, graphData);
    </script>
</body>
</html>
'''

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"Generated: {output_path}")


# ============================================================================
# FILE WATCHER
# ============================================================================

def watch_directory(root_path: str, output_path: str, interval: float = 2.0):
    """Watch for file changes and regenerate the graph."""
    print(f"Watching for changes (Ctrl+C to stop)...")

    last_hash = None

    try:
        while True:
            # Calculate hash of all Python files
            current_hash = hashlib.md5()

            for filepath in Path(root_path).rglob('*.py'):
                if any(part in IGNORE_DIRS for part in filepath.parts):
                    continue
                try:
                    stat = filepath.stat()
                    current_hash.update(f"{filepath}:{stat.st_mtime}".encode())
                except:
                    pass

            hash_digest = current_hash.hexdigest()

            if hash_digest != last_hash:
                if last_hash is not None:
                    print(f"\nChanges detected, regenerating...")

                analyzer = DependencyAnalyzer(root_path)
                graph_data = analyzer.scan()
                generate_html(graph_data, output_path)
                last_hash = hash_digest

            import time
            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nStopped watching.")


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Codebase Visualizer - Generate dependency graphs')
    parser.add_argument('directory', nargs='?', default='.', help='Directory to scan')
    parser.add_argument('--output', '-o', default='codebase_graph.html', help='Output HTML file')
    parser.add_argument('--watch', '-w', action='store_true', help='Watch for changes')

    args = parser.parse_args()

    root_path = Path(args.directory).resolve()
    output_path = root_path / args.output

    if not root_path.exists():
        print(f"Error: Directory not found: {root_path}")
        sys.exit(1)

    # Initial scan and generate
    analyzer = DependencyAnalyzer(str(root_path))
    graph_data = analyzer.scan()
    generate_html(graph_data, str(output_path))

    print(f"\nStats:")
    print(f"  Files: {graph_data['stats']['total_files']}")
    print(f"  Connections: {graph_data['stats']['total_connections']}")
    print(f"  Orphans: {graph_data['stats']['orphan_files']}")
    print(f"  Hubs: {graph_data['stats']['hub_files']}")
    print(f"  Total lines: {graph_data['stats']['total_lines']:,}")

    if args.watch:
        watch_directory(str(root_path), str(output_path))
    else:
        print(f"\nOpen {output_path} in a browser to view the graph.")


if __name__ == '__main__':
    main()
