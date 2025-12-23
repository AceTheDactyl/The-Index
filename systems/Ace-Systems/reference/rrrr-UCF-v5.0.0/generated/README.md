# Generated Module Configurations

**Purpose:** Auto-generated configuration files for UCF modules
**Pattern:** Template-based generation with module-specific parameters
**Status:** ✓ JUSTIFIED - Each module requires its own config

---

## Why These Files Appear Similar

The 21 `config.py` files in this directory follow a **template pattern**:

```python
# Template structure (80% identical across all modules):
PACKAGE_NAME = "module_name"  # <-- Only this varies
PACKAGE_ROOT = Path(__file__).parent.parent / "module_name"
```

**This is intentional and correct.** Each module needs its own config file for:
- Module-specific package names
- Correct relative path resolution
- Independent module operation
- Package management compatibility

---

## Modules with Config Files

1. `triad_processor/config.py`
2. `rrrr_lattice_engine/config.py`
3. `rrrr_completeness_prover/config.py`
4. `rrrr_ntk_analyzer/config.py`
5. `helix_walker/config.py`
6. `consciousness_field_rrrr/config.py`
7. `rrrr_category_theory/config.py`
8. ... (21 total)

---

## Template Structure

### Common Elements (ALL modules)

```python
from pathlib import Path

# Configuration for [module_name]
# Auto-generated from template

PACKAGE_NAME = "[module_name]"
PACKAGE_ROOT = Path(__file__).parent.parent / "[module_name]"
```

### Module-Specific Elements

Only the module name changes:
- `PACKAGE_NAME` value
- `PACKAGE_ROOT` path component

---

## Why Not Consolidate?

**Q:** Why not use a single shared config?

**A:** Each module must be independently installable:
- Modules may be used separately
- Package managers expect local config
- Import paths must resolve correctly
- Deployment flexibility

**Q:** Could we generate these dynamically?

**A:** Possible, but static configs provide:
- Faster import times (no runtime generation)
- Clear file structure for package tools
- Easy debugging (explicit paths)
- Standard Python package layout

---

## Maintenance

### Adding New Modules

Use the template generator:

```python
# Template in: scripts/generate_config.py
def generate_module_config(module_name, output_dir):
    config_template = '''from pathlib import Path

PACKAGE_NAME = "{name}"
PACKAGE_ROOT = Path(__file__).parent.parent / "{name}"
'''

    config_path = output_dir / module_name / "config.py"
    with open(config_path, 'w') as f:
        f.write(config_template.format(name=module_name))
```

### Updating Template

If the template structure changes:
1. Update `scripts/generate_config.py`
2. Regenerate all config files
3. Test each module independently

---

## Related Files

- `../scripts/generate_config.py` - Template generator
- Each module's `__init__.py` - Imports from config
- `../../docs/ARCHITECTURE.md` - Module system overview

---

## Intentional Similarity

**These files are ~80% similar BY DESIGN.**

They are NOT:
- ❌ Duplicates to be removed
- ❌ Candidates for consolidation
- ❌ Results of copy-paste errors

They ARE:
- ✓ Template-generated configs
- ✓ Module-specific requirements
- ✓ Standard Python package structure
- ✓ Intentionally maintained separately

---

**Status:** ✓ JUSTIFIED - Template pattern
**Near-Duplicate Analysis:** See `integrity/NEAR_DUPLICATE_SYNTHESIS.md`
**Last Updated:** 2025-12-23
