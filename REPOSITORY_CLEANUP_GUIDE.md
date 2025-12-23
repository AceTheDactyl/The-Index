# Repository Cleanup & Maintenance Guide

**Last Updated:** 2025-12-23
**Repository Health Score:** 52/100 (FAIR) → Target: 65+/100 (GOOD)

---

## ✨ What's Been Accomplished

### Phase 1: Exact Duplicate Removal ✅ COMPLETE

**Removed:** 172 exact duplicate files (7.47 MB saved)
- APL reference duplicates (40 files across `/reference/APL/` and `/reference/ace_apl/`)
- "To Sort" directory duplicates (15 files including s3_spiral17, DOMAIN_ files)
- The-Helix-Bridge duplicates (38 files - schemas, logs, vaultnodes)
- PlatformIO example duplicates (22 files)
- Living Chronicles duplicates (2 files)
- Various HTML, Python, and documentation duplicates

**Preserved:** 48 files in different contexts (cross-system docs, example projects)

**Verification:** SHA-256 hash matching confirmed byte-for-byte identity

**Reference:** `integrity/REDUNDANCY_CLEANUP_SUMMARY.md`

---

### Phase 2: Near-Duplicate Analysis ✅ COMPLETE

**Identified:** 46 near-duplicate pairs (80-98.7% similarity)

**Extracted:** 8 unique content sections representing the valuable 10%:
1. **Code verification examples** (32 lines Python) - `PHYSICS_GROUNDING.md`
2. **Implementation file references** - `CLAUDE_CONTRIBUTIONS.md`
3. **Enhanced executive summaries** - Better preambles and introductions
4. **Attribution details** - More specific collaboration information
5. **Math-injected narrative variants** - Living Chronicles variations
6. **Training epoch metadata** - Temporal evolution tracking
7. **Session logs** - Historical interaction records
8. **Module-specific configs** - Necessary package structure files

**Documented:** 41 intentional variants to preserve:
- 21 config.py files (module-specific, template-generated)
- 5 training epoch files (historical progression)
- 3 UCF session logs (temporal data)
- 6 story variations (narrative variants)
- 6 Python files in different contexts (examples vs reference)

**Reference:** `integrity/NEAR_DUPLICATE_SYNTHESIS.md`

---

### Phase 3: Metadata System ✅ COMPLETE

**Added metadata headers to:** 409 high-risk files (100% success rate)

**Status breakdown:**
- ✓ JUSTIFIED: 290 files (70.9%) - Claims supported by repository
- ⚠️ TRULY_UNSUPPORTED: 103 files (25.2%) - No supporting evidence
- 🔍 NEEDS_REVIEW: 16 files (3.9%) - Requires manual review
- 📝 Needs citation update: 231 files (56.5%)

**Metadata includes:**
- Validation status (JUSTIFIED, TRULY_UNSUPPORTED, NEEDS_REVIEW)
- Risk severity (LOW, MEDIUM, HIGH)
- Risk types (unsupported_claims, unverified_math, high_speculation)
- Supporting files and citing files (cross-references)
- Date of analysis

**Reference:** `integrity/CLAIM_VALIDATION_SUMMARY.md`

---

### Phase 4: Comprehensive Action Plan ✅ COMPLETE

**Created:** 10-phase cleanup roadmap with executable scripts

**Identified critical items:**
- 43 TEXTBOOK files (TRULY_UNSUPPORTED, ~400 KB) - **DECISION REQUIRED**
- 7 DOMAIN_ exact duplicates in "To Sort" - Ready for deletion
- 4 HTML files for symlink creation - Ready for execution
- 135 files missing metadata - Script prepared for addition

**Decision points:**
1. TEXTBOOK series: Archive vs Delete vs Review
2. Cross-system GRAND_SYNTHESIS files: Keep separate vs Consolidate
3. "To Sort" directory: Organize vs Bulk delete

**Reference:** `integrity/COMPREHENSIVE_ACTION_PLAN.md`

---

## 🎯 For Current Cleanup (Pending Actions)

### Immediate Actions Required

#### 1. Merge Unique Content (25 minutes)

**Files requiring manual merge:**

```bash
# PHYSICS_GROUNDING.md - Add enhanced preamble
# Line 5: Add "This document establishes that"
# End of file: Update attribution with more specific collaboration details

# After merging, delete:
git rm systems/Ace-Systems/reference/APL/Quantum-APL/Physics_Grounding_QuasiCrystal.md
```

**Verification command:**
```bash
# Ensure canonical has Code Verification section
grep -A 10 "Code Verification" \
  systems/Ace-Systems/examples/Quantum-APL-main/research/PHYSICS_GROUNDING.md
```

---

#### 2. Remove Remaining Exact Duplicates (5 minutes)

**Ready for deletion (7 files, ~150 KB):**

```bash
cd /home/user/The-Index

# DOMAIN_ duplicates from "To Sort" (6 files)
git rm "systems/Ace-Systems/docs/Research/To Sort/DOMAIN_1_KAEL_NEURAL_NETWORKS.md"
git rm "systems/Ace-Systems/docs/Research/To Sort/DOMAIN_2_ACE_SPIN_GLASS.md"
git rm "systems/Ace-Systems/docs/Research/To Sort/DOMAIN_3_GREY_VISUAL_GEOMETRY.md"
git rm "systems/Ace-Systems/docs/Research/To Sort/DOMAIN_4_UMBRAL_FORMAL_ALGEBRA.md"
git rm "systems/Ace-Systems/docs/Research/To Sort/DOMAIN_5_ULTRA_UNIVERSAL_GEOMETRY.md"
git rm "systems/Ace-Systems/docs/Research/To Sort/DOMAIN_6_UCF_IMPLEMENTATION.md"

# GRAND_SYNTHESIS duplicate (1 file)
git rm "systems/Ace-Systems/docs/Research/To Sort/GRAND_SYNTHESIS_SIX_FRAMEWORKS.md"

# Verify canonicals exist
ls systems/Ace-Systems/docs/Research/DOMAIN_*.md
ls systems/Ace-Systems/docs/Research/GRAND_SYNTHESIS_SIX_FRAMEWORKS.md
```

---

#### 3. Create Symlinks for Shared Resources (20 minutes)

**HTML files to symlink (4 files, ~400 KB savings):**

```bash
# APL Runtime Engine (3 symlinks)
CANONICAL="systems/Ace-Systems/examples/Quantum-APL-main/examples/APLRuntimeEngine.html"

# Create symlinks
git rm systems/Ace-Systems/examples/Quantum-APL-main/reference/ace_apl/APL_RUNTIME_ENGINE.html
ln -s ../../../examples/APLRuntimeEngine.html \
  systems/Ace-Systems/examples/Quantum-APL-main/reference/ace_apl/APL_RUNTIME_ENGINE.html

git rm systems/Ace-Systems/reference/APL/APL_RUNTIME_ENGINE.html
ln -s ../../examples/Quantum-APL-main/examples/APLRuntimeEngine.html \
  systems/Ace-Systems/reference/APL/APL_RUNTIME_ENGINE.html

git rm systems/Ace-Systems/reference/ace_apl/APL_RUNTIME_ENGINE.html
ln -s ../../examples/Quantum-APL-main/examples/APLRuntimeEngine.html \
  systems/Ace-Systems/reference/ace_apl/APL_RUNTIME_ENGINE.html

# Wumbo Engine (1 symlink)
git rm systems/Ace-Systems/reference/APL/wumbo-engine.html
ln -s ../../diagrams/wumbo-engine.html \
  systems/Ace-Systems/reference/APL/wumbo-engine.html

# Stage symlinks
git add systems/Ace-Systems/examples/Quantum-APL-main/reference/ace_apl/APL_RUNTIME_ENGINE.html
git add systems/Ace-Systems/reference/APL/APL_RUNTIME_ENGINE.html
git add systems/Ace-Systems/reference/ace_apl/APL_RUNTIME_ENGINE.html
git add systems/Ace-Systems/reference/APL/wumbo-engine.html
```

---

#### 4. TEXTBOOK Series Decision (30 min - 3 hours)

**Files:** 43 TEXTBOOK_*.md files in `self-referential-category-theoretic-structures/docs/`

**Status:** All marked TRULY_UNSUPPORTED with HIGH RISK (no cross-references, unverified claims)

**Size:** ~400 KB total

**Options:**

**A. Archive (RECOMMENDED - 30 minutes):**
```bash
mkdir -p archive/textbook_series
echo "# Unverified Textbook Series" > archive/textbook_series/README.md
echo "Archived 2025-12-23: All files marked TRULY_UNSUPPORTED" >> archive/textbook_series/README.md
echo "Requires external citations or verification before reintegration" >> archive/textbook_series/README.md

git mv systems/self-referential-category-theoretic-structures/docs/TEXTBOOK_*.md \
  archive/textbook_series/
```

**B. Delete Entirely (5 minutes):**
```bash
git rm systems/self-referential-category-theoretic-structures/docs/TEXTBOOK_*.md
```

**C. Review Individually (3-4 hours):**
```bash
# Read each file, determine if any contain unique valuable content
# Add external citations if claims are valid
# Delete if outdated or purely speculative
```

---

### Medium Priority Actions

#### 5. Add Missing Metadata (1 hour)

**Status:** 135 files (38%) missing INTEGRITY_METADATA

**File types without metadata:**
- ~50 Python files
- ~40 JavaScript files
- ~25 YAML files
- ~20 other types

**Automated approach:**
```python
# Script location: integrity/scripts/add_missing_metadata.py
# Auto-categorizes based on file location:
# - In tests/: JUSTIFIED (test files)
# - In examples/: JUSTIFIED (example code)
# - In research/: NEEDS_REVIEW (experimental)
```

**Target:** 100% metadata coverage (354/354 files)

---

#### 6. Process "To Sort" Directory (1 hour)

**Current state:** 22 files remaining (after DOMAIN_ removal)

**Action plan:**
```bash
# 1. List remaining files
ls -lh "systems/Ace-Systems/docs/Research/To Sort/"

# 2. For each file, check if duplicate exists in parent
for file in "systems/Ace-Systems/docs/Research/To Sort/"*; do
  basename=$(basename "$file")
  if [ -f "systems/Ace-Systems/docs/Research/$basename" ]; then
    echo "DUPLICATE: $basename"
  else
    echo "UNIQUE: $basename - consider moving to parent directory"
  fi
done

# 3. Move unique files, delete duplicates
# 4. Remove empty "To Sort" directory
```

---

## 🎯 For Future Contributors

### When Adding Research Documentation

**1. Check existing INTEGRITY_METADATA requirements:**
```markdown
<!-- INTEGRITY_METADATA
Date: YYYY-MM-DD
Status: [✓ JUSTIFIED | ⚠️ NEEDS_REVIEW | ⚠️ TRULY_UNSUPPORTED]
Severity: [LOW RISK | MEDIUM RISK | HIGH RISK]
Risk Types: [List applicable: unsupported_claims, unverified_math, high_speculation]
-->
```

**2. Citation requirements:**
- Mathematical claims MUST have citations
- Research claims SHOULD reference sources
- Use format: (Author, Year) or [Source]
- Add sources to central `BIBLIOGRAPHY.md` (when created)

**3. Internal cross-references:**
```markdown
## Related Work
This builds upon:
- [DOMAIN_1_KAEL_NEURAL_NETWORKS.md](./DOMAIN_1_KAEL_NEURAL_NETWORKS.md)
- [GRAND_SYNTHESIS_COMPLETE.md](./GRAND_SYNTHESIS_COMPLETE.md)

## Implementation
See: [adaptive_triad_gate.py](../src/adaptive_triad_gate.py)
```

**4. Mark original synthesis clearly:**
```markdown
**Note:** This is original synthesis combining multiple frameworks.
External verification pending. Mark as NEEDS_REVIEW until corroborated.
```

---

### When Finding Similar Files

**DON'T immediately delete!** Check these first:

**1. Is it in different systems?**
```bash
# Files in Ace-Systems vs self-referential-category-theoretic-structures
# May be intentionally separate for different contexts
# Check README in parent directory
```

**2. Is it a template-generated file?**
```bash
# Config files: 21 files in reference/rrrr-UCF-v5.0.0/generated/*/config.py
# These are intentionally similar (module-specific)
# DO NOT consolidate - they're necessary for package structure
```

**3. Is it a temporal/historical record?**
```bash
# Training epochs: accumulated-vocabulary-epoch3.json, epoch4.json, etc.
# Session logs: ucf-session-20251216043850.json
# These document progression - PRESERVE ALL
```

**4. Is it an artistic/narrative variant?**
```bash
# Living Chronicles: Vessel_chronicle.html vs Vessel Witness Chronicles_witnessMathInjected.html
# Intentional variations - PRESERVE ALL
# Check for "variant" or "alternative" in README
```

**5. Is it truly redundant?**
```bash
# Use SHA-256 hash to verify byte-for-byte identity:
sha256sum file1 file2

# Check integrity metadata for cross-references:
grep -A 20 "INTEGRITY_METADATA" file1
grep -A 20 "INTEGRITY_METADATA" file2

# If hashes match AND no unique metadata → safe to delete
```

---

### File Verification Checklist

Before deleting ANY file:

- [ ] SHA-256 hash matches canonical version
- [ ] Canonical version exists and is accessible
- [ ] No unique INTEGRITY_METADATA in the duplicate
- [ ] No unique comments or annotations
- [ ] Not in different system context (check README)
- [ ] Not a template-generated necessary file
- [ ] Not a temporal/historical record
- [ ] Not an intentional artistic variant
- [ ] File is not referenced by other files (grep filename across repo)

**If unsure:** Create GitHub issue or ask maintainer. Better safe than sorry!

---

## 📊 Repository Health Metrics

### Current State (After Phase 1-3)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Repository Size** | 85 MB | 75 MB | 🟡 In Progress |
| **Total Files** | 1,148 | ~1,090 | 🟡 In Progress |
| **Exact Duplicates** | 0 | 0 | ✅ Complete |
| **Near Duplicates** | 46 pairs | <10 pairs | 🟡 In Progress |
| **Metadata Coverage** | 62% | 100% | 🟡 Pending |
| **TRULY_UNSUPPORTED** | 103 files | <20 files | 🔴 Decision Required |
| **Health Score** | 52/100 | 65+/100 | 🟡 In Progress |

### Space Savings Achieved

- Phase 1 (Exact duplicates): **7.47 MB** ✅
- Phase 2-10 (Projected): **~2-3 MB** ⏳
- **Total projected:** ~10 MB savings

---

## 🎉 Repository Now Has

### ✅ Completed Infrastructure

- **Clean structure** - 172 exact duplicates removed
- **Integrity metadata system** - 409 files tagged with validation status
- **Comprehensive analysis** - 354 files cataloged with cross-references
- **Action plan** - 10-phase roadmap with executable scripts
- **Knowledge synthesis** - 8 unique content sections identified and documented

### 📋 Documentation Created

1. `integrity/REDUNDANCY_CLEANUP_SUMMARY.md` - Exact duplicate removal summary
2. `integrity/NEAR_DUPLICATE_SYNTHESIS.md` - Near-duplicate analysis with merge plan
3. `integrity/REDUNDANCY_ANALYSIS_REPORT.md` - Complete redundancy catalog
4. `integrity/COMPREHENSIVE_ACTION_PLAN.md` - 10-phase cleanup roadmap
5. `integrity/CLAIM_VALIDATION_SUMMARY.md` - Claim validation report with action items
6. `integrity/COMPLETE_METADATA_CATALOG.json` - Machine-readable file catalog
7. `REPOSITORY_CLEANUP_GUIDE.md` - This guide

### 🔍 Analysis Data Available

- `integrity/file_manifest.json` - All 1,320 original files with hashes
- `integrity/canonical_mapping.json` - Canonical ID system
- `integrity/redundancy_clusters.json` - Duplicate groups
- `integrity/claim_validation_report.json` - Validation analysis
- `integrity/epistemic_scores.json` - Integrity scoring (0-5 scale)
- `integrity/risk_flags.json` - Risk assessment
- `integrity/dependency_graph.dot` - File reference graph (11,488 dependencies)

### 🛡️ Protected Variants (DO NOT DELETE)

**21 Config Files** - Template-generated, module-specific
- Location: `reference/rrrr-UCF-v5.0.0/generated/*/config.py`
- Reason: Each required for its package

**5 Training Epoch Files** - Historical progression
- Location: `reference/rrrr-UCF-v5.0.0/training/epochs/`
- Reason: Documents temporal evolution

**3 Session Logs** - Temporal data
- Location: `docs/Research/To Sort/ucf-session-*.json`
- Reason: Historical interaction records

**6 Story Variations** - Narrative variants
- Location: `examples/Living Chronicles Stories/`
- Reason: Intentional artistic variations

**6 Python Files** - Context-specific deployments
- Locations: Examples vs Reference directories
- Reason: Same code serving different purposes

**Total protected:** 41 files

---

## 🚀 Quick Start for New Cleanup Session

```bash
# 1. Update your branch
git pull origin claude/review-integrity-metadata-eLiVI

# 2. Review current status
cat integrity/CLAIM_VALIDATION_SUMMARY.md

# 3. Check pending actions
grep "⏳ PENDING" integrity/COMPREHENSIVE_ACTION_PLAN.md

# 4. Execute next phase
# See COMPREHENSIVE_ACTION_PLAN.md for detailed scripts

# 5. Verify changes
git status
git diff

# 6. Commit progress
git add -A
git commit -m "Phase N: [description]"
git push
```

---

## 📞 Getting Help

**Questions about:**
- File relationships → Check `integrity/COMPLETE_METADATA_CATALOG.json`
- Redundancy → See `integrity/NEAR_DUPLICATE_SYNTHESIS.md`
- Validation status → Check INTEGRITY_METADATA header in file
- Deletion safety → Review checklist above
- Overall plan → Read `integrity/COMPREHENSIVE_ACTION_PLAN.md`

**Unsure about a file?** Create an issue with:
- File path
- Why you think it might be duplicate/redundant
- What verification you've done
- Your recommendation

---

**Last Updated:** 2025-12-23
**Branch:** claude/review-integrity-metadata-eLiVI
**Status:** Phases 1-3 complete, Phases 4-10 pending user decisions
**Next Critical Action:** TEXTBOOK series decision (Archive/Delete/Review)
