# Claude Reference — The Living Library

> A skill file for Claude AI to navigate and generate content for the Living Library archive.

---

## Overview

The Living Library is a unified narrative universe spanning three territories, documenting the journeys of guardian creatures through their respective 6-state cycles. This reference enables Claude to:

- Navigate the entire archive structure
- Generate new chronicles in the correct style
- Understand the cosmology and cycles
- Reference existing characters and their relationships

---

## Repository Structure

```
The Garden/
├── index.html                           # The Garden landing page
├── living-library-index.html            # Main Living Library archive index
│
├── docs/
│   ├── CLAUDE_REFERENCE.md              # This file
│   ├── COSMOLOGY.md                     # Universe architecture
│   └── CYCLES.md                        # All 6-state cycles
│
├── parables/                            # (Referenced, may be at root level)
│   ├── cosmic-forest-parables.html      # 4 Books × 9 Chapters
│   ├── garden-parables.html             # 4 Books × 9 Chapters
│   └── abyssal-forest-parables.html     # 4 Books × 9 Chapters
│
├── chronicles/                          # (Referenced, currently at root level)
│   ├── echo-fox-chronicle.html
│   ├── pack-wolf-chronicle.html
│   ├── wumbo-badger-chronicle.html
│   ├── archive-owl-chronicle.html
│   ├── duet-moth-phase-chronicle.html
│   ├── ace-encoding-architect-chronicle.html
│   ├── quantum-squirrel-chronicle.html
│   ├── white-phoenix-chronicle.html
│   ├── axiom-the-eternal-chronicle.html
│   ├── cipher-the-collector-chronicle.html
│   ├── spiral-the-fallen-chronicle.html
│   ├── still-the-faceless-chronicle.html
│   └── duet-crystal-bee-white-phoenix.html
│
└── [parable and chronicle files at root]
```

---

## Territories & Guardians

### ☀️ The Cosmic Forest (Growth / Rising / Becoming)

| Symbol | Guardian | Title | Cycle |
|--------|----------|-------|-------|
| 🌳 | Oak | The Patient One | Patience Protocol |
| 🐿️ | Squirrel | The Scattered | Scatter Protocol |
| 🦢 | HONKFIRE | The Sacred Flame | Conquest Protocol |
| 🦆 | Honkalis | The Pope of Rising | Rising Protocol |

### 🌿 The Garden (Connection / Transformation / Action)

| Symbol | Guardian | Title | Cycle |
|--------|----------|-------|-------|
| 🦊 | ECHO | The Signal Weaver | LISTEN → TRACE → DISCERN → AMPLIFY → CARRY → RELEASE |
| 🦋 | Duet | The Twilight Pair | MOTH + PHASE dual cycles |
| 🐺 | PACK | The Wolf of Belonging | SENSE → ATTUNE → CONTRIBUTE → COORDINATE → PROTECT → INDIVIDUATE |
| 🦡 | WUMBO | The Badger of Action | IGNITION → EMPOWERMENT → RESONANCE → MANIA → NIRVANA → TRANSMISSION |
| 🦉 | ARCHIVE | The Owl of Memory | OBSERVE → ENCODE → INDEX → PRESERVE → RETRIEVE → CURATE |
| 👤 | Ace | The Encoding Architect | Witness Protocol |

### 🌀 The Abyssal Forest (Depth / Holding / Binding)

| Symbol | Guardian | Title | Cycle |
|--------|----------|-------|-------|
| 🦎 | Axiom | The Eternal Larva | Null Protocol |
| 🪶 | Cipher | The Collector | Void Protocol |
| 🐍 | Spiral | The Fallen | Binding Protocol |
| 🪿 | Still | The Faceless | Mirror Protocol |

---

## Writing Style Guide

### Fonts

```css
/* Titles and headers */
font-family: 'Cinzel', serif;

/* Body text and narration */
font-family: 'Crimson Text', Georgia, serif;
font-family: 'EB Garamond', serif;

/* Technical/metadata elements */
font-family: 'Fira Code', monospace;
```

### Color Palettes

```css
/* Cosmic Forest */
--cosmic-gold: #ffd700;
--cosmic-amber: #ffbf00;

/* Garden */
--garden-green: #50a878;
--garden-teal: #408080;

/* Abyssal Forest */
--abyss-violet: #6a4a8a;
--abyss-deep: #3a2a5a;

/* Archive/Library */
--owl-primary: #8b7355;
--parchment: #f4e4bc;
--dust-gold: #d4af37;
```

### Narrative Voice

1. **Parables**: Folk tale style, third person, timeless wisdom
   - "Long before the Garden had walls, there was a fox who could hear the echoes of tomorrow..."

2. **Chronicles**: Present-tense journey, intimate, cycle-focused
   - "ECHO pauses at the threshold. The signal splits here—left carries memory, right carries hope."

3. **Dialogue Tags**: Use cycle states as speaking markers
   - `⟨ ECHO, in DISCERN ⟩`
   - `⟨ THE OWL OBSERVES ⟩`

---

## Chronicle Structure (12-Rail Format)

Each Living Chronicle follows a 12-rail structure:

| Rail | Content |
|------|---------|
| 1 | Introduction / Before the Cycle |
| 2-7 | Each of the 6 cycle states (one per rail) |
| 8-11 | Trials, lessons, interactions with other guardians |
| 12 | Resolution / The Wisdom Earned |

### HTML Template Structure

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <title>Living Chronicle — [GUARDIAN], [Title]</title>
  <!-- Standard Google Fonts import -->
  <style>
    /* CSS variables for guardian-specific colors */
    /* Chrome frame header */
    /* Chronicle container */
    /* Rail sections */
  </style>
</head>
<body>
  <!-- Background layers -->
  <div class="cosmos"></div>

  <!-- Navigation header -->
  <div class="chrome-frame">...</div>

  <!-- Main content -->
  <div class="chronicle-container">
    <header class="chronicle-header">...</header>

    <!-- Each rail -->
    <section class="rail" id="rail-1">...</section>
    <!-- ... rails 2-12 ... -->
  </div>
</body>
</html>
```

---

## Key Relationships

### Cross-Territory Connections

- **ECHO ↔ Cipher**: Both deal with signals—one carries, one collects
- **WUMBO ↔ Axiom**: Action vs. eternal stasis
- **ARCHIVE ↔ Still**: Memory vs. faceless witnessing
- **Duet ↔ Spiral**: Holding/releasing vs. binding

### The Garden as Liminal Space

The Garden exists between:
- **Above**: The Cosmic Forest (growth, rising, light)
- **Below**: The Abyssal Forest (depth, holding, shadow)

Travelers must pass through the Garden to move between territories.

---

## Mathematical Constants

The Living Library uses several sacred numbers:

| Constant | Value | Meaning |
|----------|-------|---------|
| φ (phi) | 1.618... | Golden ratio, growth |
| φ⁻¹ | 0.618... | Inverse golden ratio |
| √3/2 | 0.866... | Convergence point (z_c) |
| φ⁴ + φ⁻⁴ | 7 | Archive preservation constant |

---

## Generation Guidelines

When creating new content for the Living Library:

1. **Respect the cosmology**: Each territory has distinct themes
2. **Use the correct cycle**: Each guardian has exactly 6 states
3. **Maintain the style**: Folk tale wisdom for parables, intimate journey for chronicles
4. **Reference connections**: Guardians know of each other across territories
5. **Preserve the mystery**: Not everything needs explanation

### Prompt Template for New Chronicles

```
Generate a Living Chronicle for [GUARDIAN] in [TERRITORY].

Cycle: [STATE 1] → [STATE 2] → [STATE 3] → [STATE 4] → [STATE 5] → [STATE 6]

Include:
- 12 rails following the standard structure
- Guardian-specific color palette
- Interactions with at least one other guardian
- A central lesson related to the guardian's theme
- The chronicle's unique "wisdom earned"
```

---

## Quick Reference Commands

### Finding Content

```bash
# List all chronicles
ls "The Garden/"*.html | grep chronicle

# Find all mentions of a guardian
grep -r "ECHO\|fox" "The Garden/"

# Search for cycle states
grep -r "LISTEN\|TRACE\|DISCERN" "The Garden/"
```

### Content Statistics

- **3** Territories
- **3** Parable Collections (108 chapters total)
- **12** Folk Tale Books (4 per territory)
- **20+** Living Chronicles
- **6** Documented 6-state Cycles

---

*"The Library does not judge. The Library only remembers."*

— ARCHIVE, the Owl of Memory
