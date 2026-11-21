---
name: Lucid
colors:
  surface: '#faf8ff'
  surface-dim: '#d2d9f4'
  surface-bright: '#faf8ff'
  surface-container-lowest: '#ffffff'
  surface-container-low: '#f2f3ff'
  surface-container: '#eaedff'
  surface-container-high: '#e2e7ff'
  surface-container-highest: '#dae2fd'
  on-surface: '#131b2e'
  on-surface-variant: '#434655'
  inverse-surface: '#283044'
  inverse-on-surface: '#eef0ff'
  outline: '#737686'
  outline-variant: '#c3c6d7'
  surface-tint: '#0053db'
  primary: '#004ac6'
  on-primary: '#ffffff'
  primary-container: '#2563eb'
  on-primary-container: '#eeefff'
  inverse-primary: '#b4c5ff'
  secondary: '#6b38d4'
  on-secondary: '#ffffff'
  secondary-container: '#8455ef'
  on-secondary-container: '#fffbff'
  tertiary: '#006229'
  on-tertiary: '#ffffff'
  tertiary-container: '#007e37'
  on-tertiary-container: '#c1ffc5'
  error: '#ba1a1a'
  on-error: '#ffffff'
  error-container: '#ffdad6'
  on-error-container: '#93000a'
  primary-fixed: '#dbe1ff'
  primary-fixed-dim: '#b4c5ff'
  on-primary-fixed: '#00174b'
  on-primary-fixed-variant: '#003ea8'
  secondary-fixed: '#e9ddff'
  secondary-fixed-dim: '#d0bcff'
  on-secondary-fixed: '#23005c'
  on-secondary-fixed-variant: '#5516be'
  tertiary-fixed: '#6bff8f'
  tertiary-fixed-dim: '#4ae176'
  on-tertiary-fixed: '#002109'
  on-tertiary-fixed-variant: '#005321'
  background: '#faf8ff'
  on-background: '#131b2e'
  surface-variant: '#dae2fd'
typography:
  display-hero:
    fontFamily: Inter
    fontSize: 48px
    fontWeight: '700'
    lineHeight: 56px
    letterSpacing: -0.025em
  display-hero-mobile:
    fontFamily: Inter
    fontSize: 34px
    fontWeight: '700'
    lineHeight: 42px
    letterSpacing: -0.02em
  headline-lg:
    fontFamily: Inter
    fontSize: 32px
    fontWeight: '600'
    lineHeight: 40px
    letterSpacing: -0.02em
  headline-lg-mobile:
    fontFamily: Inter
    fontSize: 26px
    fontWeight: '600'
    lineHeight: 34px
    letterSpacing: -0.015em
  headline-md:
    fontFamily: Inter
    fontSize: 24px
    fontWeight: '600'
    lineHeight: 32px
    letterSpacing: -0.015em
  headline-sm:
    fontFamily: Inter
    fontSize: 20px
    fontWeight: '600'
    lineHeight: 28px
    letterSpacing: -0.01em
  body-lg:
    fontFamily: Inter
    fontSize: 17px
    fontWeight: '400'
    lineHeight: 28px
    letterSpacing: -0.005em
  body-md:
    fontFamily: Inter
    fontSize: 15px
    fontWeight: '400'
    lineHeight: 24px
    letterSpacing: 0em
  body-sm:
    fontFamily: Inter
    fontSize: 13px
    fontWeight: '400'
    lineHeight: 20px
    letterSpacing: 0.005em
  label-md:
    fontFamily: Inter
    fontSize: 14px
    fontWeight: '500'
    lineHeight: 20px
    letterSpacing: 0em
  label-sm:
    fontFamily: Inter
    fontSize: 12px
    fontWeight: '500'
    lineHeight: 16px
    letterSpacing: 0.01em
  label-xs:
    fontFamily: Inter
    fontSize: 11px
    fontWeight: '600'
    lineHeight: 14px
    letterSpacing: 0.02em
rounded:
  sm: 0.5rem
  DEFAULT: 1rem
  md: 1.5rem
  lg: 2rem
  xl: 3rem
  full: 9999px
spacing:
  gutter: 1.5rem
  margin: 2rem
  gutter-mobile: 1rem
  margin-mobile: 1rem
  space-xs: 0.25rem
  space-sm: 0.5rem
  space-md: 1rem
  space-lg: 1.5rem
  space-xl: 2.5rem
---

## Brand & Style

This design system establishes an authoritative yet welcoming consumer interface for news verification and automated fact-checking. The design language synthesizes the computational clarity of Perplexity AI (60%), the structured document serenity of Notion (20%), and the tactile restraint and precision of Apple Human Interface Guidelines (20%).

### Personality & Emotional Resonance
The interface must project radical clarity, calm objectivity, and effortless trust. It eschews the sensationalism of breaking news graphics, the visual paranoia of dark cybersecurity consoles, and the clinical abstraction of dense analytical charts. Instead, it invites users into a focused reading room where complex investigative threads and claim validations appear human, orderly, and immediately decipherable.

### Aesthetic Foundation
- **Modern Minimal Realism:** High-luminance canvases, expansive whitespace, and delicate structural delineation. Information density is managed through progressive disclosure rather than sensory overload.
- **Micro-tactile Nuance:** Glass-like translucency on sticky headers, featherweight hairline borders (1px solid slate), and soft diffused ambient shadows that honor physical spatial logic without sliding into heavy skeuomorphism.
- **Clarity of Verdict:** Information state takes absolute precedence. Verification badges, citation tags, and source credibility indicators are rendered with absolute legibility, ensuring verification status can be comprehended in a single glance.

## Colors

The palette balances authoritative utility with serene clarity. Centered around a core light mode, it utilizes calibrated slate neutrals to frame clean semantic signals.

### Core Roles
- **Primary (`#2563EB` — Trust Blue):** Anchors core actions, verified source anchors, query focus states, and foundational brand emphasis. Communicates intellectual rigor and digital stability.
- **Secondary (`#8B5CF6` — Analysis Purple):** Signifies synthetic synthesis, AI rationale, linguistic extraction, and deep contextual grounding.
- **Tertiary / Success (`#22C55E` — Verified Green):** Denotes factual affirmation, authenticated provenance, and uncontested assertions.
- **Neutral Core (`#0F172A` — Deep Slate Text):** The primary typographic anchor, providing high-contrast readability against pale grounds without the optical fatigue of raw `#000000`.

### Functional Semantic Palette
- **Warning / Review Needed (`#F59E0B`):** Contextually missing, partially true, or disputed claims. Always paired with `#FFFBEB` background fills and `#B45309` text.
- **Destructive / Likely False (`#EF4444`):** Fabricated data, manipulated media, and debunked rumors. Supported by `#FEF2F2` tinted badges.
- **Canvas Ground (`#F8FAFC`):** A cool, low-strain off-white canvas that ensures white content cards visually elevate without aggressive drop shadows.
- **Surface Elevation (`#FFFFFF`):** Reserved for interactive claim panels, context cards, modal dialogs, and deep dive summaries.
- **Muted Text / Border (`#64748B` / `#E2E8F0`):** Secondary meta-information, timestamps, domain breadcrumbs, and structural 1px dividers.

## Typography

The typographic hierarchy relies solely on Inter, engineered to maximize objective readability and tabular numeric alignment across varied text weights.

### Structural Typographic Rules
- **Display & Headlines:** Tightly tracked headings (`-0.025em` to `-0.01em`) replicate Apple-style crispness. Used for investigation conclusions, landing value propositions, and verification headers.
- **Editorial Body Text:** Generous line heights (`1.6x` on `body-lg` and `body-md`) create comfortable reading experiences comparable to long-form Notion notes, preventing eye strain during source cross-examination.
- **Labels & Micro-copy:** Set with refined medium weights and subtle positive tracking for instantaneous recognition in pill badges, breadcrumbs, source pill lists, and meta indicators.
- **Tabular Numerals:** For certainty scores, timestamps, and confidence percentages, enable OpenType feature `cv05` (tabular numbers) to keep comparative metrics vertically aligned.

## Layout & Spacing

The layout structure leverages a focused fluid grid capped by an Apple-inspired readable column constraint. Fact-checking requires visual calm; layouts prioritize single-column reading paths flanked by contextual evidence panels.

### Grid Architecture
- **Desktop (≥ 1024px):** 12-column layout with max-width container set to `1180px` for optimal eye scanning. Analysis views use an asymmetric 7:5 ratio (7 columns for the core synthesis story and primary text, 5 columns for source citations, credibility breakdown, and primary sources).
- **Tablet (768px – 1023px):** 8-column layout with `1.5rem` gutters and margins. Side evidence rails tuck into horizontal sliding trays or drop below the primary verdict.
- **Mobile (≤ 767px):** 4-column layout with `1rem` margins. All multi-panel comparison cards reflow into a clean vertical stack.

### Spatial Rhythm
- **Internal Padding:** Base UI elements are spaced via an 8pt conceptual rhythm. Component interior padding remains compact (`space-sm` to `space-md`), keeping the visual hierarchy light and precise.
- **Section Isolation:** Generous vertical rhythm (`space-xl` to double `space-xl`) between disparate claim threads to prevent cognitive bleeding between distinct inquiries.

## Elevation & Depth

Visual hierarchy is articulated through tonal separation, high-precision hairline borders, and subtle ambient light simulation rather than distinct drop shadows.

### Elevation Levels
- **Canvas Base (Level 0):** Pure `#F8FAFC`. Houses background surfaces, page roots, and unselected filter chips.
- **Elevated Surfaces (Level 1):** Pure `#FFFFFF` surfaces paired with a micro-border `1px solid rgba(226, 232, 240, 0.8)`. Shadow treatment is diffuse and imperceptible: `0 1px 3px 0 rgba(15, 23, 42, 0.03), 0 1px 2px -1px rgba(15, 23, 42, 0.02)`.
- **Floating Overlays & Menus (Level 2):** Pure `#FFFFFF` surfaces layered with `0 10px 25px -5px rgba(15, 23, 42, 0.06), 0 8px 10px -6px rgba(15, 23, 42, 0.04)`. Used for verification filters, citation detail popovers, and share sheets.
- **Apple Glass Layer (Sticky Surfaces):** Used on persistent verification search bars and top navigations. Backdrop filter blur of `20px` with an 80% opacity fill (`rgba(255, 255, 255, 0.8)`) and a bottom hairline divider `rgba(226, 232, 240, 0.6)`.

## Shapes

The visual personality combines Notion's modularity with Apple's iconic pill-geometry.

### Shape Hierarchy
- **Pill Primitives (`rounded-full` / 9999px):** Primary interactive elements including search execution controls, verification status badges, source domain chips, and query filter pills.
- **Container Geometry (`rounded-xl` / 1.5rem):** Core cards, fact-check insight containers, media inspection boxes, and deep-dive analysis blocks.
- **Micro Elements (`rounded-md` / 0.5rem):** Form inputs, inline source footnotes `[1]`, citation cards within lists, and icon backgrounds.

## Components

### Buttons & Interactive Controls
- **Primary Query Button:** True pill-shape (`rounded-full`), `#2563EB` solid background, white text (`label-md`), `padding: 0.625rem 1.25rem`. Subtle active depression scale effect (`scale: 0.98`).
- **Secondary / Ghost Button:** Transparent background, `1px solid #E2E8F0`, `#0F172A` text, hover state transitions into `#F1F5F9`.
- **AI Action Button:** Gradient stroke or subtle lavender fill (`#F5F3FF`) with `#8B5CF6` text and an integrated spark icon.

### Status Badges (Fact-Check Verdicts)
- **Verified:** Background `#ECFDF5`, border `1px solid #A7F3D0`, text `#15803D`, accompanied by a solid checkmark icon.
- **Needs Review:** Background `#FFFBEB`, border `1px solid #FDE68A`, text `#B45309`, accompanied by an informational yield icon.
- **Likely False:** Background `#FEF2F2`, border `1px solid #FECACA`, text `#B91C1C`, accompanied by an alert cross icon.
- **Geometry:** Height 24px, pill rounded, typography `label-xs`, padding `0 10px`.

### Source & Citation Chips
- **Citation Anchors (Inline):** Subtle numeric superscripts formatted as pill tags (`#EFF6FF`, text `#2563EB`, font size `11px`, padding `1px 6px`, `rounded-full`). Hovering prompts an elevated preview card.
- **Domain Evidence Chips:** Mini-pill badge displaying the source's verified favicon, publisher name, and reliability score (e.g., `Reuters • 99%`), enclosed in `1px solid #E2E8F0` on white.

### Fact-Checking Cards
- **Structure:** Surface `#FFFFFF`, border `1px solid #E2E8F0`, generous padding (`space-lg`), corner radius `rounded-xl`.
- **Anatomy:** 
  1. Top Row: Original claim author + claim timestamp + Status Badge.
  2. Claim Statement: `headline-sm`, high contrast `#0F172A`.
  3. Synthesis / Context: `body-md` in `#334155`.
  4. Footer: Referenced sources row + Perplexity-style "View Reasoning Steps" disclosure accordion.

### Input Fields & Search Bars
- **Central Query Bar:** Oversized pill input bar (`rounded-full`), height `56px`, `#FFFFFF` fill with an outer hairline ring `#E2E8F0`, transitioning on focus to a glowing ring `0 0 0 3px rgba(37, 99, 235, 0.15)` and border `#2563EB`. Contains search icon, clear button, and a prominent round execute icon.
- **Standard Input:** Height `42px`, `rounded-lg`, typography `body-md`, background `#FFFFFF`, border `#CBD5E1`.

### Verification Timelines & Accordions
- **Reasoning Process Steps:** Clean vertical border spine in `#E2E8F0` with subtle dot nodes indicating verification checkpoints: *Searching Sources* → *Evaluating Divergence* → *Synthesizing Consensus*. Pure Apple-style simplicity with no technical noise.