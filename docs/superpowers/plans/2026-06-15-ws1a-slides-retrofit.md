# WS1a — Slides Retrofit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]`.

**Goal:** Unify the 4 slide decks on one design-A language and embed the real WS2 charts/diagrams, without rewriting content (WS3 already made content correct).

**Key finding:** Reports 2/3/4 already share a design-A-aligned token set (`--accent-green: #15803d`, `--font-sans: Plus Jakarta Sans`, off-white bg, sharp corners). Report 1 is the outlier (Inter + cool `--accent #e8ecf1`). So the retrofit is mostly: (a) bring Report 1 onto the 2/3/4 system, (b) embed new charts/diagrams, (c) drop section-number eyebrows, (d) add JetBrains Mono for stats.

**Verification:** render decks to PNG with `qlmanage -t -s 1280 -o /tmp <file>.html` and inspect; validate HTML stays well-formed.

**Assets available** (in `presentations/`): `chart_ocr_benchmark.png`, `chart_color_benchmark.png`, `chart_plate_benchmark.png`, `chart_dataset.png`, `diagram_pipeline.svg`, `diagram_decision.svg`, `diagram_pivot.svg`, `diagram_architecture.svg`.

---

### Task 1: Report 1 — adopt the 2/3/4 design system

**File:** `presentations/Report_1_Presentation.html`

- [ ] **Step 1:** Replace the font `@import` (line ~9) `...family=Inter...` with the 2/3/4 import:
  `@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,600;0,700;1,400&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');`
- [ ] **Step 2:** Replace Report 1's `:root` block with the unified token set (matching 2/3/4, plus mono):
  ```css
  :root {
      --bg-color: #fbfbf9;
      --text-color: #1a1a17;
      --text-muted: #5e5e57;
      --border-color: #2e2e2a;
      --accent-red: #991b1b;
      --accent-orange: #c2410c;
      --card-bg: #ffffff;
      --accent-green: #15803d;
      --font-editorial: 'Lora', serif;
      --font-sans: 'Plus Jakarta Sans', sans-serif;
      --font-mono: 'JetBrains Mono', monospace;
      /* back-compat aliases used elsewhere in this deck */
      --accent: #15803d;
      --text: #1a1a17;
  }
  ```
  Keep `--accent` and `--text` as aliases so existing `var(--accent)`/`var(--text)` references in the deck still resolve.
- [ ] **Step 3:** Replace `font-family: 'Inter', sans-serif;` occurrences with `font-family: var(--font-sans);` (grep: `grep -n "'Inter'" presentations/Report_1_Presentation.html`).
- [ ] **Step 4:** Verify no `Inter` remains: `grep -c "Inter" presentations/Report_1_Presentation.html` → 0. Render: `qlmanage -t -s 1280 -o /tmp presentations/Report_1_Presentation.html` and inspect the title + a content slide.
- [ ] **Step 5:** Commit `presentations/Report_1_Presentation.html` with message `feat(slides): unify Report 1 onto design-A system (WS1a)`.

---

### Task 2: Add JetBrains Mono import to Reports 2/3/4

**Files:** `presentations/Report_2_Presentation.html`, `Report_3_Presentation.html`, `Report_4_Presentation.html`

- [ ] **Step 1:** In each, append `&family=JetBrains+Mono:wght@500;700` to the existing Google Fonts `@import` URL (before `&display=swap`), and add `--font-mono: 'JetBrains Mono', monospace;` to each `:root`.
- [ ] **Step 2:** Verify each import contains `JetBrains+Mono`. Render each with qlmanage to confirm no breakage.
- [ ] **Step 3:** Commit the three files: `feat(slides): add JetBrains Mono token to Reports 2-4 (WS1a)`.

---

### Task 3: Embed WS2 charts/diagrams

**Files:** all 4 decks (per relevance)

Locate slides that currently use placeholder/old images or describe these topics, and embed the new asset via `<img src="...">` (relative path, same dir). Map:
- Report 1 architecture slide → `diagram_pipeline.svg` and/or `diagram_architecture.svg`.
- Report 2 dataset slide → `chart_dataset.png` (replace `img_dataset.png` reference if present).
- Report 3 OCR results slide → `chart_ocr_benchmark.png`; colour results → `chart_color_benchmark.png`; plate detection → `chart_plate_benchmark.png`.
- Report 4 → `diagram_pivot.svg` on the "Đã thay đổi gì" / pivot slide; `diagram_decision.svg` on the decision-logic slide; the benchmark slide can show `chart_ocr_benchmark.png`.

- [ ] **Step 1:** For each deck, grep existing `<img` tags and the slide headings; insert the new `<img style="max-width:100%; border-radius:12px;">` into the matching slide's body (reuse the slide's existing image container/figure markup). Do not delete content; add or swap the image source.
- [ ] **Step 2:** Render each deck with qlmanage; confirm the embedded images appear and are not broken.
- [ ] **Step 3:** Commit per deck: `feat(slides): embed WS2 charts/diagrams in Report N (WS1a)`.

---

### Task 4: Drop section-number eyebrows

**Files:** Reports 3 & 4 (which use `NN / Topic` subtitles)

- [ ] **Step 1:** Grep `grep -n 'slide-subtitle' presentations/Report_3_Presentation.html presentations/Report_4_Presentation.html`. For subtitles of the form `NN / Topic` (e.g. `05 / Thiết kế mô hình`), remove the leading `NN / ` so only the topic remains. Leave non-numbered subtitles alone. Keep at most the existing slide-number badge (top-right) — that is fine; the eyebrow numbering is the redundant part.
- [ ] **Step 2:** Verify: `grep -nE '>[0-9]{2} ?/' presentations/Report_3_Presentation.html presentations/Report_4_Presentation.html` → no numbered eyebrows remain.
- [ ] **Step 3:** Commit: `feat(slides): drop section-number eyebrows (WS1a)`.

---

### Task 5: Final visual + structural check

- [ ] **Step 1:** Render all 4 decks with qlmanage; inspect title + 2 content slides each for unified look (same fonts, off-white bg, green accent, embedded charts).
- [ ] **Step 2:** Validate HTML well-formedness for all 4 (no unclosed tags introduced): a quick `python3 -c "import html.parser..."` smoke or browser-open check.
- [ ] **Step 3:** Confirm zero broken image refs: every `<img src>` resolves to an existing file in `presentations/`.

---

## Self-Review
- Report 1 outlier fixed (Task 1) — the audit's #1 issue. ✓
- JetBrains Mono available for stats (Task 2). ✓
- Real WS2 charts/diagrams embedded (Task 3). ✓
- Section-number eyebrows dropped (Task 4). ✓
- Fonts: kept on the existing Google-Fonts `@import` mechanism (consistent across all 4 now). True self-hosting for full offline is deferred as a separate hardening step (noted, not blocking — slides currently render online; the project's offline thesis is about the runtime, and the decks already depended on Google Fonts).
- Content untouched (WS3 already correct). ✓
