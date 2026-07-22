# Swiss R3 + R4 Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the separate Swiss R3 and R4 presentations with one 35-slide Report 4 defense deck that uses the R3 Lemon Green design, includes 33 substantive R3+R4 slides, and has one synchronized speaker script.

**Architecture:** Use the current `swiss_r3/index.html` as the visual and runtime base, preserving its CSS, navigation, motion, index, and low-power behavior. Rebuild only the slide section and presentation metadata, reuse evidence assets from both decks, then place the consolidated deck, outline, and script under `swiss_r4/`. Validate structural invariants with a small Node test and the Swiss validator, then inspect a rendered screenshot of every slide.

**Tech Stack:** Single-file HTML/CSS/JavaScript presentation, existing Swiss slide runtime, Node.js validation scripts, Playwright/Chromium for visual QA, Markdown for outline and script.

---

## File Map

- Replace: `reports/presentations/swiss_r4/index.html` — final 35-slide presentation.
- Replace: `reports/presentations/swiss_r4/OUTLINE.md` — exact final slide map.
- Create: `reports/presentations/swiss_r4/SCRIPT.md` — authoritative 35-slide Vietnamese talk track.
- Create: `reports/presentations/swiss_r4/validate-deck.mjs` — local structural and asset validator.
- Delete: `reports/presentations/swiss_r3/` — superseded presentation and copied assets.
- Delete: `reports/presentations/Report_3_Script.md` — superseded script.
- Delete: `reports/presentations/Report_4_Script.md` — superseded script.
- Preserve: `reports/presentations/_swiss_capture/` and other source assets.

### Task 1: Establish the final asset set and structural test

**Files:**
- Create: `reports/presentations/swiss_r4/validate-deck.mjs`
- Modify: `reports/presentations/swiss_r4/images/`

- [ ] **Step 1: Inventory every asset referenced by both decks**

Run:

```bash
rg -o 'images/[^"'\'' )]+' reports/presentations/swiss_r3/index.html reports/presentations/swiss_r4/index.html | sort -u
```

Expected: a deduplicated list of image references from both source decks.

- [ ] **Step 2: Consolidate only referenced final assets**

Copy the required R3 evidence assets into `reports/presentations/swiss_r4/images/`
with stable semantic names. Retain the R4 demo, architecture, API, and pivot
evidence. Remove files that are not referenced by the final deck.

- [ ] **Step 3: Write the structural validator**

Create a Node ES module that reads `index.html`, `OUTLINE.md`, and `SCRIPT.md` and
asserts:

```js
const sections = [...html.matchAll(/<section class="slide\b/g)];
assert.equal(sections.length, 35);
assert.equal([...html.matchAll(/data-layout="(?:S\d{2})"/g)].length, 33);
assert.equal((outline.match(/^\| \d{2} \|/gm) || []).length, 35);
assert.equal((script.match(/^## Slide \d{2}:/gm) || []).length, 35);
assert.equal([...new Set([...script.matchAll(/^## Slide (\d{2}):/gm)].map(m => m[1]))].length, 35);
assert(!/\[必填\]|TBD|TODO/.test(html + outline + script));
```

The validator must also extract local `images/...` references from the HTML and
fail if any referenced file is missing.

- [ ] **Step 4: Run the validator before implementation**

Run:

```bash
node reports/presentations/swiss_r4/validate-deck.mjs
```

Expected: FAIL because the old R4 deck and scripts do not meet the new 35-slide
contract.

- [ ] **Step 5: Commit the validator and asset preparation**

```bash
git add -- reports/presentations/swiss_r4/validate-deck.mjs reports/presentations/swiss_r4/images
git commit -m "test: define consolidated Swiss deck contract"
```

### Task 2: Build the 35-slide consolidated HTML deck

**Files:**
- Replace: `reports/presentations/swiss_r4/index.html`

- [ ] **Step 1: Copy the R3 presentation runtime as the final visual base**

Use the complete R3 file as the starting point so the final deck inherits:

- Lemon Green variables, including `--accent:#C5E803` and black `--accent-on`
- R3 typography, spacing, slide chrome, navigation, keyboard, index, animation,
  and low-power code
- registered Swiss layout CSS already exercised in R3

Change the document title and deck-level metadata to Report 4 Final Defense.

- [ ] **Step 2: Replace the slide region with the approved 35-slide sequence**

Implement exactly the slide map in
`docs/superpowers/specs/2026-07-21-swiss-r3-r4-consolidation-design.md`.
Slides 02–34 must each use a registered `data-layout="Sxx"`. Slide 01 is the R3
accent cover and slide 35 is the R3-style split closing.

Use a varied rhythm with at least eight distinct layout IDs. Do not use the same
main structure three times consecutively. Keep every main title above its content
axis and reserve the bottom navigation safe area.

- [ ] **Step 3: Preserve factual boundaries**

Visible copy must state:

- PaddleOCR is the deployed runtime OCR.
- PaddleOCR exact match is about 81% on the frozen 16-image set, with CER 0.031.
- Brand classification is diagnostic only at about 35%.
- Color accuracy 86.3% is measured on VCoR and is not a parking-CCTV claim.
- Security-gate performance is 69% detection at 2.5% false alarms.
- CTC/ONNX records 0/64 exact match, CER about 0.66, and
  `deployment_ready: false`.
- CTC/ONNX is not deployed and does not replace PaddleOCR.

- [ ] **Step 4: Apply the R3 design without R4 blue remnants**

Run:

```bash
rg -n '#002FA7|0,47,167|linear-gradient|box-shadow|border-radius:[^0]' reports/presentations/swiss_r4/index.html
```

Expected: no R4 IKB theme tokens, gradients, shadows, or non-zero rounded
corners in the final deck.

- [ ] **Step 5: Run the Swiss static validator**

Run:

```bash
node /Users/konalyn/.agents/skills/guizang-ppt-skill/scripts/validate-swiss-deck.mjs reports/presentations/swiss_r4/index.html
```

Expected: PASS with all substantive layouts registered and no prohibited
structure warnings.

- [ ] **Step 6: Commit the consolidated HTML**

```bash
git add -- reports/presentations/swiss_r4/index.html
git commit -m "feat: consolidate R3 evidence into Swiss R4 deck"
```

### Task 3: Write the synchronized outline and speaker script

**Files:**
- Replace: `reports/presentations/swiss_r4/OUTLINE.md`
- Create: `reports/presentations/swiss_r4/SCRIPT.md`

- [ ] **Step 1: Write the exact 35-row outline**

The outline must identify slide number, `data-layout` or cover/closing role,
theme, narrative job, evidence asset, and the one claim that the slide advances.
It must explicitly label slides 02–34 as the 33 substantive slides.

- [ ] **Step 2: Write one script section per slide**

Use headings in the exact form:

```markdown
## Slide 01: Report 4 Final Defense
```

Each section must provide a concise Vietnamese talk track and end with a natural
transition. Keep the deployed-vs-experimental distinctions explicit. Do not add
duplicate English translations.

- [ ] **Step 3: Run the local validator**

Run:

```bash
node reports/presentations/swiss_r4/validate-deck.mjs
```

Expected: PASS with 35 HTML slides, 35 outline rows, 35 unique script headings,
33 substantive layouts, no missing assets, and no placeholder text.

- [ ] **Step 4: Commit the outline and script**

```bash
git add -- reports/presentations/swiss_r4/OUTLINE.md reports/presentations/swiss_r4/SCRIPT.md
git commit -m "docs: add synchronized Swiss R4 slide script"
```

### Task 4: Remove superseded R3/R4 artifacts

**Files:**
- Delete: `reports/presentations/swiss_r3/`
- Delete: `reports/presentations/Report_3_Script.md`
- Delete: `reports/presentations/Report_4_Script.md`

- [ ] **Step 1: Confirm the final deck and script pass before deletion**

Run:

```bash
node reports/presentations/swiss_r4/validate-deck.mjs
```

Expected: PASS.

- [ ] **Step 2: Remove the explicitly superseded outputs**

Delete `reports/presentations/swiss_r3/`,
`reports/presentations/Report_3_Script.md`, and
`reports/presentations/Report_4_Script.md`. Do not remove source captures,
report documents, PDFs, or the older non-Swiss Report 3/4 HTML files.

- [ ] **Step 3: Verify only one Swiss presentation and script remain**

Run:

```bash
find reports/presentations -maxdepth 2 \( -path '*swiss_r[34]*' -o -name 'Report_[34]_Script.md' \) -print | sort
```

Expected: only the final `reports/presentations/swiss_r4/` directory and its
contents.

- [ ] **Step 4: Commit the removal**

```bash
git add -- reports/presentations/swiss_r3 reports/presentations/Report_3_Script.md reports/presentations/Report_4_Script.md
git commit -m "chore: remove superseded Swiss presentation artifacts"
```

### Task 5: Render and visually verify every slide

**Files:**
- Verify: `reports/presentations/swiss_r4/index.html`
- Verify: `reports/presentations/swiss_r4/SCRIPT.md`

- [ ] **Step 1: Serve the final deck locally**

Run:

```bash
python3 -m http.server 4173 --directory reports/presentations/swiss_r4
```

Expected: the deck is available at `http://127.0.0.1:4173/`.

- [ ] **Step 2: Capture all 35 slides at 1920×1080**

Use Playwright with animations disabled or wait for each slide animation to
settle. Navigate one slide at a time and save screenshots outside the repository
under a temporary QA directory.

- [ ] **Step 3: Inspect every slide at full size**

Check:

- no clipped or wrapped hero title
- no content below the navigation safe area
- no missing image, distorted crop, or tiny screenshot
- no accidental overlap
- Lemon Green is the only accent
- type hierarchy matches R3
- slide order and script order match
- title is slide 01 and "Thank you for listening" is slide 35

Fix every issue found and repeat the affected capture.

- [ ] **Step 4: Run final automated checks**

Run:

```bash
node reports/presentations/swiss_r4/validate-deck.mjs
node /Users/konalyn/.agents/skills/guizang-ppt-skill/scripts/validate-swiss-deck.mjs reports/presentations/swiss_r4/index.html
git diff --check
git status --short
```

Expected: both validators pass; no whitespace errors; only intentional commits
and the user's pre-existing unrelated untracked file remain.

- [ ] **Step 5: Commit visual QA fixes if any**

```bash
git add -- reports/presentations/swiss_r4/index.html reports/presentations/swiss_r4/OUTLINE.md reports/presentations/swiss_r4/SCRIPT.md
git commit -m "fix: polish consolidated Swiss deck after visual QA"
```
