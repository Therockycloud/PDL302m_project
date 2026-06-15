# Design Spec — UI Redesign + Content Consistency + Code Fixes

**Date:** 2026-06-15
**Project:** PDL302m — Smart Parking Security (Plate-Primary Verification)
**Scope:** 4 workstreams — unified UI redesign (slides + Streamlit), supporting imagery, content consistency, code fixes.

---

## 1. Goal & Context

The system that was *built* (plate-primary, PaddleOCR, EfficientNet-B0 / MobileNetV3, colour = soft warning, brand dropped) diverged from the system the *docs & slides describe* (3-factor hard cross-verification, EasyOCR, ResNet50/MobileNetV2, ≥95% targets). Three prior reviews (academic, code, UI) confirmed: strong engineering substrate, but documentation/slides lie about it, the slide set has two clashing design systems, and the Streamlit theme carries leftover "slop" (violet gradient title, stale identifiers, dead hover).

This spec unifies the visual language, fills empty slide space with real content, re-aligns all narrative to the **honest pivot story**, and cleans the code.

### Content truth decision (locked: **Option C — Hybrid pivot narrative**)
Docs & slides tell the *journey*: proposed 3-factor cross-verification → experiments showed brand (~29%) and colour (~60% / 14% depending on set) too weak for hard denial → **settled on plate-primary, colour as soft warning, brand dropped**, OCR chosen by Benchmark C (PaddleOCR 81% vs EasyOCR 0%). This keeps the literature/proposal value while being truthful and matching the live demo.

---

## 2. Workstreams & Sequencing

```
WS3 Content truth  ─►  WS1 UI redesign  +  WS2 Imagery   ─►  (WS4 Code fixes, independent)
 (foundation)          (slides + Streamlit)  (diagrams/charts)
```

**Rationale:** diagrams and slide content must reflect the *true* architecture, so content truth (WS3 decisions) is settled first; WS1+WS2 execute together; WS4 code fixes run independently any time.

---

## 3. Locked Design System — "Clean Light Systems"

Applies to **both** slides and Streamlit (one language).

### Typography (self-hosted `.woff2`, no Google `<link>` — matches the project's offline-first thesis)
- **Display / body:** Plus Jakarta Sans
- **Numbers / metrics / code:** JetBrains Mono (all stats use mono)
- **Watermark numerals only:** a squared geometric face (Chakra Petch or Rajdhani) — squarer than mono per user note
- **Banned:** Inter (per high-end skill), serif as default

### Color tokens
| Token | Value | Use |
|---|---|---|
| `--bg` | `#fafaf9` | page background |
| `--surface` | `#ffffff` | cards |
| `--ink` | `#18181b` | primary text |
| `--muted` | `#71717a` | secondary text |
| `--hairline` | `#e4e4e7` | dividers (replaces thick borders) |
| `--accent` | `#15803d` | single accent (forest green), all states/CTAs |
| `--accent-dim` | `#f0fdf4` | accent tint |
| `--alert` | `#b91c1c` | UNREGISTERED / MISMATCH |
| `--warn-fg` / `--warn-bg` | `#b45309` / `#fffbeb` | colour soft-warning |
| `--feed-dark` | `#0b0f14` | video surface |

Single accent locked across the whole page (Color Consistency). **Unify the existing dashboard `#00875a` → `#15803d`.**

### Shape (Shape Consistency Lock)
- Cards `10–12px`, metric tiles `8px`, pills/chips `6–8px`
- **Video feed = sharp (radius 0)** — deliberate exception, framed by detection brackets
- Soft, tinted, diffused shadows only (e.g. `0 18px 40px -22px rgba(21,128,61,0.3)`). No harsh black `shadow-md`. No thick gray 1px borders — use `--hairline` / soft rings.

### Background texture — "Forest Aurora" (locked)
Fixed, `pointer-events:none`, low opacity:
- 3 radial blooms: green `rgba(21,128,61,.13)` @ top-right, teal `rgba(13,148,136,.11)` @ bottom-left, lime `rgba(132,204,22,.06)` @ center
- + SVG fractal-noise grain ~3%

### Motifs & supporting components (fill empty space with real content)
- **Detection corner-brackets** (green L-shapes) framing feed/visuals — on-theme (object detection), recurring brand device
- **Real visual per content slide** (diagram / chart / screenshot) in the empty zone, asymmetric split
- **Callout chips** (small pills: status + mono stat)
- **Squared watermark numeral** (faint, max 1/slide, title/section openers only)
- **No emoji / no icons** (locked) — verdicts and labels are text only
- Eyebrow labels: **max 1 per 3 sections**; **drop all `01 /` section-numbering**

### High-end material qualities (from high-end-visual-design, the parts that transfer)
- **Double-bezel** cards (outer shell `rgba(0,0,0,.04)` + inner core with inset highlight) for Streamlit tiles
- Squircle/refined radii within the scale above, macro-whitespace, refined `cubic-bezier(0.32,0.72,0,1)` hover transitions
- **Not used** (don't fit a real-time monitor / Streamlit): scroll-reveal choreography, hamburger morph, Fluid Island nav, Framer Motion. Honesty: Streamlit styling is CSS-injection only.

---

## 4. WS1 — UI Redesign

### 4a. Slides (`presentations/Report_1..4_Presentation.html`)
- Rebuild all 4 decks on the **single** locked system (fixes Report 1 cool/Inter vs 2–4 warm/Lora split).
- Self-host fonts into `presentations/fonts/`; remove Google Fonts `<link>`.
- Apply Forest Aurora bg + grain; detection-bracket motif; squared watermark numerals on openers.
- Title-slide metric strip (mono); content slides use asymmetric split with a real visual.
- Remove section-number eyebrows; enforce eyebrow ≤ 1/3 sections.
- Replace any em-dash `—` with hyphen/restructure (per design skill).
- Fix cross-deck **instructor/group name** inconsistency → one correct value on all title slides (confirm correct name with user during impl).
- Zero broken images (already true; keep).

### 4b. Streamlit (`main/src/utils/visual.py`, `main/src/ui/dashboard.py`)
- Rewrite the CSS factory: **rename** `create_glassmorphic_css()` → `build_theme_css()`, `.glass-card` → `.card`, `--neon-green` → `--accent`.
- **Remove** the 3-stop violet gradient `.neon-title` → solid `--ink` (or single `--accent`).
- **Remove** dead hover rules (`:hover { border-color: transparent; box-shadow: none }`).
- Apply: Forest Aurora app bg + grain, double-bezel bento layout, **sharp video feed** with detection brackets + bbox label, mono metrics (FPS/latency), compact squared metric tiles.
- Result card: large "KẾT QUẢ" title, enlarged mono plate, verdict as **bold colored text, no box, no icon** (`AUTHORIZED` green / `UNREGISTERED`·`MISMATCH` red), colour soft-warning as amber text block.
- Keep all existing functional behavior (input modes, ParkingSession, matcher) unchanged — visual layer only.

---

## 5. WS2 — Imagery (all vector + real-data charts + real screenshots, offline)

Generated to match the locked design system; **diagrams reflect the true plate-primary architecture**.

**SVG diagrams (hand-authored):**
1. 2-stage pipeline: YOLOv8 vehicle→plate → PaddleOCR → decision
2. System architecture: camera → app → models → CSV → dashboard
3. Decision-flow: plate match? → AUTHORIZED / colour mismatch → soft warning / not in DB → UNREGISTERED
4. **Pivot-story diagram**: 3-factor proposed → experiments weak → plate-primary

**Charts rendered from real data (matplotlib, styled to tokens):**
5. OCR benchmark: PaddleOCR 81% vs EasyOCR 0% (`docs/benchmarks/ocr_benchmark.csv`)
6. Colour CNN benchmark: MobileNetV3 / EfficientNet / ResNet50 (`color_benchmark.csv`)
7. Plate detector mAP (`plate_benchmark.csv`)
8. Dataset distribution: 8 brands (1,209) · 8 colours (1,130)

**Real screenshots:** redesigned Streamlit (AUTHORIZED + ALERT states), demo frame with bbox.
**Restyle/keep:** training curves (re-theme background).

Output to `presentations/` (and `presentations/evidence/`), replacing/augmenting existing `img_*.png`.

---

## 6. WS3 — Content Consistency (align to code truth, narrate the pivot)

Per-file edits (truthful, pivot framing where it helps the academic story):
- **OCR:** EasyOCR → **PaddleOCR** as delivered engine in `Report_3` §3.2, `Report_4` diagram, slide R3; cite Benchmark C; EasyOCR = fallback.
- **Models:** unify to **EfficientNet-B0 / MobileNetV3-Small** in `Report_1` (§1.4, §3.2, §4.2) + slide R1; one line noting the change from the proposed ResNet50/MobileNetV2.
- **Decision logic:** state **plate-primary, colour soft-warning, brand dropped** consistently in all 4 reports + slides + `PROJECT.md`; rewrite `Report_4` §6 conclusion to match real metrics (no "≥95% achieved").
- **Latency:** one consistent table (state condition, e.g. "~1.6s from 2nd vehicle; cold-start 4.5s"); reconcile the 1.6s/2.19s/<1.0s discrepancy.
- **Colour accuracy:** reconcile 14.16% (Report 3) vs 59.73% (Benchmark A) — state which set each measures; flag the preprocessing/labeling discrepancy.
- **Citations:** verify or replace placeholder-looking refs (Smith&Patel, Wang&Choi, Jang&Lim, Chen, Lin, Lima) with real sources (reuse `related_work.md`).
- **Dataset scope:** `Report_1` §6 mark "planned vs delivered" (1,209 / 1,130 / 5).
- **`PROJECT.md`:** update to runtime reality (PaddleOCR, torch colour, light theme, plate-primary).
- **README:** fix "15 unit tests" → actual (28 passed / 5 skipped); align any stale TF-runtime claims.

## 7. WS4 — Code Fixes

- **`requirements.txt`:** remove `tensorflow` from runtime; add `requirements-train.txt` (TF/Keras for `train.py` + `run_evaluation.py` only). Keeps runtime free of the Paddle/TF conflict.
- **Uncommitted `classifiers.py` diff (TF `__call__` change):** revert — orphaned (dashboard uses `torch_color`, not these), untested at runtime. Confirm with user.
- **`detector.py` uncommitted diff (prefer `.onnx`):** keep (good).
- **`visual.py` uncommitted diff:** superseded by the WS1 rewrite.
- Identifier renames per §4b; remove dead hover; remove violet gradient.
- Run `pytest` after to confirm still 28 passed / 5 skipped.

---

## 8. Non-Goals
- No retraining / new experiments (brand/colour accuracy stays as measured).
- No change to pipeline logic, ParkingSession, matcher, or DB schema.
- No new product features; visual + content + hygiene only.

## 9. Verification
- Slides: open all 4 in browser, check unified look, no broken images, no Google-Fonts dependency offline, no em-dash, eyebrow count ≤ ceil(sections/3).
- Streamlit: launch via `run_ui.sh`, verify both AUTHORIZED and ALERT render, no violet gradient, metrics mono, feed sharp.
- Code: `KMP_DUPLICATE_LIB_OK=TRUE pytest -q` green; fresh `pip install -r requirements.txt` imports without TF.
- Content: cross-read 4 reports + slides + README for one consistent system description.

## 10. Risks
- Self-hosting fonts may shift slide layout slightly → re-check each deck.
- Renaming CSS identifiers could break selectors referenced in `dashboard.py` → grep all usages before rename.
- Citation replacement requires verifying real sources (avoid swapping one fabricated ref for another).
