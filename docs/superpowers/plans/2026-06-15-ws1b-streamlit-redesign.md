# WS1b — Streamlit Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-skin the Streamlit dashboard to the locked "Clean Light Systems" design (design A): off-white + ink + single forest-green accent, JetBrains-mono metrics, Forest-Aurora background, double-bezel bento, sharp-cornered video feed with detection-bracket frame, verdicts as bold colour text (no icons/boxes). Remove all leftover "slop" (violet gradient title, dead hover, stale `glassmorphic`/`neon` identifiers).

**Architecture:** All visual logic lives in `main/src/utils/visual.py` (a pure CSS/overlay factory) consumed by `main/src/ui/dashboard.py`. We keep functional behaviour (input modes, ParkingSession, matcher) untouched and only change the presentation layer. Pipeline/model code under `main/src/models|engine` is NOT touched.

**Tech Stack:** Streamlit (CSS injected via `st.markdown(unsafe_allow_html=True)`), OpenCV overlay (BGR), pytest for the pure-function units, Claude preview tools to launch + screenshot for visual verification.

**Design tokens (locked):** bg `#fafaf9`, surface `#ffffff`, ink `#18181b`, muted `#71717a`, hairline `#e4e4e7`, accent `#15803d`, accent-dim `#f0fdf4`, alert `#b91c1c`, warn `#b45309`/`#fffbeb`, feed-dark `#0b0f14`. Fonts: Plus Jakarta Sans (display/body) + JetBrains Mono (numbers). Radii: card 10–12px, metric 8px, video 0 (sharp). No emoji/icons in verdict text.

**Font strategy:** load Plus Jakarta Sans + JetBrains Mono via `@import url('https://fonts.googleapis.com/...')` at the top of the injected CSS, with a system fallback stack (`-apple-system, Segoe UI, Roboto, sans-serif`). (Self-hosting `.woff2` is reserved for the slides in WS1a, which are presented offline; the dashboard runs interactively where the fallback stack is acceptable.)

**Run convention:** pytest via `cd main && KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/Caskroom/miniforge/base/bin/python -m pytest`. Launch via `bash main/run_ui.sh` (or preview tools).

---

### Task 1: Recolour the detection overlay (design A)

**Files:**
- Modify: `main/src/utils/visual.py` (the `_AUTHORIZED_BGR`/`_ALERT_BGR`/`_UNKNOWN_BGR` constants, lines ~20-22)
- Test: `main/tests/test_visual_theme.py`

- [ ] **Step 1: Write the failing test**

```python
# main/tests/test_visual_theme.py
"""Guards the WS1b 'Clean Light Systems' theme tokens in visual.py."""
import numpy as np
from src.utils import visual


def test_overlay_uses_forest_green_for_authorized():
    # #15803d -> BGR (61, 128, 21)
    assert visual._AUTHORIZED_BGR == (61, 128, 21)


def test_overlay_uses_alert_red():
    # #b91c1c -> BGR (28, 28, 185)
    assert visual._ALERT_BGR == (28, 28, 185)


def test_draw_overlay_returns_same_shape_without_mutating():
    img = np.zeros((120, 200, 3), dtype=np.uint8)
    dets = [{"bbox": [10, 10, 80, 50], "plate_text": "30F-12345"}]
    out = visual.draw_detection_overlay(img, dets, {"status": "AUTHORIZED"})
    assert out.shape == img.shape
    assert out is not img            # must be a copy
    assert int(img.sum()) == 0       # source not mutated
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd main && KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/Caskroom/miniforge/base/bin/python -m pytest tests/test_visual_theme.py -v`
Expected: the two colour tests FAIL (current values are `(136, 255, 0)` and `(102, 51, 255)`).

- [ ] **Step 3: Update the BGR constants**

In `main/src/utils/visual.py` replace:

```python
_AUTHORIZED_BGR = (136, 255, 0)   # #00ff88 in BGR
_ALERT_BGR = (102, 51, 255)       # #ff3366 in BGR
_UNKNOWN_BGR = (0, 200, 255)      # amber-ish
```

with:

```python
_AUTHORIZED_BGR = (61, 128, 21)   # #15803d (forest green) in BGR
_ALERT_BGR = (28, 28, 185)        # #b91c1c (alert red) in BGR
_UNKNOWN_BGR = (9, 89, 180)       # #b45309 (amber warn) in BGR
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd main && KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/Caskroom/miniforge/base/bin/python -m pytest tests/test_visual_theme.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add main/src/utils/visual.py main/tests/test_visual_theme.py
git commit -m "feat(ui): recolour detection overlay to design-A tokens (WS1b)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Rewrite the CSS factory → `build_theme_css()` (design A)

**Files:**
- Modify: `main/src/utils/visual.py` (rename + rewrite `create_glassmorphic_css`)
- Test: extend `main/tests/test_visual_theme.py`

- [ ] **Step 1: Add failing tests for the new CSS factory**

Append to `main/tests/test_visual_theme.py`:

```python
def test_build_theme_css_exists_and_is_string():
    css = visual.build_theme_css()
    assert isinstance(css, str) and len(css) > 500


def test_theme_uses_accent_token_not_neon():
    css = visual.build_theme_css()
    assert "--accent: #15803d" in css
    assert "--neon-green" not in css            # stale token removed


def test_theme_has_no_violet_gradient_or_glass_blur():
    css = visual.build_theme_css()
    assert "#5b31df" not in css                 # violet LILA gradient gone
    assert "backdrop-filter" not in css         # glass blur gone
    assert "glass-card" not in css              # class renamed to .card
```

- [ ] **Step 2: Run to verify failure**

Run: `cd main && KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/Caskroom/miniforge/base/bin/python -m pytest tests/test_visual_theme.py -k theme -v`
Expected: FAIL (`build_theme_css` undefined; `create_glassmorphic_css` still has `--neon-green`, `backdrop-filter`, `glass-card`, `#5b31df`).

- [ ] **Step 3: Rename and rewrite the function**

In `main/src/utils/visual.py`, rename `def create_glassmorphic_css() -> str:` to `def build_theme_css() -> str:` and replace its entire body so the returned CSS implements design A. The CSS must include (at minimum):

```css
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;700&display=swap');

:root {
    --bg: #fafaf9;
    --surface: #ffffff;
    --ink: #18181b;
    --muted: #71717a;
    --hairline: #e4e4e7;
    --accent: #15803d;
    --accent-dim: #f0fdf4;
    --alert: #b91c1c;
    --warn-fg: #b45309;
    --warn-bg: #fffbeb;
    --feed-dark: #0b0f14;
    --radius: 12px;
}

/* Forest-Aurora app background (fixed, low-opacity, pointer-events none via ::before) */
.stApp {
    background-color: var(--bg);
    background-image:
        radial-gradient(circle at 90% 6%, rgba(21,128,61,0.10), transparent 40%),
        radial-gradient(circle at 6% 94%, rgba(13,148,136,0.08), transparent 42%),
        radial-gradient(circle at 50% 50%, rgba(132,204,22,0.05), transparent 55%);
    font-family: 'Plus Jakarta Sans', -apple-system, 'Segoe UI', Roboto, sans-serif;
    color: var(--ink);
}

/* Title: solid ink, NO gradient */
.app-title { font-size: 1.6rem; font-weight: 800; color: var(--ink); letter-spacing: -0.3px; }

/* Double-bezel card */
.card { background: rgba(0,0,0,0.04); padding: 6px; border-radius: 14px; }
.card > .card-inner { background: var(--surface); border-radius: 10px; padding: 16px;
    box-shadow: 0 18px 40px -22px rgba(21,128,61,0.3); }

/* Metric tile (compact, squared, mono value) */
.metric-box { background: var(--surface); border-radius: 8px; padding: 10px 12px;
    box-shadow: 0 8px 20px -16px rgba(21,128,61,0.3); }
.metric-value { font-family: 'JetBrains Mono', monospace; font-size: 1.3rem; font-weight: 700; color: var(--ink); }
.metric-label { font-size: 0.5rem; letter-spacing: 1.5px; color: var(--muted); }

/* Sharp video feed + detection corner-brackets */
.feed-wrap { position: relative; background: var(--feed-dark); }
.feed-wrap .bracket { position: absolute; width: 24px; height: 24px; border: 2px solid var(--accent); }

/* Verdict: bold colour text, NO box, NO icon */
.verdict-ok { font-size: 1.2rem; font-weight: 800; color: var(--accent); }
.verdict-bad { font-size: 1.2rem; font-weight: 800; color: var(--alert); }
.soft-warn { color: var(--warn-fg); background: var(--warn-bg); padding: 8px 12px; border-radius: 8px; font-size: 0.8rem; }

/* Buttons: single accent, no violet */
.stButton > button { background: var(--accent) !important; color: #fff !important;
    border: none !important; border-radius: 8px !important; font-weight: 700 !important;
    transition: transform 0.2s cubic-bezier(0.32,0.72,0,1) !important; }
.stButton > button:active { transform: scale(0.98) !important; }

section[data-testid="stSidebar"] { background: #f4f4f5 !important; border-right: none !important; }
```

The function returns this wrapped in `<style>...</style>`. Remove EVERY occurrence of `backdrop-filter`, `--neon-green`, `glass-card`, the 3-stop `linear-gradient(... #5b31df ...)` title, and the dead `:hover { border-color: transparent }` rules from the old body.

- [ ] **Step 4: Run to verify pass**

Run: `cd main && KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/Caskroom/miniforge/base/bin/python -m pytest tests/test_visual_theme.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add main/src/utils/visual.py main/tests/test_visual_theme.py
git commit -m "feat(ui): rewrite CSS factory as build_theme_css design-A (WS1b)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Status badge + alarm helpers → bold text, no icon/box

**Files:**
- Modify: `main/src/utils/visual.py` (`get_status_css`, `get_alarm_html`)
- Test: extend `main/tests/test_visual_theme.py`

- [ ] **Step 1: Add failing tests**

Append to `main/tests/test_visual_theme.py`:

```python
def test_status_css_authorized_is_borderless_green():
    css = visual.get_status_css("AUTHORIZED")
    assert "#15803d" in css
    assert "border: none" in css or "border:none" in css


def test_alarm_html_has_no_emoji_or_border():
    html = visual.get_alarm_html("MISMATCH")
    assert "⚠" not in html and "🚨" not in html      # no emoji/icon
    assert "border: none" in html or "border:none" in html
    assert "#b91c1c" in html or "#de350b" in html      # alert colour
```

- [ ] **Step 2: Run to verify failure**

Run: `cd main && KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/Caskroom/miniforge/base/bin/python -m pytest tests/test_visual_theme.py -k "status or alarm" -v`
Expected: FAIL (current `get_status_css` AUTHORIZED uses `#00875a`; `get_alarm_html` contains the `⚠️` emoji).

- [ ] **Step 3: Update both helpers**

In `get_status_css`, change the AUTHORIZED branch colour from `#00875a` to `#15803d` (and its `rgba(0,135,90,...)` background to `rgba(21,128,61,0.1)`), keep `border: none`. Update the MISMATCH/UNREGISTERED branch to `--alert` `#b91c1c` / `rgba(185,28,28,0.1)`.

In `get_alarm_html`, remove the leading `⚠️ ` from the text, set colour to `#b91c1c`, background `rgba(185,28,28,0.1)`, `border: none`, keep the `pulse-red` animation. The text becomes `f"ALERT — Vehicle status: {status.upper()}"`.

- [ ] **Step 4: Run to verify pass**

Run: `cd main && KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/Caskroom/miniforge/base/bin/python -m pytest tests/test_visual_theme.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add main/src/utils/visual.py main/tests/test_visual_theme.py
git commit -m "feat(ui): borderless green/red status + icon-free alarm (WS1b)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Wire dashboard.py to the new theme + layout

**Files:**
- Modify: `main/src/ui/dashboard.py`

- [ ] **Step 1: Update the import + call site**

In `dashboard.py`, change the import `from src.utils.visual import ( create_glassmorphic_css, ... )` to use `build_theme_css` instead, and update the injection call (the `st.markdown(create_glassmorphic_css(), unsafe_allow_html=True)`) to `st.markdown(build_theme_css(), unsafe_allow_html=True)`.

- [ ] **Step 2: Replace any stale class names used in dashboard markup**

Grep the file: `grep -n "glass-card\|neon-title\|create_glassmorphic_css" main/src/ui/dashboard.py`. For each hit, replace `glass-card` → `card` (and add a `<div class="card-inner">` wrapper where a `.card` is used as a content container), and replace any `neon-title` usage with `app-title`.

- [ ] **Step 3: Apply the bento layout for the live result panel**

Locate the section that renders the live feed + metrics + result (search for `st.columns`, `metric-box`, and the verdict/status rendering). Re-arrange into the design-A bento: a wider feed column (sharp `.feed-wrap` with four `.bracket` corner spans) and a narrower right column holding the compact `.metric-box` row (FPS, latency — mono via `.metric-value`), then a `.card`/`.card-inner` result block whose verdict uses `.verdict-ok` / `.verdict-bad` (bold colour text, NO icon/box) and a `.soft-warn` block for a colour mismatch. Use `st.markdown(..., unsafe_allow_html=True)` for the custom HTML blocks. Keep all existing data plumbing (the values fed in) unchanged — only the wrapping markup/classes change.

- [ ] **Step 4: Verify import + no stale identifiers remain**

Run: `grep -n "create_glassmorphic_css\|glass-card\|neon-title\|#5b31df" main/src/ui/dashboard.py`
Expected: no matches.
Run (import sanity): `cd main && KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/Caskroom/miniforge/base/bin/python -c "import ast; ast.parse(open('src/ui/dashboard.py').read()); print('parse-ok')"`
Expected: `parse-ok`.

- [ ] **Step 5: Commit**

```bash
git add main/src/ui/dashboard.py
git commit -m "feat(ui): wire dashboard to build_theme_css + bento layout (WS1b)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Launch + visual verification (and capture screenshots for WS2)

**Files:** none (verification + screenshot artifacts)

- [ ] **Step 1: Full unit suite still green**

Run: `cd main && KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/Caskroom/miniforge/base/bin/python -m pytest -q`
Expected: previous baseline + new theme tests, 0 failures.

- [ ] **Step 2: Launch the dashboard**

Use the Claude preview tools: `preview_start` with command `bash main/run_ui.sh` (Streamlit serves on 8501). If preview tools are unavailable, run `bash main/run_ui.sh` in the background and open `http://localhost:8501`.

- [ ] **Step 3: Drive the default-video flow and check for errors**

Use `preview_snapshot` / `preview_console_logs` / `preview_logs` to confirm: the app renders the light theme, no Python tracebacks, the "Play Default Parking Video" control works, the feed shows the sharp detection-bracket frame, metrics render in mono, and a verdict renders as bold colour text (no icon/box).

- [ ] **Step 4: Capture screenshots (these feed WS2)**

`preview_screenshot` for: (a) an AUTHORIZED state, (b) an ALERT (MISMATCH/UNREGISTERED) state. Save under `presentations/` (e.g. `img_dashboard_verified.png`, `img_dashboard_alert.png`) so WS1a slides can embed the real redesigned UI. Confirm there is no violet gradient anywhere and the accent is the single forest green.

- [ ] **Step 5: Report** the screenshots + any console/log issues. If issues found, fix in `visual.py`/`dashboard.py` and re-verify from Step 1.

---

## Self-Review

**Spec coverage (spec §4b):**
- Rename `create_glassmorphic_css`→`build_theme_css`, `.glass-card`→`.card`, `--neon-green`→`--accent` → Task 2 Step 3, Task 4 Step 2 ✓
- Remove violet `.neon-title` gradient → Task 2 (test asserts `#5b31df` absent) ✓
- Remove dead hover rules → Task 2 Step 3 ✓
- Forest-Aurora bg + grain, double-bezel bento, sharp feed + detection brackets, mono metrics → Tasks 2 & 4 ✓
- Result card: large title, mono plate, verdict bold colour text no box/icon, amber soft-warning → Task 3 + Task 4 Step 3 ✓
- Keep functional behaviour unchanged → Tasks state "data plumbing unchanged" ✓
- Unify accent `#00875a`→`#15803d` → Tasks 1 & 3 ✓

**Placeholder scan:** Task 4 Step 3 (bento re-layout) is described structurally rather than as a single exact diff because it depends on the current `st.columns` block; it gives explicit class names, structure, and a "keep data plumbing unchanged" constraint plus grep/parse verification. All pure-function edits (Tasks 1–3) have exact before/after.

**Type/name consistency:** `build_theme_css` (not `build_theme()`), classes `.card`/`.card-inner`/`.metric-box`/`.metric-value`/`.feed-wrap`/`.bracket`/`.verdict-ok`/`.verdict-bad`/`.soft-warn` used identically across Tasks 2 and 4. Tokens match the locked palette.

**Note for executor:** `visual.py` currently has uncommitted working-tree changes (the earlier light-theme pass) — that is the expected starting point; build on top of it. Do not touch model/engine/pipeline code. Streamlit reads CSS from `build_theme_css()` only; there is no separate stylesheet file.
