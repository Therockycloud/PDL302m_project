# Swiss R3 + R4 Consolidation Design

## Goal

Create one final Report 4 defense presentation that incorporates the necessary
Report 3 model-and-results material. The new deck must use the existing Swiss R3
visual system, contain exactly 33 substantive slides plus a title slide and a
"Thank you for listening" slide, and replace the old Swiss R3 and R4 outputs.

## Deliverables

- `reports/presentations/swiss_r4/index.html`
- `reports/presentations/swiss_r4/OUTLINE.md`
- `reports/presentations/swiss_r4/SCRIPT.md`

The old `reports/presentations/swiss_r3/` directory will be removed. The existing
`reports/presentations/swiss_r4/` files will be replaced by the consolidated
deck. The obsolete top-level `Report_3_Script.md` and `Report_4_Script.md` files
will be removed so the new `SCRIPT.md` is the only authoritative speaker script.

## Visual Direction

The consolidated deck inherits the current Swiss R3 design:

- Lemon Green accent `#C5E803`
- warm white, near-black, and neutral grey palette
- Inter/Helvetica-style sans-serif typography
- large, lightweight headlines and strong grid alignment
- square corners, no gradients, no shadows, and no mixed accent colors
- registered Swiss layouts only, with `data-layout` on substantive slides
- existing R3 navigation, keyboard controls, motion recipes, and low-power mode

The R4 IKB-blue styling will not carry into the final deck. Existing R3 and R4
screenshots, diagrams, charts, and code captures will be reused where they
provide evidence.

## Narrative

By the end, the defense panel should understand how the team moved from measured
model results in Report 3 to a defensible, CPU-first parking verification product
in Report 4, including what was deployed, what failed, and why the final design
is plate-primary.

The story follows:

1. Security problem and product promise
2. Product scope, verdicts, architecture, and pipeline
3. Report 3 model selection, training, and measured results
4. Integrated runtime and security-gate evidence
5. Report 4 CTC/ONNX experiment and explicit non-deployment decision
6. Design pivot, lessons, roadmap, and final synthesis

Repeated R3/R4 introductions, duplicate dashboards, duplicate verdict evidence,
and repeated KPI summaries will be removed.

## Slide Architecture

The final deck contains 35 slides:

| # | Role | Planned content |
|---|---|---|
| 01 | Title | Report 4 Final Defense with Report 3 model evidence |
| 02 | Content | Security problem: plate swapping is more than OCR |
| 03 | Content | Product promise and plate-primary principle |
| 04 | Content | Working dashboard overview |
| 05 | Content | Four UI verdicts |
| 06 | Content | CPU, one-camera, offline-first scope |
| 07 | Content | FastAPI and Streamlit architecture |
| 08 | Content | End-to-end detect-to-decision pipeline |
| 09 | Content | Model selection and reference architectures |
| 10 | Content | YOLOv8n training configuration |
| 11 | Content | Plate detector KPI |
| 12 | Content | Plate detector benchmark evidence |
| 13 | Content | EasyOCR-to-PaddleOCR pivot |
| 14 | Content | PaddleOCR accuracy and CER evidence |
| 15 | Content | Vietnamese two-line spatial sorting |
| 16 | Content | Paddle runtime and configuration proof |
| 17 | Content | EfficientNet-B0 brand-classifier design |
| 18 | Content | Brand result and diagnostic-only decision |
| 19 | Content | MobileNetV3-Small color-classifier design |
| 20 | Content | Color training-curve evidence |
| 21 | Content | Color KPI and VCoR caveat |
| 22 | Content | Hyperparameter and ablation path |
| 23 | Content | Integrated latency and runtime KPI |
| 24 | Content | Security gate before and after threshold 0.40 |
| 25 | Content | Detection rate and false-alarm result |
| 26 | Content | Verdict logic in the product |
| 27 | Content | AUTHORIZED, UNREGISTERED, and mismatch evidence |
| 28 | Content | `/verify` API evidence |
| 29 | Content | Motivation for lightweight CTC OCR |
| 30 | Content | Synthetic and pseudo-label training path |
| 31 | Content | CTC measured result and deployment gate |
| 32 | Content | Domain gap, anti-leakage, and no-deploy decision |
| 33 | Content | Initial design versus delivered design |
| 34 | Content | Lessons, next steps, and closing synthesis |
| 35 | Closing | Thank you for listening and Q&A |

Slides 02–34 are the 33 substantive slides requested by the user. Slides 01 and
35 are excluded from that count.

## Speaker Script

`SCRIPT.md` will contain one numbered section per slide, matching all 35 slides
exactly. Each section will include:

- a concise Vietnamese talk track written for oral delivery
- the key evidence or caveat that must be stated accurately
- a short transition into the next slide

The script will not repeat English translations unless visible deck content
requires them. It will preserve the important caveats: PaddleOCR remains the
runtime OCR, brand recognition is diagnostic only, the 86.3% color result is on
VCoR rather than parking CCTV, and CTC/ONNX remains `deployment_ready: false`.

## Replacement and Verification

Implementation will stage the new deck in a temporary directory, validate it,
and only then replace the old outputs. Verification includes:

- exactly 35 slides total
- exactly 33 substantive slides between title and closing
- all slide numbers and script headings aligned
- registered Swiss layouts and a varied layout rhythm
- no missing local images
- no placeholder text
- no multiple accent colors, gradients, shadows, or rounded corners
- Swiss deck validator passes
- browser review of every slide at presentation size
- navigation, index, keyboard controls, and low-power mode work
- obsolete Swiss R3 and old script files are absent
