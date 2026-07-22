# Report 4 Final Defense — Swiss R3 Lemon Green

**Output:** `reports/presentations/swiss_r4/index.html`

**Script:** `reports/presentations/swiss_r4/SCRIPT.md`

**Accent:** Lemon Green `#C5E803` with `--accent-on:#0a0a0a`

**Count:** 35 slides total = title + 33 substantive slides + closing

Slides 02–34 are the 33 substantive slides requested for the combined Report 3
and Report 4 defense. Slide 01 and slide 35 are excluded from that count.

| # | Layout | Theme | Narrative job | Evidence | Primary claim |
| 01 | SWISS-COVER-ASCII | Accent | Open the final defense | — | The shipped system is plate-primary and measured honestly |
| 02 | S03 | Split | Define the security problem | `14-img-problem.png` | Plate swapping is more than an OCR error |
| 03 | S22 | Light | State the product principle | `15-img-architecture.png` | The registered plate is the primary identity key |
| 04 | S22 | Light | Show the working product | `01-streamlit-home.png` | The final system runs through a real Streamlit dashboard |
| 05 | S22 | Light | Show the authorized outcome | `06-demo-authorized.png` | A registered plate is allowed automatically |
| 06 | S22 | Light | Bound the deployment scope | `19-dashboard-verified.png` | The product is CPU, one-camera, and offline-first |
| 07 | S22 | Light | Explain the integrated architecture | `30-diagram-architecture.png` | FastAPI orchestrates detection, OCR, diagnostics, and decision |
| 08 | S11 | Light | Walk through inference | Architecture diagram | Detect, read, diagnose, then verify |
| 09 | S04 | Light | Ground model selection | `16-img-tech-stack.png` | Each model was selected for a distinct task and CPU constraint |
| 10 | S21 | Grey | Show detector configuration | YOLO code/config captures | YOLOv8n was fine-tuned at 640 px for a lightweight detector |
| 11 | S06 | Light | Report detector result | Detector KPI | Plate localization reaches 98.96% mAP@0.5 |
| 12 | S22 | Light | Provide detector benchmark evidence | `chart_plate_benchmark.png` | Transfer learning beats scratch at similar CPU cost |
| 13 | S08 | Light | Explain the OCR pivot | OCR benchmark | EasyOCR failed the held-out set; PaddleOCR became runtime |
| 14 | S22 | Light | Show OCR evidence | `chart_ocr_benchmark.png` | PaddleOCR measures about 81% exact and CER 0.031 |
| 15 | S13 | Grey | Explain Vietnamese plate ordering | Decision/pipeline visual | Spatial sorting reconstructs two-line plates correctly |
| 16 | S19 | Dark | Explain the deployed OCR path | Runtime summary | PaddleOCR is the only deployed runtime OCR |
| 17 | S19 | Light | Present the brand classifier | Brand architecture capture | EfficientNet-B0 was tested as a diagnostic branch |
| 18 | S03 | Dark split | State the brand decision | `17-brand-curves.png` | About 35% accuracy is insufficient for gate logic |
| 19 | S19 | Light | Present the color classifier | Color architecture | MobileNetV3-Small balances color accuracy and CPU cost |
| 20 | S22 | Light | Show color training evidence | `18-color-curves.png` | VCoR plus full fine-tuning and TTA produced the best color result |
| 21 | S06 | Grey | Report color KPI with caveat | Color KPI | 86.3% is a VCoR result, not a parking-CCTV claim |
| 22 | S07 | Light | Summarize tuning and ablation | Benchmark bars | Measured changes, not architecture prestige, determined the final setup |
| 23 | S22 | Light | Report integrated runtime | Dashboard evidence | Warm API and approach-lock meet the near-real-time target |
| 24 | S08 | Light | Explain the security threshold change | Gate comparison | Threshold 0.40 trades recall for a usable false-alarm rate |
| 25 | S07 | Dark | State security performance | Gate benchmark | The gate detects 69% of swaps at 2.5% false alarms |
| 26 | S22 | Light | Show the soft-warning outcome | `08-demo-mismatch.png` | Attribute mismatch warns but does not override the plate key |
| 27 | S22 | Light | Show the unregistered outcome | `07-demo-unregistered.png` | A plate missing from the registry goes to manual review |
| 28 | S22 | Light | Show the backend interface | `02-fastapi-docs.png` | `/verify` exposes the measured pipeline through FastAPI |
| 29 | S22 | Light | Introduce the CTC experiment | `09-fastapi-verify.png` | ONNX was investigated only as a latency experiment |
| 30 | S04 | Grey | Explain the CTC training split | Four-stage training path | Training data stays separate from the 64 held-out car plates |
| 31 | S06 | Accent | Report the failed CTC result | CTC metrics | 0/64 exact and CER about 0.66 fail the replacement gate |
| 32 | S08 | Grey | Diagnose the CTC failure | Domain comparison | Moto-to-car domain gap supports an explicit no-deploy decision |
| 33 | S22 | Light | Explain the final design pivot | `32-diagram-decision.png` | Plate-primary with soft diagnostics is safer than hard multi-attribute voting |
| 34 | S04 | Light | Synthesize lessons and roadmap | Six-point synthesis | Measure first, ship only what passes, then close the domain gap |
| 35 | SWISS-CLOSING-ASCII | Split accent | Thank the audience and open Q&A | — | Thank you for listening; the team is ready to defend the measured choices |

## Accuracy guardrails

- PaddleOCR remains the deployed runtime OCR.
- PaddleOCR is reported as about 81% exact match on the frozen 16-image set,
  with CER 0.031.
- Brand classification remains diagnostic only at about 35% accuracy.
- Color accuracy 86.3% is reported on VCoR and is not generalized to CCTV.
- Security performance is 69% detection at 2.5% false alarms.
- CTC/ONNX records 0/64 exact match, CER about 0.66, and
  `deployment_ready: false`; it is not deployed.
