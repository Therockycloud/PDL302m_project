# Vietnamese Plate Recognizer and Single Product Camera Design

**Date:** 2026-07-13

**Status:** Approved for planning

**Primary goal:** Make license-plate recognition faster and more accurate on low-end Docker CPU deployments, while keeping the demo synchronized and presenting only the Product camera.

## 1. Scope

This change replaces the generic PaddleOCR runtime path with a lightweight recognizer trained for Vietnamese license plates. YOLOv8n remains responsible for locating the vehicle and plate. The runtime must require four matching, high-confidence OCR results within an eight-frame evidence window before authorizing or rejecting a plate.

The existing backend approach-lock remains the activation mechanism for this phase. A new browser-side reverse-motion detector is explicitly out of scope so model quality and inference latency remain the focus.

The demo will remove the visible Source video pane. Its video element remains hidden as the only media clock and frame decoder. The Product canvas is the only camera view and receives separate play, pause, and seek controls below the image.

## 2. Success Criteria

The candidate recognizer may replace PaddleOCR only when all of these conditions hold:

- Whole-plate exact-match accuracy is at least 90% on real, manually verified,
  held-out data. On the frozen 16-crop CCTV set this requires at least 15/16
  exact matches; the candidate must also reach at least 90% on the expanded
  real test set described below.
- The current PaddleOCR result of 81.2% exact match remains the baseline that
  the new model must materially outperform, not the deployment threshold.
- Mean character error rate is no worse than the current baseline of 0.031.
- OCR p95 latency on the constrained Docker CPU is materially lower than PaddleOCR.
- The end-to-end parking path locks only after four matching high-confidence reads from no more than eight successfully processed evidence frames.
- Exhausting the evidence window without a lock returns `UNCERTAIN`; it never opens the barrier.
- Product playback stays synchronized with the hidden media clock and does not accumulate a frame queue.
- Cold-start, warm p50, warm p95, per-stage, and gate-to-verdict latency are measured from actual runs. The desired gate-to-verdict latency is 500–1000 ms, but accuracy and the 4/8 safety rule take priority on hardware that cannot meet it.

No report or slide may present an estimated metric as a measured result.

## 3. Model Architecture

The primary candidate is a MobileNetV3-Small image encoder followed by a sequence projection and CTC decoder. A PP-LCNet or SVTR-Tiny recognizer may be retained as a benchmark candidate, but only one measured winner is deployed.

- Input: RGB plate crop normalized to `192 x 64` while preserving aspect ratio with padding.
- Vocabulary: digits `0-9`, uppercase letters `A-Z`, the CTC blank token, and no punctuation.
- Output: normalized alphanumeric plate text, sequence confidence, and optional per-character confidence for diagnostics.
- Export: ONNX with a fixed input shape and ONNX Runtime CPU inference.
- Decoding: greedy CTC is the latency baseline. Vietnamese plate-format constraints may reject an impossible sequence but must not silently rewrite ambiguous characters into a registered plate.
- Two-line plates: detect a compact aspect ratio, split rows, order top-to-bottom, and concatenate into a single recognition strip before inference.

The recognizer consumes an already-localized plate crop. It must not run a second text detector.

## 4. Dataset Design

The current 16 hand-labelled CCTV plate crops remain a frozen regression set.
They are too small to train a recognizer and must never be used for training
or model selection. A second held-out test set must contain at least 100
manually verified real crops, grouped by source and vehicle so repeated frames
cannot inflate the 90% exact-match result.

Training data will combine:

1. Synthetic Vietnamese plates generated from valid layouts and realistic typography.
2. Domain augmentations covering downscaling, motion blur, defocus, low light, glare, perspective, partial occlusion, sensor noise, and JPEG compression.
3. Real crops from the existing plate-detection dataset. PaddleOCR may create candidate pseudo-labels, but only high-confidence, format-valid samples enter the candidate pool, and pseudo-labelled samples remain identifiable.
4. Manually verified real crops and labelled frames extracted from parking videos.

Train, validation, and test splits must be grouped by source image, video, and vehicle identity. Neighboring frames of the same vehicle must not cross split boundaries. Synthetic-only accuracy and pseudo-label agreement are diagnostics, not claims of real-world accuracy.

The dataset build must be reproducible from scripts and a manifest containing image path, normalized label, source type, group identifier, split, and verification status.

## 5. Training and Export

Training uses CTC loss, deterministic seeds, early stopping on real validation exact-match, and checkpoint selection by exact-match followed by CER. The training output includes:

- the best native checkpoint;
- the exported ONNX model;
- vocabulary and preprocessing metadata;
- training history and curves;
- a machine-readable evaluation report.

A smoke configuration must train quickly on a tiny subset for CI. Full training is an explicit script and is not run inside normal unit tests.

ONNX parity is required: decoded text and confidence from the exported model must match the native checkpoint within documented numerical tolerance on a fixed sample set.

## 6. Runtime Data Flow

1. The existing vehicle detector and approach-lock observe sampled frames and open the evidence window while the vehicle reverses toward the configured parking ROI.
2. The dedicated plate YOLO localizes the plate within the target vehicle crop.
3. The new ONNX recognizer reads the localized crop.
4. The session records the normalized text and OCR confidence for that frame.
5. A plate locks only when the same high-confidence text appears four times within eight evidence frames.
6. Once the plate locks, the colour classifier runs once on the best target-vehicle crop. Colour remains a soft warning and is not part of the plate lock.
7. The matcher produces `AUTHORIZED` or `UNREGISTERED`. If the eight-frame window expires without a lock, the result is `UNCERTAIN` and the barrier stays closed.

PaddleOCR is not allowed to contribute an authorization vote. It may run once on the best crop after an uncertain window for explicitly labelled diagnostic output. A missing or incompatible primary ONNX model must fail readiness visibly instead of silently changing the deployed engine.

## 7. Browser Demo

The hidden `<video>` remains the single decoder and clock. `requestVideoFrameCallback` draws every decoded frame onto the visible Product canvas. The browser attempts to sample at 100 ms intervals, but permits only one request in flight and drops obsolete frames rather than queuing them.

The visible component contains:

- one Product camera canvas;
- state, media time, measured display FPS, and processing diagnostics;
- play/pause, seek, current time, and duration controls below the canvas.

Seeking resets backend trajectory and evidence state. A final Streamlit rerun resumes from the same media time without restarting playback. The Source video caption and visible source frame are removed.

## 8. Error and Safety Behaviour

- Invalid or unreadable plate crops add no high-confidence vote.
- Inference exceptions are logged and cannot authorize a vehicle.
- Invalid Vietnamese plate structure may remain diagnostic evidence but cannot lock.
- Missing ONNX assets or metadata fail the model health/readiness check.
- An expired 8-frame window returns `UNCERTAIN`.
- Browser request failure clears the in-flight flag so a newer frame can retry; it does not enqueue the failed frame.
- A seek, new video activation, or stopped session aborts the active request and clears all evidence.

## 9. Verification

Verification includes:

- unit tests for preprocessing, two-line conversion, CTC decoding, format validation, confidence calculation, ONNX reader behaviour, and 4/8 aggregation;
- native-versus-ONNX parity tests;
- training smoke tests and manifest split-leakage tests;
- evaluation against the frozen 16-crop CCTV set and a newly labelled, vehicle-grouped real-video set;
- side-by-side PaddleOCR and candidate OCR benchmarks for exact match, CER, model size, cold latency, warm p50, and warm p95;
- end-to-end Docker runs on `parking_case_real.mp4` and `sample_parking.mp4` under constrained CPU and memory;
- real browser tests for a single visible Product camera, controls, seek reset, synchronized playback, no backlog, and correctly timed final verdicts.

If the candidate scores below 90% whole-plate exact match on either required
real test set, PaddleOCR remains the deployed runtime and the failure is
documented. Character accuracy, synthetic accuracy, or pseudo-label agreement
cannot substitute for this gate. The project must not trade away measured
accuracy merely to report a lower latency.

## 10. Documentation and Presentation Deliverables

Measured results and the final deployed architecture must be synchronized across:

- root `README.md` and `main/README.md`;
- `docs/model_specifications.md`;
- a new OCR benchmark CSV, Markdown report, JSON evidence, and chart;
- `reports/documents/Report_2_Data_Tasks.md`;
- `reports/documents/Report_3_Model_Results.md`;
- `reports/documents/Report_4_Final_Report.md`;
- Report 3 and Report 4 presentation HTML, speaker scripts, benchmark assets, and release exports;
- the professional Report 4 deck if it continues to be distributed.

The documents must describe the real dataset split, architecture, training configuration, baseline comparison, measured Docker performance, limitations, and whether the candidate actually replaced PaddleOCR.

## 11. Delivery Boundaries

- Work is committed directly to `main` as requested.
- Existing user-owned changes, especially `CLAUDE.md`, are not staged or modified.
- Model weights are committed only if repository size and licensing permit it; otherwise the reproducible artifact-generation command and checksum are documented.
- Model development may be delegated to an implementation agent after this specification is reviewed. The primary agent owns integration, verification, Docker benchmarking, browser testing, documentation consistency, and final commits.
