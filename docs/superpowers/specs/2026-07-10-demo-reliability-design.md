# Demo Reliability Design

**Date:** 2026-07-10

## Goal

Make the parking-security demo deterministic on a fresh checkout and reliable
under the Streamlit video's real-time pacing, without expanding scope into the
production API/webcam architecture.

## Scope

Phase 1 contains three linked fixes:

1. Preserve OCR recognition confidence separately from plate-detector
   confidence, and use recognition confidence for plate locking.
2. Ensure the default-video session receives enough evidence to decide while
   playback is behind schedule and source frames are dropped.
3. Make the default demo-video dependency explicit and verifiable when the
   application starts.

Out of scope: remote webcam support, API queueing, process separation, and a
full dependency-lock redesign.

## OCR Confidence Contract

`PaddleOCRReader.read_plate` will return a structured reading containing
`text` and `ocr_conf`. For PaddleOCR 2.x, the confidence is the arithmetic
mean of the recognised line confidences after the same top-to-bottom ordering
used to construct the plate text. For PaddleOCR 3.x, it is the arithmetic mean
of the available recognition scores. No readable text produces an empty string
and confidence `0.0`.

`PlateReader.read` will return `text`, `ocr_conf`, `plate_det_conf`, and
`plate_bbox`. `ParkingSession` passes `ocr_conf` to `DecisionEngine` as
`plate_conf`; detector confidence remains diagnostic data only.

The lock path continues to require repeated, identical readings above
`lock_conf`. A single-image API request remains backward-compatible: it returns
the existing verdict shape but includes both confidences for observability.

## Real-time Video Behaviour

The display loop may drop frames to preserve wall-clock playback. The decision
gate must not depend on an arbitrary count of *displayed* frames, because a
slow machine can otherwise observe too few samples in the readable approach
window.

The session will expose a source-frame-aware sampling path: when the UI skips
source frames, it will advance the session frame counter by the same number.
The default demo configuration will use a sampling interval that still yields
at least five detection samples across the calibrated approach window. The
video integration test will simulate a 30 FPS source and overloaded playback,
then require a deterministic `DECIDED` verdict for the bundled sample.

The displayed FPS remains measured wall-clock FPS; no synthetic performance
number is introduced.

## Demo Video Distribution

The app will treat `main/data/test/sample_parking.mp4` as a required demo
artifact. A setup helper will download it only from the documented source,
validate a SHA-256 checksum, and fail with a copy-paste command if unavailable.
The UI's default-video option will check the artifact before starting rather
than silently completing with no video.

The project documentation will explain the setup command and identify the
source and checksum. The video itself remains excluded from ordinary Git blobs;
release packaging or the setup helper supplies it.

## Error Handling and Observability

- OCR engine errors keep their existing non-crashing video behaviour, but logs
  identify whether detector localisation or OCR recognition failed.
- A missing or checksum-invalid video shows an actionable Streamlit error.
- Result payloads record `ocr_conf` and `plate_det_conf` so debugging a lock is
  possible without guessing which model supplied the confidence.

## Verification

Tests will cover:

- PaddleOCR v2 and v3 result mapping with recognition confidence.
- A high detector confidence with low OCR confidence that must not lock.
- Repeated high OCR confidence reads that must lock.
- A paced/drop-frame default-video run that reaches `DECIDED`.
- Missing and checksum-invalid demo-video setup paths.

Full native tests, the Docker test command from the documented working
directory, and a real default-video smoke run must pass before Phase 1 is
considered complete.
