# Synchronized Event-Driven Parking Demo Design

## Goal

Simulate a real-time parking camera visually while keeping the Origin and
Product cam on the same source timestamp. Expensive vehicle, plate, OCR, and
colour inference starts only when the system identifies a vehicle beginning
to reverse into the parking area.

## Success Criteria

- Origin playback remains browser-native and visually smooth.
- Origin and Product cam render from one HTML5 media clock, with at most one
  source-frame difference.
- Streamlit reruns never drive video playback or frame synchronization.
- Motion/trajectory sampling does not block either visual pane.
- The UI enters `REVERSING — VERIFYING` within 100 ms of the trigger result.
- When plate evidence is readable, the target final-verdict latency is
  500–1200 ms from the first confirmed reversing frame.
- If evidence is insufficient by the deadline, return `UNCERTAIN`, keep the
  barrier closed, and continue collecting instead of guessing.
- Every final verdict includes the source timestamp of the evidence frame.

## Architecture

A single Streamlit custom component owns one hidden/shared HTML5 video source.
It renders the source into two visual surfaces:

1. Origin displays the unmodified source.
2. Product cam draws the same video frame into a canvas using
   `requestVideoFrameCallback`, then applies the latest available detection
   and decision overlay.

Because both panes derive from the same decoded video frame and media clock,
model latency cannot shift Product cam behind Origin.

The component sends sampled JPEG frames directly to FastAPI over HTTP. It
allows only one request in flight, so slow inference creates frame dropping
rather than request queues. Streamlit is not involved in the sampling loop.
It remains responsible for page layout and displaying durable summary
metrics/results.

## Backend Session

FastAPI exposes a demo-frame endpoint with a browser-generated session ID,
source timestamp, and JPEG frame. A per-session controller owns the
`ParkingSession` state and processes frames sequentially.

The controller has two stages:

- Observation: lightweight sampled vehicle detection and trajectory state.
- Verification: once the reversing trigger opens, collect useful frames and
  run plate detection, OCR, colour classification, and decision aggregation.

The response contains:

```json
{
  "source_time_s": 11.24,
  "state": "REVERSING_VERIFYING",
  "overlay_results": [],
  "decision": null,
  "latency_ms": 83.1
}
```

A completed response may include a final decision. The component caches the
latest response and draws it over the current Product canvas without pausing
playback.

## Timing Semantics

There are three distinct timestamps:

- `media_time`: current Origin/Product frame time.
- `evidence_time`: source time of the frame used for inference.
- `decision_time`: wall-clock time at which FastAPI returns a verdict.

The UI must not label an old inference result as the current source frame.
It displays the evidence timestamp with the result. Visual synchronization is
measured from Origin versus Product media time, not from the age of an overlay.

## Playback Controls

Play, pause, seek, and restart are controlled by the shared HTML5 video.
Seeking resets the backend demo session because trajectory history from the
old timeline is no longer valid. Product canvas immediately follows the new
media position. Pausing stops frame sampling while preserving the last result.

## Error Handling

- A failed or timed-out frame request does not pause playback; the component
  shows a non-blocking degraded status and retries on a later sample.
- Only one inference request may be active per session.
- Invalid session IDs, timestamps, or images return a 400 response.
- Missing models return 503.
- Backend session state expires after inactivity to prevent memory growth.
- `UNCERTAIN` is a safe result and never opens the barrier.

## Testing

Python unit tests cover session isolation, sequential/single-flight processing,
seek reset, invalid input, response timestamps, and verdict state mapping.
Frontend tests cover the shared-frame renderer, one-request-in-flight gate,
seek reset request, and evidence timestamp display. API integration tests use
fake pipeline collaborators. Docker browser verification checks that Origin and
Product media times stay within one frame while a deliberately slow fake
inference response is active.

## Scope

This design changes only Upload Video simulation and its transport to FastAPI.
Upload Image and Webcam behavior remain intact. It does not promise that the
current CPU models always produce a final verdict within 1200 ms when the
plate is unreadable; it guarantees smooth synchronized visualization and safe,
truthful timing while targeting that verdict window when evidence is available.

