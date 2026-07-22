# Registry Tab + Webcam Fix

**Date:** 2026-07-14  
**Status:** Approved

## Registry (new Input Mode)

- Sidebar mode **`Registry`** alongside Upload Image / Upload Video / Webcam.
- Isolated main-area UI: no detector/OCR/video session.
- Gallery cards: photo (or placeholder), plate, brand, color, Delete.
- Add form: plate, brand, color, optional image upload.
- Persist rows in `main/data/database.csv`; photos in `main/data/registry/photos/<normalized_plate>.jpg`.
- After add/delete: rewrite CSV, reload `DatabaseMatcher` in dashboard session state so other modes see updates.
- Existing CSV rows without photos show a placeholder.

## Webcam fix

- Root cause: Streamlit runs in Docker; `cv2.VideoCapture(0)` cannot see the host camera.
- Replace server-side OpenCV capture with a browser `getUserMedia` component that samples frames to existing `POST /demo/frame` (same path as Upload Video).
- Keep ParkingSession multi-frame gate; do not use single-shot `st.camera_input` as the only path.
- Clear UX when permission denied.

## Non-goals

- Edit existing registry fields (add/delete only).
- Change detect/OCR/lock algorithms.
- Device index picker beyond default browser camera (unless trivial).
