# Original User Request

## Initial Request — 2026-06-11T03:52:56Z

An upgrade to the Vehicle Anti-Theft System to transition it into a production-grade application featuring real-world parking video stream testing, a polished light-mode UI, a clean borderless design, and visual polish of presentation slides.

Working directory: /Users/konalyn/Documents/FPT Materials/DPL302m/PDL302m_project
Integrity mode: development

## Requirements

### R1. Light-Mode & Borderless UI Redesign
- Re-theme the Streamlit interface to use a light-mode color palette (light backgrounds, dark text, clean contrasting accents).
- Remove all border lines, card borders, and box outlines from visual containers in the custom CSS, resulting in a flat, premium design.

### R2. Real Parking Video Simulation
- The agent team must automatically find and download a short, open-source real-world video (`.mp4`) simulating a car entering a parking spot or gate, and save it in `main/data/test/sample_parking.mp4`.
- Add a "Play Default Parking Video" option in the UI to allow testing the E2E pipeline on this real video feed with a single click.

### R3. Pipeline Optimization for CPU
- Ensure the deep learning pipeline (YOLOv8-nano detector + EasyOCR + brand/color classifiers) is optimized for lightweight CPU execution to ensure smooth playback in the UI.

### R4. Presentation Slides Visual Audit & Polish
- Audit all HTML presentation slide files (located in `presentations/`) to find any missing images or empty visual placeholders.
- If a slide is missing an illustration or has an empty image slot, generate or extract high-quality relevant screenshots or diagrams (such as model structures or pipeline flows) and embed them into the slides to ensure a high professional standard.

## Acceptance Criteria

### UI styling
- [ ] Streamlit interface renders in a clean light theme.
- [ ] Card containers, metrics panels, and feed boxes have no visible borders or thick outlines.

### Video testing
- [ ] A real video file of a car parking exists at `main/data/test/sample_parking.mp4`.
- [ ] The Streamlit UI can run and process this default video, drawing bounding boxes and license plate text overlays in real-time.
- [ ] Latency metrics (FPS, avg latency) and matching results are dynamically updated in the UI during video playback.

### Presentation slides
- [ ] All presentation slides in `presentations/` are fully populated with relevant images/diagrams.
- [ ] No broken image links or empty visual placeholders exist in any slide.
