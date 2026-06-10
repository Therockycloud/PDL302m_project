# Project Plan: README Translation, Showcase Images, and Reports 2 & 3

This project plan details the implementation strategy for translating the onboarding README to Vietnamese, generating high-fidelity mockup screenshots of the Streamlit dashboard, creating slide deck presentations for Report 2 (Data Tasks) and Report 3 (Models & Results), and updating Report 4 with visual showcases.

## Project Type: WEB & BACKEND

---

## Success Criteria
1. **Vietnamese README:** The onboarding guide in `main/README.md` is translated into natural, professional Vietnamese while preserving all multi-platform commands and troubleshooting details.
2. **Showcase Images:** Two realistic, high-fidelity mockups of the Streamlit dashboard (Verified and Alarm states) generated and placed in `presentations/`.
3. **HTML Presentations:** `Report_2_Presentation.html` and `Report_3_Presentation.html` generated in the requested **Light, High-Contrast Editorial style** with **strict sharp corners (`border-radius: 0px`)** and the **Purple Ban** respected.
4. **Report 4 Update:** `Report_4_Presentation.html` updated with the generated dashboard showcase images.
5. **Quality Gate:** Passes the Antigravity checklist validator script.

---

## Tech Stack
*   **Presentations:** HTML5, CSS3 (Vanilla, Editorial Serif & Sans-Serif font pairs), JavaScript (Slide navigation & interactive controls).
*   **Documentation:** GitHub Flavored Markdown (Vietnamese translation).
*   **Media Generation:** High-fidelity AI image generation tool.

---

## File Structure

```
.
├── main/
│   └── README.md                      # [MODIFY] Translate to Vietnamese
└── presentations/
    ├── img_dashboard_verified.png      # [NEW] Mockup image for verified entry
    ├── img_dashboard_alert.png         # [NEW] Mockup image for security alert
    ├── Report_2_Presentation.html     # [NEW] Slide deck for Data Tasks
    ├── Report_3_Presentation.html     # [NEW] Slide deck for Model & Results
    └── Report_4_Presentation.html     # [MODIFY] Embed new showcase images
```

---

## Task Breakdown

### Phase 1: Showcase Images Generation
#### Task 1: Generate Streamlit Dashboard Mockup Images
*   **Agent:** `frontend-specialist`
*   **Skills:** `frontend-design`
*   **Priority:** P0
*   **Dependencies:** None
*   **Description:** Generate two high-fidelity visual representations of the system UI:
    1.  Streamlit dashboard in a verified state with a green barrier indicator.
    2.  Streamlit dashboard in a mismatch/alarm state with a flashing red alert header.
*   **INPUT:** UI mockup prompt strings.
*   **OUTPUT:** `presentations/img_dashboard_verified.png` and `presentations/img_dashboard_alert.png`.
*   **VERIFY:** Image files exist in the directory and correctly render the dashboard visual states.

---

### Phase 2: Documentation
#### Task 2: Translate README.md to Vietnamese
*   **Agent:** `documentation-writer`
*   **Skills:** `i18n-localization`
*   **Priority:** P1
*   **Dependencies:** None
*   **Description:** Translate the entire `main/README.md` to Vietnamese. Maintain all code blocks, Conda command references, variables, and path structures exactly as they are.
*   **INPUT:** Current English `main/README.md`.
*   **OUTPUT:** Updated Vietnamese `main/README.md`.
*   **VERIFY:** Ensure commands for macOS/Linux/Windows are correct and readable.

---

### Phase 3: Presentation Slide Decks
#### Task 3: Create Report 2 Presentation (Data Tasks)
*   **Agent:** `frontend-specialist`
*   **Skills:** `frontend-design`
*   **Priority:** P2
*   **Dependencies:** Task 1
*   **Description:** Create the HTML slide deck for Report 2: Data Tasks.
*   **INPUT:** Slide outline (Data collection, splits, EDA distributions, wrangling, crop resize, augmentations, Q&A).
*   **OUTPUT:** `presentations/Report_2_Presentation.html`.
*   **VERIFY:** Open slides, verify `border-radius: 0px`, verify keyboard navigation, and check contrast levels.

#### Task 4: Create Report 3 Presentation (Model & Results)
*   **Agent:** `frontend-specialist`
*   **Skills:** `frontend-design`
*   **Priority:** P2
*   **Dependencies:** Task 1
*   **Description:** Create the HTML slide deck for Report 3: Model & Results.
*   **INPUT:** Slide outline (YOLOv8 config, mAP localization, EasyOCR post-processing, EfficientNet-B0 brand logs, MobileNetV3-Small color logs, tuning, failures/diagnostics, Q&A).
*   **OUTPUT:** `presentations/Report_3_Presentation.html`.
*   **VERIFY:** Open slides, verify sharp geometry, check color palette, and test keyboard triggers.

#### Task 5: Upgrade Report 4 Presentation
*   **Agent:** `frontend-specialist`
*   **Skills:** `frontend-design`
*   **Priority:** P3
*   **Dependencies:** Task 1
*   **Description:** Update `presentations/Report_4_Presentation.html` to replace raw code blocks or text tables in Slide 9 with the newly generated high-fidelity mockup images (`img_dashboard_verified.png` and `img_dashboard_alert.png`).
*   **INPUT:** `presentations/Report_4_Presentation.html` and mock images.
*   **OUTPUT:** Updated `presentations/Report_4_Presentation.html`.
*   **VERIFY:** Confirm images display correctly in Slide 9 when navigating.

#### Task 6: Git Remote Synchronization & Push
*   **Agent:** `devops-engineer`
*   **Skills:** `bash-linux`
*   **Priority:** P3
*   **Dependencies:** None
*   **Description:** Configure the Git remote repository address and push both the `main` and `test/streamlit-only` branches to ensure all deliverables are synchronized with the host.
*   **INPUT:** Remote repository URL.
*   **OUTPUT:** Pushed branches on the remote repository.
*   **VERIFY:** Verify remote tracking status using `git remote -v` and `git branch -a`.

---

## Phase X: Final Verification

Run the master project checklist validator to guarantee that the new HTML slides and translated documentation pass all rules:
```bash
/opt/homebrew/Caskroom/miniforge/base/bin/python .agents/scripts/checklist.py .
```

### Manual Visual Auditing
- Verify that **no purple/violet** hex codes are introduced.
- Verify that **all elements** have sharp corners (`border-radius: 0px` or `border-radius: none`).
- Open `Report_2_Presentation.html`, `Report_3_Presentation.html`, and `Report_4_Presentation.html` in a web browser to verify visual fidelity and navigation transitions.
