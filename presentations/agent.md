# Guidelines for Project Presentations & Slide Decks

This directory contains resources for preparing the oral group presentations (Reports 1 to 4) required by the DPL302m course.

---

## General Design & Content Constraints (Mandatory)
* **Strict Rectangular Design:** All visual layouts, cards, buttons, badges, and images in HTML presentations or Web UIs must have completely sharp corners. **Never use rounded corners (always set `border-radius: 0px` or omit it entirely).**
* **Mandatory Literature Review:** Every report and its corresponding presentation slide deck must include a section/slide detailing the related papers and existing projects that were referenced during development (literature review).

---

## Slide Structure Recommendations

### 1. Report 1 Presentation: Project Proposal (10-16 Slides)
* **Slide 1:** Title Slide (Project Name, Team Members, Student IDs, Instructor Name).
* **Slide 2:** Motivation / Problem Statement (Car theft/plate swapping issue in Vietnam, loss of property).
* **Slide 3:** Proposed Solution (Multi-task cross-verification: Plates + Brand + Color).
* **Slide 4:** High-Level Architecture Diagram (Flowchart showing pipeline stages).
* **Slide 5:** Technology Stack & Rationale (YOLOv8, PaddleOCR, TensorFlow/Keras, Streamlit).
* **Slide 6:** Dataset Strategy (VinBigData, Kaggle License Plates, scraped car brands).
* **Slide 7:** Project Scope & Key Metrics (Latency < 1s, Accuracy >= 90%).
* **Slide 8:** Related Papers & Reference Projects (Literature Review).
* **Slide 9:** Work Breakdown Structure (WBS) & Role Assignment.
* **Slide 10:** Risk Management (Handling poor lighting, angled plates).
* **Slide 11:** Gantt Chart / Project Timeline.
* **Slide 12:** Q&A.

---

### 2. Report 2 Presentation: Data Tasks (10-12 Slides)
* **Slide 1:** Title Slide.
* **Slide 2:** Data Collection Summary (Total images, source links, splits: Train/Val/Test).
* **Slide 3:** Exploratory Data Analysis (EDA) - Bounding Box Sizes & Distributions.
* **Slide 4:** EDA - Class Balance for Car Brands & Colors.
* **Slide 5:** Challenges in Data (Low resolution, blur, night shots).
* **Slide 6:** Data Wrangling (Cropping plates, resizing classifiers inputs to 224x224).
* **Slide 7:** Augmentation Strategies (Brightness scaling, affine transforms, Gaussian blur).
* **Slide 8:** Sample processed inputs for License Plate detection, OCR, and Classifiers.
* **Slide 9:** Data Pipeline Verification & Summary.
* **Slide 10:** Q&A.

---

### 3. Report 3 Presentation: Model & Results (15-18 Slides)
* **Slide 1:** Title Slide.
* **Slide 2:** Model Architectures Overview.
* **Slide 3:** License Plate Detection (YOLOv8 configuration, hyperparameters, training loss curve).
* **Slide 4:** YOLOv8 Bounding Box Localization Results (mAP@0.5, Precision, Recall).
* **Slide 5:** Character Recognition (PaddleOCR/EasyOCR integration, post-processing rules).
* **Slide 6:** OCR Performance Metrics (Word accuracy, sample reads).
* **Slide 7:** Vehicle Brand Classifier (ResNet50 architecture, Transfer Learning vs. Scratch).
* **Slide 8:** Brand Classifier Training Logs (Accuracy/Loss curves, over-fitting prevention).
* **Slide 9:** Vehicle Color Classifier (MobileNetV2 architecture).
* **Slide 10:** Color Classifier Training Logs.
* **Slide 11:** Hyperparameter Tuning Experiments (Keras Tuner/Optuna trials).
* **Slide 12:** Integrated Pipeline Test Metrics (Accuracy of combined classification).
* **Slide 13:** Model Failures & Error Analysis (CLO3).
* **Slide 14:** Q&A.

---

### 4. Report 4 Presentation: Final Defense (12-15 Slides)
* **Slide 1:** Title Slide.
* **Slide 2:** Overall System Architecture & Database Integration (CSV).
* **Slide 3:** Real-time Pipeline Processing (Threading / Asynchronous design).
* **Slide 4:** Video Demo / Live Webcam Demonstration (Showing correct vehicle vs. mismatched vehicle).
* **Slide 5:** Latency Breakdown (Detection ms, OCR ms, Classification ms, Matching ms).
* **Slide 6:** Accuracy on End-to-End Test Set (Fake Plate Detection Rate).
* **Slide 7:** Implementation Challenges & Solutions.
* **Slide 8:** Future Scope (Deployment on edge devices like Raspberry Pi, cloud sync).
* **Slide 9:** Summary of Contributions by each Team Member.
* **Slide 10:** References & Course Materials mapped.
* **Slide 11:** Q&A.
