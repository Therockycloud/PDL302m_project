# Guidelines for Project Reports (Reports 1 - 4)

This folder contains the official reports required for the DPL302m ongoing assessments. Each report represents a milestone in the project's development.

---

## General Design & Content Constraints (Mandatory)
* **Strict Rectangular Design:** All visual layouts, cards, buttons, badges, and images in HTML presentations or Web UIs must have completely sharp corners. **Never use rounded corners (always set `border-radius: 0px` or omit it entirely).**
* **Mandatory Literature Review:** Every report and its corresponding presentation slide deck must include a section/slide detailing the related papers and existing projects that were referenced during development (literature review).

---


## 1. [Report 1: Project Proposal](file:///Users/konalyn/Documents/FPT%20Materials/DPL302m/PDL302m_project/docs/Report_1_Proposal.md) (10%)
* **Objective:** Define the project scope, problem statement, team roles, and system architecture.
* **Reading Script:** A detailed presentation script for the team is available at [Presentation_Script.md](file:///Users/konalyn/Documents/FPT%20Materials/DPL302m/PDL302m_project/docs/Presentation_Script.md).
* **Key Sections Required:**
  1. **Problem Statement:** The issue of car theft, plate swapping, and fake license plates in Vietnamese parking lots.
  2. **Project Scope:** Focus on cars, targeting a latency of $<1$ second per vehicle and detection/recognition accuracy $\ge 90\%$.
  3. **High-Level System Architecture:** Flowchart showing: Input Frame $\rightarrow$ License Plate Detection (YOLOv8) + Vehicle Detection $\rightarrow$ Plate OCR + Vehicle Brand/Color Classification $\rightarrow$ Matching Verification against local CSV database $\rightarrow$ Security Action (Open Barie or Alert Warning).
  4. **Technology Stack:** PyTorch, Ultralytics YOLOv8, PaddleOCR/EasyOCR, TensorFlow/Keras (for classifiers), Streamlit (for Web Demo).
  5. **Team Work Breakdown Structure (WBS):** Allocation of tasks among the 3-4 members.

---

## 2. [Report 2: Data Tasks](file:///Users/konalyn/Documents/FPT%20Materials/DPL302m/PDL302m_project/docs/Report_2_Data_Tasks.md) (10%)
* **Objective:** Collect, process, and analyze the data required for all sub-models.
* **Key Sections Required:**
  1. **Data Sources:** 
     - Car License Plates: [Kaggle Car License Plate Dataset](https://www.kaggle.com/datasets/datnguyen1111/vietnamese-car-license-plate-detection).
     - Car Brands and Colors: Kaggle datasets (e.g., Stanford Cars or scraped Google Images for local brands like VinFast, Toyota, Hyundai, Honda, Mazda).
  2. **Data Wrangling & Preprocessing:** Image resizing, grayscale conversion for OCR, normalization.
  3. **Data Augmentation:** Techniques used to handle variations in lighting (day/night, headlights), shadows, and camera angles.
  4. **Exploratory Data Analysis (EDA):** Charts showing distribution of car brands, colors, plate locations, and size distribution of bounding boxes.

---

## 3. [Report 3: Model & Results](file:///Users/konalyn/Documents/FPT%20Materials/DPL302m/PDL302m_project/docs/Report_3_Model_Results.md) (30%)
* **Objective:** Document the training, optimization, and evaluation of all models.
* **Key Sections Required:**
  1. **Model Architecture Details:**
     - YOLOv8-nano/small for License Plate Bounding Box Detection.
     - EasyOCR/PaddleOCR for character recognition.
     - ResNet50 (Transfer Learning) for Car Brand Classification.
     - MobileNetV2 (Transfer Learning) for Car Color Classification.
  2. **Training & Hyperparameter Tuning:** Regularization strategies (Dropout, BatchNorm), learning rate schedules, optimizer choices (Adam/SGD), and tuning results (using Keras Tuner/Optuna).
  3. **Evaluation Metrics:**
     - YOLOv8: mAP@0.5, Precision, Recall.
     - OCR: Character-level accuracy and Word-level accuracy.
     - Classifiers: Accuracy, Confusion Matrix, and F1-score.

---

## 4. [Report 4: Final Report & Group Defense](file:///Users/konalyn/Documents/FPT%20Materials/DPL302m/PDL302m_project/docs/Report_4_Final_Report.md) (10%)
* **Objective:** Summarize the integrated system, test outcomes, and final demo interface.
* **Key Sections Required:**
  1. **Pipeline Integration:** How the asynchronous/synchronous steps run in sequence (bounding boxes crop $\rightarrow$ inputs to classifiers).
  2. **Local Database Verification:** CSV database schema mapping License Plate $\leftrightarrow$ Brand, Color.
  3. **Streamlit UI Demo:** Screen captures of the system handling correct plates vs fake/mismatched plates (displaying flashing RED alarm and warning beep).
  4. **Inference Latency & FPS Performance:** Processing time breakdown for each model and overall end-to-end latency.
  5. **Project Retrospective & Future Improvements:** What went well, challenges, and next-generation ideas (e.g., cloud sync, multi-camera coordination).
