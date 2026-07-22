# Related Work — How ALPR/Parking Systems Are Built (and why our choices fit Vietnam)

What commercial and developed-country Automatic License-Plate Recognition (ALPR)
/ smart-parking systems use, and how this project maps onto it.

## The standard pipeline (matches ours)

| Stage | Industry / state-of-the-art | This project |
|---|---|---|
| **Detection** | Cascaded **vehicle → plate → character** with the **YOLO** family (v8/v9/nano); suppresses false positives, runs on constrained hardware | 2-stage **YOLOv8n** vehicle → plate (mAP50 0.99) |
| **OCR** | **Segmentation-free CRNN + CTC** — LPRNet (first real-time), CR-NET/SCR-Net, or Tesseract/PaddleOCR for multi-script | **PaddleOCR**, a CRNN+CTC engine (Benchmark C: 81% vs EasyOCR 0%) |
| **Attributes (make/colour/type)** | **Secondary classifiers** after the detector — NVIDIA **DeepStream** uses a primary detector (DashCamNet/TrafficCamNet) + secondary **VehicleMakeNet (ResNet18)** for make and a colour classifier | **MobileNetV3-Small** colour classifier on the vehicle crop (brand dropped) |
| **Frame efficiency** | Process **one representative frame per vehicle** — *Visual Rhythm* / *Accumulative Line Analysis* cut compute ~3× vs naive per-frame | **Frame sampling + parking-trigger gate** — heavy pipeline runs once per parked car |
| **Edge hardware** | **Jetson Nano/TX2/Orin, Raspberry Pi 4B, FPGA**; on-prem | Nano/mobile models exported to **ONNX**, sized for the same edge class |

**Commercial reference points:** Adaptive Recognition **Carmen Nano** (on-prem ANPR
on Jetson, extracts plate + make/model/colour/type), **Plate Recognizer**, and
**NVIDIA DeepStream** parking reference designs (primary detector + secondary
colour/type classifiers, per-vehicle tracking).

## Why these choices fit Vietnam

- **Cost:** commercial stacks (Carmen, DeepStream + Jetson) carry licence/hardware
  cost. Our stack (YOLOv8n + PaddleOCR + MobileNetV3, all open-source, ONNX) gives
  the same cascaded architecture at near-zero licence cost — appropriate for
  local deployment budgets.
- **Vietnamese plates:** the training data includes VN plates (`xemay` motorbikes,
  `CarLongPlate` cars), covering the **2-line motorbike** format that Western
  single-line systems don't emphasise.
- **Garage CCTV conditions:** fluorescent lighting washes colours out — addressed
  with **domain-randomisation augmentation** rather than an expensive
  domain-specific dataset.
- **Same efficiency idea:** our parking-trigger gating is the open equivalent of
  the industry's "one representative frame per vehicle" optimisation, which is
  what makes edge deployment feasible.

## Sources

- [ALPR using YOLO + Small Language Models for Edge (ResearchGate, 2025)](https://www.researchgate.net/publication/398519776_Automatic_License_Plate_Recognition_ALPR_using_YOLO_and_Small_Language_Models_for_Edge_Explainable_Surveillance_Systems)
- [Layout-Independent ALPR Based on the YOLO detector (arXiv 1909.01754)](https://arxiv.org/pdf/1909.01754)
- [Memory/Time-Efficient ALPR Based on YOLOv5 (PMC9317241)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9317241/)
- [Advanced deep learning for automated LPR (Nature Sci. Reports, 2025)](https://www.nature.com/articles/s41598-025-24967-9)
- [Efficient Video-Based ALPR Using YOLO and Visual Rhythm (arXiv 2501.02270)](https://arxiv.org/pdf/2501.02270)
- [NVIDIA DeepStream — Real-Time License Plate Detection & Recognition](https://developer.nvidia.com/blog/creating-a-real-time-license-plate-detection-and-recognition-app/)
- [NVIDIA Transfer Learning Toolkit — VehicleMakeNet (ResNet18)](https://developer.nvidia.com/blog/training-custom-pretrained-models-using-tlt/)
- [DeepStream Reference Design — Automatic Parking Lot Vehicle Registration (RidgeRun)](https://developer.ridgerun.com/wiki/index.php/DeepStream_Reference_Designs/Reference_Designs/Automatic_Parking_Lot_Vehicle_Registration)
- [Carmen Nano — ANPR/LPR for NVIDIA Jetson (Adaptive Recognition)](https://adaptiverecognition.com/products/carmen-nano/)
- [Plate Recognizer — ALPR Research](https://platerecognizer.com/alpr-research/)
