# Color Classifier Fine-Tune Report — Deployed Model (VCoR held-out test)

> Đánh giá tái lập (reproducible) cho model **đang chạy ở runtime** (`main/data/models/color_MobileNetV3Small.pt`), đo trên tập TEST giữ-riêng của VCoR (Kaggle), split 70/15/15 stratified seed=42 — cùng split dùng khi fine-tune trên Colab. Sinh bởi `main/scripts/eval_color_deployed.py`, tái sử dụng hàm load/split/eval từ `main/scripts/colab_train_color.py`.

> baseline frozen-backbone (data cũ, trước VCoR) ≈ 0.5508 (55.1%) — xem `baseline_note`.

**Data layout:** `vcor`  |  **Tổng ảnh (pool, 8 lớp):** 5881  |  **Test split:** 889 ảnh

**Test Accuracy (plain, no TTA):** 0.8526  
**Test Accuracy (TTA, hflip-averaged):** 0.8628  
**Test Macro-F1 (plain):** 0.8287  
**Test Macro-F1 (TTA):** 0.8407

## Per-Class Metrics (TTA predictions)

| Class | Precision | Recall |
|-------|-----------|--------|
| Black | 0.8395 | 0.7727 |
| Blue | 0.9545 | 0.9245 |
| Brown | 0.9800 | 0.8033 |
| Grey | 0.5966 | 0.7634 |
| Red | 0.9784 | 0.9927 |
| Silver | 0.6316 | 0.6154 |
| White | 0.8085 | 0.8736 |
| Yellow | 0.9762 | 0.9840 |

## Confusion Matrix (TTA)

Rows = true class, Columns = predicted class.

| | Black | Blue | Brown | Grey | Red | Silver | White | Yellow |
|---|---|---|---|---|---|---|---|---|
| Black | 68 | 4 | 0 | 15 | 0 | 1 | 0 | 0 |
| Blue | 4 | 147 | 0 | 3 | 0 | 3 | 1 | 1 |
| Brown | 5 | 1 | 98 | 12 | 2 | 1 | 1 | 2 |
| Grey | 3 | 1 | 1 | 71 | 1 | 14 | 2 | 0 |
| Red | 0 | 0 | 1 | 0 | 136 | 0 | 0 | 0 |
| Silver | 0 | 1 | 0 | 15 | 0 | 48 | 14 | 0 |
| White | 1 | 0 | 0 | 1 | 0 | 9 | 76 | 0 |
| Yellow | 0 | 0 | 0 | 2 | 0 | 0 | 0 | 123 |

## Levers (fine-tune recipe, Colab GPU)

- class_weighted_loss (inverse frequency, mean-normalized)
- label_smoothing=0.1
- test_time_augmentation (orig + hflip softmax average)

**Lưu ý trung thực:** số liệu trên đo trên VCoR (ảnh web sạch, không phải CCTV bãi xe thật) — hiệu năng triển khai thực tế sẽ thấp hơn do domain gap (ánh sáng/độ phân giải CCTV); xem caveat đầy đủ trong các báo cáo chính.
