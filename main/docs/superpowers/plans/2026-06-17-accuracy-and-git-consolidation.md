# Accuracy Improvement + Git Consolidation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use `- [ ]` checkboxes.

**Goal:** Dọn git về `main`, nâng độ chính xác Color (deploy) + Brand (experimental) qua làm sạch nhãn + recipe, dùng video thật làm phép thử miền, rồi cập nhật Report 2.

**Architecture:** Làm tuần tự theo phase; commit sau mỗi bước trên `main`. Giữ nguyên logic quyết định (matcher/decision_engine/parking_session). Test set đóng băng làm anchor so sánh.

**Tech Stack:** Python, TensorFlow/Keras, Ultralytics YOLO, OpenCV, pytest, git.

**Spec:** `docs/superpowers/specs/2026-06-17-model-accuracy-improvement-design.md`

**Quy ước chung:**
- Interpreter: dùng env theo memory (KMP_DUPLICATE_LIB_OK=TRUE). Train/eval chạy bằng python có TensorFlow.
- Mỗi task kết thúc bằng 1 commit. KHÔNG push trừ khi được yêu cầu.
- KHÔNG đụng: `src/utils/matching.py`, `src/engine/decision_engine.py`, `src/engine/parking_session.py`, và `data/processed/classifiers/*/test`.

---

## PHASE 0 — Git consolidation (housekeeping, làm trước)

### Task 0.1: Commit working tree đang dở
**Files:** không sửa code; chỉ git.

- [ ] **Step 1:** Xem trạng thái
  Run: `git status --short`
  Expected: thấy `main/docs/` (spec+plan mới), `presentations/Report_2_Presentation.html` (M), `Report_2_HUST_Style.html` (D), `Report_2_Script.md` (??), `professional_deck/images/` (??).
- [ ] **Step 2:** Commit tài liệu planning (của mình) trước
  ```bash
  git add main/docs/superpowers
  git commit -m "docs(plan): add accuracy-improvement spec + implementation plan"
  ```
- [ ] **Step 3:** XÁC NHẬN với user phần presentation R2 (M/D/??) là WIP hay commit luôn. Nếu commit:
  ```bash
  git add presentations/
  git commit -m "chore(report2): snapshot pending R2 presentation/script changes"
  ```
  Nếu là WIP chưa muốn commit → `git stash push -- presentations/` để working tree sạch trước khi merge.
- [ ] **Step 4:** Verify sạch
  Run: `git status --short` → Expected: trống.

### Task 0.2: Merge `fix-report-2` (chỉ sửa 1 doc)
**Files:** `docs/Report_2_Data_Tasks.md` (đến từ branch).

- [ ] **Step 1:** Xác nhận nội dung unique
  Run: `git show --stat a4ecd97` → Expected: chỉ `docs/Report_2_Data_Tasks.md`.
- [ ] **Step 2:** Merge
  ```bash
  git merge fix-report-2 -m "merge(fix-report-2): academic rewrite of Report_2_Data_Tasks.md"
  ```
  Expected: clean merge (main không đụng file này gần đây). Nếu conflict → mở file, giữ bản rewrite học thuật, `git add` + `git commit`.
- [ ] **Step 3:** Verify
  Run: `git log --oneline -1 -- docs/Report_2_Data_Tasks.md`

### Task 0.3: Gỡ worktree + xóa branch đã gộp
**Files:** không; chỉ git. **KHÔNG đụng `docs-and-presentations`, `test/streamlit-only`.**

- [ ] **Step 1:** Gỡ 2 worktree
  ```bash
  git worktree remove .worktrees/fix-report-2
  git worktree remove .worktrees/training2
  git worktree prune
  ```
- [ ] **Step 2:** Xóa các branch đã là ancestor/đã merge
  ```bash
  git branch -d fix-report-2 feature/improve-data-pipeline redesign/ui-content-code-overhaul training2
  ```
  Expected: `-d` thành công vì đều đã merged. (Nếu báo chưa merged → DỪNG, báo user.)
- [ ] **Step 3:** Verify còn lại đúng kỳ vọng
  Run: `git branch -vv`
  Expected: chỉ còn `main`, `docs-and-presentations`, `test/streamlit-only`.
- [ ] **Step 4:** (không commit — branch ops không tạo working change)

---

## PHASE 1 — Crop/domain consistency (classify trên vehicle crop)

> Vấn đề: path Upload-Image (`dashboard.py`) và API (`app.py`) classify **full frame**, lệch với lúc train (ảnh xe). Sửa: nếu có vehicle detector → crop xe lớn nhất → classify trên crop; không có → fallback full image (giữ tương thích).

### Task 1.1: Helper crop xe lớn nhất (TDD)
**Files:**
- Create: `src/utils/crop.py`
- Test: `tests/test_crop.py`

- [ ] **Step 1: Viết test thất bại**
  ```python
  # tests/test_crop.py
  import numpy as np
  from src.utils.crop import largest_vehicle_crop

  class _FakeDet:
      def __init__(self, dets): self._d = dets
      def detect(self, frame): return self._d

  def test_returns_largest_crop():
      img = np.zeros((100, 100, 3), dtype=np.uint8)
      det = _FakeDet([
          {"bbox": (0, 0, 10, 10), "crop": img[:10, :10]},
          {"bbox": (0, 0, 50, 60), "crop": img[:60, :50]},
      ])
      out = largest_vehicle_crop(img, det)
      assert out.shape == (60, 50, 3)

  def test_no_detection_returns_full_image():
      img = np.zeros((30, 40, 3), dtype=np.uint8)
      out = largest_vehicle_crop(img, _FakeDet([]))
      assert out.shape == img.shape
  ```
- [ ] **Step 2: Chạy test → FAIL**
  Run: `pytest tests/test_crop.py -v` → Expected: FAIL (module chưa có).
- [ ] **Step 3: Cài đặt tối thiểu**
  ```python
  # src/utils/crop.py
  """Pick the largest detected vehicle crop, falling back to the full frame."""
  from __future__ import annotations
  import numpy as np

  def largest_vehicle_crop(image: np.ndarray, vehicle_detector) -> np.ndarray:
      """Return the crop of the largest detected vehicle, or the full image.

      vehicle_detector must expose ``detect(frame) -> list[dict]`` with each
      dict having ``bbox=(x1,y1,x2,y2)`` and ``crop`` (np.ndarray).
      """
      if vehicle_detector is None:
          return image
      try:
          dets = vehicle_detector.detect(image)
      except Exception:
          return image
      if not dets:
          return image
      best = max(dets, key=lambda d: (d["bbox"][2] - d["bbox"][0]) * (d["bbox"][3] - d["bbox"][1]))
      crop = best.get("crop")
      if crop is None or crop.size == 0:
          return image
      return crop
  ```
- [ ] **Step 4: Chạy test → PASS**
  Run: `pytest tests/test_crop.py -v` → Expected: PASS.
- [ ] **Step 5: Commit**
  ```bash
  git add src/utils/crop.py tests/test_crop.py
  git commit -m "feat(crop): largest_vehicle_crop helper for inference consistency"
  ```

### Task 1.2: Dùng helper trong dashboard + API
**Files:**
- Modify: `src/ui/dashboard.py` (vùng classify ảnh, quanh dòng 260-274 — `brand_clf.predict(image)` / `color_clf.predict(image)`)
- Modify: `src/api/app.py` (vòng lặp per-plate — `brand_clf.predict(image)` / `color_clf.predict(image)`)

- [ ] **Step 1:** Trong cả 2 file, lấy vehicle detector nếu có (đã có `VehicleDetector` được dựng cho ParkingSession trong dashboard; với API thêm tải `VehicleDetector` ở lifespan dùng `cfg["detector"]`). Tạo `clf_input` một lần trước vòng phân loại:
  ```python
  from src.utils.crop import largest_vehicle_crop
  veh_det = models.get("vehicle_detector")  # dashboard: _models.get(...) cho API
  clf_input = largest_vehicle_crop(image, veh_det)
  ```
- [ ] **Step 2:** Thay `predict(image)` → `predict(clf_input)` cho brand và color (cả 2 file).
- [ ] **Step 3:** Đảm bảo có khóa `vehicle_detector` trong dict models. Dashboard: thêm `models["vehicle_detector"] = vehicle_det` nơi đã tạo `vehicle_det`. API: trong `_lifespan`, thêm `_models["vehicle_detector"] = VehicleDetector(model_path=<detector model>, conf=cfg["detector"].get("conf_threshold",0.3))`.
- [ ] **Step 4:** Test khói: chạy app/dashboard import không lỗi
  Run: `python -c "import src.api.app"` (với PYTHONPATH đúng) → Expected: không exception.
- [ ] **Step 5: Commit**
  ```bash
  git add src/ui/dashboard.py src/api/app.py
  git commit -m "fix(inference): classify brand/color on vehicle crop, not full frame"
  ```

---

## PHASE 2 — Augmentation tách theo task

### Task 2.1: Tham số hóa augmentation trong `load_split_dataset`
**Files:**
- Modify: `src/datasets/vehicle_dataset.py` (hàm `load_split_dataset`, dòng 5-51)
- Modify: `train.py` (2 chỗ gọi `load_split_dataset`, ~dòng 276 và 343)
- Test: `tests/test_dataset.py` (thêm case)

- [ ] **Step 1:** Thêm tham số `task: str = "color"` vào `load_split_dataset`. Xây `augmentation_model` theo task:
  ```python
  def _build_aug(task: str):
      layers = [
          tf.keras.layers.RandomFlip("horizontal"),
          tf.keras.layers.RandomRotation(0.1),
          tf.keras.layers.RandomZoom(0.1),
          tf.keras.layers.RandomTranslation(0.1, 0.1),
      ]
      if task == "color":
          # Màu: chỉ độ sáng/tương phản nhẹ. CẤM hue/saturation.
          layers += [tf.keras.layers.RandomBrightness(0.15, value_range=(0.0, 1.0)),
                     tf.keras.layers.RandomContrast(0.15)]
      else:  # brand: nhạy hình/logo → mạnh hơn
          layers += [tf.keras.layers.RandomContrast(0.3)]
      return tf.keras.Sequential(layers)
  ```
  Thay dòng 24-28 dùng `augmentation_model = _build_aug(task)`.
- [ ] **Step 2:** Cập nhật 2 call site trong `train.py`: brand → `load_split_dataset(..., task="brand")`, color → `load_split_dataset(..., task="color")`.
- [ ] **Step 3:** Thêm test shape/range vẫn đúng:
  ```python
  # tests/test_dataset.py (thêm)
  def test_load_split_accepts_task(tmp_path):
      import tensorflow as tf
      from src.datasets.vehicle_dataset import load_split_dataset
      # tạo cây train/val/test tối thiểu 1 ảnh/lớp x 2 lớp
      import numpy as np, os
      from PIL import Image
      for split in ("train","val","test"):
          for cls in ("a","b"):
              d = tmp_path/split/cls; d.mkdir(parents=True)
              Image.fromarray(np.zeros((8,8,3),np.uint8)).save(d/"x.jpg")
      tr,_,_,names = load_split_dataset(str(tmp_path), batch_size=2, img_height=8, img_width=8, task="brand")
      x,y = next(iter(tr))
      assert float(tf.reduce_max(x)) <= 1.0 + 1e-3
      assert names == ["a","b"]
  ```
- [ ] **Step 4:** Run: `pytest tests/test_dataset.py -v` → Expected: PASS.
- [ ] **Step 5: Commit**
  ```bash
  git add src/datasets/vehicle_dataset.py train.py tests/test_dataset.py
  git commit -m "feat(aug): per-task augmentation (color=geom+brightness, brand=stronger)"
  ```

---

## PHASE 3 — Label audit (nhẹ, human-in-the-loop)

### Task 3.1: Script liệt kê ảnh train nghi sai nhãn
**Files:**
- Create: `scripts/audit_labels.py`

- [ ] **Step 1:** Viết script: load model `.keras` hiện có (color hoặc brand theo arg), chạy predict trên **train split**, in các ảnh model đoán **sai với confidence ≥ ngưỡng** (sắp giảm dần), kèm path + nhãn folder + nhãn dự đoán + conf. Ghi CSV ra `data/models/<task>_label_suspects.csv`.
  ```python
  # scripts/audit_labels.py  (CLI: python scripts/audit_labels.py color --thr 0.8)
  # - dựng model qua src.models.classifiers, load_weights từ data/models/<task>_classifier.keras
  # - đọc data/processed/classifiers/<colors|brands>/train bằng image_dataset_from_directory(shuffle=False)
  # - so argmax(pred) vs nhãn; lọc sai & conf>=thr; xuất CSV path,true,pred,conf
  ```
- [ ] **Step 2:** Chạy thử
  Run: `python scripts/audit_labels.py color --thr 0.8`
  Expected: in N ảnh nghi sai + ghi CSV.
- [ ] **Step 3:** Người review CSV → sửa nhãn (di chuyển ảnh sang folder đúng) hoặc xóa ảnh rác. **Chỉ trong train/val.**
- [ ] **Step 4: Commit** (script + thay đổi nhãn nếu có)
  ```bash
  git add scripts/audit_labels.py data/processed/classifiers
  git commit -m "chore(data): label audit script + fix mislabeled train/val images"
  ```

---

## PHASE 4 — Recipe fine-tune + retrain Color + eval

### Task 4.1: EarlyStopping theo macro-F1 + retrain color
**Files:**
- Modify: `train.py` (callbacks fine-tune / trainer) — đổi monitor sang val macro-F1 nếu khả dụng, else giữ val_accuracy.

- [ ] **Step 1:** Trong `train.py`, thêm metric F1 vào compile color/brand (dùng `tf.keras.metrics.F1Score(average="macro")` trên softmax) và EarlyStopping `monitor="val_f1_score", mode="max", patience=5, restore_best_weights=True`.
- [ ] **Step 2:** Retrain color (clean data + aug mới + fine-tune):
  Run: `python train.py color --fine_tune`
  Expected: chạy 2-stage, lưu `data/models/color_classifier.keras`.
- [ ] **Step 3:** Eval trên test ĐÓNG BĂNG:
  Run: `python src/engine/run_evaluation.py` (hoặc lệnh eval hiện có) → ghi `data/models/color_classifier_test_report.json`.
- [ ] **Step 4:** Ghi lại số before/after color (54.2% → ?). So sánh với baseline; chỉ giữ nếu **cao hơn**.
- [ ] **Step 5: Commit**
  ```bash
  git add train.py data/models/color_classifier.keras data/models/color_classifier_test_report.json
  git commit -m "feat(train): color retrain (clean+aug+F1 early-stop); update test report"
  ```

---

## PHASE 5 — Video thật làm phép thử miền

### Task 5.1: Runner chạy pipeline lên `parking_case_real.mp4`
**Files:**
- Create: `scripts/run_on_video.py`

- [ ] **Step 1:** Viết script: dựng `VehicleDetector + PlateReader + color_clf + DecisionEngine + ParkingTrigger + ParkingSession` (tái dùng cách dashboard dựng), đọc video bằng cv2, feed từng frame vào `session.process_frame`, in `state` + `decision` cuối + thời điểm trigger. In rõ: plate đọc được, color dự đoán (kỳ vọng **YELLOW**), action.
  CLI: `python scripts/run_on_video.py --source data/test/parking_case_real.mp4`
- [ ] **Step 2:** Chạy
  Run: `python scripts/run_on_video.py --source data/test/parking_case_real.mp4`
  Expected: pipeline chạy hết video, in 1 decision; color ≈ YELLOW, có plate. Nếu trigger không bắn → ghi chú để calibrate ROI (`scripts/calibrate_roi.py`).
- [ ] **Step 3:** Ghi kết quả quan sát vào `docs/superpowers/specs/` (1 đoạn: domain-gap thật).
- [ ] **Step 4: Commit**
  ```bash
  git add scripts/run_on_video.py docs/superpowers
  git commit -m "feat(eval): end-to-end runner on real parking video + domain-gap note"
  ```

---

## PHASE 6 — Brand experimental (KHÔNG đụng hệ thống)

### Task 6.1: Retrain brand với clean + aug mạnh, báo cáo
**Files:** dùng lại `train.py`, `scripts/audit_labels.py`.

- [ ] **Step 1:** Audit nhãn brand: `python scripts/audit_labels.py brand --thr 0.8` → review/sửa.
- [ ] **Step 2:** Retrain: `python train.py brand --fine_tune`
- [ ] **Step 3:** Eval frozen test → `data/models/brand_classifier_test_report.json`; ghi before/after (35.3% → ?).
- [ ] **Step 4:** Xác nhận matcher/decision/session KHÔNG đổi (brand vẫn ngoài hệ thống)
  Run: `grep -rn "brand" src/utils/matching.py src/engine/decision_engine.py src/engine/parking_session.py` → Expected: trống.
- [ ] **Step 5: Commit**
  ```bash
  git add train.py data/models/brand_classifier.keras data/models/brand_classifier_test_report.json data/processed/classifiers
  git commit -m "experiment(brand): retrain (clean+strong aug); report-only, not wired to system"
  ```

---

## PHASE 7 — Cập nhật Report 2 (bước cuối)

### Task 7.1: Đồng bộ số + sửa lập luận + verify citation
**Files:**
- Modify: `presentations/Report_2_Presentation.html` (Slide 9: số color/brand + Macro-F1)
- Modify: `presentations/Report_2_Script.md` (đồng bộ narration nếu cần)

- [ ] **Step 1:** Thay 48.3 → số color cuối; 32.8 → số brand cuối; cập nhật Macro-F1 tương ứng (grep `48.3`, `32.8`, `12.5%` trong HTML).
- [ ] **Step 2:** Sửa lập luận brand: 35% đo trên test **sạch** → vấn đề là fine-grained + ít data (blur chỉ làm tệ thêm), thay vì đổ cho "ảnh giám sát mờ".
- [ ] **Step 3:** Verify citation "Yang et al. (2025)" (và Krause 2013, Chen 2014) bằng web — nếu Yang 2025 không tồn tại, thay bằng nguồn augmentation thật.
- [ ] **Step 4:** Mở HTML kiểm tra hiển thị (không vỡ layout).
- [ ] **Step 5: Commit**
  ```bash
  git add presentations/Report_2_Presentation.html presentations/Report_2_Script.md
  git commit -m "docs(report2): sync slide metrics to final models + fix brand rationale + verify citations"
  ```

---

## Verification (end-to-end)
- `pytest -q` → tất cả test pass (gồm `test_crop.py`, `test_dataset.py`).
- `git branch -vv` → còn `main` + 2 branch stripped (đã để yên).
- `data/models/color_classifier_test_report.json` → accuracy > 54.2%.
- `python scripts/run_on_video.py --source data/test/parking_case_real.mp4` → ra decision, color≈YELLOW.
- `grep -rn "brand" src/utils/matching.py src/engine/decision_engine.py src/engine/parking_session.py` → trống (hệ thống không dùng brand).
- Report 2 hiển thị số khớp test report cuối.

## Self-review notes (gaps để ý khi thực thi)
- Chữ ký gọi `load_split_dataset` trong `train.py` cần khớp khi thêm `task=` (kiểm tra kwargs thực tế ~dòng 276, 343).
- Path "Upload Image" trong `dashboard.py` cần biến `models`/`image` đúng scope của hàm chứa dòng 260-274.
- API cần thực sự tải `VehicleDetector` ở lifespan trước khi dùng `largest_vehicle_crop`.
- `tf.keras.metrics.F1Score` cần TF ≥ 2.12; nếu thiếu → fallback monitor `val_accuracy`.
