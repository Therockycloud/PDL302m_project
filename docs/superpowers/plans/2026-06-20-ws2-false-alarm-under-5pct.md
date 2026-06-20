# WS-2: Báo động giả < 5% (confidence-gating + gộp cụm màu trung tính) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: subagent-driven-development. Sonnet thực thi từng task (`model: sonnet`), Opus verify (đặc biệt cổng FA<5% bằng `eval_security.py`). Steps checkbox `- [ ]`.

**Goal:** Kéo tỉ lệ báo động giả (false-alarm) từ 14.5% xuống **<5%** bằng 2 cơ chế, kèm **bằng chứng before/after** (bảng + biểu đồ PNG) cho report/presentation.

**Architecture:** `verify_vehicle` chỉ bật `color_warning` khi (a) màu lệch **khác cụm** (cụm trung tính Black/Grey/Silver/White coi như tương đương) **và** (b) confidence màu ≥ ngưỡng. Confidence được luồn từ `color_clf.predict` → `aggregate` → `verify_vehicle`.

**Spec nguồn:** [specs/2026-06-20-plate-read-approach-latency-fa-api-design.md](../specs/2026-06-20-plate-read-approach-latency-fa-api-design.md) §3 WS-2. Quyết định user: "Cả hai" cơ chế + bắt buộc chụp bằng chứng FA giảm mạnh.

**Đánh đổi đã biết:** giảm FA sẽ giảm detection (98.5%) — đo & báo cáo trung thực, KHÔNG giấu.

---

## File structure

| File | Thay đổi |
|---|---|
| `main/src/utils/matching.py` | `verify_vehicle(plate, color, color_conf=None)` + neutral-cluster + gating |
| `main/configs/config.yaml` | block `decision: {color_warn_conf: 0.60, neutral_colors: [Black,Grey,Silver,White]}` |
| `main/src/engine/decision_engine.py` | truyền `color_conf` (đã tính) vào `verify_vehicle` |
| `main/scripts/eval_security.py` | dòng 279 truyền `pred_conf`; đo lại before/after |
| `main/tests/test_matching.py` | test neutral-merge + gating |
| `docs/benchmarks/security_eval.md` + `.json` | số mới + so sánh |
| `docs/benchmarks/security_fa_before_after.png` (mới) | biểu đồ cột |
| `reports/documents/Report_4_Final_Report.md` §4.3 | cập nhật số + nhúng chart |

---

## Task 1: `verify_vehicle` — neutral-cluster + confidence-gating

**Files:** Modify `main/src/utils/matching.py`, `main/tests/test_matching.py`

- [ ] **Step 1 — Test đỏ.** Thêm vào `test_matching.py` (DB tạm có 1 xe RED `51A-001`, 1 xe GREY `51A-002`):
```python
def test_neutral_cluster_no_warning():
    # registered GREY, detected SILVER (both neutral) -> NO warning even high conf
    r = matcher.verify_vehicle("51A-002", "SILVER", 0.95)
    assert r["status"] == "AUTHORIZED" and r["color_warning"] is False

def test_cross_cluster_high_conf_warns():
    r = matcher.verify_vehicle("51A-001", "BLUE", 0.95)   # RED reg, BLUE det
    assert r["color_warning"] is True and r["action"] == "ALLOW_WARN"

def test_low_conf_no_warning():
    r = matcher.verify_vehicle("51A-001", "BLUE", 0.40)   # mismatch but conf<0.60
    assert r["color_warning"] is False and r["action"] == "ALLOW"

def test_exact_match_no_warning():
    r = matcher.verify_vehicle("51A-001", "RED", 0.95)
    assert r["color_warning"] is False
```
Run `... -m pytest tests/test_matching.py -q` → **FAIL**.

- [ ] **Step 2 — Implement.** Trong `matching.py`:
  - Hằng số module `NEUTRAL = {"BLACK","GREY","SILVER","WHITE"}` (đọc override từ config nếu có, else default).
  - Hằng số `COLOR_WARN_CONF = 0.60` (đọc từ config `decision.color_warn_conf` nếu có).
  - Helper `_colors_equivalent(c1, c2)`: `c1 == c2 or (c1 in NEUTRAL and c2 in NEUTRAL)`.
  - `verify_vehicle(self, detected_plate, detected_color, color_conf=None)`: nhánh plate-registered: nếu `_colors_equivalent(clean_color, registered_color)` → ALLOW (no warning). Ngược lại, chỉ ALLOW_WARN khi `color_conf is None or color_conf >= COLOR_WARN_CONF`; nếu conf < ngưỡng → ALLOW (no warning, vì màu không đáng tin để cảnh báo). Giữ nguyên nhánh UNREGISTERED.
  - `color_conf=None` mặc định để caller cũ (2 tham số) không gãy.

- [ ] **Step 3 — Run + cập nhật test cũ nếu lệch.** `... -m pytest tests/test_matching.py -q` → PASS. Nếu test cũ nào assert "neutral↔neutral mismatch → warning" thì cập nhật theo logic mới (ghi lý do).

- [ ] **Step 4 — Commit.**
```bash
git add main/src/utils/matching.py main/tests/test_matching.py
git commit -m "feat(decision): neutral-cluster equivalence + colour-confidence gating to cut false alarms (WS-2)"
```

---

## Task 2: Config + luồng color_conf qua decision_engine

**Files:** Modify `main/configs/config.yaml`, `main/src/engine/decision_engine.py`, `main/tests/test_decision_engine.py`

- [ ] **Step 1 — Config.** Thêm block:
```yaml
decision:
  color_warn_conf: 0.60
  neutral_colors: [Black, Grey, Silver, White]
```
- [ ] **Step 2 — Test đỏ.** Trong `test_decision_engine.py`: với lock-aware path, frame có `color_conf` thấp + màu lệch khác cụm → verdict `color_warning False` (vì gating). Dùng `_FakeMatcher` gọi `verify_vehicle` thật-logic hoặc kiểm engine truyền `color_conf` xuống (assert matcher nhận đúng conf). Run → FAIL.
- [ ] **Step 3 — Implement.** `_aggregate_lock_aware`: đổi `self.matcher.verify_vehicle(plate, color)` → `self.matcher.verify_vehicle(plate, color, color_conf)` (biến `color_conf` đã tính sẵn trong path này). Path legacy giữ 2 tham số.
- [ ] **Step 4 — Run.** `... -m pytest tests/test_decision_engine.py -q` → PASS.
- [ ] **Step 5 — Commit.**
```bash
git add main/configs/config.yaml main/src/engine/decision_engine.py main/tests/test_decision_engine.py
git commit -m "feat(decision): thread colour confidence into verify_vehicle; decision config block (WS-2)"
```

---

## Task 3: Đo lại an ninh (eval_security.py truyền conf) + số before/after

**Files:** Modify `main/scripts/eval_security.py`

- [ ] **Step 1 — Sửa truyền conf.** Dòng ~279: `t["result"] = matcher.verify_vehicle(t["plate"], pred_colour, pred_conf)`.
- [ ] **Step 2 — Chạy đo (Sonnet chạy, báo số).**
```bash
cd main && KMP_DUPLICATE_LIB_OK=TRUE /opt/homebrew/Caskroom/miniforge/base/bin/python scripts/eval_security.py
```
Ghi lại 4 số MỚI: plate-swap detection rate, miss, **false-alarm rate (mục tiêu <5%)**, unregistered. Đối chiếu số CŨ (detection 98.5%, FA 14.5%, unreg 100%).
- [ ] **Step 3 — Commit** (script + json/md output do script tự sinh):
```bash
git add main/scripts/eval_security.py docs/benchmarks/security_eval.md docs/benchmarks/security_eval.json
git commit -m "eval(security): measure FA/detection under new gated colour-warning logic (WS-2)"
```

> **CỔNG OPUS:** Opus tự chạy lại `eval_security.py`, xác nhận **FA < 5%**. Nếu FA chưa <5% → Opus quyết định tinh chỉnh (nâng `color_warn_conf`, hoặc xét lại cụm) rồi giao Sonnet sửa + đo lại. KHÔNG sang Task 4 tới khi FA<5% đo được.

---

## Task 4: Bằng chứng before/after (biểu đồ + reports)

**Files:** Create `main/scripts/plot_fa_before_after.py` (sinh PNG); Create `docs/benchmarks/security_fa_before_after.png`; Modify `reports/documents/Report_4_Final_Report.md` (§4.3), `docs/benchmarks/security_eval.md`

- [ ] **Step 1 — Script biểu đồ.** `plot_fa_before_after.py`: dùng matplotlib vẽ 2 nhóm cột (Before vs After) cho **False-alarm** và **Plate-swap detection** (số before lấy từ §4.3 cũ: FA 14.5%, det 98.5%; số after lấy từ `security_eval.json` mới). Lưu `docs/benchmarks/security_fa_before_after.png` (dpi≥150, có title/nhãn % trên cột).
- [ ] **Step 2 — Chạy sinh PNG.** `cd main && <py> scripts/plot_fa_before_after.py`. Xác nhận file PNG tồn tại, mở đọc được.
- [ ] **Step 3 — Cập nhật Report 4 §4.3:** bảng kết quả thêm cột/ghi chú **Before (14.5% FA / 98.5% det) → After (<số mới>)**, nhúng ảnh `![FA before/after](../../docs/benchmarks/security_fa_before_after.png)`, và **đoạn trung thực** nêu rõ đánh đổi: FA giảm mạnh nhờ gating+neutral-merge, detection giảm còn <số mới> (chủ yếu mất các ca tráo trong cụm trung tính — đúng giới hạn đã biết). Cập nhật `security_eval.md` tương tự.
- [ ] **Step 4 — Commit.**
```bash
git add main/scripts/plot_fa_before_after.py docs/benchmarks/security_fa_before_after.png reports/documents/Report_4_Final_Report.md docs/benchmarks/security_eval.md
git commit -m "docs(security): FA before/after chart + Report 4 honest tradeoff update (WS-2)"
```

---

## Verify cuối (Opus)
- [ ] `... -m pytest -q` ≥ 61 passed, 0 failed.
- [ ] Tự chạy `eval_security.py` → **FA < 5%** (đọc số thật); detection mới ghi nhận trung thực.
- [ ] Mở `security_fa_before_after.png` xác nhận chart đúng số.
- [ ] Đọc §4.3 Report 4: có before/after + chart + đoạn đánh đổi trung thực. Tick plan + Progress log.
```
