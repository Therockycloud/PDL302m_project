# Quy tắc làm việc — Dự án DPL302m

## Chính sách dùng model (BẮT BUỘC)

- **Thực thi (implementation) → CHỈ spawn subagent Sonnet.**
  - Khi gọi tool `Agent`, luôn đặt `model: "sonnet"`.
  - Mức reasoning/effort **vừa đủ** để hoàn thành task — không hơn. Prompt gọn, đủ context để chạy một mạch.
  - **TUYỆT ĐỐI KHÔNG spawn subagent Opus** (tốn token, không cần thiết).

- **Opus chỉ dùng cho PLAN và KIỂM THỬ.**
  - Plan: phân tích yêu cầu, thiết kế hướng đi, chia task, viết spec/plan.
  - Kiểm thử/thẩm định: review & xác minh kết quả của subagent (đọc diff, chạy test, render/đọc file, đối chiếu yêu cầu).
  - **Không dùng Opus để tự tay thực thi** (viết code, sửa file hàng loạt, build, xuất file…). Việc đó giao Sonnet.

## Luồng chuẩn

1. **Opus** plan → 2. **Sonnet** thực thi (commit sau mỗi bước) → 3. **Opus** verify.

## Ghi chú thực thi

- Commit sau mỗi bước hoàn thành, dùng **explicit path** trong `git add` (không `git add -A` ở gốc repo).
- **Không tự push** trừ khi được yêu cầu.
- Subagent prompt phải self-contained: nêu rõ **file cần sửa, mục tiêu, ràng buộc, cách verify**.
- Nếu task nhỏ/tầm thường mà spawn agent còn tốn hơn tự làm thì Opus làm trực tiếp (ngoại lệ hiếm, vẫn ưu tiên Sonnet cho execution).
