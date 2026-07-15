# Tích hợp Local LLM (Option A - Ollama) và Hỗ trợ chuyển đổi mô hình linh hoạt

## Goal
Tích hợp mô hình ngôn ngữ lớn chạy cục bộ (Qwen2.5-3B-Instruct qua Ollama) vào pipeline NER và suy luận ngữ cảnh, đảm bảo kiến trúc cấu hình tách biệt để dễ dàng thay đổi mô hình/nhà cung cấp sau này.

## Tasks
- [ ] **Task 1: Tạo file cấu hình `configs/model.yaml`** → Verify: File tồn tại chứa các cấu hình về `provider`, `model_name`, `base_url`, `temperature`.
- [ ] **Task 2: Xây dựng client dùng chung `src/clinlp/llm/client.py`** → Verify: Hàm khởi tạo client OpenAI-compatible hoạt động tốt với Ollama.
- [ ] **Task 3: Triển khai prompt & trích xuất `src/clinlp/ner/llm_extract.py`** → Verify: Hàm `llm_ner_fn` nhận diện thực thể, nhãn loại, và các assertions qua JSON output của LLM.
- [ ] **Task 4: Cập nhật `src/clinlp/ner/pipeline_adapter.py`** → Verify: Hỗ trợ chuyển đổi giữa chế độ `rules` (baseline) và `llm` (hoặc `hybrid`).
- [ ] **Task 5: Kiểm thử tích hợp hệ thống** → Verify: Chạy thử trên một tài liệu mẫu (ví dụ `1.txt`) ở chế độ LLM, xuất kết quả JSON hợp lệ mà không có lỗi vị trí.

## Done When
- [ ] Pipeline chạy thành công ở chế độ LLM với mô hình `qwen2.5:3b` trên Windows qua Ollama.
- [ ] Việc đổi sang mô hình khác (ví dụ: `gemma2:2b` hoặc API vLLM từ xa) chỉ cần thay đổi nội dung trong tệp cấu hình `configs/model.yaml` mà không cần chỉnh sửa mã nguồn Python.
- [ ] Vượt qua toàn bộ unit tests hiện có (`pytest tests/`).

## Notes
- Do giới hạn VRAM 4GB, việc sử dụng các thư viện như `outlines` hay `vllm` trực tiếp trên máy Windows này không khả thi (thiếu hỗ trợ WSL2/GPU đầy đủ hoặc quá nặng). Vì vậy, ta sẽ sử dụng tính năng **Structured JSON Output** gốc của Ollama thông qua OpenAI-compatible API Client.
- Luôn giữ nguyên tắc: LLM chỉ trả về `text`, `type`, và `assertions`. Việc tính toán `position` (chỉ mục ký tự) vẫn do code Python (`PositionResolver`) đảm nhận để đảm bảo độ chính xác tuyệt đối, tránh trôi lệch offset.
