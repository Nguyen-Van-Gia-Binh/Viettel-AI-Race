# Walkthrough: Tích hợp Local LLM với cơ chế chuyển đổi linh hoạt

Chúng ta đã hoàn thành tích hợp local LLM (Qwen2.5-3B) qua Ollama vào hệ thống trích xuất thực thể lâm sàng và nhận diện thuộc tính ngữ cảnh. Thiết kế này tuân thủ đúng yêu cầu dễ chuyển đổi mô hình (model switching) và tự động fallback về hệ thống luật khi gặp sự cố.

## Thay đổi đã thực hiện (Changes Made)

1. **Cấu hình độc lập**: Tạo file `configs/model.json` cấu hình các thuộc tính của LLM (nhà cung cấp `ollama`, tên model `qwen2.5:3b`, base URL cổng API OpenAI-compatible, và chế độ chạy pipeline `llm`).
2. **Unified Client (`src/clinlp/llm/client.py`)**: Tận dụng thư viện `httpx` (đã có sẵn trong hệ thống) để tạo kết nối chuẩn OpenAI-compatible. Nhờ vậy, chúng ta không cần thêm thư viện bên thứ 3 và có thể dễ dàng đổi qua vLLM hay server GPU ngoài chỉ qua cấu hình.
3. **LLM Extraction Logic (`src/clinlp/ner/llm_extract.py`)**:
   - Thiết kế prompt y khoa tiếng Việt tối ưu cho việc trích xuất đồng thời 5 loại thực thể lâm sàng và 3 nhãn ngữ cảnh.
   - Thêm thuật toán cân chỉnh chữ hoa/thường (`_align_casing_with_source`) để tự động map các thực thể LLM trả về về dạng chữ chính xác trong văn bản thô trước khi đưa vào `PositionResolver`, giải quyết triệt để lỗi trôi offset vị trí.
4. **Cơ chế định tuyến & Fallback (`src/clinlp/ner/pipeline_adapter.py`)**:
   - Tách logic baseline thành `rules_ner_fn`.
   - Nếu `pipeline.mode` được đặt là `"llm"`, hệ thống sẽ ưu tiên chạy qua LLM. Trong trường hợp xảy ra lỗi sinh từ/phân tích JSON của LLM, hệ thống sẽ tự động bắt ngoại lệ và chạy `rules_ner_fn` dự phòng để đảm bảo pipeline luôn sinh ra kết quả hợp lệ, không bị gián đoạn.

## Kiểm thử & Xác thực (Verification Results)

1. **Unit Test Suite mới (`tests/test_llm_extract.py`)**:
   - Đã tạo các ca kiểm thử giả lập (mocking) cuộc gọi HTTP để kiểm tra tính chính xác của:
     - Khả năng tự sửa lỗi hoa/thường của thực thể.
     - Khả năng phân tích định dạng JSON trả về từ LLM sang đúng Schema Pydantic.
     - Khả năng định tuyến giữa LLM mode và Rules mode.
2. **Kết quả**: Chạy thành công toàn bộ **62 tests** (`pytest`):
   ```bash
   ============================= 62 passed in 4.18s ==============================
   ```

## Hướng dẫn sử dụng & Chuyển đổi mô hình

* **Để chạy LLM cục bộ (Option A)**:
  1. Khởi động Ollama trên Windows và kéo mô hình:
     ```powershell
     ollama run qwen2.5:3b
     ```
  2. Cấu hình tệp `configs/model.json` đặt `"mode": "llm"`:
     ```json
     {
       "llm": {
         "provider": "ollama",
         "model_name": "qwen2.5:3b",
         "base_url": "http://localhost:11434/v1",
         "api_key": "ollama",
         "temperature": 0.0,
         "max_tokens": 2048
       },
       "pipeline": {
         "mode": "llm"
       }
     }
     ```
* **Để quay lại Baseline bằng luật (Rules)**:
  * Đổi cấu hình `"mode"` trong `"pipeline"` thành `"rules"` (hoặc xóa file `model.json`).
* **Để đổi mô hình sang 7B (Khi có server mạnh hơn)**:
  * Chạy mô hình 7B trên Ollama hoặc server khác, sau đó thay đổi `"model_name"` và `"base_url"` tương ứng.
