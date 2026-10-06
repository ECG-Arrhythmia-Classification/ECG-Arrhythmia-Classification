# ECG Studio

Web React + FastAPI tích hợp dữ liệu, tiền xử lý và ba checkpoint đã huấn luyện của nhóm. Mặc định chạy **CNN thật**; có thể đối chiếu prediction với BiLSTM và Transformer, xem kết quả đánh giá test set, tải báo cáo JSON và sơ đồ hệ thống.

Web nhận **một heartbeat RAW gồm 180 mẫu ở 360 Hz**. API gọi trực tiếp `preprocessing.py`: bandpass Butterworth 0,5–40 Hz bậc 4, sau đó Z-score từng heartbeat. Thứ tự nhãn đầu ra là **N, S, V, F, Q**. Đây là công cụ trình diễn học thuật; xác suất softmax chưa được hiệu chuẩn và không phải độ chính xác của một dự đoán.

## Chạy local trên Windows

Từ thư mục gốc repository, dùng Python 3.12 trở lên và Node.js 20 trở lên:

```powershell
.\web\scripts\setup.ps1
.\web\scripts\run-local.ps1
```

Mở `http://127.0.0.1:5173`; API tại `http://127.0.0.1:8000`, tài liệu API tại `http://127.0.0.1:8000/docs`. Script setup cài PyTorch CPU và dependency vào môi trường ảo của project, tải dữ liệu Git LFS nếu cần và kiểm tra dataset; không huấn luyện lại model. Cần kết nối mạng cho lần cài đầu tiên. Nếu PowerShell chặn script, có thể chạy từng lệnh thủ công bên dưới. Giữ terminal của `run-local.ps1` mở; nhấn **Ctrl+C** để dừng cả hai server.

Để chọn heartbeat từ test set, tải các tệp RAW được quản lý bằng Git LFS:

```powershell
git lfs install
git lfs pull
.\.venv\Scripts\python.exe verify_dataset.py
```

Không cần tải lại MIT-BIH gốc hoặc tạo split mới để demo. Checkpoint và kết quả đánh giá nằm trong repo; các mẫu test cần tệp `processed_data/split/X_test.npy` và `y_test.npy` đã tải đầy đủ, không phải pointer Git LFS.

## Chạy thủ công

Tạo môi trường ảo và cài dependency tại gốc repo:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r web/backend/requirements-trained.txt
npm --prefix web ci
```

Chạy API trong terminal thứ nhất:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app:app --app-dir web/backend --host 127.0.0.1 --port 8000
```

Chạy frontend trong terminal thứ hai:

```powershell
npm --prefix web run dev
```

macOS/Linux dùng `python3 -m venv .venv`, sau đó thay đường dẫn Python bằng `.venv/bin/python`. Các lệnh npm và uvicorn giữ nguyên. Web local kết nối API ở cổng 8000; có thể đổi bằng **Kết nối model** hoặc `VITE_API_URL` trong `web/.env.local` trước khi chạy/build Vite.

## Sử dụng

1. Chọn một heartbeat thật trong test set hoặc tải CSV/JSON chứa một cột/mảng tín hiệu RAW 180 giá trị. Nhãn đi kèm mẫu test là nhãn tham chiếu, không phải prediction.
2. Xem biểu đồ trước/sau tiền xử lý; pipeline của checkpoint được API cố định theo pipeline đánh giá của nhóm.
3. Chạy CNN để xem nhóm nhịp dự đoán và xác suất của cả năm lớp. Chạy so sánh để đối chiếu cả ba model trên cùng đầu vào.
4. Mở phần đánh giá để xem Accuracy, Precision, Recall, Macro F1, confusion matrix, benchmark và kiểm thử nhiễu đã lưu của nhóm.
5. Xuất JSON kết quả, xem sơ đồ kiến trúc và dùng [kịch bản demo](docs/DEMO_GUIDE.md) để trình bày. [Báo cáo tích hợp](docs/INTEGRATION_REPORT.md) ghi phần đã làm và phép kiểm chứng checkpoint/API.

Chọn CNN làm mặc định vì đây là model tốt nhất theo Accuracy và Macro F1 trong kết quả đánh giá hiện có:

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| CNN | 91,49% | 0,5869 |
| BiLSTM | 80,39% | 0,3419 |
| Transformer | 63,47% | 0,4852 |

Nguồn: `results/evaluation/evaluation_results.json`, cùng 18.098 heartbeat test. Điểm toàn tập không bảo đảm một mẫu sẽ được dự đoán đúng. CNN còn yếu ở lớp S và F; khi trình bày cần đọc cả Macro F1 và confusion matrix, không chỉ Accuracy.

## Checkpoint và API

| Model | Kiến trúc trong repo | Checkpoint được nạp trực tiếp |
|---|---|---|
| CNN | `cnn_model.py` | `results/cnn/best_cnn_model.pt` |
| BiLSTM | `rnn_model.py` | `saved_models/best_rnn_model.pth` |
| Transformer | `transformer_model.py` | `results/transformer/best_model.pt` |

Không cần export TorchScript để chạy các checkpoint này. Backend chạy CPU, dùng evaluation mode và inference mode. Model hoặc dependency thiếu/lỗi phải được báo rõ; chế độ trained không tự thay bằng prototype.

API có `GET /health`, `GET /models`, `GET /examples`, `GET /examples/{index}`, `GET /evaluation`, `POST /preprocess`, `POST /predict`. Chi tiết định dạng và cấu hình checkpoint: [MODEL_INTEGRATION.md](docs/MODEL_INTEGRATION.md).

Prototype chỉ dùng khi chủ động đặt `ECG_RUNTIME=demo` hoặc chọn chế độ minh họa trong frontend. Mẫu tổng hợp và điểm prototype được gắn nhãn demo; chúng không thay thế checkpoint, dữ liệu test hoặc kết quả đánh giá.

## Kiểm tra và build

```powershell
npm --prefix web test
npm --prefix web run build
.\.venv\Scripts\python.exe -m unittest discover -s web/backend -p "test_*.py"
.\.venv\Scripts\python.exe -m unittest discover -s web/scripts -p "test_*.py"
```

Đối chiếu adapter API với kiến trúc/checkpoint gốc và chạy lại toàn test set, không ghi đè artifact đánh giá của nhóm:

```powershell
.\.venv\Scripts\python.exe web/scripts/verify_trained_integration.py --full-test --report .local/integration-verification.json
```

Lệnh này cần PyTorch và RAW test đã tải. Báo cáo ghi phiên bản runtime, SHA-256 checkpoint, độ lệch xác suất API/model gốc và kết quả đối chiếu confusion matrix; chạy cùng pipeline để phân biệt lỗi tích hợp với chất lượng checkpoint.

Build tĩnh nằm trong `web/dist/`. GitHub Pages hoặc Sites chỉ phục vụ frontend; muốn người khác chạy model thật qua web online cần triển khai FastAPI riêng bằng HTTPS, cài dependency/checkpoint và cấu hình `ECG_ALLOWED_ORIGINS`. Đặt `VITE_API_URL` thành URL API khi build frontend. Không kết nối API HTTP từ một trang HTTPS.

Chạy setup/demo không commit, push hay deploy. Website online chỉ cập nhật sau một lần build/publish do chủ repo thực hiện.
