# Tích hợp model đã huấn luyện

Backend mặc định dùng `ECG_RUNTIME=trained`: nạp trực tiếp kiến trúc và checkpoint của nhóm từ repository. TorchScript là tùy chọn cho model thay thế, không phải bước bắt buộc cho ba model hiện có.

## Hợp đồng dữ liệu

| Thành phần | Quy ước |
|---|---|
| Đầu vào web/API trained | Một heartbeat RAW, 180 mẫu, 360 Hz |
| Test RAW | `processed_data/split/X_test.npy`, `y_test.npy` |
| Tiền xử lý | `preprocessing.py`: Butterworth 0,5–40 Hz bậc 4, `sosfiltfilt`, Z-score mỗi heartbeat |
| CNN / Transformer | Tensor float32 `[batch, 1, 180]` |
| BiLSTM | Tensor float32 `[batch, 180, 1]` |
| Đầu ra native | Logits `[batch, 5]` |
| Thứ tự lớp | `0=N, 1=S, 2=V, 3=F, 4=Q` |
| Xác suất | Softmax năm logits; hữu hạn, tổng bằng 1 |

Không gửi tín hiệu từ `processed_data/preprocessed/` vào API: pipeline sẽ lọc và chuẩn hóa lần thứ hai. Tệp ECG dài hơn một heartbeat cần được cắt đúng R-peak theo dataset trước khi tải lên; web không tự nhận diện R-peak hay suy ra sampling rate từ CSV. Không resample tùy ý thành 180 điểm để coi là cùng dữ liệu với test set.

Backend import pipeline của nhóm thay vì viết lại bộ lọc. API trả lại tín hiệu đã xử lý cùng metadata trong `preprocessing`, để frontend hiển thị chính dữ liệu dùng cho inference.

## Nạp native checkpoint

| Model id | Model | Trọng số | Cách nạp |
|---|---|---|---|
| `cnn` | `ECGCNN` trong `cnn_model.py` | `results/cnn/best_cnn_model.pt` | Lấy `model_state_dict` trong checkpoint dict |
| `rnn` | `ECG_RNN`, BiLSTM hai lớp, hidden size 32 | `saved_models/best_rnn_model.pth` | Nạp trực tiếp `state_dict` |
| `transformer` | `ECGTransformer` trong `transformer_model.py` | `results/transformer/best_model.pt` | Nạp trực tiếp `state_dict` |

Transformer dùng embedding dimension 32, 4 heads, 2 encoder layers và patch size 4 theo code đánh giá. CNN/BiLSTM cũng dùng cấu hình tương ứng trong `evaluation/evaluate_models.py`. Khi thay checkpoint, phải giữ kiến trúc, shape và thứ tự lớp tương ứng hoặc sửa adapter cùng lúc. Backend đưa model về CPU và gọi `.eval()` trước inference.

Không tự train, tuning hoặc đổi trọng số khi khởi động web. Checkpoint thiếu, sai state_dict hoặc output lỗi sẽ được báo unavailable/HTTP 503. Không có fallback prototype trong runtime trained.

CNN là lựa chọn mặc định theo kết quả đánh giá hiện có: Accuracy 91,49%, Macro F1 0,5869. Đây là căn cứ chọn model cho demo, không phải cam kết chất lượng y tế. Việc tích hợp dùng pipeline giống code evaluation; lịch sử huấn luyện CNN vẫn cần nhóm xác nhận vì `train_cnn.py` có nhánh fallback dữ liệu RAW khi thiếu dữ liệu đã xử lý.

## Biến môi trường

Backend đọc biến môi trường của process, **không tự nạp `.env`**. `web/backend/.env.example` là mẫu cấu hình; xem giá trị thực tế trong file trước khi áp dụng. Mặc định trained không cần khai báo đường dẫn checkpoint.

| Biến | Ý nghĩa |
|---|---|
| `ECG_RUNTIME` | `trained` mặc định; `demo` khi chủ động muốn prototype |
| `ECG_<MODEL>_CHECKPOINT` | Override đường dẫn; đường dẫn tương đối tính từ gốc repo |
| `ECG_<MODEL>_CHECKPOINT_FORMAT` | `native` cho checkpoint của nhóm; `torchscript` cho file đã export |
| `ECG_<MODEL>_PIPELINE` | `team` cho RAW 180 mẫu; chỉ dùng pipeline khác nếu đúng model huấn luyện |
| `ECG_<MODEL>_INPUT_LAYOUT` | `channels_first` cho `[1,1,L]`; `sequence` cho `[1,L,1]` |
| `ECG_<MODEL>_OUTPUT_KIND` | `logits` mặc định; `probabilities` khi model đã softmax |
| `ECG_ALLOWED_ORIGINS` | Origin frontend được phép gọi API; cấu hình khi triển khai online |
| `ECG_TORCH_THREADS` | Số CPU Torch threads từ 1 đến 32; mặc định 2 cho native inference |

`<MODEL>` là `CNN`, `RNN` hoặc `TRANSFORMER`. Chỉ đặt override khi cần thay model; giữ mặc định để dùng checkpoint hiện có.

Ví dụ chạy trained bằng PowerShell tại gốc repo:

```powershell
$env:ECG_RUNTIME='trained'
.\.venv\Scripts\python.exe -m uvicorn app:app --app-dir web/backend --host 127.0.0.1 --port 8000
```

Ví dụ dùng TorchScript CNN do nhóm tự export:

```powershell
$env:ECG_CNN_CHECKPOINT='checkpoints/cnn.ts.pt'
$env:ECG_CNN_CHECKPOINT_FORMAT='torchscript'
$env:ECG_CNN_PIPELINE='team'
$env:ECG_CNN_INPUT_LAYOUT='channels_first'
$env:ECG_CNN_OUTPUT_KIND='logits'
```

File `.ts.pt` không có sẵn trong repo. Khi export, khởi tạo đúng model, nạp trọng số, chuyển CPU/evaluation, rồi dùng `torch.jit.script` hoặc trace phù hợp. So sánh output native và TorchScript trên cùng heartbeat đã tiền xử lý; không dùng random/untrained model làm bản export trình diễn kết quả thật.

## Dữ liệu thật và kết quả đánh giá

`GET /examples` liệt kê mẫu đại diện từ RAW test set; `GET /examples/{index}` trả tín hiệu và nhãn tham chiếu của heartbeat tương ứng. Mẫu được lấy từ dữ liệu thật khi các tệp Git LFS có mặt. API phải báo thiếu dữ liệu nếu tệp không tồn tại hoặc còn là LFS pointer, không thay bằng tín hiệu tổng hợp.

`GET /evaluation` đọc các artifact đã lưu:

- `results/evaluation/evaluation_results.json`: metric và confusion matrix ba model.
- `results/benchmark/benchmark_results.json`: số parameter và thời gian inference.
- `results/robustness/robustness_results.json`: clean, 30 dB, 20 dB, 10 dB.

Bảng tổng hợp của nhóm cũng được lưu ở `results/summary/model_comparison.csv`; endpoint tạo các dòng hiển thị từ ba JSON ở trên.

Các số này là kết quả offline của nhóm, không phải đánh giá được chạy lại mỗi khi tải trang. Timing benchmark được đo theo batch trên môi trường chạy của nhóm; không so trực tiếp với độ trễ một request trên máy người xem. Chưa có thời gian huấn luyện RNN/Transformer đầy đủ; không điền số giả vào báo cáo.

Xuất CSV RAW khi cần demo upload:

```powershell
.\.venv\Scripts\python.exe web/scripts/export_heartbeat.py --split test --index 0 --output exports/test-beat-0.csv
```

Script chỉ xuất cột `signal` và in nhãn tham chiếu. Khi tên file đã tồn tại cần `--force` để ghi đè. CSV mẫu tổng hợp trong `web/public/samples/` dùng cho prototype, không dùng làm test set của model thật.

Thử prediction trực tiếp bằng Python:

```python
import httpx
import numpy as np

raw = np.load("processed_data/split/X_test.npy", mmap_mode="r")[0]
response = httpx.post(
    "http://127.0.0.1:8000/predict",
    json={"model": "cnn", "signal": raw.tolist()},
    timeout=20,
)
response.raise_for_status()
result = response.json()
print(result["prediction"], result["is_demo"], result["preprocessing"])
```

Kiểm tra `is_demo=false` và runtime/pipeline của `/models` trước khi ghi lại kết quả thật. Nhãn tham chiếu và lớp dự đoán có thể khác nhau; hiển thị cả hai khi demo mẫu test. Xác suất cao của một mẫu không thay thế Accuracy/F1 toàn tập và không phải xác suất bệnh lý đã được hiệu chuẩn.

## Triển khai cho người xem online

Frontend tĩnh không chạy PyTorch. Cần host FastAPI riêng, cung cấp code kiến trúc, `preprocessing.py`, checkpoint và các artifact kết quả. Muốn dùng thư viện mẫu test online, cung cấp cả hai tệp RAW test; không cần train/validation cho inference.

Cài `web/backend/requirements-trained.txt`, dùng HTTPS, đặt `ECG_ALLOWED_ORIGINS` thành origin frontend thực tế, rồi cấu hình `VITE_API_URL` khi build React. Origin không chứa đường dẫn repository hoặc dấu `/` cuối. Không đưa token/secret vào biến `VITE_*`.

Kiểm tra `/health`, `/models`, prediction của một mẫu thật và tải JSON sau triển khai. Web đã host trước đây chỉ thay đổi khi chủ repo chủ động build/publish bản mới; chạy local không cập nhật website online.
