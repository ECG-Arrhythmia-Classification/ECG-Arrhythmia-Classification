# Tích hợp checkpoint của nhóm

Repo hiện có pipeline dữ liệu và web demo; chưa có checkpoint huấn luyện sẵn. Các đường dẫn `checkpoints/*.ts.pt` dưới đây là ví dụ cho đầu ra nhóm sẽ tạo sau này.

## Thống nhất dữ liệu trước khi huấn luyện

| Thành phần | Hợp đồng |
|---|---|
| Heartbeat | Một đoạn 180 mẫu ở 360 Hz |
| Dữ liệu RAW | `processed_data/split/X_<train\|val\|test>.npy` |
| Nhãn | `0=N, 1=S, 2=V, 3=F, 4=Q` |
| Tiền xử lý team | Butterworth bandpass 0,5–40 Hz bậc 4, `sosfiltfilt`, Z-score mỗi heartbeat |
| Dữ liệu sau xử lý | `processed_data/preprocessed/`, tạo bởi `python preprocessing.py` |
| CNN | Tensor float32 `[batch, 1, 180]` |
| RNN / Transformer | Tensor float32 `[batch, 180, 1]` |
| Đầu ra | Tensor `[batch, 5]`; inference một mẫu chấp nhận `[1,5]` hoặc `[5]` |

**`data_loader.py` hiện đọc `processed_data/split/` (RAW), chưa tự đọc thư mục `preprocessed`.** Khi huấn luyện cho adapter `team`, dùng dữ liệu từ `processed_data/preprocessed/` hoặc gọi cùng `preprocess_data` trong loader của model. Giữ split theo record hiện có; chỉ áp dụng augmentation trên train. Validation và test dùng cùng pipeline, không augmentation.

Thứ tự 5 giá trị đầu ra bắt buộc là **N, S, V, F, Q**. Không đảo thứ tự lớp giữa checkpoint và API. Nếu model sử dụng shape hoặc preprocessing khác, phải khai báo/cập nhật adapter tương ứng trước khi sử dụng.

## Export TorchScript

Cài PyTorch vào môi trường chạy API khi cần checkpoint:

```bash
python -m pip install torch
```

Từ code huấn luyện của nhóm, đưa model đã nạp trọng số lên CPU và chuyển sang evaluation. Ví dụ cho model CNN có đầu vào `[batch,1,180]`:

```python
from pathlib import Path
import torch

# model phải là model của nhóm đã được nạp trọng số đã huấn luyện.
model = model.cpu().eval()
scripted = torch.jit.script(model)
with torch.inference_mode():
    output = scripted(torch.zeros(1, 1, 180, dtype=torch.float32))
assert output.shape == (1, 5)
assert torch.isfinite(output).all()
Path("checkpoints").mkdir(exist_ok=True)
scripted.save("checkpoints/cnn.ts.pt")
```

Ví dụ này không huấn luyện hay tạo model thật. Với RNN/Transformer, probe bằng `torch.zeros(1,180,1)`. Nếu model trả về tuple/dict (ví dụ logits và hidden state), bọc lại để TorchScript trả về **một Tensor** chứa logits/probabilities. Kiểm tra đầu ra TorchScript khớp model gốc trên cùng heartbeat đã tiền xử lý.

## Cấu hình backend

Mỗi model có các biến môi trường:

| Biến | Giá trị |
|---|---|
| `ECG_<MODEL>_CHECKPOINT` | File TorchScript; đường dẫn tương đối tính từ gốc repo |
| `ECG_<MODEL>_PIPELINE` | `team` mặc định khi có checkpoint; `demo` chỉ khi checkpoint được huấn luyện theo đúng pipeline demo 256 mẫu |
| `ECG_<MODEL>_INPUT_LAYOUT` | `channels_first` cho `[1,1,L]`; `sequence` cho `[1,L,1]` |
| `ECG_<MODEL>_OUTPUT_KIND` | `logits` mặc định; hoặc `probabilities` nếu model đã softmax |

`<MODEL>` là `CNN`, `RNN` hoặc `TRANSFORMER`. Bỏ biến `CHECKPOINT` để model tương ứng tiếp tục chạy prototype demo. Checkpoint được khai báo nhưng thiếu/lỗi sẽ báo unavailable / HTTP 503; API không âm thầm thay bằng demo. Probability phải hữu hạn, nằm trong `[0,1]` và tổng bằng 1.

PowerShell, từ gốc repo và sau khi kích hoạt môi trường ảo:

```powershell
$env:ECG_CNN_CHECKPOINT='checkpoints/cnn.ts.pt'
$env:ECG_CNN_PIPELINE='team'
$env:ECG_CNN_INPUT_LAYOUT='channels_first'
$env:ECG_CNN_OUTPUT_KIND='logits'
python -m uvicorn app:app --app-dir web/backend --host 127.0.0.1 --port 8000
```

macOS / Linux:

```bash
export ECG_CNN_CHECKPOINT=checkpoints/cnn.ts.pt
export ECG_CNN_PIPELINE=team
export ECG_CNN_INPUT_LAYOUT=channels_first
export ECG_CNN_OUTPUT_KIND=logits
python -m uvicorn app:app --app-dir web/backend --host 127.0.0.1 --port 8000
```

`web/backend/.env.example` liệt kê cấu hình cho cả 3 model. Backend **không tự nạp dotenv**. Nếu sao chép thành `web/backend/.env`, sửa giá trị rồi nạp file vào process trước khi chạy; ví dụ Bash:

```bash
set -a
source web/backend/.env
set +a
```

Không nạp nguyên file ví dụ khi chưa có checkpoint: các đường dẫn ví dụ chưa tồn tại sẽ khiến model unavailable.

## Thử từ dữ liệu của repo

```bash
python web/scripts/export_heartbeat.py --split test --index 0 --output exports/test-beat-0.csv
```

Dataset `.npy` không có trong bản clone Git; trước khi chạy exporter cần nhận RAW split chung của nhóm hoặc chuẩn bị MIT-BIH và chạy pipeline trong README gốc. Tệp CSV xuất ra có 180 giá trị **RAW** ở cột `signal`. Tải tệp này vào web local, bật chế độ FastAPI, kiểm tra kết nối rồi chạy model tương ứng. Nhãn in bởi exporter là nhãn tham chiếu để kiểm tra, không phải kết quả model.

Với checkpoint `team`, API thực hiện tiền xử lý của `preprocessing.py` và không resample; các toggle preprocessing demo trên frontend không thay đổi pipeline này. `POST /predict` nhận RAW và trả lại tín hiệu đã xử lý trong `preprocessing.signal`, cùng metadata của pipeline. Không gửi `processed_data/preprocessed/` vào API vì sẽ lọc/Z-score lần thứ hai. UI có thể tạo mẫu tổng hợp 180 điểm khi chọn model pipeline `team` để kiểm tra kết nối và chạy thử checkpoint. Mẫu này vẫn chỉ là minh họa; cần CSV RAW thật của repo để đối chiếu model và không suy ra chất lượng model từ tín hiệu tổng hợp.

Ví dụ request khi đã có API local:

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

`httpx` đã có trong dependency của API; có thể dùng Swagger `/docs` thay thế.

## Đưa checkpoint lên web online

Hiện workflow chỉ kiểm tra và build, chưa cấu hình publish. Khi chủ repo quyết định triển khai, có thể dùng GitHub Pages để phục vụ frontend tĩnh. Pages không chạy backend: cần triển khai FastAPI ở dịch vụ/server riêng có HTTPS, cài dependency và PyTorch, cung cấp checkpoint và biến môi trường. Đặt `ECG_ALLOWED_ORIGINS` thành origin thực tế của frontend, ví dụ `https://your-team.github.io` (không thêm đường dẫn repository hoặc dấu `/` cuối).

Trong frontend, nhập URL HTTPS qua **Kết nối model**. Hoặc đặt `VITE_API_URL` lúc build, ví dụ `VITE_API_URL=https://api.example.com`; đây là địa chỉ công khai, không chứa token. Khi URL thay đổi cần build lại để cập nhật giá trị mặc định. Không kết nối API HTTP từ trang web HTTPS vì trình duyệt có thể chặn mixed content; chạy web local để thử API local.

Trước khi chia sẻ kết quả thật, kiểm tra `/models` báo `torchscript`, `is_demo=false`, shape/pipeline đúng; kiểm tra JSON kết quả cũng ghi `is_demo=false`. Đánh giá accuracy/F1/confusion matrix trên toàn test set bằng code evaluation của nhóm. Một lần dự đoán trên web không phải báo cáo chất lượng model.
