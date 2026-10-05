# ECG Studio

Web của dự án để tải một heartbeat, xem tín hiệu, chạy phân loại N/S/V/F/Q và so sánh CNN, RNN/LSTM/GRU, Transformer. Giao diện React chạy độc lập; FastAPI là phần tùy chọn để tích hợp checkpoint của nhóm.

**Trạng thái hiện tại:** repo đã có trọng số CNN (`results/cnn/best_cnn_model.pt`) và RNN (`saved_models/best_rnn_model.pth`). Web vẫn chạy prototype demo đến khi nhóm export và cấu hình TorchScript; các trọng số trên không được API tự nạp. Chế độ demo dùng tín hiệu tổng hợp, xác suất không phải độ chính xác trên MIT-BIH. Đây là công cụ trình diễn học thuật, không dùng để chẩn đoán.

## Chạy web

Thực hiện các lệnh từ thư mục gốc repository. Khuyến nghị Node.js 24 để khớp CI; web hỗ trợ Node.js 20 trở lên:

```bash
npm --prefix web ci
npm --prefix web run dev
```

Mở `http://127.0.0.1:5173`. Chế độ demo chạy ngay, không cần Python hoặc dữ liệu MIT-BIH. Có các tệp minh họa trong `web/public/samples/`.

## Chạy API tùy chọn

Khuyến nghị Python 3.12 để khớp CI; backend hỗ trợ Python 3.10 trở lên. Từ gốc repo, tạo và kích hoạt môi trường ảo:

```bash
python -m venv .venv
```

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Cài dependency và chạy API:

```bash
python -m pip install -r web/backend/requirements.txt
python -m uvicorn app:app --app-dir web/backend --host 127.0.0.1 --port 8000
```

Trong web local, chọn **Kết nối model → FastAPI / checkpoint**, nhập `http://127.0.0.1:8000`, kiểm tra kết nối rồi áp dụng. Có thể đặt URL mặc định bằng `VITE_API_URL` trong `web/.env.local`; Vite đọc biến này lúc chạy/build. API không tự đọc `web/backend/.env.example`.

API có `GET /health`, `GET /models`, `POST /preprocess`, `POST /predict`; tài liệu tương tác tại `http://127.0.0.1:8000/docs`.

## Dùng dữ liệu và model của nhóm

- Nhóm dùng heartbeat **180 mẫu, 360 Hz**, nhãn `0=N, 1=S, 2=V, 3=F, 4=Q`.
- RAW split ở `processed_data/split/` được lưu bằng Git LFS. Cài Git LFS, chạy `git lfs pull` rồi `python verify_dataset.py` từ gốc repo trước khi export; xem hướng dẫn trong README gốc. Không tự tạo split khác cho từng model.
- Khi đã có RAW split, xuất một heartbeat để tải vào web:

```bash
python web/scripts/export_heartbeat.py --split test --index 0 --output exports/test-beat-0.csv
```

Script chỉ xuất cột `signal` và in nhãn tham chiếu; không chỉnh dữ liệu nguồn. Chạy lại với cùng tên cần `--force`. Tệp CSV không phải kết quả dự đoán và không tự được thêm vào Git.

- Adapter checkpoint mặc định dùng pipeline `team`: bandpass 0,5–40 Hz bậc 4 rồi Z-score từ `preprocessing.py`, giữ đúng 180 mẫu. Chỉ cấu hình pipeline này khi khớp tiền xử lý lúc huấn luyện; `train_cnn.py` có thể dùng RAW split khi thiếu dữ liệu preprocessed nên cần đối chiếu cấu hình và log của lần train trước khi tích hợp trọng số CNN. Khi dùng `team`, tải RAW vào API để tránh xử lý hai lần.
- Demo dùng pipeline riêng, resample 256 mẫu. Kết quả demo không dùng để đánh giá model đã huấn luyện.
- Hướng dẫn export TorchScript, cấu hình shape, lớp đầu ra và chạy từ web online: [MODEL_INTEGRATION.md](docs/MODEL_INTEGRATION.md).

## Kiểm tra và build

```bash
npm --prefix web test
npm --prefix web run build
python -m unittest discover -s web/backend -p "test_*.py"
python -m unittest discover -s web/scripts -p "test_*.py"
```

Build tĩnh được ghi vào `web/dist/`. Workflow hiện chỉ kiểm tra và build; chưa cấu hình publish. Chủ repo có thể chủ động cấu hình GitHub Pages hoặc dịch vụ hosting để phục vụ build này sau. Pages không chạy Python hay PyTorch. Muốn web online dùng checkpoint thật, cần triển khai FastAPI riêng bằng HTTPS và cấu hình CORS cho đúng origin của web. Người xem vẫn dùng được demo khi chưa có API.
