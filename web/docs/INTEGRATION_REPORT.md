# Báo cáo tích hợp ECG Studio

Ngày: **06/10/2026**. Tích hợp dựa trên code và artifact của nhóm ở `main`, commit nền `562d3f6`. Không huấn luyện lại hoặc sửa checkpoint để cải thiện số liệu; mục tiêu là đưa model thật và pipeline đã đánh giá vào web, giữ output có thể đối chiếu được.

## Phần đã tích hợp

| Yêu cầu output web | Kết quả triển khai |
|---|---|
| Upload/chọn ECG | CSV/JSON RAW; thư viện heartbeat thật từ cùng test set, có nhãn tham chiếu và index |
| Biểu đồ ECG | Hiển thị tín hiệu đầu vào và tín hiệu API thực sự dùng sau tiền xử lý |
| Preprocessing | Gọi trực tiếp `preprocessing.py`: 180 mẫu/360 Hz, bandpass 0,5–40 Hz bậc 4, Z-score |
| Prediction | CNN mặc định, nạp `best_cnn_model.pt` cùng kiến trúc gốc; không dùng prototype trong trained |
| Loại nhịp và confidence | Nhãn N/S/V/F/Q; phân bố softmax năm lớp; nhãn tham chiếu được hiển thị riêng |
| So sánh model | CNN, BiLSTM và Transformer nhận cùng heartbeat, cùng pipeline team |
| Sơ đồ kiến trúc | `web/public/architecture.svg`: luồng offline, checkpoint, runtime API và kết quả đã lưu |
| Demo và trình bày | `DEMO_GUIDE.md`: kịch bản demo 4–5 phút và dàn ý ghép slide cuối |
| Bổ sung cho output cơ bản | Metric toàn tập, confusion matrix, benchmark/robustness, xuất JSON prediction và evaluation, script kiểm tra đối chiếu |

Backend chạy CPU, nạp native `state_dict` hoặc `model_state_dict`, kiểm tra đúng shape và output, dùng evaluation/inference mode. Checkpoint, dữ liệu hoặc API thiếu/lỗi được báo rõ. Runtime prototype là lựa chọn minh họa chủ động, không phải fallback khi model thật hỏng.

## Kiểm chứng checkpoint và API

Đã chạy trong môi trường ảo của project: **Python 3.12.2**, **PyTorch 2.14.1+cpu**, **NumPy 2.5.3**, CPU với hai Torch threads.

- **27/27 kiểm thử API** đạt.
- **8/8 kiểm thử exporter** đạt.
- **8/8 kiểm thử frontend** đạt; production build Vite thành công.
- Dùng **35 heartbeat thật cho mỗi model**, gồm mẫu từ đủ năm lớp và mẫu ngẫu nhiên với seed cố định, để đối chiếu API với model gốc.
- Tín hiệu sau tiền xử lý, nhãn và xác suất API khớp phép chạy checkpoint gốc trong sai số số học cho phép.
- Chạy lại toàn bộ **18.098 heartbeat test cho cả ba checkpoint**; confusion matrix mỗi model **khớp chính xác** artifact đánh giá đã lưu.

Đã khởi động bằng `run-local.ps1` và thao tác trên web: chạy CNN, so sánh ba model, chọn mẫu N/Q, upload CSV RAW, xem biểu đồ sau preprocessing, xem trang thực nghiệm và kiểm tra nội dung JSON xuất. Với heartbeat #0, nhãn thực tế Q nhưng BiLSTM dự đoán N; giao diện hiển thị “khác nhãn”, không sửa nhãn hay kết quả để tạo cảm giác luôn đúng. Kiểm tra ở bố cục desktop và điện thoại không thấy tràn ngang toàn trang; bảng rộng cuộn trong vùng riêng. Không ghi nhận lỗi JavaScript trong các luồng đã kiểm tra.

Trình duyệt trong ứng dụng chưa cung cấp được sự kiện tải Blob trong lần kiểm thử. Vì vậy JSON có thêm hộp xem trước chứa đúng nội dung xuất, văn bản có thể chọn và nút sao chép chủ động; nút tải tệp vẫn được giữ cho trình duyệt hỗ trợ. Nội dung JSON đã đối chiếu có 180 mẫu RAW, nhãn tham chiếu, prediction, phân bố xác suất, pipeline và SHA-256 checkpoint thật. Không coi lần click download trong trình duyệt này là bằng chứng tệp đã tải thành công.

| Model | Accuracy chạy lại | Macro F1 chạy lại | Sai lệch xác suất API/gốc lớn nhất |
|---|---:|---:|---:|
| CNN | 91,4852% | 0,586909 | 3,0112 × 10⁻⁷ |
| BiLSTM | 80,3901% | 0,341870 | 7,2250 × 10⁻⁸ |
| Transformer | 63,4656% | 0,485228 | 2,6003 × 10⁻⁷ |

Nguồn kiểm chứng: `web/scripts/verify_trained_integration.py`; kết quả máy hiện tại lưu ở `.local/integration-verification.json`. Script tạo báo cáo riêng, không ghi đè kết quả của thành viên đánh giá.

Chạy lại từ gốc repo sau khi setup và tải Git LFS:

```powershell
$env:ECG_RUNTIME='trained'
.\.venv\Scripts\python.exe web/scripts/verify_trained_integration.py --full-test --report .local/integration-verification.json
```

SHA-256 checkpoint đã kiểm chứng:

| Checkpoint | SHA-256 |
|---|---|
| CNN | `3e12bd9f71e852beb7ae6272f37831acf949af609c299220c66fc7012bc1220c` |
| BiLSTM | `999c9d5fef53f2893ab8f160a244c54dc286b147ec2a4bc5fa760d4e1c2a1e5f` |
| Transformer | `f67e8d789d230801634297ef681077f7c698ebddb4927fe34cc4e1d543ca95b1` |

## Cách đọc output

CNN được chọn mặc định vì Accuracy và Macro F1 tốt nhất ở kết quả hiện có. Tích hợp đúng không làm tăng chất lượng checkpoint: CNN vẫn yếu ở S/F, với F1 lần lượt khoảng 0,2246 và 0,0102. Không kết luận chỉ từ Accuracy 91,49% hoặc một heartbeat dự đoán đúng.

Softmax chưa được hiệu chuẩn thành xác suất chẩn đoán. Benchmark hiển thị là phép đo theo batch đã lưu của nhóm, khác độ trễ từng request trên máy demo. Robustness hiển thị là kết quả thí nghiệm đã lưu; tích hợp không tự chạy lại robustness khi người xem mở trang.

Nhóm vẫn cần bổ sung thời gian huấn luyện RNN/Transformer và phần phân tích ưu/nhược điểm vào báo cáo chung. Web không điền số liệu giả cho các đầu ra còn thiếu này.

## Chạy và chia sẻ

```powershell
.\web\scripts\setup.ps1
.\web\scripts\run-local.ps1
```

Web ở `http://127.0.0.1:5173`, API ở `http://127.0.0.1:8000`. Hướng dẫn chi tiết: `web/README.md` và `MODEL_INTEGRATION.md`.

Các thay đổi được giữ local để nhóm xem. Website đã host không tự cập nhật; muốn chia sẻ model thật online cần deploy FastAPI có HTTPS và cấu hình URL/CORS cho frontend. Chạy hoặc kiểm chứng project không commit, push hay publish.
