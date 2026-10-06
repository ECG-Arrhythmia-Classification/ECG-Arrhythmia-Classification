# Kịch bản demo và trình bày

Tài liệu dành cho phần web/tích hợp của dự án. Chạy tại thư mục gốc repo theo `web/README.md`, mở web local và kiểm tra API trước buổi demo.

## Chuẩn bị

1. Cài dependency bằng `.\web\scripts\setup.ps1`.
2. Tải RAW test bằng `git lfs pull`; chạy `.\.venv\Scripts\python.exe verify_dataset.py`.
3. Khởi động `.\web\scripts\run-local.ps1`, mở `http://127.0.0.1:5173`.
4. Kiểm tra ba model sẵn sàng và runtime trained. Mẫu trong thư viện test phải có nhãn tham chiếu; không dùng mẫu tổng hợp làm minh chứng chất lượng checkpoint.
5. Chuẩn bị một CSV RAW 180 điểm để minh họa upload:

```powershell
.\.venv\Scripts\python.exe web/scripts/export_heartbeat.py --split test --index 0 --output exports/test-beat-0.csv
```

Thử đủ một lần trước buổi trình bày để cài đặt/cold start không ảnh hưởng demo. Có thể tải JSON kết quả để lưu bằng chứng; không chỉnh prediction cho trùng nhãn tham chiếu.

## Demo 4–5 phút

| Bước | Thao tác | Nội dung nói |
|---|---|---|
| 1. Kiến trúc | Mở sơ đồ hệ thống | Dữ liệu chung → tiền xử lý chung → ba kiến trúc → đánh giá → web React/FastAPI |
| 2. Dữ liệu | Chọn heartbeat thật từ test | Một heartbeat 180 điểm, 360 Hz; nhãn N/S/V/F/Q; nhãn tham chiếu được giữ riêng |
| 3. Tiền xử lý | Xem tín hiệu trước/sau | Dùng chính bộ lọc 0,5–40 Hz và Z-score của nhóm; không tạo pipeline riêng cho web |
| 4. Prediction | Chạy CNN mặc định | Checkpoint đã huấn luyện trả logits; softmax tạo phân bố năm lớp; ghi rõ prediction và nhãn tham chiếu |
| 5. So sánh | Chạy cả ba model | Cùng heartbeat/cùng tiền xử lý; có thể bất đồng; thời gian request trên máy demo khác benchmark theo batch |
| 6. Upload | Tải CSV đã chuẩn bị | Người xem tự nhập dữ liệu đúng format và nhận output; không cần chọn mẫu tích hợp sẵn |
| 7. Đánh giá | Mở metric, confusion matrix và robustness | CNN tốt nhất ở kết quả hiện có; đọc Macro F1 và lớp thiểu số cùng Accuracy |
| 8. Lưu kết quả | Xuất JSON | Lưu runtime, model, prediction, probabilities và pipeline để nhóm đối chiếu |

Khi model dự đoán sai một mẫu, dùng confusion matrix để giải thích giới hạn thay vì đổi nhãn dữ liệu. CNN có Accuracy 91,49% nhưng Macro F1 0,5869, đặc biệt yếu ở S/F; độ chính xác toàn tập chịu ảnh hưởng class imbalance.

## Dàn ý ghép slide cuối

1. **Bài toán và dữ liệu** — MIT-BIH, năm nhóm nhãn, heartbeat/sampling rate, split chung; dùng thống kê của thành viên dữ liệu.
2. **Pipeline** — dùng `web/public/architecture.svg`, giải thích luồng offline và inference online.
3. **Ba model** — CNN, BiLSTM, Transformer; dùng cấu hình/checkpoint thực tế, ghép training curve từ các thành viên model khi có.
4. **Kết quả chung** — Accuracy, Macro F1, confusion matrix, parameter/inference benchmark; ghi nguồn `results/` và điều kiện đo.
5. **Robustness** — clean và nhiễu 30/20/10 dB; so sánh mức giảm từ `results/robustness/`.
6. **Web demo** — input RAW → biểu đồ/preprocessing → prediction/probabilities → so sánh → JSON export.
7. **Giới hạn và hướng tiếp theo** — class imbalance/lớp S-F, calibration, bổ sung thời gian train còn thiếu và xác nhận lịch sử tiền xử lý khi train; không đặt số liệu chưa đo.

Tài liệu này là dàn ý và kịch bản; không phải slide `.pptx` hay báo cáo khoa học đã hoàn chỉnh. Ghép các kết quả gốc của nhóm và kiểm tra lại nguồn số liệu trước khi nộp.

## Output cần kiểm tra khi nghiệm thu

- Web/API chạy được, checkpoint thật sẵn sàng.
- Chọn hoặc upload heartbeat RAW và có biểu đồ ECG.
- Hiển thị tiền xử lý thực tế, prediction và xác suất năm lớp.
- So sánh cả ba checkpoint trên cùng dữ liệu; tách nhãn tham chiếu khỏi prediction.
- Xem số liệu đánh giá/confusion matrix/robustness có nguồn rõ ràng.
- Có sơ đồ kiến trúc, JSON xuất được và kịch bản demo.
- Dữ liệu sai format, API thiếu hoặc checkpoint lỗi được báo rõ; không chuyển thành kết quả demo mà người xem không biết.
