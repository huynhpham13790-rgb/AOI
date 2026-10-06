# 1. Tổng quan về Vision AI và hệ thống AOI

## 1.1. Kiểm tra ngoại quan trong sản xuất

Kiểm tra ngoại quan (visual inspection) là khâu xác định sản phẩm có lỗi nhìn thấy được trên bề mặt hay không: vết xước, nứt, mẻ, bẩn, thiếu chi tiết, sai màu, biến dạng... Cách làm truyền thống là công nhân quan sát bằng mắt. Cách này có các nhược điểm:

- **Không ổn định**: phụ thuộc kinh nghiệm, độ mỏi mắt và sự tập trung của từng người.
- **Chậm**: khó theo kịp dây chuyền tốc độ cao.
- **Khó truy vết**: không lưu lại hình ảnh và lý do loại bỏ sản phẩm.
- **Chi phí nhân công** lớn khi sản lượng tăng.

## 1.2. AOI (Automated Optical Inspection)

AOI là hệ thống kiểm tra quang học tự động: dùng camera chụp sản phẩm rồi phần mềm phân tích ảnh để đưa ra kết luận **OK** (đạt) hoặc **NG** (No Good, lỗi). AOI được dùng phổ biến trong:

- Điện tử: kiểm tra mạch in (PCB), mối hàn, linh kiện dán (SMT).
- Cơ khí: bề mặt chi tiết gia công, ốc vít, đai ốc.
- Dược phẩm, thực phẩm: viên thuốc, viên nang, bao bì, nông sản.
- Vật liệu: thép cán, vải, gỗ, gạch.

Một hệ thống AOI điển hình gồm:

| Thành phần | Vai trò |
|---|---|
| Camera công nghiệp + ống kính | Thu ảnh sản phẩm với độ phân giải và tốc độ phù hợp |
| Hệ chiếu sáng | Làm nổi bật lỗi (đèn vòng, đèn nền, đèn góc thấp...) |
| Cảm biến / trigger | Báo có sản phẩm đi qua để chụp đúng thời điểm |
| Máy tính xử lý | Chạy thuật toán xử lý ảnh / AI |
| Cơ cấu chấp hành | Đèn báo, còi, xi lanh gạt sản phẩm NG, gửi tín hiệu tới PLC |
| Phần mềm giám sát | Hiển thị kết quả, thống kê, lưu nhật ký |

## 1.3. Hai cách tiếp cận xử lý ảnh

### a) Xử lý ảnh truyền thống (rule-based)

Dùng các phép toán như lọc nhiễu, phân ngưỡng, phát hiện biên, so khớp mẫu (template matching), đo kích thước... Kỹ sư phải tự đặt luật cho từng loại lỗi.

- Ưu điểm: nhanh, dễ giải thích, không cần nhiều dữ liệu.
- Nhược điểm: nhạy với thay đổi ánh sáng và vị trí; khó áp dụng với lỗi có hình dạng tự nhiên, ngẫu nhiên (vết bẩn, vết nứt); mỗi sản phẩm phải viết lại luật.

### b) Vision AI (học sâu)

Mạng nơ-ron tích chập (CNN) tự học đặc trưng từ dữ liệu ảnh. Có ba dạng bài toán chính:

| Bài toán | Đầu ra | Dữ liệu cần | Mô hình tiêu biểu |
|---|---|---|---|
| Phân loại (classification) | OK/NG hoặc tên loại lỗi | Ảnh gán nhãn theo lớp | ResNet, EfficientNet |
| Phát hiện đối tượng (detection) | Hộp bao quanh lỗi + loại lỗi | Ảnh gán hộp bao | YOLO, Faster R-CNN |
| Phát hiện bất thường (anomaly detection) | Điểm bất thường + bản đồ vùng lỗi | **Chỉ cần ảnh OK** | PatchCore, PaDiM, EfficientAD |

**Phát hiện bất thường** đặc biệt phù hợp với công nghiệp vì:
- Sản phẩm lỗi rất hiếm (tỷ lệ lỗi thường < 1%), khó thu đủ ảnh lỗi để huấn luyện.
- Lỗi mới, chưa từng gặp vẫn được phát hiện vì mô hình chỉ học "thế nào là bình thường".
- Không cần gán nhãn thủ công: chỉ cần gom ảnh sản phẩm đạt.

## 1.4. Hướng tiếp cận của đề tài

Đề tài kết hợp hai mô hình:

1. **PatchCore** (phát hiện bất thường): học từ ảnh OK, quyết định **OK/NG** và chỉ ra **vùng lỗi** trên ảnh.
2. **ResNet18** (phân loại): khi ảnh là NG, xác định **loại lỗi** (nứt, xước, bẩn, biến dạng...).

Kết hợp thêm các bước xử lý ảnh truyền thống (resize, cắt vùng quan tâm, chuẩn hóa, lọc Gauss bản đồ nhiệt, tìm contour để khoanh vùng lỗi).

Do không có camera công nghiệp và phần cứng thật, hệ thống được demo trên laptop: ảnh đầu vào lấy bằng **tải ảnh lên** (thay cho camera) hoặc webcam laptop, kết quả OK/NG hiển thị trên giao diện web và ghi nhật ký.

## 1.5. Các chỉ tiêu đánh giá

Với ma trận nhầm lẫn (NG là lớp dương):

| | Dự đoán NG | Dự đoán OK |
|---|---|---|
| **Thực tế NG** | TP (bắt đúng lỗi) | FN (bỏ sót lỗi) |
| **Thực tế OK** | FP (báo nhầm) | TN (đúng là OK) |

- **Độ chính xác (Accuracy)** = (TP + TN) / tổng.
- **Tỷ lệ phát hiện lỗi (Recall)** = TP / (TP + FN). Quan trọng nhất: lỗi lọt ra thị trường gây thiệt hại lớn.
- **Tỷ lệ bỏ sót** = FN / (TP + FN) = 1 − Recall.
- **Tỷ lệ báo nhầm (False alarm rate)** = FP / (FP + TN). Báo nhầm nhiều làm lãng phí sản phẩm tốt và mất thời gian kiểm tra lại.
- **AUROC**: diện tích dưới đường ROC, đánh giá khả năng tách OK/NG không phụ thuộc ngưỡng (1.0 là hoàn hảo).
- **AUROC pixel**: đánh giá khả năng khoanh đúng vùng lỗi so với mask chuẩn.
- **Thời gian xử lý**: thời gian từ lúc nhận ảnh tới khi có kết quả, quyết định tốc độ dây chuyền tối đa.
