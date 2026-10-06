# 3. Mô hình AI

Hệ thống dùng hai mô hình nối tiếp:

```
            ┌──────────────────────────┐   điểm < ngưỡng
 ảnh  ───▶  │ PatchCore (phát hiện)     │ ─────────────────▶  OK
            │ WideResNet50 + memory bank│
            └────────────┬─────────────┘
                         │ điểm ≥ ngưỡng
                         ▼
            ┌──────────────────────────┐
            │ ResNet18 (phân loại lỗi)  │ ─────────────────▶  NG + loại lỗi
            └──────────────────────────┘                       + bản đồ vùng lỗi
```

## 3.1. PatchCore: phát hiện lỗi (OK/NG) và khoanh vùng lỗi

Tham khảo: Roth et al., *Towards Total Recall in Industrial Anomaly Detection*, CVPR 2022. Mã nguồn: [`aoi/patchcore.py`](../aoi/patchcore.py).

### Nguyên lý

1. **Trích đặc trưng**: đưa ảnh 224×224 qua WideResNet50 (đã huấn luyện trên ImageNet, giữ nguyên trọng số). Lấy đầu ra của `layer2` (512 kênh, 28×28) và `layer3` (1024 kênh, 14×14, phóng lên 28×28). Ghép lại được **784 vector 1536 chiều**, mỗi vector mô tả một ô 8×8 pixel. Trước khi ghép, lấy trung bình vùng 3×3 để mỗi ô mang thông tin ngữ cảnh xung quanh.
   - Không dùng tầng sâu hơn (`layer4`) vì đặc trưng quá trừu tượng, thiên về phân loại vật thể ImageNet, kém nhạy với lỗi nhỏ.
2. **Memory bank**: gom đặc trưng của tất cả ô trong mọi ảnh OK (ví dụ 209 ảnh × 784 ô ≈ 164.000 vector).
3. **Coreset**: giữ lại 1% số vector bằng thuật toán *k-center greedy*: lần lượt chọn vector xa tập đã chọn nhất, để tập con phủ đều không gian đặc trưng. Giúp giảm bộ nhớ và tăng tốc 100 lần mà gần như không giảm độ chính xác. Để tính nhanh, khoảng cách được tính trên phép chiếu ngẫu nhiên 128 chiều.
4. **Suy luận**: với mỗi ô của ảnh cần kiểm tra, tính khoảng cách Euclid tới vector gần nhất trong memory bank. Ô càng xa thì càng "lạ" so với mọi sản phẩm OK đã thấy.
   - **Bản đồ bất thường**: lưới 28×28 khoảng cách, phóng lên 224×224 và lọc Gauss (σ = 4) cho mượt.
   - **Điểm ảnh** = giá trị lớn nhất của bản đồ.

### Chọn ngưỡng

Ảnh là **NG** nếu điểm ≥ ngưỡng. Ngưỡng được chọn trên nửa **dev** sao cho **F1** lớn nhất (cân bằng giữa phát hiện lỗi và báo nhầm). Ngưỡng lưu trong `models/<sản phẩm>/patchcore.pt`.

Trong thực tế có thể hạ ngưỡng để giảm tỷ lệ bỏ sót (chấp nhận báo nhầm nhiều hơn), tùy chi phí của từng loại sai.

### Khoanh vùng lỗi

Các điểm ảnh có giá trị bản đồ ≥ ngưỡng được tách thành vùng (contour, OpenCV). Vùng có diện tích ≥ 30 px được vẽ hộp bao màu đỏ.

### Vì sao chọn PatchCore

- Chỉ cần ảnh OK, không cần ảnh lỗi và không cần gán nhãn.
- Không phải huấn luyện mạng (không lan truyền ngược), chỉ trích đặc trưng và chọn coreset: 8–27 giây mỗi sản phẩm trên GPU.
- Thuộc nhóm có độ chính xác cao nhất trên MVTec AD (AUROC ảnh ~99% trong bài báo gốc).
- Cho bản đồ vùng lỗi, dễ giải thích kết quả cho người vận hành.

## 3.2. ResNet18: phân loại loại lỗi

Mã nguồn: [`aoi/classifier.py`](../aoi/classifier.py).

- Mạng ResNet18 huấn luyện trước trên ImageNet, thay lớp cuối bằng lớp có số đầu ra bằng số loại lỗi của sản phẩm (học chuyển giao, *transfer learning*).
- Dữ liệu: ảnh NG của nửa dev, nhãn là tên thư mục (không gán nhãn tay). Mỗi loại lỗi chỉ có khoảng 8–12 ảnh nên phải dùng học chuyển giao và tăng cường dữ liệu.
- Huấn luyện 40 epoch, AdamW (lr = 3·10⁻⁴), lịch cosine, *label smoothing* 0,1.
- Chỉ chạy khi PatchCore kết luận NG.

## 3.3. Huấn luyện

```bash
python scripts/train.py                          # 5 sản phẩm mặc định
python scripts/train.py --categories bottle      # một sản phẩm
```

Đầu ra cho mỗi sản phẩm trong `models/<sản phẩm>/`:

| Tệp | Nội dung |
|---|---|
| `patchcore.pt` | Memory bank (float16), ngưỡng, thang màu bản đồ nhiệt |
| `classifier.pt` | Trọng số ResNet18 + danh sách loại lỗi |
| `train_info.json` | Số ảnh, kích thước memory bank, ngưỡng, thời gian huấn luyện |

## 3.4. Tối ưu đã áp dụng

- Coreset 1% thay vì giữ toàn bộ memory bank: memory bank chỉ còn 1.600–3.100 vector, khoảng 5–10 MB/sản phẩm (float16), suy luận nhanh.
- Tính khoảng cách theo khối 8192 vector để không tràn bộ nhớ GPU.
- Ảnh huấn luyện bộ phân loại được thu nhỏ một lần và giữ trong RAM.
- Chạy được cả trên CPU (chậm hơn), tự chọn GPU nếu có.
