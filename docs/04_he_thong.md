# 4. Hệ thống kiểm tra tự động

## 4.1. Luồng xử lý

Theo đề cương: **camera → xử lý ảnh → AI → kết quả OK/NG**. Vì demo trên laptop, không có camera công nghiệp, bước "camera" được thay bằng **tải ảnh lên** hoặc **webcam laptop**.

```
┌────────────┐   ┌──────────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐
│ Nguồn ảnh  │──▶│ Tiền xử lý       │──▶│ PatchCore    │──▶│ ResNet18     │──▶│ Kết quả          │
│ - tải lên  │   │ - RGB            │   │ - điểm       │   │ (nếu NG)     │   │ - OK / NG        │
│ - webcam   │   │ - resize 256     │   │ - bản đồ lỗi │   │ - loại lỗi   │   │ - vùng lỗi       │
│ - ảnh mẫu  │   │ - crop 224       │   │ - so ngưỡng  │   │              │   │ - nhật ký CSV    │
└────────────┘   │ - chuẩn hóa      │   └──────────────┘   └──────────────┘   │ - lưu ảnh NG     │
                 └──────────────────┘                                        └──────────────────┘
```

Toàn bộ luồng nằm trong lớp `Inspector` ([`aoi/inspector.py`](../aoi/inspector.py)):

```python
from aoi.inspector import Inspector

insp = Inspector("bottle")
res = insp.inspect("anh_chai.png")
print(res.verdict, res.score, res.defect, res.time_ms)
```

## 4.2. Giao diện web (Streamlit)

```bash
streamlit run app.py
```

Mở trình duyệt tại `http://localhost:8501`.

**Thanh bên trái**
- *Sản phẩm*: chọn mô hình tương ứng (chai, đai ốc, hạt phỉ, ốc vít, viên nang).
- *Nguồn ảnh*:
  - **Tải ảnh lên**: chọn một hoặc nhiều ảnh (png/jpg/bmp), mô phỏng camera chụp sản phẩm trên dây chuyền.
  - **Ảnh mẫu (bộ dữ liệu)**: lấy ngẫu nhiên ảnh từ tập holdout để demo nhanh.
  - **Webcam laptop**: chụp trực tiếp bằng webcam.
- *Lưu nhật ký và ảnh NG*: bật/tắt ghi log.

**Vùng kết quả**
- Thống kê lô: tổng số ảnh, số OK, số NG, tỷ lệ đạt; bảng tóm tắt khi kiểm nhiều ảnh.
- Với từng ảnh:
  - Nhãn lớn **OK - ĐẠT** (xanh) hoặc **NG - LỖI** (đỏ).
  - Điểm bất thường, ngưỡng, loại lỗi, thời gian xử lý.
  - Ba ảnh: ảnh sau tiền xử lý, bản đồ nhiệt (đỏ = bất thường), ảnh khoanh vùng lỗi.
  - Biểu đồ xác suất các loại lỗi (khi NG).

## 4.3. Nhật ký và truy vết

Khi bật lưu nhật ký:
- `logs/inspection_log.csv`: mỗi dòng một lần kiểm tra (thời gian, sản phẩm, tên ảnh, kết quả, điểm, ngưỡng, loại lỗi, thời gian xử lý, đường dẫn ảnh NG).
- `logs/ng_images/`: lưu bản sao ảnh NG để kiểm tra lại hoặc bổ sung dữ liệu huấn luyện.

## 4.4. Dòng lệnh

Kiểm tra hàng loạt ảnh hoặc thư mục không cần giao diện:

```bash
python scripts/inspect_images.py --category metal_nut duong_dan/anh1.png thu_muc_anh/
```

```
OK  điểm=1.912  ngưỡng=2.480  lỗi=-             38 ms  ...\good\000.png
NG  điểm=3.704  ngưỡng=2.480  lỗi=scratch       41 ms  ...\scratch\003.png
```

## 4.5. Hướng tích hợp phần cứng thực tế

Bản demo không có phần cứng, nhưng `Inspector` tách biệt khỏi giao diện nên dễ gắn vào dây chuyền:

| Thay thế | Cách làm |
|---|---|
| Tải ảnh lên → camera công nghiệp | Dùng SDK camera (Basler pylon, Hikrobot MVS) hoặc `cv2.VideoCapture`, chụp khi cảm biến quang báo có sản phẩm, gọi `insp.inspect(frame)` |
| Màn hình → tín hiệu điều khiển | Gửi `OK`/`NG` qua Serial (Arduino bật đèn/còi, servo gạt) hoặc Modbus TCP / OPC UA tới PLC để xi lanh loại sản phẩm |

Ví dụ gửi kết quả qua cổng Serial (cần `pip install pyserial`):

```python
import serial
port = serial.Serial("COM3", 9600)
res = insp.inspect(frame)
port.write(b"1" if res.verdict == "NG" else b"0")
```
