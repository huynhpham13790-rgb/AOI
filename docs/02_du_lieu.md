# 2. Dữ liệu

## 2.1. Lựa chọn bộ dữ liệu

Yêu cầu: dữ liệu ảnh sản phẩm công nghiệp **đã có nhãn sẵn**, không phải gán nhãn thủ công. Một số bộ đã khảo sát:

| Bộ dữ liệu | Nội dung | Nhãn có sẵn |
|---|---|---|
| **MVTec AD** (chọn) | 15 loại sản phẩm/bề mặt, 5.354 ảnh | OK/NG, tên loại lỗi, mask pixel vùng lỗi |
| VisA (Amazon) | 12 loại (PCB, viên nang, kẹo...) | OK/NG + mask |
| NEU-DET | Thép cán nóng, 6 loại lỗi | Hộp bao |
| DeepPCB | Mạch in, 6 loại lỗi | Hộp bao |
| Casting Product (Kaggle) | Bánh công tác đúc | OK/NG |

**MVTec AD** được chọn vì:
- Là bộ chuẩn (benchmark) phổ biến nhất cho kiểm tra ngoại quan công nghiệp, dễ so sánh với các công trình khác.
- Có đủ ba loại nhãn: OK/NG, loại lỗi và mask vùng lỗi, đáp ứng cả phát hiện, phân loại và khoanh vùng.
- Tập huấn luyện chỉ gồm ảnh OK, đúng với điều kiện thực tế nhà máy.

Giấy phép: CC BY-NC-SA 4.0, chỉ dùng cho học tập/nghiên cứu phi thương mại.
Trích dẫn: Bergmann et al., *MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection*, CVPR 2019.

## 2.2. Các sản phẩm sử dụng

Đề tài dùng 5 sản phẩm dạng "vật thể" (giống sản phẩm rời trên băng chuyền):

| Sản phẩm | Mã | Ảnh huấn luyện (OK) | Ảnh test (OK / NG) | Các loại lỗi |
|---|---|---|---|---|
| Chai thủy tinh | `bottle` | 209 | 20 / 63 | broken_large (vỡ lớn), broken_small (mẻ nhỏ), contamination (bẩn) |
| Đai ốc kim loại | `metal_nut` | 220 | 22 / 93 | bent (cong), color (sai màu), flip (lật ngược), scratch (xước) |
| Hạt phỉ | `hazelnut` | 391 | 40 / 70 | crack (nứt), cut (vết cắt), hole (lỗ), print (vết in) |
| Ốc vít | `screw` | 320 | 41 / 119 | manipulated_front (hỏng đầu), scratch_head (xước mũ), scratch_neck (xước cổ), thread_side, thread_top (hỏng ren) |
| Viên nang | `capsule` | 219 | 23 / 109 | crack (nứt), faulty_imprint (in lỗi), poke (thủng), scratch (xước), squeeze (bẹp) |

Có thể dùng thêm 10 loại còn lại bằng tham số `--categories`.

## 2.3. Tải dữ liệu

```bash
python scripts/download_mvtec.py                     # 5 sản phẩm mặc định (~1,5 GB)
python scripts/download_mvtec.py --categories all    # toàn bộ 15 loại (~5 GB)
```

Script tải từ bản sao trên HuggingFace (`foersben/mvtec-ad`) vì link gốc của MVTec yêu cầu đăng ký. Cấu trúc sau khi tải:

```
data/mvtec_ad/<sản phẩm>/
├── train/good/             # ảnh OK để huấn luyện PatchCore
├── test/good/              # ảnh OK để kiểm thử
├── test/<loại lỗi>/        # ảnh NG, tên thư mục = nhãn loại lỗi
└── ground_truth/<loại lỗi>/  # mask vùng lỗi (trắng = lỗi)
```

## 2.4. Chia dữ liệu

MVTec AD chỉ có ảnh lỗi trong tập test. Để vừa chọn ngưỡng, vừa huấn luyện bộ phân loại lỗi mà vẫn đánh giá khách quan, tập test được chia **phân tầng theo loại lỗi** (seed cố định = 42) thành hai nửa:

| Phần | Dùng cho |
|---|---|
| `train/good` | Huấn luyện PatchCore (memory bank) |
| Test – nửa **dev** (50%) | Chọn ngưỡng OK/NG; huấn luyện bộ phân loại loại lỗi |
| Test – nửa **holdout** (50%) | Đánh giá cuối cùng, mô hình chưa từng thấy |

## 2.5. Tiền xử lý ảnh

1. Chuyển sang RGB (ảnh xám như `screw` được nhân 3 kênh).
2. Thu nhỏ cạnh ngắn về 256 px, cắt giữa 224×224 (bỏ phần nền ở mép).
3. Chuẩn hóa theo trung bình/độ lệch chuẩn của ImageNet (vì mạng backbone huấn luyện trên ImageNet).

Riêng khi huấn luyện bộ phân loại, áp dụng tăng cường dữ liệu: xoay ±15°, cắt ngẫu nhiên 75–100%, thay đổi nhẹ độ sáng/tương phản. **Không** lật ảnh và **không** đổi màu vì một số lỗi chính là "lật ngược" (`flip`) và "sai màu" (`color`).
