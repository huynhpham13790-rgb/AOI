"""Cấu hình chung cho toàn bộ dự án AOI."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "mvtec_ad"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
LOGS_DIR = ROOT / "logs"

# Các sản phẩm dùng mặc định (tập con của 15 loại trong MVTec AD)
DEFAULT_CATEGORIES = ["bottle", "metal_nut", "hazelnut", "screw", "capsule"]
ALL_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]

# Tên hiển thị tiếng Việt
CATEGORY_VI = {
    "bottle": "Chai thủy tinh", "cable": "Cáp điện", "capsule": "Viên nang",
    "carpet": "Thảm", "grid": "Lưới kim loại", "hazelnut": "Hạt phỉ",
    "leather": "Da", "metal_nut": "Đai ốc kim loại", "pill": "Viên thuốc",
    "screw": "Ốc vít", "tile": "Gạch", "toothbrush": "Bàn chải",
    "transistor": "Transistor", "wood": "Gỗ", "zipper": "Khóa kéo",
}

# Tiền xử lý ảnh
RESIZE = 256
CROP = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# PatchCore
CORESET_RATIO = 0.01   # giữ lại 1% patch đặc trưng làm memory bank
NUM_NEIGHBORS = 1

# Chia tập test MVTec thành 2 nửa: "dev" (chọn ngưỡng + huấn luyện bộ phân loại lỗi)
# và "holdout" (đánh giá cuối cùng, mô hình chưa từng thấy).
SPLIT_SEED = 42
DEV_FRACTION = 0.5
