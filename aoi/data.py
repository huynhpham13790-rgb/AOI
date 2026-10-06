"""Đọc dữ liệu MVTec AD và tiền xử lý ảnh."""
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from . import config as C

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


@dataclass
class Sample:
    path: Path
    label: int          # 0 = OK (good), 1 = NG (có lỗi)
    defect: str         # "good" hoặc tên loại lỗi
    mask_path: Path | None


def eval_transform():
    return T.Compose([
        T.Resize(C.RESIZE),
        T.CenterCrop(C.CROP),
        T.ToTensor(),
        T.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD),
    ])


def load_image(src) -> Image.Image:
    """Nhận đường dẫn, bytes hoặc PIL Image; trả về ảnh RGB."""
    if isinstance(src, Image.Image):
        img = src
    elif isinstance(src, (bytes, bytearray)):
        import io
        img = Image.open(io.BytesIO(src))
    else:
        img = Image.open(src)
    return img.convert("RGB")


def preprocess_display(img: Image.Image) -> np.ndarray:
    """Ảnh RGB (CROP x CROP, uint8) cùng phép cắt với ảnh đưa vào mô hình, dùng để hiển thị."""
    t = T.Compose([T.Resize(C.RESIZE), T.CenterCrop(C.CROP)])
    return np.array(t(img))


def defect_crop(img: Image.Image, amap: np.ndarray, frac: float = 0.5) -> Image.Image:
    """Cắt vùng quanh điểm bất thường nhất, lấy từ ảnh gốc (độ phân giải cao).

    amap là bản đồ CROP x CROP trong hệ tọa độ sau tiền xử lý (resize RESIZE + center crop CROP);
    vùng cắt có cạnh bằng `frac` cạnh ảnh đã tiền xử lý.
    """
    w, h = img.size
    s = C.RESIZE / min(w, h)
    ox, oy = (w * s - C.CROP) / 2, (h * s - C.CROP) / 2
    v, u = np.unravel_index(np.argmax(amap), amap.shape)
    cx, cy = (u + ox) / s, (v + oy) / s
    half = frac * C.CROP / s / 2
    cx = min(max(cx, half), w - half)
    cy = min(max(cy, half), h - half)
    return img.crop((int(cx - half), int(cy - half), int(cx + half), int(cy + half)))


def load_mask(path: Path | None) -> np.ndarray:
    if path is None:
        return np.zeros((C.CROP, C.CROP), dtype=np.uint8)
    m = Image.open(path).convert("L")
    m = T.Compose([T.Resize(C.RESIZE, interpolation=T.InterpolationMode.NEAREST), T.CenterCrop(C.CROP)])(m)
    return (np.array(m) > 127).astype(np.uint8)


def train_samples(category: str) -> list[Sample]:
    d = C.DATA_DIR / category / "train" / "good"
    return [Sample(p, 0, "good", None) for p in sorted(d.iterdir()) if p.suffix.lower() in IMG_EXTS]


def test_samples(category: str) -> list[Sample]:
    root = C.DATA_DIR / category
    out = []
    for d in sorted((root / "test").iterdir()):
        for p in sorted(d.iterdir()):
            if p.suffix.lower() not in IMG_EXTS:
                continue
            if d.name == "good":
                out.append(Sample(p, 0, "good", None))
            else:
                m = root / "ground_truth" / d.name / f"{p.stem}_mask.png"
                out.append(Sample(p, 1, d.name, m if m.exists() else None))
    return out


def split_test(category: str) -> tuple[list[Sample], list[Sample]]:
    """Chia tập test thành dev/holdout, phân tầng theo loại lỗi, cố định seed."""
    by_defect: dict[str, list[Sample]] = {}
    for s in test_samples(category):
        by_defect.setdefault(s.defect, []).append(s)
    rng = random.Random(C.SPLIT_SEED)
    dev, hold = [], []
    for defect in sorted(by_defect):
        items = by_defect[defect][:]
        rng.shuffle(items)
        k = int(round(len(items) * C.DEV_FRACTION))
        dev += items[:k]
        hold += items[k:]
    return dev, hold


def defect_types(category: str) -> list[str]:
    return sorted(d.name for d in (C.DATA_DIR / category / "test").iterdir() if d.is_dir() and d.name != "good")


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, samples: list[Sample], transform=None):
        self.samples = samples
        self.transform = transform or eval_transform()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.transform(load_image(self.samples[i].path))


def overlay_heatmap(rgb: np.ndarray, amap: np.ndarray, vmin: float, vmax: float, alpha=0.5) -> np.ndarray:
    """Chồng bản đồ bất thường (đã chuẩn hóa theo [vmin, vmax]) lên ảnh."""
    norm = np.clip((amap - vmin) / max(vmax - vmin, 1e-8), 0, 1)
    heat = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return (rgb * (1 - alpha) + heat * alpha).astype(np.uint8)


def draw_defect_boxes(rgb: np.ndarray, amap: np.ndarray, threshold: float, min_area: int = 30) -> np.ndarray:
    """Khoanh vùng lỗi: các vùng có điểm bất thường vượt ngưỡng."""
    out = rgb.copy()
    binary = (amap >= threshold).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(out, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return out
