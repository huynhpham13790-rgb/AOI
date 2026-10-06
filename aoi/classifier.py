"""Bộ phân loại loại lỗi (ResNet18 fine-tune).

Chỉ chạy khi PatchCore đã kết luận NG, để trả lời câu hỏi "lỗi gì?".
Đầu vào mặc định là vùng cắt quanh điểm bất thường nhất do PatchCore tìm ra,
giúp nhìn rõ lỗi nhỏ (vết xước trên ốc vít, viên nang...).
Nhãn lấy trực tiếp từ tên thư mục của MVTec AD (không gán nhãn thủ công).
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.models import ResNet18_Weights, resnet18

from . import config as C
from .data import defect_crop


def train_transform():
    # Không lật ảnh, không đổi màu: một số loại lỗi chính là "bị lật" (flip) hoặc "sai màu" (color).
    return T.Compose([
        T.Resize((C.RESIZE, C.RESIZE)),
        T.RandomRotation(15),
        T.RandomResizedCrop(C.CROP, scale=(0.75, 1.0), ratio=(0.9, 1.1)),
        T.ColorJitter(brightness=0.15, contrast=0.15),
        T.ToTensor(),
        T.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD),
    ])


def infer_transform():
    return T.Compose([
        T.Resize((C.CROP, C.CROP)),
        T.ToTensor(),
        T.Normalize(C.IMAGENET_MEAN, C.IMAGENET_STD),
    ])


def classifier_input(img: Image.Image, amap: np.ndarray, mode: str) -> Image.Image:
    """Ảnh đưa vào bộ phân loại: vùng cắt quanh lỗi ("crop") hoặc toàn ảnh ("full")."""
    if mode == "crop":
        return defect_crop(img, amap)
    return T.CenterCrop(min(img.size))(img)


class _DS(torch.utils.data.Dataset):
    def __init__(self, images, labels, tf):
        # Thu nhỏ một lần rồi giữ trong RAM (ảnh MVTec gốc tới 1024px, đọc lại mỗi epoch rất chậm)
        self.images = [T.Resize((C.RESIZE, C.RESIZE))(im) for im in images]
        self.labels, self.tf = labels, tf

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.tf(self.images[i]), self.labels[i]


def build_model(num_classes: int) -> torch.nn.Module:
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
    return m


def train_classifier(images: list[Image.Image], labels: list[int], num_classes: int, epochs: int = 40,
                     device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes).to(device)
    loader = torch.utils.data.DataLoader(_DS(images, labels, train_transform()), batch_size=16, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    for _ in range(epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
        sched.step()
    return model.eval()


class DefectClassifier:
    def __init__(self, model: torch.nn.Module, classes: list[str], device: str, mode: str = "crop"):
        self.model, self.classes, self.device, self.mode = model.to(device).eval(), classes, device, mode
        self.tf = infer_transform()

    @torch.no_grad()
    def predict(self, images: list[Image.Image]) -> np.ndarray:
        """Trả về xác suất N x num_classes."""
        x = torch.stack([self.tf(im) for im in images]).to(self.device)
        return torch.softmax(self.model(x), dim=1).cpu().numpy()

    def save(self, path: Path):
        torch.save({"state_dict": self.model.state_dict(), "classes": self.classes, "mode": self.mode}, path)

    @classmethod
    def load(cls, path: Path, device: str | None = None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        m = build_model(len(ckpt["classes"]))
        m.load_state_dict(ckpt["state_dict"])
        return cls(m, ckpt["classes"], device, ckpt.get("mode", "full"))
