"""PatchCore: phát hiện bất thường chỉ cần ảnh sản phẩm tốt (OK) để huấn luyện.

Roth et al., "Towards Total Recall in Industrial Anomaly Detection", CVPR 2022.

Ý tưởng:
1. Dùng mạng CNN đã huấn luyện trên ImageNet (WideResNet50) trích đặc trưng cục bộ
   cho từng ô (patch) 8x8 pixel của ảnh.
2. Gom đặc trưng của tất cả patch trong ảnh OK thành "memory bank", rút gọn bằng
   coreset (chọn tập con đại diện).
3. Khi kiểm tra: patch nào cách xa mọi patch OK trong memory bank -> bất thường.
   Điểm của ảnh = điểm lớn nhất trong các patch.
"""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from torchvision.models import Wide_ResNet50_2_Weights, wide_resnet50_2
from tqdm import tqdm

from . import config as C


class FeatureExtractor(torch.nn.Module):
    """Lấy đặc trưng tầng giữa (layer2, layer3) của WideResNet50."""

    def __init__(self):
        super().__init__()
        net = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        self.stem = torch.nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1)
        self.layer2, self.layer3 = net.layer2, net.layer3
        self.eval().requires_grad_(False)

    @torch.no_grad()
    def forward(self, x):
        f2 = self.layer2(self.stem(x))     # B x 512 x 28 x 28
        f3 = self.layer3(f2)               # B x 1024 x 14 x 14
        # Gộp vùng lân cận 3x3 để mỗi patch "nhìn" được ngữ cảnh xung quanh
        f2 = F.avg_pool2d(f2, 3, 1, 1)
        f3 = F.avg_pool2d(f3, 3, 1, 1)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([f2, f3], dim=1)  # B x 1536 x 28 x 28


def greedy_coreset(features: torch.Tensor, ratio: float, proj_dim: int = 128, seed: int = 0) -> torch.Tensor:
    """Chọn tập con các patch sao cho phủ đều không gian đặc trưng (k-center greedy)."""
    n = features.shape[0]
    k = max(1, int(n * ratio))
    g = torch.Generator(device="cpu").manual_seed(seed)
    proj = torch.randn(features.shape[1], proj_dim, generator=g).to(features.device)
    z = features @ proj  # chiếu ngẫu nhiên để tính khoảng cách nhanh hơn
    idx = [int(torch.randint(n, (1,), generator=g))]
    min_d = torch.cdist(z, z[idx]).squeeze(1)
    for _ in tqdm(range(k - 1), desc="coreset", leave=False):
        i = int(torch.argmax(min_d))
        idx.append(i)
        min_d = torch.minimum(min_d, torch.cdist(z, z[i:i + 1]).squeeze(1))
    return torch.tensor(idx, device=features.device)


class PatchCore:
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.extractor = FeatureExtractor().to(self.device)
        self.memory: torch.Tensor | None = None
        self.feat_hw = (28, 28)

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        f = self.extractor(x.to(self.device))
        self.feat_hw = f.shape[-2:]
        return f.permute(0, 2, 3, 1).reshape(-1, f.shape[1])  # (B*H*W) x D

    def fit(self, loader, coreset_ratio: float = C.CORESET_RATIO):
        feats = torch.cat([self._embed(x) for x in tqdm(loader, desc="trích đặc trưng", leave=False)])
        idx = greedy_coreset(feats, coreset_ratio)
        self.memory = feats[idx].contiguous()
        return self

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        """Trả về (điểm ảnh: B, bản đồ bất thường: B x CROP x CROP)."""
        b = x.shape[0]
        f = self._embed(x)
        d = torch.cat([torch.cdist(chunk, self.memory).min(dim=1).values for chunk in f.split(8192)])
        amap = d.reshape(b, 1, *self.feat_hw)
        amap = F.interpolate(amap, size=(C.CROP, C.CROP), mode="bilinear", align_corners=False)
        amap = amap.squeeze(1).cpu().numpy()
        amap = np.stack([gaussian_filter(m, sigma=4) for m in amap])
        scores = amap.reshape(b, -1).max(axis=1)
        return scores, amap

    def predict_loader(self, loader):
        scores, maps = [], []
        for x in loader:
            s, m = self.predict(x)
            scores.append(s)
            maps.append(m.astype(np.float16))
        return np.concatenate(scores), np.concatenate(maps)

    def save(self, path: Path, **meta):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"memory": self.memory.half().cpu(), "feat_hw": tuple(self.feat_hw), **meta}, path)

    @classmethod
    def load(cls, path: Path, device: str | None = None):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        m = cls(device)
        m.memory = ckpt["memory"].float().to(m.device)
        m.feat_hw = ckpt["feat_hw"]
        meta = {k: v for k, v in ckpt.items() if k not in ("memory", "feat_hw")}
        return m, meta
