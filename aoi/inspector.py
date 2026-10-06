"""Pipeline kiểm tra hoàn chỉnh: ảnh -> tiền xử lý -> AI -> kết quả OK/NG."""
import time
from dataclasses import dataclass, field

import numpy as np

from . import config as C
from .classifier import DefectClassifier, classifier_input
from .data import draw_defect_boxes, eval_transform, load_image, overlay_heatmap, preprocess_display
from .patchcore import PatchCore


@dataclass
class InspectionResult:
    verdict: str                     # "OK" hoặc "NG"
    score: float                     # điểm bất thường của ảnh
    threshold: float
    defect: str | None               # loại lỗi dự đoán (None nếu OK)
    defect_probs: dict = field(default_factory=dict)
    time_ms: float = 0.0
    image: np.ndarray | None = None      # ảnh sau tiền xử lý
    heatmap: np.ndarray | None = None    # ảnh chồng bản đồ nhiệt
    boxes: np.ndarray | None = None      # ảnh có khoanh vùng lỗi


class Inspector:
    """Bộ kiểm tra cho một loại sản phẩm."""

    def __init__(self, category: str, device: str | None = None):
        d = C.MODELS_DIR / category
        self.category = category
        self.patchcore, meta = PatchCore.load(d / "patchcore.pt", device)
        self.threshold = float(meta["threshold"])
        self.map_min, self.map_max = float(meta["map_min"]), float(meta["map_max"])
        cls_path = d / "classifier.pt"
        self.classifier = DefectClassifier.load(cls_path, self.patchcore.device) if cls_path.exists() else None
        self.tf = eval_transform()

    def inspect(self, src, visualize: bool = True) -> InspectionResult:
        t0 = time.perf_counter()
        img = load_image(src)
        x = self.tf(img).unsqueeze(0)
        scores, amap = self.patchcore.predict(x)
        score, amap = float(scores[0]), amap[0]
        verdict = "NG" if score >= self.threshold else "OK"
        defect, probs = None, {}
        if verdict == "NG" and self.classifier is not None:
            p = self.classifier.predict([classifier_input(img, amap, self.classifier.mode)])[0]
            probs = {c: float(v) for c, v in zip(self.classifier.classes, p)}
            defect = max(probs, key=probs.get)
        elapsed = (time.perf_counter() - t0) * 1000

        res = InspectionResult(verdict, score, self.threshold, defect, probs, elapsed)
        if visualize:
            rgb = preprocess_display(img)
            res.image = rgb
            res.heatmap = overlay_heatmap(rgb, amap, self.map_min, self.map_max)
            res.boxes = draw_defect_boxes(rgb, amap, self.threshold) if verdict == "NG" else rgb
        return res


def available_categories() -> list[str]:
    if not C.MODELS_DIR.exists():
        return []
    return sorted(p.name for p in C.MODELS_DIR.iterdir() if (p / "patchcore.pt").exists())
