"""Huấn luyện mô hình cho từng loại sản phẩm.

Với mỗi sản phẩm:
1. PatchCore: học từ ảnh OK trong train/good (không cần ảnh lỗi).
2. Chọn ngưỡng OK/NG trên nửa "dev" của tập test (tối đa F1).
3. ResNet18: học phân loại loại lỗi từ ảnh NG của nửa "dev"
   (mặc định trên vùng cắt quanh điểm bất thường do PatchCore tìm ra).
Nửa "holdout" được giữ lại cho scripts/evaluate.py.

Dùng:
    python scripts/train.py
    python scripts/train.py --categories bottle metal_nut
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aoi import config as C  # noqa: E402
from aoi.classifier import DefectClassifier, classifier_input, train_classifier  # noqa: E402
from aoi.data import ImageDataset, defect_types, load_image, split_test, train_samples  # noqa: E402
from aoi.patchcore import PatchCore  # noqa: E402


def best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    p, r, t = precision_recall_curve(labels, scores)
    f1 = 2 * p * r / np.maximum(p + r, 1e-12)
    return float(t[np.argmax(f1[:-1])])


def train_category(cat: str, epochs: int, cls_input: str) -> dict:
    torch.manual_seed(0)
    out_dir = C.MODELS_DIR / cat
    out_dir.mkdir(parents=True, exist_ok=True)
    loader = lambda s: torch.utils.data.DataLoader(ImageDataset(s), batch_size=16, num_workers=0)  # noqa: E731

    t0 = time.time()
    train = train_samples(cat)
    pc = PatchCore().fit(loader(train))
    t_fit = time.time() - t0

    dev, _ = split_test(cat)
    labels = np.array([s.label for s in dev])
    scores, maps = pc.predict_loader(loader(dev))
    thr = best_f1_threshold(labels, scores)
    good_maps = maps[labels == 0].astype(np.float32)
    map_min = float(np.percentile(good_maps, 50)) if len(good_maps) else float(scores.min())
    map_max = float(np.percentile(scores[labels == 1], 90))
    pc.save(out_dir / "patchcore.pt", threshold=thr, map_min=map_min, map_max=map_max, category=cat)

    classes = defect_types(cat)
    t1 = time.time()
    ng = [i for i, s in enumerate(dev) if s.label == 1]
    images = [classifier_input(load_image(dev[i].path), maps[i].astype(np.float32), cls_input) for i in ng]
    clf = train_classifier(images, [classes.index(dev[i].defect) for i in ng], len(classes), epochs=epochs)
    DefectClassifier(clf, classes, pc.device, cls_input).save(out_dir / "classifier.pt")
    t_cls = time.time() - t1

    info = {
        "category": cat, "train_ok_images": len(train), "dev_images": len(dev),
        "memory_bank": list(pc.memory.shape), "threshold": thr,
        "defect_classes": classes, "classifier_input": cls_input,
        "fit_seconds": round(t_fit, 1), "classifier_seconds": round(t_cls, 1),
    }
    (out_dir / "train_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="+", default=C.DEFAULT_CATEGORIES)
    ap.add_argument("--epochs", type=int, default=40, help="số epoch cho bộ phân loại lỗi")
    ap.add_argument("--cls-input", choices=["crop", "full"], default="crop",
                    help="đầu vào bộ phân loại lỗi: vùng cắt quanh lỗi (crop) hoặc toàn ảnh (full)")
    args = ap.parse_args()
    for cat in args.categories:
        info = train_category(cat, args.epochs, args.cls_input)
        print(f"[{cat}] memory bank {info['memory_bank']}, ngưỡng {info['threshold']:.3f}, "
              f"PatchCore {info['fit_seconds']}s, phân loại {info['classifier_seconds']}s")


if __name__ == "__main__":
    main()
