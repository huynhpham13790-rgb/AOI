"""Đánh giá hệ thống trên nửa "holdout" của tập test (mô hình chưa từng thấy).

Chỉ tiêu (theo mục 6 đề cương):
- Độ chính xác (Accuracy)
- Tỷ lệ phát hiện lỗi (Recall trên ảnh NG)
- Tỷ lệ báo lỗi nhầm (ảnh OK bị báo NG) và tỷ lệ bỏ sót lỗi
- AUROC mức ảnh và mức pixel (khả năng khoanh vùng lỗi)
- Độ chính xác phân loại loại lỗi
- Thời gian xử lý mỗi ảnh (ms) qua đúng pipeline Inspector

Kết quả ghi vào results/: metrics.json, summary.md và các hình.

Dùng:
    python scripts/evaluate.py
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import (ConfusionMatrixDisplay, confusion_matrix, roc_auc_score,  # noqa: E402
                             roc_curve)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aoi import config as C  # noqa: E402
from aoi.classifier import classifier_input  # noqa: E402
from aoi.data import ImageDataset, load_image, load_mask, split_test  # noqa: E402
from aoi.inspector import Inspector, available_categories  # noqa: E402

plt.rcParams["font.family"] = "DejaVu Sans"


def evaluate_category(cat: str, fig_dir: Path) -> dict:
    insp = Inspector(cat)
    _, hold = split_test(cat)
    labels = np.array([s.label for s in hold])
    loader = torch.utils.data.DataLoader(ImageDataset(hold), batch_size=16)
    scores, maps = insp.patchcore.predict_loader(loader)
    pred = (scores >= insp.threshold).astype(int)

    tp = int(((pred == 1) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    masks = np.stack([load_mask(s.mask_path) for s in hold])
    pixel_auroc = roc_auc_score(masks.ravel(), maps.astype(np.float32).ravel())

    # Phân loại loại lỗi trên ảnh NG của holdout
    ng = np.where(labels == 1)[0]
    probs = insp.classifier.predict([
        classifier_input(load_image(hold[i].path), maps[i].astype(np.float32), insp.classifier.mode) for i in ng])
    y_true = [insp.classifier.classes.index(hold[i].defect) for i in ng]
    y_pred = probs.argmax(1)
    cls_acc = float((y_pred == np.array(y_true)).mean())

    # Thời gian xử lý thực tế qua pipeline (đọc ảnh -> tiền xử lý -> AI -> kết quả)
    for s in hold[:3]:
        insp.inspect(s.path, visualize=False)  # khởi động GPU
    times = [insp.inspect(s.path, visualize=False).time_ms for s in hold[:30]]

    # Hình: ma trận nhầm lẫn OK/NG, ma trận nhầm lẫn loại lỗi, đường ROC
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ConfusionMatrixDisplay(confusion_matrix(labels, pred, labels=[0, 1]), display_labels=["OK", "NG"]).plot(
        ax=ax[0], colorbar=False, cmap="Blues")
    ax[0].set_title(f"{cat}: OK/NG")
    ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred, labels=range(len(insp.classifier.classes))),
                           display_labels=insp.classifier.classes).plot(
        ax=ax[1], colorbar=False, cmap="Oranges", xticks_rotation=45)
    ax[1].set_title(f"{cat}: loại lỗi")
    fpr, tpr, _ = roc_curve(labels, scores)
    ax[2].plot(fpr, tpr, label=f"AUROC = {roc_auc_score(labels, scores):.3f}")
    ax[2].plot([0, 1], [0, 1], "--", color="gray")
    ax[2].set_xlabel("Tỷ lệ báo nhầm (FPR)")
    ax[2].set_ylabel("Tỷ lệ phát hiện (TPR)")
    ax[2].legend(loc="lower right")
    ax[2].set_title(f"{cat}: ROC")
    fig.tight_layout()
    fig.savefig(fig_dir / f"{cat}_metrics.png", dpi=110)
    plt.close(fig)

    save_examples(insp, hold, scores, labels, fig_dir / f"{cat}_examples.png")

    return {
        "category": cat, "holdout_images": len(hold), "ok_images": int((labels == 0).sum()),
        "ng_images": int((labels == 1).sum()), "threshold": insp.threshold,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "accuracy": (tp + tn) / len(hold), "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "false_alarm_rate": fp / max(fp + tn, 1), "miss_rate": fn / max(fn + tp, 1),
        "image_auroc": float(roc_auc_score(labels, scores)), "pixel_auroc": float(pixel_auroc),
        "defect_cls_accuracy": cls_acc, "defect_classes": insp.classifier.classes,
        "time_ms_mean": float(np.mean(times)), "time_ms_p95": float(np.percentile(times, 95)),
    }


def save_examples(insp: Inspector, hold, scores, labels, path: Path, n: int = 4):
    """Lưới ảnh minh họa: ảnh gốc / bản đồ nhiệt / khoanh vùng cho vài ảnh OK và NG."""
    rng = np.random.default_rng(0)
    pick = list(rng.choice(np.where(labels == 0)[0], 1, replace=False)) + \
        list(rng.choice(np.where(labels == 1)[0], n - 1, replace=False))
    fig, ax = plt.subplots(len(pick), 3, figsize=(9, 3 * len(pick)))
    for r, i in enumerate(pick):
        res = insp.inspect(hold[i].path)
        truth = "OK" if labels[i] == 0 else f"NG ({hold[i].defect})"
        pred = res.verdict + (f" ({res.defect})" if res.defect else "")
        for c, (img, title) in enumerate([(res.image, f"Thực tế: {truth}"),
                                          (res.heatmap, f"Điểm: {res.score:.2f} / ngưỡng {res.threshold:.2f}"),
                                          (res.boxes, f"Dự đoán: {pred}")]):
            ax[r, c].imshow(img)
            ax[r, c].set_title(title, fontsize=9, color="green" if res.verdict == "OK" or c != 2 else "red")
            ax[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def write_summary(rows: list[dict], path: Path):
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    lines = [
        "# Kết quả đánh giá",
        "",
        f"Đánh giá trên nửa holdout của tập test MVTec AD (mô hình chưa từng thấy). Thiết bị: {device}.",
        "",
        "| Sản phẩm | Số ảnh (OK/NG) | Accuracy | Tỷ lệ phát hiện lỗi | Tỷ lệ báo nhầm | Tỷ lệ bỏ sót | "
        "AUROC ảnh | AUROC pixel | Phân loại loại lỗi | Thời gian (ms) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {C.CATEGORY_VI.get(r['category'], r['category'])} (`{r['category']}`) | "
            f"{r['holdout_images']} ({r['ok_images']}/{r['ng_images']}) | {r['accuracy']:.1%} | {r['recall']:.1%} | "
            f"{r['false_alarm_rate']:.1%} | {r['miss_rate']:.1%} | {r['image_auroc']:.3f} | {r['pixel_auroc']:.3f} | "
            f"{r['defect_cls_accuracy']:.1%} ({len(r['defect_classes'])} lớp) | {r['time_ms_mean']:.0f} |")
    keys = ["accuracy", "recall", "false_alarm_rate", "miss_rate", "image_auroc", "pixel_auroc",
            "defect_cls_accuracy", "time_ms_mean"]
    m = {k: np.mean([r[k] for r in rows]) for k in keys}
    lines.append(
        f"| **Trung bình** | {sum(r['holdout_images'] for r in rows)} | **{m['accuracy']:.1%}** | "
        f"**{m['recall']:.1%}** | **{m['false_alarm_rate']:.1%}** | **{m['miss_rate']:.1%}** | "
        f"**{m['image_auroc']:.3f}** | **{m['pixel_auroc']:.3f}** | **{m['defect_cls_accuracy']:.1%}** | "
        f"**{m['time_ms_mean']:.0f}** |")
    lines += ["", "Hình chi tiết từng sản phẩm:", ""]
    for r in rows:
        lines += [f"### {C.CATEGORY_VI.get(r['category'], r['category'])}", "",
                  f"![metrics](figures/{r['category']}_metrics.png)", "",
                  f"![examples](figures/{r['category']}_examples.png)", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="+", default=None)
    args = ap.parse_args()
    cats = args.categories or available_categories()
    fig_dir = C.RESULTS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for cat in cats:
        r = evaluate_category(cat, fig_dir)
        rows.append(r)
        print(f"[{cat}] acc {r['accuracy']:.1%} | phát hiện {r['recall']:.1%} | báo nhầm {r['false_alarm_rate']:.1%} "
              f"| AUROC {r['image_auroc']:.3f}/{r['pixel_auroc']:.3f} | loại lỗi {r['defect_cls_accuracy']:.1%} "
              f"| {r['time_ms_mean']:.0f} ms")
    (C.RESULTS_DIR / "metrics.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    write_summary(rows, C.RESULTS_DIR / "summary.md")
    print(f"Đã ghi {C.RESULTS_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
