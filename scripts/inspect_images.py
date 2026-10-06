"""Kiểm tra ảnh từ dòng lệnh (không cần giao diện).

Dùng:
    python scripts/inspect_images.py --category bottle anh1.png anh2.jpg thu_muc_anh/
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aoi.data import IMG_EXTS  # noqa: E402
from aoi.inspector import Inspector  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", required=True)
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args()
    files = []
    for p in map(Path, args.paths):
        files += sorted(f for f in p.rglob("*") if f.suffix.lower() in IMG_EXTS) if p.is_dir() else [p]
    insp = Inspector(args.category)
    n_ng = 0
    for f in files:
        r = insp.inspect(f, visualize=False)
        n_ng += r.verdict == "NG"
        print(f"{r.verdict}  điểm={r.score:.3f}  ngưỡng={r.threshold:.3f}  "
              f"lỗi={r.defect or '-':<12} {r.time_ms:6.0f} ms  {f}")
    print(f"Tổng: {len(files)} ảnh, OK={len(files) - n_ng}, NG={n_ng}")


if __name__ == "__main__":
    main()
