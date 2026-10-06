"""Tải bộ dữ liệu MVTec AD (bản sao trên HuggingFace: foersben/mvtec-ad).

Dữ liệu đã có sẵn nhãn: ảnh OK/NG, tên loại lỗi (theo thư mục) và mask vùng lỗi.
Giấy phép: CC BY-NC-SA 4.0 (chỉ dùng cho mục đích phi thương mại / học tập).

Dùng:
    python scripts/download_mvtec.py                       # 5 sản phẩm mặc định
    python scripts/download_mvtec.py --categories bottle screw
    python scripts/download_mvtec.py --categories all
"""
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from aoi import config as C  # noqa: E402

REPO = "foersben/mvtec-ad"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main/{{path}}?recursive=true"
FILE = f"https://huggingface.co/datasets/{REPO}/resolve/main/{{path}}"


def list_files(category: str) -> list[dict]:
    r = requests.get(API.format(path=category), timeout=60)
    r.raise_for_status()
    return [f for f in r.json() if f["type"] == "file"]


def fetch(f: dict) -> None:
    dst = C.DATA_DIR / f["path"]
    if dst.exists() and dst.stat().st_size == f["size"]:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(3):
        try:
            r = requests.get(FILE.format(path=f["path"]), timeout=120)
            r.raise_for_status()
            dst.write_bytes(r.content)
            return
        except requests.RequestException:
            if attempt == 2:
                raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="+", default=C.DEFAULT_CATEGORIES)
    args = ap.parse_args()
    cats = C.ALL_CATEGORIES if args.categories == ["all"] else args.categories
    for cat in cats:
        files = list_files(cat)
        with ThreadPoolExecutor(16) as ex:
            list(tqdm(ex.map(fetch, files), total=len(files), desc=cat))
    print(f"Xong. Dữ liệu ở {C.DATA_DIR}")


if __name__ == "__main__":
    main()
