"""Giao diện kiểm tra AOI (Streamlit).

Chạy:  streamlit run app.py

Nguồn ảnh:
- Tải ảnh lên (một hoặc nhiều ảnh) thay cho camera công nghiệp.
- Webcam của laptop (tùy chọn).
- Ảnh mẫu từ tập holdout của MVTec AD để demo nhanh.
"""
import csv
import hashlib
import random
from datetime import datetime

import pandas as pd
import streamlit as st
from PIL import Image

from aoi import config as C
from aoi.data import split_test
from aoi.inspector import Inspector, available_categories

LOG_FILE = C.LOGS_DIR / "inspection_log.csv"
NG_DIR = C.LOGS_DIR / "ng_images"

st.set_page_config(page_title="AOI - Kiểm tra ngoại quan", page_icon="🔍", layout="wide")


@st.cache_resource(show_spinner="Đang nạp mô hình...")
def get_inspector(cat: str) -> Inspector:
    return Inspector(cat)


def write_log(cat: str, name: str, res, img: Image.Image):
    C.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    now = datetime.now()
    saved = ""
    if res.verdict == "NG":
        NG_DIR.mkdir(parents=True, exist_ok=True)
        saved = NG_DIR / f"{now:%Y%m%d_%H%M%S_%f}_{cat}.png"
        img.save(saved)
    new = not LOG_FILE.exists()
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "product", "image", "verdict", "score", "threshold", "defect", "time_ms", "saved_ng"])
        w.writerow([now.isoformat(timespec="seconds"), cat, name, res.verdict, f"{res.score:.4f}",
                    f"{res.threshold:.4f}", res.defect or "", f"{res.time_ms:.1f}", saved])


def show_result(name: str, res):
    color = "#16a34a" if res.verdict == "OK" else "#dc2626"
    label = "OK - ĐẠT" if res.verdict == "OK" else "NG - LỖI"
    st.markdown(
        f"<div style='background:{color};color:white;padding:10px 16px;border-radius:8px;"
        f"font-size:26px;font-weight:700'>{label}"
        f"<span style='font-size:15px;font-weight:400;margin-left:16px'>{name}</span></div>",
        unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Điểm bất thường", f"{res.score:.3f}")
    c2.metric("Ngưỡng", f"{res.threshold:.3f}")
    c3.metric("Loại lỗi", res.defect or "—")
    c4.metric("Thời gian xử lý", f"{res.time_ms:.0f} ms")
    i1, i2, i3 = st.columns(3)
    i1.image(res.image, caption="Ảnh sau tiền xử lý", width="stretch")
    i2.image(res.heatmap, caption="Bản đồ nhiệt vùng bất thường", width="stretch")
    i3.image(res.boxes, caption="Khoanh vùng lỗi", width="stretch")
    if res.defect_probs:
        probs = pd.DataFrame({"Loại lỗi": list(res.defect_probs), "Xác suất": list(res.defect_probs.values())})
        st.bar_chart(probs, x="Loại lỗi", y="Xác suất", height=200)
    st.divider()


def main():
    st.title("🔍 Hệ thống Vision AI kiểm tra lỗi ngoại quan (AOI)")
    cats = available_categories()
    if not cats:
        st.error("Chưa có mô hình. Chạy `python scripts/download_mvtec.py` rồi `python scripts/train.py` trước.")
        return

    with st.sidebar:
        st.header("Cấu hình")
        cat = st.selectbox("Sản phẩm", cats, format_func=lambda c: f"{C.CATEGORY_VI.get(c, c)} ({c})")
        source = st.radio("Nguồn ảnh", ["Tải ảnh lên", "Ảnh mẫu (bộ dữ liệu)", "Webcam laptop"])
        save_log = st.checkbox("Lưu nhật ký và ảnh NG", value=True)
        st.caption("Luồng xử lý: Ảnh → Tiền xử lý → PatchCore (OK/NG) → ResNet18 (loại lỗi) → Kết quả")

    insp = get_inspector(cat)
    items: list[tuple[str, Image.Image]] = []

    if source == "Tải ảnh lên":
        files = st.file_uploader("Chọn ảnh sản phẩm", type=["png", "jpg", "jpeg", "bmp"], accept_multiple_files=True)
        items = [(f.name, Image.open(f).convert("RGB")) for f in files or []]
    elif source == "Ảnh mẫu (bộ dữ liệu)":
        if not (C.DATA_DIR / cat).exists():
            st.warning("Không tìm thấy dữ liệu MVTec AD trên máy.")
        else:
            n = st.slider("Số ảnh ngẫu nhiên", 1, 12, 4)
            if st.button("Lấy ảnh ngẫu nhiên") or "samples" not in st.session_state \
                    or st.session_state.get("samples_cat") != cat:
                _, hold = split_test(cat)
                st.session_state.samples = random.sample(hold, min(n, len(hold)))
                st.session_state.samples_cat = cat
            items = [(f"{s.defect}/{s.path.name}", Image.open(s.path).convert("RGB"))
                     for s in st.session_state.samples]
    else:
        shot = st.camera_input("Chụp ảnh sản phẩm")
        if shot:
            items = [("webcam.png", Image.open(shot).convert("RGB"))]

    if not items:
        st.info("Chọn hoặc tải ảnh lên để bắt đầu kiểm tra.")
        return

    # Streamlit chạy lại toàn bộ script sau mỗi thao tác: giữ kết quả theo nội dung ảnh
    # để không kiểm tra và ghi nhật ký lặp lại cùng một ảnh.
    done = st.session_state.setdefault("done", {})
    results = []
    for name, img in items:
        key = (cat, name, hashlib.md5(img.tobytes()).hexdigest())
        if key not in done:
            done[key] = insp.inspect(img)
            if save_log:
                write_log(cat, name, done[key], img)
        results.append((name, done[key]))

    n_ng = sum(r.verdict == "NG" for _, r in results)
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Tổng số ảnh", len(results))
    s2.metric("OK", len(results) - n_ng)
    s3.metric("NG", n_ng)
    s4.metric("Tỷ lệ đạt", f"{(len(results) - n_ng) / len(results):.0%}")
    if len(results) > 1:
        st.dataframe(pd.DataFrame([{
            "Ảnh": n, "Kết quả": r.verdict, "Điểm": round(r.score, 3), "Loại lỗi": r.defect or "",
            "Thời gian (ms)": round(r.time_ms)} for n, r in results]), width="stretch", hide_index=True)
    st.divider()
    for name, res in results:
        show_result(name, res)


main()
