"""Hệ thống Vision AI kiểm tra lỗi ngoại quan sản phẩm (AOI)."""
import sys

# Console Windows mặc định cp1252, không in được tiếng Việt
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8")
