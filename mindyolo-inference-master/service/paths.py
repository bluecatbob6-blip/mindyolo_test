"""内嵌 MindYOLO 源码根路径（与 yolov5-inference 内嵌 `service/yolov5` 对齐，不依赖仓库外目录）。"""
from pathlib import Path

_SERVICE_DIR = Path(__file__).resolve().parent
BUNDLED_MINDYOLO_ROOT = str(_SERVICE_DIR / "mindyolo")
