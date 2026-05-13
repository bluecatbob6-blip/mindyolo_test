import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional


MODEL_CONFIG_MAP = {
    "yolov5": {
        "n": "configs/yolov5/yolov5n.yaml",
        "s": "configs/yolov5/yolov5s.yaml",
        "m": "configs/yolov5/yolov5m.yaml",
        "l": "configs/yolov5/yolov5l.yaml",
        "x": "configs/yolov5/yolov5x.yaml",
    },
    "yolov10": {
        "n": "configs/yolov10/yolov10n.yaml",
        "s": "configs/yolov10/yolov10s.yaml",
        "m": "configs/yolov10/yolov10m.yaml",
        "b": "configs/yolov10/yolov10b.yaml",
        "l": "configs/yolov10/yolov10l.yaml",
        "x": "configs/yolov10/yolov10x.yaml",
    },
}

IMAGE_SUFFIX = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_SUFFIX = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".m4v"}


@dataclass
class InferenceOptions:
    mindyolo_root: str
    model_type: str
    model_scale: str
    backend: str
    model_path: str
    image_path: str
    config: str
    device_target: str
    img_size: int
    conf_thres: float
    iou_thres: float
    output_dir: str
    extra_args: Optional[List[str]] = None


class inference:
    """平台《算子开发规范》推理类：`model_path`、`load_model`、`run(source)`；子进程调用内嵌或自定义 `mindyolo_root` 下的 `demo/predict.py` / `deploy/mslite_predict.py`。"""

    def __init__(self, model_path: str, options: InferenceOptions):
        self.model_path = os.path.abspath(model_path)
        self._base_options = replace(
            options,
            model_path=self.model_path,
            image_path="",
        )
        self.mindyolo_root = os.path.abspath(self._base_options.mindyolo_root)
        self.model = self.load_model()

    def load_model(self) -> str:
        """校验模型路径存在；权重由子进程加载，`self.model` 为规范化路径字符串。"""
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"未找到模型文件: {self.model_path}")
        return self.model_path

    def run(self, source: str) -> List[str]:
        self.options = replace(self._base_options, image_path=source)
        return self._execute()

    def _resolve_config(self) -> str:
        if self.options.config:
            config_path = os.path.abspath(self.options.config)
        else:
            config_rel = MODEL_CONFIG_MAP[self.options.model_type][self.options.model_scale]
            config_path = os.path.join(self.mindyolo_root, config_rel)
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"未找到配置文件: {config_path}")
        return config_path

    def _build_ckpt_cmd(self, config_path: str) -> list[str]:
        script = os.path.join(self.mindyolo_root, "demo", "predict.py")
        if not os.path.isfile(script):
            raise FileNotFoundError(f"未找到脚本: {script}")

        cmd = [
            sys.executable,
            script,
            "--config",
            config_path,
            "--weight",
            os.path.abspath(self.options.model_path),
            "--image_path",
            os.path.abspath(self.options.image_path),
            "--device_target",
            self.options.device_target,
            "--img_size",
            str(self.options.img_size),
            "--conf_thres",
            str(self.options.conf_thres),
            "--iou_thres",
            str(self.options.iou_thres),
            "--save_dir",
            os.path.abspath(self.options.output_dir),
        ]
        return cmd

    def _build_mindir_cmd(self, config_path: str) -> list[str]:
        script = os.path.join(self.mindyolo_root, "deploy", "mslite_predict.py")
        if not os.path.isfile(script):
            raise FileNotFoundError(f"未找到脚本: {script}")

        cmd = [
            sys.executable,
            script,
            "--config",
            config_path,
            "--mindir_path",
            os.path.abspath(self.options.model_path),
            "--image_path",
            os.path.abspath(self.options.image_path),
            "--img_size",
            str(self.options.img_size),
            "--conf_thres",
            str(self.options.conf_thres),
            "--iou_thres",
            str(self.options.iou_thres),
            "--result_folder",
            os.path.abspath(self.options.output_dir),
        ]
        return cmd

    def _find_latest_result(self, output_dir: Optional[str] = None) -> str:
        root = Path(output_dir or self.options.output_dir)
        if not root.exists():
            return str(root)
        files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIX]
        if not files:
            return str(root)
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return str(files[0].resolve())

    def _run_single_image(self, image_path: str, output_dir: Optional[str] = None) -> str:
        original_image_path = self.options.image_path
        original_output_dir = self.options.output_dir
        try:
            if output_dir:
                self.options.output_dir = output_dir
            self.options.image_path = image_path
            os.makedirs(self.options.output_dir, exist_ok=True)
            if not os.path.isfile(self.options.image_path):
                raise FileNotFoundError(f"未找到图片: {self.options.image_path}")
            if not os.path.isfile(self.options.model_path):
                raise FileNotFoundError(f"未找到模型文件: {self.options.model_path}")
            if self.options.model_type not in MODEL_CONFIG_MAP:
                raise ValueError(f"不支持的 model_type: {self.options.model_type}")
            if self.options.model_scale not in MODEL_CONFIG_MAP[self.options.model_type]:
                valid = ",".join(MODEL_CONFIG_MAP[self.options.model_type].keys())
                raise ValueError(f"{self.options.model_type} 不支持 model_scale={self.options.model_scale}，可选: {valid}")

            config_path = self._resolve_config()
            if self.options.backend == "ckpt":
                cmd = self._build_ckpt_cmd(config_path)
            elif self.options.backend == "mindir":
                cmd = self._build_mindir_cmd(config_path)
            else:
                raise ValueError(f"不支持的 backend: {self.options.backend}")

            if self.options.extra_args:
                cmd.extend(self.options.extra_args)

            print("执行命令:", " ".join(cmd))
            subprocess.run(cmd, check=True, cwd=self.mindyolo_root)
            return self._find_latest_result(self.options.output_dir)
        finally:
            self.options.image_path = original_image_path
            self.options.output_dir = original_output_dir

    def _is_url_file(self, source: str) -> bool:
        lower = source.lower()
        url_path = lower.split("?", 1)[0]
        return lower.startswith(("http://", "https://")) and Path(url_path).suffix.lower() in IMAGE_SUFFIX.union(VIDEO_SUFFIX)

    def _download_url_file(self, source: str) -> str:
        suffix = Path(source.split("?", 1)[0]).suffix or ".jpg"
        fd, local_path = tempfile.mkstemp(prefix="mindyolo_url_", suffix=suffix)
        os.close(fd)
        urllib.request.urlretrieve(source, local_path)
        return local_path

    def _list_source_images(self, source: str) -> List[str]:
        if os.path.isfile(source):
            if Path(source).suffix.lower() in IMAGE_SUFFIX:
                return [os.path.abspath(source)]
            raise ValueError(f"不是支持的图片文件: {source}")
        if os.path.isdir(source):
            images: List[str] = []
            for root, _, files in os.walk(source):
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    if Path(file_path).suffix.lower() in IMAGE_SUFFIX:
                        images.append(os.path.abspath(file_path))
            if not images:
                raise ValueError(f"目录中未找到可推理图片: {source}")
            return sorted(images)
        raise FileNotFoundError(f"未找到待推理数据: {source}")

    def _is_video_source(self, source: str) -> bool:
        if source.isnumeric():
            return True
        lower = source.lower()
        if lower.startswith(("rtsp://", "rtmp://")):
            return True
        url_path = lower.split("?", 1)[0]
        if lower.startswith(("http://", "https://")) and Path(url_path).suffix.lower() not in IMAGE_SUFFIX:
            return True
        return os.path.isfile(source) and Path(source).suffix.lower() in VIDEO_SUFFIX

    def _run_video_source(self, source: str) -> List[str]:
        import cv2

        capture_source = int(source) if source.isnumeric() else source
        cap = cv2.VideoCapture(capture_source)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频/流: {source}")

        os.makedirs(self.options.output_dir, exist_ok=True)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        if fps <= 0 or fps > 120:
            fps = 25

        source_name = "stream" if source.isnumeric() or "://" in source else Path(source).stem
        output_path = os.path.abspath(os.path.join(self.options.output_dir, f"{source_name}_result.mp4"))
        writer = None
        frame_idx = 0
        try:
            with tempfile.TemporaryDirectory(prefix="mindyolo_frames_") as temp_dir:
                frame_dir = os.path.join(temp_dir, "frames")
                result_dir = os.path.join(temp_dir, "results")
                os.makedirs(frame_dir, exist_ok=True)
                os.makedirs(result_dir, exist_ok=True)

                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    frame_path = os.path.join(frame_dir, f"frame_{frame_idx:06d}.jpg")
                    cv2.imwrite(frame_path, frame)
                    result_image = self._run_single_image(frame_path, result_dir)
                    result_frame = cv2.imread(result_image)
                    if result_frame is None:
                        result_frame = frame
                    if writer is None:
                        height, width = result_frame.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    writer.write(result_frame)
                    frame_idx += 1
                    shutil.rmtree(result_dir, ignore_errors=True)
                    os.makedirs(result_dir, exist_ok=True)
        finally:
            cap.release()
            if writer is not None:
                writer.release()

        if frame_idx == 0:
            raise ValueError(f"视频/流没有读取到帧: {source}")
        return [output_path]

    def _execute(self) -> List[str]:
        source = self.options.image_path
        if not os.path.isfile(self.options.model_path):
            raise FileNotFoundError(f"未找到模型文件: {self.options.model_path}")
        temp_source = ""
        try:
            if self._is_url_file(source):
                temp_source = self._download_url_file(source)
                source = temp_source
            if self._is_video_source(source):
                return self._run_video_source(source)
            results = []
            for image_path in self._list_source_images(source):
                results.append(self._run_single_image(image_path))
            return results
        finally:
            if temp_source and os.path.exists(temp_source):
                os.remove(temp_source)
