import argparse
import logging
import os
import sys

from config import config
from myutils import mylog
from myutils import myminio
from myutils import strconv
from service import BUNDLED_MINDYOLO_ROOT, InferenceOptions, inference


def flag_parse(first: bool):
    """处理可选项参数：首次 first=True 完整解析；之后 first=False 仅解析已知参数，与 yolov5-inference workflow_main 一致。"""
    parser = argparse.ArgumentParser(description="处理可选项参数")
    parser.add_argument("--model-path", dest="model_path", required=True, help="模型文件的绝对路径")
    parser.add_argument("--data-path", dest="data_path", required=True, help="待检测数据的绝对路径")
    parser.add_argument("--log-path", dest="log_path", required=True, help="日志文件的绝对路径")
    parser.add_argument("--result-minio-path", dest="result_minio_path", required=False, help="结果保存的 minio 位置")
    parser.add_argument("--endpoint", dest="endpoint", required=False, help="minio 的服务地址，如: ip:port")
    parser.add_argument("--access-key", dest="access_key", required=False, help="minio 的访问用户名")
    parser.add_argument("--secret-key", dest="secret_key", required=False, help="minio 的访问密码")
    parser.add_argument("--secure", dest="secure", required=False, help="minio 服务是不是 https 加密服务")
    # MindYOLO 扩展（默认值与通用推理入口一致）
    parser.add_argument("--model-type", dest="model_type", default="yolov5", choices=["yolov5", "yolov10"])
    parser.add_argument("--model-scale", dest="model_scale", default="s")
    parser.add_argument("--backend", dest="backend", default="ckpt", choices=["ckpt", "mindir"])
    parser.add_argument("--config", dest="config_path", default="")
    parser.add_argument("--mindyolo-root", dest="mindyolo_root", default=BUNDLED_MINDYOLO_ROOT)
    parser.add_argument("--device-target", dest="device_target", default="Ascend")
    parser.add_argument("--img-size", dest="img_size", type=int, default=640)
    parser.add_argument("--conf-thres", dest="conf_thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", dest="iou_thres", type=float, default=0.65)
    parser.add_argument("--output-dir", dest="output_dir", default="./runs_infer", help="推理结果输出目录")

    if not first:
        return parser.parse_known_args()[0]
    return parser.parse_args()


def merge_minio_env(args: argparse.Namespace) -> None:
    """命令行未给出时，从环境变量读取 MinIO 配置（命令行优先）。支持 MINIO_ENDPOINT / MINIO_ACCESS_KEY / MINIO_SECRET_KEY / MINIO_SECURE；结果路径可由 MINIO_RESULT_MINIO_PATH 或 MINIO_BUCKET[+MINIO_RESULT_PREFIX] 组成。"""
    def env(key: str, default: str = "") -> str:
        v = os.environ.get(key)
        return (v if v is not None else default).strip()

    if not (getattr(args, "endpoint", None) or "").strip() and env("MINIO_ENDPOINT"):
        args.endpoint = env("MINIO_ENDPOINT")
    if not (getattr(args, "access_key", None) or "").strip() and env("MINIO_ACCESS_KEY"):
        args.access_key = env("MINIO_ACCESS_KEY")
    if not (getattr(args, "secret_key", None) or "").strip() and env("MINIO_SECRET_KEY"):
        args.secret_key = env("MINIO_SECRET_KEY")
    sec = getattr(args, "secure", None)
    if sec is None or (isinstance(sec, str) and not str(sec).strip()):
        if env("MINIO_SECURE"):
            args.secure = env("MINIO_SECURE")
    if not (getattr(args, "result_minio_path", None) or "").strip():
        if env("MINIO_RESULT_MINIO_PATH"):
            args.result_minio_path = env("MINIO_RESULT_MINIO_PATH")
        elif env("MINIO_BUCKET"):
            bucket = env("MINIO_BUCKET").strip("/")
            prefix = env("MINIO_RESULT_PREFIX", "infer-results").strip("/")
            args.result_minio_path = f"{bucket}/{prefix}/" if prefix else f"{bucket}/"


def _minio_ready(args) -> bool:
    p = (args.result_minio_path or "").strip()
    if not p:
        return False
    if not (args.endpoint or "").strip() or not (args.access_key or "").strip() or not (args.secret_key or "").strip():
        return False
    return True


def init():
    """初始化日志；若同时提供 --result-minio-path 与 MinIO 四参数则初始化客户端以便推理后上传。"""
    args = flag_parse(first=True)
    merge_minio_env(args)
    if _minio_ready(args):
        sec_raw = args.secure if args.secure is not None else ""
        secure = strconv.str2bool(str(sec_raw).strip()) if str(sec_raw).strip() else False
        config.init_global_config(
            endpoint=args.endpoint.strip(),
            access_key=args.access_key.strip(),
            secret_key=args.secret_key.strip(),
            secure=secure,
        )
    config.global_config.model_path = args.model_path
    config.global_config.model_type = args.model_type
    config.global_config.model_scale = args.model_scale
    config.global_config.backend = args.backend
    config.global_config.config_path = args.config_path
    config.global_config.mindyolo_root = args.mindyolo_root
    config.global_config.device_target = args.device_target
    config.global_config.img_size = args.img_size
    config.global_config.conf_thres = args.conf_thres
    config.global_config.iou_thres = args.iou_thres
    mylog.init_log(args.log_path)


def data_inference(model_path: str, data_path: str, result_minio_path: str, output_dir: str):
    options = InferenceOptions(
        mindyolo_root=config.global_config.mindyolo_root,
        model_type=config.global_config.model_type,
        model_scale=config.global_config.model_scale,
        backend=config.global_config.backend,
        model_path=model_path,
        image_path="",
        config=config.global_config.config_path,
        device_target=config.global_config.device_target,
        img_size=config.global_config.img_size,
        conf_thres=config.global_config.conf_thres,
        iou_thres=config.global_config.iou_thres,
        output_dir=output_dir,
        extra_args=None,
    )
    logging.info(f"准备推理位于目录：{data_path} 中的数据")
    result_path = inference(model_path, options).run(data_path)
    logging.info("预测完毕")
    logging.info(f"预测结果保存在：{result_path}")

    has_minio = getattr(config.global_config, "minio_client", None) is not None and (result_minio_path or "").strip()
    if has_minio:
        logging.info("准备上传预测文件到 MinIO")
        if isinstance(result_path, str):
            try:
                myminio.upload_file_or_folder(config.global_config.minio_client, result_minio_path, result_path)
            except Exception as e:
                logging.error(f"上传预测文件：{result_path} 失败, 原因：{e}")
        else:
            for p in result_path:
                try:
                    myminio.upload_file_or_folder(config.global_config.minio_client, result_minio_path, p)
                except Exception as e:
                    logging.error(f"上传预测文件：{p} 失败，原因：{e}")
        logging.info("预测文件上传流程结束")


if __name__ == "__main__":
    init()
    args = flag_parse(first=False)
    merge_minio_env(args)
    try:
        data_inference(config.global_config.model_path, args.data_path, args.result_minio_path, args.output_dir)
    except Exception as e:
        logging.error(f"预测出错，错误原因{e}")
        sys.exit(-1)
