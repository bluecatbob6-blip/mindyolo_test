import argparse
import logging
import sys
from fastapi import FastAPI
import uvicorn
from http_api.router.inference import router
from config import config
from myutils.mylog import init_log
from myutils import myminio
from myutils import strconv
from http_api.router import inference
from service import BUNDLED_MINDYOLO_ROOT

# fastapi定义路由
app = FastAPI()
app.include_router(router)


def flag_parse(first: bool):
    # 与 yolov5-inference http_main MinIO 参数一致；其余为 MindYOLO 扩展
    parser = argparse.ArgumentParser(description="处理识别图片相关的可选项参数")
    parser.add_argument("--model-minio-path", dest="model_minio_path", required=True, help="模型文件在 minio 中的位置", type=str)
    parser.add_argument("--model-type", dest="model_type", default="yolov5", choices=["yolov5", "yolov10"])
    parser.add_argument("--model-scale", dest="model_scale", default="s")
    parser.add_argument("--backend", dest="backend", default="ckpt", choices=["ckpt", "mindir"])
    parser.add_argument("--config", dest="config_path", default="")
    parser.add_argument("--mindyolo-root", dest="mindyolo_root", default=BUNDLED_MINDYOLO_ROOT)
    parser.add_argument("--device-target", dest="device_target", default="Ascend")
    parser.add_argument("--img-size", dest="img_size", type=int, default=640)
    parser.add_argument("--conf-thres", dest="conf_thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", dest="iou_thres", type=float, default=0.65)

    # minio
    parser.add_argument("--endpoint", dest="endpoint", required=True, help="minio 的服务地址，如: ip:port")
    parser.add_argument("--access-key", dest="access_key", required=True, help="minio 的访问用户名")
    parser.add_argument("--secret-key", dest="secret_key", required=True, help="minio 的访问密码")
    parser.add_argument("--secure", dest="secure", required=True, help="minio 服务是不是 https 加密服务")

    if not first:
        return parser.parse_known_args()[0]
    else:
        return parser.parse_args()


def run_web_server():
    uvicorn.run(app, host="0.0.0.0", port=8000)


def init():
    args = flag_parse(first=True)
    secure = strconv.str2bool(args.secure)
    config.init_global_config(endpoint=args.endpoint, access_key=args.access_key, secret_key=args.secret_key, secure=secure)
    
    log_path = "/var/log/logout.log"
    # init log（无写权限时 mylog 会落到 ./runs_batch/logout.log，避免 macOS 本地 PermissionError）
    init_log(log_path)
    config.global_config.log_path = log_path
    config.global_config.model_type = args.model_type
    config.global_config.model_scale = args.model_scale
    config.global_config.backend = args.backend
    config.global_config.config_path = args.config_path
    config.global_config.mindyolo_root = args.mindyolo_root
    config.global_config.device_target = args.device_target
    config.global_config.img_size = args.img_size
    config.global_config.conf_thres = args.conf_thres
    config.global_config.iou_thres = args.iou_thres

    # 下载模型
    save_path = "./models"
    try:
        model_path = myminio.down_object(config.global_config.minio_client, args.model_minio_path, save_path)
        config.global_config.model_path = model_path
    except Exception as err:
        logging.error(err)
        sys.exit(-1)
    inference.init_infer_class(config.global_config.model_path)

        
if __name__ == '__main__':
    init()
    run_web_server()
    