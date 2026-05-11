import argparse
import logging
import os
import sys

from config import config
from myutils import mylog
from myutils import myminio
from myutils import strconv
from service import InferenceOptions, MindYOLOInference


def flag_parse(first: bool):
    # 与 yolov5-inference batch_main 参数风格一致
    parser = argparse.ArgumentParser(description="处理识别图片相关的可选项参数")
    parser.add_argument("--model-minio-path", dest="model_minio_path", required=True, help="模型文件在 minio 中的位置", type=str)
    parser.add_argument("--data-minio-path", dest="data_minio_path", required=True, help="待预测的文件在 minio 中的位置", type=str)
    parser.add_argument("--result-minio-path", dest="result_minio_path", required=True, help="数据预测完之后的结果文件存放到 minio 中的位置", type=str)
    parser.add_argument("--model-type", dest="model_type", default="yolov5", choices=["yolov5", "yolov10"])
    parser.add_argument("--model-scale", dest="model_scale", default="s")
    parser.add_argument("--backend", dest="backend", default="ckpt", choices=["ckpt", "mindir"])
    parser.add_argument("--config", dest="config_path", default="")
    parser.add_argument("--mindyolo-root", dest="mindyolo_root", default="../mindyolo-master")
    parser.add_argument("--device-target", dest="device_target", default="Ascend")
    parser.add_argument("--img-size", dest="img_size", type=int, default=640)
    parser.add_argument("--conf-thres", dest="conf_thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", dest="iou_thres", type=float, default=0.65)
    parser.add_argument("--endpoint", dest="endpoint", required=True, help="minio 的服务地址，如: ip:port")
    parser.add_argument("--access-key", dest="access_key", required=True, help="minio 的访问用户名")
    parser.add_argument("--secret-key", dest="secret_key", required=True, help="minio 的访问密码")
    parser.add_argument("--secure", dest="secure", required=True, help="minio 服务是不是 https 加密服务")

    if not first:
        return parser.parse_known_args()[0]
    return parser.parse_args()


def init():
    """程序初始化（与 yolov5-inference batch_main 一致：先初始化 MinIO 与日志，再下载模型）。"""
    args = flag_parse(first=True)
    secure = strconv.str2bool(args.secure)
    config.init_global_config(endpoint=args.endpoint, access_key=args.access_key, secret_key=args.secret_key, secure=secure)
    mylog.init_log()
    config.global_config.model_type = args.model_type
    config.global_config.model_scale = args.model_scale
    config.global_config.backend = args.backend
    config.global_config.config_path = args.config_path
    config.global_config.mindyolo_root = args.mindyolo_root
    config.global_config.device_target = args.device_target
    config.global_config.img_size = args.img_size
    config.global_config.conf_thres = args.conf_thres
    config.global_config.iou_thres = args.iou_thres

    save_path = "./models"
    try:
        model_path = myminio.down_object(config.global_config.minio_client, args.model_minio_path, save_path)
        config.global_config.model_path = model_path
    except Exception as err:
        logging.error(err)
        sys.exit(-1)


def load_path(path: str, file_type_unfilter_list: list) -> list:
    """与 yolov5-inference batch_main 一致：文件返回单元素列表，目录返回一级子文件中后缀在列表内的路径。"""
    result = list()
    if os.path.isfile(path):
        result.append(os.path.abspath(path))
        return result

    p = os.listdir(path)
    for f in p:
        if os.path.splitext(f)[-1][1:].lower() in file_type_unfilter_list:
            result.append(os.path.join(path, f))

    return result


def batch_inference(model_path: str, data_minio_path: str, result_minio_path: str):
    save_data_parent_dir = "./data"
    if not os.path.exists(save_data_parent_dir):
        os.mkdir(save_data_parent_dir)

    save_data_path = myminio.down_object(config.global_config.minio_client, data_minio_path, save_data_parent_dir)

    output_dir = os.path.abspath("./result")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    logging.info(f"准备预测位于 minio 路径: {data_minio_path} 下的文件")
    options = InferenceOptions(
        mindyolo_root=config.global_config.mindyolo_root,
        model_type=config.global_config.model_type,
        model_scale=config.global_config.model_scale,
        backend=config.global_config.backend,
        model_path=model_path,
        image_path=save_data_path,
        config=config.global_config.config_path,
        device_target=config.global_config.device_target,
        img_size=config.global_config.img_size,
        conf_thres=config.global_config.conf_thres,
        iou_thres=config.global_config.iou_thres,
        output_dir=output_dir,
    )
    result_path = MindYOLOInference(options).run()
    logging.info("预测完毕")

    logging.info("准备上传预测文件")
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
    logging.info("预测文件上传成功")

if __name__ == '__main__':
    init()
    args = flag_parse(first=False)
    try:
        batch_inference(config.global_config.model_path, args.data_minio_path, args.result_minio_path)
    except Exception as e:
        logging.error(f"请检查给定的待识别对象 minio 路径: {args.data_minio_path} 是否正确，错误原因：{e}")
        sys.exit(-1)

