from fastapi import APIRouter, UploadFile, Form
from fastapi.responses import FileResponse
from typing import List
from myutils import file_utils
from config import config
from http_api.model.response import response as myresponse
import logging
import json
from service import InferenceOptions, MindYOLOInference

router = APIRouter(
    prefix="/api/v1"
)


def init_infer_class(model_path: str):
    config.global_config.model_path = model_path


async def get_upload_files(upload_files: List[UploadFile]):
    """返回上传的文件的信息"""
    files_data = dict()
    for file in upload_files:
        files_data.update({file.filename: file.file.read()})
        await file.close()
    return files_data


# 推理接口
@router.post("/inference")
async def ctrl_inference(upload_files: List[UploadFile] = None, structured_data: str = Form(default=None)):
    logging.info("开始推理")
    data_source = None
    if upload_files is None and structured_data == None:
        return myresponse.fail_with_msg("请提供待预测数据的表单字段。")
    elif upload_files is not None and structured_data != None:
        return myresponse.fail_with_msg("参数错误，暂不支持同时预测文件和结构化数据。")
    elif upload_files is not None:
        try:
            data_list = file_utils.save_front_upload_data(upload_files)  # 将前端上传的图片保存到本地文件中
            data_source = data_list[0]["data_path"]
        except Exception as e:
            logging.error("upload data failed", e)
            return myresponse.fail_with_msg(str(e))
    elif structured_data != None:
        try:
            _ = json.loads(structured_data)
        except Exception as e:
            return myresponse.fail_with_msg(f"解析 structured_data 字段失败，其值是: '{structured_data}'，请检查是否符合规范，详细原因：{e}")
        return myresponse.fail_with_msg("当前组件仅支持图片文件推理，暂不支持 structured_data。")
    
    try:
        logging.info(f"data_source is {data_source}")
        options = InferenceOptions(
            mindyolo_root=config.global_config.mindyolo_root,
            model_type=config.global_config.model_type,
            model_scale=config.global_config.model_scale,
            backend=config.global_config.backend,
            model_path=config.global_config.model_path,
            image_path=data_source,
            config=config.global_config.config_path,
            device_target=config.global_config.device_target,
            img_size=config.global_config.img_size,
            conf_thres=config.global_config.conf_thres,
            iou_thres=config.global_config.iou_thres,
            output_dir="./http_result",
        )
        result = MindYOLOInference(options).run()
        if upload_files is not None:
            return FileResponse(result[0])
    except Exception as e:
        logging.error(f"推理失败，原因：{e}")
        return myresponse.fail_with_msg(f"推理失败，原因：{e}")
    

# 用于服务状态检查的接口
@router.get("/ready")
async def ready():
    return "ok"


@router.get("/logs")
async def print_log():
    return FileResponse(config.global_config.log_path)

