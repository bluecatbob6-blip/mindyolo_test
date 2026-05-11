from minio import Minio

class Config():
    minio_client: Minio
    model_path: str
    log_path: str
    model_type: str
    model_scale: str
    backend: str
    config_path: str
    mindyolo_root: str
    device_target: str
    img_size: int
    conf_thres: float
    iou_thres: float


global_config = Config()


def init_global_config(endpoint: str, access_key: str, secret_key: str, secure: bool): 
    global_config.minio_client = Minio(
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=secure
    )

