import logging
import os
from minio import Minio


def down_folder(client: Minio, bucket_name: str, object_prefix: str, save_folder: str):
    """从 minio 中下载某对象, client: minio 客户端, bucket_name: 对象所处的存储桶名称, object_prefix: 对象去除存储桶之后的前缀, save_folder: 对象保存到本地的位置"""
    objs = client.list_objects(bucket_name, object_prefix)

    for obj in objs:
        obj_name = str(obj.object_name)
        save_path = os.path.join(save_folder, obj_name)
        if obj_name[-1] == "/":
            down_folder(client, bucket_name, obj_name, save_folder)
        else:
            client.fget_object(bucket_name, obj_name, save_path)


def down_object(minio_client: Minio, minio_object_path: str, save_path: str) -> str:
    """从 minio 中下载文件或者文件夹，并返回文件下载的绝对路径"""
    bucket_name = minio_object_path.split("/")[0]
    if bucket_name == "":
        raise Exception(f"给定的对象文件的 minio 地址：{minio_object_path} 有误，原因：存储桶为空。")
    prefix = minio_object_path.replace(f"{bucket_name}/", "", 1)

    try:
        logging.info(f"准备从 minio 中下载对象文件, 对象在 minio 中的位置是: {minio_object_path}")
        if minio_object_path[-1] == "/":
            down_folder(minio_client, bucket_name, prefix, save_path)
            p = minio_object_path[len(bucket_name) + 1:]
            result_path = os.path.join(save_path, p)
        else:
            result_path = os.path.join(save_path, os.path.basename(minio_object_path))
            minio_client.fget_object(bucket_name, prefix, result_path)
        logging.info(f"对象文件下载成功, 保存到本地的位置是: {os.path.abspath(result_path)}")
    except Exception as err:
        logging.error(f"从 minio 中下载对象文件失败，原因是：{err}")

    result_path = os.path.abspath(result_path)
    return result_path


# todo 1. 判断存储桶是否存在，不存在就新建，2. 上传文件
def upload_file_or_folder(minio_client: Minio, result_minio_path: str, ff_path: str):
    bucket_name = result_minio_path.split("/")[0]
    if bucket_name == "":
        Exception(f"给定的对象文件的 minio 地址：{result_minio_path} 有误，原因：存储桶为空。")
    
    # 判断存储桶是否存在，不存在的话就新建
    if not minio_client.bucket_exists(bucket_name):
        minio_client.make_bucket(bucket_name)
    
    if os.path.isfile(ff_path):
        
        object_name = os.path.join(result_minio_path[len(f"{bucket_name}/"):], os.path.basename(ff_path))
        minio_client.fput_object(bucket_name, object_name, ff_path)
        return
    
    for root, _, files in os.walk(ff_path):
        for f in files:
            f_abs_path = os.path.join(root, f)
            suffix = os.path.relpath(f_abs_path, os.path.abspath(ff_path))
            object_name = os.path.join(result_minio_path[len(f"{bucket_name}/"):], os.path.basename(ff_path), suffix)
            minio_client.fput_object(bucket_name, object_name, f_abs_path)

