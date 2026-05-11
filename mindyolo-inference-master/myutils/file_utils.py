import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import shutil

def save_front_upload_data(files):
    """
    前端上传图片保存到本地
    """
    save_data_dir = "./data/"
    if not os.path.exists(save_data_dir):
        os.mkdir(save_data_dir)
    data_list = list()
    for file in files:
        try:
            suffix = Path(file.filename).suffix
            with NamedTemporaryFile(delete=False, suffix=suffix, dir=save_data_dir) as tmp:
                shutil.copyfileobj(file.file, tmp)
                tmp_file_name = Path(tmp.name).name
            data = {"data_name": file.filename, "data_path": save_data_dir + tmp_file_name}
            data_list.append(data)
        finally:
            file.file.close
    return data_list
