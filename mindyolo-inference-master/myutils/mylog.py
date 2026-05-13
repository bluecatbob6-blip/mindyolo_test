from config import config
import logging
import os
from typing import Optional

_VAR_LOG_DEFAULT = "/var/log/logout.log"


def _pick_log_path(log_path: Optional[str]) -> str:
    """未指定或与默认 /var/log/logout.log 等价时：若 /var/log 不可写则落到当前工作目录 ./runs_batch/logout.log（便于 macOS 本地调试）。"""
    if log_path is None or not str(log_path).strip():
        candidate = _VAR_LOG_DEFAULT
    else:
        candidate = os.path.expanduser(str(log_path).strip())
    if os.path.abspath(candidate) != os.path.abspath(_VAR_LOG_DEFAULT):
        return candidate
    log_dir = os.path.dirname(candidate)
    if log_dir and os.path.isdir(log_dir) and os.access(log_dir, os.W_OK):
        return candidate
    return os.path.abspath(os.path.join(os.getcwd(), "runs_batch", "logout.log"))


def init_log(log_path: Optional[str] = None):
    # 对logger进行配置——日志等级&输出格式
    LOG_FORMAT = "%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s"
    log_path = _pick_log_path(log_path)
    log_dir = os.path.dirname(log_path)
    if log_dir and not os.path.isdir(log_dir):
        try:
            os.makedirs(log_dir, exist_ok=True)
        except OSError as e:
            raise OSError(
                f"无法创建日志目录「{os.path.abspath(log_dir)}」: {e}\n"
                "提示：以「/」开头的路径表示系统根目录（如 /runs_local 会在磁盘根下建目录，macOS 上常不可写）。"
                "若日志放在当前工程下，请用相对路径，例如 ./runs_local/workflow_test1.log"
            ) from e
    log_path = os.path.abspath(log_path)
    config.log_path = log_path

    logging.basicConfig(level=logging.INFO,
                        format=LOG_FORMAT,
                        datefmt='%m-%d %H:%M',
                        filename=log_path,
                        filemode='w')

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter(LOG_FORMAT)
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)