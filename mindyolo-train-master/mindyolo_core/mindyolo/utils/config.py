import argparse
import collections
import os
from copy import deepcopy
import yaml

try:
    collectionsAbc = collections.abc
except AttributeError:
    collectionsAbc = collections

__all__ = ["parse_args", "merge_dataset_yaml_into_args"]


def parse_args(parser):
    parser_config = argparse.ArgumentParser(description="Config", add_help=False)
    parser_config.add_argument(
        "-c", "--config", type=str, default="", help="YAML config file specifying default arguments."
    )

    args_config, remaining = parser_config.parse_known_args()

    # Do we have a config file to parse?
    if args_config.config:
        cfg, _, _ = load_config(args_config.config)
        cfg = Config(cfg)
        parser.set_defaults(**cfg)
        parser.set_defaults(config=args_config.config)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    return Config(vars(args))


def load_config(file_path):
    BASE = "__BASE__"
    assert os.path.splitext(file_path)[-1] in [".yaml", ".yml"], f"[{file_path}] not yaml format."
    cfg_default, cfg_helper, cfg_choices = _parse_yaml(file_path)

    # NOTE: cfgs outside have higher priority than cfgs in _BASE_
    if BASE in cfg_default:
        all_base_cfg_default = {}
        all_base_cfg_helper = {}
        all_base_cfg_choices = {}
        base_yamls = list(cfg_default[BASE])
        for base_yaml in base_yamls:
            if base_yaml.startswith("~"):
                base_yaml = os.path.expanduser(base_yaml)
            if not base_yaml.startswith("/"):
                base_yaml = os.path.join(os.path.dirname(file_path), base_yaml)

            base_cfg_default, base_cfg_helper, base_cfg_choices = load_config(base_yaml)
            all_base_cfg_default = _merge_config(base_cfg_default, all_base_cfg_default)
            all_base_cfg_helper = _merge_config(base_cfg_helper, all_base_cfg_helper)
            all_base_cfg_choices = _merge_config(base_cfg_choices, all_base_cfg_choices)

        del cfg_default[BASE]
        return (
            _merge_config(cfg_default, all_base_cfg_default),
            _merge_config(cfg_helper, all_base_cfg_helper),
            _merge_config(cfg_choices, all_base_cfg_choices),
        )

    return cfg_default, cfg_helper, cfg_choices


def _parse_yaml(yaml_path):
    """
    Parse the yaml config file.

    Args:
        yaml_path: Path to the yaml config.
    """
    with open(yaml_path, "r") as fin:
        try:
            cfgs = yaml.load_all(fin.read(), Loader=yaml.FullLoader)
            cfgs = [x for x in cfgs]
            if len(cfgs) == 1:
                cfg = cfgs[0]
                cfg_helper = {}
                cfg_choices = {}
            elif len(cfgs) == 2:
                cfg, cfg_helper = cfgs
                cfg_choices = {}
            elif len(cfgs) == 3:
                cfg, cfg_helper, cfg_choices = cfgs
            else:
                raise ValueError("At most 3 docs (config, description for help, choices) are supported in config yaml")
        except:
            raise ValueError("Failed to parse yaml")
    return cfg, cfg_helper, cfg_choices


def _merge_config(config, base):
    """Merge config"""
    new = deepcopy(base)
    for k, v in config.items():
        if k in new and isinstance(new[k], dict) and isinstance(config[k], collectionsAbc.Mapping):
            new[k] = _merge_config(config[k], new[k])
        else:
            new[k] = config[k]
    return new


def _yolov5_style_to_mindyolo_data(doc: dict, yaml_dir: str) -> dict:
    """Ultralytics data.yaml（path/train/val/names）转为 MindYOLO args.data 常用字段。
    path / train / val 相对路径均相对于 data yaml 文件所在目录解析（与 YOLOv5 一致）。"""
    if "train" not in doc:
        return {}

    root_raw = str(doc.get("path", "") or ".").strip()
    if root_raw and root_raw != ".":
        root_abs = os.path.normpath(os.path.join(yaml_dir, root_raw))
    else:
        root_abs = yaml_dir

    def joinp(p):
        p = str(p).strip() if p is not None else ""
        if not p:
            return ""
        if os.path.isabs(p):
            return os.path.normpath(p)
        return os.path.normpath(os.path.join(root_abs, p))

    train_p = joinp(doc["train"])
    val_raw = doc.get("val") or doc.get("train") or ""
    val_p = joinp(val_raw) if str(val_raw).strip() else train_p
    test_raw = doc.get("test") or ""
    test_p = joinp(test_raw) if str(test_raw).strip() else ""

    names = doc.get("names")
    if isinstance(names, dict):

        def sort_key(x):
            try:
                return int(x)
            except (ValueError, TypeError):
                return str(x)

        keys = sorted(names.keys(), key=sort_key)
        names_list = [names[k] for k in keys]
    elif isinstance(names, (list, tuple)):
        names_list = [str(x) for x in names]
    else:
        names_list = []

    nc = doc.get("nc")
    if nc is None:
        nc = len(names_list)
    nc = int(nc)

    return {
        "dataset_name": str(doc.get("dataset_name", "yolov5_data")),
        "train_set": train_p,
        "val_set": val_p if val_p else train_p,
        "test_set": test_p if test_p else (val_p if val_p else train_p),
        "nc": nc,
        "names": names_list,
    }


def merge_dataset_yaml_into_args(args, filepath: str) -> None:
    """
    将独立数据 yaml 合并进 args.data（与 YOLOv5 的 --data 类似，对应平台 PLATFORM_DATA_YAML / --dataset-yaml）。
    支持 MindYOLO 数据段（顶层 data: 或含 train_set 的平级字段）及 Ultralytics 风格（path/train/val/names）。
    若外部文件中 train_transforms / test_transforms 为 []，则跳过，避免覆盖模型 yaml + hyp 中的增强配置。
    """
    filepath = os.path.abspath(os.path.expanduser(filepath.strip()))
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"dataset yaml not found: {filepath}")
    yaml_dir = os.path.dirname(filepath)
    with open(filepath, "r", encoding="utf-8") as fin:
        doc = yaml.load(fin.read(), Loader=yaml.FullLoader)
    if not doc:
        return

    if isinstance(doc, dict) and "data" in doc and isinstance(doc["data"], dict):
        ext = dict(doc["data"])
    elif isinstance(doc, dict) and "train_set" in doc:
        ext = dict(doc)
    elif isinstance(doc, dict) and "train" in doc and "path" in doc:
        ext = _yolov5_style_to_mindyolo_data(doc, yaml_dir)
        if not ext:
            raise ValueError(f"Cannot parse as YOLOv5 data yaml: {filepath}")
    else:
        raise ValueError(
            f"Unrecognized dataset yaml: {filepath}. "
            "Expected MindYOLO (data.train_set / train_set) or YOLOv5 (path, train, val, names)."
        )

    for k, v in ext.items():
        if k in ("train_transforms", "test_transforms") and isinstance(v, list) and len(v) == 0:
            continue
        args.data[k] = v


class Config(dict):
    """
    Configuration namespace. Convert dictionary to members.
    """

    def __init__(self, cfg_dict):
        super(Config, self).__init__()
        for k, v in cfg_dict.items():
            setattr(self, k, Config(v) if isinstance(v, dict) else v)

    def __setattr__(self, name, value):
        self[name] = value
        self.__dict__.update({name: value})

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __str__(self):
        return config_format_func(self)

    def __repr__(self):
        return self.__str__()


def config_format_func(config, prefix=""):
    """
    Args:
        config: dict-like object
    Returns:
        formatted str
    """
    msg = ""
    if prefix:
        prefix += "."

    for k, v in config.__dict__.items():
        if isinstance(v, Config):
            msg += config_format_func(v, prefix=str(k))
        else:
            msg += format(prefix + str(k), "<40") + format(str(v), "<") + "\n"
    return msg
