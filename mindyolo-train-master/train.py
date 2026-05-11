import argparse
import glob
import os
import platform
import shutil
import subprocess
import sys


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


def str2bool(value: str) -> bool:
    return str(value).lower() in {"1", "true", "yes", "y", "on"}


def resolve_default_config(model_type: str, model_scale: str) -> str:
    if model_type not in MODEL_CONFIG_MAP:
        raise ValueError(f"不支持的 model_type: {model_type}")
    if model_scale not in MODEL_CONFIG_MAP[model_type]:
        valid_scales = ",".join(MODEL_CONFIG_MAP[model_type].keys())
        raise ValueError(f"{model_type} 不支持 model_scale={model_scale}，可选: {valid_scales}")
    return MODEL_CONFIG_MAP[model_type][model_scale]


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MindYOLO 通用训练入口（平台 JSON 与 yolov5_train_config 字段对齐）")

    # 与 yolov5_train_config.json / mindyolo_train_config.json 一致的 flag_name（供前端拼接）
    parser.add_argument("--dataset_dir", "--dataset-dir", dest="dataset_dir", default="", help="数据集根目录")
    parser.add_argument("--best_weight", "--best-weight", dest="best_weight", default="", help="最优权重输出路径")
    parser.add_argument("--log-path", dest="log_path", default="", help="训练日志路径")
    parser.add_argument("--visual-path", dest="visual_path", default="", help="可视化/训练输出目录（对齐 tensorboard data_path）")
    parser.add_argument(
        "--data",
        dest="platform_data_yaml",
        default="",
        help="平台数据集 yaml；MindYOLO 训练仍以 --cfg/--config 指向的 yaml 为主，若需替换数据集请在 yaml 内配置或自行合并",
    )
    parser.add_argument("--cfg", dest="cfg", default="", help="模型/训练 yaml，对应 MindYOLO --config（优先于 model_type+model_scale）")
    parser.add_argument("--img", dest="platform_img", type=int, default=None, help="图像边长，映射 --img_size")
    parser.add_argument("--epochs", type=int, default=300, help="训练轮数")
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None, help="单设备 batch，映射 --per_batch_size")
    parser.add_argument("--optimizer", dest="optimizer", default="", help="优化器（MindYOLO 以 yaml 为准；可结合 extra_args 透传）")
    parser.add_argument("--workers", type=int, default=None, help="数据加载并行度（尝试透传）")
    parser.add_argument("--save-period", dest="save_period", type=int, default=None, help="保存周期（映射 keep_checkpoint 相关行为见 extra_args）")
    parser.add_argument("--patience", type=int, default=None, help="早停耐心轮")

    # 原有 MindYOLO 专用参数（脚本直接调试或与平台并存）
    parser.add_argument("--mindyolo_root", "--mindyolo-root", dest="mindyolo_root", default="./mindyolo_core", help="mindyolo 根目录")
    parser.add_argument(
        "--model_type",
        "--model-type",
        dest="model_type",
        choices=["yolov5", "yolov10"],
        default="yolov5",
        help="模型类型（未指定 --cfg/--config 时使用）",
    )
    parser.add_argument("--model_scale", "--model-scale", dest="model_scale", default="s", help="模型规模 n/s/m/l/x/b")
    parser.add_argument("--config", default="", help="显式配置文件路径（优先级低于 --cfg）")
    parser.add_argument("--device_target", "--device-target", dest="device_target", default="Ascend", help="Ascend/GPU/CPU")
    parser.add_argument("--is_parallel", "--is-parallel", dest="is_parallel", default="False", help="是否并行训练")
    parser.add_argument("--per_batch_size", "--per-batch-size", dest="per_batch_size", type=int, default=8, help="单卡 batch（平台未传 batch-size 时使用）")
    parser.add_argument("--img_size", "--img-size", dest="img_size", type=int, default=640, help="训练尺寸（平台未传 --img 时使用）")
    parser.add_argument("--weight", default="", help="初始权重路径")
    parser.add_argument("--save_dir", "--save-dir", dest="save_dir", default="./runs", help="输出目录（平台未传 visual-path 时使用）")
    parser.add_argument("--extra_args", "--extra-args", dest="extra_args", nargs=argparse.REMAINDER, help="透传给 MindYOLO train.py 的额外参数")
    return parser.parse_args()


def resolve_config_path(args: argparse.Namespace, mindyolo_root: str) -> str:
    if args.cfg:
        p = os.path.abspath(args.cfg)
        if os.path.isfile(p):
            return p
        raise FileNotFoundError(f"--cfg 指定的文件不存在: {p}")
    if args.config:
        p = os.path.abspath(args.config)
        if os.path.isfile(p):
            return p
        raise FileNotFoundError(f"--config 指定的文件不存在: {p}")
    config_rel = resolve_default_config(args.model_type, args.model_scale)
    return os.path.join(mindyolo_root, config_rel)


def resolve_save_dir(args: argparse.Namespace) -> str:
    if args.visual_path.strip():
        return os.path.abspath(args.visual_path.strip())
    return os.path.abspath(args.save_dir)


def resolve_per_batch_size(args: argparse.Namespace) -> int:
    if args.batch_size is not None:
        return args.batch_size
    return args.per_batch_size


def resolve_img_size(args: argparse.Namespace) -> int:
    if args.platform_img is not None:
        return args.platform_img
    return args.img_size


def maybe_copy_best_ckpt(save_dir: str, best_weight_dst: str) -> None:
    if not best_weight_dst.strip():
        return
    patterns = [
        os.path.join(save_dir, "**", "*best*.ckpt"),
        os.path.join(save_dir, "**", "*Best*.ckpt"),
        os.path.join(save_dir, "**", "*best*.pth"),
    ]
    found = []
    for pat in patterns:
        found.extend(glob.glob(pat, recursive=True))
    if not found:
        print(f"[MindYOLO] 未在 {save_dir} 下找到 best ckpt，跳过复制到 {best_weight_dst}")
        return
    found.sort(key=os.path.getmtime, reverse=True)
    src = found[0]
    dst = os.path.abspath(best_weight_dst.strip())
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    print(f"[MindYOLO] 已复制最优权重: {src} -> {dst}")


def main() -> None:
    args = build_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    preferred_root = os.path.abspath(os.path.join(base_dir, args.mindyolo_root))
    fallback_root = os.path.abspath(os.path.join(base_dir, "../mindyolo-master"))
    mindyolo_root = preferred_root if os.path.isdir(preferred_root) else fallback_root
    source_train_py = os.path.join(mindyolo_root, "train.py")
    if not os.path.isfile(source_train_py):
        raise FileNotFoundError(f"未找到 {source_train_py}")

    config_path = resolve_config_path(args, mindyolo_root)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"未找到配置文件 {config_path}")

    save_dir = resolve_save_dir(args)
    per_batch_size = resolve_per_batch_size(args)
    img_size = resolve_img_size(args)

    env = os.environ.copy()
    if args.dataset_dir.strip():
        env["DATASET_DIR"] = os.path.abspath(args.dataset_dir.strip())
    if args.platform_data_yaml.strip():
        env["PLATFORM_DATA_YAML"] = os.path.abspath(args.platform_data_yaml.strip())

    command = [
        sys.executable,
        source_train_py,
        "--config",
        config_path,
        "--device_target",
        args.device_target,
        "--is_parallel",
        "True" if str2bool(args.is_parallel) else "False",
        "--epochs",
        str(args.epochs),
        "--per_batch_size",
        str(per_batch_size),
        "--img_size",
        str(img_size),
        "--save_dir",
        save_dir,
    ]

    if args.weight:
        command.extend(["--weight", os.path.abspath(args.weight)])

    extra_args = list(args.extra_args) if args.extra_args else []

    if args.workers is not None:
        print("[MindYOLO] --workers 由 MindYOLO 配置 yaml 中 data.num_parallel_workers 控制，请在数据集 yaml 或 extra_args 中配置")

    if args.save_period is not None:
        print("[MindYOLO] --save-period 请通过 extra_args 传入 MindYOLO 支持的 checkpoint 相关参数（如 --keep_checkpoint_max）")

    if args.patience is not None:
        print(f"[MindYOLO] --patience={args.patience} 需 MindYOLO 训练 yaml / callback 支持，当前未自动映射")

    if args.device_target.upper() == "CPU":
        if "--ms_mode" not in extra_args:
            extra_args.extend(["--ms_mode", "1"])
        if "--ms_jit" not in extra_args:
            extra_args.extend(["--ms_jit", "False"])
        if "--ms_amp_level" not in extra_args:
            extra_args.extend(["--ms_amp_level", "O0"])

    command.extend(extra_args)

    if platform.system() == "Darwin":
        env.setdefault("OBJC_DISABLE_INITIALIZE_FORK_SAFETY", "YES")

    print("执行命令:", " ".join(command))
    subprocess.run(command, check=True, cwd=mindyolo_root, env=env)

    maybe_copy_best_ckpt(save_dir, args.best_weight)


if __name__ == "__main__":
    main()
