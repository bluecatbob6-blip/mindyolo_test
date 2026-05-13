#!/usr/bin/env python3
"""
根据平台前端导出的训练 JSON（mindyolo_train_config.json / yolov5_train_config.json）
读取 input / output / log / visualization / config 中的路径与数值，拼接命令后启动
run.command + run.required_args（默认 python3 train.py）并附加各 flag。

用法:
  python3 scripts/run_platform_train_json.py --json mindyolo_train_config.json
  python3 scripts/run_platform_train_json.py --json yolov5_train_config.json --dry-run
  python3 scripts/run_platform_train_json.py -- --device-target CPU --epochs 3
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _empty(v) -> bool:
    if v is None:
        return True
    if isinstance(v, str) and not v.strip():
        return True
    return False


def _config_effective_value(item: dict):
    """与前端约定一致：列表型可选项的 value 为枚举，选用 default_value；否则非空 value 优先。"""
    val = item.get("value")
    default = item.get("default_value")
    if item.get("front_type") == "list" and isinstance(val, list):
        return default
    if not _empty(val) and not (isinstance(val, list) and item.get("type_of_list") == "single"):
        return val
    return default


def _append_flag(cmd: list[str], flag: str, value) -> None:
    if value is None:
        return
    if isinstance(value, str) and not value.strip():
        return
    cmd.extend([flag, str(value)])


def build_train_argv(data: dict) -> list[str]:
    cmd: list[str] = []
    run = data.get("run") or {}
    cmd.extend(run.get("command") or ["python3"])
    for a in run.get("required_args") or []:
        cmd.append(a)

    for block in data.get("input") or []:
        fn, path = block.get("flag_name"), (block.get("path") or "").strip()
        if fn and path:
            _append_flag(cmd, fn, path)

    for block in data.get("output") or []:
        fn, path = block.get("flag_name"), (block.get("path") or "").strip()
        if fn and path:
            _append_flag(cmd, fn, path)

    log = data.get("log") or {}
    if isinstance(log, dict):
        lf, lp = log.get("flag_name"), (log.get("path") or "").strip()
        if lf and lp:
            _append_flag(cmd, lf, lp)

    vis = data.get("visualization") or {}
    if isinstance(vis, dict):
        vf, vp = vis.get("flag_name"), (vis.get("data_path") or "").strip()
        if vf and vp:
            _append_flag(cmd, vf, vp)

    for item in data.get("config") or []:
        fn = item.get("flag_name")
        if not fn:
            continue
        ev = _config_effective_value(item)
        if ev is None or (isinstance(ev, str) and not ev.strip()):
            continue
        _append_flag(cmd, fn, ev)

    return cmd


def main() -> int:
    root = _repo_root()
    ap = argparse.ArgumentParser(description="按平台训练 JSON 启动 train.py")
    ap.add_argument(
        "--json",
        type=Path,
        default=root / "mindyolo_train_config.json",
        help="平台训练组件 JSON 路径",
    )
    ap.add_argument("--dry-run", action="store_true", help="只打印命令，不执行")
    ap.add_argument("extra", nargs=argparse.REMAINDER, help="附加参数（建议先写 --）")
    ns = ap.parse_args()
    path = ns.json.expanduser().resolve()
    if not path.is_file():
        print(f"找不到 JSON: {path}", file=sys.stderr)
        return 1

    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    argv = build_train_argv(payload)
    extra = ns.extra or []
    if extra and extra[0] == "--":
        extra = extra[1:]
    argv = argv + extra

    # required_args 里是 train.py，需相对仓库根目录
    os.chdir(root)
    print("工作目录:", root)
    print("执行:", subprocess.list2cmdline(argv))
    if ns.dry_run:
        return 0
    return subprocess.call(argv)


if __name__ == "__main__":
    raise SystemExit(main())
