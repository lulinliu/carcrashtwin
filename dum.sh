#!/usr/bin/env bash
set -euo pipefail
set -x  # 临时：打印每个命令，便于看是不是提前退出

try:
    import torch
    print("CUDA visible:", torch.cuda.is_available(), "devices:", torch.cuda.device_count())
except Exception as e:
    print("Torch import failed:", repr(e))

try:
    import tokenizers
    print("tokenizers:", tokenizers.__file__)
except Exception as e:
    print("tokenizers import failed:", repr(e))

try:
    import decord
    print("decord:", decord.__file__)
    from decord import VideoReader, cpu
    print("decord:", decord.__file__)
    so = pathlib.Path(decord.__file__).with_name("libdecord.so")
    print("libdecord.so exists:", so.exists(), "->", so)
except Exception as e:
    print("decord import failed:", repr(e))

print("Dummy OK.")
PY
