#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1; unset PYTHONPATH; export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
python - <<'PY'
import os, pathlib, torch, tokenizers
import decord
from decord import VideoReader, cpu
print("Node:", os.uname().nodename)
print("CUDA visible:", torch.cuda.is_available(), "devices:", torch.cuda.device_count())
print("tokenizers:", tokenizers.__file__)
print("decord:", decord.__file__)
print("Dummy OK.")
PY
