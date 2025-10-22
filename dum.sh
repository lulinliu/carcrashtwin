#!/usr/bin/env bash
set -euo pipefail
set -x  # 临时：打印每个命令，便于看是不是提前退出

# (可选) 激活 conda；否则就别用 CONDA_PREFIX
set +u
source /scratch/10102/hh29499/anaconda/etc/profile.d/conda.sh || true
conda activate cosmos-predict2 2>/dev/null || true
set -u

# 不要在 set -u 下直接用未定义变量；用 “存在就加” 的写法
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi

# 避免 ~/.local 抢包（decord 问题的根因之一）
export PYTHONNOUSERSITE=1
unset PYTHONPATH

# 用 python -u 取消缓冲，确保 srun 下即时看到输出
python -u - <<'PY'
import os, pathlib, sys
print("Node:", os.uname().nodename, "PID:", os.getpid())
sys.stdout.flush()

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
    from decord import VideoReader, cpu
    print("decord:", decord.__file__)
    so = pathlib.Path(decord.__file__).with_name("libdecord.so")
    print("libdecord.so exists:", so.exists(), "->", so)
except Exception as e:
    print("decord import failed:", repr(e))

print("Dummy OK.")
PY
