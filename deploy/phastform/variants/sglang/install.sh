#!/usr/bin/env bash
# install.sh: build ~/leaseholder/venvs/sglang, the environment of the sglang variant (SGLang 0.5.22, torch
# 2.14.1+cu130), from requirements.txt. Run it once on the GPU node before variant.sh use sglang; the frozen box does
# not depend on it. It reuses the curand header links that deploy/phastform/bootstrap.sh makes for FlashInfer's JIT.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
uv=${UV:-$HOME/.local/bin/uv}
venv=$HOME/leaseholder/venvs/sglang
[[ -x $venv/bin/python ]] || "$uv" venv -q --python 3.12 "$venv"
# torch's CUDA 13.0 wheels come from PyTorch's index. FlashInfer's JIT needs ninja, which the lock includes. The
# lock pins its one pre-release (cuda-tile) exactly, so sync needs no pre-release flag (and does not accept one).
"$uv" pip sync -q --python "$venv/bin/python" --extra-index-url https://download.pytorch.org/whl/cu130 \
  --index-strategy unsafe-best-match "$here/requirements.txt"
mkdir -p "$HOME/leaseholder/cache/flashinfer-sglang"
"$venv/bin/python" -c 'import sglang, torch; print("install: sglang", sglang.__version__, "torch", torch.__version__)'
