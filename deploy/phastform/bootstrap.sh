#!/usr/bin/env bash
# bootstrap.sh: build the Phastform frozen-box environment under ~/leaseholder. Safe to re-run.
#   venvs/engine            vLLM 0.32.0 with the audio extra, serving the LLM and Whisper
#   venvs/tts               Kokoro-FastAPI 0.9.1-rc1 (torch 2.8.0+cu128)
#   vendor/Kokoro-FastAPI   source checkout and model weights
#   cache/                  FlashInfer JIT workspace and the curand header links
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$HOME/leaseholder
uv=${UV:-$HOME/.local/bin/uv}
mkdir -p "$root"/{venvs,vendor,cache/flashinfer,cache/curand_include,runs}

venv() { # venv NAME REQUIREMENTS UV_ARGS...
  local name=$1 req=$2
  shift 2
  [[ -x $root/venvs/$name/bin/python ]] || "$uv" venv -q --python 3.12 "$root/venvs/$name"
  "$uv" pip sync -q --python "$root/venvs/$name/bin/python" "$@" "$req"
}

venv engine "$here/requirements/engine.txt" --torch-backend cu130
venv tts "$here/requirements/tts.txt" --torch-backend cu128

# Kokoro-FastAPI publishes no tag for 0.9.1-rc1, so the measured commit is the version pin.
kokoro=$root/vendor/Kokoro-FastAPI
[[ -d $kokoro/.git ]] || git clone -q https://github.com/remsky/Kokoro-FastAPI.git "$kokoro"
git -C "$kokoro" checkout -q ebd4ec9
"$uv" pip install -q --python "$root/venvs/tts/bin/python" --no-deps -e "$kokoro"
if [[ ! -f $kokoro/api/src/models/v1_0/kokoro-v1_0.pth ]]; then
  (cd "$kokoro" && "$root/venvs/tts/bin/python" docker/scripts/download_model.py --output api/src/models/v1_0)
fi

# FlashInfer's JIT needs curand headers that /usr/local/cuda-13.0 lacks on this host; expose the venv's copies.
ln -sf "$root"/venvs/engine/lib/python3.12/site-packages/nvidia/cu13/include/curand*.h "$root/cache/curand_include/"

echo "bootstrap: engine $("$root/venvs/engine/bin/python" -c 'import vllm, torch; print("vllm", vllm.__version__, "torch", torch.__version__)')"
echo "bootstrap: tts $("$root/venvs/tts/bin/python" -c 'import torch; print("torch", torch.__version__)')"
