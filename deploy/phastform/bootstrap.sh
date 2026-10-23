#!/usr/bin/env bash
# bootstrap.sh: build a GPU node's frozen-box environment under ~/leaseholder (Phastform; Gray reuses it). Safe to
# re-run.
#   venvs/engine            vLLM 0.32.0 with the audio extra, serving the LLM and Whisper
#   venvs/tts               Kokoro-FastAPI 0.9.1-rc1 (torch 2.8.0+cu128)
#   vendor/Kokoro-FastAPI   source checkout and model weights
#   cache/                  FlashInfer JIT workspace and the curand header links
#   ~/.cache/huggingface    the LLM and Whisper weights, at pinned revisions
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$HOME/leaseholder
uv=${UV:-$HOME/.local/bin/uv}
# The locks were resolved with uv 0.11.27; uv 0.10.8 could not resolve the engine lock's torchcodec==0.17.0+cu130
# through --torch-backend.
version=$("$uv" --version | awk '{print $2}')
[[ $(printf '%s\n' 0.11.27 "$version" | sort -V | head -1) == 0.11.27 ]] ||
  { echo "bootstrap: uv $version is older than 0.11.27; set UV to a newer uv" >&2; exit 2; }
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

# The units run offline (HF_HUB_OFFLINE=1), so the models are fetched here, at the revisions their configs pin.
for config in "$here/engine.yaml" "$here/stt.yaml"; do
  read -r model revision < <(awk '$1 == "model:" {m = $2} $1 == "revision:" {r = $2} END {print m, r}' "$config")
  [[ -n $revision ]] || { echo "bootstrap: $config pins no revision" >&2; exit 2; }
  "$root/venvs/engine/bin/hf" download -q "$model" --revision "$revision" >/dev/null
done

# FlashInfer's JIT needs curand headers that /usr/local/cuda-13.0 lacks on this host; expose the venv's copies.
ln -sf "$root"/venvs/engine/lib/python3.12/site-packages/nvidia/cu13/include/curand*.h "$root/cache/curand_include/"

echo "bootstrap: engine $("$root/venvs/engine/bin/python" -c 'import vllm, torch; print("vllm", vllm.__version__, "torch", torch.__version__)')"
echo "bootstrap: tts $("$root/venvs/tts/bin/python" -c 'import torch; print("torch", torch.__version__)')"
