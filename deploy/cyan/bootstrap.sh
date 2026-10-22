#!/usr/bin/env bash
# bootstrap.sh: prepare cyan, the control host. Safe to re-run; stop lh-voice-worker first if a re-run may change
# the voice venv's packages.
#   bin/livekit-server                  LiveKit server 1.13.10
#   venvs/voice                         lh-voice-worker and lh-voice, from python/leaseholder_voice/uv.lock
#   venvs/bench                         lh-bench and the AIPerf plugins, from python/leaseholder_bench/uv.lock
#   venvs/smg                           SMG 1.13.0 for E2's arm 8b (lh-smg.service), from requirements/smg.txt
#   ~/.config/leaseholder/livekit.env   API key pair for the local LiveKit server and the voice worker
#   ~/.ssh/lh_tunnel_ed25519            the tunnel key; authorize it on each GPU node with deploy/authorize-tunnel.sh
set -euo pipefail
src=$(cd "$(dirname "$0")/../.." && pwd)
root=$HOME/leaseholder
uv=${UV:-$HOME/.local/bin/uv}
[[ -x $uv ]] || uv=$(command -v uv) || { echo "bootstrap: no uv; set UV to its path" >&2; exit 2; }
mkdir -p "$root"/{bin,venvs,data,runs} "$HOME/.config/leaseholder" "$HOME/.ssh"

livekit=1.13.10
if ! "$root/bin/livekit-server" --version 2>/dev/null | grep -q "$livekit"; then
  curl -fsSL "https://github.com/livekit/livekit/releases/download/v$livekit/livekit_${livekit}_linux_amd64.tar.gz" |
    tar -xz -C "$root/bin" livekit-server
fi

# Each package in its own venv, exactly as its lock file says, with the test extra: the voice plane's integration
# test and the bench tests run from these venvs too (UV_PROJECT_ENVIRONMENT, as in their READMEs).
venv() { # venv NAME PACKAGE
  UV_PROJECT_ENVIRONMENT=$root/venvs/$1 "$uv" sync -q --locked --python 3.12 --extra test \
    --project "$src/python/$2"
}
venv voice leaseholder_voice
venv bench leaseholder_bench
[[ -x $root/venvs/smg/bin/python ]] || "$uv" venv -q --python 3.12 "$root/venvs/smg"
"$uv" pip sync -q --python "$root/venvs/smg/bin/python" "$src/deploy/cyan/requirements/smg.txt"

# The worker's models ship in its wheels (the v1-mini end-of-turn model in livekit-local-inference, Silero's ONNX
# file in its plugin), so there is nothing to download. Check that they load offline, as lh-voice-worker.service
# runs them (HF_HUB_OFFLINE=1, inside a namespace with no route out).
HF_HUB_OFFLINE=1 "$root/venvs/voice/bin/python" -c \
  'from livekit.local_inference import init_eot; init_eot(); from livekit.plugins import silero; silero.VAD.load()'

env_file=$HOME/.config/leaseholder/livekit.env
if [[ ! -f $env_file ]]; then
  (
    umask 077
    key=lh$(od -An -N6 -tx1 /dev/urandom | tr -d ' \n')
    secret=$(head -c 32 /dev/urandom | base64 | tr -d '/+=\n')
    printf 'LIVEKIT_KEYS="%s: %s"\nLIVEKIT_API_KEY=%s\nLIVEKIT_API_SECRET=%s\nLIVEKIT_URL=ws://127.0.0.1:17880\n' \
      "$key" "$secret" "$key" "$secret" >"$env_file"
  )
fi

[[ -f $HOME/.ssh/lh_tunnel_ed25519 ]] ||
  ssh-keygen -q -t ed25519 -N '' -C "lh-tunnel@$(hostname -s)" -f "$HOME/.ssh/lh_tunnel_ed25519"
[[ -f $HOME/.config/leaseholder/tunnel-phastform.env ]] ||
  echo "bootstrap: write LH_TUNNEL_TARGET=<user>@<gpu node> to ~/.config/leaseholder/tunnel-phastform.env" >&2
version() { "$root/venvs/$1/bin/python" -c 'import importlib.metadata as m, sys; print(sys.argv[1], m.version(sys.argv[1]))' "$2"; }
echo "bootstrap: $("$root/bin/livekit-server" --version)"
echo "bootstrap: voice $(version voice livekit-agents); its models load offline"
echo "bootstrap: bench $(version bench aiperf)"
echo "bootstrap: smg $(version smg smg)"
