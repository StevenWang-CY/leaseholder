#!/usr/bin/env bash
# bootstrap.sh: prepare cyan, the control host. Safe to re-run.
#   bin/livekit-server             LiveKit server 1.13.10
#   ~/.config/leaseholder/livekit.env   API key pair for the local LiveKit server and the voice worker
#   ~/.ssh/lh_tunnel_ed25519       the tunnel key; authorize it on each GPU node with deploy/authorize-tunnel.sh
set -euo pipefail
root=$HOME/leaseholder
mkdir -p "$root"/{bin,venvs,data,runs} "$HOME/.config/leaseholder" "$HOME/.ssh"

livekit=1.13.10
if ! "$root/bin/livekit-server" --version 2>/dev/null | grep -q "$livekit"; then
  curl -fsSL "https://github.com/livekit/livekit/releases/download/v$livekit/livekit_${livekit}_linux_amd64.tar.gz" |
    tar -xz -C "$root/bin" livekit-server
fi

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
echo "bootstrap: $("$root/bin/livekit-server" --version)"
