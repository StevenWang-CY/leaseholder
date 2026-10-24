#!/usr/bin/env bash
# authorize-tunnel.sh CLIENT GPU_NODE: let CLIENT's tunnel key (~/.ssh/lh_tunnel_ed25519 on CLIENT) forward
# GPU_NODE's loopback service ports and nothing else: the engine (8000), STT (8100) and TTS (8880). Writes one
# restricted authorized_keys entry on GPU_NODE, replacing any earlier entry for the same key. Uses LH_SSH like
# deploy.sh.
set -euo pipefail
client=${1:?usage: authorize-tunnel.sh CLIENT GPU_NODE}
node=${2:?usage: authorize-tunnel.sh CLIENT GPU_NODE}
ssh=${LH_SSH:-ssh}
key=$($ssh "$client" cat .ssh/lh_tunnel_ed25519.pub)
permit=""
for port in 8000 8100 8880; do permit+="permitopen=\"127.0.0.1:$port\","; done
printf '%s\nrestrict,port-forwarding,%scommand="/bin/false" %s\n' "${key% *}" "$permit" "$key" |
  $ssh "$node" 'umask 077; mkdir -p .ssh; touch .ssh/authorized_keys
    read -r keyblob; read -r line
    { grep -vF "$keyblob" .ssh/authorized_keys || true; printf "%s\n" "$line"; } >.ssh/authorized_keys.new
    mv .ssh/authorized_keys.new .ssh/authorized_keys
    echo "authorized: ${line##* } ($(grep -cF "$keyblob" .ssh/authorized_keys) entry)"'
