#!/usr/bin/env bash
# authorize-tunnel.sh CLIENT GPU_NODE: let CLIENT's tunnel key (~/.ssh/lh_tunnel_ed25519 on CLIENT) forward
# GPU_NODE's loopback service ports and nothing else: the engine (8000), STT (8100) and TTS (8880), with no shell
# and no listener of its own there. Writes one restricted authorized_keys entry on GPU_NODE, replacing any earlier
# entry for the same key. Uses LH_SSH like deploy.sh.
set -euo pipefail
client=${1:?usage: authorize-tunnel.sh CLIENT GPU_NODE}
node=${2:?usage: authorize-tunnel.sh CLIENT GPU_NODE}
ssh=${LH_SSH:-ssh}
key=$($ssh "$client" cat .ssh/lh_tunnel_ed25519.pub)
# The key's type and blob identify its entry; the comment after them is optional.
read -r type blob _ <<<"$key"
[[ -n $blob ]] || { echo "authorize-tunnel: no public key in $client's .ssh/lh_tunnel_ed25519.pub" >&2; exit 2; }
permit=""
for port in 8000 8100 8880; do permit+="permitopen=\"127.0.0.1:$port\","; done
# port-forwarding re-enables remote forwards (-R) too, which permitopen does not limit: the only listener the key may
# ask for is a privileged port, which the user's session cannot bind.
permit+='permitlisten="127.0.0.1:1",'
printf '%s\nrestrict,port-forwarding,%scommand="/bin/false" %s\n' "$type $blob" "$permit" "$key" |
  $ssh "$node" 'umask 077; mkdir -p .ssh; touch .ssh/authorized_keys
    read -r keyblob; read -r line
    { grep -vF "$keyblob" .ssh/authorized_keys || true; printf "%s\n" "$line"; } >.ssh/authorized_keys.new
    mv .ssh/authorized_keys.new .ssh/authorized_keys
    echo "authorized: ${line##* } ($(grep -cF "$keyblob" .ssh/authorized_keys) entry)"'
