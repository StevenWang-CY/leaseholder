#!/usr/bin/env bash
# deploy.sh HOST: sync this repository to HOST:~/leaseholder/src and install HOST's units (deploy/<short-name>/).
# Set LH_SSH to the ssh command for your hosts, e.g. LH_SSH="ssh -F $HOME/.ssh/penn_config".
set -euo pipefail
host=${1:?usage: deploy.sh HOST}
name=${host%%.*}
repo=$(cd "$(dirname "$0")/.." && pwd)
ssh=${LH_SSH:-ssh}
[[ -d $repo/deploy/$name ]] || { echo "deploy: no deploy/$name directory" >&2; exit 2; }
$ssh "$host" mkdir -p leaseholder/src
rsync -a --delete -e "$ssh" --exclude .git --exclude .claude --exclude target --exclude __pycache__ --exclude '*.egg-info' \
  --exclude .venv --exclude .pytest_cache --exclude .ruff_cache "$repo/" "$host:leaseholder/src/"
# The commit synced, which run records name (lh-bench e2 run).
rev=$(git -C "$repo" describe --always --dirty 2>/dev/null || echo unknown)
$ssh "$host" "echo $rev >leaseholder/src/DEPLOYED"
$ssh "$host" "bash leaseholder/src/deploy/$name/install.sh"
