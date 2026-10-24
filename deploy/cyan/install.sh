#!/usr/bin/env bash
# install.sh: link cyan's units into the user manager and enable them at boot, and make the gateway's cache-salt
# secret once. Lingering must be on (`loginctl enable-linger`), or the user manager stops with the last SSH session.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
units=$HOME/.config/systemd/user
mkdir -p "$units"
ln -sf "$here"/systemd/* "$units/"
# 32 random bytes in hex, the key of every tenant's cache_salt (cache_salt_secret_file in gateway.toml). Never
# replaced: a new secret changes every salt, which orphans every tenant's prefix cache.
salt_secret=$HOME/.config/leaseholder/cache-salt-secret
if [[ ! -f $salt_secret ]]; then
  mkdir -p "$(dirname "$salt_secret")"
  (umask 077 && od -An -N32 -tx1 /dev/urandom | tr -d ' \n' >"$salt_secret.new" && echo >>"$salt_secret.new")
  mv "$salt_secret.new" "$salt_secret"
fi
systemctl --user daemon-reload
systemctl --user enable -q lh-voicenet.service lh-livekit.service lh-tunnel-phastform.service lh-voice-worker.service
loginctl show-user "$(id -un)" --property=Linger
