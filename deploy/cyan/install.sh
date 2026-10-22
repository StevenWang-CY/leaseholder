#!/usr/bin/env bash
# install.sh: link cyan's units into the user manager and enable them at boot.
# Lingering must be on (`loginctl enable-linger`), or the user manager stops with the last SSH session.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
units=$HOME/.config/systemd/user
mkdir -p "$units"
ln -sf "$here"/systemd/* "$units/"
systemctl --user daemon-reload
systemctl --user enable -q lh-voicenet.service lh-livekit.service lh-tunnel-phastform.service lh-voice-worker.service
loginctl show-user "$(id -un)" --property=Linger
