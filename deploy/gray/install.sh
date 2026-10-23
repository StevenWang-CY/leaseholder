#!/usr/bin/env bash
# install.sh: link the frozen-box units (deploy/phastform/systemd) into Gray's user manager, with Gray's drop-ins. Gray is
# a second GPU node in booked windows, shared with other work, so leaseholder.target is not enabled at boot: start it
# for a window and stop it after. Lingering must be on (`loginctl enable-linger`).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
units=$HOME/.config/systemd/user
mkdir -p "$units"
ln -sf "$here"/../phastform/systemd/* "$units/"
for conf in "$here"/systemd/*.conf; do
  unit=$(basename "$conf" .conf)
  mkdir -p "$units/$unit.service.d"
  ln -sf "$conf" "$units/$unit.service.d/gray.conf"
done
systemctl --user daemon-reload
loginctl show-user "$(id -un)" --property=Linger
