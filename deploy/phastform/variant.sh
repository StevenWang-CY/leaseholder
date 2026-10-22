#!/usr/bin/env bash
# variant.sh on|off NAME: apply or remove a variant of the frozen box, then restart it.
# A variant is a directory variants/NAME/ of systemd drop-ins, one UNIT.conf per unit it changes (lh-engine.conf may add
# engine flags, which beat engine.yaml). "on" links each into ~/.config/systemd/user/UNIT.service.d/ as
# variant-NAME.conf; "off" removes those links. The box restarts in both cases: ~2 min with warm caches.
set -euo pipefail
action=${1:?on|off} name=${2:?variant name}
dir=$(cd "$(dirname "$0")" && pwd)/variants/$name
[[ -d $dir ]] || { echo "variant: no such variant: $dir" >&2; exit 2; }
units=$HOME/.config/systemd/user
systemctl --user stop leaseholder.target lh-engine lh-stt lh-tts
for conf in "$dir"/*.conf; do
  link=$units/$(basename "$conf" .conf).service.d/variant-$name.conf
  case $action in
    on) mkdir -p "$(dirname "$link")"; ln -sf "$conf" "$link" ;;
    off) rm -f "$link" ;;
    *) echo "variant: action must be on or off" >&2; exit 2 ;;
  esac
done
systemctl --user daemon-reload
systemctl --user start leaseholder.target
systemctl --user is-active lh-stt lh-tts lh-engine
