#!/usr/bin/env bash
# run.sh: on the GPU node, switch lh-engine to the plugin check variant, run the live checks against it, and
# restore the frozen configuration (also on failure).
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$HOME/leaseholder
dropin=$HOME/.config/systemd/user/lh-engine.service.d
export LH_LIVE_ENGINE=http://127.0.0.1:8000
export LH_PROBE_LOG=$root/runs/engine-check/probe.jsonl
mkdir -p "$dropin" "$(dirname "$LH_PROBE_LOG")"
rm -f "$LH_PROBE_LOG"

restore() {
  rm -f "$dropin/engine-check.conf"
  systemctl --user daemon-reload
  systemctl --user restart lh-engine
}
trap restore EXIT

ln -sf "$here/engine-check.conf" "$dropin/engine-check.conf"
systemctl --user daemon-reload
systemctl --user restart lh-engine # returns once /health answers (ExecStartPost)

# pytest is not part of the engine environment; keep it beside it instead of changing the frozen venv.
tools=$root/cache/testtools
[[ -d $tools/pytest ]] || "${UV:-$HOME/.local/bin/uv}" pip install -q --target "$tools" pytest
PYTHONPATH=$tools:$here/../../src "$root/venvs/engine/bin/python" -m pytest -q -p no:cacheprovider -rA \
  "$here/test_live_engine.py"
