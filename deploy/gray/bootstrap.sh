#!/usr/bin/env bash
# bootstrap.sh: Gray's environment is the frozen box's (deploy/phastform/bootstrap.sh). Safe to re-run.
exec "$(cd "$(dirname "$0")" && pwd)/../phastform/bootstrap.sh"
