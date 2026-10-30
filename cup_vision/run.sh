#!/bin/sh
set -eu
REPO_DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$REPO_DIR"
# Configure CUP_PYTHON on the machine hosting viam-server, not on the client laptop.
CUP_PYTHON=${CUP_PYTHON:-"$REPO_DIR/.conda/bin/python"}
if [ ! -x "$CUP_PYTHON" ]; then
    echo "Set CUP_PYTHON to a Python interpreter with cup_vision/requirements.txt installed" >&2
    exit 1
fi
exec "$CUP_PYTHON" -m cup_vision "$@"
