#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec "$SCRIPT_DIR/.venv/bin/python" -m src.serve \
  --archive "$SCRIPT_DIR/FamilyPhotoArchive"

