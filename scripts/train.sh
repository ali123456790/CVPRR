#!/usr/bin/env bash
set -e

# Ensure the project is in the Python path
export PYTHONPATH=$(pwd):$PYTHONPATH

echo "Starting DyGRAV training..."
python -m dygrav.cli.train "$@"
