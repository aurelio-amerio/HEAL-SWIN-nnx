#!/bin/bash
# HTCondor executable wrapper for the spherical GRF flow-matching example.
#
#   $1 = repo root (the uv project directory containing pyproject.toml)
set -euo pipefail

cd "$1"

# The script picks the device itself: JAX_PLATFORMS=cuda for the main process,
# cpu for spawned grain simulation workers. Unset any inherited value so a
# stray JAX_PLATFORMS=cpu from the submit environment can't force CPU-only
# training.
unset JAX_PLATFORMS
exec uv run python examples/spherical_grf_flowmatch.py
