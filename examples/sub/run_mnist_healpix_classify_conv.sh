#!/bin/bash
# HTCondor executable wrapper for the spherical-MNIST HealConv classifier.
#
#   $1 = repo root (the uv project directory containing pyproject.toml)
#
# Runs the training through uv's `examples` extra (grain / datasets / healpy).
set -euo pipefail

cd "$1"

# The script picks the device itself: JAX_PLATFORMS=cuda for the main process,
# cpu for spawned grain data-loader workers. Unset any inherited value so a
# stray JAX_PLATFORMS=cpu from the submit environment can't force CPU-only
# training.
unset JAX_PLATFORMS
exec uv run python examples/mnist_healpix_classify_conv.py
