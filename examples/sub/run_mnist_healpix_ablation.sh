#!/bin/bash
# HTCondor executable wrapper for the HealSwin shift/pos-embed ablation.
#
#   $1 = repo root (the uv project directory containing pyproject.toml)
#   $2 = pos_embed       (rel_bias | rope_axial | rope_mixed)
#   $3 = shift_strategy  (nest_roll | nest_grid_shift | nest_grid_shift_exact | ring_shift)
#
# Runs the training through uv (grain / datasets / healpy are project deps).
set -euo pipefail

cd "$1"

# The script picks the device itself: JAX_PLATFORMS=cuda for the main process,
# cpu for spawned grain data-loader workers. Unset any inherited value so a
# stray JAX_PLATFORMS=cpu from the submit environment can't force CPU-only
# training.
unset JAX_PLATFORMS
exec uv run python examples/mnist_healpix_ablation.py \
    --pos-embed "$2" --shift-strategy "$3"
