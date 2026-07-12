# Parity environment

Pinned legacy environment (Python 3.8, torch 1.8.0+cpu, timm 0.4.12, healpy
1.15.2) matching `references/HEAL-SWIN/setup.py`, used ONLY to generate the
golden fixtures in `tests/goldens/`. The main test suite never needs this env.

Regenerate all goldens:

    cd parity
    uv sync
    uv run python generate_goldens.py

Fixtures are written to `../tests/goldens/`. Commit them.
