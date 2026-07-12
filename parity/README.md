# Parity environment

Pinned legacy environment (Python 3.8, torch 1.8.0+cpu, timm 0.4.12, healpy
1.15.2) matching `references/HEAL-SWIN/setup.py`, used ONLY to generate the
golden fixtures in `tests/goldens/`. The main test suite never needs this env.

Regenerate all goldens: see "Reference clamp patch" below for the required
procedure (the pristine reference crashes on some of the golden configs, so a
transient patch must be applied to `references/HEAL-SWIN` before running
`generate_goldens.py`).

In headless environments set MPLBACKEND=Agg (the scripts default to it programmatically).

Fixtures are written to `../tests/goldens/`. Commit them.

## Reference clamp patch

The pristine reference at `references/HEAL-SWIN` crashes for configs that
trigger the window-clamp branch: when the input resolution is small enough
that `window_size` (and `shift_size`) get clamped down, `SwinTransformerBlock`
still passes the original, unclamped `window_size` argument to
`WindowAttention` instead of the clamped `self.window_size`. This is a latent
bug in the upstream reference — it is never triggered by upstream's own
resolutions, only by the smaller 32x64 golden configs used here. It causes a
shape mismatch inside `WindowAttention` and crashes.

`reference-clamp-fix.patch` fixes this with a minimal 2-line change (one line
each in `heal_swin/models_torch/swin_hp_transformer.py` and
`heal_swin/models_torch/swin_transformer.py`): pass `self.window_size` instead
of `window_size` to `WindowAttention`. Its semantics — clamped window size —
are exactly what the nnx port implements, so applying it does not change what
"correct" means for parity purposes; it only unblocks the reference from
crashing on the small golden configs.

Goldens for the full-model cases (and, for simplicity, all goldens) are
therefore generated with this patch applied to the reference checkout. The
patch is applied transiently for generation only — the reference checkout in
this repo remains pristine (unpatched) at all other times; do not commit a
patched reference.

Regeneration procedure:

    cd references/HEAL-SWIN && git apply ../../parity/reference-clamp-fix.patch && cd -
    cd parity && uv run python generate_goldens.py
    cd references/HEAL-SWIN && git checkout -- . && cd -

Verify the reference is pristine again afterwards with
`git -C references/HEAL-SWIN status --porcelain` (should print nothing).

Pass `--only models64` instead of running the full script to regenerate just
the float64 gradient goldens (`tests/goldens/{case}_f64.npz` for `hp_base`,
`hp_ring`, `hp_cos_v2`, `flat_base`, `flat_cos_v2`) used by
`tests/test_parity_f64.py`; the same clamp-patch procedure applies since these
cases go through the same small golden configs.
