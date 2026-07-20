# Design: compute `dtype` knob (bf16 mixed precision) for HealSwin, SwinUnet, and HealConv

**Date:** 2026-07-20
**Status:** Approved
**Builds on:** `2026-07-20-param-dtype-design.md` (the `param_dtype` knob, landed) — this is
the "separate computation dtype" follow-up that spec listed as out of scope.
**Reference implementation:** GenSBI mixed-precision branch
(`GenSBI/docs/superpowers/2026-07-20-mixed-precision-summary.md`), especially
`gensbi/models/flux1` and `tests/models/flux1/test_flux1_precision.py`.

## Goal

fp32 master weights with bf16 compute, GenSBI-contract style: a `dtype` field on all
three Params dataclasses, threaded into every compute layer, with knob-independent
fp32 islands for the numerics that bf16 handles badly, explicit exit casts wherever an
island output enters the stream through a non-self-healing op, and a test battery that
locks internal activation dtypes (the silent fp32-re-promotion leak class GenSBI fixed
in five models has zero endpoint signal — only spy tests catch it).

Decisions made during brainstorming:

- **All three models**: `HealSwinParams`, `SwinParams`, `HealConvParams`.
- **Default `dtype="bfloat16"`** (GenSBI transformer convention); the calibrated
  drift-lock test is the guardrail. `param_dtype` default stays `"float32"`.
- **Models emit fp32** regardless of the knob (final projections and the standalone
  encoder endpoint are fp32 islands); inputs are cast to fp32 at the door.
- **Explicit per-layer threading** (approach A) — not promotion-following, not a
  jmp-style global policy. Leaks become impossible-by-construction plus test-locked,
  matching the GenSBI idiom exactly.

## 1. Config surface

Each Params dataclass's precision group becomes:

```python
# precision
param_dtype: str = "float32"   # parameter storage (master weights)
dtype: str = "bfloat16"        # compute/matmul dtype
```

`dtype` is validated by the same `canonical_float_dtype` helper as `param_dtype`
(canonical string stored so `json.dumps(dataclasses.asdict(params))` keeps working;
non-float and garbage values raise `ValueError`; `float64` rejected unless
`jax_enable_x64` is on). The helper gains a `field_name` argument so error messages
name the offending knob. The two knobs are independent — any (floating, floating)
pair is legal: pure-fp32 (`dtype="float32"`) and pure-bf16
(`param_dtype="bfloat16"`) both remain expressible, matching `Flux1Params`.

## 2. The contract

Ported verbatim from the GenSBI summary:

| Rule | Implementation |
|---|---|
| fp32 master weights | `param_dtype` default `"float32"`; grads/optimizer state fp32 automatically |
| Compute knob | `dtype` threaded into every `nnx.Linear` / `nnx.Conv` |
| fp32 islands (knob-independent) | all LayerNorms, attention logits→softmax, RoPE, `l2_normalize`, final projections — **constructed** with `dtype=jnp.float32`, never post-hoc |
| Islands emit compute dtype | explicit `.astype` at every non-self-healing exit (Section 4) — except the emit-fp32 endpoints |
| Inputs fp32 at the door | encoder-entry cast becomes `jnp.asarray(x, jnp.float32)` (replaces the `param_dtype` cast); the first compute layer downcasts |
| Models emit fp32 | decoder `output` conv and encoder final `norm` constructed fp32; `HealSwin(x)`, `SwinUnet(x)`, `HealConv(x)`, `*Encoder(x)` all return fp32 |
| No loss scaling | bf16 shares fp32's exponent range; losses live in GenSBI-examples, out of scope |

## 3. The attention island (`WindowAttention`, both Swin models)

Precision walk-through of `__call__` under `dtype=bf16`:

1. **`qkv` linear** — compute dtype.
2. **`l2_normalize`** (`layers.py`) — becomes a knob-independent fp32 island: upcast
   to fp32, `rsqrt(sum(x²) + eps)` in fp32, cast back to the input dtype. (In bf16
   the head_dim sum-of-squares loses ~5 bits and `eps=1e-12` is meaningless next to
   bf16 resolution.) Mirrors GenSBI's QKNorm: fp32 math, emit stream dtype.
3. **RoPE** — already the correct shape (`apply_rope` computes fp32, returns input
   dtype; `rope_freqs` Param and rotation tables fp32). Untouched.
4. **Logits** — `jnp.einsum(..., preferred_element_type=jnp.float32)` replaces
   `q @ k.swapaxes(-2, -1)`: bf16 operands, fp32 accumulation/output. Native on
   tensor cores — bf16-priced flops, fp32 logits.
5. **fp32 logits region** — everything between matmul and softmax stays fp32:
   `logit_scale` (fp32 Param) `exp(min(...))` and multiply; rel-bias table (fp32
   Param) gather + add; window-mask Buffer add (the existing
   `mask.astype(attn.dtype)` use-site cast now resolves to fp32); `jax.nn.softmax`.
   Rationale: cosine attention bounds the logits (|logit| ≤ scale ≤ 100) so bf16
   would not overflow — the issue is *resolution* (8 mantissa bits ⇒ logit
   quantization ~0.5 at scale 100 visibly distorts attention weights), and step 4's
   fp32 output makes the region free apart from bandwidth on an N×N window matrix
   (N = window_size, typically 16).
6. **Exit cast** — probabilities `.astype(dtype)` immediately after softmax, then
   `attn_drop`, `attn @ v` in bf16, `proj` linear in compute dtype.

`swin.py`'s mirrored `WindowAttention` gets the identical treatment.

## 4. Norm islands and the leak map

Two promotion mechanisms drive every decision here:

- **JAX binary-op promotion:** bf16 + fp32 → fp32, silently, in adds/concats. Any
  fp32 island output *combined* with the bf16 stream re-promotes everything
  downstream — with zero endpoint signal (params and output test correctly either
  way).
- **nnx layer input promotion:** a Linear/Conv constructed with `dtype=bf16` casts
  its inputs to bf16 before the matmul (`promote_dtype`), acting as a dtype
  firewall. fp32 flowing *directly* into one is absorbed — "self-healing".

All `nnx.LayerNorm`s are constructed `dtype=jnp.float32`. Each norm's fan-out is
audited; explicit exit casts go only where the consumer does not self-heal:

**Cast required (non-self-healing consumers):**

| Site | Consumer | Fix |
|---|---|---|
| `HealSwinBlock.norm1`/`norm2` post-norm residuals (mirrored in `SwinBlock`, `HealConvBlock` — six sites) | residual add (`shortcut +` — DropPath is dtype-transparent) | `self.norm1(x).astype(dtype)` inside the residual expression; without it the *first* block of every stage permanently re-promotes the trunk |
| `PatchExpand` trailing norm | next `DecoderStage`'s block residuals + skip concat | module gains a `dtype` kwarg; `__call__` ends `return self.norm(x).astype(self.dtype)` |
| `PatchEmbed` optional norm (`patch_embed_norm=True`) | first `EncoderStage`'s block residuals (via transparent `pos_drop`) | same exit cast (latent leak — default is False) |

**No cast (self-healing or endpoint consumers):**

| Site | Why |
|---|---|
| `PatchMerging` norm | feeds its own `reduction` linear on the same line — fp32 lives for exactly one op, inside the module |
| decoder `norm_up` | feeds `FinalPatchExpand.expand` linear on the next line |
| encoder final `norm` | **is** an endpoint: standalone encoder emits fp32 by contract; inside the U-Net the decoder's first `PatchExpand.expand` linear self-heals |
| `FinalPatchExpand` trailing norm | sole consumer is the emit-fp32 `output` conv — the `norm → output` tail is deliberately fp32 end-to-end; casting here would round activations right before an fp32 layer for no benefit. This is why `FinalPatchExpand` differs from `PatchExpand`: their consumers differ |

**Compute-dtype layers** (`dtype=params.dtype`): `qkv`/`proj`, `Mlp.fc1/fc2` (gelu
runs bf16 — smooth, bounded-gradient, fine), `PatchMerging.reduction`,
`PatchExpand.expand`, `FinalPatchExpand.expand`, `PatchEmbed.proj`, decoder
`concat_back_dim` linears (which also self-heal the skip concat), HealConv's
`dwconv`.

**fp32 Params that need no use-site cast:** `logit_scale`, the rel-bias table, and
`rope_freqs` are consumed only inside fp32 islands; the island's single exit cast
covers them. (Contrast with GenSBI's `condition_embedding` bug, where an fp32 Param
was added directly to the bf16 stream.)

**HealConv specifics:** same block-level residual casts; the validity-mask Buffer
keeps its existing use-site cast (now resolving to compute dtype — the 0/1 mask is
exact in any float dtype); output conv is the fp32 endpoint like the others.

## 5. Threading mechanics and edge cases

- Shared blocks (`Mlp`, `PatchMerging`, `PatchExpand`, `FinalPatchExpand`,
  `PatchEmbed`) gain `dtype="bfloat16"` alongside `param_dtype`, forwarded to their
  Linear/Conv constructors. Their internal LayerNorms are **hard-coded**
  `dtype=jnp.float32` — islands are knob-independent, so norm dtype is deliberately
  not a parameter.
- Untouched: `DropPath`/`Dropout`/`Identity` (dtype-transparent); all Buffers
  (integer permutations stay integer, masks stay stored fp32 with their existing
  use-site casts; no Buffer gains a dtype parameter); `use_checkpoint`/`nnx.remat`
  (replays the forward with the same dtypes); RoPE machinery.
- **fp32 fallback invariant:** `dtype="float32"` must reproduce today's behavior.
  Every island already computes fp32 and the door cast matches the current
  fp32-default cast. The one site needing care is the logits einsum — for fp32
  operands `preferred_element_type=jnp.float32` should be numerically identical to
  the current matmul, but XLA lowering may differ; the suite pins this (exact if it
  passes, tolerance-relaxed with a comment if lowering differs).
- Serialization: `dtype` rides along in `dataclasses.asdict` JSON. No checkpoint
  restore path exists in this repo; fp32-default master weights keep any future one
  safe.

## 6. Testing

Invariant style, no golden values, modeled on `test_flux1_precision.py`.

**6.1 Params validation** (`test_params.py`, all three classes): `dtype=jnp.bfloat16`
→ stored `"bfloat16"`, JSON works; `"int32"`/garbage/`None`/un-enabled float64 raise
`ValueError` whose message names `dtype` (verifies the `field_name` refactor); knob
independence (`param_dtype="bfloat16", dtype="float32"` constructs).

**6.2 Precision battery** (new `tests/test_precision.py`, parametrized over the three
models; CPU-sized fixtures, `drop*_rate=0`):

- *Master weights fp32*: every leaf of `nnx.state(model, nnx.Param)` under the
  default knob, iterated with paths so failures name the parameter. The
  parametrization covers all manually-created Params: `logit_scale` (always),
  `rope_freqs` (default `rope_mixed` variant), and the rel-bias table (a
  `pos_embed="rel_bias"` variant).
- *Outputs fp32 + finite*: full model and standalone encoder.
- *Grads fp32*: `nnx.grad` on a scalar loss yields an all-fp32 tree.

**6.3 Spy tests (leak locks).** Class-level monkeypatch of block `__call__`
recording the entry activation dtype, run eagerly, asserting `bfloat16` under the
default knob. Block *i*'s entry dtype is block *i−1*'s residual output, so recording
*all* entries and asserting on `seen[1:]` sweeps every post-norm residual cast.
Probe map:

| Probe | Locks |
|---|---|
| all `HealSwinBlock` entries (encoder + decoder) | six residual casts, `PatchMerging` self-heal, `PatchExpand` exit cast, concat path |
| same with `patch_embed_norm=True` | `PatchEmbed` norm exit cast |
| `output`-conv input dtype | the deliberate fp32 `FinalPatchExpand → output` tail |
| softmax input fp32 / `attn @ v` operand bf16 | both boundaries of the attention logits island |
| `SwinBlock` / `HealConvBlock` equivalents | the mirrored casts |

Every spy is **RED-verified during implementation**: remove the corresponding cast,
confirm the test fails, restore. A spy that never went red proves nothing; this is
an explicit per-cast-site plan step, not left to diligence.

**6.4 Calibrated drift lock.** Post-implementation assessment step: per model, fixed
config/weight seed, ~10 seeded smooth synthetic inputs (not white noise), measure
`err = ‖o32 − o16‖∞ / ‖o32‖∞` (and normalized L2) between same-seed
`dtype="float32"` and `dtype="bfloat16"` models; record a one-off per-stage
decomposition (encoder-only vs full U-Net) in the implementation report. Then lock a
**two-sided** test:

```python
assert max(errs) < B * 3     # upper: mixed-precision quality regressed
assert max(errs) > B / 50    # lower canary: bf16 compute actually happening
```

with `B` = measured worst case, committed alongside a comment recording measured
values, date, and config. The upper bound catches a dropped cast or removed island;
the lower bound catches the dual failure (threading silently broken → everything
fp32 → every tolerance test passes while the knob is dead). Scope caveat: this locks
*computation* error on CPU; training-quality validation stays a GPU-side manual
gate (GenSBI precedent), but the leak class is dtype-structural and reproduces on
CPU.

**6.5 Existing-suite audit.** Geometry/permutation tests are dtype-blind. Structural
invariants (jit≡eager, remat≡no-remat, batch independence) stay on the bf16 default
so they now exercise bf16 paths. Tests asserting tighter-than-bf16 numerics get
`dtype="float32"` pinned with a comment. The existing pure-bf16 grad smoke test
(`param_dtype="bfloat16"`) stays as pure-bf16-mode coverage. **Sequencing:** thread
everything with the default temporarily at `"float32"` → full suite passes untouched
(threading is behavior-neutral) → flip the default to `"bfloat16"` → triage failures
into "pin fp32" vs "genuine bug". The flip commit is the audit.

## Out of scope

- Loss / optimizer / EMA precision, loss scaling — training-loop concerns
  (GenSBI-examples); the fp32-grads test locks this repo's half of the contract.
- Checkpoint restore casting — no restore path in `src/`.
- GPU convergence validation — manual gate on the user's side after merge.
