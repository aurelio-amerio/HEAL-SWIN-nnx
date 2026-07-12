from flax import nnx


class Buffer(nnx.Variable):
    """Non-trainable constant state: index permutations, attention masks,
    relative position indices. Excluded from ``nnx.Param`` filters, so
    optimizers and ``nnx.grad`` never touch it, but it travels through
    ``nnx.split``/``nnx.merge``/``nnx.jit`` as regular pytree state."""
