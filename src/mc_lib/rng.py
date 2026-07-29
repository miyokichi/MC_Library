"""Reproducible per-chunk random number generation.

This module is the library's *single source of seeding*.  A master ``seed``
deterministically produces one independent :class:`numpy.random.Generator` per
chunk via :class:`numpy.random.SeedSequence` spawning, exposed as a lazy,
re-runnable :class:`~mc_lib.stream.Stream`.  This guarantees:

* the same ``seed`` always yields identical results,
* two different chunks never share a random stream, and
* the k-th chunk's generator is the same whether the run consumes 3 chunks or
  three million -- so a fixed-N run and an adaptive run agree chunk for chunk,

regardless of how many chunks the engine eventually decides to use.

Everything downstream derives from :func:`generator_stream`: the sampling layer
maps it to arrays of samples, and the engine maps it to executed chunks.  Note
what flows through the stream is the *seed*, never the sampled data -- that is
what lets ``backend="processes"`` ship a tiny generator to each worker and draw
the (large) samples there, instead of pickling millions of rows.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from .stream import Stream

__all__ = ["generator_stream", "spawn_generators", "Generator"]

# Re-exported so other modules share a single name for the RNG type.
Generator = np.random.Generator


def generator_stream(seed: int) -> Stream[Generator]:
    """An infinite lazy stream of independent, reproducible chunk generators.

    Spawning one child at a time from the root :class:`~numpy.random.SeedSequence`
    is equivalent to spawning them all at once (``SeedSequence`` counts the
    children it has produced), so the stream can be consumed lazily -- and
    without knowing the chunk count up front -- while staying bit-identical to a
    bulk spawn.
    """

    def gen() -> Iterator[Generator]:
        root = np.random.SeedSequence(seed)
        while True:
            (child,) = root.spawn(1)
            yield np.random.default_rng(child)

    return Stream(gen)


def spawn_generators(seed: int, n_chunks: int) -> list[Generator]:
    """The first ``n_chunks`` generators of :func:`generator_stream`.

    Parameters
    ----------
    seed:
        Master seed controlling the whole simulation.
    n_chunks:
        Number of independent chunk streams to produce.
    """
    if n_chunks < 0:
        raise ValueError(f"n_chunks must be >= 0, got {n_chunks}")
    return generator_stream(seed).take(n_chunks).to_list()
