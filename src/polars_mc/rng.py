"""Reproducible per-chunk random number generation.

A single ``seed`` deterministically produces one independent
:class:`numpy.random.Generator` per chunk via :class:`numpy.random.SeedSequence`
spawning.  This guarantees:

* the same ``seed`` always yields identical results, and
* two different chunks never share a random stream,

regardless of how many chunks the engine decides to use.
"""

from __future__ import annotations

import numpy as np

__all__ = ["spawn_generators"]


def spawn_generators(seed: int, n_chunks: int) -> list[np.random.Generator]:
    """Create ``n_chunks`` independent, reproducible generators from ``seed``.

    Parameters
    ----------
    seed:
        Master seed controlling the whole simulation.
    n_chunks:
        Number of independent chunk streams to produce.
    """
    if n_chunks < 0:
        raise ValueError(f"n_chunks must be >= 0, got {n_chunks}")

    root = np.random.SeedSequence(seed)
    children = root.spawn(n_chunks)
    return [np.random.default_rng(child) for child in children]
