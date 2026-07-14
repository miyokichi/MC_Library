"""Distributions as reproducible, lazy, infinite sample streams.

Where the batch engine draws a fixed number of trials up front, the streaming
layer turns a distribution into an *unbounded* :class:`~polars_mc.stream.Stream`
of NumPy sample chunks.  Each chunk is drawn from its own independent generator,
spawned deterministically from the master ``seed`` via
:class:`numpy.random.SeedSequence` -- the same mechanism as
:func:`polars_mc.rng.spawn_generators`, generalised so the number of chunks need
not be known in advance.

Reproducibility invariant (identical to the batch engine): for a given ``seed``
and ``chunk`` size, the k-th chunk is always the same array, no matter how many
chunks are eventually consumed.  Re-iterating the stream restarts from the first
chunk.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

from .distributions import Distribution
from .stream import Stream

__all__ = ["DEFAULT_STREAM_CHUNK", "sample_stream", "input_stream"]

# Streaming favours a smaller chunk than the batch engine so that convergence is
# checked often enough to stop promptly, while still amortising NumPy overhead.
DEFAULT_STREAM_CHUNK = 10_000


def _root_seed_sequence(seed: int) -> np.random.SeedSequence:
    return np.random.SeedSequence(seed)


def sample_stream(
    dist: Distribution,
    seed: int = 0,
    *,
    chunk: int = DEFAULT_STREAM_CHUNK,
) -> Stream[NDArray[np.generic]]:
    """An infinite stream of length-``chunk`` sample arrays from ``dist``.

    Parameters
    ----------
    dist:
        The distribution to draw from.
    seed:
        Master seed; identical seed + chunk size gives bit-identical chunks.
    chunk:
        Number of samples per emitted array.
    """
    if chunk < 1:
        raise ValueError(f"chunk must be >= 1, got {chunk}")

    def gen() -> Iterator[NDArray[np.generic]]:
        root = _root_seed_sequence(seed)
        while True:
            (child,) = root.spawn(1)
            rng = np.random.default_rng(child)
            yield dist.sample(rng, chunk)

    return Stream(gen)


def input_stream(
    inputs: dict[str, Distribution],
    seed: int = 0,
    *,
    chunk: int = DEFAULT_STREAM_CHUNK,
) -> Stream[dict[str, NDArray[np.generic]]]:
    """An infinite stream of ``{name: sample array}`` chunks for several inputs.

    Every input in a given chunk is drawn from the *same* per-chunk generator (in
    dict order), matching how the batch engine samples a chunk -- so a streaming
    run and a batch run share the same reproducibility guarantees.
    """
    if chunk < 1:
        raise ValueError(f"chunk must be >= 1, got {chunk}")
    if not inputs:
        raise ValueError("inputs must contain at least one distribution.")

    def gen() -> Iterator[dict[str, NDArray[np.generic]]]:
        root = _root_seed_sequence(seed)
        while True:
            (child,) = root.spawn(1)
            rng = np.random.default_rng(child)
            yield {name: dist.sample(rng, chunk) for name, dist in inputs.items()}

    return Stream(gen)
