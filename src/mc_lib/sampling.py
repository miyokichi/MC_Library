"""Distributions as reproducible, lazy, infinite sample streams.

This layer is a thin ``map`` over :func:`mc_lib.rng.generator_stream`: take
the library's one stream of per-chunk generators and draw from a distribution
with each.  The engine maps the *same* generator stream to executed chunks, so a
streaming run and a batch run share one seeding mechanism rather than two
lookalike copies of it.

Reproducibility invariant (identical to the batch engine): for a given ``seed``
and ``chunk`` size, the k-th chunk is always the same array, no matter how many
chunks are eventually consumed.  Re-iterating the stream restarts from the first
chunk.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .distributions import Distribution
from .rng import Generator, generator_stream
from .stream import Stream

__all__ = ["DEFAULT_STREAM_CHUNK", "sample_stream", "input_stream"]

# Streaming favours a smaller chunk than the batch engine so that convergence is
# checked often enough to stop promptly, while still amortising NumPy overhead.
DEFAULT_STREAM_CHUNK = 10_000


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

    def draw(rng: Generator) -> NDArray[np.generic]:
        return dist.sample(rng, chunk)

    return generator_stream(seed).map(draw)


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

    def draw(rng: Generator) -> dict[str, NDArray[np.generic]]:
        return {name: dist.sample(rng, chunk) for name, dist in inputs.items()}

    return generator_stream(seed).map(draw)
