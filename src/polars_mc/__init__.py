"""polars_mc: vectorized Monte Carlo simulation on top of Polars.

The high-level entry point is :class:`Simulation`, but its three stages are also
usable on their own:

* **random generation** -- :func:`sample`, :func:`sample_chunks`,
  :func:`sample_arrays`
* **statistics** -- :func:`summarize`, :class:`Summarizer`,
  :func:`summarize_partial`, :func:`merge`
* **chunked map-reduce** -- :func:`map_reduce_chunks`, :func:`chunk_sizes`

Example
-------
>>> import numpy as np
>>> from polars_mc import Simulation, Normal
>>> def trial(w, h):
...     area = w * h
...     return {"area": area, "passed": area >= 48.0}
>>> sim = Simulation(
...     inputs={"w": Normal(10.0, 0.2), "h": Normal(5.0, 0.1)},
...     trial=trial,
...     outputs={"area": ["mean", "std", ("q", 0.95)], "passed": ["mean"]},
... )
>>> result = sim.run(1_000_000, seed=42)
>>> round(result.value("area", "mean"), 1)
50.0
"""

from __future__ import annotations

from .aggregate import DEFAULT_QUANTILE_SAMPLE_SIZE
from .chunking import DEFAULT_CHUNK_SIZE, Backend, chunk_sizes, map_reduce_chunks
from .distributions import (
    Bernoulli,
    Constant,
    Distribution,
    Exponential,
    LogNormal,
    Normal,
    Triangular,
    Uniform,
)
from .engine import Simulation
from .result import SimulationResult, StatResult
from .sampling import sample, sample_arrays, sample_chunks
from .summarize import Stats, Summarizer, merge, summarize, summarize_partial

__version__ = "0.1.0"

__all__ = [
    # High-level
    "Simulation",
    "SimulationResult",
    # Distributions
    "Distribution",
    "Normal",
    "Uniform",
    "LogNormal",
    "Triangular",
    "Exponential",
    "Bernoulli",
    "Constant",
    # Random generation (standalone)
    "sample",
    "sample_chunks",
    "sample_arrays",
    # Statistics (standalone)
    "summarize",
    "summarize_partial",
    "merge",
    "Summarizer",
    "Stats",
    "StatResult",
    # Chunked map-reduce (standalone)
    "map_reduce_chunks",
    "chunk_sizes",
    "Backend",
    # Constants
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_QUANTILE_SAMPLE_SIZE",
]
