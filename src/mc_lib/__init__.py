"""mc_lib: vectorized Monte Carlo simulation on top of Polars.

Write a single *chunk trial* as a Polars expression pipeline (one row = one
trial); the engine handles reproducible random sampling, chunking, Polars-driven
parallel evaluation and mergeable aggregation.

Example
-------
>>> import polars as pl
>>> from mc_lib import Simulation, Normal
>>> def trial(df: pl.LazyFrame) -> pl.LazyFrame:
...     return df.with_columns(
...         area=pl.col("w") * pl.col("h"),
...     ).with_columns(passed=pl.col("area") >= 48.0)
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
from .engine import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_QUANTILE_SAMPLE_SIZE,
    Backend,
    Simulation,
)
from .estimate import (
    Estimate,
    estimate_mean,
    estimate_pi,
    expectation,
    integrate,
    probability,
)
from .result import SimulationResult
from .sampling import DEFAULT_STREAM_CHUNK, input_stream, sample_stream
from .stream import Stream

__version__ = "0.1.0"

__all__ = [
    # Batch engine
    "Simulation",
    "SimulationResult",
    "Backend",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_QUANTILE_SAMPLE_SIZE",
    # Distributions
    "Distribution",
    "Normal",
    "Uniform",
    "LogNormal",
    "Triangular",
    "Exponential",
    "Bernoulli",
    "Constant",
    # Composable streaming core
    "Stream",
    "sample_stream",
    "input_stream",
    "DEFAULT_STREAM_CHUNK",
    # Adaptive estimators
    "Estimate",
    "estimate_mean",
    "expectation",
    "probability",
    "integrate",
    "estimate_pi",
]
