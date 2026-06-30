"""polars_mc: vectorized Monte Carlo simulation on top of Polars.

Write a single *chunk trial* as a Polars expression pipeline (one row = one
trial); the engine handles reproducible random sampling, chunking, Polars-driven
parallel evaluation and mergeable aggregation.

Example
-------
>>> import polars as pl
>>> from polars_mc import Simulation, Normal
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
from .result import SimulationResult

__version__ = "0.1.0"

__all__ = [
    "Simulation",
    "SimulationResult",
    "Backend",
    "Distribution",
    "Normal",
    "Uniform",
    "LogNormal",
    "Triangular",
    "Exponential",
    "Bernoulli",
    "Constant",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_QUANTILE_SAMPLE_SIZE",
]
