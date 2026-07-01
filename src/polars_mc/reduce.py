"""Reduce a Polars DataFrame into small, mergeable per-column accumulators.

This is the shared core used both by the trial engine (:mod:`polars_mc.chunk`)
and by the standalone statistics API (:mod:`polars_mc.summarize`).  It never
returns the rows themselves -- only moments and a bounded reservoir per column.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import polars as pl
from numpy.typing import ArrayLike

from .aggregate import ColumnAccumulator, ColumnPlan, Moments, Reservoir

__all__ = ["DataLike", "to_dataframe", "accumulate_columns"]

# Accepted input for the statistics API: a Polars frame or a mapping of arrays.
DataLike = pl.DataFrame | pl.LazyFrame | Mapping[str, ArrayLike]


def to_dataframe(data: DataLike) -> pl.DataFrame:
    """Normalise supported inputs to an eager DataFrame."""
    if isinstance(data, pl.LazyFrame):
        return data.collect()
    if isinstance(data, pl.DataFrame):
        return data
    if isinstance(data, Mapping):
        return pl.DataFrame({key: np.asarray(value) for key, value in data.items()})
    raise TypeError(
        "data must be a polars DataFrame/LazyFrame or a mapping of arrays, got "
        f"{type(data).__name__}."
    )


def _moment_expressions(plans: tuple[ColumnPlan, ...]) -> list[pl.Expr]:
    exprs: list[pl.Expr] = [pl.len().alias("__n")]
    for i, plan in enumerate(plans):
        col = pl.col(plan.name).cast(pl.Float64)
        exprs.extend(
            [
                col.mean().alias(f"__{i}_mean"),
                col.var(ddof=0).alias(f"__{i}_varp"),
                col.min().alias(f"__{i}_min"),
                col.max().alias(f"__{i}_max"),
            ]
        )
    return exprs


def _moments_from_row(row: dict[str, object], index: int, n: int) -> Moments:
    if n == 0:
        return Moments()
    mean = float(row[f"__{index}_mean"])  # type: ignore[arg-type]
    varp_raw = row[f"__{index}_varp"]
    varp = 0.0 if varp_raw is None else float(varp_raw)  # type: ignore[arg-type]
    return Moments(
        count=n,
        mean=mean,
        m2=varp * n,
        minimum=float(row[f"__{index}_min"]),  # type: ignore[arg-type]
        maximum=float(row[f"__{index}_max"]),  # type: ignore[arg-type]
    )


def accumulate_columns(
    df: pl.DataFrame,
    plans: tuple[ColumnPlan, ...],
    rng: np.random.Generator,
    capacity: int,
) -> dict[str, ColumnAccumulator]:
    """Reduce ``df`` to one :class:`ColumnAccumulator` per plan column.

    ``rng`` supplies the reservoir keys; ``capacity`` bounds the per-column
    quantile sample.
    """
    missing = [p.name for p in plans if p.name not in df.columns]
    if missing:
        raise KeyError(
            f"missing requested column(s): {missing}. Available: {df.columns}."
        )

    stats_row = df.select(_moment_expressions(plans)).row(0, named=True)
    n = int(stats_row["__n"])

    accumulators: dict[str, ColumnAccumulator] = {}
    for i, plan in enumerate(plans):
        moments = _moments_from_row(stats_row, i, n)
        reservoir: Reservoir | None = None
        if plan.needs_sample:
            if n > 0:
                values = df.get_column(plan.name).cast(pl.Float64).to_numpy()
                reservoir = Reservoir.from_values(values, capacity, rng)
            else:
                reservoir = Reservoir.empty(capacity)
        accumulators[plan.name] = ColumnAccumulator(moments=moments, reservoir=reservoir)
    return accumulators
