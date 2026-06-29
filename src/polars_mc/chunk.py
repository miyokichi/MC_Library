"""Execution of a single chunk.

A chunk is the unit of work the engine schedules: ``n`` trials laid out as ``n``
rows.  We

1. sample the input columns with NumPy (reproducible per chunk),
2. apply the user trial to the whole chunk at once, and
3. reduce the chunk to small, mergeable per-column accumulators.

Only the tiny accumulators leave the chunk -- never the millions of rows.

Two trial styles are supported:

* **array style** -- an ordinary Python function whose parameters are input
  names; it receives NumPy arrays (length ``n``) and returns
  ``{output_name: array}``.  Because NumPy broadcasts, the same code that would
  read naturally for a single trial runs vectorized over the whole chunk::

      def trial(width, height):
          area = width * height
          return {"area": area, "passed": area >= 48.0}

* **frame style** -- a function ``(pl.LazyFrame) -> pl.LazyFrame | pl.DataFrame``
  expressed with Polars expressions.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import polars as pl
from numpy.typing import NDArray

from .aggregate import ColumnAccumulator, ColumnPlan, Moments
from .distributions import Distribution

__all__ = ["TrialFn", "TrialStyle", "classify_trial", "run_chunk"]

# Either an array-style function ``(**input_arrays) -> {name: array}`` or a
# frame-style function ``(LazyFrame) -> LazyFrame | DataFrame``.
TrialFn = Callable[..., Any]
TrialStyle = Literal["array", "frame"]


def classify_trial(trial: TrialFn, input_names: set[str]) -> TrialStyle:
    """Infer the trial style from its signature.

    A trial whose parameters are all input names (or which accepts ``**kwargs``)
    is treated as array style; anything else is frame style.
    """
    try:
        sig = inspect.signature(trial)
    except (TypeError, ValueError):
        return "frame"

    accepts_var_kw = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    named = {
        p.name
        for p in sig.parameters.values()
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.POSITIONAL_ONLY,
        )
    }
    if accepts_var_kw or (named and named <= input_names):
        return "array"
    return "frame"


def _sample_inputs(
    inputs: dict[str, Distribution],
    n: int,
    rng: np.random.Generator,
) -> dict[str, NDArray[np.generic]]:
    return {name: dist.sample(rng, n) for name, dist in inputs.items()}


def _coerce_output(name: str, value: object, n: int) -> NDArray[np.generic]:
    array = np.asarray(value)
    if array.ndim == 0:
        return np.full(n, array)
    if array.shape[0] != n:
        raise ValueError(
            f"trial output {name!r} has length {array.shape[0]}, expected {n}."
        )
    return array


def _apply_array_trial(
    trial: TrialFn,
    arrays: dict[str, NDArray[np.generic]],
    n: int,
) -> pl.DataFrame:
    sig = inspect.signature(trial)
    accepts_var_kw = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    kwargs = arrays if accepts_var_kw else {
        name: arrays[name] for name in sig.parameters if name in arrays
    }
    result = trial(**kwargs)
    if not isinstance(result, dict):
        raise TypeError(
            "An array-style trial must return a dict of {output_name: array}, "
            f"got {type(result).__name__}."
        )
    columns: dict[str, NDArray[np.generic]] = dict(arrays)
    for name, value in result.items():
        columns[name] = _coerce_output(name, value, n)
    return pl.DataFrame(columns)


def _apply_frame_trial(
    trial: TrialFn,
    arrays: dict[str, NDArray[np.generic]],
) -> pl.DataFrame:
    result = trial(pl.DataFrame(arrays).lazy())
    if isinstance(result, pl.LazyFrame):
        return result.collect()
    if isinstance(result, pl.DataFrame):
        return result
    raise TypeError(
        "A frame-style trial must return a polars LazyFrame or DataFrame, got "
        f"{type(result).__name__}."
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


def _draw_sample(
    out: pl.DataFrame,
    name: str,
    n: int,
    sample_size: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    values = out.get_column(name).cast(pl.Float64).to_numpy()
    if sample_size >= n:
        return np.ascontiguousarray(values, dtype=np.float64)
    idx = rng.choice(n, size=sample_size, replace=False)
    return np.ascontiguousarray(values[idx], dtype=np.float64)


def run_chunk(
    inputs: dict[str, Distribution],
    trial: TrialFn,
    plans: tuple[ColumnPlan, ...],
    n: int,
    rng: np.random.Generator,
    sample_size: int,
    style: TrialStyle,
) -> dict[str, ColumnAccumulator]:
    """Run one chunk and return per-column accumulators.

    ``sample_size`` is the (uniform) number of rows to retain per
    quantile-tracked column; the engine sizes it so the pooled sample stays
    near the configured budget.
    """
    arrays = _sample_inputs(inputs, n, rng)
    if style == "array":
        out = _apply_array_trial(trial, arrays, n)
    else:
        out = _apply_frame_trial(trial, arrays)

    missing = [p.name for p in plans if p.name not in out.columns]
    if missing:
        raise KeyError(
            f"trial output is missing requested column(s): {missing}. "
            f"Available columns: {out.columns}."
        )

    stats_row = out.select(_moment_expressions(plans)).row(0, named=True)
    n_actual = int(stats_row["__n"])

    accumulators: dict[str, ColumnAccumulator] = {}
    for i, plan in enumerate(plans):
        moments = _moments_from_row(stats_row, i, n_actual)
        samples: list[NDArray[np.float64]] = []
        if plan.needs_sample and n_actual > 0 and sample_size > 0:
            samples.append(_draw_sample(out, plan.name, n_actual, sample_size, rng))
        accumulators[plan.name] = ColumnAccumulator(moments=moments, samples=samples)
    return accumulators
